import numpy as np


class NaturalNeighborInterpolation:
    def __init__(
        self, k=None, r=None, min_dist=0, min_neighbor=0, invdistpower=2,
        verbose=0, audit=False
    ):
        """
        The NaturalNeighborInterpolation class supports Vornonoi Neighbor
        Averaging with inverse distance weighting. This object interface
        is modeled after scikit-learn.

        Arguments
        ---------
        k : int
            Number of nearest neighbors to test
        r : float
            Radius (not used if k is set) for use by tree.query_ball_point
        min_dist : float
            Double k for any point where the k neighbors are all closer than
            min_dist
        min_neighbor : int
            Increase the search radius by 10% until the number of neighbors
            is >= min_neighbor
        invdistpower : float
            Weights are currently only related to distance and must be some
            power thereof. Default is 2.
        audit : bool
            Keep a tally of neighbors, distances, and weights for each
            prediction point these are accessed by the get_audit function.
        verbose : int
            level of verbosity

        Example
        -------
        nni = NaturalNeighborInterpolation(k=4, min_neighbor=0, audit=True)
        nni.fit([[0, 0], [0, 2], [4, 1], [1, 0]], [1, 2, 3, 4])
        print(nni.predict([[0.5, 0.5]]))
        [2.46428571]
        print(nni.get_audit())
        {
            'X': [array([0., 1., 0., 4.])], 'Y': [array([0., 0., 2., 1.])],
            'Z': [array([1, 4, 2, 3])],
            'W': [array([0.44642857, 0.44642857, 0.08928571, 0.01785714])]
        }

        Notes
        -----
        Currently, I have only implemented inverse distance weights, which
        only use the Delaunay triangles and not the Voronoi Polygons. The
        Delaunay implementation is about twice as fast, but is more limited
        in terms of weights that can be applied. To implement Sibson or
        Laplace weights, the Delaunay calculation would need to be replaced
        with the slower Voronoi. Then, the elements of the polygons could
        be used to construct alternate weights.

        Sibson weights are the captured area weights. These are the areas that
        were previously part of another polygon, but are "captured" by the new
        polygon when the point is added.

        Laplace weights are the interface length divided by the distance of
        the neighbor.

        If these other weights are added, the Voronoi diagram take longer to
        calculate.

        https://en.wikipedia.org/wiki/Natural_neighbor_interpolation
        https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-natural-neighbor-works.htm
        https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-natural-neighbor-works.htm
        """
        # All non-model inputs should be coordinates in the order of X, y
        self.set_params(
            k=k, min_dist=min_dist,
            r=r, min_neighbor=min_neighbor,
            invdistpower=invdistpower,
            audit=audit, verbose=verbose,
        )

    def set_params(
        self, k=None, r=None, min_dist=0, min_neighbor=0, invdistpower=2,
        verbose=0, audit=False
    ):
        """
        See __init__ for description of params
        """
        from warnings import warn
        if k is None and r is None:
            raise ValueError('k or r can be None, but not both')
        if k is not None and r is not None:
            warn('k and r were provided; k takes precedence; r is set to None')
            r = None
            raise ValueError('k or r can be None, but not both')
        self.verbose = verbose
        self.invdistpower = invdistpower
        # allow for a partial delaunay rebuild based on the neighbors of
        # neighbors
        # self.incremental = incremental
        self.audit = audit
        # Fast sort is an alternative KDTree style sorting algorithm
        #
        # self.fastsort = fastsort
        self.k = k
        self.min_dist = min_dist
        self.r = r
        self.min_neighbor = min_neighbor

    def get_params(self, deep=0):
        """
        Return model parameters

        If deep, return a deep copy
        """
        out = dict(
            k=self.k, min_dist=self.mindist,
            r=self.r, min_neighbor=self.min_neighbor,
            invdistpower=self.invdistpower,
            audit=self.audit, verbose=self.verbose,
        )
        if deep:
            from copy import deepcopy
            out = deepcopy(out)
        return out

    def get_audit(self):
        """
        Returns a dictionary of X, Y, Z, and weights (W) from the
        last predict call where audit was True

        Example
        """
        return dict(X=self.NBX, Y=self.NBY, Z=self.NBZ, W=self.NBW)

    def fit(self, p, y):
        from scipy.spatial import cKDTree
        # from .voronoi import fastsort
        # All non-model inputs should be coordinates in the order of X, y
        pv = np.asarray(p)
        # if self.fastsort:
        #     self.tree = fastsort(pv[:])
        # else:
        self.tree = cKDTree(pv[:])

        self._y = np.asarray(y)
        return None

    def predict(self, p):
        import scipy.spatial

        pv = np.asarray(p)
        # Internalize properties that will be reused within the loop
        audit = self.audit
        invdistpower = self.invdistpower
        verbose = self.verbose
        kmindist = self.min_dist
        rminneighbor = self.min_neighbor
        tree = self.tree
        k = self.k
        r = self.r
        fitz = self._y
        # if self.incremental:
        #     trim = self.trim

        if audit:
            NBW = []
            NBX = []
            NBY = []
            NBZ = []
        GZ = []
        nr = pv.shape[0]
        for ri, xy in enumerate(pv):
            if verbose > 0:
                print(f'\r{ri/nr:6.1%}', end='')
            # The incremental build based on known simplex changes did not
            # speed up the process. It is being commented out to document
            # that it was tested, and to leave a place for potential future
            # effort.
            #
            # if self.incremental:
            #     modvert = trim.simplices[trim.find_simplex(xy)]
            #     midx = np.unique(
            #         trim.simplices[
            #             np.in1d(
            #                 trim.simplices,
            #                 modvert[0]
            #             ).reshape(-1, 3).any(1)
            #         ])
            #     newxyg = np.vstack([trim.points[midx], xy])
            # else:

            # These are the k nearest neighbors
            tn = tree.data.shape[0]
            if k is not None:
                tmpk = k
                kdists, kidx = tree.query(xy, tmpk)
                kn = kidx.shape[0]
                kdmax = kdists.max()
                while kdmax < kmindist and kn < tn:
                    tmpk = tmpk * 2
                    kdists, kidx = tree.query(xy, tmpk)
                    kdmax = kdists.max()
                    kn = kidx.shape[0]
                # If the furthest is closer than mindist, throw an error
                assert(kdmax > kmindist)
            elif r is not None:
                tmpr = r
                kidx = tree.query_ball_point(xy, tmpr)
                rn = len(kidx)
                while rn < rminneighbor:
                    tmpr = 1.1 * tmpr
                    kidx = tree.query_ball_point(xy, tmpr)
                    rn = len(kidx)

                kdists = ((xy - tree.data[kidx])**2).sum(-1)**.5

            else:
                raise KeyError('r or k must be non None')

            # Otherwise get the subset in a new coordinate array
            newxyg = np.vstack([tree.data[kidx], xy])

            # Identify the last one as the target
            targetid = newxyg.shape[0] - 1

            # Build a partial Delaunay triangulation
            tric = scipy.spatial.Delaunay(newxyg)

            # Where the target point is part of the simplices, take all the
            # unique points including the targetid. So the neighbors index
            # (nidx) incldues the target id
            nidx = np.unique(
                tric.simplices[(tric.simplices[:] == targetid).any(1)]
            )

            # The targetid should be removed from the independent variables (x)
            xidx = nidx != targetid
            if not xidx.any():
                # If the target is not in a simplex, then it is a duplicate
                # point and the other point is the new value.
                pidx = ((xy - tree.data)**2).sum(1).argmin(0)
                if audit:
                    NBW.append(1)
                    NBX.append(tree.data[pidx, 0])
                    NBY.append(tree.data[pidx, 1])
                    NBZ.append(fitz[pidx])
                outz = fitz[pidx]
                GZ.append(outz)
                continue

            # The independent variables are the natural neighbors
            nnidx = nidx[xidx]

            # if incremental:
            #     inz = fitz[midx][nnidx]
            #     outp = trim.points[midx][nnidx]
            #     dist = ((outp - xy)**2).sum(1)**.5
            # else:
            outp = tree.data[kidx][nnidx]
            inz = fitz[kidx][nnidx]
            dist = kdists[nnidx]

            w = (1/dist)**invdistpower
            w /= w.sum()
            outz = (inz.T * w).sum(-1)
            if audit:
                NBW.append(w)
                NBX.append(outp[:, 0])
                NBY.append(outp[:, 1])
                NBZ.append(inz)
            GZ.append(outz)
        if verbose > 0:
            print()
        if audit:
            self.NBW = NBW
            self.NBX = NBX
            self.NBY = NBY
            self.NBZ = NBZ
        return np.asarray(GZ)

    def threaded_predict(self, p, threads):
        npts = len(p)

        def wrapper(p, r):
            result = self.predict(p)
            r.append(result)

        if threads > 1:
            import threading
            npt = (npts // threads)
            if (npts % threads) != 0:
                npt += 1

            mythreads = []
            threadresults = [[] for i in range(threads)]
            for ti in range(threads):
                starti = ti * npt
                endi = starti + npt
                td = threading.Thread(
                    target=wrapper, args=(p[starti:endi], threadresults[ti])
                )
                td.start()
                mythreads.append(td)
            for td in mythreads:
                td.join()

            return np.concatenate([r[0] for r in threadresults], axis=0)


def eVNA(ox, oy, oz, my, mx, k=None, r=None, **kwds):
    nna = NaturalNeighborInterpolation(k=k, r=r, **kwds)
    nna.fit(np.array([ox, oy]).T, oz)
    return nna.predict(np.array([mx, my]).T)
