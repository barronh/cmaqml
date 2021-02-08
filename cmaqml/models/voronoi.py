import numpy as np
from warnings import warn

class fastsort:
    def __init__(self, xy):
        """
        Functionality like cKDTree, but for some reason faster
        when searching on a grid...
        """
        self.data = xy

    def query(self, xy, k=1, distance_upper_bound=np.inf):
        obsxy = self.data[:]
        dist = ((obsxy - xy[..., None, :])**2).sum(-1)**0.5
        if self.data.shape[0] <= k:
            return dist, np.indices(self.xy.shape)
        else:
            kidx = np.argsort(dist, axis=1)[:, :k]
            pidx = np.indices(kidx.shape)[0]
            kdist = dist[pidx, kidx]
            if not np.isfinite(distance_upper_bound):
                return kdist, kidx
            else:
                dkidx = kidx[kdist < distance_upper_bound]
                dkdist = kdist[dkidx]
                return dkdist, dkidx


def VNA(tree, xy, obsz, inversedistancepower=2, **vn_kwds):
    """
    Return Voronoi Neighbor Averages
    
    vna = \sum_i{w_i*z_i}
    
    where:
        i \in Voronoi neighbors
        w_i = (1 / dist_i**inversedistancepower)
    
    Arguments
    ---------
    tree : scipy.stats.cKDtree
        must implement the query method and the data attribute.
    xy : coordinates
        dimensions(n,2)
    obsz : array
        values at tree.data coordinates
    inversedistancepower: scalar
        used to calculate weights
    vn_kwds :  mappable
        VoronoiNeighbors keywords
    """
    dist, idx = VoronoiNeighbors(tree, xy, **vn_kwds)
    vna_z = obsz[idx]
    
    # VNA and eVNA use inverse distance weighting
    if inversedistancepower > 0:
        invdist = (1 / dist)**inversedistancepower
        weight = invdist / invdist.sum()
    else:
        weight = np.ones_like(vna_z) / vna_z.size

    navals = np.isnan(vna_z)
    naweight = (weight * navals).sum()
    if naweight != 0:
        if naweight < 0.25:
            warn('NA values removed when < 25% of original weight')
            weight = weight[~navals]
            vna_z = vna_z[~navals]
            weight /= weight.sum()
        else:
            warn('NA values > 25%, expect NA outputs')
            pass

    vna_out = (vna_z * weight).sum()

    return vna_out, weight, vna_z

def VoronoiNeighbors(
    tree, xy, k=100, **kd_kwds
):
    """
    Return Neighbors on a Voronoi diagram (excuding xy)
    
    Arguments
    ---------
    tree : scipy.stats.cKDtree
        must implement the query method and the data attribute.
    xy : coordinates
        dimensions(n,2)
    k : int
        nearest points to search for Voronoi neighbors
    kd_kwds : mappable
        KDTree query keywords
    
    Returns
    -------
    dist, idx : array
        distances and indices of nearest k neighbors
    """
    from scipy.spatial import Voronoi
    k = min(k, tree.data.shape[0])
    dist, idx = tree.query(xy, k)
    dist = dist[0]
    idx = idx[0]
    xyg = np.vstack([tree.data[idx], xy])

    # Create a set of verticies with obs and model
    # then create a Voronoi diagram
    #  --Consider optimizing via a Voronoi copy and add_points
    gcidx = xyg.shape[0] - 1
    vor = Voronoi(xyg)
    vor.close()

    # For QA
    # plt.interactive(True)
    # voronoi_plot_2d(vor)

    # ridge_points "Indicies of the points between which each
    #               Voronoi ridge lies"[1]
    #  - points sharing a ridge are neighbors
    #  - the last point is the grid cell
    #
    # [1] https://docs.scipy.org/doc/scipy-0.18.1/reference/
    #     generated/scipy.spatial.Voronoi.html
    shared_edges = [xy for xy in vor.ridge_points if gcidx in xy]
    
    # Unique pairs excluding the grid cell piont are the closest
    # observations.
    neighboridx = np.sort(np.unique(shared_edges).ravel())[:-1]
    
    # Neighbors 
    return dist[neighboridx], idx[neighboridx]
