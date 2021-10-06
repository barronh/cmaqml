__all__ = [
    'sklearn_LinearRegression', 'sklearn_RandomForestRegressor',
    'scipy_linregress', 'cmaqml_vna', 'cmaqml_evna',
]

import numpy as np


class cmaq_regression:
    def __init__(self, xkeys, ykey, **kwds):
        """
        Thin wrapper for sklearn regression models

        Primary purpose is to add __str__ method
        """
        self.xkeys = xkeys
        self.verbose = kwds.get('verbose', 0)
        self.ykey = ykey
        self._model = None

    def fit(self, p, y):
        return self._model.fit(p.values, y)

    def predict(self, p):
        return self._model.predict(p)

    def display(self):
        return str(self)


class cmaqml_vna(cmaq_regression):
    def __init__(self, xkeys, ykey, enhanced=False, fastsort=False, **kwds):
        """
        xkeys : list
            keys to use a predictive variables. Must include an x coordinate
            and a ycoordinate. If enhanced, it should also include a model
            prediction keys
        ykey : str
            key to use as target
        enhanced : bool
            Use VNA of factor rather to adjust model
        verbose : int
            level of verbosity
        fastsort : bool
            if fastsort, then use custom sorted for faster performance than
            cKDTree
        kwds : mappable
            See VNA and cKDTree
        """
        # All non-model inputs should be coordinates in the order of X, y
        cmaq_regression.__init__(self, xkeys, ykey, **kwds)
        self.enhanced = enhanced
        if enhanced:
            modkey = xkeys[-1]
            self.modidx = xkeys.index(modkey)
        else:
            modkey = None
            self.modidx = None
        self.modkey = modkey
        self.fastsort = fastsort

        self.coordidx = [i for i, k in enumerate(self.xkeys) if k != modkey]
        self.kwds = kwds

    def fit(self, p, y):
        from scipy.spatial import cKDTree
        from .voronoi import fastsort
        # All non-model inputs should be coordinates in the order of X, y
        pv = np.asarray(p)
        if self.fastsort:
            self.tree = fastsort(pv[:, self.coordidx])
        else:
            self.tree = cKDTree(pv[:, self.coordidx])

        assert((self.tree.data == pv[:, self.coordidx]).all())
        # Correction factor
        self._obs = np.asarray(y)
        if self.enhanced:
            self._model = pv[:, self.modidx]
            self._z = self._obs / self._model
        else:
            self._z = self._obs
        return None

    def predict(self, p):
        from .voronoi import VNA
        pv = np.asarray(p)

        if self.enhanced:
            zs = pv[:, self.modidx]
        else:
            zs = pv[:, 0] * 0
        vna_out = np.zeros_like(zs)
        vna_count = np.zeros_like(zs)
        # vna_std = np.zeros_like(zs)
        xys = pv[:, self.coordidx]
        if self.verbose:
            n = zs.size
            modn = int(n / 100)
            if modn < 1:
                modn = 1

        tree = self.tree
        for idx, z in np.ndenumerate(zs):
            xy = xys[idx, :]
            if self.verbose > 0 and (idx[0] % modn) == 0:
                print(f'\r{idx[0] / n:.1%}', end='')
            known_pidx, = np.where((tree.data == xy).all(1))
            if known_pidx.size > 0:
                vna_out[idx] = self._z[known_pidx].mean()
            else:
                out = VNA(
                    tree, xy, self._z, **self.kwds
                )
                vna_out[idx] = out[0]
                vna_count[idx] = out[1].size
                # std[idx] = vna_z.std()

        if self.verbose > 0:
            print(flush=True)

        if self.verbose > 0:
            print(f'VNA: {vna_out.mean():.1f}+/-{vna_out.std():.1f}')
            # print(f'Std: {std.mean():.1f}+/-{std.std():.1f}')
            print(f'Count: {vna_count.mean():.1f}+/-{vna_count.std():.1f}')

        if self.enhanced:
            out = zs * vna_out
        else:
            out = vna_out
        return out

    def display(self):
        return f'eVNA(**{self.kwds})'


class cmaqml_evna(cmaqml_vna):
    def __init__(self, xkeys, ykey, fastsort=False, **kwds):
        """
        Thin wrapper around cmaqml_vna that forces enhanced=True
        """
        cmaqml_vna.__init__(self, xkeys, ykey, enhanced=True, **kwds)


class sklearn_RandomForestRegressor(cmaq_regression):
    def __init__(self, xkeys, ykey, **kwds):
        """
        Thin wrapper to sklearn.ensemble.RandomForestRegressor
        """
        from sklearn.ensemble import RandomForestRegressor
        cmaq_regression.__init__(self, xkeys, ykey, **kwds)
        self._model = RandomForestRegressor(**kwds)

    def __str__(self):
        return str(self._model.get_params())

    def display(self):
        from sklearn import tree
        tree_txts = [
            '\t' + tree.export_text(
                estimator, feature_names=self.xkeys
            ).replace('\n', '\n\t')
            for estimator in self._model.estimators_
        ]
        outstr = (
            f'sklearn_RandomForestRegressor: {self._model.get_params()}\n'
            + '\n'.join([
                f'\ttree{ti:03d}:\n{tt}\n' for ti, tt in enumerate(tree_txts)
            ])
        )
        return outstr


class sklearn_LinearRegression(cmaq_regression):
    def __init__(self, xkeys, ykey):
        """
        Thin wrapper to sklearn.ensemble.RandomForestRegressor
        """
        from sklearn.linear_model import LinearRegression
        cmaq_regression.__init__(self, xkeys, ykey)
        self._model = LinearRegression()

    def __str__(self):
        outstr = (
            f'{self.ykey} = {self._model.intercept_:.6e} + '
            + ' + '.join([
                f'{coef:.6e} * {xkey}'
                for xkey, coef in zip(self.xkeys, self._model.coef_)
            ])
        )
        return outstr


class scipy_linregress(cmaq_regression):
    """
    Thin wrapper for scipy.stats.linregress
    """
    def __init__(self, xkeys, ykey):
        if len(xkeys) > 1:
            raise ValueError('scipy.stats.linregress can only take one xkey')
        cmaq_regression.__init__(self, xkeys, ykey)

    def fit(self, p, y):
        from scipy.stats import linregress
        self._lr = linregress(p.values[:, 0], y)

    def predict(self, x):
        return self._lr.slope * x + self._lr.intercept

    def __str__(self):
        return (
            f'{self.ykey} = {self._lr.slope} * {self.xkey}'
            + f' + {self._lr.intercept}'
        )

    def display(self):
        return str(self)
