
class cmaq_regression:
    def __init__(self, xkeys, ykey, **kwds):
        """
        Thin wrapper for sklearn regression models
        
        Primary purpose is to add __str__ method
        """
        self.xkeys = xkeys
        self.ykey = ykey
        self._model = None

    def fit(self, p, y):
        return self._model.fit(p.values, y)
    
    def predict(self, p):
        return self._model.predict(p)
    
    def display(self):
        return str(self)

class sklearn_RandomForestRegressor(cmaq_regression):
    def __init__(self, xkeys, ykey, **kwds):
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
        ntrees = len(self._model.estimators_)
        outstr = (
            f'sklearn_RandomForestRegressor: {self._model.get_params()} trees\n'
            + '\n'.join([
                f'\ttree{ti:03d}:\n{tt}\n' for ti, tt in enumerate(tree_txts)
            ])
        )
        return outstr

        
class sklearn_LinearRegresson(cmaq_regression):
    def __init__(self, xkeys, ykey):
        from sklearn import linear_model
        cmaq_regression.__init__(self, xkeys, ykey)
        self._model = linear_model.LinearRegression()
    
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
    scipy.stats.linregress wrapper
    """
    def __init__(self, xkeys, ykey):
        cmaq_regression.__init__(self, xkeys, ykey)
        
    def fit(self, p, y):
        from scipy.stats import linregress
        self._lr = linregress(p.values[:, 0], y)
    
    def predict(self, x):
        return self._lr.slope * x + self._lr.intercept

    def __str__(self):
        return f'{self.ykey} = {self._lr.slope} * {self.xkey} + {self._lr.intercept}'
    
    def display(self):
        return str(self)
