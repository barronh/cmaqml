from pykrige.rk import Krige
from sklearn.model_selection import GridSearchCV
from loadobs import train_and_testdfs


df = train_and_testdfs(1)
vp = []

for psill in [1, .75, .5, .25, 0]:
    nugget = 1 - psill
    for drange in [5, 10, 20, 25]:
        vp.append(dict(psill=psill, range=drange, nugget=nugget))

param_dict = {
    "method": ["universal"],
    "variogram_model": ["exponential"],
    "variogram_parameters": vp
    # "nlags": [4, 6, 8],
    # "weight": [True, False]
}

estimator = GridSearchCV(
    Krige(), param_dict, verbose=True, return_train_score=True
)
# run the gridsearch
estimator.fit(X=df.filter(['I', 'J']).values.astype('f'), y=df.Y.values)
if hasattr(estimator, "best_score_"):
    print("best_score R2 = {:.3f}".format(estimator.best_score_))
    print("best_params = ", estimator.best_params_)

print("\nCV results::")
if hasattr(estimator, "cv_results_"):
    for key in [
        "mean_test_score",
        "mean_train_score",
        "param_method",
        "param_variogram_model",
    ]:
        print(" - {} : {}".format(key, estimator.cv_results_[key]))
