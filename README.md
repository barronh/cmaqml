Universal Kriging for CMAQ
==========================

    author: Barron H. Henderson
    original date: 2020-02-01
    last updated: 2020-02-05
    contributors: <your name here>


Status
------

Under active development. Currently, working with Ozone and PM for a single
test day. Only ready for developers to test and help develop.

Prerequisites
-------------

* Python >= 3.6
* numpy >= 2
* pykrige >= 1.5.1
* Optional:
  * sklearn >= 0.24

Overview
--------

Apply Universal Krigging as implemented in `pykrige` to CMAQ fields. This
is an example of regression kriging where the mean is first removed using
another model. The simplest example is a linear model, but cmaqkrig supports
multilinear regression and Random Forest as well. The options here will expand
over time.

* `yhat = m CMAQ + b`
    * `CMAQ` : CMAQ concentrations in ppb
    * `m`, `b` : parameters fit by scipy.stats.linregress where y is
                 AQS 1st maximum 8-hour average ozone
    * `yhat` : estimate based on CMAQ
    * `e` : `e = obs - yhat`; bias that is assumed to have spatial correlation

* `UK_ERROR = Krig(e)`
* `UK_TOTAL = yhat + UK_ERROR`

Blending
--------

In addition, cmaqkrig provides a mechanism that allows the mean estimation
model and the UniversalKriging system to be optimized in subdomains and then
reconstruct a complete surface by blending. Subdomains currently support
splitting on latitIn addition to spatial subdomains, I use a urban/rural division as
well.

Mean Estimation Models
----------------------

Before kriging the residual, the package estimates the best fit of the model to
observations using linear regression, multiple linear regression, or Random
Forest. In upcoming versions, we are likely to support extended voronoi
neighbor averaging, and custom models using the scipy.optimize framework. The
models for the mean are accessed via the config.json file "regression_options"
"model" key.

* scipy_linregress: provides access to scipy.stats.linregress for univariate
  linear regression,
* sklearn_LinearRegression: provides uni- or multi-variate regression via
  sklearn.linear_model.LinearRegression, or
* sklearn_RandomForestRegressor: provides Ensemble Random Forest modeling via
  sklearn.ensemble.RandomForestRegressor
  
Any sklearn model is capable of being added. The challenge is in finding the
right way to export the model as a text representation for meta-data. To add
a new model from sklearn, follow the templates sklearn_LinearRegression and
sklearn_RandomForestRegressor in scripts/models.py.

Please submit any additions back to the project.

Annotated Directory Structure
-----------------------------

```
.
|-- README.md
|-- config.json
|   # Fitting parameters and spatial domain splitting parameters
|-- scripts
|   |-- cmaq_uk.py
|   |   # Work horse: applies universal krigging to partial domains:
|   |   # East (E) or West (W), North (N) or South (S), Urban (U) or Rural (R)
|   |-- loadobs.py
|   |   # easy access to AQS datasets, used by cmaq_uk.py
|   |-- blend.py
|   |   # Create a composite surface from East/West, North/South, Urban/Rural
|   |-- models.py
|   |   # Wrappers for scipy and sklearn estimation models to support meta-data
|   |   # representation in the output FILEDESC attribute
|   |-- validate_figs.py
|   |   # Create validation figures including statistics from a single
|   |   # withholding
|   |-- validate_stats.py
|   |   # Create validation statistics from multiple witholdings
|   |-- make_maps.py
|   |   # Script for visualization
|   `-- fitting.py
|       # not complete. Ideally, optimize UK settings for application to domains
|-- input/
|   |-- daily_44201_20160715.zip
|   |   # subset of AQS; right now not part of repository for testing
|   |-- daily_88101_20160115.zip
|   |   # subset of AQS; right now not part of repository for testing
|   `-- CMAQ.20160715.nc
|       # subset of CMAQ. right now not part of repository
``-- output
    |-- UK.<YYYYMMDD>.<querykey>.nc
    |   # outputs from cmaq_uk.py 
    |   # template where
    |   #  * YYYYMMDD is the date
    |   #  * querykey in: (EN|ES|WN|WS|ALL)_(URB|RUR|BOTH)
    `-- UK.YYYYMMDD.FUSED.<querykey>.nc where 
        # outputs from blend.py
        # where querykey in ALL_URB, ALL_RUR, oroutputs from blend.py
```