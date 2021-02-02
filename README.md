Universal Kriging for CMAQ
==========================

    author: Barron H. Henderson
    date: 2020-02-01

Overview
--------

Apply Universal Krigging as implemented in `pykrige` to CMAQ fields. Here, I
allow the UK to be optimized in subdomains and then reconstruct a complete
surface. In addition to subdomains, I use a urban/rural division as well.

* `yhat = m CMAQ + b`
    * CMAQ : CMAQ concentrations in ppb
    * m, b : parameters fit by scipy.stats.linregress where y is
             AQS 1st maximum 8-hour average ozone
    * yhat : estimate based on CMAQ

* `UK = Krig(yhat)`
    * spatially krigged surface



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
|   |-- test_df.py
|   |   # easy access to AQS datasets, used by cmaq_uk.py
|   |-- blend.py
|   |   # Create a composite surface from East/West, North/South, Urban/Rural
|   |-- 02_kriging3D.py
|   |   # Example from https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/examples/02_kriging3D.html
|   `-- fitting.py
|       # not complete. Ideally, optimize UK settings for application to domains
|-- input/
|   |-- daily_44201_20160715.zip
|   |   # subset of AQS
|   `-- CMAQ.20160715.nc
|       # subset of CMAQ?
|-- output
|   |-- UK.<YYYYMMDD>.<querykey>.nc
|   |   # outputs from cmaq_uk.py 
|   |   # template where
|   |   #  * YYYYMMDD is the date
|   |   #  * querykey in: (EN|ES|WN|WS|ALL)_(URB|RUR|BOTH)
|   `-- UK.YYYYMMDD.FUSED.<querykey>.nc where 
|       # outputs from blend.py
|       # where querykey in ALL_URB, ALL_RUR, oroutputs from blend.py
`-- venv
    # Virtual environment for this example
```
