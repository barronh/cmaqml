How to Use on Colab
===================

Step 1: Install Libraries
-------------------------

pip install pykrige sklearn scipy pseudonetcdf pycno

*Note*: use the --user option if you do not have admin rights

Step 2: Get cmaqkrig
--------------------

wget https://github.com/barronh/cmaqkrig/archive/main.zip
unzip main.zip

Step 3: Upload CMAQ, GPW file, and GRIDCRO2D file
-------------------------------------------------

* The CMAQ file should be an hr2day combine file that has
  a single value that is the predictor.
* The GPW file should have a DENS variable with units pop/km2
* The GRIDCRO2D file should have HT with units meters above sea level

Step 4: Download observations
-----------------------------

cmaqkrig expects an path to a file download from EPAs pregenerated
https://aqs.epa.gov/aqsweb/airdata/download_files.html

Step 5: Edit config.json
------------------------

* Make sure all the paths point to your files.
* Make sure krig_thin matches your expectations
  * krig_thin=1 means use all the data
  * krig_thin=2 means use every other data point
  * ...
  * krig_thin=N means use every N data point
  * kriging is the slowest part, so use a high value
    for testing your configuration and a lower value
    for real applications
* Make sure the incmaqkey and obs_defn are appropriate
  * for ozone, for example O3_8HRMAX and first_max_value * 1000
  * for PM25, for example PM25_FRM and arithmetic_mean

Step 6: Run
-----------

Run either the validation or production scripts

  * prod.sh runs one fitting with all data.
  * validation.sh take longer because it runs 10x as many fittings

Step 7: Review
--------------

Make sure you look at the validation figure and statistics. Remember, this
is under active development.
