export OPENBLAS_NUM_THREADS=1
time python -m cmaqml
time python scripts/validate_figs.py 0 output/UK.%Y%m%d.ALL.prod.nc
time python scripts/make_maps.py 20160115
