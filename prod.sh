export OPENBLAS_NUM_THREADS=1
time python scripts/cmaq_uk.py
time python scripts/blend.py 20160715
time python scripts/validate_figs.py 0 output/UK.%Y%m%d.ALL_BOTH.prod.nc
time python scripts/validate_figs.py 0 blend/UK.%Y%m%d.ALL_BLEND.prod.nc
time python scripts/validate_figs.py 0 blend/UK.%Y%m%d.BLEND_BOTH.prod.nc
time python scripts/validate_figs.py 0 blend/UK.%Y%m%d.BLEND_BLEND.prod.nc
time python scripts/make_maps.py 20160715
