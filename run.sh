source venv/bin/activate
export OPENBLAS_NUM_THREADS=1
time python scripts/cmaq_uk.py 20160715
time python scripts/blend.py 20160715
time python scripts/make_maps.py 20160715
time python scripts/validate.py
deactivate
