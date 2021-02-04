export OPENBLAS_NUM_THREADS=1
for vi in {1..10} 0
do
  echo Validation $vi
  time python scripts/cmaq_uk.py --validate=${vi}
  time python scripts/blend.py --validate=${vi} 20160715
done
time python scripts/validate_stats.py output/UK.%Y%m%d.ALL_BOTH.{suffix}.nc csv/UK.20160715.ALL_BOTH.validation.csv
time python scripts/validate_stats.py output/UK.%Y%m%d.ALL_BLEND.{suffix}.nc csv/UK.20160715.ALL_BLEND.validation.csv
time python scripts/validate_stats.py output/UK.%Y%m%d.BLEND_BOTH.{suffix}.nc csv/UK.20160715.BLEND_BOTH.validation.csv
time python scripts/validate_stats.py output/UK.%Y%m%d.BLEND_BLEND.{suffix}.nc csv/UK.20160715.BLEND_BLEND.validation.csv

