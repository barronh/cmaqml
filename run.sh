export OPENBLAS_NUM_THREADS=1
for vi in {01..10} 0
do
  echo Validation $vi
  time python scripts/cmaq_uk.py --validate=${vi}
  time python scripts/blend.py --validate=${vi} 20160715
  if [ $vi -eq 0 ] then
      time python scripts/validate_figs.py ${vi} blend/UK.%Y%m%d.BLEND_BLEND.prod.nc
  else
      time python scripts/validate_figs.py ${vi} blend/UK.%Y%m%d.BLEND_BLEND.test${vi}.nc
  fi
done
time python scripts/make_maps.py 20160715
