export OPENBLAS_NUM_THREADS=1
for vi in {1..1} 0
do
  echo Validation $vi
  time python scripts/cmaq_uk.py --validate=${vi}
  time python scripts/blend.py --validate=${vi} 20160715
  if [ ${vi} -ne 0 ]; then
    time python scripts/validate.py ${vi} FUSED.URBRUR
  fi
done
time python scripts/make_maps.py 20160715
