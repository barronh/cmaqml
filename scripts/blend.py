import PseudoNetCDF as pnc
import numpy as np
from glob import glob
from opts import cfg
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--validate', type=int, default=0)
parser.add_argument('date', help='YYYYMMDD')
args = parser.parse_args()
date = args.date


ewdiv = cfg['longitude_center']
nsdiv = cfg['latitude_center']
urdiv = cfg['urban_minimum_density']
u0 = cfg["urban_blend"]["0"]
u1 = cfg["urban_blend"]["1"]
n0 = cfg["latitude_blend"]["0"]
n1 = cfg["latitude_blend"]["1"]
e0 = cfg["longitude_blend"]["0"]
e1 = cfg["longitude_blend"]["1"]

if args.validate == 0:
    suffix = '.prod.nc'
else:
    suffix = f'.test{args.validate:02}.nc'

cmaqkey = 'UK'
poppath = cfg['pop_path']

popf = pnc.pncopen(poppath, format='ioapi')

# Time three is present day
pop_per_km2 = popf.variables['DENS'][3, 0]

paths = sorted(glob(f'output/UK.{date}.*{suffix}'))

ukfiles = {
    path.split(f'{suffix}')[0].split('.')[-1]: pnc.pncopen(path, format='ioapi')
    for path in paths
}
I, J = np.meshgrid(
    np.arange(popf.NCOLS),
    np.arange(popf.NROWS)
)
longitude, latitude = popf.ij2ll(I, J)

outf = ukfiles['EN_RUR'].copy()
delattr(outf, 'lags')
delattr(outf, 'empirical_semivariance')
delattr(outf, 'predicted_semivariance')
fusedefn = (
    "\n Fractions:"
    + f"\n North: (lat - {n0}) / ({n1} - {n0})"
    + f"\n East : (lon - {e0}) / ({e1} - {e0})"
)
outf.FILEDESC = (
    "Fused:\n" + "\n - ".join([p for p in paths])
    + fusedefn
    + f"\n Urban: (pop_per_km2 - {u0}) / ({u1} - {u0})"
)
urbf = outf.copy()
urbf.FILEDESC = (
    "Fused:\n" + "\n - ".join([p for p in paths if 'S_URB.' in p or 'N_URB.' in p])
    + fusedefn
)
rurf = outf.copy()
rurf.FILEDESC = (
    "Fused:\n" + "\n - ".join([p for p in paths if 'S_RUR.' in p or 'N_RUR.' in p])
    + fusedefn
)
bothf = outf.copy()
bothf.FILEDESC = (
    "Fused:\n" + "\n - ".join([p for p in paths if 'S_BOTH.' in p or 'N_BOTH.' in p])
    + fusedefn
)
allf = outf.copy()
allf.FILEDESC = (
    "Fused:\n" + "\n - ".join([p for p in paths if 'ALL_RUR.' in p or 'ALL_URB.' in p])
    + fusedefn
)
# cmaqkeys = [k for k in outf.variables if k != 'TFLAG']
cmaqkeys = ['UK_TOTAL', 'UK_ERROR', 'SD', 'Q', 'Y']

ufrac = np.maximum(0, np.minimum(1, (pop_per_km2 - u0) / (u1 - u0)))
nfrac = np.maximum(0, np.minimum(1, (latitude - n0) / (n1 - n0)))
efrac = np.maximum(0, np.minimum(1, (longitude - e0) / (e1 - e0)))
rfrac = 1 - ufrac
sfrac = 1 - nfrac
wfrac = 1 - efrac

for cmaqkey in cmaqkeys:
    outf.variables[cmaqkey][0, 0] = (
        ukfiles['WN_URB'].variables[cmaqkey][0, 0] * (wfrac * nfrac * ufrac)
        + ukfiles['WN_RUR'].variables[cmaqkey][0, 0] * (wfrac * nfrac * rfrac)
        + ukfiles['WS_URB'].variables[cmaqkey][0, 0] * (wfrac * sfrac * ufrac)
        + ukfiles['WS_RUR'].variables[cmaqkey][0, 0] * (wfrac * sfrac * rfrac)
        + ukfiles['EN_URB'].variables[cmaqkey][0, 0] * (efrac * nfrac * ufrac)
        + ukfiles['EN_RUR'].variables[cmaqkey][0, 0] * (efrac * nfrac * rfrac)
        + ukfiles['ES_URB'].variables[cmaqkey][0, 0] * (efrac * sfrac * ufrac)
        + ukfiles['ES_RUR'].variables[cmaqkey][0, 0] * (efrac * sfrac * rfrac)
    )
    urbf.variables[cmaqkey][0, 0] = (
        ukfiles['WN_URB'].variables[cmaqkey][0, 0] * (wfrac * nfrac)
        + ukfiles['WS_URB'].variables[cmaqkey][0, 0] * (wfrac * sfrac)
        + ukfiles['EN_URB'].variables[cmaqkey][0, 0] * (efrac * nfrac)
        + ukfiles['ES_URB'].variables[cmaqkey][0, 0] * (efrac * sfrac)
    )
    rurf.variables[cmaqkey][0, 0] = (
        ukfiles['WN_RUR'].variables[cmaqkey][0, 0] * (wfrac * nfrac)
        + ukfiles['WS_RUR'].variables[cmaqkey][0, 0] * (wfrac * sfrac)
        + ukfiles['EN_RUR'].variables[cmaqkey][0, 0] * (efrac * nfrac)
        + ukfiles['ES_RUR'].variables[cmaqkey][0, 0] * (efrac * sfrac)
    )
    bothf.variables[cmaqkey][0, 0] = (
        ukfiles['WN_BOTH'].variables[cmaqkey][0, 0] * (wfrac * nfrac)
        + ukfiles['WS_BOTH'].variables[cmaqkey][0, 0] * (wfrac * sfrac)
        + ukfiles['EN_BOTH'].variables[cmaqkey][0, 0] * (efrac * nfrac)
        + ukfiles['ES_BOTH'].variables[cmaqkey][0, 0] * (efrac * sfrac)
    )
    allf.variables[cmaqkey][0, 0] = (
        ukfiles['ALL_URB'].variables[cmaqkey][0, 0] * (ufrac)
        + ukfiles['ALL_RUR'].variables[cmaqkey][0, 0] * (rfrac)
    )

outf.save(
    f'blend/UK.{date}.BLEND_BLEND{suffix}',
    complevel=1, verbose=0
)
bothf.save(
    f'blend/UK.{date}.BLEND_BOTH{suffix}', complevel=1, verbose=0
)
allf.save(
    f'blend/UK.{date}.ALL_BLEND{suffix}', complevel=1, verbose=0
)

urbf.save(
    f'blend/UK.{date}.BLEND_URB{suffix}', complevel=1, verbose=0
)
rurf.save(
    f'blend/UK.{date}.BLEND_RUR{suffix}', complevel=1, verbose=0
)
