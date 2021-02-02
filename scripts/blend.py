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
    validate_suffix = ''
else:
    validate_suffix = f'.{args.validate:02d}'
cmaqkey = 'UK'
poppath = cfg['pop_path']

popf = pnc.pncopen(poppath, format='ioapi')

# Time three is present day
DENS = popf.variables['DENS'][3, 0]

paths = sorted(glob(f'output/UK.{date}.[EW][SN]_???.nc{validate_suffix}'))

ukfiles = {
    path.split('.nc')[0].split('.')[-1]: pnc.pncopen(path, format='ioapi')
    for path in paths
}
I, J = np.meshgrid(
    np.arange(popf.NCOLS),
    np.arange(popf.NROWS)
)
lon, lat = popf.ij2ll(I, J)

N = (lat >= nsdiv)
S = ~N
E = (lon >= ewdiv)
W = ~E
U = (DENS >= urdiv)
R = ~U

outf = ukfiles['WN_RUR'].copy()
urbf = ukfiles['WN_RUR'].copy()
rurf = ukfiles['WN_RUR'].copy()
# cmaqkeys = [k for k in outf.variables if k != 'TFLAG']
cmaqkeys = ['UK_TOTAL', 'UK_ERROR', 'SD', 'Q', 'Y']

ismine = np.ones(N.shape, dtype='bool')

ufrac = np.maximum(0, np.minimum(1, (DENS - u0) / (u1 - u0)))
nfrac = np.maximum(0, np.minimum(1, (lat - n0) / (n1 - n0)))
efrac = np.maximum(0, np.minimum(1, (lon - e0) / (e1 - e0)))
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

outf.save(
    f'output/UK.{date}.FUSED.URBRUR.nc{validate_suffix}',
    complevel=1, verbose=0
)
outf.save(
    f'output/UK.{date}.FUSED.BOTH.nc{validate_suffix}', complevel=1, verbose=0
)
urbf.save(
    f'output/UK.{date}.FUSED.URB.nc{validate_suffix}', complevel=1, verbose=0
)
rurf.save(
    f'output/UK.{date}.FUSED.RUR.nc{validate_suffix}', complevel=1, verbose=0
)
