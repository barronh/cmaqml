import PseudoNetCDF as pnc
import os
import datetime
import numpy as np


inpaths = [
    '/home/bhenders/TestData/MCIP/GRIDCRO2D.12US2.35L.160101',
    '/work/ROMO/2016platform/CMAQv531/2016fh/12US2/POST/O3/O3_8HRMAX.LST.Y_24.2016fh.v531.12US2.5-9.ncf',
    '/work/ROMO/2016platform/CMAQv531/2016fh/12US2/POST/dailyavg/dailyavg.LST.Y_24.2016fh.v531.12US2.01.ncf',
    '/work/ROMO/gis_rasters/gpw-v4/gpw_v4_une_atotpopbt_densy_12US2.IOAPI.nc'
]
varkeys = [
    'HT',
    'O3_8HRMAX',
    'PM25_FRM',
    'DENS'
]
for inpath, varkey in zip(inpaths, varkeys):
    outpath = 'input/' + os.path.basename(inpath).replace('12US2', '108US2').replace('.ncf', '').replace('.nc', '') + '.nc'
    tmpf = pnc.pncopen(inpath, format='ioapi').subset([varkey])
    if 'dailyavg' in inpath:
        times = tmpf.getTimes()
        mydate = datetime.datetime(2016, 1, 15, tzinfo=datetime.timezone.utc)
        tidx = np.array([t == mydate for t in times])
        tmpf = tmpf.slice(TSTEP=tidx)
    elif 'O3_8HR' in inpath:
        times = tmpf.getTimes()
        mydate = datetime.datetime(2016, 7, 15, tzinfo=datetime.timezone.utc)
        tidx = np.array([t == mydate for t in times])
        tmpf = tmpf.slice(TSTEP=tidx)
    outf = tmpf.apply(ROW=lambda y: y[:-3].reshape(-1, 9).mean(1), COL=lambda x: x.reshape(-1, 9).mean(1))
    outf.NCOLS = len(outf.dimensions['COL'])
    outf.NROWS = len(outf.dimensions['ROW'])
    outf.YCELL = 108000.
    outf.XCELL = 108000.
    if 'gpw' in inpath:
        outf.TSTEP = 10000
        outf.updatetflag(overwrite=True)
    outf.variables.move_to_end('TFLAG', last=False)
    outf.GDNAM = "108US2".ljust(16)
    outf.FILEDESC = f"{varkey} from\n{inpath}\naveraged over 9 cells in each dimension"
    # diskf = outf.save('input/dailyavg.LST.Y_24.2016fh.v531.108US2.01.ncf')
    diskf = outf.save(outpath, complevel=1, verbose=0)
    diskf.close()
