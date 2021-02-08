import time
import os
import PseudoNetCDF as pnc
import pandas as pd
import numpy as np
from pykrige.uk import UniversalKriging
from .obs import loadobs, train_and_testdfs, add_gridded_meta
from .opts import loadcfg
from . import models

class timer:
    def __init__(self):
        self._starttimes = {}
        self._endtimes = {}
        self._currentkey = None

    def __call__(self, key):
        t = time.time()
        okey = self._currentkey
        self._starttimes[key] = t
        if okey is not None:
            self._endtimes[okey] = t
            dt = t - self._starttimes[okey]
            print(f'{okey} {dt:.1f} seconds')
        self._currentkey = key    

    def total(self):
        t = time.time()
        if self._currentkey not in self._endtimes:
            self._endtimes[self._currentkey] = t
        tot_dt = 0
        for key, start in self._starttimes.items():
            end = self._endtimes[key]
            dt = end - start
            tot_dt += dt
        print(f'Total {tot_dt:.1f} seconds')
            

def cmaqtmpl(qf, cmaqkey):
    """
    Return a template to carry CMAQ single time-slice

    Arguments
    ---------
    qf : PseudoNetCDFFile
        file that has CMAQ data
    cmaqkey : str
        label of CMAQ target prediction

    Returns
    -------
    tmpf : PseudoNetCDFFile
        has latitude, longitude, and cmaqkey
    """
    longitude, latitude = qf.ij2ll(
        *np.meshgrid(np.arange(qf.NCOLS), np.arange(qf.NROWS))
    )
    sqf = qf.subset([cmaqkey]).slice(TSTEP=0)
    zqv = sqf.variables[cmaqkey]
    sqf.copyVariable(zqv, key='x' + cmaqkey, withdata=False)
    latv = sqf.copyVariable(zqv, key='latitude', withdata=False)
    latv[:] = latitude
    lonv = sqf.copyVariable(zqv, key='longitude', withdata=False)
    lonv[:] = longitude
    keepkeys = ('longitude', 'latitude', cmaqkey, 'x' + cmaqkey)
    for key in list(sqf.variables):
        if key not in keepkeys:
            del sqf.variables[key]
    return sqf


def gridded_x(infiles, xkeys):
    """
    Convert files with variables to a vectorized
    set of variables. The last file with any xkey
    will be used for xkey

    Arguments
    ---------
    infiles : list
        PseudoNetCDFFiles
    xkeys : list
        keys to get from infiles

    Returns:
    outv : array
        shaped gridsize, len(xkeys) where each column is a x variable
    """
    x2d_list = []
    for xkey in xkeys:
        for infile in infiles[::-1]:
            if xkey in infile.variables:
                x2d_list.append(infile.variables[xkey][:].ravel())
                break
        else:
            raise KeyError(f'{xkey} not found')

    x2d = np.array(x2d_list).T
    return x2d


def run(cfg, validate=0):
    """
    Run utility

    Arguments
    ---------
    cfg : str or mappable
        if str, it should be a path loadable by loadcfg
        otherwise, it should match loadcfg results
    validate : int
        random_state to use for validation or 0 for a production run

    Returns
    -------
    outf : PseudoNetCDFFile
        Also saved to disk
    """

    if isinstance(cfg, str):
        cfg = loadcfg(cfg)

    # Open a GRIDDESC definition
    gf = pnc.pncopen(
        cfg['griddesc_path'], format='griddesc', GDNAM=cfg['domain']
    )

    # Create a list of meta files
    metafiles = []
    for gmcfg in cfg['gridded_meta']:
        mf = pnc.pncopen(
            gmcfg['path_template'],
            format=gmcfg.get('format', 'ioapi')
        )
        smf = mf.subset(gmcfg['var_keys'])
        if 'slice' in gmcfg:
            tmf = smf.slice(**gmcfg['slice'])
        else:
            tmf = smf

        for key in tmf.variables:
            if key not in gmcfg['var_keys']:
                del tmf.variables[key]

        metafiles.append(tmf)

    # load observations with metadata
    obsdf = loadobs(cfg, gf, metafiles)

    transform = cfg['transform']
    transformforward = cfg['transformforward']
    transformbackward = cfg['transformbackward']

    cmaqkey = cfg['model_inkey']
    aqskey = cfg['obs_key']

    if validate == 0:
        traindf = obsdf
        suffix = 'prod'
    else:
        traindf = train_and_testdfs(validate)['train']
        suffix = f'test{validate:02}'

    gridx = np.arange(gf.NCOLS, dtype='f') * gf.XCELL + gf.XCELL / 2
    gridy = np.arange(gf.NROWS, dtype='f') * gf.YCELL + gf.YCELL / 2
    atmpl = cfg['model_template']
    outtmpl = cfg['output_template']

    traindf.loc[:, 'xO'] = transformforward(traindf[aqskey].values)

    first = True
    outf = None
    for thisdate, ddf in traindf.groupby(['date']):
        print(atmpl, thisdate)
        cmaqpath = thisdate.strftime(atmpl)
        print(cmaqpath)
        qf = pnc.pncopen(cmaqpath, format='ioapi')

        times = pd.to_datetime(
            [t.replace(tzinfo=None) for t in qf.getTimes()]
        )
        ti = (times == thisdate).argmax()

        zqv = qf.variables[cmaqkey][ti]
        if first:
            sqf = cmaqtmpl(qf, cmaqkey)

        sqf.SDATE = qf.variables['TFLAG'][ti, 0, 0]
        sqf.STIME = qf.variables['TFLAG'][ti, 0, 1]
        sqf.variables[cmaqkey][:] = zqv
        sqf.variables['x' + cmaqkey][:] = transformforward(zqv)

        # Already has other meta-files
        # adding CMAQ
        add_gridded_meta(ddf, [sqf])

        for querykey, querystr in cfg['query_definitions'].items():
            print(querykey, flush=True)
            runtime = timer()
            runtime('prep')
            krig_opts = cfg['krig_opts'].copy()
            outpath = thisdate.strftime(outtmpl.format(**locals()))
            outpath = outpath

            if os.path.exists(outpath):
                print(f'Keeping {outpath}')
                continue

            querystring = querystr.format(**cfg)
            df = ddf.query(querystring).copy().iloc[::cfg['thin_dataset']]

            print('Obs shape', df.shape, flush=True)
            pt_obs = df[aqskey]

            print(f'{aqskey} mean: {pt_obs.mean():.2f}', flush=True)

            # Linear regression
            xkeys = cfg["regression_options"]["xkeys"]
            ykey = cfg["regression_options"]["ykey"]
            # model:
            # * scipy_linregress,
            # * sklearn_RandomForestRegressor,
            # * sklearn_LinearRegresson
            model = getattr(
                models, cfg["regression_options"]["model"]
            )(
                xkeys, ykey, **cfg["regression_options"]["model_options"]
            )

            runtime('Model fit')
            model.fit(df.loc[:, xkeys], df.loc[:, ykey])

            runtime('Model predict')
            # X is transformed, so Y has transformed units
            df['xY'] = model.predict(df.loc[:, xkeys])
            df['xYres'] = df['xY'] - df['xO']
            pt_linest = transformbackward(df['xY'])
            print(f'Model point mean: {pt_linest.mean():.2f}', flush=True)

            # Predict using model at grid
            outshape = sqf.variables[cmaqkey].shape
            x2d = gridded_x(metafiles + [sqf], xkeys)
            xY2d = model.predict(x2d).reshape(outshape)

            # Prepare template output file
            outf = gf.subset([])
            outunit = getattr(qf.variables[cmaqkey], 'units', 'unknown')

            # Copy CMAQ variable
            cmaqvar = outf.createVariable(
                'Q', 'f', ('TSTEP', 'LAY', 'ROW', 'COL'),
                long_name='CMAQ', var_desc='CMAQ prediction',
                units=outunit
            )
            cmaqvar[:] = sqf.variables[cmaqkey][:]

            # Prepare Linear Fit of CMAQ to Obs variable
            linvar = outf.createVariable(
                'Y', 'f', ('TSTEP', 'LAY', 'ROW', 'COL'),
                long_name='mCMAQplB', units=outunit,
                var_desc=f'untransform({model})'
            )
            linvar.description = f"{model.display()}"

            Y2d = transformbackward(xY2d)
            linvar[0, 0] = Y2d[:]
            print(f'Model grid mean: {Y2d.mean():.2f}', flush=True)
            runtime('Krig')


            ###########################################################
            # Create the 2D universal kriging object and solves for the
            # two-dimension kriged error and variance.
            if cfg["thin_krig_dataset"] > 0:
                print('Init UK...', flush=True)
                uk = UniversalKriging(
                    df.X.values.astype('f')[::cfg["thin_krig_dataset"]],
                    df.Y.values.astype('f')[::cfg["thin_krig_dataset"]],
                    df.xYres.values[::cfg["thin_krig_dataset"]],
                    **krig_opts
                )
                print('Exec UK...', flush=True)
                runtime('Krig exec')
                pxkerr, pxsdvar = uk.execute(
                    "points", df.X.values.astype('f'), df.Y.values.astype('f')
                )
                pt_ukest = transformbackward(df.xY - pxkerr)
                print(f'UnivKrig pnt mean: {pt_ukest.mean():.2f}', flush=True)

                # Calculate UniversalKriging at all grid points
                # Retruns the transformed error term and the variance of
                # the error term
                xkerr, xsdvar = uk.execute("grid", gridx, gridy)
                # When duplicate x,y coordinates, the variance can be negative
                # masking negative valuse and replacing with 0
                mxsdvar = np.ma.masked_less(xsdvar, 0)
                ninvalid = mxsdvar.mask.sum()

                xsderr = mxsdvar.filled(0)**0.5
                if ninvalid > 0:
                    print('*** WARNING ****')
                    print('Negative variances found. Replacing with 0.')
                    print('xVAR (min, max):', xsdvar.min(), xsdvar.max())
                    print('N:', mxsdvar.mask.sum())
                    print('@:', np.where(mxsdvar.mask))
                    print('After masking')
                    print('xSD (min, max):', xsderr.min(), xsderr.max())
                    print('*** WARNING ****')

                # Convert UK and SD data to original units
                kerr = transformbackward(xkerr)
                sderr = transformbackward(xsderr)
                kout = transformbackward(xY2d - xkerr)

                print('UK err at grid', 'mean', transformbackward(xkerr).mean())
                print('UK out at grid', 'mean', kout.mean())

                # Prepare UK variable
                ukvar = outf.createVariable(
                    'UK_TOTAL', 'f', ('TSTEP', 'LAY', 'ROW', 'COL'),
                    long_name='UnivKrigOutput', units=outunit,
                    var_desc=(
                        'UnivKrig = untransform(xY + UK(xY - xO));'
                        + ' where x indicates transformed'
                    )
                )
                ukvar[:] = kout
                ukvar.description = f"""pykrige.UniversalKriging
Model function: {uk.variogram_model}({uk.variogram_model_parameters});
Statistics: {uk.get_statistics()}
Lags: {uk.lags}
Semivariance: {uk.semivariance}
Variance Model: {uk.variogram_function(uk.variogram_model_parameters, uk.lags)}
"""
                ukevar = outf.copyVariable(ukvar, key='UK_ERROR', withdata=False)
                ukevar.long_name = 'UnivKrigOfRes',
                ukevar.var_desc = (
                    'UnivKrig = untransform(UK(xY - xO)); where x indicates'
                    + ' transformed'
                )
                ukevar[:] = kerr

                uksdvar = outf.copyVariable(ukvar, key='SD', withdata=False)
                uksdvar.long_name = 'UnivKrigStdError'
                uksdvar.var_desc = 'untransform(stddev(transform(UK))'
                uksdvar[:] = sderr

                outf.lags = uk.lags
                outf.empirical_semivariance = uk.semivariance
                outf.predicted_semivariance = uk.variogram_function(
                    uk.variogram_model_parameters, uk.lags
                )

            print('Save...', flush=True)

            # FILEDESC
            filedesc = f"""Model: {cmaqpath}
AQS Query: {querystring}
Date: {thisdate}
Transformation: {transform}
"""
            setattr(outf, 'FILEDESC', filedesc)
            # Save file to disk
            outf.save(outpath, complevel=1, verbose=0)
            runtime.total()
    return outf