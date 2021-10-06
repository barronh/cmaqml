__all__ = ['run']
import time
import os
import PseudoNetCDF as pnc
import pandas as pd
import numpy as np
from pykrige.uk import UniversalKriging
from .obs import loadobs, train_and_testdfs, add_gridded_meta
from .opts import loadcfg, loadmetafiles
from . import models


class timer:
    def __init__(self, verbose):
        self._starttimes = {}
        self._endtimes = {}
        self._currentkey = None
        self.verbose = verbose

    def __call__(self, key):
        t = time.time()
        okey = self._currentkey
        self._starttimes[key] = t
        if okey is not None:
            self._endtimes[okey] = t
            dt = t - self._starttimes[okey]
            logger(f'{okey} {dt:.1f} seconds', 1, self.verbose)
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
        logger(f'Total {tot_dt:.1f} seconds', 1, self.verbose)


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
    yv = sqf.copyVariable(zqv, key='Y', withdata=False)
    xv = sqf.copyVariable(zqv, key='X', withdata=False)
    xv[:], yv[:] = sqf.ll2xy(longitude, latitude)
    keepkeys = ('longitude', 'latitude', 'X', 'Y', cmaqkey, 'x' + cmaqkey)
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


def logger(msg, level=1, printlevel=1):
    if printlevel >= level:
        print(msg, flush=True)


def run(cfg, validate=0, verbose=1, overwrite=False):
    """
    Run utility

    Arguments
    ---------
    cfg : str or mappable
        if str, it should be a path loadable by loadcfg
        otherwise, it should match loadcfg results
    validate : int
        random_state to use for validation or 0 for a production run
    verbose : int
        Add or remove verbosity (higher is more verbose)
    overwrite : bool
        If overwrite is True, clobber existing files with os.remove

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

    # load metadata files
    metafiles = loadmetafiles(cfg)

    # load observations with metadata
    obsdf = loadobs(cfg['obs'], gf, metafiles, verbose=verbose)

    transform = cfg['regression_options']['transform']
    transformforward = cfg['transformforward']
    transformbackward = cfg['transformbackward']

    cmaqkey = cfg['model']['model_inkey']
    aqskey = cfg['obs']['obs_key']

    if validate == 0:
        traindf = obsdf
        suffix = 'prod'
    else:
        traindf = train_and_testdfs(obsdf, None, validate)['train']
        print(traindf.shape)
        suffix = f'test{validate:02}'

    gridx = np.arange(gf.NCOLS, dtype='f') * gf.XCELL + gf.XCELL / 2
    gridy = np.arange(gf.NROWS, dtype='f') * gf.YCELL + gf.YCELL / 2
    atmpl = cfg['model']['model_template']
    outtmpl = cfg['output_template']

    traindf.loc[:, 'xO'] = transformforward(traindf[aqskey].values)

    first = True
    outf = None
    outpaths = {}
    for thisdate, ddf in traindf.groupby(['date']):
        logger(f'{atmpl} {thisdate}', 1, verbose)
        cmaqpath = thisdate.strftime(atmpl)
        logger(cmaqpath, 1, verbose)
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
            logger(querykey, 1, verbose)
            runtime = timer(verbose)
            runtime('prep')
            krig_opts = cfg['krig_opts'].copy()
            outpath = thisdate.strftime(outtmpl.format(**locals()))
            outpaths.setdefault(querykey, []).append(outpath)

            if os.path.exists(outpath):
                if overwrite:
                    os.remove(outpath)
                else:
                    logger(f'Keeping {outpath}', 1, verbose)
                    continue

            querystring = querystr.format(**cfg)
            df = ddf.query(querystring).copy(
            ).iloc[::cfg['obs']['thin_dataset']]

            logger(f'Obs shape {df.shape}', 1, verbose)
            pt_obs = df[aqskey]

            logger(f'{aqskey} mean: {pt_obs.mean():.2f}', 1, verbose)

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
            logger(f'Model point mean: {pt_linest.mean():.2f}', 1, verbose)

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
            logger(f'Model grid mean: {Y2d.mean():.2f}', 1, verbose)
            runtime('Krig')

            ###########################################################
            # Create the 2D universal kriging object and solves for the
            # two-dimension kriged error and variance.
            if cfg["thin_krig_dataset"] > 0:
                logger('Init UK...', 1, verbose)
                uk = UniversalKriging(
                    df.X.values.astype('f')[::cfg["thin_krig_dataset"]],
                    df.Y.values.astype('f')[::cfg["thin_krig_dataset"]],
                    df.xYres.values[::cfg["thin_krig_dataset"]],
                    **krig_opts
                )
                logger('Exec UK...', 1, verbose)
                runtime('Krig exec')
                pxkerr, pxsdvar = uk.execute(
                    "points", df.X.values.astype('f'), df.Y.values.astype('f')
                )
                pt_ukest = transformbackward(df.xY - pxkerr)
                logger(f'UnivKrig pnt mean: {pt_ukest.mean():.2f}', 1, verbose)

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
                    logger('*** WARNING ****', 1, verbose)
                    logger(
                        'Negative variances found. Replacing with 0.',
                        1, verbose
                    )
                    logger(
                        f'xVAR (min, max): {xsdvar.min()}, {xsdvar.max()}',
                        1, verbose
                    )
                    logger(f'N: {mxsdvar.mask.sum()}', 1, verbose)
                    logger(f'@: {np.where(mxsdvar.mask)}', 1, verbose)
                    logger('After masking', 1, verbose)
                    logger(
                        f'xSD (min, max): {xsderr.min()}, {xsderr.max()}',
                        1, verbose
                    )
                    logger('*** WARNING ****', 1, verbose)

                # Convert UK and SD data to original units
                kerr = transformbackward(xkerr)
                sderr = transformbackward(xsderr)
                kout = transformbackward(xY2d - xkerr)

                logger(
                    f'UK err at grid mean: {transformbackward(xkerr).mean()}',
                    1, verbose
                )
                logger(f'UK out at grid mean: {kout.mean()}', 1, verbose)

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
                ukevar = outf.copyVariable(
                    ukvar, key='UK_ERROR', withdata=False
                )
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

            logger('Save...', 1, verbose)

            # FILEDESC
            filedesc = f"""Model: {cmaqpath}
AQS Query: {querystring}
Date: {thisdate}
Transformation: {transform}
"""
            setattr(outf, 'FILEDESC', filedesc)
            outf.SDATE = int(thisdate.strftime('%Y%j'))
            outf.STIME = int(thisdate.strftime('%H%M%S'))
            outf.TSTEP = 240000
            # Save file to disk
            diskf = outf.save(outpath, complevel=1, verbose=0)
            diskf.close()
            runtime.total()

    return outpaths
