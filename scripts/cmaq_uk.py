import time
import PseudoNetCDF as pnc
import pandas as pd
import numpy as np
from pykrige.uk import UniversalKriging
from scipy.stats.mstats import linregress
from loadobs import aqsdf, train_and_testdfs, gf, pop_per_km2, elevation
from opts import cfg
import os
import argparse
import models

parser = argparse.ArgumentParser()
parser.add_argument('--validate', default=0, type=int)

args = parser.parse_args()

transform = cfg['transform']
transformforward = cfg['transformforward']
transformbackward = cfg['transformbackward']

cmaqkey = cfg['model_inkey']
aqskey = cfg['obs_key']

if args.validate == 0:
    alldf = aqsdf
    suffix = 'prod'
else:
    alldf = train_and_testdfs(args.validate)['train']
    suffix = f'test{args.validate:02}'

gridx = np.arange(gf.NCOLS, dtype='f') * gf.XCELL + gf.XCELL / 2
gridy = np.arange(gf.NROWS, dtype='f') * gf.YCELL + gf.YCELL / 2
longitude, latitude = gf.ij2ll(*np.meshgrid(np.arange(gf.NCOLS), np.arange(gf.NROWS)))
atmpl = cfg['model_template']
outtmpl = cfg['output_template']

for thisdate, ddf in alldf.groupby(['date']):
    print(atmpl, thisdate)
    cmaqpath = thisdate.strftime(atmpl)
    print(cmaqpath)
    qf = pnc.pncopen(cmaqpath, format='ioapi')

    # m2path = thisdate.strftime('METCRO2D.%Y%m%d.nc')
    # mf = pnc.pncopen(m2path, format='ioapi')

    times = pd.to_datetime(
        [t.replace(tzinfo=None) for t in qf.getTimes()]
    )
    ti = (times == thisdate).argmax()
    zqf = qf.variables[cmaqkey][ti]

    for querykey, querystr in cfg['query_definitions'].items():
        print(querykey, flush=True)
        times = []
        times.append(time.time())
        krig_opts = cfg['krig_opts'].copy()
        outpath = thisdate.strftime(outtmpl.format(**globals()))
        outpath = outpath

        if os.path.exists(outpath):
            print(f'Keeping {outpath}')
            continue

        querystring = querystr.format(**cfg)
        df = ddf.query(querystring).copy()
        print('Obs shape', df.shape, flush=True)
        pt_obs = df[aqskey]
        print(f'{aqskey} mean: {pt_obs.mean():.2f}', flush=True)

        i, j = df.I.values, df.J.values
        # CMAQ
        df['Q'] = zqf[0, j, i]

        # Transformed CMAQ
        df['xQ'] = transformforward(df.Q)
        df['xO'] = transformforward(df[aqskey].values)
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
        
        model.fit(df.loc[:, xkeys], df.loc[:, ykey])

        # X is transformed, so Y has transformed units
        df['xY'] = model.predict(df.loc[:, xkeys])
        df['xYres'] = df['xY'] - df['xO']
        pt_linest = transformbackward(df['xY'])
        print(f'Linear point mean: {pt_linest.mean():.2f}', flush=True)

        # Temperature
        # df['T'] = zqf[0, j, i]

        ###########################################################
        # Create the 2D universal kriging object and solves for the
        # two-dimension kriged error and variance.
        print('Init UK...', flush=True)
        uk = UniversalKriging(
            df.X.values.astype('f')[::cfg["thin_krig_dataset"]],
            df.Y.values.astype('f')[::cfg["thin_krig_dataset"]],
            df.xYres.values[::cfg["thin_krig_dataset"]],
            **krig_opts
        )
        print('Exec UK...', flush=True)

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

        # Convert kriged error to kriged output
        xQ = transformforward(zqf[0])
        x2d = np.array([eval(xkey).ravel() for xkey in xkeys]).T
        xY2d = model.predict(x2d).reshape(xQ.shape)
        kout = transformbackward(xY2d - xkerr)
        print('UK err at grid', 'mean', transformbackward(xkerr).mean())
        print('UK out at grid', 'mean', kout.mean())
        
        print('Save...', flush=True)
        # Prepare template output file
        outf = gf.renameVariables(DUMMY='UK_TOTAL')

        # Prepare UK variable
        ukvar = outf.variables['UK_TOTAL']
        ukvar[:] = np.nan
        ukvar.long_name = 'UnivKrigOutput'
        ukvar.var_desc = (
            'UnivKrig = untransform(xY + UK(xY - xO));'
            + ' where x indicates transformed'
        )
        ukvar.units = 'ppb'

        ukevar = outf.copyVariable(
            outf.variables['UK_TOTAL'], key='UK_ERROR', withdata=False
        )
        ukevar[:] = np.nan
        ukevar.long_name = 'UnivKrig of residual'
        ukevar.var_desc = (
            'UnivKrig = untransform(UK(xY - xO)); where x indicates'
            + ' transformed'
        )
        ukevar.units = 'ppb'

        # Prepare UK standard deviation variable
        uksdvar = outf.copyVariable(ukvar, key='SD', withdata=False)
        uksdvar.units = ukvar.units
        uksdvar.var_desc = 'untransform(stddev(transform(UK))'

        # Prepare CMAQ variable
        cmaqvar = outf.copyVariable(ukvar, key='Q', withdata=False)
        cmaqvar.long_name = 'CMAQ'
        cmaqvar.var_desc = 'CMAQ prediction'
        cmaqvar.units = getattr(
            qf.variables[cmaqkey], 'units', 'unknown'
        ).strip()

        # Prepare Linear Fit of CMAQ to Obs variable
        linvar = outf.copyVariable(ukvar, key='Y', withdata=False)
        linvar.long_name = 'mCMAQplB'
        linvar.units = cmaqvar.units
        linvar.var_desc = (
            f'untransform({model})'
        )

        # Convert UK and SD data to original units
        kerr = transformbackward(xkerr)
        sderr = transformbackward(xsderr)

        Y2d = transformbackward(xY2d)
        # Populate file data
        ukvar[0, 0] = kout[:]
        # square transform backward loses sign of kerr[:]
        # so the error in units is calculated by difference.
        ukevar[0, 0] = Y2d[:] - kout[:]
        uksdvar[0, 0] = sderr[:]
        cmaqvar[0, 0] = zqf[0]
        linvar[0, 0] = Y2d[:]

        # FILEDESC
        filedesc = f"""pykrige.UniversalKriging
Model: {cmaqpath}
AQS Query: {querystring}
Date: {thisdate}
Initial Model: {model.display()}
Transformation: {transform}
Model function: {uk.variogram_model}({uk.variogram_model_parameters});
Statistics: {uk.get_statistics()}
Lags: {uk.lags}
Semivariance: {uk.semivariance}
Variance Model: {uk.variogram_function(uk.variogram_model_parameters, uk.lags)}
"""
        setattr(outf, 'FILEDESC', filedesc)
        outf.lags = uk.lags
        outf.empirical_semivariance = uk.semivariance
        outf.predicted_semivariance = uk.variogram_function(
            uk.variogram_model_parameters, uk.lags
        )
        # Save file to disk
        outf.save(outpath, complevel=1, verbose=0)
        times.append(time.time())
        timesegs = np.diff(times)
        totaltime = np.sum(timesegs)
        print(querykey, f'{totaltime:.1f} seconds')
