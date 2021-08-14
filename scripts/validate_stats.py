import PseudoNetCDF as pnc
import numpy as np
from cmaqml.opts import loadcfg
from cmaqml.obs import loadobs, train_and_testdfs
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json')
parser.add_argument('--validate', type=int, action='append', default=[1])
parser.add_argument('strftime')
parser.add_argument('outpath')
args = parser.parse_args()

cfg = loadcfg(args.config)
gf = pnc.pncopen(cfg['griddesc_path'], format='griddesc', GDNAM=cfg['domain'])
obsdf = loadobs(cfg['obs'], gf, [])
transformforward = cfg['transformforward']
transformbackward = cfg['transformbackward']

outfile = open(args.outpath, 'w')

print('Model,date,withholding,RMSE,NMBPCT,R,Coverage', file=outfile)
for validate in args.validate:
    testdf = train_and_testdfs(obsdf, None, validate)['test']
    suffix = f'test{validate:02d}'
    for date in np.unique(testdf.date.dt.to_pydatetime()):
        myvals = testdf.query(f'date == "{date}"')
        ukpath = date.strftime(
            args.strftime.format(suffix=suffix)
        )
        #print(ukpath)
        datestr = date.strftime('%Y-%m-%d')
        uk_f = pnc.pncopen(ukpath, format='ioapi')
        i = testdf.loc[:, 'I']
        j = testdf.loc[:, 'J']
        tempdf = testdf.copy()
        tempdf['UK'] = uk_f.variables['UK_TOTAL'][0, 0][j, i]
        tempdf['Y'] = uk_f.variables['Y'][0, 0][j, i]
        tempdf['Q'] = uk_f.variables['Q'][0, 0][j, i]

        transstd = transformforward(uk_f.variables['SD'][0, 0][j, i])
        transmean = transformforward(tempdf['UK'])
        tempdf['UK_LO'] = transformbackward(transmean - 1.96 * transstd)
        tempdf['UK_HI'] = transformbackward(transmean + 1.96 * transstd)

        tempdf.eval('UKBias = UK - O', inplace=True)
        tempdf.eval('YBias = Y - O', inplace=True)
        tempdf.eval('QBias = Q - O', inplace=True)
        URMSE = np.sqrt(np.mean(tempdf.UKBias**2))
        UNMB = tempdf.UKBias.mean() / tempdf.O.mean() * 100
        UR = tempdf.filter(['UK', 'O']).corr().iloc[0, 1]
        YRMSE = np.sqrt(np.mean(tempdf.YBias**2))
        YNMB = tempdf.YBias.mean() / tempdf.O.mean() * 100
        YR = tempdf.filter(['Y', 'O']).corr().iloc[0, 1]
        QRMSE = np.sqrt(np.mean(tempdf.QBias**2))
        QNMB = tempdf.QBias.mean() / tempdf.O.mean() * 100
        QR = tempdf.filter(['Q', 'O']).corr().iloc[0, 1]
        # Coverage
        UCov = (
            (tempdf.O > tempdf.UK_LO)
            & (tempdf.O < tempdf.UK_HI)
        ).mean()
        tempdf.sort_values(by='O', inplace=True)
        # ax.fill_between(tempdf.O, y1=tempdf.UK_LO, y2=tempdf.UK_HI)
        print(f'CMAQ,{datestr},{validate},{QRMSE:.6e},{QNMB:.2f},{QR:.6e},nan', file=outfile)
        print(f'LinCMAQ,{datestr},{validate},{YRMSE:.6e},{YNMB:.2f},{YR:.6e},nan', file=outfile)
        print(f'UK,{datestr},{validate},{URMSE:.6e},{UNMB:.2f},{UR:.6e},{UCov:.6e}', file=outfile)
        

outfile.close()
