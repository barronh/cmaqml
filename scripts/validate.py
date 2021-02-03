from loadobs import train_and_testdfs
import PseudoNetCDF as pnc
import numpy as np
import matplotlib.pyplot as plt
from opts import cfg
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('validate', type=int, default=1)
parser.add_argument('querykey')
args = parser.parse_args()

transformforward = cfg['transformforward']
transformbackward = cfg['transformbackward']

testdf = train_and_testdfs(args.validate)['test']

for date in np.unique(testdf.date.dt.to_pydatetime()):
    myvals = testdf.query(f'date == "{date}"')
    ukpath = date.strftime(
        f'output/UK.%Y%m%d.{args.querykey}.nc.{args.validate:02d}'
    )
    fused_uk_f = pnc.pncopen(ukpath, format='ioapi')
    i = testdf.loc[:, 'I']
    j = testdf.loc[:, 'J']
    tempdf = testdf.copy()
    tempdf['UK'] = fused_uk_f.variables['UK_TOTAL'][0, 0][j, i]
    tempdf['Y'] = fused_uk_f.variables['Y'][0, 0][j, i]
    tempdf['Q'] = fused_uk_f.variables['Q'][0, 0][j, i]

    transstd = transformforward(fused_uk_f.variables['SD'][0, 0][j, i])
    transmean = transformforward(tempdf['UK'])
    tempdf['UK_LO'] = transformbackward(transmean - 1.96 * transstd)
    tempdf['UK_HI'] = transformbackward(transmean + 1.96 * transstd)

    tempdf.eval('UKBias = UK - O', inplace=True)
    tempdf.eval('YBias = Y - O', inplace=True)
    tempdf.eval('QBias = Q - O', inplace=True)
    RMSE = np.sqrt(np.mean(tempdf.UKBias**2))
    NMB = tempdf.UKBias.mean() / tempdf.O.mean()
    R = tempdf.filter(['UK', 'O']).corr().iloc[0, 1]
    YRMSE = np.sqrt(np.mean(tempdf.YBias**2))
    YNMB = tempdf.YBias.mean() / tempdf.O.mean()
    YR = tempdf.filter(['Y', 'O']).corr().iloc[0, 1]
    QRMSE = np.sqrt(np.mean(tempdf.QBias**2))
    QNMB = tempdf.QBias.mean() / tempdf.O.mean()
    QR = tempdf.filter(['Q', 'O']).corr().iloc[0, 1]
    Coverage = (
        (tempdf.O > tempdf.UK_LO)
        & (tempdf.O < tempdf.UK_HI)
    ).mean()
    tempdf.sort_values(by='O', inplace=True)
    # plt.fill_between(tempdf.O, y1=tempdf.UK_LO, y2=tempdf.UK_HI)
    eb = plt.errorbar(
        tempdf.O, tempdf.UK,
        yerr=np.array([tempdf.UK - tempdf.UK_LO, tempdf.UK_HI - tempdf.UK]),
        linestyle='none', marker='o',
        label=f'U: RMSE: {RMSE:.2f}; NMB: {NMB:.1%}, R: {R:.2f}; Cov: {Coverage:.4f}'
    )
    ys = plt.scatter(
        tempdf.O, tempdf.Y, color='b',
        label=f'Y: RMSE: {YRMSE:.2f}; NMB: {YNMB:.1%}, R: {YR:.2f}'
    )
    qs = plt.scatter(
        tempdf.O, tempdf.Q, color='g',
        label=f'Q: RMSE: {QRMSE:.2f}; NMB: {QNMB:.1%}, R: {QR:.2f}'
    )
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    x = np.array([min(xmin, ymin), max(xmax, ymax)])
    plt.plot(x, x, linestyle='-', color='k')
    plt.xlim(*x)
    plt.ylim(*x)
    # plt.scatter(tempdf.O, tempdf.UK)
    lh = [eb, ys, qs]
    plt.legend(lh, [l.get_label() for l in lh], bbox_to_anchor=(0.5, .9), loc='lower center')
    plt.xlabel('Withheld Observed')
    plt.ylabel('Grid Cell prediction (95% CI)')
    plt.savefig(
        date.strftime(
            f'figs/validation{args.validate:02d}_{args.querykey}_%Y%m%d.png'
        )
    )
