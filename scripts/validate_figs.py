from loadobs import train_and_testdfs
import PseudoNetCDF as pnc
import numpy as np
import matplotlib.pyplot as plt
from opts import cfg
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('validate', type=int, default=1)
parser.add_argument('strftime')
args = parser.parse_args()

if args.validate == 0:
    suffix = '.prod.nc'
else:
    suffix = f'.test{args.validate:02}.nc'

transformforward = cfg['transformforward']
transformbackward = cfg['transformbackward']

testdf = train_and_testdfs(args.validate)['test']

for date in np.unique(testdf.date.dt.to_pydatetime()):
    myvals = testdf.query(f'date == "{date}"')
    ukpath = date.strftime(
        args.strftime
    )
    figpath = os.path.join(
        'figs',
        f'{os.path.basename(ukpath)}_validate.png'
    )

    uk_f = pnc.pncopen(ukpath, format='ioapi')
    i = testdf.loc[:, 'I']
    j = testdf.loc[:, 'J']
    tempdf = testdf.copy()
    tempdf['UK'] = uk_f.variables['UK_TOTAL'][0, 0][j, i]
    tempdf['Y'] = uk_f.variables['Y'][0, 0][j, i]
    tempdf['Q'] = uk_f.variables['Q'][0, 0][j, i]

    transstd = transformforward(uk_f.variables['SD'][0, 0][j, i])
    transmean = transformforward(tempdf['UK'])
    tempdf['UK_LO'] = transformbackward(transmean - 3 * transstd)
    tempdf['UK_HI'] = transformbackward(transmean + 3 * transstd)

    tempdf.eval('UKBias = UK - O', inplace=True)
    tempdf.eval('YBias = Y - O', inplace=True)
    tempdf.eval('QBias = Q - O', inplace=True)
    URMSE = np.sqrt(np.mean(tempdf.UKBias**2))
    UNMB = tempdf.UKBias.mean() / tempdf.O.mean()
    UR = tempdf.filter(['UK', 'O']).corr().iloc[0, 1]
    YRMSE = np.sqrt(np.mean(tempdf.YBias**2))
    YNMB = tempdf.YBias.mean() / tempdf.O.mean()
    YR = tempdf.filter(['Y', 'O']).corr().iloc[0, 1]
    QRMSE = np.sqrt(np.mean(tempdf.QBias**2))
    QNMB = tempdf.QBias.mean() / tempdf.O.mean()
    QR = tempdf.filter(['Q', 'O']).corr().iloc[0, 1]
    # Coverage
    UCov = (
        (tempdf.O > tempdf.UK_LO)
        & (tempdf.O < tempdf.UK_HI)
    ).mean()
    tempdf.sort_values(by='O', inplace=True)
    fig, ax = plt.subplots(1, 1)
    # ax.fill_between(tempdf.O, y1=tempdf.UK_LO, y2=tempdf.UK_HI)
    eb = ax.errorbar(
        tempdf.O, tempdf.UK,
        yerr=np.array([tempdf.UK - tempdf.UK_LO, tempdf.UK_HI - tempdf.UK]),
        linestyle='none', marker='o',
        label=(
            f'U: RMSE: {URMSE:.2f}; NMB: {UNMB:.1%}, R: {UR:.2f};'
            + f' Cov: {UCov:.4f}'
        ), zorder=1
    )
    ys = ax.scatter(
        tempdf.O, tempdf.Y, color='b', marker='^',
        label=f'Y: RMSE: {YRMSE:.2f}; NMB: {YNMB:.1%}, R: {YR:.2f}', zorder=2
    )
    qs = ax.scatter(
        tempdf.O, tempdf.Q, color='g', marker='+',
        label=f'Q: RMSE: {QRMSE:.2f}; NMB: {QNMB:.1%}, R: {QR:.2f}', zorder=3
    )
    xmin, xmax = ax.set_xlim()
    ymin, ymax = ax.set_ylim()
    x = np.array([min(xmin, ymin), max(xmax, ymax)])
    ax.plot(x, x, linestyle='-', color='k')
    ax.set_xlim(*x)
    ax.set_ylim(*x)
    # ax.scatter(tempdf.O, tempdf.UK)
    lh = [eb, ys, qs]
    ax.legend(
        lh, [l.get_label() for l in lh],
        bbox_to_anchor=(0., .9), loc='lower left'
    )
    ax.set_xlabel('Withheld Observed')
    ax.set_ylabel('Grid Cell prediction (95% CI)')
    fig.savefig(figpath)
