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

    transstd = transformforward(fused_uk_f.variables['SD'][0, 0][j, i])
    transmean = transformforward(tempdf['UK'])
    tempdf['UK_LO'] = transformbackward(transmean - 1.96 * transstd)
    tempdf['UK_HI'] = transformbackward(transmean + 1.96 * transstd)

    tempdf.eval('Bias = UK - O', inplace=True)
    RMSE = np.sqrt(np.mean(tempdf.Bias**2))
    NMB = tempdf.Bias.mean() / tempdf.O.mean()
    R = tempdf.filter(['UK', 'O']).corr().iloc[0, 1]
    Coverage = (
        (tempdf.O > tempdf.UK_LO)
        & (tempdf.O < tempdf.UK_HI)
    ).mean()
    tempdf.sort_values(by='O', inplace=True)
    # plt.fill_between(tempdf.O, y1=tempdf.UK_LO, y2=tempdf.UK_HI)
    plt.errorbar(
        tempdf.O, tempdf.UK,
        yerr=np.array([tempdf.O - tempdf.UK_LO, tempdf.UK_HI - tempdf.O]),
        linestyle='none', marker='o'
    )
    # plt.scatter(tempdf.O, tempdf.UK)
    plt.title(
        f'RMSE: {RMSE:.2f}; NMB: {NMB:.1%}, R: {R:.2f}; Cov: {Coverage:.4f}'
    )
    plt.xlabel('Withheld Observed')
    plt.ylabel('Grid Cell prediction (95% CI)')
    plt.savefig(
        date.strftime(
            f'figs/validation{args.validate:02d}_{args.querykey}_%Y%m%d.png'
        )
    )
