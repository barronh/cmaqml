import pandas as pd
import numpy as np


def renamer(k):
    return k.lower().replace(' ', '_').replace('1st', 'first')


def readaqs(cfg, verbose=1):
    aqspath = cfg['obs_path']
    obs_key = cfg["obs_key"]
    obs_defn = cfg["obs_defn"]
    aqsdf = pd.read_csv(aqspath)
    aqsdf['date'] = pd.to_datetime(aqsdf['Date Local'])
    # aqsdf = aqsdf[
    #     (aqsdf['date'].dt.month == 1) & (aqsdf['date'].dt.day == 15)
    # ].copy()
    # Archive testdata, remove derived columns
    # aqsdf.drop('date', axis=1).to_csv(
    #     'input/daily_88101_20160115.csv', index=False
    # )

    aqsdf.rename(columns=renamer, inplace=True)
    aqsdf.eval(f'{obs_key} = {obs_defn}', inplace=True)
    isnegative = aqsdf.loc[:, obs_key] <= 0
    if isnegative.sum() > 0:
        if verbose > 0:
            print(
                'Removing negative values:',
                isnegative.sum(), np.where(isnegative)
            )
        aqsdf.drop(aqsdf[isnegative].index, inplace=True)
        if verbose > 0:
            print('Done', aqsdf.shape)

    return aqsdf
