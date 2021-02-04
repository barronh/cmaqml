import PseudoNetCDF as pnc
import pandas as pd
from opts import cfg

aqspath = cfg['obs_path']
poppath = cfg['pop_path']
gdpath = cfg['griddesc_path']
domname = cfg['domain']
obs_key = cfg["obs_key"]
obs_defn = cfg["obs_defn"]

gf = pnc.pncopen(gdpath, format='griddesc', GDNAM=domname)

del gf.variables['TFLAG']

gf.TSTEP = 240000
gf.SDATE = 1970001
gf.STIME = 0
gf.updatetflag(overwrite=True)
popf = pnc.pncopen(poppath, format='ioapi')
aqsdf = pd.read_csv(aqspath)
aqsdf['date'] = pd.to_datetime(aqsdf['Date Local'])
# aqsdf = aqsdf[
#     (aqsdf['date'].dt.month == 7) & (aqsdf['date'].dt.day == 15)
# ].copy()
# Archive testdata, remove derived columns
# aqsdf.drop('date', axis=1).to_csv(
#     'input/daily_44201_20160715.zip', index=False
# )


def renamer(k):
    return k.lower().replace(' ', '_').replace('1st', 'first')


aqsdf.rename(columns=renamer, inplace=True)
i, j = gf.ll2ij(
    aqsdf.longitude.values,
    aqsdf.latitude.values, clean='mask'
)
x, y = gf.ll2xy(
    aqsdf.longitude.values,
    aqsdf.latitude.values
)

aqsdf['I'] = i
aqsdf['J'] = j
aqsdf['X'] = x
aqsdf['Y'] = y
aqsdf = aqsdf.loc[~(i.mask | j.mask)]
# time 3 is nominal present day
DENS = popf.variables['DENS'][3, 0]

aqsdf['pop_per_km2'] = DENS[aqsdf.J.values, aqsdf.I.values]
aqsdf.eval(f'{obs_key} = {obs_defn}', inplace=True)


def train_and_testdfs(random_state):
    testdf = aqsdf.groupby(['date'], as_index=False).sample(
        frac=.1, replace=False, random_state=random_state
    )
    traindf = aqsdf.drop(testdf.index)
    return dict(train=traindf, test=testdf)
