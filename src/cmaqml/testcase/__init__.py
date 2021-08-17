__all__ = ['paths']

import os


paths = {
    'config': 'config.json',
    'GRIDCRO': 'GRIDCRO2D.108US2.35L.160101.nc',
    'GRIDDESC': 'GRIDDESC',
    'MOD_O3': 'O3_8HRMAX.LST.Y_24.2016fh.v531.108US2.5-9.nc',
    'OBS_O3': 'daily_44201_20160715.zip',
    'OBS_PM25': 'daily_88101_20160115.zip',
    'MOD_PM25': 'dailyavg.LST.Y_24.2016fh.v531.108US2.01.nc',
    'GPW_POP': 'gpw_v4_une_atotpopbt_densy_108US2.IOAPI.nc',
}
paths['input'] = __path__[0]
for k in paths:
    paths[k] = os.path.join(os.path.dirname(__file__), 'input', paths[k])
