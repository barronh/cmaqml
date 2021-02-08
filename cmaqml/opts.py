__all__ = ['loadcfg']

import json
import numpy as np


def loadcfg(cfgpath):
    cfg = json.load(open(cfgpath))
    transform = cfg['transform']
    cfg["transformforward"] = {
        'none': lambda x: x,
        'sqrt': lambda x: np.sqrt(x),
        'log': lambda x: np.log(x),
    }[transform]
    cfg["transformbackward"] = {
        'none': lambda x: x,
        'sqrt': lambda x: x**2,
        'log': lambda x: np.exp(x),
    }[transform]
    return cfg
