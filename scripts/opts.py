__all__ = ['cfg']

import json
import numpy as np


cfg = json.load(open('config.json'))
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
