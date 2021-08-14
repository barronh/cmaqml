__all__ = ['loadcfg', 'loadmetafiles']

import json
import numpy as np

def removecomments(cfg):
    for key in list(cfg):
        if key.startswith('_'):
            cfg.pop(key)
        else:
            val = cfg[key]
            if isinstance(val, dict):
                removecomments(val)
            

def loadcfg(cfgpath, comments=False):
    cfg = json.load(open(cfgpath))
    if 'regression_options' in cfg:
        transform = cfg['regression_options']['transform']
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
    
    removecomments(cfg)

    return cfg

def loadmetafiles(cfg):
    import PseudoNetCDF as pnc
    
    # Create a list of meta files
    metafiles = []
    for gmcfg in cfg['gridded_meta']:
        mf = pnc.pncopen(
            gmcfg['path_template'],
            format=gmcfg.get('format', 'ioapi')
        )
        smf = mf.subset(gmcfg['var_keys'])
        if 'slice' in gmcfg:
            tmf = smf.slice(**gmcfg['slice'])
        else:
            tmf = smf

        for key in tmf.variables:
            if key not in gmcfg['var_keys']:
                del tmf.variables[key]

        metafiles.append(tmf)
    
    return metafiles
