__all__ = ['loadobs', 'train_and_testdfs']

from .aqs import readaqs


def selectreader(obs_format):
    obs_key = obs_format.lower()
    _keys = {
        'aqs': readaqs
    }
    if obs_key not in _keys:
        raise IOError(f'Unknown format: choices {list(_keys)}')

    return _keys[obs_key]


def loadobs(cfg, gf, metafiles):
    """
    cfg : mappable
        dictionary of options
    gf : pseudonetcdf
        supports ll2ij and ll2xy
    metafiles: list of files
        see add_gridded_meta
    """
    reader = selectreader(cfg['obs_format'])
    obsdf = reader(cfg)
    i, j = gf.ll2ij(
        obsdf.longitude.values,
        obsdf.latitude.values, clean='mask'
    )
    x, y = gf.ll2xy(
        obsdf.longitude.values,
        obsdf.latitude.values
    )

    obsdf['I'] = i
    obsdf['J'] = j
    obsdf['X'] = x
    obsdf['Y'] = y
    obsdf = obsdf.loc[~(i.mask | j.mask)]
    obsdf = add_gridded_meta(obsdf, metafiles)
    return obsdf


def add_gridded_meta(df, metafiles):
    """
    Add meta data from gridded files (e.g., GRIDCRO2D)

    Arguments
    ---------
    df : pandas.DataFrame
        must include longitude and latitude in decimal degrees
    metafiles : list of PseudoNetCDF files
        each file must support ll2ij and contain only variables
        with the last two dimensions j, i
    """
    for mf in metafiles:
        i, j = mf.ll2ij(df.longitude, df.latitude)
        for varkey, var in mf.variables.items():
            df.loc[:, varkey] = var[..., j, i].squeeze()

    return df


def train_and_testdfs(df, group_keys, random_state):
    if group_keys is None:
        gdf = df
    else:
        gdf = df.groupby(group_keys, as_index=False)

    testdf = gdf.groupby(['date'], as_index=False).sample(
        frac=.1, replace=False, random_state=random_state
    )
    traindf = df.drop(df.index)
    return dict(train=traindf, test=testdf)
