import PseudoNetCDF as pnc
import matplotlib.pyplot as plt
import pycno
import argparse
from opts import cfg

parser = argparse.ArgumentParser()
parser.add_argument('date', help='YYYYMMDD')
args = parser.parse_args()
date = args.date

popf = pnc.pncopen(cfg['pop_path'], format='ioapi')
# time 3 is nominal present day
DENS = popf.variables['DENS'][3, 0]

blendf = pnc.pncopen(f'blend/UK.{date}.BLEND_BLEND.prod.nc', format='ioapi')
blendrf = pnc.pncopen(f'blend/UK.{date}.BLEND_RUR.prod.nc', format='ioapi')
blenduf = pnc.pncopen(f'blend/UK.{date}.BLEND_URB.prod.nc', format='ioapi')

cno = pycno.cno(proj=blendf.getproj(withgrid=True))
norm = plt.Normalize(vmin=20, vmax=100)


def plotmasked(ax, var, norm):
    # apply some mask
    # mvar = np.ma.masked_where(DENS == 0, var[:])
    mvar = var
    return ax.pcolormesh(mvar, norm=norm)


def composite(key, typ):
    nwf = pnc.pncopen(f'output/UK.{date}.WN_{typ}.prod.nc', format='ioapi')
    nef = pnc.pncopen(f'output/UK.{date}.EN_{typ}.prod.nc', format='ioapi')
    swf = pnc.pncopen(f'output/UK.{date}.WS_{typ}.prod.nc', format='ioapi')
    sef = pnc.pncopen(f'output/UK.{date}.ES_{typ}.prod.nc', format='ioapi')
    blendf = pnc.pncopen(f'output/UK.{date}.ALL_{typ}.prod.nc', format='ioapi')

    fig = plt.figure(figsize=(6, 8), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)
    nwax = fig.add_subplot(gs[0, 0])
    nwax.set_xlabel('NW')
    neax = fig.add_subplot(gs[0, 1])
    neax.set_xlabel('NE')
    swax = fig.add_subplot(gs[3, 0])
    swax.set_xlabel('SW')
    seax = fig.add_subplot(gs[3, 1])
    seax.set_xlabel('SE')
    mainax = fig.add_subplot(gs[1:3, :])

    # norm = plt.Normalize()

    p = plotmasked(mainax, var=blendf.variables[key][0, 0], norm=norm)
    plt.colorbar(p, ax=mainax, label='ppm')
    p = plotmasked(neax, var=nef.variables[key][0, 0], norm=norm)
    p = plotmasked(nwax, var=nwf.variables[key][0, 0], norm=norm)
    p = plotmasked(seax, var=sef.variables[key][0, 0], norm=norm)
    p = plotmasked(swax, var=swf.variables[key][0, 0], norm=norm)
    for ax in [neax, nwax, seax, swax, mainax]:
        cno.draw('MWDB_Coasts_USA_3.cnob', ax=ax)
    return fig


def ruralurban(key):
    fig, axx = plt.subplots(
        3, 1, figsize=(6, 8), sharex=True, sharey=True,
        gridspec_kw=dict(left=.15, right=0.8, top=.95, bottom=.05)
    )

    cax = fig.add_axes([.825, .1, .025, .8])
    # norm = plt.Normalize()
    p = axx[1].pcolormesh(blendf.variables[key][0, 0], norm=norm)
    cno.draw('MWDB_Coasts_USA_3.cnob', ax=axx[1])
    axx[1].set_ylabel('Blend')
    fig.colorbar(p, label='ppb', cax=cax)
    axx[0].pcolormesh(blenduf.variables[key][0, 0], norm=norm)
    axx[0].set_ylabel('Urban')
    cno.draw('MWDB_Coasts_USA_3.cnob', ax=axx[0])
    axx[2].pcolormesh(blendrf.variables[key][0, 0], norm=norm)
    axx[2].set_ylabel('Rural')
    cno.draw('MWDB_Coasts_USA_3.cnob', ax=axx[2])
    return fig


for key in ['UK_TOTAL', 'Y']:
    fig = composite(key, 'URB')
    fig.suptitle(f'Urban {key} {date}')
    fig.savefig(f'figs/Urban_{key}_{date}.png')

    fig = composite(key, 'RUR')
    fig.suptitle(f'Rural {key} {date}')
    fig.savefig(f'figs/Rural_{key}_{date}.png')

    fig = ruralurban(key)
    fig.savefig(f'figs/Blend_{key}_{date}.png')
