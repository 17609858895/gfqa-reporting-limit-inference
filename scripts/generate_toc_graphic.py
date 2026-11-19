"""Table-of-contents graphic at its final 3.25 x 1.75 in size, drawn from completed analyses.

Left: station blind-spot shares (as Fig. 1a). Upper right: cadmium event prevalence under the
reporting rules (as Fig. 4a). Lower right: annual cadmium screening in France (as Fig. 5e).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from figkit_compact import FIG_ROOT, INK, ROLE, SEQ_LAV, basemap, setup, _palette_audit
import generate_main_figures_06_10 as g610
import monitoring_panels as mon

WIDTH, HEIGHT, DPI = 3.25, 1.75, 600
OUT = FIG_ROOT / 'TOC_graphic'


def draw():
    setup()
    plt.rcParams.update({'font.size': 7.5, 'axes.linewidth': .7, 'xtick.major.width': .6, 'ytick.major.width': .6,
                         'xtick.major.size': 2.2, 'ytick.major.size': 2.2, 'xtick.major.pad': 1.5,
                         'ytick.major.pad': 1.5, 'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8})
    fig = plt.figure(figsize=(WIDTH, HEIGHT))

    # Blind-spot map
    ax = fig.add_axes([0.0, 0.215, 0.525, 0.60])
    basemap(ax, extent=(-128, 42, 10, 70), aspect=1.95)
    ax.set_anchor('C')
    s = mon.smap
    ax.scatter(s['x'], s['y'], c=s['und'] * 100, cmap=SEQ_LAV, vmin=0, vmax=100, s=2.2, edgecolors=INK,
               linewidths=.08, zorder=3)
    fig.text(.02, .93, 'Reporting-limit blind spot', fontsize=8, fontweight='bold', color=INK, va='center')
    fig.text(.02, .835, 'Non-detect limit above the threshold', fontsize=6.6, color=INK, va='center')
    cax = fig.add_axes([0.06, 0.13, 0.36, 0.035])
    import matplotlib as mpl
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 100), cmap=SEQ_LAV), cax=cax,
                      orientation='horizontal', ticks=[0, 50, 100])
    cb.ax.tick_params(labelsize=6.3, length=1.8, width=.5, pad=1)
    cb.outline.set_linewidth(.5)
    cb.set_label('Blind-spot share of events (%)', fontsize=6.6, labelpad=1.5)

    # Cadmium prevalence under reporting rules
    r = g610.DB[g610.DB['label'].eq('Cd')].iloc[0]
    bx = fig.add_axes([0.60, 0.58, 0.37, 0.24])
    bx.hlines(0, r.ndc * 100, r.dl * 100, color='#D9DDE2', lw=6, zorder=1)
    bx.scatter([r.half * 100], [0], marker='s', s=26, color=ROLE['half'], edgecolor=INK, lw=.5, zorder=3)
    bx.scatter([r.model * 100], [0], marker='o', s=26, color=ROLE['model'], edgecolor=INK, lw=.5, zorder=3)
    bx.text(r.half * 100, .55, f'Half-limit\n{r.half * 100:.0f}%', ha='center', va='bottom', fontsize=6.6,
            color=INK, linespacing=.95)
    bx.text(r.model * 100, .55, f'Model\n{r.model * 100:.0f}%', ha='center', va='bottom', fontsize=6.6,
            color=INK, linespacing=.95)
    bx.set(xlim=(0, 70), ylim=(-.6, 1.9), yticks=[])
    bx.set_xticks([0, 20, 40, 60])
    for side in ['top', 'right', 'left']:
        bx.spines[side].set_visible(False)
    bx.set_xlabel('Cd exceedance (%)', fontsize=6.8, labelpad=1)
    fig.text(.60, .93, 'Same records, two rules', fontsize=7.2, fontweight='bold', color=INK, va='center')

    # French cadmium screening series
    tx = fig.add_axes([0.60, 0.13, 0.37, 0.25])
    series = g610.SERIES[('Cd', 'France')]
    tx.fill_between(series.index, 0, series['und'] * 100, color=ROLE['nd_undetermined'], alpha=.35, lw=0)
    tx.plot(series.index, series['half'] * 100, color=ROLE['half'], lw=1.1, marker='s', ms=1.8, mec=INK, mew=.3)
    tx.plot(series.index, series['model'] * 100, color=ROLE['model'], lw=1.1, marker='o', ms=1.8, mec=INK, mew=.3)
    tx.set(ylim=(0, 105), yticks=[0, 50, 100], xticks=[2005, 2015])
    for side in ['top', 'right']:
        tx.spines[side].set_visible(False)
    tx.text(.98, .95, 'France, Cd', transform=tx.transAxes, ha='right', va='top', fontsize=6.6, color=INK)
    tx.set_ylabel('%', fontsize=6.8, labelpad=1)
    for axis in [bx, tx]:
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontweight('bold')
            label.set_color(INK)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = draw()
    audit = _palette_audit(fig)
    assert not audit['unexpected_colors'], audit['unexpected_colors']
    stem = OUT / 'TOC_graphic'
    fig.savefig(stem.with_suffix('.png'), dpi=DPI, facecolor='white', metadata={'Software': 'Matplotlib'})
    fig.savefig(stem.with_suffix('.pdf'), dpi=DPI, facecolor='white',
                metadata={'Creator': 'Scientific figure export', 'Author': 'CHONG LIU'})
    plt.close(fig)
    with Image.open(stem.with_suffix('.png')) as im:
        assert im.size == (round(WIDTH * DPI), round(HEIGHT * DPI)), im.size
        im.convert('RGB').save(stem.with_suffix('.tif'), dpi=(DPI, DPI), compression='tiff_lzw')
    print(stem, 'size', WIDTH, 'x', HEIGHT, 'in at', DPI, 'dpi')


if __name__ == '__main__':
    main()
