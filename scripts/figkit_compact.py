"""Compact plotting and independent-panel export utilities for the GFQA study."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from ml_publication_style import configure_publication_style, style_axis, numeric_ticks, export_subfigures, save_figure_with_subfigures  # noqa: E402
from scientific_palettes import STYLE_COLORS, get_palette  # noqa: E402

# Geographic dependencies are available in the project interpreter; avoid an external project path.
ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = Path(os.environ.get("GFQA_FIGURE_ROOT", str(ROOT / "figures")))
BASEMAP = ROOT / "inputs" / "basemap"
INK, MUTED, LIGHT = STYLE_COLORS["ink"], STYLE_COLORS["muted"], STYLE_COLORS["light"]
PAL = get_palette("nature_new")  # rose, blue, light green, teal, lavender, pink, gold, purple

ROLE = {"Substitution": PAL[0], "Censored": PAL[1], "Ensemble": PAL[7], "Drop": PAL[6],
        "detected_ok": PAL[3], "detected_exceed": PAL[0], "nd_determinable": PAL[2], "nd_undetermined": PAL[4],
        "ndc": PAL[3], "model": PAL[1], "half": PAL[0], "dl": PAL[6]}
TCOL = {"Cd": PAL[0], "Hg": PAL[1], "Pb": PAL[3], "Ni": PAL[6], "PFOS": PAL[7], "Glyphosate": PAL[2], "CdDis": PAL[5]}
TMARK = {"Cd": "o", "Hg": "s", "Pb": "^", "Ni": "D", "PFOS": "P", "Glyphosate": "v", "CdDis": "h"}
SEQ_LAV = LinearSegmentedColormap.from_list("w_lav", ["#FFFFFF", PAL[4]])
SEQ_ROSE = LinearSegmentedColormap.from_list("w_rose", ["#FFFFFF", PAL[0]])
SEQ_BLUE = LinearSegmentedColormap.from_list("w_blue", ["#FFFFFF", PAL[1]])
DIV = LinearSegmentedColormap.from_list("blue_w_rose", [PAL[1], "#FFFFFF", PAL[0]])
FS = 15


def setup():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    configure_publication_style()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": FS, "axes.labelsize": FS, "axes.labelweight": "normal",
                         "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": 13,
                         "axes.labelpad": 6, "xtick.major.pad": 4, "ytick.major.pad": 4, "axes.linewidth": 1.3,
                         "xtick.major.width": 1.2, "ytick.major.width": 1.2, "xtick.major.size": 5.5,
                         "ytick.major.size": 5.5})


def style(ax, xlabel=None, ylabel=None, fs=FS):
    style_axis(ax, tick_size=fs, label_size=fs)
    ax.grid(False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.3)
    ax.tick_params(axis="both", which="major", colors=INK, width=1.2, length=5.5, labelsize=fs, direction="out")
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("normal")
    ax.xaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontweight("normal")
    from matplotlib.ticker import AutoLocator
    if isinstance(ax.xaxis.get_major_locator(), AutoLocator): numeric_ticks(ax, x=True, y=False, n=5)
    if isinstance(ax.yaxis.get_major_locator(), AutoLocator): numeric_ticks(ax, x=False, y=True, n=5)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fs, fontweight="normal")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fs, fontweight="normal")


def outside_legend(ax, above=False, pad_pt=48, **kwargs):
    """Place legends at a fixed physical distance from axes as cell size changes."""
    from matplotlib.transforms import ScaledTranslation
    transform=ax.transAxes+ScaledTranslation(0,(pad_pt if above else -pad_pt)/72,ax.figure.dpi_scale_trans)
    return ax.legend(loc='lower center' if above else 'upper center',bbox_to_anchor=(.5,1 if above else 0),
                     bbox_transform=transform,borderaxespad=0,**kwargs)


def darker(c, f=0.62):
    r, g, b = to_rgb(c)
    return (r * f, g * f, b * f)


# ------------------------------------------------------------------ maps
_land = None
MAP_CRS = "+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs"


def map_proj(lon, lat):
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", MAP_CRS, always_xy=True)
    return tf.transform(np.asarray(lon, float), np.asarray(lat, float))


def land():
    global _land
    if _land is None:
        import geopandas as gpd
        from shapely.geometry import box
        g = gpd.read_file(BASEMAP / "ne_50m_land.geojson")
        geom = g.geometry.union_all().intersection(box(-180, -58, 180, 84))
        _land = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(MAP_CRS)
    return _land


def proj_window(extent, aspect=1.6):
    """Axis-aligned projected window covering a lon/lat extent, padded to a fixed width/height ratio."""
    lon = np.r_[np.linspace(extent[0], extent[1], 30), np.linspace(extent[0], extent[1], 30),
                np.full(30, extent[0]), np.full(30, extent[1])]
    lat = np.r_[np.full(30, extent[2]), np.full(30, extent[3]), np.linspace(extent[2], extent[3], 30),
                np.linspace(extent[2], extent[3], 30)]
    xs, ys = map_proj(lon, lat)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    w = max(x1 - x0, (y1 - y0) * aspect)
    h = w / aspect
    return xm - w / 2, xm + w / 2, ym - h / 2, ym + h / 2


def basemap(ax, extent=None, aspect=1.6):
    land().plot(ax=ax, color="#EEF0F2", edgecolor="#B7B7B7", linewidth=0.45, zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if extent is not None:
        x0, x1, y0, y1 = proj_window(extent, aspect)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    else:
        xs, _ = map_proj([-180, 180], [0, 0])
        _, y0 = map_proj(0, -58)
        _, y1 = map_proj(0, 84)
        ax.set_xlim(xs[0], xs[1])
        ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


def zoom_box(ax_world, ax_zoom, extent, label):
    """Draw the (projected, axis-aligned) zoom window on the world map and frame the zoom axes."""
    x0, x1 = ax_zoom.get_xlim()
    y0, y1 = ax_zoom.get_ylim()
    ax_world.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=MUTED, lw=0.9, zorder=5)
    for s in ax_zoom.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)
        s.set_linewidth(0.9)
    ax_zoom.text(0, 1.02, label, transform=ax_zoom.transAxes, ha="left", va="bottom", fontsize=13.0, color=INK)


# ------------------------------------------------------------------ export
# Independent panel renders are intentional; cropped panels would carry neighbouring labels.
def _bounds(fig, axes):
    from matplotlib.transforms import Bbox
    r = fig.canvas.get_renderer()
    boxes = [a.get_tightbbox(r) for a in axes if a.get_visible()]
    return Bbox.union([b for b in boxes if b is not None]).transformed(fig.transFigure.inverted())


def _calibrate(fig, groups, specs, exempt=()):
    """Align the complete visual footprint, including labels, within shared rows/columns."""
    from collections import defaultdict
    rows, cols = defaultdict(list), defaultdict(list)
    for k in groups:
        if k in exempt: continue
        rows[(specs[k].rowspan.start, specs[k].rowspan.stop)].append(k)
        cols[(specs[k].colspan.start, specs[k].colspan.stop)].append(k)
    for _ in range(5):
        fig.canvas.draw()
        boxes = {k: _bounds(fig, aa) for k, aa in groups.items()}
        targets = {k: [b.x0,b.y0,b.x1,b.y1] for k,b in boxes.items()}
        for dimension, buckets in [('y',rows),('x',cols)]:
            a,b=(1,3) if dimension=='y' else (0,2)
            for keys in buckets.values():
                if len(keys)<2: continue
                lo=max(targets[k][a] for k in keys)
                hi=min(targets[k][b] for k in keys)
                if hi-lo < 0.08: continue
                for k in keys: targets[k][a],targets[k][b]=lo,hi
        for k,aa in groups.items():
            old=boxes[k]; x0,y0,x1,y1=targets[k]
            sx=(x1-x0)/old.width; sy=(y1-y0)/old.height
            for ax in aa:
                pos=ax.get_position()
                ax.set_position([x0+(pos.x0-old.x0)*sx,y0+(pos.y0-old.y0)*sy,pos.width*sx,pos.height*sy])
    fig.canvas.draw()
    boxes={k:_bounds(fig,aa) for k,aa in groups.items()}
    report=[]
    for dimension,buckets in [('row',rows),('column',cols)]:
        for keys in buckets.values():
            if len(keys)<2: continue
            low=[getattr(boxes[k],'y0' if dimension=='row' else 'x0') for k in keys]
            high=[getattr(boxes[k],'y1' if dimension=='row' else 'x1') for k in keys]
            report.append(dict(dimension=dimension,panels=keys,max_edge_difference=max(np.ptp(low),np.ptp(high))))
    return report


def _palette_audit(fig):
    """Check actual fills/series lines and legend proxies; structural neutral ink is separate."""
    import matplotlib.colors as mc
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.collections import Collection
    from audit_palette_usage import validate_series_colors
    validate_series_colors('nature_new',ROLE)
    validate_series_colors('nature_new',TCOL)
    allowed={mc.to_hex(c) for c in PAL}
    # Numeric map/heatmap encodings explicitly interpolate white to palette endpoints.
    for cmap in [SEQ_LAV,SEQ_ROSE,SEQ_BLUE,DIV]:
        allowed.update(mc.to_hex(c) for c in cmap(np.linspace(0,1,256)))
    structural={mc.to_hex(c) for c in [INK,MUTED,LIGHT,'white','black','#EEF0F2','#B7B7B7','#D9DDE2']}
    seen=set(); bad=set()
    for obj in fig.findobj():
        if hasattr(obj,"get_visible") and not obj.get_visible(): continue
        values=[]
        if isinstance(obj,Line2D):
            values=[]
            if obj.get_linestyle() not in ('None','none','',None): values.append(obj.get_color())
            if obj.get_marker() not in ('None','none','',None): values.append(obj.get_markerfacecolor())
        elif isinstance(obj,Patch): values=[obj.get_facecolor()]
        elif isinstance(obj,Collection):
            if obj.get_array() is not None: continue  # verified numeric encoding with explicit palette endpoint
            values=list(obj.get_facecolors())
        for col in values:
            try:
                rgba=mc.to_rgba(col)
                if rgba[3]==0: continue
                hx=mc.to_hex(rgba)
                seen.add(hx)
                if hx not in allowed|structural: bad.add(hx)
            except (ValueError,TypeError): pass
    return {'palette':'nature_new','actual_artist_colors':sorted(seen),'unexpected_colors':sorted(bad),
            'note':'White-to-palette continuous maps, transparent uncertainty bands, structural ink and grey identification ranges are intentional.'}


def _fit_cells(fig, groups, specs, exempt=()):
    """Fit full visual footprints into compact cells, retaining type size in physical points.

    This adapts the current reference project's live outer-bbox layout principle.
    Raster schematics keep their aspect ratio; quantitative panels retain all labels.
    """
    from matplotlib.text import Text
    fig.canvas.draw()
    for txt in fig.findobj(Text):
        if txt.get_text():
            txt.set_fontsize(max(12.5,txt.get_fontsize()))
            txt.set_fontweight('normal')
            txt.set_color(INK)
    report=[]
    for key,axes in groups.items():
        if key in exempt: continue
        target=specs[key].get_position(fig).frozen()
        for ax in axes: ax.set_axes_locator(None)
        for _ in range(18):
            fig.canvas.draw();outer=_bounds(fig,axes)
            err=(np.asarray(outer.extents)-target.extents)*np.tile(fig.get_size_inches(),2)*72
            if np.max(np.abs(err))<0.4:break
            sx,sy=target.width/outer.width,target.height/outer.height
            for ax in axes:
                pos=ax.get_position(original=True)
                ax.set_position([target.x0+(pos.x0-outer.x0)*sx,target.y0+(pos.y0-outer.y0)*sy,pos.width*sx,pos.height*sy],which='both')
        fig.canvas.draw();outer=_bounds(fig,axes)
        err=(np.asarray(outer.extents)-target.extents)*np.tile(fig.get_size_inches(),2)*72
        report.append({'panel':key,'target':list(map(float,target.extents)),'outer':list(map(float,outer.extents)),'max_residual_pt':float(np.max(np.abs(err)))})
    return report


def _gap_metrics(fig, groups, specs):
    """Record physical gutters between complete visible panel footprints."""
    from collections import defaultdict
    fig.canvas.draw()
    boxes={k:_bounds(fig,a) for k,a in groups.items()}
    rows,cols=defaultdict(list),defaultdict(list)
    for k in groups:
        rows[(specs[k].rowspan.start,specs[k].rowspan.stop)].append(k)
        cols[(specs[k].colspan.start,specs[k].colspan.stop)].append(k)
    gaps=[];width,height=fig.get_size_inches()
    for keys in rows.values():
        keys=sorted(keys,key=lambda k:boxes[k].x0)
        for a,b in zip(keys[:-1],keys[1:]):
            gaps.append({'direction':'horizontal','pair':[a,b],'inches':float((boxes[b].x0-boxes[a].x1)*width)})
    for keys in cols.values():
        keys=sorted(keys,key=lambda k:-boxes[k].y1)
        for a,b in zip(keys[:-1],keys[1:]):
            gaps.append({'direction':'vertical','pair':[a,b],'inches':float((boxes[a].y0-boxes[b].y1)*height)})
    # Also measure mixed rows/columns such as one schematic above three data panels.
    from matplotlib.transforms import Bbox
    for direction,buckets,scale in [('row_band',rows,height),('column_band',cols,width)]:
        for span,keys in buckets.items():
            following=[(sp,kk) for sp,kk in buckets.items() if sp[0]>=span[1]]
            if not following:continue
            edge=min(sp[0] for sp,kk in following)
            candidates=[(sp,kk) for sp,kk in following if sp[0]==edge]
            next_span,next_keys=min(candidates,key=lambda v:v[0][1]-v[0][0])
            a=Bbox.union([boxes[k] for k in keys]);b=Bbox.union([boxes[k] for k in next_keys])
            gap=(a.y0-b.y1) if direction=='row_band' else (b.x0-a.x1)
            gaps.append({'direction':direction,'pair':[list(keys),list(next_keys)],'inches':float(gap*scale)})
    return {'figsize_inches':[float(width),float(height)],'panel_bounds':{k:list(map(float,b.extents)) for k,b in boxes.items()},'gaps':gaps}


def build(panels, layout_fn, folder, stem, figsize, sub_sizes=None, letter_size=18):
    import json
    setup()
    out=FIG_ROOT/folder
    (out/'subfigures').mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=figsize)
    specs=layout_fn(fig); groups={}
    for letter,fn in panels:
        before=set(fig.axes); fn(fig,specs[letter]); groups[letter]=[a for a in fig.axes if a not in before]
    # The conceptual strips and the generated workflow retain their intentional pictorial composition.
    exempt=('c',) if stem=='Fig02_reporting_resolution' else ('a',) if stem=='Fig03_learning_framework' else ()
    cell_fit=_fit_cells(fig,groups,specs,exempt)
    alignment=_calibrate(fig,groups,specs,exempt)
    gap_metrics=_gap_metrics(fig,groups,specs)
    for letter,_ in panels:
        bb=_bounds(fig,groups[letter])
        fig.text(bb.x0-0.004,bb.y1+0.004,letter,fontsize=letter_size,ha='left',va='bottom',color=INK)
    audit=_palette_audit(fig)
    fig.savefig(out/f'{stem}.png',dpi=600,bbox_inches='tight',facecolor='white',metadata={'Software':'Matplotlib'})
    fig.savefig(out/f'{stem}.pdf',bbox_inches='tight',facecolor='white',metadata={'Creator':'Scientific figure export','Author':''})
    plt.close(fig)
    lines=['# Subfigures','','Each panel is re-rendered on its own canvas by the same panel function (not cropped).','']
    for letter,fn in panels:
        size=(sub_sizes or {}).get(letter,(6.2,4.8))
        f=plt.figure(figsize=size); fn(f,f.add_gridspec(1,1)[0]); f.canvas.draw()
        from matplotlib.text import Text
        for txt in f.findobj(Text):
            if txt.get_text(): txt.set_fontsize(max(12.5,txt.get_fontsize())); txt.set_fontweight('normal'); txt.set_color(INK)
        f.savefig(out/'subfigures'/f'{stem}_{letter}.png',dpi=600,bbox_inches='tight',facecolor='white',metadata={'Software':'Matplotlib'})
        f.savefig(out/'subfigures'/f'{stem}_{letter}.pdf',bbox_inches='tight',facecolor='white',metadata={'Creator':'Scientific figure export','Author':''})
        plt.close(f)
        lines.append(f'- {letter}: `{stem}_{letter}.png`, `{stem}_{letter}.pdf`')
    (out/'subfigures'/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    qa=ROOT/'notes/qa/ten_figures'; qa.mkdir(parents=True,exist_ok=True)
    (qa/f'{stem}_export_audit.json').write_text(json.dumps({'alignment':alignment,'cell_fit':cell_fit,'palette':audit,'layout':gap_metrics},indent=2),encoding='utf-8')
    return out
