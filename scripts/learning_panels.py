"""Censored-learning architecture and fitted scale diagnostics."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import norm

from figkit_compact import FS, INK, MUTED, PAL, ROLE, ROOT, SEQ_BLUE, TCOL, TMARK, build, darker, style, outside_legend

sys.path.insert(0, str(ROOT / "scripts"))
from censml import fit_sigma  # noqa: E402
from ensembles import CensoredSuperLearner, _spread  # noqa: E402

RES = ROOT / "results"
METALS = ["Cd", "Hg", "Pb"]
OOF = {t: pd.read_parquet(RES / "preds" / f"oof_{t}_real.parquet") for t in METALS}
LEARN = [c for c in OOF["Cd"].columns if c not in ("y_lo", "y_hi", "station", "truth")]
FAM = json.load(open(RES / "bench_Cd_real.json", encoding="utf-8"))
W = {t: json.load(open(RES / f"bench_{t}_real.json", encoding="utf-8"))["weights"] for t in METALS}


def box(ax, x, y, w, h, fc, text=None, fs=12, ec=INK, lw=0.9, alpha=1.0, ha="center", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6", facecolor=fc, edgecolor=ec,
                                lw=lw, alpha=alpha))
    if text:
        ax.text(x + (w / 2 if ha == "center" else 0.8), y + h / 2, text, ha=ha, va="center", fontsize=fs, color=tc)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, color=INK, lw=1.4))


def panel_a(fig, ss):
    """Pictorial background with editable, regular-weight labels; no data plot is generated."""
    import matplotlib.image as mpimg
    ax=fig.add_subplot(ss)
    if ss.get_gridspec().nrows==1 and ss.get_gridspec().ncols==1:
        ax.set_position([0.01,0.01,0.98,0.98])
    ax.imshow(mpimg.imread(ROOT/'assets/method_background.png'),extent=(0,1,1,0),aspect=1/3)
    ax.set_xlim(0,1); ax.set_ylim(1,0); ax.axis('off')
    fig.canvas.draw()
    scale=ax.get_position().width*fig.get_size_inches()[0]/17
    labels=[('Freshwater\nobservations',.074,.31,'center'),('Interval\nlabels',.21,.31,'center'),
            ('Detected',.229,.418,'center'),('Non-detect\ninterval',.21,.666,'center'),
            ('Model benchmark',.315,.062,'left'),('22 base\nlearners',.402,.188,'center'),
            ('Out-of-fold\npredictions',.579,.188,'center'),('CSL /\nD-CSL',.724,.188,'center'),
            ('Held-out\nstations',.925,.18,'center'),('Known-truth\ntests',.925,.345,'center'),
            ('Archive inference',.315,.528,'left'),('Station\ncross-fitting',.387,.667,'center'),
            ('LGB-Tobit',.549,.628,'center'),('Exceedance\nprobability',.758,.58,'center'),
            ('Threshold\nprevalence',.758,.725,'center'),('Within-station\ntrends',.758,.872,'center'),
            ('Follow-up\npriorities',.920,.845,'center')]
    narrow={'Detected','Held-out\nstations','Known-truth\ntests','Exceedance\nprobability','Threshold\nprevalence','Within-station\ntrends'}
    for text,x,y,ha in labels:
        fs=21 if text=='Detected' else 22 if text in narrow else 26
        ax.text(x,y,text,ha=ha,va='center',fontfamily='Arial',fontweight='normal',fontsize=fs*scale,
                color=INK,linespacing=1.05)


def panel_b(fig, ss):
    ax = fig.add_subplot(ss)
    mu, s, dl, eqs = -1.45, 0.45, np.log10(0.2), np.log10(0.08)
    x = np.linspace(-3.4, 0.2, 400)
    f = norm.pdf(x, mu, s)
    ax.plot(x, f, color=PAL[6], lw=1.6, ls="--", label="Predictive density")
    t = np.where(x <= dl, f / norm.cdf((dl - mu) / s), 0)
    ax.plot(x[x <= dl], t[x <= dl], color=PAL[1], lw=2.2, label="Given Y < DL")
    m = (x >= eqs) & (x <= dl)
    ax.fill_between(x[m], 0, t[m], color=PAL[0], alpha=0.8, lw=0, label="P(Y > EQS | Y < DL)")
    ax.axvline(dl, color=INK, lw=1.4)
    ax.axvline(eqs, color=INK, lw=1.4, ls=":")
    ax.text(dl + 0.04, 0.06 * t.max(), "DL", fontsize=13.0, color=INK)
    ax.text(eqs - 0.05, 0.06 * t.max(), "EQS", fontsize=13.0, color=INK, ha="right")
    p = (norm.cdf((dl - mu) / s) - norm.cdf((eqs - mu) / s)) / norm.cdf((dl - mu) / s)
    xm = (eqs + dl) / 2
    ax.annotate(f"{p:.2f}", xy=(xm, 0.35 * t.max()), xytext=(xm + 0.5, 0.62 * t.max()), fontsize=13.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=1))
    ax.set_xticks([-3, -2, -1, 0], ["0.001", "0.01", "0.1", "1"])
    ax.set_ylim(0, t.max() * 1.45)
    style(ax, "Cd (µg L$^{-1}$)", "Density")
    outside_legend(ax,above=True,pad_pt=4,ncol=1,fontsize=12.5,frameon=False,handlelength=1.4)


def panel_c(fig, ss):
    g = ss.subgridspec(1, 2, wspace=0.08)
    keep = sorted({k for t in METALS for key in ["CSL", "D-CSL"] for k, v in W[t][key].items() if v > 0.02},
                  key=lambda k: -sum(W[t]["D-CSL"].get(k, 0) + W[t]["CSL"].get(k, 0) for t in METALS))
    axes = []
    for j, key in enumerate(["CSL", "D-CSL"]):
        ax = fig.add_subplot(g[j])
        m = np.array([[W[t][key].get(k, 0.0) for t in METALS] for k in keep])
        im = ax.imshow(m, cmap=SEQ_BLUE, vmin=0, vmax=0.6, aspect="auto")
        for (i, jj), v in np.ndenumerate(m):
            if v > 0.005:
                ax.text(jj, i, f"{v:.2f}", ha="center", va="center", fontsize=11.5, color=INK)
        ax.set_xticks(range(3), METALS)
        ax.set_yticks(range(len(keep)), keep if j == 0 else [])
        ax.set_title(key, fontsize=13.5, color=INK, pad=6)
        style(ax)
        ax.tick_params(axis="y", length=0 if j else 5.5)
        for s_ in ["left", "bottom"]:
            ax.spines[s_].set_visible(False)
        axes.append(ax)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1, aspect=25)
    cb.set_label("Stacking weight", fontsize=13.0)
    cb.ax.tick_params(labelsize=11, length=3)
    cb.outline.set_edgecolor(INK)


_SCALE_CACHE={}

def panel_d(fig, ss):
    ax = fig.add_subplot(ss)
    for t in METALS:
        o = OOF[t]
        Z, lo, hi = o[LEARN].to_numpy(), o["y_lo"].to_numpy(), o["y_hi"].to_numpy()
        if t not in _SCALE_CACHE:
            m = CensoredSuperLearner(hetero=True).fit(Z, lo, hi)
            mu, _ = m.predict(Z)
            sp = _spread(Z)
            edges = np.quantile(sp, np.linspace(0, 1, 11))
            xs, ys = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                k = (sp >= a) & (sp <= b)
                xs.append(np.exp(np.median(sp[k])))
                ys.append(fit_sigma(mu[k], lo[k], hi[k]))
            _SCALE_CACHE[t]=(xs,ys,m.c.copy())
        xs,ys,coef=_SCALE_CACHE[t]
        ax.scatter(xs, ys, marker=TMARK[t], s=55, color=TCOL[t], edgecolor=INK, zorder=3, label=t)
        g = np.geomspace(min(xs), max(xs), 50)
        ax.plot(g, np.exp(coef[0] + coef[1] * np.log(g)), color=TCOL[t], lw=1.8)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.15, 0.3, 0.6, 1.2]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    style(ax, "Base-learner spread (log$_{10}$ units)", "Censored residual scale σ")
    ax.legend(loc="upper left", fontsize=12.0, frameon=False)

