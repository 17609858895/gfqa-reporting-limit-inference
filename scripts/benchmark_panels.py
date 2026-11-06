"""Censored benchmark evidence panels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from figkit_compact import INK, MUTED, PAL, ROLE, ROOT, SEQ_BLUE, SEQ_ROSE, TCOL, TMARK, build, darker, style

RES = ROOT / "results"
METALS = ["Cd", "Hg", "Pb"]
B = pd.concat([pd.read_csv(RES / f"bench_{t}_real.csv") for t in METALS])
B["rank"] = B.groupby("target")["cnll"].rank()
ORDER = B.groupby("model")["rank"].mean().sort_values().index.tolist()
FAMILY = B.drop_duplicates("model").set_index("model")["family"]
FAM_ORDER = ["Substitution", "Censored", "Ensemble"]
HIGHLIGHT = {"D-CSL": ROLE["Ensemble"], "CSL": PAL[4], "XGB-AFT-N": ROLE["Censored"],
             "XGBoost": ROLE["Substitution"], "Stack-L2": PAL[6]}


def panel_a(fig, ss):
    g = ss.subgridspec(1, 3, width_ratios=[0.07, 1, 1], wspace=0.06)
    strip = fig.add_subplot(g[0])
    cols = [ROLE[FAMILY[m]] for m in ORDER]
    strip.imshow(np.array([[(*__import__("matplotlib.colors", fromlist=["to_rgb"]).to_rgb(c),) for c in cols]]).transpose(1, 0, 2),
                 aspect="auto")
    strip.set_xticks([])
    strip.set_yticks(range(len(ORDER)), ORDER, fontsize=12.5)
    for s_ in strip.spines.values():
        s_.set_visible(False)
    strip.tick_params(axis="y", length=0, pad=4)
    for j, (metric, cmap, lab, fmt, rev) in enumerate([("cindex", SEQ_BLUE, "Interval C-index", "{:.3f}", False),
                                                       ("cnll", SEQ_ROSE, "Censored NLL", "{:.2f}", True)]):
        ax = fig.add_subplot(g[j + 1])
        m = B.pivot(index="model", columns="target", values=metric).loc[ORDER, METALS].to_numpy()
        # column-wise scaling: colour shows the rank-like position within each metal
        lo_, hi_ = np.nanmin(m, axis=0), np.nanmax(m, axis=0)
        show = (hi_ - m) / (hi_ - lo_) if rev else (m - lo_) / (hi_ - lo_)
        ax.imshow(show, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        for (i, k), v in np.ndenumerate(m):
            ax.text(k, i, fmt.format(v), ha="center", va="center", fontsize=11.0, color=INK)
        ax.set_xticks(range(3), METALS)
        ax.xaxis.tick_top()
        ax.set_yticks([])
        ax.set_xlabel(lab, fontsize=13.5)
        ax.xaxis.set_label_position("bottom")
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.tick_params(axis="x", length=0, labelsize=12)
    handles = [__import__("matplotlib.patches", fromlist=["Patch"]).Patch(facecolor=ROLE[f], edgecolor=INK, label=f) for f in FAM_ORDER]
    strip.legend(handles=handles, loc="upper left", bbox_to_anchor=(-6.2, -0.02), ncol=3, fontsize=12.0, frameon=False,
                 handlelength=1.1, columnspacing=0.8)


def panel_b(fig, ss):
    ax = fig.add_subplot(ss)
    w = 0.24
    for i, t in enumerate(METALS):
        for j, f in enumerate(FAM_ORDER):
            v = B[(B["target"] == t) & (B["family"] == f)]["cnll"].to_numpy()
            x = i + (j - 1) * w
            ax.boxplot([v], positions=[x], widths=w * 0.8, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=ROLE[f], edgecolor=INK, lw=0.9), medianprops=dict(color=INK, lw=1.4),
                       whiskerprops=dict(color=INK, lw=0.9), capprops=dict(color=INK, lw=0.9))
            ax.scatter(np.full(len(v), x) + np.random.default_rng(j).uniform(-0.05, 0.05, len(v)), v, s=18, color=ROLE[f], edgecolor=INK, linewidth=0.35,
                       alpha=0.6, zorder=3)
    ax.set_xticks(range(3), METALS)
    style(ax, None, "Censored NLL (test)")


def panel_c(fig, ss):
    ax = fig.add_subplot(ss)
    w = 0.24
    for i, t in enumerate(METALS):
        for j, f in enumerate(FAM_ORDER):
            g = B[(B["target"] == t) & (B["family"] == f)]
            v = g["exc_p_exceed_undetermined"].to_numpy()
            x = i + (j - 1) * w
            ax.bar(x, np.median(v), width=w * 0.85, color=ROLE[f], edgecolor=INK, lw=0.9, label=f if i == 0 else None)
            ax.scatter(np.full(len(v), x) + np.random.default_rng(j).uniform(-0.05, 0.05, len(v)), v, s=18, color=ROLE[f], edgecolor=INK, linewidth=0.35,
                       alpha=0.6, zorder=3)
            if f == "Ensemble":
                sv = g.loc[g["model"] == "Stack-L2", "exc_p_exceed_undetermined"].iloc[0]
                ax.annotate("Stack-L2", (x, sv), xytext=(x + 0.13, sv + 0.06), fontsize=11.0, color=INK,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xticks(range(3), METALS)
    ax.set_ylim(0, 1)
    style(ax, None, "Mean P(exceed), blind spot")
    ax.legend(loc="upper right", fontsize=11.5, frameon=False, ncol=1, handlelength=1.1)


def panel_d(fig, ss):
    ax = fig.add_subplot(ss)
    R = B.pivot(index="model", columns="target", values="rank")[METALS]
    for m in R.index:
        col = HIGHLIGHT.get(m, PAL[2])
        lw, z = (2.4, 3) if m in HIGHLIGHT else (1.0, 1)
        ax.plot(range(3), R.loc[m], color=col, lw=lw, zorder=z, marker="o", markersize=5 if m in HIGHLIGHT else 3,
                markeredgecolor=INK if m in HIGHLIGHT else col, markeredgewidth=0.6)
    # repel end labels so they never overlap
    ends = sorted(((R.loc[m, "Pb"], m) for m in HIGHLIGHT), key=lambda t: t[0])
    pos = []
    for y, m in ends:
        y_lab = max(y, pos[-1] + 1.9) if pos else y
        pos.append(y_lab)
        ax.text(2.12, y_lab, m, va="center", fontsize=12.0, color=INK)
        ax.plot([2.0, 2.1], [y, y_lab], color=MUTED, lw=0.7)
    ax.set_ylim(27, 0)
    ax.set_xlim(-0.15, 2.6)
    ax.set_xticks(range(3), METALS)
    ax.set_yticks([1, 5, 10, 15, 20, 25])
    style(ax, None, "Rank (censored NLL)")

