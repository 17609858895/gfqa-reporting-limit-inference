"""Censored benchmark evidence panels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figkit_compact import INK, MUTED, ROLE, ROOT, SEQ_BLUE, SEQ_ROSE, outside_legend, style

RES = ROOT / "results"
METALS = ["Cd", "Hg", "Pb"]
B = pd.concat([pd.read_csv(RES / f"bench_{t}_real.csv") for t in METALS])
B["rank"] = B.groupby("target")["cnll"].rank()
ORDER = B.groupby("model")["rank"].mean().sort_values().index.tolist()
FAMILY = B.drop_duplicates("model").set_index("model")["family"]
FAM_ORDER = ["Substitution", "Censored", "Ensemble"]
LABELLED_MODELS = {"D-CSL": "-", "CSL": "--", "XGB-AFT-N": "-",
                   "XGBoost": "-", "Stack-L2": ":"}


def family_keys():
    return [Patch(facecolor=ROLE[family], edgecolor=INK, label=family) for family in FAM_ORDER]


def panel_a(fig, ss):
    g = ss.subgridspec(2, 3, width_ratios=[0.067, 1, 1], height_ratios=[24, 1],
                      wspace=0.055, hspace=0.075)
    strip = fig.add_subplot(g[0, 0])
    cols = [ROLE[FAMILY[m]] for m in ORDER]
    strip.imshow(np.array([to_rgb(c) for c in cols])[:, None, :], aspect="auto")
    strip.set_xticks([])
    strip.set_yticks(range(len(ORDER)), ORDER, fontsize=11.7)
    for s_ in strip.spines.values():
        s_.set_visible(False)
    strip.tick_params(axis="y", length=0, pad=4)
    for j, (metric, cmap, lab, fmt, rev) in enumerate([("cindex", SEQ_BLUE, "Interval C-index", "{:.3f}", False),
                                                       ("cnll", SEQ_ROSE, "Censored NLL", "{:.2f}", True)]):
        ax = fig.add_subplot(g[0, j + 1])
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
    key_axis = fig.add_subplot(g[1, :])
    key_axis.axis("off")
    key_axis.legend(handles=family_keys(), loc="center", ncol=3, fontsize=11.5,
                    frameon=False, handlelength=1.05, columnspacing=0.85)


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
    if ss.get_gridspec().get_geometry() == (1, 1):
        outside_legend(ax, pad_pt=27, handles=family_keys(), ncol=3, fontsize=11.5,
                       frameon=False, handlelength=1.05, columnspacing=.85)


def panel_c(fig, ss):
    ax = fig.add_subplot(ss)
    w = 0.24
    for i, t in enumerate(METALS):
        for j, f in enumerate(FAM_ORDER):
            g = B[(B["target"] == t) & (B["family"] == f)]
            v = g["exc_p_exceed_undetermined"].to_numpy()
            x = i + (j - 1) * w
            ax.bar(x, np.median(v), width=w * 0.85, color=ROLE[f], edgecolor=INK, lw=0.8)
            xpos = np.full(len(v), x) + np.random.default_rng(j).uniform(-0.05, 0.05, len(v))
            is_stack = g["model"].eq("Stack-L2").to_numpy()
            ax.scatter(xpos[~is_stack], v[~is_stack], s=18, color=ROLE[f], edgecolor=INK,
                       linewidth=0.35, alpha=0.6, zorder=3)
            if is_stack.any():
                ax.scatter(xpos[is_stack], v[is_stack], s=34, marker="D", color=ROLE[f],
                           edgecolor=INK, linewidth=0.7, zorder=4)
    ax.set_xticks(range(3), METALS)
    ax.set_ylim(0, 1)
    style(ax, None, "Mean P(exceed), blind spot")
    stack_key = Line2D([], [], marker="D", ls="", markerfacecolor=ROLE["Ensemble"],
                       markeredgecolor=INK, markersize=5.5, label="Stack-L2")
    if ss.get_gridspec().get_geometry() == (1, 1):
        outside_legend(ax, pad_pt=27, handles=family_keys() + [stack_key], ncol=4,
                       fontsize=10.8, frameon=False, handlelength=1.0, columnspacing=.65)
    else:
        ax.legend(handles=[stack_key], loc="upper right", fontsize=11.5, frameon=False,
                  handlelength=1.0, borderaxespad=.25)


def panel_d(fig, ss):
    ax = fig.add_subplot(ss)
    R = B.pivot(index="model", columns="target", values="rank")[METALS]
    for m in R.index:
        col = ROLE[FAMILY[m]]
        labelled = m in LABELLED_MODELS
        ax.plot(range(3), R.loc[m], color=col, lw=1.9 if labelled else 0.8,
                alpha=1 if labelled else .43, zorder=3 if labelled else 1,
                ls=LABELLED_MODELS.get(m, "-"), marker="o", markersize=4.7 if labelled else 2.5,
                markeredgecolor=INK if labelled else col, markeredgewidth=0.55)
    # repel end labels so they never overlap
    ends = sorted(((R.loc[m, "Pb"], m) for m in LABELLED_MODELS), key=lambda t: t[0])
    pos = []
    for y, m in ends:
        y_lab = max(y, pos[-1] + 2.05) if pos else y
        pos.append(y_lab)
        ax.text(2.12, y_lab, m, va="center", fontsize=11.4, color=INK)
        ax.plot([2.0, 2.1], [y, y_lab], color=MUTED, lw=0.7)
    ax.set_ylim(27, 0)
    ax.set_xlim(-0.15, 2.62)
    ax.set_xticks(range(3), METALS)
    ax.set_yticks([1, 5, 10, 15, 20, 25])
    style(ax, None, "Rank (censored NLL)")
    if ss.get_gridspec().get_geometry() == (1, 1):
        outside_legend(ax, pad_pt=27, handles=family_keys(), ncol=3, fontsize=11.5,
                       frameon=False, handlelength=1.05, columnspacing=.85)

