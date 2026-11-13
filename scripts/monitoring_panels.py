"""Observation coverage, reporting resolution, and interval concepts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from figkit_compact import (FS, INK, MUTED, PAL, ROLE, ROOT, SEQ_LAV, TCOL, TMARK, basemap, build, map_proj, style,
                     zoom_box, outside_legend)

DP, TAB = ROOT / "data_processed", ROOT / "tables"
THRESH = {"Cd": 0.08, "Hg": 0.07, "PFOS": 0.00065, "Pb": 1.2, "Ni": 4.0, "Glyphosate": 0.1,
          "NO2N": 0.03, "Atrazine": 0.6, "PFOA": 0.1, "NH4N": 0.2, "As": 10.0, "TP": 0.1}
LABEL = {"NO2N": "NO$_2$-N", "NH4N": "NH$_4$-N"}

# ---------------------------------------------------------------- data
st = pd.read_parquet(DP / "stations_clean.parquet")[["station", "country", "lat", "lon"]]
parts, dlr = [], []
for t in ["Cd", "Hg", "Pb"]:
    d = pd.read_parquet(DP / f"target_{t}.parquet", columns=["station", "cens", "y_hi"])
    d["und"] = (d["cens"] == 1) & (d["y_hi"] > np.log10(THRESH[t]))
    parts.append(d[["station", "und"]])
smap = pd.concat(parts).groupby("station").agg(und=("und", "mean"), n=("und", "size")).reset_index().merge(st, on="station")
smap = smap.dropna(subset=["lat", "lon"])
smap = smap[smap["n"] >= 5].sort_values("und")
smap["x"], smap["y"] = map_proj(smap["lon"], smap["lat"])

for t in ["Cd", "Hg", "Pb", "PFOS", "Ni", "Glyphosate"]:
    d = pd.read_parquet(DP / f"target_{t}.parquet", columns=["cens", "y_hi"])
    r = 10 ** d.loc[d["cens"] == 1, "y_hi"].to_numpy() / THRESH[t]
    dlr.append(pd.DataFrame({"target": t, "ratio": r}))
dlr = pd.concat(dlr)

cats = []
for t, thr in THRESH.items():
    d = pd.read_parquet(DP / f"target_{t}.parquet", columns=["cens", "y_lo", "y_hi"])
    nd, l = d["cens"] == 1, np.log10(thr)
    cats.append({"target": t, "a": ((~nd) & (d["y_lo"] <= l)).mean(), "b": ((~nd) & (d["y_lo"] > l)).mean(),
                 "c": (nd & (d["y_hi"] <= l)).mean(), "d": (nd & (d["y_hi"] > l)).mean()})
cats = pd.DataFrame(cats).set_index("target").sort_values("d")

prof = pd.read_csv(TAB / "T_param_profile.csv")
meta = pd.read_parquet(DP / "GEMStat_parameter_metadata.parquet")[["Parameter Code", "Parameter Group"]]
prof = prof.merge(meta, left_on="param", right_on="Parameter Code", how="left")
grp = prof["Parameter Group"].fillna("")
prof["class"] = np.select([grp.str.contains("Metal"), grp.str.contains("Pesticide|POP|PAH|Organic|Pharma|Halocarbon|Phenyl|Volatile"),
                           grp.str.contains("Nutrient")], ["Metals", "Organic micropollutants", "Nutrients"], "Other")
prof = prof[prof["rows"] >= 1000]


# ---------------------------------------------------------------- panels
def size_of(n):
    return 6 + 10 * np.log10(n)


def _pts(ax, scale=1.0):
    ax.scatter(smap["x"], smap["y"], c=smap["und"] * 100, cmap=SEQ_LAV, vmin=0, vmax=100, s=size_of(smap["n"]) * scale,
               edgecolors=INK, linewidths=0.23, zorder=3)


def panel_a(fig, ss):
    g = ss.subgridspec(2, 1, height_ratios=[2.10, 1], hspace=0.12)
    ax = fig.add_subplot(g[0])
    basemap(ax)
    _pts(ax)
    zooms = [("Europe", (-11, 33, 36, 70)), ("Mexico", (-117, -87, 14.5, 32.5)), ("Southern Canada", (-125, -62, 43, 58))]
    zg = g[1].subgridspec(1, 3, wspace=0.045)
    for k, (name, ext) in enumerate(zooms):
        z = fig.add_subplot(zg[k])
        basemap(z, extent=ext)
        _pts(z)
        zoom_box(ax, z, ext, name)
    cax = ax.inset_axes([0.046, 0.105, 0.21, 0.030])
    import matplotlib as mpl
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 100), cmap=SEQ_LAV), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 25, 50, 75, 100])
    cb.set_label("Blind-spot share (%)", fontsize=12.5)
    cb.ax.tick_params(labelsize=11, length=3)
    cb.outline.set_edgecolor(INK)
    ax.text(0.046, 0.36, "Events per station", transform=ax.transAxes, fontsize=12.5, color=INK)
    for n, xx in zip([10, 100, 1000], [0.064, 0.15, 0.238]):
        ax.scatter([xx], [0.286], s=size_of(n), transform=ax.transAxes, facecolors="white", edgecolors=INK, lw=0.7)
        ax.text(xx, 0.223, f"{n:,}", transform=ax.transAxes, ha="center", va="center", fontsize=11.5, color=INK)
    ax.text(0.98, 0.03, f"{len(smap):,} stations, {smap['country'].nunique()} countries", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=13.0, color=INK)


def panel_b(fig, ss, legend_ncol=None):
    ax = fig.add_subplot(ss)
    order = [("a", "Detected ≤ threshold", ROLE["detected_ok"]), ("b", "Detected > threshold", ROLE["detected_exceed"]),
             ("c", "Non-detect, DL ≤ threshold", ROLE["nd_determinable"]), ("d", "Non-detect, DL > threshold", ROLE["nd_undetermined"])]
    y = np.arange(len(cats))
    left = np.zeros(len(cats))
    for k, lab, col in order:
        v = cats[k].to_numpy() * 100
        ax.barh(y, v, left=left, color=col, edgecolor=INK, linewidth=0.6, height=0.74, label=lab)
        left += v
    for i, v in enumerate(cats["d"].to_numpy() * 100):
        if v >= 3:
            ax.text(100 - v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=12.5, color=INK)
    ax.set_yticks(y, [LABEL.get(t, t) for t in cats.index])
    ax.set_xlim(0, 100)
    style(ax, "Share of events (%)")
    single_panel = ss.get_gridspec().get_geometry() == (1, 1)
    outside_legend(ax, pad_pt=39, ncol=legend_ncol or (2 if single_panel else 4), fontsize=11.7,
                   columnspacing=.95, handlelength=1.05, frameon=False)


def panel_c(fig, ss, legend_ncol=None, count_y=None):
    ax = fig.add_subplot(ss)
    cols = {"Metals": PAL[0], "Organic micropollutants": PAL[1], "Nutrients": PAL[2], "Other": PAL[6]}
    for k in ["Other", "Nutrients", "Metals", "Organic micropollutants"]:
        g = prof[prof["class"] == k]
        ax.scatter(g["lt_share"] * 100, g["n_distinct_dl"].clip(lower=1), s=np.sqrt(g["rows"]) / 7, facecolor=cols[k],
                   edgecolor=INK, linewidth=0.45, alpha=0.9, label=k)
    ax.set_yscale("log")
    ax.set_xlim(-3, 103)
    style(ax, "Non-detect share (%)", "Distinct reporting limits")
    ax.tick_params(axis="y", which="minor", length=0)
    standalone = ss.get_gridspec().get_geometry() == (1, 1)
    ax.legend(loc="upper right", fontsize=11.5, handletextpad=0.2, markerscale=0.8, borderaxespad=0.2,
              ncol=legend_ncol or (1 if standalone else 4), columnspacing=0.8, frameon=False)
    ax.text(0.98, count_y or (0.70 if standalone else 0.83), f"{len(prof)} parameters", transform=ax.transAxes, ha="right",
            fontsize=12.0, color=INK)


def panel_d(fig, ss):
    ax = fig.add_subplot(ss)
    order = ["Cd", "PFOS", "Hg", "Pb", "Glyphosate", "Ni"]
    data = [np.log10(dlr.loc[dlr["target"] == t, "ratio"].clip(10 ** -2.4, 10 ** 3.4)) for t in order]
    parts = ax.violinplot(data, positions=range(len(order)), widths=0.8, showextrema=False)
    for b, t in zip(parts["bodies"], order):
        b.set_facecolor(TCOL[t])
        b.set_edgecolor(INK)
        b.set_linewidth(0.7)
        b.set_alpha(0.95)
    for i, v in enumerate(data):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color=INK, lw=2.4)
        ax.scatter([i], [med], s=26, color="white", edgecolor=INK, zorder=3)
        share = np.mean(v > np.log10(0.3)) * 100
        ax.text(i, 1.01, f"{share:.0f}%", ha="center", va="bottom", fontsize=12.0, color=INK,
                transform=ax.get_xaxis_transform())
    ax.axhline(0, color=INK, ls="--", lw=1.1)
    ax.axhline(np.log10(0.3), color=MUTED, ls=":", lw=1.3)
    ax.set_xticks(range(len(order)), order)
    ax.set_yticks([-2, -1, 0, 1, 2, 3], ["0.01", "0.1", "1", "10", "100", "1000"])
    ax.set_ylim(-2.6, 3.6)
    style(ax, None, "DL / threshold (non-detects)")


def panel_e(fig, ss):
    """Concept strip: a log-concentration axis, four event types and four reporting rules."""
    ax = fig.add_subplot(ss)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    # concentration axis
    ax.annotate("", xy=(44, 4), xytext=(2, 4), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(23, 1.0, "log concentration", ha="center", fontsize=13.0, color=INK)
    ax.plot([28, 28], [3, 27], color=PAL[0], lw=2.2)
    ax.text(28, 28.2, "Threshold T", ha="center", fontsize=12.5, color=INK)
    rows = [(23, "Detected", 18, None, ROLE["detected_ok"], "below threshold"),
            (18, "Detected", 36, None, ROLE["detected_exceed"], "above threshold"),
            (13, "Non-detect", None, 20, ROLE["nd_determinable"], "below threshold"),
            (8, "Non-detect", None, 38, ROLE["nd_undetermined"], "blind spot")]
    for y, lab, x, dl, col, out in rows:
        ax.text(1, y, lab, va="center", fontsize=13.0, color=INK)
        if x is not None:
            ax.scatter([x], [y], s=110, color=col, edgecolor=INK, zorder=3)
        else:
            ax.add_patch(FancyBboxPatch((12, y - 1.1), dl - 12, 2.2, boxstyle="round,pad=0,rounding_size=0.8",
                                        facecolor=col, edgecolor=INK, lw=0.8))
            ax.plot([dl, dl], [y - 1.8, y + 1.8], color=INK, lw=1.6)
            ax.text(dl + 0.8, y + 1.6, "DL", fontsize=11.5, color=INK)
        ax.text(50, y, out, ha="right", va="center", fontsize=13.0, color=INK, style="italic" if out == "blind spot" else "normal")
    # rules table for the blind-spot event
    x0 = 53
    ax.text(x0, 26.9, "Blind-spot reporting rules", fontsize=12.5, color=INK)
    rules = [("Assume below threshold", "below threshold", ROLE["ndc"]),
             ("Half-limit substitution", "DL/2 > T?", ROLE["half"]),
             ("Full-limit substitution", "above threshold", ROLE["dl"]),
             ("Censored model", "P(Y > T | Y < DL, x)", ROLE["model"])]
    for k, (name, res, col) in enumerate(rules):
        yy = 21.5 - k * 5
        ax.scatter(x0 + 0.7, yy, s=68, marker="s", color=col, edgecolor=INK, linewidth=0.6)
        ax.text(x0 + 2.7, yy, name, va="center", fontsize=12.5, color=INK)
        ax.text(98.1, yy, res, va="center", ha="right", fontsize=12.5, color=INK)

