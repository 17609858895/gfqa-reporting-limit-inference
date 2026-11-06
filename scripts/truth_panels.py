"""Known-concentration validation panels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from figkit_compact import INK, MUTED, PAL, ROLE, ROOT, TCOL, TMARK, build, darker, style, outside_legend

RES, TAB = ROOT / "results", ROOT / "tables"
SYN = pd.concat([pd.read_csv(RES / f"bench_{t}_synth.csv") for t in ["Cd", "Pb"]])
ABL = pd.concat([pd.read_csv(RES / f"ablation_{t}.csv") for t in ["Cd", "Pb"]])
ABL = ABL[ABL["scenario"].str.startswith("level")]
def panel_a(fig, ss):
    # Preserve the failed RSF result instead of suppressing negative R-squared.
    grid=ss.subgridspec(1,2,width_ratios=[0.75,4],wspace=0.08)
    left=fig.add_subplot(grid[0]); ax=fig.add_subplot(grid[1],sharey=left)
    for aa in [left,ax]:
        aa.axhspan(-0.1,0.1,color=PAL[2],alpha=0.45,lw=0)
        aa.axhline(0,color=INK,lw=0.9)
    for t in ['Cd','Pb']:
        g=SYN[SYN['target']==t]
        for f in ['Substitution','Censored','Ensemble']:
            h=g[g['family']==f]
            for aa in [left,ax]:
                aa.scatter(h['r2_truth'],h['bias_hidden'],marker=TMARK[t],s=60,color=ROLE[f],edgecolor=INK,lw=0.7,
                           label=f if t=='Cd' else None,zorder=3)
        d=g[g['model']=='D-CSL'].iloc[0]
        ax.annotate(f'D-CSL ({t})',(d['r2_truth'],d['bias_hidden']),xytext=(0.64,-0.28) if t=='Cd' else (0.59,-0.18),textcoords='data',
                    fontsize=11.5,color=INK,ha='center',arrowprops=dict(arrowstyle='-',color=MUTED,lw=0.8))
    left.set_xlim(-0.75,-0.58); ax.set_xlim(0.49,0.74)
    left.set_ylim(-0.52,0.44)
    style(left,None,'Hidden-value bias (log$_{10}$)'); style(ax,'R$^2$ against truth',None)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y',left=False,labelleft=False)
    left.set_xticks([-0.67],['−0.67']); ax.set_xticks([0.5,0.6,0.7])
    left.annotate('RSF',(-0.666595,-0.45271),xytext=(1,12),textcoords='offset points',fontsize=11.5,color=INK,ha='center')
    kw=dict(color=INK,clip_on=False,lw=1)
    left.plot((0.96,1.04),(-0.015,0.015),transform=left.transAxes,**kw)
    ax.plot((-0.015,0.015),(-0.015,0.015),transform=ax.transAxes,**kw)
    from matplotlib.lines import Line2D
    handles,labels=ax.get_legend_handles_labels()
    handles += [Line2D([],[],marker=TMARK[t],ls='',markerfacecolor='white',markeredgecolor=INK,markersize=8,label=t) for t in ['Cd','Pb']]
    outside_legend(ax,above=True,pad_pt=4,handles=handles,ncol=3,fontsize=12.5,frameon=False,handletextpad=.25,columnspacing=.5)



def panel_b(fig, ss):
    ax = fig.add_subplot(ss)
    mm = {"Sub-DL/2": (ROLE["Substitution"], "-", "o"), "Sub-DL": (PAL[5], "-", "s"), "Drop-ND": (PAL[6], ":", "v"),
          "XGB-AFT-N": (ROLE["Censored"], "--", "D"), "CSL": (ROLE["Ensemble"], "--", "^")}
    for m, (col, ls, mk) in mm.items():
        h = ABL[ABL["model"] == m].assign(err=lambda x: (x["prev_pred"] - x["prev_true"]) * 100)
        a = h.groupby("scenario").agg(c=("cens_share", "mean"), e=("err", "mean"), s=("err", "std"), n=("err", "size")).sort_values("c")
        ax.plot(a["c"] * 100, a["e"], color=col, ls=ls, marker=mk, markeredgecolor=INK, markersize=7, lw=2.2, label=m)
        if (a["n"] > 1).any():
            ax.fill_between(a["c"] * 100, a["e"] - a["s"], a["e"] + a["s"], color=col, alpha=0.25, lw=0)
    ax.axhline(0, color=INK, lw=0.9)
    style(ax, "Censored share (%)", "Prevalence error (pp)")
    ax.legend(loc="upper left", fontsize=11.5, ncol=2, frameon=False, columnspacing=0.8, handlelength=1.6)


def panel_c(fig, ss):
    ax = fig.add_subplot(ss)
    keep = ["D-CSL", "CSL", "XGB-AFT-N", "CatBoost-AFT", "LGB-Tobit", "Mean-Top5", "Stack-L2", "LightGBM", "RandomForest",
            "XGBoost", "Ridge"]
    y = np.arange(len(keep))
    for k, (t, off) in enumerate([("Cd", -0.18), ("Pb", 0.18)]):
        g = SYN[SYN["target"] == t].set_index("model").loc[keep]
        err = (g["prev_pred"] - g["prev_true"]) * 100
        ax.barh(y + off, err, height=0.34, color=[ROLE[f] for f in g["family"]], edgecolor=INK, lw=0.7,
                hatch="" if t == "Cd" else "///")
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y, keep)
    ax.invert_yaxis()
    style(ax, "Prevalence error (pp)", None)
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(facecolor="white", edgecolor=INK, label="Cd"), mp.Patch(facecolor="white", edgecolor=INK, hatch="///", label="Pb")],
              loc="lower left", fontsize=11.5, frameon=False)


