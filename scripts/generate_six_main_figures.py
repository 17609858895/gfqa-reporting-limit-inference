"""Six consolidated main figures, one per Results section, each composed in three or four rows.

The established panel builders are reused unchanged in content.  Five panel types are added from
completed analyses (no learner is refitted): the country-by-contaminant blind-spot matrix, a
transposed benchmark strip, station-year annual-mean screening, reporting-limit histories and
country/water-type breakdowns of follow-up priorities.
"""
from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

os.environ.setdefault('GFQA_FIGURE_QA_ROOT', str(Path(__file__).resolve().parents[1] / 'figures/qa'))

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

from figkit_compact import INK, MUTED, PAL, ROLE, ROOT, SEQ_BLUE, SEQ_LAV, SEQ_ROSE, TCOL, TMARK, build, style
import benchmark_panels as bench
import generate_main_figures_06_10 as g610
import learning_panels as lrn
import monitoring_panels as mon
import truth_panels as truth

DP, RES, TAB = ROOT / 'data_processed', ROOT / 'results', ROOT / 'tables'
METALS = ['Cd', 'Hg', 'Pb']
WIDTH = 13.2
standalone = g610.standalone
SHORT = g610.SHORT
NAMES = {'Netherlands (-the )': 'Netherlands', 'United States of America (the)': 'USA',
         'Macedonia (the former Yugoslav Republic of)': 'North Macedonia'}
short_country = lambda c: NAMES.get(c, SHORT(c))


# ------------------------------------------------------------------ Figure 1, new panel
BS_TARGETS = ['Cd', 'PFOS', 'Hg', 'Pb', 'Glyphosate', 'Ni']
MIN_COUNTRY_EVENTS = 500


def _country_matrix():
    st = pd.read_parquet(DP / 'stations_clean.parquet')[['station', 'country']]
    rows = []
    for t in BS_TARGETS:
        d = pd.read_parquet(DP / f'target_{t}.parquet', columns=['station', 'cens', 'y_hi']).merge(st, on='station', how='left')
        d['bs'] = (d['cens'] == 1) & (d['y_hi'] > np.log10(mon.THRESH[t]))
        g = d.groupby('country').agg(n=('bs', 'size'), bs=('bs', 'mean')).reset_index()
        g['t'] = t
        rows.append(g)
    R = pd.concat(rows)
    share = R.pivot(index='country', columns='t', values='bs').reindex(columns=BS_TARGETS)
    n = R.pivot(index='country', columns='t', values='n').reindex(columns=BS_TARGETS).fillna(0)
    keep = n['Cd'] >= MIN_COUNTRY_EVENTS
    share = share[keep].where(n[keep] >= MIN_COUNTRY_EVENTS)
    return share.sort_values('Cd', ascending=False) * 100, n[keep]


CM, CM_N = _country_matrix()


def country_blindspot(fig, ss):
    g = ss.subgridspec(2, 1, height_ratios=[1, .032], hspace=.075)
    ax = fig.add_subplot(g[0])
    cmap = SEQ_LAV.copy()
    cmap.set_bad('#EEF0F2')
    m = np.ma.masked_invalid(CM.to_numpy())
    ax.imshow(m, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    for (i, j), v in np.ndenumerate(CM.to_numpy()):
        if np.isfinite(v):
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=10.3, color=INK)
    ax.set_xticks(range(len(BS_TARGETS)), BS_TARGETS)
    ax.set_yticks(range(len(CM)), [short_country(c) for c in CM.index])
    style(ax)
    ax.xaxis.tick_top()
    ax.tick_params(axis='both', length=0, labelsize=11.2)
    ax.tick_params(axis='x', labelsize=12, top=True, labeltop=True, bottom=False, labelbottom=False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cax = fig.add_subplot(g[1])
    pos = cax.get_position()
    cax.set_position([pos.x0 + pos.width * .2, pos.y0, pos.width * .6, pos.height])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 100), cmap=SEQ_LAV), cax=cax,
                      orientation='horizontal', ticks=[0, 25, 50, 75, 100])
    cb.set_label('Blind-spot share (%)', fontsize=12.5)
    cb.ax.tick_params(labelsize=11, length=3)
    cb.outline.set_edgecolor(INK)


def layout1(fig):
    g = fig.add_gridspec(4, 2, height_ratios=[7.35, 4.15, 3.75, 2.05], left=.02, right=.98, bottom=.012, top=.985,
                         hspace=.072, wspace=.085)
    return dict(a=g[0, :], b=g[1, 0], c=g[1, 1], d=g[2, 0], e=g[2, 1], f=g[3, :])


# ------------------------------------------------------------------ Figure 2, new panel
def benchmark_strip(fig, ss):
    """Transposed benchmark matrix: models along the columns, ordered by mean NLL rank."""
    g = ss.subgridspec(3, 1, height_ratios=[.34, 3, 3], hspace=.05)
    strip = fig.add_subplot(g[0])
    strip.imshow(np.array([to_rgb(ROLE[bench.FAMILY[m]]) for m in bench.ORDER])[None, :, :], aspect='auto')
    strip.set_xticks([])
    strip.set_yticks([])
    for s_ in strip.spines.values():
        s_.set_visible(False)
    strip.legend(handles=bench.family_keys(), loc='lower right', bbox_to_anchor=(1, 1.25), ncol=3, fontsize=11.5,
                 frameon=False, handlelength=1.05, columnspacing=.85, borderaxespad=0)
    for k, (metric, cmap, lab, fmt, rev) in enumerate([('cindex', SEQ_BLUE, 'C-index', '{:.3f}', False),
                                                       ('cnll', SEQ_ROSE, 'NLL', '{:.2f}', True)]):
        ax = fig.add_subplot(g[k + 1])
        m = bench.B.pivot(index='model', columns='target', values=metric).loc[bench.ORDER, METALS].to_numpy().T
        lo, hi = np.nanmin(m, axis=1)[:, None], np.nanmax(m, axis=1)[:, None]
        show = (hi - m) / (hi - lo) if rev else (m - lo) / (hi - lo)
        ax.imshow(show, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=9.4, color=INK)
        ax.set_yticks(range(3), METALS)
        ax.set_xticks(range(len(bench.ORDER)), bench.ORDER if k == 1 else [], rotation=90)
        style(ax)
        ax.set_ylabel(lab, fontsize=13.5, labelpad=8)
        ax.tick_params(axis='both', length=0, labelsize=11.4)
        for s_ in ax.spines.values():
            s_.set_visible(False)


def layout2(fig):
    g = fig.add_gridspec(4, 12, height_ratios=[3.15, 3.7, 3.55, 4.15], left=.02, right=.98, bottom=.015, top=.985,
                         hspace=.085, wspace=.6)
    return dict(a=g[0, :], b=g[1, :], c=g[2, :6], d=g[2, 6:], e=g[3, :4], f=g[3, 4:8], g=g[3, 8:])


# ------------------------------------------------------------------ Figure 3
def align_bottom_row(left, right, key=-1):
    """Give the right-hand panels the baseline of the left-hand panel and centre the shared key on its legend."""
    def fn(fig, groups):
        base = groups[left][0].get_position()
        for k in right:
            ax = groups[k][0]
            p = ax.get_position()
            ax.set_position([p.x0, base.y0, p.width, p.y1 - base.y0])
        fig.canvas.draw()
        legend = groups[left][0].get_legend()
        bb = legend.get_window_extent().transformed(fig.transFigure.inverted())
        x0, x1 = groups[right[0]][0].get_position().x0, groups[right[-1]][0].get_position().x1
        fig.legends[key].set_bbox_to_anchor(((x0 + x1) / 2, (bb.y0 + bb.y1) / 2), transform=fig.transFigure)
    return fn


def layout3(fig):
    g = fig.add_gridspec(4, 20, height_ratios=[4.25, 4.45, 4.35, .26], left=.02, right=.98, bottom=.012, top=.985,
                         hspace=.085, wspace=.55)
    g610.shared_key(fig, g[3, 8:], g610.metal_keys(), 3)
    return dict(a=g[0, :11], b=g[0, 11:], c=g[1, :10], d=g[1, 10:], e=g[2, :8], f=g[2, 8:15], g=g[2, 15:])


# ------------------------------------------------------------------ Figure 4, new panel
AC = pd.read_csv(TAB / 'T_annual_compliance_ci.csv').merge(
    pd.read_csv(TAB / 'T_annual_compliance.csv')[['target', 'undetermined_share']], on='target')
AC_LABEL = {'Cd (dissolved), hardness': 'Dissolved Cd, hardness'}
AC = AC.sort_values('undetermined_share').reset_index(drop=True)


def annual_compliance(fig, ss):
    g = ss.subgridspec(1, 2, width_ratios=[3.0, 1.25], wspace=.3)
    ax = fig.add_subplot(g[0])
    side = fig.add_subplot(g[1], sharey=ax)
    y = np.arange(len(AC))
    ax.hlines(y, AC['lower'] * 100, AC['upper'] * 100, color='#D9DDE2', lw=8.5, zorder=1, label='Interval-compatible range')
    ax.scatter(AC['lower'] * 100, y, marker='|', s=135, color=ROLE['ndc'], lw=2.3, zorder=2, label='Lower bound')
    ax.scatter(AC['upper'] * 100, y, marker='|', s=135, color=ROLE['dl'], lw=2.3, zorder=2, label='Upper bound')
    for key, col, label, mk in [('half', ROLE['half'], 'Half-limit', 's'), ('expected', ROLE['model'], 'Censored model', 'o')]:
        err = np.vstack([AC[key] - AC[key + '_lo'], AC[key + '_hi'] - AC[key]]) * 100
        ax.errorbar(AC[key] * 100, y, xerr=err, fmt=mk, color=col, mec=INK, mew=.65, ms=7.3, ecolor=INK,
                    elinewidth=.9, zorder=3, label=label)
    ax.set_yticks(y, [AC_LABEL.get(t, t) for t in AC['target']])
    ax.set_xlim(0, 100)
    ax.set_ylim(-.6, len(AC) - .4)
    style(ax, 'Annual means above T (%)', None)
    v = AC['flip_of_half'] * 100
    side.barh(y, v, height=.56, color=ROLE['detected_ok'], ec=INK, lw=.65, zorder=2)
    side.errorbar(v, y, xerr=np.vstack([v - AC['flip_of_half_lo'] * 100, AC['flip_of_half_hi'] * 100 - v]),
                  fmt='none', ecolor=INK, elinewidth=.9, capsize=0, zorder=3)
    for i, value in enumerate(v):
        side.text(value + 4.5, i, f'{value:.0f}', va='center', fontsize=12, color=INK)
    side.set_xlim(0, 100)
    side.set_xticks([0, 50, 100])
    style(side, None, None)
    side.text(0, len(AC) - .45, 'Likely\nbelow (%)', fontsize=11.5, color=INK, va='bottom', ha='left')
    side.tick_params(axis='y', left=False, labelleft=False)
    side.spines['left'].set_visible(False)
    if standalone(ss):
        g610.below_legend(ax, pad=40, handles=g610.rule_keys(), ncol=2, fontsize=11.5, handlelength=1.4,
                          columnspacing=.8, handletextpad=.4)


def layout4(fig):
    g = fig.add_gridspec(5, 20, height_ratios=[5.1, .3, 4.55, 4.3, .26], left=.02, right=.98, bottom=.012, top=.985,
                         hspace=.085, wspace=.55)
    g610.shared_key(fig, g[1, :], g610.rule_keys(), 5)
    g610.shared_key(fig, g[4, 8:], g610.transfer_keys(), 4)
    return dict(a=g[0, :11], b=g[0, 11:], c=g[2, :8], d=g[2, 8:], e=g[3, :8], f=g[3, 8:14], g=g[3, 14:])


# ------------------------------------------------------------------ Figure 5, new panel
DL_CASES = [('Cd', 'France'), ('Cd', 'Italy'), ('Hg', 'France'), ('Hg', 'Italy'), ('Hg', 'Mexico'), ('Pb', 'Mexico')]


def _limit_history(target, country):
    d = pd.read_parquet(RES / f'crossfit_{target}.parquet', columns=['country', 'date', 'nd', 'y_hi'])
    d = d[d['country'].eq(country)].assign(year=lambda x: x['date'].dt.year)
    n = d.groupby('year').size()
    nd = d[d['nd']].assign(r=lambda x: 10 ** x['y_hi'] / g610.THRESH[target])
    q = nd.groupby('year')['r'].quantile([.25, .5, .75]).unstack()
    q = q[n.reindex(q.index).ge(150)]
    return q.reindex(range(q.index.min(), q.index.max() + 1))


DL_HISTORY = {case: _limit_history(*case) for case in DL_CASES}


def limit_histories(fig, ss):
    g = ss.subgridspec(2, len(DL_CASES), height_ratios=[1, .12], wspace=.11, hspace=.0)
    first = None
    for k, (t, c) in enumerate(DL_CASES):
        ax = fig.add_subplot(g[0, k], sharey=first) if first is not None else fig.add_subplot(g[0, k])
        first = first or ax
        h = DL_HISTORY[(t, c)]
        ax.axhspan(1, 2, color=ROLE['nd_undetermined'], alpha=.22, lw=0, zorder=0)
        ax.axhline(1, color=INK, ls='--', lw=1.0, zorder=1)
        ax.axhline(2, color=MUTED, ls=':', lw=1.2, zorder=1)
        ax.fill_between(h.index, h[.25], h[.75], color=TCOL[t], alpha=.28, lw=0, step=None, zorder=2)
        ax.plot(h.index, h[.5], color=TCOL[t], marker=TMARK[t], ms=5.2, mec=INK, mew=.55, lw=1.9, zorder=3)
        ax.set_yscale('log')
        ax.set_ylim(.05, 60)
        ax.yaxis.set_major_locator(FixedLocator([.1, 1, 10]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_major_locator(FixedLocator([2005, 2015] if h.index.min() < 2010 else [2014, 2020]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}'))
        style(ax, None, 'Non-detect DL / T' if k == 0 else None, fs=13.5)
        ax.tick_params(axis='x', labelsize=10.8)
        if k:
            ax.tick_params(axis='y', labelleft=False)
        ax.text(.96, .95, f'{t}, {c}', transform=ax.transAxes, ha='right', va='top', fontsize=12, color=INK)
        if k == len(DL_CASES) - 1:
            from matplotlib.transforms import blended_transform_factory
            edge = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(1.03, 1, 'T', transform=edge, va='center', fontsize=11.5, color=INK)
            ax.text(1.03, 2.2, '2T', transform=edge, va='center', fontsize=11.5, color=INK)
    foot = fig.add_subplot(g[1, :])
    foot.set_axis_off()
    foot.text(.5, -.2, 'Year', transform=foot.transAxes, ha='center', va='top', fontsize=13.5, color=INK)


def layout5(fig):
    g = fig.add_gridspec(3, 20, height_ratios=[4.95, 2.85, 4.45], left=.02, right=.98, bottom=.012, top=.985,
                         hspace=.09, wspace=.55)
    return dict(a=g[0, :10], b=g[0, 10:], c=g[1, :], d=g[2, :12], e=g[2, 12:])


# ------------------------------------------------------------------ Figure 6, new panels
def _priority_rows():
    rows = []
    for t in METALS:
        g = g610.COUNTRY_PRIORITY[g610.COUNTRY_PRIORITY.target.eq(t)].sort_values('blind_spot', ascending=False).head(5)
        rows += [(f'{t} {SHORT(r.country)}', r) for r in g.itertuples()]
    return rows


def country_volume(fig, ss):
    """Blind-spot volume and expected hidden exceedances on the rows of the country triage."""
    ax = fig.add_subplot(ss)
    rows = _priority_rows()
    y = np.arange(len(rows))
    for i, (_, r) in enumerate(rows):
        col = TCOL[r.target]
        ax.plot([r.expected_hidden_exceedances, r.blind_spot], [i, i], color=MUTED, lw=1.0, zorder=1)
        ax.scatter(r.blind_spot, i, marker=TMARK[r.target], s=46, facecolor='white', edgecolor=col, lw=1.3, zorder=3)
        ax.scatter(r.expected_hidden_exceedances, i, marker=TMARK[r.target], s=50, color=col, ec=INK, lw=.6, zorder=3)
        ax.text(4.2e5, i, f'{r.expected_hidden_exceedances / r.blind_spot * 100:.1f}', va='center', fontsize=11.7,
                color=INK)
    ax.text(4.2e5, -1.0, 'Hidden\n(%)', fontsize=10.7, color=INK, va='bottom')
    ax.set_xscale('log')
    ax.set_xlim(.3, 2.5e5)
    ax.xaxis.set_major_locator(FixedLocator([1, 1e2, 1e4]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_yticks(y, [label for label, _ in rows] if standalone(ss) else [])
    ax.set_ylim(len(rows) - .5, -.5)
    style(ax, 'Events (log scale)', None, fs=13.5)
    if not standalone(ss):
        ax.tick_params(axis='y', left=False)
    keys = [Line2D([], [], marker='o', ls='', mfc='white', mec=INK, mew=1.2, ms=7, label='Blind-spot events'),
            Line2D([], [], marker='o', ls='', mfc=MUTED, mec=INK, mew=.6, ms=7, label='Expected hidden exceedances')]
    g610.below_legend(ax, handles=keys, ncol=1 if not standalone(ss) else 2, fontsize=12.5, handlelength=1.0,
                      handletextpad=.35, columnspacing=.7)


def priority_capture(fig, ss):
    ax = fig.add_subplot(ss)
    for t in METALS:
        s = g610.STATION_PRIORITY[g610.STATION_PRIORITY.target.eq(t)]
        total = s.expected_hidden_exceedances.sum()
        for key, ls, lw in [('expected_hidden_exceedances', '-', 2.3), ('uncertain_events', ':', 1.6)]:
            o = s.sort_values(key, ascending=False)
            cum = np.cumsum(o.expected_hidden_exceedances.to_numpy()) / total
            fraction = np.arange(1, len(o) + 1) / len(o)
            ax.plot(fraction * 100, cum * 100, color=TCOL[t], lw=lw, ls=ls, label=t if ls == '-' else None)
            if ls == '-':
                index = np.searchsorted(fraction, .2)
                ax.scatter([20], [cum[index] * 100], marker=TMARK[t], s=58, color=TCOL[t], ec=INK, lw=.7, zorder=3)
    ax.plot([0, 100], [0, 100], color=MUTED, ls='--', lw=1)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    style(ax, 'Stations targeted, ranked (%)', 'Expected hidden exceedances\ncaptured (%)')
    handles = g610.metal_keys() + [Line2D([], [], color=INK, lw=2.0, label='Posterior sum'),
                                  Line2D([], [], color=INK, lw=1.5, ls=':', label='Uncertain-event count')]
    ax.legend(handles=handles, loc='lower right', fontsize=11.6, frameon=False, ncol=1, handlelength=1.6)


WT_ORDER = ['River', 'Lake', 'Reservoir', 'Wetland', 'Groundwater']
WT = (g610.STATION_PRIORITY[g610.STATION_PRIORITY.target.isin(METALS)]
      .groupby(['wtype', 'target']).agg(stations=('station', 'size'), uncertain=('uncertain_events', 'sum'),
                                         expected=('expected_hidden_exceedances', 'sum')).reset_index())
WT['yield'] = WT['expected'] / WT['uncertain']


def water_type_yield(fig, ss):
    ax = fig.add_subplot(ss)
    w = .25
    for j, t in enumerate(METALS):
        g = WT[WT.target.eq(t)].set_index('wtype').reindex(WT_ORDER)
        x = np.arange(len(WT_ORDER)) + (j - 1) * w
        ax.bar(x, g['yield'] * 100, width=w * .9, color=TCOL[t], ec=INK, lw=.65, label=t)
    ax.set_xticks(range(len(WT_ORDER)), WT_ORDER)
    ax.set_ylim(0, 24)
    style(ax, None, 'Expected hidden exceedances\nper uncertain event (%)')
    ax.tick_params(axis='x', labelsize=12)
    ax.legend(loc='upper right', ncol=3, fontsize=12, frameon=False, handlelength=1.0, columnspacing=.8)


def align_priority_rows(fig, groups):
    """Keep the volume rows on the same baseline as the triage rows."""
    left, right = groups['b'][0], groups['c'][0]
    a, b = left.get_position(), right.get_position()
    right.set_position([b.x0, a.y0, b.width, a.height])


def layout6(fig):
    g = fig.add_gridspec(3, 20, height_ratios=[4.55, 5.9, 4.35], left=.02, right=.98, bottom=.012, top=.985,
                         hspace=.085, wspace=.55)
    return dict(a=g[0, :], b=g[1, :13], c=g[1, 13:], d=g[2, :10], e=g[2, 10:])


# ------------------------------------------------------------------ specifications
SPEC = {
    1: dict(stem='Fig01_blind_spot', layout=layout1, size=(WIDTH, 17.75), exempt=('f',),
            panels=[('a', mon.panel_a), ('b', partial(mon.panel_b, legend_ncol=2)), ('c', country_blindspot),
                    ('d', mon.panel_d), ('e', partial(mon.panel_c, legend_ncol=2, count_y=.755)), ('f', mon.panel_e)],
            subs={'a': (13.2, 8.45), 'b': (7.0, 4.9), 'c': (6.6, 5.2), 'd': (6.3, 4.7), 'e': (6.5, 4.7), 'f': (13.2, 2.75)}),
    2: dict(stem='Fig02_censored_learning', layout=layout2, size=(WIDTH, 15.3), exempt=('a',),
            panels=[('a', lrn.panel_a), ('b', benchmark_strip), ('c', bench.panel_b), ('d', bench.panel_c),
                    ('e', lrn.panel_b), ('f', lrn.panel_c), ('g', lrn.panel_d)],
            subs={'a': (17, 4.5), 'b': (13.2, 4.0), 'c': (6.4, 4.2), 'd': (6.4, 4.3), 'e': (6, 4.8), 'f': (6.4, 5.4),
                  'g': (6, 4.8)}),
    3: dict(stem='Fig03_known_truth', layout=layout3, size=(WIDTH, 14.2), exempt=(), after=align_bottom_row('e', ['f', 'g']),
            panels=[('a', truth.panel_a), ('b', truth.panel_b), ('c', truth.panel_c), ('d', g610.reliability),
                    ('e', partial(g610.recensor_triage, ncol=3, fontsize=10.8)), ('f', g610.sensitivity('prev')),
                    ('g', g610.sensitivity('fc'))],
            subs={'a': (6.7, 4.45), 'b': (6.1, 4.35), 'c': (8.6, 4.15), 'd': (6.1, 4.7), 'e': (7.1, 4.7),
                  'f': (6.6, 4.6), 'g': (6.6, 4.6)}),
    4: dict(stem='Fig04_reporting_rules', layout=layout4, size=(WIDTH, 15.4), exempt=(),
            panels=[('a', g610.rule_prevalence), ('b', annual_compliance), ('c', g610.prevalence_ratio),
                    ('d', g610.prevalence_maps), ('e', g610.bound_frontier), ('f', g610.country_transfer('cindex')),
                    ('g', g610.country_transfer('cov_lower'))],
            subs={'a': (9.5, 6.3), 'b': (7.4, 4.6), 'c': (6.5, 5.0), 'd': (8.5, 4.3), 'e': (6.7, 5.0), 'f': (6.0, 4.9),
                  'g': (6.0, 4.9)}),
    5: dict(stem='Fig05_reporting_trends', layout=layout5, size=(WIDTH, 13.35), exempt=(),
            panels=[('a', g610.trend_association), ('b', g610.trend_forest), ('c', limit_histories),
                    ('d', g610.trend_cases), ('e', g610.year_removal)],
            subs={'a': (6.3, 4.9), 'b': (9.0, 5.2), 'c': (13.2, 3.1), 'd': (11.0, 4.4), 'e': (6.1, 4.9)}),
    6: dict(stem='Fig06_measurement_priorities', layout=layout6, size=(WIDTH, 15.6), exempt=(), after=align_priority_rows,
            panels=[('a', g610.followup_maps), ('b', g610.followup_triage), ('c', country_volume),
                    ('d', priority_capture), ('e', water_type_yield)],
            subs={'a': (9.2, 4.7), 'b': (7.5, 6.5), 'c': (7.6, 6.5), 'd': (6.2, 5.0), 'e': (6.4, 4.6)}),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*', type=int, default=list(SPEC))
    args = parser.parse_args()
    for number in args.only:
        spec = SPEC[number]
        build(spec['panels'], spec['layout'], spec['stem'], spec['stem'], figsize=spec['size'], sub_sizes=spec['subs'],
              exempt=spec['exempt'], after_fit=spec.get('after'))
        print(spec['stem'], 'done', flush=True)
