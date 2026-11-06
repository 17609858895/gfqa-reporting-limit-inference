"""Assemble figure-only summaries from existing analytical outputs; no model fitting."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COUNTRY_TARGETS = ['Cd', 'Hg', 'Pb', 'Ni', 'Glyphosate', 'PFOS', 'CdDis']


def country_prevalence(root):
    pieces = []
    for target in COUNTRY_TARGETS:
        frame = pd.read_csv(root / 'results' / f'prevalence_rules_{target}.csv')
        for country, group in frame.groupby('country'):
            n = group['n'].sum()
            if n < 1000:
                continue
            pieces.append({'country': country, 'n': int(n),
                           **{key: np.average(group[key], weights=group['n'])
                              for key in ['half', 'model', 'und']}, 't': target})
    return pd.DataFrame(pieces, columns=['country', 'n', 'half', 'model', 'und', 't'])


def robustness_variants(root):
    hidden = pd.read_csv(root / 'results' / 'hidden_exceedance_summary.csv')
    hidden['variant'] = hidden['variant'].fillna('main')
    pseudo = pd.read_csv(root / 'tables' / 'T_pseudo_blindspot.csv')
    pseudo['variant'] = pseudo['variant'].fillna('main')
    dist = pd.read_csv(root / 'tables' / 'T_distribution_sensitivity.csv')
    rows = []
    def add(label, target, source_target=None, variant='main'):
        source_target = source_target or target
        h = hidden[hidden.target.eq(source_target) & hidden.variant.eq(variant)].iloc[0]
        p = pseudo[pseudo.target.eq(source_target) & pseudo.variant.eq(variant)].iloc[0]
        rows.append({'label': label, 't': target, 'prev': h.prev_model,
                     'fc': p['false_compliance_in_likely(<0.05)']})
    for target in ['Cd', 'Hg', 'Pb']:
        add('Main (LightGBM-Tobit, normal)', target)
        for distribution in ['normal', 'logistic', 'extreme']:
            d = dist[dist.target.eq(target) & dist.dist.eq(distribution)].iloc[0]
            rows.append({'label': f'XGB-AFT, {distribution} errors', 't': target,
                         'prev': d.prev_model, 'fc': d.false_compliance})
        add('No country feature', target, variant='_noCountry')
        add('No year feature', target, variant='_noYear')
        add('LOD/LOQ values re-labelled', target, source_target=target + 'LQ')
    return pd.DataFrame(rows, columns=['label', 't', 'prev', 'fc'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.project_root / 'tables'; output.mkdir(exist_ok=True)
    country_prevalence(args.project_root).to_csv(output / 'T_country_rule_prevalence.csv', index=False)
    robustness_variants(args.project_root).to_csv(output / 'T_robustness_variants.csv', index=False)


if __name__ == '__main__':
    main()
