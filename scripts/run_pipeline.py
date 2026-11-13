"""Print a serial reproduction plan; execute it only with the explicit --run flag."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    'prepare': [('s00_prepare_inputs.py', '--basemaps'), ('s01_zip_to_parquet.py', '4')],
    'profile': [('s02_profile_censoring.py',)],
    'datasets': [('s03_build_datasets.py',), ('s05_compliance_blindspot.py',)],
    'benchmark': [('s06_benchmark.py', 'real', 'Cd', 'Pb', 'Hg'),
                  ('s06_benchmark.py', 'synth', 'Cd', 'Pb')],
    'crossfit': [('s09_hidden_exceedance.py', 'Cd', 'Pb', 'Hg', 'Ni', 'PFOS',
                  'Glyphosate', 'CdDis', '--no-shap')],
    'sensitivity': [
        ('s09_hidden_exceedance.py', 'Cd', 'Pb', 'Hg', '--tag', '_noMX', '--exclude', 'Mexico', '--no-shap'),
        ('s09_hidden_exceedance.py', 'Cd', 'Pb', 'Hg', '--tag', '_noCountry', '--drop-country', '--no-shap'),
        ('s09_hidden_exceedance.py', 'Cd', 'Pb', 'Hg', '--tag', '_noYear', '--drop-features', 'year', '--no-shap'),
        ('s17_lodloq_targets.py', 'Cd', 'Pb', 'Hg'),
        ('s09_hidden_exceedance.py', 'CdLQ', 'PbLQ', 'HgLQ', '--no-shap'),
        ('s11_pseudo_blindspot.py', 'Cd', 'Pb', 'Hg', 'Ni', 'PFOS', 'Glyphosate', 'CdDis'),
        ('s11_pseudo_blindspot.py', 'Cd', 'Pb', 'Hg', '--tag', '_noMX'),
        ('s11_pseudo_blindspot.py', 'Cd', 'Pb', 'Hg', '--tag', '_noCountry'),
        ('s11_pseudo_blindspot.py', 'Cd', 'Pb', 'Hg', '--tag', '_noYear'),
        ('s11_pseudo_blindspot.py', 'CdLQ', 'PbLQ', 'HgLQ'),
        ('s07_censoring_ablation.py', 'Cd', 'Pb', '--seeds', '0', '1', '2'),
        ('s08_country_transfer.py', 'Cd', 'Pb', 'Hg'),
        ('s12_hardness_eqs.py', 'Cd', 'CdDis'),
        ('s13_trend_artifacts.py', 'Cd', 'Pb', 'Hg'),
        ('s13_trend_artifacts.py', 'Cd', 'Pb', 'Hg', '--tag', '_noYear'),
        ('s14_prevalence_ci.py',),
        ('s15_distribution_sensitivity.py', 'Cd', 'Pb', 'Hg'),
        ('s16_retest_priority.py',),
        ('s18_annual_compliance.py',),
        ('s18_annual_compliance.py', '--ci'),
    ],
    'summaries': [('s10_summarize.py',), ('assemble_figure_inputs.py',)],
    'figures': [('generate_six_main_figures.py',)],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', *STAGES], default='all')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    stages = STAGES if args.stage == 'all' else [args.stage]
    commands = [(stage, command) for stage in stages for command in STAGES[stage]]
    for i, (stage, command) in enumerate(commands, 1):
        print(f'{i:02d} [{stage}] python scripts/' + ' '.join(command), flush=True)
    if not args.run:
        print('Plan only. Add --run to execute; benchmark and crossfit stages train models.')
        return
    env = os.environ.copy()
    env.update({'PYTHONIOENCODING': 'utf-8', 'OMP_WAIT_POLICY': 'PASSIVE', 'KMP_BLOCKTIME': '0',
                'N_MAX': '50000', 'GFQA_FIGURE_ROOT': str(ROOT / 'figures')})
    for folder in ['results', 'tables', 'logs']:
        (ROOT / folder).mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for i, (stage, command) in enumerate(commands, 1):
        log = ROOT / 'logs' / f'{run_id}_{i:02d}_{stage}_{Path(command[0]).stem}.log'
        with log.open('w', encoding='utf-8') as stream:
            stream.write('UTC ' + datetime.now(timezone.utc).isoformat() + '\n')
            stream.write('Command: python scripts/' + ' '.join(command) + '\n'); stream.flush()
            result = subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / command[0]), *command[1:]],
                                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            stream.write(f'\nExit code: {result.returncode}\n')
        if result.returncode:
            raise SystemExit(f'Stopped after failure; inspect {log.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
