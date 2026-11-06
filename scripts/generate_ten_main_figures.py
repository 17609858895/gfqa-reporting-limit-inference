"""Regenerate ten figures from analytical outputs without base-learner training."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
for name in ['fig01_monitoring_landscape.py','fig02_reporting_resolution.py',
             'fig03_learning_framework.py','fig04_censored_benchmark.py',
             'fig05_known_truth.py','generate_main_figures_06_10.py']:
    subprocess.run([sys.executable,'-W','ignore',str(HERE/name)],cwd=HERE,check=True)
