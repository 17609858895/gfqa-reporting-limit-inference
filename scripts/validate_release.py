"""Validate the code-only release without importing analysis or plotting modules."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'CODE_MANIFEST.json'
THIRD_PARTY = {'numpy', 'pandas', 'pyarrow', 'scipy', 'sklearn', 'xgboost', 'lightgbm',
               'catboost', 'sksurv', 'torch', 'shap', 'matplotlib', 'geopandas',
               'pyproj', 'shapely', 'PIL', 'joblib', 'threadpoolctl'}
ALLOWED_SUFFIXES = {'.py', '.md', '.txt', '.json'}
IGNORED = {'.git', '__pycache__', '.venv'}
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|'
                    r'AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
PERSONAL_PATH = re.compile(r'(?i)[A-Z]:[/\\](?:Users|OneDrive|Anaconda)[/\\]')
EMAIL = re.compile(r'[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')


def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file()
                  and not any(part in IGNORED for part in p.relative_to(ROOT).parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    inventory = files()
    local = {p.stem for p in (ROOT / 'scripts').glob('*.py')}
    problems, imports = [], set()
    for path in inventory:
        rel = path.relative_to(ROOT).as_posix()
        if path.name == '.gitignore':
            pass
        elif rel == 'assets/method_background.png':
            pass
        elif path.suffix not in ALLOWED_SUFFIXES:
            problems.append(f'Unexpected file type: {rel}')
        if path.suffix in ALLOWED_SUFFIXES or path.name == '.gitignore':
            text = path.read_text(encoding='utf-8-sig')
            for pattern, label in [(SECRET, 'credential-like content'),
                                   (PERSONAL_PATH, 'personal absolute path'), (EMAIL, 'email address')]:
                if pattern.search(text):
                    problems.append(f'{label}: {rel}')
            if path.suffix == '.py':
                tree = ast.parse(text, filename=rel)
                compile(tree, rel, 'exec')  # in-memory compilation only
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.update(alias.name.split('.')[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.add(node.module.split('.')[0])
    missing = sorted(imports - set(sys.stdlib_module_names) - local - THIRD_PARTY)
    problems.extend(f'Unresolved static import: {name}' for name in missing)
    records = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in inventory if p != MANIFEST]
    if args.write_manifest:
        MANIFEST.write_text(json.dumps({'algorithm': 'SHA-256', 'files': records}, indent=2) + '\n', encoding='utf-8')
    elif not MANIFEST.exists():
        problems.append('CODE_MANIFEST.json is missing')
    elif json.loads(MANIFEST.read_text(encoding='utf-8'))['files'] != records:
        problems.append('File inventory or hashes differ from CODE_MANIFEST.json')
    result = {'pass': not problems, 'files': len(records) + int(MANIFEST.exists()),
              'python_files': sum(p.suffix == '.py' for p in inventory),
              'original_artwork_assets': sum(p.suffix == '.png' for p in inventory),
              'total_bytes': sum(p.stat().st_size for p in files()),
              'unresolved_imports': missing, 'problems': problems}
    print(json.dumps(result, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
