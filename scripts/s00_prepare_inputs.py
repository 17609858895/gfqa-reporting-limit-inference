"""Verify the independently obtained release and prepare its three metadata tables.

The release is read directly from ZIP. Measurement conversion is performed by s01.
Optional Natural Earth downloads are auxiliary basemaps, not monitoring observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOI = '10.5281/zenodo.18459694'
ZIP_NAME = 'GFQA_v3.zip'
ZIP_BYTES = 201278791
ZIP_MD5 = '00f3ea19ce529753977eb3aeb08fbc47'
ZIP_SHA256 = '82b169dcbf5abf30066020b176b82474b23a2dec0a108a351045f31a6476e92a'
METADATA = ['GEMStat_methods_metadata.csv', 'GEMStat_parameter_metadata.csv',
            'GEMStat_station_metadata.csv']
BASEMAP_VERSION = 'v5.1.2'
BASEMAP_BASE = ('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/'
                + BASEMAP_VERSION + '/geojson/')
BASEMAP_FILES = ['ne_50m_land.geojson', 'ne_110m_admin_0_countries.geojson']


def verify_release(path):
    if path.stat().st_size != ZIP_BYTES:
        raise ValueError('The ZIP size does not match the fixed GFQA v3 release.')
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            md5.update(chunk); sha.update(chunk)
    if md5.hexdigest() != ZIP_MD5 or sha.hexdigest() != ZIP_SHA256:
        raise ValueError('The ZIP checksum does not match the fixed GFQA v3 release.')
    return {'doi': DOI, 'filename': ZIP_NAME, 'bytes': ZIP_BYTES,
            'md5': md5.hexdigest(), 'sha256': sha.hexdigest(), 'data_license': 'CC BY 4.0'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basemaps', action='store_true', help='Also download pinned Natural Earth GeoJSONs.')
    args = parser.parse_args()
    for folder in ['inputs', 'data_processed', 'results', 'tables', 'logs']:
        (ROOT / folder).mkdir(exist_ok=True)
    path = ROOT / 'inputs' / ZIP_NAME
    if not path.exists():
        raise FileNotFoundError(f'Download {ZIP_NAME} from https://doi.org/{DOI} into inputs/ first.')
    provenance = verify_release(path)
    import pandas as pd
    with zipfile.ZipFile(path) as archive:
        for name in METADATA:
            with archive.open(name) as stream:
                frame = pd.read_csv(stream, encoding='latin-1', dtype=str)
            frame.to_parquet(ROOT / 'data_processed' / name.replace('.csv', '.parquet'), index=False)
            print(name, len(frame), flush=True)
    if args.basemaps:
        folder = ROOT / 'inputs' / 'basemap'; folder.mkdir(exist_ok=True)
        provenance['basemaps'] = []
        for name in BASEMAP_FILES:
            url = BASEMAP_BASE + name
            with urlopen(url, timeout=120) as response:
                data = response.read()
            json.loads(data)  # fail before writing if the response is not JSON
            (folder / name).write_bytes(data)
            provenance['basemaps'].append({'file': name, 'url': url,
                                          'sha256': hashlib.sha256(data).hexdigest()})
    (ROOT / 'inputs' / 'input_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print('Release verified; metadata prepared. Run s01 to convert measurement CSVs.')


if __name__ == '__main__':
    main()
