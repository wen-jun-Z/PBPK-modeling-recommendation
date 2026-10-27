"""Acquire pinned public pregnancy assets without changing the frozen experiment.

No clinical observations are opened and no native solver is invoked. Existing
catalog projects are referenced by exact bytes rather than duplicated. Public
MoBi projects and pregnancy building blocks are ancillary source material; they
never replace a catalog candidate or justify rebuilding it as a four-compartment
model. A complete native simulation still needs lossless export and native audit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'artifacts/pbpk-model-assets/execution_repairs_20261001/pregnancy_sources'
OWNER = 'Open-Systems-Pharmacology'
PREGNANCY_COMMIT = '9fbb41b59b52497faffefbbec0c50de45103d84f'
LICENSE_COMMIT = '41f2b7d1951ca8770220a301dec8a876840bcb1d'
CATALOG = ROOT / 'experiments/model_recommendation/frontier_core_145_20260921/inputs/candidates.jsonl'
CANDIDATES = {
    '0c8ef880-cbe1-548e-8c23-2e9258eeee8f': ('caffeine', 'Pregnancy-Models'),
    '833b07ad-a4c8-5a4f-8abc-97a685ae883d': ('olanzapine', 'Olanzapine-Model'),
    'f29f55d1-c73b-5bcf-8324-eddcc66ec778': ('quetiapine', 'Quetiapine-Model'),
}
MAX_FILE_BYTES = 128 * 1024**2


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def inside(path, root=ROOT):
    value = Path(path).resolve()
    if not value.is_relative_to(root.resolve()) or value == root.resolve():
        raise ValueError('Asset paths must remain inside this project')
    if any('sealed' in part.casefold() for part in value.parts):
        raise ValueError('Sealed clinical assets are forbidden')
    return value


def write_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'
    with path.open('x', encoding='utf8') as handle:
        handle.write(payload)


def fetch_asset(spec, output):
    repo, commit, source_path, role = spec
    destination = inside(output / repo / source_path)
    url = f'https://raw.githubusercontent.com/{OWNER}/{repo}/{commit}/{urllib.parse.quote(source_path)}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError('Never overwrite acquired source assets')
    partial = destination.with_name(destination.name + '.partial')
    request = urllib.request.Request(url, headers={'User-Agent': 'PBPK-project-pinned-public-assets/1'})
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=45) as response, partial.open('xb') as handle:
            length = response.headers.get('Content-Length')
            if length is not None and int(length) > MAX_FILE_BYTES:
                raise ValueError('Public source asset exceeds the bounded download budget')
            while chunk := response.read(1024**2):
                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    raise ValueError('Public source asset exceeds the bounded download budget')
                handle.write(chunk)
        if total == 0:
            raise ValueError('Public source asset is empty')
        partial.rename(destination)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return {'repository': f'https://github.com/{OWNER}/{repo}', 'commit': commit,
            'repository_path': source_path, 'download_url': url,
            'path': str(destination.relative_to(ROOT)), 'sha256': sha_file(destination),
            'bytes': total, 'role': role, 'license': 'GPL-2.0 per pinned repository README'}


def acquire(output=DEFAULT_OUTPUT, *, resume=False):
    output = inside(output)
    prior = None
    if output.exists() and not resume:
        raise FileExistsError('Use a new destination; source acquisitions are immutable')
    if resume:
        prior_path = output / 'acquisition_manifest.json'
        prior = json.loads(prior_path.read_text())
        if prior['status'] != 'partial' or (output / 'acquisition_manifest_completion.json').exists():
            raise ValueError('Only an unfinished immutable acquisition can be resumed once')
        for row in prior['downloaded_assets']:
            if sha_file(inside(ROOT / row['path'])) != row['sha256']:
                raise ValueError('Previously acquired source changed')
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise OSError('At least 2 GiB project filesystem free space required')
    catalog = {row['candidate_id']: row for row in map(json.loads, CATALOG.open())}
    existing, specs = [], []
    for cid, (compound, repo) in CANDIDATES.items():
        candidate = catalog[cid]
        provenance = candidate['provenance']
        source = inside(ROOT / provenance['source_path'])
        digest = sha_file(source)
        if digest != provenance['source_sha256']:
            raise ValueError(f'Catalog source bytes changed: {cid}')
        commit = provenance['repository_commit']
        existing.append({'candidate_id': cid, 'compound': compound,
            'repository': f'https://github.com/{OWNER}/{repo}', 'commit': commit,
            'path': str(source.relative_to(ROOT)), 'sha256': digest, 'bytes': source.stat().st_size,
            'role': 'Exact catalog source project; embedded complete simulations require native export',
            'license_status': provenance['license_status']})
        specs.append((repo, commit, 'README.md', 'Pinned author model scope, dependencies and GPLv2 declaration'))
    specs += [
        ('Pregnancy-Models', PREGNANCY_COMMIT, 'Models/Caffeine/Caffeine_Pregnancy.mbp3',
         'Full author MoBi caffeine pregnancy project; ancillary exact-commit source'),
        ('Pregnancy-Models', PREGNANCY_COMMIT, 'BuildingBlocks/SpatialStructure_Pregnancy.pkml',
         'Complete author pregnancy spatial structure; not substituted into catalog candidates'),
        ('Pregnancy-Models', PREGNANCY_COMMIT, 'BuildingBlocks/PassiveTransport_Pregnancy.pkml',
         'Author pregnancy passive transport building block; not a runnable simulation'),
        ('Pregnancy-Models', PREGNANCY_COMMIT, 'BuildingBlocks/PassiveTransport_Pregnancy_with-fetal-fu.pkml',
         'Alternative author fetal-fraction-unbound transport; acquired but never applied automatically'),
        ('Suite', LICENSE_COMMIT, 'LICENSE', 'Full GPLv2 license linked by author repositories'),
    ]
    if prior:
        finished = {(r['repository'].split('/')[-1], r['commit'], r['repository_path']) for r in prior['downloaded_assets']}
        specs = [spec for spec in specs if spec[:3] not in finished]
    else:
        output.mkdir(parents=True)
    assets, failures = list(prior['downloaded_assets']) if prior else [], []
    with ThreadPoolExecutor(max_workers=1 if resume else 3) as pool:
        jobs = [(spec, pool.submit(fetch_asset, spec, output)) for spec in specs]
        for spec, job in jobs:
            try:
                assets.append(job.result())
            except Exception as exc:
                failures.append({'repository': spec[0], 'commit': spec[1], 'path': spec[2],
                                 'error': type(exc).__name__ + ': ' + str(exc)})
    manifest = {'schema': 'pregnancy-native-source-acquisition-v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'catalog': {'path': str(CATALOG.relative_to(ROOT)), 'sha256': sha_file(CATALOG)},
        'implementation': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha_file(Path(__file__))},
        'existing_exact_catalog_projects': existing, 'downloaded_assets': assets,
        'failures': failures, 'status': 'complete' if not failures else 'partial',
        'solver_calls': 0, 'clinical_observations_accessed': False,
        'frozen_experiment_modified': False, 'four_compartment_replacement_used': False,
        'limitations': [
            'MoBi project and building-block files are not interchangeable with complete simulation PKML.',
            'Original pregnancy population physiology and maternal/fetal mechanisms must remain intact.',
            'Candidate execution requires its own source-bound complete simulation and native import audit.',
            'Availability of source assets does not establish applicability to nonpregnant clinical targets.']}
    if prior:
        manifest['resumed_acquisition'] = {'path': str(prior_path.relative_to(ROOT)), 'sha256': sha_file(prior_path)}
    write_json(output / ('acquisition_manifest_completion.json' if prior else 'acquisition_manifest.json'), manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = acquire(args.output, resume=args.resume)
    print(json.dumps({'status': result['status'], 'downloaded_assets': len(result['downloaded_assets']),
                      'bytes': sum(row['bytes'] for row in result['downloaded_assets']),
                      'failures': result['failures']}, ensure_ascii=False))
    raise SystemExit(0 if result['status'] == 'complete' else 1)
