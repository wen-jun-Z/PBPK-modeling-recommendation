"""Audit author-provided native companions of the fifteen frozen JSON sources.

No later release, similar drug model or native numerical result is substituted.
Existing verified files are referenced, never needlessly copied. Same-commit
coexistence is provenance evidence, not proof of parameter equivalence.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/model_recommendation/execution_validation_20261001'
OUT = ROOT / 'artifacts/pbpk-model-assets/execution_repairs_20261001/json_companion_sources'
PREFIXES = ('0a4fbffe','3d0876ba','54f5bcdc','64730253','6a7f9778','6f2efe23','77d4db0e',
            '7b45f757','86ab3bd9','998eb58f','c965f50e','ccc06c1b','d3b81c1b','e874015f','f6bf2b17')
MAX_TREE_BYTES = 3 * 1024**2
MAX_DOCUMENT_BYTES = 512 * 1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def local(path):
    p = (ROOT / path).resolve(strict=True)
    if not p.is_relative_to(ROOT) or any(x.casefold().startswith('sealed') for x in p.parts):
        raise ValueError('PROJECT_UNSEALED_RESOURCE_REQUIRED')
    return p


def ref(path):
    p = local(path)
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'bytes': p.stat().st_size}


def fetch(url, limit):
    with urlopen(Request(url, headers={'User-Agent': 'PBPK-same-commit-companion-audit'}), timeout=25) as r:
        if r.geturl() != url: raise ValueError('CROSS_REPOSITORY_REDIRECT_NOT_ACCEPTED')
        body = r.read(limit + 1)
        if len(body) > limit: raise ValueError('METADATA_DOWNLOAD_SIZE_LIMIT')
        return body


def tree_document(repository, commit):
    url = f'https://github.com/{repository}/tree/{commit}'
    body = fetch(url, MAX_TREE_BYTES)
    routes = []
    for script in re.findall(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>', body.decode(), re.S):
        try: payload = json.loads(script).get('payload', {})
        except ValueError: continue
        if 'codeViewTreeRoute' in payload: routes.append(payload['codeViewTreeRoute'])
    if len(routes) != 1: raise ValueError('EXACTLY_ONE_AUTHOR_TREE_REQUIRED')
    route = routes[0]
    if route.get('refInfo', {}).get('currentOid') != commit or route.get('path') != '/':
        raise ValueError('AUTHOR_TREE_COMMIT_OR_ROOT_MISMATCH')
    items = route['tree']['items']
    if route['tree']['totalCount'] != len(items): raise ValueError('TRUNCATED_AUTHOR_ROOT_TREE')
    return {'url': url, 'repository': repository, 'commit': commit, 'currentOid': commit,
            'response_sha256': hashlib.sha256(body).hexdigest(), 'items': items,
            'total_count': len(items), 'scope': 'root_only; full nested inventory from original complete collection manifest'}


def license_document(repository, commit, source_directory, manifest):
    source = local(Path(source_directory) / 'README.md')
    declared = [r for r in manifest['files'] if r['path'] == 'README.md']
    if len(declared) != 1 or sha(source) != declared[0]['sha256']:
        raise ValueError('FROZEN_AUTHOR_README_SHA_MISMATCH')
    url = f'https://raw.githubusercontent.com/{repository}/{commit}/README.md'
    body = fetch(url, MAX_DOCUMENT_BYTES)
    if hashlib.sha256(body).hexdigest() != declared[0]['sha256']:
        raise ValueError('OFFICIAL_FROZEN_README_DIFFERS_FROM_LOCAL')
    text = body.decode('utf-8-sig')
    for section in re.split(r'(?m)^#{1,6}\s+', text):
        if not re.match(r'licen[cs]e\s*\n', section, re.I): continue
        if re.search(r'\b(?:the|this) model(?: code|s)? (?:is|are) distributed under (?:the )?\[?GPL\s*v?2\b', ' '.join(section.split()), re.I):
            return {**ref(source), 'url': url, 'license_id': 'GPL-2.0',
                    'scope': 'explicit_original_repository_model_grant', 'grant_text': section.strip()}
    raise ValueError('NO_EXPLICIT_MODEL_LICENSE_GRANT')


def audit_one(receipt_path, audit_rows):
    receipt = json.loads(receipt_path.read_text())
    source = local(receipt['source_path'])
    if sha(source) != receipt['source_sha256']: raise ValueError('JSON_SOURCE_SHA_CHANGED')
    manifest_path = source.parent.parent / 'asset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    repository, commit = manifest['repository'], manifest['commit']
    if not re.fullmatch(r'Open-Systems-Pharmacology/[A-Za-z0-9_.-]+', repository) or not re.fullmatch(r'[a-f0-9]{40}', commit):
        raise ValueError('ORIGINAL_AUTHOR_REPOSITORY_AND_FULL_COMMIT_REQUIRED')
    source_rows = [x for x in audit_rows if x['catalog_provenance']['source_path'] == receipt['source_path']]
    if not source_rows or any(x['catalog_provenance']['repository_commit'] != commit or
                              x['catalog_provenance']['source_sha256'] != receipt['source_sha256'] for x in source_rows):
        raise ValueError('FROZEN_CATALOGUE_SOURCE_BINDING_MISMATCH')
    source_relative = str(source.relative_to(local(manifest['source_directory'])))
    source_entries = [x for x in manifest['files'] if x['path'] == source_relative]
    if len(source_entries) != 1 or source_entries[0]['sha256'] != receipt['source_sha256']:
        raise ValueError('COLLECTION_MANIFEST_JSON_BINDING_MISMATCH')
    row = {'json_source': ref(source), 'candidate_ids': sorted({x['candidate_id'] for x in source_rows}),
           'same_parent_scenario_ids': sorted({x['scenario_id'] for x in source_rows if x['identity_match']}),
           'repository': repository, 'commit': commit, 'original_asset_manifest': ref(manifest_path),
           'original_migration_receipt': ref(receipt_path), 'companions': [],
           'observations_read': False, 'ode_executed': False, 'scientific_equivalence_verified': False,
           'execution_authorized_by_this_receipt': False}
    tree = tree_document(repository, commit)
    dirname = OUT / (repository.split('/')[-1] + '__' + commit[:12])
    dirname.mkdir(parents=True, exist_ok=True)
    treefile = dirname / 'official_root_tree.json'
    treefile.write_text(json.dumps(tree, indent=2, ensure_ascii=False) + '\n')
    row['official_frozen_root_tree'] = ref(treefile)
    try: row['license'] = license_document(repository, commit, manifest['source_directory'], manifest)
    except (ValueError, OSError) as exc: row['license_error'] = str(exc)
    full_manifest = manifest.get('collection_scope') != 'selected_sources_with_git_blob_verification'
    native_entries = [x for x in manifest['files'] if Path(x['path']).suffix.casefold() in {'.pksim5', '.pkml'}]
    remote_native = {i['path'] for i in tree['items'] if i['contentType'] == 'file' and Path(i['path']).suffix.casefold() in {'.pksim5', '.pkml'}}
    if not remote_native.issubset({x['path'] for x in native_entries}):
        raise ValueError('AUTHOR_ROOT_NATIVE_FILE_MISSING_FROM_EXISTING_COLLECTION')
    # Source models can contain original study references. Only simulation names
    # are selected; no ObservedData or numerical result payload is inspected.
    snapshot = json.loads(source.read_text(encoding='utf-8-sig'))
    source_names = sorted(s.get('Name') for s in snapshot.get('Simulations', []))
    for entry in native_entries:
        candidate = local(Path(manifest['source_directory']) / entry['path'])
        if sha(candidate) != entry['sha256'] or candidate.stat().st_size != entry['bytes']:
            raise ValueError('EXISTING_NATIVE_COMPANION_HASH_MISMATCH')
        native = {**ref(candidate), 'source': 'existing_verified_same_commit_author_artifact',
                  'url': f'https://raw.githubusercontent.com/{repository}/{commit}/{quote(entry["path"], safe="/")}',
                  'same_basename_stem': candidate.stem == source.stem,
                  'parameter_equivalence_verified': False, 'structure_equivalence_verified': False}
        if candidate.suffix == '.pksim5':
            with sqlite3.connect(candidate.as_uri() + '?mode=ro&immutable=1', uri=True) as db:
                names = sorted(x[0] for x in db.execute('SELECT b.Name FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId'))
            native.update(source_simulation_count=len(source_names), native_simulation_count=len(names),
                          simulation_name_multiset_equal=source_names == names,
                          json_only_names=sorted(set(source_names)-set(names)), native_only_names=sorted(set(names)-set(source_names)))
            if sha(candidate) != entry['sha256']: raise ValueError('ORIGINAL_NATIVE_SOURCE_CHANGED')
        row['companions'].append(native)
    row['complete_original_inventory'] = full_manifest
    row['status'] = ('same_commit_native_companion_available_requires_semantic_validation' if row['companions'] else
                     'no_native_companion_in_complete_frozen_repository' if full_manifest else 'native_inventory_incomplete')
    row['latest_release_substitution_permitted'] = False
    row['additional_blockers'] = [] if 'license' in row else ['explicit_original_license_not_verified']
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'manifest.json'
    if target.exists(): raise FileExistsError('Existing companion audit is immutable')
    audit = BASE / 'inputs/candidate_audit.jsonl'
    rows = [json.loads(x) for x in audit.read_text().splitlines() if x.strip()]
    receipts = []
    for prefix in PREFIXES:
        matches = list((BASE/'migrations').glob(prefix+'*/receipt.json'))
        if len(matches) != 1: raise ValueError('Ambiguous requested source prefix ' + prefix)
        receipts.append(matches[0])
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(path, pool.submit(audit_one, path, rows)) for path in receipts]
        for path, future in futures:
            try: row = future.result()
            except Exception as exc: row = {'original_migration_receipt': ref(path), 'status':'audit_incomplete', 'reason': type(exc).__name__+': '+str(exc)}
            results.append(row)
            print(json.dumps({'source': row.get('json_source',{}).get('path'), 'status':row['status'], 'companions':len(row.get('companions',[]))}), flush=True)
    result = {'version':'same-commit-json-native-companion-audit-v1','created_at':datetime.now(timezone.utc).isoformat(),
              'candidate_audit':ref(audit),'collector':ref(Path(__file__)),
              'limits':{'requests_max':30,'parallel_requests_max':3,'timeout_per_request_seconds':25,
                        'tree_bytes_max':MAX_TREE_BYTES,'license_document_bytes_max':MAX_DOCUMENT_BYTES,
                        'new_native_model_bytes_downloaded':0},
              'claim':'Co-presence and source hashes do not prove native/JSON scientific equivalence. Do not substitute the companion before exact model and settings validation.',
              'sources':results}
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
