"""Read-only bridge from an authenticated JSON source to its own native build.

A shared basename, matching drug or neighboring directory is never provenance.
Only a hash-bound completed build of the exact verified snapshot partition, or
an authenticated export that equals it, can supply an executable saved model.
This module does not construct models, run ODEs or inspect observed outcomes.
"""
from copy import deepcopy
from collections import OrderedDict
import threading
import hashlib
import json
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET

import recommendation_candidate_migration as migration
from . import snapshot_repair, source_pkml

ROOT = migration.ROOT
VERSION = 'source-bound-json-native-bridge-v1'
_INVENTORY_CACHE = OrderedDict()
_INVENTORY_LOCK = threading.RLock()
_INVENTORY_CACHE_LIMIT = 64
# Set only to the reviewed immutable build registry; old source receipts stay unchanged.
SUPPLEMENTAL_BUILDS = 'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/json_native_builds_20261002/supplemental_completed_008.json'
SUPPLEMENTAL_BUILDS_SHA = '9dfd17f941be20b433392d54968fa8baf498f111afd53b6faa05e7481997d0da'
IMPLEMENTATION_FILES = tuple(sorted(set(snapshot_repair.IMPLEMENTATION_FILES) | {
    'scripts/recommendation_execution_repairs/json_native_source_proof.py',
    'scripts/recommendation_execution_repairs/source_pkml.py',
    'scripts/recommendation_execution_repairs/json_fixed_process_defaults_proof.py',
    'scripts/recommendation_execution_repairs/json_expression_profile_proof.py',
}))


def _require(condition, message):
    if not condition:
        raise ValueError('JSON native bridge: ' + message)


def _read(path):
    _require(path.stat().st_size <= 64 * 1024**2, 'bounded JSON resource required')
    return json.loads(path.read_text())


def _exact(a, b):
    if type(a) in (int, float) and type(b) in (int, float):
        import math
        return math.isfinite(a) and math.isfinite(b) and a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_exact(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_exact(x, y) for x, y in zip(a, b))
    return a == b


def _same_snapshot(a, b):
    def normalized(document):
        value = deepcopy(document)
        # PK-Sim may omit an empty top-level collection on export. This is
        # equivalent only if the partition does not reference any population.
        if value.get('Populations') == [] and not any(s.get('Population') for s in value.get('Simulations', [])):
            value.pop('Populations')
        return migration._normalize(migration.clean_snapshot(value))
    return _exact(normalized(a), normalized(b))


def _names(document):
    rows = document.get('Simulations')
    _require(isinstance(rows, list) and 1 <= len(rows) <= 256, 'bounded source simulations required')
    names = [r.get('Name') for r in rows]
    _require(all(isinstance(n, str) and n for n in names) and len(names) == len(set(names)),
             'unique source simulation names required')
    return names


def _partition_matches(document, verified):
    if document.get('Version') != 80:
        return False
    names = _names(document)
    if not set(names) <= set(_names(verified)):
        return False
    expected = deepcopy(verified)
    expected['Simulations'] = [s for s in verified['Simulations'] if s['Name'] in names]
    if _same_snapshot(document, expected):
        return True
    # Existing complex proof drops only populations unused by this exact batch.
    required = {s['Population'] for s in expected['Simulations'] if s.get('Population')}
    if 'Populations' in expected:
        expected['Populations'] = [p for p in expected['Populations'] if p['Name'] in required]
    return _same_snapshot(document, expected)


def _references(refs, root):
    unique = {}
    for ref in refs:
        _require(isinstance(ref, dict) and set(ref) >= {'path', 'sha256'}, 'invalid evidence reference')
        clean = {'path': ref['path'], 'sha256': ref['sha256']}
        _require(clean['path'] not in unique or unique[clean['path']] == clean,
                 'conflicting evidence reference')
        migration.checked(clean, root)
        unique[clean['path']] = clean
    return unique


def _process(ref, *, root, refs, mode, script_sha):
    _require(refs.get(ref['path']) == ref, 'process was not bound by source receipt')
    path = migration.checked(ref, root)
    process = _read(path)
    _require(process.get('status') == 'completed' and type(process.get('returncode')) is int
             and process['returncode'] == 0 and process.get('runtime_unchanged') is True
             and process.get('registry_modified') is False,
             'native stage did not complete with unchanged runtime')
    _require(process.get('script_sha256') == script_sha
             and migration.inside(process.get('script', ''), root) == root / migration.R_SCRIPT,
             'native stage worker identity differs')
    command = process.get('command', [])
    _require(len(command) == 6 and command[1] == '--vanilla' and command[3] == mode
             and migration.inside(command[2], root) == root / migration.R_SCRIPT,
             'native stage command differs')
    resources = process.get('runtime_binding', {}).get('locked_resources')
    _require(isinstance(resources, list) and bool(resources), 'native stage runtime lock missing')
    runtime_refs = _references(resources, root)
    return process, migration.inside(command[5], root), list(runtime_refs.values())


def _inventory(project, names, root):
    # Cache only parsed immutable SQLite/XML inventory. Every call still hashes
    # the complete native file and every upstream receipt/runtime reference.
    path = migration.checked(project, root)
    key = (str(path), project['sha256'], tuple(sorted(names)),
           migration.sha(Path(__file__)), migration.sha(Path(source_pkml.__file__)))
    with _INVENTORY_LOCK:
        cached = _INVENTORY_CACHE.get(key)
        if cached is not None:
            _require(migration.sha(path) == project['sha256'], 'cached native project changed while read')
            _INVENTORY_CACHE.move_to_end(key)
            return deepcopy(cached)
    rows = source_pkml.simulation_inventory(path, project['sha256'], project_root=root)
    _require(len(rows) == len(names) and {r['name'] for r in rows} == set(names),
             'saved project simulation coverage differs from exact JSON partition')
    result = []
    with sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        for row in rows:
            blob = db.execute('SELECT Data FROM CONTENTS WHERE Id=?', (row['content_id'],)).fetchone()[0]
            document = ET.fromstring(source_pkml.decompress_model_blob(blob))
            _require(document.get('id') == row['simulation_id'] and document.get('name') == row['name']
                     and document.find('Model') is not None, 'saved model identity/content differs')
            result.append({'name': row['name'], 'id': row['simulation_id'],
                           'source_blob_sha256': hashlib.sha256(blob).hexdigest()})
    _require(migration.sha(path) == project['sha256'], 'native project changed while read')
    with _INVENTORY_LOCK:
        _INVENTORY_CACHE[key] = deepcopy(result)
        _INVENTORY_CACHE.move_to_end(key)
        while len(_INVENTORY_CACHE) > _INVENTORY_CACHE_LIMIT:
            _INVENTORY_CACHE.popitem(last=False)
    return result


def _verified(provenance, root):
    resolved = snapshot_repair.resolve_repaired_snapshot(provenance, project_root=root)
    receipt_path = Path(resolved['receipt_path'])
    receipt = _read(receipt_path)
    verified = _read(migration.checked(receipt['snapshot'], root))
    _require(verified.get('Version') == 80, 'verified native snapshot required')
    _names(verified)
    source = {'path': receipt['source_path'], 'sha256': receipt['source_sha256']}
    _require(source == {'path': provenance.get('source_path'), 'sha256': provenance.get('source_sha256')},
             'verified receipt belongs to another catalog source')
    receipt_ref = migration.reference(receipt_path, root)
    refs = _references([*receipt['provenance_refs'], source, receipt_ref, receipt['snapshot']], root)
    script = [r for r in receipt['implementation'] if r['path'] == migration.R_SCRIPT]
    _require(len(script) == 1, 'migration worker identity missing from source proof')
    return receipt, verified, refs, source, receipt_ref, script[0]['sha256']


def _validate_supplemental_readbacks(entry, refs, root, script_sha):
    _require(entry.get('readback_required') is True and isinstance(entry.get('readbacks'), list)
             and bool(entry['readbacks']), 'supplemental native builds require explicit readback')
    seen = set()
    comparison_only_projects = set()
    for item in entry['readbacks']:
        _require(isinstance(item, dict) and set(item) == {'native_source','partition_input','export_snapshot','comparison'},
                 'supplemental readback fields differ')
        for ref in item.values():
            _require(refs.get(ref.get('path')) == ref, 'supplemental readback evidence is unregistered')
        native, inp, exported, comparison = (item[k] for k in
            ('native_source','partition_input','export_snapshot','comparison'))
        _require(native['path'] not in seen, 'supplemental native readback is ambiguous')
        seen.add(native['path'])
        original = _read(migration.checked(inp, root)); readback = _read(migration.checked(exported, root))
        audit = _read(migration.checked(comparison, root))
        _require(original.get('Version') == readback.get('Version') == 80, 'supplemental native version differs')
        _require(audit.get('accepted') is True and audit.get('numeric_tolerance') == {'absolute':0,'relative':0}
                 and audit.get('compound_parameters_fitted') is False, 'supplemental comparison changed numeric inputs')
        if _same_snapshot(original,readback):
            _require(audit.get('exact_normalized_document_equality') is True, 'exact readback was not recorded accurately')
        else:
            _require(audit.get('exact_normalized_document_equality') is False, 'semantic equivalence cannot claim literal equality')
            proof=audit.get('semantic_proof',{})
            _require(refs.get(proof.get('path'))==proof, 'semantic proof is not frozen')
            rule=audit.get('equivalence_rule')
            if rule=='source_bound_fixed_process_calculation_defaults':
                from . import json_fixed_process_defaults_proof as verifier
            elif rule=='source_bound_native_expression_profile_serialization':
                from . import json_expression_profile_proof as verifier
            else:
                raise ValueError('JSON native bridge: unregistered semantic readback rule')
            result=verifier.verify(inp,exported,native,proof,project_root=root)
            _require(result.get('accepted') is True, 'source-bound native semantic verification failed')
            content=_read(migration.checked(proof,root))
            evidence=content.get('evidence_refs',content.get('provenance_refs'))
            _require(isinstance(evidence,list) and evidence, 'semantic evidence is absent')
            _require(all(refs.get(r.get('path'))==r for r in evidence), 'semantic evidence is not fully frozen')
            if rule=='source_bound_native_expression_profile_serialization':
                # The verifier has independently reproduced this literal-original
                # project's full models. It is evidence, never an execution source.
                build_ref=content['original_equivalence']['build_result']
                _require(refs.get(build_ref.get('path'))==build_ref, 'original comparison build is not frozen')
                original_native=_read(migration.checked(build_ref,root))['native']
                _require(refs.get(original_native.get('path'))==original_native
                         and Path(original_native['path']).suffix=='.pksim5'
                         and original_native['path']!=native['path'],
                         'original comparison native project is not uniquely frozen')
                comparison_only_projects.add(original_native['path'])
        stages = []
        for ref in refs.values():
            if ref['path'].endswith('/stage_receipt.json'):
                stage = _read(migration.checked(ref, root))
                if stage.get('mode') == 'project_to_snapshot' and stage.get('input') == native:
                    process, output, runtime = _process(stage['process'], root=root, refs=refs,
                        mode='project_to_snapshot', script_sha=script_sha)
                    if migration.inside(exported['path'],root).parent != output:
                        continue
                    _require(stage.get('status') == 'completed' and stage.get('copy_before_sha256') == native['sha256']
                             and stage.get('original_after_sha256') == native['sha256'],
                             'supplemental export native source changed')
                    stages.append(ref)
        _require(len(stages) == 1, 'supplemental export lacks a unique original-native process')
    projects = {r['path'] for r in refs.values() if Path(r['path']).suffix == '.pksim5'}
    _require(not seen.intersection(comparison_only_projects)
             and seen | comparison_only_projects == projects,
             'supplemental native project has no exact readback or verified original-model comparison')


def _supplemental_refs(source, receipt_ref, verified_ref, root):
    if SUPPLEMENTAL_BUILDS is None and SUPPLEMENTAL_BUILDS_SHA is None:
        return []
    _require(isinstance(SUPPLEMENTAL_BUILDS, str) and isinstance(SUPPLEMENTAL_BUILDS_SHA, str),
             'supplemental build registry requires both fixed path and hash')
    ref = {'path': SUPPLEMENTAL_BUILDS, 'sha256': SUPPLEMENTAL_BUILDS_SHA}
    registry = _read(migration.checked(ref, root))
    _require(registry.get('schema_version') == 'source-bound-supplemental-native-builds-v1'
             and registry.get('clinical_outcomes_accessed') is False
             and isinstance(registry.get('entries'), list), 'invalid supplemental build registry')
    matches = [r for r in registry['entries'] if r.get('source_original') == source]
    _require(len(matches) <= 1, 'supplemental source registration is ambiguous')
    if not matches:
        return []
    entry = matches[0]
    _require(entry.get('source_receipt') == receipt_ref and entry.get('verified_snapshot') == verified_ref,
             'supplemental build belongs to another verified source revision')
    evidence = entry.get('provenance_refs')
    _require(isinstance(evidence, list) and bool(evidence), 'supplemental build evidence missing')
    refs = _references(evidence, root)
    receipt = _read(migration.checked(receipt_ref, root))
    scripts = [r for r in receipt['implementation'] if r['path'] == migration.R_SCRIPT]
    _require(len(scripts) == 1, 'supplemental source worker identity missing')
    _validate_supplemental_readbacks(entry, refs, root, scripts[0]['sha256'])
    return [ref, *refs.values()]


def resolve_native_sources(provenance, *, project_root=ROOT):
    """Return only source-authenticated native projects, without running a build."""
    root = Path(project_root).resolve(strict=True)
    receipt, verified, refs, source, receipt_ref, script_sha = _verified(provenance, root)
    supplemental = _supplemental_refs(source, receipt_ref, receipt['snapshot'], root)
    refs = _references([*refs.values(), *supplemental], root)
    common = {'version': VERSION, 'source_original': source, 'source_receipt': receipt_ref,
              'verified_snapshot': receipt['snapshot'], 'clinical_outcomes_accessed': False,
              'native_process_started': False, 'source_model_identity_preserved': True}
    candidates = {}; rejected = []
    def admit(project, document, mode, evidence):
        if not _partition_matches(document, verified):
            rejected.append({'native_source': project, 'reason': 'not_exact_final_verified_snapshot_partition'})
            return
        _require(refs.get(project['path']) == project, 'native project was not bound by source receipt')
        key = project['path']
        if key in candidates:
            _require(candidates[key]['native_source'] == project
                     and {r['name'] for r in candidates[key]['allowed_simulations']} == set(_names(document)),
                     'native source has contradictory bridge registrations')
            return
        allowed = _inventory(project, _names(document), root)
        proof_refs = _references([source, receipt_ref, receipt['snapshot'], project, *supplemental, *evidence], root)
        candidate = {**common, 'native_source': project, 'allowed_simulations': allowed,
                     'bridge_mode': mode, 'proof_refs': list(proof_refs.values())}
        key = project['path']
        if key not in candidates:
            candidates[key] = candidate
        else:
            _require(candidates[key]['native_source'] == project
                     and candidates[key]['allowed_simulations'] == allowed,
                     'native source has contradictory bridge registrations')
    # Current bounded builds. Output identity is bound by the immutable receipt
    # refs and the real process's output directory; never an unconstrained glob.
    for ref in refs.values():
        if not ref['path'].endswith('/stage_receipt.json'):
            continue
        stage = _read(migration.checked(ref, root))
        if stage.get('mode') != 'snapshot_to_project':
            continue
        inp = stage.get('input', {})
        _require(refs.get(inp.get('path')) == inp, 'build input missing from source receipt')
        document = _read(migration.checked(inp, root))
        if not _partition_matches(document, verified):
            continue
        _require(stage.get('status') == 'completed' and all(stage.get(k) == inp['sha256'] for k in
                 ('copy_before_sha256', 'copy_after_sha256', 'original_after_sha256')),
                 'JSON build input changed or native build failed')
        process, out, runtime = _process(stage['process'], root=root, refs=refs,
                                         mode='snapshot_to_project', script_sha=script_sha)
        projects = [r for r in refs.values() if Path(r['path']).suffix == '.pksim5'
                    and migration.inside(r['path'], root).parent == out]
        _require(len(projects) == 1, 'exactly one source-bound native build output required')
        _require(Path(projects[0]['path']).stem == Path(inp['path']).stem,
                 'native build basename differs from exact input')
        admit(projects[0], document, 'exact_verified_json_build', [ref, inp, stage['process'], *runtime])
    # Partitioned/resumed exports have the same strict source chain with a
    # standalone stage receipt, rather than the earlier aggregate v3 receipt.
    for ref in refs.values():
        if not ref['path'].endswith('/stage_receipt.json'):
            continue
        stage = _read(migration.checked(ref, root))
        if stage.get('mode') != 'project_to_snapshot':
            continue
        inp = stage.get('input', {})
        _require(refs.get(inp.get('path')) == inp and Path(inp['path']).suffix == '.pksim5',
                 'partition export project is absent from source receipt')
        _require(stage.get('status') == 'completed'
                 and stage.get('original_after_sha256') == inp['sha256']
                 and stage.get('copy_before_sha256') == inp['sha256'], 'partition export source changed')
        process, out, runtime = _process(stage['process'], root=root, refs=refs,
                                         mode='project_to_snapshot', script_sha=script_sha)
        exported = [r for r in refs.values() if Path(r['path']).suffix == '.json'
                    and migration.inside(r['path'], root).parent == out]
        if not exported:
            # An earlier failed export may retain its completed process receipt,
            # while a later authenticated resume supplies the actual final output.
            # It cannot prove any native project, but cannot erase another proof.
            rejected.append({'native_source':inp, 'stage_receipt':ref,
                             'reason':'partition_export_has_no_hash_bound_output'})
            continue
        _require(len(exported) == 1, 'exactly one source-bound partition export required')
        document = _read(migration.checked(exported[0], root))
        admit(inp, document, 'exact_verified_partition_export', [ref, inp, exported[0], stage['process'], *runtime])
    # Historical v3 exports have explicit input/output references. The exported
    # solver document must equal the final verified document, not an earlier
    # reconstruction that still needed source-parameter restoration.
    for ref in refs.values():
        if not ref['path'].endswith('/receipt.json') or ref == receipt_ref:
            continue
        history = _read(migration.checked(ref, root))
        if history.get('version') != migration.VERSION:
            continue
        _require(history.get('source_path') == source['path']
                 and history.get('source_sha256') == source['sha256'], 'historical source identity differs')
        for stage in history.get('stages', []):
            if stage.get('mode') != 'project_to_snapshot' or not stage.get('output'):
                continue
            inp, exported = stage['input'], stage['output']
            _require(refs.get(inp['path']) == inp and refs.get(exported['path']) == exported,
                     'native export chain is absent from source receipt')
            document = _read(migration.checked(exported, root))
            if not _partition_matches(document, verified):
                continue
            _require(stage.get('status') == 'completed' and stage.get('source_input_unchanged') is True
                     and stage.get('source_input_sha256_after') == inp['sha256']
                     and stage.get('copied_input_sha256_before') == inp['sha256'], 'native export input changed')
            process, out, runtime = _process(stage['process_manifest'], root=root, refs=refs,
                                              mode='project_to_snapshot', script_sha=script_sha)
            _require(migration.inside(exported['path'], root).parent == out,
                     'native export is outside recorded process output')
            admit(inp, document, 'exact_verified_native_export', [ref, exported, stage['process_manifest'], *runtime])
    all_names = sorted(_names(verified))
    covered = sorted({s['name'] for candidate in candidates.values() for s in candidate['allowed_simulations']})
    missing = sorted(set(all_names) - set(covered))
    return {**common, 'candidates': [candidates[k] for k in sorted(candidates)],
            'rejected_native_artifacts': rejected, 'all_source_simulations': all_names,
            'available_native_simulations': covered, 'unbuilt_simulations': missing,
            'native_coverage_complete': not missing,
            'new_native_build_required': bool(missing)}


def verify_native_source(original_provenance, native_candidate_proof, *, project_root=ROOT):
    """Recompute proof; no caller-supplied project or label can select a source."""
    result = resolve_native_sources(original_provenance, project_root=project_root)
    matches = [p for p in result['candidates'] if p == native_candidate_proof]
    _require(len(matches) == 1, 'selected native source proof is not exact or uniquely source-bound')
    return matches[0]
