"""Authenticate one unchanged preparation's historical code before outcomes.

This compatibility rule is limited to the registered original preparation and
its complete 24-file archive. It does not validate candidate predictions, waive
any other barrier, modify a preparation, or permit outcome access.
"""
from pathlib import Path
import re

RUN = 'experiments/model_recommendation/execution_validation_20261001'
PREPARATION = RUN + '/preparation.json'
PREPARATION_SHA = '5a5d7c275d4c1e925085d22040f2db2bd6434cdf95a1528c0a74128340a2d843'
ARCHIVE_BASE = RUN + '/engineering_amendments/repair_04_source_representation_05'
ARCHIVE_MANIFEST = ARCHIVE_BASE + '/preservation.json'
ARCHIVE_MANIFEST_SHA = '484af0ed8791789e0af0d6d49e2b04ae5026cd60882763d3182e9b3844652a14'
MANIFEST = RUN + '/engineering_amendments/json_native_bridge_followup_20261002/preparation_binding_followup/preparation_code_binding.json'
# Fail closed until the independently prepared manifest is pinned before install.
MANIFEST_SHA = '29d806b2b7f8ed3f0ece58697142aa33c9bffb3c7b83ea6ecbe6ce25623c08b6'
SCHEMA = 'pbpk-historical-preparation-code-binding-v1'
REVISION = 'engineering_repair_04'
ORIGINAL_REVISION = 'engineering_repair_02'
CODE_COUNT = 24


def _require(condition, reason):
    if not condition:
        raise ValueError('Historical preparation: ' + reason)


def _reference(value):
    _require(isinstance(value, dict) and set(value) == {'path', 'sha256'},
             'reference must have exact path and sha256 fields')
    path, checksum = value['path'], value['sha256']
    _require(isinstance(path, str) and bool(path), 'reference path missing')
    parsed = Path(path)
    _require(not parsed.is_absolute() and parsed.as_posix() == path and '..' not in parsed.parts,
             'reference path is not a canonical project-relative path')
    _require(not any(p.casefold().startswith('sealed') for p in parsed.parts),
             'sealed resource forbidden')
    _require(isinstance(checksum, str) and re.fullmatch(r'[0-9a-f]{64}', checksum) is not None,
             'reference checksum invalid')
    return value


def _checked(f, value):
    value = _reference(value)
    root = Path(f.root).resolve(strict=True)
    target = f.path(value['path'])
    _require(target.is_relative_to(root) and target != root and target.is_file(),
             'resource outside project or not a file')
    _require(not any(p.casefold().startswith('sealed') for p in target.relative_to(root).parts),
             'resolved sealed resource forbidden')
    _require(f.checked(value) == target, 'checked resource path differs')
    return target


def _bound(protocol, value):
    value = _reference(value)
    inputs = protocol.get('inputs')
    _require(isinstance(inputs, list) and all(isinstance(r, dict) for r in inputs),
             'protocol input list invalid')
    matches = [r for r in inputs if r.get('path') == value['path']]
    _require(matches == [value], 'resource must be uniquely and exactly frozen: ' + value['path'])


def _unique_refs(values, label, *, expected_count=None):
    _require(isinstance(values, list) and (expected_count is None or len(values) == expected_count),
             label + ' reference count differs')
    result = {}
    for value in values:
        value = _reference(value)
        _require(value['path'] not in result, label + ' has duplicate paths')
        result[value['path']] = value
    return result


def verify_preparation_binding(f, run, protocol, frozen_code_sha256):
    """Verify all old/new bytes and archive mappings without opening outcomes."""
    _require(getattr(f, 'unsealed', None) is False, 'outcome gate must remain closed')
    _require(not getattr(f, 'opened_outcomes', {}), 'outcomes were already opened')
    root = Path(f.root).resolve(strict=True)
    _require(f.path(run) == f.path(RUN), 'only the fixed registered experiment is supported')
    _require(protocol.get('execution_revision') == REVISION, 'protocol revision differs')
    _require(protocol.get('outcome_access', {}).get('sealed_values_read_by_preparation') is False
             and protocol.get('outcome_access', {}).get('all_predictions_and_rankings_locked_before_scoring') is True,
             'protocol outcome barrier differs')
    manifest_ref = _reference({'path': MANIFEST, 'sha256': MANIFEST_SHA})
    preparation_ref = {'path': PREPARATION, 'sha256': PREPARATION_SHA}
    archive_ref = {'path': ARCHIVE_MANIFEST, 'sha256': ARCHIVE_MANIFEST_SHA}
    for value in (manifest_ref, preparation_ref, archive_ref):
        _bound(protocol, value)
        _checked(f, value)
    manifest = f.read(f.path(MANIFEST))
    _require(isinstance(manifest, dict) and set(manifest) == {
        'schema_version', 'execution_revision', 'sealed_observations_accessed',
        'preparation', 'archive_manifest', 'code_count', 'code'}, 'manifest fields differ')
    _require(manifest['schema_version'] == SCHEMA and manifest['execution_revision'] == REVISION,
             'manifest schema or revision differs')
    _require(manifest['sealed_observations_accessed'] is False
             and manifest['preparation'] == preparation_ref
             and manifest['archive_manifest'] == archive_ref
             and type(manifest['code_count']) is int and manifest['code_count'] == CODE_COUNT,
             'manifest fixed preparation/archive/count binding differs')
    preparation = f.read(f.path(PREPARATION))
    _require(preparation.get('schema_version') == 'pbpk-execution-preparation-freeze-v1'
             and preparation.get('execution_revision') == ORIGINAL_REVISION
             and preparation.get('sealed_outcomes_accessed') is False,
             'original preparation identity/outcome barrier differs')
    original_code = _unique_refs(preparation.get('code'), 'original preparation code', expected_count=CODE_COUNT)
    original_inputs = _unique_refs(preparation.get('inputs'), 'original preparation inputs')
    for value in original_inputs.values():
        _checked(f, value)
    preserved = f.read(f.path(ARCHIVE_MANIFEST))
    _require(preserved.get('sealed_outcomes_accessed') is False and isinstance(preserved.get('code'), list),
             'archive preservation manifest invalid')
    archive_code = {}
    for entry in preserved['code']:
        _require(isinstance(entry, dict) and set(entry) == {'original', 'archived'},
                 'archive mapping fields differ')
        original, archived = _reference(entry['original']), _reference(entry['archived'])
        _require(original['path'] not in archive_code, 'archive has duplicate original paths')
        _require(archived == {'path': ARCHIVE_BASE + '/frozen_files/' + original['path'],
                              'sha256': original['sha256']}, 'archive path or preserved bytes differ')
        archive_code[original['path']] = entry
    current_refs = _unique_refs(protocol.get('code'), 'current protocol code')
    _require(isinstance(frozen_code_sha256, dict), 'current verified code map missing')
    resolved_code = {}
    for path, value in current_refs.items():
        resolved = f.path(path)
        _require(resolved.is_relative_to(root), 'current code outside project')
        canonical = str(resolved.relative_to(root))
        _require(canonical not in resolved_code, 'current code has duplicate resolved paths')
        resolved_code[canonical] = value['sha256']
    _require(resolved_code == frozen_code_sha256, 'caller verified code map differs from protocol')
    entries = manifest['code']
    _require(isinstance(entries, list) and len(entries) == CODE_COUNT, 'complete 24-code mapping required')
    mapped = set()
    changed = []
    for entry in entries:
        _require(isinstance(entry, dict) and set(entry) == {'original', 'archived', 'current'},
                 'binding code entry fields differ')
        original, archived, current = (_reference(entry[k]) for k in ('original', 'archived', 'current'))
        path = original['path']
        _require(path not in mapped and path in original_code, 'duplicate or extra original code mapping')
        mapped.add(path)
        _require(original == original_code[path], 'old code differs from the fixed preparation')
        _require(archive_code.get(path) == {'original': original, 'archived': archived},
                 'old code lacks the exact independently preserved mapping')
        _bound(protocol, archived)
        archived_path = _checked(f, archived)
        _require(str(archived_path.relative_to(root)) == archived['path'],
                 'historical code archive path must not redirect')
        _require(current['path'] == path and current_refs.get(path) == current,
                 'new code differs from current protocol binding')
        current_path = _checked(f, current)
        _require(frozen_code_sha256.get(str(current_path.relative_to(root))) == current['sha256'],
                 'new actual code differs from caller verified binding')
        if original['sha256'] != current['sha256']:
            changed.append(path)
    _require(mapped == set(original_code), 'original preparation code coverage incomplete')
    _require(getattr(f, 'unsealed', None) is False and not getattr(f, 'opened_outcomes', {}),
             'historical preparation audit must never open outcomes')
    return {'accepted': True, 'preparation': preparation_ref, 'binding_manifest': manifest_ref,
            'archive_manifest': archive_ref, 'historical_code_files': CODE_COUNT,
            'original_inputs_verified': len(original_inputs), 'changed_code_paths': sorted(changed),
            'old_bytes_verified_against_actual_archive': True,
            'current_bytes_verified_against_protocol': True,
            'sealed_observations_accessed': False}
