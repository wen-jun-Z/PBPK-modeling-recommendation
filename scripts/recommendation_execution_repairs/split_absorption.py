"""Preserve the author's two absorption inputs for one rosuvastatin dose.

This contract is restricted to five frozen original candidates. The original
native model supplies both fractions and the lag; clinical data never supply
absorption parameters. A second input is not an extra clinical administration.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET

VERSION = 'source-bound-split-absorption-v1'
REGISTRY = 'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/json_native_bridge_followup_20261002/split_absorption_source_registry.json'
REGISTRY_SHA256 = '20aab59ffd628ac6c902b47286ec9aee70538cc8c7c153a84848f4d5cb64dab1'
PARENT = 'Rosuvastatin'


def _require(ok, message):
    if not ok: raise ValueError('SPLIT_ABSORPTION_' + message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _checked(ref, root):
    path = (Path(root) / ref['path']).resolve(strict=True)
    _require(path.is_relative_to(Path(root).resolve()) and
        not any(s.lower().startswith('sealed') for s in path.relative_to(root).parts), 'PROJECT_METADATA_ONLY')
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256'], 'SOURCE_CHANGED:' + ref['path'])
    return path


def _registration(provenance, contract, root):
    registry_ref = {'path': REGISTRY, 'sha256': REGISTRY_SHA256}
    document = json.loads(_checked(registry_ref, root).read_text())
    _require(document['schema_version'] == 'source-bound-split-absorption-registry-v1' and
        document['clinical_outcomes_accessed'] is False, 'REGISTRY_CONTRACT')
    original = {'path': provenance['source_path'], 'sha256': provenance['source_sha256']}
    records = [r for r in document['sources'] if r['source_original'] == original]
    _require(len(records) == 1 and records[0]['clinical_contract_sha256'] == _digest(contract), 'FROZEN_SOURCE_AND_CLINICAL_CONTRACT')
    _require(contract['compound'] == 'rosuvastatin' and contract['route'] == 'Oral' and
        contract['dose_mg'] == 40 and contract['administration_times_min'] == [0], 'SINGLE_CLINICAL_DOSE')
    _checked(original, root)
    refs = [registry_ref, original, *document['author_evidence']]
    for ref in document['author_evidence']:
        _require('split dose approach with lag time' in _checked(ref, root).read_text(), 'AUTHOR_ABSORPTION_EVIDENCE')
    return original, refs


def _shape(metadata):
    from . import complete_candidate as c
    groups, assignments = c._metadata_events(metadata)
    _require(len(groups) == 2 and len(assignments) == 2, 'EXACT_TWO_ORIGINAL_INPUTS')
    ordered = []
    for key, group in groups.items():
        a = assignments[key]
        _require(a['administered_compounds'] == [PARENT] and a['application_type'] == 'Oral', 'PURE_ORAL_PARENT_INPUTS')
        _require('Infusion time' not in group, 'NO_INFUSION_INPUT')
        dose, start = group['Dose'], group['Start time']
        _require(dose['dimension'] == 'Mass' and start['dimension'] == 'Time', 'NATIVE_BASE_DIMENSIONS')
        _require(all(not r.get('formula_id') and not r.get('rhs_formula_id') for r in (dose, start)), 'EXPLICIT_SOURCE_DOSE_TIME')
        mass = Fraction(str(dose['value'])) * 1000000
        time = Fraction(str(start['value']))
        _require(mass > 0 and time >= 0, 'POSITIVE_SOURCE_DOSE_TIME')
        ordered.append((time, key, mass, dose['id'], start['id']))
    ordered.sort()
    total = sum((r[2] for r in ordered), Fraction(0))
    _require([r[2] / total for r in ordered] == [Fraction(317, 500), Fraction(183, 500)] and
        ordered[1][0] - ordered[0][0] == 138, 'ORIGINAL_FRACTIONS_AND_LAG')
    return {'source_total_dose_mg': float(total), 'source_event_count': 2,
        'events': [{'event_prefix': r[1], 'dose_parameter_id': r[3], 'start_parameter_id': r[4],
                    'original_dose_mg': float(r[2]), 'original_start_time_min': float(r[0]),
                    'fraction': float(r[2] / total), 'target_dose_mg': float(40 * r[2] / total),
                    'target_start_time_min': float(r[0] - ordered[0][0])} for r in ordered]}


def _metadata_from_native(native, simulation_id, blob_sha, root):
    from . import source_pkml
    path = _checked(native, root)
    with sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        rows = db.execute('SELECT c.Data FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId '
            'JOIN CONTENTS c ON c.Id=b.ContentId WHERE s.SimulationId=?', (simulation_id,)).fetchall()
    _require(len(rows) == 1 and hashlib.sha256(rows[0][0]).hexdigest() == blob_sha, 'EXACT_SAVED_MODEL_BLOB')
    document = ET.fromstring(source_pkml.decompress_model_blob(rows[0][0]))
    _require(document.get('id') == simulation_id, 'EXACT_SIMULATION_ID')
    return source_pkml.describe_saved_simulation(document)


def selection_options(metadata, provenance, contract, source_policy, execution_source, execution_sha, *, project_root):
    from . import complete_candidate as c
    root = Path(project_root).resolve(strict=True)
    original, refs = _registration(provenance, contract, root)
    native = {'path': str(Path(execution_source).resolve().relative_to(root)), 'sha256': execution_sha}
    _checked(native, root)
    if Path(original['path']).suffix == '.pksim5': _require(native == original, 'ORIGINAL_PROJECT_IDENTITY')
    options, excluded = [], []
    internal = {**deepcopy(contract), 'administration_times_min': [0., 138.]}
    for row in metadata['simulations']:
        try:
            shape = _shape(row)
            proposed, rejected = c.select_source_simulation({'simulations': [row]}, internal, source_policy=source_policy)
            _require(len(proposed) == 1 and not rejected and not proposed[0].get('application_expansion'), 'ORIGINAL_SOURCE_DESIGN')
            policy = {'version': VERSION, 'source_original': original, 'native_source': native,
                'clinical_contract_sha256': _digest(contract), 'simulation_id': row['simulation_id'],
                'simulation_name': row['name'], 'source_model_blob_sha256': row['source_blob_sha256'],
                'source_metadata_sha256': _digest(c._original_metadata(row)), 'clinical_administration_count': 1,
                'internal_absorption_event_count': 2, 'clinical_total_dose_mg': 40., 'lag_minutes': 138.,
                'proof_refs': [*refs, native], 'evidence_refs': [*refs, native], 'clinical_outcomes_accessed': False,
                'absorption_parameters_changed': False, **shape}
            option = proposed[0]
            key = list(option['sort_key']); key[4] = 0
            key[5] = abs(math.log(shape['source_total_dose_mg'] / contract['dose_mg']))
            option.update(split_absorption_contract=policy, sort_key=tuple(key))
            options.append(option)
        except (ValueError, KeyError, TypeError) as exc:
            excluded.append({'simulation_id': row.get('simulation_id'), 'simulation': row.get('name'), 'reason': str(exc)})
    options.sort(key=lambda r: r['sort_key'])
    return options, excluded


def revalidate_contract(policy, clinical_contract, source_metadata, *, project_root):
    from . import complete_candidate as c
    _require(policy.get('version') == VERSION and policy.get('clinical_outcomes_accessed') is False and
        policy.get('absorption_parameters_changed') is False, 'POLICY_IDENTITY')
    original = policy['source_original']
    _, refs = _registration({'source_path': original['path'], 'source_sha256': original['sha256']}, clinical_contract, project_root)
    native = policy['native_source']
    if Path(original['path']).suffix == '.pksim5': _require(native == original, 'ORIGINAL_PROJECT_IDENTITY')
    _require(policy['proof_refs'] == [*refs, native] and policy.get('evidence_refs') == [*refs, native], 'SOURCE_PROOF_REFERENCES')
    raw = _metadata_from_native(native, policy['simulation_id'], policy['source_model_blob_sha256'], project_root)
    _require(raw == c._original_metadata(source_metadata) and _digest(raw) == policy['source_metadata_sha256'], 'ORIGINAL_METADATA_CHANGED')
    _require(raw['name'] == policy['simulation_name'], 'ORIGINAL_SIMULATION_NAME')
    shape = _shape(raw)
    _require(all(policy.get(k) == v for k, v in shape.items()) and
        policy.get('clinical_administration_count') == 1 and policy.get('internal_absorption_event_count') == 2 and
        policy.get('clinical_total_dose_mg') == 40. and policy.get('lag_minutes') == 138., 'ABSORPTION_CONTRACT_CHANGED')
    return policy


def build_split_patches(inventory, clinical_contract, source_metadata, policy, *, project_root):
    from . import complete_candidate as c
    revalidate_contract(policy, clinical_contract, source_metadata, project_root=project_root)
    native_groups = {}
    for row in inventory.get('parameters', []):
        if c.native_pkml.ADMINISTRATION.fullmatch(row.get('path', '')):
            prefix, field = c._event_prefix(row['path']); key = c._normalized_event(prefix)
            _require(field not in native_groups.setdefault(key, {}), 'DUPLICATE_NATIVE_PARAMETER')
            native_groups[key][field] = row
    _require(set(native_groups) == {e['event_prefix'] for e in policy['events']}, 'NATIVE_EVENT_SET')
    patches, events = [], []
    for index, event in enumerate(policy['events']):
        key = event['event_prefix']; actual = native_groups[key]
        for field, wanted, expected_id, conversion in (
                ('Dose', event['original_dose_mg'], event['dose_parameter_id'], c.MASS_MG),
                ('Start time', event['original_start_time_min'], event['start_parameter_id'], c.TIME_MIN)):
            row = actual[field]
            _require(row['id'] == expected_id and row['unit'] in conversion and type(row['value']) in (int, float) and
                math.isclose(row['value'] * conversion[row['unit']], wanted, rel_tol=1e-12, abs_tol=1e-12), 'NATIVE_ORIGINAL_DOSE_TIME')
        view = deepcopy(source_metadata)
        view['administrations'] = [a for a in view['administrations'] if c._normalized_event(a['event_prefix']) == key]
        view['event_parameters'] = [r for r in view['event_parameters'] if c._normalized_event(c._event_prefix(r['path'])[0]) == key]
        single = {'parameters': [r for r in inventory['parameters'] if not c.native_pkml.ADMINISTRATION.fullmatch(r.get('path', '')) or
            c._normalized_event(c._event_prefix(r['path'])[0]) == key]}
        target = {**deepcopy(clinical_contract), 'dose_mg': event['target_dose_mg'], 'administration_times_min': [event['target_start_time_min']]}
        new, rows = c.build_patches(single, target, view, native_compound_name=PARENT)
        _require(len(rows) == 1 and len(new) == 2, 'EXACT_TWO_INPUT_PARAMETERS')
        rows[0].update(target_administration_index=0, absorption_input_index=index,
            disposition='original_split_absorption_input_of_single_clinical_dose', original_absorption_fraction=event['fraction'])
        patches.extend(new); events.extend(rows)
    _require(math.isclose(sum(r['dose_mg'] for r in events), clinical_contract['dose_mg'], rel_tol=0, abs_tol=1e-12), 'TOTAL_DOSE_NOT_PRESERVED')
    return patches, events


def verify_native_readback(native, policy):
    from . import complete_candidate as c
    rows = native.get('parameter_changes', [])
    changes = {c._normalized_event(r['path']): r for r in rows}
    _require(len(rows) == len(changes) == 4, 'READBACK_EVENT_COUNT')
    checked = []
    for e in policy['events']:
        for field, value, identifier, units in [('Dose', e['target_dose_mg'], e['dose_parameter_id'], c.MASS_MG),
                ('Start time', e['target_start_time_min'], e['start_parameter_id'], c.TIME_MIN)]:
            row = changes.get(e['event_prefix'] + '|ProtocolSchemaItem|' + field, {})
            after = row.get('after', {})
            _require(row.get('id') == identifier and after.get('unit') in units and
                type(after.get('value')) in (int, float) and math.isclose(after['value'] * units[after['unit']], value,
                rel_tol=1e-12, abs_tol=1e-12), 'READBACK_DOSE_OR_LAG')
            checked.append({'event_prefix': e['event_prefix'], 'field': field, 'value': value,
                            'native_id': identifier, 'verified': True})
    return {'accepted': True, 'policy_sha256': _digest(policy), 'clinical_administration_count': 1,
        'internal_absorption_event_count': 2, 'total_dose_mg': 40., 'lag_minutes': 138.,
        'events': checked, 'clinical_outcomes_accessed': False}
