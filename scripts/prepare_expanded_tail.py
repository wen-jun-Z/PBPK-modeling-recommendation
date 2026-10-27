#!/usr/bin/env python3
"""Prepare one evidence/RF input contract for the separate twenty-drug run.

Reuses the existing twenty-drug-excluded RF predictions and weights. This
module never fits a model, reconstructs a clearance from a clinical curve,
decodes target observations, or executes a native simulation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.evidence_blueprint import assemble_evidence_blueprint
from pbpk_kb.blueprints.pk_parameter_learning import normalize_structure

VERSION = 'expanded-tail-unified-inputs-1'
DEFAULT = ROOT / 'experiments/blueprint_design/expanded_tail_20260930'
ORIGINAL = 'experiments/blueprint_design/robust_structure_tail_20260930/protocol.json'
INVENTORY = 'experiments/blueprint_design/optimization_20260929/cohort_inventory.json'
RF_FIELDS = {'fraction_unbound': ('human_fup', 'fraction'),
             'total_clearance': ('human_CL_mL_min_kg', 'ml/min/kg')}
CL_SOURCE_SCOPE = {
    'Dexmedetomidine': 'regulatory_summary_not_asserted_independent_of_development',
    'Moxifloxacin': 'directly_reported_development_study_clearance_not_curve_recalculation',
    'Remdesivir': 'separate_mass_balance_study',
    'Alprazolam': 'independent_third_study',
    'Fluconazole': 'independent_third_study',
    'Furosemide': 'independent_third_study',
}
ORIGINAL_SENSITIVITY_FLAGS = {
    'Dexmedetomidine': ['healthy_to_surgical_population_transfer', 'sparse_digitized_development_curve'],
    'Midazolam': ['patent_target_source', 'representative_demographics_unconfirmed'],
    'Nemonoxacin': ['participant_nonoverlap_unconfirmed', 'registry_identifier_discrepancy'],
}


class Sources:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.pins = {}

    def path(self, path):
        raw = Path(path)
        p = (raw if raw.is_absolute() else self.root / raw).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError('Input or output escaped the project')
        return p

    def pin(self, path, expected=None):
        p = self.path(path)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        name = str(p.relative_to(self.root))
        if expected is not None and digest != expected:
            raise ValueError('Source hash changed: ' + name)
        if name in self.pins and self.pins[name] != digest:
            raise ValueError('Source changed during preparation: ' + name)
        self.pins[name] = digest
        return {'path': name, 'sha256': digest, 'location': 'Frozen source or unified construction input'}

    def read(self, source):
        reference = self.pin(source['path'], source['sha256']) if isinstance(source, dict) else self.pin(source)
        return json.loads(self.path(reference['path']).read_text())

    def refs(self, obj):
        if isinstance(obj, dict):
            if isinstance(obj.get('path'), str) and isinstance(obj.get('sha256'), str):
                self.pin(obj['path'], obj['sha256'])
            for value in obj.values():
                self.refs(value)
        elif isinstance(obj, list):
            for value in obj:
                self.refs(value)

    def unchanged(self):
        for name, digest in list(self.pins.items()):
            self.pin(name, digest)


def save_new(store, run, path, obj):
    """Idempotent append inside the dedicated inputs directory; never replace."""
    run = store.path(run)
    path = store.path(path)
    if not path.is_relative_to(run / 'inputs') or path.is_symlink():
        raise ValueError('Prepared artifacts belong only to this run inputs')
    payload = (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Refusing to replace a changed prepared input: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(payload)
    return store.pin(path)


def audit_split(split, predictions, structures):
    """Recheck recorded membership and current standardized twenty identities."""
    external = split['external_exclusions']
    if set(structures) != set(external) or set(predictions) != set(external) or len(external) != 20:
        raise ValueError('The existing RF must exclude exactly the same twenty drugs')
    rows = {r['index']: r for r in split['rows']}
    if len(rows) != len(split['rows']):
        raise ValueError('Duplicate parameter-table row')
    memberships = {}
    for kind in ('train', 'test'):
        indices = split[kind + '_indices']
        if len(indices) != len(set(indices)) or len(indices) != split[kind + '_count']:
            raise ValueError('RF denominator differs from membership')
        selected = [rows[i] for i in indices]
        if any(r['status'] != kind for r in selected):
            raise ValueError('RF membership differs from row disposition')
        if set(indices) != {i for i, r in rows.items() if r['status'] == kind}:
            raise ValueError('RF membership omitted a declared row')
        memberships[kind] = selected
    if set(split['train_indices']) & set(split['test_indices']):
        raise ValueError('Train/test row overlap')
    if {r['scaffold_group'] for r in memberships['train']} & {r['scaffold_group'] for r in memberships['test']}:
        raise ValueError('Train/test scaffold overlap')
    records = []
    for name, smiles in structures.items():
        identity = normalize_structure(smiles)
        exclusion = external[name]
        prediction = predictions[name]
        for key in ('identity_smiles', 'scaffold_group', 'scaffold_smiles'):
            if identity[key] != exclusion[key] or identity[key] != prediction[key]:
                raise ValueError('Structure disagrees with RF exclusion: ' + name)
        if prediction['training_structure_overlap'] is not False or prediction['training_scaffold_overlap'] is not False:
            raise ValueError('RF prediction records overlap')
        counts = {kind + '_' + key + '_overlap': sum(r[key] == identity[key] for r in selected)
                  for kind, selected in memberships.items() for key in ('identity_smiles', 'scaffold_group')}
        if any(counts.values()):
            raise ValueError('Clinical structure/scaffold overlaps RF data: ' + name)
        for endpoint, _ in RF_FIELDS.values():
            value = prediction['predictions'][endpoint]['random_forest']
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Missing or invalid RF prediction: ' + name)
        if prediction['predictions']['human_fup']['random_forest'] > 1:
            raise ValueError('RF fraction unbound exceeds one')
        records.append({'drug': name, **identity, **counts})
    dispositions = dict(Counter(r['status'] for r in rows.values()))
    if dispositions != {'excluded': 111, 'train': 1072, 'test': 100}:
        raise ValueError('The locked twenty-drug RF denominator changed')
    for endpoint, audit in split['label_audit'].items():
        for kind in ('train', 'test'):
            if not set(audit[kind + '_indices']) <= set(split[kind + '_indices']):
                raise ValueError('Endpoint labels escaped their RF split')
    return {'records': records, 'row_dispositions': dispositions, 'unique_drugs': len(records),
            'weights_reused': True, 'weights_deserialized': False, 'rf_refitted': False,
            'clinical_curves_used_for_RF_training': False,
            'interpretation': 'Verifies frozen provenance and memberships; does not claim RF predictive accuracy.'}


def refresh_evidence(rows, drug, compound_id, prediction_ref, prediction, *, fallback=False):
    """Replace RF values only; retain approved empirical evidence verbatim."""
    if len({r['parameter'] for r in rows}) != len(rows):
        raise ValueError('Conflicting primary evidence must be resolved before preparation')
    selected = deepcopy(rows)
    by_name = {r['parameter']: r for r in selected}
    cl = by_name.get('total_clearance')
    if cl and cl['basis'] != 'predicted' and drug not in CL_SOURCE_SCOPE:
        raise ValueError('An unreviewed clinical CL prior cannot enter unified inputs: ' + drug)
    actions = []
    for parameter, (endpoint, unit) in RF_FIELDS.items():
        original = by_name.get(parameter)
        replace = fallback or original is None or original['basis'] == 'predicted'
        if not replace:
            if original['basis'] not in {'measured', 'derived'}:
                raise ValueError('Non-RF input is not approved empirical evidence')
            actions.append({'parameter': parameter, 'action': 'retained_empirical',
                'source_scope': CL_SOURCE_SCOPE[drug] if parameter == 'total_clearance' else 'published_binding_input'})
            continue
        item = {'parameter': parameter, 'compound_id': compound_id,
            'value': prediction['predictions'][endpoint]['random_forest'], 'unit': unit,
            'basis': 'predicted', 'source': deepcopy(prediction_ref),
            'context': {'species': 'human', 'matrix': 'plasma', 'route': 'intravenous'},
            'prediction_method': {'name': 'random_forest', 'version': 'expanded20-20260929',
                'training_overlap': False, 'applicability': {'target_structure_and_scaffold_excluded': True,
                    'human_IV_parameter_model': True}},
            'notes': 'Existing twenty-drug-excluded RF; no clinical development or target curve labels used.'}
        if original is not None:
            selected[selected.index(original)] = item
        else:
            selected.append(item)
        actions.append({'parameter': parameter, 'action': 'rf_fallback_view' if fallback else 'refreshed_rf'})
    protected = [r for r in rows if r['parameter'] == 'renal_clearance' or
                 (not fallback and r['parameter'] in RF_FIELDS and r['basis'] != 'predicted')]
    for row in protected:
        if next((r for r in selected if r['parameter'] == row['parameter']), None) != row:
            raise ValueError('Empirical evidence changed')
    by_name = {r['parameter']: r for r in selected}
    renal = by_name.get('renal_clearance'); total = by_name['total_clearance']
    if renal and (renal['unit'] != total['unit'] or not 0 <= renal['value'] < total['value']):
        raise ValueError('RF total CL conflicts with retained renal evidence; no silent adjustment')
    return selected, actions


def prepare_inputs(run_dir=DEFAULT, extension_cases_ref=None, *, root=ROOT):
    store = Sources(root); run = store.path(run_dir)
    if not run.is_dir() or run.parent != store.root / 'experiments/blueprint_design' or not run.name.startswith('expanded_tail_'):
        raise ValueError('Create the dedicated expanded_tail_ batch before preparing inputs')
    if (run / 'protocol.json').exists() or (run / 'state.json').exists():
        raise ValueError('Inputs must freeze before experiment protocol/state creation')
    original = store.read(ORIGINAL)
    original_state = store.read(str(Path(ORIGINAL).parent / 'state.json'))
    if original_state['status'] != 'complete' or store.read(original_state['protocol']) != original:
        raise ValueError('The original cohort protocol is not locked by its completed run')
    for name, digest in original['source_pins'].items():
        store.pin(name, digest)
    extension_ref = extension_cases_ref or store.pin(run / 'inputs/extension_cases.json')
    extension = store.read(extension_ref)
    inventory = store.read(INVENTORY)
    for name, digest in inventory['source_pins'].items():
        store.pin(name, digest)
    for name, digest in extension.get('source_pins', {}).items():
        store.pin(name, digest)
    prior_refs = inventory['priors']; store.refs(prior_refs)
    split = store.read(prior_refs['split']); predictions_doc = store.read(prior_refs['predictions'])
    predictions = predictions_doc['external_predictions']
    if predictions_doc['clinical_curves_used_for_RF_training'] is not False or predictions_doc['all_twenty_drug_structures_and_scaffolds_excluded'] is not True:
        raise ValueError('RF prediction provenance is incomplete')
    source = split['source']; wheel_ref = store.pin(source['source_path'], source['source_sha256'])
    store.refs(predictions_doc['source'])
    if predictions_doc['source']['sha256'] != wheel_ref['sha256']:
        raise ValueError('Predictions and split name different RF training archives')
    with zipfile.ZipFile(store.path(wheel_ref['path'])) as archive:
        if hashlib.sha256(archive.read(source['csv_member'])).hexdigest() != source['csv_sha256']:
            raise ValueError('RF training CSV differs from the locked wheel')
    store.pin('src/pbpk_kb/blueprints/pk_parameter_learning.py', split['module_sha256'])
    structures = {}
    inventory_by_drug = {r['drug']: r for r in inventory['records']}
    for name, row in inventory_by_drug.items():
        physical = store.read(row['physchem']) if row['cohort_role'] == 'existing_ten_development' else None
        structures[name] = physical['identity']['isomeric_smiles'] if physical else row['identity']['isomeric_smiles']
        if not physical:
            store.refs(row['identity'])
    rf_audit = audit_split(split, predictions, structures)
    cases = [dict(deepcopy(c), cohort='original') for c in original['cases']] + [dict(deepcopy(c), cohort='extension') for c in extension['cases']]
    for case in cases:
        if case['cohort'] == 'original':
            case['reference_kind'] = 'original_reported'
            case['sensitivity_flags'] = list(dict.fromkeys([
                *case.get('sensitivity_flags', []), *ORIGINAL_SENSITIVITY_FLAGS.get(case['drug'], [])]))
    if (len(original['cases']) != 10 or len(extension['cases']) != 10 or
            len({c['case_id'] for c in cases}) != 20 or len({c['compound_id'] for c in cases}) != 20 or
            {c['drug'] for c in cases} != set(predictions)):
        raise ValueError('Unified input requires the same twenty unique cases and drugs')
    # Original RF uses drug-name keys. This transparent view only renames keys
    # for the existing audit interface; all split rows and weights are reused.
    split_view = deepcopy(split)
    split_view['external_exclusions'] = {c['case_id']: split['external_exclusions'][c['drug']] for c in cases}
    split_view['original_split_source'] = deepcopy(prior_refs['split'])
    split_view['case_to_original_exclusion_key'] = {c['case_id']: c['drug'] for c in cases}
    split_ref = save_new(store, run, run / 'inputs/unified/parameter_split_cases.json', split_view)
    audit = []
    for case in cases:
        drug, cid = case['drug'], case['case_id']; row = inventory_by_drug[drug]
        if case['compound_id'] != row['compound_id']:
            raise ValueError('Clinical compound identity differs from RF inventory')
        store.refs(case)
        physical = store.read(case['physchem'])
        if normalize_structure(physical['identity']['isomeric_smiles']) != normalize_structure(structures[drug]):
            raise ValueError('Construction structure differs from RF exclusion')
        original_requests = case['requests']['evidence']
        requests = {role: store.read(original_requests[role]) for role in ('development', 'target')}
        for request in requests.values():
            store.refs(request)
            ledger = store.read(request['evidence_ledger'])
            if ledger['evidence'] != request['evidence']:
                raise ValueError('Request evidence differs from its ledger')
            if request['compound_id'] != case['compound_id']:
                raise ValueError('Request compound identity changed')
        if requests['development']['evidence'] != requests['target']['evidence']:
            raise ValueError('Development and target require the same construction evidence')
        prediction = predictions[drug]; values = prediction['predictions']; folder = run / 'inputs/unified' / cid
        packaged = {'compound_id': case['compound_id'], 'method': 'random_forest',
            'predictions': {'clearance_ml_min_kg': values['human_CL_mL_min_kg']['random_forest'],
                'fraction_unbound': values['human_fup']['random_forest'], 'vss_l_kg': values['human_VDss_L_kg']['random_forest']},
            'all_parameter_predictions': deepcopy(prediction), 'training_overlap': False,
            'same_scaffold_training_overlap': False,
            'provenance': {'split': split_ref, 'original_split': prior_refs['split'], 'model': prior_refs['model'],
                'wheel': wheel_ref, 'existing_predictions': prior_refs['predictions'], 'original_prediction_key': drug},
            'no_clinical_outcomes_used': True, 'predicted_vss_used_for_pbpk_construction': False,
            'rf_refitted': False}
        prediction_ref = save_new(store, run, folder / 'predicted_pk.json', packaged)
        refreshed = {}; case_audit = {'case_id': cid, 'drug': drug, 'cohort': case['cohort'], 'views': {}}
        for view in ('evidence', 'rf'):
            selected, actions = refresh_evidence(requests['development']['evidence'], drug,
                case['compound_id'], prediction_ref, prediction, fallback=view == 'rf')
            ledger_ref = save_new(store, run, folder / (view + '_ledger.json'), {'evidence': selected})
            refreshed[view] = {}
            for role, request in requests.items():
                output = deepcopy(request); output['evidence'] = selected; output['evidence_ledger'] = ledger_ref
                result = assemble_evidence_blueprint(output, store.root)
                if result['status'] != 'assembled':
                    raise ValueError('Unified construction preflight rejected ' + drug + '/' + view + '/' + role + ': ' + str(result))
                refreshed[view][role] = save_new(store, run, folder / (view + '_' + role + '.json'), output)
            case_audit['views'][view] = actions
        case['requests'] = refreshed
        case['unified_input_source_requests'] = deepcopy(original_requests)
        case['unified_rf_prediction'] = prediction_ref
        case_audit['retained_evidence'] = [deepcopy(e) for e in requests['development']['evidence']
            if e['parameter'] in ('fraction_unbound', 'total_clearance', 'renal_clearance') and e['basis'] != 'predicted']
        audit.append(case_audit)
    store.pin('scripts/prepare_expanded_tail.py')
    store.pin('tests/test_blueprints/test_expanded_tail_inputs.py')
    for name, digest in original['paper_guard'].items():
        store.pin(name, digest)
    coverage = {'cases': len(cases), 'cohorts': dict(Counter(c['cohort'] for c in cases)),
        'study_clusters': len({c['cluster_id'] for c in cases}),
        'source_sensitivity_flagged_drugs': [c['drug'] for c in cases if c.get('sensitivity_flags')],
        'preflight_request_count': 4 * len(cases), 'rf_prediction_values_verified': 2 * len(cases),
        'retained_empirical_inputs': dict(Counter(e['parameter'] for r in audit for e in r['retained_evidence']))}
    manifest = {'version': VERSION, 'cases': cases, 'coverage': coverage, 'rf_audit': rf_audit,
        'evidence_audit': audit, 'rf_sources': prior_refs, 'source_pins': dict(store.pins),
        'paper_guard': deepcopy(original['paper_guard']), 'paper_updated': False,
        'target_observations_decoded': False, 'clinical_clearance_reconstructed': False,
        'native_simulations_executed': False, 'rf_refitted': False,
        'interpretation': 'All twenty drugs use the same RF exclusions. Empirical evidence retains its original source limitations. Previous ten-drug predictions are not reused as unified results.'}
    store.unchanged()
    save_new(store, run, run / 'inputs/unified_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = prepare_inputs(args.run_dir)
    print(json.dumps({'status': 'inputs_prepared', 'coverage': result['coverage'],
        'rf_rows': result['rf_audit']['row_dispositions'], 'source_pins': len(result['source_pins']),
        'native_simulations_executed': False}, indent=2))


if __name__ == '__main__':
    main()
