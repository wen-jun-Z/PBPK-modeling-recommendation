#!/usr/bin/env python3
"""Prepare evidence-only native models for distribution-axis engineering checks.

No clinical reference concentrations are opened here. The preset interventions
check implementation and cannot establish clinical accuracy or choose a model.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, run_evidence_blueprint
from pbpk_kb.blueprint_policy.distribution_axes import run_distribution_axes

RUN = ROOT / 'experiments/blueprint_design/architecture_20260930'
BASE = ROOT / 'experiments/blueprint_design/evidence_policy_20260929/protocol.json'
DRUGS = ('Cefepime', 'Remdesivir')
CANDIDATES = [
    {'partition_scale': 1., 'bidirectional_exchange_scale': 1.},
    {'partition_scale': .5, 'bidirectional_exchange_scale': 1.},
    {'partition_scale': 2., 'bidirectional_exchange_scale': 1.},
    {'partition_scale': 1., 'bidirectional_exchange_scale': .1},
    {'partition_scale': 1., 'bidirectional_exchange_scale': 10.},
]


def read(path):
    return json.loads(inside(path, ROOT, exists=True).read_text())


def ref(path):
    path = inside(path, ROOT, exists=True)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def checked(record):
    path = inside(record['path'], ROOT, exists=True)
    if sha256(path) != record['sha256']:
        raise ValueError('Changed source: ' + record['path'])
    return read(path)


def concentration_unit(value):
    """Accept the equivalent native micro symbols without changing scale."""
    return value.replace('\u00b5', 'u').replace('\u03bc', 'u').lower()


def prepare():
    source = read(BASE)
    items = []
    for drug in DRUGS:
        case = next(c for c in source['cases'] if c['drug'] == drug)
        unit = next(u for u in source['units'] if u['case_id'] == case['case_id']
            and u['scenario'] == 'available' and u['sampling'] == 'full' and u['method'] == 'evidence_policy')
        config = source['computations'][unit['computation_id']]
        # Only the uncalibrated development REQUEST, no fitted mapping/curve.
        request = checked(config['request'])
        items.append({'drug': drug, 'request': deepcopy(config['request']),
            'compound_id': request['compound_id'], 'partition_method': request['partition_method']})
    protocol = {'version': 'distribution-axes-engineering-1', 'cases': items,
        'candidates': CANDIDATES, 'native_threads': 2,
        'hypothesis': 'Partition extent and bidirectional exchange speed can be changed independently in the original human physiology.',
        'clinical_reference_outcomes_used': False, 'parameters_optimized': False,
        'clinical_accuracy_evaluated': False, 'native_source': 'Newly assembled from original drug evidence; never reuse a fitted drug model.',
        'protected': ['evidence_logP', 'fraction_unbound', 'pKa', 'organ_clearance_inputs', 'organ_volumes', 'blood_flows'],
        'scope': 'Adult single IV; 16 nonmucosal interstitial-cellular neighborhoods; GI mucosa remain at their original values.',
        'limitations': ['Shared partition scaling is not a unique physiological parameter or a measured Vss.',
            'Unchanged organ-clearance inputs do not establish unchanged observed systemic CL.',
            'Perturbation sensitivity does not prove parameter identifiability or clinical improvement.'],
        'source_protocol': ref(BASE)}
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'protocol.json').exists():
        if read(RUN / 'protocol.json') != protocol:
            raise ValueError('Keep the existing protocol; do not overwrite different inputs')
    else:
        save(RUN / 'protocol.json', protocol)
    if not (RUN / 'state.json').exists():
        save(RUN / 'state.json', {'status': 'prepared', 'protocol': ref(RUN / 'protocol.json'), 'base_models': {}})
    return protocol


def build_bases():
    protocol = prepare(); state = read(RUN / 'state.json')
    for case in protocol['cases']:
        drug = case['drug']
        if drug in state['base_models']:
            checked(state['base_models'][drug]['execution_manifest'])
            continue
        folder = RUN / 'work' / drug.lower() / 'base'
        request = checked(case['request'])
        if folder.exists():
            result = read(folder / 'result.json')
            if read(folder / 'request.json') != EvidenceBlueprintRequest.model_validate(request).model_dump(mode='json'):
                raise ValueError('Interrupted base request differs')
        else:
            result = run_evidence_blueprint(request=request, output_dir=folder, project_root=ROOT, execute=True)
        if result.get('status') != 'predicted':
            state.update(status='blocked', reason=f'{drug}: {result.get("status")}')
            save(RUN / 'state.json', state)
            raise RuntimeError(state['reason'])
        state['base_models'][drug] = {'execution_manifest': ref(result['execution_manifest']),
            'result': ref(folder / 'result.json'), 'prediction': ref(folder / 'prediction_locked.json')}
        state['status'] = 'building_bases'
        save(RUN / 'state.json', state)
        print(json.dumps({'drug': drug, 'status': 'fresh_evidence_model_verified'}, ensure_ascii=False), flush=True)
    state['status'] = 'base_models_ready'; save(RUN / 'state.json', state)
    return {'status': state['status'], 'models': len(state['base_models'])}


def run():
    protocol = prepare(); state = read(RUN / 'state.json')
    checked(state['protocol'])
    state.setdefault('distribution_runs', {})
    for case in protocol['cases']:
        drug = case['drug']
        source = state['base_models'][drug]['execution_manifest']
        checked(source)
        if drug in state['distribution_runs']:
            result = checked(state['distribution_runs'][drug])
        else:
            folder = RUN / 'work' / drug.lower() / 'distribution'
            if folder.exists():
                raise ValueError(f'Interrupted output requires inspection: {folder}')
            state.update(status='running_engineering_check', current_drug=drug)
            save(RUN / 'state.json', state)
            result = run_distribution_axes(execution_manifest_path=ROOT / source['path'],
                source_manifest_sha256=source['sha256'], candidates=protocol['candidates'],
                output_dir=folder, project_root=ROOT, timeout_seconds=180, cpu_threads=2)
            state['distribution_runs'][drug] = ref(folder / 'distribution_manifest.json')
        if result.get('status') != 'completed':
            state.update(status='blocked', reason=result.get('error', 'Native check failed'))
            save(RUN / 'state.json', state)
            raise RuntimeError(state['reason'])
        save(RUN / 'state.json', state)
        print(json.dumps({'drug': drug, 'native_candidates': len(result['runs']),
            'status': 'native_readback_verified'}), flush=True)
    state.update(status='native_runs_complete'); state.pop('current_drug', None)
    save(RUN / 'state.json', state)
    return verify()


def verify():
    protocol = prepare(); state = read(RUN / 'state.json')
    checked(state['protocol'])
    summaries = []
    for case in protocol['cases']:
        drug = case['drug']
        native = checked(state['distribution_runs'][drug])
        baseline = checked(state['base_models'][drug]['prediction'])
        if native['status'] != 'completed' or native['candidates'] != protocol['candidates']:
            raise ValueError('Incomplete or different native candidates')
        for path, digest in native['source_pins'].items():
            if sha256(inside(path, ROOT, exists=True)) != digest:
                raise ValueError('Changed native source or code: ' + path)
        identity = native['runs'][0]['curve']
        if (identity['time_min'] != baseline['time_min']
                or concentration_unit(identity['unit']) != concentration_unit(baseline['unit'])
                or len(identity['concentration']) != len(baseline['concentration'])):
            raise ValueError('Identity run changed time grid or units')
        # Existing native CSV export rounds to 8 significant digits.
        identity_matches = all(math.isclose(a, b, rel_tol=2e-6, abs_tol=1e-9)
            for a, b in zip(identity['concentration'], baseline['concentration']))
        if not identity_matches:
            raise ValueError('Identity intervention does not reproduce the native base')
        changes = []
        for item in native['runs']:
            artifact = ROOT / state['distribution_runs'][drug]['path']
            artifact = artifact.parent / item['pkml_file']
            if sha256(artifact) != item['pkml_sha256']:
                raise ValueError('Modified candidate artifact')
            curve = item['curve']
            if (curve['time_min'] != identity['time_min'] or curve['unit'] != identity['unit']
                    or len(curve['concentration']) != len(identity['concentration'])):
                raise ValueError('Candidate time grid or units differ')
            difference = max(abs(a - b) for a, b in zip(curve['concentration'], identity['concentration']))
            changes.append({'scales': item['scales'], 'curve_points': len(curve['time_min']),
                'changed_paths_read_back': len(item['changed']),
                'protected_paths_read_back': len(item['protected']),
                'max_absolute_concentration_change_vs_identity': difference,
                'concentration_unit': curve['unit']})
        summaries.append({'drug': drug, 'identity_matches_evidence_base': identity_matches,
            'candidates': changes})
    report = {'status': 'engineering_checks_passed', 'clinical_accuracy_evaluated': False,
        'parameters_optimized': False, 'observations_used': False, 'mass_balance_checked': False,
        'verification': summaries, 'protocol': state['protocol'],
        'distribution_runs': state['distribution_runs'],
        'claim_boundary': 'Implementation and native response only; no evidence of clinical improvement or identified parameters.'}
    save(RUN / 'verification.json', report)
    state.update(status=report['status'], verification=ref(RUN / 'verification.json'))
    save(RUN / 'state.json', state)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'build-bases', 'run', 'verify', 'status'))
    command = parser.parse_args().command
    result = {'prepare': prepare, 'build-bases': build_bases, 'run': run, 'verify': verify,
        'status': lambda: read(RUN / 'state.json')}[command]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
