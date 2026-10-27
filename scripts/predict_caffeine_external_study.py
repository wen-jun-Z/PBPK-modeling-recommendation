#!/usr/bin/env python3
"""Freeze all completed main-method predictions before an external curator runs.

No observed external endpoint or digitized curve is read by this command.
Identical fitted models share physical simulations, not method identities.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.design_runtime import execute_snapshot
from pbpk_kb.blueprints.experiment_runtime import digest, file_sha, save
from pbpk_kb.blueprints.native_edits import apply_parameter_values, compile_candidate


def local_path(value):
    p = Path(value)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if p == ROOT or not p.is_relative_to(ROOT):
        raise ValueError('All inputs and outputs must remain in this project')
    return p


def provenance(path, location, kind='user_request'):
    checksum = file_sha(path)
    return {'provenance_id': str(uuid5(NAMESPACE_URL, str(path) + location + checksum)),
            'provenance_kind': kind, 'source_id': str(path.relative_to(ROOT)),
            'source_location': location, 'source_checksum': checksum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', required=True)
    parser.add_argument('--development', required=True)
    parser.add_argument('--input-package', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    design_path, development, package_path, output = map(local_path,
        (args.design, args.development, args.input_package, args.output))
    if output.exists():
        raise ValueError('Use a new output directory; prediction locks are immutable')
    design = json.loads(design_path.read_text())
    if (design.get('contains_observed_pk_values') is not False
            or design['study_id'] != 'caffeine_sherman2022'
            or design['source_independence']['status'] != 'verified_disjoint'):
        raise ValueError('This reviewed adapter requires the outcome-free Sherman design')
    package = json.loads(package_path.read_text())
    case = package['case']
    completion = development / 'completion.json'
    if not completion.is_file():
        raise ValueError('Development must be terminal before external prediction')
    results = json.loads((development / 'results.json').read_text())['methods']
    if len(results) != 27 or any(not r.get('selected_rebuild_verified') for r in results):
        raise ValueError('Require all 27 registered main method/seed selections')
    oral = next(s for s in case['scenarios'] if s['design_request']['requirement']['route'] == 'oral')
    selections, models = [], {}
    for result in results:
        identity = f"{result['method']}_seed_{result['seed']}"
        path = development / 'methods' / identity / 'selected_parameters.json'
        parameters = json.loads(path.read_text())
        action = result['selected_candidate_id'].rsplit('__', 1)[0]
        key = digest({'action': action, 'values': parameters['values']})[:20]
        models.setdefault(key, {'action': action, 'values': parameters['values'],
                                'source': str(path.relative_to(ROOT))})
        selections.append({'method': result['method'], 'seed': result['seed'],
                           'candidate_id': result['selected_candidate_id'], 'model_key': key,
                           'parameters_path': str(path.relative_to(ROOT)), 'parameters_sha256': file_sha(path),
                           'result_path': str((path.parent / 'result.json').relative_to(ROOT)),
                           'result_sha256': file_sha(path.parent / 'result.json')})
    output.mkdir(parents=True)
    policy = {'schema_version': 'pbpk-external-prediction-protocol-v1',
        'study_id': design['study_id'], 'design_sha256': file_sha(design_path),
        'input_package_sha256': file_sha(package_path), 'development_completion_sha256': file_sha(completion),
        'selections': selections, 'unique_models': models,
        'population_approximation': {
            'age_years': 35.5, 'sex_weights': {'MALE': 13/18, 'FEMALE': 5/18},
            'height_weight': 'locked PK-Sim generated defaults, not measured study values',
            'population': 'European_ICRP_2002 source base, ethnicity unreported in external study',
            'seed': 162446752, 'water_ml_per_kg': 3.5,
            'genotype': 'source default CYP1A2; no fitted genotype correction',
            'limitations': ['Two sex representatives do not reproduce individual variability.',
                'Pooled demographics used for both dose arms; dose-specific details unavailable.',
                'Water amount and body size are source defaults, not claimed study measurements.',
                'Published geometric means of individual maxima are only approximately comparable.']},
        'time_grid': {'start_h': 0, 'stop_h': 48, 'points_per_hour': 100},
        'endpoint_policy': {'primary': 'sex-weighted geometric mean of nominal-sampling Cmax',
            'secondary': ['dense-grid Cmax', 'fixed-window AUC0-48', 'predicted-LLOQ nominal-tlast AUC sensitivity'],
            'auc_last_equivalent': False, 'lloq_ng_per_ml': 10, 'molecular_weight_g_per_mol': 194.19},
        'evidence_access': 'none; existing development selections only; no retrieval/LLM/fit on external study',
        'cpu_threads_per_worker': 2, 'workers': 2, 'max_total_output_bytes': 2*1024**3,
        'model_selection_frozen': True, 'written_at_utc': datetime.now(timezone.utc).isoformat()}
    protocol_path = output / 'prediction_protocol.json'
    save(protocol_path, policy)
    population_source = provenance(protocol_path, 'Prespecified population approximations, no PK outcomes')
    study_source = provenance(design_path, 'Methods: oral solution, absolute dose, nominal sampling; no PK outcomes')
    jobs = [(arm, key, model, sex) for arm in design['cases'] for key, model in models.items()
            for sex in ('MALE', 'FEMALE')]

    def predict(job):
        arm, key, model, sex = job
        request = deepcopy(oral['design_request'])
        request['requirement'].update(dose=arm['dose']['value'], dose_unit='mg', route='oral', formulation='solution')
        instance = request['instance']
        label = arm['case_id'] + '_' + key + '_' + sex
        instance['formulation']['Name'] = 'FORM_SHERMAN_SOLUTION'
        request['requirement']['formulation'] = instance['formulation']['Name']
        instance['simulation_name'] = label
        instance['individual']['Name'] = 'IND_' + label
        instance['individual']['OriginData'].update(Gender=sex, Age={'Value':35.5, 'Unit':'year(s)'})
        instance['human_source'] = population_source
        instance['protocol']['Name'] = 'PROTO_' + label
        for p in instance['protocol']['Parameters']:
            if p['Name'] == 'InputDose':
                p.update(Value=arm['dose']['value'], Unit='mg')
        instance['protocol_source'] = study_source
        instance['formulation_source'] = study_source
        instance['output_schema'] = [{'Parameters': [
            {'Name':'Start time', 'Value':0., 'Unit':'h'},
            {'Name':'End time', 'Value':48., 'Unit':'h'},
            {'Name':'Resolution', 'Value':100., 'Unit':'pts/h'}]}]
        # The designer's studies field denotes observed data usable by design.
        # External dose/demographic metadata is permitted, external observations
        # are not. The external study identity/role remains in the locked protocol.
        request['studies'] = []
        source = provenance(ROOT / model['source'], 'Frozen development fitted values', 'calibration_run')
        request = apply_parameter_values(request, case['parameter_search']['parameters'], model['values'], source)
        variant = compile_candidate(request, model['action'], project_root=ROOT,
                                    parameter_specs=case['parameter_search']['parameters'])
        directory = output / 'native' / label
        manifest = execute_snapshot(snapshot=variant['snapshot'], output_dir=directory, project_root=ROOT,
                                    timeout_seconds=240, cpu_threads=2, max_output_bytes=192*1024**2)
        if not manifest['engineering_accepted'] or len(manifest['curve_outputs']) != 1:
            raise RuntimeError('External native execution failed: ' + label)
        path = directory / manifest['curve_outputs'][0]['path']
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        column = oral['target_output'] + ' [µmol/l]'
        times = [float(r['Time [min]']) for r in rows]
        values = [float(r[column]) for r in rows]
        if (len(times) != len(values) or len(times) < 2 or times[-1] < 2880
                or any(not math.isfinite(v) or v < -1e-10 for v in values)
                or any(b <= a for a, b in zip(times, times[1:]))):
            raise ValueError('Invalid native external curve')
        result = {'case_id':arm['case_id'], 'model_key':key, 'sex':sex,
            'time_min':times, 'concentration':values, 'time_unit':'min', 'unit':'µmol/l',
            'native_csv_path':str(path.relative_to(ROOT)), 'native_csv_sha256':file_sha(path),
            'execution_manifest':str((directory/'execution_manifest.json').relative_to(ROOT)),
            'request_sha256':digest(request)}
        print(json.dumps({'completed_prediction':label,'points':len(times)}), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        predictions = list(pool.map(predict, jobs))
    records = []
    for arm in design['cases']:
        path = output / (arm['case_id'] + '.json')
        save(path, {'case_id':arm['case_id'], 'models':[p for p in predictions if p['case_id']==arm['case_id']],
            'selections':selections, 'protocol_sha256':file_sha(protocol_path), 'contains_observed_pk_values':False})
        records.append({'case_id':arm['case_id'], 'path':str(path.relative_to(ROOT)), 'sha256':file_sha(path)})
    save(output/'prediction_lock.json', {'schema_version':'pbpk-external-predictions-lock-v1',
        'design_sha256':file_sha(design_path), 'locked_at_utc':datetime.now(timezone.utc).isoformat(),
        'predictions':records, 'model_selection_frozen':True,
        'protocol_path':str(protocol_path.relative_to(ROOT)), 'protocol_sha256':file_sha(protocol_path),
        'method_seed_count':len(selections), 'unique_model_count':len(models),
        'method_selection_independent_of_external_outcomes':True})
    print(json.dumps({'status':'predictions_locked','lock_path':str((output/'prediction_lock.json').relative_to(ROOT)),
                      'method_seed_count':len(selections),'physical_predictions':len(predictions)}), flush=True)


if __name__ == '__main__':
    main()
