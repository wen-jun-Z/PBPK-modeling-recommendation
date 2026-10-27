#!/usr/bin/env python3
"""Authenticate all frozen Furosemide method selections, then seal endpoint scores.

The immutable original scorer supplies the unchanged endpoint arithmetic. This
wrapper adds the complete 57-job/source barrier and a metric-free public receipt.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'runtime/blueprint_furosemide_seed_fix_20260920'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


PREDICTOR = module('furosemide_methods_prediction_contract',
                   ROOT/'scripts/predict_furosemide_methods_external.py')
BASE = module('furosemide_immutable_endpoint_arithmetic',
              STAGE/'scripts/score_blueprint_locked_endpoints.py')


def validate_job_denominator(selections):
    expected = {(block, method, seed, budget, dimension)
        for block, (budget, dimension, methods) in PREDICTOR.BLOCKS.items()
        for method, seed in PREDICTOR.expected_jobs(methods)}
    actual = set()
    for row in selections:
        block, method, seed = row['block_id'], row['method'], row['seed']
        key = (block, method, seed, row['solver_budget'], row['exact_free_parameter_count'])
        if (type(seed) is not int or type(row['solver_budget']) is not int or
            (row['exact_free_parameter_count'] is not None and type(row['exact_free_parameter_count']) is not int) or
            row['selection_id'] != f'{block}__{method}_seed_{seed}' or key in actual):
            raise ValueError('Invalid or duplicate frozen method-job identity')
        actual.add(key)
    if actual != expected or len(selections) != 57:
        raise ValueError('All 57 registered block/method/seed/budget selections are required')


def validate_native_csv(row, lock_path):
    base = lock_path.parent/'native'/(row['model_key']+'_'+row['sex'])
    path = PREDICTOR.local(row['native_csv_path'])
    if not path.is_relative_to(base.resolve()) or path.stat().st_size > 64*1024**2:
        raise ValueError('Native curve path escaped its registered prediction model directory')
    if PREDICTOR.file_sha(path) != row['native_csv_sha256']:
        raise ValueError('Native prediction CSV checksum changed')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        native = list(csv.DictReader(stream))
    column = 'Organism|PeripheralVenousBlood|Furosemide|Plasma (Peripheral Venous Blood) [µmol/l]'
    times = [float(r['Time [min]']) for r in native]
    values = [float(r[column]) for r in native]
    PREDICTOR.validate_curve(times, values)
    if row['time_min'] != times or row['concentration'] != values:
        raise ValueError('Locked concentration arrays differ from their unmodified native CSV')


def validate_locked_inputs(lock_path, design_path):
    """Public, outcome-free validation; returns the immutable scorer's dict."""
    lock_path, design_path = PREDICTOR.local(lock_path), PREDICTOR.local(design_path)
    checked = BASE.validate_locked_inputs(lock_path, design_path)
    lock, protocol = checked['lock'], checked['protocol']
    PREDICTOR.validate_design(checked['design'])
    if (lock.get('method_job_count') != 57 or type(lock.get('method_job_count')) is not int or
        lock.get('strict_main_eligible') is not False or protocol.get('strict_main_eligible') is not False or
        lock.get('qualification_tier') != 'transportability_sensitivity' or
        protocol.get('qualification_tier') != 'transportability_sensitivity' or
        not protocol.get('qualification_reasons') or protocol.get('external_observations_read') is not False):
        raise ValueError('The complete denominator and approximate validation tier must remain frozen')
    validate_job_denominator(checked['selections'])
    # Authenticate the protocol registry by independently rebuilding the actual
    # completed source suite, not by trusting arbitrary hash targets from a lock.
    frozen = {}
    package_path = PREDICTOR.local(protocol['input_package_path'])
    case = PREDICTOR.read(package_path, frozen)['case']
    PREDICTOR.read(design_path, frozen)
    suite = PREDICTOR.local(protocol['suite_path'])
    selections, models = PREDICTOR.load_selections(suite, case, package_path, frozen)
    PREDICTOR.freeze_prediction_sources(frozen)
    if (protocol['input_package_sha256'] != PREDICTOR.file_sha(package_path) or
        protocol['selections'] != selections or protocol['models'] != models or
        protocol['frozen_input_sha256'] != frozen):
        raise ValueError('Frozen selections, candidate contracts or complete source registry changed')
    PREDICTOR.verify_frozen(frozen)
    policy = PREDICTOR.metadata_exclusion_policy(checked['design'], design_path)
    if protocol['evidence_exclusion'] != policy:
        raise ValueError('External-publication exclusion policy changed')
    predictions = json.loads(BASE.verified(lock['predictions'][0]))
    if (set(protocol['models']) != {s['model_key'] for s in selections if s['model_key'] is not None} or
        lock.get('physical_native_predictions') != len(predictions['models']) or
        len(predictions['models']) != 2*len(models)):
        raise ValueError('Physical predictions do not cover every registered model and sex')
    for row in predictions['models']:
        if row['sex'] not in {'male', 'female'}:
            raise ValueError('Native sex labels must remain canonical')
        validate_native_csv(row, lock_path)
    return checked


def score_and_seal(lock_path, design_path, output):
    output = PREDICTOR.local(output, exists=False)
    if output.exists():
        raise FileExistsError('Evaluation output is immutable; refused before opening sealed outcomes')
    checked = validate_locked_inputs(lock_path, design_path)
    # Reserve the output before unsealing; a concurrent invocation cannot reopen
    # outcomes and overwrite a completed evaluation.
    output.mkdir(parents=True, exist_ok=False)
    sealed = output/'sealed'
    sealed.mkdir(mode=0o700)
    PREDICTOR.verify_frozen(checked['protocol']['frozen_input_sha256'])
    result = BASE.score_locked(lock_path, design_path)
    if result['method_job_denominator'] != 57 or len(result['results']) != 57:
        raise ValueError('Immutable endpoint scorer did not preserve all method rows')
    payload = (json.dumps(result, indent=2, allow_nan=False)+'\n').encode()
    if len(payload) > 8*1024**2:
        raise ValueError('Sealed endpoint output exceeds its bounded size')
    target = sealed/'endpoint_scores.json'
    with target.open('xb') as handle:
        os.chmod(target, 0o600)
        handle.write(payload)
    counts = {}
    for name in ('Cmax', 'AUC_infinity'):
        endpoints = [e for r in result['results'] for e in r['endpoints'] if e['endpoint'] == name]
        if len(endpoints) != 57 or any(type(e['evaluable']) is not bool for e in endpoints):
            raise ValueError('Endpoint denominator or evaluability contract changed')
        evaluable = sum(e['evaluable'] for e in endpoints)
        counts[name] = {'evaluable':evaluable, 'failed_retained':57-evaluable, 'total':57}
    receipt = {'status':'scored_and_sealed', 'method_jobs_expected':57, 'method_rows_scored':57,
        'independent_drugs':1, 'independent_studies':1, 'strict_main_eligible':False,
        'qualification_tier':'transportability_sensitivity',
        'qualification_reasons':checked['protocol']['qualification_reasons'],
        'metrics_exposed_to_method_development':False, 'lock_sha256':checked['lock_sha256'],
        'endpoint_counts':counts, 'scorer_sha256':PREDICTOR.file_sha(Path(__file__)),
        'immutable_endpoint_scorer_sha256':PREDICTOR.file_sha(STAGE/'scripts/score_blueprint_locked_endpoints.py')}
    PREDICTOR.save(output/'scoring_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('lock', 'design', 'output'):
        parser.add_argument('--'+flag, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    receipt = score_and_seal(PREDICTOR.local(args.lock), PREDICTOR.local(args.design), args.output)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
