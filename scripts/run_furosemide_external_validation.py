#!/usr/bin/env python3
"""Resume independent evaluation after the selected-contract adapter repair.

Completed method results, model parameters and the failed attempt are immutable.
Only children created here can be stopped; no method search is rerun.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))

from run_blueprint_followup import RegisteredSuite, identity
from run_rosuvastatin_validation_followup import checksum, run_child_step, verify_frozen
from watch_blueprint_suite import ProcessScope, inside, read_json, write_json

STAGE = ROOT/'runtime/blueprint_furosemide_seed_fix_20260920'
SUITE = ROOT/'experiments/blueprint_design/furosemide_mechanism_suite_20260920_seed_fix_v2'
PACKAGE = ROOT/'experiments/blueprint_design/development_inputs/mechanisms_20260920/furosemide/input_package.json'
DESIGN = ROOT/'experiments/blueprint_design/independent_cohort_20260920/active_protocols_v3/prediction_metadata/furosemide_pmc8416862_trial2_baseline.json'
PREFIX = 'blueprint_furosemide_external_contract_fix_20260920'
DEFAULT_OUTPUT = 'experiments/blueprint_design/external_validation/furosemide_2021_contract_fix_20260920'


def freeze_sources():
    scripts = ('run_furosemide_external_validation.py', 'predict_furosemide_methods_external.py',
               'score_furosemide_methods_endpoints.py', 'run_rosuvastatin_validation_followup.py',
               'run_blueprint_followup.py', 'watch_blueprint_suite.py')
    paths = [*(ROOT/'scripts'/name for name in scripts), PACKAGE, DESIGN,
             *(STAGE/'src').rglob('*.py'), *(STAGE/'scripts').glob('*.py'),
             STAGE/'source_registration.json', *(ROOT/'src/pbpk_kb/model_runtime/R').glob('*.R')]
    return {str(path.relative_to(ROOT)): checksum(path) for path in paths}


def check_previous_attempt(scope):
    registration = read_json(ROOT/'runtime/deployment/blueprint_furosemide_validation_pipeline_20260920_seed_fix_v2.json')
    status = read_json(ROOT/'runtime/deployment/blueprint_furosemide_validation_pipeline_status_seed_fix_v2.json')
    if registration['pid'] != status['pid'] or status.get('status') != 'stopped':
        raise RuntimeError('Previous pipeline must already be stopped')
    if scope.state(registration['pid'], registration['start_ticks']) is not None:
        raise RuntimeError('Previous registered pipeline is still alive')
    for step in status.get('steps', []):
        if scope.state(step['pid'], step['start_ticks']) is not None:
            raise RuntimeError('Previous registered child is still alive')
    return {'registration_sha256': checksum(ROOT/'runtime/deployment/blueprint_furosemide_validation_pipeline_20260920_seed_fix_v2.json'),
            'status_sha256': checksum(ROOT/'runtime/deployment/blueprint_furosemide_validation_pipeline_status_seed_fix_v2.json'),
            'previous_prediction_output_bytes': status['steps'][-1].get('output_bytes')}


def validate_receipt(receipt, lock):
    expected = {'status': 'scored_and_sealed', 'method_jobs_expected': 57, 'method_rows_scored': 57,
                'independent_drugs': 1, 'independent_studies': 1, 'strict_main_eligible': False,
                'metrics_exposed_to_method_development': False, 'lock_sha256': checksum(lock)}
    if any(type(receipt.get(key)) is not type(value) or receipt.get(key) != value
           for key, value in expected.items()):
        raise RuntimeError('Scoring must preserve the full frozen denominator and scientific limitations')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    scope = ProcessScope()
    owner = scope.state(os.getpid())
    if owner is None or not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Authenticated Linux project processes and pidfd support are required')
    suite = RegisteredSuite(SUITE, scope)
    if not suite.sample():
        raise RuntimeError('All 57 methods must finish and release their registered children')
    prior = check_previous_attempt(scope)
    frozen = freeze_sources()
    if args.preflight_only:
        print(json.dumps({'status': 'preflight_passed', 'methods_ready': True,
                          'frozen_files': len(frozen), 'outcomes_read': False,
                          'started_processes': 0, 'previous_attempt': prior}, indent=2))
        return
    base = inside(args.output, root=ROOT)
    deployment = ROOT/'runtime/deployment'
    registration = deployment/(PREFIX+'.json')
    status_path = deployment/(PREFIX+'_status.json')
    logs = {name: deployment/(PREFIX+'_'+name+'.log') for name in ('predictions', 'scoring')}
    if any(p.exists() or p.is_symlink() for p in (base, registration, status_path, *logs.values())):
        raise FileExistsError('Do not overwrite or automatically repeat a registered attempt')
    state = {'status': 'starting', 'stage': 'predictions', **identity(owner),
             'registered_at': datetime.now(timezone.utc).isoformat(), 'suite': str(SUITE.relative_to(ROOT)),
             'output': str(base.relative_to(ROOT)), 'previous_attempt': prior, 'frozen_input_sha256': frozen,
             'automatic_retries': 0, 'methods_retrained': False, 'method_jobs_expected': 57,
             'prediction_timeout_seconds': 4*3600+120, 'prediction_max_bytes': 2*1024**3,
             'score_timeout_seconds': 1800, 'score_max_bytes': 512*1024**2,
             'poll_seconds': 5, 'max_log_bytes': 20*1024**2, 'steps': [],
             'monitor_limitations': 'SIGKILL bypasses cleanup; unobserved detach-and-exit cannot be recovered without cgroups.'}
    with registration.open('x') as stream:
        json.dump(state, stream, indent=2)
    write_json(status_path, state)
    stopped = False

    def request_stop(*_):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    try:
        if stopped or not suite.sample():
            raise RuntimeError('Stopped or method descendants became active before predictions')
        verify_frozen(frozen)
        python = str(ROOT/'runtime/app-env/bin/python')
        predictions, scores = base/'predictions', base/'scores'
        command = [python, str(ROOT/'scripts/predict_furosemide_methods_external.py'),
                   '--suite', str(SUITE), '--design', str(DESIGN), '--input-package', str(PACKAGE),
                   '--output', str(predictions)]
        run_child_step(name='predictions', command=command, output=predictions,
                       timeout=state['prediction_timeout_seconds'], byte_limit=state['prediction_max_bytes'],
                       scope=scope, owner=owner, state=state, status_path=status_path, frozen=frozen,
                       stopping=lambda: stopped, log_path=logs['predictions'])
        if stopped:
            raise RuntimeError('Stop requested before sealed scoring')
        from score_furosemide_methods_endpoints import validate_locked_inputs
        lock = predictions/'prediction_lock.json'
        validate_locked_inputs(lock, DESIGN)
        frozen[str(lock.relative_to(ROOT))] = checksum(lock)
        state['prediction_lock_sha256'] = checksum(lock)
        command = [python, str(ROOT/'scripts/score_furosemide_methods_endpoints.py'),
                   '--lock', str(lock), '--design', str(DESIGN), '--output', str(scores)]
        run_child_step(name='scoring', command=command, output=scores,
                       timeout=state['score_timeout_seconds'], byte_limit=state['score_max_bytes'],
                       scope=scope, owner=owner, state=state, status_path=status_path, frozen=frozen,
                       stopping=lambda: stopped, log_path=logs['scoring'])
        receipt = read_json(scores/'scoring_receipt.json')
        validate_receipt(receipt, lock)
        state.update(status='completed', completed_at=datetime.now(timezone.utc).isoformat(), receipt=receipt)
        write_json(status_path, state)
    except BaseException as exc:
        state.update(status='stopped', reason=type(exc).__name__+': '+str(exc))
        write_json(status_path, state)
        raise


if __name__ == '__main__':
    main()
