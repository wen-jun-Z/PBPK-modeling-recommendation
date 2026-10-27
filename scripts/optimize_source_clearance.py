#!/usr/bin/env python3
"""One mechanism-parameter development run, preserving the complete PBPK model.

Known historical studies are explicitly re-registered for calibration/selection.
This is not an unseen-data experiment and does not fit the troublesome troughs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from pbpk_kb.blueprints.heldout_evaluation import evaluate_locked_predictions, lock_predictions
from pbpk_kb.blueprints.source_adaptation_audit import audit_execution
from pbpk_kb.blueprints.source_context import install_calibrated_parameters
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from evaluate_source_context import native_job
from watch_blueprint_suite import ProcessScope

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'experiments/blueprint_design/pk_guided_pilot_20260921'
FIT_CASES = ('rifampicin_iv_source', 'rifampicin_iv_600mg_30min_pk')
SELECT_CASES = ('rifampicin_iv_dose_infusion',)
PARAMETER = 'Rifampicin-AADAC-Nakajima 2011|kcat'
ASSUMPTIONS = [
    'Historical development: these studies and earlier outcomes have already been inspected; no independent validation claim.',
    'Fit the two Sanofi/FDA IV doses as one study, not two independent studies.',
    'One default adult source individual represents aggregate concentrations; missing clinical covariates and variability are not fitted.',
    'Nitti serum observations use the source model plasma mapping; matrix equivalence is an explicit unverified assumption.',
    'CL diagnostic discrepancies motivate a clearance hypothesis, not automatic acceptance of PK constraints or proof of an AADAC defect.',
    'One shared kcat transfers to all IV/oral regimens; induction, transport, fu, distribution and absorption remain source assumptions.',
    'The half-to-double search window is a conservative development window inside source PI bounds, not a physiological confidence interval.',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(path.read_text())


def parameter_spec(cases):
    reference = next(c['source'] for c in cases if c['case_id'] == FIT_CASES[0])
    source_path = inside(reference['path'], ROOT, exists=True)
    if sha256(source_path) != reference['sha256']:
        raise ValueError('Source PBPK model has changed')
    source = load(source_path)
    pi = next(p for p in source['ParameterIdentifications'] if p['Name'] == 'PI OSP10')
    registered = next(p for p in pi['IdentificationParameters'] if p['Name'] == 'Esterase kcat')
    if not all(p.endswith('|' + PARAMETER) for p in registered['LinkedParameters']):
        raise ValueError('Official identification no longer binds the same mechanism')
    values = {p['Name']: p for p in registered['Parameters']}
    value = values['Start value']['Value']
    if values['Start value']['Unit'] != '1/min' or value != 9.865:
        raise ValueError('Source mechanism version changed; review it before fitting')
    low, high = value / 2, value * 2
    if not values['MinValue']['Value'] <= low < high <= values['MaxValue']['Value']:
        raise ValueError('Development window must remain inside source search bounds')
    return {'parameter_id': 'aadac_kcat', 'path': PARAMETER, 'unit': '1/min',
            'source_value': value, 'lower_bound': low, 'upper_bound': high, 'scaling': 'log',
            'evidence_reference': reference['path'] + '#ParameterIdentifications[PI OSP10].Esterase kcat; source native value independently read back'}


def score(rows, curves, method):
    locked = {'method': method, 'seed': 0, 'candidate_id': method, 'status': 'completed',
              'predictions_sha256': lock_predictions(curves)}
    return evaluate_locked_predictions(rows, curves, locked, device='cpu')


def run(args):
    from pbpk_kb.blueprints.mechanism_calibration import fit_select_source_mechanism
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1..4 native workers')
    baseline = inside(args.baseline, ROOT, exists=True)
    summary = load(baseline / 'summary.json')
    if summary['status'] != 'completed' or sha256(baseline / 'predictions_locked.json') != summary['prediction_lock_sha256']:
        raise ValueError('A completed, unchanged baseline is required')
    protocol_before = load(baseline / 'protocol.json')
    cases = protocol_before['cases']
    prepared = load(baseline / 'prepared_models.json')
    base_jobs = load(baseline / 'predictions_locked.json')['jobs']
    # Observations are already known development data. The role assignment
    # below is frozen before any new fit and no evaluation row is passed to it.
    observations = load(baseline / 'observations.json')
    spec = parameter_spec(cases)
    output = inside(args.output, ROOT)
    output.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    state = {'status': 'running', 'stage': 'source_readback', 'started_at': now(),
             'cases': len(cases), 'completed_native_jobs': 0, 'planned_native_jobs': 6}
    save(output / 'runner.json', {**ProcessScope().state(os.getpid()), 'started_at': now()})
    save(ROOT / 'runtime/deployment/source_adaptation_pilot.json',
         {'output': str(output.relative_to(ROOT)), 'registered_at': now()})

    def update(**values):
        state.update(values, elapsed_seconds=round(time.monotonic() - begin, 2))
        save(output / 'progress.json', state)

    update()
    try:
        refs = [baseline / p for p in ('protocol.json', 'summary.json', 'prepared_models.json',
                                      'predictions_locked.json', 'observations.json', 'pk_checks.json')]
        code = [Path(__file__), ROOT / 'scripts/evaluate_source_context.py']
        code += [ROOT / 'src/pbpk_kb/blueprints' / (name + '.py') for name in (
            'mechanism_calibration', 'source_calibration', 'source_context', 'source_adaptation_audit',
            'parameter_batch', 'design_runtime', 'heldout_evaluation', 'design_scoring')]
        code += [ROOT / 'src/pbpk_kb/model_runtime/R' / (name + '.R') for name in (
            'blueprint_parameter_batch', 'source_adaptation_audit', 'blueprint_snapshot_run')]
        protocol = {'frozen_at': now(), 'baseline': str(baseline.relative_to(ROOT)),
            'cases': cases, 'calibration_cases': list(FIT_CASES), 'selection_cases': list(SELECT_CASES),
            'development_evaluation_cases': [c['case_id'] for c in cases if c['case_id'] not in FIT_CASES + SELECT_CASES],
            'parameter_spec': spec, 'context_assumptions': ASSUMPTIONS,
            'selection_rule': 'One fixed candidate must improve study-equal selection log-MSE and not worsen AUC/Cmax absolute-log errors beyond numerical tolerance 1e-6; otherwise retain source.',
            'prediction_reuse': 'Prior completed baseline curves are reused and checksum-bound; candidate is newly solved for every Rifampicin scenario regardless of promotion.',
            'scientific_status': 'historical_development_not_independent_validation',
            'max_native_rebuilds': 6, 'max_rounds': 2, 'workers': args.workers,
            'inputs': [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)} for p in refs],
            'code': [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)} for p in code]}
        save(output / 'protocol.json', protocol)
        receipts = {}
        inputs = {'calibration': [], 'selection': []}
        for case in cases:
            case_id = case['case_id']
            if case_id not in FIT_CASES + SELECT_CASES:
                continue
            role = 'calibration' if case_id in FIT_CASES else 'selection'
            native = base_jobs[case_id + '/context']
            snapshot = prepared[case_id]['context']['snapshot']
            expected = deepcopy(case['native_expectations']['adapted'])
            expected['expectations'].append({'path': PARAMETER, 'value': spec['source_value'], 'unit': spec['unit']})
            audit_dir = output / 'source_readback' / case_id
            receipt = audit_execution(snapshot=snapshot, execution_manifest=native['execution_manifest'],
                output_dir=audit_dir, project_root=ROOT, **expected)
            if not receipt['accepted']:
                raise ValueError('Source mechanism readback failed: ' + case_id)
            receipt_path = audit_dir / 'verified_source.json'
            save(receipt_path, receipt)
            for row in observations:
                if row['case_id'] != case_id:
                    continue
                item = {'case_id': row['scenario_id'], 'study_group_id': row['study_group_id'], 'role': role,
                    'pkml_path': native['pkml_path'], 'target_output': case['target_output'],
                    'observations': deepcopy(row['observations'])}
                if role == 'selection':
                    item['sampling_type'] = row['sampling_type']
                inputs[role].append(item)
                receipts[item['case_id']] = {'path': str(receipt_path.relative_to(ROOT)), 'sha256': sha256(receipt_path)}
        update(stage='mechanism_calibration_and_selection')
        fitted = fit_select_source_mechanism(inputs['calibration'], inputs['selection'], [spec],
            output / 'calibration_and_selection', ROOT, source_readback_receipts=receipts,
            disclosed_context_assumptions=ASSUMPTIONS, max_rounds=2, device='cpu')
        save(output / 'decision.json', fitted)
        if fitted['status'] != 'completed':
            raise ValueError('Mechanism calibration/selection failed; source retained, inspect decision.json')
        candidate = fitted['candidate_values']
        selected = fitted['decision']['candidate_promoted']
        candidate_results = {}
        rif_cases = [c for c in cases if c['compound_name'] == 'Rifampicin']
        if len(rif_cases) != protocol['max_native_rebuilds']:
            raise ValueError('Frozen native rebuild count changed')
        update(stage='candidate_rebuild_and_readback', planned_native_jobs=len(rif_cases))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for case in rif_cases:
                grid = prepared[case['case_id']]['context']
                snapshot = install_calibrated_parameters(grid['snapshot'], [spec], candidate,
                    str((output / 'calibration_and_selection/calibration/calibration_result.json').relative_to(ROOT)))
                future = pool.submit(native_job, case, 'candidate', snapshot, grid['manifest'], output, spec, candidate[0])
                futures[future] = case['case_id']
            for future in as_completed(futures):
                cid = futures[future]
                candidate_results[cid] = future.result()
                update(completed_native_jobs=state['completed_native_jobs'] + 1)
                print(cid, candidate_results[cid]['status'], flush=True)
        jobs = {}
        for case in cases:
            cid = case['case_id']
            base = base_jobs[cid + '/context']
            jobs[cid + '/baseline'] = {**base, 'reused_baseline': True}
            cand = candidate_results.get(cid, {**base, 'shared_unchanged_control': True})
            jobs[cid + '/candidate'] = cand
            jobs[cid + '/selected'] = cand if selected and cid in candidate_results else base
        save(output / 'predictions_locked.json', {'locked_at': now(), 'jobs': jobs,
             'candidate_promoted_before_evaluation': selected})
        update(stage='historical_development_evaluation')
        evaluation = {}
        for arm in ('baseline', 'candidate', 'selected'):
            evaluation[arm] = {}
            for phase, ids in [('calibration', FIT_CASES), ('selection', SELECT_CASES),
                              ('development_evaluation', tuple(protocol['development_evaluation_cases']))]:
                evaluation[arm][phase] = {}
                for sampling in sorted({r['sampling_type'] for r in observations if r['case_id'] in ids}):
                    rows = [dict(r, role=phase, study_role=phase, dataset_role=phase)
                            for r in observations if r['case_id'] in ids and r['sampling_type'] == sampling]
                    curves = {r['scenario_id']: jobs[r['case_id'] + '/' + arm]['curve'] for r in rows}
                    evaluation[arm][phase][sampling] = score(rows, curves, arm + '_' + phase)
        save(output / 'evaluations.json', evaluation)
        passed = sum(r['status'] == 'passed' for r in candidate_results.values())
        stable = all(sha256(ROOT / r['path']) == r['sha256'] for r in protocol['code'] + protocol['inputs'])
        result = {'status': 'completed' if passed == len(rif_cases) and stable else 'completed_with_failures',
            'completed_at': now(), 'elapsed_seconds': round(time.monotonic() - begin, 2),
            'candidate_promoted': selected, 'source_kcat': spec['source_value'],
            'candidate_kcat': candidate[0], 'retained_kcat': fitted['retained_values'][0], 'unit': spec['unit'],
            'new_native_rebuilds': len(rif_cases), 'native_rebuilds_passed': passed,
            'curve_count': len(observations), 'parameter_count': 1,
            'calibration_studies': sorted({r['study_group_id'] for r in inputs['calibration']}),
            'selection_studies': sorted({r['study_group_id'] for r in inputs['selection']}),
            'code_and_inputs_unchanged': stable, 'independent_validation': False,
            'prediction_lock_sha256': sha256(output / 'predictions_locked.json'),
            'metrics': {a: {p: {g: e['tiers']['reconstruction'] for g, e in gs.items()} for p, gs in ps.items()}
                        for a, ps in evaluation.items()}}
        save(output / 'summary.json', result)
        update(status=result['status'], stage='completed', successful_native_jobs=passed)
        print(json.dumps({k: v for k, v in result.items() if k != 'metrics'}, ensure_ascii=False, indent=2), flush=True)
        return result
    except Exception as error:
        update(status='failed', stage='failed', reason=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path(BASELINE))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    raise SystemExit(0 if run(parser.parse_args())['status'] == 'completed' else 2)
