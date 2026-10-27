#!/usr/bin/env python3
"""Evidence-preserving capacity refinement of the frozen 100-compound cohort.

Four nominal methods; conditional fu stress uses the same two perturbed inputs
for parameter-only and capacity-guided PBPK. No clinical curves or held-out
endpoint values enter parameter selection, assembly or capacity anchoring.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np

from run_purpose_hundred import Runner as BaseRunner, Monitor, Paused, digest, now
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
from pbpk_kb.blueprint_policy import transport_calibration as engine
from pbpk_kb.blueprint_policy import capacity_guided_design as capacity

VERSION = 'purpose-capacity-refinement-1'
DEFAULT = ROOT / 'experiments/blueprint_design/purpose_refinement_20261001'
SOURCE = ROOT / 'experiments/blueprint_design/purpose_hundred_20260930'
METHODS = ('baseline', 'parameter_only', 'capacity_only', 'combined')
PHASES = ('nominal', 'fu_low', 'fu_high')


def evidence_request(original, prior, source, folder, *, phase='nominal'):
    """Only replace qualified predicted inputs; measured/derived values survive."""
    request = deepcopy(original)
    rows = {r['parameter']: r for r in request['evidence']}
    changes = {}
    for parameter in ('total_clearance', 'fraction_unbound'):
        row = rows[parameter]
        if row['basis'] != 'predicted':
            continue
        value = prior['methods']['selected']['predict_nominal'][parameter]
        if parameter == 'fraction_unbound' and phase != 'nominal':
            bound = 'lower' if phase == 'fu_low' else 'upper'
            value = prior['methods']['selected']['ranges'][parameter]['80'][bound]
        changes[parameter] = {'old_evidence': deepcopy(row), 'new_value': value,
            'basis': 'predicted', 'phase': phase, 'predictor': prior['selected_models'][parameter],
            'source': source, 'interpretation': 'Scaffold-held-out parameter prediction; fu stress is conditional, not clinical truth'}
    folder.mkdir(parents=True, exist_ok=True)
    change_path = folder / 'parameter_source.json'
    save(change_path, changes)
    change_ref = {'path': str(change_path.relative_to(ROOT)), 'sha256': sha256(change_path)}
    for parameter, change in changes.items():
        row = rows[parameter]
        row['value'] = change['new_value']
        row['source'] = dict(change_ref, location=parameter)
        row['prediction_method'] = {'name': change['predictor'], 'version': VERSION,
            'training_overlap': False, 'applicability': {'target_structure_and_scaffold_excluded': True,
                'human_IV_parameter_model': True, 'clinical_curve_used': False}}
        row['notes'] = 'Training-only model selection; target references are forbidden inputs.'
    ledger = folder / 'evidence.json'
    save(ledger, {'evidence': request['evidence']})
    request['evidence_ledger'] = {'path': str(ledger.relative_to(ROOT)), 'sha256': sha256(ledger),
                                  'location': 'Full evidence ledger including protected original inputs'}
    return EvidenceBlueprintRequest.model_validate(request).model_dump(mode='json', exclude_none=True)


def volume_evidence(request, prior, source, method):
    """No observed evaluation value can enter the Vss sidecar."""
    return [{'parameter': 'vss', 'compound_id': request['compound_id'],
        'value': prior['methods'][method]['predict_nominal']['vss'], 'unit': 'l/kg',
        'basis': 'predicted', 'source': dict(source, location=request['compound_id'] + '/' + method + '/vss'),
        'context': {'species': 'human', 'matrix': 'plasma', 'route': 'intravenous',
            'study_role': 'independent_parameters', 'independent_of_target': True},
        'prediction_method': {'name': 'random_forest' if method == 'rf' else prior['selected_models']['vss'],
            'version': VERSION, 'training_overlap': False,
            'applicability': {'target_structure_and_scaffold_excluded': True, 'human_IV_parameter_model': True}}}]


def fold_change(a, b):
    if a is None or b is None or not all(math.isfinite(x) and x >= 0 for x in (a, b)):
        return None
    if min(a, b) == 0:
        return 1. if max(a, b) == 0 else None
    return max(a / b, b / a)


def conditional_stability(nominal, stresses):
    """fu-only drift, explicitly not a clinical interval or validation score."""
    result = {}
    for name, endpoint in nominal.get('endpoints', {}).items():
        value = endpoint.get('value')
        changes = []
        for stress in stresses:
            other = stress.get('endpoints', {}).get(name, {}).get('value')
            change = (abs(value - other) if name == 'tmax' and value is not None and other is not None
                      else fold_change(value, other))
            changes.append(change)
        complete = len(stresses) == 2 and all(v is not None for v in changes)
        result[name] = {'max_change': max(changes) if complete else None, 'complete': complete,
            'metric': 'absolute_h' if name == 'tmax' else 'fold',
            'clinical_reliability_claimed': False}
    return result


class RefinementRunner(BaseRunner):
    def __init__(self, project_root=ROOT):
        super().__init__(project_root=project_root)
        self.default = self.root / 'experiments/blueprint_design/purpose_refinement_20261001'

    def own_run(self, path):
        path = inside(path, self.root)
        if path.parent != self.root / 'experiments/blueprint_design' or not path.name.startswith('purpose_refinement_'):
            raise ValueError('Use a dedicated project purpose_refinement_ directory')
        return path

    def prepare(self, run=None, workers=14):
        run = self.own_run(run or self.default)
        if not 1 <= workers <= 14:
            raise ValueError('One to fourteen workers required')
        if any((run / name).exists() for name in ('protocol.json', 'state.json', 'work', 'results.json')):
            raise FileExistsError('Preserve this batch; use run or resume')
        old = self.read(SOURCE / 'protocol.json')
        old_state = self.read(SOURCE / 'state.json')
        if old_state['status'] != 'complete' or old_state['protocol'] != self.ref(SOURCE / 'protocol.json'):
            raise ValueError('Previous batch is not a completed frozen comparator')
        self.verify(old['source_pins']); self.verify(old['code_pins'])
        self.verify(old_state['report_artifact_pins'])
        prior_path = run / 'priors/predictions.json'
        if not prior_path.is_file():
            raise ValueError('Train and freeze the adaptive prior before batch preparation')
        from pbpk_kb.blueprint_learning.adaptive_parameter_prior import fit_adaptive_prior
        bundle = fit_adaptive_prior(self.root / 'artifacts/pk_predictors/pksmart/pksmart-3.0.1-py3-none-any.whl',
            cohort_manifest=SOURCE / 'cohort.json', source_baseline_audit=SOURCE / 'priors/audit.json',
            output_dir=run / 'priors', project_root=self.root, n_jobs=4)
        if not bundle['reused']:
            raise ValueError('Preparation must verify an already frozen prior, not retrain it')
        prior_ref = self.ref(prior_path)
        audit = self.read(run / 'priors/audit.json')
        # The predictor supplies explicit dataset/split/code integrity receipts.
        source_pins = {**old['source_pins'], **old['code_pins'], **old_state['report_artifact_pins']}
        source_pins.update({str(p.relative_to(self.root)): sha256(p) for p in [SOURCE / 'protocol.json', prior_path, run / 'priors/audit.json', run / 'priors/model.joblib']})
        source_pins.update(audit.get('source_pins', {}))
        cases = old['cases']
        if len(cases) != 100 or len({c['identity_smiles'] for c in cases}) != 100:
            raise ValueError('This development batch requires all frozen 100 identities')
        plans = {}
        for case in cases:
            cid, compound = case['case_id'], case['compound_id']
            request = deepcopy(old['case_protocols'][cid]['input']['request'])
            self.adapter.checked_input(old['case_protocols'][cid]['input'])
            prior = bundle['predictions'][compound]
            if prior['identity_smiles'] != case['identity_smiles']:
                raise ValueError('Parameter prediction identity mismatch')
            folder = SOURCE / 'work' / cid / 'baseline'
            receipt = self.read(folder / 'receipt.json')
            self.verify(receipt['artifact_pins'])
            baseline_ref, physiology_ref = self.ref(folder / 'prediction.json'), self.ref(folder / 'physiology/perfusion_physiology.json')
            for item in (baseline_ref, physiology_ref):
                source_pins[item['path']] = item['sha256']
            # Verify the RF arm reproduces frozen CL/fu nominal inputs.
            for parameter in ('total_clearance', 'fraction_unbound'):
                original = next(r for r in request['evidence'] if r['parameter'] == parameter)
                if original['basis'] == 'predicted' and not math.isclose(original['value'], prior['methods']['rf']['predict_nominal'][parameter], rel_tol=1e-11):
                    raise ValueError('RF comparator differs from its frozen training/data contract')
            plans[cid] = {'request': request, 'prior': prior, 'prior_ref': prior_ref,
                'baseline_prediction': baseline_ref, 'baseline_physiology': physiology_ref,
                'decision': {'scenarios': [{'scenario_id': s} for s in PHASES]}}
        code = engine._code_pins(self.root)
        for name in ['scripts/run_purpose_refinement.py', 'scripts/run_purpose_hundred.py',
                     'scripts/design_evidence_pbpk.py', 'src/pbpk_kb/blueprint_policy/capacity_guided_design.py',
                     'src/pbpk_kb/blueprint_policy/purpose_refinement_evaluation.py',
                     'src/pbpk_kb/blueprint_policy/hundred_evaluation.py',
                     'src/pbpk_kb/blueprint_learning/adaptive_parameter_prior.py']:
            code[name] = self.ref(name)['sha256']
        protocol = {'version': VERSION, 'created_at': now(), 'cases': cases, 'case_protocols': plans,
            'expected_compounds': 100, 'workers': workers, 'native_threads': 2, 'native_timeout_seconds': 180,
            'monitor_seconds': 1200, 'directory_limit_bytes': 10 * 1024**3,
            'source_pins': source_pins, 'code_pins': code, 'paper_guard': self.paper_pins(),
            'baseline_results': self.ref(SOURCE / 'results.json'), 'methods': list(METHODS),
            'maximum_new_native_runs': 300, 'nominal_comparisons': 400,
            'conditional_fu_stress': {'phases': list(PHASES), 'range': '80% held-out absolute residual stress range',
                'fixed': 'CL and nominal predicted Vss, demographics, blood flow, dose and exchange scale; fu-dependent capacities are recomputed',
                'interpretation': 'Conditional missing-fu robustness, not total predictive uncertainty'},
            'reference_scope': '62 strict aggregate PK reference identities; no clinical concentration curves',
            'purpose_decision': 'Computed research prediction with parameter/structural gaps; unsupported construction blocks; no clinical confidence certificate',
            'target_used_for_model_selection_or_anchoring': False, 'paper_updated': False,
            'development_status': 'All100 previously inspected; not confirmatory validation'}
        self.verify(source_pins); self.verify(code)
        save(run / 'protocol.json', protocol)
        save(run / 'state.json', {'version': VERSION, 'protocol': self.ref(run / 'protocol.json'),
            'status': 'prepared', 'outcomes': {}, 'progress': {}, 'monitor': {},
            'consecutive_infrastructure_errors': 0, 'all_predictions_locked_before_scoring': False,
            'monitor_stopped_before_scoring': False})
        return {'status': 'prepared', 'cases': 100, 'methods': list(METHODS), 'native_maximum': 300}

    def load(self, run, *, locked=False, artifacts=True):
        run = self.own_run(run)
        state = self.read(run / 'state.json')
        protocol = self.checked(state['protocol'])
        if state['version'] != VERSION or protocol['version'] != VERSION:
            raise ValueError('Version changed')
        self.verify(protocol['source_pins']); self.verify(protocol['code_pins'])
        if self.paper_pins() != protocol['paper_guard']:
            raise ValueError('Paper changed')
        expected = {c['case_id'] for c in protocol['cases']}
        if len(expected) != 100 or not set(state['outcomes']) <= expected:
            raise ValueError('Frozen cohort changed')
        for cid, reference in state['outcomes'].items():
            outcome = self.checked(reference)
            if outcome['case_id'] != cid or outcome['protocol'] != state['protocol'] or outcome['status'] != 'locked':
                raise ValueError('Outcome identity or lock differs')
            if set(outcome['methods']) != set(METHODS):
                raise ValueError('Outcome omitted a method')
            if artifacts:
                self.verify(outcome['artifact_pins'])
        if locked and (state['status'] not in {'predictions_locked', 'complete'} or set(state['outcomes']) != expected
                       or not state['all_predictions_locked_before_scoring']):
            raise ValueError('All100 predictions and failures must lock before scoring')
        return protocol, state

    def _simulate(self, physiology, request, evidence, folder, *, anchored):
        try:
            decision = capacity.anchor_partition(physiology, request, evidence) if anchored else {
                'status': 'retain_unanchored', 'log_parameters': {'clearance_scale': 0., 'partition_scale': 0., 'bidirectional_exchange_scale': 0.}}
            if decision['status'] == 'blocked':
                return {'status': 'failed', 'reason': 'capacity_anchor_blocked', 'decision': decision, 'endpoints': {}}
            prediction = capacity.simulate_design(physiology, request, log_parameters=decision['log_parameters'])
        except (ValueError, ArithmeticError) as error:
            return {'status': 'failed', 'reason': str(error), 'endpoints': {}}
        prediction['decision'] = decision
        # A compressed complete trajectory enables independent NCA-tail checks.
        trajectory = prediction.pop('full_trajectory')
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'trajectory.npz', time_h=np.asarray(trajectory['time_h']),
                            concentration_mg_l=np.asarray(trajectory['concentration_mg_l']))
        prediction['full_trajectory_ref'] = self.ref(folder / 'trajectory.npz')
        return prediction

    def _phase(self, run, protocol, case, phase, monitor):
        cid = case['case_id']; plan = protocol['case_protocols'][cid]
        folder = run / 'work' / cid / phase
        receipt_path = folder / 'receipt.json'
        if receipt_path.exists():
            receipt = self.read(receipt_path)
            self.verify(receipt['artifact_pins'])
            return self.checked(receipt['prediction'])
        monitor.check()
        interrupted = folder.exists() and any(folder.iterdir())
        folder.mkdir(parents=True, exist_ok=True)
        if interrupted:
            result = {'status': 'failed', 'failure_kind': 'interrupted_native', 'phase': phase,
                'reason': 'Incomplete artifacts retained; no automatic retry', 'methods': {}}
        else:
            save(folder / 'started.json', {'phase': phase, 'started_at': now()})
            methods = {}
            try:
                if phase == 'nominal':
                    base_physiology = self.checked(plan['baseline_physiology'])
                    methods['baseline'] = self.checked(plan['baseline_prediction'])
                    methods['capacity_only'] = self._simulate(base_physiology, plan['request'],
                        volume_evidence(plan['request'], plan['prior'], plan['prior_ref'], 'rf'), folder / 'capacity_only', anchored=True)
                request = evidence_request(plan['request'], plan['prior'], plan['prior_ref'], folder, phase=phase)
                if phase == 'nominal':
                    rows = {r['parameter']: r for r in request['evidence']}
                    original = {r['parameter']: r for r in plan['request']['evidence']}
                    same = all(rows[k]['value'] == original[k]['value'] for k in ('total_clearance', 'fraction_unbound'))
                else:
                    same = False
                physiology = (self.checked(plan['baseline_physiology']) if same else
                              engine._physiology(request, folder / 'native_source', self.root, monitor.check))
                vs = volume_evidence(request, plan['prior'], plan['prior_ref'], 'selected')
                methods['parameter_only'] = self._simulate(physiology, request, [], folder / 'parameter_only', anchored=False)
                methods['combined'] = self._simulate(physiology, request, vs, folder / 'combined', anchored=True)
                result = {'status': 'completed', 'phase': phase, 'methods': methods,
                    'native_reused': same, 'request': request,
                    'clinical_observations_used': False, 'locked_at': now()}
                monitor.record_result(False)
            except Paused:
                raise
            except (ValueError, ArithmeticError) as error:
                result = {'status': 'failed', 'phase': phase, 'failure_kind': 'model_or_input_failure',
                          'reason': str(error), 'methods': methods}
                monitor.record_result(False)
            except Exception as error:
                result = {'status': 'failed', 'phase': phase, 'failure_kind': 'infrastructure_failure',
                          'reason': type(error).__name__ + ': ' + str(error), 'methods': methods}
                monitor.record_result(True)
        save(folder / 'prediction.json', result)
        save(receipt_path, {'prediction': self.ref(folder / 'prediction.json'), 'artifact_pins': self.artifacts(folder)})
        with monitor.lock:
            progress = monitor.state['progress'].setdefault(cid, {'completed': 0, 'failed': 0})
            progress['completed'] += 1
            progress['failed'] += result['status'] == 'failed'
            monitor.persist()
        return result

    def _case(self, run, protocol, case, monitor):
        cid = case['case_id']; folder = run / 'work' / cid
        phases = {phase: self._phase(run, protocol, case, phase, monitor) for phase in PHASES}
        nominal = phases['nominal']['methods']
        methods = {m: nominal.get(m, {'status': 'failed', 'endpoints': {}, 'reason': phases['nominal'].get('reason')}) for m in METHODS}
        # This comparator is already independently locked; a new construction
        # interruption cannot turn the retained original prediction into failure.
        methods['baseline'] = self.checked(protocol['case_protocols'][cid]['baseline_prediction'])
        stability = {m: conditional_stability(methods[m], [phases[p]['methods'].get(m, {}) for p in PHASES[1:]])
                     for m in ('parameter_only', 'combined')}
        outcome = {'case_id': cid, 'status': 'locked', 'protocol': monitor.state['protocol'],
            'methods': methods, 'conditional_fu_stability': stability,
            'phases': {p: self.ref(folder / p / 'prediction.json') for p in PHASES},
            'purpose_status': 'research_only_not_clinically_validated', 'locked_at': now(),
            'artifact_pins': self.artifacts(folder, excluded=('outcome.json',))}
        save(folder / 'outcome.json', outcome)
        with monitor.lock:
            monitor.state['outcomes'][cid] = self.ref(folder / 'outcome.json')
            monitor.persist()
        return cid

    def report(self, run=None):
        from pbpk_kb.blueprint_policy.hundred_evaluation import _records
        from pbpk_kb.blueprint_policy.purpose_refinement_evaluation import score_refinement
        run = self.own_run(run or self.default)
        with (run / '.run.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            protocol, state = self.load(run, locked=True)
            if state['status'] == 'complete':
                self.verify(state['report_artifact_pins'])
                return self.checked(state['results'])
            # First access to evaluation labels in this runtime, after ALL locks.
            baseline = self.checked(protocol['baseline_results'])
            by_method = {'baseline': baseline, **{m: {'records': [], 'all_predictions_locked_before_scoring': True} for m in METHODS[1:]}}
            stability = []
            for case in protocol['cases']:
                outcome = self.checked(state['outcomes'][case['case_id']])
                reference = self.checked(case['evaluation'])
                for method in METHODS[1:]:
                    p = dict(outcome['methods'][method], scenario_id='baseline')
                    pseudo = {'predictions': [p], 'construction_status': 'predicted' if p['status'] == 'predicted' else 'failed',
                        'assessment': {'endpoint_assessments': {}}, 'input_only_score': None}
                    rows = list(_records(case, pseudo, reference))
                    for row in rows:
                        row['fixed_auto_same_nominal_prediction'] = False
                        row['method'] = method
                        row['purpose_status'] = outcome['purpose_status']
                    by_method[method]['records'].extend(rows)
                stability.append({'case_id': case['case_id'], 'drug_name': case['drug_name'],
                                  'methods': outcome['conditional_fu_stability']})
            result = score_refinement(protocol, by_method, self.root)
            result.update(conditional_fu_stability=stability,
                all_predictions_locked_before_scoring=True, paper_updated=False,
                baseline_uses_old_nominal_only=True, clinical_curve_validation_n=0)
            save(run / 'results.json', result)
            import csv
            records = result.get('records') or [dict(row, method=m) for m, b in by_method.items() for row in b['records']]
            keys = sorted(set().union(*(r.keys() for r in records)))
            with (run / 'results.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
                for row in records:
                    writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
            self.verify(protocol['source_pins']); self.verify(protocol['code_pins'])
            if self.paper_pins() != protocol['paper_guard']:
                raise ValueError('Paper changed during report')
            state.update(status='complete', results=self.ref(run / 'results.json'), reported_at=now(),
                report_artifact_pins={str((run / name).relative_to(self.root)): sha256(run / name) for name in ('results.json', 'results.csv')})
            save(run / 'state.json', state)
            return result

    def recorded_process(self, run):
        import psutil
        path = run / 'pid.json'
        if not path.exists(): return None
        p = self.read(path)
        try:
            process = psutil.Process(p['pid'])
            if process.create_time() != p['create_time'] or process.status() == psutil.STATUS_ZOMBIE: return None
            if process.uids().real != os.getuid() or Path(process.cwd()).resolve() != self.root:
                raise ValueError('PID ownership changed')
            args = process.cmdline(); script = str(self.root / 'scripts/run_purpose_refinement.py')
            if p['script'] != script or p['run'] != str(run) or script not in args or args[args.index('--run') + 1] != str(run):
                raise ValueError('PID no longer belongs to this batch')
            return {'pid': p['pid'], 'running': True}
        except psutil.NoSuchProcess:
            return None

    def background(self, run=None):
        import psutil
        run = self.own_run(run or self.default)
        protocol, state = self.load(run)
        if state['status'] == 'complete': return {'status': 'complete'}
        if (run / 'PAUSE').exists(): return {'status': 'paused'}
        with (run / '.launch.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            live = self.recorded_process(run)
            if live: return {'status': 'already_running', **live}
            script = self.root / 'scripts/run_purpose_refinement.py'
            env = os.environ.copy()
            cache = self.root / 'runtime/cache/purpose_refinement'; cache.mkdir(parents=True, exist_ok=True)
            env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', TMPDIR=str(cache))
            with (run / 'run.log').open('ab') as log:
                child = subprocess.Popen([sys.executable, '-B', str(script), 'run', '--run', str(run), '--report-after-run'],
                    cwd=self.root, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, env=env)
            save(run / 'pid.json', {'pid': child.pid, 'create_time': psutil.Process(child.pid).create_time(),
                                   'script': str(script), 'run': str(run), 'started_at': now()})
            return {'status': 'started', 'pid': child.pid, 'workers': protocol['workers']}

    def status(self, run=None):
        run = self.own_run(run or self.default)
        protocol, state = self.load(run, artifacts=False)
        failed_methods = {m: 0 for m in METHODS}
        for reference in state['outcomes'].values():
            outcome = self.checked(reference)
            for m, prediction in outcome['methods'].items():
                failed_methods[m] += prediction['status'] != 'predicted'
        return {'status': state['status'], 'cases_locked': len(state['outcomes']), 'cases_total': len(protocol['cases']),
            'phases_completed': sum(p['completed'] for p in state['progress'].values()),
            'phases_total': 300, 'failed_phases': sum(p['failed'] for p in state['progress'].values()),
            'failed_nominal_methods_in_locked_cases': failed_methods,
            'pause_reason': state.get('pause_reason'), 'monitor': state['monitor'], 'process': self.recorded_process(run),
            'results': state.get('results'), 'paper_updated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'status', 'pause', 'resume', 'report'])
    parser.add_argument('--run', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--report-after-run', action='store_true')
    args = parser.parse_args(); runner = RefinementRunner()
    if args.command == 'prepare':
        result = runner.prepare(args.run, args.workers)
    elif args.command == 'run':
        result = runner.background(args.run) if args.background else runner.execute(args.run)
    elif args.command == 'resume':
        result = runner.resume(args.run, background=args.background)
    else:
        result = getattr(runner, args.command)(args.run)
    if args.command in {'run', 'resume'} and not args.background and args.report_after_run and result['status'] in {'predictions_locked', 'complete'}:
        runner.report(args.run); result = runner.status(args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__': main()
