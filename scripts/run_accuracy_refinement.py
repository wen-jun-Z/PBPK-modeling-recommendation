#!/usr/bin/env python3
"""Compare richer structure learning and learned exchange on the frozen 100 drugs.

Target labels are opened only by report, after every prediction/failure locks.
Direct ML parameters are a diagnostic, not a simulated concentration curve.
The source paper and previous results are immutable comparators.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_purpose_hundred import Runner as BaseRunner, Paused, now, digest
from run_purpose_refinement import evidence_request, volume_evidence
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprint_policy import capacity_guided_design as capacity
from pbpk_kb.blueprint_policy import transport_calibration as engine
from pbpk_kb.blueprint_learning.adaptive_parameter_prior import _digest as prior_digest

import numpy as np

VERSION = 'accuracy-rich-exchange-1'
PRIOR_VERSION = 'rich-scaffold-five-endpoint-prior-1'
DEFAULT = ROOT / 'experiments/blueprint_design/accuracy_refinement_20261001'
SOURCE = ROOT / 'experiments/blueprint_design/purpose_refinement_20261001'
METHODS = ('previous_combined', 'rich_fixed_exchange', 'rich_learned_exchange', 'direct_ml')
ENDPOINTS = ('clearance', 'vss', 'terminal_half_life', 'mrt')
SEED = 20261001


class IntegrityError(ValueError):
    """Artifact/source corruption must stop execution, not become a scored failure."""


def rich_request(original, prior, source, folder):
    """Reuse evidence protection while attributing changed rows to the new prior."""
    request = evidence_request(original, prior, source, folder)
    for row in request['evidence']:
        if row['parameter'] in {'total_clearance','fraction_unbound'} and row['basis']=='predicted':
            row['prediction_method']['version'] = PRIOR_VERSION
            row['notes'] = 'Rich-structure scaffold-CV selection on training compounds only; target labels excluded.'
    ledger = folder/'evidence.json'
    save(ledger, {'evidence': request['evidence']})
    request['evidence_ledger']['sha256'] = sha256(ledger)
    return request


def rich_volume(request, prior, source):
    records = volume_evidence(request, prior, source, 'selected')
    for record in records:
        record['prediction_method']['version'] = PRIOR_VERSION
    return records


def direct_parameters(prior, request):
    """Only model predictions; canonical units follow the frozen human weight."""
    values = prior['methods']['selected']['predict_nominal']
    weight = float(request['human']['weight_kg'])
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('Positive human weight required')
    converted = {'clearance': (values['total_clearance'] * weight * .06, 'l/h'),
        'vss': (values['vss'] * weight, 'l'),
        'terminal_half_life': (values['terminal_half_life'], 'h'), 'mrt': (values['mrt'], 'h')}
    if any(not math.isfinite(v) or v <= 0 for v, _ in converted.values()):
        raise ValueError('Model parameters must be finite and positive')
    return {'status': 'predicted', 'parameter_diagnostic_only': True, 'curve_simulated': False,
        'endpoints': {k: {'value': float(v), 'unit': unit, 'status': 'predicted'}
                      for k, (v, unit) in converted.items()}}


def half_evidence(request, prior, source):
    return [{'parameter': 'terminal_half_life', 'compound_id': request['compound_id'],
        'value': prior['methods']['selected']['predict_nominal']['terminal_half_life'], 'unit': 'h',
        'basis': 'predicted', 'source': dict(source, location=request['compound_id'] + '/selected/terminal_half_life'),
        'context': {'species': 'human', 'matrix': 'plasma', 'route': 'intravenous',
            'study_role': 'independent_parameters', 'independent_of_target': True},
        'prediction_method': {'name': prior['selected_models']['terminal_half_life'], 'version': PRIOR_VERSION,
            'training_overlap': False, 'applicability': {'target_structure_and_scaffold_excluded': True,
                'human_IV_parameter_model': True}}}]


def statistics(rows, *, seed=SEED):
    errors = np.array([r['absolute_ln_error'] for r in rows], float)
    if not len(errors):
        return {'n': 0, 'gmfe': None, 'gmfe_ci95': None, 'within_twofold_n': 0, 'within_twofold_rate': None}
    draw = np.random.default_rng(seed).integers(len(errors), size=(2000, len(errors)))
    return {'n': len(errors), 'gmfe': float(np.exp(errors.mean())),
        'gmfe_ci95': np.exp(np.quantile(errors[draw].mean(axis=1), [.025, .975])).tolist(),
        'within_twofold_n': int((errors <= math.log(2)).sum()),
        'within_twofold_rate': float((errors <= math.log(2)).mean())}


def summarize(records):
    """Always report own denominators AND common-case paired comparisons."""
    output = {}
    for endpoint in ENDPOINTS:
        rows = [r for r in records if r['endpoint'] == endpoint]
        scored = {m: {r['case_id']: r for r in rows if r['method'] == m and r['absolute_ln_error'] is not None}
                  for m in METHODS}
        common = set.intersection(*(set(scored[m]) for m in METHODS))
        pairs = {}
        for other in METHODS[1:]:
            ids = sorted(set(scored[METHODS[0]]) & set(scored[other]))
            left, right = [scored[METHODS[0]][i] for i in ids], [scored[other][i] for i in ids]
            delta = np.array([r['absolute_ln_error'] - l['absolute_ln_error'] for l, r in zip(left, right)])
            draws = np.random.default_rng(SEED).integers(len(ids), size=(2000, len(ids))) if ids else None
            pairs[other] = {'n': len(ids), 'previous': statistics(left), 'candidate': statistics(right),
                'gmfe_ratio': float(np.exp(delta.mean())) if ids else None,
                'ratio_ci95': np.exp(np.quantile(delta[draws].mean(axis=1), [.025, .975])).tolist() if ids else None,
                'improved': int((delta < -1e-8).sum()), 'worse': int((delta > 1e-8).sum())}
        output[endpoint] = {'reference_n': len({r['case_id'] for r in rows if r['reference'] is not None}),
            'own_scorable': {m: statistics(list(scored[m].values())) for m in METHODS},
            'common_all_methods': {m: statistics([scored[m][i] for i in sorted(common)]) for m in METHODS},
            'paired_with_previous': pairs,
            'unscorable_with_reference': {m: sum(r['method'] == m and r['reference'] is not None
                and r['prediction'] is None for r in rows) for m in METHODS}}
    return output


class AccuracyRunner(BaseRunner):
    def __init__(self, project_root=ROOT, *, adapter=None):
        super().__init__(project_root=project_root, adapter=adapter)
        self.default = self.root / 'experiments/blueprint_design/accuracy_refinement_20261001'
        self.source = self.root / 'experiments/blueprint_design/purpose_refinement_20261001'

    def verify(self, pins):
        try:
            return super().verify(pins)
        except (ValueError, OSError) as error:
            raise IntegrityError(str(error)) from error

    def own_run(self, path):
        path = inside(path, self.root)
        if path.parent != self.root / 'experiments/blueprint_design' or not path.name.startswith('accuracy_refinement_'):
            raise ValueError('A project accuracy_refinement_ batch is required')
        return path

    def prepare(self, run=None, workers=14):
        run = self.own_run(run or self.default)
        if type(workers) is not int or not 1 <= workers <= 14:
            raise ValueError('One to fourteen workers required')
        if any((run / n).exists() for n in ('protocol.json', 'state.json', 'work', 'results.json')):
            raise FileExistsError('Preserve existing experiment; run or resume instead')
        old, old_state = self.read(self.source / 'protocol.json'), self.read(self.source / 'state.json')
        if old_state['status'] != 'complete' or old_state['protocol'] != self.ref(self.source / 'protocol.json'):
            raise ValueError('Frozen comparator is incomplete')
        self.verify(old['source_pins']); self.verify(old['code_pins']); self.verify(old_state['report_artifact_pins'])
        prior_path = run / 'priors/predictions.json'
        bundle = self.read(prior_path)
        self.verify({v['path']: v['sha256'] for v in bundle['artifacts'].values()})
        audit = self.checked(bundle['artifacts']['audit'])
        if (bundle.get('version') != PRIOR_VERSION
                or audit.get('version') != bundle['version']
                or bundle.get('input_fingerprint') != prior_digest(audit.get('frozen_input'))
                or audit.get('input_fingerprint') != bundle['input_fingerprint']
                or audit.get('predictions_digest') != prior_digest(bundle['predictions'])
                or audit.get('target_predictions_locked_before_scoring') is not True
                or any(audit.get(key) is not False for key in ('target_observation_files_opened',
                    'calibration_labels_used_for_selection', 'target_labels_used_for_selection'))):
            raise IntegrityError('Rich-prior completion, fingerprint or outcome isolation contract differs')
        self.verify(audit['source_pins'])
        self.verify(audit.get('code_pins', {}))
        if (len(old['cases']) != 100
                or any(len({c[key] for c in old['cases']}) != 100
                       for key in ('case_id', 'compound_id', 'identity_smiles'))
                or set(bundle['predictions']) != {c['compound_id'] for c in old['cases']}):
            raise IntegrityError('Exactly the same one hundred unique identities must be predicted')
        pins = {**old['source_pins'], **old['code_pins'], **old_state['report_artifact_pins']}
        for item in (self.source / 'protocol.json', self.source / 'state.json', prior_path):
            pins.update({self.ref(item)['path']: self.ref(item)['sha256']})
        pins.update({v['path']: v['sha256'] for v in bundle['artifacts'].values()})
        pins.update(audit.get('source_pins', {})); pins.update(audit.get('code_pins', {}))
        plans = {}
        for case in old['cases']:
            cid, compound = case['case_id'], case['compound_id']
            prior = bundle['predictions'][compound]
            if prior['identity_smiles'] != case['identity_smiles']:
                raise ValueError('Prediction identity differs from frozen cohort')
            for key in ('training_scaffold_overlap', 'training_structure_overlap', 'calibration_scaffold_overlap',
                        'calibration_structure_overlap', 'cohort_labels_used', 'target_labels_used'):
                if prior.get(key) is not False:
                    raise ValueError('Missing exclusion guarantee: ' + key)
            reference = old_state['outcomes'][cid]
            outcome = self.checked(reference)
            if outcome.get('case_id') != cid or outcome.get('status') != 'locked' or 'combined' not in outcome.get('methods', {}):
                raise IntegrityError('Comparator outcome identity or completeness differs')
            self.verify(outcome['artifact_pins'])
            pins[reference['path']] = reference['sha256']
            plans[cid] = {'request': old['case_protocols'][cid]['request'], 'prior': prior,
                'prior_ref': self.ref(prior_path), 'previous_outcome': reference}
        code = engine._code_pins(self.root)
        for name in ['scripts/run_accuracy_refinement.py', 'scripts/run_purpose_hundred.py',
                'scripts/run_purpose_refinement.py', 'src/pbpk_kb/blueprint_policy/capacity_guided_design.py',
                'src/pbpk_kb/blueprint_policy/learned_exchange_design.py',
                'src/pbpk_kb/blueprint_learning/rich_parameter_prior.py',
                'src/pbpk_kb/blueprint_policy/hundred_evaluation.py']:
            code[name] = self.ref(name)['sha256']
        protocol = {'version': VERSION, 'created_at': now(), 'cases': old['cases'], 'case_protocols': plans,
            'expected_compounds': 100, 'methods': METHODS, 'workers': workers, 'native_threads': 2,
            'native_timeout_seconds': 180, 'monitor_seconds': 1200, 'directory_limit_bytes': 10*1024**3,
            'source_pins': pins, 'code_pins': code, 'paper_guard': self.paper_pins(),
            'exchange_scale_bounds': [.01, 100.], 'half_target_basis': 'sampled_nca',
            'half_search_budget': 33,
            'half_policy': 'best_valid_NCA_prior_agreement_if_better_than_PS1; no_clinical_matching_claim',
            'selection': 'Scaffold training CV chooses parameter regressors; target labels never choose model or PS',
            'mrt_input_role': 'Direct-ML diagnostic only; simulated MRT follows physically linked Vss/CL',
            'evaluation_scope': 'Aggregate PK labels, not clinical curves; all targets previously viewed development',
            'paper_updated': False, 'decision_layer_expanded': False}
        self.verify(pins); self.verify(code)
        save(run / 'protocol.json', protocol)
        save(run / 'state.json', {'version': VERSION, 'protocol': self.ref(run / 'protocol.json'),
            'status': 'prepared', 'outcomes': {}, 'progress': {}, 'monitor': {},
            'consecutive_infrastructure_errors': 0, 'all_predictions_locked_before_scoring': False,
            'monitor_stopped_before_scoring': False})
        return {'status': 'prepared', 'cases': 100, 'new_native_maximum': 100, 'methods': METHODS}

    def load(self, run, *, locked=False, artifacts=True):
        run = self.own_run(run)
        state = self.read(run / 'state.json'); protocol = self.checked(state['protocol'])
        if state['version'] != VERSION or protocol['version'] != VERSION:
            raise ValueError('Version changed')
        self.verify(protocol['source_pins']); self.verify(protocol['code_pins'])
        if self.paper_pins() != protocol['paper_guard']:
            raise ValueError('Paper changed')
        expected = {c['case_id'] for c in protocol['cases']}
        if (len(protocol['cases']) != 100 or len(expected) != 100
                or set(protocol['case_protocols']) != expected
                or any(len({c[key] for c in protocol['cases']}) != 100
                       for key in ('compound_id', 'identity_smiles'))
                or not set(state['outcomes']) <= expected):
            raise ValueError('Frozen cohort changed')
        for cid, ref in state['outcomes'].items():
            outcome = self.checked(ref)
            if outcome['case_id'] != cid or outcome['protocol'] != state['protocol'] or outcome['status'] != 'locked':
                raise ValueError('Outcome identity or lock changed')
            if set(outcome['methods']) != set(METHODS):
                raise ValueError('Missing method')
            if any(v.get('status') not in {'predicted','failed'} for v in outcome['methods'].values()):
                raise IntegrityError('All methods require terminal prediction or failure status')
            if artifacts:
                self.verify(outcome['artifact_pins'])
        if locked and (state['status'] not in {'predictions_locked', 'complete'}
                or set(state['outcomes']) != expected or not state['all_predictions_locked_before_scoring']
                or not state.get('monitor_stopped_before_scoring')):
            raise ValueError('Every prediction and failure must lock before scoring')
        return protocol, state

    def _case(self, run, protocol, case, monitor):
        from pbpk_kb.blueprint_policy.learned_exchange_design import simulate_learned_design
        cid = case['case_id']; plan = protocol['case_protocols'][cid]
        folder = run / 'work' / cid
        case_receipt = folder / 'case.lock.json'
        if case_receipt.exists():
            locked = self.read(case_receipt)
            outcome = self.checked(locked['outcome']); self.verify(outcome['artifact_pins'])
            if (outcome['protocol'] != monitor.state['protocol'] or outcome['status'] != 'locked'
                    or outcome['case_id'] != cid or set(outcome['methods']) != set(METHODS)
                    or locked['protocol'] != monitor.state['protocol']):
                raise IntegrityError('Uncommitted outcome lock mismatch')
            with monitor.lock:
                monitor.state['outcomes'][cid] = self.ref(folder / 'outcome.json'); monitor.persist()
            return cid
        started = time.monotonic(); folder.mkdir(parents=True, exist_ok=True)
        previous = self.checked(plan['previous_outcome'])['methods']['combined']
        methods = {'previous_combined': previous, 'direct_ml': direct_parameters(plan['prior'], plan['request'])}
        with monitor.lock:
            progress = monitor.state['progress'].setdefault(cid, {'method_receipts': {}})

        def seal_method(method, prediction):
            path = folder / (method + '.json')
            if path.exists():
                raise IntegrityError('Unsealed method output retained; refusing overwrite: ' + method)
            save(path, prediction)
            with monitor.lock:
                progress.setdefault('method_receipts', {})[method] = self.ref(path)
                monitor.persist()
            methods[method] = prediction

        for method, reference in progress.get('method_receipts', {}).items():
            if method not in METHODS[1:3]:
                raise IntegrityError('Unexpected cached prediction method')
            prediction = self.checked(reference)
            if prediction.get('full_trajectory_ref'):
                self.verify({prediction['full_trajectory_ref']['path']: prediction['full_trajectory_ref']['sha256']})
            if prediction.get('status') not in {'predicted', 'failed'}:
                raise IntegrityError('Cached method is not terminal')
            methods[method] = prediction
        if (folder / 'outcome.json').exists():
            # Resume the narrow interruption between the complete outcome and
            # its case receipt using independently state-sealed method files.
            outcome = self.read(folder / 'outcome.json')
            if (set(methods) != set(METHODS) or outcome.get('methods') != methods
                    or outcome.get('case_id') != cid or outcome.get('protocol') != monitor.state['protocol']
                    or outcome.get('status') != 'locked'):
                raise IntegrityError('Unsealed outcome differs from independently locked predictions')
            self.verify(outcome['artifact_pins'])
            save(case_receipt, {'outcome': self.ref(folder/'outcome.json'), 'protocol': monitor.state['protocol']})
            with monitor.lock:
                monitor.state['outcomes'][cid] = self.ref(folder/'outcome.json'); monitor.persist()
            return cid
        reason, infrastructure = None, False
        try:
            monitor.check()
            phys_path = folder / 'physiology.json'
            receipt_path = folder / 'native_receipt.json'
            if receipt_path.exists():
                source = progress.get('native_receipt')
                if source is None:
                    raise IntegrityError('Unsealed native receipt retained; cannot silently reuse')
                receipt = self.checked(source)
                if receipt['request_sha256'] != sha256(folder / 'request.json'):
                    raise IntegrityError('Resumed request changed')
                self.verify(receipt['artifact_pins']); physiology = self.checked(receipt['physiology'])
                request = self.checked(receipt['request'])
            else:
                if (folder / 'native_source').exists():
                    raise IntegrityError('Interrupted native construction retained; no silent retry')
                request = rich_request(plan['request'], plan['prior'], plan['prior_ref'], folder)
                request['assumptions'] = [s for s in request['assumptions']
                    if not s.startswith('No Vss, half-life') and not s.startswith('RF train')]
                request['assumptions'].extend(['Predicted CL/fu/Vss/half-life use frozen training-only scaffold CV.',
                    'No target clinical endpoint enters construction; independently predicted half-life guides PS.'])
                save(folder / 'request.json', request)
                physiology = engine._physiology(request, folder / 'native_source', self.root, monitor.check)
                save(phys_path, physiology)
                save(receipt_path, {'request_sha256': sha256(folder / 'request.json'),
                    'request': self.ref(folder/'request.json'), 'physiology': self.ref(phys_path),
                    'artifact_pins': self.artifacts(folder, excluded=('native_receipt.json',))})
                with monitor.lock:
                    progress['native_receipt'] = self.ref(receipt_path); monitor.persist()
            vss = rich_volume(request, plan['prior'], plan['prior_ref'])
            half = half_evidence(request, plan['prior'], plan['prior_ref'])
            for method in METHODS[1:3]:
                monitor.check()
                if method in methods:
                    continue
                try:
                    if method == 'rich_fixed_exchange':
                        anchor = capacity.anchor_partition(physiology, request, vss)
                        if anchor['status'] == 'blocked':
                            prediction = {'status': 'failed', 'reason': 'capacity_anchor_blocked', 'decision': anchor, 'endpoints': {}}
                        else:
                            prediction = capacity.simulate_design(physiology, request, log_parameters=anchor['log_parameters'])
                            prediction['decision'] = anchor
                    else:
                        prediction = simulate_learned_design(physiology, request, vss, half,
                            ps_bounds=tuple(protocol['exchange_scale_bounds']), fallback=True,
                            target_basis=protocol['half_target_basis'],
                            max_trajectory_calls=protocol['half_search_budget'])
                    trajectory = prediction.pop('full_trajectory', None)
                    if trajectory is not None:
                        path = folder / (method + '.npz')
                        if path.exists():
                            raise IntegrityError('Interrupted trajectory retained; refusing overwrite: ' + method)
                        np.savez_compressed(path, **{k: np.asarray(v) for k, v in trajectory.items()})
                        prediction['full_trajectory_ref'] = self.ref(path)
                    seal_method(method, prediction)
                except IntegrityError:
                    raise
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                    seal_method(method, {'status': 'failed', 'reason': str(error), 'endpoints': {}})
        except (Paused, IntegrityError):
            raise
        except (ValueError, ArithmeticError) as error:
            reason = str(error)
        except Exception as error:
            infrastructure = True; reason = type(error).__name__ + ': ' + str(error)
        for method in METHODS[1:3]:
            if method not in methods:
                seal_method(method, {'status': 'failed', 'reason': reason, 'endpoints': {}})
        outcome = {'case_id': cid, 'status': 'locked', 'protocol': monitor.state['protocol'], 'methods': methods,
            'elapsed_seconds': time.monotonic()-started, 'failure_reason': reason,
            'infrastructure_failure': infrastructure, 'target_observations_used': False, 'locked_at': now(),
            'artifact_pins': self.artifacts(folder, excluded=('outcome.json','case.lock.json'))}
        save(folder / 'outcome.json', outcome)
        save(case_receipt, {'outcome': self.ref(folder/'outcome.json'), 'protocol': monitor.state['protocol']})
        with monitor.lock:
            monitor.state['outcomes'][cid] = self.ref(folder / 'outcome.json')
            progress.update(completed=1, failed=int(bool(reason)))
            monitor.record_result(infrastructure)
        return cid

    def report(self, run=None):
        from pbpk_kb.blueprint_policy.hundred_evaluation import _reference_values, _prediction_value, ENDPOINT_UNITS
        run = self.own_run(run or self.default)
        with (run / '.run.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            protocol, state = self.load(run, locked=True)
            if state['status'] == 'complete':
                self.verify(state['report_artifact_pins']); return self.checked(state['results'])
            records, failures, audits = [], {m: [] for m in METHODS}, []
            for case in protocol['cases']:
                outcome = self.checked(state['outcomes'][case['case_id']])
                # First decode of held-out outcomes in this runner, after the complete barrier.
                reference = _reference_values(self.checked(case['evaluation']), case)
                for method, prediction in outcome['methods'].items():
                    if prediction['status'] != 'predicted':
                        failures[method].append({'case_id': case['case_id'], 'drug_name': case['drug_name'],
                            'reason': prediction.get('reason'), 'decision': prediction.get('decision')})
                    for endpoint in ENDPOINTS:
                        value, missing_reason = _prediction_value(prediction, endpoint)
                        truth = reference.get(endpoint)
                        error = abs(math.log(value/truth)) if truth is not None and value is not None else None
                        records.append({'case_id': case['case_id'], 'drug_name': case['drug_name'], 'method': method,
                            'endpoint': endpoint, 'unit': ENDPOINT_UNITS[endpoint], 'prediction': value,
                            'reference': truth, 'fold_error': math.exp(error) if error is not None else None,
                            'absolute_ln_error': error, 'missing_reason': missing_reason,
                            'reference_kind': 'aggregate_clinical_PK_parameter_not_matched_study',
                            'construction_status': prediction['status'], 'parameter_only': method == 'direct_ml'})
                audits.append({'case_id': case['case_id'], 'drug_name': case['drug_name'],
                    'exchange': outcome['methods']['rich_learned_exchange'].get('learned_exchange_audit'),
                    'elapsed_seconds': outcome['elapsed_seconds']})
            result = {'version': VERSION, 'registered_compounds': 100, 'methods': METHODS,
                'endpoint_metrics': summarize(records), 'failures': failures, 'structure_audits': audits,
                'all_predictions_locked_before_scoring': True, 'clinical_curve_validation_n': 0,
                'paper_updated': False, 'bootstrap': {'unit': 'compound', 'repeats': 2000, 'seed': SEED},
                'limitations': ['Previously inspected development cohort, not new blind validation.',
                    'Only strict identity-matched aggregate CL/Vss/half/MRT are scored; missing labels remain null.',
                    'Direct ML does not generate a PBPK curve and may have inconsistent independent endpoints.',
                    'Up to 33 actual trajectories select valid sampled-NCA agreement with the independently predicted half-life prior, improving on PS=1 when feasible; no clinical-label matching claim.',
                    'No clinical decision reliability or organ exposure validation is claimed.']}
            save(run / 'results.json', result)
            with (run / 'results.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
            self.verify(protocol['source_pins']); self.verify(protocol['code_pins'])
            if self.paper_pins() != protocol['paper_guard']:
                raise ValueError('Paper changed during scoring')
            state.update(status='complete', reported_at=now(), results=self.ref(run / 'results.json'),
                report_artifact_pins={str((run/n).relative_to(self.root)): sha256(run/n) for n in ('results.json', 'results.csv')})
            save(run / 'state.json', state)
            return result

    def recorded_process(self, run):
        import psutil
        if not (run / 'pid.json').exists(): return None
        receipt = self.read(run / 'pid.json')
        script = str(self.root / 'scripts/run_accuracy_refinement.py')
        if receipt['script'] != script or receipt['run'] != str(run): raise ValueError('Unrelated PID receipt')
        try:
            proc = psutil.Process(receipt['pid'])
            if proc.create_time() != receipt['create_time'] or proc.status() == psutil.STATUS_ZOMBIE: return None
            if proc.uids().real != os.getuid(): raise ValueError('PID owner changed')
            if Path(proc.cwd()).resolve() != self.root: raise ValueError('PID directory changed')
            args = proc.cmdline()
            if script not in args or args[args.index('--run')+1] != str(run): raise ValueError('PID task changed')
            return {'pid': receipt['pid'], 'running': True}
        except psutil.NoSuchProcess:
            return None

    def background(self, run=None):
        import psutil
        run = self.own_run(run or self.default); protocol, state = self.load(run)
        if state['status'] == 'complete': return {'status': 'complete'}
        if (run / 'PAUSE').exists(): return {'status': 'paused'}
        with (run / '.launch.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            live = self.recorded_process(run)
            if live: return {'status': 'already_running', **live}
            script = self.root / 'scripts/run_accuracy_refinement.py'
            cache = self.root / 'runtime/cache/accuracy_refinement'; cache.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                MKL_NUM_THREADS='2', TMPDIR=str(cache))
            with (run / 'run.log').open('ab') as log:
                child = subprocess.Popen([sys.executable, '-B', str(script), 'run', '--run', str(run), '--report-after-run'],
                    cwd=self.root, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, env=env)
            save(run / 'pid.json', {'pid': child.pid, 'create_time': psutil.Process(child.pid).create_time(),
                'script': str(script), 'run': str(run), 'started_at': now()})
            return {'status': 'started', 'pid': child.pid, 'workers': protocol['workers']}

    def status(self, run=None):
        run = self.own_run(run or self.default); protocol, state = self.load(run, artifacts=False)
        return {'status': state['status'], 'cases_locked': len(state['outcomes']), 'cases_total': 100,
            'failed_constructions': {m: sum(self.checked(r)['methods'][m]['status'] != 'predicted'
                for r in state['outcomes'].values()) for m in METHODS},
            'monitor': state['monitor'], 'process': self.recorded_process(run),
            'pause_reason': state.get('pause_reason'), 'results': state.get('results'), 'paper_updated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'status', 'pause', 'resume', 'report'])
    parser.add_argument('--run', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--report-after-run', action='store_true')
    args = parser.parse_args(); runner = AccuracyRunner()
    if args.command == 'prepare': result = runner.prepare(args.run, args.workers)
    elif args.command == 'run': result = runner.background(args.run) if args.background else runner.execute(args.run)
    elif args.command == 'resume': result = runner.resume(args.run, background=args.background)
    else: result = getattr(runner, args.command)(args.run)
    if args.command in ('run', 'resume') and not args.background and args.report_after_run and result['status'] in ('predictions_locked', 'complete'):
        runner.report(args.run); result = runner.status(args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__': main()
