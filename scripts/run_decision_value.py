#!/usr/bin/env python3
"""Evaluate purpose decisions using the same frozen two-candidate PBPK pool.

No native simulation is repeated.  Reused simulations are not new independent
experiments and their cost cannot be claimed as decision-layer compute savings.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

from run_purpose_refinement import RefinementRunner
from pbpk_kb.blueprints.source_workflow import save, sha256
from pbpk_kb.blueprint_policy.purpose_decision_value import decide_value
from pbpk_kb.blueprint_policy.decision_value_evaluation import (
    ENDPOINTS, METHODS, checked, digest, evaluate_decisions)

VERSION = 'decision-value-runner-1'
DEFAULT = ROOT / 'experiments/blueprint_design/decision_value_20261001'


def now():
    return datetime.now(timezone.utc).isoformat()


class Runner:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.source = self.root / 'experiments/blueprint_design/purpose_refinement_20261001'

    def path(self, run):
        path = Path(run).resolve()
        if path.parent != self.root / 'experiments/blueprint_design' or not path.name.startswith('decision_value_'):
            raise ValueError('Use a project decision_value_ batch')
        return path

    def ref(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Source escaped project')
        return {'path': str(path.relative_to(self.root)), 'sha256': sha256(path)}

    def verify(self, pins):
        for name, value in pins.items():
            if self.ref(self.root / name)['sha256'] != value:
                raise ValueError('Source or code changed: ' + name)

    def read(self, path):
        return json.loads(Path(path).read_text())

    def load(self, run):
        run = self.path(run)
        state = self.read(run / 'state.json')
        protocol = checked(state['protocol'], self.root)
        if protocol['version'] != VERSION:
            raise ValueError('Runner version mismatch')
        self.verify(protocol['source_pins']); self.verify(protocol['code_pins'])
        if RefinementRunner(self.root).paper_pins() != protocol['paper_guard']:
            raise ValueError('Paper changed')
        return protocol, state

    def prepare(self, run):
        run = self.path(run)
        if any((run / n).exists() for n in ('protocol.json', 'state.json', 'decisions.json', 'results.json')):
            raise FileExistsError('Preserve the existing frozen decision batch')
        oldrunner = RefinementRunner(self.root)
        old, oldstate = oldrunner.load(self.source, locked=True)
        prior_path = run / 'reference_prior.json'
        from pbpk_kb.blueprint_learning.endpoint_reference_prior import fit_reference_prior
        if not prior_path.is_file():
            raise ValueError('Prepare requires a completed independent reference artifact')
        prior = fit_reference_prior(self.root, prior_path)
        self.verify(prior['source_pins'])
        if prior['body_weight_kg'] != 70:
            raise ValueError('Unexpected calibration reference body-weight scaling')
        if set(prior['targets']) != {c['compound_id'] for c in old['cases']}:
            raise ValueError('Reference distributions must cover exactly the frozen cohort')
        pins = dict(old['source_pins']); pins.update(old['code_pins'])
        pins.update(prior['source_pins'])
        requests = {}
        for case in old['cases']:
            cid = case['case_id']
            outcome_ref = oldstate['outcomes'][cid]
            outcome = checked(outcome_ref, self.root)
            phase_ref = outcome['phases']['nominal']
            phase = checked(phase_ref, self.root)
            request = phase['request']
            if prior['targets'][case['compound_id']]['identity_smiles'] != case['identity_smiles']:
                raise ValueError('Calibration-reference target identity mismatch')
            if request['compound_id'] != case['compound_id'] or request['human']['weight_kg'] != 70:
                raise ValueError('Reference scaling and requested identity/human differ')
            requests[cid] = deepcopy(request)
            for ref in (outcome_ref, phase_ref): pins[ref['path']] = ref['sha256']
            pins.update(outcome['artifact_pins'])
        for path in (self.source / 'protocol.json', prior_path):
            ref = self.ref(path); pins[ref['path']] = ref['sha256']
        code_names = ['scripts/run_decision_value.py',
            'src/pbpk_kb/blueprint_policy/purpose_decision_value.py',
            'src/pbpk_kb/blueprint_policy/decision_value_evaluation.py',
            'src/pbpk_kb/blueprint_learning/endpoint_reference_prior.py']
        code = {n: self.ref(self.root / n)['sha256'] for n in code_names}
        protocol = {'version': VERSION, 'created_at': now(), 'expected_compounds': 100,
            'cases': old['cases'], 'requests': requests, 'candidate_outcomes': oldstate['outcomes'],
            'methods': list(METHODS), 'endpoints': list(ENDPOINTS),
            'candidate_pool': ['parameter_only', 'combined'], 'reference_prior': self.ref(prior_path),
            'source_pins': pins, 'code_pins': code, 'paper_guard': old['paper_guard'],
            'calibration_neighbors': 20, 'minimum_calibration_neighbors': 10,
            'decision_objective': 'minimize expected absolute natural log error of the requested primary endpoint',
            'fixed_rule_fallback': 'Same candidate feasibility and legal fallback for all methods',
            'ranking_rule': 'fixed methods: 1-max_training_similarity; other methods: expected_log_error',
            'tie_tolerance': 1e-8, 'review_threshold': .5, 'large_error_threshold_fold': 2,
            'coverage_levels': [.2, .4, .6, .8, 1.], 'bootstrap_repeats': 2000, 'seed': 20261001,
            'new_native_runs': 0, 'reused_native_budget': old['maximum_new_native_runs'],
            'purpose_contract': '400 different primary-purpose requests; each selects one complete model, never a per-endpoint hybrid',
            'claims_not_tested': ['benefit of acquiring another source', 'clinical safety', 'prospective validation', 'compute savings'],
            'target_labels_used_for_policy_fit': False, 'development_cohort_previously_inspected': True}
        self.verify(pins); self.verify(code)
        save(run / 'protocol.json', protocol)
        save(run / 'state.json', {'status': 'prepared', 'protocol': self.ref(run / 'protocol.json'), 'completed_decisions': 0})
        return {'status': 'prepared', 'requests': 400, 'decisions': 2000, 'new_native_runs': 0}

    def run(self, run):
        run = self.path(run)
        with (run / '.run.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            protocol, state = self.load(run)
            if state['status'] in {'decisions_locked', 'complete'}:
                checked(state['decisions'], self.root)
                return {'status': state['status'], 'reused_locked_decisions': True}
            if (run / 'decisions.json').exists():
                raise ValueError('Uncommitted decisions retained; use a new batch')
            prior = checked(protocol['reference_prior'], self.root)
            records, started = [], time.perf_counter()
            state.update(status='deciding', started_at=now()); save(run / 'state.json', state)
            for case in protocol['cases']:
                cid = case['case_id']; request = protocol['requests'][cid]
                target = prior['targets'][case['compound_id']]
                outcome = checked(protocol['candidate_outcomes'][cid], self.root)
                pool = {m: outcome['methods'][m] for m in protocol['candidate_pool']}
                distributions = {ep: {'unit': data['unit'],
                    'local_log_reference_values': data['local']['reference_values_log'] if data['local']['n'] >= 10 else [],
                    'global_log_reference_values': data['global']['reference_values_log'] if data['global']['n'] >= 10 else [],
                    'role': 'training_reference_distribution',
                    'data_role': 'independent_calibration_not_predictor_fit',
                    'source': protocol['reference_prior'], 'compound_id': case['compound_id']}
                    for ep, data in target['endpoints'].items()}
                for endpoint in ENDPOINTS:
                    purpose_request = deepcopy(request)
                    purpose_request['target']['primary_endpoint'] = endpoint
                    for method in METHODS:
                        t = time.perf_counter()
                        decision = decide_value(purpose_request, pool, distributions, primary_endpoint=endpoint, method=method)
                        elapsed = time.perf_counter() - t
                        selected = decision['selected_candidate']
                        selected_prediction = pool[selected] if selected is not None else None
                        value = (selected_prediction['endpoints'][endpoint]['value'] if selected_prediction is not None else None)
                        input_score = 1 - target['max_training_similarity']
                        ranking_score = input_score if method in {'fixed_capacity', 'fixed_native'} else decision['expected_log_error']
                        records.append({'case_id': cid, 'compound_id': case['compound_id'], 'endpoint': endpoint,
                            'method': method, 'status': decision['status'], 'action': decision['action'],
                            'selected_candidate': selected, 'prediction_value': value,
                            'selected_prediction_sha256': digest(selected_prediction) if selected_prediction else None,
                            'whole_model_source': protocol['candidate_outcomes'][cid],
                            'risk': decision['risk'], 'expected_log_error': decision['expected_log_error'],
                            'ranking_score': ranking_score, 'input_only_score': input_score,
                            'review_required': decision['review_required'], 'reasons': decision['reasons'],
                            'candidate_assessments': decision['candidate_assessments'],
                            'conditional_fu_assessment': outcome['conditional_fu_stability'].get(selected, {}).get(endpoint),
                            'input_gate_status': decision['input_gate']['status'],
                            'input_gate_reasons': decision['input_gate']['reasons'],
                            'decision_seconds': elapsed, 'clinical_validation': False})
                state.update(completed_decisions=len(records), updated_at=now()); save(run / 'state.json', state)
            self.verify(protocol['source_pins']); self.verify(protocol['code_pins'])
            artifact = {'version': VERSION, 'all_decisions_locked_before_scoring': True,
                'locked_at': now(), 'records': records, 'records_sha256': digest(records),
                'elapsed_seconds': time.perf_counter() - started, 'new_native_runs': 0}
            save(run / 'decisions.json', artifact)
            state.update(status='decisions_locked', decisions=self.ref(run / 'decisions.json'), locked_at=artifact['locked_at'])
            save(run / 'state.json', state)
            return {'status': state['status'], 'records': len(records), 'seconds': artifact['elapsed_seconds']}

    def report(self, run):
        run = self.path(run)
        with (run / '.run.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            p, s = self.load(run)
            if s['status'] == 'complete':
                self.verify(s['report_pins'])
                return {'status': 'complete', 'results': s['results']}
            if s['status'] != 'decisions_locked':
                raise ValueError('Cannot score before every purpose decision is locked')
            locked = checked(s['decisions'], self.root)
            result = evaluate_decisions(p, locked, self.root)
            result['protocol'] = s['protocol']; result['decision_artifact'] = s['decisions']
            result['run_seconds'] = locked['elapsed_seconds']; result['new_native_runs'] = 0
            save(run / 'results.json', result)
            rows = result['records']; keys = list(rows[0])
            with (run / 'results.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
                for row in rows:
                    writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
            self.verify(p['source_pins']); self.verify(p['code_pins'])
            if RefinementRunner(self.root).paper_pins() != p['paper_guard']:
                raise ValueError('Paper changed during evaluation')
            s.update(status='complete', results=self.ref(run / 'results.json'), reported_at=now(),
                report_pins={str((run / name).relative_to(self.root)): sha256(run / name) for name in ('results.json', 'results.csv')})
            save(run / 'state.json', s)
            return {'status': 'complete', 'results': s['results'], 'requests': 400, 'decisions': 2000}

    def status(self, run):
        p, s = self.load(run)
        return {'status': s['status'], 'decisions': s['completed_decisions'], 'total_decisions': 2000,
                'results': s.get('results'), 'new_native_runs': 0, 'paper_updated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'report', 'status'])
    parser.add_argument('--run', type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = getattr(Runner(), args.command)(args.run)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
