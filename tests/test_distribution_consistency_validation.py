"""The fallback runner conserves frozen inputs and separates coverage from error."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_distribution_consistency_validation.py'
SPEC = importlib.util.spec_from_file_location('distribution_consistency_runner_test', SCRIPT)
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)


def failed_entry():
    grid = np.linspace(math.log(.01), math.log(100.), 17)
    grid[np.abs(grid) < 1e-12] = 0.
    return {'status': 'failed', 'reason': 'no_usable_sampled_NCA_model', 'curve': None, 'endpoints': None,
        'design_audit': {'version': 'learned-exchange-design-2', 'target_basis': 'sampled_nca',
            'ps_bounds': [.01, 100.], 'grid_points': 17, 'trajectory_budget': 33, 'search_attempts': 17,
            'target_observations_used': False, 'linear_trajectory_calls': 0,
            'reasons': [r.FLOOR_REASON], 'trials': [
            {'status': 'failed', 'reason': r.FLOOR_REASON, 'trajectory_attempted': False,
             'log_ps': float(x), 'ps_multiplier': float(math.exp(x))} for x in grid]}}


@pytest.mark.parametrize('fault', ['success', 'mixed_reason', 'trial_attempted', 'trial_success', 'no_trials'])
def test_fallback_excludes_other_failures_or_successes(fault):
    entry = failed_entry()
    if fault == 'success': entry['status'] = 'predicted'
    elif fault == 'mixed_reason': entry['design_audit']['reasons'].append('unrelated_failure')
    elif fault == 'trial_attempted': entry['design_audit']['trials'][0]['trajectory_attempted'] = True
    elif fault == 'trial_success': entry['design_audit']['trials'][0]['status'] = 'predicted'
    else: entry['design_audit']['trials'] = []
    assert not r.floor_failure(entry)


def test_only_exclusive_pretrajectory_floor_failure_is_eligible():
    assert r.floor_failure(failed_entry())


def task_fixture(monkeypatch, blocked=False):
    request = {'compound_id': 'c', 'human': {'weight_kg': 70}, 'dosing': {'dose_mg': 20},
        'target': {'window_end_h': 24}, 'partition_method': 'PKSimStandard',
        'evidence': [{'parameter': 'total_clearance', 'value': 1.2, 'basis': 'predicted'}],
        'evidence_ledger': {'path': 'frozen_ledger.json'}}
    sidecars = {'vss': [{'value': .5}], 'half': [{'value': 2.}]}
    core = {'design_version': 'learned-exchange-design-2', 'code_pins': {}, 'parameter_prior': {'sha256': 'p'*64}}
    key = r.retained.key_for(request, sidecars, core)
    original = {'key': key, 'case_id': 'case', 'request': request, 'sidecars': sidecars,
        'core': core, 'construction_blocked': ['pKa count exceeds 3'] if blocked else None}
    entry = failed_entry()
    if blocked:
        entry.update(reason='construction_blocked', design_audit={'reasons': original['construction_blocked']})
    entry.update(request=deepcopy(request), request_sha256=r.value.model._digest(request),
        target_observations_used=False, observations_used=False)
    outcome = {'key': key, 'entry': entry, 'locked_at': 'frozen-before-scores'}
    monkeypatch.setattr(r.value.old, 'checked', lambda receipt: deepcopy(outcome))
    monkeypatch.setattr(r.value.old, 'collect_refs', lambda obj, pins: None)
    receipt = {'path': 'source_outcome.json', 'sha256': 's'*64}
    task = r.candidate_task(original, receipt, 'new-design-1', {'path': 'module.py', 'sha256': 'a'*64})
    return original, outcome, task


def test_new_core_keeps_exact_parameter_inputs_and_old_request(monkeypatch):
    original, _, task = task_fixture(monkeypatch)
    assert task['key'] != original['key']
    assert task['request'] == original['request'] and task['sidecars'] == original['sidecars']
    assert task['core']['parameter_prior'] == original['core']['parameter_prior']
    assert task['core']['base_design_version'] == original['core']['design_version']
    assert task['core']['design_version'] == 'new-design-1'
    assert task['fallback_eligible'] is True
    assert original['core']['design_version'] == 'learned-exchange-design-2'


def test_helper_change_is_bound_into_scientific_key(monkeypatch):
    original, _, task = task_fixture(monkeypatch)
    changed = r.candidate_task(original, task['source_prediction_receipt'], 'new-design-1',
        {'path': 'module.py', 'sha256': 'a'*64}, {'helper.py': 'b'*64})
    assert changed['key'] != task['key']
    assert changed['core']['code_pins']['helper.py'] == 'b'*64


@pytest.mark.parametrize('fault', ['ledger', 'sidecar', 'core', 'source_key', 'request', 'observation_flag'])
def test_source_receipt_requires_complete_request_not_only_semantic_fingerprint(monkeypatch, fault):
    _, outcome, task = task_fixture(monkeypatch)
    if fault == 'ledger': task['request']['evidence_ledger']['path'] = 'other_ledger.json'
    elif fault == 'sidecar': task['source_task']['sidecars']['vss'][0]['value'] = 3
    elif fault == 'core': task['source_task']['core']['parameter_prior']['sha256'] = 'other'
    elif fault == 'source_key': outcome['key'] = 'different'
    elif fault == 'request': outcome['entry']['request']['human']['weight_kg'] = 90
    else: outcome['entry']['target_observations_used'] = True
    monkeypatch.setattr(r.value.old, 'checked', lambda receipt: deepcopy(outcome))
    with pytest.raises(ValueError): r.source_entry(task)


class Monitor:
    def __init__(self):
        self.state = {'outcomes': {}}
        self.lock, self.halted = threading.Lock(), threading.Event()
    def check(self): pass
    def update(self, **values): self.state.update(values)


def test_pka_blocker_is_preserved_without_loading_module(monkeypatch, tmp_path):
    _, _, task = task_fixture(monkeypatch, blocked=True)
    def forbidden(name): raise AssertionError('A construction blocker must not start the distribution module')
    monkeypatch.setattr(r.importlib, 'import_module', forbidden)
    (tmp_path/'models').mkdir(); monitor = Monitor()
    assert r.solve(task, tmp_path, monitor) == task['key']
    saved = r.value.old.read(tmp_path/'models'/f"{task['key'][:20]}.json")
    assert saved['entry']['status'] == 'failed'
    assert saved['entry']['reason'] == 'construction_blocked'
    assert saved['reused'] is True and saved['new_native_calls'] == 0
    assert saved['new_linear_trajectory_calls'] == 0


def test_fallback_receives_only_frozen_prediction_and_observation_free_inputs(monkeypatch, tmp_path):
    _, outcome, task = task_fixture(monkeypatch)
    calls = []
    def design(request, vss, half, output, root, *, source_entry):
        calls.append((request, vss, half, source_entry))
        result = deepcopy(source_entry)
        result.update(native_solver_calls=1, linear_trajectory_calls=17,
            distribution_audit={'status': 'schmitt_attempt_failed'})
        return result
    monkeypatch.setattr(r.importlib, 'import_module', lambda name: SimpleNamespace(VERSION='new-design-1', run_design=design))
    (tmp_path/'models').mkdir(); monitor = Monitor()
    r.solve(task, tmp_path, monitor)
    assert len(calls) == 1 and calls[0][3] == outcome['entry']
    assert calls[0][:3] == (task['request'], task['sidecars']['vss'], task['sidecars']['half'])
    saved = r.value.old.read(tmp_path/'models'/f"{task['key'][:20]}.json")
    assert saved['new_native_calls'] == 1 and saved['new_linear_trajectory_calls'] == 17
    assert saved['entry']['target_observations_used'] is False
    assert 'reference' not in calls[0][3]


def test_completed_recovery_never_reruns_native(monkeypatch, tmp_path):
    _, outcome, task = task_fixture(monkeypatch)
    entry = deepcopy(outcome['entry']); entry.update(native_solver_calls=1, linear_trajectory_calls=17)
    folder = tmp_path/'models'/task['key'][:20]; folder.mkdir(parents=True)
    r.save(folder/'completion.json', {'key': task['key'], 'candidate': r.CANDIDATE,
        'exact_input_digest': r.source_runner.exact_input_digest(task),
        'source_prediction_receipt': task['source_prediction_receipt'], 'entry': entry})
    monkeypatch.setattr(r.importlib, 'import_module', lambda name: pytest.fail('A completed task must not run again'))
    r.solve(task, tmp_path, Monitor())
    saved = r.value.old.read(tmp_path/'models'/f"{task['key'][:20]}.json")
    assert saved['new_native_calls'] == 1  # Total batch work, not an additional recovered call.


def test_complete_module_result_recovers_after_interruption_before_completion(monkeypatch, tmp_path):
    _, outcome, task = task_fixture(monkeypatch)
    folder = tmp_path/'models'/task['key'][:20]; folder.mkdir(parents=True)
    binding = {'key': task['key'], 'candidate': r.CANDIDATE,
        'exact_input_digest': r.source_runner.exact_input_digest(task),
        'source_prediction_receipt': task['source_prediction_receipt'], 'interrupted_attempts': []}
    r.save(folder.parent/(folder.name+'.invocation.json'), binding)
    entry = deepcopy(outcome['entry']); entry.update(version='new-design-1', native_solver_calls=1, linear_trajectory_calls=17)
    r.save(folder/'result.json', entry)
    def forbidden(*args, **kwargs): pytest.fail('Already completed native work must not rerun')
    monkeypatch.setattr(r.importlib, 'import_module', lambda name: SimpleNamespace(VERSION='new-design-1', run_design=forbidden))
    r.solve(task, tmp_path, Monitor())
    saved = r.value.old.read(tmp_path/'models'/f"{task['key'][:20]}.json")
    assert saved['new_native_calls'] == 1 and (folder/'completion.json').is_file()


@pytest.mark.parametrize('process', [
    {'status': 'blocked'}, {'stop_reason': 'timeout'}, {'error': 'missing runtime'}, {'returncode': 1}])
def test_infrastructure_failures_are_not_clinical_failures(monkeypatch, process):
    monkeypatch.setattr(r.value.old, 'checked', lambda ref: {'process': process})
    assert r.infrastructure_failure({'reason': 'native_source_failed', 'execution_ref': {}}) is not None


def test_successful_process_with_scientific_readback_failure_stays_scientific(monkeypatch):
    monkeypatch.setattr(r.value.old, 'checked', lambda ref: {'process': {'status': 'completed', 'returncode': 0}})
    assert r.infrastructure_failure({'reason': 'native_source_failed', 'execution_ref': {}}) is None


def test_native_infrastructure_failure_pauses_without_locking_scientific_outcome(monkeypatch, tmp_path):
    _, outcome, task = task_fixture(monkeypatch)
    def checked(receipt):
        return {'process': {'status': 'failed', 'returncode': 1}} if receipt.get('path') == 'execution.json' else deepcopy(outcome)
    monkeypatch.setattr(r.value.old, 'checked', checked)
    def design(request, vss, half, output, root, *, source_entry):
        return {**source_entry, 'reason': 'native_source_failed', 'execution_ref': {'path': 'execution.json'},
            'native_solver_calls': 1, 'linear_trajectory_calls': 0}
    monkeypatch.setattr(r.importlib, 'import_module', lambda name: SimpleNamespace(VERSION='new-design-1', run_design=design))
    (tmp_path/'models').mkdir(); monitor = Monitor()
    with pytest.raises(r.value.old.Pause, match='infrastructure failure'): r.solve(task, tmp_path, monitor)
    assert not monitor.state['outcomes']
    assert (tmp_path/'models'/f"{task['key'][:20]}.infrastructure.json").is_file()


def test_recovery_rejects_different_source_or_core(monkeypatch):
    _, outcome, task = task_fixture(monkeypatch)
    completion = {'key': task['key'], 'candidate': r.CANDIDATE,
        'exact_input_digest': r.source_runner.exact_input_digest(task),
        'source_prediction_receipt': {'path': 'other.json'}, 'entry': outcome['entry']}
    with pytest.raises(ValueError, match='Recovery receipt'): r.validate_completion(completion, task)


def test_runner_scope_and_lock_block_observation_access(monkeypatch, tmp_path):
    assert r.own(r.DEFAULT) == r.DEFAULT
    with pytest.raises(ValueError): r.own(r.SOURCE)
    monkeypatch.setattr(r, 'own', lambda path: path)
    monkeypatch.setattr(r.value, 'verify', lambda protocol: None)
    r.save(tmp_path/'protocol.json', {'tasks': {'one': {}}})
    r.save(tmp_path/'state.json', {'protocol': r.value.old.ref(tmp_path/'protocol.json'),
        'all_predictions_locked_before_scoring': False, 'outcomes': {}})
    monkeypatch.setattr(r.base, 'ClinicalStore', lambda *args: pytest.fail('Target observations opened before lock'))
    with pytest.raises(ValueError, match='Lock every prediction'): r.report(tmp_path)


def comparison_fixture():
    cases = [{'case_id': str(i), 'compound_id': str(i), 'cluster_id': str(i),
        'validation_stratum': 'old_20' if i == 0 else 'new_compounds'} for i in range(3)]
    def data(recovered):
        rows = []
        for case in cases:
            for condition in r.CONDITIONS:
                k = int(case['case_id']); scored = k == 0 or (recovered and k == 1)
                rows.append({'case_id': case['case_id'], 'condition': condition,
                    'attempt_status': 'predicted' if scored else 'failed',
                    'curve_metrics': {'gmfe': (2 if k == 0 else 9) if scored else None},
                    'reference_family': 'observation_derived', 'endpoints': {
                        endpoint: {'reference': None if endpoint == 'vss' else 1,
                            'error': None if not scored or endpoint == 'vss' else (.0 if endpoint == 'tmax' else 2)}
                        for endpoint in r.value.UNITS}})
        return {'units': rows}
    return cases, data(True), data(False)


def test_recovered_case_coverage_does_not_change_common_case_accuracy_or_denominator():
    cases, after, before = comparison_fixture()
    result = r.comparison_statistics(after, before, {'cases': cases, 'changed_case_ids': ['0', '1', '2']})
    a = result['candidate_over_source_qualified']['all']['available']
    assert a['requested'] == 3 and a['candidate_scored'] == 2 and a['source_scored'] == 1
    assert a['common_candidate_curve']['estimate'] == a['common_source_curve']['estimate'] == 2
    assert a['common_curve_ratio']['estimate'] == 1 and a['common_curve_ratio']['requested_drugs'] == 3
    assert a['newly_scored_case_ids'] == ['1'] and a['lost_scored_case_ids'] == []
    assert a['newly_scored_curve']['estimate'] == pytest.approx(9)
    vss = a['endpoints']['observation_derived']['vss']
    assert vss['reference_available'] == vss['candidate_scored'] == vss['source_scored'] == 0
    assert a['endpoints']['observation_derived']['tmax']['mean_paired_error_difference_h'] == 0
    assert len(result['original_evidence_group_case_ids']) == 3
    assert result['denominator_reduced_for_benefit_claim'] is False


def test_report_comparison_rejects_dropped_failure_or_changed_reference():
    cases, after, before = comparison_fixture()
    with pytest.raises(ValueError, match='retain every requested'):
        r.comparison_statistics({'units': after['units'][:-1]}, before, {'cases': cases, 'changed_case_ids': []})
    after['units'][0]['endpoints']['auc']['reference'] = 2
    with pytest.raises(ValueError, match='scalar references differ'):
        r.comparison_statistics(after, before, {'cases': cases, 'changed_case_ids': []})
