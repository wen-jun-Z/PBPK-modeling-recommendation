"""Joint design covers all requests; fallback selection and scoring stay sealed."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_joint_partition_validation.py'
SPEC = importlib.util.spec_from_file_location('joint_partition_runner_test', SCRIPT)
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)


def task_fixture(monkeypatch, status='predicted', blocked=False):
    request = {'compound_id': 'c', 'human': {'weight_kg': 70}, 'dosing': {'dose_mg': 20},
        'target': {'window_end_h': 24}, 'partition_method': 'PKSimStandard',
        'evidence': [{'parameter': 'total_clearance', 'value': 1.2, 'basis': 'predicted'}],
        'evidence_ledger': {'path': 'frozen_ledger.json'}}
    sidecars = {'vss': [{'value': .5}], 'half': [{'value': 2.}]}
    core = {'design_version': 'learned-exchange-design-2', 'code_pins': {}, 'parameter_prior': {'sha256': 'p'*64}}
    key = r.retained.key_for(request, sidecars, core)
    original = {'key': key, 'case_id': 'case', 'request': request, 'sidecars': sidecars,
        'core': core, 'construction_blocked': ['pKa count exceeds 3'] if blocked else None}
    grid = np.linspace(math.log(.01), math.log(100.), 17); grid[np.abs(grid)<1e-12] = 0.
    entry = {'status': status, 'request': deepcopy(request), 'request_sha256': r.value.model._digest(request),
        'observations_used': False, 'target_observations_used': False,
        'curve': None if status == 'failed' else {'mock_curve': True},
        'endpoints': None if status == 'failed' else {'observations_used': False},
        'native_source_ref': {'path': 'native.json'}, 'execution_ref': {'path': 'execution.json'},
        'physiology_ref': {'path': 'physiology.json'}}
    if status == 'failed':
        entry.update(reason='no_usable_sampled_NCA_model', design_audit={
            'version': 'learned-exchange-design-2', 'target_basis': 'sampled_nca', 'ps_bounds': [.01, 100.],
            'grid_points': 17, 'trajectory_budget': 33, 'search_attempts': 17, 'target_observations_used': False,
            'linear_trajectory_calls': 0, 'reasons': [r.infra.FLOOR_REASON],
            'trials': [{'status': 'failed', 'reason': r.infra.FLOOR_REASON, 'trajectory_attempted': False,
                'log_ps': float(x), 'ps_multiplier': math.exp(x)} for x in grid]})
    if blocked: entry.update(status='failed', reason='construction_blocked', design_audit={'reasons': ['pKa count exceeds 3']})
    native = {'request_sha256': r.value.model._digest(request), 'partition_method': 'PKSimStandard',
        'status': 'predicted', 'execution_ref': entry['execution_ref'], 'adapter_sha256': r.sha256(Path(r.value.model.__file__))}
    outcome = {'key': key, 'entry': entry, 'locked_at': 'frozen-before-scores'}
    sources = {'outcome.json': outcome, 'native.json': native, 'physiology.json': {}}
    monkeypatch.setattr(r.value.old, 'checked', lambda receipt: deepcopy(sources[receipt['path']]))
    monkeypatch.setattr(r.value.old, 'collect_refs', lambda obj, pins: None)
    module = SimpleNamespace(__file__=str(SCRIPT), VERSION='joint-test-1', CODE_PINS={'helper.py': 'h'*64})
    task = r.candidate_task(original, {'path': 'outcome.json', 'sha256': 's'*64}, module)
    return original, outcome, task, sources, module


@pytest.mark.parametrize('status', ['predicted', 'failed'])
def test_all_nonblocked_sources_get_new_joint_design(monkeypatch, status):
    original, _, task, _, _ = task_fixture(monkeypatch, status)
    assert task['joint_design_required'] is True
    assert task['conservative_use_joint'] is (status == 'failed')
    assert task['key'] != original['key'] and task['request'] == original['request']
    assert task['sidecars'] == original['sidecars'] and task['core']['parameter_prior'] == original['core']['parameter_prior']
    assert task['core']['code_pins']['helper.py'] == 'h'*64


def test_pka_blocker_stays_failed_without_joint_call(monkeypatch):
    _, _, task, _, _ = task_fixture(monkeypatch, blocked=True)
    assert task['joint_design_required'] is False and task['conservative_use_joint'] is False


@pytest.mark.parametrize('fault', ['formula', 'request', 'adapter', 'failed_native'])
def test_standard_native_source_must_be_exact(monkeypatch, fault):
    original, _, task, sources, module = task_fixture(monkeypatch)
    native = sources['native.json']
    if fault == 'formula': native['partition_method'] = 'Schmitt'
    elif fault == 'request': native['request_sha256'] = 'different'
    elif fault == 'adapter': native['adapter_sha256'] = 'different'
    else: native['status'] = 'failed'
    with pytest.raises(ValueError, match='exact frozen Standard'):
        r.candidate_task(original, task['source_prediction_receipt'], module)


def test_conservative_policy_is_source_diagnostic_not_joint_quality(monkeypatch):
    original, outcome, task, sources, module = task_fixture(monkeypatch, status='failed')
    assert task['conservative_use_joint']
    outcome['entry']['design_audit']['reasons'] = ['other_failure']
    changed = r.candidate_task(original, task['source_prediction_receipt'], module)
    assert changed['joint_design_required'] and not changed['conservative_use_joint']


class Monitor:
    def __init__(self):
        self.state = {'outcomes': {}}; self.lock = threading.Lock(); self.halted = threading.Event()
    def check(self): pass
    def update(self, **values): self.state.update(values)


def test_successful_source_can_change_without_floor_only_assertion(monkeypatch, tmp_path):
    _, outcome, task, _, _ = task_fixture(monkeypatch)
    changed = deepcopy(outcome['entry']); changed['curve'] = {'new_joint_curve': True}
    changed.update(native_solver_calls=0, linear_trajectory_calls=33)
    monkeypatch.setattr(r, 'invoke_design', lambda task, folder, original: deepcopy(changed))
    monkeypatch.setattr(r.value, '_native_fields', lambda entry: None)
    (tmp_path/'models').mkdir(); monitor = Monitor(); r.solve(task, tmp_path, monitor)
    item = r.value.old.read(tmp_path/'models'/f"{task['key'][:20]}.json")
    assert item['entry']['curve'] == changed['curve'] and item['reused'] is False
    assert item['new_native_calls'] == 0 and item['new_linear_trajectory_calls'] == 33


@pytest.mark.parametrize('native,linear', [(1, 10), (0, 34), (0, -1)])
def test_zero_native_and_linear_budget_enforced(monkeypatch, tmp_path, native, linear):
    _, outcome, task, _, _ = task_fixture(monkeypatch)
    changed = deepcopy(outcome['entry']); changed.update(native_solver_calls=native, linear_trajectory_calls=linear)
    monkeypatch.setattr(r, 'invoke_design', lambda task, folder, original: deepcopy(changed))
    monkeypatch.setattr(r.value, '_native_fields', lambda entry: None)
    (tmp_path/'models').mkdir(); monitor = Monitor()
    with pytest.raises(r.value.old.Pause, match='zero native'): r.solve(task, tmp_path, monitor)
    assert not monitor.state['outcomes']


def test_pka_blocker_does_not_import_runtime(monkeypatch, tmp_path):
    _, _, task, _, _ = task_fixture(monkeypatch, status='failed', blocked=True)
    monkeypatch.setattr(r, 'invoke_design', lambda *a: pytest.fail('Blocked requests must remain blocked'))
    (tmp_path/'models').mkdir(); monitor = Monitor(); r.solve(task, tmp_path, monitor)
    item = r.value.old.read(tmp_path/'models'/f"{task['key'][:20]}.json")
    assert item['entry']['reason'] == 'construction_blocked' and item['reused'] is True
    assert item['new_native_calls'] == item['new_linear_trajectory_calls'] == 0


def test_completion_recovery_bound_to_new_candidate(monkeypatch, tmp_path):
    _, outcome, task, _, _ = task_fixture(monkeypatch)
    folder = tmp_path/'models'/task['key'][:20]; folder.mkdir(parents=True)
    entry = deepcopy(outcome['entry']); entry.update(native_solver_calls=0, linear_trajectory_calls=17)
    r.save(folder/'completion.json', {'key': task['key'], 'candidate': r.CANDIDATE,
        'exact_input_digest': r.source_runner.exact_input_digest(task),
        'source_prediction_receipt': task['source_prediction_receipt'], 'entry': entry})
    monkeypatch.setattr(r, 'invoke_design', lambda *args: pytest.fail('Recovery must not run new trajectories'))
    r.solve(task, tmp_path, Monitor())
    item = r.value.old.read(tmp_path/'models'/f"{task['key'][:20]}.json")
    assert item['new_linear_trajectory_calls'] == 17


def test_policy_score_blocks_before_any_observation_access(monkeypatch):
    monkeypatch.setattr(r.base, 'ClinicalStore', lambda *a: pytest.fail('Observations accessed before lock'))
    with pytest.raises(ValueError, match='Lock all joint'):
        r.conservative_rows({'tasks': {'a': {}}}, {'all_predictions_locked_before_scoring': False, 'outcomes': {}})


def test_conservative_policy_uses_precommitted_predictions_even_if_joint_error_is_worse(monkeypatch):
    case = {'case_id': 'c', 'compound_id': 'c', 'validation_stratum': 'new_compounds'}
    request = {'compound_id': 'c'}
    entry_source = {'request': request, 'status': 'predicted', 'target_observations_used': False, 'curve': {'marker': 'source'}}
    entry_joint = {'request': request, 'status': 'predicted', 'target_observations_used': False, 'curve': {'marker': 'joint'}}
    protocol = {'cases': [case], 'tasks': {
        'joint-key': {'request': request, 'conservative_use_joint': True, 'source_task': {'key': 'original-failed'}},
        'source-key': {'request': request, 'conservative_use_joint': False, 'source_task': {'key': 'original-success'},
            'source_prediction_receipt': {'path': 'source'}}},
        'units': [{'case_id': 'c', 'condition': condition, 'key': 'joint-key' if condition == 'available' else 'source-key',
            'conservative_policy': 'joint' if condition == 'available' else 'source'} for condition in r.CONDITIONS]}
    state = {'all_predictions_locked_before_scoring': True, 'outcomes': {'joint-key': {'path': 'joint'}, 'source-key': {'path': 'unused'}}}
    monkeypatch.setattr(r.value.old, 'checked', lambda receipt: {'key': 'joint-key', 'locked_at': 'frozen', 'entry': deepcopy(entry_joint)})
    monkeypatch.setattr(r.infra, 'source_entry', lambda task: deepcopy(entry_source))
    monkeypatch.setattr(r.base, 'ClinicalStore', lambda *a: SimpleNamespace(unchanged=lambda: None))
    monkeypatch.setattr(r.value.evaluation, 'clinical_truth', lambda *a: ({}, {}, {}))
    def score(case, receipt, entry, *args):
        # The newly designed result is deliberately worse; this cannot alter selection.
        return {'attempt_status': entry['status'], 'curve_metrics': {'gmfe': 9 if entry['curve']['marker'] == 'joint' else 1}}
    monkeypatch.setattr(r.value, 'score_one', score)
    monkeypatch.setattr(r.value, 'drift', lambda case, nominal, target: {'nominal': nominal['curve']['marker'], 'target': target['curve']['marker']})
    rows = r.conservative_rows(protocol, state)
    available = next(row for row in rows if row['condition'] == 'available')
    missing = next(row for row in rows if row['condition'] == 'missing_clearance')
    assert available['curve_metrics']['gmfe'] == 9 and available['policy_source'] == 'joint'
    assert missing['curve_metrics']['gmfe'] == 1 and missing['policy_source'] == 'source'
    assert missing['prediction_drift'] == {'nominal': 'joint', 'target': 'source'}
    assert len(rows) == 5 and all(row['prediction_reused'] for row in rows)


def test_report_scope_and_lock_boundary(monkeypatch, tmp_path):
    assert r.own(r.DEFAULT) == r.DEFAULT
    with pytest.raises(ValueError): r.own(r.SOURCE)
    monkeypatch.setattr(r, 'own', lambda path: path); monkeypatch.setattr(r.value, 'verify', lambda p: None)
    r.save(tmp_path/'protocol.json', {'tasks': {'one': {}}})
    r.save(tmp_path/'state.json', {'protocol': r.value.old.ref(tmp_path/'protocol.json'),
        'all_predictions_locked_before_scoring': False, 'outcomes': {}})
    monkeypatch.setattr(r.base, 'ClinicalStore', lambda *args: pytest.fail('Clinical observations opened early'))
    with pytest.raises(ValueError, match='Lock every prediction'): r.report(tmp_path)


def cases():
    return [{'case_id': str(i), 'compound_id': str(i), 'cluster_id': 'paired' if i < 2 else str(i)} for i in range(4)]


def test_benefit_denominators_include_unpaired_failures_and_strict_meaningful_changes():
    result = r.benefit_statistics({'0': .95, '1': 1.05, '2': .999}, cases())
    assert result['benefit'] == 1 and result['harm'] == 1 and result['within_5pct'] == 1
    assert result['requested_drugs'] == 4 and result['common_scored_drugs'] == 3 and result['unpaired_drugs'] == 1
    assert result['requested_rates']['benefit']['fraction'] == .25
    assert result['common_scored_rates']['benefit']['fraction'] == pytest.approx(1/3)


def test_cluster_bootstrap_preserves_repeated_study_draw_weights():
    group = cases(); output = r.benefit_statistics({'0': .8, '1': .9, '2': 1.2}, group, resamples=500, seed=8)
    names = sorted({c['cluster_id'] for c in group})
    numerator = np.array([sum(c['cluster_id'] == x and c['case_id'] in {'0', '1'} for c in group) for x in names])
    denominator = np.array([sum(c['cluster_id'] == x for c in group) for x in names])
    draws = np.random.default_rng(8).integers(0, len(names), size=(500, len(names)))
    expected = np.quantile(numerator[draws].sum(axis=1)/denominator[draws].sum(axis=1), [.025, .975])
    assert output['requested_rates']['benefit']['ci95'] == pytest.approx(expected)


def gate_data():
    return {'candidate_over_source_qualified': {'all': {condition: {
        'candidate_scored': 29, 'source_scored': 27, 'common_curve_ratio': {'estimate': .99},
        'meaningful_changes': {'benefit': 6, 'harm': 2}} for condition in r.CONDITIONS}}}


def test_gate_requires_coverage_nonregression_error_nonregression_and_real_benefit():
    assert r.adoption_gate(gate_data())['passed']
    coverage = gate_data(); coverage['candidate_over_source_qualified']['all']['missing_both']['candidate_scored'] = 26
    assert not r.adoption_gate(coverage)['passed']
    errors = gate_data(); errors['candidate_over_source_qualified']['all']['available']['common_curve_ratio']['estimate'] = 1.01
    assert not r.adoption_gate(errors)['passed']
    wins = gate_data(); wins['candidate_over_source_qualified']['all']['missing_clearance']['meaningful_changes']['benefit'] = 2
    assert not r.adoption_gate(wins)['passed']


def test_no_tiny_numeric_gain_is_counted_as_five_percent_benefit():
    result = r.benefit_statistics({'0': 1-1e-10, '1': 1+1e-10}, cases())
    assert result['benefit'] == result['harm'] == 0 and result['within_5pct'] == 2
