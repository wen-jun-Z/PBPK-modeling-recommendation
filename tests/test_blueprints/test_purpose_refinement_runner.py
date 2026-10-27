"""Scientific input protection and partial-failure retention for refinement."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_purpose_refinement as run


@pytest.fixture
def inputs():
    source = ROOT / 'experiments/blueprint_design/purpose_hundred_20260930'
    protocol = json.loads((source / 'protocol.json').read_text())
    case = protocol['cases'][0]
    original = protocol['case_protocols'][case['case_id']]['input']['request']
    prior = json.loads((ROOT / 'experiments/blueprint_design/purpose_refinement_20261001/priors/predictions.json').read_text())['predictions'][case['compound_id']]
    return deepcopy(case), deepcopy(original), deepcopy(prior)


def test_measured_clearance_is_protected_while_predicted_fu_changes(inputs, tmp_path):
    case, original, prior = inputs
    row = next(r for r in original['evidence'] if r['parameter'] == 'total_clearance')
    row['basis'] = 'measured'; row.pop('prediction_method', None)
    before = deepcopy(row)
    result = run.evidence_request(original, prior, {'path': 'source.json', 'sha256': 'a'*64}, tmp_path, phase='fu_low')
    actual = {r['parameter']: r for r in result['evidence']}
    assert {k: actual['total_clearance'][k] for k in before} == before
    assert actual['fraction_unbound']['value'] == prior['methods']['selected']['ranges']['fraction_unbound']['80']['lower']
    assert actual['fraction_unbound']['basis'] == 'predicted'
    assert original['evidence'] != result['evidence']


def test_conditional_stability_does_not_turn_missing_or_zero_into_stability():
    nominal = {'endpoints': {'vss': {'value': 10}, 'tmax': {'value': .5}, 'c_last': {'value': 0}}}
    good = {'endpoints': {'vss': {'value': 20}, 'tmax': {'value': .75}, 'c_last': {'value': 1}}}
    x = run.conditional_stability(nominal, [good, good])
    assert x['vss']['max_change'] == 2
    assert x['tmax']['max_change'] == .25
    assert not x['c_last']['complete']
    assert all(v['max_change'] is None for v in run.conditional_stability(nominal, [good, {}]).values())


def test_infeasible_anchor_never_runs_or_makes_fake_prediction(monkeypatch, tmp_path):
    runner = run.RefinementRunner()
    monkeypatch.setattr(run.capacity, 'anchor_partition', lambda *a, **k: {'status': 'blocked', 'reason': 'below vascular floor'})
    def bad(*a, **k): raise AssertionError('infeasible anchor must not simulate')
    monkeypatch.setattr(run.capacity, 'simulate_design', bad)
    result = runner._simulate({}, {}, [], tmp_path, anchored=True)
    assert result['status'] == 'failed' and not result['endpoints']
    assert not list(tmp_path.iterdir())


class FakeMonitor:
    def __init__(self):
        self.lock = threading.RLock(); self.state = {'progress': {}}; self.failures = []
    def check(self, **kwargs): pass
    def persist(self, **kwargs): pass
    def record_result(self, value): self.failures.append(value)


def test_later_infrastructure_failure_retains_finished_method(inputs, tmp_path, monkeypatch):
    case, original, prior = inputs
    runner = run.RefinementRunner()
    monkeypatch.setattr(run, 'evidence_request', lambda original, *a, **kw: original)
    monkeypatch.setattr(run.engine, '_physiology', lambda *a, **kw: {})
    calls = []
    def simulate(*args, **kwargs):
        calls.append(kwargs['anchored'])
        if kwargs['anchored']: raise OSError('simulated archive write failure')
        return {'status': 'predicted', 'endpoints': {'vss': {'value': 10}}}
    monkeypatch.setattr(runner, '_simulate', simulate)
    protocol = {'case_protocols': {case['case_id']: {'request': original, 'prior': prior, 'prior_ref': {}}}}
    monitor = FakeMonitor()
    result = runner._phase(tmp_path, protocol, case, 'fu_low', monitor)
    assert result['status'] == 'failed'
    assert result['methods']['parameter_only']['status'] == 'predicted'
    assert monitor.failures == [True]
    # A completed failure receipt is reused, not an infinite retry.
    again = runner._phase(tmp_path, protocol, case, 'fu_low', monitor)
    assert again == result and calls == [False, True]


def test_interrupted_native_phase_is_retained_without_retry(inputs, tmp_path, monkeypatch):
    case, original, prior = inputs
    runner = run.RefinementRunner()
    phase = tmp_path / 'work' / case['case_id'] / 'fu_low'; phase.mkdir(parents=True)
    (phase / 'started.json').write_text('{}')
    def bad(*a, **k): raise AssertionError('must not retry native')
    monkeypatch.setattr(run.engine, '_physiology', bad)
    result = runner._phase(tmp_path, {'case_protocols': {case['case_id']: {}}}, case, 'fu_low', FakeMonitor())
    assert result['status'] == 'failed' and result['failure_kind'] == 'interrupted_native'


def test_scoring_barrier_runs_before_reading_any_outcome_or_reference(tmp_path, monkeypatch):
    runner = run.RefinementRunner()
    monkeypatch.setattr(runner, 'own_run', lambda p: tmp_path)
    def not_locked(*args, **kwargs):
        assert kwargs['locked']; raise ValueError('all predictions must lock')
    monkeypatch.setattr(runner, 'load', not_locked)
    def forbidden(*args): raise AssertionError('no label/result may open before lock barrier')
    monkeypatch.setattr(runner, 'checked', forbidden)
    with pytest.raises(ValueError, match='must lock'):
        runner.report(tmp_path)


def test_new_nominal_interruption_cannot_erase_old_comparator(inputs, tmp_path, monkeypatch):
    case, original, prior = inputs
    runner = run.RefinementRunner()
    source = tmp_path / 'old_prediction.json'
    old = {'status': 'predicted', 'endpoints': {'vss': {'value': 10}}}
    source.write_text(json.dumps(old))
    baseline = runner.ref(source)
    monitor = FakeMonitor()
    monitor.state.update(protocol={'path': 'protocol.json', 'sha256': 'a'*64}, outcomes={})
    def incomplete(folder, protocol, case, phase, monitor):
        p = folder / 'work' / case['case_id'] / phase
        p.mkdir(parents=True)
        result = {'status': 'failed', 'reason': 'interrupted', 'methods': {}}
        (p / 'prediction.json').write_text(json.dumps(result))
        return result
    monkeypatch.setattr(runner, '_phase', incomplete)
    runner._case(tmp_path, {'case_protocols': {case['case_id']: {'baseline_prediction': baseline}}}, case, monitor)
    result = runner.checked(monitor.state['outcomes'][case['case_id']])
    assert result['methods']['baseline'] == old
    assert all(result['methods'][m]['status'] == 'failed' for m in run.METHODS[1:])
