"""Synthetic orchestration contracts; never load a clinical target or fit."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(PROJECT / 'scripts'))
    spec = importlib.util.spec_from_file_location('selective_runner_test', PROJECT / 'scripts/run_selective_blood_partition.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'RUN', tmp_path / 'run'); m.RUN.mkdir()
    monkeypatch.setattr(m.base, 'normalized', deepcopy)
    monkeypatch.setattr(m.base, 'pins_in', lambda value, pins: None)
    monkeypatch.setattr(m, 'verify', lambda pins: None)
    monkeypatch.setattr(m, 'ref', lambda p: {'path': str(Path(p).relative_to(tmp_path)), 'sha256': m.base.sha256(p)})
    def checked(value):
        if isinstance(value, dict) and 'path' in value and 'sha256' in value:
            p = tmp_path / value['path']
            if m.base.sha256(p) != value['sha256']: raise ValueError('Changed artifact hash')
            return m.read(p)
        return deepcopy(value)
    monkeypatch.setattr(m, 'checked', checked)
    return m


def setup_clinical(m, monkeypatch, n=5):
    req = {'compound_id': 'dummy', 'human': {'weight_kg': 70.},
           'dosing': {'dose_mg': 1., 'infusion_minutes': 30.},
           'evidence': [{'parameter': 'molecular_weight', 'value': 500.}]}
    target = deepcopy(req); target['dosing']['infusion_minutes'] = 60.
    source = {'role': 'five-visible-observations'}
    case = {'case_id': 'dummy', 'sampling_sources': {'five': source},
            'requests': {'evidence': {'development': req, 'target': target}},
            'target_observations': {'never_read': True}}
    best = {'objective': .1, 'log_parameters': [0.] * 3}
    three = {'case_id': 'dummy', 'sampling': 'five', 'status': 'predicted', 'method': 'direct_relaxed',
             'training_source': source, 'target_observations_used': False,
             'formula': 'PKSimStandard', 'parameters': best,
             'selection': {'formula': 'PKSimStandard', 'parameters': best, 'basis': 'development_only'},
             'formula_scores': [{'formula': 'PKSimStandard', 'best': best},
                                {'formula': 'RodgersRowland', 'best': {'objective': .2}}]}
    fitting_request = deepcopy(req); fitting_request['partition_method'] = 'PKSimStandard'
    phys_ref = {'native': 'dummy'}
    four = {'version': m.old.optimizer.VERSION, 'variant': 'blood_partition',
            'request': fitting_request, 'training_source': source, 'development_physiology': phys_ref,
            'call_limit': m.old.BUDGET, 'calls': 20, 'starts': [{}] * m.old.STARTS,
            'no_warm_start': True, 'target_observations_used': False, 'visible_points': n,
            'status': 'fitted', 'best': {'log_parameters': [0., 0., 0., .2]}}
    job = {'kind': 'clinical', 'case_id': 'dummy', 'sampling': 'five', 'formula': 'PKSimStandard'}
    protocol = {'cases': [case], 'source_pins': {}, 'jobs': {'unit': job},
                'controls': {'unit': {'three': three, 'four_fit': four}},
                'physiology_cache': {'dummy__development__PKSimStandard': phys_ref}}
    m.save(m.RUN / 'protocol.json', protocol)
    monkeypatch.setattr(m, 'get_phys', lambda *args: {})
    def training(source_given, request_given, root):
        assert source_given == source
        assert request_given == fitting_request
        return {'observations': [dict(time_min=i+1) for i in range(n)]}
    monkeypatch.setattr(m.base, '_training_document', training)
    monkeypatch.setattr(m.old.optimizer, '_visible', lambda req, train: (np.arange(n) + 1., np.ones(n)))
    monkeypatch.setattr(m.old.optimizer, '_initialization', lambda *args: (-np.ones(4)*5, np.ones(4)*5, None, None, None))
    monkeypatch.setattr(m.base.engine, 'prediction_grid', lambda req: np.array([.1, 1.]))
    calls = []
    def analyze(phys, req, times, obs, protocol, guard):
        calls.append((times.copy(), obs.copy()))
        return {'status': 'profiled', 'calls': 9, 'failures': 0,
                'decision': {'accepted': False}, 'necessity_only_decision': {'accepted': True}}
    monkeypatch.setattr(m, 'analyze', analyze)
    def simulate(phys, req, times, x):
        lock = m.read(m.RUN / 'work/unit/decision.json')
        assert lock['target_observations_used'] is False
        assert lock['formula'] == 'PKSimStandard'
        assert req['dosing']['infusion_minutes'] == 60.
        assert len(calls) == 1
        return {'concentration_mg_l': np.ones(len(times)), 'mass_balance': {}, 'observation': 'plasma'}
    monkeypatch.setattr(m.blood, 'simulate', simulate)
    return protocol, job, calls


def test_clinical_uses_development_formula_and_five_only_then_locks(runner, monkeypatch):
    p, job, calls = setup_clinical(runner, monkeypatch)
    result = runner.execute(p, 'unit', job)
    unit = runner.checked(result['result'])
    assert len(calls) == 1 and len(calls[0][0]) == 5
    assert unit['choices'] == {'always_three': 'three', 'always_four': 'four',
                               'necessity_only': 'four', 'selective': 'three'}
    assert unit['prediction_calls'] == 1


def test_five_rejects_full_curve_before_profile(runner, monkeypatch):
    p, job, calls = setup_clinical(runner, monkeypatch, n=6)
    with pytest.raises(ValueError, match='Five-point isolation'):
        runner.execute(p, 'unit', job)
    assert calls == []


@pytest.mark.parametrize('changed', [
    {'variant': 'exchange_shape'}, {'version': 'unrelated'}, {'call_limit': 10},
    {'starts': []}, {'no_warm_start': False}, {'target_observations_used': True},
    {'visible_points': 9}, {'development_physiology': {'native': 'different'}},
    {'status': 'failed'}, {'best': None}, {'best': {'log_parameters': [0., 0., 0., 90.]}},
])
def test_ineligible_frozen_four_axis_fit_never_reaches_profile(runner, monkeypatch, changed):
    p, job, calls = setup_clinical(runner, monkeypatch)
    p['controls']['unit']['four_fit'].update(changed)
    with pytest.raises(ValueError, match='Frozen'):
        runner.execute(p, 'unit', job)
    assert calls == []


def test_cannot_switch_to_four_axis_favorite_formula(runner, monkeypatch):
    p, job, calls = setup_clinical(runner, monkeypatch)
    job['formula'] = 'RodgersRowland'
    with pytest.raises(ValueError, match='Job formula'):
        runner.execute(p, 'unit', job)
    assert calls == []


def test_baseline_choice_rechecked_against_own_development_scores(runner, monkeypatch):
    p, job, calls = setup_clinical(runner, monkeypatch)
    p['controls']['unit']['three']['formula_scores'][1]['best']['objective'] = .01
    with pytest.raises(ValueError, match='Baseline formula'):
        runner.execute(p, 'unit', job)
    assert calls == []


def test_resume_returns_same_wrapper_without_refitting(runner, monkeypatch):
    p, job, calls = setup_clinical(runner, monkeypatch)
    first = runner.execute(p, 'unit', job)
    second = runner.execute(p, 'unit', job)
    assert first == second and len(calls) == 1


def test_resume_rejects_changed_protocol(runner, monkeypatch):
    p, job, calls = setup_clinical(runner, monkeypatch)
    runner.execute(p, 'unit', job)
    p['changed'] = True; runner.save(runner.RUN / 'protocol.json', p)
    with pytest.raises(ValueError, match='Resume inputs'):
        runner.execute(p, 'unit', job)
    assert len(calls) == 1


def test_guard_stops_work_on_hash_change(runner, monkeypatch):
    p, job, calls = setup_clinical(runner, monkeypatch)
    monkeypatch.setattr(runner, 'verify', lambda _: (_ for _ in ()).throw(ValueError('Hash changed')))
    with pytest.raises(ValueError, match='Hash changed'):
        runner.execute(p, 'unit', job)
    assert calls == []


def test_report_refuses_partial_global_lock(runner):
    protocol = {'jobs': {'unit': {}}, 'source_pins': {}}
    runner.save(runner.RUN / 'protocol.json', protocol)
    runner.save(runner.RUN / 'state.json', {'status': 'running', 'protocol': runner.ref(runner.RUN / 'protocol.json'), 'units': {}})
    with pytest.raises(ValueError, match='All decisions/predictions'):
        runner.report()


def row(x, mse, successful=True):
    return {'log_rbc_scale': x, 'mse': mse, 'successful': successful,
            'best': {'log_parameters': [0., 0., 0., x], 'mse': mse}}


def test_profile_candidates_use_nested_best_and_only_successful_fits(runner):
    p = {'profile': [row(0., .1), row(.5, .01), row(-.5, 0., successful=False)]}
    assert runner.profile_candidates(p) == {'three': [0.] * 4, 'four': [0., 0., 0., .5]}


def test_missing_null_never_becomes_a_fake_baseline(runner):
    with pytest.raises(ValueError, match='No successful null'):
        runner.profile_candidates({'profile': [row(.2, .1)]})


def test_profile_rejects_best_with_changed_fixed_coordinate(runner):
    bad = row(0., .1); bad['best']['log_parameters'][3] = .1
    with pytest.raises(ValueError, match='fixed-coordinate'):
        runner.profile_candidates({'profile': [bad]})


def test_synthetic_summary_keeps_failures_and_success_denominators(runner):
    subset = [{'decision': {'accepted': True}, 'necessity_only': {'accepted': True},
               'candidate_failure': None, 'status': 'profiled', 'errors': {m: 2. for m in runner.METHODS}},
              {'decision': {'accepted': False}, 'necessity_only': {'accepted': False},
               'candidate_failure': 'null did not converge', 'status': 'incomplete', 'errors': {}}]
    result = runner.summarize_synthetic(subset)
    assert result['n'] == 2 and result['failed_candidates'] == 1
    assert result['transfer_gmfe']['selective'] == pytest.approx(2.)
    assert result['transfer_scored_counts']['selective'] == 1

