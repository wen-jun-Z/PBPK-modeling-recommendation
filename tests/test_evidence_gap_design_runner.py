"""Guard the new evidence-gap runner's acquisition boundaries and adoption rule."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(PROJECT / 'scripts'))
    spec = importlib.util.spec_from_file_location('gap_runner_test', PROJECT / 'scripts/run_evidence_gap_design.py')
    r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
    monkeypatch.setattr(r, 'ROOT', tmp_path); monkeypatch.setattr(r, 'RUN', tmp_path / 'run'); r.RUN.mkdir()
    monkeypatch.setattr(r.base, 'normalized', deepcopy)
    monkeypatch.setattr(r.base, 'pins_in', lambda obj, pins: None)
    monkeypatch.setattr(r, 'verify', lambda pins: None)
    monkeypatch.setattr(r, 'ref', lambda path: {'path': str(Path(path).relative_to(tmp_path)), 'sha256': r.base.sha256(path)})
    def checked(source):
        if 'path' not in source:
            return deepcopy(source)
        path = tmp_path / source['path']
        if r.base.sha256(path) != source['sha256']:
            raise ValueError('Changed source hash')
        return r.read(path)
    monkeypatch.setattr(r, 'checked', checked)
    return r


def setup(r, monkeypatch, feasible=True):
    req = dict(compound_id='synthetic', human={'weight_kg': 70.}, dosing={'route': 'intravenous', 'dose_mg': 5., 'infusion_minutes': 1.}, evidence=[dict(parameter='molecular_weight', unit='g/mol', value=500.)])
    full = dict(compound_id='synthetic', study_id='development', role='calibration_data_reconstruction', matrix='plasma', time_unit='min', unit='umol/l', human=req['human'], dosing=req['dosing'], molecular_weight_g_mol=500., raw_observations={'secret': 1}, observations=[dict(time_min=float(i), concentration_umol_per_l=10./i) for i in range(1, 8)])
    split = r.data.split_development(full, req, 'early_four'); r.save(r.RUN / 'visible.json', split)
    case = dict(case_id='synthetic', target_contract={'metadata': True}, requests={'evidence': {'development': req, 'target': req}}, sampling_sources={'full': {'vault': True}})
    dk = 'synthetic__early_four'; spec = dict(case_id='synthetic', regime='early_four', visible_source=r.ref(r.RUN / 'visible.json'))
    jobs = {dk + f'__point_{ix:02d}': dict(design_key=dk, case_id='synthetic', regime='early_four', candidate_id=ix, time_h=t) for ix, t in zip(split['design_only']['candidate_ids'], split['design_only']['candidate_times_h'])}
    p = dict(source_pins={}, cases=[case], designs={dk: spec}, jobs=jobs, fit_budget_per_formula=16384, fit_starts=16, physiology_cache={f'synthetic__development__{f}': {'phys': f} for f in r.base.FORMULAS})
    r.save(r.RUN / 'protocol.json', p)
    baseline = dict(identity=dict(design_key=dk, candidate_id=None), protocol_sha256=r.base.sha256(r.RUN / 'protocol.json'))
    r.save(r.RUN / 'baseline.json', baseline)
    decision = dict(design_key=dk, protocol_sha256=r.base.sha256(r.RUN / 'protocol.json'), visible_source=spec['visible_source'], target_contract=case['target_contract'], candidate_ids=split['design_only']['candidate_ids'], candidate_times_h=split['design_only']['candidate_times_h'], choices={m: {g: 0 for g in r.GOALS} for m in ('request_directed', 'linear_covariance', 'time_farthest', 'maximum_variance')}, baseline=r.ref(r.RUN / 'baseline.json'), hidden_values_used=False, target_values_used=False, output_pins={})
    r.save(r.RUN / 'decision.json', decision)
    r.save(r.RUN / 'actions_locked.json', dict(protocol=r.ref(r.RUN / 'protocol.json'), designs={dk: {'result': r.ref(r.RUN / 'decision.json')}}))
    lock = r.ref(r.RUN / 'actions_locked.json')
    monkeypatch.setattr(r.base, '_training_document', lambda *args: deepcopy(full))
    monkeypatch.setattr(r.prior, 'get_phys', lambda *args: {})
    fitted = []
    def fit(predict, request, training, physiology, **kwargs):
        assert len(training['observations']) == 5 and 'raw_observations' not in training
        fitted.append(deepcopy(training))
        return dict(best={'objective': float(len(fitted)), 'log_parameters': [0.]*4} if feasible else None, calls=3, failures=0, variant='blood_partition', call_limit=16384, starts=[{}]*16, visible_points=5, no_warm_start=True, target_observations_used=False)
    monkeypatch.setattr(r.old.optimizer, 'fit', fit)
    monkeypatch.setattr(r.base.engine, 'prediction_grid', lambda req: np.array([.1, 1.]))
    monkeypatch.setattr(r.old, 'simulate', lambda *args: dict(concentration_mg_l=np.ones(2), mass_balance={}))
    return p, case, dk, jobs, lock, fitted


def test_reveal_exactly_one_after_lock_and_selection_before_prediction(runner, monkeypatch):
    r = runner; p, _, _, jobs, lock, fitted = setup(r, monkeypatch)
    key = next(iter(jobs))
    def simulate(*args):
        assert (r.RUN / 'work' / key / 'selection.json').exists()
        return dict(concentration_mg_l=np.ones(2), mass_balance={})
    monkeypatch.setattr(r.old, 'simulate', simulate)
    entry = r.execute(p, key, jobs[key], lock); result = r.checked(entry['result'])
    assert result['status'] == 'predicted' and len(fitted) == 2
    assert [x['time_min'] for x in fitted[0]['observations']] == [1, 2, 3, 4, 5]
    assert result['resources']['objective_calls'] == 6
    r.validate_unit(p, key, result, lock)
    r.execute(p, key, jobs[key], lock)
    assert len(fitted) == 2, 'Exact restart must not refit completed unit'


def test_incomplete_lock_rejected_before_reading_hidden_values(runner, monkeypatch):
    r = runner; p, _, _, jobs, _, _ = setup(r, monkeypatch)
    r.save(r.RUN / 'bad_lock.json', dict(protocol=r.ref(r.RUN / 'protocol.json'), designs={}))
    monkeypatch.setattr(r.base, '_training_document', lambda *args: pytest.fail('Vault opened before lock'))
    key = next(iter(jobs))
    with pytest.raises(ValueError, match='All actions'):
        r.execute(p, key, jobs[key], r.ref(r.RUN / 'bad_lock.json'))


def test_failed_fits_retained_without_prediction(runner, monkeypatch):
    r = runner; p, _, _, jobs, lock, fitted = setup(r, monkeypatch, feasible=False)
    key = next(iter(jobs)); result = r.checked(r.execute(p, key, jobs[key], lock)['result'])
    assert result['status'] == 'failed' and result['resources']['prediction_calls'] == 0
    assert result['reason'] == 'No feasible formula fit' and len(fitted) == 2


@pytest.mark.parametrize('mutation', ['goal', 'contract', 'baseline', 'index'])
def test_wrong_regime_or_endpoint_receipts_rejected(runner, monkeypatch, mutation):
    r = runner; p, _, dk, _, _, _ = setup(r, monkeypatch)
    decision = r.read(r.RUN / 'decision.json')
    if mutation == 'goal':
        decision['choices']['request_directed'].pop('auc_inf')
    elif mutation == 'contract':
        decision['target_contract'] = {'wrong': True}
    elif mutation == 'baseline':
        r.save(r.RUN / 'wrong.json', dict(identity=dict(design_key='late_four', candidate_id=None), protocol_sha256=decision['protocol_sha256']))
        decision['baseline'] = r.ref(r.RUN / 'wrong.json')
    else:
        decision['choices']['request_directed']['cmax'] = True
    with pytest.raises(ValueError):
        r.validate_decision(p, dk, decision)


def test_partial_visible_data_cannot_be_overwritten(runner, monkeypatch):
    r = runner; p, case, dk, _, _, _ = setup(r, monkeypatch)
    training = r.checked(p['designs'][dk]['visible_source'])['visible_training']
    folder = r.RUN / 'partial'; r.save(folder / 'training.json', {'changed': True})
    with pytest.raises(ValueError, match='partially completed'):
        r.fit_predict(p, case, training, folder, r.Guard(p), {'example': 1})
    assert r.read(folder / 'training.json') == {'changed': True}


def test_profile_duplicates_do_not_multiply_mass(runner):
    r = runner
    rows = [dict(successful=True, mse=mse, best={'log_parameters': x}) for mse, x in [(0.01, [0.]*4), (0.001, [1e-8]*4), (0.002, [1.]*4), (4., [2.]*4)]]
    rows.append(dict(successful=False, mse=None, best=None))
    chosen = r.profile_members(dict(lower4=[-10.]*4, upper4=[10.]*4, profile=rows), .1)
    assert [i for i, _ in chosen] == [1, 2]


def gate_fixture(r):
    summary = {s: {g: {m: {'endpoint': {'n_cases': 10, 'estimate': 1.2}, 'curve': {'n_cases': 10, 'estimate': 1.3}} for m in r.POLICIES} for g in r.GOALS} for s in r.REGIMES}
    comparison = {s: {g: {m: {'estimate': .98, 'ratio_below_one': 6} for m in r.POLICIES if m != 'request_directed'} for g in r.GOALS} for s in r.REGIMES}
    return summary, comparison, {'example': {'status': 'predicted'}}


def test_frozen_gate_supports_only_primary_regime(runner):
    r = runner; summary, comparisons, units = gate_fixture(r)
    assert all(r.expansion_checks(summary, comparisons, units).values())
    comparisons['late_four']['cmax']['time_farthest']['estimate'] = 1.01
    comparisons['early_four']['cmax']['time_farthest']['estimate'] = .5
    assert not r.expansion_checks(summary, comparisons, units)['one_percent_better_all_generic']


@pytest.mark.parametrize('kind', ['missing', 'secondary_harm', 'failed', 'empty'])
def test_missing_failure_and_secondary_harm_cannot_pass(runner, kind):
    r = runner; summary, comparisons, units = gate_fixture(r)
    if kind == 'missing':
        summary['late_four']['cmax']['request_directed']['endpoint']['n_cases'] = 9
    elif kind == 'secondary_harm':
        summary['early_four']['terminal_half_life']['request_directed']['endpoint']['estimate'] = 2.
    elif kind == 'failed':
        units['example']['status'] = 'failed'
    else:
        units = {}
    assert not all(r.expansion_checks(summary, comparisons, units).values())
