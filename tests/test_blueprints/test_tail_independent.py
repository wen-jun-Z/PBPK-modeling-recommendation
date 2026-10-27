"""Independent scientific contracts; native physiology and fitting are stubbed."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import tail_exchange_calibration as cal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_robust_blueprint as frozen
import run_tail_blueprint as adapter


def test_phase_weights_use_only_times_and_allocate_half_to_each_supported_phase():
    times = np.array([.25, .5, 1., 2., 3., 5., 8., 12., 24.])
    plan = cal.tail_weights(times, .5)
    assert plan['supported'] and plan['tail_indices'] == [6, 7, 8]
    weights = np.array(plan['weights'])
    assert weights[plan['body_indices']].sum() == pytest.approx(.5)
    assert weights[plan['tail_indices']].sum() == pytest.approx(.5)
    assert weights.sum() == pytest.approx(1.)
    assert plan['selection_uses_concentrations'] is False


@pytest.mark.parametrize('times,infusion', [([.1, .2, .3], 1.), ([1., 2., 3.], 0.)])
def test_missing_phase_support_really_uses_uniform_fallback(times, infusion):
    plan = cal.tail_weights(times, infusion)
    assert not plan['supported'] and plan['fallback']
    np.testing.assert_allclose(plan['weights'], np.full(len(times), 1 / len(times)))


@pytest.mark.parametrize('n', [7, 8, 9, 13, 16])
def test_development_folds_are_disjoint_have_six_training_points_and_protected_interior(n):
    times = np.geomspace(.1, 24., n)
    folds = cal.development_folds(times, 1.)
    assert len(folds) == 2
    for fold in folds:
        train, held = set(fold['training_indices']), set(fold['heldout_indices'])
        assert len(train) >= 6 and held
        assert not train & held and train | held == set(range(n))
        assert fold['selection_uses_concentrations'] is False
    assert max(folds[0]['heldout_indices']) == n - 1
    assert not set(folds[1]['heldout_indices']) & set(folds[1]['protected_interior_indices'])
    assert not set(folds[0]['heldout_indices']) & set(folds[1]['heldout_indices'])
    assert folds == cal.development_folds(times, 1.)


def test_six_points_cannot_be_misreported_as_successful_cv():
    with pytest.raises(ValueError, match='seven'):
        cal.development_folds([.1, .2, .5, 1., 2., 8.], .5)


def test_incomplete_cv_prior_is_excluded_and_tie_prefers_stronger_prior():
    plan = {'prior_weights': [.01, .1], 'folds': [{}, {}]}
    stages = {'cv_0.01_0': {'heldout_score': .001}, 'cv_0.01_1': {'heldout_score': None},
              'cv_0.1_0': {'heldout_score': 4.}, 'cv_0.1_1': {'heldout_score': 4.}}
    selected = cal._cv_selection(stages, plan)
    assert selected['selected']['penalty_weight'] == .1
    assert selected['candidates'][0]['score'] is None
    for stage in stages.values(): stage['heldout_score'] = 1.
    assert cal._cv_selection(stages, plan)['selected']['penalty_weight'] == .1
    for stage in stages.values(): stage['heldout_score'] = None
    assert cal._cv_selection(stages, plan)['selected'] is None


def mappings(variant):
    return [{'construction_partition_method': formula, 'variant': variant, 'eligible': True,
             'decision': {'objective': objective}, 'formula_selection_score': score,
             'clearance_prior_policy': {'penalty_weight': .01}}
            for formula, objective, score in [('PKSimStandard', .001, 9.), ('RodgersRowland', 4., 1.)]]


def test_guarded_formula_selection_uses_cv_while_tail_uses_full_objective():
    runner = adapter.create_runner()
    guarded = runner.api.choose_formula(mappings('development_guarded'))
    tail = runner.api.choose_formula(mappings('tail_balanced'))
    assert guarded['construction_partition_method'] == 'RodgersRowland'
    assert tail['construction_partition_method'] == 'PKSimStandard'
    invalid = mappings('development_guarded'); invalid[0]['formula_selection_score'] = None
    with pytest.raises(ValueError, match='finite'):
        runner.api.choose_formula(invalid)
    invalid[0]['eligible'] = False
    assert runner.api.choose_formula(invalid)['construction_partition_method'] == 'RodgersRowland'


def test_adapter_does_not_mutate_frozen_runner_and_preserves_baseline_semantics():
    old_api, old_choose, old_methods = frozen.api, frozen.api.choose_formula, frozen.METHODS
    first, second = adapter.create_runner(), adapter.create_runner()
    assert first.api is not second.api and first.api is not old_api
    assert frozen.api is old_api and frozen.api.choose_formula is old_choose and frozen.METHODS == old_methods
    assert first.variant('current') == 'evidence_flexible'
    assert first.runner_binding['baseline']['source_method'] == 'evidence_flexible'
    assert first.__file__.endswith('/scripts/run_tail_blueprint.py')


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    root = tmp_path
    cal.save(root / 'evidence.json', {'role': 'independent synthetic parameter evidence'})
    source = {**cal._ref(root / 'evidence.json', root), 'location': 'synthetic parameter'}
    context = {'species': 'human', 'matrix': 'plasma', 'route': 'intravenous'}
    prior = {'parameter': 'total_clearance', 'compound_id': 'synthetic:A', 'value': 2.,
        'unit': 'ml/min/kg', 'basis': 'predicted', 'source': source, 'context': context,
        'prediction_method': {'name': 'RF', 'version': 'fixture-1', 'training_overlap': False,
            'applicability': {'human_IV_parameter_model': True, 'target_structure_and_scaffold_excluded': True}}}
    request = {'entry_reason': 'model_missing', 'compound_id': 'synthetic:A', 'compound_name': 'Synthetic A',
        'human': {'age_years': 30., 'weight_kg': 80., 'height_cm': 175., 'sex': 'MALE', 'population': 'European', 'seed': 17},
        'dosing': {'route': 'intravenous', 'dose_mg': 20., 'start_h': 0., 'infusion_minutes': 30.},
        'target': {'window_start_h': 0., 'window_end_h': 12., 'endpoints': ['plasma_concentration_time'],
                   'primary_endpoint': 'plasma_concentration_time'}, 'partition_method': 'PKSimStandard',
        'evidence_ledger': source, 'evidence': [prior,
            {'parameter': 'renal_clearance', 'compound_id': 'synthetic:A', 'value': .5, 'unit': 'ml/min/kg',
             'basis': 'measured', 'source': source, 'context': context},
            {'parameter': 'molecular_weight', 'compound_id': 'synthetic:A', 'value': 500., 'unit': 'g/mol',
             'basis': 'measured', 'source': source}], 'forbidden_sources': []}
    (root / 'policy.py').write_text('# synthetic frozen policy\n')
    monkeypatch.setattr(cal, '_code_pins', lambda _: {'policy.py': cal.sha256(root / 'policy.py')})
    traces, calls, current = [], [], {}
    def physiology(*_):
        calls.append('physiology'); return {'fixture': True}
    def simulate(physiology, request, times, variant, keys, x):
        calls.append('linear')
        return {'concentration_mg_l': np.exp(1.4 + x[1] - (.5 + np.exp(x[0])) * np.asarray(times))}
    def optimize(residual, initial, **_):
        value = residual(initial)
        progress = cal._read(current['output'] / 'progress.json')
        traces.append({'stage': progress['incomplete_start']['stage'], 'initial': initial.tolist(),
                       'residual': value.tolist()})
        return SimpleNamespace(x=initial, success=True, message='synthetic optimization', nfev=1,
                               jac=np.zeros((len(value), 3)))
    monkeypatch.setattr(cal, '_physiology', physiology)
    monkeypatch.setattr(cal, '_simulate', simulate)
    monkeypatch.setattr(cal, 'least_squares', optimize)
    def fit(name, final_value=.02):
        output = root / name; current['output'] = output
        curve = {'compound_id': request['compound_id'], 'role': 'calibration_data_reconstruction',
            'matrix': 'plasma', 'unit': 'umol/l', 'time_unit': 'min', 'human': request['human'],
            'dosing': request['dosing'], 'observations': [{'time_min': t, 'concentration_umol_per_l': v}
                for t, v in zip([15., 30., 60., 120., 180., 240., 480.], [8., 7., 5., 2., .8, .2, final_value])]}
        source_path = root / (name + '.json'); cal.save(source_path, curve)
        reference = {**cal._ref(source_path, root), 'role': 'calibration_data_reconstruction'}
        return cal.fit_mapping(reference_request=request, training_source=reference, output_dir=output,
                               project_root=root, variant='development_guarded')
    return SimpleNamespace(root=root, request=request, fit=fit, traces=traces, calls=calls)


def test_heldout_concentration_cannot_change_fold_residuals_or_initialization(synthetic):
    first = synthetic.fit('original', .02)
    first_traces = deepcopy(synthetic.traces); synthetic.traces.clear()
    second = synthetic.fit('heldout_changed', 2.)
    for name in ('cv_0.01_0', 'cv_0.1_0'):
        a = [r for r in first_traces if r['stage'] == name]
        b = [r for r in synthetic.traces if r['stage'] == name]
        assert len(a) == len(b) == 8 and a == b
        assert first['stages'][name]['training_indices'] == list(range(6))
        assert first['stages'][name]['heldout_indices'] == [6]
        assert first['stages'][name]['heldout_score'] != second['stages'][name]['heldout_score']
    assert first['linear_solver_calls'] + second['linear_solver_calls'] == synthetic.calls.count('linear')
    assert first['physical_family'] == second['physical_family'] == 'evidence_flexible'
    assert first['comparison_curve_used'] is second['comparison_curve_used'] is False


def test_completed_cv_resume_does_not_refit_or_reexport_physiology(synthetic):
    first = synthetic.fit('resume')
    before = len(synthetic.calls)
    resumed = synthetic.fit('resume')
    assert resumed['actual_new_calls'] == 0 and len(synthetic.calls) == before
    assert resumed['decision'] == first['decision']
    assert resumed['cv_selection'] == first['cv_selection']
    assert resumed['linear_solver_calls'] == first['linear_solver_calls']
