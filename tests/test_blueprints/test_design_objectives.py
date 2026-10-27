"""Selection objectives must respect requests, study isolation and missing data."""
from copy import deepcopy
import math

import pytest
from pydantic import ValidationError

from pbpk_kb.blueprints.design_objectives import (
    PKDesignObjective, assess_objective, validate_objective_data,
)


def scenario(identifier='a', group='study_a', *, role='selection', values=(1., 1.), times=(0., 1.)):
    return {'scenario_id': identifier, 'study_group_id': group, 'role': role,
            'time_unit': 'min', 'unit': 'umol/l', 'auc_evaluable': True, 'cmax_evaluable': True,
            'observations': [{'time': time, 'value': value} for time, value in zip(times, values, strict=True)]}


def score(scenarios, predictions, *, curve=None):
    if curve is None:
        grouped = {}
        for scene, values in zip(scenarios, predictions, strict=True):
            actual = [point.get('concentration_umol_per_l', point.get('value')) for point in scene['observations']]
            mse = sum((math.log(predicted) - math.log(observed)) ** 2
                      for predicted, observed in zip(values, actual, strict=True)) / len(values)
            grouped.setdefault(scene['study_group_id'], []).append(mse)
        curve = math.sqrt(sum(sum(group) / len(group) for group in grouped.values()) / len(grouped))
    return {'valid': True, 'log_rmse': curve,
            'scenarios': {scene['scenario_id']: {'valid': True, 'predictions_umol_per_l': values}
                          for scene, values in zip(scenarios, predictions, strict=True)}}


def auc_objective(**kwargs):
    return PKDesignObjective(primary='observed_window_auc', max_curve_log_rmse=1., **kwargs)


def test_curve_and_auc_requests_can_rank_same_candidates_differently():
    scenes = [scenario()]
    # Candidate A gets the average exposure exactly right, but has a worse shape.
    a = score(scenes, [[0.5, 1.5]])
    b = score(scenes, [[1.1, 1.1]])
    default = PKDesignObjective()
    assert assess_objective(scenes, b, default)['primary_error'] < assess_objective(scenes, a, default)['primary_error']
    assert assess_objective(scenes, a, auc_objective())['primary_error'] < assess_objective(scenes, b, auc_objective())['primary_error']
    assert assess_objective(scenes, a, auc_objective())['metrics']['curve_log_rmse'] == a['log_rmse']


def test_cmax_request_uses_sampled_peak_not_simulation_peak():
    scenes = [scenario(values=(1., 4.), times=(2., 6.))]
    result = assess_objective(scenes, score(scenes, [[2., 4.]]),
                              PKDesignObjective(primary='sampled_cmax', max_curve_log_rmse=1.))
    assert result['valid'] and result['constraints_passed']
    assert result['primary_error'] == 0.
    assert result['metrics']['cmax_fold_error'] == 1.
    assert result['metrics']['auc_fold_error'] == pytest.approx(1.2)


def test_endpoint_winner_is_ineligible_when_curve_constraint_fails():
    scenes = [scenario()]
    result = assess_objective(scenes, score(scenes, [[0.1, 1.9]]), auc_objective())
    assert result['valid'] is True
    assert result['primary_error'] == 0.
    assert result['constraints_passed'] is False
    assert result['violations'] == ['curve_log_rmse exceeds requested maximum 1.0']


def test_studies_equal_and_arms_equal_within_study():
    scenes = [scenario('a', 'many'), scenario('b', 'many'), scenario('c', 'single')]
    result = assess_objective(scenes, score(scenes, [[2., 2.], [8., 8.], [1., 1.]]), PKDesignObjective())
    expected = ((math.log(2.) + math.log(8.)) / 2 + 0.) / 2
    assert result['metrics']['auc_log_error'] == pytest.approx(expected)
    assert result['metrics']['cmax_log_error'] == pytest.approx(expected)
    assert result['metrics']['auc_fold_error'] == pytest.approx(2.)
    assert len(result['study_metrics']) == 2
    assert result['study_metrics'][0]['scenario_ids'] == ['a', 'b']


def test_replicating_arms_does_not_increase_a_studys_weight():
    scenes = [scenario('a', 'many'), scenario('b', 'single')]
    base = assess_objective(scenes, score(scenes, [[4., 4.], [1., 1.]]), PKDesignObjective())
    scenes.insert(1, scenario('copy', 'many'))
    repeated = assess_objective(scenes, score(scenes, [[4., 4.], [4., 4.], [1., 1.]]), PKDesignObjective())
    assert repeated['metrics'] == base['metrics']


def test_positive_and_negative_bias_cannot_cancel():
    scenes = [scenario('a'), scenario('b', 'study_b')]
    result = assess_objective(scenes, score(scenes, [[2., 2.], [0.5, 0.5]]), PKDesignObjective())
    assert result['metrics']['auc_fold_error'] == pytest.approx(2.)


def test_auc_sorts_times_with_predictions_and_uses_only_observed_window():
    scenes = [scenario(values=(1., 4., 2.), times=(10., 2., 4.))]
    result = assess_objective(scenes, score(scenes, [[2., 8., 4.]]), auc_objective())
    assert result['metrics']['auc_fold_error'] == pytest.approx(2.)


@pytest.mark.parametrize('kwargs', [
    {'primary': 'auc'}, {'unknown_option': 2.}, {'selection_tolerance': -0.01},
    {'max_curve_log_rmse': -1.}, {'max_auc_fold_error': 0.99}, {'max_cmax_fold_error': 0.},
    {'selection_tolerance': float('nan')}, {'max_curve_log_rmse': float('inf')},
    {'max_auc_fold_error': '2'}, {'max_cmax_fold_error': True},
    {'primary': 'sampled_cmax'}, {'primary': 'observed_window_auc'},
])
def test_policy_rejects_unknown_coerced_nonfinite_and_unconstrained_values(kwargs):
    with pytest.raises(ValidationError):
        PKDesignObjective(**kwargs)


def test_zero_error_and_inclusive_constraints_are_valid():
    scenes = [scenario()]
    objective = PKDesignObjective(max_curve_log_rmse=0., max_auc_fold_error=1., max_cmax_fold_error=1.,
                                  selection_tolerance=0.)
    result = assess_objective(scenes, score(scenes, [[1., 1.]], curve=0.), objective)
    assert result['valid'] and result['constraints_passed']
    assert result['primary_error'] == 0.
    assert result['violations'] == []


@pytest.mark.parametrize('field', ['auc_evaluable', 'cmax_evaluable'])
@pytest.mark.parametrize('flag', [False, None, 1, 'true'])
def test_requested_endpoint_requires_explicit_eligibility(field, flag):
    scenes = [scenario()]
    scenes[0][field] = flag
    objective = PKDesignObjective(**{'max_' + field.removesuffix('_evaluable') + '_fold_error': 2.})
    with pytest.raises(ValueError, match=field):
        validate_objective_data({'scenarios': scenes}, objective)
    result = assess_objective(scenes, score(scenes, [[1., 1.]]), objective)
    assert not result['valid'] and not result['constraints_passed']


def test_missing_flag_rejected_without_silently_removing_study():
    scenes = [scenario('a'), scenario('b', 'study_b')]
    del scenes[1]['auc_evaluable']
    result = assess_objective(scenes, score(scenes, [[1., 1.], [2., 2.]]), auc_objective())
    assert result['valid'] is False
    assert result['primary_error'] is None
    optional = assess_objective(scenes, score(scenes, [[1., 1.], [2., 2.]]), PKDesignObjective())
    assert optional['valid'] is True
    assert optional['metrics']['auc_log_error'] is None
    assert optional['metrics']['auc_fold_error'] is None
    assert optional['metrics']['cmax_fold_error'] == pytest.approx(math.sqrt(2.))


@pytest.mark.parametrize('times,values', [((0.,), (1.,)), ((1., 1.), (1., 2.))])
def test_single_or_duplicate_times_allow_curve_but_not_endpoints(times, values):
    scenes = [scenario(values=values, times=times)]
    result = assess_objective(scenes, score(scenes, [list(values)]), PKDesignObjective())
    assert result['valid'] is True
    assert result['metrics']['auc_log_error'] is None
    assert result['metrics']['cmax_log_error'] is None
    with pytest.raises(ValueError, match='unique observation times'):
        validate_objective_data({'scenarios': scenes}, auc_objective())


@pytest.mark.parametrize('change', [
    lambda s: s.update(role='test'),
    lambda s: s.update(study_group_id=''),
    lambda s: s.update(time_unit='h'),
    lambda s: s.update(unit='ng/ml'),
    lambda s: s.update(observations=[]),
    lambda s: s['observations'][0].update(time=-1.),
    lambda s: s['observations'][0].update(time=True),
    lambda s: s['observations'][0].update(value=0.),
    lambda s: s['observations'][0].update(value=-1.),
    lambda s: s['observations'][0].update(value=float('nan')),
    lambda s: s['observations'][0].update(value='1'),
])
def test_malformed_observations_reject_before_search(change):
    scenes = [scenario()]
    change(scenes[0])
    with pytest.raises(ValueError):
        validate_objective_data({'scenarios': scenes}, PKDesignObjective())


def test_duplicate_scene_and_cross_role_study_rejected():
    scenes = [scenario(), scenario()]
    with pytest.raises(ValueError, match='unique'):
        validate_objective_data({'scenarios': scenes}, PKDesignObjective())
    scenes[1]['scenario_id'] = 'b'
    scenes[1]['role'] = 'calibration'
    with pytest.raises(ValueError, match='cross'):
        validate_objective_data({'scenarios': scenes}, PKDesignObjective())


@pytest.mark.parametrize('mutation', [
    lambda result: result.update(valid=False),
    lambda result: result.update(log_rmse=-1.),
    lambda result: result.update(log_rmse=float('inf')),
    lambda result: result['scenarios'].clear(),
    lambda result: result['scenarios'].update(unrelated={'valid': True}),
    lambda result: result['scenarios']['a'].update(valid=False),
    lambda result: result['scenarios']['a'].update(predictions_umol_per_l=[1.]),
    lambda result: result['scenarios']['a'].update(predictions_umol_per_l=[0., 1.]),
    lambda result: result['scenarios']['a'].update(predictions_umol_per_l=[float('inf'), 1.]),
    lambda result: result['scenarios']['a'].update(predictions_umol_per_l=[True, 1.]),
])
def test_bad_prediction_cannot_become_an_eligible_candidate(mutation):
    scenes = [scenario()]
    result = score(scenes, [[1., 1.]])
    mutation(result)
    assessed = assess_objective(scenes, result, PKDesignObjective())
    assert assessed['valid'] is False
    assert assessed['constraints_passed'] is False
    assert assessed['violations']


def test_requested_endpoint_constraints_are_checked_independently():
    scenes = [scenario()]
    result = assess_objective(scenes, score(scenes, [[3., 3.]]),
                              PKDesignObjective(max_auc_fold_error=2., max_cmax_fold_error=2.))
    assert result['valid'] is True and result['constraints_passed'] is False
    assert len(result['violations']) == 2


def test_validation_and_scoring_do_not_mutate_inputs_or_follow_holdouts():
    scenes = [scenario()]
    case = {'scenarios': scenes, 'external_scenarios': object(), 'source_audit': object()}
    original = deepcopy(scenes)
    result = score(scenes, [[1., 1.]])
    saved_result = deepcopy(result)
    validate_objective_data(case, PKDesignObjective())
    assess_objective(scenes, result, PKDesignObjective())
    assert scenes == original
    assert result == saved_result


@pytest.mark.parametrize('unit', ['µmol/l', 'μmol/l', 'umol/l'])
def test_supported_concentration_spellings(unit):
    scenes = [scenario()]
    scenes[0]['unit'] = unit
    validate_objective_data({'scenarios': scenes}, auc_objective())
    assert assess_objective(scenes, score(scenes, [[1., 1.]]), auc_objective())['valid']


def test_normalized_runtime_observations_are_self_describing_without_unit_guessing():
    scene = scenario()
    del scene['time_unit']
    del scene['unit']
    scene['observations'] = [{'time_min': 2., 'concentration_umol_per_l': 1.},
                              {'time_min': 5., 'concentration_umol_per_l': 3., 'unit': 'µmol/l'}]
    validate_objective_data({'scenarios': [scene]}, auc_objective())
    result = assess_objective([scene], score([scene], [[2., 6.]]), auc_objective())
    assert result['valid']
    assert result['metrics']['auc_fold_error'] == pytest.approx(2.)


@pytest.mark.parametrize('field', ['time_unit', 'unit'])
def test_legacy_observations_without_units_are_not_guessed(field):
    scene = scenario()
    del scene[field]
    with pytest.raises(ValueError, match='explicit'):
        validate_objective_data({'scenarios': [scene]}, PKDesignObjective())


@pytest.mark.parametrize('points', [
    [{'time_min': 2., 'concentration_umol_per_l': 1., 'unit': 'ng/ml'}],
    [{'time_min': 2., 'value': 1.}],
    [{'time_min': 2.}],
    [{'concentration_umol_per_l': 1.}],
    [{'time_min': 2., 'concentration_umol_per_l': 1., 'time': 2., 'value': 1.}],
])
def test_normalized_rows_reject_ambiguous_fields_or_conflicting_units(points):
    scene = scenario()
    scene['observations'] = points
    with pytest.raises(ValueError):
        validate_objective_data({'scenarios': [scene]}, PKDesignObjective())


def test_forged_curve_summary_cannot_bypass_explicit_constraint():
    scenes = [scenario()]
    # Correct AUC alone must not rescue a false near-perfect curve score.
    result = assess_objective(scenes, score(scenes, [[0.1, 1.9]], curve=0.01), auc_objective())
    assert result['valid'] is False
    assert result['constraints_passed'] is False
    assert result['metrics']['curve_log_rmse'] == 0.01  # Original summary is retained, never repaired.
    assert any('disagrees' in violation for violation in result['violations'])


def test_curve_summary_from_arm_weighted_mean_cannot_bypass_study_weighting():
    scenes = [scenario('a', 'many'), scenario('b', 'many'), scenario('c', 'single')]
    false_curve = math.sqrt(2 * math.log(2.) ** 2 / 3)
    result = assess_objective(scenes, score(scenes, [[2., 2.], [2., 2.], [1., 1.]], curve=false_curve),
                              PKDesignObjective())
    assert result['valid'] is False
    assert any('disagrees' in violation for violation in result['violations'])


def test_summary_comparison_accepts_only_roundoff_and_preserves_original():
    scenes = [scenario()]
    result = score(scenes, [[2., 2.]])
    result['log_rmse'] *= 1 + 1e-9
    assessment = assess_objective(scenes, result, PKDesignObjective())
    assert assessment['valid'] is True
    assert assessment['metrics']['curve_log_rmse'] == result['log_rmse']
    result['log_rmse'] *= 1 + 1e-7
    assert assess_objective(scenes, result, PKDesignObjective())['valid'] is False
