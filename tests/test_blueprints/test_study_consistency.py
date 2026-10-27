from copy import deepcopy
import json

import pytest

from pbpk_kb.blueprints.study_consistency import apply_study_consistency


SOURCE = 'partition_rodgers_rowland'


def _row(identifier, *, cal=(0.4, 0.5), sel=(0.3, 0.6), eligible=True):
    def objective(role, values):
        return {'valid': True, 'constraints_passed': True, 'study_metrics': [
            {'study_group_id': role + str(index),
             'scenario_ids': [role + str(index) + '_arm1'] +
                             ([role + str(index) + '_arm2'] if index == 0 else []),
             'curve_log_rmse': value, 'auc_log_error': value / 2,
             'cmax_log_error': value * 2}
            for index, value in enumerate(values)]}
    return {'candidate_id': identifier, 'selection_eligible': True,
            'purpose_eligible': eligible, 'result': {
                'calibration_objective': objective('calibration', cal),
                'selection_objective': objective('selection', sel)}}


def _inputs():
    candidates = {
        'source': {'candidate_id': 'source', 'action': SOURCE, 'free_parameter_ids': []},
        'fitted': {'candidate_id': 'fitted', 'action': SOURCE, 'free_parameter_ids': ['clearance']},
    }
    return [_row('source'), _row('fitted', cal=(0.3, 0.4), sel=(0.2, 0.5))], candidates


def _apply(records, by_id, primary='curve'):
    return apply_study_consistency(records, by_id, primary=primary, source_action=SOURCE)


def _details(report, identifier='fitted'):
    return next(row for row in report['candidate_reports'] if row['candidate_id'] == identifier)


def test_improvement_passes_with_actual_study_counts_and_zero_extra_solver_calls():
    records, candidates = _inputs()
    eligible, report = _apply(records, candidates)
    assert [r['candidate_id'] for r in eligible] == ['source', 'fitted']
    assert report['status'] == 'applied'
    assert report['source_baseline']['study_counts'] == {'calibration': 2, 'selection': 2}
    assert len(_details(report)['study_differences']) == 4  # Six arms are four studies.
    assert _details(report)['worst_delta'] == pytest.approx(-0.1)
    assert _details(report)['worst_regression'] == 0
    assert report['solver_calls'] == 0
    assert report['cross_validation_performed'] is False
    assert report['independent_validation_performed'] is False
    json.dumps(report, allow_nan=False)


def test_no_mutation_or_aliasing_of_caller_records_or_reference_study_rows():
    records, candidates = _inputs()
    original = deepcopy((records, candidates))
    eligible, report = _apply(records, candidates)
    assert (records, candidates) == original
    eligible[0]['result']['selection_objective']['valid'] = False
    _details(report)['study_differences'][0]['scenario_ids'].append('changed')
    report['source_baseline']['study_values']['selection']['selection0']['scenario_ids'].append('new')
    assert (records, candidates) == original


def test_exact_ties_retain_source_and_allow_downstream_simple_model_preference():
    records, candidates = _inputs()
    records[1] = _row('fitted')
    eligible, report = _apply(records, candidates)
    assert len(eligible) == 2
    assert _details(report)['worst_delta'] == 0
    assert all(item['no_worse'] for item in _details(report)['study_differences'])


@pytest.mark.parametrize('role', ['calibration', 'selection'])
def test_one_worse_study_is_rejected_despite_better_average(role):
    records, candidates = _inputs()
    rows = records[1]['result'][role + '_objective']['study_metrics']
    rows[0]['curve_log_rmse'] = 0.01
    baseline = records[0]['result'][role + '_objective']['study_metrics'][1]['curve_log_rmse']
    rows[1]['curve_log_rmse'] = baseline + 0.01
    eligible, report = _apply(records, candidates)
    assert [r['candidate_id'] for r in eligible] == ['source']
    assert _details(report)['reasons'] == ['worse_than_source:' + role + ':' + role + '1']
    assert _details(report)['worst_regression'] == pytest.approx(0.01)


@pytest.mark.parametrize(('primary', 'key'), [
    ('curve', 'curve_log_rmse'), ('observed_window_auc', 'auc_log_error'),
    ('sampled_cmax', 'cmax_log_error'),
])
def test_only_requested_primary_metric_controls_guard(primary, key):
    records, candidates = _inputs()
    scene = records[1]['result']['selection_objective']['study_metrics'][0]
    for metric in ['curve_log_rmse', 'auc_log_error', 'cmax_log_error']:
        scene[metric] = 0.01 if metric == key else 100.0
    eligible, report = _apply(records, candidates, primary)
    assert len(eligible) == 2
    assert report['metric'] == key
    assert {item['metric'] for item in _details(report)['study_differences']} == {key}


def test_optional_unavailable_endpoints_do_not_block_curve_primary():
    records, candidates = _inputs()
    for record in records:
        for role in ['calibration', 'selection']:
            for study in record['result'][role + '_objective']['study_metrics']:
                study['auc_log_error'] = study['cmax_log_error'] = None
    assert len(_apply(records, candidates)[0]) == 2
    assert _apply(records, candidates, 'observed_window_auc')[1]['status'] == 'blocked'


@pytest.mark.parametrize(('baseline', 'candidate', 'passed'), [
    (0.0, 1e-10, True), (0.0, 1.01e-10, False),
    (1.0, 1.0 + 9e-9, True), (1.0, 1.0 + 1.1e-8, False),
    (0.1, 0.119, False),  # The existing 0.02 selection tie rule is not used here.
])
def test_only_numerical_tolerance_is_allowed(baseline, candidate, passed):
    records, candidates = _inputs()
    records[0]['result']['selection_objective']['study_metrics'][0]['curve_log_rmse'] = baseline
    records[1]['result']['selection_objective']['study_metrics'][0]['curve_log_rmse'] = candidate
    _, report = _apply(records, candidates)
    assert _details(report)['passed'] is passed


@pytest.mark.parametrize('bad', [None, True, False, '0.2', -0.1, float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('index', [0, 1])
def test_nonfinite_negative_and_non_numeric_primary_metrics_fail_closed(bad, index):
    records, candidates = _inputs()
    records[index]['result']['calibration_objective']['study_metrics'][0]['curve_log_rmse'] = bad
    eligible, report = _apply(records, candidates)
    if index == 0:
        assert eligible == []
        assert report['status'] == 'blocked'
        assert report['reason'].startswith('source_reference_invalid:')
    else:
        assert [r['candidate_id'] for r in eligible] == ['source']
        assert _details(report)['reasons'] == ['study_metric_not_finite_nonnegative:calibration0']


def test_invalid_request_constraints_do_not_disqualify_source_as_reference_or_restore_eligibility():
    records, candidates = _inputs()
    records[0]['purpose_eligible'] = False
    records[0]['result']['selection_objective']['constraints_passed'] = False
    eligible, report = _apply(records, candidates)
    assert report['status'] == 'applied'
    assert [r['candidate_id'] for r in eligible] == ['fitted']
    assert report['source_baseline']['purpose_eligible'] is False
    assert report['source_baseline']['request_constraints_passed']['selection'] is False
    assert _details(report, 'source')['reasons'] == ['prior_purpose_ineligible']


@pytest.mark.parametrize('eligibility', ['selection_eligible', 'purpose_eligible'])
def test_guard_does_not_rehabilitate_otherwise_ineligible_fitted_candidate(eligibility):
    records, candidates = _inputs()
    records[1][eligibility] = False
    eligible, report = _apply(records, candidates)
    assert [r['candidate_id'] for r in eligible] == ['source']
    assert _details(report)['passed'] is False
    assert len(_details(report)['study_differences']) == 4


@pytest.mark.parametrize(('mutation', 'reason'), [
    ('missing', 'source_reference_missing'),
    ('ambiguous', 'source_reference_ambiguous'),
    ('unevaluated', 'source_reference_not_evaluated'),
    ('selection_invalid', 'source_reference_selection_ineligible'),
    ('free_effective', 'source_reference_not_fixed'),
    ('identity', 'source_reference_identity_mismatch'),
    ('duplicate_evaluation', 'ambiguous_duplicate_evaluation'),
])
def test_missing_invalid_or_ambiguous_reference_blocks_every_candidate(mutation, reason):
    records, candidates = _inputs()
    if mutation == 'missing':
        del candidates['source']
    elif mutation == 'ambiguous':
        candidates['second_source'] = {**candidates['source'], 'candidate_id': 'second_source'}
    elif mutation == 'unevaluated':
        records.pop(0)
    elif mutation == 'selection_invalid':
        records[0]['selection_eligible'] = False
    elif mutation == 'free_effective':
        records[0]['result']['effective_free_parameter_ids'] = ['clearance']
    elif mutation == 'identity':
        candidates['source']['candidate_id'] = 'another'
    elif mutation == 'duplicate_evaluation':
        records.append(deepcopy(records[0]))
    eligible, report = _apply(records, candidates)
    assert eligible == []
    assert report['status'] == 'blocked'
    assert report['reason'] == reason
    assert all(not row['passed'] for row in report['candidate_reports'])


@pytest.mark.parametrize('role', ['calibration', 'selection'])
@pytest.mark.parametrize('index', [0, 1])
def test_invalid_objective_is_not_accepted_even_with_finite_metrics(role, index):
    records, candidates = _inputs()
    records[index]['result'][role + '_objective']['valid'] = False
    eligible, report = _apply(records, candidates)
    assert len(eligible) == (0 if index == 0 else 1)
    assert (report['reason'] if index == 0 else _details(report)['reasons'][0]).endswith(role + '_objective_invalid')


@pytest.mark.parametrize('mutation', ['missing_study', 'extra_study', 'renamed_study',
                                       'duplicate_study', 'cross_role_study', 'wrong_explicit_role',
                                       'missing_arm', 'duplicate_arm', 'cross_study_arm'])
def test_matching_study_role_and_observation_groups_are_required(mutation):
    records, candidates = _inputs()
    studies = records[1]['result']['selection_objective']['study_metrics']
    if mutation == 'missing_study':
        studies.pop()
    elif mutation == 'extra_study':
        studies.append({**studies[1], 'study_group_id': 'extra', 'scenario_ids': ['extra_arm']})
    elif mutation == 'renamed_study':
        studies[0]['study_group_id'] = 'renamed'
    elif mutation == 'duplicate_study':
        studies.append(deepcopy(studies[0]))
    elif mutation == 'cross_role_study':
        studies[0]['study_group_id'] = 'calibration0'
    elif mutation == 'wrong_explicit_role':
        studies[0]['role'] = 'calibration'
    elif mutation == 'missing_arm':
        studies[0]['scenario_ids'].pop()
    elif mutation == 'duplicate_arm':
        studies[0]['scenario_ids'].append(studies[0]['scenario_ids'][0])
    elif mutation == 'cross_study_arm':
        studies[1]['scenario_ids'] = studies[0]['scenario_ids'][:]
    eligible, report = _apply(records, candidates)
    assert [r['candidate_id'] for r in eligible] == ['source']
    assert _details(report)['passed'] is False


def test_study_and_arm_order_does_not_change_the_guard():
    records, candidates = _inputs()
    for role in ['calibration', 'selection']:
        rows = records[1]['result'][role + '_objective']['study_metrics']
        rows.reverse()
        for row in rows:
            row['scenario_ids'].reverse()
    assert len(_apply(records, candidates)[0]) == 2


def test_all_request_ineligible_candidates_yield_no_delivery_not_an_implicit_source_fallback():
    records, candidates = _inputs()
    for row in records:
        row['purpose_eligible'] = False
    eligible, report = _apply(records, candidates)
    assert eligible == []
    assert report['status'] == 'applied'
    assert report['reason'] == 'no_eligible_candidate_passes_study_consistency'


@pytest.mark.parametrize('primary', ['invalid', None, []])
def test_unsupported_primary_is_fail_closed(primary):
    records, candidates = _inputs()
    eligible, report = _apply(records, candidates, primary)
    assert eligible == []
    assert report['reason'] == 'unsupported_primary_objective'
