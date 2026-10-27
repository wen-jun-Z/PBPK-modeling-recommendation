"""Analytic measurement-choice examples; no clinical observation is supplied."""
from copy import deepcopy
import inspect
import json

import numpy as np
import pytest

from pbpk_kb.blueprint_policy.endpoint_evidence_design import choose


def example():
    a = np.array([-1., -1., 1., 1.])
    b = np.array([-1., 1., -1., 1.])
    return {"candidate_times_h": [2., 8.], "observed_times_h": [1., 3.],
            "concentrations_mg_l": np.exp(np.column_stack([a, 3 * b])),
            "endpoints": {"auc": np.exp(a), "half_life": np.exp(b)}}


def test_requested_endpoint_changes_choice_and_curve_variance_is_not_goal_utility():
    result = choose(**example())
    policy = result["policies"]
    assert policy["goal_directed"]["auc"]["selected_time_h"] == 2.
    assert policy["goal_directed"]["half_life"]["selected_time_h"] == 8.
    assert policy["max_variance"]["selected_time_h"] == 8.
    assert result["goals"]["auc"]["utility_log_variance_reduction"] == pytest.approx([1 / 1.01, 0])
    assert result["goals"]["half_life"]["utility_log_variance_reduction"] == pytest.approx([0, 9 / 9.01])


def test_rank_one_update_equals_conditional_block_covariance():
    values = example()
    result = choose(**values, log_noise_sd=.2)
    log_joint = np.column_stack([np.log(values["endpoints"]["auc"]),
                                 np.log(values["concentrations_mg_l"])])
    full_covariance = np.cov(log_joint, rowvar=False, bias=True)
    for candidate in range(2):
        expected = full_covariance[0, 0] - full_covariance[0, candidate + 1] ** 2 / (
            full_covariance[candidate + 1, candidate + 1] + .2 ** 2)
        assert result["goals"]["auc"]["conditional_variance_log"][candidate] == pytest.approx(expected)


def test_uniform_random_is_an_expected_action_not_a_sample():
    result = choose(**example())
    random = result["policies"]["uniform_random"]
    assert random["selected_index"] is None and random["selected_time_h"] is None
    assert random["probabilities"] == [.5, .5]
    assert random["expected_goal_utility"]["auc"] == pytest.approx(.5 / 1.01)
    assert result == choose(**example())


def test_farthest_uses_log_time_and_earlier_tie():
    values = example()
    assert choose(**values)["policies"]["time_farthest"]["selected_time_h"] == 8.
    values["observed_times_h"] = [1., 16.]
    assert choose(**values)["policies"]["time_farthest"]["selected_time_h"] == 2.


def test_constant_endpoint_explicitly_falls_back_to_farthest():
    values = example()
    values["endpoints"] = {"constant": [1.3] * 4}
    result = choose(**values)
    policy = result["policies"]["goal_directed"]["constant"]
    assert policy["selected_time_h"] == 8.
    assert policy["reason"] == "no_endpoint_disagreement"
    assert policy["fallback"] == "time_farthest"
    assert policy["scores"] == [0., 0.]


def test_duplicate_members_allowed_without_fake_variance():
    values = example()
    values["concentrations_mg_l"] = np.tile([1.3, 2.7], (6, 1))
    values["endpoints"] = {"constant": [9.7] * 6}
    result = choose(**values)
    assert result["sample_count"] == 6
    assert result["concentration_log_variance"] == [0., 0.]
    assert result["policies"]["max_variance"]["reason"] == "no_concentration_disagreement"
    assert result["policies"]["max_variance"]["selected_time_h"] == 8.


def test_score_tie_tolerance_is_absolute_and_prefers_earlier():
    values = example()
    a = np.log(values["endpoints"]["auc"])
    values["concentrations_mg_l"] = np.exp(np.column_stack([a, (1 + 1e-14) * a]))
    result = choose(**values)
    assert result["policies"]["max_variance"]["selected_time_h"] == 2.
    assert result["policies"]["goal_directed"]["auc"]["selected_time_h"] == 2.


def test_single_candidate_is_valid_for_all_policies():
    result = choose([2.], [1., 3.], [[1.], [2.]], {"auc": [3., 4.]})
    assert result["policies"]["uniform_random"]["probabilities"] == [1.]
    for name in ("max_variance", "time_farthest"):
        assert result["policies"][name]["selected_index"] == 0
    assert result["policies"]["goal_directed"]["auc"]["selected_index"] == 0


def test_noise_increases_conditional_uncertainty_without_changing_data():
    low = choose(**example(), log_noise_sd=.05)
    high = choose(**example(), log_noise_sd=.2)
    assert low["goals"]["auc"]["conditional_variance_log"][0] < high["goals"]["auc"]["conditional_variance_log"][0]
    assert low["concentration_finite_spread"] == high["concentration_finite_spread"]


def test_prediction_units_do_not_change_log_utility():
    values = example()
    expected = choose(**values)
    values["concentrations_mg_l"] *= 1000
    values["endpoints"] = {key: value * 100 for key, value in values["endpoints"].items()}
    result = choose(**values)
    for goal in result["goals"]:
        assert result["goals"][goal]["utility_log_variance_reduction"] == pytest.approx(
            expected["goals"][goal]["utility_log_variance_reduction"], abs=1e-14)


def test_inputs_and_numpy_random_state_are_unchanged_and_output_is_json():
    values = example()
    old = deepcopy(values)
    state = np.random.get_state()
    result = choose(**values)
    assert values["candidate_times_h"] == old["candidate_times_h"]
    assert values["observed_times_h"] == old["observed_times_h"]
    np.testing.assert_array_equal(values["concentrations_mg_l"], old["concentrations_mg_l"])
    for key in values["endpoints"]:
        np.testing.assert_array_equal(values["endpoints"][key], old["endpoints"][key])
    after = np.random.get_state()
    assert after[0] == state[0] and after[2:] == state[2:]
    np.testing.assert_array_equal(after[1], state[1])
    json.dumps(result, allow_nan=False)
    assert "observations" not in inspect.signature(choose).parameters


@pytest.mark.parametrize("field,value", [
    ("candidate_times_h", []), ("candidate_times_h", [0., 8.]),
    ("candidate_times_h", [8., 2.]), ("candidate_times_h", [2., 2.]),
    ("candidate_times_h", [1., 8.]), ("candidate_times_h", [2., np.inf]),
    ("candidate_times_h", [2., np.nan]), ("candidate_times_h", ["2", "8"]),
    ("observed_times_h", []), ("observed_times_h", [3., 1.]),
    ("observed_times_h", [1., 1.]), ("observed_times_h", [-1., 3.]),
    ("concentrations_mg_l", [[1., 2.]]),
    ("concentrations_mg_l", [[1.], [2.], [3.], [4.]]),
    ("concentrations_mg_l", [[1., 0.], [2., 3.], [4., 5.], [6., 7.]]),
    ("concentrations_mg_l", [[1., np.inf], [2., 3.], [4., 5.], [6., 7.]]),
    ("endpoints", {}), ("endpoints", {"bad": [1., 2.]}),
    ("endpoints", {"bad": [1., 2., 3., 0.]}), ("endpoints", {"bad": [1., 2., 3., np.nan]}),
    ("endpoints", {"bad": [[1., 2., 3., 4.]]}), ("endpoints", {"": [1., 2., 3., 4.]}),
    ("endpoints", {1: [1., 2., 3., 4.]})
])
def test_invalid_shapes_values_names_and_temporal_overlap_fail_closed(field, value):
    values = example()
    values[field] = value
    with pytest.raises(ValueError):
        choose(**values)


@pytest.mark.parametrize("noise", [0., -1., np.nan, np.inf, 1e300, 1e-300, True, ".1", None])
def test_invalid_noise_rejected(noise):
    with pytest.raises(ValueError, match="log_noise_sd"):
        choose(**example(), log_noise_sd=noise)


def test_non_gaussian_disclaimer_and_full_finite_spread_are_returned():
    result = choose(**example())
    assert "not posterior" in " ".join(result["assumptions"])
    assert "not a guarantee" in " ".join(result["assumptions"])
    assert result["goals"]["auc"]["finite_spread"]["log_span"] == pytest.approx(2.)
    assert result["concentration_finite_spread"]["log_span"] == pytest.approx([2., 6.])
