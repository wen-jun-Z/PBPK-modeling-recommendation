"""Training-only model-selection contracts; no clinical labels or native runs."""
import math
import numpy as np
import pytest
from pbpk_kb.blueprint_learning import clearance_refinement as cl


def test_candidate_units_and_predeclared_formulas():
    auxiliary = math.log10(.6) - math.log10(2.) - math.log10(.06)
    assert 10. ** auxiliary == pytest.approx(5.)
    c = cl.candidate_predictions([0., 2.], [2., 0.], [1., 1.], [4., 4.])
    assert np.allclose(c["retained_rf_log_mean"], [1., 1.])
    assert np.allclose(c["retained_rf_auxiliary_log_median"], [1., 1.])
    with pytest.raises(ValueError):
        cl.candidate_predictions([0.], [0., 1.], [0.], [0.])


def test_cl_only_selection_tie_guard_and_shared_denominators():
    all_same = {name: np.array([0., 1.]) for name in cl.CANDIDATES}
    result = cl.select_candidate([0., 1.], all_same, [0, 1])
    assert result["selected"] == "retained_projected_cl"
    assert all(r["n"] == 2 for r in result["scores"].values())
    all_same["xgb_depth3_log_l1"] = np.array([0., 0.])
    result = cl.select_candidate([0., 0.], all_same, [0, 1])
    assert result["selected"] == "xgb_depth3_log_l1"
    assert result["independent_post_selection_evaluation"] is False
    with pytest.raises(ValueError):
        cl.select_candidate([0.], all_same, [0])


def test_strict_threshold_counts_and_tail_diagnostics():
    predictions = {name: np.log10([2., 3., 5., 1.]) for name in cl.CANDIDATES}
    row = cl.select_candidate(np.zeros(4), predictions, [0, 0, 1, 1])["scores"][cl.CANDIDATES[0]]
    assert row["threshold_comparison"] == "<"
    assert row["within_counts"] == {"2": 1, "3": 2, "5": 3}
    assert row["fold_error_max"] == pytest.approx(5.)


def test_l1_changes_objective_only_and_pins_fail_closed(tmp_path):
    l1 = cl.make_l1_estimator(1, "cpu")
    original = cl.rich.make_estimator("rich_xgb_depth3", cl.shared.SEED, 1, "cpu")
    a, b = l1.named_steps["model"].get_params(), original.named_steps["model"].get_params()
    assert {key for key in a if a[key] != b[key]
            and not (isinstance(a[key], float) and isinstance(b[key], float)
                     and math.isnan(a[key]) and math.isnan(b[key]))} == {"objective"}
    assert a["objective"] == "reg:absoluteerror" and a["n_estimators"] == 500
    path = tmp_path / "source.txt"; path.write_text("fixed")
    pins = {"source.txt": cl.shared._sha(path)}
    cl._verify(tmp_path, pins)
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        cl._verify(tmp_path, pins)
