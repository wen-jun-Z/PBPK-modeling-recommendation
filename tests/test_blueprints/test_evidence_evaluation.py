"""Clinical metric alignment and failures in the fixed multi-drug evaluation."""
import math

import pytest

from pbpk_kb.blueprints.evidence_evaluation import aggregate_cases, observed_rows, score_case


def test_equal_time_grid_scoring_preserves_windows_and_derived_endpoints():
    times = [0., 60., 120., 180., 240., 300., 360.]
    values = [8 * math.exp(-t/120) for t in times]
    curve = {"time_min": times, "concentration": values, "time_unit": "min", "unit": "umol/l"}
    # Exclude initial time; scoring must never substitute the model full AUC.
    observations = {"observations": [{"time_min": t, "concentration_umol_per_l": v/2}
                                    for t,v in zip(times[1:],values[1:])]}
    request = {"evidence": [{"parameter": "molecular_weight", "value": 200.}],
               "dosing": {"route": "intravenous", "dose_mg": 10., "start_h": 0., "infusion_minutes": 0.},
               "target": {"terminal_fit": {"start_h": 3., "end_h": 6., "min_points": 4,
                    "min_adjusted_r2": .98, "min_span_half_lives": 1., "max_auc_extrapolated_fraction": .2,
                    "max_aumc_extrapolated_fraction": .2}}}
    result = score_case(curve, observations, request)
    assert result["gmfe"] == pytest.approx(2.)
    assert result["auc"]["fold_error"] == pytest.approx(2.)
    assert result["observed_window_h"] == [1.,6.]
    assert result["sampled_tmax"]["absolute_error_h"] == 0.
    assert result["terminal_half_life"]["fold_error"] == pytest.approx(1.)
    assert result["terminal_half_life"]["observed"]["value_h"] == pytest.approx(math.log(2)*2)


def test_macro_summary_equal_weights_drugs_and_studies_and_keeps_failure_denominator():
    rows = []
    for drug,study,value in [("a","one",1.),("a","one",1.),("a","two",4.),("b","three",8.)]:
        rows.append({"compound_id":drug,"study_id":study,"role":"comparison", "scoring_status":"completed", "metrics":{"gmfe":value}})
    rows.append({"compound_id":"c","study_id":"four","role":"comparison","scoring_status":"failed"})
    result = aggregate_cases(rows)["comparison"]
    assert result["attempted_cases"] == 5 and result["scored_cases"] == 4 and result["compound_count"] == 3
    metric = result["metrics"]["gmfe"]
    assert metric["by_compound"] == pytest.approx({"a":2.,"b":8.})
    assert metric["macro_geometric_mean"] == pytest.approx(4.)
    assert metric["eligible_cases"] == 4 and metric["eligible_studies"] == 3


@pytest.mark.parametrize("times,values", [([1.,1.],[1.,2.]),([1.,2.],[0.,2.]),([-1.,2.],[1.,2.]),([1.,2.],[True,2.]),([1.,2.],[1.,float('nan')])])
def test_invalid_observations_do_not_silently_disappear(times,values):
    with pytest.raises(ValueError):
        observed_rows({"observations":[{"time_min":t,"concentration_umol_per_l":v} for t,v in zip(times,values)]})


def test_observation_extrapolation_is_rejected():
    with pytest.raises(ValueError,match="cover"):
        score_case({"time_min":[0.,60.],"concentration":[1.,.5],"time_unit":"min","unit":"umol/l"},
                   {"observations":[{"time_min":60.,"concentration_umol_per_l":1.},{"time_min":120.,"concentration_umol_per_l":.5}]}, {})
