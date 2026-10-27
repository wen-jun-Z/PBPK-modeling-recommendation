"""Mocked adapter contracts: synthetic development data, no native or target fits."""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import axes_calibration as cal
from pbpk_kb.blueprint_policy import axes_optimizer
from pbpk_kb.blueprint_policy import native_axes_runtime


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path
    request = {
        "compound_id": "synthetic:A", "compound_name": "Synthetic A", "evidence": [],
        "evidence_ledger": [], "partition_method": "synthetic-formula",
        "human": {"weight_kg": 70.},
        "dosing": {"route": "intravenous", "dose_mg": 100.},
    }
    observed = {
        "compound_id": request["compound_id"], "role": "calibration_data_reconstruction",
        "matrix": "plasma", "unit": "umol/l", "time_unit": "min",
        "human": deepcopy(request["human"]), "dosing": deepcopy(request["dosing"]),
        "observations": [{"time_min": float(i), "concentration_umol_per_l": value}
                         for i, value in enumerate([1., .8, .5, .2, .1], start=1)],
    }
    cal.save(root / "development.json", observed)
    source = {"path": "development.json", "sha256": cal.sha256(root / "development.json"),
              "role": "calibration_data_reconstruction", "location": "synthetic fixture"}
    code_path = root / "synthetic_policy.py"
    code_path.write_text("# pinned synthetic policy\n")
    monkeypatch.setattr(cal, "_code_pins", lambda _: {"synthetic_policy.py": cal.sha256(code_path)})

    class Validated:
        def __init__(self, value):
            self.value = value

        def model_dump(self, **kwargs):
            return deepcopy(self.value)

    class Request:
        @staticmethod
        def model_validate(value):
            return Validated(value)

    monkeypatch.setattr(cal, "EvidenceBlueprintRequest", Request)
    state = {"base_calls": 0, "trial_calls": [], "outcomes": []}

    def base(req, output, project_root, before_trial):
        receipt = output / "completion.json"
        if receipt.exists():
            result = cal._read(receipt)
            assert result["request"] == req
            cal._verify(result["artifact_pins"], project_root)
            return result
        if before_trial:
            before_trial()
        state["base_calls"] += 1
        output.mkdir(parents=True)
        manifest = output / "native_manifest.json"
        cal.save(manifest, {"engineering_accepted": True, "fixture": True})
        result = {"request": deepcopy(req), "execution_manifest": cal._ref(manifest, project_root),
                  "actual_native_solver_calls": 1, "charged_native_attempts": 1,
                  "artifact_pins": cal._artifacts(output, project_root)}
        cal.save(receipt, result)
        return result

    def native(*, scales, output_dir, project_root, **kwargs):
        folder = Path(output_dir)
        folder.mkdir(parents=True)
        state["trial_calls"].append({"parameters": deepcopy(scales), "folder": folder})
        outcome = state["outcomes"].pop(0) if state["outcomes"] else "predicted"
        # Infrastructure failure here occurs after entering the solver. It must
        # remain included in actual solver-call totals when its retry succeeds.
        count = 0 if outcome == "scientific_failed" else 1
        cal.save(folder / "solver_receipt.json", {"actual_native_solver_calls": count})
        cal.save(folder / "native_audit.json", {"status": outcome, "parameters": scales})
        if outcome != "predicted":
            return {"status": outcome, "actual_native_solver_calls": count,
                    "failure": "synthetic worker or capacity failure"}
        z = np.array([math.log(scales[name]) / math.log(axes_optimizer.BOUNDS[name][1])
                      for name in axes_optimizer.PARAMETERS])
        delta = np.array([z[0] + .3, z[1] - .4, z[2] + .2, z[0] - z[1], z[1] + z[2] - .1])
        values = np.array([r["concentration_umol_per_l"] for r in observed["observations"]]) * np.exp(delta)
        curve = {"unit": "µmol/l", "time_unit": "min",
                 "time_min": [r["time_min"] for r in observed["observations"]],
                 "concentration": values.tolist()}
        return {"status": "predicted", "actual_native_solver_calls": count, "curve": curve}

    monkeypatch.setattr(cal, "_base", base)
    monkeypatch.setattr(native_axes_runtime, "run_axes_trial", native)
    return root, request, source, observed, state


def fit(case, **options):
    root, request, source, _, _ = case
    defaults = {"reference_request": request, "training_source": source,
                "output_dir": root / "fit", "project_root": root, "max_evaluations": 12}
    defaults.update(options)
    return cal.fit_mapping(**defaults)


def test_base_and_all_trials_share_one_declared_budget(case):
    result = fit(case, max_evaluations=9)
    state = case[-1]
    assert result["eligible"] and result["decision"]["status"] == "ok"
    assert state["base_calls"] == 1
    assert result["evaluation_count"] == len(state["trial_calls"]) == 8
    assert result["total_native_calls"] == result["actual_solver_calls"] == 9
    assert result["actual_new_calls"] == 9
    assert result["optimizer"]["native_trial_budget"] == 8
    assert result["native_budget_includes_base"] and result["base_solver_calls"] == 1
    assert not result["comparison_curve_used"]


def test_native_predicted_status_and_loss_are_aligned_with_optimizer(case):
    result = fit(case, max_evaluations=5)
    values = [r["concentration_umol_per_l"] for r in case[3]["observations"]]
    for row in result["evaluations"]:
        assert row["status"] == "ok"
        vector = axes_optimizer.residual_vector(row, values)
        assert len(vector) == len(values) + 3
        assert float(vector @ vector) == pytest.approx(row["objective"])


def test_cached_replay_after_pause_is_deterministic(case):
    root, _, _, _, state = case
    paused = False

    def stop_once():
        nonlocal paused
        if len(state["trial_calls"]) == 4 and not paused:
            paused = True
            raise cal.InfrastructurePause("synthetic scheduled pause")

    with pytest.raises(cal.InfrastructurePause, match="scheduled pause"):
        fit(case, before_trial=stop_once)
    assert not (root / "fit/mapping_locked.json").exists()
    resumed = fit(case)
    resumed_calls = deepcopy(state["trial_calls"])
    assert state["base_calls"] == 1
    assert len(resumed_calls) == 11
    assert resumed["actual_new_calls"] == 7
    reference = fit(case, output_dir=root / "fresh")
    assert [row["parameters"] for row in resumed["evaluations"]] == [
        row["parameters"] for row in reference["evaluations"]]
    assert [row["objective"] for row in resumed["evaluations"]] == [
        row["objective"] for row in reference["evaluations"]]
    assert [row["parameters"] for row in resumed_calls] == [
        row["parameters"] for row in state["trial_calls"][11:]]


def test_infrastructure_retry_and_solver_entry_both_count_against_budget(case):
    case[-1]["outcomes"] = ["predicted", "infrastructure_failed"]
    with pytest.raises(cal.InfrastructurePause):
        fit(case, max_evaluations=7)
    result = fit(case, max_evaluations=7)
    assert result["total_native_calls"] == result["actual_solver_calls"] == 7
    assert len(case[-1]["trial_calls"]) == 6
    assert result["evaluation_count"] == 5  # one attempted trial had no completed score
    assert result["budget_exhausted"] and result["eligible"]


def test_cleanup_preserves_selected_failed_base_and_compact_trials_and_allows_reuse(case):
    case[-1]["outcomes"] = ["predicted", "scientific_failed"]
    result = fit(case)
    root = case[0]
    selected = root / result["decision"]["native_output_dir"]
    failed = [root / row["native_output_dir"] for row in result["evaluations"] if row["status"] != "ok"]
    removed = cal.cleanup_unselected(output_dir=root / "fit", project_root=root)
    assert removed["removed_count"] > 0 and selected.is_dir()
    assert failed and all(path.is_dir() for path in failed)
    assert (root / "fit/base/completion.json").is_file()
    for row in result["evaluations"]:
        assert (root / "fit/trials" / row["cache_key"] / "trial.json").is_file()
    count = len(case[-1]["trial_calls"])
    replay = fit(case)
    assert replay["actual_new_calls"] == 0 and len(case[-1]["trial_calls"]) == count
    assert replay["decision"] == result["decision"]


@pytest.mark.parametrize("change", ["code", "source", "budget", "active_parameters"])
def test_changed_inputs_or_code_cannot_reuse_locked_mapping(case, change):
    fit(case, max_evaluations=5)
    root = case[0]
    options = {"max_evaluations": 5}
    if change == "code":
        (root / "synthetic_policy.py").write_text("# changed policy\n")
    elif change == "source":
        (root / "development.json").write_text("{}")
    elif change == "budget":
        options["max_evaluations"] = 6
    else:
        options["active_parameters"] = ("clearance_scale",)
    count = len(case[-1]["trial_calls"])
    with pytest.raises(ValueError):
        fit(case, **options)
    assert len(case[-1]["trial_calls"]) == count


def test_forbidden_comparison_source_cannot_be_used_as_training_data(case):
    case[1]["forbidden_sources"] = [dict(case[2])]
    with pytest.raises(ValueError, match="Comparison observations cannot enter calibration"):
        fit(case)
    assert case[-1]["base_calls"] == 0 and not case[-1]["trial_calls"]


def test_development_conditions_must_match_fitting_request(case):
    request = deepcopy(case[1])
    request["dosing"]["dose_mg"] = 200.
    with pytest.raises(ValueError, match="Development study conditions differ"):
        fit(case, reference_request=request)
    assert not case[-1]["trial_calls"]


def test_prediction_rejects_changed_drug_evidence_before_native_work(case):
    result = fit(case, max_evaluations=5)
    request = deepcopy(case[1])
    request["partition_method"] = "different-formula"
    count = len(case[-1]["trial_calls"])
    with pytest.raises(ValueError, match="drug evidence or formula differs"):
        cal.predict_mapping(request=request, mapping_path=case[0] / "fit/mapping_locked.json",
                            output_dir=case[0] / "prediction", project_root=case[0])
    assert len(case[-1]["trial_calls"]) == count and result["eligible"]


@pytest.mark.parametrize("unit", ["umol/l", "µmol/l", "μmol/l", "µmol/L"])
def test_native_molar_unit_aliases_are_normalized_without_mutating_input(unit):
    curve = {"time_unit": "min", "unit": unit, "time_min": [0., 1., 2.],
             "concentration": [-1e-12, 1., .5]}
    original = deepcopy(curve)
    normalized = cal.normalize_curve(curve)
    assert normalized["unit"] == "umol/l"
    assert normalized["concentration"] == [0., 1., .5]
    assert normalized["numerical_negative_values_clipped"] == 1
    assert curve == original


@pytest.mark.parametrize("updates", [
    {"unit": "mg/l"}, {"time_unit": "h"}, {"time_min": [0., 0.]},
    {"concentration": [-.1, 1.]}, {"concentration": [float("nan"), 1.]},
])
def test_wrong_units_or_invalid_native_trajectory_are_rejected(updates):
    curve = {"time_unit": "min", "unit": "umol/l", "time_min": [0., 1.], "concentration": [0., 1.]}
    curve.update(updates)
    with pytest.raises(ValueError):
        cal.normalize_curve(curve)


def test_score_uses_observed_times_and_fixed_penalties_with_no_extrapolation():
    observed = {"observations": [{"time_min": 1., "concentration_umol_per_l": 1.},
                                  {"time_min": 2., "concentration_umol_per_l": 2.}]}
    curve = {"time_unit": "min", "unit": "umol/l", "time_min": [0., 1., 2., 3.],
             "concentration": [0., 2., 4., 1.]}
    parameters = dict(zip(axes_optimizer.PARAMETERS, [2., 10., 100.]))
    result = cal.score_curve(curve, observed, parameters)
    assert result["predicted_at_observed_times"] == [2., 4.]
    assert result["objective"] == pytest.approx(math.log(2) ** 2 + .1 + .01 + .01)
    assert result["curve_metrics"]["gmfe"] == pytest.approx(2.)
    vector = axes_optimizer.residual_vector(dict(result, parameters=parameters), [1., 2.])
    assert float(vector @ vector) == pytest.approx(result["objective"])
    observed["observations"][-1]["time_min"] = 4.
    with pytest.raises(ValueError, match="extrapolation is forbidden"):
        cal.score_curve(curve, observed, parameters)
