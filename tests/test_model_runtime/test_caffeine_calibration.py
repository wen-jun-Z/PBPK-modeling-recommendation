from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from pbpk_kb.model_runtime.caffeine_calibration import (
    RuntimeParameter,
    _resolve_pkml,
    concentration_metrics,
)


def _row(observed: float, predicted: float, *, included: bool = True) -> dict[str, str]:
    return {
        "observed_umol_per_l": str(observed),
        "predicted_umol_per_l": str(predicted),
        "included": str(included).lower(),
    }


def test_concentration_metrics_reproduce_locked_definitions() -> None:
    metrics = concentration_metrics(
        [
            _row(0.0, 0.0),
            _row(1.0, 0.5),
            _row(2.0, 2.0),
            _row(4.0, 8.0),
            _row(100.0, 1.0, included=False),
        ]
    )

    assert metrics["included_count"] == 4
    assert metrics["positive_count"] == 3
    assert math.isclose(metrics["gmfe"], math.exp(2 * math.log(2) / 3))
    assert math.isclose(metrics["geometric_bias"], 1.0)
    assert metrics["two_fold_fraction"] == 1.0
    assert metrics["zero_baseline_pass"] is True


def test_concentration_metrics_keep_zero_baseline_out_of_log_metrics() -> None:
    metrics = concentration_metrics([_row(0.0, 1e-8), _row(1.0, 1.0)])

    assert metrics["gmfe"] == 1.0
    assert metrics["zero_baseline_pass"] is False


def test_concentration_metrics_reject_negative_prediction() -> None:
    with pytest.raises(Exception, match="Negative concentration"):
        concentration_metrics([_row(1.0, -0.1)])


def test_renal_runtime_parameter_requires_six_source_values() -> None:
    with pytest.raises(ValidationError, match="six source subjects"):
        RuntimeParameter(
            parameter_id="caffeine_renal_plasma_clearance",
            runtime_path="renal",
            origin="derived",
            value=0.01,
            unit="ml/min/kg",
            calibrate=False,
            bootstrap_values=[0.01],
        )


def test_r_protocol_contains_locked_execution_steps() -> None:
    script = (
        Path(__file__).parents[2]
        / "pbpk_kb"
        / "model_runtime"
        / "R"
        / "caffeine_fit_validate.R"
    ).read_text(encoding="utf-8")

    for required in (
        'method = "L-BFGS-B"',
        "numeric_hessian",
        "runSimulationBatches",
        "batch$getVariableParameters()",
        '"batch_api_contract.json"',
        '"calibration_runtime_failed"',
        "multi_start$objective < penalty_objective",
        'execution_mode == "diagnostic"',
        '"calibration_diagnostic_points.csv"',
        '"calibration_parameter_profiles.csv"',
        'result_status <- if (is.null(result$status)) "cached"',
        "holdout_or_external_scenarios_run = FALSE",
        "sample_count",
        "partial_spearman",
        '"external_study_validated"',
    ):
        assert required in script


def test_pkml_resolution_prefers_locked_first_run(tmp_path: Path) -> None:
    path = tmp_path / "run_1" / "CAL_BLANCHARD_IV" / "CAL_BLANCHARD_IV.pkml"
    path.parent.mkdir(parents=True)
    path.write_text("pkml", encoding="utf-8")

    assert _resolve_pkml(tmp_path, "CAL_BLANCHARD_IV") == path.resolve()
