from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from pbpk_kb.blueprints.caffeine_v3_parameters import AbsorptionSetProtocol
from pbpk_kb.model_runtime.caffeine_v3_identification import (
    VALIDATION_SCENARIOS,
    CaffeineV3IdentificationThresholds,
    CaffeineV3UncertaintyProtocol,
    _set_reproduction_checks,
    _set_validation_reproduction_checks,
)


def test_v3_thresholds_lock_point_and_set_identification_gates() -> None:
    thresholds = CaffeineV3IdentificationThresholds()

    assert thresholds.minimum_converged_starts == 5
    assert thresholds.top_run_count == 5
    assert thresholds.profile_grid_points == 31
    assert thresholds.disposition_observation_count == 17
    assert thresholds.disposition_parameter_count == 1
    assert thresholds.prediction_set_fold_span_max == 1.10


def test_v3_uncertainty_preserves_right_censored_set_semantics() -> None:
    protocol = CaffeineV3UncertaintyProtocol(
        fraction_unbound_mean=0.65,
        fraction_unbound_sd=0.03,
        renal_clearance_bootstrap_values=[1, 2, 3, 4, 5, 6],
    )

    assert protocol.sample_count == 512
    assert protocol.clearance_sampling == "truncated_normal_from_iv_hessian"
    assert protocol.permeability_sampling == (
        "stratified_log_uniform_over_admissible_numerical_set"
    )
    assert protocol.permeability_upper_bound_semantics == (
        "right_censored_at_numerical_guard"
    )


def test_v3_validation_scenarios_remain_locked_and_external() -> None:
    assert VALIDATION_SCENARIOS == (
        "HOLD_WAHLLANDER_IV",
        "EXT_EUDRACT_PO",
        "EXT_PMC535_PO",
        "EXT2_PMC572_PO_MALE",
        "EXT2_PMC572_PO_FEMALE",
    )


def test_v3_identification_r_script_locks_order_and_no_validation_read() -> None:
    root = Path(__file__).parents[2]
    script = (
        root
        / "pbpk_kb"
        / "model_runtime"
        / "R"
        / "caffeine_v3_identify.R"
    ).read_text(encoding="utf-8")

    for required in (
        'calibration_names <- c("CAL_BLANCHARD_IV", "CAL_BLANCHARD_PO")',
        'point_ids, clearance_id',
        'set_ids, permeability_id',
        'method = "L-BFGS-B"',
        "clearance_starts",
        "profile_grid_points",
        "disposition_observation_count",
        "permeability_candidates",
        "build_plateau_comparisons",
        "derive_admissible_set",
        "absorption_candidates.csv",
        "absorption_set_manifest.json",
        '"right_censored_numerical_guard"',
        "validation_package_loaded = FALSE",
        '"identification_accepted_validation_locked"',
    ):
        assert required in script
    assert "locked_validation_package_path" not in script


def test_v3_validation_r_script_requires_full_set_and_repeated_predictions() -> None:
    root = Path(__file__).parents[2]
    script = (
        root
        / "pbpk_kb"
        / "model_runtime"
        / "R"
        / "caffeine_v3_validate.R"
    ).read_text(encoding="utf-8")

    identification_gate = script.index(
        "Identification manifest does not unlock validation."
    )
    package_read = script.index("validation_package <- fromJSON")
    assert identification_gate < package_read
    for required in (
        "admissible_candidate_ids",
        "run_validation_set(1)",
        "run_validation_set(2)",
        "set_validation_metrics.csv",
        "set_secondary_validation_metrics.csv",
        "set_endpoint_stability.csv",
        "prediction_packages_repeatable",
        "prediction_set_fold_span_max",
        "lhs_matrix",
        "uncertainty_parameter_samples.csv",
        "visual_predictive_check_quantiles.csv",
        "uncertainty_observation_coverage.csv",
        "uncertainty_prcc.csv",
        '"externally_validated_with_set_identified_permeability"',
        "parameter_identifiability = FALSE",
        "prediction_identifiability = fully_validated",
        "registration_allowed = fully_validated",
    ):
        assert required in script


def test_python_auditor_reproduces_contiguous_absorption_set(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "absorption_candidates.csv"
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "repeat_id",
                "candidate_id",
                "auc_iv_0_24",
                "auc_po_0_24",
                "f_model",
                "tmax_po_min",
                "status",
            ),
        )
        writer.writeheader()
        for repeat_id in (1, 2):
            for candidate_id in range(1, 72):
                writer.writerow(
                    {
                        "repeat_id": repeat_id,
                        "candidate_id": candidate_id,
                        "auc_iv_0_24": 10,
                        "auc_po_0_24": 10,
                        "f_model": 1,
                        "tmax_po_min": 30,
                        "status": "ok",
                    }
                )

    comparison_path = tmp_path / "absorption_plateau_comparisons.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("repeat_id", "lower_candidate_id", "accepted"),
        )
        writer.writeheader()
        for repeat_id in (1, 2):
            for candidate_id in range(1, 62):
                writer.writerow(
                    {
                        "repeat_id": repeat_id,
                        "lower_candidate_id": candidate_id,
                        "accepted": True,
                    }
                )

    set_manifest_path = tmp_path / "absorption_set_manifest.json"
    set_manifest_path.write_text(
        json.dumps(
            {
                "admissible_candidate_ids": list(range(1, 72)),
                "accepted": True,
            }
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(absorption_set_protocol=AbsorptionSetProtocol())

    checks = _set_reproduction_checks(
        candidate_path=candidate_path,
        comparison_path=comparison_path,
        set_manifest_path=set_manifest_path,
        config=config,  # type: ignore[arg-type]
    )

    assert all(checks.values())


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_python_auditor_recalculates_full_set_validation(
    tmp_path: Path,
) -> None:
    scenario_names = [f"scenario_{index}" for index in range(1, 6)]
    point_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    secondary_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []
    for repeat_id in (1, 2):
        for scenario_name in scenario_names:
            dataset_id = f"dataset_{scenario_name}"
            point_rows.append(
                {
                    "repeat_id": repeat_id,
                    "candidate_id": 1,
                    "dataset_id": dataset_id,
                    "included": True,
                    "observed_umol_per_l": 1,
                    "predicted_umol_per_l": 1,
                    "source_artifact_id": "artifact",
                    "source_location": "table 1",
                }
            )
            metric_rows.append(
                {
                    "repeat_id": repeat_id,
                    "candidate_id": 1,
                    "dataset_id": dataset_id,
                    "included_count": 1,
                    "positive_count": 1,
                    "gmfe": 1,
                    "geometric_bias": 1,
                    "log_rmse": 0,
                    "two_fold_fraction": 1,
                    "zero_baseline_pass": True,
                    "finite_nonnegative": True,
                    "accepted": True,
                }
            )
            for endpoint, unit in (
                ("auc_0_24_h_umol_h_per_l", "umol*h/l"),
                ("cmax_umol_per_l", "umol/l"),
                ("tmax_h", "h"),
            ):
                endpoint_rows.append(
                    {
                        "repeat_id": repeat_id,
                        "candidate_id": 1,
                        "scenario_name": scenario_name,
                        "endpoint": endpoint,
                        "predicted_value": 1,
                        "unit": unit,
                    }
                )
        for endpoint in ("auc_0_t", "auc_0_inf", "cmax", "tmax", "half_life"):
            secondary_rows.append(
                {
                    "repeat_id": repeat_id,
                    "candidate_id": 1,
                    "endpoint": endpoint,
                    "predicted_value": 1,
                    "observed_value": 1,
                    "fold_error": 1,
                    "absolute_error": 0,
                    "accepted": True,
                    "metric_id": "metric",
                    "source_artifact_id": "artifact",
                    "source_location": "table 2",
                }
            )
            endpoint_rows.append(
                {
                    "repeat_id": repeat_id,
                    "candidate_id": 1,
                    "scenario_name": "EXT2_PMC572_WEIGHTED",
                    "endpoint": endpoint,
                    "predicted_value": 1,
                    "unit": "h" if endpoint in {"tmax", "half_life"} else "unit",
                }
            )

    stability_rows = [
        {
            "scenario_name": row["scenario_name"],
            "endpoint": row["endpoint"],
            "candidate_count": 1,
            "minimum": 1,
            "maximum": 1,
            "max_min_fold_span": 1,
            "accepted": True,
        }
        for row in endpoint_rows
        if row["repeat_id"] == 1
    ]
    _write_rows(tmp_path / "set_point_predictions_and_residuals.csv", point_rows)
    _write_rows(tmp_path / "set_validation_metrics.csv", metric_rows)
    _write_rows(
        tmp_path / "set_secondary_validation_metrics.csv", secondary_rows
    )
    _write_rows(tmp_path / "set_endpoint_predictions.csv", endpoint_rows)
    _write_rows(tmp_path / "set_endpoint_stability.csv", stability_rows)
    config = SimpleNamespace(
        thresholds=CaffeineV3IdentificationThresholds()
    )

    checks = _set_validation_reproduction_checks(
        validation_root=tmp_path,
        config=config,  # type: ignore[arg-type]
        admissible_count=1,
    )

    assert all(checks.values())
