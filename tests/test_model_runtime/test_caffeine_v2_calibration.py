from __future__ import annotations

from pathlib import Path

from pbpk_kb.model_runtime.caffeine_v2_calibration import (
    CaffeineV2CalibrationThresholds,
    VALIDATION_SCENARIOS,
    _artifact_inventory_checks,
    _metric_reproduction_checks,
    _optional_close,
    _profile_crossing,
)
from pbpk_kb.model_assets.license_policy import sha256_file


def test_v2_thresholds_lock_paper_grade_calibration_gates() -> None:
    thresholds = CaffeineV2CalibrationThresholds()

    assert thresholds.minimum_converged_starts == 5
    assert thresholds.top_run_count == 5
    assert thresholds.profile_grid_points == 21
    assert thresholds.profile_confidence_level == 0.95
    assert thresholds.boundary_margin_fraction == 0.01
    assert thresholds.hessian_condition_number_max == 100000000.0


def test_validation_scenarios_exclude_both_calibration_arms() -> None:
    assert VALIDATION_SCENARIOS == (
        "HOLD_WAHLLANDER_IV",
        "EXT_EUDRACT_PO",
        "EXT_PMC535_PO",
        "EXT2_PMC572_PO_MALE",
        "EXT2_PMC572_PO_FEMALE",
    )


def test_v2_r_protocol_locks_split_mapping_and_identifiability() -> None:
    script = (
        Path(__file__).parents[2]
        / "pbpk_kb"
        / "model_runtime"
        / "R"
        / "caffeine_v2_fit_validate.R"
    ).read_text(encoding="utf-8")

    for required in (
        'calibration_names <- c("CAL_BLANCHARD_IV", "CAL_BLANCHARD_PO")',
        "validation_package_loaded = FALSE",
        'method = "L-BFGS-B"',
        "theta_from_z",
        "runtime_values_for",
        "cl_n3_paraxanthine",
        "cl_n1_theobromine",
        "cl_n7_theophylline",
        "numeric_hessian_z",
        "profile_grid_points",
        "crossing_nearest_best",
        "both_profile_likelihood_intervals_finite",
        "both_calibration_arms_meet_curve_standards",
        '"calibration_runtime_failed"',
    ):
        assert required in script


def test_v2_validation_protocol_locks_unlock_and_full_analysis() -> None:
    script = (
        Path(__file__).parents[2]
        / "pbpk_kb"
        / "model_runtime"
        / "R"
        / "caffeine_v2_validate.R"
    ).read_text(encoding="utf-8")

    for required in (
        '"calibration_accepted_validation_locked"',
        "Calibration manifest does not unlock validation",
        "locked_validation_package_sha256",
        "validation_package_loaded_at",
        "point_predictions_and_residuals.csv",
        "secondary_validation_metrics.csv",
        "sample_count",
        "lhs_matrix",
        "visual_predictive_check_quantiles.csv",
        "uncertainty_observation_coverage.csv",
        "partial_spearman",
        "uncertainty_prcc.csv",
        'all(prcc_table$status == "ok")',
        "validation_execution_progress.json",
        "Validation checkpoint identity drifted; resume refused.",
        "Resumed uncertainty parameter sample matrix drifted.",
        "write_rds",
        '"external_study_validated"',
        "registration_allowed = fully_validated",
    ):
        assert required in script


def test_v2_auditor_verifies_exact_artifact_inventory(tmp_path: Path) -> None:
    artifact = tmp_path / "result.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")

    checks = _artifact_inventory_checks(
        root=tmp_path,
        inventory=[
            {
                "relative_path": "result.csv",
                "sha256": sha256_file(artifact),
                "byte_size": artifact.stat().st_size,
            }
        ],
        excluded_names={"manifest.json"},
    )

    assert all(checks.values())


def test_v2_auditor_recalculates_reported_curve_metrics(
    tmp_path: Path,
) -> None:
    points = tmp_path / "points.csv"
    points.write_text(
        "dataset_id,observed_umol_per_l,predicted_umol_per_l,included,"
        "source_artifact_id,source_location\n"
        "dataset-a,1,1,TRUE,artifact-a,page 1\n"
        "dataset-a,2,2,TRUE,artifact-a,page 1\n",
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "dataset_id,included_count,positive_count,gmfe,geometric_bias,"
        "log_rmse,two_fold_fraction,zero_baseline_pass,accepted\n"
        "dataset-a,2,2,1,1,0,1,TRUE,TRUE\n",
        encoding="utf-8",
    )

    checks = _metric_reproduction_checks(
        point_path=points,
        metric_path=metrics,
        thresholds=CaffeineV2CalibrationThresholds(),
    )

    assert all(checks.values())


def test_v2_auditor_recalculates_bilateral_profile_crossings() -> None:
    rows = [
        {"fixed_z": "0", "objective": "3"},
        {"fixed_z": "0.4", "objective": "1.5"},
        {"fixed_z": "0.5", "objective": "1"},
        {"fixed_z": "0.6", "objective": "1.5"},
        {"fixed_z": "1", "objective": "3"},
    ]

    left = _profile_crossing(
        rows,
        best_z=0.5000000000000003,
        side="left",
        threshold=2.0,
    )
    right = _profile_crossing(
        rows,
        best_z=0.5000000000000003,
        side="right",
        threshold=2.0,
    )

    assert left is not None and round(left, 12) == round(4 / 15, 12)
    assert right is not None and round(right, 12) == round(11 / 15, 12)
    assert _optional_close(None, None)
    assert not _optional_close(0.5, None)
