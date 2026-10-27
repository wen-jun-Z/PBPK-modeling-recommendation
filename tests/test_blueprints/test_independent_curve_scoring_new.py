import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("independent_curve_score", ROOT / "scripts/score_locked_independent_curves.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(values, statuses=None):
    statuses = statuses or ["quantified"] * len(values)
    return {"subject_id": "test", "sequence": "TR", "period": 2, "treatment": "R",
            "observations": [{"nominal_time_h": i, "concentration_ng_ml": value,
                              "observation_status": status, "source_token": str(value)}
                             for i, (value, status) in enumerate(zip(values, statuses))]}


def test_existing_natural_log_metric_and_source_missing_positions_preserved():
    r = record([None, 100, None, 0, 200], ["left_censored_below_LLOQ", "quantified", "source_missing_unspecified", "quantified", "quantified"])
    out = MODULE.score_subject(r, [0, 60, 120, 180, 240], [0, 2, 3, 0, 4], 100, 10)
    assert out["status"] == "evaluated"
    assert out["log_rmse"] == pytest.approx(0.6931471805599453)
    assert [x["source_index"] for x in out["points"]] == list(range(5))
    assert [x["included"] for x in out["points"]] == [False, True, False, False, True]


@pytest.mark.parametrize("invalid", [0, -1])
def test_nonpositive_prediction_fails_without_clipping(invalid):
    out = MODULE.score_subject(record([100, 200]), [0, 60], [1, invalid], 100, 10)
    assert out["status"] == "failed"
    assert out["log_rmse"] is None
    assert out["error"] == "nonpositive_prediction_at_observation"


def test_below_assay_limit_is_not_promoted_to_quantified_log_target():
    out = MODULE.score_subject(record([5, 100]), [0, 60], [0, 1], 100, 10)
    assert out["status"] == "evaluated"
    assert out["points"][0]["exclusion_reason"] == "below_source_LLOQ_not_quantifiable"


def dump(path, data):
    path.write_text(json.dumps(data))
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def lock_fixture(tmp_path):
    design_path, design_sha = dump(tmp_path / "design.json", {
        "case_id": "synthetic", "schema_version": "pbpk-independent-curve-design-v1",
        "population": {"n": 2, "male_n": 1, "female_n": 1}, "sampling": {"time_h": [0, 1.33]},
        "sealed_observation_reference": {"path": "MUST_NOT_READ_SEALED"}})
    protocol_path, protocol_sha = dump(tmp_path / "protocol.json", {
        "case_id": "synthetic", "design_sha256": design_sha,
        "prediction_aggregation": "sex_weighted_reference_concentration",
        "sex_weights": {"male": 0.5, "female": 0.5}, "molecular_weight_g_mol": 100, "model_key": "source"})
    pred = {"case_id": "synthetic", "models": [{"model_key": "source", "sex": sex, "time_min": [0, 79.8], "concentration": [0, 1], "unit": "umol/l"} for sex in ["MALE", "FEMALE"]]}
    prediction_path, prediction_sha = dump(tmp_path / "prediction.json", pred)
    case = {"case_id": "synthetic", "design_path": design_path, "design_sha256": design_sha, "protocol_path": protocol_path, "protocol_sha256": protocol_sha}
    freeze_path, freeze_sha = dump(tmp_path / "freeze.json", {"cases": [case]})
    lock = {"schema_version": "pbpk-external-predictions-lock-v1", "model_selection_frozen": True,
            "config_freeze": {"path": freeze_path, "sha256": freeze_sha},
            "predictions": [{**case, "path": prediction_path, "sha256": prediction_sha}]}
    lock_path = tmp_path / "lock.json"
    dump(lock_path, lock)
    return lock_path, lock, pred


def test_validation_does_not_open_sealed_and_preserves_rounded_grid(tmp_path):
    path, _, _ = lock_fixture(tmp_path)
    _, checked = MODULE.validate_locked_inputs(path)
    assert checked[0]["time_min"][1] == pytest.approx(79.8)


def test_changed_prediction_hash_rejected_before_sealed(tmp_path):
    path, lock, pred = lock_fixture(tmp_path)
    pred["models"][0]["concentration"][1] = 2
    dump(Path(lock["predictions"][0]["path"]), pred)
    with pytest.raises(ValueError, match="Hash mismatch"):
        MODULE.validate_locked_inputs(path)


def test_rehashed_but_shifted_grid_still_rejected(tmp_path):
    path, lock, pred = lock_fixture(tmp_path)
    pred["models"][0]["time_min"][1] = 80
    _, sha = dump(Path(lock["predictions"][0]["path"]), pred)
    lock["predictions"][0]["sha256"] = sha
    dump(path, lock)
    with pytest.raises(ValueError, match="shifted"):
        MODULE.validate_locked_inputs(path)


def test_final_lock_cannot_drop_preregistered_case(tmp_path):
    path, lock, _ = lock_fixture(tmp_path)
    freeze_path = Path(lock["config_freeze"]["path"])
    frozen = json.loads(freeze_path.read_text())
    frozen["cases"].append({**frozen["cases"][0], "case_id": "second"})
    _, sha = dump(freeze_path, frozen)
    lock["config_freeze"]["sha256"] = sha
    dump(path, lock)
    with pytest.raises(ValueError, match="case set differs"):
        MODULE.validate_locked_inputs(path)


def test_equivalent_pre_run_protocol_map_requires_original_hashed_design(tmp_path):
    path, lock, _ = lock_fixture(tmp_path)
    item = lock["predictions"][0]
    config = {"protocols": {"dose": {"path": item["protocol_path"], "sha256": item["protocol_sha256"]}},
              "frozen_input_sha256": {item["design_path"]: item["design_sha256"]}}
    _, sha = dump(Path(lock["config_freeze"]["path"]), config)
    lock["config_freeze"]["sha256"] = sha
    dump(path, lock)
    assert MODULE.validate_locked_inputs(path)[1][0]["case_id"] == "synthetic"
