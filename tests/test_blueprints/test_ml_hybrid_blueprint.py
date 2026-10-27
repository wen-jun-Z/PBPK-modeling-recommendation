"""Prediction-source isolation and bounded ML-to-PBPK mapping contracts."""
from copy import deepcopy
import inspect
import json

import pytest

from pbpk_kb.blueprints import evidence_blueprint as eb
from pbpk_kb.blueprints import ml_hybrid_blueprint as ml


@pytest.fixture
def ml_case(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "inputs").mkdir(parents=True)
    split = root / "inputs/split.json"
    split.write_text('{"purpose":"synthetic fixed split"}')
    model = root / "inputs/model.bin"
    model.write_bytes(b"synthetic model hash fixture")
    predictions = root / "inputs/prediction.json"
    record = {"compound_id": "test:1", "method": "random_forest", "training_overlap": False,
              "same_scaffold_training_overlap": False,
              "predictions": {"clearance_ml_min_kg": 1., "vss_l_kg": 1.7, "fraction_unbound": .7},
              "provenance": {name: {"path": str(path.relative_to(root)), "sha256": ml.sha256(path)}
                             for name, path in (("split", split), ("model", model))}}
    predictions.write_text(json.dumps(record))
    source = {"path": "inputs/prediction.json", "sha256": ml.sha256(predictions), "location": "predictions"}
    other = root / "inputs/physicochemical.json"
    other.write_text('{"purpose":"synthetic independent physicochemical source"}')
    values = {"molecular_weight": 194.19, "molecular_formula": "C8H10N4O2", "logp": -.1,
              "fraction_unbound": .7, "pka_types": [], "solubility": 20., "solubility_ph": 7.,
              "binding_partner": "Albumin", "elimination_scope": "hepatic_only", "total_clearance": 1.}
    evidence = []
    for key, value in values.items():
        predicted = key in {"total_clearance", "fraction_unbound"}
        evidence.append({"parameter": key, "compound_id": "test:1", "value": value,
                         "unit": eb.UNITS[key], "basis": "predicted" if predicted else "measured",
                         "source": deepcopy(source) if predicted else {"path": "inputs/physicochemical.json",
                             "sha256": ml.sha256(other), "location": key},
                         "context": {"species": "human", "matrix": "plasma", "route": "iv"},
                         **({"prediction_method": {"name": "random_forest", "version": "test",
                             "training_overlap": False, "applicability": "synthetic test"}} if predicted else {})})
    ledger = root / "inputs/ledger.json"
    ledger.write_text(json.dumps({"evidence": evidence}))
    request = {"entry_reason": "model_missing", "compound_id": "test:1", "compound_name": "TestDrug",
               "human": {"age_years": 30., "weight_kg": 70., "height_cm": 175., "sex": "MALE",
                         "population": "European_ICRP_2002"},
               "dosing": {"route": "intravenous", "dose_mg": 10., "infusion_minutes": 15.},
               "target": {"endpoints": ["auc", "cmax"], "primary_endpoint": "auc",
                          "window_start_h": 0., "window_end_h": 12.},
               "partition_method": "PKSimStandard", "evidence": evidence,
               "evidence_ledger": {"path": "inputs/ledger.json", "sha256": ml.sha256(ledger), "location": "evidence"}}
    monkeypatch.setattr(eb, "load_standard_human_model", lambda _: {"purpose": "generic unit fixture"})
    return root, request, source


def fake_run(snapshot, request, output, root, *, audit):
    ml.save(output / "prediction_locked.json", {"purpose": "synthetic curve"})
    ml.save(output / "endpoints.json", {"parameter_construction": "original shared helper text",
                                       "parameters_fitted": True, "construction_parameters_fitted": True})
    return {"status": "predicted", "clearance_l_h": 4.2, "vss_l": 119., "native_audits_passed": audit}


def test_mapping_uses_prediction_targets_not_measured_evidence(ml_case, monkeypatch):
    root, request, source = ml_case
    before = deepcopy(request)
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs["audit"])
        return fake_run(*args, **kwargs)
    monkeypatch.setattr(ml, "_run", run)
    result = ml.run_ml_hybrid_case(request=request, predicted_pk_source=source, output_dir=root / "mapped",
                                   project_root=root, max_evaluations=1)
    fit = json.loads((root / "mapped/mapping_locked.json").read_text())
    assert result["status"] == "predicted"
    assert calls == [False, True]
    assert result["native_evaluations"] == 2
    assert fit["targets"] == {"clearance_l_h": pytest.approx(4.2), "vss_l": 119.}
    assert fit["target_basis"] == "predicted"
    assert fit["observations_used"] is False and fit["clinical_validation"] is False
    assert "ML_predicted_target" in fit["optimizer"]["objective"]
    assert request == before
    assert result["direct_prediction_path"]
    endpoints = json.loads((root / "mapped/final_prediction/endpoints.json").read_text())
    assert endpoints["parameter_construction"] == ml.CONSTRUCTION
    assert endpoints["construction_target_basis"] == "predicted"
    assert not any("observ" in parameter or "curve" in parameter
                   for parameter in inspect.signature(ml.run_ml_hybrid_case).parameters)


def test_direct_baseline_has_same_horizon_and_audits_without_mapping(ml_case, monkeypatch):
    root, request, source = ml_case
    seen = []
    def run(snapshot, *args, **kwargs):
        seen.append((snapshot, kwargs["audit"]))
        return fake_run(snapshot, *args, **kwargs)
    monkeypatch.setattr(ml, "_run", run)
    result = ml.run_ml_direct_case(request=request, predicted_pk_source=source,
                                   output_dir=root / "direct", project_root=root)
    assert result["status"] == "predicted" and result["native_evaluations"] == 1
    assert seen[0][1] is True
    schema = seen[0][0]["Simulations"][0]["OutputSchema"]
    assert next(p["Value"] for p in schema[-1]["Parameters"] if p["Name"] == "End time") == 1344.
    endpoints = json.loads((root / "direct/endpoints.json").read_text())
    assert endpoints["parameters_fitted"] is False
    assert endpoints["construction_parameters_fitted"] is False
    assert not (root / "direct/mapping_locked.json").exists()


@pytest.mark.parametrize("change", ["source_hash", "identity", "overlap", "scaffold_overlap", "method",
                                    "model_hash", "split_hash", "fu_range", "vss_missing", "source_location"])
def test_source_identity_and_training_provenance_checked_before_native(ml_case, monkeypatch, change):
    root, request, source = ml_case
    path = root / source["path"]
    record = json.loads(path.read_text())
    if change == "source_hash": source["sha256"] = "0" * 64
    elif change == "source_location": source["location"] = ""
    elif change == "identity": record["compound_id"] = "test:other"
    elif change == "overlap": record["training_overlap"] = True
    elif change == "scaffold_overlap": record["same_scaffold_training_overlap"] = True
    elif change == "method": record["method"] = "unknown"
    elif change == "model_hash": record["provenance"]["model"]["sha256"] = "0" * 64
    elif change == "split_hash": record["provenance"]["split"]["sha256"] = "0" * 64
    elif change == "fu_range": record["predictions"]["fraction_unbound"] = 1.5
    elif change == "vss_missing": record["predictions"].pop("vss_l_kg")
    if change not in {"source_hash", "source_location"}:
        path.write_text(json.dumps(record))
        source["sha256"] = ml.sha256(path)
    monkeypatch.setattr(ml, "_run", lambda *a, **k: pytest.fail("Invalid source reached execution"))
    with pytest.raises(ValueError):
        ml.run_ml_hybrid_case(request=request, predicted_pk_source=source, output_dir=root / "invalid", project_root=root)
    assert not (root / "invalid").exists()


@pytest.mark.parametrize("parameter,change", [("total_clearance", "measured"), ("fraction_unbound", "measured"),
                                            ("total_clearance", "value"), ("fraction_unbound", "source"),
                                            ("fraction_unbound", "overlap")])
def test_predicted_inputs_cannot_be_relabelled_or_mismatched(ml_case, monkeypatch, parameter, change):
    root, request, source = ml_case
    entry = next(e for e in request["evidence"] if e["parameter"] == parameter)
    if change == "measured": entry["basis"] = "measured"
    elif change == "value": entry["value"] *= 2
    elif change == "source": entry["source"]["path"] = "inputs/physicochemical.json"
    elif change == "overlap": entry["prediction_method"]["training_overlap"] = True
    monkeypatch.setattr(ml, "_run", lambda *a, **k: pytest.fail("Mismatched prediction reached execution"))
    with pytest.raises(ValueError):
        ml.run_ml_direct_case(request=request, predicted_pk_source=source, output_dir=root / "invalid", project_root=root)


@pytest.mark.parametrize("budget", [0, 13, True, 1.5])
def test_budget_is_bounded_before_execution(ml_case, budget):
    root, request, source = ml_case
    with pytest.raises(ValueError, match="max_evaluations"):
        ml.run_ml_hybrid_case(request=request, predicted_pk_source=source, output_dir=root / "invalid",
                             project_root=root, max_evaluations=budget)


def test_failed_mapping_keeps_unaccepted_attempt_and_never_runs_final(ml_case, monkeypatch):
    root, request, source = ml_case
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs["audit"])
        result = fake_run(*args, **kwargs)
        result["clearance_l_h"] = 1.
        return result
    monkeypatch.setattr(ml, "_run", run)
    result = ml.run_ml_hybrid_case(request=request, predicted_pk_source=source, output_dir=root / "failed",
                                   project_root=root, max_evaluations=1)
    assert calls == [False]
    assert result["status"] == "mapping_failed"
    assert result["mapping_status"] == "budget_exhausted"
    assert result["prediction_path"] is None and result["endpoints_path"] is None
    assert result["direct_prediction_path"]
    assert not (root / "failed/final_prediction").exists()


def test_final_audited_run_must_preserve_mapping_acceptance(ml_case, monkeypatch):
    root, request, source = ml_case
    def run(*args, **kwargs):
        result = fake_run(*args, **kwargs)
        if kwargs["audit"]:
            result["vss_l"] = 20.
        return result
    monkeypatch.setattr(ml, "_run", run)
    result = ml.run_ml_hybrid_case(request=request, predicted_pk_source=source, output_dir=root / "disagrees",
                                   project_root=root, max_evaluations=1)
    assert result["status"] == "final_mapping_verification_failed"
    assert result["prediction_path"]  # Preserved for diagnosis, explicitly not accepted.


@pytest.mark.parametrize("change", ["partition", "clinical_tail", "forbidden"])
def test_fixed_construction_cannot_use_comparison_guidance(ml_case, change):
    root, request, source = ml_case
    if change == "partition": request["partition_method"] = "RodgersRowland"
    elif change == "clinical_tail": request["target"]["terminal_fit"] = {"start_h": 8., "end_h": 12.}
    else: request["forbidden_sources"] = [deepcopy(source)]
    with pytest.raises(ValueError):
        ml.run_ml_direct_case(request=request, predicted_pk_source=source, output_dir=root / "invalid", project_root=root)


def test_existing_outputs_not_overwritten(ml_case):
    root, request, source = ml_case
    (root / "existing").mkdir()
    with pytest.raises(FileExistsError):
        ml.run_ml_direct_case(request=request, predicted_pk_source=source, output_dir=root / "existing", project_root=root)
