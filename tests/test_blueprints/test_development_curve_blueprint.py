"""Development-curve provenance and execution guards; native runs are mocked."""
from copy import deepcopy
import json

import pytest

from pbpk_kb.blueprints import development_curve_blueprint as dc
from pbpk_kb.blueprints import evidence_blueprint as eb
from pbpk_kb.blueprints import pk_anchor_moments as moments
from pbpk_kb.blueprints import pk_anchored_blueprint as pk
from test_pk_anchored_blueprint import evidence_case  # reuse source/ledger fixture


@pytest.fixture
def development_case(evidence_case):
    root, request, anchor = evidence_case
    document = {
        "compound_id": request["compound_id"], "study_id": "test:development",
        "role": "calibration_data_reconstruction", "matrix": "plasma",
        "time_unit": "min", "unit": "umol/l",
        "observations": [
            {"time_min": 15., "concentration_umol_per_l": 1., "unit": "umol/l"},
            {"time_min": 60., "concentration_umol_per_l": .5, "unit": "umol/l"},
            {"time_min": 240., "concentration_umol_per_l": .1, "unit": "umol/l"},
        ],
    }
    path = root / "inputs/development.json"
    path.write_text(json.dumps(document))
    source = {"path": "inputs/development.json", "sha256": pk.sha256(path),
              "role": "calibration_data_reconstruction", "location": "observations"}
    return root, request, anchor, document, source


def _write_training(case):
    root, _, _, document, source = case
    path = root / source["path"]
    path.write_text(json.dumps(document))
    source["sha256"] = pk.sha256(path)


def _native_success(snapshot, request, output, root, *, audit):
    curve = {"time_min": [0., 15., 60., 240., 720.],
             "concentration": [0., 1., .5, .1, .001],
             "time_unit": "min", "unit": "umol/l"}
    (output / "prediction_locked.json").write_text(json.dumps(curve))
    (output / "endpoints.json").write_text(json.dumps({
        "observations_used": False, "parameters_fitted": True,
        "construction_parameters_fitted": True,
        "endpoint_estimator_fitted_pbpk_parameters": False,
        "parameter_construction": "CL and optional Vss evidence anchors; no comparison curve fitting",
        "endpoints": {"auc": {"value": 1.}},
    }))
    return {"status": "predicted", "clearance_l_h": 4.2, "vss_l": 119.,
            "native_audits_passed": audit, "prediction_sha256": pk.sha256(output / "prediction_locked.json")}


def _mapping(case):
    _, request, _, _, source = case
    normalized = eb.EvidenceBlueprintRequest.model_validate(request).model_dump(mode="json", exclude_none=True)
    return {"version": dc.VERSION, "eligible": True,
            "decision": {"lipophilicity": -.1, "clearance_scale": 1.},
            "drug_signature": pk._drug_signature(normalized),
            "training_source": source, "target_vss_anchor": None}


def test_training_source_is_verified_without_mutating_inputs(development_case):
    root, request, _, document, source = development_case
    original = deepcopy(development_case[1:])
    assert dc._training_document(source, request, root) == document
    assert development_case[1:] == original


@pytest.mark.parametrize("mutation", [
    "source_role", "document_role", "missing_role", "wrong_drug", "missing_drug",
    "matrix", "time_unit", "concentration_unit", "missing_unit", "row_unit", "row_time_unit",
    "unsorted", "duplicate", "negative_time", "zero_concentration", "nan", "bool", "one_point",
])
def test_incompatible_training_records_rejected(development_case, mutation):
    root, request, _, document, source = development_case
    if mutation == "source_role": source["role"] = "comparison"
    elif mutation == "document_role": document["role"] = "comparison"
    elif mutation == "missing_role": document.pop("role")
    elif mutation == "wrong_drug": document["compound_id"] = "test:other"
    elif mutation == "missing_drug": document.pop("compound_id")
    elif mutation == "matrix": document["matrix"] = "blood"
    elif mutation == "time_unit": document["time_unit"] = "h"
    elif mutation == "concentration_unit": document["unit"] = "mg/l"
    elif mutation == "missing_unit": document.pop("unit")
    elif mutation == "row_unit": document["observations"][0]["unit"] = "mg/l"
    elif mutation == "row_time_unit": document["observations"][0]["time_unit"] = "h"
    elif mutation == "unsorted": document["observations"].reverse()
    elif mutation == "duplicate": document["observations"][1]["time_min"] = 15.
    elif mutation == "negative_time": document["observations"][0]["time_min"] = -1.
    elif mutation == "zero_concentration": document["observations"][0]["concentration_umol_per_l"] = 0.
    elif mutation == "nan": document["observations"][0]["concentration_umol_per_l"] = float("nan")
    elif mutation == "bool": document["observations"][0]["concentration_umol_per_l"] = True
    elif mutation == "one_point": document["observations"] = document["observations"][:1]
    _write_training(development_case)
    with pytest.raises(ValueError):
        dc._training_document(source, request, root)


@pytest.mark.parametrize("mutation", ["hash", "forbidden_path", "forbidden_hash", "oversize"])
def test_source_integrity_and_comparison_exclusions(development_case, mutation):
    root, request, _, _, source = development_case
    if mutation == "hash": source["sha256"] = "0" * 64
    elif mutation == "forbidden_path":
        request["forbidden_sources"] = [{**source, "sha256": "0" * 64}]
    elif mutation == "forbidden_hash":
        request["forbidden_sources"] = [{**source, "path": "inputs/comparison-copy.json"}]
    else:
        path = root / source["path"]
        path.write_text(" " * (2 * 1024**2 + 1))
        source["sha256"] = pk.sha256(path)
    with pytest.raises(ValueError):
        dc._training_document(source, request, root)


def test_source_cannot_escape_project(development_case):
    root, request, _, _, source = development_case
    (root.parent / "external.json").write_text("{}")
    source["path"] = "../external.json"
    with pytest.raises(ValueError, match="inside"):
        dc._training_document(source, request, root)


def test_fit_preserves_baseline_evidence_and_records_development_use(development_case, monkeypatch):
    root, request, anchor, _, source = development_case
    before = deepcopy(request)
    ledger = root / request["evidence_ledger"]["path"]
    evidence_bytes = ledger.read_bytes()
    training_bytes = (root / source["path"]).read_bytes()
    monkeypatch.setattr(dc, "_run", _native_success)
    fit = dc.fit_development_mapping(reference_request=request, training_source=source,
        vss_anchor=anchor, output_dir=root / "fit", project_root=root, max_evaluations=1)
    assert fit["eligible"] is True
    assert fit["evaluation_count"] == 1
    assert fit["decision"]["clearance_scale"] == 1.
    assert fit["target_clearance_l_h"] == pytest.approx(4.2)
    assert fit["target_vss_l"] == pytest.approx(119.)
    assert fit["development_curve_used"] is True
    assert fit["comparison_curve_used"] is False
    assert fit["independent_validation"] is False
    assert request == before
    assert ledger.read_bytes() == evidence_bytes
    assert (root / source["path"]).read_bytes() == training_bytes
    endpoint = json.loads((root / "fit/attempt_01/endpoints.json").read_text())
    assert endpoint["development_curve_used"] is True
    assert endpoint["observations_used"] is True
    assert endpoint["endpoint_estimator_uses_observations"] is False
    assert endpoint["endpoint_estimator_fitted_pbpk_parameters"] is False


def test_annotate_distinguishes_fitted_construction_from_endpoint_extraction(development_case):
    root, _, _, _, source = development_case
    output = root / "annotation"
    output.mkdir()
    path = output / "endpoints.json"
    path.write_text(json.dumps({"observations_used": False, "parameters_fitted": True,
        "construction_parameters_fitted": True, "endpoint_estimator_fitted_pbpk_parameters": False,
        "endpoints": {"auc": {"value": 123.}}}))
    dc._annotate(output, source)
    result = json.loads(path.read_text())
    assert result["observations_used"] is True
    assert result["endpoint_estimator_uses_observations"] is False
    assert result["development_curve_used"] is True
    assert result["independent_validation"] is False
    assert result["parameters_fitted"] is True
    assert result["endpoint_estimator_fitted_pbpk_parameters"] is False
    assert result["endpoints"] == {"auc": {"value": 123.}}
    assert "development curve calibration" in result["parameter_construction"]


@pytest.mark.parametrize("status", ["native_failed", "parameter_readback_failed", "administration_readback_failed"])
def test_failed_initial_native_run_cannot_create_eligible_mapping(development_case, monkeypatch, status):
    root, request, _, _, source = development_case
    monkeypatch.setattr(dc, "_run", lambda *a, **k: {"status": status})
    fit = dc.fit_development_mapping(reference_request=request, training_source=source,
        vss_anchor=None, output_dir=root / "fit", project_root=root, max_evaluations=4)
    assert fit["eligible"] is False
    assert fit["decision"] is None
    assert fit["status"] == "initial_evaluation_failed"
    assert fit["failed_evaluation_count"] == 1
    assert fit["evaluations"][0]["eligible_for_selection"] is False


def test_fit_detects_training_source_mutation_during_native_run(development_case, monkeypatch):
    root, request, _, _, source = development_case
    def change_source(*args, **kwargs):
        result = _native_success(*args, **kwargs)
        (root / source["path"]).write_text("{}")
        return result
    monkeypatch.setattr(dc, "_run", change_source)
    with pytest.raises(ValueError, match="changed"):
        dc.fit_development_mapping(reference_request=request, training_source=source,
            vss_anchor=None, output_dir=root / "fit", project_root=root, max_evaluations=1)
    assert not (root / "fit/mapping_locked.json").exists()


@pytest.mark.parametrize("mutation", ["version", "ineligible", "missing_decision", "wrong_drug", "changed_evidence", "partition"])
def test_predict_rejects_ineligible_or_different_mapping_before_execution(development_case, monkeypatch, mutation):
    root, request, _, _, _ = development_case
    mapping = _mapping(development_case)
    if mutation == "version": mapping["version"] = "unsupported"
    elif mutation == "ineligible": mapping["eligible"] = False
    elif mutation == "missing_decision": mapping["decision"] = None
    elif mutation == "wrong_drug": request["compound_name"] = "AnotherDrug"
    elif mutation == "changed_evidence": request["evidence"][0]["value"] += 1.
    elif mutation == "partition": request["partition_method"] = "RodgersRowland"
    path = root / "mapping.json"
    path.write_text(json.dumps(mapping))
    monkeypatch.setattr(dc, "_run", lambda *a, **k: pytest.fail("Ineligible mapping reached native run"))
    with pytest.raises(ValueError, match="eligible or belongs"):
        dc.predict_with_development_mapping(request=request, mapping_path=path,
            output_dir=root / "prediction", project_root=root)
    assert not (root / "prediction").exists()


def test_eligible_mapping_allows_new_human_and_dose_without_refitting(development_case, monkeypatch):
    root, request, _, _, _ = development_case
    mapping = _mapping(development_case)
    path = root / "mapping.json"
    path.write_text(json.dumps(mapping))
    original_mapping = path.read_bytes()
    request["human"]["weight_kg"] = 60.
    request["dosing"]["dose_mg"] = 20.
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs["audit"])
        return _native_success(*args, **kwargs)
    monkeypatch.setattr(dc, "_run", run)
    result = dc.predict_with_development_mapping(request=request, mapping_path=path,
        output_dir=root / "prediction", project_root=root)
    assert result["status"] == "predicted"
    assert result["development_curve_used"] is True
    assert result["comparison_curve_used"] is False
    assert calls == [True]
    assert path.read_bytes() == original_mapping
    construction = json.loads((root / "prediction/construction.json").read_text())
    assert construction["mapping_sha256"] == pk.sha256(path)


def test_prediction_propagates_native_failure(development_case, monkeypatch):
    root, request, _, _, _ = development_case
    path = root / "mapping.json"
    path.write_text(json.dumps(_mapping(development_case)))
    monkeypatch.setattr(dc, "_run", lambda *a, **k: {"status": "parameter_readback_failed"})
    result = dc.predict_with_development_mapping(request=request, mapping_path=path,
        output_dir=root / "prediction", project_root=root)
    assert result["status"] == "parameter_readback_failed"
    assert not (root / "prediction/prediction_locked.json").exists()


def test_prediction_detects_mapping_mutation_during_execution(development_case, monkeypatch):
    root, request, _, _, _ = development_case
    path = root / "mapping.json"
    path.write_text(json.dumps(_mapping(development_case)))
    def change_mapping(*args, **kwargs):
        result = _native_success(*args, **kwargs)
        path.write_text("{}")
        return result
    monkeypatch.setattr(dc, "_run", change_mapping)
    with pytest.raises(ValueError, match="mapping changed"):
        dc.predict_with_development_mapping(request=request, mapping_path=path,
            output_dir=root / "prediction", project_root=root)


def test_existing_fit_and_prediction_outputs_are_immutable(development_case, monkeypatch):
    root, request, _, _, source = development_case
    output = root / "existing"
    output.mkdir()
    marker = output / "preserved.txt"
    marker.write_text("keep")
    path = root / "mapping.json"
    path.write_text(json.dumps(_mapping(development_case)))
    monkeypatch.setattr(dc, "_run", lambda *a, **k: pytest.fail("Existing output was reused"))
    with pytest.raises(FileExistsError):
        dc.fit_development_mapping(reference_request=request, training_source=source,
            vss_anchor=None, output_dir=output, project_root=root)
    with pytest.raises(FileExistsError):
        dc.predict_with_development_mapping(request=request, mapping_path=path,
            output_dir=output, project_root=root)
    assert marker.read_text() == "keep"


def test_cl_only_integral_fallback_never_fills_clinical_terminal_endpoints(development_case, monkeypatch):
    root, request, _, _, _ = development_case
    output = root / "cl-only"
    output.mkdir()
    curve = {"time_min": [0., 15., 60.], "concentration": [0., 1., .5],
             "time_unit": "min", "unit": "umol/l"}
    (output / "prediction_locked.json").write_text(json.dumps(curve))
    endpoints = {"endpoints": {
        "clearance": {"status": "not_estimable", "value_l_per_h": None},
        "terminal_half_life": {"status": "not_estimable", "value_h": None},
        "vss": {"status": "not_estimable", "value_l": None},
    }}
    clinical_path = output / "endpoints.json"
    clinical_path.write_text(json.dumps(endpoints))
    clinical_before = clinical_path.read_bytes()
    native = {"status": "predicted", "clearance_l_h": None, "vss_l": None,
              "native_audits_passed": True}
    monkeypatch.setattr(dc, "_run", lambda *a, **kw: deepcopy(native))
    calls = []
    accepted_moment = {"status": "ok", "clearance_l_h": 4.2, "vss_l": None,
                       "vss_status": "not_estimable"}
    def evaluate_moment(actual_curve, actual_request, *, require_vss):
        calls.append(require_vss)
        assert actual_curve == curve
        assert actual_request == request
        return deepcopy(accepted_moment)
    monkeypatch.setattr(moments, "model_integral_pk", evaluate_moment)
    result = dc._run_development({}, request, output, root, audit=True, require_vss=False)
    assert calls == [False]
    assert result["status"] == "predicted"
    assert result["clearance_l_h"] == 4.2
    assert result["vss_l"] is None
    assert result["native_audits_passed"] is True
    assert "converged model AUC" in result["pk_anchor_estimation_basis"]
    assert clinical_path.read_bytes() == clinical_before
    assert json.loads((output / "clearance_only_integral.json").read_text()) == accepted_moment


def test_cl_only_integral_fallback_rejects_failed_convergence_gate(development_case, monkeypatch):
    root, request, _, _, _ = development_case
    output = root / "invalid-cl-only"
    output.mkdir()
    (output / "prediction_locked.json").write_text(json.dumps({"synthetic_curve": True}))
    native = {"status": "predicted", "clearance_l_h": None, "vss_l": None}
    monkeypatch.setattr(dc, "_run", lambda *a, **kw: deepcopy(native))
    rejected = {"status": "not_estimable", "clearance_l_h": None, "vss_l": None,
                "reason": "Remaining AUC exceeds declared tolerance"}
    calls = []
    def failed_gate(*args, require_vss):
        calls.append(require_vss)
        return deepcopy(rejected)
    monkeypatch.setattr(moments, "model_integral_pk", failed_gate)
    result = dc._run_development({}, request, output, root, audit=False, require_vss=False)
    assert calls == [False]
    assert result == native
    assert json.loads((output / "clearance_only_integral.json").read_text()) == rejected


@pytest.mark.parametrize("status,clearance,require_vss", [
    ("predicted", None, True),
    ("predicted", 4.2, False),
    ("parameter_readback_failed", None, False),
    ("administration_readback_failed", None, False),
    ("native_failed", None, False),
])
def test_cl_only_fallback_cannot_bypass_required_vss_or_native_audits(
        development_case, monkeypatch, status, clearance, require_vss):
    root, request, _, _, _ = development_case
    output = root / "no-rescue"
    output.mkdir()
    native = {"status": status, "clearance_l_h": clearance, "vss_l": None}
    monkeypatch.setattr(dc, "_run", lambda *a, **kw: deepcopy(native))
    monkeypatch.setattr(moments, "model_integral_pk", lambda *a, **kw:
                        pytest.fail("CL-only rescue bypassed its eligibility guard"))
    result = dc._run_development({}, request, output, root, audit=True, require_vss=require_vss)
    assert result == native
    assert not (output / "clearance_only_integral.json").exists()
