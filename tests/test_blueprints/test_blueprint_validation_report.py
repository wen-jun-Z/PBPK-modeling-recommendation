"""Scientific-statistics checks; synthetic fixtures are not experiment results."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import report_blueprint_validation as report


def test_condition_scheduling_order_does_not_change_matrix_identity():
    report.validate_condition_matrix({"methods":list(reversed(report.METHODS)),"sampling":["five","full"]})


@pytest.mark.parametrize("sampling",[["full","full"],["full","five","five"],["full"],["full","unknown"]])
def test_duplicate_or_changed_sampling_is_rejected(sampling):
    with pytest.raises(ValueError):
        report.validate_condition_matrix({"methods":list(report.METHODS),"sampling":sampling})


def test_duplicate_methods_are_rejected():
    with pytest.raises(ValueError):
        report.validate_condition_matrix({"methods":["ours"]*len(report.METHODS),"sampling":list(report.SAMPLING)})


def test_shared_study_drugs_are_resampled_together():
    result = report.clustered_geomean({"a":1., "b":9., "c":4.}, {"a":"shared", "b":"shared", "c":"other"})
    assert result["estimate"] == pytest.approx(36**(1/3))
    # The shared cluster has geometric mean 3; no bootstrap draw may isolate 1 or 9.
    assert result["ci95"] == pytest.approx([3.,4.])
    assert result["n_clusters"] == 2


def test_paired_ratio_keeps_failure_denominator():
    result = report.paired_ratio({"a":2.,"b":6.,"c":None},{"a":4.,"b":3.,"c":5.},
                                {"a":"s1","b":"s2","c":"s3"})
    assert result["estimate"] == pytest.approx(1.)
    assert result["attempted_pairs"] == 3 and result["complete_pairs"] == 2
    assert result["incomplete_case_ids"] == ["c"]
    assert result["per_case_ratio"] == {"a":.5,"b":2.}


def test_bootstrap_is_deterministic_and_dictionary_order_independent():
    a = report.clustered_geomean({"a":1.,"b":3.,"c":7.},{"a":"a","b":"b","c":"c"})
    b = report.clustered_geomean({"c":7.,"a":1.,"b":3.},{"c":"c","a":"a","b":"b"})
    assert a == b


def test_one_study_does_not_fabricate_confidence_interval():
    result = report.clustered_geomean({"a":2.,"b":8.},{"a":"same","b":"same"})
    assert result["estimate"] == pytest.approx(4.) and result["ci95"] is None
    assert report.clustered_geomean({}, {})["estimate"] is None


@pytest.mark.parametrize("value",[0.,-1.,float("nan"),float("inf")])
def test_invalid_values_rejected_by_bootstrap(value):
    with pytest.raises(ValueError):
        report.clustered_geomean({"a":value},{"a":"s1"})


def test_fold_error_and_tmax_have_distinct_dimensions():
    assert report.endpoint_error(2.,8.,"clearance","L/h") == (4.,"fold_error")
    assert report.endpoint_error(8.,2.,"clearance","L/h") == (4.,"fold_error")
    assert report.endpoint_error(15.,30.,"tmax","min") == (.25,"absolute_error_h")
    assert report.endpoint_error(0.,.5,"tmax","h") == (.5,"absolute_error_h")
    assert report.endpoint_error(3.,None,"vss","L")[0] is None


def test_lod_proxy_must_be_declared_and_not_called_lloq():
    policy = {"lloq_ng_ml":None,"prediction_cutoff_proxy":"reported_lod","lod_ng_ml":150.}
    with pytest.raises(ValueError): report.cutoff_from_policy(policy,set())
    assert report.cutoff_from_policy(policy,{"lod_used_as_prediction_cutoff"}) == (150.,"reported_lod_proxy")


def test_unlocked_run_cannot_decode_clinical_outcomes(tmp_path, monkeypatch):
    protocol = {"version":"blueprint-validation-1", "cases":[], "outcome":"must_not_read.json"}
    (tmp_path/"protocol.json").write_text(json.dumps(protocol))
    (tmp_path/"summary.json").write_text(json.dumps({"status":"running"}))
    def forbidden(*args, **kwargs):
        raise AssertionError("An unlocked run must not decode outcomes")
    monkeypatch.setattr(report,"_case_reference",forbidden)
    with pytest.raises(ValueError,match="locked"):
        report.report(tmp_path,project_root=tmp_path)
    assert not (tmp_path/"results.json").exists()


def test_tampered_ref_is_rejected(tmp_path):
    path = tmp_path/"prediction.json"; path.write_text('{"value":2}')
    store = report.Store(tmp_path)
    with pytest.raises(ValueError,match="lock mismatch"):
        store.ref({"path":"prediction.json","sha256":"0"*64})


def test_target_dose_mismatch_rejected_before_nca():
    class MemoryStore:
        def ref(self, name):
            if name == "native":
                return {"time_min":[0,60],"concentration":[0,1],"unit":"umol/l","time_unit":"min"}
            return {"human":{"weight_kg":70},"dosing":{"dose_mg":20,"infusion_minutes":60,"start_h":0}}
    with pytest.raises(ValueError,match="dose/weight/infusion"):
        report._score_prediction({"nca_policy":{"weight_kg":70,"dose_ng":10e6,"infusion_h":1}},
            {},{}, {}, {"prediction":"native","request":"request"},MemoryStore())


def test_failed_integrity_keeps_all_attempts_without_publishing_comparison(tmp_path, monkeypatch):
    # pytest is invoked with a basetemp inside project/runtime, preserving project scope.
    cases = [{"case_id":f"synthetic_{i}","drug":f"Synthetic fixture {i}","study_id":f"s{i}",
              "cluster_id":f"s{i}","transfer_group":"unchanged","effective_evidence_ablation":False} for i in range(10)]
    units = {f'{c["case_id"]}__{s}__{m}':{"case_id":c["case_id"],"sampling":s,"method":m,"status":"failed",
        "request":{"path":"runtime/absent_synthetic_request.json","sha256":"0"*64},"failure":"synthetic test fixture"}
        for c in cases for s in report.SAMPLING for m in report.METHODS}
    (tmp_path/"protocol.json").write_text(json.dumps({"version":"blueprint-validation-1","cases":cases,
        "methods":list(report.METHODS),"sampling":list(report.SAMPLING),"bootstrap":{"seed":report.SEED,"resamples":report.RESAMPLES},
        "source_pins":{"runtime/absent_synthetic_request.json":"0"*64}}))
    (tmp_path/"summary.json").write_text(json.dumps({"status":"predictions_locked","all_attempts_locked_before_scoring":True,"units":units}))
    def forbidden(*args, **kwargs): raise AssertionError("Corrupt-source batch cannot read clinical outcomes")
    monkeypatch.setattr(report,"_case_reference",forbidden)
    result = report.report(tmp_path, project_root=ROOT)
    assert result["status"] == "integrity_failed"
    assert result["automation"]["attempted"] == 100 and len(result["units"]) == 100
    assert result["data_profile"]["csv_rows"] == 1100
    assert (tmp_path/"results.csv").exists() and not (tmp_path/"comparison.pdf").exists()
