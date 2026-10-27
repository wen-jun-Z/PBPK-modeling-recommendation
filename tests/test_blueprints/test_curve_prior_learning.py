"""Small synthetic contract tests; no native PBPK or real clinical training."""
from copy import deepcopy
import math

import joblib
import numpy as np
import pytest

from pbpk_kb.blueprint_learning import curve_prior as cp
from test_pk_anchored_blueprint import evidence_case


@pytest.fixture
def data(evidence_case):
    _, template, _ = evidence_case
    rows = []
    for drug in range(9):
        compound = "synthetic:" + str(drug)
        req = deepcopy(template)
        req.update(compound_id=compound, compound_name="Synthetic " + str(drug))
        req["human"]["age_years"] += drug
        req["dosing"].update(dose_mg=10.+drug, start_h=0.)
        for evidence in req["evidence"]:
            evidence["compound_id"] = compound
            if evidence["parameter"] == "logp":
                evidence["value"] += .2*drug
        logp = next(e["value"] for e in req["evidence"] if e["parameter"] == "logp")
        for sampling in ("five", "full"):
            times = [5., 15., 60., 180., 480.] if sampling == "five" else [5., 10., 15., 30., 60., 180., 480.]
            doc = {"compound_id": compound, "study_id": "synthetic-development-" + str(drug),
                "role": "calibration_data_reconstruction", "matrix": "plasma", "time_unit": "min", "unit": "umol/l",
                "molecular_weight_g_mol": 194.19, "human": deepcopy(req["human"]), "dosing": deepcopy(req["dosing"]),
                "observations": [{"time_min": t, "concentration_umol_per_l": (1.+drug/10)*math.exp(-t/(200.+drug*10))}
                                 for t in times]}
            for formula in cp.PARTITIONS:
                rows.append({"compound_id": compound, "document": deepcopy(doc), "request": deepcopy(req),
                    "sampling": sampling, "partition_method": formula,
                    "decision": {"lipophilicity": logp+(drug-4)*.12,
                                 "clearance_scale": math.exp((drug-4)*.05)},
                    "source_refs": {"development": {"path": "synthetic/development_"+str(drug)+".json", "sha256": "a"*64},
                                    "teacher": {"path": "synthetic/teacher_"+str(drug)+".json", "sha256": "b"*64}}})
    return rows


def test_feature_scale_units_and_local_dose_normalization(data):
    row=data[0];doc,req=row["document"],row["request"]
    values=cp.curve_features(doc,req)
    assert values.shape==(len(cp.FEATURE_NAMES),)==(30,)
    expected=doc["observations"][0]["concentration_umol_per_l"]*194.19/1000/(10./70.)
    assert values[cp.FEATURE_NAMES.index("log_dose_normalized_curve_q000")]==pytest.approx(math.log(expected))
    scaled_doc,scaled_req=deepcopy(doc),deepcopy(req)
    for r in scaled_doc["observations"]:r["concentration_umol_per_l"]*=2
    scaled_doc["dosing"]["dose_mg"]*=2;scaled_req["dosing"]["dose_mg"]*=2
    modified=cp.curve_features(scaled_doc,scaled_req)
    idx=cp.FEATURE_NAMES.index("log_dose_mg_per_kg")
    assert modified[idx]-values[idx]==pytest.approx(math.log(2))
    np.testing.assert_allclose(np.delete(modified,idx),np.delete(values,idx),atol=1e-12)
    shifted_doc,shifted_req=deepcopy(doc),deepcopy(req)
    shifted_doc["dosing"]["start_h"]=shifted_req["dosing"]["start_h"]=1.
    for r in shifted_doc["observations"]:r["time_min"]+=60
    np.testing.assert_allclose(cp.curve_features(shifted_doc,shifted_req),values)


def test_features_never_use_ids_hidden_full_rows_or_endpoint_labels(data):
    row=data[0];doc,req=deepcopy(row["document"]),deepcopy(row["request"])
    original=cp.curve_features(doc,req)
    doc.update(study_id="different",compound_name="different",raw_observations=[{"huge_hidden_label":1e100}],
               clinical_reference={"clearance":1e100},endpoints={"cmax":1e200})
    req["compound_name"]="irrelevant name";req["target"]={"reference_clearance":1e100}
    np.testing.assert_array_equal(cp.curve_features(doc,req),original)
    assert not any("study_id" in k or "compound_id" in k or "reference" in k for k in cp.FEATURE_NAMES)


@pytest.mark.parametrize("bad",["concentration_unit","time_unit","individual_unit","identity","mw","role","dosing","zero","unordered","few","forbidden"])
def test_invalid_units_conditions_and_target_sources_fail(data,bad):
    row=deepcopy(data[0]);doc,req=row["document"],row["request"]
    if bad=="concentration_unit":doc["unit"]="ng/ml"
    elif bad=="time_unit":doc["time_unit"]="h"
    elif bad=="individual_unit":doc["observations"][0]["unit"]="mg/l"
    elif bad=="identity":doc["compound_id"]="another"
    elif bad=="mw":doc["molecular_weight_g_mol"]*=1000
    elif bad=="role":doc["role"]="heldout_clinical_evaluation"
    elif bad=="dosing":doc["dosing"]["dose_mg"]*=2
    elif bad=="zero":doc["observations"][0]["concentration_umol_per_l"]=0
    elif bad=="unordered":doc["observations"].reverse()
    elif bad=="few":doc["observations"].pop()
    else:
        doc["source"]={"path":"heldout.json","sha256":"c"*64}
        req["forbidden_sources"]=[doc["source"]]
    with pytest.raises(ValueError):cp.curve_features(doc,req)


def test_leave_one_compound_out_excludes_all_formulas_and_sampling_before_decoding(data):
    model=cp.fit_prior(data,"synthetic:0","rf")
    assert model["excluded_rows"]==4
    assert model["training_compound_ids"]==["synthetic:"+str(i) for i in range(1,9)]
    assert len(model["training_rows"])==32
    poisoned=deepcopy(data)
    for row in poisoned:
        if row["compound_id"]=="synthetic:0":
            row["document"]={"must_not_be_decoded":True};row["decision"]={"lipophilicity":float("inf")}
    second=cp.fit_prior(poisoned,"synthetic:0","rf")
    row=data[0]
    assert cp.predict_prior(model,row["document"],row["request"],row["partition_method"])==cp.predict_prior(
        second,row["document"],row["request"],row["partition_method"])
    x=np.asarray([cp._inputs(r["document"],r["request"],r["partition_method"]) for r in data if r["compound_id"]!="synthetic:0"])
    np.testing.assert_allclose(model["scaler"].mean_,x.mean(axis=0))
    assert model["teacher_is_clinical_ground_truth"] is False


def test_minimum_training_compounds_and_teacher_guards(data):
    with pytest.raises(ValueError,match="eight"):
        cp.fit_prior(data[:-4],"synthetic:0","rf")
    bad=deepcopy(data);bad[4]["decision"]["clearance_scale"]=4.1
    with pytest.raises(ValueError,match="bounds"):cp.fit_prior(bad,"synthetic:0","rf")
    bad=deepcopy(data);bad[4]["request"]["forbidden_sources"]=[bad[4]["source_refs"]["teacher"]]
    with pytest.raises(ValueError,match="Held-out"):cp.fit_prior(bad,"synthetic:0","rf")
    bad=deepcopy(data);bad[4]["compound_id"]="fake identity"
    with pytest.raises(ValueError,match="identities"):cp.fit_prior(bad,"synthetic:0","rf")


@pytest.mark.parametrize("algorithm",["rf","mlp"])
def test_serializable_bounded_prediction_and_heldout_checks(data,tmp_path,algorithm):
    model=cp.fit_prior(data,"synthetic:0",algorithm,epochs=3)
    row=data[0];prediction=cp.predict_prior(model,row["document"],row["request"],row["partition_method"])
    path=tmp_path/(algorithm+".joblib");joblib.dump(model,path);reloaded=joblib.load(path)
    assert prediction==cp.predict_prior(reloaded,row["document"],row["request"],row["partition_method"])
    assert .25<=prediction["clearance_scale"]<=4.
    assert abs(prediction["lipophilicity"]+.1)<=2.5
    assert prediction["native_execution_required"] is True
    assert prediction["clinical_target_values_used"] is False
    other=data[4]
    with pytest.raises(ValueError,match="excluded compound"):
        cp.predict_prior(model,other["document"],other["request"],other["partition_method"])
    if algorithm=="mlp":
        info=model["implementation"]
        assert set(info["inner_training_compound_ids"]).isdisjoint(info["inner_validation_compound_ids"])
        assert set(info["inner_training_compound_ids"])|set(info["inner_validation_compound_ids"])==set(model["training_compound_ids"])
        assert 1<=info["selected_epochs"]<=3
        train=[r for r in data if r["compound_id"] in info["inner_training_compound_ids"]]
        xx=np.asarray([cp._inputs(r["document"],r["request"],r["partition_method"]) for r in train])
        np.testing.assert_allclose(info["inner_scaler_mean"],xx.mean(axis=0))
        assert info["final_refit_uses_all_outer_training_compounds"] is True


def test_formula_is_explicit_and_predictions_reject_schema_mismatch(data):
    model=cp.fit_prior(data,"synthetic:0","rf");row=data[0]
    x=cp._inputs(row["document"],row["request"],"RodgersRowland")
    np.testing.assert_array_equal(x[-2:],[0.,1.])
    model["feature_names"]=["wrong"]
    with pytest.raises(ValueError,match="schema"):
        cp.predict_prior(model,row["document"],row["request"],"RodgersRowland")
