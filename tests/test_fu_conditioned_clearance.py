"""Training isolation and bounded evidence updates, without real model fitting."""
from copy import deepcopy
import math

import joblib
import numpy as np
import pytest

from pbpk_kb.blueprint_learning import fu_conditioned_clearance as prior


def rows(n=36):
    return [{"index":i,"status":"train","scaffold_group":"group"+str(i//2),
             "identity_smiles":"S"+str(i)} for i in range(n)]


def test_inner_and_outer_scaffold_groups_are_disjoint():
    data = rows()
    for outer in prior.scaffold_folds(data,range(len(data))):
        held = set(outer["validation_indices"])
        for inner in prior.scaffold_folds(data,outer["train_indices"]):
            assert not held & set(inner["train_indices"]+inner["validation_indices"])
            ga = {data[i]["scaffold_group"] for i in inner["train_indices"]}
            gb = {data[i]["scaffold_group"] for i in inner["validation_indices"]}
            assert not ga & gb


def test_ineligible_or_duplicate_training_rows_are_rejected():
    data = rows()
    data[0]["status"] = "excluded_target_identity"
    with pytest.raises(ValueError,match="eligible"):
        prior.scaffold_folds(data,range(len(data)))
    with pytest.raises(ValueError,match="unique"):
        prior.scaffold_folds(rows(),[0,0,2])


def test_view_weights_sum_to_one_per_drug_and_missing_cl_is_not_imputed():
    vx, y, w, records = prior.build_views(np.arange(4).reshape(-1,1),[0,1,2,3],
        {0:1.,1:2.,2:3.},{0:-1.,2:-2.,3:-3.},{0:-.2,1:-.3,2:-.4,3:-.5})
    assert len(y)==5
    for i in range(3):
        assert sum(w[j] for j,r in enumerate(records) if r["row_index"]==i)==1.
    assert all(r["row_index"]!=3 for r in records)
    assert records[0]["log10_fu_input"]==-.2 and records[1]["log10_fu_input"]==-1.
    np.testing.assert_allclose(vx[:,-1],[0,1,0,0,1])
    np.testing.assert_allclose(y,[1,1,2,3,3])


def test_offset_target_uses_actual_view_fu_not_unobserved_true_fu():
    _, y, _, _ = prior.build_views(np.zeros((2,1)),[0,1],{0:1.,1:2.},
        {0:-2.},{0:-.3,1:-.5},offset=True)
    np.testing.assert_allclose(y,[1.3,3.,2.5])


def test_final_fu_consistency_tolerates_gpu_roundoff_but_rejects_changed_baseline():
    source = {"parameters":{"test":{"fraction_unbound":.1}}}
    passed = prior.verify_final_fu(source,["test"],[-1.+1e-6])
    assert passed["test"]["log10_difference"]==pytest.approx(1e-6)
    with pytest.raises(ValueError,match="baseline differs"):
        prior.verify_final_fu(source,["test"],[-1.+1e-4])
    with pytest.raises(ValueError,match="match all"):
        prior.verify_final_fu(source,["test"],[float("nan")])


def test_fit_counts_distinguish_unique_tasks_from_retries():
    result = prior.fit_counts({"fits":{"a":{},"b":{}},"fit_attempts":{"a":1,"b":3,"c":1}})
    assert result == {"actual_fit_attempts":5,"unique_fit_tasks_started":3,
        "unique_fit_tasks_completed":2,"retried_fit_attempts":2}


def test_delta_is_bounded_then_shrunk_and_identical_input_is_exact_identity():
    np.testing.assert_allclose(prior.bounded_delta([10,-10,1],[0,0,1],.5),
        [prior.DELTA_LIMIT/2,-prior.DELTA_LIMIT/2,0.])
    with pytest.raises(ValueError,match="beta"):
        prior.bounded_delta([1],[0],.3)


def test_conditional_both_queries_use_same_measured_flag_and_offset_adds_fu():
    class Model:
        def predict(self,x):
            assert np.all(x[:,-1]==1)
            return np.zeros(len(x))+.7
    np.testing.assert_allclose(prior._predict_conditional(Model(),np.zeros((2,3)),[-1.,-.2],"offset"),[-.3,.5])


def test_selection_improves_overall_without_sacrificing_either_tail_or_p90():
    y = np.array([-2.,-.5,.5,1.5,2.])
    direct = y+np.array([.2,.2,-.1,-.2,-.2])
    delta = y-direct
    choice = prior.select_candidate(y,direct,{"conditioned":delta,"offset":np.zeros(len(y))})
    assert choice["kind"]=="conditioned" and choice["beta"]==1.
    assert choice["scores"]["conditioned:1"]["gmfe"]==pytest.approx(1.)


def test_selection_rejects_overall_gain_that_hurts_low_tail():
    y = np.array([-1.,.3,.4,1.5,1.6])
    direct = y+.2
    delta = np.array([.01,-.2,-.2,-.2,-.2])
    choice = prior.select_candidate(y,direct,{"conditioned":delta,"offset":np.zeros(len(y))})
    assert choice["kind"]=="none" and choice["beta"]==0.


class ToyConditional:
    def predict(self,x):
        assert np.all(x[:,-1]==1)
        return x[:,-2]*3.


def artifact(root,beta=.5):
    model = root/"model.joblib"
    selection = {"kind":"conditioned" if beta else "none","beta":beta}
    joblib.dump({"input_fingerprint":"fixed","selection":selection,
        "target_features":{"test":np.array([1.,2.])},"conditional_model":ToyConditional()},model)
    value = {"version":prior.VERSION,"input_fingerprint":"fixed","source_refs":{},"code_pins":{},
        "parameters":{"test":{"fraction_unbound":.1}},"selection":selection,
        "source_parameter_prior":{"path":"source.json","sha256":"fixture"},
        "model_receipt":{"path":"model.joblib","sha256":prior.shared._sha(model)}}
    value["result_sha256"] = prior.shared._digest(value)
    return value


def test_inference_only_uses_qualified_fu_keeps_source_fu_and_exposes_three_deltas(tmp_path):
    value = artifact(tmp_path)
    out = prior.predict_update(tmp_path,value,"test",.5,.1,evidence_status="derived")
    assert out["factor"]==pytest.approx(math.sqrt(2.))
    assert out["raw_delta_log10"]>out["bounded_delta_log10"]
    assert out["bounded_delta_log10"]==prior.DELTA_LIMIT
    assert out["applied_delta_log10"]==prior.DELTA_LIMIT*.5
    assert out["uses_clinical_observations"] is False
    with pytest.raises(ValueError,match="measured/derived"):
        prior.predict_update(tmp_path,value,"test",.5,.1,evidence_status="predicted")
    with pytest.raises(ValueError,match="frozen"):
        prior.predict_update(tmp_path,value,"test",.5,.2)


def test_same_fu_missing_fu_and_beta_zero_return_exact_identity_without_model_load(tmp_path,monkeypatch):
    value = artifact(tmp_path)
    monkeypatch.setattr(prior.joblib,"load",lambda *_:pytest.fail("No model may be loaded for no-op update"))
    assert prior.predict_update(tmp_path,value,"test",.1,.1)["factor"]==1.
    assert prior.predict_update(tmp_path,value,"test",None,.1)["factor"]==1.
    zero = artifact(tmp_path,0.)
    assert prior.predict_update(tmp_path,zero,"test",.7,.1)["factor"]==1.


@pytest.mark.parametrize("invalid",[0.,-1.,1.1,float("nan"),True])
def test_invalid_observed_fu_is_rejected_even_for_zero_selected_beta(tmp_path,invalid):
    with pytest.raises(ValueError,match="fraction"):
        prior.predict_update(tmp_path,artifact(tmp_path,0.),"test",invalid,.1)


class ToyFit:
    def __init__(self,kind,records):
        self.kind,self.records = kind,records
    def fit(self,x,y,**kwargs):
        self.records.append((self.kind,x.copy(),np.asarray(y).copy(),kwargs))
        self.mean = float(np.mean(y))
        return self
    def predict(self,x):
        if self.kind=="fu":
            return np.full(len(x),min(self.mean,0.))
        if self.kind in prior.KINDS:
            return .1*x[:,-2]+self.mean
        return np.full(len(x),self.mean)


def test_end_to_end_fake_training_keeps_base_parameters_and_resumes_without_cv_model_files(tmp_path,monkeypatch):
    data = rows()
    ids = list(range(len(data)))
    labels = {"total_clearance":{i:(i%7-3)*.6 for i in ids},
              "fraction_unbound":{i:-.1*(i%9+1) for i in ids if i%4}}
    source = {"parameters":{"test":{"fraction_unbound":10.**np.mean(list(labels["fraction_unbound"].values())),
        "total_clearance":3.,"vss":2.,"mrt":4.,"terminal_half_life":6.}},
        "units":{},"targets":{"test":{"identity_smiles":"S99"}},"per_compound_audit":{"test":{}},
        "audit":{"human_parameter_training_indices":{k:sorted(v) for k,v in labels.items()}}}
    original = deepcopy(source)
    cfg = {"source_refs":{},"code_pins":{},"eligible_indices":ids,"outer_folds":prior.scaffold_folds(data,ids),
        "device":"cpu","source_parameter_prior":{"path":"source","sha256":"fake"}}
    state = {"config":cfg,"input_fingerprint":"fake","fits":{},"fold_results":{}}
    monkeypatch.setattr(prior,"_prepare",lambda *_:(state,source,data,labels,{"rows":data}))
    monkeypatch.setattr(prior.rich,"rich_features",lambda smiles:(np.array([[float(s[1:]),1.] for s in smiles]),
        {"descriptor_names":["id","one"]}))
    records = []
    class InterruptedFit(ToyFit):
        def fit(self,x,y,**kwargs):
            checkpoint = __import__("json").loads((tmp_path/"output/training_state.json").read_text())
            assert checkpoint["actual_fit_attempts"]==1
            raise RuntimeError("Simulated first fitting interruption")
    with pytest.raises(RuntimeError,match="Simulated"):
        prior.train(tmp_path,output_dir="output",device="cpu",estimator_factory=lambda kind,*_:InterruptedFit(kind,records))
    assert prior.fit_counts(state)["actual_fit_attempts"]==1
    result = prior.train(tmp_path,output_dir="output",device="cpu",estimator_factory=lambda kind,*_:ToyFit(kind,records))
    assert result["parameters"]==original["parameters"] and source==original
    assert result["audit"]["actual_fit_attempts"]==result["audit"]["unique_fit_tasks_completed"]+1
    assert result["audit"]["retried_fit_attempts"]==1
    assert result["audit"]["final_fu_baseline_consistency_passed"] is True
    assert len(result["audit"]["fold_results"])==3
    assert sum(len(f["paired_oof"]) for f in result["audit"]["fold_results"].values())==len(labels["fraction_unbound"])
    assert not list((tmp_path/"output").glob("*fold*.joblib"))
    assert [p.name for p in (tmp_path/"output").glob("*.joblib")]==["model.joblib"]
    for kind,x,y,kwargs in records:
        if kind in prior.KINDS:
            weight = kwargs["model__sample_weight"]
            for i in set(x[:,0]):
                assert weight[x[:,0]==i].sum()==1.
    count = len(records)
    again = prior.train(tmp_path,output_dir="output",device="cpu",estimator_factory=lambda kind,*_:ToyFit(kind,records))
    assert len(records)==count and again["result_sha256"]==result["result_sha256"]
