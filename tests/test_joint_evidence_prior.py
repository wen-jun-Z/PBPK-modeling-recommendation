"""Fixed joint updates must preserve clinical evidence and target isolation."""
from copy import deepcopy

import numpy as np
import pytest

from pbpk_kb.blueprint_learning import joint_evidence_prior as joint


def inputs(pattern="both"):
    source={"path":"data/independent.json","sha256":"a"*64}
    context={"species":"human","matrix":"plasma","route":"intravenous","study_role":"development"}
    def record(p,value,unit,basis="predicted"):
        return {"parameter":p,"compound_id":"drug","value":value,"unit":unit,"basis":basis,
            "source":deepcopy(source),"context":deepcopy(context)}
    cl=record("total_clearance",60.,"L/h","derived")
    fu=record("fraction_unbound",.2,"fraction","measured")
    if pattern in {"none","fu_only"}: cl.update(value=12.,basis="predicted")
    if pattern in {"none","CL_only"}:fu.update(value=.1,basis="predicted")
    request={"compound_id":"drug","human":{"weight_kg":100.},"evidence":[cl,fu],
        "forbidden_sources":[{"path":"target.json","sha256":"b"*64}]}
    sides={"vss":[record("vss",2.,"L/kg")],"half":[record("terminal_half_life",4.,"h")]}
    coefficients={name:{p:{"inputs":list(evidence),"values":[.2]*len(evidence),"n":100}
        for p in joint.OUTPUTS if p not in evidence}for name,evidence in joint.PATTERNS.items()}
    artifact={"version":joint.VERSION,"shrinkage":.5,"coefficients":coefficients,
        "current_target_priors":{"drug":{"total_clearance":2.,"fraction_unbound":.1,"vss":2.,"terminal_half_life":4.}},
        "original_target_priors":{"drug":{"total_clearance":1.,"fraction_unbound":.1,"vss":2.,"terminal_half_life":4.}}}
    return request,sides,artifact,{"path":"prior/predictions.json","sha256":"c"*64}


def test_ridge_matches_independent_matrix_solution_and_is_zero_intercept():
    x=np.array([[1.,0.],[-1.,2.],[.5,-.5],[2.,1.]])
    y=np.array([1.,-.5,2.,3.])
    expected=np.linalg.solve(x.T@x+.04*np.eye(2),x.T@y)
    np.testing.assert_allclose(joint.residual_coefficients(x,y),expected)
    np.testing.assert_array_equal(joint.residual_coefficients(np.zeros((3,2)),[1.,2.,3.]),[0.,0.])
    with pytest.raises(ValueError):joint.residual_coefficients([[1.,float("nan")]],[2.])


def test_held_labels_do_not_change_their_predictions():
    x=np.array([[1.,0.],[0.,1.]]*3);y=np.array([1.,2.]*3);center=np.zeros(6);folds=[0,0,1,1,2,2]
    first=joint.crossfit_relation(x,y,center,folds,"vss")
    y[:2]=[-100.,100.]
    second=joint.crossfit_relation(x,y,center,folds,"vss")
    assert first["conditioned_oof_log10"][:2]==second["conditioned_oof_log10"][:2]
    assert first["coefficients"]!=second["coefficients"]


def test_no_evidence_is_exact_identity():
    request,sides,artifact,receipt=inputs("none")
    r,s,a=joint.condition_inputs(request,sides,artifact,receipt)
    assert (r,s)==(request,sides) and r is not request and s is not sides
    assert a["pattern"]=="none" and not a["applied"]


def test_cl_only_preserves_total_and_renal_and_updates_missing_inputs():
    request,sides,artifact,receipt=inputs("CL_only")
    renal=deepcopy(request["evidence"][0]);renal.update(parameter="renal_clearance",value=3.,basis="measured")
    request["evidence"].append(renal);before=deepcopy((request,sides))
    r,s,a=joint.condition_inputs(request,sides,artifact,receipt)
    assert (request,sides)==before
    assert r["evidence"][0]==request["evidence"][0] and r["evidence"][2]==renal
    assert a["innovations_log10"]["total_clearance"]==pytest.approx(np.log10(5.))
    ratio=5.**.1
    assert r["evidence"][1]["value"]==pytest.approx(.1*ratio)
    assert s["vss"][0]["value"]==pytest.approx(2.*ratio)
    assert s["half"][0]["basis"]=="predicted"
    assert a["request_modified"] and a["sidecars_modified"]


def test_fu_only_updates_current_complementary_cl_not_original_cl():
    request,sides,artifact,receipt=inputs("fu_only")
    request["evidence"][1]["context"].pop("route")
    r,s,a=joint.condition_inputs(request,sides,artifact,receipt)
    assert r["evidence"][1]==request["evidence"][1]
    assert r["evidence"][0]["value"]==pytest.approx(12.*2.**.1)
    assert a["pattern"]=="fu_only"
    assert a["changed_parameters"]==["total_clearance","vss","terminal_half_life"]


def test_both_preserves_all_request_evidence():
    request,sides,artifact,receipt=inputs("both")
    r,s,a=joint.condition_inputs(request,sides,artifact,receipt)
    assert r==request and not a["request_modified"]
    assert s["vss"][0]["value"]==pytest.approx(2.*10.**.1)
    assert a["pattern"]=="both" and a["sidecars_modified"]


def test_percent_fu_and_liter_vss_are_converted_correctly():
    request,sides,artifact,receipt=inputs("fu_only")
    request["evidence"][1].update(value=20.,unit="percent")
    sides["vss"][0].update(value=200.,unit="L")
    r,s,a=joint.condition_inputs(request,sides,artifact,receipt)
    assert a["innovations_log10"]["fraction_unbound"]==pytest.approx(np.log10(2.))
    assert s["vss"][0]["value"]==pytest.approx(200.*2.**.1)
    assert r["evidence"][1]["value"]==20.


def test_predicted_fu_is_physically_bounded():
    request,sides,artifact,receipt=inputs("CL_only")
    artifact["coefficients"]["CL_only"]["fraction_unbound"]["values"]=[10.]
    r,_,_=joint.condition_inputs(request,sides,artifact,receipt)
    assert r["evidence"][1]["value"]==1.


def test_measured_sidecar_is_not_replaced():
    request,sides,artifact,receipt=inputs("both")
    measured=deepcopy(sides["vss"][0]);measured.update(basis="derived",value=3.)
    sides["vss"].append(measured)
    _,new,a=joint.condition_inputs(request,sides,artifact,receipt)
    assert new["vss"]==sides["vss"] and a["changed_parameters"]==["terminal_half_life"]


@pytest.mark.parametrize("bad",["source","context","independence"])
def test_target_evidence_cannot_activate_updates(bad):
    request,sides,artifact,receipt=inputs("both")
    rec=request["evidence"][1]
    if bad=="source":rec["source"]["sha256"]="b"*64
    elif bad=="context":rec["context"]["study_role"]="target"
    else:rec["context"]["independent_of_target"]=False
    with pytest.raises(ValueError):joint.condition_inputs(request,sides,artifact,receipt)


def test_predicted_total_and_independent_renal_rejected():
    request,sides,artifact,receipt=inputs("fu_only")
    renal=deepcopy(request["evidence"][1]);renal.update(parameter="renal_clearance",value=1.,unit="L/h")
    request["evidence"].append(renal)
    with pytest.raises(ValueError,match="renal split"):joint.condition_inputs(request,sides,artifact,receipt)


def test_old_cl_center_and_double_conditioning_rejected():
    request,sides,artifact,receipt=inputs("fu_only")
    request["evidence"][0]["value"]=6.
    with pytest.raises(ValueError,match="current frozen center"):joint.condition_inputs(request,sides,artifact,receipt)
    request,sides,artifact,receipt=inputs("both")
    r,s,_=joint.condition_inputs(request,sides,artifact,receipt)
    with pytest.raises(ValueError,match="current frozen center"):joint.condition_inputs(r,s,artifact,receipt)


@pytest.mark.parametrize("value",[0.,-1.,1.01,float("nan")])
def test_invalid_observed_fu_rejected(value):
    request,sides,artifact,receipt=inputs("fu_only");request["evidence"][1]["value"]=value
    with pytest.raises(ValueError):joint.condition_inputs(request,sides,artifact,receipt)
