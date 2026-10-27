"""Independent evidence cannot mutate protected inputs or leak scoring labels."""
from copy import deepcopy
import math

import numpy as np
import pytest

from pbpk_kb.blueprint_learning import evidence_conditioned_prior as ec


def fixture():
    source = {"path":"data/independent.json", "sha256":"a"*64, "location":"independent clinical CL"}
    context = {"species":"human", "matrix":"plasma", "route":"intravenous", "study_role":"development"}
    cl = {"parameter":"total_clearance", "compound_id":"drug", "basis":"derived",
        "value":60., "unit":"l/h", "source":source, "context":context}
    renal = {"parameter":"renal_clearance", "compound_id":"drug", "basis":"measured",
        "value":3., "unit":"l/h", "source":source, "context":context}
    request = {"compound_id":"drug", "human":{"weight_kg":100.}, "evidence":[cl,renal],
        "forbidden_sources":[{"path":"target.json", "sha256":"b"*64}]}
    sides = {key:[{"compound_id":"drug", "parameter":parameter, "value":value, "unit":unit,
        "basis":"predicted", "source":source, "context":context,
        "prediction_method":{"name":"original", "version":"original", "training_overlap":False,
            "applicability":{"target_structure_and_scaffold_excluded":True,"human_IV_parameter_model":True}}}]
        for key,parameter,value,unit in [("vss","vss",2.,"l/kg"),("half","terminal_half_life",4.,"h")]}
    artifact = {"version":ec.VERSION, "shrinkage":.5,
        "coefficients":{"vss":.2,"terminal_half_life":-.4},
        "original_target_priors":{"drug":{"total_clearance":1.,"vss":2.,"terminal_half_life":4.}}}
    receipt = {"path":"prior/predictions.json", "sha256":"c"*64}
    return request,sides,artifact,receipt


def test_ridge_relation_uses_signed_residuals_and_fixed_shrink():
    assert ec.residual_slope([1.,-1.],[2.,-2.]) == pytest.approx(4/2.02)
    assert ec.residual_slope([0.,0.],[2.,-2.]) == 0.
    with pytest.raises(ValueError): ec.residual_slope([1.],[float("nan")])


def test_held_endpoint_labels_cannot_change_their_crossfit_prediction():
    folds = np.array([0,0,1,1,2,2]);yc=np.array([1.,2.,1.,2.,1.,2.]);pc=np.zeros(6)
    pz=np.zeros(6);yz=np.array([2.,4.,2.,4.,2.,4.])
    first=ec.crossfit_correction(yc,pc,yz,pz,folds)
    yz[:2]=[-100.,100.]
    second=ec.crossfit_correction(yc,pc,yz,pz,folds)
    assert first['conditioned_oof_log10'][:2] == second['conditioned_oof_log10'][:2]
    assert first['folds'][0]['slope'] == second['folds'][0]['slope']
    assert first['slope'] != second['slope']


@pytest.mark.parametrize("value,unit", [(10.,"ml/min/kg"),(1000.,"ml/min"),(60.,"l/h"),(.6,"l/h/kg"),(1.,"l/min"),(.01,"l/min/kg")])
def test_cl_units_convert_to_same_weight_normalized_value(value,unit):
    assert ec.cl_ml_min_kg({"value":value,"unit":unit},100.) == pytest.approx(10.)


def test_only_predicted_sidecars_change_and_original_center_is_used():
    request,sides,artifact,receipt=fixture();before=deepcopy((request,sides))
    new,audit=ec.condition_sidecars(request,sides,artifact,receipt)
    assert (request,sides)==before
    assert audit['cl_innovation_log10']==pytest.approx(1.)
    assert new['vss'][0]['value']==pytest.approx(2*10**.1)
    assert new['half'][0]['value']==pytest.approx(4*10**-.2)
    assert audit['changed_parameters']==['vss','terminal_half_life']
    assert new['vss'][0]['basis']=='predicted' and new['vss'][0]['source']['sha256']==receipt['sha256']


def test_predicted_cl_does_not_activate_and_measured_sidecar_wins():
    request,sides,artifact,receipt=fixture();request['evidence'][0]['basis']='predicted'
    new,audit=ec.condition_sidecars(request,sides,artifact,receipt)
    assert new==sides and not audit['applied']
    request['evidence'][0]['basis']='derived';measured=deepcopy(sides['vss'][0]);measured.update(basis='measured',value=8.)
    sides['vss'].append(measured)
    new,audit=ec.condition_sidecars(request,sides,artifact,receipt)
    assert new['vss']==sides['vss'] and audit['changed_parameters']==['terminal_half_life']


def test_target_sources_and_contexts_cannot_activate_conditioning():
    request,sides,artifact,receipt=fixture();request['evidence'][0]['source']['sha256']='b'*64
    with pytest.raises(ValueError,match='Target observation'):ec.condition_sidecars(request,sides,artifact,receipt)
    request,sides,artifact,receipt=fixture();request['evidence'][0]['context']['study_role']='target'
    with pytest.raises(ValueError,match='context'):ec.condition_sidecars(request,sides,artifact,receipt)
    request,sides,artifact,receipt=fixture();request['evidence'][0]['context']['independent_of_target']=False
    with pytest.raises(ValueError,match='context'):ec.condition_sidecars(request,sides,artifact,receipt)


def test_prior_changed_from_frozen_center_is_not_silently_conditioned_twice():
    request,sides,artifact,receipt=fixture();sides['half'][0]['value']=9.
    with pytest.raises(ValueError,match='frozen original center'):ec.condition_sidecars(request,sides,artifact,receipt)


def test_vss_liter_sidecar_and_zero_innovation():
    request,sides,artifact,receipt=fixture();sides['vss'][0].update(unit='L',value=200.)
    new,_=ec.condition_sidecars(request,sides,artifact,receipt)
    assert new['vss'][0]['value']==pytest.approx(200*10**.1)
    request['evidence'][0].update(value=1.,unit='ml/min/kg')
    new,audit=ec.condition_sidecars(request,sides,artifact,receipt)
    assert new==sides and not audit['applied']
