"""Expansion preserves a fixed algorithm and proves cohort exclusion afresh."""
from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from pbpk_kb.blueprint_learning import expanded_selected_prior as expanded


def cohort():
    return expanded.target_cohort({"targets":[{"compound_id":"drug","smiles":"c1ccccc1"}]})


def test_only_identities_and_bounded_metadata_accepted():
    document={"targets":[{"compound_id":"drug","smiles":"c1ccccc1","drug":"name","role":"candidate"}],
              "version":"cohort-1","source_refs":{},"rule":"Exclude all candidates before training"}
    assert expanded.target_cohort(document)==cohort()
    document["targets"][0]["observed_clearance"]=9.
    with pytest.raises(ValueError,match="numeric outcome"):expanded.target_cohort(document)
    with pytest.raises(ValueError):expanded.target_cohort({"targets":document["targets"],"clinical_curve":[1,2]})


def test_animal_target_labels_are_never_parsed_and_human_columns_ignored(monkeypatch):
    # Benzene identity and a benzene-scaffold relative are both protected. Their
    # invalid numeric strings must never enter the conversion function.
    payload=b"smiles_r,dog_CL_mL_min_kg,monkey_CL_mL_min_kg,rat_CL_mL_min_kg,human_CL_mL_min_kg\nc1ccccc1,DO_NOT_PARSE,DO_NOT_PARSE,DO_NOT_PARSE,SECRET\nCc1ccccc1,DO_NOT_PARSE,DO_NOT_PARSE,DO_NOT_PARSE,SECRET\nCC,1,10,100,DO_NOT_PARSE\n"
    rows=expanded.read_animal_rows(payload,cohort(),interpret_labels=True)
    assert [r["status"] for r in rows]==["excluded_target_identity","excluded_target_generic_scaffold","train"]
    assert "log10_cl" not in rows[0] and "log10_cl" not in rows[1]
    assert rows[2]["log10_cl"]=={"dog":0.,"monkey":1.,"rat":2.}
    assert all("log10_cl" not in r for r in expanded.read_animal_rows(payload,cohort()))


def test_overlap_gate_checks_animal_as_well_as_human_and_generic_scaffold():
    row={"index":4,"status":"train",**expanded.shared.normalize_structure("Cc1ccccc1")}
    overlaps=expanded.overlap_audit(cohort(),[],[row])
    assert expanded.requires_refit(overlaps)
    assert overlaps["drug"]["animal"]["identity_indices"]==[]
    assert overlaps["drug"]["animal"]["generic_scaffold_indices"]==[4]
    row["status"]="excluded_target_generic_scaffold"
    assert not expanded.requires_refit(expanded.overlap_audit(cohort(),[],[row]))


def test_normalization_salts_and_shared_acyclic_group_are_protected():
    targets=expanded.target_cohort({"targets":[{"compound_id":"d","smiles":"CC.[Na+]"}]})
    row={"index":0,"status":"train",**expanded.shared.normalize_structure("CCC")}
    assert expanded.requires_refit(expanded.overlap_audit(targets,[row],[]))


class Constant:
    def __init__(self,value):self.value=value
    def predict(self,x):return np.full(len(x),self.value)


def test_fixed_mean_fusion_does_not_reproject_or_change_other_parameters(monkeypatch):
    monkeypatch.setattr(expanded.rich,"rich_features",lambda identities:(np.zeros((len(identities),2)),{"descriptor_names":["a"]}))
    monkeypatch.setattr(expanded.kernel,"morgan_binary_features",lambda identities:np.ones((len(identities),4)))
    values={"total_clearance":1.,"fraction_unbound":.2,"vss":.3,"mrt":.8,"terminal_half_life":.4}
    shared={"features":{"descriptor_names":["a"]},"models":{p:{m:Constant(values[p]) for m in methods}
            for p,methods in expanded.shared.METHODS.items()},
            "consistency_projection":{"enabled":True,"inference_residual_variances":[1.,2.,3.]}}
    complementary={"feature_audit":{"descriptor_names":["a"]},"auxiliary_order":["dog","monkey","rat"],
        "animal_models":{s:Constant(0.) for s in expanded.animal.SPECIES},
        "human_models":{m:Constant(2.) for m in expanded.shared.METHODS["total_clearance"]},"kernel_model":Constant(3.)}
    original=deepcopy(shared["consistency_projection"])
    final,base,parts=expanded.predict_parameters(shared,complementary,cohort())
    assert final["drug"]["fraction_unbound"]==1.  # fixed physical-log clipping
    assert final["drug"]["total_clearance"]==pytest.approx(10.**np.mean([parts[k]["drug"] for k in parts]))
    for p in values:
        if p!="total_clearance":assert final["drug"][p]==base["drug"][p]
    assert final["drug"]["total_clearance"]!=base["drug"]["total_clearance"]
    assert shared["consistency_projection"]==original


def test_frozen_twenty_drug_model_inference_reconstructs_existing_parameters():
    """Model-only regression check: no clinical observations or scores read."""
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/"experiments/blueprint_design/clinical_pbpk_comparison_20261004/parameter_model/predictions_audit.json").read_text())
    cl=json.loads((root/"experiments/blueprint_design/blueprint_cl_refinement_complementary_20261004/prior/predictions.json").read_text())
    assert expanded.shared._sha(Path(old["model_path"]))==old["model_sha256"]
    assert expanded.shared._sha(root/cl["model_receipt"]["path"])==cl["model_receipt"]["sha256"]
    shared=joblib.load(old["model_path"]);complementary=joblib.load(root/cl["model_receipt"]["path"])
    final,base,_=expanded.predict_parameters(shared,complementary,old["audit"]["targets"])
    for cid,row in base.items():
        for p,value in row.items():assert value==pytest.approx(old["parameters"][cid][p],rel=1e-10,abs=1e-12)
        assert final[cid]["total_clearance"]==pytest.approx(cl["predictions"][cid],rel=1e-10,abs=1e-12)
