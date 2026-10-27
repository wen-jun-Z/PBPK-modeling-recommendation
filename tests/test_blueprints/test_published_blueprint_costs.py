"""Saved baseline cost accounting and source isolation; no scientific fits."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from published_blueprint_costs import COMPARATOR_DIRECTORY, load_baseline_costs


def fixture(*, duplicate_cell=False, missing_drug=False, missing_cost=False,
            different_training=False, missing_pin=False, repair=False, reuse_fit=False,
            observed_points=5, invalid_calls=None):
    documents, accessed = {}, []

    def encoded(document):
        return json.dumps(document, sort_keys=True, allow_nan=True).encode()

    def put(path, document):
        documents[path] = deepcopy(document)
        return {"path": path, "sha256": hashlib.sha256(encoded(document)).hexdigest()}

    def checked(ref):
        accessed.append(ref["path"])
        value = documents[ref["path"]]
        if hashlib.sha256(encoded(value)).hexdigest() != ref["sha256"]:
            raise ValueError("Changed source hash")
        return deepcopy(value)

    cases, jobs, cells, finished = {}, {}, [], {}
    training_refs = {}
    for i in range(30):
        cid = f"drug_{i:02d}"
        training = put(f"{cid}/training.json", {"case": cid, "observations": [{}] * observed_points})
        training_refs[cid] = training
        fits = []
        for formula in ("PKSimStandard", "RodgersRowland"):
            fit = {"case": cid, "formula": formula, "visible_points": 5,
                   "target_observations_used": False, "calls": 10 + i, "wall_seconds": 2. + i,
                   "training_source": training}
            if missing_cost and i == 0:
                fit.pop("wall_seconds")
            if invalid_calls is not None and i == 0:
                fit["calls"] = invalid_calls
            fits.append({"formula": formula, "fit": put(f"{cid}/{formula}.json", fit)})
        document = {"status": "predicted", "training": training, "fits": fits}
        if repair and i == 0:
            original = put(f"{cid}/original.json", document)
            document = deepcopy(document)
            if not reuse_fit:
                entries = [{"formula": entry["formula"], "fit": documents[entry["fit"]["path"]]}
                           for entry in fits]
                document["fits"] = put(f"{cid}/repaired_fits.json", entries)
            document["original_source"] = original
        result = put(f"{cid}/result.json", document)
        cases[cid] = {"frozen_pbpk_prediction": result}
        jobs[cid + "__external"] = {"case_id": cid, "adapter": {"training_source": training}}
        cells.append({"case_id": cid, "info": "time_five", "method": "two_compartment",
                      "visible_points": 5, "training_source": training, "job": cid + "__two"})
    if duplicate_cell:
        cells[-1] = deepcopy(cells[0])
    comparator_ref = put(f"{COMPARATOR_DIRECTORY}/protocol.json", {"cells": cells})
    for cid, training in training_refs.items():
        source_training = training_refs["drug_01"] if different_training and cid == "drug_00" else training
        doc = {"protocol": comparator_ref, "status": "predicted", "target_observations_used": False,
               "job": {"case_id": cid, "model": "two_compartment", "training_source": source_training},
               "fit": {"residual_calls": 20}, "wall_seconds": .25,
               "predictions": {"linear_weight": {"concentration": [1., .5]}}}
        finished[cid + "__two"] = put(f"{cid}/two.json", doc)
    state_ref = put(f"{COMPARATOR_DIRECTORY}/state.json", {
        "status": "complete", "protocol": comparator_ref, "completed": finished})
    protocol = {"cases": cases, "jobs": jobs, "source_pins": {
        comparator_ref["path"]: comparator_ref["sha256"], state_ref["path"]: state_ref["sha256"]}}
    if missing_drug:
        protocol["cases"].pop("drug_29")
    if missing_pin:
        protocol["source_pins"].pop(state_ref["path"])
    return protocol, checked, documents, accessed


def test_exact_aggregation_and_all_sources_checked():
    protocol, checked, _, accessed = fixture()
    result = load_baseline_costs(protocol, checked=checked)
    assert result["pbpk"]["n"] == result["two_compartment"]["n"] == 30
    assert result["pbpk"]["median_calls"] == 49
    assert result["pbpk"]["median_seconds"] == 33
    assert result["two_compartment"]["median_calls"] == 20
    assert result["two_compartment"]["median_seconds"] == .25
    assert "checkpoint I/O" in result["two_compartment"]["timing_scope"]
    for group in result.values():
        for row in group["rows"]:
            assert all(ref["path"] in accessed for ref in row["source_refs"])


@pytest.mark.parametrize("reuse_fit, expected_calls, artifacts", [(False, 40, 4), (True, 20, 2)])
def test_pre_repair_work_included_but_referenced_fit_not_counted_twice(reuse_fit, expected_calls, artifacts):
    protocol, checked, _, _ = fixture(repair=True, reuse_fit=reuse_fit)
    row = load_baseline_costs(protocol, checked=checked)["pbpk"]["rows"][0]
    assert row["calls"] == expected_calls
    assert row["seconds"] == expected_calls / 5
    assert row["pre_repair_documents"] == 1
    assert row["unique_formula_fit_artifacts"] == artifacts


@pytest.mark.parametrize("options", [
    {"missing_drug": True}, {"duplicate_cell": True}, {"missing_cost": True},
    {"different_training": True}, {"missing_pin": True}, {"observed_points": 4},
    {"invalid_calls": -1}, {"invalid_calls": float("nan")}, {"invalid_calls": True},
])
def test_invalid_or_incomplete_records_are_rejected(options):
    protocol, checked, _, _ = fixture(**options)
    with pytest.raises(ValueError):
        load_baseline_costs(protocol, checked=checked)


def test_source_mutation_cannot_silently_change_reported_cost():
    protocol, checked, documents, _ = fixture()
    documents["drug_00/PKSimStandard.json"]["wall_seconds"] = 0
    with pytest.raises(ValueError, match="Changed source hash"):
        load_baseline_costs(protocol, checked=checked)
