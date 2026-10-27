"""Scientific input isolation and the clinical prediction-lock barrier."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_cl_gap_comparison as comparison


SOURCE = {"path": "experiment/locked_prior.json", "sha256": "a" * 64}
PROTECTED_DRUGS = {
    "Dexmedetomidine", "Moxifloxacin", "Remdesivir",
    "Alprazolam", "Fluconazole", "Furosemide",
}


def request_fixture(*, basis="predicted", method="random_forest", weight=50.0):
    context = {"species": "human", "matrix": "plasma", "route": "intravenous"}
    return {
        "compound_id": "test:compound", "human": {"weight_kg": weight},
        "dosing": {"dose_mg": 25, "infusion_minutes": 30},
        "evidence_ledger": {"path": "original.json", "sha256": "b" * 64},
        "evidence": [
            {"parameter": "fraction_unbound", "basis": "predicted", "value": .04,
             "unit": "fraction", "prediction_method": {"name": "random_forest"}},
            {"parameter": "total_clearance", "basis": basis, "value": 3.0,
             "unit": "ml/min/kg", "prediction_method": {"name": method},
             "source": {"path": "original_cl.json"}, "context": context},
            {"parameter": "renal_clearance", "basis": "measured", "value": .2,
             "unit": "ml/min/kg", "context": context},
        ],
    }


def test_only_rf_total_clearance_changes_and_caller_input_is_immutable():
    original = request_fixture()
    before = deepcopy(original)
    changed, changes = comparison.replace_cl(original, 1.75, "tabpfn", SOURCE)
    assert original == before
    assert changed is not original
    assert len(changes) == 1 and changes[0]["parameter"] == "total_clearance"
    for field in set(original) - {"evidence"}:
        assert changed[field] == before[field]
    for old, new in zip(before["evidence"], changed["evidence"]):
        if old["parameter"] == "total_clearance":
            assert new["value"] == 1.75
            assert new["context"] == old["context"]
            assert new["unit"] == old["unit"]
        else:
            assert new == old
    changed["human"]["weight_kg"] = 999
    changed["evidence"][0]["value"] = .99
    assert original == before


@pytest.mark.parametrize("basis,method", [
    ("measured", "random_forest"), ("derived", "random_forest"),
    ("predicted", "independent_external_predictor"),
])
def test_independent_clearance_is_protected(basis, method):
    request = request_fixture(basis=basis, method=method)
    changed, changes = comparison.replace_cl(request, 999.0, "tabpfn", SOURCE)
    assert changed == request and changed is not request
    assert changes == []


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "3.2"])
def test_invalid_clearance_is_not_silently_adopted(value):
    with pytest.raises(ValueError, match="finite and positive"):
        comparison.replace_cl(request_fixture(), value, "tabpfn", SOURCE)


def test_absolute_clearance_uses_actual_request_weight_and_preserves_renal_input():
    first, _ = comparison.replace_cl(request_fixture(weight=50), 2.0, "tabpfn", SOURCE)
    second, _ = comparison.replace_cl(request_fixture(weight=80), 2.0, "tabpfn", SOURCE)
    for request, weight in ((first, 50), (second, 80)):
        actual = comparison.clearance_inputs(request)
        assert actual["total_clearance_l_h"] == pytest.approx(2.0 * weight * 60 / 1000)
        assert actual["renal_clearance_l_h"] == pytest.approx(.2 * weight * 60 / 1000)
        assert actual["renal_input_fixed"]
    wrong_unit = request_fixture()
    wrong_unit["evidence"][1]["unit"] = "L/h"
    with pytest.raises(ValueError, match="no implicit conversion"):
        comparison.replace_cl(wrong_unit, 2.0, "tabpfn", SOURCE)


def test_frozen_cohort_has_exactly_24_replaceable_and_six_protected_drugs():
    state, protocol, _, choices = comparison.common.source_bundle()
    assert len(protocol["cases"]) == 30
    assert len({c["case_id"] for c in protocol["cases"]}) == 30
    protected, replaceable = set(), set()
    for case in protocol["cases"]:
        before = deepcopy(case)
        cid = case["case_id"]
        choice = choices[cid + "__late_four"]["cmax"]["sources"]["time_repaired"]
        unit = comparison.checked(state["units"][choice["unit_id"]]["result"])
        assert len(comparison.checked(unit["training"])["observations"]) == 5
        for phase in ("development", "target"):
            request = comparison.frozen.gap.request(case, phase, "PKSimStandard")
            changed, changes = comparison.replace_cl(request, 1.2345, "tabpfn", SOURCE)
            if comparison.eligible_cl(request):
                replaceable.add(case["drug"])
                assert [r["parameter"] for r in changes] == ["total_clearance"]
            else:
                protected.add(case["drug"])
                assert not changes and changed == request
            assert [e for e in request["evidence"] if e["parameter"] != "total_clearance"] == [
                e for e in changed["evidence"] if e["parameter"] != "total_clearance"]
        assert case == before
    assert len(replaceable) == 24
    assert protected == PROTECTED_DRUGS
    assert not protected & replaceable


def lock_fixture():
    protocol = {"jobs": {f"case_{i}": {} for i in range(30)}}
    records = {method + "__" + cid: {"result": {"path": method + cid}, "reused": False}
               for method in comparison.METHODS for cid in protocol["jobs"]}
    return protocol, {"status": "predictions_locked", "jobs_done": records}


@pytest.mark.parametrize("defect", ["still_running", "missing_method_case", "unexpected_case"])
def test_incomplete_locks_reject_before_opening_prediction_or_target_files(monkeypatch, defect):
    protocol, state = lock_fixture()
    if defect == "still_running":
        state["status"] = "running"
    elif defect == "missing_method_case":
        state["jobs_done"].pop("tabpfn__case_29")
    else:
        state["jobs_done"]["tabpfn__unexpected"] = {"result": {"path": "wrong"}, "reused": False}
    def forbidden_read(*args):
        pytest.fail("A data file was read before all ninety attempts locked")
    monkeypatch.setattr(comparison, "checked", forbidden_read)
    with pytest.raises(ValueError, match="All ninety"):
        comparison.require_prediction_locks(protocol, state)


def test_prediction_lock_verifies_saved_artifacts_and_accepts_locked_failures(monkeypatch):
    protocol, state = lock_fixture()
    seen = []
    monkeypatch.setattr(comparison, "checked", lambda ref: {
        "status": "failed", "output_pins": {ref["path"]: "c" * 64}})
    monkeypatch.setattr(comparison.frozen, "verify", lambda pins: seen.append(pins))
    assert comparison.require_prediction_locks(protocol, state)
    assert len(seen) == 90
    def corrupted(pins):
        raise ValueError("Changed output hash")
    monkeypatch.setattr(comparison.frozen, "verify", corrupted)
    with pytest.raises(ValueError, match="Changed output hash"):
        comparison.require_prediction_locks(protocol, state)


def test_report_cannot_read_clinical_outcomes_before_prediction_lock(monkeypatch):
    protocol, state = lock_fixture()
    state["status"] = "running"
    monkeypatch.setattr(comparison, "read", lambda path: protocol if path.name == "protocol.json" else state)
    def forbidden_score(*args):
        pytest.fail("Clinical outcomes were opened before prediction lock")
    monkeypatch.setattr(comparison.frozen.data, "score_case", forbidden_score)
    with pytest.raises(ValueError, match="All ninety"):
        comparison.report()
