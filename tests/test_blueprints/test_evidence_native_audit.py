"""Native readback and clearance-capacity guards, independent of ODE execution."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from pbpk_kb.blueprints import evidence_native_audit as audit


def _snapshot():
    def p(name, value, unit=None):
        return {"Name": name, "Value": value, **({"Unit": unit} if unit else {})}
    return {"Version": 80, "Simulations": [{}], "Compounds": [{
        "Name": "ToyDrug", "Parameters": [p("Molecular weight", 194.19, "g/mol")],
        "Lipophilicity": [{"Parameters": [p("Lipophilicity", -.1, "Log Units")]}],
        "FractionUnbound": [{"Parameters": [p("Fraction unbound (plasma, reference value)", .5)]}],
        "Solubility": [{"Parameters": [p("Solubility at reference pH", 20., "mg/ml"), p("Reference pH", 7.)]}],
        "PlasmaProteinBindingPartner": "Albumin", "PkaTypes": [{"Type": "Acid", "Pka": 4.}],
        "Processes": [{"InternalName": "LiverClearance", "DataSource": "EvidenceAssembly", "Species": "Human",
            "Parameters": [p("Plasma clearance", 1., "ml/min/kg"), p("Fraction unbound (experiment)", .5),
                           p("Lipophilicity (experiment)", -.1, "Log Units")]}]}]}


def _physical(organ="hepatic", cl=1., unit="ml/min/kg"):
    process = {"organ": organ, "stem": "Native|Clearance", "clearance_unit": unit}
    fields = {"Plasma clearance": cl, "Body weight": 70., "Fraction unbound (experiment)": .5}
    if organ == "hepatic":
        fields.update({"Blood flow rate (liver)": .4, "Blood flow rate (portal vein)": 1.1,
                       "Blood/Plasma concentration ratio": 1., "Volume (liver)": 2., "Fraction intracellular (liver)": .75})
        specific = .07 * 1.5 / (.5 * (1.5 - .07) * 1.5)
    else:
        fields.update({"Blood flow rate (kidney)": 1., "Hematocrit": .4, "Volume (kidney)": .3})
        specific = .07 * .6 / (.5 * (.6 - .07) * .3)
    fields["Specific clearance"] = specific
    return process, {process["stem"] + "|" + key: value for key, value in fields.items()}


@pytest.mark.parametrize("organ", ["hepatic", "renal"])
@pytest.mark.parametrize("unit,normalized_cl", [("ml/min/kg", 1.), ("l/h/kg", .06), ("ml/h/kg", 60.)])
def test_native_clearance_formula_units_and_organ_capacity(organ, unit, normalized_cl):
    process, values = _physical(organ, normalized_cl, unit)
    result = audit._capacity(process, values)
    assert result["accepted"]
    assert result["plasma_clearance_l_min"] == pytest.approx(.07)
    assert result["native_formula_matches"] and result["native_fallback_1000000_absent"]
    assert result["recomputed_specific_clearance_per_min"] == pytest.approx(values[process["stem"] + "|Specific clearance"])


@pytest.mark.parametrize("organ", ["hepatic", "renal"])
@pytest.mark.parametrize("mutation", ["above_flow", "fallback", "wrong_specific", "zero_fu", "fu_above_one", "invalid_fraction"])
def test_impossible_clearance_or_wrong_native_value_rejected(organ, mutation):
    process, values = _physical(organ)
    stem = process["stem"] + "|"
    if mutation == "above_flow": values[stem + "Plasma clearance"] = 1000.
    elif mutation == "fallback": values[stem + "Specific clearance"] = 1000000.
    elif mutation == "wrong_specific": values[stem + "Specific clearance"] *= 2.
    elif mutation == "zero_fu": values[stem + "Fraction unbound (experiment)"] = 0.
    elif mutation == "fu_above_one": values[stem + "Fraction unbound (experiment)"] = 1.1
    else: values[stem + ("Fraction intracellular (liver)" if organ == "hepatic" else "Hematocrit")] = 1.1
    assert audit._capacity(process, values)["accepted"] is False


def test_native_plan_includes_explicit_properties_and_neutral_unused_pka_slots():
    plan, processes = audit._plan(_snapshot())
    expected = {row["path"]: row.get("expected") for row in plan}
    assert expected["ToyDrug|Molecular weight"] == 194.19
    assert expected["ToyDrug|Compound type 0"] == -1
    assert expected["ToyDrug|pKa value 0"] == 4.
    assert expected["ToyDrug|Compound type 1"] == 0
    assert expected["ToyDrug|pKa value 2"] == 0.
    assert expected["ToyDrug|Plasma protein binding partner"] == 1
    assert len(plan) == len({r["path"] for r in plan})
    assert processes[0]["organ"] == "hepatic"


@pytest.mark.parametrize("mutation", ["duplicate", "specific_override", "unsupported_process", "bool", "nan", "missing_fu"])
def test_unreviewed_or_ambiguous_native_input_plan_rejected(mutation):
    snapshot = _snapshot()
    compound = snapshot["Compounds"][0]
    process = compound["Processes"][0]
    if mutation == "duplicate": compound["Processes"].append(deepcopy(process))
    elif mutation == "specific_override": process["Parameters"].append({"Name": "Specific clearance", "Value": .1, "Unit": "1/min"})
    elif mutation == "unsupported_process": process["InternalName"] = "MetabolizingEnzyme"
    elif mutation == "missing_fu": process["Parameters"] = process["Parameters"][:1]
    else: compound["Parameters"][0]["Value"] = True if mutation == "bool" else float("nan")
    with pytest.raises(ValueError): audit._plan(snapshot)


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    root = tmp_path / "project"
    native = root / "native"
    native.mkdir(parents=True)
    snapshot = _snapshot()
    source = native / "snapshot.json";source.write_text(json.dumps(snapshot))
    pkml = native / "generated.pkml";pkml.write_text("Synthetic native receipt fixture, not an executable model")
    manifest_path = native / "execution_manifest.json"
    manifest = {"engineering_accepted": True, "source_unchanged": True,
                "source_sha256": audit.sha256(source), "pkml_outputs": [{"path": pkml.name, "sha256": audit.sha256(pkml)}]}
    manifest_path.write_text(json.dumps(manifest))
    database = root / "formula.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE tab_calculation_method_rates (calculation_method TEXT, formula_rate TEXT, formula TEXT)")
        for name in audit._FORMULAS.values():
            db.execute("INSERT INTO tab_calculation_method_rates VALUES (?,?,?)", ("LinksCommon", name, "unit fixture formula"))
    monkeypatch.setattr(audit, "_DATABASE", "formula.sqlite")
    monkeypatch.setattr(audit, "_DATABASE_SHA", audit.sha256(database))
    state = {"mutation": None, "calls": 0}
    def read_only_runtime(**kwargs):
        state["calls"] += 1
        request_path = Path(kwargs["args"][0])
        request = json.loads(request_path.read_text())
        assert kwargs["args"][1] == audit.sha256(request_path)
        assert kwargs["timeout_seconds"] == 180 and kwargs["cpu_threads"] == 2
        physical, values = _physical()
        rows = []
        for planned in request["parameters"]:
            field = planned["path"].rsplit("|", 1)[-1]
            value = planned.get("expected", values.get(physical["stem"] + "|" + field))
            assert value is not None
            rows.append({"path": planned["path"], "unit": planned["unit"], "value": value, "status": "evaluated"})
        result = {"request_sha256": audit.sha256(request_path), "pkml_sha256": audit.sha256(pkml),
                  "parameters": rows, "solver_calls": 0, "model_modified": False}
        mutation = state["mutation"]
        if mutation == "input_mismatch": rows[0]["value"] *= 2
        elif mutation == "unit": rows[0]["unit"] = "mg/mol"
        elif mutation == "mapping": rows.reverse()
        elif mutation == "solver": result["solver_calls"] = 1
        elif mutation == "modified": result["model_modified"] = True
        elif mutation == "manifest": manifest_path.write_text(json.dumps({**manifest, "changed": True}))
        elif mutation == "pkml": pkml.write_text("changed native data")
        elif mutation == "fallback":
            next(r for r in rows if r["path"].endswith("|Specific clearance"))["value"] = 1000000.
        Path(request["result_path"]).write_text(json.dumps(result))
        return {"status": "completed", "returncode": 0}
    monkeypatch.setattr(audit, "_run_bounded_r", read_only_runtime)
    return root, snapshot, manifest_path, state


def test_readback_acceptance_does_not_claim_scientific_or_mass_balance_validation(native_fixture):
    root, snapshot, manifest, state = native_fixture
    result = audit.audit_evidence_execution(snapshot, manifest, root / "audit", root)
    assert result["accepted"] is True and result["reasons"] == []
    assert result["scientific_accepted"] is False and result["mass_balance_checked"] is False
    assert result["solver_calls"] == 0 and result["model_modified"] is False
    assert result["clearance_checks"][0]["accepted"]
    assert state["calls"] == 1
    assert json.loads((root / "audit/audit.json").read_text()) == result


@pytest.mark.parametrize("mutation", ["input_mismatch", "unit", "fallback"])
def test_failed_readback_is_reported_and_cannot_pass(native_fixture, mutation):
    root, snapshot, manifest, state = native_fixture
    state["mutation"] = mutation
    result = audit.audit_evidence_execution(snapshot, manifest, root / "audit", root)
    assert result["accepted"] is False and result["status"] == "failed"
    assert result["reasons"]


@pytest.mark.parametrize("mutation", ["mapping", "solver", "modified", "manifest", "pkml"])
def test_binding_changes_or_non_readonly_behavior_cannot_pass(native_fixture, mutation):
    root, snapshot, manifest, state = native_fixture
    state["mutation"] = mutation
    with pytest.raises(ValueError):
        audit.audit_evidence_execution(snapshot, manifest, root / "audit", root)
    assert not (root / "audit/audit.json").exists()


def test_wrong_snapshot_cannot_invoke_native_worker(native_fixture):
    root, snapshot, manifest, state = native_fixture
    snapshot["Compounds"][0]["Parameters"][0]["Value"] *= 2
    with pytest.raises(ValueError, match="Snapshot does not match"):
        audit.audit_evidence_execution(snapshot, manifest, root / "audit", root)
    assert state["calls"] == 0
