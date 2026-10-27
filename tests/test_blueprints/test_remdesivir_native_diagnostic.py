"""Native diagnostic binding contracts, without running or fitting a model."""
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("remdesivir_diagnostic", ROOT / "scripts/diagnose_remdesivir_native.py")
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


@pytest.fixture
def model(tmp_path):
    root = ET.Element("Root")
    next_id = iter(range(1000))

    def add(path, tag, **attrs):
        node = root
        for i, name in enumerate(path.split("|")):
            children = node.find("Children")
            if children is None:
                children = ET.SubElement(node, "Children")
            found = next((x for x in children if x.get("name") == name), None)
            if found is None:
                found = ET.SubElement(children, tag if i == len(path.split("|")) - 1 else "Container",
                                      name=name, id=str(next(next_id)))
            node = found
        node.attrib.update(attrs)
        return node

    for path in ["Organism|PeripheralVenousBlood|Remdesivir|Plasma (Peripheral Venous Blood)",
                 "Organism|VenousBlood|Plasma|Remdesivir|Concentration in container",
                 "Organism|ArterialBlood|Plasma|Remdesivir|Concentration in container"]:
        add(path, "Observer", dim="Concentration (molar)")
    urine = add("Organism|Kidney|Urine|Remdesivir", "MoleculeAmount", dim="Amount")
    for region in ("Periportal", "Pericentral"):
        stem = "Organism|Liver|" + region + "|Intracellular|"
        parent = add(stem + "Remdesivir", "MoleculeAmount", dim="Amount")
        product = add(stem + "Remdesivir-Undefined Liver Metabolite", "MoleculeAmount", dim="Amount")
        reaction = add(stem + "Liver clearance", "Reaction")
        ET.SubElement(ET.SubElement(reaction, "Educts"), "ReactionPartner", partner=parent.get("id"), stoichiometricCoefficient="1")
        ET.SubElement(ET.SubElement(reaction, "Products"), "ReactionPartner", partner=product.get("id"), stoichiometricCoefficient="1")
    for name in ("Dose", "Start time", "Infusion time"):
        add("Events|Dose|ProtocolSchemaItem|" + name, "Parameter")
    add("Events|Dose|Remdesivir", "MoleculeAmount", dim="Amount")
    outer = ET.Element("Document")
    ET.SubElement(ET.SubElement(outer, "Simulation"), "Model").append(root)
    path = tmp_path / "source.pkml"
    ET.ElementTree(outer).write(path)
    return path, outer, add, urine


def test_binding_preserves_source_and_labels_hepatic_amounts_as_bookkeeping(model):
    path, _, _, _ = model
    before = path.read_bytes()
    rows, reactions, protocol = DIAGNOSTIC.discover(path)
    assert path.read_bytes() == before
    assert sum(r["kind"] == "blood_plasma" for r in rows) == 3
    counters = [r for r in rows if r["kind"] == "hepatic_product_counter"]
    assert len(counters) == 2
    assert all(r["path"].endswith("Remdesivir-Undefined Liver Metabolite") for r in counters)
    assert not any("GS-" in r["path"] for r in rows)
    assert all(r["stoichiometry"] == "1:1" for r in reactions)
    assert len(protocol) == 3


@pytest.mark.parametrize("fault", ["amount_dimension", "reaction_stoichiometry", "urine_outgoing"])
def test_unverified_amount_or_cumulative_counter_is_rejected(model, fault):
    path, outer, add, urine = model
    if fault == "amount_dimension":
        urine.set("dim", "Concentration (molar)")
    elif fault == "reaction_stoichiometry":
        outer.find(".//Products/ReactionPartner").set("stoichiometricCoefficient", "2")
    else:
        add("Neighborhoods|Unexpected urine outlet", "Transport", sourceAmount=urine.get("id"))
    ET.ElementTree(outer).write(path)
    with pytest.raises(ValueError):
        DIAGNOSTIC.discover(path)


def test_target_model_cannot_be_used_as_development_diagnostic(model, tmp_path, monkeypatch):
    path, _, _, _ = model
    base = tmp_path / "experiments/blueprint_design/architecture_20260930/clinical_pilot/work"
    folder = base / "remdesivir__full__extent_exchange_clearance"
    mapping = folder / "RodgersRowland/mapping_locked.json"
    target = folder / "target/execution"
    mapping.parent.mkdir(parents=True)
    target.mkdir(parents=True)
    (target / "candidate.pkml").write_bytes(path.read_bytes())
    mapping.write_text(json.dumps({"compound_id": "pubchem:121304016", "development_curve_used": True,
        "decision": {"native_output_dir": str(target.relative_to(tmp_path))}}))
    monkeypatch.setattr(DIAGNOSTIC, "ROOT", tmp_path)
    monkeypatch.setattr(DIAGNOSTIC.sys, "argv", ["diagnose", "--mapping", str(mapping), "--output", "unused"])
    with pytest.raises(ValueError, match="never target model"):
        DIAGNOSTIC.main()
    assert not (tmp_path / "unused").exists()
