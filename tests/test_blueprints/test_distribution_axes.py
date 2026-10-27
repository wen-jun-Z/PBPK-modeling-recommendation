"""Semantic native binding and readback contracts, without clinical observations."""
from copy import deepcopy
import json
from xml.etree import ElementTree as ET

import pytest

from pbpk_kb.blueprint_policy import distribution_axes as axes


def _fixture_pkml(tmp_path):
    transfer = ET.Element("SimulationTransfer")
    root = ET.SubElement(ET.SubElement(ET.SubElement(transfer, "Simulation"), "Model"), "Root")
    nodes = {(): root}

    def add(path, tag="Container", **attrs):
        if path in nodes:
            return nodes[path]
        parent = add(path[:-1])
        children = parent.find("Children")
        if children is None:
            children = ET.SubElement(parent, "Children")
        node = ET.SubElement(children, tag, name=path[-1], id="id"+str(len(nodes)), **attrs)
        nodes[path] = node
        return node

    for index, tissue in enumerate(axes.TISSUES):
        first = add(tissue + ("Interstitial",))
        second = add(tissue + ("Intracellular",))
        # Opaque neighborhood names force semantic ID resolution.
        base = ("Neighborhoods", "unrelated_name_"+str(index))
        add(base, "Neighborhood", firstNeighbor=first.get("id"), secondNeighbor=second.get("id"))
        for name in (axes.PARTITION, *axes.FACTORS):
            add(base + ("Drug", name), "Parameter", value="1")
        add(tissue + ("Volume",), "Parameter", dim="Volume", value="1")
    for name in ("Lipophilicity", "Fraction unbound (plasma)", "Fraction unbound (plasma, reference value)", "Permeability", "pKa value 0"):
        add(("Drug", name), "Parameter", value="1")
    add(("Drug-Total Hepatic Clearance-EvidenceAssembly", "Plasma clearance"), "Parameter", dim="Flow per weight", value="1")
    # Decoy mucosa must not become a 17th distribution tissue.
    mucosa = ("Organism", "SmallIntestine", "Mucosa", "Duodenum")
    first, second = add(mucosa + ("Interstitial",)), add(mucosa + ("Intracellular",))
    base = ("Neighborhoods", "Muscle_int_Muscle_cell_lookalike")
    add(base, "Neighborhood", firstNeighbor=first.get("id"), secondNeighbor=second.get("id"))
    for name in (axes.PARTITION, *axes.FACTORS):
        add(base + ("Drug", name), "Parameter", value="1")
    path = tmp_path / "source.pkml"
    ET.ElementTree(transfer).write(path)
    return path, transfer, nodes


def test_semantic_neighborhood_binding_ignores_names_and_mucosa(tmp_path):
    path, _, _ = _fixture_pkml(tmp_path)
    plan = axes.discover_plan(path, "Drug")
    assert len(plan["changed"]) == 48
    assert len({x["tissue"] for x in plan["changed"]}) == 16
    assert all("lookalike" not in x["path"] for x in plan["changed"])
    assert all(x["unit"] == "fraction" for x in plan["changed"])
    for tissue in axes.TISSUES:
        rows = [x for x in plan["changed"] if x["tissue"] == "|".join(tissue)]
        assert [x["axis"] for x in rows] == [axes.AXES[0], axes.AXES[1], axes.AXES[1]]
    assert "mucosa" in plan["scope"]
    assert "Drug|pKa value 0" in {x["path"] for x in plan["protected"]}


@pytest.mark.parametrize("damage", ["neighbor", "duplicate_id", "dimension", "missing"])
def test_invalid_native_semantics_rejected(tmp_path, damage):
    path, tree, nodes = _fixture_pkml(tmp_path)
    neighborhood = nodes[("Neighborhoods", "unrelated_name_0")]
    parameter = nodes[("Neighborhoods", "unrelated_name_0", "Drug", axes.PARTITION)]
    if damage == "neighbor":
        neighborhood.set("secondNeighbor", nodes[("Organism", "Brain", "Intracellular")].get("id"))
    elif damage == "duplicate_id":
        parameter.set("id", neighborhood.get("id"))
    elif damage == "dimension":
        parameter.set("dim", "Volume")
    else:
        parameter.set("name", "Different partition quantity")
    ET.ElementTree(tree).write(path)
    with pytest.raises(ValueError):
        axes.discover_plan(path, "Drug")


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), "1"])
def test_invalid_scale_rejected(value):
    with pytest.raises(ValueError, match="finite and positive"):
        axes.validate_candidates([{axes.AXES[0]: value, axes.AXES[1]: 1.}])


def test_candidates_do_not_accept_other_axes_or_unbounded_count():
    with pytest.raises(ValueError):
        axes.validate_candidates([{"partition_scale": 1., "clearance_scale": 1.}])
    with pytest.raises(ValueError):
        axes.validate_candidates([{name: 1. for name in axes.AXES}] * 9)
    assert axes.validate_candidates([{name: 1 for name in axes.AXES}]) == [{name: 1. for name in axes.AXES}]


def test_readback_verifies_changes_and_protection_not_echoed_scale():
    specs = [{"path": "Exact|K", "axis": "partition_scale"}]
    candidate = {"partition_scale": 2., "bidirectional_exchange_scale": .5}
    rows = [{"path": "Exact|K", "before": 3., "after": 6., "unit": ""}]
    axes._validate_readback(rows, specs, candidate, changed=True)
    for mutation in ({"after": 3.}, {"unit": "l"}, {"path": "Other|K"}, {"after": None}):
        bad = [dict(rows[0], **mutation)]
        with pytest.raises(ValueError):
            axes._validate_readback(bad, specs, candidate, changed=True)
    with pytest.raises(ValueError, match="mismatch"):
        axes._validate_readback(rows, specs, candidate, changed=False)
    protected = [dict(rows[0], after=3.)]
    axes._validate_readback(protected, specs, candidate, changed=False)


def _source(tmp_path):
    folder = tmp_path / "native"
    folder.mkdir()
    pkml, _, _ = _fixture_pkml(folder)
    snapshot = {"Version": 80, "Compounds": [{"Name": "Drug", "Lipophilicity": [{"Name": "EvidenceLogP"}],
        "FractionUnbound": [{"Name": "EvidenceFu"}], "Processes": [{"DataSource": "EvidenceAssembly"}]}],
        "Simulations": [{}], "Protocols": [{"ApplicationType": "Intravenous", "DosingInterval": "Single"}]}
    snapshot_path = folder / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))
    manifest = {"engineering_accepted": True, "source_unchanged": True,
        "source_sha256": axes._sha(snapshot_path), "pkml_outputs": [{"path": pkml.name, "sha256": axes._sha(pkml)}]}
    manifest_path = folder / "execution_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, snapshot_path, manifest, snapshot


def test_source_hash_and_fitted_snapshot_rejection(tmp_path):
    manifest_path, snapshot_path, manifest, snapshot = _source(tmp_path)
    bound = axes._bind(manifest_path, axes._sha(manifest_path), tmp_path)
    assert bound[-1] == "Drug"
    with pytest.raises(ValueError, match="manifest hash"):
        axes._bind(manifest_path, "0"*64, tmp_path)
    snapshot["Compounds"][0]["Lipophilicity"][0]["Name"] = "PKAnchorEffectiveMembraneAffinity"
    snapshot_path.write_text(json.dumps(snapshot))
    manifest["source_sha256"] = axes._sha(snapshot_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Fresh evidence"):
        axes._bind(manifest_path, axes._sha(manifest_path), tmp_path)


def test_oral_source_rejected_before_execution(tmp_path):
    manifest_path, snapshot_path, manifest, snapshot = _source(tmp_path)
    snapshot["Protocols"][0]["ApplicationType"] = "Oral"
    snapshot_path.write_text(json.dumps(snapshot))
    manifest["source_sha256"] = axes._sha(snapshot_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="single-IV"):
        axes._bind(manifest_path, axes._sha(manifest_path), tmp_path)


def test_external_model_path_rejected(tmp_path):
    manifest_path, _, manifest, _ = _source(tmp_path)
    manifest["pkml_outputs"][0]["path"] = "../../external.pkml"
    external = tmp_path.parent / "external.pkml"
    external.write_text("outside")
    manifest["pkml_outputs"][0]["sha256"] = axes._sha(external)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside the project"):
        axes._bind(manifest_path, axes._sha(manifest_path), tmp_path)
