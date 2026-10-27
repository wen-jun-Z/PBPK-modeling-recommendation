"""Capacity aggregation and native interface contracts for perfusion export."""
from copy import deepcopy
from xml.etree import ElementTree as ET

import pytest

from pbpk_kb.blueprint_policy.perfusion_export import (
    BLOOD, COMPARTMENTS, ORGANS, PORTAL, WEIGHTS, assemble_export, discover_plan,
)


@pytest.fixture
def native_fixture(tmp_path):
    root = ET.Element("Root")
    identifiers, parameters = {}, {}

    def add(path, tag="Container", value=None):
        node = root
        parts = path.split("|")
        for index, part in enumerate(parts):
            children = node.find("Children")
            if children is None:
                children = ET.SubElement(node, "Children")
            found = next((c for c in children if c.get("name") == part), None)
            if found is None:
                found = ET.SubElement(children, tag if index == len(parts) - 1 else "Container",
                                      name=part, id=str(len(identifiers)))
                identifiers["|".join(parts[:index + 1])] = found
            node = found
        if value is not None:
            parameters[path] = value
        return node

    def param(path, value):
        return add(path, "Parameter", value)

    for organ in ORGANS + tuple(BLOOD.values()):
        regions = ([organ + "|Periportal", organ + "|Pericentral"] if organ == "Liver" else
                   [organ, organ + "|Mucosa|Duodenum"] if organ == "SmallIntestine" else [organ])
        for region in regions:
            kinds = COMPARTMENTS[:2] if organ in BLOOD.values() else COMPARTMENTS
            for kind in kinds:
                stem = "Organism|" + region + "|" + kind
                add(stem)
                add(stem + "|Drug", "MoleculeAmount")
                param(stem + "|Volume", 0. if region.endswith("Pericentral") else 1.)
            if organ not in BLOOD.values():
                for first, second, kp in (("Plasma", "Interstitial", .5), ("Interstitial", "Intracellular", 2.)):
                    path = "Neighborhoods|" + region.replace("|", "_") + "_" + first + "_" + second
                    n = add(path, "Neighborhood")
                    n.set("firstNeighbor", identifiers["Organism|" + region + "|" + first].get("id"))
                    n.set("secondNeighbor", identifiers["Organism|" + region + "|" + second].get("id"))
                    param(path + "|Drug|Partition coefficient (" + second.lower() + "/plasma)", kp)
        if organ in ORGANS:
            param("Organism|" + organ + "|Blood flow rate", 14. if organ == "Lung" else 1.)
    param("Organism|PortalVein|Blood flow rate", 5.)
    for organ in WEIGHTS:
        param("Organism|" + organ + "|Peripheral blood flow fraction", 1. if organ == "Skin" else 0.)
    for name, value in (("Blood/Plasma concentration ratio", .75), ("Partition coefficient (blood cells/plasma)", .5),
                        ("Fraction unbound (plasma)", .1), ("Molecular weight", 500.), ("Lipophilicity", 2.)):
        param("Drug|" + name, value)
    param("Organism|Weight", 70.)
    document = ET.Element("Document")
    ET.SubElement(ET.SubElement(document, "Simulation"), "Model").append(root)
    path = tmp_path / "native.pkml"
    ET.ElementTree(document).write(path)
    plan = discover_plan(path, "Drug")
    rows = [{"path": r["path"], "unit": r["unit"], "value": parameters[r["path"]]} for r in plan["parameters"]]
    return plan, rows, document, path


def test_two_zone_liver_and_gut_mucosa_count_each_amount_once(native_fixture):
    plan, rows, _, _ = native_fixture
    value = assemble_export(plan, rows, [])
    organs = {r["name"]: r for r in value["organs"]}
    assert len(organs) == 15
    assert len(organs["Liver"]["components"]) == 8
    assert organs["Liver"]["volume_l"] == 4.  # Native zero-sized zone is legitimate.
    assert organs["Liver"]["vascular_capacity_l"] == 1.5
    assert organs["Liver"]["extravascular_capacity_l"] == 2.5
    assert organs["SmallIntestine"]["volume_l"] == 8.
    assert organs["SmallIntestine"]["extravascular_capacity_l"] == 5.
    assert value["blood"]["portal"]["volume_l"] == 2.
    assert value["circulation"]["liver_total_flow_l_h"] == 360.
    assert value["hematocrit"] == .5


@pytest.mark.parametrize("path,value", [
    ("Drug|Blood/Plasma concentration ratio", 2.),
    ("Organism|PortalVein|Blood flow rate", 6.),
    ("Organism|Lung|Blood flow rate", 13.),
    ("Organism|Bone|Plasma|Volume", -1.),
])
def test_inconsistent_physiology_or_partition_is_rejected(native_fixture, path, value):
    plan, rows, _, _ = native_fixture
    changed = deepcopy(rows)
    next(r for r in changed if r["path"] == path)["value"] = value
    with pytest.raises(ValueError):
        assemble_export(plan, changed, [])


def test_partition_binding_uses_neighbor_ids_not_neighborhood_name(native_fixture):
    plan, _, document, path = native_fixture
    neighborhood = next(document.iter("Neighborhood"))
    neighborhood.set("secondNeighbor", neighborhood.get("firstNeighbor"))
    ET.ElementTree(document).write(path)
    with pytest.raises(ValueError, match="Missing semantic compartment partition"):
        discover_plan(path, "Drug")
