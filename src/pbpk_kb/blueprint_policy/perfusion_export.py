"""Read fresh native physiology/partition capacities for a reduced model.

No ODE, calibration, physiological edits, or outcome access. Compartment amounts
are aggregated once, including GI mucosa and both hepatic zones. The exported
equilibrium capacities are definitions for a prospective perfusion model, not
native transient tissue equilibrium or clinical Vss claims.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from defusedxml import ElementTree as ET

from . import distribution_axes as source
from pbpk_kb.blueprints import evidence_native_audit as clearance

VERSION = "native-perfusion-physiology-1"
WORKER = "src/pbpk_kb/model_runtime/R/export_perfusion_physiology.R"
ORGANS = ("Bone", "Brain", "Fat", "Gonads", "Heart", "Kidney", "Liver", "Stomach",
          "SmallIntestine", "LargeIntestine", "Lung", "Muscle", "Pancreas", "Skin", "Spleen")
BLOOD = {"arterial": "ArterialBlood", "venous": "VenousBlood", "portal": "PortalVein"}
COMPARTMENTS = ("Plasma", "BloodCells", "Interstitial", "Intracellular")
PORTAL = ("Stomach", "SmallIntestine", "LargeIntestine", "Pancreas", "Spleen")
WEIGHTS = ("ArterialBlood", "Bone", "Fat", "Muscle", "Skin")


def discover_plan(pkml, compound):
    root = ET.parse(pkml).getroot().find("Simulation/Model/Root")
    if root is None:
        raise ValueError("Native model root missing")
    paths, ids = {}, {}

    def walk(node, prefix=()):
        for child in node.findall("./Children/*"):
            name = child.get("name")
            if not name or any(c in name for c in "|\r\n"):
                raise ValueError("Invalid native path component")
            path = prefix + (name,)
            if path in paths or child.get("id") in ids:
                raise ValueError("Duplicate native path/id")
            paths[path] = child
            if child.get("id"):
                ids[child.get("id")] = path
            walk(child, path)

    walk(root)
    parameters = {}

    def parameter(path, unit):
        text = "|".join(path)
        node = paths.get(path)
        if node is None or node.tag != "Parameter":
            raise ValueError("Required exact native parameter absent: " + text)
        spec = {"path": text, "unit": unit}
        if text in parameters and parameters[text] != spec:
            raise ValueError("Inconsistent parameter unit")
        parameters[text] = spec
        return text

    partition = {}
    for path, node in paths.items():
        if node.tag != "Neighborhood":
            continue
        first, second = ids.get(node.get("firstNeighbor")), ids.get(node.get("secondNeighbor"))
        if not first or not second:
            raise ValueError("Unresolved neighborhood identity")
        if first[:-1] != second[:-1] or first[:1] != ("Organism",) or first[1] not in ORGANS:
            continue
        first_amount, second_amount = paths.get(first + (compound,)), paths.get(second + (compound,))
        first_active = first_amount is not None and first_amount.tag == "MoleculeAmount"
        second_active = second_amount is not None and second_amount.tag == "MoleculeAmount"
        if not first_active and not second_active:
            continue  # Uninstantiated parent liver interfaces coexist with the two active zones.
        if first_active != second_active:
            raise ValueError("Only one native interface endpoint has the drug")
        if (first[-1], second[-1]) == ("Interstitial", "Intracellular"):
            kind, leaf = "Intracellular", "Partition coefficient (intracellular/plasma)"
        elif (first[-1], second[-1]) == ("Plasma", "Interstitial"):
            kind, leaf = "Interstitial", "Partition coefficient (interstitial/plasma)"
        else:
            continue
        key = first[:-1] + (kind,)
        if key in partition:
            raise ValueError("Ambiguous compartment partition")
        partition[key] = parameter(path + (compound, leaf), "fraction")
    global_paths = {name: parameter((compound, leaf), unit) for name, leaf, unit in (
        ("blood_to_plasma_ratio", "Blood/Plasma concentration ratio", "fraction"),
        ("rbc_kp_plasma", "Partition coefficient (blood cells/plasma)", "fraction"),
        ("fraction_unbound_plasma", "Fraction unbound (plasma)", "fraction"),
        ("molecular_weight_g_mol", "Molecular weight", "g/mol"),
        ("logp", "Lipophilicity", "Log Units"))}
    global_paths["weight_kg"] = parameter(("Organism", "Weight"), "kg")
    organs, blood = [], {}
    for name in ORGANS + tuple(BLOOD.values()):
        region_groups = {}
        for path, node in paths.items():
            if (len(path) >= 3 and path[:2] == ("Organism", name) and path[-1] in COMPARTMENTS
                    and node.tag == "Container" and paths.get(path + (compound,)) is not None
                    and paths[path + (compound,)].tag == "MoleculeAmount"):
                molecule = paths[path + (compound,)]
                if molecule.tag != "MoleculeAmount":
                    raise ValueError("Compartment lacks parent amount")
                region_groups.setdefault(path[:-1], []).append(path)
        expected = set(COMPARTMENTS[:2] if name in BLOOD.values() else COMPARTMENTS)
        components = []
        for region, compartments in sorted(region_groups.items()):
            if {p[-1] for p in compartments} != expected:
                raise ValueError("Incomplete native region compartments: " + "|".join(region))
            for path in sorted(compartments):
                kind = path[-1]
                kp_path = (None if kind == "Plasma" else global_paths["rbc_kp_plasma"] if kind == "BloodCells"
                           else partition.get(path))
                if kind in {"Interstitial", "Intracellular"} and kp_path is None:
                    raise ValueError("Missing semantic compartment partition: " + "|".join(path))
                components.append({"path": "|".join(path), "region": "|".join(region), "kind": kind,
                    "volume_path": parameter(path + ("Volume",), "l"), "kp_path": kp_path})
        if not components:
            raise ValueError("No native compartment volumes for " + name)
        row = {"name": name, "components": components}
        if name in ORGANS:
            row["flow_path"] = parameter(("Organism", name, "Blood flow rate"), "l/min")
            organs.append(row)
        else:
            blood[next(k for k, v in BLOOD.items() if v == name)] = row
    liver_regions = {x["region"] for x in next(o for o in organs if o["name"] == "Liver")["components"]}
    if liver_regions != {"Organism|Liver|Periportal", "Organism|Liver|Pericentral"}:
        raise ValueError("Hepatic two-zone mapping differs")
    weights = {name: parameter(("Organism", name, "Peripheral blood flow fraction"), "fraction") for name in WEIGHTS}
    portal_flow = parameter(("Organism", "PortalVein", "Blood flow rate"), "l/min")
    return {"organs": organs, "blood": blood, "global_paths": global_paths,
            "peripheral_weight_paths": weights, "portal_flow_path": portal_flow,
            "parameters": list(parameters.values()),
            "excluded_organ_containers": sorted(p[1] for p, n in paths.items() if len(p) == 2
                and p[0] == "Organism" and n.get("containerType") == "Organ"
                and p[1] not in ORGANS + tuple(BLOOD.values()))}


def assemble_export(plan, rows, processes):
    expected = [r["path"] for r in plan["parameters"]]
    if [r.get("path") for r in rows] != expected:
        raise ValueError("Native readback does not cover exact ordered plan")
    for spec, row in zip(plan["parameters"], rows):
        if (row.get("unit") != spec["unit"] or isinstance(row.get("value"), bool)
                or not isinstance(row.get("value"), (int, float)) or not math.isfinite(row["value"])):
            raise ValueError("Nonfinite or noncanonical native readback")
    values = {r["path"]: r["value"] for r in rows}
    result = {k: values[p] for k, p in plan["global_paths"].items()}
    if result["molecular_weight_g_mol"] <= 0 or result["weight_kg"] <= 0 or not 0 < result["fraction_unbound_plasma"] <= 1:
        raise ValueError("Invalid native drug or body inputs")

    def aggregate(spec):
        components = []
        for entry in spec["components"]:
            volume, kp = values[entry["volume_path"]], values[entry["kp_path"]] if entry["kp_path"] else 1.
            if volume < 0 or kp <= 0:
                raise ValueError("Negative compartment volume or nonpositive partition")
            components.append({**entry, "volume_l": volume, "kp_plasma": kp, "capacity_l": volume * kp})
        volumes = {kind: sum(r["volume_l"] for r in components if r["kind"] == kind) for kind in COMPARTMENTS}
        vascular = sum(r["capacity_l"] for r in components if r["kind"] in COMPARTMENTS[:2])
        extravascular = sum(r["capacity_l"] for r in components if r["kind"] in COMPARTMENTS[2:])
        volume = sum(volumes.values())
        if volume <= 0:
            raise ValueError("Aggregate organ/blood volume must be positive")
        ev = volumes["Interstitial"] + volumes["Intracellular"]
        row = {"name": spec["name"], "volume_l": volume, "native_organ_volume_l": None,
            "volume_definition": "sum of disjoint parent-containing plasma/RBC/interstitial/intracellular native compartments",
            "plasma_volume_l": volumes["Plasma"], "rbc_volume_l": volumes["BloodCells"],
            "interstitial_volume_l": volumes["Interstitial"], "intracellular_volume_l": volumes["Intracellular"],
            "vascular_volume_l": volumes["Plasma"] + volumes["BloodCells"], "extravascular_volume_l": ev,
            "vascular_capacity_l": vascular, "extravascular_capacity_l": extravascular,
            "kp_extravascular_plasma": extravascular / ev if ev else None,
            "kp_total_plasma": (vascular + extravascular) / volume, "components": components}
        if "flow_path" in spec:
            row.update(flow_l_min=values[spec["flow_path"]], flow_l_h=60 * values[spec["flow_path"]], flow_path=spec["flow_path"])
            if row["flow_l_min"] <= 0:
                raise ValueError("Nonpositive organ blood flow")
        return row

    result["organs"] = [aggregate(row) for row in plan["organs"]]
    result["blood"] = {name: aggregate(row) for name, row in plan["blood"].items()}
    venous = result["blood"]["venous"]
    hct = venous["rbc_volume_l"] / venous["volume_l"]
    implied_bp = 1 - hct + hct * result["rbc_kp_plasma"]
    if not math.isclose(implied_bp, result["blood_to_plasma_ratio"], rel_tol=1e-6):
        raise ValueError("Native B/P and blood-cell equilibrium partition disagree")
    result.update(hematocrit=hct, hematocrit_definition="venous native RBC volume / (plasma + RBC volume)",
                  equilibrium_blood_to_plasma_ratio_check=implied_bp)
    flows = {o["name"]: o["flow_l_min"] for o in result["organs"]}
    portal = values[plan["portal_flow_path"]]
    if not math.isclose(sum(flows[o] for o in PORTAL), portal, rel_tol=1e-6):
        raise ValueError("Portal tributary flows do not balance")
    if not math.isclose(sum(v for k, v in flows.items() if k != "Lung"), flows["Lung"], rel_tol=1e-6):
        raise ValueError("Arterial organ flows do not balance cardiac flow")
    result["circulation"] = {"portal_contributors": list(PORTAL), "portal_blood_state_required": True,
        "hepatic_arterial_flow_l_h": 60 * flows["Liver"], "portal_flow_l_h": 60 * portal,
        "liver_total_flow_l_h": 60 * (flows["Liver"] + portal), "cardiac_flow_l_h": 60 * flows["Lung"],
        "liver_flow_field_definition": "direct hepatic arterial inflow; portal inflow is separate",
        "topology": "venous -> lung -> arterial; portal tributaries -> portal blood -> liver -> venous; other systemic organs -> venous"}
    weights = {k: values[p] for k, p in plan["peripheral_weight_paths"].items()}
    if any(v < 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1., rel_tol=1e-8):
        raise ValueError("Native peripheral observation weights do not sum to one")
    result["peripheral_venous_weights"] = weights
    result["peripheral_venous_definition"] = "native weights multiply arterial/organ vascular plasma concentrations; reduced equilibrium observer requires validation"
    clearances = []
    for process in processes:
        stem = process["stem"]
        factors = {"ml/min/kg": .001, "l/h/kg": 1 / 60, "ml/h/kg": .001 / 60}
        per_kg = values[stem + "|Plasma clearance"]
        factor = factors[process["clearance_unit"]]
        reference_weight = values[stem + "|Body weight"]
        clearances.append({"organ": process["organ"], "input_value": per_kg, "input_unit": process["clearance_unit"],
            "reference_body_weight_kg": reference_weight, "input_at_reference_weight_l_min": per_kg * factor * reference_weight,
            "input_at_subject_weight_l_min": per_kg * factor * result["weight_kg"], "stem": stem,
            "native_inversion_audit": clearance._capacity(process, values)})
    result["clearance"] = {"processes": clearances,
        "fixed_renal_input_l_min": sum(x["input_at_subject_weight_l_min"] for x in clearances if x["organ"] == "renal"),
        "nominal_total_input_l_min": sum(x["input_at_subject_weight_l_min"] for x in clearances),
        "definition": "source per-kg plasma clearance inputs times actual subject weight; not model dose/AUC or independent systemic-CL validation"}
    result["excluded_organ_containers"] = plan["excluded_organ_containers"]
    result["capacity_definition"] = "sum(V_plasma + V_RBC*K_RBC/plasma + V_interstitial*K_int/plasma + V_cell*K_cell/plasma), each compartment once"
    result["scientific_accepted"] = False
    return result


def _check_native(native, request, request_path):
    if (native.get("version") != VERSION or native.get("solver_calls") != 0
            or native.get("model_modified") is not False
            or native.get("pkml_sha256") != request["pkml_sha256"]
            or native.get("request_sha256") != source._sha(request_path)):
        raise ValueError("Read-only export contract differs")


def export_physiology(*, execution_manifest_path, source_manifest_sha256, output_dir, project_root):
    from pbpk_kb.blueprints.design_runtime import run_bounded_r
    root = Path(project_root).resolve(strict=True)
    manifest, snapshot_path, pkml, compound = source._bind(execution_manifest_path, source_manifest_sha256, root)
    snapshot = json.loads(snapshot_path.read_text())
    plan = discover_plan(pkml, compound)
    audit_parameters, processes = clearance._plan(snapshot)
    specs = {r["path"]: r for r in plan["parameters"]}
    for spec in audit_parameters:
        if spec["path"] in specs and specs[spec["path"]]["unit"] != spec["unit"]:
            raise ValueError("Clearance readback unit conflict")
        specs[spec["path"]] = {"path": spec["path"], "unit": spec["unit"]}
    plan["parameters"] = list(specs.values())
    output = source._inside(output_dir, root, exists=False)
    pins = {str(p.relative_to(root)): source._sha(p) for p in (
        manifest, snapshot_path, pkml, root / WORKER, Path(__file__).resolve(),
        Path(source.__file__).resolve(), Path(clearance.__file__).resolve())}
    request = {"version": VERSION, "pkml_path": str(pkml), "pkml_sha256": source._sha(pkml),
        "parameters": plan["parameters"], "result_path": str(output / "native_readback.json")}
    request_path = output / "request.json"
    path = output / "perfusion_physiology.json"
    if output.exists():
        cached = json.loads(path.read_text())
        if (cached.get("source_pins") != pins or cached.get("source_execution_manifest_sha256") != source_manifest_sha256
                or json.loads(request_path.read_text()) != request):
            raise ValueError("Existing physiology cache belongs to different source/code")
        for name, digest in cached.get("artifact_pins", {}).items():
            artifact = source._inside(name, root)
            if artifact.parent != output or source._sha(artifact) != digest:
                raise ValueError("Cached native readback artifact changed")
        if set(cached.get("artifact_pins", {})) != {str(p.relative_to(root)) for p in (request_path, output / "native_readback.json")}:
            raise ValueError("Cached readback hash coverage missing")
        native = json.loads((output / "native_readback.json").read_text())
        _check_native(native, request, request_path)
        reconstructed = assemble_export(plan, native["parameters"], processes)
        if any(cached.get(key) != value for key, value in reconstructed.items()):
            raise ValueError("Cached physiology differs from native readback reconstruction")
        return {**cached, "manifest": {"path": str(path.relative_to(root)), "sha256": source._sha(path)}}
    output.mkdir(parents=True, exist_ok=False)
    source._write(request_path, request)
    runtime = run_bounded_r(script=root / WORKER, args=[str(request_path), source._sha(request_path)], output_dir=output,
        project_root=root, timeout_seconds=180, cpu_threads=2, max_output_bytes=128 * 1024**2)
    if runtime.get("status") != "completed" or runtime.get("returncode") != 0:
        raise ValueError("Bounded native export failed; inspect retained process log")
    native = json.loads((output / "native_readback.json").read_text())
    _check_native(native, request, request_path)
    if any(source._sha(root / p) != digest for p, digest in pins.items()):
        raise ValueError("Source or exporter changed during readback")
    result = {"version": VERSION, "compound": compound, "source_kind": "fresh_evidence_native_model",
              "source_pins": pins, "source_execution_manifest_sha256": source_manifest_sha256,
              "solver_calls": 0, "observations_used": False, "runtime": runtime,
              "artifact_pins": {str(p.relative_to(root)): source._sha(p) for p in (request_path, output / "native_readback.json")},
              **assemble_export(plan, native["parameters"], processes)}
    source._write(path, result)
    return {**result, "manifest": {"path": str(path.relative_to(root)), "sha256": source._sha(path)}}
