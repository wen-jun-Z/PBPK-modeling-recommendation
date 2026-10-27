"""Read PK descriptors from complete native models and already exported curves.

No ODE, fitting, observed data, Kp prediction, or source-model mutation. Native
NCA is scenario-specific; physchem and NCA distribution volumes are distinct.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3
from xml.etree import ElementTree as ET

from .parameter_batch import _quantity_path
from .source_workflow import inside, read_curve, sha256

VERSION = "source-pk-profile-v1"
_BASE = "artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173"
_DATABASE = _BASE + "/src/Db/PKSimDB.sqlite"
_DATABASE_SHA = "aec61433ddddf39ec5c832bda90a6977a25ea5427e77edc08f2d97a751fe1469"
_SOURCE_FILES = [
    _BASE + "/src/PKSim.Core/Services/VSSCalculator.cs",
    _BASE + "/src/PKSim.Core/Services/PKAnalysesTask.cs",
    _BASE + "/src/PKSim.Core/Model/OrganType.cs",
    _BASE + "/src/PKSim.Infrastructure/ORM/Repositories/OrganTypeRepository.cs",
    "artifacts/pbpk-model-assets/template_sources/ospsuite_r_v12.4.3/R/utilities-pk-analysis.R",
    "artifacts/pbpk-model-assets/template_sources/ospsuite_r_v12.4.3/R/utilities-simulation-results.R",
]


def _run_bounded_r(**kwargs):
    from .design_runtime import run_bounded_r
    return run_bounded_r(**kwargs)


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _bind(snapshot, execution_manifest, target_output, output_dir, root):
    target_output = _quantity_path(target_output, "target output")
    directory = inside(output_dir, root)
    if directory.exists():
        raise FileExistsError("PK profile output already exists")
    if isinstance(execution_manifest, dict):
        path = inside(execution_manifest.get("manifest_path", ""), root, exists=True)
        manifest = json.loads(path.read_text())
        if manifest != execution_manifest:
            raise ValueError("Execution manifest differs from persisted manifest")
    else:
        path = inside(execution_manifest, root, exists=True)
        manifest = json.loads(path.read_text())
    if not manifest.get("engineering_accepted") or not manifest.get("source_unchanged"):
        raise ValueError("A successful immutable native execution is required")
    source = inside(path.parent / "snapshot.json", root, exists=True)
    if sha256(source) != manifest.get("source_sha256") or json.loads(source.read_text()) != snapshot:
        raise ValueError("Snapshot differs from executed source")
    artifacts = []
    for field, suffix in (("pkml_outputs", ".pkml"), ("curve_outputs", ".csv")):
        items = manifest.get(field)
        if not isinstance(items, list) or len(items) != 1:
            raise ValueError("Exactly one complete PKML and one native CSV are required")
        artifact = inside(path.parent / items[0]["path"], root, exists=True)
        if (not artifact.is_relative_to(path.parent) or artifact.suffix.lower() != suffix
                or sha256(artifact) != items[0].get("sha256")):
            raise ValueError("Native artifact path or checksum changed")
        artifacts.append(artifact)
    simulations = snapshot.get("Simulations", [])
    if snapshot.get("Version") != 80 or len(simulations) != 1 or len(simulations[0].get("Compounds", [])) != 1:
        raise ValueError("Profile requires one Version=80 simulation and selected drug")
    compound = simulations[0]["Compounds"][0]["Name"]
    expected = f"Organism|PeripheralVenousBlood|{compound}|Plasma (Peripheral Venous Blood)"
    if target_output != expected:
        raise ValueError("Only registered peripheral venous plasma can define systemic PK descriptors")
    curve = read_curve(path.parent, manifest, target_output)
    return directory, path, manifest, artifacts, compound, curve


def _physchem_plan(pkml, compound, root):
    """Map the pinned VSSCalculator sum to exact native quantities, not Kp formulae."""
    database = inside(_DATABASE, root, exists=True)
    if sha256(database) != _DATABASE_SHA:
        raise ValueError("Pinned physiology database changed")
    with sqlite3.connect("file:" + str(database) + "?mode=ro", uri=True) as connection:
        types = dict(connection.execute("SELECT organ_name,organ_type FROM tab_organ_types WHERE container_type='ORGAN'"))
    model = ET.parse(pkml).getroot().find("Simulation/Model/Root")
    if model is None:
        raise ValueError("Native model root is absent")
    paths, ids, parents, elements = {}, {}, {}, []

    def walk(node, path):
        for child in node.findall("./Children/*"):
            name = child.get("name")
            if not name:
                continue
            child_path = path + (name,)
            if child_path in paths:
                raise ValueError("Native model paths are ambiguous")
            paths[child_path] = child
            if child.get("id"):
                ids[child.get("id")] = child_path
            parents[id(child)] = node
            elements.append((child_path, child))
            walk(child, child_path)
    walk(model, ())
    parameters = [{"path": "Organism|Weight", "unit": "kg"},
                  {"path": "Organism|Hematocrit", "unit": "fraction"},
                  {"path": compound + "|Fraction unbound (plasma)", "unit": "fraction"}]
    for path, element in elements:
        if (path[0] == "Events" and len(path) >= 4 and path[-2] == "ProtocolSchemaItem"
                and path[-1] in {"Dose", "Start time", "Infusion time"}):
            parameters.append({"path": "|".join(path), "unit": "mg" if path[-1] == "Dose" else "min"})
    partition = {}
    for path, element in elements:
        if len(path) < 4 or path[-2:] != (compound, "Partition coefficient (intracellular/plasma)"):
            continue
        ancestor = parents[id(element)]
        while ancestor is not model and ancestor.tag != "Neighborhood":
            ancestor = parents.get(id(ancestor), model)
        if ancestor.tag != "Neighborhood":
            raise ValueError("Partition coefficient lacks a native neighborhood")
        neighbor = ids.get(ancestor.get("firstNeighbor"))
        if not neighbor or len(neighbor) < 3:
            raise ValueError("Native first-neighbor binding is absent")
        # VSSCalculator uses FirstNeighbor.ParentContainer.Name, not name parsing.
        partition.setdefault(neighbor[-2], []).append("|".join(path))
    organs, excluded = [], []
    for path, element in elements:
        if len(path) != 2 or path[0] != "Organism" or element.get("containerType") != "Organ":
            continue
        name = path[1]
        kind = types.get(name, "Unknown")
        if kind not in {"TissueOrgansNotInGiTract", "GiTractOrgans", "VascularSystem"}:
            excluded.append({"organ": name, "database_type": kind})
            continue
        volume = "|".join(path + ("Volume",))
        parameters.append({"path": volume, "unit": "l"})
        row = {"organ": name, "type": kind, "volume_path": volume}
        if kind != "VascularSystem":
            fraction = "|".join(path + ("Fraction vascular",))
            parameters.append({"path": fraction, "unit": "fraction"})
            choices = partition.get(name, [])
            if len(choices) > 1:
                raise ValueError("Ambiguous organ partition coefficient; do not silently choose one")
            row.update(fraction_vascular_path=fraction, partition_path=choices[0] if choices else None,
                       absent_partition_policy="zero_as_in_pinned_VSSCalculator" if not choices else None)
            if choices:
                parameters.append({"path": choices[0], "unit": "fraction"})
        organs.append(row)
    if not organs or not any(x["type"] == "VascularSystem" for x in organs):
        raise ValueError("Native organ mapping is incomplete")
    return {"organs": organs, "excluded_organs": excluded, "parameters": parameters,
            "calculator": "sum_tissue((native_Kcell_plasma+fraction_vascular*(1-Hct))*V)+sum_blood(V*(1-Hct))",
            "interpretation": "Pinned source sum using executed native physiology; not a direct call to PKSim VSSCalculator"}


def _context(snapshot, curve):
    from .source_output_grid import _source_applications
    from .source_adaptation import _named
    simulation = snapshot["Simulations"][0]
    selection = simulation["Compounds"][0]
    protocol = _named(snapshot["Protocols"], selection["Protocol"]["Name"], "Protocols")
    applications = _source_applications(protocol)
    routes = [item["ApplicationType"] for item, _ in applications]
    individual = _named(snapshot["Individuals"], simulation["Individual"], "Individuals")
    origin = individual.get("OriginData", {})
    species = str(origin.get("Species", "unknown")).lower()
    sex = str(origin.get("Gender", "unknown")).lower()
    age = origin.get("Age", {})
    age_years = (age.get("Value") if age.get("Unit") in {"year(s)", "year", "years"}
                 and _number(age.get("Value")) and age["Value"] >= 0 else None)
    selected_processes = selection.get("Processes", {})
    drug = _named(snapshot["Compounds"], selection["Name"], "Compounds")
    names = {row.get("Name") for row in selected_processes if isinstance(row, dict)}
    process_types = [row.get("InternalName") for row in drug.get("Processes", [])
                     if str(row.get("Molecule")) + "-" + str(row.get("DataSource")) in names]
    nonlinear = [name for name in process_types if "_MM" in str(name) or "Binding" in str(name)]
    return {"compound": selection["Name"], "compound_name": selection["Name"],
            "species": species, "sex": sex if sex in {"male", "female"} else "unknown",
            "age_years": age_years, "population": "unknown", "source_population": origin.get("Population"),
            "population_reason": "Source population identifies physiology defaults; clinical healthy/disease status is not established",
            "matrix": "plasma",
            "route": "iv" if set(routes) <= {"Intravenous", "IntravenousBolus"} else "oral" if set(routes) == {"Oral"} else "mixed",
            "dosing": "single" if len(routes) == 1 else "repeated",
            "dose_mg": None, "infusion_minutes": None, "weight_kg": None,
            "individual": individual["Name"], "nonlinear_selected_processes": nonlinear,
            "human_origin_data": individual.get("OriginData"), "protocol": protocol["Name"],
            "application_count": len(applications), "routes": routes,
            "application_start_times_min": [start for _, start in applications],
            "single_iv": len(routes) == 1 and routes[0] in {"Intravenous", "IntravenousBolus"},
            "single_oral": routes == ["Oral"], "interaction_count": len(simulation.get("Interactions", [])),
            "selected_processes": selected_processes,
            "curve_start_min": curve["time_min"][0], "curve_end_min": curve["time_min"][-1],
            "curve_point_count": len(curve["time_min"]),
            "scope": "Current individual, dose, formulation and mechanism; not a transferable dose-independent constant"}


def _metric(endpoint, definition, value, unit, reason, *, eligible=False, provenance=None):
    return {"endpoint": endpoint, "definition": definition, "value": value, "unit": unit,
            "status": "available" if value is not None else "blocked", "eligible_for_constraint": eligible,
            "reason": reason, "provenance": provenance or {}}


def _assess(native, request, context, plan):
    if (native.get("version") != VERSION or native.get("request_sha256") != request["request_sha256"]
            or any(native.get(k) != request[k] for k in ("pkml_sha256", "csv_sha256", "target_output"))
            or type(native.get("solver_calls")) is not int or native["solver_calls"] != 0
            or native.get("model_modified") is not False):
        raise ValueError("Native profile identity/read-only contract mismatch")
    rows = native.get("parameters", [])
    if len(rows) != len(request["parameters"]) or any(row.get("path") != spec["path"] or row.get("requested_unit") != spec["unit"]
            for row, spec in zip(rows, request["parameters"], strict=True)):
        raise ValueError("Native parameter readback does not cover exact requested paths")
    values = {r["path"]: r for r in rows}

    def value(path):
        row = values[path]
        if row.get("status") != "completed" or not _number(row.get("value")):
            raise ValueError("Unavailable native parameter: " + path)
        return row["value"]
    try:
        body_weight = value("Organism|Weight")
        if body_weight <= 0:
            raise ValueError("Invalid body weight")
        context["weight_kg"] = body_weight
    except ValueError:
        body_weight = None
    events = {}
    for path, row in values.items():
        if path.startswith("Events|") and "|ProtocolSchemaItem|" in path:
            try:
                amount = value(path)
                if amount < 0:
                    raise ValueError("Negative native administration quantity")
                events.setdefault(path.rsplit("|", 1)[0], {})[path.rsplit("|", 1)[1]] = amount
            except ValueError:
                events.setdefault(path.rsplit("|", 1)[0], {})["error"] = "Native administration readback failed"
    context["native_administrations"] = [{"path": path, **row} for path, row in sorted(events.items())]
    context["administration_readback_verified"] = (len(events) == context["application_count"]
        and all("Dose" in row and "Start time" in row and "error" not in row for row in events.values()))
    if context["application_count"] == 1 and context["administration_readback_verified"]:
        event = next(iter(events.values()))
        context["dose_mg"] = event["Dose"]
        context["infusion_minutes"] = event.get("Infusion time", 0. if context["routes"] == ["IntravenousBolus"] else None)
        context["actual_start_min"] = event["Start time"]
    metrics = []
    fu_path = context["compound"] + "|Fraction unbound (plasma)"
    try:
        fu = value(fu_path)
        if not 0 < fu <= 1:
            raise ValueError("Native plasma unbound fraction is outside (0,1]")
        metrics.append(_metric("fu", "fu_plasma", fu, "fraction", "Actual native plasma binding, not source reference fu",
                               eligible=True, provenance=values[fu_path]))
    except ValueError as error:
        metrics.append(_metric("fu", "fu_plasma", None, "fraction", str(error)))
    try:
        hct = value("Organism|Hematocrit")
        if not body_weight or not 0 <= hct < 1:
            raise ValueError("Invalid native body weight or hematocrit")
        contributions = []
        for organ in plan["organs"]:
            volume = value(organ["volume_path"])
            if volume < 0:
                raise ValueError("Negative organ volume")
            if organ["type"] == "VascularSystem":
                term = volume * (1 - hct)
            else:
                fraction = value(organ["fraction_vascular_path"])
                coefficient = value(organ["partition_path"]) if organ["partition_path"] else 0.
                if not 0 <= fraction <= 1 or coefficient < 0:
                    raise ValueError("Invalid native partition or vascular fraction")
                term = volume * (coefficient + fraction * (1 - hct))
            contributions.append({**organ, "contribution_l": term})
        vss = sum(x["contribution_l"] for x in contributions)
        item = _metric("vdss", "vdss_physchem", vss, "L",
            "Diagnostic native partition/volume sum; not NCA Vss or a clinical constraint",
            provenance={"value_l_per_kg": vss / body_weight, "body_weight_kg": body_weight,
                        "contributions": contributions, "excluded_organs": plan["excluded_organs"],
                        "method": plan["interpretation"], "formula": plan["calculator"]})
        missing = [row["organ"] for row in contributions if row.get("absent_partition_policy")]
        item["missing_organs"] = missing
        if missing:
            item.update(status="incomplete", reason="Diagnostic partial source-formula sum; organ partition mapping absent for "
                        + ", ".join(missing) + "; PKML versus PKSim-object equivalence is unverified")
        metrics.append(item)
    except (ValueError, KeyError) as error:
        metrics.append(_metric("vdss", "vdss_physchem", None, "L", str(error)))
    nca = native.get("nca", {})
    nca_rows = nca.get("parameters", [])
    if len({row["name"] for row in nca_rows}) != len(nca_rows):
        raise ValueError("Native NCA parameter names are ambiguous")
    parameters = {row["name"]: row for row in nca_rows}

    def pk(name):
        row = parameters.get(name, {})
        vals = row.get("values", [])
        return vals[0] if len(vals) == 1 and _number(vals[0]) else None
    tail = pk("FractionAucLastToInf")
    if tail is None:
        tail = pk("FractionAucEndToInf")
    quality = {"auc_extrapolated_fraction": tail, "tail_screen_threshold": .2,
               "tail_screen_interpretation": "20% is an engineering screening rule, not a universal scientific acceptance criterion",
               "auc_infinity": parameters.get("AUC_inf"), "auc_observed_window": parameters.get("AUC_tEnd"),
               "mrt": parameters.get("MRT"), "half_life": parameters.get("Thalf"),
               "aumc_tail_error": "Not returned by native API; AUC tail fraction does not bound MRT/Vss extrapolation error",
               "native_nca_status": nca.get("status"), "native_nca_error": nca.get("error")}
    for endpoint, name, true_definition, apparent_definition in (
            ("cl", "CL", "cl_systemic_iv", "cl_over_f"), ("vdss", "Vss", "vdss_iv", "vdss_over_f")):
        definition = true_definition if context["single_iv"] else apparent_definition if context["single_oral"] else true_definition
        raw = pk(name)
        row = parameters.get(name, {})
        output_unit = "L/h" if endpoint == "cl" else "L"
        reasons = []
        if not context["single_iv"] and not context["single_oral"]:
            reasons.append("Repeated/mixed/non-IV protocol cannot define a single-IV systemic descriptor")
            raw = None
        unit = row.get("unit")
        converted = None
        if raw is not None and raw > 0 and body_weight:
            # Native PK parameter base units are explicitly retained below.
            factors = ({"l/min/kg": body_weight * 60, "ml/min/kg": body_weight * .06,
                        "l/h/kg": body_weight, "l/min": 60., "ml/min": .06, "l/h": 1.}
                       if endpoint == "cl" else
                       {"l/kg": body_weight, "ml/kg": body_weight / 1000, "l": 1., "ml": .001})
            if unit in factors:
                converted = raw * factors[unit]
            else:
                reasons.append("Unsupported native NCA unit: " + str(unit))
        if converted is None:
            reasons.append("Native PK quantity unavailable or invalid")
        if tail is None or not 0 <= tail <= .2:
            reasons.append("Terminal extrapolation fraction absent or fails declared engineering screen")
        if context["interaction_count"]:
            reasons.append("Source interactions/induction make NCA scenario-specific; no automatic constraint eligibility")
        if not context["single_iv"]:
            reasons.append("Apparent oral quantity depends on unknown bioavailability and absorption; not systemic CL/Vss")
        if context["curve_start_min"] > min(context["application_start_times_min"]):
            reasons.append("Export starts after first administration")
        if not context["administration_readback_verified"] or not context.get("dose_mg"):
            reasons.append("Single administered native dose not verified")
        # Even a passing IV descriptor requires same-dose/context comparison.
        metrics.append(_metric(endpoint, definition, converted, output_unit,
            "; ".join(reasons) or "Single-IV scenario-equivalent native NCA; compare only matching dose and context",
            eligible=converted is not None and not reasons,
            provenance={"native": row, "body_weight_kg": body_weight, "quality": quality,
                        "method": "OSPSuite calculatePKAnalyses(importResultsFromCSV); no solver calls",
                        "constraint_scope": "same_dose_same_context_iv_only",
                        "nonlinear_process_warning": context["nonlinear_selected_processes"],
                        "dose_independent_constant": False}))
    return metrics, quality


def extract_source_pk_profile(snapshot, execution_manifest, target_output, output_dir, project_root):
    """Return native fu, source-formula physchem Vss and qualified native NCA.

    Existing snapshot, PKML and CSV are checksum-bound. The result distinguishes
    true single-IV descriptors from apparent oral and repeated-dose quantities.
    """
    root = Path(project_root).resolve(strict=True)
    directory, manifest_path, manifest, artifacts, compound, curve = _bind(
        snapshot, execution_manifest, target_output, output_dir, root)
    context = _context(snapshot, curve)
    plan = _physchem_plan(artifacts[0], compound, root)
    sources = [{"path": path, "sha256": sha256(inside(path, root, exists=True))} for path in _SOURCE_FILES]
    directory.mkdir(parents=True)
    request = {"version": VERSION, "pkml_path": str(artifacts[0]), "pkml_sha256": sha256(artifacts[0]),
               "csv_path": str(artifacts[1]), "csv_sha256": sha256(artifacts[1]), "target_output": target_output,
               "parameters": plan["parameters"], "result_path": str(directory / "native_profile.json")}
    request_path = directory / "request.json"
    _write(request_path, request)
    request_sha = sha256(request_path)
    runtime = _run_bounded_r(script=root / "src/pbpk_kb/model_runtime/R/source_pk_profile.R",
        args=[str(request_path), request_sha], output_dir=directory, project_root=root,
        timeout_seconds=180, cpu_threads=2, max_output_bytes=64 * 1024**2)
    result = {"version": VERSION, "status": "failed", "metrics": [], "context": context,
              "solver_calls": 0, "model_modified": False, "runtime": runtime,
              "provenance": {"execution_manifest": str(manifest_path), "execution_manifest_sha256": sha256(manifest_path),
                  "pkml_sha256": request["pkml_sha256"], "csv_sha256": request["csv_sha256"],
                  "request_sha256": request_sha, "source_definitions": sources,
                  "database_sha256": _DATABASE_SHA, "physchem_plan": plan,
                  "runtime_version_binding": manifest.get("runtime_binding", {})}}
    try:
        if runtime.get("status") != "completed" or runtime.get("returncode") != 0:
            raise ValueError("Native profile reader failed")
        if sha256(artifacts[0]) != request["pkml_sha256"] or sha256(artifacts[1]) != request["csv_sha256"]:
            raise ValueError("Native artifacts changed during PK extraction")
        native = json.loads((directory / "native_profile.json").read_text())
        metrics, quality = _assess(native, {**request, "request_sha256": request_sha}, context, plan)
        result.update(status="completed", metrics=metrics, nca_quality=quality,
                      ospsuite_version=native.get("ospsuite_version"))
    except (ValueError, OSError, KeyError, TypeError) as error:
        result["error"] = str(error)
    _write(directory / "source_pk_profile.json", result)
    return result
