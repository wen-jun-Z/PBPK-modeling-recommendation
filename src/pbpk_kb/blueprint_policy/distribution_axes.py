"""Audited native distribution perturbations of a fresh evidence-built IV model.

Two tied axes preserve the anatomical model: tissue Kcell/plasma and symmetric
cellular exchange. These are construction hypotheses, not identified biological
parameters or clinical Vss. No observations, fitting, or endpoint selection.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from defusedxml import ElementTree as ET

VERSION = "native-distribution-axes-1"
AXES = ("partition_scale", "bidirectional_exchange_scale")
PARTITION = "Partition coefficient (intracellular/plasma)"
FACTORS = ("P_int_cell_factor", "P_cell_int_factor")
TISSUES = tuple(("Organism", organ) for organ in (
    "Bone", "Brain", "Fat", "Gonads", "Heart", "Kidney", "Stomach",
    "SmallIntestine", "LargeIntestine", "Lung", "Muscle", "Pancreas", "Skin", "Spleen",
)) + (("Organism", "Liver", "Periportal"), ("Organism", "Liver", "Pericentral"))
SCOPE = "16 nonmucosal tissue interstitial/intracellular neighborhoods; GI mucosa, lumen, saliva, blood-cell and endothelial interfaces excluded"


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inside(path, root, *, exists=True):
    path = Path(path)
    path = (path if path.is_absolute() else root / path).resolve(strict=exists)
    if path == root or not path.is_relative_to(root):
        raise ValueError("Distribution paths must remain inside the project")
    return path


def _write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _positive(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("Distribution scales must be finite and positive")
    return float(value)


def validate_candidates(candidates):
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 8:
        raise ValueError("Require 1..8 explicit distribution candidates")
    result = []
    for item in candidates:
        if not isinstance(item, dict) or set(item) != set(AXES):
            raise ValueError("Each candidate requires exactly the two distribution scales")
        result.append({name: _positive(item[name]) for name in AXES})
    return result


def discover_plan(pkml_path, compound):
    """Bind tissue interfaces by resolved neighbor IDs, never path wildcards."""
    if not isinstance(compound, str) or not compound or any(c in compound for c in "|*?\n\r"):
        raise ValueError("Compound must be an exact native path component")
    model = ET.parse(pkml_path).getroot().find("Simulation/Model/Root")
    if model is None:
        raise ValueError("Missing native model root")
    paths, ids = {}, {}

    def walk(node, prefix=()):
        for child in node.findall("./Children/*"):
            name = child.get("name")
            if not name or any(c in name for c in "|\n\r"):
                raise ValueError("Invalid native quantity name")
            path = prefix + (name,)
            if path in paths:
                raise ValueError("Duplicate native path")
            paths[path] = child
            identifier = child.get("id")
            if identifier:
                if identifier in ids:
                    raise ValueError("Duplicate native model ID")
                ids[identifier] = path
            walk(child, path)
    walk(model)
    selected, changed = {}, []
    for path, node in paths.items():
        if node.tag != "Neighborhood":
            continue
        first, second = ids.get(node.get("firstNeighbor")), ids.get(node.get("secondNeighbor"))
        if not first or not second:
            raise ValueError("Unresolved native neighborhood binding")
        if first[:-1] not in TISSUES or first[-1] != "Interstitial":
            continue
        if second != first[:-1] + ("Intracellular",):
            raise ValueError("Tissue neighborhood has inconsistent intracellular binding")
        tissue = first[:-1]
        if tissue in selected:
            raise ValueError("Duplicate tissue exchange neighborhood")
        selected[tissue] = path
    if set(selected) != set(TISSUES):
        raise ValueError("Expected exactly the 16 reviewed tissue interfaces")
    for tissue in TISSUES:
        path = selected[tissue]
        for leaf, axis in ((PARTITION, AXES[0]), *((name, AXES[1]) for name in FACTORS)):
            key = path + (compound, leaf)
            node = paths.get(key)
            if node is None or node.tag != "Parameter" or node.get("dim", "") not in ("", "Dimensionless", "Fraction"):
                raise ValueError("Missing or dimensional distribution quantity: " + "|".join(key))
            changed.append({"path": "|".join(key), "unit": "fraction", "axis": axis,
                            "tissue": "|".join(tissue)})
    protected = []
    for path, node in paths.items():
        if node.tag != "Parameter":
            continue
        leaf = path[-1]
        physiology = (path[0] == "Organism" and leaf in {
            "Volume", "Blood flow rate", "Weight", "Age", "Hematocrit", "Plasma protein scale factor"
        }) or leaf.startswith("Surface area")
        physchem = path[0] == compound and (leaf in {
            "Lipophilicity", "Molecular weight", "Fraction unbound (plasma)",
            "Fraction unbound (plasma, reference value)", "Permeability"
        } or leaf.startswith("pKa value "))
        clearance = leaf in {"Plasma clearance", "Specific clearance", "Fraction unbound (experiment)", "Lipophilicity (experiment)"}
        if physiology or physchem or clearance:
            protected.append({"path": "|".join(path), "dimension": node.get("dim", "")})
    required = {compound + "|" + name for name in (
        "Lipophilicity", "Fraction unbound (plasma)", "Fraction unbound (plasma, reference value)", "Permeability")}
    if not required.issubset({p["path"] for p in protected}):
        raise ValueError("Protected native drug inputs are incomplete")
    return {"scope": SCOPE, "compound": compound, "changed": changed, "protected": protected}


def _bind(execution_manifest_path, source_manifest_sha256, root):
    manifest_path = _inside(execution_manifest_path, root)
    if _sha(manifest_path) != source_manifest_sha256:
        raise ValueError("Source execution manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("engineering_accepted") or not manifest.get("source_unchanged"):
        raise ValueError("Successful unchanged source execution required")
    snapshot_path = _inside(manifest_path.parent / "snapshot.json", root)
    if _sha(snapshot_path) != manifest.get("source_sha256"):
        raise ValueError("Source snapshot hash mismatch")
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get("Version") != 80 or len(snapshot.get("Compounds", [])) != 1 or len(snapshot.get("Simulations", [])) != 1:
        raise ValueError("One evidence-built drug/simulation required")
    compound = snapshot["Compounds"][0]
    if ([x.get("Name") for x in compound.get("Lipophilicity", [])] != ["EvidenceLogP"]
            or [x.get("Name") for x in compound.get("FractionUnbound", [])] != ["EvidenceFu"]
            or any(p.get("DataSource") != "EvidenceAssembly" for p in compound.get("Processes", []))):
        raise ValueError("Fresh evidence assembly required; calibrated drug models are excluded")
    protocols = snapshot.get("Protocols", [])
    if len(protocols) != 1 or protocols[0].get("ApplicationType") not in {"Intravenous", "IntravenousBolus"}:
        raise ValueError("This distribution extension is single-IV only")
    if protocols[0].get("DosingInterval") != "Single":
        raise ValueError("This distribution extension requires single dosing")
    artifacts = manifest.get("pkml_outputs", [])
    if len(artifacts) != 1:
        raise ValueError("One native PKML required")
    pkml = _inside(manifest_path.parent / artifacts[0]["path"], root)
    if not pkml.is_relative_to(manifest_path.parent) or _sha(pkml) != artifacts[0]["sha256"]:
        raise ValueError("Native source PKML path/hash mismatch")
    return manifest_path, snapshot_path, pkml, compound["Name"]


def _validate_readback(rows, specs, candidate, *, changed):
    if not isinstance(rows, list) or [r.get("path") for r in rows] != [r["path"] for r in specs]:
        raise ValueError("Native readback paths differ from the locked plan")
    for row, spec in zip(rows, specs):
        before, after = row.get("before"), row.get("after")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (before, after)):
            raise ValueError("Readback contains nonfinite native values")
        expected = before * candidate[spec["axis"]] if changed else before
        if changed and (before < 0 or after < 0 or row.get("unit") not in ("", "1", "fraction", "dimensionless")):
            raise ValueError("Distribution values must be nonnegative and dimensionless")
        if not math.isclose(after, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Native changed/protected parameter readback mismatch")


def run_distribution_axes(*, execution_manifest_path, source_manifest_sha256, candidates,
                          output_dir, project_root, timeout_seconds=600, cpu_threads=2,
                          max_output_bytes=128 * 1024**2):
    """Run 1..8 tied-axis candidates, each from the same fresh source PKML.

    No physiology/clearance paths are varied. Native derived systemic clearance
    may still change through distribution; unchanged CL inputs do not prove
    unchanged systemic CL. Source artifact integrity does not prove provenance.
    """
    from pbpk_kb.blueprints.design_runtime import run_bounded_r
    root = Path(project_root).resolve(strict=True)
    values = validate_candidates(candidates)
    manifest, snapshot, pkml, drug = _bind(execution_manifest_path, source_manifest_sha256, root)
    plan = discover_plan(pkml, drug)
    destination = _inside(output_dir, root, exists=False)
    if destination.exists():
        raise FileExistsError("Use a new immutable distribution output directory")
    target = f"Organism|PeripheralVenousBlood|{drug}|Plasma (Peripheral Venous Blood)"
    pins = {str(p): _sha(p) for p in (manifest, snapshot, pkml, Path(__file__),
        root / "src/pbpk_kb/model_runtime/R/blueprint_distribution_axes.R")}
    request = {"version": VERSION, "pkml_path": str(pkml), "pkml_sha256": _sha(pkml),
        "source_manifest_sha256": source_manifest_sha256, "target_output": target,
        "plan": plan, "candidates": values, "native_cpu_cores": cpu_threads,
        "output_dir": str(destination), "result_path": str(destination / "native_result.json")}
    destination.mkdir(parents=True)
    request_path = destination / "request.json"
    _write(request_path, request)
    result = {"version": VERSION, "status": "failed", "scope": SCOPE,
        "source_pins": pins, "candidates": values, "observations_used": False,
        "scientific_accepted": False, "mass_balance_checked": False, "runs": [],
        "limitations": ["Tied distribution hypotheses; no parameter identifiability or accuracy claim",
            "Cellular partition scale is not a clinical Vss multiplier",
            "GI mucosal and other excluded interfaces retain their native values",
            "Fixed clearance inputs do not imply fixed model systemic clearance"]}
    runtime = run_bounded_r(script=root / "src/pbpk_kb/model_runtime/R/blueprint_distribution_axes.R",
        args=[str(request_path), _sha(request_path)], output_dir=destination, project_root=root,
        timeout_seconds=timeout_seconds, cpu_threads=cpu_threads, max_output_bytes=max_output_bytes)
    result["runtime"] = runtime
    if runtime.get("status") == "completed" and runtime.get("returncode") == 0:
        try:
            native = json.loads((destination / "native_result.json").read_text())
            if (native.get("request_sha256") != _sha(request_path) or native.get("pkml_sha256") != _sha(pkml)
                    or native.get("source_manifest_sha256") != source_manifest_sha256
                    or native.get("target_output") != target or len(native.get("runs", [])) != len(values)):
                raise ValueError("Native result identity differs from locked request")
            for index, (run, candidate) in enumerate(zip(native["runs"], values)):
                if run.get("candidate_index") != index or run.get("scales") != candidate:
                    raise ValueError("Native candidate identity differs")
                _validate_readback(run.get("changed"), plan["changed"], candidate, changed=True)
                _validate_readback(run.get("protected"), plan["protected"], candidate, changed=False)
                curve = run.get("curve", {})
                t, c = curve.get("time_min", []), curve.get("concentration", [])
                if (len(t) < 2 or len(t) != len(c) or curve.get("time_unit") != "min"
                        or not curve.get("unit") or any(not math.isfinite(v) for v in t + c)
                        or any(b <= a for a, b in zip(t, t[1:])) or min(c) < -1e-10):
                    raise ValueError("Native plasma curve invalid")
                exported = _inside(destination / run["pkml_file"], root)
                if exported.parent != destination or _sha(exported) != run.get("pkml_sha256"):
                    raise ValueError("Changed PKML hash/path mismatch")
            if any(_sha(Path(p)) != digest for p, digest in pins.items()):
                raise ValueError("Frozen source/code changed during execution")
            result.update(status="completed", runs=native["runs"], native_result_sha256=_sha(destination / "native_result.json"))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result["error"] = str(exc)
    else:
        result["error"] = "Native distribution worker failed; inspect execution.log"
    _write(destination / "distribution_manifest.json", result)
    return result
