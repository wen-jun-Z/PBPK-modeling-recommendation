"""One-solve native CL/partition/exchange trials from raw evidence construction.

The two distribution axes are unchanged. Only native hepatic plasma-clearance
inputs may additionally scale; native specific clearance remains a formula and
all net renal inputs remain fixed. No observations or fit decisions enter here.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from . import distribution_axes as distribution
from pbpk_kb.blueprints import evidence_native_audit as clearance_audit

VERSION = "native-three-axes-trial-1"
PARAMETERS = ("clearance_scale", *distribution.AXES)
WORKER = "src/pbpk_kb/model_runtime/R/blueprint_axes_trial.R"


def validate_scales(scales):
    if not isinstance(scales, dict) or set(scales) != set(PARAMETERS):
        raise ValueError("Trial requires exactly clearance, partition and bidirectional exchange scales")
    return {key: distribution._positive(scales[key]) for key in PARAMETERS}


def prepare_axes_source(*, execution_manifest_path, source_manifest_sha256, project_root):
    """Validate immutable raw source and return the exact trial/readback plan."""
    root = Path(project_root).resolve(strict=True)
    manifest, snapshot_path, pkml, compound = distribution._bind(
        execution_manifest_path, source_manifest_sha256, root)
    snapshot = json.loads(snapshot_path.read_text())
    plan = distribution.discover_plan(pkml, compound)
    audit_parameters, processes = clearance_audit._plan(snapshot)
    hepatic = [p for p in processes if p["organ"] == "hepatic"]
    changed_clearances = [{"path": p["stem"] + "|Plasma clearance", "unit": p["clearance_unit"],
        "axis": "clearance_scale"} for p in hepatic]
    changing = {row["path"] for row in changed_clearances}
    derived = {p["stem"] + "|Specific clearance" for p in hepatic}
    plan["protected"] = [p for p in plan["protected"] if p["path"] not in changing | derived]
    plan.update(clearance_changes=changed_clearances, clearance_parameters=audit_parameters,
        clearance_processes=processes,
        clearance_policy="nonrenal_only_with_fixed_net_renal" if any(p["organ"] == "renal" for p in processes) else "total_hepatic_input",
        clearance_axis_available=bool(hepatic))
    database = distribution._inside(clearance_audit._DATABASE, root)
    if distribution._sha(database) != clearance_audit._DATABASE_SHA:
        raise ValueError("Native clearance formula database changed")
    sources = (manifest, snapshot_path, pkml, database, Path(__file__).resolve(),
        Path(distribution.__file__).resolve(), Path(clearance_audit.__file__).resolve(), root / WORKER)
    return {"version": VERSION, "pkml_path": str(pkml), "pkml_sha256": distribution._sha(pkml),
        "source_manifest_sha256": source_manifest_sha256, "source_manifest_path": str(manifest),
        "source_snapshot_path": str(snapshot_path), "compound": compound,
        "target_output": f"Organism|PeripheralVenousBlood|{compound}|Plasma (Peripheral Venous Blood)",
        "plan": plan, "source_pins": {str(p): distribution._sha(p) for p in sources}}


def validate_clearance_readback(rows, plan, scales):
    """Independently reconstruct frozen native inversion/capacity from readback."""
    specs = plan["clearance_parameters"]
    if not isinstance(rows, list) or [r.get("path") for r in rows] != [r["path"] for r in specs]:
        raise ValueError("Clearance readback path binding changed")
    scaled = {p["path"] for p in plan["clearance_changes"]}
    derived = {p["stem"] + "|Specific clearance" for p in plan["clearance_processes"] if p["organ"] == "hepatic"}
    before_values, after_values = {}, {}
    for spec, row in zip(specs, rows):
        before, after = row.get("before"), row.get("after")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (before, after)):
            raise ValueError("Nonfinite native clearance readback")
        if row.get("unit") != spec["unit"]:
            raise ValueError("Clearance canonical unit changed")
        path = row["path"]
        if "expected" in spec and not math.isclose(before, spec["expected"], rel_tol=spec["relative_tolerance"], abs_tol=1e-12):
            raise ValueError("Native source differs from raw evidence input")
        if path not in derived:
            expected = before * scales["clearance_scale"] if path in scaled else before
            if not math.isclose(after, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Clearance mutation/protection readback mismatch")
        before_values[path], after_values[path] = before, after
    baseline = [clearance_audit._capacity(p, before_values) for p in plan["clearance_processes"]]
    changed = [clearance_audit._capacity(p, after_values) for p in plan["clearance_processes"]]
    return {"source": baseline, "trial": changed, "source_accepted": all(x["accepted"] for x in baseline),
        "accepted": all(x["accepted"] for x in changed)}


def _read_counter(folder, request_sha256=None):
    receipt = folder / "solver_receipt.json"
    if not receipt.exists():
        return 0
    record = json.loads(receipt.read_text())
    if request_sha256 is not None and record.get("request_sha256") != request_sha256:
        raise ValueError("Native solve counter belongs to another request")
    count = record.get("actual_native_solver_calls")
    if type(count) is not int or count not in (0, 1):
        raise ValueError("Invalid native solve counter")
    return count


def run_axes_trial(*, execution_manifest_path, source_manifest_sha256, scales,
                   output_dir, project_root, timeout_seconds=180, cpu_threads=2,
                   max_output_bytes=128 * 1024**2):
    """Return predicted/scientific_failed/infrastructure_failed plus solve count.

    One charged trial can use zero solves if capacity rejects it. A persisted
    receipt records entry to the single native solver call, including failed or
    interrupted solves. Source construction solves are counted by the caller.
    """
    from pbpk_kb.blueprints.design_runtime import run_bounded_r
    root = Path(project_root).resolve(strict=True)
    values = validate_scales(scales)
    source = prepare_axes_source(execution_manifest_path=execution_manifest_path,
        source_manifest_sha256=source_manifest_sha256, project_root=root)
    output = distribution._inside(output_dir, root, exists=False)
    if output.exists():
        raise FileExistsError("Each native trial needs a new immutable directory")
    output.mkdir(parents=True)
    request = {**source, "scales": values, "native_cpu_cores": cpu_threads,
        "output_dir": str(output), "result_path": str(output / "native_result.json")}
    request_path = output / "request.json"
    distribution._write(request_path, request)
    request_sha = distribution._sha(request_path)
    result = {"version": VERSION, "status": "infrastructure_failed", "scales": values,
        "trial_attempts": 1, "actual_native_solver_calls": 0, "source_build_solver_calls_included": False,
        "source_pins": source["source_pins"], "source_manifest_sha256": source_manifest_sha256,
        "scope": distribution.SCOPE, "clearance_policy": source["plan"]["clearance_policy"],
        "observations_used": False, "scientific_accepted": False, "mass_balance_checked": False,
        "limitations": ["Input plasma clearance is not a systemic model-clearance constraint",
            "Tied partition and exchange scales are unvalidated construction parameters",
            "Unchanged native net renal input does not prove observed renal clearance is preserved"]}
    runtime = run_bounded_r(script=root / WORKER, args=[str(request_path), request_sha],
        output_dir=output, project_root=root, timeout_seconds=timeout_seconds,
        cpu_threads=cpu_threads, max_output_bytes=max_output_bytes)
    result["runtime"] = runtime
    result["actual_native_solver_calls"] = _read_counter(output, request_sha)
    if runtime.get("status") != "completed" or runtime.get("returncode") != 0:
        result["failure"] = "Bounded native worker interrupted or failed; inspect execution.log"
    else:
        try:
            native = json.loads((output / "native_result.json").read_text())
            if (native.get("version") != VERSION or native.get("request_sha256") != request_sha
                    or native.get("pkml_sha256") != source["pkml_sha256"]
                    or native.get("source_manifest_sha256") != source_manifest_sha256
                    or native.get("scales") != values or native.get("target_output") != source["target_output"]
                    or native.get("actual_native_solver_calls") != result["actual_native_solver_calls"]):
                raise ValueError("Native result identity/solver count differs from request")
            status = native.get("status")
            if status not in {"predicted", "scientific_failed", "infrastructure_failed"}:
                raise ValueError("Unknown native trial status")
            if native.get("clearance_readback"):
                checks = validate_clearance_readback(native["clearance_readback"], source["plan"], values)
                result["clearance_checks"] = checks
                if status == "predicted" and not checks["accepted"]:
                    raise ValueError("Predicted trial failed independent native capacity reconstruction")
            if status == "predicted":
                if result["actual_native_solver_calls"] != 1 or not result.get("clearance_checks", {}).get("accepted"):
                    raise ValueError("Prediction requires one native solve and passed capacity audit")
                distribution._validate_readback(native.get("changed"), source["plan"]["changed"], values, changed=True)
                distribution._validate_readback(native.get("protected"), source["plan"]["protected"], values, changed=False)
                curve = native.get("curve", {})
                times, concentrations = curve.get("time_min", []), curve.get("concentration", [])
                if (len(times) < 2 or len(times) != len(concentrations) or curve.get("time_unit") != "min"
                        or not curve.get("unit") or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in times + concentrations)
                        or any(b <= a for a, b in zip(times, times[1:])) or min(concentrations) < -1e-10):
                    raise ValueError("Invalid native plasma concentration output")
                result["curve"] = curve
            if native.get("exported_pkml"):
                artifact = distribution._inside(output / native["exported_pkml"]["path"], root)
                if artifact.parent != output or distribution._sha(artifact) != native["exported_pkml"]["sha256"]:
                    raise ValueError("Changed model artifact path/hash mismatch")
                result["exported_pkml"] = {"path": str(artifact.relative_to(root)), "sha256": distribution._sha(artifact)}
            if distribution._sha(request_path) != request_sha or any(distribution._sha(Path(p)) != digest for p, digest in source["source_pins"].items()):
                raise ValueError("Raw source/code changed during native trial")
            result.update(status=status, failure=native.get("failure"), changed=native.get("changed", []),
                protected=native.get("protected", []), clearance_readback=native.get("clearance_readback", []),
                native_result={"path": str((output / "native_result.json").relative_to(root)),
                    "sha256": distribution._sha(output / "native_result.json")})
        except (ValueError, TypeError, KeyError, OSError) as exc:
            result.update(status="infrastructure_failed", failure="Native audit/contract failure: " + str(exc))
            result.pop("curve", None)
    manifest = output / "axes_trial_manifest.json"
    distribution._write(manifest, result)
    return {**result, "manifest": {"path": str(manifest.relative_to(root)), "sha256": distribution._sha(manifest)}}
