"""Bounded native OSPSuite parameter batches; no fitting or scientific decisions."""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

BATCH_VERSION = "osp-parameter-batch-1.1.0"
_R_SCRIPT = Path(__file__).resolve().parents[1] / "model_runtime/R/blueprint_parameter_batch.R"


def _run_bounded_r(**kwargs: Any) -> dict[str, Any]:
    from .design_runtime import run_bounded_r
    return run_bounded_r(**kwargs)


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _inside(path: Path, root: Path, *, must_exist: bool) -> Path:
    resolved = Path(path).resolve(strict=must_exist)
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Batch paths must remain inside the project root.")
    if any(ord(char) < 32 for char in str(resolved)):
        raise ValueError("Batch paths must not contain control characters.")
    return resolved


def _text(value: Any, label: str, maximum: int = 2048) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or len(value) > maximum or any(ord(char) < 32 for char in value)):
        raise ValueError(f"{label} must be explicit bounded text without surrounding whitespace.")
    return value


def _quantity_path(value: Any, label: str) -> str:
    path = _text(value, label)
    if any(part in {"", ".", ".."} for part in path.split("|")) or "*" in path or "?" in path:
        raise ValueError(f"{label} must be a full exact quantity path, without wildcards.")
    return path


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Parameter values and native curves must contain real numbers.")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Numeric value is not representable as a finite float.") from exc
    if not math.isfinite(result):
        raise ValueError("Parameter values and native curves must be finite.")
    return result


def _positive_int(value: Any, label: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f"{label} must be a positive integer within the supported limit.")
    return value


def _write_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(_json(payload) + "\n")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Native result contains duplicate JSON keys.")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise ValueError(f"Native result contains nonfinite constant {value}.")


def _validated_runs(native: Any, request: dict[str, Any], request_checksum: str) -> list[dict[str, Any]]:
    if not isinstance(native, dict) or native.get("request_checksum") != request_checksum:
        raise ValueError("Native result does not match the locked request.")
    if native.get("pkml_sha256") != request["pkml_sha256"] or native.get("target_output") != request["target_output"]:
        raise ValueError("Native result model or target binding differs from the request.")
    if type(native.get("native_cpu_cores")) is not int or native['native_cpu_cores'] != request['native_cpu_cores']:
        raise ValueError("Native batch core limit differs from the locked request.")
    runs = native.get("runs")
    if not isinstance(runs, list) or len(runs) != len(request["values"]):
        raise ValueError("Native batch result count differs from requested parameter vectors.")
    by_index: dict[int, dict[str, Any]] = {}
    run_ids: set[str] = set()
    units: set[str] = set()
    for row in runs:
        if not isinstance(row, dict):
            raise ValueError("Native batch run must be an object.")
        index = row.get("vector_index")
        if type(index) is not int or index in by_index or not 0 <= index < len(runs):
            raise ValueError("Native vector indices must form a unique complete request mapping.")
        run_id = _text(row.get("run_id"), "Native run ID", 256)
        if run_id in run_ids:
            raise ValueError("Native run IDs must be unique.")
        run_ids.add(run_id)
        returned_values = row.get("values")
        if (not isinstance(returned_values, list)
                or len(returned_values) != len(request["parameter_specs"])
                or [_number(value) for value in returned_values] != request["values"][index]):
            raise ValueError("Native run parameter values do not match their request vector.")
        times, concentrations = row.get("time_min"), row.get("concentration")
        if (not isinstance(times, list) or not isinstance(concentrations, list)
                or len(times) < 2 or len(times) != len(concentrations)):
            raise ValueError("Native curve must have matching time and concentration arrays.")
        times = [_number(value) for value in times]
        concentrations = [_number(value) for value in concentrations]
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("Native times must increase strictly; no interpolation is performed.")
        if row.get("time_unit") != "min":
            raise ValueError("Native time metadata must explicitly declare minutes.")
        unit = _text(row.get("unit"), "Native concentration unit", 128)
        units.add(unit)
        by_index[index] = dict(row, time_min=times, concentration=concentrations)
    if len(units) != 1:
        raise ValueError("The same native output must retain one unit across parameter vectors.")
    return [by_index[index] for index in range(len(runs))]


def run_parameter_batch(
    *, pkml_path: Path, parameter_specs: list[dict[str, str]], values: list[list[float]],
    target_output: str, output_dir: Path, project_root: Path, timeout_seconds: int = 180,
    cpu_threads: int = 2, max_output_bytes: int = 128 * 1024**2,
) -> dict[str, Any]:
    """Execute 1..64 parameter vectors against one SHA-locked native model.

    Invalid inputs fail before creating output. Runtime/contract failures return
    a failed manifest with no accepted curves. Callers own logical scenario
    budget accounting; this adapter does not read any observation dataset.
    """
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("project_root must be a directory.")
    pkml = _inside(pkml_path, root, must_exist=True)
    if not pkml.is_file() or pkml.suffix.lower() != ".pkml":
        raise ValueError("pkml_path must identify a project-native PKML file.")
    destination = _inside(output_dir, root, must_exist=False)
    # Isolated Python development copies still execute the unchanged audited
    # project R worker, whose allowlist is enforced by design_runtime.
    script = _inside(root / 'src/pbpk_kb/model_runtime/R/blueprint_parameter_batch.R', root, must_exist=True)
    if destination.exists():
        raise FileExistsError("Parameter batch output directory already exists.")
    _positive_int(timeout_seconds, "timeout_seconds", 3600)
    _positive_int(cpu_threads, "cpu_threads", 32)
    _positive_int(max_output_bytes, "max_output_bytes", 2 * 1024**3)
    if max_output_bytes < 1024**2:
        raise ValueError("max_output_bytes must allow at least 1 MiB of bounded runtime metadata.")
    if not isinstance(parameter_specs, list) or not 1 <= len(parameter_specs) <= 64:
        raise ValueError("Provide 1..64 explicit variable parameter specifications.")
    specs = []
    for spec in parameter_specs:
        if not isinstance(spec, dict) or set(spec) != {"path", "unit"}:
            raise ValueError("Each parameter specification requires exactly path and unit.")
        specs.append({"path": _quantity_path(spec["path"], "Parameter path"),
                      "unit": _text(spec["unit"], "Parameter unit", 128)})
    if len({item["path"] for item in specs}) != len(specs):
        raise ValueError("Variable parameter paths must be unique.")
    if not isinstance(values, list) or not 1 <= len(values) <= 64:
        raise ValueError("A parameter batch must contain 1..64 vectors.")
    normalized = []
    for row in values:
        if not isinstance(row, list) or len(row) != len(specs):
            raise ValueError("Every parameter vector must match the parameter specification count.")
        normalized.append([_number(value) for value in row])
    request = {"adapter_version": BATCH_VERSION, "pkml_path": str(pkml), "pkml_sha256": _sha(pkml),
               "parameter_specs": specs, "values": normalized,
               "native_cpu_cores": cpu_threads,
               "target_output": _quantity_path(target_output, "Target output")}
    request_checksum = hashlib.sha256(_json(request).encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    config_path, native_path = destination / "request.json", destination / "native_results.json"
    config = {**request, "request_checksum": request_checksum, "result_path": str(native_path)}
    _write_new(config_path, config)
    manifest: dict[str, Any] = {
        "adapter_version": BATCH_VERSION, "status": "failed", "request_checksum": request_checksum,
        "pkml_sha256": request["pkml_sha256"], "parameter_specs": specs,
        "target_output": request["target_output"], "requested_vector_count": len(normalized),
        "completed_vector_count": 0, "runs": [], "backend": "OSPSuite native CPU",
        "native_cpu_cores": cpu_threads,
        "scientific_status": "not_evaluated", "interpolation_performed": False,
    }
    runtime = _run_bounded_r(script=script, args=[str(config_path), _sha(config_path)],
                             output_dir=destination, project_root=root,
                             timeout_seconds=timeout_seconds, cpu_threads=cpu_threads,
                             max_output_bytes=max_output_bytes)
    manifest["runtime"] = runtime
    if runtime.get("status") != "completed" or runtime.get("returncode") != 0:
        manifest["error"] = "NATIVE_BATCH_EXECUTION_FAILED"
    else:
        try:
            if _sha(pkml) != request["pkml_sha256"]:
                raise ValueError("PKML changed during native execution.")
            if native_path.is_symlink() or not native_path.is_file() or native_path.stat().st_size > max_output_bytes:
                raise ValueError("Native results are absent, unsafe, or exceed the output byte limit.")
            native = json.loads(native_path.read_text(encoding="utf-8"),
                                object_pairs_hook=_unique_object, parse_constant=_bad_constant)
            manifest["runs"] = _validated_runs(native, request, request_checksum)
            manifest.update(status="completed", completed_vector_count=len(manifest["runs"]),
                            native_results_sha256=_sha(native_path),
                            ospsuite_version=native.get("ospsuite_version"))
        except (OSError, ValueError, TypeError) as exc:
            manifest.update(error="NATIVE_BATCH_RESULT_INVALID", error_detail=str(exc))
    # Keep curves once on disk; the compact manifest indexes native_results.
    persisted = dict(manifest)
    persisted["runs"] = [{key: value for key, value in run.items()
                          if key not in {"time_min", "concentration"}} for run in manifest["runs"]]
    if native_path.is_file() and not native_path.is_symlink():
        persisted["native_results_file"] = native_path.name
    payload_bytes = (_json(persisted) + "\n").encode("utf-8")
    existing_bytes = sum(path.stat().st_size for path in destination.rglob("*")
                         if path.is_file() and not path.is_symlink())
    if existing_bytes + len(payload_bytes) > max_output_bytes:
        manifest.update(status="failed", error="BATCH_OUTPUT_BYTE_LIMIT", runs=[],
                        completed_vector_count=0, manifest_persisted=False)
        return manifest
    _write_new(destination / "parameter_batch_manifest.json", persisted)
    manifest["manifest_path"] = str(destination / "parameter_batch_manifest.json")
    return manifest
