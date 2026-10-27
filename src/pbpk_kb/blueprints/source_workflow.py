"""Project-local source PBPK adaptation and native execution entry point.

Execution and request realization are engineering checks. Neither is a claim
of clinical qualification or a substitute for comparison with observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

from .design_runtime import execute_snapshot


def inside(path: Path | str, root: Path, *, exists: bool = False) -> Path:
    root = root.resolve(strict=True)
    target = (root / path).resolve(strict=exists)
    if target == root or not target.is_relative_to(root):
        raise ValueError("Resource must remain inside the project")
    return target


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value) -> None:
    """Atomic bounded JSON; caller validates the destination project directory."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if len(payload.encode()) > 32 * 1024**2:
        raise ValueError("JSON artifact exceeds 32 MiB")
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_curve(directory: Path, manifest: dict, target: str) -> dict:
    """Read one checksum-bound native molar output with its exact output path."""
    outputs = manifest.get("curve_outputs", [])
    if len(outputs) != 1:
        raise ValueError("Expected exactly one native curve export")
    path = inside(outputs[0]["path"], directory, exists=True)
    if sha256(path) != outputs[0]["sha256"]:
        raise ValueError("Native curve checksum changed")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        matches = [name for name in reader.fieldnames or []
                   if name.replace("μ", "u").replace("µ", "u") == target + " [umol/l]"]
        if len(matches) != 1 or "Time [min]" not in (reader.fieldnames or []):
            raise ValueError("Native output path or units differ from the requested output")
        rows = list(reader)
    if len({row["IndividualId"] for row in rows}) != 1:
        raise ValueError("One individual is required per pilot prediction")
    times = [float(row["Time [min]"]) for row in rows]
    values = [float(row[matches[0]]) for row in rows]
    if (len(times) < 2 or any(not math.isfinite(x) for x in times + values)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("Native curve has invalid or unordered values")
    return {"time_min": times, "concentration": values, "time_unit": "min", "unit": "umol/l"}


def design_source_model(*, source_path: Path, source_sha256: str,
                        simulation_name: str, request: dict,
                        output_dir: Path, project_root: Path,
                        execute: bool = False, native_expectations: dict | None = None,
                        timeout_seconds: int = 240, cpu_threads: int = 2,
                        pk_evidence: dict | None = None, pk_context: dict | None = None,
                        predictor_input: dict | None = None) -> dict:
    from .source_adaptation import load_source_model, adapt_source_model
    if (pk_evidence is None) != (pk_context is None):
        raise ValueError('PK evidence and target context must be supplied together')
    if (pk_evidence is not None and not execute) or (predictor_input is not None and pk_evidence is None):
        raise ValueError('PK checks require evidence, target context and execution; use check_blueprint_pk.py for existing outputs')
    root = project_root.resolve(strict=True)
    output = inside(output_dir, root)
    output.mkdir(parents=True, exist_ok=False)
    source = load_source_model(source_path, source_sha256, simulation_name, root)
    blueprint = adapt_source_model(source, request, root)
    save(output / "blueprint.json", blueprint)
    result = {"status": blueprint["status"], "clinical_qualified": False,
              "blueprint_path": str((output / "blueprint.json").relative_to(root)),
              "native_executed": False, "native_snapshot_verified": False,
              "request_realization_verified": False,
              "independent_expectations_supplied": native_expectations is not None}
    if blueprint["status"] == "adapted":
        save(output / "snapshot.json", blueprint["snapshot"])
    if execute and blueprint["status"] == "adapted":
        from .source_adaptation_audit import audit_execution, derive_administration_expectations
        expectations = (native_expectations if native_expectations is not None
                        else derive_administration_expectations(blueprint["snapshot"]))
        execution = execute_snapshot(snapshot=blueprint["snapshot"], output_dir=output / "native",
                                     project_root=root, timeout_seconds=timeout_seconds,
                                     max_output_bytes=128 * 1024**2, cpu_threads=cpu_threads)
        result.update(native_executed=True, native_status=execution["status"],
                      execution_manifest=execution["manifest_path"])
        result["status"] = "native_failed"
        if execution.get("engineering_accepted"):
            audit = audit_execution(snapshot=blueprint["snapshot"], execution_manifest=execution,
                                    output_dir=output / "readback", project_root=root,
                                    **expectations)
            save(output / "readback_result.json", audit)
            verified = audit.get("accepted") is True
            result.update(status="engineering_verified" if verified else "readback_failed",
                          native_snapshot_verified=verified,
                          request_realization_verified=verified and native_expectations is not None)
            if verified and pk_evidence is not None:
                from .pk_constraints import check_source_pk
                pk_report = check_source_pk(snapshot=blueprint['snapshot'], execution_manifest=execution['manifest_path'],
                    evidence=pk_evidence, target_context=pk_context, predictor_input=predictor_input,
                    output_dir=output / 'pk_check', project_root=root)
                result.update(pk_check_status=pk_report['status'], pk_check_report=pk_report['report_path'],
                              pk_constraint_comparison_count=pk_report['constraint_comparison_count'])
                if pk_report['status'] != 'completed':
                    result['status'] = 'pk_check_failed'
    save(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Adapt a full source PBPK model to a structured request.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--simulation", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expectations", type=Path,
                        help="JSON {expectations:[{path,value,unit}],expected_event_count:N} for native readback")
    parser.add_argument('--pk-evidence', type=Path, help='Optional PK evidence JSON; requires --pk-context and --execute')
    parser.add_argument('--pk-context', type=Path, help='Drug identity and clinical context for PK comparison')
    parser.add_argument('--pk-predictor-input', type=Path, help='Optional checksummed SMILES identity for local PKSmart prediction')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    request = json.loads(inside(args.request, root, exists=True).read_text())
    expectations = (json.loads(inside(args.expectations, root, exists=True).read_text())
                    if args.expectations else None)
    def optional_json(path):
        return json.loads(inside(path, root, exists=True).read_text()) if path else None
    result = design_source_model(source_path=inside(args.source, root, exists=True),
                                 source_sha256=args.sha256, simulation_name=args.simulation,
                                 request=request, output_dir=args.output, project_root=root,
                                 execute=args.execute, native_expectations=expectations,
                                 pk_evidence=optional_json(args.pk_evidence), pk_context=optional_json(args.pk_context),
                                 predictor_input=optional_json(args.pk_predictor_input))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"adapted", "engineering_verified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
