"""Resume completed native partitions without accepting their scientific content.

The old per-stage receipt did not persist the final JSON output hash. Its next
stage *did* persist the build project's hash. Consequently this module reuses
only that authenticated project and requires a fresh bounded JSON export. Every
source-global definition and selected simulation must still equal the original
cleaned source exactly. Full native-field comparison remains a later gate.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import recommendation_candidate_migration as migration

VERSION = "authenticated-native-partition-resume-v1"
R_SCRIPT = migration.R_SCRIPT
IMPLEMENTATION_FILES = (
    "scripts/recommendation_execution_repairs/native_partition_resume.py",
    "scripts/recommendation_candidate_migration.py",
    "src/pbpk_kb/blueprints/source_adaptation.py",
    "src/pbpk_kb/blueprints/design_runtime.py",
    R_SCRIPT,
)


def _inside(value, root, *, exists=True):
    return migration.inside(value, root, exists=exists)


def _ref(path, root):
    path = _inside(path, root)
    return {"path": str(path.relative_to(root)), "sha256": migration.sha(path)}


def _checked(ref, root):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("EXACT_PATH_SHA_REFERENCE_REQUIRED")
    path = _inside(ref["path"], root)
    if migration.sha(path) != ref["sha256"]:
        raise ValueError("PARTITION_RESOURCE_SHA_CHANGED:" + ref["path"])
    return path


def _read(path):
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError("BOUNDED_PARTITION_JSON_REQUIRED")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("DUPLICATE_JSON_KEY:" + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("NONFINITE_JSON_VALUE:" + value)

    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=nonfinite)


def _exact(a, b):
    """No tolerance, boolean/numeric coercion, list sorting or dropped fields."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_exact(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_exact(x, y) for x, y in zip(a, b))
    return a == b


def _names(snapshot):
    simulations = snapshot.get("Simulations")
    if not isinstance(simulations, list) or not 1 <= len(simulations) <= 256:
        raise ValueError("BOUNDED_NONEMPTY_SIMULATION_LIST_REQUIRED")
    names = [row.get("Name") for row in simulations]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("UNIQUE_SIMULATION_NAMES_REQUIRED")
    return names


def _source(provenance, root):
    ref = {"path": provenance["source_path"], "sha256": provenance["source_sha256"]}
    path = _checked(ref, root)
    if path.suffix != ".json":
        raise ValueError("JSON_ORIGINAL_SOURCE_REQUIRED")
    original = migration.clean_snapshot(_read(path))
    _names(original)
    return ref, original


def _stage(stage_dir, mode, expected_input, root):
    receipt_path = _inside(stage_dir / "stage_receipt.json", root)
    receipt = _read(receipt_path)
    process_path = _checked(receipt["process"], root)
    if process_path != stage_dir / "process_manifest.json":
        raise ValueError("STAGE_PROCESS_PATH_DIFFERS")
    process = _read(process_path)
    if (receipt.get("mode") != mode or receipt.get("status") != "completed"
            or receipt.get("stop_reason") is not None
            or process.get("status") != "completed" or type(process.get("returncode")) is not int
            or process.get("returncode") != 0
            or process.get("stop_reason") is not None
            or process.get("runtime_unchanged") is not True
            or process.get("registry_modified") is not False):
        raise ValueError("NATIVE_STAGE_NOT_COMPLETED_AND_UNCHANGED")
    if (type(process.get("timeout_seconds")) is not int
            or not 1 <= process["timeout_seconds"] <= 600
            or type(process.get("cpu_threads")) is not int
            or process["cpu_threads"] not in (1, 2)):
        raise ValueError("NATIVE_STAGE_RESOURCE_LIMITS_DIFFER")
    if receipt.get("input") != expected_input:
        raise ValueError("STAGE_INPUT_CHAIN_DIFFERS")
    _checked(receipt["input"], root)
    if (receipt.get("copy_before_sha256") != expected_input["sha256"]
            or receipt.get("original_after_sha256") != expected_input["sha256"]):
        raise ValueError("NATIVE_STAGE_ORIGINAL_OR_COPY_CHANGED")
    copy_after = receipt.get("copy_after_sha256")
    if not isinstance(copy_after, str) or len(copy_after) != 64 or any(c not in "0123456789abcdef" for c in copy_after):
        raise ValueError("COPIED_INPUT_AFTER_HASH_REQUIRED")
    if mode == "snapshot_to_project" and copy_after != expected_input["sha256"]:
        raise ValueError("JSON_COPY_CHANGED")
    if (process.get("script") != R_SCRIPT
            or process.get("script_sha256") != migration.sha(root / R_SCRIPT)):
        raise ValueError("NATIVE_WORKER_CHANGED")
    command = process.get("command", [])
    expected_tail = ["--vanilla", str(root / R_SCRIPT), mode,
                     str(stage_dir / "input"), str(stage_dir / "output")]
    if len(command) != 6 or command[1:] != expected_tail:
        raise ValueError("NATIVE_STAGE_COMMAND_DIFFERS")
    executable = _inside(command[0], root)
    if (_inside(process.get("process_manifest_path", ""), root) != process_path
            or _inside(process.get("log_path", ""), root) != stage_dir / "execution.log"):
        raise ValueError("NATIVE_STAGE_RECORDED_PATH_DIFFERS")
    binding = process.get("runtime_binding", {})
    resources = binding.get("locked_resources", [])
    if not resources or binding.get("gpu_used_by_native_solver") is not False:
        raise ValueError("LOCKED_NATIVE_RUNTIME_REQUIRED")
    resource_refs = [{"path": row["path"], "sha256": row["sha256"]} for row in resources]
    for ref in resource_refs:
        _checked(ref, root)
    if _ref(executable, root) not in resource_refs:
        raise ValueError("NATIVE_EXECUTABLE_NOT_BOUND")
    log = _inside(stage_dir / "execution.log", root)
    marker = "MIGRATION_STAGE_FINISHED_WITHOUT_ODE " + mode
    if marker not in log.read_text().splitlines():
        raise ValueError("NATIVE_NO_ODE_COMPLETION_MARKER_MISSING")
    return receipt, process, [_ref(receipt_path, root), _ref(process_path, root),
                              _ref(log, root), *resource_refs, _ref(root / R_SCRIPT, root)]


def authenticate_completed_partition(provenance, partition_dir, *, project_root=ROOT):
    """Return only a historically hash-bound project; never reuse old JSON.

    Requires BOTH old stages to have completed. No native process is started,
    and no file is written. The old export's input hash authenticates the build
    project, even when an interrupted source-level run never wrote its receipt.
    """
    root = Path(project_root).resolve(strict=True)
    part = _inside(partition_dir, root)
    source_ref, original = _source(provenance, root)
    scoped_path = _inside(part / "source_model_only.json", root)
    scoped = _read(scoped_path)
    names = _names(scoped)
    source_names = _names(original)
    if any(name not in source_names for name in names):
        raise ValueError("PARTITION_SIMULATION_NOT_IN_SOURCE")
    selected = [row for row in original["Simulations"] if row["Name"] in names]
    expected = deepcopy(original)
    expected["Simulations"] = selected
    if not _exact(scoped, expected):
        raise ValueError("PARTIAL_INPUT_DIFFERS_FROM_CLEANED_SOURCE")
    scoped_ref = _ref(scoped_path, root)
    _, build, refs = _stage(part / "build", "snapshot_to_project", scoped_ref, root)
    export_receipt = _read(_inside(part / "export/stage_receipt.json", root))
    project_ref = export_receipt["input"]
    project = _checked(project_ref, root)
    outputs = sorted((part / "build/output").glob("*.pksim5"))
    if len(outputs) != 1 or project != outputs[0]:
        raise ValueError("EXACTLY_ONE_BOUND_BUILD_PROJECT_REQUIRED")
    _, export, export_refs = _stage(part / "export", "project_to_snapshot", project_ref, root)
    if not _exact(build["runtime_binding"], export["runtime_binding"]):
        raise ValueError("BUILD_EXPORT_NATIVE_RUNTIME_DIFFERS")
    # Old output is deliberately neither read nor considered hash-authenticated:
    # its SHA was not written by the old successful per-stage receipt.
    refs += export_refs + [source_ref, scoped_ref, project_ref]
    refs = list({(ref["path"], ref["sha256"]): ref for ref in refs}.values())
    return {"version": VERSION, "status": "authenticated_project_pending_reexport",
            "source": source_ref, "partition_path": str(part.relative_to(root)),
            "simulation_names": names, "native_project": project_ref,
            "native_snapshot": None,
            "historical_hash_anchor": _ref(part / "export/stage_receipt.json", root),
            "old_export_output_reused": False, "scientific_accepted": False,
            "final_source_comparison_required": True, "ode_run": False,
            "sealed_observations_accessed": False, "provenance_refs": refs,
            "implementation": [_ref(root / path, root) for path in IMPLEMENTATION_FILES]}


def collect_completed_partitions(provenance, attempt_dirs, *, project_root=ROOT):
    """List authenticated completed partitions; report rejected paths separately.

    ``attempt_dirs`` names source-specific directories, not broad project trees.
    A changed original source is a global hard error, never a skippable partition.
    Duplicate simulation sets retain the first authenticated attempt, and any
    partially overlapping set is rejected to prevent duplicate simulations.
    """
    root = Path(project_root).resolve(strict=True)
    source_ref, _ = _source(provenance, root)
    certified, rejected, covered = [], [], set()
    for directory in attempt_dirs:
        directory = _inside(directory, root)
        for part in sorted(directory.glob("partition_[0-9][0-9][0-9]")):
            if not part.is_dir():
                continue
            try:
                row = authenticate_completed_partition(provenance, part, project_root=root)
                names = set(row["simulation_names"])
                if names & covered:
                    raise ValueError("DUPLICATE_OR_OVERLAPPING_PARTITION_NOT_REUSED")
                covered |= names
                certified.append(row)
            except (ValueError, KeyError, OSError) as exc:
                rejected.append({"partition_path": str(part.relative_to(root)),
                                 "reason": type(exc).__name__ + ":" + str(exc)})
    _checked(source_ref, root)
    return {"version": VERSION, "source": source_ref, "partitions": certified,
            "rejected": rejected, "scientific_accepted": False,
            "final_source_comparison_required": True}


def resume_partition(artifact, output_dir, *, stage_runner, project_root=ROOT,
                     timeout_seconds=600, cpu_threads=1, stop_file=None):
    """Explicit future-attempt action: re-export a bound project, pin output SHA.

    ``stage_runner`` follows snapshot_repair._native_stage's public call shape.
    Return ``snapshot`` as a reference and ``native_document`` for later strict
    partition merge/full-source proof. This operation alone never accepts one.
    """
    root = Path(project_root).resolve(strict=True)
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600:
        raise ValueError("BOUNDED_TIMEOUT_REQUIRED")
    if type(cpu_threads) is not int or cpu_threads not in (1, 2):
        raise ValueError("BOUNDED_CPU_THREADS_REQUIRED")
    if stop_file is not None:
        stop_file = _inside(stop_file, root, exists=False)
        if stop_file.exists():
            raise InterruptedError("Partition resume stopped before creation")
    provenance = {"source_path": artifact["source"]["path"],
                  "source_sha256": artifact["source"]["sha256"]}
    current = authenticate_completed_partition(provenance, artifact["partition_path"], project_root=root)
    if not _exact(current, artifact):
        raise ValueError("AUTHENTICATED_PARTITION_CHANGED_BEFORE_REUSE")
    destination = _inside(output_dir, root, exists=False)
    destination.mkdir(parents=True, exist_ok=False)
    output, stage_refs = stage_runner("project_to_snapshot", _checked(artifact["native_project"], root),
        destination / "export", root=root, timeout_seconds=timeout_seconds,
        cpu_threads=cpu_threads, stop_file=stop_file)
    _, fresh_process, checked_stage_refs = _stage(destination / "export", "project_to_snapshot",
                                                 artifact["native_project"], root)
    old_process = _read(_inside(root / artifact["partition_path"] / "export/process_manifest.json", root))
    if not _exact(fresh_process["runtime_binding"], old_process["runtime_binding"]):
        raise ValueError("REEXPORT_NATIVE_RUNTIME_DIFFERS")
    outputs = sorted((destination / "export/output").glob("*.json"))
    output = _inside(output, root)
    if len(outputs) != 1 or output != outputs[0]:
        raise ValueError("EXACTLY_ONE_FRESH_EXPORT_OUTPUT_REQUIRED")
    for ref in stage_refs:
        _checked(ref, root)
    native = _read(output)
    if native.get("Version") != 80 or set(_names(native)) != set(artifact["simulation_names"]):
        raise ValueError("FRESH_NATIVE_PARTITION_SCOPE_DIFFERS")
    after = authenticate_completed_partition(provenance, artifact["partition_path"], project_root=root)
    if not _exact(after, artifact):
        raise ValueError("AUTHENTICATED_PARTITION_CHANGED_DURING_REEXPORT")
    snapshot = _ref(output, root)
    proof = {"version": VERSION, "status": "reexported_pending_final_comparison",
             "created_at": datetime.now(timezone.utc).isoformat(),
             "original_partition": artifact, "snapshot": snapshot,
             "scientific_accepted": False, "final_source_comparison_required": True,
             "ode_run": False, "sealed_observations_accessed": False,
             "provenance_refs": [*artifact["provenance_refs"], *checked_stage_refs,
                                  *stage_refs, snapshot],
             "implementation": artifact["implementation"]}
    certificate = destination / "resume_certificate.json"
    migration.save(certificate, proof)
    _checked(snapshot, root)
    return {"snapshot": snapshot, "native_document": native,
            "certificate": _ref(certificate, root), "proof": proof,
            "provenance_refs": [*proof["provenance_refs"], _ref(certificate, root)]}


def verify_resume_certificate(certificate, *, project_root=ROOT):
    """Verify an already frozen resumed artifact, still pending final comparison."""
    root = Path(project_root).resolve(strict=True)
    proof = _read(_checked(certificate, root))
    if (proof.get("version") != VERSION or proof.get("status") != "reexported_pending_final_comparison"
            or proof.get("scientific_accepted") is not False
            or proof.get("final_source_comparison_required") is not True
            or proof.get("ode_run") is not False
            or proof.get("sealed_observations_accessed") is not False):
        raise ValueError("PARTITION_RESUME_CERTIFICATE_SCOPE_DIFFERS")
    for ref in proof["provenance_refs"]:
        _checked(ref, root)
    if proof.get("implementation") != [_ref(root / path, root) for path in IMPLEMENTATION_FILES]:
        raise ValueError("PARTITION_RESUME_IMPLEMENTATION_CHANGED")
    artifact = proof["original_partition"]
    provenance = {"source_path": artifact["source"]["path"],
                  "source_sha256": artifact["source"]["sha256"]}
    current = authenticate_completed_partition(provenance, artifact["partition_path"], project_root=root)
    if not _exact(current, artifact):
        raise ValueError("RESUMED_PARTITION_SOURCE_CHAIN_CHANGED")
    native = _read(_checked(proof["snapshot"], root))
    if native.get("Version") != 80 or set(_names(native)) != set(artifact["simulation_names"]):
        raise ValueError("RESUMED_NATIVE_SCOPE_DIFFERS")
    return {"proof": proof, "snapshot": proof["snapshot"], "native_document": native}
