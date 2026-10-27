"""Bounded project-local Linux native execution; engineering evidence only."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import selectors
import signal
import struct
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUNTIME_VERSION = "blueprint-native-runtime-0.1.0"
_SOURCE_ROOT = "artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173"
_DATABASE_SHA = "aec61433ddddf39ec5c832bda90a6977a25ea5427e77edc08f2d97a751fe1469"
_RUNNER = "src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path, *, exists: bool = False) -> Path:
    resolved = path.resolve(strict=exists)
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Path must remain inside the project")
    return resolved


def _write_json(path: Path, value: dict, *, replace: bool = False) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    target = path.with_name(path.name + ".tmp") if replace else path
    with target.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if replace:
        target.replace(path)


def _version(path: Path) -> dict:
    """Read the PE version resource without loading/executing a DLL."""
    raw = path.read_bytes()
    offset = raw.find(struct.pack("<I", 0xFEEF04BD))
    if offset < 0 or offset + 52 > len(raw):
        raise ValueError("Native DLL has no readable fixed version resource: " + path.name)
    values = struct.unpack_from("<13I", raw, offset)
    ms, ls = values[2], values[3]
    version = f"{ms >> 16}.{ms & 65535}.{ls >> 16}.{ls & 65535}"
    informational = re.findall(r"12\.3\.137\+[0-9a-f]{40}", raw.decode("ascii", errors="ignore"))
    return {"file_version": version, "informational_version": informational[0] if informational else None}


def _runtime_binding(root: Path) -> dict:
    lib = root / "runtime/ospsuite-env/lib/R/library/ospsuite/lib"
    resources = [
        root / "runtime/ospsuite-env/bin/Rscript", root / "runtime/ospsuite-dotnet/dotnet",
        root / "runtime/ospsuite-env/lib/R/library/ospsuite/DESCRIPTION",
        root / "runtime/ospsuite-env/lib/R/library/rSharp/DESCRIPTION",
        lib / "PKSim.Core.dll", lib / "PKSim.R.dll", lib / "PKSim.CLI.Core.dll",
        lib / "OSPSuite.Core.dll", lib / "libOSPSuite.SimModelNative.so",
        lib / "libOSPSuite.SimModelSolver_CVODES.so", lib / "PKSimDB.sqlite",
        root / _SOURCE_ROOT / "src/Db/PKSimDB.sqlite",
    ]
    locked = []
    for path in resources:
        path = _inside(path, root, exists=True)
        if not path.is_file():
            raise ValueError("Runtime resource is not a regular file")
        resource = {"path": str(path.relative_to(root)), "sha256": _sha(path), "byte_size": path.stat().st_size}
        if path.suffix == ".dll":
            resource.update(_version(path))
        locked.append(resource)
    database = next(item for item in locked if item["path"] == str((lib / "PKSimDB.sqlite").relative_to(root)))
    source_database = locked[-1]
    if database["sha256"] != _DATABASE_SHA or source_database["sha256"] != _DATABASE_SHA:
        raise ValueError("Runtime/source PKSimDB differs from the pinned standard-model database")
    core_version = _version(lib / "PKSim.Core.dll")
    engine_version = core_version["file_version"]
    if any(item.get("file_version") != engine_version for item in locked
           if Path(item["path"]).name in {"PKSim.Core.dll", "PKSim.R.dll", "PKSim.CLI.Core.dll"}):
        raise ValueError("PK-Sim runtime DLL versions are inconsistent")
    if engine_version != "12.3.137.0":
        raise ValueError("Native engine changed; register its actual version before using this adapter")
    return {
        "source_definition_version": "12.3.173", "actual_engine_version": engine_version,
        "actual_engine_informational_version": core_version["informational_version"],
        "source_runtime_version_match": False, "source_engine_equivalence": "not_verified",
        "database_sha256": database["sha256"], "database_matches_source": True,
        "native_solver_backend": "CPU_CVODES", "gpu_used_by_native_solver": False,
        "locked_resources": locked,
    }


def _binding_unchanged(binding: dict, root: Path) -> bool:
    return all((path := _inside(root / item["path"], root, exists=True)).is_file()
               and _sha(path) == item["sha256"] for item in binding["locked_resources"])


def _environment(root: Path, temporary: Path, cpu_threads: int) -> dict[str, str]:
    prefix = root / "runtime/ospsuite-env"
    library = prefix / "lib/R/library"
    dotnet = root / "runtime/ospsuite-dotnet"
    # Deliberately do not copy credentials or other service settings into the worker.
    environment = {
        "PATH": f"{dotnet}:{prefix / 'bin'}:/usr/bin:/bin", "DOTNET_ROOT": str(dotnet),
        "R_LIBS_USER": str(library), "R_LIBS_SITE": str(library),
        "LD_LIBRARY_PATH": ":".join(map(str, (library / "ospsuite/lib", prefix / "lib", prefix / "lib/R/lib"))),
        "DOTNET_CLI_HOME": str(temporary), "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1", "TMPDIR": str(temporary), "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "",
    }
    for key in ("DOTNET_PROCESSOR_COUNT", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = str(cpu_threads)
    return environment


def _directory_bytes(directory: Path) -> int:
    total = 0
    for parent, subdirs, files in os.walk(directory, followlinks=False):
        for name in subdirs + files:
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError("Worker output contains a symlink")
        for name in files:
            try:
                total += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass  # A native temporary file can disappear while being inspected.
    return total


def _stop_owned_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def run_bounded_r(
    *, script: Path, args: list[str], output_dir: Path, project_root: Path,
    timeout_seconds: int = 180, cpu_threads: int = 2,
    max_output_bytes: int = 256 * 1024**2,
    stop_file: Path | None = None,
) -> dict[str, Any]:
    """Run a fixed project R worker in its own process group and bounded folder.

    ``output_dir`` must already exist; execution.log/process_manifest.json are
    exclusive per invocation. Native disk writes are polled every 0.2 seconds;
    this is an actively enforced threshold, not a filesystem quota. The log is
    piped and capped before writing. No unrelated process is inspected or stopped.
    """
    root = project_root.resolve(strict=True)
    directory = _inside(output_dir, root, exists=True)
    script = _inside(script, root, exists=True)
    stop = _inside(stop_file, root, exists=False) if stop_file is not None else None
    script_root = root / "src/pbpk_kb/model_runtime/R"
    if not script.is_relative_to(script_root) or script.suffix != ".R" or not script.is_file():
        raise ValueError("Only fixed project model_runtime/R scripts can be executed")
    if not directory.is_dir():
        raise ValueError("output_dir must be an existing directory")
    for name, value, minimum, maximum in (
        ("timeout_seconds", timeout_seconds, 1, 3600), ("cpu_threads", cpu_threads, 1, 32),
        ("max_output_bytes", max_output_bytes, 1024**2, 2 * 1024**3),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    if not isinstance(args, list) or any(not isinstance(item, str) or "\x00" in item for item in args):
        raise ValueError("args must be a list of strings without NUL")
    log_path, manifest_path = directory / "execution.log", directory / "process_manifest.json"
    if log_path.exists() or manifest_path.exists():
        raise FileExistsError("Execution logs already exist; use a new per-call folder")
    temporary = directory / "temporary"
    temporary.mkdir(exist_ok=False)
    result: dict[str, Any] = {
        "runtime_adapter_version": RUNTIME_VERSION, "status": "initializing", "returncode": None,
        "stop_reason": None, "created_at": datetime.now(UTC).isoformat(), "elapsed_seconds": 0.0,
        "timeout_seconds": timeout_seconds, "max_output_bytes": max_output_bytes,
        "cpu_threads": cpu_threads, "log_path": str(log_path), "process_manifest_path": str(manifest_path),
        "script": str(script.relative_to(root)), "script_sha256": _sha(script),
        "runtime_binding": None, "engineering_accepted": False, "scientific_accepted": False,
        "registry_modified": False, "pid": None, "uid": os.getuid(),
    }
    _write_json(manifest_path, result)
    started = time.monotonic()
    child = None
    interrupt = None
    try:
        if stop is not None and stop.exists():
            result['stop_reason'] = 'PROJECT_STOP_REQUESTED'
            raise InterruptedError('This project requested native execution to stop')
        result["runtime_binding"] = _runtime_binding(root)
        if _directory_bytes(directory) >= max_output_bytes:
            raise ValueError("Input folder already exceeds the output budget")
        command = [str(root / "runtime/ospsuite-env/bin/Rscript"), "--vanilla", str(script), *args]
        result["command"] = command
        with log_path.open("xb") as log:
            child = subprocess.Popen(command, cwd=root, env=_environment(root, temporary, cpu_threads),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            result.update(pid=child.pid, status="running")
            _write_json(manifest_path, result, replace=True)
            log_limit = min(8 * 1024**2, max_output_bytes // 4)
            written = 0
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                while True:
                    for key, _ in selector.select(timeout=0.2):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        room = log_limit - written
                        if room > 0:
                            log.write(chunk[:room])
                            written += min(room, len(chunk))
                        if len(chunk) > room:
                            result["stop_reason"] = "LOG_SIZE_LIMIT"
                    log.flush()
                    if stop is not None and stop.exists():
                        result['stop_reason'] = 'PROJECT_STOP_REQUESTED'
                    if _directory_bytes(directory) > max_output_bytes:
                        result["stop_reason"] = "OUTPUT_SIZE_LIMIT"
                    if time.monotonic() - started > timeout_seconds:
                        result["stop_reason"] = result["stop_reason"] or "EXECUTION_TIMEOUT"
                    if result["stop_reason"]:
                        _stop_owned_child(child)
                        break
                    if child.poll() is not None and not selector.get_map():
                        break
            result["returncode"] = child.wait(timeout=5)
        result["runtime_unchanged"] = _binding_unchanged(result["runtime_binding"], root)
        if not result["runtime_unchanged"]:
            result["stop_reason"] = "RUNTIME_CHANGED_DURING_EXECUTION"
        if _sha(script) != result["script_sha256"]:
            result["stop_reason"] = "WORKER_SCRIPT_CHANGED_DURING_EXECUTION"
        result["status"] = "completed" if result["returncode"] == 0 and not result["stop_reason"] else "failed"
    except BaseException as exc:
        result["status"] = "blocked" if child is None else "failed"
        result["stop_reason"] = result["stop_reason"] or ("EXECUTION_INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "RUNTIME_EXECUTION_ERROR")
        result["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            interrupt = exc
    finally:
        if child is not None:
            _stop_owned_child(child)
            result["returncode"] = child.returncode
            if child.stdout is not None:
                child.stdout.close()
        result["elapsed_seconds"] = time.monotonic() - started
        try:
            result["output_bytes"] = _directory_bytes(directory)
        except ValueError as exc:
            result["output_bytes"] = None
            result.update(status="failed", stop_reason="UNSAFE_OUTPUT", error=str(exc))
        _write_json(manifest_path, result, replace=True)
    if interrupt is not None:
        raise interrupt
    return result


def _curve(path: Path, directory: Path) -> dict:
    rows, invalid, positive = 0, 0, 0
    times: dict[float, float] = {}
    monotone = True
    minima: list[float] = []
    maxima: list[float] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        header_valid = len(header) >= 3 and header[0] == "IndividualId" and header[1].startswith("Time [")
        for cells in reader:
            rows += 1
            try:
                values = [float(value) for value in cells]
            except ValueError:
                invalid += 1
                continue
            if len(values) != len(header) or len(values) < 3 or not all(math.isfinite(v) for v in values):
                invalid += 1
                continue
            person, current_time = values[:2]
            monotone = monotone and (person not in times or current_time >= times[person])
            times[person] = current_time
            positive += sum(value > 0 for value in values[2:])
            if not minima:
                minima, maxima = values[2:].copy(), values[2:].copy()
            else:
                minima = [min(a, b) for a, b in zip(minima, values[2:])]
                maxima = [max(a, b) for a, b in zip(maxima, values[2:])]
    return {"path": str(path.relative_to(directory)), "sha256": _sha(path), "byte_size": path.stat().st_size,
            "rows": rows, "columns": header, "invalid_numeric_rows": invalid, "positive_signal_values": positive,
            "time_monotone_within_individual": monotone, "header_valid": header_valid,
            "output_minima": minima, "output_maxima": maxima,
            "finite_curve": header_valid and rows > 1 and invalid == 0 and monotone}


def _snapshot_names(snapshot: dict) -> list[str]:
    if not isinstance(snapshot, dict) or type(snapshot.get("Version")) is not int or snapshot["Version"] != 80:
        raise ValueError("Expected a Version=80 snapshot object")
    simulations = snapshot.get("Simulations")
    if not isinstance(simulations, list) or not 1 <= len(simulations) <= 24:
        raise ValueError("Snapshot must contain 1..24 simulations")
    names = []
    for simulation in simulations:
        name = simulation.get("Name") if isinstance(simulation, dict) else None
        if not isinstance(name, str) or not name.strip() or any(char in name for char in "/\\\x00\n\r"):
            raise ValueError("Simulation names must be nonempty and cannot contain filesystem separators")
        if simulation.get("Model") != "4Comp":
            raise ValueError("Only the standard 4Comp snapshot is supported")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Simulation names must be unique")
    return names


def execute_snapshot(
    *, snapshot: dict, output_dir: Path, project_root: Path, timeout_seconds: int = 180,
    max_output_bytes: int = 256 * 1024**2, cpu_threads: int = 2,
) -> dict[str, Any]:
    """Execute a fresh snapshot folder and return audited native engineering outputs."""
    names = _snapshot_names(snapshot)
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if len(payload.encode()) > 8 * 1024**2:
        payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    if len(payload.encode()) > 8 * 1024**2:
        raise ValueError("Snapshot exceeds the bounded 8 MiB input limit")
    root = project_root.resolve(strict=True)
    directory = _inside(output_dir, root)
    directory.mkdir(parents=True, exist_ok=False)
    input_path = directory / "snapshot.json"
    input_path.write_text(payload, encoding="utf-8")
    source_sha = _sha(input_path)
    outputs = directory / "outputs"
    outputs.mkdir()
    manifest_path = directory / "execution_manifest.json"
    result: dict[str, Any] = {
        "runtime_adapter_version": RUNTIME_VERSION, "status": "initializing", "manifest_path": str(manifest_path),
        "source_sha256": source_sha, "expected_simulations": names,
        "engineering_accepted": False, "scientific_accepted": False, "registry_modified": False,
        "independent_observations_verified": False, "pkml_outputs": [], "curve_outputs": [],
        "claim_boundary": "Native engineering execution only; source/engine version equivalence and predictive accuracy are not established.",
    }
    _write_json(manifest_path, result)
    try:
        process = run_bounded_r(script=root / _RUNNER, args=[str(input_path), str(outputs)],
                                output_dir=directory, project_root=root, timeout_seconds=timeout_seconds,
                                cpu_threads=cpu_threads, max_output_bytes=max_output_bytes)
        result.update(process=process, runtime_binding=process.get("runtime_binding"))
        result["source_unchanged"] = _sha(input_path) == source_sha
        if process["status"] != "completed":
            result.update(status="failed", stop_reason=process["stop_reason"])
        else:
            pkmls = sorted(outputs.rglob("*.pkml"))
            curves = sorted(outputs.rglob("*-Results.csv"))
            result["pkml_outputs"] = [{"path": str(p.relative_to(directory)), "sha256": _sha(p),
                                       "byte_size": p.stat().st_size} for p in pkmls]
            result["curve_outputs"] = [_curve(p, directory) for p in curves]
            index_path = outputs / (outputs.name + ".csv")
            indexed = []
            if index_path.is_file():
                with index_path.open(encoding="utf-8-sig", newline="") as handle:
                    indexed = [row.get("Sim_id") for row in csv.DictReader(handle)]
            native_path = outputs / "native_inventory.json"
            native_values = outputs / "native_parameters.csv"
            if native_path.is_file():
                result["native_inventory"] = json.loads(native_path.read_text())
            native_rows, native_finite = 0, 0
            if native_values.is_file():
                with native_values.open(encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        native_rows += 1
                        if row.get("status") == "evaluated_native_value":
                            try:
                                native_finite += math.isfinite(float(row["value"]))
                            except (KeyError, ValueError):
                                pass
                result["native_parameters"] = {"path": str(native_values.relative_to(directory)), "sha256": _sha(native_values),
                                               "rows": native_rows, "finite_rows": native_finite, "units": "native_base_units"}
            pair_matches = all(p.with_name(p.name.removesuffix("-Results.csv") + ".pkml").is_file() for p in curves)
            checks = {
                "source_unchanged": result["source_unchanged"],
                "simulation_index_matches": len(indexed) == len(set(indexed)) == len(names) and set(indexed) == set(names),
                "artifact_counts_match": len(pkmls) == len(curves) == len(names),
                "pkml_nonempty": bool(pkmls) and all(p.stat().st_size > 0 for p in pkmls),
                "curve_pkml_pairs_match": pair_matches,
                "finite_curves": bool(curves) and all(c["finite_curve"] for c in result["curve_outputs"]),
                "nonzero_prediction": bool(curves) and all(c["positive_signal_values"] > 0 for c in result["curve_outputs"]),
                "native_values_extracted": native_finite > 0 and native_path.is_file(),
            }
            result.update(checks=checks, engineering_accepted=all(checks.values()))
            result["status"] = "engineering_passed" if result["engineering_accepted"] else "engineering_failed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", engineering_accepted=False)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            _write_json(manifest_path, result, replace=True)
            raise
    _write_json(manifest_path, result, replace=True)
    return result
