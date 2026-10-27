"""History-bound resumption: no native execution, source relaxation or old JSON trust."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from recommendation_execution_repairs import native_partition_resume as m


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def example(tmp_path):
    root = tmp_path
    for filename in m.IMPLEMENTATION_FILES:
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture " + filename)
    executable = root / "runtime/native/Rscript"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"synthetic native executable; never executed")
    db = root / "runtime/native/database"
    db.write_bytes(b"synthetic database")
    binding = {"gpu_used_by_native_solver": False, "actual_engine_version": "fixture",
               "locked_resources": [m._ref(executable, root), m._ref(db, root)]}
    original = {"Version": 77,
                "Compounds": [{"Name": "Drug", "Parameters": [{"Name": "CL", "Value": 1.0}]}],
                "Individuals": [{"Name": "I", "Weight": 70.0}],
                "Simulations": [{"Name": "A", "Parameters": [{"Path": "Drug|flag", "Value": 0}]},
                                {"Name": "B"}, {"Name": "C"}]}
    source = root / "source.json"
    write(source, original)
    provenance = {"source_path": "source.json", "source_sha256": m.migration.sha(source)}
    part = root / "old/partition_001"
    scoped = m.migration.clean_snapshot(original)
    scoped["Simulations"] = scoped["Simulations"][:2]
    write(part / "source_model_only.json", scoped)

    def stage(mode, input_path, directory, **kwargs):
        directory.mkdir(parents=True)
        (directory / "output").mkdir()
        log = directory / "execution.log"
        log.write_text("MIGRATION_STAGE_BEGIN " + mode + "\nMIGRATION_STAGE_FINISHED_WITHOUT_ODE " + mode + "\n")
        manifest = {"status": "completed", "returncode": 0, "stop_reason": None,
                    "runtime_unchanged": True, "registry_modified": False,
                    "script": m.R_SCRIPT, "script_sha256": m.migration.sha(root / m.R_SCRIPT),
                    "cpu_threads": 1, "timeout_seconds": 600,
                    "command": [str(executable), "--vanilla", str(root / m.R_SCRIPT), mode,
                                str(directory / "input"), str(directory / "output")],
                    "process_manifest_path": str(directory / "process_manifest.json"),
                    "log_path": str(log), "runtime_binding": deepcopy(binding)}
        write(directory / "process_manifest.json", manifest)
        ref = m._ref(input_path, root)
        receipt = {"mode": mode, "status": "completed", "stop_reason": None,
                   "input": ref, "copy_before_sha256": ref["sha256"],
                   "copy_after_sha256": ref["sha256"], "original_after_sha256": ref["sha256"],
                   "process": m._ref(directory / "process_manifest.json", root)}
        write(directory / "stage_receipt.json", receipt)
        output = directory / "output" / ("model.pksim5" if mode == "snapshot_to_project" else "model.json")
        if mode == "snapshot_to_project":
            output.write_bytes(b"synthetic saved project")
        else:
            write(output, {**deepcopy(scoped), "Version": 80})
        return output, [m._ref(directory / "stage_receipt.json", root),
                        m._ref(directory / "process_manifest.json", root), m._ref(output, root)]

    project, _ = stage("snapshot_to_project", part / "source_model_only.json", part / "build")
    stage("project_to_snapshot", project, part / "export")
    return root, provenance, part, stage


def refresh_process_hash(part, stage="build"):
    path = part / stage / "stage_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["process"]["sha256"] = m.migration.sha(part / stage / "process_manifest.json")
    write(path, receipt)


def test_completed_chain_authenticates_only_project_and_no_old_json(example):
    root, provenance, part, _ = example
    # Unanchored old JSON must never be consumed, even if corrupt or absent.
    (part / "export/output/model.json").write_text("intentionally invalid old output")
    artifact = m.authenticate_completed_partition(provenance, part, project_root=root)
    assert artifact["simulation_names"] == ["A", "B"]
    assert artifact["native_snapshot"] is None
    assert artifact["status"] == "authenticated_project_pending_reexport"
    assert artifact["scientific_accepted"] is False
    assert artifact["old_export_output_reused"] is False


@pytest.mark.parametrize("mutation", ["global_numeric", "global_bool", "simulation", "version"])
def test_source_partial_exactness_has_no_tolerance_or_type_coercion(example, mutation):
    root, provenance, part, _ = example
    path = part / "source_model_only.json"
    scoped = json.loads(path.read_text())
    if mutation == "global_numeric":
        scoped["Compounds"][0]["Parameters"][0]["Value"] += 1e-14
    elif mutation == "global_bool":
        scoped["Compounds"][0]["Parameters"][0]["Value"] = True
    elif mutation == "simulation":
        scoped["Simulations"][0]["Parameters"][0]["Value"] = 1
    else:
        scoped["Version"] = 80
    write(path, scoped)
    with pytest.raises(ValueError, match="PARTIAL_INPUT_DIFFERS"):
        m.authenticate_completed_partition(provenance, part, project_root=root)


@pytest.mark.parametrize("stage,status", [("build", "failed"), ("export", "running")])
def test_failed_and_unfinished_partitions_are_not_collected(example, stage, status):
    root, provenance, part, _ = example
    path = part / stage / "process_manifest.json"
    process = json.loads(path.read_text())
    process["status"] = status
    write(path, process)
    refresh_process_hash(part, stage)
    result = m.collect_completed_partitions(provenance, [part.parent], project_root=root)
    assert result["partitions"] == []
    assert "NATIVE_STAGE_NOT_COMPLETED" in result["rejected"][0]["reason"]


@pytest.mark.parametrize("target", ["source.json", "old/partition_001/build/output/model.pksim5",
                                    "runtime/native/database", m.R_SCRIPT])
def test_source_project_runtime_or_worker_drift_is_rejected(example, target):
    root, provenance, part, _ = example
    (root / target).write_bytes((root / target).read_bytes() + b" drift")
    with pytest.raises(ValueError, match="SHA_CHANGED|NATIVE_WORKER_CHANGED"):
        m.authenticate_completed_partition(provenance, part, project_root=root)


def test_missing_next_stage_hash_anchor_is_not_salvaged_from_filename(example):
    root, provenance, part, _ = example
    (part / "export/stage_receipt.json").unlink()
    result = m.collect_completed_partitions(provenance, [part.parent], project_root=root)
    assert result["partitions"] == []


def test_new_export_gets_complete_output_hash_and_remains_pending(example):
    root, provenance, part, stage = example
    artifact = m.authenticate_completed_partition(provenance, part, project_root=root)
    result = m.resume_partition(artifact, root / "new/partition_001", stage_runner=stage, project_root=root)
    verified = m.verify_resume_certificate(result["certificate"], project_root=root)
    assert verified["native_document"]["Version"] == 80
    assert verified["proof"]["scientific_accepted"] is False
    assert verified["proof"]["final_source_comparison_required"] is True
    assert verified["snapshot"] in verified["proof"]["provenance_refs"]
    path = root / result["snapshot"]["path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA_CHANGED"):
        m.verify_resume_certificate(result["certificate"], project_root=root)


def test_changed_project_after_inspection_and_stop_before_creation(example):
    root, provenance, part, stage = example
    artifact = m.authenticate_completed_partition(provenance, part, project_root=root)
    stop = root / "stop.requested"
    stop.touch()
    destination = root / "must_not_exist"
    with pytest.raises(InterruptedError):
        m.resume_partition(artifact, destination, stage_runner=stage, project_root=root, stop_file=stop)
    assert not destination.exists()
    (root / artifact["native_project"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA_CHANGED"):
        m.resume_partition(artifact, destination, stage_runner=stage, project_root=root)
    assert not destination.exists()


def test_duplicate_attempts_do_not_duplicate_simulations(example):
    root, provenance, part, _ = example
    result = m.collect_completed_partitions(provenance, [part.parent, part.parent], project_root=root)
    assert len(result["partitions"]) == 1
    assert "OVERLAPPING" in result["rejected"][0]["reason"]
