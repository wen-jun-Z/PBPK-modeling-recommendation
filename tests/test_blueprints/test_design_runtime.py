"""Only synthetic files and tiny Python worker processes; never starts R/PK-Sim."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pbpk_kb.blueprints import design_runtime as runtime


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    script = root / "src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R"
    script.parent.mkdir(parents=True)
    script.write_text("# synthetic worker fixture only\n")
    return root


def _snapshot() -> dict:
    return {"Version": 80, "Individuals": [{"Name": "Person", "Seed": 17, "OriginData": {"Species": "Human"}}],
            "Compounds": [{"Name": "ToyDrug"}], "Simulations": [{"Name": "ToySimulation", "Model": "4Comp", "Individual": "Person"}]}


def _fake_binding(root: Path) -> dict:
    return {"locked_resources": [], "actual_engine_version": "12.3.137.0",
            "source_definition_version": "12.3.173", "database_matches_source": True,
            "source_runtime_version_match": False, "source_engine_equivalence": "not_verified"}


def _outputs(path: Path, *, nonfinite: bool = False) -> None:
    (path / "ToySimulation.pkml").write_text("<synthetic_not_a_real_model />")
    last = "nan" if nonfinite else "1.0"
    (path / "ToySimulation-Results.csv").write_text(
        "IndividualId,Time [h],Organism|ToyDrug|Concentration [mg/l]\n0,0,0\n0,1," + last + "\n")
    (path / "outputs.csv").write_text("Sim_id\nToySimulation\n")
    (path / "native_parameters.csv").write_text(
        "model_file,parameter_path,value,unit,dimension,status\n"
        "ToySimulation.pkml,Organism|Liver|Volume,1.0,l,Volume,evaluated_native_value\n")
    (path / "native_inventory.json").write_text(json.dumps({"finite_parameter_rows": 1, "scientific_accepted": False}))


def _mock_success(monkeypatch: pytest.MonkeyPatch, *, nonfinite: bool = False, mutate_input: bool = False) -> None:
    def fake_run(**kwargs: object) -> dict:
        args = kwargs["args"]
        _outputs(Path(args[1]), nonfinite=nonfinite)
        if mutate_input:
            Path(args[0]).write_text("{}")
        return {"status": "completed", "returncode": 0, "stop_reason": None,
                "runtime_binding": _fake_binding(kwargs["project_root"])}
    monkeypatch.setattr(runtime, "run_bounded_r", fake_run)


def test_snapshot_execution_validates_artifacts_without_counting_index_as_curve(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_success(monkeypatch)
    snapshot = _snapshot()
    original = copy.deepcopy(snapshot)
    result = runtime.execute_snapshot(snapshot=snapshot, project_root=fake_project, output_dir=fake_project / "results")
    assert snapshot == original
    assert result["engineering_accepted"] is True
    assert result["scientific_accepted"] is False
    assert result["registry_modified"] is False
    assert len(result["curve_outputs"]) == len(result["pkml_outputs"]) == 1
    assert result["native_parameters"]["finite_rows"] == 1
    assert result["runtime_binding"]["source_engine_equivalence"] == "not_verified"
    saved = json.loads(Path(result["manifest_path"]).read_text())
    assert saved == result


def test_nonfinite_curve_is_not_accepted(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_success(monkeypatch, nonfinite=True)
    result = runtime.execute_snapshot(snapshot=_snapshot(), project_root=fake_project, output_dir=fake_project / "results")
    assert result["status"] == "engineering_failed"
    assert result["checks"]["finite_curves"] is False


def test_input_mutation_is_detected(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_success(monkeypatch, mutate_input=True)
    result = runtime.execute_snapshot(snapshot=_snapshot(), project_root=fake_project, output_dir=fake_project / "results")
    assert result["engineering_accepted"] is False
    assert result["checks"]["source_unchanged"] is False


def test_native_exit_zero_without_outputs_is_failure(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "run_bounded_r", lambda **kwargs: {"status": "completed", "returncode": 0, "stop_reason": None})
    result = runtime.execute_snapshot(snapshot=_snapshot(), project_root=fake_project, output_dir=fake_project / "results")
    assert result["engineering_accepted"] is False
    assert result["checks"]["artifact_counts_match"] is False


def test_failed_worker_still_saves_execution_manifest(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "run_bounded_r", lambda **kwargs: {"status": "failed", "returncode": -15, "stop_reason": "EXECUTION_TIMEOUT"})
    result = runtime.execute_snapshot(snapshot=_snapshot(), project_root=fake_project, output_dir=fake_project / "results")
    assert result["status"] == "failed"
    assert result["stop_reason"] == "EXECUTION_TIMEOUT"
    assert Path(result["manifest_path"]).is_file()


def test_output_reuse_and_outside_project_are_rejected(fake_project: Path) -> None:
    existing = fake_project / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        runtime.execute_snapshot(snapshot=_snapshot(), project_root=fake_project, output_dir=existing)
    with pytest.raises(ValueError, match="inside"):
        runtime.execute_snapshot(snapshot=_snapshot(), project_root=fake_project, output_dir=fake_project.parent / "outside")


def test_duplicate_simulation_identifiers_are_rejected_before_output_creation(fake_project: Path) -> None:
    snapshot = _snapshot()
    snapshot["Simulations"] *= 2
    with pytest.raises(ValueError, match="unique"):
        runtime.execute_snapshot(snapshot=snapshot, project_root=fake_project, output_dir=fake_project / "results")
    assert not (fake_project / "results").exists()


def _tiny_worker(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    """Replace only the native executable with a tiny owned Python subprocess."""
    real_popen = subprocess.Popen
    monkeypatch.setattr(runtime, "_runtime_binding", _fake_binding)
    def popen(_command: list[str], **kwargs: object) -> subprocess.Popen:
        return real_popen([sys.executable, "-c", source], **kwargs)
    monkeypatch.setattr(runtime.subprocess, "Popen", popen)


def _run(fake_project: Path, **overrides: object) -> dict:
    output = fake_project / "worker"
    output.mkdir()
    return runtime.run_bounded_r(script=fake_project / runtime._RUNNER, args=[], output_dir=output,
                                 project_root=fake_project, **overrides)


def test_tiny_worker_completes_and_records_pid_without_importing_model(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _tiny_worker(monkeypatch, "print('tiny fixture')")
    result = _run(fake_project)
    assert result["status"] == "completed"
    assert result["returncode"] == 0
    assert result["pid"] > 0
    assert Path(result["log_path"]).read_text().strip() == "tiny fixture"
    assert json.loads(Path(result["process_manifest_path"]).read_text()) == result


def test_timeout_stops_only_owned_tiny_worker(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _tiny_worker(monkeypatch, "import time; time.sleep(10)")
    result = _run(fake_project, timeout_seconds=1)
    assert result["stop_reason"] == "EXECUTION_TIMEOUT"
    assert result["returncode"] != 0
    assert result["status"] == "failed"
    assert result["elapsed_seconds"] < 6


def test_log_cap_prevents_unbounded_log_writes(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _tiny_worker(monkeypatch, "import sys; sys.stdout.write('x' * 400000); sys.stdout.flush()")
    result = _run(fake_project, max_output_bytes=1024**2)
    assert result["stop_reason"] == "LOG_SIZE_LIMIT"
    assert Path(result["log_path"]).stat().st_size <= 1024**2 // 4


def test_native_output_size_limit_stops_tiny_worker(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _tiny_worker(monkeypatch, "from pathlib import Path; import time; Path('worker/large.bin').write_bytes(b'x' * 1500000); time.sleep(10)")
    result = _run(fake_project, max_output_bytes=1024**2)
    assert result["stop_reason"] == "OUTPUT_SIZE_LIMIT"
    assert result["status"] == "failed"


def test_missing_runtime_is_saved_as_blocked(fake_project: Path) -> None:
    result = _run(fake_project)
    assert result["status"] == "blocked"
    assert result["returncode"] is None
    assert Path(result["process_manifest_path"]).is_file()


def test_script_outside_fixed_worker_directory_is_rejected(fake_project: Path) -> None:
    script = fake_project / "arbitrary.R"
    script.write_text("stop('must not run')")
    output = fake_project / "worker"
    output.mkdir()
    with pytest.raises(ValueError, match="fixed project"):
        runtime.run_bounded_r(script=script, args=[], output_dir=output, project_root=fake_project)


def test_invalid_cpu_limit_never_launches_child(fake_project: Path) -> None:
    with pytest.raises(ValueError, match="cpu_threads"):
        _run(fake_project, cpu_threads=0)


def test_existing_project_stop_flag_prevents_child_launch(fake_project, monkeypatch):
    flag = fake_project / 'stop.flag'
    flag.touch()
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **k: pytest.fail('No child may start after project stop'))
    result = _run(fake_project, stop_file=flag)
    assert result['stop_reason'] == 'PROJECT_STOP_REQUESTED'
    assert result['pid'] is None and result['status'] == 'blocked'


def test_project_stop_flag_cancels_only_its_owned_worker(fake_project, monkeypatch):
    _tiny_worker(monkeypatch, "from pathlib import Path; import time; Path('stop.flag').touch(); time.sleep(30)")
    result = _run(fake_project, stop_file=fake_project / 'stop.flag', timeout_seconds=30)
    assert result['stop_reason'] == 'PROJECT_STOP_REQUESTED'
    assert result['returncode'] != 0
    assert result['elapsed_seconds'] < 6


def test_stop_flag_outside_project_is_rejected(fake_project):
    with pytest.raises(ValueError, match='inside'):
        _run(fake_project, stop_file=fake_project.parent / 'not_this_project.flag')
