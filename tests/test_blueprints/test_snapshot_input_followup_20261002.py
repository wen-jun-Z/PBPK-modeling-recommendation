"""Staged input encoding regression; all native boundaries are Python mocks."""
from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = ROOT / "experiments/model_recommendation/execution_validation_20261001/engineering_amendments/snapshot_input_followup_20261002"
LIVE = ROOT / "src/pbpk_kb/blueprints/design_runtime.py"
STAGED = AMENDMENT / "staged/src/pbpk_kb/blueprints/design_runtime.py"
LIMIT = 8 * 1024**2


@pytest.fixture
def runtime(monkeypatch):
    module = types.ModuleType("staged_snapshot_input_runtime")
    module.__file__ = str(LIVE)
    exec(compile(STAGED.read_text(), str(LIVE), "exec"), module.__dict__)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("No native or other child process allowed"))
    monkeypatch.setattr(module, "run_bounded_r", lambda **k: pytest.fail("Native boundary must be explicitly mocked"))
    return module


def snapshot():
    return {"Version": 80, "Simulations": [{"Name": "ToySimulation", "Model": "4Comp"}]}


def compact(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def mock_boundary(runtime, monkeypatch, *, mutate=False, nonfinite=False):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        source, outputs = map(Path, kwargs["args"])
        assert source.is_file()
        (outputs / "ToySimulation.pkml").write_text("<synthetic_not_a_real_model />")
        last = "nan" if nonfinite else "1.0"
        (outputs / "ToySimulation-Results.csv").write_text(
            "IndividualId,Time [h],Organism|ToyDrug|Concentration [mg/l]\n0,0,0\n0,1," + last + "\n")
        (outputs / "outputs.csv").write_text("Sim_id\nToySimulation\n")
        (outputs / "native_parameters.csv").write_text(
            "model_file,parameter_path,value,unit,dimension,status\n"
            "ToySimulation.pkml,Organism|Liver|Volume,1.0,l,Volume,evaluated_native_value\n")
        (outputs / "native_inventory.json").write_text('{"scientific_accepted": false}')
        if mutate:
            source.write_text("{}")
        return {"status": "completed", "returncode": 0, "stop_reason": None}

    monkeypatch.setattr(runtime, "run_bounded_r", run)
    return calls


def execute(runtime, tmp_path, value):
    return runtime.execute_snapshot(snapshot=value, project_root=tmp_path,
                                    output_dir=tmp_path / "execution", timeout_seconds=600,
                                    cpu_threads=2, max_output_bytes=128 * 1024**2)


def test_compact_large_snapshot_preserves_values_and_engineering_gates(runtime, monkeypatch, tmp_path):
    value = snapshot()
    value["SyntheticPadding"] = [{"Ordered": ["é", "温度", 1.25, None, False]}] * 110000
    assert len((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()) > LIMIT
    assert len(compact(value)) < LIMIT
    calls = mock_boundary(runtime, monkeypatch)
    result = execute(runtime, tmp_path, value)
    written = (tmp_path / "execution/snapshot.json").read_bytes()
    assert written == compact(value)
    assert json.loads(written) == value
    assert result["source_sha256"] == hashlib.sha256(written).hexdigest()
    assert result["source_unchanged"] and result["engineering_accepted"]
    assert result["scientific_accepted"] is False
    assert result["independent_observations_verified"] is False
    assert result["registry_modified"] is False
    assert len(calls) == 1
    assert calls[0]["timeout_seconds"] == 600
    assert calls[0]["cpu_threads"] == 2
    assert calls[0]["max_output_bytes"] == 128 * 1024**2


def test_small_snapshot_keeps_original_pretty_bytes(runtime, monkeypatch, tmp_path):
    value = snapshot()
    value["Labels"] = ["温度", "é", "a\\b", "line\nbreak"]
    mock_boundary(runtime, monkeypatch)
    execute(runtime, tmp_path, value)
    assert (tmp_path / "execution/snapshot.json").read_bytes() == (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


@pytest.mark.parametrize("extra", [0, 1])
def test_compact_utf8_byte_boundary_includes_final_newline(runtime, monkeypatch, tmp_path, extra):
    value = snapshot()
    value["SyntheticPadding"] = ""
    padding = LIMIT - len(compact(value)) + extra
    value["SyntheticPadding"] = "é" * (padding // 2) + "x" * (padding % 2)
    assert len(compact(value)) == LIMIT + extra
    calls = mock_boundary(runtime, monkeypatch)
    if extra:
        with pytest.raises(ValueError, match="bounded 8 MiB"):
            execute(runtime, tmp_path, value)
        assert not (tmp_path / "execution").exists()
        assert calls == []
    else:
        result = execute(runtime, tmp_path, value)
        assert result["engineering_accepted"]
        assert (tmp_path / "execution/snapshot.json").stat().st_size == LIMIT
        assert len(calls) == 1


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_rejected_before_any_output(runtime, tmp_path, invalid):
    value = snapshot()
    value["Forbidden"] = invalid
    with pytest.raises(ValueError, match="Out of range float"):
        execute(runtime, tmp_path, value)
    assert not (tmp_path / "execution").exists()


def test_existing_output_remains_exclusive(runtime, tmp_path):
    output = tmp_path / "execution"
    output.mkdir()
    sentinel = output / "snapshot.json"
    sentinel.write_text("preserved")
    with pytest.raises(FileExistsError):
        execute(runtime, tmp_path, snapshot())
    assert sentinel.read_text() == "preserved"


@pytest.mark.parametrize("failure", ["mutation", "nonfinite_output"])
def test_original_failure_checks_remain_required(runtime, monkeypatch, tmp_path, failure):
    mock_boundary(runtime, monkeypatch, mutate=failure == "mutation", nonfinite=failure == "nonfinite_output")
    result = execute(runtime, tmp_path, snapshot())
    assert result["status"] == "engineering_failed"
    assert result["engineering_accepted"] is False
    assert result["scientific_accepted"] is False
    check = "source_unchanged" if failure == "mutation" else "finite_curves"
    assert result["checks"][check] is False


def test_preflight_covers_all_ready_inputs_without_native_or_scoring():
    report = json.loads((AMENDMENT / "ready_payload_preflight.json").read_text())
    assert report["ready_jobs"] == 74
    assert report["snapshot_jobs"] == report["mocked_native_boundary_calls"] == 34
    assert report["complete_pkml_jobs"] == 40
    assert report["snapshot_compact_over_8_mib"] == []
    assert report["source_adaptation_default_over_8_mib"] == []
    assert report["rejected"] == []
    assert report["ode_run"] is False
    assert report["sealed_observations_accessed"] is False
