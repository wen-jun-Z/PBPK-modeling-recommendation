"""Mock worker tests: no OSP loading, native compilation, solver or GPU call."""
import hashlib
import json

import pytest

from pbpk_kb.blueprints import parameter_batch as adapter


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    pkml = root / "synthetic.pkml"
    pkml.write_text("<synthetic-unit-fixture />")
    script = root / "src/pbpk_kb/model_runtime/R/blueprint_parameter_batch.R"
    script.parent.mkdir(parents=True)
    script.write_text("# Never executed by this test\n")
    monkeypatch.setattr(adapter, "_R_SCRIPT", script)
    settings = {
        "pkml_path": pkml, "parameter_specs": [{"path": "Organism|Liver|CL", "unit": "L/h"}],
        "values": [[1.0], [2.0]], "target_output": "Organism|Plasma|Fixture|Concentration",
        "output_dir": root / "batch", "project_root": root,
    }
    state = {"calls": [], "change": None, "runtime_failure": None}

    def fake_runner(**kwargs):
        state["calls"].append(kwargs)
        request_path = root / "batch/request.json"
        assert kwargs["args"] == [str(request_path), hashlib.sha256(request_path.read_bytes()).hexdigest()]
        request = json.loads(request_path.read_text())
        native = {
            "request_checksum": request["request_checksum"], "pkml_sha256": request["pkml_sha256"],
            "target_output": request["target_output"], "ospsuite_version": "synthetic-version",
            "native_cpu_cores": request["native_cpu_cores"],
            "runs": [{"vector_index": index, "run_id": f"fixture-run-{index}", "values": vector,
                      "time_min": [0, 1, 2], "concentration": [0, 1, 0.5],
                      "time_unit": "min", "unit": "µmol/l", "dimension": "Concentration (molar)"}
                     for index, vector in enumerate(request["values"])],
        }
        if state["change"]:
            state["change"](native)
        if state["runtime_failure"]:
            return {"status": "failed", "returncode": None, "stop_reason": state["runtime_failure"]}
        (root / "batch/native_results.json").write_text(json.dumps(native))
        return {"status": "completed", "returncode": 0, "elapsed_seconds": 0,
                "runtime_binding": {"mock": True}}

    monkeypatch.setattr(adapter, "_run_bounded_r", fake_runner)
    return settings, state


def test_native_run_mapping_and_no_interpolation_are_preserved(setup):
    settings, state = setup
    state["change"] = lambda native: native["runs"].reverse()
    result = adapter.run_parameter_batch(**settings)
    assert result["status"] == "completed"
    assert result["completed_vector_count"] == 2
    assert [row["vector_index"] for row in result["runs"]] == [0, 1]
    assert [row["values"] for row in result["runs"]] == [[1.0], [2.0]]
    assert result["runs"][0]["time_min"] == [0, 1, 2]
    assert result["scientific_status"] == "not_evaluated"
    assert result["interpolation_performed"] is False
    assert state["calls"][0]["cpu_threads"] == 2
    assert state["calls"][0]["timeout_seconds"] == 180
    persisted = json.loads((settings["output_dir"] / "parameter_batch_manifest.json").read_text())
    assert "concentration" not in persisted["runs"][0]
    assert persisted["native_results_file"] == "native_results.json"


@pytest.mark.parametrize("count", [1, 64])
def test_supported_batch_sizes_keep_complete_correspondence(setup, count):
    settings, _ = setup
    settings["values"] = [[float(index)] for index in range(count)]
    result = adapter.run_parameter_batch(**settings)
    assert result["status"] == "completed"
    assert result["completed_vector_count"] == count


def test_native_core_limit_is_explicit_in_hashed_request_and_result(setup):
    settings, state = setup
    result = adapter.run_parameter_batch(**settings, cpu_threads=4)
    request = json.loads((settings['output_dir'] / 'request.json').read_text())
    assert request['native_cpu_cores'] == result['native_cpu_cores'] == 4
    assert state['calls'][0]['cpu_threads'] == 4


@pytest.mark.parametrize('reported', [None, True, 3])
def test_unconfirmed_native_core_limit_invalidates_result(setup, reported):
    settings, state = setup
    state['change'] = lambda native: native.update(native_cpu_cores=reported)
    result = adapter.run_parameter_batch(**settings)
    assert result['status'] == 'failed'
    assert 'core limit' in result['error_detail']


@pytest.mark.parametrize("field,bad", [
    ("values", []), ("values", [[1]] * 65), ("values", [[1, 2]]),
    ("values", [[True]]), ("values", [[float("nan")]]), ("values", [[float("inf")]]),
    ("parameter_specs", []),
    ("parameter_specs", [{"path": "*|CL", "unit": "L/h"}]),
    ("parameter_specs", [{"path": "Organism||CL", "unit": "L/h"}]),
    ("parameter_specs", [{"path": "Organism|CL", "unit": ""}]),
    ("parameter_specs", [{"path": "Organism|CL", "unit": "L/h", "formula": "invented"}]),
    ("target_output", "Organism|*|Concentration"), ("target_output", "bad\npath"),
    ("cpu_threads", True), ("cpu_threads", 0), ("cpu_threads", 65), ("cpu_threads", 33),
    ("timeout_seconds", 0), ("timeout_seconds", 3601),
    ("max_output_bytes", 0), ("max_output_bytes", 100), ("max_output_bytes", 3 * 1024**3),
])
def test_invalid_input_fails_before_creating_output_or_worker(setup, field, bad):
    settings, state = setup
    settings[field] = bad
    with pytest.raises(ValueError):
        adapter.run_parameter_batch(**settings)
    assert not settings["output_dir"].exists()
    assert state["calls"] == []


def test_duplicate_parameter_paths_are_rejected(setup):
    settings, state = setup
    settings["parameter_specs"] *= 2
    settings["values"] = [[1, 2]]
    with pytest.raises(ValueError, match="unique"):
        adapter.run_parameter_batch(**settings)
    assert not state["calls"]


@pytest.mark.parametrize("field", ["pkml_path", "output_dir"])
def test_paths_cannot_escape_project(setup, field):
    settings, state = setup
    outside = settings["project_root"].parent / "outside.pkml"
    outside.write_text("synthetic outside fixture")
    settings[field] = outside
    with pytest.raises(ValueError, match="inside the project"):
        adapter.run_parameter_batch(**settings)
    assert not state["calls"]


def test_symlink_to_nonproject_pkml_is_rejected(setup):
    settings, state = setup
    outside = settings["project_root"].parent / "outside.pkml"
    outside.write_text("synthetic outside fixture")
    linked = settings["project_root"] / "linked.pkml"
    linked.symlink_to(outside)
    settings["pkml_path"] = linked
    with pytest.raises(ValueError, match="inside the project"):
        adapter.run_parameter_batch(**settings)
    assert not state["calls"]


def test_existing_output_is_never_overwritten(setup):
    settings, state = setup
    settings["output_dir"].mkdir()
    preserved = settings["output_dir"] / "existing.txt"
    preserved.write_text("keep")
    with pytest.raises(FileExistsError):
        adapter.run_parameter_batch(**settings)
    assert preserved.read_text() == "keep"
    assert not state["calls"]


@pytest.mark.parametrize("corruption", [
    "count", "duplicate_index", "duplicate_run_id", "parameter_values", "checksum", "pkml_checksum",
    "target", "nan", "nonincreasing_time", "different_lengths", "time_unit", "different_units",
])
def test_invalid_native_result_never_returns_partial_success(setup, corruption):
    settings, state = setup

    def corrupt(native):
        rows = native["runs"]
        if corruption == "count":
            rows.pop()
        elif corruption == "duplicate_index":
            rows[1]["vector_index"] = 0
        elif corruption == "duplicate_run_id":
            rows[1]["run_id"] = rows[0]["run_id"]
        elif corruption == "parameter_values":
            rows[1]["values"] = [999]
        elif corruption == "checksum":
            native["request_checksum"] = "f" * 64
        elif corruption == "pkml_checksum":
            native["pkml_sha256"] = "f" * 64
        elif corruption == "target":
            native["target_output"] = "another"
        elif corruption == "nan":
            rows[1]["concentration"][1] = float("nan")
        elif corruption == "nonincreasing_time":
            rows[1]["time_min"] = [0, 1, 1]
        elif corruption == "different_lengths":
            rows[1]["concentration"] = [0]
        elif corruption == "time_unit":
            rows[1]["time_unit"] = "h"
        else:
            rows[1]["unit"] = "mg/l"
    state["change"] = corrupt
    result = adapter.run_parameter_batch(**settings)
    assert result["status"] == "failed"
    assert result["error"] == "NATIVE_BATCH_RESULT_INVALID"
    assert result["runs"] == [] and result["completed_vector_count"] == 0


@pytest.mark.parametrize("reason", ["timeout", "output_limit", "process_failed"])
def test_worker_failure_is_explicit_and_has_no_accepted_curves(setup, reason):
    settings, state = setup
    state["runtime_failure"] = reason
    result = adapter.run_parameter_batch(**settings)
    assert result["status"] == "failed" and result["runs"] == []
    assert result["error"] == "NATIVE_BATCH_EXECUTION_FAILED"
    assert result["runtime"]["stop_reason"] == reason


def test_changed_pkml_during_worker_is_rejected(setup):
    settings, state = setup
    state["change"] = lambda native: settings["pkml_path"].write_text("changed fixture")
    result = adapter.run_parameter_batch(**settings)
    assert result["status"] == "failed"
    assert "PKML changed" in result["error_detail"]


def test_postprocessing_respects_directory_byte_limit(setup):
    settings, state = setup
    settings["max_output_bytes"] = 1024**2
    state["change"] = lambda native: native.update(padding="x" * 1024**2)
    result = adapter.run_parameter_batch(**settings)
    assert result["status"] == "failed"
    assert result["error"] in {"BATCH_OUTPUT_BYTE_LIMIT", "NATIVE_BATCH_RESULT_INVALID"}
    assert result["runs"] == []
