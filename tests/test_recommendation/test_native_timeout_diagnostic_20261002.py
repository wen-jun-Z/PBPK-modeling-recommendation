"""Diagnostic launch boundaries, intercepted before any native process exists."""
from hashlib import sha256
from pathlib import Path
import tempfile

import pytest

from pbpk_kb.blueprints import design_runtime
from scripts.recommendation_execution_repairs import pipeline


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "src/pbpk_kb/model_runtime/R/native_timeout_diagnostic_20261002.R"
LIMIT_BYTES = 1024 ** 3


class NativeLaunchIntercepted(RuntimeError):
    """The launcher reached Popen; this test deliberately starts no process."""


@pytest.fixture
def scratch():
    temporary_root = ROOT / "runtime/tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native_timeout_boundary_", dir=temporary_root) as directory:
        yield Path(directory)


@pytest.fixture(autouse=True)
def forbid_native_processes(monkeypatch):
    def reject_launch(*args, **kwargs):
        raise NativeLaunchIntercepted("Unexpected launch intercepted; no child created")

    monkeypatch.setattr(design_runtime.subprocess, "Popen", reject_launch)


def test_diagnostic_wrapper_is_outside_the_frozen_pipeline_code_set():
    assert WRAPPER.is_file()
    frozen = set(pipeline.code_files())
    assert ROOT / "src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R" in frozen
    assert WRAPPER not in frozen


def test_registered_diagnostic_launch_preserves_arguments_and_resource_bounds(scratch, monkeypatch):
    request = scratch / "request.json"
    request_bytes = b'{ "action": "run", "diagnostic_boundary_test": true }\n'
    request.write_bytes(request_bytes)
    request_sha = sha256(request_bytes).hexdigest()
    output = scratch / "native"
    output.mkdir()
    captured = {}
    binding = {"locked_resources": [], "diagnostic_boundary_test": True}

    def bind_runtime(root):
        assert root == ROOT
        return binding

    def isolated_environment(root, temporary, cpu_threads):
        assert root == ROOT and temporary == output / "temporary"
        assert cpu_threads == 1
        captured["environment"] = {"TMPDIR": str(temporary), "OMP_NUM_THREADS": "1"}
        return captured["environment"]

    def capture_launch(command, **kwargs):
        captured["command"] = command
        captured["options"] = kwargs
        raise NativeLaunchIntercepted("Diagnostic launch intercepted before process creation")

    monkeypatch.setattr(design_runtime, "_runtime_binding", bind_runtime)
    monkeypatch.setattr(design_runtime, "_environment", isolated_environment)
    monkeypatch.setattr(design_runtime.subprocess, "Popen", capture_launch)
    result = design_runtime.run_bounded_r(
        script=WRAPPER, args=[str(request), request_sha], output_dir=output,
        project_root=ROOT, timeout_seconds=1800, cpu_threads=1,
        max_output_bytes=LIMIT_BYTES,
    )

    assert captured["command"] == [
        str(ROOT / "runtime/ospsuite-env/bin/Rscript"), "--vanilla", str(WRAPPER),
        str(request), request_sha,
    ]
    assert captured["options"]["cwd"] == ROOT
    assert captured["options"]["env"] == captured["environment"]
    assert captured["options"]["start_new_session"] is True
    assert captured["options"]["stdout"] == design_runtime.subprocess.PIPE
    assert captured["options"]["stderr"] == design_runtime.subprocess.STDOUT
    assert result["timeout_seconds"] == 1800
    assert result["cpu_threads"] == 1
    assert result["max_output_bytes"] == LIMIT_BYTES
    assert result["runtime_binding"] == binding
    assert result["pid"] is None and result["returncode"] is None
    assert result["status"] == "blocked" and result["stop_reason"] == "RUNTIME_EXECUTION_ERROR"
    assert result["error"].startswith("NativeLaunchIntercepted:")
    assert request.read_bytes() == request_bytes
    assert sha256(request.read_bytes()).hexdigest() == request_sha


def test_staged_amendment_script_is_rejected_before_runtime_binding(scratch, monkeypatch):
    staged = scratch / "engineering_amendments/staged/native_timeout_diagnostic_20261002.R"
    staged.parent.mkdir(parents=True)
    staged.write_text("# Deliberately outside the fixed R worker directory.\n")
    output = scratch / "native"
    output.mkdir()

    def unexpected_runtime_binding(root):
        pytest.fail("An unauthorized script must be rejected before runtime binding")

    monkeypatch.setattr(design_runtime, "_runtime_binding", unexpected_runtime_binding)
    with pytest.raises(ValueError, match="Only fixed project model_runtime/R scripts"):
        design_runtime.run_bounded_r(
            script=staged, args=[str(scratch / "request.json"), "a" * 64],
            output_dir=output, project_root=ROOT, timeout_seconds=1800,
            cpu_threads=1, max_output_bytes=LIMIT_BYTES,
        )
    assert not (output / "process_manifest.json").exists()
    assert not (output / "execution.log").exists()
