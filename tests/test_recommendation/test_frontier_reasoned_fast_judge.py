"""Independent CPU checks of the isolated causal-convolution optimization.

No CUDA execution or model weights: installed native modules are imported only.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
os.environ.update(CUDA_VISIBLE_DEVICES="", USE_HUB_KERNELS="0", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                  HF_HOME=str(ROOT / "runtime/hf_cache/frontier"),
                  HF_HUB_CACHE=str(ROOT / "runtime/hf_cache/frontier/hub"),
                  TRITON_CACHE_DIR=str(ROOT / "runtime/deployment/frontier_reasoned_fast_setup/triton_cache"),
                  TORCHINDUCTOR_CACHE_DIR=str(ROOT / "runtime/deployment/frontier_reasoned_fast_setup/torchinductor_cache"))
sys.path.insert(0, str(ROOT / "scripts"))

import frontier_modern_judge as modern
import frontier_reasoned_judge as original
import frontier_reasoned_fast_judge as fast


@pytest.fixture(scope="module")
def weights():
    return json.loads(modern.WEIGHTS.read_text())


def test_fast_class_keeps_entire_frozen_generation_and_evidence_path():
    assert issubclass(fast.FastReasonedGPUJudge, original.ReasonedGPUJudge)
    for name in ("generate", "_render", "_messages", "_run_stage", "_context", "_append_artifact", "select_batch_size"):
        assert getattr(fast.FastReasonedGPUJudge, name) is getattr(original.ReasonedGPUJudge, name)
    assert modern.digest(Path(original.__file__)) == fast.PARENT_SHA256


def test_wrong_interpreter_rejected_without_loading_model(monkeypatch):
    monkeypatch.setattr(sys, "prefix", str(ROOT / "runtime/judge-env"))
    with pytest.raises((ValueError, RuntimeError)):
        fast.verify_fast_environment()


@pytest.mark.parametrize("name,value", [("USE_HUB_KERNELS", "1"), ("HF_HUB_OFFLINE", "0")])
def test_network_or_hub_kernel_enablement_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises((ValueError, RuntimeError)):
        fast.verify_fast_environment()


def test_actual_isolated_environment_imports_pinned_extension_without_cuda():
    import torch
    import causal_conv1d_cuda

    assert Path(sys.prefix).resolve() == fast.FAST_ENV.resolve()
    assert Path(causal_conv1d_cuda.__file__).resolve().is_relative_to(fast.FAST_ENV)
    assert not torch.cuda.is_initialized()
    verified = fast.verify_fast_environment()
    assert verified["environment"] == "runtime/judge-fast-env"
    assert verified["gpu_kernel_executed_by_verification"] is False
    assert verified["dependencies"]["wheel_sha256"] == fast.WHEEL_SHA256
    extension_hashes = {value for key, value in verified["dependencies"]["installed_files_sha256"].items()
                        if key.endswith(".so")}
    assert extension_hashes == {fast.EXTENSION_SHA256}
    assert not torch.cuda.is_initialized()


def test_dispatcher_selects_real_official_implementation_and_retains_reference():
    import causal_conv1d
    from transformers.models.qwen3_5 import modeling_qwen3_5

    functions, metadata = fast.dispatcher_functions()
    assert set(functions) == {"causal_conv1d_fn", "causal_conv1d_update"}
    for name, dispatcher in functions.items():
        assert dispatcher is getattr(modeling_qwen3_5, name)
        captured = inspect.getclosurevars(dispatcher).nonlocals
        assert captured["implementation"] is getattr(causal_conv1d, name)
        assert captured["torch_function"] is not captured["implementation"]
        assert captured["is_new_implementation"] is True
        assert metadata[name]["optimized_selected"] is True
        assert metadata[name]["implementation_source_sha256"] == original.sha(inspect.getsource(captured["implementation"]))
        assert metadata[name]["reference_source_sha256"] == original.sha(inspect.getsource(captured["torch_function"]))


@pytest.mark.parametrize("name", ["causal_conv1d_fn", "causal_conv1d_update"])
def test_silent_reference_fallback_is_rejected(monkeypatch, name):
    from transformers.models.qwen3_5 import modeling_qwen3_5

    actual = getattr(modeling_qwen3_5, name)
    reference = inspect.getclosurevars(actual).nonlocals["torch_function"]
    monkeypatch.setattr(modeling_qwen3_5, name, reference)
    with pytest.raises(ValueError, match="unverified or reference"):
        fast.dispatcher_functions()


@pytest.mark.parametrize("artifact", ["wheel", "extension", "parent"])
def test_modified_artifact_hash_is_rejected_without_altering_files(monkeypatch, artifact):
    original_digest = modern.digest
    def corrupt(path):
        name = Path(path).name
        if ((artifact == "wheel" and name.endswith(".whl")) or
                (artifact == "extension" and name.startswith("causal_conv1d_cuda") and name.endswith(".so")) or
                (artifact == "parent" and Path(path) == Path(original.__file__))):
            return "0" * 64
        return original_digest(path)
    monkeypatch.setattr(modern, "digest", corrupt)
    with pytest.raises(ValueError, match="changed|mismatch"):
        fast.dependency_artifacts()


def test_qwen_convolution_geometry_is_recomputed_from_pinned_config(weights):
    config = json.loads((Path(weights["path"]) / "config.json").read_text())["text_config"]
    actual = fast.qwen_conv_geometry(weights)
    assert actual["linear_layers"] == sum(layer == "linear_attention" for layer in config["layer_types"]) == 48
    assert actual["total_layers"] == len(config["layer_types"]) == 64
    qk = config["linear_num_key_heads"] * config["linear_key_head_dim"]
    value = config["linear_num_value_heads"] * config["linear_value_head_dim"]
    assert actual["channels"] == qk + qk + value == 10240
    assert actual["width"] == actual["cache_state_length"] == config["linear_conv_kernel_dim"] == 4
    assert actual["activation"] == config["hidden_act"] == "silu"
    assert actual["dtype"] == config["dtype"] == "bfloat16"


def test_unpinned_model_revision_or_config_is_rejected(weights):
    bad_revision = {**weights, "revision": "wrong_revision"}
    with pytest.raises(ValueError, match="revision"):
        fast.qwen_conv_geometry(bad_revision)
    bad_config = {**weights, "sha256": {**weights["sha256"], "config.json": "0" * 64}}
    with pytest.raises(ValueError, match="config changed"):
        fast.qwen_conv_geometry(bad_config)


def test_fast_protocol_changes_only_dispatch_and_requires_fresh_admission(weights):
    before = original.generation_protocol(weights)
    after = fast.generation_protocol(weights)
    for key, value in before.items():
        if key not in ("version", "implementation_sha256"):
            assert after[key] == value, key
    assert after["phase1"]["max_new_tokens"] == 1024
    assert after["phase2"]["max_new_tokens"] == 96
    assert after["phase2"]["repair_max_new_tokens"] == 160
    assert after["batch_token_budget"] == 32000 and after["gpu_memory_cap_gib"] == 36
    assert after["reasoning_parent_backend_sha256"] == fast.PARENT_SHA256
    assert after["label_reuse"] is False
    assert after["scientific_accuracy_verified"] is False
    assert after["semantic_entailment_verified"] is False
    assert "identical labels are not assumed" in after["kernel_equivalence"]
