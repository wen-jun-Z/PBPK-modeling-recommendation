"""The frozen native-thinking judge with verified official causal-conv kernels.

Only kernel dispatch changes. Prompts, weights, budgets, parsing, raw journals,
and acceptance gates remain the parent experiment's responsibility. Floating
point equivalence is measured separately; identical model answers are not assumed.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import sys

import frontier_reasoned_judge as parent

ROOT = parent.ROOT
FAST_ENV = ROOT / "runtime/judge-fast-env"
SETUP = ROOT / "runtime/deployment/frontier_reasoned_fast_setup"
WHEEL_MANIFEST = SETUP / "causal_conv1d_wheel_manifest.json"
IMPORT_VERIFICATION = SETUP / "import_verification.json"
PARENT_SHA256 = "dd5d3ec7390f4a8d435434b4ce7c16cca1eddfefc8a7a7ab0ea66b176078a6f5"
WHEEL_SHA256 = "a1fcd5d022b67ddfc9e1c291f890004c1c081a03d66b532f6eebe31219168e0e"
EXTENSION_SHA256 = "dd99b802fffde36715320745d5da5c8902a81d5e522aebe87d3df085cbd23ad6"
VERSION = "pbpk-native-qwen-thinking-official-causal-conv-v1"


def _scoped(path, within=ROOT):
    path = Path(path)
    if not path.is_file() or not path.resolve().is_relative_to(within.resolve()):
        raise ValueError("Fast backend artifact is absent or outside its project scope")
    return path


def dependency_artifacts():
    """Read-only CPU provenance; callable from the orchestration interpreter."""
    digest = parent.modern.digest
    if digest(_scoped(parent.__file__)) != PARENT_SHA256:
        raise ValueError("Frozen reasoning parent changed")
    manifest = json.loads(_scoped(WHEEL_MANIFEST).read_text())
    imported = json.loads(_scoped(IMPORT_VERIFICATION).read_text())
    wheel = _scoped(SETUP / manifest["name"], SETUP)
    if (manifest["digest"] != "sha256:" + WHEEL_SHA256 or
            manifest["tag"] != "v1.7.0" or wheel.stat().st_size != manifest["size"] or
            digest(wheel) != WHEEL_SHA256):
        raise ValueError("Official causal-conv wheel provenance mismatch")
    files = {}
    for rel, expected in imported["files"].items():
        path = _scoped(ROOT / rel, FAST_ENV)
        actual = digest(path)
        if actual != expected:
            raise ValueError("Installed causal-conv artifact changed")
        files[rel] = actual
    extensions = [key for key in files if Path(key).name.startswith("causal_conv1d_cuda") and key.endswith(".so")]
    if len(extensions) != 1 or files[extensions[0]] != EXTENSION_SHA256:
        raise ValueError("Pinned causal-conv CUDA extension missing")
    package = ROOT / extensions[0]
    for name in ("causal_conv1d_interface.py", "cpp_functions.py"):
        source = _scoped(package.parent / "causal_conv1d" / name, FAST_ENV)
        files[str(source.relative_to(ROOT))] = digest(source)
    if imported.get("causal_conv1d") != "1.7.0" or imported.get("torch") != "2.8.0+cu128":
        raise ValueError("Fast environment version contract changed")
    return {"wheel_manifest_sha256": digest(WHEEL_MANIFEST),
            "import_verification_sha256": digest(IMPORT_VERIFICATION),
            "wheel_sha256": WHEEL_SHA256, "wheel_source": manifest["release_url"],
            "installed_files_sha256": files, "parent_backend_sha256": PARENT_SHA256,
            "declared_versions": {name: imported[name] for name in
                                  ("python", "torch", "cuda_build", "cxx11abi", "causal_conv1d")}}


def qwen_conv_geometry(weights=None):
    weights = weights or json.loads(parent.modern.WEIGHTS.read_text())
    config = _scoped(Path(weights["path"]) / "config.json")
    if parent.modern.digest(config) != weights["sha256"]["config.json"]:
        raise ValueError("Pinned Qwen config changed")
    if weights.get("repo") != parent.modern.REPO or weights.get("revision") != parent.modern.REVISION:
        raise ValueError("Unexpected Qwen revision")
    text = json.loads(config.read_text())["text_config"]
    result = {"linear_layers": text["layer_types"].count("linear_attention"),
              "total_layers": text["num_hidden_layers"],
              "channels": 2 * text["linear_num_key_heads"] * text["linear_key_head_dim"] +
                          text["linear_num_value_heads"] * text["linear_value_head_dim"],
              "width": text["linear_conv_kernel_dim"], "activation": text["hidden_act"],
              "dtype": "bfloat16", "cache_state_length": text["linear_conv_kernel_dim"]}
    if result != {"linear_layers": 48, "total_layers": 64, "channels": 10240,
                  "width": 4, "activation": "silu", "dtype": "bfloat16", "cache_state_length": 4}:
        raise ValueError("Actual Qwen convolution geometry differs from the frozen benchmark")
    return result


def dispatcher_functions():
    """Verify the official Transformers dispatch closure, without GPU work."""
    if os.environ.get("USE_HUB_KERNELS") != "0":
        raise ValueError("Only the pinned local causal-conv implementation is allowed")
    import causal_conv1d
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from transformers.integrations import hub_kernels
    if hub_kernels._kernels_enabled:
        raise ValueError("Hub kernel dispatch is unexpectedly enabled")
    functions, metadata = {}, {}
    for name in ("causal_conv1d_fn", "causal_conv1d_update"):
        wrapped = getattr(modeling, name)
        closure = inspect.getclosurevars(wrapped).nonlocals
        implementation = closure.get("implementation")
        if (closure.get("is_new_implementation") is not True or
                implementation is not getattr(causal_conv1d, name) or
                closure.get("package") != "causal_conv1d"):
            raise ValueError("Qwen convolution would use an unverified or reference implementation")
        reference = closure["torch_function"]
        functions[name] = wrapped
        metadata[name] = {"implementation_module": implementation.__module__,
                          "implementation_name": implementation.__name__,
                          "implementation_source_sha256": parent.sha(inspect.getsource(implementation)),
                          "reference_source_sha256": parent.sha(inspect.getsource(reference)),
                          "dispatcher_file_sha256": parent.modern.digest(_scoped(hub_kernels.__file__)),
                          "modeling_file_sha256": parent.modern.digest(_scoped(modeling.__file__)),
                          "optimized_selected": True}
    return functions, metadata


def verify_fast_environment():
    if Path(sys.prefix).resolve() != FAST_ENV.resolve():
        raise ValueError("Fast reasoning must run in the isolated project judge-fast-env")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("USE_HUB_KERNELS") != "0":
        raise ValueError("Offline local-kernel environment required")
    dependencies = dependency_artifacts()
    import torch
    import causal_conv1d
    import causal_conv1d_cuda
    from importlib.metadata import version
    extension = _scoped(causal_conv1d_cuda.__file__, FAST_ENV)
    _scoped(causal_conv1d.__file__, FAST_ENV)
    if (torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8" or
            not torch._C._GLIBCXX_USE_CXX11_ABI or version("causal-conv1d") != "1.7.0" or
            parent.modern.digest(extension) != EXTENSION_SHA256):
        raise ValueError("Active extension ABI or environment differs from the verified installation")
    _, dispatcher = dispatcher_functions()
    return {"environment": str(FAST_ENV.relative_to(ROOT)), "dependencies": dependencies,
            "dispatcher": dispatcher, "gpu_kernel_executed_by_verification": False}


def generation_protocol(weights=None):
    result = parent.generation_protocol(weights)
    result.update(version=VERSION, implementation_sha256=parent.modern.digest(Path(__file__)),
                  reasoning_parent_backend_sha256=PARENT_SHA256,
                  official_causal_conv=dependency_artifacts(), convolution_geometry=qwen_conv_geometry(weights),
                  kernel_equivalence="Numerical/cache validation and full judge admission required; identical labels are not assumed",
                  label_reuse=False)
    return result


class ReasonedGPUJudge(parent.ReasonedGPUJudge):
    def __init__(self, weights, memory_gib=36, loading_callback=None):
        verified = verify_fast_environment()
        geometry = qwen_conv_geometry(weights)
        super().__init__(weights, memory_gib, loading_callback)
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
        layers = [module for module in self.model.modules() if isinstance(module, Qwen3_5GatedDeltaNet)]
        if len(layers) != geometry["linear_layers"]:
            raise ValueError("Loaded Qwen convolution layer count differs from pinned config")
        for layer in layers:
            weight = layer.conv1d.weight
            if (list(weight.shape) != [geometry["channels"], 1, geometry["width"]] or
                    weight.dtype != self.torch.bfloat16 or weight.device.type != "cuda" or
                    layer.activation != geometry["activation"]):
                raise ValueError("Loaded Qwen convolution shape, dtype, device or activation mismatch")
        _, dispatch = dispatcher_functions()
        if dispatch != verified["dispatcher"]:
            raise ValueError("Convolution dispatcher changed during model loading")
        self.metadata.update(backend="native_qwen_two_phase_thinking_official_causal_conv",
                             causal_conv1d="official_1.7.0_cuda_extension", fast_environment=verified,
                             loaded_convolution_layers=len(layers),
                             reasoning_protocol=generation_protocol(weights))


FastReasonedGPUJudge = ReasonedGPUJudge
