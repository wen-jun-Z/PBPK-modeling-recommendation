from __future__ import annotations

import os
import hashlib
from pathlib import Path


# Support both the src layout and a directly installed package.
_PACKAGE_PARENT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = _PACKAGE_PARENT.parent if _PACKAGE_PARENT.name == "src" else _PACKAGE_PARENT
SU700_ROOT = Path("/Volumes/SU700")

def configured_path(environment_name: str, default: Path) -> Path:
    value = os.getenv(environment_name, "").strip()
    return Path(value).expanduser() if value else default


def raw_output_root() -> Path:
    if os.getenv("PBPK_RAW_OUTPUT_ROOT"):
        return configured_path("PBPK_RAW_OUTPUT_ROOT", PROJECT_ROOT / "raw")
    default = (
        SU700_ROOT / "pbpk retrieval resul"
        if SU700_ROOT.is_dir()
        else PROJECT_ROOT / "raw"
    )
    return configured_path("PBPK_RAW_OUTPUT_ROOT", default)


def processing_output_root() -> Path:
    if os.getenv("PBPK_PROCESSING_OUTPUT_ROOT"):
        return configured_path("PBPK_PROCESSING_OUTPUT_ROOT", PROJECT_ROOT / "data_process")
    default = (
        SU700_ROOT / "pbpk processing result"
        if SU700_ROOT.is_dir()
        else PROJECT_ROOT / "data_process"
    )
    return configured_path("PBPK_PROCESSING_OUTPUT_ROOT", default)


def recommendation_quality_dir() -> Path:
    default = (
        PROJECT_ROOT
        / "data_process"
        / "final_evidence"
        / "run_20260713T163840Z"
        / "07_llm_quality_validation"
    )
    return configured_path("PBPK_RECOMMENDATION_QUALITY_DIR", default)


def recommendation_model_cache_dir() -> Path:
    default = PROJECT_ROOT / "runtime" / "hf_cache" / "embedding"
    return configured_path("PBPK_RECOMMENDATION_MODEL_CACHE_DIR", default)


def model_matching_root() -> Path:
    artifact_root = PROJECT_ROOT / "experiments" / "model_recommendation" / "runs"
    default = (
        artifact_root
        if artifact_root.is_dir()
        else PROJECT_ROOT / "data_process" / "model_matching"
    )
    return configured_path("PBPK_MODEL_MATCHING_ROOT", default)


LEGACY_PATH_PREFIXES = (
    (
        Path("/root/autodl-tmp/pbpk_stage1_public_kb_ui_plan_updated_sync"),
        PROJECT_ROOT,
    ),
    (
        Path("/root/autodl-tmp/pbpk_stage1_public_kb_ui_plan_updated"),
        PROJECT_ROOT,
    ),
    (
        Path("/root/autodl-tmp/pbpk-model-assets/model_matching"),
        model_matching_root(),
    ),
    (
        Path("/root/autodl-tmp/pbpk-model-assets"),
        PROJECT_ROOT / "artifacts" / "pbpk-model-assets",
    ),
    (Path("/root/autodl-tmp/hf_cache"), PROJECT_ROOT / "runtime" / "hf_cache"),
    (Path("/root/autodl-tmp/models"), PROJECT_ROOT / "runtime" / "models"),
    (
        Path("/root/autodl-tmp/external"),
        PROJECT_ROOT / "dependencies" / "external",
    ),
    (
        Path("/root/autodl-tmp/ncbi-BioREx"),
        PROJECT_ROOT / "dependencies" / "ncbi-BioREx",
    ),
    (
        Path("/root/autodl-tmp/osp-source"),
        PROJECT_ROOT / "artifacts" / "osp-source",
    ),
)


def resolve_recorded_path(value: str | Path, *, base: Path | None = None) -> Path:
    """Resolve immutable manifest paths without rewriting their provenance."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        return ((base or PROJECT_ROOT) / path).resolve()
    for legacy_prefix, current_prefix in LEGACY_PATH_PREFIXES:
        try:
            suffix = path.relative_to(legacy_prefix)
        except ValueError:
            continue
        return (current_prefix / suffix).resolve()
    # In the isolated deployment, unknown historical locations are provenance,
    # not permission to inspect another account's filesystem.
    if os.getenv("PBPK_PROJECT_BOUNDARY") == "1" and not path.is_relative_to(PROJECT_ROOT):
        digest = hashlib.sha256(str(path).encode()).hexdigest()
        return PROJECT_ROOT / "runtime/deployment/unavailable_recorded_assets" / digest / path.name
    try:
        if path.exists():
            return path
    except OSError:
        pass
    return path
