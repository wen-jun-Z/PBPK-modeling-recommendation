from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os

import pytest


def test_special_population_is_same_drug_without_promoting_ddi_probe():
    from pbpk_kb.storage.import_reference_models import matching_family
    candidate = {"candidate_kind": "special_population", "model_family": "Cefazolin Pregnancy",
                 "identity_bindings": [{"canonical_name": "Cefazolin"}]}
    assert matching_family(candidate) == "Cefazolin"
    candidate.update(candidate_kind="interaction", model_family="Rifampicin-Cefazolin-DDI")
    assert matching_family(candidate) == "Rifampicin-Cefazolin-DDI"


@pytest.mark.skipif(os.environ.get("PBPK_RUN_PROJECT_DB_TESTS") != "1", reason="Requires the explicitly enabled project database")
def test_reference_import_preserves_accepted_rows_and_is_idempotent():
    import psycopg
    from pbpk_kb.config.settings import get_settings
    from pbpk_kb.paths import PROJECT_ROOT
    from pbpk_kb.storage.import_reference_models import register_reference_catalog

    bundle = PROJECT_ROOT / "experiments/model_recommendation/model_expansion_20260909/catalog_candidates.json"
    if not bundle.exists():
        pytest.skip("Public acquisition fixture is not present")
    candidates = json.loads(bundle.read_text())
    selected = [r for r in candidates if r["model_family"] in {"Atorvastatin", "7E3"}]
    assert len(selected) == 2
    with psycopg.connect(get_settings().database_url.replace("postgresql+psycopg://", "postgresql://")) as connection:
        try:
            result = register_reference_catalog(connection, selected)
            assert result["existing_rows_unchanged"]
            assert result["added_counts"]["public_pbpk_model_validation_run"] == 0
            again = register_reference_catalog(connection, selected)
            assert all(v == 0 for v in again["added_counts"].values())
            promoted = json.loads(json.dumps(selected[0]))
            promoted["reusable_candidate"] = True
            with pytest.raises(ValueError, match="cannot confer"):
                register_reference_catalog(connection, [promoted])
        finally:
            connection.rollback()

from pbpk_kb.model_assets.models import (
    ExecutionProbeStatus,
    LicenseDecision,
    LicenseStatus,
    OSPModelVersionAssetReference,
    OSPModelVersionSnapshot,
    ParseStatus,
)
from pbpk_kb.storage.import_public_model_catalog import (
    PublicModelCatalogImportError,
    load_public_model_catalog_bundle,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def model() -> OSPModelVersionSnapshot:
    checksum = "a" * 64
    project_path = "Midazolam/Midazolam-Model.pksim5"
    return OSPModelVersionSnapshot(
        model_id="model:midazolam",
        model_version_id="model-version:midazolam:v12.3.1",
        model_family="Midazolam",
        project_key="b" * 32,
        project_relative_path=project_path,
        project_sha256=checksum,
        source_id="osp_pbpk_model_library",
        repository_url="https://example.test/osp.git",
        repository_commit="c" * 40,
        software="PK-Sim",
        software_version="12.3.173",
        license=LicenseDecision(
            status=LicenseStatus.UNRESOLVED,
            reason_code="LICENSE_FILE_MISSING",
        ),
        static_parse_status=ParseStatus.REGISTERED_NOT_EXECUTABLE,
        execution_probe_status=ExecutionProbeStatus.PASSED,
        pkml_parse_status=ParseStatus.PARSED,
        reusable_candidate=False,
        compounds=["Midazolam"],
        metabolites=[],
        species=["Human"],
        populations=["European"],
        routes=["Oral"],
        formulations=["Solution"],
        enzymes=["CYP3A4"],
        transporters=[],
        compartments=["Liver", "Kidney", "Plasma"],
        process_types=["absorption", "metabolism", "renal_excretion"],
        output_names=["Plasma concentration"],
        simulation_count=1,
        parameter_record_count=10,
        component_record_count=5,
        process_record_count=3,
        report_count=1,
        dataset_reference_count=2,
        validation_metric_count=0,
        assets=[
            OSPModelVersionAssetReference(
                asset_kind="pksim_project",
                relative_path=project_path,
                sha256=checksum,
                source_location=f"GITHUB_BLOB:commit/{project_path}",
            )
        ],
        report_paths=["Midazolam/Midazolam_evaluation_report.md"],
        blocking_reasons=["LICENSE_FILE_MISSING"],
    )


def build_bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "catalog"
    bundle.mkdir()
    model_path = bundle / "model_versions.jsonl"
    simulation_path = bundle / "simulations.jsonl"
    model_path.write_text(
        json.dumps(model().model_dump(mode="json"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    simulation_path.write_text(
        json.dumps(
            {
                "project_key": "b" * 32,
                "simulation_id": "simulation-1",
                "simulation_name": "Midazolam oral",
                "pkml_source": {
                    "relative_path": "exports/s0001/simulation.pkml",
                    "sha256": "d" * 64,
                    "source_location": "OSP_EXECUTION_RESULT:simulation.pkml",
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = bundle / "summary.json"
    write_json(
        summary_path,
        {
            "gate1_model_version_acceptance": True,
            "repository_commit": "c" * 40,
            "model_version_count": 1,
            "simulation_count": 1,
            "output_checksums": {
                "model_versions.jsonl": sha256(model_path),
                "simulations.jsonl": sha256(simulation_path),
            },
        },
    )
    audit_path = tmp_path / "audit.json"
    write_json(
        audit_path,
        {
            "accepted": True,
            "input_checksums": {
                "model_version_summary": sha256(summary_path)
            },
        },
    )
    return bundle, audit_path


def test_loads_checksum_locked_retrieval_only_catalog(tmp_path: Path) -> None:
    bundle, audit = build_bundle(tmp_path)

    result = load_public_model_catalog_bundle(
        bundle_directory=bundle,
        gate1_audit_path=audit,
    )

    assert len(result.models) == 1
    assert len(result.simulations) == 1
    assert result.models[0].license.status == LicenseStatus.UNRESOLVED
    assert result.models[0].reusable_candidate is False
    assert result.models[0].execution_probe_status == ExecutionProbeStatus.PASSED


def test_rejects_catalog_content_after_checksum_drift(tmp_path: Path) -> None:
    bundle, audit = build_bundle(tmp_path)
    with (bundle / "simulations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(
        PublicModelCatalogImportError,
        match="Catalog checksum for simulations.jsonl",
    ):
        load_public_model_catalog_bundle(
            bundle_directory=bundle,
            gate1_audit_path=audit,
        )
