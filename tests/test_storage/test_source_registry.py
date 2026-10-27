from pathlib import Path

from pbpk_kb.storage.import_processed_evidence import EvidenceImporter


def test_source_registry_upsert_preserves_runtime_version_when_yaml_has_nulls() -> None:
    source = Path("pbpk_kb/storage/source_registry.py").read_text(encoding="utf-8")

    assert "source_version = COALESCE(EXCLUDED.source_version, source_registry.source_version)" in source
    assert "checksum = COALESCE(EXCLUDED.checksum, source_registry.checksum)" in source


def test_processed_import_refreshes_traceable_source_snapshot_metadata() -> None:
    importer = object.__new__(EvidenceImporter)
    statements: list[str] = []
    importer._exec_sql = statements.append

    importer._refresh_source_snapshot_metadata()

    assert len(statements) == 1
    sql = statements[0]
    assert "('pmc_oa'::text, 'pubmed'::text)" in sql
    assert "('openfda_drug_label'::text, 'drugs_fda'::text)" in sql
    assert "string_agg(" in sql
    assert "digest(" in sql
    assert "raw.raw_checksum" in sql
    assert "UPDATE source_registry" in sql

from pathlib import Path

from pbpk_kb.config.sources import load_source_registry


def test_source_registry_loads_priority_sources() -> None:
    sources = load_source_registry(Path("pbpk_kb/config/sources.yaml"))
    source_ids = {source.source_id for source in sources}

    assert len(sources) >= 18
    assert {
        "osp_pbpk_model_library",
        "pkdb",
        "pubmed",
        "dailymed",
        "pubchem",
    }.issubset(source_ids)
    assert all(source.access_url for source in sources)
    assert all(source.connector.endswith(".py") for source in sources)
