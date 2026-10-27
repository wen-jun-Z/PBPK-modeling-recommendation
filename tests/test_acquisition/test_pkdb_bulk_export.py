from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from pbpk_kb.acquisition.pkdb_bulk_export import (
    EXPECTED_TABLES,
    PKDBBulkExportError,
    inspect_pkdb_bulk_export,
)


def _write_export(path: Path, *, blank_observations: bool = False) -> None:
    values = {
        "studies": "PKDB00001",
        "groups": "10",
        "individuals": "20",
        "interventions": "30",
        "outputs": "40",
        "timecourses": "50",
        "scatters": "60",
    }
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for resource, (filename, key) in EXPECTED_TABLES.items():
            if blank_observations and resource in {"outputs", "timecourses"}:
                archive.writestr(filename, '""\n')
            else:
                archive.writestr(filename, f"{key}\n{values[resource]}\n")
        archive.writestr("info_nodes.csv", "sid\nroot\n")
        archive.writestr("README.md", "PK-DB export\n")
        archive.writestr("TERMS_OF_USE.md", "Terms\n")


def _filter_counts() -> dict[str, int | str]:
    return {"uuid": "test-uuid", **{resource: 1 for resource in EXPECTED_TABLES}}


def test_pkdb_export_accepts_matching_unique_object_counts() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "export.zip"
        _write_export(archive)

        audit = inspect_pkdb_bulk_export(
            archive_path=archive,
            output_directory=root / "inspection",
            filter_payload=_filter_counts(),
            compound="caffeine",
            source_url="https://pk-db.com/api/v1/filter/?download=true",
        )

        assert audit["accepted_for_observation_processing"] is True
        assert audit["count_mismatches"] == []
        assert audit["table_audits"]["timecourses"]["actual_unique_object_count"] == 1
        assert (root / "inspection" / "members" / "outputs.csv").is_file()
        assert (root / "inspection" / "export_audit.json").is_file()


def test_pkdb_export_rejects_http_success_with_blank_observation_tables() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "export.zip"
        _write_export(archive, blank_observations=True)

        audit = inspect_pkdb_bulk_export(
            archive_path=archive,
            output_directory=root / "inspection",
            filter_payload=_filter_counts(),
            compound="caffeine",
            source_url="https://pk-db.com/api/v1/filter/?download=true",
        )

        assert audit["accepted_for_observation_processing"] is False
        assert "outputs_observation_table_missing" in audit["gap_codes"]
        assert "timecourses_observation_table_missing" in audit["gap_codes"]
        assert audit["table_audits"]["outputs"]["actual_unique_object_count"] == 0


def test_pkdb_export_rejects_path_traversal_without_writing_outside_target() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "malicious.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED) as payload:
            payload.writestr("../escape.txt", "unsafe")

        with pytest.raises(PKDBBulkExportError, match="traversal"):
            inspect_pkdb_bulk_export(
                archive_path=archive,
                output_directory=root / "inspection",
                filter_payload=_filter_counts(),
                compound="caffeine",
                source_url="https://pk-db.com/api/v1/filter/?download=true",
            )

        assert not (root / "escape.txt").exists()
        assert not (root / "inspection").exists()


def test_verified_empty_export_is_a_source_gap_but_corruption_and_other_failures_are_not(tmp_path):
    import json
    from hashlib import sha256
    from pbpk_kb.acquisition.source_recovery import verified_pkdb_observation_gap
    archive = tmp_path / "export.zip"
    _write_export(archive, blank_observations=True)
    url = "https://pk-db.com/api/v1/filter/?studies__substance=caffeine&studies__licence__in=open&download=true"
    inspect_pkdb_bulk_export(archive_path=archive, output_directory=tmp_path / "inspection",
                            filter_payload=_filter_counts(), compound="caffeine", source_url=url)
    path = tmp_path / "inspection/export_audit.json"
    def asset(identifier, key, file):
        return {"asset_id": identifier, "request_key": key, "compound": "caffeine", "source_id": "pkdb",
                "url": url, "storage_path": str(file), "checksum": sha256(file.read_bytes()).hexdigest(), "status_code": 200}
    assets = [asset("archive", "open_bulk_export", archive), asset("audit", "open_bulk_export_audit", path)]
    assets[1]["metadata"] = {"export_asset_id": "archive"}
    failure = {"request_id": "failed-export", "request_key": "open_bulk_export_incomplete",
               "source_id": "pkdb", "compound": "caffeine", "url": url}
    proof = verified_pkdb_observation_gap(failure, assets, root=tmp_path)
    assert proof["status"] == "source_gap_metadata_only"
    assert proof["accepted_for_observation_processing"] is False
    assert verified_pkdb_observation_gap({**failure, "request_key": "open_bulk_export_invalid"}, assets, root=tmp_path) is None
    assert verified_pkdb_observation_gap({**failure, "compound": "metoprolol"}, assets, root=tmp_path) is None
    content = path.read_bytes()
    audit = json.loads(content)
    audit["count_mismatches"][0]["resource"] = "groups"
    path.write_text(json.dumps(audit))
    assets[1]["checksum"] = sha256(path.read_bytes()).hexdigest()
    assert verified_pkdb_observation_gap(failure, assets, root=tmp_path) is None
    path.write_bytes(content)
    assets[1]["checksum"] = sha256(content).hexdigest()
    archive.write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="audited bytes"):
        verified_pkdb_observation_gap(failure, assets, root=tmp_path)
