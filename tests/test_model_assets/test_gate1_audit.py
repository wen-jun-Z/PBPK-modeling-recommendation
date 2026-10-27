from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from pbpk_kb.model_assets.gate1_audit import (
    _iter_jsonl,
    _process_participant_errors,
    _sha256_file,
    write_gate1_audit,
)


def test_gate1_audit_helpers_read_plain_and_gzip_jsonl(
    tmp_path: Path,
) -> None:
    rows = [{"id": "a"}, {"id": "b"}]
    plain = tmp_path / "rows.jsonl"
    compressed = tmp_path / "rows.jsonl.gz"
    payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    )
    plain.write_text(payload, encoding="utf-8")
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        handle.write(payload)

    assert list(_iter_jsonl(plain)) == rows
    assert list(_iter_jsonl(compressed)) == rows
    assert len(_sha256_file(plain)) == 64


def test_gate1_audit_report_is_written_atomically(tmp_path: Path) -> None:
    report = {"accepted": True, "checks": {"coverage": True}}
    output = tmp_path / "gate1_audit.json"

    checksum = write_gate1_audit(output, report)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert checksum == _sha256_file(output)
    assert not output.with_suffix(".json.tmp").exists()


def test_process_participant_audit_distinguishes_names_from_ids() -> None:
    scientific_name = SimpleNamespace(
        identifier="Hydroxy-Itraconazole",
        name="Hydroxy-Itraconazole",
        component_type="compound",
        resolution_status="resolved_by_component_name",
    )
    record = SimpleNamespace(
        parser_version="1.2.0",
        educt_identifiers=[],
        educts=[],
        educt_participants=[],
        product_identifiers=[],
        products=[],
        product_participants=[],
        modifier_identifiers=["Hydroxy-Itraconazole"],
        modifiers=["Hydroxy-Itraconazole"],
        modifier_participants=[scientific_name],
        unresolved_partner_identifiers=[],
    )

    assert _process_participant_errors(record) == []

    opaque_id = "LslFaZsuYUqJvhMBiG-hSg"
    record.modifier_identifiers = [opaque_id]
    record.modifiers = [opaque_id]
    record.modifier_participants = [
        SimpleNamespace(
            identifier=opaque_id,
            name=opaque_id,
            component_type="compound",
            resolution_status="resolved_by_component_id",
        )
    ]

    assert _process_participant_errors(record) == [
        "modifier_component_id_used_as_name"
    ]
