from __future__ import annotations

from collections import Counter
from typing import Any


def public_model_request_defaults(compound: str) -> dict[str, Any]:
    normalized = compound.strip().casefold()
    defaults: dict[str, Any] = {
        "species": "human",
        "route": "",
        "formulation": "",
        "population": "",
        "dose": 0.0,
        "dose_unit": "",
        "frequency": "",
        "regimen_mode": "unspecified",
        "model_structure": "unspecified",
        "enzymes": "",
        "transporters": "",
        "endpoints": [],
    }
    if normalized == "rosuvastatin":
        defaults.update(
            {
                "route": "oral",
                "formulation": "Tablet (Rosuvastatin)",
                "dose": 20.0,
                "dose_unit": "mg",
                "frequency": "once daily",
                "regimen_mode": "multiple",
                "endpoints": ["concentration_time_profile"],
            }
        )
    elif normalized == "caffeine":
        defaults.update(
            {
                "route": "oral",
                "formulation": "aqueous solution",
                "dose": 5.0,
                "dose_unit": "mg/kg",
                "frequency": "single dose",
                "regimen_mode": "single",
                "model_structure": "whole_body",
                "enzymes": "CYP1A2",
                "endpoints": ["concentration_time_profile"],
            }
        )
    return defaults


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("payload_json")
    return value if isinstance(value, dict) else {}


def parameter_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        provenance = row.get("provenance_json") or []
        first_source = provenance[0] if provenance else {}
        value = row.get("value")
        initial_value = row.get("initial_value")
        output.append(
            {
                "parameter": row.get("parameter_name"),
                "semantic_id": row.get("semantic_id"),
                "origin": row.get("origin"),
                "value": value if value is not None else initial_value,
                "value_status": (
                    "fixed"
                    if value is not None
                    else "initial estimate"
                    if initial_value is not None
                    else "missing"
                ),
                "unit": row.get("canonical_unit"),
                "blocking": row.get("blocking"),
                "source_id": first_source.get("source_id"),
                "source_location": first_source.get("source_location"),
                "source_checksum": first_source.get("source_checksum"),
            }
        )
    return output


def process_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        str(_payload(row).get("process_type") or "unspecified")
        for row in rows
    )
    return [
        {"process_type": name, "process_count": count}
        for name, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def process_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        payload = _payload(row)
        evidence = payload.get("evidence") or []
        first_source = evidence[0] if evidence else {}
        output.append(
            {
                "name": payload.get("name"),
                "type": payload.get("process_type"),
                "enzyme": payload.get("enzyme_semantic_id"),
                "transporter": payload.get("transporter_semantic_id"),
                "input_compartment": payload.get("input_compartment_id"),
                "output_compartment": payload.get("output_compartment_id"),
                "equation": payload.get("equation_template_id"),
                "source_location": first_source.get("source_location")
                or row.get("source_location"),
            }
        )
    return output


def compartment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": _payload(row).get("name"),
            "type": _payload(row).get("compartment_type"),
            "semantic_id": _payload(row).get("semantic_id"),
            "parent_compartment": _payload(row).get(
                "parent_compartment_id"
            ),
            "source_location": row.get("source_location"),
        }
        for row in rows
    ]


def simulation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scenario": row.get("scenario_id"),
            "repeat": row.get("run_ordinal"),
            "status": row.get("status"),
            "outputs": row.get("output_count"),
            "result_manifest_sha256": row.get(
                "result_manifest_sha256"
            ),
            "started_at": row.get("started_at"),
            "finished_at": row.get("finished_at"),
            "source_location": row.get("source_location"),
        }
        for row in rows
    ]


def artifact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "kind": row.get("artifact_kind"),
            "path": row.get("relative_path"),
            "bytes": row.get("byte_size"),
            "sha256": row.get("sha256"),
            "source_location": row.get("source_location"),
        }
        for row in rows
    ]
