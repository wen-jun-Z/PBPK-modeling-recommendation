from __future__ import annotations

import json
from pathlib import Path

from pbpk_kb.blueprints.caffeine_v2_parameters import (
    build_caffeine_v2_parameter_manifest,
)
from pbpk_kb.model_assets.execution_plan import canonical_sha256


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    payload["checksum"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_manifest(root: Path) -> Path:
    parent = {
        "manifest_version": "caffeine_parameter_manifest_v1",
        "accepted": True,
        "parameters": [
            {
                "parameter_id": parameter_id,
                "name": name,
                "origin": origin,
                "value": value,
                "unit": unit,
                "provenance": [{"artifact_id": "source", "source_location": "x"}],
            }
            for parameter_id, name, origin, value, unit in (
                ("caffeine_molecular_weight", "Molecular weight", "observed", 194.19, "g/mol"),
                ("caffeine_pka_base", "Base pKa", "observed", 0.7, "dimensionless"),
                ("caffeine_water_solubility_25c", "Solubility", "observed", 21600.0, "mg/l"),
                ("caffeine_fraction_unbound_plasma", "Fraction unbound", "observed", 0.68, "fraction"),
                ("caffeine_renal_plasma_clearance", "Renal clearance", "derived", 0.010330512664107847, "ml/min/kg"),
                ("caffeine_effective_lipophilicity", "Lipophilicity", "estimated", -0.1, "Log Units"),
            )
        ],
    }
    _write_manifest(root / "caffeine_parameter_manifest_v1.json", parent)

    sha = "a" * 64
    artifacts = [
        {
            "artifact_id": f"artifact_{index}",
            "role": "test",
            "source_url": "https://example.test/source",
            "relative_path": f"source_{index}",
            "sha256": sha,
            "byte_size": 1,
            "license": "test",
            "redistribution": "public_reference",
            "response_metadata_path": None,
            "response_metadata_sha256": None,
        }
        for index in range(6)
    ]
    fractions = [
        {
            "pathway_id": pathway,
            "fraction": value,
            "reported_sd": sd,
            "source_artifact_id": "lelo_1986_partial_clearance_pubmed_xml",
            "source_location": "abstract",
            "source_text": "locked pathway statement",
            "source_text_sha256": canonical_sha256("locked pathway statement"),
        }
        for pathway, value, sd in (
            ("n3_paraxanthine", 0.839, 0.054),
            ("n1_theobromine", 0.121, 0.041),
            ("n7_theophylline", 0.040, 0.014),
        )
    ]
    domains = [
        {
            "parameter_id": "total_cyp1a2_specific_clearance",
            "unit": "1/min",
            "lower_bound": 0.0,
            "upper_bound": 0.35848049857049,
            "use": "search_domain_only",
            "source_artifact_id": "fendt_2021_parameter_identification_xml",
            "source_location": "specific clearance",
        },
        {
            "parameter_id": "specific_intestinal_permeability_transcellular",
            "unit": "dm/min",
            "lower_bound": 0.0,
            "upper_bound": 3.85111864778853e-05,
            "use": "search_domain_only",
            "source_artifact_id": "fendt_2021_parameter_identification_xml",
            "source_location": "permeability",
        },
    ]
    contamination = [
        {
            "item": f"excluded_{index}",
            "decision": "excluded_from_v2_model_input",
            "reason": "development-data dependent",
            "source_artifact_id": "artifact_0",
        }
        for index in range(4)
    ]
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "builder_version": "caffeine-v2-source-builder-1.0.0",
        "package_version": "caffeine_parameter_sources_v2",
        "generated_at": "2026-08-01T00:00:00Z",
        "artifacts": artifacts,
        "pathway_fractions": fractions,
        "search_domains": domains,
        "contamination_ledger": contamination,
        "checks": {"fixture": True},
        "accepted": True,
    }
    path = root / "source_manifest.json"
    _write_manifest(path, payload)
    return path


def _dataset_manifest(path: Path) -> None:
    payload: dict[str, object] = {
        "dataset_version": "caffeine_dataset_v3",
        "accepted": True,
        "datasets": [
            {
                "dataset_id": "blanchard_1983_iv_5mgkg",
                "split": "calibration",
            },
            {
                "dataset_id": "blanchard_1983_oral_5mgkg",
                "split": "calibration",
            },
            {
                "dataset_id": "wahllander_1989_iv_3mgkg",
                "split": "internal_holdout",
            },
        ],
    }
    _write_manifest(path, payload)


def test_builds_two_parameter_pathway_model_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source_manifest = _source_manifest(source_root)
    dataset_manifest = tmp_path / "dataset.json"
    _dataset_manifest(dataset_manifest)
    output = tmp_path / "parameters"

    manifest = build_caffeine_v2_parameter_manifest(
        source_manifest_path=source_manifest,
        dataset_manifest_path=dataset_manifest,
        output_dir=output,
    )

    assert manifest.accepted is True
    assert len(manifest.parameters) == 11
    assert len(manifest.calibration_starts) == 25
    estimated = {item.parameter_id for item in manifest.parameters if item.calibrate}
    assert estimated == {
        "total_cyp1a2_specific_clearance",
        "specific_intestinal_permeability_transcellular",
    }
    total = next(
        item
        for item in manifest.parameters
        if item.parameter_id == "total_cyp1a2_specific_clearance"
    )
    assert [item.multiplier for item in total.runtime_mappings] == [
        0.839,
        0.121,
        0.04,
    ]
    lipophilicity = next(
        item
        for item in manifest.parameters
        if item.parameter_id == "caffeine_effective_lipophilicity"
    )
    assert lipophilicity.origin == "observed"
    assert lipophilicity.calibrate is False
    assert (output / "caffeine_parameter_manifest.json").is_file()
    assert (output / "calibration_start_grid.json").is_file()
