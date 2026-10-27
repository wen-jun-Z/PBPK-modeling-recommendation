"""Append source-backed, unexecuted model candidates without changing accepted runs.

This is deliberately separate from the execution-validated Gate 1 importer.
No validation labels, simulation outputs or numerical evidence are manufactured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from pbpk_kb.config.settings import get_settings
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.public_collection import PROJECT_ROOT, write_json
from pbpk_kb.model_assets.snapshot_catalog import name_key, prepare_snapshot_catalog

SOURCE = "osp_pbpk_model_library"


def _id(kind: str, *parts: str) -> UUID:
    return uuid5(NAMESPACE_URL, "pbpk-reference-catalog:" + json.dumps([kind, *parts], ensure_ascii=True))


def matching_family(candidate: dict) -> str:
    """Population-specific model records still belong to the parent drug family."""
    if candidate["candidate_kind"] == "interaction":
        return candidate["model_family"]
    if candidate.get("model_coverage_target"):
        return candidate["model_coverage_target"]
    names = {r["canonical_name"] for r in candidate["identity_bindings"]}
    if len(names) == 1 and (candidate["candidate_kind"] == "special_population"
                           or candidate["model_family"].casefold().endswith("-pediatrics")):
        return next(iter(names))
    return candidate["model_family"]


def _insert(connection, table: str, values: dict) -> bool:
    query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING RETURNING 1").format(
        sql.Identifier(table), sql.SQL(",").join(map(sql.Identifier, values)),
        sql.SQL(",").join(sql.Placeholder() for _ in values),
    )
    return connection.execute(query, list(values.values())).fetchone() is not None


def catalog_counts(connection):
    tables = ("compound_identity", "public_pbpk_model_registry", "public_pbpk_model_version",
              "public_pbpk_model_asset", "public_pbpk_model_validation_run")
    return {table: connection.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()[0] for table in tables}


def catalog_fingerprints(connection):
    return {table: dict(connection.execute(sql.SQL("SELECT {}::text, md5(row_to_json(t)::text) FROM {} t ORDER BY {} ").format(
        sql.Identifier(key), sql.Identifier(table), sql.Identifier(key))).fetchall())
        for table, key in (("compound_identity", "compound_id"), ("public_pbpk_model_registry", "model_id"),
                           ("public_pbpk_model_version", "model_version_id"),
                           ("public_pbpk_model_validation_run", "validation_run_id"))}


def validate_reference_descriptor(candidate: dict):
    if (candidate["execution_probe_status"] != "not_run" or candidate["reusable_candidate"] is not False
            or candidate["pkml_parse_status"] != "registered_not_executable"
            or candidate["static_parse_status"] != "registered_not_executable"):
        raise ValueError("Reference registration cannot confer execution or reuse acceptance")
    if not candidate["compounds"] or candidate["simulation_count"] < 1:
        raise ValueError("A candidate needs a model with compounds and simulations")
    project_found = False
    for asset in candidate["assets"]:
        path = PROJECT_ROOT / asset["relative_path"]
        if not path.resolve().is_relative_to(PROJECT_ROOT.resolve()):
            raise ValueError("Model asset escaped the project")
        if sha256_file(path) != asset["sha256"]:
            raise ValueError("Model asset checksum mismatch")
        if asset["asset_kind"] in {"pksim_snapshot", "pksim_project"} and asset["relative_path"] == candidate["project_relative_path"]:
            project_found = asset["sha256"] == candidate["project_sha256"]
    if not project_found:
        raise ValueError("No verified model snapshot")


def register_reference_catalog(connection, candidates: list[dict]) -> dict:
    """Caller owns the transaction; use one transaction for validation and import."""
    before = catalog_counts(connection)
    fingerprints = catalog_fingerprints(connection)
    identities = {}
    for compound_id, preferred, normalized, synonyms in connection.execute(
            "SELECT compound_id, preferred_name::text, normalized_name::text, synonyms FROM compound_identity"):
        for name in (preferred, normalized, *(synonyms or [])):
            if name:
                identities.setdefault(name_key(name), set()).add(compound_id)
    families = {}
    for model_id, model_name, external in connection.execute(
            "SELECT model_id,model_name,external_model_id FROM public_pbpk_model_registry WHERE source_id = %s", (SOURCE,)):
        key = name_key(model_name.removesuffix(" PBPK model"))
        if key in families:
            raise ValueError("Existing model family is ambiguous: " + key)
        families[key] = (model_id, external)
    seen_sha = set()
    seen_semantic = set()
    for checksum, metadata in connection.execute("SELECT project_sha256,metadata_json FROM public_pbpk_model_version"):
        seen_sha.add(checksum)
        if (metadata or {}).get("semantic_sha256"):
            seen_semantic.add(metadata["semantic_sha256"])
    added, skipped = [], []
    for candidate in candidates:
        validate_reference_descriptor(candidate)
        if candidate["project_sha256"] in seen_sha or candidate["semantic_sha256"] in seen_semantic:
            skipped.append({"path": candidate["project_relative_path"], "reason": "identical_registered_model_content"})
            continue
        source_location = candidate["assets"][0]["source_location"]
        administered_ids = {}
        for name in sorted({b["canonical_name"] for b in candidate["identity_bindings"]}):
            key = name_key(name)
            if len(identities.get(key, set())) > 1:
                raise ValueError("Compound identity is ambiguous: " + name)
            if key not in identities:
                compound_id = _id("compound", key)
                _insert(connection, "compound_identity", {
                    "compound_id": compound_id, "preferred_name": name, "normalized_name": name.casefold(),
                    "synonyms": [], "source_id": SOURCE,
                    "source_location": source_location + "#" + candidate["source_pointers"]["compounds"],
                    "curation_status": "machine_extracted",
                })
                identities[key] = {compound_id}
            administered_ids[name] = next(iter(identities[key]))
        family_key = name_key(candidate["model_family"])
        primary = next((v for n, v in administered_ids.items() if name_key(n) == family_key), None)
        if primary is None and len(administered_ids) == 1:
            primary = next(iter(administered_ids.values()))
        metadata = {key: candidate[key] for key in (
            "model_family", "candidate_kind", "semantic_sha256", "parser_version", "snapshot_schema_version",
            "administered_compounds", "identity_bindings", "unresolved_identity_names", "metabolites", "source_pointers", "report_paths", "blocking_reasons")}
        metadata.update(project_key=hashlib.sha256((candidate["repository"] + ":" + candidate["project_sha256"]).encode()).hexdigest()[:32],
                        registration_kind="source_backed_retrieval_only", planned_simulation_names=candidate["simulation_names"],
                        validation_grade="not_executed", scientific_execution_verified=False,
                        source_collection_scope=candidate.get("source_collection_scope", "complete_repository_archive"))
        metadata["model_record_family"] = candidate["model_family"]
        metadata["model_family"] = matching_family(candidate)
        if family_key not in families:
            model_id = _id("family", family_key)
            external = "osp-reference-model:" + family_key
            _insert(connection, "public_pbpk_model_registry", {
                "model_id": model_id, "model_name": candidate["model_family"] + " PBPK model", "compound_id": primary,
                "external_model_id": external, "model_type": "PBPK snapshot", "software": "PK-Sim",
                "species": ", ".join(candidate["species"]), "population": ", ".join(candidate["populations"]),
                "route": ", ".join(candidate["routes"]), "model_source": "public_repository",
                "source_id": SOURCE, "source_variant_id": SOURCE,
                "repository_url": candidate["repository_url"], "repository_commit": candidate["repository_commit"],
                "license_spdx_id": candidate["license"].get("spdx_id"),
                "model_file_path": candidate["project_relative_path"], "model_checksum": candidate["project_sha256"],
                "source_snapshot_checksum": candidate["source_snapshot_sha256"],
                "static_parse_status": "registered_not_executable", "execution_probe_status": "not_run",
                "validation_status": "retrieval_only", "validation_grade": "not_executed", "run_label_grade": 0,
                "ranking_label": "retrieval_only", "reusable_candidate": False,
                "simulation_count": candidate["simulation_count"], "parameter_record_count": 0,
                "component_record_count": 0, "process_record_count": 0,
                "source_location": source_location, "curation_status": "machine_extracted", "metadata_json": Jsonb(metadata),
                "notes": "Source-backed reference only. Snapshot scenarios have not been executed or independently validated.",
            })
            families[family_key] = (model_id, external)
        model_id, external = families[family_key]
        version_id = _id("version", candidate["repository"], candidate["semantic_sha256"])
        values = {
            "model_version_id": version_id, "model_id": model_id,
            "external_model_version_id": "osp-reference-version:" + str(version_id),
            "version_label": candidate["repository_commit"][:12] + " (unexecuted)",
            "repository_url": candidate["repository_url"], "repository_commit": candidate["repository_commit"],
            "project_relative_path": candidate["project_relative_path"], "project_sha256": candidate["project_sha256"],
            "source_snapshot_sha256": candidate["source_snapshot_sha256"], "software": "PK-Sim",
            "software_version": candidate["software_version"], "license_status": candidate["license"]["status"],
            "license_spdx_id": candidate["license"].get("spdx_id"),
            "license_source_location": candidate["license"].get("source_location"),
            "license_source_checksum": candidate["license"].get("source_checksum"),
            "static_parse_status": "registered_not_executable", "execution_probe_status": "not_run",
            "pkml_parse_status": "registered_not_executable", "reusable_candidate": False,
            "simulation_count": candidate["simulation_count"], "parameter_record_count": 0,
            "component_record_count": 0, "process_record_count": 0,
            "report_count": len(candidate["report_paths"]), "dataset_reference_count": candidate["dataset_reference_count"],
            "source_id": SOURCE, "source_variant_id": SOURCE, "source_location": source_location,
            "metadata_json": Jsonb(metadata),
        }
        for key in ("compounds", "species", "populations", "routes", "formulations", "enzymes", "transporters",
                    "compartments", "process_types", "output_names"):
            values[key] = candidate[key]
        inserted = _insert(connection, "public_pbpk_model_version", values)
        if not inserted:
            raise ValueError("Version identity conflict requires review")
        for asset in candidate["assets"]:
            _insert(connection, "public_pbpk_model_asset", {
                "model_asset_id": _id("asset", str(version_id), asset["asset_kind"], asset["relative_path"], asset["sha256"]),
                "model_version_id": version_id, **asset,
            })
        seen_sha.add(candidate["project_sha256"])
        seen_semantic.add(candidate["semantic_sha256"])
        added.append({"model_id": str(model_id), "model_version_id": str(version_id), "model_family": candidate["model_family"],
                      "repository": candidate["repository"], "project_relative_path": candidate["project_relative_path"],
                      "candidate_kind": candidate["candidate_kind"], "administered_compounds": candidate["administered_compounds"]})
    after = catalog_counts(connection)
    current = catalog_fingerprints(connection)
    preserved = all(current[table].get(key) == fingerprint for table, rows in fingerprints.items() for key, fingerprint in rows.items())
    if not preserved:
        raise ValueError("Previously registered rows changed during reference import")
    return {"before": before, "after": after, "added_counts": {k: after[k] - before[k] for k in before},
            "existing_rows_unchanged": preserved, "added_versions": added, "skipped": skipped,
            "scientifically_verified_new_versions": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.directory.resolve()
    if not output.is_relative_to(PROJECT_ROOT.resolve()):
        parser.error("directory must belong to this project")
    print(json.dumps(prepare_snapshot_catalog(output), ensure_ascii=False))
    if args.prepare_only:
        return
    candidates = json.loads((output / "catalog_candidates.json").read_text())
    with psycopg.connect(get_settings().database_url.replace("postgresql+psycopg://", "postgresql://")) as connection:
        connection.execute("SELECT pg_advisory_xact_lock(hashtext('pbpk-reference-catalog-import'))")
        result = register_reference_catalog(connection, candidates)
    # A repeat run must not erase the first import's net growth in the final record.
    history_path = output / "catalog_import.json"
    if history_path.exists():
        prior = json.loads(history_path.read_text())
        result["before"] = prior["before"]
        result["added_counts"] = {k: result["after"][k] - prior["before"][k] for k in result["before"]}
        result["added_versions"] = prior["added_versions"] + result["added_versions"]
        result = {**prior, **result}
    write_json(history_path, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"added_versions", "skipped"}}))


if __name__ == "__main__":
    main()
