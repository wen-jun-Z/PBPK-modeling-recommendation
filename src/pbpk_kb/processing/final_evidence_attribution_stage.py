from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from pbpk_kb.processing.final_evidence import (
    COMPOUND_ROLES,
    LLMAttributionAgent,
    build_compound_terms,
    decide_attribution,
)
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout


REQUIRED_UPSTREAM_STAGES = (
    "inventory",
    "parse",
    "deterministic_extract",
    "llm_extract",
    "table_recovery",
    "completeness",
)
REQUIRED_PROVENANCE_FIELDS = (
    "evidence_id",
    "run_id",
    "source_id",
    "raw_asset_id",
    "storage_path",
    "checksum",
    "source_location",
    "source_text",
    "source_text_hash",
    "extraction_method",
    "extractor_name",
    "extractor_version",
    "curation_status",
)
REFERENCE_FILES = (
    ("parsed_document_id", "parsed_documents.jsonl", "parsed_document_id"),
    ("chunk_id", "text_chunks.jsonl", "chunk_id"),
    ("parsed_table_id", "parsed_tables.jsonl", "parsed_table_id"),
    ("recovered_table_id", "recovered_tables.jsonl", "recovered_table_id"),
    ("table_row_id", "recovered_table_rows.jsonl", "table_row_id"),
)
SALT_OR_FORM_WORDS = {
    "anhydrous",
    "besylate",
    "calcium",
    "dihydrate",
    "fumarate",
    "hemihydrate",
    "hydrobromide",
    "hydrochloride",
    "hydrate",
    "magnesium",
    "maleate",
    "mesylate",
    "monohydrate",
    "phosphate",
    "potassium",
    "sodium",
    "succinate",
    "sulfate",
    "tartrate",
}
AMBIGUOUS_IDENTITY_SYNONYMS = {"awake", "choice", "coumarins", "result", "stim"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run provenance validation and compound attribution without deduplication or embedding."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm-mode", choices=["off", "live"], default="off")
    parser.add_argument("--llm-limit", type=int, default=0, help="Maximum live LLM calls; 0 means all pending rows.")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-concurrency", type=int, default=16)
    parser.add_argument("--llm-timeout-seconds", type=int, default=60)
    parser.add_argument("--llm-max-retries", type=int, default=4)
    parser.add_argument("--llm-reprocess-evidence-id", action="append", default=[])
    args = parser.parse_args()

    stage = ProvenanceAttributionStage(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        llm_mode=args.llm_mode,
        llm_limit=max(0, args.llm_limit),
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_concurrency=max(1, args.llm_concurrency),
        llm_timeout_seconds=max(5, args.llm_timeout_seconds),
        llm_max_retries=max(1, args.llm_max_retries),
        llm_reprocess_evidence_ids=set(args.llm_reprocess_evidence_id),
    )
    print(json.dumps(stage.run(), ensure_ascii=False, indent=2, sort_keys=True))


class ProvenanceAttributionStage:
    def __init__(
        self,
        *,
        run_dir: Path,
        output_dir: Path,
        llm_mode: str = "off",
        llm_limit: int = 0,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_concurrency: int = 16,
        llm_timeout_seconds: int = 60,
        llm_max_retries: int = 4,
        llm_reprocess_evidence_ids: set[str] | None = None,
        database_url: str | None = None,
    ) -> None:
        self.layout = raw_processing_layout(run_dir)
        self.output_dir = output_dir
        self.llm_mode = llm_mode
        self.llm_limit = max(0, llm_limit)
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_concurrency = max(1, llm_concurrency)
        self.llm_timeout_seconds = max(5, llm_timeout_seconds)
        self.llm_max_retries = max(1, llm_max_retries)
        self.llm_reprocess_evidence_ids = set(llm_reprocess_evidence_ids or set())
        self.database_url = database_url
        self.manifest_dir = self.output_dir / "00_run_manifest"
        self.attribution_dir = self.output_dir / "01_compound_attribution"

    def run(self) -> dict[str, Any]:
        self._prepare_output_dirs()
        upstream_state = load_json(self.layout.path("pipeline_state.json"))
        validate_upstream_state(upstream_state)

        candidates = list(iter_jsonl(self.layout.path("evidence_candidates.jsonl")))
        inventory = {
            str(row["raw_asset_id"]): row
            for row in iter_jsonl(self.layout.path("raw_inventory.jsonl"))
        }
        ensure_unique_candidate_ids(candidates)

        reference_ids = collect_reference_ids(candidates)
        targets = sorted({
            normalize_name(str(row.get("compound") or ""))
            for row in [*inventory.values(), *candidates]
            if row.get("compound")
        })
        if upstream_state.get("run_root"):
            plan = load_json(Path(upstream_state["run_root"]) / "download_plan.json")
            planned = sorted({normalize_name(str(name)) for name in plan.get("compounds", [])})
            if planned:
                if set(targets) - set(planned):
                    raise ValueError("Evidence compounds differ from the accepted acquisition plan.")
                targets = planned
        if not targets:
            raise ValueError("The accepted acquisition has no target compounds.")
        identity_builder = CompoundIdentityBuilder(targets)
        references = self._load_references(reference_ids, identity_builder)
        compounds = identity_builder.build()
        database_url = self.database_url
        if database_url is None and os.getenv("PBPK_PROJECT_BOUNDARY") == "1":
            from pbpk_kb.config.settings import get_settings
            database_url = get_settings().database_url
        if database_url:
            import psycopg
            from psycopg.rows import dict_row
            with psycopg.connect(database_url.replace("postgresql+psycopg://", "postgresql://", 1),
                                 row_factory=dict_row) as connection:
                existing = connection.execute(
                    "SELECT compound_id,preferred_name,normalized_name,pubchem_cid,chembl_id,inchi_key "
                    "FROM compound_identity WHERE lower(preferred_name::text)=ANY(%s) "
                    "OR lower(normalized_name::text)=ANY(%s)", (targets, targets),
                ).fetchall()
            bind_existing_compound_identities(compounds, existing)

        provenance_rows = [
            validate_candidate_provenance(row, inventory=inventory, references=references)
            for row in candidates
        ]
        provenance_summary = summarize_provenance(provenance_rows, candidates)
        atomic_write_jsonl(self.manifest_dir / "provenance_validation.jsonl", provenance_rows)
        atomic_write_json(self.manifest_dir / "provenance_summary.json", provenance_summary)

        passed_ids = {
            str(row["evidence_id"])
            for row in provenance_rows
            if row["provenance_status"] == "passed"
        }
        attribution = build_rule_attributions(
            candidates=[row for row in candidates if str(row["evidence_id"]) in passed_ids],
            compounds=compounds,
            references=references,
            source_scope_identity=identity_builder.source_scope_identity,
        )
        llm_requests: list[dict[str, Any]] = []
        llm_responses: list[dict[str, Any]] = []
        if self.llm_mode == "live":
            attribution, llm_requests, llm_responses = self._run_live_llm(attribution, compounds)

        pending = [row for row in attribution if row["attribution_status"] == "pending_llm"]
        attribution_summary = summarize_attribution(
            attribution,
            candidate_count=len(candidates),
            provenance_passed=len(passed_ids),
            llm_mode=self.llm_mode,
            llm_requests=len(llm_requests),
            llm_responses=llm_responses,
        )

        attribution_summary["compound_identity_count"] = len(compounds)
        atomic_write_jsonl(self.attribution_dir / "compound_identity.jsonl", compounds)
        atomic_write_jsonl(self.attribution_dir / "compound_attribution.jsonl", attribution)
        atomic_write_jsonl(self.attribution_dir / "llm_attribution_pending.jsonl", pending)
        atomic_write_jsonl(self.attribution_dir / "llm_attribution_requests.jsonl", llm_requests)
        atomic_write_jsonl(self.attribution_dir / "llm_attribution_responses.jsonl", llm_responses)
        atomic_write_json(self.attribution_dir / "attribution_summary.json", attribution_summary)

        state = build_pipeline_state(
            upstream_state=upstream_state,
            run_dir=self.layout.root,
            output_dir=self.output_dir,
            provenance_summary=provenance_summary,
            attribution_summary=attribution_summary,
        )
        quality_report = {
            "generated_at": utc_now(),
            "run_id": upstream_state.get("run_id"),
            "candidate_bottom_table_immutable": True,
            "provenance": provenance_summary,
            "compound_attribution": attribution_summary,
            "later_steps_executed": False,
            "next_required_step": (
                "complete_llm_attribution"
                if attribution_summary["llm_pending_count"]
                else "domain_quality_gates"
            ),
        }
        atomic_write_json(self.manifest_dir / "pipeline_state.json", state)
        atomic_write_json(self.manifest_dir / "evidence_quality_report.json", quality_report)
        return quality_report

    def _prepare_output_dirs(self) -> None:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.attribution_dir.mkdir(parents=True, exist_ok=True)

    def _load_references(
        self,
        reference_ids: dict[str, set[str]],
        identity_builder: "CompoundIdentityBuilder",
    ) -> dict[str, dict[str, dict[str, Any]]]:
        references: dict[str, dict[str, dict[str, Any]]] = {}

        parsed_records: dict[str, dict[str, Any]] = {}
        parsed_record_path = self.layout.path("parsed_records.jsonl")
        wanted_records = reference_ids["parsed_record_id"]
        for row in iter_jsonl(parsed_record_path):
            identity_builder.observe(row)
            record_id = str(row.get("parsed_record_id") or "")
            if record_id in wanted_records:
                parsed_records[record_id] = row
        references["parsed_record_id"] = parsed_records

        for candidate_field, filename, record_field in REFERENCE_FILES:
            references[candidate_field] = load_selected_jsonl(
                self.layout.path(filename),
                record_field,
                reference_ids[candidate_field],
            )
        return references

    def _run_live_llm(
        self,
        attribution_rows: list[dict[str, Any]],
        compounds: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        agent = LLMAttributionAgent(
            mode="live",
            model_name=self.llm_model,
            base_url=self.llm_base_url,
            timeout_seconds=self.llm_timeout_seconds,
        )
        pending_indexes = [
            index
            for index, row in enumerate(attribution_rows)
            if row["attribution_status"] == "pending_llm"
        ]
        if self.llm_limit:
            pending_indexes = pending_indexes[: self.llm_limit]

        request_path = self.attribution_dir / "llm_attribution_requests.jsonl"
        response_path = self.attribution_dir / "llm_attribution_responses.jsonl"
        requests = load_unique_audit_rows(request_path, "evidence_id")
        responses = load_unique_audit_rows(response_path, "evidence_id")
        response_by_id = {str(item.get("evidence_id") or ""): item for item in responses}

        for index in pending_indexes:
            evidence_id = str(attribution_rows[index]["evidence_id"])
            prior = None if evidence_id in self.llm_reprocess_evidence_ids else response_by_id.get(evidence_id)
            if prior and (prior.get("validation") or {}).get("status") == "valid":
                apply_validated_attribution(attribution_rows[index], prior["validation"]["candidate"])

        pending_indexes = [
            index for index in pending_indexes
            if attribution_rows[index]["attribution_status"] == "pending_llm"
        ]
        if not pending_indexes:
            return attribution_rows, requests, responses

        request_path.parent.mkdir(parents=True, exist_ok=True)
        lock = threading.Lock()
        with request_path.open("a", encoding="utf-8") as request_file, response_path.open("a", encoding="utf-8") as response_file:
            def process(index: int) -> tuple[int, dict[str, Any], dict[str, Any]]:
                row = attribution_rows[index]
                request_row = {
                    "evidence_id": row["evidence_id"],
                    "requested_at": utc_now(),
                    "model": agent.model_name,
                    "allowed_compounds": [item["preferred_name"] for item in compounds],
                    "allowed_roles": sorted(COMPOUND_ROLES),
                    "evidence": row["candidate_context"],
                }
                with lock:
                    append_jsonl(request_file, request_row)
                response: dict[str, Any] = {"status": "not_started"}
                validated: dict[str, Any] = {"status": "invalid", "reason": "not_started"}
                attempts = 0
                for attempts in range(1, self.llm_max_retries + 1):
                    response = agent.attribute(row=llm_agent_row(row), compounds=compounds)
                    validated = validate_llm_attribution_response(response, row=row, compounds=compounds)
                    if validated["status"] == "valid":
                        break
                    if attempts < self.llm_max_retries:
                        time.sleep(min(8.0, 0.75 * (2 ** (attempts - 1))))
                response_row = {
                    "evidence_id": row["evidence_id"],
                    "responded_at": utc_now(),
                    "attempts": attempts,
                    "model": agent.model_name,
                    "raw_response": response,
                    "validation": validated,
                }
                prior_response = response_by_id.get(str(row["evidence_id"]))
                if prior_response:
                    response_row["previous_attempts"] = [
                        *(prior_response.get("previous_attempts") or []),
                        {key: value for key, value in prior_response.items() if key != "previous_attempts"},
                    ]
                with lock:
                    append_jsonl(response_file, response_row)
                return index, request_row, response_row

            completed = 0
            with ThreadPoolExecutor(max_workers=self.llm_concurrency) as executor:
                futures = {executor.submit(process, index): index for index in pending_indexes}
                for future in as_completed(futures):
                    index, request_row, response_row = future.result()
                    requests.append(request_row)
                    responses.append(response_row)
                    if response_row["validation"]["status"] == "valid":
                        apply_validated_attribution(attribution_rows[index], response_row["validation"]["candidate"])
                    completed += 1
                    if completed == 1 or completed % 25 == 0 or completed == len(pending_indexes):
                        valid_count = sum((item.get("validation") or {}).get("status") == "valid" for item in responses)
                        print(
                            f"llm_attribution completed={completed}/{len(pending_indexes)} valid_total={valid_count}",
                            flush=True,
                        )

        return (
            attribution_rows,
            unique_latest_rows(requests, "evidence_id"),
            unique_latest_rows(responses, "evidence_id"),
        )


def bind_existing_compound_identities(compounds: list[dict[str, Any]], existing: Iterable[dict[str, Any]]) -> None:
    """Connect new evidence to exact registered drugs without merging chemical forms."""
    by_name: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in existing:
        for field in ("preferred_name", "normalized_name"):
            if row.get(field):
                by_name[normalize_name(str(row[field]))][str(row["compound_id"])] = row
    bindings = []
    for compound in compounds:
        matches = by_name.get(normalize_name(compound["preferred_name"]), {})
        if len(matches) > 1:
            raise ValueError("Ambiguous registered compound identity: " + compound["preferred_name"])
        if not matches:
            continue
        current = next(iter(matches.values()))
        for old_key, new_key in (("pubchem_cid", "pubchem_cid"), ("chembl_id", "chembl_id"),
                                 ("inchi_key", "inchikey")):
            old, new = current.get(old_key), compound.get(new_key)
            if old and new and str(old) != str(new):
                raise ValueError("Registered compound stable identifier conflict: " + compound["preferred_name"])
        bindings.append((compound, current))
    for compound, current in bindings:
        compound["compound_id"] = str(current["compound_id"])
        compound["existing_registry_binding"] = {
            "compound_id": str(current["compound_id"]),
            "preferred_name": str(current["preferred_name"]),
            "method": "exact_name_with_stable_identifier_conflict_check",
        }


class CompoundIdentityBuilder:
    def __init__(self, compounds: Iterable[str]) -> None:
        self.targets = tuple(normalize_name(item) for item in compounds)
        self.rows: dict[str, dict[str, Any]] = {}
        self.assertion_keys: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        self.source_scope_identity: dict[str, str] = {}
        for target in self.targets:
            self.rows[target] = {
                "compound_id": str(uuid5(NAMESPACE_URL, f"pbpk:compound:{target}")),
                "preferred_name": target,
                "normalized_name": target,
                "synonyms": {target},
                "pubchem_cids": set(),
                "inchikeys": set(),
                "rxnorm_rxcuis": set(),
                "rxnorm_names": set(),
                "chembl_ids": set(),
                "chembl_exact_identity": False,
                "uniis": set(),
                "dailymed_active_ingredient_names": set(),
                "fda_active_ingredient_names": set(),
                "related_forms": [],
                "identity_provenance": [],
            }

    def observe(self, row: dict[str, Any]) -> None:
        target = normalize_name(str(row.get("compound") or ""))
        if target not in self.rows:
            return
        source_id = str(row.get("source_id") or "")
        request_key = str((row.get("metadata_json") or {}).get("request_key") or "")
        record = row.get("record_json")
        if not isinstance(record, dict):
            return

        if source_id == "pubchem":
            self._observe_pubchem(target, request_key, record, row)
        elif source_id == "rxnorm":
            self._observe_rxnorm(target, request_key, record, row)
        elif source_id == "chembl":
            self._observe_chembl(target, record, row)
        elif source_id == "dailymed" and row.get("parser_name") == "dailymed_spl_parser":
            self._observe_dailymed(target, record, row)
        elif source_id == "openfda_drug_label":
            self._observe_openfda(target, record, row)

    def build(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for target in self.targets:
            row = self.rows[target]
            pubchem_cids = sorted(row.pop("pubchem_cids"))
            inchikeys = sorted(row.pop("inchikeys"))
            rxcuis = sorted(row.pop("rxnorm_rxcuis"))
            chembl_ids = sorted(row.pop("chembl_ids"))
            uniis = sorted(row.pop("uniis"))
            for field in (
                "synonyms",
                "rxnorm_names",
                "dailymed_active_ingredient_names",
                "fda_active_ingredient_names",
            ):
                row[field] = sorted(row[field], key=lambda value: (value.lower(), value))
            row["pubchem_cid"] = pubchem_cids[0] if pubchem_cids else None
            row["pubchem_cids"] = pubchem_cids
            row["inchikey"] = inchikeys[0] if inchikeys else None
            row["inchikeys"] = inchikeys
            row["rxnorm_rxcui"] = rxcuis[0] if rxcuis else None
            row["rxnorm_rxcuis"] = rxcuis
            row["chembl_id"] = chembl_ids[0] if chembl_ids else None
            row["chembl_ids"] = chembl_ids
            row["unii"] = uniis
            row["identity_status"] = identity_status(row)
            result.append(row)
        return result

    def _observe_pubchem(
        self,
        target: str,
        request_key: str,
        record: dict[str, Any],
        row: dict[str, Any],
    ) -> None:
        if request_key == "properties_by_name":
            for item in ((record.get("PropertyTable") or {}).get("Properties") or []):
                if not isinstance(item, dict):
                    continue
                self._add_value(target, "pubchem_cids", item.get("CID"), row)
                self._add_value(target, "inchikeys", item.get("InChIKey"), row)
        elif request_key == "synonyms_by_name":
            information = (record.get("InformationList") or {}).get("Information") or []
            for item in information:
                for synonym in (item.get("Synonym") or [])[:80] if isinstance(item, dict) else []:
                    self._add_name(target, str(synonym), row, source_field="pubchem_synonym")

    def _observe_rxnorm(
        self,
        target: str,
        request_key: str,
        record: dict[str, Any],
        row: dict[str, Any],
    ) -> None:
        if request_key == "rxcui_by_name":
            for rxcui in ((record.get("idGroup") or {}).get("rxnormId") or []):
                self._add_value(target, "rxnorm_rxcuis", rxcui, row)
        if request_key == "drugs_by_name":
            for group in ((record.get("drugGroup") or {}).get("conceptGroup") or []):
                for concept in (group.get("conceptProperties") or []) if isinstance(group, dict) else []:
                    name = str(concept.get("name") or "")
                    relation = classify_target_name(name, target)
                    if relation == "exact":
                        self._add_value(target, "rxnorm_names", name, row)
                        self._add_value(target, "synonyms", name, row)
                        self._add_value(target, "rxnorm_rxcuis", concept.get("rxcui"), row)
                    elif relation == "related_form":
                        self._add_related_form(target, name, row, "salt_or_form")

    def _observe_chembl(self, target: str, record: dict[str, Any], row: dict[str, Any]) -> None:
        for molecule in record.get("molecules") or []:
            if not isinstance(molecule, dict) or not chembl_exact_match(molecule, target):
                continue
            self.rows[target]["chembl_exact_identity"] = True
            self._add_value(target, "chembl_ids", molecule.get("molecule_chembl_id"), row)
            structures = molecule.get("molecule_structures") or {}
            self._add_value(target, "inchikeys", structures.get("standard_inchi_key"), row)
            self._add_name(target, str(molecule.get("pref_name") or ""), row, source_field="chembl_pref_name")
            for synonym in molecule.get("molecule_synonyms") or []:
                if isinstance(synonym, dict):
                    self._add_name(
                        target,
                        str(synonym.get("molecule_synonym") or ""),
                        row,
                        source_field="chembl_synonym",
                    )
            break

    def _observe_dailymed(self, target: str, record: dict[str, Any], row: dict[str, Any]) -> None:
        active_ingredients = [
            item
            for item in (record.get("active_ingredients") or [])
            if isinstance(item, dict) and item.get("name")
        ]
        relations = {
            classify_target_name(str(item.get("name") or ""), target)
            for item in active_ingredients
        }
        scope_key = source_scope_key(str(row.get("source_id") or ""), str(row.get("raw_asset_id") or ""))
        if active_ingredients and "unrelated" not in relations and relations & {"exact", "related_form"}:
            self.source_scope_identity[scope_key] = "exact_single_target_active_ingredient"
        elif relations & {"exact", "related_form"} and "unrelated" in relations:
            self.source_scope_identity[scope_key] = "combination_product"

        for item in record.get("active_moieties") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if classify_target_name(name, target) == "exact":
                self._add_value(target, "synonyms", name, row)
                self._add_value(target, "uniis", item.get("unii"), row)
        for item in record.get("active_ingredients") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            relation = classify_target_name(name, target)
            if relation == "exact":
                self._add_value(target, "dailymed_active_ingredient_names", name, row)
                self._add_value(target, "synonyms", name, row)
                self._add_value(target, "uniis", item.get("unii"), row)
            elif relation == "related_form":
                self._add_value(target, "dailymed_active_ingredient_names", name, row)
                self._add_related_form(target, name, row, "salt_or_form", stable_id=item.get("unii"))

    def _observe_openfda(self, target: str, record: dict[str, Any], row: dict[str, Any]) -> None:
        for result_index, result in enumerate(record.get("results") or []):
            if not isinstance(result, dict):
                continue
            openfda = result.get("openfda") or {}
            names = [
                str(name)
                for field in ("generic_name", "substance_name")
                for name in (openfda.get(field) or [])
            ]
            matched_names = [name for name in names if classify_target_name(name, target) != "unrelated"]
            if not matched_names:
                continue
            normalized_names = {normalize_name(name) for name in names if name}
            relations = {classify_target_name(name, target) for name in normalized_names}
            scope_key = source_scope_key(
                str(row.get("source_id") or ""),
                str(row.get("raw_asset_id") or ""),
                result_index=result_index,
            )
            if normalized_names and "unrelated" not in relations:
                self.source_scope_identity[scope_key] = "exact_single_target_active_ingredient"
            elif relations & {"exact", "related_form"} and "unrelated" in relations:
                self.source_scope_identity[scope_key] = "combination_product"
            uniis = [str(value) for value in (openfda.get("unii") or []) if value]
            single_unii = uniis[0] if len(uniis) == 1 else None
            matched_relations = {classify_target_name(name, target) for name in matched_names}
            parent_unii = single_unii if matched_relations == {"exact"} else None
            related_form_unii = single_unii if "related_form" in matched_relations else None
            for name in matched_names:
                relation = classify_target_name(name, target)
                self._add_value(target, "fda_active_ingredient_names", name, row)
                if relation == "exact":
                    self._add_value(target, "synonyms", name, row)
                    if parent_unii:
                        self._add_value(target, "uniis", parent_unii, row)
                else:
                    self._add_related_form(target, name, row, "salt_or_form", stable_id=related_form_unii)

    def _add_name(
        self,
        target: str,
        name: str,
        row: dict[str, Any],
        *,
        source_field: str,
    ) -> None:
        clean = clean_identity_name(name)
        if not safe_direct_synonym(clean):
            return
        relation = classify_target_name(clean, target)
        if relation == "related_form":
            self._add_related_form(target, clean, row, "salt_or_form")
            return
        self._add_value(target, "synonyms", clean, row, assertion_type=source_field)

    def _add_value(
        self,
        target: str,
        field: str,
        value: Any,
        row: dict[str, Any],
        *,
        assertion_type: str | None = None,
    ) -> None:
        if value in (None, ""):
            return
        text = str(value).strip()
        if not text:
            return
        self.rows[target][field].add(text)
        self._add_provenance(target, assertion_type or field, text, row)

    def _add_related_form(
        self,
        target: str,
        name: str,
        row: dict[str, Any],
        relationship: str,
        *,
        stable_id: Any = None,
    ) -> None:
        clean = clean_identity_name(name)
        key = (relationship, clean.lower(), str(row.get("source_id") or ""))
        existing = {
            (item["relationship"], item["name"].lower(), item["source_id"])
            for item in self.rows[target]["related_forms"]
        }
        if key in existing:
            return
        self.rows[target]["related_forms"].append(
            {
                "name": clean,
                "relationship": relationship,
                "stable_id": str(stable_id) if stable_id else None,
                **source_reference(row),
            }
        )

    def _add_provenance(
        self,
        target: str,
        assertion_type: str,
        assertion_value: str,
        row: dict[str, Any],
    ) -> None:
        key = (assertion_type, assertion_value.lower(), str(row.get("source_id") or ""))
        if key in self.assertion_keys[target]:
            return
        self.assertion_keys[target].add(key)
        self.rows[target]["identity_provenance"].append(
            {
                "assertion_type": assertion_type,
                "assertion_value": assertion_value,
                **source_reference(row),
            }
        )


def validate_upstream_state(state: dict[str, Any]) -> None:
    stages = state.get("stages") or {}
    failures = []
    for stage_name in REQUIRED_UPSTREAM_STAGES:
        stage = stages.get(stage_name) or {}
        if stage.get("status") != "completed" or stage.get("accepted") is not True:
            failures.append(stage_name)
    if failures:
        raise RuntimeError(f"Required upstream stages are not completed and accepted: {', '.join(failures)}")
    embedding = stages.get("embedding") or {}
    if embedding.get("status") not in (None, "not_started"):
        raise RuntimeError("Raw-processing embedding must remain not_started before compound attribution.")


def ensure_unique_candidate_ids(candidates: list[dict[str, Any]]) -> None:
    counts = Counter(str(row.get("evidence_id") or "") for row in candidates)
    duplicates = sorted(evidence_id for evidence_id, count in counts.items() if count > 1)
    if duplicates:
        raise RuntimeError(f"Duplicate evidence_id values in candidate bottom table: {duplicates[:10]}")


def collect_reference_ids(candidates: list[dict[str, Any]]) -> dict[str, set[str]]:
    fields = {"parsed_record_id", *(item[0] for item in REFERENCE_FILES)}
    return {
        field: {
            str(row[field])
            for row in candidates
            if row.get(field) not in (None, "")
        }
        for field in fields
    }


def load_selected_jsonl(path: Path, id_field: str, wanted: set[str]) -> dict[str, dict[str, Any]]:
    if not wanted:
        return {}
    selected: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        record_id = str(row.get(id_field) or "")
        if record_id in wanted:
            selected[record_id] = row
    return selected


def validate_candidate_provenance(
    candidate: dict[str, Any],
    *,
    inventory: dict[str, dict[str, Any]],
    references: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    evidence_id = str(candidate.get("evidence_id") or "")
    reasons: list[str] = []
    missing_fields = [field for field in REQUIRED_PROVENANCE_FIELDS if candidate.get(field) in (None, "")]
    if missing_fields:
        reasons.extend(f"missing_required_field:{field}" for field in missing_fields)

    source_text = str(candidate.get("source_text") or "")
    expected_hash = sha256_text(source_text)
    if candidate.get("source_text_hash") != expected_hash:
        reasons.append("source_text_hash_mismatch")

    raw_asset_id = str(candidate.get("raw_asset_id") or "")
    raw_asset = inventory.get(raw_asset_id)
    if raw_asset is None:
        reasons.append("raw_asset_not_found")
    else:
        compare_reference_fields(candidate, raw_asset, "raw_inventory", reasons)

    resolution_methods: list[str] = []
    reference_details: list[dict[str, Any]] = []
    provided_reference_count = 0
    for field in ("parsed_record_id", *(item[0] for item in REFERENCE_FILES)):
        reference_id = candidate.get(field)
        if not reference_id:
            continue
        provided_reference_count += 1
        reference = references.get(field, {}).get(str(reference_id))
        if reference is None:
            reasons.append(f"reference_not_found:{field}")
            continue
        compare_reference_fields(candidate, reference, field, reasons)
        matched, method = source_text_in_reference(source_text, field=field, reference=reference)
        reference_details.append(
            {
                "reference_type": field,
                "reference_id": str(reference_id),
                "source_text_resolved": matched,
                "resolution_method": method,
            }
        )
        if matched:
            resolution_methods.append(method)

    if provided_reference_count == 0:
        reasons.append("no_parsed_or_structured_reference")
    elif not resolution_methods:
        reasons.append("source_text_not_resolved_in_references")

    reasons = sorted(set(reasons))
    return {
        "evidence_id": evidence_id,
        "run_id": candidate.get("run_id"),
        "source_id": candidate.get("source_id"),
        "raw_asset_id": raw_asset_id,
        "provenance_status": "excluded" if reasons else "passed",
        "exclusion_reasons": reasons,
        "source_text_hash_verified": candidate.get("source_text_hash") == expected_hash,
        "raw_asset_resolved": raw_asset is not None,
        "source_text_resolution_methods": sorted(set(resolution_methods)),
        "references": reference_details,
        "validated_at": utc_now(),
    }


def compare_reference_fields(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    reference_name: str,
    reasons: list[str],
) -> None:
    for field in ("raw_asset_id", "source_id", "checksum", "storage_path"):
        reference_value = reference.get(field)
        candidate_value = candidate.get(field)
        if reference_value not in (None, "") and str(reference_value) != str(candidate_value):
            reasons.append(f"{reference_name}_{field}_mismatch")


def source_text_in_reference(
    source_text: str,
    *,
    field: str,
    reference: dict[str, Any],
) -> tuple[bool, str]:
    if field == "parsed_record_id":
        matched = json_value_contains_text(reference.get("record_json"), source_text)
        return matched, "parsed_record_json_value"
    if field == "parsed_document_id":
        return normalized_contains(reference.get("text"), source_text), "parsed_document_text"
    if field == "chunk_id":
        return normalized_contains(reference.get("chunk_text"), source_text), "text_chunk"
    if field == "parsed_table_id":
        return normalized_contains(reference.get("table_text"), source_text) or json_value_contains_text(
            reference.get("cells_json"), source_text
        ), "parsed_table"
    if field == "recovered_table_id":
        return normalized_contains(reference.get("table_text"), source_text) or json_value_contains_text(
            reference.get("columns_json"), source_text
        ), "recovered_table"
    if field == "table_row_id":
        return normalized_contains(reference.get("row_text"), source_text) or json_value_contains_text(
            reference.get("cells_json"), source_text
        ), "recovered_table_row"
    return False, "unsupported_reference"


def normalized_contains(container: Any, needle: str) -> bool:
    if container in (None, "") or not needle:
        return False
    normalized_container = normalize_whitespace(str(container))
    normalized_needle = normalize_whitespace(needle)
    return normalized_needle == normalized_container or normalized_needle in normalized_container


def json_value_contains_text(value: Any, source_text: str) -> bool:
    target = normalize_whitespace(source_text)
    if not target:
        return False
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
        elif current is not None:
            scalar = normalize_whitespace(str(current))
            if target == scalar or target in scalar:
                return True
    return False


def summarize_provenance(
    rows: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    statuses = Counter(str(row["provenance_status"]) for row in rows)
    reasons = Counter(
        reason
        for row in rows
        for reason in row.get("exclusion_reasons") or []
    )
    methods = Counter(
        method
        for row in rows
        for method in row.get("source_text_resolution_methods") or []
    )
    return {
        "generated_at": utc_now(),
        "candidate_count": len(candidates),
        "validated_count": len(rows),
        "passed_count": statuses["passed"],
        "excluded_count": statuses["excluded"],
        "status_counts": dict(sorted(statuses.items())),
        "exclusion_reason_counts": dict(sorted(reasons.items())),
        "resolution_method_counts": dict(sorted(methods.items())),
        "candidate_bottom_table_modified": False,
        "accepted": len(rows) == len(candidates),
    }


def build_rule_attributions(
    *,
    candidates: list[dict[str, Any]],
    compounds: list[dict[str, Any]],
    references: dict[str, dict[str, dict[str, Any]]],
    source_scope_identity: dict[str, str],
) -> list[dict[str, Any]]:
    terms = build_compound_terms(compounds)
    compound_by_id = {str(item["compound_id"]): item for item in compounds}
    compound_by_name = {str(item["preferred_name"]).lower(): item for item in compounds}
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        retrieval_name = normalize_name(str(candidate.get("compound") or ""))
        retrieval_compound = compound_by_name.get(retrieval_name)
        adapted = adapt_candidate_for_attribution(
            candidate,
            retrieval_compound,
            references,
            source_scope_identity,
        )
        decision = decide_attribution(adapted, terms, compound_by_id, compound_by_name)
        public_model_review = requires_public_model_role_review(adapted, decision.compound_role)
        llm_required = decision.llm_required or public_model_review
        llm_trigger_reason = (
            "public_pbpk_model_nonfinal_role"
            if public_model_review
            else decision.llm_trigger_reason
        )
        if llm_required and not has_allowed_source_name_span(
            str(adapted.get("source_text") or ""), compounds
        ):
            # The span-constrained agent cannot assign a name absent from the
            # quoted source. This is an evidence limitation, not an LLM outage.
            # Preserve retrieval identity for the audit, but never promote the
            # row into the final evidence roles or invent a supporting quote.
            decision.compound_role = "shared_or_unlinked"
            decision.attribution_method = "rule_no_source_name_span"
            decision.attribution_confidence = 0.0
            decision.matched_text = None
            decision.matched_synonym = None
            decision.attribution_reason = (
                "No allowed compound name or synonym occurs verbatim in source_text. "
                "Retained as unlinked evidence; retrieval identity and surrounding metadata "
                "do not supply a quote for LLM attribution."
            )
            llm_required = False
            llm_trigger_reason = "no_source_name_span"
        status = "pending_llm" if llm_required else "resolved"
        output.append(
            {
                "evidence_id": decision.evidence_id,
                "candidate_compound": candidate.get("compound"),
                "compound_id": decision.compound_id,
                "compound_name": decision.compound_name,
                "compound_role": decision.compound_role,
                "attribution_method": decision.attribution_method,
                "attribution_confidence": round(float(decision.attribution_confidence), 3),
                "matched_text": decision.matched_text,
                "matched_synonym": decision.matched_synonym,
                "attribution_reason": decision.attribution_reason,
                "llm_required": llm_required,
                "llm_trigger_reason": llm_trigger_reason,
                "attribution_stage": "rule",
                "attribution_status": status,
                "candidate_context": {
                    key: adapted.get(key)
                    for key in (
                        "evidence_id",
                        "evidence_type",
                        "compound_name",
                        "source_id",
                        "source_title",
                        "source_location",
                        "parameter",
                        "condition",
                        "population",
                        "route",
                        "dose",
                        "source_text",
                    )
                },
            }
        )
    return output


def has_allowed_source_name_span(source_text: str, compounds: list[dict[str, Any]]) -> bool:
    """Use the attribution agent's exact-span eligibility, without changing text."""
    for compound in compounds:
        for term in [str(compound.get("preferred_name") or ""), *(compound.get("synonyms") or [])]:
            if not isinstance(term, str) or not term.strip() or term.strip().isdigit():
                continue
            if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", source_text, re.IGNORECASE):
                return True
    return False


def requires_public_model_role_review(row: dict[str, Any], role: str) -> bool:
    if str(row.get("evidence_type") or "") != "public_pbpk_model" or role == "target_compound":
        return False
    text = str(row.get("source_text") or "").lower()
    return "pbpk" in text or "physiologically based pharmacokinetic" in text or "physiologically-based pharmacokinetic" in text


def adapt_candidate_for_attribution(
    candidate: dict[str, Any],
    retrieval_compound: dict[str, Any] | None,
    references: dict[str, dict[str, dict[str, Any]]],
    source_scope_identity: dict[str, str],
) -> dict[str, Any]:
    adapted = dict(candidate)
    adapted["compound_id"] = str(retrieval_compound["compound_id"]) if retrieval_compound else ""
    adapted["compound_name"] = str(retrieval_compound["preferred_name"]) if retrieval_compound else ""
    metadata = dict(candidate.get("metadata_json") or {})
    identity_scope = source_scope_identity_for_candidate(candidate, source_scope_identity)
    if identity_scope:
        metadata["document_compound_identity"] = identity_scope
    adapted["metadata_json"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    document_id = candidate.get("parsed_document_id")
    document = references.get("parsed_document_id", {}).get(str(document_id)) if document_id else None
    if document:
        adapted["source_title"] = document.get("section_title") or document.get("title") or ""
        adapted["document_type"] = document.get("content_kind") or ""
    return adapted


def summarize_attribution(
    rows: list[dict[str, Any]],
    *,
    candidate_count: int,
    provenance_passed: int,
    llm_mode: str,
    llm_requests: int,
    llm_responses: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(str(row["attribution_status"]) for row in rows)
    role_counts = Counter(str(row["compound_role"]) for row in rows)
    method_counts = Counter(str(row["attribution_method"]) for row in rows)
    trigger_counts = Counter(
        str(row["llm_trigger_reason"])
        for row in rows
        if row.get("llm_trigger_reason")
    )
    llm_validation = Counter(
        str((row.get("validation") or {}).get("status") or "unknown")
        for row in llm_responses
    )
    pending_count = status_counts["pending_llm"]
    return {
        "generated_at": utc_now(),
        "candidate_count": candidate_count,
        "provenance_passed_count": provenance_passed,
        "attribution_count": len(rows),
        "resolved_count": status_counts["resolved"],
        "llm_pending_count": pending_count,
        "status_counts": dict(sorted(status_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "llm_trigger_counts": dict(sorted(trigger_counts.items())),
        "llm_mode": llm_mode,
        "llm_request_count": llm_requests,
        "llm_response_validation_counts": dict(sorted(llm_validation.items())),
        "rules_completed": len(rows) == provenance_passed,
        "accepted": len(rows) == provenance_passed and pending_count == 0,
    }


def build_pipeline_state(
    *,
    upstream_state: dict[str, Any],
    run_dir: Path,
    output_dir: Path,
    provenance_summary: dict[str, Any],
    attribution_summary: dict[str, Any],
) -> dict[str, Any]:
    attribution_complete = attribution_summary["accepted"]
    stages: dict[str, Any] = {
        "provenance": {
            "status": "completed",
            "accepted": provenance_summary["accepted"],
            "passed_count": provenance_summary["passed_count"],
            "excluded_count": provenance_summary["excluded_count"],
            "summary_path": "00_run_manifest/provenance_summary.json",
        },
        "compound_attribution": {
            "status": "completed" if attribution_complete else "rules_completed",
            "accepted": attribution_complete,
            "resolved_count": attribution_summary["resolved_count"],
            "llm_pending_count": attribution_summary["llm_pending_count"],
            "summary_path": "01_compound_attribution/attribution_summary.json",
        },
    }
    for stage_name in (
        "domain_quality_gates",
        "deterministic_deduplication",
        "semantic_deduplication",
        "conflict_analysis",
        "evidence_layers",
        "human_review",
        "rag_index",
        "database_ui_export",
    ):
        stages[stage_name] = {"status": "not_started"}
    return {
        "run_id": upstream_state.get("run_id"),
        "source_raw_processing_run": str(run_dir),
        "output_dir": str(output_dir),
        "current_stage": "compound_attribution" if attribution_complete else "compound_attribution_llm_pending",
        "updated_at": utc_now(),
        "stages": stages,
    }


def validate_llm_attribution_response(
    response: dict[str, Any],
    *,
    row: dict[str, Any],
    compounds: list[dict[str, Any]],
) -> dict[str, Any]:
    if response.get("status") != "ok" or not isinstance(response.get("candidate"), dict):
        return {"status": "invalid", "reason": str(response.get("status") or "missing_candidate")}
    candidate = response["candidate"]
    if str(candidate.get("evidence_id") or "") != str(row["evidence_id"]):
        return {"status": "invalid", "reason": "evidence_id_mismatch"}
    allowed = {str(item["preferred_name"]).lower(): item for item in compounds}
    compound_name = str(candidate.get("compound_name") or "").strip().lower()
    role = str(candidate.get("compound_role") or "").strip()
    matched_text = str(candidate.get("matched_text") or "")
    source_text = str((row.get("candidate_context") or {}).get("source_text") or "")
    if compound_name not in allowed:
        return {"status": "invalid", "reason": "compound_not_in_identity_table"}
    if role not in COMPOUND_ROLES:
        return {"status": "invalid", "reason": "invalid_compound_role"}
    if not matched_text or matched_text not in source_text:
        return {"status": "invalid", "reason": "matched_text_not_verbatim"}
    try:
        confidence = float(candidate.get("confidence"))
    except (TypeError, ValueError):
        return {"status": "invalid", "reason": "invalid_confidence"}
    if not 0.0 <= confidence <= 1.0:
        return {"status": "invalid", "reason": "confidence_out_of_range"}
    return {
        "status": "valid",
        "candidate": {
            "evidence_id": row["evidence_id"],
            "compound_id": str(allowed[compound_name]["compound_id"]),
            "compound_name": str(allowed[compound_name]["preferred_name"]),
            "compound_role": role,
            "matched_text": matched_text,
            "role_reason": str(candidate.get("role_reason") or "LLM attribution review"),
            "confidence": round(confidence, 3),
        },
    }


def llm_agent_row(row: dict[str, Any]) -> dict[str, str]:
    context = row.get("candidate_context") or {}
    return {key: "" if value is None else str(value) for key, value in context.items()}


def apply_validated_attribution(row: dict[str, Any], candidate: dict[str, Any]) -> None:
    row.update(
        {
            "compound_id": candidate["compound_id"],
            "compound_name": candidate["compound_name"],
            "compound_role": candidate["compound_role"],
            "matched_text": candidate["matched_text"],
            "matched_synonym": candidate["matched_text"],
            "attribution_reason": candidate["role_reason"],
            "attribution_confidence": candidate["confidence"],
            "attribution_method": "llm_attribution_review",
            "attribution_stage": "llm",
            "attribution_status": "resolved",
            "llm_required": False,
        }
    )


def identity_status(row: dict[str, Any]) -> str:
    required = (
        bool(row.get("pubchem_cid")),
        bool(row.get("inchikey")),
        bool(row.get("rxnorm_rxcui")),
        bool(row.get("chembl_id")) and row.get("chembl_exact_identity") is True,
        bool(row.get("unii")),
    )
    return "complete_core_identity" if all(required) else "partial_core_identity"


def chembl_exact_match(molecule: dict[str, Any], target: str) -> bool:
    names = [str(molecule.get("pref_name") or "")]
    for synonym in molecule.get("molecule_synonyms") or []:
        if isinstance(synonym, dict):
            names.append(str(synonym.get("molecule_synonym") or ""))
    for cross_reference in molecule.get("cross_references") or []:
        if isinstance(cross_reference, dict):
            names.extend(
                [
                    str(cross_reference.get("xref_id") or ""),
                    str(cross_reference.get("xref_name") or ""),
                ]
            )
    return any(normalize_name(name) == target for name in names)


def classify_target_name(name: str, target: str) -> str:
    normalized = normalize_name(name)
    if normalized == target:
        return "exact"
    target_tokens = target.split()
    tokens = normalized.split()
    if tokens[: len(target_tokens)] == target_tokens:
        remainder = tokens[len(target_tokens) :]
        if remainder and all(token in SALT_OR_FORM_WORDS for token in remainder):
            return "related_form"
    return "unrelated"


def clean_identity_name(value: str) -> str:
    return normalize_whitespace(unicodedata.normalize("NFKC", value)).strip(" ,;|")


def safe_direct_synonym(value: str) -> bool:
    if not 3 <= len(value) <= 100 or not any(character.isalpha() for character in value):
        return False
    lowered = value.lower()
    if normalize_name(value) in AMBIGUOUS_IDENTITY_SYNONYMS:
        return False
    if any(marker in lowered for marker in ("http://", "https://", "inchi=", " and ", "/")):
        return False
    if re.fullmatch(r"\d{2,7}-\d{2}-\d", value):
        return False
    if re.fullmatch(r"[A-Z]{1,8}-?\d{3,}[A-Z0-9-]*", value):
        return False
    if re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", value):
        return False
    return True


def source_scope_key(source_id: str, raw_asset_id: str, *, result_index: int | None = None) -> str:
    suffix = f":result:{result_index}" if result_index is not None else ""
    return f"{source_id.lower()}:{raw_asset_id}{suffix}"


def source_scope_identity_for_candidate(
    candidate: dict[str, Any],
    source_scope_identity: dict[str, str],
) -> str | None:
    source_id = str(candidate.get("source_id") or "").lower()
    raw_asset_id = str(candidate.get("raw_asset_id") or "")
    if source_id == "openfda_drug_label":
        match = re.search(r"\.results\[(\d+)\]", str(candidate.get("source_location") or ""))
        if match:
            return source_scope_identity.get(
                source_scope_key(source_id, raw_asset_id, result_index=int(match.group(1)))
            )
    return source_scope_identity.get(source_scope_key(source_id, raw_asset_id))


def source_reference(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": row.get("source_id"),
        "raw_asset_id": row.get("raw_asset_id"),
        "parsed_record_id": row.get("parsed_record_id"),
        "source_location": row.get("source_location"),
        "checksum": row.get("checksum"),
    }


def normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return normalize_whitespace(normalized)


def normalize_whitespace(value: str) -> str:
    return " ".join(str(value).split())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            yield value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
    handle.write("\n")
    handle.flush()


def load_unique_audit_rows(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return unique_latest_rows(iter_jsonl(path), key)


def unique_latest_rows(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            continue
        if value not in latest:
            order.append(value)
        latest[value] = row
    return [latest[value] for value in order]


if __name__ == "__main__":
    main()
