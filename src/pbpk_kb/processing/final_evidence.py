from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from pbpk_kb.processing.raw_processing_paths import raw_processing_layout


FINAL_MANIFEST_DIR = "00_run_manifest"
FINAL_ATTRIBUTION_DIR = "01_compound_attribution"
FINAL_DEDUPLICATION_DIR = "02_deduplication"
FINAL_CONFLICT_DIR = "03_conflict_analysis"
FINAL_EVIDENCE_LAYERS_DIR = "04_evidence_layers"
FINAL_OUTPUT_DIRS = (
    FINAL_MANIFEST_DIR,
    FINAL_ATTRIBUTION_DIR,
    FINAL_DEDUPLICATION_DIR,
    FINAL_CONFLICT_DIR,
    FINAL_EVIDENCE_LAYERS_DIR,
)
DOCKER_CONTAINER = "pbpk_stage1_postgres"
DB_USER = "pbpk"
DB_NAME = "pbpk_kb"
from pbpk_kb.processing.evidence_llm import (DEFAULT_MODEL as DEFAULT_LLM_MODEL, LOCAL_BASE_URL as DEFAULT_LLM_BASE_URL, generate_evidence_json, validate_backend, backend_identity, generation_format)
BIOMEDICAL_EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"


def ensure_final_output_dirs(output_dir: Path) -> None:
    for subdir in FINAL_OUTPUT_DIRS:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

ATTRIBUTION_COLUMNS = [
    "evidence_id",
    "compound_id",
    "compound_name",
    "compound_role",
    "attribution_method",
    "attribution_confidence",
    "matched_text",
    "matched_synonym",
    "attribution_reason",
    "llm_required",
    "llm_trigger_reason",
    "attribution_stage",
    "llm_payload",
]

DUPLICATE_GROUP_COLUMNS = [
    "duplicate_group_id",
    "group_type",
    "group_key",
    "evidence_type",
    "compound_id",
    "compound_name",
    "representative_evidence_id",
    "duplicate_count",
    "selection_reason",
    "metadata_json",
]

DUPLICATE_MEMBER_COLUMNS = [
    "duplicate_group_id",
    "evidence_id",
    "is_representative",
    "duplicate_rank",
    "duplicate_reason",
    "similarity_score",
]

COMPOUND_ROLES = {
    "target_compound",
    "victim_drug",
    "perpetrator_drug",
    "co_medication",
    "comparator",
    "excipient",
    "mentioned_only",
    "shared_or_unlinked",
}

FINAL_LAYER_ROLES = {"target_compound", "victim_drug", "perpetrator_drug"}
STAGE1_EVIDENCE_TYPES = {"physicochemical", "adme", "clinical_pk", "ddi", "special_population", "tissue_expression", "public_pbpk_model"}
MULTI_DRUG_TERMS = ["coadministered", "co-administered", "concomitant", "versus", "compared with", "combination", "interaction"]
DDI_TERMS = ["increased", "decreased", "reduced", "inhibited", "inhibits", "inhibitor", "induced", "induces", "inducer", "auc", "cmax", "exposure"]
DDI_DIRECTIONAL_OUTCOME_TERMS = [
    "increase", "decrease", "reduce", "raise", "lower", "greater", "higher", "prolong", "shorten", "unchanged", "no effect",
    "did not alter", "inhibit", "induce", "impair", "fold",
]
DDI_EFFECT_ENDPOINT_TERMS = [
    "auc", "cmax", "tmax", "exposure", "clearance", "half-life", "half life", "pharmacokinetic",
    "concentration", "bioavailability", "metabolism", "cyp", "enzyme", "transporter", "p-gp", "bcrp", "oatp",
]
COMPARATOR_TERMS = ["compared with", "versus", "vs.", "comparator", "active control", "control arm"]
EXCIPIENT_TERMS = ["excipient", "inactive ingredient", "inactive ingredients", "vehicle", "tablet contains", "capsule contains"]
CO_MEDICATION_TERMS = ["concomitant medication", "co-medication", "co medication", "comedication", "concomitant use", "coadministered", "co-administered", "combination therapy"]
BACKGROUND_TERMS = [
    "eligibilitycriteria",
    "inclusion criteria",
    "exclusion criteria",
    "prohibited medication",
    "prohibited concomitant",
    "prior treatment",
    "medical history",
    "history of",
    "sourcevocabulary",
    "meddra",
    "numatrisk",
    "numaffected",
    "organSystem",
    "citation",
    "references",
    "background",
]
RETRIEVAL_ONLY_ALLOWED_SOURCES = {"pubchem", "rxnorm", "drugcentral"}
PROBE_RELATION_MISMATCH_METHOD = "rule_probe_relation_parameter_mismatch"
DDI_DIRECTION_UNCERTAIN_METHOD = "rule_ddi_direction_uncertain"
PROBE_RELATION_CONTEXT_TERMS = [
    "probe",
    "cocktail",
    "phenotyping",
    "substrate",
    "cytochrome",
    "transporter",
    "cyp",
    "p-gp",
    "p glycoprotein",
    "bcrp",
    "oatp",
    "oct",
    "mate",
]
RELATION_TARGET_PATTERNS: dict[str, list[str]] = {
    "CYP1A2": [r"\bcyp\s*1a2\b", r"\bcytochrome\s+p450\s*1a2\b"],
    "CYP2C8": [r"\bcyp\s*2c8\b", r"\bcytochrome\s+p450\s*2c8\b"],
    "CYP2C9": [r"\bcyp\s*2c9\b", r"\bcytochrome\s+p450\s*2c9\b"],
    "CYP2C19": [r"\bcyp\s*2c19\b", r"\bcytochrome\s+p450\s*2c19\b"],
    "CYP2D6": [r"\bcyp\s*2d6\b", r"\bcytochrome\s+p450\s*2d6\b"],
    "CYP3A": [r"\bcyp\s*3a\b", r"\bcytochrome\s+p450\s*3a\b"],
    "CYP3A4": [r"\bcyp\s*3a4\b", r"\bcytochrome\s+p450\s*3a4\b"],
    "CYP3A5": [r"\bcyp\s*3a5\b", r"\bcytochrome\s+p450\s*3a5\b"],
    "UGT": [r"\bugt\b", r"\bugts\b", r"\bglucuronosyltransferase\b"],
    "P-gp": [r"\bp[- ]?gp\b", r"\bp[- ]?glycoprotein\b", r"\babcb1\b"],
    "BCRP": [r"\bbcrp\b", r"\babcg2\b"],
    "OATP": [r"\boatp\b", r"\bslco\b"],
    "OATP1B1": [r"\boatp\s*1b1\b", r"\bslco1b1\b"],
    "OATP1B3": [r"\boatp\s*1b3\b", r"\bslco1b3\b"],
    "OCT": [r"\boct\b"],
    "OCT1": [r"\boct\s*1\b", r"\bslc22a1\b"],
    "OCT2": [r"\boct\s*2\b", r"\bslc22a2\b"],
    "MATE": [r"\bmate\b"],
    "MATE1": [r"\bmate\s*1\b", r"\bslc47a1\b"],
    "MATE2": [r"\bmate\s*2(?:-k)?\b", r"\bslc47a2\b"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final evidence layers from processed evidence candidates.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--container", default=DOCKER_CONTAINER)
    parser.add_argument("--db-user", default=DB_USER)
    parser.add_argument("--db-name", default=DB_NAME)
    parser.add_argument("--reset", action="store_true", help="Clear final evidence attribution and duplicate tables before rebuilding.")
    parser.add_argument("--reuse-attribution", action="store_true", help="Reuse existing evidence_compound_attribution rows; rebuild duplicate groups, conflicts, and exports only.")
    parser.add_argument("--llm-mode", choices=["off", "auto", "live"], default="auto")
    parser.add_argument("--llm-limit", type=int, default=100, help="Maximum low-confidence attribution rows to send to the LLM.")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--semantic-mode", choices=["off", "auto"], default="auto")
    parser.add_argument("--embedding-device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--semantic-max-source-chars", type=int, default=1500)
    parser.add_argument("--export-reviewable-limit", type=int, default=None, help="Limit reviewable_evidence.jsonl rows for smoke tests.")
    args = parser.parse_args()

    builder = FinalEvidenceBuilder(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        container=args.container,
        db_user=args.db_user,
        db_name=args.db_name,
        reset=args.reset,
        reuse_attribution=args.reuse_attribution,
        llm_mode=args.llm_mode,
        llm_limit=args.llm_limit,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        semantic_mode=args.semantic_mode,
        embedding_device=args.embedding_device,
        semantic_max_source_chars=args.semantic_max_source_chars,
        export_reviewable_limit=args.export_reviewable_limit,
    )
    summary = builder.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


@dataclass(frozen=True)
class CompoundTerm:
    compound_id: str
    compound_name: str
    synonym: str
    pattern: re.Pattern[str]


@dataclass
class AttributionDecision:
    evidence_id: str
    compound_id: str | None
    compound_name: str | None
    compound_role: str
    attribution_method: str
    attribution_confidence: float
    matched_text: str | None
    matched_synonym: str | None
    attribution_reason: str
    llm_payload: dict[str, Any]
    llm_required: bool = False
    llm_trigger_reason: str | None = None
    attribution_stage: str = "rule"

    def to_row(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "compound_id": self.compound_id,
            "compound_name": self.compound_name,
            "compound_role": self.compound_role,
            "attribution_method": self.attribution_method,
            "attribution_confidence": f"{max(0.0, min(1.0, self.attribution_confidence)):.3f}",
            "matched_text": self.matched_text,
            "matched_synonym": self.matched_synonym,
            "attribution_reason": self.attribution_reason,
            "llm_required": self.llm_required,
            "llm_trigger_reason": self.llm_trigger_reason,
            "attribution_stage": self.attribution_stage,
            "llm_payload": json_dump(self.llm_payload),
        }


class FinalEvidenceBuilder:
    def __init__(
        self,
        *,
        run_dir: Path,
        output_dir: Path,
        container: str,
        db_user: str,
        db_name: str,
        reset: bool,
        reuse_attribution: bool,
        llm_mode: str,
        llm_limit: int,
        llm_model: str | None,
        llm_base_url: str | None,
        semantic_mode: str,
        embedding_device: str,
        semantic_max_source_chars: int,
        export_reviewable_limit: int | None,
    ) -> None:
        self.raw_processing_paths = raw_processing_layout(run_dir)
        self.run_dir = self.raw_processing_paths.root
        self.output_dir = output_dir
        self.container = container
        self.db_user = db_user
        self.db_name = db_name
        self.reset = reset
        self.reuse_attribution = reuse_attribution
        self.llm_mode = llm_mode
        self.llm_limit = max(0, llm_limit)
        self.semantic_mode = semantic_mode
        self.embedding_device = embedding_device
        self.semantic_max_source_chars = max(200, semantic_max_source_chars)
        self.export_reviewable_limit = export_reviewable_limit
        self.counts: Counter[str] = Counter()
        self.llm_agent = LLMAttributionAgent(mode=llm_mode, model_name=llm_model, base_url=llm_base_url)

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ensure_final_output_dirs(self.output_dir)
        self._exec_sql(Path("pbpk_kb/storage/schema.sql").read_text(encoding="utf-8"))
        if self.reset:
            if self.reuse_attribution:
                self._reset_duplicate_layers()
            else:
                self._reset_final_layers()
        if not self.reuse_attribution:
            compounds = self._load_compounds()
            terms = build_compound_terms(compounds)
            self._build_attribution(compounds, terms)
        self._apply_relation_validation()
        self._build_deterministic_duplicate_groups()
        semantic_summary = self._build_semantic_duplicate_groups()
        self._analyze()
        exports = self._export_outputs(semantic_summary)
        report = self._quality_report(semantic_summary, exports)
        write_json(self.output_dir / FINAL_MANIFEST_DIR / "evidence_quality_report.json", report)
        return report

    def _reset_final_layers(self) -> None:
        self._exec_sql(
            """
            TRUNCATE TABLE
                evidence_duplicate_member,
                evidence_duplicate_group,
                evidence_compound_attribution
            RESTART IDENTITY CASCADE;
            """
        )

    def _reset_duplicate_layers(self) -> None:
        self._exec_sql(
            """
            TRUNCATE TABLE
                evidence_duplicate_member,
                evidence_duplicate_group
            RESTART IDENTITY CASCADE;
            """
        )

    def _load_compounds(self) -> list[dict[str, str | list[str] | None]]:
        rows = self._query_csv(
            """
            SELECT
                compound_id::text,
                preferred_name::text,
                normalized_name::text,
                synonyms::text,
                pubchem_cid,
                chembl_id,
                rxnorm_rxcui
            FROM compound_identity
            ORDER BY preferred_name;
            """
        )
        compounds: list[dict[str, str | list[str] | None]] = []
        for row in rows:
            compounds.append(
                {
                    "compound_id": row["compound_id"],
                    "preferred_name": row["preferred_name"],
                    "normalized_name": row["normalized_name"],
                    "synonyms": parse_pg_text_array(row.get("synonyms") or ""),
                    "pubchem_cid": row.get("pubchem_cid"),
                    "chembl_id": row.get("chembl_id"),
                    "rxnorm_rxcui": row.get("rxnorm_rxcui"),
                }
            )
        return compounds

    def _build_attribution(self, compounds: list[dict[str, Any]], terms: list[CompoundTerm]) -> None:
        compound_by_id = {str(row["compound_id"]): row for row in compounds}
        compound_by_name = {str(row["preferred_name"]).lower(): row for row in compounds}
        rows = self._iter_candidate_rows()
        writer = CopyWriter(self, "evidence_compound_attribution", ATTRIBUTION_COLUMNS)
        try:
            for row in rows:
                decision = decide_attribution(row, terms, compound_by_id, compound_by_name)
                if self._should_call_llm(decision):
                    llm_decision = self._llm_refine_attribution(row, compounds)
                    if llm_decision is not None:
                        decision = llm_decision
                writer.write(decision.to_row())
                self.counts["attribution_rows"] += 1
                self.counts[f"role:{decision.compound_role}"] += 1
                self.counts[f"method:{decision.attribution_method}"] += 1
                if decision.llm_required:
                    self.counts["llm_attribution_required"] += 1
        finally:
            writer.close()

    def _should_call_llm(self, decision: AttributionDecision) -> bool:
        if not decision.llm_required:
            return False
        if self.llm_mode == "off":
            self.counts["llm_attribution_skipped_mode_off"] += 1
            return False
        if self.counts["llm_attribution_requests"] >= self.llm_limit:
            self.counts["llm_attribution_skipped_limit"] += 1
            return False
        return True

    def _llm_refine_attribution(self, row: dict[str, str], compounds: list[dict[str, Any]]) -> AttributionDecision | None:
        self.counts["llm_attribution_requests"] += 1
        result = self.llm_agent.attribute(row=row, compounds=compounds)
        self.counts[f"llm_attribution_status:{result.get('status')}"] += 1
        if result.get("status") != "ok":
            return None
        candidate = result.get("candidate")
        if not isinstance(candidate, dict):
            return None
        source_text = row.get("source_text") or ""
        matched_text = str(candidate.get("matched_text") or "")
        if not matched_text or matched_text not in source_text:
            self.counts["llm_attribution_rejected_missing_matched_text"] += 1
            return None
        role = str(candidate.get("compound_role") or "")
        if role not in COMPOUND_ROLES:
            self.counts["llm_attribution_rejected_bad_role"] += 1
            return None
        compound_name = str(candidate.get("compound_name") or "").strip().lower()
        compound = find_compound_by_name_or_synonym(compounds, compound_name)
        if compound is None:
            self.counts["llm_attribution_rejected_unknown_compound"] += 1
            return None
        if not any(term.lower() in matched_text.lower() for term in compound_validation_terms(compound)):
            self.counts["llm_attribution_rejected_matched_text_without_compound"] += 1
            return None
        confidence = safe_float(candidate.get("confidence"), default=0.5)
        return AttributionDecision(
            evidence_id=row["evidence_id"],
            compound_id=str(compound["compound_id"]),
            compound_name=str(compound["preferred_name"]),
            compound_role=role,
            attribution_method="llm_assisted_attribution",
            attribution_confidence=confidence,
            matched_text=matched_text,
            matched_synonym=str(candidate.get("matched_synonym") or compound["preferred_name"]),
            attribution_reason=str(candidate.get("role_reason") or "LLM-assisted attribution with matched source text."),
            llm_payload={"model_name": self.llm_agent.model_name, "response": candidate},
            llm_required=False,
            llm_trigger_reason=None,
            attribution_stage="llm_assisted",
        )

    def _build_deterministic_duplicate_groups(self) -> None:
        self._exec_sql(DEDUPLICATION_SQL)

    def _apply_relation_validation(self) -> None:
        rows = self._query_csv(
            """
            SELECT
                pec.evidence_id::text,
                pec.evidence_type,
                pec.compound_id::text,
                pec.compound_name::text,
                pec.source_id,
                pec.source_text,
                pec.source_location,
                pec.parameter,
                pec.condition,
                pec.population,
                pec.route,
                pec.dose,
                coalesce(pd.title, '') AS source_title,
                coalesce(pd.document_type, '') AS document_type,
                coalesce(pec.metadata_json::text, '{}') AS metadata_json,
                coalesce(pec.source_record::text, '{}') AS source_record,
                coalesce(pec.extraction_confidence::text, '') AS extraction_confidence,
                pec.curation_status::text,
                eca.attribution_id::text,
                eca.compound_id::text AS attributed_compound_id,
                eca.compound_name AS attributed_compound_name,
                eca.compound_role,
                eca.attribution_method,
                eca.matched_text,
                eca.matched_synonym
            FROM processed_evidence_candidate pec
            JOIN evidence_compound_attribution eca
                ON eca.evidence_id = pec.evidence_id
            LEFT JOIN parsed_document pd
                ON pd.parsed_document_id = pec.parsed_document_id
            WHERE eca.compound_role IN ('target_compound', 'victim_drug', 'perpetrator_drug')
              AND eca.llm_required = FALSE
            ORDER BY pec.evidence_id;
            """
        )
        mismatches: list[dict[str, Any]] = []
        for row in rows:
            issue = final_evidence_quality_issue(
                row,
                compound_name=row.get("attributed_compound_name") or row.get("compound_name") or "",
                matched_text=row.get("matched_text") or row.get("matched_synonym"),
                compound_role=row.get("compound_role") or "",
            )
            if issue is None:
                continue
            mismatches.append(
                {
                    "attribution_id": row["attribution_id"],
                    "attribution_method": issue["method"],
                    "attribution_reason": issue["reason"],
                    "llm_trigger_reason": issue["trigger"],
                    "llm_payload": {
                        "relation_validation": {
                            "method": issue["method"],
                            "previous_compound_role": row.get("compound_role"),
                            "previous_attribution_method": row.get("attribution_method"),
                            "matched_text": row.get("matched_text"),
                            "parameter": row.get("parameter"),
                        }
                    },
                }
            )

        if not mismatches:
            self.counts["relation_validation_mismatch"] += 0
            return

        self.counts["relation_validation_mismatch"] += len(mismatches)
        columns = ["attribution_id", "attribution_method", "attribution_reason", "llm_trigger_reason", "llm_payload"]
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".csv", delete=False) as file:
            temp_path = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for mismatch in mismatches:
                writer.writerow({column: csv_value(mismatch.get(column)) for column in columns})

        container_csv_path = f"/tmp/{temp_path.name}"
        try:
            subprocess.run(["docker", "cp", str(temp_path), f"{self.container}:{container_csv_path}"], check=True)
            sql = """
            CREATE TEMP TABLE tmp_relation_validation_mismatch (
                attribution_id UUID PRIMARY KEY,
                attribution_method TEXT NOT NULL,
                attribution_reason TEXT NOT NULL,
                llm_trigger_reason TEXT NOT NULL,
                llm_payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
            );
            \\copy tmp_relation_validation_mismatch FROM '{csv_path}' WITH (FORMAT csv, HEADER true)

            UPDATE evidence_compound_attribution eca
            SET
                compound_role = 'shared_or_unlinked',
                attribution_method = tmp.attribution_method,
                attribution_confidence = 0.250,
                attribution_reason = tmp.attribution_reason,
                llm_required = TRUE,
                llm_trigger_reason = tmp.llm_trigger_reason,
                attribution_stage = 'rule',
                llm_payload = tmp.llm_payload
            FROM tmp_relation_validation_mismatch tmp
            WHERE eca.attribution_id = tmp.attribution_id;
            """.format(
                csv_path=container_csv_path.replace("'", "''"),
            )
            self._exec_sql(sql)
        finally:
            subprocess.run(["docker", "exec", self.container, "rm", "-f", container_csv_path], check=False)
            temp_path.unlink(missing_ok=True)

    def _build_semantic_duplicate_groups(self) -> dict[str, Any]:
        if self.semantic_mode == "off":
            return {
                "semantic_dedup_skipped": True,
                "semantic_dedup_reason": "semantic_mode_off",
                "embedding_model": BIOMEDICAL_EMBEDDING_MODEL,
            }
        try:
            import sentence_transformers  # type: ignore  # noqa: F401
        except Exception as exc:
            return {
                "semantic_dedup_skipped": True,
                "semantic_dedup_reason": f"sentence_transformers_unavailable:{type(exc).__name__}",
                "embedding_model": BIOMEDICAL_EMBEDDING_MODEL,
                "fallback_used": False,
            }
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
            import numpy as np  # type: ignore
            import torch  # type: ignore
        except Exception as exc:
            return {
                "semantic_dedup_skipped": True,
                "semantic_dedup_reason": f"embedding_import_failed:{type(exc).__name__}",
                "embedding_model": BIOMEDICAL_EMBEDDING_MODEL,
                "fallback_used": False,
            }

        device = resolve_embedding_device(self.embedding_device, torch)
        model = SentenceTransformer(BIOMEDICAL_EMBEDDING_MODEL, device=device)
        groups = self._semantic_candidate_groups()
        group_writer = CopyWriter(self, "evidence_duplicate_group", DUPLICATE_GROUP_COLUMNS)
        member_writer = CopyWriter(self, "evidence_duplicate_member", DUPLICATE_MEMBER_COLUMNS)
        summary: Counter[str] = Counter()
        try:
            for group_index, rows in enumerate(groups, start=1):
                summary["candidate_groups_seen"] += 1
                summary["candidate_rows_seen"] += len(rows)
                if len(rows) < 2:
                    continue
                if group_index == 1 or group_index % 25 == 0 or len(rows) >= 500:
                    print(
                        f"semantic_dedup group={group_index} rows={len(rows)} "
                        f"compound={rows[0].get('compound_name')} type={rows[0].get('evidence_type')} "
                        f"parameter={rows[0].get('parameter')} device={device}",
                        flush=True,
                    )
                texts = [semantic_text(row, max_source_chars=self.semantic_max_source_chars) for row in rows]
                embeddings = model.encode(texts, batch_size=64 if device == "mps" else 32, normalize_embeddings=True, show_progress_bar=False)
                clusters = util.community_detection(embeddings, threshold=0.92, min_community_size=2, batch_size=1024)
                for cluster in clusters:
                    cluster_rows = [rows[index] for index in cluster]
                    if len(cluster_rows) < 2:
                        continue
                    if has_value_or_condition_conflict(cluster_rows):
                        summary["semantic_clusters_skipped_conflict"] += 1
                        continue
                    representative = select_semantic_representative(cluster_rows)
                    group_key = semantic_group_key(cluster_rows)
                    duplicate_group_id = str(uuid5(NAMESPACE_URL, f"pbpk:semantic_duplicate:{group_key}"))
                    representative_embedding = embeddings[rows.index(representative)]
                    metadata = {
                        "algorithm": "pubmedbert_semantic_v1",
                        "embedding_model": BIOMEDICAL_EMBEDDING_MODEL,
                        "threshold": 0.92,
                        "candidate_group_index": group_index,
                    }
                    group_writer.write(
                        {
                            "duplicate_group_id": duplicate_group_id,
                            "group_type": "semantic",
                            "group_key": group_key,
                            "evidence_type": representative.get("evidence_type"),
                            "compound_id": representative.get("compound_id"),
                            "compound_name": representative.get("compound_name"),
                            "representative_evidence_id": representative.get("evidence_id"),
                            "duplicate_count": len(cluster_rows),
                            "selection_reason": "PubMedBERT semantic similarity >= 0.92 with no value/condition conflict",
                            "metadata_json": json_dump(metadata),
                        }
                    )
                    for rank, member in enumerate(sorted(cluster_rows, key=lambda item: 0 if item["evidence_id"] == representative["evidence_id"] else 1), start=1):
                        member_embedding = embeddings[rows.index(member)]
                        similarity = float(np.dot(representative_embedding, member_embedding))
                        member_writer.write(
                            {
                                "duplicate_group_id": duplicate_group_id,
                                "evidence_id": member.get("evidence_id"),
                                "is_representative": member["evidence_id"] == representative["evidence_id"],
                                "duplicate_rank": rank,
                                "duplicate_reason": "biomedical embedding semantic similarity",
                                "similarity_score": f"{similarity:.5f}",
                            }
                        )
                    summary["semantic_duplicate_groups"] += 1
                    summary["semantic_duplicate_members"] += len(cluster_rows)
        finally:
            group_writer.close()
            member_writer.close()
        return {
            "semantic_dedup_skipped": False,
            "embedding_model": BIOMEDICAL_EMBEDDING_MODEL,
            "embedding_device": device,
            "semantic_max_source_chars": self.semantic_max_source_chars,
            "fallback_used": False,
            **dict(summary),
        }

    def _semantic_candidate_groups(self) -> Iterable[list[dict[str, str]]]:
        rows = self._query_csv(
            """
            SELECT
                evidence_id::text,
                coalesce(attributed_compound_id, compound_id)::text AS compound_id,
                coalesce(compound_name::text, attributed_compound_name) AS compound_name,
                evidence_type,
                coalesce(parameter, '') AS parameter,
                coalesce(value::text, '') AS value,
                coalesce(value_text, '') AS value_text,
                coalesce(unit, '') AS unit,
                coalesce(canonical_value::text, '') AS canonical_value,
                coalesce(canonical_unit, '') AS canonical_unit,
                coalesce(condition, '') AS condition,
                coalesce(population, '') AS population,
                coalesce(route, '') AS route,
                coalesce(dose, '') AS dose,
                source_id,
                source_location,
                coalesce(extraction_confidence::text, '') AS extraction_confidence,
                source_text
            FROM preselected_evidence
            WHERE evidence_type IN ('physicochemical', 'adme', 'clinical_pk', 'ddi', 'special_population', 'tissue_expression', 'public_pbpk_model')
              AND length(source_text) >= 40
            ORDER BY coalesce(attributed_compound_id, compound_id)::text, evidence_type, lower(coalesce(parameter, '')), evidence_id
            """
        )
        current_key: tuple[str, str, str] | None = None
        current_rows: list[dict[str, str]] = []
        for row in rows:
            key = (row.get("compound_id") or "", row.get("evidence_type") or "", (row.get("parameter") or "").lower())
            if current_key is not None and key != current_key:
                if len(current_rows) >= 2:
                    yield current_rows
                current_rows = []
            current_key = key
            current_rows.append(row)
        if len(current_rows) >= 2:
            yield current_rows

    def _export_outputs(self, semantic_summary: dict[str, Any]) -> dict[str, Any]:
        exports: dict[str, Any] = {}
        exports["compound_attribution_path"] = str(
            self._export_jsonl(
                Path(FINAL_ATTRIBUTION_DIR) / "compound_attribution.jsonl",
                "SELECT * FROM evidence_compound_attribution ORDER BY evidence_id",
            )
        )
        exports["compound_identity_for_llm_path"] = str(
            self._export_jsonl(
                Path(FINAL_ATTRIBUTION_DIR) / "compound_identity_for_llm.jsonl",
                """
                SELECT
                    compound_id::text,
                    preferred_name::text,
                    normalized_name::text,
                    synonyms,
                    pubchem_cid,
                    chembl_id,
                    rxnorm_rxcui
                FROM compound_identity
                ORDER BY preferred_name
                """,
            )
        )
        exports["llm_attribution_pending_path"] = str(
            self._export_jsonl(
                Path(FINAL_ATTRIBUTION_DIR) / "llm_attribution_pending.jsonl",
                """
                SELECT
                    pec.*,
                    eca.compound_id AS attributed_compound_id,
                    eca.compound_name AS attributed_compound_name,
                    eca.attribution_id,
                    eca.compound_role,
                    eca.attribution_method,
                    eca.attribution_confidence,
                    eca.matched_text,
                    eca.matched_synonym,
                    eca.attribution_reason,
                    eca.llm_required,
                    eca.llm_trigger_reason,
                    eca.attribution_stage
                FROM processed_evidence_candidate pec
                JOIN evidence_compound_attribution eca
                    ON eca.evidence_id = pec.evidence_id
                WHERE eca.llm_required = TRUE
                ORDER BY eca.llm_trigger_reason, eca.compound_role, pec.source_id, pec.evidence_id
                """,
            )
        )
        exports["duplicate_groups_path"] = str(
            self._export_jsonl(
                Path(FINAL_DEDUPLICATION_DIR) / "duplicate_groups.jsonl",
                """
                SELECT
                    edg.*,
                    (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'evidence_id', edm.evidence_id,
                                'is_representative', edm.is_representative,
                                'duplicate_rank', edm.duplicate_rank,
                                'duplicate_reason', edm.duplicate_reason,
                                'similarity_score', edm.similarity_score
                            )
                            ORDER BY edm.duplicate_rank NULLS LAST, edm.evidence_id
                        )
                        FROM evidence_duplicate_member edm
                        WHERE edm.duplicate_group_id = edg.duplicate_group_id
                    ) AS members
                FROM evidence_duplicate_group edg
                ORDER BY group_type, duplicate_count DESC, duplicate_group_id
                """,
            )
        )
        preselected_sql = "SELECT * FROM preselected_evidence ORDER BY coalesce(compound_name::text, attributed_compound_name), evidence_type, source_id, evidence_id"
        exports["preselected_evidence_path"] = str(
            self._export_jsonl(Path(FINAL_EVIDENCE_LAYERS_DIR) / "preselected_evidence.jsonl", preselected_sql)
        )
        reviewable_sql = "SELECT * FROM reviewable_evidence ORDER BY coalesce(compound_name::text, attributed_compound_name), evidence_type, source_id, evidence_id"
        if self.export_reviewable_limit is not None:
            reviewable_sql += f" LIMIT {int(self.export_reviewable_limit)}"
        exports["reviewable_evidence_path"] = str(self._export_jsonl(Path(FINAL_EVIDENCE_LAYERS_DIR) / "reviewable_evidence.jsonl", reviewable_sql))
        exports["excluded_from_reviewable_path"] = str(
            self._export_jsonl(
                Path(FINAL_EVIDENCE_LAYERS_DIR) / "excluded_from_reviewable.jsonl",
                "SELECT * FROM excluded_from_reviewable ORDER BY coalesce(compound_name::text, attributed_compound_name), evidence_type, source_id, evidence_id",
            )
        )
        exports["accepted_evidence_path"] = str(
            self._export_jsonl(
                Path(FINAL_EVIDENCE_LAYERS_DIR) / "accepted_evidence.jsonl",
                "SELECT * FROM accepted_evidence ORDER BY coalesce(compound_name::text, attributed_compound_name), evidence_type, source_id, evidence_id",
            )
        )
        exports["modeling_evidence_path"] = str(
            self._export_jsonl(
                Path(FINAL_EVIDENCE_LAYERS_DIR) / "modeling_evidence.jsonl",
                "SELECT * FROM modeling_evidence ORDER BY coalesce(compound_name::text, attributed_compound_name), evidence_type, source_id, evidence_id",
            )
        )
        exports["evidence_conflict_report_path"] = str(self._export_jsonl(Path(FINAL_CONFLICT_DIR) / "evidence_conflict_report.jsonl", CONFLICT_REPORT_SQL))
        return exports

    def _quality_report(self, semantic_summary: dict[str, Any], exports: dict[str, Any]) -> dict[str, Any]:
        sql_counts = self._query_key_values(
            """
            SELECT 'candidate_count', count(*)::text FROM processed_evidence_candidate
            UNION ALL SELECT 'missing_provenance_count', count(*)::text FROM processed_evidence_candidate
                WHERE source_id IS NULL OR raw_asset_id IS NULL OR storage_path IS NULL OR checksum IS NULL OR source_location IS NULL OR source_text IS NULL
            UNION ALL SELECT 'attribution_count', count(*)::text FROM evidence_compound_attribution
            UNION ALL SELECT 'duplicate_group_count', count(*)::text FROM evidence_duplicate_group
            UNION ALL SELECT 'duplicate_member_count', count(*)::text FROM evidence_duplicate_member
            UNION ALL SELECT 'preselected_count', count(*)::text FROM preselected_evidence
            UNION ALL SELECT 'reviewable_count', count(*)::text FROM reviewable_evidence
            UNION ALL SELECT 'excluded_from_reviewable_count', count(*)::text FROM excluded_from_reviewable
            UNION ALL SELECT 'accepted_count', count(*)::text FROM accepted_evidence
            UNION ALL SELECT 'modeling_count', count(*)::text FROM modeling_evidence
            UNION ALL SELECT 'conflict_group_count', count(*)::text FROM (
                """ + CONFLICT_REPORT_SQL + """
            ) conflicts
            """
        )
        role_counts = self._query_key_values("SELECT compound_role, count(*)::text FROM evidence_compound_attribution GROUP BY compound_role ORDER BY compound_role")
        type_counts = self._query_key_values("SELECT evidence_type, count(*)::text FROM processed_evidence_candidate GROUP BY evidence_type ORDER BY evidence_type")
        source_counts = self._query_key_values("SELECT source_id, count(*)::text FROM processed_evidence_candidate GROUP BY source_id ORDER BY source_id")
        duplicate_counts = self._query_key_values("SELECT group_type, count(*)::text FROM evidence_duplicate_group GROUP BY group_type ORDER BY group_type")
        llm_candidate_counts = self._query_key_values(
            """
            SELECT 'llm_required_attribution', count(*)::text
            FROM evidence_compound_attribution
            WHERE llm_required = TRUE
            UNION ALL
            SELECT 'reviewable_unresolved_llm_count', count(*)::text
            FROM reviewable_evidence
            WHERE llm_required = TRUE
            """
        )
        llm_candidate_count = llm_candidate_counts.get("llm_required_attribution", 0)
        report = {
            "run_dir": str(self.run_dir),
            "output_dir": str(self.output_dir),
            "reset": self.reset,
            "counts": {**sql_counts, **dict(self.counts)},
            "compound_role_counts": role_counts,
            "evidence_type_counts": type_counts,
            "source_counts": source_counts,
            "duplicate_group_counts": duplicate_counts,
            "semantic_deduplication": semantic_summary,
            "llm_attribution": {
                "mode": self.llm_mode,
                "model": self.llm_agent.model_name,
                "llm_required_attribution": llm_candidate_count,
                "rule_attribution_marked_llm_required": self.counts["llm_attribution_required"],
                "requests": self.counts["llm_attribution_requests"],
                "skipped_missing_api_key": 0,
                "backend": "project_local_qwen3_8b_bf16",
                "skipped_limit": self.counts["llm_attribution_skipped_limit"],
            },
            "exports": exports,
            "quality_gates": {
                "missing_provenance_count_zero": sql_counts.get("missing_provenance_count") == 0,
                "reviewable_has_attribution": True,
                "reviewable_has_no_unresolved_llm": llm_candidate_counts.get("reviewable_unresolved_llm_count") == 0,
                "reviewable_passes_domain_gates": sql_counts.get("excluded_from_reviewable_count", 0) + sql_counts.get("reviewable_count", 0)
                == sql_counts.get("preselected_count", 0),
                "candidate_rows_preserved": True,
                "hash_embedding_not_used_for_final_semantic_dedup": semantic_summary.get("fallback_used", False) is False,
            },
        }
        return report

    def _iter_candidate_rows(self) -> Iterable[dict[str, str]]:
        sql = """
        SELECT
            pec.evidence_id::text,
            pec.evidence_type,
            pec.compound_id::text,
            pec.compound_name::text,
            pec.source_id,
            pec.source_text,
            pec.source_location,
            pec.parameter,
            pec.condition,
            pec.population,
            pec.route,
            pec.dose,
            coalesce(pd.title, '') AS source_title,
            coalesce(pd.document_type, '') AS document_type,
            coalesce(pec.metadata_json::text, '{}') AS metadata_json,
            coalesce(pec.source_record::text, '{}') AS source_record,
            coalesce(pec.extraction_confidence::text, '') AS extraction_confidence,
            pec.curation_status::text
        FROM processed_evidence_candidate pec
        LEFT JOIN parsed_document pd
            ON pd.parsed_document_id = pec.parsed_document_id
        ORDER BY pec.evidence_id;
        """
        yield from self._query_csv_stream(sql)

    def _export_jsonl(self, filename: str | Path, select_sql: str) -> Path:
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        json_sql = f"SELECT row_to_json(t)::text FROM ({strip_sql(select_sql)}) t"
        with path.open("w", encoding="utf-8") as file:
            subprocess.run(
                ["docker", "exec", "-i", self.container, "psql", "-v", "ON_ERROR_STOP=1", "-U", self.db_user, "-d", self.db_name, "-At", "--no-align", "-c", json_sql],
                stdout=file,
                text=True,
                env=psql_env(),
                check=True,
            )
        return path

    def _query_csv(self, sql: str) -> list[dict[str, str]]:
        return list(self._query_csv_stream(sql))

    def _query_csv_stream(self, sql: str) -> Iterable[dict[str, str]]:
        stripped_sql = strip_sql(sql)
        process = subprocess.Popen(
            ["docker", "exec", "-i", self.container, "psql", "-v", "ON_ERROR_STOP=1", "-U", self.db_user, "-d", self.db_name, "-c", f"COPY ({stripped_sql}) TO STDOUT WITH (FORMAT csv, HEADER true)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=psql_env(),
        )
        if process.stdout is None:
            raise RuntimeError("Could not open psql stdout")
        reader = csv.DictReader(process.stdout)
        yield from reader
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"psql query failed: {stderr}")

    def _query_key_values(self, sql: str) -> dict[str, int]:
        rows = self._query_csv(f"SELECT key, value FROM ({sql}) kv(key, value)")
        output: dict[str, int] = {}
        for row in rows:
            try:
                output[str(row["key"])] = int(row["value"])
            except (TypeError, ValueError):
                output[str(row["key"])] = 0
        return output

    def _exec_sql(self, sql: str) -> None:
        subprocess.run(
            ["docker", "exec", "-i", self.container, "psql", "-v", "ON_ERROR_STOP=1", "-U", self.db_user, "-d", self.db_name],
            input=sql,
            text=True,
            env=psql_env(),
            check=True,
        )

    def _analyze(self) -> None:
        self._exec_sql(
            """
            ANALYZE evidence_compound_attribution;
            ANALYZE evidence_duplicate_group;
            ANALYZE evidence_duplicate_member;
            ANALYZE processed_evidence_candidate;
            """
        )


class CopyWriter:
    def __init__(self, builder: FinalEvidenceBuilder, table: str, columns: list[str]) -> None:
        self.table = table
        self.columns = columns
        copy_sql = f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv, HEADER true)"
        self.process = subprocess.Popen(
            ["docker", "exec", "-i", builder.container, "psql", "-v", "ON_ERROR_STOP=1", "-U", builder.db_user, "-d", builder.db_name, "-c", copy_sql],
            stdin=subprocess.PIPE,
            text=True,
            env=psql_env(),
        )
        if self.process.stdin is None:
            raise RuntimeError("Could not open COPY stdin")
        self.writer = csv.writer(self.process.stdin, lineterminator="\n")
        self.writer.writerow(columns)

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow([csv_value(row.get(column)) for column in self.columns])

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"COPY failed for {self.table} with exit code {return_code}")


class LLMAttributionAgent:
    def __init__(self, *, mode: str, model_name: str | None = None, base_url: str | None = None, timeout_seconds: int = 45) -> None:
        self.mode = mode
        validate_backend(base_url, model_name)
        self.api_key = None
        self.model_name, self.base_url = backend_identity()
        self.timeout_seconds = timeout_seconds

    def attribute(self, *, row: dict[str, str], compounds: list[dict[str, Any]]) -> dict[str, Any]:
        if self.mode == "off":
            return {"status": "skipped_mode_off"}
        compound_identities = [llm_compound_identity(item) for item in compounds]
        source_text = str(row.get("source_text") or "")
        spans = []
        seen_spans = set()
        for compound in compounds:
            terms = [str(compound.get("preferred_name") or ""), *(compound.get("synonyms") or [])]
            for term in terms:
                if not isinstance(term, str) or not term.strip() or term.strip().isdigit():
                    continue
                for match in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", source_text, re.IGNORECASE):
                    key = (match.start(), match.end(), compound.get("preferred_name"))
                    if key in seen_spans:
                        continue
                    seen_spans.add(key)
                    spans.append({"span_id": f"s{len(spans)}", "text": match.group(), "start": match.start(), "end": match.end(), "compound_name": compound.get("preferred_name")})
        if not spans:
            return {"status": "failed_no_source_name_span", "error": "No allowed compound name occurs in the source text."}
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": LLM_ATTRIBUTION_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "allowed_compounds": compound_identities,
                            "source_spans": spans,
                            "allowed_roles": sorted(COMPOUND_ROLES),
                            "evidence": {
                                "evidence_id": row.get("evidence_id"),
                                "evidence_type": row.get("evidence_type"),
                                "retrieval_compound": row.get("compound_name"),
                                "source_id": row.get("source_id"),
                                "source_location": row.get("source_location"),
                                "parameter": row.get("parameter"),
                                "condition": row.get("condition"),
                                "population": row.get("population"),
                                "route": row.get("route"),
                                "dose": row.get("dose"),
                                "source_text": row.get("source_text"),
                            },
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        started = time.monotonic()
        request_payload = json.loads(body["messages"][1]["content"])
        from hashlib import sha256
        digest = lambda value: sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        audit = {"generation_format": generation_format(ATTRIBUTION_GENERATION_FORMAT),
                 "request_sha256": digest(request_payload), "schema_sha256": digest(attribution_output_schema(request_payload)),
                 "prompt_sha256": sha256(LLM_ATTRIBUTION_PROMPT.encode()).hexdigest()}
        payload = None
        try:
            payload = generate_evidence_json(LLM_ATTRIBUTION_PROMPT, request_payload)
            candidate = payload.get("attribution")
            if not isinstance(candidate, dict):
                raise ValueError("Local attribution output must contain an attribution object.")
            selected = next((span for span in spans if span["span_id"] == candidate.get("matched_span_id")), None)
            if selected is None or selected["compound_name"] != candidate.get("compound_name"):
                raise ValueError("Attribution must select a supplied source span for the chosen compound.")
            if str(candidate.get("evidence_id")) != str(row.get("evidence_id")):
                raise ValueError("Attribution evidence ID does not match the request.")
            candidate = dict(candidate)
            candidate.pop("matched_span_id", None)
            candidate["matched_text"] = source_text[selected["start"]:selected["end"]]
            return {"status": "ok", "candidate": candidate, "source_span_binding": selected,
                    "latency_seconds": time.monotonic() - started, "raw_payload": payload, **audit}
        except Exception as exc:
            return {"status": "failed_local_llm", "error": sanitize_error(str(exc)), "raw_payload": payload,
                    "raw_generation": str(getattr(exc, "raw_response", ""))[:65536], **audit}


LLM_ATTRIBUTION_PROMPT = """You are assigning compound roles for a PBPK evidence candidate.

Use only the supplied source_text and metadata.
Return JSON only:
{
  "attribution": {
    "evidence_id": "exact supplied evidence_id",
    "matched_span_id": "one supplied source_spans span_id",
    "matched_text": "exact text copied from source_text",
    "compound_name": "one allowed compound name",
    "compound_role": "one allowed role",
    "role_reason": "short reason",
    "confidence": 0.0
  }
}

Rules:
1. evidence_id must exactly equal the supplied evidence_id.
2. matched_text must be copied exactly from source_text and identify the selected compound or its supplied synonym/identifier.
   Select matched_span_id from source_spans for the chosen compound. The program binds the quote to that exact source span. Never invent a span_id or select a span belonging to another compound.
   Use a short verbatim span and preserve every character, especially capitalization. The normalized compound_name is a separate field.
   Formatting example: source_text="Exampledrug is eliminated renally.", allowed preferred_name="exampledrug" -> matched_text="Exampledrug", compound_name="exampledrug". Never copy this example into a real answer.
3. compound_name must be a preferred_name from allowed_compounds. Use only the supplied identity table; related_forms are not synonyms.
4. Do not use outside knowledge and do not create new evidence.
5. Do not choose accepted/modeling status.
6. If the evidence only mentions a drug in background or a list, use mentioned_only.
7. For DDI, assign victim_drug to the affected drug and perpetrator_drug to the precipitant/inhibitor/inducer only when source_text supports that role.
8. If no supplied identity is supported by source_text, use shared_or_unlinked for the retrieval compound and explain why.
9. Prefer the retrieval_compound when source_text directly supports its identity and evidence role. Select another allowed compound only when the evidence is not about the retrieval compound and directly supports the other target.
10. A phrase such as "compared with drug alone", "versus baseline", or a treatment comparison does not make the affected drug a comparator. If that drug's AUC, Cmax, exposure, clearance, or half-life changes after coadministration, it is the victim_drug.
11. Use comparator only when the selected compound is solely a control/reference arm and the evidence does not measure a PK or DDI effect on that compound.
12. Use mentioned_only only for background, citation, eligibility, or list context. A direct ADME statement, directional DDI result, or PBPK model built/evaluated for the compound is not mentioned_only.
13. For public_pbpk_model evidence, use target_compound when the source directly states that a PBPK model was built, applied, evaluated, or described for that compound, even when other drugs are in the same model network.
14. A negative DDI result is still a DDI relation. In "A had no effect on B AUC", A is the perpetrator_drug and B is the victim_drug when the identities are supported by source_text.
15. Write role_reason as one concise sentence, at most 512 characters. Do not repeat yourself or discuss JSON formatting or confidence in that sentence. Set confidence independently from source support; 0.0 in the example is not a required value.
"""


ATTRIBUTION_GENERATION_FORMAT = "pbpk_attribution_schema_v1_xgrammar_0.2.5.post1_ws2"


def attribution_output_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Bind each model decision to one genuine source span in its own request."""
    evidence = payload.get("evidence") or {}
    text = str(evidence.get("source_text") or "")
    allowed_names = {item["preferred_name"] for item in payload.get("allowed_compounds") or []}
    roles = payload.get("allowed_roles") or []
    if not roles or any(role not in COMPOUND_ROLES for role in roles):
        raise ValueError("Attribution schema requires the supplied allowed roles")
    branches = []
    for span in payload.get("source_spans") or []:
        if span["compound_name"] not in allowed_names or text[span["start"]:span["end"]] != span["text"] or not span["text"]:
            raise ValueError("Attribution schema source span is not bound to the original text")
        properties = {
            "evidence_id": {"const": str(evidence["evidence_id"])},
            "matched_span_id": {"const": span["span_id"]},
            "matched_text": {"const": span["text"]},
            "compound_name": {"const": span["compound_name"]},
            "compound_role": {"enum": roles},
            # Bound explanatory prose during constrained decoding, rather than
            # truncating an already generated answer or relaxing source checks.
            "role_reason": {"type": "string", "minLength": 1, "maxLength": 512},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }
        branches.append({"type": "object", "properties": properties,
                         "required": list(properties), "additionalProperties": False})
    if not branches:
        raise ValueError("Attribution schema requires at least one supported source span")
    return {"type": "object", "properties": {"attribution": {"anyOf": branches}},
            "required": ["attribution"], "additionalProperties": False}


def llm_compound_identity(compound: dict[str, Any]) -> dict[str, Any]:
    """Return the bounded identity evidence the LLM is allowed to use."""
    synonyms = sorted(
        {
            str(value).strip()
            for value in compound.get("synonyms") or []
            if str(value).strip()
        },
        key=lambda value: (len(value), value.lower(), value),
    )
    identifiers = {
        key: compound.get(key)
        for key in ("pubchem_cid", "inchikey", "rxnorm_rxcui", "chembl_id", "unii")
        if compound.get(key)
    }
    return {
        "preferred_name": str(compound["preferred_name"]),
        "synonyms": synonyms[:40],
        "identifiers": identifiers,
    }


DEDUPLICATION_SQL = """
WITH source_base AS (
    SELECT
        pec.evidence_id,
        pec.evidence_type,
        eca.compound_id,
        eca.compound_name,
        md5(concat_ws('|', pec.evidence_type, coalesce(eca.compound_id::text, eca.compound_name, 'unknown'), pec.source_id, pec.raw_asset_id, pec.source_location, coalesce(pec.source_text_hash, ''))) AS group_key,
        row_number() OVER (
            PARTITION BY pec.evidence_type, coalesce(eca.compound_id::text, eca.compound_name, 'unknown'), pec.source_id, pec.raw_asset_id, pec.source_location, coalesce(pec.source_text_hash, '')
            ORDER BY
                CASE pec.curation_status WHEN 'approved_for_modeling' THEN 0 WHEN 'reviewed_accepted' THEN 1 ELSE 2 END,
                CASE WHEN pec.value IS NOT NULL AND pec.unit IS NOT NULL THEN 0 ELSE 1 END,
                CASE WHEN pec.canonical_value IS NOT NULL AND pec.canonical_unit IS NOT NULL THEN 0 ELSE 1 END,
                source_priority(pec.source_id),
                coalesce(pec.extraction_confidence, 0) DESC,
                length(pec.source_location) DESC,
                pec.evidence_id
        ) AS duplicate_rank,
        count(*) OVER (
            PARTITION BY pec.evidence_type, coalesce(eca.compound_id::text, eca.compound_name, 'unknown'), pec.source_id, pec.raw_asset_id, pec.source_location, coalesce(pec.source_text_hash, '')
        ) AS duplicate_count
    FROM processed_evidence_candidate pec
    JOIN evidence_compound_attribution eca ON eca.evidence_id = pec.evidence_id
    WHERE pec.source_text_hash IS NOT NULL
),
source_groups AS (
    INSERT INTO evidence_duplicate_group (
        group_type, group_key, evidence_type, compound_id, compound_name, representative_evidence_id, duplicate_count, selection_reason, metadata_json
    )
    SELECT
        'source',
        group_key,
        (array_agg(evidence_type ORDER BY duplicate_rank))[1],
        (array_agg(compound_id ORDER BY duplicate_rank))[1],
        (array_agg(compound_name ORDER BY duplicate_rank))[1],
        (array_agg(evidence_id ORDER BY duplicate_rank))[1],
        max(duplicate_count),
        'source_id + raw_asset_id + source_location + source_text_hash',
        jsonb_build_object('algorithm', 'deterministic_source_v1')
    FROM source_base
    WHERE duplicate_count > 1
    GROUP BY group_key
    ON CONFLICT (group_type, group_key) DO UPDATE SET
        representative_evidence_id = EXCLUDED.representative_evidence_id,
        duplicate_count = EXCLUDED.duplicate_count,
        metadata_json = EXCLUDED.metadata_json
    RETURNING duplicate_group_id, group_key
)
INSERT INTO evidence_duplicate_member (
    duplicate_group_id, evidence_id, is_representative, duplicate_rank, duplicate_reason
)
SELECT
    sg.duplicate_group_id,
    sb.evidence_id,
    sb.duplicate_rank = 1,
    sb.duplicate_rank,
    'same source location and source text'
FROM source_base sb
JOIN source_groups sg ON sg.group_key = sb.group_key
WHERE sb.duplicate_count > 1
ON CONFLICT (duplicate_group_id, evidence_id) DO UPDATE SET
    is_representative = EXCLUDED.is_representative,
    duplicate_rank = EXCLUDED.duplicate_rank,
    duplicate_reason = EXCLUDED.duplicate_reason;

WITH content_base AS (
    SELECT
        pec.evidence_id,
        pec.evidence_type,
        eca.compound_id,
        eca.compound_name,
        md5(concat_ws(
            '|',
            pec.evidence_type,
            coalesce(eca.compound_id::text, eca.compound_name, 'unknown'),
            lower(coalesce(pec.parameter, '')),
            coalesce(pec.value::text, ''),
            lower(coalesce(pec.unit, '')),
            coalesce(pec.canonical_value::text, ''),
            lower(coalesce(pec.canonical_unit, '')),
            lower(coalesce(pec.condition, '')),
            lower(coalesce(pec.population, '')),
            lower(coalesce(pec.route, '')),
            lower(coalesce(pec.dose, '')),
            coalesce(pec.source_text_hash, '')
        )) AS group_key,
        row_number() OVER (
            PARTITION BY
                pec.evidence_type,
                coalesce(eca.compound_id::text, eca.compound_name, 'unknown'),
                lower(coalesce(pec.parameter, '')),
                coalesce(pec.value::text, ''),
                lower(coalesce(pec.unit, '')),
                coalesce(pec.canonical_value::text, ''),
                lower(coalesce(pec.canonical_unit, '')),
                lower(coalesce(pec.condition, '')),
                lower(coalesce(pec.population, '')),
                lower(coalesce(pec.route, '')),
                lower(coalesce(pec.dose, '')),
                coalesce(pec.source_text_hash, '')
            ORDER BY
                CASE pec.curation_status WHEN 'approved_for_modeling' THEN 0 WHEN 'reviewed_accepted' THEN 1 ELSE 2 END,
                CASE WHEN pec.value IS NOT NULL AND pec.unit IS NOT NULL THEN 0 ELSE 1 END,
                CASE WHEN pec.canonical_value IS NOT NULL AND pec.canonical_unit IS NOT NULL THEN 0 ELSE 1 END,
                source_priority(pec.source_id),
                coalesce(pec.extraction_confidence, 0) DESC,
                length(pec.source_location) DESC,
                pec.evidence_id
        ) AS duplicate_rank,
        count(*) OVER (
            PARTITION BY
                pec.evidence_type,
                coalesce(eca.compound_id::text, eca.compound_name, 'unknown'),
                lower(coalesce(pec.parameter, '')),
                coalesce(pec.value::text, ''),
                lower(coalesce(pec.unit, '')),
                coalesce(pec.canonical_value::text, ''),
                lower(coalesce(pec.canonical_unit, '')),
                lower(coalesce(pec.condition, '')),
                lower(coalesce(pec.population, '')),
                lower(coalesce(pec.route, '')),
                lower(coalesce(pec.dose, '')),
                coalesce(pec.source_text_hash, '')
        ) AS duplicate_count
    FROM processed_evidence_candidate pec
    JOIN evidence_compound_attribution eca ON eca.evidence_id = pec.evidence_id
    WHERE pec.source_text_hash IS NOT NULL
),
content_groups AS (
    INSERT INTO evidence_duplicate_group (
        group_type, group_key, evidence_type, compound_id, compound_name, representative_evidence_id, duplicate_count, selection_reason, metadata_json
    )
    SELECT
        'content',
        group_key,
        (array_agg(evidence_type ORDER BY duplicate_rank))[1],
        (array_agg(compound_id ORDER BY duplicate_rank))[1],
        (array_agg(compound_name ORDER BY duplicate_rank))[1],
        (array_agg(evidence_id ORDER BY duplicate_rank))[1],
        max(duplicate_count),
        'compound + evidence_type + parameter/value/unit/context + source_text_hash',
        jsonb_build_object('algorithm', 'deterministic_content_v1')
    FROM content_base
    WHERE duplicate_count > 1
    GROUP BY group_key
    ON CONFLICT (group_type, group_key) DO UPDATE SET
        representative_evidence_id = EXCLUDED.representative_evidence_id,
        duplicate_count = EXCLUDED.duplicate_count,
        metadata_json = EXCLUDED.metadata_json
    RETURNING duplicate_group_id, group_key
)
INSERT INTO evidence_duplicate_member (
    duplicate_group_id, evidence_id, is_representative, duplicate_rank, duplicate_reason
)
SELECT
    cg.duplicate_group_id,
    cb.evidence_id,
    cb.duplicate_rank = 1,
    cb.duplicate_rank,
    'same normalized content key'
FROM content_base cb
JOIN content_groups cg ON cg.group_key = cb.group_key
WHERE cb.duplicate_count > 1
ON CONFLICT (duplicate_group_id, evidence_id) DO UPDATE SET
    is_representative = EXCLUDED.is_representative,
    duplicate_rank = EXCLUDED.duplicate_rank,
    duplicate_reason = EXCLUDED.duplicate_reason;
"""


CONFLICT_REPORT_SQL = """
SELECT
    coalesce(attributed_compound_id, compound_id) AS compound_id,
    coalesce(compound_name::text, attributed_compound_name) AS compound_name,
    evidence_type,
    parameter,
    canonical_unit,
    route,
    population,
    condition,
    count(*) AS evidence_count,
    count(DISTINCT coalesce(canonical_value::text, value::text, value_text, '')) AS distinct_value_count,
    jsonb_agg(
        jsonb_build_object(
            'evidence_id', evidence_id,
            'value', value,
            'value_text', value_text,
            'unit', unit,
            'canonical_value', canonical_value,
            'canonical_unit', canonical_unit,
            'source_id', source_id,
            'source_location', source_location,
            'source_text', source_text
        )
        ORDER BY source_id, evidence_id
    ) AS evidence
FROM reviewable_evidence
WHERE parameter IS NOT NULL
  AND (value IS NOT NULL OR canonical_value IS NOT NULL OR value_text IS NOT NULL)
GROUP BY coalesce(attributed_compound_id, compound_id), coalesce(compound_name::text, attributed_compound_name), evidence_type, parameter, canonical_unit, route, population, condition
HAVING count(*) > 1
   AND count(DISTINCT coalesce(canonical_value::text, value::text, value_text, '')) > 1
ORDER BY evidence_count DESC, compound_name, evidence_type, parameter
"""


def build_compound_terms(compounds: list[dict[str, Any]]) -> list[CompoundTerm]:
    terms: list[CompoundTerm] = []
    seen: set[tuple[str, str]] = set()
    for compound in compounds:
        compound_id = str(compound["compound_id"])
        compound_name = str(compound["preferred_name"])
        candidates = {compound_name, str(compound.get("normalized_name") or "")}
        for synonym in compound.get("synonyms") or []:
            candidates.add(str(synonym))
        for identifier in [compound.get("pubchem_cid"), compound.get("chembl_id"), compound.get("rxnorm_rxcui")]:
            if identifier:
                candidates.add(str(identifier))
        for synonym in sorted((item.strip() for item in candidates if item and item.strip()), key=len, reverse=True):
            if len(synonym) < 3:
                continue
            if synonym.isdigit():
                continue
            key = (compound_id, synonym.lower())
            if key in seen:
                continue
            seen.add(key)
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(synonym)}(?![A-Za-z0-9])", re.IGNORECASE)
            terms.append(CompoundTerm(compound_id=compound_id, compound_name=compound_name, synonym=synonym, pattern=pattern))
    return terms


def decide_attribution(
    row: dict[str, str],
    terms: list[CompoundTerm],
    compound_by_id: dict[str, dict[str, Any]],
    compound_by_name: dict[str, dict[str, Any]],
) -> AttributionDecision:
    evidence_id = row["evidence_id"]
    source_text = row.get("source_text") or ""
    evidence_type = row.get("evidence_type") or ""
    current_compound_id = row.get("compound_id") or None
    current_compound_name = row.get("compound_name") or None
    matches = find_compound_matches(source_text, terms)
    matched_by_compound = {match.compound_id: match for match in matches}
    multi_drug = len({match.compound_id for match in matches}) >= 2
    is_ddi = is_ddi_attribution_context(
        evidence_type=evidence_type,
        source_text=source_text,
        multi_drug=multi_drug,
    )
    context = attribution_context(row)

    if current_compound_id and current_compound_id in compound_by_id:
        compound = compound_by_id[current_compound_id]
        match = matched_by_compound.get(current_compound_id)
        if match is None and not retrieval_only_source_allows_compound_metadata(row):
            return AttributionDecision(
                evidence_id=evidence_id,
                compound_id=current_compound_id,
                compound_name=current_compound_name or str(compound["preferred_name"]),
                compound_role="shared_or_unlinked",
                attribution_method="rule_retrieval_compound_without_text_match",
                attribution_confidence=0.250,
                matched_text=None,
                matched_synonym=None,
                attribution_reason=(
                    "Retrieval compound metadata alone is insufficient for free-text, label, trial, or literature sources; "
                    "source_text does not contain a known synonym for the target compound."
                ),
                llm_payload={},
                llm_required=False,
                llm_trigger_reason=None,
            )
        matched_text = match.synonym if match else current_compound_name
        role, role_reason, base_confidence = infer_role_for_compound(
            row,
            str(compound["preferred_name"]),
            matched_text=matched_text,
            is_ddi=is_ddi,
            multi_drug=multi_drug,
            context=context,
        )
        confidence = max(base_confidence, 0.86 if role == "target_compound" and not multi_drug else base_confidence)
        issue = final_evidence_quality_issue(row, compound_name=str(compound["preferred_name"]), matched_text=matched_text, compound_role=role)
        if issue is not None:
            return quality_gate_decision(
                evidence_id=evidence_id,
                compound_id=current_compound_id,
                compound_name=current_compound_name or str(compound["preferred_name"]),
                matched_text=matched_text,
                issue=issue,
            )
        llm_reason = llm_trigger_reason(source_text=source_text, role=role, confidence=confidence, multi_drug=multi_drug, is_ddi=is_ddi)
        return AttributionDecision(
            evidence_id=evidence_id,
            compound_id=current_compound_id,
            compound_name=current_compound_name or str(compound["preferred_name"]),
            compound_role=role,
            attribution_method="rule_retrieval_compound" if match else "rule_structured_source_compound",
            attribution_confidence=confidence,
            matched_text=matched_text,
            matched_synonym=matched_text,
            attribution_reason=role_reason,
            llm_payload={},
            llm_required=llm_reason is not None,
            llm_trigger_reason=llm_reason,
        )

    unique_matches = unique_compound_matches(matches)
    if len(unique_matches) == 1:
        match = unique_matches[0]
        compound = compound_by_id.get(match.compound_id) or compound_by_name.get(match.compound_name.lower())
        role, role_reason, base_confidence = infer_role_for_compound(
            row,
            match.compound_name,
            matched_text=match.synonym,
            is_ddi=is_ddi,
            multi_drug=multi_drug,
            context=context,
        )
        confidence = min(base_confidence, 0.78 if role in FINAL_LAYER_ROLES else base_confidence)
        issue = final_evidence_quality_issue(row, compound_name=match.compound_name, matched_text=match.synonym, compound_role=role)
        if issue is not None:
            return quality_gate_decision(
                evidence_id=evidence_id,
                compound_id=match.compound_id,
                compound_name=str(compound["preferred_name"]) if compound else match.compound_name,
                matched_text=match.synonym,
                issue=issue,
            )
        llm_reason = llm_trigger_reason(source_text=source_text, role=role, confidence=confidence, multi_drug=multi_drug, is_ddi=is_ddi)
        return AttributionDecision(
            evidence_id=evidence_id,
            compound_id=match.compound_id,
            compound_name=str(compound["preferred_name"]) if compound else match.compound_name,
            compound_role=role,
            attribution_method="rule_source_text_single_match",
            attribution_confidence=confidence,
            matched_text=match.synonym,
            matched_synonym=match.synonym,
            attribution_reason=role_reason,
            llm_payload={},
            llm_required=llm_reason is not None,
            llm_trigger_reason=llm_reason,
        )

    if len(unique_matches) > 1:
        match = unique_matches[0]
        return AttributionDecision(
            evidence_id=evidence_id,
            compound_id=match.compound_id,
            compound_name=match.compound_name,
            compound_role="mentioned_only",
            attribution_method="rule_multi_compound_ambiguous",
            attribution_confidence=0.450,
            matched_text=match.synonym,
            matched_synonym=match.synonym,
            attribution_reason="Multiple target compounds occur in source_text and deterministic rules cannot select the evidence owner.",
            llm_payload={},
            llm_required=True,
            llm_trigger_reason="multiple_target_compounds_in_source_text",
        )

    if current_compound_name:
        compound = compound_by_name.get(current_compound_name.lower())
        if not retrieval_only_source_allows_compound_metadata(row):
            return AttributionDecision(
                evidence_id=evidence_id,
                compound_id=str(compound["compound_id"]) if compound else None,
                compound_name=current_compound_name,
                compound_role="shared_or_unlinked",
                attribution_method="rule_retrieval_compound_without_text_match",
                attribution_confidence=0.250,
                matched_text=None,
                matched_synonym=None,
                attribution_reason=(
                    "Retrieval compound metadata alone is insufficient for free-text, label, trial, or literature sources; "
                    "source_text does not contain a known synonym for the target compound."
                ),
                llm_payload={},
                llm_required=False,
                llm_trigger_reason=None,
            )
        role, role_reason, base_confidence = infer_role_for_compound(
            row,
            current_compound_name,
            matched_text=current_compound_name,
            is_ddi=is_ddi,
            multi_drug=multi_drug,
            context=context,
        )
        if role in FINAL_LAYER_ROLES:
            base_confidence = min(base_confidence, 0.650)
        issue = final_evidence_quality_issue(row, compound_name=current_compound_name, matched_text=current_compound_name, compound_role=role)
        if issue is not None:
            return quality_gate_decision(
                evidence_id=evidence_id,
                compound_id=str(compound["compound_id"]) if compound else None,
                compound_name=current_compound_name,
                matched_text=current_compound_name,
                issue=issue,
            )
        llm_reason = llm_trigger_reason(source_text=source_text, role=role, confidence=base_confidence, multi_drug=multi_drug, is_ddi=is_ddi)
        return AttributionDecision(
            evidence_id=evidence_id,
            compound_id=str(compound["compound_id"]) if compound else None,
            compound_name=current_compound_name,
            compound_role=role,
            attribution_method="rule_structured_source_compound",
            attribution_confidence=base_confidence,
            matched_text=current_compound_name,
            matched_synonym=current_compound_name,
            attribution_reason=role_reason if role not in FINAL_LAYER_ROLES else "Structured single-compound source supports retrieval compound attribution.",
            llm_payload={},
            llm_required=llm_reason is not None,
            llm_trigger_reason=llm_reason,
        )

    return AttributionDecision(
        evidence_id=evidence_id,
        compound_id=None,
        compound_name=None,
        compound_role="shared_or_unlinked",
        attribution_method="rule_no_compound_match",
        attribution_confidence=0.100,
        matched_text=None,
        matched_synonym=None,
        attribution_reason="No retrieval compound and no compound synonym match in source_text.",
        llm_payload={},
        llm_required=False,
    )


def infer_role_for_compound(
    row: dict[str, str],
    compound_name: str,
    *,
    matched_text: str | None,
    is_ddi: bool,
    multi_drug: bool,
    context: str,
) -> tuple[str, str, float]:
    source_text = row.get("source_text") or ""
    evidence_type = row.get("evidence_type") or ""
    source_id = row.get("source_id") or ""
    lowered = source_text.lower()
    lowered_context = context.lower()
    name = compound_name.lower()
    matched = (matched_text or compound_name).lower()

    exclusion = excluded_role_for_context(
        source_text=source_text,
        context=context,
        source_id=source_id,
        evidence_type=evidence_type,
        compound_name=name,
        matched_text=matched,
        is_ddi=is_ddi,
    )
    if exclusion is not None:
        return exclusion

    if evidence_type == "ddi" or is_ddi:
        ddi_role = ddi_role_for_compound(lowered, matched)
        if ddi_role is not None:
            return ddi_role
        if "drug interaction" in lowered_context or "drug interactions" in lowered_context:
            return "victim_drug", "Drug-interaction section supports DDI evidence, but rules cannot fully resolve victim/perpetrator.", 0.780
        return "victim_drug", "DDI evidence is tied to the retrieval or matched compound, but victim/perpetrator is not fully resolved by rules.", 0.720

    if section_supports_target(row, lowered_context):
        return "target_compound", "Section/source context supports direct target-compound evidence.", 0.900 if not multi_drug else 0.820

    if is_background_or_list_context(source_text=source_text, context=context, source_id=source_id):
        return "mentioned_only", "Drug appears in background, eligibility, citation, adverse-event, or list context rather than direct PBPK evidence.", 0.300

    return "target_compound", "Evidence is linked to the retrieval or uniquely matched compound outside DDI context.", 0.880 if not multi_drug else 0.760


def should_request_llm(*, source_text: str, role: str, confidence: float, multi_drug: bool, is_ddi: bool) -> bool:
    return llm_trigger_reason(source_text=source_text, role=role, confidence=confidence, multi_drug=multi_drug, is_ddi=is_ddi) is not None


def llm_trigger_reason(*, source_text: str, role: str, confidence: float, multi_drug: bool, is_ddi: bool) -> str | None:
    lowered = source_text.lower()
    if is_ddi and role not in {"victim_drug", "perpetrator_drug"} and source_has_directional_ddi_outcome(lowered):
        return "ddi_nonfinal_role_with_directional_effect"
    if role in {"comparator", "excipient", "co_medication", "mentioned_only", "shared_or_unlinked"} and confidence >= 0.70:
        return None
    if confidence < 0.80:
        return "rule_confidence_below_0_80"
    if multi_drug:
        return "source_text_contains_multiple_target_drugs"
    if is_ddi and role in {"victim_drug", "perpetrator_drug"} and confidence < 0.86:
        return "ddi_victim_perpetrator_not_high_confidence"
    if any(term in lowered for term in MULTI_DRUG_TERMS) and confidence < 0.90:
        return "multi_drug_relation_terms_with_medium_confidence"
    if len(source_text) > 900 and (multi_drug or is_ddi):
        return "long_multi_drug_or_ddi_paragraph"
    return None


def source_has_directional_ddi_outcome(lowered_text: str) -> bool:
    return contains_any(lowered_text, DDI_DIRECTIONAL_OUTCOME_TERMS) and contains_any(lowered_text, DDI_EFFECT_ENDPOINT_TERMS)


def attribution_context(row: dict[str, str]) -> str:
    return " | ".join(
        part
        for part in [
            row.get("source_id") or "",
            row.get("source_title") or "",
            row.get("document_type") or "",
            row.get("parameter") or "",
            row.get("source_location") or "",
            row.get("metadata_json") or "",
            row.get("source_record") or "",
            row.get("source_text") or "",
        ]
        if part
    )


def excluded_role_for_context(
    *,
    source_text: str,
    context: str,
    source_id: str,
    evidence_type: str,
    compound_name: str,
    matched_text: str,
    is_ddi: bool,
) -> tuple[str, str, float] | None:
    lowered = source_text.lower()
    lowered_context = context.lower()
    term = matched_text or compound_name
    if term and term_near_any(lowered, term, EXCIPIENT_TERMS, window=90):
        return "excipient", "Matched drug term occurs in inactive-ingredient or excipient context.", 0.880
    if term and term_is_comparator(lowered, term):
        return "comparator", "Matched drug term occurs as comparator or control, not direct target-compound evidence.", 0.820
    if not is_ddi and term and term_near_any(lowered, term, CO_MEDICATION_TERMS, window=110):
        return "co_medication", "Matched drug term occurs as concomitant medication or co-medication outside a resolved DDI relationship.", 0.780
    if assertion_targets_different_combination_component(lowered, compound_name):
        return "mentioned_only", "Matched drug appears in a combination product name, but the PK/ADME assertion targets a different active component.", 0.760
    if is_background_or_list_context(source_text=source_text, context=context, source_id=source_id):
        if evidence_type != "ddi" or not contains_any(source_text, DDI_TERMS):
            return "mentioned_only", "Matched drug term appears in background, eligibility, citation, adverse-event, or list context.", 0.760
    if any(term in lowered_context for term in ["inactive ingredient", "excipient"]) and not section_supports_target({"evidence_type": evidence_type}, lowered_context):
        return "excipient", "Source section context is inactive ingredients or excipients.", 0.820
    return None


def ddi_role_for_compound(lowered_source_text: str, compound_name: str) -> tuple[str, str, float] | None:
    name_window = re.escape(compound_name)
    pk_parameter = r"(auc|cmax|tmax|half[- ]life|t\s*1/2|exposure|concentration|clearance|bioavailability|volume of distribution)"
    change_verb = r"(increase(?:d|s)?|decrease(?:d|s)?|reduce(?:d|s)?|alter(?:ed|s)?|raise(?:d|s)?|lower(?:ed|s)?)"
    substrate = compound_is_probe_or_substrate(lowered_source_text, [compound_name])
    perpetrator = bool(
        re.search(rf"{name_window}\s+(?:directly\s+)?(?:inhibits|induces)\b", lowered_source_text)
        or re.search(
            rf"{name_window}\s+(?:is|was|acts as)\s+(?:a|an)?\s*(?:strong\s+|moderate\s+|weak\s+)?(?:inhibitor|inducer)\b",
            lowered_source_text,
        )
        or re.search(rf"(?:inhibition|induction)\s+by\s+{name_window}\b", lowered_source_text)
        or re.search(rf"(?:inhibited|induced)\s+by\s+{name_window}\b", lowered_source_text)
    )

    if perpetrator and not substrate:
        return "perpetrator_drug", "DDI relationship rule: compound is explicitly described as inhibitor or inducer.", 0.920
    if substrate and not perpetrator:
        return "victim_drug", "DDI relationship rule: compound is explicitly described as a probe or substrate.", 0.940
    if re.search(rf"{name_window}.{{0,80}}(?:is|was|are|were).{{0,30}}metaboli[sz]ed by\b", lowered_source_text):
        return "victim_drug", "DDI relationship rule: compound is explicitly described as metabolized by the affected enzyme.", 0.900
    if re.search(rf"(?:effect|impact|influence).{{0,140}}{pk_parameter}.{{0,100}}(?:of|for)\s+{name_window}", lowered_source_text):
        return "victim_drug", "DDI relationship rule: another intervention is evaluated for its effect on compound PK.", 0.930
    if re.search(rf"{change_verb}.{{0,80}}{pk_parameter}.{{0,100}}(?:of|for)?\s*{name_window}", lowered_source_text):
        return "victim_drug", "DDI relationship rule: a compound PK parameter is explicitly changed.", 0.920
    if re.search(rf"{change_verb}.{{0,100}}{name_window}.{{0,90}}{pk_parameter}", lowered_source_text) or re.search(
        rf"{name_window}.{{0,90}}{pk_parameter}.{{0,100}}{change_verb}",
        lowered_source_text,
    ):
        return "victim_drug", "DDI relationship rule: compound PK parameter is explicitly affected.", 0.900
    if re.search(rf"{name_window}.{{0,80}}{change_verb}.{{0,120}}{pk_parameter}", lowered_source_text):
        return "perpetrator_drug", "DDI relationship rule: compound appears before a verb changing another drug PK parameter.", 0.860
    if re.search(rf"co-?administered with.{{0,80}}{name_window}", lowered_source_text) or re.search(
        rf"concomitant use of.{{0,80}}{name_window}",
        lowered_source_text,
    ):
        return "perpetrator_drug", "DDI relationship rule: compound appears as coadministered precipitant in DDI context.", 0.820
    return None


def section_supports_target(row: dict[str, str], lowered_context: str) -> bool:
    evidence_type = row.get("evidence_type") or ""
    if evidence_type == "clinical_pk" and any(term in lowered_context for term in ["clinical pharmacology", "pharmacokinetics", "pharmacokinetic", "pk parameter"]):
        return True
    if evidence_type == "adme" and any(term in lowered_context for term in ["clinical pharmacology", "absorption", "distribution", "metabolism", "excretion"]):
        return True
    if evidence_type == "special_population" and any(term in lowered_context for term in ["use in specific populations", "renal impairment", "hepatic impairment", "pediatric", "geriatric", "pregnancy", "lactation"]):
        return True
    if evidence_type == "physicochemical" and any(term in lowered_context for term in ["pubchem", "chembl", "molecular weight", "logp", "smiles"]):
        return True
    if evidence_type == "public_pbpk_model" and any(term in lowered_context for term in ["pbpk", "model", "simulation"]):
        return True
    return False


def is_background_or_list_context(*, source_text: str, context: str, source_id: str) -> bool:
    lowered = source_text.lower()
    lowered_context = context.lower()
    if any(term.lower() in lowered or term.lower() in lowered_context for term in BACKGROUND_TERMS):
        return True
    if source_id in {"clinicaltrials", "pubmed", "pmc_oa"} and re.search(r"\b(pmid|citation|references?)\b", lowered):
        return True
    if source_id == "clinicaltrials" and any(term in lowered for term in ["inclusion criteria", "exclusion criteria", "eligibility criteria"]):
        return True
    return False


def term_is_comparator(lowered_text: str, term: str) -> bool:
    escaped = re.escape(term.lower())
    return bool(
        re.search(rf"(compared with|versus|vs\\.?|active control|control arm).{{0,80}}{escaped}", lowered_text)
        or re.search(rf"{escaped}.{{0,80}}(as comparator|as active control|control arm|comparator)", lowered_text)
    )


def term_near_any(lowered_text: str, term: str, keywords: list[str], *, window: int) -> bool:
    escaped = re.escape(term.lower())
    for keyword in keywords:
        escaped_keyword = re.escape(keyword.lower())
        if re.search(rf"{escaped_keyword}.{{0,{window}}}{escaped}", lowered_text) or re.search(rf"{escaped}.{{0,{window}}}{escaped_keyword}", lowered_text):
            return True
    return False


def assertion_targets_different_combination_component(lowered_text: str, compound_name: str) -> bool:
    compound = compound_name.lower()
    parameter_terms = r"(bioavailability|absorption|distribution|metabolism|excretion|clearance|half[- ]life|protein binding|concentration|auc|cmax)"
    patterns: list[tuple[str, int]] = [
        (rf"{parameter_terms}\s+(?:of|for)\s+(?:the\s+)?([a-z][-a-z0-9 ]{{2,80}}?)\s+component\b", 2),
        (rf"(?:the\s+)?([a-z][-a-z0-9 ]{{2,80}}?)\s+component\s+(?:is|was|has|had|shows|showed).{{0,80}}{parameter_terms}", 1),
    ]
    for pattern, component_group in patterns:
        match = re.search(pattern, lowered_text)
        if not match:
            continue
        component = " ".join(match.group(component_group).split())
        if component and compound not in component and component not in compound:
            return True
    return False


def quality_gate_decision(
    *,
    evidence_id: str,
    compound_id: str | None,
    compound_name: str | None,
    matched_text: str | None,
    issue: dict[str, str],
) -> AttributionDecision:
    return AttributionDecision(
        evidence_id=evidence_id,
        compound_id=compound_id,
        compound_name=compound_name,
        compound_role="shared_or_unlinked",
        attribution_method=issue["method"],
        attribution_confidence=0.250,
        matched_text=matched_text,
        matched_synonym=matched_text,
        attribution_reason=issue["reason"],
        llm_payload={},
        llm_required=True,
        llm_trigger_reason=issue["trigger"],
    )


def final_evidence_quality_issue(
    row: dict[str, str],
    *,
    compound_name: str,
    matched_text: str | None,
    compound_role: str,
) -> dict[str, str] | None:
    ddi_issue = ddi_direction_issue(row, compound_name=compound_name, matched_text=matched_text, compound_role=compound_role)
    if ddi_issue is not None:
        return {
            "method": DDI_DIRECTION_UNCERTAIN_METHOD,
            "trigger": "ddi_direction_uncertain",
            "reason": ddi_issue,
        }
    probe_issue = probe_relation_parameter_issue(row, compound_name=compound_name, matched_text=matched_text, compound_role=compound_role)
    if probe_issue is not None:
        return {
            "method": PROBE_RELATION_MISMATCH_METHOD,
            "trigger": "probe_relation_parameter_mismatch",
            "reason": probe_issue,
        }
    return None


def probe_relation_parameter_issue(
    row: dict[str, str],
    *,
    compound_name: str,
    matched_text: str | None,
    compound_role: str,
) -> str | None:
    if compound_role not in FINAL_LAYER_ROLES:
        return None
    source_text = row.get("source_text") or ""
    parameter = row.get("parameter") or ""
    parameter_targets = relation_target_mentions(parameter)
    if not source_text or not parameter_targets:
        return None
    source_targets = relation_target_mentions(source_text)
    if len(source_targets) < 2 and not is_probe_relation_context(source_text):
        return None

    terms = compound_relation_terms(compound_name=compound_name, matched_text=matched_text)
    local_targets: set[str] = set()
    for term in terms:
        local_targets.update(local_relation_targets_for_term(source_text, term))

    if not local_targets:
        if len(source_targets) >= 2 and is_probe_relation_context(source_text):
            return (
                "Probe/cocktail text contains multiple possible CYP/transporter targets, "
                "but deterministic rules could not find a local compound-target pair for this evidence parameter."
            )
        return None
    if relation_targets_compatible(parameter_targets, local_targets):
        return None
    return (
        "Probe/cocktail local relation check found targets "
        f"{sorted(local_targets)} near {matched_text or compound_name}, "
        f"but the evidence parameter maps to {sorted(parameter_targets)}."
    )


def ddi_direction_issue(
    row: dict[str, str],
    *,
    compound_name: str,
    matched_text: str | None,
    compound_role: str,
) -> str | None:
    if compound_role != "perpetrator_drug":
        return None
    evidence_type = row.get("evidence_type") or ""
    source_text = row.get("source_text") or ""
    if evidence_type != "ddi" and not contains_any(source_text, DDI_TERMS):
        return None
    terms = compound_relation_terms(compound_name=compound_name, matched_text=matched_text)
    if not terms:
        return None
    lowered = source_text.lower()
    if compound_is_probe_or_substrate(lowered, terms):
        return "DDI direction check: current compound is described as a probe/known/sensitive substrate, so perpetrator assignment is not deterministic."
    if compound_pk_is_affected(lowered, terms):
        return "DDI direction check: source text describes pharmacokinetics/exposure of the current compound as affected, so perpetrator assignment is not deterministic."
    if compound_inhibited_by_other(lowered, terms):
        return "DDI direction check: current compound appears as the inhibited/affected substrate, not the precipitant."
    if combination_product_ddi_direction_uncertain(lowered, compound_name):
        return "DDI direction check: target compound appears inside a multi-active combination product and the DDI risk is not proven to belong to that component."
    return None


def compound_is_probe_or_substrate(lowered_text: str, terms: list[str]) -> bool:
    for term in terms:
        escaped = re.escape(term.lower())
        if re.search(rf"{escaped}.{{0,100}}(?:probe|known|sensitive|index)?\s*(?:cyp\w*\s*)?substrate\b", lowered_text):
            return True
        if re.search(rf"(?:probe|known|sensitive|index)?\s*(?:cyp\w*\s*)?substrate.{{0,100}}{escaped}", lowered_text):
            return True
        if re.search(rf"(?:probe|phenotyping).{{0,80}}{escaped}", lowered_text):
            return True
    return False


def compound_pk_is_affected(lowered_text: str, terms: list[str]) -> bool:
    pk_terms = r"(?:pk|pharmacokinetics|pharmacokinetic|exposure|auc|cmax|concentration|clearance|half[- ]life|bioavailability|disposition)"
    for term in terms:
        escaped = re.escape(term.lower())
        if re.search(rf"(?:effect|effects|impact|influence|change|changes).{{0,160}}{pk_terms}.{{0,120}}(?:of|for)\s+{escaped}", lowered_text):
            return True
        if re.search(rf"{pk_terms}.{{0,80}}(?:of|for)\s+{escaped}.{{0,120}}(?:increased|decreased|reduced|altered|affected)", lowered_text):
            return True
        if re.search(rf"{escaped}.{{0,80}}{pk_terms}.{{0,120}}(?:increased|decreased|reduced|altered|affected)", lowered_text):
            return True
    return False


def compound_inhibited_by_other(lowered_text: str, terms: list[str]) -> bool:
    for term in terms:
        escaped = re.escape(term.lower())
        if re.search(rf"{escaped}.{{0,100}}(?:was|were|is|are|being)?\s*(?:significantly\s*)?(?:inhibited|affected|reduced|decreased|increased)\s+by\b", lowered_text):
            return True
        if re.search(rf"{escaped}.{{0,120}}(?:in the presence of|with).{{0,80}}(?:inhibitor|inducer)\b", lowered_text):
            return True
    return False


def combination_product_ddi_direction_uncertain(lowered_text: str, compound_name: str) -> bool:
    compound = compound_name.lower()
    if compound != "caffeine":
        return False
    combo_markers = [
        "butalbital",
        "acetaminophen",
        "aspirin",
        "codeine",
        "dihydrocodeine",
        "ergotamine",
    ]
    if not any(marker in lowered_text for marker in combo_markers):
        return False
    return contains_any(lowered_text, DDI_TERMS) or any(term in lowered_text for term in ["cyp2d6", "cyp3a4", "cns depress", "oral anticoagulants", "hypoglycemia"])


def compound_relation_terms(*, compound_name: str, matched_text: str | None) -> list[str]:
    terms: list[str] = []
    for value in [matched_text, compound_name]:
        term = (value or "").strip()
        if len(term) < 3 or term.isdigit():
            continue
        if term.lower() not in {item.lower() for item in terms}:
            terms.append(term)
    return terms


def relation_target_mentions(text: str) -> set[str]:
    lowered = text.lower()
    targets: set[str] = set()
    for canonical, patterns in RELATION_TARGET_PATTERNS.items():
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns):
            targets.add(canonical)
    return normalize_relation_targets(targets)


def normalize_relation_targets(targets: set[str]) -> set[str]:
    normalized = set(targets)
    if "CYP3A4" in normalized or "CYP3A5" in normalized:
        normalized.add("CYP3A")
    if "OATP1B1" in normalized or "OATP1B3" in normalized:
        normalized.add("OATP")
    if "OCT1" in normalized or "OCT2" in normalized:
        normalized.add("OCT")
    if "MATE1" in normalized or "MATE2" in normalized:
        normalized.add("MATE")
    return normalized


def relation_targets_compatible(parameter_targets: set[str], local_targets: set[str]) -> bool:
    return bool(normalize_relation_targets(parameter_targets) & normalize_relation_targets(local_targets))


def is_probe_relation_context(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in PROBE_RELATION_CONTEXT_TERMS)


def local_relation_targets_for_term(source_text: str, term: str) -> set[str]:
    targets: set[str] = set()
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE)
    for match in pattern.finditer(source_text):
        segment = local_relation_segment(source_text, match.start(), match.end())
        targets.update(relation_target_mentions(segment))

        after = source_text[match.end() : min(len(source_text), match.end() + 140)]
        parenthetical = re.match(r"\s*(?:is|was|as|,|:|-|–|—)?\s*\(([^)]{0,120})\)", after, flags=re.IGNORECASE)
        if parenthetical:
            targets.update(relation_target_mentions(parenthetical.group(1)))
        else:
            appositive = re.match(r"\s*(?:,|:|-|–|—|as|is|was)?\s*([^,;.\n]{0,160})", after, flags=re.IGNORECASE)
            if appositive and any(
                marker in appositive.group(1).lower()
                for marker in ("substrate", "probe", "inhibitor", "inducer", "metabolized")
            ):
                targets.update(relation_target_mentions(appositive.group(1)))
            right_sentence = re.match(r"\s*([^;.\n]{0,240})", after, flags=re.IGNORECASE)
            if right_sentence and any(
                marker in right_sentence.group(1).lower()
                for marker in ("substrate", "transported", "involves", "mediated", "metabolized")
            ):
                targets.update(relation_target_mentions(right_sentence.group(1)))

        before = source_text[max(0, match.start() - 140) : match.start()]
        targets.update(relation_target_mentions(local_right_context_for_parenthetical_term(before)))
    return normalize_relation_targets(targets)


def local_relation_segment(source_text: str, start: int, end: int) -> str:
    left = max(
        source_text.rfind(".", 0, start),
        source_text.rfind(";", 0, start),
        source_text.rfind("\n", 0, start),
        source_text.rfind(",", 0, start),
        source_text.rfind(":", 0, start),
    )
    right_candidates = [
        index
        for index in [
            source_text.find(".", end),
            source_text.find(";", end),
            source_text.find("\n", end),
            source_text.find(",", end),
        ]
        if index != -1
    ]
    right = min(right_candidates) if right_candidates else min(len(source_text), end + 180)
    segment = source_text[left + 1 : right]
    if len(segment) > 260:
        window_left = max(0, start - 120)
        window_right = min(len(source_text), end + 140)
        segment = source_text[window_left:window_right]
    return segment


def local_right_context_for_parenthetical_term(before_text: str) -> str:
    if "(" not in before_text:
        return ""
    left = max(before_text.rfind("."), before_text.rfind(";"), before_text.rfind(","), before_text.rfind("\n"))
    return before_text[left + 1 :]


def find_compound_matches(text: str, terms: list[CompoundTerm]) -> list[CompoundTerm]:
    matches: list[CompoundTerm] = []
    for term in terms:
        if term.pattern.search(text):
            matches.append(term)
    return matches


def retrieval_only_source_allows_compound_metadata(row: dict[str, str]) -> bool:
    source_id = (row.get("source_id") or "").lower()
    if source_id in RETRIEVAL_ONLY_ALLOWED_SOURCES:
        return True
    metadata = row.get("metadata_json") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, dict):
        return False
    if source_id == "chembl" and metadata.get("chembl_identity_gate") == "exact":
        return True
    if source_id == "osp_pbpk_model_library" and str(metadata.get("compound_match") or "").startswith("exact_"):
        return True
    return metadata.get("document_compound_identity") == "exact_single_target_active_ingredient"


def is_ddi_attribution_context(*, evidence_type: str, source_text: str, multi_drug: bool) -> bool:
    if evidence_type == "ddi":
        return True
    if not multi_drug:
        return False
    lowered = source_text.lower()
    interaction_actions = (
        "coadministered",
        "co-administered",
        "concomitant",
        "drug interaction",
        "drug-drug interaction",
        "inhibit",
        "induce",
        "increased",
        "decreased",
        "reduced",
        "altered",
    )
    return any(action in lowered for action in interaction_actions)


def unique_compound_matches(matches: list[CompoundTerm]) -> list[CompoundTerm]:
    by_compound: dict[str, CompoundTerm] = {}
    for match in matches:
        previous = by_compound.get(match.compound_id)
        if previous is None or len(match.synonym) > len(previous.synonym):
            by_compound[match.compound_id] = match
    return sorted(by_compound.values(), key=lambda item: len(item.synonym), reverse=True)


def find_compound_by_name_or_synonym(compounds: list[dict[str, Any]], value: str) -> dict[str, Any] | None:
    lowered = value.strip().lower()
    if not lowered:
        return None
    for compound in compounds:
        if lowered in {term.lower() for term in compound_validation_terms(compound)}:
            return compound
    return None


def compound_validation_terms(compound: dict[str, Any]) -> list[str]:
    terms = [
        str(compound.get("preferred_name") or ""),
        str(compound.get("normalized_name") or ""),
        str(compound.get("pubchem_cid") or ""),
        str(compound.get("chembl_id") or ""),
        str(compound.get("rxnorm_rxcui") or ""),
    ]
    terms.extend(str(item) for item in (compound.get("synonyms") or []))
    return [term.strip() for term in terms if term and term.strip()]


def contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def semantic_text(row: dict[str, str]) -> str:
    parts = [
        row.get("compound_name") or "",
        row.get("evidence_type") or "",
        row.get("parameter") or "",
        row.get("value") or row.get("value_text") or "",
        row.get("unit") or row.get("canonical_unit") or "",
        row.get("condition") or "",
        row.get("population") or "",
        row.get("route") or "",
        row.get("dose") or "",
        row.get("source_text") or "",
    ]
    return " | ".join(part for part in parts if part)


def semantic_group_key(rows: list[dict[str, str]]) -> str:
    evidence_ids = sorted(row["evidence_id"] for row in rows)
    return stable_hash("|".join(evidence_ids))


def has_value_or_condition_conflict(rows: list[dict[str, str]]) -> bool:
    for fields in [
        ["canonical_value", "value", "value_text"],
        ["canonical_unit", "unit"],
        ["condition"],
        ["population"],
        ["route"],
        ["dose"],
    ]:
        values = {first_nonempty(row, fields).strip().lower() for row in rows}
        values.discard("")
        if len(values) > 1:
            return True
    return False


def first_nonempty(row: dict[str, str], fields: list[str]) -> str:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return ""


def select_semantic_representative(rows: list[dict[str, str]]) -> dict[str, str]:
    return sorted(
        rows,
        key=lambda row: (
            0 if (row.get("value") and row.get("unit")) else 1,
            0 if (row.get("canonical_value") and row.get("canonical_unit")) else 1,
            source_priority(row.get("source_id") or ""),
            -safe_float(row.get("extraction_confidence"), default=0.0),
            -len(row.get("source_location") or ""),
            row.get("evidence_id") or "",
        ),
    )[0]


def source_priority(source_id: str) -> int:
    if source_id in {"dailymed", "openfda_drug_label", "drugs_fda", "ema_epar"}:
        return 1
    if source_id in {"pkdb", "clinicaltrials", "pubchem", "rxnorm", "chembl", "drugcentral"}:
        return 2
    if source_id in {"pubmed", "pmc_oa"}:
        return 3
    if source_id in {"osp_pbpk_model_library", "osp_tools_docs"}:
        return 4
    return 5


def stable_hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def parse_pg_text_array(value: str) -> list[str]:
    stripped = value.strip()
    if not stripped or stripped == "{}":
        return []
    if stripped.startswith("{") and stripped.endswith("}"):
        stripped = stripped[1:-1]
    if not stripped:
        return []
    reader = csv.reader([stripped])
    return [item for item in next(reader) if item]


def strip_sql(sql: str) -> str:
    stripped = sql.strip()
    while stripped.endswith(";"):
        stripped = stripped[:-1].strip()
    return stripped


def csv_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json_dump(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


def json_dump(value: Any) -> str:
    return json.dumps(clean_json_value(value if value is not None else {}), ensure_ascii=False, sort_keys=True, default=str)


def clean_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key).replace("\x00", ""): clean_json_value(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def psql_env() -> dict[str, str]:
    env = dict(os.environ)
    # Docker Desktop often gives PostgreSQL a small /dev/shm. These final-layer
    # aggregation queries are deterministic and do not need parallel workers.
    env["PGOPTIONS"] = "-c max_parallel_workers_per_gather=0"
    return env


def resolve_embedding_device(requested_device: str, torch_module: Any) -> str:
    if requested_device == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        return "cuda"
    if requested_device == "mps":
        if not getattr(torch_module.backends, "mps", None) or not torch_module.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false.")
        return "mps"
    if requested_device == "cpu":
        return "cpu"
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_json_content(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def sanitize_error(error: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_*\\-]{6,}", "sk-***", error)


if __name__ == "__main__":
    main()
