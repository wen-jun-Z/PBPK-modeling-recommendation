from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import NAMESPACE_URL, uuid5

from pbpk_kb.processing.final_evidence import FINAL_LAYER_ROLES, source_priority
from pbpk_kb.processing.final_evidence_attribution_stage import (
    atomic_write_json,
    atomic_write_jsonl,
    iter_jsonl,
    load_json,
    utc_now,
)
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout


FINAL_TYPE_ROLES = {
    "physicochemical": {"target_compound"},
    "adme": {"target_compound"},
    "clinical_pk": {"target_compound"},
    "ddi": {"victim_drug", "perpetrator_drug"},
    "special_population": {"target_compound"},
    "public_pbpk_model": {"target_compound"},
    "tissue_expression": {"target_compound"},
}
NATURAL_LANGUAGE_METHODS = {"rule_based", "llm_assisted"}
PHYSICOCHEMICAL_PARAMETERS = {
    "molecularformula",
    "molecularweight",
    "fullmwt",
    "inchikey",
    "canonicalsmiles",
    "connectivitysmiles",
    "isomericsmiles",
    "xlogp",
    "alogp",
    "logp",
    "logd",
    "tpsa",
    "psa",
    "pka",
    "solubility",
    "ionization",
    "fractionunbound",
    "proteinbinding",
    "hba",
    "hbd",
    "hydrogenbonddonorcount",
    "hydrogenbondacceptorcount",
}
ADME_TERMS = (
    "absorption",
    "bioavailability",
    "distribution",
    "metabolism",
    "metabolite",
    "excretion",
    "excreted",
    "elimination",
    "permeability",
    "protein binding",
    "fraction unbound",
    "intrinsic clearance",
    "renal clearance",
    "hepatic clearance",
    "biliary",
    "urinary",
    "fecal",
    "substrate",
    "transporter",
    "cyp",
    "oatp",
    "oct",
    "mate",
    "p-gp",
    "p-glycoprotein",
    "bcrp",
)
ADME_EXCLUSION_TERMS = (
    "intestinal glucose absorption",
    "receptor activation",
    "receptor binding",
    "tumor growth",
    "antitumor",
    "behavioral",
    "locomotor",
    "anticonvulsant",
    "analgesic",
    "pain score",
    "blood pressure response",
    "pharmacodynamic endpoint",
    "plant biosynthesis",
    "leaf tissue",
    "seedling",
)
PK_PARAMETER_TERMS = (
    "auc",
    "cmax",
    "tmax",
    "half life",
    "halflife",
    "clearance",
    "volume of distribution",
    "bioavailability",
    "exposure",
    "concentration",
    "mean residence time",
    "mrt",
    "fraction excreted",
    "elimination rate",
    "peak trough",
)
PK_EXCLUSION_TERMS = (
    "efficacy endpoint",
    "clinical response",
    "adverse event rate",
    "pain score",
    "exercise performance",
    "blood pressure response",
)
DDI_DIRECTION_TERMS = (
    "increase",
    "decrease",
    "reduce",
    "raise",
    "lower",
    "enhance",
    "unchanged",
    "no effect",
    "did not alter",
    "inhibit",
    "induce",
    "fold",
)
DDI_EFFECT_TERMS = PK_PARAMETER_TERMS + (
    "cyp",
    "enzyme",
    "substrate",
    "inhibitor",
    "inducer",
    "transporter",
    "p-gp",
    "bcrp",
    "oatp",
    "oct",
    "mate",
)
DDI_RELATION_TERMS = (
    "coadminister",
    "co-administer",
    "concomitant",
    "combination",
    "with ",
    "effect of",
    "inhibition by",
    "induction by",
    "pretreatment",
    "interaction",
    "compared with",
    "versus",
)
SPECIAL_POPULATION_TERMS = {
    "pregnancy": ("pregnan", "maternal", "fetal", "foetal"),
    "lactation": ("lactat", "breastfeeding", "breast milk"),
    "pediatric": ("pediatric", "paediatric", "child", "adolescent", "neonate", "neonatal", "infant"),
    "geriatric": ("geriatric", "elderly", "older adult", "aged 65"),
    "renal_impairment": (
        "renal impairment", "kidney impairment", "renal failure", "chronic kidney", "decreased renal function",
        "acute kidney injury", " aki ", " arf ",
    ),
    "hepatic_impairment": (
        "hepatic impairment", "hepatic dysfunction", "liver impairment", "liver dysfunction", "chronic liver disease",
        "cirrhosis", "liver failure", "alcoholic patient", "alcoholic liver disease", "alcoholism",
    ),
    "genotype": ("genotype", "polymorphism", "poor metabolizer", "extensive metabolizer", "cyp2c19 pm"),
    "sex_difference": ("sex difference", "gender difference", "male and female", "men and women"),
    "obesity": ("obese", "obesity", "body mass index"),
    "critical_illness": ("critically ill", "critical illness", "intensive care"),
    "heart_failure": ("heart failure", "cardiac failure"),
}
ANIMAL_SPECIES = {
    "rat": (" rat ", "rats", "rat model"),
    "mouse": (" mouse ", "mice", "murine"),
    "dog": (" dog ", "dogs", "canine"),
    "monkey": ("monkey", "monkeys", "macaque", "macaques", "macaca", "cynomolgus", "primate", "primates"),
    "cat": ("cat", "cats", "feline"),
    "rabbit": ("rabbit", "rabbits"),
    "pig": (" pig ", "pigs", "porcine"),
}
PBPK_TERMS = ("pbpk", "physiologically based pharmacokinetic", "physiologically-based pharmacokinetic")
MODEL_EVIDENCE_TERMS = (
    "develop",
    "build",
    "built",
    "construct",
    "appl",
    "simulat",
    "predict",
    "reproduc",
    "describ",
    "implement",
    "analy",
    "estimat",
    "incorporat",
    "sensitivity",
    "validat",
    "verif",
    "evaluat",
    "qualif",
    "performance",
    "model file",
    "pk-sim",
    "simcyp",
    "gastroplus",
)
PUBLIC_MODEL_MENTION_PATTERNS = (
    re.compile(r"\bto date\b.{0,160}\b(?:few|other)\b.{0,160}\bpbpk models?\b", re.IGNORECASE),
    re.compile(r"\bpbpk models?\b.{0,120}\b(?:have|has) been (?:previously )?(?:reported|published)\b", re.IGNORECASE),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final-evidence domain gates and deterministic deduplication.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reset-downstream",
        action="store_true",
        help="Rebuild domain/deterministic outputs after semantic processing, preserving semantic results only when the input hash is unchanged.",
    )
    args = parser.parse_args()
    result = FinalEvidenceQualityStage(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        reset_downstream=args.reset_downstream,
    ).run()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


class FinalEvidenceQualityStage:
    def __init__(self, *, run_dir: Path, output_dir: Path, reset_downstream: bool = False) -> None:
        self.layout = raw_processing_layout(run_dir)
        self.output_dir = output_dir
        self.manifest_dir = output_dir / "00_run_manifest"
        self.attribution_dir = output_dir / "01_compound_attribution"
        self.domain_dir = output_dir / "02_domain_gates"
        self.dedup_dir = output_dir / "03_deduplication"
        self.reset_downstream = reset_downstream

    def run(self) -> dict[str, Any]:
        state_path = self.manifest_dir / "pipeline_state.json"
        state = load_json(state_path)
        previous_semantic_stage = dict((state.get("stages") or {}).get("semantic_deduplication") or {})
        previous_semantic_input_hash = file_sha256(self.dedup_dir / "semantic_dedup_input.jsonl")
        validate_stage_order(state, reset_downstream=self.reset_downstream, output_dir=self.output_dir)
        candidates = list(iter_jsonl(self.layout.path("evidence_candidates.jsonl")))
        attributions = list(iter_jsonl(self.attribution_dir / "compound_attribution.jsonl"))
        enriched = join_candidates_and_attributions(candidates, attributions)

        gate_rows: list[dict[str, Any]] = []
        passed: list[dict[str, Any]] = []
        review: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for row in enriched:
            result = evaluate_domain_gate(row)
            gate_rows.append(result)
            derived = {**row, **domain_result_fields(result)}
            if result["domain_gate_status"] == "passed":
                passed.append(derived)
            elif result["domain_gate_status"] == "needs_manual_review":
                review.append(derived)
            else:
                excluded.append(derived)

        domain_summary = summarize_domain_gates(gate_rows, candidate_count=len(enriched))
        self.domain_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(self.domain_dir / "domain_gate_results.jsonl", gate_rows)
        atomic_write_jsonl(self.domain_dir / "excluded_by_domain_gate.jsonl", excluded)
        atomic_write_jsonl(self.domain_dir / "domain_review_queue.jsonl", review)
        atomic_write_json(self.domain_dir / "domain_gate_summary.json", domain_summary)

        deterministic_input = passed + review
        groups, members, representatives = deterministic_deduplicate(deterministic_input)
        semantic_input = [row for row in representatives if semantic_input_eligible(row)]
        dedup_summary = summarize_deterministic_dedup(
            input_rows=deterministic_input,
            groups=groups,
            members=members,
            representatives=representatives,
            semantic_input=semantic_input,
        )
        self.dedup_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(self.dedup_dir / "deterministic_duplicate_groups.jsonl", groups)
        atomic_write_jsonl(self.dedup_dir / "deterministic_duplicate_members.jsonl", members)
        atomic_write_jsonl(self.dedup_dir / "deterministic_representatives.jsonl", representatives)
        atomic_write_jsonl(self.dedup_dir / "semantic_dedup_input.jsonl", semantic_input)
        atomic_write_json(self.dedup_dir / "deterministic_dedup_summary.json", dedup_summary)

        semantic_input_hash = file_sha256(self.dedup_dir / "semantic_dedup_input.jsonl")
        semantic_reused = (
            self.reset_downstream
            and previous_semantic_stage.get("status") == "completed"
            and previous_semantic_stage.get("accepted") is True
            and previous_semantic_input_hash is not None
            and previous_semantic_input_hash == semantic_input_hash
            and semantic_outputs_complete(self.dedup_dir)
        )

        state["current_stage"] = "conflict_analysis_ready" if semantic_reused else "semantic_deduplication_ready"
        state["updated_at"] = utc_now()
        state["stages"]["domain_quality_gates"] = {
            "status": "completed",
            "accepted": True,
            "passed_count": domain_summary["passed_count"],
            "needs_manual_review_count": domain_summary["needs_manual_review_count"],
            "excluded_count": domain_summary["excluded_count"],
            "summary_path": "02_domain_gates/domain_gate_summary.json",
        }
        state["stages"]["deterministic_deduplication"] = {
            "status": "completed",
            "accepted": True,
            "representative_count": dedup_summary["representative_count"],
            "semantic_input_count": dedup_summary["semantic_input_count"],
            "summary_path": "03_deduplication/deterministic_dedup_summary.json",
        }
        state["stages"]["semantic_deduplication"] = previous_semantic_stage if semantic_reused else {"status": "not_started"}
        state["stages"]["conflict_analysis"] = {"status": "not_started"}
        state["stages"]["evidence_layers"] = {"status": "not_started"}
        state["stages"]["human_review"] = {
            "status": "optional",
            "required_for_pipeline": False,
            "requires_human_action": False,
        }
        state["stages"]["rag_index"] = {"status": "not_started"}
        state["stages"]["database_ui_export"] = {"status": "not_started"}
        atomic_write_json(state_path, state)

        quality_path = self.manifest_dir / "evidence_quality_report.json"
        quality = load_json(quality_path)
        quality.update(
            {
                "generated_at": utc_now(),
                "domain_quality_gates": domain_summary,
                "deterministic_deduplication": dedup_summary,
                "semantic_deduplication_reused": semantic_reused,
                "semantic_input_sha256": semantic_input_hash,
                "later_steps_executed": False,
                "next_required_step": "conflict_analysis" if semantic_reused else "biomedical_embedding_semantic_deduplication",
            }
        )
        atomic_write_json(quality_path, quality)
        return {
            "run_id": state.get("run_id"),
            "domain_quality_gates": domain_summary,
            "deterministic_deduplication": dedup_summary,
            "semantic_deduplication_reused": semantic_reused,
            "semantic_input_sha256": semantic_input_hash,
            "next_required_step": "conflict_analysis" if semantic_reused else "biomedical_embedding_semantic_deduplication",
        }


def validate_stage_order(state: dict[str, Any], *, reset_downstream: bool = False, output_dir: Path | None = None) -> None:
    attribution = (state.get("stages") or {}).get("compound_attribution") or {}
    if attribution.get("status") != "completed" or attribution.get("accepted") is not True:
        raise RuntimeError("Compound attribution must be completed and accepted before domain gates.")
    if int(attribution.get("llm_pending_count") or 0) != 0:
        raise RuntimeError("Compound attribution still has LLM-pending evidence.")
    semantic = (state.get("stages") or {}).get("semantic_deduplication") or {}
    if semantic.get("status") not in {None, "not_started"} and not reset_downstream:
        raise RuntimeError("Semantic deduplication already started; refusing to run earlier stages out of order.")
    if reset_downstream and output_dir is not None:
        reviewed_paths = [
            output_dir / "05_evidence_layers" / "accepted_evidence.jsonl",
            output_dir / "05_evidence_layers" / "modeling_evidence.jsonl",
        ]
        if any(path.exists() and path.stat().st_size for path in reviewed_paths):
            raise RuntimeError("Refusing to reset deterministic outputs after human-reviewed evidence exists.")


def join_candidates_and_attributions(
    candidates: list[dict[str, Any]],
    attributions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_ids = [str(row.get("evidence_id") or "") for row in candidates]
    attribution_by_id = {str(row.get("evidence_id") or ""): row for row in attributions}
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("Candidate bottom table contains duplicate evidence_id values.")
    missing = [value for value in candidate_ids if value not in attribution_by_id]
    extra = sorted(set(attribution_by_id) - set(candidate_ids))
    if missing or extra:
        raise RuntimeError(f"Candidate/attribution evidence_id mismatch: missing={len(missing)} extra={len(extra)}")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        evidence_id = str(candidate["evidence_id"])
        attribution = attribution_by_id[evidence_id]
        if attribution.get("attribution_status") != "resolved":
            raise RuntimeError(f"Unresolved compound attribution: {evidence_id}")
        row = dict(candidate)
        row.update(
            {
                "retrieval_compound": candidate.get("compound"),
                "attributed_compound_id": attribution.get("compound_id"),
                "attributed_compound_name": attribution.get("compound_name"),
                "compound_id": attribution.get("compound_id"),
                "compound_name": attribution.get("compound_name"),
                "compound_role": attribution.get("compound_role"),
                "attribution_method": attribution.get("attribution_method"),
                "attribution_stage": attribution.get("attribution_stage"),
                "attribution_confidence": attribution.get("attribution_confidence"),
                "attribution_reason": attribution.get("attribution_reason"),
                "matched_text": attribution.get("matched_text"),
                "provenance_status": "passed",
            }
        )
        rows.append(row)
    return rows


def evaluate_domain_gate(row: dict[str, Any]) -> dict[str, Any]:
    from pbpk_kb.processing.source_context import analyte_mismatch_reason
    evidence_type = str(row.get("evidence_type") or "")
    role = str(row.get("compound_role") or "")
    result = {
        "evidence_id": row["evidence_id"],
        "evidence_type": evidence_type,
        "attributed_compound_id": row.get("attributed_compound_id"),
        "attributed_compound_name": row.get("attributed_compound_name"),
        "compound_role": role,
        "domain_gate_status": "passed",
        "domain_classification": None,
        "reasons": [],
        "evaluated_at": utc_now(),
    }
    allowed_roles = FINAL_TYPE_ROLES.get(evidence_type)
    if allowed_roles is None:
        return reject(result, "unsupported_evidence_type")
    if role not in allowed_roles:
        return reject(result, "compound_role_not_supported_for_evidence_type")
    mismatch = analyte_mismatch_reason(row)
    if mismatch:
        return reject(result, mismatch)
    species = explicit_animal_species(row)
    if species:
        result["species"] = species
        if evidence_type == "clinical_pk":
            # Preserve preclinical/comparative records for review rather than
            # letting the clinical table silently treat them as human PK.
            result["domain_gate_status"] = "needs_manual_review"
            result["domain_classification"] = "animal_pk_requires_species_review"
            result["reasons"] = ["explicit_animal_species_in_clinical_pk"]
            return result
    evaluator = DOMAIN_EVALUATORS[evidence_type]
    return evaluator(row, result)


def explicit_animal_species(row: dict[str, Any]) -> list[str]:
    raw_text = " ".join(str(row.get(key) or "") for key in ("population", "condition", "source_text"))
    text = normalize_words(raw_text)
    species = {name for name, terms in ANIMAL_SPECIES.items()
               if any(re.search(r"\b" + re.escape(term.strip()) + r"\b", text) for term in terms)}
    if "cat" in species and not any(not m.group(0).isupper() for m in re.finditer(r"\b(?:cats?|feline)\b", raw_text, re.I)):
        species.remove("cat")
    context = (row.get("metadata_json") or {}).get("source_context") or {}
    if context.get("match_status") == "matched" and context.get("animal_study"):
        species.update(context.get("species") or [])
    return sorted(species)


def physicochemical_gate(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    parameter = normalize_key(row.get("parameter"))
    if parameter not in PHYSICOCHEMICAL_PARAMETERS:
        return reject(result, "not_a_supported_pure_compound_property")
    result["domain_classification"] = "pure_compound_property"
    return result


def adme_gate(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    parameter = normalize_words(row.get("parameter"))
    text = normalize_words(join_text_fields(row))
    if not contains_any(parameter, ADME_TERMS) and not contains_any(text, ADME_TERMS):
        return reject(result, "missing_target_compound_adme_process")
    # A drug's effect on water or electrolyte transport is a pharmacodynamic
    # endpoint, not absorption/excretion of that drug. Match the parameter so
    # ordinary dosing-water context does not exclude valid drug disposition.
    non_drug_transport = re.search(
        r"\b(?:water|glucose|sodium|potassium|chloride|electrolytes?)\s+(?:reabsorption|absorption|excretion)\b"
        r"|\b(?:reabsorption|absorption|excretion)\s+of\s+(?:water|glucose|sodium|potassium|chloride|electrolytes?)\b",
        parameter,
    )
    if non_drug_transport or contains_any(text, ADME_EXCLUSION_TERMS) or contains_any(parameter, ADME_EXCLUSION_TERMS):
        return reject(result, "pharmacodynamic_or_non_drug_adme_context")
    result["domain_classification"] = "target_compound_adme"
    return result


def clinical_pk_gate(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    parameter = normalize_words(row.get("parameter"))
    text = normalize_words(join_text_fields(row))
    if not contains_any(parameter, PK_PARAMETER_TERMS):
        return reject(result, "missing_supported_pk_parameter")
    if contains_any(text, PK_EXCLUSION_TERMS) and not has_numeric_pk_context(text):
        return reject(result, "efficacy_safety_or_non_pk_endpoint")
    if not contains_any(text, PK_PARAMETER_TERMS) and not has_numeric_pk_context(text):
        return reject(result, "source_text_does_not_support_pk_parameter")
    result["domain_classification"] = "clinical_pk_parameter_or_exposure"
    return result


def ddi_gate(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = normalize_words(join_text_fields(row))
    missing: list[str] = []
    if not contains_any(text, DDI_RELATION_TERMS):
        missing.append("interacting_drug_relation")
    if not contains_any(text, DDI_EFFECT_TERMS):
        missing.append("effect_or_mechanism")
    if not contains_any(text, DDI_DIRECTION_TERMS):
        missing.append("direction")
    if "probe" in text and "direction" in missing:
        missing.append("probe_identity_without_effect")
    if missing:
        result["domain_gate_status"] = "needs_manual_review"
        result["domain_classification"] = "ddi_direction_or_relation_unresolved"
        result["reasons"] = sorted(set(missing))
        return result
    result["domain_classification"] = "directional_ddi"
    return result


def special_population_gate(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = f" {normalize_words(join_text_fields(row))} "
    populations = [name for name, terms in SPECIAL_POPULATION_TERMS.items() if contains_any(text, terms)]
    species = explicit_animal_species(row)
    if not populations and not species:
        return reject(result, "missing_explicit_population_condition_or_species")
    if species:
        result["domain_classification"] = "animal_special_population"
        result["species"] = species
    else:
        result["domain_classification"] = "human_special_population"
        result["population_categories"] = populations
    return result


def public_model_gate(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
    source_id = str(row.get("source_id") or "")
    parameter = normalize_words(row.get("parameter"))
    text = normalize_words(join_text_fields(row))
    if (
        source_id == "osp_pbpk_model_library"
        and parameter == "model file"
        and metadata.get("model_file_path")
        and metadata.get("model_storage_path")
        and metadata.get("model_checksum")
    ):
        result["domain_classification"] = "downloadable_public_model"
        return result
    source_text = str(row.get("source_text") or "")
    if source_id in {"pubmed", "pmc_oa"} and any(pattern.search(source_text) for pattern in PUBLIC_MODEL_MENTION_PATTERNS):
        result["domain_classification"] = "model_mention_only"
        return reject(result, "model_mention_without_pbpk_development_or_evaluation")
    if contains_any(text, PBPK_TERMS):
        result["domain_classification"] = "published_pbpk_model"
        return result
    if contains_any(text, ("simcyp", "gastroplus")) and contains_any(text, ("model", "simulator")) and contains_any(text, MODEL_EVIDENCE_TERMS):
        result["domain_classification"] = "published_pbpk_model"
        return result
    result["domain_classification"] = "model_mention_only"
    return reject(result, "model_mention_without_pbpk_development_or_evaluation")


def tissue_expression_gate(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = normalize_words(join_text_fields(row))
    if "expression" not in text or not contains_any(text, ("tissue", "liver", "kidney", "intestin", "brain", "lung")):
        return reject(result, "missing_explicit_tissue_expression_measurement")
    result["domain_classification"] = "tissue_expression_measurement"
    return result


DOMAIN_EVALUATORS: dict[str, Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]] = {
    "physicochemical": physicochemical_gate,
    "adme": adme_gate,
    "clinical_pk": clinical_pk_gate,
    "ddi": ddi_gate,
    "special_population": special_population_gate,
    "public_pbpk_model": public_model_gate,
    "tissue_expression": tissue_expression_gate,
}


def reject(result: dict[str, Any], reason: str) -> dict[str, Any]:
    result["domain_gate_status"] = "excluded"
    result["reasons"] = [reason]
    return result


def domain_result_fields(result: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "domain_gate_status": result["domain_gate_status"],
        "domain_classification": result.get("domain_classification"),
        "domain_gate_reasons": result.get("reasons") or [],
    }
    if result.get("population_categories"):
        fields["population_categories"] = result["population_categories"]
    if result.get("species"):
        fields["species"] = result["species"]
    return fields


def deterministic_deduplicate(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    active = {str(row["evidence_id"]): row for row in rows}
    all_groups: list[dict[str, Any]] = []
    all_members: list[dict[str, Any]] = []
    stages: list[tuple[str, Callable[[dict[str, Any]], str | None]]] = [
        ("evidence_id", evidence_id_key),
        ("source", source_duplicate_key),
        ("content", content_duplicate_key),
    ]
    for group_type, key_function in stages:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in active.values():
            key = key_function(row)
            if key:
                buckets[key].append(row)
        for key, members in sorted(buckets.items()):
            if len(members) < 2:
                continue
            representative = select_deterministic_representative(members)
            group_id = str(uuid5(NAMESPACE_URL, f"pbpk:deterministic:{group_type}:{key}"))
            ordered = sorted(
                members,
                key=lambda row: representative_sort_key(row),
            )
            all_groups.append(
                {
                    "duplicate_group_id": group_id,
                    "group_type": group_type,
                    "group_key": key,
                    "evidence_type": representative.get("evidence_type"),
                    "compound_id": representative.get("attributed_compound_id"),
                    "compound_name": representative.get("attributed_compound_name"),
                    "representative_evidence_id": representative["evidence_id"],
                    "duplicate_count": len(ordered),
                    "selection_reason": "strategy_priority_order",
                }
            )
            for rank, member in enumerate(ordered, start=1):
                is_representative = member["evidence_id"] == representative["evidence_id"]
                all_members.append(
                    {
                        "duplicate_group_id": group_id,
                        "group_type": group_type,
                        "evidence_id": member["evidence_id"],
                        "is_representative": is_representative,
                        "duplicate_rank": rank,
                        "duplicate_reason": f"deterministic_{group_type}_match",
                    }
                )
                if not is_representative:
                    active.pop(str(member["evidence_id"]), None)
    representatives = sorted(active.values(), key=lambda row: str(row["evidence_id"]))
    return all_groups, all_members, representatives


def evidence_id_key(row: dict[str, Any]) -> str | None:
    value = str(row.get("evidence_id") or "")
    return stable_key([value]) if value else None


def source_duplicate_key(row: dict[str, Any]) -> str | None:
    values = [row.get("raw_asset_id"), row.get("source_location"), row.get("source_text_hash")]
    # One sentence/table row can support multiple compounds and parameters.
    fact_key = content_duplicate_key(row)
    return stable_key([*values, fact_key]) if all(values) and fact_key else None


def content_duplicate_key(row: dict[str, Any]) -> str | None:
    compound = row.get("attributed_compound_id") or row.get("attributed_compound_name")
    source_hash = row.get("source_text_hash")
    if not compound or not source_hash:
        return None
    value = row.get("canonical_value") if row.get("canonical_value") is not None else row.get("value")
    unit = row.get("canonical_unit") or row.get("unit")
    condition = "" if is_structured_table_fact(row) else normalize_words(row.get("condition"))
    return stable_key(
        [
            compound,
            normalize_key(row.get("evidence_type")),
            normalize_key(row.get("parameter")),
            normalize_scalar(value),
            normalize_scalar(row.get("canonical_value_low") if row.get("canonical_value_low") is not None else row.get("value_low")),
            normalize_scalar(row.get("canonical_value_high") if row.get("canonical_value_high") is not None else row.get("value_high")),
            normalize_words(row.get("value_text")),
            normalize_words(row.get("value_qualifier")),
            normalize_key(unit),
            condition,
            normalize_words(row.get("population")),
            normalize_words(row.get("route")),
            normalize_words(row.get("dose")),
            json.dumps(row.get("species") or [], sort_keys=True),
            source_hash,
        ]
    )


def is_structured_table_fact(row: dict[str, Any]) -> bool:
    return bool(
        row.get("extraction_method") == "source_structured_table"
        or row.get("parsed_table_id")
        or row.get("recovered_table_id")
        or row.get("table_row_id")
    )


def select_deterministic_representative(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(rows, key=representative_sort_key)[0]


def representative_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if row.get("provenance_status") == "passed" else 1,
        -safe_float(row.get("attribution_confidence")),
        0 if row.get("value") is not None and row.get("unit") else 1,
        0 if row.get("canonical_value") is not None and row.get("canonical_unit") else 1,
        source_priority(str(row.get("source_id") or "")),
        -source_location_precision(str(row.get("source_location") or "")),
        -safe_float(row.get("extraction_confidence")),
        str(row.get("evidence_id") or ""),
    )


def semantic_input_eligible(row: dict[str, Any]) -> bool:
    if row.get("domain_gate_status") != "passed":
        return False
    if row.get("compound_role") not in FINAL_LAYER_ROLES:
        return False
    if row.get("extraction_method") not in NATURAL_LANGUAGE_METHODS:
        return False
    text = str(row.get("source_text") or "").strip()
    if len(text) < 40 or len(re.findall(r"[A-Za-z]{3,}", text)) < 5:
        return False
    if row.get("parsed_table_id") or row.get("recovered_table_id") or row.get("table_row_id"):
        return False
    return True


def summarize_domain_gates(rows: list[dict[str, Any]], *, candidate_count: int) -> dict[str, Any]:
    status = Counter(str(row["domain_gate_status"]) for row in rows)
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: Counter[str] = Counter()
    classifications: Counter[str] = Counter()
    for row in rows:
        by_type[str(row["evidence_type"])][str(row["domain_gate_status"])] += 1
        reasons.update(str(value) for value in row.get("reasons") or [])
        if row.get("domain_classification"):
            classifications[str(row["domain_classification"])] += 1
    return {
        "generated_at": utc_now(),
        "candidate_count": candidate_count,
        "evaluated_count": len(rows),
        "passed_count": status["passed"],
        "needs_manual_review_count": status["needs_manual_review"],
        "excluded_count": status["excluded"],
        "status_counts": dict(sorted(status.items())),
        "evidence_type_status_counts": {key: dict(sorted(value.items())) for key, value in sorted(by_type.items())},
        "reason_counts": dict(sorted(reasons.items())),
        "classification_counts": dict(sorted(classifications.items())),
        "accepted": len(rows) == candidate_count and sum(status.values()) == candidate_count,
    }


def summarize_deterministic_dedup(
    *,
    input_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    members: list[dict[str, Any]],
    representatives: list[dict[str, Any]],
    semantic_input: list[dict[str, Any]],
) -> dict[str, Any]:
    group_types = Counter(str(row["group_type"]) for row in groups)
    removed_ids = {
        str(row["evidence_id"])
        for row in members
        if row.get("is_representative") is False
    }
    return {
        "generated_at": utc_now(),
        "input_count": len(input_rows),
        "duplicate_group_count": len(groups),
        "duplicate_member_count": len(members),
        "non_representative_count": len(removed_ids),
        "representative_count": len(representatives),
        "semantic_input_count": len(semantic_input),
        "group_type_counts": dict(sorted(group_types.items())),
        "candidate_rows_deleted": 0,
        "accepted": len(representatives) + len(removed_ids) == len(input_rows),
    }


def join_text_fields(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("parameter", "value_text", "condition", "population", "source_location", "source_text")
    )


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def normalize_words(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".12g")
    return normalize_words(value)


def contains_any(value: str, terms: Iterable[str]) -> bool:
    return any(term in value for term in terms)


def has_numeric_pk_context(text: str) -> bool:
    return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:ng|pg|mg|ug|µg|mcg|mol|mmol|umol|µmol|h|hr|min|%)\b", text))


def stable_key(values: Iterable[Any]) -> str:
    payload = json.dumps(["" if value is None else str(value) for value in values], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def source_location_precision(value: str) -> int:
    selectors = sum(value.count(token) for token in ("#page=", "#chunk=", "#xml_path=", "#json_path=", "#table="))
    return selectors * 10000 + len(value)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_outputs_complete(dedup_dir: Path) -> bool:
    required = [
        dedup_dir / "semantic_duplicate_groups.jsonl",
        dedup_dir / "semantic_duplicate_members.jsonl",
        dedup_dir / "semantic_conflict_candidates.jsonl",
        dedup_dir / "semantic_dedup_state.jsonl",
        dedup_dir / "semantic_dedup_summary.json",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        summary = load_json(dedup_dir / "semantic_dedup_summary.json")
    except (OSError, RuntimeError, json.JSONDecodeError):
        return False
    return summary.get("accepted") is True


if __name__ == "__main__":
    main()
