from __future__ import annotations

from collections import Counter
from typing import Any

from pbpk_kb.qa.citation_builder import attach_citation_ids, chunk_citation, evidence_citation, graph_citation, model_citation
from pbpk_kb.qa.retrieval import RetrievalResult


MODELING_APPROVED_STATUSES = {"approved_for_modeling"}


def has_extracted_value(row: dict[str, Any]) -> bool:
    return row.get("numeric_value") is not None or bool(row.get("value_text"))


def format_value(row: dict[str, Any]) -> str:
    value = row.get("numeric_value")
    value_text = row.get("value_text")
    unit = row.get("unit")
    if (value is not None or value_text) and not unit:
        return "value omitted because unit is missing"
    if value is not None:
        return f"{value} {unit}".strip()
    if value_text:
        return f"{value_text} {unit}".strip()
    return "value not extracted"


def evidence_sentence(row: dict[str, Any]) -> str:
    citation = row["citation_id"]
    parameter = row.get("parameter_name") or "parameter"
    value = format_value(row)
    source_id = row.get("source_id") or "unknown_source"
    status = row.get("curation_status") or "unknown_status"
    return f"- [{citation}] {parameter}: {value}; source={source_id}; status={status}."


def model_sentence(row: dict[str, Any]) -> str:
    citation = row["citation_id"]
    model_name = row.get("model_name") or "unnamed model"
    source_id = row.get("source_id") or "unknown_source"
    validation = row.get("validation_status") or "validation not indexed"
    software = row.get("software") or "software not indexed"
    return f"- [{citation}] {model_name}; software={software}; validation={validation}; source={source_id}."


def graph_sentence(row: dict[str, Any]) -> str:
    citation = row["citation_id"]
    target = row.get("target_name") or row.get("gene_symbol") or "target"
    relationship = row.get("relationship_type") or "relationship"
    source_id = row.get("source_id") or "unknown_source"
    status = row.get("curation_status") or "unknown_status"
    return f"- [{citation}] {relationship}: {target}; source={source_id}; status={status}."


def find_gaps(evidence_rows: list[dict[str, Any]], retrieval_notes: list[str]) -> list[str]:
    gaps = list(retrieval_notes)
    missing_units = [row for row in evidence_rows if has_extracted_value(row) and not row.get("unit")]
    incomplete_context = [row for row in evidence_rows if not row.get("context")]
    not_modeling_approved = [
        row
        for row in evidence_rows
        if row.get("curation_status") not in MODELING_APPROVED_STATUSES
    ]
    if missing_units:
        gaps.append(f"{len(missing_units)} evidence record(s) have extracted values without units.")
    if incomplete_context:
        gaps.append(f"{len(incomplete_context)} evidence record(s) have incomplete context fields.")
    if evidence_rows and len(not_modeling_approved) == len(evidence_rows):
        statuses = {row.get("curation_status") for row in evidence_rows}
        if statuses == {"auto_validated"}:
            gaps.append(
                "All matched structured evidence is auto-validated but not "
                "modeling-approved; model-specific suitability must still be checked."
            )
        else:
            gaps.append(
                "No matched structured evidence is modeling-approved; inspect its "
                "curation status before using it as a model input."
            )
    return gaps


def generate_answer(question: str, retrieval: RetrievalResult) -> dict[str, Any]:
    evidence_rows = attach_citation_ids(retrieval.structured_evidence, "E")
    model_rows = attach_citation_ids(retrieval.model_records, "M")
    graph_rows = attach_citation_ids(retrieval.graph_relationships, "G")
    chunk_rows = attach_citation_ids(retrieval.document_chunks, "D")

    citations = [
        *(evidence_citation(row, index) for index, row in enumerate(evidence_rows, start=1)),
        *(model_citation(row, index) for index, row in enumerate(model_rows, start=1)),
        *(graph_citation(row, index) for index, row in enumerate(graph_rows, start=1)),
        *(chunk_citation(row, index) for index, row in enumerate(chunk_rows, start=1)),
    ]
    status_counts = Counter(row.get("curation_status") or "unknown" for row in evidence_rows)
    gaps = find_gaps(evidence_rows, retrieval.retrieval_notes)

    if not evidence_rows and not model_rows and not graph_rows and not chunk_rows:
        summary = "The KB does not contain enough source-backed public evidence to answer this question."
    elif evidence_rows or model_rows or graph_rows:
        compound_name = retrieval.compound.get("preferred_name") if retrieval.compound else "the requested compound/question"
        summary = (
            f"The KB found {len(evidence_rows)} structured evidence record(s) and "
            f"{len(model_rows)} public model record(s) and {len(graph_rows)} graph relationship(s) for {compound_name}. "
            "Every item below includes provenance and curation status; non-approved records are not modeling-approved."
        )
    else:
        summary = (
            "The KB found document chunks but no matching structured evidence. "
            "Treat the answer as a source-text lead, not as extracted parameter evidence."
        )

    answer_lines = [summary]
    if evidence_rows:
        answer_lines.append("\nKey structured evidence:")
        answer_lines.extend(evidence_sentence(row) for row in evidence_rows[:8])
    if model_rows:
        answer_lines.append("\nPublic model registry:")
        answer_lines.extend(model_sentence(row) for row in model_rows[:5])
    if graph_rows:
        answer_lines.append("\nKnowledge graph relationships:")
        answer_lines.extend(graph_sentence(row) for row in graph_rows[:8])
    if chunk_rows:
        answer_lines.append("\nDocument chunk leads:")
        for row in chunk_rows[:3]:
            text = " ".join((row.get("chunk_text") or "").split())
            if len(text) > 220:
                text = f"{text[:217]}..."
            answer_lines.append(f"- [{row['citation_id']}] {text}")
    if gaps:
        answer_lines.append("\nUncertainties / gaps:")
        answer_lines.extend(f"- {gap}" for gap in gaps)

    return {
        "question": question,
        "answer": "\n".join(answer_lines),
        "compound": retrieval.compound,
        "key_evidence": evidence_rows,
        "model_records": model_rows,
        "graph_relationships": graph_rows,
        "document_chunks": chunk_rows,
        "citations": citations,
        "curation_status_counts": dict(status_counts),
        "uncertainties_and_gaps": gaps,
        "retrieval_summary": {
            "structured_evidence_count": len(evidence_rows),
            "model_record_count": len(model_rows),
            "graph_relationship_count": len(graph_rows),
            "document_chunk_count": len(chunk_rows),
            "retrieval_order": ["structured_evidence", "public_model_registry", "graph_relationships", "document_chunks"],
        },
    }
