from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.local_llm import (
    DEFAULT_LOCAL_LLM,
    LocalStructuredGenerator,
    StructuredGeneration,
)


PROMPT_VERSION = "pbpk-ddi-relation-extractor-1.1.0"
VALIDATOR_VERSION = "pbpk-ddi-relation-validator-1.1.0"
MIN_CONFIDENCE = 0.80
EFFECT_VALUES = {"increased", "decreased", "no_effect", "changed", "unknown"}
RELATION_TRIGGER = re.compile(
    r"\b(?:co-?administ(?:er|ered|ration|ring)|interact(?:ion|s|ed)?|"
    r"increas(?:e|ed|es|ing)|decreas(?:e|ed|es|ing)|reduc(?:e|ed|es|ing)|"
    r"prolong(?:s|ed|ing)?|impair(?:s|ed|ing)?|inhibit(?:s|ed|ion|ing)?|"
    r"induc(?:e|ed|es|tion|ing)|affect(?:s|ed|ing)?|alter(?:s|ed|ing)?|"
    r"exposure|clearance|bioavailability|auc|cmax)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You extract drug-drug interaction events from one biomedical evidence passage.
Return exactly one JSON object with an `events` array. Use only node IDs supplied in the payload.
Create one event per explicitly supported victim/perpetrator pair. Do not infer a relation from
mere co-mention. For complex passages, use only pairs listed in supervised_candidate_pairs.
The target compound role in the payload is binding. Each event must contain:
victim_node_id, perpetrator_node_id, supporting_quote, interaction_effect, parameter_node_ids,
mediator_node_ids, confidence, and reason. supporting_quote must be an exact contiguous substring
of source_text containing both drug mentions and the interaction assertion. interaction_effect
must be increased, decreased, no_effect, changed, or unknown. Use [] when no event is asserted.
Never invent an entity, identifier, measurement, mechanism, parameter, or direction."""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve complex DDI roles with deterministic constraints and a local structured LLM."
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--mentions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--supervised-signals", type=Path)
    parser.add_argument("--supervised-summary", type=Path)
    parser.add_argument("--model", default=DEFAULT_LOCAL_LLM)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    summary = run_validation(
        evidence_path=args.evidence,
        mentions_path=args.mentions,
        output_dir=args.output_dir,
        supervised_signals_path=args.supervised_signals,
        supervised_summary_path=args.supervised_summary,
        model_name=args.model,
        resume=args.resume,
        prepare_only=args.prepare_only,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def run_validation(
    *,
    evidence_path: Path,
    mentions_path: Path,
    output_dir: Path,
    supervised_signals_path: Path | None = None,
    supervised_summary_path: Path | None = None,
    model_name: str = DEFAULT_LOCAL_LLM,
    resume: bool = False,
    prepare_only: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        str(row["evidence_id"]): row
        for row in _iter_jsonl(evidence_path)
        if row.get("evidence_type") == "ddi"
    }
    mentions_by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in _iter_jsonl(mentions_path):
        evidence_id = str(mention.get("evidence_id") or "")
        if evidence_id in evidence:
            mentions_by_evidence[evidence_id].append(mention)
    supervised_gate_required = supervised_signals_path is not None
    supervised_summary = _validate_supervised_inputs(
        supervised_signals_path,
        supervised_summary_path,
    )
    signals = _load_signals(supervised_signals_path)
    tasks = [
        build_relation_task(
            row,
            mentions_by_evidence.get(evidence_id, []),
            signals.get(evidence_id, []),
            supervised_gate_required=supervised_gate_required,
        )
        for evidence_id, row in evidence.items()
    ]
    _atomic_jsonl(output_dir / "ddi_relation_tasks.jsonl", tasks)

    task_counts = Counter(task["routing"] for task in tasks)
    base_summary = {
        "accepted": False,
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha256(evidence_path),
        "mentions_path": str(mentions_path),
        "mentions_sha256": _sha256(mentions_path),
        "supervised_signals_path": str(supervised_signals_path) if supervised_signals_path else None,
        "supervised_signals_sha256": (
            _sha256(supervised_signals_path) if supervised_signals_path and supervised_signals_path.exists() else None
        ),
        "supervised_summary_path": str(supervised_summary_path) if supervised_summary_path else None,
        "supervised_summary_sha256": (
            _sha256(supervised_summary_path)
            if supervised_summary_path and supervised_summary_path.exists()
            else None
        ),
        "supervised_model_accepted": (
            bool(supervised_summary.get("accepted")) if supervised_summary else None
        ),
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "model_name": model_name,
        "ddi_evidence_count": len(tasks),
        "routing_counts": dict(sorted(task_counts.items())),
    }
    if prepare_only:
        _atomic_json(output_dir / "ddi_relation_summary.json", {**base_summary, "status": "prepared"})
        return {**base_summary, "status": "prepared"}

    state_path = output_dir / "ddi_relation_state.jsonl"
    completed = {
        str(row["evidence_id"]): row for row in _iter_jsonl(state_path)
    } if resume and state_path.exists() else {}
    if state_path.exists() and not resume:
        raise RuntimeError(f"State exists at {state_path}; use --resume or a new output directory.")

    llm_tasks = [task for task in tasks if task["routing"] == "llm_complex"]
    generator = LocalStructuredGenerator(model_name) if any(
        task["evidence_id"] not in completed for task in llm_tasks
    ) else None
    for index, task in enumerate(tasks, start=1):
        evidence_id = str(task["evidence_id"])
        if evidence_id in completed:
            continue
        if task["routing"] == "deterministic_unique_counterpart":
            decision = deterministic_decision(task)
        elif task["routing"] == "llm_complex":
            if generator is None:
                raise RuntimeError("Local LLM was not initialized for an unresolved complex task.")
            generation = generator.generate_json(
                system_prompt=SYSTEM_PROMPT,
                payload=_llm_payload(task),
                max_new_tokens=1024,
            )
            decision = validate_model_decision(task, generation)
        else:
            decision = unresolved_decision(task, str(task["routing"]))
        _append_jsonl(state_path, decision)
        completed[evidence_id] = decision
        if index % 10 == 0 or index == len(tasks):
            print(json.dumps({"processed": index, "total": len(tasks)}, sort_keys=True), flush=True)

    ordered = [completed[str(task["evidence_id"])] for task in tasks]
    _atomic_jsonl(output_dir / "validated_ddi_relation_decisions.jsonl", ordered)
    status_counts = Counter(row["status"] for row in ordered)
    method_counts = Counter(row["decision_method"] for row in ordered)
    accepted_events = sum(len(row.get("events") or []) for row in ordered if row["status"] == "accepted")
    accepted = (
        len(ordered) == len(tasks)
        and all(row.get("source_text_hash") == task.get("source_text_hash") for row, task in zip(ordered, tasks))
        and all(
            event.get("validation_status") == "auto_validated"
            for row in ordered
            for event in row.get("events") or []
        )
    )
    summary = {
        **base_summary,
        "accepted": accepted,
        "status": "completed",
        "status_counts": dict(sorted(status_counts.items())),
        "decision_method_counts": dict(sorted(method_counts.items())),
        "accepted_event_count": accepted_events,
    }
    _atomic_json(output_dir / "ddi_relation_summary.json", summary)
    return summary


def build_relation_task(
    row: dict[str, Any],
    mentions: list[dict[str, Any]],
    supervised_signals: list[dict[str, Any]] | None = None,
    *,
    supervised_gate_required: bool = False,
) -> dict[str, Any]:
    target_node_id = f"Compound:{row['compound_id']}"
    compounds: dict[str, dict[str, Any]] = {}
    parameters: dict[str, str] = {}
    mediators: dict[str, str] = {}
    for mention in mentions:
        node_id = str(mention.get("linked_node_id") or "")
        if node_id.startswith("Compound:"):
            entry = compounds.setdefault(node_id, {"node_id": node_id, "mentions": []})
            text = str(mention.get("text") or "")
            if text and text not in entry["mentions"]:
                entry["mentions"].append(text)
        elif node_id.startswith("PKParameter:"):
            parameters[node_id] = str(mention.get("text") or "")
        elif node_id.startswith(("Enzyme:", "Transporter:", "Protein:")):
            mediators[node_id] = str(mention.get("text") or "")
    others = sorted(node_id for node_id in compounds if node_id != target_node_id)
    positive_target_pairs = _positive_target_pairs(
        target_node_id,
        supervised_signals or [],
    )
    role = str(row.get("compound_role") or "")
    routing = "unresolved_no_counterpart"
    if role in {"victim_drug", "perpetrator_drug"} and len(others) == 1:
        routing = "deterministic_unique_counterpart"
    elif role in {"victim_drug", "perpetrator_drug"} and len(others) > 1:
        routing = (
            "llm_complex"
            if not supervised_gate_required or positive_target_pairs
            else "unresolved_supervised_re_no_positive_target_pair"
        )
    return {
        "evidence_id": str(row["evidence_id"]),
        "source_id": str(row["source_id"]),
        "source_location": str(row["source_location"]),
        "source_text": str(row["source_text"]),
        "source_text_hash": str(row["source_text_hash"]),
        "target_node_id": target_node_id,
        "target_name": str(row.get("compound_name") or row.get("attributed_compound_name") or ""),
        "target_role": role,
        "candidate_compounds": [compounds[node_id] for node_id in sorted(compounds)],
        "parameter_candidates": [
            {"node_id": node_id, "mention": parameters[node_id]} for node_id in sorted(parameters)
        ],
        "mediator_candidates": [
            {"node_id": node_id, "mention": mediators[node_id]} for node_id in sorted(mediators)
        ],
        "supervised_relation_signals": supervised_signals or [],
        "supervised_candidate_pairs": positive_target_pairs,
        "supervised_gate_required": supervised_gate_required,
        "routing": routing,
    }


def deterministic_decision(task: dict[str, Any]) -> dict[str, Any]:
    target_node_id = str(task["target_node_id"])
    other_ids = [
        item["node_id"] for item in task["candidate_compounds"] if item["node_id"] != target_node_id
    ]
    if len(other_ids) != 1:
        return unresolved_decision(task, "deterministic_counterpart_not_unique")
    if task["target_role"] == "victim_drug":
        victim_node_id, perpetrator_node_id = target_node_id, other_ids[0]
    elif task["target_role"] == "perpetrator_drug":
        victim_node_id, perpetrator_node_id = other_ids[0], target_node_id
    else:
        return unresolved_decision(task, "target_role_not_binding")
    event = {
        "victim_node_id": victim_node_id,
        "perpetrator_node_id": perpetrator_node_id,
        "supporting_quote": task["source_text"],
        "interaction_effect": infer_effect(str(task["source_text"])),
        "parameter_node_ids": [item["node_id"] for item in task["parameter_candidates"]],
        "mediator_node_ids": [item["node_id"] for item in task["mediator_candidates"]],
        "confidence": 0.99,
        "reason": "target role plus one uniquely linked non-target compound",
    }
    errors = validate_event(task, event)
    if errors:
        return unresolved_decision(task, *errors)
    return _decision(
        task,
        status="accepted",
        decision_method="deterministic_unique_counterpart",
        events=[{**event, "validation_status": "auto_validated"}],
    )


def validate_model_decision(
    task: dict[str, Any],
    generation: StructuredGeneration,
) -> dict[str, Any]:
    if generation.parsed is None:
        return _decision(
            task,
            status="unresolved",
            decision_method="qwen_structured_relation_extraction",
            events=[],
            raw_model_output=generation.raw_text,
            validation_errors=[generation.error or "invalid_json"],
        )
    model_events = generation.parsed.get("events")
    if not isinstance(model_events, list):
        return _decision(
            task,
            status="unresolved",
            decision_method="qwen_structured_relation_extraction",
            events=[],
            raw_model_output=generation.raw_text,
            validation_errors=["events_must_be_a_list"],
        )
    accepted_events = []
    rejected_events = []
    seen: set[tuple[str, str, str]] = set()
    for event in model_events:
        if not isinstance(event, dict):
            rejected_events.append({"event": event, "validation_errors": ["event_must_be_an_object"]})
            continue
        errors = validate_event(task, event)
        key = (
            str(event.get("victim_node_id") or ""),
            str(event.get("perpetrator_node_id") or ""),
            str(event.get("supporting_quote") or ""),
        )
        if key in seen:
            errors.append("duplicate_event")
        seen.add(key)
        if errors:
            rejected_events.append({"event": event, "validation_errors": sorted(set(errors))})
        else:
            accepted_events.append({**event, "validation_status": "auto_validated"})
    return _decision(
        task,
        status="accepted" if accepted_events else "unresolved",
        decision_method="qwen_structured_relation_extraction",
        events=accepted_events,
        rejected_events=rejected_events,
        raw_model_output=generation.raw_text,
        validation_errors=[] if accepted_events else ["no_model_event_passed_independent_validation"],
    )


def validate_event(task: dict[str, Any], event: dict[str, Any]) -> list[str]:
    errors = []
    candidates = {item["node_id"]: item for item in task["candidate_compounds"]}
    parameters = {item["node_id"] for item in task["parameter_candidates"]}
    mediators = {item["node_id"] for item in task["mediator_candidates"]}
    victim = str(event.get("victim_node_id") or "")
    perpetrator = str(event.get("perpetrator_node_id") or "")
    if victim not in candidates:
        errors.append("victim_not_in_candidate_set")
    if perpetrator not in candidates:
        errors.append("perpetrator_not_in_candidate_set")
    if victim and victim == perpetrator:
        errors.append("victim_and_perpetrator_are_identical")
    if task["target_role"] == "victim_drug" and victim != task["target_node_id"]:
        errors.append("victim_conflicts_with_binding_target_role")
    if task["target_role"] == "perpetrator_drug" and perpetrator != task["target_node_id"]:
        errors.append("perpetrator_conflicts_with_binding_target_role")
    if task.get("supervised_gate_required") and task.get("routing") == "llm_complex":
        accepted_pairs = {
            frozenset((str(pair["target_node_id"]), str(pair["counterpart_node_id"])))
            for pair in task.get("supervised_candidate_pairs") or []
        }
        if frozenset((victim, perpetrator)) not in accepted_pairs:
            errors.append("pair_not_accepted_by_supervised_relation_gate")

    quote = str(event.get("supporting_quote") or "")
    source_text = str(task["source_text"])
    if not quote or quote not in source_text:
        errors.append("supporting_quote_not_verbatim")
    else:
        for node_id, label in ((victim, "victim"), (perpetrator, "perpetrator")):
            mention_values = candidates.get(node_id, {}).get("mentions") or []
            if not any(str(value).casefold() in quote.casefold() for value in mention_values):
                errors.append(f"{label}_mention_missing_from_quote")
        if not RELATION_TRIGGER.search(quote):
            errors.append("interaction_trigger_missing_from_quote")

    try:
        confidence = float(event.get("confidence"))
    except (TypeError, ValueError):
        confidence = -1.0
    if confidence < MIN_CONFIDENCE or confidence > 1.0:
        errors.append("confidence_below_threshold_or_invalid")
    if event.get("interaction_effect") not in EFFECT_VALUES:
        errors.append("invalid_interaction_effect")
    parameter_ids = event.get("parameter_node_ids")
    if not isinstance(parameter_ids, list) or any(item not in parameters for item in parameter_ids):
        errors.append("parameter_id_not_in_candidate_set")
    mediator_ids = event.get("mediator_node_ids")
    if not isinstance(mediator_ids, list) or any(item not in mediators for item in mediator_ids):
        errors.append("mediator_id_not_in_candidate_set")
    return sorted(set(errors))


def unresolved_decision(task: dict[str, Any], *errors: str) -> dict[str, Any]:
    return _decision(
        task,
        status="unresolved",
        decision_method=str(task["routing"]),
        events=[],
        validation_errors=list(errors),
    )


def infer_effect(text: str) -> str:
    if re.search(r"\b(?:no|not|without)\b.{0,40}\b(?:effect|change|affect|alter)\b", text, re.I):
        return "no_effect"
    if re.search(r"\b(?:increase|increased|higher|elevated|prolonged)\b", text, re.I):
        return "increased"
    if re.search(r"\b(?:decrease|decreased|lower|reduced|impaired)\b", text, re.I):
        return "decreased"
    if RELATION_TRIGGER.search(text):
        return "changed"
    return "unknown"


def _llm_payload(task: dict[str, Any]) -> dict[str, Any]:
    return {
        key: task[key]
        for key in (
            "evidence_id",
            "source_text",
            "target_node_id",
            "target_name",
            "target_role",
            "candidate_compounds",
            "parameter_candidates",
            "mediator_candidates",
            "supervised_relation_signals",
            "supervised_candidate_pairs",
        )
    }


def _decision(
    task: dict[str, Any],
    *,
    status: str,
    decision_method: str,
    events: list[dict[str, Any]],
    rejected_events: list[dict[str, Any]] | None = None,
    raw_model_output: str | None = None,
    validation_errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "evidence_id": task["evidence_id"],
        "status": status,
        "decision_method": decision_method,
        "events": events,
        "rejected_events": rejected_events or [],
        "raw_model_output": raw_model_output,
        "validation_errors": validation_errors or [],
        "source_text_hash": task["source_text_hash"],
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "supervised_relation_signals": task.get("supervised_relation_signals") or [],
        "supervised_candidate_pairs": task.get("supervised_candidate_pairs") or [],
        "supervised_gate_required": bool(task.get("supervised_gate_required")),
    }


def _load_signals(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if path is None or not path.exists():
        return output
    for row in _iter_jsonl(path):
        output[str(row["evidence_id"])].append(row)
    return output


def _validate_supervised_inputs(
    signals_path: Path | None,
    summary_path: Path | None,
) -> dict[str, Any] | None:
    if signals_path is None:
        if summary_path is not None:
            raise ValueError("--supervised-summary requires --supervised-signals.")
        return None
    if not signals_path.exists():
        raise FileNotFoundError(signals_path)
    if summary_path is None or not summary_path.exists():
        raise ValueError(
            "A supervised benchmark summary is required when relation signals are supplied."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("accepted"):
        raise ValueError("Supervised relation benchmark did not pass its locked thresholds.")
    expected_sha = str(summary.get("signals_sha256") or "")
    actual_sha = _sha256(signals_path)
    if expected_sha != actual_sha:
        raise ValueError("Supervised relation signals checksum does not match benchmark summary.")
    return summary


def _positive_target_pairs(
    target_node_id: str,
    signals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: dict[frozenset[str], dict[str, Any]] = {}
    for signal in signals:
        if not signal.get("positive_signal"):
            continue
        entity1 = str(signal.get("entity1_node_id") or "")
        entity2 = str(signal.get("entity2_node_id") or "")
        if target_node_id not in {entity1, entity2}:
            continue
        counterpart = entity2 if entity1 == target_node_id else entity1
        if not counterpart.startswith("Compound:") or counterpart == target_node_id:
            continue
        key = frozenset((target_node_id, counterpart))
        candidate = {
            "pair_id": str(signal.get("pair_id") or ""),
            "target_node_id": target_node_id,
            "counterpart_node_id": counterpart,
            "predicted_label": str(signal.get("predicted_label") or ""),
            "predicted_confidence": signal.get("predicted_confidence"),
            "positive_probability": signal.get("positive_probability"),
            "model_commit": signal.get("model_commit"),
            "runner_version": signal.get("runner_version"),
        }
        current = output.get(key)
        if current is None or float(candidate.get("positive_probability") or 0.0) > float(
            current.get("positive_probability") or 0.0
        ):
            output[key] = candidate
    return sorted(output.values(), key=lambda row: (row["counterpart_node_id"], row["pair_id"]))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            yield value


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
