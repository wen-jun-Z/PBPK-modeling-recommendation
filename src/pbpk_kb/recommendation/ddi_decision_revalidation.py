from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.ddi_relation_validation import validate_event


REVALIDATOR_VERSION = "pbpk-ddi-decision-revalidator-1.0.0"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind supervised candidate-pair context and revalidate DDI decisions."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = revalidate_decisions(
        tasks_path=args.tasks,
        decisions_path=args.decisions,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def revalidate_decisions(
    *,
    tasks_path: Path,
    decisions_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    tasks = {str(row["evidence_id"]): row for row in _iter_jsonl(tasks_path)}
    decisions = {str(row["evidence_id"]): row for row in _iter_jsonl(decisions_path)}
    if len(tasks) != sum(1 for _ in _iter_jsonl(tasks_path)):
        raise ValueError("Duplicate evidence_id in DDI tasks.")
    if len(decisions) != sum(1 for _ in _iter_jsonl(decisions_path)):
        raise ValueError("Duplicate evidence_id in DDI decisions.")
    if set(tasks) != set(decisions):
        missing = sorted(set(tasks) - set(decisions))
        unknown = sorted(set(decisions) - set(tasks))
        raise ValueError(f"DDI task/decision IDs differ; missing={missing[:5]}, unknown={unknown[:5]}.")

    output: list[dict[str, Any]] = []
    invalid_event_count = 0
    for evidence_id in sorted(tasks):
        task = tasks[evidence_id]
        decision = decisions[evidence_id]
        if decision.get("source_text_hash") != task.get("source_text_hash"):
            raise ValueError(f"{evidence_id} source_text_hash differs between task and decision.")
        accepted_events = []
        rejected_events = list(decision.get("rejected_events") or [])
        for event in decision.get("events") or []:
            errors = validate_event(task, event)
            if errors:
                invalid_event_count += 1
                rejected_events.append(
                    {
                        "event": event,
                        "validation_errors": errors,
                        "revalidation_version": REVALIDATOR_VERSION,
                    }
                )
            else:
                accepted_events.append(
                    {
                        **event,
                        "validation_status": "auto_validated",
                        "revalidation_version": REVALIDATOR_VERSION,
                    }
                )
        status = "accepted" if accepted_events else "unresolved"
        validation_errors = list(decision.get("validation_errors") or [])
        if decision.get("events") and not accepted_events:
            validation_errors.append("no_event_passed_decision_revalidation")
        output.append(
            {
                **decision,
                "status": status,
                "events": accepted_events,
                "rejected_events": rejected_events,
                "validation_errors": sorted(set(validation_errors)),
                "supervised_candidate_pairs": task.get("supervised_candidate_pairs") or [],
                "supervised_gate_required": bool(task.get("supervised_gate_required")),
                "routing": task.get("routing"),
                "revalidation_version": REVALIDATOR_VERSION,
            }
        )

    output_path = output_dir / "validated_ddi_relation_decisions.jsonl"
    _atomic_jsonl(output_path, output)
    status_counts = Counter(row["status"] for row in output)
    method_counts = Counter(str(row["decision_method"]) for row in output)
    summary = {
        "accepted": invalid_event_count == 0 and len(output) == len(tasks),
        "revalidator_version": REVALIDATOR_VERSION,
        "tasks_path": str(tasks_path),
        "tasks_sha256": _sha256(tasks_path),
        "decisions_path": str(decisions_path),
        "decisions_sha256": _sha256(decisions_path),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "decision_count": len(output),
        "accepted_event_count": sum(len(row["events"]) for row in output),
        "invalid_event_count": invalid_event_count,
        "status_counts": dict(sorted(status_counts.items())),
        "decision_method_counts": dict(sorted(method_counts.items())),
    }
    _atomic_json(output_dir / "ddi_decision_revalidation_summary.json", summary)
    if not summary["accepted"]:
        raise RuntimeError("DDI decision revalidation failed; inspect its summary and rejected events.")
    return summary


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            yield value


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            )
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
