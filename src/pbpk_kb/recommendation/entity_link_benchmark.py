from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.entity_link_validation import (
    SYSTEM_PROMPT,
    VALIDATOR_VERSION,
    _candidate_options,
    _contexts_for,
    _validate_model_decision,
)
from pbpk_kb.recommendation.local_llm import (
    DEFAULT_LOCAL_LLM,
    LocalStructuredGenerator,
    StructuredGeneration,
    _extract_json_object,
)


MIN_ACCURACY = 0.90
MAX_FALSE_ACCEPT_RATE = 0.05
MIN_VALID_JSON_RATE = 0.99
MIN_QUOTE_VALIDITY = 1.00


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the locked silver benchmark for entity context reranking.")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--resolver-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_LOCAL_LLM)
    parser.add_argument("--positive-count", type=int, default=30)
    parser.add_argument("--replay-report", type=Path)
    args = parser.parse_args()
    evidence = {str(row["evidence_id"]): row for row in _iter_jsonl(args.evidence)}
    decisions = list(_iter_jsonl(args.resolver_decisions))
    examples = build_locked_silver_examples(decisions, evidence, positive_count=args.positive_count)
    if args.replay_report:
        generator = ReplayStructuredGenerator.from_report(args.replay_report)
    else:
        generator = LocalStructuredGenerator(args.model)
    result = run_benchmark(examples, generator)
    if args.replay_report:
        result["replayed_from"] = str(args.replay_report)
        result["replayed_from_sha256"] = _sha256(args.replay_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["accepted"]:
        raise SystemExit(2)


def build_locked_silver_examples(
    decisions: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    *,
    positive_count: int,
) -> list[dict[str, Any]]:
    positives = []
    for decision in sorted(decisions, key=lambda item: str(item.get("normalized_name") or "")):
        if decision.get("pubchem", {}).get("status") != "resolved":
            continue
        if decision.get("rxnorm", {}).get("status") != "resolved":
            continue
        contexts = _contexts_for(decision, evidence)
        candidates = _candidate_options(decision)
        if len(candidates) != 1 or not contexts:
            continue
        if not any(str(decision["name"]).lower() in item["text"].lower() for item in contexts):
            continue
        positives.append(
            {
                "example_id": f"positive:{decision['normalized_name']}",
                "label": "accepted",
                "mention": decision["name"],
                "roles": decision.get("roles") or [],
                "contexts": contexts,
                "candidates": candidates,
                "expected_candidate_id": candidates[0]["candidate_id"],
            }
        )
        if len(positives) >= positive_count:
            break
    if len(positives) < 2:
        raise ValueError("At least two dual-source silver positives are required.")
    negatives = []
    for index, positive in enumerate(positives):
        wrong_candidate = positives[(index + 1) % len(positives)]["candidates"][0]
        negatives.append(
            {
                "example_id": f"hard-negative:{positive['mention']}:{wrong_candidate['candidate_id']}",
                "label": "unresolved",
                "mention": positive["mention"],
                "roles": positive["roles"],
                "contexts": positive["contexts"],
                "candidates": [wrong_candidate],
                "expected_candidate_id": None,
            }
        )
    return [item for pair in zip(positives, negatives) for item in pair]


def run_benchmark(examples: list[dict[str, Any]], generator: Any) -> dict[str, Any]:
    rows = []
    for index, example in enumerate(examples, start=1):
        raw = generator.generate_json(
            system_prompt=SYSTEM_PROMPT,
            payload={
                "mention": example["mention"],
                "roles": example["roles"],
                "contexts": example["contexts"],
                "candidates": example["candidates"],
            },
        )
        decision = _validate_model_decision(
            raw,
            mention=example["mention"],
            contexts=example["contexts"],
            candidates=example["candidates"],
            min_confidence=0.80,
        )
        predicted = decision["decision"]
        correct = (
            predicted == "accepted"
            and example["label"] == "accepted"
            and decision["selected_candidate_id"] == example["expected_candidate_id"]
        ) or (
            predicted in {"ambiguous", "unresolved"}
            and example["label"] == "unresolved"
        )
        rows.append(
            {
                "example_id": example["example_id"],
                "label": example["label"],
                "decision": decision,
                "valid_json": raw.parsed is not None,
                "correct": correct,
            }
        )
        print(json.dumps({"benchmark": index, "total": len(examples), "correct": correct}), flush=True)
    count = len(rows)
    negatives = [item for item in rows if item["label"] == "unresolved"]
    accepted_rows = [item for item in rows if item["decision"]["decision"] == "accepted"]
    accuracy = sum(item["correct"] for item in rows) / count
    false_accept_rate = (
        sum(item["decision"]["decision"] == "accepted" for item in negatives) / len(negatives)
        if negatives
        else 0.0
    )
    valid_json_rate = sum(item["valid_json"] for item in rows) / count
    quote_validity = (
        sum(not item["decision"]["validation_errors"] for item in accepted_rows) / len(accepted_rows)
        if accepted_rows
        else 0.0
    )
    checks = {
        "decision_accuracy": accuracy >= MIN_ACCURACY,
        "false_accept_rate": false_accept_rate <= MAX_FALSE_ACCEPT_RATE,
        "valid_json_rate": valid_json_rate >= MIN_VALID_JSON_RATE,
        "accepted_quote_validity": quote_validity >= MIN_QUOTE_VALIDITY,
    }
    return {
        "accepted": all(checks.values()),
        "benchmark_type": "locked_silver_dual_source_positive_and_mismatched_hard_negative",
        "gold_claim": False,
        "model_name": generator.model_name,
        "validator_version": VALIDATOR_VERSION,
        "example_count": count,
        "metrics": {
            "decision_accuracy": accuracy,
            "false_accept_rate": false_accept_rate,
            "valid_json_rate": valid_json_rate,
            "accepted_quote_validity": quote_validity,
        },
        "thresholds": {
            "min_accuracy": MIN_ACCURACY,
            "max_false_accept_rate": MAX_FALSE_ACCEPT_RATE,
            "min_valid_json_rate": MIN_VALID_JSON_RATE,
            "min_quote_validity": MIN_QUOTE_VALIDITY,
        },
        "checks": checks,
        "examples": rows,
    }


class ReplayStructuredGenerator:
    def __init__(self, model_name: str, raw_outputs: list[str]) -> None:
        self.model_name = model_name
        self.raw_outputs = raw_outputs
        self.position = 0

    @classmethod
    def from_report(cls, path: Path) -> "ReplayStructuredGenerator":
        report = json.loads(path.read_text(encoding="utf-8"))
        raw_outputs = [str(item["decision"].get("raw_model_output") or "") for item in report["examples"]]
        return cls(str(report["model_name"]), raw_outputs)

    def generate_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        max_new_tokens: int = 768,
    ) -> StructuredGeneration:
        del system_prompt, payload, max_new_tokens
        if self.position >= len(self.raw_outputs):
            raise RuntimeError("Replay report contains fewer outputs than the locked benchmark.")
        raw = self.raw_outputs[self.position]
        self.position += 1
        try:
            parsed = _extract_json_object(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            return StructuredGeneration(None, raw, f"{type(exc).__name__}:{exc}")
        return StructuredGeneration(parsed, raw, None)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
