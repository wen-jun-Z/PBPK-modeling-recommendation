from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from pbpk_kb.recommendation.graph_prompt import (
    GraphPromptCorpus,
    RetrievedSubgraph,
    assemble_recommendation,
    validate_recommendation,
)
from pbpk_kb.recommendation.online_agent import (
    _atomic_json,
    refresh_reassembled_run_manifest,
)
from pbpk_kb.recommendation.schemas import ModelingRequest
from pbpk_kb.storage.import_recommendation_runs import _validated_online_row


class ComponentRunReassemblyError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComponentRunReassemblyError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ComponentRunReassemblyError(f"Expected JSONL objects: {path}")
    return rows


def _selection(run_dir: Path) -> dict[str, list[str]]:
    row = _read_json(run_dir / "llm_selection.json")
    selection = row.get("selection_ids")
    required = {
        "component_ids",
        "evidence_ids",
        "public_model_ids",
        "gap_codes",
    }
    if not isinstance(selection, dict) or set(selection) != required:
        raise ComponentRunReassemblyError(
            f"Invalid persisted LLM selection: {run_dir.name}"
        )
    if any(
        not isinstance(selection[key], list)
        or any(not isinstance(value, str) for value in selection[key])
        for key in required
    ):
        raise ComponentRunReassemblyError(
            f"Persisted selection IDs must be string lists: {run_dir.name}"
        )
    return {key: list(selection[key]) for key in sorted(required)}


def _retrieved_subgraphs(
    *,
    run_dir: Path,
    corpus: GraphPromptCorpus,
) -> list[RetrievedSubgraph]:
    retrieval = _read_json(run_dir / "online_retrieval.json")
    selected = list(retrieval.get("selected_subgraph_ids") or ())
    reranked = _read_jsonl(run_dir / "reranked_subgraphs.jsonl")
    reranked_ids = [str(row.get("subgraph_id") or "") for row in reranked]
    if selected != reranked_ids or len(selected) != len(set(selected)):
        raise ComponentRunReassemblyError(
            f"Retrieval selection/reranking drift: {run_dir.name}"
        )
    missing = [subgraph_id for subgraph_id in selected if subgraph_id not in corpus.subgraphs]
    if missing:
        raise ComponentRunReassemblyError(
            f"Runtime subgraphs are unavailable: {run_dir.name}: {missing}"
        )
    return [
        RetrievedSubgraph(
            subgraph_id=str(row["subgraph_id"]),
            hop=int(row.get("hop") or 0),
            score=float(row.get("score") or 0.0),
            score_components={
                str(key): float(value)
                for key, value in (row.get("score_components") or {}).items()
            },
            row=corpus.subgraphs[str(row["subgraph_id"])],
            vector=None,
        )
        for row in reranked
    ]


def reassemble_component_runs(
    *,
    source_dir: Path,
    output_dir: Path,
    graph_dir: Path,
    subgraph_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    source_runs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if not source_runs:
        raise ComponentRunReassemblyError("Source recommendation batch is empty.")
    corpus = GraphPromptCorpus(
        graph_dir=graph_dir,
        subgraph_dir=subgraph_dir,
    )
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    summaries: list[dict[str, Any]] = []
    try:
        for source_run in source_runs:
            run_dir = temporary / source_run.name
            shutil.copytree(source_run, run_dir)
            request = ModelingRequest.model_validate(
                _read_json(run_dir / "recommendation_request.json")
            )
            subgraphs = _retrieved_subgraphs(run_dir=run_dir, corpus=corpus)
            recommendation = assemble_recommendation(
                corpus=corpus,
                request=request,
                subgraphs=subgraphs,
                selection=_selection(run_dir),
            )
            validation = validate_recommendation(
                corpus=corpus,
                request=request,
                recommendation=recommendation,
                allowed_subgraph_ids={row.subgraph_id for row in subgraphs},
            )
            if validation["status"] != "recommendation_validated":
                raise ComponentRunReassemblyError(
                    f"Reassembled recommendation failed validation: "
                    f"{source_run.name}: {validation['errors']}"
                )
            _atomic_json(run_dir / "recommendation_draft.json", recommendation)
            recommendation["status"] = "recommendation_validated"
            _atomic_json(run_dir / "recommendation_final.json", recommendation)
            _atomic_json(run_dir / "recommendation_validation.json", validation)
            source_manifest = _read_json(
                source_run / "recommendation_run_manifest.json"
            )
            refresh_reassembled_run_manifest(
                run_dir=run_dir,
                runtime_artifact_hashes=dict(
                    source_manifest.get("runtime_artifact_hashes") or {}
                ),
                reason=(
                    "Reassembled from checksum-preserved retrieval and LLM "
                    "selection with exact target-compound citation context."
                ),
            )
            validated = _validated_online_row(run_dir)
            summaries.append(
                {
                    "run_id": source_run.name,
                    "compound": validated["request"]["compound"],
                    "component_count": len(
                        recommendation.get("component_recommendations") or ()
                    ),
                    "citation_count": len(recommendation.get("citations") or ()),
                    "accepted_parameter_count": len(
                        recommendation.get("parameter_candidates") or ()
                    ),
                    "rejected_parameter_binding_count": len(
                        recommendation.get("parameter_candidate_rejections") or ()
                    ),
                    "gap_count": len(recommendation.get("evidence_gaps") or ()),
                }
            )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "completed",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "run_count": len(summaries),
        "runs": sorted(summaries, key=lambda row: str(row["compound"]).casefold()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reassemble a locked component-recommendation batch without "
            "rerunning retrieval, GNN inference, or the LLM."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--subgraph-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = reassemble_component_runs(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        graph_dir=args.graph_dir,
        subgraph_dir=args.subgraph_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
