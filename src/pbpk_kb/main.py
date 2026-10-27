from __future__ import annotations

import argparse
import json
from pathlib import Path

from pbpk_kb.pipeline_utils import (
    DEFAULT_COMPOUNDS,
    DEFAULT_PROCESSING_OUTPUT_ROOT,
    DEFAULT_RAW_OUTPUT_ROOT,
    DEFAULT_SOURCE_REGISTRY,
    PipelineConfig,
    UnifiedStage1Pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete PBPK Stage 1 acquisition, processing, final-evidence, RAG, and database pipeline."
    )
    parser.add_argument("--compounds", default=",".join(DEFAULT_COMPOUNDS))
    parser.add_argument("--raw-output-root", type=Path, default=DEFAULT_RAW_OUTPUT_ROOT)
    parser.add_argument("--processing-output-root", type=Path, default=DEFAULT_PROCESSING_OUTPUT_ROOT)
    parser.add_argument("--sources-yaml", type=Path, default=DEFAULT_SOURCE_REGISTRY)
    parser.add_argument(
        "--raw-run-root",
        type=Path,
        help="Resume from an existing raw run. Incomplete runs are finalized and reconciled before processing.",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Create a new acquisition run even when a matching incomplete unified manifest exists.",
    )
    parser.add_argument(
        "--shared-source",
        action="append",
        default=[],
        help="Refresh a source-wide snapshot, such as osp_pbpk_model_library; repeat as needed.",
    )
    parser.add_argument("--include-combination-labels", action="store_true")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-concurrency", type=int, default=32)
    parser.add_argument(
        "--llm-max-requests",
        type=int,
        default=None,
        help="Operational cap for new raw-processing LLM calls. Omit for all eligible chunks.",
    )
    parser.add_argument("--embedding-model", default="NeuML/pubmedbert-base-embeddings")
    parser.add_argument("--embedding-device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--allow-cpu-embedding", action="store_true")
    parser.add_argument("--skip-database-import", action="store_true")
    parser.add_argument("--append-database", action="store_true")
    parser.add_argument("--database-url")
    parser.add_argument("--docker-container", default="pbpk_stage1_postgres")
    parser.add_argument("--db-user", default="pbpk")
    parser.add_argument("--db-name", default="pbpk_kb")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the exact ordered plan and resolved paths without downloading or processing data.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    compounds = tuple(
        item.strip().lower()
        for item in args.compounds.split(",")
        if item.strip()
    )
    config = PipelineConfig(
        compounds=compounds,
        raw_output_root=args.raw_output_root,
        processing_output_root=args.processing_output_root,
        source_registry_path=args.sources_yaml,
        raw_run_root=args.raw_run_root,
        resume_incomplete=not args.new_run,
        shared_sources=tuple(args.shared_source),
        include_combination_labels=args.include_combination_labels,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_concurrency=args.llm_concurrency,
        llm_max_requests=args.llm_max_requests,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=max(1, args.embedding_batch_size),
        allow_cpu_embedding=args.allow_cpu_embedding,
        import_database=not args.skip_database_import,
        replace_database=not args.append_database,
        database_url=args.database_url,
        docker_container=args.docker_container,
        db_user=args.db_user,
        db_name=args.db_name,
    )
    pipeline = UnifiedStage1Pipeline(config)
    result = pipeline.plan() if args.plan else pipeline.run()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
