from __future__ import annotations

import argparse
import json
from typing import Any

from pbpk_kb.connectors.registry import CONNECTOR_REGISTRY, get_connector
from pbpk_kb.storage.source_registry import sync_source_registry_from_connector_results


def run_sources(
    *,
    sources: list[str],
    query: str,
    retmax: int,
    output_root: str,
    allow_network: bool,
    sync_source_registry: bool = False,
    database_url: str | None = None,
    timeout_seconds: int = 30,
) -> list[dict[str, Any]]:
    source_ids = list(CONNECTOR_REGISTRY) if sources == ["all"] else sources
    results: list[dict[str, Any]] = []
    for source_id in source_ids:
        connector = get_connector(source_id)(
            output_root=output_root,
            allow_network=allow_network,
            timeout_seconds=timeout_seconds,
        )
        result = connector.run(query=query, retmax=retmax)
        results.append(result.model_dump(mode="json"))
    if sync_source_registry:
        synced_sources = sync_source_registry_from_connector_results(results, database_url=database_url)
        synced_by_source = {source["source_id"]: source for source in synced_sources}
        for result in results:
            result["source_registry_sync"] = synced_by_source.get(result["source_id"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PBPK Stage 1 public source connectors.")
    parser.add_argument(
        "--source",
        action="append",
        choices=["all", *CONNECTOR_REGISTRY.keys()],
        default=None,
        help="Source ID to run. Repeat for multiple sources. Defaults to all Milestone 2 sources.",
    )
    parser.add_argument("--query", default="midazolam", help="Minimal connector query term.")
    parser.add_argument("--retmax", type=int, default=3, help="Maximum records per connector.")
    parser.add_argument("--output-root", default=".", help="Root directory for raw/, parsed/, and ingestion_logs/.")
    parser.add_argument("--no-network", action="store_true", help="Disable live source retrieval and write placeholders.")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="Per-request connector timeout in seconds.")
    parser.add_argument(
        "--sync-source-registry",
        action="store_true",
        help="Upsert source_version and source checksum into the PostgreSQL source_registry table after the run.",
    )
    parser.add_argument("--database-url", default=None, help="Optional SQLAlchemy database URL for source registry sync.")
    args = parser.parse_args()

    results = run_sources(
        sources=args.source or ["all"],
        query=args.query,
        retmax=args.retmax,
        output_root=args.output_root,
        allow_network=not args.no_network,
        sync_source_registry=args.sync_source_registry,
        database_url=args.database_url,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
