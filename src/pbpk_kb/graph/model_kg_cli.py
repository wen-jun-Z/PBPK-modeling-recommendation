from __future__ import annotations

import argparse
import json
from pathlib import Path

from pbpk_kb.graph.model_kg_audit import (
    audit_model_kg,
    write_model_kg_audit,
)
from pbpk_kb.graph.model_kg_builder import build_model_kg
from pbpk_kb.graph.model_kg_merge import merge_model_kgs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and independently audit the PBPK Model KG."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="Build a versioned Model KG from an accepted Gate 1 package.",
    )
    build.add_argument(
        "--model-versions",
        type=Path,
        required=True,
    )
    build.add_argument(
        "--gate1-audit",
        type=Path,
        required=True,
    )
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser(
        "audit",
        help="Audit schema, provenance, isolation, and snapshot parity.",
    )
    audit.add_argument("--graph", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    merge = subparsers.add_parser(
        "merge",
        help="Merge independently audited Model KG snapshots.",
    )
    merge.add_argument(
        "--graph",
        type=Path,
        action="append",
        required=True,
    )
    merge.add_argument(
        "--audit",
        type=Path,
        action="append",
        required=True,
    )
    merge.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "build":
        summary = build_model_kg(
            model_version_directory=args.model_versions,
            gate1_audit_path=args.gate1_audit,
            plan_path=args.plan,
            output_directory=args.output,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == "merge":
        summary = merge_model_kgs(
            graph_directories=args.graph,
            audit_paths=args.audit,
            output_directory=args.output,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    report = audit_model_kg(args.graph)
    report_sha256 = write_model_kg_audit(args.output, report)
    print(
        json.dumps(
            {
                **report,
                "report_path": str(args.output),
                "report_sha256": report_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
