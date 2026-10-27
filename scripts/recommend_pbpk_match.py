"""Rank 280 frozen PBPK candidates for a new structured request JSON.

The request accepts task_type, compounds, and features only. Inference encodes
new request nodes locally and restores a development checkpoint without fitting,
external APIs, service startup or PBPK execution. Research outputs stay uncertain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pbpk_kb.recommendation.pbpk_match.data import scoped, sha
from pbpk_kb.recommendation.pbpk_match.serving import recommend_request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--output", type=Path, help="New project JSON destination; existing files are never overwritten")
    args = parser.parse_args(argv)
    request_path = scoped(args.request_json)
    destination = scoped(args.output) if args.output is not None else None
    if destination is not None and destination.exists():
        raise FileExistsError("Output exists; choose a new recommendation result path")
    if request_path.stat().st_size > 65536:
        raise ValueError("Structured request JSON exceeds 64 KiB")
    request = json.loads(request_path.read_text())
    result = recommend_request(args.checkpoint, request, args.top_k, device=args.device,
                               batch_size=args.embedding_batch_size, chunk_size=args.chunk_size)
    result["request_file_sha256"] = sha(request_path)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    brief = {key: result[key] for key in ("status", "version", "mode", "abstain", "needs_catalog_review", "abstention_reason", "policy_eligible_candidate_count",
                                        "candidate_count", "requested_top_k", "top_k_with_boundary_ties_count",
                                        "probability_calibration", "query_compounds_not_exactly_named_in_catalog")}
    brief["output"] = str(destination) if destination is not None else None
    brief["preview"] = result["top_k_with_boundary_ties"][:10]
    brief["preview_is_truncated"] = len(result["top_k_with_boundary_ties"]) > 10
    print(json.dumps(brief, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
