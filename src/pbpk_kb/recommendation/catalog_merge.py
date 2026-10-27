from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.entity_extraction import normalize_text


MERGER_VERSION = "pbpk-entity-catalog-merger-1.0.0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge validated entity catalogs with collision checks.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary = merge_catalogs(args.input, args.output, args.summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def merge_catalogs(
    input_paths: list[Path],
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    by_node: dict[str, dict[str, Any]] = {}
    input_manifests = []
    for path in input_paths:
        rows = list(_iter_jsonl(path))
        input_manifests.append(
            {"path": str(path), "sha256": _sha256(path), "entry_count": len(rows)}
        )
        for row in rows:
            node_id = str(row["node_id"])
            current = by_node.get(node_id)
            if current is not None and current != row:
                by_node[node_id] = _merge_same_node(current, row)
            else:
                by_node[node_id] = row

    alias_index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for node_id, row in by_node.items():
        for alias in [row.get("preferred_name"), *(row.get("aliases") or [])]:
            normalized = normalize_text(str(alias or ""))
            if normalized:
                alias_index[(str(row["node_type"]), normalized)].add(node_id)
    collisions = [
        {
            "node_type": node_type,
            "normalized_alias": alias,
            "node_ids": sorted(node_ids),
        }
        for (node_type, alias), node_ids in sorted(alias_index.items())
        if len(node_ids) > 1
    ]
    if collisions:
        raise ValueError(f"Ambiguous aliases across merged catalogs: {collisions[:5]}")
    rows = sorted(by_node.values(), key=lambda item: str(item["node_id"]))
    _atomic_jsonl(output_path, rows)
    summary = {
        "accepted": True,
        "merger_version": MERGER_VERSION,
        "inputs": input_manifests,
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "output_entry_count": len(rows),
        "ambiguous_alias_collision_count": 0,
    }
    _atomic_json(summary_path, summary)
    return summary


def _merge_same_node(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if left.get("node_type") != right.get("node_type"):
        raise ValueError(f"Node type collision for {left.get('node_id')}.")
    left_ids = dict(left.get("standard_identifiers") or {})
    right_ids = dict(right.get("standard_identifiers") or {})
    conflicts = {
        key: (left_ids[key], value)
        for key, value in right_ids.items()
        if key in left_ids and left_ids[key] != value
    }
    if conflicts:
        raise ValueError(f"Identifier collision for {left.get('node_id')}: {conflicts}")
    aliases = sorted(
        {
            str(left.get("preferred_name") or ""),
            str(right.get("preferred_name") or ""),
            *(str(item) for item in left.get("aliases") or []),
            *(str(item) for item in right.get("aliases") or []),
        }
        - {""},
        key=str.lower,
    )
    return {
        **left,
        "aliases": aliases,
        "standard_identifiers": {**left_ids, **right_ids},
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            yield row


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
