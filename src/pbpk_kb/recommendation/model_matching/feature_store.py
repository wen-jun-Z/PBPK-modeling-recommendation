from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FEATURE_STORE_VERSION = "pbpk-node-embedding-store-1.2.0"
DEFAULT_BIOMEDICAL_MODEL = "NeuML/pubmedbert-base-embeddings"


def _iter_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}.")
                yield value


def _atomic_json(path: Path, value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        default=str,
    ).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (
                json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def node_text(node: Mapping[str, Any]) -> str:
    attributes = node.get("attributes") or {}
    context = [
        f"type {node.get('node_type')}",
        f"concept {node.get('label')}",
        f"PBPK hierarchy {node.get('hierarchy')}",
    ]
    if attributes.get("canonical_unit"):
        context.append(f"unit {attributes['canonical_unit']}")
    return "; ".join(context)


def collect_nodes(graph_paths: Iterable[Path]) -> list[dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for graph in _iter_jsonl(graph_paths):
        for node in graph.get("nodes") or []:
            key = str(node.get("semantic_key") or "")
            if key:
                nodes.setdefault(key, node)
    return [nodes[key] for key in sorted(nodes)]


def build_embedding_store(
    *,
    graph_paths: list[Path],
    output_directory: Path,
    model_name: str,
    model_path: Path | None,
    device: str,
    batch_size: int,
    molecular_embeddings_path: Path | None = None,
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    output_directory.mkdir(parents=True, exist_ok=False)
    nodes = collect_nodes(graph_paths)
    texts = [node_text(node) for node in nodes]
    resolved_model = str(model_path) if model_path is not None else model_name
    model = SentenceTransformer(
        resolved_model,
        device=device,
        local_files_only=model_path is not None,
    )
    biomedical_vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    biomedical_dimensions = (
        int(biomedical_vectors.shape[1]) if len(nodes) else 0
    )
    molecular_by_smiles: dict[str, list[float]] = {}
    molecular_dimensions = 0
    molecular_checksum = None
    molecular_method = None
    if molecular_embeddings_path is not None:
        from rdkit import Chem

        molecular_checksum = hashlib.sha256(
            molecular_embeddings_path.read_bytes()
        ).hexdigest()
        for row in _iter_jsonl([molecular_embeddings_path]):
            smiles = str(row.get("canonical_smiles") or "").strip()
            vector = [float(value) for value in row.get("embedding") or []]
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None or not vector:
                continue
            canonical = Chem.MolToSmiles(
                molecule, canonical=True, isomericSmiles=True
            )
            if molecular_dimensions and len(vector) != molecular_dimensions:
                raise ValueError("MolCLR embedding dimensions are inconsistent.")
            molecular_dimensions = len(vector)
            existing = molecular_by_smiles.get(canonical)
            if existing is not None and existing != vector:
                raise ValueError(f"Conflicting MolCLR vectors for {canonical}.")
            molecular_by_smiles[canonical] = vector
            method = str(row.get("embedding_method") or "")
            if molecular_method is not None and method != molecular_method:
                raise ValueError("Molecular embedding file mixes methods.")
            molecular_method = method
        if not molecular_by_smiles:
            raise ValueError("No valid MolCLR embeddings were loaded.")

    rows = []
    missing_molecular_nodes: list[str] = []
    for node, text, biomedical_vector in zip(
        nodes, texts, biomedical_vectors, strict=True
    ):
        if molecular_embeddings_path is None:
            vector = [float(value) for value in biomedical_vector]
            feature_source = "biomedical_sentence_embedding"
        elif node.get("node_type") == "MolecularStructure":
            from rdkit import Chem

            molecule = Chem.MolFromSmiles(str(node.get("label") or ""))
            canonical = (
                Chem.MolToSmiles(
                    molecule, canonical=True, isomericSmiles=True
                )
                if molecule is not None
                else ""
            )
            molecular_vector = molecular_by_smiles.get(canonical)
            if molecular_vector is None:
                missing_molecular_nodes.append(str(node["semantic_key"]))
                continue
            vector = [0.0] * biomedical_dimensions + molecular_vector
            feature_source = "official_molclr_pretrained_gin"
        else:
            vector = [float(value) for value in biomedical_vector] + [
                0.0
            ] * molecular_dimensions
            feature_source = "biomedical_sentence_embedding"
        rows.append(
            {
                "semantic_key": node["semantic_key"],
                "node_type": node.get("node_type"),
                "text": text,
                "feature_source": feature_source,
                "embedding": vector,
            }
        )
    if missing_molecular_nodes:
        raise ValueError(
            "MolCLR does not cover molecular graph nodes: "
            + ", ".join(sorted(missing_molecular_nodes)[:20])
        )
    vector_checksum = _atomic_jsonl(output_directory / "node_embeddings.jsonl", rows)
    manifest = {
        "feature_store_version": FEATURE_STORE_VERSION,
        "representation_kind": (
            "type_specific_biomedical_and_molclr_embedding"
            if molecular_embeddings_path is not None
            else "biomedical_sentence_embedding"
        ),
        "model_name": model_name,
        "resolved_model_path": str(model_path) if model_path is not None else None,
        "model_revision": model_path.name if model_path is not None else None,
        "device": device,
        "node_count": len(rows),
        "dimensions": (
            biomedical_dimensions + molecular_dimensions
            if rows
            else 0
        ),
        "biomedical_dimensions": biomedical_dimensions,
        "molecular_dimensions": molecular_dimensions,
        "molecular_embeddings_path": (
            str(molecular_embeddings_path.resolve())
            if molecular_embeddings_path is not None
            else None
        ),
        "molecular_embeddings_sha256": molecular_checksum,
        "molecular_embedding_method": molecular_method,
        "molecular_node_count": sum(
            row["node_type"] == "MolecularStructure" for row in rows
        ),
        "normalized": True,
        "graph_inputs": [str(path) for path in graph_paths],
        "graph_input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in graph_paths
        },
        "node_embeddings_sha256": vector_checksum,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def load_embedding_store(
    directory: Path,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    values = {
        str(row["semantic_key"]): [float(value) for value in row["embedding"]]
        for row in _iter_jsonl([directory / "node_embeddings.jsonl"])
    }
    dimensions = int(manifest["dimensions"])
    if values and any(len(vector) != dimensions for vector in values.values()):
        raise ValueError("Node embedding dimensions do not match the manifest.")
    return values, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build versioned biomedical node embeddings for PBPK graphs."
    )
    parser.add_argument("--graphs", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_BIOMEDICAL_MODEL)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Optional locked local snapshot path for reproducible offline loading.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--molecular-embeddings", type=Path)
    args = parser.parse_args()
    result = build_embedding_store(
        graph_paths=args.graphs,
        output_directory=args.output,
        model_name=args.model,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        molecular_embeddings_path=args.molecular_embeddings,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
