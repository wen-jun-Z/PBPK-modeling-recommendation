from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pbpk_kb.recommendation.model_matching.benchmark import _atomic_json, _atomic_jsonl


MOLCLR_FEATURE_VERSION = "official-molclr-pretrained-gin-1.1.0"
OFFICIAL_REPOSITORY = "https://github.com/yuyangw/MolCLR"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _load_official_model_class(repository: Path) -> type:
    module_path = repository / "models" / "ginet_molclr.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"MolCLR model module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("official_molclr_ginet", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load MolCLR model module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module.GINet


def _molecule_data(smiles: str):
    import torch
    from rdkit import Chem
    from torch_geometric.data import Data

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse canonical_smiles")

    chirality = {
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    }
    bond_types = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    bond_directions = {
        Chem.rdchem.BondDir.NONE: 0,
        Chem.rdchem.BondDir.ENDUPRIGHT: 1,
        Chem.rdchem.BondDir.ENDDOWNRIGHT: 2,
    }

    atoms: list[list[int]] = []
    for atom in molecule.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        if not 1 <= atomic_number <= 118:
            raise ValueError(f"Unsupported atomic number: {atomic_number}")
        if atom.GetChiralTag() not in chirality:
            raise ValueError(f"Unsupported MolCLR chirality: {atom.GetChiralTag()}")
        atoms.append([atomic_number - 1, chirality[atom.GetChiralTag()]])

    rows: list[int] = []
    columns: list[int] = []
    edge_features: list[list[int]] = []
    for bond in molecule.GetBonds():
        if bond.GetBondType() not in bond_types:
            raise ValueError(f"Unsupported MolCLR bond type: {bond.GetBondType()}")
        if bond.GetBondDir() not in bond_directions:
            raise ValueError(f"Unsupported MolCLR bond direction: {bond.GetBondDir()}")
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = [
            bond_types[bond.GetBondType()],
            bond_directions[bond.GetBondDir()],
        ]
        rows.extend([begin, end])
        columns.extend([end, begin])
        edge_features.extend([feature, feature])

    edge_index = torch.tensor([rows, columns], dtype=torch.long)
    if edge_features:
        edge_attr = torch.tensor(edge_features, dtype=torch.long)
    else:
        edge_attr = torch.empty((0, 2), dtype=torch.long)
    return Data(
        x=torch.tensor(atoms, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def _canonical_isomeric_smiles(smiles: str) -> str:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse structure SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def build_molclr_features(
    *,
    structures_path: Path,
    repository: Path,
    checkpoint: Path,
    output_directory: Path,
    device: str = "cuda",
    batch_size: int = 64,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.data import Batch

    if output_directory.exists():
        raise FileExistsError(f"Output directory already exists: {output_directory}")
    if not structures_path.is_file():
        raise FileNotFoundError(structures_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    origin = _git(repository, "config", "--get", "remote.origin.url")
    if "yuyangw/MolCLR" not in origin:
        raise RuntimeError(
            f"Expected the official MolCLR repository ({OFFICIAL_REPOSITORY}); got {origin}"
        )
    repository_commit = _git(repository, "rev-parse", "HEAD")
    license_path = repository / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError(f"MolCLR LICENSE not found: {license_path}")

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("MolCLR checkpoint is not a state dictionary")

    # These dimensions are verified from the official checkpoint tensors. Strict
    # loading below rejects architectural drift or a different checkpoint family.
    embedding_dimension = int(state["x_embedding1.weight"].shape[1])
    feature_dimension = int(state["feat_lin.weight"].shape[0])
    layer_count = len(
        {
            key.split(".")[1]
            for key in state
            if key.startswith("gnns.") and key.endswith("mlp.0.weight")
        }
    )
    model_class = _load_official_model_class(repository)
    model = model_class(
        num_layer=layer_count,
        emb_dim=embedding_dimension,
        feat_dim=feature_dimension,
        drop_ratio=0,
        pool="mean",
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    valid_by_structure: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for row in _read_jsonl(structures_path):
        compound = row.get("compound")
        isomeric_smiles = str(row.get("isomeric_smiles") or "").strip()
        connectivity_smiles = str(row.get("canonical_smiles") or "").strip()
        smiles = isomeric_smiles or connectivity_smiles
        if not compound or not smiles:
            failures.append(
                {
                    "compound": compound,
                    "status": "unavailable_structure",
                    "reason": row.get("failure") or "canonical_smiles is missing",
                }
            )
            continue
        try:
            canonical = _canonical_isomeric_smiles(str(smiles))
            entry = valid_by_structure.get(canonical)
            if entry is None:
                entry = {
                    "canonical_smiles": canonical,
                    "data": _molecule_data(canonical),
                    "rows": [],
                }
                valid_by_structure[canonical] = entry
            entry["rows"].append(
                {
                    **row,
                    "canonical_smiles": canonical,
                    "structure_representation": (
                        "canonicalized_isomeric_smiles"
                        if isomeric_smiles
                        else "canonicalized_connectivity_smiles_fallback"
                    ),
                    "source_isomeric_smiles": isomeric_smiles or None,
                    "source_connectivity_smiles": connectivity_smiles or None,
                }
            )
        except (TypeError, ValueError) as exc:
            failures.append(
                {
                    "compound": compound,
                    "status": "unsupported_structure",
                    "reason": str(exc),
                }
            )

    unique_structures = [valid_by_structure[key] for key in sorted(valid_by_structure)]
    embeddings: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(unique_structures), batch_size):
            structures = unique_structures[start : start + batch_size]
            batch = Batch.from_data_list(
                [structure["data"] for structure in structures]
            ).to(device)
            features, _projection = model(batch)
            features = functional.normalize(features, p=2, dim=1).cpu().tolist()
            for structure, embedding in zip(structures, features, strict=True):
                for row in structure["rows"]:
                    embeddings.append(
                        {
                            "compound": row["compound"],
                            "canonical_smiles": row["canonical_smiles"],
                            "structure_representation": row[
                                "structure_representation"
                            ],
                            "source_isomeric_smiles": row[
                                "source_isomeric_smiles"
                            ],
                            "source_connectivity_smiles": row[
                                "source_connectivity_smiles"
                            ],
                            "inchi_key": row.get("inchi_key"),
                            "embedding": embedding,
                            "embedding_dimension": feature_dimension,
                            "embedding_method": "official_molclr_pretrained_gin",
                            "source_id": row.get("source_id"),
                            "source_url": row.get("source_url"),
                            "structure_response_sha256": row.get(
                                "response_sha256"
                            ),
                        }
                    )

    output_directory.mkdir(parents=True, exist_ok=False)
    embedding_sha256 = _atomic_jsonl(
        output_directory / "molclr_embeddings.jsonl", embeddings
    )
    failure_sha256 = _atomic_jsonl(
        output_directory / "molclr_failures.jsonl", failures
    )
    manifest = {
        "version": MOLCLR_FEATURE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_role": "compound_domain_auxiliary_feature_only",
        "scientific_claim_eligible": False,
        "repository": OFFICIAL_REPOSITORY,
        "repository_origin": origin,
        "repository_commit": repository_commit,
        "license_path": str(license_path),
        "license_sha256": _sha256(license_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_architecture": {
            "backbone": "GIN",
            "num_layers": layer_count,
            "hidden_dimension": embedding_dimension,
            "feature_dimension": feature_dimension,
            "pool": "mean",
        },
        "structures_path": str(structures_path),
        "structures_sha256": _sha256(structures_path),
        "device": device,
        "batch_size": batch_size,
        "resolved_structure_count": len(embeddings),
        "unique_structure_count": len(unique_structures),
        "failure_count": len(failures),
        "embedding_sha256": embedding_sha256,
        "failure_sha256": failure_sha256,
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build compound features from the official pretrained MolCLR GIN."
    )
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = build_molclr_features(
        structures_path=args.structures,
        repository=args.repository,
        checkpoint=args.checkpoint,
        output_directory=args.output,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
