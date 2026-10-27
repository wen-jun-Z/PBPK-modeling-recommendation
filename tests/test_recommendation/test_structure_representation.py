from __future__ import annotations

import json
from pathlib import Path

from pbpk_kb.recommendation.model_matching.execution_experiment import _structure_index
from pbpk_kb.recommendation.model_matching.graphs import semantic_key
from pbpk_kb.recommendation.model_matching.molecular_features import (
    _canonical_isomeric_smiles,
)


def test_execution_structure_index_preserves_enantiomers(tmp_path: Path) -> None:
    path = tmp_path / "structures.jsonl"
    rows = [
        {
            "compound": "left",
            "canonical_smiles": "CC(O)C(=O)O",
            "isomeric_smiles": "C[C@H](O)C(=O)O",
            "source_id": "pubchem",
        },
        {
            "compound": "right",
            "canonical_smiles": "CC(O)C(=O)O",
            "isomeric_smiles": "C[C@@H](O)C(=O)O",
            "source_id": "pubchem",
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    structures, manifest = _structure_index(path)

    assert structures["left"]["canonical_smiles"] != structures["right"][
        "canonical_smiles"
    ]
    assert structures["left"]["structure_representation"] == (
        "canonicalized_isomeric_smiles"
    )
    assert structures["right"]["structure_representation"] == (
        "canonicalized_isomeric_smiles"
    )
    assert manifest["resolved_compound_count"] == 2


def test_molclr_structure_key_preserves_enantiomers() -> None:
    left = _canonical_isomeric_smiles("C[C@H](O)C(=O)O")
    right = _canonical_isomeric_smiles("C[C@@H](O)C(=O)O")

    assert left != right
    assert semantic_key("MolecularStructure", left) != semantic_key(
        "MolecularStructure", right
    )
