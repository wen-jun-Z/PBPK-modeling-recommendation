"""Isolation, missing-label, scale and baseline checks on small synthetic tables."""
import csv
import json

import pytest

from pbpk_kb.blueprints import pk_parameter_learning as learning


SMILES = ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1", "C1CCCCC1",
          "C1CCCC1", "C1CCC1", "C1CC1", "C1CCNCC1", "C1CCOC1", "c1cn[nH]c1",
          "c1ncncn1", "c1cc2ccccc2cc1", "CCO", "CCCO"]


def table(tmp_path, changes=None, smiles=None):
    path = tmp_path / "table.csv"
    rows = [{"smiles_r": smile, **{key: str(.1 + index / 20) for key in learning.ENDPOINTS},
             "human_fup": str(.1 + index / 100)} for index, smile in enumerate(smiles or SMILES)]
    if changes:
        for index, values in changes.items():
            rows[index].update(values)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["smiles_r", *learning.ENDPOINTS])
        writer.writeheader()
        writer.writerows(rows)
    return learning.load_parameter_dataset(path)


def test_salt_stereochemistry_tautomer_and_acyclic_identity_policy():
    assert learning.normalize_structure("CC(=O)[O-].[Na+]")["identity_smiles"] == learning.normalize_structure("CC(=O)O")["identity_smiles"]
    assert learning.normalize_structure("C[C@H](O)F")["identity_smiles"] == learning.normalize_structure("C[C@@H](O)F")["identity_smiles"]
    assert learning.normalize_structure("O=c1cccc[nH]1")["identity_smiles"] == learning.normalize_structure("Oc1ccccn1")["identity_smiles"]
    assert learning.normalize_structure("CCO")["scaffold_group"] == learning.normalize_structure("CCCC")["scaffold_group"] == "ACYCLIC"


def test_external_drug_and_same_scaffold_excluded_before_sampling(tmp_path):
    data = table(tmp_path)
    split = learning.make_scaffold_split(data, exclude_smiles={"benzene": "c1ccccc1"}, train_limit=6, test_limit=3)
    by_index = {row["index"]: row for row in split["rows"]}
    assert by_index[0]["reason"] == "external_test_identity"
    assert by_index[1]["reason"] == "external_test_scaffold"
    assert set(split["train_indices"]).isdisjoint(split["test_indices"])
    assert len(split["rows"]) == len(data.rows)
    assert split == learning.make_scaffold_split(data, exclude_smiles={"benzene": "c1ccccc1"}, train_limit=6, test_limit=3)
    assert len(split["train_indices"]) <= 6 and len(split["test_indices"]) <= 3
    json.dumps(split, allow_nan=False)


def test_split_does_not_depend_on_labels_and_deduplicates_connectivity(tmp_path):
    data = table(tmp_path, smiles=SMILES + ["c1ccccc1"])
    first = learning.make_scaffold_split(data, train_limit=6, test_limit=3)
    for row in data.rows:
        row["labels_log10"] = {key: None for key in learning.ENDPOINTS}
    second = learning.make_scaffold_split(data, train_limit=6, test_limit=3)
    assert first["train_indices"] == second["train_indices"]
    assert first["test_indices"] == second["test_indices"]
    assert first["rows"][-1]["reason"] == "duplicate_connectivity"
    assert first["rows"][-1]["representative_index"] == 0


def test_scaffold_and_acyclic_groups_never_cross_split(tmp_path):
    data = table(tmp_path)
    for seed in range(4):
        split = learning.make_scaffold_split(data, train_limit=8, test_limit=5, seed=seed)
        train_groups = {data.rows[i]["scaffold_group"] for i in split["train_indices"]}
        test_groups = {data.rows[i]["scaffold_group"] for i in split["test_indices"]}
        assert train_groups.isdisjoint(test_groups)


def test_scale_and_invalid_labels_are_explicit(tmp_path):
    data = table(tmp_path, {0: {"human_fup": "0.01", "human_CL_mL_min_kg": "1"},
                            1: {"human_fup": "0", "human_thalf": "NaN"},
                            2: {"human_fup": "1.1", "human_mrt": "inf"}})
    assert data.rows[0]["labels_log10"]["human_fup"] == -2
    assert data.rows[0]["labels_log10"]["human_CL_mL_min_kg"] == 1
    assert data.rows[1]["labels_log10"]["human_fup"] is None
    assert data.rows[1]["label_exclusions"]["human_thalf"] == "missing"
    assert data.rows[2]["label_exclusions"]["human_mrt"] == "not_finite"


def test_endpoint_training_uses_only_present_training_labels_and_train_median(tmp_path):
    data = table(tmp_path)
    split = learning.make_scaffold_split(data, train_limit=8, test_limit=5)
    missing_train, missing_test = split["train_indices"][0], split["test_indices"][0]
    data.rows[missing_train]["labels_log10"]["human_CL_mL_min_kg"] = None
    data.rows[missing_test]["labels_log10"]["human_CL_mL_min_kg"] = None
    for row in data.rows:
        row["labels_log10"]["human_thalf"] = None
    result = learning.fit_parameter_pilot(data, split, n_jobs=1)
    audit = result.audit["label_audit"]["human_CL_mL_min_kg"]
    assert missing_train in audit["train_missing_indices"]
    assert missing_test in audit["test_missing_indices"]
    assert result.metrics["human_CL_mL_min_kg"]["random_forest"]["n"] == len(split["test_indices"]) - 1
    expected = sorted(data.rows[i]["labels_log10"]["human_CL_mL_min_kg"] for i in audit["train_indices"])
    assert result.medians_log10["human_CL_mL_min_kg"] == expected[len(expected) // 2]
    assert result.metrics["human_thalf"]["status"] == "no_training_labels"
    assert "human_thalf" not in result.models
    assert not any(row["endpoint"] == "human_thalf" for row in result.predictions)
    predicted = result.predict("c1ccccc1")
    assert 0 < predicted["predictions"]["human_fup"]["random_forest"] <= 1
    assert predicted["predictions"]["human_thalf"]["random_forest"] is None
    assert predicted["eligible_as_measured_evidence"] is False
    for record in [result.audit, result.metrics, result.predictions, predicted]:
        json.dumps(record, allow_nan=False)


def test_metrics_compare_physical_multiplicative_error():
    metrics = learning._metrics([0., 1.], [1., 1.])
    assert metrics["gmfe"] == pytest.approx(10 ** .5)
    assert metrics["within_2fold"] == .5
    assert metrics["log10_rmse"] == pytest.approx((.5) ** .5)
    assert learning._metrics([], [])["gmfe"] is None


def test_rejects_tampered_split_and_unbounded_workers(tmp_path):
    data = table(tmp_path)
    split = learning.make_scaffold_split(data, train_limit=8, test_limit=5)
    with pytest.raises(ValueError, match="n_jobs"):
        learning.fit_parameter_pilot(data, split, n_jobs=5)
    split["test_indices"].append(split["train_indices"][0])
    with pytest.raises(ValueError, match="leakage"):
        learning.fit_parameter_pilot(data, split)
