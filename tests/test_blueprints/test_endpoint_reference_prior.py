"""Reference priors must be dimensional, input-only and frozen-source bound."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.blueprint_learning import endpoint_reference_prior as reference

ROOT = Path(__file__).resolve().parents[2]


def calibration(n):
    return [{"index": i, "calibration_id": f"id_{i}", "identity_smiles": f"identity_{i}",
             "residual_log": (i - 10) / 10} for i in range(n)]


def test_dimensional_proxies_and_half_life_is_explicitly_a_proxy():
    actual = reference.endpoint_proxies(2, 3)
    assert actual == pytest.approx({"clearance": 8.4, "vss": 210,
                                    "mrt": 25, "terminal_half_life": math.log(2) * 25})
    assert set(actual) == {"clearance", "vss", "mrt", "terminal_half_life"}


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_parameter_never_builds_reference(bad):
    with pytest.raises(ValueError, match="positive"):
        reference.endpoint_proxies(bad, 1)


@pytest.mark.parametrize("endpoint,value", [("clearance", .42), ("vss", 7),
                                            ("mrt", .1), ("terminal_half_life", .1)])
def test_source_log10_labels_convert_once_to_natural_log_physical_units(endpoint, value):
    assert math.exp(reference.observation_log(-1, endpoint)) == pytest.approx(value)
    assert reference.observation_log(None, endpoint) is None


def test_other_endpoints_do_not_receive_fake_reference_values():
    with pytest.raises(ValueError, match="lacks"):
        reference.observation_log(1, "cmax")


def test_local_twenty_neighbors_and_global_residuals_preserve_signed_errors():
    rows = calibration(30)
    sims = [i / 30 for i in range(30)]
    local = reference.empirical_distribution(10, rows, sims, local=True)
    global_ = reference.empirical_distribution(10, rows, sims, local=False)
    assert local["n"] == 20 and global_["n"] == 30
    assert local["calibration_indices"] == list(reversed(range(10, 30)))
    assert local["weights"] == [1 / 20] * 20
    assert local["reference_values_log"] == pytest.approx([math.log(10) + rows[i]["residual_log"] for i in reversed(range(10, 30))])
    assert min(global_["reference_values_log"]) < math.log(10)
    assert max(global_["reference_values_log"]) > math.log(10)


def test_equal_similarity_ties_resolve_by_identity_hash_not_input_order():
    rows = calibration(30)
    left = reference.empirical_distribution(1, rows, [0] * 30, local=True)
    right = reference.empirical_distribution(1, list(reversed(rows)), [0] * 30, local=True)
    expected = sorted(rows, key=lambda row: hashlib.sha256(row["identity_smiles"].encode()).hexdigest())[:20]
    assert left == right
    assert left["calibration_ids"] == [r["calibration_id"] for r in expected]


@pytest.mark.parametrize("n,assessed", [(0, False), (9, False), (10, True), (19, True), (20, True)])
def test_insufficient_calibration_is_unassessed_without_padding(n, assessed):
    result = reference.empirical_distribution(1, calibration(n), [.1] * n, local=True)
    if assessed:
        assert result["n"] == n
        assert sum(result["weights"]) == pytest.approx(1)
    else:
        assert result["status"] == "unassessed"
        assert result["reference_values_log"] == []


def test_tanimoto_excludes_descriptors_and_uses_binary_morgan_bits():
    query = np.zeros(1024); query[:2] = 1
    pool = np.zeros((3, 1024)); pool[0, :2] = 1; pool[1, 1:3] = 1; pool[2, 3] = 1
    assert reference.tanimoto(query, pool) == pytest.approx([1, 1/3, 0])
    with pytest.raises(ValueError, match="Morgan"):
        reference.tanimoto(np.zeros(1030), np.zeros((2, 1030)))
    pool[0, 0] = 4
    with pytest.raises(ValueError, match="binary"):
        reference.tanimoto(query, pool)


def test_real_frozen_estimators_use_only_calibration_labels_and_reuse_one_artifact(tmp_path, monkeypatch):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip("Use --basetemp in this project's runtime directory")
    source_dir = ROOT / reference.SOURCE_RUN
    if not (source_dir / "priors/model.joblib").exists():
        pytest.skip("Frozen deployment artifacts are not present")
    audit = json.loads((source_dir / "priors/audit.json").read_text())
    cal = set(audit["calibration_indices"])
    actual_load = reference.prior.base.load_parameter_dataset
    actual_read = Path.read_text
    touched = []

    class ForbiddenLabels(dict):
        def __getitem__(self, key):
            raise AssertionError("Attempted to access a training or held-out target label")
        def get(self, key, default=None):
            raise AssertionError("Attempted to access a training or held-out target label")

    def guard_labels(path):
        dataset = actual_load(path)
        for row in dataset.rows:
            if row["index"] not in cal:
                row["labels_log10"] = ForbiddenLabels(row["labels_log10"])
        return dataset

    def guard_read(path, *args, **kwargs):
        if "evaluation" in path.parts or path.name in {"results.json", "results.csv"}:
            raise AssertionError("Held-out outcome file was opened")
        touched.append(str(path))
        return actual_read(path, *args, **kwargs)

    monkeypatch.setattr(reference.prior.base, "load_parameter_dataset", guard_labels)
    monkeypatch.setattr(Path, "read_text", guard_read)
    output = tmp_path / "reference_prior.json"
    artifact = reference.fit_reference_prior(ROOT, output)
    assert len(artifact["targets"]) == 100
    assert artifact["split_audit"]["train_count"] == 817
    assert artifact["split_audit"]["calibration_count"] == 100
    assert not artifact["split_audit"]["target_observation_files_opened"]
    assert {p.name for p in tmp_path.iterdir()} == {"reference_prior.json"}
    for target in artifact["targets"].values():
        assert 0 <= target["max_training_similarity"] <= 1
        assert not target["target_observations_used"]
        for endpoint, entry in target["endpoints"].items():
            assert entry["unit"] == reference.ENDPOINTS[endpoint][1]
            assert entry["local"]["n"] == 20
            assert set(entry["local"]["calibration_indices"]) <= cal
            assert set(entry["global"]["calibration_indices"]) <= cal
    assert reference.fit_reference_prior(ROOT, output) == artifact
    changed = copy.deepcopy(artifact)
    next(iter(changed["targets"].values()))["max_training_similarity"] = .999
    output.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="artifact changed"):
        reference.fit_reference_prior(ROOT, output)
