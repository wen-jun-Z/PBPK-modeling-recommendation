"""Isolation, finite-sample CQR and resumable fixed-context CL tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.blueprint_learning import foundation_clearance_prior as prior


def grouped_rows(n=120):
    return [{"index": i, "status": "train", "scaffold_group": "group_" + str(i // 3),
             "identity_smiles": "identity_" + str(i)} for i in range(n)]


def test_fixed_scaffold_calibration_and_three_oof_folds_are_disjoint():
    rows = grouped_rows()
    split = prior.scaffold_split(rows, range(len(rows)))
    assert split == prior.scaffold_split(rows, reversed(range(len(rows))))
    assert len(split["calibration_indices"]) == 24
    assert len(split["final_context_indices"]) == 96
    held = [i for fold in split["folds"] for i in fold["validation_indices"]]
    assert sorted(held) == split["final_context_indices"]
    groups = lambda indices: {rows[i]["scaffold_group"] for i in indices}
    for fold in split["folds"]:
        assert not groups(fold["train_indices"]) & groups(fold["validation_indices"])
        assert not groups(fold["train_indices"] + fold["validation_indices"]) & groups(split["calibration_indices"])


def test_split_rejects_ineligible_rows_and_insufficient_scaffolds():
    rows = grouped_rows()
    rows[0]["status"] = "excluded_target_identity"
    with pytest.raises(ValueError, match="eligible"):
        prior.scaffold_split(rows, range(len(rows)))
    with pytest.raises(ValueError, match="five"):
        prior.scaffold_split(grouped_rows(9), range(9))


def test_cqr_uses_exact_noninterpolated_order_statistic_and_never_shrinks():
    c = prior.cqr_calibration(np.zeros(19), np.ones(19), np.arange(19))
    assert c["rank_one_based"] == 18
    assert c["qhat_log10"] == 16.
    assert prior.cqr_calibration(np.zeros(10), np.ones(10), np.ones(10) * .5)["qhat_log10"] == 0.


def test_cqr_rejects_unattainable_rank_and_crossed_quantiles():
    with pytest.raises(ValueError, match="unattainable"):
        prior.cqr_calibration(np.zeros(8), np.ones(8), np.zeros(8))
    with pytest.raises(ValueError, match="crossing"):
        prior.cqr_calibration(np.ones(10), np.zeros(10), np.zeros(10))
    with pytest.raises(ValueError, match="finite"):
        prior.cqr_calibration(np.zeros(10), np.ones(10), np.full(10, np.nan))


def test_fold_queries_never_fit_calibration_and_imputer_uses_only_context():
    x = np.column_stack([np.arange(50), np.r_[np.ones(30), np.full(20, 10000.)]])
    x[0, 1] = np.nan
    y = {i: i / 100 for i in range(50)}
    calls = []
    class Model:
        def fit(self, values, labels):
            calls.append((values.copy(), labels.copy()))
            assert set(values[:, 0]) == set(range(30))
            assert values[0, 1] == 1.
            return self
        def predict(self, values, output_type, quantiles):
            assert output_type == "full" and quantiles == [.05, .95]
            middle = values[:, 0] / 100
            return {"median": middle, "quantiles": [middle - .1, middle + .1]}
    record, target = prior.fit_context_predict(x, y, list(range(50)),
        {"train_indices": list(range(30)), "calibration_indices": list(range(30, 40)),
         "validation_indices": list(range(40, 50))}, np.array([[99., np.nan]]), "unused", 2,
        factory=lambda *_: Model())
    assert len(calls) == 1
    assert record["metrics"]["coverage"] == 1.
    assert record["metrics"]["gmfe"] == 1.
    assert record["imputer_statistics"][1] == 1.
    assert target["median"].tolist() == [.99]


def test_calibration_labels_are_read_only_after_fit_and_predict():
    phase = {"predicted": False}
    class Labels(dict):
        def __getitem__(self, index):
            if index >= 20:
                assert phase["predicted"], "Calibration/held labels were read before prediction"
            return index / 100.
    class Model:
        def fit(self, x, y): return self
        def predict(self, x, **kwargs):
            phase["predicted"] = True
            return {"median": np.zeros(len(x)), "quantiles": [-np.ones(len(x)), np.ones(len(x))]}
    prior.fit_context_predict(np.arange(40).reshape(-1, 1), Labels(), list(range(40)),
        {"train_indices": list(range(20)), "calibration_indices": list(range(20, 30)),
         "validation_indices": list(range(30, 40))}, np.empty((0, 1)), "unused", 1,
        factory=lambda *_: Model())


def test_fit_rejects_any_calibration_context_overlap():
    with pytest.raises(ValueError, match="disjoint"):
        prior.fit_context_predict(np.zeros((20, 2)), dict(enumerate(range(20))), list(range(20)),
            {"train_indices": [0, 1], "calibration_indices": [1, 2]}, np.zeros((1, 2)), "unused", 1)


def test_quantile_crossing_is_not_fixed_by_sorting():
    class Invalid:
        def predict(self, x, **kwargs):
            return {"median": np.zeros(len(x)), "quantiles": [np.ones(len(x)), np.ones(len(x)) * 2]}
    with pytest.raises(ValueError, match="crossing"):
        prior._predict_distribution(Invalid(), np.zeros((3, 2)))


def test_features_drop_morgan_not_maccs_and_are_under_500(monkeypatch):
    values = np.arange(2 * 1391).reshape(2, 1391)
    monkeypatch.setattr(prior.rich, "rich_features", lambda _: (values, {"descriptor_names": list(range(200))}))
    features, audit = prior.molecular_features(["a", "b"])
    np.testing.assert_array_equal(features, values[:, 1024:])
    assert audit["dimensions"] == 367 and audit["morgan_bits"] == 0
    assert audit["animal_features"] is False


def test_merge_changes_only_cl_and_keeps_units_in_physical_space():
    source = {"targets": {"a": {}}, "parameters": {"a": {"total_clearance": 99., "fraction_unbound": .2,
        "vss": 2., "terminal_half_life": 3., "mrt": 4.}}}
    original = deepcopy(source)
    params, intervals = prior.merge_predictions(source, {"q05": np.array([0.]), "median": np.array([1.]),
                                                        "q95": np.array([2.])}, .3)
    assert source == original
    assert params["a"]["total_clearance"] == 10.
    for name in ("fraction_unbound", "vss", "terminal_half_life", "mrt"):
        assert params["a"][name] == source["parameters"]["a"][name]
    assert intervals["a"]["lower"] == pytest.approx(10 ** -.3)
    assert intervals["a"]["upper"] == pytest.approx(10 ** 2.3)
    assert intervals["a"]["clinical_coverage_guaranteed"] is False


def test_metrics_are_geometric_not_arithmetic_fold_mean():
    result = prior.interval_metrics([0., 0.], [0., 2.], [-1., -1.], [3., 3.])
    assert result["gmfe"] == 10.
    assert result["within_twofold_n"] == 1
    assert result["coverage_n"] == 2


def fake_run(monkeypatch, tmp_path):
    rows = grouped_rows(60)
    split = prior.scaffold_split(rows, range(60))
    checkpoint = tmp_path / "model.ckpt"; checkpoint.write_bytes(b"test-only")
    source_input = tmp_path / "frozen_source.json"; source_input.write_text("{}")
    ref = lambda path: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    source = {"targets": {"target": {"identity_smiles": "target"}},
        "parameters": {"target": {"total_clearance": 8., "fraction_unbound": .3, "vss": 2., "terminal_half_life": 3., "mrt": 4.}},
        "units": deepcopy(prior.shared.UNITS), "per_compound_audit": {"target": {"training_structure_overlap": False}},
        "audit": {"human_parameter_training_indices": {p: list(range(60)) for p in prior.shared.PARAMETERS}}}
    config = {"split": split, "source_refs": {source_input.name: ref(source_input)["sha256"], checkpoint.name: ref(checkpoint)["sha256"]},
        "code_pins": {}, "checkpoint": ref(checkpoint), "source_parameter_prior": ref(source_input),
        "software": {"tabpfn": "test-fake"}, "calibration_scope": "new CL only"}
    state = {"version": prior.VERSION, "stage": "prepared", "input_fingerprint": "test-fingerprint",
             "config": config, "completed_context_fits": 0, "fold_results": {}}
    def prepared(root, targets, src, output, cp, jobs):
        path = output / "training_state.json"
        if path.exists():
            loaded = json.loads(path.read_text())
            assert loaded.pop("state_sha256") == prior.shared._digest(loaded)
        else:
            loaded = deepcopy(state)
        return loaded, deepcopy(source), rows, {i: i / 100. for i in range(60)}, {"train_count": 60}
    monkeypatch.setattr(prior, "_prepare", prepared)
    def features(identities):
        return np.array([[int(s.split("_")[1]) if s.startswith("identity_") else 99., 1.] for s in identities]), {"dimensions": 2}
    monkeypatch.setattr(prior, "molecular_features", features)
    return source, split, source_input


def test_complete_resume_and_context_never_contains_calibration(tmp_path, monkeypatch):
    source, split, _ = fake_run(monkeypatch, tmp_path)
    fits = []
    class Model:
        def fit(self, x, y): fits.append(x[:, 0].astype(int).tolist()); return self
        def predict(self, x, **kwargs):
            mid = x[:, 0] / 100.
            return {"median": mid, "quantiles": [mid - .1, mid + .1]}
    output = tmp_path / "prior"; output.mkdir()
    result = prior.train(tmp_path, output_dir=output, estimator_factory=lambda *_: Model())
    assert len(fits) == 4
    assert all(not set(fit) & set(split["calibration_indices"]) for fit in fits)
    assert result["audit"]["oof_metrics"]["gmfe"] == 1.
    assert result["parameters"]["target"]["vss"] == source["parameters"]["target"]["vss"]
    with np.load(output / "context.npz") as context:
        assert context["row_indices"].tolist() == split["final_context_indices"]
        assert not set(context["row_indices"]) & set(split["calibration_indices"])
    resumed = prior.train(tmp_path, output_dir=output, estimator_factory=lambda *_: pytest.fail("Must reuse committed result"))
    assert resumed == result
    assert {p.name for p in output.iterdir()} == {"training.lock", "training_state.json", "context.npz", "predictions.json"}


def test_checkpointed_fold_is_not_refitted_after_interruption(tmp_path, monkeypatch):
    _, _, _ = fake_run(monkeypatch, tmp_path)
    fits = []
    class Model:
        def fit(self, x, y): fits.append(len(y)); return self
        def predict(self, x, **kwargs):
            mid = x[:, 0] / 100.
            return {"median": mid, "quantiles": [mid - .1, mid + .1]}
    output = tmp_path / "prior"; output.mkdir()
    def interrupt(event):
        if event["stage"] == "fold_complete": raise InterruptedError("planned test interruption")
    with pytest.raises(InterruptedError):
        prior.train(tmp_path, output_dir=output, estimator_factory=lambda *_: Model(), progress=interrupt)
    assert len(fits) == 1
    result = prior.train(tmp_path, output_dir=output, estimator_factory=lambda *_: Model())
    assert len(fits) == 4 and result["audit"]["context_fits"] == 4


def test_changed_pinned_source_is_rejected_before_resume(tmp_path, monkeypatch):
    _, _, source = fake_run(monkeypatch, tmp_path)
    source.write_text('{"changed": true}')
    output = tmp_path / "prior"; output.mkdir()
    with pytest.raises(ValueError, match="changed"):
        prior.train(tmp_path, output_dir=output, estimator_factory=lambda *_: pytest.fail("Do not fit"))


def test_budget_and_symlinks_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(prior, "MAX_BYTES", 3)
    path = tmp_path / "tiny"; path.write_bytes(b"1234")
    with pytest.raises(RuntimeError, match="10 GiB"):
        prior._check_budget(tmp_path)
    path.unlink()
    (tmp_path / "pointer").symlink_to(tmp_path / "does-not-exist")
    with pytest.raises(ValueError, match="Symlinks"):
        prior._check_budget(tmp_path)


def test_cli_sets_project_only_caches_and_offline_before_ml_imports(monkeypatch):
    import ast
    import os
    import runpy
    import sys
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/train_foundation_clearance_prior.py"
    fields = {"HF_HOME": "hf", "TABPFN_MODEL_CACHE_DIR": "tabpfn", "TORCH_HOME": "torch",
              "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda"}
    for key in (*fields, "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TABPFN_DISABLE_TELEMETRY"):
        monkeypatch.setenv(key, "must-be-overridden")
    monkeypatch.setattr(sys, "path", list(sys.path))
    runpy.run_path(str(script), run_name="_foundation_cli_test")
    for key, directory in fields.items():
        assert os.environ[key] == str(root / "runtime/cache/foundation" / directory)
    for key in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TABPFN_DISABLE_TELEMETRY"):
        assert os.environ[key] == "1"
    # All explicit cache setup statements precede the first project/ML import.
    tree = ast.parse(script.read_text())
    project_import = next(n.lineno for n in tree.body if isinstance(n, ast.ImportFrom)
                          and (n.module or "").startswith("pbpk_kb"))
    assert all(n.lineno < project_import for n in tree.body if isinstance(n, ast.For))


def checkpoint_bundle(tmp_path, monkeypatch):
    directory = tmp_path / Path(prior.CHECKPOINT_SOURCE).parent
    directory.mkdir(parents=True)
    checkpoint = tmp_path / prior.CHECKPOINT
    checkpoint.write_bytes(b"synthetic-test-checkpoint")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(prior, "CHECKPOINT_SHA256", sha(checkpoint))
    monkeypatch.setattr(prior, "CHECKPOINT_BYTES", checkpoint.stat().st_size)
    (directory / "LICENSE.txt").write_text("Synthetic test licence")
    (directory / "config.json").write_text("{}")
    metadata = {"repository": "https://huggingface.co/Prior-Labs/TabPFN-v2-reg",
        "revision": prior.CHECKPOINT_REVISION, "package": "tabpfn==" + prior.EXPECTED_TABPFN_VERSION,
        "checkpoint": {"path": prior.CHECKPOINT, "sha256": sha(checkpoint), "size_bytes": checkpoint.stat().st_size},
        "source_files": {name: sha(directory / name) for name in ("LICENSE.txt", "config.json")}}
    path = tmp_path / prior.CHECKPOINT_SOURCE
    path.write_text(json.dumps(metadata))
    return path, metadata


def test_checkpoint_metadata_license_and_config_all_become_sources(tmp_path, monkeypatch):
    _, metadata = checkpoint_bundle(tmp_path, monkeypatch)
    checkpoint, refs = prior.checkpoint_sources(tmp_path, prior.CHECKPOINT)
    assert checkpoint["sha256"] == metadata["checkpoint"]["sha256"]
    assert set(refs) == {prior.CHECKPOINT_SOURCE, prior.CHECKPOINT,
        str(Path(prior.CHECKPOINT_SOURCE).parent / "LICENSE.txt"),
        str(Path(prior.CHECKPOINT_SOURCE).parent / "config.json")}


@pytest.mark.parametrize("field,value", [("revision", "another-revision"),
    ("package", "tabpfn==99"), ("checkpoint.sha256", "0" * 64),
    ("checkpoint.path", "unexpected.ckpt"), ("checkpoint.size_bytes", 1)])
def test_changed_official_checkpoint_declaration_is_rejected(tmp_path, monkeypatch, field, value):
    path, metadata = checkpoint_bundle(tmp_path, monkeypatch)
    if "." in field:
        parent, child = field.split("."); metadata[parent][child] = value
    else:
        metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="fixed"):
        prior.checkpoint_sources(tmp_path, prior.CHECKPOINT)


def test_changed_checkpoint_or_source_bytes_are_rejected(tmp_path, monkeypatch):
    _, _ = checkpoint_bundle(tmp_path, monkeypatch)
    (tmp_path / prior.CHECKPOINT).write_bytes(b"different-weights")
    with pytest.raises(ValueError, match="fixed official"):
        prior.checkpoint_sources(tmp_path, prior.CHECKPOINT)
    (tmp_path / prior.CHECKPOINT).write_bytes(b"synthetic-test-checkpoint")
    (tmp_path / Path(prior.CHECKPOINT_SOURCE).parent / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="checksum"):
        prior.checkpoint_sources(tmp_path, prior.CHECKPOINT)
