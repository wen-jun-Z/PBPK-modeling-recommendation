"""Protect request conditions and held-out membership at the new interface."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts/run_component_model_integration.py"
_SPEC = importlib.util.spec_from_file_location("component_integration_inference", _PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def query():
    return {"query_id": "q", "compounds": ["drug"], "task_type": "plasma_concentration_time_prediction",
            "text": "Dose 25 mg orally, fasted; predict Cmax.",
            "features": {"species": ["human"], "routes": ["oral"], "endpoints": ["cmax"],
                         "enzymes": ["cyp2c9"]}}


def test_only_soft_mechanism_columns_can_change():
    from pbpk_kb.recommendation.pbpk_match_v2.data import FIELDS
    original = query()
    augmented = deepcopy(original)
    augmented["text"] += "\nSoft preferences: renal excretion."
    augmented["features"].update(enzymes=["cyp2c9", "cyp3a4"], process_types=["excretion"])
    candidates = [{"candidate_id": "m", "compounds": ["drug"], "text": "Public description",
                   "features": {"enzymes": ["cyp3a4"], "process_types": ["excretion"]}}]
    frozen = np.arange(204, dtype=np.float32).reshape(1, 204)
    before = frozen.copy()
    result = runner.soft_features(original, augmented, candidates, frozen)
    protected = [i for field in FIELDS if field not in runner.SOFT_FIELDS
                 for i in range(FIELDS.index(field) * 5, FIELDS.index(field) * 5 + 5)] + list(range(60, 204))
    np.testing.assert_array_equal(result[:, protected], frozen[:, protected])
    np.testing.assert_array_equal(frozen, before)
    assert not np.array_equal(result, frozen)
    assert original == query()


@pytest.mark.parametrize("mutation", ["dose_text", "species", "route", "compound", "task", "remove_enzyme"])
def test_components_cannot_replace_user_conditions(mutation):
    original = query()
    changed = deepcopy(original)
    if mutation == "dose_text":
        changed["text"] = changed["text"].replace("25", "50")
    elif mutation == "species":
        changed["features"]["species"] = ["mouse"]
    elif mutation == "route":
        changed["features"]["routes"] = ["intravenous"]
    elif mutation == "compound":
        changed["compounds"] = ["another"]
    elif mutation == "task":
        changed["task_type"] = "ddi_exposure_prediction"
    else:
        changed["features"]["enzymes"] = []
    with pytest.raises(ValueError):
        runner.soft_features(original, changed, [], np.zeros((0, 204), np.float32))


def test_no_candidate_or_label_values_change_fold_assignment():
    protocol = {"groups": ["a", "b", "c"], "folds": [
        {"outer_train": [1, 2], "outer_test": [0]},
        {"outer_train": [0], "outer_test": [1, 2]}]}
    assert runner.exact_fold_membership([0, 2], protocol) == {0: 0, 2: 1}
    protocol["groups"][1] = "a"
    with pytest.raises(ValueError, match="group"):
        runner.exact_fold_membership([0], protocol)


def test_holdout_cannot_be_missing_or_duplicated():
    for folds in [[], [{"outer_test": [0], "outer_train": []}] * 2]:
        with pytest.raises(ValueError, match="unique"):
            runner.exact_fold_membership([0], {"folds": folds, "groups": ["a"]})


def test_changed_checkpoint_cannot_be_rebound_as_frozen(monkeypatch):
    original_sha = runner.sha
    target = runner.OLD / "runs/seed17/fold0/outer/heads.pt"
    monkeypatch.setattr(runner, "sha", lambda p: "0" * 64 if runner.scoped(p) == target else original_sha(p))
    with pytest.raises(ValueError, match="checkpoint artifact changed"):
        runner.verify_baseline()


@pytest.mark.parametrize("changed", ["ledger", "source"])
def test_bridge_cannot_rebind_changed_validated_inputs(tmp_path, monkeypatch, changed):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    directory = tmp_path / "run"
    ledger = directory / "inputs/augmented_queries.jsonl"
    source = directory / "components/validated.json"
    runner.write(ledger, {"text": "Original request plus validated components"})
    runner.write(source, {"status": "validated"})
    runner.write(directory / "inputs/bridge_manifest.json", {
        "state": "complete", "requests": 186,
        "inputs": runner.ref(ledger), "sources": [runner.ref(source)]})
    assert runner.verify_bridge_inputs(directory) == ledger
    runner.write(ledger if changed == "ledger" else source, {"edited_after_validation": True})
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        runner.verify_bridge_inputs(directory)


def test_inference_adapter_replays_a_real_frozen_fold():
    """Real saved heads, original inputs; no training or new outcome access."""
    from pbpk_match_light_inputs import load_cache, visible_inputs
    cache, visible = load_cache(), visible_inputs()
    protocol = runner.read(runner.OLD / "protocol.json")
    index = protocol["folds"][0]["outer_test"][0]
    requested = runner.read(runner.ROOT / "experiments/model_recommendation/pbpk_match_v2/dataset.json").get("execution_requested_endpoints", {})
    result = runner.infer_fold(cache["light_query_embeddings"][0, [index]],
        visible["condition_features"][0, [index]], cache["light_candidate_embeddings"],
        [visible["query_records"][index]], [requested.get(str(index))],
        runner.OLD / "runs/seed17/fold0/outer", device="cpu")
    with np.load(runner.OLD / "predictions/seed17.npz", allow_pickle=False) as original:
        for key in ("scores", "base_scores", "probabilities"):
            np.testing.assert_allclose(result[key], original[key][0, [index]], rtol=2e-5, atol=2e-5)
        assert int(np.argmax(result["scores"][0])) == int(np.argmax(original["scores"][0, index]))
