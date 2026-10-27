"""Information-boundary and resampling checks for the sampling-policy runner."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "contextual_runner_under_review", ROOT / "scripts/run_contextual_sampling_policy.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def menus():
    designs, labels = [], {}
    # Two compounds share one study. Every compound has both layouts.
    for i, group in enumerate(("shared", "shared", "g1", "g2", "g3", "g4")):
        for regime in ("early_four", "late_four"):
            key = f"drug{i}__{regime}"
            x = np.zeros((3, 12))
            x[:, 0] = [0, 1, -1]
            x[:, 11] = [0, 1 + i / 10, -1 - i / 10]
            designs.append(dict(
                design_key=key, case_id=f"drug{i}", drug_study_group=group,
                regime=regime, candidate_ids=[2, 7, 9], times=[1., 2., 3.],
                X=x.tolist(), time_index=0,
            ))
            labels[key] = dict(cmax=[2., 1., 4.], curve=[2., 1.2, 3.])
    return designs, labels


def test_label_whitelist_rejects_whole_request_before_any_outcome_lookup():
    designs, _ = menus()

    class NoRead:
        def __getitem__(self, key):
            pytest.fail("An outcome was read before the complete whitelist check")

    # The authorized menu deliberately comes first.
    with pytest.raises(ValueError, match="Held-out label requested"):
        runner.labels_for([designs[0], designs[-1]], NoRead(), {"drug0"})


def test_label_lookup_uses_only_authorized_menu_and_exact_candidate_ids():
    designs, _ = menus()
    d = designs[0]
    seen = []

    class AuthorizedOnly:
        def __getitem__(self, key):
            assert key[0] == d["design_key"]
            assert key[1] in {d["design_key"] + f"__point_{i:02d}" for i in (2, 7, 9)}
            seen.append(key)
            return {"endpoints": {"cmax": {"error": 2.}}, "curve_metrics": {"gmfe": 3.}}

    result = runner.labels_for([d], AuthorizedOnly(), {"drug0"})
    assert len(seen) == 3
    assert result == {d["design_key"]: {"cmax": [2.] * 3, "curve": [3.] * 3}}


def test_inner_tuning_keeps_shared_study_and_both_layouts_out(monkeypatch):
    designs, labels = menus()
    by_x = {str(d["X"]): d["drug_study_group"] for d in designs}
    calls = []

    def train(train_designs, train_labels, *, dimensions, alpha):
        keys = {d["design_key"] for d in train_designs}
        assert set(train_labels) == keys
        groups = {d["drug_study_group"] for d in train_designs}
        for group in groups:
            assert {d["design_key"] for d in designs if d["drug_study_group"] == group} <= keys
        calls.append((groups, keys))
        return {"groups": groups}

    def choose(model, x, times, time_index):
        assert by_x[str(x)] not in model["groups"]
        return {"selected_index": 1}

    monkeypatch.setattr(runner.policy, "train", train)
    monkeypatch.setattr(runner.policy, "choose", choose)
    result = runner.tune(designs, labels, dimensions=12)
    assert len(calls) == 5 * len(runner.ALPHAS)
    assert len(result["trials"]) == len(runner.ALPHAS)
    assert result["alpha"] == max(runner.ALPHAS)  # Equal results: fixed stronger-ridge tie rule.


def test_outer_choices_and_tuning_are_invariant_to_heldout_outcome_changes(monkeypatch):
    designs, labels = menus()
    spec = dict(evaluation="synthetic", train_case_ids=[f"drug{i}" for i in range(5)],
                heldout_case_ids=["drug5"])
    protocol = dict(inputs="inputs", outcome_store="outcomes")
    saved, heldout_multiplier = {}, [1.]

    def checked(which):
        if which == "inputs":
            return {"designs": deepcopy(designs)}
        assert which == "outcomes"
        records = []
        for d in designs:
            for k, candidate_id in enumerate(d["candidate_ids"]):
                factor = heldout_multiplier[0] if d["case_id"] == "drug5" else 1.
                records.append(dict(
                    design_key=d["design_key"], candidate=d["design_key"] + f"__point_{candidate_id:02d}",
                    endpoints={"cmax": {"error": labels[d["design_key"]]["cmax"][k] * factor}},
                    curve_metrics={"gmfe": labels[d["design_key"]]["curve"][k] / factor},
                ))
        return {"records": records}

    monkeypatch.setattr(runner, "read", lambda path: protocol)
    monkeypatch.setattr(runner, "checked", checked)
    monkeypatch.setattr(runner, "guard", lambda p: None)
    monkeypatch.setattr(runner, "save", lambda path, value: saved.update(deepcopy(value)))
    monkeypatch.setattr(runner, "ref", lambda path: {"path": "not-written"})
    runner.fold_job("synthetic", spec)
    first = deepcopy(saved)
    # Poison held-out outcomes, including values that would fail fitting validation.
    heldout_multiplier[0] = 1e12
    runner.fold_job("synthetic", spec)
    for key in ("choices", "models", "tuning"):
        assert saved[key] == first[key]
    for dimension in ("8", "12"):
        assert "drug5" not in saved["models"][dimension]["full"]["train_case_ids"]
        assert "g4" not in saved["tuning"][dimension]["train_groups"]


def test_bootstrap_preserves_shared_study_pairs_and_excludes_missing_reference():
    rows = []
    # Each study has opposite paired log effects. Study resampling must keep
    # the pair together and therefore give a degenerate ratio=1 interval.
    for cid, group, ratio in (("a", "s1", .5), ("b", "s1", 2.),
                              ("c", "s2", .25), ("d", "s2", 4.)):
        rows.extend([
            dict(case_id=cid, method="time", study_group=group, errors={"cmax": 4.}),
            dict(case_id=cid, method="contextual", study_group=group, errors={"cmax": 4. * ratio}),
        ])
    rows.extend([
        dict(case_id="missing", method="time", study_group="s3", errors={"cmax": None}),
        dict(case_id="missing", method="contextual", study_group="s3", errors={"cmax": 1.}),
    ])
    result = runner.bootstrap(rows[::-1], "contextual", "cmax")
    assert result["n"] == 4 and result["clusters"] == 2
    assert result["improved"] == 2 and result["worse"] == 2
    np.testing.assert_allclose([result["ratio"], *result["ci95"]], 1., atol=1e-14)
