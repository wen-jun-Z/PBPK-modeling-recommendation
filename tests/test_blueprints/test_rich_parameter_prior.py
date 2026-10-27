"""Small real models verify split isolation, six-candidate selection and resume."""
from copy import deepcopy
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.blueprint_learning import rich_parameter_prior as rich

ROOT = Path(__file__).resolve().parents[2]
SMILES = ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "CCc1ccncc1", "C1CCCCC1", "C1CCCC1", "C1CCC1", "C1CC1",
          "c1ccoc1", "c1ccsc1", "c1ncc[nH]1", "c1ncncc1", "C1COCCN1", "C1CCNCC1", "C1CCOC1", "C1COC1",
          "c1ccc2ccccc2c1", "O=C1CCCCC1", "O=C1NCCC1", "C1CC2CCC1C2", "c1ccc2[nH]ccc2c1", "O=C1OC=CC=C1", "C1CN1", "C1CO1"]


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip("Use project-local --basetemp")
    data = [{"smiles_r": s, "human_CL_mL_min_kg": -.4 + (i % 7) * .2,
             "human_VDss_L_kg": -.2 + (i % 5) * .3, "human_fup": .05 + (i % 8) * .08,
             "human_mrt": .3 + i * .02, "human_thalf": .4 + i * .03} for i, s in enumerate(SMILES)]
    data[6]["human_fup"] = ""
    data[7]["human_thalf"] = ""
    source = tmp_path / "data.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["smiles_r", *rich.base_prior.base.ENDPOINTS])
        writer.writeheader(); writer.writerows(data)
    dataset = rich.base_prior.base.load_parameter_dataset(source)
    split = rich.base_prior.base.make_scaffold_split(dataset, exclude_smiles={"target_a": SMILES[0], "target_b": SMILES[2]},
                                                    train_limit=24, test_limit=3, seed=20260930)
    audit = {"source": dataset.provenance, "source_pins": {str(source.relative_to(ROOT)): rich.base_prior._sha(source)},
        "external_exclusions": split["external_exclusions"], "cohort_count": 2, "train_indices": split["train_indices"],
        "calibration_indices": split["test_indices"], "train_count": split["train_count"], "calibration_count": split["test_count"],
        "rows": [{**r, "status": "calibration" if r["status"] == "test" else r["status"]} for r in split["rows"]],
        "frozen_input": {"seed": 20260930}}
    source_run = tmp_path / "source_run"
    (source_run / "priors").mkdir(parents=True)
    (source_run / "priors/audit.json").write_text(json.dumps(audit))
    (source_run / "cohort.json").write_text(json.dumps({"cases": [{"compound_id": cid, "identity_smiles": identity["identity_smiles"],
          "evaluation": {"path": "DO_NOT_READ_TARGET_EVALUATION.json"}} for cid, identity in split["external_exclusions"].items()]}))
    recipes = deepcopy(rich.RECIPES)
    for recipe in recipes.values():
        if "n_estimators" in recipe:
            recipe["n_estimators"] = 3
    monkeypatch.setattr(rich, "RECIPES", recipes)
    monkeypatch.setattr(rich, "DESCRIPTORS", rich.DESCRIPTORS[:5])
    args = {"project_root": ROOT, "output_dir": tmp_path / "output", "source_run": source_run,
            "wheel": source, "n_jobs": 1, "device": "cpu"}
    return args, dataset, audit


def test_full_features_include_complete_2d_morgan_and_maccs_without_labels():
    x, audit = rich.rich_features(["CCO", "c1ccccc1"])
    assert x.shape == (2, 1191 + len(rich.DESCRIPTORS))
    assert len(rich.DESCRIPTORS) > 200
    assert np.isin(x[:, :1191], [0, 1]).all()
    assert audit["descriptor_names"] == [name for name, _ in rich.DESCRIPTORS]


def test_fold_imputation_and_scaling_do_not_inspect_validation_extremes():
    model = rich.make_estimator("rich_svr", 1, 1, "cpu")
    fit = np.array([[1., 2., np.nan], [3., np.nan, np.nan], [5., 8., np.nan]])
    model.fit(fit, [0, 1, 2])
    assert model.named_steps["impute"].statistics_ == pytest.approx([3, 5, 0])
    assert model.named_steps["scale"].mean_ == pytest.approx([3, 5, 0])
    model.predict([[1e10, 1e10, 1e10]])
    assert model.named_steps["scale"].mean_ == pytest.approx([3, 5, 0])


def test_six_predeclared_candidates_and_geometric_ensemble_use_only_oof_logs():
    truth = np.array([0., 1.])
    oof = {name: truth + (1 if j % 2 else -1) for j, name in enumerate(rich.BASE_CANDIDATES)}
    selected, scores, values = rich.candidate_scores(oof, truth, "total_clearance")
    assert len(scores) == 6 and selected == "rich_geometric_ensemble"
    assert values[selected] == pytest.approx(truth)
    assert scores[selected] == pytest.approx(0)


def test_fu_bound_is_applied_to_oof_and_every_prediction():
    assert rich.physical_log_predictions([1., -2.], "fraction_unbound") == pytest.approx([0, -2])
    assert rich.physical_log_predictions([1., -2.], "vss") == pytest.approx([1, -2])
    with pytest.raises(ValueError, match="Nonfinite"):
        rich.physical_log_predictions([float("nan")], "vss")


def test_real_small_forests_svr_xgb_preserve_splits_missing_labels_and_complete_lock(synthetic, monkeypatch):
    args, dataset, baseline = synthetic
    actual_load = rich.base_prior.base.load_parameter_dataset
    excluded = set(range(len(dataset.rows))) - set(baseline["train_indices"]) - set(baseline["calibration_indices"])
    class Forbidden(dict):
        def __getitem__(self, key):
            raise AssertionError("Held-out cohort or scaffold-neighbor label read")
    def guard_load(path):
        result = actual_load(path)
        for i in excluded:
            result.rows[i]["labels_log10"] = Forbidden(result.rows[i]["labels_log10"])
        return result
    monkeypatch.setattr(rich.base_prior.base, "load_parameter_dataset", guard_load)
    fitted = rich.fit_rich_prior(**args)
    audit = rich._read_audit(args["output_dir"] / "audit.json")
    assert audit["train_indices"] == baseline["train_indices"]
    assert audit["calibration_indices"] == baseline["calibration_indices"]
    assert not audit["target_observation_files_opened"] and not audit["calibration_labels_used_for_selection"]
    assert len(fitted["predictions"]) == 2 and audit["target_predictions_locked_before_scoring"]
    assert set(p.name for p in args["output_dir"].iterdir()) == {"model.joblib", "audit.json", "predictions.json", "status.json"}
    for parameter, rows in audit["label_audit"].items():
        assert not set(rows["train_missing_indices"]) & set(rows["train_indices"])
        for method, folds in audit["cv_progress"][parameter].items():
            for fold in folds.values():
                fit = {dataset.rows[i]["scaffold_group"] for i in fold["train_indices"]}
                held = {dataset.rows[i]["scaffold_group"] for i in fold["validation_indices"]}
                assert not fit & held
                assert not set(fold["train_indices"] + fold["validation_indices"]) & set(baseline["calibration_indices"])
        assert audit["selection"][parameter]["selected_log10_mae"] <= audit["selection"][parameter]["old_extra_trees_log10_mae"]
    for pred in fitted["predictions"].values():
        assert not pred["cohort_labels_used"]
        for branch in pred["methods"].values():
            assert set(branch["predict_nominal"]) == set(rich.PARAMETERS)
            assert 0 < branch["predict_nominal"]["fraction_unbound"] <= 1
    def forbidden_fit(*a, **k):
        raise AssertionError("Completed run retrained instead of reusing")
    monkeypatch.setattr(rich, "make_estimator", forbidden_fit)
    assert rich.fit_rich_prior(**args) == fitted
    # Target prediction tampering cannot silently become a legitimate resume.
    path = args["output_dir"] / "predictions.json"
    changed = json.loads(path.read_text())
    changed["predictions"]["target_a"]["methods"]["selected"]["predict_nominal"]["vss"] *= 2
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="predictions changed"):
        rich.fit_rich_prior(**args)


def test_resume_reuses_completed_oof_folds_after_interruption(synthetic, monkeypatch):
    args, _, _ = synthetic
    factory = rich.make_estimator
    calls = []
    def interrupt(method, *a, **k):
        calls.append(method)
        if len(calls) == 2:
            raise RuntimeError("Injected interruption after first complete fold")
        return factory(method, *a, **k)
    monkeypatch.setattr(rich, "make_estimator", interrupt)
    with pytest.raises(RuntimeError, match="Injected"):
        rich.fit_rich_prior(**args)
    original = rich._read_audit(args["output_dir"] / "audit.json")["cv_progress"]["total_clearance"]["old_extra_trees"]["0"]
    monkeypatch.setattr(rich, "make_estimator", factory)
    rich.fit_rich_prior(**args)
    resumed = rich._read_audit(args["output_dir"] / "audit.json")["cv_progress"]["total_clearance"]["old_extra_trees"]["0"]
    assert resumed == original


def test_source_mutation_and_foreign_files_cannot_be_overwritten(synthetic):
    args, _, _ = synthetic
    args["wheel"].write_text(args["wheel"].read_text() + "\n")
    with pytest.raises(ValueError, match="source changed"):
        rich.fit_rich_prior(**args)


def test_consistency_projection_respects_units_and_changes_uncertain_parameters_more():
    source = np.array([[math.log10(10), math.log10(1), math.log10(5)]])
    projected = rich.project_consistency(source, [1, 4, 1])
    actual = 10. ** projected[0]
    assert actual[1] == pytest.approx(.06 * actual[0] * actual[2])
    delta = np.abs(projected - source)[0]
    assert delta[1] == pytest.approx(4 * delta[0])
    with pytest.raises(ValueError, match="positive"):
        rich.project_consistency(source, [0, 1, 1])


def projection_audit():
    indices = list(range(6))
    truth = np.array([[.1*i, .1*i+.2+math.log10(.06), .2] for i in indices])
    predicted = truth + (.2 + .03*np.arange(6))[:,None] * np.array([-1,1,-1])[None,:]
    folds = [{"fold":f,"validation_indices":[2*f,2*f+1]} for f in range(3)]
    audit = {"folds": folds, "selection": {}, "cv_progress": {}}
    for j,p in enumerate(rich.CONSISTENCY_PARAMETERS):
        audit["selection"][p] = {"oof_indices": indices,"selected_oof_log10":predicted[:,j].tolist()}
        audit["cv_progress"][p] = {"old_extra_trees": {str(f["fold"]): {"out_of_fold": [
            {"row_index":i,"observed_log10":float(truth[i,j])} for i in f["validation_indices"]]} for f in folds}}
    return audit


def test_projection_weights_exclude_own_entire_fold_and_enable_only_guarded_improvement():
    audit = projection_audit()
    original = rich.crossfit_consistency(audit)
    assert original["enabled"] and original["corrected_endpoint_equal_log10_mae"] < 1e-12
    for fold in original["folds"]:
        assert not set(fold["held_indices"]) & set(fold["weight_fit_indices"])
    changed = deepcopy(audit)
    changed["cv_progress"]["total_clearance"]["old_extra_trees"]["0"]["out_of_fold"][0]["observed_log10"] += 2
    altered = rich.crossfit_consistency(changed)
    assert altered["folds"][0]["residual_variances"] == original["folds"][0]["residual_variances"]
    assert not altered["enabled"]


def test_calibration_label_changes_do_not_change_selection_or_target_nominals(synthetic):
    args, _, baseline = synthetic
    first = rich.fit_rich_prior(**args)
    old_audit = rich._read_audit(args["output_dir"] / "audit.json")
    with args["wheel"].open() as stream:
        rows = list(csv.DictReader(stream))
    for i in baseline["calibration_indices"]:
        for label in rich.base_prior.base.ENDPOINTS:
            if rows[i][label] != "":
                rows[i][label] = float(rows[i][label]) * .1 if label == "human_fup" else float(rows[i][label]) + .8
    source = args["wheel"].with_name("different_calibration.csv")
    with source.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=["smiles_r",*rich.base_prior.base.ENDPOINTS])
        writer.writeheader();writer.writerows(rows)
    audit = deepcopy(baseline)
    audit["source"] = rich.base_prior.base.load_parameter_dataset(source).provenance
    audit["source_pins"] = {str(source.relative_to(ROOT)):rich.base_prior._sha(source)}
    source_run = args["source_run"].with_name("different_source_run")
    (source_run/"priors").mkdir(parents=True)
    (source_run/"priors/audit.json").write_text(json.dumps(audit))
    (source_run/"cohort.json").write_text((args["source_run"]/"cohort.json").read_text())
    output = args["output_dir"].with_name("different_output")
    second = rich.fit_rich_prior(**{**args,"wheel":source,"source_run":source_run,"output_dir":output})
    new_audit = rich._read_audit(output/"audit.json")
    assert new_audit["selection"] == old_audit["selection"]
    assert new_audit["consistency_projection"] == old_audit["consistency_projection"]
    assert new_audit["calibration"] != old_audit["calibration"]
    for cid in first["predictions"]:
        for branch in ("baseline","selected"):
            assert first["predictions"][cid]["methods"][branch]["predict_nominal"] == second["predictions"][cid]["methods"][branch]["predict_nominal"]
