"""Actual small forests exercise train-only selection and frozen exclusions."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from pbpk_kb.blueprint_learning import adaptive_parameter_prior as prior

ROOT = Path(__file__).resolve().parents[2]
SMILES = [
    "c1ccccc1", "Cc1ccccc1", "c1ccncc1", "CCc1ccncc1",
    "C1CCCCC1", "C1CCCC1", "C1CCC1", "C1CC1", "c1ccoc1", "c1ccsc1",
    "c1ncc[nH]1", "c1ncncc1", "C1COCCN1", "C1CCNCC1", "C1CCOC1", "C1COC1",
    "c1ccc2ccccc2c1", "O=C1CCCCC1", "O=C1NCCC1", "C1CC2CCC1C2",
    "c1ccc2[nH]ccc2c1", "O=C1OC=CC=C1", "C1CN1", "C1CO1",
]


@pytest.fixture
def synthetic(tmp_path):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip("Use --basetemp inside the project runtime directory")
    rows = [{"smiles_r": smiles, "human_CL_mL_min_kg": -.4 + (i % 7)*.2,
             "human_VDss_L_kg": -.2 + (i % 5)*.3, "human_fup": .05 + (i % 8)*.08,
             "human_mrt": .3 + i*.02, "human_thalf": .4 + i*.03} for i, smiles in enumerate(SMILES)]
    rows[6]["human_fup"] = ""  # Endpoint missingness does not change scaffold membership.
    def write_dataset(name, data):
        path = tmp_path / (name + ".csv")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["smiles_r", *prior.base.ENDPOINTS])
            writer.writeheader(); writer.writerows(data)
        return path
    source = write_dataset("data", rows)
    dataset = prior.base.load_parameter_dataset(source)
    exclusions = {"target_a": SMILES[0], "target_b": SMILES[2]}
    split = prior.base.make_scaffold_split(dataset, exclude_smiles=exclusions,
                                          train_limit=len(rows), test_limit=3, seed=20260930)
    learned = prior.base.fit_parameter_pilot(dataset, split, n_jobs=1)
    model = tmp_path / "baseline.joblib"
    joblib.dump(learned.models, model)
    audited = deepcopy(split["rows"])
    for row in audited:
        if row["status"] == "test":
            row["status"] = "calibration"
    audit = {"source": dataset.provenance,
             "source_pins": {str(source.relative_to(ROOT)): prior._sha(source)},
             "external_exclusions": split["external_exclusions"], "cohort_count": 2,
             "train_indices": split["train_indices"], "calibration_indices": split["test_indices"],
             "train_count": split["train_count"], "calibration_count": split["test_count"],
             "rows": audited, "frozen_input": {"seed": 20260930},
             "forest_parameters": learned.audit["forest_parameters"], "model": prior._ref(model, ROOT)}
    audit_path = tmp_path / "baseline_audit.json"
    audit_path.write_text(json.dumps(audit))
    manifest = {"cases": [{"case_id": cid, "compound_id": cid,
        "identity_smiles": prior.base.normalize_structure(smiles)["identity_smiles"],
        "evaluation": {"path": str(tmp_path / "DO_NOT_READ_heldout_evaluation.json"), "sha256": "0" * 64}}
        for cid, smiles in exclusions.items()]}
    manifest_path = tmp_path / "cohort.json"
    manifest_path.write_text(json.dumps(manifest))
    args = {"cohort_manifest": manifest_path, "source_baseline_audit": audit_path,
            "output_dir": tmp_path / "fitted", "project_root": ROOT, "n_jobs": 1}
    return rows, source, audit, args, write_dataset


def test_real_forests_preserve_baseline_and_scaffold_exclusion_with_three_artifacts(synthetic):
    _, source, baseline, args, _ = synthetic
    result = prior.fit_adaptive_prior(source, **args)
    audit = result["audit"]
    assert audit["train_indices"] == baseline["train_indices"]
    assert audit["calibration_indices"] == baseline["calibration_indices"]
    assert all(r["status"] == "verified" for r in audit["rf_baseline_equivalence"].values())
    assert not audit["cohort_evaluation_files_opened"]
    assert not audit["selection_uses_calibration_labels"] and not audit["selection_uses_cohort_labels"]
    dataset = prior.base.load_parameter_dataset(source)
    protected = {v["scaffold_group"] for v in audit["external_exclusions"].values()}
    for fold in audit["folds"]:
        fit, validation = fold["train_indices"], fold["validation_indices"]
        fit_scaffolds = {dataset.rows[i]["scaffold_group"] for i in fit}
        validation_scaffolds = {dataset.rows[i]["scaffold_group"] for i in validation}
        assert not fit_scaffolds & validation_scaffolds
        assert not (fit_scaffolds | validation_scaffolds) & protected
        assert not set(fit + validation) & set(audit["calibration_indices"])
    for p, candidates in audit["cross_validation"].items():
        for candidate in candidates.values():
            rows = candidate["out_of_fold"]
            assert len(rows) == audit["label_audit"][p]["train_count"]
            assert candidate["mean_absolute_log10_error"] == pytest.approx(np.mean([r["absolute_log10_error"] for r in rows]))
        assert audit["selection"][p]["cv_mae_improvement_vs_rf"] >= 0
    models = joblib.load(args["output_dir"] / "model.joblib")
    assert models["rf"]["vss"].n_estimators == 200
    assert models["rf"]["vss"].min_samples_leaf == 3
    for record in result["predictions"].values():
        assert set(record["methods"]) == {"rf", "selected"}
        for branch in record["methods"].values():
            assert set(branch["predict_nominal"]) == set(prior.PARAMETERS)
            for p, ranges in branch["ranges"].items():
                narrow, wide = ranges["80"], ranges["90"]
                assert 0 < wide["lower"] <= narrow["lower"] <= narrow["central"] <= narrow["upper"] <= wide["upper"]
                assert not wide["clinical_confidence_interval"]
                assert wide["unit"] == prior.PARAMETERS[p][1]
        assert not record["cohort_labels_used"]
    assert {p.name for p in args["output_dir"].iterdir()} == {"model.joblib", "audit.json", "predictions.json"}
    reused = prior.fit_adaptive_prior(source, **args)
    assert reused["reused"] and reused["predictions"] == result["predictions"]
    path = args["output_dir"] / "predictions.json"
    bundle = json.loads(path.read_text())
    bundle["predictions"]["target_a"]["methods"]["rf"]["predict_nominal"]["vss"] *= 2
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="predictions changed"):
        prior.fit_adaptive_prior(source, **args)


def test_calibration_and_excluded_labels_cannot_change_selection_or_nominal_predictions(synthetic):
    rows, source, baseline, args, write = synthetic
    original = prior.fit_adaptive_prior(source, **args)
    changed = deepcopy(rows)
    for index in baseline["calibration_indices"]:
        changed[index]["human_CL_mL_min_kg"] += 1.5
        changed[index]["human_VDss_L_kg"] += 1.1
        changed[index]["human_fup"] = .0001
    # Target molecules and same-scaffold neighbours are excluded altogether.
    for index in range(4):
        changed[index]["human_CL_mL_min_kg"] = 3.
        changed[index]["human_VDss_L_kg"] = 4.
        changed[index]["human_fup"] = .00001
    modified_source = write("altered_labels", changed)
    modified_audit = deepcopy(baseline)
    modified_audit["source"] = prior.base.load_parameter_dataset(modified_source).provenance
    modified_audit["source_pins"] = {str(modified_source.relative_to(ROOT)): prior._sha(modified_source)}
    changed_audit_path = args["source_baseline_audit"].with_name("altered_audit.json")
    changed_audit_path.write_text(json.dumps(modified_audit))
    altered = prior.fit_adaptive_prior(modified_source, **{**args, "source_baseline_audit": changed_audit_path,
                                                          "output_dir": args["output_dir"].with_name("altered")})
    assert original["audit"]["selection"] == altered["audit"]["selection"]
    assert original["audit"]["cross_validation"] == altered["audit"]["cross_validation"]
    assert original["audit"]["calibration"] != altered["audit"]["calibration"]
    for cid in original["predictions"]:
        for method in ("rf", "selected"):
            assert original["predictions"][cid]["methods"][method]["predict_nominal"] == altered["predictions"][cid]["methods"][method]["predict_nominal"]


def test_excluded_target_and_neighbour_labels_cannot_change_even_stress_ranges(synthetic):
    rows, source, baseline, args, write = synthetic
    original = prior.fit_adaptive_prior(source, **args)
    changed = deepcopy(rows)
    for index in range(4):
        changed[index]["human_CL_mL_min_kg"] = 4.
        changed[index]["human_VDss_L_kg"] = 4.
        changed[index]["human_fup"] = .000001
    source2 = write("target_labels_changed", changed)
    baseline["source"] = prior.base.load_parameter_dataset(source2).provenance
    baseline["source_pins"] = {str(source2.relative_to(ROOT)): prior._sha(source2)}
    args["source_baseline_audit"].write_text(json.dumps(baseline))
    result = prior.fit_adaptive_prior(source2, **{**args, "output_dir": args["output_dir"].with_name("target_changed")})
    assert result["predictions"] == original["predictions"]
    assert result["audit"]["selection"] == original["audit"]["selection"]
    assert result["audit"]["calibration"] == original["audit"]["calibration"]


@pytest.mark.parametrize("kind", ["target", "scaffold", "overlap", "source_hash", "cohort_changed"])
def test_rejects_leakage_or_changed_frozen_inputs_before_fit(synthetic, monkeypatch, kind):
    _, source, baseline, args, _ = synthetic
    def forbidden(*a, **k):
        raise AssertionError("Model fit started before input validation")
    monkeypatch.setattr(prior, "_forest", forbidden)
    if kind in {"target", "scaffold"}:
        added = 0 if kind == "target" else 1
        old = baseline["train_indices"][0]
        baseline["train_indices"][0] = added
        baseline["rows"][old]["status"] = "unused"
        baseline["rows"][added]["status"] = "train"
    elif kind == "overlap":
        baseline["calibration_indices"][0] = baseline["train_indices"][0]
    elif kind == "source_hash":
        baseline["source"]["source_sha256"] = "0" * 64
    else:
        manifest = json.loads(args["cohort_manifest"].read_text())
        manifest["cases"][0]["identity_smiles"] = "C1CCCCC1"
        args["cohort_manifest"].write_text(json.dumps(manifest))
    args["source_baseline_audit"].write_text(json.dumps(baseline))
    with pytest.raises(ValueError):
        prior.fit_adaptive_prior(source, **args)


def test_range_zero_residual_bounds_and_units_are_explicit():
    residuals, quantiles = prior._residuals([-1., 0., 1.], [-1., 0., 1.])
    assert not residuals.any() and quantiles == {"80": 0., "90": 0.}
    zero = prior._stress(-.5, 0., "vss", 3, "80")
    assert zero["degenerate"] and zero["unit"] == "l/kg"
    bounded = prior._stress(-.1, 1., "fraction_unbound", 20, "90")
    assert bounded["upper"] == 1. and bounded["upper_clipped_to_one"]
    assert bounded["upper_before_physical_bound"] > 1
    clearance = prior._stress(-1., 2., "total_clearance", 20, "90")
    assert clearance["lower"] == .001 and not clearance["renal_clearance_used_for_clipping"]
    for truth, prediction in [([], []), ([0.], [float("nan")]), ([0., 1.], [0.])]:
        with pytest.raises(ValueError):
            prior._residuals(truth, prediction)
    with pytest.raises(ValueError):
        prior._stress(.1, 1., "fraction_unbound", 2, "90")


def test_existing_incomplete_output_is_retained(synthetic):
    _, source, _, args, _ = synthetic
    args["output_dir"].mkdir()
    marker = args["output_dir"] / "model.joblib"
    marker.write_bytes(b"interrupted artifact")
    with pytest.raises(FileExistsError, match="Incomplete"):
        prior.fit_adaptive_prior(source, **args)
    assert marker.read_bytes() == b"interrupted artifact"


@pytest.mark.parametrize("jobs", [0, 5, True])
def test_parallel_worker_bound(synthetic, jobs):
    _, source, _, args, _ = synthetic
    with pytest.raises(ValueError, match="n_jobs"):
        prior.fit_adaptive_prior(source, **{**args, "n_jobs": jobs})
