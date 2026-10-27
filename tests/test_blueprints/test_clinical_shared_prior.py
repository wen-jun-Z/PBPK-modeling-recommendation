"""Small real fits check cohort exclusion, fixed models and saved audit integrity."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import zipfile

import joblib
import pytest

from pbpk_kb.blueprint_learning import clinical_shared_prior as prior

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip("Use a project-local --basetemp")
    smiles = ["c1ccncc1", "c1ccccc1", "C1CCCCC1", "CC1CC1", "c1ccoc1", "C1CCC1",
              "C[C@H](O)Cl", "C[C@@H](O)Cl", "C1(C)CC1", "broken", "C1CCCCCC1",
              "c1ccc2ccccc2c1", "CC1CCCC1"]
    rows = [{"smiles_r": s, "human_CL_mL_min_kg": str(-.2 + i * .08),
             "human_VDss_L_kg": str(.1 + i * .03), "human_fup": str(.1 + i * .04),
             "human_thalf": str(.3 + i * .04), "human_mrt": str(.2 + i * .02)}
            for i, s in enumerate(smiles)]
    for i in (0, 1, 2, 8, 9):
        for column in rows[i]:
            if column != "smiles_r":
                rows[i][column] = "EXCLUDED_LABEL_MUST_NOT_BE_INTERPRETED"
    rows[4]["human_thalf"] = ""
    rows[10]["human_fup"] = "1.5"
    rows[11]["human_fup"] = "0"
    wheel = tmp_path / "training.whl"
    def write_rows():
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(prior.CSV_MEMBER, stream.getvalue())
    write_rows()
    monkeypatch.setattr(prior, "WHEEL", wheel)
    monkeypatch.setattr(prior, "EXPECTED_SOURCE_ROWS", len(rows))
    recipes = deepcopy(prior.rich.RECIPES)
    for recipe in recipes.values():
        if "n_estimators" in recipe:
            recipe["n_estimators"] = 3
    monkeypatch.setattr(prior.rich, "RECIPES", recipes)
    monkeypatch.setattr(prior.rich, "DESCRIPTORS", prior.rich.DESCRIPTORS[:3])
    monkeypatch.setattr(prior, "RF_RECIPE", {**prior.RF_RECIPE, "n_estimators": 3})
    original_label = prior._label
    def guard_label(value, endpoint):
        assert value != "EXCLUDED_LABEL_MUST_NOT_BE_INTERPRETED"
        return original_label(value, endpoint)
    monkeypatch.setattr(prior, "_label", guard_label)
    class StructureOnly(dict):
        def get(self, key, *args):
            assert key in {"compound_id", "smiles"}, "Target observations must not be accessed"
            return super().get(key, *args)
    targets = [StructureOnly(compound_id="target", smiles="c1ccncc1", auc="FORBIDDEN"),
               StructureOnly(compound_id="unseen", smiles="C1CCCCCCC1", evaluation="DO_NOT_OPEN.json")]
    return {"targets": targets, "output_dir": tmp_path / "output", "project_root": ROOT,
            "device": "cpu", "n_jobs": 1}, rows, write_rows


def test_parent_identity_preserves_stereo_and_generic_scaffold_excludes_heteroatoms():
    left = prior.normalize_structure("C[C@H](O)Cl")
    right = prior.normalize_structure("C[C@@H](O)Cl")
    assert left["identity_smiles"] != right["identity_smiles"]
    assert left["scaffold_group"] == right["scaffold_group"] == "ACYCLIC"
    groups = {prior.normalize_structure(s)["scaffold_group"] for s in
              ["c1ccncc1", "c1ccccc1", "C1CCCCC1", "Cc1ccccc1"]}
    assert len(groups) == 1
    assert prior.normalize_structure("CCN.Cl")["identity_smiles"] == prior.normalize_structure("CCN")["identity_smiles"]
    with pytest.raises(ValueError, match="Invalid SMILES"):
        prior.normalize_structure("broken")


def test_real_fit_accounts_for_every_raw_row_and_uses_same_training_for_rf(synthetic):
    args, rows, _ = synthetic
    events = []
    result = prior.fit_shared_prior(**args, progress=events.append)
    audit = result["audit"]
    assert audit["raw_row_count"] == len(rows) == sum(audit["row_disposition_counts"].values())
    assert audit["normalization_failure_count"] == 1
    assert audit["duplicate_parent_count"] == 1
    assert audit["target_identity_exclusion_count"] == 1
    assert audit["target_generic_scaffold_exclusion_count"] == 2
    assert audit["train_count"] == 8
    assert audit["rows"][6]["status"] == audit["rows"][7]["status"] == "train"
    assert audit["rows"][8]["representative_index"] == 3
    assert audit["label_exclusions"]["fraction_unbound"] == {
        "10": "fraction_outside_(0,1]", "11": "fraction_outside_(0,1]"}
    assert audit["label_exclusions"]["terminal_half_life"] == {"4": "missing"}
    assert not audit["target_observation_files_opened"] and not audit["target_labels_used"]
    assert set(result["parameters"]) == set(result["rf_parameters"]) == {"target", "unseen"}
    assert result["units"] == {"total_clearance": "mL/min/kg", "fraction_unbound": "fraction",
                               "vss": "L/kg", "terminal_half_life": "h", "mrt": "h"}
    for values in result["parameters"].values():
        assert set(values) == set(prior.PARAMETERS)
        assert all(isinstance(v, float) and v > 0 for v in values.values())
        assert values["fraction_unbound"] <= 1
    for parameter in prior.RF_PARAMETERS:
        assert audit["rf_shared_predictor"]["training_indices"][parameter] == audit["parameter_training_indices"][parameter]
    assert all(0 < values["fraction_unbound"] <= 1 for values in result["rf_parameters"].values())
    groups = {r["index"]: r.get("scaffold_group") for r in audit["rows"]}
    protected = {r["scaffold_group"] for r in audit["targets"].values()}
    for parameter, records in audit["consistency_oof"].items():
        coverage = []
        for fold in records["folds"].values():
            fit = {groups[i] for i in fold["train_indices"]}
            held = {groups[i] for i in fold["validation_indices"]}
            assert not fit & held and not (fit | held) & protected
            coverage.extend(fold["validation_indices"])
        assert sorted(coverage) == audit["parameter_training_indices"][parameter]
    assert len([e for e in events if e["stage"] == "consistency_oof"]) == 36
    assert len([e for e in events if e["stage"] == "final_fit"]) == 19
    assert events[-1]["stage"] == "completed"
    assert {p.name for p in args["output_dir"].iterdir()} == {"model.joblib", "predictions_audit.json"}
    assert json.loads(Path(result["audit_path"]).read_text()) == result
    saved = joblib.load(result["model_path"])
    assert set(saved["models"]["fraction_unbound"]) == {"rich_xgb_depth3"}
    assert set(saved["models"]["total_clearance"]) == set(prior.rich.ENSEMBLE_MEMBERS)
    assert set(saved["rf_shared_predictor"]) == set(prior.RF_PARAMETERS)
    projection = audit["consistency_projection"]
    assert not projection["uses_calibration_or_target_labels"]
    if projection["enabled"]:
        for values in result["parameters"].values():
            assert values["vss"] == pytest.approx(.06 * values["total_clearance"] * values["mrt"])


def test_complete_result_reuses_without_fitting_and_rejects_tampering(synthetic, monkeypatch):
    args, _, _ = synthetic
    result = prior.fit_shared_prior(**args)
    def forbid_fit(*args, **kwargs):
        raise AssertionError("A completed unchanged prior was fitted again")
    monkeypatch.setattr(prior, "_fit", forbid_fit)
    assert prior.fit_shared_prior(**{**args, "targets": list(reversed(args["targets"]))}) == result
    with pytest.raises(ValueError, match="changed"):
        prior.fit_shared_prior(**{**args, "targets": [{"compound_id": "different", "smiles": "C1CCCCCCC1"}]})
    path = Path(result["audit_path"])
    modified = json.loads(path.read_text())
    modified["parameters"]["target"]["vss"] *= 2
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="changed"):
        prior.fit_shared_prior(**args)


def test_excluded_labels_have_no_effect_on_either_predictor(synthetic):
    args, rows, write_rows = synthetic
    first = prior.fit_shared_prior(**args)
    for i in (0, 1, 2, 8, 9):
        for column in rows[i]:
            if column != "smiles_r":
                rows[i][column] = "99999999"
    write_rows()
    second = prior.fit_shared_prior(**{**args, "output_dir": args["output_dir"].with_name("changed_excluded_labels")})
    assert first["parameters"] == second["parameters"]
    assert first["rf_parameters"] == second["rf_parameters"]
    assert first["audit"]["consistency_projection"] == second["audit"]["consistency_projection"]
    assert first["input_fingerprint"] != second["input_fingerprint"]


def test_invalid_targets_row_count_and_external_output_fail_before_training(synthetic, monkeypatch):
    args, _, _ = synthetic
    with pytest.raises(ValueError, match="inside this project"):
        prior.fit_shared_prior(**{**args, "output_dir": ROOT.parent / "outside"})
    with pytest.raises(ValueError, match="unique"):
        prior.fit_shared_prior(**{**args, "targets": [args["targets"][0]] * 2})
    with pytest.raises(ValueError, match="target structure"):
        prior.fit_shared_prior(**{**args, "targets": []})
    monkeypatch.setattr(prior, "EXPECTED_SOURCE_ROWS", 1283)
    with pytest.raises(ValueError, match="Expected 1283"):
        prior.fit_shared_prior(**args)
    assert not args["output_dir"].exists()
