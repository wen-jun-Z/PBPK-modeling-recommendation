"""Scientific-contract tests for new source curation, without native fitting."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sampling_expansion_cohort", ROOT / "scripts/prepare_sampling_expansion_cohort.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_rows_require_distinct_positive_times_and_positive_values():
    rows = [{"time_min": float(t), "concentration_umol_per_l": 1.} for t in range(1, 6)]
    assert len(m.validate_rows(rows)) == 5
    with pytest.raises(ValueError, match="Duplicate"):
        m.validate_rows(rows + [rows[0]])
    with pytest.raises(ValueError, match="Insufficient"):
        m.validate_rows([dict(r, concentration_umol_per_l=0.) if i == 0 else r for i, r in enumerate(rows)])


def fixture_archive():
    s = m.SPECS["Montelukast"]
    dev_sim, dev_obs = s["development"]
    return {"ObservedData": [{"Name": dev_obs, "ExtendedProperties": [
                {"Name": "Species", "Value": "Human"}, {"Name": "Compartment", "Value": "Plasma"},
                {"Name": "Study Id", "Value": "Cheng1996"}],
                "BaseGrid": {"Unit": "h", "Values": [.1, .2, .5, 1., 2., 4.]},
                "Columns": [{"Unit": "mg/l", "Values": [2., 3., 2., 1., .5, .1],
                             "DataInfo": {"Origin": "Observation"}}]}],
            "Individuals": [{"Name": "human", "OriginData": {"Species": "Human", "Gender": "MALE", "Age": {"Value": 30.}}}],
            "Protocols": [{"Name": "iv", "ApplicationType": "Intravenous", "DosingInterval": "Single",
                "Parameters": [{"Name": "Start time", "Value": 0., "Unit": "h"},
                               {"Name": "InputDose", "Value": 3., "Unit": "mg"},
                               {"Name": "Infusion time", "Value": .25, "Unit": "h"}]}],
            "Compounds": [{"Name": "DO_NOT_EXPORT", "FittedClearance": 999.}],
            "Simulations": [{"Name": dev_sim, "Individual": "human", "ObservedData": [dev_obs],
                "Compounds": [{"Name": "Montelukast", "Protocol": {"Name": "iv"},
                               "Processes": [{"FittedRate": 999.}]}]}]}


def fixture_setup(monkeypatch, archive):
    source = {"compound_names": ["Montelukast"], "path": m.SPECS["Montelukast"]["archive"], "sha256": "a"*64}
    monkeypatch.setattr(m, "checked", lambda s: archive)
    monkeypatch.setattr(m, "ref", lambda p, location="$", **kw: {"path": str(p), "sha256": "a"*64, "location": location, **kw})
    return {"pubchem_cid": 5281040, "molecular_weight_g_mol": 500.}, {"entries": [source]}


def test_only_observations_and_nominal_protocol_exported(monkeypatch):
    archive = fixture_archive()
    identity, inventory = fixture_setup(monkeypatch, archive)
    first = m.archive_curve("Montelukast", "development", identity, inventory)
    archive["Compounds"][0]["FittedClearance"] = -1000.
    archive["Simulations"][0]["Compounds"][0]["Processes"][0]["FittedRate"] = -1000.
    assert first == m.archive_curve("Montelukast", "development", identity, inventory)
    assert first["dosing"]["dose_mg"] == 3.
    assert first["dosing"]["infusion_minutes"] == 15.
    assert first["observations"][0] == {"time_min": 6., "concentration_umol_per_l": 4.}


def test_different_arms_are_not_called_independent_studies():
    for name in ("Montelukast", "Finerenone"):
        assert m.SPECS[name]["cross_study"] is False
        assert m.SPECS[name]["cluster_id"]
        assert any("within_study" in f for f in m.SPECS[name]["flags"])
    assert "Vancomycin" not in m.SPECS
    assert "Propofol" not in m.SPECS


def test_mismatched_observation_link_is_rejected(monkeypatch):
    archive = fixture_archive()
    identity, inventory = fixture_setup(monkeypatch, archive)
    archive["Simulations"][0]["ObservedData"] = []
    with pytest.raises(ValueError, match="not linked"):
        m.archive_curve("Montelukast", "development", identity, inventory)


def test_repeated_dosing_is_not_silently_reduced(monkeypatch):
    archive = fixture_archive()
    identity, inventory = fixture_setup(monkeypatch, archive)
    archive["Protocols"][0]["DosingInterval"] = "DI_12_12"
    with pytest.raises(ValueError, match="repeated-dose"):
        m.archive_curve("Montelukast", "development", identity, inventory)


def test_clinical_matrix_not_guessed(monkeypatch):
    archive = fixture_archive()
    identity, inventory = fixture_setup(monkeypatch, archive)
    archive["ObservedData"][0]["ExtendedProperties"][1]["Value"] = "Whole Blood"
    with pytest.raises(ValueError, match="Human plasma"):
        m.archive_curve("Montelukast", "development", identity, inventory)
