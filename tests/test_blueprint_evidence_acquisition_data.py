"""Acquisition isolation contracts using synthetic development data only."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "blueprint_acquisition_data_test", PROJECT / "scripts/blueprint_evidence_acquisition_data.py")
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


@pytest.fixture
def documents():
    request = {"compound_id": "synthetic-compound",
               "human": {"weight_kg": 70., "age_years": 30.},
               "dosing": {"route": "intravenous", "dose_mg": 10., "infusion_minutes": 30., "start_h": 0.},
               "evidence": [{"parameter": "molecular_weight", "value": 300., "unit": "g/mol"}]}
    full = {key: deepcopy(request[key]) for key in ("compound_id", "human", "dosing")}
    full.update(study_id="development-only", role=data.ROLE, matrix="plasma",
                time_unit="min", unit="umol/l", molecular_weight_g_mol=300.,
                observations=[{"time_min": float(i), "concentration_umol_per_l": 10. / i}
                              for i in range(1, 10)])
    five = deepcopy(full)
    five["observations"] = [deepcopy(full["observations"][i]) for i in (0, 2, 4, 6, 8)]
    return request, full, five


def test_menu_contains_only_times_and_stable_source_indices(documents):
    request, full, five = documents
    result = data.split_development(full, five, request)
    assert set(result) == {"visible_training", "design_only"}
    assert result["visible_training"] == five
    assert result["design_only"] == {
        "observed_times_h": [i / 60. for i in (1, 3, 5, 7, 9)],
        "candidate_times_h": [i / 60. for i in (2, 4, 6, 8)],
        "candidate_ids": [1, 3, 5, 7]}


def test_hidden_values_cannot_change_any_design_input(documents):
    request, full, five = documents
    original = data.split_development(full, five, request)
    for index in original["design_only"]["candidate_ids"]:
        full["observations"][index]["concentration_umol_per_l"] *= 1e5
    assert data.split_development(full, five, request) == original


def test_source_metadata_cannot_leak_hidden_concentrations(documents):
    request, full, five = documents
    for document in (full, five):
        document["raw_observations"] = deepcopy(full["observations"])
        document["target"] = {"observations": "not a design input"}
        document["selection"] = {"full_values": [100., 200.]}
        document["observations"][0]["source_curve"] = [999., 998.]
    split = data.split_development(full, five, request)
    visible = split["visible_training"]
    assert not {"raw_observations", "target", "selection"} & visible.keys()
    assert set(visible["observations"][0]) == {"time_min", "concentration_umol_per_l"}
    assert set(data.reveal_one(full, five, 2. / 60.)["observations"][0]) == {
        "time_min", "concentration_umol_per_l"}


def test_reveal_only_one_additional_row_and_preserve_inputs(documents):
    request, full, five = documents
    snapshots = deepcopy(documents)
    revealed = data.reveal_one(full, five, 4. / 60.)
    assert len(revealed["observations"]) == 6
    assert [row["time_min"] for row in revealed["observations"]] == [1., 3., 4., 5., 7., 9.]
    assert revealed["observations"][2] == full["observations"][3]
    assert documents == snapshots
    revealed["human"]["weight_kg"] = 1.
    revealed["observations"][0]["concentration_umol_per_l"] = 1.
    assert documents == snapshots


def test_nonselected_values_cannot_change_revealed_training(documents):
    request, full, five = documents
    original = data.reveal_one(full, five, 2. / 60.)
    for index in (3, 5, 7):
        full["observations"][index]["concentration_umol_per_l"] = 76543.
    assert data.reveal_one(full, five, 2. / 60.) == original


@pytest.mark.parametrize("t", [1./60., 3./60., 0., 20., float("nan"), float("inf"), -1., True, "0.5"])
def test_reveal_rejects_non_candidate_time(documents, t):
    _, full, five = documents
    with pytest.raises(ValueError):
        data.reveal_one(full, five, t)


@pytest.mark.parametrize("key,value", [
    ("compound_id", "another-compound"), ("study_id", "target-study"),
    ("human", {"weight_kg": 80., "age_years": 30.}),
    ("dosing", {"route": "intravenous", "dose_mg": 20., "infusion_minutes": 30., "start_h": 0.}),
    ("molecular_weight_g_mol", 301.), ("role", "target_observations"),
    ("matrix", "blood"), ("unit", "mg/l"), ("time_unit", "h"),
])
def test_rejects_mismatched_sources(documents, key, value):
    request, full, five = documents
    five[key] = value
    with pytest.raises(ValueError):
        data.split_development(full, five, request)
    with pytest.raises(ValueError):
        data.reveal_one(full, five, 2./60.)


@pytest.mark.parametrize("key", ["compound_id", "human", "dosing"])
def test_request_must_match_development(documents, key):
    request, full, five = documents
    request[key] = None
    with pytest.raises(ValueError, match="request"):
        data.split_development(full, five, request)


@pytest.mark.parametrize("field,value", [
    ("time_min", 1.0000001), ("concentration_umol_per_l", 10.00000001),
    ("unit", "mg/l"), ("time_unit", "h"),
    ("concentration_umol_per_l", float("nan")), ("concentration_umol_per_l", 0.),
    ("concentration_umol_per_l", True),
])
def test_visible_rows_require_exact_values_and_units(documents, field, value):
    request, full, five = documents
    five["observations"][0][field] = value
    with pytest.raises(ValueError):
        data.split_development(full, five, request)


def test_exactly_five_initial_points_and_at_least_one_candidate(documents):
    request, full, five = documents
    with pytest.raises(ValueError, match="Exactly five"):
        data.split_development(full, full, request)
    with pytest.raises(ValueError, match="hidden"):
        data.split_development(five, five, request)


def test_duplicate_or_unsorted_full_times_rejected(documents):
    request, full, five = documents
    full["observations"][1]["time_min"] = full["observations"][0]["time_min"]
    with pytest.raises(ValueError, match="strictly increasing"):
        data.split_development(full, five, request)


@pytest.mark.parametrize("mw", [None, [], [{"parameter": "molecular_weight", "value": 300., "unit": "kg/mol"}],
                               [{"parameter": "molecular_weight", "value": 301., "unit": "g/mol"}]])
def test_request_weight_units_required(documents, mw):
    request, full, five = documents
    request["evidence"] = mw
    with pytest.raises(ValueError):
        data.split_development(full, five, request)


def test_no_request_not_an_isolation_bypass(documents):
    _, full, five = documents
    with pytest.raises(ValueError, match="explicit development request"):
        data.split_development(full, five, None)


def test_visible_result_is_a_deep_copy(documents):
    request, full, five = documents
    snapshot = deepcopy(documents)
    result = data.split_development(full, five, request)
    result["visible_training"]["observations"][0]["time_min"] = 100.
    result["visible_training"]["human"]["weight_kg"] = 1.
    result["design_only"]["candidate_times_h"][0] = 100.
    assert documents == snapshot
