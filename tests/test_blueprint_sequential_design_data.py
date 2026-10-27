"""Time-only layouts, vault isolation and sequential budget contracts."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import blueprint_sequential_design_data as data


@pytest.fixture
def inputs():
    request = {"compound_id": "synthetic",
               "human": {"weight_kg": 70., "age_years": 30., "sex": "MALE"},
               "dosing": {"route": "intravenous", "dose_mg": 10., "infusion_minutes": 30., "start_h": 0.},
               "evidence": [{"parameter": "molecular_weight", "value": 300., "unit": "g/mol"}]}
    document = {key: deepcopy(request[key]) for key in ("compound_id", "human", "dosing")}
    document.update(study_id="development-study", role="calibration_data_reconstruction",
                    matrix="plasma", unit="umol/l", time_unit="min", molecular_weight_g_mol=300.,
                    observations=[{"time_min": float(i), "concentration_umol_per_l": 10./(i+1)}
                                  for i in range(13)])
    return document, request


def test_two_fixed_layouts_are_distinct_mixed_and_time_only(inputs):
    full, request = inputs
    results = [data.split_development(full, request, layout) for layout in (0, 1)]
    assert results[0]["design_only"]["visible_ids"] != results[1]["design_only"]["visible_ids"]
    for index, result in enumerate(results):
        assert result == data.split_development(full, request, index)
        menu = result["design_only"]
        assert set(menu) == {"visible_ids", "candidate_ids", "visible_times_h", "candidate_times_h"}
        assert len(menu["visible_ids"]) == 4 and len(result["visible_training"]["observations"]) == 4
        assert set(menu["visible_ids"]) & set(range(1, 7))
        assert set(menu["visible_ids"]) & set(range(7, 13))
        assert set(menu["visible_ids"]) | set(menu["candidate_ids"]) == set(range(1, 13))
        assert not set(menu["visible_ids"]) & set(menu["candidate_ids"])
        assert menu["visible_times_h"] == [i/60. for i in menu["visible_ids"]]
        assert menu["candidate_times_h"] == [i/60. for i in menu["candidate_ids"]]


@pytest.mark.parametrize("layout", [0, 1])
def test_hidden_concentrations_cannot_change_exported_initial_inputs(inputs, layout):
    full, request = inputs
    original = data.split_development(full, request, layout)
    for i in original["design_only"]["candidate_ids"]:
        full["observations"][i]["concentration_umol_per_l"] *= 1e6
    assert data.split_development(full, request, layout) == original


def test_all_concentration_changes_leave_time_layout_unchanged(inputs):
    full, request = inputs
    old = [data.split_development(full, request, i)["design_only"] for i in (0, 1)]
    for i, row in enumerate(full["observations"]): row["concentration_umol_per_l"] = i+1234.
    assert [data.split_development(full, request, i)["design_only"] for i in (0, 1)] == old


def test_layout_duplicate_rejection_consumes_deterministic_stream(inputs, monkeypatch):
    full, request = inputs
    sequences = {data.SEEDS[0]: [[1, 2, 3, 7]],
                 data.SEEDS[1]: [[1, 2, 3, 7], [2, 3, 4, 8]]}
    class Generator:
        def __init__(self, seed): self.rows = iter(deepcopy(sequences[seed]))
        def sample(self, population, size): return next(self.rows)
    monkeypatch.setattr(data.random, "Random", Generator)
    assert data.split_development(full, request, 0)["design_only"]["visible_ids"] == [1, 2, 3, 7]
    assert data.split_development(full, request, 1)["design_only"]["visible_ids"] == [2, 3, 4, 8]


def test_nonmixed_draw_is_rejected(inputs, monkeypatch):
    full, request = inputs
    seqs = {data.SEEDS[0]: [[1, 2, 3, 4], [1, 3, 8, 12]],
            data.SEEDS[1]: [[7, 8, 9, 10], [2, 4, 7, 11]]}
    class Generator:
        def __init__(self, seed): self.rows = iter(deepcopy(seqs[seed]))
        def sample(self, population, size): return next(self.rows)
    monkeypatch.setattr(data.random, "Random", Generator)
    assert data.split_development(full, request, 0)["design_only"]["visible_ids"] == [1, 3, 8, 12]
    assert data.split_development(full, request, 1)["design_only"]["visible_ids"] == [2, 4, 7, 11]


def test_source_metadata_and_per_row_extras_never_leave_vault(inputs):
    full, request = inputs
    full.update(raw_observations=deepcopy(full["observations"]), source_curve=[99.],
                selection={"hidden_values": [123.]}, schema_version={"hidden": [1.]},
                compound_name={"hidden": [2.]}, study_label={"hidden": [3.]})
    for row in full["observations"]: row["raw_curve"] = [987., 654.]
    split = data.split_development(full, request)
    docs = [split["visible_training"],
            data.select_ids(full, request, [1, 2, 3, 4]),
            data.reveal(full, split["visible_training"], split["design_only"]["candidate_ids"][0], request)]
    for doc in docs:
        assert set(doc) == set(data._TRAINING_FIELDS) | {"observations"}
        assert all(set(row) == {"time_min", "concentration_umol_per_l"} for row in doc["observations"])


def test_three_reveals_equal_registered_union_and_fourth_is_rejected(inputs):
    full, request = inputs
    original = deepcopy(inputs)
    split = data.split_development(full, request)
    visible_ids = split["design_only"]["visible_ids"][:]
    current = split["visible_training"]
    for budget, candidate in enumerate(split["design_only"]["candidate_ids"][:3], start=1):
        current = data.reveal(full, current, candidate, request)
        visible_ids.append(candidate)
        assert len(current["observations"]) == 4+budget
        assert current == data.select_ids(full, request, visible_ids)
    with pytest.raises(ValueError, match="at most seven"):
        data.reveal(full, current, split["design_only"]["candidate_ids"][3], request)
    assert inputs == original


def test_reveal_changes_only_one_hidden_value(inputs):
    full, request = inputs
    split = data.split_development(full, request)
    candidate = split["design_only"]["candidate_ids"][0]
    original = data.reveal(full, split["visible_training"], candidate, request)
    for other in split["design_only"]["candidate_ids"][1:]:
        full["observations"][other]["concentration_umol_per_l"] *= 100000
    assert data.reveal(full, split["visible_training"], candidate, request) == original


@pytest.mark.parametrize("candidate", [0, -1, 1000, True, 1.5, "1", None])
def test_rejects_invalid_or_nonpositive_time_candidate(inputs, candidate):
    full, request = inputs
    current = data.split_development(full, request)["visible_training"]
    with pytest.raises(ValueError): data.reveal(full, current, candidate, request)


def test_rejects_already_visible_candidate(inputs):
    full, request = inputs
    split = data.split_development(full, request)
    with pytest.raises(ValueError, match="already visible"):
        data.reveal(full, split["visible_training"], split["design_only"]["visible_ids"][0], request)


@pytest.mark.parametrize("ids", [[1, 2, 3], list(range(1, 9)), [1, 2, 3, 3],
                                 [1, 2, 3, True], [1, 2, 3, 0], [1, 2, 3, 99],
                                 [1, 2, 3, 4.], {1, 2, 3, 4}])
def test_select_ids_enforces_four_to_seven_distinct_eligible_rows(inputs, ids):
    full, request = inputs
    with pytest.raises(ValueError): data.select_ids(full, request, ids)


def test_select_ids_canonicalizes_order_and_deep_copies(inputs):
    full, request = inputs
    a = data.select_ids(full, request, [9, 2, 4, 6])
    assert a == data.select_ids(full, request, [2, 4, 6, 9])
    a["human"]["weight_kg"] = 1.
    a["observations"][0]["concentration_umol_per_l"] = 999.
    assert full["human"]["weight_kg"] == 70.
    assert full["observations"][2]["concentration_umol_per_l"] != 999.


@pytest.mark.parametrize("change", ["study", "compound", "dose", "human", "mw", "row_value", "row_time", "unit", "role"])
def test_reveal_rejects_changed_current_data_identity_or_values(inputs, change):
    full, request = inputs
    split = data.split_development(full, request)
    current = split["visible_training"]
    if change == "study": current["study_id"] = "target-study"
    elif change == "compound": current["compound_id"] = "other"
    elif change == "dose": current["dosing"]["dose_mg"] *= 2
    elif change == "human": current["human"]["weight_kg"] *= 2
    elif change == "mw": current["molecular_weight_g_mol"] += 1
    elif change == "row_value": current["observations"][0]["concentration_umol_per_l"] += 1e-10
    elif change == "row_time": current["observations"][0]["time_min"] += 1e-10
    elif change == "unit": current["unit"] = "mg/l"
    else: current["role"] = "target_observations"
    with pytest.raises(ValueError):
        data.reveal(full, current, split["design_only"]["candidate_ids"][0], request)


@pytest.mark.parametrize("index", [-1, 2, True, 0., "0", None])
def test_unknown_layout_index_rejected(inputs, index):
    full, request = inputs
    with pytest.raises(ValueError): data.split_development(full, request, index)


def test_less_than_seven_positive_points_cannot_support_all_budgets(inputs):
    full, request = inputs
    full["observations"] = full["observations"][:7]
    with pytest.raises(ValueError, match="seven positive"):
        data.split_development(full, request)


def test_human_metadata_cannot_carry_hidden_curve(inputs):
    full, request = inputs
    full["human"]["hidden_values"] = [123.]
    request["human"]["hidden_values"] = [123.]
    with pytest.raises(ValueError, match="unknown fields"):
        data.split_development(full, request)
