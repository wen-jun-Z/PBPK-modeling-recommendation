"""Bounded numerical extension invariants; no native solver is started."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path

import pytest

from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2] / "scripts/supplement_ten_drug_horizons.py"
    spec = importlib.util.spec_from_file_location("horizon_supplement_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot():
    def interval(a, b):
        return {"Parameters": [{"Name": "Start time", "Value": a, "Unit": "h"},
            {"Name": "End time", "Value": b, "Unit": "h"}, {"Name": "Resolution", "Value": 60., "Unit": "pts/h"}]}
    return {"Compounds": [{"immutable": "drug"}], "Individuals": [{"immutable": "person"}],
            "Protocols": [{"immutable": "dose"}], "Version": 80,
            "Simulations": [{"Name": "original", "OutputSelections": ["plasma"],
                             "OutputSchema": [interval(0., .5), interval(.5, 168.)]}]}


def test_append_preserves_all_original_intervals_and_every_model_value(runner):
    original = snapshot()
    frozen = deepcopy(original)
    result = runner.append_output_interval(original, 336.)
    assert original == frozen
    assert result["Simulations"][0]["OutputSchema"][:-1] == original["Simulations"][0]["OutputSchema"]
    result["Simulations"][0]["OutputSchema"].pop()
    assert result == original
    for horizon in (336., 672., 1344.):
        original = runner.append_output_interval(original, horizon)
    assert len(original["Simulations"][0]["OutputSchema"]) == 5


def test_only_frozen_horizons_and_strict_extension_are_allowed(runner):
    with pytest.raises(ValueError):
        runner.append_output_interval(snapshot(), 200.)
    once = runner.append_output_interval(snapshot(), 336.)
    with pytest.raises(ValueError):
        runner.append_output_interval(once, 336.)


def test_alignment_requires_existing_original_samples_and_stable_concentrations(runner):
    old = {"time_min": [0., 1., 2.], "concentration": [0., 2., 1.], "unit": "umol/l", "time_unit": "min"}
    new = {**old, "time_min": [0., 1., 2., 3.], "concentration": [0., 2., 1., .5]}
    assert runner.compare_original_curve(old, new)["status"] == "passed"
    new["concentration"][1] = 2.01
    assert runner.compare_original_curve(old, new)["status"] == "failed"
    new["concentration"][1] = 2.
    new["time_min"][1] = 1.001
    result = runner.compare_original_curve(old, new)
    assert result["status"] == "failed" and result["original_grid_retained"] is False


def test_long_half_life_becomes_estimable_without_changing_main_window_or_gates():
    rate = .012085
    def curve(end):
        times = list(range(end + 1))
        return {"time_min": [60. * t for t in times], "concentration": [10 * math.exp(-rate * t) for t in times],
                "unit": "umol/l", "time_unit": "min"}
    request = {"dosing": {"route": "intravenous", "start_h": 0., "infusion_minutes": 0., "dose_mg": 100.},
        "evidence": [{"parameter": "molecular_weight", "value": 1000.}],
        "target": {"endpoints": ["auc"], "primary_endpoint": "auc", "window_start_h": 0., "window_end_h": 24.}}
    short = summarize_extended_prediction(curve(168), request)
    long = summarize_extended_prediction(curve(672), request)
    assert short["endpoints"]["terminal_half_life"]["status"] == "not_estimable"
    assert long["endpoint_counts"]["completed"] == 11
    assert long["endpoints"]["terminal_half_life"]["value_h"] == pytest.approx(math.log(2) / rate)
    assert short["endpoints"]["auc"] == long["endpoints"]["auc"]
    assert short["endpoints"]["c_last"] == long["endpoints"]["c_last"]
    assert short["diagnostics"]["quality_gates"] == long["diagnostics"]["quality_gates"]
