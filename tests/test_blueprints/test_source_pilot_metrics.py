"""Engineering fidelity must detect wrong dosing and missing predictions."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_source_adaptation import compare_curves


def curve(scale=1):
    return {"time_min": list(range(12)), "concentration": [i * scale for i in range(12)]}


def test_matching_source_reference():
    assert compare_curves(curve(), curve())["accepted"]


def test_wrong_dose_is_not_source_reproduction():
    assert not compare_curves(curve(2), curve())["accepted"]


def test_missing_prediction_and_short_window_fail():
    assert not compare_curves(None, curve())["accepted"]
    truncated = curve()
    truncated["time_min"].pop()
    truncated["concentration"].pop()
    assert not compare_curves(truncated, curve())["accepted"]


def test_extra_peak_between_reference_times_cannot_pass():
    unexpected = curve()
    unexpected["time_min"].insert(1, 0.5)
    unexpected["concentration"].insert(1, 1000)
    assert not compare_curves(unexpected, curve())["accepted"]
