"""Three-axis native contract: nonlinear CL inversion and fixed renal evidence."""
from copy import deepcopy
import json

import pytest

from pbpk_kb.blueprint_policy import native_axes_runtime as runtime


def _clearance_fixture(scale=2.):
    processes = [{"organ": "hepatic", "stem": "Drug-Hepatic", "clearance_unit": "ml/min/kg"},
                 {"organ": "renal", "stem": "Drug-Renal", "clearance_unit": "ml/min/kg"}]
    common = {"Body weight": (70., "kg"), "Fraction unbound (experiment)": (.5, "fraction")}
    liver = {**common, "Plasma clearance": (1., "ml/min/kg"),
        "Blood flow rate (liver)": (1.5, "l/min"), "Blood flow rate (portal vein)": (.5, "l/min"),
        "Blood/Plasma concentration ratio": (1., "fraction"), "Volume (liver)": (1.5, "l"),
        "Fraction intracellular (liver)": (.8, "fraction"),
        "Specific clearance": (.07*2/(.5*(2-.07)*1.2), "1/min")}
    kidney = {**common, "Plasma clearance": (.2, "ml/min/kg"),
        "Blood flow rate (kidney)": (1., "l/min"), "Hematocrit": (.4, "fraction"),
        "Volume (kidney)": (.3, "l"),
        "Specific clearance": (.014*.6/(.5*(.6-.014)*.3), "1/min")}
    rows, specs = [], []
    for process, fields in zip(processes, (liver, kidney)):
        for name, (value, unit) in fields.items():
            path = process["stem"] + "|" + name
            after = value
            if process["organ"] == "hepatic":
                if name == "Plasma clearance":
                    after *= scale
                elif name == "Specific clearance":
                    cl = .07 * scale
                    after = cl*2/(.5*(2-cl)*1.2) if cl < 2 else 1e6
            specs.append({"path": path, "unit": unit})
            if name == "Plasma clearance":
                specs[-1].update(expected=value, relative_tolerance=1e-8)
            rows.append({"path": path, "before": value, "after": after, "unit": unit})
    plan = {"clearance_parameters": specs, "clearance_processes": processes,
        "clearance_changes": [{"path": "Drug-Hepatic|Plasma clearance", "unit": "ml/min/kg", "axis": "clearance_scale"}]}
    scales = {"clearance_scale": scale, "partition_scale": 1., "bidirectional_exchange_scale": 1.}
    return rows, plan, scales


def test_clearance_uses_nonlinear_native_inversion_not_scaled_specific_cl():
    rows, plan, scales = _clearance_fixture()
    checked = runtime.validate_clearance_readback(rows, plan, scales)
    assert checked["accepted"]
    liver = next(r for r in rows if r["path"] == "Drug-Hepatic|Specific clearance")
    assert liver["after"] != liver["before"] * scales["clearance_scale"]
    liver["after"] = liver["before"] * scales["clearance_scale"]
    assert not runtime.validate_clearance_readback(rows, plan, scales)["accepted"]


def test_net_renal_input_and_specific_clearance_remain_unchanged():
    rows, plan, scales = _clearance_fixture()
    assert all(r["before"] == r["after"] for r in rows if r["path"].startswith("Drug-Renal|"))
    for name in ("Plasma clearance", "Specific clearance"):
        bad = deepcopy(rows)
        next(r for r in bad if r["path"] == "Drug-Renal|" + name)["after"] *= 2
        with pytest.raises(ValueError, match="protection"):
            runtime.validate_clearance_readback(bad, plan, scales)


def test_flow_capacity_and_native_fallback_are_not_accepted():
    rows, plan, scales = _clearance_fixture(40.)
    checked = runtime.validate_clearance_readback(rows, plan, scales)
    assert not checked["accepted"]
    assert checked["source"][0]["accepted"]
    assert not checked["trial"][0]["within_flow_capacity"]
    assert not checked["trial"][0]["native_fallback_1000000_absent"]


def test_trial_can_repair_raw_source_capacity_without_hiding_source_failure():
    rows, plan, scales = _clearance_fixture(.5)
    # Source CL 40 mL/min/kg exceeds flow; trial CL 20 is representable.
    plasma = next(r for r in rows if r["path"] == "Drug-Hepatic|Plasma clearance")
    plasma.update(before=40., after=20.)
    specific = next(r for r in rows if r["path"] == "Drug-Hepatic|Specific clearance")
    specific.update(before=1e6, after=1.4*2/(.5*(2-1.4)*1.2))
    next(s for s in plan["clearance_parameters"] if s["path"] == plasma["path"])["expected"] = 40.
    checked = runtime.validate_clearance_readback(rows, plan, scales)
    assert checked["accepted"] and not checked["source_accepted"]


@pytest.mark.parametrize("change", ["unit", "path", "source", "finite", "missing"])
def test_readback_integrity_is_distinct_from_scientific_capacity(change):
    rows, plan, scales = _clearance_fixture()
    if change == "unit": rows[0]["unit"] = "mg"
    elif change == "path": rows[0]["path"] = "wrong"
    elif change == "source":
        next(r for r in rows if r["path"] == "Drug-Hepatic|Plasma clearance")["before"] *= 2
    elif change == "finite": rows[0]["after"] = float("nan")
    else: rows.pop()
    with pytest.raises(ValueError):
        runtime.validate_clearance_readback(rows, plan, scales)


@pytest.mark.parametrize("value", [False, 0., -1., float("inf"), float("nan"), "1"])
def test_clearance_scale_requires_finite_positive_value(value):
    with pytest.raises(ValueError):
        runtime.validate_scales({"clearance_scale": value, "partition_scale": 1., "bidirectional_exchange_scale": 1.})


def test_exact_axis_names_required():
    with pytest.raises(ValueError): runtime.validate_scales({"clearance_scale": 1.})
    with pytest.raises(ValueError): runtime.validate_scales({"clearance_scale": 1., "partition_scale": 1., "bidirectional_exchange_scale": 1., "fu_scale": 1.})


def test_counter_records_attempted_solve_separately_from_trial(tmp_path):
    assert runtime._read_counter(tmp_path, "hash") == 0
    receipt = tmp_path / "solver_receipt.json"
    receipt.write_text(json.dumps({"request_sha256": "hash", "actual_native_solver_calls": 1}))
    assert runtime._read_counter(tmp_path, "hash") == 1
    with pytest.raises(ValueError, match="another request"):
        runtime._read_counter(tmp_path, "other")
    receipt.write_text(json.dumps({"request_sha256": "hash", "actual_native_solver_calls": True}))
    with pytest.raises(ValueError, match="counter"):
        runtime._read_counter(tmp_path, "hash")
