"""Formula isolation, input barriers, trajectory averaging and real exchange NCA."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import clinical_pbpk_comparison as comparison
from pbpk_kb.blueprints import evidence_blueprint as evidence_module
from tests.test_blueprints.test_evidence_blueprint import request_data
from tests.test_blueprints.test_permeability_model import physiology
from tests.test_blueprints.test_capacity_guided_design import volume
from tests.test_blueprints.test_learned_exchange_design import half_life


@pytest.fixture
def formula_database(request_data, monkeypatch):
    root, request = request_data
    path = root / "formulas.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE tab_model_calculation_methods (model TEXT, calculation_method TEXT)")
        db.execute("CREATE TABLE tab_species_calculation_methods (species TEXT, calculation_method TEXT)")
        for method in comparison.PARTITION_METHODS.values():
            db.execute("INSERT INTO tab_model_calculation_methods VALUES ('4Comp',?)", (method,))
            db.execute("INSERT INTO tab_species_calculation_methods VALUES ('Human',?)", (method,))
    resource = {"role": "database", "path": "formulas.sqlite", "sha256": comparison.sha256(path)}
    monkeypatch.setattr(comparison, "_method_source", lambda r, m: ({"method": m}, {"resources": [resource]}))
    return root, request


@pytest.fixture
def native_stub(formula_database, monkeypatch):
    root, request = formula_database
    captured = []
    times = np.unique(np.r_[np.linspace(0., 48., 289), np.linspace(48., 168., 121), .25])
    values = np.where(times < .25, times / .25, np.exp(-.1 * (times-.25)))
    curve = {"time_min": (times * 60.).tolist(), "concentration": values.tolist(),
             "time_unit": "min", "unit": "umol/l"}

    def execute(**kwargs):
        captured.append(deepcopy(kwargs))
        folder = kwargs["output_dir"]
        folder.mkdir(parents=True)
        comparison.save(folder / "snapshot.json", kwargs["snapshot"])
        target = kwargs["snapshot"]["Simulations"][0]["OutputSelections"][0]
        path = folder / "curve.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["IndividualId", "Time [min]", target + " [µmol/l]"])
            writer.writerows((0, t, c) for t, c in zip(curve["time_min"], curve["concentration"]))
        result = {"engineering_accepted": True, "manifest_path": str(folder / "execution_manifest.json"),
            "runtime_binding": {"actual_engine_version": "synthetic"},
            "curve_outputs": [{"path": "curve.csv", "sha256": comparison.sha256(path)}]}
        comparison.save(folder / "execution_manifest.json", result)
        return result

    def administration(**kwargs):
        kwargs["output_dir"].mkdir()
        result = {"accepted": True, "reasons": []}
        comparison.save(kwargs["output_dir"] / "audit.json", result)
        return result

    def parameters(snapshot, execution, folder, root):
        folder.mkdir()
        result = {"accepted": True, "reasons": []}
        comparison.save(folder / "audit.json", result)
        return result

    monkeypatch.setattr(comparison, "execute_snapshot", execute)
    monkeypatch.setattr(comparison, "audit_execution", administration)
    monkeypatch.setattr(comparison, "derive_administration_expectations", lambda snapshot: {})
    monkeypatch.setattr(comparison, "audit_evidence_execution", parameters)
    return root, request, captured, curve


@pytest.mark.parametrize("method", list(comparison.PARTITION_METHODS))
def test_five_formula_override_is_only_two_selections(formula_database, method):
    root, request = formula_database
    original = evidence_module.assemble_evidence_blueprint(request, root)["snapshot"]
    untouched = deepcopy(original)
    changed, _ = comparison._partition_snapshot(original, method, root)
    expected = [comparison.PARTITION_METHODS[method], comparison.PERMEABILITY_METHOD]
    assert changed["Compounds"][0]["CalculationMethods"] == expected
    assert changed["Simulations"][0]["Compounds"][0]["CalculationMethods"] == expected
    changed["Compounds"][0]["CalculationMethods"] = original["Compounds"][0]["CalculationMethods"]
    changed["Simulations"][0]["Compounds"][0]["CalculationMethods"] = original["Simulations"][0]["Compounds"][0]["CalculationMethods"]
    assert changed == untouched and original == untouched
    assert original["Compounds"][0]["Processes"][0]["Parameters"][1]["Value"] == 1.


def test_unknown_or_unbound_formula_rejected(formula_database):
    root, request = formula_database
    snapshot = evidence_module.assemble_evidence_blueprint(request, root)["snapshot"]
    with pytest.raises(ValueError, match="Unsupported"):
        comparison._partition_snapshot(snapshot, "choose_using_target_curve", root)
    with sqlite3.connect(root / "formulas.sqlite") as db:
        db.execute("DELETE FROM tab_species_calculation_methods WHERE calculation_method=?",
                   (comparison.PARTITION_METHODS["Schmitt"],))
    with pytest.raises(ValueError, match="not bound"):
        comparison._partition_snapshot(snapshot, "Schmitt", root)


def test_native_execution_keeps_units_readbacks_horizon_and_no_fitting(native_stub):
    root, request, captured, curve = native_stub
    result = comparison.run_native_partition(request, root / "result", root, "Berezhkovskiy")
    assert result["status"] == "predicted" and result["curve"] == curve
    assert not result["observations_used"] and not result["endpoints"]["parameters_fitted"]
    call = captured[0]
    assert call["cpu_threads"] == 2 and call["timeout_seconds"] == 180
    assert call["snapshot"]["Simulations"][0]["OutputSchema"][-1]["Parameters"][1]["Value"] == 168.
    assert all(result["readback_checks"].values())
    processes = call["snapshot"]["Compounds"][0]["Processes"]
    assert [p["InternalName"] for p in processes] == ["LiverClearance"]
    assert result["endpoints"]["main_window"]["end_h"] == 12.
    assert comparison._read_ref(result["result_ref"], root)["curve"] == curve
    with pytest.raises(FileExistsError):
        comparison.run_native_partition(request, root / "result", root, "Berezhkovskiy")


def test_target_evidence_blocked_before_any_source_read(native_stub, monkeypatch):
    root, request, _, _ = native_stub
    request["evidence"][0]["context"]["study_role"] = "target"
    monkeypatch.setattr(comparison, "assemble_evidence_blueprint", lambda *a: pytest.fail("source must stay unread"))
    with pytest.raises(ValueError, match="Target outcome"):
        comparison.run_native_partition(request, root / "blocked", root, "PKSimStandard")
    assert not (root / "blocked").exists()


def test_failed_native_readback_not_scored(native_stub, monkeypatch):
    root, request, _, _ = native_stub
    def bad(snapshot, execution, folder, root):
        folder.mkdir()
        record = {"accepted": False, "reasons": ["clearance_capacity_or_formula_failed"]}
        comparison.save(folder / "audit.json", record)
        return record
    monkeypatch.setattr(comparison, "audit_evidence_execution", bad)
    monkeypatch.setattr(comparison, "summarize_extended_prediction", lambda *a: pytest.fail("invalid run cannot be scored"))
    result = comparison.run_native_partition(request, root / "failed", root, "PKSimStandard")
    assert result["status"] == "failed" and result["curve"] is None
    assert result["readback_diagnostics"]["parameters"] == ["clearance_capacity_or_formula_failed"]


def test_mean_curves_is_mean_before_endpoint_and_requires_explicit_grid():
    def curve(t, v):
        return {"time_min": t, "concentration": v, "time_unit": "min", "unit": "umol/l"}
    a, b = curve([0., 1., 2.], [0., 6., 0.]), curve([0., 1., 2.], [0., 0., 4.])
    result = comparison.mean_native_curves([a, b])
    assert result["concentration"] == [0., 3., 2.]
    assert max(result["concentration"]) == 3.  # Not mean individual Cmax=5.
    assert result["ensemble_audit"]["weights"] == [.5, .5]
    c = curve([0., 2.], [0., 4.])
    with pytest.raises(ValueError, match="explicitly declared"):
        comparison.mean_native_curves([a, c])
    explicit = comparison.mean_native_curves([a, c], target_times_min=[0., 1., 2.])
    assert explicit["concentration"] == [0., 4., 2.]
    with pytest.raises(ValueError, match="extrapolate"):
        comparison.mean_native_curves([a, c], target_times_min=[0., 3.])
    with pytest.raises(ValueError, match="Every ensemble"):
        comparison.mean_native_curves([a, None])


def test_ours_reuses_bound_native_grid_and_executes_real_sampled_nca(native_stub, physiology, monkeypatch):
    root, request, captured, _ = native_stub
    native = comparison.run_native_partition(request, root / "standard", root, "PKSimStandard")
    source = root / "sidecars.json"
    source.write_text('{"fixture": "independent synthetic predictions"}')
    records = [volume(100.), half_life(20.)]
    for record in records:
        record["compound_id"] = request["compound_id"]
        record["source"] = {"path": "sidecars.json", "sha256": comparison.sha256(source), "location": record["parameter"]}
    def export(**kwargs):
        folder = kwargs["output_dir"]
        folder.mkdir()
        comparison.save(folder / "physiology.json", physiology)
        return {**physiology, "manifest": comparison._ref(folder / "physiology.json", root)}
    monkeypatch.setattr(comparison, "export_physiology", export)
    result = comparison.run_ours(request, [records[0]], [records[1]], root / "ours", root, native_result=native)
    assert result["status"] == "predicted", result
    assert len(captured) == 1 and result["native_solver_calls"] == 0
    np.testing.assert_allclose(result["curve"]["time_min"], native["curve"]["time_min"], rtol=1e-14)
    assert result["design_audit"]["target_basis"] == "sampled_nca"
    assert result["linear_trajectory_calls"] <= 33
    assert result["mass_balance"]["maximum_relative_error"] < 1e-8
    assert not result["exact_moments_used_to_fill_NCA"]
    assert result["endpoints"] == comparison.summarize_extended_prediction(result["curve"], request)
    mutated = deepcopy(native)
    mutated["curve"]["concentration"][1] *= 2
    with pytest.raises(ValueError, match="persisted receipt"):
        comparison.run_ours(request, [records[0]], [records[1]], root / "bad_reuse", root, native_result=mutated)
    assert not (root / "bad_reuse").exists()
