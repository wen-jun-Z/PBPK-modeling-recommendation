"""Unit/receipt checks use synthetic curves, never clinical reference values."""
import copy
import csv
import hashlib
import json

import pytest

from pbpk_kb.blueprints.renal_extension import (
    add_renal_outputs, read_renal_outputs, window_renal_endpoints,
)


@pytest.fixture
def snapshot():
    return {"Version": 80, "Compounds": [{"Name": "TestDrug", "Parameters": [
        {"Name": "Molecular weight", "Value": 200., "Unit": "g/mol"}],
        "Processes": [{"InternalName": "KidneyClearance", "DataSource": "TestSource",
                       "Parameters": [{"Name": "Plasma clearance", "Value": 1.}]}]}],
        "Protocols": [{"Name": "Dose", "ApplicationType": "Intravenous", "DosingInterval": "Single",
                       "Parameters": [{"Name": "InputDose", "Value": 10., "Unit": "mg"},
                                      {"Name": "Start time", "Value": 0., "Unit": "h"},
                                      {"Name": "Infusion time", "Value": 60., "Unit": "min"}]}],
        "Simulations": [{"Name": "Test", "Compounds": [{"Name": "TestDrug", "Protocol": {"Name": "Dose"},
            "Processes": [{"SystemicProcessType": "Renal", "Name": "Renal Clearances-TestSource"}]}],
            "OutputSelections": ["Organism|PeripheralVenousBlood|TestDrug|Plasma (Peripheral Venous Blood)"]}]}


def receipt(tmp_path, snapshot, *, percent=False, amounts=(0., 10., 20.), times=(0., 60., 120.), omit_plasma=False):
    extended = add_renal_outputs(snapshot)
    source = tmp_path/"snapshot.json"; source.write_text(json.dumps(extended))
    output = tmp_path/"outputs.csv"
    headers = ["IndividualId", "Time [min]", "Organism|Kidney|Urine|TestDrug [µmol]",
               "Organism|Kidney|Urine|TestDrug|Fraction excreted to urine" + (" [%]" if percent else "")]
    if not omit_plasma:
        headers.append("Organism|PeripheralVenousBlood|TestDrug|Plasma (Peripheral Venous Blood) [µmol/l]")
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(headers)
        for t,a in zip(times, amounts):
            writer.writerow([0,t,a,a*.2/10*(100 if percent else 1)]+([] if omit_plasma else [10.]))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = tmp_path/"execution_manifest.json"
    manifest.write_text(json.dumps({"engineering_accepted": True, "source_unchanged": True,
        "source_sha256": sha(source), "curve_outputs": [{"path": output.name, "sha256": sha(output)}]}))
    return manifest, extended


def read(tmp_path, snapshot, **kwargs):
    manifest, extended = receipt(tmp_path, snapshot, **kwargs)
    return read_renal_outputs(manifest, extended, tmp_path)


def plasma():
    return {"time_min": [0.,60.,120.], "time_unit": "min", "concentration": [10.,10.,10.], "unit": "µmol/l"}


def test_append_preserves_all_inputs_and_output_order(snapshot):
    before = copy.deepcopy(snapshot); extended = add_renal_outputs(snapshot)
    assert snapshot == before
    assert extended["Simulations"][0]["OutputSelections"][:1] == before["Simulations"][0]["OutputSelections"]
    assert add_renal_outputs(extended) == extended
    extended["Simulations"][0]["OutputSelections"] = before["Simulations"][0]["OutputSelections"]
    assert extended == before


def test_missing_renal_process_refused(snapshot):
    snapshot["Compounds"][0]["Processes"] = []
    with pytest.raises(ValueError, match="KidneyClearance"):
        add_renal_outputs(snapshot)


def test_unselected_renal_process_refused(snapshot):
    snapshot["Simulations"][0]["Compounds"][0]["Processes"] = []
    with pytest.raises(ValueError, match="not selected"):
        add_renal_outputs(snapshot)


def test_overlapping_gfr_refused(snapshot):
    snapshot["Simulations"][0]["Compounds"][0]["Processes"].append({"SystemicProcessType": "GFR"})
    with pytest.raises(ValueError, match="GFR"):
        add_renal_outputs(snapshot)


def test_native_molar_amount_and_fraction_units(tmp_path, snapshot):
    r = read(tmp_path, snapshot)
    assert r["accepted"] and all(r["checks"].values())
    assert r["curves"]["amount_mg"] == [0.,2.,4.]
    assert r["curves"]["fraction"] == [0.,.2,.4]
    assert not r["full_mass_balance_checked"]


def test_native_percent_fraction_is_normalized_once(tmp_path, snapshot):
    r = read(tmp_path, snapshot, percent=True)
    assert r["accepted"] and r["curves"]["fraction"] == [0.,.2,.4]


def test_decreasing_cumulative_recovery_fails(tmp_path, snapshot):
    r = read(tmp_path, snapshot, amounts=(0.,20.,10.))
    assert not r["accepted"] and not r["checks"]["cumulative_amount_monotone"]


def test_above_dose_recovery_fails_without_clipping(tmp_path, snapshot):
    r = read(tmp_path, snapshot, amounts=(0.,20.,60.))
    assert not r["accepted"] and r["curves"]["amount_mg"][-1] == 12.
    assert not r["checks"]["amount_not_above_total_dose"]


def test_changed_output_receipt_refused(tmp_path, snapshot):
    manifest, extended = receipt(tmp_path, snapshot)
    with (tmp_path/"outputs.csv").open("a") as handle:handle.write("\n")
    with pytest.raises(ValueError, match="SHA256"):
        read_renal_outputs(manifest, extended, tmp_path)


def test_window_interpolation_auc_and_clearance_units(tmp_path, snapshot):
    r = read(tmp_path, snapshot)
    e = window_renal_endpoints(r, plasma(), 1.5, 200.)
    assert e["amount_mg"] == 3.
    assert e["fraction_percent"] == pytest.approx(30.)
    assert e["plasma_auc_mg_h_l"] == 3.
    assert e["renal_clearance_l_h"] == 1.


def test_window_extrapolation_refused(tmp_path, snapshot):
    with pytest.raises(ValueError, match="extrapolation"):
        window_renal_endpoints(read(tmp_path, snapshot), plasma(), 3., 200.)


def test_different_molecular_weight_refused(tmp_path, snapshot):
    with pytest.raises(ValueError, match="molecular weights"):
        window_renal_endpoints(read(tmp_path, snapshot), plasma(), 1., 300.)


def test_failed_physical_checks_cannot_be_scored(tmp_path, snapshot):
    r = read(tmp_path, snapshot, amounts=(0.,20.,60.))
    with pytest.raises(ValueError, match="physical checks"):
        window_renal_endpoints(r, plasma(), 1., 200.)


def test_recovery_cannot_exceed_amount_already_infused(tmp_path, snapshot):
    r = read(tmp_path, snapshot, times=(0.,30.,120.), amounts=(0.,35.,40.))
    assert not r["accepted"] and not r["checks"]["amount_not_above_administered_dose"]
    assert r["checks"]["amount_not_above_total_dose"]


def test_missing_native_plasma_output_refused(tmp_path, snapshot):
    with pytest.raises(ValueError, match="Missing/ambiguous"):
        read(tmp_path, snapshot, omit_plasma=True)
