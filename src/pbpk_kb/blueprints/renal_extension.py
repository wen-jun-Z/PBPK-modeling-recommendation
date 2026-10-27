"""Native parent-drug urinary recovery without changing model parameters.

Only one compound and one single IV administration are supported. Urine is the
PK-Sim collecting compartment, not a measured urine volume/concentration.
Bounds and amount/fraction agreement do NOT establish whole-body mass balance.
No observed concentration, clinical endpoint or fitted target enters this module.
"""
from __future__ import annotations

import bisect
import copy
import csv
import hashlib
import json
import math
import re
from pathlib import Path

VERSION = "native-renal-output-v1"
RTOL = 1e-6
ATOL_MG = 1e-9


def _number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric value")
    if positive and value <= 0:
        raise ValueError("Expected a positive value")
    return float(value)


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _inside(path, root):
    path = Path(path)
    resolved = (root / path if not path.is_absolute() else path).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Renal readback files must remain inside the project")
    return resolved


def _metadata(snapshot):
    if (snapshot.get("Version") != 80 or len(snapshot.get("Compounds", [])) != 1
            or len(snapshot.get("Simulations", [])) != 1):
        raise ValueError("Renal extension requires one Version80 compound and simulation")
    compound, simulation = snapshot["Compounds"][0], snapshot["Simulations"][0]
    name = compound["Name"]
    if not isinstance(name, str) or not name or any(c in name for c in "|/\\\n\r"):
        raise ValueError("Invalid compound name")
    selected = simulation.get("Compounds", [])
    if len(selected) != 1 or selected[0].get("Name") != name:
        raise ValueError("Simulation must select the same single compound")
    processes = [p for p in compound.get("Processes", []) if p.get("InternalName") == "KidneyClearance"]
    if len(processes) != 1:
        raise ValueError("An explicit KidneyClearance process is required; output selection alone is insufficient")
    renal = processes[0]
    chosen = selected[0].get("Processes", [])
    expected = "Renal Clearances-" + renal["DataSource"]
    if sum(p.get("SystemicProcessType") == "Renal" and p.get("Name") == expected for p in chosen) != 1:
        raise ValueError("The KidneyClearance process is not selected in the simulation")
    if any(p.get("SystemicProcessType") == "GFR" for p in chosen):
        raise ValueError("KidneyClearance cannot overlap an additional selected GFR process")
    clearance = [p for p in renal.get("Parameters", []) if p.get("Name") in {"Plasma clearance", "Specific clearance"}]
    if not clearance or not any(_number(p["Value"]) > 0 for p in clearance):
        raise ValueError("Selected renal clearance must be positive")
    parameters = {p["Name"]: p for p in compound["Parameters"]}
    mw = parameters["Molecular weight"]
    if mw.get("Unit") != "g/mol":
        raise ValueError("Molecular weight must use g/mol")
    protocol_name = selected[0]["Protocol"]["Name"]
    protocols = [p for p in snapshot.get("Protocols", []) if p.get("Name") == protocol_name]
    if len(protocols) != 1:
        raise ValueError("Expected one selected administration protocol")
    protocol = protocols[0]
    if protocol.get("DosingInterval") != "Single" or protocol.get("ApplicationType") not in {"Intravenous", "IntravenousBolus"}:
        raise ValueError("Only a single IV infusion or bolus is supported")
    pars = {p["Name"]: p for p in protocol["Parameters"]}
    if pars["InputDose"].get("Unit") != "mg":
        raise ValueError("Dose must be an absolute mg amount, not mg/kg")
    def hours(p):
        scale = {"h": 1., "min": 1/60}.get(p.get("Unit"))
        if scale is None:
            raise ValueError("Administration time must use min or h")
        return _number(p["Value"]) * scale
    start = hours(pars["Start time"])
    infusion = hours(pars["Infusion time"]) if protocol["ApplicationType"] == "Intravenous" else 0.
    if start < 0 or infusion < 0 or (protocol["ApplicationType"] == "Intravenous" and infusion == 0):
        raise ValueError("Invalid IV administration times")
    plasma = "Organism|PeripheralVenousBlood|" + name + "|Plasma (Peripheral Venous Blood)"
    outputs = simulation.get("OutputSelections", [])
    if plasma not in outputs or len(outputs) != len(set(outputs)):
        raise ValueError("Existing peripheral venous plasma output must be retained exactly once")
    amount = "Organism|Kidney|Urine|" + name
    return {"compound_name": name, "dose_mg": _number(pars["InputDose"]["Value"], positive=True),
            "molecular_weight_g_mol": _number(mw["Value"], positive=True),
            "dose_start_h": start, "infusion_h": infusion, "plasma_path": plasma,
            "amount_path": amount, "fraction_path": amount + "|Fraction excreted to urine"}


def add_renal_outputs(snapshot):
    """Deep-copy snapshot; append native urine amount and fraction, changing no input."""
    meta = _metadata(snapshot)
    extended = copy.deepcopy(snapshot)
    selections = extended["Simulations"][0]["OutputSelections"]
    for key in ("amount_path", "fraction_path"):
        if meta[key] not in selections:
            selections.append(meta[key])
    if len(selections) > 64:
        raise ValueError("Renal selections exceed the existing 64-output snapshot limit")
    return extended


def _column(header, wanted):
    matches = []
    for i, label in enumerate(header):
        match = re.fullmatch(r"(.*?)\s+\[([^\]]*)\]", label)
        path, unit = (match.group(1), match.group(2)) if match else (label, "")
        if path == wanted:
            matches.append((i, unit))
    if len(matches) != 1:
        raise ValueError("Missing/ambiguous native output: " + wanted)
    return matches[0]


def _unit(unit):
    return unit.replace("μ", "u").replace("µ", "u").strip().lower()


def _amount_factor(unit, mw):
    factors = {"mg": 1., "g": 1000., "ug": .001, "ng": 1e-6,
               "mol": mw*1000., "mmol": mw, "umol": mw/1000., "nmol": mw/1e6}
    try:
        return factors[_unit(unit)]
    except KeyError as exc:
        raise ValueError("Unsupported native urinary amount unit: " + unit) from exc


def _checks(times, amounts, fractions, metadata):
    dose = metadata["dose_mg"]
    tolerance = ATOL_MG + RTOL*dose
    fraction_tolerance = tolerance/dose
    def administered(t):
        elapsed = t-metadata["dose_start_h"]
        if elapsed < 0:
            return 0.
        return dose if metadata["infusion_h"] == 0 else dose*min(1., elapsed/metadata["infusion_h"])
    return {
        "finite_series": all(math.isfinite(x) for series in (times, amounts, fractions) for x in series),
        "nonnegative": min(amounts) >= -tolerance and min(fractions) >= -fraction_tolerance,
        "time_strictly_increasing": all(b > a for a,b in zip(times, times[1:])),
        "cumulative_amount_monotone": all(b-a >= -tolerance for a,b in zip(amounts, amounts[1:])),
        "cumulative_fraction_monotone": all(b-a >= -fraction_tolerance for a,b in zip(fractions, fractions[1:])),
        "amount_not_above_total_dose": max(amounts) <= dose+tolerance,
        "fraction_not_above_one": max(fractions) <= 1.+fraction_tolerance,
        "amount_not_above_administered_dose": all(v <= administered(t)+tolerance for t,v in zip(times, amounts)),
        "fraction_matches_amount_over_dose": all(abs(f-a/dose) <= fraction_tolerance for f,a in zip(fractions, amounts)),
    }


def read_renal_outputs(execution_manifest_path, snapshot, project_root):
    """Read SHA-bound native CSV; return unaltered series, units and physical checks.

    ``fraction`` is 0..1, never percent. Small numerical negatives are retained,
    with declared tolerances; values are never clipped, monotonized or refitted.
    """
    root = Path(project_root).resolve(strict=True)
    manifest_path = _inside(execution_manifest_path, root)
    manifest_sha = _sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("engineering_accepted") or not manifest.get("source_unchanged"):
        raise ValueError("A successful unchanged native execution is required")
    source = _inside(manifest_path.parent/"snapshot.json", root)
    if _sha(source) != manifest.get("source_sha256") or json.loads(source.read_text()) != snapshot:
        raise ValueError("Snapshot differs from its frozen native execution")
    metadata = _metadata(snapshot)
    if len(manifest.get("curve_outputs", [])) != 1:
        raise ValueError("Expected exactly one native results CSV")
    artifact = manifest["curve_outputs"][0]
    path = _inside(manifest_path.parent/artifact["path"], root)
    if not path.is_relative_to(manifest_path.parent) or _sha(path) != artifact.get("sha256"):
        raise ValueError("Native CSV path or SHA256 mismatch")
    times, amounts, fractions, plasma = [], [], [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[0] != "IndividualId":
            raise ValueError("Expected the native IndividualId column")
        ti, tu = _column(header, "Time")
        ai, au = _column(header, metadata["amount_path"])
        fi, fu = _column(header, metadata["fraction_path"])
        pi, pu = _column(header, metadata["plasma_path"])
        if _unit(tu) not in {"h", "min"}:
            raise ValueError("Unsupported native time unit")
        if _unit(fu) not in {"", "fraction", "%"}:
            raise ValueError("Unsupported native fraction unit")
        time_factor, fraction_factor = (1/60 if _unit(tu) == "min" else 1.), (.01 if fu == "%" else 1.)
        amount_factor = _amount_factor(au, metadata["molecular_weight_g_mol"])
        person = None
        for count, row in enumerate(reader, 1):
            if count > 500000 or len(row) != len(header):
                raise ValueError("Malformed or oversized native CSV")
            values = [float(v) for v in row]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Native CSV contains nonfinite values")
            if person is None:
                person = values[0]
            if values[0] != person:
                raise ValueError("Population output is not supported")
            point = (values[ti]*time_factor, values[ai]*amount_factor,
                     values[fi]*fraction_factor, values[pi])
            if times and point[0] == times[-1]:
                if point != (times[-1], amounts[-1], fractions[-1], plasma[-1]):
                    raise ValueError("Conflicting native values at duplicate times")
                continue
            for series, value in zip((times, amounts, fractions, plasma), point):
                series.append(value)
    if len(times) < 2:
        raise ValueError("Insufficient native urine output points")
    checks = _checks(times, amounts, fractions, metadata)
    checks["plasma_output_retained"] = metadata["plasma_path"] in snapshot["Simulations"][0]["OutputSelections"]
    if _sha(path) != artifact["sha256"] or _sha(source) != manifest["source_sha256"] or _sha(manifest_path) != manifest_sha:
        raise ValueError("Native readback files changed while being read")
    return {"version": VERSION, "accepted": all(checks.values()),
            "status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "failed_checks": [key for key,value in checks.items() if not value],
            "curves": {"time_h": times, "amount_mg": amounts, "fraction": fractions,
                       "plasma_concentration": plasma, "plasma_unit": pu},
            "metadata": metadata, "units": {"time_h": "h", "amount_mg": "mg", "fraction": "fraction"},
            "native_units": {"time": tu, "amount": au, "fraction": fu, "plasma": pu},
            "tolerances": {"relative_to_dose": RTOL, "absolute_mg": ATOL_MG},
            "source": {"manifest": str(manifest_path.relative_to(root)), "manifest_sha256": manifest_sha,
                       "csv": str(path.relative_to(root)), "csv_sha256": artifact["sha256"],
                       "snapshot_sha256": manifest["source_sha256"]},
            "full_mass_balance_checked": False,
            "limitation": "Native parent-drug urinary recovery bounds and internal consistency only; not whole-body mass conservation or clinical validation."}


def _series(times, values):
    if len(times) != len(values) or len(times) < 2:
        raise ValueError("Expected matching time/value arrays with at least two points")
    out_t, out_v = [], []
    for t,v in zip(times, values):
        t,v = _number(t), _number(v)
        if out_t and t == out_t[-1] and v == out_v[-1]:
            continue
        if out_t and t <= out_t[-1]:
            raise ValueError("Time series must increase; conflicting duplicate times are invalid")
        out_t.append(t); out_v.append(v)
    if len(out_t) < 2:
        raise ValueError("Expected at least two distinct times")
    return out_t, out_v


def _interpolate(times, values, t):
    if t < times[0] or t > times[-1]:
        raise ValueError("Requested renal window exceeds native coverage; extrapolation is forbidden")
    right = bisect.bisect_left(times, t)
    if times[right] == t:
        return values[right]
    left = right-1
    return values[left]+(values[right]-values[left])*(t-times[left])/(times[right]-times[left])


def window_renal_endpoints(renal_curve, plasma_native, window_h, mw):
    """Compute Ae(0,T), percentage of dose and CLr=Ae/AUC(0,T), without extrapolation.

    Accept the complete result of read_renal_outputs and the existing locked
    plasma prediction (time_min, time_unit='min', concentration, unit). Plasma
    AUC uses linear trapezoids on the native grid with interpolated boundaries.
    """
    if renal_curve.get("accepted") is not True or not all(renal_curve.get("checks", {}).values()):
        raise ValueError("Renal readback must pass its physical checks before endpoint calculation")
    end = _number(window_h, positive=True)
    mw = _number(mw, positive=True)
    metadata = renal_curve["metadata"]
    if not math.isclose(mw, metadata["molecular_weight_g_mol"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Plasma/urinary molecular weights differ")
    if metadata["dose_start_h"] != 0:
        raise ValueError("This pilot uses collection windows from the dosing start at zero")
    curve = renal_curve["curves"]
    rt, ra = _series(curve["time_h"], curve["amount_mg"])
    rf_t, rf = _series(curve["time_h"], curve["fraction"])
    if plasma_native.get("time_unit") != "min":
        raise ValueError("Locked plasma prediction must declare time_unit=min")
    factor = {"umol/l": mw/1000., "nmol/l": mw/1e6, "mg/l": 1., "ng/ml": .001,
              "ug/ml": 1.}.get(_unit(plasma_native.get("unit", "")))
    if factor is None:
        raise ValueError("Unsupported native plasma concentration unit")
    pt, pc = _series([_number(t)/60 for t in plasma_native["time_min"]],
                     [_number(v)*factor for v in plasma_native["concentration"]])
    if min(pc) < -1e-12:
        raise ValueError("Negative plasma concentration is not valid for renal AUC")
    ae = _interpolate(rt, ra, end)-_interpolate(rt, ra, 0.)
    fraction = _interpolate(rf_t, rf, end)-_interpolate(rf_t, rf, 0.)
    grid = [0.]+[t for t in pt if 0 < t < end]+[end]
    concentrations = [_interpolate(pt, pc, t) for t in grid]
    auc = math.fsum((b-a)*(u+v)/2 for a,b,u,v in zip(grid,grid[1:],concentrations,concentrations[1:]))
    tolerance = ATOL_MG+RTOL*metadata["dose_mg"]
    if ae < 0 or ae > metadata["dose_mg"]+tolerance or not math.isfinite(auc) or auc <= 0:
        raise ValueError("Invalid window recovery or nonpositive plasma AUC")
    return {"window_h": [0., end], "amount_mg": ae, "fraction_percent": fraction*100.,
            "renal_clearance_l_h": ae/auc, "plasma_auc_mg_h_l": auc,
            "dose_mg": metadata["dose_mg"], "molecular_weight_g_mol": mw,
            "definition": "Native cumulative parent-drug urine Ae(0,T) / matched plasma AUC(0,T); no clinical values used",
            "interpolation": "linear, within native coverage only", "auc_integration": "linear trapezoid",
            "full_mass_balance_checked": False}
