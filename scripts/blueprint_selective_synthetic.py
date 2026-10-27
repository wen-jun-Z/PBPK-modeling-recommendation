"""Outcome-free synthetic cases for the selective RBC calibration experiment.

This module reads no files, fits no model, and never receives clinical outcomes.
The caller supplies native physiology and the development request/time design.
Ground truth and clean transfer data stay outside ``calibration_input``.
Synthetic evidence is not additional clinical evidence or drug coverage.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from pbpk_kb.blueprint_policy import adaptive_exchange_model
from pbpk_kb.blueprint_policy import blood_partition_calibration
from pbpk_kb.blueprint_policy.perfusion_calibration import clearance_inputs, prediction_grid


VERSION = "selective-rbc-synthetic-cases-1"
SCENARIOS = ("correct_RBC", "wrong_RBC", "nuisance_error", "wrong_exchange")
NOISE_LEVELS = (.05, .15)
NOISE_STREAM = 202610081


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(name + " must be finite and " + ("positive" if positive else "nonnegative"))
    return float(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _design(times_h, five_indices):
    raw = np.asarray(times_h, dtype=object)
    if raw.ndim != 1 or len(raw) < 5:
        raise ValueError("Full design needs at least five explicit times")
    times = np.asarray([_number(x, "Full time", positive=True) for x in raw], float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Full times must be strictly increasing")
    indices = list(five_indices)
    if (len(indices) != 5 or any(isinstance(i, bool) or not isinstance(i, Integral) for i in indices)
            or indices != sorted(set(indices)) or min(indices) < 0 or max(indices) >= len(times)):
        raise ValueError("Five indices must be five ordered distinct valid full-design indices")
    return times, [int(i) for i in indices]


def _simulate(physiology, request, times, truth4, variant):
    if variant == "blood_partition":
        result = blood_partition_calibration.simulate(physiology, request, times, truth4)
    else:
        # Validate unchanged native/protected RBC inputs without changing them.
        blood_partition_calibration.derive_physiology(physiology, request, 0.)
        cl = clearance_inputs(request)
        result = adaptive_exchange_model.simulate(physiology, times_h=times,
            dose_mg=request["dosing"]["dose_mg"],
            infusion_duration_h=request["dosing"]["infusion_minutes"] / 60.,
            total_clearance_l_h=cl["total_clearance_l_h"],
            renal_clearance_l_h=cl["renal_clearance_l_h"], family="adaptive_exchange",
            log_parameters=dict(zip(adaptive_exchange_model.PARAMETERS["adaptive_exchange"],
                                    [0., 0., 0., .8])), observation="auto")
    values = np.asarray(result["concentration_mg_l"], float)
    if (values.shape != np.asarray(times).shape or not np.isfinite(values).all()
            or np.any(values < 0) or not np.any(values > 0)):
        raise ValueError("Generator returned invalid concentrations")
    return values, deepcopy(result.get("mass_balance", {}))


def build_case(cid, physiology, request, full_times_h, five_indices, scenario, log_sigma, seed):
    """Generate shared noisy full/five data and a clean altered-infusion curve.

    ``truth4`` is always [log CL, log tissue K, log PS, log RBC multiplier].
    In the misspecified exchange scenario the RBC truth is zero and the separate
    generator alpha is 0.8. Wrong RBC truth is 0.2 for odd seeds and 5 for even
    seeds. The transfer infusion factor depends only on a SHA256 bit of ``cid``;
    it is independent of noise seeds and the RBC-error direction.
    """
    if not isinstance(cid, str) or not cid or scenario not in SCENARIOS:
        raise ValueError("Explicit case ID and known synthetic scenario are required")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    sigma = _number(log_sigma, "Log noise standard deviation")
    times, indices = _design(full_times_h, five_indices)
    if not isinstance(request, dict) or not request.get("compound_id"):
        raise ValueError("An identified development request is required")
    dose = request.get("dosing", {})
    if dose.get("route") != "intravenous" or dose.get("start_h", 0.) != 0.:
        raise ValueError("The synthetic pilot requires a single IV infusion at time zero")
    duration = _number(dose.get("infusion_minutes"), "Infusion duration", positive=True)
    _number(dose.get("dose_mg"), "Dose", positive=True)
    _number(request.get("target", {}).get("window_end_h"), "Requested window", positive=True)
    original_physiology_sha, original_request_sha = _digest(physiology), _digest(request)
    truth4 = [0., 0., 0., 0.]
    if scenario == "wrong_RBC":
        truth4[3] = math.log(.2 if seed % 2 else 5.)
    elif scenario == "nuisance_error":
        truth4[:3] = [math.log(1.5), math.log(.7), math.log(2.)]
    variant = "exchange_shape" if scenario == "wrong_exchange" else "blood_partition"
    clean, training_mass = _simulate(physiology, request, times, truth4, variant)
    if np.any(clean <= 0):
        raise ValueError("Lognormal development observations require positive clean concentrations")
    compound_stream = int.from_bytes(hashlib.sha256(cid.encode()).digest()[:4], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), NOISE_STREAM, compound_stream]))
    observed = clean * np.exp(rng.normal(0., sigma, len(times)))
    if not np.isfinite(observed).all() or np.any(observed <= 0):
        raise ValueError("Generated noisy concentrations are nonfinite or nonpositive")
    transfer_request = deepcopy(request)
    factor = .5 if hashlib.sha256(cid.encode()).digest()[0] & 1 else 2.
    transfer_request["dosing"]["infusion_minutes"] = duration * factor
    transfer_times = prediction_grid(transfer_request)
    transfer_values, transfer_mass = _simulate(physiology, transfer_request, transfer_times, truth4, variant)
    cl = clearance_inputs(request)
    renal = cl["renal_clearance_l_h"] or 0.
    kp = float(physiology["rbc_kp_plasma"]) * math.exp(truth4[3])
    hct = float(physiology["hematocrit"])
    assert _digest(physiology) == original_physiology_sha and _digest(request) == original_request_sha
    datasets = {}
    for sampling, chosen in (("full", list(range(len(times)))), ("five", indices)):
        datasets[sampling] = {"request": deepcopy(request), "times_h": times[chosen].tolist(),
                              "observations_mg_l": observed[chosen].tolist()}
    result = {
        "version": VERSION, "case_id": cid, "scenario": scenario, "seed": int(seed), "log_sigma": sigma,
        "truth4": truth4, "generator_variant": variant,
        "generator_log_parameters": [0., 0., 0., .8] if variant == "exchange_shape" else list(truth4),
        "protected_truth_context": {"rbc_kp_plasma": kp, "blood_to_plasma_ratio": (1-hct)+hct*kp,
            "fraction_unbound_plasma": float(physiology["fraction_unbound_plasma"]),
            "renal_clearance_l_h": renal,
            "total_clearance_l_h": renal + (cl["total_clearance_l_h"]-renal)*math.exp(truth4[0]),
            "RBC_correction_required": scenario == "wrong_RBC",
            "candidate_family_contains_generator": variant == "blood_partition",
            "evidence_ledger_changed": False, "clinical_evidence": False},
        "calibration": datasets,
        "transfer": {"request": transfer_request, "times_h": transfer_times.tolist(),
                     "observations_mg_l": transfer_values.tolist(), "infusion_factor": factor,
                     "noise_added": False, "clinical_observations": False},
        "provenance": {"physiology_content_sha256": original_physiology_sha,
            "request_content_sha256": original_request_sha,
            "source_execution_manifest_sha256": physiology.get("source_execution_manifest_sha256"),
            "full_time_design_h": times.tolist(), "five_indices": indices,
            "five_observations_are_exact_full_subset": True,
            "noise_model": "independent additive zero-mean Gaussian on log concentration; median-unbiased lognormal measurements",
            "noise_stream": NOISE_STREAM, "compound_noise_stream": compound_stream,
            "noise_pairing": "Shared standardized noise within compound across scenarios/noise levels; independent compound streams",
            "transfer_assignment": "SHA256(case_id) first-byte parity; independent of noise seed",
            "fitted_clinical_parameters_used": False, "clinical_target_observations_read": False,
            "generator_calls": 2, "training_mass_balance": training_mass,
            "transfer_mass_balance": transfer_mass,
            "interpretation": "Known-truth synthetic mechanism test; engineering false activation, not clinical validation or a calibrated hypothesis test"}}
    json.dumps(result, allow_nan=False)
    return result


def calibration_input(case, sampling):
    """Return only a copied request and standard training document, without truth.

    This adapter uses a whitelist: scoring fields and scenario identities are
    never forwarded to the fitting or activation interface.
    """
    if sampling not in {"full", "five"} or case.get("version") != VERSION:
        raise ValueError("Unknown synthetic case version or sampling")
    visible = case["calibration"][sampling]
    request = deepcopy(visible["request"])
    rows = [r for r in request["evidence"] if r.get("parameter") == "molecular_weight"]
    if len(rows) != 1:
        raise ValueError("Exactly one molecular weight is required")
    mw = _number(rows[0].get("value"), "Molecular weight", positive=True)
    times, values = visible["times_h"], visible["observations_mg_l"]
    if len(times) != len(values) or (sampling == "five" and len(times) != 5):
        raise ValueError("Visible synthetic times and observations differ")
    training = {"compound_id": request["compound_id"], "human": deepcopy(request["human"]),
        "dosing": deepcopy(request["dosing"]), "matrix": "plasma", "time_unit": "min", "unit": "umol/l",
        "observations": [{"time_min": _number(t, "Visible time", positive=True)*60.,
                          "concentration_umol_per_l": _number(c, "Visible concentration", positive=True)*1000./mw}
                         for t, c in zip(times, values)]}
    return request, training
