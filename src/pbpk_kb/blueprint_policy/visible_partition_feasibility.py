"""Visible-data mass bound for an unmeasured RBC calibration floor.

For a nonnegative amount state A and plasma observer h, C=h@A cannot
exceed delivered_dose*max(h). This necessary condition does not establish
predictive accuracy. When the frozen relative RBC floor makes even a visible
development observation impossible, extend that floor to include K_RBC=1.
No target observation, hidden development point, drug rule or new axis is used.
K_RBC=1 is a neutral engineering reference, not measured red-cell binding.
"""
from __future__ import annotations

from copy import deepcopy
import math
from types import FunctionType

import numpy as np

from . import blood_partition_calibration as blood
from . import distribution_calibration_optimizer as frozen
from . import permeability_model as exchange
from .perfusion_calibration import clearance_inputs

VERSION = "visible-mass-bound-rbc-floor-1"
ORIGINAL_MINIMUM_SCALE = .05
NEUTRAL_RBC_PARTITION = 1.
RELATIVE_TOLERANCE = 1e-10


def _bound(physiology, request, times, observed, scale):
    derived = blood.derive_physiology(physiology, request, math.log(scale))
    cl = clearance_inputs(request)
    system = exchange.build_system(derived,
        total_clearance_l_h=cl["total_clearance_l_h"],
        renal_clearance_l_h=cl["renal_clearance_l_h"], observation="auto")
    observer = np.asarray(system.observation_vector)
    if not np.isfinite(observer).all() or np.any(observer < 0) or observer.max() <= 0:
        raise ValueError("Mass bound requires a finite nonnegative plasma observer")
    dose = float(request["dosing"]["dose_mg"])
    duration = float(request["dosing"]["infusion_minutes"]) / 60.
    delivered = np.full_like(times, dose) if duration == 0 else dose * np.minimum(times / duration, 1.)
    upper = delivered * float(observer.max())
    violations = observed > upper * (1. + RELATIVE_TOLERANCE)
    rows = [{"time_h": float(t), "observed_mg_l": float(o),
             "delivered_dose_mg": float(m), "concentration_upper_bound_mg_l": float(u),
             "observed_over_bound": float(o / u) if u > 0 else None,
             "violates_mass_bound": bool(v)}
            for t, o, m, u, v in zip(times, observed, delivered, upper, violations)]
    return {"minimum_rbc_scale": float(scale), "effective_rbc_partition": derived["rbc_kp_plasma"],
            "blood_to_plasma_ratio": derived["blood_to_plasma_ratio"],
            "observation": system.metadata["observation"],
            "observation_weights": deepcopy(system.metadata["observation_weights"]),
            "maximum_observer_coefficient_l_inverse": float(observer.max()),
            "visible_points": rows, "violations": int(violations.sum()),
            "maximum_observed_over_bound": max((r["observed_over_bound"] for r in rows
                                                if r["observed_over_bound"] is not None), default=None)}


def feasibility(physiology, request, training):
    """Audit only the exact visible four/five-point training document supplied.

    A missing bound violation is not evidence of goodness of fit. A remaining
    violation after the extension is retained as a model/data incompatibility.
    Existing qualified measured B/P or RBC evidence always blocks repair.
    """
    times, observed = frozen._visible(request, training)
    checked = blood.derive_physiology(physiology, request, 0.)
    policy = checked["blood_partition_audit"]["policy"]
    protected = bool(policy["explicit_measured_records"])
    old_scale = 1. if protected else ORIGINAL_MINIMUM_SCALE
    before = _bound(physiology, request, times, observed, old_scale)
    proposed = min(old_scale, NEUTRAL_RBC_PARTITION / physiology["rbc_kp_plasma"])
    repair = bool(before["violations"] and not protected and proposed < old_scale)
    lower_scale = proposed if repair else old_scale
    after = _bound(physiology, request, times, observed, lower_scale) if repair else deepcopy(before)
    reason = ("qualified_measured_partition_is_protected" if protected else
              "visible_concentration_exceeds_family_mass_bound" if repair else
              "neutral_reference_cannot_extend_the_existing_floor" if before["violations"] else
              "no_proven_visible_mass_conflict")
    return {"version": VERSION, "repair_required": repair, "reason": reason,
            "protected_measured_partition": protected,
            "protected_records": deepcopy(policy["explicit_measured_records"]),
            "native_rbc_partition": physiology["rbc_kp_plasma"],
            "original_minimum_rbc_scale": old_scale, "minimum_rbc_scale": lower_scale,
            "neutral_reference_rbc_partition": NEUTRAL_RBC_PARTITION,
            "before": before, "after": after,
            "remaining_visible_mass_conflict": bool(after["violations"]),
            "target_observations_used": False, "hidden_development_observations_used": False,
            "new_fitted_parameters": 0, "native_files_modified": False,
            "interpretation": "Necessary mass feasibility only; effective fitted RBC partition is not a measurement"}


def initialization(request, physiology, variant, starts, audit):
    """Frozen guesses/bounds except the proved unmeasured lower RBC bound.

    Preserve all old starts, except the fourth coordinate of the last start,
    which explores the newly admitted neutral RBC reference. No warm start.
    """
    lower, upper, guesses, absolute, cl = frozen._initialization(request, physiology, variant, starts)
    if variant == "blood_partition" and audit["repair_required"]:
        lower[3] = math.log(audit["minimum_rbc_scale"])
        guesses[-1, 3] = math.log(NEUTRAL_RBC_PARTITION / physiology["rbc_kp_plasma"])
    return lower, upper, guesses, absolute, cl


def fit(predict, request, training, physiology, *, variant="blood_partition", budget=16384,
        starts=16, before_trial=None):
    """The exact frozen objective/solver/budget, with a visible-only bound repair.

    The predictor remains the existing four-axis blood partition simulator.
    FunctionType gives this call a private initializer; old globals are never
    changed. Unaffected tasks take the original code path exactly.
    """
    if variant != "blood_partition":
        raise ValueError("Visible RBC feasibility repair supports blood_partition only")
    audit = feasibility(physiology, request, training)
    if audit["protected_measured_partition"]:
        raise ValueError("Four-axis fitting cannot modify measured partition; use a separately validated fixed-RBC fitter")
    initializer = lambda req, phys, kind, n: initialization(req, phys, kind, n, audit)
    namespace = {**frozen.fit.__globals__, "_initialization": initializer}
    execute = FunctionType(frozen.fit.__code__, namespace, frozen.fit.__name__,
                           frozen.fit.__defaults__, frozen.fit.__closure__)
    execute.__kwdefaults__ = deepcopy(frozen.fit.__kwdefaults__)
    result = execute(predict, request, training, physiology, variant=variant, budget=budget,
                     starts=starts, before_trial=before_trial)
    result["optimizer_version"] = result["version"]
    result["version"] = VERSION
    result["partition_feasibility"] = audit
    result["parameterization_changed"] = audit["repair_required"]
    return result
