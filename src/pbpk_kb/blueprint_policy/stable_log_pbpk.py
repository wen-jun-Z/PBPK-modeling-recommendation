"""Log-concentration adapter for the unchanged vascular--tissue PBPK equations.

Ordinary predictions retain the frozen solver's exact floating-point values.
For underflow or unreliable small tails, the absorbing elimination state is
removed and the remaining matrix is shifted by its spectral abscissa. This is
the identity exp(T*t)=exp(alpha*t)*exp((T-alpha*I)*t), not a concentration floor,
parameter constraint, new physiological assumption, or likelihood adjustment.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigvals, expm

from . import blood_partition_calibration as blood
from . import permeability_model as exchange

VERSION = "unchanged-pbpk-stable-log-propagation-1"
# These are numerical (prediction-only) criteria, never observation thresholds.
ABSOLUTE_CONCENTRATION_TRIGGER = 1e-250
RELATIVE_TRAJECTORY_TRIGGER = 1e-12


def shifted_log_observation(system, times_h, dose_mg, infusion_duration_h):
    """Evaluate the same positive linear system directly in log concentration.

The last state must be an absorbing mass ledger named Eliminated. Concentration
at a genuinely zero-exposure time is -inf; positive values below float64's range
remain finite logs. Matrix coefficients and observation weights are unchanged.
"""
    times = np.asarray(times_h, float)
    if (times.ndim != 1 or not times.size or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(np.diff(times) < 0)):
        raise ValueError("Sorted finite nonnegative times required")
    dose, duration = float(dose_mg), float(infusion_duration_h)
    if not all(math.isfinite(v) and v >= 0 for v in (dose, duration)):
        raise ValueError("Finite nonnegative dose and duration required")
    matrix = np.asarray(system.matrix, float)
    inputs = np.asarray(system.input_vector, float)
    observer = np.asarray(system.observation_vector, float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] < 2 or inputs.shape != (len(matrix),)
            or observer.shape != inputs.shape or not np.isfinite(matrix).all()
            or not np.isfinite(inputs).all() or not np.isfinite(observer).all()
            or system.state_names[-1] != "Eliminated"):
        raise ValueError("Finite amount matrix with a final elimination ledger required")
    if np.any(inputs < 0) or np.any(observer < 0) or inputs[-1] != 0 or observer[-1] != 0:
        raise ValueError("Nonnegative active inputs and concentration observer required")
    if not math.isclose(float(inputs.sum()), 1., rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("Dose input must sum to one")
    if np.any(matrix[:, -1] != 0):
        raise ValueError("Elimination ledger must be absorbing")
    offdiag = matrix.copy()
    np.fill_diagonal(offdiag, 0.)
    if np.any(offdiag < 0):
        raise ValueError("Positive amount dynamics require a Metzler matrix")
    column_scale = np.maximum(np.abs(matrix).sum(axis=0), 1.)
    mass_residual = float(np.max(np.abs(matrix.sum(axis=0))/column_scale))
    if mass_residual > 1e-12:
        raise ValueError("Amount matrix does not conserve the elimination ledger")
    if dose == 0:
        return np.full(times.shape, -np.inf), dict(method="shifted_active_expm",
            alpha_per_h=None, matrix_mass_residual=mass_residual, matrix_unchanged=True)

    active = matrix[:-1, :-1]
    h = observer[:-1]
    b = inputs[:-1]
    alpha = float(np.max(eigvals(active).real))
    if not math.isfinite(alpha):
        raise ArithmeticError("Nonfinite active-matrix spectral shift")
    shifted = active-alpha*np.eye(len(active))
    if duration > 0:
        augmented = np.zeros((len(active)+1, len(active)+1))
        augmented[:-1, :-1] = active
        augmented[:-1, -1] = dose/duration*b
        initial = np.zeros(len(active)+1)
        initial[-1] = 1.
        at_end = (expm(augmented*duration)@initial)[:-1]
    else:
        at_end = dose*b
    if not np.isfinite(at_end).all() or np.any(at_end < 0):
        raise ArithmeticError("Invalid positive-system infusion endpoint")

    values = []
    for t in times:
        if duration > 0 and t <= duration:
            amounts = (expm(augmented*t)@initial)[:-1]
            log_scale = 0.
        else:
            elapsed = float(t-duration)
            amounts = expm(shifted*elapsed)@at_end
            log_scale = alpha*elapsed
        if not np.isfinite(amounts).all() or np.any(amounts < 0):
            raise ArithmeticError("Invalid positive-system shifted propagation")
        concentration = float(h@amounts)
        if concentration == 0:
            if t > 0 and np.any(h > 0):
                raise ArithmeticError("Zero projected concentration after shifted propagation")
            values.append(-np.inf)
        elif concentration > 0 and math.isfinite(concentration):
            values.append(math.log(concentration)+log_scale)
        else:
            raise ArithmeticError("Invalid projected concentration")
    return np.asarray(values), dict(method="shifted_active_expm", alpha_per_h=alpha,
        matrix_mass_residual=mass_residual, matrix_unchanged=True)


def predict_log(physiology, request, times_h, log_parameters, *, return_diagnostics=False):
    """Return log mg/L using the frozen predictions unless a tail is unreliable.

Triggers: a nonpositive prediction at positive time; any positive prediction
below 1e-250 mg/L; a positive-time dynamic range greater than 1e12; or clipped
negative states in the old solver. Every trigger uses simulations alone.
"""
    times = np.asarray(times_h, float)
    reason = []
    original = None
    try:
        original = blood.simulate(physiology, request, times_h, log_parameters)
    except ArithmeticError as exc:
        reason.append("original_propagation_error: "+str(exc))
    if original is not None:
        concentrations = np.asarray(original["concentration_mg_l"], float)
        positive_time = concentrations[times > 0]
        if not np.isfinite(concentrations).all() or np.any(positive_time <= 0):
            reason.append("nonpositive_or_nonfinite_concentration")
        positive = positive_time[positive_time > 0]
        if len(positive) and np.min(positive) < ABSOLUTE_CONCENTRATION_TRIGGER:
            reason.append("absolute_underflow_guard")
        if len(positive) and np.min(positive) < np.max(positive)*RELATIVE_TRAJECTORY_TRIGGER:
            reason.append("relative_tail_guard")
        if original["mass_balance"]["roundoff_negative_values_clipped"]:
            reason.append("clipped_negative_states")
        if not reason:
            with np.errstate(divide="ignore"):
                result = np.log(concentrations)
            diag = dict(method="frozen_solver_exact", fallback_reasons=[],
                original_solver=original["solver"], original_mass_balance=original["mass_balance"],
                matrix_unchanged=True, version=VERSION)
            return (result, diag) if return_diagnostics else result

    x = np.asarray(log_parameters, float)
    derived = blood.derive_physiology(physiology, request, float(x[3]))
    clearance = blood.clearance_inputs(request)
    system = exchange.build_system(derived,
        total_clearance_l_h=clearance["total_clearance_l_h"],
        renal_clearance_l_h=clearance["renal_clearance_l_h"],
        log_parameters=dict(zip(blood.PARAMETERS[:3], x[:3].tolist())), observation="auto")
    result, diag = shifted_log_observation(system, times_h,
        request["dosing"]["dose_mg"], request["dosing"]["infusion_minutes"]/60.)
    diag.update(version=VERSION, fallback_reasons=reason,
        original_solver=original["solver"] if original else None,
        original_mass_balance=original["mass_balance"] if original else None)
    return (result, diag) if return_diagnostics else result
