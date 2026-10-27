"""One continuous exchange-shape axis around the frozen flow-exchange model.

The baseline organ return time constants define a flow-weighted, centered log
contrast. Its coefficient changes only reversible vascular/tissue conductances;
circulation, capacities, observer and aggregate central clearance are retained.
The contrast is recomputed from each supplied subject before calibration scales
are applied. It does not use observations, clinical endpoints or fitted values.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np

from . import perfusion_model as perfusion
from . import robust_exchange_model as baseline

VERSION = "linear-adaptive-exchange-model-1"
PHYSIOLOGY_VERSION = perfusion.PHYSIOLOGY_VERSION
PARAMETERS = {"adaptive_exchange": (*baseline.PARAMETERS["evidence_flexible"],
                                     "exchange_heterogeneity_scale")}
# Fitting bounds belong to the calibrator. This wider limit is only an
# engineering guard against extreme exponentiation, not physiological evidence.
MAX_ABSOLUTE_ALPHA = 2.
EXCHANGE_INTERPRETATION = (
    "PS_i = Q_effective_i * exp(log_exchange_scale + alpha*h_i); "
    "h_i = log(tau0_i/(1 h)) - sum_j(w_j*log(tau0_j/(1 h))); "
    "tau0_i = K_extra0_plasma_i/(BP*Q_effective_i); w_i = Q_effective_i/sum_j(Q_effective_j); "
    "alpha = log(exchange_heterogeneity_scale); baseline capacities are uncalibrated; "
    "effective exchange hypothesis, not measured membrane permeability"
)


def build_system(physiology, *, total_clearance_l_h, renal_clearance_l_h=None,
                 family="adaptive_exchange", log_parameters=None, observation="auto"):
    """Build a conservative Metzler generator; alpha=0 is exactly the baseline.

    Negative alpha expands the spread of tissue return time constants; positive
    alpha compresses it. Neither these tissue constants nor the fourth fitted
    parameter identify the system's terminal half-life without sufficient data.
    """
    if family not in PARAMETERS:
        raise ValueError("Unknown adaptive-exchange model family")
    if log_parameters is None:
        log_parameters = {}
    if not isinstance(log_parameters, Mapping) or set(log_parameters) - set(PARAMETERS[family]):
        raise ValueError("Unexpected log parameter for this model family")
    alpha = perfusion._number(log_parameters.get("exchange_heterogeneity_scale", 0.),
                              "log_parameters.exchange_heterogeneity_scale")
    if abs(alpha) > MAX_ABSOLUTE_ALPHA:
        raise ValueError("Exchange heterogeneity alpha exceeds the engineering safety range")
    original_logs = {name: log_parameters.get(name, 0.)
                     for name in baseline.PARAMETERS["evidence_flexible"]}
    reference = baseline.build_system(physiology, total_clearance_l_h=total_clearance_l_h,
        renal_clearance_l_h=renal_clearance_l_h, family="evidence_flexible",
        log_parameters=original_logs, observation=observation)
    bp = reference.metadata["blood_to_plasma_ratio"]
    organs = {organ["name"]: organ for organ in physiology["organs"]}
    exchanges = deepcopy(reference.metadata["exchange_edges"])
    flows = [edge["effective_flow_l_h"] for edge in exchanges]
    total_flow = perfusion._number(math.fsum(flows), "exchange_centering_flow_l_h", positive=True)
    weights = [q / total_flow for q in flows]
    log_times = []
    for edge in exchanges:
        organ = organs[edge["organ"]]
        # Explicitly use the uncalibrated input, not the reference's scaled
        # capacities: x_K must not change h during optimization.
        k0 = perfusion._number(organ["extravascular_capacity_l"],
                              f"{edge['organ']}.extravascular_capacity_l", positive=True)
        log_times.append(math.log(k0) - math.log(bp) - math.log(edge["effective_flow_l_h"]))
    center = math.fsum(w * value for w, value in zip(weights, log_times))
    contrasts = [value - center for value in log_times]
    index = {name: i for i, name in enumerate(reference.state_names)}
    matrix = reference.matrix if alpha == 0. else reference.matrix.copy()
    for edge, weight, log_time, contrast in zip(exchanges, weights, log_times, contrasts):
        try:
            multiplier = math.exp(alpha * contrast)
            time_constant = math.exp(log_time)
            ps = perfusion._number(edge["exchange_conductance_l_h"] * multiplier,
                                  f"{edge['organ']}.exchange_conductance_l_h", positive=True)
        except OverflowError as exc:
            raise ValueError("Adaptive exchange rate or baseline time constant overflow") from exc
        vi, ei = index[edge["vascular_state"]], index[edge["extravascular_state"]]
        if alpha != 0.:
            matrix[ei, vi] = ps / reference.blood_equivalent_capacities_l[vi]
            matrix[vi, ei] = ps / reference.blood_equivalent_capacities_l[ei]
        edge.update(exchange_conductance_l_h=ps,
            extravascular_return_rate_per_h=ps / reference.blood_equivalent_capacities_l[ei],
            alpha=alpha, baseline_return_time_constant_h=time_constant,
            centering_flow_weight=weight, centered_log_time_constant=contrast,
            exchange_heterogeneity_multiplier=multiplier)
    if alpha != 0.:
        # Recompute departure rates from positive transfers instead of adding
        # a conductance difference, which can cancel at very small PS.
        np.fill_diagonal(matrix, 0.)
        np.fill_diagonal(matrix, -matrix.sum(axis=0))
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite adaptive-exchange generator")
        matrix.setflags(write=False)
    metadata = {**deepcopy(reference.metadata), "version": VERSION, "family": family,
        "parameters": {**reference.metadata["parameters"], "exchange_heterogeneity_scale": math.exp(alpha)},
        "log_parameters": {**reference.metadata["log_parameters"], "exchange_heterogeneity_scale": alpha},
        "exchange_edges": exchanges, "exchange_alpha": alpha,
        "exchange_interpretation": EXCHANGE_INTERPRETATION,
        "exchange_heterogeneity": {
            "alpha": alpha, "centering": "effective_flow_weighted_log_unscaled_return_time_h",
            "weighted_log_time_center": center, "reference_time_h": 1.,
            "baseline_partition_scale": 1., "recomputed_for_each_supplied_physiology": True,
            "calibrated_partition_scale_used_in_contrast": False,
            "target_observations_used": False,
            "alpha_zero_reproduces_frozen_flow_exchange": True,
            "engineering_alpha_range": [-MAX_ABSOLUTE_ALPHA, MAX_ABSOLUTE_ALPHA],
            "tissue_time_constants_are_not_system_terminal_half_life": True},
        "structural_scope": "three blood pools and 15 vascular/extravascular pairs; one shared exchange-shape axis"}
    return perfusion.PerfusionSystem(matrix, reference.state_names, reference.input_vector,
        reference.observation_vector, reference.venous_plasma_vector,
        reference.blood_equivalent_capacities_l, metadata)


def simulate(physiology, *, times_h, dose_mg, infusion_duration_h, total_clearance_l_h,
             renal_clearance_l_h=None, family="adaptive_exchange", log_parameters=None,
             observation="auto", return_amounts=False):
    """Propagate with the frozen spectral/exponential solver and mass audit."""
    times = np.asarray(times_h)
    if (times.ndim != 1 or not times.size or times.dtype.kind not in "iuf"
            or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) < 0)):
        raise ValueError("times_h must be a nonempty nonnegative sorted numeric vector")
    times = times.astype(float)
    dose = perfusion._number(dose_mg, "dose_mg", nonnegative=True)
    duration = perfusion._number(infusion_duration_h, "infusion_duration_h", nonnegative=True)
    if not isinstance(return_amounts, bool):
        raise ValueError("return_amounts must be boolean")
    system = build_system(physiology, total_clearance_l_h=total_clearance_l_h,
        renal_clearance_l_h=renal_clearance_l_h, family=family,
        log_parameters=log_parameters, observation=observation)
    delivered = np.full_like(times, dose) if duration == 0 else dose * np.minimum(times / duration, 1.)
    values, condition = perfusion._spectral_states(system, times, dose, duration)
    method = "vectorized_spectral"
    tolerance = 1e-8 * max(dose, 1.)
    if (values is None or values.min() < -tolerance
            or np.max(np.abs(values.sum(axis=1) - delivered)) > tolerance):
        values = perfusion._expm_states(system, times, dose, duration)
        method = "matrix_exponential_fallback"
    if not np.isfinite(values).all() or values.min() < -tolerance:
        raise ArithmeticError("Linear propagation produced invalid or materially negative amounts")
    raw_negative = int(np.sum(values < 0))
    values = np.maximum(values, 0.)
    values[times == 0] = dose * system.input_vector if duration == 0 else 0.
    absolute_error = float(np.max(np.abs(values.sum(axis=1) - delivered)))
    relative_error = absolute_error / max(dose, np.finfo(float).tiny)
    if absolute_error > tolerance:
        raise ArithmeticError("Linear propagation failed mass conservation")
    result = {"status": "predicted", "version": VERSION, "time_h": times,
        "concentration_mg_l": values @ system.observation_vector,
        "venous_plasma_mg_l": values @ system.venous_plasma_vector,
        "observation": system.metadata["observation"],
        "observation_weights": system.metadata["observation_weights"],
        "parameters": system.metadata["parameters"],
        "effective_total_clearance_l_h": system.metadata["effective_total_clearance_l_h"],
        "mass_balance_relative_error": relative_error,
        "mass_balance": {"maximum_absolute_error_mg": absolute_error,
            "maximum_relative_error": relative_error, "dose_mg": dose,
            "final_delivered_mg": float(delivered[-1]), "final_in_body_mg": float(values[-1, :-1].sum()),
            "final_eliminated_mg": float(values[-1, -1]), "roundoff_negative_values_clipped": raw_negative},
        "solver": {"method": method, "eigenvector_condition_number": condition},
        "model": system.metadata}
    if return_amounts:
        result.update(state_names=system.state_names[:-1], amounts_mg=values[:, :-1],
                      eliminated_mg=values[:, -1], delivered_mg=delivered)
    return result
