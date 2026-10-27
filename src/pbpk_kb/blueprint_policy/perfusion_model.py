"""Linear perfusion-limited PBPK using explicitly supplied human physiology.

State amounts and doses are mg, time is h, flow/clearance is L/h. Organ
partition capacities are supplied in plasma-equivalent litres; the vascular
capacity is retained while only extravascular capacity is scaled. Elimination
is an aggregate systemic plasma-clearance sink, not a validated organ route.
No drug database, clinical curve, fitted model, or native process is accessed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
from scipy.linalg import eig, expm, solve

VERSION = "linear-perfusion-model-1"
PHYSIOLOGY_VERSION = "native-perfusion-physiology-1"
FAST_TISSUES = frozenset({"Heart", "Lung", "Liver", "Kidney", "Spleen", "Stomach",
                          "SmallIntestine", "LargeIntestine"})
SLOW_TISSUES = frozenset({"Bone", "Brain", "Fat", "Gonads", "Muscle", "Pancreas", "Skin"})
PARAMETERS = {
    "global": ("clearance_scale", "partition_scale"),
    "grouped": ("clearance_scale", "fast_partition_scale", "slow_partition_scale"),
}
GROUPING_INTERPRETATION = (
    "Predetermined two-group simplification inspired by tissue composition; "
    "not an exact reproduction of Yau's four groups or a target-selected grouping"
)
_BLOOD_NAMES = {"arterial": "ArterialBlood", "venous": "VenousBlood", "portal": "PortalBlood"}
_ELIMINATED = "Eliminated"


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    value = float(value)
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"Inconsistent {label}: {actual} versus {expected} L/h")


@dataclass(frozen=True)
class PerfusionSystem:
    """Generator includes one cumulative elimination state as its last row."""

    matrix: np.ndarray
    state_names: tuple[str, ...]
    input_vector: np.ndarray
    observation_vector: np.ndarray
    venous_plasma_vector: np.ndarray
    blood_equivalent_capacities_l: np.ndarray
    metadata: dict


def build_system(physiology, *, total_clearance_l_h, renal_clearance_l_h=None,
                 family="global", log_parameters=None, observation="auto"):
    """Construct a mass-conserving linear system from an exported physiology.

    ``log_parameters`` holds natural logarithms of the named positive scales.
    Missing active scales default to one. ``renal_clearance_l_h`` is an
    externally qualified net renal component of *total* plasma clearance; it
    is held fixed while the nonrenal remainder is scaled, but both are removed
    through the declared aggregate systemic sink.
    """
    if not isinstance(physiology, Mapping) or physiology.get("version") != PHYSIOLOGY_VERSION:
        raise ValueError("Unsupported perfusion physiology schema")
    if family not in PARAMETERS:
        raise ValueError("Unknown perfusion model family")
    if log_parameters is None:
        log_parameters = {}
    if not isinstance(log_parameters, Mapping) or set(log_parameters) - set(PARAMETERS[family]):
        raise ValueError("Unexpected log parameter for this model family")
    scales = {}
    for name in PARAMETERS[family]:
        value = _number(log_parameters.get(name, 0.), f"log_parameters.{name}")
        try:
            scales[name] = _number(math.exp(value), name, positive=True)
        except OverflowError as exc:
            raise ValueError("Parameter scale overflow") from exc
    bp = _number(physiology["blood_to_plasma_ratio"], "blood_to_plasma_ratio", positive=True)
    clearance = _number(total_clearance_l_h, "total_clearance_l_h", nonnegative=True)
    renal = (0. if renal_clearance_l_h is None else
             _number(renal_clearance_l_h, "renal_clearance_l_h", nonnegative=True))
    if renal > clearance:
        raise ValueError("Qualified renal clearance exceeds the total clearance input")
    nonrenal = (clearance - renal) * scales["clearance_scale"]
    effective = _number(renal + nonrenal, "effective clearance", nonnegative=True)

    blood = physiology["blood"]
    organs = physiology["organs"]
    if not isinstance(blood, Mapping) or not isinstance(organs, (list, tuple)) or not organs:
        raise ValueError("Blood and organ compartments must be explicitly supplied")
    organ_names = [organ["name"] for organ in organs]
    if (any(not isinstance(name, str) or not name for name in organ_names)
            or len(set(organ_names)) != len(organ_names)
            or set(organ_names) & (set(_BLOOD_NAMES.values()) | {_ELIMINATED})):
        raise ValueError("Organ names must be unique and distinct from blood states")
    if not {"Lung", "Liver"} <= set(organ_names):
        raise ValueError("Lung and liver compartments are required")
    if family == "grouped" and set(organ_names) - (FAST_TISSUES | SLOW_TISSUES):
        raise ValueError("An organ is outside the predetermined tissue groups")

    circulation = physiology["circulation"]
    cardiac = _number(circulation["cardiac_flow_l_h"], "cardiac_flow_l_h", positive=True)
    hepatic_arterial = _number(circulation["hepatic_arterial_flow_l_h"], "hepatic_arterial_flow_l_h", nonnegative=True)
    portal_flow = _number(circulation["portal_flow_l_h"], "portal_flow_l_h", nonnegative=True)
    liver_total = _number(circulation["liver_total_flow_l_h"], "liver_total_flow_l_h", nonnegative=True)
    portal_names = circulation["portal_contributors"]
    if (not isinstance(portal_names, (list, tuple)) or len(set(portal_names)) != len(portal_names)
            or not set(portal_names) <= set(organ_names) - {"Lung", "Liver"}):
        raise ValueError("Portal contributors must be distinct systemic organs")
    if portal_flow > 0 and "portal" not in blood:
        raise ValueError("Positive portal flow requires an explicit portal blood volume")
    blood_keys = [key for key in _BLOOD_NAMES if key in blood]
    if not {"arterial", "venous"} <= set(blood_keys):
        raise ValueError("Arterial and venous blood compartments are required")
    names = tuple([_BLOOD_NAMES[key] for key in blood_keys] + organ_names + [_ELIMINATED])
    index = {name: i for i, name in enumerate(names)}
    capacities, organ_capacity, flow = [], {}, {}
    for key in blood_keys:
        capacities.append(_number(blood[key]["volume_l"], f"{key}.volume_l", positive=True))
    for organ in organs:
        name = organ["name"]
        vascular = _number(organ["vascular_capacity_l"], f"{name}.vascular_capacity_l", nonnegative=True)
        extravascular = _number(organ["extravascular_capacity_l"], f"{name}.extravascular_capacity_l", nonnegative=True)
        scale = (scales["partition_scale"] if family == "global" else
                 scales["fast_partition_scale" if name in FAST_TISSUES else "slow_partition_scale"])
        plasma_capacity = _number(vascular + extravascular * scale, f"{name}.plasma_capacity_l", positive=True)
        capacities.append(plasma_capacity / bp)
        organ_capacity[name] = {"vascular_capacity_l": vascular,
            "baseline_extravascular_capacity_l": extravascular,
            "extravascular_scale": scale, "plasma_equivalent_capacity_l": plasma_capacity,
            "blood_equivalent_capacity_l": plasma_capacity / bp}
        flow[name] = _number(organ["flow_l_h"], f"{name}.flow_l_h", nonnegative=True)
    _close(flow["Lung"], cardiac, "pulmonary versus cardiac flow")
    _close(flow["Liver"], hepatic_arterial, "liver direct arterial flow")
    _close(sum(flow[name] for name in portal_names), portal_flow, "summed portal inflow")
    _close(hepatic_arterial + portal_flow, liver_total, "total hepatic flow")
    _close(sum(q for name, q in flow.items() if name != "Lung"), cardiac, "systemic arterial outflow")

    n = len(names)
    matrix = np.zeros((n, n))
    input_vector = np.zeros(n)
    input_vector[index["VenousBlood"]] = 1.
    edges = []

    def connect(source, target, q):
        if q == 0:
            return
        rate = q / capacities[index[source]]
        matrix[index[source], index[source]] -= rate
        matrix[index[target], index[source]] += rate
        edges.append({"source": source, "target": target, "flow_l_h": q})

    connect("VenousBlood", "Lung", cardiac)
    connect("Lung", "ArterialBlood", cardiac)
    for name in organ_names:
        if name == "Lung":
            continue
        connect("ArterialBlood", name, flow[name])
        if name in portal_names:
            connect(name, "PortalBlood", flow[name])
        elif name == "Liver":
            connect(name, "VenousBlood", liver_total)
        else:
            connect(name, "VenousBlood", flow[name])
    if "portal" in blood:
        connect("PortalBlood", "Liver", portal_flow)

    venous_vector = np.zeros(n)
    venous_vector[index["VenousBlood"]] = 1 / (capacities[index["VenousBlood"]] * bp)
    loss_rate = effective * venous_vector[index["VenousBlood"]]
    matrix[index["VenousBlood"], index["VenousBlood"]] -= loss_rate
    matrix[index[_ELIMINATED], index["VenousBlood"]] += loss_rate

    supplied_weights = physiology.get("peripheral_venous_weights")
    if observation not in {"auto", "venous", "native_peripheral_venous"}:
        raise ValueError("Unknown observation type")
    use_peripheral = observation == "native_peripheral_venous" or observation == "auto" and bool(supplied_weights)
    observation_vector = np.zeros(n)
    if use_peripheral:
        if not isinstance(supplied_weights, Mapping) or not supplied_weights:
            raise ValueError("Native peripheral observation requires exported fixed weights")
        weights = {name: _number(weight, f"observer.{name}", nonnegative=True)
                   for name, weight in supplied_weights.items()}
        if set(weights) - (set(organ_names) | {"ArterialBlood"}):
            raise ValueError("Peripheral observer refers to an unsupported compartment")
        if not math.isclose(sum(weights.values()), 1., rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("Peripheral observer weights must sum to one without refitting")
        for name, weight in weights.items():
            observation_vector[index[name]] = weight / (capacities[index[name]] * bp)
        observation_name = "native_peripheral_venous"
    else:
        weights = {"VenousBlood": 1.}
        observation_vector[:] = venous_vector
        observation_name = "venous"
    for array in (matrix, input_vector, observation_vector, venous_vector):
        array.setflags(write=False)
    capacity_array = np.asarray(capacities)
    capacity_array.setflags(write=False)
    metadata = {
        "version": VERSION, "family": family, "parameters": scales,
        "log_parameters": {name: float(log_parameters.get(name, 0.)) for name in PARAMETERS[family]},
        "blood_to_plasma_ratio": bp, "organ_capacities": organ_capacity, "flow_edges": edges,
        "effective_total_clearance_l_h": effective,
        "clearance_components_l_h": {"qualified_renal_fixed": renal, "scaled_nonrenal": nonrenal},
        "clearance_interpretation": "aggregate_systemic_venous_plasma_sink_not_validated_organ_elimination",
        "renal_component_provided": renal_clearance_l_h is not None,
        "observation": observation_name, "observation_weights": weights,
        "observation_limitation": (None if use_peripheral else
            "Central venous plasma is used; no peripheral sampling-site equivalence is asserted"),
        "grouping_interpretation": GROUPING_INTERPRETATION if family == "grouped" else "one_global_extravascular_scale",
        "tissue_groups": {"fast": sorted(FAST_TISSUES), "slow": sorted(SLOW_TISSUES)} if family == "grouped" else None,
        "units": {"amount": "mg", "time": "h", "flow": "L/h", "clearance": "L/h", "concentration": "mg/L"},
    }
    return PerfusionSystem(matrix, names, input_vector, observation_vector, venous_vector,
                           capacity_array, metadata)


def _spectral_states(system, times, dose, duration):
    """Vectorized exact propagation; return None for an unsafe eigensystem."""
    eigenvalues, vectors = eig(system.matrix)
    condition = float(np.linalg.cond(vectors))
    if not math.isfinite(condition) or condition > 1e10:
        return None, condition
    coefficients = solve(vectors, system.input_vector, assume_a="gen", check_finite=False)
    if duration == 0:
        factors = dose * np.exp(times[:, None] * eigenvalues[None, :])
    else:
        during = np.minimum(times, duration)
        after = np.maximum(times - duration, 0.)
        factors = np.empty((len(times), len(eigenvalues)), dtype=complex)
        nonzero = eigenvalues != 0
        factors[:, nonzero] = np.expm1(during[:, None] * eigenvalues[None, nonzero]) / eigenvalues[None, nonzero]
        factors[:, ~nonzero] = during[:, None]
        factors *= (dose / duration) * np.exp(after[:, None] * eigenvalues[None, :])
    values = (factors * coefficients[None, :]) @ vectors.T
    if not np.isfinite(values).all() or np.max(np.abs(values.imag)) > 1e-9 * max(dose, 1.):
        return None, condition
    return values.real, condition


def _expm_states(system, times, dose, duration):
    """Stable fallback, using an augmented constant-input state during infusion."""
    matrix = system.matrix
    if duration == 0:
        return np.array([expm(matrix * t) @ (dose * system.input_vector) for t in times])
    augmented = np.zeros((len(matrix) + 1, len(matrix) + 1))
    augmented[:-1, :-1] = matrix
    augmented[:-1, -1] = (dose / duration) * system.input_vector
    start = np.zeros(len(matrix) + 1)
    start[-1] = 1.
    at_end = (expm(augmented * duration) @ start)[:-1]
    return np.array([(expm(augmented * t) @ start)[:-1] if t <= duration else
                     expm(matrix * (t - duration)) @ at_end for t in times])


def simulate(physiology, *, times_h, dose_mg, infusion_duration_h, total_clearance_l_h,
             renal_clearance_l_h=None, family="global", log_parameters=None,
             observation="auto", return_amounts=False):
    """Return sampled plasma mg/L and a mass-balance audit (arrays stay arrays).

    Zero infusion duration denotes an IV bolus into venous blood at t=0.
    Positive duration denotes a constant infusion of ``dose_mg / duration``.
    Duplicate sample times are allowed; times must be nonnegative and sorted.
    Amount arrays are optional to avoid retaining large tissue grids in fits.
    """
    times = np.asarray(times_h)
    if (times.ndim != 1 or not times.size or times.dtype.kind not in "iuf"
            or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) < 0)):
        raise ValueError("times_h must be a nonempty nonnegative sorted numeric vector")
    times = times.astype(float)
    dose = _number(dose_mg, "dose_mg", nonnegative=True)
    duration = _number(infusion_duration_h, "infusion_duration_h", nonnegative=True)
    if not isinstance(return_amounts, bool):
        raise ValueError("return_amounts must be boolean")
    system = build_system(physiology, total_clearance_l_h=total_clearance_l_h,
        renal_clearance_l_h=renal_clearance_l_h, family=family,
        log_parameters=log_parameters, observation=observation)
    delivered = np.full_like(times, dose) if duration == 0 else dose * np.minimum(times / duration, 1.)
    values, condition = _spectral_states(system, times, dose, duration)
    method = "vectorized_spectral"
    tolerance = 1e-8 * max(dose, 1.)
    if (values is None or values.min() < -tolerance
            or np.max(np.abs(values.sum(axis=1) - delivered)) > tolerance):
        values = _expm_states(system, times, dose, duration)
        method = "matrix_exponential_fallback"
    if not np.isfinite(values).all() or values.min() < -tolerance:
        raise ArithmeticError("Linear propagation produced invalid or materially negative amounts")
    raw_negative = int(np.sum(values < 0))
    values = np.maximum(values, 0.)
    if duration == 0:
        values[times == 0] = dose * system.input_vector
    else:
        values[times == 0] = 0.
    errors = values.sum(axis=1) - delivered
    absolute_error = float(np.max(np.abs(errors)))
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
        "model": system.metadata,
    }
    if return_amounts:
        result.update(state_names=system.state_names[:-1], amounts_mg=values[:, :-1],
                      eliminated_mg=values[:, -1], delivered_mg=delivered)
    return result
