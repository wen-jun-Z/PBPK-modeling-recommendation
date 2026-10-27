"""Equilibrated metabolizing liver with finite exchange in the other organs.

The qualified nonrenal systemic plasma clearance is assigned to one locally
equilibrated hepatic pool, as an explicit structural hypothesis. A flow-based
inversion preserves its central plasma dose/AUC meaning; the renal component
remains a fixed central sink. This is not a claim of exclusively hepatic drug
metabolism, measured enzyme intrinsic clearance, or measured membrane PS.
The generic solver neither identifies drugs nor reads clinical observations.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from scipy.linalg import solve

from . import permeability_model as exchange
from . import perfusion_model as propagation

VERSION = "linear-equilibrated-hepatic-exchange-model-1"
PHYSIOLOGY_VERSION = exchange.PHYSIOLOGY_VERSION
PARAMETERS = exchange.PARAMETERS
LIVER_STATE = "Liver|Equilibrated"


def build_system(physiology, *, total_clearance_l_h, renal_clearance_l_h=None,
                 family="exchange", log_parameters=None, observation="auto"):
    """Construct 3 blood pools, 14 organ pairs, one liver and an elimination ledger.

    The local hepatic plasma-equivalent sink is C_NR*BP*Q/(BP*Q-C_NR).
    It acts on A_liver/(K_vascular+K_extra) and has no additional fu multiplier.
    Its definition is effective local clearance, not an enzyme CLint estimate.
    """
    reference = exchange.build_system(physiology,
        total_clearance_l_h=total_clearance_l_h,
        renal_clearance_l_h=renal_clearance_l_h, family=family,
        log_parameters=log_parameters, observation=observation)
    bp = reference.metadata["blood_to_plasma_ratio"]
    renal = reference.metadata["clearance_components_l_h"]["qualified_renal_fixed"]
    nonrenal = reference.metadata["clearance_components_l_h"]["scaled_nonrenal"]
    q = physiology["circulation"]["liver_total_flow_l_h"]
    capacity = propagation._number(bp * q, "hepatic plasma-clearance flow capacity", positive=True)
    if nonrenal < 0. or nonrenal >= capacity:
        raise ValueError("Nonrenal systemic plasma clearance must satisfy 0 <= CL_NR < BP*Q_H")
    local = propagation._number(nonrenal * capacity / (capacity - nonrenal),
                                "hepatic local plasma clearance", nonnegative=True)
    names = []
    for name in reference.state_names:
        if name == "Liver|Vascular":
            names.append(LIVER_STATE)
        elif name != "Liver|Extravascular":
            names.append(name)
    names = tuple(names)
    index = {name: i for i, name in enumerate(names)}
    old_index = {name: i for i, name in enumerate(reference.state_names)}
    hepatic_plasma_capacity = reference.metadata["organ_capacities"]["Liver"]["plasma_equivalent_capacity_l"]
    capacities = np.asarray([hepatic_plasma_capacity / bp if name == LIVER_STATE else
                            reference.blood_equivalent_capacities_l[old_index[name]]
                            for name in names[:-1]])
    n = len(names)
    matrix = np.zeros((n, n))
    input_vector = np.zeros(n)
    input_vector[index["VenousBlood"]] = 1.

    def transfer(source, target, conductance):
        rate = conductance / capacities[index[source]]
        if not math.isfinite(rate):
            raise ValueError("Nonfinite hepatic-exchange transfer rate")
        matrix[index[source], index[source]] -= rate
        matrix[index[target], index[source]] += rate

    def mapped(name):
        return LIVER_STATE if name == "Liver|Vascular" else name

    edges, exchanges = [], []
    for edge in reference.metadata["flow_edges"]:
        source, target = mapped(edge["source"]), mapped(edge["target"])
        transfer(source, target, edge["flow_l_h"])
        edges.append({"source": source, "target": target, "flow_l_h": edge["flow_l_h"]})
    for edge in reference.metadata["exchange_edges"]:
        if edge["organ"] == "Liver":
            continue
        vascular, extra = edge["vascular_state"], edge["extravascular_state"]
        transfer(vascular, extra, edge["exchange_conductance_l_h"])
        transfer(extra, vascular, edge["exchange_conductance_l_h"])
        exchanges.append(deepcopy(edge))
    venous_vector = np.zeros(n)
    venous_vector[index["VenousBlood"]] = 1. / (capacities[index["VenousBlood"]] * bp)
    renal_rate = renal * venous_vector[index["VenousBlood"]]
    hepatic_rate = local / hepatic_plasma_capacity
    for source, rate in [("VenousBlood", renal_rate), (LIVER_STATE, hepatic_rate)]:
        matrix[index[source], index[source]] -= rate
        matrix[index["Eliminated"], index[source]] += rate
    observation_vector = np.zeros(n)
    for name, weight in reference.metadata["observation_weights"].items():
        vascular = LIVER_STATE if name == "Liver" else (
            name if name in {"VenousBlood", "ArterialBlood"} else name + "|Vascular")
        observation_vector[index[vascular]] = weight / (capacities[index[vascular]] * bp)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite hepatic-exchange generator")

    # Unit-bolus moments validate the clearance conversion independently of
    # a finite simulation horizon, an NCA cutoff or any target observations.
    expected_total = reference.metadata["effective_total_clearance_l_h"]
    if expected_total == 0.:
        moments = {"status": "undefined_zero_clearance", "unit_dose_mg": 1.,
                   "central_clearance_equivalent": None, "observer_clearance_equivalent": None}
    else:
        integrated = solve(-matrix[:-1, :-1], input_vector[:-1], assume_a="gen")
        central_auc = float(venous_vector[:-1] @ integrated)
        observed_auc = float(observation_vector[:-1] @ integrated)
        if (not np.isfinite(integrated).all() or integrated.min() < -1e-10
                or not math.isfinite(central_auc) or central_auc <= 0.
                or not math.isfinite(observed_auc) or observed_auc <= 0.):
            raise ArithmeticError("Invalid infinite-time hepatic-exchange moments")
        achieved = 1. / central_auc
        if not math.isclose(achieved, expected_total, rel_tol=1e-8, abs_tol=1e-10):
            raise ArithmeticError("Hepatic clearance inversion failed systemic dose/AUC audit")
        moments = {"status": "verified", "unit_dose_mg": 1.,
            "central_venous_auc_mg_h_l": central_auc, "observer_auc_mg_h_l": observed_auc,
            "central_venous_dose_over_auc_l_h": achieved,
            "observer_dose_over_auc_l_h": 1. / observed_auc,
            "expected_systemic_plasma_clearance_l_h": expected_total,
            "central_clearance_equivalent": True,
            "observer_clearance_equivalent": math.isclose(1. / observed_auc, expected_total, rel_tol=1e-8, abs_tol=1e-10),
            "definition": "Exact unit-bolus integrated body states from -A^-1*b; native nonhepatic peripheral and central venous plasma share total AUC"}
    organ_capacities = deepcopy(reference.metadata["organ_capacities"])
    for field in ("vascular_state", "extravascular_state"):
        organ_capacities["Liver"].pop(field)
    organ_capacities["Liver"].update(equilibrated_state=LIVER_STATE,
        state_blood_equivalent_capacity_l=hepatic_plasma_capacity / bp,
        readback_definition="one amount equals summed hepatic vascular+scaled extravascular equilibrium capacities")
    metadata = {**deepcopy(reference.metadata), "version": VERSION,
        "organ_capacities": organ_capacities, "flow_edges": edges, "exchange_edges": exchanges,
        "structural_scope": "three blood pools, 14 vascular/extravascular organ pairs, one equilibrated liver",
        "exchange_interpretation": "Flow-proportional effective exchange applies only to 14 nonhepatic organs; liver is locally equilibrated",
        "perfusion_limit": "nonhepatic PS to infinity gives perfusion distribution with the same hepatic local and central renal elimination topology",
        "clearance_interpretation": "qualified_nonrenal_systemic_plasma_CL_assigned_to_equilibrated_liver_as_structural_hypothesis; renal_component_stays_central",
        "hepatic_clearance": {"state": LIVER_STATE, "hepatic_total_flow_l_h": q,
            "blood_to_plasma_ratio": bp, "flow_capacity_plasma_l_h": capacity,
            "qualified_nonrenal_systemic_l_h": nonrenal,
            "local_plasma_equivalent_clearance_l_h": local,
            "hepatic_inlet_extraction_fraction": nonrenal / capacity,
            "plasma_equivalent_capacity_l": hepatic_plasma_capacity,
            "first_order_loss_rate_per_h": hepatic_rate,
            "central_renal_clearance_l_h": renal,
            "central_nonrenal_clearance_l_h": 0.,
            "formula": "C_local=CL_NR*(BP*Q_H)/(BP*Q_H-CL_NR); loss=C_local*A_liver/K_liver_plasma",
            "limitation": "effective local plasma-equivalent clearance, not measured enzyme intrinsic clearance; extrahepatic metabolism is not represented"},
        "infinite_time_moments": moments}
    for array in (matrix, input_vector, observation_vector, venous_vector, capacities):
        array.setflags(write=False)
    return propagation.PerfusionSystem(matrix, names, input_vector, observation_vector,
                                       venous_vector, capacities, metadata)


def simulate(physiology, *, times_h, dose_mg, infusion_duration_h, total_clearance_l_h,
             renal_clearance_l_h=None, family="exchange", log_parameters=None,
             observation="auto", return_amounts=False):
    """Sample plasma mg/L with the frozen solver's public result contract."""
    times = np.asarray(times_h)
    if (times.ndim != 1 or not times.size or times.dtype.kind not in "iuf"
            or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) < 0)):
        raise ValueError("times_h must be a nonempty nonnegative sorted numeric vector")
    times = times.astype(float)
    dose = propagation._number(dose_mg, "dose_mg", nonnegative=True)
    duration = propagation._number(infusion_duration_h, "infusion_duration_h", nonnegative=True)
    if not isinstance(return_amounts, bool):
        raise ValueError("return_amounts must be boolean")
    system = build_system(physiology, total_clearance_l_h=total_clearance_l_h,
        renal_clearance_l_h=renal_clearance_l_h, family=family,
        log_parameters=log_parameters, observation=observation)
    delivered = np.full_like(times, dose) if duration == 0 else dose * np.minimum(times / duration, 1.)
    values, condition = propagation._spectral_states(system, times, dose, duration)
    method = "vectorized_spectral"
    tolerance = 1e-8 * max(dose, 1.)
    if (values is None or values.min() < -tolerance
            or np.max(np.abs(values.sum(axis=1) - delivered)) > tolerance):
        values = propagation._expm_states(system, times, dose, duration)
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
