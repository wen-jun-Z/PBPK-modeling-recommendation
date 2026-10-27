"""Finite effective exchange between native vascular/extravascular capacities.

Each organ has two amounts. The shared exchange conductance is defined as
PS_i = effective organ blood flow * exchange_scale (L/h). This is a declared
flow-proportional effective-exchange hypothesis, not measured membrane PS.
The frozen perfusion model supplies circulation validation and propagation;
its central systemic plasma-clearance sink and fixed renal component remain.
No clinical observations or native processes are accessed here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np

from . import perfusion_model as perfusion

VERSION = "linear-finite-exchange-model-1"
PHYSIOLOGY_VERSION = perfusion.PHYSIOLOGY_VERSION
PARAMETERS = {"exchange": ("clearance_scale", "partition_scale",
                            "bidirectional_exchange_scale")}
EXCHANGE_INTERPRETATION = (
    "flow-proportional effective exchange PS_i=Q_effective_i*exchange_scale; "
    "not measured membrane permeability; hepatic Q is total hepatic inflow"
)


def build_system(physiology, *, total_clearance_l_h, renal_clearance_l_h=None,
                 family="exchange", log_parameters=None, observation="auto"):
    """Build vascular flow and reversible exchange with one elimination ledger.

    Natural-log parameters must be finite; fitting bounds are imposed by the
    caller. Broader positive values permit engineering checks of the PS limit.
    At equilibrium A_extra/K_extra_plasma = A_vascular/K_vascular_plasma.
    """
    if family not in PARAMETERS:
        raise ValueError("Unknown finite-exchange model family")
    if log_parameters is None:
        log_parameters = {}
    if not isinstance(log_parameters, Mapping) or set(log_parameters) - set(PARAMETERS[family]):
        raise ValueError("Unexpected log parameter for this model family")
    scales = {}
    for name in PARAMETERS[family]:
        value = perfusion._number(log_parameters.get(name, 0.), f"log_parameters.{name}")
        try:
            scales[name] = perfusion._number(math.exp(value), name, positive=True)
        except OverflowError as exc:
            raise ValueError("Parameter scale overflow") from exc

    # Reuse the already verified topology/observation/clearance contract. This
    # constructs a matrix only, with no propagation or native solver execution.
    reference = perfusion.build_system(physiology,
        total_clearance_l_h=total_clearance_l_h,
        renal_clearance_l_h=renal_clearance_l_h, family="global",
        log_parameters={name: log_parameters.get(name, 0.)
                        for name in ("clearance_scale", "partition_scale")},
        observation=observation)
    organs = physiology["organs"]
    if {o["name"] for o in organs} != perfusion.FAST_TISSUES | perfusion.SLOW_TISSUES:
        raise ValueError("Finite-exchange scope requires all 15 native organs")
    organ_names = {o["name"] for o in organs}
    blood_names = [name for name in reference.state_names
                   if name not in organ_names and name != "Eliminated"]
    names = tuple(blood_names + [f"{o['name']}|{kind}" for o in organs
                               for kind in ("Vascular", "Extravascular")] + ["Eliminated"])
    index = {name: i for i, name in enumerate(names)}
    reference_index = {name: i for i, name in enumerate(reference.state_names)}
    bp = reference.metadata["blood_to_plasma_ratio"]
    capacities = [float(reference.blood_equivalent_capacities_l[reference_index[name]])
                  for name in blood_names]
    organ_capacity = deepcopy(reference.metadata["organ_capacities"])
    for organ in organs:
        name = organ["name"]
        vascular = perfusion._number(organ["vascular_capacity_l"], f"{name}.vascular_capacity_l", positive=True)
        extra = perfusion._number(organ["extravascular_capacity_l"] * scales["partition_scale"],
                                  f"{name}.extravascular_capacity_l", positive=True)
        capacities.extend((vascular / bp, extra / bp))
        organ_capacity[name].update(vascular_blood_equivalent_capacity_l=vascular / bp,
            extravascular_plasma_equivalent_capacity_l=extra,
            extravascular_blood_equivalent_capacity_l=extra / bp,
            vascular_state=f"{name}|Vascular", extravascular_state=f"{name}|Extravascular")

    n = len(names)
    matrix = np.zeros((n, n))
    input_vector = np.zeros(n)
    input_vector[index["VenousBlood"]] = 1.
    edges, exchanges = [], []

    def transfer(source, target, conductance):
        rate = conductance / capacities[index[source]]
        if not math.isfinite(rate):
            raise ValueError("Nonfinite flow/exchange rate")
        matrix[index[source], index[source]] -= rate
        matrix[index[target], index[source]] += rate

    def vascular_name(name):
        return f"{name}|Vascular" if name in organ_names else name

    for edge in reference.metadata["flow_edges"]:
        source, target = vascular_name(edge["source"]), vascular_name(edge["target"])
        transfer(source, target, edge["flow_l_h"])
        edges.append({"source": source, "target": target, "flow_l_h": edge["flow_l_h"]})
    for organ in organs:
        name = organ["name"]
        q = (physiology["circulation"]["liver_total_flow_l_h"] if name == "Liver"
             else organ["flow_l_h"])
        ps = perfusion._number(q * scales["bidirectional_exchange_scale"],
                              f"{name}.exchange_conductance_l_h", positive=True)
        vascular, extra = f"{name}|Vascular", f"{name}|Extravascular"
        transfer(vascular, extra, ps)
        transfer(extra, vascular, ps)
        exchanges.append({"organ": name, "effective_flow_l_h": q,
            "exchange_conductance_l_h": ps, "vascular_state": vascular, "extravascular_state": extra})

    venous_vector = np.zeros(n)
    venous_vector[index["VenousBlood"]] = 1. / (capacities[index["VenousBlood"]] * bp)
    loss = reference.metadata["effective_total_clearance_l_h"] * venous_vector[index["VenousBlood"]]
    matrix[index["VenousBlood"], index["VenousBlood"]] -= loss
    matrix[index["Eliminated"], index["VenousBlood"]] += loss
    observation_vector = np.zeros(n)
    for name, weight in reference.metadata["observation_weights"].items():
        vascular = vascular_name(name)
        observation_vector[index[vascular]] = weight / (capacities[index[vascular]] * bp)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite finite-exchange generator")
    for array in (matrix, input_vector, observation_vector, venous_vector):
        array.setflags(write=False)
    capacity_array = np.asarray(capacities)
    capacity_array.setflags(write=False)
    metadata = {**deepcopy(reference.metadata), "version": VERSION, "family": family,
        "parameters": scales,
        "log_parameters": {name: float(log_parameters.get(name, 0.)) for name in PARAMETERS[family]},
        "organ_capacities": organ_capacity, "flow_edges": edges, "exchange_edges": exchanges,
        "exchange_interpretation": EXCHANGE_INTERPRETATION,
        "exchange_flux_definition": "PS_i*(A_vascular/(K_vascular_plasma/BP)-BP*A_extra/K_extra_plasma)",
        "blood_partition_assumption": "instantaneous local plasma/RBC equilibrium within each vascular state",
        "structural_scope": "three blood pools and 15 organ vascular/extravascular pairs",
        "perfusion_limit": "same capacities, blood flow, observer and central systemic clearance as frozen global perfusion model"}
    return perfusion.PerfusionSystem(matrix, names, input_vector, observation_vector,
                                    venous_vector, capacity_array, metadata)


def simulate(physiology, *, times_h, dose_mg, infusion_duration_h, total_clearance_l_h,
             renal_clearance_l_h=None, family="exchange", log_parameters=None,
             observation="auto", return_amounts=False):
    """Same sampled mg/L and conservation contract as the frozen perfusion model.

    The short propagation audit is shared in behavior with that model because
    its public simulate entry point cannot accept a separately built system.
    Spectral and fallback exponential implementations are reused directly.
    """
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
