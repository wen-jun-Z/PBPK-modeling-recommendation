"""Moment identities, evidence barriers and unchanged primary NCA semantics."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pbpk_kb.blueprint_policy import capacity_guided_design as design
from pbpk_kb.blueprint_policy import permeability_model as model
from tests.test_blueprints.test_permeability_model import physiology
from tests.test_blueprints.test_evidence_decision import request as make_request, evidence


def volume(value=100., *, basis="predicted", unit="l"):
    row = evidence("vss", value, unit, basis, context={
        "species": "human", "matrix": "plasma", "route": "intravenous",
        "study_role": "independent_parameters", "independent_of_target": True})
    if basis == "predicted":
        row["prediction_method"] = {"name": "synthetic_rf", "version": "1",
            "training_overlap": False, "applicability": {
                "target_structure_and_scaffold_excluded": True,
                "human_IV_parameter_model": True}}
    return row


def test_moments_match_independent_time_integrals_with_observer_delay(physiology):
    system = model.build_system(physiology, total_clearance_l_h=8., renal_clearance_l_h=2.,
        log_parameters={"partition_scale": math.log(1.4), "bidirectional_exchange_scale": math.log(.3)})
    a = system.matrix[:-1, :-1]
    observers = np.array([system.observation_vector[:-1], system.venous_plasma_vector[:-1]])
    n = len(a)
    initial = np.r_[system.input_vector[:-1], np.zeros(4)]

    def derivative(t, values):
        concentrations = observers @ values[:n]
        return np.r_[a @ values[:n], concentrations, t * concentrations]

    solution = solve_ivp(derivative, (0., 2000.), initial, method="Radau", rtol=2e-9, atol=2e-12)
    assert solution.success and np.abs(solution.y[:n, -1]).max() < 1e-12
    auc, aumc = solution.y[n:n + 2, -1], solution.y[n + 2:, -1]
    exact = design.exact_moments(physiology, total_clearance_l_h=8., renal_clearance_l_h=2.,
        log_parameters={"partition_scale": math.log(1.4), "bidirectional_exchange_scale": math.log(.3)})
    for i, record in enumerate((exact, exact["central_venous"])):
        assert record["clearance_l_h"] == pytest.approx(1. / auc[i], rel=2e-8)
        assert record["mrt_h"] == pytest.approx(aumc[i] / auc[i], rel=2e-8)
        assert record["vss_l"] == pytest.approx(aumc[i] / auc[i] ** 2, rel=2e-8)
    # Peripheral observation carries a circulation delay, so total capacity is
    # the central-venous identity, not the requested observer's identity.
    assert exact["central_venous"]["vss_l"] == pytest.approx(exact["equilibrium_plasma_capacity_l"])
    assert exact["vss_l"] > exact["equilibrium_plasma_capacity_l"]
    assert exact["observer_vss_offset_from_central_l"] > 0.


@pytest.mark.parametrize("scale", [1e-4, .2, 1., 2000.])
def test_affine_anchor_has_no_arbitrary_partition_clipping(physiology, scale):
    baseline = design.anchor_partition(physiology, make_request(), [])
    affine = baseline["capacity_mapping"]
    target = affine["observer_vss_floor_l"] + scale * affine["vss_per_partition_scale_l"]
    result = design.anchor_partition(physiology, make_request(), [volume(target)])
    assert result["status"] == "anchored"
    assert result["partition_scale"] == pytest.approx(scale, rel=2e-8)
    assert result["anchored_moments"]["vss_l"] == pytest.approx(target, rel=1e-8)
    assert result["vss_basis"] == "predicted"
    assert result["construction_target_role"] == "predicted_prior"
    assert not result["clinical_validation"] and not result["curve_fitting_performed"]
    json.dumps(result, allow_nan=False)


def test_capacity_anchor_preserves_vascular_flow_renal_and_relative_tissue(physiology):
    request = make_request()
    request["evidence"].append(evidence("renal_clearance", 1., "ml/min/kg", context={
        "species": "human", "matrix": "plasma", "route": "intravenous"}))
    logs = {"clearance_scale": math.log(2.), "partition_scale": math.log(3.),
            "bidirectional_exchange_scale": math.log(.7)}
    original_request, original_physiology = deepcopy(request), deepcopy(physiology)
    result = design.anchor_partition(physiology, request, [volume(100.)], log_parameters=logs)
    assert result["status"] == "anchored"
    common = {"total_clearance_l_h": 21., "renal_clearance_l_h": 4.2}
    before = model.build_system(physiology, **common, log_parameters=logs)
    after = model.build_system(physiology, **common, log_parameters=result["log_parameters"])
    mask = np.array([n.endswith("|Extravascular") for n in before.state_names[:-1]])
    np.testing.assert_array_equal(before.blood_equivalent_capacities_l[~mask], after.blood_equivalent_capacities_l[~mask])
    ratios = after.blood_equivalent_capacities_l[mask] / before.blood_equivalent_capacities_l[mask]
    np.testing.assert_allclose(ratios, result["partition_scale"] / 3., rtol=2e-15)
    for key in ("flow_edges", "exchange_edges", "blood_to_plasma_ratio", "clearance_components_l_h"):
        assert before.metadata[key] == after.metadata[key]
    assert result["anchored_moments"]["clearance_components_l_h"]["qualified_renal_fixed"] == pytest.approx(4.2)
    assert result["log_parameters"]["clearance_scale"] == logs["clearance_scale"]
    assert result["log_parameters"]["bidirectional_exchange_scale"] == logs["bidirectional_exchange_scale"]
    assert request == original_request and physiology == original_physiology


@pytest.mark.parametrize("which", ["blood", "observer", "below_observer"])
def test_infeasible_volume_blocks_without_clipping(physiology, which):
    base = design.anchor_partition(physiology, make_request(), [])
    floor = base["capacity_mapping"]["observer_vss_floor_l"]
    blood = base["capacity_mapping"]["immutable_vascular_plasma_capacity_l"]
    target = {"blood": blood, "observer": floor, "below_observer": (blood + floor) / 2.}[which]
    result = design.anchor_partition(physiology, make_request(), [volume(target)])
    assert result["status"] == "blocked"
    assert result["target_vss_l"] == target and result["vss_basis"] == "predicted"
    assert result["partition_scale"] == 1. and result["anchored_moments"] is None
    assert result["log_parameters"] == base["log_parameters"]


def test_qualified_independent_volume_has_priority_and_unit_conversion(physiology):
    result = design.anchor_partition(physiology, make_request(), [volume(200.), volume(1., basis="derived", unit="l/kg")])
    assert result["status"] == "anchored"
    assert result["target_vss_l"] == 70.
    assert result["vss_basis"] == "derived"
    assert result["selected_vss_evidence"]["value"] == 1.
    assert result["selected_vss_evidence"]["unit"] == "l/kg"
    conflict = design.anchor_partition(physiology, make_request(), [volume(100., basis="measured"), volume(101., basis="derived")])
    assert conflict["status"] == "blocked" and "equal_priority_vss_conflict" in conflict["reasons"]


@pytest.mark.parametrize("issue", ["none", "missing_context", "species", "scaffold", "overlap", "unknown_basis"])
def test_unknown_or_unqualified_volume_retains_original_model(physiology, issue):
    row = volume()
    if issue == "missing_context":
        row.pop("context")
    elif issue == "species":
        row["context"]["species"] = "rat"
    elif issue == "scaffold":
        row["prediction_method"]["applicability"]["target_structure_and_scaffold_excluded"] = False
    elif issue == "overlap":
        row["prediction_method"]["training_overlap"] = True
    elif issue == "unknown_basis":
        row["basis"] = "unknown"
    result = design.anchor_partition(physiology, make_request(), [] if issue == "none" else [row])
    assert result["status"] == "retain_unanchored"
    assert result["partition_scale"] == 1. and result["selected_vss_evidence"] is None


@pytest.mark.parametrize("issue", ["role", "independence", "path", "hash", "compound", "source", "curve"])
def test_evaluation_or_unbound_volume_cannot_enter_anchor(physiology, issue):
    row, request = volume(), make_request()
    if issue == "role":
        row["context"]["study_role"] = "evaluation"
    elif issue == "independence":
        row["context"]["independent_of_target"] = False
    elif issue == "path":
        request["forbidden_sources"] = [{"path": row["source"]["path"], "sha256": "b" * 64}]
    elif issue == "hash":
        request["forbidden_sources"] = [{"path": "another/file", "sha256": row["source"]["sha256"]}]
    elif issue == "compound":
        row["compound_id"] = "wrong"
    elif issue == "source":
        row["source"]["sha256"] = "unbound"
    elif issue == "curve":
        row["clinical_curve"] = [1., 2.]
    result = design.anchor_partition(physiology, request, [row])
    assert result["status"] == "blocked" and result["anchored_moments"] is None


@pytest.mark.parametrize("value,unit", [(True, "l"), (0., "l"), (-1., "l"), (math.nan, "l"), (math.inf, "l"), (1., "ml"), (1., "l/h")])
def test_invalid_volume_or_units_cannot_be_converted_to_success(physiology, value, unit):
    result = design.anchor_partition(physiology, make_request(), [volume(value, unit=unit)])
    assert result["status"] == "blocked" and "invalid_vss_value_or_unit" in result["reasons"]


def test_adapter_retains_failed_nca_beside_exact_moments_and_correct_diagnostics(physiology):
    request = make_request()
    request["target"]["window_end_h"] = 2.
    result = design.simulate_design(physiology, request, times_h=[0., .1, .5, 1., 2.])
    assert result["status"] == "predicted"
    assert result["endpoints"]["clearance"]["value"] is None
    assert result["endpoints"]["vss"]["value"] is None
    assert result["exact_moments"]["clearance_l_h"] == pytest.approx(21.)
    assert result["exact_moments"]["vss_l"] > 0
    assert result["terminal_diagnostics"] == result["nca_summary"]["diagnostics"]
    assert result["terminal_diagnostics"]
    assert not result["exact_moments_used_to_fill_NCA"]
    assert result["full_trajectory"]["time_h"] == [0., .1, .5, 1., 2.]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("infusion_minutes", [0., 30.])
def test_adapter_full_tail_dose_linearity_and_mass_balance(physiology, infusion_minutes):
    request = make_request()
    request["dosing"]["infusion_minutes"] = infusion_minutes
    request["target"]["window_end_h"] = 4.
    times = np.unique(np.r_[0., np.geomspace(1e-4, 400., 300), .5, 4.])
    original = design.simulate_design(physiology, request, times_h=times)
    request["dosing"]["dose_mg"] *= 3.
    triple = design.simulate_design(physiology, request, times_h=times)
    assert original["curve"]["time_h"][-1] == 4.
    assert original["full_trajectory"]["time_h"][-1] == 400.
    assert len(original["full_trajectory"]["time_h"]) == len(times)
    np.testing.assert_allclose(triple["full_trajectory"]["concentration_mg_l"],
        np.asarray(original["full_trajectory"]["concentration_mg_l"]) * 3., rtol=1e-9, atol=1e-12)
    assert original["exact_moments"] == triple["exact_moments"]
    assert original["mass_balance"]["maximum_absolute_error_mg"] < 1e-7
    assert min(original["full_trajectory"]["concentration_mg_l"]) >= 0
    assert original["native_solver_calls"] == 0 and original["linear_solver_calls"] == 1


def test_zero_clearance_has_no_finite_moments_and_short_grid_is_rejected(physiology):
    with pytest.raises(ValueError, match="positive"):
        design.exact_moments(physiology, total_clearance_l_h=0.)
    with pytest.raises(ValueError, match="window"):
        design.simulate_design(physiology, make_request(), times_h=[0., 1., 2.])
