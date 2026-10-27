"""Independent circulation and numerical checks for the new perfusion solver."""
from copy import deepcopy
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pbpk_kb.blueprint_policy import perfusion_model as model


@pytest.fixture
def physiology():
    # A deliberately synthetic subject. Capacities differ from anatomical
    # volumes and from volume * global BP, exposing accidental substitution.
    flows = {"Lung": 60., "Heart": 3., "Liver": 6., "Kidney": 12.,
             "Spleen": 2., "Stomach": 2., "SmallIntestine": 5.,
             "LargeIntestine": 1., "Bone": 3., "Brain": 5., "Fat": 3.,
             "Gonads": 1., "Muscle": 10., "Pancreas": 1., "Skin": 6.}
    volumes = {"Lung": .7, "Heart": .3, "Liver": 1.7, "Kidney": .3,
               "Spleen": .2, "Stomach": .3, "SmallIntestine": .7,
               "LargeIntestine": .5, "Bone": 8., "Brain": 1.3,
               "Fat": 15., "Gonads": .1, "Muscle": 25., "Pancreas": .1,
               "Skin": 3.}
    return {"version": model.PHYSIOLOGY_VERSION, "blood_to_plasma_ratio": 1.7,
        "blood": {"arterial": {"volume_l": .7}, "venous": {"volume_l": 2.},
                  "portal": {"volume_l": .2}},
        "organs": [{"name": name, "volume_l": volumes[name], "flow_l_h": q,
                    "vascular_volume_l": .1 * volumes[name],
                    "extravascular_volume_l": .9 * volumes[name],
                    "vascular_capacity_l": .13 * volumes[name],
                    "extravascular_capacity_l": 1.8 * volumes[name]}
                   for name, q in flows.items()],
        "circulation": {"portal_contributors": ["Stomach", "SmallIntestine",
                         "LargeIntestine", "Pancreas", "Spleen"],
                        "cardiac_flow_l_h": 60., "hepatic_arterial_flow_l_h": 6.,
                        "portal_flow_l_h": 11., "liver_total_flow_l_h": 17.},
        "peripheral_venous_weights": {"ArterialBlood": 0., "Bone": 0., "Fat": 0.,
                                       "Muscle": .287, "Skin": .713}}


def run(physiology, **overrides):
    kwargs = dict(times_h=[0., .01, .1, .5, 1., 4., 12.], dose_mg=100.,
                  infusion_duration_h=.5, total_clearance_l_h=8.)
    kwargs.update(overrides)
    return model.simulate(physiology, **kwargs)


def test_circulation_is_conservative_and_portal_is_not_double_counted(physiology):
    system = model.build_system(physiology, total_clearance_l_h=8.)
    idx = {name: i for i, name in enumerate(system.state_names)}
    matrix = system.matrix
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0.)
    assert np.all(off_diagonal >= 0)
    np.testing.assert_allclose(matrix.sum(axis=0), 0., atol=2e-14)
    edges = {(r["source"], r["target"]): r["flow_l_h"]
             for r in system.metadata["flow_edges"]}
    assert edges[("ArterialBlood", "Liver")] == 6.
    assert edges[("PortalBlood", "Liver")] == 11.
    assert edges[("Liver", "VenousBlood")] == 17.
    assert edges[("Stomach", "PortalBlood")] == 2.
    assert ("Stomach", "VenousBlood") not in edges
    assert matrix[idx["Liver"], idx["PortalBlood"]] == pytest.approx(11. / .2)
    assert matrix[idx["Eliminated"], idx["VenousBlood"]] == pytest.approx(8. / (2. * 1.7))
    assert not matrix.flags.writeable


@pytest.mark.parametrize("family,logs,expected", [
    ("global", {"partition_scale": math.log(.1)}, {"Muscle": .1, "Liver": .1}),
    ("grouped", {"fast_partition_scale": math.log(3.),
                 "slow_partition_scale": math.log(.2)},
     {"Liver": 3., "Lung": 3., "Spleen": 3., "Pancreas": .2, "Skin": .2})])
def test_partition_scales_only_extravascular_native_capacity(physiology, family, logs, expected):
    system = model.build_system(physiology, total_clearance_l_h=8., family=family,
                                log_parameters=logs)
    for organ in physiology["organs"]:
        name = organ["name"]
        if name not in expected:
            continue
        result = system.metadata["organ_capacities"][name]
        capacity = organ["vascular_capacity_l"] + expected[name] * organ["extravascular_capacity_l"]
        assert result["vascular_capacity_l"] == organ["vascular_capacity_l"]
        assert result["plasma_equivalent_capacity_l"] == pytest.approx(capacity)
        assert result["blood_equivalent_capacity_l"] == pytest.approx(capacity / 1.7)
    if family == "grouped":
        assert "target-selected" in system.metadata["grouping_interpretation"]


def test_bolus_uses_blood_volume_and_plasma_ratio(physiology):
    result = run(physiology, times_h=[0., .01], infusion_duration_h=0.,
                 observation="venous", return_amounts=True)
    assert result["concentration_mg_l"][0] == pytest.approx(100. / (2. * 1.7))
    assert result["amounts_mg"][0].sum() == 100.
    assert result["eliminated_mg"][0] == 0.
    assert result["amounts_mg"][0, result["state_names"].index("VenousBlood")] == 100.


@pytest.mark.parametrize("clearance", [0., 8., 100.])
@pytest.mark.parametrize("duration", [0., .5])
def test_body_plus_eliminated_equals_delivered_at_every_time(physiology, clearance, duration):
    times = np.r_[0., np.geomspace(1e-6, 240., 121)]
    result = run(physiology, times_h=times, total_clearance_l_h=clearance,
                 infusion_duration_h=duration, return_amounts=True)
    np.testing.assert_allclose(result["amounts_mg"].sum(axis=1) + result["eliminated_mg"],
                               result["delivered_mg"], rtol=1e-10, atol=1e-9)
    assert result["mass_balance_relative_error"] < 1e-10
    assert np.all(np.diff(result["eliminated_mg"]) >= -1e-10)
    assert np.all(result["amounts_mg"] >= 0.)
    if clearance == 0.:
        np.testing.assert_array_equal(result["eliminated_mg"], 0.)


def test_native_observer_uses_fixed_vascular_plasma_weights(physiology):
    result = run(physiology, return_amounts=True)
    amounts = result["amounts_mg"]
    capacities = result["model"]["organ_capacities"]
    expected = sum(w * amounts[:, result["state_names"].index(name)] /
                   capacities[name]["plasma_equivalent_capacity_l"]
                   for name, w in [("Muscle", .287), ("Skin", .713)])
    np.testing.assert_allclose(result["concentration_mg_l"], expected, rtol=1e-14)
    assert result["observation"] == "native_peripheral_venous"
    assert result["model"]["observation_limitation"] is None
    assert not np.allclose(result["concentration_mg_l"], result["venous_plasma_mg_l"])


def test_missing_native_observer_is_explicitly_central_venous(physiology):
    del physiology["peripheral_venous_weights"]
    result = run(physiology)
    assert result["observation"] == "venous"
    assert "no peripheral" in result["model"]["observation_limitation"]
    np.testing.assert_array_equal(result["concentration_mg_l"], result["venous_plasma_mg_l"])
    with pytest.raises(ValueError, match="exported fixed weights"):
        run(physiology, observation="native_peripheral_venous")


def test_qualified_renal_clearance_remains_fixed(physiology):
    result = run(physiology, total_clearance_l_h=10., renal_clearance_l_h=4.,
                 log_parameters={"clearance_scale": math.log(3.)})
    assert result["effective_total_clearance_l_h"] == pytest.approx(22.)
    assert result["model"]["clearance_components_l_h"] == {
        "qualified_renal_fixed": 4., "scaled_nonrenal": pytest.approx(18.)}
    unqualified = run(physiology, total_clearance_l_h=10.,
                      log_parameters={"clearance_scale": math.log(3.)})
    assert unqualified["effective_total_clearance_l_h"] == pytest.approx(30.)
    assert "not_validated_organ" in result["model"]["clearance_interpretation"]
    with pytest.raises(ValueError, match="exceeds"):
        run(physiology, total_clearance_l_h=10., renal_clearance_l_h=11.)


def independent_ode(physiology, times, dose, duration, *, family, logs, clearance):
    """Separate concentration/flow balance, not the implementation's matrix."""
    organs = {o["name"]: o for o in physiology["organs"]}
    names = ["ArterialBlood", "VenousBlood", "PortalBlood", *organs, "Eliminated"]
    idx = {name: i for i, name in enumerate(names)}
    bp = physiology["blood_to_plasma_ratio"]
    portal = set(physiology["circulation"]["portal_contributors"])
    q = {name: o["flow_l_h"] for name, o in organs.items()}
    qp, qt = sum(q[name] for name in portal), q["Lung"]
    capacity = {}
    # Explicit test membership deliberately does not use model.FAST_TISSUES.
    fast = {"Heart", "Lung", "Liver", "Kidney", "Spleen", "Stomach", "SmallIntestine", "LargeIntestine"}
    for name, organ in organs.items():
        key = ("partition_scale" if family == "global" else
               "fast_partition_scale" if name in fast else "slow_partition_scale")
        capacity[name] = (organ["vascular_capacity_l"] +
                          organ["extravascular_capacity_l"] * math.exp(logs.get(key, 0.)))
    effective = clearance * math.exp(logs.get("clearance_scale", 0.))

    def rhs(amounts, input_rate):
        concentrations = {name: bp * amounts[idx[name]] / cap for name, cap in capacity.items()}
        for key, name in [("arterial", "ArterialBlood"), ("venous", "VenousBlood"), ("portal", "PortalBlood")]:
            concentrations[name] = amounts[idx[name]] / physiology["blood"][key]["volume_l"]
        ca, cv, cp = (concentrations[k] for k in ["ArterialBlood", "VenousBlood", "PortalBlood"])
        change = np.zeros(len(names))
        change[idx["ArterialBlood"]] = qt * concentrations["Lung"] - sum(q[n] for n in organs if n != "Lung") * ca
        change[idx["Lung"]] = qt * (cv - concentrations["Lung"])
        change[idx["PortalBlood"]] = sum(q[n] * concentrations[n] for n in portal) - qp * cp
        returning = 0.
        for name in organs:
            if name == "Lung":
                continue
            outflow = q[name] + (qp if name == "Liver" else 0.)
            change[idx[name]] = q[name] * ca - outflow * concentrations[name]
            if name == "Liver":
                change[idx[name]] += qp * cp
            if name not in portal:
                returning += outflow * concentrations[name]
        loss = effective * cv / bp
        change[idx["VenousBlood"]] = returning - qt * cv - loss + input_rate
        change[idx["Eliminated"]] = loss
        return change

    initial = np.zeros(len(names))
    if duration == 0.:
        initial[idx["VenousBlood"]] = dose
        solution = solve_ivp(lambda _, a: rhs(a, 0.), (0., times[-1]), initial,
                             method="Radau", rtol=2e-11, atol=2e-12, dense_output=True)
        assert solution.success
        return solution.sol(times).T
    first = solve_ivp(lambda _, a: rhs(a, dose / duration), (0., duration), initial,
                      method="Radau", rtol=2e-11, atol=2e-12, dense_output=True)
    second = solve_ivp(lambda _, a: rhs(a, 0.), (duration, times[-1]), first.y[:, -1],
                       method="Radau", rtol=2e-11, atol=2e-12, dense_output=True)
    assert first.success and second.success
    return np.array([first.sol(t) if t <= duration else second.sol(t) for t in times])


@pytest.mark.parametrize("duration", [0., .35])
@pytest.mark.parametrize("family,logs", [
    ("global", {"clearance_scale": math.log(.7), "partition_scale": math.log(1.8)}),
    ("grouped", {"clearance_scale": math.log(1.3), "fast_partition_scale": math.log(.3),
                 "slow_partition_scale": math.log(3.)})])
def test_all_amounts_agree_with_independently_coded_solve_ivp(physiology, duration, family, logs):
    times = np.array([0., .001, .1, .35 - 1e-9, .35, .35 + 1e-9, .7, 2., 8.])
    result = run(physiology, times_h=times, infusion_duration_h=duration,
                 family=family, log_parameters=logs, return_amounts=True)
    independent = independent_ode(physiology, times, 100., duration, family=family, logs=logs, clearance=8.)
    np.testing.assert_allclose(result["amounts_mg"], independent[:, :-1], rtol=2e-8, atol=3e-9)
    np.testing.assert_allclose(result["eliminated_mg"], independent[:, -1], rtol=2e-8, atol=3e-9)


def test_infusion_endpoint_is_continuous_and_short_infusion_approaches_bolus(physiology):
    result = run(physiology, times_h=[.5 - 1e-9, .5, .5 + 1e-9], return_amounts=True)
    np.testing.assert_allclose(result["amounts_mg"][0], result["amounts_mg"][2], atol=4e-7, rtol=1e-7)
    bolus = run(physiology, times_h=[.001, .1, 1., 10.], infusion_duration_h=0.)
    short = run(physiology, times_h=[.001, .1, 1., 10.], infusion_duration_h=1e-10)
    np.testing.assert_allclose(short["concentration_mg_l"], bolus["concentration_mg_l"], rtol=2e-7, atol=1e-9)


def test_dense_grid_uses_vectorized_spectral_and_matches_dense_exponential(physiology, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Ordinary well-conditioned propagation should not perform per-time expm")
    monkeypatch.setattr(model, "_expm_states", forbidden)
    times = np.linspace(0., 80., 8101)
    result = run(physiology, times_h=times, infusion_duration_h=0., return_amounts=True)
    assert result["solver"]["method"] == "vectorized_spectral"
    system = model.build_system(physiology, total_clearance_l_h=8.)
    for i in [1, 50, 1000, 8100]:
        reference = expm(system.matrix * times[i]) @ (100. * system.input_vector)
        np.testing.assert_allclose(result["amounts_mg"][i], reference[:-1], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("duration", [0., .5])
def test_fallback_is_equivalent_and_labelled(physiology, monkeypatch, duration):
    reference = run(physiology, infusion_duration_h=duration, return_amounts=True)
    monkeypatch.setattr(model, "_spectral_states", lambda *args: (None, 1e12))
    result = run(physiology, infusion_duration_h=duration, return_amounts=True)
    assert result["solver"]["method"] == "matrix_exponential_fallback"
    np.testing.assert_allclose(result["amounts_mg"], reference["amounts_mg"], rtol=1e-9, atol=1e-10)


def test_zero_dose_and_duplicate_times_and_no_large_optional_state_output(physiology):
    result = run(physiology, times_h=[0., .5, .5, 10.], dose_mg=0.)
    np.testing.assert_array_equal(result["concentration_mg_l"], 0.)
    assert result["mass_balance_relative_error"] == 0.
    assert "amounts_mg" not in result
    assert "state_names" not in result


@pytest.mark.parametrize("times", [[], [-1., 0.], [1., .1], [np.nan], [np.inf], [True], [[0., 1.]], ["0"]])
def test_invalid_sampling_times_are_rejected(physiology, times):
    with pytest.raises(ValueError, match="times_h"):
        run(physiology, times_h=times)


@pytest.mark.parametrize("case,match", [
    ("total_as_direct", "direct arterial"), ("cardiac", "systemic arterial"),
    ("portal", "summed portal"), ("missing_portal", "explicit portal"),
    ("duplicate_organ", "unique"), ("negative_capacity", "nonnegative"),
    ("observer_sum", "sum to one"), ("observer_unknown", "unsupported compartment")])
def test_inconsistent_physiology_is_rejected(physiology, case, match):
    if case == "total_as_direct":
        next(o for o in physiology["organs"] if o["name"] == "Liver")["flow_l_h"] = 17.
    elif case == "cardiac":
        next(o for o in physiology["organs"] if o["name"] == "Skin")["flow_l_h"] = 9.
    elif case == "portal":
        physiology["circulation"]["portal_flow_l_h"] = 12.
    elif case == "missing_portal":
        del physiology["blood"]["portal"]
    elif case == "duplicate_organ":
        physiology["organs"].append(deepcopy(physiology["organs"][0]))
    elif case == "negative_capacity":
        physiology["organs"][0]["vascular_capacity_l"] = -.1
    elif case == "observer_sum":
        physiology["peripheral_venous_weights"]["Muscle"] = .3
    elif case == "observer_unknown":
        physiology["peripheral_venous_weights"]["SecretTissue"] = 0.
    with pytest.raises(ValueError, match=match):
        run(physiology)


@pytest.mark.parametrize("kwargs,match", [
    ({"dose_mg": -1.}, "dose_mg"), ({"infusion_duration_h": -1.}, "duration"),
    ({"total_clearance_l_h": float("nan")}, "finite"),
    ({"log_parameters": {"partition_scale": float("inf")}}, "finite"),
    ({"log_parameters": {"partition_scale": 1000.}}, "overflow"),
    ({"log_parameters": {"target_fitted_kp": 0.}}, "Unexpected"),
    ({"family": "extra_permeability"}, "family")])
def test_invalid_parameters_are_rejected(physiology, kwargs, match):
    with pytest.raises(ValueError, match=match):
        run(physiology, **kwargs)
