"""Independent balances and the perfusion limit of finite effective exchange."""
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pbpk_kb.blueprint_policy import residence_exchange_model as model
from pbpk_kb.blueprint_policy import perfusion_model as perfusion


@pytest.fixture
def physiology():
    flows = {"Lung": 60., "Heart": 3., "Liver": 6., "Kidney": 12.,
             "Spleen": 2., "Stomach": 2., "SmallIntestine": 5.,
             "LargeIntestine": 1., "Bone": 3., "Brain": 5., "Fat": 3.,
             "Gonads": 1., "Muscle": 10., "Pancreas": 1., "Skin": 6.}
    return {"version": perfusion.PHYSIOLOGY_VERSION, "blood_to_plasma_ratio": 1.7,
        "blood": {"arterial": {"volume_l": .7}, "venous": {"volume_l": 2.},
                  "portal": {"volume_l": .2}},
        "organs": [{"name": name, "flow_l_h": q, "vascular_capacity_l": .1 + i * .025,
                    "extravascular_capacity_l": .8 + i * .6}
                   for i, (name, q) in enumerate(flows.items())],
        "circulation": {"portal_contributors": ["Stomach", "SmallIntestine",
                         "LargeIntestine", "Pancreas", "Spleen"],
                        "cardiac_flow_l_h": 60., "hepatic_arterial_flow_l_h": 6.,
                        "portal_flow_l_h": 11., "liver_total_flow_l_h": 17.},
        "peripheral_venous_weights": {"ArterialBlood": 0., "Bone": 0., "Fat": 0.,
                                       "Muscle": .287, "Skin": .713}}


def run(physiology, **overrides):
    kwargs = dict(times_h=[0., .01, .1, .35, .5, 1., 4., 12.], dose_mg=100.,
                  infusion_duration_h=.35, total_clearance_l_h=8., renal_clearance_l_h=2.)
    kwargs.update(overrides)
    return model.simulate(physiology, **kwargs)


def test_topology_exchange_and_capacities_have_physical_units(physiology):
    system = model.build_system(physiology, total_clearance_l_h=8., renal_clearance_l_h=2.,
        log_parameters={"partition_scale": math.log(3.), "bidirectional_exchange_scale": math.log(.4)})
    assert len(system.state_names) == 34
    index = {name: i for i, name in enumerate(system.state_names)}
    edges = {(e["source"], e["target"]): e["flow_l_h"] for e in system.metadata["flow_edges"]}
    assert edges[("ArterialBlood", "Liver|Vascular")] == 6.
    assert edges[("PortalBlood", "Liver|Vascular")] == 11.
    assert edges[("Liver|Vascular", "VenousBlood")] == 17.
    assert edges[("Lung|Vascular", "ArterialBlood")] == 60.
    assert ("Stomach|Vascular", "VenousBlood") not in edges
    for organ, exchange in zip(physiology["organs"], system.metadata["exchange_edges"]):
        name = organ["name"]
        vascular, extra = index[name + "|Vascular"], index[name + "|Extravascular"]
        cv, ce = organ["vascular_capacity_l"] / 1.7, organ["extravascular_capacity_l"] * 3 / 1.7
        assert system.blood_equivalent_capacities_l[vascular] == pytest.approx(cv)
        assert system.blood_equivalent_capacities_l[extra] == pytest.approx(ce)
        ps = ce * .4
        assert exchange["exchange_conductance_l_h"] == pytest.approx(ps)
        assert system.matrix[extra, vascular] == pytest.approx(ps / cv)
        assert system.matrix[vascular, extra] == pytest.approx(ps / ce)
        # Equal whole-blood-equivalent concentrations give zero net exchange.
        assert system.matrix[extra, vascular] * cv + system.matrix[extra, extra] * ce == pytest.approx(0., abs=1e-12)
    offdiag = system.matrix.copy()
    np.fill_diagonal(offdiag, 0.)
    assert offdiag.min() >= 0.
    np.testing.assert_allclose(system.matrix.sum(axis=0), 0., atol=1e-12)
    assert not system.matrix.flags.writeable
    assert "not measured permeability" in system.metadata["exchange_interpretation"]


@pytest.mark.parametrize("duration", [0., .35])
@pytest.mark.parametrize("exchange", [.01, 1., 100.])
def test_mass_conservation_across_exchange_fit_bounds(physiology, duration, exchange):
    result = run(physiology, times_h=np.r_[0., np.geomspace(1e-6, 1344., 150)],
        infusion_duration_h=duration, log_parameters={"bidirectional_exchange_scale": math.log(exchange)},
        return_amounts=True)
    np.testing.assert_allclose(result["amounts_mg"].sum(axis=1) + result["eliminated_mg"],
                               result["delivered_mg"], atol=2e-8, rtol=1e-10)
    assert result["mass_balance_relative_error"] < 1e-9
    assert result["amounts_mg"].min() >= 0.
    assert np.diff(result["eliminated_mg"]).min() >= -1e-8


def independent_ode(physiology, times, duration, logs):
    """Concentration/flow equations built independently of the model matrix."""
    organs = {o["name"]: o for o in physiology["organs"]}
    names = ["ArterialBlood", "VenousBlood", "PortalBlood"]
    names += [name + "|" + pool for name in organs for pool in ("Vascular", "Extravascular")]
    names += ["Eliminated"]
    index = {name: i for i, name in enumerate(names)}
    bp = physiology["blood_to_plasma_ratio"]
    portal = set(physiology["circulation"]["portal_contributors"])
    q = {name: o["flow_l_h"] for name, o in organs.items()}
    qp, qt = sum(q[name] for name in portal), q["Lung"]
    kscale, pscale = math.exp(logs["partition_scale"]), math.exp(logs["bidirectional_exchange_scale"])
    clearance = 2. + (8. - 2.) * math.exp(logs["clearance_scale"])

    def rhs(_, amounts, input_rate):
        cv = {name: bp * amounts[index[name + "|Vascular"]] / o["vascular_capacity_l"]
              for name, o in organs.items()}
        ce = {name: bp * amounts[index[name + "|Extravascular"]] / (o["extravascular_capacity_l"] * kscale)
              for name, o in organs.items()}
        ca, cc, cp = [amounts[index[name]] / physiology["blood"][key]["volume_l"]
                     for name, key in [("ArterialBlood", "arterial"), ("VenousBlood", "venous"), ("PortalBlood", "portal")]]
        change = np.zeros(len(names))
        change[index["ArterialBlood"]] = qt * (cv["Lung"] - ca)
        change[index["PortalBlood"]] = sum(q[n] * cv[n] for n in portal) - qp * cp
        returning = 0.
        for name in organs:
            qo = q[name] + (qp if name == "Liver" else 0.)
            incoming = qt * cc if name == "Lung" else q[name] * ca + (qp * cp if name == "Liver" else 0.)
            exchange = (organs[name]["extravascular_capacity_l"] * kscale / bp) * pscale * (cv[name] - ce[name])
            change[index[name + "|Vascular"]] = incoming - qo * cv[name] - exchange
            change[index[name + "|Extravascular"]] = exchange
            if name not in portal and name != "Lung":
                returning += qo * cv[name]
        loss = clearance * cc / bp
        change[index["VenousBlood"]] = returning - qt * cc - loss + input_rate
        change[index["Eliminated"]] = loss
        return change

    initial = np.zeros(len(names))
    if duration == 0.:
        initial[index["VenousBlood"]] = 100.
        sol = solve_ivp(lambda t, a: rhs(t, a, 0.), (0., times[-1]), initial,
                        method="Radau", rtol=2e-10, atol=2e-11, dense_output=True)
        assert sol.success
        return sol.sol(times).T
    first = solve_ivp(lambda t, a: rhs(t, a, 100. / duration), (0., duration), initial,
                      method="Radau", rtol=2e-10, atol=2e-11, dense_output=True)
    second = solve_ivp(lambda t, a: rhs(t, a, 0.), (duration, times[-1]), first.y[:, -1],
                       method="Radau", rtol=2e-10, atol=2e-11, dense_output=True)
    assert first.success and second.success
    return np.array([first.sol(t) if t <= duration else second.sol(t) for t in times])


@pytest.mark.parametrize("duration", [0., .35])
@pytest.mark.parametrize("exchange", [.03, 30.])
def test_all_amounts_match_independent_ode_solver(physiology, duration, exchange):
    logs = {"clearance_scale": math.log(.7), "partition_scale": math.log(2.),
            "bidirectional_exchange_scale": math.log(exchange)}
    times = np.array([0., .001, .1, .35 - 1e-9, .35, .35 + 1e-9, .7, 2., 8.])
    result = run(physiology, times_h=times, infusion_duration_h=duration,
                 log_parameters=logs, return_amounts=True)
    reference = independent_ode(physiology, times, duration, logs)
    np.testing.assert_allclose(result["amounts_mg"], reference[:, :-1], rtol=5e-8, atol=2e-8)
    np.testing.assert_allclose(result["eliminated_mg"], reference[:, -1], rtol=5e-8, atol=2e-8)


@pytest.mark.parametrize("duration", [0., .35])
def test_increasing_exchange_converges_to_frozen_perfusion(physiology, duration):
    times = np.array([.001, .01, .1, .35, .5, 1., 4., 12.])
    logs = {"clearance_scale": math.log(.7), "partition_scale": math.log(2.)}
    kwargs = dict(times_h=times, dose_mg=100., infusion_duration_h=duration,
                  total_clearance_l_h=8., renal_clearance_l_h=2., return_amounts=True)
    limit = perfusion.simulate(physiology, **kwargs, log_parameters=logs)
    errors, state_errors = [], []
    for scale in [1e2, 1e4, 1e6]:
        result = model.simulate(physiology, **kwargs,
            log_parameters={**logs, "bidirectional_exchange_scale": math.log(scale)})
        errors.append(np.max(np.abs(result["concentration_mg_l"] - limit["concentration_mg_l"])))
        summed = []
        for name in limit['state_names']:
            if name in result['state_names']:
                summed.append(result['amounts_mg'][:, result['state_names'].index(name)])
            else:
                summed.append(sum(result['amounts_mg'][:, result['state_names'].index(name+'|'+pool)]
                                  for pool in ('Vascular', 'Extravascular')))
        state_errors.append(np.max(np.abs(np.array(summed).T-limit['amounts_mg']))/100.)
    assert errors[1] < errors[0] * .03
    assert errors[2] < errors[1] * .03
    # A finite 10^6/h rate still has an O(1/rate) early-time boundary layer;
    # the two preceding scale comparisons check convergence rather than an
    # exact identity to the instantaneous-perfusion equations.
    np.testing.assert_allclose(result["concentration_mg_l"], limit["concentration_mg_l"], rtol=2e-4, atol=1e-5)
    # Finite-rate approximation within 0.001% of dose, with first-order
    # convergence of every aggregated state. Independent ODE tests above
    # use much tighter tolerances for the actual finite-rate equations.
    assert state_errors[1] < state_errors[0] * .03
    assert state_errors[2] < state_errors[1] * .03
    assert state_errors[-1] < 1e-5


def test_observer_uses_only_vascular_plasma_and_fixed_renal(physiology):
    result = run(physiology, log_parameters={"clearance_scale": math.log(3.)}, return_amounts=True)
    expected = np.zeros(len(result["time_h"]))
    organs = {o["name"]: o for o in physiology["organs"]}
    for name, weight in [("Muscle", .287), ("Skin", .713)]:
        expected += weight * result["amounts_mg"][:, result["state_names"].index(name + "|Vascular")] / organs[name]["vascular_capacity_l"]
    np.testing.assert_allclose(result["concentration_mg_l"], expected, rtol=1e-14)
    assert result["effective_total_clearance_l_h"] == pytest.approx(20.)
    assert result["model"]["clearance_components_l_h"]["qualified_renal_fixed"] == 2.
    assert "not_validated_organ" in result["model"]["clearance_interpretation"]


def test_no_clearance_equilibrates_without_loss(physiology):
    result = run(physiology, times_h=[0., 1000.], total_clearance_l_h=0., renal_clearance_l_h=0.,
                 infusion_duration_h=0., return_amounts=True)
    assert result["eliminated_mg"][-1] == 0.
    for organ in physiology["organs"]:
        cv = result["amounts_mg"][-1, result["state_names"].index(organ["name"] + "|Vascular")] / organ["vascular_capacity_l"]
        ce = result["amounts_mg"][-1, result["state_names"].index(organ["name"] + "|Extravascular")] / organ["extravascular_capacity_l"]
        assert cv == pytest.approx(ce, rel=1e-9)


def test_vanishing_exchange_leaves_extravascular_pools_empty(physiology):
    result = run(physiology, times_h=[0., .01, .35, 1., 4.],
                 log_parameters={"bidirectional_exchange_scale": math.log(1e-12)},
                 return_amounts=True)
    extra = [i for i, name in enumerate(result["state_names"]) if name.endswith("|Extravascular")]
    assert result["amounts_mg"][:, extra].sum(axis=1).max() < 1e-7
    assert result["concentration_mg_l"].max() > 0.


def test_fallback_and_zero_dose(physiology, monkeypatch):
    reference = run(physiology)
    monkeypatch.setattr(perfusion, "_spectral_states", lambda *args: (None, 1e12))
    result = run(physiology)
    assert result["solver"]["method"] == "matrix_exponential_fallback"
    np.testing.assert_allclose(result["concentration_mg_l"], reference["concentration_mg_l"], rtol=1e-9, atol=1e-9)
    zero = run(physiology, dose_mg=0.)
    np.testing.assert_array_equal(zero["concentration_mg_l"], 0.)


@pytest.mark.parametrize("kwargs", [
    {"family": "global"}, {"log_parameters": {"bidirectional_exchange_scale": float("nan")}},
    {"log_parameters": {"bidirectional_exchange_scale": 1000.}},
    {"log_parameters": {"extra_param": 0.}}, {"times_h": [1., 0.]}, {"times_h": [True]},
    {"renal_clearance_l_h": 10.}, {"dose_mg": -1.}, {"return_amounts": 1}])
def test_invalid_contracts_are_rejected(physiology, kwargs):
    with pytest.raises(ValueError):
        run(physiology, **kwargs)


@pytest.mark.parametrize("field", ["vascular_capacity_l", "extravascular_capacity_l"])
def test_zero_pool_capacity_is_rejected(physiology, field):
    physiology["organs"][0][field] = 0.
    with pytest.raises(ValueError, match="positive"):
        run(physiology)


def test_fast_circulation_limit_is_not_misrepresented_as_unique_organ_information(physiology):
    from copy import deepcopy
    from scipy.linalg import expm
    # A common tissue return rate has a two-pool aggregate limit. Verify this
    # explicitly rather than claiming tissue-specific kinetics are identified.
    times = np.array([.1, .35, 1., 4., 12.])
    bp = physiology['blood_to_plasma_ratio']
    vc = bp*sum(b['volume_l'] for b in physiology['blood'].values()) + sum(o['vascular_capacity_l'] for o in physiology['organs'])
    vp = sum(o['extravascular_capacity_l'] for o in physiology['organs'])
    rate = .4
    q = rate*vp
    matrix = np.array([[-(8.+q)/vc, q/vp], [q/vc, -q/vp]])
    expected = np.array([(expm(matrix*t) @ np.array([100.,0.]))[0]/vc for t in times])
    errors = []
    for speed in (100., 1000.):
        fast = deepcopy(physiology)
        for organ in fast['organs']:
            organ['flow_l_h'] *= speed
        for key in ('cardiac_flow_l_h', 'hepatic_arterial_flow_l_h', 'portal_flow_l_h', 'liver_total_flow_l_h'):
            fast['circulation'][key] *= speed
        predicted = run(fast, times_h=times, infusion_duration_h=0., log_parameters={'bidirectional_exchange_scale': math.log(rate)})
        errors.append(np.max(np.abs(predicted['concentration_mg_l']/expected-1.)))
    assert errors[1] < errors[0]*.15
    assert errors[1] < .001
