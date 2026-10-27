"""Independent equations and moment audits for hepatic clearance relocation."""
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pbpk_kb.blueprint_policy import hepatic_exchange_model as model
from pbpk_kb.blueprint_policy import perfusion_model as propagation


@pytest.fixture
def physiology():
    flows = {"Lung": 60., "Heart": 3., "Liver": 6., "Kidney": 12.,
             "Spleen": 2., "Stomach": 2., "SmallIntestine": 5.,
             "LargeIntestine": 1., "Bone": 3., "Brain": 5., "Fat": 3.,
             "Gonads": 1., "Muscle": 10., "Pancreas": 1., "Skin": 6.}
    return {"version": model.PHYSIOLOGY_VERSION, "blood_to_plasma_ratio": 1.7,
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


def test_hepatic_state_capacity_and_distinct_sinks(physiology):
    system = model.build_system(physiology, total_clearance_l_h=8., renal_clearance_l_h=2.,
        log_parameters={"clearance_scale": math.log(2.), "partition_scale": math.log(3.)})
    names = system.state_names
    assert len(names) == 33
    assert "Liver|Equilibrated" in names
    assert "Liver|Vascular" not in names and "Liver|Extravascular" not in names
    liver = next(o for o in physiology["organs"] if o["name"] == "Liver")
    capacity = liver["vascular_capacity_l"] + 3 * liver["extravascular_capacity_l"]
    assert system.blood_equivalent_capacities_l[names.index("Liver|Equilibrated")] == pytest.approx(capacity / 1.7)
    assert len(system.metadata["exchange_edges"]) == 14
    assert all(e["organ"] != "Liver" for e in system.metadata["exchange_edges"])
    edges = {(e["source"], e["target"]): e["flow_l_h"] for e in system.metadata["flow_edges"]}
    assert edges[("ArterialBlood", "Liver|Equilibrated")] == 6.
    assert edges[("PortalBlood", "Liver|Equilibrated")] == 11.
    assert edges[("Liver|Equilibrated", "VenousBlood")] == 17.
    local = 12. * (1.7 * 17.) / (1.7 * 17. - 12.)
    ledger = system.matrix[-1]
    assert ledger[names.index("VenousBlood")] == pytest.approx(2. / (2. * 1.7))
    assert ledger[names.index("Liver|Equilibrated")] == pytest.approx(local / capacity)
    assert np.count_nonzero(ledger) == 2
    assert system.metadata["hepatic_clearance"]["central_nonrenal_clearance_l_h"] == 0.
    assert system.metadata["hepatic_clearance"]["local_plasma_equivalent_clearance_l_h"] == pytest.approx(local)
    np.testing.assert_allclose(system.matrix.sum(axis=0), 0., atol=1e-12)
    offdiag = system.matrix.copy()
    np.fill_diagonal(offdiag, 0.)
    assert offdiag.min() >= 0.


@pytest.mark.parametrize("clscale", [.25, 1., 4.])
@pytest.mark.parametrize("partition,exchange", [(.1, .01), (10., 100.)])
def test_unit_dose_moments_preserve_systemic_and_native_observer_clearance(physiology, clscale, partition, exchange):
    system = model.build_system(physiology, total_clearance_l_h=8., renal_clearance_l_h=2.,
        log_parameters={"clearance_scale": math.log(clscale), "partition_scale": math.log(partition),
                        "bidirectional_exchange_scale": math.log(exchange)})
    moments = system.metadata["infinite_time_moments"]
    assert moments["central_clearance_equivalent"] is True
    assert moments["observer_clearance_equivalent"] is True
    assert moments["central_venous_dose_over_auc_l_h"] == pytest.approx(2. + 6. * clscale, rel=1e-10)
    assert moments["observer_auc_mg_h_l"] == pytest.approx(moments["central_venous_auc_mg_h_l"], rel=1e-10)
    # Independent integral solution also resolves the expected renal/hepatic
    # fractions of dose, detecting an extra unremoved central nonrenal sink.
    integrated = np.linalg.solve(-system.matrix[:-1, :-1], system.input_vector[:-1])
    names = system.state_names
    renal_fraction = system.matrix[-1, names.index("VenousBlood")] * integrated[names.index("VenousBlood")]
    hepatic_fraction = system.matrix[-1, names.index("Liver|Equilibrated")] * integrated[names.index("Liver|Equilibrated")]
    assert renal_fraction == pytest.approx(2. / (2. + 6. * clscale), rel=1e-10)
    assert renal_fraction + hepatic_fraction == pytest.approx(1., rel=1e-10)


def test_a_hepatic_outflow_observer_is_not_mislabelled_as_systemic_auc(physiology):
    physiology["peripheral_venous_weights"] = {"Liver": 1.}
    system = model.build_system(physiology, total_clearance_l_h=8., renal_clearance_l_h=2.)
    moments = system.metadata["infinite_time_moments"]
    assert moments["central_clearance_equivalent"] is True
    assert moments["observer_clearance_equivalent"] is False
    assert moments["observer_auc_mg_h_l"] < moments["central_venous_auc_mg_h_l"]


@pytest.mark.parametrize("duration", [0., .35])
@pytest.mark.parametrize("exchange", [.01, 100.])
def test_body_plus_eliminated_conserves_dose(physiology, duration, exchange):
    result = run(physiology, times_h=np.r_[0., np.geomspace(1e-6, 1344., 150)],
                 infusion_duration_h=duration, return_amounts=True,
                 log_parameters={"bidirectional_exchange_scale": math.log(exchange)})
    np.testing.assert_allclose(result["amounts_mg"].sum(axis=1) + result["eliminated_mg"],
                               result["delivered_mg"], atol=2e-8, rtol=1e-10)
    assert result["mass_balance_relative_error"] < 1e-9
    assert result["amounts_mg"].min() >= 0.
    assert np.diff(result["eliminated_mg"]).min() >= -1e-8


def independent_ode(physiology, times, duration, logs):
    """Explicit inlet/outlet concentration balances, without implementation A."""
    organs = {o["name"]: o for o in physiology["organs"]}
    names = ["ArterialBlood", "VenousBlood", "PortalBlood"]
    for name in organs:
        names += ["Liver|Equilibrated"] if name == "Liver" else [name + "|Vascular", name + "|Extravascular"]
    names += ["Eliminated"]
    idx = {name: i for i, name in enumerate(names)}
    bp = physiology["blood_to_plasma_ratio"]
    portal = set(physiology["circulation"]["portal_contributors"])
    q = {name: o["flow_l_h"] for name, o in organs.items()}
    qp, qt = sum(q[name] for name in portal), q["Lung"]
    kscale, pscale = math.exp(logs["partition_scale"]), math.exp(logs["bidirectional_exchange_scale"])
    nonrenal = 6. * math.exp(logs["clearance_scale"])
    local = nonrenal * bp * (q["Liver"] + qp) / (bp * (q["Liver"] + qp) - nonrenal)

    def rhs(_, amounts, input_rate):
        cv, ce = {}, {}
        for name, organ in organs.items():
            if name == "Liver":
                cv[name] = bp * amounts[idx["Liver|Equilibrated"]] / (organ["vascular_capacity_l"] + kscale * organ["extravascular_capacity_l"])
            else:
                cv[name] = bp * amounts[idx[name + "|Vascular"]] / organ["vascular_capacity_l"]
                ce[name] = bp * amounts[idx[name + "|Extravascular"]] / (kscale * organ["extravascular_capacity_l"])
        ca, cc, cp = [amounts[idx[name]] / physiology["blood"][key]["volume_l"]
                     for name, key in [("ArterialBlood", "arterial"), ("VenousBlood", "venous"), ("PortalBlood", "portal")]]
        change = np.zeros(len(names))
        change[idx["ArterialBlood"]] = qt * (cv["Lung"] - ca)
        change[idx["PortalBlood"]] = sum(q[n] * cv[n] for n in portal) - qp * cp
        returning = 0.
        hepatic_loss = local * cv["Liver"] / bp
        for name in organs:
            qo = q[name] + (qp if name == "Liver" else 0.)
            incoming = qt * cc if name == "Lung" else q[name] * ca + (qp * cp if name == "Liver" else 0.)
            if name == "Liver":
                change[idx["Liver|Equilibrated"]] = incoming - qo * cv[name] - hepatic_loss
            else:
                flux = q[name] * pscale * (cv[name] - ce[name])
                change[idx[name + "|Vascular"]] = incoming - qo * cv[name] - flux
                change[idx[name + "|Extravascular"]] = flux
            if name not in portal and name != "Lung":
                returning += qo * cv[name]
        renal_loss = 2. * cc / bp
        change[idx["VenousBlood"]] = returning - qt * cc - renal_loss + input_rate
        change[idx["Eliminated"]] = renal_loss + hepatic_loss
        return change

    initial = np.zeros(len(names))
    if duration == 0.:
        initial[idx["VenousBlood"]] = 100.
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
def test_all_states_match_independent_ode(physiology, duration, exchange):
    logs = {"clearance_scale": math.log(.7), "partition_scale": math.log(2.),
            "bidirectional_exchange_scale": math.log(exchange)}
    times = np.array([0., .001, .1, .35 - 1e-9, .35, .35 + 1e-9, .7, 2., 8.])
    result = run(physiology, times_h=times, infusion_duration_h=duration,
                 log_parameters=logs, return_amounts=True)
    independent = independent_ode(physiology, times, duration, logs)
    np.testing.assert_allclose(result["amounts_mg"], independent[:, :-1], rtol=5e-8, atol=2e-8)
    np.testing.assert_allclose(result["eliminated_mg"], independent[:, -1], rtol=5e-8, atol=2e-8)


@pytest.mark.parametrize("renal,total", [(0., 0.), (2., 2.), (None, 8.)])
def test_zero_clearance_renal_only_and_unsplit_cases(physiology, renal, total):
    result = run(physiology, renal_clearance_l_h=renal, total_clearance_l_h=total,
                 return_amounts=True)
    if total == 0.:
        np.testing.assert_array_equal(result["eliminated_mg"], 0.)
        assert result["model"]["infinite_time_moments"]["status"] == "undefined_zero_clearance"
    if renal == 2.:
        assert result["model"]["hepatic_clearance"]["local_plasma_equivalent_clearance_l_h"] == 0.
    if renal is None:
        assert result["model"]["hepatic_clearance"]["central_renal_clearance_l_h"] == 0.


@pytest.mark.parametrize("excess", [0., 1e-10, 1., 1000.])
def test_hepatic_flow_capacity_is_a_hard_gate(physiology, excess):
    with pytest.raises(ValueError, match="CL_NR < BP"):
        run(physiology, total_clearance_l_h=2. + 1.7 * 17. + excess)


def test_native_observer_uses_nonhepatic_vascular_states(physiology):
    result = run(physiology, return_amounts=True)
    organs = {o["name"]: o for o in physiology["organs"]}
    expected = sum(w * result["amounts_mg"][:, result["state_names"].index(name + "|Vascular")] / organs[name]["vascular_capacity_l"]
                   for name, w in [("Muscle", .287), ("Skin", .713)])
    np.testing.assert_allclose(result["concentration_mg_l"], expected, atol=1e-14)


def test_fallback_and_zero_dose(physiology, monkeypatch):
    reference = run(physiology)
    monkeypatch.setattr(propagation, "_spectral_states", lambda *args: (None, 1e12))
    result = run(physiology)
    assert result["solver"]["method"] == "matrix_exponential_fallback"
    np.testing.assert_allclose(result["concentration_mg_l"], reference["concentration_mg_l"], rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(run(physiology, dose_mg=0.)["concentration_mg_l"], 0.)


@pytest.mark.parametrize("kwargs", [{"times_h": [True]}, {"times_h": [1., 0.]},
    {"dose_mg": -1.}, {"return_amounts": 1}, {"family": "global"},
    {"log_parameters": {"hepatic_extra_parameter": 0.}}])
def test_invalid_contracts_rejected(physiology, kwargs):
    with pytest.raises(ValueError):
        run(physiology, **kwargs)
