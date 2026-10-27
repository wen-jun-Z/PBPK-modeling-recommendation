"""Synthetic physiology only: partition bookkeeping, provenance and mass."""
from copy import deepcopy
import math

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import blood_partition_calibration as model
from pbpk_kb.blueprint_policy import permeability_model as exchange


def region(name, vascular_volume, extra_volume, kp, *, hct=.47, tissue_kp=3.):
    volumes = {"Plasma": vascular_volume * (1 - hct), "BloodCells": vascular_volume * hct,
               "Interstitial": extra_volume * .3, "Intracellular": extra_volume * .7}
    parts = [{"path": name + "|" + kind, "kind": kind, "volume_l": volume,
              "kp_plasma": 1. if kind == "Plasma" else kp if kind == "BloodCells" else tissue_kp,
              "volume_path": name + "|" + kind + "|Volume", "kp_path": None if kind == "Plasma" else "partition"}
             for kind, volume in volumes.items() if volume > 0]
    for part in parts:
        part["capacity_l"] = part["volume_l"] * part["kp_plasma"]
    capacity = volumes["Plasma"] + volumes["BloodCells"] * kp
    return {"name": name, "volume_l": vascular_volume + extra_volume,
            "plasma_volume_l": volumes["Plasma"], "rbc_volume_l": volumes["BloodCells"],
            "interstitial_volume_l": volumes["Interstitial"], "intracellular_volume_l": volumes["Intracellular"],
            "vascular_volume_l": vascular_volume, "extravascular_volume_l": extra_volume,
            "vascular_capacity_l": capacity, "extravascular_capacity_l": extra_volume * tissue_kp,
            "kp_extravascular_plasma": tissue_kp if extra_volume else None,
            "kp_total_plasma": (capacity + extra_volume * tissue_kp) / (vascular_volume + extra_volume),
            "components": parts}


@pytest.fixture
def physiology():
    flows = {"Lung": 60., "Heart": 3., "Liver": 6., "Kidney": 12.,
             "Spleen": 2., "Stomach": 2., "SmallIntestine": 5., "LargeIntestine": 1.,
             "Bone": 3., "Brain": 5., "Fat": 3., "Gonads": 1., "Muscle": 10., "Pancreas": 1., "Skin": 6.}
    kp = 26.94
    organs = []
    for i, (name, q) in enumerate(flows.items()):
        row = region(name, .05 + .01 * i, .3 + .2 * i, kp)
        row.update(flow_l_h=q, flow_l_min=q / 60.)
        organs.append(row)
    bp = .53 + .47 * kp
    return {"version": "native-perfusion-physiology-1", "source_kind": "fresh_evidence_native_model",
            "source_execution_manifest_sha256": "a" * 64, "source_pins": {"source": "b" * 64},
            "artifact_pins": {"native_readback": "c" * 64}, "solver_calls": 0, "observations_used": False,
            "rbc_kp_plasma": kp, "blood_to_plasma_ratio": bp, "hematocrit": .47,
            "equilibrium_blood_to_plasma_ratio_check": bp, "fraction_unbound_plasma": .07,
            "blood": {"arterial": region("ArterialBlood", .7, 0., kp),
                      "venous": region("VenousBlood", 2., 0., kp), "portal": region("PortalVein", .2, 0., kp)},
            "organs": organs,
            "circulation": {"portal_contributors": ["Stomach", "SmallIntestine", "LargeIntestine", "Pancreas", "Spleen"],
                            "cardiac_flow_l_h": 60., "hepatic_arterial_flow_l_h": 6.,
                            "portal_flow_l_h": 11., "liver_total_flow_l_h": 17.},
            "peripheral_venous_weights": {"Muscle": .287, "Skin": .713}}


@pytest.fixture
def req():
    context = {"species": "human", "matrix": "plasma", "route": "intravenous"}
    return {"compound_id": "mock", "human": {"weight_kg": 70.},
            "dosing": {"dose_mg": 100., "route": "intravenous", "infusion_minutes": 30., "start_h": 0.},
            "evidence": [{"parameter": "molecular_weight", "value": 400.},
                         {"parameter": "total_clearance", "value": 8. / 4.2, "unit": "ml/min/kg", "context": context},
                         {"parameter": "renal_clearance", "value": 2. / 4.2, "unit": "ml/min/kg", "context": context}]}


def measured(req, parameter, value, **overrides):
    row = {"parameter": parameter, "value": value, "compound_id": "mock", "unit": "1", "basis": "measured",
           "context": {"species": "human"}, "source": {"path": "source.json", "sha256": "d" * 64}}
    row.update(overrides)
    req["evidence"].append(row)
    return row


def test_identity_reproduces_frozen_exchange_with_unchanged_inputs(physiology, req):
    old_phys, old_request = deepcopy(physiology), deepcopy(req)
    times = np.array([0., .001, .1, .5, 1., 4., 12.])
    first_three = [.1, -.3, .2]
    old = exchange.simulate(physiology, times_h=times, dose_mg=100., infusion_duration_h=.5,
                           total_clearance_l_h=8., renal_clearance_l_h=2.,
                           log_parameters=dict(zip(model.PARAMETERS[:3], first_three)))
    new = model.simulate(physiology, req, times, first_three + [0.])
    np.testing.assert_allclose(new["concentration_mg_l"], old["concentration_mg_l"], atol=1e-11, rtol=1e-10)
    assert physiology == old_phys and req == old_request
    assert new["blood_partition_audit"]["source_rbc_kp_plasma"] == physiology["rbc_kp_plasma"]
    assert new["blood_partition_audit"]["derived_not_native_readback"]


def test_consistent_derived_compartments_and_no_changes_to_other_inputs(physiology, req):
    derived = model.derive_physiology(physiology, req, math.log(.1))
    kp = physiology["rbc_kp_plasma"] * .1
    assert derived["blood_to_plasma_ratio"] == pytest.approx(.53 + .47 * kp)
    assert derived["circulation"] == physiology["circulation"]
    assert derived["fraction_unbound_plasma"] == physiology["fraction_unbound_plasma"]
    for old, new in zip([*physiology["blood"].values(), *physiology["organs"]],
                        [*derived["blood"].values(), *derived["organs"]]):
        assert new["vascular_capacity_l"] == pytest.approx(old["plasma_volume_l"] + old["rbc_volume_l"] * kp)
        assert new["kp_total_plasma"] == pytest.approx((new["vascular_capacity_l"] + old["extravascular_capacity_l"]) / old["volume_l"])
        for key, value in old.items():
            if key not in {"vascular_capacity_l", "kp_total_plasma", "components"}:
                assert new[key] == value
        for a, b in zip(old["components"], new["components"]):
            if a["kind"] == "BloodCells":
                assert b["kp_plasma"] == pytest.approx(kp)
                assert b["capacity_l"] == pytest.approx(a["volume_l"] * kp)
            else:
                assert a == b
    assert derived["source_kind"] == "derived_blood_partition_model"
    with pytest.raises(ValueError, match="untouched"):
        model.derive_physiology(derived, req, 0.)


@pytest.mark.parametrize("scale", [.01, 1., 10.])
@pytest.mark.parametrize("duration", [0., 30.])
def test_mass_conservation_and_fixed_net_renal_clearance(physiology, req, scale, duration):
    req["dosing"]["infusion_minutes"] = duration
    result = model.simulate(physiology, req, np.r_[0., np.geomspace(1e-5, 240., 100)],
                            [math.log(.5), math.log(2.), math.log(.3), math.log(scale)])
    assert result["mass_balance"]["maximum_relative_error"] < 1e-8
    assert result["effective_total_clearance_l_h"] == pytest.approx(5.)
    assert result["model"]["clearance_components_l_h"] == pytest.approx({"qualified_renal_fixed": 2., "scaled_nonrenal": 3.})


def test_whole_blood_and_plasma_consistent_after_partition_change(physiology, req):
    physiology["peripheral_venous_weights"] = {}
    req["dosing"]["infusion_minutes"] = 0.
    scale = .03
    result = model.simulate(physiology, req, [0., .1], [0., 0., 0., math.log(scale)])
    bp = .53 + .47 * physiology["rbc_kp_plasma"] * scale
    plasma = result["concentration_mg_l"][0]
    whole_blood = 100. / physiology["blood"]["venous"]["volume_l"]
    assert plasma * bp == pytest.approx(whole_blood)
    derived = model.derive_physiology(physiology, req, math.log(scale))
    system = exchange.build_system(derived, total_clearance_l_h=8., renal_clearance_l_h=2.)
    for key, state in [("arterial", "ArterialBlood"), ("venous", "VenousBlood"), ("portal", "PortalBlood")]:
        assert system.blood_equivalent_capacities_l[system.state_names.index(state)] == physiology["blood"][key]["volume_l"]
    for organ in derived["organs"]:
        capacity = system.blood_equivalent_capacities_l[system.state_names.index(organ["name"] + "|Vascular")]
        assert capacity * bp == pytest.approx(organ["vascular_capacity_l"])
    np.testing.assert_allclose(system.matrix.sum(axis=0), 0., atol=1e-10)


@pytest.mark.parametrize("parameter", ["blood_to_plasma_ratio", "rbc_kp_plasma", "blood_plasma_ratio", "K_rbc"])
def test_measured_partition_is_protected_and_identity_checked(physiology, req, parameter):
    value = physiology["rbc_kp_plasma"] if parameter in {"rbc_kp_plasma", "K_rbc"} else physiology["blood_to_plasma_ratio"]
    row = measured(req, parameter, value)
    result = model.derive_physiology(physiology, req, 0.)
    assert result["blood_partition_audit"]["policy"]["rule"] == "preserve_measured_partition"
    with pytest.raises(ValueError, match="protected"):
        model.derive_physiology(physiology, req, .01)
    row["value"] = value * 2
    with pytest.raises(ValueError, match="measured"):
        model.derive_physiology(physiology, req, 0.)


@pytest.mark.parametrize("update", [{"basis": "predicted"}, {"basis": "unknown"}, {"basis": "derived"},
                                    {"context": {}}, {"context": {"species": "rat"}}, {"source": {}},
                                    {"compound_id": "another"}, {"unit": "%"}])
def test_unknown_or_mixed_explicit_sources_fail_closed(physiology, req, update):
    measured(req, "blood_to_plasma_ratio", physiology["blood_to_plasma_ratio"])
    measured(req, "rbc_kp_plasma", physiology["rbc_kp_plasma"], **update)
    with pytest.raises(ValueError, match="explicit blood-partition"):
        model.derive_physiology(physiology, req, 0.)


@pytest.mark.parametrize("update", [{"source_kind": "unknown"}, {"observations_used": True},
                                    {"source_execution_manifest_sha256": "unknown"}, {"source_pins": {}},
                                    {"artifact_pins": {}}, {"solver_calls": 1},
                                    {"rbc_kp_plasma_source": "mixed"}])
def test_native_source_contract_is_required(physiology, req, update):
    physiology.update(update)
    with pytest.raises(ValueError):
        model.derive_physiology(physiology, req, 0.)


def test_native_component_capacity_and_blood_hct_inconsistency_rejected(physiology, req):
    wrong = deepcopy(physiology)
    wrong["organs"][0]["components"][0]["capacity_l"] *= 2
    with pytest.raises(ValueError, match="component capacity"):
        model.derive_physiology(wrong, req, 0.)
    wrong = deepcopy(physiology)
    wrong["blood"]["arterial"] = region("ArterialBlood", .7, 0., wrong["rbc_kp_plasma"], hct=.3)
    with pytest.raises(ValueError, match="hematocrit"):
        model.derive_physiology(wrong, req, 0.)
    wrong = deepcopy(physiology)
    wrong["blood_to_plasma_ratio"] *= 2
    with pytest.raises(ValueError, match="B/P"):
        model.derive_physiology(wrong, req, 0.)


@pytest.mark.parametrize("logs", [[0., 0., 0.], [0., 0., 0., float("nan")],
                                 [0., 0., 0., 1000.], [0., 0., 0., -1000.],
                                 [True, 0., 0., 0.],
                                 [False, False, False, False], ["0", "0", "0", "0"]])
def test_invalid_log_parameters_rejected(physiology, req, logs):
    with pytest.raises(ValueError):
        model.simulate(physiology, req, [0., .1], logs)
