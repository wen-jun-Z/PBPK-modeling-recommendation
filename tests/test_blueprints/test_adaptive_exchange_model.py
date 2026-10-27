"""Independent physical and numerical checks for the exchange-shape extension."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pbpk_kb.blueprint_policy import adaptive_exchange_model as model
from pbpk_kb.blueprint_policy import robust_exchange_model as baseline

spec = importlib.util.spec_from_file_location('_adaptive_physiology_fixture',
    Path(__file__).with_name('test_perfusion_model.py'))
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


@pytest.fixture
def physiology():
    return fixture_module.physiology.__wrapped__()


def simulate(physiology, **overrides):
    arguments = dict(times_h=np.r_[0., np.geomspace(1e-5, 168., 65)], dose_mg=100.,
        infusion_duration_h=.5, total_clearance_l_h=8., renal_clearance_l_h=2.)
    arguments.update(overrides)
    return model.simulate(physiology, **arguments)


@pytest.mark.parametrize('logs', [{}, {'clearance_scale': math.log(.05), 'partition_scale': math.log(.1),
    'bidirectional_exchange_scale': math.log(.01)}, {'clearance_scale': math.log(20.),
    'partition_scale': math.log(10.), 'bidirectional_exchange_scale': math.log(100.)}])
@pytest.mark.parametrize('duration', [0., .5])
def test_zero_alpha_is_exact_frozen_system_and_trajectory(physiology, logs, duration):
    kwargs = dict(total_clearance_l_h=8., renal_clearance_l_h=2., log_parameters=logs)
    original = baseline.build_system(physiology, **kwargs)
    extended = model.build_system(physiology, **kwargs)
    assert original.state_names == extended.state_names
    for name in ('matrix', 'input_vector', 'observation_vector', 'venous_plasma_vector',
                 'blood_equivalent_capacities_l'):
        np.testing.assert_array_equal(getattr(original, name), getattr(extended, name))
    arguments = dict(times_h=np.r_[0., np.geomspace(1e-5, 168., 65)], dose_mg=100.,
        infusion_duration_h=duration, return_amounts=True, **kwargs)
    expected = baseline.simulate(physiology, **arguments)
    actual = model.simulate(physiology, **arguments)
    for name in ('concentration_mg_l', 'venous_plasma_mg_l', 'amounts_mg', 'eliminated_mg'):
        np.testing.assert_array_equal(actual[name], expected[name])


def test_shape_preserves_flow_observer_capacity_and_fixed_renal(physiology):
    logs = {'clearance_scale': math.log(.2), 'partition_scale': math.log(2.),
            'bidirectional_exchange_scale': math.log(.3), 'exchange_heterogeneity_scale': -.4}
    changed = model.build_system(physiology, total_clearance_l_h=8., renal_clearance_l_h=2., log_parameters=logs)
    original = baseline.build_system(physiology, total_clearance_l_h=8., renal_clearance_l_h=2.,
        log_parameters={k: v for k, v in logs.items() if k != 'exchange_heterogeneity_scale'})
    assert changed.metadata['flow_edges'] == original.metadata['flow_edges']
    assert changed.metadata['observation_weights'] == original.metadata['observation_weights']
    for field in ('input_vector', 'observation_vector', 'venous_plasma_vector', 'blood_equivalent_capacities_l'):
        np.testing.assert_array_equal(getattr(changed, field), getattr(original, field))
    assert changed.metadata['effective_total_clearance_l_h'] == pytest.approx(3.2)
    assert changed.metadata['clearance_components_l_h']['qualified_renal_fixed'] == 2.
    np.testing.assert_array_equal(changed.matrix[-1], original.matrix[-1])
    edges = changed.metadata['exchange_edges']
    assert sum(e['centering_flow_weight'] * e['centered_log_time_constant'] for e in edges) == pytest.approx(0., abs=1e-14)
    assert sum(e['centering_flow_weight'] for e in edges) == pytest.approx(1.)
    assert next(e for e in edges if e['organ'] == 'Liver')['effective_flow_l_h'] == 17.
    assert not changed.matrix.flags.writeable


def test_contrast_ignores_fitted_capacity_and_recomputes_for_new_subject(physiology):
    def edges(subject, scale):
        system = model.build_system(subject, total_clearance_l_h=8.,
            log_parameters={'partition_scale': math.log(scale), 'exchange_heterogeneity_scale': .3})
        return {e['organ']: e for e in system.metadata['exchange_edges']}
    a, b = edges(physiology, .1), edges(physiology, 10.)
    for organ in a:
        assert a[organ]['centered_log_time_constant'] == b[organ]['centered_log_time_constant']
        assert a[organ]['exchange_conductance_l_h'] == b[organ]['exchange_conductance_l_h']
    other = deepcopy(physiology)
    next(o for o in other['organs'] if o['name'] == 'Fat')['extravascular_capacity_l'] *= 4.
    c = edges(other, .1)
    assert c['Fat']['centered_log_time_constant'] > a['Fat']['centered_log_time_constant']
    assert c['Fat']['exchange_conductance_l_h'] > a['Fat']['exchange_conductance_l_h']
    # Centering also adjusts the remaining organs, rather than carrying h from
    # the training subject into the new person's prediction.
    assert c['Muscle']['centered_log_time_constant'] < a['Muscle']['centered_log_time_constant']


def test_alpha_changes_relative_return_times_by_the_declared_power(physiology):
    systems = [model.build_system(physiology, total_clearance_l_h=8.,
        log_parameters={'exchange_heterogeneity_scale': alpha}) for alpha in (0., -.5, .5)]
    ratios = []
    for system in systems:
        e = {r['organ']: r for r in system.metadata['exchange_edges']}
        ratios.append(e['Lung']['extravascular_return_rate_per_h'] / e['Fat']['extravascular_return_rate_per_h'])
    assert ratios[0] > 1
    assert ratios[1] == pytest.approx(ratios[0] ** 1.5)
    assert ratios[2] == pytest.approx(ratios[0] ** .5)


@pytest.mark.parametrize('alpha', [-.5, .5])
@pytest.mark.parametrize('duration', [0., .5])
@pytest.mark.parametrize('scales', [(.05, .1, .01), (20., 10., 100.)])
def test_boundaries_are_nonnegative_conservative_and_stable(physiology, alpha, duration, scales):
    logs = dict(zip(model.PARAMETERS['adaptive_exchange'][:3], map(math.log, scales)))
    logs['exchange_heterogeneity_scale'] = alpha
    system = model.build_system(physiology, total_clearance_l_h=8., renal_clearance_l_h=2., log_parameters=logs)
    off = system.matrix.copy(); np.fill_diagonal(off, 0.)
    assert off.min() >= 0
    np.testing.assert_allclose(system.matrix.sum(axis=0), 0., atol=2e-10)
    assert np.linalg.eigvals(system.matrix[:-1, :-1]).real.max() < 0
    actual = simulate(physiology, log_parameters=logs, infusion_duration_h=duration, return_amounts=True)
    assert actual['amounts_mg'].min() >= 0
    assert np.diff(actual['eliminated_mg']).min() >= -1e-8
    np.testing.assert_allclose(actual['amounts_mg'].sum(axis=1) + actual['eliminated_mg'],
        actual['delivered_mg'], atol=2e-8, rtol=1e-10)


def test_dose_linearity_zero_clearance_and_infinite_auc_identity(physiology):
    logs = {'exchange_heterogeneity_scale': -.4, 'bidirectional_exchange_scale': math.log(.2)}
    one = simulate(physiology, log_parameters=logs, dose_mg=30.)
    two = simulate(physiology, log_parameters=logs, dose_mg=90.)
    np.testing.assert_allclose(two['concentration_mg_l'], 3 * one['concentration_mg_l'], atol=1e-10, rtol=1e-10)
    zero = simulate(physiology, log_parameters=logs, total_clearance_l_h=0., renal_clearance_l_h=None, return_amounts=True)
    np.testing.assert_array_equal(zero['eliminated_mg'], 0.)
    np.testing.assert_allclose(zero['amounts_mg'].sum(axis=1), zero['delivered_mg'], atol=1e-8)
    system = model.build_system(physiology, total_clearance_l_h=8., log_parameters=logs)
    integrated_unit_amounts = np.linalg.solve(-system.matrix[:-1, :-1], system.input_vector[:-1])
    for observer in (system.observation_vector, system.venous_plasma_vector):
        assert observer[:-1] @ integrated_unit_amounts == pytest.approx(1 / 8., rel=1e-11)


def test_prediction_matches_independent_concentration_balance(physiology):
    # Assemble derivatives directly from flow and reversible concentration
    # differences; do not integrate the production system matrix as an oracle.
    logs = {'exchange_heterogeneity_scale': -.35, 'partition_scale': math.log(1.8),
            'bidirectional_exchange_scale': math.log(.12)}
    system = model.build_system(physiology, total_clearance_l_h=8., log_parameters=logs)
    index = {name: i for i, name in enumerate(system.state_names)}
    capacities = system.blood_equivalent_capacities_l
    def derivative(t, amounts, infusion_rate):
        concentrations = amounts[:-1] / capacities
        out = np.zeros_like(amounts)
        for edge in system.metadata['flow_edges']:
            i, j = index[edge['source']], index[edge['target']]
            flux = edge['flow_l_h'] * concentrations[i]
            out[i] -= flux; out[j] += flux
        for edge in system.metadata['exchange_edges']:
            i, j = index[edge['vascular_state']], index[edge['extravascular_state']]
            flux = edge['exchange_conductance_l_h'] * (concentrations[i] - concentrations[j])
            out[i] -= flux; out[j] += flux
        eliminated = 8. * concentrations[index['VenousBlood']] / physiology['blood_to_plasma_ratio']
        out[index['VenousBlood']] += infusion_rate - eliminated
        out[-1] += eliminated
        return out
    times = np.array([0., .03, .2, .5, .8, 2., 6.])
    a = solve_ivp(lambda t, x: derivative(t, x, 200.), (0., .5), np.zeros(len(index)),
        t_eval=times[times <= .5], method='DOP853', rtol=1e-10, atol=1e-11)
    b = solve_ivp(lambda t, x: derivative(t, x, 0.), (.5, 6.), a.y[:, -1],
        t_eval=times[times > .5], method='DOP853', rtol=1e-10, atol=1e-11)
    expected = np.vstack([a.y.T, b.y.T]) @ system.observation_vector
    actual = simulate(physiology, times_h=times, log_parameters=logs, renal_clearance_l_h=None)
    np.testing.assert_allclose(actual['concentration_mg_l'], expected, rtol=1e-7, atol=2e-9)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, 2.01, -2.01])
def test_unsafe_alpha_is_rejected(physiology, bad):
    with pytest.raises(ValueError):
        model.build_system(physiology, total_clearance_l_h=8.,
            log_parameters={'exchange_heterogeneity_scale': bad})
