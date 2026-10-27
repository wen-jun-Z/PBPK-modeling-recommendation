"""Conservation, fixed physiology and the alpha-zero structural control."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pbpk_kb.blueprint_policy import robust_exchange_model as model
from pbpk_kb.blueprint_policy import permeability_model as original

spec = importlib.util.spec_from_file_location('_flow_fixture', Path(__file__).with_name('test_permeability_model.py'))
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


@pytest.fixture
def physiology():
    return fixture_module.physiology.__wrapped__()


@pytest.mark.parametrize('logs', [{}, {'clearance_scale': math.log(.08), 'partition_scale': math.log(3.),
    'bidirectional_exchange_scale': math.log(.011)}, {'clearance_scale': math.log(14.),
    'partition_scale': math.log(.13), 'bidirectional_exchange_scale': math.log(45.)}])
def test_alpha_zero_reproduces_frozen_matrix_and_trajectory_exactly(physiology, logs):
    arguments = {'total_clearance_l_h': 8., 'renal_clearance_l_h': 2., 'log_parameters': logs}
    a = original.build_system(physiology, **arguments)
    b = model.build_system(physiology, family='evidence_flexible', **arguments)
    for field in ('matrix', 'input_vector', 'observation_vector', 'venous_plasma_vector', 'blood_equivalent_capacities_l'):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    kwargs = dict(times_h=np.r_[0., np.geomspace(1e-5, 168., 80)], dose_mg=100., infusion_duration_h=.5, **arguments)
    old = original.simulate(physiology, **kwargs)
    new = model.simulate(physiology, family='evidence_flexible', **kwargs)
    np.testing.assert_array_equal(old['concentration_mg_l'], new['concentration_mg_l'])


def test_balanced_exchange_keeps_q_capacities_and_renal_while_changing_only_ps(physiology):
    logs = {'clearance_scale': math.log(.07), 'partition_scale': math.log(2.4), 'bidirectional_exchange_scale': math.log(.17)}
    kwargs = dict(total_clearance_l_h=8., renal_clearance_l_h=2., log_parameters=logs)
    a = original.build_system(physiology, **kwargs)
    b = model.build_system(physiology, family='balanced_exchange', **kwargs)
    np.testing.assert_array_equal(a.blood_equivalent_capacities_l, b.blood_equivalent_capacities_l)
    np.testing.assert_array_equal(a.observation_vector, b.observation_vector)
    assert a.metadata['flow_edges'] == b.metadata['flow_edges']
    assert b.metadata['clearance_components_l_h']['qualified_renal_fixed'] == 2.
    assert b.metadata['effective_total_clearance_l_h'] == pytest.approx(2. + 6. * .07)
    index = {name: i for i, name in enumerate(b.state_names)}
    for organ, edge in zip(physiology['organs'], b.metadata['exchange_edges']):
        q = 17. if organ['name'] == 'Liver' else organ['flow_l_h']
        capacity = organ['extravascular_capacity_l'] * 2.4 / 1.7
        expected = .17 * math.sqrt(q * capacity)
        assert edge['exchange_conductance_l_h'] == pytest.approx(expected)
        assert edge['extravascular_return_rate_per_h'] == pytest.approx(expected / capacity)
        vi, ei = index[edge['vascular_state']], index[edge['extravascular_state']]
        cv, ce = b.blood_equivalent_capacities_l[[vi, ei]]
        assert b.matrix[ei, vi] * cv + b.matrix[ei, ei] * ce == pytest.approx(0., abs=1e-11)
    np.testing.assert_allclose(b.matrix.sum(axis=0), 0., atol=1e-11)
    off = b.matrix.copy(); np.fill_diagonal(off, 0)
    assert off.min() >= 0
    assert b.metadata['exchange_alpha'] == .5


@pytest.mark.parametrize('family', ['evidence_flexible', 'balanced_exchange'])
@pytest.mark.parametrize('duration', [0., .5])
@pytest.mark.parametrize('scales', [(.05, .1, .01), (1., 1., 1.), (20., 10., 100.)])
def test_broad_prior_bounds_remain_nonnegative_and_conservative(physiology, family, duration, scales):
    result = model.simulate(physiology, times_h=np.r_[0., np.geomspace(1e-6, 1344., 100)], dose_mg=100.,
        infusion_duration_h=duration, total_clearance_l_h=8., renal_clearance_l_h=2., family=family,
        log_parameters=dict(zip(model.PARAMETERS[family], map(math.log, scales))), return_amounts=True)
    np.testing.assert_allclose(result['amounts_mg'].sum(axis=1) + result['eliminated_mg'], result['delivered_mg'], atol=2e-8, rtol=1e-10)
    assert result['amounts_mg'].min() >= 0
    assert result['mass_balance_relative_error'] < 1e-9


def test_balanced_spectral_solution_matches_independent_adaptive_ode(physiology):
    # This independently integrates the conservative matrix, rather than using
    # the same spectral propagation that is being checked.
    logs = {'clearance_scale': math.log(.13), 'partition_scale': math.log(2.), 'bidirectional_exchange_scale': math.log(.09)}
    system = model.build_system(physiology, total_clearance_l_h=8., renal_clearance_l_h=2., family='balanced_exchange', log_parameters=logs)
    times = np.array([0., .02, .2, .5, 1., 3., 8.])
    a = solve_ivp(lambda t, x: system.matrix @ x + 200 * system.input_vector,
        (0., .5), np.zeros(len(system.state_names)), t_eval=times[times <= .5], rtol=1e-10, atol=1e-11)
    b = solve_ivp(lambda t, x: system.matrix @ x, (.5, 8.), a.y[:, -1],
        t_eval=times[times > .5], rtol=1e-10, atol=1e-11)
    numerical = np.vstack([a.y.T, b.y.T]) @ system.observation_vector
    actual = model.simulate(physiology, times_h=times, dose_mg=100., infusion_duration_h=.5,
        total_clearance_l_h=8., renal_clearance_l_h=2., family='balanced_exchange', log_parameters=logs)
    np.testing.assert_allclose(actual['concentration_mg_l'], numerical, rtol=1e-7, atol=2e-9)
