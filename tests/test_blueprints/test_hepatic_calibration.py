"""Reuse isolation contracts for the new three-parameter transport family."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import hepatic_calibration as cal

spec = importlib.util.spec_from_file_location('_perfusion_contracts',
    Path(__file__).with_name('test_perfusion_calibration.py'))
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)


@pytest.fixture
def fitting(tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, 'cal', cal)
    original = cal.fit_mapping
    def fit(**kwargs):
        kwargs['variant'] = 'hepatic_exchange'
        return original(**kwargs)
    monkeypatch.setattr(cal, 'fit_mapping', fit)
    return contracts.fitting.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize('contract', [
    'test_sealed_fit_selection_resume_and_cost',
    'test_fit_refuses_target_observations',
    'test_completed_fit_refuses_source_or_code_changes',
    'test_subset_conditions_must_match_request',
    'test_mapping_cannot_rewrite_both_start_scores_and_selected_decision'])
def test_hepatic_isolation_contract(fitting, contract):
    getattr(contracts, contract)(fitting)


def test_hepatic_starts_use_three_declared_axes():
    keys, lower, upper, starts = cal.parameter_starts('hepatic_exchange')
    assert keys == ('clearance_scale', 'partition_scale', 'bidirectional_exchange_scale')
    np.testing.assert_allclose(np.exp(lower), [.25, .1, .01])
    np.testing.assert_allclose(np.exp(upper), [4., 10., 100.])
    assert starts.shape == (16, 3)
    np.testing.assert_array_equal(starts[0], np.zeros(3))
    np.testing.assert_array_equal(starts, cal.parameter_starts('hepatic_exchange')[3])


def test_penalty_and_recorded_curve_loss_match_selected_objective(fitting):
    result = fitting[-1]()
    chosen = result['decision']
    x = np.array(chosen['log_parameters'])
    expected = (chosen['data_only_ln_rmse']**2 + .1*(x[0]/np.log(2))**2
                + .01*(x[1]/np.log(10))**2 + .01*(x[2]/np.log(100))**2)
    assert chosen['objective'] == pytest.approx(expected)
    assert result['active_parameters'] == list(cal.PARAMETERS['hepatic_exchange'])


def test_hepatic_routing_is_identity_based_not_curve_based():
    remde = {'compound_id': 'pubchem:121304016', 'compound_name': 'Remdesivir'}
    assert cal.route_policy(remde) == 'hepatic_equilibrium'
    assert cal.route_policy({**remde, 'unrelated_target_score': 999}) == 'hepatic_equilibrium'
    assert cal.route_policy({'compound_id': 'other', 'compound_name': 'Other'}) == 'frozen_flow_exchange'
    for bad in ({**remde, 'compound_id': 'other'}, {**remde, 'compound_name': 'Other'}):
        with pytest.raises(ValueError, match='identity'):
            cal.route_policy(bad)


def test_unchanged_drugs_reuse_frozen_equations(monkeypatch):
    from pbpk_kb.blueprint_policy import permeability_model
    captured = []
    def simulate(physiology, **kwargs):
        captured.append((physiology, kwargs))
        return {'sentinel': 1}
    monkeypatch.setattr(permeability_model, 'simulate', simulate)
    req = contracts.request()
    keys = cal.PARAMETERS['hepatic_exchange']
    assert cal._simulate({'native': True}, req, [0., 1.], 'hepatic_exchange', keys, np.zeros(3)) == {'sentinel': 1}
    args = captured[0][1]
    assert args['renal_clearance_l_h'] == pytest.approx(2.4)
    assert args['total_clearance_l_h'] == pytest.approx(9.6)
    assert args['family'] == 'exchange'
