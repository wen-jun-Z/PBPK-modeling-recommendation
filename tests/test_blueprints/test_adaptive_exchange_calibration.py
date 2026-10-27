"""New calibrated exchange contract; synthetic data and no native execution."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import adaptive_exchange_calibration as cal
from pbpk_kb.blueprint_policy import robust_exchange_model as frozen_model
from pbpk_kb.blueprint_policy import adaptive_exchange_model as model
from pbpk_kb.blueprint_policy.tail_exchange_calibration import tail_weights

spec = importlib.util.spec_from_file_location('_adaptive_fit_contracts', Path(__file__).with_name('test_perfusion_calibration.py'))
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)
spec = importlib.util.spec_from_file_location('_adaptive_flow_fixture', Path(__file__).with_name('test_permeability_model.py'))
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)


def qualify(req, basis='predicted'):
    row = req['evidence'][0]
    row.update(compound_id=req['compound_id'], basis=basis,
        source={'path': 'prior.json', 'sha256': 'a'*64, 'location': 'Independent synthetic construction prior'})
    if basis == 'predicted':
        row['prediction_method'] = {'name': 'RF', 'version': 'frozen-test-1', 'training_overlap': False,
            'applicability': {'human_IV_parameter_model': True, 'target_structure_and_scaffold_excluded': True}}
    return req


@pytest.mark.parametrize('basis,bounds,penalty', [('derived', [.25, 4.], .1),
    ('measured', [.25, 4.], .1), ('predicted', [.05, 20.], .01)])
def test_identical_three_axis_starts_and_signed_alpha_have_evidence_bounds(basis, bounds, penalty):
    req = qualify(contracts.request(), basis)
    assert cal.clearance_policy(req)['penalty_weight'] == penalty
    first = cal.parameter_starts('relaxed_tail', req)
    second = cal.parameter_starts('adaptive_exchange', req)
    np.testing.assert_array_equal(first[3], second[3][:, :3])
    np.testing.assert_array_equal(second[3], cal.parameter_starts('adaptive_exchange', req)[3])
    np.testing.assert_allclose(np.exp(first[1]), [bounds[0], .02, .01])
    np.testing.assert_allclose(np.exp(first[2]), [bounds[1], 10., 100.])
    assert second[1][-1] == -.5 and second[2][-1] == .5
    assert first[3].shape == (16, 3) and second[3].shape == (16, 4)
    np.testing.assert_array_equal(second[3][0], np.zeros(4))
    assert ((second[3] >= second[1]) & (second[3] <= second[2])).all()
    altered = deepcopy(req); altered['target']['unread_outcome'] = [1000, 2000]
    np.testing.assert_array_equal(cal.parameter_starts('adaptive_exchange', altered)[3], second[3])


def test_alpha_penalty_is_signed_symmetric_and_not_exponentiated():
    prior = {'penalty_weight': .01}
    for alpha in [-.5, 0., .5]:
        x = np.array([0., 0., 0., alpha])
        residual = cal.parameter_penalty(x, prior)
        assert len(residual) == 4
        assert residual[-1] == pytest.approx(math.sqrt(.1)*alpha/.5)
        assert float(residual@residual) == pytest.approx(.1*(alpha/.5)**2)
    assert len(cal.parameter_penalty(np.zeros(3), prior)) == 3
    with pytest.raises(ValueError):
        cal.parameter_penalty(np.zeros(5), prior)


@pytest.mark.parametrize('variant', ['relaxed_tail', 'adaptive_exchange'])
def test_real_simulation_preserves_renal_for_all_calibration_axes(variant):
    physiology = flow.physiology.__wrapped__()
    request = qualify(contracts.request())
    keys = cal.PARAMETERS[variant]
    x = [math.log(.05), math.log(.02), math.log(.1)] + ([.5] if len(keys) == 4 else [])
    prediction = cal._simulate(physiology, request, [0., .1, .5, 2., 12.], variant, keys, x)
    assert prediction['effective_total_clearance_l_h'] == pytest.approx(2.4 + (9.6-2.4)*.05)
    physical = model if variant == 'adaptive_exchange' else frozen_model
    system = physical.build_system(physiology, family='adaptive_exchange' if len(keys) == 4 else 'evidence_flexible',
        total_clearance_l_h=9.6, renal_clearance_l_h=2.4, log_parameters=dict(zip(keys, x)))
    assert system.metadata['clearance_components_l_h']['qualified_renal_fixed'] == 2.4
    np.testing.assert_allclose(system.matrix.sum(axis=0), 0., atol=1e-10)


@pytest.fixture(params=['relaxed_tail', 'adaptive_exchange'])
def fitting(request, tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, 'cal', cal)
    actual = cal.fit_mapping

    def fit(**kwargs):
        kwargs['variant'] = request.param
        return actual(**kwargs)

    monkeypatch.setattr(cal, 'fit_mapping', fit)
    fixture = contracts.fitting.__wrapped__(tmp_path, monkeypatch)
    root, req, source, calls, execute = fixture
    qualify(req)
    cal.save(root/'prior.json', {'kind': 'synthetic independent prior'})
    req['evidence'][0]['source']['sha256'] = cal.sha256(root/'prior.json')
    return fixture


@pytest.mark.parametrize('contract', ['test_sealed_fit_selection_resume_and_cost',
    'test_fit_refuses_target_observations', 'test_completed_fit_refuses_source_or_code_changes',
    'test_subset_conditions_must_match_request', 'test_mapping_cannot_rewrite_both_start_scores_and_selected_decision'])
def test_frozen_source_target_isolation_and_resume_contracts(fitting, contract):
    getattr(contracts, contract)(fitting)


def test_weighted_objective_and_unweighted_metrics_are_separately_correct(fitting):
    root, request, source, calls, execute = fitting
    result = execute(); chosen = result['decision']; x = np.asarray(chosen['log_parameters'])
    observed = cal._read(root/source['path'])
    plan = tail_weights([r['time_min']/60 for r in observed['observations']], .5)
    errors = np.asarray(chosen['development_log_errors'])
    weighted = float(np.asarray(plan['weights'])@(errors**2))
    penalty = cal.parameter_penalty(x, result['clearance_prior_policy'])
    assert chosen['objective'] == pytest.approx(weighted + float(penalty@penalty))
    assert chosen['weighted_data_loss'] == pytest.approx(weighted)
    assert chosen['data_only_ln_rmse'] == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert chosen['curve_metrics']['gmfe'] == pytest.approx(np.exp(np.mean(abs(errors))))
    assert chosen['data_jacobian_includes_prior_rows'] is False
    assert chosen['data_jacobian_rows'] == len(errors) and chosen['data_jacobian_columns'] == len(x)
    assert chosen['data_jacobian_rank'] <= 2  # Stub predictor has only two physical input axes.
    assert result['formula_selection_score'] == chosen['objective']
    assert result['exchange_heterogeneity_alpha'] == (x[3] if len(x) == 4 else 0.)
    if len(x) == 4:
        assert chosen['parameters']['exchange_heterogeneity_scale'] == math.exp(x[3])


def test_data_jacobian_diagnostics_exclude_prior_identifiability():
    keys = cal.PARAMETERS['adaptive_exchange']
    data = np.column_stack([np.ones(5), np.arange(5), np.zeros(5), np.zeros(5)])
    lower = np.array([-2., -4., -5., -.5]); upper = np.array([2., 3., 5., .5])
    x = np.array([0., -4., 0., .5])
    result = cal._diagnostics(data, x, lower, upper, keys)
    assert result['data_jacobian_rank'] == 2
    assert result['data_jacobian_includes_prior_rows'] is False
    assert result['bound_hits']['partition_scale'] == 'lower'
    assert result['bound_hits']['exchange_heterogeneity_scale'] == 'upper'


def test_failed_simulations_keep_correct_residual_dimension_and_costs(fitting, monkeypatch):
    root, req, source, calls, execute = fitting
    def invalid(*args, **kwargs):
        calls.append('failed_linear')
        raise ValueError('synthetic impossible trial')
    monkeypatch.setattr(cal, '_simulate', invalid)
    result = execute()
    assert result['status'] == 'failed' and result['eligible'] is False
    assert result['decision'] is None and result['formula_selection_score'] is None
    assert len(result['starts']) == 16
    assert result['failed_evaluation_count'] == result['linear_solver_calls'] == calls.count('failed_linear')
    assert all(r['status'] == 'scientific_failed' and r['data_only_ln_rmse'] is None for r in result['starts'])
    assert all(r['objective'] == (5 + len(result['active_parameters']))*1e12 for r in result['starts'])
    before = len(calls)
    assert execute()['actual_new_calls'] == 0 and len(calls) == before


def test_resume_rejects_changed_loss_definition_before_new_simulation(fitting, monkeypatch):
    root, req, source, calls, execute = fitting
    execute(); before = len(calls)
    changed = deepcopy(cal.CONFIG); changed['alpha_penalty_weight'] = .2
    monkeypatch.setattr(cal, 'CONFIG', changed)
    with pytest.raises(ValueError, match='dependencies changed'):
        execute()
    assert len(calls) == before


def test_target_prediction_is_locked_and_reused_without_reading_target_outcomes(fitting, monkeypatch):
    root, req, source, calls, execute = fitting
    cal.save(root/'target_secret.json', {'clinical_concentration': [90123, 80912]})
    req['forbidden_sources'] = [cal._ref(root/'target_secret.json', root)]
    result = execute()
    mapping = root/result['mapping']['path']; mapping_hash = cal.sha256(mapping)
    original_read = cal._read

    def safe_read(path):
        assert Path(path) != root/'target_secret.json'
        return original_read(path)

    monkeypatch.setattr(cal, '_read', safe_read)
    stub_simulation = cal._simulate

    def simulate(*args, **kwargs):
        prediction = stub_simulation(*args, **kwargs)
        return {**prediction, 'mass_balance': {'status': 'synthetic'},
            'observation': 'peripheral_venous', 'effective_total_clearance_l_h': 4.8}

    monkeypatch.setattr(cal, '_simulate', simulate)
    # Endpoint arithmetic has independent real tests; here exercise sealing,
    # source isolation, output reuse and immutable calibration parameters.
    import pbpk_kb.blueprints.extended_prediction as endpoints
    monkeypatch.setattr(endpoints, 'summarize_extended_prediction',
        lambda curve, request: {'endpoints': {'auc': {'value': 1.0}}})
    target = deepcopy(req); target['dosing']['dose_mg'] *= 2
    folder = root/'target'
    oral = deepcopy(target); oral['dosing']['route'] = 'oral'
    before = len(calls)
    with pytest.raises(ValueError, match='intravenous'):
        cal.predict_mapping(request=oral, mapping_path=mapping,
            output_dir=root/'oral', project_root=root)
    assert len(calls) == before and not (root/'oral').exists()
    prediction = cal.predict_mapping(request=target, mapping_path=mapping,
        output_dir=folder, project_root=root)
    assert prediction['status'] == 'predicted'
    assert cal.sha256(mapping) == mapping_hash
    receipt = cal._read(folder/'completion.json')
    assert receipt['construction']['target_outcomes_used'] is False
    assert receipt['construction']['exchange_heterogeneity_alpha'] == result['exchange_heterogeneity_alpha']
    before = len(calls)
    reused = cal.predict_mapping(request=target, mapping_path=mapping, output_dir=folder, project_root=root)
    assert reused['actual_new_calls'] == 0 and len(calls) == before
    changed = deepcopy(target); changed['dosing']['dose_mg'] *= 2
    with pytest.raises(ValueError, match='Prediction inputs changed'):
        cal.predict_mapping(request=changed, mapping_path=mapping, output_dir=folder, project_root=root)
    assert len(calls) == before
    (folder/'prediction_locked.json').write_text('{}')
    with pytest.raises(ValueError):
        cal.predict_mapping(request=target, mapping_path=mapping, output_dir=folder, project_root=root)
