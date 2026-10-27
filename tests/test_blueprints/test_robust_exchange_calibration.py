"""Prior provenance, target isolation and resumable robust calibration."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import robust_exchange_calibration as cal

spec = importlib.util.spec_from_file_location('_robust_fit_contracts', Path(__file__).with_name('test_perfusion_calibration.py'))
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)


def qualify(req, basis='derived'):
    row = req['evidence'][0]
    row.update(compound_id=req['compound_id'], basis=basis,
        source={'path': 'prior.json', 'sha256': 'a' * 64, 'location': 'Synthetic independent construction prior'})
    if basis == 'predicted':
        row['prediction_method'] = {'name': 'RF', 'version': 'frozen-test-1', 'training_overlap': False,
            'applicability': {'human_IV_parameter_model': True, 'target_structure_and_scaffold_excluded': True}}
    return req


@pytest.mark.parametrize('basis,bounds,penalty', [('derived', [.25, 4.], .1), ('measured', [.25, 4.], .1), ('predicted', [.05, 20.], .01)])
def test_prior_strength_is_selected_only_from_qualified_evidence(basis, bounds, penalty):
    req = qualify(contracts.request(), basis)
    policy = cal.clearance_policy(req)
    assert policy['scale_bounds'] == bounds
    assert policy['penalty_weight'] == penalty
    req['target']['unread_outcome'] = 'not an input to this policy'
    assert cal.clearance_policy(req) == policy
    for variant in cal.PARAMETERS:
        keys, lo, hi, starts = cal.parameter_starts(variant, req)
        np.testing.assert_allclose(np.exp(lo), [bounds[0], .1, .01])
        np.testing.assert_allclose(np.exp(hi), [bounds[1], 10., 100.])
        assert starts.shape == (16, 3)
        np.testing.assert_array_equal(starts[0], np.zeros(3))


@pytest.mark.parametrize('mutation', ['target', 'nonhuman', 'unqualified_rf', 'assumption', 'wrong_unit', 'duplicate'])
def test_unqualified_clearance_cannot_get_relaxed_bounds(mutation):
    req = qualify(contracts.request(), 'predicted')
    row = req['evidence'][0]
    if mutation == 'target': row['context']['study_role'] = 'target'
    elif mutation == 'nonhuman': row['context']['species'] = 'rat'
    elif mutation == 'unqualified_rf': row['prediction_method']['training_overlap'] = True
    elif mutation == 'assumption': row['basis'] = 'assumption'
    elif mutation == 'wrong_unit': row['unit'] = 'L/h'
    else: req['evidence'].append(deepcopy(row))
    with pytest.raises(ValueError): cal.clearance_policy(req)


@pytest.fixture(params=['evidence_flexible', 'balanced_exchange'])
def fitting(request, tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, 'cal', cal)
    original = cal.fit_mapping
    def fit(**kwargs):
        kwargs['variant'] = request.param
        return original(**kwargs)
    monkeypatch.setattr(cal, 'fit_mapping', fit)
    result = contracts.fitting.__wrapped__(tmp_path, monkeypatch)
    root, req, source, calls, execute = result
    qualify(req, 'predicted')
    cal.save(root / 'prior.json', {'kind': 'synthetic independent prior'})
    req['evidence'][0]['source']['sha256'] = cal.sha256(root / 'prior.json')
    return result


@pytest.mark.parametrize('contract', ['test_sealed_fit_selection_resume_and_cost',
    'test_fit_refuses_target_observations', 'test_completed_fit_refuses_source_or_code_changes',
    'test_subset_conditions_must_match_request', 'test_mapping_cannot_rewrite_both_start_scores_and_selected_decision'])
def test_robust_fit_isolation(fitting, contract):
    getattr(contracts, contract)(fitting)


def test_locked_objective_records_the_evidence_prior_and_three_axes(fitting):
    result = fitting[-1]()
    chosen = result['decision']; x = np.array(chosen['log_parameters'])
    weight = result['clearance_prior_policy']['penalty_weight']
    expected = chosen['data_only_ln_rmse'] ** 2 + weight * (x[0] / np.log(2)) ** 2 + .01 * (x[1] / np.log(10)) ** 2 + .01 * (x[2] / np.log(100)) ** 2
    assert chosen['objective'] == pytest.approx(expected)
    assert weight == .01
    np.testing.assert_allclose(np.exp(result['effective_log_bounds']['lower']), [.05, .1, .01])
    assert result['active_parameters'] == ['clearance_scale', 'partition_scale', 'bidirectional_exchange_scale']
