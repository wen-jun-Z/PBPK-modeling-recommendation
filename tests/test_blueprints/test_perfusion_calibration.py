from copy import deepcopy
import math

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import perfusion_calibration as cal


def request():
    context = {'species': 'human', 'matrix': 'plasma', 'route': 'intravenous'}
    return {'compound_id': 'synthetic:A', 'compound_name': 'Synthetic A',
        'human': {'weight_kg': 80.}, 'dosing': {'route': 'intravenous', 'dose_mg': 20., 'infusion_minutes': 30.},
        'target': {'window_start_h': 0., 'window_end_h': 12.}, 'partition_method': 'PKSimStandard',
        'evidence_ledger': [], 'evidence': [
            {'parameter': 'total_clearance', 'value': 2., 'unit': 'ml/min/kg', 'context': context},
            {'parameter': 'renal_clearance', 'value': .5, 'unit': 'ml/min/kg', 'context': context},
            {'parameter': 'molecular_weight', 'value': 500., 'unit': 'g/mol'}]}


def test_clearance_uses_request_weight_and_preserves_renal_split():
    req = request()
    cl = cal.clearance_inputs(req)
    assert cl['total_clearance_l_h'] == pytest.approx(9.6)
    assert cl['renal_clearance_l_h'] == pytest.approx(2.4)
    req['human']['weight_kg'] = 40
    lighter = cal.clearance_inputs(req)
    assert lighter['total_clearance_l_h'] == pytest.approx(4.8)
    assert lighter['renal_clearance_l_h'] == pytest.approx(1.2)


@pytest.mark.parametrize('mutation', ['oral', 'unit', 'renal_exceeds_total', 'duplicate_hepatic'])
def test_clearance_invalid_evidence_rejected(mutation):
    req = request()
    if mutation == 'oral': req['evidence'][0]['context']['route'] = 'oral'
    elif mutation == 'unit': req['evidence'][0]['unit'] = 'L/h'
    elif mutation == 'renal_exceeds_total': req['evidence'][1]['value'] = 3
    else: req['evidence'].append({**req['evidence'][0], 'parameter': 'hepatic_clearance'})
    with pytest.raises(ValueError): cal.clearance_inputs(req)


def test_starts_do_not_receive_or_depend_on_curve():
    for variant, count in [('perfusion_global', 2), ('perfusion_grouped', 3)]:
        keys, lower, upper, starts = cal.parameter_starts(variant)
        assert starts.shape == (16, count)
        assert (starts[0] == 0).all()
        assert ((starts >= lower) & (starts <= upper)).all()
        np.testing.assert_array_equal(starts, cal.parameter_starts(variant)[3])


def test_prediction_grid_includes_dose_end_without_outcome_times():
    req = request(); req['dosing']['infusion_minutes'] = 3.17
    t = cal.prediction_grid(req)
    assert t[0] == 0 and t[-1] == 1344
    assert np.all(np.diff(t) > 0)
    assert req['dosing']['infusion_minutes']/60 in t
    assert req['target']['window_end_h'] in t


@pytest.fixture
def fitting(tmp_path, monkeypatch):
    req = request()
    values = [8., 7., 5., 2., .8]
    doc = {'compound_id': req['compound_id'], 'role': 'calibration_data_reconstruction',
        'matrix': 'plasma', 'unit': 'umol/l', 'time_unit': 'min', 'human': req['human'], 'dosing': req['dosing'],
        'observations': [{'time_min': t, 'concentration_umol_per_l': v}
                         for t, v in zip([15., 30., 60., 120., 240.], values)]}
    cal.save(tmp_path/'development.json', doc)
    source = {**cal._ref(tmp_path/'development.json', tmp_path), 'role': 'calibration_data_reconstruction'}
    (tmp_path/'policy.py').write_text('# immutable policy')
    monkeypatch.setattr(cal, '_code_pins', lambda root: {'policy.py': cal.sha256(root/'policy.py')})
    class Validated:
        def __init__(self, value): self.value = value
        def model_dump(self, **kwargs): return deepcopy(self.value)
    class Request:
        @staticmethod
        def model_validate(value): return Validated(value)
    monkeypatch.setattr(cal, 'EvidenceBlueprintRequest', Request)
    calls = []
    def physiology(req, output, root, check):
        if check: check()
        calls.append('source')
        return {'fixture': True}
    def simulate(physiology, req, times, variant, keys, x):
        calls.append('linear')
        t = np.array(times)
        return {'concentration_mg_l': np.exp(1.4+x[1] - (.5+math.exp(x[0]))*t)}
    monkeypatch.setattr(cal, '_physiology', physiology)
    monkeypatch.setattr(cal, '_simulate', simulate)
    def fit(**kwargs):
        return cal.fit_mapping(reference_request=req, training_source=source, output_dir=tmp_path/'fit',
            project_root=tmp_path, variant='perfusion_global', **kwargs)
    return tmp_path, req, source, calls, fit


def test_sealed_fit_selection_resume_and_cost(fitting):
    root, req, source, calls, fit = fitting
    result = fit()
    assert result['eligible'] and len(result['starts']) == 16
    assert result['linear_solver_calls'] == calls.count('linear')
    assert result['decision']['objective'] == min(x['objective'] for x in result['starts'])
    assert result['total_native_calls'] == 1 and not result['comparison_curve_used']
    before = len(calls)
    again = fit()
    assert again['actual_new_calls'] == 0 and len(calls) == before
    cleanup = cal.cleanup_unselected(output_dir=root/'fit', project_root=root)
    assert cleanup['removed_count'] == 0
    assert fit()['decision'] == result['decision']


def test_fit_refuses_target_observations(fitting):
    root, req, source, calls, fit = fitting
    req['forbidden_sources'] = [source]
    with pytest.raises(ValueError, match='Comparison observations'):
        fit()
    assert not calls


def test_completed_fit_refuses_source_or_code_changes(fitting):
    root, req, source, calls, fit = fitting
    fit()
    (root/'policy.py').write_text('# changed policy')
    with pytest.raises(ValueError, match='dependencies changed'):
        fit()


def test_subset_conditions_must_match_request(fitting):
    root, req, source, calls, fit = fitting
    req['human']['weight_kg'] = 60
    with pytest.raises(ValueError, match='Development conditions differ'):
        fit()
    assert not calls


def test_mapping_cannot_rewrite_both_start_scores_and_selected_decision(fitting):
    root, req, source, calls, fit = fitting
    fit()
    path = root/'fit/mapping_locked.json'
    mapping = cal._read(path)
    for row in mapping['starts']:
        row['objective'] *= .5
    mapping['decision']['objective'] *= .5
    cal.save(path, mapping)
    with pytest.raises(ValueError, match='development records'):
        fit()
