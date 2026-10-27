"""Isolation and source-integrity checks for feature-only reconstruction."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import blueprint_sampling_stability_features as module


def ref(path):
    return {'path': path, 'sha256': 'a' * 64}


def test_private_reader_overrides_only_exact_decision_without_global_mutation(monkeypatch):
    decision, ordinary = ref('decision.json'), ref('fit.json')
    reads = []
    def reader(source):
        reads.append(source)
        return {'real': source['path']}
    namespace = {'checked': reader}
    exec('def builder(p, key, entry):\n return checked(entry["result"]), checked(p["other"])', namespace)
    monkeypatch.setattr(module.expansion, 'checked', reader)
    monkeypatch.setattr(module.expansion, 'make_features', namespace['builder'])
    rewritten = {'new_model_values': [2., 3.]}
    pins = {}
    builder = module._feature_builder(rewritten, decision, pins)
    output, native = builder({'other': ordinary}, 'unused', {'result': decision})
    assert output == rewritten and output is not rewritten
    assert native == {'real': 'fit.json'} and reads == [ordinary]
    assert namespace['checked'] is reader and module.expansion.checked is reader
    assert set(pins) == {'decision.json', 'fit.json'}
    altered = dict(decision, sha256='b' * 64)
    with pytest.raises(ValueError, match='Conflicting source hashes'):
        builder({'other': ordinary}, 'unused', {'result': altered})


def test_refresh_uses_model_only_values_keeps_committee_and_marks_missing_goal(monkeypatch):
    decision, visible, contract, target, phys = map(ref, ['decision', 'visible', 'contract', 'target', 'phys'])
    case = {'case_id': 'test', 'target_contract': contract,
        'requests': {'evidence': {'target': target}}}
    old = dict(design_key='test__late_four', hidden_values_used=False, target_values_used=False,
        visible_source=visible, candidate_ids=[1, 2], candidate_times_h=[.5, 1.],
        members=[dict(formula='F', log_parameters=[i, 0, 0, 0]) for i in [0, 1]],
        concentrations_mg_l=[[1., 2.], [3., 4.]],
        choices={m: {g: 0 for g in module.GOALS} for m in
            ['time_farthest', 'maximum_variance', 'request_directed', 'linear_covariance']}, output_pins={})
    old_copy = deepcopy(old)
    menu = dict(candidate_ids=[1, 2], candidate_times_h=[.5, 1.], observed_times_h=[2., 3., 4., 5.])
    docs = {'decision': old, 'visible': {'design_only': menu},
        'contract': dict(source_kind='target_protocol_metadata_only', target_observation_values_used=False)}
    reads, simulations, design_calls = [], [], []
    def reader(source):
        reads.append(source['path'])
        return docs[source['path']]
    def simulate(variant, physiology, req, grid, parameters):
        simulations.append(parameters)
        return {'model_value': parameters[0] + 1.}
    def choose(times, visible_times, concentrations, goals, **kwargs):
        design_calls.append((concentrations.copy(), goals, kwargs))
        goal = next(iter(goals))
        return {'policies': {m: {goal: {'selected_index': 1}} for m in
            ['nonlinear_goal_directed', 'goal_directed']}}
    monkeypatch.setattr(module.expansion, 'checked', reader)
    monkeypatch.setattr(module.expansion, 'verify', lambda pins: None)
    monkeypatch.setattr(module.expansion.gap, 'get_case', lambda p, cid: case)
    monkeypatch.setattr(module.expansion.gap, 'request', lambda *args: {'metadata': True})
    monkeypatch.setattr(module.expansion.prior, 'get_phys', lambda *args: {'physiology': True})
    monkeypatch.setattr(module.expansion.base.engine, 'prediction_grid', lambda req: [0., 1.])
    monkeypatch.setattr(module.expansion.old, 'simulate', simulate)
    monkeypatch.setattr(module.expansion.prior, 'curve_document', lambda pred, grid, req: pred)
    monkeypatch.setattr(module.expansion.data, 'model_goals', lambda c, pred, req:
        dict(values={'cmax': pred['model_value'], 'auc_inf': None, 'terminal_half_life': pred['model_value']},
             target_observation_values_used=False))
    monkeypatch.setattr(module.expansion.gap.design, 'choose', choose)
    p = {'physiology_cache': {'test__target__F': phys}}
    new, pins = module._refresh_decision(p, 'test__late_four', {'result': decision},
        {'visible_source': visible, 'case_id': 'test'})
    assert old == old_copy
    assert reads == ['decision', 'visible', 'contract']
    assert simulations == [m['log_parameters'] for m in old['members']]
    assert new['concentrations_mg_l'] == old['concentrations_mg_l']
    assert new['goal_values']['cmax'] == [1., 2.]
    assert new['details']['auc_inf']['status'] == 'unsupported'
    assert new['choices']['request_directed']['auc_inf'] is None
    assert len(design_calls) == 2
    assert all(call[2] == {'log_noise_sd': .1, 'quadrature_nodes': 31} for call in design_calls)


def test_rebuild_rejects_incomplete_original_study():
    with pytest.raises(ValueError, match='must be complete'):
        module.rebuild_original_features({}, {'status': 'running', 'designs': {}})
