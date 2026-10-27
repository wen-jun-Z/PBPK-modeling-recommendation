"""Frozen-source reuse and no-training/no-target boundaries for the new mapping."""
from copy import deepcopy
from pathlib import Path

import pytest

from pbpk_kb.blueprint_policy import joint_partition_runtime as r


def inputs():
    req = {'compound_id': 'synthetic', 'human': {'weight_kg': 70},
        'dosing': {'route': 'intravenous', 'start_h': 0., 'single_dose': True},
        'target': {'window_end_h': 24.}, 'forbidden_sources': [],
        'evidence': [{'parameter': 'molecular_weight', 'value': 100., 'unit': 'g/mol', 'source': {}, 'context': {}},
            {'parameter': 'total_clearance', 'value': 5., 'unit': 'ml/min/kg', 'source': {}, 'context': {'route': 'iv'}},
            {'parameter': 'renal_clearance', 'value': 1., 'unit': 'ml/min/kg', 'source': {}, 'context': {'route': 'iv'}}]}
    execution = {'path': 'native_execution', 'sha256': 'a'*64}
    entry = {'request': req, 'status': 'predicted', 'target_observations_used': False,
        'observations_used': False, 'request_sha256': r.original._digest(req),
        'native_source_ref': {'path': 'native_source'}, 'execution_ref': execution,
        'physiology_ref': {'path': 'physiology'}}
    native = {'request_sha256': entry['request_sha256'], 'partition_method': 'PKSimStandard',
        'adapter_sha256': r.sha256(Path(r.original.__file__)), 'status': 'predicted',
        'observations_used': False, 'execution_ref': execution,
        'curve': {'time_min': [0., 30., 60.], 'concentration': [999., 999., 999.]}}
    phys = {'version': 'native-perfusion-physiology-1', 'observations_used': False,
        'source_execution_manifest_sha256': 'a'*64, 'source_pins': {}, 'artifact_pins': {}}
    return req, entry, native, phys


def bind(monkeypatch, native, phys):
    monkeypatch.setattr(r.original, '_read_ref', lambda ref, root: deepcopy(native if ref['path']=='native_source' else phys))


@pytest.mark.parametrize('change', ['request', 'source_observation', 'formula', 'adapter', 'derived_phys', 'phys_observation'])
def test_wrong_or_contaminated_construction_source_rejected(tmp_path, monkeypatch, change):
    req, entry, native, phys = inputs()
    if change == 'request': entry['request_sha256'] = 'b'*64
    elif change == 'source_observation': native['observations_used'] = True
    elif change == 'formula': native['partition_method'] = 'Schmitt'
    elif change == 'adapter': native['adapter_sha256'] = 'b'*64
    elif change == 'derived_phys': phys['mapping_version'] = 'some-derived-physiology'
    else: phys['observations_used'] = True
    bind(monkeypatch, native, phys)
    with pytest.raises(ValueError):
        r.run_design(req, [], [], tmp_path/'out', tmp_path, source_entry=entry)
    assert not (tmp_path/'out').exists()


def test_source_hashes_rechecked_before_build(tmp_path, monkeypatch):
    req, entry, native, phys = inputs()
    (tmp_path/'readback.json').write_text('{}')
    phys['artifact_pins'] = {'readback.json': 'b'*64}
    bind(monkeypatch, native, phys)
    with pytest.raises(ValueError, match='readback or source changed'):
        r.run_design(req, [], [], tmp_path/'out', tmp_path, source_entry=entry)


def test_reduced_prediction_with_frozen_clearance_no_native_solver(tmp_path, monkeypatch):
    req, entry, native, phys = inputs(); before = deepcopy(req)
    bind(monkeypatch, native, phys)
    def no_native(*args, **kwargs): raise AssertionError('This run may not rebuild native physiology')
    monkeypatch.setattr(r.original, 'run_native_partition', no_native)
    seen = {}
    def joint(physiology, request, vss, half, **kwargs):
        seen.update(request=deepcopy(request), kwargs=kwargs)
        return {'status': 'predicted', 'learned_exchange_audit': {'linear_trajectory_calls': 20},
            'joint_partition_audit': {'alpha': .2},
            'full_trajectory': {'time_h': [0., .5, 1.], 'concentration_mg_l': [0., .8, .2]},
            'mass_balance': {'maximum_relative_error': 1e-12}, 'observer': 'venous_plasma',
            'exact_moments': {'clearance_l_h': 21., 'vss_l': 40.}}
    monkeypatch.setattr(r.joint, 'simulate_joint_design', joint)
    monkeypatch.setattr(r.original, 'summarize_extended_prediction',
                        lambda curve, request: {'observations_used': False, 'last': curve['concentration'][-1]})
    out = r.run_design(req, [], [], tmp_path/'out', tmp_path, source_entry=entry)
    assert req == before
    assert [x['value'] for x in seen['request']['evidence']] == [100., 5., 1.]
    assert seen['kwargs']['times_h'].tolist() == [0., .5, 1.]
    assert seen['kwargs']['max_trajectory_calls'] == 33
    assert out['curve']['concentration'] == [0., 8., 2.]
    assert out['endpoints']['last'] == 2.
    assert out['new_training_calls'] == out['native_solver_calls'] == 0
    assert out['linear_trajectory_calls'] == 20
    assert out['exact_moments_used_to_fill_NCA'] is False


def test_impossible_parameter_combination_remains_a_failure(tmp_path, monkeypatch):
    req, entry, native, phys = inputs(); bind(monkeypatch, native, phys)
    monkeypatch.setattr(r.joint, 'simulate_joint_design', lambda *a, **k: {
        'status': 'failed', 'reason': 'no_positive_distribution_root',
        'learned_exchange_audit': {'linear_trajectory_calls': 0}, 'joint_partition_audit': {}})
    out = r.run_design(req, [], [], tmp_path/'out', tmp_path, source_entry=entry)
    assert out['status'] == 'failed' and out['curve'] is None and out['endpoints'] is None
    assert out['reason'] == 'no_positive_distribution_root'


def test_any_mutation_of_raw_physiology_is_a_hard_error(tmp_path, monkeypatch):
    req, entry, native, phys = inputs(); bind(monkeypatch, native, phys)
    def mutate(physiology, *args, **kwargs):
        physiology['observations_used'] = True
        return {'status': 'failed'}
    monkeypatch.setattr(r.joint, 'simulate_joint_design', mutate)
    with pytest.raises(ValueError, match='mutated a frozen input'):
        r.run_design(req, [], [], tmp_path/'out', tmp_path, source_entry=entry)
