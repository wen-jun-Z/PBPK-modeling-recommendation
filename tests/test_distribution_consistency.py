"""No retraining, score-based formula selection, or silent failure removal."""
from copy import deepcopy
from pathlib import Path
import math

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import distribution_consistency as d


def request():
    return {"compound_id": "synthetic", "human": {"weight_kg": 70},
            "dosing": {"route": "intravenous", "start_h": 0., "single_dose": True},
            "target": {"window_end_h": 24.}, "evidence": [], "forbidden_sources": []}


def failed():
    return {"status": "failed", "request": request(), "target_observations_used": False,
            "reason": "no_usable_sampled_NCA_model", "curve": None, "endpoints": None,
            "design_audit": {"version": d.LEARNED_VERSION, "target_basis": "sampled_nca",
                "ps_bounds": [.01, 100.], "grid_points": 17, "trajectory_budget": 33,
                "search_attempts": 17, "target_observations_used": False, "reasons": [d.FLOOR_CONFLICT],
                "linear_trajectory_calls": 0,
                "trials": [{"status": "failed", "trajectory_attempted": False,
                            "reason": d.FLOOR_CONFLICT, "log_ps": float(x), "ps_multiplier": math.exp(x)}
                           for x in np.linspace(math.log(.01), math.log(100.), 17)]}}


def successful():
    return {"status": "predicted", "request": request(), "target_observations_used": False,
            "curve": {"time_min": [0, 60, 120], "concentration": [0, 2, 1],
                      "time_unit": "min", "unit": "umol/l"},
            "endpoints": {"observations_used": False, "scalar": 42},
            "native_solver_calls": 1, "linear_trajectory_calls": 33}


def test_only_proved_capacity_failure_triggers_fallback():
    assert d.eligible_capacity_failure(failed())
    assert not d.eligible_capacity_failure(successful())
    x = failed(); x['reason'] = 'native_source_failed'
    assert not d.eligible_capacity_failure(x)
    x = failed(); x['design_audit']['trials'] = []
    assert not d.eligible_capacity_failure(x)


@pytest.mark.parametrize('mutation', ['observation', 'mixed_reason', 'attempted_curve', 'positive_calls'])
def test_incomplete_or_mixed_failure_proof_does_not_trigger(mutation):
    x = failed(); a = x['design_audit']
    if mutation == 'observation': a['target_observations_used'] = True
    elif mutation == 'mixed_reason': a['trials'].append({**a['trials'][0], 'reason': 'timeout'})
    elif mutation == 'attempted_curve': a['trials'][0]['trajectory_attempted'] = True
    else: a['linear_trajectory_calls'] = 1
    assert not d.eligible_capacity_failure(x)


@pytest.mark.parametrize('mutation', ['truncated', 'repeated_point', 'different_budget', 'mismatched_multiplier'])
def test_all_fixed_grid_points_must_be_proved_failed(mutation):
    x = failed(); a = x['design_audit']
    if mutation == 'truncated': a['trials'] = a['trials'][:-1]
    elif mutation == 'repeated_point': a['trials'][0] = deepcopy(a['trials'][1])
    elif mutation == 'different_budget': a['trajectory_budget'] = 100
    else: a['trials'][0]['ps_multiplier'] = 10.
    assert not d.eligible_capacity_failure(x)


def test_successful_prediction_is_preserved_without_native_or_training(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No new native solve or parameter inference for a valid baseline')
    monkeypatch.setattr(d.original, 'run_native_partition', forbidden)
    entry = successful(); before = deepcopy(entry)
    result = d.run_design(entry['request'], [], [], tmp_path/'out', tmp_path, source_entry=entry)
    assert entry == before
    assert result['curve'] == entry['curve'] and result['endpoints'] == entry['endpoints']
    assert result['native_solver_calls'] == result['linear_trajectory_calls'] == 0
    assert result['distribution_audit']['baseline_retained_exactly'] is True
    assert result['distribution_audit']['shared_predictor_retrained'] is False
    assert result['distribution_audit']['selection_uses_clinical_scores'] is False
    assert (tmp_path/'out/result.json').is_file()


def test_other_failures_are_retained_not_rerouted(tmp_path):
    entry = failed(); entry['reason'] = 'unsupported_ionization'
    result = d.run_design(entry['request'], [], [], tmp_path/'out', tmp_path, source_entry=entry)
    assert result['status'] == 'failed' and result['reason'] == 'unsupported_ionization'
    assert result['native_solver_calls'] == 0
    assert not result['distribution_audit']['fallback_attempted']


def test_measured_blood_partition_is_protected(tmp_path):
    entry = failed()
    entry['request']['evidence'] = [{"parameter": "blood_to_plasma_ratio", "basis": "measured",
                                    "value": 2., "source": {}, "context": {}}]
    result = d.run_design(entry['request'], [], [], tmp_path/'out', tmp_path, source_entry=entry)
    assert result['status'] == 'failed'
    assert result['distribution_audit']['reason'] == 'qualified_blood_measurement_blocks_formula_override'
    assert result['native_solver_calls'] == 0


def test_target_evidence_or_foreign_request_rejected_before_output(tmp_path):
    entry = successful(); req = deepcopy(entry['request']); req['compound_id'] = 'other'
    with pytest.raises(ValueError, match='exact request'):
        d.run_design(req, [], [], tmp_path/'out', tmp_path, source_entry=entry)
    assert not (tmp_path/'out').exists()
    req = request(); req['evidence'] = [{'parameter': 'fraction_unbound',
              'source': {}, 'context': {'study_role': 'target'}}]
    entry['request'] = req
    with pytest.raises(ValueError, match='Target outcome'):
        d.run_design(req, [], [], tmp_path/'out', tmp_path, source_entry=entry)


def test_alternative_must_be_a_real_schmitt_native_run_and_failure_stays(tmp_path, monkeypatch):
    entry = failed(); entry.update(native_source_ref={'path': 'original'},
        physiology_ref={'path': 'phys'}, execution_ref={'path': 'execution'})
    native = {'status': 'predicted', 'partition_method': 'PKSimStandard',
              'request_sha256': d.original._digest(entry['request']),
              'adapter_sha256': d.sha256(Path(d.original.__file__)),
              'execution_ref': entry['execution_ref']}
    monkeypatch.setattr(d.original, '_read_ref', lambda ref, root:
        native if ref['path'] == 'original' else {'blood_to_plasma_ratio': 100., 'rbc_kp_plasma': 220.})
    calls = []
    def solve(req, folder, root, method):
        calls.append((deepcopy(req), method))
        return {'status': 'failed', 'reason': 'native_readback_failed',
                'result_ref': {'path': 'alternative'}, 'execution_ref': {'path': 'alt_execution'}}
    monkeypatch.setattr(d.original, 'run_native_partition', solve)
    result = d.run_design(entry['request'], [], [], tmp_path/'out', tmp_path, source_entry=entry)
    assert calls == [(entry['request'], 'Schmitt')]
    assert result['status'] == 'failed' and result['reason'] == 'native_source_failed'
    assert result['native_solver_calls'] == 1 and result['linear_trajectory_calls'] == 0
    assert result['distribution_audit']['alternative_outcome'] == 'native_readback_failed'


def test_endpoint_scoring_source_cannot_be_reused(tmp_path):
    entry = successful(); entry['endpoints']['observations_used'] = True
    with pytest.raises(ValueError, match='observations'):
        d.run_design(entry['request'], [], [], tmp_path/'out', tmp_path, source_entry=entry)


def test_final_prediction_uses_reduced_trajectory_and_preserves_clearance(tmp_path, monkeypatch):
    entry = failed()
    entry['request']['evidence'] = [
        {'parameter': 'molecular_weight', 'value': 100., 'unit': 'g/mol', 'context': {}, 'source': {}},
        {'parameter': 'total_clearance', 'value': 5., 'unit': 'ml/min/kg', 'context': {'route': 'iv'}, 'source': {}},
        {'parameter': 'renal_clearance', 'value': 1., 'unit': 'ml/min/kg', 'context': {'route': 'iv'}, 'source': {}}]
    entry.update(native_source_ref={'path': 'original'}, physiology_ref={'path': 'phys'},
                 execution_ref={'path': 'execution'})
    req = deepcopy(entry['request'])
    source_native = {'status': 'predicted', 'partition_method': 'PKSimStandard',
        'request_sha256': d.original._digest(req), 'execution_ref': entry['execution_ref'],
        'adapter_sha256': d.sha256(Path(d.original.__file__))}
    monkeypatch.setattr(d.original, '_read_ref', lambda ref, root: source_native if ref['path'] == 'original'
        else {'blood_to_plasma_ratio': 100., 'rbc_kp_plasma': 220.})
    native_curve = {'time_min': [0., 60.], 'concentration': [999., 999.]}
    monkeypatch.setattr(d.original, 'run_native_partition', lambda *a: {
        'status': 'predicted', 'partition_method': 'Schmitt', 'request_sha256': d.original._digest(req),
        'result_ref': {'path': 'new_native'}, 'execution_ref': {'path': 'new_execution', 'sha256': 'a'*64},
        'curve': native_curve})
    monkeypatch.setattr(d.schmitt_physiology, 'export_physiology', lambda **kwargs: {
        'manifest': {'path': 'new_phys'}, 'blood_to_plasma_ratio': .7, 'rbc_kp_plasma': .3})
    received = {}
    def simulate(physiology, model_req, vss, half, **kwargs):
        received.update(request=deepcopy(model_req), options=kwargs)
        return {'status': 'predicted', 'learned_exchange_audit': {'linear_trajectory_calls': 17},
                'full_trajectory': {'time_h': [0., 1.], 'concentration_mg_l': [.8, .2]},
                'mass_balance': {'relative_error': 0.}, 'observer': 'venous_plasma'}
    monkeypatch.setattr(d, 'simulate_learned_design', simulate)
    monkeypatch.setattr(d.original, 'summarize_extended_prediction',
        lambda curve, request: {'observations_used': False, 'last': curve['concentration'][-1]})
    result = d.run_design(req, [], [], tmp_path/'out', tmp_path, source_entry=entry)
    assert req == entry['request']
    assert result['curve']['concentration'] == [8., 2.]
    assert result['curve']['concentration'] != native_curve['concentration']
    assert result['endpoints']['last'] == 2.
    assert [r['value'] for r in received['request']['evidence']] == [100., 5., 1.]
    assert received['options']['max_trajectory_calls'] == 33
    assert received['options']['times_h'].tolist() == [0., 1.]
    assert result['native_solver_calls'] == 1 and result['linear_trajectory_calls'] == 17
    assert result['distribution_audit']['alternative_outcome'] == 'constructed'


def test_completed_result_cannot_be_overwritten(tmp_path):
    entry = successful()
    d.run_design(entry['request'], [], [], tmp_path/'out', tmp_path, source_entry=entry)
    with pytest.raises(FileExistsError):
        d.run_design(entry['request'], [], [], tmp_path/'out', tmp_path, source_entry=entry)
