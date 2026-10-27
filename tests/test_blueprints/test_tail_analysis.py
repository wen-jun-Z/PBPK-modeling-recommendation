"""Fixed risk protocol, isolated routing and actual stage-cost accounting."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import analyze_tail_blueprint as analysis
import analyze_robust_blueprint as frozen_analysis
from pbpk_kb.blueprint_policy import predictive_risk as frozen_risk


@pytest.fixture(scope='module')
def historical():
    return json.loads((ROOT / 'experiments/blueprint_design/robust_structure_20260930/analysis.json').read_text())


def test_fixed_model_outputs_match_frozen_analysis_and_do_not_run_nested_selection(historical):
    original_nested = frozen_risk.evaluate_nested_rmse_policy
    original_import = frozen_analysis.importlib
    collector, model = analysis.analysis_helpers()
    record = historical['primary_risk']['evidence_flexible']
    result = analysis.fixed_risk_evaluation(model, record['records'])
    assert result['metrics'] == record['metrics']
    assert result['by_sampling'] == record['by_sampling']
    assert result['nested_threshold_selection_performed'] is False
    assert 'nested_rmse_policy' not in result
    assert frozen_risk.evaluate_nested_rmse_policy is original_nested
    assert frozen_analysis.importlib is original_import
    assert model is not frozen_risk and collector is not frozen_analysis


def test_private_collector_routes_new_variants_without_changing_frozen_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(analysis.tail_engine, '_simulate', lambda *a: calls.append(('tail', a[3])) or {})
    monkeypatch.setattr(analysis.baseline_engine, '_simulate', lambda *a: calls.append(('baseline', a[3])) or {})
    collector, _ = analysis.analysis_helpers()
    engine = collector.importlib.import_module('pbpk_kb.blueprint_policy.robust_exchange_calibration')
    for variant in ('tail_balanced', 'development_guarded', 'evidence_flexible'):
        engine._simulate({}, {}, [], variant, [], [])
    assert calls == [('tail', 'tail_balanced'), ('tail', 'development_guarded'), ('baseline', 'evidence_flexible')]
    with pytest.raises(ValueError, match='Unsupported'):
        engine.engine('unrecognized_variant')


def test_current_replay_features_still_match_source_available_records(historical):
    collector, _ = analysis.analysis_helpers()
    rows = historical['primary_risk']['evidence_flexible']['records']
    chosen = rows[0]
    audit = historical['replay_audit'][chosen['computation_id']]
    features, replay = collector.replay_development(analysis.Store(ROOT), audit['mapping_source'])
    assert replay['rmse_absolute_difference'] < 1e-10
    assert replay['native_calls'] == 0 and replay['curve_saved'] is False
    assert features[chosen['endpoint']] == chosen['features']


def test_scenario_diagnostics_include_only_fixed_threshold_scores(historical):
    collector, model = analysis.analysis_helpers()
    records = [{**r, 'method': 'current'} for r in historical['primary_risk']['evidence_flexible']['records']]
    evaluation = analysis.fixed_risk_evaluation(model, records)
    changed = {**records[0], 'scenario': 'missing_fu'}
    result = analysis.scenario_diagnostics(collector, records + [changed], [], {'current': evaluation})
    assert len(result) == 1
    for field in ('available_scores', 'missing_input_scores'):
        assert 'nested_warning' not in result[0][field]
        assert set(result[0][field]['probabilities']) == set(evaluation['metrics'])
    assert result[0]['probability_change']['risk_model'] == pytest.approx(0.)


def test_actual_stage_costs_are_counted_once_and_cv_failure_does_not_invent_full_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis.tail_engine, '_validated', lambda *a: None)
    store = analysis.Store(tmp_path)
    formulas = ['PKSimStandard', 'RodgersRowland']; fits = {}
    for formula in formulas:
        stages = {'cv_0.01_0': {'status': 'complete', 'starts': [{}] * 8, 'heldout_score': None}}
        eligible = formula == 'PKSimStandard'
        if eligible:
            stages['cv_0.01_0']['heldout_score'] = .1
            stages['full'] = {'status': 'complete', 'starts': [{}] * 16}
        doc = {'eligible': eligible, 'stages': stages, 'total_native_calls': 1, 'linear_solver_calls': 100,
               'source_pins': {}, 'code_pins': {}, 'retained_artifact_pins': {}, 'cost_count_exact': True}
        path = tmp_path / (formula + '.json'); path.write_text(json.dumps(doc))
        fits['x__' + formula] = {'mapping': {'path': path.name, 'sha256': analysis.sha256(path)}}
    resources = {'total_native_calls': 3, 'linear_solver_calls': 201}
    protocol = {'computations': {'x': {}}, 'cached_ids': ['old'], 'formulas': formulas,
                'optimization_budget': {'full_fit_stages': 2, 'cv_stages': 2}}
    state = {'fits': fits, 'computations': {'x': {'resources': resources}}}
    result = analysis.actual_costs(store, protocol, state)
    actual = result['actual']
    assert actual['full_formula_slots_locked'] == 2
    assert actual['full_stages_completed'] == 1 and actual['full_stages_skipped_after_cv_failure'] == 1
    assert actual['cv_stages_completed'] == 2 and actual['cv_stages_without_score'] == 1
    assert actual['full_optimizer_starts_recorded'] == actual['cv_optimizer_starts_recorded'] == 16
    assert actual['new_native_calls'] == 3 and actual['new_target_native_calls'] == 1
    assert actual['new_fit_linear_solver_calls'] == 200 and actual['new_target_linear_solver_calls'] == 1
    assert result['reused_predictions'] == 1
    broken = deepcopy(state); broken['computations']['x']['resources']['total_native_calls'] = 1
    with pytest.raises(ValueError, match='undercounts'):
        analysis.actual_costs(store, protocol, broken)


def test_analysis_refuses_unfinished_batch_before_opening_scoring_sources(tmp_path, monkeypatch):
    runner = SimpleNamespace(own_run=lambda path: path,
        load=lambda *a, **k: ({}, {'status': 'predictions_locked'}))
    monkeypatch.setattr(analysis, 'create_runner', lambda: runner)
    with pytest.raises(ValueError, match='must be complete'):
        analysis.analyze(tmp_path)


def test_historical_analysis_is_a_live_ref_and_no_runtime_cache_is_required(historical):
    path = ROOT / 'experiments/blueprint_design/robust_structure_20260930/analysis.json'
    assert analysis.sha256(path) == analysis.PREVIOUS_ANALYSIS_SHA256
    assert not (ROOT / 'runtime/cache/robust_structure/risk_baseline.json').exists()
    assert historical['historical_risk']['result']['nested_rmse_policy']['metrics']['coverage'] == 0
