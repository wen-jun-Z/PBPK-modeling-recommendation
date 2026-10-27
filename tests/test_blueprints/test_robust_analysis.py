"""Source/replay tests for the separate robust-structure analysis artifact."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import analyze_robust_blueprint as analysis
from pbpk_kb.blueprint_policy import transport_calibration


@pytest.fixture(scope='module')
def original():
    return analysis.risk.collect_current_records(ROOT)


def test_replay_never_builds_exports_or_refits_and_reproduces_development_features(original, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Analysis may not build/export/refit')
    monkeypatch.setattr(transport_calibration, '_physiology', forbidden)
    monkeypatch.setattr(transport_calibration, 'fit_mapping', forbidden)
    store = analysis.Store(ROOT)
    rows = [r for r in original['records'] if r['sampling'] == 'full']
    for case in sorted({r['case_id'] for r in rows}):
        selected = [r for r in rows if r['case_id'] == case]
        features, audit = analysis.replay_development(store, selected[0]['development_sources']['mapping'])
        assert audit['native_calls'] == 0
        assert audit['linear_solver_calls'] == 2
        assert audit['curve_saved'] is False
        assert audit['rmse_absolute_difference'] < 1e-10
        assert audit['support']['clearance_bounds'] == [.25, 4.]
        for record in selected:
            for name, value in features[record['endpoint']].items():
                prior = record['features'][name]
                assert value is prior if value is None or prior is None else value == pytest.approx(prior, abs=1e-10)


def test_replay_checks_every_retained_artifact_hash_before_simulation(original, monkeypatch):
    source = original['records'][0]['development_sources']['mapping']
    store = analysis.Store(ROOT)
    real_ref = store.ref
    mapping = real_ref(source)
    key = next(iter(mapping['retained_artifact_pins']))
    mapping['retained_artifact_pins'][key] = '0' * 64
    monkeypatch.setattr(store, 'ref', lambda ref: mapping if ref == source else real_ref(ref))
    monkeypatch.setattr(transport_calibration, '_simulate', lambda *a, **k: pytest.fail('Corrupted source reached solver'))
    with pytest.raises(ValueError, match='Source lock mismatch'):
        analysis.replay_development(store, source)


def test_replay_requires_physiology_to_be_explicitly_retained(original, monkeypatch):
    source = original['records'][0]['development_sources']['mapping']
    store = analysis.Store(ROOT)
    real_ref = store.ref
    mapping = real_ref(source)
    name = str((Path(source['path']).parent / 'physiology/perfusion_physiology.json'))
    del mapping['retained_artifact_pins'][name]
    monkeypatch.setattr(store, 'ref', lambda ref: mapping if ref == source else real_ref(ref))
    with pytest.raises(ValueError, match='not retained'):
        analysis.replay_development(store, source)


def test_fixed_scenario_scores_reproduce_available_oof_and_ignore_scenario_labels(original):
    records = [r for r in original['records'] if r['sampling'] == 'full']
    evaluation = analysis.risk.evaluate_records(records)
    for record in evaluation['records']:
        result = analysis.fixed_fold_predictions(evaluation, record)
        assert record['cluster_id'] not in result['training_clusters']
        for name, expected in record['held_out_predictions'].items():
            assert result['probabilities'][name] == pytest.approx(expected, abs=1e-14)
        changed = deepcopy(record)
        changed['matched_fold_error'] = 10. if record['matched_fold_error'] <= 2 else 1.
        assert analysis.fixed_fold_predictions(evaluation, changed) == result


def test_scenario_diagnostic_preserves_missing_available_reference_without_aborting(original):
    records = [{**r, 'method': 'current', 'scenario': 'available'} for r in original['records'] if r['sampling'] == 'full']
    evaluation = analysis.risk.evaluate_records(records)
    missing = {**records[0], 'scenario': 'missing_fu'}
    unavailable = {k: v for k, v in records[0].items() if k != 'matched_fold_error'}
    unavailable['reason'] = 'missing_or_invalid_clinical_reference_or_matched_error'
    result = analysis.scenario_diagnostics(records[1:] + [missing], {'current': evaluation}, [unavailable])
    assert len(result) == 1
    assert result[0]['clinical_reference_pair_complete'] is False
    assert result[0]['available_reference_missing_reason'] == unavailable['reason']
    assert result[0]['missing_input_scores'] is not None
    assert result[0]['probability_change']['risk_model'] == pytest.approx(0.)
    result = analysis.scenario_diagnostics([missing], {'current': evaluation})
    assert result[0]['paired_feature_diagnostic_status'] == 'available_features_missing'
    assert result[0]['missing_input_scores'] is not None
    assert result[0]['probability_change'] is None


def test_scenario_diagnostic_keeps_missing_outer_fold_explicit(original):
    records = [{**r, 'method': 'current', 'scenario': 'missing_fu'} for r in original['records'][:1]]
    result = analysis.scenario_diagnostics(records, {'current': {'folds': []}})
    assert result[0]['missing_input_scores'] is None
    assert result[0]['probability_change'] is None


def test_target_gate_is_bound_to_receipt_and_state_without_observations():
    state = json.loads((ROOT / 'experiments/blueprint_design/blueprint_capability_20260930/state.json').read_text())
    comp = next(v for v in state['computations'].values() if v.get('status') == 'predicted')
    result = analysis.locked_target_gate(analysis.Store(ROOT), comp)
    assert set(result) == set(analysis.risk.ENDPOINT_FAMILY)
    assert all(type(v) is bool for v in result.values())
    broken = deepcopy(comp)
    broken['dense_endpoints']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Source lock mismatch'):
        analysis.locked_target_gate(analysis.Store(ROOT), broken)


def test_collector_replays_equal_input_aliases_once_and_keeps_reference_denominators():
    run = ROOT / 'experiments/blueprint_design/blueprint_capability_20260930'
    state = json.loads((run / 'state.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    result = json.loads((run / 'results.json').read_text())
    original = next(r for r in result['units'] if r['drug'] == 'Remdesivir' and
                    r['scenario'] == 'calibration' and r['method'] == 'always_three' and r['sampling'] == 'full')
    rows = [{**original, 'method': 'current', 'scenario': scenario, 'scientific_inputs_changed': False}
            for scenario in ('available', 'missing_fu')]
    mock_protocol = {'cases': protocol['cases'], 'units': rows}
    records, excluded, replay = analysis.collect_records(analysis.Store(ROOT), mock_protocol, state, {'units': rows})
    assert len(replay) == 1
    assert len(records) == 8  # Four existing references times two explicit scenario aliases.
    assert len(excluded) == 4  # MRT/Vss references are absent; they never become good labels.
    assert {r['scenario'] for r in records} == {'available', 'missing_fu'}
    assert all(r['reason'] == 'missing_or_invalid_clinical_reference_or_matched_error' for r in excluded)


def test_statistical_source_locks_fail_on_replaced_file(tmp_path):
    path = tmp_path / 'source.json'
    path.write_text('{}')
    store = analysis.Store(tmp_path)
    digest = analysis.sha256(path)
    analysis.verify_pins(store, {'source.json': digest})
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError):
        analysis.verify_pins(store, {'source.json': digest})


def test_analysis_cannot_start_before_complete_or_after_publication_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis.batch, 'own_run', lambda path: tmp_path)
    monkeypatch.setattr(analysis.batch, 'load', lambda *a, **k: ({'paper_guard': {}}, {'status': 'predictions_locked'}))
    with pytest.raises(ValueError, match='must be complete'):
        analysis.analyze(tmp_path)
    monkeypatch.setattr(analysis.batch, 'load', lambda *a, **k: ({'paper_guard': {}}, {'status': 'complete'}))
    monkeypatch.setattr(analysis.batch, 'paper_pins', lambda: {'new_paper_file': 'hash'})
    with pytest.raises(ValueError, match='Publication files changed'):
        analysis.analyze(tmp_path)


def test_source_refs_cannot_escape_project(tmp_path):
    store = analysis.Store(tmp_path)
    with pytest.raises(ValueError, match='outside the project'):
        store.checksum('../outside')


def test_rf_nested_split_and_weights_are_pinned_and_values_match_without_loading_weights(monkeypatch):
    import joblib
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('Provenance audit must not execute model pickle'))
    protocol = json.loads((ROOT / 'experiments/blueprint_design/vascular_tissue_ten_drug_20260930/protocol.json').read_text())
    store = analysis.Store(ROOT)
    audit = analysis.audit_rf_provenance(store, protocol['cases'])
    assert audit['evidence_values_verified'] == 20
    assert audit['unique_request_prediction_split_model_files'] == 22
    assert {r['training_rows'] for r in audit['records']} == {1097}
    assert all(r['identity_overlap_count'] == r['scaffold_overlap_count'] == 0 for r in audit['records'])
    assert any(name.endswith('parameter_models.joblib') for name in store.pins)
    assert any(name.endswith('parameter_split.json') for name in store.pins)
