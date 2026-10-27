"""Isolation, baseline fairness, checkpoint and report barriers without native runs."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
runner = importlib.import_module('run_blueprint_architecture')


@pytest.fixture
def batch(tmp_path, monkeypatch):
    root = tmp_path
    group = root / 'experiments/blueprint_design/architecture_20260930'
    base = root / 'experiments/blueprint_design/evidence_policy_20260929'
    base.mkdir(parents=True)
    monkeypatch.setattr(runner, 'ROOT', root)
    monkeypatch.setattr(runner, 'GROUP', group)
    monkeypatch.setattr(runner, 'BASE', base)
    names = ['scripts/run_blueprint_architecture.py', 'scripts/run_blueprint_learning.py',
        'scripts/report_blueprint_validation.py', 'src/pbpk_kb/blueprint_policy/adaptive_calibration.py',
        'src/pbpk_kb/blueprint_policy/adaptive_optimizer.py',
        'src/pbpk_kb/blueprint_policy/axes_calibration.py', 'src/pbpk_kb/blueprint_policy/axes_optimizer.py',
        'src/pbpk_kb/blueprint_policy/native_axes_runtime.py', 'src/pbpk_kb/model_runtime/R/blueprint_axes_trial.R']
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture\n')
    monkeypatch.setattr(runner, '__file__', str(root / names[0]))
    def document(name, value):
        path = base / (name + '.json')
        runner.save(path, value)
        return runner.ref(path)
    human, dosing, target = {'weight_kg': 70}, {'dose_mg': 100}, {'window_end_h': 24}
    original = {'human': human, 'dosing': dosing, 'target': target, 'partition_method': 'PKSimStandard'}
    request = document('request', original)
    target_request = document('target_request', original)
    reference = document('DO_NOT_READ_TARGET_OUTCOMES', {'secret_outcomes': [9, 8, 7]})
    prediction = document('frozen_prediction', {'immutable_prediction': True})
    cases, units, computations, results = [], [], {}, {}
    for drug in runner.PILOT_DRUGS:
        cid = drug.lower()
        cases.append({'case_id': cid, 'drug': drug, 'study_id': cid + '_target', 'cluster_id': cid,
            'reference_source': reference})
        for sampling in runner.SAMPLING:
            training = document(cid + '_' + sampling, {'human': human, 'dosing': dosing,
                'observations': [{'time_min': i + 1, 'concentration_umol_per_l': 1 / (i + 1)}
                    for i in range(5 if sampling == 'five' else 6)]})
            for method in ('evidence_policy', 'two_compartment', 'two_compartment_prior'):
                identity = '__'.join((cid, sampling, method))
                units.append({'case_id': cid, 'sampling': sampling, 'method': method, 'scenario': 'available',
                    'computation_id': identity})
                computations[identity] = {'request': request, 'target_request': target_request, 'training_source': training}
                results[identity] = {'status': 'predicted', 'prediction': prediction, 'request': target_request,
                    'resources': {'total_native_calls': 49}}
    protocol = {'version': 'blueprint-evidence-policy-1', 'cases': cases, 'source_pins': {}, 'units': units, 'computations': computations}
    runner.save(base / 'protocol.json', protocol)
    runner.save(base / 'state.json', {'status': 'complete', 'computations': results})
    return root, group / 'clinical_pilot', protocol, reference


def test_prepare_reuses_identical_inputs_without_reading_target_or_warm_starts(batch, monkeypatch):
    root, run, source, reference = batch
    original = runner.read
    opened = []
    def guarded(path):
        actual = runner.inside(path, root, exists=True)
        opened.append(actual)
        if actual == root / reference['path']:
            raise AssertionError('Target clinical values reached prepare')
        return original(path)
    monkeypatch.setattr(runner, 'read', guarded)
    result = runner.prepare(run)
    assert result == {'drugs': 4, 'new_formula_fits': 64, 'new_predictions': 32,
        'reused_predictions': 16, 'comparison_units': 48}
    protocol = original(run / 'protocol.json')
    assert protocol['native_evaluations_per_formula'] == 48
    assert set(protocol['methods']) == {
        'clearance_only', 'extent_clearance', 'extent_exchange_clearance',
        'old_trust_region', 'two_compartment', 'two_compartment_prior'}
    assert not any(u['method'] == 'frozen_policy' for u in protocol['units'])
    assert sum(c['method'] == 'old_trust_region' for c in protocol['computations'].values()) == 8
    state = original(run / 'state.json')
    assert all(key.endswith(('two_compartment', 'two_compartment_prior')) for key in state['computations'])
    for identity, computation in protocol['computations'].items():
        expected = source['computations']['__'.join((computation['case_id'], computation['sampling'], 'evidence_policy'))]
        assert computation['request'] == expected['request']
        assert computation['training_source'] == expected['training_source']
        assert not any(k in computation for k in ('warm_start', 'mapping', 'previous_fit'))
    assert reference['path'] in protocol['source_pins']
    assert root / reference['path'] not in opened
    assert any('Wise1999' in text for text in protocol['fairness'])
    assert any('predeclared exploratory' in text for text in protocol['fairness'])


def test_prepare_rejects_unfair_baseline_subset(batch):
    root, run, protocol, _ = batch
    target = protocol['computations']['remdesivir__five__two_compartment']
    target['training_source'] = deepcopy(protocol['computations']['remdesivir__full__two_compartment']['training_source'])
    runner.save(runner.BASE / 'protocol.json', protocol)
    with pytest.raises(ValueError, match='different development subset'):
        runner.prepare(run)
    assert not run.exists()


def test_prepare_rejects_hidden_full_curve_metadata(batch):
    root, run, source, _ = batch
    record = source['computations']['remdesivir__five__evidence_policy']['training_source']
    value = runner.checked(record)
    value['raw_observations'] = [{'concentration': 999}]
    runner.save(root / record['path'], value)
    updated = runner.ref(root / record['path'])
    for configuration in source['computations'].values():
        if configuration['training_source']['path'] == record['path']:
            configuration['training_source'] = updated
    runner.save(runner.BASE / 'protocol.json', source)
    with pytest.raises(ValueError, match='forbidden full-curve'):
        runner.prepare(run)


def test_matrix_barrier_rejects_missing_attempt_before_reference_access(batch, monkeypatch):
    _, run, _, _ = batch
    runner.prepare(run)
    import report_blueprint_validation as reports
    monkeypatch.setattr(reports, '_case_reference', lambda *args: pytest.fail('Target read before lock'))
    with pytest.raises(ValueError, match='terminal before scoring'):
        runner.report(run)


def test_formula_selection_needs_both_and_uses_only_development_objective():
    mappings = [{'construction_partition_method': 'PKSimStandard', 'eligible': True, 'decision': {'objective': .4}},
        {'construction_partition_method': 'RodgersRowland', 'eligible': True, 'decision': {'objective': .3}}]
    assert runner.choose_formula(mappings)['construction_partition_method'] == 'RodgersRowland'
    with pytest.raises(ValueError, match='Both predeclared formulas'):
        runner.choose_formula(mappings[:1])
    for item in mappings:
        item['eligible'] = False
    assert runner.choose_formula(mappings) is None


def test_target_null_default_equivalence_does_not_erase_actual_terminal_fit():
    target = {'window_end_h': 24, 'endpoints': ['auc']}
    assert runner.target_signature({**target, 'terminal_fit': None}) == target
    assert runner.target_signature({**target, 'terminal_fit': {'start_h': 3}}) != target


def test_formula_and_prediction_barriers_monitor_stop_and_checkpoint_resume(batch, monkeypatch):
    _, run, _, _ = batch
    runner.prepare(run, workers=3)
    events, controls = [], []
    lock = threading.Lock()
    class Control:
        def __init__(self, path, protocol, state):
            self.run, self.protocol, self.state = path, protocol, state
            self.lock = threading.RLock()
            self.closed, self.halted = threading.Event(), threading.Event()
            self.finished = False
            controls.append(self)
        def persist(self, **values):
            with self.lock:
                self.state.update(values)
                runner.save(self.run / 'state.json', self.state)
        def check(self, force=False):
            if self.halted.is_set():
                raise runner.Paused('test pause')
        def pause(self, reason):
            self.halted.set()
            self.persist(status='paused', pause_reason=reason)
        def watch(self):
            self.closed.wait(30)
            self.finished = True
    monkeypatch.setattr(runner, 'Controller', Control)
    def fit(**kwargs):
        kwargs['before_trial']()
        folder = kwargs['output_dir']
        method = folder.parent.name.rsplit('__', 1)[1]
        assert kwargs['max_evaluations'] == 48
        if method == 'old_trust_region':
            assert kwargs['variant'] == 'trust_region'
            assert 'active_parameters' not in kwargs
        else:
            assert kwargs['active_parameters'] == runner.AXES_METHODS[method]
            assert 'variant' not in kwargs
        folder.mkdir(parents=True)
        formula = kwargs['reference_request']['partition_method']
        mapping = {'status': 'budget_exhausted', 'eligible': True, 'construction_partition_method': formula,
            'decision': {'objective': .2 if formula == 'RodgersRowland' else .4}, 'total_native_calls': 1,
            'failed_evaluation_count': 0, 'elapsed_seconds': .1, 'actual_solver_calls': 1, 'native_budget_includes_base': True}
        runner.save(folder / 'mapping_locked.json', mapping)
        with lock:
            events.append(('fit', str(folder)))
        return {**mapping, 'mapping': runner.ref(folder / 'mapping_locked.json')}
    def predict(**kwargs):
        folder = kwargs['output_dir']
        if (folder / 'completion.json').exists():
            return dict(runner.read(folder / 'result.json'), actual_new_calls=0)
        with lock:
            assert sum(e[0] == 'fit' for e in events) == 64
            events.append(('predict', str(kwargs['output_dir'])))
        assert kwargs['request']['partition_method'] == 'RodgersRowland'
        folder.mkdir(parents=True)
        runner.save(folder / 'request.json', kwargs['request'])
        runner.save(folder / 'prediction_locked.json', {'stub': 'immutable native prediction'})
        runner.save(folder / 'completion.json', {'complete': True})
        calls = 1 if folder.parent.name.endswith('__old_trust_region') else 2
        result = {'status': 'predicted', 'elapsed_seconds': .1, 'actual_new_calls': calls, 'actual_native_solver_calls': calls}
        runner.save(folder / 'result.json', result)
        return result
    engine = SimpleNamespace(fit_mapping=fit, predict_mapping=predict, cleanup_unselected=lambda **kw: {'removed_count': 0})
    monkeypatch.setattr(runner, 'engine_for', lambda method: engine)
    def report(path):
        assert controls[-1].finished
        protocol, state = runner.read(path / 'protocol.json'), runner.read(path / 'state.json')
        runner._validate_locked(protocol, state)
        with lock:
            assert sum(e[0] == 'predict' for e in events) == 32
            events.append(('report', 'after every prediction'))
        result = {'units': [1] * 48}
        runner.save(path / 'results.json', result)
        return result
    monkeypatch.setattr(runner, 'report', report)
    assert runner.execute(run) == {'status': 'complete', 'units': 48}
    assert runner.read(run / 'state.json')['monitor_stopped_before_scoring'] is True
    state = runner.read(run / 'state.json')
    for identity, result in state['computations'].items():
        if identity.endswith('__old_trust_region'):
            assert result['resources']['total_native_calls'] == 3
        elif identity.endswith(tuple('__' + m for m in runner.AXES_METHODS)):
            assert result['resources']['total_native_calls'] == 4
            assert result['resources']['actual_solver_calls'] == 4
    # Recover an immutable target whose receipt preceded the runner checkpoint.
    lost = 'remdesivir__five__extent_clearance'
    state['computations'].pop(lost)
    state['status'] = 'paused'
    runner.save(run / 'state.json', state)
    assert runner.execute(run) == {'status': 'complete', 'units': 48}
    assert runner.read(run / 'state.json')['computations'][lost]['resources']['total_native_calls'] == 4
    before = deepcopy(events)
    assert runner.execute(run) == {'status': 'complete', 'reused': True}
    assert events == before


def test_scope_does_not_touch_cohort_group_or_other_runs(batch):
    _, run, _, _ = batch
    with pytest.raises(ValueError, match='immediate child'):
        runner.own_run(run.parent)
    with pytest.raises(ValueError, match='immediate child'):
        runner.own_run(run / 'nested')
    assert runner.own_run(run) == run


def test_existing_batch_is_never_overwritten(batch):
    _, run, _, _ = batch
    runner.prepare(run)
    before = (run / 'protocol.json').read_bytes()
    with pytest.raises(FileExistsError):
        runner.prepare(run)
    assert (run / 'protocol.json').read_bytes() == before


@pytest.mark.parametrize('damage', ['over_budget', 'base_uncharged', 'solver_overcharge'])
def test_native_budget_counts_base_and_rejects_extra_solver_work(damage):
    record = {'total_native_calls': 48, 'actual_solver_calls': 48, 'native_budget_includes_base': True}
    if damage == 'over_budget':
        record['total_native_calls'] = 49
    elif damage == 'base_uncharged':
        record['native_budget_includes_base'] = False
    else:
        record['actual_solver_calls'] = 49
    with pytest.raises(ValueError):
        runner._fit_budget(record, 'extent_exchange_clearance')


def test_no_native_engine_for_frozen_analytic_baselines():
    with pytest.raises(ValueError, match='Unknown native'):
        runner.engine_for('two_compartment')


def test_engineering_directory_is_outside_runner_scope(batch):
    _, run, _, _ = batch
    with pytest.raises(ValueError, match='immediate child'):
        runner.own_run(run.parent / 'work')


def test_changed_input_version_invalidates_frozen_protocol_before_report(batch):
    _, run, _, _ = batch
    runner.prepare(run)
    protocol = runner.read(run / 'protocol.json')
    state = runner.read(run / 'state.json')
    protocol['native_evaluations_per_formula'] = 24
    with pytest.raises(ValueError, match='Changed architecture native budget'):
        runner._validate_locked(protocol, state)
