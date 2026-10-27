"""Frozen-control, resource and target-access contracts; no native simulations."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
runner = importlib.import_module('run_blueprint_refinement')
REAL_REPORT = runner.report


@pytest.fixture
def batch(tmp_path, monkeypatch):
    root = tmp_path
    group = root / 'experiments/blueprint_design/architecture_20260930'
    base = group / 'clinical_pilot'
    base.mkdir(parents=True)
    for name, value in (('ROOT', root), ('GROUP', group), ('BASE', base)):
        monkeypatch.setattr(runner, name, value)
    names = ('scripts/run_blueprint_refinement.py', 'scripts/run_blueprint_architecture.py',
        'scripts/run_blueprint_learning.py', 'scripts/report_blueprint_validation.py',
        'src/pbpk_kb/blueprint_policy/refined_axes_calibration.py',
        'src/pbpk_kb/blueprint_policy/refined_axes_optimizer.py')
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic frozen source\n')
    monkeypatch.setattr(runner, '__file__', str(root / names[0]))
    engine = SimpleNamespace(VERSION='synthetic-refinement-1', CONFIG={'same_loss': True},
        __file__=str(root / names[4]),
        _code_pins=lambda project_root: {name: runner.sha256(root / name) for name in names[4:]})
    monkeypatch.setattr(runner, 'engine', lambda: engine)
    def document(name, value):
        path = base / (name + '.json')
        runner.save(path, value)
        return runner.ref(path)
    human, dosing, target = {'weight_kg': 70}, {'dose_mg': 100}, {'window_end_h': 24}
    request_value = {'human': human, 'dosing': dosing, 'target': target, 'partition_method': 'PKSimStandard'}
    request = document('development_request', request_value)
    target_request = document('target_request', request_value)
    reference = document('DO_NOT_READ_TARGET_OBSERVATIONS', {'secret': [9, 8, 7]})
    old_mapping = document('DO_NOT_READ_OLD_MAPPING', {'decision': {'parameters': {'secret_warm_start': 99}}})
    old_scores = document('DO_NOT_READ_OLD_SCORES', {'secret_scores': [99]})
    prediction = document('frozen_prediction', {'immutable_prediction': True})
    cases, units, computations, results, fits = [], [], {}, {}, {}
    for drug in runner.PILOT_DRUGS:
        cid = drug.lower()
        cases.append({'case_id': cid, 'drug': drug, 'study_id': cid + '_target', 'cluster_id': cid,
            'reference_source': reference})
        for sampling in runner.SAMPLING:
            training = document(cid + '_' + sampling, {'human': human, 'dosing': dosing,
                'observations': [{'time_min': i + 1, 'concentration_umol_per_l': 1 / (i + 1)}
                    for i in range(5 if sampling == 'five' else 6)]})
            for method in runner.BASELINES:
                identity = runner.unit_id(cid, sampling, method)
                units.append({'id': identity, 'case_id': cid, 'sampling': sampling, 'method': method})
                results[identity] = {'status': 'predicted', 'prediction': prediction, 'request': target_request,
                    'resources': {'total_native_calls': 80, 'actual_solver_calls': 79},
                    'source_pins': {training['path']: training['sha256']}}
                if method in runner.frozen.VARIANTS:
                    computations[identity] = {'request': request, 'target_request': target_request, 'training_source': training,
                        'method': method, 'sampling': sampling, 'case_id': cid}
                    for formula in runner.FORMULAS:
                        fits[runner.fit_id(identity, formula)] = {'mapping': old_mapping}
                else:
                    results[identity]['original_resources'] = {'analytic_cpu_residual_calls': 100, 'native_execution_calls': 0}
                    results[identity]['resources'] = {'actual_new_native_calls': 0, 'cache_reused': True}
    protocol = {'version': runner.frozen.VERSION, 'cases': cases, 'source_pins': {names[1]: runner.sha256(root / names[1])},
        'methods': list(runner.BASELINES), 'sampling': list(runner.SAMPLING), 'formulas': list(runner.FORMULAS),
        'native_evaluations_per_formula': 48, 'units': units, 'computations': computations}
    runner.save(base / 'protocol.json', protocol)
    state = {'version': runner.frozen.VERSION, 'status': 'complete', 'protocol': runner.ref(base / 'protocol.json'),
        'fits': fits, 'computations': results, 'results': old_scores,
        'all_attempts_locked_before_scoring': True, 'monitor_stopped_before_scoring': True}
    runner.save(base / 'state.json', state)
    return SimpleNamespace(root=root, run=group / 'clinical_refinement_v2', source=protocol, state=state,
        opaque=[reference, old_mapping, old_scores], engine=engine, reference=reference)


def seal_source(batch):
    runner.save(runner.BASE / 'protocol.json', batch.source)
    batch.state['protocol'] = runner.ref(runner.BASE / 'protocol.json')
    runner.save(runner.BASE / 'state.json', batch.state)


def test_prepare_exact_matrix_and_inputs_without_decoding_targets_fits_or_scores(batch, monkeypatch):
    original = runner.read
    opaque = {batch.root / item['path'] for item in batch.opaque}
    opened = []
    def guarded(path):
        actual = runner.inside(path, batch.root, exists=True)
        assert actual not in opaque, 'Outcome or historical mapping reached preparation'
        opened.append(actual)
        return original(path)
    monkeypatch.setattr(runner, 'read', guarded)
    before = {p: p.read_bytes() for p in runner.BASE.rglob('*') if p.is_file()}
    result = runner.prepare(batch.run)
    assert result == {'drugs': 4, 'new_formula_fits': 16, 'new_predictions': 8,
        'reused_predictions': 48, 'comparison_units': 56}
    protocol, state = original(batch.run / 'protocol.json'), original(batch.run / 'state.json')
    assert state['fits'] == {}
    assert len(state['computations']) == 48
    assert protocol['native_evaluations_per_formula'] == 48
    assert protocol['engine']['config'] == batch.engine.CONFIG
    for value in protocol['computations'].values():
        old = batch.source['computations'][runner.unit_id(value['case_id'], value['sampling'], 'extent_exchange_clearance')]
        for name in ('request', 'training_source', 'target_request'):
            assert value[name] == old[name]
        assert not any(name in value for name in ('mapping', 'warm_start', 'active_parameters'))
    assert all(item['path'] in protocol['source_pins'] for item in batch.opaque)
    assert not opaque.intersection(opened)
    assert before == {p: p.read_bytes() for p in runner.BASE.rglob('*') if p.is_file()}
    assert runner.frozen.METHODS == runner.BASELINES  # Shared module globals were not patched.
    assert any('Wise1999' in text for text in protocol['fairness'])


def test_reuse_retains_original_analytic_and_native_costs(batch):
    runner.prepare(batch.run)
    state = runner.read(batch.run / 'state.json')
    for identity, value in state['computations'].items():
        assert value['resources']['actual_new_native_calls'] == 0
        if identity.endswith(('__two_compartment', '__two_compartment_prior')):
            assert value['original_resources']['analytic_cpu_residual_calls'] == 100
        else:
            assert value['original_resources']['total_native_calls'] == 80
            assert value['resources']['cached_native_calls'] == 80


@pytest.mark.parametrize('damage', ['source_code', 'old_mapping', 'old_prediction', 'old_scores', 'state_protocol'])
def test_prepare_rejects_changed_frozen_artifacts(batch, damage):
    if damage == 'source_code':
        (batch.root / 'scripts/run_blueprint_architecture.py').write_text('# changed\n')
    elif damage == 'state_protocol':
        batch.state['protocol']['sha256'] = '0' * 64
        runner.save(runner.BASE / 'state.json', batch.state)
    else:
        record = {'old_mapping': batch.opaque[1], 'old_scores': batch.opaque[2],
            'old_prediction': next(iter(batch.state['computations'].values()))['prediction']}[damage]
        (batch.root / record['path']).write_text('{"changed": true}\n')
    with pytest.raises(ValueError):
        runner.prepare(batch.run)
    assert not batch.run.exists()


@pytest.mark.parametrize('damage', ['native_subset', 'analytic_subset', 'target', 'full_curve'])
def test_prepare_rejects_different_comparison_information(batch, damage):
    identity = runner.unit_id('remdesivir', 'five', 'clearance_only')
    if damage == 'native_subset':
        batch.source['computations'][identity]['training_source'] = batch.source['computations'][identity.replace('__five__', '__full__')]['training_source']
    elif damage == 'analytic_subset':
        batch.state['computations'][runner.unit_id('remdesivir', 'five', 'two_compartment')]['source_pins'] = {}
    elif damage == 'target':
        record = batch.state['computations'][identity]['request']
        value = runner.checked(record)
        value['target']['window_end_h'] = 48
        runner.save(runner.BASE / 'different_target.json', value)
        batch.state['computations'][identity]['request'] = runner.ref(runner.BASE / 'different_target.json')
    else:
        record = batch.source['computations'][identity]['training_source']
        value = runner.checked(record)
        value['raw_observations'] = [{'concentration': 999}]
        runner.save(batch.root / record['path'], value)
        changed = runner.ref(batch.root / record['path'])
        for config in batch.source['computations'].values():
            if config['training_source']['path'] == record['path']:
                config['training_source'] = changed
        for result in batch.state['computations'].values():
            if record['path'] in result['source_pins']:
                result['source_pins'][record['path']] = changed['sha256']
    seal_source(batch)
    with pytest.raises(ValueError):
        runner.prepare(batch.run)
    assert not batch.run.exists()


def test_report_barrier_precedes_target_reference_access(batch, monkeypatch):
    runner.prepare(batch.run)
    import report_blueprint_validation as reports
    monkeypatch.setattr(reports, '_case_reference', lambda *args: pytest.fail('Target read before complete prediction lock'))
    with pytest.raises(ValueError, match='Every fit and prediction'):
        runner.report(batch.run)


@pytest.mark.parametrize('damage', ['over_budget', 'base_uncharged', 'solver_overcharge', 'solver_omitted'])
def test_fit_cost_contract_includes_base_and_failed_attempts(damage):
    mapping = {'total_native_calls': 48, 'actual_solver_calls': 47, 'native_budget_includes_base': True}
    if damage == 'over_budget':
        mapping['total_native_calls'] = 49
    elif damage == 'base_uncharged':
        mapping['native_budget_includes_base'] = False
    elif damage == 'solver_overcharge':
        mapping['actual_solver_calls'] = 49
    else:
        mapping.pop('actual_solver_calls')
    with pytest.raises(ValueError):
        runner.fit_budget(mapping)


@pytest.fixture
def simulated_execution(batch, monkeypatch):
    runner.prepare(batch.run, workers=3)
    events, controls = [], []
    event_lock = threading.Lock()
    class Control:
        def __init__(self, run, protocol, state):
            self.run, self.protocol, self.state = run, protocol, state
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
                raise runner.Paused(self.state.get('pause_reason', 'test pause'))
        def pause(self, reason):
            self.halted.set()
            self.persist(status='paused', pause_reason=reason)
        def watch(self):
            self.closed.wait(30)
            self.finished = True
    monkeypatch.setattr(runner, 'Controller', Control)
    def fit(**kwargs):
        kwargs['before_trial']()
        assert kwargs['max_evaluations'] == 48
        assert 'active_parameters' not in kwargs and 'variant' not in kwargs
        folder = kwargs['output_dir']
        folder.mkdir(parents=True)
        formula = kwargs['reference_request']['partition_method']
        value = {'status': 'budget_exhausted', 'eligible': True, 'construction_partition_method': formula,
            'decision': {'objective': .2 if formula == 'RodgersRowland' else .4}, 'total_native_calls': 48,
            'failed_evaluation_count': 1, 'elapsed_seconds': .1, 'actual_solver_calls': 47, 'native_budget_includes_base': True}
        runner.save(folder / 'mapping_locked.json', value)
        with event_lock:
            events.append(('fit', str(folder)))
        return {**value, 'mapping': runner.ref(folder / 'mapping_locked.json')}
    def predict(**kwargs):
        assert callable(kwargs['before_trial'])
        kwargs['before_trial']()
        folder = kwargs['output_dir']
        if (folder / 'completion.json').exists():
            return dict(runner.read(folder / 'result.json'), actual_new_calls=0)
        state = runner.read(batch.run / 'state.json')
        assert len(state['fits']) == 16
        assert len(runner.checked(state['formula_selection'])['choices']) == 8
        assert kwargs['request']['partition_method'] == 'RodgersRowland'
        with event_lock:
            assert sum(event[0] == 'fit' for event in events) == 16
            events.append(('predict', str(folder)))
        folder.mkdir(parents=True)
        runner.save(folder / 'request.json', kwargs['request'])
        runner.save(folder / 'prediction_locked.json', {'stub': 'immutable prediction'})
        runner.save(folder / 'completion.json', {'complete': True})
        result = {'status': 'predicted', 'elapsed_seconds': .1, 'actual_new_calls': 2, 'actual_native_solver_calls': 2}
        runner.save(folder / 'result.json', result)
        return result
    batch.engine.fit_mapping = fit
    batch.engine.predict_mapping = predict
    batch.engine.cleanup_unselected = lambda **kwargs: {'removed_count': 0}
    def report(run):
        assert controls[-1].finished
        protocol, state = runner.load_batch(run)
        runner.validate_matrix(protocol, state, locked=True)
        with event_lock:
            assert sum(event[0] == 'predict' for event in events) == 8
            events.append(('report', 'after all targets'))
        result = {'units': [1] * 56}
        runner.save(run / 'results.json', result)
        return result
    monkeypatch.setattr(runner, 'report', report)
    return batch, events, controls


def test_all_fit_formula_and_target_barriers_and_checkpoint_recovery(simulated_execution):
    batch, events, _ = simulated_execution
    assert runner.execute(batch.run) == {'status': 'complete', 'units': 56}
    state = runner.read(batch.run / 'state.json')
    for identity, result in state['computations'].items():
        if identity.endswith('__refined_axes'):
            assert result['resources']['total_native_calls'] == 98
            assert result['resources']['actual_solver_calls'] == 96
        else:
            assert result['resources']['actual_new_native_calls'] == 0
    # A target receipt can survive an interruption before its state checkpoint.
    missing = runner.unit_id('remdesivir', 'five', runner.METHOD)
    state['computations'].pop(missing)
    state['status'] = 'paused'
    runner.save(batch.run / 'state.json', state)
    assert runner.execute(batch.run) == {'status': 'complete', 'units': 56}
    restored = runner.read(batch.run / 'state.json')['computations'][missing]
    assert restored['resources']['total_native_calls'] == 98
    assert sum(event[0] == 'fit' for event in events) == 16
    assert sum(event[0] == 'predict' for event in events) == 8
    assert runner.execute(batch.run) == {'status': 'complete', 'reused': True}


def test_engine_configuration_change_prevents_any_new_fit(batch):
    runner.prepare(batch.run)
    batch.engine.CONFIG['same_loss'] = False
    with pytest.raises(ValueError, match='configuration changed'):
        runner.execute(batch.run)
    assert not (batch.run / 'work').exists()


def test_source_change_after_prepare_prevents_scoring_or_new_work(batch):
    runner.prepare(batch.run)
    (runner.BASE / 'state.json').write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match='Frozen source/code'):
        runner.execute(batch.run)
    assert not (batch.run / 'work').exists()


def test_directory_scope_and_existing_batch_protection(batch):
    for path in (runner.GROUP, runner.BASE, runner.GROUP / 'work', batch.run / 'nested'):
        with pytest.raises(ValueError, match='dedicated immediate'):
            runner.own_run(path)
    runner.prepare(batch.run)
    before = (batch.run / 'protocol.json').read_bytes()
    with pytest.raises(FileExistsError):
        runner.prepare(batch.run)
    assert (batch.run / 'protocol.json').read_bytes() == before


def test_cleanup_only_visits_declared_new_fits(simulated_execution):
    batch, _, _ = simulated_execution
    runner.execute(batch.run)
    cleaned = []
    def cleanup(**kwargs):
        assert kwargs['output_dir'].is_relative_to(batch.run / 'work')
        assert kwargs['output_dir'].parent.name.endswith('__refined_axes')
        cleaned.append(kwargs['output_dir'])
        return {'removed_count': 0}
    batch.engine.cleanup_unselected = cleanup
    assert runner.cleanup(batch.run) == {'fits_checked': 16, 'removed_count': 0}
    assert len(set(cleaned)) == 16


def test_status_counts_only_new_work_as_progress(batch):
    runner.prepare(batch.run)
    value = runner.status(batch.run)
    assert value['fits_total'] == 16 and value['new_predictions_total'] == 8
    assert value['frozen_baselines_reused'] == 48 and value['comparison_units'] == 56


def test_sealed_report_scores_all_controls_and_preserves_costs_and_sensitivity(simulated_execution, monkeypatch):
    batch, _, _ = simulated_execution
    runner.execute(batch.run)
    state = runner.read(batch.run / 'state.json')
    state.update(status='predictions_locked')
    state.pop('results')
    runner.save(batch.run / 'state.json', state)
    import report_blueprint_validation as reports
    reads, scores = [], []
    def reference(case, store):
        locked = runner.read(batch.run / 'state.json')
        assert len(locked['computations']) == 56 and locked['monitor_stopped_before_scoring']
        reads.append(case['case_id'])
        store.ref(case['reference_source'])
        return {'approximations': []}, {'unit': 'umol/l', 'time_h': [1, 2]}, {
            key: {'value': 1., 'unit': unit} for key, unit in reports.UNITS.items()}
    def score(case, clinical, curve, endpoints, unit, store):
        store.ref(unit['prediction'])
        scores.append(case['case_id'])
        return {'curve_metrics': {'gmfe': 2., 'ln_rmse': .5, 'n': 2},
            'endpoints': {key: {'value': 2., 'error': 1. if key == 'tmax' else 2., 'unit': value['unit']}
                for key, value in endpoints.items()}}, None, None
    monkeypatch.setattr(reports, '_case_reference', reference)
    monkeypatch.setattr(reports, '_score_prediction', score)
    result = REAL_REPORT(batch.run)
    assert len(reads) == 4 and len(scores) == 56
    assert set(result['statistics']['paired_curve_gmfe_ratios']['full'][runner.METHOD]) == set(runner.BASELINES)
    assert result['sensitivity_without_moxifloxacin']['by_sampling']['full'][runner.METHOD]['curve']['attempted'] == 3
    assert all(r['original_resources'] for r in result['units'] if r['method'] != runner.METHOD)
    assert str((batch.run / 'state.json').relative_to(batch.root)) not in result['source_pins']
    state.update(status='complete', results=runner.ref(batch.run / 'results.json'))
    runner.save(batch.run / 'state.json', state)
    assert REAL_REPORT(batch.run) == result
    assert len(reads) == 4 and len(scores) == 56


def test_changed_formula_lock_is_rejected_before_target_resume(simulated_execution):
    batch, _, _ = simulated_execution
    runner.execute(batch.run)
    state = runner.read(batch.run / 'state.json')
    state['status'] = 'paused'
    state['computations'].pop(runner.unit_id('remdesivir', 'five', runner.METHOD))
    runner.save(batch.run / 'state.json', state)
    selection = runner.read(batch.run / 'formulas_locked.json')
    selection['choices'][runner.unit_id('remdesivir', 'five', runner.METHOD)]['formula'] = 'PKSimStandard'
    runner.save(batch.run / 'formulas_locked.json', selection)
    with pytest.raises(ValueError, match='selection changed'):
        runner.execute(batch.run)
