"""Twenty-drug protocol/lifecycle isolation, using no native solver or clinical data."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_expanded_tail as expanded
import run_robust_blueprint as frozen
import score_expanded_tail as scorer
from test_robust_runner import batch as base_batch, install_fake_execution


@pytest.fixture
def batch(base_batch, monkeypatch):
    base = base_batch
    counters, _ = install_fake_execution(base, monkeypatch)
    ordinary_fit = base.native.fit_mapping

    def fitting(**kwargs):
        result = ordinary_fit(**kwargs)
        path = base.root / result['mapping']['path']
        mapping = json.loads(path.read_text())
        mapping.update(source_pins={}, code_pins={}, retained_artifact_pins={})
        if kwargs['variant'] == 'development_guarded':
            standard = kwargs['reference_request']['partition_method'] == 'PKSimStandard'
            mapping['formula_selection_score'] = 2. if standard else 1.
            mapping['decision']['objective'] = .001 if standard else 100.
            mapping['clearance_prior_policy'] = {'penalty_weight': .01}
        source = base.document(path, mapping)
        return {**mapping, 'mapping': source}

    base.native.fit_mapping = fitting
    base.native.CONFIG = {'starts': 16, 'cv_starts': 8, 'cv': {
        'folds': ['late', 'interior'], 'rf_penalty_weights': [.01, .1],
        'observed_penalty_weights': [.1]}}
    base.native.clearance_policy = lambda req: {'rule': 'qualified_rf' if next(
        e for e in req['evidence'] if e['parameter'] == 'total_clearance')['basis'] == 'predicted'
        else 'qualified_observed'}
    for name in set(expanded.CODE) | set(scorer.CODE_FILES):
        path = base.root / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# fixture source\n')
    runner = expanded.create_runner(project_root=base.root,
        engine_provider=lambda _: base.native, controller_class=frozen.Controller)
    cases = []
    for index in range(20):
        source_case = base.cases[index % 10]
        case = deepcopy(source_case)
        case_id = f'drug{index}'
        case.update(case_id=case_id, compound_id=case_id, drug=case_id,
            study_id=case_id + '_target', cluster_id=case_id,
            cohort='original' if index < 10 else 'extension', sensitivity_flags=[])
        training = runner.read(base.base / f'drug{index % 10}/training.json')
        training.update(compound_id=case_id, study_id=case_id + '_development')
        training['observations'] = [{'time_min': 10*j, 'concentration_umol_per_l': 1/j}
            for j in range(1, 9)]
        training_ref = base.document(base.base / f'expanded/{case_id}/training.json', training)
        case['sampling_sources'] = {'full': {**training_ref, 'role': 'calibration_data_reconstruction'}}
        for method, roles in case['requests'].items():
            for role, source in list(roles.items()):
                request = runner.checked(source)
                request.update(compound_id=case_id, compound_name=case_id)
                for item in request['evidence']:
                    item['compound_id'] = case_id
                roles[role] = base.document(base.base / f'expanded/{case_id}/{method}_{role}.json', request)
        cases.append(case)
    pins = {}
    runner.api.pin_refs(cases, pins)
    base.document(runner.DEFAULT / 'inputs/unified_manifest.json', {
        'cases': cases, 'source_pins': pins, 'paper_guard': runner.paper_pins()})
    base.expanded, base.expanded_run, base.expanded_cases = runner, runner.DEFAULT, cases
    base.counters = counters
    return base


def test_prepare_twenty_drugs_without_target_decode_and_no_old_cache(batch, monkeypatch):
    runner = batch.expanded
    read = runner.read
    opened = []

    def guarded(path):
        resolved = (batch.root / path).resolve()
        assert resolved != batch.root / batch.outcome['path'], 'Target decoded before prediction lock'
        opened.append(resolved)
        return read(path)

    monkeypatch.setattr(runner, 'read', guarded)
    monkeypatch.setattr(runner.api, 'read', guarded)
    result = runner.prepare(batch.expanded_run, workers=1)
    protocol, state = runner.load(batch.expanded_run)
    assert result['drugs'] == 20 and result['logical_units'] == 240
    assert len(protocol['units']) == len({unit['id'] for unit in protocol['units']}) == 240
    assert len(protocol['computations']) == result['unique_predictions'] == 120
    assert protocol['cached_ids'] == [] and state['computations'] == {}
    assert protocol['optimization_budget'] == {'full_fit_stages': 240, 'cv_stages': 272,
        'full_optimizer_starts': 3840, 'cv_optimizer_starts': 2176, 'optimizer_starts': 6016}
    assert {k: len(v) for k, v in protocol['changed_case_ids'].items()} == {
        'available': 0, 'missing_fu': 8, 'missing_cl': 6, 'missing_both': 8}
    assert batch.root / batch.outcome['path'] not in opened
    for computation in protocol['computations'].values():
        training = runner.checked(computation['training_source'])
        assert 'hidden_reference_metadata' not in training
        assert all(set(row) == {'time_min', 'concentration_umol_per_l'} for row in training['observations'])


def test_complete_predictions_lock_before_score_and_reports_are_idempotent(batch, monkeypatch):
    runner = batch.expanded
    runner.prepare(batch.expanded_run, workers=1)
    calls = []

    def score(case, computation, store):
        protocol, state = runner.load(batch.expanded_run, locked=True)
        assert state['all_attempts_locked_before_scoring'] and state['monitor_stopped_before_scoring']
        assert len(state['computations']) == len(protocol['computations']) == 120
        assert state['formulas_locked_before_target_prediction']
        calls.append(case['case_id'])
        store.ref(case['reference_source'])
        return {'curve_metrics': {'gmfe': 1.2}, 'endpoints': {
            'auc': {'value': 2., 'reference': 2., 'error': 1., 'error_metric': 'fold_error',
                'unit': 'ng.h/ml', 'reference_kind': 'original_reported' if case['cohort'] == 'original' else 'observation_derived'},
            'tmax': {'value': .5, 'reference': .5, 'error': 0., 'error_metric': 'absolute_error_h',
                'unit': 'h', 'reference_kind': 'original_reported' if case['cohort'] == 'original' else 'observation_derived'}}}

    monkeypatch.setattr(scorer, 'score_case', score)
    outcome = runner.execute(batch.expanded_run)
    assert outcome == {'status': 'complete', 'units': 240}
    assert batch.counters['fit'] == 240 and batch.counters['predict'] == 120
    assert len(calls) == 120
    protocol, state = runner.load(batch.expanded_run, locked=True)
    for identity, specification in protocol['computations'].items():
        if specification['method'] == 'development_guarded':
            assert state['computations'][identity]['selected_formula'] == 'RodgersRowland'
    result = runner.report(batch.expanded_run)
    assert len(result['units']) == 240 and len(result['summary']) == 48
    assert len(result['stability']) == 180 and result['unique_scored_predictions'] == 120
    assert result['paper_updated'] is False and result['paper_guard_verified'] is True
    combined = next(row for row in result['summary'] if row['cohort'] == 'all_20')
    assert combined['endpoint_errors']['auc__original_reported']['n'] == 10
    assert combined['endpoint_errors']['auc__observation_derived']['n'] == 10
    assert combined['endpoint_errors']['tmax__original_reported']['mean_absolute_error_h'] == 0
    digest = runner.sha256(batch.expanded_run / 'results.json')
    runner.report(batch.expanded_run)
    runner.execute(batch.expanded_run)
    assert digest == state['results']['sha256'] == runner.sha256(batch.expanded_run / 'results.json')
    assert batch.counters['fit'] == 240 and batch.counters['predict'] == len(calls) == 120


def test_partial_batch_cannot_decode_clinical_reference(batch, monkeypatch):
    runner = batch.expanded
    runner.prepare(batch.expanded_run, workers=1)
    decoded = []
    monkeypatch.setattr(scorer, 'score_case', lambda *args: decoded.append(args))
    with pytest.raises(ValueError, match='lock'):
        runner.report(batch.expanded_run)
    assert decoded == []


@pytest.mark.parametrize('change', ['source', 'paper_contents', 'paper_file_set'])
def test_changed_source_or_paper_prevents_resume(batch, change):
    runner = batch.expanded
    runner.prepare(batch.expanded_run, workers=1)
    if change == 'source':
        source = batch.expanded_cases[0]['sampling_sources']['full']['path']
        (batch.root / source).write_text('{}')
    elif change == 'paper_contents':
        batch.paper.write_text('paper changed')
    else:
        (batch.paper.parent / 'unapproved_figure.svg').write_text('<svg/>')
    with pytest.raises(ValueError, match='[Cc]hang'):
        runner.load(batch.expanded_run)


def test_cli_run_isolation_and_instances_do_not_modify_frozen_globals(batch):
    old = frozen.METHODS, frozen.VERSION, frozen.__file__, frozen.api.choose_formula
    runner = expanded.create_runner(project_root=batch.root,
        engine_provider=lambda _: batch.native, controller_class=frozen.Controller)
    assert (frozen.METHODS, frozen.VERSION, frozen.__file__, frozen.api.choose_formula) == old
    assert runner.api is not frozen.api and runner.api is not batch.expanded.api
    assert runner.variant('current') == 'evidence_flexible'
    assert runner.__file__ == str(batch.root / 'scripts/run_expanded_tail.py')
    assert runner.own_run(runner.DEFAULT) == runner.DEFAULT
    for path in (batch.run, batch.root / 'expanded_tail_wrong_parent', runner.DEFAULT / 'expanded_tail_nested'):
        with pytest.raises(ValueError, match='dedicated'):
            runner.own_run(path)
    symlink = runner.DEFAULT.with_name('expanded_tail_symlink')
    symlink.symlink_to(runner.DEFAULT, target_is_directory=True)
    with pytest.raises(ValueError, match='dedicated'):
        runner.own_run(symlink)


def test_summary_preserves_failed_denominators_and_reference_kinds_without_promotion():
    cases = [{'case_id': f'drug{i}', 'drug': f'drug{i}', 'cluster_id': f'study{i}',
        'cohort': 'original' if i < 10 else 'extension'} for i in range(20)]
    rows = []
    scenarios = ('available', 'missing_fu', 'missing_cl', 'missing_both')
    for case in cases:
        for scenario in scenarios:
            for method in expanded.METHODS:
                row = {**case, 'method': method, 'scenario': scenario,
                    'curve_metrics': {'gmfe': 2. if method == 'current' else 1.5},
                    'endpoints': {'auc': {'error': 1.3, 'error_metric': 'fold_error',
                        'reference_kind': 'original_reported' if case['cohort'] == 'original' else 'observation_derived'}}}
                if case['case_id'] == 'drug19' and method == 'development_guarded':
                    row.pop('curve_metrics')
                    row['endpoints'] = {}
                rows.append(row)
    summary, pairs = expanded.summarize_records(rows, cases, scenarios, expanded.METHODS)
    lookup = {(r['cohort'], r['scenario'], r['method']): r for r in summary}
    for cohort, expected in [('all_20', 20), ('original_10', 10), ('extension_10', 10)]:
        row = lookup[cohort, 'available', 'current']
        assert row['requested'] == row['scored'] == expected
    row = lookup['all_20', 'available', 'development_guarded']
    assert row['requested'] == 20 and row['scored'] == 19 and row['failed'] == 1
    assert row['endpoint_errors']['auc__original_reported']['n'] == 10
    assert row['endpoint_errors']['auc__observation_derived']['n'] == 9
    comparison = pairs['all_20']['available']['development_guarded']
    assert comparison['attempted_pairs'] == 20 and comparison['complete_pairs'] == 19
    assert comparison['incomplete_case_ids'] == ['drug19']
    checks = expanded.nonregression_checks(rows, summary, cases)
    assert checks['tail_balanced']['all_four_aggregate_errors_nonincreasing'] is True
    assert checks['development_guarded']['all_four_aggregate_errors_nonincreasing'] is False
    assert len(checks['development_guarded']['incomplete_pairs']) == 4
    assert all(value['automatic_promotion'] is False for value in checks.values())


@pytest.mark.parametrize('missing_side', ['available', 'missing_fu'])
def test_stability_keeps_curve_comparison_when_one_side_has_no_endpoints(missing_side):
    base = {'case_id': 'drug', 'drug': 'Drug', 'cohort': 'extension', 'cluster_id': 'study',
        'method': 'current', 'scientific_inputs_changed': True, 'attempt_status': 'predicted',
        'prediction_source': 'locked_curve', 'endpoints': {'auc': {'value': 2., 'unit': 'ng.h/ml'}}}
    rows = [{**deepcopy(base), 'scenario': scenario} for scenario in ('available', 'missing_fu')]
    next(row for row in rows if row['scenario'] == missing_side).pop('endpoints')
    store = SimpleNamespace(ref=lambda _: {'time_min': [0, 30, 60, 120], 'concentration': [0, 2, 1, .5]})
    cases = [{'case_id': 'drug', 'nca_policy': {'sampling_time_h': [.5, 1., 2.]}}]
    result = expanded.prediction_stability(rows, cases, store)
    assert len(result) == 1 and result[0]['status'] == 'compared'
    assert result[0]['curve_prediction_drift'] == 1.
    assert result[0]['endpoint_prediction_drift'] == {}
