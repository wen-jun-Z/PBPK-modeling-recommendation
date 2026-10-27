"""Frozen-control reuse and twenty-drug lifecycle; no clinical/native execution."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_adaptive_twenty as adaptive
import run_expanded_tail as expanded
import run_robust_blueprint as frozen
import score_expanded_tail as scorer
from test_expanded_tail_runner import batch as expanded_batch, base_batch


@pytest.fixture
def batch(expanded_batch, monkeypatch):
    base = expanded_batch
    previous = base.expanded
    # The source fixture has 40 distinct input combinations. drug13 has an
    # observed fu but RF-only CL; independently observed CL makes its four
    # missingness combinations distinct, giving the real batch's 42 controls.
    case = base.expanded_cases[13]
    for role, reference in list(case['requests']['evidence'].items()):
        request = previous.checked(reference)
        record = next(e for e in request['evidence'] if e['parameter'] == 'total_clearance')
        record.update(value=4., basis='measured')
        record.pop('prediction_method', None)
        case['requests']['evidence'][role] = base.document(base.root / reference['path'], request)
    pins = {}
    previous.api.pin_refs(base.expanded_cases, pins)
    base.document(base.expanded_run / 'inputs/unified_manifest.json', {
        'cases': base.expanded_cases, 'source_pins': pins, 'paper_guard': previous.paper_pins()})
    base.native._validated = lambda mapping, path, root: None
    for name in set(adaptive.CODE) | set(scorer.CODE_FILES):
        path = base.root / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# fixture source\n')

    # Exercise the source's preparation, fit/prediction locks and completion.
    # Its scientific scoring is outside this runner test's scope.
    def source_report(run):
        protocol, state = previous.load(run, locked=True)
        assert state['formulas_locked_before_target_prediction']
        result = {'units': protocol['units'], 'source_pins': {}}
        previous.save(run / 'results.json', result)
        return result
    monkeypatch.setattr(previous, 'report', source_report)
    prepared = previous.prepare(base.expanded_run, workers=1)
    assert prepared['unique_predictions'] == 126
    assert previous.execute(base.expanded_run) == {'status': 'complete', 'units': 240}
    base.source_protocol, base.source_state = previous.load(base.expanded_run, locked=True)
    base.source_hashes = {str(path.relative_to(base.root)): previous.sha256(path)
        for path in base.expanded_run.rglob('*.json')}
    base.counters.clear()

    # Make the full-development choice disagree strongly with a decoy CV
    # score, so selection cannot silently inherit the previous guarded rule.
    ordinary_fit = base.native.fit_mapping
    def fitting(**kwargs):
        result = ordinary_fit(**kwargs)
        path = base.root / result['mapping']['path']
        mapping = previous.read(path)
        if kwargs['variant'] in ('relaxed_tail', 'adaptive_exchange'):
            standard = mapping['construction_partition_method'] == 'PKSimStandard'
            mapping['decision']['objective'] = .1 if standard else 10.
            mapping['formula_selection_score'] = 100. if standard else .0001
        reference = base.document(path, mapping)
        return {**mapping, 'mapping': reference}
    base.native.fit_mapping = fitting
    base.adaptive = adaptive.create_runner(project_root=base.root,
        engine_provider=lambda _: base.native, controller_class=frozen.Controller)
    base.adaptive_run = base.adaptive.DEFAULT
    return base


def assert_source_unchanged(batch):
    assert batch.source_hashes == {str(path.relative_to(batch.root)): batch.adaptive.sha256(path)
        for path in batch.expanded_run.rglob('*.json')}


def test_prepare_reuses_only_exact_tail_controls_without_decoding_target(batch, monkeypatch):
    runner = batch.adaptive
    read = runner.read
    decoded = []
    def guarded(path):
        actual = (batch.root / path).resolve()
        assert actual != batch.root / batch.outcome['path'], 'Target decoded before prediction lock'
        decoded.append(actual)
        return read(path)
    monkeypatch.setattr(runner, 'read', guarded)
    monkeypatch.setattr(runner.api, 'read', guarded)
    outcome = runner.prepare(batch.adaptive_run, workers=1)
    protocol, state = runner.load(batch.adaptive_run)
    assert outcome['logical_units'] == len(protocol['units']) == 240
    assert len({u['id'] for u in protocol['units']}) == 240
    assert outcome['new_predictions'] == len(protocol['computations']) == 84
    assert outcome['new_fits'] == 168
    assert outcome['cached_predictions'] == len(state['computations']) == 42
    assert outcome['baseline_source_method'] == 'tail_balanced'
    assert protocol['optimization_budget']['cv_stages'] == 0
    assert protocol['optimization_budget']['optimizer_starts'] == 2688
    assert {s['method'] for s in protocol['computations'].values()} == {'relaxed_tail', 'adaptive_exchange'}
    expected = {u['computation_id'] for u in batch.source_protocol['units'] if u['method'] == 'tail_balanced'}
    wrong = {u['computation_id'] for u in batch.source_protocol['units'] if u['method'] == 'current'}
    assert set(protocol['cached_ids']) == expected
    assert not expected & wrong
    for identity, cached in state['computations'].items():
        original = batch.source_state['computations'][identity]
        assert cached['source_method'] == 'tail_balanced'
        assert cached['resources'] == {'total_native_calls': 0, 'linear_solver_calls': 0,
            'wall_seconds': 0, 'cache_reused': True}
        assert cached['original_resources'] == original['resources']
        for field in ('mapping', 'request', 'prediction', 'completion', 'selected_formula', 'dense_endpoints'):
            assert cached[field] == original[field]
    for specification in protocol['computations'].values():
        training = runner.checked(specification['training_source'])
        assert 'hidden_reference_metadata' not in training
        assert all(set(row) == {'time_min', 'concentration_umol_per_l'} for row in training['observations'])
    assert batch.root / batch.outcome['path'] not in decoded
    assert batch.counters['fit'] == batch.counters['predict'] == 0
    assert_source_unchanged(batch)


def test_lifecycle_locks_all_predictions_before_score_and_resume_is_idempotent(batch, monkeypatch):
    runner = batch.adaptive
    runner.prepare(batch.adaptive_run, workers=1)
    protocol, _ = runner.load(batch.adaptive_run)
    expected_ids = set(protocol['computations']) | set(protocol['cached_ids'])
    expected_fits = {runner.key(cid, formula) for cid in protocol['computations'] for formula in runner.FORMULAS}
    score_calls = []
    def score(case, computation, store):
        # report/load performs the strong whole-batch hash check once. A fresh
        # checkpoint read here checks the temporal gate without O(n^2) hashing.
        state = json.loads((batch.adaptive_run / 'state.json').read_text())
        assert state['all_attempts_locked_before_scoring']
        assert state['monitor_stopped_before_scoring']
        assert state['formulas_locked_before_target_prediction']
        assert set(state['computations']) == expected_ids
        assert set(state['fits']) == expected_fits
        assert all(item['status'] == 'predicted' for item in state['computations'].values())
        score_calls.append(case['case_id'])
        store.ref(case['reference_source'])
        return {'curve_metrics': {'gmfe': 1.2}, 'endpoints': {
            'auc': {'value': 2., 'reference': 2., 'error': 1., 'error_metric': 'fold_error',
                'unit': 'ng.h/ml', 'reference_kind': 'original_reported' if case['cohort'] == 'original' else 'observation_derived'}}}
    monkeypatch.setattr(scorer, 'score_case', score)
    assert runner.execute(batch.adaptive_run) == {'status': 'complete', 'units': 240}
    assert batch.counters['fit'] == 168 and batch.counters['predict'] == 84
    assert len(score_calls) == 126
    _, state = runner.load(batch.adaptive_run, locked=True)
    for identity in protocol['computations']:
        assert state['computations'][identity]['selected_formula'] == 'PKSimStandard'
    result = runner.report(batch.adaptive_run)
    assert len(result['units']) == 240 and len(result['summary']) == 48
    assert result['unique_scored_predictions'] == 126 and len(result['stability']) == 180
    assert result['paper_updated'] is False and result['paper_guard_verified'] is True
    assert 'tail_balanced' in result['baseline_semantics']
    digest = runner.sha256(batch.adaptive_run / 'results.json')
    assert runner.report(batch.adaptive_run) == result
    assert runner.execute(batch.adaptive_run) == {'status': 'complete', 'reused': True}
    assert digest == state['results']['sha256'] == runner.sha256(batch.adaptive_run / 'results.json')
    assert batch.counters['fit'] == 168 and batch.counters['predict'] == 84 and len(score_calls) == 126
    assert_source_unchanged(batch)


def test_partial_batch_cannot_score_or_decode_target(batch, monkeypatch):
    runner = batch.adaptive
    runner.prepare(batch.adaptive_run, workers=1)
    decoded = []
    monkeypatch.setattr(scorer, 'score_case', lambda *args: decoded.append(args))
    with pytest.raises(ValueError, match='lock'):
        runner.report(batch.adaptive_run)
    assert decoded == []


@pytest.mark.parametrize('change', ['code', 'paper_contents', 'paper_file_set', 'source_state'])
def test_changed_code_paper_or_locked_source_blocks_resume(batch, change):
    runner = batch.adaptive
    runner.prepare(batch.adaptive_run, workers=1)
    if change == 'code':
        (batch.root / 'scripts/run_adaptive_twenty.py').write_text('# changed after lock\n')
    elif change == 'paper_contents':
        batch.paper.write_text('changed manuscript')
    elif change == 'paper_file_set':
        (batch.paper.parent / 'new_unapproved_figure.svg').write_text('<svg/>')
    else:
        source_state = deepcopy(batch.source_state)
        source_state['status'] = 'paused'
        batch.document(batch.expanded_run / 'state.json', source_state)
    with pytest.raises(ValueError, match='[Cc]hang'):
        runner.load(batch.adaptive_run)
    assert batch.counters['fit'] == batch.counters['predict'] == 0


def test_source_cannot_label_evidence_flexible_as_cached_tail(batch):
    # Keep reference hashes valid so rejection tests the baseline identity,
    # rather than merely detecting that a JSON file was edited.
    protocol = deepcopy(batch.source_protocol)
    identity = next(u['computation_id'] for u in protocol['units'] if u['method'] == 'tail_balanced')
    protocol['computations'][identity]['method'] = 'current'
    reference = batch.document(batch.expanded_run / 'protocol.json', protocol)
    state = deepcopy(batch.source_state); state['protocol'] = reference
    batch.document(batch.expanded_run / 'state.json', state)
    with pytest.raises(ValueError, match='successful tail prediction'):
        batch.adaptive.prepare(batch.adaptive_run, workers=1)
    assert not batch.adaptive_run.exists()


def test_instances_preserve_old_runner_api_and_reject_other_directories(batch):
    old = (frozen.METHODS, frozen.VERSION, frozen.__file__, frozen.api.choose_formula,
           expanded.METHODS, expanded.VERSION, batch.expanded.api.choose_formula)
    runner = adaptive.create_runner(project_root=batch.root,
        engine_provider=lambda _: batch.native, controller_class=frozen.Controller)
    assert old == (frozen.METHODS, frozen.VERSION, frozen.__file__, frozen.api.choose_formula,
                   expanded.METHODS, expanded.VERSION, batch.expanded.api.choose_formula)
    assert runner.api is not frozen.api and runner.api is not batch.expanded.api and runner.api is not batch.adaptive.api
    assert runner.variant('current') == 'tail_balanced'
    assert batch.expanded.variant('current') == 'evidence_flexible'
    assert runner.own_run(runner.DEFAULT) == runner.DEFAULT
    for wrong in (batch.expanded_run, batch.run, runner.DEFAULT / 'adaptive_twenty_nested'):
        with pytest.raises(ValueError, match='dedicated'):
            runner.own_run(wrong)
    symlink = runner.DEFAULT.with_name('adaptive_twenty_symlink')
    symlink.symlink_to(runner.DEFAULT, target_is_directory=True)
    with pytest.raises(ValueError, match='dedicated'):
        runner.own_run(symlink)
    assert_source_unchanged(batch)


def test_formula_selection_uses_full_development_and_never_cv_or_target_scores():
    mappings = [{'variant': 'adaptive_exchange', 'construction_partition_method': formula,
        'eligible': True, 'decision': {'objective': objective},
        'formula_selection_score': cv, 'target_gmfe': target}
        for formula, objective, cv, target in [('PKSimStandard', 1., 100., 999.), ('RodgersRowland', 2., .001, 1.)]]
    original = deepcopy(mappings)
    assert adaptive.select_formula(mappings) is mappings[0]
    assert mappings == original
    for row in mappings:
        row.pop('formula_selection_score'); row.pop('target_gmfe')
    assert adaptive.select_formula(mappings) is mappings[0]
    mappings[0]['decision']['objective'] = float('nan')
    with pytest.raises(ValueError, match='Finite development'):
        adaptive.select_formula(mappings)
