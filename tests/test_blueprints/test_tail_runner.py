"""Independent runner bindings and cache lifecycle; no native execution."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_tail_blueprint as tail
import run_robust_blueprint as frozen
import report_blueprint_validation as scoring
from test_robust_runner import batch as base_batch, install_fake_execution


@pytest.fixture
def batch(base_batch, monkeypatch):
    base = base_batch
    frozen.prepare(base.run, workers=1)
    counters, _ = install_fake_execution(base, monkeypatch)
    old_fit = base.native.fit_mapping
    def fitting(**kwargs):
        out = old_fit(**kwargs)
        path = base.root / out['mapping']['path']
        document = json.loads(path.read_text())
        document.update(source_pins={}, code_pins={}, retained_artifact_pins={})
        if kwargs['variant'] == 'development_guarded':
            # Deliberately conflict: correct CV selection is RodgersRowland.
            document['formula_selection_score'] = 2. if kwargs['reference_request']['partition_method'] == 'PKSimStandard' else 1.
            document['decision']['objective'] = .01 if kwargs['reference_request']['partition_method'] == 'PKSimStandard' else 100.
            document['clearance_prior_policy'] = {'penalty_weight': .01}
        reference = base.document(path, document)
        return {**document, 'mapping': reference}
    base.native.fit_mapping = fitting
    base.native._validated = lambda *a: None
    base.native.CONFIG = {'starts': 16, 'cv_starts': 8, 'cv': {'folds': ['late', 'interior'],
        'rf_penalty_weights': [.01, .1], 'observed_penalty_weights': [.1]}}
    base.native.clearance_policy = lambda request: {'rule': 'qualified_rf' if
        next(x for x in request['evidence'] if x['parameter'] == 'total_clearance')['basis'] == 'predicted' else 'qualified_observed'}
    # The source protocol contract was captured before this test-only CONFIG
    # extension; restore it during the frozen mock execution.
    configured = deepcopy(base.native.CONFIG); base.native.CONFIG = {'starts': 16}
    frozen.execute(base.run)
    base.native.CONFIG = configured
    for name in tail.CODE_FILES:
        path = base.root / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# fixture\n')
    candidate = tail.create_runner(project_root=base.root, source_name=base.run.name,
        default_name='robust_structure_tail_fixture', engine_provider=lambda _: base.native,
        controller_class=frozen.Controller)
    base.tail = candidate; base.tail_run = candidate.DEFAULT; base.counters = counters
    return base


def test_instance_owns_configuration_and_does_not_mutate_frozen_module_or_api():
    before = frozen.METHODS, frozen.VERSION, frozen.__file__, frozen.api.choose_formula
    first, second = tail.create_runner(), tail.create_runner(default_name='robust_structure_tail_other')
    assert (frozen.METHODS, frozen.VERSION, frozen.__file__, frozen.api.choose_formula) == before
    assert first.api is not frozen.api and first.api is not second.api
    assert first.DEFAULT != second.DEFAULT
    assert first.variant('current') == 'evidence_flexible'
    assert first.__file__.endswith('/scripts/run_tail_blueprint.py')


def test_prepare_reuses_exact_evidence_flexible_controls_without_decoding_target(batch, monkeypatch):
    r = batch.tail; original_read = r.api.read
    def guarded(path):
        resolved = r.api.path(path) if hasattr(r.api, 'path') else (batch.root / path).resolve()
        assert resolved != batch.root / batch.outcome['path'], 'Clinical outcome decoded during preparation'
        return original_read(path)
    monkeypatch.setattr(r, 'read', guarded); monkeypatch.setattr(r.api, 'read', guarded)
    result = r.prepare(batch.tail_run, workers=1)
    assert {k: result[k] for k in ('unique_new_fits', 'new_predictions', 'cached_predictions', 'logical_units')} == {
        'unique_new_fits': 80, 'new_predictions': 40, 'cached_predictions': 20, 'logical_units': 120}
    protocol, state = r.load(batch.tail_run)
    assert protocol['optimization_budget'] == {'full_fit_stages': 80, 'cv_stages': 136,
        'full_optimizer_starts': 1280, 'cv_optimizer_starts': 1088, 'optimizer_starts': 2368}
    source = r.read(batch.run / 'state.json')
    for cid in protocol['cached_ids']:
        assert cid.startswith('evidence_flexible_')
        assert state['computations'][cid]['mapping'] == source['computations'][cid]['mapping']
        assert state['computations'][cid]['prediction'] == source['computations'][cid]['prediction']
        assert state['computations'][cid]['resources']['total_native_calls'] == 0
    assert all(item['method'] in ('tail_balanced', 'development_guarded') for item in protocol['computations'].values())


def test_guarded_selection_ignores_full_objective_and_has_no_missing_cv_fallback():
    r = tail.create_runner()
    mappings = [{'variant': 'development_guarded', 'construction_partition_method': name, 'eligible': True,
        'decision': {'objective': score}, 'formula_selection_score': cv, 'clearance_prior_policy': {'penalty_weight': .01}}
        for name, score, cv in [('PKSimStandard', .01, 3.), ('RodgersRowland', 100., .1)]]
    assert r.api.choose_formula(mappings)['construction_partition_method'] == 'RodgersRowland'
    mappings[0]['formula_selection_score'] = None
    with pytest.raises(ValueError, match='finite'):
        r.api.choose_formula(mappings)


def test_guarded_cross_formula_exact_cv_tie_prefers_stronger_selected_prior():
    r = tail.create_runner()
    mappings = [{'variant': 'development_guarded', 'construction_partition_method': name, 'eligible': True,
        'decision': {'objective': 0. if name == 'PKSimStandard' else 100.},
        'formula_selection_score': .2, 'clearance_prior_policy': {'penalty_weight': penalty}}
        for name, penalty in [('PKSimStandard', .01), ('RodgersRowland', .1)]]
    assert r.api.choose_formula(mappings)['construction_partition_method'] == 'RodgersRowland'


def test_report_complete_is_idempotent_and_guarded_prediction_uses_cv_formula(batch, monkeypatch):
    r = batch.tail; r.prepare(batch.tail_run, workers=1)
    monkeypatch.setattr(scoring, '_case_reference', lambda *a: (None, None, None))
    monkeypatch.setattr(scoring, '_score_prediction', lambda *a: ({'curve_metrics': {'gmfe': 1.2},
        'endpoints': {'auc': {'value': 2., 'reference': 2., 'error': 1., 'error_metric': 'fold_error'}}}, None, None))
    before_fits, before_predictions = batch.counters['fit'], batch.counters['predict']
    r.execute(batch.tail_run)
    assert batch.counters['fit'] - before_fits == 80
    assert batch.counters['predict'] - before_predictions == 40
    protocol, state = r.load(batch.tail_run, locked=True)
    for cid, spec in protocol['computations'].items():
        if spec['method'] == 'development_guarded':
            assert state['computations'][cid]['selected_formula'] == 'RodgersRowland'
    digest = r.sha256(batch.tail_run / 'results.json')
    result = r.report(batch.tail_run)
    assert result['baseline_semantics']['source_method'] == 'evidence_flexible'
    assert r.sha256(batch.tail_run / 'results.json') == digest == state['results']['sha256']
    r.execute(batch.tail_run)
    assert batch.counters['fit'] - before_fits == 80


def test_changed_cached_binding_or_added_paper_file_blocks_loading(batch):
    r = batch.tail; r.prepare(batch.tail_run, workers=1)
    protocol, state = r.load(batch.tail_run)
    identity = protocol['cached_ids'][0]
    state['computations'][identity]['source_method'] = 'balanced_exchange'
    r.save(batch.tail_run / 'state.json', state)
    with pytest.raises(ValueError, match='semantic'):
        r.load(batch.tail_run)
    state['computations'][identity]['source_method'] = 'evidence_flexible'; r.save(batch.tail_run / 'state.json', state)
    extra = batch.paper.parent / 'new_figure.svg'; extra.write_text('new paper artifact')
    with pytest.raises(ValueError, match='file set'):
        r.load(batch.tail_run)


def test_background_launches_new_adapter_not_frozen_script(batch, monkeypatch):
    r = batch.tail; r.prepare(batch.tail_run, workers=1)
    import psutil
    observed = []
    class Child:
        pid = 123456
    monkeypatch.setattr(r.subprocess, 'Popen', lambda args, **kwargs: observed.append(args) or Child())
    monkeypatch.setattr(psutil, 'Process', lambda pid: type('P', (), {'create_time': lambda self: 3.})())
    r.background(batch.tail_run)
    assert observed[0][2] == str(batch.root / 'scripts/run_tail_blueprint.py')
    assert observed[0][3] == 'run'


def test_status_exposes_cv_progress_before_a_full_mapping_finishes(batch):
    r = batch.tail; r.prepare(batch.tail_run, workers=1)
    protocol, _ = r.load(batch.tail_run)
    cid = next(k for k, v in protocol['computations'].items() if v['method'] == 'development_guarded')
    folder = batch.tail_run / 'work' / cid / 'PKSimStandard'
    batch.document(folder / 'input.json', {'variant': 'development_guarded'})
    batch.document(folder / 'progress.json', {'stages': {'cv_0.01_0': {'starts': [{}] * 8, 'status': 'complete'},
        'cv_0.01_1': {'starts': [{}] * 3}}, 'incomplete_start': {'stage': 'cv_0.01_1', 'index': 3}})
    status = r.status(batch.tail_run)
    assert status['fits_completed'] == 0
    progress = status['optimization_progress']
    assert progress['cv_stages_completed'] == 1 and progress['cv_optimizer_starts_completed'] == 11
    assert progress['planned']['optimizer_starts'] == 2368
    assert progress['active_stages'] == {'cv_0.01_1': 1}
