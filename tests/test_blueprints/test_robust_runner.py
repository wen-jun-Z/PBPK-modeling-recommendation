"""Protocol, cache, isolation and recovery checks; no native model is executed."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_robust_blueprint as runner
import report_blueprint_validation as scoring


@pytest.fixture
def batch(tmp_path, monkeypatch):
    root = tmp_path
    group = root / 'experiments/blueprint_design'
    base, cap, run = (group / name for name in ('baseline', 'capability', 'robust_structure_fixture'))
    for directory in (base, cap): directory.mkdir(parents=True)
    api = runner.previous.transport_runner.create_runner(project_root=root)
    monkeypatch.setattr(runner, 'api', api)
    for name, value in [('ROOT', root), ('BASE', base), ('CAPABILITY', cap), ('DEFAULT', run)]:
        monkeypatch.setattr(runner, name, value)
    for module in (runner, runner.inputs):
        for name in ('read', 'ref', 'checked', 'verify'):
            monkeypatch.setattr(module, name, getattr(api, name))
    names = ('scripts/run_robust_blueprint.py', 'scripts/run_blueprint_capability.py',
        'scripts/run_blueprint_evidence_policy.py', 'scripts/run_blueprint_learning.py',
        'scripts/run_blueprint_transport.py', 'scripts/report_blueprint_validation.py', 'engine.py')
    for name in names:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# fixture\n')
    paper = root / 'results/manuscript/main.tex'; paper.parent.mkdir(parents=True); paper.write_text('frozen paper')
    native = SimpleNamespace(VERSION='fixture-engine', CONFIG={'starts': 16},
        _code_pins=lambda _: {'engine.py': runner.sha256(root / 'engine.py')})
    monkeypatch.setattr(runner, 'engine', lambda _: native)
    def document(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        runner.save(path, value); return api.ref(path)
    source = document(base / 'visible_source.json', {'kind': 'input evidence'})
    source.setdefault('location', 'fixture evidence')
    outcome = document(base / 'DO_NOT_DECODE_TARGET.json', {'target_values': [9123, 8123]})
    cases, computations, cap_units, cached = [], {}, [], {}
    for i in range(10):
        cid = f'drug{i}'
        prediction_method = {'name': 'fixture RF', 'version': '1',
            'training_overlap': 'excluded', 'applicability': 'fixture only'}
        def evidence(parameter, value, unit, basis):
            return {'parameter': parameter, 'compound_id': cid, 'value': value, 'unit': unit,
                'basis': basis, 'source': source, 'context': {'species': 'human'},
                **({'prediction_method': prediction_method} if basis == 'predicted' else {})}
        rows = [evidence('molecular_weight', 300., 'g/mol', 'measured'),
            evidence('molecular_formula', 'C2H6', 'text', 'derived'),
            evidence('fraction_unbound', .5 if i < 4 else .12 + i / 100, 'fraction', 'measured' if i < 4 else 'predicted'),
            evidence('total_clearance', 4. if i < 3 else 2., 'ml/min/kg', 'measured' if i < 3 else 'predicted')]
        if i < 2: rows.append(evidence('renal_clearance', .5, 'ml/min/kg', 'measured'))
        request = {'entry_reason': 'model_missing', 'compound_id': cid, 'compound_name': cid,
            'human': {'age_years': 30, 'weight_kg': 70, 'height_cm': 175, 'sex': 'MALE', 'population': 'European'},
            'dosing': {'route': 'intravenous', 'dose_mg': 100, 'infusion_minutes': 30},
            'target': {'endpoints': ['plasma_concentration_time'], 'primary_endpoint': 'plasma_concentration_time',
                       'window_start_h': 0, 'window_end_h': 4},
            'partition_method': 'PKSimStandard', 'evidence': rows, 'evidence_ledger': source,
            'forbidden_sources': []}
        target = deepcopy(request); target['human']['weight_kg'] = 80; target['dosing']['dose_mg'] = 200
        rf = deepcopy(request)
        for row in rf['evidence']:
            if row['parameter'] == 'fraction_unbound':
                row.update(value=.12 + i / 100, basis='predicted', prediction_method=prediction_method)
            if row['parameter'] == 'total_clearance':
                row.update(value=2., basis='predicted', prediction_method=prediction_method)
        training = {'compound_id': cid, 'study_id': cid + '_development', 'human': request['human'],
            'dosing': request['dosing'], 'observations': [
                {'time_min': 10 * j, 'concentration_umol_per_l': 1 / j, 'target_label': 999}
                for j in range(1, 7)], 'hidden_reference_metadata': {'target_label': 999}}
        training_ref = document(base / cid / 'training.json', training)
        case = {'case_id': cid, 'drug': cid, 'study_id': cid + '_target', 'cluster_id': cid,
            'compound_id': cid, 'reference_source': outcome, 'nca_policy': {'sampling_time_h': [.25, .5, 1.]},
            'requests': {'evidence': {'development': document(base / cid / 'dev.json', request),
                                    'target': document(base / cid / 'target.json', target)},
                         'rf': {'development': document(base / cid / 'rf.json', rf)}}}
        cases.append(case)
        computations[cid + '__full__flow_exchange'] = {'training_source': training_ref}
        for scenario in ('available', 'missing_fu'):
            construction = deepcopy(request)
            if scenario == 'missing_fu':
                construction['evidence'] = [r for r in construction['evidence'] if r['parameter'] != 'fraction_unbound']
                construction['evidence'].append(deepcopy(next(r for r in rf['evidence'] if r['parameter'] == 'fraction_unbound')))
            prediction_request = deepcopy(construction)
            prediction_request['human'] = target['human']; prediction_request['dosing'] = target['dosing']
            folder = cap / 'old' / cid / scenario
            document(folder / 'input.json', {'request': construction})
            mapping = document(folder / 'mapping.json', {'version': native.VERSION, 'training_source': training_ref})
            pred = document(folder / 'prediction.json', {'time_min': [0, 30, 60, 240],
                'concentration': [0, 2, 1, .1], 'unit': 'umol/l', 'time_unit': 'min'})
            identity = cid + '__' + scenario
            cap_units.append({'case_id': cid, 'scenario': scenario, 'method': 'evidence_policy',
                              'sampling': 'full', 'computation_id': identity})
            cached[identity] = {'status': 'predicted', 'mapping': mapping, 'prediction': pred,
                'request': document(folder / 'request.json', prediction_request),
                'resources': {'total_native_calls': 3, 'linear_solver_calls': 10, 'wall_seconds': 1}}
    original = {'cases': cases, 'computations': computations, 'source_pins': {}}
    original_ref = document(base / 'protocol.json', original)
    cap_ref = document(cap / 'protocol.json', {'cases': cases, 'units': cap_units, 'source_pins': {}})
    document(base / 'state.json', {'status': 'complete', 'protocol': original_ref})
    document(cap / 'state.json', {'status': 'complete', 'protocol': cap_ref, 'computations': cached})
    return SimpleNamespace(root=root, run=run, base=base, cap=cap, cases=cases, api=api, native=native,
                           outcome=outcome, document=document, paper=paper)


def requests(batch):
    case = batch.cases[0]
    base = runner.checked(case['requests']['evidence']['development'])
    target = runner.checked(case['requests']['evidence']['target'])
    rf = runner.checked(case['requests']['rf']['development'])
    train = runner.read(batch.base / 'drug0/training.json')
    return base, target, rf, train


def test_prepare_uses_visible_inputs_and_exact_old_controls(batch, monkeypatch):
    original = runner.read
    opened = []
    def guarded(path):
        actual = runner.inside(path, batch.root, exists=True); opened.append(actual)
        assert actual != batch.root / batch.outcome['path'], 'Target observations reached preparation'
        return original(actual)
    monkeypatch.setattr(runner, 'read', guarded); monkeypatch.setattr(batch.api, 'read', guarded)
    result = runner.prepare(batch.run, workers=1)
    assert result['logical_units'] == 120
    assert result['cached_predictions'] == 14
    assert result['changed_drugs'] == {'available': 0, 'missing_fu': 4, 'missing_cl': 3, 'missing_both': 4}
    protocol, state = runner.load(batch.run)
    assert len(protocol['units']) == len({u['id'] for u in protocol['units']}) == 120
    assert all(v['resources']['total_native_calls'] == 0 and v['original_resources']['total_native_calls'] == 3
               for v in state['computations'].values())
    for spec in protocol['computations'].values():
        request = runner.checked(spec['request']); training = runner.checked(spec['training_source'])
        assert 'hidden_reference_metadata' not in request and 'hidden_reference_metadata' not in training
        assert all(set(r) == {'time_min', 'concentration_umol_per_l'} for r in training['observations'])
        assert request['evidence_ledger']['path'].startswith(str(batch.run.relative_to(batch.root)))
    assert batch.root / batch.outcome['path'] not in opened


def test_scientific_cache_ignores_only_provenance_and_separates_training_methods(batch):
    base, target, _, training = requests(batch)
    identifier = runner.signature(base, target, training, 'current')
    variant = deepcopy(base); variant['evidence'].reverse()
    for row in variant['evidence']:
        row['source'] = {'path': 'other.json', 'sha256': 'f' * 64, 'location': 'alternative provenance'}
    assert runner.signature(variant, target, training, 'current') == identifier
    assert runner.signature(base, target, training, 'balanced_exchange') != identifier
    fewer = deepcopy(training); fewer['observations'].pop()
    assert runner.signature(base, target, fewer, 'current') != identifier
    other_dose = deepcopy(target); other_dose['dosing']['dose_mg'] *= 2
    assert runner.signature(base, other_dose, training, 'current') != identifier
    changed_drug = deepcopy(target); changed_drug['evidence'][2]['value'] /= 2
    with pytest.raises(ValueError, match='[Tt]arget.*evidence'):
        runner.signature(base, changed_drug, training, 'current')


def test_integer_float_request_equivalence_reuses_scientific_cache(batch):
    base, target, _, training = requests(batch)
    original = runner.signature(base, target, training, 'current')
    equivalent_base, equivalent_target = deepcopy(base), deepcopy(target)
    for request in (equivalent_base, equivalent_target):
        for section, fields in [('human', ('age_years', 'weight_kg', 'height_cm')),
                                ('dosing', ('dose_mg', 'infusion_minutes')),
                                ('target', ('window_start_h', 'window_end_h'))]:
            for field in fields:
                request[section][field] = float(request[section][field])
    assert runner.signature(equivalent_base, equivalent_target, training, 'current') == original


def test_missing_clearance_fills_rf_but_preserves_renal_and_original_inputs(batch):
    base, _, rf, _ = requests(batch)
    rows = [deepcopy(r) for r in base['evidence'] if r['parameter'] != 'total_clearance']
    before = deepcopy(rows)
    selected, actions = runner.fill_visible(rows, rf['evidence'])
    assert rows == before
    values = {r['parameter']: r for r in selected}
    assert values['total_clearance']['value'] == 2
    assert values['total_clearance']['basis'] == 'predicted'
    assert values['renal_clearance'] == next(r for r in before if r['parameter'] == 'renal_clearance')
    assert actions == [{'parameter': 'total_clearance', 'action': 'rf_fallback'}]


@pytest.mark.parametrize('fault', ['nan_cl', 'inf_cl', 'renal_exceeds_total', 'duplicate_renal', 'wrong_unit'])
def test_invalid_clearance_cannot_be_frozen(batch, fault):
    base, _, rf, _ = requests(batch); rows = deepcopy(base['evidence'])
    total = next(r for r in rows if r['parameter'] == 'total_clearance')
    if fault == 'nan_cl': total['value'] = float('nan')
    elif fault == 'inf_cl': total['value'] = float('inf')
    elif fault == 'wrong_unit': total['unit'] = 'L/h'
    elif fault == 'renal_exceeds_total': total['value'] = .1
    else: rows.append(deepcopy(next(r for r in rows if r['parameter'] == 'renal_clearance')))
    with pytest.raises(ValueError): runner.fill_visible(rows, rf['evidence'])


def test_paper_hash_change_blocks_resume_before_any_execution(batch):
    runner.prepare(batch.run, workers=1)
    batch.paper.write_text('concurrent paper change')
    with pytest.raises(ValueError, match='[Cc]hanged'):
        runner.load(batch.run)


def test_report_refuses_partial_batch_before_decoding_target(batch, monkeypatch):
    runner.prepare(batch.run, workers=1)
    calls = []
    monkeypatch.setattr(scoring, '_case_reference', lambda *a: calls.append(a))
    with pytest.raises(ValueError, match='lock'): runner.report(batch.run)
    assert calls == []


def install_fake_execution(batch, monkeypatch, *, interrupt=False):
    counters = Counter(); interrupted = []
    class Controller:
        def __init__(self, run, protocol, state):
            self.run, self.state = run, state
            self.lock = threading.RLock(); self.closed = threading.Event(); self.halted = threading.Event()
        def persist(self, **values):
            self.state.update(values); runner.save(self.run / 'state.json', self.state)
        def pause(self, reason):
            self.halted.set(); self.persist(status='paused', pause_reason=str(reason))
        def check(self, force=False):
            if self.halted.is_set(): raise runner.Paused('fixture interrupted')
        def watch(self): self.closed.wait(5)
    monkeypatch.setattr(runner, 'Controller', Controller)
    def fit_mapping(**kw):
        counters['fit'] += 1
        out = kw['output_dir']; request = kw['reference_request']
        batch.document(out / 'input.json', {'request': request})
        mapping = {'version': batch.native.VERSION, 'variant': kw['variant'], 'status': 'fitted', 'eligible': True,
            'construction_partition_method': request['partition_method'], 'decision': {'objective': 1.},
            'training_source': kw['training_source'], 'total_native_calls': 1, 'actual_solver_calls': 1,
            'linear_solver_calls': 1, 'failed_evaluation_count': 0, 'elapsed_seconds': .01}
        return {**mapping, 'mapping': batch.document(out / 'mapping.json', mapping)}
    def predict_mapping(**kw):
        counters['predict'] += 1; out = kw['output_dir']; out.mkdir(parents=True, exist_ok=True)
        if interrupt and not interrupted:
            interrupted.append(out); (out / 'incomplete.log').write_text('interrupt')
            raise RuntimeError('simulated power loss after partial prediction')
        batch.document(out / 'request.json', kw['request'])
        batch.document(out / 'prediction_locked.json', {'time_min': [0, 30, 60, 240],
            'concentration': [0, 2, 1, .1], 'unit': 'umol/l', 'time_unit': 'min'})
        batch.document(out / 'endpoints.json', {'endpoints': {'auc': {'value': 999}}})
        batch.document(out / 'completion.json', {'sealed': True})
        batch.document(out / 'result.json', {'actual_new_calls': 1, 'linear_solver_calls': 1, 'elapsed_seconds': .01})
        return {'status': 'predicted'}
    batch.native.fit_mapping = fit_mapping; batch.native.predict_mapping = predict_mapping
    def locked_report(run):
        protocol, state = runner.load(run, locked=True)
        assert state['monitor_stopped_before_scoring'] and state['formulas_locked_before_target_prediction']
        result = {'units': protocol['units'], 'source_pins': {}}
        runner.save(run / 'results.json', result); counters['report'] += 1
        return result
    monkeypatch.setattr(runner, 'report', locked_report)
    return counters, interrupted


def test_resume_preserves_completed_fits_and_uses_explicit_prediction_retry(batch, monkeypatch):
    result = runner.prepare(batch.run, workers=1)
    counters, interrupted = install_fake_execution(batch, monkeypatch, interrupt=True)
    with pytest.raises(runner.Paused): runner.execute(batch.run)
    assert counters['fit'] == result['unique_new_fits'] and counters['report'] == 0
    assert (interrupted[0] / 'incomplete.log').exists()
    runner.execute(batch.run)
    assert counters['fit'] == result['unique_new_fits']
    assert counters['predict'] == result['new_predictions'] + 1
    assert (interrupted[0].with_name('target_retry_1') / 'completion.json').exists()
    assert (interrupted[0] / 'incomplete.log').exists()
    assert counters['report'] == 1


def test_a_nonterminal_checkpoint_cannot_be_scored_as_completed(batch, monkeypatch):
    runner.prepare(batch.run, workers=1); install_fake_execution(batch, monkeypatch)
    runner.execute(batch.run)
    state = runner.read(batch.run / 'state.json'); cid = next(iter(state['computations']))
    state['status'] = 'predictions_locked'; state['computations'][cid]['status'] = 'running'
    runner.save(batch.run / 'state.json', state)
    with pytest.raises(ValueError): runner.load(batch.run, locked=True)


def test_candidate_failure_pairs_cannot_pass_nonregression_or_disappear():
    summary = [{'scenario': s, 'method': m, 'scored': 10, 'failed': 0,
        'curve_gmfe': {'estimate': 1.}} for s in runner.SCENARIOS for m in runner.METHODS]
    summary[0].update(scored=0, failed=10, curve_gmfe={'estimate': None})
    records = []
    for m in runner.METHODS:
        records.append({'case_id': 'a', 'drug': 'a', 'scenario': 'available', 'method': m,
            'curve_metrics': {} if m == 'current' else {'gmfe': 1.}, 'endpoints': {}})
    result = runner.candidate_checks(records, summary)
    for value in result.values():
        assert not value['all_four_aggregate_curve_errors_nonincreasing']
        assert not value['automatic_promotion']
        assert value['incomplete_pairs']


def test_candidate_checks_keep_scalar_and_tmax_regressions():
    summary = [{'scenario': s, 'method': m, 'scored': 10, 'failed': 0,
        'curve_gmfe': {'estimate': 1.}} for s in runner.SCENARIOS for m in runner.METHODS]
    records = [{'case_id': 'a', 'drug': 'a', 'scenario': 'available', 'method': m,
        'curve_metrics': {'gmfe': 1.}, 'endpoints': {
            'auc': {'value': 5, 'reference': 5, 'error': 1 if m == 'current' else 2, 'error_metric': 'fold_error'},
            'tmax': {'value': 0, 'reference': 0, 'error': 0 if m == 'current' else .5, 'error_metric': 'absolute_error_h'}}}
        for m in runner.METHODS]
    result = runner.candidate_checks(records, summary)
    for value in result.values():
        assert {r['endpoint'] for r in value['endpoint_error_regressions']} == {'auc', 'tmax'}


def test_report_uses_matched_values_and_retains_zero_tmax_error(batch, monkeypatch):
    original_report = runner.report
    runner.prepare(batch.run, workers=1); install_fake_execution(batch, monkeypatch)
    runner.execute(batch.run)
    state = runner.read(batch.run / 'state.json'); state['status'] = 'predictions_locked'
    runner.save(batch.run / 'state.json', state)
    monkeypatch.setattr(runner, 'report', original_report)
    monkeypatch.setattr(scoring, '_case_reference', lambda *_: ({}, {}, {}))
    def score(case, clinical, curve, endpoints, computation, store):
        return {'curve_metrics': {'gmfe': 1.5, 'ln_rmse': .3, 'n': 3}, 'endpoints': {
            'auc': {'value': 7, 'reference': 8, 'error': 8 / 7, 'error_metric': 'fold_error', 'weak_tail': False},
            'tmax': {'value': 0, 'unit': 'h', 'reference': 0, 'error': 0, 'error_metric': 'absolute_error_h', 'weak_tail': False}}}, {}, 300
    monkeypatch.setattr(scoring, '_score_prediction', score)
    result = runner.report(batch.run)
    assert len(result['units']) == 120
    assert all(r['endpoints']['auc']['value'] == 7 for r in result['units'])
    assert all(r['endpoint_errors']['tmax']['n'] == 10 and r['endpoint_errors']['tmax']['mean_absolute_error_h'] == 0
               for r in result['summary'])
    assert all(r['endpoint_prediction_drift']['auc'] == 1 for r in result['stability'])
    assert all(r['tmax_prediction_shift_h'] == 0 and 'tmax' not in r['endpoint_prediction_drift']
               for r in result['stability'])
    assert batch.paper.read_text() == 'frozen paper'
