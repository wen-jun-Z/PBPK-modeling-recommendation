"""Publication locks, provenance and missing-label semantics; no model execution."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
paper = importlib.import_module('render_blueprint_architecture_paper')


@pytest.fixture
def matrix():
    methods = ['extent_exchange_clearance', 'flow_exchange', 'hepatic_exchange',
               'two_compartment', 'two_compartment_prior']
    cases = [{'case_id': d.lower(), 'drug': d, 'study_id': d + '_target', 'cluster_id': d}
             for d in paper.DRUGS]
    protocol = {'version': 'synthetic-hepatic-1', 'cases': cases, 'methods': methods,
                'sampling': ['full', 'five'], 'formulas': ['PKSimStandard', 'RodgersRowland'],
                'units': [], 'computations': {}}
    state = {'version': protocol['version'], 'status': 'complete', 'fits': {}, 'computations': {},
             'formulas_locked_before_target_prediction': True,
             'all_attempts_locked_before_scoring': True, 'monitor_stopped_before_scoring': True}
    result = {'version': protocol['version'], 'all_attempts_locked_before_scoring': True, 'units': []}
    selection = {'basis': 'development_objective_only', 'choices': {}}
    for case in cases:
        for sampling in paper.SAMPLING:
            for method in methods:
                uid = '__'.join((case['case_id'], sampling, method))
                ref = {'path': uid + '/prediction.json', 'sha256': 'a' * 64}
                unit = {'id': uid, 'case_id': case['case_id'], 'sampling': sampling, 'method': method}
                protocol['units'].append(unit)
                state['computations'][uid] = {'status': 'predicted', 'prediction': ref}
                record = dict(case, unit_id=uid, sampling=sampling, method=method,
                              attempt_status='predicted', prediction_source=ref)
                result['units'].append(record)
                if method == 'hepatic_exchange':
                    protocol['computations'][uid] = unit
                    for formula in protocol['formulas']:
                        mapping = {'path': uid + '/' + formula + '/mapping.json', 'sha256': 'b' * 64}
                        state['fits'][uid + '__' + formula] = {'mapping': mapping}
                    selection['choices'][uid] = {'formula': formula, 'mapping': mapping}
                    state['computations'][uid]['mapping'] = mapping
    return SimpleNamespace(protocol=protocol, state=state, result=result, selection=selection)


def test_complete_matrix_supports_explicit_new_candidate(matrix):
    cases, units = paper.validate_matrix(matrix.protocol, matrix.state, matrix.result, matrix.selection)
    assert len(cases) == 4 and len(units) == 40


def test_adopted_method_display_does_not_inherit_later_unselected_methods(matrix):
    matrix.protocol['methods'].append('residence_exchange')
    assert paper.display_methods(matrix.protocol, 'flow_exchange') == (
        'extent_exchange_clearance', 'flow_exchange', 'two_compartment', 'two_compartment_prior')


def test_publication_still_requires_every_declared_control(matrix):
    matrix.protocol['methods'].remove('two_compartment')
    with pytest.raises(ValueError, match='required frozen comparator'):
        paper.display_methods(matrix.protocol, 'flow_exchange')


def test_baseline_display_requires_explicit_adoption_and_declared_method(matrix):
    _, units = paper.validate_matrix(matrix.protocol, matrix.state, matrix.result, matrix.selection)
    with pytest.raises(ValueError, match='declared new experiment'):
        paper.validate_display_role(matrix.protocol, matrix.state, units, None,
                                    'flow_exchange', 'new_candidate')
    with pytest.raises(ValueError, match='explicitly retained finite-exchange'):
        paper.validate_display_role(matrix.protocol, matrix.state, units, None,
                                    'two_compartment', 'adopted_baseline')
    assert paper.validate_display_role(matrix.protocol, matrix.state, units, None,
                                       'hepatic_exchange', 'new_candidate') is None


@pytest.mark.parametrize('damage', ['duplicate_result', 'missing_result', 'wrong_study',
                                  'changed_prediction', 'missing_fit', 'different_formula_mapping',
                                  'target_selection', 'different_version'])
def test_rejects_material_provenance_and_matrix_errors(matrix, damage):
    if damage == 'duplicate_result':
        matrix.result['units'].append(deepcopy(matrix.result['units'][0]))
    elif damage == 'missing_result':
        matrix.result['units'].pop()
    elif damage == 'wrong_study':
        matrix.result['units'][0]['study_id'] = 'another_target'
    elif damage == 'changed_prediction':
        matrix.result['units'][0]['prediction_source'] = {'path': 'different.json', 'sha256': 'f' * 64}
    elif damage == 'missing_fit':
        matrix.state['fits'].pop(next(iter(matrix.state['fits'])))
    elif damage == 'different_formula_mapping':
        choice = next(iter(matrix.selection['choices'].values()))
        choice['mapping'] = {'path': 'different_mapping.json', 'sha256': 'c' * 64}
    elif damage == 'target_selection':
        matrix.selection['basis'] = 'target_gmfe'
    else:
        matrix.result['version'] = 'another-batch'
    with pytest.raises(ValueError):
        paper.validate_matrix(matrix.protocol, matrix.state, matrix.result, matrix.selection)


@pytest.mark.parametrize('missing', ['status', 'formulas_locked_before_target_prediction',
                                   'all_attempts_locked_before_scoring', 'monitor_stopped_before_scoring'])
def test_lock_gates_run_before_any_observation_read(tmp_path, matrix, missing, monkeypatch):
    run = tmp_path / 'run'
    run.mkdir()
    state = matrix.state
    state[missing] = 'running' if missing == 'status' else False
    (run / 'state.json').write_text(json.dumps(state))
    monkeypatch.setattr(paper, '_case_reference', lambda *_: pytest.fail('Target observations decoded early'))
    with pytest.raises(ValueError, match='pre-scoring lock'):
        paper.load_data(run, root=tmp_path)


def test_nested_source_pin_mismatch_is_fatal_without_decoding_source(tmp_path):
    source = tmp_path / 'opaque.json'
    source.write_text('not JSON, because hashing must not decode this artifact')
    store = paper.Store(tmp_path)
    reference = {'path': 'opaque.json', 'sha256': paper.digest(source)}
    paper.pinned_references({'nested': [reference]}, store)
    source.write_text('changed')
    with pytest.raises(ValueError, match='lock mismatch'):
        paper.pinned_references({'source_pins': {'opaque.json': reference['sha256']}}, store)


def test_rejects_sources_outside_project(tmp_path):
    with pytest.raises(ValueError, match='outside the project'):
        paper.pinned_references({'path': '../elsewhere.json', 'sha256': 'f' * 64}, paper.Store(tmp_path))


def test_table_keeps_missing_clast_reference_and_weak_tail_without_imputation(matrix):
    cases = {c['case_id']: c for c in matrix.protocol['cases']}
    lookup, references = {}, {}
    for cid, case in cases.items():
        refs = {e: {'value': None if e == 'c_last' else 2, 'unit': 'h' if e == 'tmax' else 'ng/mL'}
                for e in paper.ENDPOINTS}
        references[cid] = ({}, {'time_h': [0, 1, 2]}, refs)
        for sampling in paper.SAMPLING:
            for method in ('hepatic_exchange', 'two_compartment'):
                estimates = {e: {'value': 4, 'error': None if e == 'c_last' else 2, 'unit': ref['unit'],
                                 'weak_tail': e == 'terminal_half_life'} for e, ref in refs.items()}
                lookup[cid, sampling, method] = {'endpoints': estimates, 'curve_metrics': {'gmfe': 1.5}}
    data = SimpleNamespace(cases=cases, lookup=lookup, references=references, candidate='hepatic_exchange',
                           labels={'hepatic_exchange': 'Evidence-routed PBPK'})
    text = paper.table_text(data)
    clast = [line for line in text.splitlines() if line.startswith(r'$C_{\mathrm{last}}$')]
    assert len(clast) == 8
    assert all(line.count(r'\textemdash{}') == 3 for line in clast)  # O and both errors; P* remains explicit.
    assert text.count(r'4^{\dagger}') == 16
    assert 'not a long-horizon native PBPK endpoint' in text
    assert 'same frozen PK evidence' in text
    assert r'T_{\max}' in text and 'absolute error in the stated time unit' in text


def test_numeric_checks_do_not_accept_changed_missing_labels_or_nan():
    with pytest.raises(ValueError):
        paper.same_number(None, 1, 'missing O')
    with pytest.raises(ValueError):
        paper.same_number(float('nan'), float('nan'), 'NaN')
    with pytest.raises(ValueError):
        paper.same_number(1, 1.01, 'changed score')


def test_profile_axes_use_target_window_not_extended_numerical_tail():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases, references, curves, lookup, summaries = {}, {}, {}, {}, {}
    methods = tuple(paper.LABELS)
    for drug in paper.DRUGS:
        cases[drug] = {'drug': drug, 'target_request_metadata': {'window_end_h': 10,
                       'dosing': {'dose_mg': 100, 'infusion_minutes': 30}}}
        references[drug] = ({}, {'unit': 'ng/mL', 'time_h': [1, 10], 'concentration': [100, 1]}, {})
        for sampling in paper.SAMPLING:
            curves[drug, sampling, 'flow_exchange'] = {'time_h': [0, 1, 10, 1344],
                                                       'concentration': [0, 100, 1, 1e-180]}
            for method in methods:
                lookup[drug, sampling, method] = {'curve_metrics': {'gmfe': 1.5}}
                summaries[sampling, method] = 1.5
    data = SimpleNamespace(cases=cases, references=references, curves=curves, lookup=lookup,
                           summaries=summaries, methods=methods, labels=paper.LABELS, candidate='flow_exchange')
    fig = paper.figure(data)
    try:
        for ax in fig.axes[:4]:
            assert all(max(line.get_xdata()) <= 10 for line in ax.lines)
            assert ax.get_ylim()[0] > .1
            assert ax.get_xlim()[1] > 10  # Last observed marker must not be cut in half.
    finally:
        plt.close(fig)
