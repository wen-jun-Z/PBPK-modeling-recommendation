"""Own-model scope may defer comparisons, never missing model folds or labels."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('pbpk_match_v2_own_reporting', ROOT / 'scripts/summarize_pbpk_match_v2_own.py')
own = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(own)

# Reuse the existing artifact-producing fixture, including real receipt hashes
# and OOF reassembly checks, rather than mocking the authoritative input reader.
LEGACY_SPEC = importlib.util.spec_from_file_location('own_reporting_fixture', Path(__file__).with_name('test_pbpk_match_v2_reporting.py'))
legacy = importlib.util.module_from_spec(LEGACY_SPEC)
LEGACY_SPEC.loader.exec_module(legacy)
files = legacy.files


def available():
    external = dict.fromkeys(own.r.EXTERNAL)
    trained = {'full': dict.fromkeys(own.r.SEEDS)}
    pending = [f'{m}/seed{s}' for m in own.DEFERRED for s in own.r.SEEDS]
    return external, trained, pending


def test_only_explicitly_deferred_missing_runs_pass_gate():
    external, trained, pending = available()
    assert own.completion_gate(external, trained, pending) == sorted(pending)
    assert len(pending) == 12


@pytest.mark.parametrize('missing', ['seed', 'external', 'full_integrity', 'deferred_integrity', 'unknown'])
def test_scope_never_masks_required_or_corrupt_artifacts(missing):
    external, trained, pending = available()
    if missing == 'seed':
        trained['full'].pop(41)
    elif missing == 'external':
        external.pop('prorank')
    elif missing == 'full_integrity':
        pending.append('integrity/full/seed41/fold4/document_scores')
    elif missing == 'deferred_integrity':
        pending.append('integrity/mlp/seed17/fold0')
    else:
        pending.append('unregistered/seed17')
    with pytest.raises(RuntimeError):
        own.completion_gate(external, trained, pending)


def scope():
    return {'schema': 'pbpk-match-v2-execution-scope-1', 'mode': 'own_model_only',
        'original_protocol_sha256': 'original', 'active_methods': ['full'],
        'deferred_methods': list(own.DEFERRED), 'seeds': list(own.r.SEEDS),
        'outer_folds': 5, 'inner_folds': 3, 'expected_fit_jobs': 60,
        'external_methods': list(own.r.EXTERNAL)}


@pytest.mark.parametrize('field,value', [('mode', 'all'), ('active_methods', ['full', 'mlp']),
    ('seeds', [17]), ('outer_folds', 1), ('inner_folds', 1), ('expected_fit_jobs', 300),
    ('original_protocol_sha256', 'other'), ('external_methods', [])])
def test_scope_binds_original_protocol_and_unchanged_training(field, value):
    value_scope = scope()
    own.validate_scope(value_scope, 'original')
    value_scope[field] = value
    with pytest.raises(ValueError):
        own.validate_scope(value_scope, 'original')


def test_reused_calibration_does_not_read_held_out_labels_or_pk_errors():
    folds = [{'outer_train': [j for j in range(6) if j not in ix], 'outer_test': ix}
             for ix in ([0, 1], [2, 3], [4, 5])]
    data = {'groups': [f'g{i}' for i in range(6)], 'folds': folds,
        'applicability': np.tile([0, 0, 1, 2], (6, 1)),
        'error_records': [{'absolute_log_error': 2.0}]}
    scores = np.arange(24.).reshape(6, 4)
    probabilities, audit = own.r.external_probabilities(data, scores)
    altered = deepcopy(data)
    altered['applicability'][:2] = [2, 1, 0, 0]
    altered['error_records'] = [{'absolute_log_error': 999.0}]
    other, other_audit = own.r.external_probabilities(altered, scores)
    np.testing.assert_array_equal(probabilities[:2], other[:2])
    assert audit['folds'][0] == other_audit['folds'][0]
    assert all(not row['uses_clinical_errors'] for row in audit['folds'])


def test_statistics_family_is_explicit_and_excludes_deferred_models(monkeypatch):
    family = [*own.r.EXTERNAL, 'full', own.r.NO_ERROR]
    reports = {m: {'quality': {}, 'execution': {'position_rows': [{'rank': k} for k in (1, 2, 3)]}}
               for m in family}
    monkeypatch.setattr(own.r.ev, 'paired_quality', lambda *a, **k: {'computed': True})
    calls = []
    def paired(rows, methods, baseline):
        calls.append((len(rows), methods, baseline))
        return {'computed': True}
    monkeypatch.setattr(own.r.ev, 'paired_method_statistics', paired)
    result = own.paired(reports, {'query_records': [{'query_id': 'q'}], 'groups': ['g']})
    assert result['comparison_family'] == family
    assert {row['rank'] for row in result['execution']} == {1, 2, 3}
    assert len(calls) == 3
    assert all(methods == family for _, methods, _ in calls)
    assert not set(own.DEFERRED) & set(result['comparison_family'])


def test_full_own_report_keeps_seed_metrics_and_never_completes_original_protocol(files, monkeypatch):
    data, out = files
    project = out.parent
    scripts = project / 'scripts'
    scripts.mkdir()
    producer = scripts / 'summarize_pbpk_match_v2_own.py'
    producer.write_text('synthetic scoped report producer')
    monkeypatch.setattr(own, '__file__', str(producer))
    control = scripts / 'pbpk_match_v2_own_scope.py'
    control.write_text('synthetic own-only scope supervisor')
    contract = scope()
    contract['original_protocol_sha256'] = own.r.sha(out / 'protocol.json')
    contract['control_script_sha256'] = own.r.sha(control)
    scope_path = project / 'scope.json'
    own.r.write(scope_path, contract)
    result = own.summarize(data, out, scope_path=scope_path)
    assert result['state'] == 'complete_own_only'
    assert result['completed'] and not result['original_protocol_completed']
    assert set(result['single_seed_reports']['full']) == {'17', '23', '41'}
    assert set(result['reports']) == {'f2llm_v2_06b', 'full', 'no_error_branch'}
    assert {row['rank'] for row in result['reports']['full']['execution']['position_rows']} == {1, 2, 3}
    assert len(result['inference_integrity']['full']) == 9
    assert result['seed_mean_sd']['full']
    assert result['reports']['full']['quality']['risk_coverage']
    assert (out / 'report_own_only/summary.json').exists()
    assert (out / 'report_own_only/paper_execution.tex').exists()
    assert not (out / 'report').exists()
