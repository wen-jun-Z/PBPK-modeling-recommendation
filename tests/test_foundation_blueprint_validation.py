"""CL-only intervention, exact-cache reuse and observation-lock regressions."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import threading

import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_foundation_blueprint_validation.py'
SPEC = importlib.util.spec_from_file_location('foundation_blueprint_runner_test', SCRIPT)
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)


def prior_fixture():
    parameters = {'c': {'total_clearance': 2., 'fraction_unbound': .3, 'vss': 1.2,
                         'terminal_half_life': 8., 'mrt': 6.}}
    subject = {'normalized_identity': 'CCO', 'generic_scaffold': '',
               'human_training_overlap': False, 'target_structure_and_scaffold_excluded': True}
    old = {'parameters': deepcopy(parameters), 'targets': [{'compound_id': 'c', 'smiles': 'CCO'}],
        'units': {'total_clearance': 'mL/min/kg', 'fraction_unbound': 'fraction', 'vss': 'L/kg',
                  'terminal_half_life': 'h', 'mrt': 'h'},
        'audit': {'human_training_structures': 1047}, 'per_compound_audit': {'c': subject}}
    receipt = {'path': 'prior.json', 'sha256': 'p'*64}
    prior = {**deepcopy(old), 'version': r.PRIOR_VERSION, 'source_parameter_prior': receipt,
        'method_name': 'TabPFN v2 CL', 'algorithm_version': 'fixed-v2-1',
        'audit': {k: False for k in ('target_labels_used', 'target_observation_files_opened',
            'training_structure_overlap', 'training_generic_scaffold_overlap', 'clinical_results_used_for_selection')},
        'training_metadata': {'eligible_cl_labels': 1045, 'context_fit_cl_labels': 836, 'calibration_cl_labels': 209}}
    prior['audit']['eligible_human_training_structures'] = 1047
    prior['audit'].update(final_CL_context_count=836, CL_calibration_count=209, eligible_CL_labels=1045,
        human_parameter_training_indices={'total_clearance': list(range(836))})
    prior['parameters']['c']['total_clearance'] = 3.
    prior['result_sha256'] = r.base.training_digest(prior)
    return prior, old, receipt


def rehash(prior):
    prior['result_sha256'] = r.base.training_digest({k:v for k,v in prior.items() if k != 'result_sha256'})


def task_fixture(basis='predicted', renal=False, blocked=False):
    evidence = [{'parameter': 'total_clearance', 'compound_id': 'c', 'value': 2., 'unit': 'ml/min/kg',
        'basis': basis, 'source': {'path': 'old.json'}, 'context': {'matrix': 'plasma'},
        'prediction_method': {'name': 'frozen'}, 'notes': 'unchanged source'},
        {'parameter': 'fraction_unbound', 'value': .3, 'basis': 'predicted', 'unit': 'fraction'},
        {'parameter': 'elimination_scope', 'value': 'hepatic_and_renal' if renal else 'hepatic_only', 'basis': 'assumption'}]
    if renal: evidence.append({'parameter': 'renal_clearance', 'value': .5, 'basis': 'derived', 'unit': 'ml/min/kg'})
    request = {'compound_id': 'c', 'human': {'weight_kg': 70}, 'dosing': {'dose_mg': 20},
        'target': {'window_end_h': 24}, 'partition_method': 'PKSimStandard', 'evidence': evidence,
        'evidence_ledger': {'path': 'frozen_ledger.json'}}
    core = {'adapter_version': 'clinical-pbpk-comparison-1', 'design_version': 'learned-exchange-design-2',
        'parameter_prior': {'path': 'oldprior.json'}}
    sidecars = {'vss': [{'value': 1.2, 'basis': 'predicted'}], 'half': [{'value': 8., 'basis': 'predicted'}]}
    return {'key': r.retained.key_for(request, sidecars, core), 'case_id': 'case',
        'request': request, 'sidecars': sidecars, 'core': core,
        'construction_blocked': ['pKa representation unsupported'] if blocked else None}


def test_training_pool_is_not_context_fit_count():
    prior, old, receipt = prior_fixture()
    assert r.validate_prior(prior, old, receipt) is prior
    assert prior['training_metadata']['context_fit_cl_labels'] != 1047


@pytest.mark.parametrize('parameter', r.FIXED_PARAMETERS)
def test_other_four_priors_may_not_change(parameter):
    prior, old, receipt = prior_fixture(); prior['parameters']['c'][parameter] *= 1.01; rehash(prior)
    with pytest.raises(ValueError, match='Only predicted CL'): r.validate_prior(prior, old, receipt)


@pytest.mark.parametrize('flag', ['target_labels_used', 'target_observation_files_opened',
    'training_structure_overlap', 'training_generic_scaffold_overlap', 'clinical_results_used_for_selection'])
def test_each_training_isolation_audit_is_required(flag):
    prior, old, receipt = prior_fixture(); prior['audit'][flag] = True; rehash(prior)
    with pytest.raises(ValueError, match='Training-only'): r.validate_prior(prior, old, receipt)


def test_wrong_checksum_or_source_prior_rejected():
    prior, old, receipt = prior_fixture(); prior['parameters']['c']['total_clearance'] = 7
    with pytest.raises(ValueError, match='checksum'): r.validate_prior(prior, old, receipt)
    rehash(prior); prior['source_parameter_prior']['sha256'] = 'different'; rehash(prior)
    with pytest.raises(ValueError, match='bound explicitly'): r.validate_prior(prior, old, {'path': 'prior.json', 'sha256': 'p'*64})


def test_target_cohort_cannot_shrink_to_clinically_scored_subset():
    prior, old, receipt = prior_fixture(); prior['parameters'].pop('c'); rehash(prior)
    with pytest.raises(ValueError, match='complete frozen'): r.validate_prior(prior, old, receipt)


def test_only_predicted_cl_and_its_provenance_change():
    prior, _, receipt = prior_fixture(); original = task_fixture(); before = deepcopy(original)
    task, audit = r.replace_clearance(original, prior, receipt)
    assert original == before and audit['changed']
    assert task['sidecars'] == original['sidecars']
    assert task['request']['evidence'][1:] == original['request']['evidence'][1:]
    assert task['request']['evidence'][0]['value'] == 3.
    assert task['request']['evidence'][0]['source']['path'] == receipt['path']
    assert task['core']['design_version'] == original['core']['design_version']
    assert task['core']['foundation_clearance_prior'] == receipt
    assert r.retained.key_for(task['request'], task['sidecars'], task['core']) != original['key']


@pytest.mark.parametrize('basis', ['measured', 'derived'])
def test_independent_cl_and_renal_split_are_byte_equivalent(basis):
    prior, _, receipt = prior_fixture(); original = task_fixture(basis, renal=True)
    task, audit = r.replace_clearance(original, prior, receipt)
    assert task == original and not audit['changed']


def test_no_predicted_cl_replacement_through_renal_side_channel():
    prior, _, receipt = prior_fixture()
    with pytest.raises(ValueError, match='organ split'):
        r.replace_clearance(task_fixture(renal=True), prior, receipt)


def test_equal_cl_preserves_full_source_request_and_ledger():
    prior, _, receipt = prior_fixture(); prior['parameters']['c']['total_clearance'] = 2.
    original = task_fixture(); task, audit = r.replace_clearance(original, prior, receipt)
    assert task == original and not audit['changed'] and audit['before'] == audit['after']


def test_construction_blocker_stays_in_matrix_without_application():
    prior, _, receipt = prior_fixture(); original = task_fixture(blocked=True)
    task, audit = r.replace_clearance(original, prior, receipt)
    assert task == original and not audit['changed'] and 'blocker' in audit['reason']


def result_fixture(task):
    return {'status': 'failed', 'reason': 'scientific_failure', 'observations_used': False,
        'request_sha256': r.value.model._digest(task['request']),
        'experiment_fingerprint': task['key'], 'experiment_candidate': r.CANDIDATE,
        'native_solver_calls': 1, 'linear_trajectory_calls': 0, 'curve': None, 'endpoints': None}


def test_complete_bound_result_recovers_without_native(monkeypatch, tmp_path):
    task = task_fixture(); folder = tmp_path/'model'; folder.mkdir()
    result = result_fixture(task); r.save(folder/'result.json', result)
    monkeypatch.setattr(r.value.old, 'collect_refs', lambda *args: None)
    monkeypatch.setattr(r.value.old, 'ref', lambda path: {'path': str(path), 'sha256': 'receipt'})
    monkeypatch.setattr(r.base.execution.separate, '_executor', lambda *args: pytest.fail('Native must not repeat'))
    assert r.invoke(task, folder)['status'] == 'failed'


def test_infrastructure_failure_pauses_instead_of_becoming_scientific_failure(monkeypatch, tmp_path):
    task = task_fixture(); result = result_fixture(task); result['reason'] = 'native_source_failed'
    result['execution_ref'] = {'path': 'process.json'}
    monkeypatch.setattr(r.value.old, 'checked', lambda ref: {'process': {'returncode': 9}})
    monkeypatch.setattr(r.base.execution.separate, '_executor', lambda task: lambda *a, **kw: result)
    with pytest.raises(r.value.old.Pause, match='infrastructure'): r.invoke(task, tmp_path/'model')


@pytest.mark.parametrize('fault', ['digest', 'candidate', 'fingerprint'])
def test_other_scientific_inputs_cannot_recover(monkeypatch, tmp_path, fault):
    task = task_fixture(); folder = tmp_path/'model'; folder.mkdir(); result = result_fixture(task)
    field = {'digest': 'request_sha256', 'candidate': 'experiment_candidate', 'fingerprint': 'experiment_fingerprint'}[fault]
    result[field] = 'different'; r.save(folder/'result.json', result)
    with pytest.raises(ValueError, match='different inputs'): r.invoke(task, folder)


def test_report_refuses_before_targets_or_previous_scores_are_opened(monkeypatch, tmp_path):
    monkeypatch.setattr(r, 'own', lambda path: Path(path))
    r.save(tmp_path/'state.json', {'protocol': {'path': 'protocol'}, 'outcomes': {},
                                 'all_predictions_locked_before_scoring': False})
    monkeypatch.setattr(r.value.old, 'checked', lambda ref: {'tasks': {'not_finished': {}}})
    monkeypatch.setattr(r.value, 'verify', lambda protocol: None)
    monkeypatch.setattr(r.base, 'ClinicalStore', lambda *args: pytest.fail('Targets must stay sealed'))
    with pytest.raises(ValueError, match='Lock every'): r.report(tmp_path)


def test_expected_source_counts_with_synthetic_cl_shift_no_simulation():
    source = r.value.old.read(r.SOURCE/'protocol.json')
    source_prior = r.value.old.checked(source['parameter_prior'])
    prior = deepcopy(source_prior); prior['method_name'] = 'fixture'; prior['algorithm_version'] = 'test'
    for p in prior['parameters'].values(): p['total_clearance'] *= 1.01
    changed = blocked = retained = 0
    for original in source['tasks'].values():
        task, audit = r.replace_clearance(original, prior, {'path': 'unused_test_artifact.json', 'sha256': 't'*64})
        assert task['sidecars'] == original['sidecars']
        changed += audit['changed']; blocked += bool(task.get('construction_blocked')); retained += not audit['changed']
    assert (changed, blocked, retained) == (37, 2, 12)
    assert len(source['units']) == 155


def test_directory_budget_includes_prior_not_just_clinical(monkeypatch, tmp_path):
    monkeypatch.setattr(r, 'BATCH', tmp_path)
    monkeypatch.setattr(r.retained.Monitor, 'check', lambda self, force=False: None)
    (tmp_path/'prior').mkdir(); (tmp_path/'prior/model').write_bytes(b'0123456789')
    clinical = tmp_path/'clinical'; clinical.mkdir()
    monitor = r.Monitor(clinical, {'directory_limit_bytes': 8}, {'outcomes': {}})
    with pytest.raises(r.value.old.Pause, match='batch exceeds'): monitor.check(force=True)
    assert monitor.halted.is_set()


def test_prepare_creates_ledgers_and_locks_metadata_without_scores(monkeypatch, tmp_path):
    source = r.value.old.read(r.SOURCE/'protocol.json')
    prior = r.value.old.checked(source['parameter_prior']); prior = deepcopy(prior)
    prior['method_name'] = 'synthetic_test_only'; prior['algorithm_version'] = 'test-only'
    for row in prior['parameters'].values(): row['total_clearance'] *= 1.01
    prior['checkpoint_receipt'] = prior['model_receipts']['shared']
    prior['context_receipt'] = prior['model_receipts']['complementary']
    prior['uncertainty'] = {cid: {'lower': p['total_clearance']/3, 'median': p['total_clearance'],
        'upper': p['total_clearance']*3, 'unit': 'mL/min/kg', 'calibration': 'split_CQR',
        'nominal_coverage': .9, 'clinical_coverage_guaranteed': False} for cid, p in prior['parameters'].items()}
    destination = tmp_path/'isolated_prepare'
    monkeypatch.setattr(r, 'own', lambda path: Path(path))
    monkeypatch.setattr(r, 'checked_prior', lambda path, source: (prior, source['parameter_prior']))
    monkeypatch.setattr(r.value.model, 'assemble_evidence_blueprint', lambda *a: {'status': 'assembled'})
    original_checked = r.value.old.checked
    def checked(receipt):
        assert not receipt['path'].endswith('/results.json'), 'Preparation must not decode clinical scores'
        return original_checked(receipt)
    monkeypatch.setattr(r.value.old, 'checked', checked)
    result = r.prepare(destination)
    assert result['logical_units'] == 155 and result['new_native_calls_expected'] == 37
    assert len(list((destination/'inputs').glob('*.json'))) == 37
    protocol = r.value.old.read(destination/'protocol.json')
    assert protocol['unchanged_parameters'] == list(r.FIXED_PARAMETERS)
    actions = [v['action'] for v in protocol['uncertainty_decisions'].values()]
    assert actions.count('construction_blocked') == 2
    assert actions.count('preserve_independent_clearance') == 10
    assert actions.count('request_clearance_evidence') == 37
    assert len(protocol['stability_case_ids']) == 6
    assert r.value.old.read(destination/'state.json')['all_predictions_locked_before_scoring'] is False


def stability_fixture():
    cases = [{'case_id': name, 'compound_id': name, 'cluster_id': name} for name in ('a', 'outside')]
    rows = [{'case_id': name, 'condition': condition, 'curve_metrics': {'gmfe': gmfe},
        'prediction_drift': {'curve_geometric_fold_change': 2.}}
        for name in ('a', 'outside') for condition, gmfe in (('available', 2.), ('missing_clearance', 4.))]
    return {'units': rows}, {'cases': cases, 'stability_case_ids': ['a']}


def test_stability_group_is_frozen_and_not_chosen_from_best_scores():
    source, protocol = stability_fixture(); candidate = deepcopy(source)
    for row in candidate['units']:
        if row['condition'] == 'missing_clearance':
            row['curve_metrics']['gmfe'] = 3. if row['case_id'] == 'a' else 400.
            row['prediction_drift']['curve_geometric_fold_change'] = 1.8 if row['case_id'] == 'a' else 100.
    stats = r.stability_statistics(candidate, source, protocol)
    assert stats['passed'] and stats['case_ids'] == ['a']
    assert stats['candidate']['missing_over_available_error']['estimate'] == pytest.approx(1.5)


def test_better_error_does_not_hide_increased_prediction_drift():
    source, protocol = stability_fixture(); candidate = deepcopy(source)
    row = candidate['units'][1]; row['curve_metrics']['gmfe'] = 3.
    row['prediction_drift']['curve_geometric_fold_change'] = 2.1
    stats = r.stability_statistics(candidate, source, protocol)
    assert not stats['passed'] and stats['checks']['missing_over_available_error_not_increased']
    assert not stats['checks']['prediction_drift_not_increased']
