"""Fu evidence may adjust only predicted CL; exact no-ops stay exact caches."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_fu_conditioned_blueprint_validation.py'
SPEC = importlib.util.spec_from_file_location('fu_conditioned_blueprint_runner_test', SCRIPT)
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)


def fixture(cl_basis='predicted', fu_basis='measured', blocked=False):
    request = {'compound_id': 'c', 'human': {'weight_kg': 70}, 'dosing': {'dose_mg': 50},
        'target': {'window_end_h': 24}, 'partition_method': 'PKSimStandard',
        'evidence_ledger': {'path': 'original.json'}, 'forbidden_sources': [],
        'evidence': [{'parameter': 'total_clearance', 'compound_id': 'c', 'basis': cl_basis,
                     'unit': 'ml/min/kg', 'value': 2., 'context': {'study_role': 'independent_parameters'}},
                    {'parameter': 'fraction_unbound', 'compound_id': 'c', 'basis': fu_basis,
                     'unit': 'fraction', 'value': .2, 'source': {'path': 'fu_source.json', 'sha256': 'e'*64},
                     'context': {'species': 'human', 'matrix': 'plasma', 'study_role': 'development',
                                 'underlying_study_overlap': 'unknown'}},
                    {'parameter': 'elimination_scope', 'value': 'hepatic_only', 'basis': 'assumption'}]}
    sidecars = {'vss': [{'parameter': 'vss', 'value': 1.2}], 'half': [{'parameter': 'terminal_half_life', 'value': 8.}]}
    core = {'design_version': 'learned-exchange-design-2', 'adapter_version': 'clinical-pbpk-comparison-1',
            'parameter_prior': {'path': 'baseline.json', 'sha256': 'b'*64}}
    task = {'case_id': 'case', 'request': request, 'sidecars': sidecars, 'core': core,
        'key': r.retained.key_for(request, sidecars, core), 'construction_blocked': ['pKa'] if blocked else None}
    prior = {'parameters': {'c': {'total_clearance': 2., 'fraction_unbound': .1}},
             'selection': {'kind': 'conditioned', 'beta': .5}}
    receipt = {'path': 'test_prior.json', 'sha256': 'a'*64}
    return task, prior, receipt


def update_function(raw=.2, beta=.5):
    def predict(root, artifact, cid, observed_fu, prior_fu, *, evidence_status):
        bounded = max(-math.log10(2), min(math.log10(2), raw)); applied = beta*bounded
        return {'factor': 10**applied, 'beta': beta, 'candidate': 'conditioned',
            'raw_delta_log10': raw, 'bounded_delta_log10': bounded, 'applied_delta_log10': applied,
            'observed_fu': observed_fu, 'prior_fu': prior_fu, 'uses_clinical_observations': False,
            'reason': 'training-only update', 'receipt': {'path': 'model', 'sha256': 'c'*64}}
    return predict


def test_shrinkage_is_separate_from_clipping_and_only_cl_changes():
    original, prior, receipt = fixture(); before = deepcopy(original)
    task, audit = r.transformed_task(original, prior, receipt, update_function(raw=.2, beta=.5))
    assert original == before and audit['changed'] and not audit['shift_clipped']
    assert audit['after'] == pytest.approx(2*10**.1)
    assert task['sidecars'] == original['sidecars']
    assert task['request']['evidence'][1:] == original['request']['evidence'][1:]
    assert task['request']['evidence_ledger'] == original['request']['evidence_ledger']
    assert task['core']['design_version'] == original['core']['design_version']


def test_genuine_delta_cap_is_counted_before_beta_shrinkage():
    task, prior, receipt = fixture()
    prior['selection']['beta'] = .25
    changed, audit = r.transformed_task(task, prior, receipt, update_function(raw=2., beta=.25))
    assert audit['shift_clipped']
    assert changed['request']['evidence'][0]['value'] == pytest.approx(2*2**.25)


@pytest.mark.parametrize('basis', ['measured', 'derived'])
def test_independent_cl_and_renal_split_are_never_updated(basis):
    task, prior, receipt = fixture(cl_basis=basis)
    task['request']['evidence'].append({'parameter': 'renal_clearance', 'value': .8, 'basis': 'measured'})
    changed, audit = r.transformed_task(task, prior, receipt, lambda *a, **kw: pytest.fail('Protected CL must not call model'))
    assert changed == task and not audit['eligible'] and not audit['changed']


@pytest.mark.parametrize('basis', ['predicted', 'assumption'])
def test_no_real_fu_does_not_call_predictor(basis):
    task, prior, receipt = fixture(fu_basis=basis)
    changed, audit = r.transformed_task(task, prior, receipt, lambda *a, **kw: pytest.fail('No fu evidence'))
    assert changed == task and not audit['eligible']


def test_general_binding_assumption_is_not_qualified_fu():
    task, prior, receipt = fixture(); task['request']['evidence'][1]['context']['general_protein_binding_adopted_for_plasma'] = True
    changed, audit = r.transformed_task(task, prior, receipt, lambda *a, **kw: pytest.fail('Unknown matrix'))
    assert changed == task and not audit['eligible']


def test_explicit_plasma_with_unknown_underlying_study_overlap_remains_eligible():
    task, _, _ = fixture()
    assert r.eligibility(task)[0]  # No invented independence flag on old reviewed sources.


@pytest.mark.parametrize('fault', ['matrix', 'target_role', 'forbidden_source', 'independence', 'renal_split', 'fu_zero'])
def test_invalid_or_leaking_fu_input_is_rejected(fault):
    task, _, _ = fixture(); fu = task['request']['evidence'][1]
    if fault == 'matrix': fu['context']['matrix'] = 'whole_blood'
    elif fault == 'target_role': fu['context']['study_role'] = 'evaluation'
    elif fault == 'forbidden_source': task['request']['forbidden_sources'] = [fu['source']]
    elif fault == 'independence': fu['context']['independent_of_target'] = False
    elif fault == 'fu_zero': fu['value'] = 0
    else: task['request']['evidence'].append({'parameter': 'renal_clearance', 'value': .5})
    with pytest.raises(ValueError): r.eligibility(task)


def test_beta_zero_is_exact_input_cache_identity():
    task, prior, receipt = fixture()
    prior['selection']['beta'] = 0
    changed, audit = r.transformed_task(task, prior, receipt, update_function(beta=0))
    assert changed == task and audit['eligible'] and not audit['changed']
    assert r.source_runner.exact_input_digest(changed) == r.source_runner.exact_input_digest(task)


def test_identical_fu_requires_exact_unit_factor():
    task, prior, receipt = fixture(); prior['parameters']['c']['fraction_unbound'] = .2
    changed, audit = r.transformed_task(task, prior, receipt, update_function(raw=0))
    assert changed == task and not audit['changed']
    with pytest.raises(ValueError, match='exact no-op'):
        r.transformed_task(task, prior, receipt, update_function(raw=.1))


@pytest.mark.parametrize('fault', ['factor_ignores_beta', 'outcome_used', 'beta', 'prior_fu'])
def test_model_receipt_must_match_fixed_formula_and_inputs(fault):
    task, prior, receipt = fixture()
    def predict(*args, **kwargs):
        out = update_function()(*args, **kwargs)
        if fault == 'factor_ignores_beta': out['factor'] = 10**out['bounded_delta_log10']
        elif fault == 'outcome_used': out['uses_clinical_observations'] = True
        elif fault == 'beta': out['beta'] = .9
        else: out['prior_fu'] = .3
        return out
    with pytest.raises(ValueError): r.transformed_task(task, prior, receipt, predict)


def test_chemical_blocker_remains_without_model_call():
    task, prior, receipt = fixture(blocked=True)
    out, audit = r.transformed_task(task, prior, receipt, lambda *a, **kw: pytest.fail('Blocked construction'))
    assert out == task and not audit['eligible']


def test_prior_does_not_replace_any_baseline_parameter():
    source = r.value.old.read(r.SOURCE/'protocol.json'); original = r.value.old.checked(source['parameter_prior'])
    prior = deepcopy(original); prior.update(version=r.PRIOR_VERSION, source_parameter_prior=source['parameter_prior'])
    prior['selection'] = {'kind': 'conditioned', 'beta': .5}
    prior['result_sha256'] = r.base.training_digest({k:v for k,v in prior.items() if k != 'result_sha256'})
    assert r.validate_prior(prior, original, source['parameter_prior']) is prior
    cid = next(iter(prior['parameters'])); prior['parameters'][cid]['total_clearance'] *= 1.01
    prior['result_sha256'] = r.base.training_digest({k:v for k,v in prior.items() if k != 'result_sha256'})
    with pytest.raises(ValueError, match='baseline parameters'): r.validate_prior(prior, original, source['parameter_prior'])


def test_real_source_has_eight_eligible_unique_inputs_without_native():
    source = r.value.old.read(r.SOURCE/'protocol.json'); prior = r.value.old.checked(source['parameter_prior'])
    prior['selection'] = {'kind': 'conditioned', 'beta': .5}
    audits = {}
    for key, original in source['tasks'].items():
        changed, audit = r.transformed_task(original, prior, source['parameter_prior'], update_function())
        assert changed['sidecars'] == original['sidecars']
        audits[key] = audit
    assert sum(a['eligible'] for a in audits.values()) == 8
    assert sum(a['changed'] for a in audits.values()) == 8
    assert sum(not a['changed'] for a in audits.values()) == 41
    groups = {con: {u['case_id'] for u in source['units'] if u['condition'] == con and audits[u['key']]['eligible']}
              for con in r.CONDITIONS}
    assert {k: len(v) for k,v in groups.items()} == {'all_ml': 0, 'available': 4,
        'missing_clearance': 8, 'missing_fu': 0, 'missing_both': 0}


def test_report_keeps_targets_sealed_before_prediction_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(r, 'own', lambda path: Path(path))
    r.save(tmp_path/'state.json', {'protocol': {'path': 'protocol'}, 'outcomes': {},
                                 'all_predictions_locked_before_scoring': False})
    monkeypatch.setattr(r.value.old, 'checked', lambda ref: {'tasks': {'unfinished': {}}})
    monkeypatch.setattr(r.value, 'verify', lambda protocol: None)
    monkeypatch.setattr(r.base, 'ClinicalStore', lambda *a: pytest.fail('No target access before locking'))
    with pytest.raises(ValueError, match='Lock every'): r.report(tmp_path)


def test_zero_update_cannot_pass_as_new_benefit():
    comparison = {'candidate_over_source_qualified': {'all': {con: {
        'candidate_scored': 27, 'source_scored': 27, 'common_curve_ratio': {'estimate': 1.}}
        for con in r.CONDITIONS}}}
    updates = {'applied_unique_requests': 0, 'by_condition': {con: {'actually_updated': {
        'meaningful_changes': {'benefit': 0, 'harm': 0}}} for con in r.CONDITIONS}}
    assert not r.adoption_gate(comparison, {'passed': True}, updates)['passed']


def test_new_run_recovery_namespace_is_not_foundation_candidate(monkeypatch, tmp_path):
    task, _, _ = fixture(); task['candidate'] = r.CANDIDATE
    folder = tmp_path/'model'; folder.mkdir()
    result = {'status': 'failed', 'reason': 'scientific_failure', 'observations_used': False,
        'request_sha256': r.value.model._digest(task['request']), 'experiment_fingerprint': task['key'],
        'experiment_candidate': r.CANDIDATE, 'native_solver_calls': 1, 'linear_trajectory_calls': 0}
    r.save(folder/'result.json', result)
    monkeypatch.setattr(r.value.old, 'collect_refs', lambda *a: None)
    monkeypatch.setattr(r.value.old, 'ref', lambda p: {'path': str(p), 'sha256': 'a'*64})
    monkeypatch.setattr(r.base.execution.separate, '_executor', lambda *a: pytest.fail('No second native execution'))
    assert r.invoke(task, folder)['status'] == 'failed'


def test_new_batch_monitor_does_not_scan_foundation_batch(monkeypatch, tmp_path):
    monkeypatch.setattr(r, 'BATCH', tmp_path)
    monkeypatch.setattr(r.retained.Monitor, 'check', lambda self, force=False: None)
    (tmp_path/'prior').mkdir(); (tmp_path/'prior/model').write_bytes(b'0123456789')
    clinical = tmp_path/'clinical'; clinical.mkdir()
    monitor = r.Monitor(clinical, {'directory_limit_bytes': 8}, {'outcomes': {}})
    with pytest.raises(r.value.old.Pause, match='batch exceeds'): monitor.check(force=True)


def test_actual_trainer_noop_api_on_all_frozen_inputs_without_training():
    module = r.importlib.import_module(r.MODULE)
    source = r.value.old.read(r.SOURCE/'protocol.json'); original = r.value.old.checked(source['parameter_prior'])
    artifact = deepcopy(original)
    artifact.update(version=r.PRIOR_VERSION, source_parameter_prior=source['parameter_prior'],
        model_receipt=original['model_receipts']['shared'], selection={'kind': 'none', 'beta': 0.})
    artifact['result_sha256'] = r.base.training_digest({k:v for k,v in artifact.items() if k != 'result_sha256'})
    r.validate_prior(artifact, original, source['parameter_prior'])
    eligible = 0
    for task in source['tasks'].values():
        result, audit = r.transformed_task(task, artifact, source['parameter_prior'], module.predict_update)
        assert result == task and not audit['changed']
        eligible += audit['eligible']
    assert eligible == 8


def test_full_prepare_metadata_matrix_without_native_or_clinical_scores(monkeypatch, tmp_path):
    source = r.value.old.read(r.SOURCE/'protocol.json'); prior = r.value.old.checked(source['parameter_prior'])
    prior.update(selection={'kind': 'conditioned', 'beta': .5}, model_receipt=prior['model_receipts']['shared'])
    module = r.importlib.import_module(r.MODULE)
    def update(*a, **kw):
        result = update_function()(*a, **kw)
        result.pop('receipt'); result['model_receipt'] = prior['model_receipt']
        return result
    monkeypatch.setattr(module, 'predict_update', update)
    monkeypatch.setattr(r, 'own', lambda path: Path(path))
    monkeypatch.setattr(r, 'checked_prior', lambda path, source: (prior, source['parameter_prior']))
    monkeypatch.setattr(r.value.model, 'assemble_evidence_blueprint', lambda *a: {'status': 'assembled'})
    original_checked = r.value.old.checked
    def checked(receipt):
        assert not receipt['path'].endswith('/results.json'), 'No baseline clinical score may be parsed'
        return original_checked(receipt)
    monkeypatch.setattr(r.value.old, 'checked', checked)
    destination = tmp_path/'isolated_prepare'; result = r.prepare(destination)
    assert result['logical_units'] == 155 and result['unique_predictions'] == 49
    assert result['new_native_calls_expected'] == 8 and result['exact_reused_predictions'] == 41
    assert len(list((destination/'inputs').glob('*.json'))) == 8
    protocol = r.value.old.read(destination/'protocol.json')
    assert len(protocol['changed_case_ids']) == 12
    assert len(protocol['stability_case_ids']) == 6 and len(protocol['fu_stability_case_ids']) == 8
    assert not r.value.old.read(destination/'state.json')['all_predictions_locked_before_scoring']


def test_adoption_uses_each_actual_update_group_not_all_unchanged_drugs():
    comparison = {'candidate_over_source_qualified': {'all': {con: {
        'candidate_scored': 27, 'source_scored': 27, 'common_curve_ratio': {'estimate': .98}}
        for con in r.CONDITIONS}}}
    updates = {'applied_unique_requests': 8, 'by_condition': {con: {'actually_updated': {
        'meaningful_changes': {'benefit': 1, 'harm': 0}}} for con in r.CONDITIONS}}
    assert r.adoption_gate(comparison, {'passed': True}, updates)['passed']
    updates['by_condition']['available']['actually_updated']['meaningful_changes']['benefit'] = 0
    assert not r.adoption_gate(comparison, {'passed': True}, updates)['passed']
