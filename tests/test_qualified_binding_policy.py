"""The source rule is score-blind, conserves other inputs and reuses exact data."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_qualified_binding_validation.py'
SPEC=importlib.util.spec_from_file_location('qualified_binding_runner_test',SCRIPT)
r=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(r)


def record(basis='derived',assumed=True):
    return {'parameter':'fraction_unbound','compound_id':'c','value':.3,'unit':'fraction','basis':basis,
        'source':{'path':'source.json','sha256':'a'*64,'location':'Human protein binding'},
        'context':{'species':'human','matrix':'plasma','underlying_study_overlap':'unknown',
                   'general_protein_binding_adopted_for_plasma':assumed}}


def fallback():
    value=record('predicted',False);value['value']=.17
    value['prediction_method']={'name':'frozen_model','version':'1','training_overlap':False,
        'applicability':{'human_IV_parameter_model':True,'target_structure_and_scaffold_excluded':True}}
    value['source']={'path':'prior.json','sha256':'b'*64,'location':'Frozen own-compound prior'}
    return value


def tasks():
    request={'compound_id':'c','human':{'weight_kg':70},'dosing':{'dose_mg':30},'target':{'window_end_h':24},
        'evidence_ledger':{'path':'available.json'},'partition_method':'PKSimStandard',
        'evidence':[record(),{'parameter':'total_clearance','value':3,'basis':'predicted','source':{'path':'prior.json'}}]}
    original={'key':'available','request':request,'sidecars':{'vss':[1],'half':[2]},'core':{'version':'unchanged'}}
    ml=deepcopy(original);ml['key']='all_ml';ml['request']['evidence'][0]=fallback()
    ml['request']['evidence'].reverse();ml['request']['evidence_ledger']={'path':'all_ml.json'}
    return original,ml


def test_only_explicit_matrix_assumption_rejects_and_preserves_source():
    original,_=tasks();saved=deepcopy(original['request'])
    changed,audit=r.policy.qualify_binding(original['request'],fallback())
    assert original['request']==saved
    assert changed['evidence'][0]==fallback() and changed['evidence'][1:]==saved['evidence'][1:]
    assert audit['rejected_evidence']==saved['evidence'][0]
    assert audit['reason']==r.policy.REJECTION and audit['clinical_outcomes_used'] is False


@pytest.mark.parametrize('assumed',[False,None,'true'])
def test_unknown_study_overlap_alone_does_not_reject(assumed):
    original,_=tasks();original['request']['evidence'][0]['context']['general_protein_binding_adopted_for_plasma']=assumed
    result,audit=r.policy.qualify_binding(original['request'],fallback())
    assert result==original['request'] and audit['changed'] is False


@pytest.mark.parametrize('fault',['compound','unit','overlap','target_source','value'])
def test_fallback_must_remain_a_qualified_same_compound_prediction(fault):
    original,_=tasks();prior=fallback()
    if fault=='compound':prior['compound_id']='another'
    elif fault=='unit':prior['unit']='percent'
    elif fault=='overlap':prior['prediction_method']['training_overlap']=True
    elif fault=='value':prior['value']=0
    else:original['request']['forbidden_sources']=[prior['source']]
    with pytest.raises(ValueError):r.policy.qualify_binding(original['request'],prior)


def test_exact_all_ml_request_is_bound_only_after_fu_only_proof():
    original,ml=tasks();before=deepcopy(original)
    selected,audit=r.bind_existing_request(original,ml)
    assert original==before and selected==ml
    assert r.exact_input_digest(selected)==r.exact_input_digest(ml)
    assert audit['non_fu_evidence_preserved'] is True
    assert selected['request']['evidence_ledger']==ml['request']['evidence_ledger']


@pytest.mark.parametrize('changed',['total_cl','renal_cl','sidecars','dose','source_context'])
def test_all_ml_reuse_cannot_drop_other_measured_inputs_or_solver_conditions(changed):
    original,ml=tasks()
    if changed=='total_cl':original['request']['evidence'][1].update(basis='measured',value=5)
    elif changed=='renal_cl':original['request']['evidence'].append({'parameter':'renal_clearance','basis':'measured','value':1})
    elif changed=='sidecars':original['sidecars']['vss']=[3]
    elif changed=='dose':original['request']['dosing']['dose_mg']=50
    else:original['request']['evidence'][1]['source']={'path':'independent.json'}
    with pytest.raises(ValueError,match='Needs new native'):r.bind_existing_request(original,ml)


def test_old_plasma_evidence_is_exact_identity_without_using_scores():
    original,ml=tasks();original['request']['evidence'][0]['context'].pop('general_protein_binding_adopted_for_plasma')
    original['request']['evidence'][0]['context']['underlying_study_overlap']='unknown'
    selected,audit=r.bind_existing_request(original,ml)
    assert selected==original and audit['changed'] is False


def test_runner_is_bounded_and_report_stays_blocked_before_lock(tmp_path,monkeypatch):
    with pytest.raises(ValueError):r.own(r.SOURCE)
    assert r.own(r.DEFAULT)==r.DEFAULT
    monkeypatch.setattr(r,'own',lambda path:path);monkeypatch.setattr(r.value,'verify',lambda protocol:None)
    r.save(tmp_path/'protocol.json',{'tasks':{'one':{}}})
    r.save(tmp_path/'state.json',{'protocol':r.value.old.ref(tmp_path/'protocol.json'),
        'all_predictions_locked_before_scoring':False,'outcomes':{}})
    def forbidden(*args):raise AssertionError('Observation data must not open before the prediction lock')
    monkeypatch.setattr(r.base,'ClinicalStore',forbidden)
    with pytest.raises(ValueError,match='Lock every prediction'):r.report(tmp_path)


def test_benefit_denominator_keeps_source_rejected_cases(monkeypatch):
    cases=[{'case_id':str(i),'compound_id':str(i),'cluster_id':str(i),'validation_stratum':'new_compounds'} for i in range(3)]
    def data(reject=False):
        return {'units':[{'case_id':c['case_id'],'condition':con,'curve_metrics':{'gmfe':
            (2 if reject and c['case_id']=='0' else 1.8) if con=='available' else 2}} for c in cases for con in r.CONDITIONS]}
    protocol={'cases':cases,'original_evidence_changed_case_ids':['0','1','2'],'current_evidence_changed_case_ids':['1','2']}
    out=r.comparison_statistics(data(True),data(False),protocol)
    assert out['evidence_benefit_original_group']['qualified']['requested_drugs']==3
    assert out['evidence_benefit_original_group']['qualified']['meaningful_improved_drugs_5pct']==2
    assert out['denominator_reduced_for_benefit_claim'] is False
