"""Bounded lifecycle, held-out isolation and scientific denominators; no native work."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_accuracy_refinement as run


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    run.save(path, value)
    return path


@pytest.fixture
def batch(tmp_path, monkeypatch):
    root = tmp_path/'project';root.mkdir()
    runner = run.AccuracyRunner(root, adapter=SimpleNamespace(simulate_scenario=lambda *a: None))
    folder = runner.default
    def refdoc(path, value):
        save(path, value);return runner.ref(path)
    for name in ('scripts/run_accuracy_refinement.py','scripts/run_purpose_hundred.py',
            'scripts/run_purpose_refinement.py','src/pbpk_kb/blueprint_policy/capacity_guided_design.py',
            'src/pbpk_kb/blueprint_policy/learned_exchange_design.py',
            'src/pbpk_kb/blueprint_learning/rich_parameter_prior.py',
            'src/pbpk_kb/blueprint_policy/hundred_evaluation.py','engine.py'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture\n')
    monkeypatch.setattr(run.engine,'_code_pins',lambda root:{'engine.py':run.sha256(root/'engine.py')})
    paper=root/'results/manuscript/main.tex';paper.parent.mkdir(parents=True);paper.write_text('frozen paper')
    priorbase={'methods':{'selected':{'predict_nominal':{'total_clearance':2.,'fraction_unbound':.2,
        'vss':3.,'terminal_half_life':4.,'mrt':5.}}},
        'selected_models':{k:'test_model' for k in ('total_clearance','fraction_unbound','vss','terminal_half_life','mrt')},
        **{k:False for k in ('training_scaffold_overlap','training_structure_overlap',
            'calibration_scaffold_overlap','calibration_structure_overlap','cohort_labels_used','target_labels_used')}}
    predictions={};cases=[];plans={};outcomes={};pins={}
    for i in range(100):
        cid=f'drug_{i}';compound=f'compound:{i}';identity=f'identity{i}'
        evaluation=refdoc(runner.source/'evaluations'/f'{cid}.json',{'compound_id':compound,
            'identity_smiles':identity,'evaluation_scope':'aggregated_human_IV_PK_parameters','weight_kg':70,
            'clinical_curve':None,'endpoints':{'clearance':{'value':8.4,'unit':'l/h'},
                'vss':{'value':210.,'unit':'l'},'terminal_half_life':{'value':4.,'unit':'h'},
                'mrt':{'value':5.,'unit':'h'}}})
        pins[evaluation['path']]=evaluation['sha256']
        cases.append({'case_id':cid,'compound_id':compound,'identity_smiles':identity,'drug_name':cid,
            'evaluation_scope':'aggregated_human_IV_PK_parameters','evaluation':evaluation})
        predictions[compound]={'identity_smiles':identity,**deepcopy(priorbase)}
        plans[cid]={'request':{'compound_id':compound,'human':{'weight_kg':70.},'assumptions':[]}}
        outcomes[cid]=refdoc(runner.source/'work'/cid/'outcome.json',{'case_id':cid,'status':'locked',
            'methods':{'combined':run.direct_parameters(priorbase,plans[cid]['request'])},'artifact_pins':{}})
    old={'cases':cases,'case_protocols':plans,'source_pins':pins,'code_pins':{}}
    oldref=refdoc(runner.source/'protocol.json',old)
    refdoc(runner.source/'state.json',{'status':'complete','protocol':oldref,
        'outcomes':outcomes,'report_artifact_pins':{}})
    model=runner.source/'model.bin';model.write_bytes(b'fixture model')
    frozen={'version':'rich-scaffold-five-endpoint-prior-1','configuration':'fixture'}
    audit={'version':frozen['version'],'frozen_input':frozen,'input_fingerprint':run.digest(frozen),
        'predictions_digest':run.digest(predictions),'target_predictions_locked_before_scoring':True,
        'source_pins':{'engine.py':run.sha256(root/'engine.py')},
        **{k:False for k in ('target_observation_files_opened','calibration_labels_used_for_selection','target_labels_used_for_selection')}}
    auditref=refdoc(folder/'priors/audit.json',audit)
    priorbundle={'version':frozen['version'],'input_fingerprint':run.digest(frozen),'predictions':predictions,
        'artifacts':{'model':runner.ref(model),'audit':auditref}}
    refdoc(folder/'priors/predictions.json',priorbundle)
    return SimpleNamespace(root=root,runner=runner,folder=folder,cases=cases,prior=priorbase,
        audit=audit,priorbundle=priorbundle,refdoc=refdoc,paper=paper)


def test_direct_ml_unit_conversion_is_weight_normalized_and_not_a_curve():
    prior={'methods':{'selected':{'predict_nominal':{'total_clearance':20.,'vss':2.,
        'terminal_half_life':3.,'mrt':4.}}}}
    result=run.direct_parameters(prior,{'human':{'weight_kg':70.}})
    assert result['endpoints']['clearance']=={'value':84.,'unit':'l/h','status':'predicted'}
    assert result['endpoints']['vss']['value']==140.
    assert not result['curve_simulated']
    with pytest.raises(ValueError):run.direct_parameters(prior,{'human':{'weight_kg':0.}})


def test_missing_predictions_keep_reference_denominator_and_pair_only_common_cases():
    records=[]
    for case in ('a','b'):
        for method in run.METHODS:
            missing=case=='b' and method=='rich_learned_exchange'
            records.append({'case_id':case,'method':method,'endpoint':'vss','reference':1.,
                'prediction':None if missing else 2.,'absolute_ln_error':None if missing else run.math.log(2.)})
    summary=run.summarize(records)['vss']
    assert summary['reference_n']==2
    assert summary['own_scorable']['previous_combined']['n']==2
    assert summary['own_scorable']['rich_learned_exchange']['n']==1
    assert summary['unscorable_with_reference']['rich_learned_exchange']==1
    assert summary['paired_with_previous']['rich_learned_exchange']['n']==1
    assert summary['paired_with_previous']['rich_learned_exchange']['gmfe_ratio']==1.
    assert all(v['n']==1 for v in summary['common_all_methods'].values())


def test_prepare_hashes_target_without_decoding_it_and_freezes_prior(batch,monkeypatch):
    r=batch.runner;read=r.read
    def sentinel(path):
        assert '/evaluations/' not in str(path)
        return read(path)
    monkeypatch.setattr(r,'read',sentinel)
    r.prepare(batch.folder,workers=1)
    p,s=r.load(batch.folder)
    assert len(p['cases'])==100 and len(p['case_protocols'])==100 and not s['outcomes']
    assert p['half_target_basis']=='sampled_nca' and p['half_search_budget']==33
    assert p['half_policy']=='best_valid_NCA_prior_agreement_if_better_than_PS1; no_clinical_matching_claim'
    assert 'src/pbpk_kb/blueprint_learning/rich_parameter_prior.py' in p['code_pins']
    assert not (batch.folder/'work').exists()
    (batch.root/'engine.py').write_text('changed')
    with pytest.raises(run.IntegrityError):r.load(batch.folder)


@pytest.mark.parametrize('corruption',['prediction_digest','missing_isolation_flag','source_hash'])
def test_prepare_rejects_forged_or_incomplete_prior_contract(batch,corruption):
    audit=deepcopy(batch.audit)
    if corruption=='prediction_digest':audit['predictions_digest']='0'*64
    elif corruption=='missing_isolation_flag':audit.pop('target_labels_used_for_selection')
    else:audit['source_pins']['engine.py']='0'*64
    source=batch.refdoc(batch.folder/'priors/audit.json',audit)
    bundle=deepcopy(batch.priorbundle);bundle['artifacts']['audit']=source
    batch.refdoc(batch.folder/'priors/predictions.json',bundle)
    with pytest.raises(run.IntegrityError):batch.runner.prepare(batch.folder)
    assert not (batch.folder/'protocol.json').exists()


def test_report_requires_full_lock_and_stopped_monitor_before_any_target_decode(batch,monkeypatch):
    r=batch.runner;r.prepare(batch.folder,workers=1);read=r.read
    def sentinel(path):
        assert '/evaluations/' not in str(path)
        return read(path)
    monkeypatch.setattr(r,'read',sentinel)
    with pytest.raises(ValueError,match='lock'):r.report(batch.folder)


class Monitor:
    def __init__(self,runner,folder,state):
        self.runner,self.folder,self.state=runner,folder,state;self.lock=threading.RLock();self.failures=[]
    def check(self,**kwargs):pass
    def persist(self,**kwargs):
        self.state.update(kwargs);run.save(self.folder/'state.json',self.state)
    def record_result(self,value):self.failures.append(value);self.persist()


def install_simulation(batch,monkeypatch,*,pause=True):
    from pbpk_kb.blueprint_policy import learned_exchange_design
    counts=Counter()
    def native(request,folder,root,check):
        counts['native']+=1;folder.mkdir(parents=True);(folder/'readback.txt').write_text('fixed native')
        return {'synthetic':True}
    def prediction():
        return {'status':'predicted','endpoints':{'vss':{'value':210.,'unit':'l','status':'completed'}},
            'full_trajectory':{'time_h':[0.,1.],'concentration_mg_l':[0.,1.]}}
    def fixed(*args,**kwargs):counts['fixed']+=1;return prediction()
    def learned(*args,**kwargs):
        assert kwargs['target_basis']=='sampled_nca' and kwargs['max_trajectory_calls']==33
        counts['learned']+=1
        if pause and counts['learned']==1:raise run.Paused('planned interruption')
        return {**prediction(),'learned_exchange_audit':{'selected_PS':1.}}
    monkeypatch.setattr(run,'rich_request',lambda original,*args:deepcopy(original))
    monkeypatch.setattr(run.engine,'_physiology',native)
    monkeypatch.setattr(run.capacity,'anchor_partition',lambda *args:{'status':'anchored','log_parameters':{}})
    monkeypatch.setattr(run.capacity,'simulate_design',fixed)
    monkeypatch.setattr(learned_exchange_design,'simulate_learned_design',learned)
    return counts


def test_partial_resume_reuses_native_and_finished_method_and_rejects_outcome_tamper(batch,monkeypatch):
    r=batch.runner;r.prepare(batch.folder,workers=1);p,s=r.load(batch.folder)
    counts=install_simulation(batch,monkeypatch);m=Monitor(r,batch.folder,s);case=p['cases'][0]
    with pytest.raises(run.Paused):r._case(batch.folder,p,case,m)
    assert counts=={'native':1,'fixed':1,'learned':1} and not s['outcomes']
    r._case(batch.folder,p,case,m)
    assert counts=={'native':1,'fixed':1,'learned':2}
    outcome=r.checked(s['outcomes'][case['case_id']]);assert set(outcome['methods'])==set(run.METHODS)
    r._case(batch.folder,p,case,m);assert counts=={'native':1,'fixed':1,'learned':2}
    outcome['methods']['rich_fixed_exchange']['endpoints']['vss']['value']=999.
    run.save(batch.folder/'work'/case['case_id']/'outcome.json',outcome)
    with pytest.raises(run.IntegrityError):r._case(batch.folder,p,case,m)


def test_partial_native_corruption_pauses_instead_of_scoring_model_failure(batch,monkeypatch):
    r=batch.runner;r.prepare(batch.folder,workers=1);p,s=r.load(batch.folder)
    counts=install_simulation(batch,monkeypatch);m=Monitor(r,batch.folder,s);case=p['cases'][0]
    with pytest.raises(run.Paused):r._case(batch.folder,p,case,m)
    (batch.folder/'work'/case['case_id']/'physiology.json').write_text('{"synthetic":"tampered"}')
    with pytest.raises(run.IntegrityError):r._case(batch.folder,p,case,m)
    assert not s['outcomes'] and counts=={'native':1,'fixed':1,'learned':1}


def test_all_hundred_complete_before_label_decode_and_resume_does_not_reexecute(batch,monkeypatch):
    r=batch.runner;r.prepare(batch.folder,workers=2)
    counts=install_simulation(batch,monkeypatch,pause=False);read=r.read;opened=[]
    def guarded(path):
        if '/evaluations/' in str(path):
            state=json.loads((batch.folder/'state.json').read_text())
            assert state['all_predictions_locked_before_scoring'] and state['monitor_stopped_before_scoring']
            assert len(state['outcomes'])==100
            opened.append(str(path))
        return read(path)
    monkeypatch.setattr(r,'read',guarded)
    assert r.execute(batch.folder)['status']=='predictions_locked'
    assert not opened and counts=={'native':100,'fixed':100,'learned':100}
    result=r.report(batch.folder)
    assert len(opened)==100 and result['registered_compounds']==100
    assert result['endpoint_metrics']['vss']['reference_n']==100
    assert result['endpoint_metrics']['terminal_half_life']['unscorable_with_reference']['rich_fixed_exchange']==100
    assert len(result['structure_audits'])==100 and all(a['exchange']=={'selected_PS':1.} for a in result['structure_audits'])
    r.execute(batch.folder);r.report(batch.folder)
    assert counts=={'native':100,'fixed':100,'learned':100} and len(opened)==100


def test_qualified_measured_clearance_survives_rich_prior_replacement(tmp_path):
    frozen=ROOT/'experiments/blueprint_design/purpose_hundred_20260930/protocol.json'
    protocol=json.loads(frozen.read_text());case=protocol['cases'][0]
    request=deepcopy(protocol['case_protocols'][case['case_id']]['input']['request'])
    prior={'methods':{'selected':{'predict_nominal':{'total_clearance':999.,'fraction_unbound':.123}}},
        'selected_models':{'total_clearance':'rich_model','fraction_unbound':'rich_model'}}
    cl=next(r for r in request['evidence'] if r['parameter']=='total_clearance')
    cl['basis']='measured';cl.pop('prediction_method',None);before=deepcopy(cl)
    changed=run.rich_request(request,prior,{'path':'fixture.json','sha256':'a'*64},tmp_path)
    actual={r['parameter']:r for r in changed['evidence']}
    assert all(actual['total_clearance'][k]==v for k,v in before.items())
    assert actual['fraction_unbound']['value']==.123
    assert actual['fraction_unbound']['prediction_method']['version']==run.PRIOR_VERSION
    assert changed['evidence_ledger']['sha256']==run.sha256(tmp_path/'evidence.json')
