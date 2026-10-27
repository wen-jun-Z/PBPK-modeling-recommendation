"""Missing author inputs remain unknown; every other engineering failure blocks."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from recommendation_execution_repairs import source_unavailability as u, engineering_gate as g, pipeline as p, scoring


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    def write(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(value if isinstance(value,str) else json.dumps(value))
        return {'path':name,'sha256':u.sha(path)}
    source={'Version':77,'Formulations':[dict(u.FORMULATION)],'Simulations':[
        {'Name':name,'Compounds':[{'Name':'Repaglinide_DT','Protocol':{'Formulations':[
          {'Name':u.FORMULATION['Name'],'Key':'Formulation'}]}}]} for name in sorted(u.SIMULATIONS)]}
    source_ref=write('source.json',source)
    receipt={'source_path':source_ref['path'],'source_sha256':source_ref['sha256'],'status':'failed',
             'reasons':[u.SOURCE_FAILURE],**{key:False for key in ('ode_run','sealed_observations_accessed','fitting_performed','scientific_inapplicability_inferred')}}
    receipt_ref=write('receipt.json',receipt)
    log="Table formulation '"+u.FORMULATION['Name']+"' requires at least one point to be used in a simulation.\nCannot load Simulation '3.2.R_Tornio et al 2014_male_DDI'\nDIRECT_SNAPSHOT_LOAD_DONE\n"
    audit={'original_source':source_ref,'missing_formulation':dict(u.FORMULATION),
      'status':'original_formulation_input_missing; exact_recovery_not_established',
      'exact_original_table_identity_verified':False,'reference_curve_substituted':False,
      'clinical_outcomes_accessed':False,'fitting_performed':False,'failure_is_scientific_inapplicability':False,
      'affected_frozen_pairs':[{'scenario_id':a,'candidate_id':b} for a,b in sorted(u.PAIRS)],
      'source_simulations_requiring_missing_formulation':sorted(u.SIMULATIONS),
      'native_diagnostic':write('native.log',log),'same_commit_tree':write('tree.json',{}),
      'published_releases':write('releases.json',[]),'available_reference_formulation':write('reference.xml','not substituted')}
    audit_ref=write('audit.json',audit)
    monkeypatch.setattr(u,'SOURCE',source_ref);monkeypatch.setattr(u,'RECEIPT',receipt_ref);monkeypatch.setattr(u,'AUDIT',audit_ref)
    row={'source_sha256':source_ref['sha256'],'status':'failed','reasons':[u.SOURCE_FAILURE],'receipt':receipt_ref}
    provenance={'source_path':source_ref['path'],'source_sha256':source_ref['sha256']}
    def refresh():
        source_ref.update(write('source.json',source));receipt.update(source_sha256=source_ref['sha256'])
        receipt_ref.update(write('receipt.json',receipt));audit_ref.update(write('audit.json',audit))
        row['source_sha256']=source_ref['sha256'];provenance['source_sha256']=source_ref['sha256']
    return {'root':tmp_path,'source':source,'receipt':receipt,'audit':audit,'row':row,'provenance':provenance,'refresh':refresh}


def test_original_failure_is_accounted_separately_not_changed_to_verified(evidence):
    f=evidence;before=deepcopy(f['row'])
    r=u.assess_source_unavailable(f['row'],provenance=f['provenance'],project_root=f['root'])
    assert r['accepted'] and r['scientific_applicability']=='unknown'
    assert not r['source_execution_verified'] and not r['numerical_prediction_available']
    assert r['original_preservation_receipt_status']=='failed' and f['row']==before
    assert len(r['source_facts'])==2 and len(r['verified_references'])==7


@pytest.mark.parametrize('change',['extra_table','duplicate_formulation','missing_simulation','extra_simulation','wrong_dependency','duplicate_dependency','version','invented_values'])
def test_bound_but_semantically_different_source_is_rejected(evidence,change):
    f=evidence;s=f['source']
    if change=='extra_table':s['Formulations'][0]['Table']=[{'X':1,'Y':2}]
    elif change=='duplicate_formulation':s['Formulations'].append(dict(u.FORMULATION))
    elif change=='missing_simulation':s['Simulations'].pop()
    elif change=='extra_simulation':s['Simulations'].append({'Name':'unreviewed'})
    elif change=='wrong_dependency':s['Simulations'][0]['Compounds'][0]['Protocol']['Formulations'][0]['Name']='other'
    elif change=='duplicate_dependency':s['Simulations'][0]['Compounds']*=2
    elif change=='version':s['Version']=80
    else:s['Formulations'][0]['Parameters']=[{'Name':'inferred','Value':1}]
    f['refresh']()
    assert not u.assess_source_unavailable(f['row'],project_root=f['root'])['accepted']


@pytest.mark.parametrize('change',['outcomes','fitted','replaced','positive_label','wrong_pairs','missing_simulation','changed_source','success_receipt','wrong_failure','missing_native_evidence'])
def test_other_missing_or_changed_evidence_fails_closed(evidence,change):
    f=evidence;a=f['audit'];r=f['receipt']
    if change=='outcomes':a['clinical_outcomes_accessed']=True
    elif change=='fitted':r['fitting_performed']=True
    elif change=='replaced':a['reference_curve_substituted']=True
    elif change=='positive_label':a['failure_is_scientific_inapplicability']=True
    elif change=='wrong_pairs':a['affected_frozen_pairs'][0]['candidate_id']='other'
    elif change=='missing_simulation':a['source_simulations_requiring_missing_formulation'].pop()
    elif change=='changed_source':a['original_source']={'path':'another','sha256':'x'}
    elif change=='success_receipt':r['status']='verified'
    elif change=='wrong_failure':r['reasons']=['EXECUTION_TIMEOUT']
    else:(f['root']/a['native_diagnostic']['path']).write_text('native failed for another reason')
    f['refresh']()
    assert not u.assess_source_unavailable(f['row'],project_root=f['root'])['accepted']


@pytest.mark.parametrize('resource',['source.json','audit.json','receipt.json','native.log','tree.json','releases.json','reference.xml'])
def test_all_registered_evidence_is_hash_pinned(evidence,resource):
    f=evidence;(f['root']/resource).write_text('tampered')
    assert not u.assess_source_unavailable(f['row'],project_root=f['root'])['accepted']


@pytest.mark.parametrize('change',['source','status','reason','receipt','provenance'])
def test_general_engineering_failures_cannot_use_source_absence_exception(evidence,change):
    f=evidence;row=deepcopy(f['row']);prov=dict(f['provenance'])
    if change=='source':row['source_sha256']='0'*64
    elif change=='status':row['status']='verified'
    elif change=='reason':row['reasons']=['EXECUTION_TIMEOUT']
    elif change=='receipt':row['receipt']={**row['receipt'],'sha256':'0'*64}
    else:prov['source_path']='other.json'
    assert not u.assess_source_unavailable(row,provenance=prov,project_root=f['root'])['accepted']


def plan_fixture(f):
    sid,cid=sorted(u.PAIRS)[0]
    scenario={'scenario_id':sid,'case_id':'registered','normalized_design':{},'protocol_fields':{}}
    candidate={'candidate_id':cid,'catalog_provenance':f['provenance']}
    plan={'scenario_id':sid,'candidate_id':cid,'status':'blocked','reasons':['Blocked:'+u.PLAN_FAILURE],
          'sealed_observations_accessed':False,'fitting_performed':False}
    return scenario,candidate,plan


def test_exact_frozen_plan_is_unknown_and_has_causal_source_refs(evidence,monkeypatch):
    f=evidence;s,c,plan=plan_fixture(f)
    monkeypatch.setattr(g,'_context',lambda *a:(s,c,[]))
    r=g.assess_blocked_plan(plan,project_root=f['root'])
    assert r['accepted'] and r['category']=='original_source_required_input_absent'
    assert r['scientific_applicability']=='unknown'
    for key in ['candidate_id','scenario_id']:
        bad=deepcopy(plan);bad[key]='unregistered'
        assert not g.assess_blocked_plan(bad,project_root=f['root'])['accepted']
    for key in ['sealed_observations_accessed','fitting_performed']:
        bad=deepcopy(plan);bad[key]=True
        assert not g.assess_blocked_plan(bad,project_root=f['root'])['accepted']


def test_source_counts_truthful_and_native_timeout_not_waived(evidence,monkeypatch):
    f=evidence;monkeypatch.setattr(p,'ROOT',f['root'])
    source={'status':'complete','all_sources_accounted':True,'expected_sources':1,'verified_sources':0,'results':[f['row']]}
    original=deepcopy(source)
    report=p.engineering_completeness(source,[],[])
    assert report['accepted'] and source==original
    assert report['source_accounting']=={'verified_sources':0,'verified_unavailable_sources':1,'accounted_sources':1}
    for key in ['expected_sources','verified_sources']:
        bad=deepcopy(source);bad[key]+=1
        assert not p.engineering_completeness(bad,[],[])['accepted']
    bad=deepcopy(source);bad['results']*=2;bad['expected_sources']=2
    assert not p.engineering_completeness(bad,[],[])['accepted']
    job={'job_id':'timeout','scenario_id':'s','candidate_id':'c','plan':{'status':'ready'}}
    result={'scenario_id':'s','candidate_id':'c','status':'blocked','reasons':['EXECUTION_TIMEOUT']}
    assert not p.engineering_completeness(source,[job],[result])['accepted']


def test_verify_source_repairs_requires_preserved_failed_receipt_and_returns_evidence(evidence,monkeypatch):
    from recommendation_execution_repairs import snapshot_repair
    f=evidence;calls=[]
    monkeypatch.setattr(snapshot_repair,'_verify_receipt',lambda *a,**kw:calls.append((a,kw)))
    source={'status':'complete','all_sources_accounted':True,'expected_sources':1,'verified_sources':0,'results':[f['row']]}
    refs=p.verify_source_repairs(source,[{'identity_match':True,'catalog_provenance':f['provenance']}],root=f['root'])
    assert calls[0][1]['require_accepted'] is False
    assert u.RECEIPT in refs and u.AUDIT in refs and u.SOURCE in refs
    assert source['results'][0]['status']=='failed'


def test_offline_barrier_rejects_omitted_unknown_source_evidence(monkeypatch):
    run=Path('/synthetic/run');work=run/'engineering_repairs/repair_04';required={'path':'absence.json','sha256':'x'}
    gate={'accepted':True,'verified_unavailable_sources':[{'assessment':{'accepted':True,'verified_references':[required]}}]}
    refs=[{'path':str(work/name),'sha256':'x'} for name in ('source_repair_summary.json','engineering_completeness.json')]
    docs={run/'protocol.json':{'inputs':refs},work/'source_repair_summary.json':{},
          work/'engineering_completeness.json':gate,run/'inputs/execution_plan.json':{'jobs':[]}}
    class Files:
        unsealed=False;root=Path('/synthetic')
        def checked(self,ref):assert not self.unsealed;return Path(ref['path'])
        def read(self,path):assert not self.unsealed;return docs[path]
        def rows(self,path):raise AssertionError('Must reject before source receipt or outcomes')
    monkeypatch.setattr(p,'engineering_completeness',lambda *a:gate)
    f=Files()
    with pytest.raises(ValueError,match='Unavailable-case evidence'):
        scoring.verify_engineering_barrier({'run':run,'files':f,'native':{}})
    assert not f.unsealed


def test_live_source_and_two_frozen_candidate_plans_pass_without_outcomes():
    root=g.ROOT
    row={'source_sha256':u.SOURCE['sha256'],'status':'failed','reasons':[u.SOURCE_FAILURE],'receipt':u.RECEIPT}
    assert u.assess_source_unavailable(row,project_root=root)['accepted']
    jobs=json.loads((root/'experiments/model_recommendation/execution_validation_20261001/engineering_repairs/repair_04/execution_plan.json').read_text())['jobs']
    checked=0
    for job in jobs:
        if (job['scenario_id'],job['candidate_id']) in u.PAIRS:
            r=g.assess_blocked_plan(job['plan'],project_root=root)
            assert r['accepted'] and r['scientific_applicability']=='unknown';checked+=1
    assert checked==2


def test_new_modules_are_not_per_candidate_execution_dependencies():
    from recommendation_execution_repairs import complete_candidate,snapshot_candidate
    changed={f'scripts/recommendation_execution_repairs/{x}.py' for x in ('pipeline','engineering_gate','scoring','source_unavailability')}
    assert not changed.intersection(complete_candidate.IMPLEMENTATION_FILES)
    assert not changed.intersection(snapshot_candidate.IMPLEMENTATION_FILES)


def test_freeze_binds_new_audits_before_any_native_execution(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'WORK',tmp_path/'work');monkeypatch.setattr(p,'RUN',tmp_path/'run')
    monkeypatch.setattr(p,'code_files',lambda:[])
    frozen=p.freeze()
    expected=g.known_gate_evidence_refs(project_root=p.ROOT)
    registry_path='experiments/model_recommendation/execution_validation_20261001/engineering_amendments/json_source_gate_followup_20261002/json_route_sources.json'
    registry_sha='30c46247355116b2a84629f4022a5e6a92fc7e2401e1ed58d8c361e058df39e6'
    assert u.sha(p.ROOT/registry_path)==registry_sha
    definitions=json.loads((p.ROOT/registry_path).read_text())['source_definitions']
    assert set(definitions)=={
        '195c2fc857b67b14eea34b60a24ef3ae87b0c618e650aa241a05549ab69aa4d4',
        '959fd4e3a8a325eff0be12e5f428047ffba673fbd4e17dda69984a6dc4a1a1b9',
        'a5a6724a4ecd8f70d6784383854433f06442e562541a18454bddbd0d82943014',
        'af1223c0ea0544a226f5024f06daf422c48f5a5e56d232874a033f7c6d664baa'}
    new_refs=[{'path':registry_path,'sha256':registry_sha}]
    for source_sha,definition in definitions.items():
        assert definition['source']['sha256']==source_sha
        new_refs.extend(definition[key] for key in ('source','snapshot','receipt'))
    new_identities={(ref['path'],ref['sha256']) for ref in new_refs}
    expected_identities={(ref['path'],ref['sha256']) for ref in expected}
    assert len(new_refs)==len(new_identities)==len({ref['path'] for ref in new_refs})==13
    assert len(expected)==len(expected_identities)==len({ref['path'] for ref in expected})==26
    assert new_identities.issubset(expected_identities)
    assert len(expected_identities-new_identities)==13
    assert {tuple(sorted(r.items())) for r in expected}.issubset({tuple(sorted(r.items())) for r in frozen['inputs']})
    assert frozen['sealed_outcomes_accessed'] is False and frozen['fitting_performed'] is False


def test_protocol_promotion_includes_all_unavailable_plan_evidence(tmp_path,monkeypatch):
    work=tmp_path/'work';out=tmp_path/'out';rank=tmp_path/'rank'
    for path in (work,out,rank):path.mkdir()
    def write(path,value):path.write_text(json.dumps(value))
    source_ref={'path':'author_failure.json','sha256':'source'}
    plan_ref={'path':'clinical_design.json','sha256':'plan'}
    gate={'accepted':True,'verified_unavailable_sources':[{'assessment':{'accepted':True,'verified_references':[source_ref]}}],
          'verified_unavailable_cases':[{'assessment':{'accepted':True,'verified_references':[plan_ref]}}]}
    job={'job_id':'j','scenario_id':'s','candidate_id':'c','plan':{}}
    write(work/'execution_plan.json',{'jobs':[job]});write(work/'source_repair_summary.json',{})
    write(work/'predictions_lock.json',{'target_observations_read':False,'preparation':{},'execution_plan':{},
          'planned_jobs':1,'jobs':[{'job_id':'j','artifacts':[]}]})
    write(rank/'ranking_lock.json',{'state':'complete','scientific_outcomes_read':False})
    statistical=tmp_path/'protocol.json';write(statistical,{'cases':1,'candidates':1,'methods':['method']})
    monkeypatch.setattr(p,'WORK',work);monkeypatch.setattr(p,'OUT',out);monkeypatch.setattr(p,'STATISTICAL_PROTOCOL',statistical)
    monkeypatch.setattr(p.ranking,'RANK',rank);monkeypatch.setattr(p.ranking,'METHODS',['method'])
    monkeypatch.setattr(p,'freeze',lambda:{'frozen_at':'test','inputs':[],'code':[],'runtime':{}})
    checked=[];monkeypatch.setattr(p,'check',lambda ref:checked.append(ref))
    monkeypatch.setattr(p,'verify_job',lambda job:{'scenario_id':'s','candidate_id':'c'})
    monkeypatch.setattr(p,'engineering_completeness',lambda *a:gate)
    monkeypatch.setattr(p,'verify_source_repairs',lambda *a:[])
    monkeypatch.setattr(p.original,'verify_retained_ranking',lambda method:None)
    monkeypatch.setattr(p,'rows',lambda path:[{}])
    monkeypatch.setattr(p,'reference',lambda path:{'path':str(path),'sha256':'fixture'})
    published={};monkeypatch.setattr(p.ranking,'frozen_write',lambda path,value:published.update({path:value}))
    p.promote_protocol()
    protocol=published[out/'protocol.json']
    assert source_ref in protocol['inputs'] and plan_ref in protocol['inputs']
    assert source_ref in checked and plan_ref in checked
    assert protocol['outcome_access']['sealed_values_read_by_preparation'] is False
