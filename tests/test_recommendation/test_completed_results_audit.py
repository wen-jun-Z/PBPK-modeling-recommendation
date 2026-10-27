"""Synthetic report boundaries only; never call score_run or read observations."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import pytest
ROOT=Path('/home/xuyiqian/Research_Projects/Knowledge_Graph_Model_Recommendation')
BASE=ROOT/'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/json_native_bridge_followup_20261002'
spec=importlib.util.spec_from_file_location('completed_audit',BASE/'audit_completed_results.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
spec=importlib.util.spec_from_file_location('synthetic_scoring_fixture',ROOT/'tests/test_recommendation/test_execution_scoring.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def fixture():
 s=old.scenario();sid=s['scenario_id'];cids=['a','b','c'];methods=['bm25_text','new_method']
 audits={(sid,c):{'candidate_id':c,'identity_match':c!='c','family_id':'family'+c} for c in cids}
 native={(sid,'a'):{'status':'prediction_locked','prediction_locked':True},(sid,'b'):{'status':'blocked','prediction_locked':False}}
 tiers={(sid,c):{'declared_use':'reference_only','reasons':['synthetic']} for c in cids}
 ctx={'qids':[sid],'cids':cids,'methods':methods,'scenarios':{sid:s},'audits':audits,'native':native,'tiers':tiers,
      'matrices':{'bm25_text':[[1.,1.,0.]],'new_method':[[0.,1.,1.]]},'tiers_manifest':None,
      'locks':{'protocol_sha256':'p','ranking_lock_sha256':'r','predictions_lock_sha256':'n'}}
 comparisons=[]
 for c in cids:
  n={'result':native[(sid,c)],'prediction':old.prediction() if c=='a' else None} if c!='c' else None
  row=a.scorer.candidate_comparison(s,audits[(sid,c)],n,[])
  row['existing_policy_decision']=tiers[(sid,c)]
  if c=='a':row['reference_diagnostic']['endpoints']={'Cmax_reference_vs_group':{'predicted':20.,'observed':20.}}
  comparisons.append(row)
 inv={'scenario_id':sid,'requested_numerical_endpoints':['Cmax_reference_vs_group'],
      'same_parent_candidates':2,'reference_numerically_evaluable_candidates':1}
 selections=[];csv=[];curves=[]
 for method in methods:
  for stratum in a.STRATA:
   selection,rows,curve=a.reproduce_selection(ctx,comparisons,inv,method,sid,stratum)
   selections.append(selection);csv.extend(rows)
   if curve is not None:curves.append(curve)
 report={'schema_version':a.scorer.VERSION,'locks':{**ctx['locks'],'sealed_sources_read_after_all_locks':{}},
  'candidate_results':comparisons,'selection':selections,'curve_selection':curves,'observation_inventory':[inv],
  'method_endpoint_summary':a.scorer.aggregate_method_rows(csv),
  'paired_comparisons':a.statistics.paired_method_statistics(csv,methods),
  'declared_policy_audit':a.scorer.audit_declared_policy(comparisons),
  'summary':{'scenarios':1,'candidates':3,'methods':2,'compounds':1,'ranking_pairs_per_method':3,
   'all_planned_native_jobs_accounted_for':2,'native_status_counts':{'prediction_locked':1,'blocked':1},
   'all_candidate_scope_rows':3,'opened_sealed_files':0,'numerically_evaluable_reference_pairs':1,
   'direct_validated_pairs':0,'scoped_source_audit_pass_pairs':0,'source_audit_status_counts':{'unknown':2},
   'absolute_source_independence_claimed':False,'scalar_dispositions':{},'exact_recall':None,
   'missed_scientifically_suitable_candidates':None,'complete_execution_accounting':True,'complete_scientific_validation':False}}
 return report,csv,ctx


def test_synthetic_tie_unknown_and_conditional_denominators(monkeypatch):
 monkeypatch.setattr(a.scorer,'score_run',lambda *x,**kw:pytest.fail('Scoring must never be triggered'))
 report,csv,ctx=fixture();out=a.audit_payload(report,csv,ctx)
 assert out['accepted'] and out['counts']['candidate_results']==3
 selection=next(r for r in report['selection'] if r['method']=='bm25_text' and r['stratum']=='reference_diagnostic')
 assert selection['by_k']['1']['inclusion_probabilities']=={'a':.5,'b':.5,'c':0.}
 assert selection['endpoints']['Cmax_reference_vs_group']['top1_evaluable_probability']==.5
 assert selection['endpoints']['Cmax_reference_vs_group']['selected_known_conditional_absolute_log_error']==0.
 assert selection['by_k']['1']['exact_recall'] is None
 assert all(r['state']=='unavailable' and r['reason'] and r['holm_adjusted_p'] is None for r in report['paired_comparisons']['comparisons'])
 strings=[{k:'' if v is None else str(v) for k,v in r.items()} for r in csv]
 assert a.audit_payload(report,strings,ctx)['accepted']


@pytest.mark.parametrize('tamper',['missing_candidate','duplicate_candidate','negative_unknown','wrong_native','missing_selection',
 'tie','coverage','csv','summary','aggregate','paired_count','paired_reason','holm','missing_curve','policy'])
def test_each_report_mapping_or_denominator_tamper_is_rejected(tamper):
 report,csv,ctx=fixture()
 if tamper=='missing_candidate':report['candidate_results'].pop()
 elif tamper=='duplicate_candidate':report['candidate_results'][1]=deepcopy(report['candidate_results'][0])
 elif tamper=='negative_unknown':report['candidate_results'][1]['strict']['scientific_status']='incompatible'
 elif tamper=='wrong_native':report['candidate_results'][1]['native_status']='prediction_locked'
 elif tamper=='missing_selection':report['selection'].pop()
 elif tamper=='tie':report['selection'][0]['by_k']['1']['inclusion_probabilities']['a']=1.
 elif tamper=='coverage':report['selection'][1]['endpoints']['Cmax_reference_vs_group']['top1_evaluable_probability']=1.
 elif tamper=='csv':csv[1]['measured_pool_candidate_count']=2
 elif tamper=='summary':report['summary']['all_candidate_scope_rows']=2
 elif tamper=='aggregate':report['method_endpoint_summary'][0]['scenario_count']=2
 elif tamper=='paired_count':report['paired_comparisons']['comparisons'][0]['compounds']=10
 elif tamper=='paired_reason':report['paired_comparisons']['comparisons'][0]['reason']=None
 elif tamper=='holm':report['paired_comparisons']['comparisons'][0]['holm_adjusted_p']=0.
 elif tamper=='missing_curve':report['curve_selection'].pop()
 elif tamper=='policy':report['candidate_results'][0]['existing_policy_decision']['declared_use']='direct_reuse'
 with pytest.raises(ValueError):a.audit_payload(report,csv,ctx)


@pytest.mark.parametrize('jid',sorted(a.SUPERSEDED_ROSUVA_JOBS))
def test_each_superseded_rosuvastatin_execution_is_rejected(jid):
 with pytest.raises(ValueError,match='SUPERSEDED_ROSUVASTATIN'):a.reject_superseded_jobs([{'job_id':jid}])
 a.reject_superseded_jobs([{'job_id':jid.removesuffix('_r4b')+'_r4g'}])


@pytest.mark.parametrize('stage',['missing_results','missing_lock','gate_false','unbound_gate','rank_incomplete'])
def test_no_result_bytes_are_read_before_complete_gates(tmp_path,monkeypatch,stage):
 run=tmp_path/'run';work=run/'work';rank=run/'rankings';native=run/'native'
 for p in [run,work,rank,native]:p.mkdir(exist_ok=True)
 def put(p,value):p.write_text(json.dumps(value))
 for p in [run/'results.json',run/'results.csv']:p.write_text('SECRET RESULT MUST NOT BE READ')
 put(work/'engineering_completeness.json',{'accepted':stage!='gate_false'})
 put(work/'predictions_lock.json',{'target_observations_read':False,'jobs':[],'planned_jobs':0})
 f=a.PublicFiles(tmp_path)
 inputs=[] if stage=='unbound_gate' else [f.ref(work/'engineering_completeness.json'),f.ref(work/'predictions_lock.json')]
 put(run/'protocol.json',{'schema_version':'pbpk-recommendation-execution-protocol-v1','inputs':inputs,
   'outcome_access':{'all_predictions_and_rankings_locked_before_scoring':True,'sealed_values_read_by_preparation':False}})
 put(rank/'ranking_lock.json',{'state':'running' if stage=='rank_incomplete' else 'complete','scientific_outcomes_read':False})
 put(native/'predictions_lock.json',{'target_observations_read':False,'protocol':f.ref(run/'protocol.json'),'jobs':[],'planned_jobs':0})
 if stage=='missing_results':(run/'results.csv').unlink()
 elif stage=='missing_lock':(native/'predictions_lock.json').unlink()
 original=a.PublicFiles.read
 def guarded(self,p):
  if Path(p).name in ('results.json','results.csv'):pytest.fail('Opened result before complete public gate')
  return original(self,p)
 monkeypatch.setattr(a.PublicFiles,'read',guarded)
 with pytest.raises(ValueError):a.gate_before_results(tmp_path,run,work)


def test_sealed_paths_and_declared_observation_refs_cannot_be_hashed_or_read(tmp_path):
 p=tmp_path/'sealed.json';p.write_text('{}');f=a.PublicFiles(tmp_path)
 with pytest.raises(ValueError,match='SEALED_FILE'):f.read(p)
 p=tmp_path/'observation.json';p.write_text('{}');f.denied.add(p)
 with pytest.raises(ValueError,match='SEALED_FILE'):f.sha(p)
