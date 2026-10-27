"""New-cohort isolation, protected inputs, units, weighting and failed attempts."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_expanded_blueprint_validation.py'
SPEC=importlib.util.spec_from_file_location('expanded_validation_test',SCRIPT)
r=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(r)


def record(name,val,basis='predicted',unit=None):
    return {'parameter':name,'compound_id':'c','basis':basis,'value':val,
            'unit':unit or r.PARAMETER_UNITS[name],'source':{'path':'old','sha256':'a'*64,'location':'row'}}


def test_fixed_batch_and_unprepared_status_are_safe():
    assert r.own(r.DEFAULT)==r.DEFAULT
    with pytest.raises(ValueError):r.own(r.DEFAULT.parent/'another')


def test_new_parameters_preserve_independent_cl_fu_and_renal():
    rows=[record('total_clearance',5,'measured'),record('fraction_unbound',.2,'derived'),
          record('renal_clearance',1,'measured','ml/min/kg')]
    request={'compound_id':'c','evidence':rows};sides={'vss':[record('vss',2)],'half':[record('terminal_half_life',3,'measured')]}
    params={'total_clearance':9,'fraction_unbound':.9,'vss':4,'terminal_half_life':8}
    req,side=r.replace_parameters(request,sides,params,{'path':'new','sha256':'b'*64})
    assert req==request and side['half']==sides['half'] and side['vss'][0]['value']==4
    assert sides['vss'][0]['value']==2


def test_refit_cannot_reuse_old_predictions_even_if_values_coincide():
    original={'request':{'evidence':[record('total_clearance',5),record('fraction_unbound',.2)]},
              'sidecars':{'vss':[record('vss',2)],'half':[record('terminal_half_life',3)]}}
    values={'total_clearance':5,'fraction_unbound':.2,'vss':2,'terminal_half_life':3}
    assert not r.complete_cache_match(original,values,'refit_union_exclusion')
    assert r.complete_cache_match(original,values,'reuse_frozen_weights')
    assert not r.complete_cache_match(original,{**values,'vss':2.00001},'reuse_frozen_weights')


def curve_case(unit='mg/l'):
    return {'case_id':'c','compound_id':'c','study_id':'study','nca_policy':{'sampling_time_h':[0,1,2]},
        'request_template':{'human':{'weight_kg':70},'dosing':{'dose_mg':50},'evidence':[record('molecular_weight',500,'measured','g/mol')]},
        'reference_locator':{'source':{'path':'snapshot','sha256':'a'*64},'observation_index':0,
            'observation_name':'series','column_index':0,'quantity_path':'plasma|parent',
            'time_unit':'h','concentration_unit':unit,'concentration_dimension':'Concentration (mass)'}}


def snapshot(unit='mg/l'):
    return {'ObservedData':[{'Name':'series','BaseGrid':{'Values':[0,1,2],'Unit':'h'},'Columns':[
        {'Values':[1,2,3],'Unit':unit,'Dimension':'Concentration (mass)',
         'QuantityInfo':{'Path':'plasma|parent'},'RelatedColumns':[{'Values':['must not decode SD']}]}]}]}


@pytest.mark.parametrize('unit,factor',[('mg/l',2),('µg/ml',2),('µg/l',.002),('ng/ml',.002),
    ('µmol/l',1),('ng/l',.000002),('nmol/l',.001),('mol/l',1e6)])
def test_report_curve_units_follow_unit_field_and_ignore_sd(unit,factor):
    ref=r.decode_curve(curve_case(unit),snapshot(unit))
    assert ref['observations'][1]['time_min']==60
    assert ref['observations'][1]['concentration_umol_per_l']==pytest.approx(2*factor)
    assert ref['molecular_weight_g_mol']==500


def test_frozen_sampling_and_column_identity_are_checked():
    raw=snapshot();raw['ObservedData'][0]['Columns'][0]['QuantityInfo']['Path']='brain|metabolite'
    with pytest.raises(ValueError,match='identity'):r.decode_curve(curve_case(),raw)
    case=curve_case();case['nca_policy']['sampling_time_h']=[0,1,3]
    with pytest.raises(ValueError,match='schedule'):r.decode_curve(case,snapshot())


def test_multiple_studies_do_not_give_one_drug_extra_weight():
    cases=[{'case_id':'a1','compound_id':'a','cluster_id':'s1'},
           {'case_id':'a2','compound_id':'a','cluster_id':'s2'},
           {'case_id':'b','compound_id':'b','cluster_id':'s3'}]
    result=r.drug_equal({'a1':4,'a2':4,'b':1},cases,resamples=100,seed=7)
    assert result['estimate']==pytest.approx(2)
    assert result['estimate']!=pytest.approx(16**(1/3))
    assert result==r.drug_equal({'a1':4,'a2':4,'b':1},cases,resamples=100,seed=7)
    failed=r.drug_equal({'a1':4,'a2':None,'b':1},cases,resamples=100)
    assert failed['n_cases']==2 and failed['requested_cases']==3


def test_cluster_bootstrap_matches_reference_draws_with_repeated_cluster_multiplicity():
    cases=[{'case_id':str(i),'compound_id':str(i),'cluster_id':str(i//2)} for i in range(8)]
    vals={str(i):float(i+1) for i in range(8)}
    resamples,seed=241,12
    indices=np.random.default_rng(seed).integers(0,4,size=(resamples,4))
    reference=[]
    for draw in indices:
        duplicated=[vals[str(2*c+j)] for c in draw for j in (0,1)]
        reference.append(math.exp(sum(math.log(x) for x in duplicated)/len(duplicated)))
    got=r.drug_equal(vals,cases,resamples=resamples,seed=seed)
    assert got['ci95']==pytest.approx(np.quantile(reference,[.025,.975]))
    legacy=r.value.scores.clustered_geomean(vals,{c['case_id']:c['cluster_id'] for c in cases},resamples=resamples,seed=seed)
    assert got['ci95']==pytest.approx(legacy['ci95'])


def test_schema_does_not_accept_clinical_concentrations_during_prepare():
    with pytest.raises(ValueError,match='must not contain observations'):
        r._metadata_case({'observations':[1,2]}, {}, {})


def test_new_metadata_constructs_a_valid_selected_policy_request():
    source={'path':'source.json','sha256':'a'*64,'location':'independent molecular property'}
    def chemical_row(name,val,unit):
        return {'parameter':name,'compound_id':'c','value':val,'unit':unit,'basis':'derived','source':source}
    chemical={'compound_id':'c','smiles':'CC','construction_status':'ready','missing_reasons':[],
        'independent_evidence':[],'evidence':[chemical_row('molecular_weight',30.,'g/mol'),
            chemical_row('molecular_formula','C2H6','text'),chemical_row('logp',1.,'Log Units'),
            chemical_row('pka_types',[],'1')]}
    metadata={'case_id':'c','drug':'Drug','study_id':'study','cluster_id':'study',
        'human':{'age_years':30.,'weight_kg':70.,'height_cm':175.,'sex':'MALE','population':'European_ICRP_2002','seed':17},
        'dosing':{'route':'intravenous','dose_mg':50.,'start_h':0.,'infusion_minutes':30.},
        'target':{'window_start_h':0.,'window_end_h':2.},'reference_locator':{'source':source},
        'nca_policy':{'sampling_time_h':[0.,1.,2.]}}
    case=r._metadata_case(metadata,chemical,source)
    request=case['request_template']
    request['forbidden_sources']=[]
    parameters={'total_clearance':3.,'fraction_unbound':.2,'vss':1.,'terminal_half_life':3.}
    base,side=r.replace_parameters(request,{'vss':[],'half':[]},parameters,source)
    selected,decision=r.select_request(base,[],'available',source)
    assert decision['status']=='ready',decision['reasons']
    assert len(selected['target']['endpoints'])==11
    assert selected['entry_reason']=='model_missing'
    assert side['vss'][0]['prediction_method']['name']==r.PREDICTOR


def test_blocked_construction_is_a_locked_zero_native_failure(tmp_path):
    (tmp_path/'models').mkdir();key='c'*64
    task={'key':key,'case_id':'c','request':{'compound_id':'c'},'construction_blocked':['pka_exceeds_three']}
    monitor=SimpleNamespace(lock=threading.RLock(),state={'outcomes':{}},check=lambda:None)
    monitor.update=lambda **kw:monitor.state.update(kw)
    assert r.solve(task,tmp_path,monitor)==key
    item=r.value.old.checked(monitor.state['outcomes'][key])
    assert item['entry']['status']=='failed' and item['new_native_calls']==0
    assert item['entry']['design_audit']['reasons']==['pka_exceeds_three']
    assert item['entry']['target_observations_used'] is False and item['locked_at']


def test_scoring_refuses_to_read_observations_before_all_predictions_lock(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'own',lambda path:path)
    monkeypatch.setattr(r.value,'verify',lambda protocol:None)
    r.save(tmp_path/'protocol.json',{'tasks':{'missing':{}}})
    r.save(tmp_path/'state.json',{'protocol':r.value.old.ref(tmp_path/'protocol.json'),
        'all_predictions_locked_before_scoring':False,'outcomes':{}})
    def forbidden(*args):raise AssertionError('Clinical references must remain sealed')
    monkeypatch.setattr(r,'ClinicalStore',forbidden)
    with pytest.raises(ValueError,match='Lock every prediction'):r.report(tmp_path)


def test_cleanup_refuses_active_run_and_keeps_complete_reproduction_files(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'own',lambda path:path)
    monkeypatch.setattr(r.value,'verify',lambda protocol:None)
    r.save(tmp_path/'protocol.json',{})
    state={'protocol':r.value.old.ref(tmp_path/'protocol.json'),'status':'prepared','outcomes':{}}
    r.save(tmp_path/'state.json',state)
    with pytest.raises(ValueError,match='completed'):r.cleanup(tmp_path)
    state['status']='complete';r.save(tmp_path/'state.json',state)
    original=r.sha256(tmp_path/'state.json');result=r.cleanup(tmp_path)
    assert result['status']=='no_op' and result['deleted_files']==0 and result['bytes']>0
    assert r.sha256(tmp_path/'state.json')==original


def test_five_conditions_remove_all_clearance_evidence_together(monkeypatch):
    # Inspect the actual masking and allocation helper, independent of schema.
    seen=[]
    def policy(req,fallback):
        seen.append(req);return {'status':'ready','selected_evidence':req['evidence']}
    monkeypatch.setattr(r,'input_policy',policy)
    base={'compound_id':'c','evidence':[record('total_clearance',2),record('fraction_unbound',.5),
                                      record('elimination_scope','hepatic_only','assumption','text')]}
    independent=[record('total_clearance',5,'measured'),record('renal_clearance',1,'derived','ml/min/kg'),record('fraction_unbound',.1,'measured')]
    req,_=r.select_request(base,independent,'missing_clearance',{'path':'rule'})
    assert all(x['basis']=='predicted' for x in req['evidence'] if x['parameter'] in r.value.CLEARANCES)
    assert any(x['parameter']=='fraction_unbound' and x['basis']=='measured' for x in req['evidence'])


def test_dynamic_summary_uses_actual_requested_denominators(monkeypatch):
    monkeypatch.setattr(r,'RESAMPLES',30)
    cases=[{'case_id':'a','compound_id':'a','cluster_id':'s1','validation_stratum':'old_20'},
           {'case_id':'b','compound_id':'b','cluster_id':'s2','validation_stratum':'new_compounds'}]
    rows=[]
    for c in cases:
        for condition in r.CONDITIONS:
            rows.append({'case_id':c['case_id'],'condition':condition,'curve_metrics':{'gmfe':2 if c['case_id']=='a' else None},
                'attempt_status':'predicted' if c['case_id']=='a' else 'failed','reference_family':'observation_derived',
                'changed_from_nominal':False,'endpoints':{e:{'error':None,'reference':None} for e in r.value.UNITS}})
    summary,_,_=r.summarize(rows,{'cases':cases,'changed_case_ids':[]})
    total=next(x for x in summary if x['group']=='all' and x['condition']=='available')
    assert total['attempted']==2 and total['failed']==1
    threshold=total['curve']['threshold_statistics']['within_fold']
    assert threshold['2']['count']==0 and threshold['3']['count']==1
    assert threshold['3']['attempted_denominator']==2
