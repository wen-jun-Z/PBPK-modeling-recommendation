"""Independent scoring boundaries; no solver or observed outcomes are used."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest

STAGED=Path(__file__).resolve().parents[2]
PROJECT=next(p for p in (STAGED,*STAGED.parents) if (p/'src/pbpk_kb').is_dir())
sys.path[:0]=[str(PROJECT/'scripts'),str(PROJECT/'src')]
NAME='recommendation_execution_repairs.json_native_scoring'
spec=importlib.util.spec_from_file_location(NAME,STAGED/'scripts/recommendation_execution_repairs/json_native_scoring.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@pytest.fixture
def inspection(tmp_path):
    from recommendation_execution_repairs import native_pkml as n
    root=tmp_path;directory=root/'run/native/job';base=directory/'inspect';base.mkdir(parents=True)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def save(path,record):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record,sort_keys=True))
    source=directory/'source.pkml';source.write_text('<Model />')
    ref={'path':str(source.relative_to(root)),'sha256':sha(source)}
    worker=n.WORKER;adapter='scripts/recommendation_execution_repairs/native_pkml.py'
    code={worker:'a'*64,adapter:'b'*64}
    request={'version':n.VERSION,'action':'inspect','patches':[],'pkml_path':str(source),'pkml_sha256':sha(source),
        'result_path':str(base/'native_result.json'),'before_pkml_path':str(base/'before.pkml'),'after_pkml_path':str(base/'after.pkml')}
    save(base/'request.json',request)
    raw={'version':n.VERSION,'action':'inspect','request_sha256':sha(base/'request.json'),'pkml_sha256':sha(source),
         'solver_calls':0,'parameter_changes':[],'sealed_observations_accessed':False,'fitting_performed':False,
         'inventory':{'parameters':[{'path':'Dose','value':1.}]}}
    save(base/'native_result.json',raw)
    process={'status':'completed','returncode':0,'script':worker,'script_sha256':code[worker],
        'command':['Rscript','--vanilla',worker,str(base/'request.json'),sha(base/'request.json')],
        'runtime_unchanged':True,'registry_modified':False}
    save(base/'process_manifest.json',process)
    result={'version':n.VERSION,'action':'inspect','status':'inspected','native':raw,'source_sha256':sha(source),
        'sealed_observations_accessed':False,'fitting_performed':False,'worker_sha256':code[worker],
        'python_sha256':code[adapter],'process_manifest':str((base/'process_manifest.json').relative_to(root))}
    save(base/'result.json',result)
    artifact={'version':n.VERSION,'source':ref,'prediction_accepted':False,
        'files':{str(p.relative_to(root)):sha(p) for p in base.iterdir()}}
    save(base/'artifact_lock.json',artifact)
    class Files:
        def __init__(self):self.root=root;self.unsealed=False
        def path(self,p):return Path(p) if Path(p).is_absolute() else self.root/p
        def sha(self,p):return sha(self.path(p))
        def read(self,p):assert not self.unsealed;return json.loads(self.path(p).read_text())
        def checked(self,r):
            p=self.path(r['path'])
            if sha(p)!=r['sha256']:raise ValueError('hash differs')
            return p
    files=set(base.iterdir())|{source}
    return dict(f=Files(),directory=directory,base=base,plan={'implementation_sha256':code},registered={'pkml':ref},
        global_files=set(files),prediction_files=set(files),save=save)


def check_inspect(p):return m.verified_inspection(p['f'],p['directory'],p['plan'],p['registered'],p['global_files'],p['prediction_files'])


def test_real_hashed_inspection_chain_passes_without_solver(inspection):
    assert check_inspect(inspection)['parameters'][0]['value']==1.
    assert not inspection['f'].unsealed


@pytest.mark.parametrize('fault',['source','raw_request','solver','worker','destination','command','global','prediction','artifact_source','artifact_acceptance'])
def test_inspection_rejects_changes_at_each_execution_boundary(inspection,fault):
    p=inspection;f=p['f'];base=p['base'];target=base/'native_result.json';record=f.read(target)
    if fault=='source':p['registered']['pkml']['sha256']='c'*64
    elif fault=='global':p['global_files'].remove(base/'result.json')
    elif fault=='prediction':p['prediction_files'].remove(base/'artifact_lock.json')
    elif fault in ('artifact_source','artifact_acceptance'):
        target=base/'artifact_lock.json';record=f.read(target)
        if fault=='artifact_source':record['source']['path']='unrelated.pkml'
        else:record['prediction_accepted']=True
        p['save'](target,record)
    else:
        if fault=='raw_request':record['request_sha256']='c'*64
        if fault=='solver':record['solver_calls']=1
        if fault=='worker':target=base/'result.json';record=f.read(target);record['worker_sha256']='c'*64
        if fault=='destination':target=base/'request.json';record=f.read(target);record['result_path']=str(base/'wrong.json')
        if fault=='command':target=base/'process_manifest.json';record=f.read(target);record['command'][-1]='c'*64
        p['save'](target,record)
    with pytest.raises(ValueError):check_inspect(p)


@pytest.fixture
def split_case(monkeypatch):
    import recommendation_execution_repairs as package
    root=Path('/synthetic');directory=root/'run/native/job'
    source={'path':'source.pksim5','sha256':'a'*64};evidence={'path':'source-proof.json','sha256':'b'*64}
    policy={'source_original':source,'native_source':source,'simulation_id':'sim','source_model_blob_sha256':'c'*64,'evidence_refs':[evidence]}
    contract={'dose_mg':40,'administration_times_min':[0.]};metadata={'original':True}
    patches=[{'dose':25.36,'time':0.},{'dose':14.64,'time':138.}];events=deepcopy(patches)
    plan={'source_path':source['path'],'source_sha256':source['sha256'],'source_simulation_id':'sim','source_model_blob_sha256':'c'*64,
          'split_absorption_contract':deepcopy(policy),'internal_absorption_event_count':2,'contract':contract,'source_metadata':metadata}
    path=directory/'split_absorption_contract.json';docs={path:deepcopy(policy),directory.parent.parent/'protocol.json':{'inputs':[evidence]}}
    class Files:
        def __init__(self):self.root=root
        def read(self,p):return docs[p]
        def sha(self,p):return 'd'*64
        def checked(self,r):return root/r['path']
    registered={'split_absorption_contract':{'path':str(path.relative_to(root)),'sha256':'d'*64},'patches':deepcopy(patches),'administrations':deepcopy(events)}
    module=types.ModuleType('recommendation_execution_repairs.split_absorption');calls=[]
    def revalidate(p,c,md,*,project_root):
        if p!=policy or c!=contract or md!=metadata:raise ValueError('source contract differs')
        calls.append('source')
    def build(i,c,md,p,*,project_root):revalidate(p,c,md,project_root=project_root);return deepcopy(patches),deepcopy(events)
    def readback(native,p):
        if native!={'readback':'both_pulses'}:raise ValueError('native dose or time differs')
        calls.append('readback');return {'accepted':True}
    module.revalidate_contract=revalidate;module.build_split_patches=build;module.verify_native_readback=readback
    monkeypatch.setitem(sys.modules,module.__name__,module);monkeypatch.setattr(package,'split_absorption',module,raising=False)
    monkeypatch.setattr(m,'verified_inspection',lambda *a: {'native':'inventory'})
    return dict(f=Files(),directory=directory,plan=plan,registered=registered,native={'readback':'both_pulses'},global_files={path},prediction_files={path},docs=docs,path=path,calls=calls)


def check_split(p):return m.verify_split_semantics(p['f'],p['directory'],p['plan'],p['registered'],p['native'],p['global_files'],p['prediction_files'])


def test_split_absorption_rechecks_source_patches_and_actual_readback(split_case):
    assert check_split(split_case)['accepted'];assert split_case['calls']==['source','source','readback']


@pytest.mark.parametrize('fault',['source','pulse_count','suppressed','lag','request_ref','lock','native','evidence_missing','evidence_duplicate'])
def test_split_absorption_rejects_single_pulse_or_unbound_contract(split_case,fault):
    p=split_case
    if fault=='source':p['plan']['source_sha256']='f'*64
    if fault=='pulse_count':p['plan']['internal_absorption_event_count']=1
    if fault=='suppressed':p['plan']['surplus_event_suppression']={'zero':True}
    if fault=='lag':p['registered']['patches'][1]['time']=0.
    if fault=='request_ref':p['registered']['split_absorption_contract']['sha256']='f'*64
    if fault=='lock':p['prediction_files'].clear()
    if fault=='native':p['native']={'readback':'one_pulse'}
    if fault.startswith('evidence'):
        refs=p['docs'][p['directory'].parent.parent/'protocol.json']['inputs']
        if fault=='evidence_missing':refs.clear()
        else:refs.append(deepcopy(refs[0]))
    with pytest.raises(ValueError):check_split(p)


@pytest.fixture
def dual_case(monkeypatch):
    import recommendation_execution_repairs as package
    root=Path('/synthetic');directory=root/'run/native/job'
    source={'path':'native.pksim5','sha256':'a'*64};evidence={'path':'source-proof.json','sha256':'b'*64}
    dual={'identity':'original-dual'};observer={'output_path':'total-RS','original_observer_sha256':'c'*64,'molecular_weight_g_per_mol':345.416}
    dual['molecular_weight_g_per_mol']=345.416
    patches=[{'parent':'R','dose':10},{'parent':'S','dose':10}];events=deepcopy(patches)
    plan={'source_path':'original.json','source_sha256':'e'*64,'source_simulation':'CONTROL','source_simulation_id':'sim',
        'source_model_blob_sha256':'f'*64,'contract':{'dose_mg':20},'source_metadata':{'original':True},'output_path':'total-RS',
        'molecular_weight_g_per_mol':345.416,'dual_parent_contract':deepcopy(dual),'dual_parent_observer':deepcopy(observer),
        'source_output_contract':deepcopy(observer),'json_native_source_proof':{'native_source':source},'dual_parent_suppression':{}}
    paths=[directory/'dual_parent_contract.json',directory/'dual_parent_observer.json']
    docs={paths[0]:deepcopy(dual),paths[1]:deepcopy(observer),directory.parent.parent/'protocol.json':{'inputs':[evidence]}}
    class Files:
        def __init__(self):self.root=root
        def read(self,p):return docs[p]
        def sha(self,p):return 'd'*64
        def checked(self,r):return root/r['path']
    registered={'patches':deepcopy(patches),'administrations':deepcopy(events),'output_path':'total-RS','molecular_weight_g_per_mol':345.416,
        'dual_parent_contract':{'path':str(paths[0].relative_to(root)),'sha256':'d'*64},
        'dual_parent_observer':{'path':str(paths[1].relative_to(root)),'sha256':'d'*64}}
    modules={n:types.ModuleType('recommendation_execution_repairs.'+n) for n in ('json_source_design','json_racemate_contract','json_dual_native')}
    modules['json_source_design'].prepare_source_design_amendment=lambda *a,**k:None
    rc=modules['json_racemate_contract'];rc.PARENTS=('S','R');rc.OUTPUT_PATH='total-RS'
    rc.prepare_dual_parent_contract=lambda *a,**k:deepcopy(dual)
    rc.validate_native_observer=lambda *a,**k:deepcopy(observer)
    nd=modules['json_dual_native'];nd.evidence_refs=lambda d:[evidence]
    nd.source_output_contract=lambda *a,**k:deepcopy(observer)
    nd.build_patches=lambda *a,**k:(deepcopy(patches),deepcopy(events))
    nd.native_output=lambda *a:('total-RS',345.416)
    for name,module in modules.items():
        monkeypatch.setitem(sys.modules,module.__name__,module);monkeypatch.setattr(package,name,module,raising=False)
    monkeypatch.setattr(m,'verified_inspection',lambda *a:{'parameters':[]})
    monkeypatch.setattr(m.ET,'parse',lambda p:types.SimpleNamespace(getroot=lambda:object()))
    return dict(f=Files(),directory=directory,plan=plan,registered=registered,before=directory/'native/before.pkml',
        global_files=set(paths),prediction_files=set(paths),docs=docs,paths=paths)


def check_dual(p):return m.verify_bridge_semantics(p['f'],p['directory'],p['plan'],p['registered'],p['before'],p['global_files'],p['prediction_files'])


def test_dual_scorer_rederives_contract_original_observer_and_both_doses(dual_case):
    assert check_dual(dual_case)['both_parent_doses_recomputed']


@pytest.mark.parametrize('fault',['single_parent','lost_pulse','observer','dual_contract','request_ref','local_content','lock','evidence','design','expansion'])
def test_dual_scorer_rejects_unbound_or_modified_semantics(dual_case,fault):
    p=dual_case
    if fault=='single_parent':p['plan']['coadministration_isolation']={'target':['S']}
    if fault=='lost_pulse':p['registered']['patches'][0]['dose']=0
    if fault=='observer':p['plan']['source_output_contract']['original_observer_sha256']='wrong'
    if fault=='dual_contract':p['plan']['dual_parent_contract']['identity']='different'
    if fault=='request_ref':p['registered']['dual_parent_observer']['sha256']='wrong'
    if fault=='local_content':p['docs'][p['paths'][1]]['changed']=True
    if fault=='lock':p['global_files'].clear()
    if fault=='evidence':p['docs'][p['directory'].parent.parent/'protocol.json']['inputs'].clear()
    if fault=='design':p['plan']['source_design_amendment']={'changed':True}
    if fault=='expansion':p['plan']['application_expansion']={'new_event':True}
    with pytest.raises(ValueError):check_dual(p)
