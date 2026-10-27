"""Execution revision adaptation cannot weaken the pre-outcome audit barrier."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from recommendation_execution_repairs import scoring


def test_profile_requires_all_registered_repair_dependencies_and_restores_backend():
    before=scoring.backend.EXECUTION_REVISION
    impl={name:'a'*64 for name in scoring.complete_candidate.IMPLEMENTATION_FILES}
    record={'execution_revision':scoring.REVISION,'implementation_sha256':impl}
    with scoring.repaired_execution_profile():
        assert scoring.backend.verify_execution_binding(record,frozen_code_sha256=impl,
            expected_revision=scoring.REVISION,label='native')==impl
        missing=deepcopy(record);missing['implementation_sha256'].pop('scripts/recommendation_execution_repairs/model_semantics.py')
        with pytest.raises(ValueError,match='dependencies are incomplete'):
            scoring.backend.verify_execution_binding(missing,frozen_code_sha256=impl,expected_revision=scoring.REVISION,label='native')
    assert scoring.backend.EXECUTION_REVISION==before


def test_legacy_dependencies_cannot_impersonate_repaired_executor():
    impl={name:'a'*64 for name in scoring.backend.EXECUTION_IMPLEMENTATION_REQUIRED}
    with scoring.repaired_execution_profile(), pytest.raises(ValueError,match='registered adapter'):
        scoring.backend.verify_execution_binding({'execution_revision':scoring.REVISION,'implementation_sha256':impl},
            frozen_code_sha256=impl,expected_revision=scoring.REVISION,label='native')


@pytest.mark.parametrize('omit_from', ['global','prediction','none'])
def test_complete_model_audit_is_bound_in_both_locks_before_unsealing(monkeypatch,omit_from):
    run=Path('/synthetic-unsealed');native=run/'native/job/native/result.json';plock=run/'native/job/prediction_lock.json'
    docs={run/'native/predictions_lock.json':{'jobs':[{'job_id':'job','artifacts':[] if omit_from=='global' else [{'path':str(native),'sha256':'x'}]}]},
          plock:{'files':{} if omit_from=='prediction' else {str(native):'x'}},
          native:{'status':'prediction_locked','source_import_audit':{'accepted':True},'model_preservation_audit':{'accepted':True}}}
    class Files:
        unsealed=True
        def checked(self,ref):
            assert not self.unsealed
            assert ref['sha256'] in {'x','lock'}
            return Path(ref['path'])
        def read(self,path):
            assert not self.unsealed
            return docs[Path(path)]
    f=Files();bundle={'run':run,'files':f,'native':{'job':{'job_id':'job','plan':{'adapter_route':'complete_original_pksim5'},
        'result':{'prediction_locked':True,'prediction_lock':str(plock),'prediction_lock_sha256':'lock'}}}}
    monkeypatch.setattr(scoring.backend,'verify_barrier',lambda *a,**k:bundle)
    monkeypatch.setattr(scoring,'verify_engineering_barrier',lambda bundle:None)
    monkeypatch.setattr(scoring,'verify_complete_native_binding',lambda *a:None)
    with scoring.repaired_execution_profile():
        if omit_from=='none':
            assert scoring.backend.verify_barrier(run) is bundle
            assert f.unsealed
        else:
            with pytest.raises(ValueError,match='both locks'):scoring.backend.verify_barrier(run)
            assert not f.unsealed


def test_offline_scoring_cannot_omit_engineering_reports():
    run=Path('/synthetic-unsealed')
    class Files:
        unsealed=False
        def read(self,path):
            assert not self.unsealed
            assert path==run/'protocol.json'
            return {'inputs':[]}
        def checked(self,ref):return Path(ref['path'])
    f=Files()
    with pytest.raises(ValueError,match='reports are absent'):
        scoring.verify_engineering_barrier({'files':f,'run':run,'native':{}})
    assert not f.unsealed


@pytest.mark.parametrize('include_overlap',[False,True])
def test_offline_scoring_requires_registered_overlap_evidence_before_outcomes(monkeypatch,include_overlap):
    import recommendation_execution_source_audit as source_audit
    from recommendation_execution_repairs import pipeline
    root=Path('/synthetic');run=root/'run';work=run/'engineering_repairs/repair_04'
    refs=[{'path':str(work/name),'sha256':'x'} for name in ('source_repair_summary.json','engineering_completeness.json')]
    overlap={'path':str(root/'overlap.json'),'sha256':'y'}
    if include_overlap:refs.append(overlap)
    gate={'accepted':True}
    docs={run/'protocol.json':{'inputs':refs},work/'source_repair_summary.json':{},
          work/'engineering_completeness.json':gate,run/'inputs/execution_plan.json':{'jobs':[]}}
    class Files:
        unsealed=False
        def checked(self,ref):
            assert not self.unsealed
            return Path(ref['path'])
        def read(self,p):return docs[p]
        def rows(self,p):return []
    f=Files();f.root=root
    monkeypatch.setattr(pipeline,'engineering_completeness',lambda *args:gate)
    monkeypatch.setattr(pipeline,'verify_source_repairs',lambda *args,**kwargs:[])
    monkeypatch.setattr(source_audit,'known_overlap_freeze_refs',lambda **kwargs:[overlap])
    if include_overlap:scoring.verify_engineering_barrier({'files':f,'run':run,'native':{}})
    else:
        with pytest.raises(ValueError,match='source-overlap review/evidence'):
            scoring.verify_engineering_barrier({'files':f,'run':run,'native':{}})
    assert not f.unsealed


@pytest.mark.parametrize('fault', ['none','receipt_lock','source_lock','audit_lock','regimen','source_identity','native_input','outcomes','model'])
def test_repeated_applications_require_source_bound_independent_audit(monkeypatch,fault):
    from recommendation_execution_repairs import repeated_application as expansion
    directory=Path('/synthetic/job');base=directory/'repeated_application'
    ref=lambda p:{'path':str(p),'sha256':'x'}
    plan={'source_path':'original.pksim5','source_sha256':'s','source_simulation_id':'sim','source_model_blob_sha256':'blob',
          'native_compound_name':'drug','implementation_sha256':{k:'code' for k in expansion.IMPLEMENTATION_FILES},
          'contract':{'dose_mg':30.,'administration_times_min':[0.,1440.]},
          'application_expansion':{'event_prefix':'Applications|Original|Application_1'}}
    source={'path':plan['source_path'],'sha256':'s','simulation_id':'sim','source_blob_sha256':'blob'}
    receipt={'version':expansion.VERSION,'status':'expanded_pending_native_readback_and_ode',
        'source_reference':source,'source_pkml':ref(directory/'export/original.pkml'),'pkml':ref(base/'expanded.pkml'),
        'implementation_sha256':dict(plan['implementation_sha256']),
        'files':{str(base/name):'x' for name in ('expansion_plan.json','semantic_audit.json','source_audit.json')},
        **{k:False for k in ('fitting_performed','ode_executed','sealed_observations_accessed','scientific_applicability_claimed')}}
    audit={'accepted':True,'full_model_unchanged':True};source_audit={'source':source,'model_accepted':True}
    docs={base/'receipt.json':receipt,base/'expansion_plan.json':{'compound_name':'drug','dose_mg':30.,'administration_times_min':[0.,1440.],'event_prefix':'Applications|Original|Application_1'},
        base/'semantic_audit.json':audit,base/'source_audit.json':source_audit,
        directory/'export/export_manifest.json':{'pkml':dict(receipt['source_pkml'])},
        directory/'registered_native_request.json':{'pkml':dict(receipt['pkml'])}}
    all_files=set(docs)|{Path(receipt['source_pkml']['path']),Path(receipt['pkml']['path'])}
    global_files=set(all_files);prediction_files=set(all_files)
    if fault=='receipt_lock':global_files.remove(base/'receipt.json')
    if fault=='source_lock':prediction_files.remove(Path(receipt['source_pkml']['path']))
    if fault=='audit_lock':prediction_files.remove(base/'semantic_audit.json')
    if fault=='regimen':docs[base/'expansion_plan.json']['dose_mg']=60.
    if fault=='source_identity':receipt['source_reference']={**source,'simulation_id':'other'}
    if fault=='native_input':docs[directory/'registered_native_request.json']['pkml']['path']='other.pkml'
    if fault=='outcomes':receipt['sealed_observations_accessed']=True
    class Files:
        root=Path('/synthetic');unsealed=False
        def checked(self,r):
            assert not self.unsealed
            return Path(r['path'])
        def read(self,p):
            assert not self.unsealed
            return docs[p]
    calls=[]
    monkeypatch.setattr(expansion,'_read_xml',lambda p:p)
    monkeypatch.setattr(expansion,'_verify_original_source',lambda *a:calls.append('source') or source_audit)
    monkeypatch.setattr(expansion,'audit_expansion',lambda *a:calls.append('model') or ({'accepted':False} if fault=='model' else audit))
    if fault=='none':
        scoring.verify_application_expansion(Files(),directory,plan,global_files,prediction_files)
        assert calls==['source','model']
    else:
        with pytest.raises(ValueError):scoring.verify_application_expansion(Files(),directory,plan,global_files,prediction_files)


@pytest.fixture
def complete_native_chain(tmp_path,monkeypatch):
    """Real JSON/hash fixtures; no R call and no clinical outcomes."""
    from recommendation_execution_repairs import native_pkml
    root=tmp_path;directory=root/'job';native=directory/'native';native.mkdir(parents=True)
    export=directory/'export';export.mkdir()
    source=export/'original.pkml';source.write_text('<Model/>')
    before=native/'before.pkml';before.write_text('<Model before="true"/>')
    after=native/'after.pkml';after.write_text('<Model after="true"/>')
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def save(p,d):p.write_text(json.dumps(d,sort_keys=True))
    def ref(p):return {'path':str(p.relative_to(root)),'sha256':sha(p)}
    worker=native_pkml.WORKER;adapter='scripts/recommendation_execution_repairs/native_pkml.py'
    plan={'adapter_route':'complete_original_pksim5','application_expansion':None,
          'contract':{'dose_mg':30.,'administration_times_min':[0.]},
          'output_path':'Organism|Drug|Plasma','molecular_weight_g_per_mol':300.,
          'source_metadata_sha256':'metadata','implementation_sha256':{worker:'a'*64,adapter:'b'*64}}
    registered={'pkml':ref(source),'patches':[],'output_path':plan['output_path'],
                'output_times_min':[0.,60.],'molecular_weight_g_per_mol':300.,'source_metadata_sha256':'metadata'}
    request={'version':native_pkml.VERSION,'action':'run','pkml_path':str(source),'pkml_sha256':sha(source),
             'patches':[],'output_path':plan['output_path'],'output_times_min':[0.,60.],
             'result_path':str(native/'native_result.json'),'before_pkml_path':str(before),'after_pkml_path':str(after)}
    save(directory/'registered_native_request.json',registered);save(native/'request.json',request)
    raw={'version':native_pkml.VERSION,'action':'run','pkml_sha256':sha(source),
         'request_sha256':sha(native/'request.json'),'solver_calls':1,
         'sealed_observations_accessed':False,'fitting_performed':False,'parameter_changes':[],
         'output_grid_verified':True,'before_pkml_sha256':sha(before),'after_pkml_sha256':sha(after),
         'curve':{'time_min':[0.,60.],'concentration':[0.,1.],'time_unit':'min','unit':'umol/l',
                  'dimension':'Concentration (molar)','output_path':plan['output_path']}}
    result={'version':native_pkml.VERSION,'action':'run','status':'prediction_locked','source_sha256':sha(source),
            'native':deepcopy(raw),'worker_sha256':'a'*64,'python_sha256':'b'*64,
            'process_manifest':str((native/'process_manifest.json').relative_to(root)),
            'source_import_audit':{'accepted':True},'model_preservation_audit':{'accepted':True},
            'sealed_observations_accessed':False,'fitting_performed':False}
    process={'status':'completed','returncode':0,'script':str(root/worker),'script_sha256':'a'*64,
             'runtime_unchanged':True,'registry_modified':False,
             'command':[str(root/'runtime/Rscript'),'--vanilla',str(root/worker),str(native/'request.json'),sha(native/'request.json')]}
    save(native/'native_result.json',raw);save(native/'result.json',result);save(native/'process_manifest.json',process)
    save(export/'export_manifest.json',{'pkml':ref(source)})
    save(directory/'prediction.json',{'version':'outer-adapter','native_curve':raw['curve'],'endpoints':{'Cmax':1.}})
    monkeypatch.setattr(scoring.complete_candidate,'output_grid',lambda contract:[0.,60.])
    monkeypatch.setattr(scoring.complete_candidate,'prediction_from_curve',lambda curve,plan:
                        {'version':'inner-curve-converter','native_curve':curve,'endpoints':{'Cmax':max(curve['concentration'])}})
    def seal():
        paths=[native/x for x in ['request.json','native_result.json','result.json','process_manifest.json','before.pkml','after.pkml']]
        save(native/'artifact_lock.json',{'version':native_pkml.VERSION,'prediction_accepted':True,
            'source':ref(source),'files':{str(p.relative_to(root)):sha(p) for p in paths}})
        f=scoring.backend.Files(root)
        refs=[ref(p) for p in directory.rglob('*') if p.is_file()]
        locked={f.checked(r) for r in refs}
        return f,locked
    f,locked=seal()
    return {'root':root,'directory':directory,'plan':plan,'f':f,'locked':locked,
            'save':save,'sha':sha,'seal':seal,'request':request,'raw':raw,'result':result,'registered':registered,'process':process}


@pytest.mark.parametrize('with_expansion',[False,True])
def test_complete_request_chain_accepts_actual_run_with_or_without_expansion(complete_native_chain,with_expansion):
    c=complete_native_chain
    if with_expansion:c['plan']['application_expansion']={'registered':True}
    result=scoring.verify_complete_native_binding(c['f'],c['directory'],c['plan'],c['locked'],c['locked'])
    assert result['accepted'] and result['curve_and_endpoints_recomputed']
    assert c['f'].unsealed is False


@pytest.mark.parametrize('file', ['native/request.json','native/native_result.json','native/result.json',
    'native/process_manifest.json','native/artifact_lock.json','native/before.pkml','native/after.pkml',
    'registered_native_request.json','prediction.json'])
@pytest.mark.parametrize('which',['global','prediction'])
def test_actual_native_files_need_both_outer_locks(complete_native_chain,file,which):
    c=complete_native_chain;global_files=set(c['locked']);prediction_files=set(c['locked'])
    (global_files if which=='global' else prediction_files).remove(c['directory']/file)
    with pytest.raises(ValueError,match='both locks'):
        scoring.verify_complete_native_binding(c['f'],c['directory'],c['plan'],global_files,prediction_files)
    assert not c['f'].unsealed


@pytest.mark.parametrize('fault',['request_input','native_input','request_sha','result_source','nested_result',
    'patches','output','grid','native_curve_grid','native_curve_output','serialized_model','worker','command',
    'destination','solver_count','outcomes','prediction_curve','prediction_endpoint','export_input'])
def test_relocked_wrong_native_chain_cannot_impersonate_registered_run(complete_native_chain,fault):
    c=complete_native_chain;d=c['directory'];n=d/'native';save=c['save']
    if fault=='request_input':c['request']['pkml_sha256']='another-source'
    if fault=='native_input':c['raw']['pkml_sha256']='another-source'
    if fault=='request_sha':c['raw']['request_sha256']='another-request'
    if fault=='result_source':c['result']['source_sha256']='another-source'
    if fault=='nested_result':c['result']['native']['solver_calls']=2
    if fault=='patches':c['request']['patches']=[{'path':'unregistered'}]
    if fault=='output':c['request']['output_path']='another-output'
    if fault=='grid':c['request']['output_times_min']=[0.,120.]
    if fault=='native_curve_grid':c['raw']['curve']['time_min']=[0.,120.]
    if fault=='native_curve_output':c['raw']['curve']['output_path']='another-output'
    if fault=='serialized_model':c['raw']['before_pkml_sha256']='another-model'
    if fault=='worker':c['result']['worker_sha256']='another-worker'
    if fault=='command':c['process']['command'][-1]='another-request'
    if fault=='destination':c['request']['result_path']=str(n/'another-result.json')
    if fault=='solver_count':c['raw']['solver_calls']=0
    if fault=='outcomes':c['raw']['sealed_observations_accessed']=True
    if fault=='prediction_curve':
        p=json.loads((d/'prediction.json').read_text());p['native_curve']['concentration']=[0.,2.];save(d/'prediction.json',p)
    if fault=='prediction_endpoint':
        p=json.loads((d/'prediction.json').read_text());p['endpoints']['Cmax']=2.;save(d/'prediction.json',p)
    if fault=='export_input':save(d/'export/export_manifest.json',{'pkml':{'path':'another-model','sha256':'x'}})
    save(n/'request.json',c['request'])
    # Deliberately rebind all file hashes and nested payloads. Pure checksum
    # validation must not conceal the scientific/causal mismatch.
    if fault!='request_sha':c['raw']['request_sha256']=c['sha'](n/'request.json')
    if fault!='command':c['process']['command'][-1]=c['sha'](n/'request.json')
    if fault!='nested_result':c['result']['native']=deepcopy(c['raw'])
    save(n/'native_result.json',c['raw']);save(n/'result.json',c['result']);save(n/'process_manifest.json',c['process'])
    f,locked=c['seal']()
    with pytest.raises(ValueError):scoring.verify_complete_native_binding(f,d,c['plan'],locked,locked)
    assert f.unsealed is False


@pytest.fixture
def snapshot_native_chain(tmp_path, monkeypatch):
    from pbpk_kb.blueprints import design_runtime, source_adaptation_audit as audit
    root=tmp_path;d=root/'job';n=d/'native';r=d/'readback'
    (n/'outputs').mkdir(parents=True);r.mkdir()
    def write(p,value):
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def ref(p):return {'path':str(p.relative_to(root)),'sha256':sha(p)}
    worker=design_runtime._RUNNER;reader=audit._R_SCRIPT
    for path in (root/worker,root/reader,root/'runtime/ospsuite-env/bin/Rscript'):write(path,'synthetic-runtime')
    binding={'locked_resources':[ref(root/'runtime/ospsuite-env/bin/Rscript')],'engine':'synthetic'}
    monkeypatch.setattr(design_runtime,'_runtime_binding',lambda root:deepcopy(binding))
    monkeypatch.setattr(audit,'build_expectations',lambda snapshot,expectations:expectations)
    monkeypatch.setattr(audit,'assess_native_readback',lambda raw,request:
        {'accepted':raw.get('readback_valid') is True,'reasons':[]})
    output='Organism|PeripheralVenousBlood|Drug|Plasma (Peripheral Venous Blood)'
    snapshot={'Simulations':[{'Name':'s','OutputSelections':[output]}]}
    source=root/'source.json';write(source,{'original':'source'})
    plan={'candidate_id':'c','scenario_id':'q','snapshot':snapshot,'snapshot_sha256':scoring.backend.digest(snapshot),
        'source_path':'source.json','source_sha256':sha(source),'execution_source_path':'source.json','execution_source_sha256':sha(source),
        'implementation_sha256':{worker:sha(root/worker),reader:sha(root/reader)},
        'output_path':output,'molecular_weight_g_per_mol':100.,'compatibility':{},
        'contract':{'matrix':'plasma','sampling_times_min':[0.,30.,60.]},
        'native_expectations':{'expectations':[],'expected_event_count':1}}
    write(n/'snapshot.json',snapshot);write(n/'outputs/model.pkml','<Model/>')
    csv=n/'outputs/model-Results.csv'
    write(csv,'IndividualId,Time [min],'+output+' [umol/l]\n1,0,0\n1,30,2\n1,60,1\n')
    def process(folder,script,args):
        return {'status':'completed','returncode':0,'runtime_unchanged':True,'registry_modified':False,'stop_reason':None,
            'script':script,'script_sha256':sha(root/script),'process_manifest_path':str(folder/'process_manifest.json'),
            'runtime_binding':deepcopy(binding),'command':[str(root/'runtime/ospsuite-env/bin/Rscript'),'--vanilla',str(root/script),*args]}
    proc=process(n,worker,[str(n/'snapshot.json'),str(n/'outputs')])
    aproc=process(r,reader,[str(r/'request.json'),'initial'])
    manifest={'status':'engineering_passed','source_unchanged':True,'engineering_accepted':True,'registry_modified':False,
        'manifest_path':str(n/'execution_manifest.json'),'runtime_binding':deepcopy(binding),'expected_simulations':['s'],
        'pkml_outputs':[{'path':'outputs/model.pkml','sha256':sha(n/'outputs/model.pkml')}],
        'curve_outputs':[{'path':'outputs/model-Results.csv','sha256':sha(csv)}]}
    request={'adapter_version':audit.AUDIT_VERSION,'pkml_path':str(n/'outputs/model.pkml'),
        'pkml_sha256':sha(n/'outputs/model.pkml'),'result_path':str(r/'native_readback.json'),
        'expected_event_count':1,'expectations':[]}
    raw={'readback_valid':True}
    report={'status':'passed','accepted':True,'reasons':[],'solver_calls':0,'model_modified':False,
            'execution_manifest':str(n/'execution_manifest.json'),'native_readback_file':str(r/'native_readback.json')}
    curve=scoring.snapshot_candidate.read_curve(n,manifest,output)
    prediction=scoring.snapshot_candidate.prediction_from_curve(curve,plan)
    def seal():
        manifest['source_sha256']=request['source_sha256']=report['source_sha256']=sha(n/'snapshot.json')
        manifest['process']=deepcopy(proc)
        write(n/'process_manifest.json',proc);write(n/'execution_manifest.json',manifest)
        write(r/'request.json',request);raw['request_sha256']=sha(r/'request.json')
        aproc['command'][-1]=sha(r/'request.json');report['process']=deepcopy(aproc)
        write(r/'native_readback.json',raw);write(r/'process_manifest.json',aproc)
        report.update(request_sha256=sha(r/'request.json'),execution_manifest_sha256=sha(n/'execution_manifest.json'),
            pkml_sha256=sha(n/'outputs/model.pkml'),native_readback_sha256=sha(r/'native_readback.json'))
        write(r/'audit.json',report);write(d/'readback_result.json',report);write(d/'prediction.json',prediction)
        f=scoring.backend.Files(root);paths={p.resolve() for p in d.rglob('*') if p.is_file()}
        for p in paths:f.checked(ref(p))
        return f,paths
    return dict(root=root,directory=d,plan=plan,seal=seal,write=write,manifest=manifest,request=request,
                raw=raw,report=report,process=proc,audit_process=aproc,prediction=prediction,csv=csv)


def test_snapshot_full_native_chain_recomputes_every_prediction_field(snapshot_native_chain):
    c=snapshot_native_chain;f,files=c['seal']()
    result=scoring.verify_snapshot_native_binding(f,c['directory'],c['plan'],files,files)
    assert result['accepted'] and result['curve_and_endpoints_recomputed'] and not f.unsealed


@pytest.mark.parametrize('missing', ['native/snapshot.json','native/process_manifest.json','readback/request.json',
    'readback/native_readback.json','readback/process_manifest.json','readback/audit.json','readback_result.json',
    'native/outputs/model-Results.csv','native/outputs/model.pkml','prediction.json'])
def test_snapshot_requires_both_locks_for_actual_native_evidence(snapshot_native_chain,missing):
    c=snapshot_native_chain;f,files=c['seal']();inner=files-{c['directory']/missing}
    with pytest.raises(ValueError,match='both locks'):
        scoring.verify_snapshot_native_binding(f,c['directory'],c['plan'],files,inner)
    assert not f.unsealed


@pytest.mark.parametrize('fault',['input','worker','command','runtime','output','readback_input','readback_expectations',
    'readback_failed','readback_solver','prediction_point','prediction_matrix','prediction_registration','prediction_endpoint','csv_point'])
def test_relocked_snapshot_mismatches_fail_semantic_chain(snapshot_native_chain,fault):
    c=snapshot_native_chain;d=c['directory'];p=c['prediction']
    if fault=='input':c['write'](d/'native/snapshot.json',{'Simulations':[]})
    if fault=='worker':c['process']['script_sha256']='0'*64
    if fault=='command':c['process']['command'][-2]=str(d/'different.json')
    if fault=='runtime':c['process']['runtime_binding']={'locked_resources':[]}
    if fault=='output':c['plan']['output_path']='Organism|Wrong|Plasma'
    if fault=='readback_input':c['request']['pkml_path']=str(d/'other.pkml')
    if fault=='readback_expectations':c['request']['expectations']=[{'path':'changed'}]
    if fault=='readback_failed':c['raw']['readback_valid']=False
    if fault=='readback_solver':c['report']['solver_calls']=1
    if fault=='prediction_point':p['concentration_ng_ml'][1]=999.
    if fault=='prediction_matrix':p['matrix']='whole_blood'
    if fault=='prediction_registration':p.pop('sampling_grid_registration')
    if fault=='prediction_endpoint':p['endpoints']['sampled_Cmax']['value']=999.
    if fault=='csv_point':
        text=c['csv'].read_text().replace('1,30,2','1,30,3');c['write'](c['csv'],text)
        c['manifest']['curve_outputs'][0]['sha256']=hashlib.sha256(c['csv'].read_bytes()).hexdigest()
    f,files=c['seal']()
    with pytest.raises((ValueError,KeyError)):
        scoring.verify_snapshot_native_binding(f,d,c['plan'],files,files)
    assert not f.unsealed
