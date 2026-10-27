"""Native source-migration numerical invariants and bounded orchestration."""
import importlib.util
import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('candidate_migration',ROOT/'scripts/recommendation_candidate_migration.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def example():
    return {'Version':79,'Compounds':[{'Name':'Drug','Parameters':[{'Name':'CL','Value':2.,'Unit':'L/h'}]}],
        'Simulations':[{'Name':'Dose','Model':'4Comp','Parameters':[{'Path':'Applications|dose|Dose','Value':10.,'Unit':'mg'}]}],
        'ObservedData':[{'secret_concentration':1}], 'ParameterIdentifications':[{'fake':1}]}


def test_real_mapper_known_path_upgrade_and_version_only_are_allowed():
    a=example();b=deepcopy(a);b['Version']=80;b['Simulations'][0]['Parameters'][0]['Path']='Events|dose|Dose'
    r=m.compare_migration(a,b)
    assert r['accepted'] and r['numeric_checks'] == 2
    clean=m.clean_snapshot(a)
    assert 'ObservedData' not in clean and 'ParameterIdentifications' not in clean
    assert clean['Simulations'][0]['HasResults'] is False
    assert a['ObservedData']


@pytest.mark.parametrize('mutation',[
    lambda b:b['Compounds'][0]['Parameters'][0].update(Value=2.01),
    lambda b:b['Compounds'][0]['Parameters'][0].update(Unit='L/min'),
    lambda b:b['Compounds'][0].update(NewMechanism=True),
    lambda b:b['Simulations'][0].update(Model='NewModel'),
    lambda b:b.update(Compounds=[]),
])
def test_changed_source_parameters_or_structure_fail(mutation):
    a=example();b=deepcopy(a);mutation(b)
    assert m.compare_migration(a,b)['accepted'] is False


def test_named_order_preserved_semantics_only():
    a=example();a['Compounds'].append({'Name':'Drug2','Parameters':[]});b=deepcopy(a);b['Compounds'].reverse()
    assert m.compare_migration(a,b)['accepted']


def test_stop_accounts_for_all_pending_sources(tmp_path,monkeypatch):
    tasks=[{'source_sha256':str(i)*64,'candidate_ids':[str(i)]} for i in [1,2,3]]
    monkeypatch.setattr(m,'collect_tasks',lambda *a,**kw:tasks)
    stop=tmp_path/'stop.requested';stop.write_text('stop')
    out=tmp_path/'migrations'
    result=m.prepare_migrations([{}],[],out,stop_file=stop,project_root=ROOT)
    assert result['status']=='stopped' and len(result['results'])==3
    assert all(r['status']=='not_started' for r in result['results'])
    assert json.loads((out/'migration_summary.json').read_text())==result


def test_frozen_cohort_unique_migration_tasks():
    rows=[json.loads(l) for l in (ROOT/'experiments/model_recommendation/execution_validation_20261001/inputs/candidate_audit.jsonl').open()]
    tasks=m.collect_tasks(rows)
    assert len(tasks)==67 and len({t['source_sha256'] for t in tasks})==67
    assert sum(t['format']=='.json' for t in tasks)==38
    assert sum(t['format']=='.pksim5' for t in tasks)==29


def receipt_fixture(tmp_path):
    source=tmp_path/'original.json';source.write_text('{"Version":79}')
    provenance={'source_path':str(source.relative_to(ROOT)),'source_sha256':m.sha(source)}
    base=tmp_path/'migrations';directory=base/provenance['source_sha256'];directory.mkdir(parents=True)
    converted=directory/'converted_snapshot.json';converted.write_text('{"Version":80}')
    process=directory/'process_manifest.json'
    process.write_text(json.dumps({'status':'completed','returncode':0,'script_sha256':m.sha(ROOT/m.R_SCRIPT)}))
    receipt={'version':m.VERSION,'status':'converted','source_sha256':provenance['source_sha256'],
        'structural_comparison_passed':True,'comparison':{'accepted':True},'ode_run':False,'fitting_performed':False,
        'converted_source_path':str(converted.relative_to(ROOT)),'converted_source_sha256':m.sha(converted),
        'process_manifest':str(process.relative_to(ROOT)),'process_manifest_sha256':m.sha(process),
        'worker_script_sha256':m.sha(ROOT/m.R_SCRIPT),'python_adapter_sha256':m.sha(Path(m.__file__)),
        'runtime_adapter_sha256':m.sha(ROOT/m.RUNTIME_ADAPTER),
        'comparison_scope':'test fixture','native_invocations':1,
        'operation':'bounded_native_stage_retry_or_continuation',
        'stages':[{'mode':'snapshot_to_project','process_manifest':m.reference(process),'input':m.reference(source),
                   'copied_input_sha256_before':m.sha(source),'copied_input_sha256_after':m.sha(source),
                   'copied_input_sha256':m.sha(source),'copied_input_changed':False,
                   'source_input_sha256_after':m.sha(source),'source_input_unchanged':True,
                   'output':m.reference(converted)}]}
    receipt_path=directory/'receipt.json';receipt_path.write_text(json.dumps(receipt))
    return provenance,base,receipt_path,receipt,converted


def test_complete_receipt_checksum_chain_resolves(tmp_path):
    p,base,path,r,converted=receipt_fixture(tmp_path)
    resolved=m.resolve_migrated_source(p,migration_root=base)
    assert resolved['path']==converted and resolved['receipt_sha256']==m.sha(path)


@pytest.mark.parametrize('field,value',[
    ('python_adapter_sha256','0'*64),('worker_script_sha256','0'*64),
    ('structural_comparison_passed',False),('ode_run',True),('status','failed'),
])
def test_tampered_or_outdated_receipt_rejected(tmp_path,field,value):
    p,base,path,r,converted=receipt_fixture(tmp_path)
    r[field]=value;path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='MIGRATION_'):
        m.resolve_migrated_source(p,migration_root=base)


def test_converted_model_byte_drift_rejected(tmp_path):
    p,base,path,r,converted=receipt_fixture(tmp_path)
    converted.write_text('{"Version":80,"different":true}')
    with pytest.raises(ValueError,match='CHECKSUM'):
        m.resolve_migrated_source(p,migration_root=base)


def test_heterogeneous_process_multiset_and_parameter_paths_are_order_independent():
    a=example()
    a['Simulations'][0]['Compounds']=[{'Name':'Drug','Processes':[
        {'MoleculeName':'P-gp'}, {'SystemicProcessType':'GFR','Name':'GFR-study'}]}]
    a['Simulations'][0]['Parameters']=[{'Path':'Applications|dose|Dose','Value':10,'Unit':'mg'},
                                       {'Path':'Drug|Permeability','Value':1e-6,'Unit':'dm/min'}]
    b=deepcopy(a)
    b['Simulations'][0]['Compounds'][0]['Processes'].reverse()
    b['Simulations'][0]['Parameters'][0]['Path']='Events|dose|Dose'
    b['Simulations'][0]['Parameters'].reverse()
    assert m.compare_migration(a,b)['accepted']
    b['Simulations'][0]['Compounds'][0]['Processes'].append({'MoleculeName':'P-gp'})
    assert not m.compare_migration(a,b)['accepted']


def test_parameter_deletion_change_and_unknown_sequence_reordering_stay_rejected():
    a=example();a['Simulations'][0]['Parameters']=[{'Path':'Drug|pH penalty','Value':0},
                                                {'Path':'Drug|CL','Value':3,'Unit':'L/h'}]
    b=deepcopy(a);b['Simulations'][0]['Parameters'].pop(0)
    assert not m.compare_migration(a,b)['accepted']
    b=deepcopy(a);b['Simulations'][0]['Parameters'][1]['Value']=4
    assert not m.compare_migration(a,b)['accepted']
    a['UnknownSequence']=[1,2];b=deepcopy(a);b['UnknownSequence'].reverse()
    assert not m.compare_migration(a,b)['accepted']
    a=example();a['Simulations'][0]['Parameters']=[{'Path':'Drug|CL','Value':3}, {'Path':'Drug|CL','Value':4}]
    b=deepcopy(a);b['Simulations'][0]['Parameters'].reverse()
    assert not m.compare_migration(a,b)['accepted']


def test_archived_native_outputs_recover_only_proved_order_equivalence():
    rows=[json.loads(l) for l in (ROOT/'experiments/model_recommendation/execution_validation_20261001/inputs/candidate_audit.jsonl').open()]
    tasks=m.collect_tasks(rows)
    targets={
        '86c9cc70124cf03676699d65b2684030c8bf00ea8a7f4440c2d0899ebb1ac476':True,
        '0ca26837a6c1007854d95e660e8f4ed43d73eb333ce6b59cb361109118f90923':True,
        '311258546ae429125cdcd8c086c93030e1edf11591179c76efa748180e669809':True,
        'dc17440f192dd986c0e4e91eb9cf94db4d4c1d6830afa36921e1d6a32016e2b7':True,
        '7b45f757f4ee1480553c5b513da9760e7c0cdee2d8bf481ed88c942590a1bd70':False,
        '86ab3bd922d07bf7b13220f883fc947fe633e412caf368a4089a97cad73d5cc6':False,
    }
    for task in tasks:
        if task['source_sha256'] in targets:
            result=m.revalidate_old_artifact(task)
            assert result['comparison']['accepted'] is targets[task['source_sha256']]


def test_prior_success_revalidation_is_zero_new_native_calls(tmp_path,monkeypatch):
    rows=[json.loads(l) for l in (ROOT/'experiments/model_recommendation/execution_validation_20261001/inputs/candidate_audit.jsonl').open()]
    task=next(t for t in m.collect_tasks(rows) if t['source_sha256']=='999157ffa319ca75e34da85a548fb3e212be6212e2279b32a4f6ff6ea6f4e436')
    monkeypatch.setattr(m,'run_bounded_r',lambda **kw:pytest.fail('Accepted archived artifact must not be rerun'))
    result=m.migrate_one(task,tmp_path/task['source_sha256'])
    assert result['status']=='converted' and result['native_invocations']==0
    assert result['operation']=='archived_artifact_revalidation'
    assert result['inherited_original_comparison']['accepted']
    resolved=m.resolve_migrated_source(task,migration_root=tmp_path)
    assert resolved['sha256']==result['converted_source_sha256']


def test_stage_uses_remaining_total_budget_and_binds_resources(tmp_path,monkeypatch):
    source=tmp_path/'input.json';source.write_text('{"Version":79}')
    captured={}
    def native(**kwargs):
        captured.update(kwargs)
        out=Path(kwargs['args'][2])/'model.pksim5';out.write_bytes(b'test native artifact')
        process={'status':'completed','returncode':0,'script_sha256':m.sha(ROOT/m.R_SCRIPT),'stop_reason':None}
        m.save(kwargs['output_dir']/'process_manifest.json',process)
        return process
    monkeypatch.setattr(m,'run_bounded_r',native)
    result={'stages':[],'native_invocations':0}
    stop=tmp_path/'stop.requested'
    output=m._native_stage('snapshot_to_project',source,tmp_path,result,root=ROOT,
                           timeout_seconds=600,deadline=m.time.monotonic()+90,stop_file=stop)
    assert 1<=captured['timeout_seconds']<=90 and captured['cpu_threads']==2
    assert result['native_invocations']==1 and result['stages'][0]['output']['sha256']==m.sha(output)
    assert result['stages'][0]['input']['sha256']==m.sha(source)
    assert captured['stop_file']==stop


def test_completed_archived_first_stage_resumes_only_export(tmp_path,monkeypatch):
    rows=[json.loads(l) for l in (ROOT/'experiments/model_recommendation/execution_validation_20261001/inputs/candidate_audit.jsonl').open()]
    task=next(t for t in m.collect_tasks(rows) if t['source_sha256']=='079e9a826bc46d7ca9a5c0633baf575b901d27693c4d96d52a1e07dea45dc471')
    prior=m.revalidate_old_artifact(task);calls=[]
    def stage(mode,input_path,directory,result,**kwargs):
        calls.append((mode,input_path))
        fake=m.clean_snapshot(json.loads((ROOT/task['source_path']).read_text()));fake['Version']=80
        out=tmp_path/'output.json';m.save(out,fake)
        return out
    monkeypatch.setattr(m,'_native_stage',stage)
    result={'stages':[],'native_invocations':0}
    a,b=m._continue_or_retry(task,prior,tmp_path,result,root=ROOT,timeout_seconds=600)
    assert len(calls)==1 and calls[0][0]=='project_to_snapshot'
    assert str(calls[0][1]).endswith('/project/model.pksim5')
    assert result['continued_from']=='archived_completed_snapshot_to_project_stage'
    assert m.compare_migration(a,b)['accepted']


def test_real_unresolved_parameter_drop_never_triggers_blind_native_retry(tmp_path,monkeypatch):
    rows=[json.loads(l) for l in (ROOT/'experiments/model_recommendation/execution_validation_20261001/inputs/candidate_audit.jsonl').open()]
    task=next(t for t in m.collect_tasks(rows) if t['source_sha256']=='7b45f757f4ee1480553c5b513da9760e7c0cdee2d8bf481ed88c942590a1bd70')
    monkeypatch.setattr(m,'run_bounded_r',lambda **kw:pytest.fail('Real source differences cannot be retried blindly'))
    result=m.migrate_one(task,tmp_path/task['source_sha256'])
    assert result['status']=='failed' and result['native_invocations']==0
    assert any(d['kind']=='list_length' and d['path'].endswith('/Parameters')
               for d in result['comparison']['differences'])


def test_archive_implementation_drift_fails_before_revalidation(monkeypatch):
    rows=[json.loads(l) for l in (ROOT/'experiments/model_recommendation/execution_validation_20261001/inputs/candidate_audit.jsonl').open()]
    task=m.collect_tasks(rows)[0];real_sha=m.sha
    monkeypatch.setattr(m,'sha',lambda p:'0'*64 if str(p).endswith('frozen_files/scripts/recommendation_candidate_migration.py') else real_sha(p))
    with pytest.raises(ValueError,match='BOUND_RESOURCE_CHANGED'):
        m.revalidate_old_artifact(task)


def sqlite_project(path):
    with sqlite3.connect(path) as database:
        database.execute('CREATE TABLE INDIVIDUALS (Id TEXT)')
    return path


def schema_maintenance(path):
    # This is the native working-copy schema maintenance observed in the
    # archived Buprenorphine input; no model parameters/outcomes are needed.
    with sqlite3.connect(path) as database:
        database.execute('ALTER TABLE INDIVIDUALS ADD COLUMN ExpressionProfileIds TEXT')
        database.execute('CREATE TABLE EXPRESSION_PROFILES (Id TEXT)')


def fake_native_process(kwargs, *, timeout=False):
    process={'status':'stopped' if timeout else 'completed','returncode':143 if timeout else 0,
             'script_sha256':m.sha(ROOT/m.R_SCRIPT),'stop_reason':'EXECUTION_TIMEOUT' if timeout else None}
    m.save(kwargs['output_dir']/'process_manifest.json',process)
    return process


def test_native_sqlite_working_copy_maintenance_preserves_original_and_is_audited(tmp_path,monkeypatch):
    source=sqlite_project(tmp_path/'source.pksim5');original=m.sha(source)
    def native(**kwargs):
        schema_maintenance(Path(kwargs['args'][1])/'model.pksim5')
        m.save(Path(kwargs['args'][2])/'model.json',{'Version':80})
        return fake_native_process(kwargs)
    monkeypatch.setattr(m,'run_bounded_r',native)
    result={'stages':[],'native_invocations':0}
    m._native_stage('project_to_snapshot',source,tmp_path,result,root=ROOT,
                    timeout_seconds=600,deadline=m.time.monotonic()+1200)
    row=result['stages'][0]
    assert m.sha(source)==original==row['copied_input_sha256_before']==row['source_input_sha256_after']
    assert row['source_input_unchanged'] and row['copied_input_changed']
    assert row['copied_input_sha256_after']!=original


def test_native_source_drift_remains_rejected_even_when_copy_maintenance_is_allowed(tmp_path,monkeypatch):
    source=sqlite_project(tmp_path/'source.pksim5')
    def native(**kwargs):
        schema_maintenance(source)
        m.save(Path(kwargs['args'][2])/'model.json',{'Version':80})
        return fake_native_process(kwargs)
    monkeypatch.setattr(m,'run_bounded_r',native)
    result={'stages':[],'native_invocations':0}
    with pytest.raises(ValueError,match='Native stage source changed'):
        m._native_stage('project_to_snapshot',source,tmp_path,result,root=ROOT,
                        timeout_seconds=600,deadline=m.time.monotonic()+1200)
    assert result['native_invocations']==1 and result['stages'][0]['source_input_unchanged'] is False


def test_native_stage_timeout_is_accounted_and_does_not_accept_partial_export(tmp_path,monkeypatch):
    source=sqlite_project(tmp_path/'source.pksim5');original=m.sha(source)
    captured={}
    def native(**kwargs):
        captured.update(kwargs)
        schema_maintenance(Path(kwargs['args'][1])/'model.pksim5')
        m.save(Path(kwargs['args'][2])/'model.json',{'Version':80})
        return fake_native_process(kwargs,timeout=True)
    monkeypatch.setattr(m,'run_bounded_r',native)
    result={'stages':[],'native_invocations':0}
    with pytest.raises(ValueError,match='NATIVE_STAGE_FAILED:project_to_snapshot:EXECUTION_TIMEOUT'):
        m._native_stage('project_to_snapshot',source,tmp_path,result,root=ROOT,
                        timeout_seconds=600,deadline=m.time.monotonic()+90)
    assert 1<=captured['timeout_seconds']<=90 and m.sha(source)==original
    assert result['native_invocations']==1 and result['stages'][0]['output'] is None


def test_changed_archived_sqlite_copy_reexports_verified_original_and_retains_full_check(tmp_path,monkeypatch):
    rows=[json.loads(l) for l in (ROOT/'experiments/model_recommendation/execution_validation_20261001/inputs/candidate_audit.jsonl').open()]
    task=next(t for t in m.collect_tasks(rows) if t['source_sha256']=='3bfbaec480ede5e15b560455dd6b49fedc06b1347b71c2c8ead840bfebc7e2c1')
    prior=m.revalidate_old_artifact(task);calls=[]
    original_hash=m.sha(ROOT/task['source_path'])
    assert m.sha(prior['directory']/'input/model.pksim5')!=original_hash
    def stage(mode,input_path,directory,result,**kwargs):
        calls.append((mode,input_path,kwargs['deadline']))
        out=tmp_path/('new_'+str(len(calls))+('.pksim5' if mode=='snapshot_to_project' else '.json'))
        if mode=='snapshot_to_project':out.write_bytes(b'fresh project')
        else:
            document=example();document['Version']=80
            if len(calls)==3:document['Compounds'][0]['Parameters'][0]['Value']=9
            m.save(out,document)
        return out
    monkeypatch.setattr(m,'_native_stage',stage)
    result={'stages':[],'native_invocations':0}
    a,b=m._continue_or_retry(task,prior,tmp_path,result,root=ROOT,timeout_seconds=600)
    assert [c[0] for c in calls]==['project_to_snapshot','snapshot_to_project','project_to_snapshot']
    assert calls[0][1]==ROOT/task['source_path'] and len({c[2] for c in calls})==1
    assert m.sha(ROOT/task['source_path'])==original_hash
    assert result['archived_first_export_review']['old_export_reused'] is False
    assert result['continued_from']=='fresh_verified_original_project_export'
    assert m.compare_migration(a,b)['accepted'] is False  # Maintenance does not waive parameter checks.
    assert not any('/first_native_export/' in r['path'] for r in prior['bindings'])


def test_original_changed_before_retry_is_rejected_without_native_call(tmp_path,monkeypatch):
    source=sqlite_project(tmp_path/'source.pksim5')
    task={'source_path':str(source.relative_to(ROOT)),'source_sha256':m.sha(source)}
    schema_maintenance(source)
    monkeypatch.setattr(m,'_native_stage',lambda *a,**kw:pytest.fail('Changed original cannot execute'))
    with pytest.raises(ValueError,match='ORIGINAL_SOURCE_CHANGED_BEFORE_NATIVE_RETRY'):
        m._continue_or_retry(task,{'directory':tmp_path,'bindings':[],'receipt':{}},tmp_path,
                             {'stages':[],'native_invocations':0},root=ROOT,timeout_seconds=600)


def test_receipt_working_copy_before_after_binding_cannot_be_omitted(tmp_path):
    p,base,path,r,converted=receipt_fixture(tmp_path)
    r['stages'][0]['copied_input_sha256_before']='0'*64
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='MIGRATION_STAGE_INPUT_BINDING_DIFFERS'):
        m.resolve_migrated_source(p,migration_root=base)
