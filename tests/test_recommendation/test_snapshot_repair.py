"""Exact source preservation, schema mappings and immutable receipt gates."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from recommendation_execution_repairs import snapshot_repair as m
from recommendation_execution_repairs.snapshot_schema import map_legacy_expression_schema
from recommendation_execution_repairs.snapshot_readback import omitted_fields


def test_exact_comparison_never_uses_legacy_numeric_tolerance():
    a={'Version':79,'Compounds':[{'Name':'Drug','Parameters':[{'Name':'clearance','Value':1.0}]}]}
    b=deepcopy(a);b['Version']=80
    assert m.exact_comparison(a,b)['accepted']
    b['Compounds'][0]['Parameters'][0]['Value']+=1e-14
    assert not m.exact_comparison(a,b)['accepted']
    assert not m.exact_comparison({'flag':True},{'flag':1})['accepted']


def test_completed_legacy_expression_exports_are_exactly_preserved():
    rows=[json.loads(x) for x in (ROOT/m.migration.OUT/'inputs/candidate_audit.jsonl').open()]
    prefixes=('0a4fbffe','3d0876ba','64730253','6a7f9778','6f2efe23','998eb58f',
              'c965f50e','ccc06c1b','d3b81c1b','e874015f','f6bf2b17')
    checked=0
    for task in m.migration.collect_tasks(rows):
        if not task['source_sha256'].startswith(prefixes):continue
        prior=m.migration.revalidate_old_artifact(task)
        mapped=map_legacy_expression_schema(prior['original'],prior['converted'])
        assert m.exact_comparison(mapped['mapped_snapshot'],prior['converted'])['accepted']
        assert mapped['changes'];checked+=1
        changed=deepcopy(prior['converted'])
        changed['Compounds'][0]['Name']+='unexpected-change'
        assert not m.exact_comparison(mapped['mapped_snapshot'],changed)['accepted']
    assert checked==11


def test_resolver_checks_original_output_and_implementation_bindings(tmp_path):
    source=tmp_path/'model.json';source.write_text(json.dumps({'Version':80,'Compounds':[],'Simulations':[]}))
    provenance={'source_path':str(source.relative_to(ROOT)),'source_sha256':m.sha(source)}
    out=tmp_path/'repairs';r=m.repair_snapshot(provenance,out/provenance['source_sha256'])
    assert r['status']=='verified'
    resolved=m.resolve_repaired_snapshot(provenance,repair_root=out)
    assert resolved['original_path']==source
    converted=resolved['path'];before=converted.read_bytes();converted.write_bytes(before+b' ')
    with pytest.raises(ValueError,match='CHANGED'):m.resolve_repaired_snapshot(provenance,repair_root=out)
    converted.write_bytes(before)
    receipt=out/provenance['source_sha256']/'receipt.json';data=json.loads(receipt.read_text())
    data['implementation'][0]['sha256']='0'*64;receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='IMPLEMENTATION_DRIFT'):m.resolve_repaired_snapshot(provenance,repair_root=out)
    source.write_text('{}')
    with pytest.raises(ValueError,match='SOURCE_SHA'):m.resolve_repaired_snapshot(provenance,repair_root=out)


def test_unsupported_legacy_localization_fails_closed():
    source={'Version':77,'Individuals':[{'Name':'I','OriginData':{'Species':'Human'},
        'Molecules':[{'Name':'E','Type':'Enzyme','Expression':[],
            'TissueLocation':'ExtracellularMembrane','MembraneLocation':'Apical',
            'IntracellularVascularEndoLocation':'Endosomal'}]}]}
    native={'Version':80,'Individuals':[{'Name':'I','ExpressionProfiles':['E|Human|I']}],
        'ExpressionProfiles':[{'Molecule':'E','Species':'Human','Category':'I'}]}
    with pytest.raises(ValueError,match='LOCALIZATION_COMBINATION'):map_legacy_expression_schema(source,native)


def test_readback_rejects_changed_values_and_non_dimensionless_omissions():
    p={'Path':'Drug|switch','Value':0,'ValueOrigin':{'Source':'Unknown'}}
    audit={'changes':[{'field':'Parameters','simulation_name':'S','source':[p],'native_before':[]}]}
    assert omitted_fields(audit)['S'][0]['record']==p
    audit['changes'][0]['native_before']=[{**p,'Value':1}]
    with pytest.raises(ValueError,match='CHANGED_PARAMETER'):omitted_fields(audit)
    audit['changes'][0]['native_before']=[];p['Unit']='mg'
    with pytest.raises(ValueError,match='DIMENSIONLESS'):omitted_fields(audit)


def test_stop_before_creation_never_leaves_failed_empty_repair(tmp_path):
    source=tmp_path/'model.json';source.write_text('{"Version":80}')
    provenance={'source_path':str(source.relative_to(ROOT)),'source_sha256':m.sha(source)}
    stop=tmp_path/'stop.requested';stop.touch();out=tmp_path/'must-not-exist'
    with pytest.raises(InterruptedError):m.repair_snapshot(provenance,out,stop_file=stop)
    assert not out.exists()
    summary=m.prepare_repairs([],output_dir=out,stop_file=stop)
    assert summary['status']=='stopped' and not summary['all_sources_accounted']
    assert not out.exists()


def test_native_partition_merge_requires_all_simulations_and_exact_global_parameters():
    source={'Version':79,'Compounds':[{'Name':'D','Parameters':[{'Name':'CL','Value':2}]}],
            'Simulations':[{'Name':'A'},{'Name':'B'}]}
    parts=[{**deepcopy(source),'Version':80,'Simulations':[deepcopy(s)]} for s in source['Simulations']]
    result=m.merge_native_partitions(source,parts)
    assert [s['Name'] for s in result['Simulations']]==['A','B']
    with pytest.raises(ValueError,match='COVERAGE_INCOMPLETE'):m.merge_native_partitions(source,parts[:1])
    with pytest.raises(ValueError,match='IDENTITY_DIFFERS'):m.merge_native_partitions(source,parts+parts[:1])
    parts[1]['Compounds'][0]['Parameters'][0]['Value']=2.00000000000001
    with pytest.raises(ValueError,match='GLOBAL_DEFINITIONS_DIFFER'):m.merge_native_partitions(source,parts)


def test_overlay_readback_cannot_promote_nonexact_global_document(tmp_path,monkeypatch):
    original={'Version':79,'Compounds':[{'Name':'Drug','Parameters':[{'Name':'clearance','Value':1.0}]}],
              'Simulations':[{'Name':'S','Compounds':[{'Name':'Drug'}],
                              'Parameters':[{'Path':'Drug|switch','Value':0}]}]}
    native=deepcopy(original);native['Version']=80
    native['Compounds'][0]['Parameters'][0]['Value']+=1e-14
    native['Simulations'][0].pop('Parameters')
    source=tmp_path/'source.json';source.write_text(json.dumps(original))
    provenance={'source_path':str(source.relative_to(ROOT)),'source_sha256':m.sha(source)}
    monkeypatch.setattr(m,'_native_base',lambda *a,**kw:(native,[],'test-native-export'))
    monkeypatch.setattr(m,'validate_overlay',lambda *a,**kw:({'accepted':True},[]))
    bad_restored=deepcopy(original);bad_restored['Version']=80
    bad_restored['Compounds'][0]['Parameters'][0]['Value']+=1e-14
    monkeypatch.setattr(m,'repair_complex_snapshot',lambda *a,**kw:{
        'proof':{'accepted':True},'snapshot':bad_restored,'provenance_refs':[]})
    result=m.repair_snapshot(provenance,tmp_path/'repair')
    assert result['status']=='failed'
    assert result['comparison']['accepted'] is False
    assert 'FINAL_SOURCE_DOCUMENT_NOT_EXACTLY_PRESERVED' in result['reasons'][0]
    assert not (tmp_path/'repair/verified_snapshot.json').exists()


def test_real_otherprotein_and_all_25_subject_profiles_are_losslessly_mapped():
    base=ROOT/'experiments/model_recommendation/execution_validation_20261001/source_preservation_repairs/otherprotein_schema_probe_01'
    original=json.loads((base/'original_one_simulation.json').read_text())
    native=json.loads((base/'export/output/original_one_simulation.json').read_text())
    mapped=map_legacy_expression_schema(original,native)
    assert len(original['Individuals'])==25
    assert any(p['Type']=='OtherProtein' for p in mapped['mapped_snapshot']['ExpressionProfiles'])
    assert m.exact_comparison(mapped['mapped_snapshot'],native)['accepted']


def _partition_resume_fixture(tmp_path, monkeypatch, *, compound_count, simulation_count, prior_batches):
    """Synthetic completed native stages; no R, database or solver invocation."""
    from recommendation_execution_repairs import native_partition_resume as resume
    for filename in resume.IMPLEMENTATION_FILES:
        path=tmp_path/filename;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('Synthetic implementation fixture: '+filename)
    executable=tmp_path/'runtime/native/Rscript';executable.parent.mkdir(parents=True)
    executable.write_bytes(b'Synthetic executable; never run')
    runtime={'gpu_used_by_native_solver':False,
             'locked_resources':[m.reference(executable,tmp_path)],'actual_engine_version':'synthetic'}
    original={'Version':79,'Compounds':[{'Name':f'Drug{i}','Parameters':[{'Name':'CL','Value':i+1.0}]}
                  for i in range(compound_count)],
              'Individuals':[{'Name':'original person','Weight':70.}],
              'Simulations':[{'Name':f'Sim{i}'} for i in range(simulation_count)]}
    source=tmp_path/'original.json';m.save(source,original)
    task={'source_path':'original.json','source_sha256':m.sha(source),'format':'.json'}
    calls=[]

    def stage(mode,input_path,directory,*,root,timeout_seconds,cpu_threads,stop_file):
        document=json.loads(input_path.read_text())
        calls.append({'mode':mode,'directory':directory,'names':[s['Name'] for s in document['Simulations']],
                      'global_fields':{k:deepcopy(v) for k,v in document.items() if k!='Simulations'}})
        directory.mkdir();(directory/'output').mkdir()
        log=directory/'execution.log'
        log.write_text('MIGRATION_STAGE_FINISHED_WITHOUT_ODE '+mode+'\n')
        process={'status':'completed','returncode':0,'stop_reason':None,'runtime_unchanged':True,
                 'registry_modified':False,'timeout_seconds':timeout_seconds,'cpu_threads':cpu_threads,
                 'script':m.R_SCRIPT,'script_sha256':m.sha(root/m.R_SCRIPT),
                 'runtime_binding':deepcopy(runtime),'log_path':str(log),
                 'process_manifest_path':str(directory/'process_manifest.json'),
                 'command':[str(executable),'--vanilla',str(root/m.R_SCRIPT),mode,
                            str(directory/'input'),str(directory/'output')]}
        m.save(directory/'process_manifest.json',process)
        ref=m.reference(input_path,root)
        receipt={'mode':mode,'status':'completed','stop_reason':None,'input':ref,
                 'copy_before_sha256':ref['sha256'],'copy_after_sha256':ref['sha256'],
                 'original_after_sha256':ref['sha256'],
                 'process':m.reference(directory/'process_manifest.json',root)}
        m.save(directory/'stage_receipt.json',receipt)
        output=directory/'output'/('model.pksim5' if mode=='snapshot_to_project' else 'model.json')
        if mode=='project_to_snapshot':document['Version']=80
        m.save(output,document)
        return output,[m.reference(directory/'stage_receipt.json',root),receipt['process'],m.reference(output,root)]

    prior=tmp_path/m.migration.OUT/'engineering_repairs'/'native_partition_preflight'/task['source_sha256']
    clean=m.migration.clean_snapshot(original)
    for i,names in enumerate(prior_batches):
        part=prior/f'partition_{i+1:03d}';part.mkdir(parents=True)
        scoped=deepcopy(clean);scoped['Simulations']=[s for s in clean['Simulations'] if s['Name'] in names]
        input_path=part/'source_model_only.json';m.save(input_path,scoped)
        project,_=stage('snapshot_to_project',input_path,part/'build',root=tmp_path,timeout_seconds=600,cpu_threads=1,stop_file=None)
        stage('project_to_snapshot',project,part/'export',root=tmp_path,timeout_seconds=600,cpu_threads=1,stop_file=None)
    calls.clear()
    def no_archive(*args,**kwargs):raise FileNotFoundError('No old synthetic migration')
    monkeypatch.setattr(m.migration,'revalidate_old_artifact',no_archive)
    monkeypatch.setattr(m,'_native_stage',stage)
    out=tmp_path/'new_source_attempt';out.mkdir()
    return task,original,out,calls


def test_all_completed_partitions_reexport_authenticated_projects_without_rebuilding(tmp_path,monkeypatch):
    task,original,out,calls=_partition_resume_fixture(tmp_path,monkeypatch,
        compound_count=8,simulation_count=10,
        prior_batches=[[f'Sim{i}' for i in range(0,4)],[f'Sim{i}' for i in range(4,8)],['Sim8','Sim9']])
    converted,refs,_=m._native_base(task,out,root=tmp_path,timeout_seconds=600,cpu_threads=1,stop_file=None)
    assert [call['mode'] for call in calls]==['project_to_snapshot']*3
    assert [len(call['names']) for call in calls]==[4,4,2]
    assert all('authenticated_resume' in call['directory'].parts for call in calls)
    assert [row['Name'] for row in converted['Simulations']]==[row['Name'] for row in original['Simulations']]
    assert all(not (out/f'partition_{i:03d}'/'build').exists() for i in range(1,4))
    certificates=[r for r in refs if r['path'].endswith('/resume_certificate.json')]
    assert len(certificates)==3
    for ref in certificates:
        document=json.loads(m.checked(ref,tmp_path).read_text())
        assert document['scientific_accepted'] is False and document['final_source_comparison_required'] is True
        assert document['snapshot'] in document['provenance_refs']


def test_nine_compound_source_builds_one_at_a_time_and_keeps_every_simulation_and_global(tmp_path,monkeypatch):
    task,original,out,calls=_partition_resume_fixture(tmp_path,monkeypatch,
        compound_count=9,simulation_count=5,prior_batches=[['Sim0']])
    converted,_,_=m._native_base(task,out,root=tmp_path,timeout_seconds=600,cpu_threads=1,stop_file=None)
    builds=[call for call in calls if call['mode']=='snapshot_to_project']
    assert [call['names'] for call in builds]==[['Sim1'],['Sim2'],['Sim3'],['Sim4']]
    assert len([call for call in calls if call['mode']=='project_to_snapshot'])==5
    assert not (out/'partition_001/build').exists()
    assert all(len(call['names'])==1 for call in calls)
    expected_globals={k:v for k,v in m.migration.clean_snapshot(original).items() if k!='Simulations'}
    assert all(call['global_fields']==expected_globals for call in builds)
    assert len(converted['Compounds'])==9 and len(converted['Simulations'])==5
    assert [row['Name'] for row in converted['Simulations']]==[f'Sim{i}' for i in range(5)]


@pytest.mark.parametrize('proof_accepted,drift,expected_status',[(True,False,'verified'),
    (False,False,'failed'),(True,True,'failed')])
def test_mapped_json_equality_never_bypasses_required_complete_model_proof(tmp_path,monkeypatch,
        proof_accepted,drift,expected_status):
    original={'Version':79,'Compounds':[{'Name':'Drug','Parameters':[{'Name':'CL','Value':1.}]}],
              'Simulations':[{'Name':'A'},{'Name':'B'}]}
    native=m.migration.clean_snapshot(original);native['Version']=80
    assert m.exact_comparison(original,native)['accepted']
    source=tmp_path/'original.json';m.save(source,original)
    provenance={'source_path':str(source.relative_to(ROOT)),'source_sha256':m.sha(source)}
    monkeypatch.setattr(m,'_native_base',lambda *args,**kwargs:(deepcopy(native),[],'synthetic_native'))
    monkeypatch.setattr(m,'map_legacy_expression_schema',lambda *args,**kwargs:{
        'mapped_snapshot':deepcopy(native),'changes':[{'operation':'forced_equivalence_required'}],
        'requires_complete_native_model_proof':True})
    monkeypatch.setattr(m,'restore_explicit_simulation_fields',lambda *args,**kwargs:{'snapshot':deepcopy(native)})
    def forbidden_overlay(*args,**kwargs):raise AssertionError('A forced full proof took the overlay shortcut')
    monkeypatch.setattr(m,'validate_overlay',forbidden_overlay)
    full_proof_calls=[]
    def full_proof(source_document,native_document,out,**kwargs):
        full_proof_calls.append([row['Name'] for row in source_document['Simulations']])
        converted=deepcopy(native_document)
        if drift:converted['Compounds'][0]['Parameters'][0]['Value']+=1e-14
        return {'proof':{'accepted':proof_accepted},'snapshot':converted,'provenance_refs':[]}
    monkeypatch.setattr(m,'repair_complex_snapshot',full_proof)
    result=m.repair_snapshot(provenance,tmp_path/'new_repair')
    assert full_proof_calls==[['A','B']]
    assert result['status']==expected_status
    assert result['requires_complete_native_model_proof'] is True
    assert (tmp_path/'new_repair/verified_snapshot.json').exists()==(expected_status=='verified')
    if proof_accepted and not drift:
        assert 'complete_registered_source_model_and_explicit_parameter_proof' in result['operation']
    elif not proof_accepted:
        assert 'COMPLETE_SOURCE_MODEL_PROOF_NOT_ACCEPTED' in result['reasons'][0]
    else:
        assert 'FINAL_SOURCE_DOCUMENT_NOT_EXACTLY_PRESERVED' in result['reasons'][0]
