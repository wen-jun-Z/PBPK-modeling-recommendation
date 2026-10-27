"""Real hashed JSON/SQLite fixtures; no R, observed data or source substitution."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import pytest

STAGED=Path(__file__).resolve().parents[2]
PROJECT=next(p for p in (STAGED,*STAGED.parents) if (p/'src/pbpk_kb').is_dir())
sys.path[:0]=[str(PROJECT/'scripts'),str(PROJECT/'src')]
SPEC=importlib.util.spec_from_file_location('recommendation_execution_repairs.json_native_source_proof',STAGED/'scripts/recommendation_execution_repairs/json_native_source_proof.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


@pytest.fixture
def build(tmp_path,monkeypatch):
    # Each synthetic source owns its own optional build registration.
    monkeypatch.setattr(m,'SUPPLEMENTAL_BUILDS',None)
    monkeypatch.setattr(m,'SUPPLEMENTAL_BUILDS_SHA',None)
    root=tmp_path
    def write(path,value):
        data=value if isinstance(value,bytes) else json.dumps(value,sort_keys=True).encode()
        p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        return {'path':path,'sha256':hashlib.sha256(data).hexdigest()}
    doc={'Version':80,'Compounds':[{'Name':'Drug','Parameters':[{'Name':'constant','Value':1.0}]}],
         'Simulations':[{'Name':'CONTROL','Compounds':[{'Name':'Drug'}]}]}
    source=write('source/model.json',doc);verified=write('proof/verified_snapshot.json',doc)
    inp=write('proof/part.json',doc);worker=write(m.migration.R_SCRIPT,b'official migration worker')
    runtime=write('runtime/native.bin',b'pinned native runtime')
    native=root/'proof/build/output/part.pksim5';native.parent.mkdir(parents=True)
    with sqlite3.connect(native) as db:
        db.executescript('CREATE TABLE SIMULATIONS (SimulationId TEXT,SimulationMode INTEGER);CREATE TABLE BUILDING_BLOCKS (Id TEXT,Name TEXT,ContentId TEXT);CREATE TABLE CONTENTS (Id TEXT,Data BLOB);')
        blob=m.source_pkml.compress_model_blob(b'<Simulation id="sim-id" name="CONTROL"><Model><Root name="Drug" /></Model></Simulation>')
        db.execute('INSERT INTO SIMULATIONS VALUES (?,?)',('sim-id',0));db.execute('INSERT INTO BUILDING_BLOCKS VALUES (?,?,?)',('sim-id','CONTROL','blob-id'));db.execute('INSERT INTO CONTENTS VALUES (?,?)',('blob-id',blob))
    project=m.migration.reference(native,root)
    process={'status':'completed','returncode':0,'runtime_unchanged':True,'registry_modified':False,'script':worker['path'],'script_sha256':worker['sha256'],
             'command':[str(root/'runtime/Rscript'),'--vanilla',str(root/worker['path']),'snapshot_to_project',str(root/'proof/build/input'),str(native.parent)],
             'runtime_binding':{'locked_resources':[runtime]}}
    proc=write('proof/build/process_manifest.json',process)
    stage={'mode':'snapshot_to_project','status':'completed','input':inp,'process':proc,
           'copy_before_sha256':inp['sha256'],'copy_after_sha256':inp['sha256'],'original_after_sha256':inp['sha256']}
    st=write('proof/build/stage_receipt.json',stage)
    receipt={'source_path':source['path'],'source_sha256':source['sha256'],'snapshot':verified,'implementation':[worker],
             'provenance_refs':[inp,proc,st,project]}
    def freeze():return write('proof/receipt.json',receipt)
    freeze()
    # The existing independently tested source-repair authenticator is mocked
    # only at its public resolution boundary; all bridge refs/SHA checks are real.
    monkeypatch.setattr(m.snapshot_repair,'resolve_repaired_snapshot',lambda provenance,project_root: {'receipt_path':root/'proof/receipt.json'})
    return {'root':root,'write':write,'doc':doc,'source':source,'verified':verified,'inp':inp,'project':project,
            'process':process,'proc':proc,'stage':stage,'st':st,'receipt':receipt,'freeze':freeze}


def resolve(p):return m.resolve_native_sources({'source_path':p['source']['path'],'source_sha256':p['source']['sha256']},project_root=p['root'])


def update(p,key,value):
    previous=p[key];new=p['write'](previous['path'],value);p['receipt']['provenance_refs']=[new if r==previous else r for r in p['receipt']['provenance_refs']];p[key]=new;p['freeze']()


def test_exact_completed_build_preserves_original_source_and_native_identity(build):
    p=build;r=resolve(p);assert len(r['candidates'])==1
    c=r['candidates'][0];assert c['source_original']==p['source'] and c['native_source']==p['project']
    assert c['allowed_simulations'][0]['name']=='CONTROL' and c['allowed_simulations'][0]['id']=='sim-id'
    assert m.verify_native_source({'source_path':p['source']['path'],'source_sha256':p['source']['sha256']},c,project_root=p['root'])==c
    assert not r['native_process_started'] and not r['clinical_outcomes_accessed']


def test_unrelated_same_basename_cannot_be_selected(build):
    p=build;c=resolve(p)['candidates'][0];c['native_source']={**p['project'],'path':'unrelated/part.pksim5'}
    with pytest.raises(ValueError,match='not exact'):m.verify_native_source({'source_path':p['source']['path'],'source_sha256':p['source']['sha256']},c,project_root=p['root'])


def test_old_pre_restore_native_build_is_not_final_source(build):
    p=build;wrong=deepcopy(p['doc']);wrong['Compounds'][0]['Parameters'][0]['Value']=2.0
    update(p,'inp',wrong)
    p['stage']['input']=p['inp'];update(p,'st',p['stage'])
    assert resolve(p)['candidates']==[]


@pytest.mark.parametrize('fault',['returncode','runtime','command','script','output_path'])
def test_process_or_destination_cannot_be_substituted(build,fault):
    p=build;process=p['process']
    if fault=='returncode':process['returncode']=1
    if fault=='runtime':process['runtime_unchanged']=False
    if fault=='command':process['command'][3]='different_action'
    if fault=='script':process['script_sha256']='f'*64
    if fault=='output_path':process['command'][5]=str(p['root']/'another_output');(p['root']/'another_output').mkdir()
    update(p,'proc',process);p['stage']['process']=p['proc'];update(p,'st',p['stage'])
    with pytest.raises(ValueError):resolve(p)


def test_native_artifact_tampering_is_rejected(build):
    p=build;(p['root']/p['project']['path']).write_bytes(b'wrong project')
    with pytest.raises(ValueError):resolve(p)


def test_simulation_name_or_id_mismatch_is_rejected(build):
    p=build;path=p['root']/p['project']['path']
    with sqlite3.connect(path) as db:db.execute('UPDATE BUILDING_BLOCKS SET Name=?',('different',))
    new=m.migration.reference(path,p['root']);p['receipt']['provenance_refs']=[new if r==p['project'] else r for r in p['receipt']['provenance_refs']];p['freeze']()
    with pytest.raises(ValueError,match='simulation coverage'):resolve(p)


def test_simulation_partition_keeps_all_model_definitions_and_only_omits_unused_populations():
    verified={'Version':80,'Simulations':[{'Name':'A','Population':'P1'},{'Name':'B','Population':'P2'}],
              'Populations':[{'Name':'P1','Seed':1},{'Name':'P2','Seed':2}],'Compounds':[{'Name':'Drug','Value':4.0}]}
    scoped=deepcopy(verified);scoped['Simulations']=scoped['Simulations'][:1];scoped['Populations']=scoped['Populations'][:1]
    assert m._partition_matches(scoped,verified)
    scoped['Compounds'][0]['Value']=4.000000000000001;assert not m._partition_matches(scoped,verified)
    scoped['Compounds'][0]['Value']=True;assert not m._partition_matches(scoped,verified)


def test_export_only_chain_is_bound_to_actual_native_input(build):
    p=build;exported=p['write']('proof/export/output/snapshot.json',p['doc'])
    process=deepcopy(p['process']);process['command'][3]='project_to_snapshot';process['command'][4]=str(p['root']/'proof/export/input');process['command'][5]=str(p['root']/'proof/export/output')
    proc=p['write']('proof/export/process_manifest.json',process)
    stage={'mode':'project_to_snapshot','status':'completed','input':p['project'],'output':exported,'process_manifest':proc,
           'source_input_unchanged':True,'source_input_sha256_after':p['project']['sha256'],'copied_input_sha256_before':p['project']['sha256']}
    hist=p['write']('history/receipt.json',{'version':m.migration.VERSION,'source_path':p['source']['path'],'source_sha256':p['source']['sha256'],'stages':[stage]})
    p['receipt']['provenance_refs']=[p['project'],exported,proc,hist];p['freeze']()
    assert resolve(p)['candidates'][0]['bridge_mode']=='exact_verified_native_export'
    wrong=deepcopy(p['doc']);wrong['Compounds'][0]['Parameters'][0]['Value']=2.0
    p['write'](exported['path'],wrong)
    with pytest.raises(ValueError):resolve(p)


def test_upstream_receipt_cannot_replace_catalog_source(build):
    with pytest.raises(ValueError,match='another catalog source'):
        m.resolve_native_sources({'source_path':'another/model.json','source_sha256':'f'*64},project_root=build['root'])


def test_partition_export_receipt_can_authenticate_existing_project(build):
    p=build;exported=p['write']('proof/export/output/snapshot.json',p['doc'])
    process=deepcopy(p['process']);process['command'][3]='project_to_snapshot';process['command'][4]=str(p['root']/'proof/export/input');process['command'][5]=str(p['root']/'proof/export/output')
    proc=p['write']('proof/export/process_manifest.json',process)
    stage={'mode':'project_to_snapshot','status':'completed','input':p['project'],'process':proc,
           'original_after_sha256':p['project']['sha256'],'copy_before_sha256':p['project']['sha256']}
    st=p['write']('proof/export/stage_receipt.json',stage)
    p['receipt']['provenance_refs']=[p['project'],exported,proc,st];p['freeze']()
    r=resolve(p);assert r['candidates'][0]['bridge_mode']=='exact_verified_partition_export'
    assert r['native_coverage_complete'] and not r['new_native_build_required']


def test_partial_native_coverage_is_explicit_and_requires_further_build(build):
    p=build;verified=deepcopy(p['doc']);verified['Simulations'].append({'Name':'DDI','Compounds':[{'Name':'Drug'}]})
    p['verified']=p['write'](p['verified']['path'],verified);p['receipt']['snapshot']=p['verified'];p['freeze']()
    r=resolve(p);assert len(r['candidates'])==1 and r['unbuilt_simulations']==['DDI']
    assert r['new_native_build_required'] and not r['native_coverage_complete']


@pytest.mark.parametrize('fault',['none','receipt_identity','duplicate_source','manifest_tampered'])
def test_new_builds_require_fixed_supplemental_manifest_and_original_receipt(build,monkeypatch,fault):
    p=build;refs=deepcopy(p['receipt']['provenance_refs']);p['receipt']['provenance_refs']=[];receipt_ref=p['freeze']()
    assert resolve(p)['candidates']==[]
    exported=p['write']('new/export/output/part.json',p['doc'])
    process=deepcopy(p['process']);process['command'][3]='project_to_snapshot';process['command'][4]=str(p['root']/'new/export/input');process['command'][5]=str(p['root']/'new/export/output')
    proc=p['write']('new/export/process_manifest.json',process)
    stage=p['write']('new/export/stage_receipt.json',{'mode':'project_to_snapshot','status':'completed','input':p['project'],'process':proc,
        'copy_before_sha256':p['project']['sha256'],'original_after_sha256':p['project']['sha256']})
    comparison=p['write']('new/comparison.json',{'accepted':True,'exact_normalized_document_equality':True,
        'numeric_tolerance':{'absolute':0,'relative':0},'compound_parameters_fitted':False})
    refs += [exported,proc,stage,comparison]
    entry={'source_original':p['source'],'source_receipt':receipt_ref,'verified_snapshot':p['verified'],'provenance_refs':refs,
        'readback_required':True,'readbacks':[{'native_source':p['project'],'partition_input':p['inp'],
            'export_snapshot':exported,'comparison':comparison}]}
    if fault=='receipt_identity':entry['source_receipt']={**receipt_ref,'sha256':'e'*64}
    registry={'schema_version':'source-bound-supplemental-native-builds-v1','clinical_outcomes_accessed':False,'entries':[entry]}
    if fault=='duplicate_source':registry['entries'].append(deepcopy(entry))
    manifest=p['write']('bridge/supplemental.json',registry)
    monkeypatch.setattr(m,'SUPPLEMENTAL_BUILDS',manifest['path']);monkeypatch.setattr(m,'SUPPLEMENTAL_BUILDS_SHA',manifest['sha256'])
    if fault=='manifest_tampered':p['write'](manifest['path'],{**registry,'forged':True})
    if fault!='none':
        with pytest.raises(ValueError):resolve(p)
    else:
        r=resolve(p);assert len(r['candidates'])==1 and manifest in r['candidates'][0]['proof_refs']


def test_empty_unused_populations_are_only_syntactic_equivalence():
    original={'Version':80,'Simulations':[{'Name':'A'}],'Populations':[],'Value':4.0}
    exported={k:v for k,v in original.items() if k!='Populations'}
    assert m._same_snapshot(original,exported)
    wrong=deepcopy(exported);wrong['Value']=4.000000000000001
    assert not m._same_snapshot(original,wrong)
    wrong=deepcopy(original);wrong['Simulations'][0]['Population']='Required'
    wrong2={k:v for k,v in wrong.items() if k!='Populations'}
    assert not m._same_snapshot(wrong,wrong2)
    wrong=deepcopy(original);wrong['Populations']=[{'Name':'P'}]
    assert not m._same_snapshot(wrong,exported)


def test_incomplete_earlier_partition_export_cannot_erase_valid_build(build):
    p=build;process=deepcopy(p['process']);process['command'][3]='project_to_snapshot'
    out=p['root']/'empty/output';out.mkdir(parents=True)
    process['command'][4]=str(p['root']/'empty/input');process['command'][5]=str(out)
    proc=p['write']('empty/process_manifest.json',process)
    st=p['write']('empty/stage_receipt.json',{'mode':'project_to_snapshot','status':'completed','input':p['project'],'process':proc,
        'copy_before_sha256':p['project']['sha256'],'original_after_sha256':p['project']['sha256']})
    p['receipt']['provenance_refs'] += [proc,st];p['freeze']()
    r=resolve(p);assert len(r['candidates'])==1
    assert r['rejected_native_artifacts'][0]['reason']=='partition_export_has_no_hash_bound_output'


@pytest.mark.parametrize('fault',['no_readbacks','not_required','unregistered_native','no_export_process','modified_export','numeric_tolerance'])
def test_supplemental_readback_never_trusts_acceptance_boolean(build,fault):
    p=build;exported=p['write']('new/export/output/part.json',p['doc'])
    process=deepcopy(p['process']);process['command'][3]='project_to_snapshot';process['command'][4]=str(p['root']/'new/export/input');process['command'][5]=str(p['root']/'new/export/output')
    proc=p['write']('new/export/process_manifest.json',process)
    st=p['write']('new/export/stage_receipt.json',{'mode':'project_to_snapshot','status':'completed','input':p['project'],'process':proc,
        'copy_before_sha256':p['project']['sha256'],'original_after_sha256':p['project']['sha256']})
    audit={'accepted':True,'exact_normalized_document_equality':True,'numeric_tolerance':{'absolute':0,'relative':0},'compound_parameters_fitted':False}
    if fault=='numeric_tolerance':audit['numeric_tolerance']['relative']=1e-6
    comparison=p['write']('new/comparison.json',audit)
    refs={r['path']:r for r in [*p['receipt']['provenance_refs'],exported,proc,st,comparison]}
    item={'native_source':p['project'],'partition_input':p['inp'],'export_snapshot':exported,'comparison':comparison}
    entry={'readback_required':True,'readbacks':[item]}
    if fault=='no_readbacks':entry['readbacks']=[]
    if fault=='not_required':entry['readback_required']=False
    if fault=='unregistered_native':refs.pop(p['project']['path'])
    if fault=='no_export_process':refs.pop(st['path'])
    if fault=='modified_export':
        doc=deepcopy(p['doc']);doc['Compounds'][0]['Parameters'][0]['Value']=1.000000000000001
        changed=p['write'](exported['path'],doc);refs[exported['path']]=changed;item['export_snapshot']=changed
    with pytest.raises(ValueError):m._validate_supplemental_readbacks(entry,refs,p['root'],p['process']['script_sha256'])


def test_inventory_cache_preserves_hash_checks_and_returns_copies(build,monkeypatch):
    p=build;original=m.source_pkml.simulation_inventory;calls=[]
    def wrapped(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(m.source_pkml,'simulation_inventory',wrapped)
    first=m._inventory(p['project'],['CONTROL'],p['root']);first[0]['name']='caller mutation'
    second=m._inventory(p['project'],['CONTROL'],p['root'])
    assert second[0]['name']=='CONTROL' and len(calls)==1
    (p['root']/p['project']['path']).write_bytes(b'tampered after cache fill')
    with pytest.raises(ValueError):m._inventory(p['project'],['CONTROL'],p['root'])


def test_inventory_cache_cannot_reuse_different_coverage_or_native_content(build):
    p=build;m._inventory(p['project'],['CONTROL'],p['root'])
    with pytest.raises(ValueError,match='coverage'):m._inventory(p['project'],['OTHER'],p['root'])
    path=p['root']/p['project']['path']
    with sqlite3.connect(path) as db:db.execute('UPDATE BUILDING_BLOCKS SET Name=?',('OTHER',))
    changed=m.migration.reference(path,p['root'])
    with pytest.raises(ValueError,match='coverage'):m._inventory(changed,['CONTROL'],p['root'])


@pytest.mark.parametrize('fault', ['none','literal_equality','unknown_rule','unfrozen_proof',
    'unfrozen_nested_evidence','unfrozen_baseline','unbound_extra_project','execution_as_baseline','verifier_rejected'])
def test_semantic_readback_requires_full_evidence_and_separate_original_project(build, monkeypatch, fault):
    """Only a successful source-specific verifier may register comparison evidence."""
    from types import SimpleNamespace
    import recommendation_execution_repairs as package
    p=build
    doc=deepcopy(p['doc']);doc['Compounds'][0]['Parameters'][0]['Value']=1.000000000000001
    exported=p['write']('semantic/export/output/part.json',doc)
    process=deepcopy(p['process']);process['command'][3]='project_to_snapshot'
    process['command'][4]=str(p['root']/'semantic/export/input');process['command'][5]=str(p['root']/'semantic/export/output')
    proc=p['write']('semantic/export/process_manifest.json',process)
    st=p['write']('semantic/export/stage_receipt.json',{'mode':'project_to_snapshot','status':'completed','input':p['project'],
        'process':proc,'copy_before_sha256':p['project']['sha256'],'original_after_sha256':p['project']['sha256']})
    baseline=p['write']('original/baseline.pksim5',{'not_an_execution_source':True})
    if fault=='execution_as_baseline':baseline=p['project']
    build_ref=p['write']('original/build_result.json',{'native':baseline})
    evidence=p['write']('semantic/official_evidence.json',{'frozen':True})
    proof=p['write']('semantic/proof.json',{'evidence_refs':[build_ref,baseline,evidence],
        'original_equivalence':{'build_result':build_ref}})
    rule='source_bound_native_expression_profile_serialization'
    audit={'accepted':True,'exact_normalized_document_equality':fault=='literal_equality',
        'numeric_tolerance':{'absolute':0,'relative':0},'compound_parameters_fitted':False,
        'equivalence_rule':'unregistered' if fault=='unknown_rule' else rule,'semantic_proof':proof}
    comparison=p['write']('semantic/comparison.json',audit)
    refs={r['path']:r for r in [*p['receipt']['provenance_refs'],exported,proc,st,comparison,proof,build_ref,baseline,evidence]}
    if fault=='unfrozen_proof':refs.pop(proof['path'])
    if fault=='unfrozen_nested_evidence':refs.pop(evidence['path'])
    if fault=='unfrozen_baseline':refs.pop(baseline['path'])
    if fault=='unbound_extra_project':
        extra=p['write']('extra/unreviewed.pksim5',{'unregistered':True});refs[extra['path']]=extra
    calls=[]
    def verify(*args,**kwargs):
        calls.append(args);return {'accepted':fault!='verifier_rejected'}
    monkeypatch.setattr(package,'json_expression_profile_proof',SimpleNamespace(verify=verify),raising=False)
    entry={'readback_required':True,'readbacks':[{'native_source':p['project'],'partition_input':p['inp'],
        'export_snapshot':exported,'comparison':comparison}]}
    if fault=='none':
        m._validate_supplemental_readbacks(entry,refs,p['root'],p['process']['script_sha256'])
        assert len(calls)==1 and calls[0]==(p['inp'],exported,p['project'],proof)
    else:
        with pytest.raises(ValueError):
            m._validate_supplemental_readbacks(entry,refs,p['root'],p['process']['script_sha256'])
