from copy import deepcopy
from pathlib import Path
import importlib.util
import pytest
from scripts.recommendation_execution_repairs import complete_candidate as c
from scripts.recommendation_execution_repairs import json_native_candidate as bridge
from scripts.recommendation_execution_repairs import json_native_source_proof as proof

ROOT=Path('/home/xuyiqian/Research_Projects/Knowledge_Graph_Model_Recommendation')
spec=importlib.util.spec_from_file_location('existing_complete_fixture',ROOT/'tests/test_recommendation/test_complete_candidate.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
prepared=old.prepared


@pytest.fixture
def json_prepared(prepared,monkeypatch):
    root,scenario,candidate,metadata=prepared
    original=root/'author.json';original.write_text('{"original":"science"}')
    native={'path':'source.pksim5','sha256':c.sha(root/'source.pksim5')}
    candidate['provenance'].update(source_path='author.json',source_sha256=c.sha(original))
    entry={'source_original':{'path':'author.json','sha256':c.sha(original)},'native_source':native,
        'allowed_simulations':[{'name':metadata['name'],'id':metadata['simulation_id'],'source_blob_sha256':metadata['source_blob_sha256']}],
        'proof_refs':[]}
    def verified(prov,value,**kwargs):
        if value!=entry or prov['source_sha256']!=c.sha(original) or c.sha(root/native['path'])!=native['sha256']:
            raise ValueError('unbound source proof')
        return deepcopy(entry)
    monkeypatch.setattr(proof,'verify_native_source',verified)
    monkeypatch.setattr(c,'implementation_hashes',lambda *args,**kwargs:{'scripts/recommendation_execution_repairs/source_pkml.py':'code'})
    return root,scenario,candidate,metadata,entry


def test_verified_bridge_preserves_original_catalogue_and_clinical_identity(json_prepared):
    root,scenario,candidate,metadata,entry=json_prepared;original=deepcopy(candidate)
    p=c.prepare_candidate(scenario,candidate,project_root=root,verified_json_source=entry)
    assert p['status']=='ready',p
    assert candidate==original and p['source_path']=='author.json'
    assert p['source_sha256']==candidate['provenance']['source_sha256']
    assert p['execution_source_path']=='source.pksim5' and p['execution_source_sha256']==entry['native_source']['sha256']
    assert p['adapter_route']==bridge.ROUTE and p['json_native_source_proof']==entry
    assert p['candidate_sha256']==c._digest(candidate)
    assert p['contract']['dose_mg']==100. and p['native_compound_name']=='Drug'
    assert c.execution_source_reference(p,candidate,project_root=root)==entry['native_source']


@pytest.mark.parametrize('change',['project_path','project_sha','name','id','blob','original'])
def test_modified_native_source_or_simulation_proof_stays_blocked(json_prepared,change):
    root,scenario,candidate,metadata,entry=json_prepared;bad=deepcopy(entry)
    if change=='project_path':bad['native_source']['path']='other.pksim5'
    elif change=='project_sha':bad['native_source']['sha256']='f'*64
    elif change=='original':bad['source_original']['sha256']='f'*64
    else:bad['allowed_simulations'][0][{'name':'name','id':'id','blob':'source_blob_sha256'}[change]]='other'
    p=c.prepare_candidate(scenario,candidate,project_root=root,verified_json_source=bad)
    assert p['status']=='blocked' and 'unbound source proof' in p['reasons'][0]


def test_actual_metadata_must_match_proven_names_ids_and_blob(json_prepared,monkeypatch):
    root,scenario,candidate,metadata,entry=json_prepared
    metadata['name']='different native model';c._METADATA_CACHE.clear()
    p=c.prepare_candidate(scenario,candidate,project_root=root,verified_json_source=entry)
    assert p['status']=='blocked' and 'SIMULATION_BINDING_DIFFERS' in p['reasons'][0]


def test_missing_allowed_simulation_cannot_be_replaced_by_other_project_model(json_prepared,monkeypatch):
    root,scenario,candidate,metadata,entry=json_prepared
    metadata['simulation_id']='other';c._METADATA_CACHE.clear()
    p=c.prepare_candidate(scenario,candidate,project_root=root,verified_json_source=entry)
    assert p['status']=='blocked' and 'ALLOWED_SIMULATION_ABSENT' in p['reasons'][0]


def test_native_and_original_sources_revalidated_at_execution_boundary(json_prepared):
    root,scenario,candidate,metadata,entry=json_prepared
    p=c.prepare_candidate(scenario,candidate,project_root=root,verified_json_source=entry)
    (root/'source.pksim5').write_bytes(b'changed native model')
    with pytest.raises(ValueError,match='unbound source proof'):c.execution_source_reference(p,candidate,project_root=root)


def test_existing_usable_snapshot_never_invokes_bridge_or_native(monkeypatch):
    p={'status':'ready','snapshot_sha256':'same'}
    monkeypatch.setattr(bridge.snapshot_candidate,'prepare_candidate',lambda *a,**kw:p)
    monkeypatch.setattr(proof,'resolve_native_sources',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('unnecessary bridge')))
    assert bridge.prepare_candidate({}, {},project_root=ROOT) is p


@pytest.mark.parametrize('reason',['Blocked:CANDIDATE_LICENSE_NOT_RESOLVED','Blocked:SOURCE_CHECKSUM_MISMATCH','Blocked:REFERENCE_SUBJECT_APPROXIMATION_REQUIRES_EXPLICIT_PROTOCOL_PERMISSION'])
def test_source_license_or_scientific_scope_failure_never_triggers_fallback(monkeypatch,reason):
    p={'status':'blocked','reasons':[reason]}
    monkeypatch.setattr(bridge.snapshot_candidate,'prepare_candidate',lambda *a,**kw:p)
    monkeypatch.setattr(proof,'resolve_native_sources',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('unsafe fallback')))
    assert bridge.prepare_candidate({}, {},project_root=ROOT) is p


def test_partition_choice_prefers_original_single_drug_over_zero_exposure_isolation(monkeypatch):
    fallback={'status':'blocked','reasons':['Blocked:NO_SOURCE_SIMULATION_MATCHES_FIXED_EXECUTION_CONTRACT']}
    monkeypatch.setattr(bridge.snapshot_candidate,'prepare_candidate',lambda *a,**kw:fallback)
    entries=[{'native_source':{'path':str(i),'sha256':'h'}} for i in range(2)]
    monkeypatch.setattr(proof,'resolve_native_sources',lambda *a,**kw:{'candidates':entries,'new_native_build_required':False,'native_coverage_complete':True})
    def prepare(*a,verified_json_source,**kw):
        index=int(verified_json_source['native_source']['path'])
        return {'status':'ready','reasons':[],'source_selection_sort_key':[index],'source_simulation':str(index),
            'source_simulation_id':str(index),'execution_source_path':str(index),
            'coadministration_isolation':{'proved':True} if index==0 else None}
    monkeypatch.setattr(bridge.complete_candidate,'prepare_candidate',prepare)
    assert bridge.prepare_candidate({}, {'provenance':{'source_path':'original.json'}},project_root=ROOT)['source_simulation']=='1'


def test_partial_native_partition_cannot_hide_a_better_unbuilt_original_model(monkeypatch):
    monkeypatch.setattr(bridge.complete_candidate,'implementation_hashes',lambda *a,**kw:{p:'a'*64 for p in bridge.IMPLEMENTATION_FILES})
    fallback={'status':'blocked','reasons':['Blocked:NO_SOURCE_SIMULATION_MATCHES_FIXED_EXECUTION_CONTRACT']}
    monkeypatch.setattr(bridge.snapshot_candidate,'prepare_candidate',lambda *a,**kw:fallback)
    monkeypatch.setattr(proof,'resolve_native_sources',lambda *a,**kw:{'candidates':[{'native_source':{'path':'one'}}],
        'native_coverage_complete':False,'unbuilt_simulations':['eligible original model']})
    monkeypatch.setattr(bridge.complete_candidate,'prepare_candidate',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('partial selection')))
    result=bridge.prepare_candidate({}, {'provenance':{'source_path':'original.json'}},project_root=ROOT)
    assert result['status']=='blocked' and 'Full original source simulation coverage' in result['json_native_bridge_assessment']['reason']


def test_conflicting_provenance_fields_are_rejected_before_adapter(monkeypatch):
    monkeypatch.setattr(bridge.snapshot_candidate,'prepare_candidate',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('must not choose provenance')))
    with pytest.raises(ValueError,match='CONFLICTING_CANDIDATE_PROVENANCE_FIELDS'):
        bridge.prepare_candidate({}, {'provenance':{'source_path':'one.json'},'catalog_provenance':{'source_path':'two.json'}},project_root=ROOT)


def test_complete_rejects_conflicting_provenance_fields(json_prepared):
    root,scenario,candidate,metadata,entry=json_prepared
    candidate['catalog_provenance']={**candidate['provenance'],'source_path':'other.json'}
    p=c.prepare_candidate(scenario,candidate,project_root=root,verified_json_source=entry)
    assert p['status']=='blocked' and 'CONFLICTING_CANDIDATE_PROVENANCE_FIELDS' in p['reasons'][0]


@pytest.mark.parametrize('change',['missing_total','duplicate_total','missing_r_weight','r_weight_changed','unrecognized_weight_unit'])
def test_native_dual_readback_refuses_wrong_observer_or_either_parent_weight(change):
    from scripts.recommendation_execution_repairs import json_dual_native as dual
    from scripts.recommendation_execution_repairs import json_racemate_contract as rc
    plan={'dual_parent_contract':{'output_path':rc.OUTPUT_PATH,'molecular_weight_g_per_mol':345.416}}
    inventory={'observer_paths':[rc.OUTPUT_PATH], 'parameters':[
        {'path':p+'|Molecular weight','unit':'g/mol','value':345.416} for p in rc.PARENTS]}
    assert dual.native_output(inventory,plan)==(rc.OUTPUT_PATH,345.416)
    if change=='missing_total':inventory['observer_paths']=['Organism|PeripheralVenousBlood|Esomeprazole|Plasma (Peripheral Venous Blood)']
    if change=='duplicate_total':inventory['observer_paths']*=2
    if change=='missing_r_weight':inventory['parameters'].pop()
    if change=='r_weight_changed':inventory['parameters'][1]['value']=346.
    if change=='unrecognized_weight_unit':inventory['parameters'][1]['unit']='mg'
    with pytest.raises(ValueError,match='DUAL_PARENT_'):dual.native_output(inventory,plan)


def test_dual_parent_does_not_zero_one_parent_to_remove_a_third_drug(monkeypatch):
    from scripts.recommendation_execution_repairs import json_dual_native as dual
    monkeypatch.setattr(dual.racemate,'prepare_dual_parent_contract',lambda *a,**kw:{'other_administered_drugs':['Quinidine']})
    monkeypatch.setattr(dual.coadministration_isolation,'target_metadata',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('unsafe single-parent isolation')))
    def require_full_proof(*args,**kwargs):
        assert args[5]==['Esomeprazole','R-omeprazole']
        raise ValueError('both retained parents proof not established')
    monkeypatch.setattr(dual.coadministration_isolation,'prepare_source_isolation',require_full_proof)
    options,failures=dual.selection_options({'simulations':[{'name':'dual+quinidine','simulation_id':'id','source_blob_sha256':'blob'}]}, {}, {}, {}, 'source', 'sha',project_root=ROOT)
    assert options==[] and 'both retained parents proof not established' in failures[0]['reason']
