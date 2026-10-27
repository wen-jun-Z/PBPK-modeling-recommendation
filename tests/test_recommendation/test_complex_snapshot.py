"""Original-input preservation gates without R, ODE or clinical observations."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from recommendation_execution_repairs.complex_snapshot import (
    assert_no_explicit_simulation_seed,parameter_expectations,source_partition,unique)
from recommendation_execution_repairs import complex_snapshot as m


@pytest.fixture
def tmp_path():
    parent=ROOT/'runtime/tmp';parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='complex-snapshot-',dir=parent) as directory:
        yield Path(directory)


@pytest.mark.parametrize('declared',[
    {'RandomSeed':17},{'Settings':{'RandomSeed':17}},
    {'SimulationSettings':{'random_seed':0}}, {'Solver':{'Seed':1}},
])
def test_declared_simulation_seed_cannot_be_overwritten(declared):
    with pytest.raises(ValueError,match='EXPLICIT_SIMULATION_SEED'):
        assert_no_explicit_simulation_seed({'Simulations':[{'Name':'s',**declared}]})


def test_population_seed_is_preserved_and_unrelated_population_not_built():
    original={'Version':79,'Populations':[{'Name':'used','Settings':{'Seed':99}},
        {'Name':'unused','Settings':{'Seed':42}}],
        'Compounds':[{'Name':'D','Parameters':[{'Name':'x','Value':3}]}],
        'Simulations':[{'Name':'a','Population':'used'},{'Name':'b','Population':'unused'}]}
    before=deepcopy(original)
    assert_no_explicit_simulation_seed(original)
    part=source_partition(original,['a'])
    assert part['Populations']==[before['Populations'][0]]
    assert part['Compounds']==before['Compounds'] and original==before
    assert part['Simulations'][0]['Name']=='a'


def test_population_reference_must_exist():
    with pytest.raises(ValueError,match='SOURCE_POPULATION_DEPENDENCY'):
        source_partition({'Simulations':[{'Name':'s','Population':'missing'}]},['s'])


def test_effective_explicit_parameter_retains_source_origin_and_override():
    source={'Compounds':[{'Name':'D','Parameters':[{'Name':'x','Value':1,'Unit':'mg',
        'ValueOrigin':{'Source':'Unknown'}}]}]}
    sim={'Name':'s','Compounds':[{'Name':'D'}],
        'Parameters':[{'Path':'D|x','Value':2,'Unit':'mg','ValueOrigin':{'Source':'Publication','Description':'A'}}]}
    result=parameter_expectations(source,sim)
    assert result['parameters'][0]['value']==2
    assert result['parameters'][0]['source_record']==sim['Parameters'][0]
    assert result['parameters'][0]['source_scope']=='simulation:s'


def test_table_formula_does_not_get_checked_as_a_constant():
    source={'Compounds':[{'Name':'D','Parameters':[{'Name':'x','Value':1}]}]}
    sim={'Name':'s','Compounds':[{'Name':'D'}],
         'Parameters':[{'Path':'D|x','TableFormula':{'X':[0,1],'Y':[1,2]}}]}
    result=parameter_expectations(source,sim)
    assert result['parameters']==[]
    assert result['table_formulas'][0]['source_record']==sim['Parameters'][0]


def test_duplicate_simulation_identity_is_rejected():
    with pytest.raises(ValueError,match='Duplicate'):
        unique([{'Name':'same'},{'Name':'same'}],'Name')


def test_native_readback_has_no_numeric_tolerance_acceptance():
    code=(ROOT/'src/pbpk_kb/model_runtime/R/recommendation_snapshot_readback.R').read_text()
    assert 'accepted=isTRUE(actual==expected)' in code
    assert 'delta<=bound' not in code


def test_generic_registration_requires_frozen_exact_parent_source(tmp_path,monkeypatch):
    source=tmp_path/'source.json';source.write_text('{"Version":76,"Simulations":[]}')
    ref={'source_path':str(source.relative_to(ROOT)),'source_sha256':m.migration.sha(source)}
    audit=tmp_path/'candidate_audit.jsonl'
    row={'candidate_id':'C','scenario_id':'Q','identity_match':True,'catalog_provenance':ref}
    audit.write_text(json.dumps(row)+'\n')
    monkeypatch.setattr(m,'FROZEN_CANDIDATE_AUDIT',{'path':str(audit.relative_to(ROOT)),
                                                 'sha256':m.migration.sha(audit)})
    assert m.registered_source(ref,{'Version':76},ROOT)['matches']==[{'candidate_id':'C','scenario_id':'Q'}]
    audit.write_text(json.dumps({**row,'identity_match':False})+'\n')
    with pytest.raises(ValueError,match='FROZEN_CANDIDATE_AUDIT_CHANGED'):
        m.registered_source(ref,{'Version':76},ROOT)
    monkeypatch.setattr(m,'FROZEN_CANDIDATE_AUDIT',{'path':str(audit.relative_to(ROOT)),
                                                 'sha256':m.migration.sha(audit)})
    with pytest.raises(ValueError,match='SOURCE_NOT_IN_FROZEN_EXACT_PARENT_CATALOG'):
        m.registered_source(ref,{'Version':76},ROOT)


def test_changed_original_bytes_cannot_register(tmp_path,monkeypatch):
    source=tmp_path/'source.json';source.write_text('{"Version":79}')
    ref={'source_path':str(source.relative_to(ROOT)),'source_sha256':m.migration.sha(source)}
    source.write_text('{"Version":79,"changed":true}')
    with pytest.raises(ValueError,match='Unregistered exact original'):
        m.registered_source(ref,{'Version':79},ROOT)


def inactive_fixture():
    record={'Path':'Applications|P|Application_2|ProtocolSchemaItem|Volume of water/body weight',
            'Value':3.5,'Unit':'ml/kg','ValueOrigin':{'Source':'Unknown'}}
    original={'Simulations':[{'Name':'s','Parameters':[record]}]}
    registry=m.register_preserved_simulation_overrides(original,deepcopy(original))['s']
    parameter={'path':next(iter(registry)),'value':3.5,'unit':'ml/kg','source_scope':'simulation:s',
               'source_record':deepcopy(record),'inactive_override_registration':next(iter(registry.values()))}
    check={'path':parameter['path'],'source_value':3.5,'source_unit':'ml/kg','source_scope':'simulation:s',
           'source_record':deepcopy(record),'status':'missing_in_native_model','accepted':False}
    return original,{'parameters':[parameter]}, {'parameter_readbacks':[check],'all_parameters_accepted':False}


def test_inactive_override_preserves_record_without_inventing_a_native_value():
    original,expectation,row=inactive_fixture()
    result=m.compare_bilateral_readbacks(row,deepcopy(row),expectation)
    assert result['active_parameters_exact_per_build']==0 and result['inactive_override_count']==1
    inactive=result['inactive_overrides'][0]
    assert inactive['source_record']==original['Simulations'][0]['Parameters'][0]
    assert inactive['status']=='source_override_inactive_in_both_official_builds'
    assert inactive['numerical_value_validated'] is False and result['all_source_records_accounted_for'] is True
    assert 'actual_native_value' not in row['parameter_readbacks'][0]


@pytest.mark.parametrize('alteration',['drop','value','unit','path','origin'])
def test_mapped_override_cannot_be_dropped_or_changed(alteration):
    original,_,_=inactive_fixture();restored=deepcopy(original);record=restored['Simulations'][0]['Parameters'][0]
    if alteration=='drop':restored['Simulations'][0]['Parameters']=[]
    elif alteration=='value':record['Value']+=1e-14
    elif alteration=='unit':record['Unit']='l/kg'
    elif alteration=='path':record['Path']=record['Path'].replace('Application_2','Application_3')
    else:record['ValueOrigin']['Source']='Publication'
    with pytest.raises(ValueError,match='OVERRIDE_(SET|RECORD)_CHANGED'):
        m.register_preserved_simulation_overrides(original,restored)


@pytest.mark.parametrize('scope',['compound:D','individual:I','expression_profile:D|Human|Default'])
def test_missing_non_simulation_parameter_is_never_exempted(scope):
    _,expectation,row=inactive_fixture();expectation['parameters'][0]['source_scope']=scope
    row['parameter_readbacks'][0]['source_scope']=scope
    with pytest.raises(ValueError,match='UNREGISTERED'):m.validate_parameter_readbacks(row,expectation)


@pytest.mark.parametrize('tamper',['path','source_value','source_unit','source_record','fake_value','fake_acceptance','missing_record','summary'])
def test_inactive_native_report_cannot_change_or_fake_source_records(tamper):
    _,expectation,row=inactive_fixture();check=row['parameter_readbacks'][0]
    if tamper=='path':check['path']+='x'
    elif tamper=='source_value':check['source_value']=4
    elif tamper=='source_unit':check['source_unit']='l/kg'
    elif tamper=='source_record':check['source_record']['Value']=4
    elif tamper=='fake_value':check['actual_native_value']=0
    elif tamper=='fake_acceptance':check['accepted']=True
    elif tamper=='missing_record':row['parameter_readbacks']=[]
    elif tamper=='summary':row['all_parameters_accepted']=True
    with pytest.raises(ValueError):m.validate_parameter_readbacks(row,expectation)


def as_present(row,value=0.0035):
    row=deepcopy(row);row['all_parameters_accepted']=True;row['parameter_readbacks'][0].update(
        status='present_exact',accepted=True,actual_native_value=value,expected_native_value=0.0035,
        native_unit='l/kg',dimension='Volume per body weight',absolute_difference=abs(value-0.0035))
    return row


def test_one_sided_missing_override_rejects_model_conversion():
    _,expectation,row=inactive_fixture()
    with pytest.raises(ValueError,match='PRESENCE_DIFFERS'):
        m.compare_bilateral_readbacks(row,as_present(row),expectation)


def test_present_numeric_drift_still_has_zero_tolerance():
    _,expectation,row=inactive_fixture()
    with pytest.raises(ValueError,match='SOURCE_PARAMETER_NATIVE_DIFFERENCE'):
        m.compare_bilateral_readbacks(as_present(row),as_present(row,0.0035+1e-15),expectation)


def test_registration_path_cannot_be_rebound_away_from_preserved_raw_path():
    _,expectation,row=inactive_fixture();parameter=expectation['parameters'][0]
    wrong=parameter['path'].replace('Application_2','Application_20')
    parameter['path']=wrong;parameter['inactive_override_registration']['native_path']=wrong
    row['parameter_readbacks'][0]['path']=wrong
    with pytest.raises(ValueError,match='UNREGISTERED'):m.validate_parameter_readbacks(row,expectation)


def test_native_json_integer_spelling_preserves_exact_number_but_not_boolean_or_drift():
    _,expectation,row=inactive_fixture()
    expectation['parameters'][0]['source_record']['ValueOrigin']['NumericAnnotation']=1.0
    expectation['parameters'][0]['inactive_override_registration']['source_record']=deepcopy(expectation['parameters'][0]['source_record'])
    expectation['parameters'][0]['inactive_override_registration']['source_record_sha256']=m._record_digest(expectation['parameters'][0]['source_record'])
    row['parameter_readbacks'][0]['source_record']['ValueOrigin']['NumericAnnotation']=1
    assert m.validate_parameter_readbacks(row,expectation)['missing_pending_bilateral_audit']
    for wrong in [True,1+1e-15,'1']:
        row['parameter_readbacks'][0]['source_record']['ValueOrigin']['NumericAnnotation']=wrong
        with pytest.raises(ValueError,match='BINDING_CHANGED'):m.validate_parameter_readbacks(row,expectation)


def serialization_fixture(tmp_path):
    source=0.9741260583;actual=0.9741260582577684
    parameter={'path':'CYP3A4|Ontogeny factor GI','value':source,'unit':None,
        'source_scope':'expression_profile:CYP3A4|Human|source',
        'source_record':{'Path':'CYP3A4|Ontogeny factor GI','Value':source}}
    artifact=tmp_path/'official.json'
    artifact.write_text(json.dumps({'Name':parameter['path'],'Value':source,'Unit':''}))
    audit={'status':'official_snapshot_serialization_audit','serializer':'PKSim.Core.Services.IJsonSerializer',
        'dto_type':'PKSim.Core.Snapshots.Parameter','parameter_path':parameter['path'],
        'actual_native_value':actual,'native_unit':'','dimension':'Dimensionless',
        'declared_unit':'','source_unit_was_unspecified':True,
        'source_unit_value_before_serialization':actual,'serialized_value':source,
        'source_value':source,'source_value_exactly_reproduced':True,
        'serializer_assembly':'PKSim.Infrastructure, Version=12.3.137.0',
        'artifact':m.migration.reference(artifact,ROOT)}
    check={'path':parameter['path'],'status':'official_snapshot_serialization_equivalent','accepted':False,
        'source_value':source,'source_unit':None,'actual_native_value':actual,'expected_native_value':source,
        'native_unit':'','dimension':'Dimensionless','absolute_difference':abs(actual-source),
        'source_record':deepcopy(parameter['source_record']),'source_scope':parameter['source_scope'],
        'official_serialization':audit}
    return {'parameters':[parameter]},{'parameter_readbacks':[check],'all_parameters_accepted':False}


def test_official_roundtrip_is_separate_from_literal_exact_and_preserves_source(tmp_path):
    expectation,row=serialization_fixture(tmp_path)
    proof=m.compare_bilateral_readbacks(row,deepcopy(row),expectation)
    assert proof['active_parameters_exact_per_build']==0
    assert proof['official_serialization_equivalent_count']==1
    assert proof['inactive_override_count']==0
    entry=proof['official_serialization_equivalents'][0]
    assert entry['native_value_dimension_unit_identical'] is True
    assert entry['source_record']==expectation['parameters'][0]['source_record']
    assert entry['original']['numerical_tolerance_used'] is False
    assert entry['original']['literal_source_to_base_double_equality'] is False


@pytest.mark.parametrize('tamper',[
    'missing_audit','wrong_value','wrong_native','wrong_unit','wrong_dimension','wrong_path',
    'missing_assembly','missing_artifact','artifact_sha','artifact_value','artifact_path',
    'fake_numeric_acceptance','fake_summary','wrong_delta','renamed_status','one_sided_audit',
])
def test_serialization_proof_rejects_incomplete_or_changed_bindings(tmp_path,tamper):
    expectation,row=serialization_fixture(tmp_path);other=deepcopy(row)
    check=other['parameter_readbacks'][0];audit=check['official_serialization']
    if tamper=='missing_audit':check.pop('official_serialization')
    elif tamper=='wrong_value':audit['serialized_value']+=1e-10
    elif tamper=='wrong_native':audit['actual_native_value']+=1e-15
    elif tamper=='wrong_unit':audit['declared_unit']='h'
    elif tamper=='wrong_dimension':audit['dimension']='Time'
    elif tamper=='wrong_path':audit['parameter_path']='other'
    elif tamper=='missing_assembly':audit.pop('serializer_assembly')
    elif tamper=='missing_artifact':audit.pop('artifact')
    elif tamper=='artifact_sha':audit['artifact']['sha256']='0'*64
    elif tamper in {'artifact_value','artifact_path'}:
        path=ROOT/audit['artifact']['path'];encoded=json.loads(path.read_text())
        if tamper=='artifact_value':encoded['Value']+=1e-10
        else:encoded['Name']='other'
        path.write_text(json.dumps(encoded));audit['artifact']=m.migration.reference(path,ROOT)
        row['parameter_readbacks'][0]['official_serialization']['artifact']=deepcopy(audit['artifact'])
    elif tamper=='fake_numeric_acceptance':check['accepted']=True
    elif tamper=='fake_summary':other['all_parameters_accepted']=True
    elif tamper=='wrong_delta':check['absolute_difference']=0
    elif tamper=='renamed_status':check['status']='present_numeric_difference'
    elif tamper=='one_sided_audit':check['official_serialization']['source_value_exactly_reproduced']=False
    with pytest.raises(ValueError):m.compare_bilateral_readbacks(row,other,expectation)


@pytest.mark.parametrize('drift',['value','unit','dimension','display_value','assembly'])
def test_same_rounded_source_value_never_exempts_native_bilateral_drift(tmp_path,drift):
    expectation,row=serialization_fixture(tmp_path);other=deepcopy(row)
    check=other['parameter_readbacks'][0];audit=check['official_serialization']
    if drift=='value':
        check['actual_native_value']+=1e-15;audit['actual_native_value']=check['actual_native_value']
        check['absolute_difference']=abs(check['actual_native_value']-check['expected_native_value'])
    elif drift=='unit':
        check['native_unit']='unit';audit['native_unit']='unit';audit['declared_unit']='unit'
        artifact=tmp_path/'second.json';artifact.write_text(json.dumps({'Name':check['path'],'Value':check['source_value'],'Unit':'unit'}))
        audit['artifact']=m.migration.reference(artifact,ROOT)
    elif drift=='dimension':check['dimension']='Other';audit['dimension']='Other'
    elif drift=='display_value':audit['source_unit_value_before_serialization']+=1e-15
    elif drift=='assembly':audit['serializer_assembly']='OtherSerializerRuntime'
    with pytest.raises(ValueError,match='BILATERAL_NATIVE_'):m.compare_bilateral_readbacks(row,other,expectation)


def test_readback_executes_official_serializer_not_local_rounding_rule():
    code=(ROOT/m.R_SCRIPT).read_text()
    assert 'resolve("PKSim.Core.Services.IJsonSerializer")' in code
    assert 'newObjectFromName("PKSim.Core.Snapshots.Parameter")' in code
    assert 'json_serializer$call("Serialize",dto,destination)' in code
    assert 'encoded$Value==check$value' in code
    assert 'round(' not in code and 'signif(' not in code


def representation_fixture(tmp_path):
    expectation,row=serialization_fixture(tmp_path);row['name']='s'
    check=row['parameter_readbacks'][0]
    pkml=tmp_path/'original.pkml'
    pkml.write_text('<SimulationTransfer><Simulation><Model><Root name="s"><Children>'
        '<Container name="CYP3A4"><Children><Parameter name="Ontogeny factor GI" '
        'value="0.9741260582577684" dim="Dimensionless" displayUnit="" />'
        '</Children></Container></Children></Root></Model></Simulation></SimulationTransfer>')
    row['pkml_path']=str(pkml);row['pkml_sha256']=m.migration.sha(pkml)
    readback=tmp_path/'readback.json';readback.write_text(json.dumps({'simulations':[row]}))
    other=deepcopy(row);b=other['parameter_readbacks'][0]
    b.update(status='present_exact',accepted=True,actual_native_value=b['expected_native_value'],absolute_difference=0)
    b.pop('official_serialization');other['all_parameters_accepted']=True
    change=m.derive_native_representation_changes(row,other,expectation,readback)[0]
    snapshot={'Version':80,'ExpressionProfiles':[{'Molecule':'CYP3A4','Species':'Human','Category':'source',
        'Parameters':[deepcopy(expectation['parameters'][0]['source_record'])]}]}
    return snapshot,change,expectation,row,other


def test_representation_recovery_uses_exact_source_native_unit_value_and_keeps_raw_record(tmp_path):
    snapshot,change,_,_,_=representation_fixture(tmp_path);before=deepcopy(snapshot)
    fixed=m.apply_native_representation_changes(snapshot,[change])
    assert snapshot==before
    assert change['source_record']['Value']==0.9741260583
    assert fixed['ExpressionProfiles'][0]['Parameters'][0]['Value']==0.9741260582577684
    assert change['unit_changed'] is False and change['fitting_performed'] is False
    assert change['original_native_value']==0.9741260582577684


@pytest.mark.parametrize('tamper',['value','unit','scope','path','raw_value','formula','pkml_value',
    'pkml_binding','readback_sha','drop_record','duplicate','rule'])
def test_representation_recovery_rejects_changes_beyond_authenticated_constant(tmp_path,tamper):
    snapshot,change,_,row,_=representation_fixture(tmp_path)
    if tamper=='value':change['restored_record']['Value']+=1e-15
    elif tamper=='unit':change['restored_record']['Unit']='h'
    elif tamper=='scope':change['expression_profile']='CYP3A4|Human|other'
    elif tamper=='path':change['parameter_path']='CYP3A4|other'
    elif tamper=='raw_value':change['source_record']['Value']+=1e-10
    elif tamper in {'formula','pkml_value'}:
        pkml=ROOT/change['original_native_pkml']['path'];value=pkml.read_text()
        value=value.replace('value="0.9741260582577684"','formula="F" value="0.9741260582577684"' if tamper=='formula' else 'value="0.9741260582577685"')
        pkml.write_text(value);change['original_native_pkml']=m.migration.reference(pkml,ROOT)
        row['pkml_sha256']=m.migration.sha(pkml);readback=ROOT/change['original_readback']['path']
        readback.write_text(json.dumps({'simulations':[row]}));change['original_readback']=m.migration.reference(readback,ROOT)
    elif tamper=='pkml_binding':change['original_native_pkml']['sha256']='0'*64
    elif tamper=='readback_sha':change['original_readback']['sha256']='0'*64
    elif tamper=='drop_record':snapshot['ExpressionProfiles'][0]['Parameters']=[]
    elif tamper=='rule':change['rule']='generic_epsilon'
    with pytest.raises((ValueError,KeyError)):
        m.apply_native_representation_changes(snapshot,[change,deepcopy(change)] if tamper=='duplicate' else [change])


def test_representation_recovery_never_changes_compound_or_simulation_fields(tmp_path):
    snapshot,change,expectation,row,other=representation_fixture(tmp_path)
    for scope in ['compound:D','simulation:s','individual:i']:
        expectation['parameters'][0]['source_scope']=scope
        row['parameter_readbacks'][0]['source_scope']=scope
        other['parameter_readbacks'][0]['source_scope']=scope
        with pytest.raises(ValueError,match='NOT_REGISTERABLE'):
            m.derive_native_representation_changes(row,other,expectation,ROOT/change['original_readback']['path'])


@pytest.mark.parametrize('mapped_tamper',[None,'value','unit','origin'])
def test_complex_repair_binds_override_to_source_record_before_native_path_rename(tmp_path,monkeypatch,mapped_tamper):
    original,_,_=inactive_fixture();original['Version']=79
    before=deepcopy(original);source=tmp_path/'source.json';source.write_text(json.dumps(original))
    native=deepcopy(original);native['Version']=80
    mapped=deepcopy(native)
    record=mapped['Simulations'][0]['Parameters'][0]
    if mapped_tamper=='value':record['Value']+=1e-14
    elif mapped_tamper=='unit':record['Unit']='l/kg'
    elif mapped_tamper=='origin':record['ValueOrigin']['Source']='Publication'
    monkeypatch.setattr(m,'registered_source',lambda *a:{'synthetic_registered_source':True})
    monkeypatch.setattr(m,'map_legacy_expression_schema',lambda *a:{'mapped_snapshot':deepcopy(mapped),'changes':[]})
    calls=[];source_record=before['Simulations'][0]['Parameters'][0]
    class NativeCopiesChecked(Exception):pass
    def stage(mode,input_path,directory,**kwargs):
        assert mode=='snapshot_to_project'
        document=json.loads(input_path.read_text());calls.append(directory.name)
        actual=document['Simulations'][0]['Parameters'][0]
        if directory.name=='original_build':
            assert actual==source_record
            return input_path,[]
        expected=deepcopy(source_record);expected['Path']=expected['Path'].replace('Applications|','Events|',1)
        assert directory.name=='restored_build' and actual==expected
        raise NativeCopiesChecked
    def readback(project,expectations,output,**kwargs):
        parameter=expectations['s']['parameters'][0]
        assert parameter['path']==source_record['Path'].replace('Applications|','Events|',1)
        assert parameter['source_record']==source_record
        registration=parameter['inactive_override_registration']
        assert registration['original_path']==source_record['Path']
        assert registration['source_record']==source_record
        assert registration['source_record_sha256']==m._record_digest(source_record)
        return {'s':{}},[]
    monkeypatch.setattr(m,'readback',readback)
    provenance={'source_path':str(source.relative_to(ROOT)),'source_sha256':m.migration.sha(source)}
    if mapped_tamper:
        with pytest.raises(ValueError,match='EXPLICIT_SIMULATION_OVERRIDE_RECORD_CHANGED'):
            m.repair_complex_snapshot(original,native,tmp_path/'proof',source_provenance=provenance,
                                      root=ROOT,stage_runner=stage)
        assert calls==[]
    else:
        with pytest.raises(NativeCopiesChecked):
            m.repair_complex_snapshot(original,native,tmp_path/'proof',source_provenance=provenance,
                                      root=ROOT,stage_runner=stage)
        assert calls==['original_build','restored_build']
    assert original==before and json.loads(source.read_text())==before
