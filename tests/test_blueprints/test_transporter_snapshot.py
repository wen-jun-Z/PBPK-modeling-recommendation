"""Source transport inputs are copied, never invented, into standard human builds."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid
import pytest
from pbpk_kb.blueprints.development_case import _plain
from pbpk_kb.blueprints.human_model_snapshot import (build_human_model_snapshot, HumanModelSnapshotError,
    _TRANSPORT_DIRECTIONS, _TRANSPORT_PROCESS_PARAMETERS)

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('artifacts/pbpk-model-assets/public_collection_20260909/Furosemide-Model__bd0c455aa3c8/source/Furosemide-Model.json')
IDENTITY=Path('artifacts/pbpk_raw_runs/compound_runs/run_20260909T083120Z/compounds/furosemide/compound_api/pubchem/properties_by_name_e8300a4ca369.bin')


def reference(path,location):
    return {'provenance_id':str(uuid.uuid5(uuid.NAMESPACE_URL,str(path)+'#'+location)),
            'provenance_kind':'model_asset','source_id':str(path),'source_location':location,
            'source_checksum':hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}


def furosemide_inputs():
    x=json.loads((ROOT/SOURCE).read_text())
    compound=_plain(next(c for c in x['Compounds'] if c['Name']=='Furosemide'))
    compound.pop('IntestinalPermeability',None)
    mw=next(p for p in compound['Parameters'] if p['Name']=='Molecular weight')
    formula=json.loads((ROOT/IDENTITY).read_text())['PropertyTable']['Properties'][0]['MolecularFormula']
    assert formula=='C12H11ClN2O5S'
    compound['Parameters']=[mw,*[{'Name':a,'Value':1 if a=='Cl' else 0} for a in ['F','Cl','Br','I']]]
    compound['CalculationMethods']=['Cellular partition coefficient method - PK-Sim Standard','Cellular permeability - PK-Sim Standard']
    sol=next(p for p in compound['Solubility'][0]['Parameters'] if p['Name']=='Solubility at reference pH')
    assert sol['Unit']=='mg/l';sol.update(Value=sol['Value']/1000,Unit='mg/ml')
    for p in compound['Processes']:p.setdefault('Species','Human')
    original=next(i for i in x['Individuals'] if i['Name']=='Haegeli 2007, 26y, male individuum')
    individual={k:_plain(original[k]) for k in ['Name','Seed','OriginData']}
    individual['Name']='IND_Furosemide_Haegeli_transport_engineering'
    individual['OriginData'].pop('CalculationMethods',None)
    proteins={p['Molecule'] for p in compound['Processes'] if 'Molecule'in p}
    ids=[i for i in original['ExpressionProfiles'] if i.split('|')[0] in proteins]
    individual['ExpressionProfiles']=ids
    profiles=[_plain(next(p for p in x['ExpressionProfiles'] if p['Molecule']+'|'+p['Species']+'|'+p['Category']==ident)) for ident in ids]
    protocol=_plain(next(p for p in x['Protocols'] if p['Name']=='i.v. (bolus), 20 mg'))
    protocol['Name']='PROTOCOL_Furosemide_source_20mg_iv'
    sources={field:reference(SOURCE,'Compounds[Furosemide].'+field+'; source unit conversion and source parameter semantics preserved')
             for field in compound if field not in {'Name','Description','IsSmallMolecule'}}
    sources['Parameters']=reference(IDENTITY,'PropertyTable.Properties[0].MolecularFormula; MW retained from original model')
    return {'individual':individual,'compound':compound,'protocol':protocol,'simulation_name':'furosemide_transport_engineering_20mg_iv',
            'snapshot_version':80,'human_source':reference(SOURCE,'Individuals[Haegeli 2007, 26y, male individuum].OriginData only'),
            'compound_sources':sources,'protocol_source':reference(SOURCE,'Protocols[i.v. (bolus), 20 mg]'),
            'generated_demographics':[n for n in ['Weight','Height'] if n not in individual['OriginData']],
            'expression_profiles':profiles,'expression_profile_sources':{i:reference(SOURCE,'ExpressionProfiles['+i+']') for i in ids},
            'output_schema':[{'Parameters':[{'Name':'Start time','Value':0.,'Unit':'h'}, {'Name':'End time','Value':6.,'Unit':'h'}, {'Name':'Resolution','Value':12.,'Unit':'pts/h'}]}],
            'output_selections':['Organism|PeripheralVenousBlood|Furosemide|Plasma (Peripheral Venous Blood)']}


def test_native_source_transport_profiles_preserve_every_direction_and_parameter():
    inputs=furosemide_inputs();before=deepcopy(inputs)
    built=build_human_model_snapshot(**inputs)
    assert inputs==before
    assert built['snapshot']['ExpressionProfiles']==inputs['expression_profiles']
    assert built['snapshot']['Individuals']==[inputs['individual']]
    assert built['snapshot']['Protocols']==[inputs['protocol']]
    assert built['snapshot']['Compounds'][0]['Processes']==inputs['compound']['Processes']
    assert 'active_transport' in built['manifest']['clearance_mechanisms']
    assert built['manifest']['elimination_configured']
    selections=built['snapshot']['Simulations'][0]['Compounds'][0]['Processes']
    assert {'Name':'OAT3-Ebner 2015','MoleculeName':'OAT3'} in selections
    assert {'Name':'ABCC4-Prasad 2020','MoleculeName':'ABCC4'} in selections


def test_transport_native_definitions_exist_in_locked_database():
    p=ROOT/'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/Db/PKSimDB.sqlite'
    with sqlite3.connect(p.as_uri()+'?mode=ro',uri=True) as connection:
        registered={r[0] for r in connection.execute('select transport_direction from tab_transport_directions')}
        assert set.union(*_TRANSPORT_DIRECTIONS.values())|{'None','ExcretionLiver','ExcretionKidney'}==registered
        for process in _TRANSPORT_PROCESS_PARAMETERS:
            assert connection.execute('select kinetic_type,process_type,action_type from tab_processes where process=?',(process,)).fetchall()==[('MM','ActiveTransport','TRANSPORT')]


@pytest.mark.parametrize('change,code', [
    (lambda p:p['Expression'].pop(),'MISSING_TRANSPORT_DIRECTIONS'),
    (lambda p:p['Expression'][0].update(TransportDirection='ExcretionKidney'),'UNSUPPORTED_TRANSPORT_DIRECTION'),
    (lambda p:p['Expression'][0].update(Name='InventedOrgan'),'UNSUPPORTED_TRANSPORT_DIRECTION'),
    (lambda p:p.update(TransportType='invented'),'UNSUPPORTED_TRANSPORT_TYPE'),
    (lambda p:p.update(Localization='Intracellular'),'UNSUPPORTED_FIELD'),
])
def test_invalid_native_transport_profile_fails_closed(change,code):
    inputs=furosemide_inputs();profile=next(p for p in inputs['expression_profiles'] if p['Type']=='Transporter');change(profile)
    with pytest.raises(HumanModelSnapshotError,match=code):build_human_model_snapshot(**inputs)


def test_transport_cannot_bind_an_enzyme_profile():
    inputs=furosemide_inputs();p=next(p for p in inputs['compound']['Processes'] if p['InternalName']=='ActiveTransportSpecific_MM');p['Molecule']='UGT1A9'
    with pytest.raises(HumanModelSnapshotError,match='UNBOUND_ENZYME'):build_human_model_snapshot(**inputs)


def test_specific_transport_requires_capacity_and_correct_physical_unit():
    inputs=furosemide_inputs();p=next(p for p in inputs['compound']['Processes'] if p['InternalName']=='ActiveTransportSpecific_MM')
    p['Parameters']=[v for v in p['Parameters'] if v['Name'] not in {'kcat','Vmax'}]
    with pytest.raises(HumanModelSnapshotError,match='MISSING_FIELDS'):build_human_model_snapshot(**inputs)


def furosemide_tablet_inputs():
    inputs=furosemide_inputs();x=json.loads((ROOT/SOURCE).read_text())
    formulation=_plain(next(f for f in x['Formulations'] if f['FormulationType']=='Formulation_Tablet_Weibull'))
    compound=next(c for c in x['Compounds'] if c['Name']=='Furosemide')
    permeability=_plain(compound['IntestinalPermeability'])
    assert permeability[0]['Parameters'][0]['Unit']=='cm/min'
    permeability[0]['Parameters'][0].update(Value=permeability[0]['Parameters'][0]['Value']/60,Unit='cm/s')
    inputs['compound']['IntestinalPermeability']=permeability
    inputs['compound_sources']['IntestinalPermeability']=reference(SOURCE,'Compounds[Furosemide].IntestinalPermeability; cm/min divided by 60 -> cm/s')
    inputs['protocol']=_plain(next(p for p in x['Protocols'] if p['Name']=='p.o. (tab), 40 mg'))
    inputs['protocol_source']=reference(SOURCE,'Protocols[p.o. (tab), 40 mg]; historical engineering protocol only')
    inputs['formulation']=formulation
    inputs['formulation_source']=reference(SOURCE,'Formulations[Immediate release]; all four historical source release parameters unchanged')
    inputs['simulation_name']='furosemide_transport_engineering_40mg_source_tablet'
    return inputs


def test_weibull_formulation_uses_exact_sourced_release_parameters():
    inputs=furosemide_tablet_inputs();before=deepcopy(inputs)
    built=build_human_model_snapshot(**inputs)
    assert built['snapshot']['Formulations']==[before['formulation']]
    assert built['snapshot']['Simulations'][0]['Compounds'][0]['Protocol']['Formulations']==[{'Key':'Formulation','Name':before['formulation']['Name']}]
    assert built['manifest']['normalized_administration']['route']=='oral'


@pytest.mark.parametrize('name,value,code', [('Dissolution time (50% dissolved)',0.,'VALUE_OUT_OF_RANGE'),
    ('Dissolution shape',-1.,'VALUE_OUT_OF_RANGE'),('Lag time',-1.,'VALUE_OUT_OF_RANGE'),('Use as suspension',.5,'UNSUPPORTED_FORMULATION')])
def test_weibull_invalid_release_parameters_fail_closed(name,value,code):
    inputs=furosemide_tablet_inputs();next(p for p in inputs['formulation']['Parameters'] if p['Name']==name)['Value']=value
    with pytest.raises(HumanModelSnapshotError,match=code):build_human_model_snapshot(**inputs)


def test_missing_source_release_parameter_must_not_use_native_default():
    inputs=furosemide_tablet_inputs();inputs['formulation']['Parameters'].pop()
    with pytest.raises(HumanModelSnapshotError,match='MISSING_FIELDS'):build_human_model_snapshot(**inputs)


def test_furosemide_source_specific_mm_has_its_own_native_linear_counterpart():
    from pbpk_kb.blueprints.mechanism_edits import compile_native_candidate,native_candidates,ACTION
    request=json.loads((ROOT/'runtime/blueprint_mechanism_worktree_20260920/native_verification/furosemide_source_20mg_iv/design_request.json').read_text())
    candidates=native_candidates(request,project_root=ROOT)
    assert len(candidates)==4
    linear=compile_native_candidate(request,'partition_pksim_standard__'+ACTION,project_root=ROOT)
    branches=linear['edit_manifest']['branches'];assert len(branches)==1
    b=branches[0];assert b['from_identity']['molecule']=='UGT1A9'
    assert b['to_identity']['internal_name']=='MetabolizationSpecific_FirstOrder'
    original=request['instance']['compound']['Processes']
    for before,after in zip(original,linear['request']['instance']['compound']['Processes']):
        if before.get('Molecule')!='UGT1A9':assert before==after
        else:
            assert next(p for p in before['Parameters'] if p['Name']=='Enzyme concentration')==next(p for p in after['Parameters'] if p['Name']=='Enzyme concentration')
            assert next(p['Value'] for p in after['Parameters'] if p['Name']=='CLspec/[Enzyme]')==b['kcat_anchor']/b['km_umol_per_l']
