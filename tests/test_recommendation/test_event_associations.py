"""Every link must hold before an original IV event gains a drug identity."""
from copy import deepcopy
from pathlib import Path
import json
import sqlite3
import sys
import tempfile
import xml.etree.ElementTree as E
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from recommendation_execution_repairs import event_associations as m


@pytest.fixture
def tmp_path():
    parent=ROOT/'runtime/tmp';parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='event-association-',dir=parent) as folder:yield Path(folder)


def document():
    d=E.Element('IndividualSimulation',id='simulation',name='Source')
    used=E.SubElement(d,'UsedBuildingBlocks')
    for tag,attrs in [('Compound',{'id':'compound','name':'Drug'}),('SimpleProtocol',{'id':'protocol','name':'Protocol','applicationType':'Intravenous'})]:
        w=E.SubElement(used,'UsedBuildingBlock');E.SubElement(w,tag,**attrs)
    p=E.SubElement(E.SubElement(d,'Properties'),'CompoundPropertiesList')
    c=E.SubElement(p,'CompoundProperties',compound='compound');E.SubElement(c,'ProtocolProperties',protocol='protocol')
    model=E.SubElement(d,'Model');root=E.SubElement(model,'Root',id='root',name='Source')
    def child(parent,tag,identifier,name,**attrs):return E.SubElement(parent,tag,id=identifier,name=name,**attrs)
    drug=child(root,'Container','drugdef','Drug');child(drug,'Para','mw','Molecular weight',dim='Molecular weight',value='300')
    org=child(root,'Container','org','Organism');blood=child(org,'Container','blood','VenousBlood');plasma=child(blood,'Container','plasma','Plasma')
    child(plasma,'MoleculeAmount','target','Drug',dim='Amount',quantityType='Drug')
    apps=child(root,'Container','apps','Applications');group=child(apps,'EventGroup','group','Administration',eventGroupType='Intravenous')
    child(group,'Para','rate','Application rate',dim='Amount per time',value='0')
    schema=child(group,'Container','schema','ProtocolSchemaItem')
    child(schema,'Para','dose','Dose',dim='Mass',value='1')
    mass=child(schema,'Para','mass','DrugMass',dim='Amount',formula='massformula');tags=E.SubElement(mass,'Tags');E.SubElement(tags,'Tag',value='Drug')
    for identifier,name in [('start','Start time'),('infusion','Infusion time')]:
        p=child(schema,'Para',identifier,name,dim='Time',value='1');E.SubElement(p,'Origin',bb='protocol')
    state=child(group,'MoleculeAmount','state','Drug',dim='Amount',quantityType='Drug')
    child(state,'Transport','transport','Transport',dimension='Amount per time',formula='transportformula',sourceAmount='state',targetAmount='target')
    for label,formula in [('on','onformula'),('off','offformula')]:
        event=child(group,'Event',label,label);child(event,'EventAssignment',label+'assignment','Assignment',changedEntity='rate',objectPath='..|..|Application rate',formula=formula)
    cache=E.SubElement(model,'FormulaCache');formulas=E.SubElement(cache,'Formulas')
    def formula(identifier,expression,refs):
        f=E.SubElement(formulas,'Formula',id=identifier,formula=expression);paths=E.SubElement(f,'Paths')
        for alias,path in refs.items():E.SubElement(paths,'Path',**{'as':alias,'path':path})
    prefix='Applications|Administration|ProtocolSchemaItem|'
    formula('massformula','Dose / MW',{'Dose':'..|Dose','MW':'Source|Drug|Molecular weight'})
    formula('transportformula','ApplicationRate',{'ApplicationRate':'..|..|Application rate'})
    formula('onformula','DrugMass / InfusionTime',{'DrugMass':prefix+'DrugMass','InfusionTime':prefix+'Infusion time'})
    E.SubElement(formulas,'ConstantFormula',id='offformula',value='0')
    return d


def node(d,identifier):return next(n for n in d.iter() if n.get('id')==identifier)


def prove(d,metadata=None):
    md=metadata or m.source_pkml.describe_saved_simulation(d)
    return m._prove_indirect_iv(m.ModelIndex(d),md['administrations'][0],md)


def test_complete_chain_identifies_parent_not_protocol_display_name():
    d=document();node(d,'protocol').set('name','Voriconazole - iv (a misleading display name)')
    meta,audits=m._enrich_document(d)
    assert meta['administrations'][0]['administered_compounds']==['Drug']
    assert meta['administrations'][0]['compound_association_complete'] is True
    assert audits[0]['venous_target_state_path']=='Organism|VenousBlood|Plasma|Drug'


def test_external_transport_cannot_share_the_application_rate():
    d=document();other=deepcopy(node(d,'transport'));other.set('id','othertransport');other.set('name','OtherTransport');other.set('sourceAmount','target')
    other.set('formula','externalformula');node(d,'target').append(other)
    f=E.SubElement(d.find('Model/FormulaCache/Formulas'),'Formula',id='externalformula',formula='ApplicationRate')
    E.SubElement(E.SubElement(f,'Paths'),'Path',**{'as':'ApplicationRate','path':'Applications|Administration|Application rate'})
    with pytest.raises(ValueError,match='AMBIGUOUS_TRANSPORT_CONSUMERS'):prove(d)


def test_only_unreferenced_official_time_path_is_permitted():
    d=document();E.SubElement(node(d,'onformula').find('Paths'),'TimePath')
    assert prove(d)['assignment_proofs'][0]['unreferenced_time_path_count']==1
    node(d,'onformula').find('Paths/TimePath').set('as','DrugMass')
    with pytest.raises(ValueError,match='UNREFERENCED_TIME_PATH'):prove(d)


def mutate(d,case):
    setv=lambda i,k,v:node(d,i).set(k,v)
    if case=='group_route':setv('group','eventGroupType','Oral')
    elif case=='rate_dimension':setv('rate','dim','Time')
    elif case=='rate_rhs':setv('rate','rhs','dynamic')
    elif case=='state_type':setv('state','quantityType','Protein')
    elif case=='state_dimension':setv('state','dim','Mass')
    elif case=='state_name':setv('state','name','Other')
    elif case=='extra_state':E.SubElement(node(d,'group'),'MoleculeAmount',id='extra',name='Other',quantityType='Drug',dim='Amount')
    elif case=='target_type':setv('target','quantityType','Protein')
    elif case=='target_dimension':setv('target','dim','Mass')
    elif case=='transport_source':setv('transport','sourceAmount','target')
    elif case=='transport_target':setv('transport','targetAmount','state')
    elif case=='extra_transport':
        extra=deepcopy(node(d,'transport'));extra.set('id','extra');extra.set('name','Extra');node(d,'state').append(extra)
    elif case=='transport_dimension':setv('transport','dimension','Time')
    elif case=='transport_formula':setv('transportformula','formula','ApplicationRate * 2')
    elif case=='transport_path':node(d,'transportformula').find('Paths/Path').set('path','..|..|ProtocolSchemaItem|Dose')
    elif case=='mass_dimension':setv('mass','dim','Mass')
    elif case=='mass_tag':node(d,'mass').find('Tags/Tag').set('value','Other')
    elif case=='mass_formula':setv('massformula','formula','Dose * MW')
    elif case=='mass_weight_path':node(d,'massformula').findall('Paths/Path')[1].set('path','Applications|Administration|ProtocolSchemaItem|Dose')
    elif case=='dose_dimension':setv('dose','dim','Amount')
    elif case=='start_origin':node(d,'start').find('Origin').attrib.pop('bb')
    elif case=='infusion_origin':node(d,'infusion').find('Origin').set('bb','other')
    elif case=='compound_binding':d.find('Properties/CompoundPropertiesList/CompoundProperties').set('compound','other')
    elif case=='protocol_type':node(d,'protocol').tag='Compound'
    elif case=='protocol_route':setv('protocol','applicationType','Oral')
    elif case=='extra_assignment':
        extra=deepcopy(node(d,'onassignment'));extra.set('id','extra');extra.set('name','Extra');node(d,'on').append(extra)
    elif case=='assignment_target':setv('onassignment','changedEntity','dose')
    elif case=='assignment_object_path':setv('onassignment','objectPath','..|..|ProtocolSchemaItem|Dose')
    elif case=='missing_assignment_formula':setv('onassignment','formula','missing')
    elif case=='stop_nonzero':setv('offformula','value','1')
    elif case=='start_expression':setv('onformula','formula','DrugMass * InfusionTime')
    elif case=='start_path':node(d,'onformula').find('Paths/Path').set('path','Drug|Molecular weight')
    elif case=='two_stops':setv('onassignment','formula','offformula')
    elif case=='duplicate_formula_alias':node(d,'onformula').findall('Paths/Path')[1].set('as','DrugMass')
    else:raise AssertionError(case)


BROKEN_CHAINS=['group_route','rate_dimension','rate_rhs','state_type','state_dimension','state_name','extra_state',
 'target_type','target_dimension','transport_source','transport_target','extra_transport','transport_dimension',
 'transport_formula','transport_path','mass_dimension','mass_tag','mass_formula','mass_weight_path','dose_dimension',
 'start_origin','infusion_origin','compound_binding','protocol_type','protocol_route','extra_assignment','assignment_target',
 'assignment_object_path','missing_assignment_formula','stop_nonzero','start_expression','start_path','two_stops','duplicate_formula_alias']


@pytest.mark.parametrize('case',BROKEN_CHAINS)
def test_each_of_34_broken_identity_links_is_rejected(case):
    d=document();mutate(d,case)
    with pytest.raises(ValueError,match='EVENT_ASSOCIATION_'):prove(d)


def project(tmp_path):
    d=document();blob=m.source_pkml.compress_model_blob(E.tostring(d));path=tmp_path/'original.pksim5'
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE SIMULATIONS (SimulationId TEXT,SimulationMode INTEGER); CREATE TABLE BUILDING_BLOCKS (Id TEXT,Name TEXT,ContentId TEXT); CREATE TABLE CONTENTS (Id TEXT,Data BLOB);')
        db.execute('INSERT INTO SIMULATIONS VALUES (?,?)',('simulation',0));db.execute('INSERT INTO BUILDING_BLOCKS VALUES (?,?,?)',('simulation','Source','content'));db.execute('INSERT INTO CONTENTS VALUES (?,?)',('content',blob))
    return path,m.source_pkml.sha(path),m.hashlib.sha256(blob).hexdigest()


def test_source_and_blob_binding_and_original_metadata_are_required(tmp_path):
    path,sha,blob=project(tmp_path)
    result=m.enrich_source_metadata(path,sha,'simulation',blob)
    assert result['audit']['parameters_modified'] is False
    for badsha,badblob in [('0'*64,blob),(sha,'0'*64)]:
        with pytest.raises(ValueError,match='SHA_MISMATCH'):m.enrich_source_metadata(path,badsha,'simulation',badblob)
    metadata=m.source_pkml.describe_saved_simulation(document());metadata['compounds']=['Other']
    with pytest.raises(ValueError,match='METADATA_DIFFERS'):m.enrich_source_metadata(path,sha,'simulation',blob,metadata)


def test_route_absence_recomputed_no_borrowing_other_drug_route(tmp_path):
    path,sha,blob=project(tmp_path)
    cert=m.create_route_absence_certificate(path,sha,'Drug','Oral')
    assert cert['simulation_count']==1 and cert['observed_routes']==['Intravenous']
    assert m.validate_route_absence_certificate(cert)['accepted'] is True
    with pytest.raises(ValueError,match='EXPECTED_SCOPE_DIFFERS'):
        m.validate_route_absence_certificate(cert,compound='Other')
    altered=deepcopy(cert);altered['target_administration_count']=0
    with pytest.raises(ValueError,match='CERTIFICATE_CONTENT'):m.validate_route_absence_certificate(altered)
    with pytest.raises(ValueError,match='ROUTE_EXISTS'):m.create_route_absence_certificate(path,sha,'Drug','Intravenous')
    with pytest.raises(ValueError,match='ADMINISTRATION_NOT_FOUND'):m.create_route_absence_certificate(path,sha,'Voriconazole','Intravenous')


@pytest.mark.parametrize('source_index,simulation_index',[(0,0),(0,1),(1,0)])
def test_three_actual_source_identity_repairs(source_index,simulation_index):
    data=json.loads((ROOT/'runtime/deployment/event_compound_association_diagnosis_20261001/diagnosis.json').read_text())
    source=data['sources'][source_index];sim=source['simulations'][simulation_index]
    result=m.enrich_source_metadata(source['source_path'],source['source_sha256'],sim['simulation_id'],sim['source_blob_sha256'])
    assert len(result['audit']['enriched_administrations'])==1
    assert all(row['compound_association_complete'] for row in result['enriched_metadata']['administrations'])
    expected='Buprenorphine' if source_index==0 else 'Fentanyl'
    assert result['audit']['enriched_administrations'][0]['drug']==expected


def exported_document(original):
    native=deepcopy(original)
    for identifier in ('start','infusion'):
        parameter=node(native,identifier);parameter.remove(parameter.find('Origin'))
    E.SubElement(node(native,'onformula').find('Paths'),'TimePath',alias='Time')
    return native


def test_expanded_iv_uses_unchanged_original_origin_without_modifying_xml():
    original=document();native=exported_document(original)
    before=(E.tostring(original),E.tostring(native))
    with pytest.raises(ValueError,match='PROTOCOL_PARAMETER_ORIGIN_REQUIRED'):
        m._enrich_document(native)
    metadata,proofs=m._enrich_document(native,original_document=original)
    assert metadata['administrations'][0]['administered_compounds']==['Drug']
    evidence=proofs[0]['original_protocol_origin_evidence']
    assert [r['id'] for r in evidence]==['start','infusion']
    assert all(r['original_origin']=={'bb':'protocol'} and not r['native_origin_present'] for r in evidence)
    assert (E.tostring(original),E.tostring(native))==before


@pytest.mark.parametrize('case',[
    'original_origin_missing','native_origin_conflict','event_id','parameter_id',
    'parameter_value','parameter_dimension','parameter_formula','transport_formula',
    'assignment_target','mass_tag','source_protocol_binding',
])
def test_expanded_origin_evidence_rejects_changed_identity_value_or_chain(case):
    original=document();native=exported_document(original)
    if case=='original_origin_missing':
        node(original,'start').remove(node(original,'start').find('Origin'))
    elif case=='native_origin_conflict':E.SubElement(node(native,'start'),'Origin',bb='other')
    elif case=='event_id':node(native,'group').set('id','different-event')
    elif case=='parameter_id':node(native,'start').set('id','different-start')
    elif case=='parameter_value':node(native,'infusion').set('value','2')
    elif case=='parameter_dimension':node(native,'start').set('dim','Mass')
    elif case=='parameter_formula':node(native,'start').set('formula','offformula')
    elif case=='source_protocol_binding':
        original.find('Properties/CompoundPropertiesList/CompoundProperties').set('compound','other')
    else:mutate(native,case)
    with pytest.raises(ValueError,match='EVENT_ASSOCIATION_'):
        m._enrich_document(native,original_document=original)


@pytest.mark.parametrize('attrs',[{'alias':'Other'},{'as':'Time','alias':'Time'},{'alias':'Time','unexpected':'1'}])
def test_time_path_serialization_alias_is_strict(attrs):
    d=document();E.SubElement(node(d,'onformula').find('Paths'),'TimePath',**attrs)
    with pytest.raises(ValueError,match='UNREFERENCED_TIME_PATH_REQUIRED'):prove(d)
