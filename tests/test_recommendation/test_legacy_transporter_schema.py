"""Legacy transport expressions must preserve the effective native model."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from recommendation_execution_repairs.snapshot_schema import (
    _legacy_transporter, _transport_direction, map_legacy_expression_schema)
from recommendation_execution_repairs.complex_snapshot import parameter_expectations


def fixture():
    molecule={'Name':'ABCB1','Type':'Transporter','TransportType':'Efflux',
        'Parameters':[{'Name':'Reference concentration','Value':2,'Unit':'µmol/l',
                       'ValueOrigin':{'Source':'Publication','Description':'unchanged source'}}],
        'Expression':[
            {'Name':'Duodenum','MembraneLocation':'Apical','Value':1.41},
            {'Name':'Bone','MembraneLocation':'Basolateral','Value':0.0341950646},
            {'Name':'Brain','MembraneLocation':'BloodBrainBarrier','Value':0.5},
            {'Name':'Kidney','MembraneLocation':'Basolateral','Value':0.2}]}
    native={'Molecule':'ABCB1','Species':'Human','Category':'Human','Type':'Transporter',
        'TransportType':'Efflux','Expression':[
            {'Name':'Duodenum','CompartmentName':'Intracellular','TransportDirection':'EffluxMucosaIntracellularToLumen'},
            {'Name':'Bone','CompartmentName':'Interstitial','TransportDirection':'EffluxIntracellularToInterstitial'},
            {'Name':'Brain','CompartmentName':'Plasma','TransportDirection':'EffluxBrainInterstitialToPlasma'},
            {'Name':'Brain','CompartmentName':'Interstitial','TransportDirection':'EffluxBrainTissueToInterstitial'},
            {'Name':'Kidney','CompartmentName':'Intracellular','TransportDirection':'ExcretionKidney'}],
        # Official JSON output rounds doubles. It must never become input truth.
        'Parameters':[{'Path':'Organism|Bone|Intracellular|ABCB1|Relative expression','Value':0.0242518189}]}
    return molecule,native


def test_source_normalization_retains_full_precision_and_raw_provenance():
    original,native=fixture();before=deepcopy(original);changes=[]
    mapped=_legacy_transporter(original,native,changes,'Individual')
    params={p['Path']:p for p in mapped['Parameters']}
    value=params['Organism|Bone|Intracellular|ABCB1|Relative expression']['Value']
    assert value==0.0341950646/1.41
    assert value!=native['Parameters'][0]['Value']
    assert original==before
    assert changes[0]['source_molecule_record']==before
    assert changes[0]['native_numeric_values_copied'] is False
    record=params['ABCB1|Reference concentration']
    assert record=={'Path':'ABCB1|Reference concentration',**{k:v for k,v in before['Parameters'][0].items() if k!='Name'}}
    assert changes[0]['source_to_target_values'][1]['source_value']==0.0341950646
    assert changes[0]['source_to_target_values'][1]['denominator']==1.41


def test_membrane_fraction_mapping_preserves_source_localization():
    original,native=fixture();mapped=_legacy_transporter(original,native,[],'Individual')
    values={p['Path']:p['Value'] for p in mapped['Parameters']}
    assert values['Organism|SmallIntestine|Mucosa|Duodenum|Intracellular|ABCB1|Fraction expressed apical']==1
    assert values['Organism|Kidney|Intracellular|ABCB1|Fraction expressed apical']==0
    assert values['Organism|Brain|Plasma|ABCB1|Fraction expressed at blood brain barrier']==1


def test_native_directions_cannot_override_source_transport_type():
    original,native=fixture();native['Expression'][0]['TransportDirection']='InfluxLumenToMucosaIntracellular'
    with pytest.raises(ValueError,match='NATIVE_TRANSPORT_DIRECTION_DIFFERS'):
        _legacy_transporter(original,native,[],'Individual')


@pytest.mark.parametrize('mutate,reason',[
    (lambda m,n:m['Expression'][0].update(Unexpected='x'),'UNREGISTERED_LEGACY_TRANSPORT_EXPRESSION_FIELDS'),
    (lambda m,n:m['Parameters'][0].update(Path='ABCB1|x'),'MIXED_LEGACY_TRANSPORTER_PARAMETER_IDENTITY'),
    (lambda m,n:n['Expression'][0].update(Value=999),'UNEXPECTED_NATIVE_TRANSPORT_EXPRESSION_FIELDS'),
    (lambda m,n:m['Expression'][1].update(MembraneLocation='Apical'),'UNREGISTERED_LEGACY_TISSUE_MEMBRANE'),
    (lambda m,n:m['Expression'][1].update(Value=float('nan')),'FINITE_NONNEGATIVE'),
])
def test_unrecognized_or_ambiguous_fields_fail_closed(mutate,reason):
    original,native=fixture();mutate(original,native)
    with pytest.raises(ValueError,match=reason):_legacy_transporter(original,native,[],'Individual')


def test_zero_expression_and_modern_schema_are_not_rescaled():
    original,native=fixture()
    for row in original['Expression']:row['Value']=0
    mapped=_legacy_transporter(original,native,[],'Individual')
    assert all(p['Value']==0 for p in mapped['Parameters'] if p['Path'].endswith('Relative expression'))
    modern={'Name':'ABCB1',**native};modern.pop('Molecule');changes=[]
    assert _legacy_transporter(modern,native,changes,'Individual')==modern and changes==[]


def test_transporter_conversion_requires_whole_native_proof_and_removes_path_error():
    molecule,native=fixture()
    original={'Version':77,'Individuals':[{'Name':'person','OriginData':{'Species':'Human'},'Molecules':[molecule]}],
              'Simulations':[{'Name':'simulation','Individual':'person'}]}
    converted={'Version':80,'Individuals':[{'Name':'person','OriginData':{'Species':'Human'},
                'ExpressionProfiles':['ABCB1|Human|Human']}],
               'ExpressionProfiles':[native],'Simulations':deepcopy(original['Simulations'])}
    mapped=map_legacy_expression_schema(original,converted)
    assert mapped['requires_complete_native_model_proof'] is True
    snapshot=mapped['mapped_snapshot']
    expectations=parameter_expectations(snapshot,snapshot['Simulations'][0])
    assert len(expectations['parameters'])==8
    assert all('path' in p for p in expectations['parameters'])


@pytest.mark.parametrize('kind,region,compartment,direction',[
    ('Efflux','Kidney','Intracellular','ExcretionKidney'),
    ('Influx','Pericentral','Intracellular','ExcretionLiver'),
    ('Influx','BloodCells',None,'InfluxPlasmaToBloodCells'),
    ('PgpLike','VascularEndothelium',None,'EffluxInterstitialToPlasma'),
    ('PgpLike','Brain','Plasma','PgpBrainInterstitialToPlasma'),
    ('Influx','Duodenum','Intracellular','InfluxLumenToMucosaIntracellular'),
])
def test_registered_official_transport_directions(kind,region,compartment,direction):
    assert _transport_direction(kind,region,compartment)==direction
