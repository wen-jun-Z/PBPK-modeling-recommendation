"""The narrow expression projection never grants a generic numeric tolerance."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pytest

STAGED=Path(__file__).resolve().parents[2]
PROJECT=next(p for p in (STAGED,*STAGED.parents) if (p/'src/pbpk_kb').is_dir())
sys.path[:0]=[str(PROJECT/'scripts'),str(PROJECT/'src')]
spec=importlib.util.spec_from_file_location('recommendation_execution_repairs.json_expression_profile_proof',STAGED/'scripts/recommendation_execution_repairs/json_expression_profile_proof.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_full_precision_fraction_native_value_is_required():
    row={'Path':'Organism|Bone|Intracellular|ABCB1|Relative expression','Value':0.02425181886524823}
    native=ET.Element('Para',{'value':repr(row['Value']),'dim':'Fraction'})
    assert m._native_value(row,native)==(row['Value'],1.)
    native.set('value','0.0242518189')
    with pytest.raises(ValueError,match='exactly'):m._native_value(row,native)


def test_cyp2c8_preserves_exact_native_base_value_without_epsilon():
    row={'Path':'CYP2C8|t1/2 (liver)','Unit':'h','Value':23.000001424999983}
    native=ET.Element('Para',{'value':'1380.000085499999','dim':'Time','displayUnit':'h'})
    assert m._native_value(row,native)==(1380.000085499999,60.)
    native.set('value','1380.0000855')
    with pytest.raises(ValueError,match='exactly'):m._native_value(row,native)


@pytest.mark.parametrize('fault',['formula','rhs','dimension','unit','parameter','bool','nan'])
def test_scalar_proof_rejects_dynamic_or_wrong_dimension_inputs(fault):
    row={'Path':'CYP2C8|t1/2 (liver)','Unit':'h','Value':23.000001424999983}
    native=ET.Element('Para',{'value':'1380.000085499999','dim':'Time','displayUnit':'h'})
    if fault in ('formula','rhs'):native.set(fault,'dependent')
    if fault=='dimension':native.set('dim','Mass')
    if fault=='unit':row['Unit']='min'
    if fault=='parameter':row['Path']='other|t1/2 (liver)'
    if fault=='bool':row['Value']=True
    if fault=='nan':native.set('value','NaN')
    with pytest.raises(ValueError):m._native_value(row,native)


def source_documents():
    source={'Version':80,'ExpressionProfiles':[],'Simulations':[{'Name':'A','Parameters':[{'Path':'existing','Value':2.}],
        'Compounds':[{'Name':'Drug','Processes':[{'SystemicProcessType':'Hepatic'}]}]}]}
    export=deepcopy(source)
    export['Simulations'][0]['Parameters'] += [{'Path':k,**v} for k,v in m.DEFAULT_GLOBALS.items()]
    return source,export


def test_only_three_explicitly_registered_generated_globals_are_admitted():
    source,export=source_documents();result=m._simulation_exports(source,export)
    assert len(result['A'])==3 and len(source['Simulations'][0]['Parameters'])==1


@pytest.mark.parametrize('fault',['existing_value','missing_original','additional','default_value','unit','no_hepatic','other_field','extra_simulation'])
def test_entire_remainder_remains_exact(fault):
    source,export=source_documents();sim=export['Simulations'][0]
    if fault=='existing_value':sim['Parameters'][0]['Value']=2.000000000000001
    if fault=='missing_original':sim['Parameters'].pop(0)
    if fault=='additional':sim['Parameters'].append({'Path':'unsupported','Value':0.})
    if fault=='default_value':sim['Parameters'][1]['Value']=1.000000000000001
    if fault=='unit':sim['Parameters'][1]['Unit']='mg/l'
    if fault=='no_hepatic':
        source['Simulations'][0]['Compounds'][0]['Processes']=[];sim['Compounds'][0]['Processes']=[]
    if fault=='other_field':sim['Unexpected']=True
    if fault=='extra_simulation':export['Simulations'].append({'Name':'B'})
    with pytest.raises(ValueError):m._simulation_exports(source,export)


@pytest.mark.parametrize('index',[0,1,2])
def test_unknown_hash_triples_never_gain_expression_equivalence(index):
    refs=[{'path':'not-read.json','sha256':s} for s in (m.INPUT_SHA,m.EXPORT_SHA,m.NATIVE_SHA)]
    refs[index]['sha256']='f'*64
    with pytest.raises(ValueError,match='unregistered'):m.projection(*refs,project_root=PROJECT)
