from copy import deepcopy
import math
import pytest
from pbpk_kb.blueprints import source_adaptation as a


def item(time=0.,dose=10.,water=3.5):
    return {'Name':'author item','ApplicationType':'Oral','FormulationKey':'Formulation','Parameters':[
        {'Name':'Start time','Value':time,'Unit':'min'},
        {'Name':'InputDose','Value':dose,'Unit':'mg'},
        {'Name':'Volume of water/body weight','Value':water,'Unit':'ml/kg'}]}

def schema(start=0.,repeat=1.,interval=0.,dose=10.):
    return {'SchemaItems':[item(dose=dose)],'Parameters':[
        {'Name':'Start time','Value':start,'Unit':'min'},
        {'Name':'NumberOfRepetitions','Value':repeat},
        {'Name':'TimeBetweenRepetitions','Value':interval,'Unit':'min'}]}

def test_multiple_schemas_follow_official_expansion_and_do_not_mutate_source():
    p={'Schemas':[schema(1440.,3.,1440.,75.),schema(0.,dose=300.)]};old=deepcopy(p)
    items,times=a._administrations(p)
    assert times==[0.,1440.,2880.,4320.] and len(items)==2 and p==old
    assert [a._parameters(x)['InputDose']['Value'] for x in items]==[75.,300.]


def test_missing_schema_scalars_use_only_literal_official_factory_defaults():
    p={'Schemas':[{'SchemaItems':[item()]},schema(1440.)]}
    assert a._administrations(p)[1]==[0.,1440.]


@pytest.mark.parametrize('mutation,expected',[
    (lambda p:p['Schemas'][0]['SchemaItems'].append(item(2.3,3.66)), 'MULTI_ITEM_SOURCE_REGIMEN'),
    (lambda p:p['Schemas'][1]['Parameters'][0].update(Value=0.), 'AMBIGUOUS_SIMULTANEOUS'),
    (lambda p:p['Schemas'][1]['Parameters'][1].update(Value=31.), 'REPETITION_COUNT'),
    (lambda p:p['Schemas'][1]['Parameters'][1].update(Value=1.2), 'REPETITION_COUNT'),
    (lambda p:p['Schemas'][1]['Parameters'][1].update(Value=2.), 'REPETITION_COUNT'),
    (lambda p:p['Schemas'][1]['Parameters'][0].update(TableFormula={'Points':[]}), 'DYNAMIC_SOURCE_SCHEMA'),
    (lambda p:p['Schemas'][1]['SchemaItems'][0]['Parameters'][0].update(TableFormula={'Points':[]}), 'DYNAMIC_SOURCE_DOSE'),
    (lambda p:p['Schemas'][1]['SchemaItems'][0]['Parameters'][2].update(Value=5.), 'NONUNIFORM_SOURCE_VEHICLE'),
    (lambda p:p['Schemas'][1]['SchemaItems'][0].update(FormulationKey='other'), 'NONUNIFORM_SOURCE_VEHICLE'),
    (lambda p:p['Schemas'][1].update(CustomEvent={}), 'UNREVIEWED_SOURCE_SCHEMA_FIELDS'),
    (lambda p:p['Schemas'][1]['Parameters'].append({'Name':'unknown','Value':0.}), 'UNREVIEWED_SOURCE_SCHEMA_PARAMETERS'),
])
def test_ambiguous_or_nonuniform_source_regimens_remain_blocked(mutation,expected):
    p={'Schemas':[schema(),schema(1440.)]};mutation(p)
    with pytest.raises(a.SourceAdaptationError,match=expected):a._administrations(p)


def test_total_source_schedule_remains_bounded():
    p={'Schemas':[schema(0.,20.,1440.),schema(600.,20.,1440.)]}
    with pytest.raises(a.SourceAdaptationError,match='EXCEEDS_THIRTY'):a._administrations(p)


def test_absolute_prescription_rebinds_input_dose_not_physiology():
    p={'Name':'InputDose','Value':10.,'Unit':'mg/kg','ValueOrigin':{'Source':'Publication'}}
    a._bind_requested_dose(p,{'value':500.,'unit':'mg'})
    assert p=={'Name':'InputDose','Value':500.,'Unit':'mg'}


def test_weighted_to_weighted_conversion_remains_exact():
    p={'Name':'InputDose','Value':10.,'Unit':'mg/kg'}
    a._bind_requested_dose(p,{'value':1000.,'unit':'µg/kg'})
    assert p['Value']==1. and p['Unit']=='mg/kg'


def test_absolute_to_weighted_is_not_an_invented_body_weight_conversion():
    with pytest.raises(a.SourceAdaptationError,match='BODY_WEIGHT_CONVERSION'):
        a._bind_requested_dose({'Name':'InputDose','Value':10.,'Unit':'mg'}, {'value':1.,'unit':'mg/kg'})


def test_weight_normalized_table_dose_remains_blocked():
    p={'Name':'InputDose','Value':10.,'Unit':'mg/kg','TableFormula':{'Points':[[0,1]]}}
    with pytest.raises(a.SourceAdaptationError,match='WEIGHT_DOSE_TABLE'):a._bind_requested_dose(p,{'value':500.,'unit':'mg'})


def test_multi_item_split_absorption_is_never_reduced_to_single_dose():
    p={'Schemas':[{'SchemaItems':[item(0.,6.34),item(138.,3.66)],'Parameters':schema()['Parameters']}]}
    with pytest.raises(a.SourceAdaptationError,match='MULTI_ITEM_SOURCE_REGIMEN'):a._administrations(p)
