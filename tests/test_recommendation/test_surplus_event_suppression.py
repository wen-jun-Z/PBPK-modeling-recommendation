"""Saved XML event conditions, not labels, justify the future-time barrier."""
from copy import deepcopy
import xml.etree.ElementTree as ET

import pytest

from scripts.recommendation_execution_repairs.surplus_event_suppression import prove_application_barrier


def model():
    return ET.fromstring('''<Model><Root name="root" id="root"><Children>
    <Container name="Applications" id="apps"><Children><Container name="P" id="p"><Children>
    <EventGroup name="Application_1" id="application"><Children>
      <Container name="ProtocolSchemaItem" id="protocol"><Children>
        <Para name="Start time" id="start" dim="Time" value="120" />
      </Children></Container>
      <Para name="Enabled" id="enabled" value="0" />
      <Event name="Arbitrary name X" id="begin" formula="begin_condition"><Children>
        <EventAssignment name="write" id="write1" changedEntity="enabled" formula="one" />
      </Children></Event>
      <Event name="Arbitrary name Y" id="finish" formula="finish_condition"><Children>
        <EventAssignment name="write" id="write0" changedEntity="enabled" formula="zero" />
      </Children></Event>
    </Children></EventGroup></Children></Container></Children></Container>
    </Children></Root><FormulaCache><Formulas>
      <Formula id="begin_condition" formula="Time=StartTime"><Paths>
        <Path as="StartTime" path="..|ProtocolSchemaItem|Start time" dim="Time" /><TimePath />
      </Paths></Formula>
      <Formula id="finish_condition" formula="(Enabled = 1) AND (Time &gt; 0)"><Paths>
        <Path as="Enabled" path="..|Enabled" /><TimePath />
      </Paths></Formula>
      <ConstantFormula id="one" value="1" /><ConstantFormula id="zero" value="0" />
    </Formulas></FormulaCache></Model>''')


PREFIX = 'Events|P|Application_1'


def test_time_and_private_state_are_proven_without_using_event_names():
    tree = model(); before = ET.tostring(tree)
    proof = prove_application_barrier(tree, PREFIX, 'start')
    assert proof['verified'] and proof['all_application_events_accounted']
    assert len(proof['time_bound_events']) == len(proof['inactive_guarded_events']) == 1
    assert proof['time_bound_events'][0]['event_path'].endswith('Arbitrary name X')
    assert ET.tostring(tree) == before


@pytest.mark.parametrize('mutation', ['wrong_start_formula', 'wrong_start_target', 'initial_active',
                                    'dynamic_gate', 'mass_state_gate', 'start_time_writer',
                                    'unguarded_stop', 'or_bypass', 'external_writer'])
def test_unproven_event_conditions_or_gate_activation_reject_adaptation(mutation):
    tree = model()
    formulas = {node.get('id'): node for node in tree.findall('FormulaCache/Formulas/*')}
    if mutation == 'wrong_start_formula': formulas['begin_condition'].set('formula', 'Time=0')
    elif mutation == 'wrong_start_target': formulas['begin_condition'].find('Paths/Path').set('path', '..|Enabled')
    elif mutation == 'initial_active': next(n for n in tree.iter('Para') if n.get('id')=='enabled').set('value','1')
    elif mutation == 'dynamic_gate': next(n for n in tree.iter('Para') if n.get('id')=='enabled').set('rhs','one')
    elif mutation == 'mass_state_gate': next(n for n in tree.iter('Para') if n.get('id')=='enabled').tag='MoleculeAmount'
    elif mutation == 'start_time_writer': next(tree.iter('EventAssignment')).set('changedEntity','start')
    elif mutation == 'unguarded_stop': formulas['finish_condition'].set('formula','Time > 0')
    elif mutation == 'or_bypass': formulas['finish_condition'].set('formula','(Enabled = 1) AND (Time > 0) OR (Time = 0)')
    else:
        children = tree.find('Root/Children')
        event = ET.SubElement(children, 'Event', {'name':'outside','id':'outside','formula':'one'})
        ET.SubElement(event, 'EventAssignment', {'name':'write','id':'external','changedEntity':'enabled','formula':'one'})
    with pytest.raises(ValueError, match='SURPLUS'):
        prove_application_barrier(tree, PREFIX, 'start')


def test_foreign_parameter_id_cannot_be_validated_by_familiar_name():
    with pytest.raises(ValueError, match='ID_OR_DIMENSION'):
        prove_application_barrier(model(), PREFIX, 'not_the_start_parameter')


def test_original_iv_stop_after_nonnegative_infusion_is_also_future():
    tree=model(); container=next(n for n in tree.iter('Container') if n.get('id')=='protocol')
    ET.SubElement(container.find('Children'),'Para',{'id':'duration','name':'Infusion time','dim':'Time','value':'5'})
    f=next(n for n in tree.iter('Formula') if n.get('id')=='finish_condition')
    f.set('formula','Time=StartTime+InfusionTime'); f.remove(f.find('Paths'))
    paths=ET.SubElement(f,'Paths');ET.SubElement(paths,'Path',{'as':'StartTime','path':'..|ProtocolSchemaItem|Start time'})
    ET.SubElement(paths,'Path',{'as':'InfusionTime','path':'..|ProtocolSchemaItem|Infusion time'});ET.SubElement(paths,'TimePath')
    proof=prove_application_barrier(tree,PREFIX,'start')
    assert len(proof['time_bound_events'])==2
    assert sorted(e['nonnegative_infusion_offset_min'] for e in proof['time_bound_events'])==[0,5]
    next(n for n in tree.iter('Para') if n.get('id')=='duration').set('value','-1')
    with pytest.raises(ValueError,match='INFUSION_STOP_DURATION'):prove_application_barrier(tree,PREFIX,'start')
