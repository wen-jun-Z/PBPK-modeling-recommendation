"""Preserve complete executable equations/topology across XML identity changes."""
from copy import deepcopy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from recommendation_execution_repairs.model_semantics import compare_models, compare_saved_settings_xml


def model():
    return ET.fromstring('''<Model neighborhoods="n"><Root id="root" name="Root"><Children>
      <Container id="a" name="A" containerType="Compartment" mode="Physical"><Children>
        <Parameter id="p" name="P" value="2" dim="Time"/>
        <MoleculeAmount id="m" name="Drug" formula="f" dim="Amount"/>
        <MoleculeAmount id="m2" name="Metabolite" formula="f" dim="Amount"/>
        <Reaction id="r" name="Metabolism" formula="f"><Educts><Educt partner="m" stoichiometry="1"/></Educts><Products><Product partner="m2" stoichiometry="1"/></Products></Reaction>
        <Transport id="t" name="Flow" sourceAmount="m" targetAmount="m2" formula="f"/>
        <EventAssignment id="e" name="Event" changedEntity="p" objectPath="Root|A|P" formula="f"/>
      </Children></Container>
      <Container id="b" name="B" containerType="Compartment" mode="Physical"/>
      <Neighborhoods id="n" name="Neighborhoods"><Children><Neighborhood id="nb" name="A_B" firstNeighbor="a" secondNeighbor="b"/></Children></Neighborhoods>
    </Children></Root><FormulaCache><StringMap><String id="s1" s="..|P"/><String id="s2" s="P"/><String id="s3" s="Time"/></StringMap><Formulas><Formula id="f" name="rate" formula="P + Time"><Paths><Path path="s1" as="s2" dim="s3"/><TimePath/></Paths></Formula></Formulas></FormulaCache></Model>''')


def test_cache_identity_and_equivalent_legacy_aliases_do_not_change_semantics():
    a, b = model(), model()
    f = b.find('.//Formula'); f.set('id', 'new-f')
    for n in b.iter():
        if n.get('formula') == 'f': n.set('formula', 'new-f')
    p = b.find('.//Parameter'); p.tag = 'Para'; p.set('mode', 'Local')
    c = b.find('.//Container'); c.tag = 'Compartment'; c.set('visible', '1')
    b.find('.//TimePath').set('alias', 'Time')
    obj = b.find('.//EventAssignment'); obj.attrib.pop('objectPath')
    ET.SubElement(obj, 'ObjectPath', {'path': 'Root|A|P'})
    assert compare_models(a, b)['accepted']


def test_infusion_display_unit_does_not_rescale_the_serialized_base_value():
    a=model();container=a.find('.//Container');container.set('name','ProtocolSchemaItem')
    p=a.find('.//Parameter');p.set('name','Infusion time');p.set('displayUnit','min')
    b=deepcopy(a);b.find('.//Parameter').set('displayUnit','h')
    assert compare_models(a,b)['accepted']
    b.find('.//Parameter').set('value','120')
    assert not compare_models(a,b)['accepted']


def test_infusion_display_exception_does_not_ignore_wrong_dimensions_or_unknown_units():
    a=model();a.find('.//Container').set('name','ProtocolSchemaItem')
    p=a.find('.//Parameter');p.set('name','Infusion time');p.set('displayUnit','min')
    for field,value in [('dim','Mass'),('displayUnit','mg')]:
        b=deepcopy(a);b.find('.//Parameter').set(field,value)
        assert not compare_models(a,b)['accepted']


@pytest.mark.parametrize('xpath,field,value', [
    ('.//Parameter', 'value', '3'), ('.//Parameter', 'dim', 'Mass'),
    ('.//Formula', 'formula', 'P - Time'), ('.//Path', 'path', 'Root|B|P'),
    ('.//EventAssignment', 'changedEntity', 'm'), ('.//EventAssignment', 'objectPath', 'Root|A|Drug'),
    ('.//Educt', 'stoichiometry', '2'), ('.//Product', 'partner', 'm'),
    ('.//Transport', 'sourceAmount', 'm2'), ('.//Neighborhood', 'secondNeighbor', 'a'),
    ('.//Container', 'containerType', 'Other'), ('.//Container', 'unexpected_mechanism', 'enabled'),
])
def test_real_equation_topology_or_unknown_field_change_is_rejected(xpath, field, value):
    a, b = model(), model(); b.find(xpath).set(field, value)
    assert compare_models(a, b)['accepted'] is False


def test_rhs_and_unknown_nodes_are_not_ignored():
    a, b = model(), model(); b.find('.//Parameter').set('rHSFormula', 'f')
    assert not compare_models(a, b)['accepted']
    a, b = model(), model(); ET.SubElement(b.find('.//Transport'), 'NovelMechanism')
    assert not compare_models(a, b)['accepted']


def test_nonfinite_is_preserved_as_unknown_not_success_and_dangling_fails():
    a, b = model(), model()
    a.find('.//Parameter').set('value', 'NaN'); b.find('.//Parameter').set('value', 'NaN')
    result = compare_models(a, b)
    assert result['accepted']
    assert result['counts']['source']['nonfinite_placeholders']
    assert result['counts']['source']['nonfinite_is_numerical_success'] is False
    b.find('.//Transport').set('sourceAmount', 'absent')
    with pytest.raises(ValueError, match='Dangling'): compare_models(a, b)


def settings():
    return ET.fromstring('''<SimulationSettings id="s" name="Simulation" randomSeed="123"><Creation createdAt="1"/><Solver id="x" name="CVODES"><Children><Para name="Relative tolerance" value="1e-5"/></Children></Solver><OutputSchema><OutputInterval name="I"><Children><Para name="End time" value="60" dim="Time"/></Children></OutputInterval></OutputSchema><OutputSelections><QuantitySelection path="A|Drug"/></OutputSelections></SimulationSettings>''')


@pytest.mark.parametrize('xpath,field,value', [('.', 'randomSeed', '124'), ('.//Para', 'value', '0.1'), ('.//QuantitySelection', 'path', 'B|Drug'), ('.//OutputInterval', 'new_option', '1')])
def test_settings_seed_solver_output_and_unknown_fields_remain_strict(xpath, field, value):
    a, b = settings(), settings(); b.find(xpath).set(field, value)
    assert not compare_saved_settings_xml(a, b)['accepted']


def test_settings_creation_and_serialization_ids_only_are_excluded():
    a, b = settings(), settings(); b.set('id', 'new'); b.find('Creation').set('createdAt', '2')
    assert compare_saved_settings_xml(a, b)['accepted']


def test_constant_formula_unknown_fields_cannot_disappear():
    a, b = model(), model()
    ET.SubElement(b.find('FormulaCache/Formulas'), 'ConstantFormula', {'id': 'c', 'value': '2', 'dim': 'Time', 'unknown': '1'})
    b.find('.//Parameter').set('formula', 'c')
    with pytest.raises(ValueError, match='unreviewed'): compare_models(a, b)


def test_settings_own_formula_cache_expands_and_cannot_hide_changed_solver_value():
    a, b = settings(), settings()
    cache = ET.SubElement(b, 'FormulaCache'); forms = ET.SubElement(cache, 'Formulas')
    f = ET.SubElement(forms, 'ConstantFormula', {'id': 'f', 'value': '1e-5'})
    parameter = b.find('.//Para'); parameter.attrib.pop('value'); parameter.set('formula', 'f')
    assert compare_saved_settings_xml(a, b)['accepted']
    f.set('value', '1e-2')
    assert not compare_saved_settings_xml(a, b)['accepted']


def test_reset_provenance_and_moved_description_not_equations_but_origin_method_is():
    a, b = model(), model()
    a.find('.//Parameter').set('valueDescription', 'original citation')
    b.find('.//Parameter').set('isDefault', '1')
    origin = ET.SubElement(b.find('.//Parameter'), 'ValueOrigin', {'description': 'original citation'})
    assert compare_models(a, b)['accepted']
    origin.set('method', 'FittedToTarget')
    assert not compare_models(a, b)['accepted']


def test_official_tablet_display_order_change_never_waives_parameter_values():
    a,b=model(),model()
    for tree,seq in [(a,'1'),(b,'20')]:
        p=tree.find('.//Parameter');p.set('name','Tablet time delay factor')
        ET.SubElement(p,'Info',{'seq':seq})
    assert compare_models(a,b)['accepted']
    b.find('.//Parameter').set('value','20')
    assert not compare_models(a,b)['accepted']


@pytest.mark.parametrize('kind',['other_parameter','other_sequence','non_parameter'])
def test_ui_sequence_exception_is_restricted_to_documented_field(kind):
    a,b=model(),model()
    for tree,seq in [(a,'1'),(b,'20')]:
        p=tree.find('.//Parameter')
        if kind!='other_parameter':p.set('name','Tablet time delay factor')
        if kind=='non_parameter':p.tag='UnreviewedEntity'
        ET.SubElement(p,'Info',{'seq':seq})
    if kind=='other_sequence':b.find('.//Info').set('seq','21')
    assert not compare_models(a,b)['accepted']


def percentile_model():
    doc=model();p=doc.find('.//Parameter');p.tag='DistributedParameter'
    child=ET.SubElement(p,'Children')
    ET.SubElement(child,'Parameter',id='pct',name='Percentile',value='0.5',displayUnit='l')
    return doc


def test_invalid_display_annotation_on_dimensionless_distribution_percentile_is_ui_only():
    before=percentile_model();after=deepcopy(before)
    after.find('.//Parameter').attrib.pop('displayUnit')
    assert compare_models(before,after)['accepted']


@pytest.mark.parametrize('field,value',[('value','0.500000000000001'),('dim','Volume')])
def test_percentile_value_or_dimension_changes_still_fail(field,value):
    before=percentile_model();after=deepcopy(before)
    p=after.find('.//Parameter');p.attrib.pop('displayUnit');p.set(field,value)
    assert not compare_models(before,after)['accepted']


def test_percentile_exception_cannot_hide_another_parameter_unit():
    before=percentile_model();after=deepcopy(before)
    for m in (before,after):m.find('.//Parameter').set('name','Actual volume')
    after.find('.//Parameter').attrib.pop('displayUnit')
    assert not compare_models(before,after)['accepted']


@pytest.mark.parametrize('source_tag,export_tag', [('Settings','SimulationSettings'),('SimulationSettings','Settings'),('Settings','Settings')])
def test_saved_settings_property_alias_is_semantic_only(source_tag,export_tag):
    source,exported=settings(),settings();source.tag=source_tag;exported.tag=export_tag
    originals=(ET.tostring(source),ET.tostring(exported))
    assert compare_saved_settings_xml(source,exported)['accepted']
    assert (ET.tostring(source),ET.tostring(exported))==originals
    for xpath,key,value in [('.', 'randomSeed','456'),('./Solver/Children/Para','value','1e-4'),('./OutputSchema/OutputInterval/Children/Para','value','120'),('./OutputSelections/QuantitySelection','path','A|OtherDrug'),('.', 'unknownScientificFlag','1')]:
        changed=deepcopy(exported);changed.find(xpath).set(key,value)
        assert not compare_saved_settings_xml(source,changed)['accepted']


def test_arbitrary_settings_element_name_is_not_an_alias():
    source=settings();source.tag='OtherSettings'
    with pytest.raises(ValueError,match='Settings or SimulationSettings'):
        compare_saved_settings_xml(source,settings())
