from copy import deepcopy
from pathlib import Path
import importlib.util
import xml.etree.ElementTree as E
import pytest
from scripts.recommendation_execution_repairs import coadministration_isolation as iso
from scripts.recommendation_execution_repairs.event_associations import ModelIndex
ROOT=Path('/home/xuyiqian/Research_Projects/Knowledge_Graph_Model_Recommendation')
spec=importlib.util.spec_from_file_location('existing_zero_fixture',ROOT/'tests/test_recommendation/test_coadministration_isolation.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def model():
    document,metadata=old.model();index=ModelIndex(document)
    body=index.paths['Body'];formulas=document.find('./Model/FormulaCache/Formulas')
    protein=E.SubElement(body,'MoleculeAmount',name='SharedReceptor',id='receptor',dim='Amount',quantityType='OtherProtein',value='12',negativeValuesAllowed='0')
    complex=E.SubElement(body,'MoleculeAmount',name='UnrelatedName',id='complex',dim='Amount',quantityType='Complex',value='0',negativeValuesAllowed='0')
    rx=E.SubElement(body,'Reaction',name='Binding',id='binding',formula='bind')
    educts=E.SubElement(rx,'Educts');E.SubElement(educts,'ReactionPartner',partner='Other');E.SubElement(educts,'ReactionPartner',partner='receptor')
    E.SubElement(E.SubElement(rx,'Products'),'ReactionPartner',partner='complex')
    formula=E.SubElement(formulas,'Formula',id='bind',formula='Drug * Receptor - Complex');paths=E.SubElement(formula,'Paths')
    for alias,path in [('Drug','Body|Other'),('Receptor','Body|SharedReceptor'),('Complex','Body|UnrelatedName')]:E.SubElement(paths,'Path',**{'as':alias,'path':path})
    return document,metadata


def test_zero_closure_includes_typed_binding_product_without_zeroing_receptor():
    doc,metadata=model();before=E.tostring(doc);p=iso.prove_zero_manifold(doc,metadata,'Target')
    assert 'UnrelatedName' in p['removed_reaction_product_closure']
    assert 'Body|UnrelatedName' in p['zero_monitor_paths']
    assert 'Body|SharedReceptor' not in p['zero_monitor_paths']
    assert p['physical_state_count']==3 and p['zero_flux_count']==2
    assert E.tostring(doc)==before


@pytest.mark.parametrize('failure',['nonzero_initial','target_production','external_production','unknown_quantity_type','external_writer'])
def test_binding_complex_never_weakens_zero_manifold_requirements(failure):
    doc,metadata=model();index=ModelIndex(doc)
    if failure=='nonzero_initial':index.paths['Body|UnrelatedName'].set('value','1')
    elif failure=='target_production':index.paths['Body|TargetReaction'].find('Products/ReactionPartner').set('partner','complex')
    elif failure=='external_production':index.formulas['bind'].set('formula','Drug * Receptor - Complex + 1')
    elif failure=='unknown_quantity_type':index.paths['Body|UnrelatedName'].set('quantityType','OtherProtein')
    else:E.SubElement(index.paths['Applications|TargetApp|Start'],'EventAssignment',name='WriteComplex',id='writeComplex',changedEntity='complex',formula='Targetflux')
    with pytest.raises(ValueError):iso.prove_zero_manifold(doc,metadata,'Target')


def dual_model():
    import json
    document, metadata=old.model()
    document=E.fromstring(E.tostring(document).decode().replace('Target','Esomeprazole'))
    metadata=json.loads(json.dumps(metadata).replace('Target','Esomeprazole'))
    index=ModelIndex(document)
    def clone(node):return E.fromstring(E.tostring(node).decode().replace('Esomeprazole','R-omeprazole'))
    document.find('./Model/Root').find("Container[@name='Applications']").append(clone(index.paths['Applications|EsomeprazoleApp']))
    for key in ['Body|Esomeprazole','Body|EsomeprazoleMet','Body|EsomeprazoleReaction']:
        index.paths['Body'].append(clone(index.paths[key]))
    formulas=document.find('./Model/FormulaCache/Formulas')
    for key in ['Esomeprazolecondition','Esomeprazoleflux']:formulas.append(clone(index.formulas[key]))
    metadata['administrations'].append(json.loads(json.dumps(metadata['administrations'][0]).replace('Esomeprazole','R-omeprazole')))
    metadata['event_parameters'] += [json.loads(json.dumps(r).replace('Esomeprazole','R-omeprazole'))
        for r in metadata['event_parameters'] if 'Esomeprazole' in r['path']]
    return document,metadata


def test_both_registered_parents_and_both_metabolite_closures_are_retained():
    document,metadata=dual_model();before=E.tostring(document)
    proof=iso.prove_zero_manifold(document,metadata,['Esomeprazole','R-omeprazole'])
    assert proof['removed_administered_parents']==['Other']
    assert proof['retained_target_product_closure']==['Esomeprazole','EsomeprazoleMet','R-omeprazole','R-omeprazoleMet']
    assert not any('omeprazole' in x for x in proof['zero_monitor_paths'])
    view=iso.target_metadata(metadata,['Esomeprazole','R-omeprazole'])
    assert len(view['administrations'])==2 and len(metadata['administrations'])==3
    assert E.tostring(document)==before


@pytest.mark.parametrize('failure',['r_production_into_removed','missing_r','nonzero_removed_initial'])
def test_dual_retention_does_not_weaken_zero_proof(failure):
    document,metadata=dual_model();idx=ModelIndex(document)
    if failure=='r_production_into_removed':idx.paths['Body|R-omeprazoleReaction'].find('Products/ReactionPartner').set('partner','Other')
    elif failure=='missing_r':metadata['administrations']=[a for a in metadata['administrations'] if a['administered_compounds']!=['R-omeprazole']]
    else:idx.paths['Body|Other'].set('value','1')
    with pytest.raises(ValueError):iso.prove_zero_manifold(document,metadata,['Esomeprazole','R-omeprazole'])


@pytest.mark.parametrize('target',[['Esomeprazole'],['R-omeprazole','Esomeprazole'],['Esomeprazole','Other'],['Esomeprazole','R-omeprazole','Other']])
def test_no_unregistered_retained_parent_collection(target):
    document,metadata=dual_model()
    with pytest.raises(ValueError,match='EXACT_REGISTERED_RACEMATE'):iso.prove_zero_manifold(document,metadata,target)
