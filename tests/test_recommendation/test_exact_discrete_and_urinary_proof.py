from copy import deepcopy
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as E
import pytest
from scripts.recommendation_execution_repairs import coadministration_isolation as iso
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
spec=importlib.util.spec_from_file_location('domain_original_fixture',ROOT/'tests/test_recommendation/test_coadministration_isolation.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def discrete(value='0.47'):
    doc,metadata=old.model();root=doc.find('./Model/Root');formulas=doc.find('./Model/FormulaCache/Formulas')
    organism=E.SubElement(root,'Container',name='Organism',id='org')
    node=E.SubElement(organism,'DistPara',name='Hematocrit',id='hct',formula='discrete',default='0.99')
    children=E.SubElement(node,'Children');mean=E.SubElement(children,'Para',name='Mean',id='mean',value=value)
    E.SubElement(children,'Para',name='Percentile',id='percentile',value='0.01')
    f=E.SubElement(formulas,'DiscreteDistributionFormula',id='discrete')
    E.SubElement(E.SubElement(f,'Paths'),'Path',path='Mean',**{'as':'Mean'})
    proof=iso.FormulaProof(iso.association.ModelIndex(doc),[],[])
    symbol=proof.static_symbol('Organism|Hematocrit')
    return doc,metadata,mean,f,proof,symbol


def prove(p):
    proof,symbol=p[-2:]
    proof.denominator((symbol-iso.sy.Rational('0.1'))/symbol)
    return proof


def test_exact_discrete_uses_explicit_mean_not_default_or_percentile():
    p=discrete();before=E.tostring(p[0]);proof=prove(p)
    row=proof.discrete_expansions['Organism|Hematocrit']
    assert row['exact_rational_value']=='47/100' and not row['default_value_used'] and not row['percentile_used']
    assert row['native_readback_required'] and E.tostring(p[0])==before


def test_prior_successful_symbolic_domain_does_not_acquire_new_output():
    p=discrete();p[-2].domain(p[-1],'positive','UNPROVED')
    assert p[-2].discrete_expansions=={}


@pytest.mark.parametrize('fault',['zero_denominator','dynamic_mean','event_written_mean','normal_distribution','unbound_mean','missing_value','nonfinite','wrong_dimension'])
def test_no_source_default_or_assumed_physio_bound_can_replace_exact_value(fault):
    p=discrete();doc,metadata,mean,f,proof,symbol=p
    if fault=='zero_denominator':mean.set('value','0.1')
    elif fault=='dynamic_mean':mean.set('rhs','dynamic')
    elif fault=='event_written_mean':E.SubElement(doc.find('./Model/Root'),'EventAssignment',id='write',name='write',changedEntity='mean',formula='Otherflux')
    elif fault=='normal_distribution':f.tag='NormalDistributionFormula'
    elif fault=='unbound_mean':f.find('Paths/Path').set('as','Other')
    elif fault=='missing_value':mean.attrib.pop('value')
    elif fault=='nonfinite':mean.set('value','NaN')
    else:proof.index.paths['Organism|Hematocrit'].set('dim','Mass')
    proof=iso.FormulaProof(iso.association.ModelIndex(doc),[],[]);symbol=proof.static_symbol('Organism|Hematocrit')
    with pytest.raises((ValueError,KeyError)):prove((None,None,None,None,proof,symbol))


@pytest.fixture
def exact_readback():
    proof=prove(discrete());rows=list(proof.discrete_expansions.values())
    return {'static_discrete_parameter_expansions':rows,'evidence_refs':deepcopy(iso.DISCRETE_DISTRIBUTION_EVIDENCE)}, {'proof_parameter_readback':[{'path':'Organism|Hematocrit','id':'hct','dimension':'Dimensionless','unit':'','value':.47}]}


def test_native_independently_reproduces_exact_mean_and_explicit_percent_unit(exact_readback):
    proof,native=exact_readback;assert iso.verify_discrete_parameter_readback(native,proof)[0]['verified']
    native['proof_parameter_readback'][0].update(unit='%',value=47.)
    assert iso.verify_discrete_parameter_readback(native,proof)[0]['verified']


@pytest.mark.parametrize('fault',['missing','duplicate','id','dimension','unit','value','nonfinite'])
def test_native_discrete_readback_requires_exact_identity_units_and_value(exact_readback,fault):
    proof,native=deepcopy(exact_readback);rows=native['proof_parameter_readback'];r=rows[0]
    if fault=='missing':rows.clear()
    elif fault=='duplicate':rows.append(deepcopy(r))
    elif fault=='id':r['id']='other'
    elif fault=='dimension':r['dimension']='Mass'
    elif fault=='unit':r['unit']='mg'
    elif fault=='value':r['value']+=1e-12
    else:r['value']=float('nan')
    with pytest.raises(ValueError):iso.verify_discrete_parameter_readback(native,proof)


def urinary():
    doc,metadata=old.model();root=doc.find('./Model/Root');formulas=doc.find('./Model/FormulaCache/Formulas')
    org=E.SubElement(root,'Container',name='Organism',id='org')
    kidney=E.SubElement(org,'Container',name='Kidney',id='kidney');urine=E.SubElement(kidney,'Container',name='Urine',id='urine')
    E.SubElement(urine,'MoleculeAmount',name='Other',id='urineOther',quantityType='Drug',dim='Amount',value='0')
    events=E.SubElement(root,'Container',name='Events',id='events')
    outer=E.SubElement(events,'EventGroup',name='Urinary bladder emptying',id='outer',containerType='EventGroup')
    group=E.SubElement(outer,'EventGroup',name='Urinary bladder emptying_1',id='group',containerType='EventGroup')
    E.SubElement(group,'Para',name='Start time',id='urinaryTime',dim='Time',value='30')
    event=E.SubElement(group,'Event',name='Empty',id='empty',formula='trigger')
    writer=E.SubElement(event,'EventAssignment',name='Reset',id='reset',changedEntity='urineOther',formula='zero',useAsValue='1')
    trigger=E.SubElement(formulas,'Formula',id='trigger',formula='Time=StartTime');paths=E.SubElement(trigger,'Paths')
    E.SubElement(paths,'TimePath');E.SubElement(paths,'Path',path='..|Start time',**{'as':'StartTime'})
    zero=E.SubElement(formulas,'ConstantFormula',id='zero',value='0')
    path='Events|Urinary bladder emptying|Urinary bladder emptying_1|Start time'
    metadata['event_parameters'].append({'path':path,'id':'urinaryTime','dimension':'Time','value':'30'})
    return doc,metadata,group,writer,zero


def test_original_urinary_reset_preserves_zero_mass_and_full_source():
    doc,metadata,*_=urinary();before=E.tostring(doc)
    proof=iso._prove_zero_manifold_uncached(doc,metadata,'Target')
    assert proof['verified'] and proof['preserved_urinary_physiology']['verified']
    assert 'Organism|Kidney|Urine|Other' in proof['zero_monitor_paths']
    assert any(x['changed_entity']=='Organism|Kidney|Urine|Other' for x in proof['event_writers'])
    assert E.tostring(doc)==before


@pytest.mark.parametrize('fault',['nonzero','dynamic','formula_switch','non_urine_target','hidden_dose','wrong_time_identity'])
def test_urinary_label_never_hides_dose_external_input_or_parameter_change(fault):
    doc,metadata,group,writer,zero=urinary()
    if fault=='nonzero':zero.set('value','0.0001')
    elif fault=='dynamic':zero.tag='Formula';zero.set('formula','0')
    elif fault=='formula_switch':writer.set('useAsValue','0')
    elif fault=='non_urine_target':writer.set('changedEntity','Other')
    elif fault=='hidden_dose':E.SubElement(group,'Para',name='Dose',id='hidden',dim='Mass',value='1')
    else:metadata['event_parameters'][-1]['id']='wrong'
    with pytest.raises(ValueError):iso.preserved_urinary_events(doc,metadata)
