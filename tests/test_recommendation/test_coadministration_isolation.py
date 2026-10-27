"""Scientific zero-exposure invariants; labels alone never qualify a model."""
from copy import deepcopy
import xml.etree.ElementTree as E
import pytest
from scripts.recommendation_execution_repairs import coadministration_isolation as iso
from scripts.recommendation_execution_repairs.event_associations import ModelIndex


def model():
    d=E.Element('IndividualSimulation',name='Source',id='sim');m=E.SubElement(d,'Model');r=E.SubElement(m,'Root',name='Source',id='root')
    def node(parent,tag,name,ident,**attrs):return E.SubElement(parent,tag,name=name,id=ident,**attrs)
    a=node(r,'Container','Applications','apps')
    administrations=[];parameters=[]
    f=E.SubElement(E.SubElement(m,'FormulaCache'),'Formulas')
    for name in ['Target','Other']:
        g=node(a,'EventGroup',name+'App',name+'app',eventGroupType='Oral');s=node(g,'Container','ProtocolSchemaItem',name+'schema')
        for n,dim,val in [('Dose','Mass','1'),('Start time','Time','0')]:
            p=node(s,'Para',n,name+n,dim=dim,value=val);parameters.append({'path':'Applications|'+name+'App|ProtocolSchemaItem|'+n,'id':p.get('id')})
        e=node(g,'Event','Start',name+'event',formula=name+'condition')
        formula=E.SubElement(f,'Formula',id=name+'condition',formula='Time=StartTime');ps=E.SubElement(formula,'Paths')
        E.SubElement(ps,'Path',**{'as':'StartTime','path':'..|ProtocolSchemaItem|Start time'});E.SubElement(ps,'TimePath')
        administrations.append({'event_prefix':'Applications|'+name+'App','administered_compounds':[name], 'compound_association_complete':True})
    body=node(r,'Container','Body','body')
    for name in ['Target','Other','TargetMet','OtherMet']:
        node(body,'MoleculeAmount',name,name,dim='Amount',quantityType='Drug',negativeValuesAllowed='0',value='0')
    for name in ['Target','Other']:
        rx=node(body,'Reaction',name+'Reaction',name+'rx',formula=name+'flux')
        E.SubElement(E.SubElement(rx,'Educts'),'ReactionPartner',partner=name)
        E.SubElement(E.SubElement(rx,'Products'),'ReactionPartner',partner=name+'Met')
        fm=E.SubElement(f,'Formula',id=name+'flux',formula='Mass');E.SubElement(E.SubElement(fm,'Paths'),'Path',**{'as':'Mass','path':'Body|'+name})
    return d,{'administrations':administrations,'event_parameters':parameters}


def test_actual_parent_closure_retains_target_explicit_compound_metabolite():
    doc,md=model();before=E.tostring(doc);p=iso.prove_zero_manifold(doc,md,'Target')
    assert p['removed_administered_parents']==['Other']
    assert p['removed_reaction_product_closure']==['Other','OtherMet']
    assert p['retained_target_product_closure']==['Target','TargetMet']
    assert p['physical_state_count']==2 and p['zero_flux_count']==1
    assert E.tostring(doc)==before


@pytest.mark.parametrize('case',['nonzero_initial','target_production','constant_production','unknown_math','external_event','unresolved_parent'])
def test_real_zero_exposure_failures_cannot_be_called_isolated(case):
    doc,md=model();idx=ModelIndex(doc)
    if case=='nonzero_initial':idx.paths['Body|OtherMet'].set('value','1')
    elif case=='target_production':idx.paths['Body|TargetReaction'].find('Products/ReactionPartner').set('partner','Other')
    elif case=='constant_production':idx.formulas['Otherflux'].set('formula','Mass + 1')
    elif case=='unknown_math':idx.formulas['Otherflux'].set('formula','mystery(Mass)')
    elif case=='external_event':
        E.SubElement(idx.paths['Applications|TargetApp|Start'],'EventAssignment',name='WriteOther',id='outside',changedEntity='Other',formula='Otherflux')
    else:md['administrations'][1]['compound_association_complete']=False
    with pytest.raises(ValueError):iso.prove_zero_manifold(doc,md,'Target')


@pytest.mark.parametrize('expression,expected', [('0 = 0 & 1 = 1 ? 0 : 1',0),('NOT 0 ? 3 : 4',3),
    ('(0 > 1 ? 3 : 4) + 1',5),('0 = 1 OR 2 > 1 ? 8 : 9',8),('2^3',8)])
def test_native_formula_grammar_is_explicit_and_never_python_eval(expression,expected):
    doc,_=model();idx=ModelIndex(doc);p=iso.FormulaProof(idx,[],[])
    f=E.Element('Formula',formula=expression)
    assert p.expression(f,'Body|Other')==expected


def test_dynamic_volume_is_conditional_and_registered_never_static():
    doc,_=model();idx=ModelIndex(doc);other=idx.paths['Body|Other'];other.set('dim','Volume')
    p=iso.FormulaProof(idx,[],[]);v=p.value('Body|Other')
    assert v.is_positive and not p.is_static('Body|Other')
    assert p.dynamic_positive_paths=={'Body|Other'}


def test_unknown_dynamic_denominator_is_not_canceled_away():
    doc,_=model();idx=ModelIndex(doc);p=iso.FormulaProof(idx,['Body|Other'],[])
    f=E.Element('Formula',formula='X / Y');paths=E.SubElement(f,'Paths')
    E.SubElement(paths,'Path',**{'as':'X','path':'Body|Other'});E.SubElement(paths,'Path',**{'as':'Y','path':'Body|Target'})
    with pytest.raises(ValueError,match='DENOMINATOR'):p.expression(f,'Body|Other')


def test_zero_multiplier_does_not_require_a_positive_clearance_coefficient():
    doc,_=model();idx=ModelIndex(doc);p=iso.FormulaProof(idx,[],[])
    path='Applications|TargetApp|ProtocolSchemaItem|Dose'
    coefficient=p.static_symbol(path);p.cache[path]=coefficient
    f=E.Element('Formula',formula='0 * K');refs=E.SubElement(f,'Paths')
    E.SubElement(refs,'Path',**{'as':'K','path':path})
    assert p.expression(f,'Body|Other')==0
    assert coefficient.is_positive is None and not p.positive_paths
    p.denominator(coefficient)
    assert p.positive_paths=={path}


def test_static_switch_cannot_assume_a_zero_capable_coefficient_is_positive():
    doc,_=model();idx=ModelIndex(doc);p=iso.FormulaProof(idx,[],[])
    path='Applications|TargetApp|ProtocolSchemaItem|Dose'
    coefficient=p.static_symbol(path);p.cache[path]=coefficient
    f=E.Element('Formula',formula='K > 0 ? 1 : 0');refs=E.SubElement(f,'Paths')
    E.SubElement(refs,'Path',**{'as':'K','path':path})
    expression=p.expression(f,'Body|Other')
    assert expression.subs(coefficient,0)==0 and expression.subs(coefficient,1)==1


def test_selection_view_never_mutates_full_source():
    _,md=model();before=deepcopy(md);view=iso.target_metadata(md,'Target')
    assert len(view['administrations'])==1 and md==before and len(md['administrations'])==2


def patch_fixture():
    _,metadata=model()
    rows=[]
    for r in metadata['event_parameters']:
        rows.append({**r,'value':1.,'unit':'kg' if r['path'].endswith('|Dose') else 'min'})
    contract={'administration_times_min':[0.],'dose_mg':2.,'route':'Oral'}
    prefix='Events|OtherApp';event={'event_prefix':prefix,'dose_parameter_path':prefix+'|ProtocolSchemaItem|Dose',
        'start_parameter_path':prefix+'|ProtocolSchemaItem|Start time','dose_parameter_id':'OtherDose',
        'start_parameter_id':'OtherStart time','new_start_time_min':120.,'administered_compound':'Other','barrier':{'verified':True}}
    policy={'verified':True,'target':'Target','source_metadata_sha256':iso._digest(metadata),
        'target_selection_metadata_sha256':iso._digest(iso.target_metadata(metadata,'Target')),
        'contract_sha256':iso._digest(contract),'events':[event]}
    return {'parameters':rows},contract,metadata,policy


def test_native_patches_account_for_both_target_and_other_original_events():
    inv,c,md,p=patch_fixture()
    def target(rows,contract,metadata,**kw):
        assert len(rows['parameters'])==2 and len(metadata['administrations'])==1
        return [],[]
    patches,events=iso.build_isolated_patches(inv,c,md,p,target_patch_builder=target)
    assert [r['value'] for r in patches]==[0.,120.]
    assert all(r['path'].startswith('Applications|OtherApp|') for r in patches)
    assert events[0]['disposition']=='other_parent_zero_dose_event_moved_beyond_output_window'
    assert len(md['administrations'])==2


@pytest.mark.parametrize('mutation',['missing_event','duplicate_event','wrong_id','wrong_unit','wrong_contract','wrong_metadata','wrong_prefix'])
def test_isolation_patch_binding_is_exact(mutation):
    inv,c,md,p=patch_fixture()
    if mutation=='missing_event': inv['parameters'].pop()
    elif mutation=='duplicate_event': inv['parameters'].append(deepcopy(inv['parameters'][0]))
    elif mutation=='wrong_id': inv['parameters'][-1]['id']='foreign'
    elif mutation=='wrong_unit': inv['parameters'][-1]['unit']='day/kg'
    elif mutation=='wrong_contract':c['dose_mg']=3.
    elif mutation=='wrong_metadata':md['administrations'][1]['administered_compounds']=['Wrong']
    else:p['events'][0]['event_prefix']='Events|WrongApp'
    with pytest.raises((ValueError,KeyError)):
        iso.build_isolated_patches(inv,c,md,p,target_patch_builder=lambda *a,**k:([],[]))


@pytest.mark.parametrize('formula,requires_positive',[
    ('K > 0 ? 0 / K : 0',False),
    ('K = 0 ? 0 : 0 / K',False),
    ('K > 0 ? 0 : 0 / K',True),
    ('(K > 0 ? 0 / K : 0) + 0 / K',True),
])
def test_guarded_zero_volume_does_not_impose_a_global_positive_domain(formula,requires_positive):
    doc,_=model();idx=ModelIndex(doc);p=iso.FormulaProof(idx,[],[])
    path='Applications|TargetApp|ProtocolSchemaItem|Dose'
    coefficient=p.static_symbol(path);p.cache[path]=coefficient
    f=E.Element('Formula',formula=formula);refs=E.SubElement(f,'Paths')
    E.SubElement(refs,'Path',**{'as':'K','path':path})
    assert p.expression(f,'Body|Other')==0
    assert (path in p.positive_paths)==requires_positive


def test_positive_denominator_with_guarded_zero_volume_keeps_local_assumption():
    doc,_=model();idx=ModelIndex(doc);p=iso.FormulaProof(idx,[],[])
    path='Applications|TargetApp|ProtocolSchemaItem|Dose'
    coefficient=p.static_symbol(path);p.cache[path]=coefficient
    f=E.Element('Formula',formula='0 / (1 + (K > 0 ? 2/K : 0))');refs=E.SubElement(f,'Paths')
    E.SubElement(refs,'Path',**{'as':'K','path':path})
    assert p.expression(f,'Body|Other')==0
    assert path not in p.positive_paths


def test_conditional_denominator_with_a_zero_branch_is_rejected():
    doc,_=model();idx=ModelIndex(doc);p=iso.FormulaProof(idx,[],[])
    path='Applications|TargetApp|ProtocolSchemaItem|Dose';p.cache[path]=p.static_symbol(path)
    f=E.Element('Formula',formula='0 / (K > 0 ? 2/K : 0)');refs=E.SubElement(f,'Paths')
    E.SubElement(refs,'Path',**{'as':'K','path':path})
    with pytest.raises(ValueError,match='DENOMINATOR'):p.expression(f,'Body|Other')


def test_exact_rational_normalization_proves_static_partition_denominator():
    doc,_=model();idx=ModelIndex(doc);p=iso.FormulaProof(idx,[],[])
    path='Applications|TargetApp|ProtocolSchemaItem|Dose';k=p.static_symbol(path)
    value=k*(iso.sy.Rational(17,1000)*10**k-iso.sy.Rational(76,5)/(1+10**iso.sy.Rational(23,50))+iso.sy.Rational(823391,35600))
    assert value.is_nonzero is None
    p.denominator(value)
    assert p.positive_paths=={path}


def test_exact_rational_normalization_cannot_accept_a_possible_zero_denominator():
    doc,_=model();idx=ModelIndex(doc);p=iso.FormulaProof(idx,[],[])
    k=p.static_symbol('Applications|TargetApp|ProtocolSchemaItem|Dose')
    with pytest.raises(ValueError,match='DENOMINATOR'):p.denominator(k-1)
