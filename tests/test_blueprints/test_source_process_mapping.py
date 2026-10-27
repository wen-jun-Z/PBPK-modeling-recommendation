"""Native name and mapping semantics; no solver, outcomes or parameter fitting."""
from copy import deepcopy
import pytest
from pbpk_kb.blueprints import source_adaptation as a


def snapshot():
    return {'Version':80,'Individuals':[{'Name':'Subject','OriginData':{'Species':'Human'},
             'ExpressionProfiles':['Enzyme human|Human|Enzyme']}],
      'ExpressionProfiles':[{'Molecule':'Enzyme human','Species':'Human','Category':'Enzyme'}],
      'Compounds':[{'Name':'Parent','IsSmallMolecule':True,'Processes':[
          {'Molecule':'Enzyme reference','DataSource':'author-data','InternalName':'rCYP450_MM','Metabolite':'Original metabolite label'}]}],
      'Protocols':[{'Name':'Dose'}],
      'Simulations':[{'Name':'S','Model':'4Comp','Individual':'Subject','Compounds':[
           {'Name':'Parent','Protocol':{'Name':'Dose'},'Processes':[
              {'Name':'Enzyme reference-author-data','MoleculeName':'Enzyme human'}]}]}]}


def test_process_name_separate_from_physiological_molecule_mapping_is_retained():
    d=snapshot();before=deepcopy(d)
    r=a._check_dependencies(d)
    assert r['accepted'] and d==before
    assert r['active_processes']==['Enzyme reference-author-data']
    assert r['explicit_metabolite_links']==[]
    assert r['unselected_metabolite_defaults']==[{'parent':'Parent','process':'Enzyme reference-author-data','unselected_default':'Original metabolite label'}]


@pytest.mark.parametrize('fault',['name','ambiguous_name','unknown_human_molecule','wrong_source_molecule'])
def test_mapping_fix_still_rejects_unbound_or_ambiguous_names(fault):
    d=snapshot()
    if fault=='name':d['Simulations'][0]['Compounds'][0]['Processes'][0]['Name']='unbound'
    elif fault=='ambiguous_name':d['Compounds'][0]['Processes']*=2
    elif fault=='unknown_human_molecule':d['Simulations'][0]['Compounds'][0]['Processes'][0]['MoleculeName']='unknown'
    else:d['Compounds'][0]['Processes'][0]['Molecule']='wrong'
    with pytest.raises(a.SourceAdaptationError):a._check_dependencies(d)


def test_no_interaction_selection_is_preserved_as_disabled():
    d=snapshot();d['Simulations'][0]['Interactions']=[{'MoleculeName':'Enzyme human'}];before=deepcopy(d)
    r=a._check_dependencies(d)
    assert r['accepted'] and d==before
    assert r['interaction_count']==1
    assert r['disabled_interaction_selections']==[{'MoleculeName':'Enzyme human'}]


@pytest.mark.parametrize('item',[
 {'MoleculeName':'absent'}, {'Name':'active','MoleculeName':'Enzyme human'},
 {'MoleculeName':'Enzyme human','CompoundName':'unselected'},
 {'MoleculeName':'Enzyme human','Parameters':[{'Name':'enable','Value':1}]}, {},
 {'Name':'Enzyme reference-author-data','MoleculeName':'Enzyme human','CompoundName':'unselected'}])
def test_no_interaction_does_not_hide_invalid_active_selection(item):
    d=snapshot();d['Simulations'][0]['Interactions']=[item]
    with pytest.raises(a.SourceAdaptationError):a._check_dependencies(d)


def test_explicit_metabolite_remapping_uses_selected_existing_compound():
    d=snapshot();d['Compounds'].append({'Name':'Author metabolite_FM','IsSmallMolecule':True,'Processes':[]})
    selection=d['Simulations'][0]['Compounds'][0]['Processes'][0];selection['MetaboliteName']='Author metabolite_FM'
    d['Simulations'][0]['Compounds'].append({'Name':'Author metabolite_FM'})
    before=deepcopy(d);r=a._check_dependencies(d)
    assert d==before and r['accepted']
    assert r['explicit_metabolite_links']==[{'parent':'Parent','metabolite':'Author metabolite_FM','process':'Enzyme reference-author-data'}]
    assert r['retained_undosed_metabolites']==['Author metabolite_FM']


@pytest.mark.parametrize('fault',['unselected','undeclared','unreachable','unproven_enzymatic_default'])
def test_metabolite_mapping_must_remain_explicit_and_reachable(fault):
    d=snapshot();d['Compounds'].append({'Name':'M','IsSmallMolecule':True,'Processes':[]})
    d['Simulations'][0]['Compounds'][0]['Processes'][0]['MetaboliteName']='M'
    d['Simulations'][0]['Compounds'].append({'Name':'M'})
    if fault=='unselected':d['Simulations'][0]['Compounds'].pop()
    elif fault=='undeclared':d['Compounds'].pop()
    elif fault=='unreachable':d['Simulations'][0]['Compounds'][0]['Processes'][0].pop('MetaboliteName')
    else:d['Compounds'][0]['Processes'][0].pop('Metabolite')
    with pytest.raises(a.SourceAdaptationError):a._check_dependencies(d)
