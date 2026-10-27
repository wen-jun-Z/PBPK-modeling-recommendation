"""Design-only selection and complete original-model execution boundaries."""
from copy import deepcopy
import json
import hashlib
import sqlite3
from pathlib import Path

import pytest

from scripts.recommendation_execution_repairs import complete_candidate as m


def metadata(*, count=2, compound='Drug', coadmin=False):
    events, associations = [], []
    for i in range(count):
        prefix = f'Applications|Oral 100 mg|tablet|Application_{i+1}'
        for field, value, dim in [('Dose', '0.0001', 'Mass'), ('Start time', str(i*120), 'Time')]:
            events.append({'path': prefix+'|ProtocolSchemaItem|'+field, 'id': f'{i}_{field}', 'value': value, 'dimension': dim})
        associations.append({'event_prefix': prefix, 'application_type': 'Oral',
                             'administered_compounds': [compound, 'Other'] if coadmin else [compound],
                             'compound_association_complete': True, 'assignment_targets': ['Organism|Drug']})
    return {'simulation_id': 'sim', 'name': 'Drug fasted reference', 'saved_type': 'IndividualSimulation',
            'species': ['Human'], 'compounds': [compound, 'Drug metabolite'],
            'model_configuration': '4Comp|Human', 'formulations': [{'name': 'tablet', 'type': 'tablet'}],
            'population_source': [{'name': 'original individual'}], 'event_parameters': events, 'administrations': associations,
            'source_blob_sha256': 'b'*64}


def contract():
    return {'compound': 'drug', 'route': 'Oral', 'dose_mg': 100., 'administration_times_min': [0.],
            'formulation': 'tablet', 'food': 'fasted', 'infusion_minutes': None,
            'absolute_sampling_times_min': [0., 30., 120.], 'observation_origin_min': 0.}


def inventory(source):
    parameters = []
    for row in source['event_parameters']:
        parameters.append({'path': row['path'].replace('Applications|', 'Events|', 1), 'id': row['id'],
                           'value': float(row['value']), 'unit': 'kg' if row['dimension']=='Mass' else 'min'})
    parameters.append({'path': 'Drug|Molecular weight', 'value': 2e-7, 'unit': 'kg/µmol', 'id': 'mw'})
    return {'parameters': parameters, 'observer_paths': ['Organism|PeripheralVenousBlood|Drug|Plasma (Peripheral Venous Blood)']}


def suppression_policy(source, desired):
    starts = sorted([row for row in source['event_parameters'] if row['path'].endswith('|ProtocolSchemaItem|Start time')],
                    key=lambda row: (float(row['value']), row['path']))
    prefixes = [row['path'].replace('Applications|', 'Events|', 1).rsplit('|ProtocolSchemaItem|', 1)[0] for row in starts]
    end = desired['absolute_sampling_times_min'][-1]
    count = len(desired['administration_times_min'])
    return {'verified': True, 'ordered_source_event_prefixes': prefixes, 'target_event_count': count,
            'output_end_min': end, 'events': [
                {'event_prefix': prefixes[index], 'start_parameter_id': starts[index]['id'],
                 'start_parameter_path': starts[index]['path'].replace('Applications|', 'Events|', 1),
                 'new_start_time_min': end + 60. + index - count, 'dose_mg': 0.,
                 'barrier': {'verified': True, 'all_application_events_accounted': True,
                             'application_path': prefixes[index]}}
                for index in range(count, len(starts))]}


def test_metabolites_can_be_preserved_but_extra_administered_drug_rejected():
    good, _ = m.select_source_simulation({'simulations': [metadata()]}, contract())
    assert len(good) == 1
    bad, reasons = m.select_source_simulation({'simulations': [metadata(coadmin=True)]}, contract())
    assert not bad
    assert 'COADMINISTERED' in reasons[0]['reason']


@pytest.mark.parametrize('change', ['route', 'species', 'missing_association', 'too_few_events', 'unknown_target'])
def test_no_identity_route_or_event_shortcuts(change):
    source, desired = metadata(count=1), contract()
    if change == 'route': source['administrations'][0]['application_type'] = 'Intravenous'
    if change == 'species': source['species'] = ['Rat']
    if change == 'missing_association': source['administrations'][0]['compound_association_complete'] = False
    if change == 'too_few_events': desired['administration_times_min'] = [0., 60.]
    if change == 'unknown_target': source['compounds'] = ['Different']
    options, reasons = m.select_source_simulation({'simulations': [source]}, desired)
    assert not options and reasons


def test_every_surplus_event_is_zeroed_and_moved_beyond_output_without_new_events():
    source = metadata(count=3)
    patches, events = m.build_patches(inventory(source), contract(), source,
                                    suppression_policy=suppression_policy(source, contract()))
    doses = [p for p in patches if p['path'].endswith('|Dose')]
    assert [p['value'] for p in doses] == [100., 0., 0.]
    assert len(events) == 3
    assert events[2]['disposition'] == 'surplus_zero_dose_event_moved_beyond_output_window'
    assert [p['value'] for p in patches if p['path'].endswith('|Start time')] == [0., 180., 181.]
    assert all(row['start_time_min'] > 120 for row in events[1:])
    assert all(p['expected_before_unit']=='kg' for p in doses)


@pytest.mark.parametrize('mutation', ['missing_proof', 'in_window', 'wrong_parameter', 'wrong_window'])
def test_surplus_events_require_frozen_original_semantics_and_strict_future_time(mutation):
    source = metadata()
    proof = suppression_policy(source, contract())
    if mutation == 'missing_proof': proof = None
    elif mutation == 'in_window': proof['events'][0]['new_start_time_min'] = 120.
    elif mutation == 'wrong_parameter': proof['events'][0]['start_parameter_id'] = 'different'
    else: proof['output_end_min'] = 119.
    with pytest.raises(ValueError, match='SURPLUS'):
        m.build_patches(inventory(source), contract(), source, suppression_policy=proof)


def test_surplus_native_readback_requires_zero_mass_and_exact_future_trigger():
    source = metadata(); proof = suppression_policy(source, contract()); event = proof['events'][0]
    native = {'parameter_changes': [
        {'path': event['start_parameter_path'], 'id': event['start_parameter_id'],
         'after': {'value': 180., 'unit': 'min'}},
        {'path': event['event_prefix']+'|ProtocolSchemaItem|Dose', 'id': 'dose',
         'after': {'value': 0., 'unit': 'kg'}}]}
    assert m.suppression_readback(native, proof)['accepted']
    native['parameter_changes'][0]['after']['value'] = 120.
    with pytest.raises(ValueError, match='NOT_BEYOND'): m.suppression_readback(native, proof)
    native['parameter_changes'][0]['after']['value'] = 180.
    native['parameter_changes'][1]['after']['value'] = .0001
    with pytest.raises(ValueError, match='READBACK'): m.suppression_readback(native, proof)


def test_preserved_native_applications_alias_keeps_original_parameter_paths():
    source=metadata(); actual=inventory(source)
    for row in actual['parameters']:
        row['path']=row['path'].replace('Events|','Applications|',1)
    patches,events=m.build_patches(actual,contract(),source,suppression_policy=suppression_policy(source,contract()))
    assert all(p['path'].startswith('Applications|') for p in patches)
    assert events[-1]['start_time_min']==180.


def test_repeated_dose_origin_preserves_all_history_and_grid_contains_sampling():
    desired = contract()
    desired.update(administration_times_min=[0., 120.], observation_origin_min=120., absolute_sampling_times_min=[120., 150., 240.])
    patches, _ = m.build_patches(inventory(metadata(count=2)), desired, metadata(count=2))
    assert [p['value'] for p in patches if p['path'].endswith('|Start time')] == [0., 120.]
    times = m.output_grid(desired)
    assert times[0] == 0. and times[-1] == 240.
    assert {120.,150.,240.}.issubset(times)


@pytest.mark.parametrize('change', ['missing_event', 'id_changed', 'dose_unit', 'extra_event'])
def test_native_events_must_match_saved_model_independently(change):
    source = metadata(); actual = inventory(source)
    if change == 'missing_event': actual['parameters'].pop(0)
    if change == 'id_changed': actual['parameters'][0]['id']='wrong'
    if change == 'dose_unit': actual['parameters'][0]['unit']='mg/kg'
    if change == 'extra_event': actual['parameters'].append({**actual['parameters'][0], 'path':actual['parameters'][0]['path'].replace('Application_1','Application_9')})
    with pytest.raises(ValueError): m.build_patches(actual,contract(),source)


def test_molar_mass_conversion_requires_exact_parent_and_units():
    source=metadata(); actual=inventory(source); plan={'native_compound_name':'Drug'}
    _,mw=m._native_output(actual,plan)
    assert mw==pytest.approx(200.)
    actual['parameters'][-1]['path']='Drug metabolite|Molecular weight'
    with pytest.raises(ValueError):m._native_output(actual,plan)


@pytest.mark.parametrize('unit',['µmol/l','μmol/l','umol/l'])
def test_exact_micro_unit_alias_keeps_every_numeric_value_and_raw_curve(unit):
    raw={'unit':unit,'dimension':'Concentration (molar)','time_unit':'min','time_min':[0.,1.],
         'concentration':[.01,.002]};saved=deepcopy(raw)
    out=m.normalized_molar_curve(raw)
    assert out['unit']=='umol/l' and out['concentration']==raw['concentration']
    assert raw==saved and out is not raw
    if unit!='umol/l':assert out['original_native_unit']==unit


@pytest.mark.parametrize('unit,dimension',[('µmol/l',None),('µmol/l','Concentration (mass)'),
    ('μg/l','Concentration (mass)'),('mmol/l','Concentration (molar)'),('umol/l','Mass')])
def test_micro_unit_alias_never_masks_dimension_or_scale_change(unit,dimension):
    with pytest.raises(ValueError):m.normalized_molar_curve({'unit':unit,'dimension':dimension})


def test_standard_meal_start_is_preserved_and_never_patched_as_dose():
    source=metadata();meal={'path':'Events|Standard Meal|Standard Meal_1|Start time',
                           'id':'meal','value':'120','dimension':'Time'}
    source['event_parameters'].append(meal)
    actual=inventory(source)
    options,reasons=m.select_source_simulation({'simulations':[source]},contract())
    assert len(options)==1 and not reasons
    patches,events=m.build_patches(actual,contract(),source, suppression_policy=suppression_policy(source,contract()))
    assert all('Standard Meal' not in p['path'] for p in patches) and len(events)==2
    assert m._preserved_meal_parameters(source)==[meal]
    meal['path']='Events|Unregistered infusion|Infusion_1|Start time'
    with pytest.raises(ValueError,match='UNREGISTERED'):m._metadata_events(source)


def test_meal_name_cannot_hide_an_administered_drug_or_wrong_dimension():
    source=metadata();meal={'path':'Events|Standard Meal|Standard Meal_1|Start time','dimension':'Mass'}
    source['event_parameters'].append(meal)
    with pytest.raises(ValueError,match='MEAL_TIME_DIMENSION'):m._metadata_events(source)
    meal['dimension']='Time';source['administrations'].append({'event_prefix':'Events|Standard Meal|Standard Meal_1'})
    with pytest.raises(ValueError,match='HIDE_DRUG'):m._metadata_events(source)


def original_output_fixture(tmp_path, *, dimension='Molecular weight', value='2e-7', observer_drug='Drug'):
    xml=f'''<IndividualSimulation id="sim"><Model><Root id="r" name="root"><Children>
    <Container id="drug" name="Drug"><Children><Para id="mw" name="Molecular weight" dim="{dimension}" value="{value}" displayUnit="g/mol" /></Children></Container>
    <Container id="org" name="Organism"><Children><Container id="blood" name="PeripheralVenousBlood"><Children>
    <Container id="d" name="{observer_drug}"><Children><Observer id="o" name="Plasma (Peripheral Venous Blood)" /></Children></Container>
    </Children></Container></Children></Container></Children></Root><FormulaCache><Formulas /></FormulaCache></Model></IndividualSimulation>'''.encode()
    blob=m.source_pkml.compress_model_blob(xml)
    path=tmp_path/'original.pksim5'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE SIMULATIONS (SimulationId TEXT)')
        db.execute('CREATE TABLE BUILDING_BLOCKS (Id TEXT,ContentId TEXT)')
        db.execute('CREATE TABLE CONTENTS (Id TEXT,Data BLOB)')
        db.execute('INSERT INTO SIMULATIONS VALUES (?)',('sim',))
        db.execute('INSERT INTO BUILDING_BLOCKS VALUES (?,?)',('sim','content'))
        db.execute('INSERT INTO CONTENTS VALUES (?,?)',('content',blob))
    return path,hashlib.sha256(blob).hexdigest()


def test_original_base_units_and_exact_parent_observer_are_frozen(tmp_path):
    path,blob_sha=original_output_fixture(tmp_path)
    out=m.source_output_contract(path,m.sha(path),'sim',blob_sha,'Drug',project_root=tmp_path)
    # Display g/mol does not turn the stored native base value into 2e-7 g/mol.
    assert out['molecular_weight_g_per_mol']==pytest.approx(200.)
    assert out['output_path']=='Organism|PeripheralVenousBlood|Drug|Plasma (Peripheral Venous Blood)'


@pytest.mark.parametrize('kwargs',[{'dimension':'Mass'},{'value':'NaN'},{'observer_drug':'Drug metabolite'}])
def test_unresolved_original_units_nonfinite_mw_or_metabolite_output_blocks(tmp_path,kwargs):
    path,blob_sha=original_output_fixture(tmp_path,**kwargs)
    with pytest.raises(ValueError):m.source_output_contract(path,m.sha(path),'sim',blob_sha,'Drug',project_root=tmp_path)


def test_original_output_contract_requires_exact_model_blob(tmp_path):
    path,_=original_output_fixture(tmp_path)
    with pytest.raises(ValueError,match='EXACT_SELECTED'):
        m.source_output_contract(path,m.sha(path),'sim','0'*64,'Drug',project_root=tmp_path)


def test_explicit_human_model_configuration_can_supply_missing_population_origin():
    source=metadata();source.update(species=[],saved_type='PopulationSimulation')
    options,_=m.select_source_simulation({'simulations':[source]},contract())
    assert len(options)==1


def test_ir_brand_prefers_solution_over_xr_tablet_and_never_infers_ir_from_strength():
    desired=contract();desired.update(compound='quetiapine',formulation='Two25mg Seroquel film-coated reference tablets',
                                     food='Not explicitly stated in this report; do not infer fasting.')
    solution=metadata(compound='Quetiapine');solution.update(name='25 mg solution fed reference',simulation_id='solution',
                                                          formulations=[{'name':'Solution'}])
    xr=metadata(compound='Quetiapine');xr.update(name='Li 2014',simulation_id='xr',formulations=[{'name':'XR tablet'}])
    options,reasons=m.select_source_simulation({'simulations':[xr,solution]},desired)
    assert not reasons and options[0]['metadata']['simulation_id']=='solution'
    assert options[0]['target_release']=='immediate_release'
    assert options[0]['target_food_state']=='unknown'
    assert m._release('25 mg film-coated tablet','quetiapine')=='unknown'
    assert m._release('Seroquel XR 50 mg','quetiapine')=='modified_release'


def test_true_pregnancy_model_cannot_be_replaced_by_healthy_same_project_control():
    pregnant=metadata(compound='Caffeine_template');pregnant.update(simulation_id='pregnant',species=[],model_configuration=None)
    healthy=metadata(compound='Caffeine_template');healthy.update(simulation_id='healthy')
    desired=contract();desired['compound']='caffeine'
    policy={'pregnancy_candidate':True,'identity_aliases':{'caffeine':{'native_name':'Caffeine_template'}},
            'simulations':{'pregnant':{'pregnancy_structure_preserved':True,'human_pregnant_origin':True},
                           'healthy':{'pregnancy_structure_preserved':False,'human_pregnant_origin':False}}}
    options,reasons=m.select_source_simulation({'simulations':[healthy,pregnant]},desired,source_policy=policy)
    assert len(options)==1 and options[0]['metadata']['simulation_id']=='pregnant'
    assert reasons[0]['reason']=='PREGNANCY_CANDIDATE_CANNOT_USE_NONPREGNANT_CONTROL'
    _,events=m.build_patches(inventory(pregnant),desired,pregnant,native_compound_name='Caffeine_template',
                            suppression_policy=suppression_policy(pregnant,desired))
    assert len(events)==2
    with pytest.raises(ValueError,match='COADMINISTERED'):
        m.build_patches(inventory(pregnant),desired,pregnant)
    policy['simulations']['pregnant']['human_pregnant_origin']=False
    assert not m.select_source_simulation({'simulations':[pregnant]},desired,source_policy=policy)[0]
    assert not m.select_source_simulation({'simulations':[pregnant]},desired)[0]


@pytest.fixture
def pregnancy_source(tmp_path,monkeypatch):
    path,_=original_output_fixture(tmp_path)
    with sqlite3.connect(path) as db:
        blob=db.execute('SELECT Data FROM CONTENTS').fetchone()[0]
        xml=m.source_pkml.decompress_model_blob(blob).decode()
        xml=xml.replace('</IndividualSimulation>', '<UsedBuildingBlocks><RandomPopulation><OriginData species="Human" speciesPopulation="Pregnant" gender="FEMALE" /></RandomPopulation></UsedBuildingBlocks></IndividualSimulation>')
        xml=xml.replace('<Container id="blood"', ''.join(f'<Container id="{n}" name="{n}" />' for n in sorted(m.PREGNANCY_MARKERS))+'<Container id="blood"')
        blob=m.source_pkml.compress_model_blob(xml.encode());db.execute('UPDATE CONTENTS SET Data=?',(blob,))
    ref=m.reference(path,tmp_path);blob_sha=hashlib.sha256(blob).hexdigest()
    proof={'source':ref,'simulation':{'simulation_id':'sim'},'source_blob_sha256':blob_sha}
    m.save(tmp_path/'proof.json',proof)
    row={'candidate_id':'preg','source':ref,'repository_commit':'c'*40,'simulation':{'simulation_id':'sim'},
         'pregnancy_markers':sorted(m.PREGNANCY_MARKERS),'proof':m.reference(tmp_path/'proof.json',tmp_path)}
    m.save(tmp_path/'registry.json',{'records':[row]});monkeypatch.setattr(m,'PREGNANCY_REGISTRY','registry.json')
    candidate={'candidate_id':'preg','provenance':{'source_sha256':ref['sha256'],'repository_commit':'c'*40}}
    return tmp_path,path,candidate,{'simulations':[{'simulation_id':'sim','source_blob_sha256':blob_sha}]}


def test_pregnancy_qualification_reads_human_and_structure_from_original_source(pregnancy_source):
    root,path,candidate,meta=pregnancy_source
    policy=m.pregnancy_source_policy(candidate,path,meta,project_root=root)
    assert policy['pregnancy_candidate'] is True
    proof=policy['simulations']['sim']
    assert proof['human_pregnant_origin'] and proof['pregnancy_structure_preserved']
    assert proof['original_origin_data'][0]['gender']=='FEMALE'


@pytest.mark.parametrize('change',['source','commit','blob','proof'])
def test_pregnancy_qualification_rejects_unbound_registry(pregnancy_source,change):
    root,path,candidate,meta=pregnancy_source
    if change=='source':candidate['provenance']['source_sha256']='0'*64
    if change=='commit':candidate['provenance']['repository_commit']='0'*40
    if change=='blob':meta['simulations'][0]['source_blob_sha256']='0'*64
    if change=='proof':(root/'proof.json').write_text('{}')
    with pytest.raises(ValueError):m.pregnancy_source_policy(candidate,path,meta,project_root=root)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source=tmp_path/'source.pksim5';source.write_bytes(b'original model fixture')
    scenario={'scenario_id':'scenario','case_id':'case','compound':'drug',
              'normalized_design': {'dose':{'value':100.,'unit':'mg'}, 'route':'oral',
              'regimen':{'dose_times_h':[0.], 'observation_origin_h':0.},'sampling':{'times_h':[0.,.5,2.]},
              'matrix':'plasma','analyte':'parent','formulation':'tablet','food':'fasted',
              'prediction_contract':{'source_reference_subject_prediction_allowed':True}}}
    candidate={'candidate_id':'candidate','compounds':['Drug'],
               'provenance':{'source_path':'source.pksim5','source_sha256':m.sha(source),'license_status':'allowed'}}
    code={'scripts/recommendation_execution_repairs/source_pkml.py':'code'}
    monkeypatch.setattr(m,'implementation_hashes',lambda project_root=m.ROOT:code.copy())
    source_meta=metadata()
    monkeypatch.setattr(m.source_pkml,'source_simulation_metadata',lambda *a,**kw:{'simulations':[deepcopy(source_meta)]})
    monkeypatch.setattr(m,'source_output_contract',lambda *a,**kw:{
        'output_path':'Organism|PeripheralVenousBlood|Drug|Plasma (Peripheral Venous Blood)',
        'molecular_weight_g_per_mol':200.})
    monkeypatch.setattr(m,'source_suppression_policy',lambda source,sha,sid,blob,meta,desired,**kw:
                        suppression_policy(meta,desired))
    m._METADATA_CACHE.clear()
    return tmp_path,scenario,candidate,source_meta


def test_prepare_is_deterministic_without_export_inspection_or_solver(prepared,monkeypatch):
    root,scenario,candidate,_=prepared
    def forbidden(*args,**kwargs):raise AssertionError('Preparation attempted native execution')
    monkeypatch.setattr(m.source_pkml,'export_saved_model',forbidden)
    monkeypatch.setattr(m.native_pkml,'execute_complete_pkml',forbidden)
    one=m.prepare_candidate(scenario,candidate,project_root=root)
    two=m.prepare_candidate(scenario,candidate,project_root=root)
    assert one==two and one['status']=='ready'
    assert one['surplus_source_doses_to_zero']==1
    assert one['compatibility']['strict_target_match'] is False
    assert one['molecular_weight_g_per_mol']==200.


def test_json_companion_is_not_silently_substituted(prepared):
    root,scenario,candidate,_=prepared
    p=root/'snapshot.json';p.write_text('{}')
    candidate['provenance'].update(source_path='snapshot.json',source_sha256=m.sha(p))
    answer=m.prepare_candidate(scenario,candidate,project_root=root)
    assert answer['status']=='blocked'
    assert 'NO_JSON_COMPANION_SUBSTITUTION' in answer['reasons'][0]


def test_different_frozen_plan_is_rejected_before_creating_attempt(prepared):
    root,scenario,candidate,_=prepared
    plan=m.prepare_candidate(scenario,candidate,project_root=root)
    plan['source_simulation_id']='other'
    with pytest.raises(ValueError,match='FROZEN_COMPLETE'):
        m.execute_candidate(scenario,candidate,root/'out',project_root=root,frozen_plan=plan)
    assert not (root/'out').exists()


def test_immediate_frozen_execution_reuses_only_its_verified_preparation(prepared,monkeypatch):
    root,scenario,candidate,_=prepared
    license_file=root/'license.json';license_file.write_text('{}')
    plan=m.prepare_candidate(scenario,candidate,project_root=root,license_receipt=license_file)
    def repeated(*a,**kw):raise AssertionError('unexpected duplicate preparation')
    def exported(*a,**kw):raise AssertionError('verified preparation reached export')
    monkeypatch.setattr(m,'prepare_candidate',repeated)
    monkeypatch.setattr(m.source_pkml,'export_saved_model',exported)
    with pytest.raises(AssertionError,match='reached export'):
        m.execute_candidate(scenario,candidate,root/'out',project_root=root,license_receipt=license_file,frozen_plan=plan)


@pytest.mark.parametrize('changed',['source','license','scenario','code'])
def test_preparation_cache_never_hides_dependency_changes(prepared,monkeypatch,changed):
    root,scenario,candidate,_=prepared
    license_file=root/'license.json';license_file.write_text('{}')
    plan=m.prepare_candidate(scenario,candidate,project_root=root,license_receipt=license_file)
    if changed=='source':(root/'source.pksim5').write_bytes(b'changed')
    elif changed=='license':license_file.write_text('{"changed":true}')
    elif changed=='scenario':scenario['normalized_design']['dose']['value']=200.
    else:monkeypatch.setattr(m,'implementation_hashes',lambda *a,**kw:{'changed':'code'})
    def repeat(*a,**kw):raise AssertionError('dependency change forces fresh preparation')
    monkeypatch.setattr(m,'prepare_candidate',repeat)
    with pytest.raises(AssertionError,match='forces fresh preparation'):
        m.execute_candidate(scenario,candidate,root/'out',project_root=root,license_receipt=license_file,frozen_plan=plan)


def test_license_unknown_is_not_made_allowed_by_public_access(prepared,monkeypatch):
    root,scenario,candidate,_=prepared
    candidate['provenance']['license_status']='unresolved'
    monkeypatch.setattr(m,'audit_candidate_license',lambda *a,**kw:{'status':'unknown','reason':'no grant'})
    plan=m.prepare_candidate(scenario,candidate,project_root=root)
    assert plan['status']=='blocked' and 'CANDIDATE_LICENSE_NOT_RESOLVED' in plan['reasons'][0]


def test_execution_preserves_frozen_plan_and_separate_native_receipt(prepared,monkeypatch):
    root,scenario,candidate,source_meta=prepared
    plan=m.prepare_candidate(scenario,candidate,project_root=root)
    def export(source,source_sha,sid,out,**kw):
        out.mkdir(parents=True);pkml=out/'original.pkml';pkml.write_text('complete native fixture')
        result={'status':'exported_pending_native_load_and_clinical_validation','source_metadata':plan['source_metadata'],
                'source_blob_sha256':source_meta['source_blob_sha256'],'pkml':m.reference(pkml,root)}
        m.save(out/'export_manifest.json',result);return result
    def native(pkml,out,*,action='inspect',**kw):
        out.mkdir(parents=True)
        if action=='inspect':result={'status':'inspected','native':{'inventory':inventory(source_meta)}}
        else:
            parameters={row['path']:row for row in inventory(source_meta)['parameters']}
            changes=[]
            for patch in kw['patches']:
                before=parameters[patch['path']]
                value=patch['value']*1e-6 if patch['unit']=='mg' else patch['value']
                changes.append({'path':patch['path'],'id':before['id'],
                                'after':{'value':value,'unit':before['unit']}})
            result={'status':'prediction_locked','model_preservation_audit':{'accepted':True},
                    'native':{'solver_calls':1,'parameter_changes':changes,'output_grid_verified':True,
                              'curve':{'time_min':kw['output_times_min'],'concentration':[1.]*len(kw['output_times_min']),
                                       'time_unit':'min','unit':'µmol/l','dimension':'Concentration (molar)'}}}
        m.save(out/'result.json',result);return result
    monkeypatch.setattr(m.source_pkml,'export_saved_model',export)
    monkeypatch.setattr(m.native_pkml,'execute_complete_pkml',native)
    result=m.execute_candidate(scenario,candidate,root/'out',project_root=root,frozen_plan=plan)
    assert result['status']=='prediction_locked',result
    saved=json.loads((root/'out/execution_plan.json').read_text())
    assert saved==plan
    prediction=json.loads((root/'out/prediction.json').read_text())
    assert prediction['concentration_ng_ml'][0]==pytest.approx(200.)
    assert prediction['native_curve']['unit']=='umol/l' and result['curve']['unit']=='µmol/l'
    assert result['strict_target_match'] is False
    assert (root/'out/registered_native_request.json').exists()
    lock=json.loads((root/'out/prediction_lock.json').read_text())
    assert 'out/export/export_manifest.json' in lock['files']
    assert 'out/execution_receipt.json' in lock['files']


def expansion_policy(source, desired, source_ref):
    return {'version': m.repeated_application.VERSION, 'verified': True,
            'source_reference': source_ref, 'compound_name': 'Drug',
            'event_prefix': source['administrations'][0]['event_prefix'],
            'original_source_event_count': 1, 'executable_event_count': len(desired['administration_times_min']),
            'administration_times_min': desired['administration_times_min'], 'dose_mg': desired['dose_mg']}


def test_short_source_requires_bound_structural_policy_not_a_count_override():
    source, desired = metadata(count=1), contract()
    desired['administration_times_min'] = [0.,1440.]
    policy = expansion_policy(source, desired, {'simulation_id':'sim','source_blob_sha256':source['source_blob_sha256']})
    context = {'pregnancy_candidate':False,'simulations':{},'identity_aliases':{},'application_expansion_policies':{'sim':policy}}
    good,_ = m.select_source_simulation({'simulations':[source]}, desired, source_policy=context)
    assert good[0]['source_event_count'] == 1
    for field, value in [('verified',False), ('dose_mg',1.), ('executable_event_count',3), ('administration_times_min',[0.,60.])]:
        bad = deepcopy(context); bad['application_expansion_policies']['sim'][field] = value
        assert not m.select_source_simulation({'simulations':[source]},desired,source_policy=bad)[0]


def test_prepare_preserves_one_source_event_and_freezes_separate_seven_event_recipe(prepared,monkeypatch):
    root, scenario, candidate, _ = prepared
    source = metadata(count=1)
    monkeypatch.setattr(m.source_pkml,'source_simulation_metadata',lambda *a,**kw:{'simulations':[deepcopy(source)]})
    scenario['normalized_design']['regimen']={'dose_times_h':[24.*i for i in range(7)],'observation_origin_h':144.}
    calls=[]
    def qualify(ref, original, **kw):
        calls.append((ref,deepcopy(original)))
        desired={'administration_times_min':kw['administration_times_min'],'dose_mg':kw['dose_mg']}
        return expansion_policy(source,desired,ref)
    monkeypatch.setattr(m.repeated_application,'prepare_source_expansion',qualify)
    def forbidden(*a,**kw):raise AssertionError('Plan preparation invoked native')
    monkeypatch.setattr(m.source_pkml,'export_saved_model',forbidden)
    monkeypatch.setattr(m.native_pkml,'execute_complete_pkml',forbidden)
    m._METADATA_CACHE.clear()
    plan=m.prepare_candidate(scenario,candidate,project_root=root)
    assert plan['status']=='ready',plan
    assert plan['source_event_count']==1 and plan['executable_event_count']==7
    assert len(plan['source_metadata']['administrations'])==1
    assert plan['source_metadata']==m._original_metadata(source)
    assert plan['source_metadata_sha256']==m._digest(m._original_metadata(source))
    assert plan['surplus_source_doses_to_zero']==0 and plan['surplus_event_suppression'] is None
    assert plan['application_expansion']['source_reference']['source_blob_sha256']==source['source_blob_sha256']
    assert len(calls)==1 and calls[0][1]==m._original_metadata(source)


def test_failed_source_structure_qualification_stays_blocked(prepared,monkeypatch):
    root,scenario,candidate,_=prepared
    source=metadata(count=1)
    monkeypatch.setattr(m.source_pkml,'source_simulation_metadata',lambda *a,**kw:{'simulations':[deepcopy(source)]})
    scenario['normalized_design']['regimen']={'dose_times_h':[0.,24.],'observation_origin_h':24.}
    def reject(*a,**kw):raise ValueError('complete tablet transport structure missing')
    monkeypatch.setattr(m.repeated_application,'prepare_source_expansion',reject)
    m._METADATA_CACHE.clear()
    plan=m.prepare_candidate(scenario,candidate,project_root=root)
    assert plan['status']=='blocked'
    assert 'transport structure' in plan['application_expansion_assessment_failures'][0]['reason']
    assert plan['scientific_inapplicability_inferred'] is False


def test_event_association_proof_never_rewrites_original_source_metadata(prepared,monkeypatch):
    root,scenario,candidate,_=prepared
    source=metadata();source['administrations'][0].update(compound_association_complete=False,administered_compounds=[])
    monkeypatch.setattr(m.source_pkml,'source_simulation_metadata',lambda *a,**kw:{'simulations':[deepcopy(source)]})
    def enrich(path, sha, sid, blob, metadata=None,**kw):
        assert metadata==m._original_metadata(source)
        enriched=deepcopy(metadata);enriched['administrations'][0].update(compound_association_complete=True,administered_compounds=['Drug'])
        enriched['source_blob_sha256']=blob
        return {'enriched_metadata':enriched,'audit':{'original_metadata_sha256':m._digest(metadata),'source_model_blob':blob}}
    monkeypatch.setattr(m.event_associations,'enrich_source_metadata',enrich)
    m._METADATA_CACHE.clear()
    plan=m.prepare_candidate(scenario,candidate,project_root=root)
    assert plan['status']=='ready',plan
    assert plan['source_metadata']['administrations'][0]['compound_association_complete'] is False
    assert plan['source_event_association']['enriched_metadata']['administrations'][0]['compound_association_complete'] is True
    assert plan['source_metadata_sha256']==m._digest(m._original_metadata(source))


def test_expansion_readback_uses_exact_expanded_entity_ids_and_registered_regimen():
    source=metadata(count=2);source.update(metadata_scope='actual_expanded_executable_model_only',original_source_event_count=1,executable_event_count=2)
    desired=contract();desired['administration_times_min']=[0.,120.]
    policy=expansion_policy(metadata(count=1),desired,{});actual=inventory(source)
    patches,_=m.build_patches(actual,desired,source)
    params={p['path']:p for p in actual['parameters']};changes=[]
    for p in patches:
        n=params[p['path']];changes.append({'path':p['path'],'id':n['id'],'after':{'value':p['value']/1e6 if p['unit']=='mg' else p['value'],'unit':n['unit']}})
    native={'parameter_changes':changes}
    assert m.expansion_readback(native,policy,source,{'accepted':True})['accepted']
    native['parameter_changes'][-1]['id']='old-source-id'
    with pytest.raises(ValueError,match='READBACK_DIFFERS'):
        m.expansion_readback(native,policy,source,{'accepted':True})
