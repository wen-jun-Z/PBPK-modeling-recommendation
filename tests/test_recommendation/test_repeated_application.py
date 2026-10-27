"""Independent complete-application expansion boundaries; no fitting or ODE."""
from copy import deepcopy
import hashlib
import json
import sqlite3
import xml.etree.ElementTree as ET

import pytest

from scripts.recommendation_execution_repairs import repeated_application as repair


PREFIX = 'Applications|Original regimen|Original tablet|Application_1'


def document():
    doc = ET.fromstring('''<SimulationTransfer><Simulation><SimulationConfiguration>
      <SimulationSettings><Solver relativeTolerance="0.000001"/><OutputSchema end="100"/></SimulationSettings>
      <Individual name="original" weight="70"/></SimulationConfiguration><Model><Root id="root" name="Original" mode="Logical">
      <Children><Container id="body" name="Organism"><Children><Parameter id="clearance" name="Clearance" value="2"/>
      </Children></Container><Container id="drug" name="Pioglitazone"><Children>
      <Parameter id="mw" name="Molecular weight" dim="Molecular weight" value="3.5644e-7"/>
      <Parameter id="oral_total" name="Total oral drug mass" dim="Amount" value="0"/>
      <Parameter id="aggregate" name="Total drug mass" dim="Amount" formula="sum"/>
      </Children></Container><EventGroup id="apps" name="Applications"><Children>
      <EventGroup id="regimen" name="Original regimen"><Children>
      <EventGroup id="form" name="Original tablet"><Children>
      <EventGroup id="app" name="Application_1" containerType="Application" eventGroupType="Oral"><Tags>
      <Tag value="Application"/><Tag value="Application_1"/></Tags><Children>
      <Parameter id="active" name="TabletIsActive" value="0"/>
      <Parameter id="kinetics" name="Dissolution rate" value="0.42"/>
      <Container id="schema" name="ProtocolSchemaItem"><Children>
      <Parameter id="dose" name="Dose" dim="Mass" value="3E-05" displayUnit="mg"/>
      <Parameter id="start" name="Start time" dim="Time" value="60" displayUnit="h"/>
      <Parameter id="mass" name="DrugMass" dim="Amount" formula="drugmass"/>
      </Children></Container><MoleculeAmount id="tablet" name="Pioglitazone" dim="Amount" formula="initial"/>
      <Event id="event_start" name="Application_StartEvent" oneTime="1" formula="begin"><Children>
      <EventAssignment id="assignment1" name="Assignment_1" changedEntity="active" formula="one" useAsValue="1"/>
      <EventAssignment id="assignment3" name="Assignment_2" changedEntity="oral_total" formula="adjust_total" useAsValue="1"/>
      </Children></Event><Event id="event_end" name="Application_StopEvent" oneTime="1" formula="end"><Children>
      <EventAssignment id="assignment2" name="Assignment_1" changedEntity="active" formula="zero" useAsValue="0"/>
      </Children></Event></Children></EventGroup></Children></EventGroup></Children></EventGroup></Children></EventGroup>
      </Children></Root><FormulaCache><Formulas>
      <Formula id="sum" formula="P_1" dim="Amount"><Paths><Path path="Original|Applications|Original regimen|Original tablet|Application_1|ProtocolSchemaItem|DrugMass" as="P_1" dim="Amount"/></Paths></Formula>
      <Formula id="drugmass" formula="Dose/MW"><Paths><Path path="..|Dose" as="Dose"/><Path path="Original|Pioglitazone|Molecular weight" as="MW"/></Paths></Formula>
      <Formula id="initial" formula="M"><Paths><Path path="..|ProtocolSchemaItem|DrugMass" as="M"/></Paths></Formula>
      <Formula id="adjust_total" formula="M+DrugMass"><Paths><Path path="Original|Pioglitazone|Total oral drug mass" as="M"/><Path path="..|..|ProtocolSchemaItem|DrugMass" as="DrugMass"/></Paths></Formula>
      <Formula id="release" formula="k*M"><Paths><Path path="..|Dissolution rate" as="k"/><Path path="..|Pioglitazone" as="M"/></Paths></Formula>
      <Formula id="begin" formula="Time=StartTime"><Paths><TimePath/><Path path="..|ProtocolSchemaItem|Start time" as="StartTime"/></Paths></Formula>
      <Formula id="end" formula="Active=1"><Paths><Path path="..|TabletIsActive" as="Active"/></Paths></Formula>
      <ConstantFormula id="one" value="1"/><ConstantFormula id="zero" value="0"/>
      </Formulas><StringMap/></FormulaCache></Model></Simulation></SimulationTransfer>''')
    model = doc.find('Simulation/Model')
    body = next(n for n in model.iter('Container') if n.get('id') == 'body').find('Children')
    app = next(n for n in model.iter('EventGroup') if n.get('id') == 'app').find('Children')
    for i in range(12):
        ET.SubElement(body, 'MoleculeAmount', id=f'body_{i}', name=f'Lumen_{i}', dim='Amount', value='0')
        ET.SubElement(app, 'Transport', id=f'release_{i}', name=f'TabletDrugRelease_{i}',
                      sourceAmount='tablet', targetAmount=f'body_{i}', formula='release')
    return doc


def expanded():
    source = document()
    result, plan = repair.expand_document(source, compound_name='Pioglitazone', event_prefix=PREFIX,
                      administration_times_min=[1440. * i for i in range(7)], dose_mg=30.)
    return source, result, plan


def test_all_seven_doses_have_independent_states_and_original_external_targets():
    source, result, plan = expanded()
    branches = [n for n in result.iter('EventGroup') if n.get('containerType') == 'Application']
    assert len(branches) == 7
    states = [next(n.iter('MoleculeAmount')).get('id') for n in branches]
    assert len(set(states)) == 7
    for i, branch in enumerate(branches):
        transports = list(branch.iter('Transport'))
        assert len(transports) == 12
        assert {x.get('sourceAmount') for x in transports} == {states[i]}
        assert {x.get('targetAmount') for x in transports} == {f'body_{j}' for j in range(12)}
        assert all(n.get('oneTime') == '1' for n in branch.iter('Event'))
    assert repair.audit_expansion(source, result, plan)['accepted']
    assert ET.tostring(source) == ET.tostring(document()), 'Pure expansion must not mutate original'


def test_count_one_without_a_regimen_change_is_exact_identity():
    source = document()
    result, plan = repair.expand_document(source, compound_name='Pioglitazone', event_prefix=PREFIX,
                            administration_times_min=[60.], dose_mg=30.)
    assert repair._canonical(source) == repair._canonical(result)
    assert repair.audit_expansion(source, result, plan)['accepted']


@pytest.mark.parametrize('target', ['Clearance', 'Dissolution rate', 'Molecular weight'])
def test_hidden_drug_physiology_or_release_changes_are_rejected(target):
    source, result, plan = expanded()
    matches = [x for x in result.iter('Parameter') if x.get('name') == target]
    matches[-1].set('value', '9')
    assert not repair.audit_expansion(source, result, plan)['accepted']


def test_full_settings_remain_protected():
    source, result, plan = expanded()
    result.find('.//Solver').set('relativeTolerance', '0.1')
    assert not repair.audit_expansion(source, result, plan)['accepted']


def test_cloned_transport_cannot_point_to_another_dose_state():
    source, result, plan = expanded()
    branches = [n for n in result.iter('EventGroup') if n.get('containerType') == 'Application']
    next(branches[2].iter('Transport')).set('sourceAmount', next(branches[1].iter('MoleculeAmount')).get('id'))
    assert not repair.audit_expansion(source, result, plan)['accepted']


def test_external_target_cannot_be_silently_changed_to_another_body_state():
    source, result, plan = expanded()
    list(result.iter('Transport'))[-1].set('targetAmount', 'body_0')
    assert not repair.audit_expansion(source, result, plan)['accepted']


def test_alias_and_sum_must_cover_each_registered_dose_exactly_once():
    source, result, plan = expanded()
    f = next(x for x in result.find('.//FormulaCache/Formulas') if x.get('id') == 'sum')
    f.find('Paths')[3].set('as', 'P_2')
    assert not repair.audit_expansion(source, result, plan)['accepted']


def test_new_native_paths_and_aliases_use_integer_string_table_ids():
    _, result, plan = expanded()
    cache = result.find('.//FormulaCache')
    mapping = {n.get('id'): n.get('s') for n in cache.find('StringMap')}
    f = next(n for n in cache.find('Formulas') if n.get('id') == 'sum')
    for i, path in enumerate(f.find('Paths'), 1):
        assert path.get('path').isdigit() and path.get('as').isdigit()
        assert mapping[path.get('as')] == f'P_{i}'
        assert mapping[path.get('path')].endswith(f'Application_{i}|ProtocolSchemaItem|DrugMass')
    assert len(plan['added_string_map_entries']) == 14


def test_bibliographic_integer_ids_are_preserved_and_never_become_entity_uuids():
    source = document()
    param = next(n for n in source.iter('Parameter') if n.get('name') == 'Dissolution rate')
    ET.SubElement(param, 'ValueOrigin', id='27', source='Publication')
    result, plan = repair.expand_document(source, compound_name='Pioglitazone', event_prefix=PREFIX,
                      administration_times_min=[0.,1440.], dose_mg=30.)
    assert [n.get('id') for n in result.iter('ValueOrigin')] == ['27','27']
    assert all('27' not in p['id_mapping'] for p in plan['applications'])
    assert repair.audit_expansion(source, result, plan)['accepted']


def test_an_existing_string_reference_cannot_be_rebound_to_hide_a_parameter_change():
    source, result, plan = expanded()
    mapping = result.find('.//FormulaCache/StringMap')
    mapping[-1].set('s', 'clearance-override')
    assert not repair.audit_expansion(source, result, plan)['accepted']


def test_nonstandard_start_event_cannot_be_misrepresented_as_periodic_dosing():
    source = document()
    next(n for n in source.iter('Formula') if n.get('id') == 'begin').set('formula', 'Time>=StartTime')
    with pytest.raises(ValueError, match='exact StartTime'):
        repair.expand_document(source, compound_name='Pioglitazone', event_prefix=PREFIX,
                               administration_times_min=[0.,1440.], dose_mg=30.)


@pytest.mark.parametrize('kind', ['dose', 'time', 'extra_event', 'duplicate_id', 'fixed_flag', 'one_time'])
def test_regimen_and_entity_identity_cannot_be_forged(kind):
    source, result, plan = expanded()
    branches = [n for n in result.iter('EventGroup') if n.get('containerType') == 'Application']
    if kind == 'dose':
        next(x for x in branches[-1].iter('Parameter') if x.get('name') == 'Dose').set('value', '4e-5')
    elif kind == 'time':
        next(x for x in branches[-1].iter('Parameter') if x.get('name') == 'Start time').set('value', '0')
    elif kind == 'extra_event':
        parent = next(n for n in result.iter() if branches[-1] in list(n))
        parent.append(deepcopy(branches[-1]))
    elif kind == 'duplicate_id':
        next(branches[-1].iter('MoleculeAmount')).set('id', 'tablet')
    elif kind == 'fixed_flag':
        next(x for x in branches[-1].iter('Parameter') if x.get('name') == 'Dose').set('isFixedValue', '1')
    else:
        next(branches[-1].iter('Event')).set('oneTime', '0')
    assert not repair.audit_expansion(source, result, plan)['accepted']


@pytest.mark.parametrize('times,dose', [([],30), ([0.,0.],30), ([0.,float('nan')],30), ([0.],-1),
                                      ([0.],float('inf')), (list(range(65)),30), ([-1.,2.],30)])
def test_no_inferred_unbounded_or_nonfinite_regimen(times, dose):
    with pytest.raises(ValueError):
        repair.expand_document(document(), compound_name='Pioglitazone', event_prefix=PREFIX,
                               administration_times_min=times, dose_mg=dose)


def test_another_drug_or_a_shared_aggregate_is_not_accepted():
    source = document()
    with pytest.raises(ValueError):
        repair.expand_document(source, compound_name='Other', event_prefix=PREFIX,
                               administration_times_min=[0.,1440.], dose_mg=30.)
    source = document()
    next(x for x in source.iter('Parameter') if x.get('name') == 'Clearance').set('formula', 'sum')
    with pytest.raises(ValueError, match='shared'):
        repair.expand_document(source, compound_name='Pioglitazone', event_prefix=PREFIX,
                               administration_times_min=[0.,1440.], dose_mg=30.)


def test_unregistered_inbound_application_dependency_requires_separate_review():
    source = document()
    f = next(n for n in source.iter('Formula') if n.get('id') == 'release')
    f.find('Paths')[0].set('path', 'Original|' + PREFIX + '|Dissolution rate')
    with pytest.raises(ValueError, match='absolute self'):
        repair.expand_document(source, compound_name='Pioglitazone', event_prefix=PREFIX,
                               administration_times_min=[0.,1440.], dose_mg=30.)


def test_malicious_branch_rename_in_receipt_is_rejected():
    source, result, plan = expanded()
    plan['applications'][-1]['prefix'] = plan['applications'][0]['prefix']
    assert not repair.audit_expansion(source, result, plan)['accepted']


def test_clinical_observation_path_is_outside_adaptation_contract(tmp_path):
    folder = tmp_path / 'sealed_clinical'; folder.mkdir()
    path = folder / 'model.pkml'; path.write_text('<ignored/>')
    with pytest.raises(ValueError, match='unsealed'):
        repair.expand_applications(path, tmp_path/'out', source_reference={}, compound_name='Pioglitazone',
                                  event_prefix=PREFIX, administration_times_min=[0.], dose_mg=30., project_root=tmp_path)


def source_project(tmp_path):
    from scripts.recommendation_execution_repairs.source_pkml import compress_model_blob
    source = document()
    saved = ET.Element('IndividualSimulation', id='original_sim', name='Original')
    saved.append(deepcopy(source.find('Simulation/Model')))
    saved.append(deepcopy(source.find('.//SimulationSettings')))
    blocks=ET.SubElement(saved,'UsedBuildingBlocks'); wrapper=ET.SubElement(blocks,'UsedBuildingBlock')
    ET.SubElement(wrapper,'Compound',id='compound',name='Pioglitazone')
    blob = compress_model_blob(ET.tostring(saved))
    database = tmp_path/'author.pksim5'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE SIMULATIONS(SimulationId TEXT)')
        db.execute('CREATE TABLE BUILDING_BLOCKS(Id TEXT,ContentId TEXT)')
        db.execute('CREATE TABLE CONTENTS(Id TEXT,Data BLOB)')
        db.execute('INSERT INTO SIMULATIONS VALUES(?)', ('original_sim',))
        db.execute('INSERT INTO BUILDING_BLOCKS VALUES(?,?)', ('original_sim','original_content'))
        db.execute('INSERT INTO CONTENTS VALUES(?,?)', ('original_content',blob))
    pkml = tmp_path/'original.pkml'
    ET.ElementTree(source).write(pkml)
    reference = {'path': 'author.pksim5', 'sha256': repair.sha(database), 'simulation_id': 'original_sim',
                 'source_blob_sha256': hashlib.sha256(blob).hexdigest()}
    return pkml, reference


def test_file_receipt_binds_immutable_source_simulation_code_and_saved_audit(tmp_path):
    pkml, reference = source_project(tmp_path)
    original_bytes = pkml.read_bytes()
    receipt = repair.expand_applications(pkml, tmp_path/'out', source_reference=reference,
                 compound_name='Pioglitazone', event_prefix=PREFIX,
                 administration_times_min=[0.,1440.], dose_mg=30., project_root=tmp_path)
    assert receipt['status'] == 'expanded_pending_native_readback_and_ode'
    assert not receipt['ode_executed']
    assert not receipt['scientific_applicability_claimed']
    assert receipt['source_reference'] == reference
    assert receipt['implementation_sha256']
    for path, expected in receipt['files'].items():
        assert repair.sha(tmp_path/path) == expected
    assert repair.sha(tmp_path/receipt['pkml']['path']) == receipt['pkml']['sha256']
    assert pkml.read_bytes() == original_bytes
    assert json.loads((tmp_path/'out/semantic_audit.json').read_text())['accepted']
    with pytest.raises(FileExistsError):
        repair.expand_applications(pkml, tmp_path/'out', source_reference=reference,
             compound_name='Pioglitazone', event_prefix=PREFIX,
             administration_times_min=[0.,1440.], dose_mg=30., project_root=tmp_path)


@pytest.mark.parametrize('field', ['sha256', 'simulation_id', 'source_blob_sha256', 'source_pkml_parameter'])
def test_wrong_source_chain_cannot_be_laundered_by_an_expansion(tmp_path, field):
    pkml, reference = source_project(tmp_path)
    if field == 'source_pkml_parameter':
        doc = ET.parse(pkml)
        next(n for n in doc.iter('Parameter') if n.get('name') == 'Clearance').set('value','3')
        doc.write(pkml)
    else:
        reference[field] = 'wrong'
    with pytest.raises(ValueError):
        repair.expand_applications(pkml, tmp_path/'out', source_reference=reference,
             compound_name='Pioglitazone', event_prefix=PREFIX,
             administration_times_min=[0.,1440.], dose_mg=30., project_root=tmp_path)
    assert not (tmp_path/'out').exists()


def test_pure_source_policy_freezes_original_structure_without_native_calls(tmp_path, monkeypatch):
    pkml,reference=source_project(tmp_path)
    source=repair._original_document(reference,tmp_path)
    metadata=repair.describe_saved_simulation(source)
    def forbidden(*a,**kw):raise AssertionError('Eligibility called native execution')
    from scripts.recommendation_execution_repairs import source_pkml,native_pkml
    monkeypatch.setattr(source_pkml,'export_saved_model',forbidden)
    monkeypatch.setattr(native_pkml,'execute_complete_pkml',forbidden)
    policy=repair.prepare_source_expansion(reference,metadata,compound_name='Pioglitazone',event_prefix=PREFIX,
                            administration_times_min=[0.,1440.],dose_mg=30.,project_root=tmp_path)
    assert policy['verified'] and policy['original_source_event_count']==1 and policy['executable_event_count']==2
    assert len(metadata['administrations'])==1 and policy['native_execution_required'] is True
    assert policy['original_application_xml_sha256']
    wrong=deepcopy(metadata);wrong['name']='unbound'
    with pytest.raises(ValueError,match='metadata differs'):
        repair.prepare_source_expansion(reference,wrong,compound_name='Pioglitazone',event_prefix=PREFIX,
                            administration_times_min=[0.,1440.],dose_mg=30.,project_root=tmp_path)


def test_scoring_verifier_and_executable_metadata_preserve_original_count(tmp_path):
    pkml,reference=source_project(tmp_path);original=repair.describe_saved_simulation(repair._original_document(reference,tmp_path))
    untouched=deepcopy(original)
    policy=repair.prepare_source_expansion(reference,original,compound_name='Pioglitazone',event_prefix=PREFIX,
                            administration_times_min=[0.,1440.],dose_mg=30.,project_root=tmp_path)
    repair.expand_applications(pkml,tmp_path/'out',source_reference=reference,compound_name='Pioglitazone',
                              event_prefix=PREFIX,administration_times_min=[0.,1440.],dose_mg=30.,project_root=tmp_path)
    path=tmp_path/'out/receipt.json'
    actual,verification=repair.executable_metadata(path,original,expected_policy=policy,project_root=tmp_path)
    assert verification['accepted'] and actual['metadata_scope']=='actual_expanded_executable_model_only'
    assert len(actual['administrations'])==2 and len(original['administrations'])==1 and original==untouched
    assert all(a['compound_association_complete'] for a in actual['administrations'])
    assert all(a['administered_compounds']==['Pioglitazone'] for a in actual['administrations'])
    bad=deepcopy(policy);bad['dose_mg']=60.
    with pytest.raises(ValueError,match='policy differs'):
        repair.verify_expansion_receipt(path,expected_policy=bad,project_root=tmp_path)
    repaired=tmp_path/'out/repeated_application.pkml';repaired.write_text(repaired.read_text().replace('value="2"','value="999"'))
    with pytest.raises(ValueError,match='SHA differs'):
        repair.verify_expansion_receipt(path,expected_policy=policy,project_root=tmp_path)
