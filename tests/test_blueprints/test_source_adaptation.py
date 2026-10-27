"""Static source-adaptation checks; never run native solvers or score observations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pbpk_kb.blueprints import source_adaptation as adapter

ROOT = Path('/home/xuyiqian/Research_Projects/Knowledge_Graph_Model_Recommendation')
COLLECTION = ROOT / 'artifacts/pbpk-model-assets/public_collection_20260909'
MID = COLLECTION / 'Midazolam-Model__cc916b56d5e3/source/Midazolam-Model.json'
RIF = COLLECTION / 'Rifampicin-Model__1565559bf7b7/source/Rifampicin-Model.json'
ROS = COLLECTION / 'Rosuvastatin-Model__34fe50412a87/source/Rosuvastatin-Model.json'


@pytest.fixture(autouse=True)
def forbid_execution(monkeypatch):
    from pbpk_kb.blueprints import design_runtime
    monkeypatch.setattr(design_runtime, 'execute_snapshot',
        lambda **kw: pytest.fail('No native solver is permitted in static adapter tests'))


def real(path, simulation):
    return adapter.load_source_model(path, hashlib.sha256(path.read_bytes()).hexdigest(), simulation, ROOT)


def sim(snapshot):
    return snapshot['Simulations'][0]


def protocol(snapshot):
    name = sim(snapshot)['Compounds'][0]['Protocol']['Name']
    return next(p for p in snapshot['Protocols'] if p['Name'] == name)


def parameter(container, name):
    return next(p for p in container['Parameters'] if p['Name'] == name)


@pytest.mark.parametrize('path,name', [
    (MID, 'iv 0.001 mg (5 min)'),
    (MID, 'po 15 mg (tablet) - with high-fat breakfast'),
    (RIF, 'Rifampicin iv 600 mg (3 h) MD OD (7 days)'),
    (RIF, 'Rifampicin po 600 mg MD OD (16 days)')])
def test_real_noop_preserves_complete_native_model_without_observation_sections(path, name):
    raw = json.loads(path.read_text())
    original = next(s for s in raw['Simulations'] if s['Name'] == name)
    source = real(path, name)
    result = adapter.adapt_source_model(source, {}, ROOT)
    assert result['status'] == 'adapted', result['reasons']
    assert result['diff'] == []
    snapshot = result['snapshot']
    for key in ('Individuals', 'Compounds', 'ExpressionProfiles', 'Formulations', 'Protocols', 'Events', 'ObserverSets'):
        assert snapshot.get(key) == raw.get(key)
    for key in ('Parameters', 'Solver', 'OutputSchema', 'Compounds', 'Interactions', 'Events', 'ObserverSets'):
        assert sim(snapshot).get(key) == original.get(key)
    assert sim(snapshot)['HasResults'] is False
    assert not (set(snapshot) & adapter._ROOT_NON_SOLVER)
    assert not (set(sim(snapshot)) & adapter._SIM_NON_SOLVER)
    assert result['manifest']['stripped_non_solver_sections']
    assert result['manifest']['observations_used_for_adaptation'] is False
    assert result['checks']['registered_human_base_verified'] is True
    assert result['checks']['runtime_execution_verified'] is False
    assert snapshot == source['snapshot']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source['source_sha256']
    if path == RIF and '(16 days)' in name:
        assert result['checks']['disabled_process_selections'] == [
            {'MoleculeName': 'CYP3A4'}, {'MoleculeName': 'UGT1A4'}]
        assert len(result['checks']['active_processes']) == 4


def test_real_midazolam_dose_and_infusion_reach_protocol_and_higher_priority_override():
    source = real(MID, 'iv 0.001 mg (5 min)')
    result = adapter.adapt_source_model(source,
        {'simulation_name': 'MID_Request', 'dose_scale': 1000., 'infusion_minutes': 2.}, ROOT)
    assert result['status'] == 'adapted', result['reasons']
    snapshot = result['snapshot']
    assert protocol(snapshot)['Name'] == 'iv 0.001 mg (5 min)'
    assert parameter(protocol(snapshot), 'InputDose')['Value'] == 1.
    assert parameter(protocol(snapshot), 'Infusion time')['Value'] == 2.
    before, after = sim(source['snapshot'])['Parameters'], sim(snapshot)['Parameters']
    assert len(before) == len(after) == 23
    for first, second in zip(before, after, strict=True):
        if first['Path'].endswith('|Infusion time'):
            assert second['Value'] == 2. and second['Unit'] == 'min'
            assert second['Path'] == first['Path']
        else:
            assert second == first
    assert snapshot['Compounds'] == source['snapshot']['Compounds']


def test_real_source_individual_and_protocol_switch_preserve_other_parameters():
    source = real(MID, 'iv 1 mg (5 min)')
    result = adapter.adapt_source_model(source, {
        'individual_name': 'Korean (Yu 2004 study)', 'protocol_name': 'iv 1 mg (2 min)'}, ROOT)
    assert result['status'] == 'adapted', result['reasons']
    selected = sim(result['snapshot'])
    assert selected['Individual'] == 'Korean (Yu 2004 study)'
    assert selected['Compounds'][0]['Protocol']['Name'] == 'iv 1 mg (2 min)'
    for item in selected['Parameters']:
        if item['Path'].endswith('|Infusion time'):
            assert item['Path'].startswith('Events|iv 1 mg (2 min)|') and item['Value'] == 2.
        else:
            assert item in sim(source['snapshot'])['Parameters']


def test_real_repeated_dose_source_switch_preserves_schedule_and_induction():
    source = real(RIF, 'Rifampicin po 600 mg MD OD (7 days)')
    result = adapter.adapt_source_model(source, {
        'protocol_name': 'po 450 mg MD OD (7 days)',
        'individual_name': 'European (P-gp modified, CYP3A4 36 h, EHC off)'}, ROOT)
    assert result['status'] == 'adapted', result['reasons']
    schema = protocol(result['snapshot'])['Schemas'][0]
    assert parameter(schema, 'NumberOfRepetitions')['Value'] == 7.
    assert parameter(schema, 'TimeBetweenRepetitions')['Value'] == 24.
    assert parameter(schema['SchemaItems'][0], 'InputDose')['Value'] == 450.
    selected = sim(result['snapshot'])
    assert len(selected['Interactions']) == 7
    assert selected['Interactions'] == sim(source['snapshot'])['Interactions']
    assert len(selected['Parameters']) == 7
    assert all(p['Path'].startswith('Events|po 450 mg MD OD (7 days)|Oral solution|') for p in selected['Parameters'])


def test_real_repeated_scaling_changes_dose_and_infusion_without_changing_schedule():
    source = real(RIF, 'Rifampicin iv 600 mg (3 h) MD OD (7 days)')
    result = adapter.adapt_source_model(source, {'dose_scale': .5, 'infusion_minutes': 60.}, ROOT)
    assert result['status'] == 'adapted', result['reasons']
    schema = protocol(result['snapshot'])['Schemas'][0]
    assert parameter(schema['SchemaItems'][0], 'InputDose')['Value'] == 300.
    assert parameter(schema['SchemaItems'][0], 'Infusion time')['Value'] == 60.
    assert schema['Parameters'] == protocol(source['snapshot'])['Schemas'][0]['Parameters']
    assert all(p['Value'] == 60. for p in sim(result['snapshot'])['Parameters'])


def test_real_formulation_switch_rebinds_overrides_and_preserves_release_parameters():
    source = real(MID, 'po 10 mg (solution)')
    result = adapter.adapt_source_model(source, {'formulation_name': 'Tablet (Dormicum)'}, ROOT)
    assert result['status'] == 'adapted', result['reasons']
    selected = sim(result['snapshot'])
    assert selected['Compounds'][0]['Protocol']['Formulations'][0]['Name'] == 'Tablet (Dormicum)'
    assert result['snapshot']['Formulations'] == source['snapshot']['Formulations']
    for first, second in zip(sim(source['snapshot'])['Parameters'], selected['Parameters'], strict=True):
        assert second['Path'] == first['Path'].replace('|Oral solution|', '|Tablet (Dormicum)|')
        assert {k:v for k,v in second.items() if k != 'Path'} == {k:v for k,v in first.items() if k != 'Path'}


def test_v79_source_is_blocked_until_real_migration():
    source = real(ROS, json.loads(ROS.read_text())['Simulations'][0]['Name'])
    result = adapter.adapt_source_model(source, {}, ROOT)
    assert result['status'] == 'blocked' and 'snapshot' not in result
    assert 'REAL_MIGRATION' in result['reasons'][0]
    assert source['snapshot']['Version'] == 79


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, 'load_standard_human_model',
        lambda root: {'checksum': 'a'*64, 'model': {'source_version': '12.3.173'}})
    raw = {'Version': 80,
        'Individuals': [{'Name': 'Human', 'OriginData': {'Species': 'Human', 'CalculationMethods': ['native']},
            'Parameters': [{'Path': 'Organism|Liver|EHC continuous fraction', 'Value': .5}]},
            {'Name': 'Other Human', 'OriginData': {'Species': 'Human'}}],
        'Compounds': [{'Name': 'Drug', 'IsSmallMolecule': True,
            'Processes': [{'InternalName': 'LiverClearance', 'DataSource': 'Source', 'Parameters': []}]}],
        'Protocols': [{'Name': 'Dose', 'ApplicationType': 'Intravenous', 'DosingInterval': 'Single',
            'Parameters': [{'Name': 'Start time', 'Value': 0., 'Unit': 'h'},
                {'Name': 'InputDose', 'Value': 10., 'Unit': 'mg'},
                {'Name': 'Infusion time', 'Value': 5., 'Unit': 'min'}]}],
        'Simulations': [{'Name': 'Simulation', 'Model': '4Comp', 'Individual': 'Human',
            'Compounds': [{'Name': 'Drug', 'Protocol': {'Name': 'Dose'},
                'Processes': [{'Name': 'Total Hepatic Clearance-Source', 'SystemicProcessType': 'Hepatic'}]}],
            'Parameters': [{'Path': 'Events|Dose|Application_1|ProtocolSchemaItem|InputDose',
                'Value': 20000., 'Unit': 'µg', 'ValueOrigin': {'Source': 'ParameterIdentification'}}],
            'OutputSelections': ['Organism|PeripheralVenousBlood|Drug|Plasma'],
            'ObservedData': ['Curve'], 'OutputMappings': [{'ObservedData': 'Curve'}],
            'IndividualAnalyses': [{'Name': 'Old plot'}], 'HasResults': True,
            'UnknownNativeField': {'Preserve': 3}}],
        'ObservedData': [{'Name': 'Curve', 'Values': [999999.]}],
        'ParameterIdentifications': [{'Name': 'OldFit', 'Results': [999999.]}],
        'UnknownNativeRootBlock': {'Preserve': 4}}
    def write():
        path = tmp_path/'source.json'
        path.write_text(json.dumps(raw))
        return adapter.load_source_model(path, hashlib.sha256(path.read_bytes()).hexdigest(), 'Simulation', tmp_path)
    return tmp_path, raw, write


def test_unknown_native_fields_are_preserved(synthetic):
    root, raw, write = synthetic
    result = adapter.adapt_source_model(write(), {}, root)
    assert result['status'] == 'adapted'
    assert sim(result['snapshot'])['UnknownNativeField'] == {'Preserve': 3}
    assert result['snapshot']['UnknownNativeRootBlock'] == {'Preserve': 4}
    assert result['snapshot']['Individuals'][0]['Parameters'] == raw['Individuals'][0]['Parameters']


@pytest.mark.parametrize('user_request,expected', [
    ({'dose_scale': .5}, 10000.), ({'dose': {'value': .005, 'unit': 'g'}}, 5000.)])
def test_dose_override_synchronized_with_explicit_units_and_new_provenance(synthetic, user_request, expected):
    root, _, write = synthetic
    source = write()
    result = adapter.adapt_source_model(source, user_request, root)
    assert result['status'] == 'adapted', result['reasons']
    override = sim(result['snapshot'])['Parameters'][0]
    assert override['Value'] == expected and override['Unit'] == 'µg'
    assert 'ValueOrigin' not in override
    assert parameter(protocol(result['snapshot']), 'InputDose')['Value'] == 5.
    assert sim(source['snapshot'])['Parameters'][0]['Value'] == 20000.


@pytest.mark.parametrize('user_request,reason', [
    ({'new_drug': 'Other'}, 'UNSUPPORTED_REQUEST_FIELDS'),
    ({'compound_name': 'Other'}, 'NEW_DRUG'),
    ({'ddi': {'perpetrator': 'Other'}}, 'UNSUPPORTED_REQUEST_FIELDS'),
    ({'regimen': {'repetitions': 7}}, 'UNSUPPORTED_REQUEST_FIELDS'),
    ({'dose_scale': True}, 'INVALID_POSITIVE_NUMBER'),
    ({'dose_scale': 0}, 'INVALID_POSITIVE_NUMBER'),
    ({'dose_scale': float('nan')}, 'INVALID_POSITIVE_NUMBER'),
    ({'dose_scale': 2., 'dose': {'value': 10., 'unit': 'mg'}}, 'MUTUALLY_EXCLUSIVE'),
    ({'dose': {'value': 2., 'unit': 'mg/kg'}}, 'BODY_WEIGHT_CONVERSION'),
    ({'infusion_minutes': -2.}, 'INVALID_POSITIVE_NUMBER'),
    ({'simulation_name': '../escape'}, 'UNSAFE_OUTPUT'),
    ({'individual_name': 'Unknown'}, 'MISSING_REFERENCE'),
    ({'protocol_name': 'Unknown'}, 'MISSING_REFERENCE'),
    ({'formulation_name': 'Unknown'}, 'FORMULATION_CHANGE_REQUIRES'),
    ({'output_selections': ['Organism|NewOrgan|Drug']}, 'OUTPUTS_MUST'),
    ({'output_selections': []}, 'OUTPUTS_MUST')])
def test_unsupported_request_has_concrete_reason_and_no_partial_snapshot(synthetic, user_request, reason):
    root, _, write = synthetic
    source = write()
    result = adapter.adapt_source_model(source, user_request, root)
    assert result['status'] == 'blocked' and 'snapshot' not in result
    assert reason in result['reasons'][0]
    assert adapter._digest(source['snapshot']) == source['snapshot_sha256']


@pytest.mark.parametrize('change,reason', [
    ('duplicate_dose', 'DUPLICATE_LOCAL_OVERRIDE'), ('unbound_dose', 'UNBOUND_LOCAL_DOSE'),
    ('wrong_application', 'NONEXISTENT_APPLICATION'), ('multiple_drugs', 'DUPLICATE_SELECTED_COMPOUND_BINDINGS'),
    ('missing_process', 'UNBOUND_OR_AMBIGUOUS_PROCESS'), ('nonhuman', 'MUST_BE_HUMAN')])
def test_source_dependency_and_override_ambiguities_block(synthetic, change, reason):
    root, raw, write = synthetic
    selected = raw['Simulations'][0]
    if change == 'duplicate_dose':
        selected['Parameters'].append(deepcopy(selected['Parameters'][0]))
    elif change == 'unbound_dose':
        selected['Parameters'][0]['Path'] = 'Events|OtherDose|Application_1|ProtocolSchemaItem|InputDose'
    elif change == 'wrong_application':
        selected['Parameters'][0]['Path'] = 'Events|Dose|Application_2|ProtocolSchemaItem|InputDose'
    elif change == 'multiple_drugs':
        selected['Compounds'].append(deepcopy(selected['Compounds'][0]))
    elif change == 'missing_process':
        raw['Compounds'][0]['Processes'] = []
    else:
        raw['Individuals'][0]['OriginData']['Species'] = 'Rat'
    result = adapter.adapt_source_model(write(), {'dose_scale': 2.}, root)
    assert result['status'] == 'blocked' and reason in result['reasons'][0]


def test_individual_change_blocks_conflicting_physiology_override(synthetic):
    root, raw, write = synthetic
    raw['Simulations'][0]['Parameters'].append({'Path': 'Organism|Liver|Volume', 'Value': 1.5, 'Unit': 'l'})
    result = adapter.adapt_source_model(write(), {'individual_name': 'Other Human'}, root)
    assert result['status'] == 'blocked' and 'LOCAL_PHYSIOLOGY_OVERRIDE' in result['reasons'][0]


def test_source_checksum_and_in_memory_forgery_are_rejected(synthetic):
    root, _, write = synthetic
    source = write()
    with pytest.raises(adapter.SourceAdaptationError, match='CHECKSUM'):
        adapter.load_source_model(root/'source.json', 'b'*64, 'Simulation', root)
    source['snapshot']['Compounds'][0]['Name'] = 'Tampered'
    with pytest.raises(adapter.SourceAdaptationError, match='IMPORTED_SNAPSHOT_CHANGED'):
        adapter.adapt_source_model(source, {}, root)
    source['snapshot_sha256'] = adapter._digest(source['snapshot'])
    with pytest.raises(adapter.SourceAdaptationError, match='DIFFERS_FROM_VERIFIED_SOURCE'):
        adapter.adapt_source_model(source, {}, root)
    source = write()
    (root/'source.json').write_text('{}')
    with pytest.raises(adapter.SourceAdaptationError, match='CHECKSUM'):
        adapter.adapt_source_model(source, {}, root)


def test_source_stays_inside_explicit_project_root(synthetic):
    root, _, write = synthetic
    source = write()
    restricted = root/'nested'
    restricted.mkdir()
    with pytest.raises(adapter.SourceAdaptationError, match='OUTSIDE_PROJECT'):
        adapter.load_source_model(root/'source.json', source['source_sha256'], 'Simulation', restricted)


def test_duplicate_json_keys_rejected(synthetic):
    root, _, _ = synthetic
    path = root/'duplicate.json'
    path.write_text('{"Version":80,"Version":79}')
    with pytest.raises(adapter.SourceAdaptationError, match='DUPLICATE_JSON_KEY'):
        adapter.load_source_model(path, hashlib.sha256(path.read_bytes()).hexdigest(), 'S', root)


def test_protocol_switch_preserves_absolute_start_of_later_repeated_application(synthetic):
    root, raw, write = synthetic
    item = deepcopy(raw['Protocols'][0])
    item.pop('DosingInterval')
    item['Name'] = 'Item'
    old = {'Name': 'Dose', 'DosingInterval': 'Single', 'TimeUnit': 'h',
        'Schemas': [{'Name': 'Schema', 'SchemaItems': [item], 'Parameters': [
            {'Name': 'Start time', 'Value': 0., 'Unit': 'h'},
            {'Name': 'NumberOfRepetitions', 'Value': 3.},
            {'Name': 'TimeBetweenRepetitions', 'Value': 24., 'Unit': 'h'}]}]}
    target = deepcopy(old)
    target['Name'] = 'Target'
    raw['Protocols'] = [old, target]
    raw['Simulations'][0]['Parameters'] = [{
        'Path': 'Events|Dose|Application_2|ProtocolSchemaItem|Start time', 'Value': 24., 'Unit': 'h'}]
    result = adapter.adapt_source_model(write(), {'protocol_name': 'Target'}, root)
    assert result['status'] == 'adapted', result['reasons']
    override = sim(result['snapshot'])['Parameters'][0]
    assert override['Path'] == 'Events|Target|Application_2|ProtocolSchemaItem|Start time'
    assert override['Value'] == 24.  # Not the schema item's relative zero start.


def test_real_itraconazole_retains_metabolite_chain_and_interactions():
    path = COLLECTION / 'Itraconazole-Model__59b6b686ba37/source/Itraconazole-Model.json'
    source = real(path, '100 mg IV SD')
    result = adapter.adapt_source_model(source, {'dose': {'value': 150., 'unit': 'mg'}}, ROOT)
    assert result['status'] == 'adapted', result['reasons']
    assert result['snapshot']['Compounds'] == source['snapshot']['Compounds']
    assert sim(result['snapshot'])['Compounds'][0]['Name'] == 'Hydroxy-Itraconazole'
    assert result['checks']['retained_undosed_metabolites'] == [
        'Hydroxy-Itraconazole', 'Keto-Itraconazole', 'N-desalkyl-Itraconazole']
    assert len(result['checks']['explicit_metabolite_links']) == 3
    assert result['checks']['process_selections_unchanged'] is True
    assert result['checks']['interactions_unchanged'] is True
    parent = adapter._one_administered_selection(sim(result['snapshot']))
    assert parent['Name'] == 'Itraconazole'
    selected_protocol = adapter._named(result['snapshot']['Protocols'], parent['Protocol']['Name'], 'Protocols')
    assert adapter._parameters(adapter._administrations(selected_protocol)[0][0])['InputDose']['Value'] == 150


@pytest.mark.parametrize('mutation,reason', [
    ('unlinked', 'UNDOSED_COMPOUND_NOT_AN_EXPLICIT_PARENT_METABOLITE'),
    ('second_dose', 'MULTIDRUG_SIMULATION'),
    ('missing_selected_metabolite', 'NAMED_METABOLITE_NEEDS_EXPLICIT_MULTICOMPOUND_BINDING'),
    ('conflicting_metabolite', 'NONUNIQUE_OR_MISSING_REFERENCE'),
])
def test_selected_metabolites_must_be_explicit_undosed_dependencies(synthetic, mutation, reason):
    root, raw, write = synthetic
    raw['Compounds'].append({'Name': 'Metabolite', 'IsSmallMolecule': True, 'Processes': []})
    child = {'Name': 'Metabolite', 'Processes': []}
    raw['Simulations'][0]['Compounds'].insert(0, child)
    parent = raw['Simulations'][0]['Compounds'][1]
    if mutation == 'second_dose':
        child['Protocol'] = {'Name': 'Dose'}
    elif mutation in {'missing_selected_metabolite', 'conflicting_metabolite'}:
        raw['Compounds'][0]['Processes'][0]['Metabolite'] = 'Metabolite'
        parent['Processes'][0]['MetaboliteName'] = 'Metabolite' if mutation == 'missing_selected_metabolite' else 'Other'
        if mutation == 'missing_selected_metabolite':
            raw['Simulations'][0]['Compounds'].remove(child)
    result = adapter.adapt_source_model(write(), {}, root)
    assert result['status'] == 'blocked' and reason in result['reasons'][0]
