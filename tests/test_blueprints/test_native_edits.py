"""Static native snapshot edits, including source-backed cases; no solver."""
from copy import deepcopy
import hashlib
import json
import sqlite3

import pytest

from pbpk_kb.blueprints import native_edits
from test_human_model_binding import _payload, provider, PROJECT_ROOT, DEFINITION_CHECKSUM


@pytest.fixture
def bound_methods(provider, monkeypatch):
    def source(root, method):
        assert root == PROJECT_ROOT
        assert method in native_edits.PARTITION_ACTIONS.values()
        return {
            'provenance_id': '00000000-0000-0000-0000-000000000002',
            'provenance_kind': 'model_asset', 'source_id': 'synthetic-native.db',
            'source_location': method, 'source_checksum': 'b' * 64,
        }, {'checksum': DEFINITION_CHECKSUM}
    monkeypatch.setattr(native_edits, '_method_source', source)


def test_partition_choices_change_only_constitutive_native_paths(bound_methods):
    payload = _payload(oral=True)
    original = deepcopy(payload)
    specs = [{'parameter_id': 'fixture', 'unit': '1/min', 'lower_bound': 0, 'upper_bound': 1}]
    standard, rodgers = native_edits.partition_candidates(payload, project_root=PROJECT_ROOT, parameter_specs=specs)
    assert payload == original
    assert standard['edit_count'] == 0 and rodgers['edit_count'] == 1
    assert standard['semantic_hash'] != rodgers['semantic_hash']
    assert standard['parameterization_hash'] == rodgers['parameterization_hash']
    assert rodgers['parameter_specs'] == specs
    assert rodgers['snapshot']['Individuals'] == standard['snapshot']['Individuals']
    assert rodgers['snapshot']['Protocols'] == standard['snapshot']['Protocols']
    assert rodgers['snapshot']['Compounds'][0]['Processes'] == standard['snapshot']['Compounds'][0]['Processes']
    manifest = rodgers['edit_manifest']
    assert {row['path'] for row in manifest['pathdiff']} == {
        '/Compounds/0/CalculationMethods/0', '/Simulations/0/Compounds/0/CalculationMethods/0',
    }
    assert manifest['change_class'] == 'constitutive_method'
    assert manifest['topology_changed'] is False
    assert manifest['drug_parameter_values_changed'] is False
    assert manifest['compile_status'] == manifest['simulation_status'] == 'not_run'
    assert manifest['execution'] == {'solver_calls': 0, 'gpu_jobs': 0, 'llm_calls': 0}


def test_reapplying_choice_is_identity_and_scenario_is_not_structure(bound_methods):
    payload = _payload()
    first = native_edits.compile_candidate(payload, 'partition_rodgers_rowland', project_root=PROJECT_ROOT)
    repeated = native_edits.compile_candidate(first['request'], 'partition_rodgers_rowland', project_root=PROJECT_ROOT)
    assert repeated['edit_count'] == 0 and repeated['edit_manifest']['pathdiff'] == []
    assert repeated['semantic_hash'] == first['semantic_hash']
    assert repeated['parameterization_hash'] == first['parameterization_hash']
    payload['requirement']['dose'] = 20
    payload['instance']['protocol']['Parameters'][0]['Value'] = 20
    other_dose = native_edits.compile_candidate(payload, 'partition_rodgers_rowland', project_root=PROJECT_ROOT)
    assert other_dose['semantic_hash'] == first['semantic_hash']
    assert other_dose['edit_manifest']['snapshot_sha256'] != first['edit_manifest']['snapshot_sha256']


@pytest.mark.parametrize('action', ['hepatic_clearance_replace', 'invent_new_organs', '', None])
def test_unimplemented_actions_are_rejected_without_source_or_runtime_access(monkeypatch, action):
    monkeypatch.setattr(native_edits, '_method_source', lambda *args: pytest.fail('Must reject before source access'))
    with pytest.raises(ValueError, match='Unsupported native action'):
        native_edits.compile_candidate(_payload(), action, project_root=PROJECT_ROOT)


def test_generic_unapplied_modules_cannot_be_mislabelled_native(bound_methods):
    payload = _payload()
    payload['modules'] = [{'module_id': 'unimplemented', 'version': '1', 'mechanism': 'transport',
                           'provenance': [payload['instance']['human_source']]}]
    with pytest.raises(ValueError, match='Unapplied generic modules'):
        native_edits.compile_candidate(payload, 'partition_rodgers_rowland', project_root=PROJECT_ROOT)


def test_parameter_contract_hash_is_sensitive_to_bounds_but_does_not_edit_values(bound_methods):
    specs = [{'parameter_id': 'fixture', 'unit': '1/min', 'lower_bound': 0, 'upper_bound': 1}]
    first = native_edits.compile_candidate(_payload(), 'partition_pksim_standard', project_root=PROJECT_ROOT, parameter_specs=specs)
    specs[0]['upper_bound'] = 2
    second = native_edits.compile_candidate(_payload(), 'partition_pksim_standard', project_root=PROJECT_ROOT, parameter_specs=specs)
    assert first['semantic_hash'] == second['semantic_hash']
    assert first['parameterization_hash'] != second['parameterization_hash']
    assert first['snapshot'] == second['snapshot']
    assert first['parameter_specs'][0]['upper_bound'] == 1


def test_method_reference_is_bound_to_exact_native_database_category(monkeypatch, tmp_path):
    database = tmp_path / 'tiny.db'
    method = native_edits.PARTITION_ACTIONS['partition_rodgers_rowland']
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE tab_calculation_methods(calculation_method TEXT, category TEXT)')
        connection.execute('INSERT INTO tab_calculation_methods VALUES (?, ?)', (method, 'DistributionCellular'))
    digest = hashlib.sha256(database.read_bytes()).hexdigest()
    definition = {'checksum': DEFINITION_CHECKSUM, 'resources': [
        {'role': 'database', 'path': 'tiny.db', 'sha256': digest}]}
    monkeypatch.setattr(native_edits, 'load_standard_human_model', lambda root: definition)
    source, bound = native_edits._method_source(tmp_path, method)
    assert source['source_checksum'] == digest
    assert source['source_id'] == 'tiny.db'
    assert 'DistributionCellular' in source['source_location']
    assert bound == definition
    with pytest.raises(ValueError, match='absent'):
        native_edits._method_source(tmp_path, native_edits.PARTITION_ACTIONS['partition_pksim_standard'])


def _parameter_fixture():
    payload = _payload(oral=True)
    compound = payload['instance']['compound']
    mappings = []
    for name, fraction in [('a', .8), ('b', .15), ('c', .05)]:
        compound['Processes'].append({'InternalName': 'MetabolizationIntrinsic_FirstOrder', 'Molecule': 'CYP1A2',
            'Species': 'Human', 'DataSource': name, 'Parameters': [
                {'Name': 'Intrinsic clearance', 'Value': 0., 'Unit': 'l/min'},
                {'Name': 'Specific clearance', 'Value': .1 * fraction, 'Unit': '1/min'},
            ]})
        mappings.append({'runtime_path': f'ToyDrug-CYP1A2-{name}|Specific clearance', 'multiplier': fraction})
    specs = [
        {'parameter_id': 'total_cyp1a2_specific_clearance', 'unit': '1/min', 'lower_bound': 0., 'upper_bound': 1.,
         'runtime_mappings': mappings},
        {'parameter_id': 'specific_intestinal_permeability_transcellular', 'unit': 'dm/min',
         'lower_bound': 1e-9, 'upper_bound': .01,
         'runtime_mappings': [{'runtime_path': 'ToyDrug|Specific intestinal permeability (transcellular)', 'multiplier': 1.}]},
    ]
    return payload, specs


def test_parameter_application_preserves_branch_sum_units_and_unrelated_fields():
    payload, specs = _parameter_fixture()
    specs[1]['runtime_mappings'][0]['mapping_rule'] = 'Audited scalar identity; annotation is not a runtime field'
    original = deepcopy(payload)
    source = deepcopy(payload['instance']['human_source'])
    changed = native_edits.apply_parameter_values(payload, specs, [.3, .0006], source)
    assert payload == original
    before, after = original['instance'], changed['instance']
    assert after['individual'] == before['individual']
    assert after['protocol'] == before['protocol']
    assert after['compound']['CalculationMethods'] == before['compound']['CalculationMethods']
    assert after['compound']['Processes'][0] == before['compound']['Processes'][0]
    branch_values = [p['Parameters'][1]['Value'] for p in after['compound']['Processes'][1:]]
    assert branch_values == pytest.approx([.24, .045, .015])
    assert sum(branch_values) == pytest.approx(.3)
    assert after['compound']['IntestinalPermeability'][0]['Parameters'][0]['Value'] == pytest.approx(.0001)


@pytest.mark.parametrize('change', [
    lambda specs, values: values.__setitem__(0, float('inf')),
    lambda specs, values: values.__setitem__(0, 2.),
    lambda specs, values: specs[0]['runtime_mappings'][0].update(multiplier=.9),
    lambda specs, values: specs[0]['runtime_mappings'][0].update(runtime_path='Other-CYP1A2-a|Specific clearance'),
    lambda specs, values: specs[0].update(unit='ml/min/kg'),
    lambda specs, values: specs[1]['runtime_mappings'][0].update(multiplier=2.),
    lambda specs, values: specs[1]['runtime_mappings'][0].update(multiplier=True),
    lambda specs, values: specs[1]['runtime_mappings'][0].update(multiplier='1'),
    lambda specs, values: specs[1]['runtime_mappings'][0].update(multiplier=float('nan')),
    lambda specs, values: specs[1]['runtime_mappings'][0].update(multiplier=float('inf')),
    lambda specs, values: specs[1]['runtime_mappings'][0].update(multiplier=1.000000000000001),
    lambda specs, values: specs[1]['runtime_mappings'][0].pop('multiplier'),
    lambda specs, values: specs[1]['runtime_mappings'][0].update(
        runtime_path='OtherDrug|Specific intestinal permeability (transcellular)'),
    lambda specs, values: specs[1].update(unit='cm/s'),
])
def test_parameter_application_rejects_drift_before_native_build(change):
    payload, specs = _parameter_fixture()
    specs[1]['runtime_mappings'][0]['mapping_rule'] = 'Annotation must not bypass native mapping validation'
    values = [.3, .0006]
    change(specs, values)
    with pytest.raises(ValueError):
        native_edits.apply_parameter_values(payload, specs, values, payload['instance']['human_source'])


@pytest.fixture(scope='module')
def audited_caffeine_rebuild_inputs():
    from pbpk_kb.blueprints.experiment_cases import make_caffeine_experiment_case

    # This constructor verifies original source bytes and only the approved
    # historical development studies. Final-test observations are not loaded.
    case = make_caffeine_experiment_case(PROJECT_ROOT)
    assert case['observation_counts']['test_loaded'] == 0
    specs = case['parameter_search']['parameters']
    assert 'mapping_rule' in specs[1]['runtime_mappings'][0]
    source = deepcopy(specs[0]['source'])
    assert specs[1]['source']['source_id'] == source['source_id']
    assert specs[1]['source']['source_checksum'] == source['source_checksum']
    source['source_location'] = 'parameters; original source-provided initial values, static regression only'
    # Only source-backed starts are applied; this test does not estimate values
    # or use calibration/selection observations in any calculation.
    return specs, source, {scenario['scenario_id']: deepcopy(scenario['design_request'])
                           for scenario in case['scenarios']}


@pytest.mark.parametrize('scenario_id', [
    'blanchard_1983_iv_5mgkg', 'blanchard_1983_oral_5mgkg', 'wahllander_1989_iv_3mgkg',
])
@pytest.mark.parametrize('action', list(native_edits.PARTITION_ACTIONS))
def test_actual_case_parameter_specs_rebuild_all_scenarios_statically(
    audited_caffeine_rebuild_inputs, scenario_id, action, monkeypatch,
):
    from pbpk_kb.blueprints import design_runtime

    monkeypatch.setattr(design_runtime, 'execute_snapshot',
                        lambda **kwargs: pytest.fail('Static rebuild must never run a solver'))
    specs, source, requests = audited_caffeine_rebuild_inputs
    request = requests[scenario_id]
    original = deepcopy(request)
    values = [spec['initial_value'] for spec in specs]
    candidate = native_edits.compile_candidate(request, action, project_root=PROJECT_ROOT, parameter_specs=specs)
    updated = native_edits.apply_parameter_values(candidate['request'], specs, values, source)
    rebuilt = native_edits.compile_candidate(updated, action, project_root=PROJECT_ROOT, parameter_specs=specs)
    assert request == original
    snapshot = rebuilt['snapshot']
    assert snapshot['Individuals'] == candidate['snapshot']['Individuals']
    assert snapshot['Protocols'] == candidate['snapshot']['Protocols']
    compound = snapshot['Compounds'][0]
    assert native_edits.PARTITION_ACTIONS[action] in compound['CalculationMethods']
    branches = [process for process in compound['Processes'] if process.get('Molecule') == 'CYP1A2']
    actual_clearance = sum(next(parameter['Value'] for parameter in branch['Parameters']
                                if parameter['Name'] == 'Specific clearance') for branch in branches)
    assert actual_clearance == pytest.approx(values[0])
    assert compound['IntestinalPermeability'][0]['Parameters'][0]['Value'] == pytest.approx(values[1] / 6)
    assert rebuilt['request']['instance']['compound_sources']['Processes'] == source
    assert rebuilt['request']['instance']['compound_sources']['IntestinalPermeability'] == source
    assert len(snapshot['Simulations']) == 1
    assert snapshot['Simulations'][0]['HasResults'] is False
    assert 'ObservedData' not in snapshot and 'ParameterIdentifications' not in snapshot
    assert rebuilt['edit_manifest']['execution'] == {'solver_calls': 0, 'gpu_jobs': 0, 'llm_calls': 0}


def _cyp3a4_parameter_fixture():
    from test_human_model_snapshot import _cyp3a4_request

    payload = _payload()
    instance = _cyp3a4_request()
    instance.pop('expected_compound_name')
    instance['population_context'] = 'healthy_adult'
    payload['instance'] = instance
    specs = [
        {'parameter_id': 'cyp3a4_intrinsic_clearance', 'unit': 'l/min', 'lower_bound': .1, 'upper_bound': 2.,
         'source': deepcopy(instance['human_source']),
         'runtime_mappings': [{'runtime_path': 'ToyDrug-CYP3A4-toy-intrinsic|Intrinsic clearance',
                               'multiplier': 1., 'mapping_rule': 'reviewed identity'}]},
        {'parameter_id': 'lipophilicity', 'unit': 'Log Units', 'lower_bound': 1.3, 'upper_bound': 2.,
         'source': deepcopy(instance['human_source']),
         'runtime_mappings': [{'runtime_path': 'ToyDrug|Lipophilicity', 'multiplier': 1.}]},
    ]
    return payload, specs


@pytest.mark.parametrize('action', list(native_edits.PARTITION_ACTIONS))
def test_cyp3a4_and_lipophilicity_apply_rebuild_without_changing_other_inputs(bound_methods, action):
    payload, specs = _cyp3a4_parameter_fixture()
    before = deepcopy(payload)
    candidate = native_edits.compile_candidate(payload, action, project_root=PROJECT_ROOT, parameter_specs=specs)
    updated = native_edits.apply_parameter_values(candidate['request'], specs, [.6, 1.8], payload['instance']['human_source'])
    rebuilt = native_edits.compile_candidate(updated, action, project_root=PROJECT_ROOT, parameter_specs=specs)
    assert payload == before
    assert rebuilt['snapshot']['Individuals'] == candidate['snapshot']['Individuals']
    assert rebuilt['snapshot']['Protocols'] == candidate['snapshot']['Protocols']
    compound = rebuilt['snapshot']['Compounds'][0]
    assert compound['Processes'][0] == before['instance']['compound']['Processes'][0]
    assert compound['Processes'][1]['Parameters'] == [{'Name': 'Intrinsic clearance', 'Value': .6, 'Unit': 'l/min'}]
    assert compound['Lipophilicity'][0]['Parameters'][0]['Value'] == 1.8
    assert rebuilt['semantic_hash'] == candidate['semantic_hash']
    assert rebuilt['parameterization_hash'] != candidate['parameterization_hash']
    assert rebuilt['edit_manifest']['execution']['solver_calls'] == 0


@pytest.mark.parametrize('change', [
    lambda p, s: s[0].pop('source'),
    lambda p, s: s[1]['source'].update(source_checksum='invalid'),
    lambda p, s: s[0].update(unit='1/min'),
    lambda p, s: s[1].update(unit='dimensionless'),
    lambda p, s: s[0]['runtime_mappings'][0].update(runtime_path='Other-CYP3A4-toy-intrinsic|Intrinsic clearance'),
    lambda p, s: s[1]['runtime_mappings'][0].update(runtime_path='ToyDrug|Organism|Liver|Volume'),
    lambda p, s: s[0]['runtime_mappings'][0].update(multiplier=True),
    lambda p, s: s[1]['runtime_mappings'][0].update(multiplier=float('nan')),
    lambda p, s: s[1]['runtime_mappings'][0].update(multiplier=2),
    lambda p, s: p['instance']['compound']['Processes'][1]['Parameters'].append(
        {'Name': 'Specific clearance', 'Value': .1, 'Unit': '1/min'}),
    lambda p, s: p['instance']['compound']['Processes'][1].update(InternalName='MetabolizationMichaelisMenten'),
    lambda p, s: p['instance']['compound']['Processes'][1].update(Species='Rat'),
    lambda p, s: p['instance']['compound']['Processes'].append(deepcopy(p['instance']['compound']['Processes'][1])),
    lambda p, s: p['instance']['compound']['Lipophilicity'].append(deepcopy(p['instance']['compound']['Lipophilicity'][0])),
])
def test_cyp3a4_parameter_contract_rejects_arbitrary_paths_units_or_missing_source(change):
    payload, specs = _cyp3a4_parameter_fixture()
    change(payload, specs)
    with pytest.raises(ValueError):
        native_edits.apply_parameter_values(payload, specs, [.6, 1.8], payload['instance']['human_source'])


@pytest.fixture(scope='module')
def actual_alfentanil_development_case():
    path = PROJECT_ROOT / 'experiments/blueprint_design/development_inputs/alfentanil_multistudy_v1/input_package.json'
    text = path.read_text()
    # Decode only the development case. External scenarios and their observed
    # concentrations, if present elsewhere in the package, are not decoded.
    start = text.index(':', text.index('"case"')) + 1
    while text[start].isspace():
        start += 1
    case = json.JSONDecoder().raw_decode(text, start)[0]
    assert [s['role'] for s in case['scenarios']] == ['calibration', 'calibration', 'selection']
    return case


@pytest.mark.parametrize('scenario_index', [0, 1, 2])
@pytest.mark.parametrize('action', list(native_edits.PARTITION_ACTIONS))
def test_real_alfentanil_requests_compile_apply_recompile_statically(actual_alfentanil_development_case, scenario_index, action):
    case = actual_alfentanil_development_case
    specs = case['parameter_search']['parameters']
    request = case['scenarios'][scenario_index]['design_request']
    candidate = native_edits.compile_candidate(request, action, project_root=PROJECT_ROOT, parameter_specs=specs)
    updated = candidate['request']
    for spec in specs:
        updated = native_edits.apply_parameter_values(updated, [spec], [spec['initial_value']], spec['source'])
    rebuilt = native_edits.compile_candidate(updated, action, project_root=PROJECT_ROOT, parameter_specs=specs)
    compound = rebuilt['snapshot']['Compounds'][0]
    assert 'Permeability' not in compound and 'IntestinalPermeability' not in compound
    assert compound['Lipophilicity'][0]['Parameters'][0]['Value'] == pytest.approx(1.75)
    cyp = next(p for p in compound['Processes'] if p.get('Molecule') == 'CYP3A4')
    assert cyp['Parameters'] == [{'Name': 'Intrinsic clearance', 'Value': .504, 'Unit': 'l/min'}]
    assert len(rebuilt['snapshot']['ExpressionProfiles'][0]['Parameters']) == 15
    assert rebuilt['snapshot']['Protocols'] == candidate['snapshot']['Protocols']
    assert rebuilt['snapshot']['Individuals'] == candidate['snapshot']['Individuals']
    assert rebuilt['edit_manifest']['compile_status'] == 'not_run'
    assert rebuilt['edit_manifest']['execution'] == {'solver_calls': 0, 'gpu_jobs': 0, 'llm_calls': 0}


def _generic_process_spec(payload, process, parameter_name, unit, lower=0., upper=2.):
    identity = {'internal_name': process['InternalName'], 'molecule': process.get('Molecule'),
                'data_source': process['DataSource'], 'parameter_name': parameter_name}
    return {'parameter_id': 'single_process_clearance', 'native_process': identity,
            'unit': unit, 'lower_bound': lower, 'upper_bound': upper,
            'source': deepcopy(payload['instance']['human_source']),
            'runtime_mappings': [{'runtime_path': native_edits.reviewed_process_parameter_path(
                payload['instance']['compound']['Name'], identity), 'multiplier': 1.,
                'mapping_rule': 'explicit source native parameter'}]}


@pytest.mark.parametrize('molecule', ['UGT2B7', 'SULT2A1', 'CYP2D6'])
def test_generic_clearance_updates_only_selected_sourced_downstream_parameter(molecule):
    payload, _ = _cyp3a4_parameter_fixture()
    process = payload['instance']['compound']['Processes'][1]
    process['Molecule'] = molecule
    process['Parameters'].append({'Name': 'Specific clearance', 'Value': .1, 'Unit': '1/min'})
    spec = _generic_process_spec(payload, process, 'Specific clearance', '1/min')
    before = deepcopy(payload)
    updated = native_edits.apply_parameter_values(payload, [spec], [.2], spec['source'])
    assert payload == before
    assert updated['instance']['compound']['Processes'][1]['Parameters'][0] == process['Parameters'][0]
    assert updated['instance']['compound']['Processes'][1]['Parameters'][1]['Value'] == .2
    assert updated['instance']['compound']['Processes'][0] == before['instance']['compound']['Processes'][0]
    assert updated['instance']['individual'] == before['instance']['individual']
    hidden = _generic_process_spec(payload, process, 'Intrinsic clearance', 'l/min')
    with pytest.raises(ValueError, match='upstream'):
        native_edits.apply_parameter_values(payload, [hidden], [.5], hidden['source'])


@pytest.mark.parametrize('internal,parameter,unit,value', [
    ('GlomerularFiltration', 'GFR fraction', 'dimensionless', .8),
    ('KidneyClearance', 'Plasma clearance', 'ml/min/kg', 1.2),
    ('LiverClearance', 'Plasma clearance', 'l/h/kg', 1.2),
    ('LiverClearance', 'Specific clearance', '1/h', .4),
    ('rCYP450_FirstOrder', 'CLspec/[Enzyme]', 'l/µmol/min', .04),
    ('MetabolizationLiverMicrosomes_FirstOrder', 'In vitro CL for liver microsomes', 'µl/min/mg mic. protein', 1.5),
])
def test_generic_source_parameter_whitelist_keeps_native_unit(internal, parameter, unit, value):
    payload, _ = _cyp3a4_parameter_fixture()
    process = {'InternalName': internal, 'DataSource': 'reviewed-source', 'Species': 'Human',
               'Parameters': [{'Name': parameter, 'Value': .1, **({} if parameter == 'GFR fraction' else {'Unit': unit})}]}
    if internal in {'rCYP450_FirstOrder', 'MetabolizationLiverMicrosomes_FirstOrder'}:
        process['Molecule'] = 'UGT1A1'
    payload['instance']['compound']['Processes'] = [process]
    spec = _generic_process_spec(payload, process, parameter, unit, upper=1 if parameter == 'GFR fraction' else 2)
    updated = native_edits.apply_parameter_values(payload, [spec], [value], spec['source'])
    assert updated['instance']['compound']['Processes'][0]['Parameters'][0] == {**process['Parameters'][0], 'Value': value}
    if internal == 'GlomerularFiltration':
        assert spec['runtime_mappings'][0]['runtime_path'] == (
            'Neighborhoods|Kidney_pls_Kidney_ur|ToyDrug|Glomerular Filtration-reviewed-source-ToyDrug|GFR fraction')
    elif internal == 'LiverClearance':
        assert spec['runtime_mappings'][0]['runtime_path'] == 'ToyDrug-Total Hepatic Clearance-reviewed-source|' + parameter


@pytest.mark.parametrize('change', [
    lambda p, s: s.pop('source'),
    lambda p, s: s['native_process'].update(extra_path='Organism|Liver|Volume'),
    lambda p, s: s['native_process'].update(molecule='OtherEnzyme'),
    lambda p, s: s['native_process'].update(internal_name='rCYP450_MM'),
    lambda p, s: s['native_process'].update(parameter_name='Body weight'),
    lambda p, s: s['native_process'].update(data_source='../escape'),
    lambda p, s: s['runtime_mappings'][0].update(runtime_path='OtherDrug|Intrinsic clearance'),
    lambda p, s: s['runtime_mappings'][0].update(multiplier=True),
    lambda p, s: s['runtime_mappings'][0].update(multiplier=float('inf')),
    lambda p, s: s.update(unit='ml/min'),
    lambda p, s: p['instance']['compound']['Processes'][1].update(Species='Rat'),
])
def test_generic_parameter_mapping_rejects_source_identity_path_and_unit_drift(change):
    payload, _ = _cyp3a4_parameter_fixture()
    process = payload['instance']['compound']['Processes'][1]
    spec = _generic_process_spec(payload, process, 'Intrinsic clearance', 'l/min')
    change(payload, spec)
    with pytest.raises(ValueError):
        native_edits.apply_parameter_values(payload, [spec], [.5], payload['instance']['human_source'])


def test_gfr_estimation_rejects_out_of_fraction_domain_and_biliary_is_not_single_path():
    payload, _ = _cyp3a4_parameter_fixture()
    process = payload['instance']['compound']['Processes'][0]
    spec = _generic_process_spec(payload, process, 'GFR fraction', 'dimensionless', upper=2)
    with pytest.raises(ValueError, match='GFR fraction bounds'):
        native_edits.apply_parameter_values(payload, [spec], [.5], spec['source'])
    with pytest.raises(ValueError, match='no reviewed single-parameter'):
        native_edits.reviewed_process_parameter_path('ToyDrug', {
            'internal_name': 'BiliaryClearance', 'molecule': None, 'data_source': 'source', 'parameter_name': 'Plasma clearance'})


def _shared_mm_factor_fixture():
    payload, _ = _cyp3a4_parameter_fixture()
    processes, identities, mappings = [], [], []
    anchors = [.8, 13.7]
    for source, anchor, km in zip(('alpha-OH pathway', '4-OH pathway'), anchors, (.5, 2.), strict=True):
        process = {'InternalName': 'MetabolizationLiverMicrosomes_MM', 'Molecule': 'CYP3A4',
                   'DataSource': source, 'Parameters': [
                       {'Name': 'kcat', 'Value': anchor, 'Unit': '1/min'},
                       {'Name': 'Km', 'Value': km, 'Unit': 'µmol/l'},
                       {'Name': 'In vitro Vmax for liver microsomes', 'Value': .2, 'Unit': 'nmol/min/mg mic. protein'}]}
        identity = {'internal_name': process['InternalName'], 'molecule': 'CYP3A4',
                    'data_source': source, 'parameter_name': 'kcat'}
        processes.append(process); identities.append(identity)
        mappings.append({'runtime_path': native_edits.reviewed_process_parameter_path('ToyDrug', identity),
                         'multiplier': anchor, 'native_unit': '1/min'})
    payload['instance']['compound']['Processes'] = [payload['instance']['compound']['Processes'][0], *processes]
    spec = {'parameter_id': 'shared_process_parameter', 'parameterization': 'shared_factor',
            'unit': 'dimensionless', 'lower_bound': .1, 'upper_bound': 10.,
            'source': deepcopy(payload['instance']['human_source']),
            'native_processes': identities, 'source_anchor_values': anchors, 'runtime_mappings': mappings}
    return payload, spec


def _single_mm_source_context_fixture():
    payload, shared = _shared_mm_factor_fixture()
    payload['instance']['compound']['Processes'].pop()
    identity = deepcopy(shared['native_processes'][0])
    spec = {'parameter_id': 'single_process_kcat', 'unit': '1/min',
            'lower_bound': .1, 'upper_bound': 10., 'source': shared['source'],
            'native_process': identity, 'runtime_mappings': [
                {'runtime_path': native_edits.reviewed_process_parameter_path('ToyDrug', identity),
                 'multiplier': 1., 'native_unit': '1/min'}]}
    return payload, spec


@pytest.mark.parametrize('explicit_species', [False, True])
@pytest.mark.parametrize('action', list(native_edits.PARTITION_ACTIONS))
def test_single_mm_capacity_rebuild_binds_missing_species_without_inventing_it(bound_methods, action, explicit_species):
    payload, spec = _single_mm_source_context_fixture()
    if explicit_species:
        payload['instance']['compound']['Processes'][1]['Species'] = 'Human'
    original = deepcopy(payload)
    candidate = native_edits.compile_candidate(payload, action, project_root=PROJECT_ROOT, parameter_specs=[spec])
    updated = native_edits.apply_parameter_values(candidate['request'], [spec], [2.], spec['source'])
    rebuilt = native_edits.compile_candidate(updated, action, project_root=PROJECT_ROOT, parameter_specs=[spec])
    expected = deepcopy(candidate['snapshot']['Compounds'][0])
    expected['Processes'][1]['Parameters'][0]['Value'] = 2.
    assert rebuilt['snapshot']['Compounds'][0] == expected
    assert ('Species' in expected['Processes'][1]) is explicit_species
    assert rebuilt['snapshot']['Individuals'] == candidate['snapshot']['Individuals']
    assert rebuilt['snapshot']['Protocols'] == candidate['snapshot']['Protocols']
    assert rebuilt['snapshot']['ExpressionProfiles'] == candidate['snapshot']['ExpressionProfiles']
    assert payload == original


@pytest.mark.parametrize('change', [
    lambda i: i['compound']['Processes'][1].update(Species=None),
    lambda i: i['compound']['Processes'][1].update(Species='Rat'),
    lambda i: i['expression_profiles'].clear(),
    lambda i: i['expression_profiles'].append(deepcopy(i['expression_profiles'][0])),
    lambda i: i['expression_profiles'][0].update(Type='Transporter'),
    lambda i: i['expression_profiles'][0].update(Species='Rat'),
    lambda i: i['individual']['ExpressionProfiles'].clear(),
    lambda i: i['individual']['ExpressionProfiles'].append(i['individual']['ExpressionProfiles'][0]),
    lambda i: i['expression_profile_sources'].clear(),
    lambda i: i['individual']['OriginData'].update(Species='Rat'),
])
def test_single_mm_capacity_cannot_inherit_missing_or_conflicting_human_context(change):
    payload, spec = _single_mm_source_context_fixture()
    change(payload['instance'])
    with pytest.raises(ValueError):
        native_edits.apply_parameter_values(payload, [spec], [2.], spec['source'])


@pytest.mark.parametrize('action', list(native_edits.PARTITION_ACTIONS))
def test_shared_mm_factor_preserves_unequal_branches_km_and_unrelated_native_inputs(bound_methods, action):
    payload, spec = _shared_mm_factor_fixture()
    original = deepcopy(payload)
    candidate = native_edits.compile_candidate(payload, action, project_root=PROJECT_ROOT, parameter_specs=[spec])
    updated = native_edits.apply_parameter_values(candidate['request'], [spec], [2.], spec['source'])
    rebuilt = native_edits.compile_candidate(updated, action, project_root=PROJECT_ROOT, parameter_specs=[spec])
    assert payload == original
    assert rebuilt['snapshot']['Individuals'] == candidate['snapshot']['Individuals']
    assert rebuilt['snapshot']['Protocols'] == candidate['snapshot']['Protocols']
    assert rebuilt['snapshot']['ExpressionProfiles'] == candidate['snapshot']['ExpressionProfiles']
    before = candidate['snapshot']['Compounds'][0]
    after = rebuilt['snapshot']['Compounds'][0]
    expected = deepcopy(before)
    for branch, anchor in zip(expected['Processes'][1:], spec['source_anchor_values'], strict=True):
        branch['Parameters'][0]['Value'] = 2 * anchor
    assert after == expected
    assert [branch['Parameters'][0]['Value'] for branch in after['Processes'][1:]] == [1.6, 27.4]
    assert after['Processes'][1]['Parameters'][1:] == before['Processes'][1]['Parameters'][1:]
    assert rebuilt['semantic_hash'] == candidate['semantic_hash']
    assert rebuilt['parameterization_hash'] != candidate['parameterization_hash']


@pytest.mark.parametrize('change', [
    lambda p, s: s.pop('source'),
    lambda p, s: s.update(parameterization='shared_value'),
    lambda p, s: s.update(unit='1/min'),
    lambda p, s: s.update(lower_bound=-1),
    lambda p, s: s['source_anchor_values'].pop(),
    lambda p, s: s['source_anchor_values'].__setitem__(0, True),
    lambda p, s: s['runtime_mappings'][0].update(multiplier=1),
    lambda p, s: s['runtime_mappings'][0].update(multiplier=True),
    lambda p, s: s['runtime_mappings'][0].update(native_unit='1/h'),
    lambda p, s: s['runtime_mappings'][0].pop('native_unit'),
    lambda p, s: s['runtime_mappings'][0].update(runtime_path='Organism|Liver|Volume'),
    lambda p, s: s['native_processes'][0].update(parameter_name='Km'),
    lambda p, s: s['native_processes'][0].update(internal_name='rCYP450_FirstOrder'),
    lambda p, s: s['native_processes'][0].update(data_source='../escape'),
    lambda p, s: s['native_processes'][0].update(extra='ignored'),
    lambda p, s: p['instance']['compound']['Processes'][1].update(Species='Rat'),
    lambda p, s: p['instance']['compound']['Processes'].append(deepcopy(p['instance']['compound']['Processes'][1])),
    lambda p, s: p['instance']['compound']['Processes'][1]['Parameters'][0].update(Value=.9),
    lambda p, s: p['instance']['compound']['Processes'][1]['Parameters'][0].update(Unit='1/h'),
    lambda p, s: p['instance']['compound']['Processes'][1]['Parameters'][1].update(Value=0),
    lambda p, s: p['instance']['compound']['Processes'][1]['Parameters'][1].update(Unit='mg/l'),
])
def test_shared_mm_factor_rejects_untyped_paths_unit_drift_or_changed_source_anchors(change):
    payload, spec = _shared_mm_factor_fixture()
    change(payload, spec)
    with pytest.raises(ValueError):
        native_edits.apply_parameter_values(payload, [spec], [2.], payload['instance']['human_source'])


def test_shared_mm_factor_cannot_be_applied_cumulatively_or_duplicate_a_branch():
    payload, spec = _shared_mm_factor_fixture()
    updated = native_edits.apply_parameter_values(payload, [spec], [2.], spec['source'])
    with pytest.raises(ValueError, match='original kcat anchors'):
        native_edits.apply_parameter_values(updated, [spec], [2.], spec['source'])
    spec['native_processes'][1] = deepcopy(spec['native_processes'][0])
    spec['runtime_mappings'][1] = deepcopy(spec['runtime_mappings'][0])
    spec['source_anchor_values'][1] = spec['source_anchor_values'][0]
    with pytest.raises(ValueError, match='distinct branches'):
        native_edits.apply_parameter_values(payload, [spec], [2.], spec['source'])


def test_shared_mm_factor_does_not_silently_share_capacity_between_different_enzymes():
    payload, spec = _shared_mm_factor_fixture()
    spec['native_processes'][1]['molecule'] = 'CYP2D6'
    spec['runtime_mappings'][1]['runtime_path'] = native_edits.reviewed_process_parameter_path('ToyDrug', spec['native_processes'][1])
    payload['instance']['compound']['Processes'][2]['Molecule'] = 'CYP2D6'
    with pytest.raises(ValueError, match='one explicitly sourced enzyme'):
        native_edits.apply_parameter_values(payload, [spec], [2.], spec['source'])


@pytest.mark.parametrize('parameter', ['Km', 'Vmax', 'In vitro Vmax for liver microsomes', 'kcat_app'])
def test_mm_runtime_mapping_accepts_only_reviewed_downstream_kcat(parameter):
    with pytest.raises(ValueError, match='Unsupported enzyme parameter'):
        native_edits.reviewed_process_parameter_path('ToyDrug', {
            'internal_name': 'MetabolizationLiverMicrosomes_MM', 'molecule': 'CYP3A4',
            'data_source': 'alpha-OH pathway', 'parameter_name': parameter})


def test_actual_alprazolam_development_requests_preserve_source_factor_contract():
    path = PROJECT_ROOT / 'experiments/blueprint_design/development_inputs/alprazolam_reconstruction_20260920_v2/input_package.json'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'ad846d0dcb10f315a715832c7abcf292a10049cc29f5673864f3679a9a79ff36'
    # The input package contains permitted development values and only sealed
    # external references; this test never opens the external answer documents.
    case = json.loads(raw)['case']
    assert {s['role'] for s in case['scenarios']} == {'calibration', 'selection'}
    spec = next(p for p in case['parameter_search']['parameters'] if p['parameter_id'] == 'shared_process_parameter')
    assert spec['initial_value'] == 1 and spec['unit'] == 'dimensionless'
    assert spec['source_anchor_values'] == [.8066945978, 13.7322820855]
    for scenario in case['scenarios']:
        candidate = native_edits.compile_candidate(scenario['design_request'], 'partition_pksim_standard',
            project_root=PROJECT_ROOT, parameter_specs=case['parameter_search']['parameters'])
        changed = native_edits.apply_parameter_values(candidate['request'], [spec], [1.25], spec['source'])
        rebuilt = native_edits.compile_candidate(changed, 'partition_pksim_standard',
            project_root=PROJECT_ROOT, parameter_specs=case['parameter_search']['parameters'])
        before = candidate['snapshot']['Compounds'][0]['Processes']
        after = rebuilt['snapshot']['Compounds'][0]['Processes']
        expected = deepcopy(before)
        for process in expected:
            if process.get('InternalName') == 'MetabolizationLiverMicrosomes_MM':
                parameter = next(p for p in process['Parameters'] if p['Name'] == 'kcat')
                parameter['Value'] *= 1.25
        assert after == expected
        assert rebuilt['snapshot']['Individuals'] == candidate['snapshot']['Individuals']
        assert rebuilt['snapshot']['Protocols'] == candidate['snapshot']['Protocols']
