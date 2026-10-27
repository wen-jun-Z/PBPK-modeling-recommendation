"""Request-design behavior with synthetic sources; no ODE, GPU, LLM or holdout data."""
from copy import deepcopy
import json
import math
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest

from pbpk_kb.blueprints import experiment_runtime, mechanism_edits, native_design, native_design_policy
from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from pbpk_kb.blueprints.design_diagnostics import BudgetExceeded, BudgetLedger
from pbpk_kb.blueprints.experiment_runtime import digest, file_sha
from test_human_model_binding import _payload


def _source(root, name, content=None, kind='model_asset'):
    path = root / name
    path.write_text(content or 'Synthetic source: ' + name)
    return {'provenance_id': str(uuid5(NAMESPACE_URL, name)), 'provenance_kind': kind,
            'source_id': name, 'source_location': 'synthetic fixture only',
            'source_checksum': file_sha(path)}


@pytest.fixture
def design_inputs(tmp_path, monkeypatch):
    source = _source(tmp_path, 'source.json')
    request_source = _source(tmp_path, 'request.json', kind='user_request')
    request = _payload(oral=True)
    instance = request['instance']
    instance['human_source'] = instance['protocol_source'] = instance['formulation_source'] = source
    instance['compound_sources'] = {key: source for key in instance['compound_sources']}
    instance['compound']['Processes'] = [
        {'InternalName': 'MetabolizationLiverMicrosomes_MM', 'Molecule': enzyme,
         'Species': 'Human', 'DataSource': 'Synthetic', 'Parameters': [
             {'Name': 'kcat', 'Unit': '1/min', 'Value': 2.},
             {'Name': 'Km', 'Unit': 'µmol/l', 'Value': 4.}]} for enzyme in ('CYP3A4', 'CYP2C19')]
    request = BlueprintDesignRequest.model_validate(request).model_dump(mode='json')
    scenarios = []
    for index, role in enumerate(('calibration', 'selection'), start=1):
        study_source = _source(tmp_path, role + '.json')
        development = deepcopy(request)
        development['requirement']['dose'] = float(index * 10)
        next(p for p in development['instance']['protocol']['Parameters']
             if p['Name'] == 'InputDose')['Value'] = float(index * 10)
        development['studies'] = [{'study_id': role, 'role': role,
            'source_id': study_source['source_id'], 'source_checksum': study_source['source_checksum'],
            'cohort_ids': [role], 'observed_target_ids': ['plasma']}]
        scenario = {'scenario_id': role, 'role': role, 'study_group_id': role,
            'design_request': development,
            'compound_id': 'compound:synthetic', 'time_unit': 'min', 'unit': 'µmol/l',
            'observations': [{'time': 1., 'value': .25}, {'time': 2., 'value': .15}],
            'target_output': 'Organism|PeripheralVenousBlood|ToyDrug|Plasma (Peripheral Venous Blood)'}
        original = {key: scenario[key] for key in
                    ('scenario_id', 'study_group_id', 'compound_id', 'role', 'time_unit', 'unit', 'observations')}
        # This intentionally absent pointer must never be followed by the designer.
        original['source_audit'] = 'not-accessible-external-observations.json'
        study_path = tmp_path / study_source['source_id']
        study_path.write_text(json.dumps(original))
        development['studies'][0]['source_checksum'] = file_sha(study_path)
        scenarios.append(scenario)
    actions = list(native_design_policy.PARTITION_ACTIONS)
    actions += [action + '__' + mechanism_edits.ACTION for action in actions]
    specs = [{'parameter_id': name, 'initial_value': 1., 'lower_bound': .1,
              'upper_bound': 2., 'unit': 'dimensionless', 'source': source} for name in ('CL', 'Pint')]
    case = {'purpose': 'historical_development_experiment', 'case_id': 'synthetic',
            'compound_id': 'compound:synthetic', 'scenarios': scenarios,
            'parameter_search': {'parameters': specs},
            'mechanism_search': {'version': 'native-metabolism-search-v1', 'expected_structure_ids': actions}}

    def compile_variant(request, action, *, project_root, parameter_specs=None, **kwargs):
        assert project_root == tmp_path
        return {'structure_id': action, 'semantic_hash': digest(action),
            'edit_count': int('rodgers_rowland' in action) + int(mechanism_edits.ACTION in action),
            'parameter_specs': deepcopy(parameter_specs), 'request': deepcopy(request),
            'snapshot': {'synthetic': True, 'protocol': deepcopy(request['instance']['protocol'])},
            'edit_manifest': {'snapshot_sha256': digest(request)}}

    # Exercise the real candidate-space enumeration; isolate native source compilation only.
    monkeypatch.setattr(mechanism_edits, 'native_candidates', lambda request, **kw:
                        [compile_variant(request, action, **kw) for action in actions])
    monkeypatch.setattr(native_design_policy, '_snapshot_from_design', lambda *args: None)
    monkeypatch.setattr(native_design_policy, 'compile_native_candidate', compile_variant)
    monkeypatch.setattr(native_design, 'compile_native_candidate', compile_variant)
    policy = native_design_policy.NativeDesignPolicy(prediction_request=request).model_dump(mode='json')
    return SimpleNamespace(root=tmp_path, case=case, request=request, policy=policy,
                           source=source, request_source=request_source, actions=actions)


def _plan(data):
    return native_design_policy.build_design_plan(data.case, data.policy, project_root=data.root)


def _purpose_source(data, *, source_action='partition_rodgers_rowland', mutate=None):
    name = data.policy['prediction_request']['instance']['compound']['Name']
    methods = deepcopy(data.policy['prediction_request']['instance']['compound']['CalculationMethods'])
    methods = [native_design_policy.PARTITION_ACTIONS[source_action]
               if item in native_design_policy.PARTITION_ACTIONS.values() else item for item in methods]
    # Only these source model fields are required. No source observations, fitted
    # parameters or identification records are needed to establish the baseline.
    document = {'Compounds': [{'Name': name, 'CalculationMethods': methods}]}
    if mutate:
        mutate(document)
    reference = _source(data.root, 'partition_source.json', json.dumps(document))
    reference['source_location'] = f'Compounds[{name}].CalculationMethods; disclosed reconstruction'
    for request in [s['design_request'] for s in data.case['scenarios']] + [data.policy['prediction_request']]:
        request['instance']['compound_sources']['CalculationMethods'] = deepcopy(reference)
    data.policy['algorithm'] = 'purpose_driven'
    return reference


def _assert_blocked(result, text=None):
    assert result['supported'] is False and result['status'] == 'blocked'
    assert result['candidates'] == [] and result['reasons']
    if text:
        assert text in str(result['reasons'])


def _write_proofs(data, *, ratio=.1):
    proofs = []
    requests = [s['design_request'] for s in data.case['scenarios']] + [data.policy['prediction_request']]
    for index, request in enumerate(requests):
        for branch in mechanism_edits._branches(request['instance']['compound']):
            proof = {'kind': 'local_unbound_substrate_upper_bound', 'compound_id': data.case['compound_id'],
                'design_request_sha256': native_design_policy.design_request_sha256(request),
                'parameter_search_sha256': digest(data.case['parameter_search']['parameters']),
                'bound_scope': 'all_parameter_vectors_in_search_bounds',
                'process_identity': branch['from_identity'],
                'upper_bound_umol_per_l': ratio * branch['km_umol_per_l'],
                'role': 'prior', 'source_refs': [deepcopy(data.source)]}
            name = f'bound_{index}_{branch["from_identity"]["molecule"]}.json'
            (data.root / name).write_text(json.dumps(proof))
            proofs.append({'path': name, 'sha256': file_sha(data.root / name)})
    data.policy.update(kinetics_mode='allow_supported_linearization', linearization_evidence=proofs)
    return proofs


def _mutate_proof(data, mutate):
    ref = data.policy['linearization_evidence'][0]
    path = data.root / ref['path']
    proof = json.loads(path.read_text())
    mutate(proof)
    path.write_text(json.dumps(proof))
    ref['sha256'] = file_sha(path)


def test_constraints_filter_candidates_without_mutating_historical_space(design_inputs):
    data = design_inputs
    original_case = deepcopy(data.case)
    historical = experiment_runtime.candidate_space(data.case, root=data.root)
    assert len(historical) == 16
    assert _plan(data)['candidate_count'] == 6
    data.policy.update(partition_mode='source', max_free_parameters=0)
    plan = _plan(data)
    assert plan['candidate_count'] == 1
    assert plan['candidates'][0]['action'] == 'partition_pksim_standard'
    assert plan['candidates'][0]['free_parameter_ids'] == []
    assert experiment_runtime.candidate_space(data.case, root=data.root) == historical
    assert data.case == original_case
    assert plan['execution'] == {'solver_calls': 0, 'llm_calls': 0}


def test_purpose_audits_real_source_partition_without_importing_source_parameters(design_inputs):
    data = design_inputs
    _purpose_source(data)
    before = deepcopy(data.case)
    plan = _plan(data)
    assert plan['status'] == 'ready'
    assert plan['rule_candidate_id'] == 'partition_rodgers_rowland__00'
    audit = plan['source_structure_audit']
    assert audit['status'] == 'reconstructed_override'
    assert audit['source_partition_action'] == 'partition_rodgers_rowland'
    assert audit['template_partition_action'] == 'partition_pksim_standard'
    assert audit['values_used'] == ['Compounds[ToyDrug].CalculationMethods']
    assert audit['historical_fitted_parameters_imported'] is False
    baselines = plan['baseline_definitions']
    assert baselines['source_structure_initial_parameters']['eligible_candidate_ids'] == ['partition_rodgers_rowland__00']
    assert baselines['reconstructed_template_initial_parameters']['eligible_candidate_ids'] == ['partition_pksim_standard__00']
    assert all(b['is_original_complete_source_model'] is False for b in baselines.values())
    for candidate in plan['candidates']:
        source = candidate['action'] == 'partition_rodgers_rowland'
        assert candidate['matches_source_partition'] is source
        assert candidate['matches_template_partition'] is (not source)
        assert candidate['source_partition_edit_count'] == int(not source)
        assert candidate['partition_origin'] == ('original_source' if source else 'reconstructed_template')
        # Historical edit counts remain relative to the supplied template.
        assert candidate['structure_edit_count'] == int(source)
    assert data.case == before


@pytest.mark.parametrize('mode,expected', [('source', 'partition_rodgers_rowland'),
                                        ('template', 'partition_pksim_standard')])
def test_purpose_distinguishes_source_and_template_constraints(design_inputs, mode, expected):
    data = design_inputs
    _purpose_source(data)
    data.policy.update(partition_mode=mode, max_free_parameters=0)
    plan = _plan(data)
    assert plan['status'] == 'ready'
    assert [candidate['action'] for candidate in plan['candidates']] == [expected]


def test_source_preserved_label_is_separate_from_reconstructed_override(design_inputs):
    data = design_inputs
    _purpose_source(data, source_action='partition_pksim_standard')
    assert _plan(data)['source_structure_audit']['status'] == 'source_structure_preserved'


def test_purpose_joint_candidates_respect_explicit_parameter_limits(design_inputs):
    data = design_inputs
    _purpose_source(data)
    data.policy['max_free_parameters'] = 2
    plan = _plan(data)
    assert plan['candidate_count'] == 8
    assert sum(len(c['free_parameter_ids']) == 2 for c in plan['candidates']) == 2
    data.policy['parameter_modes'] = {'CL': 'fixed'}
    assert all('CL' not in c['free_parameter_ids'] for c in _plan(data)['candidates'])


@pytest.mark.parametrize('change,reason', [
    ('duplicate_name', 'exactly one matching'),
    ('wrong_name', 'exactly one matching'),
    ('missing_array', 'Compounds array'),
    ('missing_methods', 'CalculationMethods'),
    ('duplicate_methods', 'unique nonempty'),
    ('unknown_partition', 'registered source'),
    ('nonpartition_override', 'Nonpartition calculation'),
])
def test_purpose_source_ambiguity_or_unsupported_override_blocks(design_inputs, change, reason):
    def mutate(document):
        compound = document['Compounds'][0]
        if change == 'duplicate_name':
            document['Compounds'].append(deepcopy(compound))
        elif change == 'wrong_name':
            compound['Name'] = 'Another drug'
        elif change == 'missing_array':
            document.pop('Compounds')
        elif change == 'missing_methods':
            compound.pop('CalculationMethods')
        elif change == 'duplicate_methods':
            compound['CalculationMethods'].append(compound['CalculationMethods'][0])
        elif change == 'unknown_partition':
            compound['CalculationMethods'][0] = 'Cellular partition coefficient method - Unregistered'
        else:
            compound['CalculationMethods'].append('Unverified alternative permeability')
    _purpose_source(design_inputs, mutate=mutate)
    _assert_blocked(_plan(design_inputs), reason)


@pytest.mark.parametrize('change,reason', [
    ('location', 'exact compound CalculationMethods'),
    ('non_json', 'verifiable JSON model'),
    ('checksum', 'checksum mismatch'),
    ('oversized', 'exceeds 8 MiB'),
])
def test_purpose_source_reference_is_verified_and_bounded(design_inputs, change, reason):
    data = design_inputs
    reference = _purpose_source(data)
    path = data.root / reference['source_id']
    if change == 'location':
        reference['source_location'] = 'Compounds[Another drug].CalculationMethods'
    elif change == 'non_json':
        path.write_text('An unstructured source is insufficient')
        reference['source_checksum'] = file_sha(path)
    elif change == 'checksum':
        path.write_text('{}')
    else:
        # Sparse temporary fixture exercises the input cap without a large write.
        with path.open('wb') as stream:
            stream.truncate(8 * 1024 ** 2 + 1)
        reference['source_checksum'] = file_sha(path)
    for request in [s['design_request'] for s in data.case['scenarios']] + [data.policy['prediction_request']]:
        request['instance']['compound_sources']['CalculationMethods'] = deepcopy(reference)
    _assert_blocked(_plan(data), reason)


def test_purpose_requires_verifiable_source_while_legacy_accepts_legacy_fixture(design_inputs):
    data = design_inputs
    assert _plan(data)['status'] == 'ready'
    data.policy['algorithm'] = 'purpose_driven'
    _assert_blocked(_plan(data), 'source location')


def test_purpose_source_semantics_do_not_rebind_linearization_evidence(design_inputs):
    data = design_inputs
    _purpose_source(data)
    _write_proofs(data)
    plan = _plan(data)
    linearized = [c for c in plan['candidates'] if mechanism_edits.ACTION in c['action']]
    assert linearized
    assert all(c['action'].startswith('partition_pksim_standard__') for c in linearized)
    data.policy['partition_mode'] = 'source'
    plan = _plan(data)
    assert plan['status'] == 'ready'
    assert all(c['action'] == 'partition_rodgers_rowland' for c in plan['candidates'])


@pytest.mark.parametrize('objective', [{'selection_tolerance': .1}, {'max_auc_fold_error': 2.},
    {'primary': 'observed_window_auc', 'max_curve_log_rmse': .5}])
def test_legacy_cannot_silently_ignore_requested_objective(design_inputs, objective):
    data = design_inputs
    data.policy['objective'] = objective
    with pytest.raises(ValueError, match='nondefault objective'):
        native_design_policy.NativeDesignPolicy.model_validate(data.policy)
    _assert_blocked(_plan(data))
    _purpose_source(data)
    plan = _plan(data)
    assert plan['status'] == 'ready'
    assert all(plan['objective'][key] == value for key, value in objective.items())


def test_fixed_and_required_estimates_really_constrain_parameter_sets(design_inputs):
    data = design_inputs
    data.policy.update(parameter_modes={'CL': 'fixed', 'Pint': 'estimate'})
    plan = _plan(data)
    assert plan['candidate_count'] == 2
    assert all(c['free_parameter_ids'] == ['Pint'] for c in plan['candidates'])
    assert plan['guidance_informative'] is False
    assert {(c['requirement_score'], c['evidence_score']) for c in plan['candidates']} == {(1., 1.)}
    assert all(c['design_decision']['score_semantics'] == 'binary_hard_constraint_pass'
               for c in plan['candidates'])
    data.policy['max_free_parameters'] = 0
    _assert_blocked(_plan(data), 'exceed')


@pytest.mark.parametrize('change', ['target', 'population', 'compound', 'source', 'human', 'study_role'])
def test_unsupported_scope_and_inconsistent_template_block(design_inputs, change):
    data = design_inputs
    request = data.policy['prediction_request']
    if change == 'target':
        request['requirement']['targets'][0]['matrix'] = 'liver'
    elif change == 'population':
        request['requirement']['population'] = request['instance']['population_context'] = 'renal_impairment'
    elif change == 'compound':
        request['requirement']['compound_semantic_id'] = 'compound:other'
    elif change == 'source':
        request['instance']['compound_sources']['Processes']['source_location'] = 'different drug source'
    elif change == 'human':
        request['human_model']['definition_checksum'] = 'f' * 64
    else:
        data.case['scenarios'][1]['design_request']['studies'][0]['role'] = 'calibration'
    _assert_blocked(_plan(data))


@pytest.mark.parametrize('mode', ['retain_source', 'require_saturable', 'allow_supported_linearization'])
def test_linearization_is_never_enabled_without_evidence(design_inputs, mode):
    data = design_inputs
    data.policy['kinetics_mode'] = mode
    plan = _plan(data)
    assert plan['supported']
    assert all(mechanism_edits.ACTION not in c['action'] for c in plan['candidates'])


def test_require_saturable_cannot_invent_an_absent_source_mechanism(design_inputs):
    data = design_inputs
    for request in [s['design_request'] for s in data.case['scenarios']] + [data.policy['prediction_request']]:
        request['instance']['compound']['Processes'] = []
    data.policy['kinetics_mode'] = 'require_saturable'
    _assert_blocked(_plan(data), 'no supported source MM')


@pytest.mark.parametrize('ratio,enabled', [(.1, True), (.100001, False)])
def test_local_bound_threshold_requires_every_scenario_and_branch(design_inputs, ratio, enabled):
    data = design_inputs
    _write_proofs(data, ratio=ratio)
    plan = _plan(data)
    assert plan['supported'] and len(plan['linearization_checks']) == 6
    assert any(mechanism_edits.ACTION in c['action'] for c in plan['candidates']) is enabled
    # The proof binds source partition; it cannot license a changed tissue partition law.
    assert plan['candidate_count'] == (9 if enabled else 6)
    assert not any('rodgers_rowland__' + mechanism_edits.ACTION in c['action']
                   for c in plan['candidates'])
    data.policy['linearization_evidence'].pop()
    assert all(mechanism_edits.ACTION not in c['action'] for c in _plan(data)['candidates'])


@pytest.mark.parametrize('change', ['plasma', 'unit', 'scenario', 'branch', 'search_bounds',
                                  'compound', 'scope', 'calibration_source', 'selection_source'])
def test_invalid_linearization_proofs_are_rejected(design_inputs, change):
    data = design_inputs
    _write_proofs(data)
    def mutate(proof):
        if change == 'plasma':
            proof['kind'] = 'plasma_concentration_upper_bound'
        elif change == 'unit':
            proof['unit'] = 'mg/L'
        elif change == 'scenario':
            proof['design_request_sha256'] = 'f' * 64
        elif change == 'branch':
            proof['process_identity']['molecule'] = 'CYP1A2'
        elif change == 'search_bounds':
            proof['parameter_search_sha256'] = 'f' * 64
        elif change == 'compound':
            proof['compound_id'] = 'compound:other'
        elif change == 'scope':
            proof['bound_scope'] = 'source_parameter_point_only'
        elif change == 'calibration_source':
            proof.update(role='calibration', study_id='calibration')
        else:
            study = data.case['scenarios'][1]['design_request']['studies'][0]
            proof['source_refs'][0].update(source_id=study['source_id'], source_checksum=study['source_checksum'])
    _mutate_proof(data, mutate)
    _assert_blocked(_plan(data))


@pytest.mark.parametrize('tamper', ['proof', 'underlying_source'])
def test_proof_and_underlying_source_checksums_are_verified(design_inputs, tamper):
    data = design_inputs
    _write_proofs(data)
    name = data.policy['linearization_evidence'][0]['path'] if tamper == 'proof' else data.source['source_id']
    with (data.root / name).open('a') as stream:
        stream.write(' tampered')
    _assert_blocked(_plan(data), 'checksum mismatch')


def test_forbidden_source_alias_is_rejected_by_checksum(design_inputs):
    data = design_inputs
    _write_proofs(data)
    data.case['forbidden_source_checksums'] = [data.source['source_checksum']]
    _assert_blocked(_plan(data), 'forbidden source')


def test_source_path_traversal_is_rejected_without_reading_target(design_inputs):
    data = design_inputs
    data.case['parameter_search']['parameters'][0]['source'] = dict(data.source, source_id='../outside.json')
    _assert_blocked(_plan(data), 'project-relative')


@pytest.mark.parametrize('route,infusion,application', [
    ('oral', 0., 'Oral'), ('intravenous', 0., 'IntravenousBolus'), ('intravenous', 30., 'Intravenous')])
def test_template_request_materializes_dose_route_infusion_and_preserves_drug(design_inputs, route, infusion, application):
    data = design_inputs
    before = deepcopy(data.case)
    specification = {'template_scenario_id': 'calibration', 'route': route, 'dose': 40.,
                     'dose_unit': 'mg', 'infusion_minutes': infusion, 'partition_mode': 'source',
                     'max_free_parameters': 0}
    if route == 'oral':
        specification.update(water_ml_per_kg=3., formulation='solution')
    policy = native_design.policy_from_template(data.case, specification, request_source=data.request_source)
    request = policy.prediction_request.model_dump(mode='json')
    protocol = request['instance']['protocol']
    assert protocol['ApplicationType'] == application
    assert next(p['Value'] for p in protocol['Parameters'] if p['Name'] == 'InputDose') == 40.
    assert request['requirement']['dose'] == 40. and request['studies'] == []
    assert request['instance']['compound'] == before['scenarios'][0]['design_request']['instance']['compound']
    assert request['instance']['individual'] == before['scenarios'][0]['design_request']['instance']['individual']
    assert request['instance']['protocol_source'] == data.request_source
    assert data.case == before
    assert native_design_policy.build_design_plan(data.case, policy, project_root=data.root)['candidate_count'] == 1


def _fit_result(candidate, loss=.2):
    return {'status': 'completed', 'selection_log_rmse': loss, 'calibration_log_rmse': .001,
            'solver_calls': 1, 'effective_free_parameter_ids': candidate['free_parameter_ids'],
            'fitted_values': [1., 1.]}


def test_comparison_uses_selection_loss_then_complexity_not_calibration_or_guidance(design_inputs):
    plan = _plan(design_inputs)
    seen = []
    def evaluate(candidate, *, method_config):
        seen.append(candidate['candidate_id'])
        assert method_config['fixed_parameterization']
        result = _fit_result(candidate, .20 if not candidate['free_parameter_ids'] else .19)
        result['calibration_log_rmse'] = 100. if not candidate['free_parameter_ids'] else .0001
        return result
    result = native_design.compare_candidates(plan, evaluate, solver_budget=100)
    assert result['status'] == 'completed' and result['enumeration_complete']
    assert len(seen) == len(plan['candidates']) == 6
    assert result['selected_candidate']['free_parameter_ids'] == []
    assert result['selected_candidate']['structure_edit_count'] == 0
    assert result['scientific_validation_performed'] is False


@pytest.mark.parametrize('failure', ['budget', 'exception'])
def test_comparison_stops_after_unknown_cost_or_budget_failure(design_inputs, failure):
    seen = []
    def evaluate(candidate, *, method_config):
        seen.append(candidate['candidate_id'])
        if failure == 'budget':
            raise BudgetExceeded('search', 1, 0)
        raise RuntimeError('synthetic fitter error')
    result = native_design.compare_candidates(_plan(design_inputs), evaluate, solver_budget=100)
    assert result['status'] == 'failed' and len(seen) == 1
    assert result['selected_candidate'] is None and not result['enumeration_complete']
    assert not result['solver_cost_complete']


def test_comparison_allocates_equal_budgets_and_continues_known_candidate_truncation(design_inputs):
    plan = _plan(design_inputs)
    seen = []
    def evaluate(candidate, *, method_config):
        seen.append(method_config['remaining_solver_calls'])
        result = _fit_result(candidate)
        if len(seen) == 1:
            result.update(status='budget_exhausted', prediction_available=True,
                          solver_calls=method_config['remaining_solver_calls'])
        return result
    result = native_design.compare_candidates(plan, evaluate, solver_budget=100)
    assert seen == [16] * 6
    assert result['solver_calls'] == 21 and result['unallocated_remainder'] == 4
    assert result['enumeration_complete'] and not result['search_complete']
    assert result['status'] == 'budget_exhausted' and result['selected_candidate'] is not None


def test_template_study_linked_evidence_is_not_relabelled_as_prediction_prior(design_inputs):
    data = design_inputs
    request = data.case['scenarios'][0]['design_request']
    study = request['studies'][0]
    source = dict(data.source, source_id=study['source_id'], source_checksum=study['source_checksum'])
    request['evidence'] = [{'evidence_id': 'calibration-mechanism', 'kind': 'mechanism',
        'compound_semantic_id': data.case['compound_id'], 'mechanism': 'metabolism',
        'study_id': study['study_id'], 'provenance': source}]
    before = deepcopy(data.case)
    policy = native_design.policy_from_template(data.case,
        {'template_scenario_id': 'calibration', 'route': 'oral', 'dose': 20., 'dose_unit': 'mg',
         'water_ml_per_kg': 3.}, request_source=data.request_source)
    assert policy.prediction_request.studies == []
    assert all(e.study_id is None for e in policy.prediction_request.evidence)
    assert data.case == before and request['evidence'][0]['study_id'] == 'calibration'


def test_planning_compile_failure_is_persisted_as_failed(design_inputs, monkeypatch):
    data = design_inputs
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic compile failure')
    monkeypatch.setattr(native_design, '_compile', fail)
    result = native_design.run_native_design(case=data.case, policy=data.policy,
        output='compile_failed', project_root=data.root)
    assert result['status'] == 'failed' and result['model_executed'] is False
    assert json.loads((data.root / 'compile_failed/progress.json').read_text())['status'] == 'failed'
    assert json.loads((data.root / 'compile_failed/result.json').read_text())['status'] == 'failed'


def test_default_workflow_only_plans_and_never_constructs_experiment(design_inputs, monkeypatch):
    data = design_inputs
    monkeypatch.setattr(native_design, 'NativeExperiment', lambda **kwargs: pytest.fail('no fit was authorized'))
    monkeypatch.setattr(native_design, 'execute_snapshot', lambda **kwargs: pytest.fail('no solver was authorized'))
    result = native_design.run_native_design(case=data.case, policy=data.policy,
        output='planned', project_root=data.root)
    assert result['status'] == 'planned' and result['stage'] == 'planning_complete'
    assert result['solver_calls'] == 0 and result['model_executed'] is False
    assert result['scientific_validated'] is False and result['external_observations_read'] is False
    assert (data.root / 'planned/draft_snapshot.json').is_file()
    assert not (data.root / 'planned/fit').exists()


def test_blocked_workflow_persists_reason_without_solver(design_inputs, monkeypatch):
    data = design_inputs
    data.policy['prediction_request']['requirement']['targets'][0]['matrix'] = 'liver'
    monkeypatch.setattr(native_design, 'NativeExperiment', lambda **kwargs: pytest.fail('blocked request must not run'))
    result = native_design.run_native_design(case=data.case, policy=data.policy, execute=True,
        output='blocked', project_root=data.root)
    assert result['status'] == 'blocked' and result['stage'] == 'request_not_supported'
    assert json.loads((data.root / 'blocked/result.json').read_text())['reasons']


@pytest.mark.parametrize('failure,expected', [('budget', 'budget_exhausted'), ('exception', 'failed'),
                                          ('interrupt', 'interrupted')])
def test_execution_failure_closes_fitter_and_persists_status(design_inputs, monkeypatch, failure, expected):
    data = design_inputs
    closed = []
    class FakeExperiment:
        def __init__(self, **kwargs):
            pass
        def evaluator(self, context):
            if failure == 'budget':
                raise BudgetExceeded('search', 1, 0)
            if failure == 'interrupt':
                raise KeyboardInterrupt()
            raise RuntimeError('synthetic failure')
        def close(self):
            closed.append(True)
    monkeypatch.setattr(native_design, 'NativeExperiment', FakeExperiment)
    monkeypatch.setattr(native_design, 'execute_snapshot', lambda **kwargs: pytest.fail('failed fit must not predict'))
    result = native_design.run_native_design(case=data.case, policy=data.policy,
        output='failed', project_root=data.root, execute=True)
    assert result['status'] == expected and result['model_executed'] is False
    assert result['stage'] == 'incomplete_delivery' and closed == [True]
    assert json.loads((data.root / 'failed/progress.json').read_text())['status'] == expected
    assert json.loads((data.root / 'failed/result.json').read_text())['status'] == expected


def test_development_sources_are_verified_without_following_external_audit_pointer(design_inputs):
    data = design_inputs
    verified = native_design.verify_development_observations(data.case, data.root)
    assert [row['role'] for row in verified] == ['calibration', 'selection']
    assert all(row['sha256'] == file_sha(data.root / row['source_id']) for row in verified)


@pytest.mark.parametrize('field', ['scenario_id', 'study_group_id', 'compound_id', 'role',
                                 'time_unit', 'unit', 'observations'])
def test_development_source_fields_cannot_be_changed_in_memory(design_inputs, field):
    data = design_inputs
    scenario = data.case['scenarios'][0]
    scenario[field] = [{'time': 1., 'value': 100.}] if field == 'observations' else 'altered'
    with pytest.raises(ValueError):
        native_design.verify_development_observations(data.case, data.root)


@pytest.mark.parametrize('tamper', ['checksum', 'parent_path', 'duplicate_study', 'missing_study'])
def test_development_source_identity_failures_block_before_native_dispatch(design_inputs, tamper):
    data = design_inputs
    studies = data.case['scenarios'][0]['design_request']['studies']
    if tamper == 'checksum':
        (data.root / studies[0]['source_id']).write_text('{}')
    elif tamper == 'parent_path':
        studies[0]['source_id'] = '../outside.json'
    elif tamper == 'duplicate_study':
        studies.append(deepcopy(studies[0]))
    else:
        studies.clear()
    with pytest.raises(ValueError):
        native_design.verify_development_observations(data.case, data.root)


def test_default_plan_does_not_read_development_observations(design_inputs, monkeypatch):
    data = design_inputs
    monkeypatch.setattr(native_design, 'verify_development_observations',
        lambda *args: pytest.fail('planning must not read calibration/selection observations'))
    result = native_design.run_native_design(case=data.case, policy=data.policy,
        output='no_observation_read', project_root=data.root)
    assert result['status'] == 'planned'


@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('algorithm,evaluation_fit_strategy', [
    ('finite_comparison', None), ('purpose_driven', None),
    ('purpose_driven', 'residual'), ('purpose_driven', 'coordinate'),
    ('purpose_driven', 'screened_residual'), ('study_consistent', None),
    ('study_consistent', 'residual')])
def test_fake_execution_delivers_prediction_and_preserves_partial_search_status(
        design_inputs, monkeypatch, partial, algorithm, evaluation_fit_strategy):
    data = design_inputs
    purpose_driven = algorithm in {'purpose_driven', 'study_consistent'}
    consistent = algorithm == 'study_consistent'
    if purpose_driven:
        _purpose_scoring_case(data, max_free=1)
        data.policy['algorithm'] = algorithm
        data.policy['objective'] = {'primary': 'observed_window_auc', 'max_curve_log_rmse': .1}
    expected_plan = _plan(data)
    expected_fit_strategy = evaluation_fit_strategy or ('screened_residual' if consistent else 'residual' if purpose_driven else 'coordinate')
    reference_algorithm = native_design.STUDY_ALGORITHM if consistent else native_design.PURPOSE_ALGORITHM
    suffix = ('-coordinate-control' if evaluation_fit_strategy == 'coordinate' else
              '-screened-control' if evaluation_fit_strategy == 'screened_residual' and not consistent else
              '-source-start-control' if evaluation_fit_strategy == 'residual' and consistent else '')
    expected_algorithm = reference_algorithm + suffix
    events, allocations, progress_during_fit = [], [], []
    output = data.root / 'fake_execution'
    monkeypatch.setattr(native_design, 'verify_development_observations', lambda *args: [])

    class FakeExperiment:
        ids = ['CL', 'Pint']

        def __init__(self, **kwargs):
            assert kwargs['fit_strategy'] == expected_fit_strategy
            assert kwargs['fit_start_strategy'] == 'source_anchor'
            assert kwargs['require_identifiability'] is purpose_driven
            self.progress = kwargs['progress_callback']
            events.append('initialize')

        def evaluator(self, context):
            def evaluate(candidate, *, method_config):
                events.append('evaluate')
                allocations.append(method_config['remaining_solver_calls'])
                # A candidate finishing must not announce global workflow completion.
                self.progress(status='completed', stage='candidate_complete',
                              candidate_id=candidate['candidate_id'])
                progress_during_fit.append(json.loads((output / 'progress.json').read_text()))
                if purpose_driven:
                    count = len(candidate['free_parameter_ids'])
                    # True synthetic curve scoring drives goal filtering and
                    # selection. The fitted candidate meets the request; the
                    # fixed candidate fails its explicit curve-error bound.
                    result = _purpose_fit_result(candidate, data.case,
                        [.25, .15] if count else [.35, .25],
                        diagnostic={'status': 'evaluated', 'step_stability_passed': True,
                                    'rank': count, 'local_not_global': True})
                else:
                    result = _fit_result(candidate)
                calls = method_config['remaining_solver_calls'] if partial and len(allocations) == 1 else 1
                context['ledger'].consume(calls, phase='search')
                if partial and len(allocations) == 1:
                    result.update(status='budget_exhausted', prediction_available=True, solver_calls=calls)
                return result
            return evaluate

        def rebuild_selected(self, candidate, fitted, context):
            events.append('rebuild')
            context['ledger'].consume(1, phase='diagnostics')
            (context['output'] / 'selected_parameters.json').write_text(json.dumps(fitted['fitted_values']))
            return [{'accepted': True, 'synthetic': True}]

        def bounded_diagnose(self, candidate, point, free_ids, context, *, reserve_calls, phase):
            assert reserve_calls == 3 and phase == 'diagnostics'
            if purpose_driven:
                assert len(free_ids) == 1
                return {'status': 'evaluated', 'rank': len(free_ids),
                        'step_stability_passed': True, 'local_not_global': True}
            return {'status': 'unavailable', 'reason': 'synthetic_fixture_no_identifiability_claim'}

        def close(self):
            events.append('close')

    def execute(*, snapshot, output_dir, **kwargs):
        events.append('predict')
        output_dir.mkdir()
        target = data.case['scenarios'][0]['target_output']
        (output_dir / 'curve.csv').write_text('Time [min],' + target + ' [µmol/l]\n0,0\n1,0.2\n2,0.1\n')
        (output_dir / 'model.pkml').write_text('<synthetic-test-fixture/>')
        return {'engineering_accepted': True, 'curve_outputs': [{'path': 'curve.csv'}],
                'pkml_outputs': [{'path': 'model.pkml'}]}

    monkeypatch.setattr(native_design, 'NativeExperiment', FakeExperiment)
    monkeypatch.setattr(native_design, 'execute_snapshot', execute)
    result = native_design.run_native_design(case=data.case, policy=data.policy,
        output=output, project_root=data.root, execute=True, max_solver_calls=100, device='cpu',
        evaluation_fit_strategy=evaluation_fit_strategy)
    assert result['status'] == ('budget_exhausted' if partial else 'completed')
    assert result['stage'] == 'design_complete' and result['delivery_complete'] and result['model_executed']
    assert result['all_candidates_visited'] and result['search_complete'] is (not partial)
    assert result['scientific_validated'] is False and result['external_observations_read'] is False
    assert result['parameter_diagnostics']['status'] == ('evaluated' if purpose_driven else 'unavailable')
    configuration = json.loads((output / 'configuration.json').read_text())
    assert json.loads((output / 'design_plan.json').read_text()) == expected_plan
    assert configuration['evaluation_fit_strategy'] == evaluation_fit_strategy
    if evaluation_fit_strategy is None:
        assert configuration['evaluation_override'] is None
    else:
        assert configuration['evaluation_override'] == {
            'kind': 'optimizer_only', 'reference_algorithm': reference_algorithm,
            'requested_fit_strategy': evaluation_fit_strategy, 'is_legacy_algorithm': False}
    if purpose_driven:
        assert result['algorithm'] == expected_algorithm
        assert configuration['fit_strategy'] == expected_fit_strategy and configuration['require_identifiability'] is True
        assert configuration['selection_rule'].endswith('Meet requested PK bounds and local rank diagnostics; '
            'Pareto errors/complexity; primary objective then simpler model within declared tolerance')
        assert configuration['study_consistency_required'] is consistent
        if consistent:
            assert result['selection_consistency']['status'] == 'applied'
            assert result['selected_candidate_id'] in result['selection_consistency']['eligible_candidate_ids']
            assert (output / 'study_consistency.json').is_file()
        assert configuration['objective']['primary'] == 'observed_window_auc'
        assert len(result['free_parameter_ids']) == 1
        assessment = result['selection_objective']
        assert assessment['valid'] and assessment['constraints_passed']
        assert assessment['primary_error'] == assessment['metrics']['auc_log_error'] == pytest.approx(0.)
        assert assessment['metrics']['curve_log_rmse'] == pytest.approx(0.)
        assert assessment['metrics']['auc_fold_error'] == pytest.approx(1.)
    else:
        assert configuration['fit_strategy'] == 'coordinate' and configuration['require_identifiability'] is False
        assert result['selection_objective'] is None
    assert allocations == [4, 4, 18, 18, 18, 18]
    assert all(state['status'] == 'running' for state in progress_during_fit)
    assert events[-3:] == ['rebuild', 'predict', 'close']
    assert (data.root / result['native_model']).is_file()
    curve = json.loads((data.root / result['prediction_file']).read_text())
    assert curve['time_min'] == [0., 1., 2.] and curve['concentration_umol_per_l'] == [0., .2, .1]
    assert result['budget']['totals']['logical_calls'] == (11 if partial else 8)
    assert json.loads((output / 'result.json').read_text()) == result


@pytest.mark.parametrize('evaluation_fit_strategy', ['coordinate', 'residual'])
@pytest.mark.parametrize('algorithm,execute', [
    ('finite_comparison', True), ('finite_comparison', False), ('purpose_driven', False)])
def test_evaluation_optimizer_override_rejects_nonpurpose_execution_before_output(
        design_inputs, monkeypatch, evaluation_fit_strategy, algorithm, execute):
    data = design_inputs
    data.policy['algorithm'] = algorithm
    monkeypatch.setattr(native_design, 'build_design_plan',
        lambda *args, **kwargs: pytest.fail('Invalid override must fail before planning'))
    output = data.root / 'invalid_override'
    with pytest.raises(ValueError, match='requires purpose_driven execution'):
        native_design.run_native_design(case=data.case, policy=data.policy, output=output,
            project_root=data.root, execute=execute, evaluation_fit_strategy=evaluation_fit_strategy)
    assert not output.exists()


@pytest.mark.parametrize('evaluation_fit_strategy', ['', 'unknown', True, 1, [], {}])
def test_evaluation_optimizer_override_rejects_invalid_values_before_output(
        design_inputs, monkeypatch, evaluation_fit_strategy):
    data = design_inputs
    data.policy['algorithm'] = 'purpose_driven'
    monkeypatch.setattr(native_design, 'build_design_plan',
        lambda *args, **kwargs: pytest.fail('Invalid override must fail before planning'))
    output = data.root / 'invalid_override'
    with pytest.raises(ValueError, match='Evaluation fit strategy must'):
        native_design.run_native_design(case=data.case, policy=data.policy, output=output,
            project_root=data.root, execute=True, evaluation_fit_strategy=evaluation_fit_strategy)
    assert not output.exists()


@pytest.mark.parametrize('evaluation_fit_strategy', [None, 'coordinate', 'residual'])
def test_purpose_equal_budget_records_same_source_start_and_diagnostics_for_optimizer_control(
        design_inputs, monkeypatch, evaluation_fit_strategy):
    data = design_inputs
    _purpose_scoring_case(data, max_free=1)
    captured = []
    class FakeExperiment:
        def __init__(self, **kwargs):
            captured.append(kwargs)
        def evaluator(self, context):
            raise RuntimeError('Synthetic stop before numerical fitting')
        def close(self):
            pass
    monkeypatch.setattr(native_design, 'NativeExperiment', FakeExperiment)
    monkeypatch.setattr(native_design, 'verify_development_observations', lambda *args: [])
    output = data.root / 'equal_budget_control'
    result = native_design.run_native_design(case=data.case, policy=data.policy,
        output=output, project_root=data.root, execute=True, budget_strategy='equal',
        evaluation_fit_strategy=evaluation_fit_strategy)
    assert result['status'] == 'failed'
    assert len(captured) == 1
    configuration = json.loads((output / 'configuration.json').read_text())
    assert captured[0]['fit_start_strategy'] == configuration['fit_start_strategy'] == 'source_anchor'
    assert captured[0]['fit_strategy'] == configuration['fit_strategy'] == (evaluation_fit_strategy or 'residual')
    assert captured[0]['require_identifiability'] is configuration['require_identifiability'] is True
    assert configuration['budget_strategy'] == 'equal'
    assert configuration['max_solver_calls'] == 500


def test_default_draft_materializes_initial_values_with_verified_case_provenance(design_inputs, monkeypatch):
    data = design_inputs
    data.case['parameter_search']['parameters'][0]['initial_value'] = 1.3
    data.case['parameter_search']['parameters'][1]['initial_value'] = .7
    captured = []
    compile_original = native_design._compile

    def capture(request, candidate, specs, root, *, values=None, provenance=None):
        captured.append((deepcopy(values), deepcopy(provenance)))
        return compile_original(request, candidate, specs, root, values=values, provenance=provenance)

    monkeypatch.setattr(native_design, '_compile', capture)
    result = native_design.run_native_design(case=data.case, policy=data.policy,
        output='initial_draft', project_root=data.root)
    assert result['status'] == 'planned' and result['draft_is_uncalibrated']
    assert len(captured) == 1 and captured[0][0] == [1.3, .7]
    source = captured[0][1]
    source_file = data.root / source['source_id']
    assert source_file == data.root / 'initial_draft/development_case.json'
    assert source['provenance_kind'] == 'model_asset'
    assert source['source_checksum'] == file_sha(source_file)
    assert json.loads(source_file.read_text()) == data.case
    assert 'initial_value' in source['source_location']


def test_total_output_limit_stops_after_first_candidate_without_further_solver_calls(design_inputs, monkeypatch):
    data = design_inputs
    output = data.root / 'output_limit'
    sparse = output / 'sparse-test-output.bin'
    events = []
    limit = 128 * 1024**2
    monkeypatch.setattr(native_design, 'verify_development_observations', lambda *args: [])

    class FakeExperiment:
        def __init__(self, **kwargs):
            pass

        def evaluator(self, context):
            def evaluate(candidate, *, method_config):
                assert events == [], 'A second candidate must never execute after the output limit'
                events.append('evaluate')
                context['ledger'].consume(1, phase='search')
                with sparse.open('wb') as stream:
                    stream.truncate(limit + 1)
                assert sparse.stat().st_blocks * 512 < 1024**2
                return _fit_result(candidate)
            return evaluate

        def close(self):
            events.append('close')

    monkeypatch.setattr(native_design, 'NativeExperiment', FakeExperiment)
    monkeypatch.setattr(native_design, 'execute_snapshot',
        lambda **kwargs: pytest.fail('The output limit must block the final prediction solver'))
    try:
        result = native_design.run_native_design(case=data.case, policy=data.policy,
            output=output, project_root=data.root, execute=True, max_output_bytes=limit)
        assert result['status'] == 'failed' and 'output limit exceeded' in result['reason']
        assert result['model_executed'] is False and result['delivery_complete'] is False
        assert events == ['evaluate', 'close']
        assert result['budget']['totals']['logical_calls'] == 1
        assert not (output / 'prediction').exists()
        assert json.loads((output / 'result.json').read_text())['status'] == 'failed'
    finally:
        sparse.unlink(missing_ok=True)


@pytest.mark.parametrize('loss_order', ['fixed_better', 'fitted_better'])
def test_fit_priority_budget_allocation_never_depends_on_observed_losses(design_inputs, loss_order):
    plan = _plan(design_inputs)
    seen = []

    def evaluate(candidate, *, method_config):
        fixed = not candidate['free_parameter_ids']
        seen.append((fixed, method_config['remaining_solver_calls']))
        loss = .1 if fixed == (loss_order == 'fixed_better') else 1.
        return _fit_result(candidate, loss)

    result = native_design.compare_candidates(plan, evaluate, solver_budget=400,
        budget_strategy='fit_priority', scenario_count=2)
    assert seen == [(True, 4), (True, 4)] + [(False, 98)] * 4
    assert result['candidate_budget'] is None and result['budget_strategy'] == 'fit_priority'
    assert result['unallocated_remainder'] == 0
    assert sum(a['solver_budget'] for a in result['allocations']) == 400
    assert result['enumeration_complete'] and result['search_complete']


@pytest.mark.parametrize('pool', ['fixed', 'fitted'])
def test_fit_priority_all_fixed_or_all_fitted_candidates_have_valid_budgets(design_inputs, pool):
    plan = _plan(design_inputs)
    plan['candidates'] = [c for c in plan['candidates'] if bool(c['free_parameter_ids']) == (pool == 'fitted')]
    seen = []

    def evaluate(candidate, *, method_config):
        seen.append(method_config['remaining_solver_calls'])
        return _fit_result(candidate)

    result = native_design.compare_candidates(plan, evaluate, solver_budget=101,
        budget_strategy='fit_priority', scenario_count=2)
    assert seen == ([4, 4] if pool == 'fixed' else [25] * 4)
    assert result['unallocated_remainder'] == (93 if pool == 'fixed' else 1)
    assert result['status'] == 'completed'


@pytest.mark.parametrize('count,budget', [(None, 100), (True, 100), (1, 100), (2., 100),
                                       (2, 23), (2, True), (2, 5001)])
def test_fit_priority_invalid_or_insufficient_budget_rejects_before_evaluation(design_inputs, count, budget):
    def evaluate(*args, **kwargs):
        pytest.fail('Invalid preallocation must be rejected before fitting')
    with pytest.raises(ValueError):
        native_design.compare_candidates(_plan(design_inputs), evaluate, solver_budget=budget,
            budget_strategy='fit_priority', scenario_count=count)


def test_fit_priority_truncated_fit_cannot_consume_another_candidates_budget(design_inputs):
    plan = _plan(design_inputs)
    seen = []

    def evaluate(candidate, *, method_config):
        allowance = method_config['remaining_solver_calls']
        seen.append((candidate['candidate_id'], allowance))
        result = _fit_result(candidate)
        if len(seen) == 3:
            result.update(status='budget_exhausted', prediction_available=True, solver_calls=allowance)
        return result

    result = native_design.compare_candidates(plan, evaluate, solver_budget=100,
        budget_strategy='fit_priority', scenario_count=2)
    assert [share for _, share in seen] == [4, 4, 23, 23, 23, 23]
    assert result['solver_calls'] == 28 and result['status'] == 'budget_exhausted'
    assert result['enumeration_complete'] and not result['search_complete']


@pytest.mark.parametrize('failure', ['exception', 'overreported_cost', 'infrastructure'])
def test_fit_priority_fatal_errors_stop_all_remaining_candidates(design_inputs, failure):
    seen = []

    def evaluate(candidate, *, method_config):
        seen.append(candidate['candidate_id'])
        if failure == 'exception':
            raise RuntimeError('synthetic infrastructure error')
        result = _fit_result(candidate)
        if failure == 'overreported_cost':
            result['solver_calls'] = method_config['remaining_solver_calls'] + 1
        else:
            result.update(status='failed', infrastructure_failure=True)
        return result

    result = native_design.compare_candidates(_plan(design_inputs), evaluate, solver_budget=100,
        budget_strategy='fit_priority', scenario_count=2)
    assert len(seen) == 1 and result['status'] == 'failed'
    assert result['selected_candidate'] is None and not result['enumeration_complete']


def test_fixed_candidate_native_evaluator_fits_with_exact_cold_build_prediction_allowance(design_inputs, monkeypatch):
    data = design_inputs
    # Exercise the real evaluator, fit and scoped ledgers; replace only scoring/ODE work.
    experiment = object.__new__(experiment_runtime.NativeExperiment)
    experiment.case = data.case
    experiment.specs = data.case['parameter_search']['parameters']
    experiment.ids = [s['parameter_id'] for s in experiment.specs]
    experiment.initial = [s['initial_value'] for s in experiment.specs]
    context = {'ledger': BudgetLedger(500), 'output': data.root, 'model_requests': set()}
    charges = []

    def score(candidate, points, local_context, *, role='calibration', phase='search'):
        calls = experiment._score_call_cost(candidate, len(points), local_context, role=role)
        local_context['ledger'].consume(calls, phase=phase)
        charges.append((role, calls))
        for scenario in data.case['scenarios']:
            if scenario['role'] == role:
                local_context['model_requests'].add((candidate['structure_id'], scenario['scenario_id']))
        return [{'valid': True, 'log_rmse': .2} for _ in points]

    monkeypatch.setattr(experiment, 'score', score)
    plan = _plan(data)
    plan['candidates'] = [c for c in plan['candidates'] if not c['free_parameter_ids']]
    result = native_design.compare_candidates(plan, experiment.evaluator(context), solver_budget=100,
        budget_strategy='fit_priority', scenario_count=2)
    assert result['status'] == 'completed'
    assert charges == [('calibration', 2), ('selection', 2)] * 2
    assert result['solver_calls'] == context['ledger'].snapshot()['totals']['logical_calls'] == 8
    # Another parameterization of an already-requested structure cannot increase build cost.
    warm = deepcopy(plan['candidates'][0])
    warm['candidate_id'] += '_warm'
    warm_result = experiment.evaluator(context)(warm, method_config={
        'remaining_solver_calls': 4, 'fixed_parameterization': True,
        'identifiability_prescreen': False, 'identifiability_feedback': False})
    assert warm_result['status'] == 'completed' and warm_result['solver_calls'] == 2
    assert charges[-2:] == [('calibration', 1), ('selection', 1)]


@pytest.mark.parametrize('trigger', ['before_fit', 'during_fit'])
def test_resource_guard_stops_fitting_closes_runtime_and_persists_failure(design_inputs, monkeypatch, trigger):
    data = design_inputs
    events = []
    started = False

    def guard():
        if trigger == 'before_fit' or started:
            raise RuntimeError('synthetic overall resource limit')

    class FakeExperiment:
        def __init__(self, **kwargs):
            self.progress = kwargs['progress_callback']
            events.append('initialize')

        def evaluator(self, context):
            def evaluate(candidate, *, method_config):
                nonlocal started
                events.append('evaluate')
                context['ledger'].consume(1, phase='search')
                started = True
                self.progress(stage='parameter_fit')
                pytest.fail('The active fit must stop at its resource checkpoint')
            return evaluate

        def close(self):
            events.append('close')

    monkeypatch.setattr(native_design, 'NativeExperiment', FakeExperiment)
    monkeypatch.setattr(native_design, 'verify_development_observations', lambda *args: [])
    monkeypatch.setattr(native_design, 'execute_snapshot',
        lambda **kwargs: pytest.fail('A resource guard failure must not reach prediction'))
    result = native_design.run_native_design(case=data.case, policy=data.policy,
        output='guarded', project_root=data.root, execute=True, resource_guard=guard)
    assert result['status'] == 'failed' and result['stage'] == 'incomplete_delivery'
    assert 'synthetic overall resource limit' in result['reason']
    assert result['model_executed'] is False and result['delivery_complete'] is False
    assert events == (['initialize', 'close'] if trigger == 'before_fit'
                      else ['initialize', 'evaluate', 'close'])
    assert result['budget']['totals']['logical_calls'] == (0 if trigger == 'before_fit' else 1)
    for name in ('progress.json', 'result.json'):
        persisted = json.loads((data.root / 'guarded' / name).read_text())
        assert persisted['status'] == 'failed' and persisted['model_executed'] is False


def _purpose_fit_result(candidate, case, predictions, *, diagnostic=None):
    result = _fit_result(candidate)
    for role in ('calibration', 'selection'):
        scenes = [s for s in case['scenarios'] if s['role'] == role]
        rows = []
        for scene in scenes:
            errors = [math.log(p / observed['value']) for p, observed in zip(predictions, scene['observations'], strict=True)]
            rows.append({'rows': [{'valid': True, 'log_mse': sum(e*e for e in errors)/len(errors),
                'gmfe': math.exp(sum(abs(e) for e in errors)/len(errors)),
                'predictions_umol_per_l': predictions}]})
        score = experiment_runtime.aggregate_scores(scenes, rows)[0]
        result[role + '_scores'] = score
        result[role + '_log_rmse'] = score['log_rmse']
    result['identifiability'] = diagnostic or {'status': 'no_free_parameters', 'rank': 0,
                                               'step_stability_passed': True}
    return result


def _purpose_scoring_case(data, *, max_free=0):
    _purpose_source(data)
    data.policy['max_free_parameters'] = max_free
    for scene in data.case['scenarios']:
        scene.update(auc_evaluable=True, cmax_evaluable=True)
    return _plan(data)


@pytest.mark.parametrize('primary,expected', [('curve', 'partition_pksim_standard__00'),
                                            ('observed_window_auc', 'partition_rodgers_rowland__00')])
def test_purpose_request_changes_choice_with_same_predictions_and_no_weighted_score(design_inputs, primary, expected):
    data = design_inputs
    plan = _purpose_scoring_case(data)
    def evaluate(candidate, *, method_config):
        predictions = [.275, .165] if candidate['matches_template_partition'] else [.2, .2]
        return _purpose_fit_result(candidate, data.case, predictions)
    result = native_design.compare_candidates(plan, evaluate, solver_budget=100, case=data.case,
        objective={'primary': primary, 'max_curve_log_rmse': .3})
    assert result['selected_candidate']['candidate_id'] == expected
    assert len(result['pareto_candidate_ids']) == 2
    assert result['selected_result']['selection_objective']['constraints_passed']
    assert result['scientific_validation_performed'] is False


def test_purpose_auc_cannot_override_explicit_curve_error_limit(design_inputs):
    data = design_inputs
    plan = _purpose_scoring_case(data)
    def evaluate(candidate, *, method_config):
        return _purpose_fit_result(candidate, data.case,
            [.275, .165] if candidate['matches_template_partition'] else [.2, .2])
    result = native_design.compare_candidates(plan, evaluate, solver_budget=100, case=data.case,
        objective={'primary': 'observed_window_auc', 'max_curve_log_rmse': .15})
    assert result['selected_candidate']['candidate_id'] == 'partition_pksim_standard__00'
    rejected = next(r for r in result['evaluations'] if 'rodgers' in r['candidate_id'])
    assert not rejected['purpose_eligible']
    assert 'curve_log_rmse' in str(rejected['purpose_exclusion_reasons'])


@pytest.mark.parametrize('diagnostic', [
    {'status': 'unavailable'},
    {'status': 'evaluated', 'step_stability_passed': False, 'rank': 1},
    {'status': 'evaluated', 'step_stability_passed': True, 'rank': 0}])
def test_purpose_rejects_better_fitted_curve_when_parameter_diagnostics_are_unconfirmed(design_inputs, diagnostic):
    data = design_inputs
    plan = _purpose_scoring_case(data, max_free=1)
    plan['candidates'] = [c for c in plan['candidates'] if c['matches_source_partition']]
    def evaluate(candidate, *, method_config):
        return _purpose_fit_result(candidate, data.case,
            [.25, .15] if candidate['free_parameter_ids'] else [.275, .165], diagnostic=diagnostic)
    result = native_design.compare_candidates(plan, evaluate, solver_budget=100, case=data.case, objective={})
    assert result['selected_candidate']['free_parameter_ids'] == []
    assert all(not r['purpose_eligible'] for r in result['evaluations'] if r['result']['effective_free_parameter_ids'])


def test_purpose_selects_supported_joint_parameters_and_preserves_budget_truncation(design_inputs):
    data = design_inputs
    plan = _purpose_scoring_case(data, max_free=2)
    def evaluate(candidate, *, method_config):
        count = len(candidate['free_parameter_ids'])
        result = _purpose_fit_result(candidate, data.case,
            [.25, .15] if count == 2 else [.35, .25],
            diagnostic={'status': 'evaluated', 'step_stability_passed': True, 'rank': count})
        if count == 2:
            result.update(status='budget_exhausted', prediction_available=True)
        return result
    result = native_design.compare_candidates(plan, evaluate, solver_budget=400, case=data.case, objective={})
    assert len(result['selected_candidate']['free_parameter_ids']) == 2
    assert result['selected_candidate']['matches_source_partition']
    assert result['status'] == 'budget_exhausted' and not result['search_complete']
    assert result['enumeration_complete']


def test_purpose_returns_no_recommendation_when_all_candidates_violate_goal(design_inputs):
    data = design_inputs
    plan = _purpose_scoring_case(data)
    def evaluate(candidate, *, method_config):
        return _purpose_fit_result(candidate, data.case, [.275, .165])
    result = native_design.compare_candidates(plan, evaluate, solver_budget=100, case=data.case,
        objective={'max_curve_log_rmse': 0.})
    assert result['selected_candidate'] is None
    assert result['stop_reason'] == 'no_candidate_meets_requested_objective_and_local_diagnostics'
    assert result['enumeration_complete']


def test_purpose_frontier_keeps_simpler_model_within_declared_tolerance(design_inputs):
    data = design_inputs
    plan = _purpose_scoring_case(data, max_free=1)
    def evaluate(candidate, *, method_config):
        error = .1 if not candidate['free_parameter_ids'] else .09
        return _purpose_fit_result(candidate, data.case,
            [.25*math.exp(error), .15*math.exp(error)],
            diagnostic={'status': 'evaluated', 'step_stability_passed': True,
                        'rank': len(candidate['free_parameter_ids'])})
    result = native_design.compare_candidates(plan, evaluate, solver_budget=200, case=data.case, objective={})
    assert result['selected_candidate']['free_parameter_ids'] == []
    assert result['selected_candidate']['matches_source_partition']


@pytest.mark.parametrize('algorithm', ['purpose_driven', 'study_consistent'])
def test_purpose_entrypoint_uses_residual_fit_and_does_not_deliver_failed_objective(design_inputs, monkeypatch, algorithm):
    data = design_inputs
    _purpose_scoring_case(data)
    data.policy['algorithm'] = algorithm
    data.policy['objective'] = {'max_curve_log_rmse': 0.}
    seen = []
    class FakeExperiment:
        def __init__(self, **kwargs):
            assert kwargs['fit_strategy'] == ('screened_residual' if algorithm == 'study_consistent' else 'residual')
            assert kwargs['require_identifiability'] is True
        def evaluator(self, context):
            def evaluate(candidate, *, method_config):
                context['ledger'].consume(1)
                return _purpose_fit_result(candidate, data.case, [.275, .165])
            return evaluate
        def close(self):
            seen.append('closed')
    monkeypatch.setattr(native_design, 'NativeExperiment', FakeExperiment)
    monkeypatch.setattr(native_design, 'execute_snapshot', lambda **kwargs: pytest.fail('No candidate satisfied the request'))
    result = native_design.run_native_design(case=data.case, policy=data.policy, project_root=data.root,
        output='purpose_rejected', execute=True)
    assert result['algorithm'] == (native_design.STUDY_ALGORITHM if algorithm == 'study_consistent' else native_design.PURPOSE_ALGORITHM)
    assert result['stage'] == 'no_selected_model' and not result['delivery_complete']
    if algorithm == 'study_consistent':
        assert result['selection_consistency']['eligible_candidate_ids'] == []
        saved = json.loads((data.root / 'purpose_rejected/result.json').read_text())
        assert saved['selection_consistency'] == result['selection_consistency']
    assert seen == ['closed']


def test_purpose_frontier_preserves_simple_mechanism_within_error_tolerance(design_inputs):
    data = design_inputs
    plan = _purpose_scoring_case(data)
    simple = deepcopy(next(c for c in plan['candidates'] if c['matches_source_partition']))
    more_edits = deepcopy(simple)
    more_edits.update(candidate_id='additional_mechanism_edit', structure_id='additional_mechanism_edit',
                      structure_edit_count=simple['structure_edit_count'] + 1)
    plan['candidates'] = [simple, more_edits]
    def evaluate(candidate, *, method_config):
        error = .09 if candidate['candidate_id'] == more_edits['candidate_id'] else .1
        return _purpose_fit_result(candidate, data.case, [.25*math.exp(error), .15*math.exp(error)])
    result = native_design.compare_candidates(plan, evaluate, solver_budget=100, case=data.case, objective={})
    assert result['selected_candidate']['candidate_id'] == simple['candidate_id']
    assert set(result['pareto_candidate_ids']) == {simple['candidate_id'], more_edits['candidate_id']}


def test_purpose_checks_delivery_budget_before_spending_search_calls(design_inputs, monkeypatch):
    data = design_inputs
    _purpose_scoring_case(data, max_free=2)
    monkeypatch.setattr(native_design, 'NativeExperiment',
        lambda **kwargs: pytest.fail('Insufficient final budget must block before fitting starts'))
    result = native_design.run_native_design(case=data.case, policy=data.policy, project_root=data.root,
        output='insufficient_purpose_budget', execute=True, max_solver_calls=50)
    assert result['stage'] == 'insufficient_delivery_budget' and result['status'] == 'blocked'
    assert result['required_final_solver_calls'] == 11 and result['reserved_final_solver_calls'] == 10
    assert result['budget']['totals']['logical_calls'] == 0


@pytest.mark.parametrize('use_guard', [False, True])
def test_study_guard_cannot_hide_one_development_study_regression(design_inputs, use_guard):
    data = design_inputs
    plan = _purpose_scoring_case(data, max_free=1)
    def evaluate(candidate, *, method_config):
        free = candidate['free_parameter_ids']
        result = _purpose_fit_result(candidate, data.case, [.25, .15] if free else [.275, .165],
            diagnostic={'status': 'evaluated', 'step_stability_passed': True, 'rank': len(free)})
        if free:
            worse_calibration = _purpose_fit_result(candidate, data.case, [.4, .24])
            result['calibration_scores'] = worse_calibration['calibration_scores']
            result['calibration_log_rmse'] = worse_calibration['calibration_log_rmse']
        return result
    result = native_design.compare_candidates(plan, evaluate, solver_budget=400,
        case=data.case, objective={}, study_consistency=use_guard)
    selected = result['selected_candidate']
    assert bool(selected['free_parameter_ids']) is (not use_guard)
    if use_guard:
        assert selected['matches_source_partition']
        audit = result['selection_consistency']
        assert audit['status'] == 'applied'
        assert audit['cross_validation_performed'] is False
        assert any('worse_than_source:calibration:' in reason
            for row in audit['candidate_reports'] for reason in row['reasons'])


def test_study_policy_blocks_when_request_excludes_fixed_source_baseline(design_inputs):
    data = design_inputs
    _purpose_source(data)
    data.policy.update(algorithm='study_consistent', max_free_parameters=2,
                       parameter_modes={'CL': 'estimate', 'Pint': 'estimate'})
    plan = _plan(data)
    assert not plan['supported'] and plan['candidates'] == []
    assert plan['candidate_count'] == 0 and plan['static_preferred_candidate_id'] is None
    assert 'legal fixed source candidate' in str(plan['reasons'])


def test_study_guard_requires_purpose_objective_before_evaluation(design_inputs):
    with pytest.raises(ValueError, match='explicit purpose objective'):
        native_design.compare_candidates(_plan(design_inputs),
            lambda *args, **kwargs: pytest.fail('Must validate before evaluating'),
            solver_budget=100, study_consistency=True)


@pytest.mark.parametrize('audit', [None, {}, {'source_partition_action': ''}])
def test_study_guard_requires_source_audit_before_evaluation(design_inputs, audit):
    data = design_inputs
    plan = _purpose_scoring_case(data, max_free=1)
    plan['source_structure_audit'] = audit
    with pytest.raises(ValueError, match='verified source partition action'):
        native_design.compare_candidates(plan,
            lambda *args, **kwargs: pytest.fail('Must validate before evaluating'),
            solver_budget=100, objective={}, case=data.case, study_consistency=True)
