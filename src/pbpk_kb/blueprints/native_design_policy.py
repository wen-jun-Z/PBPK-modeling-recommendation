"""Small, source-checked request policy for existing executable PBPK edits.

This module plans only: it does not fit, run a solver, call an LLM, or open
selection/test observations. Equal scores mean passed hard constraints, never
invented differential evidence support. Historical candidate generation is intact.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .design_contracts import BlueprintDesignRequest
from .design_objectives import PKDesignObjective
from .experiment_runtime import candidate_space, digest, file_sha, validate_case
from .human_model_snapshot import _MM_PROCESS_PARAMETERS
from .mechanism_edits import ACTION, MechanismNotApplicable, _branches, compile_native_candidate
from .native_edits import PARTITION_ACTIONS, _snapshot_from_design
from .schemas import ProvenanceRef


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class LinearizationEvidenceRef(_StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class _ProcessIdentity(_StrictModel):
    internal_name: str = Field(min_length=1)
    molecule: str = Field(min_length=1)
    data_source: str = Field(min_length=1)
    parameter_name: Literal['kcat']


class LocalUnboundEvidence(_StrictModel):
    kind: Literal['local_unbound_substrate_upper_bound']
    compound_id: str = Field(min_length=1)
    design_request_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    parameter_search_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    bound_scope: Literal['all_parameter_vectors_in_search_bounds']
    process_identity: _ProcessIdentity
    upper_bound_umol_per_l: float = Field(ge=0)
    role: Literal['prior', 'calibration']
    source_refs: list[ProvenanceRef] = Field(min_length=1, max_length=16)
    study_id: str | None = None

    @model_validator(mode='after')
    def calibration_source_identity(self):
        if (self.role == 'calibration') != (self.study_id is not None):
            raise ValueError('Only calibration evidence must provide study_id')
        return self


class NativeDesignPolicy(_StrictModel):
    version: Literal['native-request-design-v1'] = 'native-request-design-v1'
    algorithm: Literal['finite_comparison', 'purpose_driven', 'study_consistent'] = 'finite_comparison'
    objective: PKDesignObjective = Field(default_factory=PKDesignObjective)
    prediction_request: BlueprintDesignRequest
    max_free_parameters: int = Field(default=1, ge=0, le=2)
    parameter_modes: dict[str, Literal['fixed', 'estimate', 'auto']] = Field(default_factory=dict)
    partition_mode: Literal['source', 'template', 'compare_registered'] = 'compare_registered'
    kinetics_mode: Literal['retain_source', 'require_saturable', 'allow_supported_linearization'] = 'retain_source'
    linearization_evidence: list[LinearizationEvidenceRef] = Field(default_factory=list, max_length=256)

    @model_validator(mode='after')
    def objective_requires_purpose_algorithm(self):
        if self.algorithm == 'finite_comparison' and self.objective != PKDesignObjective():
            raise ValueError('A nondefault objective requires the purpose_driven algorithm')
        return self


def design_request_sha256(request: BlueprintDesignRequest | dict) -> str:
    """Bind a proof to the complete normalized scenario, including administration."""
    payload = request.model_dump(mode='json') if isinstance(request, BlueprintDesignRequest) else request
    return digest(BlueprintDesignRequest.model_validate(payload).model_dump(mode='json'))


def _project_file(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Source paths must be project-relative without parent traversal')
    path = (root / path).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Source must be a regular file inside the project')
    return path


def _references(value):
    if isinstance(value, dict):
        if {'provenance_id', 'source_id', 'source_checksum', 'source_location'} <= value.keys():
            yield ProvenanceRef.model_validate(value).model_dump(mode='json')
        else:
            for key, item in value.items():
                if key != 'studies':
                    yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _blocked_sources(case, requests):
    ids = set(case.get('forbidden_source_ids', []))
    checksums = set(case.get('forbidden_source_checksums', []))
    forbidden_studies = set(case.get('forbidden_study_group_ids', []))
    calibration, roles, source_checksums = {}, {}, {}
    for request in requests:
        ids.update(request.get('forbidden_source_ids', []))
        forbidden_studies.update(request.get('forbidden_study_ids', []))
    for request in requests:
        for study in request.get('studies', []):
            identities = [('study', study['study_id']), ('source', study['source_id']),
                          ('checksum', study['source_checksum'])]
            identities.extend(('cohort', value) for value in study['cohort_ids'])
            for identity in identities:
                if identity in roles and roles[identity] != study['role']:
                    raise ValueError('Study sources or cohorts cannot cross data roles across scenarios')
                roles[identity] = study['role']
            previous = source_checksums.setdefault(study['source_id'], study['source_checksum'])
            if previous != study['source_checksum']:
                raise ValueError('One study source has conflicting checksums across scenarios')
            if study['role'] in {'selection', 'test'} or study['study_id'] in forbidden_studies:
                ids.add(study['source_id'])
                checksums.add(study['source_checksum'])
            elif study['role'] == 'calibration':
                calibration.setdefault(study['study_id'], set()).add((study['source_id'], study['source_checksum']))
    return ids, checksums, calibration


def _verify_ref(reference, root, blocked_ids, blocked_sha, cache):
    key = (reference['source_id'], reference['source_checksum'])
    if key[0] in blocked_ids or key[1] in blocked_sha:
        raise ValueError('A selection, test or forbidden source entered design inputs')
    path = _project_file(root, key[0])
    if path not in cache:
        cache[path] = file_sha(path)
    if cache[path] != key[1]:
        raise ValueError('Design source checksum mismatch: ' + key[0])


def _partition_source_audit(request, root):
    """Inspect only source compound names/methods, never import fitted values.

    The checksum-bound JSON asset is loaded as a document; only the allowlisted
    Compounds[].Name/CalculationMethods fields affect this audit. Source curves,
    identification records and historical fitted parameter values are not used.
    """
    instance = request['instance']
    compound = instance['compound']
    name = compound['Name']
    reference = instance['compound_sources']['CalculationMethods']
    location = f'Compounds[{name}].CalculationMethods'
    if reference['source_location'].split(';', 1)[0].strip() != location:
        raise ValueError('Partition source location must identify the exact compound CalculationMethods')
    path = _project_file(root, reference['source_id'])
    if path.stat().st_size > 8 * 1024 ** 2:
        raise ValueError('Partition source JSON exceeds 8 MiB')
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Partition source must be a verifiable JSON model asset') from exc
    if not isinstance(document, dict) or not isinstance(document.get('Compounds'), list):
        raise ValueError('Partition source JSON must provide a Compounds array')
    matches = [item for item in document['Compounds']
               if isinstance(item, dict) and item.get('Name') == name]
    if len(matches) != 1:
        raise ValueError('Partition source must contain exactly one matching compound name')

    def partition_action(methods):
        if (not isinstance(methods, list) or not methods
                or any(not isinstance(item, str) or not item for item in methods)
                or len(set(methods)) != len(methods)):
            raise ValueError('Partition CalculationMethods must be unique nonempty method names')
        actions = [key for key, value in PARTITION_ACTIONS.items() if value in methods]
        unknown = [item for item in methods if item.startswith('Cellular partition coefficient method')
                   and item not in PARTITION_ACTIONS.values()]
        if len(actions) != 1 or unknown:
            raise ValueError('One registered source and template partition method is required')
        return actions[0]

    source_methods = matches[0].get('CalculationMethods')
    template_methods = compound['CalculationMethods']
    source_action = partition_action(source_methods)
    template_action = partition_action(template_methods)
    nonpartition = lambda methods: sorted(item for item in methods if item not in PARTITION_ACTIONS.values())
    if nonpartition(source_methods) != nonpartition(template_methods):
        raise ValueError('Nonpartition calculation methods differ between source and reconstructed template')
    return {
        'status': 'source_structure_preserved' if source_action == template_action else 'reconstructed_override',
        'source_partition_action': source_action,
        'template_partition_action': template_action,
        'source_ref': deepcopy(reference),
        'values_used': [location],
        'historical_fitted_parameters_imported': False,
        'scope': 'partition_formula_only; other model fields and registered initial parameters remain unchanged',
    }


def _linearization_support(policy, named_requests, parameter_specs, root, blocked, source_refs, cache):
    """Every transformed branch in every scenario needs an independent bound."""
    blocked_ids, blocked_sha, calibration = blocked
    supplied = {}
    for ref in policy.linearization_evidence:
        _verify_ref({'source_id': ref.path, 'source_checksum': ref.sha256}, root, blocked_ids, blocked_sha, cache)
        path = _project_file(root, ref.path)
        if path.stat().st_size > 1024 ** 2:
            raise ValueError('Local-bound evidence document exceeds 1 MiB')
        doc = LocalUnboundEvidence.model_validate_json(path.read_text())
        if doc.compound_id != policy.prediction_request.requirement.compound_semantic_id:
            raise ValueError('Local-bound evidence belongs to a different compound')
        if doc.parameter_search_sha256 != digest(parameter_specs):
            raise ValueError('Local-bound evidence does not cover the registered parameter search domain')
        refs = [r.model_dump(mode='json') for r in doc.source_refs]
        for source in refs:
            _verify_ref(source, root, blocked_ids, blocked_sha, cache)
        if doc.role == 'calibration' and not any(
            (r['source_id'], r['source_checksum']) in calibration.get(doc.study_id, set()) for r in refs
        ):
            raise ValueError('Calibration bound has no matching declared calibration source')
        key = (doc.design_request_sha256, digest(doc.process_identity.model_dump(mode='json')))
        if key in supplied:
            raise ValueError('Duplicate local-bound evidence for one scenario/process')
        supplied[key] = (doc, ref.model_dump(mode='json'))
        source_refs.extend(refs)
    checks, used = [], set()
    for name, request in named_requests:
        try:
            branches = _branches(request['instance']['compound'])
        except MechanismNotApplicable as exc:
            checks.append({'scenario_id': name, 'supported': False, 'reason': str(exc)})
            continue
        for branch in branches:
            key = (design_request_sha256(request), digest(branch['from_identity']))
            item = supplied.get(key)
            ratio = item[0].upper_bound_umol_per_l / branch['km_umol_per_l'] if item else None
            if item:
                used.add(key)
            checks.append({'scenario_id': name, 'process_identity': branch['from_identity'],
                'supported': ratio is not None and ratio <= 0.1,
                'upper_bound_over_Km': ratio, 'evidence': item[1] if item else None,
                'reason': 'local_bound_within_limit' if ratio is not None and ratio <= .1
                          else 'local_bound_exceeds_limit' if item else 'local_bound_missing'})
    if supplied.keys() - used:
        raise ValueError('Local-bound evidence does not bind an actual scenario/process')
    return bool(checks) and all(row['supported'] for row in checks), checks


def build_design_plan(case, policy, *, project_root) -> dict:
    """Return a finite executable plan or explicit blocks, without running it."""
    result = {'version': 'native-request-design-v1', 'supported': False, 'status': 'blocked',
        'reasons': [], 'candidates': [], 'excluded_candidates': [], 'rule_candidate_id': None,
        'request_sha256': None, 'source_refs': [], 'prediction_scenario': None,
        'selection_basis': 'hard_constraints_then_fewest_free_parameters_and_edits',
        'guidance_informative': False, 'execution': {'solver_calls': 0, 'llm_calls': 0}}
    try:
        root = Path(project_root).resolve(strict=True)
        raw = policy.model_dump(mode='json') if isinstance(policy, NativeDesignPolicy) else policy
        policy = NativeDesignPolicy.model_validate(raw)
        result['algorithm'] = policy.algorithm
        result['objective'] = policy.objective.model_dump(mode='json')
        prediction = policy.prediction_request.model_dump(mode='json')
        result['request_sha256'] = design_request_sha256(prediction)
        result['policy_sha256'] = digest(policy.model_dump(mode='json'))
        validate_case(case)
        if prediction['studies']:
            raise ValueError('Prediction request must contain no studies or observations')
        scenarios = case['scenarios']
        if any(s['scenario_id'] == 'user_request' for s in scenarios):
            raise ValueError('user_request is reserved for the new prediction scenario')
        named = [(s['scenario_id'], BlueprintDesignRequest.model_validate(s['design_request']).model_dump(mode='json'))
                 for s in scenarios] + [('user_request', prediction)]
        requests = [r for _, r in named]
        for scenario, request in zip(scenarios, requests, strict=False):
            matching = [s for s in request['studies'] if s['study_id'] == scenario['study_group_id']]
            if not matching or any(s['role'] != scenario['role'] for s in matching):
                raise ValueError('Scenario study role does not match its declared source study')
        blocked = _blocked_sources(case, requests)
        specs = case['parameter_search']['parameters']
        ids = {s['parameter_id'] for s in specs}
        if len(specs) > 2 or set(policy.parameter_modes) - ids:
            raise ValueError('Parameter policy must name at most two registered parameters')
        required_fit = {p for p, mode in policy.parameter_modes.items() if mode == 'estimate'}
        fixed = {p for p, mode in policy.parameter_modes.items() if mode == 'fixed'}
        if len(required_fit) > policy.max_free_parameters:
            raise ValueError('Required estimated parameters exceed the free-parameter limit')
        compound_id = case['compound_id']
        anchor = requests[0]
        sources, cache = [], {}
        for name, request in named:
            requirement, instance = request['requirement'], request['instance']
            if requirement['compound_semantic_id'] != compound_id or requirement['name'] != anchor['requirement']['name']:
                raise ValueError('All scenarios and prediction must describe the same compound')
            if not instance or not request['human_model']:
                raise ValueError('A registered human template and source-bound drug instance are required')
            if requirement['population'] != 'healthy_adult':
                raise ValueError('Only healthy adult predictions are implemented')
            if request['modules'] or request['base'] is not None:
                raise ValueError('Generic modules or alternate physiology cannot replace native edits')
            if any((t['analyte'], t['matrix'], t['quantity'], t['binding']) !=
                   ('parent', 'plasma', 'concentration', 'total') for t in requirement['targets']):
                raise ValueError('Only parent total plasma concentration targets are implemented')
            if request['human_model'] != anchor['human_model']:
                raise ValueError('Human model source lock differs across scenarios')
            for field in ('compound', 'compound_sources', 'expression_profiles', 'expression_profile_sources'):
                if instance[field] != anchor['instance'][field]:
                    raise ValueError('Drug configuration/source differs across scenarios: ' + field)
            for source in _references(request):
                _verify_ref(source, root, *blocked[:2], cache)
                sources.append(source)
            _snapshot_from_design(BlueprintDesignRequest.model_validate(request), root)
            expected = f"Organism|PeripheralVenousBlood|{requirement['name']}|Plasma (Peripheral Venous Blood)"
            if name != 'user_request' and next(s for s in scenarios if s['scenario_id'] == name)['target_output'] != expected:
                raise ValueError('A development scenario does not target parent total plasma')
        for spec in specs:
            if 'source' not in spec:
                raise ValueError('Every parameter initial value and bounds need explicit source provenance')
            ref = ProvenanceRef.model_validate(spec['source']).model_dump(mode='json')
            _verify_ref(ref, root, *blocked[:2], cache)
            sources.append(ref)
        methods = anchor['instance']['compound']['CalculationMethods']
        source_actions = [key for key, value in PARTITION_ACTIONS.items() if value in methods]
        if len(source_actions) != 1:
            raise ValueError('One registered source partition method is required')
        template_partition = source_actions[0]
        source_partition = template_partition
        if policy.algorithm in {'purpose_driven', 'study_consistent'}:
            source_audit = _partition_source_audit(anchor, root)
            result['source_structure_audit'] = source_audit
            source_partition = source_audit['source_partition_action']
        # Preserving saturation does not require eligibility for linearization:
        # explicit same-enzyme interactions can forbid the latter but not MM.
        has_mm = any(process['InternalName'] in _MM_PROCESS_PARAMETERS
                     for process in anchor['instance']['compound']['Processes'])
        if policy.kinetics_mode == 'require_saturable' and not has_mm:
            raise ValueError('Saturable kinetics requested but no supported source MM branches exist')
        linear_ok, linear_checks = _linearization_support(policy, named, specs, root, blocked, sources, cache)
        result['linearization_checks'] = linear_checks
        candidates, variants = candidate_space(case, root=root), {}
        for original in candidates:
            candidate, reasons = deepcopy(original), []
            action = candidate['action']
            free = set(candidate['free_parameter_ids'])
            partition = action.split('__')[0]
            if policy.partition_mode == 'source' and partition != source_partition:
                reasons.append('partition_change_not_requested')
            if policy.partition_mode == 'template' and partition != template_partition:
                reasons.append('template_partition_change_not_requested')
            if ACTION in action and (policy.kinetics_mode != 'allow_supported_linearization' or not linear_ok):
                reasons.append('source_kinetics_required' if policy.kinetics_mode != 'allow_supported_linearization'
                               else 'linearization_evidence_incomplete_or_out_of_range')
            if ACTION in action and partition != template_partition:
                reasons.append('local_concentration_evidence_does_not_cover_changed_partition_method')
            if free & fixed:
                reasons.append('fixed_parameter_would_be_estimated')
            if not required_fit <= free:
                reasons.append('required_parameter_not_estimated')
            if len(free) > policy.max_free_parameters:
                reasons.append('free_parameter_limit_exceeded')
            if reasons:
                result['excluded_candidates'].append({'candidate_id': candidate['candidate_id'], 'reasons': reasons})
                continue
            if action not in variants:
                variants[action] = compile_native_candidate(prediction, action, project_root=root, parameter_specs=specs)
                # Check the delivery path with initial coordinates before fitting.
                # A valid batch parameter path alone does not prove that the
                # selected snapshot can later be rebuilt with those values.
                for _, request in named:
                    compile_native_candidate(request, action, project_root=root, parameter_specs=specs,
                        fitted_values=[s['initial_value'] for s in specs], parameter_provenance=specs[0]['source'])
            variant = variants[action]
            if variant['semantic_hash'] != candidate['structure_semantic_hash']:
                raise ValueError('Prediction mechanism semantics differ from the fitting candidate')
            candidate.update(requirement_score=1., evidence_score=1., guidance_is_informative_in_this_case=False,
                design_decision={'constraints_passed': True, 'score_semantics': 'binary_hard_constraint_pass',
                    'parameter_modes': {p: policy.parameter_modes.get(p, 'auto') for p in sorted(ids)},
                    'source_initial_values_retained_for_fixed_parameters': True,
                    'linearization_supported': linear_ok if ACTION in action else None,
                    'prediction_snapshot_sha256': variant['edit_manifest']['snapshot_sha256']})
            if policy.algorithm in {'purpose_driven', 'study_consistent'}:
                candidate.update(
                    matches_source_partition=partition == source_partition,
                    matches_template_partition=partition == template_partition,
                    partition_origin=('original_source' if partition == source_partition else
                                      'reconstructed_template' if partition == template_partition else
                                      'registered_alternative'),
                    source_partition_edit_count=int(partition != source_partition))
            result['candidates'].append(candidate)
        result['candidates'].sort(key=lambda c: (len(c['free_parameter_ids']), c['structure_edit_count'], c['candidate_id']))
        if not result['candidates'] or len(result['candidates']) > 24:
            raise ValueError('Constraints leave no candidates or exceed the 24-candidate bound')
        if policy.algorithm in {'purpose_driven', 'study_consistent'}:
            result['baseline_definitions'] = {
                label: {
                    'partition_action': partition,
                    'parameter_values': 'registered_initial_values',
                    'is_original_complete_source_model': False,
                    'eligible_candidate_ids': [candidate['candidate_id'] for candidate in result['candidates']
                        if candidate['action'] == partition and not candidate['free_parameter_ids']],
                }
                for label, partition in (
                    ('source_structure_initial_parameters', source_partition),
                    ('reconstructed_template_initial_parameters', template_partition))}
        refs = {digest(ref): ref for ref in sources}
        result.update(supported=True, status='ready', source_refs=sorted(refs.values(), key=digest),
            candidate_count=len(result['candidates']), rule_candidate_id=result['candidates'][0]['candidate_id'],
            static_preferred_candidate_id=result['candidates'][0]['candidate_id'],
            decision_summary={'selection_algorithm': 'bounded_enumeration',
                'request_effect': 'native_regimen_and_hard_candidate_constraints',
                'evidence_effect': 'source_verification_and_mechanism_admissibility',
                'differential_guidance_scores_used': False,
                'unfiltered_candidate_count': len(candidates),
                'excluded_candidate_count': len(result['excluded_candidates'])},
            prediction_scenario={'scenario_id': 'user_request', 'compound_id': compound_id,
                'design_request': prediction,
                'target_output': f"Organism|PeripheralVenousBlood|{prediction['requirement']['name']}|Plasma (Peripheral Venous Blood)"},
            limitations=['Static source checks do not establish predictive accuracy or validate historical parameter values.',
                         'A source-bound local-concentration bound is a supplied assertion, not an automatic proof of its biological correctness.',
                         'Only registered native choices on a supplied drug instance are supported; arbitrary drug assembly is not implemented.'])
        if policy.algorithm in {'purpose_driven', 'study_consistent'}:
            if policy.algorithm == 'study_consistent' and not any(
                    candidate['action'] == source_partition and not candidate['free_parameter_ids']
                    for candidate in result['candidates']):
                raise ValueError('Study consistency requires a legal fixed source candidate; do not override requested estimated parameters to create one')
            preferred = min(result['candidates'], key=lambda candidate: (
                len(candidate['free_parameter_ids']), candidate['source_partition_edit_count'],
                candidate['structure_edit_count'], candidate['candidate_id']))
            result['rule_candidate_id'] = result['static_preferred_candidate_id'] = preferred['candidate_id']
            result['selection_basis'] = 'requested_pk_objective_with_constraints_local_diagnostics_and_parsimony'
            result['decision_summary'].update(
                selection_algorithm='purpose_driven_finite_design',
                request_effect='native_regimen_hard_constraints_and_explicit_pk_objective',
                evidence_effect='verified_source_partition_and_registered_parameter_domains',
                runtime_fit='bounded_residual_least_squares_on_calibration_data',
                runtime_selection='local_diagnostic_gate_then_selection_objective_pareto_and_parsimony',
                objective=policy.objective.model_dump(mode='json'),
                objective_data_validation='deferred_until_execution; no endpoint observations inspected by planning')
            if policy.algorithm == 'study_consistent':
                result['decision_summary'].update(
                    runtime_fit='calibration_grid_screen_then_bounded_residual_least_squares',
                    runtime_selection='source_relative_primary_error_guard_in_every_development_study_then_original_selector',
                    stability_claim='descriptive_development_consistency_not_cross_validation',
                    source_reference_required=True)
    except (ValidationError, ValueError, TypeError, KeyError, OSError) as exc:
        detail = '; '.join('.'.join(map(str, e['loc'])) + ': ' + e['type'] for e in exc.errors()) if isinstance(exc, ValidationError) else str(exc)
        result['reasons'].append({'code': 'NATIVE_DESIGN_BLOCKED', 'detail': detail})
        result.update(supported=False, status='blocked', candidates=[], candidate_count=0,
            rule_candidate_id=None, static_preferred_candidate_id=None)
    return result
