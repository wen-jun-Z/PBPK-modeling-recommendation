"""Request-constrained PBPK design on the registered human template.

The small legal space is compared directly with the existing common fitter.
There is no learned ranker, LLM judge, or access to external test observations.
Planning is the default; numerical fitting must be explicitly requested.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .design_diagnostics import BudgetExceeded, BudgetLedger
from .design_runtime import execute_snapshot
from .experiment_runtime import NativeExperiment, digest, file_sha, save, validate_case, validate_native_parallelism
from .native_design_policy import NativeDesignPolicy, build_design_plan
from .native_edits import apply_parameter_values, compile_candidate
from .mechanism_edits import compile_native_candidate
from .search_methods import run_method_search
from .design_contracts import DesignTarget
from .design_objectives import PKDesignObjective, assess_objective, validate_objective_data
from .study_consistency import apply_study_consistency
from .schemas import ProvenanceRef

ALGORITHM = 'request-constrained-native-design-v2'
PURPOSE_ALGORITHM = 'purpose-driven-native-design-v3'
STUDY_ALGORITHM = 'study-consistent-native-design-v4'


def verify_development_observations(case, root):
    """Authenticate Dcal/Dsel inputs only; never follow external source_audit.

    These values are checked after admissible candidates have been fixed. The
    plan itself needs identities/provenance, not the selection observations.
    """
    records = []
    for scenario in case['scenarios']:
        refs = [s for s in scenario['design_request']['studies']
                if s['study_id'] == scenario['study_group_id'] and s['role'] == scenario['role']]
        if len(refs) != 1:
            raise ValueError('Each development scenario needs exactly one matching source study')
        ref = refs[0]
        relative = Path(ref['source_id'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Development observations must have a project-relative source')
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 8 * 1024**2:
            raise ValueError('Development source must be a project file smaller than 8 MiB')
        if file_sha(path) != ref['source_checksum']:
            raise ValueError('Development observation source checksum mismatch')
        original = json.loads(path.read_text())
        for key in ('scenario_id', 'study_group_id', 'compound_id', 'role', 'time_unit', 'unit', 'observations'):
            if key not in original or key not in scenario or original[key] != scenario[key]:
                raise ValueError('Development observations differ from their registered source: ' + key)
        records.append({'scenario_id': scenario['scenario_id'], 'role': scenario['role'],
                        'source_id': ref['source_id'], 'sha256': ref['source_checksum']})
    return records


class TemplateRequest(BaseModel):
    """Compact user input; drug values and human physiology stay in the template."""
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    version: Literal['pbpk-template-request-v1'] = 'pbpk-template-request-v1'
    algorithm: Literal['finite_comparison', 'purpose_driven', 'study_consistent'] = 'finite_comparison'
    objective: PKDesignObjective = Field(default_factory=PKDesignObjective)
    template_scenario_id: str = Field(min_length=1)
    population: str = 'healthy_adult'
    route: Literal['oral', 'intravenous']
    dose: float = Field(gt=0)
    dose_unit: Literal['mg', 'mg/kg']
    infusion_minutes: float = Field(default=0., ge=0)
    formulation: Literal['source', 'solution'] = 'source'
    water_ml_per_kg: float | None = Field(default=None, ge=0)
    targets: list[DesignTarget] | None = Field(default=None, min_length=1)
    max_free_parameters: int = Field(default=1, ge=0, le=2)
    parameter_modes: dict[str, Literal['fixed', 'estimate', 'auto']] = Field(default_factory=dict)
    partition_mode: Literal['source', 'template', 'compare_registered'] = 'compare_registered'
    kinetics_mode: Literal['retain_source', 'require_saturable', 'allow_supported_linearization'] = 'retain_source'

    @model_validator(mode='after')
    def administration(self):
        if self.route == 'oral' and (self.infusion_minutes or self.water_ml_per_kg is None):
            raise ValueError('Oral requests require explicit water volume and no IV infusion')
        if self.route == 'intravenous' and (self.water_ml_per_kg is not None or self.formulation != 'source'):
            raise ValueError('IV requests cannot contain oral water or formulation changes')
        return self


def policy_from_template(case, specification, *, request_source):
    """Materialize a user's regimen in the real native protocol, without fitting."""
    spec = TemplateRequest.model_validate(specification)
    source = ProvenanceRef.model_validate(request_source).model_dump(mode='json')
    if source['provenance_kind'] != 'user_request':
        raise ValueError('User protocol changes must retain user_request provenance')
    matches = [s for s in case['scenarios'] if s['scenario_id'] == spec.template_scenario_id]
    if len(matches) != 1:
        raise ValueError('Choose one registered development scenario as the human/drug template')
    request = deepcopy(matches[0]['design_request'])
    request['studies'] = []
    # Study-linked evidence remains in the development case where its role can
    # be checked. A prediction request has no observation/study role of its own.
    request['evidence'] = [e for e in request.get('evidence', []) if e.get('study_id') is None]
    requirement, instance = request['requirement'], request['instance']
    requirement.update(population=spec.population, route=spec.route, dose=spec.dose, dose_unit=spec.dose_unit)
    if spec.targets is not None:
        requirement['targets'] = [target.model_dump(mode='json') for target in spec.targets]
    instance.update(population_context=spec.population, protocol_source=source, simulation_name='SIM_User_Request')
    parameters = [{'Name': 'Start time', 'Unit': 'h', 'Value': 0.},
                  {'Name': 'InputDose', 'Unit': spec.dose_unit, 'Value': spec.dose}]
    if spec.route == 'oral':
        application = 'Oral'
        parameters.append({'Name': 'Volume of water/body weight', 'Unit': 'ml/kg', 'Value': spec.water_ml_per_kg})
        if spec.formulation == 'solution':
            instance['formulation'] = {'Name': 'User_Solution', 'FormulationType': 'Formulation_Dissolved', 'Parameters': []}
            instance['formulation_source'] = source
        if instance.get('formulation') is None:
            raise ValueError('This template has no oral formulation; request solution explicitly or supply a sourced formulation')
        requirement['formulation'] = instance['formulation']['Name']
    else:
        application = 'Intravenous' if spec.infusion_minutes > 0 else 'IntravenousBolus'
        if spec.infusion_minutes > 0:
            parameters.append({'Name': 'Infusion time', 'Unit': 'min', 'Value': spec.infusion_minutes})
        instance['formulation'] = instance['formulation_source'] = None
        requirement['formulation'] = None
    instance['protocol'] = {'Name': 'PROTO_User_Request', 'ApplicationType': application,
                            'DosingInterval': 'Single', 'Parameters': parameters}
    return NativeDesignPolicy(prediction_request=request, max_free_parameters=spec.max_free_parameters,
        parameter_modes=spec.parameter_modes, partition_mode=spec.partition_mode, kinetics_mode=spec.kinetics_mode,
        algorithm=spec.algorithm, objective=spec.objective)


def _compile(request, candidate, specs, root, *, values=None, provenance=None):
    if candidate.get('native_mechanism_search'):
        return compile_native_candidate(request, candidate['action'], project_root=root,
            parameter_specs=specs, fitted_values=values, parameter_provenance=provenance)
    variant = compile_candidate(request, candidate['action'], project_root=root, parameter_specs=specs)
    if values is not None:
        revised = apply_parameter_values(variant['request'], specs, values, provenance)
        variant = compile_candidate(revised, candidate['action'], project_root=root, parameter_specs=specs)
    return variant


def compare_candidates(plan, evaluate, *, solver_budget, selection_tolerance=0.02,
                       budget_strategy='equal', scenario_count=None, objective=None, case=None,
                       study_consistency=False):
    """Preallocate budgets without seeing any calibration or selection errors.

    Reuse E0's result and parameter validation for each individual candidate.
    Fit priority reserves build/predict calls for fixed candidates, dividing the
    rest equally among fitted candidates. No outcome-dependent budget transfer.
    Equal shares retain the previous algorithm as a reproducible comparator.
    """
    if not plan['supported'] or not plan['candidates']:
        raise ValueError('A supported nonempty design plan is required')
    if type(study_consistency) is not bool or study_consistency and objective is None:
        raise ValueError('Study consistency requires an explicit purpose objective')
    source_action = (plan.get('source_structure_audit') or {}).get('source_partition_action')
    if study_consistency and (not isinstance(source_action, str) or not source_action):
        raise ValueError('Study consistency requires a verified source partition action')
    if objective is not None:
        objective = PKDesignObjective.model_validate(objective)
        if case is None:
            raise ValueError('Purpose-driven selection requires its development case')
        validate_objective_data(case, objective)
    candidates = deepcopy(plan['candidates'])
    if not 1 <= len(candidates) <= 24 or len({c['candidate_id'] for c in candidates}) != len(candidates):
        raise ValueError('Use 1..24 unique legal candidates')
    if type(solver_budget) is not int or not len(candidates) <= solver_budget <= 5000:
        raise ValueError('Solver budget must cover each candidate and not exceed 5000')
    candidates.sort(key=lambda c: (len(c['free_parameter_ids']), c['structure_edit_count'], c['candidate_id']))
    if budget_strategy not in {'equal', 'fit_priority'}:
        raise ValueError('Unknown candidate budget strategy')
    share = solver_budget // len(candidates)
    budgets = {c['candidate_id']: share for c in candidates}
    if budget_strategy == 'fit_priority':
        if type(scenario_count) is not int or scenario_count < 2:
            raise ValueError('Fit priority requires the calibration/selection scenario count')
        # One native build plus one prediction for each development scenario.
        # This is a conservative cold cost; cache hits do not release budget.
        fixed_cost = 2 * scenario_count
        fixed = [c for c in candidates if not c['free_parameter_ids']]
        fitted = [c for c in candidates if c['free_parameter_ids']]
        if solver_budget < fixed_cost * len(candidates):
            raise ValueError('Budget cannot cover a source-initialized prediction for every candidate')
        for candidate in fixed:
            budgets[candidate['candidate_id']] = fixed_cost
        if fitted:
            fit_share = (solver_budget - fixed_cost * len(fixed)) // len(fitted)
            for candidate in fitted:
                budgets[candidate['candidate_id']] = fit_share
    records, history, allocations, calls, fatal = [], [], [], 0, None
    for candidate in candidates:
        allowance = budgets[candidate['candidate_id']]
        outcome = run_method_search(method='E0', candidates=[candidate], evaluate=evaluate,
            max_candidates=1, max_solver_calls=allowance, selection_tolerance=selection_tolerance)
        allocations.append({'candidate_id': candidate['candidate_id'], 'solver_budget': allowance,
                            'solver_calls': outcome['solver_calls']})
        records.extend(outcome['evaluations'])
        history.extend(outcome['history'])
        calls += outcome['solver_calls']
        if (not outcome['solver_cost_complete'] or outcome['stop_reason'] == 'evaluator_budget_contract_violation'
                or any(r['result'].get('infrastructure_failure') for r in outcome['evaluations'])):
            fatal = outcome['stop_reason'] or 'native_infrastructure_failure'
            break
    by_id = {c['candidate_id']: c for c in candidates}
    eligible = [r for r in records if r['selection_eligible']]
    pareto_ids, consistency_report = None, None
    if objective is not None:
        for row in records:
            fitted = row['result']
            if not row['selection_eligible']:
                row['purpose_eligible'] = False
                continue
            for role in ('calibration', 'selection'):
                fitted[role + '_objective'] = assess_objective(
                    [s for s in case['scenarios'] if s['role'] == role],
                    fitted[role + '_scores'], objective)
            free = fitted.get('effective_free_parameter_ids', by_id[row['candidate_id']]['free_parameter_ids'])
            diagnostic = fitted.get('identifiability') or {}
            diagnostic_pass = (not free or diagnostic.get('status') == 'evaluated'
                and diagnostic.get('step_stability_passed') is True
                and diagnostic.get('rank') == len(free))
            assessment = fitted['selection_objective']
            row['purpose_eligible'] = bool(assessment['valid'] and assessment['constraints_passed']
                                           and fitted['calibration_objective']['valid'] and diagnostic_pass)
            row['local_diagnostics_passed'] = bool(diagnostic_pass)
            # Do not rewrite the native outcome or hide a budget-truncated fit.
            row['purpose_exclusion_reasons'] = ([] if diagnostic_pass else ['local_parameter_diagnostics_unconfirmed'])
            row['purpose_exclusion_reasons'].extend(assessment['violations'])
            if not fitted['calibration_objective']['valid']:
                row['purpose_exclusion_reasons'].extend(
                    'calibration: ' + reason for reason in fitted['calibration_objective']['violations'])
        eligible = [r for r in eligible if r.get('purpose_eligible')]
        if study_consistency:
            eligible, consistency_report = apply_study_consistency(records, by_id,
                primary=objective.primary, source_action=source_action)
        eligible, pareto_ids = _purpose_frontier(eligible, by_id)
    selected = None
    if eligible and fatal is None:
        def selection_error(row):
            return (row['result']['selection_objective']['primary_error'] if objective is not None
                    else row['result']['selection_log_rmse'])
        tolerance = objective.selection_tolerance if objective is not None else selection_tolerance
        best_error = min(selection_error(r) for r in eligible)
        close = [r for r in eligible if selection_error(r) <= best_error + tolerance]
        selected = min(close, key=lambda r: (
            len(r['result'].get('effective_free_parameter_ids', by_id[r['candidate_id']]['free_parameter_ids'])),
            (by_id[r['candidate_id']].get('source_partition_edit_count', 0) if objective is not None else 0),
            by_id[r['candidate_id']]['structure_edit_count'], selection_error(r), r['candidate_id']))
    all_visited = len(records) == len(candidates) and fatal is None
    complete = all_visited and all(r['result']['status'] == 'completed' for r in records)
    truncated = any(r['result']['status'] == 'budget_exhausted' for r in records)
    status = ('failed' if fatal or not selected else 'completed' if complete else
              'budget_exhausted' if truncated else 'partial')
    return {'method': STUDY_ALGORITHM if study_consistency else PURPOSE_ALGORITHM if objective is not None else ALGORITHM, 'status': status,
        'stop_reason': fatal or ('no_candidate_meets_study_consistency_guard' if study_consistency and not selected else None) or ('no_candidate_meets_requested_objective_and_local_diagnostics'
                                 if objective is not None and not selected else None),
        'selected_candidate': deepcopy(by_id[selected['candidate_id']]) if selected else None,
        'selected_result': deepcopy(selected['result']) if selected else None,
        'evaluations': records, 'history': history, 'solver_calls': calls,
        'solver_cost_complete': fatal is None,
        'registered_candidate_count': len(candidates), 'evaluated_candidate_count': len(records),
        'evaluated_structure_count': len({r['structure_id'] for r in records}),
        'enumeration_complete': all_visited, 'search_complete': complete,
        'candidate_budget': share if budget_strategy == 'equal' else None,
        'budget_strategy': budget_strategy, 'allocations': allocations,
        'unallocated_remainder': solver_budget - sum(budgets.values()),
        'objective': objective.model_dump(mode='json') if objective is not None else None,
        'pareto_candidate_ids': pareto_ids,
        'selection_consistency': consistency_report,
        'scientific_validation_performed': False}


def _purpose_frontier(records, by_id):
    """Keep trade-offs in verified PK errors AND complexity, without weights.

    Including complexity means a tiny error improvement cannot silently remove
    the simpler model before the declared selection tolerance is applied.
    An endpoint participates only when calculable for every eligible candidate.
    """
    if not records:
        return [], []
    keys = [key for key in ('curve_log_rmse', 'auc_log_error', 'cmax_log_error')
            if all(r['result']['selection_objective']['metrics'].get(key) is not None for r in records)]
    vectors = {}
    for row in records:
        candidate, fitted = by_id[row['candidate_id']], row['result']
        vectors[row['candidate_id']] = [fitted['selection_objective']['metrics'][key] for key in keys] + [
            len(fitted.get('effective_free_parameter_ids', candidate['free_parameter_ids'])),
            candidate.get('source_partition_edit_count', candidate['structure_edit_count']),
            candidate['structure_edit_count']]
    frontier = [row for row in records if not any(
        all(a <= b for a, b in zip(vector, vectors[row['candidate_id']], strict=True))
        and any(a < b for a, b in zip(vector, vectors[row['candidate_id']], strict=True))
        for name, vector in vectors.items() if name != row['candidate_id'])]
    return frontier, [r['candidate_id'] for r in frontier]


def _prediction_curve(directory, execution, target):
    """Verify the actual requested observer; zero at predose is permitted."""
    if not execution.get('engineering_accepted') or len(execution.get('curve_outputs', [])) != 1:
        raise RuntimeError('Requested model did not return one accepted native curve')
    path = (directory / execution['curve_outputs'][0]['path']).resolve(strict=True)
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('Native output path leaves its run directory')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    column = target + ' [µmol/l]'
    if not rows or column not in rows[0] or 'Time [min]' not in rows[0]:
        raise RuntimeError('Native output is missing the requested plasma observer')
    times = [float(row['Time [min]']) for row in rows]
    values = [float(row[column]) for row in rows]
    if (len(times) < 2 or any(not math.isfinite(v) for v in times + values)
            or any(v < 0 for v in values) or not any(v > 0 for v in values)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise RuntimeError('Native prediction must contain ordered finite times and nonnegative exposure')
    return {'time_min': times, 'concentration_umol_per_l': values,
            'target_output': target, 'source': str(path.relative_to(directory))}


def run_native_design(*, case, policy, output, project_root, execute=False,
                      max_solver_calls=500, device='cuda', cpu_threads=2,
                      scenario_workers=2, max_output_bytes=512 * 1024**2,
                      budget_strategy='fit_priority', resource_guard=None,
                      evaluation_fit_strategy=None):
    """Plan or execute one user request; never dispatch a comparison campaign.

    Calibration/selection inputs stay unchanged. The user prediction is compiled
    separately with the selected fitted values; it is never fitted as a new
    observation. No external observation file is opened by this entry point.
    An explicit evaluation optimizer override is restricted to purpose-driven
    execution. It leaves the candidate space, budget, diagnostic requirements,
    and selector unchanged; a coordinate control is not the historical method.
    """
    root = Path(project_root).resolve(strict=True)
    directory = Path(output)
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.resolve()
    if not directory.is_relative_to(root) or directory == root or directory.exists():
        raise ValueError('Choose a new output directory inside this project')
    if type(max_solver_calls) is not int or not 50 <= max_solver_calls <= 5000:
        raise ValueError('Use a total solver budget between 50 and 5000')
    if type(max_output_bytes) is not int or not 128 * 1024**2 <= max_output_bytes <= 1024**3:
        raise ValueError('Use an output limit between 128 MiB and 1 GiB')
    if device not in {'cpu', 'cuda'} or type(execute) is not bool:
        raise ValueError('Choose CPU/CUDA and an explicit boolean execution mode')
    if budget_strategy not in {'equal', 'fit_priority'}:
        raise ValueError('Unknown candidate budget strategy')
    if resource_guard is not None and not callable(resource_guard):
        raise ValueError('Resource guard must be a callable or None')
    validate_native_parallelism(cpu_threads=cpu_threads, scenario_workers=scenario_workers)
    validate_case(case)
    policy = NativeDesignPolicy.model_validate(policy)
    purpose_driven = policy.algorithm in {'purpose_driven', 'study_consistent'}
    study_consistent = policy.algorithm == 'study_consistent'
    if evaluation_fit_strategy is not None:
        if (not isinstance(evaluation_fit_strategy, str)
                or evaluation_fit_strategy not in {'coordinate', 'residual', 'screened_residual'}):
            raise ValueError('Evaluation fit strategy must be coordinate, residual, screened_residual, or None')
        if not purpose_driven or not execute:
            raise ValueError('An evaluation fit override requires purpose_driven execution')
    plan = build_design_plan(case, policy, project_root=root)
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / 'request.json', policy.model_dump(mode='json'))
    save(directory / 'development_case.json', case)
    save(directory / 'design_plan.json', plan)
    algorithm = (STUDY_ALGORITHM if study_consistent else PURPOSE_ALGORITHM if purpose_driven else
                 ALGORITHM if budget_strategy == 'fit_priority' else 'request-constrained-native-design-v1')
    if evaluation_fit_strategy == 'coordinate':
        algorithm += '-coordinate-control'
    elif evaluation_fit_strategy == 'screened_residual' and not study_consistent:
        algorithm += '-screened-control'
    elif evaluation_fit_strategy == 'residual' and study_consistent:
        algorithm += '-source-start-control'
    fit_strategy = evaluation_fit_strategy or ('screened_residual' if study_consistent else 'residual' if purpose_driven else 'coordinate')
    config = {'algorithm': algorithm, 'case_id': case.get('case_id'),
        'case_sha256': digest(case), 'request_sha256': digest(policy.model_dump(mode='json')),
        'execution_requested': execute, 'max_solver_calls': max_solver_calls,
        'max_output_bytes': max_output_bytes, 'device': device,
        'cpu_threads': cpu_threads, 'scenario_workers': scenario_workers,
        'selection_rule': 'Dsel log-RMSE within 0.02 of best; then fewer free parameters, fewer edits, lower error, stable ID',
        'budget_strategy': budget_strategy,
        # Residual fitting always starts at the registered source values. Keep
        # that same starting policy for its coordinate optimizer control.
        'fit_start_strategy': 'source_anchor' if purpose_driven or budget_strategy == 'fit_priority' else 'legacy',
        'residual_start_screening': fit_strategy == 'screened_residual',
        'study_consistency_required': study_consistent,
        'fit_strategy': fit_strategy,
        'evaluation_fit_strategy': evaluation_fit_strategy,
        'evaluation_override': ({'kind': 'optimizer_only',
            'reference_algorithm': STUDY_ALGORITHM if study_consistent else PURPOSE_ALGORITHM,
            'requested_fit_strategy': evaluation_fit_strategy,
            'is_legacy_algorithm': False} if evaluation_fit_strategy is not None else None),
        'require_identifiability': purpose_driven,
        'objective': policy.objective.model_dump(mode='json') if purpose_driven else None,
        'candidate_budget_rule': ('reserve build/predict costs for fixed candidates; equal remaining shares for fitted candidates'
                                  if budget_strategy == 'fit_priority' else 'equal fixed shares for all candidates'),
        'fit_roles': ['calibration'], 'selection_roles': ['selection'],
        'external_observations_read': False, 'scientific_validated': False}
    if purpose_driven:
        config['selection_rule'] = 'Meet requested PK bounds and local rank diagnostics; Pareto errors/complexity; primary objective then simpler model within declared tolerance'
    if study_consistent:
        config['selection_rule'] = 'No worse primary error than verified fixed source reference in every Dcal/Dsel study; then ' + config['selection_rule']
        config['consistency_limitation'] = 'Dcal is fitted; this is a conservative development guard, not unbiased cross-validation.'
    save(directory / 'configuration.json', config)

    def update(**fields):
        state.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
        save(directory / 'progress.json', state, replace=True)

    state = {'algorithm': algorithm, 'status': 'planned', 'stage': 'constraints',
             'scientific_validated': False, 'external_observations_read': False,
             'model_executed': False, 'delivery_complete': False}

    def output_bytes():
        if resource_guard is not None:
            resource_guard()
        total = 0
        for path in directory.rglob('*'):
            if path.is_symlink():
                raise ValueError('Design outputs must not contain symlinks')
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue  # An atomic progress temporary may have been renamed.
        if total > max_output_bytes:
            raise RuntimeError('Total design output limit exceeded; no further solver calls allowed')
        return total

    update()
    if not plan['supported']:
        update(status='blocked', stage='request_not_supported', reasons=plan.get('reasons', []))
        save(directory / 'result.json', state)
        return state
    specs = case['parameter_search']['parameters']
    prediction_request = plan['prediction_scenario']['design_request']
    if not execute:
        try:
            candidate = next(c for c in plan['candidates'] if c['candidate_id'] == plan['rule_candidate_id'])
            parameter_file = directory / 'development_case.json'
            provenance = {'provenance_id': str(uuid5(NAMESPACE_URL, str(parameter_file))),
                'provenance_kind': 'model_asset', 'source_id': str(parameter_file.relative_to(root)),
                'source_location': 'parameter_search.parameters.initial_value; original per-parameter sources retained',
                'source_checksum': file_sha(parameter_file)}
            variant = _compile(prediction_request, candidate, specs, root,
                values=[s['initial_value'] for s in specs], provenance=provenance)
            save(directory / 'draft_snapshot.json', variant['snapshot'])
            save(directory / 'draft_edits.json', variant['edit_manifest'])
            output_bytes()
            update(stage='planning_complete', candidate_count=len(plan['candidates']),
                   draft_candidate_id=candidate['candidate_id'], solver_calls=0,
                   draft_is_uncalibrated=True, model_executed=False)
        except (Exception, KeyboardInterrupt) as exc:
            update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                   stage='planning_failed', reason=f'{type(exc).__name__}: {exc}')
        save(directory / 'result.json', state)
        return state

    experiment = None
    ledger = BudgetLedger(max_solver_calls, diagnostic_fraction=0.2)
    context = {'ledger': ledger, 'model_requests': set(), 'output': directory}

    if purpose_driven:
        # Final checks must be affordable for any legal winner, before fitting
        # spends the search budget. Selected structures are already built.
        calibration_count = sum(s['role'] == 'calibration' for s in case['scenarios'])
        largest_free_set = max(len(c['free_parameter_ids']) for c in plan['candidates'])
        required_final = 4 * calibration_count * largest_free_set + len(case['scenarios']) + 1
        if required_final > ledger.diagnostics_limit:
            update(status='blocked', stage='insufficient_delivery_budget',
                   reason='The reserved budget cannot diagnose and deliver every legal candidate; increase max_solver_calls.',
                   required_final_solver_calls=required_final,
                   reserved_final_solver_calls=ledger.diagnostics_limit, budget=ledger.snapshot())
            save(directory / 'result.json', state)
            return state

    def candidate_update(**fields):
        output_bytes()
        candidate_status = fields.pop('status', None)
        if candidate_status is not None:
            fields['candidate_status'] = candidate_status
        update(**fields, status='running')

    try:
        verified = verify_development_observations(case, root)
        if purpose_driven:
            validate_objective_data(case, policy.objective)
        save(directory / 'development_sources.json', {'verified': verified, 'external_observations_read': False})
        update(status='running', stage='candidate_comparison')
        experiment = NativeExperiment(case=case, output=directory / 'fit', project_root=root,
            cpu_threads=cpu_threads, scenario_workers=scenario_workers, device=device,
            max_output_bytes=max_output_bytes - 64 * 1024**2, progress_callback=candidate_update,
            fit_start_strategy=config['fit_start_strategy'], fit_strategy=config['fit_strategy'],
            require_identifiability=purpose_driven)
        evaluate = experiment.evaluator(context)

        def bounded_evaluate(candidate, *, method_config):
            output_bytes()
            value = evaluate(candidate, method_config=method_config)
            output_bytes()
            return value

        search = compare_candidates(plan, bounded_evaluate, solver_budget=ledger.search_limit,
            budget_strategy=budget_strategy, scenario_count=len(case['scenarios']),
            objective=policy.objective if purpose_driven else None, case=case,
            study_consistency=study_consistent)
        search['method'] = algorithm
        save(directory / 'search_result.json', search)
        if study_consistent:
            save(directory / 'study_consistency.json', search['selection_consistency'])
        output_bytes()
        if context.get('infrastructure_failure'):
            raise RuntimeError('Native fitting failed; see candidate result records')
        candidate, fitted = search.get('selected_candidate'), search.get('selected_result')
        if candidate is None or fitted is None:
            update(status=search['status'], stage='no_selected_model', reason=search.get('stop_reason'),
                selection_consistency=search.get('selection_consistency'))
        else:
            update(stage='selected_parameter_diagnostics')
            diagnostic = experiment.bounded_diagnose(candidate, fitted['fitted_values'],
                fitted['effective_free_parameter_ids'], context,
                reserve_calls=len(case['scenarios']) + 1, phase='diagnostics')
            save(directory / 'selected_diagnostics.json', diagnostic)
            if purpose_driven and fitted['effective_free_parameter_ids'] and not (
                diagnostic.get('status') == 'evaluated' and diagnostic.get('step_stability_passed') is True
                and diagnostic.get('rank') == len(fitted['effective_free_parameter_ids'])):
                raise RuntimeError('Selected model parameter diagnostics could not be confirmed before delivery')
            update(stage='rebuild_selected_model')
            checks = experiment.rebuild_selected(candidate, fitted, context)
            save(directory / 'rebuild_checks.json', checks)
            parameter_file = directory / 'selected_parameters.json'
            provenance = {'provenance_id': str(uuid5(NAMESPACE_URL, str(parameter_file) + file_sha(parameter_file))),
                'provenance_kind': 'calibration_run', 'source_id': str(parameter_file.relative_to(root)),
                'source_location': 'fitted_values; calibrated on Dcal, selected on Dsel',
                'source_checksum': file_sha(parameter_file)}
            variant = _compile(prediction_request, candidate, specs, root,
                               values=fitted['fitted_values'], provenance=provenance)
            save(directory / 'selected_request.json', variant['request'])
            save(directory / 'selected_snapshot.json', variant['snapshot'])
            save(directory / 'selected_edits.json', variant['edit_manifest'])
            update(stage='predict_user_request')
            native_dir = directory / 'prediction'
            remaining_bytes = max_output_bytes - output_bytes() - 2 * 1024**2
            if remaining_bytes < 1024**2:
                raise RuntimeError('Insufficient total output budget for the requested prediction')
            execution = ledger.run(lambda: execute_snapshot(snapshot=variant['snapshot'],
                output_dir=native_dir, project_root=root, cpu_threads=cpu_threads,
                timeout_seconds=180, max_output_bytes=min(32 * 1024**2, remaining_bytes)), phase='diagnostics')
            curve = _prediction_curve(native_dir, execution, plan['prediction_scenario']['target_output'])
            save(directory / 'prediction.json', curve)
            pkml = [(native_dir / row['path']).resolve(strict=True) for row in execution['pkml_outputs']]
            if len(pkml) != 1 or not pkml[0].is_relative_to(native_dir):
                raise RuntimeError('User request did not produce one project-local native model')
            output_bytes()
            update(status=search['status'], stage='design_complete', model_executed=True, delivery_complete=True,
                selected_candidate_id=candidate['candidate_id'], selected_structure=candidate['structure_id'],
                free_parameter_ids=fitted['effective_free_parameter_ids'],
                parameters=dict(zip(experiment.ids, fitted['fitted_values'], strict=True)),
                native_model=str(pkml[0].relative_to(root)),
                prediction_file=str((directory / 'prediction.json').relative_to(root)),
                calibration_log_rmse=fitted['calibration_log_rmse'],
                selection_log_rmse=fitted['selection_log_rmse'],
                selection_objective=fitted.get('selection_objective'),
                selection_consistency=search.get('selection_consistency'),
                all_candidates_visited=search['enumeration_complete'], search_complete=search['search_complete'],
                parameter_diagnostics=diagnostic,
                limitations=['Development selection is not independent validation.',
                             'Historical source parameters may already have been fitted.',
                             'Only registered mechanisms and total parent plasma concentration are supported.'])
    except BudgetExceeded as exc:
        update(status='budget_exhausted', stage='incomplete_delivery', reason=str(exc), model_executed=False)
    except (Exception, KeyboardInterrupt) as exc:
        update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
               stage='incomplete_delivery', reason=f'{type(exc).__name__}: {exc}', model_executed=False)
    finally:
        if experiment is not None:
            experiment.close()
        update(budget=ledger.snapshot())
        save(directory / 'result.json', state)
    return state
