"""Prefer the existing JSON adapter; bridge only to that JSON's proven native model.

An engineering fallback is selected using frozen source/design metadata before
any solver runs. Original catalogue identity, model coefficients, physiology
and mechanisms remain intact. No clinical outcomes enter selection.
"""
from copy import deepcopy
from pathlib import Path
import shutil

from . import snapshot_candidate, complete_candidate, json_native_source_proof

ROOT = complete_candidate.ROOT
VERSION = 'source-bound-json-native-candidate-v1'
ROUTE = 'verified_json_native_pksim5'
BLOCKED_ROUTE = 'blocked_json_native_bridge'
IMPLEMENTATION_FILES = tuple(sorted(set((*snapshot_candidate.IMPLEMENTATION_FILES,
    *complete_candidate.IMPLEMENTATION_FILES, *json_native_source_proof.IMPLEMENTATION_FILES,
    'scripts/recommendation_execution_repairs/json_native_candidate.py',
    'scripts/recommendation_execution_repairs/json_racemate_contract.py',
    'scripts/recommendation_execution_repairs/json_source_design.py',
    'scripts/recommendation_execution_repairs/json_dual_native.py'))))


def _eligible_failure(plan):
    reasons = plan.get('reasons')
    return plan.get('status') == 'blocked' and isinstance(reasons, list) and bool(reasons) and all(
        reason == 'Blocked:NO_SOURCE_SIMULATION_MATCHES_FIXED_EXECUTION_CONTRACT'
        or reason.startswith('Blocked:SOURCE_ADAPTER_BLOCKED:')
        or reason.startswith('ValueError:Output grid would reach or exceed the ')
        for reason in reasons)


def _preference(plan):
    # Across exact native partitions, preserve the complete executor's ordering.
    # Any source-only option precedes an isolation option; no solver/error used.
    key = plan.get('source_selection_sort_key')
    if not isinstance(key, list) or not key:
        raise ValueError('JSON native bridge needs the actual frozen source selection key')
    return (bool(plan.get('coadministration_isolation')), bool(plan.get('application_expansion')),
            tuple(key), plan['source_simulation'], plan['source_simulation_id'], plan['execution_source_path'])


def prepare_candidate(scenario, candidate, *, project_root=ROOT):
    root = Path(project_root).resolve(strict=True)
    if 'provenance' in candidate and 'catalog_provenance' in candidate and candidate['provenance'] != candidate['catalog_provenance']:
        raise ValueError('CONFLICTING_CANDIDATE_PROVENANCE_FIELDS')
    fallback = snapshot_candidate.prepare_candidate(scenario, candidate, project_root=root)
    if not _eligible_failure(fallback):
        return fallback
    provenance = candidate.get('provenance', candidate.get('catalog_provenance', {}))
    assessments, plans = [], []
    try:
        if Path(provenance.get('source_path', '')).suffix != '.json':
            raise ValueError('JSON native bridge requires original JSON catalogue source')
        proofs = json_native_source_proof.resolve_native_sources(provenance, project_root=root)
        if proofs.get('native_coverage_complete') is not True:
            raise ValueError('Full original source simulation coverage required before native selection; unbuilt=' + str(proofs.get('unbuilt_simulations', [])))
        for proof in proofs['candidates']:
            plan = complete_candidate.prepare_candidate(scenario, candidate, project_root=root,
                                                       verified_json_source=proof)
            assessments.append({'native_source': deepcopy(proof['native_source']),
                'status': plan['status'], 'reasons': list(plan['reasons']),
                'source_simulation': plan.get('source_simulation'),
                'excluded_source_simulations': deepcopy(plan.get('excluded_source_simulations', [])),
                'coadministration_isolation_assessment_failures': deepcopy(plan.get('coadministration_isolation_assessment_failures', [])),
                'application_expansion_assessment_failures': deepcopy(plan.get('application_expansion_assessment_failures', []))})
            if plan['status'] == 'ready':
                plans.append(plan)
        if plans:
            # The chosen complete plan carries its actual source proof, direct
            # selection key and complete bridge producer dependency union.
            return min(plans, key=_preference)
        fallback = deepcopy(fallback)
        fallback['json_native_bridge_assessment'] = {'status': 'blocked',
            'proof_source': proofs.get('source_original'),
            'new_native_build_required': proofs['new_native_build_required'],
            'rejected_native_artifacts': proofs.get('rejected_native_artifacts', []),
            'native_candidates': assessments,
            'clinical_outcomes_accessed': False}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        fallback = deepcopy(fallback)
        fallback['json_native_bridge_assessment'] = {'status': 'blocked',
            'reason': type(exc).__name__ + ':' + str(exc),
            'native_candidates': assessments, 'clinical_outcomes_accessed': False}
    # The extra assessment is part of this bridge's frozen output. Sending it
    # back through snapshot.execute_candidate would silently prepare a different
    # plan and discard that evidence.
    fallback.update(version=VERSION, adapter_route=BLOCKED_ROUTE,
        implementation_sha256=complete_candidate.implementation_hashes(root, verified_json_source=True))
    return fallback


def _execute_blocked_bridge(plan, output_dir, root, *, timeout_seconds, cpu_threads, stop_file):
    if (plan.get('version') != VERSION or plan.get('adapter_route') != BLOCKED_ROUTE
            or plan.get('status') != 'blocked'
            or plan.get('json_native_bridge_assessment', {}).get('status') != 'blocked'
            or any(plan.get(key) is not None and plan.get(key) is not False
                   for key in ('native_executed', 'prediction_locked'))
            or any(key in plan for key in ('prediction_file', 'prediction_lock', 'predicted_endpoints'))):
        raise ValueError('Blocked JSON native bridge cannot carry a prediction')
    output = complete_candidate.inside(output_dir, root, exists=False)
    if output.exists():
        raise FileExistsError('Never overwrite JSON native bridge attempts')
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 1800:
        raise ValueError('timeout_seconds must be 1..1800')
    if type(cpu_threads) is not int or cpu_threads not in (1, 2):
        raise ValueError('One or two CPU threads required')
    if stop_file is not None and complete_candidate.inside(stop_file, root, exists=False).exists():
        raise ValueError('PROJECT_STOP_REQUESTED')
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise OSError('At least 2 GiB free space required before candidate execution')
    if plan['implementation_sha256'] != complete_candidate.implementation_hashes(root, verified_json_source=True):
        raise ValueError('Blocked JSON native bridge implementation changed')
    output.mkdir(parents=True)
    snapshot_candidate._save(output / 'execution_plan.json', plan)
    result = {'version': VERSION, 'adapter_route': BLOCKED_ROUTE,
        'scenario_id': plan['scenario_id'], 'candidate_id': plan['candidate_id'],
        'execution_revision': plan['execution_revision'],
        'implementation_sha256': deepcopy(plan['implementation_sha256']),
        'status': 'blocked', 'reasons': deepcopy(plan['reasons']),
        'native_executed': False, 'prediction_locked': False,
        'sealed_observations_accessed': False, 'fitting_performed': False,
        'scientific_inapplicability_inferred': False,
        'execution_plan_sha256': complete_candidate.sha(output / 'execution_plan.json')}
    if plan['implementation_sha256'] != complete_candidate.implementation_hashes(root, verified_json_source=True):
        raise ValueError('Blocked JSON native bridge implementation changed during recording')
    snapshot_candidate._save(output / 'result.json', result)
    return result


def execute_candidate(scenario, candidate, output_dir, *, project_root=ROOT,
                      timeout_seconds=300, cpu_threads=1, stop_file=None, frozen_plan=None):
    plan = prepare_candidate(scenario, candidate, project_root=project_root)
    if frozen_plan is not None and plan != frozen_plan:
        raise ValueError('Frozen JSON native bridge plan changed')
    if plan.get('adapter_route') == BLOCKED_ROUTE:
        return _execute_blocked_bridge(plan, output_dir, Path(project_root).resolve(strict=True),
            timeout_seconds=timeout_seconds, cpu_threads=cpu_threads, stop_file=stop_file)
    if plan.get('adapter_route') == ROUTE:
        return complete_candidate.execute_candidate(scenario, candidate, output_dir,
            project_root=project_root, timeout_seconds=timeout_seconds, cpu_threads=cpu_threads,
            stop_file=stop_file, frozen_plan=plan, verified_json_source=plan['json_native_source_proof'])
    # If no proven native route exists, the original JSON attempt remains an
    # explicit engineering failure, never an inapplicability label.
    return snapshot_candidate.execute_candidate(scenario, candidate, output_dir,
        project_root=project_root, timeout_seconds=min(timeout_seconds, 600))
