"""Compare existing per-study errors with a registered fixed source candidate.

This is a deterministic development-data guard, not cross-validation. No model
is fitted or evaluated here, and no calibration study becomes a held-out study.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real


_METRICS = {'curve': 'curve_log_rmse', 'observed_window_auc': 'auc_log_error',
            'sampled_cmax': 'cmax_log_error'}
_ROLES = ('calibration', 'selection')
_ABS_TOL = 1e-10
_REL_TOL = 1e-8


def _study_values(record, metric):
    result = record.get('result')
    if not isinstance(result, dict):
        raise ValueError('missing_candidate_result')
    values, seen_groups = {}, set()
    for role in _ROLES:
        assessment = result.get(role + '_objective')
        if not isinstance(assessment, dict) or assessment.get('valid') is not True:
            raise ValueError(role + '_objective_invalid')
        rows = assessment.get('study_metrics')
        if not isinstance(rows, list) or not rows:
            raise ValueError(role + '_study_metrics_missing')
        values[role] = {}
        seen_scenarios = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(role + '_study_metric_invalid')
            group = row.get('study_group_id')
            if not isinstance(group, str) or not group:
                raise ValueError(role + '_study_group_id_invalid')
            if group in seen_groups:
                raise ValueError('duplicate_or_cross_role_study_group:' + group)
            if 'role' in row and row['role'] != role:
                raise ValueError('study_role_mismatch:' + group)
            seen_groups.add(group)
            scenarios = row.get('scenario_ids')
            if (not isinstance(scenarios, list) or not scenarios
                    or any(not isinstance(s, str) or not s for s in scenarios)
                    or len(set(scenarios)) != len(scenarios)
                    or seen_scenarios.intersection(scenarios)):
                raise ValueError('study_scenario_ids_invalid:' + group)
            seen_scenarios.update(scenarios)
            error = row.get(metric)
            if isinstance(error, bool) or not isinstance(error, Real):
                raise ValueError('study_metric_not_finite_nonnegative:' + group)
            try:
                error = float(error)
            except (OverflowError, ValueError) as exc:
                raise ValueError('study_metric_not_finite_nonnegative:' + group) from exc
            if not math.isfinite(error) or error < 0:
                raise ValueError('study_metric_not_finite_nonnegative:' + group)
            values[role][group] = {'error': error, 'scenario_ids': sorted(scenarios)}
    return values


def apply_study_consistency(records, by_id, *, primary, source_action):
    """Return eligible record copies and a full, JSON-serializable audit report.

    Both roles must have exactly the source reference's studies and curves. Only
    the declared primary metric is compared, once per study. A worse value may
    pass solely for floating-point equality (absolute 1e-10, relative 1e-8).
    Existing objective/diagnostic eligibility is never granted by this guard.
    In particular, an invalid or ambiguous source reference blocks all choices.
    """
    report = {
        'status': 'blocked', 'reason': None, 'primary': primary,
        'metric': _METRICS.get(primary) if isinstance(primary, str) else None,
        'absolute_tolerance': _ABS_TOL, 'relative_tolerance': _REL_TOL,
        'rule': 'No study may have worse primary error than the fixed source reference, within floating-point tolerance',
        'source_baseline': None, 'candidate_reports': [],
        'eligible_candidate_ids': [], 'rejected_candidate_ids': [],
        'solver_calls': 0, 'cross_validation_performed': False,
        'independent_validation_performed': False,
        'limitations': [
            'Calibration errors are in-sample fitting errors, not held-out predictions.',
            'Selection studies are already used for design selection; this is not independent validation.',
            'Study counts do not increase with dose arms, curves, or concentration samples.',
            'No-worse development errors do not establish clinical acceptability or future-study generalization.',
        ],
    }

    def blocked(reason):
        report['reason'] = reason
        if isinstance(records, list):
            for row in records:
                identifier = row.get('candidate_id') if isinstance(row, dict) else None
                report['candidate_reports'].append({
                    'candidate_id': identifier, 'passed': False,
                    'reasons': ['reference_guard_unavailable:' + reason],
                    'study_differences': [], 'worst_delta': None,
                    'worst_regression': None,
                })
                if isinstance(identifier, str):
                    report['rejected_candidate_ids'].append(identifier)
        return [], report

    if report['metric'] is None:
        return blocked('unsupported_primary_objective')
    if not isinstance(source_action, str) or not source_action:
        return blocked('source_action_missing')
    if not isinstance(records, list) or not isinstance(by_id, dict):
        return blocked('invalid_guard_inputs')
    metric = report['metric']
    reference_ids = [identifier for identifier, candidate in by_id.items()
                     if isinstance(candidate, dict)
                     and candidate.get('action') == source_action
                     and candidate.get('free_parameter_ids') == []]
    if len(reference_ids) != 1:
        return blocked('source_reference_missing' if not reference_ids else 'source_reference_ambiguous')
    source_id = reference_ids[0]
    reference_candidate = by_id[source_id]
    if reference_candidate.get('candidate_id') != source_id:
        return blocked('source_reference_identity_mismatch')
    identifiers = [r.get('candidate_id') if isinstance(r, dict) else None for r in records]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        return blocked('invalid_evaluation_candidate_id')
    if len(set(identifiers)) != len(identifiers):
        return blocked('ambiguous_duplicate_evaluation')
    references = [r for r in records if r['candidate_id'] == source_id]
    if not references:
        return blocked('source_reference_not_evaluated')
    reference = references[0]
    if reference.get('selection_eligible') is not True:
        return blocked('source_reference_selection_ineligible')
    if (isinstance(reference.get('result'), dict)
            and reference['result'].get('effective_free_parameter_ids', []) != []):
        return blocked('source_reference_not_fixed')
    try:
        reference_values = _study_values(reference, metric)
    except ValueError as exc:
        return blocked('source_reference_invalid:' + str(exc))
    report['source_baseline'] = {
        'candidate_id': source_id, 'action': source_action,
        'free_parameter_ids': [],
        'purpose_eligible': reference.get('purpose_eligible') is True,
        'request_constraints_passed': {
            role: reference['result'][role + '_objective'].get('constraints_passed') is True
            for role in _ROLES},
        'study_counts': {role: len(reference_values[role]) for role in _ROLES},
        'study_values': reference_values,
    }
    eligible = []
    for row in records:
        identifier = row['candidate_id']
        details = {'candidate_id': identifier, 'passed': False, 'reasons': [],
                   'study_differences': [], 'worst_delta': None, 'worst_regression': None}
        if row.get('selection_eligible') is not True:
            details['reasons'].append('prior_selection_ineligible')
        if row.get('purpose_eligible') is not True:
            details['reasons'].append('prior_purpose_ineligible')
        candidate = by_id.get(identifier)
        if not isinstance(candidate, dict) or candidate.get('candidate_id') != identifier:
            details['reasons'].append('candidate_identity_mismatch')
        try:
            values = _study_values(row, metric)
            for role in _ROLES:
                if set(values[role]) != set(reference_values[role]):
                    raise ValueError(role + '_study_set_mismatch')
                for group, source in reference_values[role].items():
                    if values[role][group]['scenario_ids'] != source['scenario_ids']:
                        raise ValueError('study_scenario_set_mismatch:' + role + ':' + group)
            for role in _ROLES:
                for group, source in reference_values[role].items():
                    value, baseline = values[role][group]['error'], source['error']
                    delta = value - baseline
                    passed = value <= baseline or math.isclose(value, baseline, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
                    details['study_differences'].append({
                        'role': role, 'study_group_id': group,
                        'scenario_ids': source['scenario_ids'][:],
                        'metric': metric, 'candidate_error': value,
                        'source_error': baseline, 'delta': delta,
                        'no_worse': passed,
                    })
                    if not passed:
                        details['reasons'].append('worse_than_source:' + role + ':' + group)
            details['worst_delta'] = max(item['delta'] for item in details['study_differences'])
            details['worst_regression'] = max(0.0, details['worst_delta'])
        except ValueError as exc:
            details['reasons'].append(str(exc))
        details['passed'] = not details['reasons']
        report['candidate_reports'].append(details)
        if details['passed']:
            eligible.append(deepcopy(row))
            report['eligible_candidate_ids'].append(identifier)
        else:
            report['rejected_candidate_ids'].append(identifier)
    report['status'] = 'applied'
    report['reason'] = None if eligible else 'no_eligible_candidate_passes_study_consistency'
    return eligible, report
