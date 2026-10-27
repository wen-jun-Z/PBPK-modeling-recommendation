"""Explicit PK design objectives for selection; no fitting, file I/O or simulation.

Endpoint errors use the observation window and observation sampling times only.
They are averaged over arms within each study, then equally over studies. A
missing endpoint never silently changes the studies used by a requested metric.
The supplied curve log-RMSE remains the calibration/scoring runtime's value.
"""
from __future__ import annotations

import math
from collections import defaultdict
from numbers import Real
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PKDesignObjective(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)

    primary: Literal['curve', 'observed_window_auc', 'sampled_cmax'] = 'curve'
    max_curve_log_rmse: float | None = Field(default=None, ge=0)
    max_auc_fold_error: float | None = Field(default=None, ge=1)
    max_cmax_fold_error: float | None = Field(default=None, ge=1)
    selection_tolerance: float = Field(default=0.02, ge=0)

    @model_validator(mode='after')
    def require_explicit_curve_constraint(self):
        if self.primary != 'curve' and self.max_curve_log_rmse is None:
            raise ValueError('An endpoint primary objective requires explicit max_curve_log_rmse')
        return self


def _objective(value):
    return value if isinstance(value, PKDesignObjective) else PKDesignObjective.model_validate(value)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{label} must be a finite real number')
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must be a finite real number') from exc
    if not math.isfinite(result):
        raise ValueError(f'{label} must be a finite real number')
    return result


def _requirements(objective):
    return {
        'auc': objective.primary == 'observed_window_auc' or objective.max_auc_fold_error is not None,
        'cmax': objective.primary == 'sampled_cmax' or objective.max_cmax_fold_error is not None,
    }


def _observations(scenario):
    def concentration_unit(unit):
        return isinstance(unit, str) and unit.replace('µ', 'u').replace('μ', 'u') == 'umol/l'

    if 'time_unit' in scenario and scenario['time_unit'] != 'min':
        raise ValueError('Objective observations require time_unit=min')
    if 'unit' in scenario and not concentration_unit(scenario['unit']):
        raise ValueError('Objective observations require concentration unit umol/l')
    points = scenario.get('observations')
    if not isinstance(points, list) or not points:
        raise ValueError('Objective observations must be a nonempty list')
    rows = []
    for point in points:
        if not isinstance(point, dict):
            raise ValueError('Each objective observation must be an object')
        if 'unit' in point and not concentration_unit(point['unit']):
            raise ValueError('Observation concentration unit conflicts with umol/l')
        normalized = 'time_min' in point or 'concentration_umol_per_l' in point
        if normalized:
            if 'time' in point or 'value' in point:
                raise ValueError('One observation cannot mix normalized and legacy field names')
            time = _number(point.get('time_min'), 'Observation time')
            value = _number(point.get('concentration_umol_per_l'), 'Observation concentration')
        else:
            if scenario.get('time_unit') != 'min' or not concentration_unit(scenario.get('unit')):
                raise ValueError('Legacy time/value observations require explicit min and umol/l units')
            time = _number(point.get('time'), 'Observation time')
            value = _number(point.get('value'), 'Observation concentration')
        if time < 0 or value <= 0:
            raise ValueError('Observation times must be nonnegative and concentrations positive')
        rows.append((time, value))
    return rows


def _validate_scenarios(scenarios, objective):
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError('Objective scenarios must be a nonempty list')
    identifiers, roles = set(), {}
    needed = _requirements(objective)
    for scene in scenarios:
        if not isinstance(scene, dict):
            raise ValueError('Each objective scenario must be an object')
        identifier, group, role = (scene.get(key) for key in ('scenario_id', 'study_group_id', 'role'))
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError('Objective scenario identifiers must be nonempty and unique')
        if not isinstance(group, str) or not group:
            raise ValueError('Objective study group identifiers must be nonempty')
        if role not in {'calibration', 'selection'}:
            raise ValueError('Objective design data permits calibration/selection roles only')
        if group in roles and roles[group] != role:
            raise ValueError('One objective study cannot cross calibration/selection roles')
        identifiers.add(identifier)
        roles[group] = role
        observations = _observations(scene)
        distinct_times = len({time for time, _ in observations}) == len(observations)
        for endpoint, required in needed.items():
            flag = endpoint + '_evaluable'
            if flag in scene and type(scene[flag]) is not bool:
                raise ValueError(f'{identifier}: {flag} must be an explicit boolean')
            if required and scene.get(flag) is not True:
                raise ValueError(f'{identifier}: requested {endpoint} requires {flag}=True')
            if required and (len(observations) < 2 or not distinct_times):
                raise ValueError(f'{identifier}: requested {endpoint} requires at least two unique observation times')


def validate_objective_data(case: dict, objective: PKDesignObjective | dict) -> None:
    """Reject unavailable requested endpoints before search, without opening sources.

    Only ``case['scenarios']`` is inspected; external/holdout payloads are not read.
    Selection observations are used here only to verify declared endpoint eligibility
    and the data shape. No endpoint values or candidate choices are computed.
    The caller's case validation owns the requirement to supply both data roles.
    """
    objective = _objective(objective)
    if not isinstance(case, dict):
        raise ValueError('The objective case must be an object')
    _validate_scenarios(case.get('scenarios'), objective)


def _mean(values):
    return math.fsum(value / len(values) for value in values) if values else None


def _complete_mean(values):
    return _mean(values) if values and all(value is not None for value in values) else None


def _fold(error):
    if error is None:
        return None
    try:
        return math.exp(error)
    except OverflowError:
        return None


def _endpoint_errors(scene, rows, predictions):
    result = {'auc_log_error': None, 'cmax_log_error': None}
    if len(rows) < 2 or len({time for time, _ in rows}) != len(rows):
        return result
    ordered = sorted((time, value, prediction) for (time, value), prediction in zip(rows, predictions, strict=True))
    if scene.get('auc_evaluable') is True:
        def area(column):
            return math.fsum((right[0] - left[0]) * (left[column] / 2 + right[column] / 2)
                             for left, right in zip(ordered, ordered[1:]))
        observed, predicted = area(1), area(2)
        if observed > 0 and predicted > 0 and math.isfinite(observed) and math.isfinite(predicted):
            result['auc_log_error'] = abs(math.log(predicted) - math.log(observed))
    if scene.get('cmax_evaluable') is True:
        result['cmax_log_error'] = abs(math.log(max(predictions)) - math.log(max(value for _, value in rows)))
    return result


def assess_objective(scenarios: list[dict], aggregate_score: dict,
                     objective: PKDesignObjective | dict) -> dict:
    """Assess one fitted candidate without changing its fit or original curve loss.

    ``valid`` means requested metrics are computable and the curve summary agrees
    with its predictions. Candidate eligibility requires both ``valid`` and
    ``constraints_passed``. Unavailable optional endpoints remain null rather
    than averaging a subset.
    """
    objective = _objective(objective)
    metrics = dict.fromkeys(('curve_log_rmse', 'auc_log_error', 'cmax_log_error',
                             'auc_fold_error', 'cmax_fold_error'))
    result = {'valid': False, 'primary_error': None, 'metrics': metrics,
              'constraints_passed': False, 'violations': [], 'study_metrics': []}
    try:
        _validate_scenarios(scenarios, objective)
        if not isinstance(aggregate_score, dict) or aggregate_score.get('valid') is not True:
            raise ValueError('Aggregate curve score is invalid')
        curve = _number(aggregate_score.get('log_rmse'), 'Aggregate curve log-RMSE')
        if curve < 0:
            raise ValueError('Aggregate curve log-RMSE must be nonnegative')
        metrics['curve_log_rmse'] = curve
        score_rows = aggregate_score.get('scenarios')
        if not isinstance(score_rows, dict) or set(score_rows) != {s['scenario_id'] for s in scenarios}:
            raise ValueError('Aggregate score must contain exactly the requested scenarios')
        grouped = defaultdict(list)
        for scene in scenarios:
            identifier = scene['scenario_id']
            score = score_rows[identifier]
            if not isinstance(score, dict) or score.get('valid') is not True:
                raise ValueError(f'{identifier}: curve prediction is invalid')
            observations = _observations(scene)
            predictions = score.get('predictions_umol_per_l')
            if not isinstance(predictions, list) or len(predictions) != len(observations):
                raise ValueError(f'{identifier}: predictions must match all observation rows')
            predictions = [_number(value, 'Predicted concentration') for value in predictions]
            if any(value <= 0 for value in predictions):
                raise ValueError(f'{identifier}: predictions must be positive')
            row = _endpoint_errors(scene, observations, predictions)
            row['curve_log_mse'] = _mean([(math.log(prediction) - math.log(observed)) ** 2
                                          for (_, observed), prediction in zip(observations, predictions, strict=True)])
            row['scenario_id'] = identifier
            grouped[scene['study_group_id']].append(row)
        for group, rows in grouped.items():
            result['study_metrics'].append({
                'study_group_id': group, 'scenario_ids': [row['scenario_id'] for row in rows],
                'curve_log_rmse': math.sqrt(_mean([row['curve_log_mse'] for row in rows])),
                'auc_log_error': _complete_mean([row['auc_log_error'] for row in rows]),
                'cmax_log_error': _complete_mean([row['cmax_log_error'] for row in rows]),
            })
        actual_curve = math.sqrt(_mean([row['curve_log_rmse'] ** 2 for row in result['study_metrics']]))
        if not math.isclose(actual_curve, curve, rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError('Aggregate curve log-RMSE disagrees with the study-equal prediction error')
        for endpoint in ('auc', 'cmax'):
            error = _complete_mean([row[endpoint + '_log_error'] for row in result['study_metrics']])
            metrics[endpoint + '_log_error'] = error
            metrics[endpoint + '_fold_error'] = _fold(error)
            if _requirements(objective)[endpoint] and (error is None or metrics[endpoint + '_fold_error'] is None):
                raise ValueError(f'Requested {endpoint} is unavailable or nonfinite for the complete study set')
        primary_key = {'curve': 'curve_log_rmse', 'observed_window_auc': 'auc_log_error',
                       'sampled_cmax': 'cmax_log_error'}[objective.primary]
        result['primary_error'] = metrics[primary_key]
        result['valid'] = result['primary_error'] is not None
        for name, limit in (('curve_log_rmse', objective.max_curve_log_rmse),
                            ('auc_fold_error', objective.max_auc_fold_error),
                            ('cmax_fold_error', objective.max_cmax_fold_error)):
            if limit is not None and (metrics[name] is None or metrics[name] > limit):
                result['violations'].append(f'{name} exceeds requested maximum {limit}')
        result['constraints_passed'] = result['valid'] and not result['violations']
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        result['violations'].append(str(exc))
    return result
