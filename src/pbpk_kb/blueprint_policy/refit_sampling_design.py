"""Endpoint-directed acquisition using bounded hypothetical PBPK refits.

The finite generating scenarios are an engineering approximation, not a
posterior. Hypothetical refits start at visible-data fits and are local; the
final real-data fitter still uses its independent, full multistart budget.
Only callbacks, visible concentrations and candidate times enter this module.
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping
from numbers import Integral, Real
import numpy as np
from scipy.optimize import least_squares

VERSION = 'bounded-refit-sampling-1'
PENALTY = np.array([0., .1/math.log(10.), .1/math.log(100.), .1/math.log(10.)])


class BudgetExceeded(Exception):
    pass


def _positive(values, shape):
    try:
        a = np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError('Predictions must contain real numeric values') from exc
    if a.shape != shape or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError('Predictions must be positive, finite and shape matched')
    return a


def _budget(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError('A positive actual-call budget is required')
    return int(value)


def _model(model, *, scenarios=False):
    if not isinstance(model, Mapping) or not callable(model.get('predict')):
        raise ValueError('A model with a prediction callback is required')
    try:
        x, lo, hi = [np.array(model[k], dtype=float, copy=True) for k in ('x', 'lower', 'upper')]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('Four explicit bounded log coordinates are required') from exc
    if (any(v.shape != (4,) or not np.isfinite(v).all() for v in (x, lo, hi))
            or np.any(lo >= hi) or np.any(x < lo) or np.any(x > hi)):
        raise ValueError('Four finite initial log coordinates must be within strictly ordered bounds')
    result = dict(model, x=x, lower=lo, upper=hi)
    if scenarios:
        if not callable(model.get('goal')):
            raise ValueError('Each generating model requires an endpoint callback')
        members = model.get('members')
        if not isinstance(members, (list, tuple)) or not 1 <= len(members) <= 2:
            raise ValueError('Each model must supply one or two visible-data scenarios')
        clean = []
        for member in members:
            point = np.array(member, dtype=float, copy=True)
            if (point.shape != (4,) or not np.isfinite(point).all()
                    or np.any(point < lo) or np.any(point > hi)):
                raise ValueError('Generating scenarios must obey the four coordinate bounds')
            if any(np.array_equal(point, old) for old in clean):
                raise ValueError('Generating scenarios within a model must be distinct')
            clean.append(point)
        result['members'] = clean
    return result


def _goal(callback, x):
    answer = callback(np.array(x, dtype=float, copy=True))
    if isinstance(answer, (bool, np.bool_)) or not isinstance(answer, Real) or not math.isfinite(answer) or answer <= 0:
        raise ValueError('Endpoint unavailable: require a finite positive scalar')
    return float(answer)


def local_refit(model, times, values, budget, guard=None):
    """Bounded actual nonlinear local fit with the unchanged final fit loss."""
    budget = _budget(budget)
    if guard is not None and not callable(guard):
        raise ValueError('Guard must be callable')
    model = _model(model)
    times = np.array(times, dtype=float, copy=True)
    if (times.ndim != 1 or len(times) < 4 or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError('Require at least four ordered finite nonnegative visible times')
    values = _positive(values, times.shape)
    used = failures = 0
    best = None
    reasons = {}

    def residual(x):
        nonlocal used, failures, best
        if used >= budget:
            raise BudgetExceeded()
        if guard is not None:
            guard()
        used += 1
        try:
            predicted = _positive(model['predict'](np.array(x, copy=True), times.copy()), times.shape)
            errors = np.log(predicted) - np.log(values)
            r = np.r_[errors / math.sqrt(len(values)), PENALTY[1:]*x[1:]]
            objective = float(r @ r)
            if not math.isfinite(objective):
                raise ValueError('Nonfinite objective')
            if best is None or objective < best['objective']:
                best = dict(objective=objective, x=np.asarray(x).tolist())
            return r
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            failures += 1
            key = type(exc).__name__ + ': ' + str(exc)
            reasons[key] = reasons.get(key, 0) + 1
            return np.full(len(values)+3, 1e6)

    status = 'converged'
    try:
        opt = least_squares(residual, model['x'].copy(),
            bounds=(model['lower'], model['upper']), max_nfev=128,
            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        if not opt.success:
            status = 'optimizer_limit'
    except BudgetExceeded:
        status = 'call_budget_exhausted'
    if best is None:
        status = 'failed'
    return dict(best=best, calls=used, failures=failures, reasons=reasons, status=status)


def choose(models, selected_model, visible_times, visible_values, candidate_times,
           *, proportional_sd=.1, floor_fraction=.01, local_budget=128, guard=None):
    """Minimize simulated squared log endpoint error after local refitting.

    Each model carries 1--2 distinct visible-data parameter scenarios. Equal
    scenario weights and three Gaussian quadrature nodes are fixed in advance.
    No candidate is discarded for an unfavorable simulated loss. Numerical
    failures make it ineligible and are reported; selection returns status
    failed if every action is ineligible. The engineering additive floor
    uses ONLY visible values. The positive lognormal noise is median-preserving:
    c is the generating median, not the arithmetic mean concentration.
    """
    start = time.monotonic()
    local_budget = _budget(local_budget)
    if guard is not None and not callable(guard):
        raise ValueError('Guard must be callable')
    ts = np.array(visible_times, dtype=float, copy=True)
    if ts.ndim != 1:
        raise ValueError('Visible times must be a one-dimensional array')
    ys = _positive(visible_values, ts.shape)
    candidates = np.asarray(candidate_times, dtype=float)
    if (len(ts) < 4 or np.any(np.diff(ts) <= 0) or np.any(ts <= 0)
            or not np.isfinite(ts).all() or candidates.ndim != 1 or not len(candidates)
            or not np.isfinite(candidates).all() or np.any(candidates <= 0)
            or np.any(np.diff(candidates) <= 0) or set(ts) & set(candidates)):
        raise ValueError('Need four visible positive times and distinct sorted hidden times')
    if (not isinstance(models, (list, tuple)) or not 1 <= len(models) <= 2
            or isinstance(selected_model, (bool, np.bool_)) or not isinstance(selected_model, Integral)
            or not 0 <= selected_model < len(models)
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not math.isfinite(v)
                   for v in (proportional_sd, floor_fraction))
            or proportional_sd <= 0 or floor_fraction < 0):
        raise ValueError('Invalid models or declared measurement noise')
    models = [_model(m, scenarios=True) for m in models]
    calls = failed_calls = local_fits = budget_exhaustions = goal_calls = 0
    floor = float(floor_fraction * np.max(ys))
    if not math.isfinite(floor):
        raise ValueError('Visible-only additive floor must remain finite')
    scenarios = []
    for mi, model in enumerate(models):
        for x in model['members']:
            if guard is not None:
                guard()
            c = _positive(model['predict'](np.array(x, copy=True), candidates.copy()), candidates.shape)
            if guard is not None:
                guard()
            goal_calls += 1; g = _goal(model['goal'], x)
            scenarios.append(dict(model=mi, x=list(x), concentration=c, log_goal=math.log(g)))
    if not scenarios:
        raise ValueError('No visible-data generating scenarios')
    if guard is not None:
        guard()
    goal_calls += 1
    current = _goal(models[selected_model]['goal'], models[selected_model]['x'])
    before = float(np.mean([(math.log(current)-s['log_goal'])**2 for s in scenarios]))
    rows = []
    for j, t in enumerate(candidates):
        expected = 0.; reasons = []; predicted_goals = []
        for si, scenario in enumerate(scenarios):
            c = float(scenario['concentration'][j])
            log_terms = [0., 2 * math.log(proportional_sd)]
            if floor > 0:
                log_terms.append(2 * (math.log(floor) - math.log(c)))
            sd = math.sqrt(float(np.logaddexp.reduce(log_terms)))
            for node, weight in ((-math.sqrt(3.), 1/6), (0., 2/3), (math.sqrt(3.), 1/6)):
                try:
                    simulated = c if node == 0. else math.exp(math.log(c) + sd * node)
                    if not math.isfinite(simulated) or simulated <= 0:
                        raise ValueError('Unrepresentable positive hypothetical concentration')
                except (ValueError, ArithmeticError) as exc:
                    reasons.append(dict(scenario=si, node=node, reason=str(exc)))
                    continue
                times = np.r_[ts, t]; values = np.r_[ys, simulated]
                order = np.argsort(times); trials = []
                for mi, model in enumerate(models):
                    fit = local_refit(model, times[order], values[order], local_budget, guard)
                    local_fits += 1; calls += fit['calls']; failed_calls += fit['failures']
                    budget_exhaustions += int(fit['status'] == 'call_budget_exhausted')
                    if fit['best'] is not None:
                        trials.append((fit['best']['objective'], mi, fit['best']['x']))
                if not trials:
                    reasons.append(dict(scenario=si, node=node, reason='No feasible hypothetical fit'))
                    continue
                _, chosen, x = min(trials, key=lambda v: (v[0], v[1]))
                if guard is not None:
                    guard()
                goal_calls += 1
                try:
                    g = _goal(models[chosen]['goal'], x)
                    expected += weight * (math.log(g)-scenario['log_goal'])**2 / len(scenarios)
                    predicted_goals.append(g)
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    reasons.append(dict(scenario=si, node=node, reason=str(exc)))
        rows.append(dict(time_h=float(t), expected_log_endpoint_loss=None if reasons else expected,
                         expected_reduction=None if reasons else before-expected,
                         failures=reasons, simulated_endpoints=len(predicted_goals)))
    valid = [i for i, row in enumerate(rows) if row['expected_log_endpoint_loss'] is not None]
    ix = min(valid, key=lambda i: (rows[i]['expected_log_endpoint_loss'], candidates[i])) if valid else None
    return dict(version=VERSION, status='selected' if ix is not None else 'failed', selected_index=ix,
        initial_scenario_loss=before, candidates=rows, additive_floor_mg_l=floor,
        proportional_sd=proportional_sd, local_call_budget=local_budget,
        generating_scenarios=[dict(model=s['model'], x=s['x'], log_goal=s['log_goal'],
            concentration_mg_l=s['concentration'].tolist()) for s in scenarios],
        resources=dict(refit_calls=calls, failed_calls=failed_calls, local_fits=local_fits,
            budget_exhaustions=budget_exhaustions, goal_calls=goal_calls,
            scenario_calls=len(scenarios), wall_seconds=time.monotonic()-start),
        hidden_values_used=False, target_values_used=False,
        noise_convention='Median-preserving lognormal concentration noise; additive floor determined from visible values only.',
        approximation='Finite engineering scenarios and bounded local refits; final multistart refit may select another solution.')
