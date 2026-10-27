"""Small, fixed ridge ranker for retrospective sampling-value development.

Only pre-acquisition numerical features enter this module. It has no filesystem,
clinical-data or PBPK-fitting access. Scores rank a menu; they are NOT calibrated
absolute gains, uncertainty intervals or online stopping probabilities.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    'log_time_over_infusion', 'nearest_visible_log_time_distance',
    'mean_log_concentration_over_visible_max', 'concentration_log_variance',
    'linear_explained_goal_variance', 'nonlinear_explained_goal_variance',
    'local_c_optimal_fractional_reduction', 'after_infusion',
)
ALPHA = 1.0


def finite(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind == 'b' or np.iscomplexobj(raw):
        raise ValueError(name + ' must be finite real numbers')
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(name + ' must be finite real numbers')
    return arr


def features(*, times, visible_times, visible_max, infusion_end,
             concentrations, linear_utility, nonlinear_utility,
             goal_variance, c_optimal_scores, c_optimal_before):
    """Explicit whitelist: times, visible summaries and model-only predictions."""
    t = finite(times, 'times'); vt = finite(visible_times, 'visible_times')
    c = finite(concentrations, 'concentrations')
    lu = finite(linear_utility, 'linear utility'); nu = finite(nonlinear_utility, 'nonlinear utility')
    cs = finite(c_optimal_scores, 'c-optimal scores')
    if (t.ndim != 1 or len(t) < 1 or vt.ndim != 1 or len(vt) < 1
            or c.ndim != 2 or c.shape[1] != len(t) or len(c) < 1
            or any(v.shape != t.shape for v in (lu, nu, cs))):
        raise ValueError('Unaligned feature dimensions')
    scalars = finite([visible_max, infusion_end, goal_variance, c_optimal_before], 'scalars')
    if (np.any(t <= 0) or np.any(vt <= 0) or np.any(c <= 0)
            or np.any(scalars[:2] <= 0) or np.any(scalars[2:] < 0)):
        raise ValueError('Positive times/concentrations and nonnegative variances required')
    frac = lambda a, b: np.clip(a / b, 0., 1.) if b > 1e-15 else np.zeros_like(a)
    logc = np.log(c)
    x = np.column_stack((
        np.log(t / infusion_end),
        np.min(abs(np.log(t[:, None]) - np.log(vt[None, :])), axis=1),
        logc.mean(axis=0) - np.log(visible_max), logc.var(axis=0),
        frac(lu, goal_variance), frac(nu, goal_variance),
        frac(c_optimal_before - cs, c_optimal_before), (t > infusion_end).astype(float),
    ))
    return finite(x, 'features')


def row_weights(design_ids, drug_ids):
    designs, drugs = np.asarray(design_ids), np.asarray(drug_ids)
    if designs.ndim != 1 or drugs.shape != designs.shape or not len(designs):
        raise ValueError('Nonempty aligned design/drug IDs required')
    for d in set(designs):
        if len(set(drugs[designs == d])) != 1:
            raise ValueError('A design cannot belong to multiple drugs')
    weights = np.zeros(len(designs))
    for drug in sorted(set(drugs)):
        ds = sorted(set(designs[drugs == drug]))
        for d in ds:
            mask = designs == d
            weights[mask] = 1. / len(set(drugs)) / len(ds) / mask.sum()
    return weights


def fit_ranker(x, y, design_ids, drug_ids, *, alpha=ALPHA):
    x = finite(x, 'X'); y = finite(y, 'y'); ds = np.asarray(design_ids)
    if x.ndim != 2 or x.shape[1] < 1 or y.shape != (len(x),) or ds.shape != y.shape:
        raise ValueError('Unaligned training data')
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Ridge alpha must be positive')
    w = row_weights(design_ids, drug_ids)
    xc, yc = x.copy(), y.copy()
    # Eliminate drug/layout intercepts: only the ranking of its own menu matters.
    for d in sorted(set(ds)):
        mask = ds == d
        xc[mask] -= x[mask].mean(axis=0)
        yc[mask] -= y[mask].mean()
    scale = np.sqrt(w @ (xc * xc)); scale[scale < 1e-12] = 1.
    z = xc / scale
    coef = np.linalg.solve(z.T @ (w[:, None] * z) + alpha * np.eye(x.shape[1]), z.T @ (w * yc))
    return dict(version='within-menu-weighted-ridge-1', alpha=float(alpha),
                scale=scale.tolist(), coef=coef.tolist(), train_rows=len(x),
                train_drugs=len(set(drug_ids)), train_designs=len(set(design_ids)),
                semantics='Relative menu ranking; no absolute gain calibration')


def predict_scores(model, x):
    x = finite(x, 'X'); scale = finite(model['scale'], 'scale'); coef = finite(model['coef'], 'coef')
    if x.ndim != 2 or scale.shape != (x.shape[1],) or coef.shape != scale.shape or np.any(scale <= 0):
        raise ValueError('Unaligned prediction data')
    # Within-menu centering is unnecessary at prediction: its offset is common.
    return x / scale @ coef


def choose_index(scores, times):
    scores, times = finite(scores, 'scores'), finite(times, 'times')
    if scores.ndim != 1 or not len(scores) or times.shape != scores.shape or len(set(times)) != len(times):
        raise ValueError('Aligned distinct candidate times required')
    tied = np.flatnonzero(scores >= scores.max() - 1e-12)
    return int(min(tied, key=lambda i: times[i]))


def cluster_folds(rows):
    """Every row of a drug, and every drug of a shared study, stays together."""
    if not rows:
        raise ValueError('Empty rows')
    drug_cluster = {}
    for row in rows:
        drug, cluster = row['case_id'], row['cluster_id']
        if drug in drug_cluster and drug_cluster[drug] != cluster:
            raise ValueError('A drug has conflicting study groups')
        drug_cluster[drug] = cluster
    groups = sorted(set(drug_cluster.values()))
    if len(groups) < 2:
        raise ValueError('At least two clusters required')
    for group in groups:
        train = [i for i, row in enumerate(rows) if row['cluster_id'] != group]
        test = [i for i, row in enumerate(rows) if row['cluster_id'] == group]
        yield group, train, test


def local_derivatives(x, lower, upper, visible_times, candidate_times, predict, goals):
    """Four-coordinate central/boundary differences; both callables are model-only."""
    x, lower, upper = (finite(v, 'coordinates') for v in (x, lower, upper))
    if x.shape != (4,) or lower.shape != (4,) or upper.shape != (4,) or np.any(lower >= upper):
        raise ValueError('Four bounded coordinates required')
    if np.any(x < lower) or np.any(x > upper):
        raise ValueError('Point outside bounds')
    times = np.r_[visible_times, candidate_times]; order = np.argsort(times); inverse = np.argsort(order)
    if np.any(np.diff(times[order]) <= 0):
        raise ValueError('Distinct visible and candidate times required')
    def conc(v):
        c = finite(predict(v, times[order]), 'model concentration')[inverse]
        if np.any(c <= 0):
            raise ValueError('Nonpositive model concentration')
        return c
    base = conc(x); j, g = [], []
    for axis in range(4):
        a, b = x.copy(), x.copy()
        a[axis] = max(lower[axis], x[axis] - 1e-4)
        b[axis] = min(upper[axis], x[axis] + 1e-4)
        width = b[axis] - a[axis]
        ga, gb = finite(goals(a), 'model goals'), finite(goals(b), 'model goals')
        if np.any(ga <= 0) or np.any(gb <= 0):
            raise ValueError('Nonpositive model goal')
        j.append((np.log(conc(b)) - np.log(conc(a))) / width)
        g.append((np.log(gb) - np.log(ga)) / width)
    return base, np.asarray(j).T, np.asarray(g).T
