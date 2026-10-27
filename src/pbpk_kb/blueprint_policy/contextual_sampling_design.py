"""Visible-context sampling with study-stable gain and a separate curve guard.

No clinical files, outcome readers or model fitting reside in this module.
Study-deletion lower scores are a robustness heuristic, not confidence bounds.
"""
from __future__ import annotations

import math
import numpy as np

from .learned_sampling_design import FEATURE_NAMES as ORIGINAL_FEATURES, finite

VERSION = 'visible-context-relative-gain-1'
FEATURE_NAMES = (*ORIGINAL_FEATURES, 'signed_visible_residual_endpoint_response',
    'absolute_visible_residual_endpoint_response', 'visible_rmse_times_time_gap',
    'visible_log_slope_times_relative_time')
ALPHA = 1.0
LOG_NOISE_SD = .1
LOG_TIME_BANDWIDTH = math.log(2)
LABEL_CAP = math.log(4)
MAX_CURVE_DEGRADATION = math.log(1.05)
TOL = 1e-12


def contextual_features(x, candidate_times, visible_times, observed, predicted,
                        committee_concentrations, committee_cmax):
    """Add exactly four predeclared interactions from four-point information.

    Cmax committee values must use the current metadata-only target contract.
    Neither the unseen fifth concentration nor any target concentration is an
    argument. A near-zero fitted residual is kept, not replaced by hidden data.
    """
    x = finite(x, 'original features')
    t, vt = finite(candidate_times, 'candidate times'), finite(visible_times, 'visible times')
    y, yp = finite(observed, 'visible observed'), finite(predicted, 'visible predicted')
    c, g = finite(committee_concentrations, 'committee concentrations'), finite(committee_cmax, 'committee Cmax')
    if (x.shape != (len(t), 8) or vt.shape != (4,) or y.shape != vt.shape or yp.shape != vt.shape
            or c.ndim != 2 or c.shape != (len(g), len(t)) or len(g) < 2
            or len(set(t)) != len(t) or len(set(vt)) != 4
            or any(np.any(a <= 0) for a in (t, vt, y, yp, c, g))):
        raise ValueError('Four visible points and aligned positive candidate committee required')
    r = np.log(y / yp)
    kernel_log = -.5 * ((np.log(t[:, None]) - np.log(vt[None, :])) / LOG_TIME_BANDWIDTH)**2
    kernel = np.exp(kernel_log - kernel_log.max(axis=1, keepdims=True))
    residual_proxy = kernel @ r / kernel.sum(axis=1)
    lc, lg = np.log(c), np.log(g)
    covariance = np.mean((lc-lc.mean(axis=0)) * (lg-lg.mean())[:, None], axis=0)
    response = covariance / (lc.var(axis=0)+LOG_NOISE_SD**2)
    signed = response * residual_proxy
    centered_t = np.log(vt)-np.mean(np.log(vt))
    slope = float(centered_t @ (np.log(y)-np.mean(np.log(y))) / (centered_t @ centered_t))
    distance = np.min(np.abs(np.log(t[:, None])-np.log(vt[None, :])), axis=1)
    extra = np.column_stack((signed, np.abs(signed), np.sqrt(np.mean(r*r))*distance,
                            slope*np.log(t/max(vt))))
    return np.column_stack((x, extra))


def fit(designs, labels, *, dimensions=12, alpha=ALPHA):
    """Fit separate Cmax/curve gains against time coverage; no menu intercept.

    Each drug, layout and candidate has equal nested weight. The time action
    has exactly zero features and label. Robust label clipping is fixed before
    evaluating outcomes; unclipped clinical errors remain the scored outcome.
    """
    if dimensions not in (8, 12) or len(designs) < 2 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('At least two training designs and 8/12 dimensions required')
    drug_ids = sorted({d['case_id'] for d in designs})
    case_designs = {cid:sum(d['case_id'] == cid for d in designs) for cid in drug_ids}
    rows, targets, weights = [], [], []
    seen = set()
    for d in designs:
        key = d['design_key']
        if key in seen:
            raise ValueError('Duplicate training menu')
        seen.add(key)
        x = finite(d['X'], 'X')[:, :dimensions]
        j = int(d['time_index'])
        if not 0 <= j < len(x) or x.shape[1] != dimensions:
            raise ValueError('Invalid time action or feature dimensions')
        lab = labels[key]
        errors = np.column_stack([finite(lab[e], e+' labels') for e in ('cmax', 'curve')])
        if errors.shape != (len(x), 2) or np.any(errors < 1-1e-9):
            raise ValueError('Aligned clinical fold errors required')
        rows.extend(x-x[j])
        targets.extend(np.clip(np.log(errors[j]/errors), -LABEL_CAP, LABEL_CAP))
        weights.extend([1/len(drug_ids)/case_designs[d['case_id']]/len(x)]*len(x))
    x, y, w = np.asarray(rows), np.asarray(targets), np.asarray(weights)
    scale = np.sqrt(w @ (x*x)); scale[scale < 1e-12] = 1.
    z = x/scale
    coef = np.linalg.solve(z.T @ (w[:, None]*z) + alpha*np.eye(dimensions), z.T @ (w[:, None]*y))
    return dict(version=VERSION, dimensions=dimensions, feature_names=list(FEATURE_NAMES[:dimensions]),
        scale=scale.tolist(), coef=coef.tolist(), alpha=float(alpha), label_cap=LABEL_CAP,
        train_case_ids=drug_ids, train_design_keys=sorted(seen),
        train_groups=sorted({d['drug_study_group'] for d in designs}), rows=len(x))


def train(designs, labels, *, dimensions=12, alpha=ALPHA):
    groups = sorted({d['drug_study_group'] for d in designs})
    if len(groups) < 3:
        raise ValueError('At least three training drug-study groups required')
    return dict(version=VERSION, full=fit(designs, labels, dimensions=dimensions, alpha=alpha),
        deletions=[dict(excluded_group=g, model=fit([d for d in designs if d['drug_study_group'] != g],
            labels, dimensions=dimensions, alpha=alpha)) for g in groups])


def predict(model, x, time_index):
    x = finite(x, 'X')[:, :model['dimensions']]
    return (x-x[int(time_index)])/np.asarray(model['scale']) @ np.asarray(model['coef'])


def choose(bundle, x, times, time_index, *, curve_guard=True, study_guard=True):
    times = finite(times, 'candidate times')
    if times.ndim != 1 or len(set(times)) != len(times) or np.any(times <= 0):
        raise ValueError('Distinct positive candidate times required')
    j = int(time_index)
    scores = [predict(bundle['full'], x, j)]
    if study_guard:
        scores.extend(predict(d['model'], x, j) for d in bundle['deletions'])
    scores = np.asarray(scores)
    if scores.shape[1:] != (len(times), 2) or not np.allclose(scores[:, j], 0, atol=1e-12):
        raise ValueError('Action alignment/time-reference invariant failed')
    robust = scores.min(axis=0)
    eligible = robust[:, 0] > TOL
    if curve_guard:
        eligible &= robust[:, 1] >= -MAX_CURVE_DEGRADATION
    candidates = np.flatnonzero(eligible)
    if len(candidates):
        best = robust[candidates, 0].max()
        selected = min((i for i in candidates if robust[i,0] >= best-TOL), key=lambda i:times[i])
        reason = 'largest_study_stable_gain_with_curve_guard' if curve_guard else 'largest_study_stable_gain'
    else:
        selected, reason = j, 'time_fallback_no_supported_gain'
    return dict(selected_index=int(selected), time_index=j, selected_time_h=float(times[selected]),
        changed=bool(selected != j), reason=reason, curve_guard=bool(curve_guard), study_guard=bool(study_guard),
        minimum_gain_predictions=robust.tolist(), full_gain_predictions=scores[0].tolist(),
        eligible=eligible.tolist(), semantics='Training-study robustness heuristic, not calibrated probability or clinical guarantee')
