"""Study-jackknife guard for the frozen eight-feature sampling ranker.

Training labels are supplied by the caller from its eligible historical studies.
This module neither reads clinical data nor accepts held-out outcome labels at
selection. Its consensus check measures sensitivity to the training studies; it
does not guarantee a clinical improvement or represent a calibrated probability.
"""
from __future__ import annotations

import numbers

import numpy as np

from . import learned_sampling_design as ridge

VERSION = 'study-jackknife-time-fallback-1'
MARGIN_TOLERANCE = 1e-12
FEATURE_NAMES = ridge.FEATURE_NAMES


def _ids(values, n, name):
    arr = np.asarray(values)
    if arr.shape != (n,) or any(not isinstance(v, (str, np.str_)) or not v for v in arr):
        raise ValueError(f'{name} must contain aligned nonempty string IDs')
    return arr


def train(X, y, design_ids, drug_ids, cluster_ids, exclude_cluster=None):
    """Fit one full and one leave-study-out ranker per eligible study.

    Scaling is independently estimated within every training fold. Exclusion
    removes every drug belonging to a held-out study before inspecting numeric
    values or fitting. IDs are used only for grouping and provenance, never as
    predictive features. Only the caller can enforce an original-cohort whitelist.
    """
    x, labels = np.asarray(X), np.asarray(y)
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES)
            or len(x) == 0 or labels.shape != (len(x),)):
        raise ValueError('Aligned nonempty X with eight features and y required')
    ds = _ids(design_ids, len(x), 'design_ids')
    drugs = _ids(drug_ids, len(x), 'drug_ids')
    clusters = _ids(cluster_ids, len(x), 'cluster_ids')
    for drug in set(drugs):
        if len(set(clusters[drugs == drug])) != 1:
            raise ValueError('A drug cannot belong to conflicting study clusters')
    for design in set(ds):
        if len(set(drugs[ds == design])) != 1:
            raise ValueError('A design cannot belong to multiple drugs')
    if exclude_cluster is not None and exclude_cluster not in set(clusters):
        raise ValueError('Excluded study cluster was not present in training rows')
    eligible = np.flatnonzero(clusters != exclude_cluster)
    groups = sorted(set(clusters[eligible]))
    if len(groups) < 2:
        raise ValueError('At least two eligible study clusters required for jackknife')

    def fit(indices):
        model = ridge.fit_ranker(x[indices], labels[indices], ds[indices], drugs[indices], alpha=1.)
        membership = dict(row_indices=indices.tolist(),
                          drug_ids=sorted(set(drugs[indices])),
                          design_ids=sorted(set(ds[indices])),
                          cluster_ids=sorted(set(clusters[indices])))
        return model, membership

    full, members = fit(eligible)
    jackknife = []
    for group in groups:
        indices = eligible[clusters[eligible] != group]
        model, membership = fit(indices)
        jackknife.append(dict(excluded_cluster=group, model=model,
                              training_membership=membership))
    return dict(version=VERSION, feature_names=list(FEATURE_NAMES), alpha=1.,
                margin_tolerance=MARGIN_TOLERANCE, excluded_cluster=exclude_cluster,
                full_model=full, training_membership=members,
                jackknife_models=jackknife,
                semantics='Study sensitivity guard; no guarantee of clinical gain')


def choose(bundle, X, times, time_index):
    """Use the full-ranker proposal only if all study deletions prefer it.

    The reference action is supplied by the fixed time rule. Each margin compares
    the *same full-model proposal* with that reference; jackknife argmax agreement
    is not required. A tie or any nonpositive margin keeps the time-rule action.
    """
    x = ridge.finite(X, 'X'); t = ridge.finite(times, 'times')
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or len(x) == 0
            or t.shape != (len(x),) or np.any(t <= 0)
            or len(set(t)) != len(t)):
        raise ValueError('Eight-feature menu and aligned distinct positive times required')
    if isinstance(time_index, bool) or not isinstance(time_index, numbers.Integral) or not 0 <= time_index < len(x):
        raise ValueError('time_index must be a valid integer menu index')
    if bundle.get('version') != VERSION or len(bundle.get('jackknife_models', [])) < 2:
        raise ValueError('Complete study-jackknife bundle required')
    scores = ridge.predict_scores(bundle['full_model'], x)
    proposal = ridge.choose_index(scores, t)
    time_index = int(time_index)
    margins = [dict(model='full', excluded_cluster=None,
                    margin=float(scores[proposal] - scores[time_index]))]
    for item in bundle['jackknife_models']:
        js = ridge.predict_scores(item['model'], x)
        margins.append(dict(model='jackknife', excluded_cluster=item['excluded_cluster'],
                            margin=float(js[proposal] - js[time_index])))
    adopt = proposal != time_index and all(m['margin'] > MARGIN_TOLERANCE for m in margins)
    reason = ('unanimous_positive_margin' if adopt else
              'proposal_is_time_rule' if proposal == time_index else
              'nonpositive_or_tied_study_margin')
    selected = proposal if adopt else time_index
    return dict(selected_index=selected, proposed_index=proposal, time_index=time_index,
                selected_time=float(t[selected]), adopted=bool(adopt), reason=reason,
                margins=margins, full_scores=scores.tolist())
