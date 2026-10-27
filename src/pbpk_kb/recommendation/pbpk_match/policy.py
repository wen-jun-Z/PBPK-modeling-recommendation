"""Catalog identity screening, separate from learned scores and scientific labels.

This policy never reads grades, judge outputs, or predicted applicability. It
only decides whether a catalog identity is sufficiently specified for a direct
drug-specific request. Templates may refer to other drugs. Failing this screen
does not establish biological incompatibility or a negative scientific label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .data import normalize


DIRECT_TASKS = frozenset({
    "plasma_concentration_time_prediction",
    "special_population_prediction",
    "ddi_exposure_prediction",
})
TEMPLATE_TASKS = frozenset({
    "compound_parameterization_template",
    "mechanistic_template_recommendation",
})


def _compounds(record: Mapping[str, Any]) -> list[str]:
    """Same compounds precedence and text normalization as data.fields_for."""
    features = record.get("features") or {}
    if not isinstance(features, Mapping):
        raise ValueError("features must be an object when supplied")
    values = record.get("compounds", features.get("compounds", []))
    if values is None:
        values = []
    if not isinstance(values, (list, tuple, set)) or any(not isinstance(value, str) for value in values):
        raise ValueError("compounds must be a collection of strings, not a free-text string")
    return normalize(values)


def catalog_identity_policy(query: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Screen catalog identities while preserving all candidate positions.

    For direct tasks every explicitly required compound must be registered in
    a candidate. Empty query/candidate identities and partial coverage remain
    metadata uncertainties. Unsupported task types abstain conservatively.
    Templates are not filtered by drug identity. No source or medical claim is
    inferred from the presence or absence of a catalog string.
    """
    task = str(query.get("task_type") or "").strip()
    family = "direct" if task in DIRECT_TASKS else "template" if task in TEMPLATE_TASKS else "unsupported"
    required = _compounds(query)
    required_set = set(required)
    rows = []
    for index, candidate in enumerate(candidates):
        registered = _compounds(candidate)
        registered_set = set(registered)
        missing = sorted(required_set - registered_set)
        if family == "template":
            eligible, status = True, "template_reference_allowed"
            code = "template_task_does_not_require_same_compound"
            reason = "A template reference may concern another compound; scientific transferability still requires review."
        elif family == "unsupported":
            eligible, status = False, "task_not_supported"
            code = "unsupported_task_type"
            reason = "This task has no defined catalog identity policy; no candidate is automatically admitted."
        elif not required:
            eligible, status = False, "query_identity_unknown"
            code = "query_compounds_missing"
            reason = "The direct-use request has no explicit compound identity to check."
        elif not registered:
            eligible, status = False, "candidate_identity_unknown"
            code = "candidate_compounds_missing"
            reason = "The catalog does not register a compound identity for this candidate; applicability is unconfirmed."
        elif not missing:
            eligible, status = True, "identity_covered"
            code = "all_required_compounds_registered"
            reason = "The catalog registers every required compound; population, route, endpoints and scientific applicability still require review."
        elif required_set & registered_set:
            eligible, status = False, "partial_identity_coverage"
            code = "some_required_compounds_not_registered"
            reason = "Only some required compounds are registered; this catalog entry does not establish full identity coverage."
        else:
            eligible, status = False, "different_registered_compounds"
            code = "required_compounds_not_registered"
            reason = "The catalog registers other compounds and does not establish the requested direct-use identity."
        rows.append({
            "catalog_index": index,
            "candidate_id": str(candidate.get("candidate_id", index)),
            "eligible": eligible,
            "status": status,
            "reason_code": code,
            "reason": reason,
            "required_compounds": list(required),
            "registered_compounds": registered,
            "missing_required_compounds": missing,
            "metadata_only": True,
        })
    mask = [row["eligible"] for row in rows]
    abstain = not any(mask)
    if not rows:
        abstention_reason = "catalog_empty"
    elif family == "unsupported":
        abstention_reason = "unsupported_task_type"
    elif family == "direct" and not required:
        abstention_reason = "query_compounds_missing"
    elif abstain:
        abstention_reason = "no_catalog_candidate_covers_all_required_compounds"
    else:
        abstention_reason = None
    return {
        "policy_version": "catalog-identity-v1",
        "scope": {
            "task_type": task,
            "task_family": family,
            "required_compounds": required,
            "comparison": "casefold and whitespace normalization, matching fields_for; aliases are not inferred",
            "direct_rule": "All explicitly required compounds must be registered; partial or empty identity is not admitted.",
            "template_rule": "Cross-compound references remain eligible.",
            "scientific_applicability_assessed": False,
            "grade_or_probability_threshold_applied": False,
            "labels_used": False,
        },
        "eligible_mask": mask,
        "candidate_status": rows,
        "catalog_candidate_count": len(rows),
        "eligible_candidate_count": sum(mask),
        "abstain": abstain,
        "abstention_reason": abstention_reason,
        "needs_catalog_review": abstain,
        "needs_request_review": family == "unsupported" or (family == "direct" and not required),
    }


def _rank_intervals(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    low, high = np.empty(len(scores), dtype=int), np.empty(len(scores), dtype=int)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        low[order[start:end]], high[order[start:end]] = start + 1, end
        start = end
    return low, high


def select_policy_topk(scores, policy: Mapping[str, Any], k: int = 10) -> dict[str, Any]:
    """Return eligible top-K with boundary ties, plus the complete raw audit.

    All ranks are one-based inclusive intervals. Raw ranks are computed over
    the unchanged complete score array. Policy ranks refer only to eligible
    entries. Equal boundary scores are all returned, so items may exceed K.
    An empty eligible set returns no items, never an other-drug fallback.
    """
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k < 1:
        raise ValueError("k must be a positive integer")
    values = np.asarray(scores, dtype=float)
    mask = np.asarray(policy["eligible_mask"])
    statuses = policy["candidate_status"]
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be one finite vector in original catalog order")
    if mask.shape != values.shape or (mask.size and mask.dtype != np.dtype(bool)):
        raise ValueError("eligible_mask must be boolean and match the score vector")
    mask = mask.astype(bool)
    if len(statuses) != len(values):
        raise ValueError("candidate_status must preserve every catalog entry")
    if any(row.get("catalog_index") != i or bool(row.get("eligible")) != bool(mask[i]) for i, row in enumerate(statuses)):
        raise ValueError("candidate_status and mask disagree on original catalog positions")
    raw_low, raw_high = _rank_intervals(values)
    eligible = np.flatnonzero(mask)
    policy_low, policy_high = _rank_intervals(values[eligible])
    position = {int(index): j for j, index in enumerate(eligible)}
    audit = []
    for index, status in enumerate(statuses):
        pos = position.get(index)
        audit.append({
            **status,
            "score": float(values[index]),
            "raw_rank": int(raw_low[index]),
            "raw_rank_end": int(raw_high[index]),
            "policy_rank": int(policy_low[pos]) if pos is not None else None,
            "policy_rank_end": int(policy_high[pos]) if pos is not None else None,
        })
    if len(eligible):
        ordered = eligible[np.argsort(-values[eligible], kind="stable")]
        cutoff = float(values[ordered[min(k, len(ordered)) - 1]])
        selected = ordered[values[ordered] >= cutoff]
        items = [audit[int(index)] for index in selected]
    else:
        cutoff, items = None, []
    return {
        "items": items,
        "all_candidates": audit,
        "requested_top_k": int(k),
        "returned_count": len(items),
        "cutoff_score": cutoff,
        "cutoff_ties_retained": True,
        "tie_policy": "exact score ties retain their full boundary block; catalog order is presentation only",
        "catalog_candidate_count": len(values),
        "eligible_candidate_count": len(eligible),
        "abstain": not len(eligible),
        "abstention_reason": policy.get("abstention_reason") if not len(eligible) else None,
        "needs_catalog_review": not len(eligible),
        "scope": policy["scope"],
    }
