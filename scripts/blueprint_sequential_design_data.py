"""Time-only initial layouts and strict sequential development-data reveals.

All source documents stay with a data steward. Selection uses positive-time
row indices, never concentration values. The design API receives a compact
visible training document and an index/time menu; arbitrary source metadata,
raw curves, full-file references and clinical target outcomes are not exported.
Source hashes and study independence must be verified by the experiment runner.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral
import random

from blueprint_evidence_acquisition_data import _document
from blueprint_gap_design_data import _conditions, _molecular_weight


VERSION = "sequential-evidence-data-1"
SEEDS = (20261008, 20261009)
INITIAL_POINTS = 4
MAX_VISIBLE_POINTS = 7
MAX_LAYOUT_DRAWS = 10000
_TRAINING_FIELDS = ("compound_id", "study_id", "role", "matrix", "time_unit", "unit",
                    "molecular_weight_g_mol", "human", "dosing")


def _validated(document, request, name):
    rows = _document(document, name=name)
    _conditions(request)
    for key in ("compound_id", "human", "dosing"):
        if document[key] != request.get(key):
            raise ValueError(name + " conditions differ from request: " + key)
    if document["molecular_weight_g_mol"] != _molecular_weight(request):
        raise ValueError(name + " molecular weight differs from request")
    return rows


def _safe(document, rows):
    # Even optional labels/schema/selection metadata are omitted. Only fields
    # required to identify and fit this training set leave the source vault.
    result = {key: deepcopy(document[key]) for key in _TRAINING_FIELDS}
    result["observations"] = deepcopy(rows)
    return result


def _ids(ids, rows, *, minimum=INITIAL_POINTS, maximum=MAX_VISIBLE_POINTS):
    if not isinstance(ids, (list, tuple)) or not minimum <= len(ids) <= maximum:
        raise ValueError(f"Require between {minimum} and {maximum} explicit observation IDs")
    if any(isinstance(i, bool) or not isinstance(i, Integral) for i in ids):
        raise ValueError("Observation IDs must be original integer source-row indices")
    result = sorted(int(i) for i in ids)
    if len(set(result)) != len(result):
        raise ValueError("Duplicate observation IDs are not allowed")
    if any(i < 0 or i >= len(rows) or rows[i]["time_min"] <= 0 for i in result):
        raise ValueError("Observation IDs must identify positive-time source rows")
    return result


def _layouts(rows):
    positive = [i for i, row in enumerate(rows) if row["time_min"] > 0]
    if len(positive) < MAX_VISIBLE_POINTS:
        raise ValueError("Sequential budgets require at least seven positive-time observations")
    boundary = len(positive) // 2
    earlier, later = set(positive[:boundary]), set(positive[boundary:])
    layouts = []
    for seed in SEEDS:
        generator = random.Random(seed)
        for _ in range(MAX_LAYOUT_DRAWS):
            chosen = sorted(generator.sample(positive, INITIAL_POINTS))
            if set(chosen) & earlier and set(chosen) & later and chosen not in layouts:
                layouts.append(chosen)
                break
        else:
            raise RuntimeError("Deterministic layout draws exhausted without a distinct mixed layout")
    return layouts


def split_development(full_document, request, layout_index=0):
    """Export one of two fixed, distinct mixed four-point initial layouts.

    Each drug uses independent random.Random streams seeded with 20261008 and
    20261009. Rejected all-early/all-late or duplicate draws consume the same
    deterministic stream. The positive-time rank median defines the halves.
    Identical time grids therefore produce identical index layouts regardless
    of chemical identity or concentration values.
    """
    if isinstance(layout_index, bool) or not isinstance(layout_index, Integral) or layout_index not in (0, 1):
        raise ValueError("Layout index must be integer 0 or 1")
    rows = _validated(full_document, request, "Full development data")
    visible_ids = _layouts(rows)[int(layout_index)]
    visible_set = set(visible_ids)
    candidate_ids = [i for i, row in enumerate(rows) if row["time_min"] > 0 and i not in visible_set]
    return {"visible_training": _safe(full_document, [rows[i] for i in visible_ids]),
            "design_only": {"visible_ids": visible_ids,
                            "candidate_ids": candidate_ids,
                            "candidate_times_h": [rows[i]["time_min"]/60. for i in candidate_ids],
                            "visible_times_h": [rows[i]["time_min"]/60. for i in visible_ids]}}


def select_ids(full_document, request, ids):
    """Steward-only export of a registered 4–7-point source subset.

    This is deliberately not an action-policy interface: it accepts the source
    vault. The caller must lock each requested addition before using it.
    """
    rows = _validated(full_document, request, "Full development data")
    selected = _ids(ids, rows)
    return _safe(full_document, [rows[i] for i in selected])


def reveal(full_document, current_training, candidate_id, request):
    """Reveal one new source row; preserve every current value and condition."""
    full = _validated(full_document, request, "Full development data")
    current = _validated(current_training, request, "Visible development data")
    if current_training["study_id"] != full_document["study_id"]:
        raise ValueError("Visible development study differs from the full source")
    if not INITIAL_POINTS <= len(current) < MAX_VISIBLE_POINTS:
        raise ValueError("A reveal requires 4–6 visible points and may produce at most seven")
    by_time = {row["time_min"]: (i, row) for i, row in enumerate(full)}
    current_ids = []
    for row in current:
        found = by_time.get(row["time_min"])
        if found is None or found[1] != row or row["time_min"] <= 0:
            raise ValueError("Visible times and concentrations must exactly match full development rows")
        current_ids.append(found[0])
    if isinstance(candidate_id, bool) or not isinstance(candidate_id, Integral):
        raise ValueError("Candidate ID must be an integer source-row index")
    new_id = int(candidate_id)
    if new_id in current_ids:
        raise ValueError("Candidate observation is already visible")
    selected = _ids([*current_ids, new_id], full)
    return _safe(current_training, [full[i] for i in selected])
