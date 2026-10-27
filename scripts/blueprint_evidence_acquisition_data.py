"""Separate a development-data vault from a time-only acquisition interface.

These pure helpers never read files or target observations. The caller must
verify source hashes and the development/target-study split before invoking
them. Unknown source metadata is deliberately not forwarded to a predictor:
clinical source documents can contain the complete curve in raw_observations.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real


VERSION = "blueprint-one-point-acquisition-data-1"
ROLE = "calibration_data_reconstruction"
_FIELDS = (
    "schema_version", "compound_id", "compound_name", "study_id", "study_label",
    "role", "matrix", "time_unit", "unit", "molecular_weight_g_mol", "human", "dosing",
)


def _number(value, label, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or (value < 0 if zero else value <= 0)):
        raise ValueError(label + " must be finite and " + ("nonnegative" if zero else "positive"))
    return float(value)


def _document(document, *, name):
    if not isinstance(document, dict):
        raise ValueError(name + " must be a training document")
    for key in ("compound_id", "study_id"):
        if not isinstance(document.get(key), str) or not document[key].strip():
            raise ValueError(name + " requires a nonempty " + key)
    if (document.get("role") != ROLE or document.get("matrix") != "plasma"
            or document.get("time_unit") != "min" or document.get("unit") != "umol/l"):
        raise ValueError(name + " must declare development plasma observations in min and umol/l")
    human, dosing = document.get("human"), document.get("dosing")
    if not isinstance(human, dict) or not isinstance(dosing, dict):
        raise ValueError(name + " requires explicit human and dosing conditions")
    _number(human.get("weight_kg"), name + " body weight")
    if dosing.get("route") != "intravenous" or dosing.get("start_h", 0.) != 0.:
        raise ValueError(name + " requires an intravenous dose starting at zero")
    _number(dosing.get("dose_mg"), name + " dose")
    _number(dosing.get("infusion_minutes"), name + " infusion minutes", zero=True)
    _number(document.get("molecular_weight_g_mol"), name + " molecular weight")
    rows = document.get("observations")
    if not isinstance(rows, list) or not rows:
        raise ValueError(name + " requires explicit observations")
    clean, previous = [], -math.inf
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(name + " observation must be a dictionary")
        if row.get("time_unit", "min") != "min" or row.get("unit", "umol/l") != "umol/l":
            raise ValueError(name + " per-row units differ from the document")
        t = _number(row.get("time_min"), name + " time", zero=True)
        c = _number(row.get("concentration_umol_per_l"), name + " concentration")
        if t <= previous:
            raise ValueError(name + " observation times must be strictly increasing")
        previous = t
        # Only these two fields may enter the optimizer. A row may otherwise
        # carry a source curve, a raw concentration array or extraction notes.
        clean.append({"time_min": t, "concentration_umol_per_l": c})
    return clean


def _validated_pair(full_document, five_document, request=None):
    full_rows = _document(full_document, name="Full development data")
    five_rows = _document(five_document, name="Five-point development data")
    if len(five_rows) != 5:
        raise ValueError("Exactly five visible observations are required")
    if len(full_rows) <= 5:
        raise ValueError("At least one hidden development observation is required")
    for key in ("compound_id", "study_id", "human", "dosing", "molecular_weight_g_mol"):
        if full_document[key] != five_document[key]:
            raise ValueError("Full and five-point development conditions differ: " + key)
    if request is not None:
        if not isinstance(request, dict):
            raise ValueError("Request must be a dictionary")
        for key in ("compound_id", "human", "dosing"):
            if five_document[key] != request.get(key):
                raise ValueError("Development conditions differ from request: " + key)
        evidence = request.get("evidence")
        if not isinstance(evidence, list) or any(not isinstance(row, dict) for row in evidence):
            raise ValueError("Request requires an evidence list")
        mw = [row for row in evidence if row.get("parameter") == "molecular_weight"]
        if (len(mw) != 1 or mw[0].get("unit") != "g/mol"
                or _number(mw[0].get("value"), "Request molecular weight") != five_document["molecular_weight_g_mol"]):
            raise ValueError("One matching molecular weight in g/mol is required")
    full_by_time = {row["time_min"]: row for row in full_rows}
    for row in five_rows:
        if full_by_time.get(row["time_min"]) != row:
            raise ValueError("Five-point times and concentrations must exactly match full development rows")
    return full_rows, five_rows


def _safe_document(source, rows):
    result = {key: deepcopy(source[key]) for key in _FIELDS if key in source}
    result["observations"] = deepcopy(rows)
    return result


def split_development(full_document, five_document, request):
    """Return five visible values and a separate concentration-free time menu.

    ``candidate_ids`` are zero-based row indices in the immutable full source.
    The menu contains no source document, path, hash, concentration, residual,
    fit or endpoint. Modifying unselected concentrations cannot change it.
    Validation necessarily inspects the vault, but the design algorithm only
    receives the returned safe document and menu, never either source object.
    """
    if request is None:
        raise ValueError("An explicit development request is required")
    full_rows, five_rows = _validated_pair(full_document, five_document, request)
    observed = {row["time_min"] for row in five_rows}
    candidates = [(index, row["time_min"] / 60.) for index, row in enumerate(full_rows)
                  if row["time_min"] not in observed]
    return {
        "visible_training": _safe_document(five_document, five_rows),
        "design_only": {
            "observed_times_h": [row["time_min"] / 60. for row in five_rows],
            "candidate_times_h": [t for _, t in candidates],
            "candidate_ids": [index for index, _ in candidates],
        },
    }


def reveal_one(full_training, five_training, candidate_time_h):
    """Reveal exactly one registered development value after action locking.

    This function validates the same full/five source pair and returns six
    sorted observations. The caller is responsible for locking the selected
    action before invoking this function; there is no target-data interface.
    """
    t = _number(candidate_time_h, "Candidate time in hours", zero=True)
    full_rows, five_rows = _validated_pair(full_training, five_training)
    observed = {row["time_min"] / 60. for row in five_rows}
    if t in observed:
        raise ValueError("Candidate time is already visible")
    candidates = [row for row in full_rows if row["time_min"] / 60. == t]
    if len(candidates) != 1:
        raise ValueError("Candidate time is absent from the registered development menu")
    rows = sorted([*five_rows, candidates[0]], key=lambda row: row["time_min"])
    return _safe_document(five_training, rows)
