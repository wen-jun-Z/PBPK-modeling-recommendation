"""Pure, auditable input policy for the adult single-IV plasma pilot.

This module neither reads sources nor runs simulations.  A caller must verify
source hashes, extraction accuracy and independence before supplying evidence.
``ready`` means construction inputs passed this policy, not clinical validity.
Source paths, compound names and any outcomes are never predictive features.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any

from pydantic import ValidationError

from pbpk_kb.blueprints.evidence_blueprint import (
    COMMON, UNITS, Evidence, EvidenceBlueprintRequest,
)

VERSION = "evidence-decision-1"
SUPPORTED_ENDPOINTS = frozenset({
    "plasma_concentration_time", "auc", "cmax", "tmax", "c_last",
    "terminal_half_life", "auc_inf", "clearance", "vz", "mrt", "vss",
})
METHODS = frozenset({"evidence_policy", "fixed_template", "rf_fill", "no_conflict_guard"})
FALLBACK_PARAMETERS = frozenset({"fraction_unbound", "total_clearance"})
CLEARANCES = frozenset({"total_clearance", "hepatic_clearance", "renal_clearance"})
PRIORITY = {"measured": 0, "derived": 0, "predicted": 1, "assumption": 2}


def _finite(value: Any, *, positive: bool = False, nonnegative: bool = False) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (not positive or value > 0)
            and (not nonnegative or value >= 0))


def _rf_qualified(record: dict) -> bool:
    method = record.get("prediction_method") or {}
    applicability = method.get("applicability")
    return (record.get("basis") == "predicted"
            and method.get("name") in {"random_forest", "RF"}
            and bool(method.get("version"))
            and method.get("training_overlap") is False
            and isinstance(applicability, dict)
            and applicability.get("human_IV_parameter_model") is True
            and applicability.get("target_structure_and_scaffold_excluded") is True)


def _record_issue(record: dict, compound_id: str) -> tuple[str | None, bool]:
    """Return issue and whether it blocks, rather than allowing quarantine."""
    try:
        Evidence.model_validate(record)
    except (ValidationError, TypeError, ValueError):
        return "invalid_evidence_schema", True
    parameter = record["parameter"]
    if record["compound_id"] != compound_id:
        return "compound_mismatch", True
    if parameter not in UNITS:
        return "unsupported_parameter", True
    if record["unit"] != UNITS[parameter]:
        return "noncanonical_unit", True
    value, context = record["value"], record.get("context", {})
    if record["basis"] == "predicted":
        prediction = record.get("prediction_method") or {}
        if not all(key in prediction for key in ("name", "version", "training_overlap", "applicability")):
            return "missing_prediction_metadata", True
        if not prediction["name"] or not prediction["version"] or not prediction["applicability"]:
            return "empty_prediction_metadata", True
    if parameter in CLEARANCES | {"fraction_unbound"}:
        if context.get("species") != "human":
            return "not_human_evidence", False
        if context.get("matrix") != "plasma":
            return "not_plasma_evidence", False
        if context.get("study_role") in {"target", "evaluation", "test", "validation"}:
            return "held_out_outcome_evidence", True
        if context.get("population_condition", "reference_adult") != "reference_adult":
            return "population_condition_mismatch", False
        if record["basis"] not in {"measured", "derived", "predicted"}:
            return "unsupported_pk_assumption", True
        if record["basis"] == "predicted" and not _rf_qualified(record):
            return "unqualified_pk_prediction", True
    if parameter in CLEARANCES:
        if context.get("route") not in {"iv", "intravenous"}:
            return "not_iv_clearance", False
        if context.get("clearance_type") in {"apparent", "CL/F", "intrinsic", "CLint"}:
            return "not_systemic_plasma_clearance", False
        if context.get("analyte", "parent") not in {"parent", "unchanged_parent"}:
            return "not_parent_clearance", False
        if not _finite(value, nonnegative=True):
            return "invalid_clearance_value", True
    elif parameter == "fraction_unbound":
        if not _finite(value, positive=True) or value > 1:
            return "invalid_fraction_unbound", True
    elif parameter in {"molecular_weight", "solubility", "intestinal_permeability"}:
        if not _finite(value, positive=True):
            return "invalid_positive_parameter", True
    elif parameter in {"logp", "solubility_ph"}:
        if not _finite(value):
            return "invalid_numeric_parameter", True
    elif parameter == "molecular_formula":
        tokens = re.findall(r"([A-Z][a-z]?)(\d*)", value) if isinstance(value, str) else []
        if (not tokens or "".join(a + n for a, n in tokens) != value
                or any(int(n or 1) <= 0 for _, n in tokens)):
            return "invalid_molecular_formula", True
    elif parameter == "pka_types":
        if (not isinstance(value, list) or any(
                not isinstance(item, dict) or set(item) != {"Type", "Pka"}
                or item["Type"] not in {"Acid", "Base"} or not _finite(item["Pka"])
                for item in value)):
            return "invalid_pka_specification", True
    elif parameter == "elimination_scope":
        if value not in {"hepatic_only", "renal_only", "hepatic_and_renal"}:
            return "invalid_elimination_scope", True
    elif parameter == "binding_partner":
        if value not in {"Albumin", "Alpha1AcidGlycoprotein", "Unknown"}:
            return "invalid_binding_partner", True
    return None, False


def _equal_value(left: Any, right: Any) -> bool:
    if _finite(left) and _finite(right):
        # Only numerical representation noise, not a biological tolerance.
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def decide(request: dict, fallback_evidence: list[dict], purpose: dict | None = None,
           method: str = "evidence_policy") -> dict:
    """Select qualified evidence without modifying inputs or reading any files.

    Equal-priority conflicting evidence blocks the main method even when RF is
    available.  The named conflict-guard ablation chooses first-in-input order;
    it never claims that this resolves scientific disagreement.  ``rf_fill`` is
    an explicit lower-evidence baseline, not an evidence-resolving strategy.
    """
    if method not in METHODS:
        raise ValueError("Unknown evidence-decision method: " + str(method))
    if not isinstance(request, dict) or not isinstance(fallback_evidence, list):
        raise TypeError("request must be a dict and fallback_evidence a list")
    reasons: list[str] = []
    actions: list[dict] = []
    purpose = deepcopy(purpose) if purpose is not None else {}
    if not isinstance(purpose, dict):
        raise TypeError("purpose must be a dict")
    endpoints = purpose.get("endpoints", request.get("target", {}).get("endpoints", []))
    if not isinstance(endpoints, list) or not all(isinstance(x, str) for x in endpoints):
        endpoints = []
        reasons.append("invalid_requested_endpoints")
    if not endpoints or len(endpoints) != len(set(endpoints)):
        reasons.append("empty_or_duplicate_requested_endpoints")
    supported = [x for x in endpoints if x in SUPPORTED_ENDPOINTS]
    unsupported = [x for x in endpoints if x not in SUPPORTED_ENDPOINTS]
    if unsupported:
        reasons.append("unsupported_endpoints_for_this_pilot")
    if purpose.get("population_condition", "reference_adult") != "reference_adult":
        reasons.append("unsupported_population_condition_for_this_pilot")
    if purpose.get("mechanism", "parent_plasma") != "parent_plasma":
        reasons.append("unsupported_mechanism_for_this_pilot")
    if purpose.get("dosing_regimen", "single_dose") != "single_dose":
        reasons.append("unsupported_regimen_for_this_pilot")
    if request.get("dosing", {}).get("route") != "intravenous":
        reasons.append("intravenous_only_for_this_pilot")

    compound = request.get("compound_id")
    if not isinstance(compound, str) or not compound:
        reasons.append("missing_compound_identity")
    groups: dict[str, list[dict]] = {}
    backups: dict[str, list[dict]] = {}
    evidence = request.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
        reasons.append("invalid_evidence_list")

    for origin, records, destination in (("request", evidence, groups),
                                          ("fallback", fallback_evidence, backups)):
        for index, record in enumerate(records):
            parameter = record.get("parameter") if isinstance(record, dict) else None
            entry = {"origin": origin, "index": index, "parameter": parameter}
            if origin == "fallback" and parameter not in FALLBACK_PARAMETERS:
                actions.append({**entry, "action": "ignore", "reason": "fallback_parameter_not_allowed"})
                continue
            issue, fatal = _record_issue(record, compound)
            if origin == "fallback" and issue is None and not _rf_qualified(record):
                issue, fatal = "fallback_must_be_qualified_rf", True
            if issue:
                actions.append({**entry, "action": "reject", "reason": issue})
                if fatal or method == "fixed_template":
                    reasons.append(f"{origin}:{parameter}:{issue}")
                continue
            destination.setdefault(parameter, []).append({"record": deepcopy(record), **entry})

    selected: list[dict] = []
    for parameter in sorted(set(groups) | (set(backups) if method != "fixed_template" else set())):
        primary, spare = groups.get(parameter, []), backups.get(parameter, [])
        if method == "fixed_template":
            candidates = primary
            if len(candidates) != 1:
                reasons.append(f"{parameter}:fixed_template_requires_single_record")
                continue
        elif method == "rf_fill" and parameter in FALLBACK_PARAMETERS and spare:
            candidates = spare
        else:
            candidates = primary or (spare if parameter in FALLBACK_PARAMETERS else [])
        if not candidates:
            continue
        best_rank = min(PRIORITY[x["record"]["basis"]] for x in candidates)
        best = [x for x in candidates if PRIORITY[x["record"]["basis"]] == best_rank]
        conflict = any(not _equal_value(best[0]["record"]["value"], x["record"]["value"]) for x in best[1:])
        if conflict and method != "no_conflict_guard":
            reasons.append(f"{parameter}:equal_priority_conflict")
            actions.append({"parameter": parameter, "action": "block", "reason": "equal_priority_conflict",
                            "sources": [x["record"]["source"] for x in best]})
            continue
        choice = best[0]
        selected.append(choice["record"])
        actions.append({"parameter": parameter, "origin": choice["origin"], "index": choice["index"],
                        "action": "select", "reason": ("conflict_guard_disabled_first_record" if conflict else
                            "explicit_rf_baseline" if method == "rf_fill" and choice["origin"] == "fallback" else
                            "qualified_rf_fallback" if choice["origin"] == "fallback" else "highest_priority_qualified_evidence"),
                        "source": deepcopy(choice["record"]["source"])})
        for other in primary + spare:
            if other is choice:
                continue
            actions.append({"parameter": parameter, "origin": other["origin"], "index": other["index"],
                            "action": "not_selected", "reason": "duplicate_or_lower_priority_or_baseline_override"})

    values = {record["parameter"]: record["value"] for record in selected}
    for missing in sorted(COMMON - values.keys()):
        reasons.append("missing_required_parameter:" + missing)
    scope = values.get("elimination_scope")
    present = values.keys() & CLEARANCES
    allowed = {"hepatic_only": ({"total_clearance"}, {"hepatic_clearance"}),
               "renal_only": ({"total_clearance"}, {"renal_clearance"}),
               "hepatic_and_renal": ({"total_clearance", "renal_clearance"},
                                      {"hepatic_clearance", "renal_clearance"})}
    if scope in allowed and present not in allowed[scope]:
        reasons.append("missing_or_ambiguous_clearance_allocation")
    renal = values.get("renal_clearance", values.get("total_clearance", 0) if scope == "renal_only" else 0)
    hepatic = values.get("hepatic_clearance", values.get("total_clearance", 0) - renal if scope != "renal_only" else 0)
    if hepatic < 0 or renal < 0 or hepatic + renal <= 0:
        reasons.append("invalid_clearance_allocation")
    net_renal = next((e for e in selected if e["parameter"] == "renal_clearance"), None)
    protect_renal = bool(net_renal and net_renal["basis"] in {"measured", "derived"} and renal > 0)
    normalized = deepcopy(request)
    normalized["evidence"] = deepcopy(selected)
    try:
        EvidenceBlueprintRequest.model_validate(normalized)
    except (ValidationError, TypeError, ValueError):
        reasons.append("invalid_selected_request_schema")
    reasons = list(dict.fromkeys(reasons))
    return {"policy_version": VERSION, "method": method, "status": "blocked" if reasons else "ready",
            "selected_evidence": selected, "actions": actions, "reasons": reasons,
            "supported_endpoints": supported, "unsupported_endpoints": unsupported,
            "purpose": {"endpoints": deepcopy(endpoints), "population_condition": purpose.get("population_condition", "reference_adult"),
                        "mechanism": purpose.get("mechanism", "parent_plasma"),
                        "dosing_regimen": purpose.get("dosing_regimen", "single_dose")},
            "calibration": {"allowed_parameters": ["lipophilicity", "clearance_scale"],
                            "protect_net_renal": protect_renal,
                            "clearance_scale_scope": "nonrenal_only" if protect_renal else "total_clearance",
                            "clearance_evidence_role": "construction_prior_not_target_patient_truth",
                            "lipophilicity_role": "effective_logMA_prior_not_fixed_measured_logP",
                            "fraction_unbound_role": "retain_selected_construction_input"},
            "limitations": ["source_integrity_and_extraction_must_be_checked_by_caller",
                            "ready_is_not_clinical_validation",
                            "no_target_outcomes_used",
                            "requested_NCA_endpoints_still_require_valid_prediction_windows"]}
