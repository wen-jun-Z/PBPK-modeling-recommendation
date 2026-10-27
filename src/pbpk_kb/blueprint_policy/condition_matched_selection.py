"""Pure, outcome-free parameter bundle selection for an explicit PK context.

Callers verify source receipts/extraction before this function. Compatibility
is a property of declared evidence and request conditions, never proof of
prediction accuracy. An explicitly assumed context remains an assumption.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json
import math

VERSION = "condition-matched-selection-1"
METHODS = frozenset({"joint", "independent", "unconstrained_search"})
PARAMETERS = frozenset({"total_clearance", "renal_clearance", "fraction_unbound", "vss", "terminal_half_life", "mrt"})
DEFINITIONS = {
    "total_clearance": {"systemic_total_plasma_clearance"},
    "renal_clearance": {"net_renal_plasma_clearance", "systemic_net_renal_plasma_clearance", "renal_parent_plasma_clearance"},
    "fraction_unbound": {"fraction_unbound_in_plasma"},
    "vss": {"steady_state_volume_of_distribution_plasma"},
    "terminal_half_life": {"terminal_plasma_half_life"},
    "mrt": {"systemic_plasma_mean_residence_time"},
}
MAX_COMBINATIONS = 100_000


def _number(value, *, zero=False):
    return (not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)
            and (value >= 0 if zero else value > 0))


def _route(value):
    return {"iv":"intravenous", "oral":"oral", "po":"oral",
            "intravenous":"intravenous", "not_applicable_ex_vivo":"ex_vivo", "ex_vivo":"ex_vivo"}.get(value,value)


def _normalize(record):
    """Normalize only explicit units; absolute CL/Vss require SOURCE weight."""
    p, value = record["parameter"], record.get("value")
    if not _number(value,zero=p=="renal_clearance"):
        return None, "invalid_numeric_value"
    unit = str(record.get("unit","")).lower().replace(" ","")
    weight = record["context"].get("weight_kg")
    if p in {"total_clearance","renal_clearance"}:
        factors = {"ml/min/kg":1., "l/h/kg":1000./60., "l/min/kg":1000.}
        if unit in {"ml/min","l/h","l/min"}:
            if not _number(weight):
                return None,"source_weight_required_for_absolute_clearance"
            factor = {"ml/min":1.,"l/h":1000./60.,"l/min":1000.}[unit]/weight
        elif unit in factors:
            factor = factors[unit]
        else:
            return None,"unsupported_clearance_unit"
        value, canonical = value*factor,"mL/min/kg"
    elif p=="fraction_unbound":
        if unit in {"fraction","unitless","1"}:
            pass
        elif unit in {"percent_unbound","%unbound"}:
            value /= 100.
        else:
            return None,"unsupported_fraction_unit"
        if value > 1:
            return None,"fraction_outside_0_1"
        canonical = "fraction"
    elif p=="vss":
        if unit=="l":
            if not _number(weight):
                return None,"source_weight_required_for_absolute_volume"
            value /= weight
        elif unit!="l/kg":
            return None,"unsupported_volume_unit"
        canonical = "L/kg"
    else:
        if unit in {"min","minutes"}:
            value /= 60.
        elif unit not in {"h","hours"}:
            return None,"unsupported_time_unit"
        canonical = "h"
    result = deepcopy(record)
    result.update(value=float(value),unit=canonical)
    result["normalization"] = {"input_value":record["value"],"input_unit":record["unit"],
        "source_weight_kg":weight,"target_weight_used":False}
    return result,None


def _base_issue(record, context):
    if not isinstance(record,dict) or not isinstance(record.get("candidate_id"),str) or not record["candidate_id"]:
        return "missing_candidate_id"
    if record.get("compound_id") != context["compound_id"]:
        return "compound_mismatch"
    p = record.get("parameter")
    if p not in PARAMETERS:
        return "unsupported_parameter"
    rc = record.get("context",{})
    if not isinstance(rc,dict):
        return "missing_source_context"
    if rc.get("species") != context["species"]:
        return "species_mismatch_or_missing"
    if rc.get("matrix") != context["matrix"]:
        return "matrix_mismatch_or_missing"
    if rc.get("analyte") != context["analyte"]:
        return "analyte_mismatch_or_missing"
    if rc.get("definition") not in DEFINITIONS[p]:
        return "parameter_definition_missing_or_incompatible"
    if record.get("basis") not in {"measured","derived"}:
        return "unsupported_evidence_basis"
    coupling = record.get("coupling_group")
    if coupling is not None and (not isinstance(coupling,str) or not coupling or
            not isinstance(record.get("grouped_parameters"),list) or
            p not in record["grouped_parameters"] or
            not set(record["grouped_parameters"]) <= PARAMETERS or
            record.get("pairing_level") not in {"group_summary","individual_paired"}):
        return "invalid_explicit_parameter_group"
    quality = record.get("quality_score")
    if not _number(quality,zero=True):
        return "missing_or_invalid_quality_score"
    study = rc.get("study_id")
    independence = record.get("independence",{})
    if not study or not isinstance(independence,dict) or independence.get("target_values_used") is not False:
        return "target_independence_not_declared"
    if independence.get("source_study") != study:
        return "source_study_identity_conflict"
    excluded = set(context.get("excluded_study_ids",[]))
    if context.get("target_study"):
        excluded.add(context["target_study"])
    if study in excluded or rc.get("study_role") in {"target","test","validation","evaluation"}:
        return "excluded_target_or_development_study"
    if independence.get("target_study") and study==independence["target_study"]:
        return "target_study_leakage"
    return None


def _context_issues(record, context):
    rc = record["context"]
    issues, reasons = [], []
    population = rc.get("population")
    if not population or population in {"unknown","unspecified"}:
        issues.append("source_population_unknown")
    elif population != context.get("population"):
        compatible = context.get("compatible_source_populations",[])
        if population not in compatible:
            issues.append("population_mismatch")
        else:
            reasons.append("explicitly_permitted_reference_population")
    else:
        reasons.append("population_matches_declared_context")
    target_route, source_route = _route(context.get("route")),_route(rc.get("route"))
    if record["parameter"]=="fraction_unbound":
        if not source_route:
            issues.append("source_route_or_binding_setting_unknown")
        elif source_route != target_route:
            reasons.append("fu_binding_evidence_transfer_across_route")
        binding = rc.get("binding",{})
        if binding.get("concentration_dependent") is True:
            measured_range = binding.get("concentration_range")
            required_range = context.get("concentration_range")
            if not isinstance(measured_range,dict) or not isinstance(required_range,dict):
                issues.append("concentration_dependent_fu_range_unverified")
            elif (measured_range.get("unit") != required_range.get("unit") or
                  not all(_number(x,zero=True) for x in [measured_range.get("lower"),measured_range.get("upper"),
                      required_range.get("lower"),required_range.get("upper")])):
                issues.append("incompatible_fu_concentration_range_units")
            elif not measured_range["lower"] <= required_range["lower"] <= required_range["upper"] <= measured_range["upper"]:
                issues.append("request_outside_fu_concentration_range")
    elif not source_route or source_route != target_route:
        issues.append("route_mismatch_or_unknown")
    # Only compare fields actually supplied by the request. Missing source
    # fields stay unknown and cannot be interpreted as normal/healthy.
    for key,value in context.get("conditions",{}).items():
        source_value = rc.get("conditions",{}).get(key)
        if source_value is None:
            issues.append("source_condition_unknown:"+key)
        elif source_value != value:
            issues.append("source_condition_mismatch:"+key)
    return issues,reasons


def _pair_issues(records, *, joint, coupled_fu=None):
    by = {r["parameter"]:r for r in records}
    issues, reasons = [], []
    total, renal = by.get("total_clearance"),by.get("renal_clearance")
    if renal:
        if not total:
            issues.append("renal_clearance_requires_total_clearance")
        elif renal["value"] > total["value"]+1e-12:
            issues.append("renal_clearance_exceeds_total_clearance")
        elif joint and (renal["context"].get("study_id"),renal["context"].get("arm_id")) != (
                total["context"].get("study_id"),total["context"].get("arm_id")):
            issues.append("total_and_net_renal_must_share_study_arm")
        elif joint and not total["context"].get("arm_id"):
            issues.append("clearance_pair_arm_unknown")
    same_arm_pairs = 0
    if joint:
        if total and coupled_fu and total["candidate_id"] in coupled_fu:
            fu = by.get("fraction_unbound")
            if fu is None or fu["candidate_id"] not in coupled_fu[total["candidate_id"]]:
                issues.append("preserve_explicit_cl_fu_group")
            else:
                reasons.append("preserved_explicit_cl_fu_group:"+str(total["coupling_group"]))
                if total["pairing_level"]=="group_summary":
                    reasons.append("group_summary_pairing_not_individual_correlation")
        for left,right in itertools.combinations(records,2):
            a,b = left["context"],right["context"]
            if a["study_id"]==b["study_id"]:
                if a.get("arm_id") and b.get("arm_id") and a["arm_id"] != b["arm_id"]:
                    issues.append("within_study_arm_mismatch")
                elif a.get("arm_id") and a.get("arm_id")==b.get("arm_id"):
                    same_arm_pairs += 1
                    reasons.append("same_study_arm:"+a["study_id"]+":"+a["arm_id"])
                else:
                    reasons.append("same_source_without_verified_arm_pairing")
            else:
                reasons.append("compatible_cross_study_supplement")
    return sorted(set(issues)),sorted(set(reasons)),same_arm_pairs


def _bundle(records,method,coupled_fu=None):
    selected = {r["parameter"]:r["candidate_id"] for r in records}
    payload = json.dumps({"selected":selected,"records":records},sort_keys=True,separators=(",",":"))
    issues,reasons,pairs = _pair_issues(records,joint=method=="joint",coupled_fu=coupled_fu)
    rank = (-pairs if method=="joint" else 0., -sum(r["quality_score"] for r in records),
            tuple(sorted(selected.values())))
    return {"id":"bundle_"+hashlib.sha256(payload.encode()).hexdigest()[:16],
        "selected":selected,"records":deepcopy(records),"compatibility":{"issues":issues,"reasons":reasons,
            "same_study_arm_pairs":pairs,"context_gate_applied":True,"study_arm_gate_applied":method=="joint"},
        "rank_key":rank}


def select_bundles(candidates,context,method="joint",limit=4):
    """Select auditable candidates without mutating inputs or opening files.

    context requires compound_id/species/matrix/analyte/population/route and
    target_study (or explicit excluded_study_ids). Population assumptions must
    be stated in population_basis and are propagated, never upgraded to facts.
    quality_score is source quality, not clinical prediction performance.
    """
    if method not in METHODS:
        raise ValueError("Unknown selection method")
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=256:
        raise ValueError("limit must be an integer in [1,256]")
    if not isinstance(context,dict) or any(not isinstance(context.get(k),str) or not context[k]
            for k in ("compound_id","species","matrix","analyte","population","route")):
        raise ValueError("An explicit drug/species/matrix/analyte/population/route context is required")
    if context["population"] in {"unknown","unspecified"}:
        raise ValueError("Unknown population cannot be interpreted as healthy")
    if not context.get("target_study") and not context.get("excluded_study_ids"):
        raise ValueError("Explicit target study exclusions are required")
    required = context.get("required_parameters",["total_clearance","fraction_unbound"])
    if not required or len(required)!=len(set(required)) or not set(required)<=PARAMETERS:
        raise ValueError("Invalid required parameter list")
    if not isinstance(candidates,list):
        raise TypeError("candidates must be a list")
    accepted,rejected,seen = [],[],set()
    for raw in candidates:
        cid = raw.get("candidate_id") if isinstance(raw,dict) else None
        if cid in seen:
            raise ValueError("Duplicate candidate identifiers are not permitted")
        seen.add(cid)
        issue = _base_issue(raw,context)
        record = None
        if not issue:
            record,issue = _normalize(raw)
        if issue:
            rejected.append({"candidate_id":cid,"stage":"base","reasons":[issue]})
            continue
        issues,reasons = _context_issues(record,context)
        record["context_compatibility"] = {"issues":issues,"reasons":reasons}
        if issues:
            rejected.append({"candidate_id":cid,"stage":"common_context","reasons":issues})
        else:
            accepted.append(record)
    accepted.sort(key=lambda r:r["candidate_id"])
    rejected.sort(key=lambda r:str(r["candidate_id"]))
    result = {"version":VERSION,"method":method,"context":deepcopy(context),
        "status":"ready","eligible":accepted,"rejected":rejected,"bundles":[],
        "common_context_rejections":[r for r in rejected if r["stage"]=="common_context"],
        "common_context_rejection_count":sum(r["stage"]=="common_context" for r in rejected),
        "single_record_context_gate_shared_by_all_methods":True,
        "observations_used":False,"accuracy_optimized":False,
        "interpretation":"Compatibility of declared evidence only; prediction accuracy requires independent scoring",
        "population_basis":context.get("population_basis","not_declared"),
        "reasons":[],"bundle_rejections":[]}
    groups = {p:[r for r in accepted if r["parameter"]==p] for p in sorted({r["parameter"] for r in accepted}|set(required))}
    missing = [p for p in required if not groups[p]]
    if missing:
        result.update(status="needs_evidence",reasons=["missing_required_parameter:"+p for p in missing])
        return result
    coupled_fu = {}
    for cl in groups.get("total_clearance",[]):
        if not cl.get("coupling_group") or "fraction_unbound" not in cl["grouped_parameters"]:
            continue
        related = [fu["candidate_id"] for fu in groups.get("fraction_unbound",[])
            if fu.get("coupling_group")==cl["coupling_group"]
            and "total_clearance" in fu.get("grouped_parameters",[])
            and (fu["context"]["study_id"],fu["context"].get("arm_id"))==(
                cl["context"]["study_id"],cl["context"].get("arm_id"))
            and bool(cl["context"].get("arm_id"))]
        if related:
            coupled_fu[cl["candidate_id"]]=set(related)
    result["explicit_group_preservation_rule"] = "If selected CL has an explicitly grouped eligible same-arm fu, retain that fu group; otherwise permit compatible cross-study supplementation"
    if method=="independent":
        chosen = []
        for parameter,values in groups.items():
            if not values:
                continue
            best_quality = max(r["quality_score"] for r in values)
            best = [r for r in values if r["quality_score"]==best_quality]
            if any(not math.isclose(r["value"],best[0]["value"],rel_tol=1e-9,abs_tol=1e-12) for r in best[1:]):
                result["reasons"].append("equal_priority_conflict:"+parameter)
            else:
                chosen.append(best[0])
        if result["reasons"]:
            result["status"]="blocked_conflict"
            return result
        combinations = [tuple(chosen)]
    else:
        count = math.prod(len(v) for v in groups.values())
        result["cartesian_combinations"] = count
        if count>MAX_COMBINATIONS:
            result.update(status="search_space_exceeded",reasons=["explicit_candidate_search_limit_exceeded"])
            return result
        combinations = itertools.product(*(groups[p] for p in sorted(groups)))
    valid = []
    for combination in combinations:
        bundle = _bundle(combination,method,coupled_fu)
        if bundle["compatibility"]["issues"]:
            result["bundle_rejections"].append({"id":bundle["id"],"selected":bundle["selected"],
                "reasons":bundle["compatibility"]["issues"]})
        else:
            valid.append(bundle)
    valid.sort(key=lambda b:b["rank_key"])
    result["valid_bundle_count"] = len(valid)
    result["bundles"] = valid[:limit]
    result["limit_truncated"] = len(valid)>limit
    if not valid:
        result.update(status="blocked_incompatible",reasons=["no_physically_and_methodologically_compatible_bundle"])
    return result
