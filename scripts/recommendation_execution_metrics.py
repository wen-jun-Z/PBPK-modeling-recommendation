"""Pure numerical summaries for frozen, scenario-matched PBPK execution outcomes.

No files, services, labels or acceptability thresholds are inferred here. The caller
must establish matching units, analyte, matrix, observation window, study role and
the scientific decision before providing outcomes. Reference adaptation is a
separate track, never evidence that the unmodified model was directly reusable.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


VERSION = "pbpk-execution-metrics-1.0"
TRACKS = {"direct", "reference_adaptation", "reference_diagnostic"}
SCIENTIFIC_STATES = {"validated", "incompatible", "reference_only", "unknown"}
EXECUTION_STATES = {"succeeded", "failed", "not_run"}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _exp(value):
    """Do not serialize infinity when a mathematically valid ratio overflows."""
    try:
        answer = math.exp(value)
    except OverflowError:
        return None
    return answer if math.isfinite(answer) and answer > 0 else None


def endpoint_error(predicted, observed, *, endpoint, bql=False):
    """Natural-log error for one positive, same-unit, same-definition scalar.

    Nonpositive/BQL/missing/nonfinite inputs are unknown, not zero errors or
    scientific negatives. AUC and Cmax definitions remain the caller's contract.
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a nonempty name")
    if type(bql) is not bool:
        raise ValueError("bql must be an explicit boolean")
    reason = ("below_quantification" if bql else "missing_value" if predicted is None or observed is None
              else "nonfinite_or_nonnumeric_value" if not _finite(predicted) or not _finite(observed)
              else "nonpositive_value" if predicted <= 0 or observed <= 0 else None)
    result = {"endpoint": endpoint, "state": "unknown" if reason else "valid", "reason": reason,
              "predicted": predicted if _finite(predicted) else None,
              "observed": observed if _finite(observed) else None,
              "ratio": None, "signed_log_error": None, "absolute_log_error": None,
              "fold_error": None, "ratio_or_fold_overflow": False}
    if reason:
        return result
    signed = math.log(predicted) - math.log(observed)
    ratio, fold = _exp(signed), _exp(abs(signed))
    result.update(ratio=ratio, signed_log_error=signed, absolute_log_error=abs(signed), fold_error=fold,
                  ratio_or_fold_overflow=ratio is None or fold is None)
    return result


def summarize_errors(records):
    """GMFE=exp(mean(abs(log(P/O)))); AFE=exp(mean(log(P/O)))."""
    records = list(records)
    valid = [r for r in records if r.get("state") == "valid"]
    if len({r.get("endpoint") for r in records}) > 1:
        raise ValueError("Different endpoints must be summarized separately")
    for row in valid:
        signed, absolute = row.get("signed_log_error"), row.get("absolute_log_error")
        if not _finite(signed) or not _finite(absolute) or absolute < 0 or not math.isclose(abs(signed), absolute):
            raise ValueError("Invalid endpoint-error record")
    signed = math.fsum(r["signed_log_error"] for r in valid) / len(valid) if valid else None
    absolute = math.fsum(r["absolute_log_error"] for r in valid) / len(valid) if valid else None
    return {"state": "computed" if valid else "unavailable", "record_count": len(records),
            "valid_count": len(valid), "unknown_count": len(records) - len(valid),
            "mean_signed_log_error": signed, "mean_absolute_log_error": absolute,
            "gmfe": _exp(absolute) if valid else None, "afe": _exp(signed) if valid else None,
            "weighting": "one equal weight per supplied valid record; no automatic pooling of studies or endpoints"}


def curve_metrics(observed_times, observed_values, predicted_times, predicted_values, *, below_quantification=None):
    """Score only explicit, exactly aligned positive observations/predictions.

    No interpolation, extrapolation, zero replacement or silent point deletion.
    Integrals cover [first observation, last observation]; peaks are sampled peaks,
    not native continuous Cmax. A time-zero concentration of zero invalidates these
    logarithmic metrics rather than being replaced by an arbitrary small value.
    """
    ot, ov, pt, pv = map(list, (observed_times, observed_values, predicted_times, predicted_values))
    result = {"state": "unavailable", "reason": None, "point_count": len(ot),
              "window": None, "curve": None, "observed_window_auc": None,
              "sampled_peak": None, "continuous_cmax": None,
              "policy": "strict observed-time alignment; no extrapolation; no BQL/nonpositive substitution"}
    if len(ot) < 2 or any(len(values) != len(ot) for values in (ov, pt, pv)):
        result["reason"] = "length_mismatch_or_fewer_than_two_points"
        return result
    if not all(_finite(t) and t >= 0 for t in ot + pt):
        result["reason"] = "invalid_time"
        return result
    if any(right <= left for times in (ot, pt) for left, right in zip(times, times[1:])):
        result["reason"] = "times_not_strictly_increasing"
        return result
    if ot != pt:
        result["reason"] = "time_grid_mismatch_no_implicit_interpolation_or_extrapolation"
        return result
    flags = [False] * len(ot) if below_quantification is None else list(below_quantification)
    if len(flags) != len(ot) or any(type(flag) is not bool for flag in flags):
        raise ValueError("below_quantification must have one explicit boolean per observation")
    errors = [endpoint_error(p, o, endpoint="concentration", bql=b) for p, o, b in zip(pv, ov, flags)]
    invalid = [i for i, row in enumerate(errors) if row["state"] != "valid"]
    if invalid:
        result.update(reason="invalid_concentration_points", invalid_points=[{"index": i, "reason": errors[i]["reason"]} for i in invalid])
        return result
    def integral(values):
        return math.fsum((b - a) * (x / 2 + y / 2) for a, b, x, y in zip(ot, ot[1:], values, values[1:]))
    try:
        auc_p, auc_o = integral(pv), integral(ov)
    except OverflowError:
        auc_p = auc_o = None
    result.update(state="computed", window={"start": ot[0], "end": ot[-1], "time_unit": "caller-matched"},
                  curve=summarize_errors(errors),
                  observed_window_auc=endpoint_error(auc_p, auc_o, endpoint="observed_window_auc"),
                  sampled_peak=endpoint_error(max(pv), max(ov), endpoint="sampled_peak"))
    return result


def _tie_groups(expected, ranking):
    if not isinstance(ranking, Mapping) or set(ranking) != set(expected):
        raise ValueError("ranking must contain exactly the complete expected candidate IDs")
    if not all(_finite(v) for v in ranking.values()):
        raise ValueError("Every ranking score must be finite")
    # IDs only identify outcomes. They are never used to decide a winner in a tie.
    scores = sorted(set(ranking.values()), reverse=True)
    return [[cid for cid in expected if ranking[cid] == score] for score in scores]


def _inclusion(groups, k):
    remaining, weights = k, {}
    for group in groups:
        take = min(max(remaining, 0), len(group))
        weights.update({cid: take / len(group) for cid in group})
        remaining -= take
    return weights


def _hit(groups, k, positives):
    no_hit, remaining = 1.0, k
    for group in groups:
        take = min(max(remaining, 0), len(group))
        negatives = sum(cid not in positives for cid in group)
        if take > negatives:
            return 1.0
        if take:
            no_hit *= math.comb(negatives, take) / math.comb(len(group), take)
        remaining -= take
    return 1.0 - no_hit


def _recall_bounds(weights, positives, unknown):
    """Tight expected-recall range over all binary completions of unknown IDs.

    For each possible positive count m among unknowns, extrema choose the m
    lowest/highest inclusion probabilities. No assumption about random missingness.
    """
    known = math.fsum(weights[c] for c in positives)
    low_weights = sorted(weights[c] for c in unknown)
    high_weights = list(reversed(low_weights))
    lower = upper = None
    low_sum = high_sum = 0.0
    for count in range(len(unknown) + 1):
        if count:
            low_sum += low_weights[count - 1]
            high_sum += high_weights[count - 1]
        denominator = len(positives) + count
        if denominator:
            lo, hi = (known + low_sum) / denominator, (known + high_sum) / denominator
            lower = lo if lower is None else min(lower, lo)
            upper = hi if upper is None else max(upper, hi)
    return {"lower": lower, "upper": upper, "conditional_on_at_least_one_positive": True,
            "zero_positive_completion_possible": not positives,
            "policy": "all binary unknown completions; uniform expected tie inclusion"}


def evaluate_selection(expected_candidate_ids, outcomes, ranking, *, ks=(1, 3, 5), track="direct", abstained=False,
                       endpoint_names=()):
    """Evaluate one fixed scenario and one frozen ranking, never rerank using truth.

    Outcomes have candidate_id, track (default direct), execution_status
    (succeeded/failed/not_run), scientific_status (validated/incompatible/
    reference_only/unknown), optional endpoints {name: {predicted, observed, bql}},
    and reason. Missing outcomes remain unknown. Validated requires successful
    execution plus at least one numeric endpoint; independent-study qualification
    and acceptance criteria must already have been checked by the caller.

    Reference_only is a known non-direct-positive category, not an execution
    failure. In the adaptation track it remains unknown until adaptation is tested.
    Incompatible is an explicitly evidenced scientific exclusion, never inferred
    from failed execution. Entirely unexecuted/failed pools produce no false gold.
    """
    expected = list(expected_candidate_ids)
    if not expected or any(not isinstance(c, str) or not c for c in expected) or len(set(expected)) != len(expected):
        raise ValueError("Expected candidate IDs must be nonempty, unique strings")
    if track not in TRACKS or type(abstained) is not bool:
        raise ValueError("Invalid track or abstained flag")
    ks = list(ks)
    if not ks or any(type(k) is not int or k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("ks must contain distinct positive integers")
    groups = _tie_groups(expected, ranking)
    endpoint_names = list(endpoint_names)
    if any(not isinstance(name, str) or not name for name in endpoint_names) or len(set(endpoint_names)) != len(endpoint_names):
        raise ValueError("endpoint_names must be unique nonempty strings")
    by_id, errors = {}, {}
    for original in outcomes:
        row = dict(original)
        cid = row.get("candidate_id")
        if cid not in expected or cid in by_id:
            raise ValueError("Unexpected or duplicate outcome candidate ID")
        if row.get("track", "direct") != track:
            raise ValueError("Direct and reference-adaptation outcomes must not be mixed")
        execution = row.get("execution_status", "not_run")
        state = row.get("scientific_status", "unknown")
        if execution not in EXECUTION_STATES or state not in SCIENTIFIC_STATES:
            raise ValueError("Invalid outcome status")
        endpoints = row.get("endpoints", {})
        if not isinstance(endpoints, Mapping):
            raise ValueError("endpoints must be a mapping")
        if any(not isinstance(pair, Mapping) for pair in endpoints.values()):
            raise ValueError("Each endpoint must supply a predicted/observed mapping")
        numerical = {name: endpoint_error(pair.get("predicted"), pair.get("observed"), endpoint=name,
                                          bql=pair.get("bql", False)) for name, pair in endpoints.items()}
        valid = {name: value for name, value in numerical.items() if value["state"] == "valid"}
        if valid and execution != "succeeded":
            raise ValueError("Failed/unexecuted runs cannot supply valid predictions")
        if state == "validated" and (execution != "succeeded" or not valid):
            raise ValueError("Validated requires successful execution and a valid endpoint")
        if state in {"incompatible", "reference_only"} and not row.get("reason"):
            raise ValueError("A definite exclusion/reference-only status requires its reason")
        if state == "incompatible" and execution == "failed":
            raise ValueError("Execution failure is not a scientific negative; retain unknown")
        row.update(execution_status=execution, scientific_status=state)
        by_id[cid], errors[cid] = row, numerical
    supplied_count = len(by_id)
    for cid in expected:
        if cid not in by_id:
            by_id[cid] = {"candidate_id": cid, "execution_status": "not_run", "scientific_status": "unknown", "reason": "missing_outcome"}
            errors[cid] = {}
    positives = {cid for cid in expected if by_id[cid]["scientific_status"] == "validated"}
    unknown = {cid for cid in expected if by_id[cid]["scientific_status"] == "unknown" or
               (track != "direct" and by_id[cid]["scientific_status"] == "reference_only")}
    complete = not unknown
    executed = {cid for cid in expected if by_id[cid]["execution_status"] == "succeeded"}
    evaluable_any = {cid for cid in expected if any(v["state"] == "valid" for v in errors[cid].values())}
    evaluable_all = {cid for cid in expected if all(errors[cid].get(name, {}).get("state") == "valid" for name in endpoint_names)} if endpoint_names else None
    evaluable = evaluable_all if endpoint_names else evaluable_any
    by_k = {}
    for k in ks:
        slots = min(k, len(expected))
        weights = _inclusion(groups, slots)
        hit = _hit(groups, slots, positives)
        possible_hit = _hit(groups, slots, positives | unknown)
        known_recall = math.fsum(weights[c] for c in positives) / len(positives) if positives else None
        by_k[str(k)] = {"requested_k": k, "ranking_slots": slots,
                       "executed_coverage": math.fsum(weights[c] for c in executed) / slots,
                       "evaluable_coverage": math.fsum(weights[c] for c in evaluable) / slots,
                       "evaluable_any_endpoint_coverage": math.fsum(weights[c] for c in evaluable_any) / slots,
                       "evaluable_all_requested_endpoints_coverage": math.fsum(weights[c] for c in evaluable_all) / slots if endpoint_names else None,
                       "known_positive_count": len(positives), "known_hit": hit,
                       "known_miss": 1 - hit if positives else None,
                       "known_positive_recall": known_recall,
                       "exact_recall": known_recall if complete else None,
                       "recall_bounds": _recall_bounds(weights, positives, unknown),
                       "hit_bounds": {"lower": hit, "upper": possible_hit},
                       "miss_bounds": {"lower": 1 - possible_hit, "upper": 1 - hit} if positives else None,
                       "inclusion_probabilities": weights}
    top = _inclusion(groups, 1)
    per_endpoint = {}
    for endpoint in sorted(set(endpoint_names) | {name for row in errors.values() for name in row}):
        valid = {cid: row[endpoint]["absolute_log_error"] for cid, row in errors.items()
                 if endpoint in row and row[endpoint]["state"] == "valid"}
        # Explicit out-of-scope/reference-only exclusions without this endpoint
        # are not silently assigned a numerical error. Selecting them leaves regret
        # unavailable and is counted by the separate coverage/decision metrics.
        required = {cid for cid in expected if by_id[cid]["scientific_status"] not in {"incompatible", "reference_only"} or cid in valid}
        if track != "direct":
            required |= {cid for cid in expected if by_id[cid]["scientific_status"] == "reference_only"}
        endpoint_complete = complete and required.issubset(valid)
        coverage = math.fsum(top[c] for c in valid)
        all_selected_known = all(cid in valid for cid, weight in top.items() if weight)
        selected = math.fsum(top[c] * value for c, value in valid.items()) / coverage if coverage else None
        best = min(valid.values()) if valid else None
        known_delta = max(0., selected - best) if selected is not None else None
        available = endpoint_complete and all_selected_known and not abstained
        if abstained:
            bounds = None
        elif all_selected_known:
            bounds = {"lower": known_delta, "upper": known_delta if endpoint_complete else selected}
        else:
            bounds = {"lower": 0., "upper": None, "upper_reason": "unknown_prediction_error_has_no_finite_bound"}
        per_endpoint[endpoint] = {"valid_candidate_count": len(valid), "required_candidate_count": len(required),
                                  "oracle_scope": "same-track, same-scenario successful numeric outputs; not only candidates that passed acceptance",
                                  "explicit_scope_exclusions": [cid for cid in expected if cid not in required],
                                  "endpoint_pool_complete": endpoint_complete,
                                  "top1_evaluable_probability": coverage,
                                  "selected_known_conditional_absolute_log_error": selected,
                                  "best_observed_absolute_log_error": best,
                                  "known_pool_conditional_regret": known_delta,
                                  "exact_full_pool_regret": known_delta if available else None,
                                  "full_pool_regret_bounds": bounds,
                                  "reason": None if available else "abstained" if abstained else
                                  "selected_candidate_endpoint_unknown" if not all_selected_known else "candidate_or_endpoint_pool_incomplete"}
    return {"schema_version": VERSION, "track": track, "candidate_count": len(expected),
            "processed_outcome_count": supplied_count,
            "scientific_pool_complete": complete, "known_positive_count": len(positives),
            "unknown_candidate_count": len(unknown), "abstained": abstained,
            "tie_policy": "uniform permutation within exact equal-score blocks; expected fixed-K inclusion, not ID tie-break",
            "coverage_scope": "raw ranking slots among the fixed expected pool, independent of abstention",
            "evaluable_coverage_definition": "all explicit endpoint_names valid" if endpoint_names else "at least one endpoint valid; no required endpoint set supplied",
            "required_endpoints": endpoint_names,
            "decision": {"exact_false_rejection": int(abstained) if complete and positives else None,
                         "exact_false_rejection_denominator": int(complete and bool(positives)),
                         "known_applicable_exists": bool(positives),
                         "no_applicable_established": complete and not positives,
                         "exact_correct_refusal": int(abstained) if complete and not positives else None,
                         "false_rejection_event_bounds": {"lower": int(abstained and bool(positives)),
                                                           "upper": int(abstained and bool(positives or unknown))}},
            "by_k": by_k, "endpoints": per_endpoint,
            "limitations": ["Scientific labels, units, study independence and acceptance thresholds are supplied, not established by this calculator.",
                            "Known-positive recall is not full-pool recall when any scientific outcome is unknown.",
                            "Endpoint regret is per endpoint in absolute natural-log-error units; no weighted composite.",
                            "No clinical prediction claim follows from technical execution success."]}
