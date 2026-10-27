"""Finite-state blueprint searches; all scientific evaluation is delegated.

No solver, observations, model service, shared score cache, or filesystem is
accessed here. The caller must register only legal candidate states and enforce
the remaining *logical* solver budget inside ``evaluate``. Scores returned by
that callback must be from Dcal/Dsel, never the final test set.
"""
from __future__ import annotations

from copy import deepcopy
import math
import random
from typing import Any, Callable

from .design_diagnostics import BudgetExceeded


METHODS = frozenset({"M0", "M1", "M2", "M3", "M4", "M5", "M6", "A1", "A2", "A3", "E0"})
_SUCCESS = frozenset({"ok", "success", "completed", "passed", "evaluated"})


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _ids(value: Any, name: str) -> list[str]:
    if (not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value)
            or len(value) != len(set(value))):
        raise ValueError(f"{name} must contain unique nonempty identifiers")
    return sorted(value)


def _candidates(items: list[dict]) -> list[dict]:
    if not isinstance(items, list):
        raise ValueError("candidates must be a list")
    result, seen, feature_size = [], set(), None
    for item in items:
        candidate = deepcopy(item)
        if not isinstance(candidate, dict):
            raise ValueError("Each candidate must be a dictionary")
        for name in ("candidate_id", "structure_id"):
            if not isinstance(candidate.get(name), str) or not candidate[name].strip():
                raise ValueError(f"Candidate {name} is required")
        if candidate["candidate_id"] in seen:
            raise ValueError("Duplicate candidate_id")
        seen.add(candidate["candidate_id"])
        if set(candidate) & {"selection_log_rmse", "calibration_log_rmse", "identifiability", "solver_calls"}:
            raise ValueError("Candidate inputs must not carry previously evaluated scores")
        candidate["parameter_ids"] = _ids(candidate.get("parameter_ids"), "parameter_ids")
        candidate["free_parameter_ids"] = _ids(candidate.get("free_parameter_ids"), "free_parameter_ids")
        if not set(candidate["free_parameter_ids"]) <= set(candidate["parameter_ids"]):
            raise ValueError("Free parameters must belong to the candidate")
        if len(candidate["free_parameter_ids"]) > 6:
            raise ValueError("At most six free parameters are allowed")
        features = candidate.get("features", [])
        if not isinstance(features, list):
            raise ValueError("features must be a numeric list")
        candidate["features"] = [_number(x, "feature") for x in features]
        if feature_size is not None and len(features) != feature_size:
            raise ValueError("All feature vectors must have the same length")
        feature_size = len(features)
        for name in ("requirement_score", "evidence_score"):
            candidate[name] = _number(candidate.get(name, 0.0), name)
        edits = candidate.get("structure_edit_count", 0)
        if isinstance(edits, bool) or not isinstance(edits, int) or edits < 0:
            raise ValueError("structure_edit_count must be a nonnegative integer")
        candidate["structure_edit_count"] = edits
        result.append(candidate)
    return sorted(result, key=lambda c: c["candidate_id"])


def run_method_search(
    *, method: str, candidates: list[dict], evaluate: Callable[..., dict],
    seed: int = 17, max_candidates: int = 24, max_solver_calls: int = 5000,
    selection_tolerance: float = 0.02, base_candidate_id: str | None = None,
    rule_candidate_id: str | None = None, phase1_fraction: float = 0.5,
    qwen_proposer: Callable[..., Any] | None = None, beam_width: int = 4,
    exact_free_parameter_count: int | None = None,
) -> dict:
    """Run one method with its own immutable input copy and evaluation history.

    ``evaluate(candidate, *, method_config)`` returns ``status``, incremental
    ``solver_calls`` and, when successful, nonnegative Dcal/Dsel log-RMSEs.
    Optional ``effective_free_parameter_ids`` records a subset actually fitted.
    Identifiability feedback uses ``selected_parameter_ids`` from the result or
    its ``identifiability`` dictionary, without treating local rank as truth.

    ``qwen_proposer(candidates, *, seed)`` is called once for M2 and must return
    a registered ID (or a dictionary containing that ID). No fallback proposer
    is used. M5's first stage requires one all-parameter state per structure.
    In the preregistered count-matched mode it instead uses one common parameter
    subset of the requested size, fixed before any callback is invoked. Candidate
    anchors outside the filtered space block the run rather than being replaced.
    M0--M4/E0 retain the registered parameterization; M5's subset stage and
    M6/A1/A2 may use diagnostic feedback. Final diagnostics remain the caller's
    responsibility for every method, including A3.
    """
    if method not in METHODS:
        raise ValueError("Unknown search method")
    for name, value, maximum in (("max_candidates", max_candidates, 24),
                                 ("max_solver_calls", max_solver_calls, 5000),
                                 ("beam_width", beam_width, 24)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be between 1 and {maximum}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    tolerance = _number(selection_tolerance, "selection_tolerance", nonnegative=True)
    fraction = _number(phase1_fraction, "phase1_fraction")
    if not 0 < fraction < 1:
        raise ValueError("phase1_fraction must lie between zero and one")
    if (exact_free_parameter_count is not None and
            (type(exact_free_parameter_count) is not int or not 0 <= exact_free_parameter_count <= 6)):
        raise ValueError("exact_free_parameter_count must be None or an integer in 0..6")
    original_pool = _candidates(candidates)
    pool = [candidate for candidate in original_pool if exact_free_parameter_count is None
            or len(candidate["free_parameter_ids"]) == exact_free_parameter_count]
    by_id = {c["candidate_id"]: c for c in pool}
    blocked = None
    for name, anchor in (("BASE", base_candidate_id), ("RULE", rule_candidate_id)):
        if anchor is not None and anchor not in by_id:
            blocked = name + "_CANDIDATE_OUTSIDE_LEGAL_SPACE"
            break
    rng = random.Random(seed)
    shuffled = list(by_id)
    rng.shuffle(shuffled)
    random_order = {key: i for i, key in enumerate(shuffled)}
    records: list[dict] = []
    history: list[dict] = []
    visited: set[str] = set()
    calls, stop = 0, None
    locked_structure = None
    common_structure_parameter_ids = None
    forest_type = None
    if method == "M4" and not blocked:
        try:
            from sklearn.ensemble import RandomForestRegressor
            forest_type = RandomForestRegressor
        except ImportError:
            blocked = "RANDOM_FOREST_DEPENDENCY_UNAVAILABLE"

    def guidance(candidate: dict) -> tuple:
        return (0 if method == "A1" else -candidate["requirement_score"],
                0 if method == "A2" else -candidate["evidence_score"],
                random_order[candidate["candidate_id"]])

    def free_ids(record: dict) -> list[str]:
        return record["result"].get("effective_free_parameter_ids", by_id[record["candidate_id"]]["free_parameter_ids"])

    def best(rows: list[dict]) -> dict | None:
        valid = [row for row in rows if row["selection_eligible"]]
        if not valid:
            return None
        minimum = min(row["result"]["selection_log_rmse"] for row in valid)
        close = [row for row in valid if row["result"]["selection_log_rmse"] <= minimum + tolerance]
        return min(close, key=lambda row: (
            len(free_ids(row)), by_id[row["candidate_id"]]["structure_edit_count"],
            row["result"]["selection_log_rmse"], row["candidate_id"]))

    def attempt(candidate: dict, phase: str, reason: str, *, phase_call_limit: int | None = None) -> None:
        nonlocal calls, stop
        if stop or candidate["candidate_id"] in visited:
            return
        if len(records) >= max_candidates or calls >= max_solver_calls:
            stop = "candidate_limit" if len(records) >= max_candidates else "solver_budget"
            return
        allowance = max_solver_calls - calls
        if phase_call_limit is not None:
            allowance = min(allowance, phase_call_limit - calls)
        if allowance <= 0:
            return
        diagnostic_adaptation = method in {"M6", "A1", "A2"} or (
            method == "M5" and phase == "parameter_subset")
        config = {
            "method": method, "seed": seed, "phase": phase,
            "identifiability_prescreen": diagnostic_adaptation and exact_free_parameter_count is None,
            "identifiability_feedback": diagnostic_adaptation,
            "fixed_parameterization": not diagnostic_adaptation or exact_free_parameter_count is not None,
            "exact_free_parameter_count": exact_free_parameter_count,
            "requirement_guidance": method != "A1", "evidence_guidance": method != "A2",
            "remaining_solver_calls": allowance,
        }
        visited.add(candidate["candidate_id"])
        try:
            value = evaluate(deepcopy(candidate), method_config=deepcopy(config))
        except BudgetExceeded:
            # No trustworthy cost is returned: stop instead of spending again.
            stop = "external_budget_exhausted"
            history.append({"candidate_id": candidate["candidate_id"], "phase": phase,
                            "reason": reason, "event": stop, "cost_reported": False})
            return
        except Exception as exc:
            stop = "evaluator_exception"
            history.append({"candidate_id": candidate["candidate_id"], "phase": phase,
                            "reason": reason, "event": stop, "exception_type": type(exc).__name__,
                            "cost_reported": False})
            return
        if not isinstance(value, dict):
            raise ValueError("Evaluator must return a dictionary with incremental solver_calls")
        value = deepcopy(value)
        increment = value.get("solver_calls")
        if isinstance(increment, bool) or not isinstance(increment, int) or increment < 0:
            raise ValueError("Evaluator solver_calls must be a nonnegative integer increment")
        calls += increment
        eligible = value.get("status") in _SUCCESS or (
            value.get("status") == "budget_exhausted" and value.get("prediction_available") is True)
        if eligible:
            for key in ("selection_log_rmse", "calibration_log_rmse"):
                value[key] = _number(value.get(key), key, nonnegative=True)
        if "effective_free_parameter_ids" in value:
            value["effective_free_parameter_ids"] = _ids(value["effective_free_parameter_ids"], "effective_free_parameter_ids")
            if not set(value["effective_free_parameter_ids"]) <= set(candidate["free_parameter_ids"]):
                raise ValueError("An evaluator cannot release unrequested parameters")
            if config["fixed_parameterization"] and value["effective_free_parameter_ids"] != candidate["free_parameter_ids"]:
                raise ValueError("The registered parameterization must remain fixed for this method/phase")
        diagnostic = value.get("identifiability") or {}
        if not isinstance(diagnostic, dict):
            raise ValueError("identifiability must be a dictionary")
        suggested = value.get("selected_parameter_ids", diagnostic.get("selected_parameter_ids"))
        if suggested is not None:
            suggested = _ids(suggested, "selected_parameter_ids")
            if not set(suggested) <= set(candidate["parameter_ids"]):
                raise ValueError("Diagnostic feedback references unknown parameters")
        row = {"candidate_id": candidate["candidate_id"], "structure_id": candidate["structure_id"],
               "phase": phase, "result": value, "selection_eligible": eligible,
               "suggested_parameter_ids": suggested, "method_config": config}
        records.append(row)
        history.append({"candidate_id": candidate["candidate_id"], "phase": phase, "reason": reason,
                        "solver_calls": increment, "cumulative_solver_calls": calls})
        if increment > allowance:
            stop = "evaluator_budget_contract_violation"
        elif value.get("status") == "budget_exhausted":
            stop = "external_budget_exhausted"

    def remaining() -> list[dict]:
        return [candidate for candidate in pool if candidate["candidate_id"] not in visited]

    def joint_next(options: list[dict]) -> tuple[dict, str]:
        valid = sorted((row for row in records if row["selection_eligible"]),
                       key=lambda row: (row["result"]["selection_log_rmse"], row["candidate_id"]))[:beam_width]
        if method != "A3":
            for parent in valid:
                matches = [candidate for candidate in options
                           if candidate["structure_id"] == parent["structure_id"]
                           and parent["suggested_parameter_ids"] is not None
                           and candidate["free_parameter_ids"] == parent["suggested_parameter_ids"]]
                if matches:
                    return min(matches, key=guidance), "identifiability_subset_feedback"
        untried = [candidate for candidate in options
                   if not any(row["structure_id"] == candidate["structure_id"] for row in records)]
        if untried:
            return min(untried, key=guidance), "unvisited_structure"
        if valid:
            def priority(candidate: dict) -> tuple:
                local = [row for row in valid if row["structure_id"] == candidate["structure_id"]]
                if not local:
                    return (float("inf"), float("inf"), *guidance(candidate))
                parent = min(local, key=lambda row: row["result"]["selection_log_rmse"])
                distance = len(set(candidate["free_parameter_ids"]) ^ set(free_ids(parent)))
                return (parent["result"]["selection_log_rmse"], distance, *guidance(candidate))
            return min(options, key=priority), "selection_loss_joint_neighborhood"
        return min(options, key=guidance), "guidance_without_valid_feedback"

    structures = sorted({candidate["structure_id"] for candidate in pool})
    parameters = sorted({p for candidate in pool for p in candidate["parameter_ids"]})

    def features(candidate: dict) -> list[float]:
        return (candidate["features"] + [float(candidate["structure_id"] == s) for s in structures]
                + [float(p in candidate["free_parameter_ids"]) for p in parameters]
                + [candidate["requirement_score"], candidate["evidence_score"]])

    def rf_next(options: list[dict]) -> tuple[dict, str]:
        valid = [row for row in records if row["selection_eligible"]]
        if len(valid) < 2:
            return min(options, key=lambda c: random_order[c["candidate_id"]]), "surrogate_initial_exploration"
        forest = forest_type(n_estimators=64, min_samples_leaf=1, random_state=seed, n_jobs=1)
        forest.fit([features(by_id[row["candidate_id"]]) for row in valid],
                   [row["result"]["selection_log_rmse"] for row in valid])
        predictions = [tree.predict([features(candidate) for candidate in options]) for tree in forest.estimators_]
        acquisition = []
        for index, candidate in enumerate(options):
            values = [float(prediction[index]) for prediction in predictions]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((value - mean)**2 for value in values) / len(values))
            acquisition.append((mean - std, random_order[candidate["candidate_id"]], candidate))
        return min(acquisition, key=lambda item: item[:2])[2], "random_forest_lower_confidence_bound"

    if not pool and not blocked:
        blocked = "EMPTY_EXACT_FREE_PARAMETER_SPACE" if exact_free_parameter_count is not None else "EMPTY_LEGAL_CANDIDATE_SPACE"
    if method in {"M0", "M1"} and not blocked:
        anchor = base_candidate_id if method == "M0" else rule_candidate_id
        if anchor is None:
            blocked = "BASE_CANDIDATE_REQUIRED" if method == "M0" else "RULE_CANDIDATE_REQUIRED"
        else:
            attempt(by_id[anchor], "fixed", "registered_fixed_candidate")
    elif method == "M2" and not blocked:
        if qwen_proposer is None:
            blocked = "QWEN_PROPOSER_UNAVAILABLE"
        else:
            try:
                proposal = qwen_proposer(deepcopy(pool), seed=seed)
            except Exception as exc:
                blocked = "QWEN_PROPOSER_FAILED"
                history.append({"event": blocked, "exception_type": type(exc).__name__})
            else:
                proposal_id = proposal.get("candidate_id") if isinstance(proposal, dict) else proposal
                if not isinstance(proposal_id, str) or proposal_id not in by_id:
                    blocked = "QWEN_PROPOSAL_NOT_REGISTERED"
                else:
                    attempt(by_id[proposal_id], "one_shot", "qwen_single_proposal")
    elif method == "E0" and not blocked:
        if len(pool) > max_candidates:
            blocked = "ENUMERATION_EXCEEDS_CANDIDATE_BUDGET"
        else:
            for candidate in pool:
                attempt(candidate, "enumeration", "complete_legal_space")
                if stop:
                    break
    elif method == "M5" and not blocked:
        representatives = []
        if exact_free_parameter_count is not None:
            subsets = [{tuple(c["free_parameter_ids"]) for c in pool if c["structure_id"] == structure}
                       for structure in structures]
            common = set.intersection(*subsets)
            anchor = base_candidate_id if base_candidate_id is not None else rule_candidate_id
            proposed = tuple(by_id[anchor]["free_parameter_ids"]) if anchor is not None else (
                min(common) if common else None)
            if proposed is None or proposed not in common:
                blocked = "M5_COMMON_MATCHED_PARAMETERIZATION_MISSING"
            else:
                common_structure_parameter_ids = list(proposed)
        for structure in structures:
            if blocked:
                break
            full = [candidate for candidate in pool if candidate["structure_id"] == structure
                    and candidate["free_parameter_ids"] == (candidate["parameter_ids"]
                        if exact_free_parameter_count is None else common_structure_parameter_ids)]
            if not full:
                blocked = "M5_COMMON_PARAMETERIZATION_MISSING"
                break
            representatives.append(min(full, key=guidance))
        if not blocked:
            phase_candidates = max(1, math.floor(max_candidates * fraction))
            phase_calls = max(1, math.floor(max_solver_calls * fraction))
            anchors = [base_candidate_id, rule_candidate_id]
            representatives.sort(key=lambda candidate: (
                anchors.index(candidate["candidate_id"]) if candidate["candidate_id"] in anchors else 2,
                *guidance(candidate)))
            for candidate in representatives[:phase_candidates]:
                attempt(candidate, "structure", "common_parameter_structure_comparison", phase_call_limit=phase_calls)
                if stop or calls >= phase_calls:
                    break
            # A batch can exceed a phase's *remaining* allowance before its
            # counter reaches the cap. Only an explicit phase stop permits
            # transition; global/unknown budget stops must remain terminal.
            if (stop == "external_budget_exhausted" and calls < max_solver_calls
                    and records and records[-1]["phase"] == "structure"
                    and records[-1]["result"].get("status") == "budget_exhausted"
                    and records[-1]["result"].get("budget_scope") == "phase"):
                stop = None
            chosen = best(records)
            if chosen is not None and not stop:
                locked_structure = chosen["structure_id"]
                while not stop:
                    options = [candidate for candidate in remaining() if candidate["structure_id"] == locked_structure]
                    if not options:
                        break
                    candidate, reason = joint_next(options)
                    attempt(candidate, "parameter_subset", reason)
    elif not blocked:
        for anchor in dict.fromkeys((base_candidate_id, rule_candidate_id)):
            if anchor is not None:
                attempt(by_id[anchor], "warm_start", "shared_registered_warm_start")
        while not stop and remaining():
            options = remaining()
            if method == "M3":
                candidate = min(options, key=lambda c: random_order[c["candidate_id"]])
                reason = "seeded_random_legal_order"
            elif method == "M4":
                candidate, reason = rf_next(options)
            else:
                candidate, reason = joint_next(options)
            attempt(candidate, "joint", reason)

    eligible_records = records
    if method == "M5" and locked_structure is not None:
        eligible_records = [row for row in records if row["structure_id"] == locked_structure]
    selected = best(eligible_records)
    if blocked:
        status = "blocked"
    elif stop in {"evaluator_exception", "evaluator_budget_contract_violation"}:
        status = "failed"
    elif stop:
        status = "budget_exhausted"
    else:
        status = "completed" if selected else "failed"
    return {
        "method": method, "seed": seed, "status": status, "blocked_reason": blocked,
        "stop_reason": stop, "selected_candidate": deepcopy(by_id[selected["candidate_id"]]) if selected else None,
        "selected_result": deepcopy(selected["result"]) if selected else None,
        "evaluations": records, "history": history, "solver_calls": calls,
        "solver_cost_complete": not any(row.get("cost_reported") is False for row in history),
        "locked_structure_id": locked_structure, "registered_candidate_count": len(pool),
        "input_candidate_count": len(original_pool), "exact_free_parameter_count": exact_free_parameter_count,
        "common_structure_parameter_ids": common_structure_parameter_ids,
        "evaluated_candidate_count": len(records),
        "evaluated_structure_count": len({row["structure_id"] for row in records}),
        "enumeration_complete": method == "E0" and len(records) == len(pool) and not blocked and not stop,
        "scientific_validation_performed": False,
    }
