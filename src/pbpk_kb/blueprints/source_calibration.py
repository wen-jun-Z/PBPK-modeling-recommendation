"""Small, bounded calibration of explicit inputs in an unchanged source PKML.

This is parameter calibration, not structural model design. Only caller-supplied
calibration observations are accepted. The source model is never overwritten.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

from .design_scoring import score_calibration_curves
from .parameter_batch import _inside, _number, _quantity_path, _sha, _text, run_parameter_batch

VERSION = "bounded-complete-source-calibration-v1"
PRIOR_WEIGHT = 0.02
_SPEC_FIELDS = {"parameter_id", "path", "unit", "source_value", "lower_bound",
                "upper_bound", "evidence_reference", "scaling"}
_CASE_FIELDS = {"case_id", "study_group_id", "role", "pkml_path", "target_output", "observations"}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical(value) + "\n")


def _coordinate(value, spec):
    low, high = spec["lower_bound"], spec["upper_bound"]
    if spec["scaling"] == "log":
        return (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    return (value - low) / (high - low)


def _physical(coordinate, spec):
    low, high = spec["lower_bound"], spec["upper_bound"]
    if coordinate <= 0:
        return low
    if coordinate >= 1:
        return high
    if spec["scaling"] == "log":
        value = math.exp(math.log(low) + coordinate * (math.log(high) - math.log(low)))
    else:
        value = low + coordinate * (high - low)
    # Guard roundoff at a declared physical bound.
    return min(high, max(low, value))


def _penalty(values, specs):
    terms = []
    for value, spec in zip(values, specs, strict=True):
        displacement = ((math.log(value) - math.log(spec["source_value"])) if spec["scaling"] == "log"
                        else (value - spec["source_value"]) / (spec["upper_bound"] - spec["lower_bound"]))
        terms.append(displacement * displacement)
    return PRIOR_WEIGHT * sum(terms)


def _validate(training_cases, parameter_specs, output_dir, project_root, max_rounds, device):
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("project_root must be a directory")
    destination = _inside(output_dir, root, must_exist=False)
    if destination.exists():
        raise FileExistsError("Calibration output directory already exists")
    if type(max_rounds) is not int or not 1 <= max_rounds <= 2:
        raise ValueError("max_rounds must be 1 or 2")
    if not isinstance(parameter_specs, list) or not 1 <= len(parameter_specs) <= 3:
        raise ValueError("Provide 1..3 explicitly bounded source parameters")
    specs = []
    for spec in parameter_specs:
        if not isinstance(spec, dict) or set(spec) != _SPEC_FIELDS:
            raise ValueError("Parameter specifications require exactly the documented fields")
        item = dict(spec)
        for key in ("parameter_id", "unit", "evidence_reference"):
            item[key] = _text(item[key], key)
        item["path"] = _quantity_path(item["path"], "parameter path")
        for key in ("source_value", "lower_bound", "upper_bound"):
            item[key] = _number(item[key])
        low, source, high = (item[k] for k in ("lower_bound", "source_value", "upper_bound"))
        if not low <= source <= high or high <= low or not math.isfinite(high - low):
            raise ValueError("Finite, ordered bounds must include the declared source value")
        if item["scaling"] not in {"log", "linear"} or (item["scaling"] == "log" and low <= 0):
            raise ValueError("Use linear scaling or log scaling with positive bounds")
        specs.append(item)
    for key in ("parameter_id", "path"):
        if len({s[key] for s in specs}) != len(specs):
            raise ValueError("Parameter identifiers and native paths must be unique")
    if not isinstance(training_cases, list) or not training_cases:
        raise ValueError("Provide nonempty calibration cases")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cases = []
    for case in training_cases:
        if not isinstance(case, dict) or set(case) != _CASE_FIELDS or case.get("role") != "calibration":
            raise ValueError("Only explicit calibration cases with documented fields are accepted; no holdout")
        item = dict(case)
        for key in ("case_id", "study_group_id"):
            item[key] = _text(item[key], key)
        path = _inside(item["pkml_path"], root, must_exist=True)
        if not path.is_file() or path.suffix.lower() != ".pkml":
            raise ValueError("Every calibration case needs a complete project PKML")
        item["pkml_path"], item["pkml_sha256"] = str(path), _sha(path)
        item["target_output"] = _quantity_path(item["target_output"], "target output")
        # Validate observations and explicit/automatic scoring device before output.
        score_calibration_curves(runs=[], observations=item["observations"], device=device)
        item["observations"] = json.loads(_canonical(item["observations"]))
        cases.append(item)
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Case identifiers must be unique")
    return root, destination, cases, specs, device


def _aggregate(case_scores, cases, values, specs):
    if any(not case_scores[c["case_id"]]["valid"] for c in cases):
        return {"valid": False, "log_mse": None, "prior_penalty": None, "objective": None,
                "per_study": {}, "cases": case_scores}
    grouped = {}
    for case in cases:
        grouped.setdefault(case["study_group_id"], []).append(case_scores[case["case_id"]]["log_mse"])
    per_study = {group: sum(scores) / len(scores) for group, scores in grouped.items()}
    mse = sum(per_study.values()) / len(per_study)
    prior = _penalty(values, specs)
    return {"valid": True, "log_mse": mse, "prior_penalty": prior, "objective": mse + prior,
            "per_study": per_study, "cases": case_scores}


def _grid(center, specs, round_index):
    points = []
    for index, spec in enumerate(specs):
        coordinate = _coordinate(center[index], spec)
        low, high = ((0., 1.) if round_index == 0 else
                     (max(0., coordinate - 1 / 6), min(1., coordinate + 1 / 6)))
        for step in range(7):
            point = list(center)
            point[index] = _physical(low + (high - low) * step / 6, spec)
            points.append(tuple(point))
    return points


def _sensitivity(cache, selected, specs, cases):
    """Observed grid secants, without extra ODE calls or identifiability claims."""
    result = []
    for index, spec in enumerate(specs):
        matches = [(point, score) for point, score in cache.items() if score["valid"]
                   and all(point[j] == selected[j] for j in range(len(specs)) if j != index)]
        nearest = sorted(matches, key=lambda pair: (abs(_coordinate(pair[0][index], spec)
                                                     - _coordinate(selected[index], spec)), pair[0]))[:2]
        row = {"parameter_id": spec["parameter_id"], "valid_aligned_points": len(matches),
               "status": "insufficient_aligned_points", "rms_log_prediction_per_normalized_coordinate": None,
               "interpretation": "Sampled secant diagnostic; not an identifiability or causal conclusion"}
        if len(nearest) == 2:
            (p1, s1), (p2, s2) = nearest
            delta = _coordinate(p2[index], spec) - _coordinate(p1[index], spec)
            if delta:
                groups = {}
                for case in cases:
                    a = s1["cases"][case["case_id"]]["predictions_umol_per_l"]
                    b = s2["cases"][case["case_id"]]["predictions_umol_per_l"]
                    mse = sum(((math.log(y) - math.log(x)) / delta) ** 2
                              for x, y in zip(a, b, strict=True)) / len(a)
                    groups.setdefault(case["study_group_id"], []).append(mse)
                squared = sum(sum(v) / len(v) for v in groups.values()) / len(groups)
                row.update(status="computed", parameter_values=[p1[index], p2[index]],
                           rms_log_prediction_per_normalized_coordinate=math.sqrt(squared))
        result.append(row)
    return result


def fit_bounded_source_parameters(training_cases, parameter_specs, output_dir, project_root,
                                  max_rounds=2, device="auto"):
    """Fit 1..3 declared inputs using at most 1 + 7*p*rounds vectors per case.

    Each round scores coordinate candidates together, selects one real scored
    vector, and never combines independent coordinate optima without evaluation.
    Studies have equal weight; cases within a study have equal weight. Bounds
    and source values are caller-reviewed scientific inputs, not inferred here.
    """
    root, destination, cases, specs, device = _validate(
        training_cases, parameter_specs, output_dir, project_root, max_rounds, device)
    request = {"version": VERSION, "training_cases": cases, "parameter_specs": specs,
               "max_rounds": max_rounds, "device": device, "prior_weight": PRIOR_WEIGHT}
    destination.mkdir(parents=True)
    _write(destination / "calibration_request.json", request)
    source = tuple(s["source_value"] for s in specs)
    cache, trace, batches, rounds = {}, [], [], []
    native_specs = [{"path": s["path"], "unit": s["unit"]} for s in specs]
    counts = {"unique_vectors": 0, "native_vector_attempts": 0, "native_batches": 0,
              "valid_vectors": 0, "failed_vectors": 0, "cache_hits": 0,
              "vector_limit_per_case": 1 + 7 * len(specs) * max_rounds,
              "max_parallel_cases": min(4, len(cases)), "cpu_threads_per_batch": 2}

    def evaluate(points, round_index):
        pending = []
        for point in points:
            if point in cache or point in pending:
                counts["cache_hits"] += 1
            else:
                pending.append(point)
        if not pending:
            return
        if len(pending) > 64 or counts["unique_vectors"] + len(pending) > counts["vector_limit_per_case"]:
            raise RuntimeError("Internal calibration budget exceeded")

        def run_case(item):
            index, case = item
            dispatched = False
            try:
                if _sha(Path(case["pkml_path"])) != case["pkml_sha256"]:
                    raise ValueError("Source PKML changed after request lock")
                dispatched = True
                result = run_parameter_batch(pkml_path=Path(case["pkml_path"]), parameter_specs=native_specs,
                    values=[list(v) for v in pending], target_output=case["target_output"],
                    output_dir=destination / f"round_{round_index + 1:02d}" / f"case_{index:03d}",
                    project_root=root, cpu_threads=2)
                if _sha(Path(case["pkml_path"])) != case["pkml_sha256"]:
                    raise ValueError("Source PKML changed during parameter batch")
                return result, dispatched
            except (OSError, ValueError, RuntimeError) as exc:
                return {"status": "failed", "runs": [], "error": type(exc).__name__ + ": " + str(exc)}, dispatched

        with ThreadPoolExecutor(max_workers=counts["max_parallel_cases"]) as executor:
            outcomes = list(executor.map(run_case, enumerate(cases)))
        scores = []
        for case, (outcome, dispatched) in zip(cases, outcomes, strict=True):
            batches.append({"round": round_index + 1, "case_id": case["case_id"],
                            "vector_count": len(pending), "status": outcome.get("status"),
                            "native_batch_dispatched": dispatched,
                            "manifest_path": outcome.get("manifest_path"), "error": outcome.get("error")})
            counts["native_batches"] += int(dispatched)
            counts["native_vector_attempts"] += len(pending) * int(dispatched)
            if outcome.get("status") == "completed":
                scored = score_calibration_curves(runs=outcome["runs"], observations=case["observations"], device=device)
                by_index = {row["vector_index"]: row for row in scored["rows"]}
                if set(by_index) != set(range(len(pending))):
                    raise RuntimeError("Native batch violated complete vector mapping")
                scores.append(by_index)
            else:
                scores.append({i: {"valid": False, "log_mse": None, "predictions_umol_per_l": None,
                                   "error": outcome.get("error", "native_batch_failed")} for i in range(len(pending))})
        for index, point in enumerate(pending):
            score = _aggregate({case["case_id"]: rows[index] for case, rows in zip(cases, scores, strict=True)},
                               cases, point, specs)
            cache[point] = score
            trace.append({"vector_index": len(trace), "round": round_index + 1,
                          "values": list(point), **score})
        counts["unique_vectors"] += len(pending)

    selected = source
    for round_index in range(max_rounds):
        center = selected
        evaluate(([source] if round_index == 0 else []) + _grid(center, specs, round_index), round_index)
        if not cache[source]["valid"]:
            break
        for point, score in cache.items():
            if score["valid"] and score["objective"] < cache[selected]["objective"]:
                selected = point
        rounds.append({"round": round_index + 1, "center": list(center), "selected_values": list(selected),
                       "selected_objective": cache[selected]["objective"]})

    baseline = cache[source]
    success = baseline["valid"]
    chosen = cache[selected] if success else None
    counts["valid_vectors"] = sum(row["valid"] for row in cache.values())
    counts["failed_vectors"] = len(cache) - counts["valid_vectors"]
    result = {"version": VERSION, "status": "completed" if success else "failed",
              "error": None if success else "SOURCE_BASELINE_INVALID",
              "scientific_status": "calibration_only_not_independent_validation",
              "source_values": dict(zip([s["parameter_id"] for s in specs], source, strict=True)),
              "selected_values": (dict(zip([s["parameter_id"] for s in specs], selected, strict=True)) if success else None),
              "selected_vector": list(selected) if success else None,
              "training_score": chosen, "baseline_score": baseline,
              "improved_training_objective": bool(success and chosen["objective"] < baseline["objective"]),
              "study_diagnostics": [{"study_group_id": group,
                  "case_ids": [c["case_id"] for c in cases if c["study_group_id"] == group],
                  "baseline_log_mse": baseline["per_study"].get(group),
                  "selected_log_mse": chosen["per_study"].get(group) if chosen else None}
                  for group in sorted({c["study_group_id"] for c in cases})],
              "sensitivity_diagnostics": _sensitivity(cache, selected, specs, cases) if success else [],
              "counts": counts, "rounds": rounds, "batches": batches, "trace": trace,
              "provenance": {"request_sha256": hashlib.sha256(_canonical(request).encode()).hexdigest(),
                  "implementation_sha256": _sha(Path(__file__)), "parameter_specs": specs,
                  "cases": [{k: c[k] for k in ("case_id", "study_group_id", "role", "pkml_path", "pkml_sha256", "target_output")}
                            for c in cases],
                  "native_backend": "OSPSuite CPU; one PKML load per case and round",
                  "scoring_device": device, "prior_weight": PRIOR_WEIGHT,
                  "baseline_semantics": "declared_source_parameter_vector; caller must confirm native source values",
                  "bounds_semantics": "caller-reviewed scientific limits and evidence, not automatically established",
                  "selection_data": "calibration_only", "structure_unchanged": True,
                  "source_files_overwritten": False, "fitted_pkml_exported": False}}
    _write(destination / "calibration_result.json", result)
    return result
