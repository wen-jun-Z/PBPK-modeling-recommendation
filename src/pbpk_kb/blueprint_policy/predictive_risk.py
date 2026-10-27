"""Exploratory endpoint-risk meta-model with leave-study-cluster-out scoring.

This predicts an association with a later study's protocol-matched fold error.
It is neither a clinical eligibility rule nor independent clinical validation.
All fitted preprocessing and coefficients exclude the held-out study cluster.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


VERSION = "cluster-crossfit-predictive-risk-2"
FEATURES = ("development_ln_rmse", "development_support_gap", "clearance_boundary_fraction")
FAMILIES = ("exposure", "shape", "moment")
ENDPOINT_FAMILY = {"auc_inf": "exposure", "clearance": "exposure",
    "terminal_half_life": "shape", "vz": "shape", "mrt": "moment", "vss": "moment"}
CONFIG = {"version": VERSION, "continuous_features": list(FEATURES),
    "endpoint_families": list(FAMILIES), "main_threshold": .35,
    "diagnostic_thresholds": [.35, .5, .8], "ece_bin_edges": [0., .2, .4, .6, .8, 1.],
    "label": "matched clinical-protocol scalar fold error strictly greater than 2",
    "learner": {"C": 1., "penalty": "l2", "solver": "lbfgs", "max_iter": 1000, "tol": 1e-9},
    "split": "leave entire study cluster out, including all its drugs, sampling regimes and endpoints",
    "weighting": "one total weight per study cluster; equal drugs, then sampling regimes, then endpoints",
    "preprocessing": "training-fold weighted median imputation and weighted standardization only",
    "fallback": "fewer than two training clusters or one label class: Jeffreys-smoothed training weighted prevalence",
    "support_gap": {"exposure": "fraction of locked finite-horizon development-model AUC after last development observation",
        "moment": "fraction of locked finite-horizon development-model AUMC after last development observation",
        "shape": "log1p(dense development-model half-life / span of last max(3,ceil(n_postinfusion/3)) postinfusion observations)",
        "missing": "missing half-life or fewer than three postinfusion points stays null; training-fold imputation only"},
    "boundary": "minimum log-distance of CL multiplier to its predeclared calibration bounds divided by full log-range; original baseline bounds [.25,4]",
    "interpretation": "Development cross-validation on previously inspected studies; scores and .35 warning threshold are exploratory, not clinical confidence or dose guidance."}
NESTED_CONFIG = {"learner": "fixed development-RMSE-only L2 logistic C=1, no endpoint categories",
    "selection_history": "The RMSE-only learner was selected after inspecting the original baseline results; this is one additional exploratory attempt, not a blind confirmatory analysis.",
    "inner_recall_target": .8,
    "threshold_rule": "Highest distinct inner leave-study-cluster-out probability with cluster-weighted FE>2 recall at least 0.8; warn at probability >= threshold.",
    "split": "Each outer held-out study cluster is excluded from every inner fit and threshold choice; full/five and all its drugs/endpoints stay together.",
    "fallback": "Too few clusters, any single-class fit, constant fitted predictor, constant pooled inner probabilities, or no eligible threshold: warn on every outer held-out record.",
    "interpretation": "The inner recall target is a tuning constraint, not a guarantee about held-out or clinical recall; no further threshold or recall-target tuning is allowed for this attempt."}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value):
    return _finite(value) and value > 0


def cluster_weights(records):
    """Avoid turning correlated endpoints and full/five aliases into sample size."""
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, row in enumerate(records):
        groups[row["cluster_id"]][row["case_id"]][row["sampling"]].append(i)
    weights = np.zeros(len(records))
    for drugs in groups.values():
        for samplings in drugs.values():
            for indices in samplings.values():
                weights[indices] = 1. / (len(drugs) * len(samplings) * len(indices))
    return weights


def _weighted_median(values, weights):
    good = np.isfinite(values)
    if not np.any(good):
        return 0.  # Explicitly uninformative all-missing training feature.
    v, w = values[good], weights[good]
    order = np.argsort(v, kind="stable"); v, w = v[order], w[order]
    return float(v[np.searchsorted(np.cumsum(w), w.sum() / 2, side="left")])


def _matrix(records, features):
    return np.asarray([[float(r["features"][k]) if _finite(r["features"].get(k)) else np.nan
                        for k in features] for r in records], dtype=float)


def _preprocess(train, heldout, weights, features, *, families):
    a, b = _matrix(train, features), _matrix(heldout, features)
    medians = np.array([_weighted_median(a[:, j], weights) for j in range(len(features))])
    a = np.where(np.isfinite(a), a, medians); b = np.where(np.isfinite(b), b, medians)
    mean = np.average(a, axis=0, weights=weights)
    scale = np.sqrt(np.average((a - mean)**2, axis=0, weights=weights))
    scale[scale < 1e-12] = 1.
    a, b = (a - mean) / scale, (b - mean) / scale
    if families:
        def categories(rows):
            return np.asarray([[ENDPOINT_FAMILY[r["endpoint"]] == f for f in FAMILIES] for r in rows], float)
        a = np.column_stack((a, categories(train))); b = np.column_stack((b, categories(heldout)))
    return a, b, {"features": list(features), "training_weighted_medians": medians.tolist(),
                  "training_weighted_means": mean.tolist(), "training_weighted_scales": scale.tolist(),
                  "endpoint_categories": list(FAMILIES) if families else []}


def _validate(records):
    if not records:
        raise ValueError("At least one labeled risk record is required")
    identities, case_clusters = [], {}
    for r in records:
        for name in ("case_id", "cluster_id", "sampling"):
            if not isinstance(r.get(name), str) or not r[name]:
                raise ValueError("Explicit case, cluster and sampling identities are required")
        if r.get("endpoint") not in ENDPOINT_FAMILY:
            raise ValueError("Unknown predeclared endpoint family")
        if not _finite(r.get("matched_fold_error")) or r["matched_fold_error"] < 1:
            raise ValueError("Missing/nonfinite/invalid fold error cannot become a negative label")
        if type(r.get("dense_warning")) is not bool:
            raise ValueError("The old numerical gate must be explicitly recorded")
        if not isinstance(r.get("features"), dict) or set(r["features"]) != set(FEATURES):
            raise ValueError("Exactly the three predeclared features are required")
        for value in r["features"].values():
            if value is not None and not _finite(value):
                raise ValueError("Represent a missing feature as null, not NaN or infinity")
        if r["case_id"] in case_clusters and case_clusters[r["case_id"]] != r["cluster_id"]:
            raise ValueError("All versions of one drug case must remain in one study cluster")
        case_clusters[r["case_id"]] = r["cluster_id"]
        identities.append((r["case_id"], r["sampling"], r["endpoint"]))
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate endpoint aliases cannot count as extra training records")
    if len(set(case_clusters.values())) < 2:
        raise ValueError("Leave-study-cluster-out evaluation needs at least two clusters")


def _metrics(records, probabilities):
    y = np.asarray([r["matched_fold_error"] > 2 for r in records], bool)
    q = np.asarray(probabilities, float); w = cluster_weights(records)
    if q.shape != y.shape or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("Every held-out endpoint needs a finite probability in [0,1]")
    edges = np.asarray(CONFIG["ece_bin_edges"]); bins = np.minimum(np.searchsorted(edges, q, side="right") - 1, len(edges) - 2)
    reliability, ece = [], 0.
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = bins == i; weight = w[mask].sum()
        if weight:
            probability = float(np.average(q[mask], weights=w[mask])); observed = float(np.average(y[mask], weights=w[mask]))
            ece += weight / w.sum() * abs(probability - observed)
        else:
            probability = observed = None
        reliability.append({"lower": float(lo), "upper": float(hi), "upper_inclusive": i == len(edges) - 2,
            "records": int(mask.sum()), "cluster_weight": float(weight),
            "mean_probability": probability, "observed_large_error_rate": observed})
    thresholds = []
    for threshold in CONFIG["diagnostic_thresholds"]:
        warn = q >= threshold
        tp, fp, fn, tn = (int(mask.sum()) for mask in (warn & y, warn & ~y, ~warn & y, ~warn & ~y))
        wt, wf, wn, wr = (float(w[mask].sum()) for mask in (warn & y, warn & ~y, ~warn & y, ~warn & ~y))
        div = lambda a, b: a / b if b else None
        thresholds.append({"threshold": threshold, "primary": threshold == CONFIG["main_threshold"],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn), "false_warning_rate": div(fp, fp + tn),
            "coverage": (fn + tn) / len(y), "accepted_large_error_rate": div(fn, fn + tn),
            "cluster_weighted": {"precision": div(wt, wt + wf), "recall": div(wt, wt + wn),
                "false_warning_rate": div(wf, wf + wr), "coverage": (wn + wr) / w.sum(),
                "accepted_large_error_rate": div(wn, wn + wr)}})
    return {"records": len(records), "study_clusters": len({r["cluster_id"] for r in records}),
        "large_errors": int(y.sum()), "cluster_weighted_prevalence": float(np.average(y, weights=w)),
        "brier": float(np.average((q - y)**2, weights=w)), "unweighted_brier": float(np.mean((q - y)**2)),
        "ece": float(ece), "auroc": float(roc_auc_score(y, q, sample_weight=w)) if len(set(y)) == 2 else None,
        "reliability_bins": reliability, "thresholds": thresholds}


def _record_id(record):
    return [record[k] for k in ("case_id", "sampling", "endpoint")]


def _warning_metrics(records, warnings):
    """Score a decision policy without misrepresenting warnings as probabilities."""
    y = np.asarray([r["matched_fold_error"] > 2 for r in records], bool)
    warn = np.asarray(warnings)
    if warn.dtype != bool or warn.shape != y.shape:
        raise ValueError("Every policy record needs an explicit Boolean warning")
    weights = cluster_weights(records)
    masks = (warn & y, warn & ~y, ~warn & y, ~warn & ~y)
    tp, fp, fn, tn = [int(mask.sum()) for mask in masks]
    wt, wf, wn, wr = [float(weights[mask].sum()) for mask in masks]
    div = lambda a, b: float(a / b) if b else None
    return {"records": len(records), "study_clusters": len({r["cluster_id"] for r in records}),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
        "false_warning_rate": div(fp, fp + tn), "coverage": div(fn + tn, len(records)),
        "accepted_large_error_rate": div(fn, fn + tn),
        "cluster_weighted": {"precision": div(wt, wt + wf), "recall": div(wt, wt + wn),
            "false_warning_rate": div(wf, wf + wr), "coverage": div(wn + wr, weights.sum()),
            "accepted_large_error_rate": div(wn, wn + wr)}}


def _nested_fit_rmse(train, valid):
    """Fit one nested fold, logging every identity and train-only transform."""
    weights = cluster_weights(train)
    labels = np.asarray([r["matched_fold_error"] > 2 for r in train], int)
    clusters = sorted({r["cluster_id"] for r in train})
    prevalence = float((np.dot(weights, labels) + .5) / (weights.sum() + 1.))
    a, b, preprocessing = _preprocess(train, valid, weights, FEATURES[:1], families=False)
    audit = {"training_clusters": clusters, "held_out_clusters": sorted({r["cluster_id"] for r in valid}),
        "training_records": len(train), "held_out_records": len(valid),
        "training_ids": [_record_id(r) for r in train], "held_out_ids": [_record_id(r) for r in valid],
        "training_labels": dict(Counter(map(int, labels))), "training_cluster_weight": float(weights.sum()),
        "preprocessing": preprocessing, "fallback": None}
    if len(clusters) < 2 or len(set(labels)) < 2:
        audit.update(fallback="too_few_clusters_or_single_class", coefficients=None, intercept=None, iterations=0)
        return np.full(len(valid), prevalence), audit
    model = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, tol=1e-9)
    model.fit(a, labels, sample_weight=weights)
    if int(model.n_iter_[0]) >= 1000:
        raise ValueError("Nested risk classifier exhausted its fixed optimizer budget")
    probabilities = model.predict_proba(b)[:, list(model.classes_).index(1)]
    fitted = model.predict_proba(a)[:, list(model.classes_).index(1)]
    audit.update(coefficients=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
                 iterations=int(model.n_iter_[0]))
    if float(np.ptp(fitted)) <= 1e-12:
        audit["fallback"] = "constant_fitted_predictor"
    return probabilities, audit


def _select_nested_threshold(records, probabilities):
    """Use only outer-training labels and inner held-out probabilities."""
    y = np.asarray([r["matched_fold_error"] > 2 for r in records], bool)
    q = np.asarray(probabilities, float)
    if q.shape != y.shape or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("Invalid inner held-out probabilities")
    if len(set(y)) < 2:
        return None, "single_class_threshold_data"
    if float(np.ptp(q)) <= 1e-12:
        return None, "constant_inner_probabilities"
    w = cluster_weights(records)
    for threshold in np.unique(q)[::-1]:
        recall = float(w[y & (q >= threshold)].sum() / w[y].sum())
        if recall >= NESTED_CONFIG["inner_recall_target"] - 1e-12:
            return float(threshold), None
    return None, "no_eligible_threshold"


def evaluate_nested_rmse_policy(records):
    """Single prespecified nested threshold attempt; outer labels are score-only."""
    records = deepcopy(records); _validate(records)
    warnings = np.ones(len(records), dtype=bool)
    folds, decisions = [], [None] * len(records)
    for held_cluster in sorted({r["cluster_id"] for r in records}):
        train = [r for r in records if r["cluster_id"] != held_cluster]
        vi = [i for i, r in enumerate(records) if r["cluster_id"] == held_cluster]
        valid = [records[i] for i in vi]
        probabilities, outer_fit = _nested_fit_rmse(train, valid)
        inner_probabilities = np.zeros(len(train))
        inner_folds = []
        for inner_cluster in sorted({r["cluster_id"] for r in train}):
            inner_train = [r for r in train if r["cluster_id"] != inner_cluster]
            ii = [i for i, r in enumerate(train) if r["cluster_id"] == inner_cluster]
            inner_valid = [train[i] for i in ii]
            if not inner_train:
                inner_probabilities[ii] = .5
                inner_folds.append({"training_clusters": [], "held_out_clusters": [inner_cluster],
                    "training_records": 0, "held_out_records": len(inner_valid), "training_ids": [],
                    "held_out_ids": [_record_id(r) for r in inner_valid], "fallback": "empty_inner_training_set"})
            else:
                q, audit = _nested_fit_rmse(inner_train, inner_valid)
                inner_probabilities[ii] = q
                inner_folds.append(audit)
        threshold, fallback = _select_nested_threshold(train, inner_probabilities)
        if outer_fit["fallback"]:
            fallback = "outer_" + outer_fit["fallback"]
        elif any(f["fallback"] for f in inner_folds):
            fallback = "degenerate_inner_fit"
        if fallback:
            threshold = None
        inner_warnings = np.ones(len(train), bool) if fallback else inner_probabilities >= threshold
        warnings[vi] = True if fallback else probabilities >= threshold
        folds.append({"held_out_cluster": held_cluster, "threshold": threshold, "fallback": fallback,
            "training_clusters": sorted({r["cluster_id"] for r in train}),
            "inner_oof_records": len(train), "inner_oof_clusters": len(inner_folds),
            "training_ids": [_record_id(r) for r in train], "held_out_ids": [_record_id(r) for r in valid],
            "outer_fit": outer_fit, "inner_folds": inner_folds,
            "inner_oof": [{"id": _record_id(r), "cluster_id": r["cluster_id"],
                "probability": float(q), "large_error": r["matched_fold_error"] > 2}
                for r, q in zip(train, inner_probabilities)],
            "inner_selection_metrics": _warning_metrics(train, inner_warnings),
            "held_out_metrics": _warning_metrics(valid, warnings[vi])})
        for i, q in zip(vi, probabilities):
            decisions[i] = {"id": _record_id(records[i]), "cluster_id": held_cluster,
                "probability": float(q), "threshold": threshold, "warning": bool(warnings[i]), "fallback": fallback}
    return {"config": deepcopy(NESTED_CONFIG), "folds": folds, "records": decisions,
        "metrics": _warning_metrics(records, warnings),
        "by_sampling": {sampling: _warning_metrics([r for r in records if r["sampling"] == sampling],
            warnings[[i for i, r in enumerate(records) if r["sampling"] == sampling]])
            for sampling in sorted({r["sampling"] for r in records})},
        "fallback_folds": sum(f["fallback"] is not None for f in folds),
        "no_same_cluster_labels_used_for_threshold_or_model": True}


def evaluate_records(records):
    """Pure structured-record interface; never fit and score the same cluster."""
    records = deepcopy(records); _validate(records)
    predictions = {name: np.zeros(len(records)) for name in ("risk_model", "development_rmse_only", "training_prevalence")}
    folds = []
    for held_cluster in sorted({r["cluster_id"] for r in records}):
        ti = [i for i, r in enumerate(records) if r["cluster_id"] != held_cluster]
        vi = [i for i, r in enumerate(records) if r["cluster_id"] == held_cluster]
        train, valid = [records[i] for i in ti], [records[i] for i in vi]
        y = np.asarray([r["matched_fold_error"] > 2 for r in train], int); weights = cluster_weights(train)
        prevalence = float((np.dot(weights, y) + .5) / (weights.sum() + 1.))
        train_clusters = sorted({r["cluster_id"] for r in train})
        fallback = len(train_clusters) < 2 or len(set(y)) < 2
        fold = {"held_out_cluster": held_cluster, "training_clusters": train_clusters,
            "training_records": len(train), "held_out_records": len(valid), "training_cluster_weight": float(weights.sum()),
            "training_labels": dict(Counter(map(int, y))), "smoothed_training_prevalence": prevalence,
            "fallback": "too_few_clusters_or_single_class" if fallback else None, "models": {}}
        predictions["training_prevalence"][vi] = prevalence
        for name, features, families in (("risk_model", FEATURES, True),
                                         ("development_rmse_only", FEATURES[:1], False)):
            a, b, preprocessing = _preprocess(train, valid, weights, features, families=families)
            if fallback:
                probabilities = np.full(len(valid), prevalence)
                model_audit = {"coefficients": None, "intercept": None, "iterations": 0}
            else:
                model = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, tol=1e-9)
                model.fit(a, y, sample_weight=weights)
                if int(model.n_iter_[0]) >= 1000:
                    raise ValueError("Risk classifier exhausted its fixed optimizer budget")
                probabilities = model.predict_proba(b)[:, list(model.classes_).index(1)]
                model_audit = {"coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
                               "iterations": int(model.n_iter_[0])}
            predictions[name][vi] = probabilities
            fold["models"][name] = {**preprocessing, **model_audit}
        folds.append(fold)
    predictions["dense_gate"] = np.asarray([float(r["dense_warning"]) for r in records])
    predictions["always_warn"] = np.ones(len(records))
    output = [{**r, "held_out_predictions": {name: float(q[i]) for name, q in predictions.items()}}
              for i, r in enumerate(records)]
    metrics = {name: _metrics(records, q) for name, q in predictions.items()}
    by_sampling = {}
    for sampling in sorted({r["sampling"] for r in records}):
        indices = [i for i, r in enumerate(records) if r["sampling"] == sampling]
        by_sampling[sampling] = {name: _metrics([records[i] for i in indices], q[indices]) for name, q in predictions.items()}
    return {"version": VERSION, "config": deepcopy(CONFIG), "records": output, "folds": folds,
            "metrics": metrics, "by_sampling": by_sampling,
            "nested_rmse_policy": evaluate_nested_rmse_policy(records),
            "no_same_cluster_labels_used_for_its_predictions": True,
            "limitations": [CONFIG["interpretation"], "Only nine original study clusters; uncertainty is not established by 106 correlated endpoint rows.",
                "The raw dense gate and always-warn scores are binary comparators, not estimated probabilities.",
                "Missing clinical references are excluded and counted separately; null never means low risk."]}


def development_features(*, training, prediction, endpoints, mapping, request, clearance_bounds=None):
    """Features use only a development curve/model and its frozen fit metadata."""
    observations = training["observations"]
    times = np.asarray([r["time_min"] / 60 for r in observations], float)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("At least two distinct ordered development times are required")
    t = np.asarray(prediction["time_min"], float) / 60; c = np.asarray(prediction["concentration"], float)
    if t.shape != c.shape or len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(c).all() or np.any(c < 0) or np.any(np.diff(t) < 0):
        raise ValueError("Invalid locked development-model trajectory")
    # Add the last observed time exactly to the piecewise-linear integral grid.
    end = float(times[-1])
    if end < t[0] or end > t[-1]:
        raise ValueError("Development prediction does not cover its observations")
    t, unique = np.unique(t, return_index=True); c = c[unique]
    if end not in t:
        pos = np.searchsorted(t, end); ce = np.interp(end, t, c)
        t, c = np.insert(t, pos, end), np.insert(c, pos, ce)
    late = t >= end
    fractions = {}
    for family, values in (("exposure", c), ("moment", t * c)):
        total = float(np.trapezoid(values, t)); tail = float(np.trapezoid(values[late], t[late]))
        fractions[family] = float(np.clip(tail / total, 0., 1.)) if total > 0 else None
    stop = float(request["dosing"].get("start_h", 0)) + float(request["dosing"]["infusion_minutes"]) / 60
    post = times[times >= stop]
    span = None if len(post) < 3 else float(post[-1] - post[-max(3, math.ceil(len(post) / 3))])
    half = endpoints.get("endpoints", {}).get("terminal_half_life", {}).get("value_h")
    fractions["shape"] = math.log1p(half / span) if _positive(half) and _positive(span) else None
    scale = mapping["decision"]["parameters"]["clearance_scale"]
    bounds = clearance_bounds if clearance_bounds is not None else mapping.get("optimizer_config", {}).get("bounds", {}).get("clearance_scale", [.25, 4.])
    low, high = map(float, bounds)
    if not _positive(low) or not _positive(high) or low >= high:
        raise ValueError("Invalid predeclared clearance bounds")
    if not _positive(scale) or scale < low - 1e-10 or scale > high + 1e-10:
        raise ValueError("CL multiplier is outside the frozen calibration bounds")
    boundary = max(0., min(math.log(scale / low), math.log(high / scale)) / math.log(high / low))
    rmse = mapping["decision"]["data_only_ln_rmse"]
    if not _finite(rmse) or rmse < 0:
        raise ValueError("Invalid development fit error")
    return {endpoint: dict(zip(FEATURES, (float(rmse), fractions[family], float(boundary))))
            for endpoint, family in ENDPOINT_FAMILY.items()}, {
        "development_points": len(times), "postinfusion_points": len(post), "last_development_time_h": end,
        "postinfusion_last_third_span_h": span, "dense_development_half_life_h": half,
        "model_integration_horizon_h": float(t[-1]), "clearance_bounds": [low, high], "support_fractions": fractions}


def collect_current_records(project_root):
    """Source-locked baseline collection; no new simulation and no paper writes."""
    root = Path(project_root).resolve(strict=True); pins = {}
    def read(path, expected=None):
        path = (root / path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source escapes this project")
        payload = path.read_bytes(); digest = hashlib.sha256(payload).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("Frozen risk source changed: " + str(path))
        pins[str(path.relative_to(root))] = digest
        return json.loads(payload)
    def checked(ref):
        return read(ref["path"], ref["sha256"])
    run = "experiments/blueprint_design/blueprint_capability_20260930/"
    state = read(run + "state.json")
    if state["status"] != "complete" or not state.get("all_attempts_locked_before_scoring"):
        raise ValueError("Risk development requires a completed source experiment")
    result = checked(state["results"]); protocol = checked(state["protocol"])
    transport = checked(protocol["source_protocol"])
    original_state = checked(protocol["source_state"])
    original_results = checked(original_state["results"])
    auxiliary = checked(original_results["auxiliary_source"])
    cases = {c["case_id"]: c for c in protocol["cases"]}; records, excluded = [], []
    gate = {(r["case_id"], r["sampling"], r["endpoint"]): r for r in result["tail_gate"]["records"] if r["method"] == "always_three"}
    for row in result["units"]:
        if (row["scenario"], row["method"]) != ("calibration", "always_three"):
            continue
        cid, sampling = row["case_id"], row["sampling"]
        identity = cid + "__" + sampling + "__flow_exchange"
        source = auxiliary["development"][identity]
        training = checked(transport["computations"][identity]["training_source"])
        prediction, endpoints, mapping, request = (checked(source[k]) for k in ("prediction", "endpoints", "mapping", "request"))
        features, support = development_features(training=training, prediction=prediction, endpoints=endpoints, mapping=mapping, request=request)
        for endpoint in ENDPOINT_FAMILY:
            scored = row["endpoints"][endpoint]
            gate_record = gate[cid, sampling, endpoint]
            dense = checked(gate_record["gate_source"])
            if gate_record["warning"] != (dense["endpoints"][endpoint]["status"] != "completed"):
                raise ValueError("Old gate label differs from its locked prediction")
            base = {"case_id": cid, "drug": row["drug"], "cluster_id": cases[cid]["cluster_id"], "sampling": sampling,
                "endpoint": endpoint, "features": features[endpoint], "feature_audit": support,
                "dense_warning": gate[cid, sampling, endpoint]["warning"],
                "matched_prediction_weak_tail": bool(scored.get("weak_tail")),
                "development_sources": {k: source[k] for k in ("prediction", "endpoints", "mapping", "request")},
                "training_source": transport["computations"][identity]["training_source"],
                "clinical_label_source": state["results"]}
            if not _positive(scored.get("reference")) or not _positive(scored.get("error")):
                excluded.append({**base, "reason": "missing_reference_or_matched_error"})
                continue
            records.append({**base, "matched_fold_error": scored["error"]})
    module_path = Path(__file__).resolve()
    pins[str(module_path.relative_to(root))] = hashlib.sha256(module_path.read_bytes()).hexdigest()
    return {"records": records, "excluded": excluded, "source_pins": pins,
            "source_scope": "Original always-three full/five only; 20 predictions, not all 58 aliases."}
