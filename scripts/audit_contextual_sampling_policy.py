#!/usr/bin/env python3
"""Independent JSON/NumPy audit: no production learner, runner or PBPK import.

Reconstructs weighted ridge normal equations from allowed training drugs,
inner/outer decisions, candidate scores and study-cluster bootstrap statistics.
Only audit.json is written; every experiment artifact remains unchanged.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/blueprint_design/sampling_policy_20261009"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(reference):
    path = (ROOT / reference["path"]).resolve()
    assert path.is_relative_to(ROOT) and sha(path) == reference["sha256"], path
    return read(path)


def close(a, b, description):
    assert np.allclose(a, b, rtol=2e-9, atol=2e-11, equal_nan=False), description


def choose_time(design):
    t = np.asarray(design["times"])
    distances = np.asarray(design["X"])[:, 1]
    return min(np.flatnonzero(distances >= distances.max() - 1e-12), key=lambda i: t[i])


def group_map(raw, cases):
    """Rebuild connected components directly from metadata-only source fields."""
    source = next(r for r in raw["source_manifest"] if "clinical_identity_fields_only_not_predictors" in r["roles"])
    audit = checked(source)
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def join(x, y):
        x, y = find(x), find(y)
        parent[max(x, y)] = min(x, y)
    for case in cases:
        join("case:" + case["case_id"], "compound:" + case["compound_id"])
    for record in audit["records"]:
        cid = record["case_id"]
        identities = []
        for field in ("doi", "pmid", "pmcid", "registry", "study_id"):
            value = record["fields"].get(field, {}).get("value")
            if not value:
                continue
            value = re.sub(r"\s+", " ", str(value).strip().lower())
            if field == "study_id" and len(re.sub(r"[^a-z0-9]", "", value)) <= 3:
                continue
            identities.append(("study" if field == "study_id" else field) + ":" + value)
        for identity in identities:
            join("case:" + cid, identity)
    components = defaultdict(list)
    for case in cases:
        components[find("case:" + case["case_id"])].append(case["case_id"])
    result = {}
    for members in components.values():
        group = "drugstudy:" + hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()[:16]
        result.update({cid: group for cid in members})
    assert result == raw["grouping"]["case_to_group"]
    return result


class IndependentRidge:
    """Sufficient statistics avoid repeating expensive production training."""
    def __init__(self, designs, outcomes, groups, label_cap):
        self.designs, self.outcomes, self.groups = designs, outcomes, groups
        self.statistics, self.cache = {}, {}
        by_drug = defaultdict(list)
        for design in designs.values():
            by_drug[design["case_id"]].append(design)
        for cid, menus in by_drug.items():
            a, b, rows = np.zeros((12, 12)), np.zeros((12, 2)), 0
            for d in menus:
                x = np.asarray(d["X"], float)
                x = x - x[d["time_index"]]
                labels = [outcomes[d["design_key"], d["design_key"] + f"__point_{i:02d}"] for i in d["candidate_ids"]]
                errors = np.asarray([[r["endpoints"]["cmax"]["error"], r["curve_metrics"]["gmfe"]] for r in labels])
                target = np.clip(np.log(errors[d["time_index"]] / errors), -label_cap, label_cap)
                weight = 1. / len(menus) / len(x)
                a += weight * x.T @ x
                b += weight * x.T @ target
                rows += len(x)
            self.statistics[cid] = a, b, rows

    def fit(self, ids, dimensions, alpha):
        ids = tuple(sorted(ids))
        key = ids, dimensions, float(alpha)
        if key not in self.cache:
            # Every summand belongs to this exact allowed training set.
            a = sum(self.statistics[c][0][:dimensions, :dimensions] for c in ids) / len(ids)
            b = sum(self.statistics[c][1][:dimensions] for c in ids) / len(ids)
            scale = np.sqrt(np.diag(a))
            scale[scale < 1e-12] = 1.
            coefficient = np.linalg.solve(a / np.outer(scale, scale) + alpha * np.eye(dimensions), b / scale[:, None])
            self.cache[key] = scale, coefficient
        return self.cache[key]

    def decide(self, ids, dimensions, alpha, design, curve_guard=True, study_guard=True):
        ids = tuple(sorted(ids))
        groups = sorted({self.groups[c] for c in ids})
        subsets = [ids]
        if study_guard:
            subsets += [tuple(c for c in ids if self.groups[c] != g) for g in groups]
        x = np.asarray(design["X"])[:, :dimensions]
        x = x - x[design["time_index"]]
        predicted = []
        for subset in subsets:
            scale, coefficient = self.fit(subset, dimensions, alpha)
            predicted.append(x / scale @ coefficient)
        predicted = np.asarray(predicted)
        lower = predicted.min(axis=0)
        eligible = lower[:, 0] > 1e-12
        if curve_guard:
            eligible &= lower[:, 1] >= -math.log(1.05)
        candidates = np.flatnonzero(eligible)
        j = design["time_index"]
        if len(candidates):
            best = lower[candidates, 0].max()
            j = min((i for i in candidates if lower[i, 0] >= best - 1e-12), key=lambda i: design["times"][i])
        return int(j), lower, predicted[0], eligible


def bootstrap(values, groups, seed, resamples):
    ids = sorted(values)
    logs = np.asarray([math.log(values[c]) for c in ids])
    cluster_ids = sorted({groups[c] for c in ids})
    positions = [[i for i, cid in enumerate(ids) if groups[cid] == group] for group in cluster_ids]
    sums = np.asarray([sum(logs[ix]) for ix in positions])
    counts = np.asarray([len(ix) for ix in positions])
    take = np.random.default_rng(seed).integers(0, len(cluster_ids), (resamples, len(cluster_ids)))
    draws = np.exp(sums[take].sum(axis=1) / counts[take].sum(axis=1))
    return dict(n=len(ids), clusters=len(cluster_ids), ratio=float(np.exp(logs.mean())),
        ci95=np.quantile(draws, [.025, .975]).tolist(), improved=int(sum(logs < -1e-10)),
        worse=int(sum(logs > 1e-10)), tied=int(sum(abs(logs) <= 1e-10)))


def audit():
    start = time.monotonic()
    state = read(RUN / "state.json")
    assert state["status"] == "complete", "Wait for completed evaluation"
    protocol, results = checked(state["protocol"]), checked(state["results"])
    original_hashes = {str(p): sha(p) for p in (RUN / "protocol.json", RUN / "inputs.json", RUN / "state.json", RUN / "results.json")}
    for path, digest in protocol["source_pins"].items():
        assert sha(ROOT / path) == digest, path
    inputs = checked(protocol["inputs"])
    raw = inputs["raw"]
    previous = checked(protocol["previous_protocol"])
    groups = group_map(raw, previous["cases"])
    designs = {d["design_key"]: d for d in inputs["designs"]}
    source = checked(protocol["outcome_store"])
    outcomes = {(r["design_key"], r["candidate"]): r for r in source["records"]}
    assert len(designs) == 60 and len(outcomes) == 586
    prediction_lock = checked(source["prediction_lock"])
    assert prediction_lock["locked_at"] <= source["scored_at"] < protocol["created_at"]
    assert protocol["created_at"] <= results["choices_locked_at"] <= results["scored_at"]
    assert results["choices_locked_at"] == state["choices_locked_at"]

    # Independently reconstruct all four new features from preacquisition data.
    for r in raw["designs"]:
        d = designs[r["design_key"]]
        t, v = np.asarray(d["times"]), np.asarray(r["visible"]["time_h"])
        visible_doc = checked(r["visible"]["source"])["visible_training"]
        assert len(visible_doc["observations"]) == 4
        close(v, [o["time_min"] / 60 for o in visible_doc["observations"]], "Visible time conversion")
        y = np.asarray(r["visible"]["concentration_mg_l"])
        close(y, [o["concentration_umol_per_l"] * visible_doc["molecular_weight_g_mol"] / 1000 for o in visible_doc["observations"]], "Visible concentration conversion")
        residual = np.log(y / np.asarray(r["baseline"]["visible_prediction_mg_l"]))
        weights = -.5 * ((np.log(t[:, None]) - np.log(v[None, :])) / protocol["algorithm"]["kernel_bandwidth_log"]) ** 2
        weights = np.exp(weights - weights.max(axis=1, keepdims=True))
        proxy = weights @ residual / weights.sum(axis=1)
        lc, lg = np.log(r["committee"]["concentrations_mg_l"]), np.log(r["committee"]["cmax_ng_ml"])
        response = np.mean((lc - lc.mean(axis=0)) * (lg - lg.mean())[:, None], axis=0) / (lc.var(axis=0) + .1 ** 2)
        centered = np.log(v) - np.log(v).mean()
        slope = centered @ (np.log(y) - np.log(y).mean()) / (centered @ centered)
        distance = np.min(abs(np.log(t[:, None]) - np.log(v[None, :])), axis=1)
        features = np.column_stack((r["goals"]["cmax"]["X"], response * proxy, abs(response * proxy),
            np.sqrt(np.mean(residual ** 2)) * distance, slope * np.log(t / max(v))))
        close(features, d["X"], "Context features: " + d["design_key"])
        assert choose_time(d) == d["time_index"]
        assert d["drug_study_group"] == groups[d["case_id"]]
    assert raw["isolation"]["hidden_values_used"] is False and raw["isolation"]["target_observations_used"] is False

    ridge = IndependentRidge(designs, outcomes, groups, protocol["algorithm"]["label_cap_log"])
    coefficient_checks, inner_checks, outer_checks = 0, 0, 0
    folds = {}
    for key, reference in state["folds"].items():
        fold = checked(reference)
        folds[key] = fold
        spec = protocol["folds"][key]
        train, held = set(spec["train_case_ids"]), set(spec["heldout_case_ids"])
        assert fold["train_case_ids"] == spec["train_case_ids"] and fold["heldout_case_ids"] == spec["heldout_case_ids"]
        assert not train & held and not {groups[c] for c in train} & {groups[c] for c in held}
        assert fold["heldout_outcomes_used"] is False
        assert protocol["created_at"] <= fold["locked_at"] <= results["choices_locked_at"]
        menus = [d for d in designs.values() if d["case_id"] in train]
        held_menus = [d for d in designs.values() if d["case_id"] in held]
        assert set(fold["choices"]) == {d["design_key"] for d in held_menus}
        for dim in (8, 12):
            tuning, bundle = fold["tuning"][str(dim)], fold["models"][str(dim)]
            assert set(tuning["train_case_ids"]) == train
            assert set(tuning["train_groups"]) == {groups[c] for c in train}
            for deletion, model in [(None, bundle["full"])] + [(x["excluded_group"], x["model"]) for x in bundle["deletions"]]:
                allowed = {c for c in train if groups[c] != deletion}
                assert set(model["train_case_ids"]) == allowed
                assert set(model["train_groups"]) == {groups[c] for c in allowed}
                assert set(model["train_design_keys"]) == {d["design_key"] for d in menus if d["case_id"] in allowed}
                scale, coefficient = ridge.fit(allowed, dim, model["alpha"])
                close(scale, model["scale"], "Training-only scales")
                close(coefficient, model["coef"], "Training-only normal equations")
                coefficient_checks += 1
            assert [trial["alpha"] for trial in tuning["trials"]] == protocol["algorithm"]["alpha_grid"]
            eligible_trials = []
            for trial in tuning["trials"]:
                values = defaultdict(list)
                assert len(trial["heldout_decisions"]) == len(menus)
                for entry in trial["heldout_decisions"]:
                    cid, regime, group = entry["case_id"], entry["regime"], entry["group"]
                    assert cid in train and group == groups[cid]
                    d = designs[cid + "__" + regime]
                    inner_train = {c for c in train if groups[c] != group}
                    index, _, _, _ = ridge.decide(inner_train, dim, trial["alpha"], d)
                    assert index == entry["index"], (key, dim, trial["alpha"], d["design_key"], "inner selection")
                    record = outcomes[d["design_key"], d["design_key"] + f'__point_{d["candidate_ids"][index]:02d}']
                    control = outcomes[d["design_key"], d["design_key"] + f'__point_{d["candidate_ids"][d["time_index"]]:02d}']
                    for endpoint in ("cmax", "curve"):
                        get = (lambda r: r["curve_metrics"]["gmfe"]) if endpoint == "curve" else (lambda r: r["endpoints"]["cmax"]["error"])
                        log_ratio = math.log(get(record) / get(control))
                        close(log_ratio, entry[endpoint + "_log_ratio"], "Inner label")
                        values[regime, endpoint].append(log_ratio)
                    inner_checks += 1
                means = {reg: {ep: float(np.mean(values[reg, ep])) for ep in ("cmax", "curve")} for reg in ("early_four", "late_four")}
                for reg in means:
                    for ep in means[reg]:
                        close(means[reg][ep], trial["means"][reg][ep], "Inner aggregate")
                eligible = all(means[r]["curve"] <= math.log(1.01) for r in means)
                assert eligible == trial["eligible"]
                if eligible and means["late_four"]["cmax"] < -1e-12:
                    eligible_trials.append(trial)
            expected = min(eligible_trials, key=lambda x: (x["means"]["late_four"]["cmax"], -x["alpha"]))["alpha"] if eligible_trials else None
            assert tuning["alpha"] == expected
        for d in held_menus:
            for method, dim, cg, sg in [("contextual", 12, True, True), ("original_features", 8, True, True),
                                        ("without_curve_guard", 12, False, True), ("without_study_guard", 12, True, False)]:
                stored = fold["choices"][d["design_key"]][method]
                alpha = fold["tuning"][str(dim)]["alpha"]
                if alpha is None:
                    assert stored["selected_index"] == d["time_index"]
                else:
                    index, lower, full, eligible = ridge.decide(train, dim, alpha, d, cg, sg)
                    assert index == stored["selected_index"]
                    close(lower, stored["minimum_gain_predictions"], "Outer robust scores")
                    close(full, stored["full_gain_predictions"], "Outer full scores")
                    assert eligible.tolist() == stored["eligible"]
                outer_checks += 1
        print(json.dumps(dict(audited_folds=len(folds), total=len(protocol["folds"])), ensure_ascii=False), flush=True)

    assert set(folds) == set(protocol["folds"])
    previous_choices = checked(protocol["previous_choices"])["choices"]
    seen = set()
    for row in results["rows"]:
        cid, ev, reg, method = row["case_id"], row["evaluation"], row["regime"], row["method"]
        d = designs[cid + "__" + reg]
        fold = next(f for f in folds.values() if f["evaluation"] == ev and cid in f["heldout_case_ids"])
        if method == "no_addition":
            index = None
        elif method == "time":
            index = d["time_index"]
        elif method == "frozen_stable":
            index = previous_choices[d["design_key"]]["cmax"]["decision"]["selected_index"]
        else:
            index = fold["choices"][d["design_key"]][method]["selected_index"]
        candidate = "baseline" if index is None else d["design_key"] + f'__point_{d["candidate_ids"][index]:02d}'
        expected = outcomes[d["design_key"], candidate]
        assert row["candidate"] == candidate and row["time_h"] == (None if index is None else d["times"][index])
        assert row["source"] == expected["source"] and row["endpoints"] == expected["endpoints"]
        errors = {ep: value["error"] for ep, value in expected["endpoints"].items()}
        errors["curve"] = expected["curve_metrics"]["gmfe"]
        assert errors == row["errors"] and row["changed"] == (index != d["time_index"])
        assert row["study_group"] == groups[cid]
        key = ev, reg, cid, method
        assert key not in seen
        seen.add(key)
    assert len(seen) == 700

    paired, deletion = {}, {}
    bootstrap_checks = 0
    for scope, methods in results["summaries"].items():
        ev, reg = scope.split("__")
        rows = [r for r in results["rows"] if r["evaluation"] == ev and r["regime"] == reg]
        lookup = {(r["case_id"], r["method"]): r for r in rows}
        ids = sorted({r["case_id"] for r in rows})
        for method, summary in methods.items():
            selected = [lookup[c, method] for c in ids]
            assert summary["n"] == len(selected) and summary["changed"] == sum(r["changed"] for r in selected)
            for endpoint, expected in summary["endpoints"].items():
                values = [r["errors"][endpoint] for r in selected if r["errors"][endpoint] is not None]
                close(math.exp(np.mean(np.log(values))), expected["gmfe"], "GMFE")
                assert len(values) == expected["n"] and sum(v <= 2 for v in values) == expected["within_two"]
                ratios = {c: lookup[c, method]["errors"][endpoint] / lookup[c, "time"]["errors"][endpoint]
                          for c in ids if lookup[c, method]["errors"][endpoint] is not None and lookup[c, "time"]["errors"][endpoint] is not None}
                observed = bootstrap(ratios, groups, **protocol["bootstrap"])
                for field, value in observed.items():
                    close(value, expected["paired_vs_time"][field], "Clustered bootstrap " + field)
                bootstrap_checks += 1
                if method == "contextual" and endpoint in ("cmax", "curve"):
                    paired[scope + "__" + endpoint] = observed
                    items = []
                    for group in sorted({groups[c] for c in ratios}):
                        keep = {c: v for c, v in ratios.items() if groups[c] != group}
                        excluded = [c for c in ratios if groups[c] == group]
                        items.append(dict(excluded_group=group, excluded_case_ids=excluded,
                            excluded_drugs=[lookup[c, method]["drug"] for c in excluded],
                            estimate=bootstrap(keep, groups, **protocol["bootstrap"])))
                    deletion[scope + "__" + endpoint] = dict(all_drugs=observed, leave_one_study_out=items,
                        minimum_ratio=min(x["estimate"]["ratio"] for x in items), maximum_ratio=max(x["estimate"]["ratio"] for x in items),
                        improved_drugs=[lookup[c, method]["drug"] for c, v in ratios.items() if math.log(v) < -1e-10],
                        worse_drugs=[lookup[c, method]["drug"] for c, v in ratios.items() if math.log(v) > 1e-10],
                        per_case_ratio={lookup[c, method]["drug"]: v for c, v in ratios.items()})
    primary_cmax = paired["original10_to_added20__late_four__cmax"]
    primary_curve = paired["original10_to_added20__late_four__curve"]
    support = primary_cmax["ci95"][1] < 1 and primary_curve["ratio"] <= 1.01
    assert support == results["primary_incremental_support"]
    assert original_hashes == {name: sha(name) for name in original_hashes}
    out = dict(version="independent-contextual-policy-audit-1", status="passed", created_at=time.time(),
        source_results=state["results"], audit_script_sha256=sha(Path(__file__)),
        checks=dict(source_pins=len(protocol["source_pins"]), feature_menus=60, source_groups=len(set(groups.values())),
            outer_folds=len(folds), independent_ridge_coefficient_sets=coefficient_checks,
            inner_cv_choices=inner_checks, outer_choices=outer_checks, scored_rows=len(seen),
            summary_bootstraps=bootstrap_checks, production_modules_imported=False, pbpk_calls=0,
            sources_unchanged=True, selection_lock_precedes_scoring=True,
            prior_counterfactual_scores_are_steward_only=True, independent_linear_fits_cached=len(ridge.cache)),
        primary_incremental_support=support, primary_cmax=primary_cmax, primary_curve=primary_curve,
        paired_results=paired, deletion_sensitivity=deletion,
        limits=["This verifies computational isolation and statistics, not a previously unviewed clinical test.",
            "Study components use recorded source identities; absent identifiers cannot prove participant independence.",
            "Study-deletion lower model scores are heuristics, not calibrated gain probabilities.",
            "Counterfactual labels existed before selector training; only allowed training drugs enter independent coefficient reconstruction."],
        wall_seconds=time.monotonic() - start)
    temporary = RUN / "audit.json.tmp"
    temporary.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(RUN / "audit.json")
    print(json.dumps(dict(status=out["status"], checks=out["checks"], primary_incremental_support=support,
        primary_cmax=primary_cmax, primary_curve=primary_curve, wall_seconds=out["wall_seconds"]), ensure_ascii=False))


if __name__ == "__main__":
    audit()
