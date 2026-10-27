"""Immutable, outcome-separated inputs and connected-component nested folds.

Only the frozen common request/catalogue texts and their public metadata are
features. Document judgments and completed reference-diagnostic errors live in
separate arrays/records. No source packet, hidden judge support, sealed target,
native model, or supplementary sensitivity outcome is opened by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = Path("experiments/model_recommendation")
CORE = BASE / "frontier_core_145_20260921"
FRONTIER = BASE / "frontier_comparison_20260914"
EXECUTION = BASE / "execution_validation_20261001"
FOLLOWUP = EXECUTION / "followup_20261003"
DEFAULT_OUTPUT = ROOT / BASE / "pbpk_match_v2"
VARIANTS = ("original", "reordered", "missing_evidence")
FIELDS = ("compounds", "species", "routes", "populations", "endpoints", "enzymes",
          "transporters", "process_types", "formulations", "model_structure", "software", "compartments")
NUMERIC_FIELDS = ("dose_mg", "dose_mg_per_kg", "dose_mg_per_m2", "interval_h",
                  "infusion_h", "sampling_start_h", "sampling_end_h", "auc_start_h", "auc_end_h", "dose_count")
CATEGORIES = {
    "dose_unit": ("mg", "mg/kg", "mg/m2", "other"),
    "food": ("fasted", "fed", "other"),
    "formulation": ("tablet", "capsule", "solution", "suspension", "extended_release", "immediate_release", "other"),
    "endpoint_window": ("tau", "fixed", "last", "infinity", "peak", "steady_state", "normalized", "curve"),
}
ARRAY_NAMES = ("pair_features", "pair_features_by_variant", "grades", "applicability")
SCHEMA = "pbpk-match-v2-outcome-separated-data-1"
SEED = "pbpk-match-v2-connected-input-groups-5x3-v1"

# These are immutable existing experiment boundaries, not live label journals.
PINNED = {
    FRONTIER / "inputs/queries.jsonl": "b6ea181c4cef5844a08534c2fd18e6f1c500f2cc37b9142443d0c8ea7206ff7c",
    FRONTIER / "inputs/candidates.jsonl": "5ccfc1d7b8a1a0962504f991676f88f94044d8e868b0348768314ea418025b15",
    CORE / "inputs/queries.jsonl": "d3fa1b9408df51b9b11eb3c86fa129e0a369b4f5da5d8dfb63e547917b64d159",
    CORE / "inputs/candidates.jsonl": "5ccfc1d7b8a1a0962504f991676f88f94044d8e868b0348768314ea418025b15",
    CORE / "selection_manifest.json": "3699bb1b128d3ee22b581374881ba8f3af9d0591ac79b487072413f0969b08d8",
    CORE / "assessment_repaired_v2/judgments.jsonl": "a3efec0f49da047947edb36367d8fbf3537f8d06ddaecc1996c597f2c9bbe071",
    CORE / "assessment_repaired_v2/judge_protocol.json": "4814880e144217c03acdafb499f66db417017b35c1196e66664465b32ba27db4",
    CORE / "assessment_repaired_v2/automatic_evaluation.json": "28167036e1bf7523a49c1b2554e06e5b2b051d6c0df7224c60dbe62a0df11dee",
    EXECUTION / "inputs/scenarios.jsonl": "c10062f38430c9874bd4d7043da7cc45b2cf6064f14e907e30cbd94187c60fe6",
    EXECUTION / "rankings/inputs/queries.jsonl": "05a97b89190573e5f7f1cf93876efdbaec86d5a324558d044322ca3810332e5a",
    FOLLOWUP / "primary/results.json": "8798a53c9a452817ba820cdadf948c9edcf5c5beff939c7813f2cd0cd46292e7",
    FOLLOWUP / "completion.json": "c75d5bc15d1a84ec9cbfc174ffab02c6b9fff3ba1a1f988d680a086ed696c914",
}


def scoped(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT) or "sealed" in path.relative_to(ROOT).parts:
        raise ValueError("Dataset access must stay in this project and outside sealed files")
    return path


def digest(path):
    h = hashlib.sha256()
    with scoped(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(scoped(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in scoped(path).read_text().splitlines() if line.strip()]


def normalized(values):
    if isinstance(values, str):
        values = [values]
    return sorted({re.sub(r"\s+", " ", str(v).strip().casefold()) for v in values or [] if str(v).strip()})


def fields_for(record, variant="original"):
    features = record.get("variants", {}).get(variant, {}).get("features", record.get("features", {}))
    result = {key: normalized(features.get(key, [])) for key in FIELDS}
    result["compounds"] = normalized(record.get("compounds", result["compounds"]))
    return result


def text_for(record, variant="original"):
    return record.get("variants", {}).get(variant, {}).get("text", record["text"])


def without_conditions(text):
    """Ablation removes numeric values AND food/formulation condition wording.

    Drug, route and endpoint types remain; numeric endpoint windows are masked.
    Original texts in the dataset are never rewritten.
    """
    lines = []
    for line in text.splitlines():
        if line.startswith(("Food context:", "Formulation categories explicitly named:",
                            "Regimen and observation origin:", "Nominal sampling hours:")):
            lines.append(line.split(":", 1)[0] + ": [condition omitted]")
            continue
        if line.startswith("Administration: "):
            try:
                route = json.loads(line.split(": ", 1)[1]).get("route", "unspecified")
            except (ValueError, TypeError, AttributeError):
                route = "unspecified"
            lines.append("Administration route: " + str(route))
            continue
        line = re.sub(r"(?i)\b(?:fasted|fasting|fed|tablet|capsule|solution|suspension|extended.release|immediate.release)\b", "[condition omitted]", line)
        line = re.sub(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", "[number]", line)
        lines.append(line)
    return "\n".join(lines)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def condition_values(record, variant="original", design=None):
    """Extract input conditions only; missing is None, never an inferred zero.

    Clinical structured metadata is already frozen before execution. Candidate
    conditions use only the same public catalogue text/features supplied to all
    comparators; no native parameter/simulation or hidden evidence is consulted.
    """
    text = text_for(record, variant)
    design = copy.deepcopy(design or {})
    # The common clinical query text also carries the exact pre-outcome design.
    for prefix, key in (("Administration: ", "dose"), ("Regimen and observation origin: ", "regimen")):
        if key not in design:
            line = next((s[len(prefix):] for s in text.splitlines() if s.startswith(prefix)), None)
            if line:
                try:
                    design[key] = json.loads(line)
                except ValueError:
                    pass
    values = dict.fromkeys(NUMERIC_FIELDS)
    cats = {k: [] for k in CATEGORIES}
    dose, regimen = design.get("dose") or {}, design.get("regimen") or {}
    value, unit = dose.get("value"), str(dose.get("unit", "")).casefold().replace(" ", "")
    if _finite(value) and value >= 0:
        conversions = {"mg": ("dose_mg", 1), "g": ("dose_mg", 1000), "ug": ("dose_mg", .001),
                       "µg": ("dose_mg", .001), "mg/kg": ("dose_mg_per_kg", 1), "mg/m2": ("dose_mg_per_m2", 1)}
        if unit in conversions:
            key, scale = conversions[unit]
            values[key] = float(value) * scale
        cats["dose_unit"] = [unit if unit in CATEGORIES["dose_unit"] else "other"]
    times = regimen.get("dose_times_h")
    if isinstance(times, list) and times and all(_finite(x) for x in times):
        values["dose_count"] = float(len(times))
        differences = [b - a for a, b in zip(times, times[1:])]
        if differences and all(d > 0 and math.isclose(d, differences[0], abs_tol=1e-10, rel_tol=1e-12) for d in differences):
            values["interval_h"] = differences[0]
    for key, raw in (("interval_h", regimen.get("dosing_interval_h")), ("infusion_h", regimen.get("infusion_duration_h"))):
        if _finite(raw) and raw >= 0:
            values[key] = float(raw)
    sample_times = (design.get("sampling") or {}).get("times_h")
    if isinstance(sample_times, list) and sample_times and all(_finite(t) for t in sample_times):
        values["sampling_start_h"], values["sampling_end_h"] = float(min(sample_times)), float(max(sample_times))
    food = design.get("food")
    if food is None:
        food = next((s.split(":", 1)[1] for s in text.splitlines() if s.startswith("Food context:")), "")
    food = str(food).casefold()
    if re.search(r"\bfast(?:ed|ing)\b", food):
        cats["food"] = ["fasted"]
    elif re.search(r"\bfed\b|fed_", food):
        cats["food"] = ["fed"]
    elif food and food not in ("none", "unspecified") and not re.search(r"not.specified|not.reported|unknown", food):
        cats["food"] = ["other"]
    formulations = fields_for(record, variant)["formulations"]
    formulation = str(design.get("formulation") or dose.get("formulation") or " ".join(formulations)).casefold()
    for category in CATEGORIES["formulation"][:-1]:
        pattern = category.replace("_", r"[ _-]")
        if re.search(pattern, formulation):
            cats["formulation"].append(category)
    if re.search(r"\b(?:sr|er|xr|cr)\b", formulation):
        cats["formulation"].append("extended_release")
    if not cats["formulation"] and formulation and formulation not in ("none", "unspecified"):
        cats["formulation"] = ["other"]
    endpoint_text = " ".join(s.split(":", 1)[1] for s in text.splitlines() if s.startswith(("Target endpoints:", "Output endpoints:"))).casefold()
    for category, pattern in (("tau", r"auc.?tau"), ("last", r"auc.?last"), ("infinity", r"auc.?inf"),
                              ("peak", r"cmax|peak"), ("steady_state", r"cmax_ss|auc_tau_ss"),
                              ("normalized", r"normalized"), ("curve", r"concentration.?time")):
        if re.search(pattern, endpoint_text):
            cats["endpoint_window"].append(category)
    windows = re.findall(r"auc[_ -]?(\d+(?:\.\d+)?)[_ –-]+(\d+(?:\.\d+)?)\s*h", endpoint_text)
    if len(set(windows)) == 1:
        values["auc_start_h"], values["auc_end_h"] = map(float, windows[0])
        cats["endpoint_window"].append("fixed")
    elif "tau" in cats["endpoint_window"] and values["interval_h"] is not None:
        values["auc_start_h"], values["auc_end_h"] = 0.0, values["interval_h"]
    return {"numeric": values, "categorical": {k: sorted(set(v)) for k, v in cats.items()}}


def feature_names():
    names = [f"{key}.{suffix}" for key in FIELDS for suffix in ("query_present", "candidate_present", "jaccard", "query_coverage", "listed_disjoint")]
    for key in NUMERIC_FIELDS:
        names += [f"condition.{key}.{s}" for s in ("query_log1p", "query_missing", "candidate_log1p", "candidate_missing", "both_known", "absolute_log1p_difference", "exact_match")]
    for key, categories in CATEGORIES.items():
        names += [f"condition.{key}.{side}.{category}" for category in categories for side in ("query", "candidate", "shared")]
        names += [f"condition.{key}.{side}_missing" for side in ("query", "candidate")]
    return names


def pair_feature_row(query, candidate, variant="original", design=None, candidate_condition=None,
                     query_condition=None, query_fields=None, candidate_fields=None):
    qf, cf = query_fields or fields_for(query, variant), candidate_fields or fields_for(candidate)
    row = []
    for key in FIELDS:
        a, b = set(qf[key]), set(cf[key])
        row += [bool(a), bool(b), len(a & b) / len(a | b) if a | b else 0,
                len(a & b) / len(a) if a else 0, bool(a and b and not a & b)]
    qc = query_condition or condition_values(query, variant, design)
    cc = candidate_condition or condition_values(candidate)
    for key in NUMERIC_FIELDS:
        a, b = qc["numeric"][key], cc["numeric"][key]
        ax = math.copysign(math.log1p(abs(a)), a) if a is not None else 0
        bx = math.copysign(math.log1p(abs(b)), b) if b is not None else 0
        both = a is not None and b is not None
        row += [ax, a is None, bx, b is None, both, abs(ax - bx) if both else 0, both and a == b]
    for key, categories in CATEGORIES.items():
        a, b = set(qc["categorical"][key]), set(cc["categorical"][key])
        for category in categories:
            row += [category in a, category in b, category in a & b]
        row += [not a, not b]
    return np.asarray(row, dtype=np.float32)


def atomic_property(evidence):
    """Identify explicitly typed chemical properties, not narrative evidence.

    A ChEMBL hydrogen-bond count of '0' is not a shared research passage. Its
    raw hash is retained, but only identifies evidence within the same target,
    source and property. Abstract/full-text hashes remain global.
    """
    patterns = {"chembl": r"#json_path=\$\.molecules\[\d+\]\.molecule_properties\.([A-Za-z_][A-Za-z_0-9]*)$",
                "pubchem": r"#json_path=\$\.PropertyTable\.Properties\[\d+\]\.([A-Za-z_][A-Za-z_0-9]*)$"}
    source = evidence.get("source_id")
    match = re.search(patterns[source], evidence.get("source_location", "")) if source in patterns else None
    return (source, match.group(1)) if match else None


def research_keys(query, scenario=None):
    """Conservative input-only linkage. Coarse database source_id is NOT a study."""
    keys = {"compound:" + c for c in normalized(query.get("compounds", []))}
    for e in query.get("provenance", {}).get("evidence", []):
        if e.get("evidence_text_sha256"):
            atomic = atomic_property(e)
            if atomic:
                namespace = json.dumps([*atomic, normalized(query.get("compounds", [])), e["evidence_text_sha256"]], separators=(",", ":"))
                keys.add("atomic_property_evidence:" + namespace)
            else:
                keys.add("evidence:" + e["evidence_text_sha256"])
        location = e.get("source_location", "")
        source = e.get("source_id", "")
        # Preserve article/record indices in multi-article acquisition files.
        patterns = {"pmc_oa": r"^(.+#xml_path=//article\[\d+\])", "pubmed": r"^(.+#xml_path=//PubmedArticle\[\d+\])",
                    "pkdb": r"^(.+#json_path=\$\.data\.data\[\d+\])", "openfda_drug_label": r"^(.+#json_path=\$\.results\[\d+\])"}
        match = re.match(patterns[source], location) if source in patterns else None
        if match:
            keys.add("source_record:" + match.group(1))
        elif source in ("dailymed", "osp_pbpk_model_library") and location:
            keys.add("source_document:" + location.split("#", 1)[0])
    if scenario:
        for key in ("study_id", "publication_cluster_id", "declared_publication_cluster_id"):
            if scenario.get(key):
                keys.add("study:" + scenario[key].casefold())
        p = scenario.get("protocol_fields", {})
        for key, value in scenario.get("publication_ids", {}).items():
            if value:
                keys.add("study:" + key.casefold() + ":" + str(value).casefold())
        for trial in p.get("trial_cluster_ids", []):
            keys.add("trial:" + str(trial).casefold())
        for key in ("source_article_sha256",):
            if p.get(key):
                keys.add("article_sha256:" + p[key])
        article = p.get("source_article")
        if isinstance(article, dict) and article.get("sha256"):
            keys.add("article_sha256:" + article["sha256"])
    return sorted(keys)


def connected_groups(key_rows):
    parents, owner = list(range(len(key_rows))), {}
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for i, keys in enumerate(key_rows):
        for key in keys:
            if key in owner:
                parents[find(i)] = find(owner[key])
            else:
                owner[key] = i
    members = defaultdict(list)
    for i in range(len(key_rows)):
        members[find(i)].append(i)
    ids = {}
    for indices in members.values():
        keys = sorted({k for i in indices for k in key_rows[i]})
        group = hashlib.sha256(json_bytes(keys)).hexdigest()
        ids.update({i: group for i in indices})
    return [ids[i] for i in range(len(key_rows))]


def nested_folds(groups, outer_count=5, inner_count=3, core_indices=(), execution_indices=()):
    """Input-size balanced, label-blind, indivisible connected components."""
    by_group = defaultdict(list)
    for i, group in enumerate(groups):
        by_group[group].append(i)
    if len(by_group) < outer_count:
        raise ValueError("Too few independent input groups for outer folds")
    core, execution = set(core_indices), set(execution_indices)
    def assign(keys, count, salt):
        bins = [[] for _ in range(count)]
        totals = [max(1, len(core & {i for g in keys for i in by_group[g]})),
                  max(1, len(execution & {i for g in keys for i in by_group[g]})),
                  sum(len(by_group[g]) for g in keys)]
        weights = {g: (len(core & set(by_group[g])), len(execution & set(by_group[g])), len(by_group[g])) for g in keys}
        keys = sorted(keys, key=lambda g: (tuple(-x for x in weights[g]), hashlib.sha256((SEED + salt + g).encode()).hexdigest()))
        loads = [[0, 0, 0] for _ in range(count)]
        for group in keys:
            target = min(range(count), key=lambda k: (sum(loads[k][j] / totals[j] for j in range(3)), k))
            bins[target].extend(by_group[group])
            loads[target] = [a + b for a, b in zip(loads[target], weights[group])]
        return [sorted(b) for b in bins]
    outer = assign(list(by_group), outer_count, "outer")
    folds = []
    all_indices = set(range(len(groups)))
    for i, test in enumerate(outer):
        train = sorted(all_indices - set(test))
        train_groups = {groups[j] for j in train}
        if len(train_groups) < inner_count:
            raise ValueError("Too few independent input groups for inner folds")
        validation = assign(train_groups, inner_count, f"inner-{i}")
        folds.append({"outer_test": test, "outer_train": train,
                      "inner": [{"train": sorted(set(train) - set(v)), "validation": v} for v in validation]})
    return folds


def document_labels(rows, query_ids, candidate_ids, core_ids, protocol_sha256):
    qi, ci = {s: i for i, s in enumerate(query_ids)}, {s: i for i, s in enumerate(candidate_ids)}
    grades = np.full((len(qi), len(ci)), -1, dtype=np.int8)
    applicability = np.full_like(grades, -1)
    expected = {(q, c) for q in core_ids for c in candidate_ids}
    seen = set()
    for row in rows:
        key = row["query_id"], row["candidate_id"]
        if key not in expected or key in seen or row.get("judge") != "primary" or row.get("protocol_sha256") != protocol_sha256:
            raise ValueError("Document label identity, duplicate, or protocol mismatch")
        seen.add(key)
        if row["status"] == "invalid_output":
            continue
        if row["status"] not in ("judged", "insufficient_evidence"):
            raise ValueError("Unexpected document label status")
        i, j = qi[key[0]], ci[key[1]]
        grade = row.get("grade")
        if grade is not None and (type(grade) is not int or grade not in range(4)):
            raise ValueError("Invalid ordinal document grade")
        grades[i, j] = -1 if grade is None else grade
        app = row.get("documented_applicability")
        if app is not None and type(app) is not bool:
            raise ValueError("Invalid applicability label")
        applicability[i, j] = 2 if app is None else (0 if app else 1)
    if seen != expected:
        raise ValueError("Document judgments do not cover the complete core Cartesian product")
    return grades, applicability


def _safe_record(row, candidate=False):
    allowed = ("candidate_id", "family_id", "family_name", "compounds", "features", "text") if candidate else (
        "query_id", "compound_id", "compounds", "task_type", "features", "text")
    result = {k: copy.deepcopy(row[k]) for k in allowed if k in row}
    if not candidate:
        result["variants"] = {v: {k: copy.deepcopy(value) for k, value in row.get("variants", {}).get(v, {}).items() if k in ("text", "features")}
                              for v in VARIANTS[1:]}
    return result


def _scenario_metadata(scenario):
    # IDs/research metadata remain separate from model input records.
    allowed = ("scenario_id", "case_id", "compound", "study_id", "publication_cluster_id", "publication_ids",
               "cohort_id", "normalized_design", "source_protocol", "source_role", "catalog_candidate_ids", "curve_views")
    return {k: copy.deepcopy(scenario[k]) for k in allowed if k in scenario}


def _error_labels(report, queries, candidates):
    qi = {q["query_id"]: i for i, q in enumerate(queries)}
    ci = {c["candidate_id"]: i for i, c in enumerate(candidates)}
    errors, same_parent, seen = [], defaultdict(list), set()
    for pair in report["candidate_results"]:
        i, j = qi[pair["scenario_id"]], ci[pair["candidate_id"]]
        if pair.get("identity_match") is True:
            same_parent[i].append(j)
        for scalar in pair["scalars"]:
            if scalar["state"] != "valid":
                continue
            endpoint = scalar["endpoint_key"]
            identity = (i, j, endpoint)
            if identity in seen:
                raise ValueError("Duplicate exact endpoint error label")
            seen.add(identity)
            values = [scalar[k] for k in ("predicted", "observed", "absolute_log_error", "signed_log_error")]
            if not all(_finite(v) for v in values) or min(values[:2]) <= 0:
                raise ValueError("Invalid completed scalar values")
            signed = math.log(values[0]) - math.log(values[1])
            if not math.isclose(abs(signed), values[2], abs_tol=1e-12, rel_tol=1e-12) or not math.isclose(signed, values[3], abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError("Completed scalar label does not reproduce its physical values")
            errors.append({"query_index": i, "candidate_index": j, "scenario_id": pair["scenario_id"],
                           "candidate_id": pair["candidate_id"], "endpoint": endpoint, "endpoint_key": endpoint,
                           "source_endpoint": scalar["endpoint"], "abs_log_error": values[2], "absolute_log_error": values[2],
                           "signed_log_error": values[3], "predicted": values[0], "observed": values[1],
                           "study_id": pair["publication_cluster_id"], "compound_id": pair["compound"],
                           "unit": scalar.get("unit"), "source_unit": scalar.get("source_unit"),
                           "prediction_window_h": scalar.get("prediction_window_h"),
                           "comparison_stratum": scalar.get("comparison_stratum", "reference_diagnostic")})
    requested = {}
    for selection in report["selection"]:
        if selection["stratum"] != "reference_diagnostic":
            continue
        i = qi[selection["scenario_id"]]
        endpoints = list(selection["required_endpoints"])
        if i in requested and requested[i] != endpoints:
            raise ValueError("Requested endpoints vary by ranking method")
        requested[i] = endpoints
    return errors, dict(requested), {i: sorted(set(js)) for i, js in same_parent.items()}


def assemble(stability, core, clinical, candidates, scenarios, labels, report, protocol_sha256):
    """Pure assembly from already checked inputs; label values never enter folds."""
    queries, index = [], {}
    for q in stability + clinical:
        if q["query_id"] in index:
            if queries[index[q["query_id"]]] != q:
                raise ValueError("Conflicting records with one query ID")
        else:
            index[q["query_id"]] = len(queries)
            queries.append(q)
    for q in core:
        if q["query_id"] not in index or queries[index[q["query_id"]]] != q:
            raise ValueError("Core query is not an exact subset of the stability corpus")
    candidate_ids = [c["candidate_id"] for c in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Duplicate catalogue candidate ID")
    scenario_map = {s["scenario_id"]: s for s in scenarios}
    if set(scenario_map) != {q["query_id"] for q in clinical} or len(scenario_map) != len(scenarios):
        raise ValueError("Clinical query/scenario identity mismatch")
    key_rows = [research_keys(q, scenario_map.get(q["query_id"])) for q in queries]
    atomic_rows = [{"query_id": q["query_id"], "target_compounds": normalized(q.get("compounds", [])),
                    "source_id": e["source_id"], "property": atomic_property(e)[1], "evidence_text_sha256": e["evidence_text_sha256"]}
                   for q in queries for e in q.get("provenance", {}).get("evidence", [])
                   if atomic_property(e) and e.get("evidence_text_sha256")]
    groups = connected_groups(key_rows)
    core_indices = [index[q["query_id"]] for q in core]
    execution_indices = [index[q["query_id"]] for q in clinical]
    folds = nested_folds(groups, core_indices=core_indices, execution_indices=execution_indices)
    grades, apps = document_labels(labels, list(index), candidate_ids, [q["query_id"] for q in core], protocol_sha256)
    candidate_conditions = [condition_values(c) for c in candidates]
    candidate_fields = [fields_for(c) for c in candidates]
    features = np.empty((len(VARIANTS), len(queries), len(candidates), len(feature_names())), dtype=np.float32)
    for v, variant in enumerate(VARIANTS):
        for i, q in enumerate(queries):
            design = scenario_map.get(q["query_id"], {}).get("normalized_design")
            qc, qf = condition_values(q, variant, design), fields_for(q, variant)
            for j, c in enumerate(candidates):
                features[v, i, j] = pair_feature_row(q, c, variant, design, candidate_conditions[j], qc, qf, candidate_fields[j])
    errors, requested, same_parent = _error_labels(report, queries, candidates)
    result = {
        "schema_version": SCHEMA, "query_records": [_safe_record(q) for q in queries],
        "candidate_records": [_safe_record(c, True) for c in candidates], "variant_names": list(VARIANTS),
        "query_texts": [[text_for(q, v) for q in queries] for v in VARIANTS],
        "document_texts": [c["text"] for c in candidates],
        "no_numeric_query_texts": [[without_conditions(text_for(q, v)) for q in queries] for v in VARIANTS],
        "no_numeric_document_texts": [without_conditions(c["text"]) for c in candidates],
        "pair_features": features[0], "pair_features_by_variant": features,
        "pair_feature_names": feature_names(),
        "numeric_feature_indices": list(range(FIELDS.index("formulations") * 5, (FIELDS.index("formulations") + 1) * 5)) + list(range(5 * len(FIELDS), len(feature_names()))),
        "grades": grades, "applicability": apps,
        "core_indices": core_indices, "execution_indices": execution_indices,
        "stability_indices": [index[q["query_id"]] for q in stability], "error_records": errors,
        "groups": groups, "group_keys": key_rows, "folds": folds,
        "grouping_audit": {"empty_hashes_used": any("evidence:" + hashlib.sha256(b"").hexdigest() in keys for keys in key_rows),
                           "atomic_property_hashes_namespaced": atomic_rows,
                           "reason": "Typed chemical-property scalar equality is not shared research identity; exact narrative hashes and shared article records remain global."},
        "execution_scenarios": [_scenario_metadata(scenario_map[q["query_id"]]) for q in clinical],
        "execution_requested_endpoints": requested, "same_parent_candidate_indices": same_parent,
        "input_policy": {"features_use_outcomes": False, "split_assignment_uses_labels": False,
                         "candidate_family_collapsed": False, "sealed_files_opened": 0,
                         "applicability_codes": {"Y": 0, "N": 1, "U": 2, "invalid_or_unlabeled": -1},
                         "clinical_missing_variants": "identity transform: no supplied perturbation is invented",
                         "numeric_ablation": "mask condition text and zero every numeric_feature_indices column in each variant",
                         "grouping_limit": "article/record coordinates within acquired files plus exact evidence text; unknown cross-file publication aliases are not inferred",
                         "clinical_labels": "completed main reference diagnostics only; population equivalence not asserted; supplementary outcomes excluded"},
        "group_summary": {"independent_groups": len(set(groups)), "group_sizes_descending": sorted(Counter(groups).values(), reverse=True),
                          "outer_test_sizes": [len(f["outer_test"]) for f in folds],
                          "fold_counts": [{"outer_test": {"queries": len(f["outer_test"]), "core": len(set(f["outer_test"]) & set(core_indices)),
                                                           "execution": len(set(f["outer_test"]) & set(execution_indices)), "groups": len({groups[i] for i in f["outer_test"]})},
                                           "outer_train": {"queries": len(f["outer_train"]), "core": len(set(f["outer_train"]) & set(core_indices)),
                                                            "execution": len(set(f["outer_train"]) & set(execution_indices)), "groups": len({groups[i] for i in f["outer_train"]})},
                                           "inner_core_counts": [{"train": len(set(inner["train"]) & set(core_indices)),
                                                                  "validation": len(set(inner["validation"]) & set(core_indices))} for inner in f["inner"]]} for f in folds]},
    }
    validate_dataset(result)
    return result


def validate_dataset(data):
    q, c, v, f = len(data["query_records"]), len(data["candidate_records"]), len(data["variant_names"]), len(data["pair_feature_names"])
    for name, shape in (("pair_features", (q, c, f)), ("pair_features_by_variant", (v, q, c, f)), ("grades", (q, c)), ("applicability", (q, c))):
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError("Invalid or nonfinite dataset array: " + name)
    if not np.array_equal(data["pair_features"], data["pair_features_by_variant"][0]):
        raise ValueError("Original features differ from variant zero")
    if not set(np.unique(data["grades"])).issubset({-1, 0, 1, 2, 3}) or not set(np.unique(data["applicability"])).issubset({-1, 0, 1, 2}):
        raise ValueError("Invalid label codes")
    tests = []
    for fold in data["folds"]:
        train, test = set(fold["outer_train"]), set(fold["outer_test"])
        if train & test or train | test != set(range(q)):
            raise ValueError("Outer fold index partition invalid")
        tests += fold["outer_test"]
        pairs = [(train, test)]
        for inner in fold["inner"]:
            a, b = set(inner["train"]), set(inner["validation"])
            if a & b or a | b != train:
                raise ValueError("Inner fold index partition invalid")
            pairs.append((a, b))
        for a, b in pairs:
            if {data["groups"][i] for i in a} & {data["groups"][i] for i in b}:
                raise ValueError("Input-linked group leakage across folds")
    if sorted(tests) != list(range(q)):
        raise ValueError("Outer test folds do not cover each request exactly once")


def _write_exclusive(path, raw):
    import os
    with scoped(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def build_dataset(output: Path = DEFAULT_OUTPUT) -> dict:
    output = scoped(output)
    if (output / "manifest.json").exists():
        return load_dataset(output)
    if any((output / name).exists() for name in ("dataset.json", "arrays.npz")):
        raise ValueError("Partial dataset exists; refusing to overwrite")
    sources = []
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError("Frozen source changed: " + str(path))
        sources.append({"path": str(path), "sha256": expected})
    gate = read_json(CORE / "assessment_repaired_v2/automatic_evaluation.json")
    accepted = gate.get("completion_acceptance", {})
    if gate.get("state") != "complete" or accepted.get("state") != "passed" or not all(accepted.get("checks", {}).values()):
        raise ValueError("Document assessment is not completed and accepted")
    if gate["assessment"]["committed_judgments_sha256"] != PINNED[CORE / "assessment_repaired_v2/judgments.jsonl"]:
        raise ValueError("Document completion does not bind the final labels")
    completion = read_json(FOLLOWUP / "completion.json")
    target = {"path": str(FOLLOWUP / "primary/results.json"), "sha256": PINNED[FOLLOWUP / "primary/results.json"]}
    if completion.get("completed") is not True or completion.get("original_artifacts_rechecked_unchanged") is not True or target not in next(x["files"] for x in completion["outputs"] if x["analysis"] == "primary"):
        raise ValueError("Clinical primary results have no completed bound output")
    stability, core, candidates = read_rows(FRONTIER / "inputs/queries.jsonl"), read_rows(CORE / "inputs/queries.jsonl"), read_rows(FRONTIER / "inputs/candidates.jsonl")
    clinical, scenarios = read_rows(EXECUTION / "rankings/inputs/queries.jsonl"), read_rows(EXECUTION / "inputs/scenarios.jsonl")
    if (len(stability), len(core), len(clinical), len(candidates)) != (772, 145, 41, 280) or candidates != read_rows(CORE / "inputs/candidates.jsonl"):
        raise ValueError("Frozen corpus count or catalogue-order mismatch")
    result = assemble(stability, core, clinical, candidates, scenarios,
                      read_rows(CORE / "assessment_repaired_v2/judgments.jsonl"), read_json(FOLLOWUP / "primary/results.json"),
                      PINNED[CORE / "assessment_repaired_v2/judge_protocol.json"])
    if len(result["error_records"]) != 171:
        raise ValueError("Primary scalar label count changed")
    # Conditions previously invisible to the categorical-only model must differ.
    ids = {q["query_id"]: i for i, q in enumerate(result["query_records"])}
    pairs = [("caffeine_sherman2022_solution_250mg", "caffeine_sherman2022_solution_50mg"),
             ("metformin_pmc12394267_reference_fasting", "metformin_pmc12394267_reference_fed"),
             ("voriconazole_li2023_reference_iv_4mgkg", "voriconazole_li2023_reference_iv_6mgkg")]
    for a, b in pairs:
        if np.array_equal(result["pair_features"][ids["clinical:" + a]], result["pair_features"][ids["clinical:" + b]]):
            raise ValueError("Distinct frozen clinical conditions collapsed: " + a)
    # Check again after assembly: no mutable input is silently frozen mid-change.
    for ref in sources:
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Source changed during dataset assembly")
    output.mkdir(parents=True, exist_ok=True)
    arrays = io.BytesIO()
    np.savez_compressed(arrays, **{name: result[name] for name in ARRAY_NAMES})
    _write_exclusive(output / "arrays.npz", arrays.getvalue())
    _write_exclusive(output / "dataset.json", json_bytes({k: v for k, v in result.items() if k not in ARRAY_NAMES}))
    producer = {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": digest(__file__)}
    manifest = {"schema_version": SCHEMA, "completed": True, "inputs": sources, "producer": producer,
                "outputs": [{"path": str((output / n).relative_to(ROOT)), "sha256": digest(output / n)} for n in ("dataset.json", "arrays.npz")],
                "counts": {"queries": len(result["query_records"]), "stability": 772, "core": 145, "clinical": 41, "candidates": 280, "document_pairs": 40600, "scalar_labels": 171},
                "group_summary": result["group_summary"], "grouping_audit": result["grouping_audit"],
                "scientific_results_are_labels_not_inputs": True}
    preliminary = output / "preliminary_atomic_hash_audit_20261004"
    if preliminary.is_dir():
        manifest["superseded_input_only_grouping_attempt"] = {
            "reason": "Before model fitting: namespaced structured atomic property evidence to avoid false cross-drug joins on literal 0/3.",
            "files": [{"path": str((preliminary / name).relative_to(ROOT)), "sha256": digest(preliminary / name)}
                      for name in ("data.py", "dataset.json", "arrays.npz", "manifest.json")]}
    _write_exclusive(output / "manifest.json", json_bytes(manifest))
    result["manifest"] = manifest
    return result


def load_dataset(output: Path = DEFAULT_OUTPUT) -> dict:
    output = scoped(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema_version") != SCHEMA or manifest.get("completed") is not True:
        raise ValueError("Dataset is not complete")
    if manifest.get("producer", {}).get("sha256") != digest(__file__):
        raise ValueError("Dataset producer changed; use a new output directory")
    expected_outputs = {str((output / n).relative_to(ROOT)) for n in ("dataset.json", "arrays.npz")}
    if {r["path"] for r in manifest["outputs"]} != expected_outputs:
        raise ValueError("Dataset output identity changed")
    if {r["path"]: r["sha256"] for r in manifest["inputs"]} != {str(p): h for p, h in PINNED.items()}:
        raise ValueError("Dataset input registration changed")
    for ref in [*manifest["inputs"], *manifest["outputs"], manifest["producer"],
                *manifest.get("superseded_input_only_grouping_attempt", {}).get("files", [])]:
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Frozen dataset binding changed: " + ref["path"])
    result = read_json(output / "dataset.json")
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != set(ARRAY_NAMES):
            raise ValueError("Unexpected dataset arrays")
        result.update({key: arrays[key] for key in ARRAY_NAMES})
    for key in ("execution_requested_endpoints", "same_parent_candidate_indices"):
        result[key] = {int(k): v for k, v in result[key].items()}
    result["manifest"] = manifest
    validate_dataset(result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = load_dataset(args.output) if args.check else build_dataset(args.output)
    print(json.dumps({"counts": data["manifest"]["counts"], "groups": data["group_summary"], "features": len(data["pair_feature_names"])}))
