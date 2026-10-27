"""Prospective evidence-first contract validation and full-source diagnostics.

Synthetic labels concern explicit fictional statements only. Past failures and
representative real documents never acquire fabricated gold labels. Historical
outputs are read only to identify diagnostic inputs/exposure, not to assign gold
or choose representative inputs. The frozen grading contract is unchanged.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import frontier_judge_validation as v1
import frontier_judge_validation_v2 as v2
import frontier_judge_prompt_v3 as prompt
import run_frontier_assessment as assessment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/model_recommendation/frontier_comparison_20260914"
PARENT = OUT / "assessment_qwen36_v2/validation"
PRODUCTION = OUT / "assessment_validated_v2"
PROMPT_SHA = "a759e4c3cf43c98a2311e27b0cd51600b10338c1144fc9e4c08171fef25276f9"
VERSION = "pbpk-evidence-first-validation-v3"
SEED = "pbpk-full-packet-coverage-v3-20260919"
ROLES = {"development_v1_regression": 64, "development_v2_regression": 64,
         "development_real_source": 60, "independent_admission": 32,
         "production_failure_diagnostic": 19, "prospective_source_coverage": 494}
A, B = "control-compound-aurora", "control-compound-boreal"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _q(text, task, **features):
    return {"query_id": "", "text": text, "task_type": task, "compounds": [A], "features": features}


def _scenarios():
    plasma, ddi, special, mechanism, parameter = v1.TASKS
    # New cases use contrasts/attribution, not alternative names for old cases.
    # Each entry: name, query, numbered paragraphs, grade, applicability,
    # alternative sufficient citation groups (zero-based), optional registry.
    return [
        ("background_candidate_is_not_current_candidate",
         _q(f"Predict the plasma concentration-time profile of {A} using the current candidate.", plasma),
         [f"Background comparison: a different published candidate, not the current candidate, predicts plasma concentrations of {B}.",
          f"Current candidate: its {A} PBPK equations explicitly support the requested plasma concentration-time prediction."], 3, True, [[1]], [A]),
        ("supported_endpoint_does_not_supply_other_required_endpoint",
         _q(f"Predict urinary excretion and tissue concentrations of {A} using this candidate.", plasma,
            endpoints=["urinary_excretion", "tissue_concentration"]),
         [f"For {A}, this candidate documents urinary excretion prediction.",
          f"Tissue-concentration prediction for {A} is not described or evaluated; neither support nor exclusion of this second requested endpoint is established."], None, None, [], [A]),
        ("future_plan_is_not_present_support",
         _q(f"Predict concentration-time profiles of {A} in patients with hepatic impairment using the current candidate.", special, populations=["hepatic_impairment"]),
         [f"A development plan proposes adding hepatic-impairment physiology to the {A} candidate next year.",
          "The proposed extension has not been implemented or evaluated. The current source establishes neither support nor exclusion for the proposed hepatic-impairment prediction."], None, None, [], [A]),
        ("quoted_exclusion_is_explicitly_retracted",
         _q(f"Predict oral plasma concentration-time profiles of {A} in adult humans.", plasma,
            routes=["oral"], populations=["adult"], species=["human"]),
         [f"Erratum: the earlier statement 'this {A} candidate excludes oral dosing in adult humans' was incorrect and is retracted.",
          f"The corrected specification explicitly supports oral plasma concentration-time prediction of {A} in adult humans."], 3, True, [[1]], [A]),
        ("two_requested_routes_one_explicitly_excluded",
         _q(f"Predict both oral and intravenous plasma concentration-time profiles of {A} in humans.", plasma,
            routes=["oral", "intravenous"], species=["human"]),
         [f"The {A} candidate supports oral plasma concentration-time prediction in humans.",
          f"Intravenous administration of {A} is explicitly excluded from this candidate's equations and applicable use. The candidate must not be applied to the requested intravenous prediction."], 0, False, [[1]], [A]),
        ("cross_compound_mechanism_absent_is_not_conflict",
         _q(f"For {A}, find a mechanistic reference for OATP-mediated hepatic uptake.", mechanism, transporters=["OATP"]),
         [f"This {B} model document reports physicochemical properties, including pKa and solubility.",
          "Transporter mechanisms, including OATP-mediated uptake, are not documented. No shared or conflicting uptake mechanism is established by this source."], None, None, [], [B]),
        ("nearby_transporter_name_is_explicitly_different",
         _q(f"Find a mechanistic template for OCT2-mediated active secretion of {A}.", mechanism, transporters=["OCT2"]),
         [f"The {A} candidate provides an active-secretion template implemented for MATE1 only.",
          f"The {A} candidate's specification explicitly excludes OCT2-mediated secretion; its MATE1 equations are stated to be incompatible with the requested OCT2 template."], 0, False, [[1]], [A]),
        ("previous_version_population_not_current_version",
         _q(f"Predict plasma concentration-time profiles of {A} in children with the current model version.", special, populations=["pediatric"]),
         [f"An older {A} model version was evaluated for concentration-time prediction in children.",
          f"The current version replaced that version's physiology equations. Pediatric applicability of the current {A} version has not been documented or evaluated; the source does not establish exclusion either."], None, None, [], [A]),
        ("current_version_support_overrides_historical_exclusion",
         _q(f"Predict plasma concentration-time profiles of {A} during pregnancy using the current candidate version.", special, populations=["pregnancy"]),
         [f"The obsolete {A} version excluded pregnancy prediction.",
          f"The current {A} candidate version has a documented pregnancy physiology module and explicitly supports the requested plasma concentration-time prediction during pregnancy. The obsolete version's exclusion does not describe this version."], 3, True, [[1]], [A]),
        ("isolated_parameter_not_complete_requested_workflow",
         _q(f"Provide a documented compound-parameterization workflow for {A} covering both pKa and renal clearance estimation.", parameter),
         [f"A data entry lists pKa for {A}; this is an isolated parameter entry, not a reusable complete workflow.",
          f"The source does not document a renal-clearance estimation workflow for {A}, nor establish that such a workflow is excluded."], None, None, [], [A]),
        ("mechanistic_module_not_whole_curve_prediction",
         _q(f"Directly predict complete plasma concentration-time profiles of {A} after oral dosing.", plasma, routes=["oral"]),
         [f"For {A}, this candidate documents an oral intestinal-permeability mechanism as a reusable reference module.",
          f"The reference module is insufficient to establish complete plasma concentration-time prediction of {A}; that main use is neither established nor explicitly excluded."], 1, None, [[0]], [A]),
        ("partial_renal_scope_requires_documented_adaptation",
         _q(f"Predict plasma concentration-time profiles of {A} in humans with renal impairment.", special, populations=["renal_impairment"], species=["human"]),
         [f"For humans with renal impairment, the {A} candidate explicitly provides partial support for plasma concentration-time prediction with a documented renal-transporter adaptation.",
          f"The renal-transporter adaptation for this {A} use is still incomplete; applying the candidate to the requested prediction is not yet established and is not explicitly excluded."], 2, None, [[0]], [A]),
        ("ddi_roles_require_current_direction",
         _q(f"Predict the AUC interaction change for {A} as the victim compound when {B} inhibits its metabolism.", ddi, endpoints=["auc"]),
         [f"Background experiment: {B} was the victim and {A} the inhibitor, the reverse of the requested roles.",
          f"The current registered model additionally and explicitly supports the requested direction: {A} is the victim of inhibition by {B}, with prediction of the resulting drug-drug interaction AUC change."], 3, True, [[1]], [A, B]),
        ("ddi_target_present_but_requested_role_excluded",
         _q(f"Predict the AUC drug-drug interaction change of {A} when it is the victim of inhibition.", ddi, endpoints=["auc"]),
         [f"The artifact includes {A} and {B}; compound membership alone does not establish their interaction roles.",
          f"This artifact implements {A} only as an inhibitor/perpetrator and explicitly excludes using {A} as the victim in AUC interaction prediction."], 0, False, [[1]], [A, B]),
        ("specific_dose_undocumented_not_excluded",
         _q(f"Predict oral AUC of {A} after a 200 mg dose in adult humans.", plasma,
            routes=["oral"], populations=["adult"], species=["human"], endpoints=["auc"]),
         [f"The source documents oral AUC prediction of {A} at 10 mg in adult humans.",
          f"Application of this {A} model at 200 mg has not been documented; the supplied source neither establishes support nor excludes that dose. Dose extrapolation cannot be inferred here."], None, None, [], [A]),
        ("unrequested_metabolite_gap_not_parent_scope_gap",
         _q(f"Predict plasma AUC and Cmax of parent {A}; no metabolite prediction is requested.", plasma, endpoints=["auc", "cmax"]),
         [f"The candidate explicitly supports plasma concentration-time prediction and prediction of both AUC and Cmax for parent {A}.",
          f"A separate inactive metabolite has not been studied. This missing metabolite information does not concern the documented parent-{A} AUC or Cmax predictions."], 3, True, [[0]], [A]),
    ]


def _new_controls():
    cases = []
    distractor = (None, "Repository maintenance note: an available excerpt number is not by itself evidence for a model-use decision.")
    for name, query, texts, grade, applicable, groups, compounds in _scenarios():
        for variant in ("base", "citation_permutation"):
            case_id = f"admission-v3:{name}:{variant}"
            q = deepcopy(query); q["query_id"] = case_id
            items = list(enumerate(texts))
            items = [*items, distractor] if variant == "base" else [distractor, *reversed(items)]
            body = {"documented_model_name": "Controlled fictional PBPK candidate for evidence selection",
                    "model_compounds": compounds,
                    "scope_warning": "Artificial source statements for a controlled reasoning contract, not clinical advice or expert scientific gold.",
                    "excerpts": [{"n": i, "text": text, "kind": "synthetic_contract_control", "version_specific_validation_verified": False}
                                 for i, (_, text) in enumerate(items, 1)]}
            refs = [{"source_id": f"{case_id}:{i}", "quote": text, "excerpt_sha256": v1._sha(text),
                     "document_sha256": v1._sha(text), "document_path": f"synthetic-controls-v3/{name}",
                     "location": {"char_start": 0, "char_end": len(text)}, "synthetic": True}
                    for i, (_, text) in enumerate(items, 1)]
            expected = {"grades": [grade], "documented_applicability": applicable,
                        "supporting_excerpt_numbers": [i for i, (origin, _) in enumerate(items, 1) if origin is not None],
                        "required_support_any_of": [[i for i, (origin, _) in enumerate(items, 1) if origin in group] for group in groups],
                        "source": "New explicit fictional contrast/attribution statements authored after v3 prompt freeze and before v3 judge outputs; not expert scientific gold."}
            cases.append({"case_id": case_id, "kind": "synthetic_control", "query": q,
                          "candidate_id": "synthetic:v3-candidate", "packet": {"body": body, "refs": refs,
                          "packet_sha256": v1._sha(v1._canonical(body)), "source_candidate_id": "synthetic:v3-candidate",
                          "characters": sum(len(text) for _, text in items)}, "expected": expected, "scientific_gold": False,
                          "strata": {"scenario": name, "variant": variant, "validation_role": "independent_admission",
                                     "task_type": q["task_type"], "initial_input_seen_before_v3": False}})
    return cases


def _real_case(role, query, packet, extra):
    key = v1._sha(packet["packet_sha256"] + "\n" + query["text"])
    return {"case_id": f"{role}:{key[:24]}", "kind": "real_source_pair",
            "query": {k: deepcopy(query[k]) for k in ("query_id", "text", "task_type", "compounds", "features")},
            "candidate_id": packet["source_candidate_id"], "packet": deepcopy(packet),
            "expected": None, "scientific_gold": False,
            "strata": {"validation_role": role, "task_type": query["task_type"], **extra}}


def _packet_map(packets):
    result = {}
    for cid in sorted(packets):
        p = packets[cid]
        if p["refs"]:
            result.setdefault(p["packet_sha256"], p)
    if len(result) != 247:
        raise ValueError("Expected all 247 nonempty packet classes")
    return result


def _history(queries):
    """Only input identities/exposure and failure status influence diagnostics."""
    qmap = {q["query_id"]: q for q in queries}
    seen, hashes = set(), {}
    for name in ("assessment", "assessment_validated_v2"):
        folder = OUT / name
        files = [folder / "primary_generation_journal.jsonl"]
        if name == "assessment":
            files += [folder / f"preflight_v{i}/primary_generation_journal.jsonl" for i in range(1, 6)]
        for path in files:
            if not path.exists():
                continue
            hashes[str(path.relative_to(OUT))] = _sha(path)
            for row in _rows(path):
                for qid in row["query_ids"]:
                    if qid in qmap:
                        seen.add((row["packet_sha256"], qmap[qid]["text"]))
    for name in ("assessment_qwen36", "assessment_qwen36_v2"):
        folder = OUT / name / "validation"
        if not (folder / "cases.jsonl").exists():
            continue
        cases = {c["case_id"]: c for c in _rows(folder / "cases.jsonl")}
        hashes[str((folder / "cases.jsonl").relative_to(OUT))] = _sha(folder / "cases.jsonl")
        for model in ("phi", "qwen"):
            path = folder / f"{model}_records.jsonl"
            if path.exists():
                hashes[str(path.relative_to(OUT))] = _sha(path)
                for row in _rows(path):
                    case = cases[row["case_id"]]
                    seen.add((case["packet"]["packet_sha256"], case["query"]["text"]))
    return seen, hashes


def _diagnostics(queries, packets, seen):
    cache_path = PRODUCTION / "primary_judgment_cache.jsonl"
    journal_path = PRODUCTION / "primary_generation_journal.jsonl"
    invalid = {row["generation_id"] for row in _rows(cache_path) if row["status"] == "invalid_output"}
    qmap = {q["query_id"]: q for q in queries}; pmap = _packet_map(packets)
    cases = []
    for line, row in enumerate(_rows(journal_path), 1):
        if row["generation_id"] not in invalid:
            continue
        if len(row["query_ids"]) != 1:
            raise ValueError("Expected one request in diagnostic production group")
        q, p = qmap[row["query_ids"][0]], pmap[row["packet_sha256"]]
        cases.append(_real_case("production_failure_diagnostic", q, p,
                     {"previous_status": "invalid_output", "original_generation_id": row["generation_id"],
                      "original_journal_line": line, "initial_input_seen_before_v3": True}))
    if len(cases) != 19 or len({c["case_id"] for c in cases}) != 19:
        raise ValueError("Expected exactly 19 unique preserved production failures")
    return cases, {str(cache_path.relative_to(OUT)): _sha(cache_path), str(journal_path.relative_to(OUT)): _sha(journal_path)}


def _representatives(queries, packets, seen):
    cases, counts = [], Counter()
    for packet_key, p in sorted(_packet_map(packets).items()):
        registry = {name.casefold() for name in p["body"]["model_compounds"]}
        same = [q for q in queries if registry.intersection(name.casefold() for name in q["compounds"])]
        cross = [q for q in queries if not registry.intersection(name.casefold() for name in q["compounds"])]
        chosen = set()
        for slot in range(2):
            preferred = same if slot == 0 else cross
            pool = [q for q in (preferred or queries) if q["query_id"] not in chosen]
            q = min(pool, key=lambda q: (counts[q["task_type"]], v1._sha(SEED + packet_key + str(slot) + q["query_id"])))
            chosen.add(q["query_id"]); counts[q["task_type"]] += 1
            overlap = bool(registry.intersection(name.casefold() for name in q["compounds"]))
            cases.append(_real_case("prospective_source_coverage", q, p,
                         {"packet_class_sha256": packet_key, "selection_slot": slot,
                          "compound_overlap": "same_compound" if overlap else "cross_compound",
                          "same_compound_request_available": bool(same), "cross_compound_request_available": bool(cross),
                          "initial_input_seen_before_v3": (packet_key, q["text"]) in seen,
                          "material_characters": p["characters"]}))
    return cases


def _read_parent(queries, candidates, packets):
    manifest = v2.verify_validation_v2(queries, candidates, packets, PARENT)
    return manifest, _rows(PARENT / "cases.jsonl")


def _validate_cases(cases, parent_cases, packets):
    if len(cases) != 733 or len({c["case_id"] for c in cases}) != 733:
        raise ValueError("Expected 733 unique role-specific cases")
    if dict(Counter(c["strata"].get("validation_role") for c in cases)) != ROLES:
        raise ValueError("v3 role denominator mismatch")
    parent = {c["case_id"]: c for c in parent_cases}
    coverage = []
    for c in cases:
        v1._check_packet(c["packet"])
        role = c["strata"]["validation_role"]
        if role.startswith("development_"):
            stripped = deepcopy(c)
            old_role = stripped["strata"].pop("prior_validation_role")
            stripped["strata"]["validation_role"] = old_role
            if stripped != parent.get(c["case_id"]):
                raise ValueError("Inherited v2 case or gold changed")
        elif role == "independent_admission":
            expected = c["expected"]
            valid = set(range(1, len(c["packet"]["refs"]) + 1))
            allowed = set(expected["supporting_excerpt_numbers"])
            if not allowed <= valid or any(not g or not set(g) <= allowed for g in expected["required_support_any_of"]):
                raise ValueError("Invalid controlled source support specification")
            if expected["grades"] != [None] or expected["documented_applicability"] is not None:
                if not expected["required_support_any_of"] or len(expected["required_support_any_of"]) > 3:
                    raise ValueError("Known control label needs achievable source support")
            if c.get("scientific_gold") is not False or c["case_id"] in parent:
                raise ValueError("New control provenance invalid")
        else:
            if c["expected"] is not None or c["packet"] != packets[c["candidate_id"]]:
                raise ValueError("Real case has gold or altered source")
            if role == "prospective_source_coverage": coverage.append(c)
    counts = Counter(c["packet"]["packet_sha256"] for c in coverage)
    if set(counts) != set(_packet_map(packets)) or set(counts.values()) != {2}:
        raise ValueError("Full packet coverage needs two requests for each nonempty class")
    new = [c for c in cases if c["strata"]["validation_role"] == "independent_admission"]
    if len({c["strata"]["scenario"] for c in new}) != 16 or {c["query"]["task_type"] for c in new} != set(v1.TASKS):
        raise ValueError("New scenario/task coverage incomplete")
    if {c["expected"]["grades"][0] for c in new} != {None, 0, 1, 2, 3}:
        raise ValueError("New grade coverage incomplete")


def _write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT.resolve()): raise ValueError("Output outside project")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value: raise ValueError("Frozen v3 file differs")
        return
    with path.open("xb") as handle:
        handle.write(value); handle.flush(); os.fsync(handle.fileno())


def build_validation_v3(queries, candidates, packets, out):
    out = Path(out)
    if (out / "manifest.json").exists(): return verify_validation_v3(queries, candidates, packets, out)
    if out.exists() and any(out.glob("*_records.jsonl")): raise ValueError("Cannot author after v3 outputs exist")
    if _sha(prompt.__file__) != PROMPT_SHA: raise ValueError("Frozen v3 prompt changed")
    parent_manifest, parent_cases = _read_parent(queries, candidates, packets)
    mapping = {"development_regression": "development_v1_regression", "independent_admission": "development_v2_regression", "unlabelled_real_source": "development_real_source"}
    inherited = deepcopy(parent_cases)
    for c in inherited:
        role = c["strata"]["validation_role"]
        c["strata"].update(prior_validation_role=role, validation_role=mapping[role])
    seen, history = _history(queries)
    diagnostic, diagnostic_hashes = _diagnostics(queries, packets, seen)
    history.update(diagnostic_hashes)
    representative = _representatives(queries, packets, seen)
    cases = inherited + _new_controls() + diagnostic + representative
    _validate_cases(cases, parent_cases, packets)
    data = ("\n".join(v1._canonical(c) for c in cases) + "\n").encode()
    manifest = {"version": VERSION, "frozen_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases), "role_counts": ROLES, "cases_sha256": v1._sha(data),
        "implementation_sha256": _sha(__file__), "evaluator_v1_sha256": _sha(v1.__file__),
        "v3_prompt_file_sha256": _sha(prompt.__file__), "v3_system_prompt_sha256": v1._sha(prompt.SYSTEM),
        "parent_v2_cases_sha256": parent_manifest["cases_sha256"], "parent_v2_manifest_sha256": _sha(PARENT / "manifest.json"),
        "input_sha256": v2._hash_inputs(queries, candidates, packets), "historical_input_artifact_sha256": history,
        "new_control_scenarios": 16, "new_control_variants_per_scenario": 2,
        "new_control_grade_counts": dict(Counter(str(c["expected"]["grades"][0]) for c in cases if c["strata"]["validation_role"] == "independent_admission")),
        "representative_coverage": {"nonempty_packet_classes": 247, "queries_per_packet": 2,
          "task_counts": dict(Counter(c["query"]["task_type"] for c in representative)),
          "overlap_counts": dict(Counter(c["strata"]["compound_overlap"] for c in representative)),
          "previously_seen_input_count": sum(c["strata"]["initial_input_seen_before_v3"] for c in representative),
          "selection_seed": SEED,
          "selection_rule": "Every nonempty packet sorted by SHA; slot0 prefers same-compound query when one exists, slot1 prefers cross-compound; balance available task counts and fixed SHA tie-break. Two distinct queries. Historical outputs/exposure do not affect this selection."},
        "unique_supplied_prompt_counts_by_role": {role: len({(c["packet"]["packet_sha256"], c["query"]["text"]) for c in cases if c["strata"]["validation_role"] == role}) for role in ROLES},
        "diagnostic_policy": "19 actual previous invalid outputs select observed diagnostic inputs only; no desired scientific answer is assigned or inferred from those failures.",
        "raw_vs_final_policy": "Control admission uses unrepaired raw outputs. Non-control final engineering checks replay the existing parser/scope guard on final_output after at most one real repair; final_status alone is never trusted.",
        "representative_window_policy": "Use frozen prospective_source_coverage case order for every full consecutive 128-case window; invalid final outputs remain in the denominator. No reordering to avoid a window failure.",
        "science_limit": "32 fresh controlled variants represent 16 authored fictional scenarios, not expert scientific cases. All real source pairs have no gold; format, known-label citation presence and lexical scope validation do not prove semantic citation support or PBPK accuracy.",
        "blinding": "Only the packet and query enter model inputs; role, exposure, previous failure, expected values and other judge answers are withheld.",
        "freeze_policy": "Preserve original cases/gold and historical failures. Do not revise this set, grade definitions or thresholds after seeing v3 outputs."}
    _write(out / "cases.jsonl", data)
    _write(out / "manifest.json", (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
    return verify_validation_v3(queries, candidates, packets, out)


def verify_validation_v3(queries, candidates, packets, out):
    out = Path(out)
    if not out.resolve().is_relative_to(ROOT.resolve()): raise ValueError("Input outside project")
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["version"] != VERSION or manifest["cases_sha256"] != _sha(out / "cases.jsonl"):
        raise ValueError("Frozen v3 case hash/version mismatch")
    if manifest["implementation_sha256"] != _sha(__file__) or manifest["evaluator_v1_sha256"] != _sha(v1.__file__):
        raise ValueError("Frozen validator implementation changed")
    if _sha(prompt.__file__) != PROMPT_SHA or manifest["v3_prompt_file_sha256"] != PROMPT_SHA or manifest["v3_system_prompt_sha256"] != v1._sha(prompt.SYSTEM):
        raise ValueError("Frozen v3 prompt changed")
    parent_manifest, parent_cases = _read_parent(queries, candidates, packets)
    if manifest["parent_v2_cases_sha256"] != parent_manifest["cases_sha256"] or manifest["parent_v2_manifest_sha256"] != _sha(PARENT / "manifest.json"):
        raise ValueError("Parent validation changed")
    if manifest["input_sha256"] != v2._hash_inputs(queries, candidates, packets): raise ValueError("Full source inputs changed")
    if manifest["case_count"] != 733 or manifest["role_counts"] != ROLES: raise ValueError("Frozen role denominator mismatch")
    for name, expected in manifest["historical_input_artifact_sha256"].items():
        path = OUT / name
        if not path.resolve().is_relative_to(ROOT.resolve()) or _sha(path) != expected: raise ValueError("Historical source/exposure artifact changed")
    _validate_cases(_rows(out / "cases.jsonl"), parent_cases, packets)
    return manifest


def _final_record(record):
    value = record.get("final_output")
    if value is None:
        value = record.get("attempts", [{}])[-1].get("output") if record.get("attempts") else None
    return {**record, "output": value if value is not None else record.get("output")}


def _replay(records, cases):
    by_id = {r["case_id"]: r for r in records}
    rows = []
    for c in cases:
        r = by_id.get(c["case_id"])
        row = {"case_id": c["case_id"], "observed": r is not None, "final_valid": False,
               "error": "missing_record" if r is None else None, "scope_guard_applied": False}
        if r is not None:
            try:
                answer = assessment.parse_answers(_final_record(r)["output"], [1], c["packet"]["refs"])[1]
                result = assessment.validate_explicit_scope(answer, c["query"], c["packet"]["refs"], c["packet"]["body"]["model_compounds"])
                row.update(final_valid=True, scope_guard_applied=result["applied"])
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                row["error"] = type(error).__name__ + ": " + str(error)
            if r.get("final_status") in {"valid", "invalid_output"}:
                row["reported_status_agrees"] = (r["final_status"] == "valid") == row["final_valid"]
        rows.append(row)
    return {"expected": len(cases), "observed": sum(r["observed"] for r in rows),
            "complete": len(by_id) == len(cases), "final_valid_count": sum(r["final_valid"] for r in rows),
            "final_valid_fraction": sum(r["final_valid"] for r in rows) / len(cases) if cases else None,
            "rows": rows, "semantic_entailment_verified": False}


def evaluate_validation_v3(records, cases):
    if isinstance(cases, (str, Path)): cases = _rows(cases)
    ids = {c["case_id"] for c in cases}
    if len(ids) != len(cases) or len({r["case_id"] for r in records}) != len(records) or any(r["case_id"] not in ids for r in records):
        raise ValueError("Unknown or duplicate v3 case/output")
    by_role, admission, final, final_admission, replay = {}, {}, {}, {}, {}
    for role, count in ROLES.items():
        subset = [c for c in cases if c["strata"].get("validation_role") == role]
        if len(subset) != count: raise ValueError("v3 evaluation role denominator mismatch")
        keep = {c["case_id"] for c in subset}; rows = [r for r in records if r["case_id"] in keep]
        by_role[role] = v1.evaluate_validation(rows, subset)
        admission[role] = v1.admission_metrics(by_role[role])
        final[role] = v1.evaluate_validation([_final_record(r) for r in rows], subset)
        final_admission[role] = v1.admission_metrics(final[role])
        replay[role] = _replay(rows, subset)
    coverage = replay["prospective_source_coverage"]
    ordered = coverage["rows"]
    windows = [sum(not r["final_valid"] for r in ordered[start:start + 128]) for start in range(max(0, len(ordered) - 127))]
    return {"version": VERSION, "complete": len(records) == len(cases), "expected_cases": len(cases), "observed_cases": len(records),
            "by_role": by_role, "admission_by_role": admission, "final_by_role": final,
            "final_admission_by_role": final_admission, "scope_guard_replay_by_role": replay,
            "final_validation_by_role": replay,
            "coverage_sequence": {"complete": coverage["complete"], "case_ids": [r["case_id"] for r in ordered],
               "final_valid_fraction": coverage["final_valid_fraction"], "window_size": 128,
               "full_window_count": len(windows), "max_invalid_in_full_window": max(windows) if windows else None,
               "any_full_window_above10percent": any(count / 128 > .10 for count in windows)},
            "scientific_gold_accuracy": None}
