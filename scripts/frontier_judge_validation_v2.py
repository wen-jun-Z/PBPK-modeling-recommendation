"""Prospectively frozen v2 controls; v1 cases remain development regressions.

The 16 newly authored scenarios test explicit artificial evidence contracts, not
expert PBPK findings. Four transformations per scenario are correlated repeats.
Real source pairs remain unlabelled. This module never reads judge outputs or
recommender scores when building or verifying the corpus.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import frontier_judge_validation as v1
import frontier_judge_prompt_v2 as prompt

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "experiments/model_recommendation/frontier_comparison_20260914/assessment_qwen36/validation"
PROMPT_FILE_SHA256 = "3f97e486ab455266f6632cf7faa70b23c82994840f792f7c8fa98a598e854997"
VERSION = "pbpk-judge-independent-contract-validation-v2"
ROLES = {"development_regression": 64, "independent_admission": 64, "unlabelled_real_source": 60}
LUMEN = "validation-compound-lumen"
UMBRA = "validation-compound-umbra"


def _q(text, task, **features):
    return {"query_id": "", "text": text, "task_type": task, "compounds": [LUMEN], "features": features}


def _scenarios():
    plasma, ddi, special, mechanism, parameter = v1.TASKS
    # Each source text is newly authored fiction. Required citation groups refer
    # to zero-based original paragraphs, and can require JOINT evidence. Labels
    # are determined from these explicit statements, never compound equality.
    return [
        {"name": "joint_main_use_and_scope", "task": plasma,
         "q": _q(f"Predict plasma concentration-time profiles, AUC and Cmax for {LUMEN} after intravenous administration in adult humans.",
                  plasma, routes=["intravenous"], populations=["adult"], species=["human"], endpoints=["auc", "cmax", "concentration_time"]),
         "texts": [f"The {LUMEN} PBPK candidate explicitly supports plasma concentration-time prediction; further scope details are supplied in another excerpt of this specification.",
                   "For the model use defined in the companion excerpt, the documented population is adult humans and the administration route is intravenous. The supported endpoints include both AUC and Cmax."],
         "grade": 3, "applicable": True, "groups": [[0], [1]],
         "basis": "One paragraph establishes the main prediction; the other supplies every required condition and both endpoints. Both are needed."},
        {"name": "joint_scope_one_population_unknown", "task": special,
         "q": _q(f"Predict oral plasma concentration-time profiles of {LUMEN} in premature human neonates.",
                  special, routes=["oral"], populations=["premature neonates"], species=["human"]),
         "texts": [f"For {LUMEN}, the candidate supports oral plasma concentration-time prediction in adult humans.",
                   f"The physiology of premature human neonates has not been characterized for {LUMEN}. This source provides no conclusion supporting or excluding predictions in those neonates."],
         "grade": None, "applicable": None, "groups": [],
         "basis": "The target population is an essential missing condition; adult support does not establish partial neonatal support."},
        {"name": "all_but_species_supported_explicit_exclusion", "task": special,
         "q": _q(f"Predict oral AUC for {LUMEN} in elderly humans.", special,
                  routes=["oral"], populations=["geriatric"], species=["human"], endpoints=["auc"]),
         "texts": [f"The {LUMEN} model predicts oral AUC in aged rats. Despite the matching route and aging context, its species-specific equations explicitly exclude application to humans, including elderly humans."],
         "grade": 0, "applicable": False, "groups": [[0]],
         "basis": "The documented human exclusion overrides similarities in route, endpoint and aging context."},
        {"name": "different_registered_identity_direct_prediction", "task": plasma,
         "q": _q(f"Directly predict the plasma concentration-time profile of {LUMEN} using this model.", plasma),
         "compounds": [UMBRA],
         "texts": [f"This candidate is a PBPK model of {UMBRA}. All compound-specific differential equations, parameters and predicted plasma concentrations in this document describe {UMBRA}."],
         "grade": 0, "applicable": False, "groups": [[0]],
         "basis": "For DIRECT target prediction, the defining source and registry establish another compound's identity; the target is not in COMPOUNDS."},
        {"name": "other_compound_explicit_shared_template", "task": mechanism,
         "q": _q(f"For {LUMEN}, find a mechanistic reference template representing OCT2-mediated renal secretion; no direct concentration prediction is requested.",
                  mechanism, transporters=["OCT2"]),
         "compounds": [UMBRA],
         "texts": [f"The {UMBRA} candidate implements OCT2-mediated renal secretion as a reusable mechanistic template.",
                   f"The source explicitly supports using these documented OCT2 equations as a structural reference for {LUMEN}. This transfer concerns the requested mechanistic template only, not a claim of validated {LUMEN} concentrations."],
         "grade": 3, "applicable": True, "groups": [[0, 1]],
         "basis": "Different compound identity is not an exclusion for a documented shared mechanism; the template itself is the requested main use."},
        {"name": "other_compound_conflicting_template", "task": mechanism,
         "q": _q(f"For {LUMEN}, find a mechanistic template that represents OCT2-mediated active renal secretion.", mechanism, transporters=["OCT2"]),
         "compounds": [UMBRA],
         "texts": [f"This {UMBRA} candidate contains passive filtration equations only. Its structural specification explicitly excludes carrier-mediated active secretion, including OCT2, and states that it is incompatible with an OCT2 active-secretion template request."],
         "grade": 0, "applicable": False, "groups": [[0]],
         "basis": "The exclusion comes from an explicit mechanistic conflict, not from different compound names."},
        {"name": "registry_target_omitted_from_partial_readme", "task": plasma,
         "q": _q(f"Predict plasma concentration-time profiles of {LUMEN} using the registered artifact.", plasma),
         "compounds": [LUMEN, UMBRA],
         "texts": [f"# {UMBRA} model report", f"This short report describes plasma concentration simulations of {UMBRA}. It does not describe the uses of other compounds in the registered artifact."],
         "grade": None, "applicable": None, "groups": [],
         "basis": "COMPOUNDS contains the target, so a nonexhaustive title cannot exclude it; the excerpt also cannot establish its requested prediction."},
        {"name": "registry_target_in_body_not_title", "task": ddi,
         "q": _q(f"Predict the drug-drug interaction AUC change of {LUMEN} after CYP3A inhibition.", ddi, endpoints=["auc"], enzymes=["CYP3A"]),
         "compounds": [LUMEN, UMBRA],
         "texts": [f"# {UMBRA} interaction repository", f"The registered artifact also contains {LUMEN}. Its {LUMEN} PBPK module explicitly supports prediction of drug-drug interaction AUC changes following CYP3A inhibition."],
         "grade": 3, "applicable": True, "groups": [[1]],
         "basis": "The body explicitly supports the registered target; the nonexhaustive repository title does not override that prose."},
        {"name": "heading_and_actual_support_in_same_excerpt", "task": plasma,
         "q": _q(f"Predict plasma concentration-time profiles of {LUMEN} after inhalation in adult humans.",
                  plasma, routes=["inhalation"], populations=["adult"], species=["human"]),
         "texts": [f"# Respiratory model\nThe {LUMEN} PBPK candidate explicitly supports plasma concentration-time prediction after inhalation in adult humans. This sentence describes the model's supported intended use."],
         "grade": 3, "applicable": True, "groups": [[0]],
         "basis": "Unlike a bare heading, this excerpt contains an explicit supporting body sentence; the heading must not cause that prose to be discarded."},
        {"name": "documented_scope_but_adaptation_pending", "task": plasma,
         "q": _q(f"Predict subcutaneous plasma concentration-time profiles of {LUMEN} in adult humans.",
                  plasma, routes=["subcutaneous"], populations=["adult"], species=["human"]),
         "texts": [f"The {LUMEN} candidate provides explicit partial support for subcutaneous plasma concentration-time prediction in adult humans: its systemic equations cover this use, with a documented depot-absorption adaptation.",
                   f"The depot adaptation for that {LUMEN} prediction still needs parameter calibration. Applicability to the requested subcutaneous adult-human prediction before this adaptation is not established, and is not explicitly excluded."],
         "grade": 2, "applicable": None, "groups": [[0]],
         "basis": "Unlike missing information, partial support and the required adaptation are documented; the pending adaptation leaves applicability unknown."},
        {"name": "ddi_multiple_requirements_one_endpoint_unknown", "task": ddi,
         "q": _q(f"Predict both AUC and Cmax drug-drug interaction changes of {LUMEN} after oral administration in humans with renal impairment.",
                  ddi, routes=["oral"], species=["human"], populations=["renal_impairment"], endpoints=["auc", "cmax"]),
         "texts": [f"For orally administered {LUMEN} in humans with renal impairment, the PBPK candidate supports drug-drug interaction AUC prediction.",
                   f"Cmax drug-drug interaction changes for that {LUMEN} setting have not been studied or documented. The source neither supports nor excludes this requested Cmax endpoint."],
         "grade": None, "applicable": None, "groups": [],
         "basis": "Explicit support for route, population and AUC does not supply the additionally requested Cmax endpoint; unknown is not partial or negative."},
        {"name": "unspecified_special_population_not_healthy_adult", "task": special,
         "q": _q(f"Find a PBPK model of {LUMEN} for special-population plasma concentration-time prediction; the particular special population is not specified.", special),
         "texts": [f"The {LUMEN} candidate documents plasma concentration-time prediction in healthy adult human volunteers. No special-population use is described, and the report makes no special-population exclusion claim."],
         "grade": None, "applicable": None, "groups": [],
         "basis": "Healthy adult volunteers do not establish a special-population use, even when the requested special population is unspecified."},
        {"name": "parameter_workflow_main_use_not_prediction", "task": parameter,
         "q": _q(f"Provide the documented compound parameterization workflow for {LUMEN}, including pKa, solubility and fraction unbound; I request the parameter template, not direct concentration predictions.", parameter),
         "texts": [f"This document supplies the requested {LUMEN} compound parameterization workflow, including the pKa, solubility and fraction-unbound template and their parameter estimation steps.",
                   f"The template is documented as applicable to that workflow for {LUMEN}; direct concentration predictions derived from it have not been evaluated. This limitation concerns a different use, not the supplied parameter workflow."],
         "grade": 3, "applicable": True, "groups": [[0]],
         "basis": "The requested main use is the explicit parameter workflow. An unrequested concentration-validation gap does not lower template relevance to 1."},
        {"name": "enzyme_reference_insufficient_for_ddi_main_use", "task": ddi,
         "q": _q(f"Directly predict the drug-drug interaction AUC change for {LUMEN} following enzyme inhibition.", ddi, endpoints=["auc"]),
         "texts": [f"The {LUMEN} document provides an enzyme-inhibition clearance module as a mechanistic reference for drug-drug interaction work. It is only a mechanistic starting template, insufficient to establish direct AUC-change prediction; such prediction applicability is neither established nor explicitly excluded."],
         "grade": 1, "applicable": None, "groups": [[0]],
         "basis": "A documented relevant mechanism permits the grade-1 reference exception, while direct requested DDI prediction remains unestablished."},
        {"name": "exclusion_is_for_a_different_requested_scope", "task": plasma,
         "q": _q(f"Predict oral AUC and Cmax of {LUMEN} in adult humans.",
                  plasma, routes=["oral"], populations=["adult"], species=["human"], endpoints=["auc", "cmax"]),
         "texts": [f"The {LUMEN} PBPK candidate explicitly supports prediction of both AUC and Cmax after oral dosing in adult humans; applicability to this intended use is established.",
                   f"The same candidate explicitly excludes intravenous dosing and predictions in rats. These exclusions concern those other conditions; they do not exclude the documented oral use in adult humans."],
         "grade": 3, "applicable": True, "groups": [[0]],
         "basis": "The source fully supports every requested condition. Explicit exclusion of a different route or species does not negate this request's support."},
        {"name": "unprovided_paper_is_not_candidate_parameter_evidence", "task": parameter,
         "q": _q(f"Find a documented compound parameterization template for {LUMEN} in this candidate.", parameter),
         "texts": [f"Bibliography entry: Parameterization templates for {LUMEN} — an external paper, whose contents are not supplied here.",
                   f"That bibliography entry does not state that the current candidate contains a {LUMEN} parameterization template. No candidate-specific parameter workflow is described, and no exclusion is established."],
         "grade": None, "applicable": None, "groups": [],
         "basis": "A citation to an unprovided paper is not evidence that the candidate implements the requested workflow; unknown cannot become exclusion."},
    ]


def _new_controls():
    cases = []
    for scenario in _scenarios():
        for variant in ("base", "request_paraphrase", "evidence_reordering", "support_duplicate"):
            cid = f"admission-v2:{scenario['name']}:{variant}"
            q = deepcopy(scenario["q"])
            q["query_id"] = cid
            if variant == "request_paraphrase":
                q["text"] = "Judge the candidate only for this requested purpose and its stated conditions: " + q["text"]
            items = [(index, text) for index, text in enumerate(scenario["texts"])]
            distractor = (None, "Repository maintenance note: figure files are kept separately from text documents; no model-use conclusion is stated in this note.")
            if variant == "evidence_reordering":
                items = [distractor, *reversed(items)]
            elif variant == "support_duplicate":
                items = [*items, items[0], distractor]
            else:
                items.append(distractor)
            body = {"documented_model_name": "Controlled fictional PBPK candidate",
                    "model_compounds": scenario.get("compounds", [LUMEN]),
                    "scope_warning": "Explicit fictional statements for a reading-contract check, not real scientific evidence or clinical guidance.",
                    "excerpts": [{"n": i, "text": text, "kind": "synthetic_contract_control",
                                  "version_specific_validation_verified": False}
                                 for i, (_, text) in enumerate(items, 1)]}
            refs = [{"source_id": f"{cid}:excerpt:{i}", "quote": text, "excerpt_sha256": v1._sha(text),
                     "document_sha256": v1._sha(text), "document_path": f"synthetic-controls-v2/{scenario['name']}",
                     "location": {"char_start": 0, "char_end": len(text)}, "synthetic": True}
                    for i, (_, text) in enumerate(items, 1)]
            allowed = [i for i, (origin, _) in enumerate(items, 1) if origin is not None]
            groups = [[i for i, (origin, _) in enumerate(items, 1) if origin in group]
                      for group in scenario["groups"]]
            expected = {"grades": [scenario["grade"]], "documented_applicability": scenario["applicable"],
                        "supporting_excerpt_numbers": allowed, "required_support_any_of": groups,
                        "source": "New explicit fictional contract statements authored after v2 prompt freeze and before either v2 judge execution.",
                        "reasoning_basis": scenario["basis"]}
            cases.append({"case_id": cid, "kind": "synthetic_control", "candidate_id": "synthetic:v2-candidate",
                          "query": q, "packet": {"body": body, "refs": refs,
                            "packet_sha256": v1._sha(v1._canonical(body)),
                            "source_candidate_id": "synthetic:v2-candidate",
                            "characters": sum(len(text) for _, text in items)},
                          "expected": expected, "scientific_gold": False,
                          "strata": {"scenario": scenario["name"], "variant": variant,
                                     "task_type": scenario["task"], "validation_role": "independent_admission"}})
    return cases


def _hash_inputs(queries, candidates, packets):
    return {"queries": v1._sha(v1._canonical(sorted(queries, key=lambda q: q["query_id"]))),
            "candidates": v1._sha(v1._canonical(sorted(candidates, key=lambda c: c["candidate_id"]))),
            "packets": v1._sha(v1._canonical(packets))}


def _old_inputs():
    manifest = json.loads((OLD / "manifest.json").read_text())
    data = (OLD / "cases.jsonl").read_bytes()
    if v1._sha(data) != manifest["cases_sha256"]:
        raise ValueError("v1 frozen cases were modified")
    return manifest, [json.loads(line) for line in data.splitlines() if line.strip()]


def _prompt_hash():
    digest = v1._sha(Path(prompt.__file__).read_bytes())
    if digest != PROMPT_FILE_SHA256:
        raise ValueError("Frozen v2 prompt changed")
    return digest


def _validate_cases(cases, old_cases, packets):
    if len(cases) != 188 or len({c["case_id"] for c in cases}) != 188:
        raise ValueError("Expected 188 unique validation cases")
    roles = Counter(c["strata"].get("validation_role") for c in cases)
    if dict(roles) != ROLES:
        raise ValueError("Validation role coverage mismatch")
    original = {c["case_id"]: c for c in old_cases}
    independent = []
    for case in cases:
        v1._check_packet(case["packet"])
        role = case["strata"]["validation_role"]
        if role != "independent_admission":
            stripped = deepcopy(case)
            del stripped["strata"]["validation_role"]
            if stripped != original.get(case["case_id"]):
                raise ValueError("A v1 case, source, or expected answer was changed")
            expected_role = "development_regression" if case["kind"] == "synthetic_control" else "unlabelled_real_source"
            if role != expected_role:
                raise ValueError("Legacy case assigned to wrong role")
            if role == "unlabelled_real_source":
                if case["expected"] is not None or case["packet"] != packets[case["candidate_id"]]:
                    raise ValueError("Real pair gold or source was changed")
        else:
            if case["case_id"] in original or case["kind"] != "synthetic_control" or case.get("scientific_gold") is not False:
                raise ValueError("Independent controls overlap old cases or claim scientific gold")
            independent.append(case)
            expected = case["expected"]
            support = set(expected["supporting_excerpt_numbers"])
            if any(type(n) is not int or not 1 <= n <= len(case["packet"]["refs"]) for n in support):
                raise ValueError("Invalid expected supporting citation")
            if any(not group or not set(group) <= support for group in expected["required_support_any_of"]):
                raise ValueError("Impossible expected joint citation group")
            if expected["grades"] != [None] or expected["documented_applicability"] is not None:
                if not expected["required_support_any_of"] or len(expected["required_support_any_of"]) > 3:
                    raise ValueError("Known synthetic judgment lacks achievable source support")
    scenario_counts = Counter(c["strata"]["scenario"] for c in independent)
    if len(scenario_counts) != 16 or set(scenario_counts.values()) != {4}:
        raise ValueError("Independent scene/variant denominator mismatch")
    if {c["query"]["task_type"] for c in independent} != set(v1.TASKS):
        raise ValueError("Independent task coverage mismatch")
    if {c["expected"]["grades"][0] for c in independent} != {None, 0, 1, 2, 3}:
        raise ValueError("Independent grade coverage mismatch")


def _write_once(path, data):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Validation output outside project")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Frozen v2 validation file differs")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        import os
        os.fsync(handle.fileno())


def _coverage(cases):
    new = [c for c in cases if c["strata"]["validation_role"] == "independent_admission"]
    counts = Counter(c["strata"]["scenario"] for c in new)
    return {"scenario_count": len(counts), "variants_per_scenario": 4,
            "tasks": dict(Counter(c["query"]["task_type"] for c in new)),
            "grades": dict(Counter(str(c["expected"]["grades"][0]) for c in new)),
            "scenarios": sorted(counts)}


def build_validation_v2(queries, candidates, packets, out):
    """One-time build from frozen v1 material and novel controlled statements."""
    out = Path(out)
    if (out / "manifest.json").exists():
        return verify_validation_v2(queries, candidates, packets, out)
    if out.exists() and any(out.glob("*_records.jsonl")):
        raise ValueError("Cannot author validation after v2 inference records exist")
    prompt_sha = _prompt_hash()
    old_manifest, old_cases = _old_inputs()
    inputs = _hash_inputs(queries, candidates, packets)
    if inputs != old_manifest["input_sha256"]:
        raise ValueError("Full request/candidate/material inputs changed")
    inherited = deepcopy(old_cases)
    for case in inherited:
        case["strata"]["validation_role"] = ("development_regression" if case["kind"] == "synthetic_control" else "unlabelled_real_source")
    cases = [c for c in inherited if c["kind"] == "synthetic_control"] + _new_controls() + [c for c in inherited if c["kind"] == "real_source_pair"]
    _validate_cases(cases, old_cases, packets)
    data = ("\n".join(v1._canonical(case) for case in cases) + "\n").encode()
    manifest = {"version": VERSION, "frozen_at": datetime.now(timezone.utc).isoformat(),
                "case_count": len(cases), "role_counts": ROLES, "cases_sha256": v1._sha(data),
                "implementation_sha256": v1._sha(Path(__file__).read_bytes()),
                "evaluator_v1_sha256": v1._sha(Path(v1.__file__).read_bytes()),
                "parent_v1_cases_sha256": old_manifest["cases_sha256"],
                "parent_v1_manifest_sha256": v1._sha((OLD / "manifest.json").read_bytes()),
                "v2_prompt_file_sha256": prompt_sha, "v2_system_prompt_sha256": v1._sha(prompt.SYSTEM),
                "input_sha256": inputs,
                "independent_control_coverage": _coverage(cases),
                "development_policy": "All 64 v1 controls retain identical query, packet and expectations; their outputs were reviewed, so these are development regressions, not independent admission controls.",
                "independent_policy": "16 new explicit fictional scenarios, each with four correlated transformations; authored after the fixed v2 prompt and before either v2 model's outputs. No v2 outcomes used for selection or labels.",
                "unlabelled_real_policy": "The same 60 frozen real source pairs remain unlabelled; these are descriptive comparisons, not scientific gold or accuracy validation.",
                "judge_blinding": "Only packet and query enter model inputs; role, scenario, reasoning_basis and expected values remain withheld.",
                "claim_limits": "Independent means prospective controlled contract admission, not expert scientific truth; 64 variants represent 16 scenario families, not 64 independent scientific cases.",
                "freeze_policy": "Neither inputs nor expected judgments nor admission thresholds may be changed in response to outputs. Verify reads existing artifacts without rebuilding them."}
    _write_once(out / "cases.jsonl", data)
    _write_once(out / "manifest.json", (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
    return verify_validation_v2(queries, candidates, packets, out)


def verify_validation_v2(queries, candidates, packets, out):
    """Read-only verifier; never regenerate or overwrite a frozen case."""
    out = Path(out)
    if not out.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Validation input outside project")
    manifest = json.loads((out / "manifest.json").read_text())
    data = (out / "cases.jsonl").read_bytes()
    if manifest["version"] != VERSION or manifest["cases_sha256"] != v1._sha(data):
        raise ValueError("Frozen v2 cases hash/version mismatch")
    if manifest["implementation_sha256"] != v1._sha(Path(__file__).read_bytes()) or manifest["evaluator_v1_sha256"] != v1._sha(Path(v1.__file__).read_bytes()):
        raise ValueError("Frozen validation implementation changed")
    if manifest["v2_prompt_file_sha256"] != _prompt_hash() or manifest["v2_system_prompt_sha256"] != v1._sha(prompt.SYSTEM):
        raise ValueError("Frozen v2 prompt provenance mismatch")
    old_manifest, old_cases = _old_inputs()
    if (manifest["parent_v1_cases_sha256"] != old_manifest["cases_sha256"]
            or manifest["parent_v1_manifest_sha256"] != v1._sha((OLD / "manifest.json").read_bytes())):
        raise ValueError("Parent v1 material changed")
    if manifest["input_sha256"] != _hash_inputs(queries, candidates, packets) or manifest["input_sha256"] != old_manifest["input_sha256"]:
        raise ValueError("Full material source provenance mismatch")
    if manifest["case_count"] != 188 or manifest["role_counts"] != ROLES:
        raise ValueError("Frozen role denominator mismatch")
    cases = [json.loads(line) for line in data.splitlines() if line.strip()]
    _validate_cases(cases, old_cases, packets)
    if manifest["independent_control_coverage"] != _coverage(cases):
        raise ValueError("Frozen control coverage counts differ from cases")
    return manifest


def evaluate_validation_v2(records, cases):
    """Report development, prospective controls and unlabelled pairs separately."""
    if isinstance(cases, (str, Path)):
        cases = [json.loads(line) for line in Path(cases).read_text().splitlines() if line.strip()]
    by_id = {c["case_id"]: c for c in cases}
    if len(by_id) != len(cases):
        raise ValueError("Duplicate v2 cases")
    seen = set()
    for record in records:
        if record["case_id"] not in by_id or record["case_id"] in seen:
            raise ValueError("Unknown or duplicate v2 output")
        seen.add(record["case_id"])
    reports, admission = {}, {}
    for role, count in ROLES.items():
        subset = [c for c in cases if c["strata"].get("validation_role") == role]
        if len(subset) != count:
            raise ValueError("v2 evaluation role denominator mismatch")
        ids = {c["case_id"] for c in subset}
        reports[role] = v1.evaluate_validation([r for r in records if r["case_id"] in ids], subset)
        admission[role] = v1.admission_metrics(reports[role])
    return {"version": VERSION, "complete": len(records) == len(cases), "expected_cases": len(cases),
            "observed_cases": len(records), "by_role": reports, "admission_by_role": admission,
            "primary_admission_role": "independent_admission",
            "scientific_gold_accuracy": None}
