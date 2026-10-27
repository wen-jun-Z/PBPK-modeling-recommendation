"""Prospective fictional reading controls authored after the v5 prompt freeze.

Sixteen authored situations each have two dependent citation-order variants.
These check the explicit document-reading contract, never scientific PBPK gold.
The verifier reads frozen artifacts and does not regenerate or repair them.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations, permutations
import json
from pathlib import Path

import frontier_judge_validation as v1
import frontier_judge_prompt_v5 as prompt
import run_frontier_assessment as assessment

ROOT = Path(__file__).resolve().parents[1]
ASSESS = ROOT / "experiments/model_recommendation/frontier_comparison_20260914/assessment_precedence_v5"
OLD_CASES = ROOT / "experiments/model_recommendation/frontier_comparison_20260914/assessment_evidence_first_v3/validation/cases.jsonl"
PROMPT_FILE_SHA = "f439cba420c41ba8f370b320e9d29d93d8eb7f5ed7d60ca446904e76bc3be924"
PROMPT_SYSTEM_SHA = "7e13a2c94b94b22ab47d3a52b249fa842d2ea15476128579390d856b34e9926c"
VERSION = "pbpk-v5-fresh-fictional-contract-controls-v1"
A, B, M = "fictional-compound-cobalt", "fictional-compound-ochre", "fictional-cobalt-metabolite"


def _q(text, task, **features):
    return {"query_id": "", "text": text, "task_type": task,
            "compounds": [A], "features": features}


def _scenarios():
    plasma, ddi, special, mechanism, parameter = v1.TASKS
    # name, request, source paragraphs, grade, applicability, necessary citation
    # groups (each group contains interchangeable zero-based paragraph indices),
    # registered compounds, explicit reasoning basis.
    return [
        ("observed_measurements_are_not_candidate_predictions",
         _q(f"Predict plasma AUC and Cmax of {A} using this candidate.", plasma, endpoints=["auc", "cmax"]),
         [f"The data appendix contains measured plasma AUC and Cmax values for {A}. These are observations supplied as inputs, not outputs calculated by the candidate.",
          f"The source does not describe whether the {A} candidate predicts AUC or Cmax. The presence of these observed measurements establishes neither that predictive use nor its exclusion."],
         None, None, [], [A], "Observed input measurements do not establish the candidate's requested predictive output."),
        ("registered_parent_not_the_implemented_analyte",
         _q(f"Predict the plasma concentration-time profile of parent {A}, not its metabolite.", plasma),
         [f"Both {A} and {M} are registered because the parent supplies the input dose. The current prediction equations calculate only the metabolite's plasma concentrations.",
          f"Plasma concentration-time prediction of parent {A} is explicitly excluded from this candidate; the parent analyte is not represented as a predicted state."],
         0, False, [[1]], [A, M], "Registration of an input compound does not override an explicit exclusion of its requested analyte output."),
        ("documented_output_matrix_conversion_bridge",
         _q(f"Predict plasma concentration-time profiles of {A} after oral dosing in adult humans.", plasma,
            routes=["oral"], populations=["adult"], species=["human"]),
         [f"The current {A} candidate supports whole-blood concentration-time prediction after oral dosing in adult humans.",
          "Its released output-conversion module is explicitly documented to convert those same whole-blood predictions into plasma concentration-time profiles, with that plasma prediction supported for the same dosing and population setting. This conversion belongs to the current candidate."],
         3, True, [[0], [1]], [A], "Scope from the base model plus its explicit current output conversion jointly establish the requested plasma output; conversion is documented, not assumed."),
        ("two_documented_regimens_do_not_fill_cartesian_product",
         _q(f"Predict plasma concentration-time profiles of {A} after repeated intravenous doses.", plasma, routes=["intravenous"]),
         [f"For {A}, the current candidate supports a single intravenous dose and, separately, repeated oral doses.",
          f"Repeated intravenous dosing of {A} has not been studied or documented. The two documented regimens do not establish that additional combination; it is neither supported nor excluded by this source."],
         None, None, [], [A], "A documented route in one regimen and repeated dosing in another do not establish their undocumented combination."),
        ("incidental_enzyme_is_not_a_requested_ddi_constraint",
         _q(f"Predict the drug-drug interaction AUC change of {A} with {B}. CYP2D6 appears in background material, but this request does not require a CYP2D6 mechanism.", ddi,
            endpoints=["auc"], enzymes=["cyp2d6"]),
         [f"The current model explicitly supports prediction of the drug-drug interaction AUC change for victim {A} with perpetrator {B}, using its documented CYP3A4 mechanism.",
          "CYP2D6 is mentioned only in the background discussion and is not implemented. That background mention does not limit the documented interaction use above."],
         3, True, [[0]], [A, B], "The request explicitly makes the enzyme mention incidental; documented support for its actual DDI/AUC main use is sufficient."),
        ("pairwise_ddi_studies_do_not_establish_simultaneous_triple",
         _q(f"Predict the AUC drug-drug interaction change of {A} when {B} and fictional-compound-sienna are administered simultaneously.", ddi, endpoints=["auc"]),
         [f"This registered network supports AUC drug-drug interaction prediction for {A} with {B} alone, and separately for {A} with fictional-compound-sienna alone.",
          f"The simultaneous three-compound regimen has not been documented or evaluated for {A}. Neither applicability nor an exclusion of that simultaneous regimen is established."],
         None, None, [], [A, B, "fictional-compound-sienna"], "Separate pairwise interaction support cannot be combined into an undocumented simultaneous interaction claim."),
        ("relative_ddi_ratio_explicitly_excludes_absolute_exposure",
         _q(f"Predict the absolute AUC of {A} during a drug-drug interaction; an AUC ratio alone does not fulfill this request.", ddi, endpoints=["auc"]),
         [f"The {A} interaction artifact outputs a relative AUC ratio, with its control exposure normalized to one.",
          f"The specification explicitly excludes using this {A} artifact to predict absolute AUC: it has no absolute-exposure output or conversion. Its relative AUC ratio is not an absolute AUC prediction."],
         0, False, [[1]], [A], "The requested output quantity is explicitly excluded, rather than merely unreported."),
        ("documented_partial_ddi_adaptation_is_not_missing_endpoint",
         _q(f"Predict the drug-drug interaction AUC change of {A} after oral dosing in adult humans using the current adaptation.", ddi,
            endpoints=["auc"], routes=["oral"], populations=["adult"], species=["human"]),
         [f"For oral {A} in adult humans, the current candidate explicitly documents partial support for prediction of the drug-drug interaction AUC change using an enzyme-turnover adaptation. AUC prediction in this exact scope is implemented; it is not an undocumented endpoint or population.",
          f"The documented enzyme-turnover adaptation for {A} is incomplete, so applicability of the current partial model to the requested interaction prediction has not yet been established. The source does not exclude the use."],
         2, None, [[0]], [A], "Explicit partial support for the main use and every stated scope, with an identified incomplete adaptation, supports 2/U rather than unknown or full support."),
        ("pediatric_subrange_does_not_inherit_neonatal_exclusion",
         _q(f"Predict plasma concentration-time profiles of {A} in human children aged 6 to 12 years after oral dosing; neonates are not requested.", special,
            species=["human"], populations=["pediatric"], routes=["oral"]),
         [f"The current {A} candidate explicitly supports oral plasma concentration-time prediction in human children aged 6 to 12 years.",
          f"This same {A} candidate explicitly excludes neonates younger than 28 days. The neonatal exclusion does not apply to its documented 6-to-12-year-old population."],
         3, True, [[0]], [A], "The source distinguishes disjoint age subranges; an exclusion in another subrange does not reject the explicitly supported requested one."),
        ("selected_population_branch_overrides_default_configuration",
         _q(f"Predict plasma concentration-time profiles of {A} during pregnancy with the candidate's pregnancy configuration selected.", special, populations=["pregnancy"]),
         [f"The {A} package opens in its nonpregnant default configuration. The default alone does not model pregnancy.",
          f"The same current {A} release includes a separately selectable pregnancy configuration that explicitly supports plasma concentration-time prediction during pregnancy. The request has selected that documented configuration, not the nonpregnant default."],
         3, True, [[1]], [A], "The requested current configuration is documented; default-setting limitations are not limitations of the selected branch."),
        ("frailty_exclusion_within_documented_geriatric_population",
         _q(f"Predict pharmacokinetics of {A} in frail geriatric patients.", special, populations=["geriatric"]),
         [f"The {A} candidate supports geriatric pharmacokinetic prediction only for nonfrail older adults.",
          f"The specification explicitly excludes pharmacokinetic prediction of {A} in frail geriatric patients. That excluded subgroup is the population requested here."],
         0, False, [[1]], [A], "A documented broad age group does not override an explicit exclusion of the requested nested subgroup."),
        ("current_version_structural_bridge_not_accuracy_transfer",
         _q(f"Provide the current release's mechanistic template for hepatic partition equations of {A}; no numerical prediction-accuracy claim is requested.", mechanism),
         [f"The release-v2 specification defines a reusable mechanistic template for hepatic partition equations of {A}.",
          f"The migration record explicitly states that current release v7 of {A} includes the exact equation template specified in the preceding v2 paragraph, unchanged and available as the current structural template. It does not transfer v2's numerical prediction-validation claims to v7."],
         3, True, [[0], [1]], [A], "An explicit current structural-identity bridge permits the requested template claim, while historical numerical accuracy remains unclaimed."),
        ("renal_reference_is_not_coupled_organ_template",
         _q(f"Provide a coupled hepatic-and-renal mechanistic PBPK template for {A}.", mechanism),
         [f"For {A}, the current source documents a reusable renal-transport mechanism module as a relevant reference for designing the requested coupled hepatic-and-renal template.",
          "That reference module is insufficient to supply the requested coupled-organ template. The coupled main use is not established and is not explicitly excluded; the documented module remains a reference only."],
         1, None, [[0]], [A], "The documented relevant module is reference-only for a larger requested template; it is not partial implementation of that main use."),
        ("cross_compound_workflow_has_explicit_unit_adapter",
         _q(f"Provide a transferable compound-parameterization workflow for {A} that estimates plasma binding and renal clearance.", parameter),
         [f"The parameterization workflow developed for {B} first fits plasma binding from binding observations, then estimates renal clearance using the fitted binding and excretion observations. These are actual workflow steps, not just a parameter-value table.",
          f"The current released unit-and-input adapter explicitly makes the preceding complete workflow applicable as the requested compound-parameterization template for {A}, including both plasma binding and renal clearance estimation. The adapter transfers the procedure, not {B}'s numerical parameter values."],
         3, True, [[0], [1]], [B], "Actual estimation steps plus an explicit current transfer adapter support the cross-compound workflow; no parameter values or transferability are inferred."),
        ("aggregate_clearance_identification_excludes_separate_parameters",
         _q(f"Provide a parameterization workflow that separately identifies hepatic clearance and renal clearance for {A}, rather than only their sum.", parameter),
         [f"The {A} workflow identifies a single aggregate clearance parameter from its supplied observations.",
          f"The documentation explicitly states that this {A} parameterization cannot separately identify hepatic and renal clearance and excludes using the aggregate parameter as two identified clearances. Separate identification is incompatible with this workflow."],
         0, False, [[1]], [A], "The requested separate parameter identification is expressly excluded; this is not an inference from a missing parameter name."),
        ("scaling_reference_is_not_joint_identification_workflow",
         _q(f"Provide the complete joint plasma-binding and renal-clearance parameterization workflow for {A}.", parameter),
         [f"For {A}, the source documents an allometric scaling mechanism as a reusable reference that is relevant to designing the requested joint binding-and-clearance parameterization workflow.",
          "The documented scaling reference does not supply the requested joint parameter-identification workflow. Its contribution is reference-only; applicability to performing that main workflow is unestablished and not excluded."],
         1, None, [[0]], [A], "A documented relevant mechanistic reference is 1/U for the unimplemented main workflow, rather than a partial workflow or an exclusion."),
    ]


def new_controls():
    cases = []
    for name, query, texts, grade, applicability, groups, compounds, basis in _scenarios():
        for variant in ("base", "citation_permutation"):
            cid = f"admission-v5-fresh:{name}:{variant}"
            items = list(enumerate(texts))
            distractor = (None, "Administrative note: repository diagrams use blue labels. This note makes no assertion about model uses.")
            items = [*items, distractor] if variant == "base" else [distractor, *reversed(items)]
            q = deepcopy(query); q["query_id"] = cid
            body = {"documented_model_name": "Fictional candidate for prospective v5 reading controls",
                    "model_compounds": compounds,
                    "scope_warning": "Fictional contract control; no clinical advice or scientific gold.",
                    "excerpts": [{"n": n, "text": text, "kind": "synthetic_contract_control",
                                  "version_specific_validation_verified": False}
                                 for n, (_, text) in enumerate(items, 1)]}
            refs = [{"source_id": f"{cid}:source:{n}", "quote": text,
                     "document_path": f"synthetic-controls-v5/{name}",
                     "document_sha256": v1._sha(text), "excerpt_sha256": v1._sha(text),
                     "location": {"char_start": 0, "char_end": len(text)}, "synthetic": True}
                    for n, (_, text) in enumerate(items, 1)]
            expected = {"grades": [grade], "documented_applicability": applicability,
                        "supporting_excerpt_numbers": [n for n, (i, _) in enumerate(items, 1) if i is not None],
                        "required_support_any_of": [[n for n, (i, _) in enumerate(items, 1) if i in group] for group in groups],
                        "source": "Explicit fictional source statements authored after v5 prompt freeze and before any v5 model output; not expert scientific gold.",
                        "reasoning_basis": basis}
            cases.append({"case_id": cid, "kind": "synthetic_control", "candidate_id": "synthetic:v5-fresh",
                          "query": q, "packet": {"body": body, "refs": refs,
                              "packet_sha256": v1._sha(v1._canonical(body)), "source_candidate_id": "synthetic:v5-fresh",
                              "characters": sum(len(text) for _, text in items)},
                          "expected": expected, "scientific_gold": False,
                          "strata": {"scenario": name, "variant": variant, "task_type": q["task_type"],
                                     "validation_role": "fresh_v5_admission", "seen_by_v5_before_freeze": False}})
    return cases


def expected_citation_sets(case):
    e = case["expected"]
    for count in range(4):
        for ids in combinations(e["supporting_excerpt_numbers"], count):
            if all(set(ids).intersection(group) for group in e["required_support_any_of"]):
                yield list(ids)


def expected_output(case, citations):
    e = case["expected"]
    return json.dumps([{"i": 1, "g": e["grades"][0],
                        "a": {True: "Y", False: "N", None: "U"}[e["documented_applicability"]],
                        "e": citations}], separators=(",", ":"))


def _validate_cases(cases):
    if len(cases) != 32 or len({c["case_id"] for c in cases}) != 32:
        raise ValueError("Expected 32 unique prospective controls")
    scenarios = Counter(c["strata"]["scenario"] for c in cases)
    if len(scenarios) != 16 or set(scenarios.values()) != {2}:
        raise ValueError("Expected 16 situations, two dependent variants each")
    if set(c["query"]["task_type"] for c in cases) != set(v1.TASKS):
        raise ValueError("Expected all five task types")
    old_ids = {json.loads(line)["case_id"] for line in OLD_CASES.read_text().splitlines() if line.strip()}
    for c in cases:
        if c["case_id"] in old_ids or c["kind"] != "synthetic_control" or c.get("scientific_gold") is not False:
            raise ValueError("Fresh control overlaps legacy IDs or claims scientific gold")
        v1._check_packet(c["packet"])
        e = c["expected"]
        allowed = set(e["supporting_excerpt_numbers"])
        if (any(type(n) is not int or not 1 <= n <= len(c["packet"]["refs"]) for n in allowed) or
                any(not group or not set(group) <= allowed for group in e["required_support_any_of"])):
            raise ValueError("Invalid expected citation domain or support group")
        if not list(expected_citation_sets(c)):
            raise ValueError("Expected decision has no constructible supporting citation set")
        for ids in expected_citation_sets(c):
            a = assessment.parse_answers(expected_output(c, ids), [1], c["packet"]["refs"])[1]
            assessment.validate_explicit_scope(a, c["query"], c["packet"]["refs"], c["packet"]["body"]["model_compounds"])
    for name in scenarios:
        pair = [c for c in cases if c["strata"]["scenario"] == name]
        if ({c["strata"]["variant"] for c in pair} != {"base", "citation_permutation"} or
                pair[0]["query"]["text"] != pair[1]["query"]["text"] or
                pair[0]["expected"]["grades"] != pair[1]["expected"]["grades"] or
                pair[0]["expected"]["documented_applicability"] != pair[1]["expected"]["documented_applicability"] or
                sorted(r["quote"] for r in pair[0]["packet"]["refs"]) != sorted(r["quote"] for r in pair[1]["packet"]["refs"])):
            raise ValueError("Citation permutation changed source meaning or expected judgment")


def validate_expected_outputs(cases):
    """Exhaust every accepted reference set AND its order through actual parsers."""
    from frontier_conditional_citation_decoding import conditional_citation_parser

    _validate_cases(cases)
    count = sets = 0
    for case in cases:
        for ids in expected_citation_sets(case):
            sets += 1
            for ordered in permutations(ids):
                output = expected_output(case, list(ordered))
                answer = assessment.parse_answers(output, [1], case["packet"]["refs"])[1]
                assessment.validate_explicit_scope(answer, case["query"], case["packet"]["refs"], case["packet"]["body"]["model_compounds"])
                parser = conditional_citation_parser(1, len(case["packet"]["refs"]))
                for char in output:
                    parser = parser.add_character(char)
                if not parser.can_end():
                    raise ValueError("Expected answer cannot complete frozen conditional grammar")
                record = {"case_id": case["case_id"], "output": output}
                if not v1.evaluate_validation([record], [case])["rows"][0]["control_passed"]:
                    raise ValueError("Constructed expected answer fails the actual evaluator")
                count += 1
    return {"case_count": len(cases), "accepted_citation_sets_checked": sets,
            "ordered_answers_checked": count, "parse_answers_passed": True,
            "unchanged_scope_guard_passed": True, "conditional_grammar_passed": True,
            "unchanged_control_evaluator_passed": True, "model_inference_performed": False}


def _prompt_freeze():
    freeze = json.loads((ASSESS / "prompt_freeze.json").read_text())
    if (v1._sha(Path(prompt.__file__).read_bytes()) != PROMPT_FILE_SHA or
            v1._sha(prompt.SYSTEM) != PROMPT_SYSTEM_SHA or
            freeze["prompt_file_sha256"] != PROMPT_FILE_SHA or
            freeze["system_prompt_sha256"] != PROMPT_SYSTEM_SHA):
        raise ValueError("Frozen v5 prompt changed")
    return freeze


def build_fresh(out):
    out = Path(out).resolve(); out.relative_to(ROOT)
    if any((out / n).exists() for n in ("cases.jsonl", "manifest.json")):
        raise ValueError("Refuse to replace previously authored fresh material")
    if any(ASSESS.glob("**/*records.jsonl")) or any(ASSESS.glob("**/*generation_journal.jsonl")):
        raise ValueError("Fresh cases must precede every v5 model output")
    freeze = _prompt_freeze(); cases = new_controls(); checks = validate_expected_outputs(cases)
    now = datetime.now(timezone.utc).isoformat()
    if datetime.fromisoformat(now) <= datetime.fromisoformat(freeze["frozen_at"]):
        raise ValueError("Prompt must be frozen before authoring these controls")
    data = "".join(v1._canonical(c) + "\n" for c in cases)
    manifest = {"version": VERSION, "frozen_at": now, "case_count": 32,
                "independent_authored_scenarios": 16, "dependent_variants_per_scenario": 2,
                "variant_counts": dict(Counter(c["strata"]["variant"] for c in cases)),
                "task_counts": dict(Counter(c["query"]["task_type"] for c in cases)),
                "grade_counts": dict(Counter(str(c["expected"]["grades"][0]) for c in cases)),
                "scenario_counts": dict(Counter(c["strata"]["scenario"] for c in cases)),
                "cases_sha256": v1._sha(data), "author_implementation_sha256": v1._sha(Path(__file__).read_bytes()),
                "prompt_file_sha256": PROMPT_FILE_SHA, "system_prompt_sha256": PROMPT_SYSTEM_SHA,
                "prompt_frozen_at": freeze["frozen_at"], "prompt_freeze_sha256": v1._sha((ASSESS / "prompt_freeze.json").read_bytes()),
                "legacy_733_cases_sha256": v1._sha(OLD_CASES.read_bytes()),
                "pre_freeze_cpu_validation": checks,
                "scientific_gold_accuracy": None,
                "limitations": ["Fictional explicit reading-contract statements, not expert scientific PBPK labels.",
                    "Sixteen independent authored situations with two dependent citation permutations, not 32 independent scientific cases.",
                    "Authored after disclosed prior-version development failures and the frozen v5 prompt; neither v5 judge produced outputs before freeze.",
                    "All gold follows explicit fictional prose; no legacy compatibility labels, same-drug heuristic, recommender ranking, or judge agreement assigns gold.",
                    "Passing parser, scope guard and grammar only establishes representability; it does not establish correctness on real scientific documents."]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.jsonl").write_text(data)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def verify_fresh(out):
    """Read-only: verify saved material without rebuilding or rewriting it."""
    out = Path(out).resolve(); out.relative_to(ROOT)
    manifest = json.loads((out / "manifest.json").read_text())
    freeze = _prompt_freeze()
    data = (out / "cases.jsonl").read_bytes()
    if (v1._sha(data) != manifest["cases_sha256"] or
            v1._sha(Path(__file__).read_bytes()) != manifest["author_implementation_sha256"] or
            v1._sha(OLD_CASES.read_bytes()) != manifest["legacy_733_cases_sha256"] or
            manifest["prompt_file_sha256"] != PROMPT_FILE_SHA or
            manifest["system_prompt_sha256"] != PROMPT_SYSTEM_SHA or
            manifest["prompt_freeze_sha256"] != v1._sha((ASSESS / "prompt_freeze.json").read_bytes()) or
            datetime.fromisoformat(manifest["frozen_at"]) <= datetime.fromisoformat(freeze["frozen_at"])):
        raise ValueError("Fresh control provenance or frozen digest mismatch")
    cases = [json.loads(line) for line in data.splitlines() if line.strip()]
    _validate_cases(cases)
    if manifest["case_count"] != len(cases) or manifest["scenario_counts"] != dict(Counter(c["strata"]["scenario"] for c in cases)):
        raise ValueError("Fresh control denominator mismatch")
    return manifest
