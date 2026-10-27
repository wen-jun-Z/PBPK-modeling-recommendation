"""Narrow lexical fixes and bounded evidence-focused repair for the v8 rubric.

No grades, citation numbers, case IDs or thresholds are assigned by this module.
The old model output remains immutable. Additional LLM attempts are explicit.
"""
from __future__ import annotations

from copy import deepcopy
import re

from frontier_request_scope_guard import _body_prose

VERSION = "pbpk-scope-aliases-and-focused-repair-v1"
PARAMETER = "compound_parameterization_template"
DDI = "ddi_exposure_prediction"
KINETIC = re.compile(r"\b(?:k\s*cat|V\s*max|K\s*m|k\s*inact|K\s*i)\b", re.I)
FITTING = re.compile(r"\b(?:optimi[sz](?:ed|ation|ing)|estimat(?:ed|ion|ing)|calibrat(?:ed|ion|ing)|fitted)\b", re.I)
NEGATED = re.compile(r"\b(?:not|never|no|cannot|without|unavailable|unreported|planned|future|proposed)\b", re.I)
DDGI = re.compile(r"\bDD(?:\(G\)|G)Is?\b|\bdrug[\s–—-]+drug[\s–—-]+gene[\s–—-]+interactions?\b", re.I)
HUMAN_PROFILE = re.compile(r"\b(?:standard|typical)\s+(?:European|Asian|Japanese|Chinese|Caucasian)\s+(?:male|female)\b", re.I)
PARAMETER_NAVIGATION = re.compile(
    r"^(?:the\s+)?(?:(?:compound|input|final|model|estimated|physicochemical)\s+){0,3}"
    r"parameters?(?:\s+values)?(?:\s+of\s+[^.!?;\n]{1,140})?\s+(?:are|is)\s+"
    r"(?:illustrated|listed|shown|tabulated|summari[sz]ed|provided)\s+"
    r"(?:below|above|in\s+(?:the\s+)?(?:following\s+)?table(?:\s+\d+)?)\.?$", re.I)


def filter_navigation(refs):
    """Remove only standalone parameter-table placeholders from the lexical view.

    Actual source packets, hashes and output citations are never changed.
    A real table/procedure following the placeholder remains available.
    """
    return [{**ref, "quote": "\n".join(line for line in ref.get("quote", "").splitlines()
              if not PARAMETER_NAVIGATION.fullmatch(line.strip()))} for ref in refs]


def additional_lexical_support(refs, selected):
    prose = _body_prose(refs, selected)
    supported = {}
    for text in prose:
        # The unit is an actual supplied prose line, never a heading/link/name.
        if KINETIC.search(text) and FITTING.search(text) and not NEGATED.search(text):
            supported["main_task=" + PARAMETER] = "explicit kinetic-parameter fitting/optimization"
        if DDGI.search(text):
            supported["main_task=" + DDI] = "DD(G)I / drug-drug-gene interaction spelling"
        if HUMAN_PROFILE.search(text):
            supported["species=human"] = "explicit standardized human demographic profile"
    return supported


def guard_factory(previous):
    def validate(answer, query, refs, model_compounds=None):
        view = filter_navigation(refs)
        try:
            return previous(answer, query, view, model_compounds)
        except ValueError as error:
            prefix = "Unsupported explicit scope in cited body text: "
            message = str(error)
            if not message.startswith(prefix):
                raise
            missing = set(message[len(prefix):].split(". Cite", 1)[0].split(", "))
            supported = additional_lexical_support(view, answer.get("evidence_numbers", []))
            fixed = missing & set(supported)
            if not fixed:
                raise
            adjusted = deepcopy(query)
            matched = []
            task_key = "main_task=" + str(query.get("task_type"))
            if task_key in fixed:
                adjusted["task_type"] = None
                matched.append({"dimension": "main_task", "value": query["task_type"], "lexical_form": supported[task_key]})
            if "species=human" in fixed:
                features = adjusted.setdefault("features", {})
                features["species"] = [v for v in features.get("species", []) if v.strip().casefold() != "human"]
                matched.append({"dimension": "species", "value": "human", "lexical_form": supported["species=human"]})
            result = deepcopy(previous(answer, adjusted, view, model_compounds))
            result["checked_constraints"] = matched + result.get("checked_constraints", [])
            result.update(guard_version=VERSION, corrected_lexical_aliases=matched,
                          answer_or_citation_rewritten=False, semantic_entailment_verified=False)
            return result
    return validate


def focused_feedback(error, query):
    """Same feedback policy for every failing case/candidate, without gold labels."""
    task = query.get("task_type")
    guidance = {
        PARAMETER: "A parameterization workflow must actually describe assigning, estimating or optimizing compound parameters. A statement that a model was developed/evaluated, a PK prediction, a repository link, or a table-of-contents heading alone does not describe such a workflow.",
        DDI: "Distinguish DDI exposure predictions from standalone PK or enzyme mentions. Establish every requested endpoint from the cited prose; an interaction statement alone does not establish a requested AUC output.",
        "plasma_concentration_time_prediction": "Do not infer AUC, Cmax or a concentration-time curve from a generic PK statement or an unlabeled GMFE table. Establish every explicitly requested output from supplied prose.",
    }.get(task, "Check the main requested use and every explicit scope condition against the actual cited prose.")
    return ("The previous answer still failed the frozen evidence checks: " + str(error)[:900] + "\n" + guidance +
        "\nRe-read ALL supplied excerpts. If relevant support is present elsewhere, select the actual supporting excerpt numbers (at most three) and judge the request again. Do not repeat an unsupported claim with the same citations. "
        "A missing requested condition is not partial support. If the required support cannot be established, report the appropriate reference-only or unknown judgment under the unchanged rubric. "
        "Do not manufacture evidence or treat this feedback as source material. The grade and applicability remain your evidence-based decision; no target answer is supplied.")


def protocol():
    return {"version": VERSION,
        "lexical_additions": ["kinetic constants explicitly fitted/optimized in source prose", "DD(G)I and drug-drug-gene interaction spelling", "standard/typical named human demographic profiles"],
        "placeholder_filter": "Standalone statements that parameter values are illustrated/listed/shown elsewhere cannot supply the absent values or workflow; real following tables/prose are retained.",
        "unchanged": ["all required endpoints/routes/populations", "citation existence and original source text", "original system prompt and grade definitions", "numerical admission thresholds", "unknown is not negative"],
        "extra_repair": "At most one additional evidence-focused LLM attempt after the original initial generation and one ordinary repair still fail validation; applied uniformly, never selected using control gold answers.",
        "baseline_reuse": "Replay initial outputs and ordinary repairs for both complete 765-case cohorts under the corrected guard; reuse an ordinary repair only when its actual feedback is unchanged. Preserve parent outputs and record hashes. Raw synthetic-control admission remains unchanged and receives no repair. All real-source cases follow the same bounded three-attempt policy as production. This is post-development regression, not blind validation.",
        "scientific_gold_accuracy": None,
        "program_assigns_grade_or_citation": False}
