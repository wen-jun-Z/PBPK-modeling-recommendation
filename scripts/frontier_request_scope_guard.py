"""Narrow request-aware accommodation for the frozen lexical scope guard.

A coarse concentration-time task label must not invent a curve requirement when
the actual request asks only for AUC/Cmax. This remains a necessary-condition
filter, not an entailment judgment, and never changes an answer or citation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
VERSION = "explicit-scope-actual-endpoint-request-v1"
ORIGINAL_FUNCTION_SHA256 = "0b43edf647946e90df4c481eb168a00b174cf22892de057d35d2e0c3a3fe8076"
ORIGINAL_FILE_SHA256 = "af598badfe061d545f09356cf2e1921a3818c8b98c05b24b46ad73da3c6a6a03"
TASK = "plasma_concentration_time_prediction"
ENDPOINTS = {
    "auc": r"\bAUC\w*\b|\barea\s+under\s+(?:the\s+)?(?:(?:plasma|serum|blood)\s+)?(?:concentration\S*\s+)?curve\b",
    "cmax": r"\bC[ _-]?max\b|\b(?:peak|maximum|maximal)\s+(?:(?:plasma|serum|blood)\s+)?concentrations?\b",
}
# Apply after removing endpoint expressions (e.g. 'area under the curve').
ADDITIONAL_OUTPUT = re.compile(
    r"\b(?:concentrations?|profiles?|curves?|trajector\w*|time[\s–—-]*courses?|"
    r"clearance|half[\s–—-]*life|T[ _-]?max|bioavailability|excretion|tissue\w*|"
    r"parameter\w*|mechanis\w*|DDI\w*|interactions?|sensitivity)\b|\bC\s*\(\s*t\s*\)", re.I)
PREDICTIVE = re.compile(
    r"\b(?:predict(?:s|ed|ing)?|simulat(?:e|es|ed|ing))\b|"
    r"\b(?:supports?|enables?|provides?|allows?|used\s+(?:for|to)|capable\s+of)\b"
    r"[^.;!?\n]{0,100}\b(?:predictions?|simulation)\b", re.I)
MODEL = re.compile(r"\b(?:models?|candidates?|simulators?|PBPK|PBBM)\b", re.I)
NEGATED = re.compile(
    r"\b(?:not|no|never|without|cannot|can['’]t|unsupported|unavailable|unestablished|"
    r"fails?|failed|excludes?|excluded|will|would|could|might|plans?|planned|future|"
    r"propos(?:e|ed|al)|aims?|intends?|intended\s+to)\b", re.I)
BIBLIOGRAPHY = re.compile(r"^(?:#{1,6}\s*)?(?:bibliograph\w*|references?|citations?|"
                          r"related\s+(?:papers|publications)|further\s+reading)\s*:?(?:\s|$)", re.I)
AUTHOR_YEAR_REFERENCE = re.compile(
    r"^[A-Z][\w'’–-]*[^\n]{0,120}\(\s*(?:18|19|20)\d{2}[a-z]?\s*\)\s*\.\s+", re.I)
LINK = re.compile(r"\[[^\]]*\]\([^)]*\)|\[[^\]]*\]\[[^\]]*\]|<a\b[^>]*>.*?</a>", re.I)
URL = re.compile(r"https?://\S+", re.I)


def guard_protocol():
    original = ROOT / "scripts/run_frontier_assessment.py"
    if hashlib.sha256(original.read_bytes()).hexdigest() != ORIGINAL_FILE_SHA256:
        raise ValueError("Frozen original scope guard file changed")
    return {"version": VERSION, "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "original_guard_function_sha256": ORIGINAL_FUNCTION_SHA256,
            "original_guard_file_sha256": ORIGINAL_FILE_SHA256,
            "scope": "Only endpoint-only actual requests under the coarse plasma concentration-time task",
            "required": "Cited body prose explicitly predicts/simulates every requested AUC/Cmax endpoint; original scope and identity checks still pass",
            "curve_requests": "Unchanged original main-task requirement when the request asks for concentration/time/profile/curve output",
            "citation_policy": "Selected supplied excerpts only; no headings, links, bibliography or unobserved material as new support",
            "answer_or_citation_rewritten": False, "global_pattern_mutation": False,
            "semantic_entailment_verified": False, "case_ids_or_gold_used": False}


def endpoint_only_request(query):
    if query.get("task_type") != TASK or not isinstance(query.get("text"), str):
        return ()
    features = query.get("features", {})
    if not isinstance(features, dict):
        return ()
    declared = features.get("endpoints") or []
    if not isinstance(declared, list) or any(not isinstance(x, str) for x in declared):
        return ()
    normalized = {x.strip().casefold() for x in declared}
    if not normalized <= set(ENDPOINTS):
        return ()
    text = query["text"]
    found = tuple(key for key, pattern in ENDPOINTS.items() if re.search(pattern, text, re.I))
    if not found or not normalized <= set(found):
        return ()
    remainder = text
    for pattern in ENDPOINTS.values():
        remainder = re.sub(pattern, " ", remainder, flags=re.I)
    return () if ADDITIONAL_OUTPUT.search(remainder) else found


def _body_prose(refs, selected):
    """Retain the original guard's body/link policy, excluding bibliography."""
    output = []
    for n in selected:
        quote = refs[n - 1].get("quote")
        if not isinstance(quote, str) or not quote.strip():
            continue
        in_bibliography = False
        for raw in quote.splitlines():
            line = raw.strip()
            if BIBLIOGRAPHY.match(line):
                in_bibliography = True
                continue
            if re.match(r"^#{1,6}\s", line):
                in_bibliography = False
                continue
            if not line or in_bibliography:
                continue
            remainder = URL.sub("", LINK.sub("", line))
            words = re.findall(r"\b\w+\b", remainder)
            if not words:
                continue
            if LINK.search(line) and len(words) <= 4 and not re.search(
                    r"\b(?:is|are|was|were|has|have|supports?|evaluat\w*|validat\w*|predict\w*|administ\w*|receiv\w*|includ\w*|develop\w*)\b", remainder, re.I):
                continue
            # Numeric reference-list entries and DOI records are bibliographic metadata.
            if (re.match(r"^(?:\[\d+\]|\d+[.)])\s", line) or
                    re.search(r"\bdoi\s*:", line, re.I) or AUTHOR_YEAR_REFERENCE.match(line)):
                continue
            visible = re.sub(r"\[([^\]]*)\]\([^)]*\)|\[([^\]]*)\]\[[^\]]*\]",
                             lambda m: m[1] or m[2] or "", line)
            visible = re.sub(r"<[^>]+>", " ", visible)
            output.append(URL.sub("", visible))
    return output


def _explicit_endpoint_support(refs, selected, endpoints):
    supported = set()
    for line in _body_prose(refs, selected):
        for clause in re.split(r"[.;!?\n]+", line):
            if not MODEL.search(clause) or not PREDICTIVE.search(clause) or NEGATED.search(clause):
                continue
            for name in endpoints:
                if re.search(ENDPOINTS[name], clause, re.I):
                    supported.add(name)
    return set(endpoints) <= supported


def guard_factory(original):
    """Wrap the frozen function; repeated configuration never nests wrappers."""
    seen = set()
    while hasattr(original, "__scope_guard_original__"):
        if id(original) in seen:
            raise ValueError("Cyclic scope guard wrapper")
        seen.add(id(original))
        original = original.__scope_guard_original__
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != ORIGINAL_FUNCTION_SHA256:
        raise ValueError("Scope accommodation requires the frozen original guard")

    def validate_explicit_scope(answer, query, refs, model_compounds=None):
        try:
            return original(answer, query, refs, model_compounds)
        except ValueError as error:
            # Other original guards (invalid citations, identity exclusion, schema)
            # are never intercepted by this narrowly identified main-task failure.
            message = str(error)
            if (not message.startswith("Unsupported explicit scope in cited body text:") or
                    "main_task=" + TASK not in message):
                raise
            endpoints = endpoint_only_request(query)
            if not endpoints or not _explicit_endpoint_support(refs, answer["evidence_numbers"], endpoints):
                raise
            scoped_query = deepcopy(query)
            scoped_query["task_type"] = None
            checked = original(answer, scoped_query, refs, model_compounds)
            result = deepcopy(checked)
            result["checked_constraints"].insert(0, {"dimension": "main_task", "value": TASK,
                                                     "actual_requested_endpoints": list(endpoints)})
            result.update(guard_version=VERSION, actual_request_accommodation=True,
                          original_guard_version=checked.get("guard_version"),
                          semantic_entailment_verified=False)
            return result

    validate_explicit_scope.__scope_guard_original__ = original
    return validate_explicit_scope
