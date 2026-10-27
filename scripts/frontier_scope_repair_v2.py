"""Separate source existence from contribution to a documented-use judgment.

Navigation is real text but cannot establish the absent parameter workflow.
Only the lexical checking view changes; original packets and citations do not.
"""
from copy import deepcopy
import re

import frontier_scope_repair as parent

VERSION = "pbpk-scope-contribution-separation-v2"
DETAILS_NAVIGATION = re.compile(
    r"^(?:further\s+)?(?:details?|information)\s+(?:about|on|regarding)\s+[^\n]{1,250}\s+"
    r"(?:can\s+be\s+found|(?:are|is)\s+(?:given|provided|described))\s+in\s+"
    r"(?:\[[^\]]*\]\([^)]*\)|(?:section|table|appendix)\b[^\n]*)\.?$", re.I)


def is_navigation(line):
    return bool(parent.PARAMETER_NAVIGATION.fullmatch(line.strip()) or DETAILS_NAVIGATION.fullmatch(line.strip()))


def lexical_view(refs):
    # The inherited guard already excludes headings from substantive prose.
    # Mark only navigation lines as headings in this temporary view instead of
    # deleting their contents and manufacturing an 'empty source' violation.
    return [{**ref, "quote": "\n".join(
        "## " + line if is_navigation(line) else line
        for line in ref.get("quote", "").splitlines())} for ref in refs]


def guard_factory(previous):
    corrected = parent.guard_factory(previous)

    def validate(answer, query, refs, model_compounds=None):
        result = deepcopy(corrected(answer, query, lexical_view(refs), model_compounds))
        selected = answer.get("evidence_numbers", [])
        noncontributing = []
        for number in selected:
            if type(number) is not int or not 1 <= number <= len(refs):
                continue
            lines = [line.strip() for line in refs[number - 1].get("quote", "").splitlines() if line.strip()]
            if lines and all(is_navigation(line) for line in lines):
                noncontributing.append(number)
        result.update(guard_version=VERSION, noncontributing_excerpt_numbers=noncontributing,
                      answer_or_citation_rewritten=False, semantic_entailment_verified=False)
        return result

    return validate


focused_feedback = parent.focused_feedback


def protocol():
    value = deepcopy(parent.protocol())
    value.update(version=VERSION,
        placeholder_filter="Standalone parameter-table placeholders and details-can-be-found-in-section navigation remain real nonempty citations, but contribute no scope evidence. They cannot invalidate separate adequate source prose; a decision still fails if its remaining cited prose does not support the requested use.",
        citation_quality="Record noncontributing navigation excerpt numbers separately. They are not supporting parameter evidence, are not silently removed from model output, and are not fabricated empty sources.",
        supersedes="v1 navigation view incorrectly manufactured an empty-source failure; all v1 outputs and protocols are retained.")
    return value
