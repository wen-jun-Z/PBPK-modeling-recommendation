"""Real existence, contribution and missing support are separate conditions."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_frontier_assessment as base
from frontier_request_scope_guard import guard_factory as original_guard
from frontier_scope_repair_v2 import guard_factory

GUARD = guard_factory(original_guard(base.validate_explicit_scope))
NAV = "The compound parameter values of the final PBPK model are illustrated below."
QUERY = {"task_type": "compound_parameterization_template", "features": {}, "compounds": ["drug"]}
ANSWER = {"grade": 3, "documented_applicability": True, "evidence_numbers": [1, 2]}


def test_navigation_does_not_invalidate_other_actual_parameter_workflow():
    refs = [{"quote": NAV}, {"quote": "A specific set of parameters was optimized using the Parameter Identification module provided in PK-Sim."}]
    before = deepcopy((ANSWER, QUERY, refs))
    result = GUARD(ANSWER, QUERY, refs, ["drug"])
    assert result["noncontributing_excerpt_numbers"] == [1]
    assert result["answer_or_citation_rewritten"] is False
    assert (ANSWER, QUERY, refs) == before


def test_navigation_plus_identity_cannot_supply_the_absent_workflow():
    refs = [{"quote": NAV}, {"quote": "The report evaluates the PBPK model for drug in adults."}]
    with pytest.raises(ValueError, match="main_task=compound_parameterization_template"):
        GUARD(ANSWER, QUERY, refs, ["drug"])


def test_truly_empty_selected_source_remains_invalid_even_with_other_support():
    refs = [{"quote": " "}, {"quote": "The parameters were optimized using clinical data."}]
    with pytest.raises(ValueError, match="empty source excerpt"):
        GUARD(ANSWER, QUERY, refs, ["drug"])


def test_navigation_cannot_mask_a_missing_requested_endpoint():
    query = {**QUERY, "features": {"endpoints": ["cmax"]}}
    refs = [{"quote": NAV}, {"quote": "The parameters were optimized using clinical data."}]
    with pytest.raises(ValueError, match="endpoints=cmax"):
        GUARD(ANSWER, query, refs, ["drug"])


@pytest.mark.parametrize("pointer", [
    "Details about the structural model and its parameters can be found in [Section 2.3](#23-model-parameters-and-assumptions).",
    "Details about compound properties (physicochemical, *in vitro* and clinical) can be found in [Section 2.2](#22-data).",
])
def test_section_navigation_does_not_supply_workflow_or_poison_supported_prose(pointer):
    with pytest.raises(ValueError, match="main_task=compound_parameterization_template"):
        GUARD(ANSWER, QUERY, [{"quote": pointer}, {"quote": "The model is for drug in adults."}], ["drug"])
    result = GUARD(ANSWER, QUERY, [{"quote": pointer}, {"quote": "The parameters were optimized using clinical data."}], ["drug"])
    assert result["noncontributing_excerpt_numbers"] == [1]
