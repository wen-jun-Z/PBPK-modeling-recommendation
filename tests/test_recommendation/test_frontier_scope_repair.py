"""Necessary-condition regressions for narrow source-prose spelling fixes."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_frontier_assessment as base
from frontier_request_scope_guard import guard_factory as v8_guard
from frontier_scope_repair import guard_factory, focused_feedback

GUARD = guard_factory(v8_guard(base.validate_explicit_scope))


def check(quote, task="compound_parameterization_template", **features):
    return GUARD({"grade": 3, "documented_applicability": True, "evidence_numbers": [1]},
        {"task_type": task, "text": "Find a supported model", "features": features, "compounds": ["drug"]},
        [{"quote": quote}], ["drug"])


def test_actual_kinetic_optimization_is_parameter_prose():
    result = check("The CYP2D6 k<sub>cat</sub> values were optimized for the activity scores.")
    assert result["corrected_lexical_aliases"][0]["dimension"] == "main_task"
    assert result["answer_or_citation_rewritten"] is False


@pytest.mark.parametrize("quote", ["The CYP2D6 kcat values were not optimized.",
    "The model was developed and evaluated.", "## CYP2D6 kcat values optimized",
    "[CYP2D6 kcat values optimized](https://example.invalid)", "The kcat value is 1.5."])
def test_no_workflow_heading_and_negated_estimation_stay_unsupported(quote):
    with pytest.raises(ValueError):
        check(quote)


@pytest.mark.parametrize("expression", ["DD(G)I", "DDGI", "drug-drug-gene interaction"])
def test_ddgi_spelling_retains_ddi_and_endpoint_checks(expression):
    result = check(f"The model predicts AUC during {expression}.", task="ddi_exposure_prediction", endpoints=["auc"])
    assert result["corrected_lexical_aliases"]
    with pytest.raises(ValueError, match="endpoints=cmax"):
        check(f"The model predicts AUC during {expression}.", task="ddi_exposure_prediction", endpoints=["cmax"])


def test_gene_only_interaction_is_not_drug_drug_interaction():
    with pytest.raises(ValueError, match="main_task"):
        check("The model predicts AUC during drug-gene interaction.", task="ddi_exposure_prediction")


def test_named_human_demographic_profile_is_narrow():
    result = check("CYP3A4 metabolism in the Standard European Male for DDI.",
                   task="mechanistic_template_recommendation", species=["human"])
    assert result["corrected_lexical_aliases"][0]["value"] == "human"
    with pytest.raises(ValueError, match="species=human"):
        check("CYP3A4 metabolism in male rats.", task="mechanistic_template_recommendation", species=["human"])
    with pytest.raises(ValueError, match="populations=pediatric"):
        check("CYP3A4 metabolism in the Standard European Male for DDI.",
              task="mechanistic_template_recommendation", species=["human"], populations=["pediatric"])


def test_alias_correction_cannot_hide_missing_other_dimensions_or_rewrite_answer():
    with pytest.raises(ValueError, match="species=human"):
        check("The CYP2D6 kcat values were optimized.", species=["human"])
    answer = {"grade": None, "documented_applicability": None, "evidence_numbers": []}
    assert GUARD(answer, {"task_type": "ddi_exposure_prediction", "features": {}}, [], ["drug"])["applied"] is False
    assert answer == {"grade": None, "documented_applicability": None, "evidence_numbers": []}


def test_navigation_placeholder_cannot_supply_missing_parameter_values():
    placeholder = "The compound parameter values of the final digoxin PBPK model are illustrated below."
    with pytest.raises(ValueError):
        check(placeholder)
    result = check(placeholder + "\nThe permeability parameter was estimated using the observed data.")
    assert result["applied"]
    # Describing actual estimates is substantive prose, not a navigation-only line.
    assert check("The parameter values were estimated by fitting clinical concentration data.")["applied"]


def test_feedback_has_no_case_id_grade_or_citation_override():
    q = {"task_type": "compound_parameterization_template", "case_id": "never-reveal-gold", "grade": 3}
    text = focused_feedback("main_task=compound_parameterization_template", q)
    assert "never-reveal-gold" not in text and "g=3" not in text
    assert "no target answer" in text and "at most three" in text
