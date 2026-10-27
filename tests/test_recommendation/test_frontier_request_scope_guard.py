"""CPU regression checks for interpreting endpoint-only user requests.

Uses saved v7 outputs verbatim; does not rewrite validation cases or judgments.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
sys.path.insert(0, str(ROOT / "scripts"))
import run_frontier_assessment as base
import frontier_request_scope_guard as scope

ORIGINAL = base.validate_explicit_scope
COMPOUND = "validation-compound-lumen"
FEATURES = {"endpoints": ["auc", "cmax"], "routes": ["oral"],
            "populations": ["adult"], "species": ["human"]}
REQUEST = f"Predict oral AUC and Cmax of {COMPOUND} in adult humans."
PROSE = f"The {COMPOUND} PBPK candidate explicitly supports prediction of both AUC and Cmax after oral dosing in adult humans; applicability to this intended use is established."


def inputs(prose=PROSE, request=REQUEST, features=None, grade=3, applicable=True):
    return ({"grade": grade, "documented_applicability": applicable,
             "status": "judged", "evidence_numbers": [1]},
            {"query_id": "scope-test", "text": request,
             "task_type": "plasma_concentration_time_prediction",
             "compounds": [COMPOUND], "features": deepcopy(FEATURES if features is None else features)},
            [{"quote": prose}], [COMPOUND])


@pytest.fixture(scope="module")
def actual_failures():
    folder = ROOT / "experiments/model_recommendation/frontier_comparison_20260914/assessment_reasoned_fast_v7/validation"
    cases = {c["case_id"]: c for c in map(json.loads, (folder / "cases.jsonl").read_text().splitlines())}
    records = [r for r in map(json.loads, (folder / "qwen_records.jsonl").read_text().splitlines())
               if r["case_id"].startswith("admission-v2:exclusion_is_for_a_different_requested_scope:")]
    assert len(records) == 4
    return [(r, cases[r["case_id"]]) for r in records]


def test_four_saved_v7_outputs_pass_request_guard_but_reproduce_old_rejection(actual_failures):
    wrapped = scope.guard_factory(ORIGINAL)
    for record, case in actual_failures:
        assert record["final_status"] == "invalid_output"
        answer = base.parse_answers(record["output"], {1}, case["packet"]["refs"])[1]
        assert answer["grade"] == 3 and answer["documented_applicability"] is True
        args = (answer, case["query"], case["packet"]["refs"], case["packet"]["body"]["model_compounds"])
        before = deepcopy(args)
        with pytest.raises(ValueError, match="main_task=plasma_concentration_time_prediction"):
            ORIGINAL(*args)
        result = wrapped(*args)
        assert result["applied"] is True and result["semantic_entailment_verified"] is False
        assert args == before


@pytest.mark.parametrize("extra", [" Also predict the full concentration-time curve.",
                                    " Include the plasma concentration-time profile.",
                                    " The complete plasma concentration time trajectory is required."])
def test_actual_curve_request_cannot_use_endpoint_only_exception(extra):
    with pytest.raises(ValueError):
        scope.guard_factory(ORIGINAL)(*inputs(request=REQUEST + extra))


@pytest.mark.parametrize("prose", [
    PROSE.replace("both AUC and Cmax", "AUC"),
    PROSE.replace("after oral dosing", "after dosing"),
    PROSE.replace("adult humans", "humans"),
    PROSE.replace("adult humans", "adults"),
])
def test_missing_endpoint_or_each_explicit_scope_still_rejected(prose):
    with pytest.raises(ValueError):
        scope.guard_factory(ORIGINAL)(*inputs(prose=prose))


@pytest.mark.parametrize("prose", [
    "# " + PROSE,
    "[" + PROSE + "](https://example.invalid/paper)",
    "Smith, J. (2020). Prediction of oral AUC and Cmax in adult humans. Journal of PBPK, 1, 1-4.",
    "Smith et al. (2020). A PBPK model predicts oral AUC and Cmax in adult humans. Journal of Pharmacology, 1, 1-4.",
    f"Observed oral AUC and Cmax for {COMPOUND} in adult humans were summarized as statistics, not predicted by this candidate.",
    f"The {COMPOUND} model does not support prediction of AUC and Cmax after oral dosing in adult humans.",
    f"Future plans propose prediction of AUC and Cmax after oral dosing in adult humans for {COMPOUND}.",
    f"The {COMPOUND} PBPK model is planned to predict AUC and Cmax after oral dosing in adult humans in a future release.",
])
def test_non_supporting_prose_cannot_activate_endpoint_exception(prose):
    with pytest.raises(ValueError):
        scope.guard_factory(ORIGINAL)(*inputs(prose=prose))


def test_unknown_and_negative_answers_preserve_original_semantics():
    wrapped = scope.guard_factory(ORIGINAL)
    for grade, applicable in ((None, None), (0, False), (1, None)):
        args = inputs(grade=grade, applicable=applicable)
        assert wrapped(*args) == ORIGINAL(*args)
    bad = inputs(prose="A different registered compound has incompatible model uses.", grade=0, applicable=False)
    for function in (ORIGINAL, wrapped):
        with pytest.raises(ValueError, match="Unsupported compound exclusion"):
            function(*bad)


def test_repeated_wrapping_and_threaded_calls_do_not_modify_global_guard_or_inputs():
    patterns = deepcopy(base.TASK_SUPPORT_PATTERNS)
    once = scope.guard_factory(ORIGINAL)
    twice = scope.guard_factory(once)
    assert once.__scope_guard_original__ is ORIGINAL
    assert twice.__scope_guard_original__ is ORIGINAL
    shared = inputs()
    saved = deepcopy(shared)
    def run(index):
        if index % 3 == 0:
            with pytest.raises(ValueError, match="main_task="):
                ORIGINAL(*shared)
            return "old_rejected"
        result = (once if index % 3 == 1 else twice)(*shared)
        assert result["applied"]
        return "new_accepted"
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(100)))
    assert results.count("old_rejected") == 34
    assert results.count("new_accepted") == 66
    assert base.validate_explicit_scope is ORIGINAL
    assert base.TASK_SUPPORT_PATTERNS == patterns and shared == saved


def test_unrecognized_guard_implementation_is_not_wrapped():
    def untrusted(answer, query, refs, model_compounds=None):
        return {"applied": False}
    with pytest.raises((ValueError, RuntimeError)):
        scope.guard_factory(untrusted)


@pytest.mark.parametrize("endpoint", ["AUC", "Cmax"])
def test_single_explicit_endpoint_does_not_acquire_an_unrequested_second_endpoint(endpoint):
    request = f"Predict oral {endpoint} of {COMPOUND} in adult humans."
    prose = f"The {COMPOUND} model predicts oral {endpoint} in adult humans."
    features = {**FEATURES, "endpoints": [endpoint.casefold()]}
    result = scope.guard_factory(ORIGINAL)(*inputs(prose, request, features))
    assert result["applied"] and result["actual_request_accommodation"]


def test_separate_module_import_unwraps_existing_wrapper_without_global_mutation():
    spec = importlib.util.spec_from_file_location("independent_scope_guard_test_import", scope.__file__)
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    wrapped = independent.guard_factory(scope.guard_factory(ORIGINAL))
    assert wrapped.__scope_guard_original__ is ORIGINAL
    assert wrapped(*inputs())["applied"]
    assert base.validate_explicit_scope is ORIGINAL


def test_protocol_discloses_narrow_filter_and_preserves_answers_and_gold():
    protocol = scope.guard_protocol()
    assert protocol["original_guard_function_sha256"] == scope.ORIGINAL_FUNCTION_SHA256
    assert protocol["answer_or_citation_rewritten"] is False
    assert protocol["global_pattern_mutation"] is False
    assert protocol["semantic_entailment_verified"] is False
    assert protocol["case_ids_or_gold_used"] is False
