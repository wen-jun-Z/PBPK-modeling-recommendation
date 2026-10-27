"""Unknown candidates must remain visible in actual-list diagnostics."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('reviewer_full_list', ROOT / 'scripts/reviewer_full_list_evaluation.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_perfect_judged_ndcg_does_not_confirm_unknown_first_recommendation():
    result = m.full_list_row([3., 2., 1.], [-1, 3, 0], [2, 0, 1])
    assert result['judged_only']['ndcg@10'] == 1.
    assert result['top1_unknown'] == 1.
    assert result['top1_supported'] == 0.
    assert result['top3_known_fraction'] == pytest.approx(2 / 3)


def test_exact_ties_are_averaged_without_candidate_order_selection():
    result = m.full_list_row([2., 2., 1.], [3, -1, 0], [0, 2, 1])
    reordered = m.full_list_row([2., 1., 2.], [-1, 0, 3], [2, 1, 0])
    assert result == reordered
    assert result['top1_supported'] == result['top1_unknown'] == .5
    assert result['top1_unsupported'] == 0.


def test_no_known_positive_is_ineligible_not_zero_or_perfect_ranking():
    result = m.full_list_row([3., 2., 1.], [-1, 1, 0], [2, 2, 1])
    summary = m.summarize([result])
    assert summary['requests'] == 1 and summary['ranking_eligible_requests'] == 0
    assert summary['ndcg@10'] is None
    assert summary['top1_unknown'] == 1.


def test_invalid_labels_and_nonfinite_scores_do_not_silently_enter_report():
    with pytest.raises(ValueError):
        m.full_list_row([np.nan, 1.], [0, 3], [1, 0])
    with pytest.raises(ValueError):
        m.full_list_row([2., 1.], [0, 4], [1, 0])
