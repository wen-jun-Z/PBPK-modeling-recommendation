"""Publication scope must not turn unknowns into negatives or hide missing ranks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("light_paper_metrics", ROOT / "scripts/pbpk_match_light_paper_metrics.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_binary_scope_excludes_unknown_and_invalid_and_keeps_half_threshold():
    result = m.binary_applicability(np.array([[.5, .2, .8, .7, 1., 1.]]),
                                   np.array([[0, 0, 1, 2, -1, 2]]))
    assert result["confusion"] == {"tp": 1, "fp": 1, "tn": 0, "fn": 1}
    assert result["positive_f1"] == .5
    assert result["known_pairs"] == 3
    assert result["unknown_pairs_excluded"] == 2
    assert result["invalid_pairs_excluded"] == 1


def test_known_fraction_keeps_boundary_ties_and_all_queries():
    result = m.known_topk_coverage([[5, 4, 4, 1], [4, 3, 2, 1]],
                                 [[3, -1, -1, 0], [-1, -1, 0, 0]], k=2)
    assert result["included_candidates"] == [3, 2]
    assert result["per_query"] == [1 / 3, 0]
    assert result["mean"] == 1 / 6


def test_holdout_audit_rejects_group_in_calibration():
    data = {"groups": ["held", "train"], "query_records": [{"query_id": "q0"}, {"query_id": "q1"}],
            "folds": [{"outer_train": [1], "outer_test": [0]}]}
    selection = [{"fold": 0, "selection_indices": [1], "calibration": {"fit_groups": ["held"]}}]
    with pytest.raises(ValueError, match="entered fitting"):
        m.held_out_membership(data, [0], selection)
    selection[0]["calibration"]["fit_groups"] = ["train"]
    assert m.held_out_membership(data, [0], selection)[0]["outer_fold"] == 0


def test_scalar_export_retains_endpoint_with_no_scored_top3():
    summaries = [{"endpoint": "Cmax", "rank": i, "coverage": 0., "gmfe": None} for i in (1, 2, 3)]
    reports = {key: {"execution": {"position_summary": summaries}} for key in m.METHODS}
    rows, counts = m.scalar_rows(reports, [{"endpoint": "Cmax"}])
    assert counts == {"Cmax": 1}
    assert len(rows) == 6 and all(row["gmfe"] is None for row in rows)
    summaries.pop()
    with pytest.raises(ValueError, match="Missing exact-rank"):
        m.scalar_rows(reports, [{"endpoint": "Cmax"}])
