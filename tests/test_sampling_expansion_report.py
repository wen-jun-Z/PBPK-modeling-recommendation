"""Completion isolation, scientific aggregates and visible cohort denominators."""
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sampling_expansion_report", ROOT / "scripts/report_sampling_expansion.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_completion_checked_before_any_result_or_source_read(monkeypatch):
    seen = []
    def read(path):
        seen.append(Path(path).name)
        assert Path(path).name == "state.json"
        return {"status": "acquiring"}
    monkeypatch.setattr(m, "read", read)
    monkeypatch.setattr(m, "checked", lambda ref: pytest.fail("Premature source read"))
    with pytest.raises(ValueError, match="not complete"):
        m.load_complete()
    assert seen == ["state.json"]


def test_cluster_statistics_weight_drugs_equally_and_keep_cluster_sizes():
    values = {"a": 1., "b": 2., "c": 4.}
    clusters = {"a": "shared", "b": "shared", "c": "single"}
    out = m.bootstrap(values, clusters, seed=3, resamples=2000)
    assert out["estimate"] == pytest.approx(2.)
    assert out["n_cases"] == 3 and out["n_clusters"] == 2
    assert out["ci95"] == pytest.approx([math.sqrt(2.), 4.])
    assert m.bootstrap({}, {}, seed=3, resamples=2000)["estimate"] is None
    one = m.bootstrap({"a": 1., "b": 2.}, {"a": "s", "b": "s"}, seed=3, resamples=2000)
    assert one["ci95"] is None


def test_independent_ranker_has_known_regularized_solution():
    features = {"d__early": {"case_id": "d", "candidate_ids": [0, 1],
                "goals": {"cmax": {"X": [[-1.], [1.]]}}}}
    raw = [{"design_key": "d__early", "candidate": "baseline", "endpoints": {"cmax": {"error": 2.}}},
           {"design_key": "d__early", "candidate": "d__early__point_00", "endpoints": {"cmax": {"error": 2.*math.exp(.2)}}},
           {"design_key": "d__early", "candidate": "d__early__point_01", "endpoints": {"cmax": {"error": 2.*math.exp(-.2)}}}]
    scale, coef = m.independent_ranker(features, raw, {"d"}, "cmax", 1.)
    assert scale == pytest.approx([1.]) and coef == pytest.approx([.1])


def test_random_expectation_missing_value_does_not_become_success():
    assert m.gm([1., 4.]) == pytest.approx(2.)
    assert m.gm([1., None, 4.]) is None
    with pytest.raises(ValueError):
        m.gm([1., -1.])


def test_profile_numpy_scalars_and_missing_statistics_serialize():
    import json
    raw={'flag':np.bool_(True),'count':np.int64(30),'stats':np.array([1.,np.nan])}
    assert json.loads(json.dumps(m.json_native(raw),allow_nan=False)) == {
        'flag':True,'count':30,'stats':[1.,None]}


def test_figure_keeps_all_thirty_names_and_missing_rows():
    sys.path.insert(0, str(m.SKILL))
    cases, outcomes = {}, {}
    for i in range(30):
        cid = f"d{i:02d}"
        cases[cid] = dict(drug=f"Compound {i+1:02d}", original=i < 10, cross_study=i < 28)
        outcomes[cid] = {ep: dict(ratio=None if i == 29 else .8+i/60, learned=None, time=None) for ep in ("cmax", "curve")}
    method_stats = {method: {"endpoint": {"estimate": 1.5, "ci95": [1.2, 2.], "n_cases": 19}} for method in m.METHODS}
    result = {"summary": {"added_twenty": {"late_four": {"cmax": method_stats}}}}
    fig = m.make_figure({}, result, dict(cases=cases, drug_outcomes=outcomes))
    fig.canvas.draw()
    labels = [t.get_text() for ax in fig.axes[1:] for t in ax.get_yticklabels()]
    assert len(labels) == 30 and len(set(labels)) == 30
    assert "Compound 30*" in labels
    assert sum(t.get_text() == "NA" for ax in fig.axes for t in ax.texts) == 2
    import matplotlib.pyplot as plt
    plt.close(fig)
