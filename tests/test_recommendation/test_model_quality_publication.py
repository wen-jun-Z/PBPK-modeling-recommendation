"""Checks for paper statistics that must not conceal absent positives or unknowns."""
from pathlib import Path
import hashlib
import json
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from build_model_quality_results import close_tree, positive_metrics, update_paper_index


def test_rare_positive_f1_is_not_negative_dominated_macro_f1():
    values = positive_metrics({"tp": 27, "fp": 3, "tn": 16333, "fn": 47})
    assert values["positive_f1"] == pytest.approx(54 / 104)
    assert values["positive_recall"] == pytest.approx(27 / 74)
    assert values["positive_precision"] == .9
    assert values["constant_zero_probability_brier"] == pytest.approx(74 / 16410)


def test_rejecting_everything_has_zero_recall_and_undefined_precision():
    values = positive_metrics({"tp": 0, "fp": 0, "tn": 16336, "fn": 74})
    assert values["positive_f1"] == values["positive_recall"] == 0
    assert values["positive_precision"] is None


def test_absent_positive_class_is_not_perfect_or_measured_zero():
    values = positive_metrics({"tp": 0, "fp": 0, "tn": 19, "fn": 0})
    assert values["positive_f1"] is None
    assert values["positive_recall"] is None


@pytest.mark.parametrize("actual,expected", [
    ({"metric": 0}, {"metric": None}),
    ({"count": 73}, {"count": 74}),
    ({"point": .8}, {"point": .8, "denominator": 40}),
    ({"metric": float("nan")}, {"metric": .8}),
    ({"value": [1, 2]}, {"value": [1]}),
])
def test_full_recomputation_rejects_missing_or_changed_statistics(actual, expected):
    with pytest.raises(ValueError):
        close_tree(actual, expected)


def test_full_recomputation_tolerates_only_numerical_roundoff():
    close_tree({"metric": .8 + 1e-15, "unknown": None}, {"metric": .8, "unknown": None})


def test_quality_index_preserves_engineering_scope_and_source_record_schema(tmp_path):
    folder = tmp_path / 'results/data'
    folder.mkdir(parents=True)
    path = folder / 'paper_results.json'
    old = {'model_recommendation': {'scientific_quality_metrics_available': False},
           'blueprint': {'unchanged': 7}, 'source_files': {}}
    path.write_text(json.dumps(old))
    input_file = tmp_path / 'input.json'
    input_file.write_text('{}')
    digest = hashlib.sha256(input_file.read_bytes()).hexdigest()
    update_paper_index(tmp_path, {'state': 'complete', 'scope': {'queries': 145},
                                 'source_files': {'input.json': digest}})
    new = json.loads(path.read_text())
    assert new['model_recommendation'] == old['model_recommendation']
    assert new['blueprint'] == old['blueprint']
    assert new['model_recommendation_quality_evaluation_complete'] is True
    assert new['source_files']['input.json'] == {'sha256': digest, 'bytes': 2}
