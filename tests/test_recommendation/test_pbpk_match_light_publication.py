"""Own-model publication must preserve frozen baselines and metric meanings."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("light_publication", ROOT / "scripts/publish_pbpk_match_light.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.fixture
def completed():
    """Read already completed artifacts; never infer, train or alter their files."""
    return m.merged_results()


@pytest.fixture
def table_output(tmp_path, monkeypatch):
    names = ("quality", "execution", "stability", "runtime", "generation")
    originals = {}
    for name in names:
        filename = f"model_recommendation_{name}.tex"
        originals[filename] = m.clean_blocks((m.TABLES / filename).read_text())
        (tmp_path / filename).write_text(originals[filename])
    monkeypatch.setattr(m, "TABLES", tmp_path)
    return tmp_path, originals


def test_merge_preserves_twelve_frozen_configurations_and_appends_two(completed):
    pairs = (("quality_rows", m.DATA / "model_recommendation_quality.json", "rows"),
             ("cmax_rows", m.DATA / "model_recommendation_execution.json", "main_cmax_table"))
    for result_key, path, old_key in pairs:
        original = m.read(path)[old_key]
        assert len(original) == 12
        assert completed[result_key][:12] == original
        assert len(completed[result_key]) == 14
        assert tuple(row["method_id"] for row in completed[result_key][-2:]) == m.OWN
        assert len({row["method_id"] for row in completed[result_key]}) == 14
    old_engineering = m.read(m.DATA / "paper_results.json")["model_recommendation"]
    assert completed["engineering"]["rows"][:12] == old_engineering["rows"]
    assert len(completed["engineering"]["rows"]) == 14
    assert completed["engineering"]["generation"][:3] == old_engineering["generation"]
    assert completed["baseline_values_unchanged"]
    assert not completed["training_or_baseline_inference_performed"]


def test_quality_column_uses_102_request_binary_f1_not_three_class_f1(completed):
    source = m.read(m.DATA / "pbpk_match_light_paper_metrics.json")
    assert source["scope"]["paper_binary_test_requests"] == 102
    assert len(set(source["test_query_ids"])) == 102
    for row in completed["quality_rows"][-2:]:
        details = source["quality_details"][row["method_id"]]
        binary = details["binary_applicability_102"]
        confusion = binary["confusion"]
        assert binary["threshold"] == .5
        assert sum(confusion.values()) == row["test_known_pairs"] == 16410
        assert confusion["tp"] + confusion["fn"] == row["test_positive_pairs"] == 74
        f1 = 2 * confusion["tp"] / (2 * confusion["tp"] + confusion["fp"] + confusion["fn"])
        assert row["test_positive_f1"] == f1 == .52
        assert row["test_positive_f1"] != details["three_class_applicability_145"]["per_class_f1"]["Y"]


def test_curves_and_cmax_keep_distinct_values_and_missing_positions(completed):
    own = m.OWN[1]
    cmax = next(row for row in completed["cmax_rows"] if row["method_id"] == own)
    curves = [row for row in completed["curve_position_rows"] if row["method_id"] == own]
    rosu = sorted((row for row in curves if row["compound"] == "rosuvastatin"), key=lambda row: row["position"])
    assert [row["curve_gmfe"] for row in rosu] == pytest.approx([1.7467406404433083, 1.707942415185163, 1.716081216670531])
    assert cmax["rank1_conditional_gmfe"] != rosu[0]["curve_gmfe"]
    assert all("curve_gmfe" not in row for row in completed["cmax_rows"])
    assert len(curves) == 6
    ator_missing = [row for row in curves if row["compound"] == "atorvastatin" and row["position"] > 1]
    assert len(ator_missing) == 2
    assert all(row["curve_gmfe"] is None and row["curve_coverage"] == 0 for row in ator_missing)
    assert {row["endpoint"] for row in completed["scalar_endpoint_rows"]} == set(
        m.read(m.DATA / "pbpk_match_light_paper_metrics.json")["scalar_endpoint_counts"])
    assert not any("tmax" in row["endpoint"].lower() for row in completed["scalar_endpoint_rows"])


def test_repeated_table_publication_is_idempotent_and_preserves_baseline_tex(completed, table_output):
    directory, originals = table_output
    m.tables(completed)
    first = {path.name: path.read_bytes() for path in directory.iterdir()}
    m.tables(completed)
    assert first == {path.name: path.read_bytes() for path in directory.iterdir()}
    for name, original in originals.items():
        assert m.clean_blocks((directory / name).read_text()) == original
    for name in ("quality", "execution", "stability", "runtime"):
        with (directory / f"model_recommendation_{name}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 14
        assert len({row["method_id"] for row in rows}) == 14
    for name in ("quality", "execution", "stability", "generation"):
        text = (directory / f"model_recommendation_{name}.tex").read_text()
        assert text.count(m.BEGIN) == text.count(m.END) == 1
    runtime = (directory / "model_recommendation_runtime.tex").read_text()
    assert runtime.count(m.BEGIN) == runtime.count(m.END) == 2


def test_unavailable_error_is_printed_na_not_zero(completed, table_output):
    directory, _ = table_output
    value = deepcopy(completed)
    value["cmax_rows"][-1].update(rank3_conditional_gmfe=None, rank3_coverage_percent=0.)
    m.tables(value)
    row = next(line for line in (directory / "model_recommendation_execution.tex").read_text().splitlines()
               if line.startswith("PBPK-Match &"))
    assert "NA (0.0)" in row
    assert "0.000 (0.0)" not in row
    endpoint = (directory / "model_recommendation_scalar_endpoints.tex").read_text()
    assert "NA (0.0)" in endpoint
    assert "0.000 (0.0)" not in endpoint


def test_non_generating_method_does_not_get_fictitious_success_rate(completed, table_output):
    directory, _ = table_output
    own = completed["engineering"]["generation"][-2:]
    for row in own:
        assert row["state"] == "not_applicable"
        assert row["strict_valid"] is row["postprocessed_or_retried_valid"] is row["fallback"] is None
    m.tables(completed)
    text = (directory / "model_recommendation_generation.tex").read_text()
    assert text.count("N/A (non-generative)") == 2
