"""Scalar publication keeps endpoint windows, denominators and missingness."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("endpoint_publication", ROOT / "scripts/build_recommendation_endpoint_table.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.fixture(scope="module")
def completed():
    return m.build()


@pytest.fixture
def inputs(monkeypatch):
    original = {name: m.read(ROOT / "results/data" / name) for name in
                ("model_recommendation_execution.json", "pbpk_match_light_paper_metrics.json")}
    monkeypatch.setattr(m, "read", lambda path: original[Path(path).name])
    return original


def test_all_methods_endpoints_and_exact_positions_are_retained(completed):
    rows = completed["rows"]
    assert len(rows) == 14 * 7 * 3 == 294
    assert len({(r["method_id"], r["endpoint"], r["rank"]) for r in rows}) == 294
    assert {r["stratum"] for r in rows} == {"reference_diagnostic"}
    assert len(completed["unmeasured_registered_endpoints"]) == 14
    assert all(r["valid_scalar_comparisons"] == 0 for r in completed["unmeasured_registered_endpoints"])
    assert sum(p["source_scalar_comparisons"] for p in completed["endpoint_panels"]) == 171
    assert completed["checks"]["cmax_value_checks"] == 126
    assert not any("curve" in r["endpoint"] or "tmax" in r["endpoint"].lower() for r in rows)


def test_all_three_statistics_preserve_source_units_and_denominators(completed):
    legacy = m.read(ROOT / "results/data/model_recommendation_execution.json")
    own = m.read(ROOT / "results/data/pbpk_match_light_paper_metrics.json")
    old = {(r["method_id"], r["endpoint"], r["position"]): r for r in legacy["rank_positions"]["summaries"] if r["stratum"] == "reference_diagnostic"}
    new = {(r["method"], r["endpoint"], r["rank"]): r for r in own["scalar_endpoint_rows"]}
    for row in completed["rows"]:
        key = row["method_id"], row["endpoint"], row["rank"]
        if row["method_id"] in m.OWN:
            source = new[key]
            gmfe, accuracy, coverage, n = source["gmfe"], source["within_twofold"], source["coverage"], source["scenarios"]
        else:
            source = old[key]
            gmfe, accuracy, coverage, n = source["conditional_gmfe"], source["conditional_within2fold"], source["evaluable_probability"], source["requested_scenarios"]
        assert row["gmfe"] == gmfe
        assert row["within_twofold_percent"] == (None if accuracy is None else 100 * accuracy)
        assert row["coverage_percent"] == 100 * coverage
        assert row["requested_scenarios"] == n
    cmax = next(row for row in completed["rows"] if row["method_id"] == m.OWN[1] and row["endpoint"] == "Cmax_reference_vs_group" and row["rank"] == 1)
    assert cmax["within_twofold_percent"] == pytest.approx(77.08333333333334)
    assert cmax["coverage_percent"] == pytest.approx(26 / 34 * 100)
    assert cmax["requested_scenarios"] == 34


def test_other_stratum_never_contaminates_reference_results(inputs, completed):
    rows = inputs["model_recommendation_execution.json"]["rank_positions"]["summaries"]
    poison = deepcopy(next(r for r in rows if r["stratum"] == "reference_diagnostic"))
    poison.update(stratum="another_execution_stratum", conditional_gmfe=999.)
    rows.append(poison)
    assert m.build()["rows"] == completed["rows"]


def test_missing_exact_rank_is_a_hard_failure(inputs):
    inputs["pbpk_match_light_paper_metrics.json"]["scalar_endpoint_rows"].pop()
    with pytest.raises(ValueError, match="Missing or unexpected"):
        m.build()


def test_missing_predictions_cannot_be_reported_as_zero_error(inputs):
    row = next(r for r in inputs["pbpk_match_light_paper_metrics.json"]["scalar_endpoint_rows"] if r["coverage"] == 0)
    row["gmfe"] = 0.
    with pytest.raises(ValueError, match="GMFE must"):
        m.build()


def test_cmax_reconciliation_detects_changed_old_table(inputs):
    inputs["model_recommendation_execution.json"]["main_cmax_table"][0]["rank2_conditional_gmfe"] += .1
    with pytest.raises(ValueError, match="Cmax table"):
        m.build()


def test_denominator_mismatch_is_not_silently_normalized(inputs):
    inputs["pbpk_match_light_paper_metrics.json"]["scalar_endpoint_rows"][0]["scenarios"] += 1
    with pytest.raises(ValueError, match="requested scenarios"):
        m.build()


def test_longtable_has_all_panels_repeated_headers_and_explicit_missingness(completed):
    tex = m.latex(completed)
    assert tex.count(r"\begin{longtable}") == tex.count(r"\end{longtable}") == 1
    assert r"\endfirsthead" in tex and r"\endhead" in tex
    assert r"\label{tab:model_execution}" in tex
    assert r"\ref{fig:model_execution_curves}" in tex
    assert tex.count(r"\textbf{PBPK-Match (document)}") == 7
    assert tex.count(r"\textbf{PBPK-Match}") == 7
    assert tex.count("BM25 &") == 7
    assert "NA / NA / 0.0" in tex
    assert "0.000 / 0.0 / 0.0" not in tex
    for panel in completed["endpoint_panels"]:
        assert f"$n={panel['requested_scenarios']}$, $m={panel['source_scalar_comparisons']}$" in tex


def test_csv_and_json_roundtrip_contains_all_rows(completed, tmp_path, monkeypatch):
    import csv
    import json
    monkeypatch.setattr(m, "build", lambda root: completed)
    m.publish(tmp_path)
    report = json.loads((tmp_path / "results/data/model_recommendation_endpoint_accuracy.json").read_text())
    assert report == completed
    with (tmp_path / "results/tables/model_recommendation_endpoint_accuracy.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 294
    assert any(r["gmfe"] == "" and r["coverage_percent"] == "0.0" for r in rows)


@pytest.fixture
def extension(completed):
    inventory = [{"family": key, "reference_scenarios": int(key == "Tmax"),
                  "matched_scenarios": int(key == "Tmax")} for key, _ in m.FAMILIES]
    panel = {"endpoint_key": "Tmax|oral", "family": "Tmax", "quantity": "Tmax", "source_stratum": "main:protocol_matched",
             "source_comparisons": 1, "metric_kind": "absolute_hours", "endpoint_label": "Tmax (oral)", "denominator": 1}
    rows = [{**panel, "method_id": method["method_id"], "rank": rank,
             "gmfe": None, "mae_h": 0. if rank == 1 else None, "within_twofold": None,
             "coverage": 1. if rank == 1 else 0.}
            for method in completed["methods"] for rank in (1, 2, 3)]
    return {"family_inventory": inventory, "endpoint_panels": [panel], "publication_rows": rows}


def test_eleven_outputs_are_retained_and_tmax_is_absolute_error(extension, completed):
    tex = m.extended_latex(extension, completed["methods"])
    for number, (_, title) in enumerate(m.FAMILIES, 1):
        assert f"{number}. {title}" in tex
    assert "MAE (h)" in tex
    assert "0.000 / -- / 100.0" in tex
    assert "NA / -- / 0.0" in tex
    assert tex.count("BM25 &") == 1
    assert tex.count("No qualified comparison.") == 10


def test_extension_rejects_missing_method_position(extension, completed):
    extension["publication_rows"].pop()
    with pytest.raises(ValueError, match="Incomplete endpoint extension"):
        m.extended_latex(extension, completed["methods"])


def test_compact_main_table_keeps_every_comparator_on_one_landscape_float(extension, completed):
    tex = m.extended_latex(extension, completed["methods"], compact=True)
    assert "BM25 &" in tex
    assert "PBPK-Match w/o error" in tex and r"\textbf{PBPK-Match}" in tex
    assert r"\ref{tab:model_execution_all_endpoints}" in tex
    assert r"\label{tab:model_execution}" in tex
    assert tex.count(r"\begin{landscape}") == tex.count(r"\begin{table}[p]") == 1
    assert r"\begin{longtable}" not in tex
    assert "0.00/NA/NA" in tex
    assert r"\PBPKcoverage{100.0/0.0/0.0}" in tex
    assert r"\textbf{0.00}" not in tex
    assert len(extension["publication_rows"]) == 42


def test_extension_rejects_fictitious_accuracy_and_tmax_gmfe(extension, completed):
    extension["publication_rows"][1]["mae_h"] = 0.
    with pytest.raises(ValueError, match="unavailable endpoint"):
        m.extended_latex(extension, completed["methods"])
    extension["publication_rows"][1]["mae_h"] = None
    extension["publication_rows"][0]["gmfe"] = 1.
    with pytest.raises(ValueError, match="Tmax"):
        m.extended_latex(extension, completed["methods"])


@pytest.fixture(scope="module")
def actual_extension():
    return m.read(ROOT / 'results/data/model_recommendation_extended_metrics.json')


def test_overview_selects_by_reference_count_never_observed_performance(actual_extension, completed):
    overview = m.overview_data(actual_extension, completed['methods'])
    altered = deepcopy(actual_extension)
    # Change every available error while preserving validity and coverage. The
    # definition selection must be unchanged and still deterministic if the
    # registry order changes.
    altered['endpoint_panels'].reverse()
    altered['publication_rows'].reverse()
    for row in altered['publication_rows']:
        if row['gmfe'] is not None:
            row['gmfe'] += 10.
        if row['mae_h'] is not None:
            row['mae_h'] += 10.
    other = m.overview_data(altered, completed['methods'])
    assert overview['selected_definitions'] == other['selected_definitions']
    for panel in overview['selected_definitions']:
        options = [p for p in actual_extension['endpoint_panels'] if p['family'] == panel['family']]
        expected = min(options, key=lambda p: (-p['eligible_reference_scenarios'], p['source_stratum'], p['endpoint_key']))
        assert m.panel_key(panel) == m.panel_key(expected)


def test_overview_keeps_all_14_methods_and_exact_source_values(actual_extension, completed):
    overview = m.overview_data(actual_extension, completed['methods'])
    lookup = m.unique(actual_extension['publication_rows'], ('method_id', 'endpoint_key', 'source_stratum', 'rank'))
    assert len(overview['methods']) == 14
    assert len(overview['selected_definitions']) == 11
    assert overview['display_policy']['shared_unavailable_families'] == ['C_last', 'Vss']
    assert len(overview['source_row_bindings']) == len(lookup) == 1512
    assert overview['checks']['selected_source_rows'] == 14 * 11 * 3
    assert overview['checks']['displayed_source_rows'] == 14 * 9 * 3
    for block in overview['layout_data']:
        for rendered in block['ranks']:
            row = lookup[block['method_id'], block['endpoint_key'], block['source_stratum'], rendered['rank']]
            value = row['mae_h'] if row['metric_kind'] == 'absolute_hours' else row['gmfe']
            assert rendered['error'] == value
            assert rendered['error_display'] == m.fmt(value, 2)
            assert rendered['coverage_display'] == m.fmt(100 * row['coverage'], 1)
            assert rendered['source_row_sha256'] == m.canonical_hash(row)


def test_one_main_and_two_additional_matrices_replace_multipage_longtable(actual_extension, completed):
    overview = m.overview_data(actual_extension, completed['methods'])
    assert len(overview['additional_measured_definitions']) == 15
    main = m.overview_latex(actual_extension, completed['methods'])
    extra = m.additional_matrices_latex(actual_extension, completed['methods'])
    assert main.count(r'\begin{table}[p]') == 1
    assert extra.count(r'\begin{table}[p]') == 2
    assert r'\begin{longtable}' not in main + extra
    assert r'\fontsize{7.4}{8.5}' in main
    assert r'\label{tab:model_execution_all_endpoints}' in extra
    assert r'\label{tab:model_execution_all_endpoints_extra}' in extra
    for method in completed['methods']:
        label = m.tex_text(m.SHORT_METHODS[method['method_id']])
        if method['method_id'] == m.OWN[1]:
            label = r'\textbf{' + label + '}'
        assert main.count(label + ' &') == 1
        assert extra.count(label + ' &') == 2


def test_coverage_bold_uses_unrounded_positive_values(actual_extension, completed):
    overview = m.overview_data(actual_extension, completed['methods'])
    lookup = m.unique(actual_extension['publication_rows'], ('method_id', 'endpoint_key', 'source_stratum', 'rank'))
    for block in overview['layout_data']:
        for rendered in block['ranks']:
            maximum = max(lookup[method['method_id'], block['endpoint_key'], block['source_stratum'], rendered['rank']]['coverage']
                          for method in completed['methods'])
            assert rendered['highest_coverage'] == (maximum > 0 and abs(rendered['coverage'] - maximum) <= 1e-12)
