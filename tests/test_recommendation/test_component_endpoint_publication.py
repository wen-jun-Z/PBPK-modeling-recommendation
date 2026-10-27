"""Check the component-input row uses the same frozen endpoint definitions."""
import importlib.util
import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('component_endpoint_publication', ROOT / 'scripts/publish_component_endpoint_table.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.fixture(scope='module')
def publication():
    return m.build(ROOT)


def test_all_original_rows_and_definitions_are_preserved(publication):
    combined, methods, view = publication
    original = m.table.read(ROOT / m.FROZEN)
    assert combined['publication_rows'][:1512] == original['publication_rows']
    assert combined['endpoint_panels'] == original['endpoint_panels']
    assert len(methods) == 15
    assert len(combined['publication_rows']) == 15 * 36 * 3
    assert view['replay']['rows_checked'] == 108
    assert len(view['selected_definitions']) == 11
    assert view['display_policy']['shared_unavailable_families'] == ['C_last', 'Vss']


def test_improvement_and_deterioration_are_both_retained(publication):
    combined, _, view = publication
    lookup = m.table.unique(combined['publication_rows'], ('method_id', 'endpoint_key', 'source_stratum', 'rank'))
    assert lookup[m.METHOD, 'legacy:Cmax_reference_vs_group', 'legacy:reference_individual', 1]['gmfe'] == pytest.approx(1.9002282559898755)
    assert lookup[m.METHOD, 'legacy:Cmax_reference_vs_group', 'legacy:reference_individual', 2]['gmfe'] == pytest.approx(1.650576779214363)
    assert lookup[m.METHOD, 'legacy:Cmax_reference_vs_group', 'legacy:reference_individual', 3]['gmfe'] == pytest.approx(1.9470163608515791)
    assert lookup[m.METHOD, 'legacy:curve_gmfe', 'legacy:reference_individual', 1]['gmfe'] == pytest.approx(1.8743773921758768)
    tmax = next(p for p in view['selected_definitions'] if p['family'] == 'Tmax')
    row = lookup[m.METHOD, tmax['endpoint_key'], tmax['source_stratum'], 1]
    assert row['metric_kind'] == 'absolute_hours'
    assert row['gmfe'] is None
    assert row['mae_h'] == pytest.approx(1.1148333333333331)


def test_curve_twofold_is_pointwise_and_baseline_replays(publication):
    combined, _, _ = publication
    report = m.table.read(ROOT / m.RUN / 'report/summary.json')
    for method in (m.BASE, m.METHOD):
        rows = [r for r in report['curves']['position_rows'] if r['method_id'] == method and r['position'] == 1]
        expected = sum(r['within_twofold_fraction'] for r in rows) / 2
        actual = next(r for r in combined['publication_rows'] if r['method_id'] == method
                      and r['endpoint_key'] == 'legacy:curve_gmfe' and r['rank'] == 1)
        assert actual['within_twofold'] == pytest.approx(expected)
        scenario_fraction = sum(r['curve_gmfe'] <= 2 for r in rows) / 2
        assert not math.isclose(actual['within_twofold'], scenario_fraction)


def test_mismatched_endpoint_qualification_is_rejected():
    report = m.table.read(ROOT / m.RUN / 'report/summary.json')
    original = m.table.read(ROOT / m.FROZEN)
    changed = next(r for r in report['extended_endpoints']['publication_rows'] if r['method_id'] == m.METHOD)
    changed['route'] = 'wrong_route'
    with pytest.raises(ValueError, match='endpoint definition or qualification: route'):
        m.component_rows(report, original, m.METHOD)


def test_same_source_rows_drive_table_and_complete_csv(publication):
    combined, methods, view = publication
    text = m.table.overview_latex(combined, methods)
    assert '15 recommendation configurations' in text
    assert 'PBPK-Match + components' in text
    assert '1.90/1.65/1.95' in text  # Cmax includes the adverse first-rank result.
    assert 'PBPK-Match w/o error' in text
    assert 'w/o error disables predicted-error correction' in text
    assert view['checks']['method_count'] == 15
    assert view['source_validation']['unique_sources_verified'] == 867


def test_control_report_cannot_be_published():
    with pytest.raises(ValueError, match='controls are excluded'):
        m.verify_experiment_sources({'purpose': 'zero_effect_replay_control_not_experiment'},
                                    ROOT, ROOT / m.RUN / 'report/summary.json')


def test_completion_must_bind_the_exact_summary(monkeypatch):
    path = ROOT / m.RUN / 'report/summary.json'
    report = m.table.read(path)
    completion = m.table.read(path.with_name('completion.json'))
    next(r for r in completion['outputs'] if r['path'].endswith('/summary.json'))['sha256'] = '0' * 64
    monkeypatch.setattr(m.table, 'read', lambda path: completion)
    with pytest.raises(ValueError, match='exact component summary'):
        m.verify_experiment_sources(report, ROOT, path)


@pytest.mark.parametrize('path,expected_hash,message', [
    ('../outside-project', '0' * 64, 'outside the project'),
    (m.FROZEN, '0' * 64, 'source changed'),
])
def test_unscoped_or_stale_sources_are_rejected(path, expected_hash, message):
    source = ROOT / m.RUN / 'report/summary.json'
    report = m.table.read(source)
    report['sources'] = [{'path': path, 'sha256': expected_hash}]
    with pytest.raises(ValueError, match=message):
        m.verify_experiment_sources(report, ROOT, source)
