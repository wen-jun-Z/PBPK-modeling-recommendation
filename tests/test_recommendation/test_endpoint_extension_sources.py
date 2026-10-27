"""Checks for source identity and dimensional safeguards in the endpoint extension."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'endpoint_extension_sources', ROOT / 'scripts/recommendation_endpoint_extension_sources.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope='module')
def report():
    return MODULE.build()


def test_all_scenarios_have_explicit_endpoint_availability(report):
    ids = {p['scenario_id'] for p in report['clinical_protocols']}
    assert len(ids) == 41
    observed = {(r['scenario_id'], r['endpoint']) for r in report['references']}
    missing = {(r['scenario_id'], r['endpoint']) for r in report['missing']}
    assert not observed & missing
    assert observed | missing == {(s, ep) for s in ids for ep in MODULE.ENDPOINTS}
    assert all(r['value'] is None and r['eligibility'] is False for r in report['missing'])
    assert all(r['scenario_id'] == 'clinical:' + r['case_id'] for r in report['references'])
    assert all(len(r['source_ref']['sha256']) == 64 for r in report['references'])


def test_source_inconsistency_does_not_become_a_valid_observation(report):
    last = [r for r in report['references'] if r['compound'] == 'vancomycin' and r['endpoint'] == 'C_last']
    assert len(last) == 1 and last[0]['value'] == 4.68
    assert last[0]['unit'] == 'h' and last[0]['eligibility'] is False
    assert last[0]['reason'] == 'source_unit_time_concentration_swap_unresolved'
    moxi = next(r for r in report['references'] if r['compound'] == 'moxifloxacin' and r['endpoint'] == 'Vz')
    assert moxi['quantity'] == 'Vd/F' and not moxi['eligibility']
    oral = next(r for r in report['references'] if r['compound'] == 'quetiapine' and r['endpoint'] == 'Vz')
    assert oral['quantity'] == 'Vz/F' and oral['body_weight_normalized']
    assert oral['unit'] == 'L/kg' and oral['eligibility']


def test_caffeine_dose_arms_and_assay_limits_are_bound_separately(report):
    rows = [r for r in report['references'] if r['compound'] == 'caffeine' and r['endpoint'] == 'CL']
    assert {r['dose']['value']: r['value'] for r in rows} == {50: 5.58, 250: 6.50}
    assert all(r['quantity'] == 'CL/F' for r in rows)
    assert all(len(r['source_ref']['word_bbox']) == 4 for r in rows)
    assert len({r['source_ref']['column'] for r in rows}) == 2
    p = next(p for p in report['clinical_protocols'] if p['compound'] == 'paracetamol')
    assert p['lloq_ng_ml'] == 250
    assert '0.25 µg/mL' in p['lloq_source']['text']
    # LLOQ for dapagliflozin cannot be copied from the article's other analyte DAP-FOR.
    p = next(p for p in report['clinical_protocols'] if p['compound'] == 'dapagliflozin')
    assert p['lloq_ng_ml'] is None


def test_transcription_guard_rejects_wrong_reference_column():
    source = ROOT / 'experiments/blueprint_design/independent_cohort_20260920/retrieval_round1/sealed_articles/PMC9985189.xml'
    with pytest.raises(ValueError, match='Transcription mismatch'):
        MODULE.xml_cell(source, 'Tab2', 5, 1, 7.91)  # test product 7.96, reference 7.91
