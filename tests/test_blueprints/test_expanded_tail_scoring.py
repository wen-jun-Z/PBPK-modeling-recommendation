"""Finite-window clinical scoring and reference-provenance isolation."""
from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import score_expanded_tail as scoring
from report_blueprint_validation import Store
from pbpk_kb.blueprints.source_workflow import save, sha256


@pytest.fixture
def sample(tmp_path):
    def write(name, value):
        path = tmp_path / name; save(path, value)
        return {'path': name, 'sha256': sha256(path), 'location': 'synthetic fixture'}
    source = write('source.json', {'origin': 'synthetic observed source'})
    human = {'age_years': 30., 'weight_kg': 70., 'height_cm': 175., 'sex': 'MALE', 'population': 'European', 'seed': 17}
    dosing = {'route': 'intravenous', 'dose_mg': 10., 'start_h': 0., 'infusion_minutes': 0.}
    request = {'entry_reason': 'model_missing', 'compound_id': 'synthetic:A', 'compound_name': 'Synthetic A',
        'human': human, 'dosing': dosing, 'target': {'window_start_h': 0., 'window_end_h': 4.,
            'endpoints': ['plasma_concentration_time'], 'primary_endpoint': 'plasma_concentration_time'},
        'partition_method': 'PKSimStandard', 'evidence_ledger': source, 'evidence': [
            {'parameter': 'molecular_weight', 'compound_id': 'synthetic:A', 'value': 100., 'unit': 'g/mol',
             'basis': 'measured', 'source': source}], 'forbidden_sources': []}
    reference = {'compound_id': 'synthetic:A', 'study_id': 'independent-study',
        'role': 'historical_cross_study_reference', 'matrix': 'plasma', 'unit': 'umol/l', 'time_unit': 'min',
        'molecular_weight_g_mol': 100., 'human': human, 'dosing': dosing, 'source': source,
        'observations': [{'time_min': t, 'concentration_umol_per_l': c} for t, c in [(60., 4.), (120., 2.), (240., 1.)]]}
    native = {'unit': 'umol/l', 'time_unit': 'min', 'time_min': [0., 30., 60., 90., 120., 240., 360.],
              'concentration': [0., 20., 8., 6., 4., 2., 1.]}
    fields = {'auc': ('value_mg_h_per_l', 'mg*h/l'), 'cmax': ('value_mg_per_l', 'mg/l'),
        'tmax': ('value_h', 'h'), 'c_last': ('value_mg_per_l', 'mg/l'),
        'terminal_half_life': ('value_h', 'h'), 'auc_inf': ('value_mg_h_per_l', 'mg*h/l'),
        'clearance': ('value_l_per_h', 'l/h'), 'vz': ('value_l', 'l'), 'mrt': ('value_h', 'h'), 'vss': ('value_l', 'l')}
    dense = {'comparison_observations_used': False, 'endpoint_estimator_uses_observations': False,
        'endpoints': {k: {'status': 'completed', field: 2., 'units': {field: unit}}
                      for k, (field, unit) in fields.items()}}
    case = {'case_id': 'synthetic-case', 'compound_id': 'synthetic:A', 'study_id': reference['study_id'],
        'drug': 'Synthetic A', 'reference_kind': 'observed_curve', 'reference_statistic': 'unspecified',
        'reference_source': write('reference.json', reference),
        'physchem': write('physchem.json', {'identity': {'molecular_weight_g_mol': 100.}}),
        'nca_policy': {'sampling_time_h': [1., 2., 4.], 'dose_ng': 1e7, 'weight_kg': 70., 'infusion_h': 0.}}
    computation = {'status': 'predicted', 'request': write('request.json', request),
        'prediction': write('prediction.json', native), 'dense_endpoints': write('endpoints.json', dense),
        'mapping': write('mapping.json', {'locked': True})}
    def seal():
        computation['completion'] = write('completion.json', {
            'construction': {'request': request, 'mapping': computation['mapping']},
            'artifact_pins': {computation[k]['path']: computation[k]['sha256'] for k in ('request', 'prediction', 'dense_endpoints')}})
    seal()
    return SimpleNamespace(root=tmp_path, write=write, case=case, computation=computation,
                           request=request, reference=reference, native=native, dense=dense, seal=seal)


def score(sample):
    return scoring.score_case(sample.case, sample.computation, Store(sample.root))


def test_bolus_without_lloq_scores_only_observed_window_and_retains_unscored_native_tail(sample):
    result = score(sample)
    assert result['curve_metrics']['gmfe'] == pytest.approx(2.)
    assert result['curve_metrics']['n'] == 3
    endpoints = result['endpoints']
    # Log-down areas: (400-200)/ln2 + 2*(200-100)/ln2 ng.h/mL.
    assert endpoints['auc']['reference'] == pytest.approx(400 / math.log(2))
    assert endpoints['auc']['value'] == pytest.approx(800 / math.log(2))
    assert endpoints['cmax']['value'] == 800.  # Dense peak 2000 at 0.5h is outside the observed window.
    assert endpoints['cmax']['reference'] == 400.
    assert endpoints['tmax']['value'] == endpoints['tmax']['reference'] == 1.
    assert endpoints['tmax']['error'] == 0.
    for key in scoring.WINDOW_ENDPOINTS:
        assert endpoints[key]['reference_kind'] == 'observation_derived'
    for key in set(scoring.UNITS) - scoring.WINDOW_ENDPOINTS:
        assert endpoints[key]['value'] is not None and endpoints[key]['reference'] is None and endpoints[key]['error'] is None
        assert endpoints[key]['prediction_basis'] == 'native_model_only_unscored'
    assert endpoints['clearance']['value'] == pytest.approx(2 / 70)
    assert endpoints['vss']['value'] == pytest.approx(2 / 70)
    assert result['nca_diagnostics']['assay_lloq_ng_ml'] is None
    assert result['nca_diagnostics']['synthetic_predose_added'] is False


@pytest.mark.parametrize('field', ['compound', 'study', 'dose', 'weight', 'molecular_weight', 'units', 'grid'])
def test_reference_identity_conditions_and_units_cannot_be_silently_repaired(sample, field):
    if field == 'compound': sample.reference['compound_id'] = 'other'
    elif field == 'study': sample.reference['study_id'] = 'other'
    elif field == 'dose': sample.reference['dosing'] = {**sample.reference['dosing'], 'dose_mg': 20.}
    elif field == 'weight': sample.reference['human'] = {**sample.reference['human'], 'weight_kg': 80.}
    elif field == 'molecular_weight': sample.reference['molecular_weight_g_mol'] = 101.
    elif field == 'units': sample.reference['unit'] = 'mg/l'
    else: sample.case['nca_policy']['sampling_time_h'][0] = 0.
    sample.case['reference_source'] = sample.write('reference.json', sample.reference)
    with pytest.raises(ValueError): score(sample)


def test_no_extrapolation_when_model_does_not_cover_reference_window(sample):
    sample.native['time_min'] = [120., 240., 360.]
    sample.native['concentration'] = [4., 2., 1.]
    sample.computation['prediction'] = sample.write('prediction.json', sample.native); sample.seal()
    with pytest.raises(ValueError, match='cover'): score(sample)


def test_native_output_must_match_its_completion_receipt(sample):
    sample.native['concentration'][2] = 99.
    sample.computation['prediction'] = sample.write('prediction.json', sample.native)
    with pytest.raises(ValueError, match='receipt'): score(sample)


def test_mismatched_native_array_lengths_are_not_truncated(sample):
    sample.native['concentration'].append(5.)
    sample.computation['prediction'] = sample.write('prediction.json', sample.native); sample.seal()
    with pytest.raises(ValueError, match='equal length'): score(sample)


def test_unavailable_native_tail_stays_null_and_never_becomes_a_clinical_label(sample):
    sample.dense['endpoints']['vss'] = {'status': 'not_estimable', 'value': None, 'reason': 'AUMC gate failed'}
    sample.computation['dense_endpoints'] = sample.write('endpoints.json', sample.dense); sample.seal()
    endpoint = score(sample)['endpoints']['vss']
    assert endpoint['value'] is endpoint['reference'] is endpoint['error'] is None
    assert endpoint['reason'] == 'AUMC gate failed'


def test_original_reported_branch_preserves_legacy_numbers_and_labels_provenance(monkeypatch):
    expected = {'curve_metrics': {'gmfe': 1.5}, 'endpoints': {
        'auc': {'value': 7., 'reference': 8., 'error': 8 / 7},
        'vss': {'value': 2., 'reference': None, 'error': None}}, 'curve': {'old': True}}
    before = deepcopy(expected)
    monkeypatch.setattr(scoring.legacy, '_case_reference', lambda *a: ({}, {}, {}))
    monkeypatch.setattr(scoring.legacy, '_score_prediction', lambda *a: (expected, {}, 100.))
    result = scoring.score_case({'reference_kind': 'original_reported'}, {'status': 'predicted'}, None)
    assert expected == before
    assert result['endpoints']['auc']['value'] == 7. and result['endpoints']['auc']['reference'] == 8.
    assert result['endpoints']['auc']['reference_kind'] == 'original_reported'
    assert result['endpoints']['vss']['reference_kind'] == 'unavailable'
    assert 'curve' not in result and result['scoring_status'] == 'scored'


def test_failed_prediction_is_rejected_before_reference_is_read():
    class NeverRead:
        def ref(self, *_): raise AssertionError('A failed prediction must not decode reference outcomes')
    with pytest.raises(ValueError, match='completed'):
        scoring.score_case({'reference_kind': 'observed_curve'}, {'status': 'failed'}, NeverRead())
