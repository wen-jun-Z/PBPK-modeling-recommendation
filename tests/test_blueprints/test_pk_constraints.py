"""PK semantics, leakage boundaries, and unit conversion checks."""
from copy import deepcopy
import hashlib

import pytest

from pbpk_kb.blueprints.pk_constraints import (
    compare_pk_constraints, normalise_quantity, prepare_pk_constraints,
)


@pytest.fixture
def inputs(tmp_path):
    evidence = tmp_path / 'paper.txt'
    evidence.write_text('An independently identified IV study table.')
    context = {'compound_id': 'pubchem:1', 'species': 'human', 'population': 'healthy_adult',
               'matrix': 'plasma', 'route': 'iv', 'dosing': 'single', 'dose_mg': 1., 'infusion_minutes': 5.}
    record = {'record_id': 'iv_cl', 'compound_id': 'pubchem:1', 'endpoint': 'cl',
              'definition': 'cl_systemic_iv', 'value': 6., 'unit': 'L/h', 'source_kind': 'measured',
              'study_role': 'development', 'study_id': 'study1', 'context': deepcopy(context),
              'source': {'path': 'paper.txt', 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(),
                         'locator': 'Table 1'},
              'uncertainty': {'kind': 'confidence_interval_95', 'lower': 3., 'upper': 9.}}
    return tmp_path, context, record


@pytest.mark.parametrize('endpoint,value,unit,expected', [
    ('cl', .19, 'L/h/kg', (.19*1000/60, 'mL/min/kg')),
    ('vdss', 660., 'mL/kg', (.66, 'L/kg')),
    ('fu', 17., '%', (.17, '1')),
    ('vdss', 83.5, 'L', (83.5, 'L')),
])
def test_units_preserve_absolute_vs_weight_normalised(endpoint, value, unit, expected):
    converted, label = normalise_quantity(endpoint, value, unit)
    assert converted == pytest.approx(expected[0])
    assert label == expected[1]


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), -1.])
def test_invalid_quantities_rejected(value):
    with pytest.raises(ValueError):
        normalise_quantity('cl', value, 'mL/min')


def test_heldout_never_numeric_decoded(inputs):
    root, context, record = inputs
    record.update(study_role='evaluation', value='must not read', source={'path': 'missing'})
    prepared = prepare_pk_constraints([record], context, root)
    assert prepared['records'] == [{'record_id': 'iv_cl', 'status': 'excluded',
                                   'reasons': ['evaluation_or_selection_evidence']}]


def test_dose_mismatch_and_apparent_clearance_not_constraint(inputs):
    root, context, record = inputs
    record['context']['dose_mg'] = 2.
    prepared = prepare_pk_constraints([record], context, root)
    assert prepared['constraint_count'] == 0
    assert 'dose_mg_mismatch' in prepared['records'][0]['reasons']
    record['definition'] = 'cl_over_f'
    record['context'].update(route='oral', dose_mg=1.)
    prepared = prepare_pk_constraints([record], context, root)
    assert 'definition_not_observed_systemic_pk' in prepared['records'][0]['reasons']


@pytest.mark.parametrize('field,value', [('route', 'oral'), ('dosing', 'multiple')])
def test_iv_reference_cannot_constrain_oral_or_repeated_target(inputs, field, value):
    root, context, record = inputs
    context[field] = value
    result = prepare_pk_constraints([record], context, root)
    assert result['constraint_count'] == 0
    assert 'target_not_single_iv' in result['records'][0]['reasons']


def test_predictor_overlap_and_unknown_error_not_fit_targets(inputs):
    root, context, record = inputs
    record.update(source_kind='predicted', training_overlap=True, applicability='in_domain', uncertainty=None)
    prepared = prepare_pk_constraints([record], context, root)
    assert prepared['constraint_count'] == 0
    assert 'predictor_training_overlap_or_unknown' in prepared['records'][0]['reasons']
    assert 'predictive_uncertainty_not_quantified' in prepared['records'][0]['reasons']


def test_measured_precedes_eligible_prediction_without_erasing_it(inputs):
    root, context, record = inputs
    prediction = deepcopy(record)
    prediction.update(record_id='predicted', source_kind='predicted', training_overlap=False,
                      applicability='in_domain', uncertainty={'kind': 'prediction_interval', 'lower': 1., 'upper': 10.})
    prepared = prepare_pk_constraints([prediction, record], context, root)
    assert prepared['constraint_count'] == 1
    assert prepared['records'][0]['status'] == 'diagnostic_only'
    assert prepared['records'][1]['status'] == 'constraint_candidate'


def test_model_weight_conversion_does_not_invent_observation_weight(inputs):
    root, context, record = inputs
    prepared = prepare_pk_constraints([record], context, root)
    profile = {'context': dict(context, weight_kg=50.),
               'metrics': [{'endpoint': 'cl', 'definition': 'cl_systemic_iv', 'value': 2.,
                            'unit': 'mL/min/kg', 'eligible_for_constraint': True}]}
    report = compare_pk_constraints(profile, prepared)
    row = report['comparisons'][0]
    assert row['model_value'] == row['evidence_value'] == 100.
    assert row['fold_error'] == 1.
    assert row['reported_interval_relation'] == 'within'
    assert row['eligible_for_constraint']
    assert 'weight_kg' not in prepared['records'][0]['context']
    assert not report['parameters_modified']
    assert report['composite_score'] is None


def test_wrong_native_dose_disables_constraint_even_if_declared_context_matches(inputs):
    root, context, record = inputs
    prepared = prepare_pk_constraints([record], context, root)
    profile = {'context': dict(context, dose_mg=300.), 'metrics': [
        {'endpoint': 'cl', 'definition': 'cl_systemic_iv', 'value': 100.,
         'unit': 'mL/min', 'eligible_for_constraint': True}]}
    row = compare_pk_constraints(profile, prepared)['comparisons'][0]
    assert not row['eligible_for_constraint']
    assert 'dose_mg_model_mismatch' in row['model_context_reasons']


def test_physiochemical_vss_not_silently_equated_to_iv_nca(inputs):
    root, context, record = inputs
    record.update(endpoint='vdss', definition='vdss_iv', value=1., unit='L/kg', uncertainty=None)
    prepared = prepare_pk_constraints([record], context, root)
    profile = {'context': context, 'metrics': [
        {'endpoint': 'vdss', 'definition': 'vdss_physchem', 'value': 1., 'unit': 'L/kg',
         'eligible_for_constraint': False}]}
    row = compare_pk_constraints(profile, prepared)['comparisons'][0]
    assert row['status'] == 'definition_mismatch'
    assert row['fold_error'] is None


def test_digest_and_project_boundary(inputs):
    root, context, record = inputs
    record['source']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='checksum'):
        prepare_pk_constraints([record], context, root)


def test_missing_metric_remains_explicit(inputs):
    root, context, record = inputs
    report = compare_pk_constraints({'context': context, 'metrics': []}, prepare_pk_constraints([record], context, root))
    assert report['constraint_comparison_count'] == 0
    assert report['comparisons'][0]['status'] == 'model_metric_missing'


def test_source_entry_does_not_silently_ignore_pk_request(tmp_path):
    from pbpk_kb.blueprints.source_workflow import design_source_model
    destination = tmp_path / 'never_created'
    with pytest.raises(ValueError, match='require'):
        design_source_model(source_path=tmp_path/'unused.json', source_sha256='0'*64,
                            simulation_name='unused', request={}, output_dir=destination,
                            project_root=tmp_path, execute=False, pk_evidence={'records': []},
                            pk_context={'compound_id': 'pubchem:1'})
    assert not destination.exists()


def test_explicit_forbidden_study_is_excluded_even_if_mislabelled_development(inputs):
    root, context, record = inputs
    record['value'] = 'no_numeric_access'
    result = prepare_pk_constraints([record], context, root, forbidden_study_ids=['study1'])
    assert result['records'][0]['status'] == 'excluded'
