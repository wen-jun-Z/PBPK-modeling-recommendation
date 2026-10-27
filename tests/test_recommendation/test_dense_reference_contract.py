"""Missing clinical time grids permit only source-bound reference approximations."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from recommendation_execution_repairs import clinical_contract as c
import recommendation_candidate_executor as executor
import score_recommendation_execution as scorer


def scenario(compound):
    path = ROOT/'experiments/model_recommendation/execution_validation_20261001/inputs/scenarios.jsonl'
    return next(json.loads(line) for line in path.open() if json.loads(line)['compound'] == compound)


@pytest.mark.parametrize('compound,end,points', [('furosemide', 720, 121), ('posaconazole', 8640, 1441)])
def test_registered_horizon_is_used_without_claiming_clinical_samples(compound, end, points):
    s = scenario(compound); before = deepcopy(s)
    contract = c.scenario_contract(s)
    assert s == before and s['normalized_design']['sampling']['times_h'] is None
    assert contract['sampling_times_min'] == list(range(0, end + 1, 6))
    assert contract['absolute_sampling_times_min'] == contract['sampling_times_min']
    registration = contract['sampling_grid_registration']
    assert registration['numerical_points'] == points
    assert registration['reported_clinical_sampling_grid_known'] is False
    assert registration['source_protocol']['sha256'] == s['source_protocol']['sha256']
    assert 'not AUC_last or AUC_infinity' in registration['AUC_definition']


@pytest.mark.parametrize('tamper', ['sha', 'protocol', 'real_grid', 'permission', 'repeat', 'origin', 'normalized_dose', 'normalized_matrix'])
def test_approximation_rejects_protocol_changes_and_cannot_replace_a_real_grid(tamper):
    s = scenario('furosemide')
    if tamper == 'sha': s['source_protocol']['sha256'] = '0' * 64
    elif tamper == 'protocol': s['protocol_fields']['protocol']['observation_duration_min'] = 2880
    elif tamper == 'real_grid': s['normalized_design']['sampling']['times_h'] = [0, 1, 12]
    elif tamper == 'permission': s['normalized_design']['prediction_contract']['source_reference_subject_prediction_allowed'] = False
    elif tamper == 'repeat': s['normalized_design']['regimen']['administrations'].append({'time_min': 60, 'dose': 40, 'dose_unit': 'mg'})
    elif tamper == 'normalized_dose': s['normalized_design']['dose']['value'] = 1000
    elif tamper == 'normalized_matrix': s['normalized_design']['matrix'] = 'whole_blood'
    else: s['normalized_design']['regimen']['observation_origin_h'] = 1
    with pytest.raises(ValueError): c.scenario_contract(s)


def test_unregistered_missing_grid_case_does_not_inherit_an_assumed_horizon():
    with pytest.raises(ValueError, match='SAMPLING_TIMES'):
        c.scenario_contract(scenario('efavirenz'))


def test_posaconazole_alias_requires_exact_frozen_source_and_named_variant():
    from recommendation_execution_repairs.snapshot_candidate import source_parent_identity
    contract = c.scenario_contract(scenario('posaconazole'))
    digest = contract['native_binding']['snapshot']['sha256']
    identity = source_parent_identity(contract, digest)
    assert identity['native_name'] == 'Posaconazol ohne supersaturation'
    assert identity['source_compound_or_mechanism_renamed'] is False
    assert source_parent_identity(contract, '0' * 64) is None
    contract['native_binding']['compound_name'] = 'Posaconazol mit supersaturation'
    with pytest.raises(ValueError, match='FROZEN_POSACONAZOLE'):
        source_parent_identity(contract, digest)


def dense_prediction():
    s = scenario('furosemide'); contract = c.scenario_contract(s)
    times = contract['sampling_times_min']
    curve = {'time_min': times, 'concentration': [0.] + [1.] * (len(times) - 1),
             'time_unit': 'min', 'unit': 'umol/l'}
    plan = {'contract': contract, 'molecular_weight_g_per_mol': 100., 'output_path': 'synthetic',
            'candidate_id': 'synthetic', 'scenario_id': 'synthetic', 'compatibility': {}}
    return s, executor.prediction_from_curve(curve, plan)


def test_dense_peak_is_a_distinct_approximate_endpoint_not_a_reported_sampled_peak():
    s, prediction = dense_prediction()
    assert 'sampled_Cmax' not in prediction['endpoints']
    assert prediction['endpoints']['dense_grid_Cmax']['value'] == 100
    comparison = scorer.scalar_comparison({'endpoint': 'Cmax', 'value': 100, 'unit': 'ng/mL'}, s, prediction)
    assert comparison['state'] == 'valid' and comparison['scientific_status'] == 'unknown'
    assert 'sampling times are unknown' in comparison['definition_warning']
    for endpoint in ('AUC_infinity', 'AUC_last'):
        result = scorer.scalar_comparison({'endpoint': endpoint, 'value': 1200, 'unit': 'ng*h/mL'}, s, prediction)
        assert result['state'] == 'unavailable'


@pytest.mark.parametrize('tamper', ['registration_removed', 'claims_measured_grid', 'both_endpoints',
                                 'rename_to_sampled', 'wrong_protocol', 'wrong_horizon'])
def test_dense_peak_cannot_be_scored_without_its_frozen_approximation(tamper):
    s, prediction = dense_prediction()
    if tamper == 'registration_removed': prediction.pop('sampling_grid_registration')
    elif tamper == 'claims_measured_grid': prediction['sampling_grid_registration']['reported_clinical_sampling_grid_known'] = True
    elif tamper == 'both_endpoints': prediction['endpoints']['sampled_Cmax'] = prediction['endpoints']['dense_grid_Cmax']
    elif tamper == 'rename_to_sampled':
        prediction['endpoints']['sampled_Cmax'] = prediction['endpoints'].pop('dense_grid_Cmax')
        prediction.pop('sampling_grid_registration')
    elif tamper == 'wrong_protocol': prediction['sampling_grid_registration']['source_protocol']['sha256'] = '0' * 64
    else: prediction['sampling_grid_registration']['source_reported_horizon_minutes'] = 99999
    result = scorer.scalar_comparison({'endpoint': 'Cmax', 'value': 100, 'unit': 'ng/mL'}, s, prediction)
    assert result['state'] == 'unavailable'
