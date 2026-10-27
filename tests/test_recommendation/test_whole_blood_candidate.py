"""Source-native blood observations must never be substituted with plasma.

Uses frozen design/source metadata and synthetic observations only.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import recommendation_candidate_executor as executor
import score_recommendation_execution as scoring
from recommendation_execution_repairs.snapshot_candidate import reference_source_options


def context():
    inputs = ROOT/'experiments/model_recommendation/execution_validation_20261001/inputs'
    scenario = next(json.loads(line) for line in (inputs/'scenarios.jsonl').open()
                    if json.loads(line)['compound'] == 'tacrolimus')
    candidates = [json.loads(line) for line in (inputs/'candidate_audit.jsonl').open()
                  if json.loads(line)['scenario_id'] == scenario['scenario_id']
                  and json.loads(line).get('identity_match') is True]
    return scenario, candidates


def native_fixture():
    """Pinned prior native export is a selection fixture, not a new repair pass."""
    directory = ROOT/'experiments/model_recommendation/execution_validation_20261001/engineering_repairs/snapshot_repairs_05/1b7c1fcc7a0a56e972fb21274dcd00e1e2a00e1fad2544778067930d9464224d'
    receipt = json.loads((directory/'receipt.json').read_text())
    path = ROOT/receipt['snapshot']['path']
    assert executor._sha(path) == receipt['snapshot']['sha256']
    return json.loads(path.read_text())


def test_frozen_tacrolimus_protocol_preserves_exact_matrix_and_grid():
    scenario, _ = context()
    c = executor.scenario_contract(scenario)
    assert c['matrix'] == 'whole_blood'
    assert c['sampling_times_min'] == scenario['protocol_fields']['protocol']['sampling_times_min']
    assert c['dose_mg'] == 1 and c['analyte'] == 'parent' and c['binding'] == 'total'


def test_all_four_original_tacrolimus_sources_declare_blood_controls():
    scenario, candidates = context()
    assert len(candidates) == 4
    for candidate in candidates:
        prov = candidate['catalog_provenance']
        raw = json.loads((ROOT/prov['source_path']).read_text())
        controls = []
        for sim in raw['Simulations']:
            try: selected = executor._one_administered_selection(sim)
            except ValueError: continue
            if selected['Name'] == 'Tacrolimus': controls.append(sim)
        assert controls
        assert all('Organism|PeripheralVenousBlood|Tacrolimus|Whole Blood (Peripheral Venous Blood)'
                   in sim['OutputSelections'] for sim in controls)


def test_verified_native_fixture_selects_exact_original_blood_output_without_mutation():
    scenario, candidates = context(); raw = native_fixture(); before = deepcopy(raw)
    options, rejected = reference_source_options(raw, executor.scenario_contract(scenario),
        source_sha256=candidates[0]['catalog_provenance']['source_sha256'], reference_allowed=True)
    assert options, rejected
    assert raw == before
    assert all(o['output_path'] == 'Organism|PeripheralVenousBlood|Tacrolimus|Whole Blood (Peripheral Venous Blood)'
               for o in options)


@pytest.mark.parametrize('replacement', ['Plasma (Peripheral Venous Blood)',
                                       'Whole Blood (Peripheral Venous Blood) renamed'])
def test_missing_blood_output_cannot_use_plasma_or_a_near_match(replacement):
    scenario, candidates = context()
    candidate = candidates[0]; prov = candidate['catalog_provenance']
    raw = native_fixture()
    for sim in raw['Simulations']:
        sim['OutputSelections'] = [p.replace('Whole Blood (Peripheral Venous Blood)', replacement)
                                  for p in sim.get('OutputSelections', [])]
    options, rejected = reference_source_options(raw, executor.scenario_contract(scenario),
        source_sha256=prov['source_sha256'], reference_allowed=True)
    assert not options
    assert any(r['reason'] == 'EXPLICIT_TOTAL_PARENT_WHOLE_BLOOD_OUTPUT_REQUIRED' for r in rejected)


def synthetic_prediction(matrix='whole_blood'):
    curve = {'time_min': [0., 60., 120.], 'concentration': [0., 2., 0.],
             'time_unit': 'min', 'unit': 'umol/l'}
    path = 'Organism|PeripheralVenousBlood|Drug|Whole Blood (Peripheral Venous Blood)'
    plan = {'candidate_id': 'c', 'scenario_id': 'q', 'output_path': path,
            'molecular_weight_g_per_mol': 100., 'compatibility': {},
            'contract': {'matrix': matrix, 'sampling_times_min': [0., 30., 60.]}}
    return curve, plan


def test_whole_blood_molar_conversion_has_no_plasma_scaling_factor():
    curve, plan = synthetic_prediction()
    result = executor.prediction_from_curve(curve, plan)
    assert result['matrix'] == 'whole_blood'
    assert result['sampled_concentration_ng_ml'] == [0., 100., 200.]
    assert result['endpoints']['AUC_fixed_sampling_window']['value'] == 100.


def test_prediction_cannot_relabel_plasma_as_whole_blood():
    curve, plan = synthetic_prediction()
    plan['output_path'] = plan['output_path'].replace('Whole Blood (', 'Plasma (')
    with pytest.raises(ValueError, match='exact source-native whole-blood observer'):
        executor.prediction_from_curve(curve, plan)


def test_prediction_cannot_relabel_whole_blood_as_plasma():
    curve, plan = synthetic_prediction('plasma')
    with pytest.raises(ValueError, match='cannot be relabelled'):
        executor.prediction_from_curve(curve, plan)


@pytest.mark.parametrize('observed_matrix,predicted_matrix,accepted', [
    ('whole_blood', 'whole_blood', True), ('plasma', 'plasma', True),
    ('whole_blood', 'plasma', False), ('plasma', 'whole_blood', False),
    ('serum', 'plasma', False)])
def test_scalar_scoring_requires_exact_observation_prediction_matrix(observed_matrix, predicted_matrix, accepted):
    scenario = {'normalized_design': {'matrix': observed_matrix, 'analyte': 'parent',
                                      'endpoint_contracts': [{'endpoint': 'Cmax'}]}}
    prediction = {'matrix': predicted_matrix, 'analyte': 'parent', 'binding': 'total',
                  'endpoints': {'sampled_Cmax': {'value': 20., 'unit': 'ng/mL', 'window_h': [0, 24]}}}
    row = scoring.scalar_comparison({'endpoint': 'Cmax', 'value': 20., 'unit': 'ng/mL'}, scenario, prediction)
    assert (row['state'] == 'valid') is accepted
    if accepted:
        assert row['fold_error'] == 1 and row['scientific_status'] == 'unknown'


def test_curve_schema_without_declared_blood_matrix_cannot_be_scored_as_blood():
    curve, plan = synthetic_prediction()
    prediction = executor.prediction_from_curve(curve, plan)
    doc = {'schema_version': 'pbpk-independent-aggregate-curves-sealed-v1',
           'case_id': 'synthetic', 'compound': 'Drug', 'concentration_unit': 'ng/mL',
           'time_unit': 'h', 'records': []}
    result = scoring.curve_comparison(doc, prediction, scenario={
        'compound': 'Drug', 'curve_views': [{'protocol_fields': {'case_id': 'synthetic'}}]})
    assert result[0]['state'] == 'unavailable'
