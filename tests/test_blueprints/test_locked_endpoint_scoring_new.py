"""Meaningful isolation and numerical checks using only synthetic endpoints."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('locked_endpoint_scorer', ROOT / 'scripts/score_blueprint_locked_endpoints.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
TAIL_POLICY = {'fit_window_start_h': 8, 'fit_window_end_h': 10, 'minimum_points': 5,
    'require_negative_log_slope': True, 'maximum_tail_fraction_for_primary': .01,
    'minimum_r_squared_for_primary': .99, 'do_not_clip_or_replace_failed_tail': True}


def curve(sex='MALE'):
    times = list(range(0, 601, 2))
    return {'model_key': 'synthetic', 'sex': sex, 'time_min': times,
        'concentration': [math.exp(-t / 60) for t in times], 'unit': 'µmol/l'}


@pytest.fixture
def locked_inputs():
    with tempfile.TemporaryDirectory(prefix='endpoint-scorer-', dir=ROOT / 'runtime') as directory:
        base = Path(directory)
        def write(name, data):
            p = base / name
            p.write_text(json.dumps(data))
            return {'path': str(p.relative_to(ROOT)), 'sha256': scorer.sha(p.read_bytes())}
        sealed = write('sealed.json', {'case_id': 'synthetic', 'study_id': 'synthetic-source',
            'observed_endpoints': [{'endpoint': name, 'unit': unit, 'aggregation': 'arithmetic_mean', 'value': 100,
                'source_table_id': 'synthetic', 'source_row': i, 'source_column': 1}
                for i, (name, unit) in enumerate([('Cmax', 'ng/mL'), ('AUC_infinity', 'ng*h/mL')])]})
        design = write('design.json', {'case_id': 'synthetic', 'study_id': 'synthetic-source', 'sealed_observation_reference': sealed})
        selections = [{'method': 'M0', 'seed': 1, 'block_id': block, 'model_key': 'synthetic'} for block in ['small', 'large']]
        protocol = write('protocol.json', {'sex_weights': {'male': .75, 'female': .25}, 'case_id': 'synthetic',
            'design_sha256': design['sha256'], 'selections': selections,
            'molecular_weight_g_mol': 100, 'approximation': ['synthetic reference-subject test'], 'tail_policy': TAIL_POLICY})
        pred = write('prediction.json', {'case_id': 'synthetic', 'models': [curve(), curve('FEMALE')],
            'protocol_sha256': protocol['sha256'], 'selections': selections})
        lock = {'schema_version': scorer.SCHEMA, 'model_selection_frozen': True, 'design_sha256': design['sha256'],
            'protocol': protocol, 'predictions': [{'case_id': 'synthetic', **pred}]}
        write('lock.json', lock)
        yield base, lock, write


def test_exponential_tail_and_hour_conversion():
    result = scorer.curve_endpoints(curve(), 100, TAIL_POLICY)
    assert result['Cmax_dense'] == 100
    assert result['AUC_infinity'] == pytest.approx(100, rel=1e-4)
    assert result['terminal_tail']['valid']
    assert result['terminal_tail']['slope_per_h'] == pytest.approx(-1)


def test_same_method_seed_across_blocks_preserved(locked_inputs):
    base, _, _ = locked_inputs
    result = scorer.score_locked(base / 'lock.json', base / 'design.json')
    assert len(result['results']) == 2
    assert {r['selection_id'] for r in result['results']} == {'small__M0_seed_1', 'large__M0_seed_1'}
    assert not result['strict_main_promotion']
    assert all(e['reference_subject_population_approximation'] for r in result['results'] for e in r['endpoints'])


def test_unfrozen_lock_rejected_before_missing_sealed_is_read(locked_inputs):
    base, lock, write = locked_inputs
    (base / 'sealed.json').unlink()
    lock['model_selection_frozen'] = False
    write('lock.json', lock)
    with pytest.raises(ValueError, match='model-selection freeze'):
        scorer.score_locked(base / 'lock.json', base / 'design.json')


def test_prediction_tampering_rejected_before_sealed_is_read(locked_inputs):
    base, _, _ = locked_inputs
    (base / 'sealed.json').unlink()
    (base / 'prediction.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum mismatch'):
        scorer.score_locked(base / 'lock.json', base / 'design.json')


def test_unknown_unit_rejected():
    model = curve()
    model['unit'] = 'ng/mL'
    with pytest.raises(ValueError, match='micromolar'):
        scorer.curve_endpoints(model, 100, TAIL_POLICY)


def test_invalid_terminal_phase_does_not_become_total_auc():
    model = curve()
    model['concentration'] = [1 for _ in model['time_min']]
    result = scorer.curve_endpoints(model, 100, TAIL_POLICY)
    assert result['AUC_infinity'] is None
    assert result['AUC_observed_simulation_window'] == 1000


def test_negative_concentration_is_rejected():
    model = curve()
    model['concentration'][12] = -1
    with pytest.raises(ValueError, match='negative concentration'):
        scorer.curve_endpoints(model, 100, TAIL_POLICY)


def test_tail_fraction_threshold_comes_from_locked_protocol():
    model = curve()
    model['concentration'] = [math.exp(-t / 600) for t in model['time_min']]
    result = scorer.curve_endpoints(model, 100, TAIL_POLICY)
    assert result['AUC_infinity'] is None
    assert result['terminal_tail']['tail_fraction'] > .01


def test_tail_window_cannot_change_after_freeze():
    with pytest.raises(ValueError, match='locked tail policy'):
        scorer.curve_endpoints(curve(), 100, {**TAIL_POLICY, 'fit_window_end_h': 11})


def rewrite_selected_inputs(base, lock, write, selections, models):
    protocol = json.loads((base / 'protocol.json').read_text())
    protocol['selections'] = selections
    lock['protocol'] = write('protocol.json', protocol)
    pred = json.loads((base / 'prediction.json').read_text())
    pred.update(selections=selections, models=models, protocol_sha256=lock['protocol']['sha256'])
    lock['predictions'] = [{'case_id': 'synthetic', **write('prediction.json', pred)}]
    write('lock.json', lock)


def test_failed_selection_kept_alongside_successful_predictions(locked_inputs):
    base, lock, write = locked_inputs
    pred = json.loads((base / 'prediction.json').read_text())
    pred['selections'].append({'method': 'M1', 'seed': 1, 'block_id': 'small', 'model_key': None,
        'upstream_status': 'native_execution_failed'})
    rewrite_selected_inputs(base, lock, write, pred['selections'], pred['models'])
    result = scorer.score_locked(base / 'lock.json', base / 'design.json')
    assert result['method_job_denominator'] == 3
    assert result['jobs_with_frozen_model_prediction'] == 2
    failed = result['results'][-1]
    assert failed['upstream_status'] == 'native_execution_failed'
    assert failed['retained_in_method_job_denominator']
    assert all(e['predicted'] is None and not e['evaluable'] for e in failed['endpoints'])


def test_all_failed_jobs_still_produce_complete_denominator(locked_inputs):
    base, lock, write = locked_inputs
    selections = [{'method': 'M0', 'seed': seed, 'block_id': 'small', 'model_key': None,
        'upstream_status': 'design_failed'} for seed in [1, 2]]
    rewrite_selected_inputs(base, lock, write, selections, [])
    result = scorer.score_locked(base / 'lock.json', base / 'design.json')
    assert result['method_job_denominator'] == result['jobs_without_frozen_model_prediction'] == 2
    assert result['jobs_with_frozen_model_prediction'] == 0


def test_unselected_prediction_models_rejected_before_sealed_read(locked_inputs):
    base, lock, write = locked_inputs
    pred = json.loads((base / 'prediction.json').read_text())
    pred['models'].extend([{**curve(sex), 'model_key': 'unselected'} for sex in ['MALE', 'FEMALE']])
    rewrite_selected_inputs(base, lock, write, pred['selections'], pred['models'])
    (base / 'sealed.json').unlink()
    with pytest.raises(ValueError, match='exactly match'):
        scorer.score_locked(base / 'lock.json', base / 'design.json')
