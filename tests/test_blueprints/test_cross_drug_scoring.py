"""Scientific scoring contract for zero-/five-point cross-drug predictions."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import score_cross_drug_blueprint as scoring
from report_blueprint_validation import Store

# Reuse the established synthetic clinical-reference fixture, not live data.
spec = importlib.util.spec_from_file_location('_cross_score_fixture', Path(__file__).with_name('test_expanded_tail_scoring.py'))
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)
sample = fixture_module.sample


def entry(sample):
    return {'request': sample.request, 'curve': sample.native, 'endpoints': sample.dense}


def test_real_curve_window_and_missing_terminal_references_stay_separate(sample):
    result = scoring.score_curve_reference(sample.case, entry(sample), Store(sample.root))
    assert result['curve_metrics']['gmfe'] == pytest.approx(2.)
    assert result['endpoints']['auc']['reference'] == pytest.approx(400/math.log(2))
    assert result['endpoints']['tmax']['error'] == 0
    for key in set(scoring.UNITS)-scoring.WINDOW_ENDPOINTS:
        v = result['endpoints'][key]
        assert v['value'] is not None
        assert v['reference'] is v['error'] is None
        assert v['reference_kind'] == 'unavailable'


@pytest.mark.parametrize('mutate', ['curve_unit','negative','identity','human','dosing','schedule','mw','observation_use','out_of_window'])
def test_scoring_rejects_inconsistent_reference_or_prediction(sample, mutate):
    e = entry(sample)
    if mutate == 'curve_unit': e['curve']['unit'] = 'mg/l'
    elif mutate == 'negative': e['curve']['concentration'][1] = -1
    elif mutate == 'identity': e['request']['compound_id'] = 'different'
    elif mutate == 'human': e['request']['human'] = {**e['request']['human'], 'weight_kg': 71.}
    elif mutate == 'dosing': e['request']['dosing'] = {**e['request']['dosing'], 'dose_mg': 20.}
    elif mutate == 'schedule': sample.case['nca_policy']['sampling_time_h'][0] = 0.
    elif mutate == 'mw': e['request']['evidence'][0]['value'] = 101.
    elif mutate == 'observation_use': e['endpoints']['comparison_observations_used'] = True
    else:
        e['curve']['time_min'] = [120.,240.,360.]
        e['curve']['concentration'] = [4.,2.,1.]
    with pytest.raises(ValueError):
        scoring.score_curve_reference(sample.case, e, Store(sample.root))


def test_lock_gate_checks_complete_case_coverage_before_labels():
    cases = [{'case_id': str(i)} for i in range(20)]
    protocol = {'cases': cases, 'methods': list(scoring.METHODS), 'conditions': list(scoring.CONDITIONS)}
    state = {'prediction_bundles': {c['case_id']: {} for c in cases}}
    with pytest.raises(ValueError, match='locked'): scoring.validate_locked(protocol,state)
    state['all_predictions_locked_before_scoring'] = True
    scoring.validate_locked(protocol,state)
    del state['prediction_bundles']['0']
    with pytest.raises(ValueError, match='coverage'): scoring.validate_locked(protocol,state)


def test_summary_keeps_reference_kinds_and_failure_denominators_separate():
    cases = [{'case_id': str(i), 'cluster_id': 'shared' if i < 2 else str(i),
        'drug': str(i), 'cohort': 'original' if i < 2 else 'extension'} for i in range(4)]
    rows = []
    for c in cases:
        for condition in scoring.CONDITIONS:
            for method in scoring.METHODS:
                r = {**c, 'condition': condition, 'method': method, 'curve_metrics': {'gmfe': 2.},
                    'endpoints': {'auc': {'value': 2., 'reference': 1., 'error': 2.,
                        'reference_kind': 'original_reported' if c['cohort']=='original' else 'observation_derived'}}}
                if c['case_id'] == '3' and condition=='zero' and method=='ridge':
                    r.update(curve_metrics={}, endpoints={})
                rows.append(r)
    summary, pairs = scoring.summarize(rows,cases)
    row = next(r for r in summary if (r['cohort'],r['condition'],r['method'])==('all_20','zero','ridge'))
    assert (row['requested'], row['scored'], row['failed_or_unscorable']) == (4,3,1)
    assert set(row['endpoint_errors']) == {'auc__original_reported','auc__observation_derived'}
    assert row['curve_gmfe']['n_clusters'] == 2
    assert pairs['all_20']['learned_over_rf']['zero']['ridge']['complete_pairs'] == 3


def test_json_pointer_escapes_tokens_and_preserves_whole_bundle_digest():
    ref = {'path': 'bundle.json','sha256': 'abc'}
    p = scoring._pointer(ref,'entries','a/b~c','curve')
    assert p == {**ref, 'json_pointer': '/entries/a~1b~0c/curve'}
