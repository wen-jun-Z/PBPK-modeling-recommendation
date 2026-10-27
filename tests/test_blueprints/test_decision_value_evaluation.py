from copy import deepcopy
from hashlib import sha256
import json
import math

import pytest

from pbpk_kb.blueprint_policy import decision_value_evaluation as mod


def store(root, name, value):
    data = json.dumps(value).encode()
    (root / name).write_bytes(data)
    return {'path': name, 'sha256': sha256(data).hexdigest()}


@pytest.fixture
def fixture(tmp_path):
    cases, outcomes, rows = [], {}, []
    units = {'clearance': 'l/h', 'vss': 'l', 'terminal_half_life': 'h', 'mrt': 'h'}
    for i in range(3):
        cid = f'case{i}'
        case = {'case_id': cid, 'compound_id': f'compound{i}', 'identity_smiles': f'C{i}',
                'drug_name': f'Drug {i}', 'evaluation_scope': 'aggregated_human_IV_PK_parameters'}
        ev = {k: case[k] for k in ('compound_id', 'identity_smiles', 'evaluation_scope')}
        ev.update(clinical_curve=None, weight_kg=70,
            endpoints={ep: {'value': None if i == 2 else 10., 'unit': unit} for ep, unit in units.items()})
        case['evaluation'] = store(tmp_path, f'eval{i}.json', ev)
        candidates = {name: {'status': 'predicted', 'endpoints': {
            ep: {'value': 10. if name == 'combined' else 30., 'unit': unit, 'status': 'completed'}
            for ep, unit in units.items()}} for name in ('parameter_only', 'combined')}
        outcomes[cid] = store(tmp_path, f'outcome{i}.json', {'case_id': cid, 'methods': candidates})
        cases.append(case)
        for ep in mod.ENDPOINTS:
            for method in mod.METHODS:
                selected = 'parameter_only' if method == 'fixed_native' else 'combined'
                rows.append({'case_id': cid, 'compound_id': case['compound_id'], 'endpoint': ep, 'method': method,
                    'selected_candidate': selected, 'prediction_value': candidates[selected]['endpoints'][ep]['value'],
                    'selected_prediction_sha256': mod.digest(candidates[selected]), 'whole_model_source': outcomes[cid],
                    'expected_log_error': .1, 'risk': .1 if selected == 'combined' else .9,
                    'ranking_score': i / 10, 'status': 'research_prediction'})
    protocol = {'cases': cases, 'candidate_outcomes': outcomes, 'expected_compounds': 3,
                'methods': list(mod.METHODS), 'endpoints': list(mod.ENDPOINTS)}
    locked = {'all_decisions_locked_before_scoring': True, 'records': rows, 'records_sha256': mod.digest(rows)}
    return tmp_path, protocol, locked


def relock(locked):
    locked['records_sha256'] = mod.digest(locked['records'])


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unlocked', 'edited', 'identity', 'model', 'source'])
def test_bad_lock_never_reads_evaluation(fixture, monkeypatch, change):
    root, protocol, locked = fixture
    if change == 'missing': locked['records'].pop(); relock(locked)
    elif change == 'duplicate': locked['records'][-1] = deepcopy(locked['records'][0]); relock(locked)
    elif change == 'unlocked': locked['all_decisions_locked_before_scoring'] = False
    elif change == 'edited': locked['records'][0]['risk'] = .5
    elif change == 'identity': locked['records'][0]['compound_id'] = 'wrong'; relock(locked)
    elif change == 'model': locked['records'][0]['prediction_value'] = 11.; relock(locked)
    elif change == 'source': locked['records'][0]['whole_model_source'] = protocol['candidate_outcomes']['case1']; relock(locked)
    monkeypatch.setattr(mod, '_checked_evaluation', lambda *a: pytest.fail('Evaluation opened before barrier'))
    with pytest.raises(ValueError): mod.evaluate_decisions(protocol, locked, root, repeats=20)


def test_full_result_and_no_reference_denominator(fixture):
    root, protocol, locked = fixture
    result = mod.evaluate_decisions(protocol, locked, root, repeats=30)
    assert result['registered_compounds'] == 3 and result['requests'] == 12
    assert len(result['records']) == 60
    ep = result['endpoint_metrics']['clearance']
    assert ep['reference_n'] == ep['common_scorable_n'] == 2
    fixed = ep['methods']['fixed_capacity']
    assert fixed['prediction_n'] == 3 and fixed['own_scorable']['gmfe'] == 1
    assert fixed['risk']['brier'] == pytest.approx(.01)
    assert fixed['registered_coverage_curve'][-1]['reference_missing_n'] == 1
    assert fixed['matched_coverage_curve'][-1]['n'] == 2
    assert ep['comparisons']['purpose_decision_vs_fixed_capacity']['selection']['gmfe_ratio_ci95'] == [1., 1.]


def test_selected_whole_model_readback(fixture):
    root, protocol, locked = fixture
    locked['records'][0]['selected_prediction_sha256'] = '0' * 64; relock(locked)
    with pytest.raises(ValueError, match='Whole-model'):
        mod.evaluate_decisions(protocol, locked, root)


def test_primary_missing_remains_registered(fixture):
    root, protocol, locked = fixture
    for row in locked['records']:
        if row['case_id'] == 'case0':
            row.update(selected_candidate=None, prediction_value=None, selected_prediction_sha256=None,
                       risk=None, expected_log_error=None, ranking_score=None)
    relock(locked)
    result = mod.evaluate_decisions(protocol, locked, root, repeats=20)
    ep = result['endpoint_metrics']['mrt']
    assert ep['reference_n'] == 2 and ep['common_scorable_n'] == 1
    assert ep['methods']['purpose_decision']['unavailable_n'] == 1


def test_artifact_modification_rejected(fixture):
    root, protocol, locked = fixture
    (root / 'outcome0.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        mod.evaluate_decisions(protocol, locked, root)


def test_risk_flags_and_ece_edge():
    rows = [{'risk': 1., 'within_twofold': True}, {'risk': 0., 'within_twofold': False},
            {'risk': .5, 'within_twofold': False}, {'risk': None, 'within_twofold': True}]
    r = mod._risk(rows, 4)
    assert r['n'] == 3 and r['false_alarm_n'] == r['missed_large_error_n'] == 1
    assert r['brier'] == pytest.approx(.75)
    assert r['unflagged_scorable_fraction_of_registered'] == .25
    assert sum(b['n'] for b in r['calibration_bins']) == 3


def test_ranking_gain_separate_from_selection():
    left = {str(i): {'case_id': str(i), 'ranking_score': 3-i, 'absolute_ln_error': i,
                    'selected_candidate': 'combined'} for i in range(4)}
    right = {c: dict(r, ranking_score=int(c)) for c, r in left.items()}
    selection = mod.paired(left, right, 50, 1)
    ranking = mod.ranking_pair(left, right, .5, 50, 1)
    assert selection['gmfe_ratio_right_over_left'] == 1 and selection['changed_candidate_n'] == 0
    assert ranking['gmfe_ratio_right_over_left'] == pytest.approx(math.exp(-2))
    assert ranking['resamples_reselect_by_frozen_rank']


def test_floating_ties_do_not_become_risk_signal():
    rows = [{'case_id': f'c{i}', 'ranking_score': 1 + i * 1e-14} for i in range(8)]
    a, _ = mod._ranking(rows)
    b, _ = mod._ranking([dict(r, ranking_score=1.) for r in rows])
    assert a == b
