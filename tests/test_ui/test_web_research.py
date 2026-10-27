"""Public UI contracts preserve study scope, finite metrics and bounded jobs."""
import json
from pathlib import Path
import time

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from pbpk_kb.recommendation.schemas import RecommendationRequest
from pbpk_kb.web import recommendations as rec, research


def test_completed_results_keep_denominators_and_missing_endpoints():
    summary = research.summary()
    assert summary['counts']['candidates'] == 280
    assert summary['counts']['document_requests'] == 145
    assert len(summary['endpoint_families']) == 11
    rows = research.endpoints(method_id='pbpk_match_light', family='Tmax')['items']
    assert rows and all(row['method_id'] == 'pbpk_match_light' and row['family'] == 'Tmax' for row in rows)
    assert any(row['mae_h'] is not None for row in rows)
    unknown = research.endpoints(method_id='', family='Vss')['items']
    assert unknown and all(row['gmfe'] is None for row in unknown)
    json.dumps(summary, allow_nan=False)


def test_curve_case_preserves_saved_values_and_is_not_new_simulation():
    result = research.curves('atorvastatin')
    source = research.read_json(research.CURVES)['cases'][0]
    assert result['observation']['time_h'] == source['observation']['time_h']
    assert result['curves'][0]['concentration_ng_ml'] == source['curves'][0]['concentration_ng_ml']
    assert result['observation']['source_url'].startswith('https://')
    assert result['curves'][0]['model_version_id'] == result['curves'][0]['candidate_id']
    assert len(result['rankings']) == 14
    with pytest.raises(HTTPException):
        research.curves('midazolam')


@pytest.mark.parametrize('key', ['../../runtime/deployment/app.env', '/etc/passwd', 'not-a-file'])
def test_download_is_an_exact_allowlist(key):
    with pytest.raises(HTTPException) as error:
        research.download(key)
    assert error.value.status_code == 404


def test_new_request_conditions_are_visible_without_outcome_inputs():
    request = RecommendationRequest(compound='MIDAZOLAM', route='iv', dose=5, dose_unit='mg',
        frequency='q12h', infusion_duration='30 min', required_enzymes=['CYP3A4'],
        population='healthy adults', prediction_endpoints=['Cmax'])
    query, design = rec.input_query(request)
    assert query['compounds'] == ['midazolam']
    assert query['features']['routes'] == ['intravenous']
    assert query['features']['enzymes'] == ['CYP3A4']
    assert design['regimen']['dosing_interval_h'] == 12
    assert design['regimen']['infusion_duration_h'] == .5
    assert '5.0' in query['text'] and 'healthy adults' in query['text']
    assert query['query_id'] not in query['text']
    with pytest.raises(ValidationError):
        RecommendationRequest(compound='midazolam', target_observed_cmax=4)


def test_fold_routing_never_admits_a_seen_target_compound():
    records = [{'compounds': ['a']}, {'compounds': ['b']}, {'compounds': ['a']}]
    folds = [{'outer_test': [0, 2]}, {'outer_test': [1]}]
    assert rec.select_folds(['a'], records, folds) == [0]
    assert rec.select_folds(['new'], records, folds) == [0, 1]
    with pytest.raises(ValueError):
        rec.select_folds(['a', 'b'], records, folds)


def test_active_duplicate_returns_current_job_and_other_request_is_blocked(tmp_path, monkeypatch):
    # All ephemeral test files remain under the project via pytest --basetemp.
    monkeypatch.setattr(rec, 'RUN', tmp_path)
    request = RecommendationRequest(compound='midazolam')
    payload = rec.valid_payload(request)
    digest = rec.hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    job_id = 'a' * 32
    directory = tmp_path / job_id
    directory.mkdir()
    rec.write(directory / 'status.json', {'job_id': job_id, 'status': 'running', 'stage': 'ranking',
        'message': 'Running', 'created_epoch': time.time(), 'request_sha256': digest,
        'pid': 123, 'start_ticks': '456'})
    rec.write(tmp_path / 'current.json', {'job_id': job_id})
    monkeypatch.setattr(rec, 'process_ticks', lambda pid: '456')
    response = rec.recommend(request)
    assert response.status_code == 202
    assert json.loads(response.body)['job_id'] == job_id
    with pytest.raises(HTTPException) as error:
        rec.recommend(RecommendationRequest(compound='caffeine'))
    assert error.value.status_code == 409
    assert error.value.detail['job_id'] == job_id


def test_dead_job_does_not_block_future_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, 'process_ticks', lambda pid: None)
    rec.write(tmp_path / 'status.json', {'job_id': 'b' * 32, 'status': 'running', 'stage': 'encoding',
        'created_epoch': time.time() - 20, 'pid': 123, 'start_ticks': '1'})
    assert rec.state(tmp_path)['status'] == 'failed'


@pytest.mark.parametrize('job_id', ['../app.env', '/etc/passwd', 'x' * 32])
def test_job_lookup_cannot_escape_project(job_id):
    with pytest.raises(HTTPException):
        rec.job_directory(job_id)


def _display_row(candidate, score, raw_rank, layer=None, eligible=True, blockers=None):
    return {'candidate_id': candidate, 'name': candidate, 'score': score, 'rank': raw_rank,
            'raw_rank': raw_rank, 'policy_eligible': eligible,
            'gate': {'candidate_layer': layer, 'hard_blockers': blockers or []}}


def test_display_uses_qualification_before_raw_research_score_and_keeps_all_scores():
    rows = [_display_row('pregnancy', 9, 1, 'blocked', blockers=[{'reason_code': 'PARSER_FAILED'}]),
            _display_row('unassessed', 8, 2),
            _display_row('requires-validation', 7, 3, 'same_compound_qualification_candidate'),
            _display_row('can-adapt', 3, 4, 'controlled_adaptation_candidate'),
            _display_row('passed-scope', 2, 5, 'validated_reuse'),
            _display_row('mechanism-reference', 1, 6, 'mechanistic_reference', eligible=False)]
    original = json.loads(json.dumps(rows))
    result = rec.display_result({'ranking': rows, 'top_candidates': rows[:3]})
    assert [r['candidate_id'] for r in result['top_candidates']] == ['passed-scope', 'can-adapt', 'requires-validation']
    assert [r['display_rank'] for r in result['top_candidates']] == [1, 2, 3]
    assert [r['raw_rank'] for r in result['top_candidates']] == [5, 4, 3]
    assert [r['candidate_id'] for r in result['reference_candidates']] == ['mechanism-reference']
    assert result['blocked_candidates'][0]['reason_codes'] == ['PARSER_FAILED']
    assert result['unassessed_count'] == 1
    assert result['ranking'] == rows == original
    assert rec.display_result(result) == result


def test_unchecked_or_inconsistent_gate_cannot_be_direct_reuse():
    rows = [_display_row('no-gate', 10, 1),
            _display_row('conflicting', 8, 2, 'validated_reuse', blockers=[{'reason_code': 'LICENSE_RESTRICTED'}]),
            _display_row('wrong-identity', 6, 3, 'same_compound_qualification_candidate', eligible=False)]
    result = rec.display_result({'ranking': rows})
    assert result['top_candidates'] == []
    assert result['status'] == 'no_cleared_candidate'
    assert result['blocked_candidates'][0]['direct_reuse_allowed'] is False
    assert result['unassessed_count'] == 2


def test_display_ties_stay_tied_inside_the_qualified_layer():
    rows = [_display_row(str(i), 20 - i, i + 1, 'same_compound_qualification_candidate') for i in range(9)]
    rows.extend([_display_row('tie-a', 1, 10, 'same_compound_qualification_candidate'),
                 _display_row('tie-b', 1, 10, 'same_compound_qualification_candidate')])
    result = rec.display_result({'ranking': rows})
    assert len(result['top_candidates']) == 11
    assert result['top_candidates'][-1]['display_rank'] == 10
    assert result['top_candidates'][-1]['display_rank_end'] == 11
    assert all(not r['direct_reuse_allowed'] for r in result['top_candidates'])
