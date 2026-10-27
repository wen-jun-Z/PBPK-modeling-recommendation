from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from reviewer_external_validation import document_head, OUT, LIGHT


def test_external_head_reproduces_original_heldout_document_scores():
    """New-request restoration must not silently change the learned ranker."""
    import torch
    torch.set_num_threads(2)
    protocol = json.loads((LIGHT / 'protocol.json').read_text())
    indices = [protocol['folds'][f]['outer_test'][0] for f in (0, 1)]
    with np.load(LIGHT / 'cache/inputs.npz', allow_pickle=False) as cache:
        queries = cache['raw_query_embeddings'][0, indices].astype(np.float32)
        candidates = cache['raw_candidate_embeddings'].astype(np.float32)
    with np.load(ROOT / 'experiments/model_recommendation/pbpk_match_v2/arrays.npz', allow_pickle=False) as data:
        features = data['pair_features_by_variant'][0, indices].astype(np.float32)
    with np.load(LIGHT / 'predictions/seed17.npz', allow_pickle=False) as old:
        expected = old['base_scores'][0, indices]
    for f in (0, 1):
        scores, probabilities = document_head(queries[f], features[f], candidates, LIGHT / f'runs/seed17/fold{f}/outer')
        np.testing.assert_allclose(scores, expected[f], rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(np.argsort(scores), np.argsort(expected[f]))
        np.testing.assert_allclose(probabilities.sum(-1), 1, atol=2e-7)


def test_new_rankings_exclude_all_target_compound_training_records():
    receipt = json.loads((OUT / 'rankings/manifest.json').read_text())
    protocol = json.loads((LIGHT / 'protocol.json').read_text())
    assert receipt['new_training'] is False
    assert receipt['clinical_outcome_inputs'] is False
    assert receipt['components'] is False and receipt['error_correction'] is False
    for row in receipt['fold_assignments']:
        for f in row['folds']:
            excluded = set(row['same_compound_training_records_excluded'])
            assert excluded <= set(protocol['folds'][f]['outer_test'])
            assert not excluded & set(protocol['folds'][f]['outer_train'])
    with np.load(OUT / 'rankings/scores.npz', allow_pickle=False) as a:
        for method in receipt['methods']:
            assert a[method].shape == (6, 280)
            assert np.isfinite(a[method]).all()


def test_new_inputs_keep_numerical_conditions_without_observations():
    scenarios = json.loads((OUT / 'inputs/scenarios.json').read_text())
    queries = json.loads((OUT / 'inputs/queries.json').read_text())
    from rank_recommendation_execution import build_query
    for s, q in zip(scenarios, queries):
        contaminated = dict(s, observations={'Cmax': 999999.}, observed_values={'auc': 123456.})
        assert build_query(contaminated) == q
        alternative = json.loads(json.dumps(s))
        alternative['normalized_design']['dose']['value'] *= 2
        assert build_query(alternative)['text'] != q['text']
    # The original adapter must additionally refuse outcomes in native inputs.
    from recommendation_candidate_executor import scenario_contract, Blocked
    with pytest.raises(Blocked, match='OUTCOMES_FORBIDDEN'):
        scenario_contract(dict(scenarios[0], observations={'Cmax': 1}))


def test_metformin_ambiguous_regimen_is_quarantined_before_scoring():
    policy = json.loads((OUT / 'scoring_protocol.json').read_text())
    assert 'blocked' in policy['metformin']
    assert policy['clinical_results_used_for_rule_selection'] is False
    assert (OUT / 'selection_amendment_01.json').is_file()
