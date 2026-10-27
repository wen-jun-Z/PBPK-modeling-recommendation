"""Isolation and decision checks for the study-jackknife sampling guard."""
import copy
import inspect
import json

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import learned_sampling_design as ridge
from pbpk_kb.blueprint_policy import stable_sampling_design as m


def cohort(slopes=(1., 1., 1.), scales=None):
    scales = scales or [1.] * len(slopes)
    x, y, ds, drugs, clusters = [], [], [], [], []
    for i, (slope, scale) in enumerate(zip(slopes, scales)):
        for j in range(3):
            x.append([j * scale] + [0.] * 7)
            y.append(j * slope)
            ds.append(f'design{i}'); drugs.append(f'drug{i}'); clusters.append(f'study{i}')
    return np.array(x), np.array(y), ds, drugs, clusters


def test_shared_study_exclusion_precedes_values_and_applies_to_all_models():
    x, y, ds, drugs, clusters = cohort((1., 1., 1., 1.))
    clusters[:6] = ['study0'] * 6
    a = m.train(x, y, ds, drugs, clusters, exclude_cluster='study0')
    x[:6] = np.nan; y[:6] = np.inf
    b = m.train(x, y, ds, drugs, clusters, exclude_cluster='study0')
    assert a == b
    for member in [a['training_membership']] + [j['training_membership'] for j in a['jackknife_models']]:
        assert not set(member['drug_ids']) & {'drug0', 'drug1'}
        assert 'study0' not in member['cluster_ids']
        assert set(member['row_indices']).isdisjoint(range(6))


def test_fold_local_normalization_and_fixed_alpha():
    x, y, ds, drugs, clusters = cohort(scales=[1., 4., 100.])
    bundle = m.train(x, y, ds, drugs, clusters)
    fold = next(item for item in bundle['jackknife_models'] if item['excluded_cluster'] == 'study2')
    manual = ridge.fit_ranker(x[:6], y[:6], ds[:6], drugs[:6], alpha=1.)
    assert fold['model'] == manual
    assert fold['model']['scale'][0] != bundle['full_model']['scale'][0]
    assert bundle['alpha'] == 1.
    for item in bundle['jackknife_models']:
        assert item['excluded_cluster'] not in item['training_membership']['cluster_ids']


def test_adopts_when_every_study_deletion_prefers_proposal():
    bundle = m.train(*cohort())
    result = m.choose(bundle, [[0.] * 8, [2.] + [0.] * 7], [1., 3.], 0)
    assert result['adopted'] and result['selected_index'] == result['proposed_index'] == 1
    assert result['reason'] == 'unanimous_positive_margin'
    assert all(item['margin'] > 1e-12 for item in result['margins'])


def test_disagreeing_study_deletion_keeps_time_rule():
    bundle = m.train(*cohort((10., -1., -1.)))
    result = m.choose(bundle, [[0.] * 8, [2.] + [0.] * 7], [1., 3.], 0)
    assert result['proposed_index'] == 1 and result['selected_index'] == 0
    assert not result['adopted']
    assert any(item['margin'] < 0 for item in result['margins'])


def test_tie_uses_earlier_proposal_but_still_keeps_time_rule():
    bundle = m.train(*cohort((0., 0., 0.)))
    result = m.choose(bundle, [[0.] * 8, [2.] + [0.] * 7], [3., 1.], 0)
    assert result['proposed_index'] == 1  # earlier candidate among tied scores
    assert result['selected_index'] == 0  # zero margin is not evidence to switch
    assert not result['adopted']


def test_same_action_is_explicit_no_switch():
    result = m.choose(m.train(*cohort()), [[0.] * 8, [2.] + [0.] * 7], [1., 3.], 1)
    assert result['selected_index'] == 1 and not result['adopted']
    assert result['reason'] == 'proposal_is_time_rule'


def test_no_target_outcome_interface_and_new_labels_cannot_affect_selection():
    bundle = m.train(*cohort()); original = copy.deepcopy(bundle)
    menu, times = [[0.] * 8, [2.] + [0.] * 7], [1., 3.]
    target_labels = [1., 1e10]
    result = m.choose(bundle, menu, times, 0)
    target_labels[:] = [1e10, 1.]
    assert m.choose(bundle, menu, times, 0) == result
    assert bundle == original
    assert list(inspect.signature(m.choose).parameters) == ['bundle', 'X', 'times', 'time_index']
    with pytest.raises(TypeError):
        m.choose(bundle, menu, times, 0, target_labels=target_labels)


def test_labels_are_grouped_only_ids_do_not_enter_scores():
    x, y, ds, drugs, clusters = cohort()
    a = m.train(x, y, ds, drugs, clusters)
    b = m.train(x, y, [v+'renamed' for v in ds], [v+'renamed' for v in drugs], [v+'renamed' for v in clusters])
    assert a['full_model'] == b['full_model']
    json.dumps(a)  # artifacts must not contain NumPy-only values


@pytest.mark.parametrize('kind', ['features', 'clusters', 'designs', 'unknown_exclusion', 'one_cluster'])
def test_invalid_training_contract_rejected(kind):
    x, y, ds, drugs, clusters = cohort()
    kwargs = {}
    if kind == 'features': x = x[:, :7]
    elif kind == 'clusters': clusters[0] = 'conflicting'
    elif kind == 'designs': ds[3] = ds[0]
    elif kind == 'unknown_exclusion': kwargs['exclude_cluster'] = 'absent'
    elif kind == 'one_cluster': clusters[:] = ['same'] * len(clusters)
    with pytest.raises(ValueError): m.train(x, y, ds, drugs, clusters, **kwargs)


@pytest.mark.parametrize('index', [True, -1, 2, .5])
def test_invalid_reference_index_rejected(index):
    with pytest.raises(ValueError):
        m.choose(m.train(*cohort()), [[0.] * 8, [2.] + [0.] * 7], [1., 3.], index)
