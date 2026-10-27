"""Leakage, group isolation, weighting, numerical and sampling contract checks."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from pbpk_kb.blueprint_policy import learned_sampling_design as m


def data():
    x = np.array([[0., 2.], [1., 4.], [5., 3.], [7., 3.], [9., 9.]])
    return x, np.array([1., 2., 9., 6., 3.]), ['a', 'a', 'b', 'b', 'b'], ['A', 'A', 'B', 'B', 'B']


def test_equal_drug_layout_weights_ignore_candidate_menu_size():
    w = m.row_weights(['a', 'a', 'b', 'c', 'c', 'c'], ['A', 'A', 'A', 'B', 'B', 'B'])
    assert w == pytest.approx([.125, .125, .25, 1/6, 1/6, 1/6])
    assert sum(w) == pytest.approx(1)


def test_weighted_normal_equation_independent_augmentation():
    x, y, ds, drugs = data(); model = m.fit_ranker(x, y, ds, drugs)
    w = np.array([.25, .25, 1/6, 1/6, 1/6])
    xc = x.copy(); yc = y.copy()
    for indices in ([0, 1], [2, 3, 4]):
        xc[indices] -= x[indices].mean(axis=0); yc[indices] -= y[indices].mean()
    scale = np.sqrt(w @ xc**2)
    z = xc / scale
    augmented = np.vstack((np.sqrt(w)[:, None] * z, np.eye(2)))
    target = np.r_[np.sqrt(w) * yc, [0., 0.]]
    coef = np.linalg.lstsq(augmented, target, rcond=None)[0]
    assert model['coef'] == pytest.approx(coef, rel=1e-12)
    assert model['scale'] == pytest.approx(scale)


def test_centering_removes_baseline_and_drug_difficulty():
    x, y, ds, drugs = data(); a = m.fit_ranker(x, y, ds, drugs)
    x[:2] += 100; x[2:] -= 300; y[:2] -= 90; y[2:] += 700
    b = m.fit_ranker(x, y, ds, drugs)
    assert a['coef'] == pytest.approx(b['coef']); assert a['scale'] == pytest.approx(b['scale'])


def test_training_scales_unaffected_by_extreme_holdout_and_no_mutation():
    x, y, ds, drugs = data(); before = x.copy(); model = m.fit_ranker(x, y, ds, drugs)
    frozen = repr(model)
    s = m.predict_scores(model, [[1e12, -1e12], [1., 2.]])
    assert np.isfinite(s).all() and repr(model) == frozen
    np.testing.assert_array_equal(x, before)


def test_constant_feature_and_zero_label_produce_tie_earlier_time():
    a = m.fit_ranker([[1, 2], [1, 2]], [4, 4], ['a', 'a'], ['A', 'A'])
    assert a['coef'] == [0., 0.]
    assert m.choose_index(m.predict_scores(a, [[1, 2], [1, 2]]), [4, 1]) == 1
    assert m.choose_index([1, 1 + 1e-13], [1, 4]) == 0


def test_drug_and_shared_study_are_jointly_held_out():
    rows = [dict(case_id=drug, cluster_id=cluster, design=i, candidate=j)
            for drug, cluster in [('A', 'study1'), ('B', 'study1'), ('C', 'study2'), ('D', 'study3')]
            for i in range(2) for j in range(3)]
    folds = list(m.cluster_folds(rows)); assert len(folds) == 3
    group, train, test = folds[0]
    assert group == 'study1' and len(test) == 12
    assert {rows[i]['case_id'] for i in test} == {'A', 'B'}
    assert not {rows[i]['case_id'] for i in test} & {rows[i]['case_id'] for i in train}


def test_conflicting_drug_clusters_rejected():
    with pytest.raises(ValueError, match='conflicting'):
        list(m.cluster_folds([dict(case_id='A', cluster_id='X'), dict(case_id='A', cluster_id='Y')]))


@pytest.mark.parametrize('field,value', [('x', [[np.nan, 1]]), ('y', [np.nan]), ('alpha', 0)])
def test_invalid_training_rejected(field, value):
    args = dict(x=[[1, 2]], y=[1], design_ids=['a'], drug_ids=['A'], alpha=1)
    args[field] = value
    with pytest.raises(ValueError):
        m.fit_ranker(**args)


def feature_args():
    return dict(times=[1., 3.], visible_times=[.5, 2.], visible_max=10., infusion_end=1.,
        concentrations=[[10., 1.], [5., 2.]], linear_utility=[.2, .4], nonlinear_utility=[.4, .5],
        goal_variance=1., c_optimal_scores=[.8, .2], c_optimal_before=1.)


def test_features_physical_units_and_no_target_values_interface():
    x = m.features(**feature_args()); assert x.shape == (2, 8)
    assert x[:, 0] == pytest.approx([0, np.log(3)])
    assert x[:, -2] == pytest.approx([.2, .8]); assert x[:, -1] == pytest.approx([0, 1])
    with pytest.raises(TypeError):
        m.features(**feature_args(), target_observations=[1, 2])


def test_zero_variance_has_finite_zero_utility_features():
    args = feature_args(); args.update(goal_variance=0., c_optimal_before=0., c_optimal_scores=[0., 0.])
    x = m.features(**args); np.testing.assert_array_equal(x[:, 4:7], np.zeros((2, 3)))


def test_local_derivatives_sort_inputs_and_restore_rows_without_refitting():
    calls = {'c': 0, 'g': 0}
    def predict(x, ts):
        assert np.all(np.diff(ts) > 0)
        calls['c'] += 1
        return np.exp(x[0] + x[1]*ts + x[2]*ts**2 + x[3]*ts**3)
    def goals(x):
        calls['g'] += 1
        return np.exp([x[0] + 2*x[3], x[1], x[2]])
    c, j, g = m.local_derivatives([0, 0, 0, 0], [-1]*4, [1]*4, [2, 3], [.5, 1], predict, goals)
    ts = np.array([2, 3, .5, 1])
    np.testing.assert_allclose(j, np.column_stack((np.ones(4), ts, ts**2, ts**3)), rtol=1e-10)
    np.testing.assert_allclose(g, [[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0]], atol=1e-10)
    assert calls == {'c': 9, 'g': 8}


def test_boundary_one_sided_derivative():
    c, j, g = m.local_derivatives([1, 0, 0, 0], [-1]*4, [1]*4, [1], [2],
                                lambda x,t: np.exp(np.full(len(t), x.sum())), lambda x: np.exp([x.sum()]))
    np.testing.assert_allclose(j, 1, rtol=1e-10); np.testing.assert_allclose(g, 1, rtol=1e-10)


@pytest.fixture(scope='module')
def runner():
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    import run_learned_sampling_design
    return run_learned_sampling_design


def cached_data():
    return {'records': [dict(case_id=drug, design_key=drug+'__early', candidate=cand,
                            endpoints={'cmax': {'error': value}})
                       for drug in ('A', 'B') for cand, value in [('baseline', 2), (drug+'__point_01', 1.2)]]}


def test_training_steward_rejects_heldout_row_and_never_uses_heldout_labels(runner):
    cached = cached_data(); rows = [dict(case_id='A', design_key='A__early', candidate='A__point_01')]
    y1 = runner.labels_for_training(cached, rows, {'A'}, 'cmax')
    for r in cached['records']:
        if r['case_id'] == 'B':
            r['endpoints']['cmax']['error'] = None
    assert runner.labels_for_training(cached, rows, {'A'}, 'cmax') == y1
    with pytest.raises(ValueError, match='Held-out'):
        runner.labels_for_training(cached, rows, {'B'}, 'cmax')


def test_missing_training_label_fails_not_zero_imputed(runner):
    cached = cached_data(); cached['records'][1]['endpoints']['cmax']['error'] = None
    with pytest.raises(ValueError, match='Missing'):
        runner.labels_for_training(cached, [dict(case_id='A', design_key='A__early', candidate='A__point_01')], {'A'}, 'cmax')


def test_report_requires_choice_lock_before_clinical_scoring(runner, monkeypatch):
    monkeypatch.setattr(runner, 'read', lambda p: {'status': 'features', 'protocol': {}})
    monkeypatch.setattr(runner, 'checked', lambda p: {})
    monkeypatch.setattr(runner.Guard, '__call__', lambda *a, **kw: None)
    with pytest.raises(ValueError, match='lock before scoring'):
        runner.report()
