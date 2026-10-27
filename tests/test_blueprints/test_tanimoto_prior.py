"""Numerical and input contracts for the standalone molecular kernel module."""
from io import BytesIO

import joblib
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pbpk_kb.blueprint_learning.tanimoto_prior import (
    MORGAN_BITS, TanimotoKernelRidge, morgan_binary_features, tanimoto_kernel)


def test_fixed_binary_morgan_features_and_invalid_molecules():
    x = morgan_binary_features(["CCO", "OCC", "CCN"])
    assert x.shape == (3, MORGAN_BITS) and x.dtype == np.uint8
    assert set(np.unique(x)) <= {0, 1}
    assert np.array_equal(x[0], x[1])
    assert not np.array_equal(x[0], x[2])
    for invalid in ([], "CCO", [""], ["not_a_smiles"], [None]):
        with pytest.raises(ValueError):
            morgan_binary_features(invalid)


def test_tanimoto_matches_manual_values_and_avoids_uint8_overflow():
    x = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    expected = np.array([[1, 1/3, 0], [1/3, 1, 1/2], [0, 1/2, 1]])
    assert np.allclose(tanimoto_kernel(x), expected)
    assert np.allclose(tanimoto_kernel(x[:1], x[1:]), expected[:1, 1:])
    assert tanimoto_kernel(np.ones((2, 2048), dtype=np.uint8)).tolist() == [[1., 1.], [1., 1.]]


@pytest.mark.parametrize("invalid", [[], [1, 0], [[0, 0]], [[1, 2]], [[1, np.nan]], [[1, np.inf]]])
def test_invalid_binary_features_are_rejected(invalid):
    with pytest.raises(ValueError):
        tanimoto_kernel(invalid)


def test_centered_ridge_solution_precomputed_reuse_and_batch_prediction():
    x = morgan_binary_features(["CCO", "CCN", "CCC", "CCCO"])
    y = np.array([-.3, .2, 1.2, .8])
    kernel = tanimoto_kernel(x)
    before = kernel.copy()
    a = TanimotoKernelRidge(alpha=.3, batch_size=1).fit(x, y, train_kernel=kernel)
    expected = kernel @ np.linalg.solve(kernel + .3*np.eye(4), y-y.mean()) + y.mean()
    assert np.allclose(a.predict(x), expected, rtol=0, atol=1e-12)
    assert np.array_equal(kernel, before)
    assert np.allclose(a.predict(x, cross_kernel=kernel), expected, rtol=0, atol=1e-12)
    b = TanimotoKernelRidge(alpha=.3, batch_size=99).fit(x, y+10.)
    assert np.allclose(b.predict(x), expected+10., rtol=0, atol=1e-12)
    c = TanimotoKernelRidge(alpha=2.).fit(x, y, train_kernel=kernel)
    assert not np.allclose(a.predict(x), c.predict(x))
    stream = BytesIO(); joblib.dump(a, stream); stream.seek(0)
    restored = joblib.load(stream)
    assert np.array_equal(restored.predict(x), a.predict(x))


def test_inputs_are_copied_and_constant_targets_remain_constant():
    x = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    model = TanimotoKernelRidge().fit(x, [3., 3.])
    x[0] = [0, 1]
    assert model.X_fit_[0].tolist() == [1, 0]
    assert model.predict(x).tolist() == [3., 3.]


def test_regression_parameter_shape_and_precomputed_kernel_checks():
    x = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    for alpha in (0, -1, np.nan, np.inf, True):
        with pytest.raises(ValueError):
            TanimotoKernelRidge(alpha=alpha).fit(x, [0., 1.])
    for batch in (0, 1.5, True):
        with pytest.raises(ValueError):
            TanimotoKernelRidge(batch_size=batch).fit(x, [0., 1.])
    for y in ([1.], [[1.], [2.]], [1., np.nan]):
        with pytest.raises(ValueError):
            TanimotoKernelRidge().fit(x, y)
    for k in (np.ones((1, 1)), [[1, .2], [.5, 1]], [[1, 2], [2, 1]], [[.5, .2], [.2, 1]]):
        with pytest.raises(ValueError):
            TanimotoKernelRidge().fit(x, [0., 1.], train_kernel=k)
    with pytest.raises(NotFittedError):
        TanimotoKernelRidge().predict(x)
    fitted = TanimotoKernelRidge().fit(x, [0., 1.])
    with pytest.raises(ValueError):
        fitted.predict([[1, 0, 1]])
    with pytest.raises(ValueError):
        fitted.predict(x, cross_kernel=[[1., np.nan], [0., 1.]])
