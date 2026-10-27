"""Binary Morgan/Tanimoto kernel regression for a single log-PK parameter.

This module performs no split, label transformation, candidate selection, or
clinical scoring. Callers supply training-only log targets and enforce their
own scaffold split. Molecules are used as supplied, without salt removal or
identity normalization; the shared cohort code owns those decisions.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import LinAlgWarning, solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

VERSION = "binary-morgan-tanimoto-kernel-ridge-1"
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def _binary_features(values, *, expected_features=None):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Expected a finite binary feature matrix") from error
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("Features must be a nonempty two-dimensional matrix")
    if expected_features is not None and array.shape[1] != expected_features:
        raise ValueError("Prediction and training feature dimensions differ")
    if not np.isfinite(array).all() or not np.all((array == 0.) | (array == 1.)):
        raise ValueError("Tanimoto features must contain only finite binary 0/1 values")
    if np.any(array.sum(axis=1) == 0.):
        raise ValueError("Zero fingerprints are unsupported; molecular similarity is undefined")
    return array


def morgan_binary_features(smiles):
    """Return fixed radius-2, 2048-bit binary Morgan features as uint8.

    Chirality and count simulation are disabled. Invalid, empty or zero-bit
    molecules are rejected. The input must already use the intended molecular
    identities; this function does not canonicalize salts or parent structures.
    """
    if isinstance(smiles, (str, bytes)):
        raise ValueError("Pass a sequence of SMILES, not one string")
    identities = list(smiles)
    if not identities:
        raise ValueError("At least one molecular identity is required")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS, includeChirality=False,
        countSimulation=False)
    values = np.empty((len(identities), MORGAN_BITS), dtype=np.uint8)
    for index, identity in enumerate(identities):
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"Invalid SMILES at row {index}")
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(identity)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError(f"Invalid or empty molecule at row {index}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), values[index])
    _binary_features(values)
    return values


def _kernel_unchecked(left, right):
    # The inputs are float64, preventing uint8 overflow for long fingerprints.
    intersection = left @ right.T
    union = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :] - intersection
    if np.any(union <= 0.):
        raise ValueError("Tanimoto union must be positive")
    return intersection / union


def tanimoto_kernel(features, other=None):
    """Compute x·z / (|x| + |z| - x·z) for nonzero binary fingerprints.

    A full training kernel can be computed once and sliced by scaffold-fold
    indices, then reused unchanged for several ridge penalties.
    """
    left = _binary_features(features)
    right = left if other is None else _binary_features(other, expected_features=left.shape[1])
    return _kernel_unchecked(left, right)


def _precomputed_kernel(values, shape, *, square=False):
    kernel = np.asarray(values, dtype=np.float64)
    if kernel.shape != shape or not np.isfinite(kernel).all():
        raise ValueError("Precomputed kernel has invalid shape or nonfinite values")
    if np.any(kernel < -1e-12) or np.any(kernel > 1. + 1e-12):
        raise ValueError("A Tanimoto kernel must be within [0,1]")
    if square and (not np.allclose(kernel, kernel.T, rtol=0., atol=1e-12)
                   or not np.allclose(np.diag(kernel), 1., rtol=0., atol=1e-12)):
        raise ValueError("Training kernel must be symmetric with unit diagonal")
    return kernel


class TanimotoKernelRidge(RegressorMixin, BaseEstimator):
    """Ridge regression on log targets centered by the training target mean.

    ``fit(X, y, train_kernel=None)`` and ``predict(X, cross_kernel=None)`` accept
    caller-computed Tanimoto matrices. Their dimensions/ranges are checked, but
    the caller must ensure matching row order and fingerprints; we do not
    recompute supplied kernels. No mean/scale of held-out labels is used.

    The returned values remain in the supplied log units. Exponentiation and
    physical PK interpretation belong to the caller. Instances are joblib
    serializable and expose ``get_params``/``set_params`` through BaseEstimator.
    """

    def __init__(self, alpha=1., batch_size=256):
        self.alpha = alpha
        self.batch_size = batch_size

    def fit(self, X, y, *, train_kernel=None):
        if isinstance(self.alpha, bool) or not np.isscalar(self.alpha):
            raise ValueError("alpha must be a finite positive scalar")
        alpha = float(self.alpha)
        if not math.isfinite(alpha) or alpha <= 0.:
            raise ValueError("alpha must be finite and strictly positive")
        self._validate_batch_size()
        features = _binary_features(X)
        targets = np.asarray(y, dtype=np.float64)
        if targets.shape != (features.shape[0],) or not np.isfinite(targets).all():
            raise ValueError("Targets must be a finite vector matching training rows")
        mean = float(targets.mean())
        centered = targets - mean
        if not math.isfinite(mean) or not np.isfinite(centered).all():
            raise ValueError("Centered log targets must be finite")
        kernel = (_kernel_unchecked(features, features) if train_kernel is None else
                  _precomputed_kernel(train_kernel, (len(features), len(features)), square=True))
        regularized = np.array(kernel, dtype=np.float64, order="F", copy=True)
        regularized.flat[::len(features) + 1] += alpha
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            coefficients = solve(regularized, centered, assume_a="pos",
                                 overwrite_a=True, check_finite=True)
        if not np.isfinite(coefficients).all():
            raise ValueError("Kernel solve produced nonfinite coefficients")
        self.X_fit_ = features.astype(np.uint8, copy=True)
        self.n_features_in_ = features.shape[1]
        self.y_mean_ = mean
        self.dual_coef_ = coefficients
        self.alpha_ = alpha
        return self

    def _validate_batch_size(self):
        if (isinstance(self.batch_size, bool)
                or not isinstance(self.batch_size, (int, np.integer)) or self.batch_size < 1):
            raise ValueError("batch_size must be a positive integer")

    def predict(self, X, *, cross_kernel=None):
        check_is_fitted(self, ("X_fit_", "dual_coef_", "y_mean_", "n_features_in_"))
        self._validate_batch_size()
        features = _binary_features(X, expected_features=self.n_features_in_)
        train = self.X_fit_.astype(np.float64)
        supplied = None if cross_kernel is None else _precomputed_kernel(
            cross_kernel, (len(features), len(train)))
        predictions = np.empty(len(features), dtype=np.float64)
        for start in range(0, len(features), self.batch_size):
            stop = min(start + self.batch_size, len(features))
            kernel = (_kernel_unchecked(features[start:stop], train) if supplied is None
                      else supplied[start:stop])
            predictions[start:stop] = kernel @ self.dual_coef_ + self.y_mean_
        if not np.isfinite(predictions).all():
            raise ValueError("Kernel regression produced nonfinite predictions")
        return predictions
