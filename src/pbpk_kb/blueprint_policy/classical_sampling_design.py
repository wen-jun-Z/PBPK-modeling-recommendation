"""Pinned, author-implemented PyApprox c-optimal sequential sampling baseline.

The caller supplies local concentration Jacobians and an endpoint gradient in
the SAME four parameter coordinates. Visible Jacobians must already be noise
whitened. Candidate Jacobians must also be whitened unless an explicit candidate
noise standard deviation is supplied. ``regularization`` contains four diagonal
Fisher precisions, not their square roots and not optimization penalty weights.

This adapter does not estimate gradients, fit observations, or evaluate outcomes.
The classical local linear criterion is not a calibrated clinical uncertainty.
"""

from functools import lru_cache
import hashlib
from importlib import import_module
from pathlib import Path
import sys

import numpy as np


PYAPPROX_VERSION = "2.0.0"
PYAPPROX_COMMIT = "0d9acd97c61ba11b478f76059d01031eed6321c5"
PYAPPROX_WHEEL_SHA256 = "14556913837aeb32344b88b3395ef3d0c81a0a8028dcc4a70078958789488149"
PYAPPROX_WHEEL_NAME = "pyapprox-2.0.0-py3-none-any.whl"
PYAPPROX_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/c3/b8/"
    "1ebd43ea75cc96dca43dd2fe09dedd81eafae487d7995b85650a073d6901/"
    + PYAPPROX_WHEEL_NAME
)
_DEPENDENCY_DIR = (
    Path(__file__).resolve().parents[3] / "runtime" / "blueprint_sampling_dependencies"
)
_TIE_ATOL = 1e-12


def source_files():
    """Return the absolute pinned wheel and license paths for run provenance."""
    return (
        _DEPENDENCY_DIR / PYAPPROX_WHEEL_NAME,
        _DEPENDENCY_DIR / "PyApprox-LICENSE.txt",
    )


def source_version():
    """Return reproducible upstream provenance without inspecting observations."""
    return {
        "package": "PyApprox",
        "version": PYAPPROX_VERSION,
        "commit": PYAPPROX_COMMIT,
        "release_tag": "v2.0.0",
        "license": "MIT",
        "repository": "https://github.com/sandialabs/pyapprox",
        "wheel_sha256": PYAPPROX_WHEEL_SHA256,
        "wheel_url": PYAPPROX_WHEEL_URL,
        "author_criterion": "pyapprox.expdesign.local.criteria.c_optimal.COptimalCriterion",
        "author_design_matrices": (
            "pyapprox.expdesign.local.design_matrices.least_squares.LeastSquaresDesignMatrices"
        ),
    }


@lru_cache(maxsize=1)
def _author_implementation():
    """Load the unmodified, checksum-pinned wheel, with no package installation."""
    wheel = _DEPENDENCY_DIR / PYAPPROX_WHEEL_NAME
    if not wheel.is_file():
        raise RuntimeError(f"Pinned PyApprox dependency is missing: {wheel}")
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != PYAPPROX_WHEEL_SHA256:
        raise RuntimeError("Pinned PyApprox wheel SHA-256 mismatch; refusing substituted code")
    wheel_path = str(wheel)
    loaded = sys.modules.get("pyapprox")
    if loaded is not None and not str(getattr(loaded, "__file__", "")).startswith(wheel_path + "/"):
        raise RuntimeError("A different PyApprox installation is loaded; use a fresh project process")
    sys.path.insert(0, wheel_path)
    try:
        criterion = import_module("pyapprox.expdesign.local.criteria.c_optimal").COptimalCriterion
        matrices = import_module(
            "pyapprox.expdesign.local.design_matrices.least_squares"
        ).LeastSquaresDesignMatrices
        backend = import_module("pyapprox.util.backends.numpy").NumpyBkd
        package = import_module("pyapprox")
        if package.__version__ != PYAPPROX_VERSION:
            raise RuntimeError("Pinned PyApprox version mismatch")
    finally:
        sys.path.remove(wheel_path)
    return criterion, matrices, backend


def _finite_array(value, name):
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw) or raw.dtype.kind == "b":
            raise ValueError
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real numbers") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real numbers")
    return array


def c_optimal_scores(
    visible_jacobian,
    candidate_jacobian,
    goal_gradient,
    regularization,
    noise_sd_candidate=None,
):
    """Score adding ONE candidate to a fixed visible design using author code.

    Parameters
    ----------
    visible_jacobian : array, shape (n, 4)
        Already whitened derivatives of visible observations; n may be zero.
    candidate_jacobian : array, shape (m, 4)
        Candidate derivatives, m >= 1. Already whitened when noise is omitted.
    goal_gradient : array, shape (4,)
        Local endpoint derivative in the same coordinates as both Jacobians.
    regularization : array, shape (4,)
        Nonnegative, predeclared diagonal Fisher precision. This is engineering
        stabilization; it does not assert a physiological or clinical prior.
    noise_sd_candidate : positive scalar or array, shape (m,), optional
        Divides candidate rows ONLY. Do not supply when these are whitened.

    Returns
    -------
    dict
        JSON-compatible ``scores`` (smaller is better), ``selected_index`` and
        source/diagnostic information. Ties within 1e-12 select the first row;
        callers requiring earlier-time ties must order candidates by time.

    For candidate j, stack visible rows, the one candidate row, and four
    diagonal sqrt(precision) rows. PyApprox receives unit design weights and
    explicit unit noise, so M0 = M1 = F = J.T J + diag(precision). Its actual
    adjoint implementation evaluates c.T M1^-1 M0 M1^-1 c = c.T F^-1 c.
    Weights are NOT normalized: normalization would change the variance scale.
    Singular designs fail explicitly; no pseudoinverse silently rewards them.
    """
    visible = _finite_array(visible_jacobian, "visible_jacobian")
    candidates = _finite_array(candidate_jacobian, "candidate_jacobian")
    gradient = _finite_array(goal_gradient, "goal_gradient")
    precision = _finite_array(regularization, "regularization")
    if visible.ndim != 2 or visible.shape[1] != 4:
        raise ValueError("visible_jacobian must have shape (n, 4)")
    if candidates.ndim != 2 or candidates.shape[1] != 4 or candidates.shape[0] < 1:
        raise ValueError("candidate_jacobian must have shape (m, 4), m >= 1")
    if gradient.shape != (4,):
        raise ValueError("goal_gradient must have shape (4,)")
    if precision.shape != (4,) or np.any(precision < 0):
        raise ValueError("regularization must be four nonnegative diagonal precisions")
    m = candidates.shape[0]
    if noise_sd_candidate is not None:
        noise = _finite_array(noise_sd_candidate, "noise_sd_candidate")
        if noise.ndim == 0:
            noise = np.full(m, float(noise))
        if noise.shape != (m,) or np.any(noise <= 0):
            raise ValueError("noise_sd_candidate must be positive scalar or shape (m,)")
        candidates = candidates / noise[:, None]
        if not np.all(np.isfinite(candidates)):
            raise ValueError("candidate whitening overflowed")
    criterion_type, matrix_type, backend_type = _author_implementation()
    backend = backend_type()
    ridge_rows = np.diag(np.sqrt(precision))
    scores, condition_numbers = [], []
    for index, row in enumerate(candidates):
        rows = np.vstack((visible, row[None, :], ridge_rows))
        fisher = rows.T @ rows
        if not np.all(np.isfinite(fisher)):
            raise ValueError(f"Candidate {index} Fisher matrix overflowed")
        try:
            np.linalg.cholesky(fisher)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Candidate {index} Fisher matrix is not positive definite") from exc
        matrices = matrix_type(rows, backend, noise_mult=np.ones(len(rows)))
        criterion = criterion_type(matrices, gradient, backend)
        try:
            score = float(np.asarray(criterion(np.ones((len(rows), 1)))).item())
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Candidate {index} official c-opt solve failed") from exc
        if not np.isfinite(score) or score < 0:
            raise ValueError(f"Candidate {index} official c-opt score is invalid: {score}")
        scores.append(score)
        condition_numbers.append(float(np.linalg.cond(fisher)))
    best = min(scores)
    selected = next(index for index, score in enumerate(scores) if score - best <= _TIE_ATOL)
    return {
        "criterion": "classical_local_c_optimal",
        "scores": scores,
        "selected_index": selected,
        "selected_score": scores[selected],
        "score_direction": "minimize",
        "tie_absolute_tolerance": _TIE_ATOL,
        "visible_count": int(visible.shape[0]),
        "candidate_count": int(m),
        "parameter_count": 4,
        "regularization_precision": precision.tolist(),
        "candidate_whitening": "explicit_noise_sd" if noise_sd_candidate is not None else "provided_by_caller",
        "fisher_condition_numbers": condition_numbers,
        "source": source_version(),
        "assumptions": [
            "Local linearization at a predeclared parameter estimate; no target observations are used.",
            "Visible and candidate Jacobians and goal gradient share parameter coordinates.",
            "Unit design weights count each visible measurement and one candidate exactly once.",
            "Diagonal precision is engineering regularization, not a calibrated clinical prior.",
            "The score measures local model uncertainty, not structural error or clinical prediction coverage.",
        ],
    }
