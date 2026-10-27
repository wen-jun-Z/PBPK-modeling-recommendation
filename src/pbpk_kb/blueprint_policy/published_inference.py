"""Published PSS-SVD and DRAM inference, adapted to a frozen PBPK simulator.

The sampler is the unmodified ``pymcmcstat`` package (Miles, JOSS 2019,
10.21105/joss.01417), the Python adaptation of Laine's mcmcstat. ``pss_svd``
translates PSS_SVD.m from Dadashova et al., Phil. Trans. A 2025,
10.1098/rsta.2024.0219, author source Zenodo 14366956, CC-BY 4.0.

This is NOT a reproduction of the authors' antibody brain model, million-step
experiment, or manually refined iterative selection. We use the same declared
small-molecule simulator, five log concentrations and four log multipliers as
the project comparator. Relative log-output sensitivities and the declared eta
replace the original raw-output sensitivities/threshold. The all-parameter
control is DRAM Bayesian inference, NOT the NUTS population BayesPBPK tutorial.

Only the callable's development predictions are visible here. No target-study
observations, target errors or target-based stopping rule enter this module.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import math
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np
from scipy.optimize import least_squares


VERSION = "published-pbpk-inference-20261011-v1"
METHODS = ("bayesian_dram", "pss_bayesian_dram")
PARAMETER_NAMES = ("log_clearance", "log_partition", "log_exchange", "log_blood_partition")
DEFAULT_CONFIG = {"chains": 4, "iterations": 2000, "burnin": 1000,
                  "max_iterations": 8000, "sigma_log": 0.3, "prior_sd": None,
                  "pss_eta": 1e-6, "initialize_map": True, "map_max_nfev": 300}


def default_prior_sd(n_observations: int, sigma_log: float = 0.3) -> np.ndarray:
    """Match the frozen four-axis calibration penalty; not clinical certainty."""
    if n_observations < 1 or not math.isfinite(sigma_log) or sigma_log <= 0:
        raise ValueError("positive observation count and sigma_log are required")
    scale = sigma_log / (math.sqrt(n_observations) * 0.1)
    return np.array([np.inf, scale * np.log(10), scale * np.log(100), scale * np.log(10)])


def pss_svd(sensitivity: np.ndarray, eta: float = 1e-6) -> dict:
    """Translate author PSS_SVD.m's squared singular-value ratio rule.

    At each step remove the largest absolute loading in the least identifiable
    right singular vector. Ties use the first maximum, as MATLAB ``max`` does.
    Zero matrices are handled explicitly (the MATLAB loop reaches an empty set).
    """
    matrix = np.asarray(sensitivity, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.isfinite(matrix).all():
        raise ValueError("sensitivity must be a nonempty finite matrix")
    if not 0 <= eta < 1:
        raise ValueError("eta must lie in [0, 1)")
    indices = list(range(matrix.shape[1]))
    history = []
    while indices:
        _, values, vh = np.linalg.svd(matrix, full_matrices=True)
        # More columns than observations implies an exact nullspace.
        ratio = (float(values[-1] / values[0]) ** 2
                 if values[0] > 0 and matrix.shape[1] <= matrix.shape[0] else 0.0)
        record = {"indices": indices.copy(), "squared_singular_ratio": ratio,
                  "singular_values": values.tolist()}
        if ratio > eta:
            history.append(record)
            break
        remove = int(np.argmax(np.abs(vh[-1])))
        record["removed_index"] = indices[remove]
        history.append(record)
        del indices[remove]
        matrix = np.delete(matrix, remove, axis=1)
    return {"selected_indices": indices,
            "excluded_indices": [i for i in range(np.asarray(sensitivity).shape[1]) if i not in indices],
            "eta": float(eta), "history": history,
            "algorithm_source": "Zenodo:14366956/PSS_SVD.m",
            "source_license": "CC-BY-4.0"}


def log_sensitivity(predict: Callable[[np.ndarray], np.ndarray], initial: np.ndarray,
                    lower: np.ndarray, upper: np.ndarray, step: float = 1e-4) -> np.ndarray:
    """Finite differences in log-multiplier coordinates at nominal inputs."""
    columns = []
    for j in range(len(initial)):
        a, b = initial.copy(), initial.copy()
        a[j], b[j] = max(lower[j], a[j] - step), min(upper[j], b[j] + step)
        if b[j] <= a[j]:
            raise ValueError("each parameter needs a positive bounded sensitivity interval")
        columns.append((predict(b) - predict(a)) / (b[j] - a[j]))
    return np.column_stack(columns)


def _load_libraries():
    # Dependencies belong to this project; do not alter the shared app venv.
    project = Path(__file__).resolve().parents[3]
    dependency_dir = project / "runtime" / "published-inference-python"
    if dependency_dir.is_dir() and str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))
    # mcmcplot's unused ellipse helper imports NumPy aliases removed by SciPy.
    # Restoring these identical aliases changes no likelihood or sampler code.
    import scipy
    for name in ("pi", "sin", "cos"):
        if not hasattr(scipy, name):
            setattr(scipy, name, getattr(np, name))
    from pymcmcstat.MCMC import MCMC
    import arviz as az
    return MCMC, az


def _chain_diagnostics(chains: np.ndarray, az) -> dict:
    rhats, bulk, tail = [], [], []
    for j in range(chains.shape[-1]):
        sample = chains[:, :, j]
        rhats.append(float(az.rhat(sample, method="rank")))
        bulk.append(float(az.ess(sample, method="bulk")))
        tail.append(float(az.ess(sample, method="tail")))
    finite = np.isfinite(rhats).all() and np.isfinite(bulk).all() and np.isfinite(tail).all()
    def clean(values):
        return [float(x) if np.isfinite(x) else None for x in values]
    return {"rank_normalized_split_rhat": clean(rhats), "bulk_ess": clean(bulk),
            "tail_ess": clean(tail),
            "converged": bool(finite and max(rhats) <= 1.05 and min(bulk + tail) >= 100),
            "strict_rhat_1_01": bool(finite and max(rhats) <= 1.01),
            "thresholds": {"rhat": 1.05, "bulk_and_tail_ess": 100},
            "bulk_ESS_1000_all_axes": bool(finite and min(bulk) >= 1000)}


def fit(predict: Callable[[np.ndarray], np.ndarray], observed_log: np.ndarray,
        initial: np.ndarray, lower: np.ndarray, upper: np.ndarray, *,
        method: str = "bayesian_dram", seed: int = 20261011, chains: int = 4,
        iterations: int = 2000, burnin: int = 1000, max_iterations: int = 8000,
        sigma_log: float = 0.3, prior_sd: np.ndarray | None = None,
        pss_eta: float = 1e-6, initialize_map: bool = True,
        map_max_nfev: int = 300) -> dict:
    """Fit development observations using official DRAM and optional author PSS.

    Parameters are natural-log multipliers. ``predict`` returns natural-log
    concentrations at the fixed visible development times. Non-selected PSS
    axes remain at their initial evidence values. A development-only MAP warmup
    initializes four dispersed chains; the reported point model uses posterior
    medians, not the best target outcome or the warmup optimizer output.

    Chains are extended by doubling to a predeclared maximum based only on
    rank-normalized R-hat/ESS. Nonconverged predictions are retained and flagged.
    Caller must serialize/lock the returned fit before opening target data.
    """
    started = time.perf_counter()
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    observed_log = np.asarray(observed_log, dtype=float)
    initial, lower, upper = [np.asarray(x, dtype=float) for x in (initial, lower, upper)]
    if initial.shape != (4,) or lower.shape != (4,) or upper.shape != (4,):
        raise ValueError("this protocol requires four log multipliers")
    if not all(np.isfinite(x).all() for x in (observed_log, initial, lower, upper)):
        raise ValueError("inputs must be finite")
    if observed_log.ndim != 1 or len(observed_log) < 2:
        raise ValueError("at least two development observations are required")
    if np.any(upper <= lower) or np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("initial parameters must lie within strictly ordered bounds")
    if chains < 2 or burnin < 0 or iterations - burnin < 20 or max_iterations < iterations:
        raise ValueError("at least two chains and twenty post-warmup draws are required")
    prior_sd = default_prior_sd(len(observed_log), sigma_log) if prior_sd is None else np.asarray(prior_sd, float)
    if prior_sd.shape != (4,) or np.any(prior_sd <= 0) or np.isnan(prior_sd).any():
        raise ValueError("prior_sd must have four positive entries; infinity means bounded uniform")
    if sigma_log <= 0 or not np.isfinite(sigma_log):
        raise ValueError("sigma_log must be finite and positive")
    calls, failure_count, failures = 0, 0, []

    def checked_prediction(theta):
        nonlocal calls
        calls += 1
        output = np.asarray(predict(np.asarray(theta, float)), dtype=float)
        if output.shape != observed_log.shape or not np.isfinite(output).all():
            raise ValueError("simulator must return one finite log concentration per visible observation")
        return output

    # Validate the nominal simulator before changing any sampler state.
    nominal_prediction = checked_prediction(initial)
    selection = {"selected_indices": list(range(4)), "excluded_indices": []}
    if method == "pss_bayesian_dram":
        sensitivity = log_sensitivity(checked_prediction, initial, lower, upper)
        selection = pss_svd(sensitivity, pss_eta)
        selection["sensitivity"] = sensitivity.tolist()
        selection["adaptation"] = "d(log concentration)/d(log multiplier), centered finite differences 1e-4"
        selection["author_threshold_sensitivity_only"] = {
            str(eta): pss_svd(sensitivity, eta)["selected_indices"]
            for eta in (2.5e-8, 1e-8, 2.5e-10, 1e-10, 1e-12)}
    selected = np.array(selection["selected_indices"], dtype=int)
    if len(selected) == 0:
        return {"point_parameters": initial.copy(), "posterior_samples": initial[None, :].copy(),
                "selected_indices": [], "diagnostics_dev_log_mse": float(np.mean((nominal_prediction-observed_log)**2)),
                "simulator_calls": calls, "diagnostics": {"version": VERSION, "method": method,
                "selection": selection, "converged": False, "status": "no_locally_identifiable_axes",
                "target_observations_used": False, "elapsed_seconds": time.perf_counter()-started}}

    def expand(theta):
        whole = initial.copy()
        whole[selected] = theta
        return whole

    def residual(theta):
        nonlocal failure_count
        try:
            data = (checked_prediction(expand(theta)) - observed_log) / sigma_log
            regularizer = theta / prior_sd[selected]
            return np.r_[data, regularizer]
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failure_count += 1
            if len(failures) < 20:
                failures.append({"type": type(exc).__name__, "message": str(exc)[:300]})
            return np.full(len(observed_log) + len(selected), 1e6)

    point = initial[selected].copy()
    proposal_covariance = np.eye(len(selected)) * 0.05 ** 2
    warmup = {"enabled": bool(initialize_map), "success": None}
    if initialize_map:
        opt = least_squares(residual, point, bounds=(lower[selected], upper[selected]),
                            max_nfev=map_max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        point = opt.x
        information = opt.jac.T @ opt.jac + np.eye(len(selected)) * 1e-6
        vals, vecs = np.linalg.eigh(np.linalg.inv(information))
        covariance = (vecs * np.clip(vals, 1e-6, 0.5)) @ vecs.T
        proposal_covariance = covariance * 2.38 ** 2 / len(selected)
        warmup.update(success=bool(opt.success), nfev=int(opt.nfev), cost=float(opt.cost),
                      parameters=expand(point).tolist(), message=str(opt.message))

    def ssfun(theta, _data):
        return float(np.sum(residual(theta)[:len(observed_log)] ** 2) * sigma_log ** 2)

    MCMC, az = _load_libraries()
    random_state, error_state = np.random.get_state(), np.geterr()
    samplers = []
    convergence_history = []
    rng = np.random.default_rng(seed)
    try:
        for chain_index in range(chains):
            sampler = MCMC(rngseed=int(seed + 104729 * chain_index))
            sampler.data.add_data_set(x=np.arange(len(observed_log)), y=observed_log)
            # sigma_log is frozen and shared with the other inference controls.
            # Do not specify N0: that option enables variance updates in pymcmcstat.
            sampler.model_settings.define_model_settings(sos_function=ssfun, sigma2=sigma_log**2)
            chain_initial = point + rng.multivariate_normal(np.zeros(len(selected)), proposal_covariance * 0.25)
            margin = np.minimum(1e-8, (upper[selected]-lower[selected]) * 0.01)
            chain_initial = np.clip(chain_initial, lower[selected]+margin, upper[selected]-margin)
            for j, axis in enumerate(selected):
                sampler.parameters.add_model_parameter(
                    name=PARAMETER_NAMES[axis], theta0=float(chain_initial[j]),
                    minimum=float(lower[axis]), maximum=float(upper[axis]),
                    prior_mu=0.0, prior_sigma=float(prior_sd[axis]))
            sampler.simulation_options.define_simulation_options(
                nsimu=iterations, method="dram", adaptint=100,
                lastadapt=max(100, burnin), qcov=proposal_covariance,
                updatesigma=False, waitbar=False, verbosity=0,
                save_to_json=False, save_to_txt=False, save_to_bin=False)
            sampler.run_simulation()
            samplers.append(sampler)
        total_iterations = iterations
        while True:
            posterior = np.stack([s.simulation_results.results["chain"][burnin:] for s in samplers])
            diagnostics = _chain_diagnostics(posterior, az)
            convergence_history.append({"iterations_per_chain": total_iterations, **diagnostics})
            if diagnostics["converged"] or total_iterations >= max_iterations:
                break
            next_total = min(max_iterations, total_iterations * 2)
            for chain_index, sampler in enumerate(samplers):
                # Independent reproducible continuation random streams. Library
                # retains each chain's state/covariance; no target data are read.
                np.random.seed(int(seed + 104729 * chain_index + total_iterations))
                # mcmcstat includes the retained initial state in nsimu, so
                # continuation adds nsimu-1 genuinely new chain states.
                sampler.simulation_options.nsimu = next_total - total_iterations + 1
                sampler.simulation_options.qcov = sampler.simulation_results.results["qcov"]
                sampler.simulation_options.adaptint = 0
                # Initialization populated N0 internally; resetting it prevents
                # pymcmcstat's restart path from silently enabling sigma updates.
                sampler.model_settings.N0 = None
                sampler.model_settings.sigma2 = np.array([sigma_log**2])
                sampler.simulation_options.updatesigma = False
                sampler.run_simulation(use_previous_results=True)
                if sampler.simulation_options.updatesigma:
                    raise RuntimeError("fixed observation variance was not preserved on continuation")
            total_iterations = next_total
    finally:
        np.random.set_state(random_state)
        np.seterr(**error_state)
    samples_selected = posterior.reshape(-1, len(selected))
    samples = np.tile(initial, (len(samples_selected), 1))
    samples[:, selected] = samples_selected
    point_parameters = np.median(samples, axis=0)
    point_prediction = checked_prediction(point_parameters)
    per_chain = []
    for sampler in samplers:
        chain = sampler.simulation_results.results["chain"]
        per_chain.append({"draws": int(len(chain)), "retained": int(len(chain)-burnin),
                          "move_fraction": float(np.mean(np.any(np.diff(chain, axis=0) != 0, axis=1)))})
    details = {"version": VERSION, "method": method, "selection": selection,
               "sampler": "pymcmcstat DRAM", "sampler_version": importlib.metadata.version("pymcmcstat"),
               "arviz_version": importlib.metadata.version("arviz"), "sigma_log": float(sigma_log),
               "sigma_fixed": True, "prior_mean": np.zeros(4).tolist(),
               "prior_sd": [None if np.isinf(x) else float(x) for x in prior_sd],
               "prior_sd_null_means": "bounded uniform", "prior_bounds": [lower.tolist(), upper.tolist()],
               "point_rule": "componentwise posterior median in natural-log multiplier coordinates",
               "warmup_map": warmup, "chains": per_chain, "burnin_per_chain": burnin,
               "convergence_history": convergence_history, **diagnostics,
               "development_observations": int(len(observed_log)), "target_observations_used": False,
               "simulator_calls": calls, "simulator_failure_count": failure_count,
               "first_simulator_failures": failures,
               "elapsed_seconds": time.perf_counter()-started,
               "status": "complete" if diagnostics["converged"] else "complete_with_convergence_warning",
               "adaptation_limit": "No manual algebraic-dependency or prior-update C1/C2 claims; not full author workflow."}
    return {"point_parameters": point_parameters, "posterior_samples": samples,
            "selected_indices": selected.tolist(), "diagnostics": details,
            "diagnostics_dev_log_mse": float(np.mean((point_prediction-observed_log)**2)),
            "simulator_calls": calls}
