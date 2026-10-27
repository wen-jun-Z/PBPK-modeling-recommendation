"""Float64 scoring of explicit calibration curves, without dataset or model I/O.

Only concentration errors at supplied observation times are evaluated. This
does not estimate Cmax/AUC or establish validation on an independent study.
CUDA requests never fall back to CPU. Invalid curves retain their input row and
vector index, with null predictions/metrics and an explicit error.
"""
from __future__ import annotations

import math
from collections import defaultdict
from numbers import Real
from typing import Any

import torch


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must contain real numbers.")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be representable as float64.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _concentration_unit(value: Any) -> bool:
    # Micro-sign, Greek mu and ASCII u are spelling variants, not conversions.
    return isinstance(value, str) and value.replace("µ", "u").replace("μ", "u") == "umol/l"


def _device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("Choose an explicit cpu or cuda device.") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only cpu and cuda devices are supported.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; CPU fallback is disabled.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device {index} is unavailable; CPU fallback is disabled.")
        return torch.device("cuda", index)
    return torch.device("cpu")


def score_calibration_curves(
    *, runs: list[dict], observations: list[dict], device: str = "cuda",
) -> dict[str, Any]:
    """Linearly interpolate curves without extrapolation and score in float64.

    Runs explicitly declare minutes and µmol/l (also μmol/l or umol/l). Supplied
    observations require finite nonnegative ``time_min`` and finite positive
    ``concentration_umol_per_l``. Observation ordering is preserved. Natural
    logarithms define log_mse/log_rmse; GMFE is exp(mean(abs(log(pred)-log(obs)))).

    Malformed observation data or ambiguous vector IDs reject the whole call.
    A malformed run, out-of-range observation, or nonpositive interpolated
    prediction invalidates only that run. No point or candidate is dropped.
    The caller owns calibration/selection/test isolation; this function reads
    no files or other datasets and performs no simulation or optimization.
    """
    if not isinstance(observations, list) or not observations:
        raise ValueError("Provide at least one explicit calibration observation.")
    observed_times, observed_values = [], []
    for point in observations:
        if not isinstance(point, dict):
            raise ValueError("Each observation must be an object.")
        time = _number(point.get("time_min"), "Observation times")
        concentration = _number(point.get("concentration_umol_per_l"), "Observations")
        if time < 0 or concentration <= 0:
            raise ValueError("Observation times must be nonnegative and concentrations positive.")
        observed_times.append(time)
        observed_values.append(concentration)
    if not isinstance(runs, list):
        raise ValueError("Runs must be a list of curve objects.")
    indices: set[int] = set()
    for run in runs:
        if not isinstance(run, dict) or type(run.get("vector_index")) is not int or run["vector_index"] < 0:
            raise ValueError("Every run must have an explicit nonnegative integer vector_index.")
        if run["vector_index"] in indices:
            raise ValueError("Run vector_index values must be unique.")
        indices.add(run["vector_index"])

    target_device = _device(device)
    rows = [{"vector_index": run["vector_index"], "valid": False,
             "log_mse": None, "log_rmse": None, "gmfe": None,
             "predictions_umol_per_l": None, "error": None} for run in runs]
    groups: dict[int, list[tuple[int, list[float], list[float]]]] = defaultdict(list)
    for position, run in enumerate(runs):
        try:
            if run.get("time_unit") != "min":
                raise ValueError("unsupported_time_unit: expected explicit min")
            if not _concentration_unit(run.get("unit")):
                raise ValueError("unsupported_concentration_unit: expected explicit µmol/l, μmol/l or umol/l")
            times, concentrations = run.get("time_min"), run.get("concentration")
            if (not isinstance(times, list) or not isinstance(concentrations, list)
                    or len(times) < 2 or len(times) != len(concentrations)):
                raise ValueError("invalid_curve_shape: matching arrays with at least two samples are required")
            times = [_number(value, "Curve times") for value in times]
            concentrations = [_number(value, "Curve concentrations") for value in concentrations]
            if times[0] < 0 or any(right <= left for left, right in zip(times, times[1:])):
                raise ValueError("invalid_time_grid: nonnegative strictly increasing times are required")
            if min(observed_times) < times[0] or max(observed_times) > times[-1]:
                raise ValueError("observation_outside_curve: extrapolation is forbidden")
            groups[len(times)].append((position, times, concentrations))
        except ValueError as exc:
            rows[position]["error"] = str(exc)

    with torch.inference_mode():
        obs_time = torch.tensor(observed_times, dtype=torch.float64, device=target_device)
        log_observed = torch.log(torch.tensor(observed_values, dtype=torch.float64, device=target_device))
        for group in groups.values():
            # Curves with equal sample counts are scored together, even when
            # their individual time grids differ. No padding/point masking.
            times = torch.tensor([item[1] for item in group], dtype=torch.float64, device=target_device)
            concentrations = torch.tensor([item[2] for item in group], dtype=torch.float64, device=target_device)
            query = obs_time.unsqueeze(0).expand(len(group), -1).contiguous()
            upper = torch.searchsorted(times, query).clamp(1, times.shape[1] - 1)
            lower = upper - 1
            left_time, right_time = times.gather(1, lower), times.gather(1, upper)
            fraction = (query - left_time) / (right_time - left_time)
            predictions = ((1 - fraction) * concentrations.gather(1, lower)
                           + fraction * concentrations.gather(1, upper))
            prediction_finite = torch.isfinite(predictions).all(dim=1)
            prediction_positive = (predictions > 0).all(dim=1)
            log_error = torch.log(predictions) - log_observed.unsqueeze(0)
            log_mse = torch.mean(log_error.square(), dim=1)
            log_rmse = torch.sqrt(log_mse)
            gmfe = torch.exp(torch.mean(log_error.abs(), dim=1))
            metrics = torch.stack((log_mse, log_rmse, gmfe), dim=1)
            finite_metrics = torch.isfinite(metrics).all(dim=1)
            flags = torch.stack((prediction_finite, prediction_positive, finite_metrics), dim=1).cpu().tolist()
            predicted_rows, metric_rows = predictions.cpu().tolist(), metrics.cpu().tolist()
            for item, values, predicted, flags_row in zip(group, metric_rows, predicted_rows, flags):
                row = rows[item[0]]
                if not flags_row[0]:
                    row["error"] = "nonfinite_interpolated_prediction"
                elif not flags_row[1]:
                    row["error"] = "nonpositive_prediction_at_observation"
                elif not flags_row[2]:
                    row["error"] = "nonfinite_score"
                else:
                    row.update(valid=True, log_mse=values[0], log_rmse=values[1], gmfe=values[2],
                               predictions_umol_per_l=predicted)
    return {"device": str(target_device), "gpu_used": target_device.type == "cuda",
            "observation_count": len(observations), "rows": rows}
