"""Compound-held-out learned initializations for the native PBPK optimizer.

Targets are fitted teacher decisions, never clinical parameter ground truth.
This module performs no native simulation and opens no clinical/source files.
Its caller must verify source hashes and development/target separation first.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from numbers import Integral

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

VERSION = "development-curve-prior-1.0"
SEED = 20260929
PARTITIONS = ("PKSimStandard", "RodgersRowland")
SHAPE_QUANTILES = (0., .2, .4, .6, .8, 1.)
FEATURE_NAMES = (
    "age_years", "log_weight_kg", "height_cm", "female",
    "log_molecular_weight_g_mol", "logp", "log_fraction_unbound",
    "log_clearance_ml_min_kg", "renal_clearance_fraction",
    "acid_pka_count", "base_pka_count", "minimum_pka", "maximum_pka",
    "log_dose_mg_per_kg", "log_infusion_h", "log_first_time_h", "log_last_time_h",
    "log_time_span_h", "log_observation_count", "peak_time_log_fraction",
    "log_peak_dose_normalized_kg_l", "log_observed_window_auc_h_kg_l",
    "early_log_slope_asinh_per_h", "late_log_slope_asinh_per_h",
) + tuple(f"log_dose_normalized_curve_q{int(q*100):03d}" for q in SHAPE_QUANTILES)
MODEL_FEATURE_NAMES = FEATURE_NAMES + tuple("partition_" + p for p in PARTITIONS)
TEACHER_TARGET_NAMES = ("lipophilicity_delta_over_2p5", "log_clearance_scale_over_log4")


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(name + " must be numeric")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(name + " must be finite" + (" and positive" if positive else ""))
    return value


def _references(value):
    """Only identifiers are inspected, never files or clinical label values."""
    result = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            result.append({"path": value["path"], "sha256": value["sha256"]})
        for child in value.values():
            result.extend(_references(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.extend(_references(child))
    return result


def _reject_forbidden(value, request):
    forbidden = request.get("forbidden_sources", [])
    paths = {r["path"] for r in forbidden}
    digests = {r["sha256"] for r in forbidden}
    if any(r["path"] in paths or r["sha256"] in digests for r in _references(value)):
        raise ValueError("Held-out clinical source cannot enter the learned prior")


def _evidence(request):
    items = request["evidence"]
    result = {e["parameter"]: e for e in items}
    if len(result) != len(items):
        raise ValueError("Duplicate evidence parameter")
    for entry in items:
        if entry.get("compound_id") != request["compound_id"]:
            raise ValueError("Evidence compound identity mismatch")
    _reject_forbidden(items, request)
    return result


def _value(evidence, name, unit, *, positive=False):
    record = evidence[name]
    if record["unit"] != unit:
        raise ValueError("Unexpected unit for " + name)
    return _number(record["value"], name, positive=positive)


def _clearance(evidence):
    renal = _value(evidence, "renal_clearance", "ml/min/kg") if "renal_clearance" in evidence else 0.
    if "total_clearance" in evidence:
        total = _value(evidence, "total_clearance", "ml/min/kg", positive=True)
    elif "hepatic_clearance" in evidence and "renal_clearance" in evidence:
        if evidence.get("elimination_scope", {}).get("value") != "hepatic_and_renal":
            raise ValueError("Summing clearances requires hepatic_and_renal scope")
        total = _value(evidence, "hepatic_clearance", "ml/min/kg") + renal
    else:
        raise ValueError("Missing construction clearance prior")
    if total <= 0 or not 0 <= renal <= total:
        raise ValueError("Inconsistent renal and total clearance inputs")
    return total, renal/total


def curve_features(document, request):
    """Return 30 fixed features from the explicitly supplied development rows.

    Concentration is converted from umol/L to mg/L then divided by dose mg/kg.
    Log-time interpolation stays inside the observed window; no terminal
    extrapolation, hidden/full observations, clinical NCA labels or IDs enter.
    """
    if (document.get("role") != "calibration_data_reconstruction"
            or document.get("matrix") != "plasma" or document.get("time_unit") != "min"
            or document.get("unit") != "umol/l"):
        raise ValueError("A declared development plasma curve in min and umol/l is required")
    if document.get("compound_id") != request["compound_id"]:
        raise ValueError("Development/request compound identity mismatch")
    _reject_forbidden(document, request)
    human, dosing = request["human"], request["dosing"]
    if dosing["route"] != "intravenous":
        raise ValueError("Current learned prior supports single IV infusion only")
    for key in ("age_years", "weight_kg", "height_cm", "sex", "population"):
        if document["human"].get(key) != human.get(key):
            raise ValueError("Development and request human conditions differ")
    for key in ("route", "dose_mg", "infusion_minutes"):
        if document["dosing"].get(key) != dosing.get(key):
            raise ValueError("Development and request dosing conditions differ")
    if document["dosing"].get("start_h", 0.) != dosing.get("start_h", 0.):
        raise ValueError("Development and request start times differ")
    weight = _number(human["weight_kg"], "weight", positive=True)
    dose = _number(dosing["dose_mg"], "dose", positive=True)
    infusion = _number(dosing["infusion_minutes"], "infusion duration", positive=True)/60.
    start = _number(dosing.get("start_h", 0.), "dose start")
    if start < 0 or human["sex"] not in ("MALE", "FEMALE"):
        raise ValueError("Invalid sex or administration start")
    evidence = _evidence(request)
    mw = _value(evidence, "molecular_weight", "g/mol", positive=True)
    if "molecular_weight_g_mol" in document and not math.isclose(
            _number(document["molecular_weight_g_mol"], "document MW", positive=True), mw, rel_tol=1e-8):
        raise ValueError("Development/request molecular weight mismatch")
    logp = _value(evidence, "logp", "Log Units")
    fu = _value(evidence, "fraction_unbound", "fraction", positive=True)
    if fu > 1:
        raise ValueError("Fraction unbound exceeds one")
    cl, renal_fraction = _clearance(evidence)
    pkas = evidence.get("pka_types", {}).get("value", [])
    if any(p.get("Type") not in ("Acid", "Base") for p in pkas):
        raise ValueError("Unknown pKa type")
    pka_values = [_number(p["Pka"], "pKa") for p in pkas]
    rows = document["observations"]
    if len(rows) < 5:
        raise ValueError("At least five supplied development observations are required")
    if any(r.get("unit", "umol/l") != "umol/l" or r.get("time_unit", "min") != "min" for r in rows):
        raise ValueError("Per-row concentration/time unit mismatch")
    times = np.asarray([_number(r["time_min"], "time")/60.-start for r in rows])
    values = np.asarray([_number(r["concentration_umol_per_l"], "concentration", positive=True)
                         for r in rows]) * mw / 1000. / (dose/weight)
    if not (times > 0).all() or not (np.diff(times) > 0).all():
        raise ValueError("Observation times must be positive after dosing and strictly ordered")
    logt, logc = np.log(times), np.log(values)
    fractions = (logt-logt[0])/(logt[-1]-logt[0])
    shape = np.interp(SHAPE_QUANTILES, fractions, logc)
    area = float(np.trapezoid(values, times))  # Observed window only; not a clinical AUC label.
    early = float(np.polyfit(times[:3], logc[:3], 1)[0])
    late = float(np.polyfit(times[-3:], logc[-3:], 1)[0])
    features = np.asarray([
        _number(human["age_years"], "age", positive=True), math.log(weight),
        _number(human["height_cm"], "height", positive=True), float(human["sex"] == "FEMALE"),
        math.log(mw), logp, math.log(fu), math.log(cl), renal_fraction,
        sum(p["Type"] == "Acid" for p in pkas), sum(p["Type"] == "Base" for p in pkas),
        min(pka_values, default=0.), max(pka_values, default=0.),
        math.log(dose/weight), math.log(infusion), logt[0], logt[-1], math.log(times[-1]-times[0]),
        math.log(len(times)), fractions[int(np.argmax(values))], math.log(float(max(values))),
        math.log(area), math.asinh(early), math.asinh(late), *shape], dtype=np.float64)
    if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
        raise ValueError("Nonfinite curve features")
    return features


def _inputs(document, request, partition_method):
    if partition_method not in PARTITIONS:
        raise ValueError("Unknown native partition formula")
    return np.r_[curve_features(document, request), [float(partition_method == p) for p in PARTITIONS]]


def _weights(groups):
    counts = Counter(groups)
    # Repeated formulas/full/five are correlated observations, not more drugs.
    return np.asarray([len(groups)/(len(counts)*counts[g]) for g in groups], dtype=np.float64)


def _mlp_fit(x, y, groups, seed, epochs, device):
    import torch
    from torch import nn
    requested = torch.device(device)
    if requested.type not in ("cpu", "cuda"):
        raise ValueError("Only this process's CPU/CUDA device is supported")
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA is unavailable; no silent device fallback")
    group_names = sorted(set(groups))
    random = np.random.default_rng(seed)
    ordered = random.permutation(group_names).tolist()
    validation_groups = sorted(ordered[:max(2, int(math.ceil(len(group_names)*.2)))])
    train_mask = np.asarray([g not in validation_groups for g in groups])
    validation_mask = ~train_mask
    training_groups = sorted(set(np.asarray(groups)[train_mask]))
    inner_scaler = StandardScaler().fit(x[train_mask], sample_weight=_weights(np.asarray(groups)[train_mask]))
    inner_x = inner_scaler.transform(x)
    devices = [requested.index if requested.index is not None else torch.cuda.current_device()] if requested.type == "cuda" else []

    def new_net():
        return nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Linear(32, 32),
                             nn.ReLU(), nn.Linear(32, 2)).to(requested)

    def tensor(a):
        return torch.as_tensor(a, dtype=torch.float32, device=requested)

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        net = new_net()
        optimizer = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.01)
        tx, ty = tensor(inner_x[train_mask]), tensor(y[train_mask])
        vx, vy = tensor(inner_x[validation_mask]), tensor(y[validation_mask])
        tw, vw = tensor(_weights(np.asarray(groups)[train_mask])), tensor(_weights(np.asarray(groups)[validation_mask]))
        best_loss, best_epoch, stale = math.inf, 0, 0
        history = []
        for epoch in range(1, epochs+1):
            net.train(); optimizer.zero_grad()
            loss = ((net(tx)-ty).square().mean(dim=1)*tw).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite MLP training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
            net.eval()
            with torch.no_grad():
                validation = float(((net(vx)-vy).square().mean(dim=1)*vw).mean().cpu())
            if not math.isfinite(validation):
                raise ValueError("Nonfinite MLP validation loss")
            history.append({"epoch": epoch, "training_loss": float(loss.detach().cpu()), "validation_loss": validation})
            if validation < best_loss-1e-7:
                best_loss, best_epoch, stale = validation, epoch, 0
            else:
                stale += 1
            if stale >= 30:
                break
        # Refit only outer-training compounds, for the epoch count chosen using
        # inner held-out compounds. The excluded drug never picks an epoch.
        scaler = StandardScaler().fit(x, sample_weight=_weights(groups))
        torch.manual_seed(seed)
        net = new_net(); optimizer = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.01)
        tx, ty, tw = tensor(scaler.transform(x)), tensor(y), tensor(_weights(groups))
        for _ in range(best_epoch):
            optimizer.zero_grad(); loss = ((net(tx)-ty).square().mean(dim=1)*tw).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite final MLP training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
        state = {k: v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()}
    return scaler, state, {"device_used": str(requested), "hidden_widths": [32,32],
        "optimizer": "AdamW", "learning_rate": .003, "weight_decay": .01,
        "patience": 30, "maximum_epochs": epochs, "selected_epochs": best_epoch,
        "inner_training_compound_ids": training_groups, "inner_validation_compound_ids": validation_groups,
        "inner_scaler_mean": inner_scaler.mean_.tolist(), "inner_history": history,
        "final_refit_uses_all_outer_training_compounds": True}


def fit_prior(rows, excluded_compound, algorithm, seed=SEED, device="cpu", epochs=300):
    """Fit an RF/MLP teacher prior after excluding every row of one compound.

    Row keys: document, request, compound_id, sampling, partition_method,
    decision={lipophilicity, clearance_scale}; source_refs is optional audit
    metadata (e.g. development and teacher path/SHA records), never features.
    """
    if not isinstance(excluded_compound, str) or not excluded_compound:
        raise ValueError("An explicit excluded compound is required")
    if algorithm not in ("rf", "mlp"):
        raise ValueError("algorithm must be rf or mlp")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("Invalid seed")
    if isinstance(epochs, bool) or not isinstance(epochs, Integral) or not 1 <= epochs <= 10000:
        raise ValueError("epochs must be in 1..10000")
    retained, omitted = [], 0
    for row in rows:
        # This check occurs before decoding excluded curves or teacher targets.
        if row["compound_id"] == excluded_compound:
            omitted += 1
        else:
            retained.append(row)
    groups = [r["compound_id"] for r in retained]
    if len(set(groups)) < 8:
        raise ValueError("At least eight distinct training compounds are required")
    xx, yy, provenance = [], [], []
    for row in retained:
        req, doc = row["request"], row["document"]
        if row["compound_id"] != req["compound_id"] or row["compound_id"] != doc.get("compound_id"):
            raise ValueError("Row/document/request compound identities disagree")
        if row["sampling"] not in ("full", "five"):
            raise ValueError("Unknown development sampling setting")
        if row["sampling"] == "five" and len(doc["observations"]) != 5:
            raise ValueError("Five-point rows must contain exactly five observations")
        _reject_forbidden(row.get("source_refs", {}), req)
        xx.append(_inputs(doc, req, row["partition_method"]))
        prior = _value(_evidence(req), "logp", "Log Units")
        decision = row["decision"]
        target = ((_number(decision["lipophilicity"], "teacher logMA")-prior)/2.5,
                  math.log(_number(decision["clearance_scale"], "teacher scale", positive=True))/math.log(4.))
        if any(abs(v) > 1.+1e-12 for v in target):
            raise ValueError("Teacher decision exceeds native parameter bounds")
        yy.append(target)
        provenance.append({"compound_id": row["compound_id"], "sampling": row["sampling"],
            "partition_method": row["partition_method"], "source_refs": deepcopy(row.get("source_refs", {}))})
    x, y = np.asarray(xx), np.clip(np.asarray(yy), -1., 1.)
    if algorithm == "rf":
        scaler = StandardScaler().fit(x, sample_weight=_weights(groups))
        estimator = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=2,
            max_features=1., random_state=int(seed), n_jobs=1)
        estimator.fit(scaler.transform(x), y, sample_weight=_weights(groups))
        implementation = {"device_used": "cpu", "trees": 200, "max_depth": 4, "min_samples_leaf": 2}
    else:
        scaler, estimator, implementation = _mlp_fit(x, y, groups, int(seed), int(epochs), device)
    return {"version": VERSION, "algorithm": algorithm, "seed": int(seed), "excluded_compound": excluded_compound,
        "training_compound_ids": sorted(set(groups)), "training_rows": provenance, "excluded_rows": omitted,
        "feature_names": list(MODEL_FEATURE_NAMES), "teacher_target_names": list(TEACHER_TARGET_NAMES),
        "scaler": scaler, "estimator": estimator, "implementation": implementation,
        "native_execution_required": True, "teacher_is_clinical_ground_truth": False,
        "clinical_target_values_used": False, "source_hashes_verified_by_this_module": False,
        "limitations": ["Teachers are budget-limited development fits, not clinical parameter truth.",
            "Few correlated curves do not create additional independent compounds.",
            "Initialization quality is not native PBPK prediction quality; new native runs are required.",
            "Caller must verify source hashes, identity aliases and development/target provenance."]}


def predict_prior(model, document, request, partition_method):
    """Predict one bounded initial point; no calibration or native solving here."""
    if model.get("version") != VERSION or model.get("feature_names") != list(MODEL_FEATURE_NAMES):
        raise ValueError("Incompatible prior model or feature schema")
    cid = request["compound_id"]
    if cid in model["training_compound_ids"] or cid != model["excluded_compound"]:
        raise ValueError("Prediction must be for this model's excluded compound")
    x = model["scaler"].transform(_inputs(document, request, partition_method)[None, :])
    if model["algorithm"] == "rf":
        raw = model["estimator"].predict(x)[0]
    elif model["algorithm"] == "mlp":
        weights = model["estimator"]
        # CPU NumPy inference keeps the joblib object portable across devices.
        for layer in (0,2):
            x = np.maximum(x @ weights[f"{layer}.weight"].T + weights[f"{layer}.bias"], 0.)
        raw = (x @ weights["4.weight"].T + weights["4.bias"])[0]
    else:
        raise ValueError("Unknown saved prior algorithm")
    if np.asarray(raw).shape != (2,) or not np.isfinite(raw).all():
        raise ValueError("Invalid learned parameter prediction")
    bounded = np.clip(raw, -1., 1.)
    logp = _value(_evidence(request), "logp", "Log Units")
    return {"lipophilicity": float(logp+2.5*bounded[0]),
        "clearance_scale": float(math.exp(math.log(4.)*bounded[1])),
        "training_compound_ids": list(model["training_compound_ids"]), "algorithm": model["algorithm"],
        "partition_method": partition_method, "normalized_prediction": bounded.tolist(),
        "unclipped_normalized_prediction": np.asarray(raw).tolist(),
        "clipped_to_native_bounds": bool(np.any(raw != bounded)), "native_execution_required": True,
        "development_curve_used": True, "clinical_target_values_used": False,
        "teacher_is_clinical_ground_truth": False, "version": VERSION}
