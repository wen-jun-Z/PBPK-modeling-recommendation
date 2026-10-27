"""Isolate four-point evidence gaps and protocol-matched prediction goals.

The data steward sees source documents, verifies them outside this module, and
exports only visible development values and whitelisted target protocol
metadata. The goal function accepts a predicted curve and that metadata; it
has no observed-concentration or reference-endpoint argument.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral

from blueprint_evidence_acquisition_data import _document, _number, _safe_document


VERSION = "blueprint-gap-design-data-1"
REGIMES = ("early_four", "late_four")
GOALS = ("cmax", "auc_inf", "terminal_half_life")
_HUMAN_FIELDS = {"age_years", "weight_kg", "height_cm", "sex", "population", "seed"}
_DOSING_FIELDS = {"route", "dose_mg", "start_h", "infusion_minutes"}
_POLICY_FIELDS = {"sampling_time_h", "dose_ng", "infusion_h", "weight_kg", "integration",
                  "auc_window_end_h", "c_last_time_h", "mrt_infusion_correction"}
_CONTRACT_FIELDS = {"version", "case_id", "study_id", "compound_id", "human", "dosing",
                    "molecular_weight_g_mol", "policy", "cutoff_ng_ml", "cutoff_basis",
                    "endpoint_units", "source_kind", "source_verification",
                    "target_observation_values_used"}


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be nonempty text")
    return value


def _conditions(request):
    if not isinstance(request, dict):
        raise ValueError("An explicit request dictionary is required")
    human, dosing = request.get("human"), request.get("dosing")
    if (not isinstance(human, dict) or set(human)-_HUMAN_FIELDS
            or not isinstance(dosing, dict) or set(dosing)-_DOSING_FIELDS):
        raise ValueError("Human/dosing metadata contains missing or unknown fields")
    _number(human.get("weight_kg"), "Body weight")
    for key in ("age_years", "height_cm"):
        if key in human:
            _number(human[key], key)
    if "seed" in human and (isinstance(human["seed"], bool)
            or not isinstance(human["seed"], Integral) or human["seed"] < 0):
        raise ValueError("Human seed must be a nonnegative integer")
    if "sex" in human and human["sex"] not in {"MALE", "FEMALE"}:
        raise ValueError("Unknown human sex")
    if "population" in human:
        _identity(human["population"], "Population")
    if dosing.get("route") != "intravenous" or dosing.get("start_h", 0.) != 0.:
        raise ValueError("This pilot requires an intravenous dose starting at zero")
    if isinstance(dosing.get("start_h", 0.), bool):
        raise ValueError("Dose start must be a numerical zero")
    _number(dosing.get("dose_mg"), "Dose")
    _number(dosing.get("infusion_minutes"), "Infusion duration", zero=True)
    return deepcopy(human), deepcopy(dosing)


def _molecular_weight(request):
    evidence = request.get("evidence")
    if not isinstance(evidence, list) or any(not isinstance(row, dict) for row in evidence):
        raise ValueError("Request requires an evidence list")
    rows = [row for row in evidence if row.get("parameter") == "molecular_weight"]
    if len(rows) != 1 or rows[0].get("unit") != "g/mol":
        raise ValueError("Exactly one molecular weight in g/mol is required")
    return _number(rows[0].get("value"), "Molecular weight")


def split_development(full_document, request, regime):
    """Reveal first/last four positive-time values using times alone.

    Candidate IDs are original zero-based full-source row indices. Time-zero
    rows are not acquisition candidates. Unknown source metadata is stripped.
    """
    if regime not in REGIMES:
        raise ValueError("Unknown evidence-gap regime")
    full = _document(full_document, name="Full development data")
    _conditions(request)
    for key in ("compound_id", "human", "dosing"):
        if full_document[key] != request.get(key):
            raise ValueError("Development conditions differ from request: " + key)
    if full_document["molecular_weight_g_mol"] != _molecular_weight(request):
        raise ValueError("Development molecular weight differs from request")
    positive = [(i, row) for i, row in enumerate(full) if row["time_min"] > 0]
    if len(positive) < 5:
        raise ValueError("At least five positive-time development observations are required")
    selected = positive[:4] if regime == "early_four" else positive[-4:]
    visible_ids = {i for i, _ in selected}
    candidates = [(i, row) for i, row in positive if i not in visible_ids]
    return {"visible_training": _safe_document(full_document, [row for _, row in selected]),
            "design_only": {"observed_times_h": [row["time_min"]/60. for _, row in selected],
                            "candidate_times_h": [row["time_min"]/60. for _, row in candidates],
                            "candidate_ids": [i for i, _ in candidates]}}


def reveal_one(full_training, four_training, candidate_time_h):
    """Return four visible observations plus one selected, hidden value."""
    t = _number(candidate_time_h, "Candidate time in hours")
    full = _document(full_training, name="Full development data")
    visible = _document(four_training, name="Four-point development data")
    if len(visible) != 4 or any(row["time_min"] <= 0 for row in visible):
        raise ValueError("Exactly four positive-time visible observations are required")
    _conditions(four_training)
    for key in ("compound_id", "study_id", "human", "dosing", "molecular_weight_g_mol"):
        if full_training[key] != four_training[key]:
            raise ValueError("Full and visible development conditions differ: " + key)
    source = {row["time_min"]: row for row in full}
    if any(source.get(row["time_min"]) != row for row in visible):
        raise ValueError("Visible rows must exactly match full development times and values")
    if t in {row["time_min"]/60. for row in visible}:
        raise ValueError("Candidate time is already visible")
    candidate = [row for row in full if row["time_min"] > 0 and row["time_min"]/60. == t]
    if len(candidate) != 1:
        raise ValueError("Candidate is absent from the registered development menu")
    return _safe_document(four_training, sorted([*visible, candidate[0]], key=lambda row: row["time_min"]))


def _policy(raw):
    """Retain only policy constants used by the frozen prediction scorer."""
    if not isinstance(raw, dict):
        raise ValueError("An explicit frozen NCA policy is required")
    schedule = raw.get("sampling_time_h")
    if not isinstance(schedule, (list, tuple)) or len(schedule) < 2:
        raise ValueError("NCA sampling times must be explicit")
    schedule = [_number(t, "NCA sampling time", zero=True) for t in schedule]
    if schedule[0] != 0 or any(b <= a for a, b in zip(schedule, schedule[1:])):
        raise ValueError("NCA sampling times must start at zero and strictly increase")
    result = {"sampling_time_h": schedule,
              "dose_ng": _number(raw.get("dose_ng"), "NCA dose"),
              "infusion_h": _number(raw.get("infusion_h"), "NCA infusion duration"),
              "weight_kg": _number(raw.get("weight_kg"), "NCA body weight"),
              # The frozen scorer treats every non-linear-trapezoidal text as
              # its standard linear-up/log-down estimator; do not copy prose.
              "integration": "linear_trapezoidal" if raw.get("integration") == "linear_trapezoidal"
                             else "linear_up_log_down"}
    for key in ("auc_window_end_h", "c_last_time_h"):
        if key in raw:
            result[key] = _number(raw[key], key)
            if result[key] > schedule[-1]:
                raise ValueError("Endpoint window exceeds the declared sampling schedule")
    if "mrt_infusion_correction" in raw:
        if raw["mrt_infusion_correction"] not in {"none", "subtract_half_infusion"}:
            raise ValueError("Unknown MRT infusion correction")
        result["mrt_infusion_correction"] = raw["mrt_infusion_correction"]
    return result


def _match_policy_request(policy, request):
    _conditions(request)
    pairs = ((policy["weight_kg"], request["human"]["weight_kg"]),
             (policy["dose_ng"], request["dosing"]["dose_mg"]*1e6),
             (policy["infusion_h"], request["dosing"]["infusion_minutes"]/60.))
    if not all(math.isclose(a, b, rel_tol=1e-12) for a, b in pairs):
        raise ValueError("Frozen target NCA dose/weight/infusion differs from request")


def build_target_contract(case_config, clinical_document, request, molecular_weight_g_mol=None):
    """Data-steward export: target TIMES/settings and endpoint UNITS only.

    ``clinical_document`` may be a selected case or its enclosing cases object.
    Concentration arrays and endpoint value/availability fields are never
    inspected or copied. Source identity/hash verification stays with caller.
    The resulting contract can be passed to a design algorithm independently
    of the original clinical document.
    """
    from report_blueprint_validation import cutoff_from_policy
    from run_clinical_method_comparison import clinical_unit_factor
    from pbpk_kb.blueprints.clinical_prediction_nca import UNITS

    if not isinstance(case_config, dict) or not isinstance(clinical_document, dict):
        raise ValueError("Case configuration and clinical metadata must be dictionaries")
    cid = _identity(case_config.get("case_id"), "Case ID")
    study = _identity(case_config.get("study_id"), "Study ID")
    compound = _identity(case_config.get("compound_id"), "Compound ID")
    human, dosing = _conditions(request)
    if request.get("compound_id") != compound:
        raise ValueError("Target compound differs from request")
    clinical = clinical_document
    if "cases" in clinical:
        rows = clinical["cases"]
        if not isinstance(rows, list):
            raise ValueError("Clinical cases must be a list")
        selected = [row for row in rows if isinstance(row, dict) and row.get("case_id") == cid]
        if len(selected) != 1:
            raise ValueError("Require one uniquely matching clinical metadata case")
        clinical = selected[0]
    raw = clinical.get("clinical_reference")
    if (not isinstance(raw, dict) or any(clinical.get(k) != v or raw.get(k) != v
                                      for k, v in (("case_id", cid), ("study_id", study)))):
        raise ValueError("Clinical case/study metadata differs from the frozen case")
    policy = _policy(case_config.get("nca_policy"))
    _match_policy_request(policy, request)
    flags = clinical.get("approximations", [])
    if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
        raise ValueError("Clinical approximation flags must be a text list")
    cutoff, basis = cutoff_from_policy(case_config["nca_policy"], set(flags))
    mw = _molecular_weight(request)
    if molecular_weight_g_mol is not None:
        source_mw = _number(molecular_weight_g_mol, "Source molecular weight")
        if not math.isclose(mw, source_mw, rel_tol=1e-12):
            raise ValueError("Source molecular weight differs from target request")
        mw = source_mw
    endpoints = raw.get("endpoints")
    if not isinstance(endpoints, dict):
        raise ValueError("Clinical endpoint unit metadata is required")
    units = {}
    for goal in GOALS:
        row = endpoints.get(goal) or {}
        if not isinstance(row, dict):
            raise ValueError("Endpoint unit metadata must be a dictionary")
        units[goal] = row.get("unit") or UNITS[goal]
        if not isinstance(units[goal], str):
            raise ValueError("Endpoint unit must be text")
        clinical_unit_factor(goal, units[goal], policy["weight_kg"])
    return {"version": VERSION, "case_id": cid, "study_id": study, "compound_id": compound,
            "human": human, "dosing": dosing, "molecular_weight_g_mol": mw,
            "policy": policy, "cutoff_ng_ml": _number(cutoff, "Prediction cutoff"),
            "cutoff_basis": basis, "endpoint_units": units,
            "source_kind": "target_protocol_metadata_only",
            "source_verification": "caller_must_verify_source_identity_hashes_and_extraction",
            "target_observation_values_used": False}


def model_goals(contract, predicted_curve, request):
    """Compute the same three predicted scalars as the clinical scorer.

    ``predicted_curve`` must be a native prediction (min, umol/l); this matches
    _score_prediction exactly, including float-time coalescing, unit conversion,
    clinical sampling, cutoff proxy, integration and NCA tail selection.
    Missing tails stay None; a model's infinite-time eigenvalue is never used
    to replace the study-schedule terminal estimate.
    """
    from run_clinical_method_comparison import predicted_nca, clinical_unit_factor
    from summarize_additional_clinical_baselines import prediction_in_hours, weak_endpoints

    if (not isinstance(contract, dict) or set(contract) != _CONTRACT_FIELDS
            or contract.get("version") != VERSION
            or contract.get("source_kind") != "target_protocol_metadata_only"
            or contract.get("target_observation_values_used") is not False):
        raise ValueError("Require a complete metadata-only target contract")
    human, dosing = _conditions(request)
    if (contract["compound_id"] != request.get("compound_id")
            or contract["human"] != human or contract["dosing"] != dosing):
        raise ValueError("Target request differs from the metadata contract")
    if not isinstance(contract["policy"], dict) or set(contract["policy"])-_POLICY_FIELDS:
        raise ValueError("Unknown target protocol metadata fields")
    policy = _policy(contract["policy"])
    _match_policy_request(policy, request)
    mw = _number(contract["molecular_weight_g_mol"], "Contract molecular weight")
    if not math.isclose(mw, _molecular_weight(request), rel_tol=1e-12):
        raise ValueError("Contract molecular weight differs from request")
    cutoff = _number(contract["cutoff_ng_ml"], "Prediction cutoff")
    if contract["cutoff_basis"] not in {"reported_lloq", "reported_lod_proxy", "cited_assay_calibration_lower_bound_proxy"}:
        raise ValueError("Unknown prediction cutoff basis")
    units = contract["endpoint_units"]
    if not isinstance(units, dict) or set(units) != set(GOALS):
        raise ValueError("Exactly the three goal unit definitions are required")
    if (not isinstance(predicted_curve, dict) or predicted_curve.get("unit") != "umol/l"
            or predicted_curve.get("time_unit") != "min"):
        raise ValueError("Native predicted curve must use min and umol/l")
    times, values = predicted_curve.get("time_min"), predicted_curve.get("concentration")
    if (not isinstance(times, (list, tuple)) or not isinstance(values, (list, tuple))
            or len(times) != len(values) or len(times) < 2):
        raise ValueError("Native predicted time and concentration arrays must be paired")
    clean = {"time_min": [_number(t, "Prediction time", zero=True) for t in times],
             "concentration": [_number(c, "Predicted concentration", zero=True) for c in values]}
    hours, concentrations, merged = prediction_in_hours(clean, mw)
    result = predicted_nca(policy=policy, time_h=hours, concentration_ng_ml=concentrations,
                          dose_ng=policy["dose_ng"], infusion_h=policy["infusion_h"],
                          nominal_times_h=policy["sampling_time_h"], lloq_ng_ml=cutoff,
                          weight_kg=policy["weight_kg"])
    goals, reasons = {}, {}
    for goal in GOALS:
        if not isinstance(units[goal], str):
            raise ValueError("Endpoint unit must be text")
        factor = clinical_unit_factor(goal, units[goal], policy["weight_kg"])
        row = result["endpoints"][goal]
        goals[goal] = row["value"]*factor if row["value"] is not None else None
        reasons[goal] = row.get("reason", "")
    diagnostics = deepcopy(result["diagnostics"])
    diagnostics.update(quantification_threshold_basis=contract["cutoff_basis"],
                       coalesced_float_time_duplicates=merged)
    return {"version": VERSION, "values": goals, "units": deepcopy(units),
            "reasons": reasons, "nca_status": result["status"],
            "weak_endpoints": sorted(weak_endpoints(diagnostics)), "diagnostics": diagnostics,
            "target_observation_values_used": False}
