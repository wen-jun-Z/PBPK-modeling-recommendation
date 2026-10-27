"""Portable, fixed-human PBPK case presentation and real CPU prediction.

Only exported, source-audited inputs are accepted. This module never opens the
server database, original evidence, native executables, or clinical outcomes.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import io
import json
import math
from numbers import Real
import time
from urllib.parse import urlsplit

import numpy as np

from pbpk_kb.blueprint_policy.blood_partition_calibration import simulate
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction


PARAMETER_LABELS = {
    "clearance_scale": "Nonrenal clearance multiplier",
    "partition_scale": "Tissue partition multiplier",
    "bidirectional_exchange_scale": "Bidirectional exchange multiplier",
    "blood_partition_scale": "Blood cell partition multiplier",
}
ORGAN_LABELS = {
    "Heart": "Heart", "Lung": "Lung", "Liver": "Liver", "Kidney": "Kidney",
    "Spleen": "Spleen", "Stomach": "Stomach", "SmallIntestine": "Small intestine",
    "LargeIntestine": "Large intestine", "Bone": "Bone", "Brain": "Brain", "Fat": "Fat",
    "Gonads": "Gonads", "Muscle": "Muscle", "Pancreas": "Pancreas", "Skin": "Skin",
}
EVIDENCE_LABELS = {
    "molecular_weight": "Molecular weight", "logp": "Lipophilicity", "pka": "pKa",
    "pka_types": "Acid-base dissociation type", "fraction_unbound": "Unbound plasma fraction",
    "binding_partner": "Plasma binding protein", "total_clearance": "Total clearance",
    "renal_clearance": "Renal clearance", "hepatic_clearance": "Hepatic clearance",
    "solubility": "Solubility", "blood_to_plasma_ratio": "Blood-to-plasma ratio",
}


def _bounded(value, label, maximum, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value > maximum
            or (value < 0 if zero else value <= 0)):
        raise ValueError(f"{label} must be a finite {'nonnegative' if zero else 'positive'} number no greater than {maximum:g}")
    return float(value)


def _mw(request):
    return next(float(row["value"]) for row in request["evidence"]
                if row["parameter"] == "molecular_weight")


def _result(case, request, curve, *, mass_balance=None, elapsed_seconds=None):
    summary = summarize_extended_prediction(curve, request)
    return {
        "version": "cloud-fixed-human-blueprint-1",
        "case_id": case["case_id"],
        "compound": case.get("compound") or case.get("label"),
        "request": deepcopy(request), "curve": deepcopy(curve),
        "parameters": deepcopy(case["parameters"]),
        "calibration": deepcopy(case["calibration"]),
        "physiology": deepcopy(case["physiology"]),
        "provenance": deepcopy(case.get("provenance", {})),
        "endpoint_summary": summary, "mass_balance": mass_balance,
        "elapsed_seconds": elapsed_seconds,
        "scenario_changed": (request["dosing"] != case["request"]["dosing"]
                             or request["target"] != case["request"]["target"]),
        "parameters_fitted": False, "scientific_accepted": False,
        "source_verification": "Source files checked during export; cloud uses the exported snapshot.",
        "limitations": [
            "Fixed case physiology, one compound, and a single intravenous dose; all four calibrated parameters remain fixed.",
            "Changed dosing is model extrapolation without new development observations or independent validation.",
            "New human physiology and calibration against new observations require the original server workspace.",
        ],
    }


def saved_result(case):
    """Present the actual locked prediction without recalculation or rescaling."""
    return _result(case, case["request"], case["saved_curve"])


def predict_case(case, *, dose_mg=None, infusion_minutes=None, window_end_h=None,
                 times_h=None):
    """Re-solve the existing four-parameter equations on unchanged physiology.

    ``times_h`` supports numerical replay tests; the UI always uses the bounded
    dense grid. No observations are read and no parameters are estimated.
    """
    begin = time.monotonic()
    request = deepcopy(case["request"])
    dosing = request["dosing"]
    if (dosing.get("route") != "intravenous" or dosing.get("start_h", 0) != 0
            or dosing.get("single_dose", True) is not True):
        raise ValueError("Cloud blueprints support registered single-dose intravenous cases only")
    dose = _bounded(dosing["dose_mg"] if dose_mg is None else dose_mg, "Dose (mg)", 100000)
    infusion = _bounded(dosing["infusion_minutes"] if infusion_minutes is None else infusion_minutes,
                        "Infusion duration (min)", 10080, zero=True)
    end = _bounded(request["target"]["window_end_h"] if window_end_h is None else window_end_h,
                   "Prediction horizon (h)", 168)
    if infusion / 60 >= end or end <= request["target"]["window_start_h"]:
        raise ValueError("The prediction horizon must extend beyond the infusion and the start of the observation window")
    dosing.update(dose_mg=dose, infusion_minutes=infusion)
    request["target"]["window_end_h"] = end
    if times_h is None:
        times = np.unique(np.r_[0., np.geomspace(min(1e-5, end / 10000), end, 700),
                                 np.linspace(0, end, 401), infusion / 60])
    else:
        times = np.asarray(times_h, dtype=float)
        if (times.ndim != 1 or not 2 <= len(times) <= 5000 or not np.isfinite(times).all()
                or times[0] != 0 or times[-1] < end or times[-1] > 168
                or not (np.diff(times) > 0).all()):
            raise ValueError("Replay requires a finite, increasing time grid that covers the prediction window")
    prediction = simulate(case["physiology"], request, times, case["parameters"]["log_parameters"])
    curve = {
        "time_min": (times * 60).tolist(),
        "concentration": (np.asarray(prediction["concentration_mg_l"]) * 1000 / _mw(request)).tolist(),
        "time_unit": "min", "unit": "umol/l", "observation": prediction["observation"],
    }
    return _result(case, request, curve, mass_balance=prediction["mass_balance"],
                   elapsed_seconds=time.monotonic() - begin)


def curve_csv(result):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["time_h", "concentration_ng_ml"])
    mw = _mw(result["request"])
    writer.writerows((t / 60, c * mw) for t, c in
                     zip(result["curve"]["time_min"], result["curve"]["concentration"]))
    return buffer.getvalue().encode("utf-8-sig")


def _human_label(human):
    sex = {"MALE": "Male", "FEMALE": "Female"}.get(human.get("sex"), "Sex not recorded")
    return (f"{human['age_years']:g} years · {sex} · "
            f"{human['weight_kg']:g} kg · {human['height_cm']:g} cm")


def _dosing_label(dosing):
    duration = dosing.get("infusion_minutes", 0)
    mode = f"IV infusion over {duration:g} min" if duration else "IV bolus"
    return f"{dosing['dose_mg']:g} mg · {mode}"


def _public_url(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value.strip())
        if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password:
            return value.strip()
    except ValueError:
        pass
    return None


def _chart(result):
    import altair as alt
    import pandas as pd

    visible = result["endpoint_summary"]["endpoints"]["plasma_concentration_time"]
    values = np.asarray(visible["concentration_umol_per_l"]) * _mw(result["request"])
    index = sorted(set(np.linspace(0, len(values) - 1, min(len(values), 700), dtype=int))
                   | {int(np.argmax(values))})
    frame = pd.DataFrame({"Time (h)": [visible["time_h"][i] for i in index],
                          "Plasma concentration (ng/mL)": [float(values[i]) for i in index]})
    base = alt.Chart(frame).encode(
        x=alt.X("Time (h):Q", axis=alt.Axis(grid=False), scale=alt.Scale(zero=True)),
        y=alt.Y("Plasma concentration (ng/mL):Q", scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("Time (h):Q", format=".4g"),
                 alt.Tooltip("Plasma concentration (ng/mL):Q", format=".5g")],
    )
    return (base.mark_area(color="#7F9E8C", opacity=.13)
            + base.mark_line(color="#4E7764", strokeWidth=2.5)).properties(height=340).configure_view(stroke=None)


def render_blueprints(data):
    """Render within the Cloud app; no global configuration or filesystem I/O."""
    import pandas as pd
    import streamlit as st

    @st.cache_data(show_spinner=False, ttl=3600, max_entries=40)
    def cached_prediction(case, dose, infusion, end):
        return predict_case(case, dose_mg=dose, infusion_minutes=infusion, window_end_h=end)

    st.title("Blueprint design")
    st.caption("PBPK predictions built from human physiology, drug parameter evidence, and existing calibration.")
    cases = sorted(data.get("cases", []), key=lambda row: str(row.get("compound") or row["label"]).lower())
    if not cases:
        st.info("No blueprint study cases are available yet.")
        return
    case_map = {row["case_id"]: row for row in cases}
    selected = st.selectbox("Study case", options=list(case_map), key="cloud_blueprint_case",
                            format_func=lambda key: case_map[key].get("label") or case_map[key]["compound"])
    case = case_map[selected]
    calibration = case["calibration"]
    request = case["request"]
    human_col, calibration_col = st.columns(2)
    with human_col, st.container(border=True):
        st.subheader("Prediction population")
        st.write(_human_label(request["human"]))
        st.caption(f"{len(case['physiology']['organs'])} organs · Fixed case physiology")
    with calibration_col, st.container(border=True):
        st.subheader("Calibration evidence")
        st.write(_human_label(calibration["human"]))
        st.caption(_dosing_label(calibration["dosing"]))
        st.caption(f"{calibration.get('source_label') or calibration.get('study_id') or 'Development study'}"
                   f" · {calibration['visible_points']} development observations")
    st.caption("Human characteristics are fixed by the study data. New physiology and calibration against new observations require the original server workspace.")

    with st.form(f"cloud_blueprint_form_{selected}", border=True):
        st.subheader("Dosing and prediction")
        dose_col, infusion_col, window_col = st.columns(3)
        with dose_col:
            dose = st.number_input("Dose (mg)", min_value=.000001, max_value=100000.,
                                   value=float(request["dosing"]["dose_mg"]), format="%.4f")
        with infusion_col:
            infusion = st.number_input("Infusion duration (min; 0 for bolus)", min_value=0., max_value=10080.,
                                       value=float(request["dosing"]["infusion_minutes"]), format="%.2f")
        with window_col:
            end = st.number_input("Prediction horizon (h)", min_value=.000001, max_value=168.,
                                  value=float(request["target"]["window_end_h"]), format="%.2f")
        st.caption("The existing four parameters stay fixed. Changed dosing is model extrapolation, without recalibration.")
        submit = st.form_submit_button("Generate prediction", type="primary")
    result_key = f"cloud_blueprint_result_{selected}"
    if submit:
        try:
            with st.spinner("Solving the concentration curve…"):
                st.session_state[result_key] = cached_prediction(case, dose, infusion, end)
        except (ValueError, ArithmeticError, KeyError) as exc:
            st.error(f"Prediction could not be generated: {exc}")
    result = st.session_state.get(result_key) or saved_result(case)
    active_request = result["request"]
    if result_key in st.session_state and st.button("Restore saved prediction", key=f"cloud_blueprint_reset_{selected}"):
        del st.session_state[result_key]
        st.rerun()
    st.subheader("Prediction results")
    st.caption(_dosing_label(active_request["dosing"])
               + f" · Observation window {active_request['target']['window_start_h']:g}–"
               f"{active_request['target']['window_end_h']:g} h"
               + (" · Newly calculated" if result_key in st.session_state else " · Saved case prediction"))
    endpoints = result["endpoint_summary"]["endpoints"]
    a, b, c = st.columns(3)
    a.metric("Cmax (ng/mL)", f"{endpoints['cmax']['value_mg_per_l'] * 1000:,.3g}")
    b.metric("AUC (ng·h/mL)", f"{endpoints['auc']['value_mg_h_per_l'] * 1000:,.3g}")
    c.metric("Tmax (h)", f"{endpoints['tmax']['value_h']:g}")
    st.altair_chart(_chart(result), width="stretch")
    st.caption("Curves and endpoints are model predictions. Cmax and Tmax are derived from the output grid; AUC covers the observation window shown above.")

    with st.expander("Human physiology template and model parameters"):
        st.write("The template contains arterial, venous, and portal circulation; vascular–tissue exchange in 15 organs; "
                 "tissue and blood cell partitioning; and renal and nonrenal clearance. Organ volumes and blood flows reflect each study's human characteristics.")
        st.dataframe(pd.DataFrame([{"Organ": ORGAN_LABELS.get(row["name"], row["name"]),
                                    "Volume (L)": row["volume_l"], "Blood flow (L/h)": row["flow_l_h"]}
                                   for row in result["physiology"]["organs"]]),
                     hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame([{"Calibrated parameter": label, "Multiplier": result["parameters"]["parameters"][name]}
                                   for name, label in PARAMETER_LABELS.items()]),
                     hide_index=True, width="stretch")
        st.caption("The four parameters were fitted to the development observations shown above and remain fixed. Human characteristics are not fitted parameters.")
    with st.expander("Drug parameter evidence"):
        rows = []
        for row in case.get("evidence", []):
            value = row.get("value")
            rows.append({"Parameter": EVIDENCE_LABELS.get(row.get("parameter"), row.get("parameter")),
                         "Value": json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value),
                         "Unit": row.get("unit") or "—", "Source": _public_url(row.get("source_url"))})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"Source": st.column_config.LinkColumn("Source", display_text="Read source ↗")})
        st.caption("A blank source means that a direct reference link has not been confirmed.")
    with st.expander("Additional prediction endpoints"):
        columns = [("terminal_half_life", "Terminal half-life", "value_h", "h"),
                   ("auc_inf", "AUC∞", "value_mg_h_per_l", "mg·h/L"),
                   ("clearance", "Clearance", "value_l_per_h", "L/h"),
                   ("vz", "Vz", "value_l", "L"), ("mrt", "MRT", "value_h", "h"),
                   ("vss", "Vss", "value_l", "L")]
        st.dataframe(pd.DataFrame([{"Endpoint": label,
                                    "Prediction": f"{endpoints[name][key]:.5g}" if endpoints[name].get("status") == "completed" and key in endpoints[name] else "Not estimable",
                                    "Unit": unit}
                                   for name, label, key, unit in columns]), hide_index=True, width="stretch")
        st.caption("Terminal endpoints are reported only when the existing numerical quality criteria are met.")
    left, right = st.columns(2)
    with left:
        st.download_button("Download prediction CSV", curve_csv(result),
                           file_name=f"{selected}_prediction.csv", mime="text/csv", width="stretch")
    with right:
        st.download_button("Download blueprint JSON", json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2),
                           file_name=f"{selected}_blueprint.json", mime="application/json", width="stretch")
