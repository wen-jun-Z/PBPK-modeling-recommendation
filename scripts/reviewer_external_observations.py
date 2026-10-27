#!/usr/bin/env python3
"""Extract only frozen external-study scalar endpoints after prediction release.

The parent supplies the completed prediction-lock receipt and its SHA. The lock
and every path/SHA reference it contains are verified before any table value is
read. Metformin remains unsealed numerically because its dose schedule is blocked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/model_recommendation/reviewer_revision_20261009/external_validation"


def ref(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(source):
    path = (ROOT / source["path"]).resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file() or ref(path)["sha256"] != source["sha256"]:
        raise ValueError("reference mismatch or outside project")
    return path


def iter_refs(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("path"), str) and isinstance(obj.get("sha256"), str):
            yield obj
        for v in obj.values():
            yield from iter_refs(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_refs(v)


def text(node):
    return " ".join(" ".join(node.itertext()).split())


def number(cell):
    value = cell.replace(",", "").strip()
    m = re.match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$", value)
    if not m or (m[2] and not (m[2].startswith("±") or m[2].startswith("("))):
        raise ValueError("missing, censored or ambiguous scalar cell: " + cell)
    result = float(m[1])
    if not math.isfinite(result) or result <= 0:
        raise ValueError("nonpositive or nonfinite scalar cell")
    return result


def table_spec(scene):
    pmc = scene["pmcid"]
    if pmc == "PMC12511804":
        return "T2", 1, {"Cmax": 1, "Tmax": 2, "AUC_inf": 4, "terminal_half_life": 5}, "arithmetic_mean", "arithmetic_mean"
    if pmc == "PMC12640539":
        table = "TB1" if "studyA" in scene["scenario_id"] else "TB2"
        return table, 3, {"Cmax": 1, "AUC_inf": 3, "Tmax": 5, "terminal_half_life": 6}, "arithmetic_mean", "median"
    if pmc == "PMC13104799":
        offset = 24 if scene["compound"] == "rosuvastatin" else 0
        return "jcph70185-tbl-0003", 1, {"Cmax": offset, "AUC_inf": offset + 2, "terminal_half_life": offset + 3, "Tmax": offset + 5}, "geometric_mean", "median"
    raise ValueError("No authorized unblocked table specification for " + pmc)


def extract_scene(scene, source_path):
    table_id, column, row_indices, statistic, tmax_stat = table_spec(scene)
    root = ET.parse(source_path).getroot()
    table = root.find('.//table-wrap[@id="' + table_id + '"]')
    if table is None:
        raise ValueError("Frozen source table absent")
    body_rows = table.findall(".//table/tbody/tr")
    expected = {"Cmax": r"c\s*max", "Tmax": r"t\s*max", "AUC_inf": r"auc.*(?:inf|∞)", "terminal_half_life": r"t\s*(?:1/2|half)"}
    rows = []
    for endpoint, index in row_indices.items():
        if endpoint not in scene["planned_endpoint_definitions"]:
            raise ValueError("Endpoint not frozen before outcomes")
        cells = list(body_rows[index])
        label, original = text(cells[0]), text(cells[column])
        if not re.search(expected[endpoint], label, re.I):
            raise ValueError("Frozen row no longer matches endpoint: " + label)
        reason = None
        try:
            value = number(original)
        except ValueError as exc:
            value = None
            reason = str(exc)
        unit = "h" if endpoint in {"Tmax", "terminal_half_life"} else "ng/mL" if endpoint == "Cmax" else "ng*h/mL"
        actual_stat = tmax_stat if endpoint == "Tmax" else statistic
        column_label = ("Alone (mean ± SD)" if scene["pmcid"] == "PMC12511804" else
                        "Sildenafil FCT, fast (T3)" if scene["pmcid"] == "PMC12640539" and table_id == "TB1" else
                        "Sildenafil FCT, fed (R)" if scene["pmcid"] == "PMC12640539" else
                        scene["compound"].capitalize() + " alone, geometric mean (Tmax median)")
        rows.append({"scenario_id": scene["scenario_id"], "compound": scene["compound"], "study_id": scene["study_id"],
                     "selection_stratum": scene["selection_stratum"], "endpoint": endpoint, "value": value, "unit": unit,
                     "statistic": actual_stat, "matrix": "plasma", "analyte": "parent", "value_origin": "published_scalar_table",
                     "source_ref": ref(source_path), "table_id": table_id, "source_row_0based": index, "source_column_0based": column,
                     "row_label": label, "column_label": column_label,
                     "source_row_label": label, "source_cell": original,
                     "original_value": value, "original_unit": unit, "unit_conversion_factor": 1.0,
                     "qualification": "published_scalar_unit_and_analyte_matched" if value is not None else "unqualified_missing_or_censored",
                     "reason": reason,
                     "source_xpath": f'(.//table-wrap[@id="{table_id}"]//table/tbody/tr)[{index + 1}]/*[{column + 1}]',
                     "interpretation": "Published group NCA statistic; comparison with a reference individual remains a declared approximation, not a population-matched prediction."})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-lock", required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--release-confirmation", required=True, choices=["all-frozen-native-tasks-accounted-for"])
    args = parser.parse_args()
    lock = verify({"path": args.prediction_lock, "sha256": args.lock_sha256})
    release = json.loads(lock.read_text())
    refs = list(iter_refs(release))
    if not refs:
        raise ValueError("Prediction lock lacks verifiable artifact references")
    for source in refs:
        verify(source)
    selection_path = BASE / "selection_manifest.json"
    selection_ref = ref(selection_path)
    if not any(s["path"] == selection_ref["path"] and s["sha256"] == selection_ref["sha256"] for s in refs):
        raise ValueError("Prediction lock does not bind frozen clinical selection")
    selected = json.loads(selection_path.read_text())
    for source in selected["source_receipts"]:
        verify(source)
    amendment = json.loads((BASE / "selection_amendment_01.json").read_text())
    if amendment["decision"] != "conditions_blocked_for_strict_evaluation":
        raise ValueError("Metformin amendment changed; no automatic unsealing authorized")
    rows, blocked = [], []
    for scene in selected["selected_scenarios"]:
        if scene["scenario_id"] == amendment["scenario_id"]:
            blocked.append({"scenario_id": scene["scenario_id"], "reason": amendment["reason"], "observed_values_read": False})
            continue
        rows.extend(extract_scene(scene, BASE / "sources" / (scene["pmcid"] + ".xml")))
    receipt = {"schema": "reviewer-external-observations-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "prediction_lock": ref(lock), "selection": selection_ref, "condition_amendment": ref(BASE / "selection_amendment_01.json"),
               "release_confirmation": args.release_confirmation, "observations": rows, "blocked_scenarios": blocked,
               "sources": selected["source_receipts"],
               "endpoints_added_after_prediction_lock": False, "observations_used_to_refit": False,
               "strict_new_study_numbers_first_extracted_after_this_prediction_lock": True,
               "previous_inadvertent_exposure_preserved": selected["inadvertent_exposure"], "producer": ref(Path(__file__))}
    (BASE / "observations.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"observations": len(rows), "scenarios": len({r["scenario_id"] for r in rows}), "blocked": len(blocked), "output": ref(BASE / "observations.json")}))


if __name__ == "__main__":
    main()
