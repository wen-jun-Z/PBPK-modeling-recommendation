#!/usr/bin/env python3
"""Curate one external study only after its numerical predictions are locked.

Run with runtime/app-env/bin/python. No network, model fitting, LLM or GPU.
All numerical outcomes and extraction diagnostics stay in the sealed directory;
stdout and the public receipt contain identifiers, counts and checksums only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DESIGN_SHA = "0fe4c8924bcd8ddecd0a6a642bfc90e74e9ef68c98e3bdfbfff91690dd756a74"
PDF_SHA = "f110bbd82e618788c6264168946d26db6c93191ad717a12bd5691a4bf2a93284"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(path: str | Path) -> Path:
    p = Path(path)
    p = (ROOT / p if not p.is_absolute() else p).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("All sources, locks and destinations must stay in project")
    return p


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def validate_lock(design_path: Path, lock_path: Path) -> tuple[dict, dict]:
    if sha(design_path) != DESIGN_SHA:
        raise ValueError("Design changed after source curator freeze")
    design = json.loads(design_path.read_text())
    lock = json.loads(lock_path.read_text())
    if lock.get("design_sha256") != DESIGN_SHA:
        raise ValueError("Prediction lock does not bind the frozen external design")
    if lock.get("model_selection_frozen") is not True:
        raise ValueError("Model selection has not been explicitly frozen")
    instant = datetime.fromisoformat(lock["locked_at_utc"].replace("Z", "+00:00"))
    if instant.tzinfo is None or instant > datetime.now(timezone.utc):
        raise ValueError("Invalid lock timestamp")
    expected = {x["case_id"] for x in design["cases"]}
    seen = set()
    records = lock.get("predictions", [])
    if not records:
        raise ValueError("No actual prediction artifacts locked")
    for record in records:
        if record["case_id"] not in expected:
            raise ValueError("Prediction from an unexpected study case")
        p = local(record["path"])
        # Hash only: the curator must never inspect predicted concentrations.
        if not p.is_file() or not p.stat().st_size or sha(p) != record["sha256"]:
            raise ValueError("Missing or changed locked prediction artifact")
        seen.add(record["case_id"])
    if seen != expected:
        raise ValueError("Both prespecified external dose cases must be locked")
    return design, lock


def table_endpoints(document) -> tuple[list[dict], dict]:
    """Read native PDF text positions, retaining printed unit rows independently."""
    page = document[2]
    words = page.get_text("words")
    table_title = [w for w in words if w[4] == "Table" and
                   w[0] > page.rect.width / 2 and w[1] > page.rect.height * .6]
    if len(table_title) != 1:
        raise ValueError("Ambiguous Table 1 location")
    top = table_title[0][1]
    data = sorted([w for w in words if w[1] > top and w[0] > page.rect.width / 2], key=lambda w: (w[1], w[0]))
    lines: list[list] = []
    for word in data:
        center = (word[1] + word[3]) / 2
        target = next((line for line in lines if abs(line[0] - center) < 2), None)
        if target is None:
            lines.append([center, [word]])
        else:
            target[1].append(word)
    lines.sort(key=lambda line: line[0])
    line_rows = [{"y_center": y, "words": [list(w[:5]) for w in sorted(ws, key=lambda w: w[0])]} for y, ws in lines]
    # Native table has four adjacent numeric columns: caffeine and d9-caffeine
    # for each dose. The two caffeine column starts are derived from the Cmax
    # mass-unit row, not hard-coded observed values.
    mass_row = next((ws for y, ws in lines if any(w[4].startswith("Cmax") for w in ws) and any("ng/mL" in w[4] for w in ws)), None)
    if mass_row is None:
        raise ValueError("Cannot locate native Cmax mass-unit row")
    numbers = sorted([w for w in mass_row if _number(w[4]) is not None], key=lambda w: w[0])
    if len(numbers) != 4:
        raise ValueError("Expected four Cmax columns; refusing inferred replacement")
    starts = [w[0] for w in numbers]
    bounds = [-math.inf] + [(starts[i] + starts[i + 1]) / 2 for i in range(3)] + [math.inf]
    endpoints = []
    for name, marker, unit_label, unit in [
        ("cmax", "Cmax", "ng/mL", "ng/mL"),
        ("cmax", "Cmax", "nM)", "nmol/L"),
        ("auc_last", "AUClast", "h*ng/mL", "h*ng/mL"),
        ("auc_last", "AUClast", "h*nM)", "h*nmol/L"),
    ]:
        candidates = [(y, ws) for y, ws in lines if any(w[4] == marker for w in ws) and any(unit_label in w[4] for w in ws)]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous endpoint row {name}/{unit}")
        y, ws = candidates[0]
        # A CV may wrap onto the following line within the same cell. Stop at
        # the next first-column row label, not the next text baseline.
        label_limit = starts[0] - 10
        following_labels = [yy for yy, ww in lines if yy > y + 3 and
                            any(w[0] < label_limit for w in ww)]
        row_end_y = min(following_labels) if following_labels else y + 13
        for dose, column in [(50, 0), (250, 2)]:
            values = sorted([w for w in ws if _number(w[4]) is not None and bounds[column] <= w[0] < bounds[column + 1]], key=lambda w: w[0])
            if not values:
                raise ValueError("Missing printed endpoint; no inference permitted")
            token = values[0]
            cell_words = sorted([w for yy, ww in lines if y-2 <= yy < row_end_y-2
                                 for w in ww if bounds[column] <= w[0] < bounds[column+1]],
                                key=lambda w: (w[1], w[0]))
            cell_text = " ".join(w[4] for w in cell_words)
            cv_matches = re.findall(r"\(\s*([\d.]+)\s*\)", cell_text)
            canonical = unit in {"nmol/L", "h*nmol/L"}
            endpoints.append({
                "case_id": f"caffeine_sherman2022_solution_{dose}mg",
                "study_id": "caffeine_sherman2022", "endpoint": name,
                "value": _number(token[4]), "unit": unit,
                "source_type": "published_aggregate_table_transcription",
                "source_pdf_page": 3, "source_table": "1", "source_word_bbox": list(token[:4]),
                "source_token": token[4], "population_statistic": "geometric_mean",
                "cell_native_text": cell_text,
                "reported_cv_percent": float(cv_matches[0]) if len(cv_matches) == 1 else None,
                "cv_interpretation": "published percent CV; not a confidence interval",
                "analyzed_n": 9, "time_window_h": None if name == "auc_last" else "sampled_peak",
                "auc_definition": "to_last_measured_concentration; individual_tlast_not_reported" if name == "auc_last" else None,
                "canonical_unit_representation": canonical,
                "counts_as_unique_endpoint": canonical,
                "primary_comparable": name == "cmax" and canonical,
                "do_not_average_unit_rows": True,
            })
    # Independent transcription route: read the table in native PDF text order,
    # without any x/y cell assignments, then compare both parent-arm values.
    full_text = page.get_text()
    table_text = full_text[full_text.rindex("Table 1"):]
    endpoint_labels = {
        ("cmax", "ng/mL"): ("Cmax (ng/mL)", "Cmax (nM)"),
        ("cmax", "nmol/L"): ("Cmax (nM)", "Cmax_D"),
        ("auc_last", "h*ng/mL"): ("AUClast (h*ng/mL)", "AUClast (h*nM)"),
        ("auc_last", "h*nmol/L"): ("AUClast (h*nM)", "AUClast_D"),
    }
    for (endpoint, unit), (begin, end) in endpoint_labels.items():
        section = table_text.split(begin, 1)[1].split(end, 1)[0]
        second_values = re.findall(r"(?m)^\s*([\d,]+(?:\.\d+)?)\s*(?:\([^)]*\))?\s*$", section)
        if len(second_values) != 4:
            raise ValueError("Independent text-order transcription is ambiguous")
        for dose, column in [(50, 0), (250, 2)]:
            row = next(x for x in endpoints if x["endpoint"] == endpoint and x["unit"] == unit and
                       x["case_id"].endswith(f"_{dose}mg"))
            if float(second_values[column].replace(",", "")) != row["value"]:
                raise ValueError("Independent PDF transcription routes disagree")
            row["independent_text_order_value_check"] = "passed"
    return endpoints, {"page": 3, "table": "1", "native_text_rows": line_rows,
                       "independent_text_order_check": "passed_for_both_parent_arms_and_both_unit_rows",
                       "numeric_column_starts": starts, "numeric_column_boundaries": [None if not math.isfinite(x) else x for x in bounds],
                       "ordinary_caffeine_columns_zero_based": [0, 2],
                       "deuterated_columns_excluded_zero_based": [1, 3]}


def _number(token: str) -> float | None:
    # Parenthetical CV tokens must never be misread as endpoint values.
    if token.startswith("(") or token.endswith(")"):
        return None
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def longest_run(values):
    import numpy as np
    change = np.diff(np.r_[False, values, False].astype(int))
    starts, ends = np.where(change == 1)[0], np.where(change == -1)[0]
    if not len(starts):
        return 0, 0, 0
    k = int(np.argmax(ends - starts))
    return int(ends[k] - starts[k]), int(starts[k]), int(ends[k] - 1)


def curve_extract(document, dose: int, output: Path) -> tuple[list[dict], dict]:
    """Conservative template matching, never extrapolating an invisible point."""
    import numpy as np
    from PIL import Image
    from scipy.signal import correlate2d
    import pymupdf

    xref, ymax = ((60, 8000.0) if dose == 50 else (57, 50000.0))
    pix = pymupdf.Pixmap(document, xref)
    mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(pix.n)
    if mode is None:
        raise ValueError("Unsupported source image channels")
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("L")
    original = output / f"figure_{2 if dose == 50 else 4}_original.png"
    image.save(original)
    gray = np.asarray(image, dtype=float)
    dark = gray < 155
    h, w = dark.shape
    ys = range(int(h * .65), int(h * .88))
    axis_y, run_x = max(((y, longest_run(dark[y])) for y in ys), key=lambda pair: pair[1][0])
    xs = range(int(w * .10), int(w * .23))
    axis_x, run_y = max(((x, longest_run(dark[:axis_y + 1, x])) for x in xs), key=lambda pair: pair[1][0])
    top, right = run_y[1], run_x[2]
    if run_x[0] < 400 or run_y[0] < 400:
        raise ValueError("Cannot reliably locate source plot axes")
    x_scale, y_scale = (right - axis_x) / 50, (axis_y - top) / ymax
    # The parent legend marker is on the upper row of the right-side legend.
    # Geometry refers to a label glyph, not a known observation ordinate.
    nominal_legend = (863, 265) if dose == 50 else (865, 263)
    lx, ly = nominal_legend
    # Refine the label marker by maximizing vertical ink near its outline.
    legend_window = dark[ly - 12:ly + 13, lx - 12:lx + 13]
    col_scores = legend_window.sum(axis=0)
    strong = np.flatnonzero(col_scores >= max(5, .75 * col_scores.max()))
    if len(strong) >= 2:
        lx = nominal_legend[0] - 12 + float(strong.min() + strong.max()) / 2
    template_radius = 10
    lx, ly = int(round(lx)), int(round(ly))
    template = (255 - gray[ly - template_radius:ly + template_radius + 1,
                           lx - template_radius:lx + template_radius + 1]) / 255
    # Error bars and connecting lines traverse the glyph centre and can be
    # darker than the hollow parent marker. Match the four corner regions,
    # excluding the central horizontal/vertical strips, to distinguish the
    # parent square/circle from the comparator diamond/triangle.
    yy, xx = np.mgrid[-template_radius:template_radius+1, -template_radius:template_radius+1]
    mask = ((np.abs(xx) > 3) & (np.abs(yy) > 3)).astype(float)
    centered = (template - (template * mask).sum() / mask.sum()) * mask
    target = (255 - gray) / 255
    numerator = correlate2d(target, centered, mode="same", boundary="fill")
    ones = mask
    n = mask.sum()
    total = correlate2d(target, ones, mode="same", boundary="fill")
    squares = correlate2d(target * target, ones, mode="same", boundary="fill")
    denominator = np.sqrt(np.maximum(squares - total * total / n, 0) * np.sum(centered * centered))
    scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-8)
    all_times = [5/60, 10/60, 20/60, 40/60, 1, 1.5, 2, 3, 4, 8, 10, 12, 24, 36, 48]
    rows, observations = [], []
    for time in all_times:
        record = {"time_h": time, "source_type": "raster_digitization_attempt", "status": "unresolved"}
        if time < 8:
            record["reason"] = "early_marker_cluster_or_adjacent_series; no inferred deblending"
            rows.append(record)
            continue
        if time >= 36:
            record["reason"] = "late_near_baseline_or_lloq; zero/BLQ cannot be inferred from pixels"
            rows.append(record)
            continue
        expected_x = axis_x + x_scale * time
        xlo, xhi = int(round(expected_x)) - 3, int(round(expected_x)) + 4
        window = scores[top + 12:axis_y - 3, xlo:xhi]
        candidates = []
        masked = window.copy()
        for _ in range(8):
            yy, xx = np.unravel_index(np.argmax(masked), masked.shape)
            score = float(masked[yy, xx])
            candidates.append({"x": xlo + int(xx), "y": top + 12 + int(yy), "score": score})
            masked[max(0, yy-18):yy+19] = -1
        best, second = candidates[:2]
        record.update(expected_x=expected_x, search_x_bounds=[xlo, xhi], template_matches=candidates)
        if best["score"] < .68 or best["score"] - second["score"] < .08:
            record["reason"] = "marker_match_not_unique_or_below_similarity_threshold"
            rows.append(record)
            continue
        concentration = (axis_y - best["y"]) / y_scale
        # Two pixels of marker-center uncertainty plus axis calibration endpoints.
        uncertainty = 3 / y_scale
        if concentration <= 0 or uncertainty / concentration > .25:
            record["reason"] = "pixel_resolution_bound_exceeds_25_percent_of_estimate"
            rows.append(record)
            continue
        record.update(status="accepted_pending_visual_qc", x_pixel=best["x"], y_pixel=best["y"],
                      concentration_nmol_per_l=concentration, absolute_pixel_error_bound_nmol_per_l=uncertainty,
                      time_pixel_error_bound_h=2/x_scale,
                      error_bar_definition="unresolved; not used as SD/SE")
        observations.append({"case_id": f"caffeine_sherman2022_solution_{dose}mg", "study_id": "caffeine_sherman2022",
                             "time_h": time, "concentration": concentration / 1000, "unit": "umol/L",
                             "absolute_digitization_error_bound": uncertainty / 1000,
                             "source_type": "digitized_published_group_mean_not_original_numeric_table",
                             "statistic": "published_mean_uncertainty_type_not_defined",
                             "source_figure": "2" if dose == 50 else "4", "source_page": 4,
                             "eligible_for_scoring": False, "requires_visual_qc": True})
        rows.append(record)
    audit = {"dose_mg": dose, "image_file": original.name, "image_sha256": sha(original),
             "native_resolution": [w, h], "pdf_image_xref": xref,
             "axes": {"origin_x": axis_x, "origin_y": axis_y, "top_y": top, "right_x": right,
                      "x_axis_range_h": [0, 50], "y_axis_range_nmol_per_l": [0, ymax],
                      "pixels_per_hour": x_scale, "pixels_per_nmol_per_l": y_scale,
                      "formula": "t=(x-origin_x)/pixels_per_hour; c=(origin_y-y)/pixels_per_nmol_per_l"},
             "legend_template_box": [lx-10, ly-10, lx+11, ly+11],
             "matching_rule": {"ncc_minimum": .68, "ncc_margin_minimum": .08,
                               "template_mask": "four corner regions; exclude abs(dx)<=3 or abs(dy)<=3 to suppress crossing error bars/lines",
                               "maximum_relative_pixel_error_bound": .25},
             "points": rows, "visual_qc_complete": False,
             "excluded_points_retained": True, "interpolation_used": False}
    return observations, audit


def finalize_review(output: Path, review_path: Path, lock_path: Path) -> dict:
    review = json.loads(review_path.read_text())
    audit_path = output / "extraction_audit.json"
    if review["source_audit_sha256"] != sha(audit_path):
        raise ValueError("Visual review does not bind the exact automatic proposals")
    audit = json.loads(audit_path.read_text())
    observation_path = output / "observations.json"
    observations = json.loads(observation_path.read_text())
    if observations["prediction_lock_sha256"] != sha(lock_path):
        raise ValueError("Observation/prediction-lock mismatch")
    points = []
    for decision in review["point_reviews"]:
        curve = next(x for x in audit["curves"] if x["dose_mg"] == decision["dose_mg"])
        row = next(x for x in curve["points"] if x["time_h"] == decision["time_h"])
        candidate = row["template_matches"][decision["selected_proposal_index"]]
        axes = curve["axes"]
        c = (axes["origin_y"] - candidate["y"]) / axes["pixels_per_nmol_per_l"]
        bound = 3 / axes["pixels_per_nmol_per_l"]
        if c <= 0 or bound / c > .25:
            raise ValueError("Visual symbol approval cannot override resolution bound")
        if decision["parent_symbol_distinct_from_comparator"] is not True:
            raise ValueError("Unresolved drug identity must never become an observation")
        row["automatic_decision"] = {k: row.get(k) for k in ["status", "reason"]}
        row.update(status="accepted_after_visual_symbol_qc", visual_review=decision,
                   x_pixel=candidate["x"], y_pixel=candidate["y"],
                   absolute_pixel_error_bound_nmol_per_l=bound,
                   time_pixel_error_bound_h=2/axes["pixels_per_hour"],
                   reviewed_box=[candidate["x"]-15, candidate["y"]-15, candidate["x"]+16, candidate["y"]+16])
        points.append({"case_id": f"caffeine_sherman2022_solution_{curve['dose_mg']}mg",
                       "study_id": "caffeine_sherman2022", "time_h": row["time_h"],
                       "concentration": c/1000, "unit": "umol/L",
                       "absolute_digitization_error_bound": bound/1000,
                       "time_digitization_error_bound_h": row["time_pixel_error_bound_h"],
                       "source_type": "automatic_coordinate_proposal_with_AI_visual_symbol_review; not_original_numeric_table",
                       "statistic": "published_mean; arithmetic interpretation assumed, not explicitly defined in paper",
                       "source_figure": "2" if curve["dose_mg"] == 50 else "4", "source_page": 4,
                       "eligible_for_scoring": True, "eligible_as_primary_full_curve_evidence": False,
                       "requires_visual_qc": False, "source_image_sha256": curve["image_sha256"]})
    for curve in audit["curves"]:
        curve["visual_qc_complete"] = True
    observations["partial_curve_points"] = points
    observations["endpoint_transcription_visual_qc_complete"] = True
    observations["quality_review"] = {
        "review_file": str(review_path.relative_to(ROOT)), "review_sha256": sha(review_path),
        "actor": "AI data curator; no claim of independent human labeling",
        "endpoint_checks": "native-PDF coordinate transcription versus native-text-order transcription, plus source table/column/footnote review",
        "curve_checks": "reviewed parent glyphs against source legend and competing glyphs; retained all rejected proposals and unresolved sampling times",
        "clinical_error_bar_type": "unknown; no clinical confidence intervals inferred",
        "partial_curve_mean_kind": "arithmetic interpretation is an explicit evaluation approximation"}
    audit["visual_review_file"] = str(review_path.relative_to(ROOT))
    write_json(audit_path, audit)
    write_json(observation_path, observations)
    receipt = {"schema_version": 1, "study_id": "caffeine_sherman2022",
               "case_ids": sorted({e["case_id"] for e in observations["endpoint_rows"]}),
               "design_sha256": DESIGN_SHA, "prediction_lock_sha256": sha(lock_path),
               "curator_script_sha256": sha(Path(__file__)),
               "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "sealed_output": str(output.relative_to(ROOT)),
               "endpoint_rows_including_duplicate_unit_representations": len(observations["endpoint_rows"]),
               "unique_endpoint_count": sum(e["canonical_unit_representation"] for e in observations["endpoint_rows"]),
               "accepted_partial_curve_points": len(points),
               "unresolved_curve_points": sum(x["status"] == "unresolved" for c in audit["curves"] for x in c["points"]),
               "curator_had_viewed_published_values_during_prior_source_audit": True,
               "numeric_outcome_values_shared_with_prediction_runner": False,
               "status": "curation_complete", "full_curve_available": False,
               "files": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in sorted(output.iterdir()) if p.is_file()]}
    write_json(output.parent / "curation_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", required=True)
    parser.add_argument("--prediction-lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--visual-review", help="Finalize an existing sealed extraction using an explicit reviewed proposal manifest")
    args = parser.parse_args()
    design_path, lock_path, output = local(args.design), local(args.prediction_lock), local(args.output)
    design, _ = validate_lock(design_path, lock_path)
    if args.visual_review:
        print(json.dumps(finalize_review(output, local(args.visual_review), lock_path), ensure_ascii=False))
        return
    if output.exists():
        raise ValueError("Never overwrite an existing sealed extraction")
    source = local(design["publication"]["source_pdf"])
    if sha(source) != PDF_SHA:
        raise ValueError("Source PDF bytes differ from audited publication")
    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    import pymupdf
    document = pymupdf.open(source)
    endpoints, table_audit = table_endpoints(document)
    curves, curve_audits = [], []
    for dose in [50, 250]:
        points, audit = curve_extract(document, dose, output)
        curves.extend(points)
        curve_audits.append(audit)
    write_json(output / "observations.json", {
        "study_id": design["study_id"], "source_pdf_sha256": PDF_SHA,
        "design_sha256": DESIGN_SHA, "prediction_lock_sha256": sha(lock_path),
        "endpoint_rows": endpoints, "partial_curve_points": curves,
        "source_type": "curated_published_summary_data_not_individual_raw_data",
        "endpoint_transcription_visual_qc_complete": False,
    })
    write_json(output / "extraction_audit.json", {"table": table_audit, "curves": curve_audits})
    receipt = {"schema_version": 1, "study_id": design["study_id"],
               "case_ids": [c["case_id"] for c in design["cases"]],
               "design_sha256": DESIGN_SHA, "prediction_lock_sha256": sha(lock_path),
               "curator_script_sha256": sha(Path(__file__)),
               "curator_had_viewed_published_values_during_prior_source_audit": True,
               "numeric_outcome_values_shared_with_prediction_runner": False,
               "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "sealed_output": str(output.relative_to(ROOT)),
               "endpoint_rows_including_duplicate_unit_representations": len(endpoints),
               "partial_curve_points_pending_qc": len(curves),
               "status": "extracted_pending_independent_transcription_and_visual_qc",
               "files": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in sorted(output.iterdir())]}
    write_json(output.parent / "curation_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
