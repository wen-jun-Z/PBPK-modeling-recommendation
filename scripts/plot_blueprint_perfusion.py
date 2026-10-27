#!/usr/bin/env python3
"""Plot a completed perfusion/transport comparison without reading fit data.

Usage:
    runtime/app-env/bin/python scripts/plot_blueprint_perfusion.py --run-dir RUN_DIRECTORY

Only results.json/results.csv are used for plotting; state.json supplies the
completion gate and results checksum. No engine, prediction, clinical source,
or fitting module is imported. The sole experiment output is comparison.pdf.
PNG previews and Matplotlib's cache stay under the project runtime/cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "experiments/blueprint_design/architecture_20260930/clinical_perfusion_v1"
CACHE = ROOT / "runtime/cache/perfusion_figure"
METHODS = (
    ("old_trust_region", "PBPK: original trust region"),
    ("clearance_only", "PBPK: CL"),
    ("extent_clearance", "PBPK: CL + partition"),
    ("extent_exchange_clearance", "PBPK: CL + partition + exchange"),
    ("refined_axes", "PBPK: refined three-axis search"),
    ("two_compartment", "Two compartments"),
    ("two_compartment_prior", "Two compartments + CL prior"),
    ("perfusion_global", "Perfusion: global partition"),
    ("perfusion_grouped", "Perfusion: grouped partition"),
    ("flow_exchange", "PBPK: finite tissue exchange (new)"),
)
SAMPLING = (("full", "Full development curve"), ("five", "Five development points"))
SIZE_INCHES = (7.2, 5.6)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result) or result < 1. - 1e-12:
        raise ValueError("A curve GMFE must be finite and at least one")
    return result


def same_value(first, second, label):
    first, second = number(first), number(second)
    if (first is None) != (second is None) or (
            first is not None and not math.isclose(first, second, rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError("Inconsistent completed results: " + label)


def selected_methods(document):
    """Use the declared results matrix, preserving the same semantic row order."""
    groups = document["statistics"]["by_sampling"]
    method_ids = set(groups["full"])
    known = {name for name, _ in METHODS}
    if (set(groups) != {name for name, _ in SAMPLING}
            or set(groups["five"]) != method_ids or not method_ids <= known
            or not known - {"flow_exchange"} <= method_ids):
        raise ValueError("Comparison must preserve the prior methods in both sampling subsets")
    transport = "flow_exchange" in method_ids
    if document["version"] == "blueprint-transport-pilot-1" and not transport:
        raise ValueError("Transport results must include the finite-exchange method")
    return tuple((name, label + (" (new)" if not transport and name.startswith("perfusion_") else ""))
                 for name, label in METHODS if name in method_ids)


def load_results(run):
    """Validate complete case coverage, failures, and JSON/CSV agreement."""
    paths = [run / name for name in ("state.json", "results.json", "results.csv")]
    state = json.loads(paths[0].read_text())
    if state.get("status") != "complete":
        raise ValueError("Wait for state.status == complete before plotting")
    pins = {path: sha256(path) for path in paths}
    if state.get("results", {}).get("sha256") != pins[paths[1]]:
        raise ValueError("Completed state does not seal this results.json")
    document = json.loads(paths[1].read_text())
    if document.get("version") not in {"blueprint-perfusion-pilot-1", "blueprint-transport-pilot-1"}:
        raise ValueError("Unsupported perfusion/transport comparison results version")
    if document.get("all_attempts_locked_before_scoring") is not True:
        raise ValueError("Prediction-lock evidence is missing")
    records = document["units"]
    methods = selected_methods(document)
    method_ids = {name for name, _ in methods}
    cases, drug_cases, lookup = {}, {}, {}
    for record in records:
        key = (record["sampling"], record["method"], record["case_id"])
        if key in lookup:
            raise ValueError("Duplicate method/sampling/case record")
        if record["method"] not in method_ids or record["sampling"] not in {"full", "five"}:
            raise ValueError("Unexpected method or development sampling condition")
        case = (record["drug"], record["study_id"], record["cluster_id"])
        if cases.setdefault(record["case_id"], case) != case:
            raise ValueError("Case identity changed across methods")
        if drug_cases.setdefault(record["drug"], record["case_id"]) != record["case_id"]:
            raise ValueError("This figure requires one study case per drug")
        value = number(record.get("curve_metrics", {}).get("gmfe"))
        if value is not None and (record["attempt_status"] != "predicted" or record["scoring_status"] != "scored"):
            raise ValueError("Unsuccessful record has a scored curve GMFE")
        lookup[key] = record
    if len(cases) != 4 or len({case[2] for case in cases.values()}) != 4:
        raise ValueError("This pilot figure requires four distinct drug/study clusters")
    expected = {(s, m, c) for s, _ in SAMPLING for m, _ in methods for c in cases}
    if set(lookup) != expected:
        raise ValueError("Every declared method and both subsets must retain every case")
    with paths[2].open(newline="") as handle:
        table = list(csv.DictReader(handle))
    aggregate = {(row["sampling"], row["method"]): row for row in table}
    if len(aggregate) != len(table) or set(aggregate) != {(s, m) for s, _ in SAMPLING for m, _ in methods}:
        raise ValueError("Aggregate CSV comparison matrix is incomplete or duplicated")
    ordered_cases = sorted(cases, key=lambda identity: cases[identity][0])
    summaries, eda = {}, []
    for sampling, _ in SAMPLING:
        for method, _ in methods:
            selected = [lookup[sampling, method, cid] for cid in ordered_cases]
            values = [number(row.get("curve_metrics", {}).get("gmfe")) for row in selected]
            available = [value for value in values if value is not None]
            mean = math.exp(math.fsum(map(math.log, available)) / len(available)) if available else None
            reported = document["statistics"]["by_sampling"][sampling][method]["curve"]
            row = aggregate[sampling, method]
            same_value(mean, reported["estimate"], "case geometric mean versus JSON")
            same_value(mean, row["curve_gmfe"], "case geometric mean versus CSV")
            failed = sum(item["attempt_status"] == "failed" for item in selected)
            if (int(row["scored_cases"]) != len(available) or int(row["attempted_cases"]) != 4
                    or int(row["failed_cases"]) != failed or reported["scored"] != len(available)):
                raise ValueError("Aggregate coverage disagrees with individual cases")
            # An incomplete cohort must not look comparable to a four-case mean.
            summaries[sampling, method] = mean if len(available) == 4 else None
            for record, value in zip(selected, values):
                eda.append({"sampling": sampling, "method": method, "drug": record["drug"],
                            "curve_gmfe": value})
    return document, lookup, ordered_cases, cases, summaries, eda, pins


def draw(loaded):
    import matplotlib.pyplot as plt
    from matplotlib import colors, ticker
    import numpy as np

    document, lookup, case_ids, cases, summaries, _, _ = loaded
    methods = selected_methods(document)
    n_methods = len(methods)
    transport = any(name == "flow_exchange" for name, _ in methods)
    def family(name):
        if name.startswith("two_compartment"):
            return "classical"
        if name.startswith("perfusion_"):
            return "perfusion"
        return "transport" if name == "flow_exchange" else "native"
    boundaries = [i for i in range(1, n_methods) if family(methods[i - 1][0]) != family(methods[i][0])]
    finite = [number(record.get("curve_metrics", {}).get("gmfe")) for record in lookup.values()]
    maximum = max([value for value in finite if value is not None] or [2.])
    upper = max(2., 2. ** math.ceil(math.log2(maximum)))
    norm, cmap = colors.LogNorm(vmin=1., vmax=upper), plt.get_cmap("cividis_r").copy()
    cmap.set_bad("#e5e7eb")
    fig = plt.figure(figsize=SIZE_INCHES, layout=None)
    grid = fig.add_gridspec(1, 2, left=.302, right=.925, bottom=.355, top=.845, wspace=.13)
    axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    for panel, (sampling, title) in enumerate(SAMPLING):
        ax = axes[panel]
        values = np.full((n_methods, 5), np.nan)
        text_labels = []
        for i, (method, _) in enumerate(methods):
            labels = []
            for j, cid in enumerate(case_ids):
                record = lookup[sampling, method, cid]
                value = number(record.get("curve_metrics", {}).get("gmfe"))
                if value is not None:
                    values[i, j] = value
                labels.append(f"{value:.2f}" if value is not None else
                              "FAIL" if record["attempt_status"] == "failed" else "NA")
            mean = summaries[sampling, method]
            if mean is not None:
                values[i, -1] = mean
            labels.append(f"{mean:.2f}" if mean is not None else "NA")
            text_labels.append(labels)
        mesh = ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                             edgecolors="white", linewidth=.6, rasterized=False)
        ax.set(xlim=(0, 5), ylim=(n_methods, 0))
        ax.set_xticks(np.arange(5) + .5,
            [cases[cid][0] for cid in case_ids] + ["Geomean (n = 4)"])
        ax.tick_params(axis="x", length=0, pad=4)
        plt.setp(ax.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor", fontsize=7)
        ax.set_yticks(np.arange(n_methods) + .5, [label for _, label in methods] if panel == 0 else [""] * n_methods)
        ax.tick_params(axis="y", length=0, pad=7, labelsize=7)
        if panel == 0:
            for label in ax.get_yticklabels():
                if label.get_text().endswith("(new)"):
                    label.set_fontweight("bold")
        ax.set_title(title, loc="left", fontsize=8.5, pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for boundary in boundaries:
            ax.axhline(boundary, color="white", lw=2.8)
            ax.axhline(boundary, color="#888888", lw=.4)
        ax.axvline(4., color="white", lw=2.8)
        ax.axvline(4., color="#777777", lw=.6)
        for i, row in enumerate(text_labels):
            for j, label in enumerate(row):
                value = values[i, j]
                rgb = cmap(norm(value))[:3] if np.isfinite(value) else (1., 1., 1.)
                luminance = sum(a * b for a, b in zip(rgb, (.2126, .7152, .0722)))
                ax.text(j + .5, i + .5, label, ha="center", va="center", fontsize=7.1,
                         color="white" if luminance < .47 else "#171717",
                         fontweight="bold" if j == 4 else "normal")
    # Panel letters use the same figure coordinates, independent of tick extents.
    for letter, ax in zip(("a", "b"), axes):
        fig.text(ax.get_position().x0 - .023, .879, letter, weight="bold", fontsize=10)
    cax = fig.add_axes([.947, .355, .014, .49])
    bar = fig.colorbar(mesh, cax=cax)
    bar.solids.set_rasterized(False)
    ticks = [2. ** i for i in range(int(round(math.log2(upper))) + 1)]
    bar.set_ticks(ticks)
    bar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    bar.ax.minorticks_off()
    bar.ax.tick_params(labelsize=7, length=2, pad=2)
    bar.outline.set_linewidth(.4)
    fig.text(.03, .965, "Target-curve prediction error", fontsize=12, weight="bold", va="top")
    fig.text(.03, .918, "Curve GMFE: lower is better  |  Shared logarithmic color scale", fontsize=8, va="top")
    caption = [
        "Known four-drug development cohort; previous target results informed model development.",
        "This is not blinded validation. Each cell is one drug/study; geomean weights all four drugs equally.",
        "Full and five refer to development-curve sampling; both are scored on the same target curves.",
        "Moxifloxacin retains a PK prior from its development study; five points are not its full information budget.",
        ("Finite exchange fits CL, partition and tissue exchange. All previous predictions are reused." if transport else
         "Global/grouped perfusion fit CL plus one/two tissue-partition scales. Historical predictions are reused."),
        "GMFE = exp(mean |ln(predicted / observed)|). FAIL/NA remain visible; incomplete means are NA.",
    ]
    for index, line in enumerate(caption):
        fig.text(.03, .18 - .025 * index, line, fontsize=6.8, va="top")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, help="Experiment directory (positional compatibility)")
    parser.add_argument("--run-dir", type=Path, help="Experiment directory containing complete results")
    parser.add_argument("--preview-only", action="store_true", help="Run EDA and QA before exporting the PDF")
    parser.add_argument("--skill-dir", type=Path,
                        default=Path.home() / ".codex/skills/scipilot-figure-skill")
    args = parser.parse_args()
    if args.run is not None and args.run_dir is not None:
        parser.error("Use either the positional directory or --run-dir, not both")
    run = (args.run_dir or args.run or DEFAULT_RUN).resolve()
    loaded = load_results(run)
    CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(CACHE / "mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image
    skill_scripts = args.skill_dir.resolve() / "scripts"
    if not skill_scripts.is_dir():
        raise ValueError("SciPilot figure skill is required; supply --skill-dir")
    sys.path.insert(0, str(skill_scripts))
    from profile_data import profile_data
    from setup_style import setup_style
    from visual_qa import audit_layout, render_preview
    profile = profile_data(pd.DataFrame(loaded[5]), group_cols=["sampling", "method"])
    print(json.dumps({"EDA": {"comparison_cells": profile["n_rows"], "independent_drug_study_clusters": 4,
        "cases_per_method_and_subset": 4, "methods": len(selected_methods(loaded[0])),
        "curve_gmfe": profile["columns"]["curve_gmfe"],
        "display": "Two annotated matrices retain every drug; no inferential interval is plotted"}},
        indent=2, default=lambda value: value.item()))
    setup_style(journal="nature", lang="en", use_sciplots=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.constrained_layout.use": False,
                         "savefig.bbox": None, "pdf.fonttype": 42})
    fig = draw(loaded)
    preview = CACHE / "comparison_preview.png"
    render_preview(fig, str(preview), dpi=180)
    with Image.open(preview) as original:
        original.convert("L").save(CACHE / "comparison_grayscale.png")
    issues = audit_layout(fig)
    print(json.dumps({"layout_issues": issues, "preview": str(preview)}, ensure_ascii=False))
    if any(severity == "FAIL" for severity, _ in issues):
        raise ValueError("Figure layout QA failed; inspect the temporary preview")
    for path, digest in loaded[-1].items():
        if sha256(path) != digest:
            raise ValueError("Completed results changed during plotting")
    if not args.preview_only:
        output = run / "comparison.pdf"
        temporary = CACHE / "comparison.pdf"
        fig.savefig(temporary, format="pdf", bbox_inches=None,
                    metadata={"Title": "Four-drug physiological architecture comparison",
                              "Subject": "Previously viewed development cohort; not blinded validation"})
        # Atomic publication leaves results and all immutable experiment inputs untouched.
        temporary.replace(output)
        print(json.dumps({"pdf": str(output), "size_inches": SIZE_INCHES}))
    plt.close(fig)


if __name__ == "__main__":
    main()
