#!/usr/bin/env python3
"""One English comparison figure from a COMPLETED sequential-sampling pilot.

Preview: runtime/app-env/bin/python -B scripts/plot_sequential_sampling_design.py
After visual review, repeat with --export for comparison.pdf and comparison.png.
Only state.json is read until completion has been verified. No model, fitter,
clinical source, acquisition receipt or target-value scoring function is imported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "experiments/blueprint_design/sequential_sampling_20261008"
SKILL = Path("/home/xuyiqian/.codex/skills/scipilot-figure-skill/scripts")
METHODS = ("refit_goal", "posterior_variance", "c_optimal", "time_farthest", "random")
RAW_METHODS = METHODS[:4] + ("random_0", "random_1", "random_2")
STYLE = {
    "refit_goal": ("Refit goal (ours)", "#009E73", "o", "-"),
    "posterior_variance": ("Expected variance", "#0072B2", "s", "--"),
    "c_optimal": ("Official c-opt", "#D55E00", "^", "-."),
    "time_farthest": ("Time coverage", "#CC79A7", "D", ":"),
    "random": ("Random (3 seeds)", "#606970", "v", (0, (5, 2))),
}
FIGSIZE = (7.2, 6.8)


def _load(path):
    return json.loads(Path(path).read_text())


def _valid(value):
    return value is not None and math.isfinite(value) and value > 0


def _gm(values):
    return math.exp(float(np.mean(np.log(values)))) if values and all(_valid(v) for v in values) else None


def _same(actual, expected, label):
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise ValueError(f"CSV/results mismatch for {label}: {actual} vs {expected}")
    elif not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"CSV/results mismatch for {label}: {actual} vs {expected}")


def load_complete(run_dir):
    """Gate on completion BEFORE opening any results or endpoint observations."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(ROOT):
        raise ValueError("Run directory must remain inside this project")
    state = _load(run_dir / "state.json")
    if state.get("status") != "complete":
        raise ValueError("The run is not complete; target results will not be read")
    result_path = run_dir / "results.json"
    source = state.get("results", {})
    if source.get("sha256") != hashlib.sha256(result_path.read_bytes()).hexdigest():
        raise ValueError("Completed result hash does not match state.json")
    result = _load(result_path)
    if not result["prediction_lock"] <= result["scored_at"]:
        raise ValueError("Target scoring precedes the prediction lock")
    with (run_dir / "endpoints.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(r["case_id"], r["layout"], r["method"], r["budget"], r["endpoint"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate endpoint records")
    return result, rows


def aggregate(result, rows):
    """Recompute drug-equal statistics, independently of the frozen reporter."""
    names = {}
    cmax = {}
    for row in rows:
        cid = row["case_id"]
        if cid in names and names[cid] != row["drug"]:
            raise ValueError("A case has inconsistent drug names")
        names[cid] = row["drug"]
        if row["endpoint"] == "cmax":
            key = (cid, row["layout"], row["method"], None if not row["budget"] else int(row["budget"]))
            if key in cmax:
                raise ValueError("Duplicate Cmax record")
            cmax[key] = None if not row["error"] else float(row["error"])
    if len(names) != 10:
        raise ValueError("This prespecified figure requires all ten drug records")
    case_ids = sorted(names, key=lambda cid: (names[cid].lower(), cid))
    layouts = {cid: sorted({k[1] for k in cmax if k[0] == cid and k[1] != "full"}) for cid in case_ids}
    if any(len(v) != 2 for v in layouts.values()):
        raise ValueError("Exactly two layouts per drug are required")
    expected = {(cid, layout, method, budget) for cid in case_ids for layout in layouts[cid]
                for method in RAW_METHODS for budget in range(4)}
    expected |= {(cid, "full", "full_reference", None) for cid in case_ids}
    if set(cmax) != expected:
        raise ValueError("Incomplete or unexpected Cmax comparison cells")
    for cid in case_ids:
        for layout in layouts[cid]:
            for method in RAW_METHODS[1:]:
                _same(cmax[cid, layout, method, 0], cmax[cid, layout, RAW_METHODS[0], 0],
                      f"{cid}/{layout} shared no-addition baseline")
    by_drug, curves, counts = {}, {}, {}
    for method in METHODS:
        raw = ("random_0", "random_1", "random_2") if method == "random" else (method,)
        curves[method], counts[method] = [], []
        for budget in (0, 1, 2, 3, "mean_acquired"):
            bs = (1, 2, 3) if budget == "mean_acquired" else (budget,)
            values = {cid: _gm([cmax[cid, layout, m, b] for layout in layouts[cid] for m in raw for b in bs])
                      for cid in case_ids}
            by_drug[method, budget] = values
            usable = [v for v in values.values() if _valid(v)]
            mean = _gm(usable)
            official = result["summary"][str(budget)][method]["cmax"]
            _same(mean, official["estimate"], f"{method}/{budget}/Cmax")
            if official["n_cases"] != len(usable):
                raise ValueError("Drug denominator differs from completed report")
            if isinstance(budget, int):
                curves[method].append(mean)
                counts[method].append(len(usable))
    ratios = {}
    for cid in case_ids:
        numerator = by_drug["refit_goal", "mean_acquired"][cid]
        denominator = by_drug["time_farthest", "mean_acquired"][cid]
        ratios[cid] = numerator / denominator if _valid(numerator) and _valid(denominator) else None
    pair = result["comparisons"]["mean_acquired"]["cmax"]["time_farthest"]
    _same(_gm([v for v in ratios.values() if _valid(v)]), pair["estimate"], "paired drug-equal ratio")
    for cid in case_ids:
        _same(ratios[cid], pair["per_case_ratio"].get(cid), f"{cid} paired ratio")
    # Reconstruct first attainment from endpoint errors. NULL/failed sequences
    # remain in the denominator, including unavailable full-information errors.
    first = {}
    for cid in case_ids:
        full = cmax[cid, "full", "full_reference", None]
        for layout in layouts[cid]:
            for method in RAW_METHODS:
                reached = [b for b in range(4) if _valid(cmax[cid, layout, method, b]) and _valid(full)
                           and cmax[cid, layout, method, b] <= 2
                           and cmax[cid, layout, method, b] <= 1.1 * full]
                first[cid, layout, method] = min(reached) if reached else None
    official_first = {(r["case_id"], r["layout"], r["method"]): r["first_budget"] for r in result["efficiency"]}
    if len(official_first) != len(result["efficiency"]) or first != official_first:
        raise ValueError("CSV first-attainment reconstruction differs from results.json")
    efficiency = {}
    for method in METHODS:
        raw = ("random_0", "random_1", "random_2") if method == "random" else (method,)
        efficiency[method] = [100 * float(np.mean([
            np.mean([first[cid, layout, m] is not None and first[cid, layout, m] <= budget
                     for layout in layouts[cid] for m in raw]) for cid in case_ids])) for budget in range(4)]
    return dict(case_ids=case_ids, names=names, curves=curves, counts=counts,
                ratios=ratios, paired=pair, efficiency=efficiency)


def make_figure(data):
    """The three questions and method set are fixed before reading outcomes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
    from setup_style import setup_style

    setup_style(journal="nature", lang="en", use_sciplots=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "xtick.major.width": .6, "ytick.major.width": .6,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "figure.constrained_layout.use": False})
    fig = plt.figure(figsize=FIGSIZE)
    axa = fig.add_axes((.105, .555, .375, .275))
    axb = fig.add_axes((.735, .20, .16, .63))
    axc = fig.add_axes((.105, .20, .375, .23))
    for ax in (axa, axc):
        ax.set_xlim(-.12, 3.12)
        ax.set_xticks(range(4))
        ax.set_xlabel("Additional observed points")
        ax.grid(axis="y", color="#E5E9EC", linewidth=.55, zorder=0)
    handles = []
    for method in METHODS:
        label, color, marker, linestyle = STYLE[method]
        kwargs = dict(color=color, marker=marker, linestyle=linestyle,
                      linewidth=1.2, markersize=4.0, markerfacecolor="white", markeredgewidth=.9)
        ys = [np.nan if v is None else v for v in data["curves"][method]]
        axa.plot(range(4), ys, **kwargs)
        axc.plot(range(4), data["efficiency"][method], clip_on=False, **kwargs)
        handles.append(Line2D([], [], label=label, **kwargs))
    axa.set_title("Prediction error", loc="left", pad=10)
    axa.set_ylabel(r"C$_{\mathrm{max}}$ GMFE (zoomed)")
    available = [v for vs in data["curves"].values() for v in vs if _valid(v)]
    if not available:
        raise ValueError("No finite Cmax GMFE values are available for panel a")
    lower, upper = min(available), max(available)
    margin = .125 * (upper - lower) if upper > lower else max(.005, .01 * lower)
    axa.set_ylim(max(0., lower - margin), upper + margin)
    axa.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    axa.yaxis.set_minor_locator(NullLocator())
    axa.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axc.set_title("Data efficiency (post hoc)", loc="left", pad=10)
    axc.set_ylabel("Ever reached criterion\nby budget (%)")
    axc.set_ylim(0, 103)
    axc.set_yticks((0, 25, 50, 75, 100))
    axb.set_title("Drug-level effect", loc="left", pad=10)
    axb.axvline(1., color="#9DA8AD", linewidth=.9, linestyle="--", zorder=0)
    ids = data["case_ids"]
    positions = list(range(len(ids), 0, -1))
    for cid, y in zip(ids, positions):
        value = data["ratios"][cid]
        if _valid(value):
            axb.plot(value, y, "o", color=STYLE["refit_goal"][1], markersize=4.5)
            axb.text(1.14, y, f"{100 * (value - 1):+.2f}", transform=axb.get_yaxis_transform(),
                     fontsize=7, ha="left", va="center", color="#4B5B65", clip_on=False)
        else:
            axb.text(1., y, "NA", color="#606970", fontsize=7, va="center", ha="center")
    pair = data["paired"]
    ci = pair.get("ci95")
    if ci:
        axb.hlines(0, *ci, color="#263F4C", linewidth=1.3)
    if _valid(pair["estimate"]):
        axb.plot(pair["estimate"], 0, "D", color="#263F4C", markersize=5)
        axb.text(1.14, 0, f"{100 * (pair['estimate'] - 1):+.2f}", transform=axb.get_yaxis_transform(),
                 fontsize=7, ha="left", va="center", color="#263F4C", weight="bold", clip_on=False)
    axb.text(1.14, 1.01, "Δ FE (%)", transform=axb.transAxes, fontsize=6.5, ha="left", va="bottom")
    axb.axhline(.5, color="#D8DFE3", linewidth=.7)
    axb.set_yticks(positions + [0], [data["names"][cid] for cid in ids] + ["All drugs"])
    axb.tick_params(axis="y", length=0, pad=6)
    axb.set_ylim(-.65, len(ids) + .65)
    axb.set_xscale("log", base=2)
    limits = [v for v in data["ratios"].values() if _valid(v)] + (ci or [])
    half_range = max(.04, 1.2 * max([abs(math.log2(v)) for v in limits] or [1.]))
    axb.set_xlim(2 ** (-half_range - .02), 2 ** (half_range + .02))
    exponent_ticks = [-half_range, 0, half_range]
    axb.xaxis.set_major_locator(FixedLocator([2. ** k for k in exponent_ticks]))
    axb.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}".rstrip("0").rstrip(".")))
    axb.xaxis.set_minor_locator(NullLocator())
    axb.set_xlabel("FE ratio: refit / time\n(log scale; lower favors refit)", labelpad=6)
    for label, x, y in (("a", .015, .85), ("b", .565, .85), ("c", .015, .45)):
        fig.text(x, y, label, weight="bold", fontsize=11, va="bottom")
    fig.text(.5, .988, "Sequential endpoint-directed PBPK sampling", ha="center", va="top",
             fontsize=11, weight="bold", color="#263F4C")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.51, .949),
               ncol=3, frameon=False, fontsize=7.5, columnspacing=1.35, handlelength=2.5,
               labelspacing=.85)
    min_n = min(n for values in data["counts"].values() for n in values)
    n_note = "All 10 drugs available." if min_n == 10 else f"Available drugs per point: {min_n}–10; missing results are retained as NA."
    footer = [
        "Ten development drugs; two layouts per drug. Log errors are averaged within drugs, then drugs receive equal weight.",
        "Panel a uses a zoomed y-range. Random: three permutations; lines join discrete budgets. " + n_note,
        f"Panel b: budgets 1–3; diamond/line = paired estimate/95% study-cluster bootstrap CI ({pair.get('resamples', 2000):,} resamples).",
        "Panel c: first FE ≤ 2 AND FE ≤ 1.10 × full-data FE; post hoc, not a prospective stopping rule. Failures stay in denominators.",
        "The full-data reference need not be best. Repeated development pilot; not independent clinical validation.",
    ]
    for index, line in enumerate(footer):
        fig.text(.028, .083 - .0168 * index, line, fontsize=6.4, color="#4B5B65", va="bottom")
    return fig


def write_caption(run_dir):
    """Update only our figure block inside the already existing result README."""
    path = Path(run_dir) / "README.md"
    text = path.read_text()
    start, stop = "<!-- sequential-sampling-figure:start -->", "<!-- sequential-sampling-figure:end -->"
    if start in text or stop in text:
        if text.count(start) != 1 or text.count(stop) != 1 or text.index(stop) < text.index(start):
            raise ValueError("Malformed existing figure-caption block")
        before, rest = text.split(start)
        _, after = rest.split(stop)
        text = before.rstrip() + "\n" + after.lstrip()
    caption = """
![Sequential endpoint-directed PBPK sampling](comparison.png)

**Figure. Prediction benefit and data efficiency of sequential PBPK sampling.**
**(a)** Cmax geometric mean fold error (GMFE; lower is better) after zero to three additional observations.
Log fold errors are averaged over the two layouts within each drug before the ten drugs receive equal weight;
the random comparator also averages over its three fixed permutations within each drug. Lines guide the eye
between discrete budgets and do not imply performance at fractional budgets. Panel a uses a clearly labelled,
zoomed linear y-range set to the observed range plus 12.5% margin at each end; no values are changed or smoothed.
Available-case counts are stated in the figure.
**(b)** Every drug's geometric-mean Cmax fold-error ratio (refit goal/time coverage), averaged over the two layouts
and budgets 1–3; ratios below one favor refit goal. The signed numbers show 100 × (ratio − 1), so negative values
indicate reduced fold error and positive values indicate increased fold error. The bottom diamond and horizontal line show the drug-equal
paired ratio and its descriptive 95% study-cluster bootstrap interval (2,000 resamples). Ertapenem and Ceftriaxone
remain in the same study cluster. Missing paired comparisons are marked NA rather than omitted from the drug list.
**(c)** Cumulative *ever reached criterion by budget*: the first budget, including budget zero, at which Cmax fold error
is both ≤2 and ≤1.10 times that drug's full-data-reference fold error. Each drug first averages its two layouts
(six layout/permutation sequences for random), then all ten drugs receive equal weight. Failed, unavailable and
never-reaching sequences remain in the denominator. Reaching the criterion earlier does not imply that later
budgets continue to satisfy it. The full-data fit is an information reference and is not guaranteed optimal;
these thresholds are evaluated retrospectively and do not validate an online stopping rule.
All acquisitions lock before revealing their development concentrations, and target values are used only after
predictions lock. This is a repeated development pilot, not independent clinical validation.
"""
    path.write_text(text.rstrip() + "\n\n" + start + "\n" + caption.strip() + "\n" + stop + "\n")


def audit_pdf_descendant_fonts(path):
    """Resolve Type0 fonts, which the skill's shallow descriptor check misses."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    records = []
    for page in reader.pages:
        resources = page["/Resources"].get_object()
        for _, raw in resources.get("/Font", {}).get_object().items():
            font = raw.get_object()
            for child in font.get("/DescendantFonts", [raw]):
                descendant = child.get_object()
                subtype = str(descendant.get("/Subtype", ""))
                descriptor = descendant.get("/FontDescriptor")
                descriptor = descriptor.get_object() if descriptor else {}
                sizes = {key: len(descriptor[key].get_object().get_data())
                         for key in ("/FontFile", "/FontFile2", "/FontFile3") if key in descriptor}
                if "Type3" in subtype or not sizes or not all(v > 0 for v in sizes.values()):
                    raise ValueError("PDF has a Type3 or genuinely unembedded font")
                records.append(dict(font=str(font.get("/BaseFont")), subtype=subtype, embedded_stream_bytes=sizes))
    if not records:
        raise ValueError("PDF font audit found no embedded fonts")
    sizes = [(float(page.mediabox.width) / 72, float(page.mediabox.height) / 72) for page in reader.pages]
    if len(sizes) != 1 or any(abs(actual - target) > 1e-6 for actual, target in zip(sizes[0], FIGSIZE)):
        raise ValueError("PDF page dimensions differ from the final figure dimensions")
    return dict(fonts=records, page_inches=sizes, all_embedded=True, type3=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--export", action="store_true", help="Export the two final formats after visual review")
    args = parser.parse_args()
    result, rows = load_complete(args.run_dir)
    data = aggregate(result, rows)
    qa_dir = ROOT / "runtime/test_tmp/sequential_sampling_figure"
    qa_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SKILL))
    from profile_data import profile_data, render_report
    from visual_qa import audit_layout, render_preview
    from check_figure import check_figure
    from PIL import Image
    import pandas as pd

    # EDA precedes plotting. Repeated rows are never treated as independent drugs.
    # Never pool unlike endpoint units or correlate unrelated raw concentrations.
    frame = pd.DataFrame([{k: r[k] for k in ("case_id", "drug", "layout", "method", "budget", "status", "error")}
                          for r in rows if r["endpoint"] == "cmax"])
    for column in ("case_id", "drug", "layout", "method", "status"):
        frame[column] = frame[column].astype("category")
    for column in ("budget", "error"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    profile = profile_data(frame, group_cols=["method", "budget"])
    (qa_dir / "profile.txt").write_text(render_report(profile))
    fig = make_figure(data)
    preview = qa_dir / "preview.png"
    render_preview(fig, str(preview), dpi=180)
    with Image.open(preview) as image:
        image.convert("L").save(qa_dir / "grayscale.png")
    issues = audit_layout(fig)
    (qa_dir / "layout.json").write_text(json.dumps(issues, indent=2))
    if any(severity == "FAIL" for severity, _ in issues):
        raise ValueError(f"Figure layout audit failed: {issues}")
    output = dict(status="preview_ready", preview=str(preview), layout_issues=issues,
                  drug_count=len(data["case_ids"]), paired_estimate=data["paired"]["estimate"])
    if args.export:
        run_dir = args.run_dir.resolve()
        paths = (run_dir / "comparison.pdf", run_dir / "comparison.png")
        fig.savefig(paths[0], metadata={"Creator": "PBPK sequential-sampling plotter", "CreationDate": None, "ModDate": None})
        fig.savefig(paths[1], dpi=600)
        checks = {str(path): check_figure(str(path), min_dpi=300, target_inches=FIGSIZE) for path in paths}
        font_check = audit_pdf_descendant_fonts(paths[0])
        unresolved = [(path, severity, message) for path, (issues, _) in checks.items()
                      for severity, message in issues if severity in ("WARN", "FAIL")
                      and not (path == str(paths[0]) and severity == "WARN" and 'Font may not be embedded' in message
                               and font_check["all_embedded"])]
        (qa_dir / "exports.json").write_text(json.dumps(dict(skill_checks=checks, pdf_descendant_fonts=font_check,
            shallow_font_warning_resolved=True, unresolved=unresolved), indent=2, default=str))
        if unresolved:
            raise ValueError(f"Strict export audit requires review: {checks}")
        write_caption(run_dir)
        output.update(status="exported", files=[str(path) for path in paths])
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
