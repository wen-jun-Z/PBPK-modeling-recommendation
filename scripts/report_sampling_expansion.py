#!/usr/bin/env python3
"""Independent completed-run audit and one English thirty-drug figure.

This module never imports a runner, fitter or clinical scorer. Only state.json
is read until completion is verified. Preview first, inspect comparison.png,
then use ``--export`` to create the matching vector PDF.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/blueprint_design/sampling_expansion_20261009"
OLD = ROOT / "experiments/blueprint_design/evidence_gap_design_20261008"
SKILL = Path("/home/xuyiqian/.codex/skills/scipilot-figure-skill/scripts")
METHODS = ("learned_gain", "time_farthest", "maximum_variance", "c_optimal",
           "request_directed", "linear_covariance", "uniform_random", "no_addition", "oracle")
GOALS = ("cmax", "auc_inf", "terminal_half_life")
ENDPOINTS = ("curve", "auc", "cmax", "tmax", "c_last", "terminal_half_life", "auc_inf", "clearance", "vz", "mrt", "vss")
LABELS = ("Learned gain (frozen)", "Time coverage", "Concentration variance", "PyApprox c-optimal",
          "Nonlinear utility", "Linear utility", "Exact random expectation", "No added observation", "Hindsight oracle")


def read(path):
    return json.loads(Path(path).read_text())


def inside(path):
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Only this project's files may be read")
    return p


def digest(path):
    return hashlib.sha256(inside(path).read_bytes()).hexdigest()


def checked(source):
    p = inside(source["path"])
    if digest(p) != source["sha256"]:
        raise ValueError("Source hash mismatch: " + str(p))
    return read(p)


def ref(path):
    p = inside(path)
    return dict(path=str(p.relative_to(ROOT)), sha256=digest(p))


def json_native(value):
    """Normalize NumPy/pandas descriptive-statistic scalars for JSON output."""
    if isinstance(value, np.ndarray):
        return json_native(value.tolist())
    if isinstance(value, np.generic):
        return json_native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(v) for v in value]
    return value


def save(path, document):
    p = inside(path)
    if not p.is_relative_to(RUN):
        raise ValueError("Report output must remain in this batch")
    p.write_text(json.dumps(json_native(document), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def same(a, b, label):
    if a is None or b is None:
        if a is not None or b is not None:
            raise AssertionError(label + ": null mismatch")
    elif not np.allclose(a, b, rtol=1e-9, atol=1e-11):
        raise AssertionError(label + ": numerical mismatch")


def gm(values):
    if not values or any(v is None for v in values):
        return None
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Positive finite error values required")
    return math.exp(float(np.log(values).mean()))


def bootstrap(values, clusters, *, seed, resamples):
    """Independent group-resampling implementation, preserving unequal sizes."""
    if set(values) != set(clusters):
        raise ValueError("Aligned values/cluster keys required")
    keys = sorted(values)
    names = sorted(set(clusters.values()))
    if not keys:
        return dict(estimate=None, ci95=None, n_cases=0, n_clusters=0)
    logs = {k: math.log(values[k]) for k in keys}
    result = dict(estimate=math.exp(sum(logs.values())/len(keys)), n_cases=len(keys), n_clusters=len(names))
    if len(names) == 1:
        return result | dict(ci95=None)
    draws = np.random.default_rng(seed).integers(0, len(names), size=(resamples, len(names)))
    sums = np.array([sum(logs[k] for k in keys if clusters[k] == c) for c in names])
    counts = np.array([sum(clusters[k] == c for k in keys) for c in names])
    samples = np.exp(np.sum(sums[draws], axis=1)/np.sum(counts[draws], axis=1))
    return result | dict(ci95=np.quantile(samples, [.025, .975]).tolist())


def load_complete():
    try:
        state = read(RUN / "state.json")
    except FileNotFoundError as exc:
        raise ValueError("Run state not yet available: target results will not be opened") from exc
    if state.get("status") != "complete":
        raise ValueError("Run not complete: results and clinical outcomes will not be opened")
    protocol = checked(state["protocol"])
    for path, expected in protocol["source_pins"].items():
        if digest(path) != expected:
            raise ValueError("Frozen source changed: " + path)
    # The manuscript is an independently edited deliverable, not a fit input.
    # Record concurrent changes without blocking source-verified experiment audit.
    state['_paper_current'] = ref(protocol['paper_before']['path'])
    state['_paper_unchanged'] = state['_paper_current'] == protocol['paper_before']
    result = checked(state["results"])
    if result["protocol"] != state["protocol"]:
        raise ValueError("Result protocol differs from state")
    if not state["predictions_locked_at"] <= result["scored_at"] <= state["completed_at"]:
        raise ValueError("Clinical scoring lock order invalid")
    if result["predictions_locked_at"] != state["predictions_locked_at"]:
        raise ValueError("Result prediction lock changed")
    csv_path = inside(result["endpoints"]["path"])
    if digest(csv_path) != result["endpoints"]["sha256"]:
        raise ValueError("Endpoint CSV hash mismatch")
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return state, protocol, result, rows


def independent_ranker(features, records, training_ids, goal, alpha):
    """Augmented weighted least squares; no production learner import."""
    raw = {(r["design_key"], r["candidate"]): r for r in records}
    xs, ys, ws = [], [], []
    for cid in sorted(training_ids):
        menus = sorted(k for k, value in features.items() if value["case_id"] == cid)
        for key in menus:
            d = features[key]
            x = np.asarray(d["goals"][goal]["X"], dtype=float)
            baseline = raw[key, "baseline"]["endpoints"][goal]["error"]
            y = [math.log(baseline)-math.log(raw[key, key+f"__point_{i:02d}"]["endpoints"][goal]["error"]) for i in d["candidate_ids"]]
            xs.extend(x-x.mean(axis=0)); ys.extend(np.asarray(y)-np.mean(y))
            ws.extend([1/len(training_ids)/len(menus)/len(x)]*len(x))
    x, y, w = np.asarray(xs), np.asarray(ys), np.asarray(ws)
    scale = np.sqrt(np.sum(w[:, None]*x*x, axis=0)); scale[scale < 1e-12] = 1.
    z = x/scale
    matrix = np.vstack([np.sqrt(w)[:, None]*z, math.sqrt(alpha)*np.eye(x.shape[1])])
    response = np.r_[np.sqrt(w)*y, np.zeros(x.shape[1])]
    return scale, np.linalg.lstsq(matrix, response, rcond=None)[0]


def audit_models(state, protocol):
    frozen = checked(protocol["rankers"])
    originals = set(protocol["original_case_ids"])
    assert len(originals) == 10 and set(frozen["trained_case_ids"]) == originals
    assert frozen["added_drug_labels_used"] is False and frozen["alpha"] == 1.
    features = checked(frozen["features"])
    labels = checked(frozen["labels"])
    old_choices = checked(frozen["original_oof_decisions"])
    for goal in GOALS:
        scale, coef = independent_ranker(features, labels["records"], originals, goal, 1.)
        model = frozen["models"][goal]
        same(model["scale"], scale, "Frozen training-only scale")
        same(model["coef"], coef, "Frozen original-ten model")
        assert model["train_drugs"] == 10 and model["train_designs"] == 20
    # Check the existing OOF origin, not just copy whichever choice was saved.
    clusters = {c["case_id"]: c["cluster_id"] for c in protocol["cases"] if c["case_id"] in originals}
    oof_count = 0
    for cluster, fold in old_choices["folds"].items():
        heldout = {cid for cid in originals if clusters[cid] == cluster}
        assert set(fold["heldout_case_ids"]) == heldout
        assert set(fold["train_case_ids"]) == originals-heldout
        for goal in GOALS:
            scale, coef = independent_ranker(features, labels["records"], originals-heldout, goal, 1.)
            same(fold["models"][goal]["coef"], coef, "Original OOF model")
            same(fold["models"][goal]["scale"], scale, "Original OOF scale")
            for key, value in features.items():
                if value["case_id"] not in heldout:
                    continue
                scores = np.asarray(value["goals"][goal]["X"])/scale@coef
                choice = old_choices["choices"][key][goal]
                index = int(min(np.flatnonzero(scores >= max(scores)-1e-12), key=lambda i: value["times"][i]))
                same(choice["scores"], scores, "Original OOF score")
                assert index == choice["index"]
                oof_count += 1
    added = {}
    count = 0
    lock = checked(state["action_lock"])
    assert lock["protocol"] == state["protocol"] and lock["designs"] == state["designs"]
    assert lock["locked_at"] <= state["predictions_locked_at"]
    for key, entry in state["designs"].items():
        decision = checked(entry["result"])
        ch = checked(entry["features"])
        assert ch["decision"] == entry["result"] and ch["protocol"] == state["protocol"]
        assert ch["hidden_values_used"] is False and ch["locked_at"] <= lock["locked_at"]
        assert ch["candidate_ids"] == decision["candidate_ids"]
        added[key] = ch
        for goal, detail in ch["goals"].items():
            if detail["status"] != "selected":
                assert detail["choices"]["learned_gain"] is None
                continue
            model = frozen["models"][goal]
            scores = np.asarray(detail["X"])/np.asarray(model["scale"])@np.asarray(model["coef"])
            same(scores, detail["scores"], "New-drug frozen-model score")
            index = int(min(np.flatnonzero(scores >= max(scores)-1e-12), key=lambda i: ch["candidate_times_h"][i]))
            assert index == detail["choices"]["learned_gain"]
            count += 1
    return frozen, features, old_choices, added, dict(frozen_models=3, original_oof_choices=oof_count, added_supported_choices=count)


def audit_csv(state, protocol, result, rows, original_features, original_choices, added_choices):
    raw = {(r["design_key"], r["candidate"]): r for r in result["records"]}
    if len(raw) != len(result["records"]):
        raise ValueError("Duplicate raw prediction records")
    originals = set(protocol["original_case_ids"])
    old_state = read(OLD / "state.json")
    cases = {c["case_id"]: c for c in protocol["cases"]}
    seen, decisions = set(), {}
    for cid in cases:
        for regime in ("early_four", "late_four"):
            key = cid+"__"+regime
            entry = (old_state if cid in originals else state)["designs"][key]
            decisions[key] = checked(entry["result"])
    for rec in result["records"]:
        key, candidate = rec["design_key"], rec["candidate"]
        owner = old_state if rec["case_id"] in originals else state
        expected = decisions[key]["baseline"] if candidate == "baseline" else owner["units"][candidate]["result"]
        assert rec["source"] == expected
        checked(expected)
    for row in rows:
        cid, goal, ep, method = (row[k] for k in ("case_id", "requested_goal", "endpoint", "method"))
        key = cid+"__"+row["regime"]
        if row["drug"] != cases[cid]["drug"] or row["cluster_id"] != cases[cid]["cluster_id"] or row["cohort"] != cases[cid]["expansion_cohort"]:
            raise ValueError("CSV drug, cohort or study identity differs from protocol")
        ident = (key, goal, method, ep)
        if ident in seen:
            raise ValueError("Duplicate endpoint comparison row")
        seen.add(ident)
        d = decisions[key]
        menu = [raw[key, key+f"__point_{i:02d}"] for i in d["candidate_ids"]]
        index, picked = None, []
        if method == "no_addition":
            picked = [raw[key, "baseline"]]
        elif method == "uniform_random":
            picked = menu
        elif method == "oracle":
            errors = [r["endpoints"][goal]["error"] for r in menu]
            if all(e is not None for e in errors):
                index = int(np.argmin(errors)); picked = [menu[index]]
        else:
            if cid in originals:
                index = original_choices["choices"][key][goal]["index"] if method == "learned_gain" else original_features[key]["goals"][goal]["choices"].get(method)
            else:
                index = added_choices[key]["goals"][goal]["choices"].get(method)
            if index is not None:
                picked = [menu[index]]
        time_h = None if index is None else d["candidate_times_h"][index]
        number = lambda name: None if row[name] == "" else float(row[name])
        same(number("selected_time_h"), time_h, "Chosen observation time")
        if not picked:
            same(number("error"), None, "Unsupported selector remains missing")
            continue
        values = [r["curve_metrics"]["gmfe"] if ep == "curve" else r["endpoints"][ep]["error"] for r in picked]
        error = float(np.mean(values)) if ep == "tmax" and all(v is not None for v in values) else None if ep == "tmax" else gm(values)
        same(number("error"), error, "CSV prediction error")
        if ep != "curve":
            same(number("reference"), picked[0]["endpoints"][ep]["reference"], "Clinical reference")
            same(number("prediction"), picked[0]["endpoints"][ep]["value"] if len(picked) == 1 else None, "Native endpoint prediction")
    expected = {(cid+"__"+regime, goal, method, ep) for cid in cases for regime in ("early_four", "late_four")
                for goal in GOALS for method in METHODS for ep in ENDPOINTS}
    if seen != expected:
        raise ValueError("All30 drug/method/endpoint cells must be retained including failures")
    return len(seen)


def summarize(protocol, result, rows):
    ids = {c["case_id"] for c in protocol["cases"]}
    original = set(protocol["original_case_ids"])
    scopes = {"original_ten": original, "added_twenty": ids-original, "all_thirty": ids}
    clusters = {c["case_id"]: c["cluster_id"] for c in protocol["cases"]}
    values = {}
    for row in rows:
        if row["error"]:
            values[(row["case_id"], row["regime"], row["requested_goal"], row["method"], row["endpoint"])] = float(row["error"])
    checks = 0
    for scope, members in scopes.items():
        for regime in ("early_four", "late_four"):
            for goal in GOALS:
                for method in METHODS:
                    for label, ep in (("endpoint", goal), ("curve", "curve")):
                        usable = {cid: values[cid, regime, goal, method, ep] for cid in members if (cid, regime, goal, method, ep) in values}
                        stat = bootstrap(usable, {cid: clusters[cid] for cid in usable}, **protocol["bootstrap"])
                        official = result["summary"][scope][regime][goal][method][label]
                        for field in ("estimate", "ci95"):
                            same(stat[field], official[field], "Independent aggregate " + field)
                        assert stat["n_cases"] == official["n_cases"] and stat["n_clusters"] == official["n_clusters"]
                        assert official["attempted_drugs"] == len(members)
                        checks += 1
                for method in METHODS:
                    if method == "learned_gain":
                        continue
                    ratios = {cid: values[cid, regime, goal, "learned_gain", goal]/values[cid, regime, goal, method, goal]
                              for cid in members if (cid, regime, goal, "learned_gain", goal) in values and (cid, regime, goal, method, goal) in values}
                    stat = bootstrap(ratios, {cid: clusters[cid] for cid in ratios}, **protocol["bootstrap"])
                    official = result["comparisons"][scope][regime][goal][method]
                    same(stat["estimate"], official["estimate"], "Independent paired effect")
                    same(stat["ci95"], official["ci95"], "Independent paired interval")
                    assert official["complete_pairs"] == len(ratios)
                    assert official["ratio_below_one"] == sum(v < 1 for v in ratios.values())
                    assert official["ratio_above_one"] == sum(v > 1 for v in ratios.values())
                    checks += 1
    outcomes = {}
    for cid in sorted(ids):
        outcomes[cid] = {}
        for ep in ("cmax", "curve"):
            a = values.get((cid, "late_four", "cmax", "learned_gain", ep))
            b = values.get((cid, "late_four", "cmax", "time_farthest", ep))
            outcomes[cid][ep] = dict(learned=a, time=b, ratio=a/b if a is not None and b is not None else None)
    return dict(statistic_checks=checks, drug_outcomes=outcomes,
                cases={c["case_id"]: dict(drug=c["drug"], original=c["case_id"] in original,
                                        cross_study=c.get("cross_study_pair", True)) for c in protocol["cases"]})


def audit():
    state, protocol, result, rows = load_complete()
    frozen, features, choices, added, model_checks = audit_models(state, protocol)
    count = audit_csv(state, protocol, result, rows, features, choices, added)
    statistics = summarize(protocol, result, rows)
    report = dict(status="passed", audited_at=time.time(), protocol=state["protocol"], results=state["results"],
                  audit_code=ref(Path(__file__)), endpoint_rows=count,
                  paper_unchanged=state['_paper_unchanged'], paper_before=protocol['paper_before'],
                  paper_current=state['_paper_current'], manuscript_written_by_this_report=False,
                  source_hashes=len(protocol["source_pins"]), **model_checks, **statistics,
                  scope="Original-ten OOF and added-twenty fixed-model choices independently replayed. This audits arithmetic and isolation, not clinical validity or blinded generalization.")
    save(RUN / "audit.json", report)
    return report


def make_figure(protocol, result, audited):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    from setup_style import setup_style

    setup_style(journal="nature", lang="en", use_sciplots=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7.3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.constrained_layout.use": False,
        "axes.spines.right": False, "axes.spines.top": False, "savefig.bbox": None})
    fig = plt.figure(figsize=(7.2, 9.4))
    top = fig.add_axes([.285, .705, .65, .195])
    left = fig.add_axes([.195, .19, .245, .405])
    right = fig.add_axes([.715, .19, .245, .405])
    estimates = []
    for y, (method, label) in enumerate(zip(METHODS, LABELS)):
        s = result["summary"]["added_twenty"]["late_four"]["cmax"][method]["endpoint"]
        color = "#009E73" if method == "learned_gain" else "#263F4C"
        if s["estimate"] is not None:
            estimates.append(s["estimate"])
            if s["ci95"]:
                lo, hi = s["ci95"]; estimates.extend([lo, hi])
                top.hlines(y, lo, hi, color=color, linewidth=1.2)
            top.plot(s["estimate"], y, "s" if method == "learned_gain" else "o", color=color, markersize=4.2)
        else:
            top.text(.05, y, "NA", transform=top.get_yaxis_transform(), va="center", fontsize=7)
        top.text(1.025, y, f'{s["n_cases"]}/20', transform=top.get_yaxis_transform(), va="center", fontsize=7, color="#4B5B65")
    top.set_yticks(range(len(METHODS)), LABELS); top.invert_yaxis()
    top.set_title("a  Primary evaluation: 20 added drugs", loc="left", pad=9)
    top.set_xlabel("Cmax GMFE (log scale; lower is better)")
    top.set_xscale("log"); top.minorticks_off(); top.grid(axis="x", color="#E2E8EB", linewidth=.5)
    if estimates:
        lower, upper = min(1., min(estimates)/1.12), max(estimates)*1.12
        top.set_xlim(lower, upper)
        top.xaxis.set_major_locator(FixedLocator(np.geomspace(lower, upper, 4)))
    top.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))
    allratios = [x[e]["ratio"] for x in audited["drug_outcomes"].values() for e in ("cmax", "curve") if x[e]["ratio"] is not None]
    lo, hi = min([1., *allratios])/1.15, max([1., *allratios])*1.15
    exponent = max(abs(math.log2(lo)), abs(math.log2(hi)), .15)
    ticks = [2**(-exponent), 1., 2**exponent]
    for ax, original, title in ((left, True, "b  Original 10: held-out choices"), (right, False, "c  Added 20: frozen choices")):
        ids = sorted((cid for cid, c in audited["cases"].items() if c["original"] == original), key=lambda cid: audited["cases"][cid]["drug"].lower())
        for y, cid in enumerate(ids):
            for endpoint, offset, marker, color in (("cmax", -.12, "o", "#0072B2"), ("curve", .12, "s", "#D55E00")):
                ratio = audited["drug_outcomes"][cid][endpoint]["ratio"]
                if ratio is None:
                    ax.text(1., y+offset, "NA", color=color, fontsize=6, ha="center", va="center")
                else:
                    ax.plot([1., ratio], [y+offset, y+offset], color=color, linewidth=.65, alpha=.7)
                    ax.plot(ratio, y+offset, marker, color=color, markersize=3.5, markeredgewidth=.5)
        ax.axvline(1., color="#8E999F", linewidth=.8, linestyle="--", zorder=0)
        ax.set_yticks(range(len(ids)), [audited["cases"][cid]["drug"]+("*" if not audited["cases"][cid]["cross_study"] else "") for cid in ids])
        ax.set_ylim(len(ids)-.4, -.6); ax.tick_params(axis="y", length=0, pad=5)
        ax.set_xscale("log", base=2); ax.set_xlim(2**(-exponent), 2**exponent)
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))
        ax.set_xlabel("Error ratio: learned / time\n(log scale; <1 favors learned)")
        ax.set_title(title, loc="left", pad=12, fontsize=8)
        ax.grid(axis="x", color="#E2E8EB", linewidth=.5)
    fig.text(.035, .977, "Does learned sampling transfer to additional drugs?", fontsize=11, weight="bold", color="#263F4C", va="top")
    fig.text(.035, .949, "Frozen 10-drug policy | 20 added drugs | Four late observations + one acquired point", fontsize=7.4, color="#4B5B65")
    fig.legend(handles=[Line2D([], [], marker="o", color="#0072B2", label="Cmax fold-error ratio", linestyle="None", markersize=4),
                        Line2D([], [], marker="s", color="#D55E00", label="Curve GMFE ratio", linestyle="None", markersize=4)],
               loc="upper center", bbox_to_anchor=(.57, .65), frameon=False, ncol=2, fontsize=7.5)
    notes = ["a: Drug-equal GMFE with 95% study-cluster bootstrap intervals (2,000 draws); labels show available/enrolled drugs.",
             "b–c: Every enrolled drug is shown; missing comparisons remain NA. No drug was removed because of poor performance.",
             "* Finerenone and Montelukast compare different arms within one study; these are not independent-study validations.",
             "The original 10 use their saved out-of-fold choices and cached PBPK inputs; the added 20 use fixed original-ten models.",
             "Cmax reference = measured-curve sampled maximum. Historical development evaluation; manuscript not updated by this run."]
    for i, line in enumerate(notes):
        fig.text(.035, .105-i*.018, line, fontsize=6.35, color="#4B5B65", va="bottom")
    return fig


def plot(export=False):
    state, protocol, result, rows = load_complete()
    audited = read(RUN / "audit.json")
    if audited.get("status") != "passed" or audited["results"] != state["results"]:
        raise ValueError("Current independent numerical audit required")
    sys.path.insert(0, str(SKILL))
    from profile_data import profile_data
    from visual_qa import audit_layout, render_preview
    import pandas as pd
    frame = pd.DataFrame([{**r, "error": None if r["error"] == "" else float(r["error"])} for r in rows
                          if r["regime"] == "late_four" and r["requested_goal"] == "cmax" and r["endpoint"] in ("cmax", "curve")])
    profile = profile_data(frame[["drug", "cohort", "method", "endpoint", "error"]], group_cols=["cohort", "endpoint"])
    fig = make_figure(protocol, result, audited)
    render_preview(fig, str(RUN / "comparison.png"), dpi=220)
    qa = audit_layout(fig)
    # Keep QA/EDA in the existing audit, avoiding another report artifact.
    audited["figure"] = dict(size_inches=[7.2, 9.4], layout_qa=qa, profile=profile,
                              png=ref(RUN / "comparison.png"), all_enrolled_drugs_displayed=True,
                              vector_exported=bool(export), visual_review_required=True)
    if export:
        if any(level == "FAIL" for level, message in qa):
            raise ValueError("Figure machine QA failed; refusing vector export")
        fig.savefig(RUN / "comparison.pdf", format="pdf", dpi=300)
        audited["figure"]["pdf"] = ref(RUN / "comparison.pdf")
    save(RUN / "audit.json", audited)
    path = RUN / "README.md"
    if path.exists():
        start, stop = "<!-- sampling-expansion-figure:start -->", "<!-- sampling-expansion-figure:end -->"
        text = path.read_text()
        if start in text or stop in text:
            if text.count(start) != 1 or text.count(stop) != 1 or text.index(stop) < text.index(start):
                raise ValueError("Malformed existing figure block")
            before, tail = text.split(start)
            _, after = tail.split(stop)
            text = before.rstrip()+"\n"+after.lstrip()
        caption = ("\n"+start+"\n\n![Thirty-drug sampling evaluation](comparison.png)\n\n"
                   'The upper panel compares the additional 20 drugs with study-cluster bootstrap 95% intervals; the lower panel shows per-drug error ratios against time coverage, where values below 1 indicate improvement.'
                   'Original-ten drugs retain their grouped held-out selections; the additional 20 use the model trained and frozen on the original ten only. All 30 drugs and missing entries are shown. Asterisks indicate different arms of the same study, not independent cross-study validation.'
                   'Original sources and model files passed hash checks; these results were not written into the manuscript PDF.\n\n'+stop+"\n")
        path.write_text(text.rstrip()+"\n"+caption)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return audited["figure"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "plot", "all"])
    parser.add_argument("--export", action="store_true", help="After visual review, export matching vector PDF")
    args = parser.parse_args()
    if args.command in {"audit", "all"}:
        summary = audit()
        print(json.dumps({k: summary[k] for k in ("status", "endpoint_rows", "statistic_checks", "source_hashes", "paper_unchanged")}))
    if args.command in {"plot", "all"}:
        figure = plot(args.export)
        print(json.dumps({k: figure[k] for k in ("png", "layout_qa", "vector_exported")}, ensure_ascii=False))
