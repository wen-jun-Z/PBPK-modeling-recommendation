"""Rebuild audited paper tables/figures from saved component and model experiments.

CPU only: no training, inference, network or database access. Default outputs are
under results/{data,tables,figures/experiments}; original experiments are read-only.
Run with runtime/app-env/bin/python scripts/build_paper_results.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/component_recommendation"
OUT = ROOT / "results"
VARIANTS = ("no_kg", "text_graph", "soft_graph", "hybrid")
ARCHITECTURES = ("no_gnn", "rgcn", "hgt", "graph_transformer")
SEEDS = (17, 29, 43)
# Match the final NeurIPS single-column width without secondary scaling.
FIGURE_WIDTH_INCHES = 5.5
VARIANT_NAMES = ("No KG", "Text graph", "Soft graph", "Hybrid")
ARCHITECTURE_NAMES = ("NoGNN", "R-GCN", "HGT", "Graph Transformer")
SOURCE_INDEX: dict[str, dict] = {}


def source(path: Path):
    raw = path.read_bytes()
    SOURCE_INDEX[str(path.relative_to(ROOT))] = {
        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)
    }
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return json.loads(raw)


def record_source(path: Path):
    raw = path.read_bytes()
    SOURCE_INDEX[str(path.relative_to(ROOT))] = {
        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)
    }


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"Recomputed metric mismatch: {a} != {b}")


def validate_hash(path: Path, expected: str):
    actual = SOURCE_INDEX[str(path.relative_to(ROOT))]["sha256"]
    if actual != expected:
        raise ValueError(f"Source checksum mismatch: {path.relative_to(ROOT)}")


def component_results():
    dataset = BASE / "101_gate5_locked_prompt_dataset_v9"
    generation = BASE / "104_gate5_qwen3_8b_bf16_normalized_handles"
    evaluation = BASE / "105_gate6_strict_evaluation"
    dm = source(dataset / "dataset_manifest.json")
    prompts = source(dataset / "prompt_examples.jsonl")
    silver = source(dataset / "recommendation_silver.jsonl")
    gm = source(generation / "gate5_manifest.json")
    comparison = source(generation / "gate5_comparison.json")
    raw = source(generation / "raw_variant_outputs.jsonl")
    gate6 = source(evaluation / "gate6_manifest.json")
    reported = source(evaluation / "recommendation_evaluation.json")
    retrieval = source(evaluation / "retrieval_evaluation.json")
    for name, directory in [
        ("dataset_manifest.json", dataset), ("prompt_examples.jsonl", dataset),
        ("recommendation_silver.jsonl", dataset), ("gate5_manifest.json", generation),
        ("gate5_comparison.json", generation), ("raw_variant_outputs.jsonl", generation),
    ]:
        validate_hash(directory / name, gate6["input_hashes"][name])
    for name in ("recommendation_evaluation.json", "retrieval_evaluation.json"):
        validate_hash(evaluation / name, gate6["artifacts"][name]["sha256"])
    test = {row["example_id"]: row for row in prompts if row["split"] == "test"}
    assert len(test) == reported["test_example_count"] == 25
    assert len(raw) == len(VARIANTS) * len(test) == 100
    assert len({(row["variant"], row["example_id"]) for row in raw}) == len(raw)
    assert all(row["example_id"] in test for row in raw)
    split_counts = {split: sum(r["split"] == split for r in prompts)
                    for split in ("train", "validation", "test")}
    assert split_counts == dm["summary"]["split_counts"]
    compounds = {split: {r["compound_id"] for r in prompts if r["split"] == split}
                 for split in split_counts}
    assert all(not compounds[a] & compounds[b] for a, b in
               (("train", "validation"), ("train", "test"), ("validation", "test")))
    assert {r["example_id"] for r in silver} == {r["example_id"] for r in prompts}
    records = []
    for variant, display in zip(VARIANTS, VARIANT_NAMES):
        predictions = sorted((r for r in raw if r["variant"] == variant), key=lambda r: r["example_id"])
        assert len(predictions) == len(test)
        record = {"variant": variant, "method": display, "test_requests": len(test)}
        field_metrics = {}
        for field in ("component_ids", "evidence_ids", "gap_codes", "public_model_ids"):
            tp = fp = fn = exact = nonempty = 0
            for row in predictions:
                predicted = set((row.get("selection") or {}).get(field, [])) if row["schema_valid"] else set()
                target = set(test[row["example_id"]]["target"].get(field, []))
                tp += len(predicted & target); fp += len(predicted - target); fn += len(target - predicted)
                exact += predicted == target; nonempty += bool(target)
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            metrics = {"true_positive": tp, "false_positive": fp, "false_negative": fn,
                       "f1": f1, "exact_set_accuracy": exact / len(test), "nonempty_target_count": nonempty}
            for metric, value in metrics.items():
                close(value, reported["variants"][variant][field][metric])
            field_metrics[field] = metrics
        record.update(component_micro_f1=field_metrics["component_ids"]["f1"],
                      component_exact_set_accuracy=field_metrics["component_ids"]["exact_set_accuracy"],
                      evidence_micro_f1=field_metrics["evidence_ids"]["f1"],
                      gap_micro_f1=field_metrics["gap_codes"]["f1"])
        records.append(record)
    # Stable request order is based on clinical task and identity, never score.
    def request_order(row):
        request = row["modeling_request"]
        objective = request.get("objective") or ""
        priority = {"ddi_victim": 0, "ddi_perpetrator": 1}.get(objective, 2)
        return (priority, objective, (request.get("compound") or "").casefold(), row["example_id"])

    ordered_requests = sorted(test.values(), key=request_order)
    predictions_by_key = {(row["example_id"], row["variant"]): row for row in raw}
    assert set(predictions_by_key) == {(row["example_id"], variant)
                                    for row in ordered_requests for variant in VARIANTS}
    request_details = []
    for request_index, prompt in enumerate(ordered_requests, start=1):
        for variant in VARIANTS:
            prediction = predictions_by_key[prompt["example_id"], variant]
            fields = {}
            for field in ("component_ids", "evidence_ids", "gap_codes"):
                predicted = set((prediction.get("selection") or {}).get(field, [])) if prediction["schema_valid"] else set()
                target = set(prompt["target"].get(field, []))
                tp, fp, fn = len(predicted & target), len(predicted - target), len(target - predicted)
                empty_match = not (predicted or target)
                exact = predicted == target
                saved = prediction["metrics"][field]
                for metric, value in (("true_positive", tp), ("false_positive", fp),
                                      ("false_negative", fn), ("exact_set_match", int(exact)),
                                      ("empty_target", int(not target)),
                                      ("empty_target_correct", int(empty_match))):
                    close(value, saved[metric])
                # Saved per-request F1 used zero for an empty denominator. Plot it
                # as undefined, while retaining the correct empty-set match flag.
                close(2 * tp / max(1, 2 * tp + fp + fn), saved["f1"])
                fields[field] = {"tp": tp, "fp": fp, "fn": fn,
                                 "f1": None if empty_match else 2 * tp / (2 * tp + fp + fn),
                                 "exact": exact, "empty_match": empty_match}
            request_details.append({"request_index": request_index, "example_id": prompt["example_id"],
                                    "compound_id": prompt["compound_id"],
                                    "compound_name": prompt["modeling_request"]["compound"],
                                    "objective": prompt["modeling_request"].get("objective"),
                                    "variant": variant, "fields": fields})
    assert len(request_details) == 100
    assert {row["request_index"] for row in request_details} == set(range(1, 26))
    selected = min(comparison["training_runs"], key=lambda r: r["best_validation_loss"])
    assert selected["seed"] == comparison["selected_seed"] == gm["projector"]["selected_seed"] == 29
    assert selected["checkpoint_sha256"] == gm["projector"]["checkpoint_sha256"]
    assert comparison["training_mode"] == "reused_locked_three_seed_projectors"
    assert all(row["reused_without_training"] for row in comparison["training_runs"])
    for name in ("gate6_evaluation.py", "soft_graph_prompt.py", "graph_prompt_dataset.py"):
        record_source(ROOT / "src/pbpk_kb/recommendation" / name)
    # Retrieval completeness is about target evidence/model IDs, not every positive subgraph.
    full_coverage = sum(set(r.get("positive_subgraph_ids", [])) <= set(r.get("retrieved_subgraph_ids", [])) for r in test.values())
    close(full_coverage / len(test), retrieval["complete_positive_subgraph_coverage_rate"])
    close(statistics.mean(len(set(r.get("positive_subgraph_ids", [])) & set(r.get("retrieved_subgraph_ids", []))) /
                          max(1, len(set(r.get("positive_subgraph_ids", [])))) for r in test.values()),
          retrieval["subgraph_recall_at_8"])
    record_source(ROOT / "src/pbpk_kb/recommendation/online_retrieval.py")
    protocol = {
        "conditional_dataset": {"retained_requests": len(prompts), "excluded_retrieval_failures": dm["retrieval_failure_count"],
             "pre_filter_requests": len(prompts) + dm["retrieval_failure_count"],
             "inclusion_rule": "retrieval_complete: selected subgraphs cover all silver target evidence IDs and public-model IDs",
             "positive_subgraph_full_coverage_test_count": full_coverage,
             "positive_subgraph_full_coverage_test_fraction": full_coverage / len(test)},
        "benchmark_level": "silver", "sample_unit": "modeling request", "split_counts": split_counts,
        "compound_counts": {s: len(v) for s, v in compounds.items()},
        "compound_overlap": 0, "llm": {k: gm["llm"][k] for k in ("model", "revision", "dtype")},
        "llm_frozen": True, "trained_modules": ["attention_pooling", "shared_mlp_projector"],
        "projector_seeds": list(SEEDS), "selected_projector_seed": 29,
        "selection_basis": comparison["selection_basis"], "test_generation_repetitions": 1,
        "training_mode": comparison["training_mode"], "projector_checkpoint_sha256": selected["checkpoint_sha256"],
        "training_data": "79 train silver requests; 15 validation requests; 25 held-out requests",
        "metric_definition": {"micro_f1": "2 TP / (2 TP + FP + FN), pooled across request/label assignments",
                              "exact_set_accuracy": "fraction of requests with the complete predicted set equal to the silver set"},
        "request_details": request_details,
        "gap_nonempty_targets": 12, "gap_empty_targets": 13,
        "unused_public_model_id_evaluation": {"nonempty_target_requests": 2, "positive_ids": 3,
            "reason": "ID selection against silver targets is not PBPK model ranking or execution validation"},
        "retrieval_report": retrieval,
        "limitations": [
            "KG-derived silver labels; not independent expert judgments or clinical validation.",
            "Evaluation is conditional on prior silver-target coverage: 119 retained of 125 requests; 6 retrieval failures excluded.",
            "One selected-seed generation result per variant; no repeated-generation uncertainty estimate.",
            "All prompts include constrained output candidate handles; no_kg is not an unconstrained standalone chatbot.",
            "Soft/hybrid prompts use learned projection; comparisons are not all zero-shot.",
            "Hybrid has higher exact-set accuracy but lower component micro-F1 than text_graph; no universal-best claim.",
            "The same 25-request set was used across development variants; zero cross-split compound overlap does not establish untouched external validation.",
            "Source-document/compound splits share this evaluation; do not count them as independent replications.",
            "The checkpoint hash is linked through saved manifests; this report does not rerun model weights.",
        ],
    }
    return records, protocol


def subgraph_results():
    training = BASE / "80_gate4_v3_gnn_training"
    benchmark = BASE / "79_gate4_v3_locked_benchmark"
    bm = source(benchmark / "gnn_benchmark_manifest.json")
    acceptance = source(benchmark / "preregistered_acceptance.json")
    comparison = source(training / "gnn_comparison.json")
    validate_hash(benchmark / "preregistered_acceptance.json", comparison["acceptance_sha256"])
    records = []; summary = []
    names = ("ndcg_at_10", "recall_at_10", "reciprocal_rank", "relation_macro_f1")
    expected_ids = None
    for architecture, display in zip(ARCHITECTURES, ARCHITECTURE_NAMES):
        previous = comparison["architectures"][architecture]
        saved = {r["seed"]: r for r in previous["runs"]}
        group = []
        for seed in SEEDS:
            report = source(training / architecture / f"seed_{seed}_metrics.json")
            assert report == saved[seed], (architecture, seed)
            test = report["test"]
            assert len(test["per_query"]) == test["query_count"] == 38
            ids = {row["query_id"] for row in test["per_query"]}
            assert len(ids) == 38
            if expected_ids is None: expected_ids = ids
            assert ids == expected_ids
            for metric in ("ndcg_at_10", "recall_at_10", "reciprocal_rank", "hit_at_10"):
                close(statistics.mean(r[metric] for r in test["per_query"]), test["overall"][metric])
            record = {"architecture": architecture, "method": display, "seed": seed, "test_queries": 38,
                      **{m: test["overall"][m] for m in names[:-1]},
                      "relation_macro_f1": test["relation_macro_f1"], "hit_at_10": test["overall"]["hit_at_10"]}
            records.append(record); group.append(record)
        aggregate = {"architecture": architecture, "method": display, "seed_count": 3, "test_queries": 38}
        for metric in names:
            values = [r[metric] for r in group]
            aggregate[metric + "_mean"] = statistics.mean(values)
            aggregate[metric + "_sd"] = statistics.stdev(values)
            aggregate[metric + "_population_sd"] = statistics.pstdev(values)
            if metric != "relation_macro_f1":
                close(aggregate[metric + "_mean"], previous["test_mean"][metric])
                close(aggregate[metric + "_population_sd"], previous["test_std"][metric])
            else:
                close(aggregate[metric + "_mean"], previous[metric + "_mean"])
        summary.append(aggregate)
    record_source(ROOT / "src/pbpk_kb/recommendation/gnn_benchmark.py")
    protocol = {
        "benchmark_level": "silver", "query_counts": bm["split_audit"]["query_count_by_split"],
        "candidate_subgraphs": bm["subgraph_count"], "query_categories": bm["split_audit"]["category_query_count_by_split"]["test"],
        "retrieval_split": bm["split_audit"]["method"], "relation_split": bm["relation_split_audit"]["method"],
        "relation_test_edges": bm["relation_split_audit"]["edge_count_by_split"]["test"],
        "table_field_paths": {"retrieval": "seed_*_metrics.json:test.overall.{ndcg_at_10,recall_at_10,reciprocal_rank}",
                              "relation": "seed_*_metrics.json:test.relation_macro_f1"},
        "seeds": list(SEEDS), "standard_deviation": "Sample SD across the three recorded seeds (ddof=1), recomputed from raw seed values; not CI or SEM",
        "original_standard_deviation": "Original report used population SD (ddof=0); retained under *_population_sd and independently verified",
        "metrics_source": "test, not raw_test: validation-selected category-specific teacher/graph score blending",
        "architecture_selection": comparison["selection_basis"], "selected_architecture": comparison["selected_architecture"],
        "training": acceptance["training"],
        "limitations": [
            "Provenance-backed KG silver relevance, not independent expert relevance labels.",
            "GNN comparisons include no-GNN teacher and validation-calibrated category scores, not pure standalone architecture comparisons.",
            "Only three optimization seeds on the same 38 test queries; these are not three independent test datasets.",
            "Recall is evaluated inside the saved candidate-pair pool, not exhaustive search over the model library.",
            "Hit@10 equals 1 for every method/seed; retained in CSV but omitted from plot because it cannot discriminate.",
            "HGT has the highest mean NDCG/MRR/relation F1 here, but lower Recall@10 than no_gnn.",
            "Original checkpoint-reproduction flags are historical; this script recomputes saved per-query aggregation, not GPU inference.",
            "No significance testing or confidence intervals are inferred from these data.",
        ],
    }
    return records, summary, protocol


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_tex(component, subgraph, seed_rows):
    component_metrics = ("component_micro_f1", "component_exact_set_accuracy", "evidence_micro_f1", "gap_micro_f1")
    graph_metrics = ("ndcg_at_10", "recall_at_10", "reciprocal_rank", "relation_macro_f1")
    component_lines = ['\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lcccc@{}}\n\\toprule\n',
                       'Method & Component micro F1 & Exact set accuracy & Citation micro F1 & Gap micro F1 \\\\\n\\midrule\n']
    for row in component:
        cells = []
        for metric in component_metrics:
            text = f'{row[metric]:.4f}'
            if row[metric] == max(r[metric] for r in component):
                text = r"\textbf{" + text + "}"
            cells.append(text)
        component_lines.append(row["method"] + " & " + " & ".join(cells) + " \\\\\n")
    component_lines.append("\\bottomrule\n\\end{tabular*}\n")
    (OUT / "tables/component_recommendation.tex").write_text("".join(component_lines), encoding="utf-8")
    graph_lines = ['\\begingroup\n\\fontsize{9}{11.5}\\selectfont\n\\setlength{\\tabcolsep}{2pt}\n',
                   '\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lcccc@{}}\n\\toprule\n',
                   'Method / seed & NDCG@10 & Recall@10 & MRR & Relation macro F1 \\\\\n\\midrule\n']
    for row in subgraph:
        cells = []
        for metric in graph_metrics:
            text = f'{row[metric + "_mean"]:.4f} \\pm {row[metric + "_sd"]:.4f}'
            if row[metric + "_mean"] == max(r[metric + "_mean"] for r in subgraph):
                text = r"\mathbf{" + text + "}"
            cells.append("$" + text + "$")
        graph_lines.append(row["method"] + " & " + " & ".join(cells) + " \\\\\n")
        runs = sorted((r for r in seed_rows if r["architecture"] == row["architecture"]), key=lambda r: r["seed"])
        assert [r["seed"] for r in runs] == list(SEEDS)
        for run in runs:
            cells = [f'{run[metric]:.4f}' for metric in graph_metrics]
            graph_lines.append(r"\quad Seed " + str(run["seed"]) + " & " + " & ".join(cells) + " \\\\\n")
        if row is not subgraph[-1]:
            graph_lines.append("\\addlinespace[5pt]\n")
    graph_lines.append("\\bottomrule\n\\end{tabular*}\n\\endgroup\n")
    (OUT / "tables/subgraph_retrieval.tex").write_text("".join(graph_lines), encoding="utf-8")


def plotting_environment():
    # An explicit project-only registry avoids scanning system/user font locations.
    import importlib.util
    spec = importlib.util.find_spec("matplotlib")
    mpl_root = Path(spec.origin).parent
    cache = ROOT / "runtime/cache/paper_figures"
    cache.mkdir(parents=True, exist_ok=True)
    version = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", (mpl_root / "font_manager.py").read_text()).group(1)
    payload = {"_version": version, "_FontManager__default_weight": "normal", "default_size": None,
               "defaultFamily": {"ttf": "DejaVu Sans", "afm": "Helvetica"}, "afmlist": [], "ttflist": [], "__class__": "FontManager"}
    (cache / f"fontlist-v{version}.json").write_text(json.dumps(payload))
    os.environ["MPLCONFIGDIR"] = str(cache)
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    for path in (mpl_root / "mpl-data/fonts/ttf").glob("*.ttf"):
        fm.fontManager.addfont(str(path))
    mpl.rcParams.update({"font.family": "STIXGeneral", "font.size": 9.5, "axes.titlesize": 10,
                         "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "axes.unicode_minus": False, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
                         "svg.hashsalt": "pbpk-completed-silver-paper-figures", "savefig.dpi": 300})
    import matplotlib.pyplot as plt
    return plt


def figure_audit(fig):
    from matplotlib.text import Text
    import warnings
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always"); fig.canvas.draw()
    missing = [str(w.message) for w in seen if "Glyph" in str(w.message) or "missing from font" in str(w.message)]
    assert not missing, missing
    renderer = fig.canvas.get_renderer(); canvas = fig.bbox
    cropped = []
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text(): continue
        box = obj.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > canvas.width + .5 or box.y1 > canvas.height + .5:
            cropped.append(obj.get_text())
    assert not cropped, cropped
    texts = [(obj.get_text(), obj.get_window_extent(renderer)) for obj in fig.findobj(Text)
             if obj.get_visible() and obj.get_text()]
    assert not any(re.search(r"[\u3400-\u9fff]", label) for label, _ in texts), "Expected English figure labels"
    overlap = []
    for i, (a, box) in enumerate(texts):
        for b, other in texts[i + 1:]:
            if min(box.x1, other.x1) - max(box.x0, other.x0) > 1 and min(box.y1, other.y1) - max(box.y0, other.y0) > 1:
                overlap.append((a,b))
    assert not overlap, overlap
    condition_palettes = getattr(fig, "_pbpk_score_colors_by_condition", None)
    audit = {"missing_glyph_warnings": 0, "cropped_texts": 0, "text_overlap_pairs": 0,
            "size_inches": list(fig.get_size_inches()),
            "minimum_font_points": min(obj.get_fontsize() for obj in fig.findobj(Text)
                                       if obj.get_visible() and obj.get_text()),
            "figure_dpi": 300, "score_color_range": [0, 1], "statistical_significance_claim": False,
            "label_language": "English", "font_family": "STIXGeneral",
            "color_theme": fig._pbpk_theme,
            "panel_colors": (WORKFLOW_THEMES[fig._pbpk_theme].get("panels")
                             or [WORKFLOW_THEMES[fig._pbpk_theme]["panel"]]),
            "score_colors": ((condition_palettes or {}).get("sentence_reordering")
                             or WORKFLOW_THEMES[fig._pbpk_theme].get("scores")),
            "palette_reference": "results/figures/workflow/流程图.png"}
    if condition_palettes:
        audit["score_colors_by_condition"] = condition_palettes
    return audit


def export_outlined_pdf(svg_path, png_path, size_inches):
    """Avoid OTF/CID subset remapping: preserve SVG glyph outlines as PDF paths."""
    import xml.etree.ElementTree as ET
    import numpy as np
    import pymupdf
    from PIL import Image
    svg = ET.parse(svg_path)
    # MuPDF does not preserve Matplotlib's SVG hatch patterns reliably. Use
    # explicit labels and distinct luminance levels instead of pattern fills.
    assert not any(node.tag.rsplit("}", 1)[-1] in {"text", "image", "pattern"} for node in svg.iter())
    with pymupdf.open(svg_path) as source_svg:
        converted = source_svg.convert_to_pdf()
    with pymupdf.open("pdf", converted) as pdf:
        pdf.set_metadata({})
        pdf_bytes = pdf.tobytes(garbage=4, deflate=True, no_new_id=True)
    svg_path.with_suffix(".pdf").write_bytes(pdf_bytes)
    # Audit the saved PDF, including its actual renderer, rather than the PNG alone.
    with pymupdf.open(svg_path.with_suffix(".pdf")) as pdf:
        page = pdf[0]
        expected = tuple(float(size) * 72 for size in size_inches)
        actual = (page.rect.width, page.rect.height)
        assert all(abs(a - b) < .001 for a, b in zip(actual, expected)), actual
        assert not page.get_fonts() and not page.get_images() and not page.get_text().strip()
        paths = len(page.get_drawings())
        assert paths > 0
        pixmap = page.get_pixmap(dpi=300, alpha=False)
        rendered = np.asarray(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
    with Image.open(png_path) as png:
        reference = np.asarray(png.convert("RGB"))
    # Rasterizers round the page edge differently by at most one pixel.
    assert all(abs(a - b) <= 1 for a, b in zip(rendered.shape[:2], reference.shape[:2]))
    height, width = (min(a, b) for a, b in zip(rendered.shape[:2], reference.shape[:2]))
    difference = np.abs(rendered[:height, :width].astype(float) - reference[:height, :width].astype(float))
    mae = float(difference.mean())
    changed_fraction = float(np.mean(difference.max(axis=2) > 50))
    assert mae < 6 and changed_fraction < .05, (mae, changed_fraction)
    return {"pdf_text_representation": "vector_outlines", "pdf_font_resources": 0,
            "pdf_raster_images": 0, "pdf_vector_paths": paths,
            "pdf_page_size_points": list(actual), "pdf_render_comparison_dpi": 300,
            "pdf_vs_png_mean_absolute_channel_error_0_255": mae,
            "pdf_vs_png_pixel_fraction_max_channel_error_above_50": changed_fraction,
            "pdf_render_comparison_passed": True}


# Workflow-derived accents organize the retrieval metrics. The component figure
# keeps its single green theme; each figure uses one shared 0–1 score scale.
WORKFLOW_THEMES = {
    "four_color": {"panels": ["#EFF5FA", "#F0F6F1", "#FCF7E9", "#F3EFF8"],
                   "accents": ["#6B94B7", "#6F9781", "#BE944F", "#9580B1"],
                   "track": "#E1E7E1",
                   "scores": ["#F3F6F3", "#D4E3D5", "#ADCDB7", "#7BA88E", "#4B7B63"]},
    "green": {"panel": "#F0F6F1", "accent": "#6F9781", "track": "#E1EAE3",
              "ink": "#243E31",
              "scores": ["#F2F7F3", "#D0E4D6", "#8FBDA0", "#4F926A", "#205539"]},
    "gold_green": {"panels": ["#FCF7E9", "#F0F6F1"],
                   "accents": ["#BE944F", "#6F9781"], "ink": "#293B35",
                   "scores": ["#FBF7E9", "#F2E4BC", "#DEC385", "#C49C50", "#906527"],
                   "secondary_scores": ["#F3F8F4", "#DCEADD", "#B2CEBA", "#7CA78E", "#3D7255"]},
    "purple_blue": {"panels": ["#F3EFF8", "#EFF5FA"],
                    "accents": ["#9580B1", "#6B94B7"], "ink": "#29343F",
                    "latency": "#80649C", "memory": "#5487B0",
                    "generation": ["#80649C", "#7CA6C8", "#F0F1F4"]},
}


def score_palette(theme):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        f"pbpk_score_{theme}", WORKFLOW_THEMES[theme]["scores"])


def color_key(fig, cmap, *, y, left=.59, width=.34):
    """Vector-only color key: avoids embedding raster gradients into SVG/PDF."""
    from matplotlib.patches import Rectangle
    ax = fig.add_axes([left, y, width, .026])
    for i in range(100):
        ax.add_patch(Rectangle((i / 100, 0), .0101, 1,
                               facecolor=cmap((i + .5) / 100), edgecolor="none"))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.set_axis_off()
    for x, label in [(0, "0"), (.5, "0.5"), (1, "1")]:
        fig.text(left + x * width, y - .007, label, fontsize=6.8, ha="center", va="top")
    fig.text(left - .025, y + .013, "Score", fontsize=8, ha="right", va="center")


def subgraph_matrix(plt, subgraph, summary):
    from matplotlib.patches import Rectangle
    theme = WORKFLOW_THEMES["four_color"]
    cmap = score_palette("four_color")
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 4.45))
    fig._pbpk_theme = "four_color"
    fig.text(.025, .955, "Subgraph retrieval: performance and seed stability", fontsize=11, weight="bold", color="#283E46")
    fig.text(.025, .907, "4 encoders × 4 metrics × 3 random seeds", fontsize=8.5, color="#4D6269")
    ax = fig.add_axes([.20, .17, .78, .64])
    ax.set(xlim=(0, 4), ylim=(4, 0)); ax.set_axis_off()
    metrics = ("ndcg_at_10", "recall_at_10", "reciprocal_rank", "relation_macro_f1")
    titles = ("NDCG@10", "Recall@10", "MRR", "Relation\nmacro-F1")
    labels = ("NoGNN", "R-GCN", "HGT", "Graph\nTransformer")
    for j, title in enumerate(titles):
        ax.add_patch(Rectangle((j + .025, -.49), .95, .44, clip_on=False,
                     facecolor=theme["panels"][j], edgecolor="none"))
        ax.plot([j + .025, j + .975], [-.49, -.49], color=theme["accents"][j],
                linewidth=1.5, clip_on=False)
        ax.text(j + .5, -.27, title, ha="center", va="center", fontsize=8.5, weight="bold")
    for i, row in enumerate(summary):
        ax.text(-.10, i + .48, labels[i], ha="right", va="center", fontsize=9,
                color="#243E49")
        for j, metric in enumerate(metrics):
            mean, sd = row[metric + "_mean"], row[metric + "_sd"]
            best = mean == max(r[metric + "_mean"] for r in summary)
            ax.add_patch(Rectangle((j + .025, i + .035), .95, .93,
                         facecolor=theme["panels"][j], edgecolor="#293C41" if best else "#DCE3E2",
                         linewidth=1.2 if best else .5))
            ax.text(j + .5, i + .18, f"{mean:.4f}", ha="center", va="center", fontsize=10.5,
                    weight="bold" if best else "normal", color="#143847")
            ax.text(j + .5, i + .36, f"± {sd:.4f}", ha="center", va="center", fontsize=7.3, color="#526970")
            for k, seed in enumerate(SEEDS):
                value = next(r[metric] for r in subgraph
                             if r["architecture"] == row["architecture"] and r["seed"] == seed)
                y = i + .54 + k * .16
                ax.add_patch(Rectangle((j + .075, y - .038), .52, .076,
                             facecolor=theme["track"], edgecolor="none"))
                ax.add_patch(Rectangle((j + .075, y - .038), .52 * value, .076,
                             facecolor=cmap(value), edgecolor="none"))
                ax.text(j + .79, y, f"{value:.4f}", ha="center", va="center", fontsize=6.8,
                        color="#264B59")
    fig.text(.025, .122, "Mean ± sample SD; bars: seeds 17 / 29 / 43, top to bottom (shared 0–1 scale).", fontsize=7.5, color="#41555E")
    fig.text(.025, .084, "Dark outline: highest mean per metric; not a significance indicator.", fontsize=7.5, color="#41555E")
    fig.text(.025, .035, "38 shared test queries", fontsize=8, color="#41555E")
    color_key(fig, cmap, y=.040)
    return fig


def component_matrix(plt, protocol, *, caption_notes=True):
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    theme = WORKFLOW_THEMES["green"]
    cmap = score_palette("green")
    height = 4.45 if caption_notes else 3.95
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, height))
    fig._pbpk_theme = "green"
    # When notes live in the manuscript caption, remove their reserved space
    # without changing the heatmap, request order, or physical cell sizes.
    bottom = .166 if caption_notes else (.166 * 4.45 - .5) / height
    plot_height = .810 if caption_notes else .810 * 4.45 / height
    ax = fig.add_axes([0, bottom, 1, plot_height])
    ax.set(xlim=(0, 35), ylim=(18.7, -1.65)); ax.set_axis_off()
    details = protocol["request_details"]
    lookup = {(r["request_index"], r["variant"]): r for r in details}
    assert len(lookup) == 100
    labels = ("No KG", "Text", "Soft", "Hybrid")
    groups = [(0, 11, "DDI victim"), (11, 7, "DDI perpetrator"), (18, 7, "Other scenarios")]
    for start, n, label in groups:
        ax.add_patch(Rectangle((9 + start, -1.44), n - .10, .72,
                     facecolor=theme["panel"], edgecolor="none"))
        ax.text(9 + start + (n - .10) / 2, -1.08, label, ha="center", va="center", fontsize=7.4, color="#294B58")
    blocks = [("Component\nF1", "component_ids", False),
              ("Exact-set\nmatch", "component_ids", True),
              ("Evidence\nF1", "evidence_ids", False),
              ("Gap\nF1", "gap_codes", False)]
    for group, (label, field, exact_match) in enumerate(blocks):
        ybase = group * 4.6
        ax.add_patch(Rectangle((.5, ybase -.06), 4.15, 4.12,
                     facecolor=theme["panel"], edgecolor="none"))
        ax.plot([.5, .5], [ybase-.06, ybase+4.06], color=theme["accent"], linewidth=1.6)
        ax.text(2.575, ybase + 2, label, ha="center", va="center", fontsize=8.1, weight="bold", color="#244C59")
        for i, (variant, method) in enumerate(zip(VARIANTS, labels)):
            y = ybase + i
            ax.text(8.25, y + .5, method + ("†" if group == 2 and i == 0 else ""),
                    ha="right", va="center", fontsize=8, color="#2F4550")
            for request_index in range(1, 26):
                field_data = lookup[request_index, variant]["fields"][field]
                value = float(field_data["exact"]) if exact_match else field_data["f1"]
                x = 8 + request_index
                ax.add_patch(Rectangle((x + .025, y + .045), .95, .91,
                             facecolor="#E7E9E8" if value is None else cmap(value), edgecolor="none"))
                if value is None:
                    ax.scatter(x + .5, y + .5, marker="D", s=8, facecolors="none", edgecolors="#526167", linewidths=.65)
        for x in (20, 27):
            ax.plot([x, x], [ybase-.06, ybase+4.06], color="#889B9E", linewidth=.8)
    for q in range(1, 26):
        ax.text(8.5 + q, 18.04, str(q), ha="center", va="center", fontsize=6.3, color="#50656D")
    ax.text(4.6, 18.04, "Request ID", ha="center", va="center", fontsize=7.4, color="#50656D")
    if caption_notes:
        fig.text(.025, .129, "◇ Both sets empty: F1 undefined. Exact-set cells are binary (0 / 1).", fontsize=7.5, color="#000000")
        fig.text(.025, .088, "† No KG has no candidate citations. Request order is not based on scores.", fontsize=7.5, color="#000000")
        fig.text(.025, .032, "One column = one test request", fontsize=7.5, color="#000000")
    color_key(fig, cmap, y=.032, left=.62, width=.31)
    for label in fig.findobj(Text):
        label.set_color("#000000")
    return fig


def model_stability_matrix(plt, report):
    """Reported matched-execution stability, never independent recommendation quality."""
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    row_count = len(report["rows"])
    full_height = 4.55 + max(0, row_count - 12) * .27
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, full_height))
    from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex
    fig._pbpk_theme = "gold_green"
    theme = WORKFLOW_THEMES["gold_green"]
    # Preserve the gold/green scales while keeping black numerals readable.
    palettes = {key: [to_hex(tuple(c + .15 * (1 - c) for c in to_rgb(color)))
                      for color in colors]
                for key, colors in (("reorder", theme["scores"]),
                                    ("missing", theme["secondary_scores"]))}
    fig._pbpk_score_colors_by_condition = {
        "sentence_reordering": palettes["reorder"], "evidence_removal": palettes["missing"]}
    cmaps = {key: LinearSegmentedColormap.from_list(f"pbpk_stability_{key}_black_text", colors)
             for key, colors in palettes.items()}
    ink = "#000000"
    fig.text(.025, .836, "Scoring configuration", fontsize=8, weight="bold", color=ink)
    fig.text(.400, .867, "Original\nTop-1 ties", ha="center", va="center", fontsize=7.6, color=ink)
    for j, (center, title) in enumerate(((.582, "Sentence reordering"), (.844, "Evidence removal"))):
        fig.add_artist(Rectangle((center-.127, .805), .247, .078, transform=fig.transFigure,
                                 facecolor=theme["panels"][j], edgecolor="none", zorder=0))
        fig.text(center, .862, title, ha="center", fontsize=8.3, weight="bold", color=ink)
    columns = [("reorder_top1", .470), ("reorder_top10", .555),
               ("missing_top1", .730), ("missing_top10", .815)]
    for (key, left), title in zip(columns, ("Top-1", "Top-10", "Top-1", "Top-10")):
        fig.text(left + .0375, .821, title, ha="center", fontsize=7.5, color=ink)
    for x in (.670, .930):
        fig.text(x, .821, "n", ha="center", fontsize=7.5, style="italic", color=ink)
    fig.text(.025, .798, "Full-catalog scoring", fontsize=7.2, style="italic", color="#58666B")
    changes = sum(a["group"] != b["group"] for a, b in zip(report["rows"], report["rows"][1:]))
    step = (.480 - changes * .030) / max(row_count - 1, 1)
    row_height = min(.035, .85 * step)
    previous, group_offset = None, 0.
    for i, row in enumerate(report["rows"]):
        if previous is not None and row["group"] != previous:
            group_offset += .030
        y = .765 - i * step - group_offset
        if previous is not None and row["group"] != previous:
            heading = "Grouped held-out PBPK-Match" if row["group"] == "trained" else "BM25 Top-20 reranking"
            fig.text(.025, y + .024, heading, fontsize=7.2, style="italic", color="#58666B")
        previous = row["group"]
        fig.text(.347, y, row["method"], ha="right", va="center", fontsize=7.2,
                 weight="bold" if row["group"] == "trained" else "normal", color="#303F45")
        fig.add_artist(Rectangle((.363, y-row_height/2), .074, row_height, transform=fig.transFigure,
                                facecolor="#F1F3F3", edgecolor="none"))
        fig.text(.400, y+.002, f'{100 * row["tie_fraction"]:.1f}%', ha="center", va="center", fontsize=7.2, color="#3F4A4F")
        fig.add_artist(Rectangle((.369, y-row_height*.4), .062 * row["tie_fraction"], .003,
                                transform=fig.transFigure, facecolor="#67747A", edgecolor="none"))
        for key, left in columns:
            value = row[key]
            cmap = cmaps[key.split("_")[0]]
            background = (.906, .914, .914, 1) if value is None else cmap(value)
            fig.add_artist(Rectangle((left, y-row_height/2), .075, row_height, transform=fig.transFigure,
                                    facecolor=background,
                                    edgecolor="#A4AAAB" if value is None else "none",
                                    linewidth=.3))
            marker = "†" if row["method_id"] == "gfm_rag_8m" and key.endswith("top10") else ""
            fig.text(left+.0375, y, "NA" if value is None else f"{value:.3f}{marker}",
                     ha="center", va="center", fontsize=7.7,
                     color=ink)
        for x, key in ((.670, "reorder_n"), (.930, "missing_n")):
            fig.text(x, y, str(row[key]), ha="center", va="center", fontsize=7.7, color="#3F4A4F")
    fig.text(.025, .219, "Top-1: consistency; Top-10: overlap; n: matched execution pairs.", fontsize=7.4, color="#41555E")
    fig.text(.025, .179, "NA: no unique Top-1. † Top-10 ties use frozen candidate order.", fontsize=7.4, color="#41555E")
    fig.text(.025, .139, "Input stability is not recommendation correctness.", fontsize=8, weight="bold", color=ink)
    # Each perturbation has its own labeled, equally normalized sequential scale.
    # Color keys use exactly the same lightened fills as the numeric cells.
    for key, label, y in (("reorder", "Reordering", .088), ("missing", "Evidence removal", .052)):
        ax = fig.add_axes([.64, y, .30, .019])
        for i in range(100):
            ax.add_patch(Rectangle((i/100, 0), .0101, 1,
                                   facecolor=cmaps[key]((i+.5)/100), edgecolor="none"))
        ax.set(xlim=(0,1), ylim=(0,1)); ax.set_axis_off()
        fig.text(.62, y+.0095, label, ha="right", va="center", fontsize=7.2, color=ink)
    for x, label in ((0,"0"), (.5,"0.5"), (1,"1")):
        fig.text(.64+x*.30, .043, label, ha="center", va="top", fontsize=6.8, color=ink)
    fig.text(.025, .074, "Shared 0–1 scale", fontsize=7.5, color="#41555E")
    fig.text(.025, .039, "Darker = higher stability", fontsize=7.5, color="#41555E")
    for label in fig.findobj(Text):
        label.set_color("#000000")
    # Trim only the removed title band, preserving physical row spacing and fonts.
    compact_height = round(full_height * .91, 2)
    vertical_scale = full_height / compact_height
    for label in fig.texts:
        x, y = label.get_position()
        label.set_position((x, y * vertical_scale))
    for patch in fig.artists:
        if isinstance(patch, Rectangle):
            patch.set_y(patch.get_y() * vertical_scale)
            patch.set_height(patch.get_height() * vertical_scale)
    for axis in fig.axes:
        box = axis.get_position()
        axis.set_position([box.x0, box.y0 * vertical_scale, box.width, box.height * vertical_scale])
    fig.set_size_inches(FIGURE_WIDTH_INCHES, compact_height)
    return fig


def model_operations_figure(plt, report):
    from matplotlib.patches import Patch
    import numpy as np
    rows = report["rows"]
    theme = WORKFLOW_THEMES["purple_blue"]
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 5.7))
    fig._pbpk_theme = "purple_blue"
    fig.text(.025, .963, "Model scoring: execution costs and output validity", fontsize=10.5, weight="bold", color=theme["ink"])
    fig.text(.025, .924, "Recorded costs · Heterogeneous precision and batching settings", fontsize=8, color="#48565B")
    fig.text(.355, .881, "(a) Serial latency", fontsize=8.5, weight="bold", color="#654D7B")
    fig.text(.754, .881, "(b) GPU allocation", fontsize=8.5, weight="bold", color="#355C7B")
    ax = fig.add_axes([.355, .444, .30, .397])
    mem = fig.add_axes([.755, .444, .218, .397])
    positions = np.array([i + (.55 if i >= 9 else 0) + (.55 if i >= 12 else 0) for i in range(len(rows))])
    for axis in (ax, mem):
        axis.set_ylim(float(positions[-1]) + .60, -.65)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#ABB2B5")
        axis.tick_params(axis="both", length=2, labelsize=7)
        axis.set_yticks([])
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#E7E9EE", linewidth=.5)
    ax.set(xscale="log", xlim=(.002, 90))
    ax.set_xticks([.01, .1, 1, 10], labels=["0.01", "0.1", "1", "10"])
    ax.minorticks_off()
    ax.set_xlabel("Seconds (log scale)", fontsize=7.7, labelpad=4)
    mem.set(xlim=(0, 28))
    mem.set_xticks([0, 10, 20])
    mem.set_xlabel("Peak allocated GiB", fontsize=7.7, labelpad=4)
    for y, row in zip(positions, rows):
        ax.text(-.045, y, row["method"], transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.0, color="#303F45")
        if row["p50_seconds"] is None:
            ax.text(.003, y, "Mixed modes", va="center", fontsize=7, color="#586C73")
        else:
            p50, p95 = row["p50_seconds"], row["p95_seconds"]
            ax.plot([p50, p95], [y, y], color=theme["accents"][0], linewidth=1.1)
            ax.plot(p50, y, "o", color=theme["latency"], markersize=3.5)
            ax.plot(p95, y, "s", markerfacecolor="white", markeredgecolor=theme["latency"], markersize=3.5)
        mem.barh(y, row["peak_gpu_gib"], height=.5, color=theme["memory"])
        if row["peak_gpu_gib"] == 0:
            mem.plot(0, y, "o", color="#355C7B", markersize=2, clip_on=False)
        mem.text(27.8, y, f'{row["peak_gpu_gib"]:.2f}', ha="right", va="center", fontsize=7, color=theme["ink"])
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", color=theme["latency"], linestyle="none", markersize=3.5, label="P50"),
                        Line2D([], [], marker="s", markerfacecolor="white", color=theme["latency"], linestyle="none", markersize=3.5, label="P95")],
               loc="center", bbox_to_anchor=(.51, .862), frameon=False, ncol=2, fontsize=7, handletextpad=.35, columnspacing=1)
    fig.text(.025, .358, "(c) Generation validity: 2,316 query–condition combinations per pipeline", fontsize=8.6, weight="bold", color=theme["ink"])
    colors = theme["generation"]
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#7D8194", label=t)
                        for c,t in zip(colors,["Strictly valid", "Postprocessed", "BM25 fallback"])],
               loc="center", bbox_to_anchor=(.53,.316), frameon=False, ncol=3, fontsize=7.5, handlelength=1.8, columnspacing=1.4)
    gen = fig.add_axes([.355,.177,.615,.112])
    gen.set(xlim=(0,160), ylim=(2.7,-.7));gen.set_axis_off()
    for i,row in enumerate(r for r in report["generation"] if r.get("strict_valid") is not None):
        counts = [row["strict_valid"], row["postprocessed_or_retried_valid"], row["fallback"]]
        gen.text(-3, i, row["method"], ha="right", va="center", fontsize=7.1, color="#303F45")
        left = 0
        for count, color in zip(counts, colors):
            width = count / row["total"] * 100
            gen.barh(i, width, left=left, color=color, edgecolor="#7D8194", linewidth=.35, height=.62)
            left += width
        gen.text(105, i, " / ".join(map(str,counts)), va="center", fontsize=7.3, color=theme["ink"])
    fig.text(.355,.146,"Counts: strict / postprocessed / fallback",fontsize=7.2,color="#41555E")
    note = "Legacy latency: n = 771; PBPK-Match: 50 held-out requests, query encoding included." if len(rows) > 12 else "Latency excludes the first original request (n = 771); batch costs are separate."
    fig.text(.025,.096,note,fontsize=6.8,color="#41555E")
    fig.text(.025,.060,"GPU allocation is not total device use; exclusive GPU access was not verified.",fontsize=7.2,color="#41555E")
    note = "PBPK-Match is non-generative; generation recovery is not applicable." if len(rows) > 12 else "Completed scoring includes recovered outputs and fallback rankings."
    fig.text(.025,.023,note,fontsize=7.5,color=theme["ink"])
    return fig


def figures(component, subgraph, summary, component_protocol, model_report, *, preview_only):
    plt = plotting_environment()
    destination = OUT / "figures/experiments"; destination.mkdir(parents=True, exist_ok=True)
    audits = {}
    figure_specs = [
        ("subgraph_retrieval", subgraph_matrix(plt, subgraph, summary), "integrated seed-performance matrix"),
        ("component_recommendation", component_matrix(plt, component_protocol), "paired request-level heatmap without an internal title; aggregate metrics reported in the accompanying text"),
        ("model_recommendation_stability", model_stability_matrix(plt, model_report), "matched-execution input-stability matrix with denominators and ties"),
        ("model_recommendation_operations", model_operations_figure(plt, model_report), "recorded serial costs, allocated memory and generation-validity composition"),
    ]
    for name, fig, design in figure_specs:
        audits[name] = figure_audit(fig)
        audits[name]["design"] = design
        if name == "model_recommendation_stability":
            audits[name]["score_colors_by_condition"] = fig._pbpk_score_colors_by_condition
        if name == "model_recommendation_operations":
            audits[name]["score_color_range"] = None
            audits[name]["score_colors"] = None
            theme = WORKFLOW_THEMES["purple_blue"]
            audits[name]["colors_by_measurement"] = {
                "latency": theme["latency"], "gpu_allocation": theme["memory"],
                "generation_outcomes": theme["generation"]}
            audits[name]["axis_units"] = {"latency": "seconds, log scale", "gpu": "allocated GiB",
                                          "generation": "fraction of 2316 query-condition combinations"}
        fig.savefig(destination / f"{name}.png", dpi=300, facecolor="white")
        if not preview_only:
            fig.savefig(destination / f"{name}.svg", metadata={"Date": None})
            audits[name].update(export_outlined_pdf(destination / f"{name}.svg", destination / f"{name}.png", fig.get_size_inches()))
        plt.close(fig)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-only", action="store_true", help="Write PNG previews before vector export")
    args = parser.parse_args()
    component, component_protocol = component_results()
    graph, graph_summary, graph_protocol = subgraph_results()
    frozen_graph = source(BASE / "107_runtime_graph_gate5_locked/kg_build_summary.json")
    import_record = source(ROOT / "data_process/final_evidence/run_20260909T083120Z/00_run_manifest/database_import_summary.json")
    knowledge_base = import_record["database_readback_verification"]
    assert frozen_graph["evidence_count"] == 1410
    assert frozen_graph["node_count"] == 4149 and frozen_graph["edge_count"] == 9307
    assert knowledge_base["passed"] and knowledge_base["total_serving_evidence_count"] == 2872
    assert knowledge_base["evidence_covered_compound_count"] == 30
    write_csv(OUT / "tables/component_recommendation.csv", component)
    write_csv(OUT / "tables/subgraph_retrieval.csv", graph_summary)
    write_tex(component, graph_summary, graph)
    from model_recommendation_paper import build_model_results
    model_report = build_model_results(ROOT, OUT)
    SOURCE_INDEX.update(model_report.pop("source_files"))
    quality = None
    quality_path = OUT / "data/model_recommendation_quality.json"
    if quality_path.exists():
        quality = source(quality_path)
        if quality["state"] != "complete":
            raise ValueError("Paper quality export is incomplete")
        for relative, expected_hash in quality["source_files"].items():
            record_source(ROOT / relative)
            validate_hash(ROOT / relative, expected_hash)
    audits = figures(component, graph, graph_summary, component_protocol, model_report, preview_only=args.preview_only)
    record_source(Path(__file__).resolve())
    report = {"schema_version": "pbpk-paper-results-1.2", "generated_from_saved_outputs_only": True,
              "component_recommendation": {"rows": component, "protocol": component_protocol},
              "subgraph_retrieval": {"seed_rows": graph, "summary": graph_summary, "protocol": graph_protocol},
              "model_recommendation": model_report,
              "resource_snapshots": {
                  "frozen_component_graph": {key: frozen_graph[key] for key in ("evidence_count", "node_count", "edge_count", "event_count", "node_type_counts")},
                  "later_knowledge_base": knowledge_base,
                  "later_snapshot_used_for_component_evaluation": False},
              "source_files": SOURCE_INDEX, "figure_audit": audits,
              "model_recommendation_results_included": True,
              "model_recommendation_quality_evaluation_complete": quality is not None,
              "model_recommendation_quality": ({
                  "file": "results/data/model_recommendation_quality.json", "state": quality["state"],
                  "scope": quality["scope"], "independent_scientific_validation": False
              } if quality is not None else None),
              "reproduce": "runtime/app-env/bin/python scripts/build_paper_results.py"}
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "data/paper_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"source_count": len(SOURCE_INDEX), "component_rows": len(component),
                      "subgraph_seed_rows": len(graph), "all_recomputed_metrics_match": True,
                      "figure_audit": audits}, ensure_ascii=False))


if __name__ == "__main__":
    main()
