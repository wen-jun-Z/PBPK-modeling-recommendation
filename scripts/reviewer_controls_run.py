#!/usr/bin/env python3
"""Frozen 5x3 internal controls; no LLM inference or candidate PBPK refitting."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
from pbpk_kb.recommendation.pbpk_match_v2.experiment import sha, read, write, array_write
from pbpk_kb.recommendation.pbpk_match_v2.evaluation import fit_oof_temperature, calibrate
from reviewer_controls_models import METHODS, trainer, identity_condition_rule, visible_graph

OUT = ROOT / "experiments/model_recommendation/reviewer_revision_20261009/controls"
LIGHT = ROOT / "experiments/model_recommendation/pbpk_match_light"
RULES = ("identity_condition_rule", "service_typed_rule")


def reference(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Project scope only")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def prepare():
    from run_pbpk_match_light import prepare as original_prepare
    data = original_prepare()
    old = read(LIGHT / "protocol.json")
    assert data["folds"] == old["folds"] and list(data["groups"]) == old["groups"]
    assert len(data["pair_feature_names"]) == 204
    for path, digest in old["implementation_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("Original frozen training implementation changed: " + path)
    return data


def freeze(data):
    code = [Path(__file__), ROOT / "scripts/reviewer_controls_models.py", ROOT / "scripts/pbpk_match_light_training.py",
            *sorted((ROOT / "src/pbpk_kb/recommendation/pbpk_match_v2").glob("*.py")),
            ROOT / "src/pbpk_kb/recommendation/model_matching/scoring.py",
            ROOT / "src/pbpk_kb/recommendation/model_matching/graphs.py"]
    payload = {"schema": "reviewer-information-matched-controls-v1", "methods": list(METHODS), "rule_methods": list(RULES),
        "seed": 17, "original_protocol": reference(LIGHT / "protocol.json"),
        "dataset_manifest": reference(ROOT / "experiments/model_recommendation/pbpk_match_v2/manifest.json"),
        "cache_manifest": reference(LIGHT / "cache/manifest.json"),
        "implementation_sha256": {str(p.relative_to(ROOT)): sha(p) for p in code},
        "query_ids": [r["query_id"] for r in data["query_records"]],
        "candidate_ids": [r["candidate_id"] for r in data["candidate_records"]],
        "groups": list(data["groups"]), "folds": data["folds"], "outer_folds": 5, "inner_folds": 3, "fits_per_method": 20,
        "max_epochs": 40, "patience": 5, "learning_rate": .001, "weight_decay": .001, "width": 128, "dropout": .1,
        "loss": "Unmodified PBPK-Match masked pairwise+ordinal+Y/N/U loss, same train-only class/query/family weights",
        "supervision": "Same 145 core document labels; 41 execution requests retain missing document labels; no PK error supervision",
        "error_correction": "Disabled in all controls; primary architectural comparator is original document-only PBPK-Match",
        "refit_rule": "Median of inner best epochs; no outer-fold early stopping or model selection",
        "calibration": "Original inner-OOF temperature calibration, grouped weights; no outer labels",
        "training_batch": "One complete query with every one of 280 candidates, no negative sampling",
        "models": {"structured_mlp": "204 normalized typed/numeric features -> 128 hidden -> same heads",
                   "text_mlp": "Shared projected frozen query/candidate text, concatenation -> 128 hidden -> same heads",
                   "concat_mlp": "Shared projected text concatenated with 204 features -> 128 hidden -> same heads"},
        "rules": {"identity_condition_rule": "4 drug coverage + 2 route coverage + species coverage + mean numeric/food/formulation exact condition support; missing candidate data gives no support",
                  "service_typed_rule": "Unmodified appendix-B score_candidate on same visible typed fields; no live engineering/use-tier gate; no dose support in this original service scorer"},
        "rule_probabilities": "Unavailable: deterministic rule scores are not probabilities; no invented calibration",
        "text_encoder_frozen": True, "external_methods_rerun": False, "independent_external_validation": False,
        "all_variants": list(data["variant_names"]), "statistics": "Existing unknown masks, exact ties, endpoint separation, grouped paired statistics"}
    path = OUT / "protocol.json"
    if path.exists() and read(path) != payload:
        raise ValueError("Frozen controls protocol changed; refuse mixed runs")
    if not path.exists():
        write(path, payload)
    return payload


def status(phase, **fields):
    path = OUT / "status.json"
    old = read(path) if path.exists() else {}
    payload = {**old, **fields, "phase": phase, "pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat()}
    write(path, payload)
    print(phase, fields, flush=True)
    if (OUT / "stop.requested").exists():
        raise InterruptedError("Project controls stop requested")


def integrity(module, data, indices, checkpoint, device):
    chosen = sorted(indices)[:2]
    plain = module.predict_fold(data, chosen, checkpoint, device=device)
    repeated = module.predict_fold(data, chosen, checkpoint, device=device)
    small = module.predict_fold(data, chosen, checkpoint, device=device, batch_size=140)
    order = np.random.default_rng(17).permutation(len(data["candidate_records"]))
    altered = dict(data)
    altered["candidate_records"] = [data["candidate_records"][i] for i in order]
    altered["light_candidate_embeddings"] = data["light_candidate_embeddings"][order]
    altered["pair_features_by_variant"] = data["pair_features_by_variant"][:, :, order]
    # Prediction must not read any labels or error targets.
    for key in ("grades", "applicability", "error_records"):
        altered.pop(key, None)
    permuted = module.predict_fold(altered, chosen, checkpoint, device=device)
    checks = {"repeat": np.array_equal(plain["scores"], repeated["scores"]),
              "batch_size": np.array_equal(plain["scores"], small["scores"]),
              "candidate_order": np.array_equal(plain["scores"], permuted["scores"][:, np.argsort(order)]),
              "inference_without_outcomes": True}
    if not all(checks.values()):
        raise ValueError("Inference reproducibility failed: " + str(checks))
    return {"query_indices": chosen, "candidate_count": len(order), **checks}


def fit_job(module, data, tr, va, job, *, fixed_epochs=None, device, method, fold, kind):
    config = {"method": method, "protocol_sha256": sha(OUT / "protocol.json"),
              "train_indices": tr, "validation_indices": va, "fixed_epochs": fixed_epochs}
    job.mkdir(parents=True, exist_ok=True)
    if (job / "binding.json").exists() and read(job / "binding.json") != config:
        raise ValueError("Control job binding changed")
    write(job / "binding.json", config)
    def callback(event):
        if event.get("stage") == "epoch_complete":
            status("training", method=method, fold=fold, fit=kind, epoch=event["epoch"],
                   best_epoch=event["best_epoch"], elapsed_fit_seconds=event["elapsed_seconds"])
    return module.train_fold(data, tr, va, job, seed=17, device=device, epochs=fixed_epochs or 40,
        patience=5, learning_rate=.001, fixed_epochs=fixed_epochs is not None,
        pk_epochs=1, error_train_indices=[], callback=callback)


def train_control(data, method, device):
    module = trainer(method)
    supervised = set(data["core_indices"]) | set(data["execution_indices"])
    destination = OUT / method
    for f, fold in enumerate(data["folds"]):
        folder = destination / f"runs/seed17/fold{f}"
        indices, probabilities, epochs = [], [], []
        for j, split in enumerate(fold["inner"]):
            tr = sorted(supervised & set(split["train"]))
            va = sorted(supervised & set(split["validation"]))
            job = folder / f"inner{j}"
            meta = fit_job(module, data, tr, va, job, device=device, method=method, fold=f, kind=f"inner{j}")
            prediction = module.predict_fold(data, va, job, device=device)
            array_write(job / "validation.npz", indices=np.array(va), scores=prediction["scores"], probabilities=prediction["probabilities"])
            write(job / "complete.json", {"state": "complete", "metadata": reference(job / "metadata.json"),
                "weights": reference(job / "heads.pt"), "predictions": reference(job / "validation.npz"), "binding": reference(job / "binding.json")})
            indices += va
            probabilities.append(prediction["probabilities"])
            epochs.append(meta["best_epoch"])
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate inner OOF requests")
        if set(data["groups"][i] for i in indices) & set(data["groups"][i] for i in fold["outer_test"]):
            raise ValueError("Outer labels enter calibration")
        cal = fit_oof_temperature(np.concatenate(probabilities), np.asarray(data["applicability"])[indices], [data["groups"][i] for i in indices])
        refit_epochs = max(1, int(np.median(epochs)))
        write(folder / "selection.json", {"refit_epochs": refit_epochs, "inner_best_epochs": epochs,
            "calibration": cal, "selection_indices": indices, "outer_indices_not_used": fold["outer_test"], "error_weight": 0.})
        tr = sorted(supervised & set(fold["outer_train"]))
        te = list(fold["outer_test"])
        job = folder / "outer"
        fit_job(module, data, tr, [], job, device=device, method=method, fold=f, kind="outer", fixed_epochs=refit_epochs)
        scores, probs = [], []
        for variant in data["variant_names"]:
            status("predicting", method=method, fold=f, variant=variant)
            prediction = module.predict_fold(data, te, job, variant=variant, device=device)
            scores.append(prediction["scores"])
            probs.append(calibrate(prediction["probabilities"], cal["temperature"]))
        array_write(job / "predictions.npz", indices=np.asarray(te), scores=np.asarray(scores), probabilities=np.asarray(probs))
        write(job / "integrity.json", integrity(module, data, te, job, device))
        write(job / "complete.json", {"state": "complete", "metadata": reference(job / "metadata.json"),
            "weights": reference(job / "heads.pt"), "predictions": reference(job / "predictions.npz"),
            "binding": reference(job / "binding.json"), "selection": reference(folder / "selection.json"), "integrity": reference(job / "integrity.json")})
    q, c = len(data["query_records"]), len(data["candidate_records"])
    scores = np.full((3, q, c), np.nan, np.float32)
    probabilities = np.full((3, q, c, 3), np.nan, np.float32)
    coverage = np.zeros(q, int)
    for f in range(5):
        with np.load(destination / f"runs/seed17/fold{f}/outer/predictions.npz") as arrays:
            ix = arrays["indices"]
            scores[:, ix], probabilities[:, ix] = arrays["scores"], arrays["probabilities"]
            coverage[ix] += 1
    if not (coverage == 1).all() or not np.isfinite(scores).all() or not np.isfinite(probabilities).all():
        raise ValueError("Incomplete unique held-out candidate output")
    path = destination / "predictions/seed17.npz"
    array_write(path, indices=np.arange(q), scores=scores, base_scores=scores, probabilities=probabilities)
    write(path.with_suffix(".json"), {"state": "complete", "method": method, "protocol": reference(OUT / "protocol.json"),
        "predictions": reference(path), "fits": [reference(destination / f"runs/seed17/fold{f}/{kind}/complete.json")
            for f in range(5) for kind in ("inner0", "inner1", "inner2", "outer")]})
    status("method_complete", method=method)


def rules(data):
    from pbpk_kb.recommendation.model_matching.scoring import score_candidate
    rule = identity_condition_rule(data["pair_features_by_variant"], data["pair_feature_names"])
    q, c = len(data["query_records"]), len(data["candidate_records"])
    service = np.empty((3, q, c), np.float32)
    candidates = [visible_graph(row, candidate=True) for row in data["candidate_records"]]
    cache = {}
    for v, variant in enumerate(data["variant_names"]):
        for i, query in enumerate(data["query_records"]):
            graph = visible_graph(query, variant)
            key = tuple((n.node_type, n.semantic_key, n.label) for n in graph.nodes)
            if key not in cache:
                cache[key] = np.array([score_candidate(graph, model).score for model in candidates], np.float32)
            service[v, i] = cache[key]
        status("rule_scoring", variant=variant, distinct_graphs=len(cache))
    for method, scores in zip(RULES, (rule, service)):
        path = OUT / method / "predictions/seed17.npz"
        array_write(path, indices=np.arange(q), scores=scores, base_scores=scores,
                    probabilities=np.full((3, q, c, 3), np.nan, np.float32))
        write(path.with_suffix(".json"), {"state": "complete", "method": method, "probabilities_available": False,
            "protocol": reference(OUT / "protocol.json"), "predictions": reference(path), "fits": []})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "freeze", "status"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--method", choices=(*METHODS, "rules", "all"), default="all")
    args = parser.parse_args()
    if args.action == "status":
        print((OUT / "status.json").read_text() if (OUT / "status.json").exists() else "Not started")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = prepare()
        freeze(data)
        if args.action == "freeze":
            print("Frozen", OUT / "protocol.json")
            return
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("Requested project GPU unavailable")
            free, total = torch.cuda.mem_get_info(args.device)
            if free < 2 * 1024**3:
                raise RuntimeError("Require 2GiB free; no other task will be inspected or changed")
            torch.cuda.set_per_process_memory_fraction(min(6 * 1024**3, total) / total, args.device)
        started = time.monotonic()
        try:
            status("starting", method=args.method)
            for method in METHODS if args.method == "all" else (args.method,):
                if method != "rules":
                    train_control(data, method, args.device)
            if args.method in {"all", "rules"}:
                rules(data)
            status("complete", elapsed_seconds=time.monotonic() - started)
        except BaseException as exc:
            status("failed", error=str(exc))
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
