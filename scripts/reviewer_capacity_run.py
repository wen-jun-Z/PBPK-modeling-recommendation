#!/usr/bin/env python3
"""Grouped, three-seed capacity control using frozen inputs and outcome tables."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from pbpk_kb.recommendation.pbpk_match_v2.experiment import read, write, sha, array_write
from pbpk_kb.recommendation.pbpk_match_v2.evaluation import fit_oof_temperature, calibrate
from reviewer_controls_run import prepare, reference, integrity, LIGHT
from reviewer_capacity_models import METHODS, SEEDS, trainer, parameter_counts

OUT = ROOT / "experiments/model_recommendation/reviewer_revision_20261009/capacity_control"
SCRIPT = Path(__file__).resolve()
PYTHON = ROOT / "runtime/app-env/bin/python"
JOBS = [(m, s) for s in SEEDS for m in METHODS if (m, s) != ("pbpk_interaction", 17)]


def freeze(data):
    code = [SCRIPT, ROOT / "scripts/reviewer_capacity_models.py", ROOT / "scripts/reviewer_controls_models.py",
            ROOT / "scripts/reviewer_controls_run.py", ROOT / "scripts/pbpk_match_light_training.py"]
    code += sorted((ROOT / "src/pbpk_kb/recommendation/pbpk_match_v2").glob("*.py"))
    payload = {"schema": "pbpk-capacity-interaction-ablation-v1", "methods": list(METHODS), "seeds": list(SEEDS),
        "original_protocol": reference(LIGHT / "protocol.json"),
        "dataset_manifest": reference(ROOT / "experiments/model_recommendation/pbpk_match_v2/manifest.json"),
        "cache_manifest": reference(LIGHT / "cache/manifest.json"),
        "reused_seed17": reference(LIGHT / "predictions/seed17.npz"),
        "implementation_sha256": {str(p.relative_to(ROOT)): sha(p) for p in code},
        "query_ids": [r["query_id"] for r in data["query_records"]],
        "candidate_ids": [r["candidate_id"] for r in data["candidate_records"]],
        "groups": list(data["groups"]), "folds": data["folds"],
        "variants": list(data["variant_names"]), "outer_folds": 5, "inner_folds": 3,
        "new_fits": 100, "reused_fits": 20, "parameters": parameter_counts(),
        "projection_width": 128, "fusion_width": {"pbpk_interaction": 128, "concat_capacity": 198},
        "max_epochs": 40, "patience": 5, "learning_rate": .001, "weight_decay": .001, "dropout": .1,
        "batch": "One whole request, all 280 candidates, unchanged masked loss and group/family weights",
        "selection": "Median inner best epochs; calibration from inner OOF only; no outer label selection",
        "supervision": "Same document labels only; no measured execution error targets; error correction disabled",
        "capacity_selection": "Fusion width chosen analytically by parameter count, without viewing new scores",
        "primary_analysis": "Mean across the three seed-specific per-request known-subset NDCG@10 values, then paired by frozen drug/study group; PBPK minus concat; 95% cluster bootstrap CI and paired permutation p",
        "secondary_analysis": "Each seed and rank ensemble; full145 support/unknown, applicability, same-scenario endpoint errors and coverage, stability; no best-seed reporting",
        "multiplicity": "One prespecified primary contrast; secondary families retain Holm correction and are descriptive",
        "interpretation": "Grouped internal development evidence; no equivalence claim from nonsignificance; no full-list reliability claim from judged-only ranking",
        "text_encoder_training": False, "new_llm_labels": False, "external_reruns": False, "new_pbpk_execution": False}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists() and read(path) != payload:
        raise ValueError("Frozen capacity protocol/input/code changed; refusing mixed experiment")
    if not path.exists():
        write(path, payload)
    return payload


def status(method, seed, phase, **fields):
    path = OUT / "state" / f"{method}_seed{seed}.json"
    old = read(path) if path.exists() else {}
    value = {**old, **fields, "method": method, "seed": seed, "phase": phase,
             "pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat()}
    write(path, value)
    print(f"{method} seed{seed} {phase}: {fields}", flush=True)
    if (OUT / "stop.requested").exists() and phase not in {"failed", "paused"}:
        raise InterruptedError("Project capacity control stop requested")


def verify_reference_tree(ref, seen=None):
    """Validate a fit receipt and its actual weights, including the old schema."""
    seen = set() if seen is None else seen
    path = (ROOT / ref["path"]).resolve()
    if reference(path) != ref:
        raise ValueError("Completed artifact was modified: " + ref["path"])
    if path in seen:
        return
    seen.add(path)
    if path.name != "complete.json":
        return
    payload = read(path)
    for filename, digest in payload.get("artifacts_sha256", {}).items():
        child = (path.parent / filename).resolve()
        if not child.is_relative_to(path.parent):
            raise ValueError("Artifact escapes fit directory")
        verify_reference_tree({"path": str(child.relative_to(ROOT)), "sha256": digest}, seen)
    for value in payload.values():
        if isinstance(value, dict) and set(value) == {"path", "sha256"}:
            verify_reference_tree(value, seen)


def receipt_valid(path):
    path = Path(path)
    if not path.exists():
        return False
    receipt = read(path)
    if receipt.get("state") != "complete":
        return False
    if receipt.get("protocol") != reference(OUT / "protocol.json"):
        raise ValueError("Completed prediction has another protocol")
    seen = set()
    for ref in [receipt["predictions"], *receipt.get("fits", []), *receipt.get("reused_sources", [])]:
        verify_reference_tree(ref, seen)
    return True


def publish_predictions(data, method, seed):
    destination = OUT / method
    q, c = len(data["query_records"]), len(data["candidate_records"])
    shape = (len(data["variant_names"]), q, c)
    scores, probs = np.full(shape, np.nan, np.float32), np.full((*shape, 3), np.nan, np.float32)
    covered = np.zeros(q, int)
    fits = []
    for f in range(5):
        folder = destination / f"runs/seed{seed}/fold{f}"
        with np.load(folder / "outer/predictions.npz", allow_pickle=False) as a:
            ix = a["indices"]
            if not np.array_equal(ix, data["folds"][f]["outer_test"]) or len(set(ix.tolist())) != len(ix):
                raise ValueError("Fold indices differ from unique frozen outer holdout")
            scores[:, ix], probs[:, ix] = a["scores"], a["probabilities"]
            covered[ix] += 1
        fits += [reference(folder / kind / "complete.json") for kind in ("inner0", "inner1", "inner2", "outer")]
    if not (covered == 1).all() or not np.isfinite(scores).all() or not np.isfinite(probs).all():
        raise ValueError("Incomplete unique outer holdout outputs")
    path = destination / f"predictions/seed{seed}.npz"
    array_write(path, indices=np.arange(q), scores=scores, base_scores=scores, probabilities=probs)
    write(path.with_suffix(".json"), {"state": "complete", "method": method, "seed": seed,
        "protocol": reference(OUT / "protocol.json"), "predictions": reference(path), "fits": fits})


def reuse_seed17(data):
    method, seed = "pbpk_interaction", 17
    path = OUT / method / "predictions/seed17.npz"
    if receipt_valid(path.with_suffix(".json")):
        return
    source = LIGHT / "predictions/seed17.npz"
    old = read(LIGHT / "protocol.json")
    if old["folds"] != data["folds"] or old["groups"] != list(data["groups"]):
        raise ValueError("Original seed17 uses different folds")
    sources = [reference(source), reference(LIGHT / "protocol.json")]
    old_receipt_path = source.with_suffix(".json")
    old_receipt = read(old_receipt_path)
    if (old_receipt.get("state") != "complete" or old_receipt.get("score_sha256") != sha(source)
            or old_receipt.get("protocol_sha256") != sha(LIGHT / "protocol.json")):
        raise ValueError("Original seed17 score/protocol receipt changed")
    if len(old_receipt.get("fit_receipts", [])) != 20 or len(old_receipt.get("outer_receipts", {})) != 5:
        raise ValueError("Original seed17 has incomplete fit receipts")
    sources.append(reference(old_receipt_path))
    seen = set()
    for ref in [*old_receipt["fit_receipts"], *old_receipt["outer_receipts"].values()]:
        verify_reference_tree(ref, seen)
        sources.append(ref)
    for f in range(5):
        for kind in ("inner0", "inner1", "inner2", "outer"):
            folder = LIGHT / f"runs/seed17/fold{f}/{kind}"
            sources.extend([reference(folder / "heads.pt"), reference(folder / "metadata.json")])
    with np.load(source, allow_pickle=False) as a:
        scores, probabilities = a["base_scores"], a["probabilities"]
        expected = (len(data["variant_names"]), len(data["query_records"]), len(data["candidate_records"]))
        if scores.shape != expected or probabilities.shape != (*expected, 3):
            raise ValueError("Original prediction axes mismatch")
        if not np.array_equal(a["indices"], np.arange(expected[1])):
            raise ValueError("Original query order mismatch")
        if (not np.isfinite(scores).all() or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(-1), 1., atol=1e-6, rtol=0)):
            raise ValueError("Original seed17 contains invalid scores/probabilities")
        array_write(path, indices=a["indices"], scores=scores, base_scores=scores, probabilities=probabilities)
    write(path.with_suffix(".json"), {"state": "complete", "method": method, "seed": seed,
        "protocol": reference(OUT / "protocol.json"), "predictions": reference(path), "fits": [],
        "reused_sources": sources, "reuse": "Original document-head scores; predicted error subtraction disabled. Error head was trained only after document heads froze."})
    status(method, seed, "complete", reused_fits=20)


def run_seed(data, method, seed, device):
    if (method, seed) == ("pbpk_interaction", 17):
        reuse_seed17(data)
        return
    destination = OUT / method
    if receipt_valid(destination / f"predictions/seed{seed}.json"):
        return
    module = trainer(method)
    supervised = set(data["core_indices"]) | set(data["execution_indices"])
    def fit(tr, va, job, f, kind, fixed=None):
        bind = {"protocol": reference(OUT / "protocol.json"), "method": method, "seed": seed,
                "train_indices": tr, "validation_indices": va, "fixed_epochs": fixed}
        job.mkdir(parents=True, exist_ok=True)
        if (job / "binding.json").exists() and read(job / "binding.json") != bind:
            raise ValueError("Fit binding changed")
        write(job / "binding.json", bind)
        def callback(event):
            if event.get("stage") == "epoch_complete":
                status(method, seed, "training", fold=f, fit=kind, epoch=event["epoch"],
                       best_epoch=event["best_epoch"], elapsed_fit_seconds=event["elapsed_seconds"])
        status(method, seed, "training", fold=f, fit=kind, epoch=0)
        return module.train_fold(data, tr, va, job, seed=seed, device=device, epochs=fixed or 40,
            patience=5, learning_rate=.001, fixed_epochs=fixed is not None, error_train_indices=[],
            pk_epochs=1, callback=callback)
    for f, fold in enumerate(data["folds"]):
        folder = destination / f"runs/seed{seed}/fold{f}"
        indices, probabilities, epochs = [], [], []
        for j, split in enumerate(fold["inner"]):
            tr, va = sorted(supervised & set(split["train"])), sorted(supervised & set(split["validation"]))
            job = folder / f"inner{j}"
            meta = fit(tr, va, job, f, f"inner{j}")
            pred = module.predict_fold(data, va, job, device=device)
            array_write(job / "validation.npz", indices=np.asarray(va), scores=pred["scores"], probabilities=pred["probabilities"])
            write(job / "complete.json", {"state": "complete", "weights": reference(job / "heads.pt"),
                "metadata": reference(job / "metadata.json"), "binding": reference(job / "binding.json"),
                "predictions": reference(job / "validation.npz")})
            indices.extend(va); probabilities.append(pred["probabilities"]); epochs.append(meta["best_epoch"])
        if len(indices) != len(set(indices)) or set(data["groups"][i] for i in indices) & set(data["groups"][i] for i in fold["outer_test"]):
            raise ValueError("Inner OOF/calibration leaks outer group")
        cal = fit_oof_temperature(np.concatenate(probabilities), np.asarray(data["applicability"])[indices], [data["groups"][i] for i in indices])
        n = max(1, int(np.median(epochs)))
        write(folder / "selection.json", {"refit_epochs": n, "inner_best_epochs": epochs, "calibration": cal,
            "selection_indices": indices, "outer_indices_not_used": fold["outer_test"], "error_weight": 0.})
        tr, te, job = sorted(supervised & set(fold["outer_train"])), list(fold["outer_test"]), folder / "outer"
        fit(tr, [], job, f, "outer", n)
        scores, probs = [], []
        for variant in data["variant_names"]:
            status(method, seed, "predicting", fold=f, variant=variant)
            pred = module.predict_fold(data, te, job, device=device, variant=variant)
            scores.append(pred["scores"]); probs.append(calibrate(pred["probabilities"], cal["temperature"]))
        array_write(job / "predictions.npz", indices=np.asarray(te), scores=np.asarray(scores), probabilities=np.asarray(probs))
        write(job / "integrity.json", integrity(module, data, te, job, device))
        write(job / "complete.json", {"state": "complete", "weights": reference(job / "heads.pt"),
            "metadata": reference(job / "metadata.json"), "binding": reference(job / "binding.json"),
            "predictions": reference(job / "predictions.npz"), "selection": reference(folder / "selection.json"),
            "integrity": reference(job / "integrity.json")})
    publish_predictions(data, method, seed)
    status(method, seed, "complete", completed_fits=20)


def progress():
    completed = sum(len(list((OUT / m / f"runs/seed{s}").glob("fold*/*/complete.json"))) for m, s in JOBS)
    state = read(OUT / "status.json") if (OUT / "status.json").exists() else {}
    bar = "█" * (completed * 30 // 100) + "░" * (30 - completed * 30 // 100)
    print(f"[{bar}] {completed:3d}/100 new fits | 20 original-model fits reused | {state.get('phase','Not started')}")
    for path in sorted((OUT / "state").glob("*.json")):
        row = read(path)
        print(f"  {row['method']:18s} seed{row['seed']} {row['phase']:10s} fold={row.get('fold','-')} fit={row.get('fit','-')} epoch={row.get('epoch','-')}")
    if state.get("error"):
        print('Error: ', state["error"])


def supervise(device, workers):
    with (OUT / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = prepare(); freeze(data); reuse_seed17(data)
        pending = [(m, s) for m, s in JOBS if not receipt_valid(OUT / m / f"predictions/seed{s}.json")]
        active = []
        began = time.monotonic()
        write(OUT / "status.json", {"phase": "running", "pid": os.getpid(), "workers": workers,
             "started_at": datetime.now(timezone.utc).isoformat()})
        del data
        try:
            while pending or active:
                if (OUT / "stop.requested").exists():
                    raise InterruptedError("Capacity experiment stop requested")
                while pending and len(active) < workers:
                    method, seed = pending.pop(0)
                    with (OUT / f"{method}_seed{seed}.log").open("ab") as log:
                        process = subprocess.Popen([str(PYTHON), str(SCRIPT), "worker", "--method", method,
                            "--seed", str(seed), "--device", device], cwd=ROOT, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT)
                    active.append((method, seed, process))
                for method, seed, process in active[:]:
                    result = process.poll()
                    if result is not None:
                        active.remove((method, seed, process))
                        if result:
                            raise RuntimeError(f"{method} seed{seed} failed (exit {result}); inspect its project log")
                time.sleep(2)
            report = ROOT / "scripts/reviewer_capacity_report.py"
            write(OUT / "status.json", {"phase": "reporting", "pid": os.getpid()})
            subprocess.run([str(PYTHON), str(report)], cwd=ROOT, check=True)
            write(OUT / "status.json", {"phase": "complete", "pid": os.getpid(),
                "elapsed_seconds": time.monotonic() - began, "completed_at": datetime.now(timezone.utc).isoformat()})
        except BaseException as exc:
            for _, _, process in active:
                process.terminate()
            for _, _, process in active:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            write(OUT / "status.json", {"phase": "paused" if isinstance(exc, InterruptedError) else "failed",
                "pid": os.getpid(), "error": str(exc), "updated_at": datetime.now(timezone.utc).isoformat()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "start", "run", "worker", "status", "watch"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action in {"status", "watch"}:
        while True:
            if args.action == "watch" and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            progress()
            if args.action == "status" or (OUT / "status.json").exists() and read(OUT / "status.json").get("phase") in {"complete", "failed", "paused"}:
                return
            time.sleep(5)
    if args.action == "start":
        freeze(prepare())
        if (OUT / "stop.requested").exists():
            raise InterruptedError("Remove this experiment's stop.requested only when explicitly resuming")
        with (OUT / "supervisor.log").open("ab") as log:
            process = subprocess.Popen([str(PYTHON), str(SCRIPT), "run", "--device", args.device,
                "--workers", str(args.workers)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        (OUT / "supervisor.pid").write_text(str(process.pid) + "\n")
        print(f"Started project capacity control PID {process.pid}; use {SCRIPT.name} watch")
        return
    if args.action == "run":
        supervise(args.device, args.workers); return
    data = prepare(); freeze(data)
    if args.action == "freeze":
        print(OUT / "protocol.json"); return
    if args.method is None or args.seed is None:
        parser.error("worker requires --method and --seed")
    lock_path = OUT / "state" / f"{args.method}_seed{args.seed}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        if args.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info(args.device)
            if free < 2 * 1024**3:
                raise RuntimeError("Less than 2GiB GPU memory available; checkpoint kept")
            torch.cuda.set_per_process_memory_fraction(min(2 * 1024**3, total) / total, args.device)
        try:
            run_seed(data, args.method, args.seed, args.device)
        except BaseException as exc:
            status(args.method, args.seed, "failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
