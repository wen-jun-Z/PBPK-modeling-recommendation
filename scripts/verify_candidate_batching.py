#!/usr/bin/env python3
"""Compare candidate batching against unchanged single-pair training on train data."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--candidate-count", type=int, default=33)
    parser.add_argument("--query-count", type=int, default=2)
    parser.add_argument("--mode", choices=("equivalence", "performance", "all"), default="equivalence")
    parser.add_argument("--variants", nargs="+", choices=("gmn",), default=["gmn"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch
    from pbpk_kb.recommendation.model_matching.feature_store import load_embedding_store
    from pbpk_kb.recommendation.model_matching.neural import build_matcher, tensorize_graph
    from pbpk_kb.recommendation.model_matching.training import _load_benchmark, _checkpointed_ranking_outputs

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    base = ROOT / "experiments/model_recommendation/runs/expanded_catalog_20260913"
    queries, query_graphs, model_graphs, _, labels = _load_benchmark(base / "benchmark")
    embeddings, manifest = load_embedding_store(base / "features")
    candidate_ids = sorted(key for key, graph in model_graphs.items() if graph.metadata.get("candidate_split") == "train")
    # Evenly spread real candidates; no evaluation-split data are used.
    count = min(args.candidate_count, len(candidate_ids))
    candidate_ids = [candidate_ids[index * len(candidate_ids) // count] for index in range(count)]
    train_queries = sorted((row for row in queries if row["split"] == "train"),
                           key=lambda row: (len(query_graphs[row["query_id"]].nodes), row["query_id"]))
    selected_queries = [train_queries[-1]]
    if args.query_count > 1:
        selected_queries.extend(train_queries[index * len(train_queries) // (args.query_count - 1)]
                                for index in range(args.query_count - 1))
    device = torch.device(args.device)
    models = {key: tensorize_graph(model_graphs[key], feature_dimensions=int(manifest["dimensions"]),
                                   node_embeddings=embeddings).to(device) for key in candidate_ids}
    tensors = {row["query_id"]: tensorize_graph(query_graphs[row["query_id"]],
        feature_dimensions=int(manifest["dimensions"]), node_embeddings=embeddings).to(device) for row in selected_queries}
    molecular_types = {"MolecularStructure", "MolecularFingerprint", "MolecularDescriptor"}
    molecular_node_count = sum(node.node_type in molecular_types
                              for graph in [*(model_graphs[key] for key in candidate_ids),
                                            *(query_graphs[row["query_id"]] for row in selected_queries)]
                              for node in graph.nodes)
    if molecular_node_count:
        raise ValueError("Verification expects the current molecular-free metadata training graphs")
    print(json.dumps(dict(phase="input_contract", training_split_only=True,
                          candidate_count=count, molecular_node_count=molecular_node_count)), flush=True)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def loss_step(model, row, batch_size):
        # Compare the actual old path with the new outer-microbatch checkpoint.
        model.set_projection_checkpointing(batch_size == 1)
        model.zero_grad(set_to_none=True)
        outputs = _checkpointed_ranking_outputs(model=model, query_tensor=tensors[row["query_id"]],
            candidate_ids=candidate_ids, model_tensors=models, candidate_batch_size=batch_size)
        logits = torch.stack([output["score_logit"] for output in outputs])
        grades = torch.tensor([labels[(row["query_id"], key)] for key in candidate_ids], dtype=logits.dtype, device=device)
        gains = torch.pow(2.0, grades) - 1.0
        loss = -((gains/gains.sum().clamp_min(1.0)) * logits.log_softmax(0)).sum()
        loss.backward()
        return logits.detach(), loss.detach()

    results = []
    for variant in args.variants:
        for phase in (("equivalence", "performance") if args.mode == "all" else (args.mode,)):
            torch.manual_seed(17)
            model = build_matcher("gmn",
                feature_dimensions=int(manifest["dimensions"])+12,
                hidden_dimensions=128, dropout=0.0 if phase == "equivalence" else 0.1).to(device)
            model.set_projection_checkpointing(False)
            model.train()
            for row in selected_queries:
                if phase == "equivalence":
                    reference_logits, reference_loss = loss_step(model, row, 1)
                    reference_logits, reference_loss = reference_logits.cpu(), reference_loss.cpu()
                    gradients = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
                                 for name, parameter in model.named_parameters()}
                    for batch_size in args.batch_sizes:
                        logits, loss = loss_step(model, row, batch_size)
                        torch.testing.assert_close(logits.cpu(), reference_logits, atol=2e-5, rtol=5e-4)
                        torch.testing.assert_close(loss.cpu(), reference_loss, atol=2e-5, rtol=5e-4)
                        max_gradient_error = 0.0
                        for name, parameter in model.named_parameters():
                            actual, expected = parameter.grad, gradients[name]
                            if actual is None or expected is None:
                                if (actual is None) != (expected is None):
                                    raise AssertionError(f"Gradient connectivity changed: {name}")
                                continue
                            actual = actual.detach().cpu()
                            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=5e-4,
                                                       msg=lambda msg: f"{variant} {name}: {msg}")
                            max_gradient_error = max(max_gradient_error, float((actual-expected).abs().max()))
                        result = dict(phase=phase, variant=variant, query_id=row["query_id"], candidate_count=count,
                            candidate_batch_size=batch_size, passed=True,
                            max_logit_error=float((logits.cpu()-reference_logits).abs().max()),
                            max_gradient_error=max_gradient_error, loss=float(loss.cpu()))
                        results.append(result)
                        print(json.dumps(result), flush=True)
                else:
                    for batch_size in args.batch_sizes:
                        try:
                            loss_step(model, row, batch_size)
                            synchronize()
                            if device.type == "cuda":
                                torch.cuda.reset_peak_memory_stats(device)
                            durations = []
                            for _ in range(args.repeats):
                                synchronize()
                                start = time.perf_counter()
                                logits, loss = loss_step(model, row, batch_size)
                                synchronize()
                                durations.append(time.perf_counter()-start)
                            result = dict(phase=phase, variant=variant, query_id=row["query_id"], candidate_count=count,
                                candidate_batch_size=batch_size, passed=True, seconds=durations,
                                pairs_per_second=count/(sum(durations)/len(durations)),
                                loss=float(loss.cpu()),
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
                        except torch.cuda.OutOfMemoryError:
                            model.zero_grad(set_to_none=True)
                            gc.collect()
                            torch.cuda.empty_cache()
                            result = dict(phase=phase, variant=variant, query_id=row["query_id"], candidate_count=count,
                                          candidate_batch_size=batch_size, passed=False, error="cuda_out_of_memory")
                        results.append(result)
                        print(json.dumps(result), flush=True)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if args.output:
        destination = args.output.resolve()
        if not destination.is_relative_to(ROOT):
            raise ValueError("Output must remain inside this project")
        with destination.open("x") as handle:
            json.dump(dict(device=str(device), training_split_only=True,
                           molecular_node_count=molecular_node_count,
                           results=results), handle, indent=2)


if __name__ == "__main__":
    main()
