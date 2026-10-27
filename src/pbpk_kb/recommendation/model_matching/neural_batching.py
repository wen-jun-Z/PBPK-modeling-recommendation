"""Candidate batches for GMN with unchanged parameters and objectives.

Static topology is cached; learned node states are never cached across updates.
Disjoint graph encoding and cross-graph attention perform batched computation.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


def _cached(owner: Any, name: str, key: Any, factory: Any, maximum: int = 128) -> Any:
    cache = getattr(owner, name, None)
    if cache is None:
        cache = OrderedDict()
        setattr(owner, name, cache)
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    value = factory()
    cache[key] = value
    while len(cache) > maximum:
        cache.popitem(last=False)
    return value


@dataclass
class _Selection:
    source_indices: Any
    padded_indices: Any
    mask: Any
    width: int

    def nodes(self, values: Any) -> Any:
        selected = values.index_select(0, self.source_indices)
        padded = values.new_zeros((self.mask.shape[0] * self.width, values.shape[-1]))
        return padded.index_copy(0, self.padded_indices, selected).reshape(
            self.mask.shape[0], self.width, values.shape[-1]
        )


def _selection(graphs: list[Any], choices: list[tuple[int, ...]]) -> _Selection:
    import torch

    width = max((len(indices) for indices in choices), default=0)
    sources, targets, offset = [], [], 0
    for batch_index, (graph, indices) in enumerate(zip(graphs, choices, strict=True)):
        sources.extend(offset + index for index in indices)
        targets.extend(batch_index * width + index for index in range(len(indices)))
        offset += len(graph.x)
    device = graphs[0].x.device
    mask = torch.tensor([[index < len(indices) for index in range(width)] for indices in choices],
                        dtype=torch.bool, device=device).reshape(len(graphs), width)
    return _Selection(
        torch.tensor(sources, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device), mask, width,
    )


def _disjoint(graphs: list[Any]) -> Any:
    import torch
    from pbpk_kb.recommendation.model_matching.neural import GraphTensor

    offset, edges = 0, []
    for graph in graphs:
        edges.append(graph.edge_index + offset)
        offset += len(graph.x)
    return GraphTensor(
        **{name: torch.cat([getattr(graph, name) for graph in graphs], dim=0)
           for name in ("x", "node_type", "hierarchy", "required_weight", "molecular_mask", "edge_type")},
        edge_index=torch.cat(edges, dim=1),
    )


@dataclass
class _CandidateBatch:
    originals: list[Any]
    graph: Any
    full: _Selection


def _candidate_batch(owner: Any, graphs: list[Any]) -> _CandidateBatch:
    def create() -> _CandidateBatch:
        return _CandidateBatch(
            list(graphs), _disjoint(graphs),
            _selection(graphs, [tuple(range(len(graph.x))) for graph in graphs]),
        )
    return _cached(owner, "_candidate_batches", tuple(id(graph) for graph in graphs), create, 32)


def _pool(nodes: Any, weights: Any) -> Any:
    normalized = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return (nodes * normalized.unsqueeze(-1)).sum(1)


def _attention(query: Any, model: Any, qmask: Any, mmask: Any) -> tuple[Any, Any]:
    import torch

    if query.shape[1] == 0 or model.shape[1] == 0:
        return torch.zeros_like(query), torch.zeros_like(model)
    compatible = qmask.unsqueeze(-1) & mmask.unsqueeze(1)
    similarity = torch.bmm(query, model.transpose(1, 2)) / max(1.0, query.shape[-1] ** 0.5)
    similarity = similarity.masked_fill(~compatible, -1e9)
    qattention = torch.softmax(similarity, dim=-1) * compatible.any(-1, keepdim=True)
    mattention = torch.softmax(similarity.transpose(1, 2), dim=-1) * compatible.any(1).unsqueeze(-1)
    return torch.bmm(qattention, model), torch.bmm(mattention, query)


def _gmn_features(owner: Any, query: Any, batch: _CandidateBatch) -> Any:
    import torch

    count, query_size = len(batch.originals), len(query.x)
    queries = _cached(owner, "_candidate_query_disjoints", (id(query), count),
                      lambda: (query, _disjoint([query] * count)), 128)[1]
    qnodes, mnodes = owner.encoder.initialize(queries), owner.encoder.initialize(batch.graph)
    qmask = torch.ones((count, query_size), dtype=torch.bool, device=query.x.device)
    for layer_index in range(2):
        qpadded, mpadded = qnodes.reshape(count, query_size, -1), batch.full.nodes(mnodes)
        qmatch, mmatch = _attention(qpadded, mpadded, qmask, batch.full.mask)
        qresidual = (qpadded - qmatch).reshape_as(qnodes)
        mresidual = (mpadded - mmatch).reshape(-1, mnodes.shape[-1]).index_select(
            0, batch.full.padded_indices)
        qnodes = owner.encoder.step(qnodes, queries, layer_index, qresidual)
        mnodes = owner.encoder.step(mnodes, batch.graph, layer_index, mresidual)
    qpadded, mpadded = qnodes.reshape(count, query_size, -1), batch.full.nodes(mnodes)
    qmatch, mmatch = _attention(qpadded, mpadded, qmask, batch.full.mask)
    qglobal, mglobal = _pool(qpadded, qmask.to(qnodes.dtype)), _pool(mpadded, batch.full.mask.to(mnodes.dtype))
    return torch.cat((qglobal, mglobal, (qglobal-mglobal).abs(), qglobal*mglobal,
                      _pool((qpadded-qmatch).abs(), qmask.to(qnodes.dtype)),
                      _pool((mpadded-mmatch).abs(), batch.full.mask.to(mnodes.dtype))), dim=-1)


def forward_candidates(owner: Any, query: Any, candidates: list[Any], *, query_nodes: Any = None) -> dict[str, Any]:
    if not candidates:
        raise ValueError("A candidate batch must contain at least one graph.")
    if owner.architecture != "gmn":
        raise ValueError("Candidate batching supports GMN.")
    batch = _candidate_batch(owner, candidates)
    features = _gmn_features(owner, query, batch)
    return owner.heads_from_features(features)
