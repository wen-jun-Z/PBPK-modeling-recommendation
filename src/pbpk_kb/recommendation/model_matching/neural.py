from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from pbpk_kb.recommendation.model_matching.graphs import TypedGraph


NEURAL_MATCHER_VERSION = "pbpk-neural-model-matcher-1.4.0"
AUXILIARY_NODE_FEATURE_DIMENSIONS = 12

ADAPTATION_OPERATION_TYPES = (
    "dose",
    "frequency",
    "start_time",
    "end_time",
    "infusion_duration",
)

ArchitectureName = Literal[
    "no_gnn",
    "rgcn_dual_encoder",
    "graph_transformer",
    "greed",
    "gedgnn",
    "mata_star",
    "gmn",
    "cgsim",
    "cosimgnn",
    "h2mn",
    "segmn",
    "gmn_zoomer",
    "hgt_dual_encoder",
    "cg_hhgm",
]

METHOD_FIDELITY: dict[str, str] = {
    "no_gnn": "project_baseline_node_pooling_without_message_passing",
    "rgcn_dual_encoder": "project_baseline_relation_convolution_dual_encoder",
    "graph_transformer": "project_baseline_sparse_edge_aware_graph_transformer",
    "greed": "paper_inspired_metric_preserving_siamese",
    "gedgnn": "paper_inspired_assignment_matching",
    "mata_star": "paper_inspired_learned_node_matching_without_exact_astar",
    "gmn": "faithful_core_iterative_cross_graph_attention_reimplementation",
    "cgsim": "paper_inspired_linear_node_graph_contrast",
    "cosimgnn": "paper_inspired_hierarchical_coarsen_then_match",
    "h2mn": "paper_inspired_pbpk_hierarchy_matching",
    "segmn": "paper_inspired_edge_enhanced_assignment_matching",
    "gmn_zoomer": "paper_inspired_multigranularity_matching",
    "hgt_dual_encoder": "project_baseline_relation_aware_dual_encoder",
    "cg_hhgm": "proposed_method",
}

HIERARCHY_IDS = {
    "root": 0,
    "identity": 1,
    "context": 2,
    "mechanism": 3,
    "structure": 4,
    "validation": 5,
    "other": 6,
}


@dataclass(slots=True)
class GraphTensor:
    x: Any
    node_type: Any
    hierarchy: Any
    required_weight: Any
    molecular_mask: Any
    edge_index: Any
    edge_type: Any
    semantic_types: tuple[str, ...] = ()

    def to(self, device: Any) -> "GraphTensor":
        return GraphTensor(
            x=self.x.to(device),
            node_type=self.node_type.to(device),
            hierarchy=self.hierarchy.to(device),
            required_weight=self.required_weight.to(device),
            molecular_mask=self.molecular_mask.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            semantic_types=self.semantic_types,
        )


def _bucket(value: str, buckets: int) -> tuple[int, float]:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()
    index = int.from_bytes(digest[:8], "little") % buckets
    sign = 1.0 if digest[8] & 1 else -1.0
    return index, sign


def _node_feature(node: Any, dimensions: int) -> list[float]:
    values = [0.0] * dimensions
    tokens = re.findall(
        r"[a-z0-9]+",
        f"{node.node_type} {node.label} {node.semantic_key}".casefold(),
    )
    for token in tokens:
        index, sign = _bucket(token, dimensions)
        values[index] += sign
    norm = sum(value * value for value in values) ** 0.5
    return [value / norm for value in values] if norm else values


def _signed_log1p(value: object) -> float:
    if value is None:
        return 0.0
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.0
    return math.copysign(math.log1p(abs(numeric)), numeric)


def _dose_feature(value: object, unit: object) -> tuple[float, float, float]:
    normalized = str(unit or "").strip().casefold().replace("μ", "µ")
    per_kg = normalized.endswith("/kg")
    base = normalized[:-3] if per_kg else normalized
    factors = {"g": 1000.0, "mg": 1.0, "ug": 0.001, "µg": 0.001}
    if value is None or base not in factors:
        return 0.0, 0.0, float(per_kg)
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        return 0.0, 0.0, float(per_kg)
    return math.log1p(numeric * factors[base]), 1.0, float(per_kg)


def _time_feature(value: object, unit: object) -> float:
    normalized = str(unit or "").strip().casefold()
    factors = {
        "h": 1.0,
        "hr": 1.0,
        "hour": 1.0,
        "min": 1.0 / 60.0,
        "minute": 1.0 / 60.0,
        "s": 1.0 / 3600.0,
        "sec": 1.0 / 3600.0,
        "d": 24.0,
        "day": 24.0,
    }
    if value is None or normalized not in factors:
        return 0.0
    return _signed_log1p(float(value) * factors[normalized])


def _auxiliary_node_features(
    node: Any, *, include_provenance_features: bool
) -> list[float]:
    attributes = node.attributes or {}
    dose, dose_present, dose_per_kg = _dose_feature(
        attributes.get("dose_value"), attributes.get("dose_unit")
    )
    provenance = [
        float(bool(node.required)),
        float(attributes.get("confidence") or 0.0),
        float(
            bool(attributes.get("evidence_ids"))
            or bool(attributes.get("provenance"))
            or bool(attributes.get("source_node_ids"))
        ),
        float(bool(attributes.get("conflict")) or bool(attributes.get("missing"))),
    ]
    if not include_provenance_features:
        provenance = [0.0] * 4
    return [
        *provenance,
        dose,
        dose_present,
        dose_per_kg,
        _time_feature(
            attributes.get("start_time_value"), attributes.get("start_time_unit")
        ),
        _time_feature(
            attributes.get("end_time_value"), attributes.get("end_time_unit")
        ),
        _time_feature(
            attributes.get("infusion_duration_value"),
            attributes.get("infusion_duration_unit"),
        ),
        _signed_log1p(attributes.get("count") or attributes.get("metric_count")),
        float(attributes.get("run_label_grade") or 0.0) / 3.0,
    ]


def tensorize_graph(
    graph: TypedGraph,
    *,
    feature_dimensions: int = 128,
    type_buckets: int = 64,
    relation_buckets: int = 128,
    node_embeddings: Mapping[str, Sequence[float]] | None = None,
    include_provenance_features: bool = True,
) -> GraphTensor:
    import torch

    node_index = {node.node_id: index for index, node in enumerate(graph.nodes)}
    features = []
    for node in graph.nodes:
        stored = (node_embeddings or {}).get(node.semantic_key)
        if stored is not None:
            if len(stored) != feature_dimensions:
                raise ValueError(
                    f"Embedding dimension mismatch for {node.semantic_key}: "
                    f"expected {feature_dimensions}, received {len(stored)}."
                )
            values = [float(value) for value in stored]
        else:
            values = _node_feature(node, feature_dimensions)
        features.append(
            values
            + _auxiliary_node_features(
                node,
                include_provenance_features=include_provenance_features,
            )
        )
    node_types = [_bucket(node.node_type, type_buckets)[0] for node in graph.nodes]
    hierarchies = [HIERARCHY_IDS.get(node.hierarchy, HIERARCHY_IDS["other"]) for node in graph.nodes]
    required_weights = [
        2.0 if bool(node.required) else 1.0 for node in graph.nodes
    ]
    molecular_mask = [
        node.node_type
        in {
            "MolecularStructure",
            "MolecularFingerprint",
            "MolecularDescriptor",
        }
        for node in graph.nodes
    ]
    sources: list[int] = []
    targets: list[int] = []
    relations: list[int] = []
    for edge in graph.edges:
        if edge.source_id not in node_index or edge.target_id not in node_index:
            continue
        relation = _bucket(edge.predicate, relation_buckets)[0]
        source = node_index[edge.source_id]
        target = node_index[edge.target_id]
        sources.extend((source, target))
        targets.extend((target, source))
        relations.extend((relation, relation + relation_buckets))
    if not sources:
        sources = list(range(len(graph.nodes)))
        targets = list(range(len(graph.nodes)))
        relations = [0] * len(graph.nodes)
    return GraphTensor(
        x=torch.tensor(features, dtype=torch.float32).reshape(
            len(graph.nodes), feature_dimensions + AUXILIARY_NODE_FEATURE_DIMENSIONS
        ),
        node_type=torch.tensor(node_types, dtype=torch.long),
        hierarchy=torch.tensor(hierarchies, dtype=torch.long),
        required_weight=torch.tensor(required_weights, dtype=torch.float32),
        molecular_mask=torch.tensor(molecular_mask, dtype=torch.bool),
        edge_index=torch.tensor([sources, targets], dtype=torch.long),
        edge_type=torch.tensor(relations, dtype=torch.long),
        semantic_types=tuple(node.node_type for node in graph.nodes),
    )


def build_matcher(
    architecture: ArchitectureName,
    *,
    feature_dimensions: int = 128,
    hidden_dimensions: int = 128,
    type_buckets: int = 64,
    relation_buckets: int = 256,
    dropout: float = 0.1,
    use_relation_typing: bool = True,
    use_cross_graph_attention: bool = True,
    use_pbpk_hierarchy: bool = True,
) -> Any:
    import torch

    nn = torch.nn
    functional = torch.nn.functional
    projection_checkpointing = True

    def bounded_indexed_projection(
        x: Any, indices: Any, weights: Any, *, relation: bool = False
    ) -> Any:
        """Gather at most 16 MiB of matrices without synchronizing on type IDs.

        Recompute gathered matrices during backward so chunking also bounds
        saved expanded weights across a large graph's autograd computation.
        """
        if not x.shape[0]:
            return torch.empty_like(x)
        matrix_bytes = math.prod(weights.shape[1:]) * weights.element_size()
        chunk_size = max(1, (16 * 2**20) // matrix_bytes)

        def project(values: Any, selected: Any, matrices: Any) -> Any:
            gathered = matrices.index_select(0, selected)
            if relation:
                return torch.einsum("ehd,ehdf->ehf", values, gathered)
            return torch.bmm(values.unsqueeze(1), gathered).squeeze(1)

        chunks = []
        for start in range(0, x.shape[0], chunk_size):
            values, selected = x[start:start + chunk_size], indices[start:start + chunk_size]
            if projection_checkpointing and torch.is_grad_enabled() and (x.requires_grad or weights.requires_grad):
                from torch.utils.checkpoint import checkpoint

                result = checkpoint(project, values, selected, weights,
                                    use_reentrant=False, preserve_rng_state=False)
            else:
                result = project(values, selected, weights)
            chunks.append(result)
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

    class HGTLayer(nn.Module):
        """Relation- and node-type-specific multi-head heterogeneous attention."""

        def __init__(self, heads: int = 8) -> None:
            super().__init__()
            if hidden_dimensions % heads:
                raise ValueError("hidden_dimensions must be divisible by HGT heads.")
            self.heads = heads
            self.head_dimensions = hidden_dimensions // heads
            self.query = nn.Parameter(
                torch.empty(type_buckets, hidden_dimensions, hidden_dimensions)
            )
            self.key = nn.Parameter(
                torch.empty(type_buckets, hidden_dimensions, hidden_dimensions)
            )
            self.value = nn.Parameter(
                torch.empty(type_buckets, hidden_dimensions, hidden_dimensions)
            )
            self.output = nn.Parameter(
                torch.empty(type_buckets, hidden_dimensions, hidden_dimensions)
            )
            self.relation_attention = nn.Parameter(
                torch.empty(
                    relation_buckets,
                    heads,
                    self.head_dimensions,
                    self.head_dimensions,
                )
            )
            self.relation_message = nn.Parameter(
                torch.empty(
                    relation_buckets,
                    heads,
                    self.head_dimensions,
                    self.head_dimensions,
                )
            )
            self.relation_prior = nn.Parameter(
                torch.ones(relation_buckets, heads)
            )
            self.skip = nn.Parameter(torch.ones(type_buckets))
            self.norm = nn.LayerNorm(hidden_dimensions)
            self.dropout = nn.Dropout(dropout)
            for parameter in (
                self.query,
                self.key,
                self.value,
                self.output,
                self.relation_attention,
                self.relation_message,
            ):
                nn.init.xavier_uniform_(parameter)

        @staticmethod
        def _typed_projection(x: Any, types: Any, weights: Any) -> Any:
            """Apply type-specific matrices with bounded vectorized gathers."""
            return bounded_indexed_projection(x, types, weights)

        @staticmethod
        def _relation_projection(x: Any, relations: Any, weights: Any) -> Any:
            """Apply relation matrices with bounded vectorized gathers."""
            return bounded_indexed_projection(x, relations, weights, relation=True)

        def forward(self, x: Any, graph: GraphTensor) -> Any:
            sources, targets = graph.edge_index
            relations = graph.edge_type.remainder(relation_buckets)
            query_nodes = self._typed_projection(x, graph.node_type, self.query)
            key_nodes = self._typed_projection(x, graph.node_type, self.key)
            value_nodes = self._typed_projection(x, graph.node_type, self.value)
            query = query_nodes[targets].view(
                -1, self.heads, self.head_dimensions
            )
            key = key_nodes[sources].view(-1, self.heads, self.head_dimensions)
            value = value_nodes[sources].view(
                -1, self.heads, self.head_dimensions
            )
            key = self._relation_projection(
                key, relations, self.relation_attention
            )
            value = self._relation_projection(
                value, relations, self.relation_message
            )
            logits = (
                (query * key).sum(dim=-1)
                * self.relation_prior[relations]
                / max(1.0, self.head_dimensions**0.5)
            )
            expanded_targets = targets.unsqueeze(-1).expand(-1, self.heads)
            target_max = torch.full(
                (x.shape[0], self.heads),
                -torch.inf,
                dtype=logits.dtype,
                device=logits.device,
            )
            target_max.scatter_reduce_(
                0,
                expanded_targets,
                logits,
                reduce="amax",
                include_self=True,
            )
            exponentiated = torch.exp(logits - target_max[targets])
            denominator = torch.zeros(
                (x.shape[0], self.heads),
                dtype=logits.dtype,
                device=logits.device,
            )
            denominator.index_add_(0, targets, exponentiated)
            attention = exponentiated / denominator[targets].clamp_min(1e-12)
            messages = (attention.unsqueeze(-1) * value).reshape(
                -1, hidden_dimensions
            )
            aggregate = torch.zeros_like(x)
            aggregate.index_add_(0, targets, messages)
            projected = self._typed_projection(
                aggregate, graph.node_type, self.output
            )
            alpha = torch.sigmoid(self.skip[graph.node_type]).unsqueeze(-1)
            return self.norm(
                alpha * self.dropout(projected) + (1.0 - alpha) * x
            )

    class RGCNLayer(nn.Module):
        """Relation-specific graph convolution without attention."""

        def __init__(self) -> None:
            super().__init__()
            self.relation = nn.Parameter(
                torch.empty(relation_buckets, hidden_dimensions, hidden_dimensions)
            )
            self.self_projection = nn.Linear(hidden_dimensions, hidden_dimensions)
            self.norm = nn.LayerNorm(hidden_dimensions)
            self.dropout = nn.Dropout(dropout)
            nn.init.xavier_uniform_(self.relation)

        def forward(self, x: Any, graph: GraphTensor) -> Any:
            sources, targets = graph.edge_index
            relations = graph.edge_type.remainder(relation_buckets)
            messages = torch.empty(
                (sources.shape[0], hidden_dimensions),
                dtype=x.dtype,
                device=x.device,
            )
            for relation_id in torch.unique(relations):
                selected = relations == relation_id
                messages[selected] = x[sources[selected]] @ self.relation[relation_id]
            aggregate = torch.zeros_like(x)
            aggregate.index_add_(0, targets, messages)
            degree = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            degree.index_add_(0, targets, torch.ones_like(targets, dtype=x.dtype))
            update = self.self_projection(x) + aggregate / degree.clamp_min(1.0).unsqueeze(-1)
            return self.norm(x + self.dropout(functional.gelu(update)))

    class SparseGraphTransformerLayer(nn.Module):
        """Shared-QKV edge-aware attention over observed graph edges."""

        def __init__(self, heads: int = 8) -> None:
            super().__init__()
            if hidden_dimensions % heads:
                raise ValueError("hidden_dimensions must be divisible by attention heads.")
            self.heads = heads
            self.head_dimensions = hidden_dimensions // heads
            self.query = nn.Linear(hidden_dimensions, hidden_dimensions)
            self.key = nn.Linear(hidden_dimensions, hidden_dimensions)
            self.value = nn.Linear(hidden_dimensions, hidden_dimensions)
            self.output = nn.Linear(hidden_dimensions, hidden_dimensions)
            self.relation_key = nn.Embedding(relation_buckets, hidden_dimensions)
            self.relation_value = nn.Embedding(relation_buckets, hidden_dimensions)
            self.relation_prior = nn.Embedding(relation_buckets, heads)
            self.norm = nn.LayerNorm(hidden_dimensions)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Any, graph: GraphTensor) -> Any:
            sources, targets = graph.edge_index
            relations = graph.edge_type.remainder(relation_buckets)
            query = self.query(x)[targets].view(
                -1, self.heads, self.head_dimensions
            )
            key = (
                self.key(x)[sources] + self.relation_key(relations)
            ).view(-1, self.heads, self.head_dimensions)
            value = (
                self.value(x)[sources] + self.relation_value(relations)
            ).view(-1, self.heads, self.head_dimensions)
            logits = (
                (query * key).sum(dim=-1)
                * torch.sigmoid(self.relation_prior(relations))
                / max(1.0, self.head_dimensions**0.5)
            )
            expanded_targets = targets.unsqueeze(-1).expand(-1, self.heads)
            target_max = torch.full(
                (x.shape[0], self.heads),
                -torch.inf,
                dtype=x.dtype,
                device=x.device,
            )
            target_max.scatter_reduce_(
                0, expanded_targets, logits, reduce="amax", include_self=True
            )
            exponentiated = torch.exp(logits - target_max[targets])
            denominator = torch.zeros(
                (x.shape[0], self.heads), dtype=x.dtype, device=x.device
            )
            denominator.index_add_(0, targets, exponentiated)
            attention = exponentiated / denominator[targets].clamp_min(1e-12)
            messages = (attention.unsqueeze(-1) * value).reshape(
                -1, hidden_dimensions
            )
            aggregate = torch.zeros_like(x)
            aggregate.index_add_(0, targets, messages)
            return self.norm(x + self.dropout(self.output(aggregate)))

    class MessageEncoder(nn.Module):
        def __init__(
            self,
            *,
            heterogeneous: bool,
            edge_enhanced: bool = False,
            relational_convolution: bool = False,
            sparse_graph_transformer: bool = False,
            no_message_passing: bool = False,
        ) -> None:
            super().__init__()
            self.heterogeneous = heterogeneous
            self.edge_enhanced = edge_enhanced
            self.no_message_passing = no_message_passing
            self.input = nn.Linear(feature_dimensions, hidden_dimensions)
            self.type_embedding = nn.Embedding(type_buckets, hidden_dimensions)
            self.relation_embedding = nn.Embedding(relation_buckets, hidden_dimensions)
            self.self_layers = nn.ModuleList(
                nn.Linear(hidden_dimensions, hidden_dimensions) for _ in range(2)
            )
            self.message_layers = nn.ModuleList(
                nn.Linear(hidden_dimensions, hidden_dimensions) for _ in range(2)
            )
            self.cross_layers = nn.ModuleList(
                nn.Linear(hidden_dimensions, hidden_dimensions) for _ in range(2)
            )
            self.norms = nn.ModuleList(
                nn.LayerNorm(hidden_dimensions) for _ in range(2)
            )
            self.dropout = nn.Dropout(dropout)
            self.hgt_layers = nn.ModuleList(
                HGTLayer() for _ in range(2)
            ) if heterogeneous else None
            self.rgcn_layers = nn.ModuleList(
                RGCNLayer() for _ in range(2)
            ) if relational_convolution else None
            self.graph_transformer_layers = nn.ModuleList(
                SparseGraphTransformerLayer() for _ in range(2)
            ) if sparse_graph_transformer else None

        def initialize(self, graph: GraphTensor) -> Any:
            return self.input(graph.x) + self.type_embedding(graph.node_type)

        def step(
            self,
            x: Any,
            graph: GraphTensor,
            layer_index: int,
            cross_residual: Any | None = None,
        ) -> Any:
            sources, targets = graph.edge_index
            message = self.message_layers[layer_index](x[sources])
            if self.heterogeneous or self.edge_enhanced:
                relation = self.relation_embedding(
                    graph.edge_type.remainder(relation_buckets)
                )
                message = message * torch.sigmoid(relation)
            aggregate = torch.zeros_like(x)
            aggregate.index_add_(0, targets, message)
            degree = torch.zeros(x.shape[0], device=x.device)
            degree.index_add_(0, targets, torch.ones_like(targets, dtype=x.dtype))
            aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
            update = self.self_layers[layer_index](x) + aggregate
            if cross_residual is not None:
                update = update + self.cross_layers[layer_index](cross_residual)
            return self.norms[layer_index](
                x + self.dropout(functional.gelu(update))
            )

        def forward(self, graph: GraphTensor) -> Any:
            x = self.initialize(graph)
            if self.no_message_passing:
                return x
            if self.hgt_layers is not None:
                for layer in self.hgt_layers:
                    x = layer(x, graph)
                return x
            if self.rgcn_layers is not None:
                for layer in self.rgcn_layers:
                    x = layer(x, graph)
                return x
            if self.graph_transformer_layers is not None:
                for layer in self.graph_transformer_layers:
                    x = layer(x, graph)
                return x
            for layer_index in range(len(self.self_layers)):
                x = self.step(x, graph, layer_index)
            return x

    def pool(x: Any) -> Any:
        return x.mean(dim=0)

    def hierarchy_pool(x: Any, levels: Any) -> Any:
        pooled = []
        for level in range(1, 6):
            selected = x[levels == level]
            pooled.append(selected.mean(dim=0) if selected.numel() else torch.zeros_like(x[0]))
        return torch.stack(pooled)

    def bidirectional_attention(
        query: Any,
        model: Any,
        query_types: Any | None = None,
        model_types: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        scale = max(1.0, query.shape[-1] ** 0.5)
        similarity = query @ model.transpose(0, 1) / scale
        if query_types is not None and model_types is not None:
            compatible = query_types.unsqueeze(1) == model_types.unsqueeze(0)
            query_has_match = compatible.any(dim=1, keepdim=True)
            model_has_match = compatible.any(dim=0, keepdim=True)
            compatible = torch.where(
                query_has_match,
                compatible,
                torch.ones_like(compatible),
            )
            compatible = torch.where(
                model_has_match,
                compatible,
                torch.ones_like(compatible),
            )
            similarity = similarity.masked_fill(~compatible, -1e9)
        query_match = torch.softmax(similarity, dim=1) @ model
        model_match = torch.softmax(similarity.transpose(0, 1), dim=1) @ query
        return query_match, model_match, similarity


    heterogeneous = (
        architecture in {"hgt_dual_encoder", "cg_hhgm"}
        and use_relation_typing
    )
    edge_enhanced = architecture == "segmn"
    relational_convolution = architecture == "rgcn_dual_encoder"
    sparse_graph_transformer = architecture == "graph_transformer"
    no_message_passing = architecture == "no_gnn"

    class PairMatcher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.architecture = architecture
            self.encoder = MessageEncoder(
                heterogeneous=heterogeneous,
                edge_enhanced=edge_enhanced,
                relational_convolution=relational_convolution,
                sparse_graph_transformer=sparse_graph_transformer,
                no_message_passing=no_message_passing,
            )
            hierarchical = (
                architecture
                in {
                    "h2mn",
                    "cosimgnn",
                    "gmn_zoomer",
                    "cg_hhgm",
                }
                and use_pbpk_hierarchy
            )
            if hierarchical:
                pair_dimensions = hidden_dimensions * 5 * 4
            elif architecture == "cg_hhgm" and use_cross_graph_attention:
                pair_dimensions = hidden_dimensions * 6
            else:
                pair_dimensions = hidden_dimensions * 4
            if architecture in {"gmn", "gedgnn", "mata_star", "segmn"}:
                pair_dimensions += hidden_dimensions * 2
            self.pair_head = nn.Sequential(
                nn.Linear(pair_dimensions, hidden_dimensions),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dimensions, 1),
            )
            self.adaptation_head = nn.Sequential(
                nn.Linear(pair_dimensions, hidden_dimensions // 2),
                nn.GELU(),
                nn.Linear(hidden_dimensions // 2, 1),
            )
            self.adaptation_operation_head = nn.Sequential(
                nn.Linear(pair_dimensions, hidden_dimensions // 2),
                nn.GELU(),
                nn.Linear(
                    hidden_dimensions // 2,
                    len(ADAPTATION_OPERATION_TYPES),
                ),
            )
            self.decision_head = nn.Sequential(
                nn.Linear(pair_dimensions, hidden_dimensions),
                nn.GELU(),
                nn.Linear(hidden_dimensions, 4),
            )
            self.uncertainty_head = nn.Sequential(
                nn.Linear(pair_dimensions, hidden_dimensions // 2),
                nn.GELU(),
                nn.Linear(hidden_dimensions // 2, 1),
            )

        def encode_graph(self, graph: GraphTensor) -> Any:
            if architecture == "gmn":
                raise RuntimeError("GMN node states depend on the paired graph.")
            return self.encoder(graph)

        def set_projection_checkpointing(self, enabled: bool) -> None:
            """Control memory-saving projection recomputation without changing weights."""
            nonlocal projection_checkpointing
            projection_checkpointing = bool(enabled)

        def forward_candidates(
            self, query: GraphTensor, candidates: list[GraphTensor], query_nodes: Any = None
        ) -> dict[str, Any]:
            """Score a real candidate batch with disjoint graph message passing."""
            from pbpk_kb.recommendation.model_matching.neural_batching import forward_candidates

            return forward_candidates(self, query, candidates, query_nodes=query_nodes)


        def _pair_features_from_nodes(
            self,
            query: GraphTensor,
            model: GraphTensor,
            query_nodes: Any,
            model_nodes: Any,
        ) -> Any:
            query_global = pool(query_nodes)
            model_global = pool(model_nodes)
            base = [
                query_global,
                model_global,
                torch.abs(query_global - model_global),
                query_global * model_global,
            ]
            if architecture in {
                "no_gnn",
                "rgcn_dual_encoder",
                "graph_transformer",
                "greed",
                "hgt_dual_encoder",
                "cgsim",
            }:
                return torch.cat(base)
            if architecture == "cg_hhgm" and not use_pbpk_hierarchy:
                if use_cross_graph_attention:
                    query_match, model_match, _ = bidirectional_attention(
                        query_nodes,
                        model_nodes,
                        query.node_type,
                        model.node_type,
                    )
                    base.extend(
                        [
                            pool(torch.abs(query_nodes - query_match)),
                            pool(torch.abs(model_nodes - model_match)),
                        ]
                    )
                return torch.cat(base)
            query_match, model_match, assignment = bidirectional_attention(
                query_nodes, model_nodes
            )
            if architecture in {"gmn", "gedgnn", "mata_star", "segmn"}:
                base.extend(
                    [
                        pool(torch.abs(query_nodes - query_match)),
                        pool(torch.abs(model_nodes - model_match)),
                    ]
                )
                return torch.cat(base)

            level_features = []
            for level in range(1, 6):
                selected_query = query_nodes[query.hierarchy == level]
                selected_model = model_nodes[model.hierarchy == level]
                if not selected_query.numel() or not selected_model.numel():
                    level_features.append(
                        torch.zeros(
                            hidden_dimensions * 4,
                            device=query_nodes.device,
                            dtype=query_nodes.dtype,
                        )
                    )
                    continue
                if architecture == "cosimgnn":
                    selected_query = torch.tanh(selected_query)
                    selected_model = torch.tanh(selected_model)
                if use_cross_graph_attention:
                    query_level_match, model_level_match, _ = bidirectional_attention(
                        selected_query,
                        selected_model,
                        query.node_type[query.hierarchy == level],
                        model.node_type[model.hierarchy == level],
                    )
                else:
                    query_level_match = selected_query.mean(dim=0).expand_as(
                        selected_query
                    )
                    model_level_match = selected_model.mean(dim=0).expand_as(
                        selected_model
                    )
                level_features.append(
                    torch.cat(
                        [
                            pool(selected_query),
                            pool(selected_model),
                            pool(torch.abs(selected_query - query_level_match)),
                            pool(torch.abs(selected_model - model_level_match)),
                        ]
                    )
                )
            return torch.cat(level_features)

        def _pair_features(self, query: GraphTensor, model: GraphTensor) -> Any:
            if architecture == "gmn":
                query_nodes = self.encoder.initialize(query)
                model_nodes = self.encoder.initialize(model)
                for layer_index in range(2):
                    query_match, model_match, _ = bidirectional_attention(
                        query_nodes, model_nodes
                    )
                    next_query = self.encoder.step(
                        query_nodes,
                        query,
                        layer_index,
                        query_nodes - query_match,
                    )
                    next_model = self.encoder.step(
                        model_nodes,
                        model,
                        layer_index,
                        model_nodes - model_match,
                    )
                    query_nodes, model_nodes = next_query, next_model
            else:
                query_nodes = self.encode_graph(query)
                model_nodes = self.encode_graph(model)
            return self._pair_features_from_nodes(
                query,
                model,
                query_nodes,
                model_nodes,
            )

        def _heads(self, features: Any) -> dict[str, Any]:
            score_logit = self.pair_head(features).squeeze(-1)
            adaptation = torch.sigmoid(self.adaptation_head(features).squeeze(-1))
            adaptation_operations = self.adaptation_operation_head(features)
            decision_logits = self.decision_head(features)
            uncertainty = torch.sigmoid(
                self.uncertainty_head(features).squeeze(-1)
            )
            return {
                "score_logit": score_logit,
                "score": torch.sigmoid(score_logit),
                "adaptation_cost": adaptation,
                "adaptation_operation_logits": adaptation_operations,
                "decision_logits": decision_logits,
                "uncertainty": uncertainty,
                "pair_features": features,
            }

        def forward_encoded(
            self,
            query: GraphTensor,
            model: GraphTensor,
            query_nodes: Any,
            model_nodes: Any,
        ) -> dict[str, Any]:
            if architecture == "gmn":
                raise RuntimeError("GMN requires pair-conditioned message passing.")
            return self._heads(
                self._pair_features_from_nodes(
                    query,
                    model,
                    query_nodes,
                    model_nodes,
                )
            )

        def pair_features_encoded(
            self,
            query: GraphTensor,
            model: GraphTensor,
            query_nodes: Any,
            model_nodes: Any,
        ) -> Any:
            if architecture == "gmn":
                raise RuntimeError("GMN requires pair-conditioned message passing.")
            return self._pair_features_from_nodes(
                query,
                model,
                query_nodes,
                model_nodes,
            )

        def heads_from_features(self, features: Any) -> dict[str, Any]:
            return self._heads(features)

        def forward(self, query: GraphTensor, model: GraphTensor) -> dict[str, Any]:
            return self._heads(self._pair_features(query, model))

    model = PairMatcher()
    model.matcher_metadata = {
        "supports_candidate_batching": architecture == "gmn",
        "matcher_version": NEURAL_MATCHER_VERSION,
        "architecture": architecture,
        "fidelity": METHOD_FIDELITY[architecture],
        "feature_dimensions": feature_dimensions,
        "hidden_dimensions": hidden_dimensions,
        "type_buckets": type_buckets,
        "relation_buckets": relation_buckets,
        "heterogeneous_attention": heterogeneous,
        "relational_convolution": relational_convolution,
        "sparse_graph_transformer": sparse_graph_transformer,
        "no_message_passing": no_message_passing,
        "hgt_heads": 8 if heterogeneous else None,
        "typed_cross_graph_attention": (
            architecture == "cg_hhgm" and use_cross_graph_attention
        ),
        "use_relation_typing": use_relation_typing,
        "use_cross_graph_attention": use_cross_graph_attention,
        "use_pbpk_hierarchy": use_pbpk_hierarchy,
        "supports_independent_graph_encoding": architecture != "gmn",
    }
    return model
