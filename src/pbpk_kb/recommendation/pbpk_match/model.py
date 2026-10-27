"""Small supervised matchers over frozen, public-input node embeddings.

This module does not load a language model, evidence labels, or execution grades.
``matching`` describes learned alignment of *input metadata*, not independently
verified source support. Applicability classes are always ordered ``Y, N, U``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


APPLICABILITY_CLASSES = ("Y", "N", "U")


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """A mean of real nodes; an empty set returns an exact zero vector."""
    return (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def _masked_softmax(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Stable masked softmax, including genuinely empty/all-masked node sets."""
    if values.shape[dim] == 0:
        return torch.zeros_like(values)
    masked = values.masked_fill(~mask, -torch.inf)
    any_valid = mask.any(dim=dim, keepdim=True)
    safe = torch.where(any_valid, masked, torch.zeros_like(masked))
    return torch.softmax(safe, dim=dim).masked_fill(~mask, 0.0)


class _FrozenInputMatcher(nn.Module):
    """Common input contract and identical projection for both model families."""

    def __init__(
        self,
        embedding_dim: int,
        pair_feature_dim: int,
        num_types: int,
        num_tasks: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(embedding_dim, num_types, num_tasks, hidden_dim) < 1 or pair_feature_dim < 0:
            raise ValueError("Embedding, type, task and hidden dimensions must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.embedding_dim = embedding_dim
        self.pair_feature_dim = pair_feature_dim
        self.num_types = num_types
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        # The external encoder is frozen; only this shared projection is trained.
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)
        nn.init.normal_(self.type_embedding.weight, std=0.02)
        nn.init.normal_(self.task_embedding.weight, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _prepare(
        self,
        q_embeddings: Tensor,
        q_types: Tensor,
        q_mask: Tensor,
        c_embeddings: Tensor,
        c_types: Tensor,
        c_mask: Tensor,
        pair_features: Tensor,
        task_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if q_embeddings.ndim != 3 or c_embeddings.ndim != 3:
            raise ValueError("Embeddings must have shape [batch, nodes, embedding_dim].")
        batch = q_embeddings.shape[0]
        if c_embeddings.shape[0] != batch:
            raise ValueError("Query and candidate batch sizes differ.")
        if q_embeddings.shape[-1] != self.embedding_dim or c_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("Embedding dimension does not match the model configuration.")
        if q_types.shape != q_embeddings.shape[:2] or q_mask.shape != q_types.shape:
            raise ValueError("Query types and mask must match its node dimensions.")
        if c_types.shape != c_embeddings.shape[:2] or c_mask.shape != c_types.shape:
            raise ValueError("Candidate types and mask must match its node dimensions.")
        if pair_features.shape != (batch, self.pair_feature_dim) or task_ids.shape != (batch,):
            raise ValueError("pair_features or task_ids have incorrect dimensions.")
        q_mask, c_mask = q_mask.bool(), c_mask.bool()
        q_types = q_types.long().masked_fill(~q_mask, 0)
        c_types = c_types.long().masked_fill(~c_mask, 0)
        if ((q_types < 0) | (q_types >= self.num_types)).any():
            raise ValueError("A real query node has an unknown type ID.")
        if ((c_types < 0) | (c_types >= self.num_types)).any():
            raise ValueError("A real candidate node has an unknown type ID.")
        task_ids = task_ids.long()
        if ((task_ids < 0) | (task_ids >= self.num_tasks)).any():
            raise ValueError("Unknown task ID; encode unseen tasks with the reserved unknown type.")
        dtype = self.projection[0].weight.dtype
        # Zero *before* projection: padding may carry NaNs or arbitrary IDs.
        q_raw = q_embeddings.detach().to(dtype=dtype).masked_fill(~q_mask.unsqueeze(-1), 0)
        c_raw = c_embeddings.detach().to(dtype=dtype).masked_fill(~c_mask.unsqueeze(-1), 0)
        q = self.projection(F.normalize(q_raw, dim=-1)) + self.type_embedding(q_types)
        c = self.projection(F.normalize(c_raw, dim=-1)) + self.type_embedding(c_types)
        q = q.masked_fill(~q_mask.unsqueeze(-1), 0)
        c = c.masked_fill(~c_mask.unsqueeze(-1), 0)
        return q, c, q_types, c_types, q_mask, c_mask, pair_features.to(dtype=dtype), self.task_embedding(task_ids)


class PBPKMatch(_FrozenInputMatcher):
    """Type-constrained cross-attention with explicit missing-information slots.

    ``full`` permits attention to same-type nodes plus a learned NULL slot.
    ``no_type`` removes the hard type mask but retains type embeddings.
    ``no_null`` removes the learned NULL when a legal node exists; with no legal
    node it uses a fixed zero fallback, without introducing cross-type matches.

    Neither missingness nor a NULL match is hard-coded as a negative outcome.
    Ranking and Y/N/U applicability are learned from separate supervised losses.
    No positional encodings are used, so node order cannot affect eval outputs.
    """

    def __init__(
        self,
        embedding_dim: int,
        pair_feature_dim: int,
        num_types: int,
        num_tasks: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        ablation: str = "full",
    ) -> None:
        if ablation not in {"full", "no_type", "no_null"}:
            raise ValueError("ablation must be full, no_type, or no_null.")
        super().__init__(embedding_dim, pair_feature_dim, num_types, num_tasks, hidden_dim, dropout)
        self.ablation = ablation
        self.query_attention = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_attention = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.null_node = nn.Parameter(torch.zeros(hidden_dim), requires_grad=ablation != "no_null")
        self.null_bias = nn.Parameter(torch.zeros(()), requires_grad=ablation != "no_null")
        self.match_encoder = nn.Sequential(
            nn.Linear(4 * hidden_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        # The gate depends on the request and task, not candidate length/identity.
        self.requirement_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim // 2 or 1), nn.Tanh(),
            nn.Linear(hidden_dim // 2 or 1, 1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim + pair_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.rank_head = nn.Linear(hidden_dim, 1)
        self.app_head = nn.Linear(hidden_dim, len(APPLICABILITY_CLASSES))

    def forward(
        self,
        q_embeddings: Tensor,
        q_types: Tensor,
        q_mask: Tensor,
        c_embeddings: Tensor,
        c_types: Tensor,
        c_mask: Tensor,
        pair_features: Tensor,
        task_ids: Tensor,
    ) -> dict[str, Any]:
        q, c, qt, ct, qm, cm, features, task = self._prepare(
            q_embeddings, q_types, q_mask, c_embeddings, c_types, c_mask, pair_features, task_ids
        )
        batch, query_nodes, hidden = q.shape
        legal = qm.unsqueeze(2) & cm.unsqueeze(1)
        same_type = qt.unsqueeze(2) == ct.unsqueeze(1)
        has_same_type = (legal & same_type).any(-1)
        if self.ablation != "no_type":
            legal = legal & same_type
        has_legal = legal.any(-1)
        null = self.null_node if self.ablation != "no_null" else torch.zeros_like(self.null_node)
        null = null.view(1, 1, hidden).expand(batch, 1, hidden)
        all_c = torch.cat((c, null), dim=1)
        logits = torch.bmm(self.query_attention(q), self.candidate_attention(all_c).transpose(1, 2)) / math.sqrt(hidden)
        # Compare average real-node evidence against NULL, reducing length bias.
        counts = legal.sum(-1, keepdim=True).clamp_min(1).to(logits.dtype)
        real_logits = logits[..., :-1] - counts.log()
        null_logits = logits[..., -1:] + (self.null_bias if self.ablation != "no_null" else 0)
        if self.ablation == "no_null":
            null_legal = qm & ~has_legal
        else:
            null_legal = qm
        allowed = torch.cat((legal, null_legal.unsqueeze(-1)), dim=-1)
        attention = _masked_softmax(torch.cat((real_logits, null_logits), dim=-1), allowed, dim=-1)
        aligned = torch.bmm(attention, all_c)
        null_probability = attention[..., -1]
        node_features = torch.cat(
            (q, aligned, torch.abs(q - aligned), q * aligned,
             has_legal.unsqueeze(-1).to(q.dtype), null_probability.unsqueeze(-1)), dim=-1
        )
        matches = self.match_encoder(node_features).masked_fill(~qm.unsqueeze(-1), 0)
        gate_input = torch.cat((q, task.unsqueeze(1).expand(-1, query_nodes, -1)), dim=-1)
        weights = _masked_softmax(self.requirement_gate(gate_input).squeeze(-1), qm, dim=-1)
        pooled_match = (matches * weights.unsqueeze(-1)).sum(1)
        pooled_query = (q * weights.unsqueeze(-1)).sum(1)
        fused = self.fusion(torch.cat((pooled_match, pooled_query, task, features), dim=-1))
        return {
            "rank_score": self.rank_head(fused).squeeze(-1),
            "app_logits": self.app_head(fused),
            "matching": {
                "attention": attention,
                "allowed_real_matches": legal,
                "null_probability": null_probability,
                "requirement_weights": weights,
                "has_same_type": has_same_type,
                "has_legal_match": has_legal,
            },
        }


class PairMLP(_FrozenInputMatcher):
    """Supervised same-input baseline without query-to-candidate alignment.

    Shares input projection, type/task embeddings and pair features with
    PBPKMatch. Each node set is independently mean-pooled, then combined by a
    conventional MLP. No candidate matching, NULL slot or requirement gate.
    """

    def __init__(
        self,
        embedding_dim: int,
        pair_feature_dim: int,
        num_types: int,
        num_tasks: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embedding_dim, pair_feature_dim, num_types, num_tasks, hidden_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(5 * hidden_dim + pair_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.rank_head = nn.Linear(hidden_dim, 1)
        self.app_head = nn.Linear(hidden_dim, len(APPLICABILITY_CLASSES))

    def forward(
        self,
        q_embeddings: Tensor,
        q_types: Tensor,
        q_mask: Tensor,
        c_embeddings: Tensor,
        c_types: Tensor,
        c_mask: Tensor,
        pair_features: Tensor,
        task_ids: Tensor,
    ) -> dict[str, Any]:
        q, c, _, _, qm, cm, features, task = self._prepare(
            q_embeddings, q_types, q_mask, c_embeddings, c_types, c_mask, pair_features, task_ids
        )
        q_pooled, c_pooled = _masked_mean(q, qm), _masked_mean(c, cm)
        combined = torch.cat((q_pooled, c_pooled, torch.abs(q_pooled - c_pooled), q_pooled * c_pooled, task, features), dim=-1)
        fused = self.fusion(combined)
        return {"rank_score": self.rank_head(fused).squeeze(-1), "app_logits": self.app_head(fused), "matching": {}}
