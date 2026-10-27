"""Jointly trained global matcher with a bounded typed-matching correction.

The global branch can see the whole candidate description even when a request
contains only a few typed fields. The typed branch adds a small learned
correction, rather than replacing that global view. Both branches consume the
same frozen input embeddings and metadata; no baseline predictions or labels
are inputs. These are input-field alignments, not evidence-support judgments.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .model import PBPKMatch, PairMLP


class PBPKMatchHybrid(nn.Module):
    """Global PairMLP + gated, bounded PBPKMatch residual; all active parts train.

    Each output uses ``global + sigmoid(gate) * scale * tanh(typed)``. There
    are separate scalar gates for ranking and Y/N/U applicability. With the
    default initial gate of 0.1 and scale of 1.0, each initial correction has
    magnitude at most 0.1; throughout training its magnitude stays at most 1.0
    (up to floating-point rounding).

    Input projection, type embeddings, and task embeddings are the *same module
    objects* in both branches. PyTorch deduplicates their optimizer parameters;
    gradients from both branches jointly update them. The global branch is not
    frozen. Shared modules have two state-dict names referring to the same
    parameter storage, following the normal PyTorch shared-module convention.

    ``residual_enabled=False`` constructs only PairMLP, providing an exact
    baseline degeneration without unused typed parameters or extra computation.
    """

    def __init__(
        self,
        embedding_dim: int,
        pair_feature_dim: int,
        num_types: int,
        num_tasks: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        residual_enabled: bool = True,
        residual_scale: float = 1.0,
        initial_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be finite and positive.")
        if not math.isfinite(initial_gate) or not 0 < initial_gate < 1:
            raise ValueError("initial_gate must be strictly between zero and one.")
        self.residual_enabled = bool(residual_enabled)
        self.residual_scale = float(residual_scale)
        kwargs = dict(
            embedding_dim=embedding_dim, pair_feature_dim=pair_feature_dim,
            num_types=num_types, num_tasks=num_tasks,
            hidden_dim=hidden_dim, dropout=dropout,
        )
        self.global_branch = PairMLP(**kwargs)
        if self.residual_enabled:
            self.typed_branch: PBPKMatch | None = PBPKMatch(**kwargs, ablation="full")
            self.typed_branch.projection = self.global_branch.projection
            self.typed_branch.type_embedding = self.global_branch.type_embedding
            self.typed_branch.task_embedding = self.global_branch.task_embedding
            initial_logit = math.log(initial_gate / (1.0 - initial_gate))
            self.rank_gate_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
            self.app_gate_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        else:
            self.typed_branch = None
            self.register_parameter("rank_gate_logit", None)
            self.register_parameter("app_gate_logit", None)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        inputs = (q_embeddings, q_types, q_mask, c_embeddings, c_types, c_mask, pair_features, task_ids)
        global_result = self.global_branch(*inputs)
        if self.typed_branch is None:
            return {
                "rank_score": global_result["rank_score"],
                "app_logits": global_result["app_logits"],
                "matching": {
                    "residual_enabled": False,
                    "rank_gate": global_result["rank_score"].new_zeros(()),
                    "app_gate": global_result["rank_score"].new_zeros(()),
                    "rank_correction": torch.zeros_like(global_result["rank_score"]),
                    "app_correction": torch.zeros_like(global_result["app_logits"]),
                },
            }
        typed_result = self.typed_branch(*inputs)
        rank_gate = self.rank_gate_logit.sigmoid()
        app_gate = self.app_gate_logit.sigmoid()
        rank_correction = rank_gate * self.residual_scale * typed_result["rank_score"].tanh()
        app_correction = app_gate * self.residual_scale * typed_result["app_logits"].tanh()
        return {
            "rank_score": global_result["rank_score"] + rank_correction,
            "app_logits": global_result["app_logits"] + app_correction,
            "matching": {
                **typed_result["matching"],
                "residual_enabled": True,
                "rank_gate": rank_gate,
                "app_gate": app_gate,
                "residual_scale": self.residual_scale,
                "rank_correction": rank_correction,
                "app_correction": app_correction,
                "global_rank_score": global_result["rank_score"],
                "global_app_logits": global_result["app_logits"],
            },
        }
