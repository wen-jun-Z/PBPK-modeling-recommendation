"""Information-matched small-head controls, with the frozen PBPK loss/trainer.

Only the head factory differs.  No frozen implementation file is modified.
The document-only PBPK-Match is the primary architectural comparator because
none of these controls consumes measured execution errors during training.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("structured_mlp", "text_mlp", "concat_mlp")


class ControlMatch(nn.Module):
    def __init__(self, embedding_size, feature_size, width=128, dropout=.1, *, mode):
        super().__init__()
        if mode not in METHODS:
            raise ValueError("Unknown prespecified control")
        self.mode = mode
        self.projection = None if mode == "structured_mlp" else nn.Sequential(
            nn.Linear(embedding_size, width), nn.LayerNorm(width), nn.GELU())
        size = feature_size if mode == "structured_mlp" else 2 * width
        if mode == "concat_mlp":
            size += feature_size
        self.fusion = nn.Sequential(nn.Linear(size, width), nn.LayerNorm(width),
                                    nn.GELU(), nn.Dropout(dropout))
        self.rank = nn.Linear(width, 1)
        self.applicability = nn.Linear(width, 3)
        self.ordinal_location = nn.Linear(width, 1)
        self.threshold_start = nn.Parameter(torch.tensor(-1.))
        self.threshold_steps = nn.Parameter(torch.zeros(2))

    def forward(self, query, candidates, features):
        if self.mode == "structured_mlp":
            inputs = features.float()
        else:
            q = self.projection(F.normalize(query.detach().float(), dim=-1))
            c = self.projection(F.normalize(candidates.detach().float(), dim=-1))
            if q.ndim == 1:
                q = q.expand_as(c)
            inputs = torch.cat((q, c, features.float()), -1) if self.mode == "concat_mlp" else torch.cat((q, c), -1)
        hidden = self.fusion(inputs)
        thresholds = torch.cat((self.threshold_start.reshape(1),
                                self.threshold_start + F.softplus(self.threshold_steps).cumsum(0)))
        return {"scores": self.rank(hidden).squeeze(-1), "app_logits": self.applicability(hidden),
                "ordinal_logits": self.ordinal_location(hidden) - thresholds,
                # The shared trainer expects this shape. The error branch is
                # explicitly disabled; appended features cannot affect scores.
                "representation": torch.cat((hidden, features.float()), -1)}


def trainer(mode):
    if mode not in METHODS:
        raise ValueError("Unknown control mode")
    spec = importlib.util.spec_from_file_location("reviewer_control_trainer_" + mode,
                                                ROOT / "scripts/pbpk_match_light_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LightMatch = lambda *a, **kw: ControlMatch(*a, **kw, mode=mode)
    return module


def identity_condition_rule(features, feature_names):
    """Prespecified identity/route/condition matching; no fitted coefficients.

    Drug coverage has weight 4, route 2, species 1. Other requested numeric,
    food and formulation conditions contribute their equal-weight mean (1).
    Unavailable candidate conditions contribute no exact-match evidence.
    """
    x = np.asarray(features)
    lookup = {name: i for i, name in enumerate(feature_names)}
    score = sum(weight * x[..., lookup[field + ".query_coverage"]]
                for field, weight in (("compounds", 4.), ("routes", 2.), ("species", 1.)))
    matched, requested = np.zeros(x.shape[:-1]), np.zeros(x.shape[:-1])
    for name in feature_names:
        if name.startswith("condition.") and name.endswith(".query_missing"):
            prefix = name.removesuffix("query_missing")
            exact = prefix + "exact_match"
            if exact in lookup:
                present = 1 - x[..., lookup[name]]
                requested += present
                matched += x[..., lookup[exact]] * present
    for category in ("food", "formulation"):
        prefix = "condition." + category + "."
        present = 1 - x[..., lookup[prefix + "query_missing"]]
        overlap = np.zeros(x.shape[:-1])
        for name in feature_names:
            if name.startswith(prefix + "shared."):
                overlap = np.maximum(overlap, x[..., lookup[name]])
        requested += present
        matched += overlap * present
    return (score + np.divide(matched, requested, out=np.zeros_like(matched), where=requested > 0)).astype(np.float32)


def visible_graph(record, variant="original", *, candidate=False):
    """Project only the same public typed fields into the old service graph.

    No new source/label, engineering gate, or live database is exposed here.
    """
    from pbpk_kb.recommendation.pbpk_match_v2.data import fields_for
    from pbpk_kb.recommendation.model_matching.graphs import GraphNode, TypedGraph, semantic_key, node_hierarchy
    types = {"compounds": "Compound", "species": "Species", "routes": "Route", "populations": "Population",
             "endpoints": "Endpoint", "enzymes": "Enzyme", "transporters": "Transporter", "process_types": "ADMEProcess",
             "formulations": "Formulation", "model_structure": "ModelStructure", "software": "Software", "compartments": "Compartment"}
    nodes = []
    for field, values in fields_for(record, variant).items():
        for value in sorted(set(values)):
            kind = types[field]
            key = semantic_key(kind, value)
            nodes.append(GraphNode(key, kind, key, value, node_hierarchy(kind), required=not candidate))
    identity = record["candidate_id"] if candidate else record["query_id"]
    return TypedGraph(identity, "model" if candidate else "requirement", tuple(nodes), ())
