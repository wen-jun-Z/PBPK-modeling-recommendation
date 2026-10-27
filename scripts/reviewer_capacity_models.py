"""Capacity-matched interaction ablation without changing frozen trainers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn
from reviewer_controls_models import ControlMatch

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("pbpk_interaction", "concat_capacity")
SEEDS = (17, 23, 41)


class CapacityConcat(ControlMatch):
    """Same 128-dimensional projection; 198 active fusion units match capacity.

    With 1024 text dimensions and 204 conditions: 224128 trainable parameters,
    versus 224136 in LightMatch. Width is fixed by capacity, never label scores.
    """

    def __init__(self, embedding_size, feature_size, width=128, dropout=.1):
        if width != 128:
            raise ValueError("The prespecified text projection is 128-dimensional")
        super().__init__(embedding_size, feature_size, width, dropout, mode="concat_mlp")
        hidden = 198
        self.fusion = nn.Sequential(nn.Linear(2 * width + feature_size, hidden),
                                    nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.rank = nn.Linear(hidden, 1)
        self.applicability = nn.Linear(hidden, 3)
        self.ordinal_location = nn.Linear(hidden, 1)


def trainer(method):
    if method not in METHODS:
        raise ValueError("Unknown capacity control")
    spec = importlib.util.spec_from_file_location("capacity_trainer_" + method,
                                                ROOT / "scripts/pbpk_match_light_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if method == "concat_capacity":
        module.LightMatch = CapacityConcat
    return module


def parameter_counts():
    return {method: sum(p.numel() for p in trainer(method).LightMatch(1024, 204).parameters())
            for method in METHODS}
