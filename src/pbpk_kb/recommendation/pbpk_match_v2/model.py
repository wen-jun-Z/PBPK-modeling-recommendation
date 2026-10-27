"""PBPK-Match v2 trainable heads and locally pinned Qwen LoRA encoder.

Applicability order is Y, N, U. Unknown annotation (-1) is never class N.
No evaluation label, observed concentration, or execution status is an input.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
BACKBONE_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
DEFAULT_BACKBONE = ROOT / "runtime/hf_cache/frontier/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots" / BACKBONE_REVISION
CLASSES = ("Y", "N", "U")
INSTRUCTION = ("Given a physiologically based pharmacokinetic modelling request, retrieve "
               "PBPK model descriptions relevant to the requested compounds, biological "
               "mechanisms, population, administration route and prediction task.")
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on '
          'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
          '<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def last_token(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Select the final real token with either left or right padding."""
    if hidden.shape[:2] != attention_mask.shape or not attention_mask.bool().any(1).all():
        raise ValueError("Each sequence must contain a real token")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    indices = positions.expand_as(attention_mask).masked_fill(~attention_mask.bool(), -1).max(1).values
    return hidden[torch.arange(len(hidden), device=hidden.device), indices]


def yes_no_margin(representation: Tensor, answer_weights: Tensor) -> Tensor:
    """Continuous yes-minus-no score, avoiding subtractive BF16 quantization."""
    return F.linear(representation.float(), answer_weights[1].float() - answer_weights[0].float())


class MatchHeads(nn.Module):
    """Shared 128-unit fusion with rank, ordered relevance and Y/N/U heads."""
    def __init__(self, hidden_size: int, feature_size: int, width: int = 128,
                 dropout: float = .1, residual: bool = True):
        super().__init__()
        self.hidden_size, self.feature_size, self.width = hidden_size, feature_size, width
        self.residual = residual
        self.fusion = nn.Sequential(nn.Linear(hidden_size + feature_size, width),
                                    nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout))
        self.rank = nn.Linear(width, 1)
        self.applicability = nn.Linear(width, 3)
        self.ordinal_location = nn.Linear(width, 1)
        self.threshold_start = nn.Parameter(torch.tensor(-1.))
        self.threshold_steps = nn.Parameter(torch.zeros(2))
        nn.init.zeros_(self.rank.weight)
        nn.init.zeros_(self.rank.bias)

    def forward(self, representation: Tensor, features: Tensor, base_score: Tensor | None = None) -> dict:
        dtype = self.rank.weight.dtype
        # Layer-normalization keeps pretrained hidden scales from dominating explicit conditions.
        text = F.layer_norm(representation.to(dtype), (representation.shape[-1],))
        fused = self.fusion(torch.cat((text, features.to(dtype)), -1))
        score = self.rank(fused).squeeze(-1)
        if self.residual and base_score is not None:
            score = score + base_score.float()
        thresholds = torch.cat((self.threshold_start.reshape(1),
                                self.threshold_start + F.softplus(self.threshold_steps).cumsum(0)))
        return {"scores": score, "app_logits": self.applicability(fused),
                "ordinal_logits": self.ordinal_location(fused) - thresholds,
                "representation": fused}


class ErrorHead(nn.Module):
    """Positive absolute-log-error prediction; endpoint vocab is train-only."""
    def __init__(self, input_size: int, endpoint_count: int):
        super().__init__()
        self.endpoint = nn.Embedding(endpoint_count, 8)
        self.network = nn.Sequential(nn.Linear(input_size + 8, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, representation: Tensor, endpoint: Tensor) -> Tensor:
        values = torch.cat((representation.detach().float(), self.endpoint(endpoint)), -1)
        return F.softplus(self.network(values).squeeze(-1))


class QwenMatchEncoder(nn.Module):
    """Last-token Qwen states and pretrained yes/no margin, optionally with LoRA.

    Only the final state's two vocabulary rows are evaluated: allocating the
    full [batch, tokens, vocabulary] tensor would waste almost all GPU memory.
    """
    def __init__(self, path: str | Path = DEFAULT_BACKBONE, *, device: str = "cpu",
                 trainable: bool = True, adapter_path: str | Path | None = None,
                 gradient_checkpointing: bool = True, lora_layers: int | None = 8):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_relative_to(ROOT):
            raise ValueError("The backbone must be a project-owned local snapshot")
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        lm = AutoModelForCausalLM.from_pretrained(resolved, local_files_only=True, torch_dtype=dtype,
                                                 attn_implementation="sdpa")
        ids = [self.tokenizer.convert_tokens_to_ids(word) for word in ("no", "yes")]
        self.register_buffer("answer_weights", lm.get_output_embeddings().weight[ids].detach().float().clone())
        base = lm.model
        self.hidden_size = int(lm.config.hidden_size)
        del lm
        if adapter_path is not None:
            from peft import PeftModel
            self.encoder = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=trainable)
        elif trainable:
            from peft import LoraConfig, TaskType, get_peft_model
            targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
            if lora_layers:
                # We wrap the decoder directly, whose module names begin with
                # `layers` rather than `model.layers`; explicit names avoid the
                # PEFT layers_to_transform matcher requiring a leading prefix.
                targets = [f"layers.{layer}.self_attn.{name}"
                           for layer in range(max(0, base.config.num_hidden_layers - lora_layers), base.config.num_hidden_layers)
                           for name in targets]
            config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32,
                                lora_dropout=.05, target_modules=targets, bias="none")
            self.encoder = get_peft_model(base, config)
        else:
            self.encoder = base
        self.trainable_text = trainable
        if not trainable:
            self.encoder.requires_grad_(False)
        elif gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # Non-reentrant checkpointing does not require embedding gradients.
            # This lets an unadapted prefix remain outside the autograd graph.
        self.encoder.config.use_cache = False
        self.to(device)
        self.truncations = 0
        self.tokenized_pairs = 0

    def tokenize(self, queries: list[str], documents: list[str], max_length: int) -> dict:
        if len(queries) != len(documents) or max_length < 128:
            raise ValueError("Paired texts and a token budget >=128 are required")
        prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        budget = max_length - len(prefix) - len(suffix)
        rows = []
        for query, document in zip(queries, documents):
            body = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
            tokens = self.tokenizer.encode(body, add_special_tokens=False)
            self.truncations += int(len(tokens) > budget)
            self.tokenized_pairs += 1
            rows.append(prefix + tokens[:budget] + suffix)
        batch = self.tokenizer.pad({"input_ids": rows}, padding=True, return_tensors="pt")
        return {key: value.to(self.answer_weights.device) for key, value in batch.items()}

    def forward(self, tokens: dict) -> tuple[Tensor, Tensor]:
        if not self.trainable_text:
            self.encoder.eval()
        with nullcontext() if self.trainable_text else torch.no_grad():
            hidden = self.encoder(**tokens, use_cache=False, return_dict=True).last_hidden_state
            representation = last_token(hidden, tokens["attention_mask"])
            # Subtracting two BF16 vocabulary logits can create coarse 0.25/0.5
            # score steps. Project their weight difference in FP32 instead.
            margin = yes_no_margin(representation, self.answer_weights)
            return representation.float(), margin


def document_loss(output: dict, grades: Tensor, applicability: Tensor, weights: Tensor,
                  same_drug: Tensor | None = None, class_weights: Tensor | None = None) -> tuple[Tensor, dict]:
    """Unknown-masked multi-task loss; known unequal-grade pairs only.

    Family weights affect every task. Within-drug pairs receive twice the
    ordinary pair mass, while retaining cross-drug negatives.
    """
    known_g, known_a = grades >= 0, applicability >= 0
    zero = output["scores"].sum() * 0
    ordinal = app = pair = zero
    if known_g.any():
        target = (grades[known_g, None] > torch.arange(3, device=grades.device)).float()
        loss = F.binary_cross_entropy_with_logits(output["ordinal_logits"][known_g], target, reduction="none").mean(-1)
        ordinal = (loss * weights[known_g]).sum() / weights[known_g].sum().clamp_min(1e-8)
        mask = known_g[:, None] & known_g[None, :] & (grades[:, None] > grades[None, :])
        if mask.any():
            pair_weights = (weights[:, None] * weights[None, :]).sqrt()
            if same_drug is not None:
                pair_weights = pair_weights * (1 + (same_drug[:, None] & same_drug[None, :]).float())
            pair = (F.softplus(-(output["scores"][:, None] - output["scores"][None, :]))[mask]
                    * pair_weights[mask]).sum() / pair_weights[mask].sum().clamp_min(1e-8)
    if known_a.any():
        loss = F.cross_entropy(output["app_logits"][known_a], applicability[known_a], reduction="none")
        app_weights = weights[known_a]
        if class_weights is not None:
            app_weights = app_weights * class_weights[applicability[known_a]]
        app = (loss * app_weights).sum() / app_weights.sum().clamp_min(1e-8)
    return pair + ordinal + app, {"pair": float(pair.detach()), "ordinal": float(ordinal.detach()), "app": float(app.detach())}
