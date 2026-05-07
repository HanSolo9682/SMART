"""Adapter late-interaction MMEB-V2 embedding evaluation.

This evaluator follows `eval_embedding_intermediate_late_interaction.py`, but uses a
trained VisDoc token adapter for the late-interaction vectors. The frozen
Qwen3-VL-Embedding backbone still provides the final-layer pooled anchor embedding; the
selected hidden layer is filtered with the same MaxSim token mask, passed through the
adapter, and then scored with the same formula:

    final_score = lambda_anchor * anchor_score + lambda_late * adapted_late_score

The original evaluators and launch scripts are intentionally left untouched.
"""

import copy
import hashlib
import os
import shutil
import time
import yaml
import torch
import torch.nn as nn
import random
import pickle
import json
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from datasets import concatenate_datasets
from datasets.distributed import split_dataset_by_node
from typing import Any, Dict, List, Optional, Union

from .arguments import ModelArguments, DataArguments, EvalArguments
from .utils.basic_utils import print_rank, print_master
from .utils.eval_utils.metrics import RankingMetrics
from .models import MMEBEmbeddingModel
from .data.datasets.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
from .data.collator import MultimodalEvalDataCollator


@dataclass
class LateInteractionArguments:
    enable_late_interaction: bool = field(
        default=False,
        metadata={"help": "Use token-level MaxSim late interaction in addition to EOT anchor scores."},
    )
    lambda_anchor: float = field(
        default=1.0,
        metadata={"help": "Weight for the original single-vector EOT anchor score."},
    )
    lambda_late: float = field(
        default=1.0,
        metadata={"help": "Weight for the token-level MaxSim late-interaction score."},
    )
    late_candidate_chunk_size: int = field(
        default=64,
        metadata={"help": "Number of candidates scored per MaxSim chunk."},
    )
    late_query_chunk_size: int = field(
        default=4,
        metadata={"help": "Number of queries scored per MaxSim chunk."},
    )
    late_interaction_layer: int = field(
        default=-1,
        metadata={
            "help": (
                "Transformer layer index to extract image/text token hidden states from for MaxSim "
                "late interaction. Supports negative indices (e.g. -1 = final layer). Indexing matches "
                "transformers' `outputs.hidden_states`: index 0 is the embedding layer and index N is "
                "the output of the N-th decoder layer. Example: 24."
            )
        },
    )


@dataclass
class AdapterArguments:
    adapter_type: str = field(
        default="residual",
        metadata={
            "help": "Adapter variant to evaluate: residual, self_attention, or mlp.",
            "choices": ["residual", "self_attention", "mlp"],
        },
    )
    adapter_checkpoint_dir: str = field(
        default=None,
        metadata={
            "help": (
                "Adapter checkpoint directory. May point directly at a directory containing "
                "adapter.pt, at best_adapter/, or at the run root containing last_1/best_adapter."
            )
        },
    )
    selected_layer: int = field(
        default=-1,
        metadata={
            "help": (
                "Backbone hidden-state layer used as adapter input. Negative indices are "
                "Python-style over transformers' hidden_states tuple; -1 is the final layer."
            )
        },
    )
    sanity_check: bool = field(
        default=False,
        metadata={"help": "Run a small adapter/MaxSim sanity check and exit."},
    )
    sanity_check_num_examples: int = field(
        default=2,
        metadata={"help": "Number of query/candidate examples used by --sanity_check."},
    )
    sanity_check_print_chars: int = field(
        default=2000,
        metadata={"help": "Maximum characters printed for formatted sanity-check inputs."},
    )


class LayerWeightedTokenProjectionModule(nn.Module):
    """Projection adapter used by the residual and MLP VisDoc checkpoints."""

    def __init__(
        self,
        hidden_size: int,
        residual_dim: int,
        num_input_layers: int = 1,
        projection_type: str = "linear",
        projection_mlp_hidden_dim: int = 4096,
        projection_mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_input_layers = num_input_layers
        self.hidden_size = hidden_size
        self.residual_dim = residual_dim
        self.projection_type = projection_type.lower().strip()
        self.projection_mlp_hidden_dim = projection_mlp_hidden_dim
        self.projection_mlp_dropout = projection_mlp_dropout
        if self.projection_type not in {"linear", "mlp"}:
            raise ValueError(f"Unsupported projection_type={projection_type!r}")
        self.num_output_layers = 1

        if num_input_layers > 1:
            self.layer_weight_vectors = nn.Parameter(torch.empty(self.num_output_layers, hidden_size))
            nn.init.normal_(self.layer_weight_vectors, mean=0.0, std=0.02)
            self.layer_weight_scale = hidden_size ** -0.5
        else:
            self.layer_weight_vectors = None
            self.layer_weight_scale = None

        if self.projection_type == "linear":
            self.proj = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, residual_dim, bias=False),
            )
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, projection_mlp_hidden_dim, bias=False),
                nn.GELU(),
                nn.Dropout(projection_mlp_dropout),
                nn.Linear(projection_mlp_hidden_dim, residual_dim, bias=False),
            )

    @staticmethod
    def _rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def _merge_multi_layer_states(self, hidden_states: Union[List[torch.Tensor], tuple]) -> torch.Tensor:
        if len(hidden_states) != self.num_input_layers:
            raise ValueError(f"Expected {self.num_input_layers} selected layers, got {len(hidden_states)}")
        if self.layer_weight_vectors is None:
            return hidden_states[-1]

        ref = hidden_states[0]
        dtype = ref.dtype
        normed_layer_vectors = self._rms_normalize(self.layer_weight_vectors.to(dtype))
        per_layer_scores = []
        for h in hidden_states:
            h_normed = self._rms_normalize(h)
            per_layer_scores.append(torch.einsum("bth,lh->btl", h_normed, normed_layer_vectors))
        layer_scores = torch.stack(per_layer_scores, dim=2)
        layer_weights = F.softmax(layer_scores * self.layer_weight_scale, dim=2)

        batch_size, seq_len, _ = ref.shape
        hidden_size = ref.shape[-1]
        merged = ref.new_zeros(batch_size, seq_len, self.num_output_layers, hidden_size)
        for idx, h in enumerate(hidden_states):
            merged += layer_weights[:, :, idx].unsqueeze(-1) * h.unsqueeze(2)
        return merged

    @staticmethod
    def _flatten_layer_weighted_states(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if hidden_states.dim() == 3:
            return hidden_states, None if attention_mask is None else attention_mask.clone()
        if hidden_states.dim() != 4:
            raise ValueError(f"Expected rank-3 or rank-4 hidden states, got {tuple(hidden_states.shape)}")

        batch_size, seq_len, num_layers, hidden_size = hidden_states.shape
        flattened = hidden_states.reshape(batch_size, seq_len * num_layers, hidden_size)
        if attention_mask is None:
            return flattened, None
        expanded_mask = (
            attention_mask.unsqueeze(-1)
            .expand(batch_size, seq_len, num_layers)
            .reshape(batch_size, seq_len * num_layers)
            .clone()
        )
        return flattened, expanded_mask

    @staticmethod
    def _compact_token_dimension(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if attention_mask is None:
            return hidden_states, None
        if hidden_states.dim() != 3:
            raise ValueError(f"Expected rank-3 hidden states before compaction, got {tuple(hidden_states.shape)}")

        keep_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        batch_size, _, hidden_size = hidden_states.shape
        max_tokens = int(keep_mask.sum(dim=-1).max().item()) if batch_size > 0 else 0
        max_tokens = max(max_tokens, 1)

        compacted = hidden_states.new_zeros(batch_size, max_tokens, hidden_size)
        compacted_mask = torch.zeros(batch_size, max_tokens, device=hidden_states.device, dtype=torch.bool)
        for row_idx in range(batch_size):
            row_states = hidden_states[row_idx][keep_mask[row_idx]]
            row_count = min(row_states.shape[0], max_tokens)
            if row_count > 0:
                compacted[row_idx, :row_count] = row_states[:row_count]
                compacted_mask[row_idx, :row_count] = True
        return compacted, compacted_mask

    def forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(hidden_states, (list, tuple)):
            if attention_mask is not None:
                compacted_layers = []
                compacted_mask = None
                for h in hidden_states:
                    compacted_h, current_mask = self._compact_token_dimension(h, attention_mask)
                    compacted_layers.append(compacted_h)
                    if compacted_mask is None:
                        compacted_mask = current_mask
                hidden_states = compacted_layers
                attention_mask = compacted_mask
            hidden_states = self._merge_multi_layer_states(hidden_states)
        elif attention_mask is not None:
            hidden_states, attention_mask = self._compact_token_dimension(hidden_states, attention_mask)

        residual_raw, residual_mask = self._flatten_layer_weighted_states(hidden_states, attention_mask)
        residual = self.proj(residual_raw)
        residual = F.normalize(residual, p=2, dim=-1)
        return residual_raw, residual, residual_mask


class LayerWeightedTokenSelfAttentionProjectionModule(LayerWeightedTokenProjectionModule):
    """Projection plus token self-attention adapter used by the self-attention checkpoint."""

    def __init__(
        self,
        hidden_size: int,
        residual_dim: int,
        num_input_layers: int = 1,
        self_attn_num_heads: int = 8,
        self_attn_num_layers: int = 1,
        self_attn_dropout: float = 0.0,
    ) -> None:
        nn.Module.__init__(self)
        self.num_input_layers = num_input_layers
        self.hidden_size = hidden_size
        self.residual_dim = residual_dim
        self.self_attn_num_heads = self_attn_num_heads
        self.self_attn_num_layers = self_attn_num_layers
        self.self_attn_dropout = self_attn_dropout
        self.num_output_layers = 1
        if residual_dim % self_attn_num_heads != 0:
            raise ValueError(
                f"residual_dim={residual_dim} must be divisible by self_attn_num_heads={self_attn_num_heads}"
            )

        if num_input_layers > 1:
            self.layer_weight_vectors = nn.Parameter(torch.empty(self.num_output_layers, hidden_size))
            nn.init.normal_(self.layer_weight_vectors, mean=0.0, std=0.02)
            self.layer_weight_scale = hidden_size ** -0.5
        else:
            self.layer_weight_vectors = None
            self.layer_weight_scale = None

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, residual_dim, bias=False),
        )
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=residual_dim,
                num_heads=self_attn_num_heads,
                dropout=self_attn_dropout,
                batch_first=True,
            )
            for _ in range(self_attn_num_layers)
        ])
        self.self_attn_dropouts = nn.ModuleList([
            nn.Dropout(self_attn_dropout) for _ in range(self_attn_num_layers)
        ])
        self.self_attn_norms = nn.ModuleList([
            nn.LayerNorm(residual_dim) for _ in range(self_attn_num_layers)
        ])

    @staticmethod
    def _effective_attention_mask(
        token_mask: Optional[torch.Tensor],
        ref: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if token_mask is None:
            return None, None
        valid_mask = token_mask.to(device=ref.device, dtype=torch.bool)
        effective_mask = valid_mask.clone()
        empty_rows = ~effective_mask.any(dim=-1)
        if empty_rows.any():
            effective_mask[empty_rows, 0] = True
        return valid_mask, effective_mask

    def _apply_self_attention(
        self,
        projected: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        valid_mask, effective_mask = self._effective_attention_mask(token_mask, projected)
        key_padding_mask = None if effective_mask is None else ~effective_mask
        x = projected
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        for attn, dropout, norm in zip(self.self_attn_layers, self.self_attn_dropouts, self.self_attn_norms):
            attn_out, _ = attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
            x = norm(x + dropout(attn_out))
            if valid_mask is not None:
                x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return x

    def forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(hidden_states, (list, tuple)):
            if attention_mask is not None:
                compacted_layers = []
                compacted_mask = None
                for h in hidden_states:
                    compacted_h, current_mask = self._compact_token_dimension(h, attention_mask)
                    compacted_layers.append(compacted_h)
                    if compacted_mask is None:
                        compacted_mask = current_mask
                hidden_states = compacted_layers
                attention_mask = compacted_mask
            hidden_states = self._merge_multi_layer_states(hidden_states)
        elif attention_mask is not None:
            hidden_states, attention_mask = self._compact_token_dimension(hidden_states, attention_mask)

        residual_raw, residual_mask = self._flatten_layer_weighted_states(hidden_states, attention_mask)
        projected = self.proj(residual_raw)
        attended = self._apply_self_attention(projected, residual_mask)
        residual = F.normalize(attended, p=2, dim=-1)
        if residual_mask is not None:
            residual = residual.masked_fill(~residual_mask.to(residual.device).unsqueeze(-1), 0.0)
        return residual_raw, residual, residual_mask


@dataclass
class AdapterRuntime:
    module: nn.Module
    adapter_type: str
    adapter_checkpoint_dir: str
    adapter_file: str
    adapter_fingerprint: str
    adapter_param_count: int
    missing_keys: List[str]
    unexpected_keys: List[str]
    checkpoint_metadata: Dict[str, Any]


def pad_dataset_to_divisible(dataset, world_size):
    num_samples = len(dataset)
    if num_samples % world_size == 0:
        return dataset, num_samples

    num_to_add = world_size - (num_samples % world_size)
    padded_size = num_samples + num_to_add

    padding_data = dataset.select([i % len(dataset) for i in range(num_to_add)])
    padded_dataset = concatenate_datasets([dataset, padding_data])
    return padded_dataset, padded_size


def _token_ids(tokenizer, token_strings):
    ids = set()
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in token_strings:
        token_id = vocab.get(token)
        if token_id is None and hasattr(tokenizer, "convert_tokens_to_ids"):
            token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != unk_id:
            ids.add(int(token_id))
    return ids


# def _late_keep_token_mask(input_ids, attention_mask, tokenizer):
#     keep_mask = attention_mask.bool()

#     # Gather all default special tokens from the tokenizer
#     special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    
#     # Add control tokens AND visual placeholders to the blacklist
#     special_ids.update(
#         _token_ids(
#             tokenizer,
#             [
#                 "<|endoftext|>",
#                 "<|im_start|>",
#                 "<|im_end|>",
#                 "<|vision_start|>",
#                 "<|vision_end|>",
#                 "<|image_pad|>",  # Now explicitly marked for masking
#                 "<|video_pad|>",  # Now explicitly marked for masking
#             ],
#         )
#     )

#     # Note: We completely removed the `special_ids.difference_update(...)` line.
#     # By doing this, the visual tokens remain in the special_ids set.

#     if special_ids:
#         special_tensor = torch.tensor(sorted(special_ids), device=input_ids.device)
#         # Checks if each token in input_ids matches any ID in the special_tensor
#         is_special = (input_ids.unsqueeze(-1) == special_tensor).any(dim=-1)
        
#         # Flips the boolean (so special/visual tokens become False) and applies it to the mask
#         keep_mask = keep_mask & ~is_special

#     return keep_mask



def _late_keep_token_mask(input_ids, attention_mask, tokenizer):
    keep_mask = attention_mask.bool()

    visual_placeholder_ids = _token_ids(
        tokenizer,
        ["<|image_pad|>", "<|video_pad|>"],
    )
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    special_ids.update(
        _token_ids(
            tokenizer,
            [
                "<|endoftext|>",
                "<|im_start|>",
                "<|im_end|>",
                "<|vision_start|>",
                "<|vision_end|>",
            ],
        )
    )
    special_ids.difference_update(visual_placeholder_ids)

    if special_ids:
        special_tensor = torch.tensor(sorted(special_ids), device=input_ids.device)
        is_special = (input_ids.unsqueeze(-1) == special_tensor).any(dim=-1)
        keep_mask = keep_mask & ~is_special

    return keep_mask


def _format_batch_inputs(model: MMEBEmbeddingModel, batch_inputs):
    encoder = model.encoder
    return [
        encoder.format_model_input(
            text=ele.get("text"),
            image=ele.get("image"),
            video=ele.get("video"),
            instruction=ele.get("instruction"),
            fps=ele.get("fps"),
            max_frames=ele.get("max_frames"),
        )
        for ele in batch_inputs
    ]


def _layer_tag(layer_index: int) -> str:
    return f"L{str(layer_index).replace('-', 'n')}"


def _select_layer_hidden(all_hidden_states, layer_index: int):
    """Pull out the hidden states for the requested layer.

    `all_hidden_states` is the tuple returned by transformers when `output_hidden_states=True`:
    index 0 = embedding output, index k = output of decoder block k. Negative indices are
    interpreted Python-style (e.g. -1 is the final layer).
    """
    num_states = len(all_hidden_states)
    idx = layer_index if layer_index >= 0 else num_states + layer_index
    if idx < 0 or idx >= num_states:
        raise ValueError(
            f"Layer index {layer_index} is out of range; model exposes {num_states} hidden states "
            f"(0..{num_states - 1})."
        )
    return all_hidden_states[idx]


def _canonical_adapter_type(adapter_type: str) -> str:
    normalized = adapter_type.lower().strip().replace("-", "_")
    aliases = {
        "linear": "residual",
        "token_projection": "residual",
        "token_self_attention": "self_attention",
        "self_attn": "self_attention",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"residual", "self_attention", "mlp"}:
        raise ValueError(
            f"Unsupported adapter_type={adapter_type!r}; expected residual, self_attention, or mlp."
        )
    return normalized


def _resolve_adapter_file(adapter_checkpoint_dir: str) -> str:
    if not adapter_checkpoint_dir:
        raise ValueError("--adapter_checkpoint_dir is required")

    root = Path(adapter_checkpoint_dir).expanduser()
    candidates = []
    if root.is_file():
        candidates.append(root)
    else:
        candidates.extend(
            [
                root / "adapter.pt",
                root / "best_adapter" / "adapter.pt",
                root / "last_1" / "best_adapter" / "adapter.pt",
                root / "last_1" / "epoch_2" / "adapter.pt",
                root / "last_1" / "epoch_1" / "adapter.pt",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    recursive = sorted(root.glob("**/adapter.pt")) if root.exists() and root.is_dir() else []
    best = [path for path in recursive if path.parent.name == "best_adapter"]
    if best:
        return str(best[0].resolve())
    if recursive:
        return str(recursive[-1].resolve())

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find adapter.pt under {adapter_checkpoint_dir}. Checked: {checked}")


def _checkpoint_fingerprint(adapter_file: str) -> str:
    path = Path(adapter_file)
    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _torch_load_adapter(adapter_file: str) -> Dict[str, Any]:
    try:
        return torch.load(adapter_file, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(adapter_file, map_location="cpu")


def _adapter_num_input_layers(checkpoint: Dict[str, Any]) -> int:
    selected_layers = checkpoint.get("selected_layers")
    if selected_layers is not None:
        return len(selected_layers)
    return int(checkpoint.get("use_last_n_layers", 1))


def _adapter_tag(adapter_runtime: AdapterRuntime, selected_layer: int) -> str:
    return (
        f"{adapter_runtime.adapter_type}_{_layer_tag(selected_layer)}_"
        f"{adapter_runtime.adapter_fingerprint}"
    )


def load_adapter_runtime(
    adapter_args: AdapterArguments,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> AdapterRuntime:
    adapter_type = _canonical_adapter_type(adapter_args.adapter_type)
    adapter_file = _resolve_adapter_file(adapter_args.adapter_checkpoint_dir)
    checkpoint = _torch_load_adapter(adapter_file)
    if "token_projection_module" not in checkpoint:
        raise KeyError(f"{adapter_file} does not contain a token_projection_module state dict")

    state_dict = checkpoint["token_projection_module"]
    adapter_param_count = sum(value.numel() for value in state_dict.values() if hasattr(value, "numel"))
    checkpoint_hidden_size = checkpoint.get("hidden_size")
    if checkpoint_hidden_size is not None and int(checkpoint_hidden_size) != int(hidden_size):
        raise ValueError(
            f"Adapter hidden_size={checkpoint_hidden_size} does not match backbone hidden_size={hidden_size}"
        )

    num_input_layers = _adapter_num_input_layers(checkpoint)
    if num_input_layers != 1:
        raise ValueError(
            "This evaluator applies one --selected_layer at a time, but the checkpoint expects "
            f"{num_input_layers} input layers. Use a last_1/single-layer adapter checkpoint."
        )

    residual_dim = int(checkpoint.get("residual_dim", hidden_size))
    checkpoint_adapter_type = str(checkpoint.get("adapter_type", "token_projection")).lower().strip()
    if adapter_type == "self_attention":
        if checkpoint_adapter_type not in {"token_self_attention", "self_attention"}:
            raise ValueError(
                f"Requested adapter_type='self_attention', but checkpoint adapter_type="
                f"{checkpoint_adapter_type!r}."
            )
        module = LayerWeightedTokenSelfAttentionProjectionModule(
            hidden_size=hidden_size,
            residual_dim=residual_dim,
            num_input_layers=num_input_layers,
            self_attn_num_heads=int(checkpoint.get("self_attn_num_heads", 8)),
            self_attn_num_layers=int(checkpoint.get("self_attn_num_layers", 1)),
            self_attn_dropout=float(checkpoint.get("self_attn_dropout", 0.0)),
        )
    else:
        if checkpoint_adapter_type not in {"token_projection", "residual", "linear", "mlp"}:
            raise ValueError(
                f"Requested adapter_type={adapter_type!r}, but checkpoint adapter_type="
                f"{checkpoint_adapter_type!r}."
            )
        checkpoint_projection_type = str(checkpoint.get("projection_type", "linear")).lower().strip()
        projection_type = "mlp" if adapter_type == "mlp" else "linear"
        if checkpoint_projection_type != projection_type:
            raise ValueError(
                f"Requested adapter_type={adapter_type!r}, but checkpoint projection_type="
                f"{checkpoint_projection_type!r}."
            )
        module = LayerWeightedTokenProjectionModule(
            hidden_size=hidden_size,
            residual_dim=residual_dim,
            num_input_layers=num_input_layers,
            projection_type=projection_type,
            projection_mlp_hidden_dim=int(checkpoint.get("projection_mlp_hidden_dim", 4096)),
            projection_mlp_dropout=float(checkpoint.get("projection_mlp_dropout", 0.0)),
        )

    load_result = module.load_state_dict(state_dict, strict=False)
    module = module.to(device=device, dtype=dtype)
    module.eval()
    for param in module.parameters():
        param.requires_grad = False

    metadata = {key: value for key, value in checkpoint.items() if key != "token_projection_module"}
    return AdapterRuntime(
        module=module,
        adapter_type=adapter_type,
        adapter_checkpoint_dir=adapter_args.adapter_checkpoint_dir,
        adapter_file=adapter_file,
        adapter_fingerprint=_checkpoint_fingerprint(adapter_file),
        adapter_param_count=int(adapter_param_count),
        missing_keys=list(load_result.missing_keys),
        unexpected_keys=list(load_result.unexpected_keys),
        checkpoint_metadata=metadata,
    )


def _safe_json(obj: Any, max_chars: Optional[int] = None) -> str:
    def _default(value):
        return f"<{value.__class__.__name__}>"

    text = json.dumps(obj, ensure_ascii=False, indent=2, default=_default)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>"
    return text


def _maybe_log_token_filter_counts(
    side: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    keep_mask: torch.Tensor,
    tokenizer,
    logged_sides: set,
) -> None:
    if side in logged_sides:
        return
    logged_sides.add(side)
    visual_placeholder_ids = _token_ids(tokenizer, ["<|image_pad|>", "<|video_pad|>"])
    before = attention_mask.bool().sum(dim=1).detach().cpu().tolist()
    after = keep_mask.sum(dim=1).detach().cpu().tolist()
    print_master(f"[token-filter:{side}] token count before filtering: {before}")
    print_master(f"[token-filter:{side}] token count after filtering : {after}")
    print_master(
        f"[token-filter:{side}] visual placeholder tokens kept: True "
        f"(<|image_pad|>/<|video_pad|> ids={sorted(visual_placeholder_ids)})"
    )


@torch.no_grad()
def encode_anchor_and_tokens(
    model: MMEBEmbeddingModel,
    batch_inputs,
    include_tokens: bool,
    layer_index: int,
    adapter_runtime: Optional[AdapterRuntime],
    encode_side: str,
    token_filter_log_sides: set,
    debug_state: Optional[Dict[str, Any]] = None,
):
    if not include_tokens:
        return model.encode_input(batch_inputs), None
    if adapter_runtime is None:
        raise ValueError("adapter_runtime is required when include_tokens=True")

    encoder = model.encoder
    conversations = _format_batch_inputs(model, batch_inputs)
    processed_inputs = encoder._preprocess_inputs(conversations)
    processed_inputs = {k: v.to(model.device) for k, v in processed_inputs.items()}

    # Bypass the embedding wrapper so we can request all hidden states in a single forward.
    # encoder.model is Qwen3VLForEmbedding; encoder.model.model is the underlying Qwen3VLModel.
    inner_model = encoder.model.model
    inner_outputs = inner_model(**processed_inputs, output_hidden_states=True)

    last_hidden_state = inner_outputs.last_hidden_state
    all_hidden_states = inner_outputs.hidden_states
    attention_mask = processed_inputs.get("attention_mask")

    # Anchor stays on the final layer pooled embedding to match the original scorer exactly.
    anchors = encoder._pooling_last(last_hidden_state, attention_mask)
    if model.normalize:
        anchors = F.normalize(anchors, p=2, dim=-1)

    layer_hidden = _select_layer_hidden(all_hidden_states, layer_index)

    input_ids = processed_inputs["input_ids"]
    keep_mask = _late_keep_token_mask(input_ids, attention_mask, encoder.processor.tokenizer)
    _maybe_log_token_filter_counts(
        side=encode_side,
        input_ids=input_ids,
        attention_mask=attention_mask,
        keep_mask=keep_mask,
        tokenizer=encoder.processor.tokenizer,
        logged_sides=token_filter_log_sides,
    )

    adapter_param = next(adapter_runtime.module.parameters())
    layer_hidden_for_adapter = layer_hidden.to(dtype=adapter_param.dtype)
    residual_raw, adapted_tokens, adapted_mask = adapter_runtime.module(layer_hidden_for_adapter, keep_mask)
    if adapted_mask is None:
        adapted_mask = torch.ones(
            adapted_tokens.shape[:2],
            device=adapted_tokens.device,
            dtype=torch.bool,
        )

    if debug_state is not None and encode_side not in debug_state:
        debug_state[encode_side] = {
            "formatted_input": conversations[0] if conversations else None,
            "hidden_state_shape_before_adapter": list(layer_hidden.shape),
            "token_representation_shape_after_adapter": list(adapted_tokens.shape),
            "residual_raw_shape": list(residual_raw.shape),
            "dtype": str(adapted_tokens.dtype),
            "outputs_finite": bool(torch.isfinite(adapted_tokens).all().item()),
            "token_count_before_filter": attention_mask.bool().sum(dim=1).detach().cpu().tolist(),
            "token_count_after_filter": keep_mask.sum(dim=1).detach().cpu().tolist(),
        }

    token_states = []
    for row_idx in range(adapted_tokens.shape[0]):
        row_tokens = adapted_tokens[row_idx][adapted_mask[row_idx]]
        if row_tokens.numel() == 0:
            # Empty mask is a degenerate edge case; fall back to a zero vector with the
            # correct feature dim so MaxSim shapes still line up.
            row_tokens = torch.zeros(
                1, adapted_tokens.shape[-1],
                device=adapted_tokens.device, dtype=adapted_tokens.dtype,
            )
        token_states.append(row_tokens.float().cpu().numpy().astype(np.float32))

    return anchors, token_states


@torch.no_grad()
def encode_representations(
    model: MMEBEmbeddingModel,
    loader: DataLoader,
    encode_side: str,
    full_dataset_len: int,
    include_tokens: bool,
    layer_index: int,
    adapter_runtime: Optional[AdapterRuntime],
    token_filter_log_sides: set,
    description: str = "Encoding",
    object_group=None,
    debug_state: Optional[Dict[str, Any]] = None,
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    local_anchors = []
    local_infos = []
    local_token_states = []

    model.eval()
    progress_bar = tqdm(
        loader,
        desc=f"{description} (rank {rank})",
        disable=local_rank > 0,
        ncols=120,
    )

    for batch_inputs, dataset_info in progress_bar:
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            anchors, token_states = encode_anchor_and_tokens(
                model=model,
                batch_inputs=batch_inputs,
                include_tokens=include_tokens,
                layer_index=layer_index,
                adapter_runtime=adapter_runtime,
                encode_side=encode_side,
                token_filter_log_sides=token_filter_log_sides,
                debug_state=debug_state,
            )
            anchors = anchors.detach()

        local_anchors.append(anchors)
        if include_tokens:
            local_token_states.extend(token_states)

        if encode_side == "qry":
            local_infos.extend(dataset_info)
        else:
            local_infos.extend([info.get("cand_name", "") for info in dataset_info])

    if not local_anchors:
        empty_tokens = [] if include_tokens else None
        return np.array([]), [], empty_tokens

    local_anchor_tensor = torch.cat(local_anchors, dim=0).contiguous()

    if dist.is_initialized():
        gathered_anchors = [torch.zeros_like(local_anchor_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_anchors, local_anchor_tensor)

        gathered_infos = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_infos, local_infos, group=object_group)

        gathered_tokens = None
        if include_tokens:
            gathered_tokens = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object(
                local_token_states,
                object_gather_list=gathered_tokens,
                dst=0,
                group=object_group,
            )

        if rank == 0:
            final_anchors = torch.cat(gathered_anchors, dim=0).cpu().float().numpy()
            final_infos = [info for rank_list in gathered_infos for info in rank_list]
            final_tokens = None
            if include_tokens:
                final_tokens = [tokens for rank_list in gathered_tokens for tokens in rank_list]
                final_tokens = final_tokens[:full_dataset_len]
            return (
                final_anchors[:full_dataset_len],
                final_infos[:full_dataset_len],
                final_tokens,
            )
        return None, None, None

    return (
        local_anchor_tensor.cpu().float().numpy(),
        local_infos,
        local_token_states if include_tokens else None,
    )


def _as_token_tensor(token_array, device, dtype):
    if isinstance(token_array, torch.Tensor):
        tensor = token_array.detach()
    else:
        tensor = torch.from_numpy(token_array)
    return tensor.to(device=device, dtype=dtype)


@torch.no_grad()
def compute_late_scores_for_query_batch(
    query_token_arrays,
    candidate_token_arrays,
    device,
    candidate_chunk_size: int,
):
    if not query_token_arrays:
        return torch.empty(0, len(candidate_token_arrays), device=device)

    dtype = torch.float32
    query_lengths = [max(1, len(tokens)) for tokens in query_token_arrays]
    max_query_len = max(query_lengths)
    dim = query_token_arrays[0].shape[-1]

    query_tensor = torch.zeros(
        len(query_token_arrays), max_query_len, dim, device=device, dtype=dtype
    )
    query_mask = torch.zeros(
        len(query_token_arrays), max_query_len, device=device, dtype=torch.bool
    )
    for query_idx, tokens in enumerate(query_token_arrays):
        token_tensor = _as_token_tensor(tokens, device=device, dtype=dtype)
        query_tensor[query_idx, : token_tensor.shape[0]] = token_tensor
        query_mask[query_idx, : token_tensor.shape[0]] = True

    chunk_scores = []
    for start in range(0, len(candidate_token_arrays), candidate_chunk_size):
        chunk = candidate_token_arrays[start : start + candidate_chunk_size]
        cand_lengths = [max(1, len(tokens)) for tokens in chunk]
        max_cand_len = max(cand_lengths)

        cand_tensor = torch.zeros(len(chunk), max_cand_len, dim, device=device, dtype=dtype)
        cand_mask = torch.zeros(len(chunk), max_cand_len, device=device, dtype=torch.bool)
        for cand_idx, tokens in enumerate(chunk):
            token_tensor = _as_token_tensor(tokens, device=device, dtype=dtype)
            cand_tensor[cand_idx, : token_tensor.shape[0]] = token_tensor
            cand_mask[cand_idx, : token_tensor.shape[0]] = True

        sims = torch.einsum("bqd,ckd->bqck", query_tensor, cand_tensor)
        min_value = torch.finfo(sims.dtype).min
        sims = sims.masked_fill(~cand_mask[None, None, :, :], min_value)
        max_sims = sims.max(dim=-1).values.float()
        max_sims = max_sims.masked_fill(~query_mask[:, :, None], 0.0)
        denom = query_mask.sum(dim=1).clamp_min(1).float()[:, None]
        chunk_scores.append(max_sims.sum(dim=1) / denom)

        del cand_tensor, cand_mask, sims, max_sims

    return torch.cat(chunk_scores, dim=1)


def _expected_scoring_config(
    late_args: LateInteractionArguments,
    adapter_args: AdapterArguments,
    adapter_runtime: AdapterRuntime,
):
    return {
        "enable_late_interaction": bool(late_args.enable_late_interaction),
        "lambda_anchor": float(late_args.lambda_anchor),
        "lambda_late": float(late_args.lambda_late),
        "selected_layer": int(adapter_args.selected_layer),
        "adapter_type": adapter_runtime.adapter_type,
        "adapter_file": adapter_runtime.adapter_file,
        "adapter_fingerprint": adapter_runtime.adapter_fingerprint,
        "adapter_param_count": int(adapter_runtime.adapter_param_count),
        "late_representation": "adapter_outputs",
        "tie_breaker": "query_candidate_hash",
    }


def _score_cache_matches(score_dict, expected_config):
    return (
        "num_pred" in score_dict
        and score_dict.get("scoring_config") == expected_config
    )


def _stable_hash_int(*parts):
    hasher = hashlib.blake2b(digest_size=8)
    for part in parts:
        hasher.update(str(part).encode("utf-8", errors="surrogatepass"))
        hasher.update(b"\0")
    return int.from_bytes(hasher.digest(), byteorder="big", signed=False)


def _argsort_desc_with_hash_tiebreak(scores, candidate_names, query_key):
    tie_values = np.array(
        [_stable_hash_int(query_key, cand_name) for cand_name in candidate_names],
        dtype=np.uint64,
    )
    tie_order_np = np.argsort(tie_values, kind="stable")
    tie_order = torch.as_tensor(tie_order_np, device=scores.device, dtype=torch.long)

    ordered_scores = scores.index_select(0, tie_order)
    ranked_in_tie_order = torch.argsort(ordered_scores, descending=True, stable=True)
    ranked_indices = tie_order.index_select(0, ranked_in_tie_order)
    return ranked_indices.cpu().numpy()


def _rank_global(
    model,
    qry_embeds,
    cand_embed_dict,
    query_tokens,
    cand_token_dict,
    late_args,
    local_rank,
    dataset_name,
):
    device = model.device
    cand_keys = list(cand_embed_dict.keys())
    cand_embeds = np.stack([cand_embed_dict[key] for key in cand_keys])
    cand_tensor = torch.from_numpy(cand_embeds).to(device=device, dtype=torch.float32)
    qry_tensor = torch.from_numpy(qry_embeds).to(device=device, dtype=torch.float32)

    all_ranked_indices = []
    use_late = late_args.enable_late_interaction
    query_chunk_size = max(1, late_args.late_query_chunk_size)
    candidate_chunk_size = max(1, late_args.late_candidate_chunk_size)
    candidate_tokens = None
    if use_late:
        candidate_tokens = [cand_token_dict[key] for key in cand_keys]

    progress = tqdm(
        range(0, qry_tensor.shape[0], query_chunk_size),
        desc=f"Global Ranking: {dataset_name}",
        disable=local_rank > 0,
        ncols=120,
    )
    for start in progress:
        end = min(start + query_chunk_size, qry_tensor.shape[0])
        anchor_scores = model.compute_similarity(qry_tensor[start:end], cand_tensor)
        if use_late:
            late_scores = compute_late_scores_for_query_batch(
                query_tokens[start:end],
                candidate_tokens,
                device=device,
                candidate_chunk_size=candidate_chunk_size,
            )
            scores = late_args.lambda_anchor * anchor_scores + late_args.lambda_late * late_scores
        else:
            scores = anchor_scores

        for row_idx in range(scores.shape[0]):
            all_ranked_indices.append(
                _argsort_desc_with_hash_tiebreak(
                    scores=scores[row_idx],
                    candidate_names=cand_keys,
                    query_key=f"{dataset_name}:{start + row_idx}",
                )
            )

        del anchor_scores, scores
        if use_late:
            del late_scores

    del cand_tensor, qry_tensor
    torch.cuda.empty_cache()
    return cand_keys, all_ranked_indices


def _rank_local(
    model,
    qry_embeds,
    cand_embed_dict,
    gt_infos,
    query_tokens,
    cand_token_dict,
    late_args,
    local_rank,
    dataset_name,
):
    device = model.device
    qry_tensor = torch.from_numpy(qry_embeds).to(device=device, dtype=torch.float32)
    candidate_chunk_size = max(1, late_args.late_candidate_chunk_size)

    ranked_name_lists = []
    progress = tqdm(
        enumerate(zip(qry_tensor, gt_infos)),
        total=len(gt_infos),
        desc=f"Local Ranking: {dataset_name}",
        disable=local_rank > 0,
        ncols=120,
    )
    for qid, (qry_vec, gt_info) in progress:
        cand_names = gt_info["cand_names"]
        cand_embeds = np.stack([cand_embed_dict[name] for name in cand_names])
        cand_tensor = torch.from_numpy(cand_embeds).to(device=device, dtype=torch.float32)

        anchor_scores = model.compute_similarity(qry_vec.unsqueeze(0), cand_tensor).squeeze(0)
        if late_args.enable_late_interaction:
            late_scores = compute_late_scores_for_query_batch(
                [query_tokens[qid]],
                [cand_token_dict[name] for name in cand_names],
                device=device,
                candidate_chunk_size=candidate_chunk_size,
            ).squeeze(0)
            scores = late_args.lambda_anchor * anchor_scores + late_args.lambda_late * late_scores
        else:
            scores = anchor_scores

        ranked_idx = _argsort_desc_with_hash_tiebreak(
            scores=scores,
            candidate_names=cand_names,
            query_key=f"{dataset_name}:{qid}",
        )
        ranked_name_lists.append([cand_names[i] for i in ranked_idx])

        del cand_tensor, anchor_scores, scores, ranked_idx
        if late_args.enable_late_interaction:
            del late_scores

    del qry_tensor
    torch.cuda.empty_cache()
    return ranked_name_lists


def _select_first(dataset, n: int):
    n = min(max(1, n), len(dataset))
    return dataset.select(range(n)) if hasattr(dataset, "select") else [dataset[i] for i in range(n)]


def run_sanity_check(
    model: MMEBEmbeddingModel,
    data_args: DataArguments,
    model_args: ModelArguments,
    eval_args: EvalArguments,
    late_args: LateInteractionArguments,
    adapter_args: AdapterArguments,
    adapter_runtime: AdapterRuntime,
    dataset_configs: Dict[str, Any],
    object_group=None,
):
    if not dataset_configs:
        raise ValueError("Dataset config is empty; cannot run sanity check.")

    dataset_name, task_config = next(iter(dataset_configs.items()))
    task_config = copy.deepcopy(task_config)
    print_master(f"=== Adapter sanity check: {dataset_name} ===")

    if data_args.data_basedir is not None:
        for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
            if task_config.get(key):
                task_config[key] = os.path.join(data_args.data_basedir, task_config[key])

    full_eval_qry_dataset, corpus = AutoEvalPairDataset.instantiate(
        model_args=model_args, data_args=data_args, **task_config
    )
    full_eval_cand_dataset = generate_cand_dataset(full_eval_qry_dataset, corpus)
    sample_count = min(
        max(1, adapter_args.sanity_check_num_examples),
        len(full_eval_qry_dataset),
        len(full_eval_cand_dataset),
    )
    qry_dataset = _select_first(full_eval_qry_dataset, sample_count)
    cand_dataset = _select_first(full_eval_cand_dataset, sample_count)

    qry_loader = DataLoader(
        qry_dataset,
        batch_size=sample_count,
        collate_fn=MultimodalEvalDataCollator(encode_side="qry"),
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )
    cand_loader = DataLoader(
        cand_dataset,
        batch_size=sample_count,
        collate_fn=MultimodalEvalDataCollator(encode_side="cand"),
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )

    debug_state: Dict[str, Any] = {}
    token_filter_log_sides: set = set()
    query_embeds, gt_infos, query_tokens = encode_representations(
        model=model,
        loader=qry_loader,
        encode_side="qry",
        full_dataset_len=sample_count,
        include_tokens=True,
        layer_index=adapter_args.selected_layer,
        adapter_runtime=adapter_runtime,
        token_filter_log_sides=token_filter_log_sides,
        description=f"Sanity queries: {dataset_name}",
        object_group=object_group,
        debug_state=debug_state,
    )
    cand_embeds, cand_ids, cand_tokens = encode_representations(
        model=model,
        loader=cand_loader,
        encode_side="cand",
        full_dataset_len=sample_count,
        include_tokens=True,
        layer_index=adapter_args.selected_layer,
        adapter_runtime=adapter_runtime,
        token_filter_log_sides=token_filter_log_sides,
        description=f"Sanity candidates: {dataset_name}",
        object_group=object_group,
        debug_state=debug_state,
    )

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return

    device = model.device
    qry_tensor = torch.from_numpy(query_embeds).to(device=device, dtype=torch.float32)
    cand_tensor = torch.from_numpy(cand_embeds).to(device=device, dtype=torch.float32)
    anchor_scores = model.compute_similarity(qry_tensor, cand_tensor)
    late_scores = compute_late_scores_for_query_batch(
        query_tokens,
        cand_tokens,
        device=device,
        candidate_chunk_size=max(1, late_args.late_candidate_chunk_size),
    )
    final_scores = late_args.lambda_anchor * anchor_scores + late_args.lambda_late * late_scores

    query_debug = debug_state.get("qry", {})
    cand_debug = debug_state.get("cand", {})
    print_master("Sanity query input format:")
    print_master(_safe_json(query_debug.get("formatted_input"), adapter_args.sanity_check_print_chars))
    print_master("Sanity candidate input format:")
    print_master(_safe_json(cand_debug.get("formatted_input"), adapter_args.sanity_check_print_chars))
    for side_name, side_debug in [("query", query_debug), ("candidate", cand_debug)]:
        print_master(f"Sanity {side_name} hidden-state shape before adapter: {side_debug.get('hidden_state_shape_before_adapter')}")
        print_master(f"Sanity {side_name} token representation shape after adapter: {side_debug.get('token_representation_shape_after_adapter')}")
        print_master(f"Sanity {side_name} dtype: {side_debug.get('dtype')}")
        print_master(f"Sanity {side_name} outputs finite: {side_debug.get('outputs_finite')}")
    print_master(f"Sanity MaxSim score shape: {list(late_scores.shape)}")
    print_master(f"Sanity final fused score shape: {list(final_scores.shape)}")
    print_master(f"Sanity candidate ids: {cand_ids}")
    print_master(f"Sanity gt info count: {len(gt_infos)}")
    print_master("=== Adapter sanity check complete; exiting before full evaluation. ===")


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))

    object_gather_group = (
        dist.new_group(backend="gloo") if dist.is_initialized() else None
    )

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    print_master("=== Distributed Setup Initialized ===")
    print_master(f"Master Info -> ADDR: {os.environ.get('MASTER_ADDR')}, PORT: {os.environ.get('MASTER_PORT')}")
    print_master(f"Global World Size: {world_size}")
    print_rank(f"Process Identity -> Rank: {rank}, Local Rank: {local_rank} on {torch.cuda.get_device_name()}")

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, EvalArguments, LateInteractionArguments, AdapterArguments)
    )
    model_args, data_args, eval_args, late_args, adapter_args = parser.parse_args_into_dataclasses()
    if data_args.encode_output_path is None:
        data_args.encode_output_path = eval_args.output_dir
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    layer_index = int(adapter_args.selected_layer)

    # DDP-safe model loading
    if rank == 0:
        print_master(f"[rank=0] Loading the model from: {model_args.model_name_or_path}...")
        model = MMEBEmbeddingModel.load(
            model_name_or_path=model_args.model_name_or_path,
            normalize=model_args.normalize,
            instruction=model_args.instruction,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if rank != 0:
        print_rank("Loading the model from cache...")
        time.sleep(random.randint(2 * rank, 3 * rank))
        model = MMEBEmbeddingModel.load(
            model_name_or_path=model_args.model_name_or_path,
            normalize=model_args.normalize,
            instruction=model_args.instruction,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    model = model.to(eval_args.device, dtype=torch.bfloat16)

    hidden_size = int(model.encoder.model.get_input_embeddings().embedding_dim)
    adapter_runtime = load_adapter_runtime(
        adapter_args=adapter_args,
        hidden_size=hidden_size,
        device=eval_args.device,
        dtype=torch.bfloat16,
    )
    adapter_file_tag = _adapter_tag(adapter_runtime, layer_index)

    print_master("=== Adapter late-interaction configuration ===")
    print_master(f"  base_model_path         : {model_args.model_name_or_path}")
    print_master(f"  adapter_type            : {adapter_runtime.adapter_type}")
    print_master(f"  adapter_checkpoint_dir  : {adapter_runtime.adapter_checkpoint_dir}")
    print_master(f"  resolved_adapter_file   : {adapter_runtime.adapter_file}")
    print_master(f"  adapter_fingerprint     : {adapter_runtime.adapter_fingerprint}")
    print_master(f"  adapter_params_loaded   : {adapter_runtime.adapter_param_count}")
    print_master(f"  missing_checkpoint_keys : {adapter_runtime.missing_keys}")
    print_master(f"  unexpected_ckpt_keys    : {adapter_runtime.unexpected_keys}")
    print_master(f"  enable_late_interaction : {late_args.enable_late_interaction}")
    print_master(f"  selected_hidden_layer   : {layer_index}")
    print_master("  late_representation     : adapter outputs (not raw hidden states)")
    print_master(f"  lambda_anchor           : {late_args.lambda_anchor}")
    print_master(f"  lambda_late             : {late_args.lambda_late}")
    print_master(f"  late_query_chunk_size   : {late_args.late_query_chunk_size}")
    print_master(f"  late_cand_chunk_size    : {late_args.late_candidate_chunk_size}")
    print_master(f"  output_path             : {data_args.encode_output_path}")
    print_master(f"  adapter_tag (file suffix): {adapter_file_tag}")
    print_master(
        "  final_score_formula    : "
        "final_score = lambda_anchor * anchor_score + lambda_late * adapted_late_score"
    )

    with open(data_args.dataset_config, "r") as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)

    if adapter_args.sanity_check:
        run_sanity_check(
            model=model,
            data_args=data_args,
            model_args=model_args,
            eval_args=eval_args,
            late_args=late_args,
            adapter_args=adapter_args,
            adapter_runtime=adapter_runtime,
            dataset_configs=dataset_configs,
            object_group=object_gather_group,
        )
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    token_filter_log_sides: set = set()

    for dataset_name, task_config in dataset_configs.items():
        if dist.is_initialized():
            dist.barrier()
        print_master(f"--- Evaluating {dataset_name} ---")

        # Anchor caches stay layer-agnostic (final-layer pooled embedding); adapted token
        # caches and score/pred files are namespaced by adapter type, layer, and checkpoint
        # fingerprint so they cannot collide with raw late-interaction outputs.
        query_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_qry")
        cand_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_tgt")
        query_token_path = os.path.join(
            data_args.encode_output_path, f"{dataset_name}_qry_tok_{adapter_file_tag}"
        )
        cand_token_path = os.path.join(
            data_args.encode_output_path, f"{dataset_name}_tgt_tok_{adapter_file_tag}"
        )
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_info.jsonl")

        do_query = not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path)
        do_cand = not os.path.exists(cand_embed_path)
        
        score_path = os.path.join(
            data_args.encode_output_path, f"{dataset_name}_score.json"
        )
        if os.path.exists(score_path):
            do_query = False
            do_cand = False

        # if late_args.enable_late_interaction:
        #     do_query = do_query or not os.path.exists(query_token_path)
        #     do_cand = do_cand or not os.path.exists(cand_token_path)

        if do_query or do_cand:
            if data_args.data_basedir is not None:
                for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
                    if task_config.get(key):
                        task_config[key] = os.path.join(data_args.data_basedir, task_config[key])

            try:
                full_eval_qry_dataset, corpus = AutoEvalPairDataset.instantiate(
                    model_args=model_args, data_args=data_args, **task_config
                )
                full_eval_cand_dataset = generate_cand_dataset(full_eval_qry_dataset, corpus)
                eval_qry_dataset, eval_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset

                if dist.is_initialized():
                    padded_qry_dataset, _ = pad_dataset_to_divisible(full_eval_qry_dataset, world_size)
                    padded_cand_dataset, _ = pad_dataset_to_divisible(full_eval_cand_dataset, world_size)
                    eval_qry_dataset = split_dataset_by_node(
                        padded_qry_dataset, rank=rank, world_size=world_size
                    )
                    eval_cand_dataset = split_dataset_by_node(
                        padded_cand_dataset, rank=rank, world_size=world_size
                    )
            except Exception as e:
                print_master(f"Failed to load dataset {dataset_name}, skipping {dataset_name}")
                import traceback

                traceback.print_exc()
                print_master(e)
                raise e

        if do_query:
            print_master("Encoding queries...")
            eval_qry_loader = DataLoader(
                eval_qry_dataset,
                batch_size=eval_args.per_device_eval_batch_size,
                collate_fn=MultimodalEvalDataCollator(encode_side="qry"),
                num_workers=eval_args.dataloader_num_workers,
                pin_memory=True,
                shuffle=False,
            )
            query_embeds, gt_infos, query_token_states = encode_representations(
                model=model,
                loader=eval_qry_loader,
                encode_side="qry",
                full_dataset_len=len(full_eval_qry_dataset),
                include_tokens=late_args.enable_late_interaction,
                layer_index=layer_index,
                adapter_runtime=adapter_runtime,
                token_filter_log_sides=token_filter_log_sides,
                description=f"Queries: {dataset_name}",
                object_group=object_gather_group,
            )
            if rank == 0:
                os.makedirs(os.path.dirname(query_embed_path), exist_ok=True)

                with open(query_embed_path, "wb") as f:
                    pickle.dump(query_embeds, f)

                with open(dataset_info_path, "w", encoding="utf-8") as f:
                    for info in gt_infos:
                        f.write(json.dumps(info, ensure_ascii=False) + "\n")

                if late_args.enable_late_interaction:
                    with open(query_token_path, "wb") as f:
                        pickle.dump(query_token_states, f)
                print_master(f"Successfully saved {len(query_embeds)} query embeddings to {query_embed_path}")

            if dist.is_initialized():
                dist.barrier()

        if do_cand:
            print_master("Encoding candidates...")
            eval_cand_loader = DataLoader(
                eval_cand_dataset,
                batch_size=eval_args.per_device_eval_batch_size,
                collate_fn=MultimodalEvalDataCollator(encode_side="cand"),
                num_workers=eval_args.dataloader_num_workers,
                pin_memory=True,
                shuffle=False,
            )
            cand_embeds, all_cand_ids, cand_token_states = encode_representations(
                model=model,
                loader=eval_cand_loader,
                encode_side="cand",
                full_dataset_len=len(full_eval_cand_dataset),
                include_tokens=late_args.enable_late_interaction,
                layer_index=layer_index,
                adapter_runtime=adapter_runtime,
                token_filter_log_sides=token_filter_log_sides,
                description=f"Candidates: {dataset_name}",
                object_group=object_gather_group,
            )
            if rank == 0:
                os.makedirs(os.path.dirname(cand_embed_path), exist_ok=True)

                cand_embed_dict = {
                    cand_id: embed for cand_id, embed in zip(all_cand_ids, cand_embeds)
                }
                with open(cand_embed_path, "wb") as f:
                    pickle.dump(cand_embed_dict, f)

                if late_args.enable_late_interaction:
                    cand_token_dict = {
                        cand_id: tokens for cand_id, tokens in zip(all_cand_ids, cand_token_states)
                    }
                    with open(cand_token_path, "wb") as f:
                        pickle.dump(cand_token_dict, f)
                print_master(f"Successfully saved {len(cand_embed_dict)} unique candidate embeddings to {cand_embed_path}")

            if dist.is_initialized():
                dist.barrier()

        if rank == 0:
            score_path = os.path.join(
                data_args.encode_output_path, f"{dataset_name}_score.json"
            )
            pred_path = os.path.join(
                data_args.encode_output_path, f"{dataset_name}_pred.jsonl"
            )
            score_alias_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
            pred_alias_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_pred.jsonl")
            expected_config = _expected_scoring_config(late_args, adapter_args, adapter_runtime)

            need_compute = True
            if os.path.exists(score_path) and os.path.exists(pred_path):
                try:
                    with open(score_path, "r") as f:
                        score_dict = json.load(f)
                    if _score_cache_matches(score_dict, expected_config):
                        print_master(f"Results already exist for {dataset_name}. Skipping computation.")
                        formatted = {
                            k: f"{v:.4f}"
                            for k, v in score_dict.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)
                        }
                        print_master(f"Scores: {formatted}")
                        shutil.copyfile(score_path, score_alias_path)
                        shutil.copyfile(pred_path, pred_alias_path)
                        need_compute = False
                    else:
                        print_master(f"Scoring config changed for {dataset_name}, re-computing...")
                except Exception as e:
                    print_master(f"Cache for {dataset_name} is corrupted ({e}), re-computing...")
            
            if os.path.exists(score_alias_path):
                need_compute = False

            if need_compute:
                with open(query_embed_path, "rb") as f:
                    qry_embeds = pickle.load(f)
                with open(cand_embed_path, "rb") as f:
                    cand_embed_dict = pickle.load(f)

                gt_infos = [json.loads(l) for l in open(dataset_info_path, encoding="utf-8")]

                query_tokens = cand_token_dict = None
                if late_args.enable_late_interaction:
                    with open(query_token_path, "rb") as f:
                        query_tokens = pickle.load(f)
                    with open(cand_token_path, "rb") as f:
                        cand_token_dict = pickle.load(f)

                pred_dicts = []
                rank_against_all_candidates = task_config.get("eval_type", "global") == "global"

                if rank_against_all_candidates:
                    cand_keys, ranked_indices = _rank_global(
                        model=model,
                        qry_embeds=qry_embeds,
                        cand_embed_dict=cand_embed_dict,
                        query_tokens=query_tokens,
                        cand_token_dict=cand_token_dict,
                        late_args=late_args,
                        local_rank=local_rank,
                        dataset_name=dataset_name,
                    )
                    for ranked_idx, gt_info in zip(ranked_indices, gt_infos):
                        rel_docids = (
                            gt_info["label_name"]
                            if isinstance(gt_info["label_name"], list)
                            else [gt_info["label_name"]]
                        )
                        pred_dicts.append(
                            {
                                "prediction": [cand_keys[i] for i in ranked_idx],
                                "label": rel_docids,
                                "rel_scores": gt_info.get("rel_scores", None),
                            }
                        )
                else:
                    ranked_name_lists = _rank_local(
                        model=model,
                        qry_embeds=qry_embeds,
                        cand_embed_dict=cand_embed_dict,
                        gt_infos=gt_infos,
                        query_tokens=query_tokens,
                        cand_token_dict=cand_token_dict,
                        late_args=late_args,
                        local_rank=local_rank,
                        dataset_name=dataset_name,
                    )
                    for ranked_names, gt_info in zip(ranked_name_lists, gt_infos):
                        rel_docids = (
                            gt_info["label_name"]
                            if isinstance(gt_info["label_name"], list)
                            else [gt_info["label_name"]]
                        )
                        pred_dicts.append(
                            {
                                "prediction": ranked_names,
                                "label": rel_docids,
                                "rel_scores": gt_info.get("rel_scores", None),
                            }
                        )

                metrics_to_report = task_config.get(
                    "metrics", ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]
                )
                metrics = RankingMetrics(metrics_to_report)
                score_dict = metrics.evaluate(pred_dicts)

                score_dict["num_pred"] = len(pred_dicts)
                score_dict["num_data"] = len(gt_infos)
                score_dict["scoring_config"] = expected_config

                with open(score_path, "w") as f:
                    json.dump(score_dict, f, indent=4)

                with open(pred_path, "w", encoding="utf-8") as f:
                    for pred in pred_dicts:
                        f.write(json.dumps(pred, ensure_ascii=False) + "\n")

                # shutil.copyfile(score_path, score_alias_path)
                # shutil.copyfile(pred_path, pred_alias_path)

                formatted = {
                    k: f"{v:.4f}"
                    for k, v in score_dict.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                print_master(f"Final Score for {dataset_name} [{adapter_file_tag}]: {formatted}")
                print_master(f"  score file -> {score_path}")
                print_master(f"  pred file  -> {pred_path}")
                print_master(f"  score alias -> {score_alias_path}")
                print_master(f"  pred alias  -> {pred_alias_path}")
                
        if rank == 0:
            try:
                os.remove(query_embed_path)
                os.remove(cand_embed_path)
                os.remove(query_token_path)
                os.remove(cand_token_path)
            except Exception as e:
                print(e)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
