from __future__ import annotations

import logging
import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from qwen_vl_utils.vision_process import process_vision_info
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

from .utils import ensure_file_uri, normalize_text

logger = logging.getLogger(__name__)

IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_LENGTH = 8192
DEFAULT_INSTRUCTION = "Represent the user's input."
PAD_TOKEN = "<|endoftext|>"


def _token_ids(tokenizer: Any, token_strings: Sequence[str]) -> set[int]:
    ids: set[int] = set()
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in token_strings:
        token_id = vocab.get(token)
        if token_id is None and hasattr(tokenizer, "convert_tokens_to_ids"):
            token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != unk_id:
            ids.add(int(token_id))
    return ids


def _late_keep_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
) -> torch.Tensor:
    """Match MMEB late-interaction eval token filtering.

    Keep normal non-padding tokens and visual placeholders, but remove chat and
    vision boundary special tokens from the token set used for MaxSim.
    """
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


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Training wrapper mirroring the official Qwen3-VL-Embedding model class."""

    config_class = Qwen3VLConfig
    accepts_loss_kwargs = False
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Qwen3VLForEmbeddingOutput:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class LayerWeightedTokenProjectionModule(nn.Module):
    """
    Project selected frozen-backbone token states into residual multi-vectors.

    The backbone is always run under torch.no_grad(), so the hidden states
    passed here carry no gradient w.r.t. backbone weights. Gradients only flow
    through this module's own parameters (layer_weight_vectors and proj).

    Args:
        hidden_size:       Dimension of the backbone hidden states (H).
        residual_dim:      Output dimension of the projection head (D).
        num_input_layers:  Number of backbone layers provided to the module.
                           If > 1, learn a per-token softmax over source
                           layers and emit one fused hidden state per token.
    """

    def __init__(
        self,
        hidden_size: int,
        residual_dim: int,
        num_input_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_input_layers = num_input_layers
        # One fused (B, T, H) per token. Setting this to N would emit N output
        # slots per token and multiply downstream seq-len by N.
        self.num_output_layers = 1

        if num_input_layers > 1:
            self.layer_weight_vectors = nn.Parameter(torch.empty(self.num_output_layers, hidden_size))
            nn.init.normal_(self.layer_weight_vectors, mean=0.0, std=0.02)
            self.layer_weight_scale = hidden_size ** -0.5
        else:
            self.layer_weight_vectors = None
            self.layer_weight_scale = None

        # Token projection: H → residual_dim with layer norm.
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, residual_dim, bias=False),
        )
        nn.init.xavier_uniform_(self.proj[1].weight)

    @staticmethod
    def _rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def _merge_multi_layer_states(
        self,
        hidden_states: Union[List[torch.Tensor], tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        if len(hidden_states) != self.num_input_layers:
            raise ValueError(
                f"Expected {self.num_input_layers} selected layers, got {len(hidden_states)}"
            )
        if self.layer_weight_vectors is None:
            return hidden_states[-1]

        # Stream over the N source layers to avoid materializing (B, T, N, H)
        # and its fp32 copy, which was the OOM source.
        ref = hidden_states[0]
        dtype = ref.dtype
        normed_layer_vectors = self._rms_normalize(self.layer_weight_vectors.to(dtype))  # (L, H)

        per_layer_scores = []
        for h in hidden_states:
            h_normed = self._rms_normalize(h)  # (B, T, H) in bf16
            per_layer_scores.append(torch.einsum("bth,lh->btl", h_normed, normed_layer_vectors))
        layer_scores = torch.stack(per_layer_scores, dim=2)  # (B, T, N, L)
        layer_weights = F.softmax(layer_scores * self.layer_weight_scale, dim=2)  # (B, T, N, L)

        batch_size, seq_len, _ = ref.shape
        num_output_layers = self.num_output_layers
        hidden_size = ref.shape[-1]
        merged = ref.new_zeros(batch_size, seq_len, num_output_layers, hidden_size)
        for idx, h in enumerate(hidden_states):
            # layer_weights[:, :, idx]: (B, T, L) -> (B, T, L, 1) * (B, T, 1, H)
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
            raise ValueError(f"Expected hidden states with rank 3 or 4, got shape {tuple(hidden_states.shape)}")

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
        """
        Args:
            hidden_states:
                Single tensor (B, L, H) when one backbone layer is used.
                List of N tensors each (B, L, H) when multiple backbone layers
                are fused; the list must contain exactly N selected layers.
            attention_mask:
                Token validity mask from the backbone input.

        Returns:
            residual_raw:  (B, K', H)  fused token states before projection.
            residual:      (B, K', D)  projected and L2-normalised token vectors.
            residual_mask: (B, K')     validity mask aligned with residual.
        """
        if isinstance(hidden_states, (list, tuple)):
            if attention_mask is not None:
                compacted_layers: List[torch.Tensor] = []
                compacted_mask: Optional[torch.Tensor] = None
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


class TokenProjectionQwen3VLEmbedder(nn.Module):
    """
    Token-projection residual embedder for Qwen3-VL-Embedding.

    The backbone is kept entirely frozen and is run **once** per input to
    produce:
      1. A strong single-vector anchor (same as the base model's embedding).
      2. A sequence of selected hidden states that are projected token-wise
         into late-interaction multi-vectors.

    MaxSim late interaction over these projected token vectors produces a
    residual retrieval score that is combined with the anchor score at
    training time.
    """

    def __init__(
        self,
        model_name_or_path: str,
        residual_dim: int = 256,
        anchor_dim: int = 2048,
        use_last_n_layers: int = 1,
        selected_layers: Optional[Sequence[int]] = None,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        default_instruction: str = DEFAULT_INSTRUCTION,
        query_instruction: Optional[str] = None,
        document_instruction: Optional[str] = None,
        query_encode_batch_size: Optional[int] = None,
        doc_encode_batch_size: Optional[int] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.default_instruction = default_instruction
        self.query_instruction = query_instruction
        self.document_instruction = document_instruction
        self.query_encode_batch_size = query_encode_batch_size
        self.doc_encode_batch_size = doc_encode_batch_size
        self.anchor_dim = anchor_dim
        self.use_last_n_layers = use_last_n_layers
        self.selected_layers = None if selected_layers is None else [int(layer) for layer in selected_layers]

        self.backbone = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        self.processor = Qwen3VLProcessor.from_pretrained(model_name_or_path, padding_side="right")

        hidden_size = self.backbone.get_input_embeddings().embedding_dim
        self.hidden_size = hidden_size
        self.num_backbone_layers: Optional[int] = None
        if anchor_dim > hidden_size:
            raise ValueError(f"anchor_dim={anchor_dim} exceeds hidden_size={hidden_size}")
        if self.selected_layers is not None:
            if not self.selected_layers:
                raise ValueError("selected_layers must contain at least one layer index")
            if len(set(self.selected_layers)) != len(self.selected_layers):
                raise ValueError(f"selected_layers must not contain duplicates: {self.selected_layers}")
        if use_last_n_layers < 1:
            raise ValueError(f"use_last_n_layers must be >= 1, got {use_last_n_layers}")

        self.num_fused_layers = (
            len(self.selected_layers) if self.selected_layers is not None else self.use_last_n_layers
        )

        # Freeze the entire backbone — it is always run under torch.no_grad().
        for param in self.backbone.parameters():
            param.requires_grad = False

        # The only trainable component: token-projection residual module.
        self.token_projection_module = LayerWeightedTokenProjectionModule(
            hidden_size=hidden_size,
            residual_dim=residual_dim,
            num_input_layers=self.num_fused_layers,
        )

        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen backbone in eval mode at all times.
        self.backbone.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self.token_projection_module.proj[1].weight.device

    def _resolve_layer_indices(self, backbone_hidden_states: Sequence[torch.Tensor]) -> List[int]:
        num_backbone_layers = len(backbone_hidden_states) - 1
        if num_backbone_layers <= 0:
            raise ValueError("Backbone did not return per-layer hidden states")

        self.num_backbone_layers = num_backbone_layers
        if self.selected_layers is not None:
            invalid_layers = [
                layer_idx for layer_idx in self.selected_layers
                if layer_idx < -num_backbone_layers or layer_idx >= num_backbone_layers
            ]
            if invalid_layers:
                raise ValueError(
                    f"selected_layers out of range for backbone with {num_backbone_layers} layers: "
                    f"{invalid_layers}"
                )
            resolved_layers = [
                layer_idx if layer_idx >= 0 else num_backbone_layers + layer_idx
                for layer_idx in self.selected_layers
            ]
            if len(set(resolved_layers)) != len(resolved_layers):
                raise ValueError(
                    f"selected_layers resolve to duplicate layer indices for backbone with "
                    f"{num_backbone_layers} layers: {self.selected_layers} -> {resolved_layers}"
                )
            return resolved_layers

        if self.use_last_n_layers > num_backbone_layers:
            raise ValueError(
                f"use_last_n_layers={self.use_last_n_layers} exceeds backbone depth {num_backbone_layers}"
            )
        start_layer = num_backbone_layers - self.use_last_n_layers
        return list(range(start_layer, num_backbone_layers))

    def _uses_final_hidden_state_only(self) -> bool:
        return self.selected_layers == [-1] or (
            self.selected_layers is None and self.use_last_n_layers == 1
        )

    def forward(
        self,
        queries: Sequence[Dict[str, Optional[str]]],
        docs: Sequence[Dict[str, Optional[str]]],
    ) -> Dict[str, torch.Tensor]:
        (
            query_anchor,
            query_anchor_raw,
            query_residual_raw,
            query_residual,
            query_residual_mask,
        ) = self.encode_side(
            queries, is_query=True
        )
        (
            doc_anchor,
            doc_anchor_raw,
            doc_residual_raw,
            doc_residual,
            doc_residual_mask,
        ) = self.encode_side(
            docs, is_query=False
        )
        return {
            "query_anchor": query_anchor,
            "query_anchor_raw": query_anchor_raw,
            "query_residual_raw": query_residual_raw,
            "query_residual": query_residual,
            "query_residual_mask": query_residual_mask,
            "doc_anchor": doc_anchor,
            "doc_anchor_raw": doc_anchor_raw,
            "doc_residual_raw": doc_residual_raw,
            "doc_residual": doc_residual,
            "doc_residual_mask": doc_residual_mask,
        }

    def trainable_parameter_groups(self, weight_decay: float) -> List[Dict[str, Any]]:
        no_decay: List[torch.nn.Parameter] = []
        with_decay: List[torch.nn.Parameter] = []
        for name, param in self.token_projection_module.named_parameters():
            if "norm" in name or "bias" in name:
                no_decay.append(param)
            else:
                with_decay.append(param)
        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": with_decay, "weight_decay": weight_decay},
        ]

    def save_adapter(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        state = {
            "token_projection_module": self.token_projection_module.state_dict(),
            "adapter_type": "token_projection",
            "use_last_n_layers": self.use_last_n_layers,
            "selected_layers": self.selected_layers,
            "anchor_dim": self.anchor_dim,
            "hidden_size": self.hidden_size,
            "default_instruction": self.default_instruction,
        }
        torch.save(state, os.path.join(output_dir, "adapter.pt"))
        self.processor.save_pretrained(output_dir)

    # ------------------------------------------------------------------
    # Input preprocessing helpers 
    # ------------------------------------------------------------------

    @staticmethod
    def _close_modal_items(items: Optional[Sequence[Any]]) -> None:
        if items is None:
            return
        for item in items:
            if isinstance(item, (list, tuple)):
                TokenProjectionQwen3VLEmbedder._close_modal_items(item)
                continue
            close_fn = getattr(item, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    @staticmethod
    def _append_period(text: str) -> str:
        text = text.strip()
        if text and not unicodedata.category(text[-1]).startswith("P"):
            text += "."
        return text

    def _resolve_instruction(self, is_query: bool) -> str:
        raw = self.query_instruction if is_query else self.document_instruction
        if raw is None:
            raw = self.default_instruction
        return raw.strip()

    def _build_conversation(
        self,
        example: Dict[str, Any],
        is_query: bool,
    ) -> List[Dict[str, Any]]:
        example_instruction = normalize_text(example.get("instruction"), strip_placeholders=False)
        instruction = example_instruction or self._resolve_instruction(is_query=is_query)
        text = normalize_text(example.get("text"), strip_placeholders=True)
        image = example.get("image")

        content: List[Dict[str, Any]] = []
        if image is not None:
            image_content = ensure_file_uri(image) if isinstance(image, str) else image
            content.append({
                "type": "image",
                "image": image_content,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            })
        if text:
            content.append({"type": "text", "text": text})
        if not content:
            content = [{"type": "text", "text": "NULL"}]

        conversation: List[Dict[str, Any]] = []
        if instruction:
            conversation.append({
                "role": "system",
                "content": [{"type": "text", "text": self._append_period(instruction)}],
            })
        conversation.append({"role": "user", "content": content})
        return conversation

    def _preprocess(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except Exception as exc:
            logger.error("Error in process_vision_info; falling back to NULL text batch: %s", exc)
            images = None
            video_inputs = None
            video_kwargs = {"do_sample_frames": False}
            text = self.processor.apply_chat_template(
                [[{"role": "user", "content": [{"type": "text", "text": "NULL"}]}]],
                add_generation_prompt=True,
                tokenize=False,
            )
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None
        try:
            features = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadata,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                do_resize=False,
                return_tensors="pt",
                **video_kwargs,
            )
        finally:
            self._close_modal_items(images)
            self._close_modal_items(videos)
        device_features = {k: v.to(self.device) for k, v in features.items()}
        del features, images, videos, video_metadata
        return device_features

    def _late_keep_token_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return _late_keep_token_mask(input_ids, attention_mask, self.processor.tokenizer)

    @staticmethod
    def _decode_token(tokenizer: Any, token_id: int) -> str:
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            if token is not None:
                return str(token)
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode([int(token_id)])
        return str(int(token_id))

    def debug_token_filter_report(
        self,
        example: Dict[str, Any],
        is_query: bool,
        max_tokens: int = 160,
    ) -> Dict[str, Any]:
        conversation = self._build_conversation(example, is_query=is_query)
        model_inputs = self._preprocess([conversation])
        input_ids = model_inputs["input_ids"][0].detach().cpu()
        attention_mask = model_inputs["attention_mask"][0].detach().cpu().bool()
        keep_mask = self._late_keep_token_mask(
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
        )[0].detach().cpu().bool()
        tokenizer = self.processor.tokenizer

        def _records(mask: torch.Tensor) -> List[Dict[str, Any]]:
            positions = mask.nonzero(as_tuple=False).flatten().tolist()
            records: List[Dict[str, Any]] = []
            for pos in positions[:max_tokens]:
                token_id = int(input_ids[pos].item())
                records.append({
                    "position": int(pos),
                    "id": token_id,
                    "token": self._decode_token(tokenizer, token_id),
                })
            return records

        removed_mask = attention_mask & ~keep_mask
        return {
            "side": "query" if is_query else "candidate",
            "total_non_padding": int(attention_mask.sum().item()),
            "kept_count": int(keep_mask.sum().item()),
            "removed_count": int(removed_mask.sum().item()),
            "tokens_before_filter": _records(attention_mask),
            "tokens_after_filter": _records(keep_mask),
            "tokens_removed": _records(removed_mask),
            "truncated_to": max_tokens,
        }

    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Official Qwen3-VL-Embedding pooling: last non-pad token hidden state."""
        flipped = attention_mask.flip(dims=[1])
        last_one_positions = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    @staticmethod
    def _pad_token_dimension(tensor: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if tensor.shape[1] == target_tokens:
            return tensor
        if tensor.dim() == 3:
            pad_shape = (tensor.shape[0], target_tokens - tensor.shape[1], tensor.shape[2])
        elif tensor.dim() == 2:
            pad_shape = (tensor.shape[0], target_tokens - tensor.shape[1])
        else:
            raise ValueError(f"Unsupported tensor rank for token padding: {tensor.dim()}")
        pad = tensor.new_zeros(pad_shape)
        return torch.cat([tensor, pad], dim=1)

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def _encode_side_batch(
        self,
        examples: Sequence[Dict[str, Optional[str]]],
        is_query: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single backbone forward pass → anchor + token-projection residual.

        The backbone always runs under torch.no_grad() since it is frozen.
        Gradients only flow through self.token_projection_module.

        Returns:
            anchor       (B, anchor_dim)  L2-normalised anchor vector.
            anchor_raw   (B, H)           pooled hidden state (no grad).
            residual_raw (B, K', H)       selected token states before projection.
            residual     (B, K', D)       L2-normalised token vectors.
            residual_mask(B, K')          1 for valid late-interaction tokens.
        """
        conversations = [self._build_conversation(ex, is_query=is_query) for ex in examples]
        model_inputs = self._preprocess(conversations)
        attn_mask = model_inputs["attention_mask"]
        residual_keep_mask = self._late_keep_token_mask(model_inputs["input_ids"], attn_mask)

        need_all_layers = not self._uses_final_hidden_state_only()

        # Run backbone once, frozen, no gradient tracking.
        with torch.no_grad():
            backbone_out = self.backbone.model(
                **model_inputs,
                output_hidden_states=need_all_layers,
            )
        # Free large inputs (pixel_values etc.) immediately after forward pass.
        del model_inputs

        last_hidden = backbone_out.last_hidden_state  # (B, L, H)

        # --- Anchor ---
        anchor_raw = self._pooling_last(last_hidden, attn_mask)          # (B, H)
        anchor = F.normalize(anchor_raw[:, : self.anchor_dim], p=2, dim=-1)

        # --- Hidden states for token projection ---
        if need_all_layers:
            layer_indices = self._resolve_layer_indices(backbone_out.hidden_states)

            # hidden_states[0] is the embedding output; transformer block i
            # lives at hidden_states[i + 1].
            selected_hidden_states = [
                backbone_out.hidden_states[layer_idx + 1]
                for layer_idx in layer_indices
            ]
            hidden_for_projection = (
                selected_hidden_states[0] if len(selected_hidden_states) == 1 else selected_hidden_states
            )
        else:
            hidden_for_projection = last_hidden
        # Release backbone output — frees all layer hidden states except the
        # clones we just extracted above.
        del backbone_out, last_hidden

        # --- Token-projection residual (gradients flow here) ---
        residual_raw, residual, residual_mask = self.token_projection_module(
            hidden_for_projection,
            residual_keep_mask,
        )
        return anchor, anchor_raw, residual_raw, residual, residual_mask

    def encode_side(
        self,
        examples: Sequence[Dict[str, Optional[str]]],
        is_query: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_size = self.query_encode_batch_size if is_query else self.doc_encode_batch_size
        if chunk_size is None or chunk_size <= 0 or len(examples) <= chunk_size:
            return self._encode_side_batch(examples, is_query)

        anchor_parts: List[torch.Tensor] = []
        anchor_raw_parts: List[torch.Tensor] = []
        residual_raw_parts: List[torch.Tensor] = []
        residual_parts: List[torch.Tensor] = []
        residual_mask_parts: List[torch.Tensor] = []

        for start in range(0, len(examples), chunk_size):
            end = min(start + chunk_size, len(examples))
            anchor, anchor_raw, residual_raw, residual, residual_mask = self._encode_side_batch(
                examples[start:end],
                is_query,
            )
            anchor_parts.append(anchor)
            anchor_raw_parts.append(anchor_raw)
            residual_raw_parts.append(residual_raw)
            residual_parts.append(residual)
            residual_mask_parts.append(residual_mask)

        max_tokens = max(part.shape[1] for part in residual_parts)
        residual_raw_parts = [
            self._pad_token_dimension(part, max_tokens) for part in residual_raw_parts
        ]
        residual_parts = [
            self._pad_token_dimension(part, max_tokens) for part in residual_parts
        ]
        residual_mask_parts = [
            self._pad_token_dimension(part, max_tokens) for part in residual_mask_parts
        ]

        return (
            torch.cat(anchor_parts, dim=0),
            torch.cat(anchor_raw_parts, dim=0),
            torch.cat(residual_raw_parts, dim=0),
            torch.cat(residual_parts, dim=0),
            torch.cat(residual_mask_parts, dim=0),
        )
