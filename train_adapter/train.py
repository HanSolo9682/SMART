from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR

from src.data import (
    VISDOC_QUERY_INSTRUCTION,
    TaskAwareBatchSampler,
    VisDocDataset,
    VisDocPairCollator,
)
from src.losses import (
    anchor_overlap_regularizer,
    bank_maxsim,
    contrastive_loss,
    diversity_regularizer,
    explicit_hard_negative_metrics,
)
from src.model import MAX_PIXELS, MIN_PIXELS, TokenProjectionQwen3VLEmbedder
from src.utils import set_seed

METRIC_COUNT_INDEX = 7
METRIC_VECTOR_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train token-projection late-interaction adapters on Qwen3-VL-Embedding."
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--dataset_type", type=str, default="visdoc", choices=["visdoc"])
    parser.add_argument(
        "--train_data",
        type=str,
        default="/nobackup3/hlee_nobackup_backup/dataset/Visdoc/colpali_train_set",
        help="Local Hugging Face VisDoc/ColPali parquet dataset path.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--image_cache_dir",
        type=str,
        default=None,
        help="Optional directory where embedded VisDoc page images are materialized as PNG paths.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    # Token-projection residual module
    parser.add_argument("--use_last_n_layers", type=int, default=1,
                        help="Number of backbone layers whose hidden states are fused into layer-weighted "
                             "late-interaction multivectors. 1 = only last layer; >1 = learn per-layer "
                             "pseudo-query weights over the last N layers when "
                             "--selected_layers is not provided.")
    parser.add_argument(
        "--selected_layers",
        type=str,
        default=None,
        help="Comma-separated backbone layer indices or inclusive ranges to fuse, "
             "e.g. '0,3,7,11', '15-20', or '-1' for the last layer. "
             "Overrides --use_last_n_layers.",
    )
    parser.add_argument("--residual_dim", type=int, default=256)
    parser.add_argument("--anchor_dim", type=int, default=2048)
    parser.add_argument("--query_encode_batch_size", type=int, default=None)
    parser.add_argument("--doc_encode_batch_size", type=int, default=8)

    # Preprocessing
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)

    # Training objective
    parser.add_argument("--lambda_anchor", type=float, default=1.0)
    parser.add_argument("--lambda_residual", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--hybrid_loss_weight", type=float, default=1.0)
    parser.add_argument("--residual_loss_weight_start", type=float, default=0.0)
    parser.add_argument("--residual_loss_weight_end", type=float, default=0.0)
    parser.add_argument("--diversity_weight", type=float, default=0.0)
    parser.add_argument("--anchor_overlap_weight", type=float, default=0.0)

    # Instructions
    parser.add_argument("--default_instruction", type=str, default="Represent the user's input.")
    parser.add_argument("--query_instruction", type=str, default=None)
    parser.add_argument("--document_instruction", type=str, default=None)

    # Backbone
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"])

    # Data loading
    parser.add_argument("--num_hard_negatives", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Disabled by design for this trainer. Must be 1.",
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--persistent_workers", action="store_true", default=False)

    # Optimiser
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging / checkpointing
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--show_progress_bar", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="qwen3_vl_weight_all_hidden_experiment")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument("--trace_every", type=int, default=0)
    parser.add_argument("--slow_step_seconds", type=float, default=0.0)
    parser.add_argument("--host_memory_trim_every", type=int, default=20)
    parser.add_argument("--debug_dump_num_batches", type=int, default=0)
    parser.add_argument("--debug_dump_samples", type=int, default=3)
    parser.add_argument("--debug_dump_include_negatives", action="store_true")
    parser.add_argument("--debug_dump_dir", type=str, default=None)
    parser.add_argument(
        "--debug_print_batch",
        action="store_true",
        help="Print one collated VisDoc batch and chat-template summary before training.",
    )
    parser.add_argument(
        "--debug_print_batch_exit",
        action="store_true",
        help="Exit after printing the first debug batch.",
    )
    parser.add_argument("--debug_token_filter_max_tokens", type=int, default=160)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_int_fields = {
        "use_last_n_layers": args.use_last_n_layers,
        "residual_dim": args.residual_dim,
        "anchor_dim": args.anchor_dim,
        "max_length": args.max_length,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_epochs": args.num_epochs,
        "log_every": args.log_every,
    }
    non_negative_int_fields = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_hard_negatives": args.num_hard_negatives,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "num_workers": args.num_workers,
        "save_every_steps": args.save_every_steps,
        "trace_every": args.trace_every,
        "host_memory_trim_every": args.host_memory_trim_every,
        "debug_dump_num_batches": args.debug_dump_num_batches,
        "debug_dump_samples": args.debug_dump_samples,
        "debug_token_filter_max_tokens": args.debug_token_filter_max_tokens,
    }
    for name, value in positive_int_fields.items():
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    optional_positive_int_fields = {
        "query_encode_batch_size": args.query_encode_batch_size,
        "doc_encode_batch_size": args.doc_encode_batch_size,
    }
    for name, value in optional_positive_int_fields.items():
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1 when provided, got {value}")
    for name, value in non_negative_int_fields.items():
        if value is not None and value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if args.gradient_accumulation_steps != 1:
        raise ValueError(
            "This trainer does not support gradient accumulation. "
            "Please launch with --gradient_accumulation_steps 1."
        )
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1], got {args.warmup_ratio}")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {args.val_ratio}")
    if args.temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {args.temperature}")
    if args.hybrid_loss_weight < 0.0:
        raise ValueError(f"hybrid_loss_weight must be >= 0, got {args.hybrid_loss_weight}")
    if args.residual_loss_weight_start < 0.0:
        raise ValueError(
            f"residual_loss_weight_start must be >= 0, got {args.residual_loss_weight_start}"
        )
    if args.residual_loss_weight_end < 0.0:
        raise ValueError(
            f"residual_loss_weight_end must be >= 0, got {args.residual_loss_weight_end}"
        )
    if args.max_grad_norm <= 0.0:
        raise ValueError(f"max_grad_norm must be > 0, got {args.max_grad_norm}")
    if args.num_workers == 0 and args.persistent_workers:
        raise ValueError("persistent_workers requires num_workers > 0")
    if args.dataset_type == "visdoc" and args.num_hard_negatives != 0:
        raise ValueError(
            "VisDoc training data does not contain explicit hard negatives. "
            "Use --num_hard_negatives 0 so the trainer relies on in-batch negatives."
        )
    args.selected_layers = parse_selected_layers(args.selected_layers)


def use_wandb(args: argparse.Namespace) -> bool:
    return args.use_wandb and args.wandb_mode != "disabled"


def parse_selected_layers(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    layers: List[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if range_match is not None:
            start_idx = int(range_match.group(1))
            end_idx = int(range_match.group(2))
            if end_idx < start_idx:
                raise ValueError(
                    f"selected_layers range must be ascending, got {token!r} in {value!r}"
                )
            layers.extend(range(start_idx, end_idx + 1))
            continue
        try:
            layer_idx = int(token)
        except ValueError as exc:
            raise ValueError(
                f"selected_layers must be comma-separated integers or ranges, got {value!r}"
            ) from exc
        layers.append(layer_idx)

    if not layers:
        raise ValueError("selected_layers must contain at least one layer index")
    if len(set(layers)) != len(layers):
        raise ValueError(f"selected_layers must not contain duplicates, got {value!r}")
    return layers


def build_accelerator(args: argparse.Namespace) -> Accelerator:
    accelerator = Accelerator(log_with="wandb" if use_wandb(args) else None)
    if accelerator.state.deepspeed_plugin is not None:
        raise ValueError(
            "This trainer is DDP-only and does not support DeepSpeed. "
            "Relaunch without `--use_deepspeed`."
        )
    return accelerator


def build_dataloaders(
    args: argparse.Namespace, accelerator: Accelerator
) -> tuple[DataLoader, DataLoader, TaskAwareBatchSampler, TaskAwareBatchSampler]:
    train_dataset = VisDocDataset(
        data_path=args.train_data,
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        cache_dir=args.cache_dir,
        max_samples=args.max_train_samples,
    )
    val_dataset = VisDocDataset(
        data_path=args.train_data,
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
        cache_dir=args.cache_dir,
        max_samples=args.max_val_samples,
    )

    if accelerator.is_main_process:
        accelerator.print(
            f"[data] dataset_type={args.dataset_type} train_data={args.train_data}"
        )
        if args.image_cache_dir:
            accelerator.print(f"[data] image_cache_dir={args.image_cache_dir}")
        accelerator.print(
            f"[data] train_examples={len(train_dataset)} val_examples={len(val_dataset)} "
            f"query_instruction={VISDOC_QUERY_INSTRUCTION!r}"
        )

    train_sampler = TaskAwareBatchSampler(
        task_slices=train_dataset.task_slices,
        batch_size=args.train_batch_size,
        drop_last=not args.debug_print_batch_exit,
        seed=args.seed,
        rank=accelerator.process_index,
        num_replicas=accelerator.num_processes,
    )
    val_sampler = TaskAwareBatchSampler(
        task_slices=val_dataset.task_slices,
        batch_size=args.eval_batch_size,
        drop_last=False,
        seed=args.seed,
        rank=accelerator.process_index,
        num_replicas=accelerator.num_processes,
    )

    collator = VisDocPairCollator(
        num_hard_negatives=args.num_hard_negatives,
        image_cache_dir=args.image_cache_dir,
    )
    worker_kwargs: Dict[str, Any] = {}
    if args.num_workers > 0:
        worker_kwargs["prefetch_factor"] = 2
        worker_kwargs["persistent_workers"] = args.persistent_workers

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        **worker_kwargs,
    )
    return train_loader, val_loader, train_sampler, val_sampler


def residual_loss_weight_at_step(
    start_weight: float,
    end_weight: float,
    current_step: int,
    total_steps: int,
) -> float:
    if total_steps <= 1:
        return end_weight
    clamped_step = min(max(current_step, 0), total_steps)
    progress = float(clamped_step - 1) / float(max(1, total_steps - 1))
    return start_weight + (end_weight - start_weight) * progress


def reshape_doc_outputs(
    batch_size: int,
    num_negatives: int,
    doc_anchor: torch.Tensor,
    doc_anchor_raw: torch.Tensor,
    doc_residual_raw: torch.Tensor,
    doc_residual: torch.Tensor,
    doc_residual_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pos_anchor = doc_anchor[:batch_size]
    pos_anchor_raw = doc_anchor_raw[:batch_size]
    pos_residual = doc_residual[:batch_size]
    pos_residual_raw = doc_residual_raw[:batch_size]
    pos_residual_mask = doc_residual_mask[:batch_size]

    if num_negatives == 0:
        # No hard negatives: return properly-shaped empty tensors so downstream
        # code (explicit_hard_negative_metrics, bank construction) can
        # detect the zero-negative case via neg_anchor.shape[1] == 0.
        return {
            "pos_anchor": pos_anchor,
            "neg_anchor": doc_anchor.new_empty(batch_size, 0, doc_anchor.shape[-1]),
            "pos_anchor_raw": pos_anchor_raw,
            "neg_anchor_raw": doc_anchor_raw.new_empty(batch_size, 0, doc_anchor_raw.shape[-1]),
            "pos_residual": pos_residual,
            "neg_residual": doc_residual.new_empty(batch_size, 0, doc_residual.shape[1], doc_residual.shape[2]),
            "pos_residual_raw": pos_residual_raw,
            "neg_residual_raw": doc_residual_raw.new_empty(batch_size, 0, doc_residual_raw.shape[1], doc_residual_raw.shape[2]),
            "pos_residual_mask": pos_residual_mask,
            "neg_residual_mask": doc_residual_mask.new_empty(batch_size, 0, doc_residual_mask.shape[-1]),
        }

    neg_anchor = doc_anchor[batch_size:].reshape(batch_size, num_negatives, doc_anchor.shape[-1])
    neg_anchor_raw = doc_anchor_raw[batch_size:].reshape(batch_size, num_negatives, doc_anchor_raw.shape[-1])
    neg_residual = doc_residual[batch_size:].reshape(
        batch_size, num_negatives, doc_residual.shape[1], doc_residual.shape[2]
    )
    neg_residual_raw = doc_residual_raw[batch_size:].reshape(
        batch_size, num_negatives, doc_residual_raw.shape[1], doc_residual_raw.shape[2]
    )
    neg_residual_mask = doc_residual_mask[batch_size:].reshape(
        batch_size, num_negatives, doc_residual_mask.shape[-1]
    )
    return {
        "pos_anchor": pos_anchor,
        "neg_anchor": neg_anchor,
        "pos_anchor_raw": pos_anchor_raw,
        "neg_anchor_raw": neg_anchor_raw,
        "pos_residual": pos_residual,
        "neg_residual": neg_residual,
        "pos_residual_raw": pos_residual_raw,
        "neg_residual_raw": neg_residual_raw,
        "pos_residual_mask": pos_residual_mask,
        "neg_residual_mask": neg_residual_mask,
    }


def compute_batch(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    args: argparse.Namespace,
    residual_loss_weight: float = 0.0,
) -> Optional[Dict[str, torch.Tensor]]:
    batch_size = len(batch["queries"])
    if batch_size == 0:
        return None
    docs = list(batch["positives"]) + list(batch["flat_negatives"])
    if len(docs) != batch_size * (1 + args.num_hard_negatives):
        return None

    encoded = model(batch["queries"], docs)
    query_anchor = encoded["query_anchor"]
    query_anchor_raw = encoded["query_anchor_raw"]
    query_residual_raw = encoded["query_residual_raw"]
    query_residual = encoded["query_residual"]
    query_residual_mask = encoded["query_residual_mask"]
    doc_anchor = encoded["doc_anchor"]
    doc_anchor_raw = encoded["doc_anchor_raw"]
    doc_residual_raw = encoded["doc_residual_raw"]
    doc_residual = encoded["doc_residual"]
    doc_residual_mask = encoded["doc_residual_mask"]

    doc_parts = reshape_doc_outputs(
        batch_size=batch_size,
        num_negatives=args.num_hard_negatives,
        doc_anchor=doc_anchor,
        doc_anchor_raw=doc_anchor_raw,
        doc_residual_raw=doc_residual_raw,
        doc_residual=doc_residual,
        doc_residual_mask=doc_residual_mask,
    )

    bank_anchor = torch.cat(
        [doc_parts["pos_anchor"], doc_parts["neg_anchor"].reshape(-1, doc_parts["neg_anchor"].shape[-1])],
        dim=0,
    )
    bank_residual = torch.cat(
        [
            doc_parts["pos_residual"],
            doc_parts["neg_residual"].reshape(
                -1, doc_parts["neg_residual"].shape[-2], doc_parts["neg_residual"].shape[-1]
            ),
        ],
        dim=0,
    )
    bank_residual_mask = torch.cat(
        [
            doc_parts["pos_residual_mask"],
            doc_parts["neg_residual_mask"].reshape(-1, doc_parts["neg_residual_mask"].shape[-1]),
        ],
        dim=0,
    )
    anchor_logits = query_anchor @ bank_anchor.t()
    residual_logits = bank_maxsim(
        query_residual,
        bank_residual,
        query_mask=query_residual_mask,
        doc_mask=bank_residual_mask,
    )
    logits = (args.lambda_anchor * anchor_logits) + (args.lambda_residual * residual_logits)
    anchor_loss = contrastive_loss(anchor_logits, temperature=args.temperature)
    residual_loss = contrastive_loss(residual_logits, temperature=args.temperature)

    
    # print("anchor_scores_track: ", anchor_scores_track)
    # print("residual_scores_track: ", residual_scores_track)

    hybrid_loss = contrastive_loss(logits, temperature=args.temperature)
    loss = (args.hybrid_loss_weight * hybrid_loss) + (residual_loss_weight * residual_loss)

    if args.diversity_weight > 0.0:
        loss = loss + args.diversity_weight * (
            diversity_regularizer(query_residual_raw, token_mask=query_residual_mask)
            + diversity_regularizer(
                doc_residual_raw.reshape(-1, doc_residual_raw.shape[1], doc_residual_raw.shape[2]),
                token_mask=doc_residual_mask.reshape(-1, doc_residual_mask.shape[-1]),
            )
        )
    if args.anchor_overlap_weight > 0.0:
        loss = loss + args.anchor_overlap_weight * (
            anchor_overlap_regularizer(
                query_anchor_raw,
                query_residual_raw,
                token_mask=query_residual_mask,
            )
            + anchor_overlap_regularizer(
                doc_anchor_raw.reshape(-1, doc_anchor_raw.shape[-1]),
                doc_residual_raw.reshape(-1, doc_residual_raw.shape[1], doc_residual_raw.shape[2]),
                token_mask=doc_residual_mask.reshape(-1, doc_residual_mask.shape[-1]),
            )
        )

    with torch.no_grad():
        metrics = explicit_hard_negative_metrics(
            query_anchor=query_anchor.detach(),
            query_residuals=query_residual.detach(),
            pos_anchor=doc_parts["pos_anchor"].detach(),
            pos_residuals=doc_parts["pos_residual"].detach(),
            neg_anchor=doc_parts["neg_anchor"].detach(),
            neg_residuals=doc_parts["neg_residual"].detach(),
            query_mask=query_residual_mask,
            pos_residual_mask=doc_parts["pos_residual_mask"],
            neg_residual_mask=doc_parts["neg_residual_mask"],
            lambda_anchor=args.lambda_anchor,
            lambda_residual=args.lambda_residual,
        )
    metrics["anchor_loss_sum"] = anchor_loss.detach() * metrics["count"]
    metrics["residual_loss_sum"] = residual_loss.detach() * metrics["count"]
    metrics["loss_sum"] = loss.detach() * metrics["count"]
    metrics["loss"] = loss
    return metrics


def summarize_metrics(total: torch.Tensor) -> Dict[str, float]:
    count = max(total[METRIC_COUNT_INDEX].item(), 1e-8)
    return {
        "loss": total[0].item() / count,
        "anchor_loss": total[1].item() / count,
        "residual_loss": total[2].item() / count,
        "anchor_hn_acc": total[3].item() / count,
        "hybrid_hn_acc": total[4].item() / count,
        "anchor_margin": total[5].item() / count,
        "hybrid_margin": total[6].item() / count,
        "count": count,
    }


def empty_metric_vector(device: torch.device) -> torch.Tensor:
    return torch.zeros(METRIC_VECTOR_SIZE, device=device)


def pack_metric_vector(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([
        outputs["loss_sum"],
        outputs["anchor_loss_sum"],
        outputs["residual_loss_sum"],
        outputs["anchor_hn_acc_sum"],
        outputs["hybrid_hn_acc_sum"],
        outputs["anchor_margin_sum"],
        outputs["hybrid_margin_sum"],
        outputs["count"],
    ])


def has_metrics(total: torch.Tensor) -> bool:
    return bool(total[METRIC_COUNT_INDEX].item() > 0)


def gather_metric_sums(accelerator: Accelerator, local: torch.Tensor) -> torch.Tensor:
    gathered = accelerator.gather_for_metrics(local.unsqueeze(0))
    return gathered.sum(dim=0)


def is_valid_batch(batch: Dict[str, Any], args: argparse.Namespace) -> bool:
    query_count = len(batch.get("queries", []))
    if query_count == 0:
        return False
    expected_docs = query_count * (1 + args.num_hard_negatives)
    actual_docs = len(batch.get("positives", [])) + len(batch.get("flat_negatives", []))
    return actual_docs == expected_docs


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def sanitize_path_component(value: Any, fallback: str = "unknown") -> str:
    text = str(value).strip()
    if not text:
        return fallback
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return cleaned[:80] or fallback


def dump_modal_input(modal_input: Dict[str, Any], output_dir: Path, prefix: str) -> Dict[str, Any]:
    image_path = modal_input.get("image")
    record: Dict[str, Any] = {
        "text": modal_input.get("text"),
        "instruction": modal_input.get("instruction"),
        "raw_image_path": modal_input.get("raw_image_path"),
        "image": image_path if isinstance(image_path, str) else str(type(image_path).__name__),
        "saved_image": None,
    }
    if not image_path:
        return record
    if not isinstance(image_path, str) and callable(getattr(image_path, "save", None)):
        dest_name = f"{prefix}.png"
        image_path.save(output_dir / dest_name)
        record["saved_image"] = dest_name
        return record
    image_str = str(image_path)
    if image_str.startswith(("http://", "https://")):
        return record
    if image_str.startswith("file://"):
        image_str = image_str[len("file://"):]
    source = Path(image_str)
    if not source.exists():
        return record
    suffix = source.suffix or ".img"
    dest_name = f"{prefix}{suffix}"
    shutil.copy2(source, output_dir / dest_name)
    record["saved_image"] = dest_name
    return record


def maybe_dump_batch_examples(
    args: argparse.Namespace,
    batch: Dict[str, Any],
    epoch: int,
    global_step: int,
    process_index: int,
    dumped_batches: int,
) -> int:
    if args.debug_dump_num_batches <= 0 or dumped_batches >= args.debug_dump_num_batches:
        return dumped_batches
    queries = batch.get("queries", [])
    positives = batch.get("positives", [])
    if not queries or not positives:
        return dumped_batches
    sample_count = min(args.debug_dump_samples, len(queries), len(positives))
    if sample_count <= 0:
        return dumped_batches
    dump_root = Path(args.debug_dump_dir) if args.debug_dump_dir else Path(args.output_dir) / "debug_batches"
    task_name = sanitize_path_component(batch.get("task"), fallback="unknown_task")
    batch_dir = dump_root / f"epoch_{epoch + 1:03d}_step_{global_step:07d}_rank_{process_index}_{task_name}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = batch.get("sample_indices", [])
    manifest: Dict[str, Any] = {
        "epoch": epoch + 1,
        "global_step": global_step,
        "process_index": process_index,
        "task": batch.get("task"),
        "num_query_examples": len(queries),
        "num_positives": len(positives),
        "num_negatives": len(batch.get("flat_negatives", [])),
        "saved_samples": sample_count,
        "samples": [],
    }
    for sample_idx in range(sample_count):
        sample_dir = batch_dir / f"sample_{sample_idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_record: Dict[str, Any] = {
            "sample_position": sample_idx,
            "sample_index": sample_indices[sample_idx] if sample_idx < len(sample_indices) else None,
            "metadata": batch.get("metadata", [{}])[sample_idx] if sample_idx < len(batch.get("metadata", [])) else {},
            "query": dump_modal_input(queries[sample_idx], sample_dir, "query"),
            "positive": dump_modal_input(positives[sample_idx], sample_dir, "positive"),
            "negatives": [],
        }
        if args.debug_dump_include_negatives:
            negatives = batch.get("negatives", [])
            sample_negs = negatives[sample_idx] if sample_idx < len(negatives) else []
            for neg_idx, neg in enumerate(sample_negs):
                sample_record["negatives"].append(dump_modal_input(neg, sample_dir, f"negative_{neg_idx:02d}"))
        with (sample_dir / "sample.json").open("w", encoding="utf-8") as fp:
            json.dump(sample_record, fp, ensure_ascii=False, indent=2)
        manifest["samples"].append(sample_record)
    with (batch_dir / "batch_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    return dumped_batches + 1


def trim_host_memory() -> None:
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        trim = getattr(libc, "malloc_trim", None)
        if trim is not None:
            trim(0)
    except OSError:
        pass
    # Release PyTorch's CUDA allocator cache back to the driver so that the
    # cache does not grow monotonically as sequence lengths vary across batches.
    torch.cuda.empty_cache()


def get_rss_gb() -> Optional[float]:
    status_path = Path("/proc/self/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) / (1024.0 * 1024.0)
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def maybe_trace_step(
    args: argparse.Namespace,
    output_dir: str,
    process_index: int,
    global_step: int,
    batch: Dict[str, Any],
    step_time: float,
) -> None:
    should_trace = args.trace_every > 0 and global_step % args.trace_every == 0
    is_slow = args.slow_step_seconds > 0.0 and step_time >= args.slow_step_seconds
    if not (should_trace or is_slow):
        return
    record = {
        "process_index": process_index,
        "global_step": global_step,
        "task": batch.get("task"),
        "sample_indices": batch.get("sample_indices", []),
        "num_query_examples": len(batch.get("queries", [])),
        "num_positives": len(batch.get("positives", [])),
        "num_negatives": len(batch.get("flat_negatives", [])),
        "step_time_sec": round(step_time, 4),
        "rss_gb": get_rss_gb(),
    }
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        record.update(
            {
                "cuda_device": device_index,
                "cuda_allocated_gb": round(torch.cuda.memory_allocated(device_index) / (1024.0 ** 3), 4),
                "cuda_reserved_gb": round(torch.cuda.memory_reserved(device_index) / (1024.0 ** 3), 4),
                "cuda_max_allocated_gb": round(torch.cuda.max_memory_allocated(device_index) / (1024.0 ** 3), 4),
                "cuda_max_reserved_gb": round(torch.cuda.max_memory_reserved(device_index) / (1024.0 ** 3), 4),
            }
        )
    trace_path = Path(output_dir) / f"rank_{process_index}_trace.jsonl"
    append_jsonl(trace_path, record)


def sync_any_invalid(accelerator: Accelerator, local_invalid: bool) -> bool:
    flag = torch.tensor([1 if local_invalid else 0], device=accelerator.device, dtype=torch.int32)
    gathered = accelerator.gather_for_metrics(flag)
    return bool(gathered.max().item() > 0)


def flush_train_metrics(
    accelerator: Accelerator,
    args: argparse.Namespace,
    metrics_path: Path,
    running: torch.Tensor,
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    progress_bar: Optional[tqdm] = None,
) -> None:
    if not has_metrics(running):
        return
    if accelerator.is_main_process:
        summary = summarize_metrics(running)
        log_record = {
            "split": "train",
            "epoch": epoch,
            "global_step": global_step,
            **summary,
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_jsonl(metrics_path, log_record)
        accelerator.print(
            f"[train] epoch={epoch} step={global_step} loss={summary['loss']:.4f} "
            f"anchor_loss={summary['anchor_loss']:.4f} residual_loss={summary['residual_loss']:.4f} "
            f"anchor_hn_acc={summary['anchor_hn_acc']:.4f} hybrid_hn_acc={summary['hybrid_hn_acc']:.4f} "
            f"anchor_margin={summary['anchor_margin']:.4f} hybrid_margin={summary['hybrid_margin']:.4f} "
            f"count={summary['count']:.1f}"
        )
        if use_wandb(args):
            accelerator.log(
                {
                    "train/loss": summary["loss"],
                    "train/anchor_loss": summary["anchor_loss"],
                    "train/residual_loss": summary["residual_loss"],
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=global_step,
            )
        if progress_bar is not None:
            progress_bar.set_postfix(loss=f"{summary['loss']:.4f}")
    running.zero_()


def save_adapter_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    output_dir: str,
    label: str,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_adapter(output_dir)
        accelerator.print(f"[save] {label} -> {output_dir}")
    accelerator.wait_for_everyone()


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    accelerator: Accelerator,
    args: argparse.Namespace,
    residual_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    running = empty_metric_vector(accelerator.device)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            if sync_any_invalid(accelerator, not is_valid_batch(batch, args)):
                continue
            with accelerator.autocast():
                outputs = compute_batch(model, batch, args, residual_loss_weight=residual_loss_weight)
            if sync_any_invalid(accelerator, outputs is None):
                continue
            running += gather_metric_sums(accelerator, pack_metric_vector(outputs))
            if args.host_memory_trim_every > 0 and batch_idx % args.host_memory_trim_every == 0:
                trim_host_memory()
    model.train()
    return summarize_metrics(running)


def describe_image_for_debug(image: Any) -> Optional[Dict[str, Any]]:
    if image is None:
        return None
    if isinstance(image, str):
        local_path = image[len("file://"):] if image.startswith("file://") else image
        return {
            "type": "path",
            "value": image,
            "exists": Path(local_path).exists(),
        }
    return {
        "type": type(image).__name__,
        "size": list(getattr(image, "size", [])) or None,
        "mode": getattr(image, "mode", None),
    }


def describe_modal_for_debug(modal_input: Dict[str, Any]) -> Dict[str, Any]:
    text = modal_input.get("text")
    return {
        "text": None if text is None else str(text)[:240],
        "instruction": modal_input.get("instruction"),
        "raw_image_path": modal_input.get("raw_image_path"),
        "image": describe_image_for_debug(modal_input.get("image")),
    }


def summarize_conversation_for_debug(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for turn in conversation:
        content_summary: List[Dict[str, Any]] = []
        for item in turn.get("content", []):
            item_type = item.get("type")
            record: Dict[str, Any] = {"type": item_type}
            if item_type == "text":
                record["text"] = str(item.get("text", ""))[:240]
            elif item_type == "image":
                record["image"] = describe_image_for_debug(item.get("image"))
                record["min_pixels"] = item.get("min_pixels")
                record["max_pixels"] = item.get("max_pixels")
            else:
                record["value"] = str(item)[:240]
            content_summary.append(record)
        summary.append({"role": turn.get("role"), "content": content_summary})
    return summary


def print_debug_batch(
    accelerator: Accelerator,
    args: argparse.Namespace,
    model: torch.nn.Module,
    batch: Dict[str, Any],
) -> None:
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    sample_count = min(args.debug_dump_samples, len(batch.get("queries", [])), len(batch.get("positives", [])))
    samples: List[Dict[str, Any]] = []
    for sample_idx in range(sample_count):
        query = batch["queries"][sample_idx]
        positive = batch["positives"][sample_idx]
        samples.append({
            "sample_position": sample_idx,
            "sample_index": batch.get("sample_indices", [None])[sample_idx],
            "metadata": batch.get("metadata", [{}])[sample_idx],
            "query_input": describe_modal_for_debug(query),
            "candidate_input": describe_modal_for_debug(positive),
            "query_conversation": summarize_conversation_for_debug(
                unwrapped._build_conversation(query, is_query=True)
            ),
            "candidate_conversation": summarize_conversation_for_debug(
                unwrapped._build_conversation(positive, is_query=False)
            ),
            "query_token_filter": unwrapped.debug_token_filter_report(
                query,
                is_query=True,
                max_tokens=args.debug_token_filter_max_tokens,
            ),
            "candidate_token_filter": unwrapped.debug_token_filter_report(
                positive,
                is_query=False,
                max_tokens=args.debug_token_filter_max_tokens,
            ),
        })
    record = {
        "dataset_type": args.dataset_type,
        "train_data": args.train_data,
        "image_cache_dir": args.image_cache_dir,
        "task": batch.get("task"),
        "num_queries": len(batch.get("queries", [])),
        "num_positives": len(batch.get("positives", [])),
        "num_flat_negatives": len(batch.get("flat_negatives", [])),
        "num_hard_negatives": args.num_hard_negatives,
        "visdoc_eval_alignment": {
            "query": "text plus instruction",
            "query_instruction": VISDOC_QUERY_INSTRUCTION,
            "candidate": "document page image only",
            "hard_negatives": "none; contrastive loss uses in-batch negatives",
        },
        "samples": samples,
    }
    accelerator.print("[debug_batch] first collated VisDoc batch:")
    accelerator.print(json.dumps(record, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.cache_dir is not None and not args.cache_dir.strip():
        args.cache_dir = None

    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = build_accelerator(args)
    set_seed(args.seed)

    if accelerator.is_main_process and not Path(args.train_data).exists():
        print(f"[warn] train_data does not exist locally: {args.train_data}")
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as fp:
            json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    if use_wandb(args):
        run_name = args.wandb_run_name or Path(args.output_dir).name
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "name": run_name,
                    "entity": args.wandb_entity,
                    "mode": args.wandb_mode,
                }
            },
        )

    train_loader, val_loader, train_sampler, _ = build_dataloaders(args, accelerator)
    if len(train_loader) == 0:
        raise ValueError("Training dataloader is empty. Check dataset filters and batch size.")
    has_val_loader = len(val_loader) > 0

    model = TokenProjectionQwen3VLEmbedder(
        model_name_or_path=args.model_name_or_path,
        residual_dim=args.residual_dim,
        anchor_dim=args.anchor_dim,
        use_last_n_layers=args.use_last_n_layers,
        selected_layers=args.selected_layers,
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        default_instruction=args.default_instruction,
        query_instruction=args.query_instruction,
        document_instruction=args.document_instruction,
        query_encode_batch_size=args.query_encode_batch_size,
        doc_encode_batch_size=args.doc_encode_batch_size,
        attn_implementation=args.attn_implementation,
    )

    if args.debug_print_batch:
        debug_batch = next(iter(train_loader))
        print_debug_batch(accelerator, args, model, debug_batch)
        del debug_batch
        if args.debug_print_batch_exit:
            accelerator.print("[debug_batch] exiting before optimizer/training as requested.")
            if use_wandb(args):
                accelerator.end_training()
            return

    optimizer = torch.optim.AdamW(
        model.trainable_parameter_groups(weight_decay=args.weight_decay),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = max(len(train_loader) * args.num_epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = min(1.0, float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, _lr_lambda)

    model, optimizer = accelerator.prepare(model, optimizer)

    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    best_val = float("-inf")
    global_step = 0
    dumped_batches = 0

    if accelerator.is_main_process:
        accelerator.print(
            "[config] "
            f"processes={accelerator.num_processes} "
            f"per_device_train_batch_size={args.train_batch_size} "
            f"global_train_batch_size={args.train_batch_size * accelerator.num_processes}"
        )
        if not has_val_loader:
            accelerator.print("[warn] Validation dataloader is empty; skipping validation.")

    for epoch in range(args.num_epochs):
        epoch_number = epoch + 1
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_running = empty_metric_vector(accelerator.device)
        progress_bar: Optional[tqdm] = None
        train_iterator = train_loader
        if accelerator.is_main_process and args.show_progress_bar:
            progress_bar = tqdm(
                train_loader,
                desc=f"train epoch {epoch_number}/{args.num_epochs}",
                dynamic_ncols=True,
                leave=True,
            )
            train_iterator = progress_bar

        for batch in train_iterator:
            step_start = time.monotonic()
            if sync_any_invalid(accelerator, not is_valid_batch(batch, args)):
                continue

            next_step = global_step + 1
            if accelerator.is_main_process:
                dumped_before = dumped_batches
                dumped_batches = maybe_dump_batch_examples(
                    args=args,
                    batch=batch,
                    epoch=epoch,
                    global_step=next_step,
                    process_index=accelerator.process_index,
                    dumped_batches=dumped_batches,
                )
                if dumped_batches > dumped_before:
                    dump_root = (
                        Path(args.debug_dump_dir) if args.debug_dump_dir
                        else Path(args.output_dir) / "debug_batches"
                    )
                    accelerator.print(f"[debug] dumped batch examples to {dump_root}")

            optimizer.zero_grad(set_to_none=True)
            residual_loss_weight = residual_loss_weight_at_step(
                start_weight=args.residual_loss_weight_start,
                end_weight=args.residual_loss_weight_end,
                current_step=next_step,
                total_steps=total_steps,
            )
            with accelerator.autocast():
                outputs = compute_batch(model, batch, args, residual_loss_weight=residual_loss_weight)
            if sync_any_invalid(accelerator, outputs is None):
                continue
            loss = outputs["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at global_step={next_step}: {loss.detach().item()}")

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step = next_step

            epoch_running += gather_metric_sums(accelerator, pack_metric_vector(outputs))
            step_time = time.monotonic() - step_start
            maybe_trace_step(
                args=args,
                output_dir=args.output_dir,
                process_index=accelerator.process_index,
                global_step=global_step,
                batch=batch,
                step_time=step_time,
            )

            del loss, outputs, batch

            if args.host_memory_trim_every > 0 and global_step % args.host_memory_trim_every == 0:
                trim_host_memory()

            if global_step % args.log_every == 0:
                flush_train_metrics(
                    accelerator=accelerator,
                    args=args,
                    metrics_path=metrics_path,
                    running=epoch_running,
                    epoch=epoch_number,
                    global_step=global_step,
                    optimizer=optimizer,
                    progress_bar=progress_bar,
                )

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                step_dir = os.path.join(args.output_dir, f"step_{global_step}")
                save_adapter_checkpoint(accelerator, model, step_dir, f"step {global_step}")

        flush_train_metrics(
            accelerator=accelerator,
            args=args,
            metrics_path=metrics_path,
            running=epoch_running,
            epoch=epoch_number,
            global_step=global_step,
            optimizer=optimizer,
            progress_bar=progress_bar,
        )

        if has_val_loader:
            val_residual_loss_weight = residual_loss_weight_at_step(
                start_weight=args.residual_loss_weight_start,
                end_weight=args.residual_loss_weight_end,
                current_step=global_step,
                total_steps=total_steps,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                accelerator,
                args,
                residual_loss_weight=val_residual_loss_weight,
            )
            if accelerator.is_main_process:
                val_record = {
                    "split": "val",
                    "epoch": epoch_number,
                    "global_step": global_step,
                    **val_metrics,
                }
                append_jsonl(metrics_path, val_record)
                accelerator.print(
                    f"[val] epoch={epoch_number} loss={val_metrics['loss']:.4f}"
                )
                if use_wandb(args):
                    accelerator.log(
                        {
                            "val/loss": val_metrics["loss"],
                            "epoch": epoch_number,
                        },
                        step=global_step,
                    )
            if val_metrics["hybrid_hn_acc"] > best_val:
                best_val = val_metrics["hybrid_hn_acc"]
                best_dir = os.path.join(args.output_dir, "best_adapter")
                save_adapter_checkpoint(accelerator, model, best_dir, "best adapter")

        if args.save_every_epoch:
            epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch_number}")
            save_adapter_checkpoint(accelerator, model, epoch_dir, f"epoch {epoch_number}")

    accelerator.wait_for_everyone()
    if use_wandb(args):
        accelerator.end_training()


if __name__ == "__main__":
    main()
