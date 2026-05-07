from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _as_bool_mask(mask: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return mask.to(device=ref.device, dtype=torch.bool)


def _downsample_tokens(
    states: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    max_tokens: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if states.shape[1] <= max_tokens:
        return states, token_mask
    indices = torch.linspace(
        0,
        states.shape[1] - 1,
        steps=max_tokens,
        device=states.device,
    ).round().long()
    states = states.index_select(1, indices)
    if token_mask is not None:
        token_mask = token_mask.index_select(1, indices)
    return states, token_mask


def _resolve_bank_chunk_sizes(
    batch_size: int,
    num_docs: int,
    query_tokens: int,
    doc_tokens: int,
    target_chunk_elements: int = 8_000_000,
    min_query_chunk: int = 16,
    max_doc_chunk: int = 8,
) -> tuple[int, int]:
    doc_chunk_size = target_chunk_elements // max(1, batch_size * max(1, doc_tokens) * min_query_chunk)
    doc_chunk_size = max(1, min(num_docs, doc_chunk_size, max_doc_chunk))
    query_chunk_size = target_chunk_elements // max(1, batch_size * doc_chunk_size * max(1, doc_tokens))
    query_chunk_size = max(1, min(query_tokens, query_chunk_size))
    return doc_chunk_size, query_chunk_size


def _resolve_pair_query_chunk_size(
    batch_size: int,
    query_tokens: int,
    doc_tokens: int,
    target_chunk_elements: int = 8_000_000,
) -> int:
    return max(1, min(query_tokens, target_chunk_elements // max(1, batch_size * max(1, doc_tokens))))


def bank_maxsim(
    query_residuals: torch.Tensor,
    doc_residuals: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    doc_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Late interaction scores.

    query_residuals: [B, Kq, D]
    doc_residuals: [N, Kd, D]
    returns: [B, N]
    """
    batch_size, query_tokens, _ = query_residuals.shape
    num_docs, doc_tokens, _ = doc_residuals.shape

    query_mask = _as_bool_mask(query_mask, query_residuals)
    doc_mask = _as_bool_mask(doc_mask, doc_residuals)
    doc_chunk_size, query_chunk_size = _resolve_bank_chunk_sizes(
        batch_size=batch_size,
        num_docs=num_docs,
        query_tokens=query_tokens,
        doc_tokens=doc_tokens,
    )

    if query_mask is None:
        query_counts = query_residuals.new_full((batch_size, 1), float(query_tokens))
    else:
        query_counts = query_mask.sum(dim=-1, keepdim=True).to(query_residuals.dtype).clamp_min(1.0)

    scores = query_residuals.new_empty(batch_size, num_docs)
    # Avoid torch.finfo(bf16).min: the Python double round-trip trips masked_fill's
    # overflow check on bf16. -1e4 is far more negative than any realistic sim.
    neg_fill = -1e4

    for doc_start in range(0, num_docs, doc_chunk_size):
        doc_end = min(doc_start + doc_chunk_size, num_docs)
        doc_chunk = doc_residuals[doc_start:doc_end]
        doc_chunk_t = doc_chunk.transpose(1, 2).contiguous()
        doc_chunk_mask = None if doc_mask is None else doc_mask[doc_start:doc_end]
        chunk_sum = query_residuals.new_zeros(batch_size, doc_end - doc_start)

        for query_start in range(0, query_tokens, query_chunk_size):
            query_end = min(query_start + query_chunk_size, query_tokens)
            query_chunk = query_residuals[:, query_start:query_end]
            sim = torch.einsum("bqd,cdk->bcqk", query_chunk, doc_chunk_t)
            if doc_chunk_mask is not None:
                sim = sim.masked_fill(~doc_chunk_mask.unsqueeze(0).unsqueeze(2), neg_fill)
            chunk_max = sim.max(dim=-1).values
            if query_mask is None:
                chunk_sum += chunk_max.sum(dim=-1)
            else:
                query_chunk_mask = query_mask[:, query_start:query_end].to(chunk_max.dtype)
                chunk_sum += (chunk_max * query_chunk_mask.unsqueeze(1)).sum(dim=-1)

        chunk_scores = chunk_sum / query_counts
        if doc_chunk_mask is not None:
            doc_valid = doc_chunk_mask.any(dim=-1)
            chunk_scores = chunk_scores.masked_fill(~doc_valid.unsqueeze(0), 0.0)
        scores[:, doc_start:doc_end] = chunk_scores

    return scores



def pair_maxsim(
    query_residuals: torch.Tensor,
    doc_residuals: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    doc_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pairwise late interaction for aligned batches.

    query_residuals: [B, K, D]
    doc_residuals: [B, K, D] or [B, H, K, D]
    """
    query_mask = _as_bool_mask(query_mask, query_residuals)

    if doc_residuals.ndim == 3:
        batch_size, query_tokens, _ = query_residuals.shape
        doc_tokens = doc_residuals.shape[1]
        query_chunk_size = _resolve_pair_query_chunk_size(
            batch_size=batch_size,
            query_tokens=query_tokens,
            doc_tokens=doc_tokens,
        )
        doc_mask = _as_bool_mask(doc_mask, doc_residuals)
        if query_mask is None:
            query_counts = query_residuals.new_full((batch_size,), float(query_tokens))
        else:
            query_counts = query_mask.sum(dim=-1).to(query_residuals.dtype).clamp_min(1.0)
        score_sum = query_residuals.new_zeros(batch_size)
        doc_t = doc_residuals.transpose(1, 2).contiguous()
        neg_fill = -1e4

        for query_start in range(0, query_tokens, query_chunk_size):
            query_end = min(query_start + query_chunk_size, query_tokens)
            query_chunk = query_residuals[:, query_start:query_end]
            sim = torch.matmul(query_chunk, doc_t)
            if doc_mask is not None:
                sim = sim.masked_fill(~doc_mask.unsqueeze(1), neg_fill)
            chunk_max = sim.max(dim=-1).values
            if query_mask is None:
                score_sum += chunk_max.sum(dim=-1)
            else:
                query_chunk_mask = query_mask[:, query_start:query_end].to(chunk_max.dtype)
                score_sum += (chunk_max * query_chunk_mask).sum(dim=-1)

        scores = score_sum / query_counts
        if doc_mask is not None:
            scores = scores.masked_fill(~doc_mask.any(dim=-1), 0.0)
        return scores
    if doc_residuals.ndim == 4:
        doc_mask = _as_bool_mask(doc_mask, doc_residuals)
        parts = []
        for neg_idx in range(doc_residuals.shape[1]):
            current_doc_mask = None if doc_mask is None else doc_mask[:, neg_idx]
            parts.append(
                pair_maxsim(
                    query_residuals=query_residuals,
                    doc_residuals=doc_residuals[:, neg_idx],
                    query_mask=query_mask,
                    doc_mask=current_doc_mask,
                )
            )
        return torch.stack(parts, dim=1)
    raise ValueError(f"Unsupported doc_residuals rank: {doc_residuals.ndim}")



def build_contrastive_logits(
    query_anchor: torch.Tensor,
    query_residuals: torch.Tensor,
    pos_anchor: torch.Tensor,
    pos_residuals: torch.Tensor,
    neg_anchor: torch.Tensor,
    neg_residuals: torch.Tensor,
    query_mask: Optional[torch.Tensor],
    pos_residual_mask: Optional[torch.Tensor],
    neg_residual_mask: Optional[torch.Tensor],
    lambda_anchor: float,
    lambda_residual: float,
) -> torch.Tensor:
    batch_size = query_anchor.shape[0]
    bank_anchor = torch.cat([pos_anchor, neg_anchor.reshape(-1, neg_anchor.shape[-1])], dim=0)
    bank_residuals = torch.cat(
        [pos_residuals, neg_residuals.reshape(-1, neg_residuals.shape[-2], neg_residuals.shape[-1])],
        dim=0,
    )
    bank_residual_mask = None
    if pos_residual_mask is not None and neg_residual_mask is not None:
        bank_residual_mask = torch.cat(
            [pos_residual_mask, neg_residual_mask.reshape(-1, neg_residual_mask.shape[-1])],
            dim=0,
        )
    anchor_scores = query_anchor @ bank_anchor.t()
    residual_scores = bank_maxsim(
        query_residuals,
        bank_residuals,
        query_mask=query_mask,
        doc_mask=bank_residual_mask,
    )
    logits = (lambda_anchor * anchor_scores) + (lambda_residual * residual_scores)

    assert logits.shape[0] == batch_size
    return logits



def contrastive_loss(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits / temperature, labels)



def explicit_hard_negative_metrics(
    query_anchor: torch.Tensor,
    query_residuals: torch.Tensor,
    pos_anchor: torch.Tensor,
    pos_residuals: torch.Tensor,
    neg_anchor: torch.Tensor,
    neg_residuals: torch.Tensor,
    query_mask: Optional[torch.Tensor],
    pos_residual_mask: Optional[torch.Tensor],
    neg_residual_mask: Optional[torch.Tensor],
    lambda_anchor: float,
    lambda_residual: float,
) -> Dict[str, torch.Tensor]:
    zero = query_anchor.new_tensor(0.0)
    count = torch.tensor(float(query_anchor.shape[0]), device=query_anchor.device)

    # When there are no hard negatives the HN metrics are undefined; return zeros.
    if neg_anchor.shape[1] == 0:
        return {
            "anchor_hn_acc_sum": zero,
            "hybrid_hn_acc_sum": zero,
            "anchor_margin_sum": zero,
            "hybrid_margin_sum": zero,
            "count": count,
        }

    pos_anchor_score = (query_anchor * pos_anchor).sum(dim=-1)
    neg_anchor_score = torch.einsum("bd,bhd->bh", query_anchor, neg_anchor)
    pos_residual_score = pair_maxsim(
        query_residuals,
        pos_residuals,
        query_mask=query_mask,
        doc_mask=pos_residual_mask,
    )
    neg_residual_score = pair_maxsim(
        query_residuals,
        neg_residuals,
        query_mask=query_mask,
        doc_mask=neg_residual_mask,
    )

    anchor_margin = pos_anchor_score - neg_anchor_score.max(dim=-1).values
    hybrid_margin = (
        lambda_anchor * pos_anchor_score
        + lambda_residual * pos_residual_score
        - (lambda_anchor * neg_anchor_score + lambda_residual * neg_residual_score).max(dim=-1).values
    )

    anchor_correct = anchor_margin.gt(0.0)
    hybrid_correct = hybrid_margin.gt(0.0)

    return {
        "anchor_hn_acc_sum": anchor_correct.float().sum(),
        "hybrid_hn_acc_sum": hybrid_correct.float().sum(),
        "anchor_margin_sum": anchor_margin.sum(),
        "hybrid_margin_sum": hybrid_margin.sum(),
        "count": count,
    }



def diversity_regularizer(
    residual_raw_states: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    max_tokens: int = 128,
) -> torch.Tensor:
    """Discourage residual tokens from collapsing onto each other within a side."""
    residual_raw_states, token_mask = _downsample_tokens(residual_raw_states, token_mask, max_tokens=max_tokens)
    if residual_raw_states.shape[1] <= 1:
        return residual_raw_states.new_tensor(0.0)
    states = F.normalize(residual_raw_states, p=2, dim=-1)
    sim = torch.einsum("bkd,bjd->bkj", states, states)
    valid_pairs = ~torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool).unsqueeze(0)
    if token_mask is not None:
        token_mask = token_mask.to(device=sim.device, dtype=torch.bool)
        valid_pairs = valid_pairs & token_mask.unsqueeze(1) & token_mask.unsqueeze(2)
    if not valid_pairs.any().item():
        return sim.new_tensor(0.0)
    off_diag = sim.masked_select(valid_pairs)
    return off_diag.pow(2).mean()



def anchor_overlap_regularizer(
    anchor_raw: torch.Tensor,
    residual_raw_states: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    anchor = F.normalize(anchor_raw, p=2, dim=-1).unsqueeze(1)
    residual = F.normalize(residual_raw_states, p=2, dim=-1)
    overlap = (anchor * residual).sum(dim=-1)
    if token_mask is None:
        return overlap.pow(2).mean()
    token_mask = token_mask.to(device=overlap.device, dtype=overlap.dtype)
    denom = token_mask.sum().clamp_min(1.0)
    return (overlap.pow(2) * token_mask).sum() / denom
