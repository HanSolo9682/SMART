import os
import time
import yaml
import torch
import random
import pickle
import json
import hashlib
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

from dataclasses import dataclass, field
from datetime import timedelta
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from datasets import concatenate_datasets
from datasets.distributed import split_dataset_by_node

from .arguments import ModelArguments, DataArguments, EvalArguments
from .utils.basic_utils import print_rank, print_master
from .utils.eval_utils.metrics import RankingMetrics
from src.models.gme_qwen2_vl import GmeQwen2VL
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


def _late_keep_token_mask_text_only(input_ids, attention_mask, tokenizer):
    keep_mask = attention_mask.bool()

    # Gather all default special tokens from the tokenizer
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    
    # Add control tokens AND visual placeholders to the blacklist
    special_ids.update(
        _token_ids(
            tokenizer,
            [
                "<|endoftext|>",
                "<|im_start|>",
                "<|im_end|>",
                "<|vision_start|>",
                "<|vision_end|>",
                "<|image_pad|>",  # Now explicitly marked for masking
                "<|video_pad|>",  # Now explicitly marked for masking
            ],
        )
    )

    # Note: We completely removed the `special_ids.difference_update(...)` line.
    # By doing this, the visual tokens remain in the special_ids set.

    if special_ids:
        special_tensor = torch.tensor(sorted(special_ids), device=input_ids.device)
        # Checks if each token in input_ids matches any ID in the special_tensor
        is_special = (input_ids.unsqueeze(-1) == special_tensor).any(dim=-1)
        
        # Flips the boolean (so special/visual tokens become False) and applies it to the mask
        keep_mask = keep_mask & ~is_special

    return keep_mask


def _format_batch_inputs(model: GmeQwen2VL, batch_inputs):
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


@torch.no_grad()
def encode_anchor_and_tokens(
    model,
    batch_inputs,
    include_tokens: bool,
    encode_side: str,
):
    texts = [ele.get("text") for ele in batch_inputs]
    images = []
    instructions = [ele.get("instruction") for ele in batch_inputs]
    
    for ele in batch_inputs:
        img_data = ele.get("image")
        vid_data = ele.get("video")
        
        # 1. Handle Images (Documents / Visuals)
        if img_data is not None:
            if isinstance(img_data, list) and len(img_data) > 0:
                # If a multi-page document is passed as a list, take the first page
                images.append(img_data[0])
            else:
                # It's already a single image
                images.append(img_data)
                
        # 2. Handle Videos (extract middle frame as per VLM2Vec paper)
        elif vid_data is not None:
            if isinstance(vid_data, list) and len(vid_data) > 0:
                middle_idx = len(vid_data) // 2
                images.append(vid_data[middle_idx])
            else:
                images.append(vid_data) # Fallback if it's somehow a single file string
                
        # 3. Text only
        else:
            images.append(None)

    is_query = (encode_side == "qry")

    # Delegate natively to the GME code (now guaranteed to receive max 1 image per item)
    embeddings, hidden_states, input_ids, attention_mask = model.embed(
        texts=texts, 
        images=images, 
        is_query=is_query,
        instruction=instructions[0] if instructions else None,
    )
    anchors = embeddings.detach()

    if not include_tokens:
        return anchors, None

    if encode_side == "qry":
        keep_mask = _late_keep_token_mask_text_only(input_ids, attention_mask, model.processor.tokenizer)
    else:
        # Retrieve tokenizer for dynamic masking 
        keep_mask = _late_keep_token_mask(input_ids, attention_mask, model.processor.tokenizer)

    normalized_hidden = F.normalize(hidden_states.float(), p=2, dim=-1)
        
    token_states = []
    for row_idx in range(normalized_hidden.shape[0]):
        row_tokens = normalized_hidden[row_idx][keep_mask[row_idx]]
        if row_tokens.numel() == 0:
            row_tokens = F.normalize(anchors[row_idx].float(), p=2, dim=-1).unsqueeze(0)
        token_states.append(row_tokens.cpu().numpy().astype(np.float32))

    return anchors, token_states


import gc

@torch.no_grad()
def encode_representations(
    model, # Assumed GmeQwen2VL
    loader, # Assumed DataLoader
    encode_side: str,
    full_dataset_len: int,
    include_tokens: bool,
    description: str = "Encoding",
    object_group=None,
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    local_anchors = []
    local_infos = []
    
    # FIX: Initialize a list of lists on Rank 0 to group tokens by rank
    final_tokens_per_rank = [[] for _ in range(world_size)] if (rank == 0 and include_tokens) else None
    
    # Fallback for non-distributed (single GPU)
    local_token_states = [] if (not dist.is_initialized() and include_tokens) else None

    progress_bar = tqdm(
        loader,
        desc=f"{description} (rank {rank})",
        disable=local_rank > 0,
        ncols=120,
    )

    # batch by batch encoding
    for batch_inputs, dataset_info in progress_bar:
        anchors, token_states = encode_anchor_and_tokens(
            model=model,
            batch_inputs=batch_inputs,
            include_tokens=include_tokens,
            encode_side=encode_side,
        )
        anchors = anchors.detach()

        local_anchors.append(anchors)
        
        if encode_side == "qry":
            local_infos.extend(dataset_info)
        else:
            local_infos.extend([info.get("cand_name", "") for info in dataset_info])

        # --- MEMORY EFFICIENT PER-BATCH GATHER ---
        if include_tokens:
            if dist.is_initialized():
                gathered_batch = [None for _ in range(world_size)] if rank == 0 else None
                
                dist.gather_object(
                    token_states,
                    object_gather_list=gathered_batch,
                    dst=0,
                    group=object_group,
                )
                
                if rank == 0:
                    # FIX: Append each rank's data to its specific sub-list to maintain order
                    for i, rank_data in enumerate(gathered_batch):
                        final_tokens_per_rank[i].extend(rank_data)
                    
                    del gathered_batch
                
                # Nuke the local token states on ALL ranks so they don't accumulate
                del token_states
            else:
                # Fallback for single GPU
                local_token_states.extend(token_states)
                del token_states

    if not local_anchors:
        empty_tokens = [] if include_tokens else None
        return np.array([]), [], empty_tokens

    local_anchor_tensor = torch.cat(local_anchors, dim=0).contiguous()

    if dist.is_initialized():
        gathered_anchors = [torch.zeros_like(local_anchor_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_anchors, local_anchor_tensor)

        gathered_infos = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_infos, local_infos, group=object_group)

        if rank == 0:
            final_anchors = torch.cat(gathered_anchors, dim=0).cpu().float().numpy()
            final_infos = [info for rank_list in gathered_infos for info in rank_list]

            if include_tokens:
                # FIX: Flatten the lists sequentially to perfectly match anchors and infos
                final_tokens = [token for rank_list in final_tokens_per_rank for token in rank_list]
                del final_tokens[full_dataset_len:]
            else:
                final_tokens = None

            return (
                final_anchors[:full_dataset_len],
                final_infos[:full_dataset_len],
                final_tokens,
            )
        return None, None, None

    # Fallback for non-distributed (single GPU) runs
    if include_tokens:
        del local_token_states[full_dataset_len:]
        
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

    for query_idx, tokens in enumerate(query_token_arrays):  # [num_queries, max_query len, dim (2048)]
        token_tensor = _as_token_tensor(tokens, device=device, dtype=dtype)
        query_tensor[query_idx, : token_tensor.shape[0]] = token_tensor
        query_mask[query_idx, : token_tensor.shape[0]] = True

    chunk_scores = []
    # candidate_token_arrays: [num_candidates, max_candidate len, dim (2048)]
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

        # query_tensor: [num_queries, max_query len, dim (2048)]
        # cand_tensor: [num_candidates, max_candidate len, dim (2048)]
        # sims: [num_queries, max_query len, num_candidates, max_candidate len]
        # max_sims: [num_queries, max_query len, num_candidates]

        sims = torch.einsum("bqd,ckd->bqck", query_tensor, cand_tensor)
        min_value = torch.finfo(sims.dtype).min
        sims = sims.masked_fill(~cand_mask[None, None, :, :], min_value)
        max_sims = sims.max(dim=-1).values.float()
        max_sims = max_sims.masked_fill(~query_mask[:, :, None], 0.0)
        denom = query_mask.sum(dim=1).clamp_min(1).float()[:, None]
        chunk_scores.append(max_sims.sum(dim=1) / denom)

        #import pdb; pdb.set_trace()
        # cand_tensor: [num_candidates, max_candidate len, dim (2048)]
        # cand_mask: [num_candidates, max_candidate len]

        del cand_tensor, cand_mask, sims, max_sims

    return torch.cat(chunk_scores, dim=1)



def _expected_scoring_config(late_args: LateInteractionArguments):
    return {
        "enable_late_interaction": bool(late_args.enable_late_interaction),
        "lambda_anchor": float(late_args.lambda_anchor),
        "lambda_late": float(late_args.lambda_late),
        "tie_breaker": "query_candidate_hash",
    }


def _score_cache_matches(score_dict, expected_config):
    return (
        "num_pred" in score_dict
        and score_dict.get("scoring_config", _expected_scoring_config(LateInteractionArguments()))
        == expected_config
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

import torch
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

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
    base_device = model.device
    num_gpus = 8 # Total devices as specified
    cand_keys = list(cand_embed_dict.keys())
    cand_embeds = np.stack([cand_embed_dict[key] for key in cand_keys])

    use_late = late_args.enable_late_interaction
    query_chunk_size = max(1, late_args.late_query_chunk_size)
    candidate_chunk_size = max(1, late_args.late_candidate_chunk_size)

    # ---------------------------------------------------------
    # 1. DISTRIBUTE: Split candidates across 8 GPUs upfront
    # ---------------------------------------------------------
    cand_splits = np.array_split(cand_embeds, num_gpus)
    cand_tensors = []
    for i, split in enumerate(cand_splits):
        device = torch.device(f"cuda:{i}")
        cand_tensors.append(torch.from_numpy(split).to(device=device, dtype=torch.float32))

    cand_token_splits = []
    if use_late:
        candidate_tokens = [cand_token_dict[key] for key in cand_keys]
        # Split tokens to align perfectly with the candidate tensor chunks
        start_idx = 0
        for split in cand_splits:
            end_idx = start_idx + len(split)
            cand_token_splits.append(candidate_tokens[start_idx:end_idx])
            start_idx = end_idx

    qry_tensor_full = torch.from_numpy(qry_embeds).to(dtype=torch.float32)
    all_ranked_indices = []

    progress = tqdm(
        range(0, qry_tensor_full.shape[0], query_chunk_size),
        desc=f"Global Ranking: {dataset_name}",
        disable=local_rank > 0,
        ncols=120,
    )

    # We use a ThreadPool. PyTorch releases the GIL during CUDA ops, achieving 
    # parallel multiprocessing across devices without IPC serialization overhead.
    pool = ThreadPool(processes=num_gpus)

    def compute_on_gpu(gpu_id, local_qry_chunk, local_qry_tokens):
        """Worker function bound to a specific GPU."""
        device = torch.device(f"cuda:{gpu_id}")
        
        # Move the query batch to this specific GPU
        local_qry = local_qry_chunk.to(device)
        local_cand = cand_tensors[gpu_id]

        # Compute anchor scores (Assumes model.compute_similarity doesn't lock to base_device)
        anchor_scores = model.compute_similarity(local_qry, local_cand)

        if use_late:
            local_late_scores = compute_late_scores_for_query_batch(
                local_qry_tokens,
                cand_token_splits[gpu_id],
                device=device,
                candidate_chunk_size=candidate_chunk_size,
            )
            scores = late_args.lambda_anchor * anchor_scores + late_args.lambda_late * local_late_scores
        else:
            scores = anchor_scores

        # Push back to rank0's base device for the final gather
        return scores.to(base_device)

    # ---------------------------------------------------------
    # 2. BATCH LOOP: Process, Wait, and Gather
    # ---------------------------------------------------------
    for start in progress:
        end = min(start + query_chunk_size, qry_tensor_full.shape[0])
        qry_chunk = qry_tensor_full[start:end]
        qry_tokens_chunk = query_tokens[start:end] if use_late else None

        # Build arguments for the 8 workers
        worker_args = [(i, qry_chunk, qry_tokens_chunk) for i in range(num_gpus)]

        # Map dispatches the work and WAITS for all 8 GPUs to finish the batch
        gathered_score_splits = pool.starmap(compute_on_gpu, worker_args)

        # GATHER WISELY: Concat the 8 score splits along the candidate dimension (dim=1)
        # Because we split candidates sequentially, concatenating them perfectly 
        # reconstructs the original `cand_keys` order.
        scores = torch.cat(gathered_score_splits, dim=1)

        for row_idx in range(scores.shape[0]):
            all_ranked_indices.append(
                _argsort_desc_with_hash_tiebreak(
                    scores=scores[row_idx],
                    candidate_names=cand_keys,
                    query_key=f"{dataset_name}:{start + row_idx}",
                )
            )

        # Cleanup memory for the next batch
        del gathered_score_splits, scores
        torch.cuda.empty_cache()

    # Cleanup pool and persistent GPU memory
    pool.close()
    pool.join()
    del cand_tensors, qry_tensor_full
    torch.cuda.empty_cache()

    return cand_keys, all_ranked_indices

# def _rank_global(
#     model,
#     qry_embeds,
#     cand_embed_dict,
#     query_tokens,
#     cand_token_dict,
#     late_args,
#     local_rank,
#     dataset_name,
# ):
#     device = model.device
#     cand_keys = list(cand_embed_dict.keys())
#     cand_embeds = np.stack([cand_embed_dict[key] for key in cand_keys])
#     cand_tensor = torch.from_numpy(cand_embeds).to(device=device, dtype=torch.float32)
#     qry_tensor = torch.from_numpy(qry_embeds).to(device=device, dtype=torch.float32)

#     all_ranked_indices = []
#     use_late = late_args.enable_late_interaction
#     query_chunk_size = max(1, late_args.late_query_chunk_size)
#     candidate_chunk_size = max(1, late_args.late_candidate_chunk_size)
#     candidate_tokens = None
#     if use_late:
#         candidate_tokens = [cand_token_dict[key] for key in cand_keys]

#     progress = tqdm(
#         range(0, qry_tensor.shape[0], query_chunk_size),
#         desc=f"Global Ranking: {dataset_name}",
#         disable=local_rank > 0,
#         ncols=120,
#     )
#     for start in progress:
#         end = min(start + query_chunk_size, qry_tensor.shape[0])
#         anchor_scores = model.compute_similarity(qry_tensor[start:end], cand_tensor)
#         if use_late:
#             late_scores = compute_late_scores_for_query_batch(
#                 query_tokens[start:end],
#                 candidate_tokens,
#                 device=device,
#                 candidate_chunk_size=candidate_chunk_size,
#             )
#             scores = late_args.lambda_anchor * anchor_scores + late_args.lambda_late * late_scores
#         else:
#             scores = anchor_scores

#         #import pdb; pdb.set_trace()

#         for row_idx in range(scores.shape[0]):
#             all_ranked_indices.append(
#                 _argsort_desc_with_hash_tiebreak(
#                     scores=scores[row_idx],
#                     candidate_names=cand_keys,
#                     query_key=f"{dataset_name}:{start + row_idx}",
#                 )
#             )

#         del anchor_scores, scores
#         if use_late:
#             del late_scores

#     del cand_tensor, qry_tensor
#     torch.cuda.empty_cache()
#     return cand_keys, all_ranked_indices


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
        #import pdb; pdb.set_trace()

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


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))

    object_gather_group = (
        dist.new_group(backend="gloo") if dist.is_initialized() else None
    )

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    print_master("=== Distributed Setup Initialized ===")
    print_master(f"Master Info -> ADDR: {os.environ.get('MASTER_ADDR')}, PORT: {os.environ.get('MASTER_PORT')}")
    print_master(f"Global World Size: {world_size}")
    print_rank(f"Process Identity -> Rank: {rank}, Local Rank: {local_rank} on {torch.cuda.get_device_name()}")

    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments, LateInteractionArguments))
    model_args, data_args, eval_args, late_args = parser.parse_args_into_dataclasses()
    os.makedirs(data_args.encode_output_path, exist_ok=True)


    # DDP-safe model loading
    # Step 1: Only rank 0 downloads the model
    if rank == 0:
        print_master(f"[rank=0] Loading the model from: {model_args.model_name_or_path}...")
        model = GmeQwen2VL(
            model_args.model_name_or_path,
            device=f"cuda:{rank}"
        )

    # Step 2: All processes wait until rank 0 finishes downloading
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Step 3: Non-master processes load from local cache
    if rank != 0:
        print_rank("Loading the model from cache...")
        # time.sleep(random.randint(2 * rank, 3 * rank))
        model = GmeQwen2VL(
            model_args.model_name_or_path,
            device=f"cuda:{rank}"
        )

    print_master(
        "Late interaction: "
        f"enabled={late_args.enable_late_interaction}, "
        f"lambda_anchor={late_args.lambda_anchor}, "
        f"lambda_late={late_args.lambda_late}"
    )

    with open(data_args.dataset_config, "r") as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)

    for dataset_name, task_config in dataset_configs.items():
        if dist.is_initialized():
            dist.barrier()
        print_master(f"--- Evaluating {dataset_name} ---")

        query_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_qry")
        cand_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_tgt")
        query_token_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_qry_tok")
        cand_token_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_tgt_tok")
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_info.jsonl")

        do_query = not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path)
        do_cand = not os.path.exists(cand_embed_path)
        
        if late_args.enable_late_interaction:
            do_query = do_query or not os.path.exists(query_token_path)
            do_cand = do_cand or not os.path.exists(cand_token_path)
        
        score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
        if os.path.exists(score_path):
            do_cand = False
            do_query = False

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
                description=f"Queries: {dataset_name}",
                object_group=object_gather_group,
            )
            if rank == 0:
                os.makedirs(os.path.dirname(query_embed_path), exist_ok=True)

                # Save embeddings
                with open(query_embed_path, "wb") as f:
                    pickle.dump(query_embeds, f)

                # Save dataset info in JSONL format
                with open(dataset_info_path, "w", encoding="utf-8") as f:
                    for info in gt_infos:
                        f.write(json.dumps(info, ensure_ascii=False) + "\n")

                if late_args.enable_late_interaction:
                    with open(query_token_path, "wb") as f:
                        pickle.dump(query_token_states, f)
                    del query_token_states
                    gc.collect()
                print_master(f"Successfully saved {len(query_embeds)} query embeddings to {query_embed_path}")

            if dist.is_initialized():
                dist.barrier()

        # 2. Compute candidate embeddings
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
                description=f"Candidates: {dataset_name}",
                object_group=object_gather_group,
            )
            if rank == 0:
                os.makedirs(os.path.dirname(cand_embed_path), exist_ok=True)

                # Map embeddings to dictionary: {cand_id: embedding_vector}
                # Enables fast lookup by ID during retrieval evaluation

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
                    del cand_token_dict, cand_token_states
                    gc.collect()
                print_master(f"Successfully saved {len(cand_embed_dict)} unique candidate embeddings to {cand_embed_path}")

            if dist.is_initialized():
                dist.barrier()

        # 3. Compute scores (rank 0 only)
        if rank == 0:
            score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
            pred_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_pred.jsonl")
            expected_config = _expected_scoring_config(late_args)

            # Skip computation only if both files exist and are valid
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
                        need_compute = False
                    else:
                        print_master(f"Scoring config changed for {dataset_name}, re-computing...")
                except Exception as e:
                    print_master(f"Cache for {dataset_name} is corrupted ({e}), re-computing...")

            if need_compute:
                # Load persisted embeddings and metadata
                with open(query_embed_path, "rb") as f:
                    qry_embeds = pickle.load(f)
                with open(cand_embed_path, "rb") as f:
                    cand_embed_dict = pickle.load(f)

                 # Explicitly specify UTF-8 encoding to handle non-ASCII characters in dataset metadata
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

                formatted = {
                    k: f"{v:.4f}"
                    for k, v in score_dict.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                print_master(f"Final Score for {dataset_name}: {formatted}")
        
        
        # Clean up token cache after evaluation to save disk space
        if rank == 0:
            try:
                os.remove(query_embed_path)
                os.remove(cand_embed_path)
            except Exception as e:
                print_master(f"Failed to clean up embedding cache for {dataset_name}: {e}")

            if late_args.enable_late_interaction:
                    try:
                        os.remove(query_token_path)
                        os.remove(cand_token_path)
                        print_master(f"Cleaned up token cache for {dataset_name}")
                    except Exception as e:
                        print_master(f"Failed to clean up token cache for {dataset_name}: {e}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
