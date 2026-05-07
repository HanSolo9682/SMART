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
from transformers import HfArgumentParser, AutoModel

from datasets import concatenate_datasets
from datasets.distributed import split_dataset_by_node
from multiprocessing.pool import ThreadPool

from .arguments import ModelArguments, DataArguments, EvalArguments
from .utils.basic_utils import print_rank, print_master
from .utils.eval_utils.metrics import RankingMetrics
from .data.datasets.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
from .data.collator import MultimodalEvalDataCollator


@dataclass
class LateInteractionArguments:
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


@torch.no_grad()
def encode_tokens(model, batch_inputs, encode_side: str):
    """
    Leverages Jina V4's native processor and multi_vec_emb extraction.
    """
    processor = model.processor
    texts = []
    images = []

    for ele in batch_inputs:
        text_val = ele.get("text")
        img_val = ele.get("image")

        if img_val is not None:
            images.append(img_val)
            # # Use Jina/Qwen2.5-VL's hardcoded prompt structure instead of apply_chat_template
            # prompt_text = text_val if text_val else "Describe the image."
            # prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt_text}<|im_end|>\n"
            # texts.append(prompt)
        else:
            # # Jina specifies prefixing single-modality text based on task
            # prefix = "Query: " if encode_side == "qry" else "Passage: "
            # texts.append(f"{prefix}{text_val or ''}")
            texts.append(text_val)

    # valid_images = [img for img in images if img is not None]
    # if len(valid_images) == 0:
    #     valid_images = None

    # batch = processor(
    #     text=texts,
    #     images=valid_images,
    #     padding="longest",
    #     return_tensors="pt"
    # )
    # batch = {k: v.to(model.device) for k, v in batch.items()}

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Forward pass requesting retrieval task representations
        # outputs = model(**batch, task_label="retrieval")
        if images:
            outputs = model.encode_image(images, task="retrieval", batch_size=8, return_multivector=True)
        else:
            outputs = model.encode_text(texts, task="retrieval", batch_size=32, return_multivector=True)

    return outputs
    # multi_vec_emb = outputs.multi_vec_emb
    # valid_tokens = batch["attention_mask"].bool()

    # token_states = []
    # # Jina's multi_vec_emb zeroes out padding, but we still need to strictly slice 
    # # out the unpadded valid tokens as variable-length arrays for MaxSim.
    # for emb, mask in zip(multi_vec_emb, valid_tokens):
    #     valid_emb = emb[mask]
    #     token_states.append(valid_emb.cpu().numpy().astype(np.float32))

    # return token_states


import gc

@torch.no_grad()
def encode_representations(
    model,
    loader,
    encode_side: str,
    full_dataset_len: int,
    description: str = "Encoding",
    object_group=None,
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    local_infos = []
    final_tokens_per_rank = [[] for _ in range(world_size)] if rank == 0 else None
    local_token_states = [] if not dist.is_initialized() else None

    model.eval()
    progress_bar = tqdm(
        loader,
        desc=f"{description} (rank {rank})",
        disable=local_rank > 0,
        ncols=120,
    )

    for batch_inputs, dataset_info in progress_bar:
        token_states = encode_tokens(model, batch_inputs, encode_side)

        if encode_side == "qry":
            local_infos.extend(dataset_info)
        else:
            local_infos.extend([info.get("cand_name", "") for info in dataset_info])

        # --- MEMORY EFFICIENT PER-BATCH GATHER ---
        if dist.is_initialized():
            gathered_batch = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object(
                token_states,
                object_gather_list=gathered_batch,
                dst=0,
                group=object_group,
            )
            
            if rank == 0:
                for i, rank_data in enumerate(gathered_batch):
                    final_tokens_per_rank[i].extend(rank_data)
                del gathered_batch
            
            del token_states
        else:
            local_token_states.extend(token_states)
            del token_states

    if dist.is_initialized():
        gathered_infos = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_infos, local_infos, group=object_group)

        if rank == 0:
            final_infos = [info for rank_list in gathered_infos for info in rank_list]
            final_tokens = [token for rank_list in final_tokens_per_rank for token in rank_list]
            
            del final_tokens[full_dataset_len:]
            return final_infos[:full_dataset_len], final_tokens
        
        return None, None

    # Fallback for single GPU
    del local_token_states[full_dataset_len:]
    return local_infos, local_token_states


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


def _expected_scoring_config(late_args: LateInteractionArguments):
    return {
        "tie_breaker": "query_candidate_hash",
        "late_candidate_chunk_size": late_args.late_candidate_chunk_size,
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


def _rank_global(
    query_tokens,
    cand_token_dict,
    late_args,
    local_rank,
    dataset_name,
    base_device,
):
    num_gpus = torch.cuda.device_count() or 1
    cand_keys = list(cand_token_dict.keys())
    candidate_tokens = [cand_token_dict[key] for key in cand_keys]

    query_chunk_size = max(1, late_args.late_query_chunk_size)
    candidate_chunk_size = max(1, late_args.late_candidate_chunk_size)

    # Distribute tokens mathematically rather than tensors
    split_size = (len(candidate_tokens) + num_gpus - 1) // num_gpus
    cand_token_splits = [
        candidate_tokens[i : i + split_size]
        for i in range(0, len(candidate_tokens), split_size)
    ]
    
    # Ensure splits matches number of GPUs exactly (pads with empty if fewer candidates)
    while len(cand_token_splits) < num_gpus:
         cand_token_splits.append([])

    all_ranked_indices = []

    progress = tqdm(
        range(0, len(query_tokens), query_chunk_size),
        desc=f"Global Ranking: {dataset_name}",
        disable=local_rank > 0,
        ncols=120,
    )

    pool = ThreadPool(processes=num_gpus)

    def compute_on_gpu(gpu_id, local_qry_tokens):
        device = torch.device(f"cuda:{gpu_id}")
        local_cands = cand_token_splits[gpu_id]
        
        if not local_cands:
             return torch.empty(len(local_qry_tokens), 0, device=base_device)
             
        scores = compute_late_scores_for_query_batch(
            local_qry_tokens,
            local_cands,
            device=device,
            candidate_chunk_size=candidate_chunk_size,
        )
        return scores.to(base_device)

    for start in progress:
        end = min(start + query_chunk_size, len(query_tokens))
        qry_tokens_chunk = query_tokens[start:end]

        worker_args = [(i, qry_tokens_chunk) for i in range(num_gpus)]
        gathered_score_splits = pool.starmap(compute_on_gpu, worker_args)

        scores = torch.cat(gathered_score_splits, dim=1)

        for row_idx in range(scores.shape[0]):
            all_ranked_indices.append(
                _argsort_desc_with_hash_tiebreak(
                    scores=scores[row_idx],
                    candidate_names=cand_keys,
                    query_key=f"{dataset_name}:{start + row_idx}",
                )
            )

        del gathered_score_splits, scores
        torch.cuda.empty_cache()

    pool.close()
    pool.join()
    torch.cuda.empty_cache()

    return cand_keys, all_ranked_indices


def _rank_local(
    gt_infos,
    query_tokens,
    cand_token_dict,
    late_args,
    local_rank,
    dataset_name,
    base_device,
):
    candidate_chunk_size = max(1, late_args.late_candidate_chunk_size)
    ranked_name_lists = []
    
    progress = tqdm(
        enumerate(gt_infos),
        total=len(gt_infos),
        desc=f"Local Ranking: {dataset_name}",
        disable=local_rank > 0,
        ncols=120,
    )
    
    for qid, gt_info in progress:
        cand_names = gt_info["cand_names"]
        
        scores = compute_late_scores_for_query_batch(
            [query_tokens[qid]],
            [cand_token_dict[name] for name in cand_names],
            device=base_device,
            candidate_chunk_size=candidate_chunk_size,
        ).squeeze(0)

        ranked_idx = _argsort_desc_with_hash_tiebreak(
            scores=scores,
            candidate_names=cand_names,
            query_key=f"{dataset_name}:{qid}",
        )
        ranked_name_lists.append([cand_names[i] for i in ranked_idx])

        del scores, ranked_idx

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

    # ---------------------------------------------------------
    # DDP-safe model loading using AutoModel (Jina setup)
    # ---------------------------------------------------------
    if rank == 0:
        print_master(f"[rank=0] Loading the model from: {model_args.model_name_or_path}...")
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        model.task = "retrieval" # Initialize the default task label for Jina

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    if rank != 0:
        print_rank("Loading the model from cache...")
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        model.task = "retrieval"

    model.eval()
    model = model.to(eval_args.device)

    with open(data_args.dataset_config, "r") as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)

    for dataset_name, task_config in dataset_configs.items():
        if dist.is_initialized():
            dist.barrier()
        print_master(f"--- Evaluating {dataset_name} ---")

        # Simplified paths: Only tokens matter for strictly late interaction
        query_token_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_qry_tok.pkl")
        cand_token_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_tgt_tok.pkl")
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_info.jsonl")
        score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")

        do_query = not os.path.exists(query_token_path) or not os.path.exists(dataset_info_path)
        do_cand = not os.path.exists(cand_token_path)
        
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
            gt_infos, query_token_states = encode_representations(
                model=model,
                loader=eval_qry_loader,
                encode_side="qry",
                full_dataset_len=len(full_eval_qry_dataset),
                description=f"Queries: {dataset_name}",
                object_group=object_gather_group,
            )
            
            if rank == 0:
                os.makedirs(os.path.dirname(query_token_path), exist_ok=True)
                
                with open(dataset_info_path, "w", encoding="utf-8") as f:
                    for info in gt_infos:
                        f.write(json.dumps(info, ensure_ascii=False) + "\n")

                with open(query_token_path, "wb") as f:
                    pickle.dump(query_token_states, f)
                
                del query_token_states
                gc.collect()
                print_master(f"Successfully saved query tokens to {query_token_path}")

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
            all_cand_ids, cand_token_states = encode_representations(
                model=model,
                loader=eval_cand_loader,
                encode_side="cand",
                full_dataset_len=len(full_eval_cand_dataset),
                description=f"Candidates: {dataset_name}",
                object_group=object_gather_group,
            )
            
            if rank == 0:
                os.makedirs(os.path.dirname(cand_token_path), exist_ok=True)

                cand_token_dict = {
                    cand_id: tokens for cand_id, tokens in zip(all_cand_ids, cand_token_states)
                }
                with open(cand_token_path, "wb") as f:
                    pickle.dump(cand_token_dict, f)
                    
                del cand_token_dict, cand_token_states
                gc.collect()
                print_master(f"Successfully saved candidate tokens to {cand_token_path}")

            if dist.is_initialized():
                dist.barrier()

        # ---------------------------------------------------------
        # Compute Scores
        # ---------------------------------------------------------
        if rank == 0:
            pred_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_pred.jsonl")
            expected_config = _expected_scoring_config(late_args)

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
                gt_infos = [json.loads(l) for l in open(dataset_info_path, encoding="utf-8")]

                with open(query_token_path, "rb") as f:
                    query_tokens = pickle.load(f)
                with open(cand_token_path, "rb") as f:
                    cand_token_dict = pickle.load(f)

                pred_dicts = []
                rank_against_all_candidates = task_config.get("eval_type", "global") == "global"

                if rank_against_all_candidates:
                    cand_keys, ranked_indices = _rank_global(
                        query_tokens=query_tokens,
                        cand_token_dict=cand_token_dict,
                        late_args=late_args,
                        local_rank=local_rank,
                        dataset_name=dataset_name,
                        base_device=eval_args.device,
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
                        gt_infos=gt_infos,
                        query_tokens=query_tokens,
                        cand_token_dict=cand_token_dict,
                        late_args=late_args,
                        local_rank=local_rank,
                        dataset_name=dataset_name,
                        base_device=eval_args.device,
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