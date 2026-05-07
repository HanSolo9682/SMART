from typing import Tuple, Optional, List, Union 
import torch 
from transformers.utils import logging

logger = logging.get_logger(__name__)

from transformers import Qwen3VLConfig
from transformers import Qwen3VLForConditionalGeneration
from torch import nn 
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Dict
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Literal

from torch.distributed.nn.functional import all_gather as diff_all_gather

@dataclass
class MultiLayerLossOutput(SequenceClassifierOutput):
    layer_losses: Optional[Dict[str, torch.FloatTensor]] = None

def block_normalize(x, num_blocks=2):
    """
    x: (..., d)
    num_blocks: how many equal chunks to split into (e.g., 2 → halves, 4 → quarters)

    returns (..., d) with each block L2-normalized separately.
    """
    d = x.size(-1)
    assert d % num_blocks == 0, "Embedding dim must be divisible by num_blocks."

    # Split into equal-sized blocks
    blocks = torch.split(x, d // num_blocks, dim=-1)

    # Normalize each block
    blocks = [F.normalize(b, dim=-1) for b in blocks]

    # Concatenate back
    return torch.cat(blocks, dim=-1)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=0.07, use_doc_attention=False):
        super().__init__()
        self.temp = temp
        self.use_doc_attention = use_doc_attention
        self.cos = nn.CosineSimilarity(dim=-1)
        
        
    def batch_cosine_sim(self, q_norm: torch.Tensor, c_norm: torch.Tensor) -> torch.Tensor:
        """
        q: [B, D]
        c: [B, B, D]
        Returns:
            sims: [B, B], where sims[i,j] = cosine_similarity(q[i], c[i,j,:])
        """
        B = q_norm.shape[0]
        # batch dot products → [B, B]
        qi_vs_dji_sims = torch.einsum("qd,tqd->qt", q_norm, c_norm) # Calculates how similar each query (q_i) is to other documents with respect to the i-th query (dji)- this is the setting faced during inference
        qi_vs_dij_sims = torch.einsum("qd,qtd->qt", q_norm, c_norm) # Calculates how similar each query (q_i) is to the same document with respect to the j-th query (dij) - this is to make sure that the document embeddings use the context to update in some way
        
        # Remove the diagonal elements present in qi_vs_dij_sims (since they are basically our positive pair), and keep a [B, B-1] matrix to later concat the columns
        mask = ~torch.eye(B, dtype=torch.bool, device=qi_vs_dij_sims.device)
        qi_vs_dij_sims_no_diag = qi_vs_dij_sims[mask].reshape(B, B - 1)
        sims = torch.cat((qi_vs_dji_sims, qi_vs_dij_sims_no_diag), dim=1)
           
        return sims

    def forward(self, x, y):
        if self.use_doc_attention:
            return self.batch_cosine_sim(x, y) / self.temp
        else:
            return self.cos(x.unsqueeze(1), y.unsqueeze(0)) / self.temp

class Qwen3VLRetFinetuneForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config: Qwen3VLConfig, embed_dim=3584, pad_token_id=None, late_method="disabled"):
        super().__init__(config)
        self.late_method: Literal["disabled", "late_only", "hybrid"] = late_method
        self.pad_token_id = pad_token_id
            
    def get_features(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).to(inputs_embeds.device)
                image_mask = input_ids == self.config.image_token_id
                if self.training:
                    inputs_embeds = inputs_embeds.clone()
                inputs_embeds[image_mask] = image_embeds
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
                video_mask = input_ids == self.config.video_token_id
                inputs_embeds[video_mask] = video_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] 
        embed_index = self.config.emb_token_ids[0]
        embed_indices = torch.argmax((labels == embed_index).int(), dim=1) 
        embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] # (batch_size, embed_dim)
        return embed_features 
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        inference=False,
        has_hard_negative=False,
        qids=None,
        dids=None,
        ids=None,
        ignore_token_id=None,
    ) -> Union[Tuple, Qwen3VLCausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mini_batch_size = 32 
        input_ids_list = torch.split(input_ids, mini_batch_size)
        
        if attention_mask is not None:
            attention_mask_list = torch.split(attention_mask, mini_batch_size)
        else:
            attention_mask_list = [None] * len(input_ids_list)

        if image_grid_thw is not None:
            cumsum_pixel_values = torch.cumsum(image_grid_thw[:, 1] * image_grid_thw[:, 2], dim=-1) 
            zero_tensor = torch.tensor([0], device=cumsum_pixel_values.device)
            cumsum_pixel_values = torch.cat((zero_tensor, cumsum_pixel_values))
            image_nums = 0
        
        all_hidden_states = []
        
        for i in range(len(input_ids_list)):
            batch_attention_mask = None
            if inputs_embeds is None:
                batch_inputs_embeds = self.model.language_model.embed_tokens(input_ids_list[i])
                if pixel_values is not None:
                    image_mask = input_ids_list[i] == self.config.image_token_id
                    current_image_num = torch.sum(torch.any(image_mask, dim=-1)).cpu().item()
                    if current_image_num != 0:
                        batch_pixel_values = pixel_values[cumsum_pixel_values[image_nums] : cumsum_pixel_values[image_nums + current_image_num]]
                        batch_pixel_values = batch_pixel_values.type(self.visual.dtype)
                        batch_image_embeds = self.visual(batch_pixel_values, grid_thw=image_grid_thw[image_nums:image_nums + current_image_num])[0].to(batch_inputs_embeds.device)
                        image_nums = image_nums + current_image_num
                        if self.training:
                            batch_inputs_embeds = batch_inputs_embeds.clone()
                        batch_inputs_embeds[image_mask] = batch_image_embeds
                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
                    video_mask = input_ids == self.config.video_token_id
                    inputs_embeds[video_mask] = video_embeds
                if attention_mask is not None:
                    batch_attention_mask = attention_mask_list[i].to(batch_inputs_embeds.device)

            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=batch_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=batch_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            hidden_states = outputs[0]
            all_hidden_states.append(hidden_states)

        hidden_states = torch.cat(all_hidden_states)
        
        if has_hard_negative:
            batch_size = len(hidden_states) // 3
        elif not inference:
            batch_size = len(hidden_states) // 2
        elif inference:
            batch_size = len(hidden_states)

        if inference:
            assert batch_size == len(hidden_states)

        embed_index = self.config.emb_token_ids[0]
        embed_indices = torch.argmax((labels == embed_index).int(), dim=1)         
        embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1]
        

        if inference:
            if ids is not None:
                return embed_features, hidden_states, ids 
            elif qids is not None or dids is not None:
                return embed_features, hidden_states, qids, dids 
            return embed_features, hidden_states
        
        loss_fct = nn.CrossEntropyLoss()
        
        if self.late_method != "late_only":
            if has_hard_negative:
                embed1, embed2, embed3 = embed_features[:batch_size], embed_features[batch_size:2*batch_size], embed_features[2*batch_size:]
            else:
                embed1, embed2 = embed_features[:batch_size], embed_features[batch_size:]

            if dist.is_initialized():
                if has_hard_negative:
                    embed3 = torch.cat(diff_all_gather(embed3.contiguous()), 0).contiguous()
                
                embed1 = torch.cat(diff_all_gather(embed1.contiguous()), 0).contiguous()
                embed2 = torch.cat(diff_all_gather(embed2.contiguous()), 0).contiguous()

            sim = Similarity(temp=0.05, use_doc_attention=self.use_doc_attention)

            embed1 = F.normalize(embed1, dim=-1)
            embed2 = F.normalize(embed2, dim=-1)
            cos_sim = sim(embed1, embed2)
            
            if has_hard_negative:
                embed1_embed3_cos = sim(embed1.unsqueeze(1), embed3.unsqueeze(0))
                cos_sim = torch.cat([cos_sim, embed1_embed3_cos], 1)
            
            if self.late_method == "disabled":
                nce_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
                total_loss = loss_fct(cos_sim, nce_labels)

        # LATE INTERACTION (MAXSIM) COMPUTATION
        if self.late_method != "disabled":
            q_toks = hidden_states[:batch_size]
            d_toks = hidden_states[batch_size:2*batch_size]
            
            if attention_mask is not None:
                q_mask = attention_mask[:batch_size]
                d_mask = attention_mask[batch_size:2*batch_size]
            else:
                q_mask = torch.ones(q_toks.shape[:2], device=q_toks.device, dtype=torch.bool)
                d_mask = torch.ones(d_toks.shape[:2], device=d_toks.device, dtype=torch.bool)

            if has_hard_negative:
                neg_d_toks = hidden_states[2*batch_size:]
                neg_d_mask = attention_mask[2*batch_size:] if attention_mask is not None else torch.ones(neg_d_toks.shape[:2], device=neg_d_toks.device, dtype=torch.bool)

            if dist.is_initialized():
                q_toks = torch.cat(diff_all_gather(q_toks), 0).contiguous()
                d_toks = torch.cat(diff_all_gather(d_toks), 0).contiguous()
                
                # Standard gather for boolean masks
                q_mask_list = [torch.zeros_like(q_mask) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=q_mask_list, tensor=q_mask.contiguous())
                q_mask = torch.cat(q_mask_list, 0).contiguous()

                d_mask_list = [torch.zeros_like(d_mask) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=d_mask_list, tensor=d_mask.contiguous())
                d_mask = torch.cat(d_mask_list, 0).contiguous()

                if has_hard_negative:
                    neg_d_toks = torch.cat(diff_all_gather(neg_d_toks.contiguous()), 0).contiguous()
                    
                    neg_d_mask_list = [torch.zeros_like(neg_d_mask) for _ in range(dist.get_world_size())]
                    dist.all_gather(tensor_list=neg_d_mask_list, tensor=neg_d_mask.contiguous())
                    neg_d_mask = torch.cat(neg_d_mask_list, 0).contiguous()

            def compute_maxsim(query_tokens, doc_tokens, query_m, doc_m, q_chunk_size=32, d_chunk_size=32):
                """
                Computes ColBERT MaxSim with double-chunking (Queries and Documents)
                to completely prevent 4D einsum OOM and CUDA hangs.
                """
                q_norm = F.normalize(query_tokens, p=2, dim=-1)
                d_norm = F.normalize(doc_tokens, p=2, dim=-1)
                
                B, Q, _ = q_norm.shape
                C, K, _ = d_norm.shape
                
                # Pre-compute boolean masks and denominator for the whole batch
                d_m_bool = doc_m.bool()
                q_m_bool = query_m.bool()
                denom = q_m_bool.sum(dim=1).clamp_min(1)[:, None]  # Shape: [B, 1]
                
                all_scores = []
                
                # Outer loop: Chunk over the Query batch dimension (B)
                for i in range(0, B, q_chunk_size):
                    q_chunk = q_norm[i : i + q_chunk_size]
                    q_m_chunk = q_m_bool[i : i + q_chunk_size]
                    denom_chunk = denom[i : i + q_chunk_size]
                    
                    q_chunk_max_sims = []
                    
                    # Inner loop: Chunk over the Document batch dimension (C)
                    for j in range(0, C, d_chunk_size):
                        d_chunk = d_norm[j : j + d_chunk_size]
                        d_m_chunk = d_m_bool[j : j + d_chunk_size]
                        
                        # 1. Cross-similarity for the sub-chunk
                        # Shape: [q_chunk_size, Q, d_chunk_size, K]
                        sims = torch.einsum("bqd,ckd->bqck", q_chunk, d_chunk)
                        
                        # 2. Document Masking
                        # Mask padded document tokens with the minimum possible value
                        min_val = torch.finfo(sims.dtype).min
                        sims = sims.masked_fill(~d_m_chunk[None, None, :, :], min_val)
                        
                        # 3. MaxSim calculation for this specific document chunk
                        # Max over doc tokens (dim=-1). Shape: [q_chunk_size, Q, d_chunk_size]
                        max_sims_sub = sims.max(dim=-1).values
                        
                        q_chunk_max_sims.append(max_sims_sub)
                        
                    # Recombine document chunks to get the full [q_chunk_size, Q, C] tensor
                    max_sims = torch.cat(q_chunk_max_sims, dim=-1)
                    
                    # 4. Query Masking
                    # Mask padded query tokens with 0.0 so they don't contribute to the sum
                    max_sims = max_sims.masked_fill(~q_m_chunk[:, :, None], 0.0)
                    
                    # 5. Aggregate and Average
                    # Sum over query tokens (dim=1) and divide by the number of valid query tokens
                    scores = max_sims.sum(dim=1) / denom_chunk  # Shape: [q_chunk_size, C]
                    
                    all_scores.append(scores)
                    
                # Recombine query chunks to return the final [B, C] score matrix
                return torch.cat(all_scores, dim=0)
            

            late_sim_scores = compute_maxsim(q_toks, d_toks, q_mask, d_mask)

            if has_hard_negative:
                late_sim_neg_scores = compute_maxsim(q_toks, neg_d_toks, q_mask, neg_d_mask)
                late_sim_scores = torch.cat([late_sim_scores, late_sim_neg_scores], 1)

            late_sim_scores = late_sim_scores / 0.05
            
            if self.late_method == "hybrid":
                total_loss = loss_fct(cos_sim + late_sim_scores, nce_labels)
            else:
                total_loss = loss_fct(late_sim_scores, nce_labels)

        return SequenceClassifierOutput(loss=total_loss)
