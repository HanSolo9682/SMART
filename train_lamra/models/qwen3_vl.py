from typing import Tuple, Optional, List, Union 
import torch 
from transformers.utils import logging

logger = logging.get_logger(__name__)
# from .custom_qwen2 import Qwen2VLForConditionalGeneration
from transformers import PreTrainedTokenizer, Qwen3VLForConditionalGeneration
from torch import nn 
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
import torch.nn.functional as F
from .aux_networks import *

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Qwen3VLRetForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, 
                 config, 
                 use_doc_attention: bool = False, 
                 use_full_attention: bool = False, 
                 use_up_attention: bool = False, 
                 embed_dim=3584, pad_token_id=None, 
                 use_steer_layers=False, 
                 replace_embed_layers=False,
                 use_multilayer_features=False,
                 num_layers=10
                 ):
        # super().__init__(config, use_steer_layers=use_steer_layers, replace_embed_layers=replace_embed_layers)
        super().__init__(config)
        self.use_doc_attention = use_doc_attention
        self.use_up_attention = use_up_attention
        self.pad_token_id = pad_token_id
        if self.use_doc_attention:
            self.doc_attention = BatchToBatchAttention(embed_dim=embed_dim, num_heads=1)
        if self.use_up_attention:
            self.up_attention = UpAttention(input_dim=embed_dim, hidden_dim=embed_dim + 768)
            
        self.use_multilayer_features = use_multilayer_features
        if self.use_multilayer_features:
            print("Using MultiLayer Feature Fusion")
            self.num_layers = num_layers
            # self.multilayer_embed_proj = nn.Linear(embed_dim * self.num_layers, embed_dim)
            hidden_dim = embed_dim * 4

            self.multilayer_embed_proj = nn.Sequential(
                nn.Linear(embed_dim * self.num_layers, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),

                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),

                nn.Linear(hidden_dim, embed_dim),
            )

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
        overrule_doc_attn: bool = False 
    ) -> Union[Tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)[0].to(inputs_embeds.device)
                image_mask = input_ids == self.config.image_token_id
                if self.training:
                    inputs_embeds = inputs_embeds.clone()

                inputs_embeds[image_mask] = image_embeds
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)[0].to(inputs_embeds.device)
                video_mask = input_ids == self.config.video_token_id
                inputs_embeds[video_mask] = video_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        
        if self.use_multilayer_features:
            output_hidden_states = True
        
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
            # labels=labels
        )
        
        if self.use_multilayer_features:
                last_ten = outputs.hidden_states[-10:]          # tuple of 10 tensors
                hidden_states = torch.cat(last_ten, dim=-1)
        else:
            hidden_states = outputs[0]

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
        embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] # (batch_size, embed_dim)
        
        if self.use_multilayer_features:
            embed_features = self.multilayer_embed_proj(embed_features)

        if self.use_doc_attention and not overrule_doc_attn:
            # Keep the query indices in memory, will use it later to calculate the final query vector
            query_indices = embed_indices[:batch_size]
            # identify padded positions to exclude from the later attention module
            final_attention_mask = (labels != ignore_token_id).int() 

        if inference:
            if self.use_doc_attention and not overrule_doc_attn:
                # hidden_states = F.normalize(hidden_states, dim=-1)
                # Take the whole thing cuz we will pass in docs and queries independently, so no mixing in batches
                query_attn_mask = final_attention_mask[torch.arange(len(embed_indices)), :embed_indices[0]-1]
                full_context = hidden_states[torch.arange(len(embed_indices)), :embed_indices[0]-1]  #includes the final query vector, idx(<emb>) - 2
                embed1_q = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] 
                # All of the sequences have been left padded (to max_length, therefore the <emb> position is the same)            
                embed_features = self.doc_attention(embed1_q, full_context, mode='simple', attn_mask=query_attn_mask) # (bs x N, hidden)
                
                if ids is not None:
                    return embed_features, full_context, query_attn_mask, ids 
                elif qids is not None or dids is not None:
                    return embed_features, full_context, query_attn_mask, qids, dids     
                
            if ids is not None:
                return embed_features, ids 
            
            elif qids is not None or dids is not None:
                return embed_features, qids, dids 
            return embed_features 
        
        if has_hard_negative:
            embed1, embed2, embed3 = embed_features[:batch_size], embed_features[batch_size:2*batch_size], embed_features[2*batch_size:]
        else:
            embed1, embed2 = embed_features[:batch_size], embed_features[batch_size:]
        loss_fct = nn.CrossEntropyLoss()

        if dist.is_initialized():
            if has_hard_negative:
                embed3_list = [torch.zeros_like(embed3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=embed3_list, tensor=embed3.contiguous())
                embed3_list[dist.get_rank()] = embed3 
                embed3 = torch.cat(embed3_list, 0)
            
            # Dummy vectors for allgather
            embed1_list = [torch.zeros_like(embed1) for _ in range(dist.get_world_size())]
            embed2_list = [torch.zeros_like(embed2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=embed1_list, tensor=embed1.contiguous())
            dist.all_gather(tensor_list=embed2_list, tensor=embed2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            embed1_list[dist.get_rank()] = embed1
            embed2_list[dist.get_rank()] = embed2
            # Get full batch embeddings: (bs x N, hidden)
            embed1 = torch.cat(embed1_list, 0)
            embed2 = torch.cat(embed2_list, 0)

        sim = Similarity(temp=0.05)

        # add normalization
        embed1 = F.normalize(embed1, dim=-1)
        embed2 = F.normalize(embed2, dim=-1)

        cos_sim = sim(embed1.unsqueeze(1), embed2.unsqueeze(0))

        if has_hard_negative:
            embed1_embed3_cos = sim(embed1.unsqueeze(1), embed3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, embed1_embed3_cos], 1)
        
        nce_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)

        loss = loss_fct(cos_sim, nce_labels)
        return SequenceClassifierOutput(loss=loss)

    def inference(
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
        batch_size = len(hidden_states)
        embed_index = self.config.emb_token_ids[0]
        embed_indices = torch.argmax((input_ids == embed_index).int(), dim=1) 
        embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] # (batch_size, embed_dim)
        embed_features = F.normalize(embed_features, dim=-1)
        return embed_features 
    
    # def compute_similarity_with_corpus_transform(self, candidate_features, full_context, full_attn_mask, query_features, batch_size=256):
    #     """Applies attention to the document vectors over the query embeddings as context, then does the matrix multiply and stores the similarities

    #     Args:
    #         candidate_features (torch.Tensor): Single Vector corpus embeddings (B, dim)
    #         full_context List[torch.Tensor]: Query context, to perform attention over 
    #         full_attn_mask: Attention mask to denote padded positions
    #         query_features: Query Vectors for similarity computation
    #     """
    #     num_cands = candidate_features.shape[0]
    #     sim_matrix = torch.zeros((query_features.shape[0],candidate_features.shape[0]))
    #     evolved_features = []
    #     device = next(self.doc_attention.parameters()).device
    #     with torch.no_grad():  
    #         for idx, (context, attn_mask) in enumerate(zip(full_context, full_attn_mask)):
    #             qnum = context.shape[0]
    #             # move to right device
    #             context = context.to(device)
    #             attn_mask = attn_mask.to(device) if attn_mask is not None else None
    #             qchunk = F.normalize(query_features[idx * qnum: (idx + 1) * qnum], dim=-1)

    #             with torch.autocast(device_type=device.type):
    #                 for start in range(0, num_cands, batch_size):
    #                     end = start + batch_size
    #                     cand_chunk = candidate_features[start:end]

    #                     # doc_attention for this slice
    #                     doc_embeds_chunk = F.normalize(self.doc_attention(
    #                         cand_chunk, context, mode="batch2batch", attn_mask=attn_mask
    #                     ), dim=-1)

    #                     # similarity scores
    #                     sim_matrix[idx * qnum: (idx + 1) * qnum, start:end] = torch.einsum("id,jid->ij", qchunk, doc_embeds_chunk)

    #     return sim_matrix.to(device).contiguous()

    def compute_similarity_with_corpus_transform(
        self,
        candidate_features: torch.Tensor,
        full_context: list[torch.Tensor],
        full_attn_mask: list[torch.Tensor],
        query_features: torch.Tensor,
        batch_size: int = 1024,
    ):
        """
        Applies attention to the document vectors over the query embeddings as context,
        then computes and stores the similarity matrix.

        Args:
            candidate_features (torch.Tensor): Corpus embeddings of shape (num_cands, dim)
            full_context (List[torch.Tensor]): List of context tensors for each query chunk
            full_attn_mask (List[torch.Tensor]): List of attention masks for each context
            query_features (torch.Tensor): Query feature matrix of shape (num_queries, dim)
            batch_size (int): Number of candidates processed per batch
        """
        num_cands = candidate_features.shape[0]
        num_queries = query_features.shape[0]

        # Device and dtype setup
        device = next(self.doc_attention.parameters()).device
        dtype = query_features.dtype

        # Initialize the similarity matrix on the same device
        sim_matrix = torch.zeros((num_queries, num_cands), device=device, dtype=dtype)

        offset = 0
        with torch.no_grad():
            from tqdm import tqdm
            for context, attn_mask in tqdm(zip(full_context, full_attn_mask), total=len(full_context)):
                qnum = context.shape[0]
                context = context.to(device)
                attn_mask = attn_mask.to(device) if attn_mask is not None else None

                qchunk = query_features[offset: offset + qnum].to(device)

                with torch.autocast(device_type=device.type):
                    for start in range(0, num_cands, batch_size):
                        end = min(start + batch_size, num_cands)
                        cand_chunk = candidate_features[start:end].to(device)

                        # Get attended document embeddings
                        doc_embeds_chunk = F.normalize(
                            self.doc_attention(
                                cand_chunk, context, mode="batch2batch", attn_mask=attn_mask
                            ),
                            dim=-1,
                        )

                        # Compute similarities for this slice
                        sim_matrix[offset: offset + qnum, start:end] = torch.einsum(
                            "id,jid->ij", qchunk, doc_embeds_chunk
                        )

                offset += qnum

        # Sanity check: ensure we filled the entire matrix
        assert offset == num_queries, (
            f"Offset mismatch: wrote {offset} rows, "
            f"but expected {num_queries}. Check context segmentation."
        )

        return sim_matrix.contiguous()