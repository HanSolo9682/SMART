from typing import Tuple

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM

from . import register_loader
from .base import BaseModelLoader
from models.qwen3_vl import Qwen3VLRetForConditionalGeneration
from models.qwen3_vl_finetune import Qwen3VLRetFinetuneForConditionalGeneration
# from models.qwen3_vl_finetune_hardneg import Qwen3VLRetFinetuneWithHardNegatives
@register_loader("qwen3-vl-2b")
class Qwen3VL2BModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True, pretrain=True, cache_dir='weights') -> Tuple[AutoModelForCausalLM, AutoTokenizer, None]:
        if load_model and pretrain:
            model = Qwen3VLRetForConditionalGeneration.from_pretrained(
                self.model_local_path,
                cache_dir=cache_dir, 
                **self.loading_kwargs,
            ) 
        elif load_model and not pretrain:
            # if self.use_hard_negatives:
            #     model = Qwen3VLRetFinetuneWithHardNegatives.from_pretrained(
            #     self.model_local_path, 
            #     cache_dir=cache_dir,
            #     embed_dim=1536,
            #     **self.loading_kwargs,
            # ) 


            # else:
            model = Qwen3VLRetFinetuneForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                cache_dir=cache_dir,
                embed_dim=1536,
                late_method=self.late_method
                **self.loading_kwargs,
            ) 

        # processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            min_pixels=256 * 28 * 28,  # The Floor: ~200k pixels
            max_pixels=1024 * 28 * 28  # The Ceiling: ~1M pixels
        )
        tokenizer = processor.tokenizer

        self.add_embed_token(tokenizer, model)

        return model, tokenizer, processor 

    def add_embed_token(self, tokenizer, model, emb_token="<emb>"):
        emb_tokens = [emb_token]
        num_new_tokens = tokenizer.add_tokens(emb_tokens)
        #assert len(emb_tokens) == num_new_tokens
        if len(emb_tokens) == num_new_tokens:
            model.resize_token_embeddings(len(tokenizer))

            if emb_token=="<emb>":
                emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
                model.config.emb_token_ids = emb_token_ids