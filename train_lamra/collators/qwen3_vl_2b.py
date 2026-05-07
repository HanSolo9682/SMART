from typing import Dict, Sequence

import torch

from . import register_collator
from .base import BaseDataCollator
from .qwen3_vision_process import process_vision_info


@register_collator("qwen3-vl-2b")
class Qwen3VL2BDataCollator(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        category_size = len(messages[0])
        if category_size == 3:
            has_hard_negative = True 
        else:
            has_hard_negative = False 
        
        new_messages = []
        for category in range(category_size):
            for item in messages:
                new_messages.append(item[category])

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in new_messages
        ]
        image_inputs, video_inputs = process_vision_info(new_messages)
        if self.use_doc_attention:
            inputs = self.processor(
                                    text=texts,
                                    images=image_inputs,
                                    videos=video_inputs,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt",
                                )
        else:
            inputs = self.processor(
                                    text=texts,
                                    images=image_inputs,
                                    videos=video_inputs,
                                    # padding=True,
                                    padding='max_length',
                                    max_length=1024,
                                    return_tensors="pt",
                                )

        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        labels[labels == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID

        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
        else:
            attention_mask = None 
        if 'pixel_values' in inputs:
            pixel_values = inputs['pixel_values']
        else:
            pixel_values = None 
        if 'image_grid_thw' in inputs:
            image_grid_thw = inputs['image_grid_thw']
        else:
            image_grid_thw = None
        if "position_ids" in inputs:
            position_ids = inputs["position_ids"]
        else:
            position_ids = None
        
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            has_hard_negative=has_hard_negative,
            ignore_token_id=self.IGNORE_TOKEN_ID
        )