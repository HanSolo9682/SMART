from typing import Dict, Sequence

import torch

from . import register_collator
from .base import BaseDataCollator
from .qwen3_vision_process import process_vision_info


@register_collator("qwen3_vl_2b_hardneg")
class Qwen3VL2BHardNegDataCollator(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict], max_negatives=1000) -> Dict[str, torch.Tensor]:
        category_size = len(messages[0])
        if category_size == 3:
            has_hard_negative = True 
        else:
            has_hard_negative = False 
        
        new_messages = []
        doc2query_map = [] # List to track which query does each hard neg/pos belong to
        num_negatives = 0
        for category in range(category_size):
            for idx, item in enumerate(messages):
                if category == 0:
                    # Query
                    new_messages.append(item[category])
                    doc2query_map.append(('query', idx))
                elif category == 1:
                    # Positives
                    new_messages.extend(item[category])
                    doc2query_map.extend([('pos', idx)] * len(item[category]))
                elif category == 2:
                    if num_negatives >= max_negatives:
                        break
                    # Negatives
                    new_messages.extend(item[category])
                    doc2query_map.extend([('neg', idx)] * len(item[category]))
                    num_negatives += len(item[category])
        #             print(item[category])
        #             raise Exception
        # print(num_negatives)
        
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
            try:
                 inputs = self.processor(
                                        text=texts,
                                        images=image_inputs,
                                        videos=video_inputs,
                                        padding=True,
                                        truncation=True,
                                        return_tensors="pt",
                                    )
            except ValueError:
                print(f"Truncation error occurred. Reducing max negatives to {num_negatives - 5}.")
                return self.__call__(messages, max_negatives=num_negatives - 5)

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
        
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            has_hard_negative=has_hard_negative,
            ignore_token_id=self.IGNORE_TOKEN_ID,
            doc2query_map=doc2query_map,
            batch_size=len(messages),
            neg_start=find_neg_start(doc2query_map)
        )
        
        
def find_neg_start(pairs):
    for i, (label, _) in enumerate(pairs):
        if label == "neg":
            return i
    return None