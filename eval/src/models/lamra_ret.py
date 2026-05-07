from typing import Tuple, Optional, List, Union, Dict
import torch 
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import logging
from transformers import AutoProcessor, Qwen3VLConfig
from transformers import Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

from dataclasses import dataclass

# from peft import PeftModel, PeftConfig

from PIL import Image
from urllib.parse import urlparse
from qwen_vl_utils.vision_process import process_vision_info

import os
import unicodedata
import numpy as np

logger = logging.get_logger(__name__)


@dataclass
class MultiLayerLossOutput(SequenceClassifierOutput):
    layer_losses: Optional[Dict[str, torch.FloatTensor]] = None


def block_normalize(x, num_blocks=2):
    d = x.size(-1)
    assert d % num_blocks == 0, "Embedding dim must be divisible by num_blocks."
    blocks = torch.split(x, d // num_blocks, dim=-1)
    blocks = [F.normalize(b, dim=-1) for b in blocks]
    return torch.cat(blocks, dim=-1)


class Similarity(nn.Module):
    def __init__(self, temp=0.07, use_doc_attention=False):
        super().__init__()
        self.temp = temp
        self.use_doc_attention = use_doc_attention
        self.cos = nn.CosineSimilarity(dim=-1)
        
    def batch_cosine_sim(self, q_norm: torch.Tensor, c_norm: torch.Tensor) -> torch.Tensor:
        B = q_norm.shape[0]
        qi_vs_dji_sims = torch.einsum("qd,tqd->qt", q_norm, c_norm) 
        qi_vs_dij_sims = torch.einsum("qd,qtd->qt", q_norm, c_norm) 
        
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
    def __init__(self, config: Qwen3VLConfig, use_doc_attention: bool = False, embed_dim=3584, pad_token_id=None):
        super().__init__(config)
        self.pad_token_id = pad_token_id

    def forward2(
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

        embed_index = self.config.emb_token_ids[0]
        embed_indices = torch.argmax((labels == embed_index).int(), dim=1)         
        embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1]

        return embed_features, hidden_states, attention_mask

def sample_frames(frames: List[Union[str, Image.Image]], max_segments: int) -> List[Union[str, Image.Image]]:
    duration = len(frames)
    if duration <= max_segments:
        return frames

    frame_id_array = np.linspace(0, duration - 1, max_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    sampled_frames = [ frames[frame_idx] for frame_idx in frame_id_list ]
    return sampled_frames

def is_image_path(path: str) -> bool:
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
    
    if path.startswith(('http://', 'https://')):
        # Parse URL to remove query parameters
        parsed_url = urlparse(path)
        clean_path = parsed_url.path
    else:
        clean_path = path
    
    # Check file extension
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions

def is_video_input(video) -> bool:
    if isinstance(video, str):
        return True
    
    if isinstance(video, list) and len(video) > 0:
        # Check first element to determine the type
        first_elem = video[0]
        
        if isinstance(first_elem, Image.Image):
            return True
        
        if isinstance(first_elem, str):
            return is_image_path(first_elem)
    
    return False

# =====================================================================
# EVALUATION WRAPPERS
# =====================================================================

class Qwen3VLEncoderWrapper:
    """
    Acts as the `model.encoder` object expected by the evaluation harness.
    Bypasses the custom training forward pass to extract representations cleanly.
    """
    def __init__(self, model: Qwen3VLRetFinetuneForConditionalGeneration, processor):
        self.model = model
        self.processor = processor
        self.min_pixels=256 * 28 * 28,  # The Floor: ~200k pixels
        self.max_pixels=1024 * 28 * 28  # The Ceiling: ~1M pixels

    def format_model_input(
        self, 
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict]:

        # Ensure instruction ends with punctuation
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
                instruction = instruction + '.'

        # Initialize conversation with system prompts
        content = []
        conversation = [
            # {"role": "system", "content": [{"type": "text", "text": instruction or self.default_instruction}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": "<emb>."}
        ]

        # Normalize text input to list
        if text is None:
            texts = ["\nSummarize above image in one word: "]
        elif isinstance(text, str):
            inst = instruction if instruction else self.default_instruction
            texts = [' ' + inst + '\n' + text + "\nSummarize above sentence in one word: "]
        else:
            raise Exception
            texts = text
        
        # Normalize image input to list
        if image is None:
            images = []
        elif not isinstance(image, list):
            images = [image]
        else:
            images = image
        
        # Normalize video input to list
        if video is None:
            videos = []
        elif is_video_input(video):
            videos = [video]
        else:
            # Assume it's a list of videos
            videos = video

        # Add text, image, or video content to conversation
        if not texts and not images and not videos:
            content.append({'type': 'text', 'text': "NULL"})
            return conversation

        # Process each video
        for vid in videos:
            video_content = None
            video_kwargs = {'total_pixels': self.total_pixels}
            
            if isinstance(vid, list):
                # Video as frame sequence
                video_content = vid
                if self.max_frames is not None:
                    video_content = sample_frames(video_content, self.max_frames)
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele) 
                    for ele in video_content
                ]
            elif isinstance(vid, str):
                # Video as file path
                video_content = vid if vid.startswith(('http://', 'https://')) else 'file://' + vid
                video_kwargs = {'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames}
            else:
                raise TypeError(f"Unrecognized video type: {type(vid)}")

            # Add video input to content
            if video_content:
                content.append({
                    'type': 'video', 
                    'video': video_content,
                    **video_kwargs
                })

        # Process each image
        for img in images:
            image_content = None
            
            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = img if img.startswith(('http://', 'https://')) else 'file://' + img
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")

            # Add image input to content
            if image_content:
                content.append({
                    'type': 'image', 
                    'image': image_content,
                    "max_pixels": self.max_pixels,
                })

        # Process each text
        for txt in texts:
            content.append({'type': 'text', 'text': txt})

        return conversation

    def _preprocess_inputs(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        images, video_inputs, video_kwargs = process_vision_info(
            conversations, image_patch_size=16,
            return_video_metadata=True, return_video_kwargs=True
        )

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self.processor(
            text=text, images=images, videos=videos, video_metadata=video_metadata, truncation=True, 
            padding=True, do_resize=False, return_tensors='pt',
            **video_kwargs
        )
        
        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        inputs["labels"] = labels
        return inputs

    def forward(self, processed_inputs):
        # Directly call the base LM to get hidden states without triggering NCE loss
        return self.model.forward2(
            input_ids=processed_inputs.get("input_ids"),
            attention_mask=processed_inputs.get("attention_mask"),
            pixel_values=processed_inputs.get("pixel_values"),
            pixel_values_videos=processed_inputs.get("pixel_values_videos"),
            image_grid_thw=processed_inputs.get("image_grid_thw"),
            video_grid_thw=processed_inputs.get("video_grid_thw"),
            labels=processed_inputs.get("labels"),
            output_hidden_states=True,
            return_dict=True,
        )


class MMEBEmbeddingModel:
    """
    The top-level API wrapper expected by your evaluation code.
    Handles LoRA loading and exposes the compute_similarity calculation.
    """
    def __init__(self, model, processor, normalize=True):
        self.model = model
        self.encoder = Qwen3VLEncoderWrapper(model, processor)
        self.normalize = normalize

    @property
    def device(self):
        return self.model.device

    @classmethod
    def load(cls, model_name_or_path, normalize=True, instruction=None, attn_implementation="sdpa", torch_dtype=torch.bfloat16):
        
        # # 1. Detect if the path points to a LoRA adapter
        # try:
        #     peft_config = PeftConfig.from_pretrained(model_name_or_path)
        #     base_model_path = peft_config.base_model_name_or_path
        #     is_lora = True
        #     logger.info(f"LoRA adapter detected. Loading base model: {base_model_path}")
        # except Exception:
        #     base_model_path = model_name_or_path
        #     is_lora = False
        
        # 2. Load the base custom model
        model = Qwen3VLRetFinetuneForConditionalGeneration.from_pretrained(
            model_name_or_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", max_pixels=1024 * 28 * 28)

        tokenizer = processor.tokenizer 
        tokenizer.model_max_length = 1024

        def add_embed_token(tokenizer, model, emb_token="<emb>"):
            emb_tokens = [emb_token]
            num_new_tokens = tokenizer.add_tokens(emb_tokens)
            assert len(emb_tokens) == num_new_tokens

            model.resize_token_embeddings(len(tokenizer))

            emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
            model.config.emb_token_ids = emb_token_ids

        add_embed_token(tokenizer, model)
        
        return cls(model, processor, normalize=normalize)

    def encode_input(self, batch_inputs):
        # Standard execution loop when include_tokens=False
        conversations = [
            self.encoder.format_model_input(
                text=ele.get("text"), image=ele.get("image"), video=ele.get("video"),
                instruction=ele.get("instruction"), fps=ele.get("fps"), max_frames=ele.get("max_frames")
            )
            for ele in batch_inputs
        ]
        processed_inputs = self.encoder._preprocess_inputs(conversations)
        processed_inputs = {k: v.to(self.device) for k, v in processed_inputs.items()}
        
        with torch.no_grad():
            anchors, _, _ = self.encoder.forward(processed_inputs)
        
        if self.normalize:
            anchors = F.normalize(anchors, p=2, dim=-1)
        return anchors

    def compute_similarity(self, qry, cand):
        qry = qry.to(torch.float64)
        cand = cand.to(torch.float64)
        return torch.matmul(qry, cand.T)