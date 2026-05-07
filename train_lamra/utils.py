import os
import math
from typing import List, Dict, Optional

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import transformers
from transformers import Trainer
from transformers.trainer import has_length
import logging
import wandb

logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track the last logged step to prevent redundant logging 
        # during gradient accumulation steps.
        self._last_logged_step = -1

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Define allowed classes (for PEFT compatibility)
        try:
            from peft import PeftModel, is_peft_model
            supported_classes = (torch.nn.Module, PeftModel)
        except ImportError:
            supported_classes = (torch.nn.Module,)

        # Handle non-PreTrainedModel saving
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            model_to_save = self.accelerator.unwrap_model(self.model)
            if isinstance(model_to_save, supported_classes):
                model_to_save.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                    save_embedding_layers=True,  # 👈 CUSTOM ARG HERE
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    import safetensors.torch
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, "model.safetensors"), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        else:
            # For PreTrainedModel / PeftModel
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
                save_embedding_layers=True,  # 👈 CUSTOM ARG HERE
            )

        # Save tokenizer as usual
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        outputs = model(**inputs)
        
        # Standard HF Trainer expects a `loss` attribute or key
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
        
        # Extract our custom layer_losses
        layer_losses = outputs.get("layer_losses") if isinstance(outputs, dict) else getattr(outputs, "layer_losses", None)
        
        # Log to wandb if on the main process
        if layer_losses is not None and self.is_world_process_zero():
            # Check if we've hit a logging step
            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                if self.state.global_step != self._last_logged_step:
                    if wandb.run is not None:
                        # commit=False ensures these custom metrics are bundled 
                        # with the default HF Trainer metrics (like epoch, learning_rate, etc.)
                        wandb_logs = {f"train/loss_layer_{k}": v.item() for k, v in layer_losses.items()}
                        wandb.log(wandb_logs, commit=False)
                        
                    self._last_logged_step = self.state.global_step

        return (loss, outputs) if return_outputs else loss


class NoTextOnlyBatchSampler(Sampler):
    r"""
    Sampler that tries its best to sample batches such that no batch has only 
    text (unimodal) data. This is necessary for training with deepspeed. 
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        is_text_only: Optional[List[bool]] = None,
        generator=None,
    ):
        if is_text_only is None:
            raise ValueError("`is_text_only` must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.is_text_only = is_text_only
        self.generator = generator
        self.mega_batch_size = batch_size * world_size

    def __len__(self):
        return len(self.is_text_only)

    def __iter__(self):
        # mm: multimodal, entry that has both text and image/video
        # uni: unimodal, entry that has only text
        mm_indices = [i for i, is_text_only in enumerate(self.is_text_only) if not is_text_only]
        uni_indices = [i for i, is_text_only in enumerate(self.is_text_only) if is_text_only]

        num_batches = math.ceil((len(mm_indices) + len(uni_indices)) / self.mega_batch_size)
        if len(mm_indices) < num_batches:
            raise ValueError(
                f"{len(mm_indices)} multimodal entries, {len(num_batches)} batches. "
                "Not enough multimodal data in the dataset, or the batch size is too small. " 
                "There will be at least one batch that is text-only, which doesn't work with deepspeed. "
                "Try increasing the batch size first."
            )

        # shuffle indices
        mm_indices = [mm_indices[i] for i in torch.randperm(len(mm_indices), generator=None).tolist()]
        uni_indices = [uni_indices[i] for i in torch.randperm(len(uni_indices), generator=None).tolist()]

        # distribute indices into batches
        num_uni_indices_in_mega_batch = [len(uni_indices) // num_batches] * num_batches
        for i in range(len(uni_indices) % num_batches):
            num_uni_indices_in_mega_batch[i] += 1
        
        mega_batches = []
        cur_uni_index = 0
        cur_mm_index = 0
        for i, num_uni_indices in enumerate(num_uni_indices_in_mega_batch):
            mega_batch = []
            mega_batch.extend(uni_indices[cur_uni_index:cur_uni_index + num_uni_indices])
            cur_uni_index += num_uni_indices
            assert len(mega_batch) < self.mega_batch_size

            if i < num_batches - 1:
                increment = self.mega_batch_size - len(mega_batch)
                mega_batch.extend(
                    mm_indices[cur_mm_index:cur_mm_index + increment]
                )
                cur_mm_index += increment
            else: # last batch
                mega_batch.extend(mm_indices[cur_mm_index:])
                assert len(mega_batch) <= self.mega_batch_size, "Last batch is too big."
            
            mega_batches.append(mega_batch)
        
        mega_batch_indices = torch.randperm(len(mega_batches), generator=self.generator)
        mega_batches = [mega_batches[i] for i in mega_batch_indices]
        indices = [i for mega_batch in mega_batches for i in mega_batch]
        return iter(indices)


class TrainerWithCustomSampler(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        is_text_only = self.train_dataset.is_text_only
        return NoTextOnlyBatchSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            is_text_only=is_text_only,
        )
    
    def _get_eval_sampler(self, eval_dataset: torch.utils.data.Dataset) -> Optional[torch.utils.data.Sampler]:
        is_text_only = eval_dataset.is_text_only
        return NoTextOnlyBatchSampler(
            self.args.eval_batch_size,
            world_size=self.args.world_size,
            is_text_only=is_text_only,
        )


def find_all_linear_names(named_modules: Dict, target_modules: List[str]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in named_modules.items():
        if not any([module_name in name for module_name in target_modules]):
            continue

        if isinstance(module, cls):
            lora_module_names.add(name)

    for name in list(lora_module_names):
        if 'lm_head' in name: # needed for 16-bit
            lora_module_names.remove(name)

    return list(lora_module_names)


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def merge_and_save_all(trainer, output_dir: str, tokenizer=None, processor=None):
    """Handles merging + saving full model and associated assets, even with DeepSpeed."""

    os.makedirs(output_dir, exist_ok=True)

    # Always sync CUDA to avoid race conditions
    torch.cuda.synchronize()

    # Get base model
    model = trainer.model
        
    if hasattr(model, "merge_and_unload"):
        print("[Info] Merging LoRA adapters into base model...")
        model = model.merge_and_unload()
    elif hasattr(model, "get_base_model"):
        model = model.get_base_model()

    # Save full model
    print(f"[Info] Saving full model to {output_dir}")
    model.save_pretrained(output_dir)

    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    # Save processor (e.g., Qwen2VLProcessor)
    if processor is not None:
        processor.save_pretrained(output_dir)

    # # Save image processor if separate
    # if hasattr(trainer.model, "image_processor") and trainer.model.image_processor is not None:
    #     trainer.model.image_processor.save_pretrained(output_dir)

    print(f"[✓] Full model and assets saved to {output_dir}")