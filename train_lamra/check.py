from safetensors.torch import load_file
import os

ckpt_dir = "checkpoints/attention/qwen2-vl-7b_LamRA-Ret_doc_attn"

# Load the index file (tells us which param is in which shard)
index_file = os.path.join(ckpt_dir, "model.safetensors.index.json")
import json
with open(index_file, "r") as f:
    index = json.load(f)

# Collect all parameter shards into one dict
full_state = {}
for shard_file in index["weight_map"].values():
    path = os.path.join(ckpt_dir, shard_file)
    if not os.path.exists(path):
        continue
    print(f"Loading {path} ...")
    shard_tensors = load_file(path)
    full_state.update(shard_tensors)

# Print all parameter names and shapes
for k, v in full_state.items():
    print(f"{k:60s} {tuple(v.shape)}")
