# SMART
[[arXiv]()] [[Huggingface](https://huggingface.co/collections/HanSolo9682/smart)]

This is the official repository for the paper Your Embedding Model is SMARTer Than You Think. We open source the code used to train our LamRA-Ret variants, our Qwen3-VL-Embedding adapters, and for evaluating baselines and our newly trained models.

## Training Data
In order to use LamRA, we need to first convert Colpali into the desired data format. First, clone from the official Colpali training set repository:

```bash
git clone https://huggingface.co/datasets/vidore/colpali_train_set
```

Then, run the following two lines to generate the query and candidate jsonl files:

```bash
python load_colpali.py path/to/colpali_training_set
python convert_colpali.py path/to/colpali_training_set
```
## LamRA-Ret Training
All code under `train_lamra` folder are built upon the official [LamRA](https://github.com/Code-kunkun/LamRA) repository. We here only include code that allows training from the Qwen3-VL-2B-Instruct backbone.

### Usage
See `train_lamra/scripts/lamra_ret/finetune_visdoc.sh` for the full training setup. Important things to edit include:
- `QUERY_DATA_PATH` and `CAND_POOL_PATH` (generated from the previous section)
- `late_method`, which should only be either `disabled` to disable late-interaction and only use the single-vector objective, `late_only` to only use the multi-vector objective, or `hybrid` to use the hybrid scoring objective.

Note: the query instruction file is just an empty table. In the previous section, we have already merged the instructions into each query/candidate text to match LamRA's data collator.

## Adapter Training
We only experimented with training an adapter on the SoTA Qwen3-VL-Embedding models, but feel free to build your custom adapter for other models building on top of ours.

### Usage
See `train_adapter/training_visdoc_2/8b.sh` for the full training setup. Important things to edit include:
- `TRAIN_DATA`: the Colpali repository you cloned (do not use the LamRA format here).
- `USE_LAST_N_LAYERS`: we use `1` for our adapters as we only use the last layer and it already shows SoTA multi-vector performance; feel free to experiment with more layers.


## Evaluation
All code under `eval` folder are built upon the official [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) repository.

### Usage
Before using the evaluation script, please follow the official [VLM2Vec-V2](https://github.com/TIGER-AI-Lab/VLM2Vec) repository in downloading and setting up MMEB-V2.

We include all the evaluation scripts in `eval/scripts/evaluation/mmeb_v2/eval_xxx_late_interaction.sh`. Important things to edit include:
- `DATA_BASEDIR`: where you stored your MMEB-V2 images.
- `ENABLE_LATE_INTERACTION`: enables or disables late interaction. If disabled, only single-vector/anchor is used during eval.
- `LAMBDA_ANCHOR`: the weight of the single vector embedding similarity during hybrid scoring, defaults to 1. If 0, then only late interaction is used.
- `LAMBDA_LATE`: the weight of the multi vector embedding similarity scores during hybrid scoring, defaults to 1. If 0, then only the single vector embedding is used.
- `LATE_QUERY/CANDIDATE_CHUNK_SIZE`: choose wisely for your environment to avoid OOM errors.

### Supported Models
This repository current supports the evaluation of the following baseline embedders:
- `Qwen3-VL-Embedding-2/8B` w/ or w/o adapter
- `LamRA`
- `GME-2/7B`
- `jina-embeddings-v4`

## Citation
If you find SMART helpful, we would appreciate it if you can cite us through the following:
```bibtex
@misc{zhang2026smart,
    ...
}
```