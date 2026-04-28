# TM-DLM: Topology-Masked Diffusion Language Model for Node Classification

This directory contains the training and evaluation scripts for applying masked diffusion language models (DLMs) to node classification on text-attributed graphs (TAGs).

## Overview

Instead of using GNN+LM pipelines or autoregressive LLMs, we use LLaDA (a masked discrete diffusion LM with bidirectional attention) to classify nodes by denoising a masked answer token conditioned on the node's text and optional neighbor context.

### How It Works

1. Each node is formatted as `category_infill` prompt:
   ```
   Paper: <title>. <abstract>
   Options: 0) Case Based 1) Genetic Algorithms 2) Neural Networks ...
   The category of this paper is: <class_name>
   ```
2. Neighbor node texts can include labels (recommended for latest SFT):
   ```
   Neighbor 1 [label: <class_name>]: <neighbor_title>. <neighbor_abstract>
   Neighbor 2: ...
   ```
3. We reserve a fixed answer window (`max_answer_tokens=6`) for infill/eval compatibility.
4. Training supervises only real answer tokens (`labels != -100`), and keeps reserved tail mask slots unsupervised (no eos-padding supervision).

## Key Concepts

### Topology Attention Mask

When neighbors are included, the **topology mask** controls which tokens can attend to each other:

- **Target node tokens**: can attend to all tokens (self + all neighbors)
- **Neighbor tokens**: can only attend to self + target node tokens
- **No cross-neighbor attention**: neighbors cannot see each other

This creates a star topology centered on the target node:

```
         Target
        /  |  \
      Nb1  Nb2  Nb3     (Nb1, Nb2, Nb3 cannot see each other)
```

The mask is implemented as a 4D additive attention mask `[b, 1, L, L]` where `0 = attend` and `-inf = block`, compatible with HuggingFace's attention interface.

Without the topology mask (`use_topology_mask=False`), all tokens attend to all tokens (full bidirectional attention).

### Topological Position IDs

By default, position IDs are sequential `[0, 1, ..., L-1]` across the concatenated sequence. This means neighbors appended later in the sequence get larger position IDs and are "farther" from the target in RoPE encoding.

With `--position_id_type topological`, each node's tokens get position IDs restarting from 0:

```
Sequential (default):
  pos: 0  1  2 ... 200  201 202 ... 400  401 ...
       |--- target ---|  |--- nb1 ---|  |--- nb2 ---|

Topological (per-node reset):
  pos: 0  1  2 ... 200  0   1   2 ... 199  0   1  ...
       |--- target ---|  |--- nb1 ---|    |--- nb2 ---|
```

This removes the concatenation-order bias in RoPE so all neighbors are positionally equidistant from the target. However, in frozen-model evaluation this significantly hurts performance (28.78% vs 50.74%) because the model was never pretrained with position resets. It may be useful when combined with SFT.

### Diffusion Training (SFT)

The SFT process uses the standard masked diffusion ELBO loss from the dllm framework:

1. Sample a random timestep `t`
2. Stochastically mask supervised answer tokens (`labels != -100`) with probability `1 - alpha(t)`
3. Forward pass through the model with the topology mask applied
4. Compute weighted cross-entropy at the masked position
5. Optionally add an auxiliary classification CE loss on the answer position

In latest `category_infill` training, supervised positions include:
- target class-name tokens
- (optional) neighbor class-name tokens when `include_neighbor_labels=True`

## Scripts

### `run_experiments.py` - Frozen Model Evaluation

Evaluates the pretrained LLaDA model without any fine-tuning. Masks the answer position, runs a single forward pass, and does restricted argmax over digit tokens.

```bash
# Target-only (no neighbors)
CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/run_experiments.py \
    --exp mc_target_only \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_name cora \
    --max_hops 0

# 1-hop neighbors + topology mask
CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/run_experiments.py \
    --exp mc_1hop_topo_mask \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_name cora \
    --max_hops 1 \
    --use_topology_mask True

# 1-hop + topology mask + topological position IDs
CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/run_experiments.py \
    --exp mc_1hop_topo_posid \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_name cora \
    --max_hops 1 \
    --use_topology_mask True \
    --position_id_type topological

# Evaluate a LoRA-finetuned checkpoint
CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/run_experiments.py \
    --exp sft_ep3_eval \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --lora_path .models/tmdlm-llada-8b-cora-na-na-target-ckpt/checkpoint-306 \
    --dataset_name cora \
    --max_hops 0
```

Key arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--max_hops` | 2 | 0 = target-only, 1 = 1-hop neighbors, 2 = 2-hop |
| `--use_topology_mask` | False | Apply star-topology attention mask |
| `--position_id_type` | sequential | `sequential` or `topological` (per-node reset) |
| `--lora_path` | None | Path to LoRA checkpoint for evaluating SFT models |
| `--denoising_steps` | 1 | Number of iterative denoising steps |

### `sft.py` - LoRA Fine-Tuning

Fine-tunes LLaDA with LoRA on the training split using masked diffusion loss.

```bash
CUDA_VISIBLE_DEVICES=0 python /home/lingjie7/auto-research/projects/dlm-graph/examples/tmdlm/sft.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_name cora \
    --max_hops 2 \
    --use_topology_mask True \
    --max_neighbors_per_hop 10 \
    --prompt_format category_infill \
    --max_answer_tokens 6 \
    --include_neighbor_labels True \
    --neighbor_label_format bracket \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 20 \
    --output_dir .models/tmdlm-llada-8b-cora-2hop-topo-catinfill-nbmask-noeospad-r64-ep20 \
    --gradient_checkpointing True \
    --cls_loss_weight 1.0 \
    --lora True --r 64 --lora_alpha 64 --target_modules all-linear
```

Latest paired launcher (topo + notopo):

```bash
# Cora only
GPUS=0,1 DATASETS=cora \
  bash /home/lingjie7/auto-research/projects/dlm-graph/examples/tmdlm/run_sft_cora_hops_topo_lora.sh

# Cora then PubMed
GPUS=0,1 DATASETS=cora,pubmed \
  bash /home/lingjie7/auto-research/projects/dlm-graph/examples/tmdlm/run_sft_cora_hops_topo_lora.sh
```

Key training config (latest recipe):
| Parameter | Value |
|-----------|-------|
| Prompt format | `category_infill` |
| Max hops | 2 |
| Max neighbors per hop | 10 |
| Max answer tokens | 6 |
| Neighbor labels in prompt | True (`bracket`) |
| LoRA rank | 64 |
| LoRA alpha | 64 |
| Target modules | all-linear |
| Learning rate | 5e-5 |
| Effective batch size | 16 (bs=2 x grad_accum=8) |
| Epochs | 20 |
| cls_loss_weight | 1.0 |

## Datasets

| Dataset | Classes | Train | Val | Test | Source |
|---------|---------|-------|-----|------|--------|
| Cora | 7 | 1624 | 542 | 542 | HuggingFace (xxhe/tape-cora) + PyG |
| PubMed | 3 | 60 | 500 | 999 | Local TAPE files (custom stratified split) |

## Code Structure

```
dllm/
  data/graph.py              # Dataset loading, prompt construction (build_node_sample)
  pipelines/tmdlm/
    trainer.py               # TMDLMTrainer: topology mask injection + aux classification loss
    utils.py                 # GraphDataCollator: padding, topology mask, position IDs
    sampler.py               # TMDLMSampler: iterative denoising for inference
examples/tmdlm/
    sft.py                   # LoRA fine-tuning entry point
    run_experiments.py       # Frozen model evaluation
    eval.py                  # SFT model evaluation (using sampler)
    results.md               # Full evaluation results with baselines
```

## Results

See [/home/lingjie7/auto-research/projects/dlm-graph/examples/tmdlm/results.md](/home/lingjie7/auto-research/projects/dlm-graph/examples/tmdlm/results.md) for full evaluation results and baseline comparisons.

Summary (Cora, supervised setting):

| Method | Accuracy |
|--------|----------|
| LLaGA-HO-7B | 89.22% |
| SAGN | 89.19% |
| GAT | 88.97% |
| GCN | 88.93% |
| GraphSAGE | 88.89% |
| LLaGA-ND-7B | 88.86% |
| NodeFormer | 88.23% |
| SGC | 87.97% |
| **Ours: SFT (2-hop + topo, label-on, eval_logit)** | **90.41%** |
| Ours: SFT (2-hop + no topo, label-on, eval_logit) | 90.22% |
| Ours: SFT (2-hop + topo, label-on, eval_infill lenient) | 87.45% |
| Ours: SFT (2-hop + no topo, label-on, eval_infill lenient) | 89.48% |
| Ours: Frozen (2-hop + topo, label-on, eval_logit) | 37.82% |
| Ours: Frozen (2-hop + no topo, label-on, eval_logit) | 33.03% |
| Ours: Frozen (2-hop + topo, label-on, eval_infill lenient) | 57.01% |
| Ours: Frozen (2-hop + no topo, label-on, eval_infill lenient) | 60.52% |
| Ours: Frozen MC (target-only, legacy setting) | 62.73% |

SFT source: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_noeospad_allckpts_eval_gpu01_20260425_164435/summary.csv`.
Frozen 2-hop label-on source: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_frozen_labelon_newest_gpu26_20260425_2102/jsonl`.
