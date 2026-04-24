# DLM-Graph: Diffusion Language Models for Text-Attributed Graph Learning

Applying masked diffusion language models to node classification on text-attributed graphs (TAGs). Built on the [dLLM](https://github.com/ZHZisZZ/dllm) framework with [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).

## Overview

Instead of using GNN+LM pipelines or autoregressive LLMs, we use LLaDA (a masked discrete diffusion LM with bidirectional attention) to classify nodes by denoising a masked answer token conditioned on the node's text and optional neighbor context.

Each node is formatted as a multiple-choice prompt:
```
Paper: <title>. <abstract>
Options: 0) Case Based 1) Genetic Algorithms 2) Neural Networks ...
Answer: [MASK]
Neighbor 1: <neighbor_title>. <neighbor_abstract>
...
```

The model predicts the masked answer digit via forward pass (frozen) or diffusion denoising (SFT with LoRA).

## Key Components

### Topology Attention Mask

Controls which tokens can attend to each other when neighbors are included:
- **Target node**: attends to all tokens (self + all neighbors)
- **Neighbor nodes**: attend to self + target only (no cross-neighbor attention)

```
         Target
        /  |  \
      Nb1  Nb2  Nb3     (star topology, no cross-neighbor edges)
```

### Topological Position IDs

Optional per-node RoPE position reset (`--position_id_type topological`), removing concatenation-order bias so all neighbors are positionally equidistant from the target.

```
Sequential:    0  1  ... 200  201 ... 400  401 ...
Topological:   0  1  ... 200  0   ... 199  0   ...
               |-- target --|  |-- nb1 --|  |-- nb2 --|
```

## Quick Start

### Setup

```bash
conda create -n dllm python=3.10 -y && conda activate dllm
conda install cuda=12.4 -c nvidia
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

### Frozen Model Evaluation (no training)

```bash
# Target-only
CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/run_experiments.py \
    --exp mc_target_only \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_name cora --max_hops 0

# 1-hop neighbors + topology mask
CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/run_experiments.py \
    --exp mc_1hop_topo_mask \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_name cora --max_hops 1 --use_topology_mask True
```

### SFT with LoRA

```bash
CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/sft.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_name cora --max_hops 0 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 --num_train_epochs 5 \
    --output_dir .models/tmdlm-llada-8b-cora-na-na-target-ckpt \
    --gradient_checkpointing True --cls_loss_weight 0.0
```

## Results

### Cora (7 classes, 542 test nodes, supervised setting)

Baselines from arXiv:2502.00829, Table 2.

| Method | Type | Accuracy |
|--------|------|----------|
| GCN + LLM Emb | GNN + LLM embeddings | 88.15% |
| TAPE | LLM-as-Reasoner | 88.05% |
| LLaGA | LLM + Graph Projector | 87.55% |
| GraphSAGE | GNN | 87.44% |
| GCN | GNN | 87.41% |
| **Ours: SFT (1-hop + topo mask)** | **DLM + LoRA (best: checkpoint-408)** | **84.87%** |
| Ours: SFT (1-hop, no topo mask) | DLM + LoRA (best: checkpoint-816) | 84.13% |
| Ours: SFT (2-hop + topo mask) | DLM + LoRA (best: checkpoint-510) | 83.95% |
| Ours: SFT (2-hop, no topo mask) | DLM + LoRA (best: checkpoint-510) | 84.50% |
| RoBERTa-355M | LM only | 83.17% |
| Ours: Frozen MC | DLM zero-shot | 62.73% |

### PubMed (3 classes, zero-shot)

| Method | Type | Accuracy |
|--------|------|----------|
| RoBERTa-355M | LM only, supervised | 94.84% |
| TAPE | supervised | 93.00% |
| GCN | supervised | 89.01% |
| **Ours: Frozen MC** | **DLM zero-shot** | **88.69%** |

See [examples/tmdlm/results.md](examples/tmdlm/results.md) for full results.

### Neighbor Count Sweep (`nb` = 1/3/5/10/20)

Open-ended category-infill runs logged in `experiments/experiment_log.jsonl`
(`openended_*_nb*`, `target_first`, `use_topology_mask=False`).
Metric shown is `accuracy_strict` (%).

| Dataset | Hops | nb=1 | nb=3 | nb=5 | nb=10 | nb=20 |
|---------|------|------|------|------|-------|-------|
| Cora | 1 | 60.52 | 60.33 | 61.62 | 61.44 | 61.44 |
| Cora | 2 | 57.93 | 63.84 | 65.13 | 64.21 | 64.76 |
| Cora | 3 | 57.93 | 63.84 | 65.13 | 64.21 | 64.76 |
| OGBN-Arxiv | 1 | 55.10 | 58.10 | 58.90 | 59.20 | 58.90 |
| OGBN-Arxiv | 2 | 56.30 | 59.10 | 61.10 | 58.70 | 57.60 |
| OGBN-Arxiv | 3 | 56.30 | 59.10 | 61.10 | 58.70 | 57.60 |
| OGBN-Products | 1 | 62.50 | 63.20 | 64.70 | 64.50 | 64.20 |
| OGBN-Products | 2 | 62.70 | 64.30 | 64.80 | 65.10 | 66.30 |
| OGBN-Products | 3 | 62.70 | 64.30 | 64.80 | 65.10 | 66.30 |
| PubMed | 1 | 74.77 | 80.98 | 81.68 | 81.78 | 82.08 |
| PubMed | 2 | 85.19 | 88.89 | 88.99 | 88.89 | 90.29 |
| PubMed | 3 | 85.19 | 88.89 | 88.99 | 88.89 | 90.29 |

### Neighbor Count Sweep (`nb` = 1/3/5/10/20, `use_topology_mask=True`)

Open-ended category-infill runs from
`.logs/topo_nb_sweep_debug_direct/records/*.jsonl` (run id: `debug_direct`).
Metric shown is `accuracy_strict` (%).

| Dataset | Hops | nb=1 | nb=3 | nb=5 | nb=10 | nb=20 |
|---------|------|------|------|------|-------|-------|
| Cora | 1 | 60.89 | 59.41 | 60.52 | 60.33 | 60.33 |
| Cora | 2 | 57.01 | 61.44 | 62.55 | 62.55 | 62.73 |
| Cora | 3 | 57.01 | 61.44 | 62.55 | 62.55 | 62.73 |
| PubMed | 1 | 74.97 | 76.78 | 77.58 | 77.88 | 77.98 |
| PubMed | 2 | 77.08 | 81.88 | 82.78 | 82.28 | 82.68 |
| PubMed | 3 | 77.08 | 81.88 | 82.78 | 82.28 | 82.68 |

`OGBN-Arxiv` and `OGBN-Products` topology-mask sweep rows are still running and will be appended after completion.

## Project Structure

```
dllm/
  data/graph.py                    # TAG dataset loading (Cora, PubMed)
  pipelines/tmdlm/
    trainer.py                     # TMDLMTrainer (topology mask + aux loss)
    utils.py                       # GraphDataCollator (topology mask, position IDs)
    sampler.py                     # Iterative denoising for inference
  pipelines/llada/models/
    modeling_llada.py               # LLaDA model (extended with position_ids)
  core/                            # Base dllm framework (MDLMTrainer, schedulers, etc.)

examples/tmdlm/
  sft.py                           # LoRA fine-tuning
  run_experiments.py               # Frozen model evaluation
  eval.py                          # SFT model evaluation
  results.md                       # Full results with baselines
  README.md                        # Detailed documentation
```

## Datasets

| Dataset | Classes | Train | Val | Test | Source |
|---------|---------|-------|-----|------|--------|
| Cora | 7 | 1624 | 542 | 542 | xxhe/tape-cora + PyG |
| PubMed | 3 | 60 | 500 | 999 | Local TAPE files |

## Acknowledgments

Built on [dLLM](https://github.com/ZHZisZZ/dllm) by Zhou et al. Baseline results from ["When Do LLMs Help With Node Classification?"](https://arxiv.org/abs/2502.00829).
