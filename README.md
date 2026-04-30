# DLM-Graph: Diffusion Language Models for Text-Attributed Graph Learning

Applying masked diffusion language models to node classification on text-attributed graphs (TAGs). Built on the [dLLM](https://github.com/ZHZisZZ/dllm) framework with [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).

## Overview

Instead of using GNN+LM pipelines or autoregressive LLMs, we use LLaDA (a masked discrete diffusion LM with bidirectional attention) to classify nodes by denoising a masked answer token conditioned on the node's text and optional neighbor context.

Each node is formatted as a category-infill prompt:

```
Paper: <title>. <abstract>
Answer: [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]
Neighbor 1 [<class>]: <neighbor_title>. <neighbor_abstract>
...
```

The model fills in the masked answer region via forward pass (frozen logit scoring) or iterative denoising (SFT with LoRA).

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

### Frozen Evaluation (no training)

```bash
# Logit scoring: single forward pass, argmax over class token logits
CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/eval_logit.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_name cora --max_hops 2 \
    --prompt_format category_infill --max_answer_tokens 6 \
    --include_neighbor_labels True --neighbor_label_format bracket

# Infill scoring: iterative denoising (10 steps)
CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/eval_infill.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_name cora --max_hops 2 \
    --prompt_format category_infill --max_answer_tokens 6 \
    --include_neighbor_labels True --neighbor_label_format bracket \
    --steps 10
```

### SFT with LoRA

Dataset-specific paired launchers (topo + no-topo, one GPU each):

```bash
# Cora
GPUS=0,1 bash examples/tmdlm/run_sft_cora_hops_topo_lora.sh

# PubMed
GPUS=0,1 bash examples/tmdlm/run_sft_pubmed_lora.sh

# ogbn-arxiv  (uses max_steps instead of num_train_epochs)
GPUS=0,1 bash examples/tmdlm/run_sft_arxiv_lora.sh
```

Key shared defaults: `--prompt_format category_infill`, `--include_neighbor_labels True`, `--neighbor_label_format bracket`, `--lora True --r 64 --lora_alpha 64 --target_modules all-linear`, `--learning_rate 5e-5`.

### Evaluation on SFT Checkpoints

```bash
# Logit eval on all checkpoints of a run
CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/eval_logit.py \
    --lora_path .models/<run_name>/checkpoint-<N> \
    --dataset_name cora --log_file /tmp/eval-out.jsonl [...]

# Infill eval (iterative denoising)
CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/eval_infill.py \
    --lora_path .models/<run_name>/checkpoint-<N> \
    --steps 10 --max_samples 1000 [...]
```

## Results

Baselines from [LLaGA, Table 1 (Single Focus)](https://arxiv.org/pdf/2402.08170). Our SFT uses LLaDA-8B-Instruct + LoRA (r=64, all-linear), 2-hop neighbors (max 10/hop), category-infill prompt with neighbor labels.

### Summary

| Dataset | Setting | Eval | Best Acc | vs. Best Baseline |
|---------|---------|------|----------|-------------------|
| **Cora** | SFT · no-topo | Infill | **92.07%** | +2.85 pp vs. LLaGA-HO-7B (89.22%) |
| Cora | SFT · topo | Infill | 91.51% | +2.29 pp |
| Cora | SFT · no-topo | Logit | 91.33% | +2.11 pp |
| Cora | SFT · topo | Logit | 90.77% | +1.55 pp |
| Cora | Frozen | Infill | 57.01% | — |
| **PubMed** | SFT · no-topo | Logit | **95.18%** | +0.31 pp vs. GraphSAGE (94.87%) |
| PubMed | SFT · no-topo | Infill | 94.93% | — |
| PubMed | SFT · topo | Logit | 94.47% | — |
| PubMed | SFT · topo | Infill | 94.14% | — |
| PubMed | Frozen | Logit | 87.15% | — |

---

### Cora — Checkpoint Curve (10 epochs, LoRA r=64)

Eval on 542 test nodes. Checkpoints saved every 10% of training (204 steps).

| Checkpoint | Logit · no-topo | Logit · topo | Infill · no-topo | Infill · topo |
|-----------|----------------|-------------|-----------------|--------------|
| 204 | 82.66 | 82.47 | 89.11 | 87.64 |
| 408 | 87.82 | 88.93 | 90.04 | 89.67 |
| 612 | 89.85 | 89.67 | 89.48 | 89.67 |
| 816 | 90.59 | 90.41 | 91.14 | 91.14 |
| 1020 | 91.33 | **90.77** | 91.33 | **91.51** |
| 1224 | **91.33** | — | **92.07** | — |
| *LLaGA-HO-7B* | *89.22* | | | |
| *SAGN* | *89.19* | | | |
| *GCN* | *88.93* | | | |
| *RoBERTa-355M* | *83.17* | | | |
| *Frozen LLaDA-8B* | *57.01* | | | |

Topo runs evaluated up to ckpt-1020; no-topo runs evaluated up to ckpt-1224.

---

### PubMed — Checkpoint Curve (10 epochs, LoRA r=64)

Eval on full test split. Checkpoints saved every 5% of training (370 steps).

| Checkpoint | Logit · no-topo | Logit · topo | Infill · no-topo | Infill · topo |
|-----------|----------------|-------------|-----------------|--------------|
| 370 | 91.23 | 92.11 | 93.13 | 92.98 |
| 740 | 94.35 | 94.14 | **94.93** | 93.36 |
| 1110 | 93.61 | 92.37 | **94.93** | 94.27 |
| 1480 | **95.18** | **94.47** | 94.90 | **94.14** |
| *LLaGA-HO-7B* | *95.03* | | | |
| *GraphSAGE* | *94.87* | | | |
| *GCN* | *92.96* | | | |
| *GAT* | *92.33* | | | |
| *Frozen LLaDA-8B* | *87.15* | | | |

---

### ogbn-arxiv — in progress

SFT training on GPU7 (topo, LoRA r=64, max\_steps=7400, bs=6). Results to be added.

## Datasets

| Dataset | Classes | Train | Val | Test | Source |
|---------|---------|-------|-----|------|--------|
| Cora | 7 | 1,624 | 542 | 542 | [TAPE](https://github.com/XiaoxinHe/TAPE) + PyG |
| PubMed | 3 | ~11,800 | 500 | ~3,700 | TAPE files |
| ogbn-arxiv | 40 | ~90,000 | ~29,000 | ~48,000 | OGB |

## Project Structure

```
dllm/
  data/graph.py                    # TAG dataset loading (Cora, PubMed, ogbn-*)
  pipelines/tmdlm/
    trainer.py                     # TMDLMTrainer (topology mask + aux loss)
    utils.py                       # GraphDataCollator (topology mask, position IDs)
    sampler.py                     # Iterative denoising for inference
  pipelines/llada/models/
    modeling_llada.py              # LLaDA model (extended with position_ids)
  core/                            # Base dllm framework (MDLMTrainer, schedulers, etc.)

examples/tmdlm/
  sft.py                           # LoRA fine-tuning entry point
  eval_logit.py                    # Logit-based evaluation (frozen or SFT)
  eval_infill.py                   # Infill-based evaluation (iterative denoising)
  run_sft_cora_hops_topo_lora.sh   # Cora SFT launcher
  run_sft_pubmed_lora.sh           # PubMed SFT launcher
  run_sft_arxiv_lora.sh            # ogbn-arxiv SFT launcher

analysis/
  plot_cora_sft_lineplot.py        # Cora checkpoint accuracy curves
  plot_pubmed_sft_lineplot.py      # PubMed checkpoint accuracy curves
  plot_style.json                  # Shared visual style for all plots
  baselines_cora.json              # Cora baseline numbers
  baselines_pubmed.json            # PubMed baseline numbers
```

## Acknowledgments

Built on [dLLM](https://github.com/ZHZisZZ/dllm) by Zhou et al. Baseline results from ["When Do LLMs Help With Node Classification?"](https://arxiv.org/abs/2502.00829).
