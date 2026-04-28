# DLM-Graph Results

This file tracks evaluation results from 2026-04-27 onward. Paths are absolute so each run can be traced back to its raw stdout and JSONL output.

## Current Dataset Alignment

- Dataset source: LLaGA-aligned local cache
- Dataset root: `/home/lingjie7/auto-research/projects/dlm-graph/.datasets/llaga`
- Base model: `/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct`
- Default evaluation prompt format unless otherwise stated: `category_infill`
- Default neighborhood setting unless otherwise stated: `max_hops=2`, `max_neighbors_per_hop=10`
- Default neighbor labels unless otherwise stated: `include_neighbor_labels=True`, `neighbor_label_format=bracket`

## Frozen Base Model

| Date | Dataset | Split | Method | Topology Mask | Neighbor Labels | Accuracy | Elapsed | Per-class Accuracy | Output |
|---|---|---:|---|---|---|---:|---:|---|---|
| 2026-04-27 | PubMed | test | eval_logit | False | True | 87.15 | 1082s (0.30 hr) | Experimental: 68.23; Type 1: 87.98; Type 2: 96.42 | `/home/lingjie7/auto-research/projects/dlm-graph/summaries/pubmed_llaga_frozen_eval_20260427/pubmed_llaga_frozen_notopo_logit_labelon.jsonl` |
| 2026-04-27 | PubMed | test | eval_logit | True | True | 86.49 | 2833s (0.79 hr) | Diabetes Mellitus, Experimental: 72.04; Diabetes Mellitus Type 1: 87.42; Diabetes Mellitus Type 2: 93.25 | `/home/lingjie7/auto-research/projects/dlm-graph/summaries/pubmed_llaga_frozen_eval_20260427/pubmed_llaga_frozen_topo_logit_labelon.jsonl` |
| 2026-04-27 | Cora | test | eval_logit | False | True | 57.01 | 114s (0.03 hr) | Case Based: 0.00; Genetic Algorithms: 83.33; Neural Networks: 73.75; Probabilistic Methods: 90.00; Reinforcement Learning: 92.59; Rule Learning: 0.00; Theory: 0.00 | `/home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_llaga_frozen_eval_20260427/cora_llaga_frozen_notopo_logit_labelon.jsonl` |
| 2026-04-27 | Cora | test | eval_logit | True | True | 56.27 | 247s (0.07 hr) | Case Based: 1.56; Genetic Algorithms: 83.33; Neural Networks: 73.12; Probabilistic Methods: 84.44; Reinforcement Learning: 94.44; Rule Learning: 0.00; Theory: 0.00 | `/home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_llaga_frozen_eval_20260427/cora_llaga_frozen_topo_logit_labelon.jsonl` |
| 2026-04-27 | ogbn-arxiv | test | eval_logit | False | True | failed | — | torch_sparse missing | `/home/lingjie7/auto-research/projects/dlm-graph/summaries/ogbn_arxiv_llaga_frozen_eval_20260427/ogbn_arxiv_llaga_frozen_notopo_logit_labelon.jsonl` |
| 2026-04-27 | ogbn-arxiv | test | eval_logit | True | True | failed | — | torch_sparse missing | `/home/lingjie7/auto-research/projects/dlm-graph/summaries/ogbn_arxiv_llaga_frozen_eval_20260427/ogbn_arxiv_llaga_frozen_topo_logit_labelon.jsonl` |
| 2026-04-28 | ogbn-arxiv | test | eval_logit | False | True | 37.32 | 12094s (3.36 hr) | cs.CV: 96.08; cs.DC: 79.86; cs.FL: 69.55; cs.CG: 66.45; cs.CC: 63.19 | `/home/lingjie7/auto-research/projects/dlm-graph/summaries/ogbn_arxiv_llaga_frozen_eval_20260428/ogbn_arxiv_llaga_frozen_notopo_logit_labelon.jsonl` |
| 2026-04-28 | ogbn-arxiv | test | eval_logit | True | True | 37.58 | 13562s (3.77 hr) | cs.CV: 96.23; cs.DC: 85.23; cs.FL: 70.45; cs.CG: 69.33; cs.CC: 63.77 | `/home/lingjie7/auto-research/projects/dlm-graph/summaries/ogbn_arxiv_llaga_frozen_eval_20260428/ogbn_arxiv_llaga_frozen_topo_logit_labelon.jsonl` |
| 2026-04-28 | ogbn-products | test | eval_logit | False | True | 23.07 | 3874s (1.08 hr) | Grocery: 82.17; Arts & Crafts: 81.48; Cell Phones: 78.94; CDs: 74.33; Health: 64.60 | `/home/lingjie7/auto-research/projects/dlm-graph/summaries/ogbn_products_llaga_frozen_eval_20260428/ogbn_products_llaga_frozen_notopo_logit_labelon.jsonl` |
| 2026-04-28 | ogbn-products | test | eval_logit | True | True | 23.44 | 3885s (1.08 hr) | Arts & Crafts: 84.29; Grocery: 82.56; Cell Phones: 75.78; CDs: 70.68; Health: 67.20 | `/home/lingjie7/auto-research/projects/dlm-graph/summaries/ogbn_products_llaga_frozen_eval_20260428/ogbn_products_llaga_frozen_topo_logit_labelon.jsonl` |

### PubMed Frozen Base Details

- Experiment: `pubmed_llaga_frozen_notopo_logit_labelon`
- Model: `/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct`
- Dataset: `pubmed`
- Split: `test`
- Samples: `3944`
- Correct: `3437/3944`
- Accuracy: `87.15%`
- Elapsed time: `1082.1s`
- Config:
  - `batch_size=1`
  - `max_seq_len=2048`
  - `max_neighbors_per_hop=10`
  - `max_hops=2`
  - `use_topology_mask=False`
  - `lora_path=None`
  - `prompt_layout=target_first`
  - `use_chat_template=False`
  - `prompt_format=category_infill`
  - `answer_label_style=digit0`
  - `include_neighbor_labels=True`
  - `neighbor_label_format=bracket`
- Stdout: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/pubmed_llaga_frozen_eval_20260427/pubmed_llaga_frozen_notopo_logit_labelon.out`
- JSONL: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/pubmed_llaga_frozen_eval_20260427/pubmed_llaga_frozen_notopo_logit_labelon.jsonl`

### Cora Frozen Base Details

- Experiment: `cora_llaga_frozen_notopo_logit_labelon`
- Model: `/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct`
- Dataset: `cora`
- Split: `test`
- Samples: `542`
- Correct: `309/542`
- Accuracy: `57.01%`
- Elapsed time: `113.8s`
- Per-class accuracy:
  - `Case Based`: `0.00%` (`0/64`)
  - `Genetic Algorithms`: `83.33%` (`60/72`)
  - `Neural Networks`: `73.75%` (`118/160`)
  - `Probabilistic Methods`: `90.00%` (`81/90`)
  - `Reinforcement Learning`: `92.59%` (`50/54`)
  - `Rule Learning`: `0.00%` (`0/35`)
  - `Theory`: `0.00%` (`0/67`)
- Config:
  - `batch_size=1`
  - `max_seq_len=2048`
  - `max_neighbors_per_hop=10`
  - `max_hops=2`
  - `use_topology_mask=False`
  - `lora_path=None`
  - `prompt_layout=target_first`
  - `use_chat_template=False`
  - `prompt_format=category_infill`
  - `answer_label_style=digit0`
  - `include_neighbor_labels=True`
  - `neighbor_label_format=bracket`
- Stdout: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_llaga_frozen_eval_20260427/cora_llaga_frozen_notopo_logit_labelon.out`
- JSONL: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_llaga_frozen_eval_20260427/cora_llaga_frozen_notopo_logit_labelon.jsonl`

### Cora Frozen Base Topo Details

- Experiment: `cora_llaga_frozen_topo_logit_labelon`
- Model: `/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct`
- Dataset: `cora`
- Split: `test`
- Accuracy: `56.27%`
- Elapsed time: `246.9s`
- Per-class accuracy: `Case Based: 1.56; Genetic Algorithms: 83.33; Neural Networks: 73.12; Probabilistic Methods: 84.44; Reinforcement Learning: 94.44; Rule Learning: 0.00; Theory: 0.00`
- Stdout: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_llaga_frozen_eval_20260427/cora_llaga_frozen_topo_logit_labelon.out`
- JSONL: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_llaga_frozen_eval_20260427/cora_llaga_frozen_topo_logit_labelon.jsonl`

### PubMed Frozen Base Topo Details

- Experiment: `pubmed_llaga_frozen_topo_logit_labelon`
- Model: `/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct`
- Dataset: `pubmed`
- Split: `test`
- Accuracy: `86.49%`
- Elapsed time: `2832.9s`
- Per-class accuracy: `Diabetes Mellitus, Experimental: 72.04; Diabetes Mellitus Type 1: 87.42; Diabetes Mellitus Type 2: 93.25`
- Stdout: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/pubmed_llaga_frozen_eval_20260427/pubmed_llaga_frozen_topo_logit_labelon.out`
- JSONL: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/pubmed_llaga_frozen_eval_20260427/pubmed_llaga_frozen_topo_logit_labelon.jsonl`

### ogbn-arxiv Frozen Base Failure (2026-04-27)

- Date: `2026-04-27`
- Runs attempted: no-topo on GPU3, topo on GPU7
- Failure: `ModuleNotFoundError: No module named torch_sparse` while loading LLaGA processed data via `torch.load`.
- no-topo stdout: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/ogbn_arxiv_llaga_frozen_eval_20260427/ogbn_arxiv_llaga_frozen_notopo_logit_labelon.out`
- topo stdout: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/ogbn_arxiv_llaga_frozen_eval_20260427/ogbn_arxiv_llaga_frozen_topo_logit_labelon.out`

### ogbn-arxiv Frozen Base (2026-04-28)

`torch_sparse` installed; re-ran on 2026-04-28. `batch_size=4`, `bs=1` was too slow.

| config | accuracy | elapsed |
|---|---:|---:|
| notopo | 37.32% | 12094s (3.36 hr) |
| topo   | 37.58% | 13562s (3.77 hr) |

- topo adds only +0.26 pt; consistent with Cora/PubMed trend.
- Severe class imbalance: cs.CV dominates (96%), most classes < 50%.
- Per-class accuracy (notopo, top 8): cs.CV: 96.08; cs.DC: 79.86; cs.FL: 69.55; cs.CG: 66.45; cs.CC: 63.19; cs.CE: 59.70; cs.LO: 54.16; cs.CR: 43.45
- Per-class accuracy (topo, top 8): cs.CV: 96.23; cs.DC: 85.23; cs.FL: 70.45; cs.CG: 69.33; cs.CC: 63.77; cs.CE: 52.24; cs.LO: 48.70; cs.NE: 45.54
- notopo JSONL: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/ogbn_arxiv_llaga_frozen_eval_20260428/ogbn_arxiv_llaga_frozen_notopo_logit_labelon.jsonl`
- topo JSONL: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/ogbn_arxiv_llaga_frozen_eval_20260428/ogbn_arxiv_llaga_frozen_topo_logit_labelon.jsonl`

### ogbn-products Frozen Base (2026-04-28)

First-ever eval on ogbn-products. `batch_size=4`.

| config | accuracy | elapsed |
|---|---:|---:|
| notopo | 23.07% | 3874s (1.08 hr) |
| topo   | 23.44% | 3885s (1.08 hr) |

- Topology mask again negligible (+0.37 pt).
- Much faster than ogbn-arxiv (fewer test samples / shorter sequences).
- Strong categories: Grocery & Gourmet Food, Arts & Crafts, Cell Phones, CDs & Vinyl. Long-tail retail categories very poor.
- Per-class accuracy (notopo, top 8): Grocery: 82.17; Arts & Crafts: 81.48; Cell Phones: 78.94; CDs: 74.33; Health: 64.60; Toys: 61.03; Clothing: 51.17; Tools: 51.11
- Per-class accuracy (topo, top 8): Arts & Crafts: 84.29; Grocery: 82.56; Cell Phones: 75.78; CDs: 70.68; Health: 67.20; Toys: 62.47; Tools: 54.12; Clothing: 50.23
- notopo JSONL: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/ogbn_products_llaga_frozen_eval_20260428/ogbn_products_llaga_frozen_notopo_logit_labelon.jsonl`
- topo JSONL: `/home/lingjie7/auto-research/projects/dlm-graph/summaries/ogbn_products_llaga_frozen_eval_20260428/ogbn_products_llaga_frozen_topo_logit_labelon.jsonl`

## Running / Pending

| Date | Dataset | Run | GPU | Status | Stdout | Output |
|---|---|---|---:|---|---|---|
| 2026-04-27 | Cora | `cora_llaga_frozen_notopo_logit_labelon` | 7 | finished | `/home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_llaga_frozen_eval_20260427/cora_llaga_frozen_notopo_logit_labelon.out` | `/home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_llaga_frozen_eval_20260427/cora_llaga_frozen_notopo_logit_labelon.jsonl` |
| 2026-04-27 | Cora | no-topo SFT, r64, 20 epochs, LLaGA aligned | 0 | OOM-killed at step 1224/2040 (~ep 12) | `/home/lingjie7/auto-research/projects/dlm-graph/.models/tmdlm-llada-8b-cora-2hop-notopo-catinfill-nbmask-noeospad-r64-ep20-llaga_20260427_aligned_notopo_gpu0_tmux2.log` | `/home/lingjie7/auto-research/projects/dlm-graph/.models/tmdlm-llada-8b-cora-2hop-notopo-catinfill-nbmask-noeospad-r64-ep20-llaga_20260427_aligned_notopo_gpu0_tmux2` |
| 2026-04-27 | Cora | topo SFT, r64, 20 epochs, LLaGA aligned | 1 | OOM-killed at step 1101/2040 (~ep 10) | `/home/lingjie7/auto-research/projects/dlm-graph/.models/tmdlm-llada-8b-cora-2hop-topo-catinfill-nbmask-noeospad-r64-ep20-llaga_20260427_aligned_topo_gpu1_tmux1.log` | `/home/lingjie7/auto-research/projects/dlm-graph/.models/tmdlm-llada-8b-cora-2hop-topo-catinfill-nbmask-noeospad-r64-ep20-llaga_20260427_aligned_topo_gpu1_tmux1` |

## Cora SFT LoRA Checkpoints — All-ckpt logit + infill eval (2026-04-27 / 28)

Both runs use the same SFT recipe (2-hop, max_neighbors_per_hop=10, prompt_format=category_infill, max_answer_tokens=6, include_neighbor_labels=True/bracket, position_id_type=sequential, LoRA r=64 alpha=64 all-linear). Eval split=test (542 samples). Symlinked unified RUN_TAG `llaga_20260427_aligned`.

- topo run dir: `/home/lingjie7/auto-research/projects/dlm-graph/.models/tmdlm-llada-8b-cora-2hop-topo-catinfill-nbmask-noeospad-r64-ep20-llaga_20260427_aligned`
- notopo run dir: `/home/lingjie7/auto-research/projects/dlm-graph/.models/tmdlm-llada-8b-cora-2hop-notopo-catinfill-nbmask-noeospad-r64-ep20-llaga_20260427_aligned`
- topo only has ckpts up to 1020 (training OOM-killed before ckpt-1224); notopo has ckpts up to 1224.

### eval_logit (overall ACC%)

| ckpt (step) | topo | notopo |
|---:|---:|---:|
| 204  | 82.47 | 82.66 |
| 408  | 88.93 | 87.82 |
| 612  | 89.67 | 89.85 |
| 816  | 90.41 | 90.59 |
| 1020 | 90.77 | 91.33 |
| 1224 | —     | **91.33** |

- Per-ckpt elapsed: ~130 s (topo) / ~128 s (notopo).
- JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-2hop-{topo,notopo}-llaga_20260427_aligned.jsonl`
- Worker logs: `/tmp/dlm-graph-eval-logs/eval-cora-2hop-{topo,notopo}-llaga_20260427_aligned.worker.log`
- Launch script: [`examples/tmdlm/run_eval_cora_hops_topo_lora_all_ckpts.sh`](examples/tmdlm/run_eval_cora_hops_topo_lora_all_ckpts.sh)

### eval_infill (strict = lenient ACC%, steps=10)

| ckpt (step) | topo | notopo |
|---:|---:|---:|
| 204  | 87.64 | 89.11 |
| 408  | 89.67 | 90.04 |
| 612  | 89.67 | 89.48 |
| 816  | 91.14 | 91.14 |
| 1020 | 91.51 | 91.33 |
| 1224 | —     | **92.07** |

- Per-ckpt elapsed: ~641 s (topo) / ~466 s (notopo). topo is ~37% slower because the 4D additive topology mask falls back to HF eager attention (flash-attn / SDPA fast kernels only support 2D padding masks or structured causal/sliding patterns).
- JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-2hop-{topo,notopo}-llaga_20260427_aligned-infill.jsonl`
- Worker logs (initial 2-GPU run): `/tmp/dlm-graph-eval-logs/eval-cora-2hop-{topo,notopo}-llaga_20260427_aligned-infill.worker.log`
- Worker logs (4-GPU split for remaining ckpts): `/tmp/dlm-graph-eval-logs/eval-cora-2hop-{topo,notopo}-llaga_20260427_aligned-infill-4gpu-gpu{4,5,6,7}.worker.log`
- Launch scripts: [`examples/tmdlm/run_eval_cora_hops_topo_lora_all_ckpts_infill.sh`](examples/tmdlm/run_eval_cora_hops_topo_lora_all_ckpts_infill.sh), [`examples/tmdlm/run_eval_cora_infill_4gpu_oneshot.sh`](examples/tmdlm/run_eval_cora_infill_4gpu_oneshot.sh)

### Takeaways

- Best point overall: **notopo + infill, ckpt-1224 → 92.07%**.
- topology mask (topo) does **not** help on this SFT recipe — at every comparable ckpt, notopo is within ±0.5 pt of topo (sometimes slightly higher), and topo also lost the latest ckpt to OOM.
- infill ≥ logit at every ckpt (avg gap ~+1 pt at later ckpts, larger at early ckpts e.g. ckpt-204 +5 / +6.5 pt). Iterative denoising decode is more robust than logit-argmax.
- Both metrics improve monotonically with training (one tiny dip at notopo-infill ckpt-612 vs 408 = −0.6 pt, within run-to-run noise).
- topo training crashed earlier (step 1101) than notopo (step 1224). If we re-run topo to completion the gap to notopo would likely close further but probably not reverse.

---

## Legacy Runs (pre-llaga alignment, 2026-04-25) — Archived

These runs used an older dataset split (not LLaGA-aligned). Model weights deleted; training config preserved at each run dir root.

### Cora SFT LoRA (20260425_042600) — eval_logit, notopo, 2-hop

- Run dir (weights deleted): `.models/legacy/cora_pre_llaga_20260427/tmdlm-llada-8b-cora-2hop-notopo-catinfill-nbmask-noeospad-r64-ep20-20260425_042600/`
- Config: LLaDA-8B-Instruct, LoRA r=64 alpha=64 all-linear, dropout=0.05, ep=20, 2-hop, category_infill, include_neighbor_labels=True

| ckpt | logit acc |
|---:|---:|
| 204  | 78.97 |
| 408  | 84.13 |
| 612  | 87.45 |
| 816  | 86.53 |
| 1020 | 88.19 |
| 1224 | 90.04 |
| 1428 | **90.22** |
| 1632 | 89.85 |

Per-class at best (ckpt-1428): Case Based: 87.1; Genetic Algorithms: 97.78; Neural Networks: 95.42; Probabilistic Methods: 92.65; RL: 82.14; Rule Learning: 95.24; Theory: 73.24

Note: infill results for this run were not collected (empty).

### PubMed SFT LoRA (20260425_150500) — eval_infill, checkpoint-final, 2-hop

- Run dir (weights deleted): `.models/tmdlm-llada-8b-pubmed-2hop-{notopo,topo}-catinfill-nbmask-noeospad-r64-ep20-20260425_150500/`
- Config: LLaDA-8B-Instruct, LoRA r=64 alpha=64 all-linear, dropout=0.05, ep=20, 2-hop, category_infill, include_neighbor_labels=True
- Eval: checkpoint-final (step 80/80), various diffusion steps, test split (999 samples)
- JSONL: `summaries/pubmed_final_infill_steps_gpu26_wait_20260425_212612/jsonl/`

| steps | topo acc | notopo acc |
|---:|---:|---:|
| 1  | 88.99 | 90.19 |
| 2  | 89.69 | 90.09 |
| 4  | 90.59 | 87.39 |
| 8  | **91.79** | **91.99** |

Per-class at steps=8:
- topo: Experimental: 90.39; Type 1: 88.59; Type 2: 96.40
- notopo: Experimental: 87.99; Type 1: 90.09; Type 2: 97.90

Note: These results are lower than pubmed_20260428_aligned (~94–95%) due to older dataset split.

---

## PubMed SFT LoRA Checkpoints — All-ckpt logit eval (2026-04-28)

Both runs use the same SFT recipe (2-hop, max_neighbors_per_hop=10, prompt_format=category_infill, max_answer_tokens=6, include_neighbor_labels=True/bracket, position_id_type=sequential, LoRA r=64 alpha=64 all-linear, ep=10). Eval split=test (999 samples, 333/class, custom stratified split). Unified RUN_TAG `pubmed_20260428_aligned`.

- notopo run dir: `/home/lingjie7/auto-research/projects/dlm-graph/.models/tmdlm-llada-8b-pubmed-2hop-notopo-catinfill-nbmask-noeospad-r64-ep10-pubmed_20260428_aligned`
- topo run dir: `/home/lingjie7/auto-research/projects/dlm-graph/.models/tmdlm-llada-8b-pubmed-2hop-topo-catinfill-nbmask-noeospad-r64-ep10-pubmed_20260428_aligned`
- Both have ckpts: 370, 740, 1110, 1480, 1850, 2220 (ep 2, 4, 6, 8, 10; ckpt-2220 was saved mid-run before training was killed).

### eval_logit (overall ACC%)

| ckpt (step) | topo | notopo |
|---:|---:|---:|
| 370  | 92.11 | 91.23 |
| 740  | 94.14 | 94.35 |
| 1110 | 92.37 | 93.61 |
| 1480 | **94.47** | **95.18** |
| 1850 | ❌ not yet run | ❌ not yet run |
| 2220 | ❌ not yet run | ❌ not yet run |

- Per-ckpt elapsed: ~2044s (34 min) for notopo; ~955s (16 min) for topo.
- JSONL: `/tmp/dlm-graph-eval-jsonl/eval-pubmed-2hop-{topo,notopo}-pubmed_20260428_aligned.jsonl`

### eval_infill (strict ACC%, steps=10) — partial, in progress

| ckpt (step) | topo | notopo |
|---:|---:|---:|
| 370  | 92.98 | 93.13 |
| 740  | 🔄 running (GPU6) | 94.93 |
| 1110 | ⏳ pending | 🔄 running (GPU7) |
| 1480 | ⏳ pending | ⏳ pending |
| 1850 | ❌ not yet run | ❌ not yet run |
| 2220 | ❌ not yet run | ❌ not yet run |

- Per-ckpt elapsed: ~12601s (3.5 hr) for topo; ~7536–9279s (2.1–2.6 hr) for notopo.
- JSONL: `/tmp/dlm-graph-eval-jsonl/eval-pubmed-2hop-{topo,notopo}-pubmed_20260428_aligned-infill.jsonl`
- Worker logs: `/tmp/dlm-graph-eval-logs/eval-pubmed-existing-gpu{6,7}.worker.log`
