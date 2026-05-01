# DLM-Graph: Node Classification Results

All results below are **post the prompt/option-block label-leakage fix** in
`dllm/data/graph.py` (2026-04-29). All training runs use
`include_neighbor_labels=False` (**nonb** — neighbor text only, no oracle
class labels — fair vs. LLaGA). Earlier runs that either used `nbmask`
(oracle neighbor labels) or pre-date the leakage fix are not included here.

## Model & Setup

- **Model**: LLaDA-8B-Instruct (GSAI-ML/LLaDA-8B-Instruct), 8B params, masked discrete diffusion LM.
- **Training**: LoRA r=64, alpha=64, all-linear modules, lr=5e-5, effective batch 32 (per-device 4 × grad-accum 8), max_hops=2, max_neighbors_per_hop=10, position_id_type=sequential, gradient_checkpointing on, 10 epochs unless noted.
- **Neighbor labels**: `include_neighbor_labels=False` (**nonb**) for all runs in this file.
- **Test sets**: Cora 542 nodes / 7 classes; PubMed 999 nodes / 3 classes (custom stratified split, 333/class).

## External Baselines

### Cora — supervised
Source: "When Do LLMs Help With Node Classification?" (arXiv:2502.00829), Table 2.

| Method                 | Type                  | Accuracy       |
| ---------------------- | --------------------- | -------------- |
| GCN + LLM Emb          | GNN + LLM embeddings  | 88.15 ± 1.79   |
| TAPE                   | LLM-as-Reasoner       | 88.05 ± 1.76   |
| LLaGA                  | LLM + Graph Projector | 87.55 ± 1.15   |
| GraphSAGE (ShallowEmb) | GNN                   | 87.44 ± 1.74   |
| GCN (ShallowEmb)       | GNN                   | 87.41 ± 2.08   |
| ENGINE                 | GNN + LLM             | 87.00 ± 1.60   |
| GLEM                   | GNN + LLM             | 86.81 ± 1.19   |
| GAT (ShallowEmb)       | GNN                   | 86.68 ± 1.12   |
| RoBERTa-355M           | LM only               | 83.17 ± 0.84   |
| GraphGPT               | LLM + Graph           | 82.29 ± 0.26   |

### PubMed — supervised
Source: LLaGA (arXiv:2402.08170), Table 1 (Single Focus). Note our PubMed split is custom stratified (999 test) from TAPE files, not Planetoid — comparison approximate.

| Method      | Type                  | Accuracy |
| ----------- | --------------------- | -------- |
| SAGN        | GNN                   | 95.17    |
| LLaGA-ND-7B | LLM + Graph Projector | 95.03    |
| LLaGA-HO-7B | LLM + Graph Projector | 95.03    |
| NodeFormer  | Graph Transformer     | 94.90    |
| GraphSAGE   | GNN                   | 94.87    |
| GCN         | GNN                   | 92.96    |
| GAT         | GNN                   | 92.33    |
| SGC         | GNN                   | 87.35    |

---

## §1. Cora — SFT mc_digit nonb (run tag `cora_20260429_mcdigit_nonb_fixed`)

Single-dataset Cora, mc_digit (digit answer over `{0..6}`), 510 steps total.

### Logit Eval (direct token scoring over class digits)

| Checkpoint | notopo | topo |
|-----------:|-------:|-----:|
|  26 | 78.23 | 75.46 |
|  52 | 81.55 | 81.73 |
|  78 | 85.24 | 84.50 |
| 104 | 85.61 | 83.58 |
| 130 | 86.35 | 86.72 |
| 156 | 86.16 | 87.27 |
| 182 | 89.11 | 87.82 |
| 208 | 89.11 | 87.64 |
| 234 | 88.75 | 87.45 |
| 260 | 90.41 | 88.93 |
| 286 | 89.67 | 88.93 |
| 312 | 90.59 | **90.04** ⭐ |
| **338** | **90.77** ⭐ | 89.11 |
| 364 | 90.41 | 89.11 |
| 390 | 90.22 | 89.85 |
| 416 | 90.41 | 89.48 |
| 442 | 90.59 | 89.67 |
| 468 | 90.41 | 89.67 |
| 494 | 90.22 | 89.67 |
| 510 | 90.41 | 89.48 |

JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-2hop-{notopo,topo}-nonb-cora_20260429_mcdigit_nonb_fixed.jsonl`

### Infill Eval (masked diffusion gen, 10 steps, T=0)

| Checkpoint | notopo strict | topo strict |
|-----------:|--------------:|------------:|
|  26 | 78.41 | 75.65 |
|  52 | 81.55 | 81.73 |
|  78 | 85.24 | 84.69 |
| 104 | 85.42 | 83.76 |
| 130 | 86.16 | 86.72 |
| 156 | 86.16 | 86.90 |
| 182 | 89.11 | 87.82 |
| 208 | 88.93 | 87.64 |
| 234 | 88.93 | 87.64 |
| **260** | **90.96** ⭐ | 88.56 |
| 286 | 89.48 | 88.93 |
| 312 | 90.41 | **89.85** ⭐ |
| 338 | 90.77 | 88.93 |
| 364 | 90.41 | 89.30 |
| 390 | 90.04 | 89.67 |
| 416 | 90.41 | 89.48 |
| 442 | 90.59 | 89.85 |
| 468 | 90.59 | 89.67 |
| 494 | 90.59 | 89.85 |
| 510 | 90.41 | 89.48 |

`accuracy_lenient` matches `accuracy_strict` for all rows.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-2hop-{notopo,topo}-nonb-cora_20260429_mcdigit_nonb_fixed-infill.jsonl`

### Cora extras — best 2 ckpts × {nb, hop} variations

Run on the 2 best ckpts per setting (notopo 260+338, topo 312+442).

| Setting | ckpt | logit | infill strict |
|---------|-----:|------:|--------------:|
| notopo nb=10 hop=3 | 260 | 90.41 | 90.96 |
| notopo nb=10 hop=3 | 338 | 90.77 | 90.77 |
| notopo nb=15 hop=2 | 260 | 89.48 | 89.67 |
| notopo nb=15 hop=2 | 338 | 90.22 | 90.22 |
| notopo nb=20 hop=2 | 260 | 89.48 | 89.48 |
| notopo nb=20 hop=2 | 338 | 90.59 | 90.59 |
| notopo nb=25 hop=2 | 260 | 89.85 | 89.67 |
| notopo nb=25 hop=2 | 338 | 90.04 | 90.22 |
| topo   nb=10 hop=3 | 312 | 90.04 | 89.85 |
| topo   nb=10 hop=3 | 442 | 89.67 | 89.85 |
| topo   nb=15 hop=2 | 312 | 89.85 | 89.48 |
| topo   nb=15 hop=2 | 442 | 89.67 | 89.85 |
| topo   nb=20 hop=2 | 312 | 89.85 | 89.48 |
| topo   nb=20 hop=2 | 442 | 90.04 | 90.04 |
| topo   nb=25 hop=2 | 312 | 89.85 | 89.48 |
| topo   nb=25 hop=2 | 442 | 89.67 | 89.67 |

Take-away: more neighbors / 3 hops give no meaningful gain over the default 2-hop nb=10 baseline. Best overall remains nb=10, hop=2.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-extras-{notopo,topo}-nb{10,15,20,25}-{2,3}h{,-infill}.jsonl`

---

## Cross-dataset / resampling experiments (2026-04-30)

See `experiment_log.md` for design rationale.

### §6. cora+pubmed merged catinfill nonb (run tag `cora-pubmed_20260430`, **killed**)

`prompt_format=category_infill`, `max_answer_tokens=10`, no resampling. Demonstrates class collapse on cora when pubmed dominates the gradient.

#### Logit Eval

| ckpt | cora-notopo | cora-topo | pubmed-notopo | pubmed-topo |
|---:|---:|---:|---:|---:|
|  211 | 60.70 | 66.79 | 76.29 | 76.55 |
|  422 | 63.84 | 69.37 | 77.36 | 77.18 |
|  633 | 46.31 | 70.30 | **77.48** | **77.56** |
|  844 | 48.52 | 67.71 | 76.93 | 77.03 |
| 1055 | 39.85 | —     | 77.41 | —     |
| 1266 | —     | —     | 76.65 | —     |

cora-notopo collapses from 63.84 → 39.85 between ckpt-422 and ckpt-1055; the topo mask delays collapse. Pubmed plateaus ~77% (well below the single-pubmed level on this data) due to vocabulary-level domination by the unconstrained `[Diab]` prefix tokens. Run was killed and replaced by §7.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-merged-on-{cora,pubmed}-{notopo,topo}-catinfill-nonb-cora-pubmed_20260430-logit.jsonl`

### §7. cora+pubmed merged mc_digit + balanced nonb (run tag `cora-pubmed_20260430_mcdigit_d0_bal_nonb`, **in progress**)

`prompt_format=mc_digit`, `answer_label_style=digit0`, `max_answer_tokens=1`, `--resample_strategy balance_datasets` → each dataset downsampled to min count = 1624. ~1020 steps total, both topo and notopo. Eval still progressing at time of writing.

#### Logit Eval

| ckpt | cora-notopo | cora-topo | pubmed-notopo | pubmed-topo |
|---:|---:|---:|---:|---:|
|  51 | 78.04 | 76.38 | 89.76 | 91.00 |
| 102 | 82.66 | 81.00 | 93.33 | 93.69 |
| 153 | 85.42 | 84.50 | 92.77 | 93.38 |
| 204 | 87.64 | 85.98 | 94.32 | 94.75 |
| 255 | 86.90 | 85.79 | 94.55 | 94.75 |
| 306 | **89.11** ⭐ | **88.75** ⭐ | 94.60 | 93.94 |
| 357 | 88.56 | 87.64 | 94.65 | 94.68 |
| 408 | (pending) | (pending) | (pending) | 94.37 |
| 459 | (pending) | (pending) | (pending) | 90.09 ← dip |
| 510 | (pending) | (pending) | (pending) | **94.93** ⭐ ← recovered |
| 561+ | (pending — followup will catch up) ||||

Switching to `mc_digit + balance_datasets` removes the catinfill class collapse. Cora reaches ~89% (single-cora §1 best 90.77, gap ~1.7pt). Pubmed reaches ~94.9% on topo. The pubmed-topo dip at ckpt-459 (94.7 → 90.1) is followed by a clean recovery at 510 — investigating later ckpts.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-merged-bal-on-{cora,pubmed}-{notopo,topo}-mcdigit-cora-pubmed_20260430_bal-logit.jsonl`

### §8. cora-only mc_digit boost nonb (run tag `cora_20260430_mcdigit_boost_nonb`, **in progress**)

Single-dataset cora, `topo`, mc_digit, `--resample_strategy boost --boost_spec "cora:Theory:2,cora:Rule Learning:3"`. Train 1624 → 2045 (+25.9%); ~640 steps. Compare against single-cora topo §1 baseline (90.04 @ ckpt-312) to isolate the boost effect.

#### Logit Eval

| ckpt | cora-topo |
|---:|---:|
|  32 | 75.46 |
|  64 | 83.03 |
|  96 | 84.87 |
| 128 | 85.98 |
| 160 | **87.08** |
| 192 | 86.35 |
| 224 | 86.53 |

Per-class vs §1 baseline at comparable training (data still partial): boost ckpt-64 already shows Rule Learning ≈ 94, Theory ≈ 65, vs §1 single-cora at similar step ~74 / ~50 — i.e. boost lifts the two targeted classes earlier in training. Final comparison pending later ckpts (256–640).
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-boost-topo-mcdigit-cora_20260430_boost-logit.jsonl`

---

## Summary — best per dataset / setting

| Section | Run | Setting | Best ckpt | Acc | Eval |
|---|---|---|---:|---:|---|
| §1 | cora single | mc_digit nonb notopo  | 260 | **90.96** | infill strict |
| §1 | cora single | mc_digit nonb notopo  | 338 | **90.77** | logit |
| §1 | cora single | mc_digit nonb topo    | 312 | 89.85 / 90.04 | infill / logit |
| §7 | cora+pubmed merged-bal (cora) | mc_digit nonb notopo | 306 | 89.11 | logit (in-progress) |
| §7 | cora+pubmed merged-bal (cora) | mc_digit nonb topo   | 306 | 88.75 | logit (in-progress) |
| §7 | cora+pubmed merged-bal (pubmed) | mc_digit nonb notopo | 357 | 94.65 | logit (in-progress) |
| §7 | cora+pubmed merged-bal (pubmed) | mc_digit nonb topo   | 510 | **94.93** | logit (in-progress) |
| §8 | cora boost | mc_digit nonb topo   | 160 | 87.08 | logit (in-progress) |
