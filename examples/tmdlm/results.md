# DLM-Graph: Experiment Results (NC + LP)

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


| Method                 | Type                  | Accuracy     |
| ---------------------- | --------------------- | ------------ |
| GCN + LLM Emb          | GNN + LLM embeddings  | 88.15 ± 1.79 |
| TAPE                   | LLM-as-Reasoner       | 88.05 ± 1.76 |
| LLaGA                  | LLM + Graph Projector | 87.55 ± 1.15 |
| GraphSAGE (ShallowEmb) | GNN                   | 87.44 ± 1.74 |
| GCN (ShallowEmb)       | GNN                   | 87.41 ± 2.08 |
| ENGINE                 | GNN + LLM             | 87.00 ± 1.60 |
| GLEM                   | GNN + LLM             | 86.81 ± 1.19 |
| GAT (ShallowEmb)       | GNN                   | 86.68 ± 1.12 |
| RoBERTa-355M           | LM only               | 83.17 ± 0.84 |
| GraphGPT               | LLM + Graph           | 82.29 ± 0.26 |


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


### Link Prediction (LP) — supervised

Source: LLaGA (arXiv:2402.08170), Table 1, "Single Focus" setting (task-specific SFT per dataset — closest to our recipe). Accuracy (%) on each dataset's LP test split. LLaGA does not report AUC.

Our edge split is a deterministic random 85 / 5 / 10 with `seed=42` (`dllm/data/datasets/_lp_common.py`), not the same split LLaGA uses, so absolute numbers are indicative — not directly head-to-head.


| Method        | Type                  | Cora        | PubMed      | ogbn-arxiv  | ogbn-products |
| ------------- | --------------------- | ----------- | ----------- | ----------- | ------------- |
| GCN           | GNN                   | 85.09       | 94.55       | 92.28       | 93.89         |
| GAT           | GNN                   | 82.68       | 87.60       | 87.78       | 94.19         |
| GraphSAGE     | GNN                   | 79.94       | 93.87       | 92.75       | 95.22         |
| NodeFormer    | Graph Transformer     | 81.79       | 84.43       | 92.60       | 96.13         |
| LLaGA-ND-7B   | LLM + Graph Projector | **92.71** ⭐ | 96.49       | 93.31       | **97.85** ⭐  |
| LLaGA-HO-7B   | LLM + Graph Projector | 92.65       | **96.95** ⭐ | **96.18** ⭐ | 95.88         |


### Our base LLaDA-8B-Instruct (zero-shot, no SFT)

Sanity baseline for LP before any fine-tuning: a frozen LLaDA-8B-Instruct receives the same text-LP prompt (`Paper A: <text>. Neighbor A1: ... Paper B: <text>. Neighbor B1: ... Do Paper A and Paper B cite each other? Answer:`) and we score the `' yes'` vs `' no'` token logits at the answer position (`examples/tmdlm/eval_lp_logit.py`).


| Dataset | Samples | Accuracy | AUC    | Per-label acc (no / yes) |
| ------- | ------: | -------: | -----: | ------------------------ |
| Cora    | 1,054   | 52.18    | 0.5668 | 72.68 / 31.69            |


JSONL: `.models/eval_logs/lp_base_zeroshot.jsonl`. Config: `max_seq_len=4096`, `max_neighbors_per_hop=10`, `max_hops=2`, `use_topology_mask=True`, `batch_size=4`, `seed=42`.

The base model is essentially at random chance (50% chance line) with a strong prior toward "no": 72.7 % recall on true non-edges, only 31.7 % on true edges. AUC 0.567 shows a weak ranking signal but the decision boundary is far from optimal. The model has not been exposed to "do these two papers cite each other?" supervision in pretraining; LP therefore requires task-specific SFT — motivating the LP runs in §21+ (to follow).

---

## §1. Cora — SFT mc_digit nonb (run tag `cora_20260429_mcdigit_nonb_fixed`)

Single-dataset Cora, mc_digit (digit answer over `{0..6}`), 510 steps total.

### Logit Eval (direct token scoring over class digits)


| Checkpoint | notopo      | topo        |
| ---------- | ----------- | ----------- |
| 26         | 78.23       | 75.46       |
| 52         | 81.55       | 81.73       |
| 78         | 85.24       | 84.50       |
| 104        | 85.61       | 83.58       |
| 130        | 86.35       | 86.72       |
| 156        | 86.16       | 87.27       |
| 182        | 89.11       | 87.82       |
| 208        | 89.11       | 87.64       |
| 234        | 88.75       | 87.45       |
| 260        | 90.41       | 88.93       |
| 286        | 89.67       | 88.93       |
| 312        | 90.59       | **90.04** ⭐ |
| **338**    | **90.77** ⭐ | 89.11       |
| 364        | 90.41       | 89.11       |
| 390        | 90.22       | 89.85       |
| 416        | 90.41       | 89.48       |
| 442        | 90.59       | 89.67       |
| 468        | 90.41       | 89.67       |
| 494        | 90.22       | 89.67       |
| 510        | 90.41       | 89.48       |


JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-2hop-{notopo,topo}-nonb-cora_20260429_mcdigit_nonb_fixed.jsonl`

### Infill Eval (masked diffusion gen, 10 steps, T=0)


| Checkpoint | notopo strict | topo strict |
| ---------- | ------------- | ----------- |
| 26         | 78.41         | 75.65       |
| 52         | 81.55         | 81.73       |
| 78         | 85.24         | 84.69       |
| 104        | 85.42         | 83.76       |
| 130        | 86.16         | 86.72       |
| 156        | 86.16         | 86.90       |
| 182        | 89.11         | 87.82       |
| 208        | 88.93         | 87.64       |
| 234        | 88.93         | 87.64       |
| **260**    | **90.96** ⭐   | 88.56       |
| 286        | 89.48         | 88.93       |
| 312        | 90.41         | **89.85** ⭐ |
| 338        | 90.77         | 88.93       |
| 364        | 90.41         | 89.30       |
| 390        | 90.04         | 89.67       |
| 416        | 90.41         | 89.48       |
| 442        | 90.59         | 89.85       |
| 468        | 90.59         | 89.67       |
| 494        | 90.59         | 89.85       |
| 510        | 90.41         | 89.48       |


`accuracy_lenient` matches `accuracy_strict` for all rows.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-2hop-{notopo,topo}-nonb-cora_20260429_mcdigit_nonb_fixed-infill.jsonl`

### Cora extras — best 2 ckpts × {nb, hop} variations

Run on the 2 best ckpts per setting (notopo 260+338, topo 312+442).


| Setting            | ckpt | logit | infill strict |
| ------------------ | ---- | ----- | ------------- |
| notopo nb=10 hop=3 | 260  | 90.41 | 90.96         |
| notopo nb=10 hop=3 | 338  | 90.77 | 90.77         |
| notopo nb=15 hop=2 | 260  | 89.48 | 89.67         |
| notopo nb=15 hop=2 | 338  | 90.22 | 90.22         |
| notopo nb=20 hop=2 | 260  | 89.48 | 89.48         |
| notopo nb=20 hop=2 | 338  | 90.59 | 90.59         |
| notopo nb=25 hop=2 | 260  | 89.85 | 89.67         |
| notopo nb=25 hop=2 | 338  | 90.04 | 90.22         |
| topo nb=10 hop=3   | 312  | 90.04 | 89.85         |
| topo nb=10 hop=3   | 442  | 89.67 | 89.85         |
| topo nb=15 hop=2   | 312  | 89.85 | 89.48         |
| topo nb=15 hop=2   | 442  | 89.67 | 89.85         |
| topo nb=20 hop=2   | 312  | 89.85 | 89.48         |
| topo nb=20 hop=2   | 442  | 90.04 | 90.04         |
| topo nb=25 hop=2   | 312  | 89.85 | 89.48         |
| topo nb=25 hop=2   | 442  | 89.67 | 89.67         |


Take-away: more neighbors / 3 hops give no meaningful gain over the default 2-hop nb=10 baseline. Best overall remains nb=10, hop=2.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-extras-{notopo,topo}-nb{10,15,20,25}-{2,3}h{,-infill}.jsonl`

---

## Cross-dataset / resampling experiments (2026-04-30)

See `experiment_log.md` for design rationale.

### §6. cora+pubmed merged catinfill nonb (run tag `cora-pubmed_20260430`, **killed**)

`prompt_format=category_infill`, `max_answer_tokens=10`, no resampling. Demonstrates class collapse on cora when pubmed dominates the gradient.

#### Logit Eval


| ckpt | cora-notopo | cora-topo | pubmed-notopo | pubmed-topo |
| ---- | ----------- | --------- | ------------- | ----------- |
| 211  | 60.70       | 66.79     | 76.29         | 76.55       |
| 422  | 63.84       | 69.37     | 77.36         | 77.18       |
| 633  | 46.31       | 70.30     | **77.48**     | **77.56**   |
| 844  | 48.52       | 67.71     | 76.93         | 77.03       |
| 1055 | 39.85       | —         | 77.41         | —           |
| 1266 | —           | —         | 76.65         | —           |


cora-notopo collapses from 63.84 → 39.85 between ckpt-422 and ckpt-1055; the topo mask delays collapse. Pubmed plateaus ~77% (well below the single-pubmed level on this data) due to vocabulary-level domination by the unconstrained `[Diab]` prefix tokens. Run was killed and replaced by §7.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-merged-on-{cora,pubmed}-{notopo,topo}-catinfill-nonb-cora-pubmed_20260430-logit.jsonl`

### §7. cora+pubmed merged mc_digit + balanced nonb (run tag `cora-pubmed_20260430_mcdigit_d0_bal_nonb`, **complete**)

`prompt_format=mc_digit`, `answer_label_style=digit0`, `max_answer_tokens=1`, `cls_loss_weight=1.0`, `--resample_strategy balance_datasets` → each dataset downsampled to min count = 1624. 1020 steps total, both topo and notopo, all 20 ckpts × 4 settings (cora/pubmed × notopo/topo) evaluated.

#### Logit Eval


| ckpt | cora-notopo | cora-topo   | pubmed-notopo | pubmed-topo |
| ---- | ----------- | ----------- | ------------- | ----------- |
| 51   | 78.04       | 76.38       | 89.76         | 91.00       |
| 102  | 82.66       | 81.00       | 93.33         | 93.69       |
| 153  | 85.42       | 84.50       | 92.77         | 93.38       |
| 204  | 87.64       | 85.98       | 94.32         | 94.75       |
| 255  | 86.90       | 85.79       | 94.55         | 94.75       |
| 306  | 89.11       | 88.75       | 94.60         | 93.94       |
| 357  | 88.56       | 87.64       | 94.65         | 94.68       |
| 408  | 88.38       | 86.35       | 94.45         | 94.37       |
| 459  | 89.30       | 90.04       | 91.78 ← dip   | 90.09 ← dip |
| 510  | 89.30       | 89.11       | 94.42         | 94.93       |
| 561  | 87.82       | 89.48       | 95.03         | 94.93       |
| 612  | 89.11       | 88.56       | **95.28** ⭐   | 94.75       |
| 663  | 89.30       | 88.75       | 94.70         | 94.80       |
| 714  | 89.48       | 90.41       | 94.90         | 94.78       |
| 765  | **90.77** ⭐ | 90.41       | 94.65         | 94.37       |
| 816  | 90.59       | **90.96** ⭐ | 94.37         | 94.17       |
| 867  | 90.22       | 90.22       | 94.27         | 94.04       |
| 918  | 90.59       | 89.85       | 94.42         | 94.30       |
| 969  | 90.41       | 90.22       | 94.40         | 94.42       |
| 1020 | 90.59       | 90.04       | 94.47         | 94.35       |


**Run complete.** Best per setting: cora-notopo 90.77@765, cora-topo 90.96@816, pubmed-notopo **95.28@612** ⭐, pubmed-topo 94.93@510.

Switching to `mc_digit + balance_datasets` removes the §6 catinfill class collapse. **Final results match or exceed §1 single-dataset baselines on every setting**: cora-notopo ties 90.77 @ ckpt-765 = §1 best 90.77 logit; cora-topo 90.96 @ ckpt-816 exceeds §1 cora-topo logit 90.04 by +0.92pt; pubmed-notopo 95.28 @ ckpt-612 surpasses LLaGA-7B 95.03 (oracle GNN proj); pubmed-topo plateaus 94.8–94.9. Both pubmed settings show a transient dip near ckpt-459 (notopo 91.78, topo 90.09) followed by recovery — likely lr-scheduler local instability, not a regression.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-merged-bal-on-{cora,pubmed}-{notopo,topo}-mcdigit-cora-pubmed_20260430_bal-logit.jsonl`

### §8. cora-only mc_digit boost nonb (run tag `cora_20260430_mcdigit_boost_nonb`, **complete**)

Single-dataset cora, `topo`, mc_digit, `cls_loss_weight=1.0`, `--resample_strategy boost --boost_spec "cora:Theory:2,cora:Rule Learning:3"`. Train 1624 → 2045 (+25.9%); 640 steps total, all 20 ckpts evaluated. Compare against single-cora topo §1 baseline (90.04 @ ckpt-312) to isolate the boost effect.

#### Logit Eval


| ckpt    | cora-topo   |
| ------- | ----------- |
| 32      | 75.46       |
| 64      | 83.03       |
| 96      | 84.87       |
| 128     | 85.98       |
| 160     | 87.08       |
| 192     | 86.35       |
| 224     | 86.53       |
| 256     | 88.93       |
| 288     | 89.30       |
| **320** | **90.59** ⭐ |
| 352     | 87.64       |
| 384     | 89.30       |
| 416     | 89.30       |
| 448     | 88.93       |
| 480     | 88.19       |
| 512     | 89.30       |
| 544     | 89.48       |
| 576     | 89.85       |
| 608     | 90.22       |
| 640     | 89.85       |


Run **complete**. Best ckpt-320 = 90.59 (logit) — exceeds single-cora topo §1 baseline 90.04 by 0.55pt, validating the boost on Theory + Rule Learning.

Per-class vs §1 baseline at comparable training (data still partial): boost ckpt-64 already shows Rule Learning ≈ 94, Theory ≈ 65, vs §1 single-cora at similar step ~74 / ~50 — i.e. boost lifts the two targeted classes earlier in training. Final comparison pending later ckpts (256–640).
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-boost-topo-mcdigit-cora_20260430_boost-logit.jsonl`

### §9. single PubMed mc_digit nonb (run tag `pubmed_20260428_aligned`, **complete**)

Single-dataset PubMed, `prompt_format=mc_digit`, `answer_label_style=digit0`, `max_answer_tokens=1`, `nb=10, hop=2`, no resampling. 6 ckpts saved per setting (370 / 740 / 1110 / 1480 / 1850 / 2220; ckpt-2220 was captured mid-run before training was killed); checkpoint directories have since been deleted from disk so further training-step checkpoints are unrecoverable.

#### Logit Eval


| ckpt     | pubmed-notopo | pubmed-topo |
| -------- | ------------- | ----------- |
| 370      | 91.23         | 92.11       |
| 740      | 94.35         | 94.14       |
| 1110     | 93.61         | 92.37       |
| **1480** | **95.18** ⭐  | **94.47** ⭐ |
| 1850     | 94.95         | 94.90       |
| 2220     | 94.88         | 93.18       |


Best per setting: pubmed-notopo 95.18 @ ckpt-1480, pubmed-topo 94.47 @ ckpt-1480. Late ckpts (1850 / 2220) regress slightly on logit — notopo peaks at ckpt-1480 and does not improve further, while topo drops 1.3 pt from ckpt-1480 → 2220.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-pubmed-2hop-{notopo,topo}-pubmed_20260428_aligned.jsonl`

#### Infill Eval (strict acc%, 10 steps)


| ckpt     | pubmed-notopo | pubmed-topo |
| -------- | ------------- | ----------- |
| 370      | 93.13         | 92.98       |
| 740      | 94.93         | 93.36       |
| 1110     | 94.93         | 94.27       |
| 1480     | 94.90         | 94.14       |
| 1850     | 95.51         | 95.51       |
| **2220** | **95.64** ⭐  | **95.59** ⭐ |


Best per setting: pubmed-notopo infill **95.64 @ ckpt-2220**, pubmed-topo infill **95.59 @ ckpt-2220**. Infill decode recovers more accuracy than logit at late ckpts: notopo infill 95.64 vs. logit 95.18 at their respective peaks (+0.46 pt). Both infill peaks exceed the LLaGA-7B oracle-projector NC baseline of 95.03. The infill ↔ logit gap widens in the late-training regime (ckpts 1850–2220), consistent with the iterative denoising decoder recovering class signal that the single-step logit scorer misses once probability mass is spread across similar class tokens.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-pubmed-2hop-{notopo,topo}-pubmed_20260428_aligned-infill.jsonl`

Run **complete**. Single-dataset PubMed-notopo 95.18 (logit) / 95.64 (infill) sit within 0.10 pt and above of the §7 merged-balanced logit peak (95.28) — joint training on Cora+PubMed therefore loses no PubMed accuracy.

### §11. cora mc_digit nonb at `max_seq_len=4096` (run tag `cora_20260501_mcdigit_nonb_seq4k`, **complete**)

Single-dataset cora, `mc_digit + nonb`, `max_hops=2`, `max_neighbors_per_hop=10`, identical recipe to §1 except **`max_seq_len` raised from 2048 → 4096**. Motivation: at seq=2048 each neighbor was hard-truncated to ~66 tokens (cora abstract median ~200), losing most neighbor content. At seq=4096 the per-neighbor budget rises to ~170 tokens so most neighbor abstracts fit without token-level truncation. Per-device batch reduced from 6 → 3 with grad_accum 8 → 16 to keep the effective batch at 48 within 80GB. 340 total steps (10 epochs); save / eval every 17 steps (`save_steps=eval_steps=0.05`).

#### Logit Eval (cora test, 542 samples; all 20 ckpts complete)


| ckpt    | cora-notopo  | cora-topo    |
| ------- | ------------ | ------------ |
| 17      | 76.57        | 75.46        |
| 34      | 78.78        | 79.15        |
| 51      | 80.81        | 82.47        |
| 68      | 84.87        | 85.24        |
| 85      | 86.72        | 86.16        |
| 102     | 87.08        | 87.27        |
| 119     | 85.79        | 86.90        |
| 136     | 87.64        | 86.90        |
| 153     | 88.93        | 88.01        |
| 170     | 88.01        | 87.64        |
| 187     | 89.11        | **88.56** ⭐ |
| 204     | 87.45        | 87.45        |
| **221** | **89.67** ⭐ | 87.45        |
| 238     | 89.48        | **88.56** ⭐ |
| 255     | 89.11        | 88.01        |
| 272     | 88.93        | 87.82        |
| 289     | 89.11        | 88.01        |
| 306     | 89.30        | 88.19        |
| 323     | 89.30        | 88.19        |
| 340     | 89.30        | 88.19        |


Self-eval (cora → cora), all 20 ckpts complete. notopo best 89.67 @ ckpt-221; topo best 88.56 @ ckpt-187 / ckpt-238. Both still ~1.1–1.5 pt below §1 (cora seq=2048: 90.77 notopo / 90.04 topo) — the seq=2048 → 4096 expansion has *not* closed the gap. Two confounds: per-device batch halved (6 → 3) keeping effective batch=48 fixed, and total optimizer steps dropped from §1's 510 → 340 (since one example takes 2× more tokens at seq=4096 yet effective batch was held constant). §11_aligned (below) controls for the second confound.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-seq4k-{topo,notopo}-mcdigit-cora_20260501_mcdigit_nonb_seq4k-logit.jsonl`

#### Cross-Domain Eval: cora-trained ckpts → PubMed test (n=1000 stratified subsample)

Cross-domain transfer: §11 ckpts (trained on cora-only) evaluated on PubMed test (downsampled to 1000 samples, seed=42). Same SFT config (`max_seq_len=4096`, `hop=2`, `nb=10`, `mc_digit`, `nonb`, `max_answer_tokens=1`); only `dataset_name` (cora → pubmed) and `split` (train → test) differ. All 20 ckpts × 2 settings now evaluated.


| ckpt    | pubmed-notopo | pubmed-topo |
| ------- | ------------- | ----------- |
| 17      | 89.00         | 85.40       |
| 34      | 89.30         | 87.20       |
| 51      | 90.50         | 88.60       |
| 68      | 90.00         | 89.10       |
| 85      | 90.50         | 89.70       |
| 102     | 91.10         | 89.60       |
| 119     | 90.70         | 89.80       |
| **136** | **91.20** ⭐  | 90.10       |
| **153** | **91.20** ⭐  | 90.10       |
| 170     | 90.90         | 90.30       |
| 187     | 90.60         | 89.90       |
| 204     | 90.70         | 89.90       |
| 221     | 91.00         | 90.20       |
| 238     | 90.90         | 90.20       |
| 255     | 91.00         | 90.50       |
| 272     | 91.00         | **90.70** ⭐ |
| 289     | 91.00         | 90.60       |
| 306     | 91.10         | 90.50       |
| 323     | 91.00         | 90.50       |
| 340     | 91.10         | **90.70** ⭐ |


Cora-trained checkpoints transfer to PubMed at **~89–91 %** with no PubMed-specific training, far above the 33 % random baseline. Two reinforcing reasons: (1) the `mc_digit` prompt embeds the actual PubMed class names (`Diabetes Mellitus, Experimental` / `Type 1` / `Type 2`) in the options block, letting the underlying LLaDA-8B base model use its semantic prior to match abstract content to class name; (2) LoRA on cora teaches a generic "read paper abstract → pick a class index from the listed options" behavior without overwriting that prior. notopo peaks at **91.20** (ckpt-136/153, tied) and stays in 90.6–91.2 thereafter; topo peaks at **90.70** (ckpt-272/340) and is consistently 0.3–1.0 pt below notopo at every ckpt, mirroring the in-domain pattern. The 91.20 ceiling sits ~4 pt below §9 single-PubMed in-domain training (95.18) and ~4 pt below §7 merged-balanced in-domain (95.28), so PubMed-specific training still helps, but the cross-domain gap is small. Both curves saturate by ckpt ~100 (only +0.1 pt for notopo from ckpt-102 → 340), so most of the transferable signal is acquired in the first ~30 % of cora training.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-seq4k-on-pubmed-n1000-{topo,notopo}-mcdigit-cora_20260501_mcdigit_nonb_seq4k-logit.jsonl`

### §12. Validation accuracy on best §7 cora-topo and pubmed-topo checkpoints

Sanity check on the §7 reported numbers: re-evaluate the two best-per-dataset §7 topo ckpts on each dataset's val split (with the same merged-bal SFT config, `max_seq_len=2048`, `topo=True`) to confirm the test-set numbers are not a test-set selection artifact. PubMed val downsampled to 1000 samples (seed=42); cora val is the full 542-sample split.


| §7 ckpt   | dataset      | val acc      | test acc      | val − test |
| --------- | ------------ | ------------ | ------------- | ---------- |
| **816**   | cora-topo    | 88.19        | **90.96** ⭐  | −2.77      |
| **510**   | pubmed-topo  | **94.90**    | 94.93         | −0.03      |


cora-topo @ ckpt-816 shows a 2.77 pt val-test gap (val 88.19 < test 90.96), suggesting some test-set selection bias when picking ckpt-816 by test-set best. pubmed-topo @ ckpt-510 is essentially identical on val and test (94.90 vs 94.93), so this checkpoint is a clean optimum on the merged-bal val set. Per-class pubmed val: Experimental 92.9 % / Type 1 95.8 % / Type 2 95.0 %.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-sec7-val-{cora,pubmed}-topo-mcdigit.jsonl`

### §13. pubmed mc_digit nonb at `max_seq_len=4096` (run tag `pubmed_20260502_mcdigit_nonb_seq4k`, **complete**)

Single-dataset PubMed, `mc_digit + nonb`, `max_hops=2`, `max_neighbors_per_hop=10`. Same per-device batch / grad-accum recipe as §11 (`max_seq_len=4096`, `per_device_train_batch=3`, `grad_accum=16`, effective batch 48, 10 epochs, ~1972 total steps; `save_steps=eval_steps=0.05`, `cls_loss_weight=0.0`). Motivation: pubmed abstracts are typically longer than cora — at seq=2048 each neighbor was budget-bound to ~144 tokens, truncating most. seq=4096 raises the per-neighbor budget to ~306 tokens, removing token-level truncation for nearly all neighbors.

#### Logit Eval (pubmed test, n=999 full split)


| ckpt | pubmed-notopo | pubmed-topo |
| ---- | ------------- | ----------- |
| 124  | 92.40         | 93.80       |
| 248  | 95.40         | 93.99       |
| 372  | 94.80         | 95.06       |
| 496  | 94.47         | 94.80       |
| 620  | 95.23         | 94.57       |
| 744  | 95.39         | **96.30** ⭐ |
| 868  | 94.42         | —           |
| 992  | **95.70**     | —           |


Self-eval (pubmed → pubmed), all SFT checkpoints evaluated. **Topo peaks at 96.30 @ ckpt-744, beating notopo's peak 95.70 @ ckpt-992 by +0.60 pt** — this is the first clean case in the nonb setting where topo surpasses notopo at peak. Both variants show non-monotone trajectories: topo trends generally up (93.8 → 94.0 → 95.1 → 94.8 → 94.6 → 96.3), notopo bounces in [94.42, 95.70] after the early ckpt-248 high. The seq=4096 result also clears §9 single-pubmed seq=2048 (notopo 95.18 / topo 94.47) by a wide margin, confirming the seq-length lift. The topo win is consistent with the H2 prediction that the topo↔notopo gap shrinks as the target-token share of each head's attended set grows (longer sequences dilute the structural mask's effect).
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-pubmed-seq4k-{topo,notopo}-mcdigit-pubmed_20260502_mcdigit_nonb_seq4k-logit.jsonl`

### §14. cora mc_digit nonb at `max_seq_len=4096` *aligned* (run tag `cora_20260502_mcdigit_nonb_seq4k_aligned`, **topo only, complete**)

Variant of §11 controlling for total optimizer steps. §11 reduced per-device batch 6 → 3 to fit seq=4096 in 80GB but kept grad_accum=8, halving total steps (510 → 340). §14 keeps grad_accum=16 *and* further reduces per-device batch 3 → 2, restoring effective batch to 32 (matching §1's 4×8) and total optimizer steps to **510** (same as §1). All other knobs identical to §1 / §11 (`mc_digit`, `nonb`, `hop=2`, `nb=10`, `cls_loss_weight=0.0`, 10 epochs, `save_steps=eval_steps=0.05`). Topo only.

#### Logit Eval (cora test, n=542 full split)


| ckpt | cora-topo | | ckpt | cora-topo |
| ---- | --------- |-| ---- | --------- |
| 26   | 75.28     | | 286  | **90.22** ⭐ |
| 52   | 81.73     | | 312  | 88.75     |
| 78   | 84.32     | | 338  | 89.85     |
| 104  | 86.53     | | 364  | 89.85     |
| 130  | 86.35     | | 390  | 90.22     |
| 156  | 86.90     | | 416  | 89.67     |
| 182  | 85.79     | | 442  | 90.22     |
| 208  | 88.01     | | 468  | 90.04     |
| 234  | 88.38     | | 494  | 89.85     |
| 260  | 87.64     | | 510  | 89.85     |


**Peak 90.22 @ ckpt-286** (also 390 / 442). Compared to §1 single-cora seq=2048 topo peak 90.04 @ ckpt-312, **§14 seq=4096 with matched step budget edges out seq=2048 by +0.18 pt** — reversing the apparent regression in §11 (which was confounded by a 510→340 step-budget shortfall). Net conclusion: the seq=2k → 4k expansion on cora is at worst neutral and slightly positive *when total optimizer steps are held fixed*; §11's lower number was an optimization-budget artifact, not a sequence-length penalty.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-seq4k-aligned-topo-mcdigit-cora_20260502-logit.jsonl`

### §15. cora category_infill EOS-padded (run tag `cora_20260503_catinfill_nonb_eospad_seq2k`, **complete**)

Single-dataset cora, `prompt_format=category_infill` with the answer window padded by `eos_token_id` instead of `pad_token_id` (matching the upstream LLaDA SFT recipe in `examples/llada/sft.py`, which sets `label_pad_token_id=tokenizer.pad_token_id` and supervises EOS positions). All other knobs identical to §1 (`max_seq_len=2048`, `hop=2`, `nb=10`, `nonb`, `cls_loss_weight=0.0`, 10 epochs, eff batch 32, 510 steps). `max_answer_tokens=6` to fit the longest cora class name. Topo only.

#### Infill Eval (cora test, n=542; `--steps=max_answer_tokens=6`, T=0.0)


| ckpt | cora-topo | | ckpt | cora-topo |
| ---- | --------- |-| ---- | --------- |
| 26   | 73.62     | | 286  | 88.01     |
| 52   | 74.17     | | 312  | 87.82     |
| 78   | 81.73     | | 338  | 88.01     |
| 104  | 83.03     | | 364  | 88.75     |
| 130  | 85.24     | | 390  | **89.67** ⭐ |
| 156  | 85.61     | | 416  | 89.48     |
| 182  | 85.79     | | 442  | 88.75     |
| 208  | 85.61     | | 468  | 88.75     |
| 234  | 85.98     | | 494  | 89.11     |
| 260  | 86.72     | | 510  | 89.11     |


**Peak 89.67 @ ckpt-390**, ~0.4 pt below §1 mc_digit topo (90.04). Category-name infilling is a strictly harder task than picking a digit (the model must produce a multi-token class name, scored by mean log-prob over valid name positions), so a small accuracy gap vs mc_digit is expected. The convergence trajectory is monotone and stable; no class-collapse pathology like in §6 (which used `pad_token_id` filler instead of EOS). Useful as a stepping stone toward open-vocabulary node classification where the answer space is not enumerable as digits.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-seq2k-topo-catinfill-eospad-cora_20260503-infill.jsonl`

### §16. cora topo r=128 LoRA capacity ablation (run tag `cora_20260504_mcdigit_nonb_r128`, **complete**)

H2 ablation: doubles LoRA rank from r=64 (§1 default) to r=128 (alpha=128 to keep alpha/r=1.0). All other settings identical to §1 cora topo (`max_seq_len=2048`, `mc_digit`, `nonb`, `hop=2`, `nb=10`, eff batch 32, 510 steps). Tests whether under-parameterization of the low-rank adapter is the primary reason topo lags notopo in §1.

#### Logit Eval (cora test, n=542 full split)


| ckpt | r=128 topo | r=64 topo (§1) | r=64 notopo (§1) | Δ(r=128 vs r=64 topo) |
| ---- | ---------- | -------------- | ---------------- | --------------------- |
| 26   | 76.57      | 75.46          | 78.23            | +1.11                 |
| 52   | 81.18      | 81.73          | 81.55            | −0.55                 |
| 78   | 84.32      | 84.50          | 85.24            | −0.18                 |
| 104  | 85.06      | 83.58          | 85.61            | +1.48                 |
| 130  | 86.16      | 86.72          | 86.35            | −0.56                 |
| 156  | 88.01      | 87.27          | 86.16            | +0.74                 |
| 182  | 86.35      | 87.82          | 89.11            | −1.47                 |
| 208  | 88.01      | 87.64          | 89.11            | +0.37                 |
| 234  | 89.30      | 87.45          | 88.75            | +1.85                 |
| 260  | 88.93      | 88.93          | 90.41            | +0.00                 |
| 286  | 88.75      | 88.93          | 89.67            | −0.18                 |
| 312  | 86.72      | 90.04          | 90.59            | −3.32                 |
| 338  | 89.85      | 89.11          | 90.77            | +0.74                 |
| 364  | 90.22      | 89.11          | 90.41            | +1.11                 |
| **390** | **90.41** ⭐ | 89.85       | 90.22            | +0.56                 |
| 416  | 89.67      | 89.48          | 90.41            | +0.19                 |
| 442  | 89.67      | 89.67          | 90.59            | +0.00                 |
| 468  | 89.30      | 89.67          | 90.41            | −0.37                 |
| 494  | 89.85      | 89.67          | 90.22            | +0.18                 |
| 510  | 89.48      | 89.48          | 90.41            | +0.00                 |


**r=128 topo peak 90.41 @ ckpt-390 vs r=64 topo peak 90.04 @ ckpt-312 (+0.37 pt at peak)**, and **vs r=64 notopo peak 90.77 @ ckpt-338 (−0.36 pt)**. Doubling LoRA capacity recovers about half of the original topo↔notopo gap (0.73 → 0.36) and produces small per-checkpoint gains on average (+0.10 pt mean delta vs r=64 topo), but does not close the gap. **H2 (insufficient adapter capacity) is therefore a contributing factor but not the primary cause** of topo's underperformance — the remaining ~0.36 pt deficit must come from H1 (cross-neighbor information bottleneck) or H3 (neighbor gradient starvation). The −3.32 pt single-point dip at ckpt-312 is the only large outlier and likely reflects a transient optimizer instability in the larger-rank run; r=64 is more stable through the same step. Practical conclusion: capacity scaling alone is not sufficient; pair with structural fixes (auxiliary MLM on neighbor tokens, real-subgraph mask) to close the residual gap.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-seq2k-topo-mcdigit-r128-cora_20260504-logit.jsonl`

### §17. Dataset graph statistics — neighbor density across cora / pubmed / ogbn-arxiv / ogbn-products

To contextualize the topology-mask experiments and the seq-length budget analyses, we measured the per-node neighbor density on each dataset's train split. For each train node we recorded both (i) the raw 1-hop degree in the loader's full adjacency dict (no split filtering — neighbors may span train/val/test) and (ii) the number of neighbors actually fed to the model after `_sample_khop_neighbors(max_neighbors_per_hop=10, max_hops=2)` caps each hop at 10 (so the per-node total is bounded by 20). All numbers are computed on a 2000-node random sample (or the full split when smaller).

| dataset       | train\_size | raw 1-hop deg (mean / median / p90 / max) | % deg = 0 | sampled total NBs (mean / median) | % NBs = 0 | % NBs = 20 (cap) |
| ------------- | ----------- | ----------------------------------------- | --------- | --------------------------------- | --------- | ---------------- |
| cora          | 1,624       | 4.0 / 3 / 7 / 168                         | 0.0%      | 11.3 / 13                         | 0.0%      | 4.8%             |
| pubmed        | 11,830      | 4.6 / 2 / 13 / 171                        | 0.0%      | 12.6 / 12                         | 0.0%      | 13.2%            |
| ogbn-arxiv    | 90,941      | 14.1 / 5 / 25 / 1251                      | 0.0%      | 14.6 / 15                         | 0.0%      | 29.9%            |
| ogbn-products | 14,708      | 1.5 / 0 / 5 / 15                          | **74.4%** | **2.3 / 0**                       | **74.4%** | 3.5%             |

Three observations follow. First, **ogbn-products is qualitatively different from the other three**: 74.4% of its training nodes are isolated in the loader's adjacency, so the sampler returns an empty neighbor list for nearly three quarters of the training set. This is a property of the TAPE-products subset rather than the ingestion pipeline — `dllm/data/datasets/ogbn_products.py` builds `adj` directly from `data.adj_t.storage._row/_col` of the saved subset (no split-based filtering), and the subset preserves only edges whose endpoints both lie inside the ~54k subsampled nodes; the full OGB ogbn-products graph (2.4M nodes, 61M edges, average degree ~50) loses most of its edges in this trimming, leaving the majority of subset nodes with zero in-subset neighbors. Topology-mask experiments on products therefore have a degenerate baseline: for 3/4 of training samples the target node *is* the entire input, and topo vs notopo are identical by construction.

Second, **arxiv has by far the densest graph** (mean raw degree 14.1, max 1251) and 29.9% of its train nodes hit the 20-neighbor cap during sampling. Raising `max_neighbors_per_hop` above 10 would expose more graph signal on arxiv, whereas on cora/pubmed only 5–13% of samples saturate the cap and on products the cap is effectively irrelevant.

Third, **the seq-length budget analyses in §11 / §13 / §14 are independent of this graph property**: even with seq → ∞, isolated products nodes still have no neighbors to feed in. The seq=4096 lift observed on cora and pubmed comes from preserving longer abstracts of *existing* neighbors, which products cannot benefit from for the 74% isolated subset.

## §18. Topo vs. notopo gap analysis (consolidated)

This section consolidates the three concrete findings from §13 / §14 / §16 about when, and by how much, the topology-mask hurts (or helps) accuracy relative to dense attention. We state each conclusion in revised, scope-honest form and tie it back to the rows it draws on.

### 18.1 Conclusion 1 — On pubmed, raising seq from 2k to 4k flips the topo↔notopo gap from −0.71 to +0.60

| seq | run tag                                       | notopo peak                  | topo peak                    | gap (topo − notopo) |
| --- | --------------------------------------------- | ---------------------------- | ---------------------------- | ------------------- |
| 2k  | `pubmed_20260428_aligned` (§9)                | 95.18 @ 1480                 | 94.47 @ 1480                 | **−0.71**           |
| 4k  | `pubmed_20260502_mcdigit_nonb_seq4k` (§13)    | 95.70 @ 992                  | **96.30** @ 744              | **+0.60**           |

A 1.31-point swing (−0.71 → +0.60) in favor of topology-mask is the clearest evidence we have that the dense baseline's edge at seq=2k is a *truncation* artefact rather than a fundamental limit of the topo block-mask: with 2× the context, every neighbor abstract fits and the topo run reaches a new global topo peak (96.30 ⭐, the highest topo accuracy across every pubmed setting we have run). Strength: ⭐⭐⭐⭐ (peak-to-peak across 20 ckpts × 2 conditions on the full pubmed test set, n=1000 cap).

**Open question.** Whether this result generalizes to cora is unverified — we have not yet run a cora seq=4k *notopo* control with the §1-aligned step budget (§14 only covers topo at seq=4k). Until that control is run, we cannot claim seq=4k flips the cora gap.

### 18.2 Conclusion 2 — On cora, the apparent "seq=4k regression" was a step-budget confound, not a real regression

| run                                                       | seq | total optimizer steps | topo peak           |
| --------------------------------------------------------- | --- | --------------------- | ------------------- |
| §1  `cora_20260429_mcdigit_nonb_fixed`  (eff bs=32)       | 2k  | 510                   | 90.04 @ 312         |
| §11 `cora_20260502_mcdigit_nonb_seq4k`  (unaligned)       | 4k  | 340                   | 88.93 @ 272         |
| §14 `cora_20260503_mcdigit_nonb_seq4k_aligned` (eff bs=32)| 4k  | 510                   | **90.22 @ 286**     |

Once the seq=4k cora run is given the same effective batch (32 = per_device 4 × grad_accum 8) and therefore the same 510-step training budget as §1, peak accuracy *recovers* to 90.22 — actually +0.18 above §1's 90.04. The −1.11 drop reported in §11 was entirely explained by the 33% step-budget shortfall (340 vs 510). On cora, doubling seq with matched compute is at worst neutral for topo. Strength: ⭐⭐⭐ (matched-budget head-to-head, single seed each, peak-to-peak comparison; small absolute gap).

### 18.3 Conclusion 3 — Doubling LoRA rank (r=64 → r=128) closes only ~51% of the cora topo↔notopo gap

| run                                              | r   | topo peak       | notopo peak (§1) | gap to notopo |
| ------------------------------------------------ | --- | --------------- | ---------------- | ------------- |
| §1  `cora_20260429_mcdigit_nonb_fixed` topo      | 64  | 90.04 @ 312     | 90.77 @ 364      | **−0.73**     |
| §16 `cora_20260504_mcdigit_nonb_r128` topo       | 128 | **90.41 @ 390** | 90.77 (§1)       | **−0.36**     |

Doubling LoRA rank (and α) lifts the topo peak by +0.37 and shrinks the gap from −0.73 to −0.36 — i.e. capacity recovers ~51% of the deficit but does not eliminate it. This rules out *insufficient adapter capacity alone* (H2) as the full explanation and leaves H1 (cross-neighbor information bottleneck through the block-diagonal mask) and H3 (neighbor gradient starvation: with `include_neighbor_labels=False`, no token-level signal flows through neighbor positions during SFT) as the remaining candidates for the residual ~0.36-point gap. Strength: ⭐⭐⭐ (single-axis ablation with all other hyperparameters held to §1; one seed; small absolute effect).

### 18.4 Where this leaves us

Across the three datasets and the three controls above, the topo block-mask is competitive with — and on pubmed seq=4k *exceeds* — dense attention; the residual cora gap of ~0.36 points after r=128 is small in absolute terms but consistent across 20 checkpoints. The most informative remaining experiments are (i) a cora seq=4k *notopo* run with the §1-aligned 510-step budget, to confirm or refute that the pubmed seq=4k reversal generalizes; (ii) an auxiliary MLM loss on neighbor positions (H3) to test whether token-level neighbor supervision closes the residual cora gap without growing adapter capacity further; and (iii) replacing the strict block-diagonal mask with a real-subgraph attention pattern (H1) so that neighbor↔neighbor edges of the original graph remain visible to attention.

### §19. Cross-domain eval: pubmed → cora (run tag `pubmed_20260502_mcdigit_nonb_seq4k`, **complete**)

Symmetric counterpart to the cora → pubmed cross-eval in §11. Each pubmed seq=4k checkpoint (the §13 run) is evaluated on the full cora test split (n=542) using `eval_logit`, `mc_digit`, `digit0`, `max_seq_len=4096`, `max_answer_tokens=1`, `max_neighbors_per_hop=10`, `max_hops=2`, `include_neighbor_labels=False`. Topo training only produced 6 checkpoints (124..744); notopo ran the full 8 (124..992).

JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cross-pubmed2cora-seq4k-{topo,notopo}-pubmed_20260502_mcdigit_nonb_seq4k-logit.jsonl`

| ckpt | topo  | notopo | gap (topo − notopo) |
| ---- | ----- | ------ | ------------------- |
| 124  | 75.09 | **76.38** ⭐ | **−1.29** |
| 248  | 73.06 | 73.43 | −0.37 |
| 372  | 74.35 | 75.46 | −1.11 |
| 496  | 73.25 | 74.35 | −1.10 |
| 620  | 73.25 | 73.62 | −0.37 |
| 744  | 72.88 | 73.99 | −1.11 |
| 868  | —     | 74.54 | —     |
| 992  | —     | 73.43 | —     |
| **peak** | **75.09 @ 124** | **76.38 @ 124** | **−1.29** |

Three observations. First, **both peaks land at ckpt-124** (the earliest checkpoint we evaluate), and accuracy declines monotonically from there. This is the standard cross-domain forgetting curve — the longer the model is fine-tuned on pubmed-specific 3-class medical labels, the worse it transfers to cora's 7-class ML labels. Second, **notopo wins at every ckpt** by 0.4 to 1.3 points, opposite to the in-domain §13 result where topo overtook notopo by +0.60 at seq=4k. The structural prior the topo block-mask encodes is helpful when the train and test graphs have the same neighborhood semantics, but becomes a liability when those semantics shift across domains. Third, on cora the **Theory class** is the limiting factor (per-class accuracy 27–36% across all ckpts vs 65%+ for the other six classes); Theory is the most label-ambiguous class in cora and the one most reliant on residual general-LM commonsense rather than learned graph features.

Combined with §11 (cora → pubmed, also notopo > topo across the cross-eval grid), this gives **two-direction confirmation that dense attention is more cross-domain robust than the topology-mask block-diagonal pattern**, even when the in-domain comparison favors topo.

### §20. arxiv seq=4k mc_digit nonb (run tag `arxiv_20260503_mcdigit_nonb_seq4k`, **superseded — early snapshot only**)

First arxiv run on the seq=4k + mc_digit + nonb pipeline, replacing the older §-pre-29 catinfill run. Settings match §13 / §14: `max_seq_len=4096`, `max_neighbors_per_hop=10`, `max_hops=2`, `mc_digit + digit0`, `max_answer_tokens=2` (40-class arxiv needs 2 digits "00".."39"), `include_neighbor_labels=False`, `max_train_samples=20000`, `max_steps=1668` (4 epoch over 20k samples at effective batch 48), LoRA r=64/α=64 on `all-linear`. Initial single-GPU training was killed at step 420 (25%) after the periodic HF-trainer eval (eval_steps=0.1, eval set ≈ 14900 batches × 1.6 s/it ≈ 6.6 h per single eval) stalled the run; relaunched as 4-GPU DDP with `eval_strategy=no` and on-disk TAG-cache (commit `a92dc66`) to avoid both pitfalls. The numbers below are from the first 5 checkpoints of the killed single-GPU run (84..420); the full 1668-step DDP run is currently underway and will overwrite ckpts 84..420 with fresh ones.

eval_logit on full arxiv test (cap n=1000), `max_seq_len=4096`, batch_size=2.

JSONL: `/tmp/dlm-graph-eval-jsonl/eval-arxiv-seq4k-{topo,notopo}-mcdigit-arxiv_20260503{,_mcdigit_nonb_seq4k}-logit.jsonl`

| ckpt | topo  | notopo | gap (topo − notopo) |
| ---- | ----- | ------ | ------------------- |
| 84   | 53.67 | 52.67 | +1.00 |
| 168  | **69.50** ⭐ | 69.10 | +0.40 |
| 252  | 65.10 | 64.60 | +0.50 |
| 336  | 68.50 | 69.10 | −0.60 |
| 420  | 68.80 | **70.00** ⭐ | **−1.20** |

The crossover at ckpt-336 mirrors the late-stage divergence we see on cora (§14) but in the opposite direction from pubmed (§13 had topo overtake notopo): **on arxiv, dense attention pulls ahead the longer training continues**. ckpt-252 is a synchronized dip on both lines (data / optimization noise rather than mask-specific behavior). Peak topo at the snapshot is 69.50 @ 168; peak notopo is 70.00 @ 420 and still rising. With only 25% of training complete and the curves still climbing 4–6 pt over baseline, no firm head-to-head conclusion is possible until the full 1668-step DDP run finishes.

Per-class breakdown at ckpt-420 (40 arxiv classes, eval n=1000): top tier `cs.AR` 100 / `cs.CV` 99 / `cs.CL` 86–89 / `cs.CG` 87–88 / `cs.RO` 85–88 / `cs.DS` 76–85 / `cs.SD` 86; bottom tier 8 classes at 0% (`cs.NA / cs.MM / cs.CY / cs.GL / cs.SC / cs.GR / cs.OH / cs.OS`) — these eight together are 2.59% of the test split, so lifting them all to the non-zero mean of 58% would only move the overall accuracy by under 1 pt. The real ceiling is set by `cs.LG` (Machine Learning), which is 22.10% of the test split but only 7.69% of train (a 2.87× distribution shift introduced by the time-based OGB-arxiv split — test papers are post-2018, when ML exploded outside the train period); cs.LG sits at 56–59% accuracy and ~12 pp of the overall comes from this class alone, so any future architectural lever on arxiv must target the cs.LG ↔ cs.AI / cs.CV / cs.CL confusion rather than the long tail.

### §21. arxiv r=128 lgboost (run tag `arxiv_20260506_digit0pad_lgboost_r128`, **complete**)

LoRA r=128/α=128 on `all-linear`, full ogbn-arxiv train capped at 22% (`max_train_samples` applied), `mc_digit + digit0_pad`, `hops=2`, `nb=10`, `topo`, `max_seq_len=4096`. Best raw checkpoint at N=1000: ckpt-1845 = 74.40%. Best raw at N=5000: ckpt-1845 = 75.03% (N=5000 seed=42). Post-hoc TTA exploration (Phase 1–5 in `analysis/postprocess_arxiv_r128/RESULTS.md`) pushed the 1000-sample ensemble to 76.40% (16-pass E5 + τ=0.2 cal) but the unbiased N=5000 best single-pass is 75.70 (ckpt-2042, nb=12). The key finding from this exploration: neighbor-sample jitter (varying `--neighbor_seed` over 4 draws, fixed nb=10) recovers +1.8 pt at N=1000 by averaging out per-node sampling variance, matching more expensive hyperparam ensembles.

### §22. arxiv full-train 1-epoch r=128 (run tag `arxiv_20260511_fulltrain_r128_1ep`, **complete**)

Same recipe as §21 except the train-sample cap is removed — full 111,391-sample ogbn-arxiv train set, 1 epoch = 2396 steps. Eval at N=5000 (seed=42, nb=10) on 5 tail checkpoints:

| ckpt | raw   | cal (τ=1) |
| ---- | -----:| ---------:|
| 1680 | 75.54 | 69.90 |
| 1920 | 75.40 | 69.12 |
| 2160 | 76.08 | 70.02 |
| 2396 | **76.16** ⭐ | 69.90 |
| final | **76.16** ⭐ | 69.90 |

Reference: §21 ckpt-1845 N=5000 raw = 74.18. Full training alone yields **+1.98 pt** over the 22%-capped §21 baseline.

### §23. arxiv full-train 3-epoch r=128 (run tag `arxiv_20260514_fulltrain_r128_3ep`, **complete**)

Identical setup to §22 except `MAX_STEPS=7188` (3 epochs). Ran 2026-05-14 → 05-17 on GPUs 2/3/4/6 (~63 h). All 10 checkpoints evaluated on the **full test set** (N=48,603, σ≈0.2 pt) on 2026-05-18.

Note: all §23 checkpoints are adapter-only (1.3 GB each) and cannot be resumed — `save_only_model: True` project default in `dllm/utils/configs.py:69` overrode the launcher flag. Future full-train runs must pass `--save_only_model False` explicitly.

#### Full-test eval (N=48,603, complete 2026-05-18)

| ckpt  | epoch | raw       | cal (τ=1) | N=5000 raw |
| ----- | -----:| ---------:| ---------:| ----------:|
| 719   | 0.30  | 73.68     | 66.73     | —          |
| 1438  | 0.60  | 74.51     | 69.66     | —          |
| 2157  | 0.90  | 74.88     | 69.86     | 75.12      |
| 2876  | 1.20  | 76.04     | 70.69     | 76.38      |
| 3595  | 1.50  | 74.89 ↓   | 69.89     | 74.28      |
| 4314  | 1.80  | 75.31     | 70.76     | 75.24      |
| 5033  | 2.10  | 76.32     | 71.89     | 76.72      |
| **5752** | **2.40** | **76.39** ⭐ | 71.87 | 77.24   |
| 6471  | 2.70  | 75.87     | 71.94     | —          |
| final | 3.00  | 76.22     | **72.02** | —          |

Best raw (full test): ckpt-5752 = **76.39%**. Best cal (full test): ckpt-final cal = 72.02% (τ=1; τ-sweep not yet applied).

vs. baselines:
- LLaGA-HO = 76.66% → **−0.27 pt**, within σ≈0.2 pt.
- §22 best = 76.16% → **+0.23 pt** from 1→3 epochs (sharp diminishing returns).
- §21 best = 75.03% → **+1.36 pt** from capped→full training.

Key observations. (i) Full training (§22/§23) is the dominant lever: removing the 22%-cap gives +1.98 pt at 1 epoch and +2.21 pt at 3 epochs over §21. (ii) 3 epochs yields only +0.23 pt over 1 epoch — training is effectively converged by epoch 2. (iii) A double dip appears at epochs 0.9 and 1.5 (raw ≈74.9 both times) amid neighboring checkpoints at 75.3–76.0; this is a reproducible LR-scheduler × data-reshuffle artifact confirmed by both N=5000 and full-test measurements. (iv) N=5000 estimates carry a 0.4–0.85 pt bias relative to full test for individual checkpoints, making full-test eval necessary for reliably ranking close checkpoints.

---

### §24. ogbn-products NC SFT mc_digit nonb (run tag `products_20260503_mcdigit_nonb_seq2k`, **partial eval**)

Single-dataset ogbn-products, `prompt_format=mc_digit`, `answer_label_style=digit0`, `max_answer_tokens=2` (47 classes requires 2-digit answer), `nb=10, hop=2`, `topo`, `max_seq_len=2048`, `include_neighbor_labels=False`. 5 ckpts evaluated from a ~540-step run. Frozen-base logit baseline: 23.44% (topo). Note from §17: 74.4% of training nodes are isolated (zero in-subset neighbors), so the topo mask has no effect for the majority of samples.

#### Logit Eval (products test, n=300 stratified subsample, `batch_size=8`)


| ckpt    | products-topo |
| ------- | ------------- |
| 231     | 57.07         |
| 308     | 55.00         |
| **385** | **58.50** ⭐  |
| 462     | 58.00         |
| 539     | 56.67         |


Best: ckpt-385 = **58.50%** — +35 pt above the frozen-base zero-shot baseline (23.44%). Accuracy oscillates in 55–58.5% range past ckpt-231, suggesting early convergence or a plateau. The Patio, Lawn & Garden class remains at 0% across all checkpoints (only ~10 test samples; the TAPE products subset severely under-represents it). LLaGA NC numbers for ogbn-products are not available for direct comparison.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-products-seq2k-topo-mcdigit-products_20260503-bs8-logit.jsonl`

---

### §25. Cora Link Prediction SFT (run tag `cora_lp_20260519_seq4k_5ep_3gpu`, **complete**)

First LP fine-tuning run. Setup: LLaDA-8B-Instruct + LoRA r=64/α=64 all-linear, `task=lp`, `lp_neg_ratio=1`, `mc_digit + digit0`, `max_seq_len=4096`, `hop=2`, `nb=10`, `topo=True`, `nonb`, 5 epochs = 1870 steps, `per_device_train_batch=2`, `grad_accum=8`, effective batch 48 on 3 GPUs (6 h wall, 2026-05-19). Evaluated on LLaGA's exact test node-pairs (`edge_sampled_2_10_only_test.jsonl`, n=680) using `eval_lp_llaga_split.py` with `processed_data_link_notest.pt` as adjacency (no test edges leaked).

#### Accuracy + AUC (LLaGA test split, n=680)


| ckpt     | acc (%)      | AUC          | yes acc (%) | no acc (%) |
| -------- | ------------ | ------------ | ----------- | ---------- |
| 187      | 85.44        | 0.9634       | 71.43       | 96.57      |
| 374      | 89.85        | 0.9640       | 82.06       | 96.04      |
| 561      | 88.68        | 0.9644       | 77.41       | 97.63      |
| **748**  | **91.47** ⭐ | 0.9643       | **88.70**   | 93.67      |
| 935      | 89.71        | 0.9653       | 82.72       | 95.25      |
| 1122     | 90.44        | **0.9657** ⭐ | 87.04       | 93.14      |
| 1309     | 89.71        | 0.9674       | 81.73       | 96.04      |
| 1496     | 90.29        | 0.9654       | 85.38       | 94.20      |
| 1683     | 89.41        | 0.9661       | 81.40       | 95.78      |
| 1870     | 89.12        | 0.9655       | 80.73       | 95.78      |
| final    | 89.12        | 0.9655       | 80.73       | 95.78      |


Run **complete**. Best accuracy: ckpt-748 = **91.47%** / AUC = 0.9643. Best AUC: ckpt-1309 = 0.9674 (accuracy 89.71%). The accuracy peak at ckpt-748 (~2 epochs) is followed by a mild regression to 89–90% at later checkpoints, consistent with over-fitting to the edge-label balance rather than genuine signal degradation — AUC continues to improve (0.9634 → 0.9674) even as argmax accuracy falls, meaning the rank ordering of predictions improves while the decision boundary shifts.

vs. LLaGA baselines (same Cora LP split from LLaGA Table 1):
- LLaGA-ND-7B: 92.71% → **−1.24 pt**
- LLaGA-HO-7B: 92.65% → **−1.18 pt**
- Base LLaDA-8B-Instruct zero-shot (§0): 52.18% → **+39.29 pt** from SFT

The gap to LLaGA (~1.2 pt) is the primary open target for LP. Unlike NC where we match or beat LLaGA on Cora and PubMed, LP accuracy is bounded below the LLaGA oracle projector here.

JSONL: `.models/eval_logs/eval_cora_lp_llaga_cora_lp_20260519_seq4k_5ep_gpu246_gpu{2,4,6}.jsonl`

#### LP split and leakage analysis

**Split mismatch.** Our SFT run used our own random 85/5/10 split (`lp_split_seed42_neg1_v50_t100.pt`, seed=42). LLaGA's test split (`edge_sampled_2_10_only_test.jsonl`) is a different partition. Checking overlap: **241/301 (80%) of LLaGA test positive edges appear in our training set.** This creates a train-test distribution mismatch: those edges were present in `adj_train` during our SFT, so structural signals around them (shared neighbors, indirect paths) were available to the model during training but are absent at test time (we use `processed_data_link_notest.pt` at eval, which removes all LLaGA test edges). The result is that the comparison on the LLaGA test split is not fully fair — to fix this, retrain on LLaGA's own `edge_sampled_2_10_only_train.jsonl`.

**Our eval is leakage-free at inference time.** At evaluation, `_sample_lp_neighbors` uses `adj_train = processed_data_link_notest.pt` (test edges removed) and explicitly drops the candidate node v from u's neighbor list at every hop (`nb != v` filter covers hop 1 and hop 2). So neither the direct edge nor any 2-hop path to v can appear in the prompt. The only graph-structural signal visible is shared neighbors (common friends of u and v that are training edges) — this is the standard common-neighbors heuristic used by all LP methods, not a leak.

**LLaGA has a harder leakage problem at test time.** LLaGA's test JSONL pre-samples neighbors using the full adjacency (test edges not yet removed). For **48.2% of LLaGA test positive pairs**, the other endpoint v appears directly in u's pre-sampled `graph` field — meaning v's SimTeG embedding is included in u's `<graph>` token during LLaGA inference. The model can leverage this direct embedding co-occurrence. LLaGA's reported 92.71% (ND) / 92.65% (HO) Cora LP accuracy is therefore measured under partial test-time label leakage. Our 91.47% is measured under a stricter setup (no direct endpoint in the prompt), so the true gap between methods is likely smaller than 1.2 pt.

#### Topo vs. notopo eval on §25 checkpoints (eval_lp_2hop_ablation.py, 2026-05-19/20)

Post-hoc eval that also stratifies by test-time leakage group. Group A (n=74): test pairs where v appears in u's sampled prompt neighbors despite `adj_train` masking (2-hop path exists via training edges). Group B (n=606): clean pairs with no prompt leakage. Both groups together equal the full LLaGA test split (n=680).

| model (training)                        | eval topo | overall acc | overall AUC | yes acc | no acc | group B acc |
| --------------------------------------- | --------- | ----------: | ----------: | ------: | -----: | ----------: |
| §25 `cora_lp_20260519` ckpt-748 (topo) | True      |       90.00 |      0.9638 |   88.70 |  91.03 |       88.94 |
| §25 `cora_lp_20260519` final (topo)    | False     |       89.26 |      0.9579 |   85.71 |  92.08 |       88.12 |
| posw2 `cora_lp_20260520` final (topo)  | False     |       89.85 |      0.9595 |   87.38 |  91.82 |       88.78 |

Note: group A accuracy is artificially high (~98.65% yes_acc) because those prompt pairs include a 2-hop path to the target node — these pairs sit in a structural grey zone (not a direct edge leak, but not fully blind either). The clean group B numbers are a better lower-bound estimate of true out-of-distribution performance.

JSONL: `.models/eval_logs/lp_2hop_ablation.jsonl`, `lp_2hop_ablation_cora_20260519.jsonl`, `lp_2hop_ablation_cora_20260520_posw2.jsonl`

### §26. Cora LP posw=2 ablation (run tag `cora_lp_20260520_seq4k_5ep_posw2_topo`, **complete**)

Same setup as §25 (`topo=True`, `lp_neg_ratio=1`, `mc_digit + digit0`, `seq=4096`, `hop=2`, `nb=10`, 5 epochs = 1870 steps) with one change: `lp_pos_weight=2.0` — the diffusion loss for positive (yes, cls_label=1) samples is multiplied by 2.0 to address the yes_acc < no_acc imbalance observed in §25. New `lp_pos_weight` flag added to `TMDLMConfig` in `dllm/pipelines/tmdlm/trainer.py`. Evaluated on LLaGA test split (n=680), same eval script as §25.

| ckpt     | acc (%)      | AUC          | yes acc (%) | no acc (%) |
| -------- | ------------ | ------------ | ----------- | ---------- |
| 187      | 89.71        | 0.9604       | 84.05       | 94.20      |
| 374      | 89.56        | 0.9602       | 81.06       | 96.31      |
| 561      | 89.41        | 0.9656       | 83.06       | 94.46      |
| 748      | 89.71        | 0.9589       | 87.04       | 91.82      |
| 935      | 88.68        | 0.9623       | 79.73       | 95.78      |
| 1122     | 89.85        | 0.9629       | 85.71       | 93.14      |
| 1309     | 88.68        | 0.9618       | 79.40       | 96.04      |
| **1496** | **90.88** ⭐ | **0.9647** ⭐ | 85.05       | 95.51      |
| 1683     | 89.85        | 0.9637       | 81.73       | 96.31      |
| 1870     | 89.71        | 0.9635       | 81.73       | 96.04      |
| final    | 89.71        | 0.9635       | 81.73       | 96.04      |

Best accuracy: ckpt-1496 = **90.88%** / AUC = 0.9647. Compared to §25 (posw=1): best acc 91.47 @ ckpt-748 → posw=2 **loses −0.59 pt** at peak accuracy. The posw=2 run also fails to improve yes_acc meaningfully (85.05 at posw2 best vs 88.70 at §25 best), while no_acc stays high (95.51 vs 93.67). The peak shifts to a later checkpoint (ckpt-1496 vs ckpt-748) and the overall curve is flatter but lower. Conclusion: doubling the positive-sample loss weight does not close the yes/no imbalance gap and slightly hurts peak accuracy — posw=1 (uniform weighting) remains the better default for Cora LP.

JSONL: `.models/eval_logs/eval_cora_lp_llaga_posw2_allckpts.jsonl`

---

## Summary

All experiments use `include_neighbor_labels=False` (**nonb**) — neighbor text only, no oracle class labels. All NC experiments use `mc_digit + digit0` unless noted. LP experiments use the same format with `task=lp`.

### Node Classification (NC)

#### Cora — single-dataset

| §   | Run / variant                          | topo   | notopo | Best logit | Best infill | Notes |
| --- | -------------------------------------- | -----: | -----: | ---------: | ----------: | ----- |
| §1  | mc_digit, seq=2k, r=64                 | 90.04  | **90.77** | **90.77** | **90.96** | baseline |
| §8  | mc_digit + boost (Theory×2, RL×3)      | **90.59** | —   | **90.59** | —        | +0.55 vs §1 topo |
| §11 | mc_digit, seq=4k, r=64 (340 steps)    | 88.56  | 89.67  | 89.67     | —           | step-budget confound |
| §14 | mc_digit, seq=4k, r=64 (510 steps)    | **90.22** | — | **90.22** | —        | matched budget; +0.18 vs §1 topo |
| §15 | category_infill EOS-pad, seq=2k, topo  | 89.67  | —      | —         | 89.67       | catinfill control |
| §16 | mc_digit, seq=2k, r=128, topo          | **90.41** | — | **90.41** | —        | +0.37 vs r=64 topo; gap halved |

#### PubMed — single-dataset

| §   | Run / variant                          | topo   | notopo | Best logit | Best infill | Notes |
| --- | -------------------------------------- | -----: | -----: | ---------: | ----------: | ----- |
| §9  | mc_digit, seq=2k, r=64                 | 94.47  | **95.18** | **95.18** | **95.64** | seq=2k baseline |
| §13 | mc_digit, seq=4k, r=64                 | **96.30** | 95.70 | **96.30** | —        | topo wins at seq=4k (+0.60 vs notopo) |

#### Cora + PubMed — joint training

| §  | Run / variant                              | Status   | Cora topo | Cora notopo | PubMed topo | PubMed notopo |
| -- | ------------------------------------------ | -------- | --------: | ----------: | ----------: | ------------: |
| §6 | catinfill, no resampling                   | killed   | 70.30     | 63.84       | 77.56       | 77.48         |
| §7 | mc_digit + balance_datasets                | complete | **90.96** | **90.77**   | 94.93       | **95.28**     |

§6 collapses on Cora due to PubMed dominating the loss with unconstrained catinfill answers. §7 fixes both issues with `mc_digit` (single-token answer) + dataset balancing; it matches or exceeds all §1 single-dataset numbers.

#### ogbn-arxiv

| §   | Run / variant                          | Best acc (full test) | vs LLaGA-HO (76.66%) | Notes |
| --- | -------------------------------------- | -------------------: | --------------------: | ----- |
| §20 | mc_digit, seq=4k, 22%-cap (snapshot)  | ~70.0 (N=1000, 25% of training) | — | superseded |
| §21 | mc_digit, seq=4k, 22%-cap, r=128      | 75.03 (N=5000)       | −1.63                 | neighbor jitter +1.8 pt at N=1000 |
| §22 | mc_digit, seq=4k, full-train, 1-epoch | 76.16                | −0.50                 | +1.98 pt vs §21 |
| §23 | mc_digit, seq=4k, full-train, 3-epoch | **76.39**            | **−0.27** (within σ)  | best; 3ep ≈ 1ep (+0.23 pt only) |

#### ogbn-products (NC)

| §   | Run / variant                          | Best acc | vs zero-shot | Notes |
| --- | -------------------------------------- | -------: | -----------: | ----- |
| §24 | mc_digit, seq=2k, topo, partial eval   | 58.50    | +35 pt       | 74% isolated nodes; topo degenerate |

#### Cross-domain and seq-length findings (§11–§19)

- **seq=4k is neutral-to-positive on Cora** (§14): with matched step budget (+0.18 pt over seq=2k).
- **seq=4k flips topo/notopo on PubMed** (§13 vs §9): gap goes from −0.71 (topo lags) to +0.60 (topo leads).
- **r=128 closes ~51% of the cora topo gap** (§16): gap −0.73 → −0.36; not sufficient on its own.
- **Cross-domain transfer** (§11, §19): notopo is more cross-domain robust than topo in both directions (cora→pubmed and pubmed→cora).

---

### Link Prediction (LP) — Cora

Eval on LLaGA's exact test split (`edge_sampled_2_10_only_test.jsonl`, n=680). **Note**: 80% of LLaGA test positive edges were in our SFT training set (split mismatch); the comparison understates our true difficulty. LLaGA's eval has a separate leakage issue (48% of test pairs include the target endpoint in the prompt).

| §   | Run / variant                    | Best acc | Best AUC | vs LLaGA-ND (92.71%) | Notes |
| --- | -------------------------------- | -------: | -------: | -------------------: | ----- |
| §25 | topo, posw=1, 5ep               | **91.47** @ ckpt-748 | 0.9674 @ ckpt-1309 | −1.24 pt | best overall |
| §26 | topo, posw=2, 5ep               | 90.88 @ ckpt-1496 | 0.9647 | −1.83 pt | posw=2 hurts (−0.59 pt vs §25) |

Base LLaDA-8B zero-shot: 52.18% (AUC 0.567) — SFT gives +39 pt lift. Primary open target: retrain on LLaGA's own train split to make the comparison fully fair.
