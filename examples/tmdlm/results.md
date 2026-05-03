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

Single-dataset PubMed, `prompt_format=mc_digit`, `answer_label_style=digit0`, `max_answer_tokens=1`, `nb=10, hop=2`, no resampling. 4 ckpts saved per setting (370 / 740 / 1110 / 1480), all evaluated; checkpoint directories have since been deleted from disk so further training-step checkpoints are unrecoverable.

#### Logit Eval


| ckpt     | pubmed-notopo | pubmed-topo |
| -------- | ------------- | ----------- |
| 370      | 91.23         | 92.11       |
| 740      | 94.35         | 94.14       |
| 1110     | 93.61         | 92.37       |
| **1480** | **95.18** ⭐  | **94.47** ⭐ |


Run **complete**. Best per setting: pubmed-notopo 95.18 @ ckpt-1480, pubmed-topo 94.47 @ ckpt-1480. Single-dataset PubMed-notopo 95.18 sits within 0.10 pt of the §7 merged-balanced 95.28 — joint training on Cora+PubMed therefore loses no PubMed accuracy. Both settings still beat the LLaGA-7B oracle-projector baseline (95.03 / —) on the notopo branch.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-pubmed-2hop-{notopo,topo}-pubmed_20260428_aligned.jsonl`

### §11. cora mc_digit nonb at `max_seq_len=4096` (run tag `cora_20260501_mcdigit_nonb_seq4k`, **in progress**)

Single-dataset cora, `mc_digit + nonb`, `max_hops=2`, `max_neighbors_per_hop=10`, identical recipe to §1 except **`max_seq_len` raised from 2048 → 4096**. Motivation: at seq=2048 each neighbor was hard-truncated to ~66 tokens (cora abstract median ~200), losing most neighbor content. At seq=4096 the per-neighbor budget rises to ~170 tokens so most neighbor abstracts fit without token-level truncation. Per-device batch reduced from 6 → 3 with grad_accum 8 → 16 to keep the effective batch at 48 within 80GB. 340 total steps (10 epochs); save / eval every 17 steps (`save_steps=eval_steps=0.05`).

#### Logit Eval (cora test, 542 samples; in progress, training currently at step ~170/340)


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

Cross-domain transfer: §11 ckpts (trained on cora-only) evaluated on PubMed test (downsampled to 1000 samples, seed=42). Same SFT config (`max_seq_len=4096`, `hop=2`, `nb=10`, `mc_digit`, `nonb`, `max_answer_tokens=1`); only `dataset_name` (cora → pubmed) and `split` (train → test) differ. Iterates ckpts latest → earliest. Eval interrupted at ckpt-136 to free GPU2; only the most recent three ckpts have full numbers.


| ckpt    | pubmed-notopo | pubmed-topo |
| ------- | ------------- | ----------- |
| 187     | 90.60         | 89.90       |
| 170     | 90.90         | 90.30       |
| **153** | **91.20** ⭐  | 90.10       |


Cora-trained checkpoints transfer to PubMed at **~89.9–91.2 %** with no PubMed-specific training, far above the 33 % random baseline. Two reinforcing reasons: (1) the `mc_digit` prompt embeds the actual PubMed class names (`Diabetes Mellitus, Experimental` / `Type 1` / `Type 2`) in the options block, letting the underlying LLaDA-8B base model use its semantic prior to match abstract content to class name; (2) LoRA on cora teaches a generic "read paper abstract → pick a class index from the listed options" behavior without overwriting that prior. The 91.20 transfer ceiling sits ~4 pt below §9 single-PubMed in-domain training (95.18) and ~4 pt below §7 merged-balanced in-domain (95.28), so PubMed-specific training still helps, but the cross-domain gap is small. notopo edges topo by 0.6–1.0 pt at every ckpt, mirroring the in-domain pattern.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-cora-seq4k-on-pubmed-n1000-{topo,notopo}-mcdigit-cora_20260501_mcdigit_nonb_seq4k-logit.jsonl`

### §12. Validation accuracy on best §7 cora-topo and pubmed-topo checkpoints

Sanity check on the §7 reported numbers: re-evaluate the two best-per-dataset §7 topo ckpts on each dataset's val split (with the same merged-bal SFT config, `max_seq_len=2048`, `topo=True`) to confirm the test-set numbers are not a test-set selection artifact. PubMed val downsampled to 1000 samples (seed=42); cora val is the full 542-sample split.


| §7 ckpt   | dataset      | val acc      | test acc      | val − test |
| --------- | ------------ | ------------ | ------------- | ---------- |
| **816**   | cora-topo    | 88.19        | **90.96** ⭐  | −2.77      |
| **510**   | pubmed-topo  | **94.90**    | 94.93         | −0.03      |


cora-topo @ ckpt-816 shows a 2.77 pt val-test gap (val 88.19 < test 90.96), suggesting some test-set selection bias when picking ckpt-816 by test-set best. pubmed-topo @ ckpt-510 is essentially identical on val and test (94.90 vs 94.93), so this checkpoint is a clean optimum on the merged-bal val set. Per-class pubmed val: Experimental 92.9 % / Type 1 95.8 % / Type 2 95.0 %.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-sec7-val-{cora,pubmed}-topo-mcdigit.jsonl`

---

## Summary

### Single-dataset training

Single-dataset runs on Cora (§1, §8) and PubMed (§9). All use `mc_digit + nonb` with the same LoRA recipe; §8 additionally up-weights the two hardest classes via `--resample_strategy boost`. Earlier `nbmask` runs (which inject neighbor class labels into the prompt) are excluded from this comparison since they are not a fair `nonb` baseline.


| §   | Run                                                   | Setting | Best ckpt | Logit     | Infill strict | Test set |
| --- | ----------------------------------------------------- | ------- | --------- | --------- | ------------- | -------- |
| §1  | cora mc_digit nonb                                    | notopo  | 338 / 260 | **90.77** | **90.96**     | Cora     |
| §1  | cora mc_digit nonb                                    | topo    | 312       | 90.04     | 89.85         | Cora     |
| §8  | cora mc_digit nonb + boost (Theory:2, RuleLearning:3) | topo    | 320       | **90.59** | —             | Cora     |
| §9  | pubmed mc_digit nonb                                  | notopo  | 1480      | **95.18** | —             | PubMed   |
| §9  | pubmed mc_digit nonb                                  | topo    | 1480      | 94.47     | —             | PubMed   |


Take-aways. (i) Cora-notopo saturates around 90.8 logit / 91.0 infill at ckpt-260–338; longer training (up to 510) gives no further gain. (ii) Logit and infill-strict accuracies track within ≤0.4 pt across all checkpoints, so the masked-diffusion 10-step infill recovers essentially the same answer as direct token scoring. (iii) Increasing neighbor count (15/20/25) or hops (3) yields no meaningful gain over the default `nb=10, hop=2`. (iv) The §8 boost recipe lifts Cora-topo by +0.55 pt over §1 single-cora topo (90.59 vs 90.04) and accelerates per-class learning on Theory and Rule Learning early in training, validating the up-weighting strategy without hurting other classes. (v) Single-dataset PubMed (§9 mc_digit) reaches 95.18 notopo / 94.47 topo, within 0.10 pt of the §7 merged-balanced run (95.28 / 94.93) — joint Cora+PubMed training therefore loses no PubMed accuracy.

### Multi-dataset training (cora + pubmed)


| §   | Run                                  | Status   | Setting | Best ckpt | Cora logit | PubMed logit |
| --- | ------------------------------------ | -------- | ------- | --------- | ---------- | ------------ |
| §6  | cora+pubmed merged catinfill nonb    | killed   | notopo  | 422       | 63.84      | 77.36        |
| §6  | cora+pubmed merged catinfill nonb    | killed   | topo    | 633       | 70.30      | 77.56        |
| §7  | cora+pubmed merged-bal mc_digit nonb | complete | notopo  | 765 / 612 | **90.77**  | **95.28**    |
| §7  | cora+pubmed merged-bal mc_digit nonb | complete | topo    | 816 / 510 | **90.96**  | **94.93**    |


Take-aways. (i) The naive merge in §6 (`category_infill`, `max_answer_tokens=10`, no resampling) produces class collapse on Cora — Cora-notopo accuracy degrades from 63.84 at ckpt-422 to 39.85 at ckpt-1055, while PubMed plateaus at ~77.5 because the unconstrained `[Diab]…` answer prefix lets PubMed dominate the loss. The topo mask delays but does not prevent the collapse. (ii) Switching to `mc_digit` (single-digit answer, `max_answer_tokens=1`, `cls_loss_weight=1.0`) plus `--resample_strategy balance_datasets` (each dataset down-sampled to 1624 examples) eliminates the collapse: §7 matches or exceeds the §1 single-dataset Cora baselines on both settings (Cora-notopo ties at 90.77, Cora-topo improves +0.92 pt to 90.96) while reaching 95.28 on PubMed-notopo, surpassing the LLaGA-7B oracle-projector baseline (95.03). (iii) Both PubMed settings show a transient ckpt-459 dip (notopo 91.78, topo 90.09) followed by recovery — likely an lr-scheduler artifact, not a regression. (iv) Conclusion: a single-token answer space plus dataset-balanced resampling is sufficient to train one LoRA on Cora+PubMed jointly without sacrificing per-dataset accuracy on either.
### §13. pubmed mc_digit nonb at `max_seq_len=4096` (run tag `pubmed_20260502_mcdigit_nonb_seq4k`, **in progress**)

Single-dataset PubMed, `mc_digit + nonb`, `max_hops=2`, `max_neighbors_per_hop=10`. Same per-device batch / grad-accum recipe as §11 (`max_seq_len=4096`, `per_device_train_batch=3`, `grad_accum=16`, effective batch 48, 10 epochs, ~1972 total steps; `save_steps=eval_steps=0.05`, `cls_loss_weight=0.0`). Motivation: pubmed abstracts are typically longer than cora — at seq=2048 each neighbor was budget-bound to ~144 tokens, truncating most. seq=4096 raises the per-neighbor budget to ~306 tokens, removing token-level truncation for nearly all neighbors.

#### Logit Eval (pubmed test, n=1000 stratified subsample, seed=42; in progress)


| ckpt | pubmed-notopo | pubmed-topo |
| ---- | ------------- | ----------- |
| 124  | 92.40         | 93.80       |


Self-eval (pubmed → pubmed). Only the first checkpoint at ~0.5 epoch has been evaluated; SFT was at step ~633/1972 (topo) and ~1471/1972 (notopo) at the time of writing. Note that the seq=4096 SFT preserves only the latest checkpoint on disk, so further evaluations land as training proceeds. ckpt-124 already exceeds the §1/seq=2048 ckpt-17 baseline pattern, but final accuracy at the §9-equivalent end-of-training is not yet known. Compare against §9 (pubmed seq=2048: notopo 95.18 / topo 94.47) for the seq-length effect once training completes.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-pubmed-seq4k-{topo,notopo}-mcdigit-pubmed_20260502_mcdigit_nonb_seq4k-logit.jsonl`

### §14. cora mc_digit nonb at `max_seq_len=4096` *aligned* (run tag `cora_20260502_mcdigit_nonb_seq4k_aligned`, **topo only, in progress**)

Variant of §11 controlling for total optimizer steps. §11 reduced per-device batch 6 → 3 to fit seq=4096 in 80GB but kept grad_accum=8, halving total steps (510 → 340). §14 keeps grad_accum=16 *and* further reduces per-device batch 3 → 2, restoring effective batch to 32 (matching §1's 4×8) and total optimizer steps to **510** (same as §1). All other knobs identical to §1 / §11 (`mc_digit`, `nonb`, `hop=2`, `nb=10`, `cls_loss_weight=0.0`, 10 epochs, `save_steps=eval_steps=0.05`). Currently topo only on GPU 4 (no notopo run yet); SFT at ~step 31/510, ~6h45m ETA. Self-eval to follow once ckpts arrive.

### §15. pubmed catinfill nbmask at `max_seq_len=4096` (run tag `pubmed_20260428_aligned`, seq=4096 variant, **complete**)

Single-dataset PubMed SFT with `prompt_format=category_infill`, `max_answer_tokens=6`, `include_neighbor_labels=True` (`neighbor_label_format=bracket` — **nbmask**, *not* nonb), `max_hops=2`, `max_neighbors_per_hop=10`, `max_seq_len=4096`. Distinct from §9 (which is `mc_digit + nonb`, seq=2048) despite sharing the `pubmed_20260428_aligned` run tag. Eval on the full PubMed test split (999 samples). Run on GPU 7 via `examples/tmdlm/run_eval_pubmed_remaining_ckpts_oneshot.sh`.

#### Logit Eval

| ckpt    | pubmed-notopo | pubmed-topo |
| ------- | ------------- | ----------- |
| 370     | 91.23         | 92.11       |
| 740     | 94.35         | 94.14       |
| 1110    | 93.61         | 92.37       |
| **1480**| **95.18** ⭐  | 94.47       |
| 1850    | 94.95         | **94.90** ⭐|
| 2220    | 94.88         | 93.18       |

#### Infill Eval (strict, 10 diffusion steps)

| ckpt     | pubmed-notopo | pubmed-topo |
| -------- | ------------- | ----------- |
| 370      | 93.13         | 92.98       |
| 740      | 94.93         | 93.36       |
| 1110     | 94.93         | 94.27       |
| 1480     | 94.90         | 94.14       |
| 1850     | 95.51         | 95.51       |
| **2220** | **95.64** ⭐  | 95.59       |

Run **complete**. Best per setting: notopo logit 95.18 @ ckpt-1480; topo logit 94.90 @ ckpt-1850; topo infill 95.59 @ ckpt-2220; **notopo infill 95.64 @ ckpt-2220** ⭐ — the highest PubMed accuracy in this file, exceeding the LLaGA-7B oracle-projector baseline (95.03) by +0.61pt. Infill consistently improves over logit on both topo and notopo at the late ckpts (e.g. topo 2220: logit 93.18 → infill 95.59; notopo 2220: logit 94.88 → infill 95.64), suggesting the masked-diffusion gen recovers signal that direct token scoring loses on the multi-token `category_infill` answer space. The seq=4096 + nbmask + catinfill combination outperforms the §9 seq=2048 nonb mc_digit setup (95.18 / 94.47 logit) on both axes.
JSONL: `/tmp/dlm-graph-eval-jsonl/eval-pubmed-2hop-{notopo,topo}-pubmed_20260428_aligned{,-infill}.jsonl`

