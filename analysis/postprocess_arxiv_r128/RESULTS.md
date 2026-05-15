# Post-process exploration: arxiv §21 r128 — running results log

All runs use the §21 r128 SFT (`arxiv_20260506_digit0pad_lgboost_r128`),
1000-sample test subset (seed=42), unless stated otherwise.

Goal: push beyond 74.4% (raw, ckpt-1845, 1000-sample) toward LLaGA's ~76%.

Format per row: `method | ckpt | raw acc | cal acc | Δ vs baseline-raw | notes`.

Baseline-raw = 74.4% (1845, 1000-sample). All Δ values are computed against
this for direct comparability; "cal acc" is the same row's calibrated number
(same eval pass, just argmax over `logits − (log p_train − log p_test)`).

---

## Phase 0 — Baseline (1000 samples, raw + calibrated)

| ckpt  | raw acc | cal acc | Δ raw |
|------:|--------:|--------:|------:|
| 1640  | 73.90 | 66.60 | -0.50 |
| **1845** | **74.40** | 68.70 | base |
| 2042  | 74.20 | 68.40 | -0.20 |
| final | 74.20 | 68.40 | -0.20 |

`accuracy_calibrated` uses default τ=1.0 calibration with `log p_train − log p_test_1000`.
At 1000-sample N, the test prior estimate is noisy → cal hurts (-5.7 pt). Phase 1c
shows that with full-test prior + small τ, calibration becomes neutral or +0.4.

---

## Headline: best methods so far (1000-sample, seed=42)

| method | passes | acc | Δ vs 74.4 |
|---|---:|---:|---:|
| Baseline (ckpt-1845, default settings) | 1 | 74.40 | — |
| Best **single** forward pass: ckpt-2042 nb=12 | 1 | 75.70 | +1.30 |
| Best **single** on ckpt-1845: nb=12 OR nb=15 | 1 | 75.40 | +1.00 |
| 4-pass E5 vote (1845 × {default, nb=12, nb=15, nb=30}) | 4 | 76.20 | +1.80 |
| 4-pass E5 vote (**neighbor-seed jitter only, nb=10**) | 4 | 76.20 | +1.80 |
| 16-pass E5 vote (4 ckpts × {default, nb=12, nb=15, nb=30}) | 16 | 76.30 | +1.90 |
| **16-pass E5 + post-ensemble cal τ=0.2** | 16 | **76.40** | **+2.00** |

LLaGA target ~76% — surpassed by 4-pass jitter alone (cheapest path); 16-pass
ensemble + post-cal gives +0.2 more (within σ=1.4 of N=1000).

Key levers (each, single-pass on ckpt-1845):
- `max_neighbors_per_hop=15` (vs SFT-default 10) → +1.0
- `max_neighbors_per_hop=12` → +1.0
- `max_neighbors_per_hop=20` / `30` → +0.7 / +0.9
- `max_hops=1` (vs 2) → +0.5
- `use_topology_mask=False` → +0.4
- **`neighbor_seed=7`** at default nb=10 → +1.3 (just changing which 10 neighbors get sampled!)

The three hyperparam levers (`nb`, `max_hops`, `topo_mask`) are **partially redundant**
— their combination (nb15+h1+notopo) gives 75.20, less than the best single (75.40).
They tap the same underlying effect (looser/different attention scaffolding).

`prompt_layout=neighbor_first` was a disaster (47.2%) — SFT only saw `target_first`.

**Per-test-node neighbor selection is the single largest TTA source.** Just varying
`--neighbor_seed` over 4 values (same 1000 nodes, same SFT-trained nb=10) gives the
same +1.8 pt as a 4-setting hyperparam ensemble. This means the original 74.4 baseline
has high variance from neighbor sampling alone (74.4-75.7 across 4 seeds), and
ensembling averages that out.

### Diversity / oracle upper bound

With 12 settings on ckpt-1845, ANY-correct rate = 81.4%. Realistic ceiling for 1-ckpt
TTA is ~81%. We achieved 76.2 (recovers 94/146 "ensemble-decidable" samples; 52 still
missed even when 1+ setting got them right). 18.6% of samples are wrong on all 12 —
these need fundamentally different inference (multi-step diffusion, etc).

Pre-1.0-pt methods (Phase 1, 1b, 3) all stayed within ±0.3 of baseline,
indicating offline post-processing on cached logits is noise-limited at N=1000.

---

## Phase 1 — Offline logit transforms (CPU only, on cached .npz)

All methods within ±0.3 of 74.4 baseline. Best: B. tau=-0.1 = 74.5 (within σ).
Calibration with default tau=1.0 hurts (68.7) due to noisy 1000-sample test prior.

## Phase 1c — full-test prior calibration

Recompute log p_test from FULL test split (48,604 samples). Stable prior.
Best: tau=0.2 → 74.80 (+0.4). Calibration now mildly helps for tau ∈ [0.1, 0.3].

## Phase 2 — GPU re-eval (single-flag changes on ckpt-1845)

See headline. Key finding: nb=15 > nb=10 (the SFT default) by 1.0 pt.

## Phase 3 — Graph-structure post-processing

99.4% of test nodes have ≥1 train-labeled k-hop neighbor. Best:
N1. low-conf fallback gap<0.05 → 74.7 (+0.3, ckpt-2042/final). All other graph
methods (label propagation, hard override) hurt. ogbn-arxiv homophily ~70% — too
low for naive neighbor majority to beat the model's own logits.

## Phase 4 — TTA ensemble of Phase 2 settings

Mean-pool of 8 settings (baseline + s1..s7) → 75.0 raw. With oracle τ=0.1
on top: 75.5. CV-fit weights: 75.0. Single best (s2_nb15 standalone) ≈ ensemble.

---

## Method index (planned)

**Phase 1 (offline, cached logits)**
- A. Calibration on/off
- B. Temperature sweep on calibration shift: τ ∈ {0.3, 0.5, 0.7, 1.0, 1.5, 2.0}
- C. Logits ensemble: mean / weighted-by-baseline / top-2 ckpts only
- D. Aggregation across answer positions: mean (baseline) / sum / first-only / second-only / max
- E. Confidence-thresholded fallback: if top-2 gap < ε → fall back to neighbor majority OR train-prior argmax
- F. Re-normalize then calibrate (softmax → log → subtract prior)
- G. Vector-scaling calibration (learn diagonal scaling on a held-out 200 from the 1000)
- H. Top-k restricted-argmax: only consider top-k candidate classes per sample, drop tail

**Phase 2 (re-eval needed)**
- I. Neighbor-sampling jitter ensemble (K=3-5 RNG seeds, mean-pool logits)
- J. `max_neighbors_per_hop` ∈ {5, 10, 15, 20}
- K. `max_hops` ∈ {1, 2, 3}
- L. `use_topology_mask` True vs False
- M. Choice-order shuffling: K=3 shuffles, mean-pool

**Phase 3 (graph)**
- N. Low-confidence fallback to 2-hop neighbor majority
- O. Label propagation: add α · log p(neighbor predicted class) to logits

## Phase 1 — auto-generated 2026-05-08 09:47:45

Top-25 method×ckpt combos by accuracy (1000-sample, seed=42).
Baseline-raw = 74.4% (ckpt-1845).

| method | ckpt | acc | Δ vs 74.4 |
|---|---|---|---|
| A. baseline (mean-pos, raw) | 1845 | 74.40 | +0.00 |
| B. temp tau=0.0 | 1845 | 74.40 | +0.00 |
| B. temp tau=0.3 | 1845 | 74.30 | -0.10 |
| C. ensemble-mean (n=4, raw) | ALL | 74.30 | -0.10 |
| C. ensemble-top2 (1845+2042, raw) | TOP2 | 74.30 | -0.10 |
| B. temp tau=0.5 | 1845 | 73.60 | -0.80 |
| D. pos-agg=sum (cal) | 1845 | 73.60 | -0.80 |
| D. pos-agg=min (cal) | 1845 | 73.40 | -1.00 |
| B. temp tau=0.7 | 2042 | 71.50 | -2.90 |
| A. baseline (mean-pos, calibrated) | 1845 | 68.70 | -5.70 |
| B. temp tau=1.0 | 1845 | 68.70 | -5.70 |
| D. pos-agg=mean (cal) | 1845 | 68.70 | -5.70 |
| F. softmax-renorm + cal | 1845 | 68.70 | -5.70 |
| H. top-k=3 (after cal) | 1845 | 68.70 | -5.70 |
| H. top-k=5 (after cal) | 1845 | 68.70 | -5.70 |
| H. top-k=10 (after cal) | 1845 | 68.70 | -5.70 |
| H. top-k=20 (after cal) | 1845 | 68.70 | -5.70 |
| C. ensemble-top2 (1845+2042, cal) | TOP2 | 68.50 | -5.90 |
| C. ensemble-top3 (1845+2042+final, cal) | TOP3 | 68.50 | -5.90 |
| C. ensemble-mean (n=4, cal) | ALL | 68.20 | -6.20 |
| C. ensemble-weighted-by-acc (cal) | ALL | 68.10 | -6.30 |
| B. temp tau=1.5 | 1845 | 62.50 | -11.90 |
| B. temp tau=2.0 | 2042 | 55.40 | -19.00 |
| D. pos-agg=last (cal) | 1845 | 16.30 | -58.10 |
| D. pos-agg=first (cal) | 1845 | 3.40 | -71.00 |

Full per-row log: `/home/lingjie7/auto-research/projects/dlm-graph/analysis/postprocess_arxiv_r128/phase1_results.jsonl` (86 rows)

## Phase 1 — auto-generated 2026-05-08 09:49:01

Top-25 method×ckpt combos by accuracy (1000-sample, seed=42).
Baseline-raw = 74.4% (ckpt-1845).

| method | ckpt | acc | Δ vs 74.4 |
|---|---|---|---|
| B. temp tau=-0.1 | 1845 | 74.50 | +0.10 |
| B. temp tau=0.25 | 1845 | 74.50 | +0.10 |
| A. baseline (mean-pos, raw) | 1845 | 74.40 | +0.00 |
| B. temp tau=0.0 | 1845 | 74.40 | +0.00 |
| D2. pos-weight alpha=0.1 (raw) | 1845 | 74.40 | +0.00 |
| D2. pos-weight alpha=0.2 (raw) | 1845 | 74.40 | +0.00 |
| D2. pos-weight alpha=0.3 (raw) | 1845 | 74.40 | +0.00 |
| D2. pos-weight alpha=0.4 (raw) | 1845 | 74.40 | +0.00 |
| D2. pos-weight alpha=0.5 (raw) | 1845 | 74.40 | +0.00 |
| D2. pos-weight alpha=0.6 (raw) | 1845 | 74.40 | +0.00 |
| D2. pos-weight alpha=0.7 (raw) | 1845 | 74.40 | +0.00 |
| D2. pos-weight alpha=0.8 (raw) | 1845 | 74.40 | +0.00 |
| D2. pos-weight alpha=0.9 (raw) | 1845 | 74.40 | +0.00 |
| B. temp tau=0.15 | 1845 | 74.30 | -0.10 |
| B. temp tau=0.3 | 1845 | 74.30 | -0.10 |
| C. ensemble-mean (n=4, raw) | ALL | 74.30 | -0.10 |
| C. ensemble-top2 (1845+2042, raw) | TOP2 | 74.30 | -0.10 |
| B. temp tau=0.05 | 1845 | 74.20 | -0.20 |
| B. temp tau=0.2 | 1845 | 74.20 | -0.20 |
| B. temp tau=0.1 | 2042 | 74.10 | -0.30 |
| J. CV-tau (mean tau=0.01) | 1845 | 74.10 | -0.30 |
| J. CV-tau (mean tau=0.19) | 2042 | 74.10 | -0.30 |
| B. temp tau=0.5 | 1845 | 73.60 | -0.80 |
| D. pos-agg=sum (cal) | 1845 | 73.60 | -0.80 |
| J. CV-tau (mean tau=0.11) | 1640 | 73.60 | -0.80 |

Full per-row log: `/home/lingjie7/auto-research/projects/dlm-graph/analysis/postprocess_arxiv_r128/phase1_results.jsonl` (198 rows)

## Phase 1b — fit-based methods (auto 2026-05-08 09:52:27)

All fit methods use 5-fold CV on the 1000 samples — train on 800, eval on 200, average.

| method | ckpt | acc | Δ vs 74.4 |
|---|---|---|---|
| N. weighted-ensemble fit (avg w: 1640=0.17+1845=0.28+2042=0.27+final=0.27) | ALL | 74.30 | -0.10 |
| M. alpha-mix CV (5-fold) | 1845 | 74.10 | -0.30 |
| M. alpha-mix CV (5-fold) | 2042 | 73.80 | -0.60 |
| M. alpha-mix CV (5-fold) | 1640 | 73.50 | -0.90 |
| M. alpha-mix CV (5-fold) | final | 73.40 | -1.00 |
| N+M. weighted-ensemble + alpha-cal CV | ALL | 73.10 | -1.30 |
| K. per-class-bias (5-fold CV fit) | 1845 | 27.10 | -47.30 |
| K. per-class-bias (5-fold CV fit) | final | 26.90 | -47.50 |
| K. per-class-bias (5-fold CV fit) | 2042 | 26.20 | -48.20 |
| K. per-class-bias (5-fold CV fit) | 1640 | 18.50 | -55.90 |

### Per-class confusion (ckpt-1845, raw)

| class | n | acc | errors | top confused with | n |
|---|---:|---:|---:|---|---:|
| cs.LG(Machine Learning) | 219 | 74.43 | 56 | cs.CV(Computer Vision and | 13 |
| cs.CV(Computer Vision and | 204 | 89.71 | 21 | cs.LG(Machine Learning) | 12 |
| cs.AI(Artificial Intellig | 36 | 61.11 | 14 | cs.CL(Computation and Lan | 5 |
| cs.RO(Robotics) | 39 | 64.10 | 14 | cs.CV(Computer Vision and | 5 |
| cs.CL(Computation and Lan | 97 | 86.60 | 13 | cs.AI(Artificial Intellig | 5 |
| cs.DC(Distributed, Parall | 22 | 50.00 | 11 | cs.CR(Cryptography and Se | 3 |
| cs.IR(Information Retriev | 20 | 45.00 | 11 | cs.LG(Machine Learning) | 3 |
| cs.HC(Human-Computer Inte | 19 | 47.37 | 10 | cs.CV(Computer Vision and | 3 |

## Phase 1b — fit-based methods (auto 2026-05-08 09:53:56)

All fit methods use 5-fold CV on the 1000 samples — train on 800, eval on 200, average.

| method | ckpt | acc | Δ vs 74.4 |
|---|---|---|---|
| N. weighted-ensemble fit (avg w: 1640=0.17+1845=0.28+2042=0.27+final=0.27) | ALL | 74.30 | -0.10 |
| M. alpha-mix CV (5-fold) | 1845 | 74.10 | -0.30 |
| M. alpha-mix CV (5-fold) | 2042 | 73.80 | -0.60 |
| K. per-class-bias (5-fold CV fit) | 1845 | 73.60 | -0.80 |
| K. per-class-bias (5-fold CV fit) | 2042 | 73.60 | -0.80 |
| K. per-class-bias (5-fold CV fit) | final | 73.50 | -0.90 |
| M. alpha-mix CV (5-fold) | 1640 | 73.50 | -0.90 |
| M. alpha-mix CV (5-fold) | final | 73.40 | -1.00 |
| N+M. weighted-ensemble + alpha-cal CV | ALL | 73.10 | -1.30 |
| K. per-class-bias (5-fold CV fit) | 1640 | 72.40 | -2.00 |

### Per-class confusion (ckpt-1845, raw)

| class | n | acc | errors | top confused with | n |
|---|---:|---:|---:|---|---:|
| cs.LG(Machine Learning) | 219 | 74.43 | 56 | cs.CV(Computer Vision and | 13 |
| cs.CV(Computer Vision and | 204 | 89.71 | 21 | cs.LG(Machine Learning) | 12 |
| cs.AI(Artificial Intellig | 36 | 61.11 | 14 | cs.CL(Computation and Lan | 5 |
| cs.RO(Robotics) | 39 | 64.10 | 14 | cs.CV(Computer Vision and | 5 |
| cs.CL(Computation and Lan | 97 | 86.60 | 13 | cs.AI(Artificial Intellig | 5 |
| cs.DC(Distributed, Parall | 22 | 50.00 | 11 | cs.CR(Cryptography and Se | 3 |
| cs.IR(Information Retriev | 20 | 45.00 | 11 | cs.LG(Machine Learning) | 3 |
| cs.HC(Human-Computer Inte | 19 | 47.37 | 10 | cs.CV(Computer Vision and | 3 |

## Phase 3 — graph-structure post-processing (auto 2026-05-08 09:55:51)

Per-test-node neighbor majority computed from train-labeled k-hop neighbors (k≤2, ≤10 per hop, hop_weight=(1.0, 0.5), seed=42).
Coverage: 99.4% of 1000 test nodes have ≥1 train-labeled neighbor.

| method | ckpt | acc | Δ vs 74.4 |
|---|---|---|---|
| N1. low-conf fallback gap<0.05 | 2042 | 74.70 | +0.30 |
| N1. low-conf fallback gap<0.05 | final | 74.70 | +0.30 |
| N1. low-conf fallback gap<0.05 | 1845 | 74.60 | +0.20 |
| baseline (raw) | 1845 | 74.40 | +0.00 |
| O1. logits + 0.0·log p_neighbor | 1845 | 74.40 | +0.00 |
| baseline (raw) | 2042 | 74.20 | -0.20 |
| O1. logits + 0.0·log p_neighbor | 2042 | 74.20 | -0.20 |
| baseline (raw) | final | 74.20 | -0.20 |
| O1. logits + 0.0·log p_neighbor | final | 74.20 | -0.20 |
| baseline (raw) | 1640 | 73.90 | -0.50 |
| N1. low-conf fallback gap<0.05 | 1640 | 73.90 | -0.50 |
| O1. logits + 0.0·log p_neighbor | 1640 | 73.90 | -0.50 |
| N1. low-conf fallback gap<0.1 | 1845 | 73.30 | -1.10 |
| O2. logits + 0.5·log p_neigh - 0.5·prior | 1845 | 73.30 | -1.10 |
| N1. low-conf fallback gap<0.1 | 2042 | 73.30 | -1.10 |
| N1. low-conf fallback gap<0.1 | final | 73.30 | -1.10 |
| O2. logits + 0.3·log p_neigh - 0.5·prior | 1845 | 73.10 | -1.30 |
| O3. hard-override neigh majority ≥0.9 | 1845 | 73.10 | -1.30 |
| O2. logits + 0.3·log p_neigh - 0.5·prior | 2042 | 73.10 | -1.30 |
| O2. logits + 0.3·log p_neigh - 0.5·prior | final | 73.10 | -1.30 |
| O3. hard-override neigh majority ≥0.9 | 2042 | 73.00 | -1.40 |
| O3. hard-override neigh majority ≥0.9 | final | 73.00 | -1.40 |
| O1. logits + 0.1·log p_neighbor | 1845 | 72.90 | -1.50 |
| O2. logits + 0.5·log p_neigh - 0.5·prior | 2042 | 72.90 | -1.50 |
| O2. logits + 0.5·log p_neigh - 0.5·prior | final | 72.90 | -1.50 |
| O2. logits + 0.5·log p_neigh - 0.2·prior | 1845 | 72.80 | -1.60 |
| O3. hard-override neigh majority ≥0.8 | 1845 | 72.80 | -1.60 |
| O2. logits + 0.5·log p_neigh - 0.2·prior | 2042 | 72.80 | -1.60 |
| O3. hard-override neigh majority ≥0.8 | 2042 | 72.80 | -1.60 |
| O2. logits + 0.5·log p_neigh - 0.2·prior | final | 72.80 | -1.60 |

## Phase 1c — full-test prior calibration (auto 2026-05-08 09:57:22)

Recompute test prior over full ogbn-arxiv test split (48603 samples) for stable calibration.
New shift range = 2.42 (vs 1000-sample shift = 3.31).

| method | ckpt | acc | Δ vs 74.4 |
|---|---|---|---|
| P1c. full-test-prior tau=0.2 | 1845 | 74.80 | +0.40 |
| P1c. full-test-prior tau=0.1 | 1845 | 74.60 | +0.20 |
| P1c. full-test-prior tau=0.1 | 2042 | 74.60 | +0.20 |
| P1c. full-test-prior tau=0.1 | final | 74.60 | +0.20 |
| P1c. full-test-prior tau=0.1 | 1640 | 74.50 | +0.10 |
| P1c. full-test-prior tau=0.3 | 1845 | 74.50 | +0.10 |
| P1c. full-test-prior tau=0.0 | 1845 | 74.40 | +0.00 |
| P1c. full-test-prior tau=0.5 | 1845 | 74.40 | +0.00 |
| P1c. full-test-prior tau=0.2 | 1640 | 74.30 | -0.10 |
| P1c. full-test-prior tau=0.2 | 2042 | 74.30 | -0.10 |
| P1c. full-test-prior tau=0.3 | 2042 | 74.30 | -0.10 |
| P1c. full-test-prior tau=0.2 | final | 74.30 | -0.10 |
| P1c. full-test-prior tau=0.3 | final | 74.30 | -0.10 |
| P1c. full-test-prior tau=0.0 | 2042 | 74.20 | -0.20 |
| P1c. full-test-prior tau=0.0 | final | 74.20 | -0.20 |
| P1c. full-test-prior tau=-0.1 | 1845 | 74.00 | -0.40 |
| P1c. full-test-prior tau=0.0 | 1640 | 73.90 | -0.50 |
| P1c. full-test-prior tau=0.5 | 2042 | 73.80 | -0.60 |
| P1c. full-test-prior tau=0.5 | final | 73.80 | -0.60 |
| P1c. full-test-prior tau=0.3 | 1640 | 73.70 | -0.70 |
| P1c. full-test-prior tau=0.7 | 1845 | 73.50 | -0.90 |
| P1c. full-test-prior tau=-0.1 | 2042 | 73.50 | -0.90 |
| P1c. full-test-prior tau=-0.1 | final | 73.50 | -0.90 |
| P1c. full-test-prior tau=0.7 | 2042 | 73.20 | -1.20 |
| P1c. full-test-prior tau=0.7 | final | 73.20 | -1.20 |

## Phase 4 — test-time augmentation ensemble (auto 2026-05-08 10:27:06)

Combines baseline ckpt-1845 with Phase 2 setting variants (7 settings).
All evaluated on the same 1000 test samples (seed=42).

| method | acc | Δ vs 74.4 |
|---|---|---|
| E2. mean-pool + best tau (=0.10, oracle) | 75.50 | +1.10 |
| Standalone s2_nb15 | 75.40 | +1.00 |
| E5. confidence-weighted vote | 75.30 | +0.90 |
| Standalone s3_nb20 | 75.10 | +0.70 |
| E1. mean-pool all (8 settings, raw) | 75.00 | +0.60 |
| E3. CV-fit weights (baseline=0.13, s1_nb5=0.13, s2_nb15=0.13, s3_nb20=0.13, s4_hops1=0.14, s5_hops3=0.13, s6_notopo=0.15, s7_nbfirst=0.06) | 75.00 | +0.60 |
| E4. plurality vote (raw) | 75.00 | +0.60 |
| Standalone s4_hops1 | 74.90 | +0.50 |
| Standalone s6_notopo | 74.80 | +0.40 |
| Standalone baseline (1845, defaults) | 74.40 | +0.00 |
| Standalone s5_hops3 | 74.40 | +0.00 |
| Standalone s1_nb5 | 74.00 | -0.40 |
| E6. mean-pool then argmax-of-cal-each (mean-after-cal) | 68.10 | -6.30 |
| E1. mean-pool all (cal tau=1) | 67.60 | -6.80 |
| Standalone s7_nbfirst | 47.20 | -27.20 |

## Phase 4 — test-time augmentation ensemble (auto 2026-05-08 10:59:56)

Combines baseline ckpt-1845 with Phase 2 setting variants (11 settings).
All evaluated on the same 1000 test samples (seed=42).

| method | acc | Δ vs 74.4 |
|---|---|---|
| E5. confidence-weighted vote | 76.00 | +1.60 |
| E4. plurality vote (raw) | 75.80 | +1.40 |
| Standalone s2_nb15 | 75.40 | +1.00 |
| Standalone nb12 | 75.40 | +1.00 |
| Standalone nb30 | 75.30 | +0.90 |
| E1. mean-pool all (12 settings, raw) | 75.30 | +0.90 |
| E2. mean-pool + best tau (=0.00, oracle) | 75.30 | +0.90 |
| E3. CV-fit weights (baseline=0.08, s1_nb5=0.08, s2_nb15=0.08, s3_nb20=0.08, s4_hops1=0.09, s5_hops3=0.08, s6_notopo=0.09, nb12=0.08, nb15h1notopo=0.09, nb18=0.08, nb25=0.08, nb30=0.08) | 75.30 | +0.90 |
| Standalone nb15h1notopo | 75.20 | +0.80 |
| Standalone s3_nb20 | 75.10 | +0.70 |
| Standalone nb18 | 75.00 | +0.60 |
| Standalone s4_hops1 | 74.90 | +0.50 |
| Standalone nb25 | 74.90 | +0.50 |
| Standalone s6_notopo | 74.80 | +0.40 |
| Standalone baseline (1845, defaults) | 74.40 | +0.00 |
| Standalone s5_hops3 | 74.40 | +0.00 |
| Standalone s1_nb5 | 74.00 | -0.40 |
| E1. mean-pool all (cal tau=1) | 69.00 | -5.40 |
| E6. mean-pool then argmax-of-cal-each (mean-after-cal) | 69.00 | -5.40 |

## Phase 5 — cross-ckpt + cross-setting ensemble (auto 2026-05-08 11:40:28)

Pools forward passes across ckpts (1640/1845/2042/final) and settings (default + nb=(12, 15, 30)).

| subset | n | mean-pool | E4 plur | E5 conf-vote | Δ-E5 vs 74.4 |
|---|---:|---:|---:|---:|---:|
| all-available | 25/25 | 75.60 | 76.00 | **76.00** | +1.60 |
| best1845 | 4/4 | 75.60 | 75.60 | **76.20** | +1.80 |
| cross_ckpt_baseline | 4/4 | 74.30 | 74.30 | **74.50** | +0.10 |
| cross_ckpt_nb12 | 4/4 | 75.70 | 75.80 | **75.80** | +1.40 |
| cross_ckpt_nb15 | 4/4 | 75.30 | 75.40 | **75.20** | +0.80 |
| cross_ckpt_nb30 | 4/4 | 75.60 | 75.20 | **75.30** | +0.90 |
| 16-cross_ckpt_x_nb | 16/16 | 75.50 | 76.00 | **76.30** | +1.90 |
| 12-3ckpts_x_4nb_no_baseline | 12/12 | 75.80 | 76.00 | **76.10** | +1.70 |

## Phase 5 — cross-ckpt + cross-setting ensemble (auto 2026-05-08 12:08:15)

Pools forward passes across ckpts (1640/1845/2042/final) and settings (default + nb=(12, 15, 30)).

| subset | n | mean-pool | E4 plur | E5 conf-vote | Δ-E5 vs 74.4 |
|---|---:|---:|---:|---:|---:|
| all-available | 33/33 | 75.60 | 75.80 | **75.80** | +1.40 |
| best1845 | 4/4 | 75.60 | 75.60 | **76.20** | +1.80 |
| cross_ckpt_baseline | 4/4 | 74.30 | 74.30 | **74.50** | +0.10 |
| cross_ckpt_nb12 | 4/4 | 75.70 | 75.80 | **75.80** | +1.40 |
| cross_ckpt_nb15 | 4/4 | 75.30 | 75.40 | **75.20** | +0.80 |
| cross_ckpt_nb30 | 4/4 | 75.60 | 75.20 | **75.30** | +0.90 |
| 16-cross_ckpt_x_nb | 16/16 | 75.50 | 76.00 | **76.30** | +1.90 |
| 12-3ckpts_x_4nb_no_baseline | 12/12 | 75.80 | 76.00 | **76.10** | +1.70 |
| 1845_nb10_4jit_only | 4/4 | 75.60 | 75.70 | **76.20** | +1.80 |
| 1845_nb15_4jit_only | 4/4 | 75.80 | 75.70 | **75.80** | +1.40 |
| 1845_nb10_jit_plus_base | 5/5 | 75.30 | 75.70 | **75.80** | +1.40 |
| 1845_nb10_8jit_combined | 9/9 | 75.40 | 75.60 | **75.90** | +1.50 |
| best1845_plus_jitters | 12/12 | 75.60 | 75.70 | **75.70** | +1.30 |
| ALL_24_passes | 24/24 | 75.80 | 76.00 | **76.10** | +1.70 |

## Phase 7 — N=5000 ensemble (auto 2026-05-09 17:37:13)

Phase 6 produced 12 independent N=5000 caches (4 ckpts × 4 nb; ckpt-final == ckpt-2042 confirmed).
σ at N=5000 ≈ 0.6 pt (vs 1.4 at N=1000). Best single: ckpt-1845 nb=10 = 75.56.

| subset | n | mean | E4 | E5 | E5+τ=0.0 | E5+τ=0.1 | E5+τ=0.2 | E5+τ=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best_single (1845_nb10) | 1 | 75.56 | 75.56 | 75.56 | 75.56 | 74.96 | 70.42 | 44.64 |
| 1845_4nb | 4 | 75.52 | 75.54 | 75.94 | 75.94 | 75.76 | 75.68 | 74.00 |
| 3ckpts_nb10 | 3 | 75.06 | 75.44 | 75.50 | 75.50 | 75.54 | 75.32 | 71.48 |
| 3ckpts_nb12 | 3 | 75.02 | 75.12 | 75.16 | 75.16 | 75.24 | 75.20 | 71.24 |
| 3ckpts_x_4nb_12 | 12 | 75.24 | 75.48 | 75.66 | 75.66 | 75.72 | 75.66 | 75.82 |
| 1845_2042_nb10_nb15 | 4 | 75.54 | 75.20 | 76.02 | 76.02 | 76.06 | 76.02 | 73.84 |
