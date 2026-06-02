# All Baselines and Current Results

This table consolidates the strongest tracked baselines and the current TM-DLM results. Scores are accuracy percentages. "Best listed baseline" is the strongest non-TM-DLM result currently recorded in this repository for that task and dataset.

## Consolidated Scorecard

| Task | Dataset | Our best | LLaGA best | Best local GNN/GT baseline | Best listed baseline | Delta vs best listed baseline | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| NC | Cora | 90.96 | 89.22 | 89.67 | 89.67 | +1.29 | Best |
| NC | PubMed | 96.30 | 95.03 | 95.28 | 95.28 | +1.02 | Best |
| NC | ogbn-arxiv | 76.39 | 76.66 | 76.63 | 76.66 | -0.27 | Slightly below |
| LP | Cora | 91.62 | 86.82 | 87.79 | 87.79 | +3.83 | Best |
| LP | PubMed | 95.31 | 91.41 | 89.74 | 91.41 | +3.90 | Best |
| LP | ogbn-arxiv | 96.55 | 94.15 | 94.67 | 94.67 | +1.88 | Best |

Summary: TM-DLM is best among the tracked methods on 5 of 6 task-dataset pairs. The only exception is ogbn-arxiv node classification, where TM-DLM trails LLaGA-HO by 0.27 points and the local SimTeG NodeFormer baseline by 0.24 points.

## Node Classification Details

| Dataset | Our best | Our setting | LLaGA-ND | LLaGA-HO | Best local GNN/GT | Local model | Delta vs LLaGA best | Delta vs local best |
|---|---:|---|---:|---:|---:|---|---:|---:|
| Cora | 90.96 | §1 infill | 88.86 | 89.22 | 89.67 | SimTeG NodeFormer | +1.74 | +1.29 |
| PubMed | 96.30 | §13 logit, seq=4k | 95.03 | 95.03 | 95.28 | SimTeG SGFormer | +1.27 | +1.02 |
| ogbn-arxiv | 76.39 | §23 full-train 3ep | 75.98 | 76.66 | 76.63 | SimTeG NodeFormer | -0.27 | -0.24 |

## Link Prediction Details

| Dataset | Frozen / selected reference | Our best | Our setting | LLaGA-ND | LLaGA-HO | Best local GNN/GT ACC | Local model | Delta vs LLaGA best | Delta vs local best |
|---|---:|---:|---|---:|---:|---:|---|---:|---:|
| Cora | 85.88* | 91.62 | §28, LLaGA split | 83.79 | 86.82 | 87.79 | GCN | +4.80 | +3.83 |
| PubMed | 90.89* | 95.31 | ckpt-744, LLaGA split | 91.41 | 89.18 | 89.74 | SGFormer | +3.90 | +5.57 |
| ogbn-arxiv | 95.25* | 96.55 | ckpt-2492, LLaGA split | 91.24 | 94.15 | 94.67 | GCN | +2.40 | +1.88 |

Frozen LLaDA details: [baselines/frozen_llada_lp.md](baselines/frozen_llada_lp.md).
`*` Selected LP references use no-topo SFT evals: Cora `checkpoint-final` 85.88 / AUC 0.9574; PubMed `checkpoint-final` 90.89 / AUC 0.9889; ogbn-arxiv `checkpoint-712` 95.25 / AUC 0.9925.

## Source Files

- Current method results: [ours/summary.md](ours/summary.md), [ours/detailed.md](ours/detailed.md)
- Neighbor sweep results: [ours/neighbor_sweep.md](ours/neighbor_sweep.md)
- LP cross-dataset transfer: [ours/lp_cross_dataset.md](ours/lp_cross_dataset.md)
- Frozen LLaDA LP baseline: [baselines/frozen_llada_lp.md](baselines/frozen_llada_lp.md)
- Local NC baselines: [baselines/simteg_nc.md](baselines/simteg_nc.md), [baselines/gnn_nc.md](baselines/gnn_nc.md)
- Local LP baselines: [baselines/gnn_lp.md](baselines/gnn_lp.md)
- Baseline category index: [baselines/README.md](baselines/README.md)
