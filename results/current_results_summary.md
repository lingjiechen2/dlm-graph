# Current Results Summary

All current method results use `include_neighbor_labels=False` (`nonb`), so prompts include neighbor text but no oracle neighbor class labels. Node-classification runs mostly use `mc_digit + digit0`; link-prediction runs use the same digit format with `task=lp`.

## Headline

Against the LLaGA paper Table 1 target, defined as the better of LLaGA-ND and LLaGA-HO per dataset, the current method is SOTA on 5 of 6 tracked tasks.

| Task | Our best | LLaGA target | Delta | Status |
|---|---:|---:|---:|---|
| Cora NC | 90.96 | 89.22 | +1.74 | SOTA |
| PubMed NC | 96.30 | 95.03 | +1.27 | SOTA |
| ogbn-arxiv NC | 76.39 | 76.66 | -0.27 | Slightly below, within expected noise |
| Cora LP | 91.62 | 86.82 | +4.80 | SOTA |
| PubMed LP | 95.31 | 91.41 | +3.90 | SOTA |
| ogbn-arxiv LP | 96.55 | 94.15 | +2.40 | SOTA |

## Node Classification

| Dataset | Best current result | Main comparison |
|---|---:|---|
| Cora | 90.96 | Beats LLaGA-HO 89.22 by +1.74 points. |
| PubMed | 96.30 | Beats LLaGA-ND/HO 95.03 by +1.27 points. |
| ogbn-arxiv | 76.39 | Trails LLaGA-HO 76.66 by -0.27 points. |

Main takeaways:

- `mc_digit` is the robust prompt format for supervised NC; older `category_infill` runs had collapse issues in merged training.
- Balanced Cora+PubMed joint training fixes the earlier Cora collapse and matches or exceeds the single-dataset Cora/PubMed baselines.
- Increasing PubMed context to `max_seq_len=4096` lifts the peak to 96.30 and is the clearest case where topology masking beats dense attention.
- ogbn-arxiv NC is the only task not above LLaGA-HO; full-train 3-epoch r=128 reaches 76.39, only 0.27 points short.

Full NC in-domain, replication, and cross-dataset transfer details:
[nc_experiment_results.md](nc_experiment_results.md).

## Link Prediction

| Dataset | Best current result | Main comparison |
|---|---:|---|
| Cora | 91.62 | Beats LLaGA-HO 86.82 by +4.80 points on the official LLaGA split. |
| PubMed | 95.31 | Beats LLaGA-ND 91.41 by +3.90 points. |
| ogbn-arxiv | 96.55 | Beats LLaGA-HO 94.15 by +2.40 points. |

Main takeaways:

- LP is strong across all three datasets.
- The definitive Cora LP result is the LLaGA-split run, not the earlier seed-42 split run, because the latter had split overlap with LLaGA test positives.
- Full-data arxiv LP training is needed to clear LLaGA-HO; the 10 percent data version already beats LLaGA-ND but does not clear HO.

## Frozen / Selected LP Reference

| Dataset | Reference accuracy | Current SFT best | Lift |
|---|---:|---:|---:|
| Cora | 85.88* | 91.62 | +5.74 |
| PubMed | 90.89* | 95.31 | +4.42 |
| ogbn-arxiv | 95.25* | 96.55 | +1.30 |

`*` Selected LP references use no-topo SFT evals: Cora `checkpoint-final` 85.88 / AUC 0.9574; PubMed `checkpoint-final` 90.89 / AUC 0.9889; ogbn-arxiv `checkpoint-712` 95.25 / AUC 0.9925. Full frozen zero-shot details: [frozen_llada_lp_results.md](frozen_llada_lp_results.md).

## LP Cross-Dataset Transfer

Topo final-checkpoint LP adapters transfer strongly across datasets, with all
off-diagonal accuracies above 88%.

| Train/source | Cora target | PubMed target | ogbn-arxiv target |
|---|---:|---:|---:|
| Cora topo final | -- | 92.34 | 93.54 |
| PubMed topo final | 88.82 | -- | 94.51 |
| ogbn-arxiv topo final | 90.44 | 94.08 | -- |

Full accuracy/AUC details: [lp_cross_dataset_topo_results.md](lp_cross_dataset_topo_results.md).

## Neighbor Sweep

The star-topology neighbor-count sweep over `nb={0,1,3,5,10,20}` is complete
for Cora, PubMed, and ogbn-arxiv NC/LP. The previous post-training eval default
used `max_neighbors_per_hop=10`, which remains best for 5 of 6 rows; Cora NC
peaks at `nb=3`.

Full sweep table: [neighbor_sweep_results.md](neighbor_sweep_results.md).

## Related Files

- Consolidated baseline comparison: [all_results_table.md](all_results_table.md)
- Frozen LLaDA LP baseline: [frozen_llada_lp_results.md](frozen_llada_lp_results.md)
- LP cross-dataset transfer: [lp_cross_dataset_topo_results.md](lp_cross_dataset_topo_results.md)
- Neighbor sweep results: [neighbor_sweep_results.md](neighbor_sweep_results.md)
- Full current ledger: [current_results_detailed.md](current_results_detailed.md)
- Baseline result index: [README.md](README.md#baseline-results)
