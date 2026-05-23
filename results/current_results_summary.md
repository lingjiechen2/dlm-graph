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

## Related Files

- Consolidated baseline comparison: [all_results_table.md](all_results_table.md)
- Full current ledger: [current_results_detailed.md](current_results_detailed.md)
- Baseline result index: [README.md](README.md#baseline-results)
