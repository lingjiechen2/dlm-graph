# Neighbor Sweep Results

Star-topology neighbor-count sweep for the current TM-DLM evals. Scores are
accuracy percentages. All rows use `max_hops=2`, `topology_mask_type=star`, and
`include_neighbor_labels=False`.

The standard post-training eval setting used in previous result tables is
`max_neighbors_per_hop=10`.

## Accuracy Summary

| Dataset | Task | nb=0 | nb=1 | nb=3 | nb=5 | nb=10 | nb=20 | Best |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Cora | LP | 85.15 | 85.74 | 88.24 | 89.12 | 90.29 | 87.65 | nb=10 |
| Cora | NC | 80.44 | 85.42 | 87.64 | 87.45 | 87.45 | 87.27 | nb=3 |
| PubMed | LP | 89.79 | 88.79 | 83.81 | 85.82 | 95.03 | 89.75 | nb=10 |
| PubMed | NC | 94.27 | 94.98 | 95.11 | 95.06 | 95.26 | 95.13 | nb=10 |
| ogbn-arxiv | LP | 92.30 | 92.63 | 91.31 | 93.83 | 96.44 | 90.76 | nb=10 |
| ogbn-arxiv | NC | 71.35 | 73.33 | 74.08 | 74.36 | 75.14 | 75.01 | nb=10 |

## Notes

- The strongest setting in this sweep is `nb=10` for 5 of 6 rows.
- Cora NC is the exception: `nb=3` is best at 87.64, narrowly above `nb=5/10`
  at 87.45.
- ogbn-arxiv NC initially missed `nb=3/5/10/20` because the first multi-value
  eval failed during distributed class-prior broadcast. The missing values were
  completed as separate 8-GPU eval jobs.
- The final ogbn-arxiv NC `nb=20` eval used a fast raw-label class-prior count
  path to avoid constructing full training prompts just to count labels.

## Source Logs

- LP sweep JSONL: `/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_lp_star_neighbor_sweep_20260526.jsonl`
- NC sweep JSONL: `/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_nc_star_neighbor_sweep_20260526.jsonl`
- Final ogbn-arxiv NC `nb=20` job: `arxiv_nc_star_nb20_1684846`
