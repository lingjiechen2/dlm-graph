# DLM-Graph Results

The result documentation has moved to [results/](results/).

- Current summary: [results/current_results_summary.md](results/current_results_summary.md)
- All baselines and current results: [results/all_results_table.md](results/all_results_table.md)
- Current detailed ledger: [results/current_results_detailed.md](results/current_results_detailed.md)
- Baseline result tables: [results/baselines/](results/baselines/)

The old root-level `results.md` content was an early ledger and is superseded by the files above.

<!-- frozen-llada-lp-start -->
## Frozen / Selected LLaDA LP Reference (LLaGA Splits)

Selected LP reference values on the official LLaGA LP test splits. Cora and
PubMed use no-topo SFT `checkpoint-final`; ogbn-arxiv uses no-topo
`checkpoint-712`. The original zero-shot frozen LLaDA baseline is preserved in
[results/frozen_llada_lp_results.md](results/frozen_llada_lp_results.md).

| Dataset | Samples | Accuracy | AUC | Reference checkpoint |
|---|---:|---:|---:|---|
| Cora | 680 | 85.88 | 0.9574 | no-topo `checkpoint-final` |
| PubMed | 5368 | 90.89 | 0.9889 | no-topo `checkpoint-final` |
| ogbn-arxiv | 80086 | 95.25 | 0.9925 | no-topo `checkpoint-712` |

JSONL sources:

- Frozen zero-shot baseline: `/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_frozen_llada_lp_llaga.jsonl`
- Selected no-topo SFT references: `/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_{cora,pubmed,arxiv}_lp_llaga_notopo_allckpts.jsonl`

<!-- frozen-llada-lp-end -->
