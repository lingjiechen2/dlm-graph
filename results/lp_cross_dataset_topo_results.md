# LP Cross-Dataset Transfer Results

Topo cross-dataset LP evaluation uses the final topology-masked LP SFT adapter
from one source dataset and evaluates on another dataset's official LLaGA LP
test split. All runs use 2-hop neighborhoods, 10 neighbors per hop,
`max_seq_len=4096`, `seed=42`, and topology masking enabled.

## Final-Checkpoint Topo Transfer Matrix

Rows are training/source datasets; columns are evaluation/target datasets.
Diagonal cells are omitted because they are in-domain LP results covered in
[current_results_summary.md](current_results_summary.md).

| Source checkpoint | Cora target | PubMed target | ogbn-arxiv target |
|---|---:|---:|---:|
| Cora topo final | -- | 92.34 / 0.9759 | 93.54 / 0.9832 |
| PubMed topo final | 88.82 / 0.9580 | -- | 94.51 / 0.9886 |
| ogbn-arxiv topo final | 90.44 / 0.9627 | 94.08 / 0.9848 | -- |

Values are `accuracy / AUC`.

## Run Details

| Source | Target | Job | Samples | Accuracy | AUC | Per-label acc (no / yes) | World size |
|---|---|---:|---:|---:|---:|---|---:|
| Cora | PubMed | 1677193 | 5,368 | 92.34 | 0.9759 | 91.82 / 92.88 | 1 |
| Cora | ogbn-arxiv | 1677194 | 80,086 | 93.54 | 0.9832 | 95.69 / 91.04 | 8 |
| PubMed | Cora | 1677160 | 680 | 88.82 | 0.9580 | 96.04 / 79.73 | 1 |
| PubMed | ogbn-arxiv | 1677195 | 80,086 | 94.51 | 0.9886 | 97.61 / 90.92 | 8 |
| ogbn-arxiv | Cora | 1677162 | 680 | 90.44 | 0.9627 | 89.71 / 91.36 | 1 |
| ogbn-arxiv | PubMed | 1677196 | 5,368 | 94.08 | 0.9848 | 90.82 / 97.38 | 1 |

JSONL source:
`/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_lp_cross_dataset_topo_final.jsonl`

## Source Checkpoints

- Cora topo final: `/mnt/weka/home/lingjie.chen/model/dlm-graph/tmdlm-llada-8b-cora-lp-2hop-r64-ep5-cora_lp_llaga_20260521_1751_8gpu_5ep/checkpoint-final`
- PubMed topo final: `/mnt/weka/home/lingjie.chen/model/dlm-graph/tmdlm-llada-8b-pubmed-lp-2hop-r64-ep5-pubmed_lp_llaga_20260521_1917_8gpu_5ep/checkpoint-final`
- ogbn-arxiv topo final: `/mnt/weka/home/lingjie.chen/model/dlm-graph/tmdlm-llada-8b-arxiv-lp-2hop-r64-ep5-arxiv_lp_llaga_20260521_2259_64gpu_5ep/checkpoint-final`
