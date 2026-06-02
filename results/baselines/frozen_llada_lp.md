# Frozen LLaDA-8B-Instruct LP Results

Zero-shot LP baseline using the untrained/non-SFT `GSAI-ML/LLaDA-8B-Instruct`
model on the official LLaGA LP test splits. Evaluation uses the same LP
head-to-head prompt setup as the SFT runs: `max_seq_len=4096`, 2-hop
neighborhoods, 10 neighbors per hop, sequential positions, and topology mask
enabled.

| Dataset | Samples | Accuracy | AUC | Per-label acc (no / yes) |
|---|---:|---:|---:|---|
| Cora | 680 | 52.65 | 0.5209 | 69.66 / 31.23 |
| PubMed | 5,368 | 51.12 | 0.4789 | 92.52 / 9.19 |
| ogbn-arxiv | 80,086 | 46.99 | 0.4875 | 25.69 / 71.65 |

JSONL: `/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_frozen_llada_lp_llaga.jsonl`

Takeaway: frozen LLaDA-Instruct is effectively chance on LP across all three
datasets, with dataset-dependent label bias. Task-specific SFT is responsible
for the large LP gains.
