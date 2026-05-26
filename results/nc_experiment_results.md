# Node Classification Experiment Results

All scores are accuracy percentages. Unless noted otherwise, runs use
`include_neighbor_labels=False` (`nonb`), `mc_digit`, 2-hop neighborhoods,
10 neighbors per hop, sequential positions, and topology masking for the
cross-dataset results summarized here.

## Main In-Domain Results

| Dataset | Our best | Setting / source | LLaGA best | Best local baseline | Status |
|---|---:|---|---:|---:|---|
| Cora | **90.96** | Cora+PubMed balanced topo, `mc_digit` | 89.22 | 89.67 | Best |
| PubMed | **96.30** | PubMed topo, seq=4k | 95.03 | 95.28 | Best |
| ogbn-arxiv | **76.39** | Arxiv full-train r128, 3ep | 76.66 | 76.63 | Slightly below |

## Recent Replication Runs

| Dataset / run | Best checkpoint | Accuracy | Notes |
|---|---|---:|---|
| PubMed-only topo, 24GPU 10ep | `checkpoint-496` | **95.26** | New replication run; only best checkpoint retained locally. |
| Arxiv topo, 64GPU 9ep | `checkpoint-5760` | **75.18** | `checkpoint-6399` and `checkpoint-final` both reached 75.14. |
| Cora-only topo local run | historical peak | **88.56** | Lower than the main Cora result, which comes from balanced Cora+PubMed training. |

## NC Cross-Dataset Transfer

Only off-diagonal transfer is emphasized. Self-eval values are included for
orientation when they are the relevant local checkpoint result.

| Train/source | Cora target | PubMed target | ogbn-arxiv target |
|---|---:|---:|---:|
| Cora-only topo | self ~88.56 | **90.70** | **47.56** |
| PubMed-only topo `checkpoint-496` | **73.62** | self 95.26 | **48.49** |
| Arxiv topo `checkpoint-5760` | **74.17** | **89.86** | self 75.18 |
| Arxiv topo `checkpoint-6399` | 74.17 | 89.83 | self 75.14 |
| Arxiv topo `checkpoint-final` | 74.17 | 89.83 | self 75.14 |

### Cora-Only to Arxiv Details

The final two Cora-only topo checkpoints were evaluated on the full
ogbn-arxiv test split with 8 GPUs, `digit0_pad`, and `max_answer_tokens=2`.

| Source checkpoint | Target | Accuracy |
|---|---|---:|
| `checkpoint-182` | ogbn-arxiv | **47.56** |
| `checkpoint-208` | ogbn-arxiv | 47.04 |

### Source JSONL Files

- PubMed self-eval replication:
  `/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_pubmed_nc_topo_checkpoint-*167777*.jsonl`
- PubMed -> Cora:
  `/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_pubmed_nc_cross_cora_1677803.jsonl`
- PubMed -> ogbn-arxiv:
  `/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_pubmed_nc_cross_arxiv_retry_1678763.jsonl`
- Arxiv checkpoint cross-evals:
  `/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_arxiv_nc_topo_cross_*.jsonl`
- Cora-only -> ogbn-arxiv:
  `/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_cora_nc_topo_to_arxiv_checkpoint-*.jsonl`

## Takeaways

- In-domain NC is strongest on Cora and PubMed; both beat LLaGA and the local
  GNN/graph-transformer baselines.
- ogbn-arxiv remains the difficult in-domain case: the historical best 76.39
  is close to LLaGA-HO 76.66, while the new 64GPU 9ep replication reached
  75.18.
- Cross-dataset transfer to PubMed is relatively strong: Cora -> PubMed 90.70
  and Arxiv -> PubMed 89.86.
- Cross transfer into ogbn-arxiv from smaller datasets is weak: Cora -> Arxiv
  47.56 and PubMed -> Arxiv 48.49. The main likely causes are the larger
  40-class `digit0_pad` label space and the stronger domain shift.
- Arxiv-trained adapters transfer moderately to Cora/PubMed, but they do not
  match the corresponding in-domain best checkpoints.
