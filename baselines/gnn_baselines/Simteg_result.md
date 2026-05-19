All runs use `feature_type=simteg`. Cora and PubMed have complete results. OGBN-Arxiv has two OOM failures recorded below.

## Successful Runs

| Dataset | Model | LR | Layers | Best Val Acc | Final Test Acc | Best Epoch | Epochs Ran |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | GCN | 0.0001 | 3 | 90.41% | 89.48% | 566 | 666 |
| Cora | SAGE | 0.01 | 2 | 92.25% | 88.93% | 106 | 206 |
| Cora | GAT | 0.0005 | 3 | 90.77% | 86.90% | 56 | 156 |
| Cora | GATv2 | 0.0005 | 3 | 90.77% | 89.11% | 101 | 201 |
| Cora | GraphTransformer | 0.005 | 2 | 91.33% | 87.64% | 183 | 283 |
| Cora | MixHop | 0.005 | 2 | 92.25% | 88.01% | 96 | 196 |
| Cora | DifFormer | 0.0005 | 2 | 90.96% | 88.56% | 327 | 427 |
| Cora | SGFormer | 0.001 | 4 | 89.30% | 87.64% | 123 | 223 |
| Cora | NodeFormer | 0.001 | 3 | 91.51% | 89.67% | 388 | 488 |
| PubMed | GCN | 0.005 | 2 | 93.76% | 93.56% | 68 | 168 |
| PubMed | SAGE | 0.0005 | 4 | 94.83% | 94.93% | 78 | 178 |
| PubMed | GAT | 0.001 | 2 | 92.47% | 92.06% | 146 | 246 |
| PubMed | GATv2 | 0.01 | 2 | 94.83% | 94.68% | 49 | 149 |
| PubMed | GraphTransformer | 0.01 | 4 | 94.98% | 95.03% | 54 | 154 |
| PubMed | MixHop | 0.001 | 3 | 94.93% | 95.23% | 22 | 122 |
| PubMed | DifFormer | 0.01 | 2 | 94.88% | 95.03% | 77 | 177 |
| PubMed | SGFormer | 0.005 | 2 | 95.03% | 95.28% | 83 | 183 |
| PubMed | NodeFormer | 0.001 | 3 | 94.95% | 95.16% | 191 | 291 |
| OGBN-Arxiv | GCN | 0.001 | 3 | 75.57% | 74.25% | 122 | 222 |
| OGBN-Arxiv | SAGE | 0.005 | 3 | 77.15% | 75.76% | 24 | 124 |
| OGBN-Arxiv | GAT | 0.0005 | 3 | 75.17% | 73.80% | 131 | 231 |
| OGBN-Arxiv | GATv2 | 0.0005 | 2 | 77.27% | 76.02% | 123 | 223 |
| OGBN-Arxiv | MixHop | 0.001 | 3 | 77.38% | 76.01% | 25 | 125 |
| OGBN-Arxiv | DifFormer | 0.0005 | 3 | 77.81% | 76.29% | 1615 | 1715 |
| OGBN-Arxiv | SGFormer | 0.001 | 3 | 77.27% | 76.08% | 708 | 808 |

## Best Final Test By Dataset

| Dataset | Best Model | LR | Layers | Best Val Acc | Final Test Acc |
|---|---|---:|---:|---:|---:|
| Cora | NodeFormer | 0.001 | 3 | 91.51% | 89.67% |
| PubMed | SGFormer | 0.005 | 2 | 95.03% | 95.28% |
| OGBN-Arxiv | DifFormer | 0.0005 | 3 | 77.81% | 76.29% |
