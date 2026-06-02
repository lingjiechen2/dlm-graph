# GNN Node Classification Baselines

All runs use LLaGA-processed text features. Models run via `baselines/gnn_baselines/run_gnn.py`.

## Cora

| Rank | Model | Test Acc | Val Acc | LR | Layers | Best Epoch |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GCN | 89.48% | 89.85% | 1e-4 | 3 | 359 |
| 2 | GAT | 88.93% | 90.04% | 5e-4 | 3 | 40 |
| 3 | MixHop | 88.75% | 89.67% | 5e-3 | 2 | 20 |
| 4 | DifFormer | 88.56% | 88.01% | 5e-4 | 2 | 197 |
| 4 | GATv2 | 88.56% | 89.67% | 5e-4 | 3 | 86 |
| 4 | NodeFormer | 88.56% | 90.59% | 1e-3 | 3 | 296 |
| 7 | GraphTransformer | 88.01% | 90.22% | 5e-3 | 2 | 10 |
| 8 | SAGE | 87.82% | 90.22% | 1e-2 | 2 | 58 |
| 9 | SGFormer | 87.64% | 87.82% | 1e-3 | 4 | 58 |
| 10 | GIN | 86.35% | 88.93% | — | — | — |

## PubMed

| Rank | Model | Test Acc | Val Acc | LR | Layers | Best Epoch |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MixHop | 90.04% | 89.55% | 1e-3 | 3 | 87 |
| 2 | SAGE | 89.68% | 88.28% | 5e-4 | 4 | 39 |
| 3 | GraphTransformer | 89.58% | 88.59% | 1e-2 | 4 | 34 |
| 4 | DifFormer | 89.25% | 88.59% | 1e-2 | 2 | 120 |
| 5 | GCN | 88.82% | 88.97% | 5e-3 | 2 | 213 |
| 6 | GATv2 | 88.36% | 88.33% | 1e-2 | 2 | 110 |
| 7 | GIN | 88.29% | 88.59% | — | — | — |
| 8 | GAT | 87.91% | 88.00% | 1e-3 | 2 | 217 |
| 9 | SGFormer | 87.73% | 87.75% | 5e-3 | 2 | 561 |
| 10 | NodeFormer | 86.46% | 87.14% | 1e-3 | 3 | 976 |

## Best Per Dataset

| Dataset | Best Model | Test Acc |
|---|---|---:|
| Cora | GCN | 89.48% |
| PubMed | MixHop | 90.04% |
