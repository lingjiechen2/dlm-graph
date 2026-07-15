# GNN Baselines

Included source files:
- `run_gnn.py`
- `model.py`
- `__init__.py`
- `requirements.txt`

Not copied:
- `__pycache__/`
- `data/`
- `result/`
- any other generated or cached files

# Results
Cora dataset: 
| Rank | Model | Best test acc | Corresponding val acc | lr | layers | best epoch |
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

* 10: GIN | 0.8635 | 0.8893 | 

Pubmed dataset:
| Rank | Model | Best test acc | Corresponding val acc | lr | layers | best epoch |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MixHop | 90.04% | 89.55% | 1e-3 | 3 | 87 |
| 2 | SAGE | 89.68% | 88.28% | 5e-4 | 4 | 39 |
| 3 | GraphTransformer | 89.58% | 88.59% | 1e-2 | 4 | 34 |
| 4 | DifFormer | 89.25% | 88.59% | 1e-2 | 2 | 120 |
| 5 | GCN | 88.82% | 88.97% | 5e-3 | 2 | 213 |
| 6 | GATv2 | 88.36% | 88.33% | 1e-2 | 2 | 110 |
| 8 | GAT | 87.91% | 88.00% | 1e-3 | 2 | 217 |
| 9 | SGFormer | 87.73% | 87.75% | 5e-3 | 2 | 561 |
| 10 | NodeFormer | 86.46% | 87.14% | 1e-3 | 3 | 976 |

* 7: GIN | 0.8829 | 0.8859 | 


## Hetero Graph Performance
(test accuracy %)
node classification
| model | Cornell | Texas | Washington | Wisconsin |
|---|---:|---:|---:|---:|
| gcn | 58.97% | 65.79% | 68.09% | 56.60% |
| sage | 69.23% | 81.58% | 85.11% | 90.57% |
| gin | 48.72% | 60.53% | 61.70% | 58.49% |
| gat | 51.28% | 57.89% | 63.83% | 56.60% |
| gatv2 | 43.59% | 60.53% | 59.57% | 50.94% |
| graphtransformer | 66.67% | 78.95% | 85.11% | 71.70% |
| mixhop | 76.92% | 78.95% | 78.72% | 84.91% |
| difformer | 74.36% | 78.95% | 85.11% | 92.45% |
| sgformer | 53.85% | 65.79% | 57.45% | 47.17% |
| nodeformer | 79.49% | 76.32% | 80.85% | 79.25% |

link prediction
| model | Cornell | Texas | Washington | Wisconsin |
|---|---:|---:|---:|---:|
| gcn | 75.93% | 61.61% | 70.55% | 75.82% |
| sage | 62.96% | 71.43% | 71.23% | 70.33% |
| gin | 50.00% | 58.04% | 54.11% | 59.89% |
| gat | 50.00% | 50.00% | 50.00% | 50.00% |
| gatv2 | 50.00% | 50.00% | 50.00% | 50.00% |
| graphtransformer | 57.41% | 59.82% | 68.49% | 69.78% |
| mixhop | 77.78% | 75.89% | 74.66% | 73.63% |
| difformer | 70.37% | 76.79% | 72.60% | 69.78% |
| sgformer | 65.74% | 82.14% | 77.40% | 70.33% |
| nodeformer | 68.52% | 73.21% | 72.60% | 50.00% |


## Supported Models
Because our improvements lean toward transformer-style designs, this adds comparisons with representative graph transformer architectures (best on PubMed is around 90%).
- `gcn`
- `sage`
- `gat`
- `gatv2`
- `graphtransformer`
- `nodeformer`
- `difformer`
- `sgformer`



## Example Commands

Run with LLaGA processed data:

```bash
python run_gnn.py \
  --dataset cora \
  --model gcn \
  --llaga_dataset_root "xxx" \
  --result_dir "xxx"
```

Run GATv2 on a specific GPU:

```bash
cd /home/bei4/exper/dlm-graph/baselines/llaga_gnn

python run_gnn.py \
  --dataset pubmed \
  --model gatv2 \
  --gpu 0 \
  --llaga_dataset_root "xxx" \
  --result_dir "xxx"
```

Run with official Planetoid data:

```bash
cd /home/bei4/exper/dlm-graph/baselines/llaga_gnn

python run_gnn.py \
  --source planetoid \
  --dataset cora \
  --model sage \
  --data_root "xxx" \
  --result_dir "xxx"
```
