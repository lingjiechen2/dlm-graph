# LLaGA GNN Baselines

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

## Supported Models

- `gcn`
- `sage`
- `gat`
- `gatv2`


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
