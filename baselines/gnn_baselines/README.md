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
我们的改进比较偏transformer设计，所以增加了一下graph transformer的代表性架构的对比（best in pubmed ~90%）。
- `gcn`
- `sage`
- `gat`
- `gatv2`
- `graphtransformer`
- `nodeformer`
- `difformer`
- `sgformer`

TODO: 
- Full baseline result table.
- GNN&GT+LLM embedding.

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
