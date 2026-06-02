# Baseline Results

Baselines are grouped by method category. All accuracy values are percentages.

## Category A — Pure GNN

Graph neural networks applied directly to node/edge features (no LLM backbone). Text features come from LLaGA preprocessing.

- **[gnn_nc.md](gnn_nc.md)** — GNN node classification on Cora and PubMed. Best: GCN 89.48% (Cora), MixHop 90.04% (PubMed).
- **[gnn_lp.md](gnn_lp.md)** — GNN link prediction on Cora, PubMed, and ogbn-arxiv. Best test ACC: GCN 87.79% (Cora), SGFormer 89.74% (PubMed), GCN 94.67% (ogbn-arxiv).

## Category B — GNN + SimTeG Embeddings

GNN models using SimTeG-enriched text embeddings as node features instead of raw LLaGA features. Evaluated on node classification only.

- **[simteg_nc.md](simteg_nc.md)** — SimTeG+GNN NC on Cora, PubMed, ogbn-arxiv. Best: NodeFormer 89.67% (Cora), SGFormer 95.28% (PubMed), NodeFormer 76.63% (ogbn-arxiv).

## Category C — Frozen LLM (Zero-Shot)

The LLaDA-8B-Instruct backbone evaluated without any task-specific SFT. Establishes how much of the gain comes from SFT vs. the model's priors.

- **[frozen_llada_lp.md](frozen_llada_lp.md)** — Zero-shot LP on Cora, PubMed, ogbn-arxiv. All three datasets are near chance (~50%), confirming SFT is necessary.

## Category D — LLaGA (SFT LLM Baseline)

Results quoted from the LLaGA paper (Table 1). LLaGA-ND uses node-only descriptions; LLaGA-HO uses higher-order neighborhood descriptions.

Numbers are reproduced in the consolidated comparison table:
- [../all_results_table.md](../all_results_table.md)

## Key Numbers at a Glance

| Task | Dataset | Best GNN | Best SimTeG+GNN | LLaGA-HO | TM-DLM (ours) |
|---|---|---:|---:|---:|---:|
| NC | Cora | 89.48 | 89.67 | 89.22 | **90.96** |
| NC | PubMed | 90.04 | 95.28 | 95.03 | **96.30** |
| NC | ogbn-arxiv | — | 76.63 | 76.66 | 76.39 |
| LP | Cora | 87.79 | — | 86.82 | **91.62** |
| LP | PubMed | 89.74 | — | 91.41 | **95.31** |
| LP | ogbn-arxiv | 94.67 | — | 94.15 | **96.55** |
