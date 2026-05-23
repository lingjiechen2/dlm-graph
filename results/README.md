# DLM-Graph Results

This directory is the central index for result documentation.

## Current Method Results

- [All baselines and current results](all_results_table.md): consolidated comparison table across NC and LP.
- [Current results summary](current_results_summary.md): compact scorecard and takeaways.
- [Current results detailed](current_results_detailed.md): full experiment ledger for the current TM-DLM NC and LP runs.

## Baseline Results

- [GNN node classification baselines](baselines/gnn_node_classification.md): Cora and PubMed GNN / graph-transformer NC runs.
- [SimTeG node classification baselines](baselines/simteg_node_classification.md): NC runs using `feature_type=simteg` on Cora, PubMed, and ogbn-arxiv.
- [GNN link prediction baselines](baselines/gnn_link_prediction.md): LP baseline tables with AUC, AP, and accuracy.

The runnable baseline code remains in `baselines/gnn_baselines/`; this directory stores the result tables and current experiment ledger.
