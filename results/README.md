# DLM-Graph Results

This directory is the central index for result documentation.

## Current Method Results

- [All baselines and current results](all_results_table.md): consolidated comparison table across NC and LP.
- [Paper main-results table](main_results_table.tex): LaTeX table for the essay.
- [Current results summary](current_results_summary.md): compact scorecard and takeaways.
- [Current results detailed](current_results_detailed.md): full experiment ledger for the current TM-DLM NC and LP runs.
- [Neighbor sweep results](neighbor_sweep_results.md): star-topology `max_neighbors_per_hop` sweep across Cora, PubMed, and ogbn-arxiv NC/LP.
- [Node classification experiment results](nc_experiment_results.md): NC in-domain, replication, and cross-dataset transfer summary.
- [Frozen LLaDA LP results](frozen_llada_lp_results.md): zero-shot LP baseline on the LLaGA Cora/PubMed/ogbn-arxiv test splits.
- [LP cross-dataset transfer results](lp_cross_dataset_topo_results.md): topology-masked LP final-checkpoint transfer matrix across Cora, PubMed, and ogbn-arxiv.

## Baseline Results

- [GNN node classification baselines](baselines/gnn_node_classification.md): Cora and PubMed GNN / graph-transformer NC runs.
- [SimTeG node classification baselines](baselines/simteg_node_classification.md): NC runs using `feature_type=simteg` on Cora, PubMed, and ogbn-arxiv.
- [GNN link prediction baselines](baselines/gnn_link_prediction.md): LP baseline tables with AUC, AP, and accuracy.

The runnable baseline code remains in `baselines/gnn_baselines/`; this directory stores the result tables and current experiment ledger.
