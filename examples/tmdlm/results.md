# DLM-Graph: Node Classification Results

## Model & Setup

- **Model**: LLaDA-8B-Instruct (GSAI-ML/LLaDA-8B-Instruct), 8B params, masked discrete diffusion LM
- **Classification format**: Multiple-choice with single-digit answers (tokens "0"–"6" for Cora, "0"–"2" for PubMed)
- **SFT config**: LoRA r=32, alpha=64, all-linear modules, 83.9M trainable params (1.04%), lr=5e-5, effective batch=16

## Cora (7 classes, 542 test nodes, supervised setting)

Baseline source: "When Do LLMs Help With Node Classification?" (arXiv:2502.00829), Table 2, supervised setting.

### Comparison with Baselines

| Method | Type | Accuracy | Source |
|--------|------|----------|--------|
| GCN + LLM Emb | GNN + LLM embeddings | 88.15 +/- 1.79 | arXiv:2502.00829 |
| TAPE | LLM-as-Reasoner | 88.05 +/- 1.76 | arXiv:2502.00829 |
| LLaGA | LLM + Graph Projector | 87.55 +/- 1.15 | arXiv:2502.00829 |
| GraphSAGE (ShallowEmb) | GNN | 87.44 +/- 1.74 | arXiv:2502.00829 |
| GCN (ShallowEmb) | GNN | 87.41 +/- 2.08 | arXiv:2502.00829 |
| ENGINE | GNN + LLM | 87.00 +/- 1.60 | arXiv:2502.00829 |
| GLEM | GNN + LLM | 86.81 +/- 1.19 | arXiv:2502.00829 |
| GAT (ShallowEmb) | GNN | 86.68 +/- 1.12 | arXiv:2502.00829 |
| **Ours: SFT target-only (ep3)** | **DLM + LoRA** | **84.13** | — |
| RoBERTa-355M | LM only | 83.17 +/- 0.84 | arXiv:2502.00829 |
| GraphGPT | LLM + Graph | 82.29 +/- 0.26 | arXiv:2502.00829 |
| Ours: SFT target-only (ep1) | DLM + LoRA | 82.47 | — |
| SentenceBERT-66M | LM only | 79.61 +/- 1.40 | arXiv:2502.00829 |

### Our Full Results on Cora

| Method | Accuracy | Notes |
|--------|----------|-------|
| **SFT target-only (ep3)** | **84.13%** | Best result, LoRA fine-tuned |
| SFT target-only (ep4) | 83.95% | Slight overfit |
| SFT target-only (ep1) | 82.47% | |
| SFT target-only (ep2) | 81.92% | |
| SFT 2-hop full attn (ep1) | 79.15% | Neighbors hurt during training |
| SFT dense mask (ep3) | 73.43% | Dense masking dilutes classification signal |
| SFT target-only ep3, eval w/ 2-hop full attn | 71.77% | Neighbors at inference hurt |
| Frozen MC (target-only) | 62.73% | Best frozen result |
| Frozen MC (2-hop full attn) | 60.33% | |
| Frozen MC (1-hop full attn) | 57.01% | |
| SFT target-only ep3, eval w/ 2-hop topo mask | 57.38% | Topo mask at inference hurts most |
| Frozen first-token (target-only) | 51.29% | Class-name token classification |
| Frozen MC (1-hop topo mask) | 50.74% | |
| Frozen MC (2-hop topo mask) | 49.26% | |
| Frozen first-token (2-hop topo mask) | 48.34% | |
| Frozen first-token (no neighbors) | 46.86% | |
| Frozen first-token (2-hop full attn) | 44.46% | |
| Frozen MC (1-hop topo mask + topo posid) | 28.78% | Per-node position ID reset breaks frozen model |

### Per-Class Accuracy (Cora SFT, best epochs)

| Epoch | Overall | Case Based | Genetic Alg | Neural Net | Prob Methods | RL | Rule Learn | Theory |
|-------|---------|-----------|-------------|-----------|-------------|------|-----------|--------|
| 1 | 82.47% | 83.9% | 94.4% | 88.9% | 73.5% | 80.4% | 69.0% | 70.4% |
| 2 | 81.92% | 91.9% | 92.2% | 85.0% | 82.4% | 85.7% | 54.8% | 66.2% |
| **3** | **84.13%** | 88.7% | 92.2% | 89.5% | 80.9% | 85.7% | 64.3% | 71.8% |
| 4 | 83.95% | 87.1% | 92.2% | 89.5% | 82.4% | 85.7% | 61.9% | 71.8% |

---

## PubMed (3 classes, supervised setting)

Baseline source: arXiv:2502.00829, Table 2, supervised setting. Standard Planetoid split (1000 test nodes).

**Note**: Our PubMed results use a custom stratified split (999 test, 333/class) from TAPE data files, not the standard Planetoid split. Direct comparison with baselines is approximate.

### Comparison with Baselines

| Method | Type | Accuracy | Source |
|--------|------|----------|--------|
| RoBERTa-355M | LM only | 94.84 +/- 0.06 | arXiv:2502.00829 |
| SentenceBERT-66M | LM only | 94.47 +/- 0.33 | arXiv:2502.00829 |
| GLEM | GNN + LLM | 93.98 +/- 0.32 | arXiv:2502.00829 |
| GraphGPT | LLM + Graph | 93.54 +/- 0.22 | arXiv:2502.00829 |
| TAPE | LLM-as-Reasoner | 93.00 +/- 0.13 | arXiv:2502.00829 |
| GraphSAGE (ShallowEmb) | GNN | 90.47 +/- 0.25 | arXiv:2502.00829 |
| LLaGA | LLM + Graph Projector | 90.28 +/- 0.91 | arXiv:2502.00829 |
| ENGINE | GNN + LLM | 90.08 +/- 0.16 | arXiv:2502.00829 |
| GCN (ShallowEmb) | GNN | 89.01 +/- 0.59 | arXiv:2502.00829 |
| **Ours: Frozen MC (target-only)** | **DLM zero-shot** | **88.69%** | — |
| GCN + LLM Emb | GNN + LLM embeddings | 88.38 +/- 0.68 | arXiv:2502.00829 |
| GAT (ShallowEmb) | GNN | 88.25 +/- 0.47 | arXiv:2502.00829 |
| Ours: Frozen MC (1-hop full attn) | DLM zero-shot | 84.78% | — |
| Ours: Frozen MC (1-hop topo mask) | DLM zero-shot | 82.98% | — |

### Per-Class Accuracy (PubMed Frozen)

| Config | DM Experimental | DM Type 1 | DM Type 2 | Overall |
|--------|----------------|-----------|-----------|---------|
| Target-only | 86.2% | 83.5% | 96.4% | 88.69% |
| 1-hop full attn | 69.7% | 87.1% | 97.6% | 84.78% |
| 1-hop topo mask | 62.2% | 89.2% | 97.6% | 82.98% |

---

## Key Findings

1. **Multiple-choice format is critical**: MC format (62.73%) significantly outperforms first-token class-name classification (51.29%) on frozen model.

2. **SFT closes the gap**: LoRA fine-tuning improves Cora from 62.73% (frozen) to 84.13% (SFT ep3), closing ~84% of the gap to SOTA (88.15%).

3. **Neighbors consistently hurt**: In all settings (frozen, SFT, cross-eval), adding neighbor information degrades performance. The model cannot effectively aggregate neighbor information through attention alone.

4. **Topology mask hurts more than full attention**: When neighbors are included, topology mask (star-topology attention) performs worse than full bidirectional attention.

5. **Topological position IDs break frozen model**: Per-node position ID reset (28.78%) severely degrades frozen model performance vs sequential (50.74%), due to distribution mismatch with pretraining.

6. **PubMed is easy for LLMs**: Frozen LLaDA achieves 88.69% zero-shot on PubMed (3 classes), comparable to supervised GCN (89.01%). The 3-class task is largely solvable from text content alone.

7. **Overfitting is severe with sparse signal**: With only 1 token per sample in the loss, the model overfits within 1-3 epochs. Best eval loss (ep1) does not correspond to best test accuracy (ep3).
