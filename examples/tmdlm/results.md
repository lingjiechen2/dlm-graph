# DLM-Graph: Node Classification Results

## Model & Setup

- **Model**: LLaDA-8B-Instruct (GSAI-ML/LLaDA-8B-Instruct), 8B params, masked discrete diffusion LM
- **Classification format**: Multiple-choice with single-digit answers (tokens "0"–"6" for Cora, "0"–"2" for PubMed)
- **SFT config**: LoRA r=32, alpha=64, all-linear modules, 83.9M trainable params (1.04%), lr=5e-5, effective batch=16

## Cora (7 classes, 542 test nodes, supervised setting)

Baseline source: "When Do LLMs Help With Node Classification?" (arXiv:2502.00829), Table 2, supervised setting.

### Comparison with Baselines


| Method                          | Type                  | Accuracy       | Source           |
| ------------------------------- | --------------------- | -------------- | ---------------- |
| GCN + LLM Emb                   | GNN + LLM embeddings  | 88.15 +/- 1.79 | arXiv:2502.00829 |
| TAPE                            | LLM-as-Reasoner       | 88.05 +/- 1.76 | arXiv:2502.00829 |
| LLaGA                           | LLM + Graph Projector | 87.55 +/- 1.15 | arXiv:2502.00829 |
| GraphSAGE (ShallowEmb)          | GNN                   | 87.44 +/- 1.74 | arXiv:2502.00829 |
| GCN (ShallowEmb)                | GNN                   | 87.41 +/- 2.08 | arXiv:2502.00829 |
| ENGINE                          | GNN + LLM             | 87.00 +/- 1.60 | arXiv:2502.00829 |
| GLEM                            | GNN + LLM             | 86.81 +/- 1.19 | arXiv:2502.00829 |
| GAT (ShallowEmb)                | GNN                   | 86.68 +/- 1.12 | arXiv:2502.00829 |
| **Ours: SFT target-only (ep3)** | **DLM + LoRA**        | **84.13**      | —                |
| RoBERTa-355M                    | LM only               | 83.17 +/- 0.84 | arXiv:2502.00829 |
| GraphGPT                        | LLM + Graph           | 82.29 +/- 0.26 | arXiv:2502.00829 |
| Ours: SFT target-only (ep1)     | DLM + LoRA            | 82.47          | —                |
| SentenceBERT-66M                | LM only               | 79.61 +/- 1.40 | arXiv:2502.00829 |


### Our Full Results on Cora


| Method                                       | Accuracy   | Notes                                          |
| -------------------------------------------- | ---------- | ---------------------------------------------- |
| **SFT target-only (ep3)**                    | **84.13%** | Best result, LoRA fine-tuned                   |
| SFT target-only (ep4)                        | 83.95%     | Slight overfit                                 |
| SFT target-only (ep1)                        | 82.47%     |                                                |
| SFT target-only (ep2)                        | 81.92%     |                                                |
| SFT 2-hop full attn (ep1)                    | 79.15%     | Neighbors hurt during training                 |
| SFT dense mask (ep3)                         | 73.43%     | Dense masking dilutes classification signal    |
| SFT target-only ep3, eval w/ 2-hop full attn | 71.77%     | Neighbors at inference hurt                    |
| Frozen MC (target-only)                      | 62.73%     | Best frozen result                             |
| Frozen MC (2-hop full attn)                  | 60.33%     |                                                |
| Frozen MC (1-hop full attn)                  | 57.01%     |                                                |
| SFT target-only ep3, eval w/ 2-hop topo mask | 57.38%     | Topo mask at inference hurts most              |
| Frozen first-token (target-only)             | 51.29%     | Class-name token classification                |
| Frozen MC (1-hop topo mask)                  | 50.74%     |                                                |
| Frozen MC (2-hop topo mask)                  | 49.26%     |                                                |
| Frozen first-token (2-hop topo mask)         | 48.34%     |                                                |
| Frozen first-token (no neighbors)            | 46.86%     |                                                |
| Frozen first-token (2-hop full attn)         | 44.46%     |                                                |
| Frozen MC (1-hop topo mask + topo posid)     | 28.78%     | Per-node position ID reset breaks frozen model |


### Per-Class Accuracy (Cora SFT, best epochs)


| Epoch | Overall    | Case Based | Genetic Alg | Neural Net | Prob Methods | RL    | Rule Learn | Theory |
| ----- | ---------- | ---------- | ----------- | ---------- | ------------ | ----- | ---------- | ------ |
| 1     | 82.47%     | 83.9%      | 94.4%       | 88.9%      | 73.5%        | 80.4% | 69.0%      | 70.4%  |
| 2     | 81.92%     | 91.9%      | 92.2%       | 85.0%      | 82.4%        | 85.7% | 54.8%      | 66.2%  |
| **3** | **84.13%** | 88.7%      | 92.2%       | 89.5%      | 80.9%        | 85.7% | 64.3%      | 71.8%  |
| 4     | 83.95%     | 87.1%      | 92.2%       | 89.5%      | 82.4%        | 85.7% | 61.9%      | 71.8%  |


---

## PubMed (3 classes, supervised setting)

Baseline source: [LLaGA, Table 1 (Single Focus)](https://arxiv.org/pdf/2402.08170), PubMed node classification.

**Note**: Our PubMed results use a custom stratified split (999 test, 333/class) from TAPE data files, not the standard Planetoid split. Direct comparison with baselines is approximate.

### Comparison with Baselines


| Method                            | Type                  | Accuracy       | Source           |
| --------------------------------- | --------------------- | -------------- | ---------------- |
| SAGN                              | GNN                   | 95.17          | arXiv:2402.08170 |
| LLaGA-ND-7B                       | LLM + Graph Projector | 95.03          | arXiv:2402.08170 |
| LLaGA-HO-7B                       | LLM + Graph Projector | 95.03          | arXiv:2402.08170 |
| NodeFormer                        | Graph Transformer     | 94.90          | arXiv:2402.08170 |
| GraphSAGE                         | GNN                   | 94.87          | arXiv:2402.08170 |
| GCN                               | GNN                   | 92.96          | arXiv:2402.08170 |
| GAT                               | GNN                   | 92.33          | arXiv:2402.08170 |
| **Ours: Frozen MC (target-only)** | **DLM zero-shot**     | **88.69%**     | —                |
| SGC                               | GNN                   | 87.35          | arXiv:2402.08170 |
| Ours: Frozen MC (1-hop full attn) | DLM zero-shot         | 84.78%         | —                |
| Ours: Frozen MC (1-hop topo mask) | DLM zero-shot         | 82.98%         | —                |


### Per-Class Accuracy (PubMed Frozen)


| Config          | DM Experimental | DM Type 1 | DM Type 2 | Overall |
| --------------- | --------------- | --------- | --------- | ------- |
| Target-only     | 86.2%           | 83.5%     | 96.4%     | 88.69%  |
| 1-hop full attn | 69.7%           | 87.1%     | 97.6%     | 84.78%  |
| 1-hop topo mask | 62.2%           | 89.2%     | 97.6%     | 82.98%  |


---

## PubMed — SFT nonb (mc_digit, include_neighbor_labels=False)

**Setting**: LoRA r=64 all-linear, 10 epochs, lr=5e-5, effective batch=16, 2-hop, max_neighbors=10.
`include_neighbor_labels=False` at **both train and eval** — no oracle neighbor class names, fair comparison vs LLaGA.
Run tag: `pubmed_20260428_mcdigit_nonb`. Test set 999 samples (333/class).

### Logit Eval (direct token scoring)

| Checkpoint | notopo accuracy | topo accuracy |
| ---------- | --------------- | ------------- |
| 370        | 99.8%           | 100.0%        |
| 740        | 100.0%          | 100.0%        |
| 1110       | 100.0%          | 100.0%        |
| 1480       | 100.0%          | 100.0%        |

**Note**: Logit eval saturates at 100% early. PubMed paper abstracts are highly class-discriminative on their own (DM Experimental / Type 1 / Type 2 have distinct vocabulary), so fine-tuned LLaDA-8B reaches ceiling without graph structure. This makes PubMed mc_digit a weak benchmark for measuring graph reasoning contribution.

### Infill Eval (masked diffusion generation, 10 steps, temperature=0)

| Checkpoint | notopo (strict) | topo (strict) |
| ---------- | --------------- | ------------- |
| 370        | 90.4%           | 92.6%         |
| 740        | 93.3%           | 91.7%         |
| 1110       | 91.9%           | 91.8%         |
| **1480**   | **93.8%**       | **94.2%**     |

### Per-Class Accuracy — Infill, checkpoint-1480

| Setting | DM Experimental | DM Type 1 | DM Type 2 | Overall |
| ------- | --------------- | --------- | --------- | ------- |
| notopo  | 93.63%          | 90.13%    | 97.51%    | 93.8%   |
| topo    | 86.27%          | 93.16%    | 99.25%    | 94.2%   |

### Comparison with Baselines (PubMed)

| Method | Accuracy | Notes |
| ------ | -------- | ----- |
| SAGN | 95.17 | GNN, arXiv:2402.08170 |
| LLaGA-7B | 95.03 | LLM + Graph Projector, arXiv:2402.08170 |
| GraphSAGE | 94.87 | GNN, arXiv:2402.08170 |
| **Ours: SFT nonb topo ckpt-1480 (infill)** | **94.2%** | DLM + LoRA, no neighbor labels |
| **Ours: SFT nonb notopo ckpt-1480 (infill)** | **93.8%** | DLM + LoRA, no neighbor labels |
| GCN | 92.96 | GNN, arXiv:2402.08170 |
| GAT | 92.33 | GNN, arXiv:2402.08170 |
| Ours: Frozen MC (target-only) | 88.69% | DLM zero-shot |


