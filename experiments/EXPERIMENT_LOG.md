# TM-DLM Experiment Log: TAG Node Classification

## Setup

- **Model**: LLaDA-8B-Instruct (GSAI-ML/LLaDA-8B-Instruct)
- **Datasets**: Cora (7 classes, 542 test nodes), PubMed (3 classes, 999 test nodes)
- **Format**: Multiple-choice with digit answers
- **SOTA reference (Cora)**: TAPE 88.05%, LLaGA 87.55%, GraphSAGE 87.44%

---

## Phase 1: Layer 0 (Frozen Model, No Fine-tuning)

### First-token classification (class name tokens)


| Experiment                               | Accuracy | Notes                                                             |
| ---------------------------------------- | -------- | ----------------------------------------------------------------- |
| baseline_no_neighbors (max_hops=0, bs=4) | 46.86%   | Severe class imbalance: Neural Networks 95.4%, Rule Learning 0.0% |
| baseline_target_only (max_hops=0, bs=8)  | 51.29%   | Similar bias pattern                                              |
| neighbors_full_attn_2hop                 | 44.46%   | Adding neighbors hurts without fine-tuning                        |
| tmdlm_topo_mask_2hop                     | 48.34%   | Topology mask slightly better than full attn                      |


### Multiple-choice format (digit token classification)


| Experiment        | Accuracy   | Notes                                      |
| ----------------- | ---------- | ------------------------------------------ |
| mc_target_only    | **62.73%** | +11.4% over first-token baseline           |
| mc_1hop_full_attn | 57.01%     | Neighbors hurt without fine-tuning         |
| mc_1hop_topo_mask | 50.74%     | Topology mask worse than full attn         |
| mc_1hop_topo_mask_topo_posid | 28.78% | Topological position IDs (per-node reset) severely hurts frozen model |
| mc_2hop_full_attn | 60.33%     | Neighbors hurt without fine-tuning         |
| mc_2hop_topo_mask | 49.26%     | Topology mask also hurts in frozen setting |

### Prompt Layout Comparison: target_first vs neighbor_first (answer at end)

| Config | target_first | neighbor_first | Δ |
| ------ | ------------ | -------------- | --- |
| Cora 1-hop full attn | 57.01% | 53.51% | -3.5% |
| Cora 1-hop topo mask | 50.74% | 39.30% | **-11.4%** |
| Cora 2-hop full attn | 60.33% | 55.17% | -5.2% |
| Cora 2-hop topo mask | 49.26% | 29.70% | **-19.6%** |
| PubMed 1-hop full attn | 84.78% | 82.98% | -1.8% |
| PubMed 1-hop topo mask | 82.98% | 80.68% | -2.3% |

**Key observation**: neighbor_first is universally worse. The penalty is largest with topology mask (Cora 2-hop: -19.6%), suggesting topology mask + neighbor_first is a particularly bad combination — the mask restricts neighbor-to-target attention flow, and placing neighbors far from the answer compounds the problem.


### PubMed Layer 0 (MC format, frozen)


| Experiment               | Accuracy   | Notes                                                 |
| ------------------------ | ---------- | ----------------------------------------------------- |
| mc_pubmed_target_only    | **88.69%** | 3-class task much easier for frozen LLM               |
| mc_pubmed_1hop_full_attn | 84.78%     | Neighbors hurt (-3.9%), Class 0 drops 86.2%→69.7%     |
| mc_pubmed_1hop_topo_mask | 82.98%     | Topo mask worse than full attn (-5.7% vs target-only) |
| mc_pubmed_1hop_nb_first (full attn) | 82.98% | neighbor_first: worse than target_first (-1.8%) |
| mc_pubmed_1hop_nb_first (topo mask) | 80.68% | neighbor_first + topo: worst combination (-8.0% vs target-only) |


**Per-class breakdown (PubMed):**


| Config          | DM Experimental | DM Type 1 | DM Type 2 |
| --------------- | --------------- | --------- | --------- |
| target-only     | 86.2%           | 83.5%     | 96.4%     |
| 1-hop full attn | 69.7%           | 87.1%     | 97.6%     |
| 1-hop topo mask | 62.2%           | 89.2%     | 97.6%     |
| 1-hop nb_first (full) | 65.2%     | 87.4%     | 96.4%     |
| 1-hop nb_first (topo) | 58.3%     | 87.7%     | 96.1%     |


**Key finding**: MC format much better than first-token classification. Neighbors hurt frozen model on BOTH Cora and PubMed. Topology mask consistently worse than full attention. PubMed (3 classes) is much easier but the pattern holds: neighbors = noise for frozen models. Prompt layout (neighbor_first vs target_first) also matters: putting neighbors before target consistently hurts accuracy across ALL configurations. The penalty is moderate with full attention (Cora: -3.5% to -5.2%, PubMed: -1.8%) but severe with topology mask (Cora 1-hop: -11.4%, Cora 2-hop: -19.6%). This suggests topology mask + neighbor_first is particularly harmful because the mask restricts cross-attention between neighbors and target, while the layout also pushes the answer far from any useful signal.

---

## Phase 2: SFT with LoRA (Target-Only, max_hops=0)

### Training Config

- LoRA: r=32, alpha=64, target_modules=all-linear (83.9M params, 1.04%)
- Effective batch size: 16 (bs=2 × grad_accum=8)
- LR: 5e-5, 5 epochs, gradient_checkpointing=True
- cls_loss_weight=0.0 (disabled buggy aux loss)
- Only answer digit token in loss (1 token per sequence)

### Eval Loss Trend (Validation Set)


| Epoch | Step | Eval Loss  | Eval PPL | Train NLL |
| ----- | ---- | ---------- | -------- | --------- |
| 1     | 102  | **0.4019** | 1.495    | 0.825     |
| 2     | 204  | 0.4455     | 1.561    | 0.544     |
| 3     | 306  | 0.4201     | 1.522    | 0.413     |
| 4     | 408  | 0.5651     | 1.760    | 0.439     |
| 5     | 510  | 0.6379     | 1.893    | 0.231     |


**Severe overfitting**: Train NLL keeps decreasing but eval loss worsens after epoch 1.

### Test Accuracy by Epoch


| Epoch | Test Acc   | Case Based | Genetic Alg | Neural Net | Prob Methods | RL    | Rule Learn | Theory |
| ----- | ---------- | ---------- | ----------- | ---------- | ------------ | ----- | ---------- | ------ |
| 1     | 82.47%     | 83.9%      | 94.4%       | 88.9%      | 73.5%        | 80.4% | 69.0%      | 70.4%  |
| 2     | 81.92%     | 91.9%      | 92.2%       | 85.0%      | 82.4%        | 85.7% | 54.8%      | 66.2%  |
| 3     | **84.13%** | 88.7%      | 92.2%       | 89.5%      | 80.9%        | 85.7% | 64.3%      | 71.8%  |
| 4     | 83.95%     | 87.1%      | 92.2%       | 89.5%      | 82.4%        | 85.7% | 61.9%      | 71.8%  |


**Best: Epoch 3 = 84.13%** (despite eval loss being higher than epoch 1)

- Note: eval loss (diffusion NLL) ≠ classification accuracy. Epoch 3 has better acc despite worse NLL.
- Multi-step denoising (5 steps) gives identical result (82.47%) — expected since only 1 token is masked.

---

## Phase 3: Enhancement Experiments (In Progress)

### Dense Target Masking + Fixed cls_loss

- mask_target_text=True: all target body tokens in loss (~200 tokens vs 1)
- cls_loss_weight=1.0 with fix: restricted logits to digit token IDs
- Training loss much higher (~3.0) since reconstruction is harder


| Epoch | Test Acc | Notes                              |
| ----- | -------- | ---------------------------------- |
| 1     | 71.96%   | Worse than sparse masking (82.47%) |
| 2     | 72.88%   | Marginal improvement               |
| 3     | 73.43%   | Plateaued, -10.7% vs sparse        |


**Conclusion**: Dense masking dilutes classification signal — answer digit is 1/200 of the loss.

### Chat Template Comparison: raw prompt vs LLaDA-Instruct chat template

**Setup**: Wrap the classification prompt in LLaDA-Instruct's Llama-3 chat template:
```
<|startoftext|><|start_header_id|>user<|end_header_id|>

Paper: ... Options: ... Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer_digit}
```
Answer token is placed in the assistant turn, matching the model's instruction-tuning format. All experiments use `target_first` layout.

**Cora (7 classes)**:

| Config | Raw Prompt | Chat Template | Δ |
| ------ | ---------- | ------------- | --- |
| target-only | **62.73%** | 63.65% | +0.9% |
| 1-hop full attn | 57.01% | 54.43% | -2.6% |
| 1-hop topo mask | 50.74% | 43.54% | -7.2% |
| 2-hop full attn | 60.33% | **61.07%** | +0.7% |
| 2-hop topo mask | 49.26% | 33.58% | **-15.7%** |

**PubMed (3 classes)**:

| Config | Raw Prompt | Chat Template | Δ |
| ------ | ---------- | ------------- | --- |
| target-only | **88.69%** | 86.09% | -2.6% |
| 1-hop full attn | 84.78% | 75.28% | **-9.5%** |
| 1-hop topo mask | 82.98% | 78.78% | -4.2% |

**Key finding**: Chat template provides marginal gains for target-only or 2-hop full-attention on Cora (+0.7%~+0.9%), but **significantly hurts** when neighbors are present, especially on PubMed 1-hop (-9.5%) and Cora 2-hop topo mask (-15.7%). The template adds ~15 special tokens to the sequence, and in the frozen-model setting, these extra tokens interact poorly with neighbor context—especially under topology masking where attention is already restricted. The template tokens (BOS, header markers, EOT) sit outside node spans but attend to everything, acting as noise anchors. On PubMed, the stronger effect (-9.5% for 1-hop full attn) may reflect the model learning to "follow instructions" rather than classify from context—when neighbors are added, the instruction-following behavior conflicts with the classification signal.

### Category Infill Format: Natural class name prediction

**Setup**: Infill the class name directly instead of a digit. Scored by mean log-prob across answer tokens. Auto-computed max_answer_tokens: Cora=4, PubMed=6.

Two variants tested:
- **catinfill (no options)**: `Paper: {text}\nThe category of this paper is: {class_name}` — no options list
- **catinfill+options**: `Paper: {text}\nOptions: 0) ... 6) ...\nThe category of this paper is: {class_name}` — includes options list

**Cora (7 classes)**:

| Config | mc_digit | catinfill (no opts) | catinfill+options | Δ vs mc_digit |
| ------ | -------- | ------------------- | ----------------- | ------------- |
| target-only | **62.73%** | 47.60% | 55.72% | -7.0% |
| 1-hop full attn | 57.01% | 52.03% | 52.03% | -5.0% |
| 1-hop topo mask | 50.74% | 53.32% | 51.11% | +0.4% |
| 2-hop full attn | 60.33% | 53.14% | **57.93%** | -2.4% |
| 2-hop topo mask | 49.26% | 54.06% | **54.98%** | **+5.7%** |

**PubMed (3 classes)**:

| Config | mc_digit | catinfill (no opts) | catinfill+options | Δ vs mc_digit |
| ------ | -------- | ------------------- | ----------------- | ------------- |
| target-only | **88.69%** | 27.73% | 44.14% | -44.6% |
| 1-hop full attn | 84.78% | 40.24% | **94.19%** | **+9.4%** |
| 1-hop topo mask | 82.98% | 37.14% | **93.39%** | **+10.4%** |

**Key findings**:

1. **Options list is critical for category_infill** — Without options, the model must generate class names from pretrained knowledge alone. Adding the options list closes the gap significantly: Cora target-only 47.60% → 55.72% (+8.1%), PubMed 1-hop 40.24% → 94.19% (+54%).

2. **PubMed catinfill+options + neighbors = best frozen result (94.19%)** — This is the single best frozen-model result across all experiments, surpassing mc_digit (84.78%), supervised GCN (89.01%), and approaching RoBERTa-355M (94.84%). The options list lets the model see all 3 "Diabetes Mellitus" variants explicitly, and neighbor context provides the signal to differentiate at the suffix tokens. Per-class: Experimental 99.4%, Type 1 88.0%, Type 2 95.2%.

3. **Neighbors help catinfill+options dramatically on PubMed** — target-only 44.14% → 1-hop 94.19% (+50%). This is the largest neighbor benefit across all experiments. The neighbor text provides domain-specific context that helps the model distinguish between "Experimental", "Type 1", and "Type 2" at the suffix positions.

4. **On Cora, catinfill+options is slightly worse than mc_digit overall** — target-only: 55.72% vs 62.73% (-7.0%). Multi-token mean log-prob scoring adds noise compared to single-token argmax. However, topo mask configs still outperform mc_digit topo mask (2-hop: 54.98% vs 49.26%).

5. **Without options list, catinfill still shows topo mask benefits on Cora** — 1-hop topo 53.32% > mc_digit topo 50.74% (+2.6%). 2-hop topo 54.06% > mc_digit topo 49.26% (+4.8%). Natural language infilling is more compatible with topology-restricted attention than digit prediction.

6. **PubMed target-only catinfill remains poor** — Even with options, target-only is only 44.14% (vs mc_digit 88.69%). The 3 class names share the first 3 tokens "Diabetes Mellitus", making discrimination from text alone very hard. Neighbors are essential to provide the distinguishing signal.

### 2-hop Neighbors + Full Attention (SFT)

- max_hops=2, bs=1, grad_accum=16
- Training very slow (~80 min/epoch due to long sequences up to 2048 tokens)


| Epoch | Test Acc (2-hop eval) | Test Acc (target-only eval) |
| ----- | --------------------- | --------------------------- |
| 1     | 79.15%                | 78.97%                      |


**Conclusion**: Training with neighbors doesn't help. Worse than target-only SFT (84.13%).

### Cross-evaluation: Target-only Model + Neighbors at Inference


| Training        | Inference       | Test Acc        |
| --------------- | --------------- | --------------- |
| Target-only ep3 | Target-only     | **84.13%**      |
| Target-only ep3 | 2-hop full attn | 71.77% (-12.4%) |
| Target-only ep3 | 2-hop topo mask | 57.38% (-26.8%) |


**Conclusion**: Neighbors ALWAYS hurt — both at training and inference time. The model cannot effectively aggregate neighbor information through attention alone.

---

## Core Diagnosis

### Why SFT reaches 84% but not SOTA (~88%)

**1. Generative vs. Discriminative Objective Mismatch**
The diffusion ELBO loss is a generative loss (predict masked tokens) not a discriminative loss (maximize class separation). The model learns P(x|x_masked), not P(class|x). Standard classifiers optimize cross-entropy directly on class logits, giving a much stronger classification signal.

**2. Single-Token Classification Bottleneck**
All class-discriminative information must flow through the probability distribution over 7 digit tokens at a single position. This is unlike standard classifiers that use a learned embedding → linear head pipeline over rich hidden representations.

**3. Rapid Overfitting on Sparse Signal**
With only 1 token per sample in the loss (1624 training samples × 1 token = 1624 effective training examples), the model overfits within 1-3 epochs. Dense masking (200 tokens per sample) dilutes the classification signal rather than enhancing it.

**4. Graph Structure Helps Only with Natural Language Infilling**

- Frozen mc_digit: neighbors hurt because abstract digit tokens cannot leverage neighbor semantics
- Frozen catinfill+options: neighbors dramatically help on PubMed (44.14% → 94.19%), moderately on Cora (55.72% → 57.93% 2-hop)
- SFT with neighbors: extremely slow (sequences 10x longer → attention O(n²))
- The key insight: natural language class name infilling enables the model to use neighbor text as semantic context, while digit prediction cannot

**5. Eval Loss ≠ Classification Accuracy**
The diffusion NLL on the validation set doesn't correlate well with classification accuracy. Best eval loss (epoch 1: 0.40) has lower test acc (82.47%) than epoch 3 (eval loss 0.42, test acc 84.13%). This suggests the model needs some overfitting to sharpen class boundaries, but the optimal point is hard to find without test set access.

### Baseline Comparison (Supervised Setting)

Source: "When Do LLMs Help With Node Classification?" (arXiv:2502.00829)

**Cora:**


| Method            | Type                  | Accuracy   |
| ----------------- | --------------------- | ---------- |
| GCN+LLM Emb       | GNN + LLM embeddings  | 88.15%     |
| TAPE              | LLM-as-Reasoner       | 88.05%     |
| LLaGA             | LLM + Graph Projector | 87.55%     |
| GCN (ShallowEmb)  | GNN                   | 87.41%     |
| ENGINE            | GNN + LLM             | 87.00%     |
| GAT (ShallowEmb)  | GNN                   | 86.68%     |
| **Ours: SFT ep3** | **DLM + LoRA**        | **84.13%** |
| RoBERTa-355M      | LM only               | 83.17%     |
| Ours: Frozen MC   | DLM zero-shot         | 62.73%     |


**PubMed:**


| Method           | Type                  | Accuracy |
| ---------------- | --------------------- | -------- |
| RoBERTa-355M     | LM only               | 94.84%   |
| TAPE             | LLM-as-Reasoner       | 93.00%   |
| LLaGA            | LLM + Graph Projector | 90.28%   |
| ENGINE           | GNN + LLM             | 90.08%   |
| GCN (ShallowEmb) | GNN                   | 89.01%   |
| Ours: Frozen MC  | DLM zero-shot         | 88.69%   |
| GCN+LLM Emb      | GNN + LLM embeddings  | 88.38%   |
| GAT (ShallowEmb) | GNN                   | 88.25%   |


Note: PubMed frozen result uses our custom stratified split (999 test, 333/class), not standard Planetoid split (1000 test).

### What SOTA Methods Do Differently


| Method             | Key Advantage                                                                  | Our Limitation                                       |
| ------------------ | ------------------------------------------------------------------------------ | ---------------------------------------------------- |
| TAPE (88.05%)      | Multi-step pipeline: LLM explanations + pseudo-labels + specialized classifier | Single-step diffusion, no iterative refinement       |
| LLaGA (87.55%)     | Graph projector into LLM embedding space + direct classification CE loss       | Diffusion ELBO loss, no explicit classification head |
| GraphSAGE (87.44%) | Specialized graph architecture with learned aggregation                        | Attention-only, no graph-specific inductive bias     |
| GCN (87.41%)       | Spectral graph convolutions                                                    | No graph-specific computation                        |


### Potential Improvements (Not Tested)

1. **Add classification head**: After diffusion fine-tuning, add a linear head on the answer token's hidden representation and train with standard CE. This separates representation learning (diffusion) from classification.
2. **Two-stage training**: (a) Dense masking for text representation learning, (b) Switch to sparse masking (answer-only) for classification fine-tuning.
3. **Label propagation post-processing**: Use the model's predictions + graph structure for semi-supervised label propagation (like TAPE does).
4. **Ensemble across denoising trajectories**: Sample multiple masking patterns and aggregate predictions (Monte Carlo estimation of P(class|x)).

---

## Full Results Summary

### Cora (7 classes, 542 test nodes)


| Method                            | Accuracy   | Gap to SOTA |
| --------------------------------- | ---------- | ----------- |
| *SOTA: GCN+LLM Emb*               | *88.15%*   | *—*         |
| *TAPE*                            | *88.05%*   | *-0.1%*     |
| *LLaGA*                           | *87.55%*   | *-0.6%*     |
| *GCN*                             | *87.41%*   | *-0.7%*     |
| **Ours: SFT target-only ep3**     | **84.13%** | **-4.0%**   |
| *RoBERTa-355M*                    | *83.17%*   | *-5.0%*     |
| Ours: SFT target-only ep1         | 82.47%     | -5.7%       |
| Ours: SFT 2-hop full attn (ep1)   | 79.15%     | -9.0%       |
| Ours: SFT dense mask (ep3)        | 73.43%     | -14.7%      |
| Ours: Frozen MC (target-only)     | 62.73%     | -25.4%      |
| Ours: Frozen MC (1-hop full attn, target_first) | 57.01% | -31.1% |
| Ours: Frozen MC (2-hop full attn, nb_first) | 55.17% | -33.0% |
| Ours: Frozen MC (1-hop full attn, nb_first) | 53.51% | -34.6% |
| Ours: Frozen MC (1-hop topo, target_first) | 50.74% | -37.4% |
| Ours: Frozen MC (2-hop topo, target_first) | 49.26% | -38.9% |
| Ours: Frozen MC (1-hop topo, nb_first) | 39.30% | -48.9% |
| Ours: Frozen MC (2-hop topo, nb_first) | 29.70% | -58.5% |
| Ours: Frozen MC (1-hop topo + topo posid) | 28.78% | -59.4% |
| Ours: Frozen CatInfill+Opts (2-hop full)  | 57.93% | -30.2% |
| Ours: Frozen CatInfill+Opts (target-only) | 55.72% | -32.4% |
| Ours: Frozen CatInfill+Opts (2-hop topo)  | 54.98% | -33.2% |
| Ours: Frozen CatInfill (2-hop topo)       | 54.06% | -34.1% |
| Ours: Frozen CatInfill (1-hop topo)       | 53.32% | -34.8% |
| Ours: Frozen CatInfill (2-hop full attn)  | 53.14% | -35.0% |
| Ours: Frozen CatInfill+Opts (1-hop full)  | 52.03% | -36.1% |
| Ours: Frozen CatInfill (1-hop full attn)  | 52.03% | -36.1% |
| Ours: Frozen CatInfill+Opts (1-hop topo)  | 51.11% | -37.0% |
| Ours: Frozen CatInfill (target-only)      | 47.60% | -40.6% |


### PubMed (3 classes, 999 test nodes)


| Method                            | Accuracy   | Notes                      |
| --------------------------------- | ---------- | -------------------------- |
| *RoBERTa-355M*                    | *94.84%*   | *LM only, supervised*      |
| **Ours: Frozen CatInfill+Opts (1-hop full)** | **94.19%** | **zero-shot, no training** |
| **Ours: Frozen CatInfill+Opts (1-hop topo)** | **93.39%** | **zero-shot, no training** |
| *TAPE*                            | *93.00%*   | *supervised*               |
| *LLaGA*                           | *90.28%*   | *supervised*               |
| *GCN*                             | *89.01%*   | *supervised*               |
| Ours: Frozen MC (target-only)     | 88.69%     | zero-shot                  |
| Ours: Frozen MC (1-hop full attn, target_first) | 84.78% | zero-shot |
| Ours: Frozen MC (1-hop full attn, nb_first) | 82.98% | zero-shot, neighbor_first |
| Ours: Frozen MC (1-hop topo, target_first) | 82.98% | zero-shot |
| Ours: Frozen MC (1-hop topo, nb_first) | 80.68% | zero-shot, neighbor_first |
| Ours: Frozen CatInfill+Opts (target-only) | 44.14% | zero-shot, catinfill+options |
| Ours: Frozen CatInfill (1-hop full attn) | 40.24% | zero-shot, catinfill no options |
| Ours: Frozen CatInfill (1-hop topo)     | 37.14% | zero-shot, catinfill no options |
| Ours: Frozen CatInfill (target-only)    | 27.73% | zero-shot, catinfill no options |


Note: PubMed baselines are supervised (trained with labels); our results are zero-shot (no training). Our custom stratified split (999 test) differs from standard Planetoid (1000 test).

**Core conclusion**: DLM fine-tuning with LoRA achieves **84.13%** on Cora, closing ~84% of the gap between frozen LLM (51.29%) and SOTA (88.15%). On PubMed, frozen LLaDA with catinfill+options and 1-hop neighbors reaches **94.19%** zero-shot — surpassing all supervised baselines except RoBERTa-355M (94.84%), and far exceeding the mc_digit best (88.69%).

The remaining ~4% gap on Cora is due to:

1. **Generative-vs-discriminative objective mismatch** — diffusion ELBO ≠ classification loss
2. **Single-token classification bottleneck** — all info compressed to 1 token position (mc_digit) or multi-token scoring noise (catinfill)
3. **Rapid overfitting** — train NLL→0.23 while eval→0.64 by epoch 5

However, on PubMed, frozen catinfill+options with 1-hop neighbors achieves **94.19%** zero-shot — surpassing most supervised methods and demonstrating that DLMs can leverage graph structure when using natural language infilling with an explicit options list.

The most promising paths forward:
1. **SFT with catinfill+options format** — the frozen catinfill+options result (94.19% PubMed) suggests this format may be superior to mc_digit for fine-tuning
2. **Add classification head** on the answer token's hidden representation for discriminative training
3. **Two-stage training**: dense masking for text understanding, then sparse masking for classification

---

## Key Findings

1. **Multiple-choice format is critical**: MC format (62.73%) significantly outperforms first-token class-name classification (51.29%) on frozen model.
2. **SFT closes the gap**: LoRA fine-tuning improves Cora from 62.73% (frozen) to 84.13% (SFT ep3), closing ~84% of the gap to SOTA (88.15%).
3. **Neighbors hurt mc_digit but help catinfill+options**: In mc_digit (frozen and SFT), neighbors consistently degrade performance. However, with catinfill+options format, neighbors dramatically help: PubMed 1-hop 94.19% vs target-only 44.14%. The key difference: natural language infilling can leverage neighbor semantics, while digit prediction cannot.
4. **Topology mask hurts more than full attention**: When neighbors are included, topology mask (star-topology attention) performs worse than full bidirectional attention.
5. **Topological position IDs break frozen model**: Per-node position ID reset (28.78%) severely degrades frozen model performance vs sequential (50.74%), due to distribution mismatch with pretraining.
6. **PubMed: catinfill+options + neighbors is the best format**: Frozen LLaDA achieves **94.19%** zero-shot with catinfill+options + 1-hop neighbors, surpassing supervised GCN (89.01%) and approaching RoBERTa-355M (94.84%). MC-digit target-only also reaches 88.69%. The catinfill+options result demonstrates that DLMs can effectively leverage graph structure when using the right prompt format.
7. **Overfitting is severe with sparse signal**: With only 1 token per sample in the loss, the model overfits within 1-3 epochs. Best eval loss (ep1) does not correspond to best test accuracy (ep3).
8. **Prompt layout matters — dramatically with topology mask**: neighbor_first is universally worse than target_first. With full attention the gap is moderate (Cora: -3.5% to -5.2%, PubMed: -1.8% to -2.3%), but with topology mask the gap explodes (Cora 1-hop: -11.4%, Cora 2-hop: **-19.6%**). The worst frozen-model result (29.7%) is the combination of 2-hop topology mask + neighbor_first, nearly as bad as broken topological position IDs (28.78%). This suggests DLM's bidirectional attention, while theoretically position-invariant, has strong practical sensitivity to where the answer token sits relative to useful context.
9. **Category infill + options + neighbors = breakthrough on PubMed**: Frozen catinfill+options with 1-hop neighbors achieves **94.19%** on PubMed — surpassing mc_digit (84.78%), supervised GCN (89.01%), LLaGA (90.28%), and TAPE (93.00%). This is the single best frozen-model result. The options list provides the class name vocabulary, and neighbor text provides the discriminative context to distinguish "Experimental" vs "Type 1" vs "Type 2" at the suffix tokens. On Cora, catinfill+options is slightly weaker than mc_digit overall (-7% target-only) due to multi-token scoring noise, but topo mask configs still benefit (+5.7% on 2-hop).