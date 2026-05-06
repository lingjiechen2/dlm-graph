# Experiment conclusions — topo vs. notopo

三条核心结论（基于 results.md §13 / §14 / §16 的实测数据）：

## 1. pubmed seq=4k：topo 反超 notopo

在序列足够长 + 训练足够久时，topo 的结构先验从负担变成 net positive。这给"什么时候用 topo"提供了具体答案 —— **长 seq + 长训练**。

| seq | notopo peak | topo peak | gap (topo − notopo) |
| --- | ----------- | --------- | ------------------- |
| 2k (§9)  | 95.30 @ 656 | 94.59 @ 656 | **−0.71** |
| 4k (§13) | 95.70 @ 992 | **96.30** @ 744 | **+0.60** |

1.31 pt 的反转，且 96.30 是目前所有 pubmed 设置下的全局 topo 峰值。

## 2. cora seq=4k aligned 不输 seq=2k

之前认为"长 seq 在 cora 上没用"是**优化预算混淆**的伪结论 —— §11 的 seq=4k 跑只有 340 个 step（vs §1 的 510 个）。修正后（§14 用 eff bs=32 对齐 510 step），cora 也是 **+0.18 pt**（虽小但同向）。

| run | seq | total steps | topo peak |
| --- | --- | ----------- | --------- |
| §1  cora_20260429_mcdigit_nonb_fixed | 2k | 510 | 90.04 @ 312 |
| §11 cora_20260502_mcdigit_nonb_seq4k | 4k | 340 | 88.93 @ 272 |
| §14 cora_20260503_mcdigit_nonb_seq4k_aligned | 4k | 510 | **90.22 @ 286** |

## 3. r=128 只关闭一半 cora gap

H2（LoRA 容量不足）是真实因素但**不是主因**，剩余 gap 要靠攻击 H1（结构 / 邻居信息瓶颈）或 H3（邻居梯度信号缺失）来填。

| run | r | topo peak | notopo peak (§1) | gap |
| --- | - | --------- | ---------------- | --- |
| §1  cora topo  | 64  | 90.04 @ 312 | 90.77 @ 364 | **−0.73** |
| §16 cora topo  | 128 | **90.41 @ 390** | 90.77 | **−0.36** |

容量翻倍只关闭了 ~51% 的 gap。现在 cora 是**唯一还有 gap 的数据集** —— 如果方案 4（aux MLM on neighbors / 真实子图 mask）在 cora 上也能反超，三个数据集（cora / pubmed / products）都将进入 "topo ≥ notopo" 的状态。

---

## §21. arxiv 二代 4-GPU DDP topo —— 数据 / 答案 / 容量 / 加权四件套修复（run tag `arxiv_20260506_digit0pad_lgboost_r128`，**计划中**）

针对 §20 (`arxiv_20260503`) ckpt-1668 在 acc=71.70 见顶（仍距 LLaGA 报告的 ~76% 约 4 pt）做的同时四点改动。所有变化集中在新 `RUN_TAG = arxiv_20260506_digit0pad_lgboost_r128`，名字直接编码三个核心变化（`digit0pad` 答案格式 / `lgboost` cs.LG 加权 / `r128` LoRA 容量）。

### 修复清单

| 改动 | 旧 (§20) | 新 (§21) | 原因 |
|---|---|---|---|
| arxiv 类标签 | `cs.CL idx 30 = "Computational Complexity"`（与 cs.CC 字面同名） | `cs.CL idx 30 = "Computation and Language"` | LLaGA 缓存抄错；cs.CC 33% acc / cs.CL 85% acc 两类互相挤占（fix 已合并 commit `344c379`） |
| answer label style | `digit0` → `"0".."39"`（10 个 1-token + 30 个 2-token） | `digit0_pad` → `"00".."39"`（全部 2-token） | 旧格式下 1-token 类（cs.NA/cs.MM/cs.CY 等）在第 0 位 logit 与 2-token 类前缀冲突 (`'1'` 既是 class 1 也是 10–19 的 prefix)，calibration 系统性不公平；6 个 1-token 类 acc 直接 0% |
| LoRA 容量 | r=alpha=64（167M trainable，2.05% 模型参数） | r=alpha=128（335M trainable，4.1% 模型参数） | §16 cora 验证 r=128 闭合一半 gap；arxiv 40 类 capacity 瓶颈更紧，预期收益更大 |
| class boost | none（按 train 自然分布） | `Machine Learning:3.0, Artificial Intelligence:2.0, Neural and Evolutionary Computing:2.0` | cs.LG 是 22.10% test 占比但 7.69% train 占比（OGB 时间切分 → 2.87× 分布偏移），acc 56-59% 是整体瓶颈；cs.AI / cs.NE 是 cs.LG 的混淆密集区 |

### 不变项（沿用 §20）

`max_seq_len=4096`, `max_hops=2`, `max_neighbors_per_hop=10`, `prompt_format=mc_digit`, `max_answer_tokens=2`, `include_neighbor_labels=False`, `position_id_type=sequential`, `use_topology_mask=True` (topo only), `max_train_samples=20000`, `max_steps=1668` (4 epoch over 20k @ eff_bs 48), `learning_rate=5e-5`, `save_steps=0.1` (10 ckpts), `eval_strategy=no`, `target_modules=all-linear`, `gradient_checkpointing=True`, `cls_loss_weight=0.0`. 4-GPU DDP via torchrun (per_device_batch=3, grad_accum=4, world=4 → eff_bs=48), trap auto-claims GPU 1,2,3,5 with sample_gen on exit.

### 预期收益（按贡献从大到小）

| 改动 | 预期 contribution | 凭据 |
|---|---|---|
| cs.CL fix | +1-2 pt overall | cs.CC 33→60+, cs.CL 85→88（per-class × test 占比加权） |
| `digit0_pad` | +1-2 pt overall | 解除 8 类 0% acc 的 calibration bias，cs.NA/cs.MM/cs.CY 起码能进 30%+ |
| cs.LG boost ×3 | +1-2 pt overall | cs.LG 56% → 62%+（占 22% test 时每 +1 pt 贡献整体 +0.22 pt）|
| r=128 | +0.5-1 pt | §16 cora 类比，arxiv capacity 瓶颈更紧时收益略大 |
| **合计** | **+3-5 pt** | 目标 ckpt-final acc **>= 75%**，逼近 LLaGA |

### 训练时长

r=128 比 r=64 略慢（额外 167M trainable param 的 forward/backward），预计 32 sec/step → 35-37 sec/step。1668 step × 36 sec ≈ **~17 h**。完成时刻取决于启动时间。
