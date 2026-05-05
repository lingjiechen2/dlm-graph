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
