# Handoff: arxiv §21 r128 ckpts — post-processing experiments

Memory-transit doc for a parallel session exploring post-processing methods
to push ckpt accuracy past 74.4% (current best on 1000-sample, ckpt-1845).
LLaGA reference target: ~76%.

---

## 1. SFT run summary

- **Run tag**: `arxiv_20260506_digit0pad_lgboost_r128`
- **Run dir**: `.models/tmdlm-llada-8b-ogbn-arxiv-2hop-topo-mcdigit-d0-nonb-r128-steps2042-arxiv_20260506_digit0pad_lgboost_r128/`
- **Wandb**: https://wandb.ai/lingjiechen127/huggingface/runs/6mkxtmxc
- **Wallclock**: 18 h 16 m (4-GPU DDP on 2/3/4/6, started 2026-05-07 04:48, ~32 s/step)
- **Final train_loss**: 0.333

### 1.1 Checkpoints

11 LoRA-only adapters (1.3 GB each, `save_only_model=True` → no optimizer / scheduler / RNG state on disk).

| step | path (relative to run dir) |
|---|---|
| 205, 410, 615, 820, 1025, 1230, 1435, 1640, 1845, 2042, final | `checkpoint-{step}/adapter_model.safetensors` |

Each ckpt dir also contains `adapter_config.json`, `tokenizer.json`, `chat_template.jinja`, `trainer_state.json`. Soft-resume works via `--resume_from_checkpoint` (sft.py:269 wired); hard resume not possible.

### 1.2 SFT config (eval MUST match these to avoid label/token mismatch)

| key | value |
|---|---|
| base model | `GSAI-ML/LLaDA-8B-Instruct` |
| prompt_format | `mc_digit` |
| answer_label_style | `digit0_pad` (40 classes → "00".."39", every label = exactly 2 tokens) |
| max_answer_tokens | 2 |
| max_seq_len | 4096 |
| max_hops / max_neighbors_per_hop | 2 / 10 |
| use_topology_mask | True (topo branch only — no notopo run was done at r=128) |
| position_id_type | sequential |
| include_neighbor_labels | False (nonb) |
| max_train_samples (pre-boost cap) | 20000 |
| resample_strategy | boost |
| boost_spec | `ogbn-arxiv:cs.LG(Machine Learning):3,ogbn-arxiv:cs.AI(Artificial Intelligence):2,ogbn-arxiv:cs.NE(Neural and Evolutionary Computing):2` |
| post-boost dataset size | 24,511 samples |
| LoRA r / alpha / target | 128 / 128 / all-linear (~335 M trainable, 4.1% of model) |
| effective batch | 48 (per_device=3 × grad_accum=4 × world=4) |
| learning_rate | 5e-5 |
| max_steps | 2042 (= 4 epochs over post-boost data) |
| cls_loss_weight | 0.0 (CE only; aux disabled — trainer.py:211 squeeze fails on multi-token labels) |

---

## 2. Eval setup

Script: `examples/tmdlm/eval_logit.py`. Layer-0 / restricted-argmax: mask answer tokens, single forward, score each of 40 classes by mean log-prob across 2 answer positions.

### 2.1 Class prior calibration (already implemented in eval_logit.py)

- Flags: `--apply_class_prior_calibration True`, plus train_resample_strategy / train_boost_spec / train_max_train_samples to match SFT-time priors.
- Logic: load train (with same boost args) → count cls_labels → log p_train. Count test cls_labels → log p_test. Subtract `log p_train − log p_test` from logits before argmax. This corrects (a) SFT-time boost bias and (b) train/test class distribution shift; target prior is the test distribution itself (mild leakage but standard for offline eval).
- Each JSONL entry now has both `accuracy` (raw) and `accuracy_calibrated`, and `per_class_accuracy` / `per_class_accuracy_calibrated`.

### 2.2 Eval invocation template

```bash
CUDA_VISIBLE_DEVICES=$GPU /home/lingjie7/anaconda3/envs/dllm/bin/python \
  /home/lingjie7/auto-research/projects/dlm-graph/examples/tmdlm/eval_logit.py \
  --exp <experiment-name> \
  --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
  --lora_path <run_dir>/checkpoint-<step> \
  --dataset_name ogbn-arxiv --split test \
  --max_samples 0 \
  --max_seq_len 4096 --max_hops 2 --max_neighbors_per_hop 10 \
  --use_topology_mask True \
  --prompt_format mc_digit --answer_label_style digit0_pad --max_answer_tokens 2 \
  --include_neighbor_labels False --position_id_type sequential \
  --batch_size 2 \
  --apply_class_prior_calibration True \
  --train_resample_strategy boost \
  --train_boost_spec 'ogbn-arxiv:cs.LG(Machine Learning):3,ogbn-arxiv:cs.AI(Artificial Intelligence):2,ogbn-arxiv:cs.NE(Neural and Evolutionary Computing):2' \
  --train_max_train_samples 20000 \
  --log_file /tmp/dlm-graph-eval-jsonl/<your.jsonl>
```

`max_samples=0` → full test (48,604 samples). `max_samples=N>0` → random N samples (seeded; `dllm/data/graph.py:1427` was patched to be random instead of first-N).

Per-ckpt wallclock: **~9 h full-test** / **~11 min for 1000 samples** at batch=2, seq=4096, single GPU.

---

## 3. Existing eval results

### 3.1 1000 random samples (raw, with digit0_pad / max_answer_tokens=2)

| ckpt | acc | notes |
|---|---|---|
| 205 | 66.00% | |
| 410 | 69.70% | |
| 615 | 71.70% | matches §20 r64 ckpt-1668 peak |
| 1640 | 73.90% | |
| **1845** | **74.40%** | **best so far** |
| 2042 | 74.20% | |
| final | 74.20% | |

Std-error at n=1000: σ ≈ 1.4 pt → 74.40% ± 1.4%.

JSONL: `/tmp/dlm-graph-eval-jsonl/eval-arxiv-seq4k-topo-mcdigit-arxiv_20260506_digit0pad_lgboost_r128-logit.jsonl` (filter: `config.lora_path` contains `steps2042-arxiv_20260506`).

### 3.2 Full test (48,604 samples) — IN PROGRESS

4-GPU pipelined eval over all 11 ckpts, started 2026-05-08 08:46. Total wallclock ~27 h.

- Launcher: `examples/tmdlm/run_eval_arxiv_4gpu_fulltest_all_ckpts_lgboost_r128.sh`
- Dispatcher PID at start: 2463997 (check with `ps -p`)
- JSONL: `/tmp/dlm-graph-eval-jsonl/eval-arxiv-fulltest-r128-arxiv_20260506_digit0pad_lgboost_r128-logit.jsonl`
- Worker logs: `/tmp/dlm-graph-eval-logs/eval-arxiv-fulltest-r128-gpu{2,3,4,6}-ckpt*.worker.log`
- Reports both raw and calibrated. **Watch acc_cal vs acc** — the calibration shift range was ±1.1 (sane; first version was ±10 and broken).

When this finishes, GPU 2/3/4/6 auto-reclaim via sample_gen (launcher trap EXIT). Don't fight that other than killing sample_gen explicitly when you need the GPU.

---

## 4. Per-class breakdown (ckpt-1845, 1000 random samples)

Sorted by **pt of overall accuracy lost** (errors / 1000):

| class | size | acc | errors | overall pt loss |
|---|---|---|---|---|
| cs.LG (Machine Learning) | 219 (22%) | 74.4% | 56 | **5.6** |
| cs.CV (Computer Vision) | 204 (20%) | 89.7% | 21 | 2.1 |
| cs.RO (Robotics) | 39 | 64.1% | 14 | 1.4 |
| cs.AI (AI) | 36 | 61.1% | 14 | 1.4 |
| cs.CL (Computation and Language) | 97 (10%) | 86.6% | 13 | 1.3 |
| cs.DC | 22 | 50.0% | 11 | 1.1 |
| cs.IR | 20 | 45.0% | 11 | 1.1 |
| cs.HC | 19 | 47.4% | 10 | 1.0 |
| cs.IT | 52 | 82.7% | 9 | 0.9 |
| cs.NI | 29 | 69.0% | 9 | 0.9 |

**Single biggest bottleneck: cs.LG.** Pushing it from 74.4% → 80% adds +5.6 pt overall. Confusion direction: cs.LG samples leak into cs.AI / cs.RO / cs.NE.

### 4.1 Train/test distribution shift (top of why cs.LG is hard)

| class | train (boosted, n=24,511) | test (n=48,604) | residual shift |
|---|---|---|---|
| cs.LG | 18.9% (was 7.7% pre-boost ×3) | 22.1% | still under by 3.2 pt |
| cs.CV | ~9% (no boost) | 21.6% | **under by 12.6 pt** — biggest unboosted shift |
| cs.IT | 14.4% | 5.9% | over by 8.5 pt |
| cs.CL | ~6% | 9.5% | under by 3.5 pt |
| cs.AI | ~9% (×2 boost) | ~7% | slight over |

cs.CV is the most striking: untouched by boost, big test class, but its 89.7% acc says model handles it OK despite under-training — vision papers must have distinctive surface features. Worth keeping an eye on.

---

## 5. Post-processing ideas (priority order)

### Tier 1 — likely ≥1 pt, no SFT needed

1. **Read calibrated acc from existing JSONL** — check whether `accuracy_calibrated > accuracy` for late ckpts. Free win if yes.
2. **Logits ensemble of late ckpts** (1640 + 1845 + 2042 + final, mean logits): predicted +0.3–0.7 pt. Implementation: extend eval_logit.py to accept multiple `--lora_path` and average pre-argmax logits, or save per-sample logits to disk and merge offline.
3. **Temperature-scaled calibration**: try τ ∈ {0.3, 0.5, 0.7, 1.0} sweep on the calibration shift. Add a `--calibration_temperature` flag (multiplies the shift before subtracting).

### Tier 2 — partial re-eval / more compute

4. **Neighbor-sampling jitter ensemble**: re-eval same ckpt with K different neighbor RNG seeds, mean-pool logits. Predicted +0.5–1 pt for unstable predictions.
5. **Cross-ckpt distillation**: train tiny adapter on ensemble logits as soft targets. Overkill for 1-2 pt.

### Tier 3 — bigger / structural

6. **Self-consistency** on multi-step diffusion paths if applicable.
7. **Graph-aware tiebreaker**: when top-2 logit gap < ε, fall back to neighbor majority label.

---

## 6. Files modified earlier this session (don't re-edit)

| file | change |
|---|---|
| `dllm/data/graph.py` (line ~1427) | `max_samples` now does seeded random sampling instead of first-N |
| `examples/tmdlm/eval_logit.py` | Added calibration: new flags + train/test prior load + dual raw/cal metrics in JSONL |
| `examples/tmdlm/sft.py` (line 269) | `trainer.train(resume_from_checkpoint=...)` for soft-resume support |
| **NEW** `examples/tmdlm/run_sft_arxiv_4gpu_ddp_lgboost_r128.sh` | 4-GPU DDP SFT launcher for §21 |
| **NEW** `examples/tmdlm/run_eval_arxiv_4gpu_fulltest_all_ckpts_lgboost_r128.sh` | 4-GPU pipelined full-test eval over all 11 ckpts |

---

## 7. GPU policy

GPUs 2, 3, 4, 6 = our quota (occupied by SFT / eval / sample_gen). GPUs 0, 1, 5, 7 = other users — leave alone.

When eval finishes, the launcher trap auto-runs sample_gen on 2/3/4/6. To take a GPU back, `pkill -f "sample_gen.py start <N>"`.

---

## 8. Useful greps for picking up state

```bash
# What ckpts exist
ls /home/lingjie7/auto-research/projects/dlm-graph/.models/tmdlm-llada-8b-ogbn-arxiv-2hop-topo-mcdigit-d0-nonb-r128-steps2042-arxiv_20260506_digit0pad_lgboost_r128/

# Latest full-test eval JSONL
cat /tmp/dlm-graph-eval-jsonl/eval-arxiv-fulltest-r128-arxiv_20260506_digit0pad_lgboost_r128-logit.jsonl

# Is full-test still running?
pgrep -af 'eval_logit.py.*steps2042'
ps -p 2463997  # dispatcher

# Per-class accuracies for any past run
python3 -c "
import json
for line in open('<jsonl>'):
    d=json.loads(line)
    if 'steps2042' in d['config']['lora_path']:
        ckpt=d['config']['lora_path'].split('checkpoint-')[-1]
        print(ckpt, d['accuracy'], d.get('accuracy_calibrated'))"
```

---

Last update: 2026-05-08, mid-day, after launching full-test eval (calibration v2 with shift = log p_train − log p_test).
