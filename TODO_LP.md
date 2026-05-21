# LP Migration TODO (as of 2026-05-21)

## Current state

### Cora LP — complete (§25–§28)

LLaGA-ND-7B Cora LP best = **89.41%** (Task Expert, verified from arXiv:2402.08170 Table 1).

- **§25** `cora_lp_20260519_seq4k_5ep_3gpu` (topo, posw=1, seed-42 split): best acc **91.47%** @ ckpt-748, AUC 0.9674 @ ckpt-1309. **+2.06 pt** vs LLaGA-ND. Note: 80% of LLaGA test positives in training — split mismatch.
- **§26** `cora_lp_20260520_seq4k_5ep_posw2_topo` (topo, posw=2, seed-42 split): best acc **90.88%** @ ckpt-1496. posw=2 does not help.
- **§27** `cora_lp_20260520_seq4k_3ep_hardneg05` (topo, posw=1, hard_neg_ratio=0.5, 3ep): best acc **88.53%** @ ckpt-339. Hard negatives hurt — model collapses to predicting "no" at later checkpoints.
- **§28** `cora_lp_20260521_llaga_split_5ep` (topo, posw=1, **LLaGA official train/test split**, 5ep): best acc **91.62%** @ ckpt-136, AUC 0.9633 @ ckpt-238. **+2.21 pt** vs LLaGA-ND. **Clean fair comparison — this is the definitive Cora LP result.**
- **LLaGA leakage note**: 48.2% of LLaGA test positive pairs include the target endpoint in the prompt during LLaGA inference. Our eval is clean.

### Uncommitted code changes (commit before migrating)
- `dllm/pipelines/tmdlm/trainer.py` — adds `lp_pos_weight` field to `TMDLMConfig` + applies per-sample loss weighting in `TMDLMTrainer`.
- `examples/tmdlm/run_sft_cora_lp_4gpu_ddp.sh` — exposes `LP_POS_WEIGHT` env var (default 1.0).
- `dllm/data/datasets/_lp_common.py` — adds `load_lp_llaga_split()` for official JSONL splits.
- `dllm/data/datasets/cora_lp.py` — adds `use_llaga_split` parameter.
- `examples/tmdlm/run_sft_cora_lp_llaga_split.sh` — §28 training script.

---

## Pending tasks (priority order)

### 1. ~~Retrain Cora LP on LLaGA's split~~ ✓ Done (§28)
- Completed 2026-05-21. Best acc 91.62% @ ckpt-136, surpasses all LLaGA Cora LP baselines.

### 2. PubMed LP eval (was started, not completed)
- Eval the existing PubMed NC SFT checkpoints (`pubmed_20260428_aligned`, ckpts 370/740/1110/1480) on the LP task.
- 4 settings: logit × {topo, notopo}, infill × {topo, notopo}.
- GPU7 was running logit-topo ckpt-370 when the session ended; results not saved to results.md.
- Script: adapt `eval_lp_llaga_split.py` or the existing `eval_logit.py` with `task=lp`.
- Note: PubMed LP SFT checkpoints do not yet exist — this eval is zero-shot transfer from NC SFT.

### 3. Cora LP notopo run
- §25 and §26 both train with `topo=True`. A notopo control is missing.
- Needed to confirm whether topo helps or hurts LP (analogous to the NC topo/notopo analysis in §18).

### 4. PubMed LP SFT (lower priority)
- Full LP fine-tuning on PubMed edge pairs (analogous to §25 for Cora).
- Requires PubMed LP split file; check if `edge_sampled_2_10_only_{train,test}.jsonl` exists for PubMed under `.datasets/llaga/pubmed/`.

---

## Key file locations
| Item | Path |
| --- | --- |
| LLaGA Cora LP train pairs | `.datasets/llaga/cora/edge_sampled_2_10_only_train.jsonl` |
| LLaGA Cora LP test pairs  | `.datasets/llaga/cora/edge_sampled_2_10_only_test.jsonl` |
| Cora adj (no test edges)  | `.datasets/llaga/cora/processed_data_link_notest.pt` |
| §25 ckpts                 | `.models/tmdlm-llada-8b-cora-lp-2hop-r64-ep5-cora_lp_20260519_seq4k_5ep_3gpu/` |
| §26 ckpts                 | `.models/tmdlm-llada-8b-cora-lp-2hop-r64-ep5-cora_lp_20260520_seq4k_5ep_posw2_topo/` |
| §27 ckpts                 | `.models/tmdlm-llada-8b-cora-lp-2hop-r64-ep3-cora_lp_20260520_seq4k_3ep_hardneg05/` |
| §28 ckpts                 | `.models/tmdlm-llada-8b-cora-lp-2hop-r64-ep5-cora_lp_20260521_llaga_split_5ep/` |
| LP eval script            | `examples/tmdlm/eval_lp_llaga_split.py` |
| LP SFT launch script      | `examples/tmdlm/run_sft_cora_lp_4gpu_ddp.sh` |
| §28 launch script         | `examples/tmdlm/run_sft_cora_lp_llaga_split.sh` |
| LP ablation script        | `examples/tmdlm/eval_lp_2hop_ablation.py` |
| Results log               | `examples/tmdlm/results.md` (§25–§28) |
