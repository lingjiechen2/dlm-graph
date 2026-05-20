# LP Migration TODO (as of 2026-05-20)

## Current state

### Cora LP — complete (§25, §26)
- **§25** `cora_lp_20260519_seq4k_5ep_3gpu` (topo, posw=1): best acc **91.47%** @ ckpt-748, AUC 0.9674 @ ckpt-1309. −1.2 pt vs LLaGA-ND-7B (92.71%).
- **§26** `cora_lp_20260520_seq4k_5ep_posw2_topo` (topo, posw=2): best acc **90.88%** @ ckpt-1496. posw=2 does not help — posw=1 is the better default.
- **Split mismatch (critical)**: Our SFT used our own 85/5/10 split; 80% of LLaGA test positive edges were in our training set. True head-to-head with LLaGA requires retraining on LLaGA's split.
- **LLaGA leakage note**: 48.2% of LLaGA test positive pairs include the target endpoint in the prompt during LLaGA inference (test edges not removed before neighbor sampling). Our eval is clean; the true gap to LLaGA is smaller than 1.2 pt.

### Uncommitted code changes (commit before migrating)
- `dllm/pipelines/tmdlm/trainer.py` — adds `lp_pos_weight` field to `TMDLMConfig` + applies per-sample loss weighting in `TMDLMTrainer`.
- `examples/tmdlm/run_sft_cora_lp_4gpu_ddp.sh` — exposes `LP_POS_WEIGHT` env var (default 1.0).

---

## Pending tasks (priority order)

### 1. Retrain Cora LP on LLaGA's split (highest priority)
- Use `edge_sampled_2_10_only_train.jsonl` as training pairs instead of our seed-42 random split.
- Goal: remove the 80% train-test overlap so the LLaGA test-split comparison is fair.
- Same recipe as §25: `topo=True`, `posw=1`, `seq=4096`, `hop=2`, `nb=10`, 5 epochs, 3×GPU DDP.
- Script: `examples/tmdlm/run_sft_cora_lp_4gpu_ddp.sh` with `GPUS=2,4,6`.

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
| LP eval script            | `examples/tmdlm/eval_lp_llaga_split.py` |
| LP SFT launch script      | `examples/tmdlm/run_sft_cora_lp_4gpu_ddp.sh` |
| LP ablation script        | `examples/tmdlm/eval_lp_2hop_ablation.py` |
| Results log               | `examples/tmdlm/results.md` (§25, §26) |
