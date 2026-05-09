#!/usr/bin/env bash
set -euo pipefail

# 4-GPU 1:1 eval on top-4 ckpts of §21 r128 SFT, 1000-sample with seed=42.
# Dumps per-sample 40-class scores + per-position log-probs to .npz cache for
# offline post-processing (Phase 1 of arxiv r128 post-process exploration).
#
# Usage: bash examples/tmdlm/run_eval_arxiv_4gpu_1000sample_dump_logits.sh

REPO=/home/lingjie7/auto-research/projects/dlm-graph
RUN_DIR=$REPO/.models/tmdlm-llada-8b-ogbn-arxiv-2hop-topo-mcdigit-d0-nonb-r128-steps2042-arxiv_20260506_digit0pad_lgboost_r128
WORK=$REPO/analysis/postprocess_arxiv_r128
JSONL_DIR=$WORK/eval_jsonl
LOGITS_DIR=$WORK/logits_cache
LOG_DIR=$WORK/logs
JSONL=$JSONL_DIR/baseline_1000_calibrated.jsonl
PY=/home/lingjie7/anaconda3/envs/dllm/bin/python
EVAL=$REPO/examples/tmdlm/eval_logit.py
SAMPLE_GEN=/home/lingjie7/sample_gen.py

GPUS=(2 3 4 6)
CKPTS=(1640 1845 2042 final)

BASE_MODEL=GSAI-ML/LLaDA-8B-Instruct
BOOST_SPEC='ogbn-arxiv:cs.LG(Machine Learning):3,ogbn-arxiv:cs.AI(Artificial Intelligence):2,ogbn-arxiv:cs.NE(Neural and Evolutionary Computing):2'
TRAIN_RESAMPLE=boost
TRAIN_MAX_SAMPLES=20000

mkdir -p $JSONL_DIR $LOGITS_DIR $LOG_DIR
export HF_HOME=/tmp/dlm-graph-hf-home
export TRANSFORMERS_CACHE=/tmp/dlm-graph-hf-transformers
export PYTHONPATH=$REPO

# Release sample_gen on our 4 GPUs before launching evals
echo "[setup] releasing sample_gen on GPUs ${GPUS[*]}"
for g in "${GPUS[@]}"; do
  pkill -f "sample_gen.py start $g" 2>/dev/null || true
done
sleep 3

reclaim_gpus() {
  echo "[trap] launching sample_gen on GPUs ${GPUS[*]}" >&2
  for g in "${GPUS[@]}"; do
    nohup $PY $SAMPLE_GEN start $g >> $LOG_DIR/sample_gen_gpu${g}.log 2>&1 &
    disown || true
  done
}
trap reclaim_gpus EXIT

cd $REPO

# Pre-warm train cache (single process)
echo "[prewarm] building/loading train cache for class-prior calibration..."
$PY -c "
from dllm.data.graph import load_tag_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${BASE_MODEL}')
ds = load_tag_dataset(
    'ogbn-arxiv', tokenizer=tok, split='train',
    max_seq_len=4096, max_neighbors_per_hop=10, max_hops=2, seed=42,
    max_answer_tokens=2, prompt_layout='target_first', use_chat_template=False,
    prompt_format='mc_digit', answer_label_style='digit0_pad',
    include_neighbor_labels=False, neighbor_label_format='bracket',
    max_samples=${TRAIN_MAX_SAMPLES}, resample_strategy='${TRAIN_RESAMPLE}',
    boost_spec='${BOOST_SPEC}',
)
print(f'[prewarm] train dataset ready: {len(ds)} samples')
" 2>&1 | tail -10

ts() { date '+%F %T'; }

launch_ckpt() {
  local GPU=$1 STEP=$2
  local EXP=baseline-1000-mcdigit-r128-checkpoint-${STEP}
  local CKPT=$RUN_DIR/checkpoint-${STEP}
  local NPZ=$LOGITS_DIR/ckpt-${STEP}.npz
  local WLOG=$LOG_DIR/baseline-1000-gpu${GPU}-ckpt${STEP}.worker.log
  : > $WLOG
  CUDA_VISIBLE_DEVICES=$GPU nohup $PY $EVAL \
    --exp "$EXP" \
    --model_name_or_path "$BASE_MODEL" \
    --lora_path "$CKPT" \
    --dataset_name ogbn-arxiv \
    --split test \
    --max_samples 1000 \
    --seed 42 \
    --max_seq_len 4096 \
    --max_hops 2 \
    --max_neighbors_per_hop 10 \
    --use_topology_mask True \
    --prompt_format mc_digit \
    --answer_label_style digit0_pad \
    --max_answer_tokens 2 \
    --include_neighbor_labels False \
    --position_id_type sequential \
    --batch_size 2 \
    --apply_class_prior_calibration True \
    --train_resample_strategy "$TRAIN_RESAMPLE" \
    --train_boost_spec "$BOOST_SPEC" \
    --train_max_train_samples "$TRAIN_MAX_SAMPLES" \
    --dump_logits_path "$NPZ" \
    --dump_per_position True \
    --log_file "$JSONL" \
    >> $WLOG 2>&1 &
  echo $!
}

declare -A GPU_PID
for i in "${!GPUS[@]}"; do
  G=${GPUS[$i]}
  STEP=${CKPTS[$i]}
  PID=$(launch_ckpt $G $STEP)
  GPU_PID[$G]=$PID
  echo "[$(ts)] [start] GPU $G -> ckpt $STEP (pid $PID)"
done

# Wait for all to finish
while [[ ${#GPU_PID[@]} -gt 0 ]]; do
  sleep 30
  for G in "${!GPU_PID[@]}"; do
    PID=${GPU_PID[$G]}
    if ! kill -0 $PID 2>/dev/null; then
      echo "[$(ts)] [done] GPU $G (pid $PID)"
      unset GPU_PID[$G]
    fi
  done
done

echo "[$(ts)] [all-done]"
