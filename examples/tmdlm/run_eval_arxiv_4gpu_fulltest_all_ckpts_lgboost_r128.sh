#!/usr/bin/env bash
set -euo pipefail

# 4-GPU pipelined full-test eval across all 11 ckpts of §21 r128 SFT
# (run tag arxiv_20260506_digit0pad_lgboost_r128). Reports raw accuracy AND
# class-prior-calibrated accuracy (calibration matches SFT-time boost).
#
# Usage: bash examples/tmdlm/run_eval_arxiv_4gpu_fulltest_all_ckpts_lgboost_r128.sh

REPO=/home/lingjie7/auto-research/projects/dlm-graph
RUN_DIR=$REPO/.models/tmdlm-llada-8b-ogbn-arxiv-2hop-topo-mcdigit-d0-nonb-r128-steps2042-arxiv_20260506_digit0pad_lgboost_r128
JSONL_DIR=/tmp/dlm-graph-eval-jsonl
LOG_DIR=/tmp/dlm-graph-eval-logs
JSONL=$JSONL_DIR/eval-arxiv-fulltest-r128-arxiv_20260506_digit0pad_lgboost_r128-logit.jsonl
PY=/home/lingjie7/anaconda3/envs/dllm/bin/python
EVAL=$REPO/examples/tmdlm/eval_logit.py
SAMPLE_GEN=/home/lingjie7/sample_gen.py

GPUS=(2 3 4 6)
CKPTS=(205 410 615 820 1025 1230 1435 1640 1845 2042 final)

BASE_MODEL=GSAI-ML/LLaDA-8B-Instruct
BOOST_SPEC='ogbn-arxiv:cs.LG(Machine Learning):3,ogbn-arxiv:cs.AI(Artificial Intelligence):2,ogbn-arxiv:cs.NE(Neural and Evolutionary Computing):2'
TRAIN_RESAMPLE=boost
TRAIN_MAX_SAMPLES=20000

mkdir -p $JSONL_DIR $LOG_DIR
export HF_HOME=/tmp/dlm-graph-hf-home
export TRANSFORMERS_CACHE=/tmp/dlm-graph-hf-transformers
export PYTHONPATH=$REPO

reclaim_gpus() {
  echo "[trap] launching sample_gen on GPUs ${GPUS[*]}" >&2
  for g in "${GPUS[@]}"; do
    nohup $PY $SAMPLE_GEN start $g >> $LOG_DIR/sample_gen_gpu${g}.log 2>&1 &
    disown || true
  done
}
trap reclaim_gpus EXIT

cd $REPO

# Pre-warm the train-split cache (calibration uses split=train + boost spec).
# Single-process build avoids 4 parallel jobs racing to build the same cache.
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

# Pipelined dispatch: launch initial 4, replace as GPUs free.
declare -A GPU_PID
declare -A GPU_CKPT
declare -a QUEUE=("${CKPTS[@]}")

ts() { date '+%F %T'; }

launch_ckpt() {
  local GPU=$1 STEP=$2
  local EXP=eval-arxiv-fulltest-mcdigit-arxiv_20260506_digit0pad_lgboost_r128-checkpoint-${STEP}
  local CKPT=$RUN_DIR/checkpoint-${STEP}
  local WLOG=$LOG_DIR/eval-arxiv-fulltest-r128-gpu${GPU}-ckpt${STEP}.worker.log
  : > $WLOG
  CUDA_VISIBLE_DEVICES=$GPU nohup $PY $EVAL \
    --exp "$EXP" \
    --model_name_or_path "$BASE_MODEL" \
    --lora_path "$CKPT" \
    --dataset_name ogbn-arxiv \
    --split test \
    --max_samples 0 \
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
    --log_file "$JSONL" \
    >> $WLOG 2>&1 &
  echo $!
}

# Initial fill: one ckpt per GPU
for i in "${!GPUS[@]}"; do
  if [[ ${#QUEUE[@]} -eq 0 ]]; then break; fi
  STEP=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
  G=${GPUS[$i]}
  PID=$(launch_ckpt $G $STEP)
  GPU_PID[$G]=$PID
  GPU_CKPT[$G]=$STEP
  echo "[$(ts)] [start] GPU $G -> ckpt $STEP (pid $PID)"
done

# Pipeline loop: poll every 60s, launch next ckpt on freed GPU
while [[ ${#GPU_PID[@]} -gt 0 ]]; do
  sleep 60
  for G in "${!GPU_PID[@]}"; do
    PID=${GPU_PID[$G]}
    if ! kill -0 $PID 2>/dev/null; then
      echo "[$(ts)] [done]  GPU $G ckpt ${GPU_CKPT[$G]} (pid $PID)"
      unset GPU_PID[$G]
      unset GPU_CKPT[$G]
      if [[ ${#QUEUE[@]} -gt 0 ]]; then
        STEP=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
        NEW_PID=$(launch_ckpt $G $STEP)
        GPU_PID[$G]=$NEW_PID
        GPU_CKPT[$G]=$STEP
        echo "[$(ts)] [start] GPU $G -> ckpt $STEP (pid $NEW_PID)"
      fi
    fi
  done
done

echo "[$(ts)] [all-done]"
