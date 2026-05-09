#!/usr/bin/env bash
set -euo pipefail

# Phase 2d: neighbor-sampling jitter ensemble on ckpt-1845.
# Same 1000 nodes (seed=42) but with different neighbor RNG seeds → different
# specific 1-hop/2-hop neighbor selections per node. Tests TTA via neighbor jitter.
# Requires the new `--neighbor_seed` flag and graph.py patch.

REPO=/home/lingjie7/auto-research/projects/dlm-graph
RUN_DIR=$REPO/.models/tmdlm-llada-8b-ogbn-arxiv-2hop-topo-mcdigit-d0-nonb-r128-steps2042-arxiv_20260506_digit0pad_lgboost_r128
CKPT_STEP=1845
WORK=$REPO/analysis/postprocess_arxiv_r128
JSONL_DIR=$WORK/eval_jsonl
LOGITS_DIR=$WORK/logits_cache
LOG_DIR=$WORK/logs
JSONL=$JSONL_DIR/phase2d_neighbor_jitter.jsonl
PY=/home/lingjie7/anaconda3/envs/dllm/bin/python
EVAL=$REPO/examples/tmdlm/eval_logit.py
SAMPLE_GEN=/home/lingjie7/sample_gen.py

GPUS=(2 3 4 6)
BASE_MODEL=GSAI-ML/LLaDA-8B-Instruct
BOOST_SPEC='ogbn-arxiv:cs.LG(Machine Learning):3,ogbn-arxiv:cs.AI(Artificial Intelligence):2,ogbn-arxiv:cs.NE(Neural and Evolutionary Computing):2'
TRAIN_RESAMPLE=boost
TRAIN_MAX_SAMPLES=20000

mkdir -p $JSONL_DIR $LOGITS_DIR $LOG_DIR
export HF_HOME=/tmp/dlm-graph-hf-home
export TRANSFORMERS_CACHE=/tmp/dlm-graph-hf-transformers
export PYTHONPATH=$REPO

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

# 8 jitters: 4 at nb=10 (SFT-trained value), 4 at nb=15 (best Phase 2 setting)
declare -a SETTINGS=(
  "nb10_jit7   10  7"
  "nb10_jit13  10 13"
  "nb10_jit23  10 23"
  "nb10_jit31  10 31"
  "nb15_jit7   15  7"
  "nb15_jit13  15 13"
  "nb15_jit23  15 23"
  "nb15_jit31  15 31"
)

ts() { date '+%F %T'; }

launch_setting() {
  local GPU=$1 NAME=$2 NB=$3 NSEED=$4
  local EXP=phase2d-${NAME}-ckpt${CKPT_STEP}
  local CKPT=$RUN_DIR/checkpoint-${CKPT_STEP}
  local NPZ=$LOGITS_DIR/phase2d_${NAME}_ckpt${CKPT_STEP}.npz
  local WLOG=$LOG_DIR/phase2d-${NAME}-gpu${GPU}.worker.log
  : > $WLOG
  CUDA_VISIBLE_DEVICES=$GPU nohup $PY $EVAL \
    --exp "$EXP" \
    --model_name_or_path "$BASE_MODEL" \
    --lora_path "$CKPT" \
    --dataset_name ogbn-arxiv \
    --split test \
    --max_samples 1000 \
    --seed 42 \
    --neighbor_seed "$NSEED" \
    --max_seq_len 4096 \
    --max_hops 2 \
    --max_neighbors_per_hop "$NB" \
    --use_topology_mask True \
    --prompt_format mc_digit \
    --answer_label_style digit0_pad \
    --max_answer_tokens 2 \
    --include_neighbor_labels False \
    --position_id_type sequential \
    --prompt_layout target_first \
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
declare -A GPU_NAME
declare -a QUEUE=()
for s in "${SETTINGS[@]}"; do QUEUE+=("$s"); done

for i in "${!GPUS[@]}"; do
  if [[ ${#QUEUE[@]} -eq 0 ]]; then break; fi
  ENTRY=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
  read -r NAME NB NSEED <<< "$ENTRY"
  G=${GPUS[$i]}
  PID=$(launch_setting $G $NAME $NB $NSEED)
  GPU_PID[$G]=$PID
  GPU_NAME[$G]=$NAME
  echo "[$(ts)] [start] GPU $G -> $NAME nb=$NB seed=$NSEED (pid $PID)"
done

while [[ ${#GPU_PID[@]} -gt 0 ]]; do
  sleep 30
  for G in "${!GPU_PID[@]}"; do
    PID=${GPU_PID[$G]}
    if ! kill -0 $PID 2>/dev/null; then
      echo "[$(ts)] [done] GPU $G ${GPU_NAME[$G]} (pid $PID)"
      unset GPU_PID[$G]
      unset GPU_NAME[$G]
      if [[ ${#QUEUE[@]} -gt 0 ]]; then
        ENTRY=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
        read -r NAME NB NSEED <<< "$ENTRY"
        NEW_PID=$(launch_setting $G $NAME $NB $NSEED)
        GPU_PID[$G]=$NEW_PID
        GPU_NAME[$G]=$NAME
        echo "[$(ts)] [start] GPU $G -> $NAME nb=$NB seed=$NSEED (pid $NEW_PID)"
      fi
    fi
  done
done

echo "[$(ts)] [all-done]"
