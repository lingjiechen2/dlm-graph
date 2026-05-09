#!/usr/bin/env bash
set -euo pipefail

# Phase 2: GPU re-eval sweep on ckpt-1845, varying inference-time hyperparams.
# Each setting runs 1000-sample eval (seed=42), dumps logits for offline reuse.
#
# Sweep:
#   s1: max_neighbors_per_hop=5
#   s2: max_neighbors_per_hop=15
#   s3: max_neighbors_per_hop=20
#   s4: max_hops=1
#   s5: max_hops=3
#   s6: use_topology_mask=False
#   s7: prompt_layout=neighbor_first

REPO=/home/lingjie7/auto-research/projects/dlm-graph
RUN_DIR=$REPO/.models/tmdlm-llada-8b-ogbn-arxiv-2hop-topo-mcdigit-d0-nonb-r128-steps2042-arxiv_20260506_digit0pad_lgboost_r128
CKPT_STEP=1845
WORK=$REPO/analysis/postprocess_arxiv_r128
JSONL_DIR=$WORK/eval_jsonl
LOGITS_DIR=$WORK/logits_cache
LOG_DIR=$WORK/logs
JSONL=$JSONL_DIR/phase2_sweep_1845.jsonl
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

# Pre-warm both train cache and a few common test caches
echo "[prewarm] base train cache..."
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
print(f'[prewarm] train ready: {len(ds)} samples')
" 2>&1 | tail -5

# Settings: name, then KV pairs as space-separated 'KEY=VAL' for the eval script
# Defaults baseline: max_neighbors_per_hop=10, max_hops=2, use_topology_mask=True, prompt_layout=target_first
declare -a SETTINGS=(
  "s1_nb5      max_neighbors_per_hop=5"
  "s2_nb15     max_neighbors_per_hop=15"
  "s3_nb20     max_neighbors_per_hop=20"
  "s4_hops1    max_hops=1"
  "s5_hops3    max_hops=3"
  "s6_notopo   use_topology_mask=False"
  "s7_nbfirst  prompt_layout=neighbor_first"
)

ts() { date '+%F %T'; }

launch_setting() {
  local GPU=$1 NAME=$2 OVERRIDES=$3
  local NB=10 HOPS=2 TOPO=True LAYOUT=target_first
  for kv in $OVERRIDES; do
    case $kv in
      max_neighbors_per_hop=*) NB=${kv#*=} ;;
      max_hops=*) HOPS=${kv#*=} ;;
      use_topology_mask=*) TOPO=${kv#*=} ;;
      prompt_layout=*) LAYOUT=${kv#*=} ;;
    esac
  done
  local EXP=phase2-${NAME}-ckpt${CKPT_STEP}
  local CKPT=$RUN_DIR/checkpoint-${CKPT_STEP}
  local NPZ=$LOGITS_DIR/phase2_${NAME}_ckpt${CKPT_STEP}.npz
  local WLOG=$LOG_DIR/phase2-${NAME}-gpu${GPU}.worker.log
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
    --max_hops "$HOPS" \
    --max_neighbors_per_hop "$NB" \
    --use_topology_mask "$TOPO" \
    --prompt_format mc_digit \
    --answer_label_style digit0_pad \
    --max_answer_tokens 2 \
    --include_neighbor_labels False \
    --position_id_type sequential \
    --prompt_layout "$LAYOUT" \
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

# Initial fill
for i in "${!GPUS[@]}"; do
  if [[ ${#QUEUE[@]} -eq 0 ]]; then break; fi
  ENTRY=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
  NAME=${ENTRY%% *}; OVERRIDES=${ENTRY#* }
  G=${GPUS[$i]}
  PID=$(launch_setting $G $NAME "$OVERRIDES")
  GPU_PID[$G]=$PID
  GPU_NAME[$G]=$NAME
  echo "[$(ts)] [start] GPU $G -> $NAME (pid $PID)"
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
        NAME=${ENTRY%% *}; OVERRIDES=${ENTRY#* }
        NEW_PID=$(launch_setting $G $NAME "$OVERRIDES")
        GPU_PID[$G]=$NEW_PID
        GPU_NAME[$G]=$NAME
        echo "[$(ts)] [start] GPU $G -> $NAME (pid $NEW_PID)"
      fi
    fi
  done
done

echo "[$(ts)] [all-done]"
