#!/usr/bin/env bash
set -euo pipefail

# Phase 6: replicate Phase 2c at N=5000 to tighten variance.
# 4 ckpts (1640, 1845, 2042, final) × 4 nb values (10, 12, 15, 30) = 16 runs.
# Each ~55-60 min. 4-GPU pipeline → ~4-4.5 hours total.

REPO=/home/lingjie7/auto-research/projects/dlm-graph
RUN_DIR=$REPO/.models/tmdlm-llada-8b-ogbn-arxiv-2hop-topo-mcdigit-d0-nonb-r128-steps2042-arxiv_20260506_digit0pad_lgboost_r128
WORK=$REPO/analysis/postprocess_arxiv_r128
JSONL_DIR=$WORK/eval_jsonl
LOGITS_DIR=$WORK/logits_cache
LOG_DIR=$WORK/logs
JSONL=$JSONL_DIR/phase6_5k_nb_sweep.jsonl
PY=/home/lingjie7/anaconda3/envs/dllm/bin/python
EVAL=$REPO/examples/tmdlm/eval_logit.py
SAMPLE_GEN=/home/lingjie7/sample_gen.py

GPUS=(2 3 4 6)
BASE_MODEL=GSAI-ML/LLaDA-8B-Instruct
BOOST_SPEC='ogbn-arxiv:cs.LG(Machine Learning):3,ogbn-arxiv:cs.AI(Artificial Intelligence):2,ogbn-arxiv:cs.NE(Neural and Evolutionary Computing):2'
TRAIN_RESAMPLE=boost
TRAIN_MAX_SAMPLES=20000
N_SAMPLES=5000

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

# Pre-warm caches: train (calibration) + 4 test variants (nb=10, 12, 15, 30 at N=5000)
echo "[prewarm] train cache..."
$PY -c "
from dllm.data.graph import load_tag_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${BASE_MODEL}')
load_tag_dataset(
    'ogbn-arxiv', tokenizer=tok, split='train',
    max_seq_len=4096, max_neighbors_per_hop=10, max_hops=2, seed=42,
    max_answer_tokens=2, prompt_layout='target_first', use_chat_template=False,
    prompt_format='mc_digit', answer_label_style='digit0_pad',
    include_neighbor_labels=False, neighbor_label_format='bracket',
    max_samples=${TRAIN_MAX_SAMPLES}, resample_strategy='${TRAIN_RESAMPLE}',
    boost_spec='${BOOST_SPEC}',
)
print('[prewarm] train cache ready')
" 2>&1 | tail -3

# Pre-build all 4 test caches before launching (avoids 4 parallel jobs racing to build same cache for nb=10)
for NB in 10 12 15 30; do
  echo "[prewarm] test cache nb=${NB} N=${N_SAMPLES} ..."
  $PY -c "
from dllm.data.graph import load_tag_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${BASE_MODEL}')
load_tag_dataset(
    'ogbn-arxiv', tokenizer=tok, split='test',
    max_seq_len=4096, max_neighbors_per_hop=${NB}, max_hops=2, seed=42,
    max_answer_tokens=2, prompt_layout='target_first', use_chat_template=False,
    prompt_format='mc_digit', answer_label_style='digit0_pad',
    include_neighbor_labels=False, neighbor_label_format='bracket',
    max_samples=${N_SAMPLES},
)
print(f'[prewarm] test nb=${NB} ready')
" 2>&1 | tail -3
done

# 16 runs queue: ckpt step + nb
declare -a SETTINGS=(
  "1640_nb10  1640 10"
  "1640_nb12  1640 12"
  "1640_nb15  1640 15"
  "1640_nb30  1640 30"
  "1845_nb10  1845 10"
  "1845_nb12  1845 12"
  "1845_nb15  1845 15"
  "1845_nb30  1845 30"
  "2042_nb10  2042 10"
  "2042_nb12  2042 12"
  "2042_nb15  2042 15"
  "2042_nb30  2042 30"
  "final_nb10 final 10"
  "final_nb12 final 12"
  "final_nb15 final 15"
  "final_nb30 final 30"
)

ts() { date '+%F %T'; }

launch_setting() {
  local GPU=$1 NAME=$2 STEP=$3 NB=$4
  local EXP=phase6-${NAME}
  local CKPT=$RUN_DIR/checkpoint-${STEP}
  local NPZ=$LOGITS_DIR/phase6_${NAME}.npz
  local WLOG=$LOG_DIR/phase6-${NAME}-gpu${GPU}.worker.log
  : > $WLOG
  CUDA_VISIBLE_DEVICES=$GPU nohup $PY $EVAL \
    --exp "$EXP" \
    --model_name_or_path "$BASE_MODEL" \
    --lora_path "$CKPT" \
    --dataset_name ogbn-arxiv \
    --split test \
    --max_samples "$N_SAMPLES" \
    --seed 42 \
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

# Initial fill
for i in "${!GPUS[@]}"; do
  if [[ ${#QUEUE[@]} -eq 0 ]]; then break; fi
  ENTRY=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
  read -r NAME STEP NB <<< "$ENTRY"
  G=${GPUS[$i]}
  PID=$(launch_setting $G $NAME $STEP $NB)
  GPU_PID[$G]=$PID
  GPU_NAME[$G]=$NAME
  echo "[$(ts)] [start] GPU $G -> $NAME (pid $PID)"
done

while [[ ${#GPU_PID[@]} -gt 0 ]]; do
  sleep 60
  for G in "${!GPU_PID[@]}"; do
    PID=${GPU_PID[$G]}
    if ! kill -0 $PID 2>/dev/null; then
      echo "[$(ts)] [done] GPU $G ${GPU_NAME[$G]} (pid $PID)"
      unset GPU_PID[$G]
      unset GPU_NAME[$G]
      if [[ ${#QUEUE[@]} -gt 0 ]]; then
        ENTRY=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
        read -r NAME STEP NB <<< "$ENTRY"
        NEW_PID=$(launch_setting $G $NAME $STEP $NB)
        GPU_PID[$G]=$NEW_PID
        GPU_NAME[$G]=$NAME
        echo "[$(ts)] [start] GPU $G -> $NAME (pid $NEW_PID)"
      fi
    fi
  done
done

echo "[$(ts)] [all-done]"
