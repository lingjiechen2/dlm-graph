#!/usr/bin/env bash
set -euo pipefail

# Phase 8: full-test (48,604 samples) eval of the 3 unique top ckpts
# (1640, 1845, 2042; final == 2042 confirmed by md5).
# 1 ckpt per GPU on 2/3/4. GPU 6 stays held by sample_gen throughout.
# ETA ~9.5-10 h wallclock (each ckpt independently runs to completion).

REPO=/home/lingjie7/auto-research/projects/dlm-graph
RUN_DIR=$REPO/.models/tmdlm-llada-8b-ogbn-arxiv-2hop-topo-mcdigit-d0-nonb-r128-steps2042-arxiv_20260506_digit0pad_lgboost_r128
WORK=$REPO/analysis/postprocess_arxiv_r128
JSONL_DIR=$WORK/eval_jsonl
LOGITS_DIR=$WORK/logits_cache
LOG_DIR=$WORK/logs
JSONL=$JSONL_DIR/phase8_fulltest_top3.jsonl
PY=/home/lingjie7/anaconda3/envs/dllm/bin/python
EVAL=$REPO/examples/tmdlm/eval_logit.py
SAMPLE_GEN=/home/lingjie7/sample_gen.py

# Use only 3 of our 4 GPUs; GPU 6 stays held by sample_gen the whole time.
EVAL_GPUS=(2 3 4)
HOLD_GPUS=(2 3 4 6)   # all 4 will be re-held on exit
BASE_MODEL=GSAI-ML/LLaDA-8B-Instruct
BOOST_SPEC='ogbn-arxiv:cs.LG(Machine Learning):3,ogbn-arxiv:cs.AI(Artificial Intelligence):2,ogbn-arxiv:cs.NE(Neural and Evolutionary Computing):2'
TRAIN_RESAMPLE=boost
TRAIN_MAX_SAMPLES=20000

mkdir -p $JSONL_DIR $LOGITS_DIR $LOG_DIR
export HF_HOME=/tmp/dlm-graph-hf-home
export TRANSFORMERS_CACHE=/tmp/dlm-graph-hf-transformers
export PYTHONPATH=$REPO

# Release sample_gen on the 3 EVAL GPUs only (keep GPU 6 held)
echo "[setup] releasing sample_gen on EVAL GPUs ${EVAL_GPUS[*]}; GPU 6 stays held"
for g in "${EVAL_GPUS[@]}"; do
  pkill -f "sample_gen.py start $g" 2>/dev/null || true
done
sleep 3

reclaim_gpus() {
  echo "[trap] launching sample_gen on GPUs ${HOLD_GPUS[*]}" >&2
  for g in "${HOLD_GPUS[@]}"; do
    nohup $PY $SAMPLE_GEN start $g >> $LOG_DIR/sample_gen_gpu${g}.log 2>&1 &
    disown || true
  done
}
trap reclaim_gpus EXIT

cd $REPO

# Pre-warm train cache (single process, avoids 3 parallel jobs racing)
echo "[prewarm] train cache for class-prior calibration..."
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

# Pre-warm full test cache (one process, nb=10)
echo "[prewarm] full test cache (nb=10, max_samples=0) ..."
$PY -c "
from dllm.data.graph import load_tag_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${BASE_MODEL}')
ds = load_tag_dataset(
    'ogbn-arxiv', tokenizer=tok, split='test',
    max_seq_len=4096, max_neighbors_per_hop=10, max_hops=2, seed=42,
    max_answer_tokens=2, prompt_layout='target_first', use_chat_template=False,
    prompt_format='mc_digit', answer_label_style='digit0_pad',
    include_neighbor_labels=False, neighbor_label_format='bracket',
    max_samples=0,
)
print(f'[prewarm] full test ready: {len(ds)} samples')
" 2>&1 | tail -3

# 3 ckpts mapped 1:1 to 3 GPUs
declare -a CKPTS=(1640 1845 2042)

ts() { date '+%F %T'; }

launch_ckpt() {
  local GPU=$1 STEP=$2
  local EXP=phase8-fulltest-ckpt${STEP}-nb10
  local CKPT=$RUN_DIR/checkpoint-${STEP}
  local NPZ=$LOGITS_DIR/phase8_fulltest_ckpt${STEP}_nb10.npz
  local WLOG=$LOG_DIR/phase8-fulltest-ckpt${STEP}-gpu${GPU}.worker.log
  : > $WLOG
  CUDA_VISIBLE_DEVICES=$GPU nohup $PY $EVAL \
    --exp "$EXP" \
    --model_name_or_path "$BASE_MODEL" \
    --lora_path "$CKPT" \
    --dataset_name ogbn-arxiv \
    --split test \
    --max_samples 0 \
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
for i in "${!EVAL_GPUS[@]}"; do
  if [[ $i -ge ${#CKPTS[@]} ]]; then break; fi
  G=${EVAL_GPUS[$i]}
  STEP=${CKPTS[$i]}
  PID=$(launch_ckpt $G $STEP)
  GPU_PID[$G]=$PID
  echo "[$(ts)] [start] GPU $G -> ckpt-$STEP (pid $PID)"
done

while [[ ${#GPU_PID[@]} -gt 0 ]]; do
  sleep 300  # poll every 5 min for a 10h job
  for G in "${!GPU_PID[@]}"; do
    PID=${GPU_PID[$G]}
    if ! kill -0 $PID 2>/dev/null; then
      echo "[$(ts)] [done] GPU $G (pid $PID)"
      unset GPU_PID[$G]
    fi
  done
done

echo "[$(ts)] [all-done]"
