#!/usr/bin/env bash
set -euo pipefail

# §23 3ep fulltrain — full-test (N=48,603) eval on all 10 ckpts, newest-first.
# 4-GPU pipelined on quota {2,3,4,6}. trap-EXIT holds GPUs with sample_gen.

REPO=/home/lingjie7/auto-research/projects/dlm-graph
RUN_DIR=$REPO/.models/tmdlm-llada-8b-ogbn-arxiv-2hop-topo-mcdigit-d0-nonb-r128-steps7188-arxiv_20260514_fulltrain_r128_3ep
WORK=$REPO/analysis/postprocess_arxiv_r128
JSONL_DIR=$WORK/eval_jsonl
LOG_DIR=$WORK/logs
JSONL=$JSONL_DIR/sec23_fulltest_all_ckpts.jsonl
PY=/home/lingjie7/anaconda3/envs/dllm/bin/python
EVAL=$REPO/examples/tmdlm/eval_logit.py
SAMPLE_GEN=/home/lingjie7/sample_gen.py

GPUS=(2 3 4 6)
# newest-first
CKPTS=(final 6471 5752 5033 4314 3595 2876 2157 1438 719)

BASE_MODEL=GSAI-ML/LLaDA-8B-Instruct
BOOST_SPEC='ogbn-arxiv:cs.LG(Machine Learning):3,ogbn-arxiv:cs.AI(Artificial Intelligence):2,ogbn-arxiv:cs.NE(Neural and Evolutionary Computing):2'
TRAIN_RESAMPLE=boost
TRAIN_MAX_SAMPLES=0   # full train, matches §23 SFT

mkdir -p $JSONL_DIR $LOG_DIR
export HF_HOME=/tmp/dlm-graph-hf-home
export TRANSFORMERS_CACHE=/tmp/dlm-graph-hf-transformers
export PYTHONPATH=$REPO

# Release sample_gen on our GPUs (SFT's trap-EXIT may have re-armed it)
echo "[setup] releasing sample_gen on GPUs ${GPUS[*]}"
for g in "${GPUS[@]}"; do
  pkill -f "sample_gen.py start $g" 2>/dev/null || true
done
sleep 5

reclaim_gpus() {
  echo "[trap] launching sample_gen on GPUs ${GPUS[*]}" >&2
  for g in "${GPUS[@]}"; do
    nohup $PY $SAMPLE_GEN start $g >> $LOG_DIR/sample_gen_gpu${g}.log 2>&1 &
    disown || true
  done
}
trap reclaim_gpus EXIT

cd $REPO

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
" 2>&1 | tail -5

ts() { date '+%F %T'; }

launch_ckpt() {
  local GPU=$1 STEP=$2
  local EXP=sec23-fulltest-checkpoint-${STEP}
  local CKPT=$RUN_DIR/checkpoint-${STEP}
  local WLOG=$LOG_DIR/sec23-fulltest-gpu${GPU}-ckpt${STEP}.worker.log
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

declare -A GPU_PID GPU_CKPT
declare -a QUEUE=("${CKPTS[@]}")

# Initial fill
for G in "${GPUS[@]}"; do
  [[ ${#QUEUE[@]} -eq 0 ]] && break
  STEP=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
  PID=$(launch_ckpt $G $STEP)
  GPU_PID[$G]=$PID
  GPU_CKPT[$G]=$STEP
  echo "[$(ts)] [start] GPU $G -> ckpt $STEP (pid $PID)"
done

# Pipeline
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
        NPID=$(launch_ckpt $G $STEP)
        GPU_PID[$G]=$NPID
        GPU_CKPT[$G]=$STEP
        echo "[$(ts)] [start] GPU $G -> ckpt $STEP (pid $NPID)"
      fi
    fi
  done
done

echo "[$(ts)] [all-done]"
