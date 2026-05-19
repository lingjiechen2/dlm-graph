#!/usr/bin/env bash
set -euo pipefail

# §23 3ep fulltrain — N=5000 eval on GPU 0 (user-authorized override of
# default GPU quota {2,3,4,6}; runs in parallel with the still-active §23 SFT
# on GPUs 2/3/4/6). Sequential, newest-first.

REPO=/home/lingjie7/auto-research/projects/dlm-graph
RUN_DIR=$REPO/.models/tmdlm-llada-8b-ogbn-arxiv-2hop-topo-mcdigit-d0-nonb-r128-steps7188-arxiv_20260514_fulltrain_r128_3ep
WORK=$REPO/analysis/postprocess_arxiv_r128
JSONL_DIR=$WORK/eval_jsonl
LOGITS_DIR=$WORK/logits_cache
LOG_DIR=$WORK/logs
JSONL=$JSONL_DIR/sec23_partial_5000.jsonl
PY=/home/lingjie7/anaconda3/envs/dllm/bin/python
EVAL=$REPO/examples/tmdlm/eval_logit.py

GPU=0
# Newest-first: SFT is still running, so start from the latest ckpt
CKPTS=(5752 5033 4314 3595 2876 2157 1438 719)

BASE_MODEL=GSAI-ML/LLaDA-8B-Instruct
BOOST_SPEC='ogbn-arxiv:cs.LG(Machine Learning):3,ogbn-arxiv:cs.AI(Artificial Intelligence):2,ogbn-arxiv:cs.NE(Neural and Evolutionary Computing):2'
TRAIN_RESAMPLE=boost
TRAIN_MAX_SAMPLES=0

mkdir -p $JSONL_DIR $LOGITS_DIR $LOG_DIR
export HF_HOME=/tmp/dlm-graph-hf-home
export TRANSFORMERS_CACHE=/tmp/dlm-graph-hf-transformers
export PYTHONPATH=$REPO

cd $REPO

# Train cache for class-prior calibration (CPU-only, no GPU contention)
echo "[prewarm] building/loading train cache..."
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

for STEP in "${CKPTS[@]}"; do
  CKPT=$RUN_DIR/checkpoint-${STEP}
  if [[ ! -d $CKPT ]]; then
    echo "[$(ts)] [skip] $CKPT not found"
    continue
  fi
  EXP=sec23-partial-5000-checkpoint-${STEP}
  NPZ=$LOGITS_DIR/sec23_ckpt-${STEP}_5k.npz
  WLOG=$LOG_DIR/sec23-5000-gpu${GPU}-ckpt${STEP}.worker.log
  : > $WLOG
  echo "[$(ts)] [start] GPU $GPU -> ckpt ${STEP}"
  CUDA_VISIBLE_DEVICES=$GPU $PY $EVAL \
    --exp "$EXP" \
    --model_name_or_path "$BASE_MODEL" \
    --lora_path "$CKPT" \
    --dataset_name ogbn-arxiv \
    --split test \
    --max_samples 5000 \
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
    >> $WLOG 2>&1
  echo "[$(ts)] [done]  ckpt ${STEP}"
done

echo "[$(ts)] [all-done]"
