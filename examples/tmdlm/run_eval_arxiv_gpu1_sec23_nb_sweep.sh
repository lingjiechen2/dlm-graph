#!/usr/bin/env bash
set -euo pipefail

# §23 ckpt-5752 (current best, raw=77.24 @ N=5000 nb=10) — nb sweep on GPU 1.
# Sequential, N=5000, same hops/topo/prompt as training.

REPO=/home/lingjie7/auto-research/projects/dlm-graph
RUN_DIR=$REPO/.models/tmdlm-llada-8b-ogbn-arxiv-2hop-topo-mcdigit-d0-nonb-r128-steps7188-arxiv_20260514_fulltrain_r128_3ep
WORK=$REPO/analysis/postprocess_arxiv_r128
JSONL_DIR=$WORK/eval_jsonl
LOGITS_DIR=$WORK/logits_cache
LOG_DIR=$WORK/logs
JSONL=$JSONL_DIR/sec23_ckpt5752_nb_sweep_5000.jsonl
PY=/home/lingjie7/anaconda3/envs/dllm/bin/python
EVAL=$REPO/examples/tmdlm/eval_logit.py

GPU=1
STEP=5752
NBS=(5 10 12 15 18 20 25 30)

BASE_MODEL=GSAI-ML/LLaDA-8B-Instruct
BOOST_SPEC='ogbn-arxiv:cs.LG(Machine Learning):3,ogbn-arxiv:cs.AI(Artificial Intelligence):2,ogbn-arxiv:cs.NE(Neural and Evolutionary Computing):2'
TRAIN_RESAMPLE=boost
TRAIN_MAX_SAMPLES=0
CKPT=$RUN_DIR/checkpoint-${STEP}

mkdir -p $JSONL_DIR $LOGITS_DIR $LOG_DIR
export HF_HOME=/tmp/dlm-graph-hf-home
export TRANSFORMERS_CACHE=/tmp/dlm-graph-hf-transformers
export PYTHONPATH=$REPO

cd $REPO

# Train cache already prewarmed by other dispatchers; this is a no-op if cached
echo "[prewarm] ensuring train cache..."
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
print(f'[prewarm] train cache: {len(ds)} samples')
" 2>&1 | tail -3

ts() { date '+%F %T'; }

for NB in "${NBS[@]}"; do
  EXP=sec23-ckpt5752-nb${NB}-5000
  NPZ=$LOGITS_DIR/sec23_ckpt5752_nb${NB}_5k.npz
  WLOG=$LOG_DIR/sec23-nb${NB}-gpu${GPU}-ckpt${STEP}.worker.log
  : > $WLOG
  echo "[$(ts)] [start] GPU $GPU -> nb=${NB}"
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
    --max_neighbors_per_hop ${NB} \
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
  echo "[$(ts)] [done]  nb=${NB}"
done

echo "[$(ts)] [all-done]"
