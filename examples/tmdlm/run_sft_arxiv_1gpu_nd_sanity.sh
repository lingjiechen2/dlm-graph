#!/usr/bin/env bash
set -euo pipefail

# Single-GPU sanity SFT for prompt_format=nd_describe on ogbn-arxiv.
# Goal: verify loss decreases and assistant target is produced cleanly.
# Small subset, few hundred steps. ~30-60 min on one GPU.
#
# Usage:
#   GPU=2 bash examples/tmdlm/run_sft_arxiv_1gpu_nd_sanity.sh
#
# To switch to nda_describe (title + abstract): set PROMPT_FORMAT=nda_describe.

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
SFT_SCRIPT="${REPO_ROOT}/examples/tmdlm/sft.py"
PYTHON_BIN="${PYTHON_BIN:-/home/lingjie7/anaconda3/envs/dllm/bin/python}"
SAMPLE_GEN="/home/lingjie7/sample_gen.py"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

GPU="${GPU:-2}"   # project quota: pick one of 2/3/4/6

DATASET_NAME=ogbn-arxiv
PROMPT_FORMAT="${PROMPT_FORMAT:-nd_describe}"   # or nda_describe
MAX_STEPS=300
MAX_TRAIN_SAMPLES=2000
MAX_SEQ_LEN=4096
MAX_HOPS=2
MAX_NEIGHBORS_PER_HOP=10
POSITION_ID_TYPE=sequential
USE_TOPOLOGY_MASK=True

BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
OUTPUT_ROOT="${REPO_ROOT}/.models"
RUN_TAG="${RUN_TAG:-arxiv_20260517_nd_sanity}"

LEARNING_RATE=5e-5
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=2
GRAD_ACCUM_STEPS=8                      # eff bs = 2*8 = 16
GRADIENT_CHECKPOINTING=True
CLS_LOSS_WEIGHT=0.0
SAVE_STEPS=0.5
EVAL_STRATEGY=no
REPORT_TO=wandb

LORA_R=128
LORA_ALPHA=128
LORA_TARGET_MODULES=all-linear

# nd_describe ignores answer_label_style / max_answer_tokens / boost (assistant
# target is free-form text). Keep placeholders so sft.py argparse is happy.
ANSWER_LABEL_STYLE=digit0_pad
MAX_ANSWER_TOKENS=2
RESAMPLE_STRATEGY=none
BOOST_SPEC=''

run_name="tmdlm-llada-8b-${DATASET_NAME}-${PROMPT_FORMAT}-r${LORA_R}-steps${MAX_STEPS}-${RUN_TAG}"
output_dir="${OUTPUT_ROOT}/${run_name}"
log_file="${OUTPUT_ROOT}/${run_name}.log"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

reclaim_gpus() {
  echo "[trap] launching sample_gen on GPU ${GPU}" >&2
  nohup "${PYTHON_BIN}" "${SAMPLE_GEN}" start "${GPU}" \
    >> "${OUTPUT_ROOT}/sample_gen_gpu${GPU}.log" 2>&1 &
  disown || true
}
trap reclaim_gpus EXIT

echo "[launch] run_name=${run_name}"
echo "[launch] GPU=${GPU}  prompt_format=${PROMPT_FORMAT}"
echo "[log]    ${log_file}"

# Kill any sample_gen on our GPU before training
pkill -f "sample_gen.py start ${GPU}" 2>/dev/null || true
sleep 3

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SFT_SCRIPT}" \
    --model_name_or_path "${BASE_MODEL}" \
    --dataset_name "${DATASET_NAME}" \
    --max_hops "${MAX_HOPS}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --use_topology_mask "${USE_TOPOLOGY_MASK}" \
    --position_id_type "${POSITION_ID_TYPE}" \
    --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}" \
    --prompt_format "${PROMPT_FORMAT}" \
    --answer_label_style "${ANSWER_LABEL_STYLE}" \
    --max_answer_tokens "${MAX_ANSWER_TOKENS}" \
    --include_neighbor_labels False \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --resample_strategy "${RESAMPLE_STRATEGY}" \
    --boost_spec "${BOOST_SPEC}" \
    --output_dir "${output_dir}" \
    --max_steps "${MAX_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --cls_loss_weight "${CLS_LOSS_WEIGHT}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_strategy "${EVAL_STRATEGY}" \
    --report_to "${REPORT_TO}" \
    --run_name "${run_name}" \
    --overwrite_output_dir \
    --lora True \
    --r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --target_modules "${LORA_TARGET_MODULES}" \
    >"${log_file}" 2>&1
