#!/usr/bin/env bash
set -euo pipefail

# Single-GPU sanity SFT for link prediction on ogbn-arxiv.
# ogbn-arxiv LP has ~984K train pairs (84% pos + neg @ ratio=1); for sanity we
# cap with --max_train_samples to a manageable size and run a few hundred steps.
#
# Usage:
#   GPU=2 bash examples/tmdlm/run_sft_arxiv_lp_lora_sanity.sh

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
SFT_SCRIPT="${REPO_ROOT}/examples/tmdlm/sft.py"
PYTHON_BIN="${PYTHON_BIN:-/home/lingjie7/anaconda3/envs/dllm/bin/python}"
SAMPLE_GEN="/home/lingjie7/sample_gen.py"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

GPU="${GPU:-2}"   # project quota: 2/3/4/6

DATASET_NAME=ogbn-arxiv
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MAX_NEIGHBORS_PER_HOP="${MAX_NEIGHBORS_PER_HOP:-10}"
MAX_HOPS="${MAX_HOPS:-2}"
LP_NEG_RATIO="${LP_NEG_RATIO:-1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-4000}"
MAX_STEPS="${MAX_STEPS:-300}"
USE_TOPOLOGY_MASK="${USE_TOPOLOGY_MASK:-True}"

BASE_MODEL="${BASE_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/.models}"
RUN_TAG="${RUN_TAG:-arxiv_lp_$(date +%Y%m%d)_sanity}"

LEARNING_RATE="${LEARNING_RATE:-5e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
POSITION_ID_TYPE="${POSITION_ID_TYPE:-sequential}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
SAVE_STEPS="${SAVE_STEPS:-0.5}"
REPORT_TO="${REPORT_TO:-wandb}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

run_name="tmdlm-llada-8b-${DATASET_NAME}-lp-${MAX_HOPS}hop-r${LORA_R}-steps${MAX_STEPS}-${RUN_TAG}"
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
echo "[launch] GPU=${GPU}  task=lp  dataset=${DATASET_NAME}"
echo "[log]    ${log_file}"

pkill -f "sample_gen.py start ${GPU}" 2>/dev/null || true
sleep 3

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SFT_SCRIPT}" \
    --task lp \
    --model_name_or_path "${BASE_MODEL}" \
    --dataset_name "${DATASET_NAME}" \
    --lp_neg_ratio "${LP_NEG_RATIO}" \
    --max_hops "${MAX_HOPS}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --use_topology_mask "${USE_TOPOLOGY_MASK}" \
    --position_id_type "${POSITION_ID_TYPE}" \
    --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --output_dir "${output_dir}" \
    --max_steps "${MAX_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --save_steps "${SAVE_STEPS}" \
    --report_to "${REPORT_TO}" \
    --run_name "${run_name}" \
    --overwrite_output_dir \
    --lora True \
    --r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --target_modules "${LORA_TARGET_MODULES}" \
    >"${log_file}" 2>&1
