#!/usr/bin/env bash
set -euo pipefail

# Smoke + experimental run: cora-only mc_digit nonb SFT with `boost` resampling
# applied to the two hardest classes (Theory, Rule Learning). Compares against
# the existing baseline single-cora mc_digit nonb run to see whether boosting
# the under-performing classes lifts their per-class accuracy.
#
# Usage:
#   GPU=6 bash examples/tmdlm/run_sft_cora_mcdigit_boost_nonb_lora.sh

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
SFT_SCRIPT="${REPO_ROOT}/examples/tmdlm/sft.py"
PYTHON_BIN="${PYTHON_BIN:-/home/lingjie7/anaconda3/envs/dllm/bin/python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if command -v conda >/dev/null 2>&1; then
  CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/lingjie7/anaconda3/envs/dllm}"
  conda activate "${CONDA_ENV_PATH}" || true
fi

GPU="${GPU:-6}"

DATASET_NAME=cora
PROMPT_FORMAT=mc_digit
ANSWER_LABEL_STYLE=digit0
MAX_ANSWER_TOKENS=1
NUM_EPOCHS="${NUM_EPOCHS:-10}"

RESAMPLE_STRATEGY="${RESAMPLE_STRATEGY:-boost}"
BOOST_SPEC="${BOOST_SPEC:-cora:Theory:2,cora:Rule Learning:3}"
boost_tag="$(echo "${BOOST_SPEC}" | tr ',' '_' | tr ':' '-' | tr ' ' '_')"

BASE_MODEL="${BASE_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/.models}"
RUN_TAG="${RUN_TAG:-cora_$(date +%Y%m%d)_mcdigit_boost_nonb}"

LEARNING_RATE="${LEARNING_RATE:-5e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_NEIGHBORS_PER_HOP="${MAX_NEIGHBORS_PER_HOP:-10}"
MAX_HOPS="${MAX_HOPS:-2}"
POSITION_ID_TYPE="${POSITION_ID_TYPE:-sequential}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-1.0}"
SAVE_STEPS="${SAVE_STEPS:-0.05}"
EVAL_STEPS="${EVAL_STEPS:-0.1}"
REPORT_TO="${REPORT_TO:-wandb}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

USE_TOPOLOGY_MASK="${USE_TOPOLOGY_MASK:-False}"
topo_name=$([[ "${USE_TOPOLOGY_MASK}" = "True" ]] && echo topo || echo notopo)

run_name="tmdlm-llada-8b-cora-${MAX_HOPS}hop-${topo_name}-mcdigit-d0-boost-nonb-r${LORA_R}-ep${NUM_EPOCHS}-${RUN_TAG}"
output_dir="${OUTPUT_ROOT}/${run_name}"
log_file="${OUTPUT_ROOT}/${run_name}.log"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

cmd=(
  "${PYTHON_BIN}" "${SFT_SCRIPT}"
  --model_name_or_path "${BASE_MODEL}"
  --dataset_name "${DATASET_NAME}"
  --resample_strategy "${RESAMPLE_STRATEGY}"
  --boost_spec "${BOOST_SPEC}"
  --max_hops "${MAX_HOPS}"
  --use_topology_mask "${USE_TOPOLOGY_MASK}"
  --position_id_type "${POSITION_ID_TYPE}"
  --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}"
  --prompt_format "${PROMPT_FORMAT}"
  --answer_label_style "${ANSWER_LABEL_STYLE}"
  --max_answer_tokens "${MAX_ANSWER_TOKENS}"
  --include_neighbor_labels False
  --output_dir "${output_dir}"
  --num_train_epochs "${NUM_EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
  --cls_loss_weight "${CLS_LOSS_WEIGHT}"
  --save_steps "${SAVE_STEPS}"
  --eval_steps "${EVAL_STEPS}"
  --report_to "${REPORT_TO}"
  --run_name "${run_name}"
  --overwrite_output_dir
  --lora True
  --r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --target_modules "${LORA_TARGET_MODULES}"
)

echo "[launch] dataset=${DATASET_NAME} format=${PROMPT_FORMAT} gpu=${GPU} topo=${USE_TOPOLOGY_MASK} resample=${RESAMPLE_STRATEGY} boost='${BOOST_SPEC}' -> ${output_dir}"
echo "[log] ${log_file}"

CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}" >"${log_file}" 2>&1
