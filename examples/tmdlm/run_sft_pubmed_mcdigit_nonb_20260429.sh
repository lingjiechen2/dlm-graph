#!/usr/bin/env bash
set -euo pipefail

# TM-DLM SFT on PubMed, mc_digit format, NO neighbor labels (nonb).
# Fresh run with tag pubmed_20260429_mcdigit_nonb to avoid clobbering
# previous (20260428) checkpoints.
#
# Trains both notopo and topo variants in parallel on two GPUs.
#
# Usage:
#   GPUS=0,1 bash examples/tmdlm/run_sft_pubmed_mcdigit_nonb_20260429.sh
#
# Optional env overrides:
#   NUM_EPOCHS=10
#   RUN_TAG=pubmed_20260429_mcdigit_nonb
#   LEARNING_RATE=5e-5
#   PER_DEVICE_TRAIN_BATCH_SIZE=2
#   GRAD_ACCUM_STEPS=8
#   SAVE_STEPS=0.05
#   EXTRA_ARGS="--save_total_limit 4"

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
SFT_SCRIPT="${REPO_ROOT}/examples/tmdlm/sft.py"
PYTHON_BIN="${PYTHON_BIN:-/home/lingjie7/anaconda3/envs/dllm/bin/python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if command -v conda >/dev/null 2>&1; then
  CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/lingjie7/anaconda3/envs/dllm}"
  conda activate "${CONDA_ENV_PATH}" || true
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

if [[ ! -f "${SFT_SCRIPT}" ]]; then
  echo "SFT script not found: ${SFT_SCRIPT}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1}"
if [[ ${#GPU_LIST[@]} -lt 2 ]]; then
  echo "Need at least 2 GPU ids in GPUS (example: GPUS=0,1)." >&2
  exit 1
fi

BASE_MODEL="${BASE_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
DATASET_NAME="${DATASET_NAME:-pubmed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/.models}"
RUN_TAG="${RUN_TAG:-pubmed_20260429_mcdigit_nonb}"

NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_NEIGHBORS_PER_HOP="${MAX_NEIGHBORS_PER_HOP:-10}"
MAX_HOPS="${MAX_HOPS:-2}"
POSITION_ID_TYPE="${POSITION_ID_TYPE:-sequential}"
PROMPT_FORMAT="${PROMPT_FORMAT:-mc_digit}"
ANSWER_LABEL_STYLE="${ANSWER_LABEL_STYLE:-digit0}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-1}"
INCLUDE_NEIGHBOR_LABELS="${INCLUDE_NEIGHBOR_LABELS:-False}"   # nonb
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-0.0}"
SAVE_STEPS="${SAVE_STEPS:-0.05}"
EVAL_STEPS="${EVAL_STEPS:-0.1}"
REPORT_TO="${REPORT_TO:-wandb}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

declare -a EXPERIMENTS=(
  "True topo"
  "False notopo"
)

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

declare -a PIDS=()
declare -a JOB_NAMES=()

cleanup() {
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "Stopping child jobs: ${PIDS[*]}" >&2
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for i in "${!EXPERIMENTS[@]}"; do
  read -r use_topology_mask topo_name <<< "${EXPERIMENTS[$i]}"
  gpu_id="${GPU_LIST[$i]}"

  run_name="tmdlm-llada-8b-${DATASET_NAME}-${MAX_HOPS}hop-${topo_name}-mcdigit-d0-nonb-r${LORA_R}-ep${NUM_EPOCHS}-${RUN_TAG}"
  output_dir="${OUTPUT_ROOT}/${run_name}"
  log_file="${OUTPUT_ROOT}/${run_name}.log"

  cmd=(
    "${PYTHON_BIN}" "${SFT_SCRIPT}"
    --model_name_or_path "${BASE_MODEL}"
    --dataset_name "${DATASET_NAME}"
    --max_hops "${MAX_HOPS}"
    --use_topology_mask "${use_topology_mask}"
    --position_id_type "${POSITION_ID_TYPE}"
    --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}"
    --prompt_format "${PROMPT_FORMAT}"
    --answer_label_style "${ANSWER_LABEL_STYLE}"
    --max_answer_tokens "${MAX_ANSWER_TOKENS}"
    --include_neighbor_labels "${INCLUDE_NEIGHBOR_LABELS}"
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

  if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=( ${EXTRA_ARGS} )
    cmd+=("${extra_args[@]}")
  fi

  echo "[launch] dataset=${DATASET_NAME} gpu=${gpu_id} hops=${MAX_HOPS} topo_mask=${use_topology_mask} include_neighbor_labels=${INCLUDE_NEIGHBOR_LABELS}"
  echo "[log] ${log_file}"

  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}"
  ) >"${log_file}" 2>&1 &

  PIDS+=("$!")
  JOB_NAMES+=("${run_name}")
done

echo "[info] launched pids: ${PIDS[*]}"
echo "[info] tail logs:"
for job in "${JOB_NAMES[@]}"; do
  echo "  tail -f ${OUTPUT_ROOT}/${job}.log"
done

failed=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  job="${JOB_NAMES[$i]}"
  if wait "${pid}"; then
    echo "[done] ${job} (pid=${pid})"
  else
    echo "[fail] ${job} (pid=${pid})" >&2
    failed=1
  fi
done

exit "${failed}"
