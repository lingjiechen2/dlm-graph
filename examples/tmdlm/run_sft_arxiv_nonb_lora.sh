#!/usr/bin/env bash
set -euo pipefail

# TM-DLM SFT launcher for ogbn-arxiv in the **nonb** setting
# (no neighbor labels, fair-vs-LLaGA). Runs topo + notopo in parallel on
# two GPUs. Mirrors the existing nbmask run on GPU7 but with
# `--include_neighbor_labels False`.
#
# Usage:
#   GPUS=0,1 bash examples/tmdlm/run_sft_arxiv_nonb_lora.sh
#
# Optional env vars:
#   MAX_STEPS=7400  PROMPT_FORMAT=category_infill (default)
#   MAX_ANSWER_TOKENS=10
#   PER_DEVICE_TRAIN_BATCH_SIZE=6  GRAD_ACCUM_STEPS=8
#   LEARNING_RATE=5e-5
#   RUN_TAG, OUTPUT_ROOT, USE_SRUN, EXTRA_ARGS, ...
#
# Resume:
#   EXTRA_ARGS="--resume_from_checkpoint True"

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
SFT_SCRIPT="${REPO_ROOT}/examples/tmdlm/sft.py"
PYTHON_BIN="${PYTHON_BIN:-/home/lingjie7/anaconda3/envs/dllm/bin/python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${ZSH_VERSION:-}" && -f "${HOME}/.zshrc" ]]; then
  source "${HOME}/.zshrc" || true
fi
if command -v conda >/dev/null 2>&1; then
  CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/lingjie7/anaconda3/envs/dllm}"
  conda activate "${CONDA_ENV_PATH}" || true
fi
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="$(command -v python)"
[[ -f "${SFT_SCRIPT}" ]] || { echo "SFT script not found: ${SFT_SCRIPT}" >&2; exit 1; }

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1}"
[[ ${#GPU_LIST[@]} -ge 2 ]] || { echo "Need at least 2 GPU ids in GPUS (e.g. GPUS=0,1)" >&2; exit 2; }

DATASET_NAME=ogbn-arxiv
PROMPT_FORMAT="${PROMPT_FORMAT:-category_infill}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-10}"
MAX_STEPS="${MAX_STEPS:-7400}"

BASE_MODEL="${BASE_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/.models}"
RUN_TAG="${RUN_TAG:-arxiv_$(date +%Y%m%d)_nonb}"

LEARNING_RATE="${LEARNING_RATE:-5e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-6}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_NEIGHBORS_PER_HOP="${MAX_NEIGHBORS_PER_HOP:-10}"
MAX_HOPS="${MAX_HOPS:-2}"
POSITION_ID_TYPE="${POSITION_ID_TYPE:-sequential}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-0.0}"
SAVE_STEPS="${SAVE_STEPS:-0.05}"
EVAL_STEPS="${EVAL_STEPS:-0.1}"
REPORT_TO="${REPORT_TO:-wandb}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

USE_SRUN="${USE_SRUN:-0}"
SRUN_TIME="${SRUN_TIME:-150:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-24}"

declare -a EXPERIMENTS=(
  "True topo"
  "False notopo"
)

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

declare -a PIDS=() JOB_NAMES=()
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

  run_name="tmdlm-llada-8b-${DATASET_NAME}-${MAX_HOPS}hop-${topo_name}-catinfill-nonb-r${LORA_R}-steps${MAX_STEPS}-${RUN_TAG}"
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
    --max_answer_tokens "${MAX_ANSWER_TOKENS}"
    --include_neighbor_labels False
    --output_dir "${output_dir}"
    --max_steps "${MAX_STEPS}"
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
    cmd+=( "${extra_args[@]}" )
  fi

  echo "[launch] dataset=${DATASET_NAME} format=${PROMPT_FORMAT} gpu=${gpu_id} topo=${use_topology_mask} max_steps=${MAX_STEPS} -> ${output_dir}"
  echo "[log] ${log_file}"

  if [[ "${USE_SRUN}" = "1" ]]; then
    [[ -n "${PARTITION:-}" && -n "${QUOTATYPE:-}" ]] || { echo "USE_SRUN=1 requires PARTITION and QUOTATYPE" >&2; exit 1; }
    (
      srun -p "${PARTITION}" --quotatype="${QUOTATYPE}" \
        --gres=gpu:1 --cpus-per-task="${CPUS_PER_TASK}" --time="${SRUN_TIME}" \
        "${cmd[@]}"
    ) >"${log_file}" 2>&1 &
  else
    (
      CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}"
    ) >"${log_file}" 2>&1 &
  fi

  PIDS+=("$!")
  JOB_NAMES+=("${run_name}")
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
