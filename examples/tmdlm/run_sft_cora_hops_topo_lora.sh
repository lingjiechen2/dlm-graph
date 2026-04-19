#!/usr/bin/env bash
set -euo pipefail

# Run 4 Cora SFT jobs with LoRA:
# 1-hop + topo, 1-hop + no-topo, 2-hop + topo, 2-hop + no-topo.
#
# Usage:
#   bash /home/lingjie7/auto-research/projects/dlm-graph/examples/tmdlm/run_sft_cora_hops_topo_lora.sh
#
# Optional env vars:
#   GPUS=0,1,2,3
#   NUM_EPOCHS=5
#   OUTPUT_ROOT=/home/lingjie7/auto-research/projects/dlm-graph/.models
#   RUN_TAG=mytag
#   USE_SRUN=0
#   EXTRA_ARGS="--save_total_limit 2"

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
SFT_SCRIPT="${REPO_ROOT}/examples/tmdlm/sft.py"
PYTHON_BIN="${PYTHON_BIN:-/home/lingjie7/anaconda3/envs/dllm/bin/python}"

if [[ -n "${ZSH_VERSION:-}" && -f "${HOME}/.zshrc" ]]; then
  source "${HOME}/.zshrc" || true
fi

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

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3}"
if [[ ${#GPU_LIST[@]} -lt 4 ]]; then
  echo "Need at least 4 GPU ids in GPUS (example: GPUS=0,1,2,3)." >&2
  exit 1
fi

BASE_MODEL="${BASE_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
DATASET_NAME="${DATASET_NAME:-cora}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/.models}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

NUM_EPOCHS="${NUM_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_NEIGHBORS_PER_HOP="${MAX_NEIGHBORS_PER_HOP:-10}"
POSITION_ID_TYPE="${POSITION_ID_TYPE:-sequential}"
PROMPT_FORMAT="${PROMPT_FORMAT:-mc_digit}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-0.0}"
REPORT_TO="${REPORT_TO:-wandb}"

LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

USE_SRUN="${USE_SRUN:-0}"
SRUN_TIME="${SRUN_TIME:-03:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-24}"

declare -a EXPERIMENTS=(
  "1 True topo"
  "1 False notopo"
  "2 True topo"
  "2 False notopo"
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
  read -r max_hops use_topology_mask topo_name <<< "${EXPERIMENTS[$i]}"
  gpu_id="${GPU_LIST[$i]}"

  run_name="${DATASET_NAME}-${max_hops}hop-${topo_name}-lora-${RUN_TAG}"
  output_dir="${OUTPUT_ROOT}/tmdlm-llada-8b-${run_name}"
  log_file="${OUTPUT_ROOT}/${run_name}.log"

  cmd=(
    "${PYTHON_BIN}" "${SFT_SCRIPT}"
    --model_name_or_path "${BASE_MODEL}"
    --dataset_name "${DATASET_NAME}"
    --max_hops "${max_hops}"
    --use_topology_mask "${use_topology_mask}"
    --position_id_type "${POSITION_ID_TYPE}"
    --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}"
    --prompt_format "${PROMPT_FORMAT}"
    --max_answer_tokens "${MAX_ANSWER_TOKENS}"
    --output_dir "${output_dir}"
    --num_train_epochs "${NUM_EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --cls_loss_weight "${CLS_LOSS_WEIGHT}"
    --report_to "${REPORT_TO}"
    --run_name "${run_name}"
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

  echo "[launch] gpu=${gpu_id} hops=${max_hops} topo_mask=${use_topology_mask} output=${output_dir}"
  echo "[log] ${log_file}"

  if [[ "${USE_SRUN}" == "1" ]]; then
    if [[ -z "${PARTITION:-}" || -z "${QUOTATYPE:-}" ]]; then
      echo "USE_SRUN=1 requires PARTITION and QUOTATYPE env vars." >&2
      exit 1
    fi
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
