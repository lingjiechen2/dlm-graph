#!/usr/bin/env bash
set -euo pipefail

# Evaluate 4 LoRA SFT runs on Cora:
# 1-hop + topo, 1-hop + no-topo, 2-hop + topo, 2-hop + no-topo.
#
# Why eval_logit.py by default:
# - sft.py optimizes masked answer-token denoising loss on answer positions.
# - eval_logit.py uses the same masked-answer setup with restricted class-token scoring.
# - This gives the most direct train-eval consistency for current mc_digit SFT runs.
#
# Usage:
#   bash /home/lingjie7/auto-research/projects/dlm-graph/examples/tmdlm/run_eval_cora_hops_topo_lora.sh
#
# Optional env vars:
#   RUN_TAG=20260418_185914
#   EVAL_GPUS=0,1,2,3
#   BASE_MODEL=GSAI-ML/LLaDA-8B-Instruct
#   DATASET_NAME=cora
#   SPLIT=test
#   POSITION_ID_TYPE=sequential
#   PROMPT_FORMAT=mc_digit
#   MAX_ANSWER_TOKENS=1
#   MAX_NEIGHBORS_PER_HOP=10
#   BATCH_SIZE=8
#   OUTPUT_ROOT=/home/lingjie7/auto-research/projects/dlm-graph/.models
#   JSONL_FILE=/home/lingjie7/auto-research/projects/dlm-graph/experiments/experiment_log.jsonl
#   ALLOW_MISSING=1
#   DRY_RUN=1

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
EVAL_SCRIPT="${REPO_ROOT}/examples/tmdlm/eval_logit.py"
PYTHON_BIN="${PYTHON_BIN:-/home/lingjie7/anaconda3/envs/dllm/bin/python}"

if command -v conda >/dev/null 2>&1; then
  CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/lingjie7/anaconda3/envs/dllm}"
  conda activate "${CONDA_ENV_PATH}" || true
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Eval script not found: ${EVAL_SCRIPT}" >&2
  exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/.models}"
BASE_MODEL="${BASE_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
DATASET_NAME="${DATASET_NAME:-cora}"
SPLIT="${SPLIT:-test}"
POSITION_ID_TYPE="${POSITION_ID_TYPE:-sequential}"
PROMPT_FORMAT="${PROMPT_FORMAT:-mc_digit}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-1}"
MAX_NEIGHBORS_PER_HOP="${MAX_NEIGHBORS_PER_HOP:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
DRY_RUN="${DRY_RUN:-0}"

JSONL_FILE="${JSONL_FILE:-${REPO_ROOT}/experiments/experiment_log.jsonl}"
mkdir -p "$(dirname "${JSONL_FILE}")"

IFS=',' read -r -a EVAL_GPU_LIST <<< "${EVAL_GPUS:-0,1,2,3}"
if [[ ${#EVAL_GPU_LIST[@]} -lt 4 ]]; then
  echo "Need at least 4 GPU ids in EVAL_GPUS (example: EVAL_GPUS=0,1,2,3)." >&2
  exit 1
fi

latest_run_tag() {
  local tags
  tags="$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name 'tmdlm-llada-8b-cora-*hop-*-lora-*' \
    | sed -E 's|.*/tmdlm-llada-8b-cora-[0-9]+hop-(topo|notopo)-lora-||' \
    | sort -u)"
  if [[ -z "${tags}" ]]; then
    echo ""
  else
    echo "${tags}" | tail -n 1
  fi
}

find_best_checkpoint() {
  local run_dir="$1"
  if [[ -d "${run_dir}/checkpoint-final" ]]; then
    echo "${run_dir}/checkpoint-final"
    return 0
  fi
  local latest
  latest="$(
    find "${run_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
      | awk -F- '/checkpoint-[0-9]+$/ {print $NF" "$0}' \
      | sort -n | tail -n 1 | cut -d' ' -f2-
  )"
  echo "${latest}"
}

RUN_TAG="${RUN_TAG:-$(latest_run_tag)}"
if [[ -z "${RUN_TAG}" ]]; then
  echo "Cannot infer RUN_TAG from ${OUTPUT_ROOT}. Please set RUN_TAG=..." >&2
  exit 1
fi

declare -a EXPERIMENTS=(
  "1 topo True"
  "1 notopo False"
  "2 topo True"
  "2 notopo False"
)

declare -a PIDS=()
declare -a JOBS=()

echo "Using RUN_TAG=${RUN_TAG}"
echo "Logging JSONL to ${JSONL_FILE}"

cleanup() {
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "Stopping eval jobs: ${PIDS[*]}" >&2
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for i in "${!EXPERIMENTS[@]}"; do
  read -r hops topo_name topo_bool <<< "${EXPERIMENTS[$i]}"
  gpu_id="${EVAL_GPU_LIST[$i]}"
  run_dir="${OUTPUT_ROOT}/tmdlm-llada-8b-cora-${hops}hop-${topo_name}-lora-${RUN_TAG}"
  ckpt_path="$(find_best_checkpoint "${run_dir}")"

  if [[ -z "${ckpt_path}" || ! -d "${ckpt_path}" ]]; then
    msg="Checkpoint not found for ${run_dir} (expected checkpoint-final or checkpoint-*)"
    if [[ "${ALLOW_MISSING}" == "1" ]]; then
      echo "[skip] ${msg}"
      continue
    fi
    echo "[error] ${msg}" >&2
    exit 1
  fi

  exp_name="eval-cora-${hops}hop-${topo_name}-lora-${RUN_TAG}"
  log_file="${OUTPUT_ROOT}/${exp_name}.log"

  cmd=(
    "${PYTHON_BIN}" "${EVAL_SCRIPT}"
    --exp "${exp_name}"
    --model_name_or_path "${BASE_MODEL}"
    --lora_path "${ckpt_path}"
    --dataset_name "${DATASET_NAME}"
    --split "${SPLIT}"
    --max_hops "${hops}"
    --use_topology_mask "${topo_bool}"
    --position_id_type "${POSITION_ID_TYPE}"
    --prompt_format "${PROMPT_FORMAT}"
    --max_answer_tokens "${MAX_ANSWER_TOKENS}"
    --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}"
    --batch_size "${BATCH_SIZE}"
    --log_file "${JSONL_FILE}"
  )

  echo "[launch] gpu=${gpu_id} exp=${exp_name}"
  echo "[ckpt] ${ckpt_path}"
  echo "[log] ${log_file}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] CUDA_VISIBLE_DEVICES=${gpu_id} ${cmd[*]}"
    continue
  fi

  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}"
  ) >"${log_file}" 2>&1 &

  PIDS+=("$!")
  JOBS+=("${exp_name}")
done

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "No evaluation jobs were launched."
  exit 0
fi

failed=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  job="${JOBS[$i]}"
  if wait "${pid}"; then
    echo "[done] ${job} (pid=${pid})"
  else
    echo "[fail] ${job} (pid=${pid})" >&2
    failed=1
  fi
done

exit "${failed}"
