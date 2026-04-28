#!/usr/bin/env bash
set -euo pipefail

# Evaluate all LoRA checkpoints for latest Cora SFT settings:
# - 2hop + topo
# - 2hop + notopo
#
# Each setting is pinned to one GPU and evaluated sequentially by checkpoint step.
#
# Usage:
#   bash /home/lingjie7/auto-research/projects/dlm-graph/examples/tmdlm/run_eval_cora_hops_topo_lora_all_ckpts.sh
#
# Optional env vars:
#   RUN_TAG=20260418_185914
#   EVAL_GPUS=0,1
#   OUTPUT_ROOT=/home/lingjie7/auto-research/projects/dlm-graph/.models
#   RUNS_ROOT=/home/lingjie7/auto-research/projects/dlm-graph/.models
#   WORK_LOG_ROOT=/tmp/dlm-graph-eval-logs
#   BASE_MODEL=GSAI-ML/LLaDA-8B-Instruct
#   DATASET_NAME=cora
#   SPLIT=test
#   POSITION_ID_TYPE=sequential
#   PROMPT_FORMAT=category_infill
#   MAX_ANSWER_TOKENS=6
#   INCLUDE_NEIGHBOR_LABELS=True
#   NEIGHBOR_LABEL_FORMAT=bracket
#   MAX_NEIGHBORS_PER_HOP=10
#   BATCH_SIZE=8
#   JSONL_ROOT=/tmp/dlm-graph-eval-jsonl
#   HF_HOME=/tmp/dlm-graph-hf-home
#   HF_DATASETS_CACHE=/tmp/dlm-graph-hf-datasets
#   TRANSFORMERS_CACHE=/tmp/dlm-graph-hf-transformers
#   ALLOW_MISSING=1
#   DRY_RUN=1

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
EVAL_SCRIPT="${REPO_ROOT}/examples/tmdlm/eval_infill.py"
EVAL_KIND="${EVAL_KIND:-infill}"
INFILL_STEPS="${INFILL_STEPS:-10}"
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
RUNS_ROOT="${RUNS_ROOT:-${OUTPUT_ROOT}}"
WORK_LOG_ROOT="${WORK_LOG_ROOT:-/tmp/dlm-graph-eval-logs}"
BASE_MODEL="${BASE_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
DATASET_NAME="${DATASET_NAME:-cora}"
SPLIT="${SPLIT:-test}"
POSITION_ID_TYPE="${POSITION_ID_TYPE:-sequential}"
PROMPT_FORMAT="${PROMPT_FORMAT:-category_infill}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-6}"
INCLUDE_NEIGHBOR_LABELS="${INCLUDE_NEIGHBOR_LABELS:-True}"
NEIGHBOR_LABEL_FORMAT="${NEIGHBOR_LABEL_FORMAT:-bracket}"
MAX_NEIGHBORS_PER_HOP="${MAX_NEIGHBORS_PER_HOP:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
DRY_RUN="${DRY_RUN:-0}"

JSONL_ROOT="${JSONL_ROOT:-/tmp/dlm-graph-eval-jsonl}"
mkdir -p "${JSONL_ROOT}"
mkdir -p "${WORK_LOG_ROOT}"

HF_HOME="${HF_HOME:-/tmp/dlm-graph-hf-home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/dlm-graph-hf-datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/dlm-graph-hf-transformers}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

IFS=',' read -r -a EVAL_GPU_LIST <<< "${EVAL_GPUS:-0,1}"
if [[ ${#EVAL_GPU_LIST[@]} -lt 2 ]]; then
  echo "Need at least 2 GPU ids in EVAL_GPUS (example: EVAL_GPUS=0,1)." >&2
  exit 1
fi

latest_run_tag() {
  local tags
  tags="$(find -L "${RUNS_ROOT}" -maxdepth 1 -type d -name "tmdlm-llada-8b-${DATASET_NAME}-2hop-*-catinfill-nbmask-noeospad-r*-ep*-*" \
    | sed -E "s#.*/tmdlm-llada-8b-${DATASET_NAME}-2hop-(topo|notopo)-catinfill-nbmask-noeospad-r[0-9]+-ep[0-9]+-##" \
    | sort -u)"
  if [[ -z "${tags}" ]]; then
    echo ""
  else
    echo "${tags}" | tail -n 1
  fi
}

RUN_TAG="${RUN_TAG:-$(latest_run_tag)}"
if [[ -z "${RUN_TAG}" ]]; then
  echo "Cannot infer RUN_TAG from ${RUNS_ROOT}. Please set RUN_TAG=..." >&2
  exit 1
fi

get_checkpoints_sorted() {
  local run_dir="$1"
  find -L "${run_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | awk -F- '/checkpoint-[0-9]+$/ {print $NF" "$0}' \
    | sort -n \
    | cut -d' ' -f2-
}

cd "${REPO_ROOT}"

declare -a SETTINGS=(
  "2 topo True"
  "2 notopo False"
)

declare -a PIDS=()
declare -a JOBS=()

echo "RUN_TAG=${RUN_TAG}"
echo "RUNS_ROOT=${RUNS_ROOT}"
echo "WORK_LOG_ROOT=${WORK_LOG_ROOT}"
echo "JSONL_ROOT=${JSONL_ROOT}"

cleanup() {
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "Stopping running evaluation workers: ${PIDS[*]}" >&2
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for i in "${!SETTINGS[@]}"; do
  read -r hops topo_name topo_bool <<< "${SETTINGS[$i]}"
  gpu_id="${EVAL_GPU_LIST[$i]}"
  run_dir="$(find -L "${RUNS_ROOT}" -maxdepth 1 -type d -name "tmdlm-llada-8b-${DATASET_NAME}-${hops}hop-${topo_name}-catinfill-nbmask-noeospad-r*-ep*-${RUN_TAG}" | head -n 1)"
  worker_log="${WORK_LOG_ROOT}/eval-${DATASET_NAME}-${hops}hop-${topo_name}-${RUN_TAG}-${EVAL_KIND}.worker.log"
  worker_jsonl="${JSONL_ROOT}/eval-${DATASET_NAME}-${hops}hop-${topo_name}-${RUN_TAG}-${EVAL_KIND}.jsonl"

  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    msg="Run directory not found for dataset=${DATASET_NAME}, hops=${hops}, topo=${topo_name}, tag=${RUN_TAG}"
    if [[ "${ALLOW_MISSING}" == "1" ]]; then
      echo "[skip] ${msg}"
      continue
    fi
    echo "[error] ${msg}" >&2
    exit 1
  fi

  mapfile -t ckpts < <(get_checkpoints_sorted "${run_dir}")
  if [[ ${#ckpts[@]} -eq 0 ]]; then
    msg="No checkpoint-* found in ${run_dir}"
    if [[ "${ALLOW_MISSING}" == "1" ]]; then
      echo "[skip] ${msg}"
      continue
    fi
    echo "[error] ${msg}" >&2
    exit 1
  fi

  echo "[plan] gpu=${gpu_id} setting=${hops}hop-${topo_name} checkpoints=${#ckpts[@]}"

  (
    set -euo pipefail
    : > "${worker_log}"
    for ckpt_path in "${ckpts[@]}"; do
      ckpt_name="$(basename "${ckpt_path}")"
      step="${ckpt_name#checkpoint-}"
      exp_name="eval-${DATASET_NAME}-${hops}hop-${topo_name}-lora-${RUN_TAG}-${EVAL_KIND}-${ckpt_name}"

      cmd=(
        "${PYTHON_BIN}" "${EVAL_SCRIPT}"
        --exp "${exp_name}"
        --model_name_or_path "${BASE_MODEL}"
        --lora_path "${ckpt_path}"
        --dataset_name "${DATASET_NAME}"
        --split "${SPLIT}"
        --max_hops "${hops}"
        --use_topology_mask "${topo_bool}"
        --prompt_format "${PROMPT_FORMAT}"
        --max_answer_tokens "${MAX_ANSWER_TOKENS}"
        --include_neighbor_labels "${INCLUDE_NEIGHBOR_LABELS}"
        --neighbor_label_format "${NEIGHBOR_LABEL_FORMAT}"
        --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}"
        --batch_size "${BATCH_SIZE}"
        --steps "${INFILL_STEPS}"
        --log_file "${worker_jsonl}"
      )

      {
        echo "[$(date '+%F %T')] [start] ${exp_name} gpu=${gpu_id}"
        echo "[$(date '+%F %T')] [ckpt] ${ckpt_path}"
      } >> "${worker_log}"

      if [[ ! -f "${ckpt_path}/adapter_config.json" ]]; then
        echo "[$(date '+%F %T')] [skip] missing adapter_config.json at ${ckpt_path}" >> "${worker_log}"
        continue
      fi

      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] CUDA_VISIBLE_DEVICES=${gpu_id} ${cmd[*]}" >> "${worker_log}"
        continue
      fi

      PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      HF_HOME="${HF_HOME}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
      CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" >> "${worker_log}" 2>&1
      echo "[$(date '+%F %T')] [done] ${exp_name} (step=${step})" >> "${worker_log}"
    done
    if [[ "${HOLD_GPU_AFTER:-1}" == "1" ]]; then
      echo "[$(date '+%F %T')] [hold-gpu] starting sample_gen on gpu=${gpu_id}" >> "${worker_log}"
      nohup /home/lingjie7/anaconda3/envs/dllm/bin/python /home/lingjie7/sample_gen.py start "${gpu_id}" \
        >> "${worker_log}" 2>&1 < /dev/null &
      disown || true
    fi
  ) &

  PIDS+=("$!")
  JOBS+=("eval-${hops}hop-${topo_name}")
done

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "No evaluation workers launched."
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
