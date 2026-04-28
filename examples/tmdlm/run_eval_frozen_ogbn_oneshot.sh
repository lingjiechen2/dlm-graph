#!/usr/bin/env bash
set -euo pipefail

# One-shot: frozen-base (no LoRA) eval_logit on ogbn-arxiv and ogbn-products,
# topo + notopo per dataset, pinned to one GPU per dataset.
# After each GPU finishes its 2 settings, sample_gen.py occupies the GPU.

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
EVAL_SCRIPT="${REPO_ROOT}/examples/tmdlm/eval_logit.py"
PYTHON_BIN="/home/lingjie7/anaconda3/envs/dllm/bin/python"
SAMPLE_GEN="/home/lingjie7/sample_gen.py"

BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
SPLIT="test"
PROMPT_FORMAT="category_infill"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-0}"   # 0 = auto from class names
INCLUDE_NEIGHBOR_LABELS=True
NEIGHBOR_LABEL_FORMAT="bracket"
MAX_NEIGHBORS_PER_HOP=10
MAX_HOPS=2
BATCH_SIZE="${BATCH_SIZE:-4}"
TAG="${TAG:-20260428}"

WORK_LOG_ROOT="/tmp/dlm-graph-eval-logs"
JSONL_ROOT="/tmp/dlm-graph-eval-jsonl"
SUMMARIES_ROOT="${REPO_ROOT}/summaries"
mkdir -p "${WORK_LOG_ROOT}" "${JSONL_ROOT}" "${SUMMARIES_ROOT}"

HF_HOME="${HF_HOME:-/tmp/dlm-graph-hf-home}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/dlm-graph-hf-transformers}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"
# Note: do NOT override HF_DATASETS_CACHE -- DATASET_CONFIGS resolves
# tape_dir / ogb_root / etc. from it at import time, so a /tmp override
# would point ogbn-products tape_dir to a non-existent path.

cd "${REPO_ROOT}"

# (gpu_id, dataset)
declare -a JOBS=(
  "6 ogbn-arxiv"
  "7 ogbn-products"
)

run_one_gpu() {
  local gpu_id="$1" dataset="$2"
  local sum_dir="${SUMMARIES_ROOT}/${dataset//-/_}_llaga_frozen_eval_${TAG}"
  mkdir -p "${sum_dir}"
  local worker_log="${WORK_LOG_ROOT}/eval-frozen-${dataset}-gpu${gpu_id}.worker.log"
  : > "${worker_log}"

  for setting in notopo topo; do
    case "${setting}" in
      topo)   topo_bool=True  ;;
      notopo) topo_bool=False ;;
    esac

    local exp_name="${dataset//-/_}_llaga_frozen_${setting}_logit_labelon"
    local jsonl="${sum_dir}/${exp_name}.jsonl"
    local stdout="${sum_dir}/${exp_name}.out"

    {
      echo "[$(date '+%F %T')] [start] ${exp_name} gpu=${gpu_id}"
    } >> "${worker_log}"

    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    HF_HOME="${HF_HOME}" TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
        --exp "${exp_name}" \
        --model_name_or_path "${BASE_MODEL}" \
        --dataset_name "${dataset}" \
        --split "${SPLIT}" \
        --max_hops "${MAX_HOPS}" \
        --use_topology_mask "${topo_bool}" \
        --prompt_format "${PROMPT_FORMAT}" \
        --max_answer_tokens "${MAX_ANSWER_TOKENS}" \
        --include_neighbor_labels "${INCLUDE_NEIGHBOR_LABELS}" \
        --neighbor_label_format "${NEIGHBOR_LABEL_FORMAT}" \
        --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}" \
        --batch_size "${BATCH_SIZE}" \
        --log_file "${jsonl}" \
        > "${stdout}" 2>&1

    {
      echo "[$(date '+%F %T')] [done]  ${exp_name}"
    } >> "${worker_log}"
  done

  echo "[$(date '+%F %T')] [hold-gpu] starting sample_gen on gpu=${gpu_id}" >> "${worker_log}"
  nohup "${PYTHON_BIN}" "${SAMPLE_GEN}" start "${gpu_id}" >> "${worker_log}" 2>&1 < /dev/null &
  disown || true
}

declare -a PIDS=()
for j in "${JOBS[@]}"; do
  read -r gpu_id dataset <<< "${j}"
  echo "[plan] gpu=${gpu_id} dataset=${dataset}"
  ( run_one_gpu "${gpu_id}" "${dataset}" ) &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if wait "${pid}"; then
    echo "[done] worker pid=${pid}"
  else
    echo "[fail] worker pid=${pid}" >&2
    failed=1
  fi
done
exit "${failed}"
