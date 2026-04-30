#!/usr/bin/env bash
set -euo pipefail

# Eval for ogbn-arxiv catinfill SFT checkpoints (topo, with neighbor labels).
# Auto-discovers all checkpoints in the run directory.
#
# Usage:
#   GPU_ID=6 bash run_eval_arxiv_catinfill.sh
#
# Optional env overrides:
#   MAX_SAMPLES=1000
#   BATCH_SIZE=8
#   INFILL_STEPS=10
#   RUN_TAG=arxiv_20260429
#   INCLUDE_NEIGHBOR_LABELS=True

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
PYTHON_BIN="/home/lingjie7/anaconda3/envs/dllm/bin/python"
SAMPLE_GEN="/home/lingjie7/sample_gen.py"

EVAL_LOGIT="${REPO_ROOT}/examples/tmdlm/eval_logit.py"
EVAL_INFILL="${REPO_ROOT}/examples/tmdlm/eval_infill.py"

RUN_TAG="${RUN_TAG:-arxiv_20260429}"
DATASET=ogbn-arxiv
SPLIT=test
BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
PROMPT_FORMAT=category_infill
MAX_ANSWER_TOKENS=10
INCLUDE_NEIGHBOR_LABELS="${INCLUDE_NEIGHBOR_LABELS:-True}"
NEIGHBOR_LABEL_FORMAT=bracket
MAX_NEIGHBORS_PER_HOP=10
MAX_HOPS=2
BATCH_SIZE="${BATCH_SIZE:-8}"
INFILL_STEPS="${INFILL_STEPS:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

WORK_LOG_ROOT="/tmp/dlm-graph-eval-logs"
JSONL_ROOT="/tmp/dlm-graph-eval-jsonl"
mkdir -p "${WORK_LOG_ROOT}" "${JSONL_ROOT}"

HF_HOME="${HF_HOME:-/tmp/dlm-graph-hf-home}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/dlm-graph-hf-transformers}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

PYTHONPATH="${REPO_ROOT}"
export PYTHONPATH HF_HOME TRANSFORMERS_CACHE

cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:?set GPU_ID=6}"

RUN_DIR="${REPO_ROOT}/.models/tmdlm-llada-8b-${DATASET}-2hop-topo-catinfill-nbmask-r64-steps7400-${RUN_TAG}"

get_checkpoints_sorted() {
  find -L "$1" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | awk -F- '/checkpoint-[0-9]+$/ {print $NF" "$0}' \
    | sort -n \
    | awk '{print $2}'
}

JOBS=(
  "logit  topo True"
  "infill topo True"
)

WORKER_LOG="${WORK_LOG_ROOT}/eval-arxiv-catinfill-gpu${GPU_ID}.worker.log"
: > "${WORKER_LOG}"

ts() { date '+%F %T'; }

echo "[$(ts)] [start] GPU=${GPU_ID} MAX_SAMPLES=${MAX_SAMPLES} include_neighbor_labels=${INCLUDE_NEIGHBOR_LABELS}" >> "${WORKER_LOG}"

for job in "${JOBS[@]}"; do
  read -r kind setting topo_bool <<< "${job}"

  case "${kind}" in
    logit)
      script="${EVAL_LOGIT}"
      jsonl="${JSONL_ROOT}/eval-arxiv-2hop-topo-catinfill-${RUN_TAG}.jsonl"
      extra_args=( --position_id_type sequential --max_samples "${MAX_SAMPLES}" )
      ;;
    infill)
      script="${EVAL_INFILL}"
      jsonl="${JSONL_ROOT}/eval-arxiv-2hop-topo-catinfill-${RUN_TAG}-infill.jsonl"
      extra_args=( --steps "${INFILL_STEPS}" --max_samples "${MAX_SAMPLES}" )
      ;;
  esac

  echo "[$(ts)] [job-start] ${kind} ${setting} on gpu=${GPU_ID}" >> "${WORKER_LOG}"

  mapfile -t ckpt_paths < <(get_checkpoints_sorted "${RUN_DIR}")

  if [[ ${#ckpt_paths[@]} -eq 0 ]]; then
    echo "[$(ts)] [skip] no checkpoints found in ${RUN_DIR}" >> "${WORKER_LOG}"
    continue
  fi

  for ckpt in "${ckpt_paths[@]}"; do
    step="$(basename "${ckpt}" | sed 's/checkpoint-//')"
    exp="eval-arxiv-2hop-topo-catinfill-${kind}-${RUN_TAG}-checkpoint-${step}"

    if [[ ! -f "${ckpt}/adapter_config.json" ]]; then
      echo "[$(ts)] [skip] missing ${ckpt}/adapter_config.json" >> "${WORKER_LOG}"
      continue
    fi

    echo "[$(ts)] [start] ${exp}" >> "${WORKER_LOG}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      "${PYTHON_BIN}" "${script}" \
        --exp "${exp}" \
        --model_name_or_path "${BASE_MODEL}" \
        --lora_path "${ckpt}" \
        --dataset_name "${DATASET}" \
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
        "${extra_args[@]}" \
        >> "${WORKER_LOG}" 2>&1

    echo "[$(ts)] [done]  ${exp}" >> "${WORKER_LOG}"
  done

  echo "[$(ts)] [job-done] ${kind} ${setting}" >> "${WORKER_LOG}"
done

echo "[$(ts)] [hold-gpu] starting sample_gen on gpu=${GPU_ID}" >> "${WORKER_LOG}"
nohup "${PYTHON_BIN}" "${SAMPLE_GEN}" start "${GPU_ID}" >> "${WORKER_LOG}" 2>&1 < /dev/null &
disown || true
