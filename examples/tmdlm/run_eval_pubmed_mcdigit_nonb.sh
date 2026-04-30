#!/usr/bin/env bash
set -euo pipefail

# Eval for mc_digit SFT checkpoints trained WITHOUT neighbor labels (nonb runs).
# Evaluates tmdlm-llada-8b-pubmed-2hop-{topo,notopo}-mcdigit-d0-nonb-* checkpoints
# with include_neighbor_labels=False (fair comparison setting).
#
# Usage:
#   GPU_ID=6 bash run_eval_pubmed_mcdigit_nonb.sh
#
# Optional env overrides:
#   CKPTS="370 740 1110 1480"
#   MAX_SAMPLES=1000
#   BATCH_SIZE=8
#   INFILL_STEPS=10
#   RUN_TAG=pubmed_20260428_mcdigit_nonb

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
PYTHON_BIN="/home/lingjie7/anaconda3/envs/dllm/bin/python"
SAMPLE_GEN="/home/lingjie7/sample_gen.py"

EVAL_LOGIT="${REPO_ROOT}/examples/tmdlm/eval_logit.py"
EVAL_INFILL="${REPO_ROOT}/examples/tmdlm/eval_infill.py"

RUN_TAG="${RUN_TAG:-pubmed_20260428_mcdigit_nonb}"
DATASET=pubmed
SPLIT=test
BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
PROMPT_FORMAT=mc_digit
MAX_ANSWER_TOKENS=1
INCLUDE_NEIGHBOR_LABELS=False
MAX_NEIGHBORS_PER_HOP=10
MAX_HOPS=2
BATCH_SIZE="${BATCH_SIZE:-8}"
INFILL_STEPS="${INFILL_STEPS:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

read -r -a CKPTS <<< "${CKPTS:-370 740 1110 1480}"

WORK_LOG_ROOT="/tmp/dlm-graph-eval-logs"
JSONL_ROOT="/tmp/dlm-graph-eval-jsonl"
mkdir -p "${WORK_LOG_ROOT}" "${JSONL_ROOT}"

HF_HOME="${HF_HOME:-/tmp/dlm-graph-hf-home}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/dlm-graph-hf-transformers}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

PYTHONPATH="${REPO_ROOT}"
export PYTHONPATH HF_HOME TRANSFORMERS_CACHE

cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:?set GPU_ID=6 or 7}"

JOBS=(
  "logit  notopo False"
  "logit  topo   True"
  "infill notopo False"
  "infill topo   True"
)

WORKER_LOG="${WORK_LOG_ROOT}/eval-pubmed-mcdigit-nonb-gpu${GPU_ID}.worker.log"
: > "${WORKER_LOG}"

run_dir_for() {
  case "$1" in
    topo)   echo "${REPO_ROOT}/.models/tmdlm-llada-8b-pubmed-2hop-topo-mcdigit-d0-nonb-r64-ep10-${RUN_TAG}" ;;
    notopo) echo "${REPO_ROOT}/.models/tmdlm-llada-8b-pubmed-2hop-notopo-mcdigit-d0-nonb-r64-ep10-${RUN_TAG}" ;;
  esac
}

ts() { date '+%F %T'; }

echo "[$(ts)] [start] GPU=${GPU_ID} CKPTS=${CKPTS[*]} MAX_SAMPLES=${MAX_SAMPLES} include_neighbor_labels=False (nonb model)" >> "${WORKER_LOG}"

for job in "${JOBS[@]}"; do
  read -r kind setting topo_bool <<< "${job}"
  run_dir="$(run_dir_for "${setting}")"

  case "${kind}" in
    logit)
      script="${EVAL_LOGIT}"
      jsonl="${JSONL_ROOT}/eval-pubmed-2hop-${setting}-nonb-${RUN_TAG}.jsonl"
      extra_args=( --position_id_type sequential --max_samples "${MAX_SAMPLES}" )
      ;;
    infill)
      script="${EVAL_INFILL}"
      jsonl="${JSONL_ROOT}/eval-pubmed-2hop-${setting}-nonb-${RUN_TAG}-infill.jsonl"
      extra_args=( --steps "${INFILL_STEPS}" --max_samples "${MAX_SAMPLES}" )
      ;;
  esac

  echo "[$(ts)] [job-start] ${kind} ${setting} on gpu=${GPU_ID}" >> "${WORKER_LOG}"

  for step in "${CKPTS[@]}"; do
    ckpt="${run_dir}/checkpoint-${step}"
    exp="eval-pubmed-2hop-${setting}-nonb-${kind}-${RUN_TAG}-checkpoint-${step}"

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
