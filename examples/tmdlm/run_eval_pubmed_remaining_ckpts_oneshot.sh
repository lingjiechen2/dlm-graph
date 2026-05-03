#!/usr/bin/env bash
set -euo pipefail

# One-shot PubMed eval over the 2 remaining ckpts (1850 / 2220) of the
# category_infill SFT (run_tag=pubmed_20260428_aligned). Mirrors the
# completed run_eval_pubmed_existing_ckpts_oneshot.sh which covered
# ckpts 370/740/1110/1480.
#
# Usage:
#   bash run_eval_pubmed_remaining_ckpts_oneshot.sh                  # all 8 jobs on GPU 7 (default)
#   GPU_ID=N bash run_eval_pubmed_remaining_ckpts_oneshot.sh         # all 8 jobs on GPU N
#
# Runs every (kind, setting) combination on a single GPU, sequentially:
#   logit topo, logit notopo, infill topo, infill notopo
# for each ckpt in CKPTS. Estimated wallclock on GPU 7 ~13.5h
#   (logit topo 2*16min + logit notopo 2*34min + infill topo 2*3.5h + infill notopo 2*2.5h).
# After all jobs finish, sample_gen.py is launched on the GPU to hold it.

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
PYTHON_BIN="/home/lingjie7/anaconda3/envs/dllm/bin/python"
SAMPLE_GEN="/home/lingjie7/sample_gen.py"

EVAL_LOGIT="${REPO_ROOT}/examples/tmdlm/eval_logit.py"
EVAL_INFILL="${REPO_ROOT}/examples/tmdlm/eval_infill.py"

RUN_TAG="${RUN_TAG:-pubmed_20260428_aligned}"
DATASET=pubmed
SPLIT=test
BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
PROMPT_FORMAT=category_infill
MAX_ANSWER_TOKENS=6
INCLUDE_NEIGHBOR_LABELS=True
NEIGHBOR_LABEL_FORMAT=bracket
MAX_NEIGHBORS_PER_HOP=10
MAX_HOPS=2
BATCH_SIZE="${BATCH_SIZE:-8}"
INFILL_STEPS="${INFILL_STEPS:-10}"

# Snapshot of remaining ckpts to evaluate.
CKPTS=(1850 2220)

WORK_LOG_ROOT="/tmp/dlm-graph-eval-logs"
JSONL_ROOT="/tmp/dlm-graph-eval-jsonl"
mkdir -p "${WORK_LOG_ROOT}" "${JSONL_ROOT}"

HF_HOME="${HF_HOME:-/tmp/dlm-graph-hf-home}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/dlm-graph-hf-transformers}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-7}"
declare -a JOBS=(
  "logit  topo   True"
  "logit  notopo False"
  "infill topo   True"
  "infill notopo False"
)

WORKER_LOG="${WORK_LOG_ROOT}/eval-pubmed-remaining-gpu${GPU_ID}.worker.log"
: > "${WORKER_LOG}"

run_dir_for() {
  case "$1" in
    topo)   echo "${REPO_ROOT}/.models/tmdlm-llada-8b-pubmed-2hop-topo-catinfill-nbmask-noeospad-r64-ep10-${RUN_TAG}" ;;
    notopo) echo "${REPO_ROOT}/.models/tmdlm-llada-8b-pubmed-2hop-notopo-catinfill-nbmask-noeospad-r64-ep10-${RUN_TAG}" ;;
  esac
}

ts() { date '+%F %T'; }

for job in "${JOBS[@]}"; do
  read -r kind setting topo_bool <<< "${job}"
  run_dir="$(run_dir_for "${setting}")"

  case "${kind}" in
    logit)
      script="${EVAL_LOGIT}"
      jsonl="${JSONL_ROOT}/eval-pubmed-2hop-${setting}-${RUN_TAG}.jsonl"
      extra_args=( --position_id_type sequential )
      ;;
    infill)
      script="${EVAL_INFILL}"
      jsonl="${JSONL_ROOT}/eval-pubmed-2hop-${setting}-${RUN_TAG}-infill.jsonl"
      extra_args=( --steps "${INFILL_STEPS}" )
      ;;
  esac

  echo "[$(ts)] [job-start] ${kind} ${setting} on gpu=${GPU_ID}" >> "${WORKER_LOG}"

  for step in "${CKPTS[@]}"; do
    ckpt="${run_dir}/checkpoint-${step}"
    exp="eval-pubmed-2hop-${setting}-${kind}-${RUN_TAG}-checkpoint-${step}"

    echo "[$(ts)] [start] ${exp}" >> "${WORKER_LOG}"
    if [[ ! -f "${ckpt}/adapter_config.json" ]]; then
      echo "[$(ts)] [skip] missing adapter_config.json at ${ckpt}" >> "${WORKER_LOG}"
      continue
    fi

    PYTHONPATH="${REPO_ROOT}" \
    HF_HOME="${HF_HOME}" TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
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
