#!/usr/bin/env bash
set -euo pipefail

# Shard cora LP checkpoint eval across multiple GPUs by reusing the validated
# single-GPU launcher. Each worker launcher handles releasing sample_gen before
# eval and restoring sample_gen on its GPU when it exits.
#
# Usage:
#   GPUS_CSV=2,3,4,6 \
#   MODEL_DIR=/home/lingjie7/auto-research/projects/dlm-graph/.models/<run> \
#   USE_TOPO=True \
#   bash /home/lingjie7/auto-research/projects/dlm-graph/examples/tmdlm/run_eval_cora_lp_llaga_ckpts_oneshot.sh

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
SINGLE_GPU_LAUNCHER="${REPO_ROOT}/examples/tmdlm/run_eval_cora_lp_llaga_ckpts.sh"

GPUS_CSV="${GPUS_CSV:-2,3,4,6}"
MODEL_DIR="${MODEL_DIR:-}"
USE_TOPO="${USE_TOPO:-}"
DATASET="${DATASET:-cora}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MAX_NEIGHBORS="${MAX_NEIGHBORS:-10}"
MAX_HOPS="${MAX_HOPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RUN_TAG="${RUN_TAG:-$(basename "${MODEL_DIR}")}"
SKIP_CHECKPOINTS="${SKIP_CHECKPOINTS:-}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/.logs}"
JSONL_ROOT="${JSONL_ROOT:-${REPO_ROOT}/.models/eval_logs}"

[[ -n "${MODEL_DIR}" ]] || { echo "ERROR: MODEL_DIR not set" >&2; exit 1; }
[[ -d "${MODEL_DIR}" ]] || { echo "ERROR: MODEL_DIR not found: ${MODEL_DIR}" >&2; exit 1; }
[[ -n "${USE_TOPO}" ]] || {
  echo "ERROR: USE_TOPO must be set explicitly to True or False" >&2
  exit 1
}

mkdir -p "${LOG_ROOT}" "${JSONL_ROOT}"

mapfile -t GPUS < <(tr ',' '\n' <<< "${GPUS_CSV}" | sed '/^$/d')
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "ERROR: no GPUs parsed from GPUS_CSV=${GPUS_CSV}" >&2
  exit 1
fi

mapfile -t CKPTS < <(
  find "${MODEL_DIR}" -maxdepth 1 -type d -regextype posix-extended \
    -regex '.*/checkpoint-[0-9]+' \
    | sort -t- -k2 -n \
    | xargs -n1 basename
)
if [[ -d "${MODEL_DIR}/checkpoint-final" ]]; then
  CKPTS+=("checkpoint-final")
fi

if [[ -n "${SKIP_CHECKPOINTS}" ]]; then
  IFS=',' read -r -a SKIPS <<< "${SKIP_CHECKPOINTS}"
  FILTERED=()
  for ckpt in "${CKPTS[@]}"; do
    skip_this=0
    for skip in "${SKIPS[@]}"; do
      if [[ "${ckpt}" == "${skip}" ]]; then
        skip_this=1
        break
      fi
    done
    [[ ${skip_this} -eq 0 ]] && FILTERED+=("${ckpt}")
  done
  CKPTS=("${FILTERED[@]}")
fi

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "ERROR: no checkpoints left to evaluate" >&2
  exit 1
fi

declare -a ASSIGNMENTS
for ((i=0; i<${#GPUS[@]}; i++)); do
  ASSIGNMENTS[i]=""
done

for ((i=0; i<${#CKPTS[@]}; i++)); do
  slot=$((i % ${#GPUS[@]}))
  if [[ -n "${ASSIGNMENTS[slot]}" ]]; then
    ASSIGNMENTS[slot]+=","
  fi
  ASSIGNMENTS[slot]+="${CKPTS[i]}"
done

declare -a PIDS=()
for ((i=0; i<${#GPUS[@]}; i++)); do
  gpu="${GPUS[i]}"
  filter="${ASSIGNMENTS[i]}"
  [[ -n "${filter}" ]] || continue

  worker_log="${LOG_ROOT}/eval_cora_lp_llaga_${RUN_TAG}_gpu${gpu}.log"
  worker_jsonl="${JSONL_ROOT}/eval_cora_lp_llaga_${RUN_TAG}_gpu${gpu}.jsonl"
  echo "[plan] gpu=${gpu} checkpoints=${filter}"

  (
    GPUS="${gpu}" \
    MODEL_DIR="${MODEL_DIR}" \
    CHECKPOINT_FILTER="${filter}" \
    USE_TOPO="${USE_TOPO}" \
    DATASET="${DATASET}" \
    MAX_SEQ_LEN="${MAX_SEQ_LEN}" \
    MAX_NEIGHBORS="${MAX_NEIGHBORS}" \
    MAX_HOPS="${MAX_HOPS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    LOG_FILE="${worker_jsonl}" \
    bash "${SINGLE_GPU_LAUNCHER}"
  ) > "${worker_log}" 2>&1 &
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
