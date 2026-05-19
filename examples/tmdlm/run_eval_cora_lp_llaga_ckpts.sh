#!/usr/bin/env bash
set -euo pipefail
# Eval all checkpoints in a cora LP model directory against LLaGA's test split.
#
# Usage:
#   GPUS=0 MODEL_DIR=.models/<run> bash examples/tmdlm/run_eval_cora_lp_llaga_ckpts.sh
#
# Results are appended to .models/eval_logs/lp_llaga_split.jsonl
# A per-run summary table is printed at the end.
# After eval finishes, sample_gen is re-launched on the same GPU.

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
EVAL_SCRIPT="${REPO_ROOT}/examples/tmdlm/eval_lp_llaga_split.py"
PYTHON_BIN="${PYTHON_BIN:-/home/lingjie7/anaconda3/envs/dllm/bin/python}"
SAMPLE_GEN="/home/lingjie7/sample_gen.py"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

GPU="${GPUS:-0}"
MODEL_DIR="${MODEL_DIR:-}"
DATASET="${DATASET:-cora}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MAX_NEIGHBORS="${MAX_NEIGHBORS:-10}"
MAX_HOPS="${MAX_HOPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
USE_TOPO="${USE_TOPO:-}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/.models/eval_logs/lp_llaga_split.jsonl}"
CHECKPOINT_FILTER="${CHECKPOINT_FILTER:-}"

[[ -z "${MODEL_DIR}" ]] && { echo "ERROR: MODEL_DIR not set" >&2; exit 1; }
[[ -d "${MODEL_DIR}" ]] || { echo "ERROR: MODEL_DIR not found: ${MODEL_DIR}" >&2; exit 1; }
[[ -n "${USE_TOPO}" ]] || {
  echo "ERROR: USE_TOPO must be set explicitly to True or False" >&2
  exit 1
}

mkdir -p "$(dirname "${LOG_FILE}")"

cleanup() {
  echo "[hold-gpu] starting sample_gen on gpu=${GPU}"
  nohup "${PYTHON_BIN}" "${SAMPLE_GEN}" start "${GPU}" \
    >> "${REPO_ROOT}/.models/sample_gen_gpu${GPU}.log" 2>&1 < /dev/null &
  disown || true
}
trap cleanup EXIT

echo "[release] killing sample_gen on gpu=${GPU} before eval ..."
pkill -f "sample_gen.py start ${GPU}" 2>/dev/null || true

# Collect numeric checkpoints only, then append checkpoint-final once.
mapfile -t CKPTS < <(
  find "${MODEL_DIR}" -maxdepth 1 -type d -regextype posix-extended \
    -regex '.*/checkpoint-[0-9]+' \
    | sort -t- -k2 -n
)
if [[ -d "${MODEL_DIR}/checkpoint-final" ]]; then
  CKPTS+=("${MODEL_DIR}/checkpoint-final")
fi

if [[ -n "${CHECKPOINT_FILTER}" ]]; then
  IFS=',' read -r -a REQUESTED <<< "${CHECKPOINT_FILTER}"
  FILTERED=()
  for wanted in "${REQUESTED[@]}"; do
    for ckpt in "${CKPTS[@]}"; do
      if [[ "$(basename "${ckpt}")" == "${wanted}" ]]; then
        FILTERED+=("${ckpt}")
        break
      fi
    done
  done
  CKPTS=("${FILTERED[@]}")
fi

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "No checkpoints found in ${MODEL_DIR}" >&2
  exit 1
fi

echo "Found ${#CKPTS[@]} checkpoint(s) in ${MODEL_DIR}"
echo "Dataset: ${DATASET} | GPU: ${GPU} | max_seq_len: ${MAX_SEQ_LEN} | topo: ${USE_TOPO}"
echo ""

RUN_NAME="$(basename "${MODEL_DIR}")"
declare -a RESULTS=()

for CKPT in "${CKPTS[@]}"; do
  CKPT_NAME="$(basename "${CKPT}")"
  STEP="${CKPT_NAME#checkpoint-}"
  EXP="${RUN_NAME}__${CKPT_NAME}"

  echo "[eval] ${CKPT_NAME} ..."

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --exp "${EXP}" \
    --dataset_name "${DATASET}" \
    --lora_path "${CKPT}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --max_neighbors_per_hop "${MAX_NEIGHBORS}" \
    --max_hops "${MAX_HOPS}" \
    --batch_size "${BATCH_SIZE}" \
    --use_topology_mask "${USE_TOPO}" \
    --log_file "${LOG_FILE}"

  # Extract last logged accuracy for this exp from the log file.
  ACC=$(grep "\"exp\": \"${EXP}\"" "${LOG_FILE}" 2>/dev/null \
    | tail -1 \
    | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('accuracy','?'))" 2>/dev/null || echo "?")
  RESULTS+=("${CKPT_NAME}  acc=${ACC}%")
  echo "[done] ${CKPT_NAME}  acc=${ACC}%"
  echo ""
done

echo "========================================"
echo "Summary: ${RUN_NAME}"
echo "LLaGA test split (${DATASET})"
echo "----------------------------------------"
for r in "${RESULTS[@]}"; do echo "  ${r}"; done
echo "========================================"
echo "Full results: ${LOG_FILE}"
