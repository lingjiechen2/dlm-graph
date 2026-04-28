#!/usr/bin/env bash
set -euo pipefail

# One-shot: split remaining cora 2-hop infill ckpts across 4 GPUs.
# After each GPU finishes its ckpt list, sample_gen.py occupies it.

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
EVAL_SCRIPT="${REPO_ROOT}/examples/tmdlm/eval_infill.py"
PYTHON_BIN="/home/lingjie7/anaconda3/envs/dllm/bin/python"
SAMPLE_GEN="/home/lingjie7/sample_gen.py"

RUN_TAG="${RUN_TAG:-llaga_20260427_aligned}"
DATASET_NAME="cora"
SPLIT="test"
BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
PROMPT_FORMAT="category_infill"
MAX_ANSWER_TOKENS=6
INCLUDE_NEIGHBOR_LABELS=True
NEIGHBOR_LABEL_FORMAT="bracket"
MAX_NEIGHBORS_PER_HOP=10
MAX_HOPS=2
BATCH_SIZE=8
INFILL_STEPS=10

WORK_LOG_ROOT="/tmp/dlm-graph-eval-logs"
JSONL_ROOT="/tmp/dlm-graph-eval-jsonl"
mkdir -p "${WORK_LOG_ROOT}" "${JSONL_ROOT}"

HF_HOME="/tmp/dlm-graph-hf-home"
HF_DATASETS_CACHE="/tmp/dlm-graph-hf-datasets"
TRANSFORMERS_CACHE="/tmp/dlm-graph-hf-transformers"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

cd "${REPO_ROOT}"

run_dir_topo="${REPO_ROOT}/.models/tmdlm-llada-8b-cora-2hop-topo-catinfill-nbmask-noeospad-r64-ep20-${RUN_TAG}"
run_dir_notopo="${REPO_ROOT}/.models/tmdlm-llada-8b-cora-2hop-notopo-catinfill-nbmask-noeospad-r64-ep20-${RUN_TAG}"

# Per-GPU jobs: "gpu_id setting topo_bool ckpt_step1,ckpt_step2,..."
declare -a JOBS=(
  "4 topo  True  408,612"
  "5 notopo False 612,816"
  "6 topo  True  816,1020"
  "7 notopo False 1020,1224"
)

run_one_gpu() {
  local gpu_id="$1" setting="$2" topo_bool="$3" ckpts_csv="$4"
  local run_dir
  if [[ "${setting}" == "topo" ]]; then
    run_dir="${run_dir_topo}"
  else
    run_dir="${run_dir_notopo}"
  fi

  local worker_log="${WORK_LOG_ROOT}/eval-cora-2hop-${setting}-${RUN_TAG}-infill-4gpu-gpu${gpu_id}.worker.log"
  local worker_jsonl="${JSONL_ROOT}/eval-cora-2hop-${setting}-${RUN_TAG}-infill.jsonl"

  : > "${worker_log}"
  IFS=',' read -r -a CKPTS <<< "${ckpts_csv}"
  for step in "${CKPTS[@]}"; do
    local ckpt_path="${run_dir}/checkpoint-${step}"
    local exp_name="eval-cora-2hop-${setting}-lora-${RUN_TAG}-infill-checkpoint-${step}"

    {
      echo "[$(date '+%F %T')] [start] ${exp_name} gpu=${gpu_id}"
      echo "[$(date '+%F %T')] [ckpt] ${ckpt_path}"
    } >> "${worker_log}"

    if [[ ! -f "${ckpt_path}/adapter_config.json" ]]; then
      echo "[$(date '+%F %T')] [skip] missing adapter_config.json at ${ckpt_path}" >> "${worker_log}"
      continue
    fi

    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    HF_HOME="${HF_HOME}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
        --exp "${exp_name}" \
        --model_name_or_path "${BASE_MODEL}" \
        --lora_path "${ckpt_path}" \
        --dataset_name "${DATASET_NAME}" \
        --split "${SPLIT}" \
        --max_hops "${MAX_HOPS}" \
        --use_topology_mask "${topo_bool}" \
        --prompt_format "${PROMPT_FORMAT}" \
        --max_answer_tokens "${MAX_ANSWER_TOKENS}" \
        --include_neighbor_labels "${INCLUDE_NEIGHBOR_LABELS}" \
        --neighbor_label_format "${NEIGHBOR_LABEL_FORMAT}" \
        --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}" \
        --batch_size "${BATCH_SIZE}" \
        --steps "${INFILL_STEPS}" \
        --log_file "${worker_jsonl}" \
        >> "${worker_log}" 2>&1
    echo "[$(date '+%F %T')] [done] ${exp_name} (step=${step})" >> "${worker_log}"
  done

  echo "[$(date '+%F %T')] [hold-gpu] starting sample_gen on gpu=${gpu_id}" >> "${worker_log}"
  nohup "${PYTHON_BIN}" "${SAMPLE_GEN}" start "${gpu_id}" >> "${worker_log}" 2>&1 < /dev/null &
  disown || true
}

declare -a PIDS=()
for j in "${JOBS[@]}"; do
  read -r gpu_id setting topo_bool ckpts_csv <<< "${j}"
  echo "[plan] gpu=${gpu_id} setting=${setting} topo_bool=${topo_bool} ckpts=${ckpts_csv}"
  ( run_one_gpu "${gpu_id}" "${setting}" "${topo_bool}" "${ckpts_csv}" ) &
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
