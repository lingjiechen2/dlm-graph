#!/usr/bin/env bash
set -euo pipefail

# TM-DLM SFT on Cora, mc_digit, nonb, max_seq_len=4096, ALIGNED to §1 effective batch 32.
# Same recipe as run_sft_cora_mcdigit_nonb_lora_seq4k.sh except per_device_train_batch=2
# (halved further from 3) so effective batch = 2*16 = 32 matches §1 (seq=2048: 4*8=32).
# This restores the 510 total optimizer steps that §1 had (vs 340 in the unaligned seq4k).
# Supports SETTING={both|topo|notopo} so a single GPU can run one setting.
#
# Usage:
#   GPUS=4,5             bash examples/tmdlm/run_sft_cora_mcdigit_nonb_lora_seq4k_aligned.sh
#   GPUS=4 SETTING=topo  bash examples/tmdlm/run_sft_cora_mcdigit_nonb_lora_seq4k_aligned.sh
#   GPUS=5 SETTING=notopo bash examples/tmdlm/run_sft_cora_mcdigit_nonb_lora_seq4k_aligned.sh

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
SFT_SCRIPT="${REPO_ROOT}/examples/tmdlm/sft.py"
PYTHON_BIN="${PYTHON_BIN:-/home/lingjie7/anaconda3/envs/dllm/bin/python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="$(command -v python)"
[[ -f "${SFT_SCRIPT}" ]] || { echo "SFT script not found: ${SFT_SCRIPT}" >&2; exit 1; }

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-4,5}"
SETTING="${SETTING:-both}"  # both | topo | notopo
case "${SETTING}" in
  both)
    [[ ${#GPU_LIST[@]} -ge 2 ]] || { echo "SETTING=both needs >=2 GPUs (e.g. GPUS=4,5)" >&2; exit 2; }
    ;;
  topo|notopo)
    [[ ${#GPU_LIST[@]} -ge 1 ]] || { echo "SETTING=${SETTING} needs >=1 GPU" >&2; exit 2; }
    ;;
  *) echo "SETTING must be one of: both|topo|notopo" >&2; exit 2 ;;
esac

DATASET_NAME=cora
PROMPT_FORMAT="${PROMPT_FORMAT:-mc_digit}"
ANSWER_LABEL_STYLE="${ANSWER_LABEL_STYLE:-digit0}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-1}"  # 7 classes → labels "0".."6", 1 token each
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"

BASE_MODEL="${BASE_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/.models}"
RUN_TAG="${RUN_TAG:-cora_$(date +%Y%m%d)_mcdigit_nonb_seq4k_aligned}"

LEARNING_RATE="${LEARNING_RATE:-5e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"  # aligned: 2*16=32 effective batch (matches §1 seq=2048: 4*8=32)
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"  # 2*16=32 → 510 total steps (vs 340 in unaligned seq4k)
MAX_NEIGHBORS_PER_HOP="${MAX_NEIGHBORS_PER_HOP:-10}"
MAX_HOPS="${MAX_HOPS:-2}"
POSITION_ID_TYPE="${POSITION_ID_TYPE:-sequential}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-0.0}"
SAVE_STEPS="${SAVE_STEPS:-0.05}"
EVAL_STEPS="${EVAL_STEPS:-0.1}"
REPORT_TO="${REPORT_TO:-wandb}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

declare -a EXPERIMENTS=()
case "${SETTING}" in
  both)   EXPERIMENTS=("True topo" "False notopo") ;;
  topo)   EXPERIMENTS=("True topo") ;;
  notopo) EXPERIMENTS=("False notopo") ;;
esac

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

declare -a PIDS=() JOB_NAMES=()
cleanup() {
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "Stopping child jobs: ${PIDS[*]}" >&2
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for i in "${!EXPERIMENTS[@]}"; do
  read -r use_topology_mask topo_name <<< "${EXPERIMENTS[$i]}"
  gpu_id="${GPU_LIST[$i]}"

  run_name="tmdlm-llada-8b-${DATASET_NAME}-${MAX_HOPS}hop-${topo_name}-mcdigit-d0-nonb-r${LORA_R}-ep${NUM_EPOCHS}-${RUN_TAG}"
  output_dir="${OUTPUT_ROOT}/${run_name}"
  log_file="${OUTPUT_ROOT}/${run_name}.log"

  cmd=(
    "${PYTHON_BIN}" "${SFT_SCRIPT}"
    --model_name_or_path "${BASE_MODEL}"
    --dataset_name "${DATASET_NAME}"
    --max_hops "${MAX_HOPS}"
    --max_seq_len "${MAX_SEQ_LEN}"
    --use_topology_mask "${use_topology_mask}"
    --position_id_type "${POSITION_ID_TYPE}"
    --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}"
    --prompt_format "${PROMPT_FORMAT}"
    --answer_label_style "${ANSWER_LABEL_STYLE}"
    --max_answer_tokens "${MAX_ANSWER_TOKENS}"
    --include_neighbor_labels False
    --output_dir "${output_dir}"
    --num_train_epochs "${NUM_EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --cls_loss_weight "${CLS_LOSS_WEIGHT}"
    --save_steps "${SAVE_STEPS}"
    --eval_steps "${EVAL_STEPS}"
    --report_to "${REPORT_TO}"
    --run_name "${run_name}"
    --overwrite_output_dir
    --lora True
    --r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --target_modules "${LORA_TARGET_MODULES}"
  )

  echo "[launch] dataset=${DATASET_NAME} format=${PROMPT_FORMAT} gpu=${gpu_id} topo=${use_topology_mask} epochs=${NUM_EPOCHS} seq=${MAX_SEQ_LEN} -> ${output_dir}"
  echo "[log] ${log_file}"

  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}"
  ) >"${log_file}" 2>&1 &

  PIDS+=("$!")
  JOB_NAMES+=("${run_name}")
done

failed=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  job="${JOB_NAMES[$i]}"
  gpu="${GPU_LIST[$i]}"
  if wait "${pid}"; then
    echo "[done] ${job} (pid=${pid})"
  else
    echo "[fail] ${job} (pid=${pid})" >&2
    failed=1
  fi
  # Hold GPU immediately after this run finishes (success or fail) so it's not
  # snatched by another user before any follow-up eval can grab it.
  echo "[hold-gpu] launching sample_gen on gpu=${gpu}"
  nohup "${PYTHON_BIN}" /home/lingjie7/sample_gen.py start "${gpu}" \
    > "${OUTPUT_ROOT}/sample_gen_gpu${gpu}.log" 2>&1 &
  disown || true
done

exit "${failed}"
