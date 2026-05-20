#!/usr/bin/env bash
set -euo pipefail

# 4-GPU DDP SFT for cora link prediction at max_seq_len=4096.
# 8976 train pos+neg pairs × 10 ep / eff_batch 32 ≈ 2805 steps.
# Estimated ~10 hours on 4× A6000 (per_device_batch=2, grad_accum=4).
#
# Usage:
#   GPUS=2,3,4,6 bash examples/tmdlm/run_sft_cora_lp_4gpu_ddp.sh

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
SFT_SCRIPT="${REPO_ROOT}/examples/tmdlm/sft.py"
PYTHON_BIN="${PYTHON_BIN:-/home/lingjie7/anaconda3/envs/dllm/bin/python}"
SAMPLE_GEN="/home/lingjie7/sample_gen.py"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

GPUS="${GPUS:-2,3,4,6}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NPROC=${#GPU_LIST[@]}

DATASET_NAME=cora
MAX_SEQ_LEN=4096
MAX_HOPS=2
MAX_NEIGHBORS_PER_HOP=10
LP_NEG_RATIO=1
USE_TOPOLOGY_MASK="${USE_TOPOLOGY_MASK:-True}"
LP_POS_WEIGHT="${LP_POS_WEIGHT:-1.0}"
POSITION_ID_TYPE=sequential

BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
OUTPUT_ROOT="${REPO_ROOT}/.models"
RUN_TAG="${RUN_TAG:-cora_lp_$(date +%Y%m%d)_seq4k_4gpu}"

NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE=5e-5
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=2
GRAD_ACCUM_STEPS=4                  # eff_batch = 2 * 4 * 4 = 32
GRADIENT_CHECKPOINTING=True
SAVE_STEPS=0.1                      # ~10 ckpts
EVAL_STRATEGY=no                    # avoid prior HF eval-hang issue
REPORT_TO=wandb

LORA_R=64
LORA_ALPHA=64
LORA_TARGET_MODULES=all-linear

# NCCL / DDP timeouts (matches §22 fulltrain pattern)
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

run_name="tmdlm-llada-8b-${DATASET_NAME}-lp-${MAX_HOPS}hop-r${LORA_R}-ep${NUM_EPOCHS}-${RUN_TAG}"
output_dir="${OUTPUT_ROOT}/${run_name}"
log_file="${OUTPUT_ROOT}/${run_name}.log"
master_port="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

reclaim_gpus() {
  echo "[trap] launching sample_gen on GPUs ${GPUS}" >&2
  for g in "${GPU_LIST[@]}"; do
    nohup "${PYTHON_BIN}" "${SAMPLE_GEN}" start "${g}" \
      >> "${OUTPUT_ROOT}/sample_gen_gpu${g}.log" 2>&1 &
    disown || true
  done
}
trap reclaim_gpus EXIT

echo "[launch] run_name=${run_name}"
echo "[launch] GPUs=${GPUS} nproc=${NPROC} master_port=${master_port}"
echo "[log]    ${log_file}"

# Prewarm: build the LP edge-split cache single-process. cora_lp.load() is
# cheap on cora (~30s) but doing it from all 4 ranks risks a torch.save race
# on lp_split_*.pt. Single-process prewarm avoids that.
echo "[prewarm] building cora LP train split cache..."
"${PYTHON_BIN}" - <<EOF 2>&1 | tail -3
from dllm.data.graph import load_lp_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${BASE_MODEL}")
ds = load_lp_dataset(
    "${DATASET_NAME}", tokenizer=tok, split="train",
    max_seq_len=${MAX_SEQ_LEN}, max_neighbors_per_hop=${MAX_NEIGHBORS_PER_HOP},
    max_hops=${MAX_HOPS}, seed=42, neg_ratio=${LP_NEG_RATIO},
)
print(f"[prewarm] LP train cache ready: {len(ds)} samples")
EOF

# Kill sample_gen on our GPUs before torchrun
echo "[release] killing sample_gen on GPUs ${GPUS} ..."
for g in "${GPU_LIST[@]}"; do
  pkill -f "sample_gen.py start $g" 2>/dev/null || true
done
sleep 5

CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${master_port}" \
  -- \
  "${SFT_SCRIPT}" \
    --task lp \
    --model_name_or_path "${BASE_MODEL}" \
    --dataset_name "${DATASET_NAME}" \
    --lp_neg_ratio "${LP_NEG_RATIO}" \
    --max_hops "${MAX_HOPS}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --use_topology_mask "${USE_TOPOLOGY_MASK}" \
    --position_id_type "${POSITION_ID_TYPE}" \
    --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}" \
    --output_dir "${output_dir}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_strategy "${EVAL_STRATEGY}" \
    --report_to "${REPORT_TO}" \
    --run_name "${run_name}" \
    --overwrite_output_dir \
    --lora True \
    --r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --target_modules "${LORA_TARGET_MODULES}" \
    --lp_pos_weight "${LP_POS_WEIGHT}" \
    >"${log_file}" 2>&1
