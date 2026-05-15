#!/usr/bin/env bash
set -euo pipefail

# §22 sanity-check launcher: same as §21 r128 but with FULL ogbn-arxiv train
# (no max_train_samples cap) and only 1 epoch. Boost spec unchanged from §21.
#
# Post-boost dataset size ≈ 115K samples → 2396 steps at eff_bs=48 → ~21h on
# 4-GPU DDP (2/3/4/6) at ~32 s/step.
#
# Usage:
#   GPUS=2,3,4,6 bash examples/tmdlm/run_sft_arxiv_4gpu_ddp_fulltrain_1ep.sh

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
SFT_SCRIPT="${REPO_ROOT}/examples/tmdlm/sft.py"
PYTHON_BIN="${PYTHON_BIN:-/home/lingjie7/anaconda3/envs/dllm/bin/python}"
SAMPLE_GEN="/home/lingjie7/sample_gen.py"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

GPUS="${GPUS:-2,3,4,6}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NPROC=${#GPU_LIST[@]}

DATASET_NAME=ogbn-arxiv
PROMPT_FORMAT=mc_digit
ANSWER_LABEL_STYLE=digit0_pad
MAX_ANSWER_TOKENS=2
MAX_STEPS="${MAX_STEPS:-2396}"          # 1 epoch at post-boost ~115K / eff_bs 48
MAX_TRAIN_SAMPLES=0                     # 0 = no cap, use full ~90,941 train
MAX_SEQ_LEN=4096
MAX_HOPS=2
MAX_NEIGHBORS_PER_HOP=10
POSITION_ID_TYPE=sequential
USE_TOPOLOGY_MASK=True

# Tolerate long single-process cache build during DDP barrier; default is 30 min.
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
OUTPUT_ROOT="${REPO_ROOT}/.models"
RUN_TAG="${RUN_TAG:-arxiv_20260511_fulltrain_r128_1ep}"

LEARNING_RATE=5e-5
PER_DEVICE_TRAIN_BATCH_SIZE=3
PER_DEVICE_EVAL_BATCH_SIZE=2
GRAD_ACCUM_STEPS=4                      # 3 * 4 * 4(world) = 48 eff bs
GRADIENT_CHECKPOINTING=True
CLS_LOSS_WEIGHT=0.0
SAVE_STEPS=0.1                          # 10 ckpts total
EVAL_STEPS=0.1
EVAL_STRATEGY=no
REPORT_TO=wandb

LORA_R=128
LORA_ALPHA=128
LORA_TARGET_MODULES=all-linear

RESAMPLE_STRATEGY=boost
BOOST_SPEC='ogbn-arxiv:cs.LG(Machine Learning):3,ogbn-arxiv:cs.AI(Artificial Intelligence):2,ogbn-arxiv:cs.NE(Neural and Evolutionary Computing):2'

run_name="tmdlm-llada-8b-${DATASET_NAME}-${MAX_HOPS}hop-topo-mcdigit-d0-nonb-r${LORA_R}-steps${MAX_STEPS}-${RUN_TAG}"
output_dir="${OUTPUT_ROOT}/${run_name}"
log_file="${OUTPUT_ROOT}/${run_name}.log"
master_port="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

# NOTE: sample_gen holds the GPUs during prewarm (CPU-only) and is killed
# right before torchrun to minimize idle-GPU window.
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

# Pre-build the train dataset cache BEFORE torchrun — building inside DDP's
# local_main_process_first() barrier on the full ~90K train set takes ~30+ min
# and triggers NCCL socket timeout on ranks 1-3. Building here single-process
# (no GPU needed) is safe and matches the cache_key the SFT will request.
echo "[prewarm] building full train cache (max_samples=${MAX_TRAIN_SAMPLES}, boost on)..."
"${PYTHON_BIN}" - <<EOF 2>&1 | tail -5
from dllm.data.graph import load_tag_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${BASE_MODEL}")
ds = load_tag_dataset(
    "${DATASET_NAME}", tokenizer=tok, split="train",
    max_seq_len=${MAX_SEQ_LEN}, max_neighbors_per_hop=${MAX_NEIGHBORS_PER_HOP},
    max_hops=${MAX_HOPS}, seed=42,
    max_answer_tokens=${MAX_ANSWER_TOKENS}, prompt_layout="target_first",
    use_chat_template=False, prompt_format="${PROMPT_FORMAT}",
    answer_label_style="${ANSWER_LABEL_STYLE}",
    include_neighbor_labels=False, neighbor_label_format="bracket",
    max_samples=${MAX_TRAIN_SAMPLES}, resample_strategy="${RESAMPLE_STRATEGY}",
    boost_spec="${BOOST_SPEC}",
)
print(f"[prewarm] full train cache ready: {len(ds)} samples")
EOF

# Kill sample_gen on our GPUs right before torchrun (was held during prewarm)
echo "[release] killing sample_gen on GPUs ${GPUS} before torchrun..."
for g in "${GPU_LIST[@]}"; do
  pkill -f "sample_gen.py start $g" 2>/dev/null || true
done
sleep 5  # give CUDA contexts time to release VRAM

CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${master_port}" \
  -- \
  "${SFT_SCRIPT}" \
    --model_name_or_path "${BASE_MODEL}" \
    --dataset_name "${DATASET_NAME}" \
    --max_hops "${MAX_HOPS}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --use_topology_mask "${USE_TOPOLOGY_MASK}" \
    --position_id_type "${POSITION_ID_TYPE}" \
    --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}" \
    --prompt_format "${PROMPT_FORMAT}" \
    --answer_label_style "${ANSWER_LABEL_STYLE}" \
    --max_answer_tokens "${MAX_ANSWER_TOKENS}" \
    --include_neighbor_labels False \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --resample_strategy "${RESAMPLE_STRATEGY}" \
    --boost_spec "${BOOST_SPEC}" \
    --output_dir "${output_dir}" \
    --max_steps "${MAX_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --cls_loss_weight "${CLS_LOSS_WEIGHT}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_strategy "${EVAL_STRATEGY}" \
    --report_to "${REPORT_TO}" \
    --run_name "${run_name}" \
    --overwrite_output_dir \
    --lora True \
    --r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --target_modules "${LORA_TARGET_MODULES}" \
    >"${log_file}" 2>&1
