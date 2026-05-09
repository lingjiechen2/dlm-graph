#!/usr/bin/env bash
set -euo pipefail

# §21 launcher: 4-GPU DDP, ogbn-arxiv 2hop topo, mc_digit / digit0_pad / r=128
# / boost cs.LG x3, cs.AI x2, cs.NE x2 / no neighbor labels / seq_len=4096.
#
# Usage:
#   GPUS=2,3,4,6 bash examples/tmdlm/run_sft_arxiv_4gpu_ddp_lgboost_r128.sh
#
# Default GPUs: 2,3,4,6. Auto-claims same GPUs with sample_gen on exit.

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
MAX_STEPS="${MAX_STEPS:-2042}"          # 4 epochs at post-boost ~24500 / eff_bs 48
MAX_TRAIN_SAMPLES=20000                 # pre-boost cap
MAX_SEQ_LEN=4096
MAX_HOPS=2
MAX_NEIGHBORS_PER_HOP=10
POSITION_ID_TYPE=sequential
USE_TOPOLOGY_MASK=True

BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
OUTPUT_ROOT="${REPO_ROOT}/.models"
RUN_TAG="${RUN_TAG:-arxiv_20260506_digit0pad_lgboost_r128}"

LEARNING_RATE=5e-5
PER_DEVICE_TRAIN_BATCH_SIZE=3
PER_DEVICE_EVAL_BATCH_SIZE=2
GRAD_ACCUM_STEPS=4                      # 3 * 4 * 4(world) = 48 eff bs
GRADIENT_CHECKPOINTING=True
CLS_LOSS_WEIGHT=0.0
SAVE_STEPS=0.1                          # 10 ckpts total
EVAL_STEPS=0.1
EVAL_STRATEGY=no                        # SFT log shows no eval was run
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
    --save_only_model True \
    --eval_strategy "${EVAL_STRATEGY}" \
    --report_to "${REPORT_TO}" \
    --run_name "${run_name}" \
    --overwrite_output_dir \
    --lora True \
    --r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --target_modules "${LORA_TARGET_MODULES}" \
    >"${log_file}" 2>&1
