#!/bin/bash
# Frozen eval_logit on Cora, sweeping max_neighbors_per_hop.
# Each sbatch array task runs one nb value (both flag off and on back-to-back).
#
# Usage:
#   sbatch --array=1,3,10,20 scripts/eval_nb_labels_cora_sweep.sh
#
# Notes:
#   - 1-hop only, no topology mask, frozen LLaDA-8B-Instruct.
#   - Modest resources: 1 GPU, 8 CPUs, 48G RAM (per job).
#
#SBATCH --account=bffx-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=00:45:00
#SBATCH --job-name=nb_labels_cora_sweep
#SBATCH --output=.logs/%x-nb%a-%j.out
#SBATCH --error=.logs/%x-nb%a-%j.err

set -eo pipefail
source activate dllm
export HF_DATASETS_CACHE=/projects/bffx/lingjie7/datasets/huggingface
cd /u/lingjie7/dlm-graph
export PYTHONPATH=/u/lingjie7/dlm-graph:${PYTHONPATH:-}

NB=${SLURM_ARRAY_TASK_ID}
MODEL=/projects/bffx/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct

COMMON=(
  --model_name_or_path "$MODEL"
  --dataset_name cora
  --split test
  --batch_size 8
  --max_seq_len 2048
  --max_neighbors_per_hop "$NB"
  --max_hops 1
  --seed 42
  --log_file experiments/nb_labels_cora.jsonl
)

echo "=============================================================="
echo "[nb=$NB] Run A: include_neighbor_labels = False"
echo "=============================================================="
python examples/tmdlm/eval_logit.py \
  "${COMMON[@]}" \
  --exp cora_nb${NB}_1hop_nolabel \
  --include_neighbor_labels false

echo
echo "=============================================================="
echo "[nb=$NB] Run B: include_neighbor_labels = True"
echo "=============================================================="
python examples/tmdlm/eval_logit.py \
  "${COMMON[@]}" \
  --exp cora_nb${NB}_1hop_label \
  --include_neighbor_labels true

echo "[nb=$NB] all runs complete."
