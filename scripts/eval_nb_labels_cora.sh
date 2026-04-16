#!/bin/bash
#SBATCH --account=bffx-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=01:00:00
#SBATCH --job-name=nb_labels_cora
#SBATCH --output=.logs/%x-%j.out
#SBATCH --error=.logs/%x-%j.err

set -eo pipefail
source activate dllm
export HF_DATASETS_CACHE=/projects/bffx/lingjie7/datasets/huggingface
cd /u/lingjie7/dlm-graph
# Ensure local dllm (with data.graph) wins over the pip-installed copy at /u/lingjie7/dllm
export PYTHONPATH=/u/lingjie7/dlm-graph:${PYTHONPATH:-}

MODEL=/projects/bffx/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct
COMMON=(
  --model_name_or_path "$MODEL"
  --dataset_name cora
  --split test
  --batch_size 8
  --max_seq_len 2048
  --max_neighbors_per_hop 5
  --max_hops 1
  --seed 42
  --log_file experiments/nb_labels_cora.jsonl
)

echo "=============================================================="
echo "[Run A] include_neighbor_labels = False"
echo "=============================================================="
python examples/tmdlm/eval_logit.py \
  "${COMMON[@]}" \
  --exp cora_nb5_1hop_nolabel \
  --include_neighbor_labels false

echo
echo "=============================================================="
echo "[Run B] include_neighbor_labels = True"
echo "=============================================================="
python examples/tmdlm/eval_logit.py \
  "${COMMON[@]}" \
  --exp cora_nb5_1hop_label \
  --include_neighbor_labels true

echo "All runs complete."
