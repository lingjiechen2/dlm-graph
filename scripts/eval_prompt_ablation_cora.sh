#!/bin/bash
# Prompt-format ablation: frozen eval_logit on Cora at nb=5, 1-hop.
# One sbatch = one (format) run (one job per run, per repo convention).
#
# Submit via:
#   for fmt in bracket paren sentence colon; do
#     sbatch --export=ALL,FMT=$fmt --job-name=prompt_abl_$fmt \
#       scripts/eval_prompt_ablation_cora.sh
#   done
#
#SBATCH --account=bffx-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=00:20:00
#SBATCH --output=.logs/%x-%j.out
#SBATCH --error=.logs/%x-%j.err

set -eo pipefail
source activate dllm
export HF_DATASETS_CACHE=/projects/bffx/lingjie7/datasets/huggingface
cd /u/lingjie7/dlm-graph
export PYTHONPATH=/u/lingjie7/dlm-graph:${PYTHONPATH:-}

: "${FMT:?FMT env var required (bracket|paren|sentence|colon)}"
MODEL=/projects/bffx/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct

echo "=============================================================="
echo "Prompt-ablation: nb=5, 1-hop, frozen, format=${FMT}"
echo "=============================================================="

python examples/tmdlm/eval_logit.py \
  --model_name_or_path "$MODEL" \
  --dataset_name cora \
  --split test \
  --batch_size 8 \
  --max_seq_len 2048 \
  --max_neighbors_per_hop 5 \
  --max_hops 1 \
  --seed 42 \
  --include_neighbor_labels True \
  --neighbor_label_format "${FMT}" \
  --exp "cora_nb5_1hop_label_${FMT}" \
  --log_file experiments/nb_labels_cora.jsonl

echo "[format=${FMT}] done."
