#!/bin/bash
# Prompt-format ablation: frozen eval_logit on PubMed at nb=5, 1-hop.
# One sbatch = one (format) run.
#
# Submit via:
#   sbatch --export=ALL,MODE=nolabel --job-name=prompt_abl_pm_nolabel \
#     scripts/eval_prompt_ablation_pubmed.sh
#   for fmt in bracket paren sentence colon; do
#     sbatch --export=ALL,MODE=label,FMT=$fmt --job-name=prompt_abl_pm_$fmt \
#       scripts/eval_prompt_ablation_pubmed.sh
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
#SBATCH --time=00:30:00
#SBATCH --output=.logs/%x-%j.out
#SBATCH --error=.logs/%x-%j.err

set -eo pipefail
source activate dllm
export HF_DATASETS_CACHE=/projects/bffx/lingjie7/datasets/huggingface
cd /u/lingjie7/dlm-graph
export PYTHONPATH=/u/lingjie7/dlm-graph:${PYTHONPATH:-}

: "${MODE:?MODE env var required (nolabel|label)}"
MODEL=/projects/bffx/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct

if [[ "$MODE" == "nolabel" ]]; then
  EXP="pubmed_nb5_1hop_nolabel"
  LABEL_FLAG=(--include_neighbor_labels False)
else
  : "${FMT:?FMT env var required when MODE=label (bracket|paren|sentence|colon)}"
  EXP="pubmed_nb5_1hop_label_${FMT}"
  LABEL_FLAG=(--include_neighbor_labels True --neighbor_label_format "$FMT")
fi

echo "=============================================================="
echo "PubMed prompt-ablation: nb=5, 1-hop, frozen, mode=${MODE} fmt=${FMT:--}"
echo "=============================================================="

python examples/tmdlm/eval_logit.py \
  --model_name_or_path "$MODEL" \
  --dataset_name pubmed \
  --split test \
  --batch_size 8 \
  --max_seq_len 2048 \
  --max_neighbors_per_hop 5 \
  --max_hops 1 \
  --seed 42 \
  "${LABEL_FLAG[@]}" \
  --exp "$EXP" \
  --log_file experiments/nb_labels_pubmed.jsonl

echo "[mode=${MODE} fmt=${FMT:--}] done."
