#!/bin/bash
# Prompt-format ablation: frozen eval_logit on PubMed at nb=20, 2-hop,
# category_infill format (the setting that reaches ~90% baseline per README).
# One sbatch = one (format) run.
#
# Submit via:
#   sbatch --export=ALL,MODE=nolabel --job-name=prompt_abl_pm2h_nolabel \
#     scripts/eval_prompt_ablation_pubmed_2hop.sh
#   for fmt in bracket paren sentence colon; do
#     sbatch --export=ALL,MODE=label,FMT=$fmt --job-name=prompt_abl_pm2h_$fmt \
#       scripts/eval_prompt_ablation_pubmed_2hop.sh
#   done
#
#SBATCH --account=bffx-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=01:30:00
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
  EXP="pubmed_nb20_2hop_cat_nolabel"
  LABEL_FLAG=(--include_neighbor_labels False)
else
  : "${FMT:?FMT env var required when MODE=label (bracket|paren|sentence|colon)}"
  EXP="pubmed_nb20_2hop_cat_label_${FMT}"
  LABEL_FLAG=(--include_neighbor_labels True --neighbor_label_format "$FMT")
fi

echo "=============================================================="
echo "PubMed 2-hop nb=20 category_infill: mode=${MODE} fmt=${FMT:--}"
echo "=============================================================="

python examples/tmdlm/eval_logit.py \
  --model_name_or_path "$MODEL" \
  --dataset_name pubmed \
  --split test \
  --batch_size 1 \
  --max_seq_len 4096 \
  --max_neighbors_per_hop 20 \
  --max_hops 2 \
  --prompt_format category_infill \
  --prompt_layout target_first \
  --seed 42 \
  "${LABEL_FLAG[@]}" \
  --exp "$EXP" \
  --log_file experiments/nb_labels_pubmed.jsonl

echo "[mode=${MODE} fmt=${FMT:--}] done."
