#!/bin/bash
# Replicate the README PubMed 2-hop nb=20 = 90.29% baseline.
#
# Key settings (from README "Neighbor Count Sweep" section):
#   - eval_infill.py (iterative denoising), NOT eval_logit.py
#   - prompt_format=category_infill
#   - prompt_layout=target_first
#   - use_topology_mask=False (default)
#   - steps=10 (iterative denoising steps)
#   - max_hops=2, max_neighbors_per_hop=20
#   - NO --include_neighbor_labels (oracle-off baseline)
#
# Submit:
#   sbatch --job-name=repl_pm_nb20_2hop scripts/replicate_pubmed_90.sh
#
#SBATCH --account=bffx-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=02:00:00
#SBATCH --output=.logs/%x-%j.out
#SBATCH --error=.logs/%x-%j.err

set -eo pipefail
source activate dllm
export HF_DATASETS_CACHE=/projects/bffx/lingjie7/datasets/huggingface
cd /u/lingjie7/dlm-graph
export PYTHONPATH=/u/lingjie7/dlm-graph:${PYTHONPATH:-}

MODEL=/projects/bffx/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct

echo "=============================================================="
echo "Replicate README: PubMed 2-hop nb=20 category_infill infill-eval"
echo "Target: ~90.29% (accuracy_strict)"
echo "=============================================================="

python examples/tmdlm/eval_infill.py \
  --model_name_or_path "$MODEL" \
  --dataset_name pubmed \
  --split test \
  --batch_size 1 \
  --max_seq_len 4096 \
  --max_neighbors_per_hop 20 \
  --max_hops 2 \
  --prompt_format category_infill \
  --prompt_layout target_first \
  --steps 10 \
  --temperature 0.0 \
  --remasking low_confidence \
  --seed 42 \
  --exp openended_pubmed_nb20_2hop_replicate \
  --log_file experiments/replicate_pubmed_90.jsonl

echo "[replicate] done."
