#!/bin/bash
# Replicate EXPERIMENT_LOG.md PubMed 1-hop catinfill+options = 94.19% baseline.
#
# Config (from EXPERIMENT_LOG.md line 194, historical best frozen result):
#   - eval_logit.py (multi-token mean log-prob scoring), NOT eval_infill.py
#   - prompt_format=category_infill  (options list implicit via include_options=True default)
#   - prompt_layout=target_first
#   - max_hops=1, max_neighbors_per_hop=10   (1-hop full attn, default nb)
#   - use_topology_mask=False (default)
#   - NO --include_neighbor_labels (oracle-off baseline)
#
# Target: 94.19% (Experimental 99.4%, Type 1 88.0%, Type 2 95.2%)
#
# Submit:
#   sbatch --job-name=repl_pm_1hop_94 scripts/replicate_pubmed_94.sh
#
#SBATCH --account=bffx-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=01:00:00
#SBATCH --output=.logs/%x-%j.out
#SBATCH --error=.logs/%x-%j.err

set -eo pipefail
source activate dllm
export HF_DATASETS_CACHE=/projects/bffx/lingjie7/datasets/huggingface
cd /u/lingjie7/dlm-graph
export PYTHONPATH=/u/lingjie7/dlm-graph:${PYTHONPATH:-}

MODEL=/projects/bffx/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct

echo "=============================================================="
echo "Replicate EXPERIMENT_LOG: PubMed 1-hop nb=10 catinfill+options"
echo "Target: ~94.19% (eval_logit mean log-prob scoring)"
echo "=============================================================="

python examples/tmdlm/eval_logit.py \
  --model_name_or_path "$MODEL" \
  --dataset_name pubmed \
  --split test \
  --batch_size 1 \
  --max_seq_len 4096 \
  --max_neighbors_per_hop 10 \
  --max_hops 1 \
  --prompt_format category_infill \
  --prompt_layout target_first \
  --seed 42 \
  --exp pubmed_1hop_nb10_catinfill_opts_replicate \
  --log_file experiments/replicate_pubmed_94.jsonl

echo "[replicate 94.19] done."
