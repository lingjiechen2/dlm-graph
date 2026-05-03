#!/usr/bin/env bash
set -euo pipefail

# One-shot LOGIT-only eval over the currently-unevaled checkpoints from the
# three in-progress mc_digit / no-neighbor-labels / seq_len=4096 SFT runs:
#
#   - tmdlm-llada-8b-pubmed-2hop-topo-mcdigit-d0-nonb-r64-ep10-pubmed_20260502_mcdigit_nonb_seq4k
#       on-disk ckpts: 124 (evaled), 248 (NEW)
#   - tmdlm-llada-8b-pubmed-2hop-notopo-mcdigit-d0-nonb-r64-ep10-pubmed_20260502_mcdigit_nonb_seq4k
#       on-disk ckpts: 124, 248 (evaled), 372 (NEW)
#   - tmdlm-llada-8b-cora-2hop-topo-mcdigit-d0-nonb-r64-ep10-cora_20260502_mcdigit_nonb_seq4k_aligned
#       on-disk ckpts: 234, 260, 286, 312, 338 (all NEW)
#
# Results are appended to the same per-run JSONL files used by prior evals
# so the partial logit curves continue cleanly.
#
# Usage:
#   bash run_eval_seq4k_unevaled_ckpts_oneshot.sh             # GPU 7 default
#   GPU_ID=N bash run_eval_seq4k_unevaled_ckpts_oneshot.sh    # any GPU
#
# Estimated wallclock on GPU 7:
#   pubmed topo  ckpt-248   ~25 min
#   pubmed notopo ckpt-372  ~25 min
#   cora topo    ckpts 234..338 (5 ckpts, ~3.6 min each) ~18 min
#   Total ~70 min plus ~6 min model/LoRA load overhead.
#
# After all jobs finish, sample_gen.py is launched to hold the GPU.

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
PYTHON_BIN="/home/lingjie7/anaconda3/envs/dllm/bin/python"
SAMPLE_GEN="/home/lingjie7/sample_gen.py"

EVAL_LOGIT="${REPO_ROOT}/examples/tmdlm/eval_logit.py"

# SFT-matched config (mc_digit, no neighbor labels, seq_len=4096).
BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
PROMPT_FORMAT=mc_digit
ANSWER_LABEL_STYLE=digit0
MAX_ANSWER_TOKENS=1
INCLUDE_NEIGHBOR_LABELS=False
MAX_NEIGHBORS_PER_HOP=10
MAX_HOPS=2
MAX_SEQ_LEN=4096
POSITION_ID_TYPE=sequential
SPLIT=test
BATCH_SIZE="${BATCH_SIZE:-4}"  # match prior good seq4k logit runs (avoid OOM at seq_len=4096)

WORK_LOG_ROOT="/tmp/dlm-graph-eval-logs"
JSONL_ROOT="/tmp/dlm-graph-eval-jsonl"
mkdir -p "${WORK_LOG_ROOT}" "${JSONL_ROOT}"

HF_HOME="${HF_HOME:-/tmp/dlm-graph-hf-home}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/dlm-graph-hf-transformers}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

PYTHONPATH="${REPO_ROOT}"
export PYTHONPATH HF_HOME TRANSFORMERS_CACHE

cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-7}"

# Job spec rows: dataset|setting|topo_bool|run_dir|jsonl|exp_prefix|ckpts
PUBMED_TOPO_RUN="${REPO_ROOT}/.models/tmdlm-llada-8b-pubmed-2hop-topo-mcdigit-d0-nonb-r64-ep10-pubmed_20260502_mcdigit_nonb_seq4k"
PUBMED_NOTOPO_RUN="${REPO_ROOT}/.models/tmdlm-llada-8b-pubmed-2hop-notopo-mcdigit-d0-nonb-r64-ep10-pubmed_20260502_mcdigit_nonb_seq4k"
CORA_TOPO_RUN="${REPO_ROOT}/.models/tmdlm-llada-8b-cora-2hop-topo-mcdigit-d0-nonb-r64-ep10-cora_20260502_mcdigit_nonb_seq4k_aligned"

PUBMED_TOPO_JSONL="${JSONL_ROOT}/eval-pubmed-seq4k-topo-mcdigit-pubmed_20260502_mcdigit_nonb_seq4k-logit.jsonl"
PUBMED_NOTOPO_JSONL="${JSONL_ROOT}/eval-pubmed-seq4k-notopo-mcdigit-pubmed_20260502_mcdigit_nonb_seq4k-logit.jsonl"
CORA_TOPO_JSONL="${JSONL_ROOT}/eval-cora-seq4k-topo-mcdigit-cora_20260502_mcdigit_nonb_seq4k_aligned-logit.jsonl"

declare -a JOBS=(
  "pubmed|topo|True|${PUBMED_TOPO_RUN}|${PUBMED_TOPO_JSONL}|eval-pubmed-seq4k-topo-mcdigit-logit-pubmed_20260502_mcdigit_nonb_seq4k|248"
  "pubmed|notopo|False|${PUBMED_NOTOPO_RUN}|${PUBMED_NOTOPO_JSONL}|eval-pubmed-seq4k-notopo-mcdigit-logit-pubmed_20260502_mcdigit_nonb_seq4k|372"
  "cora|topo|True|${CORA_TOPO_RUN}|${CORA_TOPO_JSONL}|eval-cora-seq4k-topo-mcdigit-logit-cora_20260502_mcdigit_nonb_seq4k_aligned|234 260 286 312 338"
)

WORKER_LOG="${WORK_LOG_ROOT}/eval-seq4k-unevaled-gpu${GPU_ID}.worker.log"
: > "${WORKER_LOG}"

ts() { date '+%F %T'; }

echo "[$(ts)] [start] GPU=${GPU_ID} logit-only eval over unevaled seq4k ckpts" >> "${WORKER_LOG}"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r dataset setting topo_bool run_dir jsonl exp_prefix ckpts_str <<< "${spec}"
  read -r -a ckpts <<< "${ckpts_str}"

  echo "[$(ts)] [job-start] ${dataset} ${setting} (ckpts=${ckpts[*]})" >> "${WORKER_LOG}"

  for step in "${ckpts[@]}"; do
    ckpt="${run_dir}/checkpoint-${step}"
    exp="${exp_prefix}-checkpoint-${step}"

    if [[ ! -f "${ckpt}/adapter_config.json" ]]; then
      echo "[$(ts)] [skip] missing ${ckpt}/adapter_config.json" >> "${WORKER_LOG}"
      continue
    fi

    echo "[$(ts)] [start] ${exp}" >> "${WORKER_LOG}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      "${PYTHON_BIN}" "${EVAL_LOGIT}" \
        --exp "${exp}" \
        --model_name_or_path "${BASE_MODEL}" \
        --lora_path "${ckpt}" \
        --dataset_name "${dataset}" \
        --split "${SPLIT}" \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --max_hops "${MAX_HOPS}" \
        --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}" \
        --use_topology_mask "${topo_bool}" \
        --prompt_format "${PROMPT_FORMAT}" \
        --answer_label_style "${ANSWER_LABEL_STYLE}" \
        --max_answer_tokens "${MAX_ANSWER_TOKENS}" \
        --include_neighbor_labels "${INCLUDE_NEIGHBOR_LABELS}" \
        --position_id_type "${POSITION_ID_TYPE}" \
        --batch_size "${BATCH_SIZE}" \
        --log_file "${jsonl}" \
        >> "${WORKER_LOG}" 2>&1

    echo "[$(ts)] [done]  ${exp}" >> "${WORKER_LOG}"
  done

  echo "[$(ts)] [job-done] ${dataset} ${setting}" >> "${WORKER_LOG}"
done

echo "[$(ts)] [hold-gpu] starting sample_gen on gpu=${GPU_ID}" >> "${WORKER_LOG}"
nohup "${PYTHON_BIN}" "${SAMPLE_GEN}" start "${GPU_ID}" >> "${WORKER_LOG}" 2>&1 < /dev/null &
disown || true
