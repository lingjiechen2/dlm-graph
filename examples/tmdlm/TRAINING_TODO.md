# TM-DLM Training TODO — PubMed

All runs: dataset=pubmed, 2-hop, max_neighbors_per_hop=10, LoRA r=64 all-linear, 10 epochs, bs=2, grad_accum=8, lr=5e-5.

## mc_digit (answer = single digit 0/1/2)

| Setting | Topo Mask | NB Labels | Run Tag | Status |
|---------|-----------|-----------|---------|--------|
| mc_digit | True  | True  | pubmed_20260428_mcdigit | ✅ done — checkpoints 370/740/1110/1480 |
| mc_digit | False | True  | pubmed_20260428_mcdigit | ✅ done — checkpoints 370/740/1110/1480 |
| mc_digit | True  | False | pubmed_20260428_mcdigit_nonb | 🔄 running GPU4 (step ~9/7400) |
| mc_digit | False | False | pubmed_20260428_mcdigit_nonb | 🔄 running GPU5 (step ~10/7400) |

Script: `run_sft_pubmed_nonb_lora.sh` (nonb), `run_sft_pubmed_index_lora.sh` (with labels)

## category_infill (answer = full class name, e.g. "Diabetes Mellitus Type 2")

| Setting | Topo Mask | NB Labels | Run Tag | Status |
|---------|-----------|-----------|---------|--------|
| category_infill | True  | False | pubmed_YYYYMMDD_catinfill_nonb | ☐ TODO |
| category_infill | False | False | pubmed_YYYYMMDD_catinfill_nonb | ☐ TODO |

Script: needs new `run_sft_pubmed_catinfill_nonb_lora.sh`
Key differences vs mc_digit: `prompt_format=category_infill`, `max_answer_tokens=4` (or 6), `answer_label_style` N/A.

## Notes

- `include_neighbor_labels=True` at **eval time** leaks oracle GT → not comparable to LLaGA baseline.
  Always eval with `include_neighbor_labels=False` for fair comparison.
- `mask_neighbor_labels=True` (train only): masks NB label tokens with `[MASK]`, trains diffusion jointly.
  Separate experiment, not yet scheduled.
- LLaGA uses graph embeddings (SBERT/E5), NOT oracle class-name text → our `nonb` setting is the fair comparison.
