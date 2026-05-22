"""Print decoded LP prompts to verify the LLaGA-split loader.

Loads N samples via ``load_lp_dataset(..., llaga_split_root=...)`` and prints
the decoded ``input_ids`` plus the supervised answer token. Use this before
launching SFT to confirm pairs / neighbors / template are correct.

Usage
-----
    python examples/tmdlm/print_lp_prompts.py \
        --dataset_name cora \
        --llaga_split_root .datasets/llaga \
        --split train \
        --n 3
"""
from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from dllm.data.graph import load_lp_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="cora", choices=["cora", "pubmed"])
    ap.add_argument("--llaga_split_root", required=True,
                    help="e.g. .datasets/llaga (resolved as <root>/<dataset>/...)")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_neighbors_per_hop", type=int, default=10)
    ap.add_argument("--max_hops", type=int, default=2)
    ap.add_argument("--model_name_or_path", default="GSAI-ML/LLaDA-8B-Instruct")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    ds = load_lp_dataset(
        args.dataset_name,
        tokenizer=tok,
        split=args.split,
        max_seq_len=args.max_seq_len,
        max_neighbors_per_hop=args.max_neighbors_per_hop,
        max_hops=args.max_hops,
        max_samples=args.n,
        llaga_split_root=args.llaga_split_root,
    )

    print(f"\n[info] dataset description: {ds.info.description}")
    print(f"[info] loaded {len(ds)} sample(s); printing {min(args.n, len(ds))}\n")

    for i, ex in enumerate(ds):
        edge = ex["edge"]
        cls = ex["cls_label"]
        label_str = "yes" if cls == 1 else "no"
        input_ids = ex["input_ids"]
        label_pos = ex["label_token_pos"]
        answer_tok_id = input_ids[label_pos]

        print("=" * 90)
        print(f"Sample {i}: edge=({edge[0]}, {edge[1]})  "
              f"cls_label={cls} ({label_str})  "
              f"seq_len={len(input_ids)}  label_pos={label_pos}")
        print("-" * 90)
        print(tok.decode(input_ids))
        print("-" * 90)
        ctx_lo = max(0, label_pos - 8)
        ctx_hi = min(len(input_ids), label_pos + 2)
        ctx_ids = input_ids[ctx_lo:ctx_hi]
        ctx_tokens = [tok.decode([t]) for t in ctx_ids]
        marker = ["<-- supervised" if (ctx_lo + j) == label_pos else "" for j in range(len(ctx_ids))]
        print(f"answer-token context (ids {ctx_lo}..{ctx_hi - 1}):")
        for tid, tstr, m in zip(ctx_ids, ctx_tokens, marker):
            print(f"  id={tid:>6d}  text={tstr!r:<20s} {m}")
        print(f"answer token: id={answer_tok_id}  decoded={tok.decode([answer_tok_id])!r}")
        print()


if __name__ == "__main__":
    main()
