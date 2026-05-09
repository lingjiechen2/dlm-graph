"""Aggregate leaderboard across all phase JSONLs."""

import json
from pathlib import Path

WORK = Path("/home/lingjie7/auto-research/projects/dlm-graph/analysis/postprocess_arxiv_r128")

rows = []
for j in WORK.glob("phase*.jsonl"):
    phase = j.stem
    for line in open(j):
        try:
            d = json.loads(line)
        except Exception:
            continue
        d["phase"] = phase
        rows.append(d)

# Filter out diagnostic-only rows
metric_rows = [r for r in rows if "acc" in r and not r.get("method", "").startswith("P. diag")]
metric_rows.sort(key=lambda r: -r["acc"])

print(f"Total rows: {len(rows)}, metric rows: {len(metric_rows)}")
print(f"\n  {'phase':<20} {'method':<55} {'ckpt':>6} {'acc':>6} {'Δ':>6}")
seen = set()
for r in metric_rows[:40]:
    key = (r.get("method"), r.get("ckpt"))
    if key in seen: continue
    seen.add(key)
    delta = r["acc"] - 74.4
    print(f"  {r['phase']:<20} {r.get('method',''):<55} {str(r.get('ckpt','')):>6} {r['acc']:>6.2f} {delta:>+6.2f}")
