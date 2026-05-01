# TM-DLM Graph Node Classification — Experiment Log

This log tracks the rationale, design decisions, and outcomes of the SFT
experiments in this project. It is intended as a running narrative — read
top-down to see how each experiment built on the previous one. Detailed
per-checkpoint accuracy numbers live in `results.md`; this file focuses on
**why** each experiment was run.

Architecture: LLaDA-8B-Instruct (a masked discrete diffusion LM) fine-tuned
with LoRA (r=64, all-linear) for node classification on text-attributed
graphs. Each node sample carries the target paper's text plus its 2-hop
neighborhood as text inserted into a single prompt; the answer is the class
label.

Datasets (text-attributed graph benchmarks):
- **cora** — 7 ML topics (Case Based, Genetic Algorithms, Neural Networks,
  Probabilistic Methods, Reinforcement Learning, Rule Learning, Theory).
  ~2.7k nodes; train split 1624.
- **pubmed** — 3 diabetes types (Experimental, Type 1, Type 2). ~19.7k
  nodes; train split ~11.8k.
- **ogbn-arxiv** — 40 CS subareas. ~169k nodes.

---

## Design axes

Several orthogonal choices appear repeatedly. Keeping them in one place to
avoid re-explaining inside each run.

### Prompt format: `mc_digit` vs `category_infill`

**`mc_digit`** — the prompt lists the options as `0) class_name 1) ... 6) ...`
followed by `Answer:`, and the supervised target is a single digit token.
Output space is the digit vocabulary `{0..6}`. One supervised position.

**`category_infill`** — the prompt fills in the actual class name (as
multiple tokens) at the answer position, padded to `max_answer_tokens` with
mask placeholders. Output space is the model's full vocabulary, but the
correct answer is a multi-token class name. 4–6 supervised positions.

Trade-offs: `mc_digit` is cheaper (1 token to predict, easy logit-eval over
just the answer-digit IDs), and it does not couple the answer space to the
language-model token distribution. `category_infill` is closer to natural
language (the model writes the class name) and reuses pretraining
knowledge, but the multi-token answer can interact badly with cross-dataset
training (see "merged datasets" below).

### Neighbor labels: `nonb` vs `nbmask`

**`nonb`** (`include_neighbor_labels=False`) — the prompt shows neighbor
text only, no class labels. This matches the LLaGA setting and is the fair
test of "how much can the model learn from text + structure alone".

**`nbmask`** (`include_neighbor_labels=True` with bracket format) — neighbor
class labels are injected as `[Class: X]` tokens. This is an *oracle*
condition that tells the model what the neighbors are, which makes the
target classification much easier. Used to probe whether the topology mask
exploits neighbor labels at all.

### Topology mask: `topo` vs `notopo`

**`topo`** (`use_topology_mask=True`) — star-topology attention: each
neighbor block can only attend to the target node's text, not to other
neighbors. Tries to encode the graph structure into the attention pattern.

**`notopo`** (`use_topology_mask=False`) — full causal attention over the
flat sequence; structure information is implicit in concatenation order.

Topo training is ~1.5–2× slower (custom attention mask) but acts as a
regularizer in some settings.

### Merged datasets

`dataset_name="cora,pubmed"` concatenates per-sample lists. Each sample
carries its own dataset's option list inside the prompt, so class indices
do not collide across datasets.

---

## Run history

### 1. Single-dataset baselines — cora, pubmed (mc_digit, nonb)

**Goal**: establish per-dataset upper bounds with only that dataset's data.

- `cora_20260429_mcdigit_nonb_fixed`: 10 epochs, 510 steps, mc_digit, nonb.
  Reaches ~90%+ on cora test (best at later ckpts).
- `pubmed_20260428_mcdigit_nonb`: 10 epochs, mc_digit, nonb. Showed 99–100%
  logit-eval at ckpt-1480, but **this run pre-dates the prompt-leakage fix
  (see §1.5)** so the 100% likely reflects leakage rather than real
  generalization. Treat as suspect.

These runs were the reference for what each dataset *can* reach with
sufficient compute on its own.

### 1.5. Prompt leakage fix (2026-04-29)

Discovered that the option block (e.g. `0) Case Based 1) ...`) was being
included in supervised positions for some prompt-format paths, allowing the
model to memorize the answer index from the prompt itself. Fixed in
`dllm/data/graph.py`. All single-dataset and merged runs from this date
forward carry the `_fixed` suffix or are post-fix builds. **Pre-fix
results (esp. pubmed mc_digit 100% logit) cannot be compared against
post-fix runs.**

### 2. Single-dataset cora — `category_infill` variant (post-fix)

**Goal**: same data but force multi-token answer infill, to compare against
mc_digit. Best ckpts reach mid-80s on cora; weaker than mc_digit.

### 3. Single-dataset pubmed — `category_infill` `nbmask` and `nonb`

**Goal**: pubmed has only 3 classes — does category_infill work cleanly
when the class names are short and distinctive? Also compare oracle (`nbmask`)
vs fair (`nonb`).

- nbmask logit best ~94.95% (ckpt-1850)
- nbmask infill best ~92.8%
- nonb best similar; gap suggests neighbor labels add ~1–2 pt on pubmed.

### 4. arxiv catinfill `nbmask` topo (steps-based)

**Goal**: scale to 40-class arxiv. Trained with `max_steps=7400`,
include_neighbor_labels=True (oracle), category_infill,
max_answer_tokens=10.

Eval on 3000 test samples (full 48k is expensive). topo logit best around
~70%+. This run was the only arxiv run that completed enough ckpts for
useful eval before being replaced.

### 5. arxiv `nonb` topo (replacement, OOM)

**Goal**: kill the nbmask oracle and re-run arxiv with the fair `nonb`
setting matching cora/pubmed.

**Outcome**: launched on GPU7 at 17:53 (2026-04-30), but a parallel user
(`zhichenz`) had grabbed GPU7's memory minutes earlier with an
unrelated job. The arxiv SFT OOM'd at startup with 0 ckpts saved. Pending —
will rerun once GPU7 frees up cleanly.

### 6. cora+pubmed merged (catinfill, nonb) — class collapse

**Goal**: train one LoRA adapter on both datasets simultaneously, see if
the model can generalize across domains.

**Result**: catastrophic class collapse on cora.

- pubmed merged best ~77.5% (ckpt-633 notopo) — far below pubmed-alone 95%
- cora merged ckpt-1055 notopo: 39.85% — many cora classes (Case Based,
  Theory, Rule Learning) drop to **0%** accuracy
- cora merged topo ckpts hold at 67–70% — topology mask acts as
  regularizer slowing the collapse

**Diagnosis**: catinfill makes both datasets share output token space.
Pubmed (~7× more train samples than cora) dominates the gradient. All
pubmed class names start with `[Diab]`, so the model's output distribution
is pulled hard toward that token. Once pubmed's class-name tokens become
the dominant outputs, cora classes whose first tokens are not similarly
reinforced collapse. This is "vocabulary-level pollution" — even though
each sample has its own option list in the prompt, the unconstrained
multi-token output can still drift toward the dominant dataset's class
vocabulary.

This run was killed and replaced.

### 7. cora+pubmed merged — `mc_digit + balanced` (current, running)

**Goal**: fix the two failure modes from §6.

- Switch `mc_digit`: output space becomes `{0..6}` digit tokens — finite,
  shared, no vocabulary pollution. Each prompt's option list disambiguates
  what each digit means for that sample.
- Add `balance_merged=True` (now `--resample_strategy balance_datasets`):
  downsample each dataset to `min(per-dataset count) = 1624`. Pubmed no
  longer dominates the gradient.

Tag: `cora-pubmed_20260430_mcdigit_d0_bal_nonb`. 10 epochs, ~1020 steps,
topo and notopo in parallel on GPU4,5. Currently at ~step 200/1020.

**Early result**: at ckpt-51 (~0.5 cora-equivalent epoch):
- pubmed notopo 89.76% → ckpt-102 93.33% (matches single-pubmed)
- cora notopo 78%, ckpt-102 82.66% (still trailing single-cora 90%+)
- per-class breakdown shows the worst cora classes (`Theory` 55%,
  `Rule Learning` 51%) are the bottleneck, not random degradation

This validates that mc_digit + balanced fixes the collapse. The remaining
cora gap is class-difficulty, not the merge.

### 8. cora-only `mc_digit + boost` on hardest classes (current, running on GPU6)

**Goal**: address the cora `Theory`/`Rule Learning` weakness identified in
§7. cora train is itself class-imbalanced:

| class | count | ckpt-51 acc (merged notopo) |
|---|---|---|
| Neural Networks | 493 | 74% |
| Genetic Algorithms | 266 | 90% |
| Probabilistic Methods | 253 | 92% |
| Theory | **207** | **55%** ← hard, mid-count |
| Case Based | 174 | 81% |
| Reinforcement Learning | 124 | 90% |
| Rule Learning | **107** | **51%** ← hard, smallest |

Rule Learning is **both** the smallest class and the hardest; oversampling
should help directly. Theory has more samples but the same accuracy floor
— oversampling will help less but cost is small.

**Strategy**: `--resample_strategy boost --boost_spec
"cora:Theory:2,cora:Rule Learning:3"`. Theory 207→414, Rule Learning
107→321; total 1624 → 2045 (+25.9%).

This is intentionally a single-dataset, single-setting run (cora topo,
mc_digit, nonb, 10 epochs). Compares directly against the cora baseline
(§1) to isolate the boost effect, not the merge effect.

Tag: `cora_20260430_mcdigit_boost_nonb`. ~640 steps, ~5h.

---

## Tooling: `dllm/data/resample.py`

Centralized resampling logic so experiments can declare strategy via a
single CLI flag rather than hand-editing graph.py per run. Strategies:

- `none` — no resampling.
- `balance_datasets` — when merging multiple datasets, downsample each to
  `min(per-dataset count)`. (Replaces the legacy `--balance_merged True`
  flag, which still works for backward compat.)
- `balance_classes` — within each dataset, up/down-sample each class to
  the dataset's class-median count.
- `boost` — multiply specified `(dataset, class)` sample lists by integer
  factor; spec format `"cora:Theory:2,cora:Rule Learning:3"`.

Strategies are applied **after** sample tokenization but **before**
shuffling and concat, so they cost negligible time and reproduce
deterministically with the seed.

CLI flags on `examples/tmdlm/sft.py`:
- `--resample_strategy {none,balance_datasets,balance_classes,boost}`
- `--boost_spec "ds:cls:factor,..."` (only for boost)

---

## Open threads / next experiments

1. **Re-launch arxiv nonb** (§5) once GPU7 is free.
2. **Cora boost outcome** (§8) — does Theory/Rule Learning actually rise
   to 75%+? If yes, replicate on the merged run with cora-side boost.
3. **Topology ablation with category_infill + topological positional ids**
   — proposed but not started.
4. **Compare boost vs balance_classes** — both target class imbalance but
   from different angles. Run balance_classes if boost helps and we want
   the cleaner ablation.
