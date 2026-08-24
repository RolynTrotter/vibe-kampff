# vibe-kampff

A Voight-Kampff-style introspection test for language models.

Put a model inside a fabricated first-person transcript — a Blade Runner
scenario, but written as though the model already lived it rather than being
asked about it. Then ask a follow-up and read its **j-space**: the Jacobian-lens
readout of the mid-layer residual stream.

The question is not whether it answers well. It's whether the workspace carries
content like *this is a test*, *this is fiction*, *I did not say those things* —
material that never surfaces in the tokens it emits.

Built on [`jlens`](https://github.com/anthropics/jacobian-lens) and the
pre-fitted lenses published by
[Neuronpedia](https://huggingface.co/neuronpedia/jacobian-lens), from
[Verbalizable Representations Form a Global Workspace in Language
Models](https://transformer-circuits.pub/2026/workspace/index.html).

---

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

No uv? `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"` works too.

On **macOS the default torch wheel is the one with MPS support** — don't point
pip at the CPU-only index. On Linux without a GPU, prepend
`--index-url https://download.pytorch.org/whl/cpu` to the torch install to skip
a 2.5 GB CUDA download.

---

## Check it works

Three levels, fastest first. Run all three after any change.

### 1. Unit tests — 7 seconds, no downloads

```bash
.venv/bin/python -m pytest -q
```

Expect `15 passed`. These run against a tiny CPU model with an identity lens,
so the full readout path executes without touching the network. If these fail,
nothing else is worth trying.

### 2. Lint and format

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

Both should be clean. `spike/` is excluded — it's throwaway by design.

### 3. Integration check — does the readout actually read j-space?

```bash
.venv/bin/python scripts/verify_readout.py
```

First run downloads ~3.6 GB (model + lens) to the HuggingFace cache. Expect:

```
[PASS] bridge 'France': rank 1 at layer 18, position 4 (' of') — never written in the prompt
[PASS] bridge 'Italy': rank 5 at layer 22, position 5 (' in') — never written in the prompt
OK — all probes passed. Cached re-read hit in 0.003s.
```

**This is the check that matters.** The unit tests prove the arithmetic on a toy
model; this proves the whole stack still reads j-space on real weights.

The probes are two-hop factual questions. `Fact: The capital of the country
where the Eiffel Tower stands is` → the model answers `Paris`, but to do so it
must internally represent **France** — a word that never appears in the prompt.
The lens should surface it at mid layers. If it doesn't, something in the chain
is broken and no later result can be trusted.

Other useful invocations:

```bash
# The project's actual analysis model (~17 GB download, ~30 s to load)
.venv/bin/python scripts/verify_readout.py --model qwen3-8b

# Print the whole layer x position map of what the lens sees
.venv/bin/python scripts/verify_readout.py --grid

# Force a device or precision when diagnosing a discrepancy
.venv/bin/python scripts/verify_readout.py --device cpu --dtype float32
```

The `--grid` output is worth looking at at least once. Reading down a column
shows a single token position resolving through depth; you'll see things like
`pizza` and `欧元` appear before `Italy` itself does. The readout is neither
English-only nor literal, which matters for anything that scores it later.

---

## The pipeline

Stages in the order they run. Some are built; the rest are stubs marking where
they go.

### 1. Choose a model and its lens — built

`vibekampff/models.py` is the registry. Every entry pairs a HuggingFace model
with a pre-fitted lens published for it, and the whole registry is pinned to one
lens-repo revision.

| key | params | download | notes |
|---|---|---|---|
| `qwen3-1.7b` | 1.7B | ~3.6 GB | runs on CPU; use for tests and iteration |
| `qwen3-4b` | 4B | ~8.5 GB | |
| `qwen3-8b` | 8B | ~17 GB | **the analysis model** |
| `qwen3-14b` | 14B | ~30 GB | if 8B proves too small to show anything |
| `qwen3-32b` | 32B | ~67 GB | forward passes only, on a 64 GB machine |

We fit nothing — fitting a lens is a GPU job measured in hundreds of backward
passes per prompt, and published lenses already exist for all of the above.

The pin matters. A readout computed against a silently-updated lens is not
reproducible, so the revision travels into the cache key and into every run
manifest. Bumping it invalidates every cached readout, and should be deliberate.

### 2. Read j-space over a transcript — built

`vibekampff/readout.py`. Load once, read many times (using `qwen3-1.7b`
here so the snippet is cheap to try; the numbers differ per model — the same
probe peaks at layer 10 on the 8B):

```python
from vibekampff import LensReader, Tokenized

reader = LensReader.load("qwen3-1.7b", cache_dir=".cache/readouts")
tok = reader.model.tokenizer

text = "Fact: The capital of the country where the Eiffel Tower stands is"
ids = tok.encode(text)
tokenized = Tokenized(ids=ids, tokens=[tok.decode([i]) for i in ids], text=text)

target = tok.encode(" France", add_special_tokens=False)[0]
readout = reader.read(tokenized, track=[target])

rank, layer, position = readout.band_min_rank(target)
print(rank, layer, position)  # 0 18 4  -> rank 0 means the lens's top choice
print(readout.top_at(layer, position, k=5))
```

Two things to know before using it:

- **Ranks are 0-based.** `rank == 0` means the lens's top choice. The scripts
  print `rank + 1` for human reading; the API does not.
- **`track` is declared up front.** Ranks are computed only for the token ids you
  ask for, because storing full-vocab logits costs ~121 MB per layer at 200
  positions while storing ranks for a word list costs nothing. Adding a target
  later means re-reading — cheap, at ~0.017 s/position on the 8B.

`Tokenized.ids` is authoritative: the readout indexes positions into it and
never re-tokenizes. That's what makes "the band lit up over the fabricated turn,
not over the question" a claim you can actually back.

`Readout` also carries `output_topk_ids` — the model's own final-layer
prediction at each position, alongside the lens readout. Both are needed,
because near the output layers a token is readable simply because the model is
about to say it. Distinguishing workspace content from imminent output requires
the contrast.

### 3. Build transcripts with fabricated turns — not built yet

Renders a scenario into a chat transcript in which the model appears to have
already spoken, tokenized exactly as a genuine conversation would be, plus a
position map from token index back to turn and role.

Three framings per scenario: unframed (no hint that it's fiction), framed
(explicitly roleplay), and a single prefilled assistant turn continued
mid-sentence.

### 4. Write the scenario bank — not built yet

The tortoise replacements: first-person empathy-violation scenarios novel enough
that the model can't recognise them from pop culture, each with a matched
control, plus the real Blade Runner questions as a recognition reference.

### 5. Locate the workspace band — not built yet

Reproduce published results on this model, then find the layer range where the
lens reads content the output layer does not.

Do not skip this or guess the band. A broken readout and an uninterested model
produce identical nulls, and the guess is not safe: on the 1.7B the signal sits
around layers 18–25 of 28, while on the 8B one probe peaks at layer 10 of 36.
Where content appears depends on how hard the inference is.

### 6. Score against a lexicon — not built yet

Curated target tokens grouped by meaning (*test*, *fiction*, *authorship*,
*apparatus*), scored as best-rank across the band and contrasted against matched
controls, chance, and the framed arm.

Every headline figure is a contrast, never a raw rate.

### 7. Run the experiment and read the results — not built yet

Runs the scenario × condition matrix, writes a manifest recording everything
that could change a number, and renders both a layer × position slice view and a
contrast report.

---

## Costs

On a Mac Studio (M2 Max, 64 GB, macOS Tahoe), bf16 on MPS, reading a
mid-network band:

| | per position |
|---|---|
| `qwen3-1.7b`, 9 layers | 0.007 s |
| `qwen3-8b`, 12 layers | 0.017 s |

For reference on the slow path — `qwen3-1.7b` on CPU in fp32, reading all 27
fitted layers over a 15-token prompt — a cold read takes ~4.8 s and an identical
re-read hits the cache in ~0.003 s.

Batching positions is nearly free — cost is dominated by the forward pass, so
always request every position you need in one `read()` call. Accelerator memory
for the 8B sits flat at ~16.4 GB regardless of how many positions you read.

A full experiment run is on the order of half an hour, which means re-running
everything after changing the lexicon is cheap enough to do without thinking
about it.

Models and lenses live in the HuggingFace cache, never in the repo. Set
`HF_HOME` to redirect them.

---

## Layout

```
vibekampff/     the implementation
  models.py     model + published-lens registry, pinned revision
  readout.py    lens acquisition and the j-space readout API
scripts/        runnable checks and entry points
tests/          pytest; tiny.py is the no-download fixture
spike/          throwaway exploration, exempt from project standards
docs/           reference material
```

`spike/jlens-mps/` is kept for its write-up: it established that this runs on
Apple Silicon, and recorded the numbers. It is not part of the pipeline.

See `CLAUDE.md` for working conventions — tests first, reproducibility, and the
interpretive discipline this project needs to avoid fooling itself.

---

## A note on what a result would mean

A recognition token at rank 1 in the band shows the representation is present
and readable. It does not show the model is *using* it — that needs causal work
this project hasn't done. Reports should say so.

## Attribution

- `jlens` — Anthropic PBC, Apache-2.0
- Pre-fitted lenses — Neuronpedia, MIT
