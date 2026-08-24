# Spike: jlens on Apple Silicon

Answers [issue #4](https://github.com/RolynTrotter/vibe-kampff/issues/4) — **does
the Jacobian lens run usefully on this machine?** This is the one question that
can invalidate the project's plan, so it gets answered before anything else is
built.

Spike code. Isolated in `spike/` on purpose; the real implementation goes at the
top level once this comes back positive. Read it, run it, mine it for the parts
worth keeping, then let it rot.

## Quick start

```bash
cd spike/jlens-mps
./setup.sh                      # venv + torch + transformers + jlens
./run.sh --model qwen3-1.7b     # fast first pass, ~3.6 GB download
./run.sh                        # the real target: Qwen3-8B, ~17 GB download
```

That's it. `run.sh` picks MPS automatically and prints PASS/FAIL per check.

Start with `qwen3-1.7b`. It exercises the identical code path for a fifth of the
download, so a broken environment surfaces in minutes instead of after a 17 GB
wait. Once it passes, run the 8B.

Useful variations:

```bash
./run.sh --device cpu                 # is a failure MPS-specific?
./run.sh --dtype float32              # is a failure precision-related?
./run.sh --precision-check            # compare bf16 vs fp32 readouts
./run.sh --model qwen3-8b --skip-cost # skip the memory sweep
```

Each run writes a JSON report to `results/`, so findings can be compared across
machines and dtypes.

## What it checks

Mapping to the five questions in #4:

| # | Check | What a failure means |
|---|---|---|
| 1 | Versions and device | `transformers>=5.5` is a hard `jlens` requirement |
| 2 | Layout resolution and hooks | `jlens` can't find Qwen3's decoder blocks, or hooks return non-finite activations |
| 3 | Lens download and load | Wrong lens for the model — `d_model` mismatch is checked explicitly |
| 4 | Readout sanity | **The important one.** See below |
| 5 | Memory and timing | Establishes the ceiling and feeds the ETA for #9 |
| 6 | Precision (opt-in) | bf16 arithmetic on MPS diverges from fp32 |

### The readout check is the one that matters

Everything else can pass while the readout is quietly meaningless. So check 4
reproduces the paper's basic result rather than merely asserting the code ran.

It uses two-hop factual probes:

> `Fact: The currency used in the country shaped like a boot is`

The model should answer `euro`. But the interesting claim is about **`Italy`** —
a word the prompt never contains, which the model must nonetheless represent
internally to answer at all. The paper's finding is that the Jacobian lens
surfaces that bridge entity in mid-layer readouts. If `Italy` doesn't show up in
the band, the readout is not working, whatever the other checks say.

The check also runs the same probe with `use_jacobian=False` — the plain
logit-lens baseline — and prints both ranks. The paper claims the Jacobian lens
beats logit lens; seeing that on your own machine is a good sign you've wired it
up correctly.

**A `FAIL` here means stop.** Don't proceed to #6 or write scenarios. A broken
readout and an uninterested model produce identical nulls later, which is exactly
the confusion this whole project has to avoid.

## Reading the verdict

**All PASS** → MPS works. Close #4, record the numbers in `FINDINGS.md`, move on
to #5 and #6.

**FAIL on readout, PASS elsewhere** → the stack runs but the numbers are wrong.
Try `--dtype float32`, then `--device cpu`. If CPU-fp32 passes and MPS-bf16
fails, it's a precision or kernel problem on MPS, not a `jlens` problem —
`--precision-check` quantifies it.

**FAIL on hooks or layout** → a `transformers` version problem far more often
than an MPS one. Check the version first.

**Crash / out of memory** → see the ladder below.

### Fallback ladder

In preference order, from #4:

1. **Works as-is on MPS** → proceed.
2. **Works with fp32** → proceed, note the memory cost (fp32 doubles the model:
   32 GB for the 8B).
3. **Works on CPU only, slowly** → still viable. The workload is a few hundred
   short forward passes, not training. Take the timing from check 5 and decide.
4. **Doesn't work locally** → move the analysis lane to a rented GPU. The backend
   abstraction in #2 is designed so this changes one config value.

Dropping to `qwen3-4b` is also legitimate — it has a published lens and halves
the memory. The project targets 8B, but a working 4B pipeline beats a stalled
8B one.

## Apple Silicon specifics

`run.sh` sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, which routes any op MPS hasn't
implemented to the CPU instead of raising. If a run *only* succeeds with this
set, that is a finding worth writing down — note which op fell back, because it
will be slow.

If you hit memory pressure with the 8B:

```bash
# Raise how much unified memory the GPU may wire down (default is ~75% of RAM).
# Resets on reboot. 48 GB of a 64 GB machine:
sudo sysctl iogpu.wired_limit_mb=49152
```

Only reach for this if check 5 actually reports a ceiling problem. Don't
pre-emptively tune.

## Costs

| Model | Download | Peak memory (bf16) |
|---|---|---|
| `qwen3-1.7b` | ~3.6 GB | ~4 GB |
| `qwen3-4b` | ~8.5 GB | ~9 GB |
| `qwen3-8b` | ~17.2 GB | ~18 GB |

Models and lenses go to the HuggingFace cache (`~/.cache/huggingface`), not the
repo. Set `HF_HOME` to redirect. Nothing large is ever committed — see
`.gitignore`.

## Notes from building this

Things that were not obvious from the docs, recorded so nobody re-derives them:

- **The bridge entity does not appear at the last token, and not in the middle
  third of layers.** My first version of the readout check looked only at the
  final position, over a guessed middle-third band — and reported a confident
  FAIL on a pipeline that was working correctly. On Qwen3-1.7B the content sits
  at layers 18-25 of 28, at the position where the entity is *described*. The
  check now sweeps the whole grid. See `FINDINGS.md`; this is the most
  consequential thing the spike turned up.
- **Readouts are not English-only and not literal.** At the `boot` prompt, layer
  19 reads `pizza` and layer 22 reads `欧元` (euro in Chinese) — before `Italy`
  itself ranks. A literal English lexicon in #8 will under-count.
- **Batching positions is nearly free.** Cost is dominated by the forward pass:
  1 position took 1.75s, 64 positions took 2.80s. Always request every position
  you need in a single `apply()` call.
- **The transport is float32 regardless of model dtype.** `JacobianLens.__init__`
  calls `.float()` on the jacobians, and `apply()` calls `.float()` on the
  selected residuals. So `J_l @ h` is fp32 even for a bf16 model.
- **But the unembed is not.** `HFLensModel.unembed` casts the transported
  residual back to `lm_head.weight.dtype` before the final norm. On a bf16 model
  the final norm and unembedding run in bf16. That's the precision risk worth
  measuring, and it's why `--precision-check` exists.
- **`from_hf(force_bos=True)` is the default** — a BOS token is prepended. It
  shifts every position index by one, which will matter for the position map in
  #3.
- **`apply(positions=None)` returns every position**, at full vocab width, for
  every requested layer. For Qwen3-8B that is `36 × seq_len × 151936 × 4` bytes.
  Always pass explicit positions.
- **The lens repo hosts many models in one repo**, so `filename=` carries the
  path (`qwen3-8b/jlens/Salesforce-wikitext/...`). Loading the wrong file gives a
  `d_model` mismatch, which check 3 catches explicitly.
- **`uv venv` does not install pip.** Use `uv pip install --python .venv/bin/python`.

## What has actually been verified, and where

Honest status, because it determines how much to trust this:

- **Verified on Linux/CPU (in the container this was written in):** the full code
  path — install, layout resolution, hooks, lens download, readout, and the
  factual-recall reproduction — against real `Qwen/Qwen3-1.7B` and its published
  lens. See `FINDINGS.md` for the numbers.
- **Not verified:** anything MPS-specific. There is no Apple GPU in that
  container. Device selection, `mps.empty_cache`, memory reporting, and bf16
  behaviour on MPS are all written but unexercised.

So: the script is known-good apart from the device. **Running it on the Mac
Studio is the actual spike** — that's the part nobody has done yet.

## Deliberately not here

- No lens *fitting*. We use published lenses (#13 covers fitting).
- No scenarios, scoring, or lexicon (#7, #8).
- No band identification. `models.py` guesses the middle third purely so there's
  something to explore with; the real band is measured in #6 and must not be
  taken from here.
- No caching, no CLI polish, no tests. This is throwaway code.

## Attribution

- `jlens` — Anthropic PBC, Apache-2.0 — https://github.com/anthropics/jacobian-lens
- Pre-fitted lenses — Neuronpedia, MIT — https://huggingface.co/neuronpedia/jacobian-lens
- Paper — https://transformer-circuits.pub/2026/workspace/index.html
