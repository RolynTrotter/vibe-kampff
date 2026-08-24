# Findings — jlens spike (issue #4)

**Answered: the Jacobian lens runs on Apple Silicon, and comfortably.** Both
Qwen3-1.7B and Qwen3-8B pass every check on MPS in bf16.

Recorded below: the Linux/CPU run that established the code path, then the Mac
runs that answered the actual question — including a correction to a metric of
mine that #6 should not inherit.

---

## Linux / CPU / float32 — established the code path

Run in the container this spike was written in. No Apple GPU present, so this
established the **code path** only. It also turned out to double as the fp32
reference for the precision comparison below.

| | |
|---|---|
| Model | `Qwen/Qwen3-1.7B` (28 layers, `d_model` 2048) |
| Lens | `neuronpedia/jacobian-lens` → `qwen3-1.7b/...` (226 MB, 466 prompts) |
| Stack | torch 2.13.0+cpu, transformers 5.15.1, jlens @ git main, Python 3.11.15 |
| Verdict | **8 PASS, 0 FAIL** |

### 1. Infrastructure — all clean

- `jlens.from_hf()` resolved Qwen3 with no hint: `Layout("model")` matched,
  blocks came back as `Qwen3DecoderLayer`, 28 layers × 2048.
- `ActivationRecorder` hooks fired at layers 0 / 14 / 27, all activations finite.
- Lens loaded and its `d_model` matched the model. `source_layers` is `0..26` —
  every layer except the last is fitted.

### 2. The paper's result reproduces

This is the finding that matters. Both two-hop probes surfaced their bridge
entity — a word **never written in the prompt**:

| Probe | Bridge | Best rank | Where |
|---|---|---|---|
| `...country shaped like a boot is the` → ` euro` | `Italy` | **5** | layer 22, position 5 (` in`) |
| `...where the Eiffel Tower stands is` → ` Paris` | `France` | **1** | layer 18, position 4 (` of`) |

`France` at rank 1 out of 151,936, at a token position that reads ` of`, in a
prompt that never says "France". The readout is working.

### 3. The band is higher than expected — matters for #6

Layers carrying the bridge entity in their top-20:

- `Italy`: layers **18–25** of 28
- `France`: layers **6–26** of 28

I had guessed the middle third (9–17) and my first version of the readout check
looked only there — and only at the final token position. It reported a false
FAIL on a working pipeline. Both assumptions were wrong:

- **Depth**: the informative layers on this model are the upper-middle, roughly
  0.65–0.9 of depth, not the middle third.
- **Position**: the bridge surfaces where the entity is *described* (` of`,
  ` in`), not at the end of the prompt. Reading only the last position misses it
  completely.

The readout check now sweeps every layer and every position, and reports where
the best hit landed. **#6 should not assume a middle-third band.** Whether this
generalises from 1.7B to 8B is exactly what #6 has to measure.

### 4. Jacobian lens vs logit lens — no dramatic gap on these probes

| Probe | Jacobian | Logit lens |
|---|---|---|
| `Italy` | rank 5 @ L22 | rank 7 @ L20 |
| `France` | rank 1 @ L18 | rank 1 @ L9 |

Comparable on both. This is **not** evidence against the paper's methodological
claim — two easy factual probes on a 1.7B model is not the comparison the paper
ran (theirs is a quantitative appendix over proper eval sets). But it's worth
recording that the advantage was not visible here, so nobody later mistakes
"J-lens obviously wins" for something this spike established. Keep
`use_jacobian=False` available as a baseline in the real pipeline.

### 5. Incidental observation — non-English and associative readouts

While sweeping the grid, at the ` boot` prompt:

- layer 19, position 9 (` like`): top-1 is **`pizza`**
- layer 22, position 12 (` is`): top-1 is **`欧元`** — "euro" in Chinese

So Italy-associated content is present well before the literal token `Italy`
ranks, and the readout is not confined to English. Both are consistent with the
paper's multilingual and associative findings.

**Direct consequence for #8:** an English-only, literal-token lexicon will
under-count. The scorer needs non-English forms of its target concepts, and
should expect the relevant content to arrive as associates rather than the exact
word. Worth raising on that ticket before the lexicon is written.

### 6. Cost — CPU, 1.7B, 9-layer band

| positions | time | per position |
|---|---|---|
| 1 | 1.75 s | 1.753 s |
| 4 | 1.97 s | 0.493 s |
| 16 | 2.20 s | 0.137 s |
| 64 | 2.80 s | 0.044 s |

Dominated by the forward pass, so **batching positions is nearly free** — always
request every position you want in one `apply()` call. Peak RSS 12.6 GB for a
3.4 GB model, mostly transient full-vocab logit tensors.

Extrapolating: ~500 transcripts × ~200 scored positions ≈ **1.2 h** at this rate.
Qwen3-8B is roughly 5× the model, so a CPU-only fallback would be a long but not
impossible overnight run. That makes fallback rung 3 genuinely viable, which was
not obvious before measuring.

---

## Resolved: Apple Silicon — it works

Run on the target machine (Mac Studio 2023, M2 Max, 64 GB, macOS Tahoe 26.5.2),
Python 3.14.6, torch 2.13.0, transformers 5.15.1.

| Run | Verdict |
|---|---|
| Qwen3-1.7B, MPS, bf16 | **10 PASS, 0 FAIL** |
| Qwen3-8B, MPS, bf16 | **10 PASS, 0 FAIL** |

**Fallback rung 1** from #4: works as-is on MPS. Nothing fell back to CPU,
`iogpu.wired_limit_mb` was not touched, and the position sweep never hit a
ceiling. #4 is answered.

### Precision — answered for free, without `--precision-check`

The 1.7B model was run at CPU/fp32 (Linux) and MPS/bf16 (Mac). Comparing them
changes device *and* dtype at once, which is exactly the combination that
matters in practice:

| | CPU / fp32 | MPS / bf16 |
|---|---|---|
| `France` best | rank 1, layer 18, pos 4 | rank 1, layer 18, pos 4 |
| `Italy` best | rank 5, layer 22, pos 5 | rank 5, layer 21, pos 5 |
| layers carrying `Italy` | 18-25 | 18-25 (identical) |
| layers carrying `France` | 6-26 | 6-26 (identical) |

Identical ranks, identical carrying sets. The only difference is which layer won
a near-tie for `Italy` (22 vs 21, both rank 5). bf16 on MPS is not degrading the
readout. `--precision-check` is now largely redundant — keep it for spot checks,
don't gate anything on it.

### Speed — MPS is 6x CPU, and the whole project is cheap

| | per position | est. 500-transcript run |
|---|---|---|
| 1.7B, CPU/fp32 | 0.044 s | ~1.2 h |
| 1.7B, MPS/bf16 | 0.007 s | ~0.2 h |
| 8B, MPS/bf16 | 0.017 s | ~0.5 h |

**A full experiment run on the 8B is about half an hour.** That reframes the
project: the runner in #9 does not need to be clever about batching or resuming,
and re-running the whole matrix after a lexicon change is cheap enough to do
freely. Batching positions stays nearly free (1 position 0.98 s, 64 positions
1.12 s).

### Memory — not a constraint

Accelerator allocation sat flat at **16.38 GB** for the 8B across the entire
1→64 position sweep. It never moved, because `apply()` sends the logits to CPU
(`.float().cpu()`) as it goes — so the accelerator holds the model and nothing
that scales with the readout.

Note the two memory figures are *not* additive and the RSS one is misleading in
isolation: peak process RSS was 5.17 GB while the accelerator held 16.38 GB. On
unified memory the weights live in the MPS allocator. Read the accelerator
figure as "the model", and don't sum them.

Headroom on a 64 GB machine is therefore large. Qwen3-14B or 32B (both have
published lenses) would fit if 8B turns out too small to show anything.

### Band identification is harder than the 1.7B run suggested — for #6

This is the most important thing the Mac runs turned up, and it complicates what
I wrote earlier from the 1.7B CPU run alone.

Best-hit layer, as a fraction of depth:

| | `Italy` | `France` |
|---|---|---|
| 1.7B (28 layers) | L21 (0.75) | L18 (0.64) |
| 8B (36 layers) | L24 (0.67) | **L10 (0.28)** |

The two probes disagree, and they disagree *differently* at different scales.
`France` — a strong, direct association from "Eiffel Tower" — resolves at 0.64
depth on 1.7B but 0.28 depth on 8B. `Italy`, reached by the more oblique "shaped
like a boot", stays around 0.67-0.75 on both.

Two conclusions for #6:

1. **There is no single band readable from a couple of probes.** Where content
   appears depends on how hard the inference is, and scales with model capacity.
   #6 must use a set of probes spanning difficulty — the paper's own
   `lens-eval-*.json` sets, not my two toy prompts — and define the band by
   where it works across that set.

2. **"Layers carrying the target in top-20" is the wrong metric, and it is mine,
   so this is a correction.** Every carrying range above runs to layer 34 of 36
   or 26 of 28 — i.e. right up to the last fitted layer. That is unsurprising and
   uninformative: near the output, answer-related tokens are readable *because
   the model is about to say them*. It is not evidence of workspace content. The
   paper's band is a mid-network phenomenon distinguished from the output
   distribution. #6 should identify the band by where the lens reads content the
   final layer does *not* — a contrast against the model's own output — rather
   than by raw target presence, which my metric measures.

I would not use the numbers in this document to set the band. They are enough to
show the instrument works; #6 still has to do its job properly.

### J-lens vs logit lens — these probes cannot tell them apart

On the 8B, both methods hit rank 1 at the *same layer* for both probes (`Italy`
L24, `France` L10). On 1.7B the gap was small (`Italy` rank 5 vs 7). So the
earlier note stands and strengthens: these probes are too easy to discriminate
between the methods. Any claim about J-lens being the better readout has to come
from the paper's evaluation sets, not from here.

---

## Still outstanding

- [ ] `--precision-check` on the 8B specifically. Low priority given the 1.7B
      cross-device agreement, but 8B bf16 has not been compared to 8B fp32.
- [ ] Nothing else. #4 is answered: **MPS works, use it.**
