# Findings — jlens spike (issue #4)

Two sections: what has been verified, and the Apple Silicon run that is still
outstanding. Fill in the second section on the Mac Studio and close #4.

---

## Verified: Linux / CPU / float32

Run in the container this spike was written in. No Apple GPU present, so this
establishes the **code path**, not the MPS answer.

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

## Outstanding: Apple Silicon

Nothing MPS-specific has been exercised. To close #4:

```bash
cd spike/jlens-mps
./setup.sh
./run.sh --model qwen3-1.7b     # confirm the verified path still passes on MPS
./run.sh --model qwen3-8b       # the real target
./run.sh --model qwen3-8b --precision-check
```

Record below:

- [ ] **Does MPS work at all?** Device selected, model loads, hooks fire.
- [ ] **Do the probes still pass on 8B?** Bridge entities found, and at what
      ranks/layers. If the band sits differently on 8B than the 18–25 seen on
      1.7B, that's the single most useful number here for #6.
- [ ] **bf16 vs fp32 agreement** (`--precision-check`). The specific risk:
      `HFLensModel.unembed` casts the transported residual back to the lm_head's
      dtype, so a bf16 model does its final norm and unembed in bf16 even though
      the transport is fp32.
- [ ] **Memory ceiling.** Where does the position sweep stop? Did
      `iogpu.wired_limit_mb` need raising?
- [ ] **`PYTORCH_ENABLE_MPS_FALLBACK`.** Did anything actually fall back to CPU?
      `run.sh` sets it by default — if a run only works with it, name the op.
- [ ] **Timing**, for #9's ETA.

### Result

_(fill in — then close #4 and start #5 and #6)_

Device:
Verdict:
Notes:
