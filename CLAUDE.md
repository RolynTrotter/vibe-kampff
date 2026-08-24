# vibe-kampff — working conventions

A Voight-Kampff-style introspection test for language models: put a model
inside a fabricated first-person transcript, then read its **j-space** (the
Jacobian-lens readout of the mid-layer residual stream) to see whether the
workspace carries content it never says out loud.

See issue #1 for the architecture and the decisions behind it.

## Tests first

**Write the test before the implementation.** Not "write tests eventually" —
write the failing test, then make it pass.

This is not ceremony here, it is the only defence the project has. Nearly
everything we measure is a number nobody can eyeball for correctness: a token's
rank at layer 18 is either right or plausibly wrong, and plausibly wrong is
indistinguishable from a real finding. A test written after the fact tends to
assert what the code already does. A test written first has to state what the
code *should* do.

Concretely:

- New behaviour starts with a test that fails for the right reason.
- Bug fixes start with a test that reproduces the bug.
- Prefer tests that check against an independently computed answer over tests
  that check self-consistency. `test_rank_matches_a_hand_computed_argsort`
  recomputes a rank the slow, obvious way and compares — that catches a wrong
  shortcut; asserting the shortcut equals itself would not.
- Keep the fast path fast: `tests/tiny.py` gives a real `LensModel` with an
  identity lens, so the full readout path runs in under two seconds with no
  downloads. Use it. Reserve real-model runs for deliberate integration checks.

Spike code under `spike/` is exempt — that's throwaway by design.

## Numbers must be reproducible

Anything that could change a result goes in the run manifest: model id and
revision, **lens revision SHA**, dtype, device, lexicon version, workspace
band, scenario bank commit, seeds, library versions. A result without a
manifest is not a result.

The lens revision is pinned in `vibekampff/models.py`. Bumping it invalidates
every cached readout and should be a deliberate, recorded decision.

## Interpretive discipline

The project is unusually easy to fool itself with. Two rules that have already
had to be enforced against my own work:

- **Presence is not use.** A recognition token at rank 1 in the band shows the
  representation is present and readable. It does not show the model is using
  it. That needs the causal work in #12. Reports must say so.
- **Every headline figure is a contrast**, never a raw rate — against a matched
  control, a chance baseline, or the framed arm. A hit-rate alone is
  uninterpretable.

Related trap, found the hard way in #4: metrics like "layers carrying the
target in top-20" run right to the output layer, because near the output the
model is *about to say* the answer. That is not workspace content. Distinguish
what the lens reads from what the model outputs.

## Layout

```
vibekampff/     the real implementation
  models.py     model + published-lens registry, pinned revision
  readout.py    lens acquisition and the j-space readout API
tests/          pytest; tiny.py is the no-download fixture
spike/          throwaway exploration, not held to these standards
docs/           reference material
```

## Environment

Analysis runs on `Qwen/Qwen3-8B` in bf16. Verified on Apple Silicon (MPS) and
on CPU. Lenses are pre-fitted and downloaded from `neuronpedia/jacobian-lens`
— we fit nothing (see #13).

```bash
uv venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
```

Never commit model or lens files. They live in the HuggingFace cache.

## Attribution

`jlens` is Anthropic PBC, Apache-2.0. The pre-fitted lenses are Neuronpedia,
MIT. Keep both credited.
