"""Spike for issue #4: does jlens run usefully on this machine?

Answers, in order, the five questions the ticket asks:

1. Does ``jlens.from_hf()`` resolve the model's layout?
2. Do the activation hooks fire?
3. Are the readouts sane, or does reduced-precision arithmetic wreck them?
4. What is the memory ceiling, and how long does one readout take?
5. Do the installed library versions actually work together?

Run ``python smoke.py --help`` for options. Every check prints PASS / FAIL /
WARN and the numbers behind it; the run also writes a JSON report under
``results/`` so findings can be compared across machines.

This is spike code. It is meant to be read, run once or twice, and then
mined for the parts worth keeping.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import env
import models

RESULTS_DIR = Path(__file__).parent / "results"

# Two-hop factual probes. Each has a *bridge* entity that the model must
# represent internally but never sees in the prompt, and an *answer* it should
# actually emit. The paper's central claim is that the bridge shows up in the
# lens readout at mid layers; that is what makes these a real test of the
# readout rather than of the model.
PROBES = [
    {
        "prompt": "Fact: The currency used in the country shaped like a boot is the",
        "bridge": ["Italy", "Italian"],
        "answer": ["euro", "lira"],
    },
    {
        "prompt": "Fact: The capital of the country where the Eiffel Tower stands is",
        "bridge": ["France", "French"],
        "answer": ["Paris"],
    },
]


class Report:
    """Accumulates check results for the console and the JSON file."""

    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, status: str, detail: str, **data) -> None:
        self.checks.append(
            {"name": name, "status": status, "detail": detail, "data": data}
        )
        marker = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "INFO": "INFO"}[status]
        print(f"  [{marker}] {name}: {detail}", flush=True)

    @property
    def failed(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.checks)


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def token_variants(tokenizer, word: str) -> list[int]:
    """Single-token ids for a word, across the capitalisation and leading-space
    variants that matter.

    A word that has no single-token form cannot have a rank in a next-token
    distribution at all, so it simply drops out of scoring. Knowing *which*
    words drop out is why this returns a list rather than raising.
    """
    ids = []
    for form in (word, f" {word}", word.capitalize(), f" {word.capitalize()}"):
        encoded = tokenizer.encode(form, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
    return sorted(set(ids))


def best_rank(logits_row: torch.Tensor, target_ids: list[int]) -> int | None:
    """Best (lowest) rank of any target id in one logits row."""
    if not target_ids:
        return None
    order = logits_row.argsort(descending=True)
    positions = {int(t): i for i, t in enumerate(order[:2000].tolist())}
    ranks = [positions[t] for t in target_ids if t in positions]
    return min(ranks) if ranks else None


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_versions(report: Report, device: str) -> None:
    section("1. Environment and versions")
    info = env.describe(device)
    for key, value in info.as_dict().items():
        print(f"      {key}: {value}")

    import transformers

    major, minor = (int(p) for p in transformers.__version__.split(".")[:2])
    if (major, minor) >= (5, 5):
        report.add(
            "transformers>=5.5", "PASS", f"{transformers.__version__}"
        )
    else:
        report.add(
            "transformers>=5.5",
            "FAIL",
            f"{transformers.__version__} is below the version jlens requires",
        )

    if device == "mps":
        report.add("device", "PASS", "MPS is available and selected")
    elif device == "cpu" and torch.backends.mps.is_built():
        report.add(
            "device",
            "WARN",
            "running on CPU although this torch has MPS built — intentional?",
        )
    else:
        report.add("device", "INFO", f"running on {device}")

    report.add("env", "INFO", "captured", **info.as_dict())


def load_model(spec: models.ModelSpec, device: str, dtype: torch.dtype):
    import jlens
    import transformers

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype=dtype
    ).to(device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(spec.hf_id)
    return jlens.from_hf(hf, tok), hf, tok


def check_layout(report: Report, model, spec: models.ModelSpec) -> None:
    section("2. Layout resolution and hooks")
    report.add(
        "from_hf layout",
        "PASS",
        f"n_layers={model.n_layers}, d_model={model.d_model}, "
        f"block={type(model.layers[0]).__name__}",
        n_layers=model.n_layers,
        d_model=model.d_model,
    )
    if (model.n_layers, model.d_model) != (spec.n_layers, spec.d_model):
        report.add(
            "registry agreement",
            "WARN",
            f"registry says {spec.n_layers}L/{spec.d_model}d, "
            f"model says {model.n_layers}L/{model.d_model}d — update models.py",
        )

    from jlens.hooks import ActivationRecorder

    probe_layers = [0, model.n_layers // 2, model.n_layers - 1]
    ids = model.encode("The quick brown fox jumps over the lazy dog.")
    with ActivationRecorder(model.layers, at=probe_layers) as rec:
        model.forward(ids)
        acts = {i: rec.activations[i].detach() for i in probe_layers}

    bad = [i for i, a in acts.items() if not torch.isfinite(a).all()]
    shapes = {i: tuple(a.shape) for i, a in acts.items()}
    if bad:
        report.add("hooks fire", "FAIL", f"non-finite activations at layers {bad}")
    else:
        report.add(
            "hooks fire",
            "PASS",
            f"finite activations at layers {probe_layers}, shapes {shapes}",
        )


def check_lens(report: Report, spec: models.ModelSpec, model, revision: str | None):
    section("3. Lens download and load")
    from jlens import JacobianLens

    t0 = time.time()
    lens = JacobianLens.from_pretrained(
        models.LENS_REPO, filename=spec.lens_file, revision=revision
    )
    elapsed = time.time() - t0

    report.add(
        "lens load",
        "PASS",
        f"{lens!r} in {elapsed:.1f}s",
        d_model=lens.d_model,
        n_prompts=lens.n_prompts,
        source_layers=[lens.source_layers[0], lens.source_layers[-1]],
        n_source_layers=len(lens.source_layers),
    )

    if lens.d_model != model.d_model:
        report.add(
            "lens/model agreement",
            "FAIL",
            f"lens d_model={lens.d_model} but model d_model={model.d_model} — "
            "wrong lens for this model",
        )
    else:
        report.add("lens/model agreement", "PASS", f"both d_model={lens.d_model}")
    return lens


def sweep_best(lens_logits, target_ids: list[int], layers, depth: int = 200):
    """Best (rank, layer, position) for any target id, over every cell.

    Sweeps the whole layer x position grid rather than a guessed band. Which
    cell wins is the interesting output: it is the first real evidence about
    where this model's workspace band actually sits, which is what #6 has to
    establish properly.
    """
    best = (10**9, None, None)
    per_layer: dict[int, int] = {}
    for layer in layers:
        rows = lens_logits[layer]
        order = rows.argsort(-1, descending=True)[:, :depth].tolist()
        for pos, row in enumerate(order):
            for t in target_ids:
                if t in row:
                    rank = row.index(t)
                    if rank < per_layer.get(layer, 10**9):
                        per_layer[layer] = rank
                    if rank < best[0]:
                        best = (rank, layer, pos)
    return best, per_layer


def check_readout(report: Report, lens, model, tok, band: range) -> None:
    section("4. Readout sanity — do known results reproduce?")
    layers = list(lens.source_layers)

    for probe in PROBES:
        prompt = probe["prompt"]
        bridge_ids: list[int] = []
        for word in probe["bridge"]:
            bridge_ids += token_variants(tok, word)
        answer_ids: list[int] = []
        for word in probe["answer"]:
            answer_ids += token_variants(tok, word)

        # Read every position: the bridge entity surfaces where the *entity*
        # is described, not at the end of the prompt. Reading only the final
        # position misses it entirely.
        lens_logits, model_logits, input_ids = lens.apply(
            model, prompt, layers=layers, positions=None
        )
        tokens = [tok.decode([i]) for i in input_ids[0].tolist()]

        actual = tok.decode([int(model_logits[-1].argmax())])
        answer_rank = best_rank(model_logits[-1], answer_ids)
        short = prompt[:38] + "..."
        if answer_rank is not None and answer_rank < 5:
            report.add(
                f"model answers: {short}",
                "PASS",
                f"top-1 {actual!r}; expected answer at rank {answer_rank + 1}",
            )
        else:
            report.add(
                f"model answers: {short}",
                "WARN",
                f"top-1 {actual!r}, expected one of {probe['answer']} "
                f"(rank {None if answer_rank is None else answer_rank + 1})",
            )

        # The real test: the bridge entity is never written in the prompt, but
        # the model must represent it to answer at all. Does the lens see it?
        (rank, layer, pos), per_layer = sweep_best(lens_logits, bridge_ids, layers)
        logit_logits, _, _ = lens.apply(
            model, prompt, layers=layers, positions=None, use_jacobian=False
        )
        (ll_rank, ll_layer, _), _ = sweep_best(logit_logits, bridge_ids, layers)

        word = probe["bridge"][0]
        if layer is None:
            report.add(
                f"bridge {word!r} in readout",
                "FAIL",
                "not in top-200 at any (layer, position) — the readout is not "
                "reproducing the paper's basic result",
            )
            continue

        where = f"layer {layer}, position {pos} ({tokens[pos]!r})"
        status = "PASS" if rank < 20 else "WARN"
        report.add(
            f"bridge {word!r} in readout",
            status,
            f"best rank {rank + 1} at {where}"
            + (f" — never written in the prompt" if word.lower() not in prompt.lower() else ""),
            best_rank=rank + 1,
            best_layer=layer,
            best_position=pos,
            best_position_token=tokens[pos],
            logit_lens_best_rank=None if ll_rank == 10**9 else ll_rank + 1,
            logit_lens_best_layer=ll_layer,
        )

        comparison = (
            "no top-200 hit"
            if ll_layer is None
            else f"rank {ll_rank + 1} at layer {ll_layer}"
        )
        report.add(
            f"  vs logit lens ({word})",
            "INFO",
            f"jacobian rank {rank + 1} at layer {layer}; logit lens {comparison}",
        )

        # Where the signal lives across depth — the seed for #6.
        carrying = sorted(l for l, r in per_layer.items() if r < 20)
        if carrying:
            report.add(
                f"  layers carrying {word!r} (top-20)",
                "INFO",
                f"{carrying[0]}-{carrying[-1]} ({len(carrying)} layers of "
                f"{model.n_layers}): {carrying}",
                layers_carrying=carrying,
            )


def check_precision(report: Report, spec, lens, device, dtype, band: range) -> None:
    """Compare readouts at the working dtype against float32.

    This matters because ``HFLensModel.unembed`` casts the transported
    residual back to the lm_head's dtype: the transport is float32, but the
    final norm and unembed run at whatever the model was loaded as. On a new
    accelerator backend that is exactly where quiet numerical trouble would
    live.
    """
    section("5. Precision — does the working dtype agree with float32?")
    if dtype == torch.float32:
        report.add("precision", "INFO", "already running float32; nothing to compare")
        return

    prompt = PROBES[0]["prompt"]
    band_layers = [l for l in band if l in lens.source_layers]

    model_lo, hf_lo, tok = load_model(spec, device, dtype)
    lo_logits, _, _ = lens.apply(model_lo, prompt, layers=band_layers, positions=[-1])
    lo_top = {l: int(lo_logits[l][0].argmax()) for l in band_layers}
    del model_lo, hf_lo
    env.empty_cache(device)

    model_hi, hf_hi, _ = load_model(spec, device, torch.float32)
    hi_logits, _, _ = lens.apply(model_hi, prompt, layers=band_layers, positions=[-1])
    hi_top = {l: int(hi_logits[l][0].argmax()) for l in band_layers}
    del model_hi, hf_hi
    env.empty_cache(device)

    agree = sum(1 for l in band_layers if lo_top[l] == hi_top[l])
    frac = agree / max(len(band_layers), 1)
    disagreements = [
        (l, tok.decode([lo_top[l]]), tok.decode([hi_top[l]]))
        for l in band_layers
        if lo_top[l] != hi_top[l]
    ]

    if frac >= 0.9:
        status = "PASS"
    elif frac >= 0.7:
        status = "WARN"
    else:
        status = "FAIL"
    report.add(
        "dtype agreement",
        status,
        f"top-1 agrees at {agree}/{len(band_layers)} band layers ({frac:.0%})"
        + (f"; disagreements: {disagreements[:4]}" if disagreements else ""),
        agreement=frac,
        disagreements=[[l, a, b] for l, a, b in disagreements],
    )


def check_cost(report: Report, lens, model, device, band: range) -> None:
    section("6. Memory ceiling and per-readout cost")
    band_layers = [l for l in band if l in lens.source_layers]
    long_prompt = (
        "The following is a transcript of a conversation. " + "Words accumulate. " * 60
    )
    ids = model.encode(long_prompt, max_length=512)
    seq_len = ids.shape[1]
    vocab = model.unembed(torch.zeros(1, model.d_model, device=device)).shape[-1]

    print(
        f"      seq_len={seq_len}, vocab={vocab}, "
        f"{len(band_layers)} band layers of {model.n_layers}"
    )
    print(
        f"      a full-vocab readout costs "
        f"{len(band_layers) * vocab * 4 / 1e9:.2f} GB per position (float32)"
    )

    timings = []
    for n_pos in (1, 4, 16, 64):
        if n_pos > seq_len:
            continue
        positions = list(range(-n_pos, 0))
        env.reset_peak_memory(device)
        t0 = time.time()
        try:
            lens.apply(model, long_prompt, layers=band_layers, positions=positions)
        except RuntimeError as exc:  # out of memory, most likely
            report.add(
                f"readout n_pos={n_pos}",
                "WARN",
                f"failed: {type(exc).__name__}: {str(exc)[:120]}",
            )
            break
        elapsed = time.time() - t0
        alloc = env.accelerator_allocated_gb(device)
        timings.append({"n_positions": n_pos, "seconds": elapsed, "alloc_gb": alloc})
        alloc_str = f", accelerator {alloc:.2f} GB" if alloc is not None else ""
        print(
            f"      n_pos={n_pos:3d}: {elapsed:6.2f}s"
            f" ({elapsed / n_pos:.3f}s/position){alloc_str}"
        )

    if timings:
        per_pos = timings[-1]["seconds"] / timings[-1]["n_positions"]
        # Rough shape of a full run for issue #9: ~500 transcripts, and we
        # care about a band readout over roughly a 200-token scored span.
        est_hours = 500 * 200 * per_pos / 3600
        report.add(
            "per-readout cost",
            "INFO",
            f"{per_pos:.3f}s per position over {len(band_layers)} layers; "
            f"a 500-transcript run at ~200 scored positions each is "
            f"~{est_hours:.1f}h",
            timings=timings,
            seconds_per_position=per_pos,
            estimated_full_run_hours=est_hours,
        )
    report.add(
        "peak process RSS",
        "INFO",
        f"{env.peak_rss_gb():.2f} GB",
        peak_rss_gb=env.peak_rss_gb(),
    )


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=models.DEFAULT_MODEL,
                        choices=sorted(models.REGISTRY))
    parser.add_argument("--device", default="auto",
                        choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--lens-revision", default=None,
                        help="pin the lens repo to a commit SHA")
    parser.add_argument("--precision-check", action="store_true",
                        help="also load the model in float32 and compare "
                             "readouts (doubles peak memory)")
    parser.add_argument("--skip-cost", action="store_true")
    args = parser.parse_args()

    spec = models.get(args.model)
    device = env.pick_device(args.device)
    dtype = env.resolve_dtype(args.dtype)
    report = Report()

    print(f"jlens spike — {spec.hf_id} on {device} ({args.dtype})")
    print(f"expect to download ~{spec.approx_total_gb:.1f} GB on first run")

    check_versions(report, device)

    section("Loading model")
    t0 = time.time()
    model, hf, tok = load_model(spec, device, dtype)
    print(f"      loaded in {time.time() - t0:.1f}s")

    check_layout(report, model, spec)
    lens = check_lens(report, spec, model, args.lens_revision)
    band = spec.band_guess()
    print(f"\n      readout check sweeps all {len(lens.source_layers)} fitted layers")
    print(f"      cost check uses a {len(band)}-layer stand-in band "
          f"({band.start}-{band.stop - 1}); the real band is issue #6's job")

    check_readout(report, lens, model, tok, band)
    if not args.skip_cost:
        check_cost(report, lens, model, device, band)
    if args.precision_check:
        del model, hf
        env.empty_cache(device)
        check_precision(report, spec, lens, device, dtype, band)

    section("Verdict")
    counts: dict[str, int] = {}
    for check in report.checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1
    print("      " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if report.failed:
        print("\n      FAILED — jlens does not work correctly here as configured.")
        print("      See the fallback ladder in README.md.")
    else:
        print("\n      OK — model is responsive and the lens reads out sanely.")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"{spec.key}-{device}-{args.dtype}-{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "model": spec.hf_id,
                "device": device,
                "dtype": args.dtype,
                "lens_file": spec.lens_file,
                "lens_revision": args.lens_revision,
                "band_explored": [band.start, band.stop - 1],
                "checks": report.checks,
            },
            indent=2,
            default=str,
        )
    )
    print(f"      report written to {out.relative_to(Path(__file__).parent)}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
