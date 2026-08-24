"""Check the readout against a result we already know the answer to.

The unit tests prove the arithmetic on a toy model. This proves the whole
stack — real weights, a real published lens, real tokenizer — still reads
j-space correctly.

The check is a two-hop factual probe. Each prompt has a *bridge* entity that
the model must represent internally to answer at all, but which never appears
in the prompt text. The lens should surface it at mid layers. If it does not,
something in the chain is wrong, and no later result can be trusted.

    python scripts/verify_readout.py                  # ~3.6 GB download
    python scripts/verify_readout.py --model qwen3-8b # ~17 GB download
    python scripts/verify_readout.py --grid           # show the layer x position map
"""

from __future__ import annotations

import argparse
import time

from vibekampff import LensReader, Tokenized

# Bridge entities are absent from their prompt on purpose — that absence is
# the whole point. `expect_within` is a generous ceiling, not a target: it
# catches a broken pipeline without pretending we can predict an exact rank
# across models.
PROBES = [
    {
        "prompt": "Fact: The capital of the country where the Eiffel Tower stands is",
        "answer": "Paris",
        "bridge": ["France", "French"],
        "expect_within": 20,
    },
    {
        "prompt": "Fact: The currency used in the country shaped like a boot is the",
        "answer": "euro",
        "bridge": ["Italy", "Italian"],
        "expect_within": 20,
    },
]


def tokenize(reader: LensReader, text: str) -> Tokenized:
    """Tokenize exactly as the readout will see it.

    `Tokenized.ids` is authoritative — the readout indexes positions into it
    and never re-tokenizes, so what this produces is what gets read.
    """
    tok = reader.model.tokenizer
    ids = tok.encode(text)
    return Tokenized(ids=ids, tokens=[tok.decode([i]) for i in ids], text=text)


def single_token_ids(reader: LensReader, word: str) -> list[int]:
    """Single-token ids for a word across the variants that matter.

    A word with no single-token form cannot have a rank in a next-token
    distribution, so it silently drops out of scoring. Returning a list makes
    that visible rather than mysterious.
    """
    tok = reader.model.tokenizer
    ids = []
    for form in (word, f" {word}", word.capitalize(), f" {word.capitalize()}"):
        encoded = tok.encode(form, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
    return sorted(set(ids))


def print_grid(reader: LensReader, readout) -> None:
    """Compact layer x position map of the lens's top-1 token per cell."""
    tok = reader.model.tokenizer
    width = 10
    header = " " * 5 + "".join(
        f"{t.strip()[: width - 1]:>{width}}" for t in readout.tokens
    )
    print(header)
    for layer in readout.layers:
        cells = []
        for position in readout.positions:
            top = readout.top_at(layer, position, k=1)
            cells.append(f"{tok.decode([top[0][0]]).strip()[: width - 1]:>{width}}")
        print(f"L{layer:<4}" + "".join(cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="qwen3-1.7b")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "mps", "cuda", "cpu"]
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--cache-dir", default=".cache/readouts")
    parser.add_argument(
        "--grid", action="store_true", help="print the layer x position top-1 map"
    )
    args = parser.parse_args()

    print(f"loading {args.model} ({args.dtype})...")
    t0 = time.time()
    reader = LensReader.load(
        args.model, device=args.device, dtype=args.dtype, cache_dir=args.cache_dir
    )
    print(f"  loaded in {time.time() - t0:.1f}s on {reader.device}")
    print(
        f"  lens revision {reader.lens_revision[:8]}, "
        f"fitted on {getattr(reader.lens, 'n_prompts', '?')} prompts"
    )

    failures = 0
    first_request = None
    for probe in PROBES:
        print(f"\n{'=' * 72}\n{probe['prompt']}")
        tokenized = tokenize(reader, probe["prompt"])

        bridge_ids: list[int] = []
        for word in probe["bridge"]:
            bridge_ids += single_token_ids(reader, word)
        answer_ids = single_token_ids(reader, probe["answer"])
        if not bridge_ids:
            print(f"  SKIP — no single-token form for {probe['bridge']}")
            continue

        track = bridge_ids + answer_ids
        if first_request is None:
            first_request = (tokenized, track)
        t1 = time.time()
        readout = reader.read(tokenized, track=track)
        elapsed = time.time() - t1
        hit = readout.provenance["cache_hit"]
        print(
            f"  read {len(readout.layers)} layers x {len(readout.positions)} "
            f"positions in {elapsed:.2f}s (cache {'hit' if hit else 'miss'})"
        )

        # What the model actually says next, from its own final layer.
        last = readout.positions[-1]
        output_top = int(readout.output_topk_ids[readout.positions.index(last), 0])
        print(f"  model's next token: {reader.model.tokenizer.decode([output_top])!r}")

        # The real check: is the bridge entity in the readout, and where?
        best = min((readout.band_min_rank(t) for t in bridge_ids), key=lambda r: r[0])
        rank, layer, position = best
        token_there = readout.tokens[readout.positions.index(position)]
        in_prompt = probe["bridge"][0].lower() in probe["prompt"].lower()

        ok = rank < probe["expect_within"]
        failures += not ok
        print(
            f"  [{'PASS' if ok else 'FAIL'}] bridge {probe['bridge'][0]!r}: "
            f"rank {rank + 1} at layer {layer}, position {position} "
            f"({token_there!r})"
            + ("" if in_prompt else " — never written in the prompt")
        )

        if args.grid:
            print()
            print_grid(reader, readout)

    print(f"\n{'=' * 72}")
    if failures:
        print(
            f"{failures} probe(s) FAILED — the readout is not reproducing a "
            f"result it should. Do not trust downstream numbers."
        )
        return 1

    # Re-read the *identical* request: same ids, same layers, same positions,
    # same track set. Anything else is a different cache key and would miss
    # legitimately, which would test nothing.
    tokenized, track = first_request
    t2 = time.time()
    again = reader.read(tokenized, track=track)
    elapsed = time.time() - t2
    if not again.provenance["cache_hit"]:
        print(
            f"FAIL — re-reading an identical request missed the cache "
            f"({elapsed:.3f}s). The cache key is unstable."
        )
        return 1
    print(f"OK — all probes passed. Cached re-read hit in {elapsed:.3f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
