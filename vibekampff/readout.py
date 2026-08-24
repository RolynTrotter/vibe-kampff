"""Lens acquisition and the j-space readout API.

Wraps `jlens` in the surface this project needs: fetch a pinned published
lens, run a transcript through the model, and reduce the layer x position
grid to something scoreable.

Two deliberate departures from `jlens`'s own convenience API:

1. **We drive the forward pass ourselves** rather than calling
   `JacobianLens.apply()`. `apply()` takes a *prompt string* and re-tokenizes
   it internally, which would silently desync from the token ids and position
   map the transcript layer produces. Reading from explicit ids is the whole
   basis of saying "the band lit up over the fabricated turn".

2. **Ranks are computed for a declared target set**, not stored for the whole
   vocabulary. Keeping full-vocab logits for every cell costs ~121 MB per
   layer at 200 positions; keeping ranks for a lexicon costs nothing. Adding
   a target after the fact means re-reading, which is cheap.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from vibekampff.models import LENS_REPO, LENS_REVISION, ModelSpec, get_model

#: Rank sentinel for "this token was not among the tracked set".
UNTRACKED = -1


@dataclass(frozen=True)
class Tokenized:
    """A rendered transcript, already tokenized.

    `ids` is the ground truth. Everything downstream indexes by position into
    this sequence, so it must be exactly what the model sees.
    """

    ids: list[int]
    tokens: list[str]
    text: str

    def __len__(self) -> int:
        return len(self.ids)

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.ids, separators=(",", ":")).encode()
        ).hexdigest()[:16]


@dataclass
class Readout:
    """The lens readout over a layer x position grid.

    Attributes:
        layers: Layer indices, ascending.
        positions: Absolute positions into the token sequence, ascending.
        tokens: The token string at each entry of `positions`.
        topk_ids: `[n_layers, n_positions, k]` token ids, best first.
        topk_logits: Matching lens logits.
        tracked_ids: Target token ids ranks were computed for.
        tracked_ranks: `[n_layers, n_positions, n_tracked]`, 0-based; 0 means
            the target was the lens's top choice at that cell.
        output_topk_ids: The model's own final-layer top-k at each position —
            what it is actually about to say. Kept because "the lens reads
            something the output does not" is the contrast that distinguishes
            workspace content from imminent output.
    """

    layers: list[int]
    positions: list[int]
    tokens: list[str]
    topk_ids: np.ndarray
    topk_logits: np.ndarray
    tracked_ids: list[int]
    tracked_ranks: np.ndarray
    output_topk_ids: np.ndarray
    provenance: dict = field(default_factory=dict)

    def _layer_index(self, layer: int) -> int:
        try:
            return self.layers.index(layer)
        except ValueError:
            raise KeyError(
                f"layer {layer} was not read; read layers {self.layers}"
            ) from None

    def _tracked_index(self, token_id: int) -> int:
        try:
            return self.tracked_ids.index(token_id)
        except ValueError:
            raise KeyError(
                f"token id {token_id} was not tracked; re-read with it in `track`"
            ) from None

    def rank(self, token_id: int, layer: int, position: int) -> int:
        """0-based rank of `token_id` at one cell."""
        p = self.positions.index(position)
        return int(
            self.tracked_ranks[
                self._layer_index(layer), p, self._tracked_index(token_id)
            ]
        )

    def band_min_rank(
        self, token_id: int, band: Sequence[int] | None = None
    ) -> tuple[int, int, int]:
        """Best rank for `token_id` across a band, and where it occurred.

        Returns `(rank, layer, position)`. This is the project's primary
        continuous measure: a hit at rank 1 is the paper's convention, but the
        rank itself carries far more than a thresholded hit does.
        """
        col = self.tracked_ranks[:, :, self._tracked_index(token_id)]
        rows = (
            range(len(self.layers))
            if band is None
            else [self._layer_index(layer) for layer in band if layer in self.layers]
        )
        if not list(rows):
            raise ValueError("band selects none of the read layers")
        sub = col[list(rows), :]
        flat = int(sub.argmin())
        r, c = divmod(flat, sub.shape[1])
        return int(sub[r, c]), self.layers[list(rows)[r]], self.positions[c]

    def top_at(self, layer: int, position: int, k: int = 5) -> list[tuple[int, float]]:
        """`(token_id, logit)` pairs at one cell, best first."""
        p = self.positions.index(position)
        li = self._layer_index(layer)
        return [
            (int(self.topk_ids[li, p, i]), float(self.topk_logits[li, p, i]))
            for i in range(min(k, self.topk_ids.shape[2]))
        ]


def fetch_lens(
    spec: ModelSpec,
    *,
    revision: str = LENS_REVISION,
    repo: str = LENS_REPO,
) -> tuple[object, Path]:
    """Download and load the published lens for `spec` at a pinned revision.

    Returns `(lens, path)`. Verifies that the lens agrees with the spec's
    `d_model` — loading the wrong file out of a multi-model repo is the
    easiest mistake to make here, and it fails loudly rather than producing
    plausible nonsense.
    """
    from huggingface_hub import hf_hub_download
    from jlens import JacobianLens

    path = Path(
        hf_hub_download(repo_id=repo, filename=spec.lens_file, revision=revision)
    )
    lens = JacobianLens.load(str(path))
    if lens.d_model != spec.d_model:
        raise ValueError(
            f"lens {spec.lens_file} has d_model={lens.d_model} but "
            f"{spec.hf_id} has d_model={spec.d_model} — wrong lens file"
        )
    return lens, path


class LensReader:
    """Reads j-space over a tokenized transcript.

    Holds a loaded model and lens; construct once and reuse. Loading is
    measured in tens of seconds and gigabytes, so building one of these per
    scenario is not viable.
    """

    def __init__(
        self,
        model,
        lens,
        *,
        spec: ModelSpec,
        lens_revision: str = LENS_REVISION,
        device: str = "cpu",
        dtype: str = "bfloat16",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.lens = lens
        self.spec = spec
        self.lens_revision = lens_revision
        self.device = device
        self.dtype = dtype
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(
        cls,
        model_key: str,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        cache_dir: str | Path | None = None,
        lens_revision: str = LENS_REVISION,
    ) -> LensReader:
        """Load model and lens from scratch. Downloads on first use."""
        import jlens
        import transformers

        spec = get_model(model_key)
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

        hf = transformers.AutoModelForCausalLM.from_pretrained(
            spec.hf_id, dtype=torch_dtype
        ).to(device)
        hf.eval()
        tok = transformers.AutoTokenizer.from_pretrained(spec.hf_id)
        model = jlens.from_hf(hf, tok)
        lens, _ = fetch_lens(spec, revision=lens_revision)
        return cls(
            model,
            lens,
            spec=spec,
            lens_revision=lens_revision,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
        )

    @property
    def provenance(self) -> dict:
        """Everything about this reader that could change a number.

        Goes straight into the run manifest. A result without one is not a
        result.
        """
        return {
            "model_key": self.spec.key,
            "model_hf_id": self.spec.hf_id,
            "n_layers": self.spec.n_layers,
            "d_model": self.spec.d_model,
            "lens_repo": LENS_REPO,
            "lens_file": self.spec.lens_file,
            "lens_revision": self.lens_revision,
            "lens_n_prompts": getattr(self.lens, "n_prompts", None),
            "device": self.device,
            "dtype": self.dtype,
        }

    # -- caching ---------------------------------------------------------

    def _cache_key(
        self,
        tokenized: Tokenized,
        layers: Sequence[int],
        positions: Sequence[int],
        top_k: int,
        track: Sequence[int],
    ) -> str:
        payload = {
            "provenance": self.provenance,
            "ids": list(tokenized.ids),
            "layers": list(layers),
            "positions": list(positions),
            "top_k": top_k,
            "track": list(track),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def _cache_load(self, key: str) -> dict | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def _cache_store(self, key: str, arrays: dict) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{key}.npz"
        tmp = path.with_suffix(".npz.tmp")
        # Write through a file handle: np.savez_compressed appends ".npz" to
        # any path that lacks it, which would silently produce
        # "<key>.npz.tmp.npz" and leave the rename with nothing to move.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(path)

    # -- the readout -----------------------------------------------------

    def read(
        self,
        tokenized: Tokenized,
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        top_k: int = 20,
        track: Sequence[int] = (),
    ) -> Readout:
        """Read the lens over a layer x position grid.

        Args:
            tokenized: The transcript. `ids` is authoritative.
            layers: Layers to read. Defaults to every fitted layer.
            positions: Absolute positions. Defaults to all of them.
            top_k: Top tokens kept per cell, for display and exploration.
            track: Token ids to compute exact ranks for. This is what
                scoring consumes; supply the lexicon here.
        """
        layers = sorted(self.lens.source_layers if layers is None else layers)
        unknown = set(layers) - set(self.lens.source_layers)
        if unknown:
            raise ValueError(
                f"layers {sorted(unknown)} are not fitted in this lens; "
                f"fitted range is {self.lens.source_layers[0]}-"
                f"{self.lens.source_layers[-1]}"
            )
        positions = (
            list(range(len(tokenized))) if positions is None else sorted(positions)
        )
        bad = [p for p in positions if not 0 <= p < len(tokenized)]
        if bad:
            raise ValueError(
                f"positions {bad} out of range for a {len(tokenized)}-token sequence"
            )
        track = list(dict.fromkeys(track))

        key = self._cache_key(tokenized, layers, positions, top_k, track)
        cached = self._cache_load(key)
        if cached is not None:
            return Readout(
                layers=layers,
                positions=positions,
                tokens=[tokenized.tokens[p] for p in positions],
                topk_ids=cached["topk_ids"],
                topk_logits=cached["topk_logits"],
                tracked_ids=track,
                tracked_ranks=cached["tracked_ranks"],
                output_topk_ids=cached["output_topk_ids"],
                provenance=self.provenance | {"cache_key": key, "cache_hit": True},
            )

        arrays = self._compute(tokenized, layers, positions, top_k, track)
        self._cache_store(key, arrays)
        return Readout(
            layers=layers,
            positions=positions,
            tokens=[tokenized.tokens[p] for p in positions],
            tracked_ids=track,
            provenance=self.provenance | {"cache_key": key, "cache_hit": False},
            **arrays,
        )

    @torch.no_grad()
    def _compute(
        self,
        tokenized: Tokenized,
        layers: list[int],
        positions: list[int],
        top_k: int,
        track: list[int],
    ) -> dict:
        from jlens.hooks import ActivationRecorder

        model = self.model
        final_layer = model.n_layers - 1
        record_at = sorted(set(layers) | {final_layer})

        input_ids = torch.tensor(
            [tokenized.ids], dtype=torch.long, device=model.input_device
        )
        with ActivationRecorder(model.layers, at=record_at) as rec:
            model.forward(input_ids)
            activations = {i: rec.activations[i].detach() for i in record_at}

        pos_index = torch.tensor(positions, dtype=torch.long)

        topk_ids = np.zeros((len(layers), len(positions), top_k), dtype=np.int32)
        topk_logits = np.zeros((len(layers), len(positions), top_k), dtype=np.float32)
        tracked_ranks = np.full(
            (len(layers), len(positions), len(track)), UNTRACKED, dtype=np.int32
        )

        for li, layer in enumerate(layers):
            residual = activations[layer][0][pos_index].float()
            logits = model.unembed(self.lens.transport(residual, layer)).float().cpu()
            values, indices = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
            topk_ids[li] = indices.numpy()
            topk_logits[li] = values.numpy()
            # Rank = how many vocabulary entries beat this one. Exact, and
            # cheaper than a full argsort over 150k entries per cell. Done one
            # target at a time on purpose: comparing all targets at once would
            # build a [positions, vocab, targets] boolean tensor, which is
            # ~900 MB at 200 positions and a 30-word lexicon.
            for ti, token_id in enumerate(track):
                beaten = (logits > logits[:, token_id : token_id + 1]).sum(dim=-1)
                tracked_ranks[li, :, ti] = beaten.numpy()
            del logits

        output = model.unembed(activations[final_layer][0][pos_index].float())
        output_topk_ids = (
            (output.float().cpu().topk(min(top_k, output.shape[-1]), dim=-1).indices)
            .numpy()
            .astype(np.int32)
        )

        return {
            "topk_ids": topk_ids,
            "topk_logits": topk_logits,
            "tracked_ranks": tracked_ranks,
            "output_topk_ids": output_topk_ids,
        }
