"""Model + published-lens registry for the spike.

Every lens here is a pre-fitted Jacobian lens published by Neuronpedia at
``neuronpedia/jacobian-lens`` (MIT), built with Anthropic's ``jlens``
(Apache-2.0). We fit nothing; we download and apply.

Sizes are the bf16 model download plus the lens file, which is what actually
has to fit in memory and on disk.
"""

from __future__ import annotations

from dataclasses import dataclass

LENS_REPO = "neuronpedia/jacobian-lens"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    lens_file: str
    n_layers: int
    d_model: int
    approx_model_gb: float
    approx_lens_gb: float

    @property
    def approx_total_gb(self) -> float:
        return self.approx_model_gb + self.approx_lens_gb

    def band_guess(self) -> range:
        """A stand-in band, used only to size the cost measurement.

        This is NOT an estimate of where the workspace band is. The readout
        check deliberately sweeps every layer instead of trusting this, and
        the first run showed why: on Qwen3-1.7B the bridge entity surfaced at
        layers 18-26 of 28, well above the middle third. Locating the real
        band is issue #6's job and must not be taken from here.

        It exists so the cost sweep reads out over a realistic *number* of
        layers, which is what drives memory and time.
        """
        return range(self.n_layers // 3, (2 * self.n_layers) // 3)


def _lens_path(slug: str, stem: str) -> str:
    return f"{slug}/jlens/Salesforce-wikitext/{stem}_jacobian_lens.pt"


REGISTRY: dict[str, ModelSpec] = {
    # Smallest real Qwen3 with a published lens. Fast enough to run on CPU,
    # which makes it the right choice for proving the code path.
    "qwen3-1.7b": ModelSpec(
        key="qwen3-1.7b",
        hf_id="Qwen/Qwen3-1.7B",
        lens_file=_lens_path("qwen3-1.7b", "Qwen3-1.7B"),
        n_layers=28,
        d_model=2048,
        approx_model_gb=3.4,
        approx_lens_gb=0.23,
    ),
    "qwen3-4b": ModelSpec(
        key="qwen3-4b",
        hf_id="Qwen/Qwen3-4B",
        lens_file=_lens_path("qwen3-4b", "Qwen3-4B"),
        n_layers=36,
        d_model=2560,
        approx_model_gb=8.0,
        approx_lens_gb=0.47,
    ),
    # The intended analysis model for the project (issue #1).
    "qwen3-8b": ModelSpec(
        key="qwen3-8b",
        hf_id="Qwen/Qwen3-8B",
        lens_file=_lens_path("qwen3-8b", "Qwen3-8B"),
        n_layers=36,
        d_model=4096,
        approx_model_gb=16.0,
        approx_lens_gb=1.17,
    ),
}

DEFAULT_MODEL = "qwen3-8b"


def get(key: str) -> ModelSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise SystemExit(
            f"unknown model {key!r}; available: {', '.join(sorted(REGISTRY))}"
        ) from None
