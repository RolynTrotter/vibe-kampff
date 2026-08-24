"""Model and published-lens registry.

Every lens named here is a pre-fitted Jacobian lens published by Neuronpedia
(`neuronpedia/jacobian-lens`, MIT), built with Anthropic's `jlens`
(Apache-2.0). This project fits nothing; see the fitting ticket for when that
changes.

The registry is pinned to a lens-repo revision on purpose. A result computed
against a silently-updated lens is not reproducible, so the revision travels
into every run manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

#: HuggingFace repo hosting the pre-fitted lenses.
LENS_REPO = "neuronpedia/jacobian-lens"

#: Pinned lens-repo revision. Bumping this invalidates every cached readout
#: (the revision is part of the cache key) and should be a deliberate,
#: recorded decision — not a drive-by update.
LENS_REVISION = "0731326edff4ae730ffc5356fe1a4728c748b3a6"


@dataclass(frozen=True)
class ModelSpec:
    """A model and the published lens fitted for it."""

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


def _lens_path(slug: str, stem: str) -> str:
    return f"{slug}/jlens/Salesforce-wikitext/{stem}_jacobian_lens.pt"


REGISTRY: dict[str, ModelSpec] = {
    # Small enough to run on CPU. The right choice for tests and for any
    # change where iteration speed matters more than the result.
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
    # The project's analysis model. Verified end to end on MPS.
    "qwen3-8b": ModelSpec(
        key="qwen3-8b",
        hf_id="Qwen/Qwen3-8B",
        lens_file=_lens_path("qwen3-8b", "Qwen3-8B"),
        n_layers=36,
        d_model=4096,
        approx_model_gb=16.0,
        approx_lens_gb=1.17,
    ),
    # Larger options, should 8B prove too small to show anything. Both have
    # published lenses and both fit on a 64 GB machine for forward passes.
    "qwen3-14b": ModelSpec(
        key="qwen3-14b",
        hf_id="Qwen/Qwen3-14B",
        lens_file=_lens_path("qwen3-14b", "Qwen3-14B"),
        n_layers=40,
        d_model=5120,
        approx_model_gb=28.0,
        approx_lens_gb=2.0,
    ),
    "qwen3-32b": ModelSpec(
        key="qwen3-32b",
        hf_id="Qwen/Qwen3-32B",
        lens_file=_lens_path("qwen3-32b", "Qwen3-32B"),
        n_layers=64,
        d_model=5120,
        approx_model_gb=64.0,
        approx_lens_gb=3.2,
    ),
}

DEFAULT_MODEL = "qwen3-8b"


def get_model(key: str) -> ModelSpec:
    """Look up a spec by key, with a useful error for typos."""
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown model {key!r}; available: {', '.join(sorted(REGISTRY))}"
        ) from None
