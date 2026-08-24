"""A tiny CPU decoder implementing jlens's LensModel, for fast tests.

Adapted from `tests/tiny.py` in anthropics/jacobian-lens (Apache-2.0). Lets
the readout API exercise its real code path — hooks, transport, unembed —
without downloading gigabytes.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.linear(hidden)


class _ByteTokenizer:
    bos_token_id = 0

    def decode(self, ids, **_kw) -> str:
        return "".join(chr(96 + int(i)) for i in ids)


class TinyDecoder(nn.Module):
    """A small residual stack with the LensModel surface."""

    def __init__(
        self, n_layers: int = 6, d_model: int = 8, vocab_size: int = 32, seed: int = 0
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_layers = n_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tokenizer = _ByteTokenizer()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_ResidualBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    @property
    def input_device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=hidden)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(residual.float()))


def tiny_lens(model: TinyDecoder, layers=None):
    """An identity-transport JacobianLens over `model`'s layers.

    Identity keeps the arithmetic checkable by hand: the lens readout at a
    layer is then exactly the logit lens at that layer, so a test can assert
    against `unembed` directly.
    """
    from jlens import JacobianLens

    layers = list(range(model.n_layers - 1)) if layers is None else list(layers)
    return JacobianLens(
        {layer: torch.eye(model.d_model) for layer in layers},
        n_prompts=1,
        d_model=model.d_model,
    )
