"""Tests for the j-space readout API."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.tiny import TinyDecoder, tiny_lens
from vibekampff.models import REGISTRY, get_model
from vibekampff.readout import LensReader, Tokenized


@pytest.fixture
def reader(tmp_path):
    model = TinyDecoder()
    return LensReader(
        model,
        tiny_lens(model),
        spec=get_model("qwen3-1.7b"),  # only used for provenance here
        device="cpu",
        dtype="float32",
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture
def tokenized():
    ids = [1, 5, 9, 14, 20, 3, 7, 11]
    return Tokenized(ids=ids, tokens=[f"t{i}" for i in ids], text="whatever")


def test_reads_every_layer_and_position_by_default(reader, tokenized):
    out = reader.read(tokenized, top_k=5)
    assert out.layers == list(reader.lens.source_layers)
    assert out.positions == list(range(len(tokenized)))
    assert out.topk_ids.shape == (len(out.layers), len(tokenized), 5)


def test_topk_is_ordered_best_first(reader, tokenized):
    out = reader.read(tokenized, top_k=5)
    diffs = np.diff(out.topk_logits, axis=-1)
    assert (diffs <= 1e-5).all(), "top-k logits must be non-increasing"


def test_ranks_are_exact_and_agree_with_topk(reader, tokenized):
    """A token at rank 0 must be the same token top-k reports first."""
    track = [3, 7, 11, 20]
    out = reader.read(tokenized, top_k=5, track=track)
    for li, _layer in enumerate(out.layers):
        for pi in range(len(out.positions)):
            for ti, token_id in enumerate(track):
                rank = out.tracked_ranks[li, pi, ti]
                if rank == 0:
                    assert out.topk_ids[li, pi, 0] == token_id


def test_rank_matches_a_hand_computed_argsort(reader, tokenized):
    """Cross-check the count-how-many-beat-it shortcut against a real sort."""
    track = [2, 13, 29]
    out = reader.read(tokenized, top_k=3, track=track)

    # Recompute one cell the slow, obvious way.
    from jlens.hooks import ActivationRecorder

    layer, position = out.layers[2], 4
    ids = torch.tensor([tokenized.ids])
    with ActivationRecorder(reader.model.layers, at=[layer]) as rec:
        reader.model.forward(ids)
        residual = rec.activations[layer].detach()[0][position].float()
    logits = reader.model.unembed(reader.lens.transport(residual, layer))
    order = logits.argsort(descending=True).tolist()

    for token_id in track:
        expected = order.index(token_id)
        got = out.rank(token_id, layer, position)
        assert got == expected, f"token {token_id}: got {got}, expected {expected}"


def test_band_min_rank_finds_the_best_cell(reader, tokenized):
    track = [6]
    out = reader.read(tokenized, top_k=3, track=track)
    rank, layer, position = out.band_min_rank(6)
    assert rank == out.tracked_ranks[:, :, 0].min()
    assert out.rank(6, layer, position) == rank


def test_band_min_rank_respects_the_band(reader, tokenized):
    out = reader.read(tokenized, top_k=3, track=[6])
    band = out.layers[1:3]
    rank, layer, _ = out.band_min_rank(6, band=band)
    assert layer in band
    assert rank >= out.band_min_rank(6)[0], "narrowing a band cannot improve the best"


def test_cache_round_trips_and_reports_a_hit(reader, tokenized):
    first = reader.read(tokenized, top_k=4, track=[3, 9])
    assert first.provenance["cache_hit"] is False
    second = reader.read(tokenized, top_k=4, track=[3, 9])
    assert second.provenance["cache_hit"] is True
    np.testing.assert_array_equal(first.topk_ids, second.topk_ids)
    np.testing.assert_array_equal(first.tracked_ranks, second.tracked_ranks)


def test_cache_key_separates_different_requests(reader, tokenized):
    a = reader.read(tokenized, top_k=4, track=[3])
    b = reader.read(tokenized, top_k=4, track=[9])
    assert a.provenance["cache_key"] != b.provenance["cache_key"]
    assert b.provenance["cache_hit"] is False


def test_cache_key_changes_with_lens_revision(reader, tokenized):
    reader.read(tokenized, top_k=4, track=[3])
    reader.lens_revision = "deadbeef"
    again = reader.read(tokenized, top_k=4, track=[3])
    assert again.provenance["cache_hit"] is False, (
        "a different lens revision must not reuse a cached readout"
    )


def test_output_topk_is_the_models_own_prediction(reader, tokenized):
    """The final-layer row must be the model's real output, not a lens read."""
    out = reader.read(tokenized, top_k=3)
    ids = torch.tensor([tokenized.ids])
    hidden = reader.model.forward(ids).last_hidden_state[0]
    expected = reader.model.unembed(hidden.float()).topk(3, dim=-1).indices
    np.testing.assert_array_equal(out.output_topk_ids, expected.numpy())


def test_rejects_unfitted_layers(reader, tokenized):
    with pytest.raises(ValueError, match="not fitted"):
        reader.read(tokenized, layers=[reader.model.n_layers - 1])


def test_rejects_out_of_range_positions(reader, tokenized):
    with pytest.raises(ValueError, match="out of range"):
        reader.read(tokenized, positions=[0, len(tokenized)])


def test_untracked_token_lookup_is_a_clear_error(reader, tokenized):
    out = reader.read(tokenized, top_k=3, track=[1])
    with pytest.raises(KeyError, match="re-read"):
        out.rank(2, out.layers[0], 0)


def test_provenance_carries_what_the_manifest_needs(reader):
    p = reader.provenance
    for key in ("model_hf_id", "lens_repo", "lens_file", "lens_revision", "dtype"):
        assert p[key], f"{key} missing from provenance"


def test_registry_entries_are_self_consistent():
    for key, spec in REGISTRY.items():
        assert spec.key == key
        assert key.replace("qwen3-", "").replace("-", "") in spec.lens_file.lower()
        assert spec.n_layers > 0 and spec.d_model > 0
