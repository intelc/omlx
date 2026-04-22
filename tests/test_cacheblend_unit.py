# SPDX-License-Identifier: Apache-2.0
"""Unit tests for omlx.patches.cacheblend."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches import cacheblend


def test_hkvd_score_selects_largest_diff_tokens():
    # 4 heads, 8 tokens, 16 dim
    rng = np.random.default_rng(42)
    k_cached = rng.normal(size=(4, 8, 16)).astype(np.float32)
    k_fresh = k_cached.copy()
    # Inject large deviation at tokens 2 and 5
    k_fresh[:, 2, :] += 10.0
    k_fresh[:, 5, :] += 10.0

    cold_mask = mx.zeros((8,), dtype=mx.bool_)  # no COLD tokens

    indices = cacheblend.hkvd_score(
        k_fresh=mx.array(k_fresh),
        k_cached=mx.array(k_cached),
        cold_token_mask=cold_mask,
        recompute_ratio=0.25,   # top-25% of 8 = 2 tokens
    )

    selected = sorted(indices.tolist())
    assert selected == [2, 5]


def test_hkvd_score_always_includes_cold_tokens():
    rng = np.random.default_rng(0)
    k = rng.normal(size=(2, 10, 8)).astype(np.float32)

    cold_mask = mx.array([False, False, False, True, False, False, False, False, False, False])
    # recompute_ratio would pick 1 token; COLD token 3 must be included
    indices = cacheblend.hkvd_score(
        k_fresh=mx.array(k),
        k_cached=mx.array(k),   # zero diff everywhere
        cold_token_mask=cold_mask,
        recompute_ratio=0.1,
    )

    assert 3 in indices.tolist()


def test_rotate_k_by_offsets_matches_direct_computation():
    """K rotated by offset θ at position 0 should equal K computed fresh at position θ.

    Uses a simple RoPE implementation inline to avoid depending on the
    model's rotary module. CacheBlend's production path uses the model's
    own rotary; this test validates the generalization math.
    """
    head_dim = 32
    num_heads = 2
    total_len = 6

    rng = np.random.default_rng(7)
    x = mx.array(rng.normal(size=(num_heads, total_len, head_dim)).astype(np.float32))

    def apply_rope_at(x, start_pos):
        # Rotate pairs (2i, 2i+1) by angle m * theta_i; theta_i = base^(-2i/d)
        base = 10000.0
        d = x.shape[-1]
        positions = mx.arange(start_pos, start_pos + x.shape[1]).astype(mx.float32)
        inv_freq = base ** (-mx.arange(0, d, 2).astype(mx.float32) / d)
        freqs = positions[:, None] * inv_freq[None, :]   # [T, d/2]
        cos = mx.cos(freqs)                              # [T, d/2]
        sin = mx.sin(freqs)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        out = mx.stack([rot_even, rot_odd], axis=-1)
        return out.reshape(x.shape)

    # Cached case: K was RoPE'd at position 0; we want it rotated to position 4.
    k_at_0 = apply_rope_at(x, start_pos=0)
    k_at_4_direct = apply_rope_at(x, start_pos=4)

    # Re-rotate k_at_0 by +4 using the offset utility
    offsets = mx.array([4] * total_len, dtype=mx.int32)
    k_at_4_via_offset = cacheblend.rotate_k_by_offsets(
        k=k_at_0,
        offsets=offsets,
        head_dim=head_dim,
        base=10000.0,
    )

    assert mx.allclose(k_at_4_via_offset, k_at_4_direct, atol=1e-5).item()


def test_sparse_q_selected_against_full_kv_matches_dense_subset():
    """sparse_attn_selected_q(Q_selected, K_full, V_full) must equal
    the corresponding rows of dense attention(Q_full, K_full, V_full)."""
    num_heads = 4
    total_len = 12
    head_dim = 16
    rng = np.random.default_rng(3)
    q = mx.array(rng.normal(size=(num_heads, total_len, head_dim)).astype(np.float32))
    k = mx.array(rng.normal(size=(num_heads, total_len, head_dim)).astype(np.float32))
    v = mx.array(rng.normal(size=(num_heads, total_len, head_dim)).astype(np.float32))
    scale = 1.0 / float(head_dim) ** 0.5

    # Causal mask: shape [total_len, total_len], True = allow attend
    allow = mx.tril(mx.ones((total_len, total_len), dtype=mx.bool_))
    mask_full = mx.where(allow, mx.array(0.0), mx.array(-1e9))   # additive

    scores_full = (q @ k.transpose(0, 2, 1)) * scale + mask_full
    attn_full = mx.softmax(scores_full, axis=-1)
    out_dense = attn_full @ v   # [H, T, D]

    selected = mx.array([3, 7, 11], dtype=mx.int32)
    out_sparse = cacheblend.sparse_attn_selected_q(
        q=q, k=k, v=v,
        selected_q_indices=selected,
        scale=scale,
    )

    expected = out_dense[:, selected, :]
    assert mx.allclose(out_sparse, expected, atol=1e-5).item()


class _FakeTokenizer:
    """Minimal tokenizer for split_prompt_on_separator tests."""
    def __init__(self):
        # Reserve a few pseudo-token ids for the separator tokens.
        self.vocab = {"A": 1, "B": 2, "C": 3, "D": 4, "Q": 5, " ": 6, "#": 7}

    def encode(self, s: str) -> list[int]:
        # Character-level so tests are predictable.
        return [self.vocab.setdefault(ch, len(self.vocab) + 1) for ch in s]


def test_split_prompt_basic():
    tok = _FakeTokenizer()
    sep = " # # "
    chunks, query_tokens = cacheblend.split_prompt_on_separator(
        prompt="AAA # # BBB # # Q", tokenizer=tok, separator=sep,
    )
    # Expect 2 non-query chunks + 1 query suffix
    assert [tok.encode("AAA"), tok.encode("BBB")] == chunks
    assert query_tokens == tok.encode("Q")


def test_split_prompt_no_separator_returns_none():
    tok = _FakeTokenizer()
    result = cacheblend.split_prompt_on_separator(
        prompt="AAABBBQ", tokenizer=tok, separator=" # # ",
    )
    assert result is None


def test_split_prompt_leading_and_trailing_separators_rejected():
    tok = _FakeTokenizer()
    # Leading separator: treat as no valid split, fall back.
    assert cacheblend.split_prompt_on_separator(
        prompt=" # # AAA # # Q", tokenizer=tok, separator=" # # ",
    ) is None
    # Trailing separator: zero-length query, reject.
    assert cacheblend.split_prompt_on_separator(
        prompt="AAA # # Q # # ", tokenizer=tok, separator=" # # ",
    ) is None


class _FakePrefixCache:
    """Captures lookup_chunk_by_standalone_hash behavior for classify tests."""
    def __init__(self, known_tokens_to_handle):
        self._known = {tuple(k): v for k, v in known_tokens_to_handle.items()}

    def lookup_chunk_by_standalone_hash(self, tokens):
        return self._known.get(tuple(tokens))


def test_classify_chunks_mixed_cached_and_cold():
    class _Handle:
        per_layer_kv = [(None, None)]
    cache = _FakePrefixCache({(1, 2, 3): _Handle()})

    meta = cacheblend.build_blend_metadata(
        chunk_token_lists=[[1, 2, 3], [4, 5, 6, 7]],
        query_tokens=[9, 9],
        prefix_cache=cache,
        chunk_min_tokens=1,
    )

    assert len(meta.chunks) == 3        # 2 doc chunks + 1 query chunk
    assert meta.chunks[0].kind == cacheblend.CACHED
    assert meta.chunks[1].kind == cacheblend.COLD
    assert meta.chunks[2].kind == cacheblend.COLD    # query always cold
    assert meta.chunks[0].start_pos == 0
    assert meta.chunks[1].start_pos == 3
    assert meta.chunks[2].start_pos == 7
    assert meta.total_len == 9
    # Cold mask: positions 3..8 are cold, 0..2 are cached
    assert meta.cold_token_mask.tolist() == [False, False, False, True, True, True, True, True, True]


def test_classify_chunks_all_cold_returns_none():
    cache = _FakePrefixCache({})
    meta = cacheblend.build_blend_metadata(
        chunk_token_lists=[[1, 2, 3]],
        query_tokens=[9],
        prefix_cache=cache,
        chunk_min_tokens=1,
    )
    assert meta is None   # signal to fall back to standard prefill
