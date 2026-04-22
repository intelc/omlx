# SPDX-License-Identifier: Apache-2.0
"""Integration tests for CacheBlend additions to BlockAwarePrefixCache.

Uses the real cache stack built via CacheFactory.create_full_cache_stack
(same pattern as tests/test_cache_factory.py::test_create_full_cache_stack_enabled).
"""

from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest

from omlx.cache.factory import CacheConfig, CacheFactory


def _build_cache(tmp_path: Path):
    config = CacheConfig(
        paged_ssd_cache_dir=tmp_path / "cache",
        model_name="test-cacheblend",
    )
    mock_model = MagicMock()
    mock_model.layers = [MagicMock() for _ in range(2)]
    stack = CacheFactory.create_full_cache_stack(config, model=mock_model)
    return stack["prefix_cache"]


def _fabricate_per_layer_kv(num_layers: int, num_heads: int, num_tokens: int, head_dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for layer in range(num_layers):
        k = mx.array(rng.normal(size=(num_heads, num_tokens, head_dim)).astype(np.float32))
        v = mx.array(rng.normal(size=(num_heads, num_tokens, head_dim)).astype(np.float32))
        out.append((k, v))
    return out


def test_commit_and_lookup_chunk_roundtrip(tmp_path: Path):
    cache = _build_cache(tmp_path)
    # Two full blocks + one partial block: 300 tokens at block_size=256.
    tokens = list(range(300))
    num_layers, num_heads, head_dim = 2, 1, 8
    per_layer_kv = _fabricate_per_layer_kv(num_layers, num_heads, len(tokens), head_dim, seed=42)

    ok = cache.commit_chunk_as_standalone(tokens, per_layer_kv)
    assert ok is True

    handle = cache.lookup_chunk_by_standalone_hash(tokens)
    assert handle is not None
    assert len(handle.per_layer_kv) == num_layers
    for layer_idx in range(num_layers):
        k_stored, v_stored = handle.per_layer_kv[layer_idx]
        k_orig, v_orig = per_layer_kv[layer_idx]
        assert k_stored.shape == k_orig.shape
        assert v_stored.shape == v_orig.shape
        assert mx.allclose(k_stored, k_orig, atol=1e-6).item()
        assert mx.allclose(v_stored, v_orig, atol=1e-6).item()


def test_lookup_returns_none_on_miss(tmp_path: Path):
    cache = _build_cache(tmp_path)
    handle = cache.lookup_chunk_by_standalone_hash([999, 998, 997, 996])
    assert handle is None


def test_commit_returns_false_without_ssd_cache():
    """If paged_ssd_cache is None, commit should fail gracefully."""
    from omlx.cache.prefix_cache import BlockAwarePrefixCache
    from omlx.cache.paged_cache import PagedCacheManager

    mock_model = MagicMock()
    mock_model.layers = [MagicMock() for _ in range(1)]

    # PagedCacheManager.__init__ signature: block_size, max_blocks, enable_caching, model_name, initial_blocks
    paged = PagedCacheManager(
        block_size=256,
        max_blocks=8,
        enable_caching=True,
        model_name="test",
    )
    cache = BlockAwarePrefixCache(mock_model, paged, paged_ssd_cache_manager=None)

    per_layer_kv = _fabricate_per_layer_kv(1, 1, 50, 8)
    ok = cache.commit_chunk_as_standalone(list(range(50)), per_layer_kv)
    assert ok is False

    handle = cache.lookup_chunk_by_standalone_hash(list(range(50)))
    assert handle is None
