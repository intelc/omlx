# SPDX-License-Identifier: Apache-2.0
"""Integration test for the CacheBlend scheduler pre-prefill hook."""

from __future__ import annotations

import types

import mlx.core as mx
import pytest

from omlx.patches import cacheblend


class _FakeTokenizer:
    def encode(self, s: str):
        return [ord(c) for c in s]


class _FakeLayerCache:
    """Stand-in for an mlx-lm KVCache just for attaching blend_metadata."""


class _FakePrefixCache:
    def __init__(self, primed: dict):
        self._primed = {tuple(k): v for k, v in primed.items()}

    def lookup_chunk_by_standalone_hash(self, tokens):
        return self._primed.get(tuple(tokens))


def _make_request(prompt: str, *, specprefill_enabled: bool = False):
    req = types.SimpleNamespace(
        prompt=prompt,
        tokenizer=_FakeTokenizer(),
        _specprefill_enabled=specprefill_enabled,
    )
    return req


def _make_settings(
    cacheblend_enabled: bool = True,
    specprefill_enabled: bool = False,
):
    return types.SimpleNamespace(
        cacheblend_enabled=cacheblend_enabled,
        cacheblend_recompute_ratio=0.2,
        cacheblend_check_layers=[1],
        cacheblend_special_str=" # # ",
        cacheblend_chunk_min_tokens=1,
        specprefill_enabled=specprefill_enabled,
    )


def test_hook_attaches_metadata_on_cache_hit(monkeypatch):
    # No-op patcher so this test doesn't need a real mlx-lm model.
    monkeypatch.setattr(cacheblend, "patch_model_for_cacheblend", lambda m: None)

    class _Handle:
        per_layer_kv = [(mx.zeros((1, 3, 8)), mx.zeros((1, 3, 8)))]

    cache = [_FakeLayerCache(), _FakeLayerCache()]
    req = _make_request("AAA # # BBB # # Q")
    tokens_AAA = _FakeTokenizer().encode("AAA")
    prefix_cache = _FakePrefixCache(primed={tuple(tokens_AAA): _Handle()})

    ok = cacheblend.try_cacheblend_prefill(
        req, model=object(), prefix_cache=prefix_cache, settings=_make_settings(),
        cache=cache,
    )

    assert ok is True
    for layer_cache in cache:
        assert hasattr(layer_cache, "blend_metadata")
    meta = cache[0].blend_metadata
    # AAA (cached) + BBB + Q
    assert meta.total_len == len(tokens_AAA) + len(_FakeTokenizer().encode("BBB")) + 1
    assert meta.recompute_ratio == 0.2
    assert meta.check_layer == 1


def test_hook_noop_when_disabled():
    cache = [_FakeLayerCache()]
    req = _make_request("A # # B # # Q")
    settings = _make_settings(cacheblend_enabled=False)
    prefix_cache = _FakePrefixCache(primed={})
    ok = cacheblend.try_cacheblend_prefill(
        req, model=object(), prefix_cache=prefix_cache, settings=settings,
        cache=cache,
    )
    assert ok is False
    assert not hasattr(cache[0], "blend_metadata")


def test_hook_noop_when_no_separator():
    cache = [_FakeLayerCache()]
    req = _make_request("no-separator-in-prompt")
    prefix_cache = _FakePrefixCache(primed={})
    ok = cacheblend.try_cacheblend_prefill(
        req, model=object(), prefix_cache=prefix_cache, settings=_make_settings(),
        cache=cache,
    )
    assert ok is False
    assert not hasattr(cache[0], "blend_metadata")


def test_hook_noop_when_all_chunks_cold():
    """If no chunk is CACHED, we fall back to standard prefill — no metadata attached."""
    cache = [_FakeLayerCache()]
    req = _make_request("AAA # # BBB # # Q")
    prefix_cache = _FakePrefixCache(primed={})   # nothing cached
    ok = cacheblend.try_cacheblend_prefill(
        req, model=object(), prefix_cache=prefix_cache, settings=_make_settings(),
        cache=cache,
    )
    assert ok is False
    assert not hasattr(cache[0], "blend_metadata")


def test_hook_raises_on_specprefill_conflict():
    cache = [_FakeLayerCache()]
    req = _make_request("A # # B # # Q", specprefill_enabled=True)
    settings = _make_settings(specprefill_enabled=True)
    prefix_cache = _FakePrefixCache(primed={})
    with pytest.raises(ValueError, match="cannot both be enabled"):
        cacheblend.try_cacheblend_prefill(
            req, model=object(), prefix_cache=prefix_cache, settings=settings,
            cache=cache,
        )


def test_commit_blend_cold_chunks_writes_to_prefix_cache():
    """End-to-end: metadata attached to cache, mock K/V available via .keys,
    commit_blend_cold_chunks should submit each COLD doc chunk to the prefix
    cache's commit_chunk_as_standalone. Query chunk is skipped.
    """
    meta = cacheblend.BlendMetadata(
        chunks=[
            cacheblend.ChunkInfo(tokens=[1, 2, 3], kind=cacheblend.COLD, start_pos=0),
            cacheblend.ChunkInfo(tokens=[4, 5], kind=cacheblend.COLD, start_pos=3),
            cacheblend.ChunkInfo(tokens=[9], kind=cacheblend.COLD, start_pos=5),  # query
        ],
        total_len=6,
    )

    class _LayerCache:
        def __init__(self):
            # Shape [1, num_heads=1, tokens=6, head_dim=4] — batched.
            self.keys = mx.arange(6 * 4, dtype=mx.float32).reshape(1, 1, 6, 4)
            self.values = -mx.arange(6 * 4, dtype=mx.float32).reshape(1, 1, 6, 4)

    layer_caches = [_LayerCache(), _LayerCache()]
    layer_caches[0].blend_metadata = meta

    req = types.SimpleNamespace(cache=layer_caches)

    committed_list = []

    class _Prefix:
        def commit_chunk_as_standalone(self, tokens, per_layer_kv):
            committed_list.append((tokens, len(per_layer_kv)))
            return True

    n = cacheblend.commit_blend_cold_chunks(req, _Prefix())
    # 2 doc chunks committed; query chunk skipped.
    assert n == 2
    assert committed_list == [([1, 2, 3], 2), ([4, 5], 2)]


def test_commit_blend_cold_chunks_noop_without_metadata():
    """No blend_metadata attached → returns 0, doesn't call prefix cache."""
    class _LayerCache:
        keys = mx.zeros((1, 1, 1, 1))
        values = mx.zeros((1, 1, 1, 1))

    req = types.SimpleNamespace(cache=[_LayerCache()])

    class _Prefix:
        def commit_chunk_as_standalone(self, *a, **kw):
            raise AssertionError("should not be called")

    n = cacheblend.commit_blend_cold_chunks(req, _Prefix())
    assert n == 0


def test_commit_blend_cold_chunks_noop_without_prefix_cache():
    """Prefix cache None → returns 0 gracefully."""
    req = types.SimpleNamespace(cache=[])
    n = cacheblend.commit_blend_cold_chunks(req, None)
    assert n == 0
