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
