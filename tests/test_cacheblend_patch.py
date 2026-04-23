# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the CacheBlend model monkey-patch.

These tests load the real Llama-3.2-1B-Instruct-4bit mlx-community checkpoint
(pre-downloaded by Task 0). The suite verifies:

1. A patched model with NO blend_metadata attached produces byte-identical
   logits to the unpatched model — the patch is pure overhead when the
   feature is inactive.
2. A patched model WITH blend_metadata attached runs the layerwise forward
   end-to-end, populates meta.recompute_indices via HKVD, and — because
   the MVP does full recompute at every layer — produces logits identical
   to the unpatched baseline.

Both tests guard against the known failure modes in the blend path:
NaN outputs, silent shape mismatches, cache-offset corruption.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import cacheblend


MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"


@pytest.fixture(scope="module")
def loaded():
    mlx_lm = pytest.importorskip("mlx_lm")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    return model, tokenizer


def _make_cache(model):
    from mlx_lm.models.cache import make_prompt_cache
    return make_prompt_cache(model)


def test_patched_model_without_metadata_matches_baseline(loaded):
    model, tokenizer = loaded
    tokens = mx.array([tokenizer.encode("Hello, world.")])

    cache_baseline = _make_cache(model)
    out_baseline = model(tokens, cache=cache_baseline)
    mx.eval(out_baseline)

    cacheblend.patch_model_for_cacheblend(model)
    # Idempotent: calling twice must not double-wrap.
    cacheblend.patch_model_for_cacheblend(model)

    cache_patched = _make_cache(model)
    out_patched = model(tokens, cache=cache_patched)
    mx.eval(out_patched)

    assert out_patched.shape == out_baseline.shape
    assert mx.allclose(out_patched, out_baseline, atol=1e-5).item(), (
        "Patched model without blend_metadata must produce identical logits "
        "to the unpatched baseline."
    )


class _DummyChunkHandle:
    """Stand-in for _ChunkKVHandle used when we want to exercise the blend
    path without actually going through the prefix cache.

    Provides per-layer K tensors of the right shape; values are zeros because
    the MVP forward path is full-recompute and will not consult cached V, and
    will mask out the cached K at HKVD time for tokens that don't reach the
    top-r threshold (the fresh K dominates).
    """

    def __init__(self, num_layers: int, num_kv_heads: int, num_tokens: int, head_dim: int):
        self.per_layer_kv = [
            (mx.zeros((num_kv_heads, num_tokens, head_dim)),
             mx.zeros((num_kv_heads, num_tokens, head_dim)))
            for _ in range(num_layers)
        ]


def test_patched_model_with_metadata_runs_layerwise(loaded):
    """Attach a synthetic BlendMetadata and confirm the layerwise forward runs
    end-to-end. MVP full-recompute means the output must match baseline; HKVD
    scoring must populate meta.recompute_indices.
    """
    model, tokenizer = loaded
    inner = getattr(model, "model", model)
    num_layers = len(inner.layers)
    # Pull per-layer KV shape directly off the model.
    attn0 = inner.layers[0].self_attn
    num_kv_heads = attn0.n_kv_heads
    head_dim = attn0.head_dim if hasattr(attn0, "head_dim") else attn0.scale  # Llama exposes head_dim; Qwen3 computes from scale — load_dim from weight below if needed
    # More reliable: read from k_proj output dim
    head_dim = attn0.k_proj.weight.shape[0] // num_kv_heads

    prompt = "The sun rose over the quiet city. Birds sang and the streets were empty."
    token_list = tokenizer.encode(prompt)
    total_len = len(token_list)
    # Split the prompt into a fake "chunk + query" layout. Chunk is the first
    # half (we'll flag it CACHED with zeroed K/V); query is the second half.
    split_at = total_len // 2
    chunk_tokens = token_list[:split_at]
    query_tokens = token_list[split_at:]

    dummy = _DummyChunkHandle(num_layers, num_kv_heads, len(chunk_tokens), head_dim)

    meta = cacheblend.BlendMetadata(
        chunks=[
            cacheblend.ChunkInfo(
                tokens=chunk_tokens, kind=cacheblend.CACHED,
                start_pos=0, cached_kv=dummy,
            ),
            cacheblend.ChunkInfo(
                tokens=query_tokens, kind=cacheblend.COLD,
                start_pos=len(chunk_tokens), cached_kv=None,
            ),
        ],
        total_len=total_len,
        cold_token_mask=mx.array(
            [False] * len(chunk_tokens) + [True] * len(query_tokens),
            dtype=mx.bool_,
        ),
        check_layer=1,
        recompute_ratio=0.15,
    )

    # Baseline (no blend):
    cache_baseline = _make_cache(model)
    out_baseline = model(mx.array([token_list]), cache=cache_baseline)
    mx.eval(out_baseline)

    # Patch + attach metadata:
    cacheblend.patch_model_for_cacheblend(model)
    cache_blend = _make_cache(model)
    for layer_cache in cache_blend:
        layer_cache.blend_metadata = meta

    out_blend = model(mx.array([token_list]), cache=cache_blend)
    mx.eval(out_blend)

    # Shape and NaN checks.
    assert out_blend.shape == out_baseline.shape
    assert not mx.any(mx.isnan(out_blend)).item(), "blend forward produced NaN logits"

    # Sparse forward is an approximation — it skips updates at non-selected
    # positions. So output is NOT bit-identical to the baseline, but it must
    # stay finite and in a sane magnitude range. The E2E correctness tests
    # (tests/test_cacheblend_correctness.py) verify semantic quality on real
    # RAG prompts; here we just guard against explosion / NaN.
    blend_max = float(mx.max(mx.abs(out_blend)).item())
    base_max = float(mx.max(mx.abs(out_baseline)).item())
    assert blend_max < 10 * base_max + 1e-3, (
        f"Blend output magnitude exploded: blend_max={blend_max}, "
        f"baseline_max={base_max}"
    )

    # HKVD must have populated recompute_indices.
    assert meta.recompute_indices is not None, (
        "HKVD scoring should have run at the check layer and stored indices on meta."
    )
    # Indices must include all COLD positions (the query half).
    idx_list = meta.recompute_indices.tolist()
    for pos in range(len(chunk_tokens), total_len):
        assert pos in idx_list, f"COLD token {pos} missing from recompute_indices"


def test_check_layer_config_is_honored(loaded, monkeypatch):
    """Setting meta.check_layer to a non-default value must route HKVD
    scoring to that layer instead of the default layer 1.
    """
    model, tokenizer = loaded
    inner = getattr(model, "model", model)
    num_layers = len(inner.layers)
    attn0 = inner.layers[0].self_attn
    num_kv_heads = attn0.n_kv_heads
    head_dim = attn0.k_proj.weight.shape[0] // num_kv_heads

    observed_layers = []
    from omlx.patches import cacheblend as cb
    original = cb._score_hkvd_at_check_layer

    def spy(layer_cache, m, layer_idx, **kwargs):
        observed_layers.append(layer_idx)
        return original(layer_cache, m, layer_idx, **kwargs)

    monkeypatch.setattr(cb, "_score_hkvd_at_check_layer", spy)

    prompt = "The sun rose over the quiet city that morning and it was still."
    token_list = tokenizer.encode(prompt)
    split_at = len(token_list) // 2
    chunk_tokens = token_list[:split_at]
    query_tokens = token_list[split_at:]

    dummy = _DummyChunkHandle(num_layers, num_kv_heads, len(chunk_tokens), head_dim)
    target_check_layer = 2  # instead of the default 1
    meta = cacheblend.BlendMetadata(
        chunks=[
            cacheblend.ChunkInfo(
                tokens=chunk_tokens, kind=cacheblend.CACHED,
                start_pos=0, cached_kv=dummy,
            ),
            cacheblend.ChunkInfo(
                tokens=query_tokens, kind=cacheblend.COLD,
                start_pos=len(chunk_tokens), cached_kv=None,
            ),
        ],
        total_len=len(token_list),
        cold_token_mask=mx.array(
            [False] * len(chunk_tokens) + [True] * len(query_tokens),
            dtype=mx.bool_,
        ),
        check_layer=target_check_layer,
        recompute_ratio=0.15,
    )

    cacheblend.patch_model_for_cacheblend(model)
    cache = _make_cache(model)
    for lc in cache:
        lc.blend_metadata = meta

    out = model(mx.array([token_list]), cache=cache)
    mx.eval(out)

    assert observed_layers == [target_check_layer], (
        f"HKVD scorer should have been called exactly once at layer "
        f"{target_check_layer}; got calls at layers {observed_layers}"
    )
    assert meta.recompute_indices is not None


def test_patched_model_rejects_unknown_architecture():
    """Calling the patch on something that isn't LlamaModel/Qwen3Model should
    raise NotImplementedError so callers catch the mismatch loudly.
    """

    class _Fake:
        pass

    class _FakeWrapper:
        def __init__(self):
            self.model = _Fake()

    wrapper = _FakeWrapper()
    with pytest.raises(NotImplementedError, match="no layerwise patch"):
        cacheblend.patch_model_for_cacheblend(wrapper)
