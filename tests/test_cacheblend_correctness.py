# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness: CacheBlend (full-recompute MVP) vs. baseline.

In the MVP layerwise forward each layer runs normally, so output MUST be
bit-identical to the un-blended path. These tests fail loudly if a future
change introduces any drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches import cacheblend


MODEL_ID_LLAMA = "mlx-community/Llama-3.2-1B-Instruct-4bit"
MODEL_ID_QWEN3 = "mlx-community/Qwen3-1.7B-4bit"

FIXTURE = Path(__file__).parent / "fixtures" / "cacheblend" / "rag_prompt_3doc.json"


def _load_fixture():
    with FIXTURE.open() as f:
        return json.load(f)


def _build_prompt(fixture):
    return fixture["separator"].join(fixture["docs"] + [fixture["query"]])


def _slice_kv_for_commit(layer_cache, n_tokens):
    """Extract (K, V) for this layer's first n_tokens filled positions, in
    the shape expected by commit_chunk_as_standalone: [num_kv_heads, n_tokens, head_dim].
    """
    k = layer_cache.keys
    v = layer_cache.values
    if k.ndim == 4:
        k = k[0]   # drop batch
        v = v[0]
    offset = int(getattr(layer_cache, "offset", k.shape[1]))
    n = min(n_tokens, offset)
    return k[:, :n, :], v[:, :n, :]


def _prewarm_chunks(model, tokenizer, prefix_cache, docs):
    """Run each doc chunk as a standalone prefill and commit its K/V as
    a root-parented prefix, so subsequent lookups find it.
    """
    from mlx_lm.models.cache import make_prompt_cache
    committed_counts = []
    for doc in docs:
        tokens = tokenizer.encode(doc)
        cache = make_prompt_cache(model)
        _ = model(mx.array([tokens]), cache=cache)
        mx.eval(_)
        per_layer_kv = [_slice_kv_for_commit(lc, len(tokens)) for lc in cache]
        ok = prefix_cache.commit_chunk_as_standalone(tokens, per_layer_kv)
        committed_counts.append(ok)
    assert all(committed_counts), "pre-warm failed — check commit_chunk_as_standalone returned True for every chunk"


def _make_prefix_cache(tmp_path: Path, model):
    from unittest.mock import MagicMock
    from omlx.cache.factory import CacheFactory
    from omlx.cache.factory import CacheConfig

    config = CacheConfig(
        paged_ssd_cache_dir=tmp_path / "cache",
        model_name="correctness-test",
    )
    # The cache only uses the model for layer-count validation; a MagicMock
    # whose .layers is a list of the right length works.
    inner = getattr(model, "model", model)
    mock_model = MagicMock()
    mock_model.layers = [MagicMock() for _ in range(len(inner.layers))]
    stack = CacheFactory.create_full_cache_stack(config, model=mock_model)
    return stack["prefix_cache"]


def _greedy_generate_logits(model, prompt_token_ids, cache, max_new_tokens: int):
    """Run a greedy rollout and return the list of first-N chosen tokens."""
    tokens = mx.array([prompt_token_ids])
    logits = model(tokens, cache=cache)
    out = []
    for _ in range(max_new_tokens):
        next_tok = int(mx.argmax(logits[0, -1, :]).item())
        out.append(next_tok)
        logits = model(mx.array([[next_tok]]), cache=cache)
        mx.eval(logits)
    return out


def _logits_distribution(model, prompt_token_ids, cache, max_steps: int):
    """Return a list of max_steps softmax distributions, advancing the
    sequence greedily (so we compare at the *same* generated token stream).
    """
    tokens = mx.array([prompt_token_ids])
    logits = model(tokens, cache=cache)
    dists = [mx.softmax(logits[0, -1, :], axis=-1)]
    for _ in range(max_steps - 1):
        tok = int(mx.argmax(logits[0, -1, :]).item())
        logits = model(mx.array([[tok]]), cache=cache)
        dists.append(mx.softmax(logits[0, -1, :], axis=-1))
    return dists


def _run_with_blend(model, tokenizer, prompt, prefix_cache, docs, query):
    from mlx_lm.models.cache import make_prompt_cache
    cacheblend.patch_model_for_cacheblend(model)
    cache = make_prompt_cache(model)
    tokenized_chunks = [tokenizer.encode(d) for d in docs]
    query_tokens = tokenizer.encode(query)
    meta = cacheblend.build_blend_metadata(
        chunk_token_lists=tokenized_chunks,
        query_tokens=query_tokens,
        prefix_cache=prefix_cache,
        chunk_min_tokens=1,
    )
    assert meta is not None, "all chunks should be CACHED after prewarm"
    meta.recompute_ratio = 0.15
    for lc in cache:
        lc.blend_metadata = meta
    return cache, tokenizer.encode(prompt)


def _run_baseline(model, tokenizer, prompt):
    from mlx_lm.models.cache import make_prompt_cache
    return make_prompt_cache(model), tokenizer.encode(prompt)


@pytest.mark.parametrize("model_id", [MODEL_ID_LLAMA, MODEL_ID_QWEN3])
def test_first_20_tokens_greedy_identical(model_id, tmp_path):
    pytest.importorskip("mlx_lm")
    from mlx_lm import load

    model, tokenizer = load(model_id)
    fixture = _load_fixture()
    prompt = _build_prompt(fixture)

    # Baseline generation (fresh model instance? No — patched model is still
    # byte-identical when no blend_metadata is attached, so reuse is fine).
    cache_base, tok_base = _run_baseline(model, tokenizer, prompt)
    baseline_toks = _greedy_generate_logits(model, tok_base, cache_base, max_new_tokens=20)

    prefix_cache = _make_prefix_cache(tmp_path, model)
    _prewarm_chunks(model, tokenizer, prefix_cache, fixture["docs"])

    cache_blend, tok_blend = _run_with_blend(
        model, tokenizer, prompt, prefix_cache, fixture["docs"], fixture["query"]
    )
    blend_toks = _greedy_generate_logits(model, tok_blend, cache_blend, max_new_tokens=20)

    assert baseline_toks == blend_toks, (
        f"MVP full-recompute blend must produce identical tokens; "
        f"baseline={baseline_toks}, blend={blend_toks}"
    )


@pytest.mark.parametrize("model_id", [MODEL_ID_LLAMA, MODEL_ID_QWEN3])
def test_first_20_tokens_kl_bounded(model_id, tmp_path):
    """KL(blend || baseline) ≤ 0.05 nats at each of the first 20 positions."""
    pytest.importorskip("mlx_lm")
    from mlx_lm import load

    model, tokenizer = load(model_id)
    fixture = _load_fixture()
    prompt = _build_prompt(fixture)

    cache_base, tok_base = _run_baseline(model, tokenizer, prompt)
    p_dists = _logits_distribution(model, tok_base, cache_base, max_steps=20)

    prefix_cache = _make_prefix_cache(tmp_path, model)
    _prewarm_chunks(model, tokenizer, prefix_cache, fixture["docs"])

    cache_blend, tok_blend = _run_with_blend(
        model, tokenizer, prompt, prefix_cache, fixture["docs"], fixture["query"]
    )
    q_dists = _logits_distribution(model, tok_blend, cache_blend, max_steps=20)

    eps = 1e-12
    for i, (p, q) in enumerate(zip(p_dists, q_dists)):
        # Cast to float32 before computing log to avoid 0 * log(0) = NaN in
        # float16 distributions (softmax of a 4-bit quantised model can have
        # exact zeros in low-probability bins).  Where p == 0 the KL term is
        # 0 by definition (lim_{p→0} p log p = 0), so we zero those entries.
        p32 = p.astype(mx.float32)
        q32 = q.astype(mx.float32)
        term = mx.where(p32 > 0, p32 * (mx.log(p32 + eps) - mx.log(q32 + eps)), mx.zeros_like(p32))
        kl = float(mx.sum(term).item())
        assert kl <= 0.05, (
            f"KL at position {i} = {kl:.6f} exceeds 0.05 nats "
            f"(model={model_id})"
        )
