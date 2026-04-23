# SPDX-License-Identifier: Apache-2.0
"""CacheBlend TTFT micro-benchmark.

Runs 4 pre-warmed doc chunks (512 tokens each) + 64-token query prompt under:
  (a) full recompute baseline
  (b) CacheBlend (MVP: full recompute with metadata plumbing)

Reports median + IQR of TTFT over N runs, records the raw numbers to JSON,
and prints the speedup. Ship-gate of 1.5x is INFORMATIONAL in the MVP
(which is full-recompute) — gate failure does not abort.
"""

from __future__ import annotations

import json
import os
import platform
import random
import re
import statistics
import string
import time
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx

from omlx.patches import cacheblend


# Model ID can be overridden via env var, e.g.
#   OMLX_BENCH_MODEL_ID=mlx-community/Meta-Llama-3.1-8B-Instruct-bf16 python ...
# The default is the 1B-4bit checkpoint Task 0 pre-downloaded — cheap to run
# and a known calibration point. Larger unquantized models move out of
# memory-bandwidth-bound territory and show CacheBlend's real speedup.
MODEL_ID = os.environ.get(
    "OMLX_BENCH_MODEL_ID", "mlx-community/Llama-3.2-1B-Instruct-4bit"
)
# RAG-representative context: 4 chunks @ ~2K tokens each + 64-token query.
# At ~8K total tokens, attention's O(L^2) starts dominating Q/K/V projection's
# O(L*D^2), which is where the sparse-attention win is visible. At the
# original 2K-nominal config attention is too small relative to projections
# for CacheBlend's savings to show up against baseline noise.
NUM_CHUNKS = 4
TOKENS_PER_CHUNK = 2048
QUERY_TOKENS = 64
RUNS = 20
WARMUP = 3
RECOMPUTE_RATIO = 0.15


def _synthetic_doc(seed: int, approx_tokens: int) -> str:
    """Stable pseudorandom prose. Use 4 chars/token as a rough approximation."""
    r = random.Random(seed)
    words = []
    for _ in range(approx_tokens * 4):
        words.append(r.choice(string.ascii_lowercase + " "))
    return "".join(words)


def _synthetic_query(approx_tokens: int) -> str:
    return "Summarize the above documents in one paragraph. " * max(1, approx_tokens // 8)


def _make_prefix_cache(tmp_dir: Path, model):
    from omlx.cache.factory import CacheFactory, CacheConfig

    config = CacheConfig(
        paged_ssd_cache_dir=tmp_dir,
        model_name="bench-cacheblend",
    )
    inner = getattr(model, "model", model)
    mock_model = MagicMock()
    mock_model.layers = [MagicMock() for _ in range(len(inner.layers))]
    stack = CacheFactory.create_full_cache_stack(config, model=mock_model)
    return stack["prefix_cache"]


def _slice_kv(layer_cache, n):
    k = layer_cache.keys
    v = layer_cache.values
    if k.ndim == 4:
        k = k[0]
        v = v[0]
    off = int(getattr(layer_cache, "offset", k.shape[1]))
    nn = min(n, off)
    return k[:, :nn, :], v[:, :nn, :]


def _prewarm(model, prefix_cache, doc_token_lists):
    from mlx_lm.models.cache import make_prompt_cache
    for toks in doc_token_lists:
        cache = make_prompt_cache(model)
        logits = model(mx.array([toks]), cache=cache)
        mx.eval(logits)
        per_layer_kv = [_slice_kv(lc, len(toks)) for lc in cache]
        ok = prefix_cache.commit_chunk_as_standalone(toks, per_layer_kv)
        assert ok, "pre-warm commit failed"


def _measure_ttft_baseline(model, prompt_token_ids):
    """Time a single prefill-to-first-token on a fresh cache, no blend."""
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(mx.array([prompt_token_ids]), cache=cache)
    # Force Metal to materialize logits before timing stops.
    _ = int(mx.argmax(logits[0, -1, :]).item())
    return time.perf_counter() - t0


def _measure_ttft_blend(model, prompt_token_ids, doc_token_lists, query_tokens, prefix_cache):
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    meta = cacheblend.build_blend_metadata(
        chunk_token_lists=doc_token_lists,
        query_tokens=query_tokens,
        prefix_cache=prefix_cache,
        chunk_min_tokens=1,
    )
    if meta is None:
        raise RuntimeError("meta should have been populated from prewarmed chunks")
    meta.recompute_ratio = RECOMPUTE_RATIO
    for lc in cache:
        lc.blend_metadata = meta
    t0 = time.perf_counter()
    logits = model(mx.array([prompt_token_ids]), cache=cache)
    _ = int(mx.argmax(logits[0, -1, :]).item())
    return time.perf_counter() - t0


def main():
    from mlx_lm import load

    print(f"Loading {MODEL_ID}...")
    model, tokenizer = load(MODEL_ID)

    print(f"Generating {NUM_CHUNKS} synthetic chunks of ~{TOKENS_PER_CHUNK} tokens + {QUERY_TOKENS}-token query...")
    doc_texts = [_synthetic_doc(i, TOKENS_PER_CHUNK) for i in range(NUM_CHUNKS)]
    query_text = _synthetic_query(QUERY_TOKENS)

    # Build the prompt at TOKEN level rather than string level. Plain
    # string-split + per-chunk encode drifts by a few tokens (tokenizer
    # prepends BOS per call, boundary merges differ in context), which
    # misaligns the blend path's HKVD scorer and silently disables sparse.
    # Here we concatenate pre-encoded token lists directly.
    doc_token_lists = [tokenizer.encode(d) for d in doc_texts]
    query_tokens_raw = tokenizer.encode(query_text)
    # Drop BOS on non-first chunks and on the query so only one BOS lands
    # at the very start of the assembled prompt.
    first_tok = doc_token_lists[0][:1] if doc_token_lists and doc_token_lists[0] else []
    bos_candidate = first_tok[0] if first_tok else None

    def _strip_leading_bos(tokens):
        if bos_candidate is not None and tokens and tokens[0] == bos_candidate:
            return tokens[1:]
        return tokens

    # Keep chunk[0]'s BOS, strip it from everything after.
    doc_token_lists = [doc_token_lists[0]] + [_strip_leading_bos(t) for t in doc_token_lists[1:]]
    query_tokens = _strip_leading_bos(query_tokens_raw)

    doc_token_counts = [len(t) for t in doc_token_lists]
    query_token_count = len(query_tokens)
    print(f"  doc tokens: {doc_token_counts}, query tokens: {query_token_count}")

    prompt_token_ids = []
    for t in doc_token_lists:
        prompt_token_ids.extend(t)
    prompt_token_ids.extend(query_tokens)
    print(f"  total prompt tokens: {len(prompt_token_ids)}")

    # Pre-warm the prefix cache with each doc chunk as standalone.
    tmp_dir = Path("/tmp/cacheblend_bench_cache")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    prefix_cache = _make_prefix_cache(tmp_dir, model)
    print("Pre-warming prefix cache...")
    _prewarm(model, prefix_cache, doc_token_lists)

    # Ensure patch is installed so no-metadata path is byte-identical (Task 9 guarantee)
    cacheblend.patch_model_for_cacheblend(model)

    print(f"\nWarmup ({WARMUP} runs each)...")
    for _ in range(WARMUP):
        _measure_ttft_baseline(model, prompt_token_ids)
        _measure_ttft_blend(model, prompt_token_ids, doc_token_lists, query_tokens, prefix_cache)

    print(f"\nMeasuring baseline TTFT ({RUNS} runs)...")
    baseline = [_measure_ttft_baseline(model, prompt_token_ids) for _ in range(RUNS)]

    print(f"Measuring blend TTFT ({RUNS} runs)...")
    blend = [_measure_ttft_blend(model, prompt_token_ids, doc_token_lists, query_tokens, prefix_cache) for _ in range(RUNS)]

    q_base = statistics.quantiles(baseline, n=4)
    q_blend = statistics.quantiles(blend, n=4)
    median_base = statistics.median(baseline)
    median_blend = statistics.median(blend)
    speedup = median_base / median_blend

    result = {
        "host": {"machine": platform.machine(), "platform": platform.platform()},
        "model_id": MODEL_ID,
        "config": {
            "chunks": NUM_CHUNKS,
            "nominal_tokens_per_chunk": TOKENS_PER_CHUNK,
            "actual_chunk_tokens": doc_token_counts,
            "query_tokens": query_token_count,
            "total_prompt_tokens": len(prompt_token_ids),
            "runs": RUNS,
            "recompute_ratio": RECOMPUTE_RATIO,
        },
        "baseline_ttft_s": {
            "median": median_base,
            "iqr": q_base[2] - q_base[0],
            "min": min(baseline),
            "max": max(baseline),
        },
        "blend_ttft_s": {
            "median": median_blend,
            "iqr": q_blend[2] - q_blend[0],
            "min": min(blend),
            "max": max(blend),
        },
        "speedup_x": speedup,
        "ship_gate_threshold": 1.5,
        "ship_gate_pass": speedup >= 1.5,
        "mvp_note": (
            "Current patch is full-recompute (layerwise plumbing only); "
            "sparse-attention optimization is a follow-up. Speedup near 1.0x "
            "is expected; gate failure informs that follow-up work."
        ),
    }

    # Per-model result files so runs on different checkpoints don't clobber
    # each other. Sanitize the model ID into a safe filename fragment.
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", MODEL_ID)
    out_path = Path(__file__).parent / f"bench_cacheblend_results_{safe_id}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)
    print(f"\nspeedup: {speedup:.3f}x   (ship-gate: 1.5x -> {'PASS' if result['ship_gate_pass'] else 'FAIL (expected in MVP)'})")
    print(f"result written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
