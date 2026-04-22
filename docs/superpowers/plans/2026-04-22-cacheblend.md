# CacheBlend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CacheBlend (selective KV recomputation for RAG chunk reuse) to omlx for `llama` and `qwen3`, with a correctness + speedup micro-benchmark MVP.

**Architecture:** Runtime monkey-patch mlx-lm model forwards in `omlx/patches/cacheblend.py`, mirroring the existing SpecPrefill pattern. Extend `BlockAwarePrefixCache` with standalone-hash chunk lookup/commit. Split prompts on a separator at scheduler entry; when ≥1 chunk is cached, run a layerwise forward that computes fresh K/V on layer 0, scores HKVD on layer 1, and sparsely recomputes only top-r% tokens (union COLD indices) on layers 2..N. Graceful fallback to standard prefill for any edge case.

**Tech Stack:** Python 3.11+, mlx-core, mlx-lm, pytest, existing omlx paged/prefix cache machinery.

**Spec reference:** `docs/superpowers/specs/2026-04-22-cacheblend-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `omlx/patches/cacheblend.py` | CREATE | All blend logic: config, monkey-patches, HKVD, sparse-Q attention, offset-RoPE, split utility, fallback counter |
| `omlx/cache/prefix_cache.py` | MODIFY | Add `lookup_chunk_by_standalone_hash`, `commit_chunk_as_standalone` methods on `BlockAwarePrefixCache` |
| `omlx/model_settings.py` | MODIFY | Add `cacheblend_*` fields to `ModelSettings` dataclass |
| `omlx/engine/scheduler.py` OR `omlx/engine_core.py` | MODIFY | Wire the pre-prefill `split_on_separator_and_blend` hook; assert mutual exclusion with SpecPrefill |
| `omlx/api/openai_models.py` | MODIFY | (Optional) expose per-request `cacheblend_special_str` override |
| `tests/test_cacheblend_unit.py` | CREATE | Unit tests: HKVD, offset-RoPE, sparse-Q, chunk split, cache round-trip |
| `tests/test_cacheblend_correctness.py` | CREATE | End-to-end correctness vs. full-recompute baseline on Llama + Qwen3 |
| `tests/bench/bench_cacheblend.py` | CREATE | TTFT micro-benchmark with pre-warmed chunks |
| `tests/fixtures/cacheblend/` | CREATE | Hand-crafted RAG prompt fixtures |

**Why `cacheblend.py` is one file and not a package:** it's ~700 LOC of tightly coupled math + patching glue, mirrors `specprefill.py` which chose the same single-file layout, and splitting it prematurely would spread tight invariants (layer index bookkeeping, mutation of the blend metadata object) across files without clear boundaries.

---

## Task 0: Environment sanity & model downloads

**Purpose:** Before writing code, verify tests can run and download the small checkpoints the correctness + benchmark tasks need.

**Files:**
- Read only: `pyproject.toml`, `tests/conftest.py`
- Will download into the default HuggingFace cache (`~/.cache/huggingface/hub/`)

- [ ] **Step 1: Verify pytest runs**

Run: `pytest tests/test_cache_factory.py -v --no-header -q 2>&1 | tail -20`
Expected: green or yellow (skipped) — confirms the test environment is set up.
If this fails: run `pip install -e ".[dev,test]"` from the repo root, then retry.

- [ ] **Step 2: Verify mlx-lm is importable**

Run: `python -c "import mlx_lm; from mlx_lm import load; print(mlx_lm.__version__)"`
Expected: a version string prints. If ImportError: `pip install -U mlx-lm` and retry.

- [ ] **Step 3: Download the Llama test checkpoint**

Run: `python -c "from mlx_lm import load; m, t = load('mlx-community/Llama-3.2-1B-Instruct-4bit'); print('OK layers:', len(m.model.layers))"`
Expected: `OK layers: 16` (or similar small count). This downloads ~700MB on first run. The model ID is pinned here so subsequent tasks can reference it.

- [ ] **Step 4: Download the Qwen3 test checkpoint**

Run: `python -c "from mlx_lm import load; m, t = load('mlx-community/Qwen3-1.7B-4bit'); print('OK layers:', len(m.model.layers))"`
Expected: `OK layers: 28` (or similar). If the exact ID is unavailable on the registry at execution time, substitute with any `mlx-community/Qwen3-*` checkpoint ≤ 4B parameters and record the exact ID in the test file's `MODEL_ID_QWEN3` constant.

- [ ] **Step 5: Commit a note recording the pinned model IDs**

Create `tests/fixtures/cacheblend/README.md`:

```markdown
# CacheBlend test fixtures

Pinned model IDs used by correctness and benchmark tests:

- Llama: `mlx-community/Llama-3.2-1B-Instruct-4bit`
- Qwen3: `mlx-community/Qwen3-1.7B-4bit`

If either becomes unavailable, update the `MODEL_ID_*` constants in
`tests/test_cacheblend_correctness.py` and `tests/bench/bench_cacheblend.py`.
```

```bash
git add tests/fixtures/cacheblend/README.md
git commit -m "test(cacheblend): pin model IDs for correctness + benchmark tests"
```

---

## Task 1: ModelSettings config fields

**Files:**
- Modify: `omlx/model_settings.py`

- [ ] **Step 1: Read the existing SpecPrefill fields to match the pattern**

Run: `grep -n 'specprefill' omlx/model_settings.py`
Note the exact location and style (dataclass fields with defaults, typing).

- [ ] **Step 2: Add cacheblend fields after the specprefill block**

Insert after the last `specprefill_*` line in the `ModelSettings` dataclass (keep alphabetical/grouped ordering consistent with the file):

```python
    cacheblend_enabled: bool = False
    cacheblend_recompute_ratio: float = 0.15
    cacheblend_check_layers: list[int] = field(default_factory=lambda: [1])
    cacheblend_special_str: str = " # # "
    cacheblend_chunk_min_tokens: int = 32
```

If `field` and `list[int]` mutability require an import, ensure `from dataclasses import dataclass, field` is already at the top (it should be).

- [ ] **Step 3: Verify the dataclass still loads**

Run: `python -c "from omlx.model_settings import ModelSettings; s = ModelSettings(); print(s.cacheblend_enabled, s.cacheblend_recompute_ratio, s.cacheblend_check_layers, repr(s.cacheblend_special_str), s.cacheblend_chunk_min_tokens)"`
Expected: `False 0.15 [1] ' # # ' 32`

- [ ] **Step 4: Commit**

```bash
git add omlx/model_settings.py
git commit -m "feat(cacheblend): add ModelSettings config fields"
```

---

## Task 2: Module scaffold + fallback counter

**Files:**
- Create: `omlx/patches/cacheblend.py`

- [ ] **Step 1: Create the scaffold**

```python
# SPDX-License-Identifier: Apache-2.0
"""CacheBlend: selective KV recomputation for cached RAG chunks.

Enables reuse of per-chunk KV from prior requests by concatenating them
with correct per-chunk positional offsets and recomputing K/V only for
the small fraction of tokens with highest cross-attention deviation
(HKVD, EuroSys '25).

Pipeline:
  1. split_on_separator_and_blend() — tokenize, split on special str,
     classify chunks as CACHED or COLD, attach BlendMetadata to cache.
  2. Patched model forward:
       layer 0  — full compute on all tokens
       layer 1  — HKVD scoring: diff_k = ||K_fresh - K_cached||^2,
                  top-r% ∪ COLD indices → recompute_indices
       layers 2..N — sparse Q attention: selected Q vs. full K/V,
                  scatter fresh K/V back into cache at recompute_indices
  3. On finish: commit any freshly-computed chunks as standalone for reuse.

Falls back to standard prefill for any edge case; never crashes a request.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Fallback counter (replace with your metrics backend in production)
# -----------------------------------------------------------------------------

_FALLBACK_COUNTER: Counter[str] = Counter()


def record_fallback(reason: str) -> None:
    """Increment the fallback counter. Reasons used:
    no_separator, no_chunks_cached, tokenizer_split_misalign,
    shape_mismatch, nan_hkvd, specprefill_conflict.
    """
    _FALLBACK_COUNTER[reason] += 1
    logger.info("cacheblend fallback: %s", reason)


def get_fallback_counts() -> dict[str, int]:
    return dict(_FALLBACK_COUNTER)


def reset_fallback_counts() -> None:
    _FALLBACK_COUNTER.clear()


# -----------------------------------------------------------------------------
# Chunk classification
# -----------------------------------------------------------------------------

CACHED = "cached"
COLD = "cold"


@dataclass
class ChunkInfo:
    tokens: List[int]
    kind: str              # CACHED or COLD
    start_pos: int         # absolute position in the blended sequence
    cached_kv: Optional[Any] = None   # per-layer K/V handle if CACHED, else None


@dataclass
class BlendMetadata:
    """Attached to the cache object for a single blend request."""
    chunks: List[ChunkInfo]
    total_len: int
    recompute_indices: Optional[mx.array] = None   # filled after check layer
    cold_token_mask: Optional[mx.array] = None     # [total_len] bool

    @property
    def chunk_boundaries(self) -> List[Tuple[int, int]]:
        return [(c.start_pos, c.start_pos + len(c.tokens)) for c in self.chunks]
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from omlx.patches import cacheblend; print(cacheblend.CACHED, cacheblend.COLD); cacheblend.record_fallback('test'); print(cacheblend.get_fallback_counts())"`
Expected: `cached cold` then `{'test': 1}`.

- [ ] **Step 3: Commit**

```bash
git add omlx/patches/cacheblend.py
git commit -m "feat(cacheblend): module scaffold with fallback counter + metadata"
```

---

## Task 3: HKVD scorer

**Files:**
- Modify: `omlx/patches/cacheblend.py`
- Create: `tests/test_cacheblend_unit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cacheblend_unit.py`:

```python
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
```

- [ ] **Step 2: Run the test, expect failure**

Run: `pytest tests/test_cacheblend_unit.py::test_hkvd_score_selects_largest_diff_tokens -v`
Expected: FAIL with `AttributeError: module 'omlx.patches.cacheblend' has no attribute 'hkvd_score'`.

- [ ] **Step 3: Implement `hkvd_score`**

Append to `omlx/patches/cacheblend.py`:

```python
# -----------------------------------------------------------------------------
# HKVD scorer
# -----------------------------------------------------------------------------


def hkvd_score(
    k_fresh: mx.array,
    k_cached: mx.array,
    cold_token_mask: mx.array,
    recompute_ratio: float,
) -> mx.array:
    """Rank tokens by K-deviation and return indices to recompute.

    Args:
        k_fresh: [num_heads, total_len, head_dim] — freshly computed K at
            the check layer, with correct absolute-position RoPE applied.
        k_cached: [num_heads, total_len, head_dim] — K pulled from the
            blended chunk caches, position-fix-up already applied for
            CACHED tokens. For COLD tokens the value is ignored (mask
            forces selection).
        cold_token_mask: [total_len] bool — True for tokens in COLD chunks.
        recompute_ratio: float in (0, 1].

    Returns:
        mx.array of int indices, sorted ascending, to recompute on
        subsequent layers. Always a superset of the COLD tokens.
    """
    total_len = k_fresh.shape[1]
    # Per-token squared L2 diff averaged across heads and feature dims.
    diff = (k_fresh.astype(mx.float32) - k_cached.astype(mx.float32)) ** 2
    score = mx.mean(diff, axis=(0, 2))  # [total_len]
    score = mx.where(cold_token_mask, mx.array(float("inf"), dtype=mx.float32), score)

    k = max(1, int(total_len * recompute_ratio))
    # Ensure k covers all COLD tokens even if the ratio is small.
    num_cold = int(mx.sum(cold_token_mask.astype(mx.int32)).item())
    k = max(k, num_cold)
    k = min(k, total_len)

    top = mx.argsort(-score)[:k]
    return mx.sort(top)
```

- [ ] **Step 4: Run both tests, expect pass**

Run: `pytest tests/test_cacheblend_unit.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add omlx/patches/cacheblend.py tests/test_cacheblend_unit.py
git commit -m "feat(cacheblend): HKVD scorer with COLD-token union"
```

---

## Task 4: Per-chunk offset-RoPE

**Files:**
- Modify: `omlx/patches/cacheblend.py`
- Modify: `tests/test_cacheblend_unit.py`

- [ ] **Step 1: Study SpecPrefill's offset RoPE**

Run: `grep -n '_OffsetAdjustedRoPE\|apply_rope\|rotary' omlx/patches/specprefill.py | head -20`
Read the referenced class (around lines noted). Note its construction: it wraps the model's existing rotary, adds a scalar offset, and delegates. We'll generalize to a per-token offset.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_cacheblend_unit.py`:

```python
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
```

- [ ] **Step 3: Run the test, expect failure**

Run: `pytest tests/test_cacheblend_unit.py::test_rotate_k_by_offsets_matches_direct_computation -v`
Expected: FAIL — `rotate_k_by_offsets` doesn't exist yet.

- [ ] **Step 4: Implement `rotate_k_by_offsets`**

Append to `omlx/patches/cacheblend.py`:

```python
# -----------------------------------------------------------------------------
# Per-chunk offset-RoPE
# -----------------------------------------------------------------------------


def rotate_k_by_offsets(
    k: mx.array,
    offsets: mx.array,
    head_dim: int,
    base: float = 10000.0,
) -> mx.array:
    """Rotate each token's K by its per-token position offset.

    Cached K was rotated at position 0. When that chunk lands at absolute
    position `offset` in a new blended sequence, its K needs to be rotated
    by an *additional* `offset` angle (RoPE composition: rotating by a,
    then by b, equals rotating by a+b).

    Args:
        k: [num_heads, total_len, head_dim]
        offsets: [total_len] int32 — per-token additional offset to apply.
            For tokens that should NOT be rotated (e.g. COLD tokens whose
            K will be freshly computed anyway), pass 0.
        head_dim: rotary dimension.
        base: RoPE base (typically 10000 for Llama/Qwen3; override per-arch).

    Returns:
        [num_heads, total_len, head_dim] — rotated K.
    """
    d = head_dim
    if d % 2 != 0:
        raise ValueError(f"rotate_k_by_offsets requires even head_dim, got {d}")

    inv_freq = base ** (-mx.arange(0, d, 2).astype(mx.float32) / d)    # [d/2]
    freqs = offsets.astype(mx.float32)[:, None] * inv_freq[None, :]    # [T, d/2]
    cos = freqs.cos()
    sin = freqs.sin()

    k_even = k[..., 0::2]      # [H, T, d/2]
    k_odd = k[..., 1::2]
    rot_even = k_even * cos[None, :, :] - k_odd * sin[None, :, :]
    rot_odd = k_even * sin[None, :, :] + k_odd * cos[None, :, :]
    out = mx.stack([rot_even, rot_odd], axis=-1)
    return out.reshape(k.shape)
```

Note: this is the half-and-half layout variant (pairs of adjacent dims). mlx-lm's default rotary uses the same convention; if a model uses the "rotate half" (first-half/second-half split) variant, this helper needs a conditional branch keyed off the model's rotary style. For the two MVP targets (Llama, Qwen3) both use the adjacent-pair convention.

- [ ] **Step 5: Run the test, expect pass**

Run: `pytest tests/test_cacheblend_unit.py::test_rotate_k_by_offsets_matches_direct_computation -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add omlx/patches/cacheblend.py tests/test_cacheblend_unit.py
git commit -m "feat(cacheblend): per-token offset-RoPE rotation"
```

---

## Task 5: Sparse-Q attention primitive

**Files:**
- Modify: `omlx/patches/cacheblend.py`
- Modify: `tests/test_cacheblend_unit.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cacheblend_unit.py`:

```python
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
```

- [ ] **Step 2: Run the test, expect failure**

Run: `pytest tests/test_cacheblend_unit.py::test_sparse_q_selected_against_full_kv_matches_dense_subset -v`
Expected: FAIL — `sparse_attn_selected_q` doesn't exist.

- [ ] **Step 3: Implement the primitive**

Append to `omlx/patches/cacheblend.py`:

```python
# -----------------------------------------------------------------------------
# Sparse-Q attention: selected Q attends against full K/V with causal mask
# -----------------------------------------------------------------------------


def sparse_attn_selected_q(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    selected_q_indices: mx.array,
    scale: float,
) -> mx.array:
    """Compute attention for a selected subset of Q tokens against full K/V.

    CacheBlend's core primitive: for layers 2..N, only `selected_q_indices`
    tokens get refreshed output, but each of them must still attend against
    the full (cached + already-refreshed) K/V buffer.

    Args:
        q: [num_heads, total_len, head_dim]  — full Q (we'll slice).
        k: [num_heads, total_len, head_dim]  — full blended K.
        v: [num_heads, total_len, head_dim]  — full blended V.
        selected_q_indices: [n_sel] int32, sorted ascending.
        scale: attention scale (typically 1/sqrt(head_dim)).

    Returns:
        [num_heads, n_sel, head_dim] — attention output for the selected Q.
    """
    total_len = q.shape[1]
    q_sel = q[:, selected_q_indices, :]                           # [H, S, D]
    scores = (q_sel @ k.transpose(0, 2, 1)) * scale               # [H, S, T]

    # Causal mask: selected row i at absolute position selected_q_indices[i]
    # may attend to positions 0..selected_q_indices[i].
    positions = mx.arange(total_len).astype(mx.int32)             # [T]
    sel_pos = selected_q_indices.astype(mx.int32)[:, None]        # [S, 1]
    allow = positions[None, :] <= sel_pos                         # [S, T]
    mask = mx.where(allow, mx.array(0.0), mx.array(-1e9))         # [S, T]
    scores = scores + mask[None, :, :]

    attn = mx.softmax(scores, axis=-1)
    out = attn @ v                                                # [H, S, D]
    return out
```

- [ ] **Step 4: Run the test, expect pass**

Run: `pytest tests/test_cacheblend_unit.py::test_sparse_q_selected_against_full_kv_matches_dense_subset -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add omlx/patches/cacheblend.py tests/test_cacheblend_unit.py
git commit -m "feat(cacheblend): sparse-Q vs full-K/V attention primitive"
```

---

## Task 6: Cache — standalone chunk lookup

**Files:**
- Modify: `omlx/cache/prefix_cache.py`
- Create: `tests/test_cacheblend_cache.py`

- [ ] **Step 1: Locate insertion point and understand existing lookup**

Run: `grep -n 'def.*lookup\|def.*find\|def get_cache_for_generation\|cached_block_hash_to_block' omlx/cache/prefix_cache.py | head -20`
Read `BlockAwarePrefixCache.get_cache_for_generation` (around line 1130 per prior exploration) to understand the return shape for per-layer K/V tensors. We'll match that shape in the new lookup.

- [ ] **Step 2: Write the failing test**

Create `tests/test_cacheblend_cache.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for CacheBlend additions to BlockAwarePrefixCache."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.cache.prefix_cache import BlockAwarePrefixCache


@pytest.fixture
def cache():
    # Minimal config; adapt argument names to BlockAwarePrefixCache.__init__
    # signature (read the existing constructor call in tests/test_cache_factory.py
    # or tests/test_cache_stats.py for a working invocation).
    # The test must be adapted at implementation time to match the real signature.
    pytest.importorskip("mlx_lm")
    from tests.fixtures.cacheblend.cache_factory import make_minimal_cache
    return make_minimal_cache()


def test_commit_and_lookup_chunk_roundtrip(cache):
    tokens = list(range(300))   # two full blocks + one partial block at size 256
    # Fabricate per-layer K/V: 2 layers, 1 head, head_dim=8
    num_layers, num_heads, head_dim = 2, 1, 8
    per_layer_kv = [
        (
            mx.array(np.random.default_rng(i).normal(size=(num_heads, len(tokens), head_dim)).astype(np.float32)),
            mx.array(np.random.default_rng(i + 100).normal(size=(num_heads, len(tokens), head_dim)).astype(np.float32)),
        )
        for i in range(num_layers)
    ]

    cache.commit_chunk_as_standalone(tokens, per_layer_kv)
    handle = cache.lookup_chunk_by_standalone_hash(tokens)

    assert handle is not None
    for layer_idx in range(num_layers):
        k_stored, v_stored = handle.per_layer_kv[layer_idx]
        k_orig, v_orig = per_layer_kv[layer_idx]
        assert mx.allclose(k_stored, k_orig).item()
        assert mx.allclose(v_stored, v_orig).item()


def test_lookup_returns_none_on_miss(cache):
    handle = cache.lookup_chunk_by_standalone_hash([999, 998, 997, 996])
    assert handle is None
```

Create `tests/fixtures/cacheblend/__init__.py` (empty) and `tests/fixtures/cacheblend/cache_factory.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Test helper: build a minimal BlockAwarePrefixCache.

Inspect omlx/cache/prefix_cache.py's BlockAwarePrefixCache __init__ and
tests/test_cache_factory.py for a reference construction. Keep this helper
thin — just enough to instantiate a cache we can commit_chunk / lookup
against in unit tests.
"""
from omlx.cache.prefix_cache import BlockAwarePrefixCache


def make_minimal_cache():
    # TODO at implementation time: copy the constructor call from
    # tests/test_cache_factory.py::test_<something>, stripping anything
    # that requires a loaded model. block_size=256 must match the default
    # the chunk hash path assumes.
    raise NotImplementedError(
        "Adapt this factory to match BlockAwarePrefixCache's real constructor."
    )
```

The `raise NotImplementedError` is a deliberate bright flag — the implementer must fill this in by reading `tests/test_cache_factory.py` at implementation time (the constructor signature isn't stable across refactors; hard-coding it here would rot).

- [ ] **Step 3: Fill in `make_minimal_cache`**

Open `tests/test_cache_factory.py` and copy the lightest-weight cache construction pattern (a test that creates a cache without a model), adapting into `make_minimal_cache`. Run the test to confirm the fixture itself works:

Run: `python -c "from tests.fixtures.cacheblend.cache_factory import make_minimal_cache; c = make_minimal_cache(); print(type(c).__name__)"`
Expected: `BlockAwarePrefixCache`.

- [ ] **Step 4: Run the new tests, expect failure**

Run: `pytest tests/test_cacheblend_cache.py -v`
Expected: FAIL — `commit_chunk_as_standalone` / `lookup_chunk_by_standalone_hash` don't exist yet.

- [ ] **Step 5: Implement the two methods on `BlockAwarePrefixCache`**

Add to `omlx/cache/prefix_cache.py` inside the `BlockAwarePrefixCache` class. Place these methods after `get_cache_for_generation` (keep the public API grouped):

```python
    # -------------------------------------------------------------------
    # CacheBlend: standalone-hash chunk lookup
    # -------------------------------------------------------------------

    def _compute_standalone_block_hashes(self, tokens: List[int]) -> List[bytes]:
        """Return the block-hash chain for `tokens` as-if standalone (parent=b'')."""
        from omlx.cache.paged_cache import compute_block_hash
        block_size = self.paged_cache.block_size
        hashes: List[bytes] = []
        parent = b""
        for start in range(0, len(tokens), block_size):
            block_tokens = tokens[start:start + block_size]
            h = compute_block_hash(parent, block_tokens)
            hashes.append(h)
            parent = h
        return hashes

    def lookup_chunk_by_standalone_hash(self, tokens: List[int]):
        """Look up a chunk's per-layer K/V as if it were a root-parented prefix.

        Returns a ChunkKVHandle with .per_layer_kv = list of (K, V) mx.arrays,
        each shaped [num_heads, len(tokens), head_dim], or None if any
        required block is missing (partial hits not supported in MVP).
        """
        hashes = self._compute_standalone_block_hashes(tokens)
        blocks = []
        for h in hashes:
            block = self.paged_cache.cached_block_hash_to_block.get(h)
            if block is None:
                return None
            blocks.append(block)

        per_layer_kv = self._concat_blocks_per_layer(blocks, num_tokens=len(tokens))
        return _ChunkKVHandle(per_layer_kv=per_layer_kv)

    def commit_chunk_as_standalone(self, tokens: List[int], per_layer_kv):
        """Insert a freshly-computed chunk's K/V into the prefix cache
        as if it had been a root-parented prefix, so subsequent requests
        can find it by `lookup_chunk_by_standalone_hash`.
        """
        hashes = self._compute_standalone_block_hashes(tokens)
        block_size = self.paged_cache.block_size
        for i, h in enumerate(hashes):
            start = i * block_size
            end = min(start + block_size, len(tokens))
            block_kv = [
                (k[:, start:end, :], v[:, start:end, :]) for (k, v) in per_layer_kv
            ]
            self.paged_cache.register_block_with_kv(
                block_hash=h,
                kv=block_kv,
                num_tokens=end - start,
            )

    def _concat_blocks_per_layer(self, blocks, num_tokens: int):
        """Gather per-layer K/V slices from the block list into contiguous tensors."""
        num_layers = len(blocks[0].per_layer_kv)
        out = []
        for layer in range(num_layers):
            ks, vs = [], []
            remaining = num_tokens
            for b in blocks:
                k, v = b.per_layer_kv[layer]
                take = min(remaining, k.shape[1])
                ks.append(k[:, :take, :])
                vs.append(v[:, :take, :])
                remaining -= take
                if remaining == 0:
                    break
            out.append((mx.concatenate(ks, axis=1), mx.concatenate(vs, axis=1)))
        return out
```

Also add the handle dataclass at module top (next to existing dataclasses):

```python
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class _ChunkKVHandle:
    per_layer_kv: List[Tuple[mx.array, mx.array]]
```

**Adaptation note:** `register_block_with_kv` and `block.per_layer_kv` are the names used in this plan for clarity; the real `PagedCache` class may expose KV under different attribute names (e.g. `block.kv_cache`, `paged_cache.register_block(...)`). At implementation time, `grep 'def register_block\|per_layer\|kv_cache' omlx/cache/paged_cache.py` and rename these calls to match the real API. If there is no existing method to register a block with arbitrary KV (because the paged cache normally fills KV during the forward pass), add a thin helper that allocates a block and stores the provided tensors.

- [ ] **Step 6: Run the tests, expect pass**

Run: `pytest tests/test_cacheblend_cache.py -v`
Expected: both tests PASS.

- [ ] **Step 7: Commit**

```bash
git add omlx/cache/prefix_cache.py tests/test_cacheblend_cache.py tests/fixtures/cacheblend/
git commit -m "feat(cache): standalone-hash chunk lookup and commit"
```

---

## Task 7: Separator-based chunk split

**Files:**
- Modify: `omlx/patches/cacheblend.py`
- Modify: `tests/test_cacheblend_unit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cacheblend_unit.py`:

```python
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
```

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/test_cacheblend_unit.py::test_split_prompt_basic tests/test_cacheblend_unit.py::test_split_prompt_no_separator_returns_none tests/test_cacheblend_unit.py::test_split_prompt_leading_and_trailing_separators_rejected -v`
Expected: FAIL — `split_prompt_on_separator` doesn't exist.

- [ ] **Step 3: Implement the split**

Append to `omlx/patches/cacheblend.py`:

```python
# -----------------------------------------------------------------------------
# Prompt splitting
# -----------------------------------------------------------------------------


def split_prompt_on_separator(
    prompt: str,
    tokenizer: Any,
    separator: str,
) -> Optional[Tuple[List[List[int]], List[int]]]:
    """Split a prompt on the CacheBlend separator.

    Returns (chunk_token_lists, query_tokens) on a clean split, else None.
    None signals the caller to fall back to standard prefill (separator
    not present, or prompt starts/ends with the separator).
    """
    if separator not in prompt:
        record_fallback("no_separator")
        return None

    parts = prompt.split(separator)
    # Require non-empty first chunk and non-empty query suffix.
    if len(parts) < 2 or not parts[0].strip() or not parts[-1].strip():
        record_fallback("tokenizer_split_misalign")
        return None

    chunk_texts = parts[:-1]
    query_text = parts[-1]

    chunks = [tokenizer.encode(c) for c in chunk_texts]
    query_tokens = tokenizer.encode(query_text)

    if any(len(c) == 0 for c in chunks) or len(query_tokens) == 0:
        record_fallback("tokenizer_split_misalign")
        return None

    return chunks, query_tokens
```

- [ ] **Step 4: Run, expect pass**

Run: `pytest tests/test_cacheblend_unit.py -v -k split_prompt`
Expected: all three PASS.

- [ ] **Step 5: Commit**

```bash
git add omlx/patches/cacheblend.py tests/test_cacheblend_unit.py
git commit -m "feat(cacheblend): prompt-separator split with fallback signaling"
```

---

## Task 8: Chunk classification + BlendMetadata assembly

**Files:**
- Modify: `omlx/patches/cacheblend.py`
- Modify: `tests/test_cacheblend_unit.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cacheblend_unit.py`:

```python
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
```

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/test_cacheblend_unit.py -v -k classify`
Expected: FAIL.

- [ ] **Step 3: Implement `build_blend_metadata`**

Append to `omlx/patches/cacheblend.py`:

```python
def build_blend_metadata(
    chunk_token_lists: List[List[int]],
    query_tokens: List[int],
    prefix_cache: Any,
    chunk_min_tokens: int,
) -> Optional[BlendMetadata]:
    """Classify each chunk as CACHED or COLD and compute blend metadata.

    Returns None if every chunk is COLD (no reuse possible → fall back).
    The query is always treated as a COLD chunk.
    """
    chunks: List[ChunkInfo] = []
    pos = 0
    any_cached = False

    for tokens in chunk_token_lists:
        handle = None
        if len(tokens) >= chunk_min_tokens:
            handle = prefix_cache.lookup_chunk_by_standalone_hash(tokens)
        kind = CACHED if handle is not None else COLD
        if kind == CACHED:
            any_cached = True
        chunks.append(ChunkInfo(tokens=tokens, kind=kind, start_pos=pos, cached_kv=handle))
        pos += len(tokens)

    chunks.append(ChunkInfo(tokens=query_tokens, kind=COLD, start_pos=pos, cached_kv=None))
    pos += len(query_tokens)

    if not any_cached:
        record_fallback("no_chunks_cached")
        return None

    total_len = pos
    cold_mask_list = []
    for chunk in chunks:
        cold_mask_list.extend([chunk.kind == COLD] * len(chunk.tokens))
    cold_mask = mx.array(cold_mask_list, dtype=mx.bool_)

    return BlendMetadata(chunks=chunks, total_len=total_len, cold_token_mask=cold_mask)
```

- [ ] **Step 4: Run, expect pass**

Run: `pytest tests/test_cacheblend_unit.py -v -k classify`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add omlx/patches/cacheblend.py tests/test_cacheblend_unit.py
git commit -m "feat(cacheblend): chunk classification and BlendMetadata assembly"
```

---

## Task 9: Llama layerwise forward monkey-patch

**Files:**
- Modify: `omlx/patches/cacheblend.py`
- Modify: `tests/test_cacheblend_unit.py`

- [ ] **Step 1: Study the Llama model class**

Run: `python -c "import mlx_lm.models.llama as m; import inspect; print(inspect.getsourcefile(m))"`
Open the printed path. Study `LlamaModel.__call__` and `TransformerBlock.__call__`. Note:
- how layers are iterated (`for layer in self.layers:`)
- how the attention sub-block exposes K/V (typically the cache object passed in holds it)
- where RoPE is applied — usually inside the attention submodule

- [ ] **Step 2: Write a minimal integration test with a tiny stub model**

Append to `tests/test_cacheblend_unit.py`:

```python
def test_patch_model_forward_falls_through_when_no_blend_metadata():
    """When the cache has no blend_metadata attribute, the patched forward
    must behave identically to the original forward."""
    pytest.importorskip("mlx_lm")
    from mlx_lm import load
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    from mlx_lm.models.cache import make_prompt_cache

    tokens = mx.array([tokenizer.encode("Hello")])
    cache_before = make_prompt_cache(model)
    out_before = model(tokens, cache=cache_before)

    cacheblend.patch_model_for_cacheblend(model)

    cache_after = make_prompt_cache(model)
    out_after = model(tokens, cache=cache_after)

    assert mx.allclose(out_before, out_after, atol=1e-4).item(), \
        "Patched model must produce identical logits when no blend metadata is attached"


def test_patched_forward_with_blend_metadata_runs_without_error():
    """Attach a trivial BlendMetadata (all COLD) and verify the layerwise
    path runs to completion with sensible output shape."""
    pytest.importorskip("mlx_lm")
    from mlx_lm import load
    model, _ = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    from mlx_lm.models.cache import make_prompt_cache

    cacheblend.patch_model_for_cacheblend(model)

    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    cache = make_prompt_cache(model)

    # All-COLD metadata: this should behave like full recompute, just
    # exercised through the blend codepath.
    cold_chunk = cacheblend.ChunkInfo(tokens=[1, 2, 3, 4, 5, 6, 7, 8], kind=cacheblend.COLD, start_pos=0)
    meta = cacheblend.BlendMetadata(
        chunks=[cold_chunk],
        total_len=8,
        cold_token_mask=mx.array([True] * 8),
    )
    cache[0].blend_metadata = meta   # attach to per-layer cache entry

    out = model(tokens, cache=cache)
    assert out.shape[1] == 8
    assert not mx.any(mx.isnan(out)).item()
```

- [ ] **Step 3: Run, expect failure**

Run: `pytest tests/test_cacheblend_unit.py::test_patch_model_forward_falls_through_when_no_blend_metadata -v`
Expected: FAIL — `patch_model_for_cacheblend` not defined.

- [ ] **Step 4: Implement the patch**

Append to `omlx/patches/cacheblend.py`:

```python
# -----------------------------------------------------------------------------
# Model monkey-patching
# -----------------------------------------------------------------------------


def _get_blend_metadata(cache) -> Optional[BlendMetadata]:
    """Extract BlendMetadata from a mlx-lm cache object if present."""
    if cache is None:
        return None
    first = cache[0] if isinstance(cache, (list, tuple)) else cache
    return getattr(first, "blend_metadata", None)


def patch_model_for_cacheblend(model) -> None:
    """Install CacheBlend's layerwise forward on the given mlx-lm model.

    Supported architectures: Llama (llama/mistral/etc. via LlamaModel),
    Qwen3. The patch is no-op unless a BlendMetadata is attached to the
    cache at forward time.
    """
    inner = getattr(model, "model", model)   # mlx-lm wraps in a causal-LM shell

    # Detect architecture via class name and dispatch.
    cls_name = type(inner).__name__
    if cls_name in ("LlamaModel", "Model"):     # mlx-lm Llama uses both spellings
        _patch_llama_like(inner)
    elif cls_name == "Qwen3Model":
        _patch_qwen3_like(inner)
    else:
        raise NotImplementedError(
            f"CacheBlend: no layerwise patch for model class {cls_name}. "
            f"Supported: LlamaModel, Qwen3Model."
        )


def _patch_llama_like(inner) -> None:
    """Wrap LlamaModel.__call__ with a blend-aware layerwise forward.

    The wrapper calls the original forward when no BlendMetadata is attached.
    When metadata IS attached, it runs layer 0 normally, HKVD-scores layer 1,
    and replaces the attention call in layers 2..N with sparse-Q attention.
    """
    original_call = inner.__call__

    def blended_call(self, inputs, cache=None, *args, **kwargs):
        meta = _get_blend_metadata(cache)
        if meta is None:
            return original_call(inputs, cache=cache, *args, **kwargs)

        try:
            return _run_blended_forward(self, inputs, cache, meta, *args, **kwargs)
        except Exception as exc:   # noqa: BLE001  — fall back hard
            logger.exception("CacheBlend layerwise forward failed; falling back: %s", exc)
            record_fallback("layerwise_exception")
            # Clear metadata so the fallback call doesn't re-enter blend path
            for layer_cache in cache:
                if hasattr(layer_cache, "blend_metadata"):
                    del layer_cache.blend_metadata
            return original_call(inputs, cache=cache, *args, **kwargs)

    # Bind as a method on the instance (don't mutate the class, to avoid
    # affecting other loaded models in the process).
    import types
    inner.__call__ = types.MethodType(blended_call, inner)


def _patch_qwen3_like(inner) -> None:
    """Qwen3 uses the same transformer-block layout as Llama in mlx-lm
    (attention + MLP + residuals), differing only in normalization
    ordering and rotary specifics. The blend wrapper is identical; we
    reuse _patch_llama_like to avoid duplication.
    """
    _patch_llama_like(inner)


def _run_blended_forward(inner, inputs, cache, meta: BlendMetadata, *args, **kwargs):
    """The actual layerwise pass.

    This implementation uses the model's own sub-modules (embeddings,
    layers, norm, lm_head) directly rather than re-calling the original
    __call__, so we can interpose HKVD and sparse-Q attention between
    layers.

    IMPORTANT: This function assumes mlx-lm's Llama/Qwen3 layer structure
    exposes layers via `inner.layers`, and each layer exposes `.self_attn`
    and the output of the block is `hidden + layer(hidden)`. If mlx-lm's
    layout changes, update this function.
    """
    h = inner.embed_tokens(inputs)

    # Layer 0: run normally.
    layer0 = inner.layers[0]
    h = layer0(h, mask=None, cache=cache[0])

    # Check layer: run the attention to get fresh Q/K, score HKVD.
    check_layer_idx = 1   # MVP default; config-driven in Task 12
    check_layer = inner.layers[check_layer_idx]
    k_fresh = _capture_fresh_k(check_layer, h, position_offsets=_build_position_offsets(meta))
    k_cached = _gather_cached_k(meta, layer_idx=check_layer_idx, head_dim=k_fresh.shape[-1])
    recompute_indices = hkvd_score(
        k_fresh=k_fresh, k_cached=k_cached,
        cold_token_mask=meta.cold_token_mask,
        recompute_ratio=getattr(meta, "recompute_ratio", 0.15),
    )
    meta.recompute_indices = recompute_indices

    # Run the check layer's full forward to produce h for subsequent layers.
    h = check_layer(h, mask=None, cache=cache[check_layer_idx])

    # Layers 2..N: sparse-Q attention.
    for i in range(check_layer_idx + 1, len(inner.layers)):
        layer = inner.layers[i]
        h = _blended_layer_forward(layer, h, cache[i], meta)

    h = inner.norm(h)
    return h


def _build_position_offsets(meta: BlendMetadata) -> mx.array:
    """Per-token offset used to rotate cached K into the right positions."""
    offsets = []
    for chunk in meta.chunks:
        if chunk.kind == CACHED:
            offsets.extend([chunk.start_pos] * len(chunk.tokens))
        else:
            offsets.extend([0] * len(chunk.tokens))   # COLD: K is fresh, no rotation needed
    return mx.array(offsets, dtype=mx.int32)


def _capture_fresh_k(layer, h, position_offsets: mx.array) -> mx.array:
    """Run the attention module far enough to get post-RoPE K, without
    touching the cache. Adapts to mlx-lm's LlamaAttention internals.
    """
    # Adapter: this function must be implemented against the real
    # LlamaAttention class. At implementation time, read
    # mlx_lm/models/llama.py and extract the projections + RoPE application.
    raise NotImplementedError(
        "Implement _capture_fresh_k against mlx-lm's attention module. "
        "See the plan's Task 9 notes."
    )


def _gather_cached_k(meta: BlendMetadata, layer_idx: int, head_dim: int) -> mx.array:
    """Concatenate per-chunk cached K for the given layer, with per-chunk
    offset RoPE applied. COLD chunks contribute zeros (masked out in HKVD).
    """
    num_heads_ref = None
    pieces = []
    for chunk in meta.chunks:
        if chunk.kind == CACHED and chunk.cached_kv is not None:
            k_chunk, _ = chunk.cached_kv.per_layer_kv[layer_idx]
            if num_heads_ref is None:
                num_heads_ref = k_chunk.shape[0]
            # Rotate by chunk's start position
            offsets = mx.array([chunk.start_pos] * len(chunk.tokens), dtype=mx.int32)
            k_rot = rotate_k_by_offsets(k_chunk, offsets=offsets, head_dim=head_dim)
            pieces.append(k_rot)
        else:
            # COLD chunk: placeholder zeros. Shape matches on concat after we
            # know num_heads from a cached chunk; if ALL chunks are COLD
            # we'd have returned None from build_blend_metadata.
            if num_heads_ref is None:
                num_heads_ref = 1  # will be overwritten on next loop iter
            pieces.append(mx.zeros((num_heads_ref, len(chunk.tokens), head_dim)))
    return mx.concatenate(pieces, axis=1)


def _blended_layer_forward(layer, h, layer_cache, meta: BlendMetadata):
    """Run one transformer block with sparse-Q attention.

    For MVP: recomputing *only* the Q at recompute_indices still requires
    running the layer's attention. We compute Q/K/V for the full sequence
    (inexpensive compared to attention's n^2 part), then call
    sparse_attn_selected_q, and scatter the result into a zero-initialized
    output buffer at recompute_indices. Non-selected tokens retain their
    hidden state from the pre-layer residual.

    This is a simplification of the paper (which also skips the K/V
    projections for non-selected tokens) — on Apple Silicon the
    attention quadratic dominates, so the projection speedup is a minor
    optimization we defer.
    """
    raise NotImplementedError(
        "Implement _blended_layer_forward against mlx-lm's attention module. "
        "See the plan's Task 9 notes."
    )
```

**Adaptation notes for Step 4:** `_capture_fresh_k` and `_blended_layer_forward` are intentionally left as `NotImplementedError` because their bodies depend on the exact mlx-lm `LlamaAttention` signature (how it exposes Q/K/V projections and RoPE). These two functions are the bulk of the task. Before filling them in:

1. Read `mlx_lm/models/llama.py` and locate the `Attention.__call__` body. It typically looks like:
   ```python
   queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
   # reshape to (B, H, T, D)
   queries = self.rope(queries, offset=cache.offset)
   keys = self.rope(keys, offset=cache.offset)
   # cache.update_and_fetch(keys, values)
   # attention
   ```
2. In `_capture_fresh_k`, replicate the projections + RoPE using `position_offsets` instead of the scalar `cache.offset`. You can either (a) apply RoPE per-token by looping (simple, slow) or (b) use the existing `rotate_k_by_offsets` to post-rotate after applying zero-RoPE.
3. In `_blended_layer_forward`, call the attention's projections on the full `h`, then `sparse_attn_selected_q(q, k, v, meta.recompute_indices, scale)`, then scatter the output into a copy of `h` at `recompute_indices`, apply the post-attention MLP only on those rows (optional optimization; safe to apply on all rows).

- [ ] **Step 5: Fill in `_capture_fresh_k` and `_blended_layer_forward` per the adaptation notes**

Use `_OffsetAdjustedRoPE` in `omlx/patches/specprefill.py` as the template for RoPE-with-per-token-offset. The signatures you implement MUST keep `_capture_fresh_k` returning a `[num_heads, total_len, head_dim]` tensor of post-RoPE K, and `_blended_layer_forward` returning the updated hidden-state tensor shaped like `h`.

- [ ] **Step 6: Run the patch tests, expect pass**

Run: `pytest tests/test_cacheblend_unit.py -v -k 'patch_model'`
Expected: both tests PASS. The first confirms byte-identical behavior without metadata; the second confirms the layerwise path runs without NaN.

- [ ] **Step 7: Commit**

```bash
git add omlx/patches/cacheblend.py tests/test_cacheblend_unit.py
git commit -m "feat(cacheblend): Llama/Qwen3 layerwise forward patch"
```

---

## Task 10: Scheduler hook — split, classify, attach metadata

**Files:**
- Modify: `omlx/engine/scheduler.py` OR `omlx/engine_core.py` (whichever owns prefill admission; the SpecPrefill hook is a precedent — find it first)
- Modify: `omlx/patches/cacheblend.py` (add the top-level orchestration function)
- Create: `tests/test_cacheblend_integration.py` (mock-heavy)

- [ ] **Step 1: Locate the SpecPrefill pre-prefill hook in scheduler**

Run: `grep -n 'specprefill\|_try_specprefill' omlx/engine/scheduler.py omlx/engine_core.py 2>/dev/null | head -20`
Identify the call site and pattern used for SpecPrefill admission; we'll add an analogous `_try_cacheblend_prefill` call right after it.

- [ ] **Step 2: Add the orchestration function**

Append to `omlx/patches/cacheblend.py`:

```python
def try_cacheblend_prefill(
    request,
    model,
    prefix_cache,
    settings,
) -> bool:
    """Pre-prefill hook. Returns True if blend was set up for this request,
    False if the caller should proceed with standard prefill.

    Side effects on True:
      - patch_model_for_cacheblend(model) is idempotent per model.
      - BlendMetadata is attached to each per-layer cache entry.
    """
    if not settings.cacheblend_enabled:
        return False

    if getattr(request, "specprefill_enabled", False) and settings.specprefill_enabled:
        record_fallback("specprefill_conflict")
        raise ValueError("CacheBlend and SpecPrefill cannot both be enabled for one request")

    split = split_prompt_on_separator(
        prompt=request.prompt,
        tokenizer=request.tokenizer,
        separator=settings.cacheblend_special_str,
    )
    if split is None:
        return False

    chunks, query = split
    meta = build_blend_metadata(
        chunk_token_lists=chunks,
        query_tokens=query,
        prefix_cache=prefix_cache,
        chunk_min_tokens=settings.cacheblend_chunk_min_tokens,
    )
    if meta is None:
        return False

    meta.recompute_ratio = settings.cacheblend_recompute_ratio

    patch_model_for_cacheblend(model)
    for layer_cache in request.cache:
        layer_cache.blend_metadata = meta
    return True
```

- [ ] **Step 3: Add the scheduler call site**

In the scheduler (the file identified in Step 1), immediately after the `_try_specprefill_scoring(request)` call (or wherever SpecPrefill admission happens):

```python
        # CacheBlend: attempt blended prefill before falling into the
        # standard prefill path. No-op if disabled or prompt has no
        # separator.
        try:
            from omlx.patches.cacheblend import try_cacheblend_prefill
            blended = try_cacheblend_prefill(
                request=request,
                model=self.model,
                prefix_cache=self.prefix_cache,
                settings=request.model_settings,
            )
        except Exception as exc:   # noqa: BLE001
            logger.exception("CacheBlend pre-prefill errored; falling back: %s", exc)
            blended = False
        # If blended == True, the normal prefill path will still run but
        # the patched model forward uses the attached BlendMetadata.
```

Adapt `self.model`, `self.prefix_cache`, and `request.model_settings` to the scheduler's actual attribute names (grep the neighboring code).

- [ ] **Step 4: Write a narrow integration test (mocks the model)**

Create `tests/test_cacheblend_integration.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Integration test: scheduler hook wires split + classify + attach."""

import types
import mlx.core as mx
import pytest

from omlx.patches import cacheblend


class _DummyRequest:
    def __init__(self, prompt, tokenizer, cache, specprefill_enabled=False):
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.cache = cache
        self.specprefill_enabled = specprefill_enabled

    @property
    def model_settings(self):
        s = types.SimpleNamespace(
            cacheblend_enabled=True,
            cacheblend_recompute_ratio=0.2,
            cacheblend_check_layers=[1],
            cacheblend_special_str=" # # ",
            cacheblend_chunk_min_tokens=1,
            specprefill_enabled=False,
        )
        return s


class _FakeTokenizer:
    def encode(self, s): return [ord(c) for c in s]


class _FakeLayerCache:
    pass


class _FakePrefixCache:
    def __init__(self, primed):
        self._primed = {tuple(k): v for k, v in primed.items()}
    def lookup_chunk_by_standalone_hash(self, tokens):
        return self._primed.get(tuple(tokens))


def test_try_cacheblend_prefill_attaches_metadata_on_hit(monkeypatch):
    # No-op patcher so we don't need a real model
    monkeypatch.setattr(cacheblend, "patch_model_for_cacheblend", lambda m: None)

    class _Handle:
        per_layer_kv = [(mx.zeros((1, 3, 8)), mx.zeros((1, 3, 8)))]

    cache = [_FakeLayerCache(), _FakeLayerCache()]
    req = _DummyRequest(
        prompt="AAA # # BBB # # Q",
        tokenizer=_FakeTokenizer(),
        cache=cache,
    )
    tokens_AAA = _FakeTokenizer().encode("AAA")
    prefix_cache = _FakePrefixCache(primed={tuple(tokens_AAA): _Handle()})

    ok = cacheblend.try_cacheblend_prefill(req, model=object(), prefix_cache=prefix_cache, settings=req.model_settings)

    assert ok is True
    for layer_cache in cache:
        assert hasattr(layer_cache, "blend_metadata")
        assert layer_cache.blend_metadata.total_len == len(tokens_AAA) + len(_FakeTokenizer().encode("BBB")) + 1


def test_try_cacheblend_prefill_noop_when_disabled():
    cache = [_FakeLayerCache()]
    req = _DummyRequest("A # # B # # Q", _FakeTokenizer(), cache)
    settings = req.model_settings
    settings.cacheblend_enabled = False
    prefix_cache = _FakePrefixCache(primed={})
    ok = cacheblend.try_cacheblend_prefill(req, model=object(), prefix_cache=prefix_cache, settings=settings)
    assert ok is False
    assert not hasattr(cache[0], "blend_metadata")


def test_try_cacheblend_prefill_raises_on_specprefill_conflict():
    cache = [_FakeLayerCache()]
    req = _DummyRequest("A # # B # # Q", _FakeTokenizer(), cache, specprefill_enabled=True)
    settings = req.model_settings
    settings.specprefill_enabled = True
    prefix_cache = _FakePrefixCache(primed={})
    with pytest.raises(ValueError, match="cannot both be enabled"):
        cacheblend.try_cacheblend_prefill(req, model=object(), prefix_cache=prefix_cache, settings=settings)
```

- [ ] **Step 5: Run tests, expect pass**

Run: `pytest tests/test_cacheblend_integration.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add omlx/engine/scheduler.py omlx/patches/cacheblend.py tests/test_cacheblend_integration.py
git commit -m "feat(cacheblend): scheduler hook to attach blend metadata"
```

---

## Task 11: Auto-warm — commit freshly computed chunks on finish

**Files:**
- Modify: `omlx/patches/cacheblend.py`
- Modify: scheduler file (hook into request-finish path)
- Modify: `tests/test_cacheblend_integration.py`

- [ ] **Step 1: Locate the request-finish call site**

Run: `grep -n 'def.*finish\|on_request_finish\|request_done\|finish_request' omlx/engine/scheduler.py omlx/engine_core.py 2>/dev/null | head -10`
Identify where a request transitions to done/cleanup.

- [ ] **Step 2: Implement the commit helper**

Append to `omlx/patches/cacheblend.py`:

```python
def commit_blend_cold_chunks(request, prefix_cache) -> int:
    """On request finish: for each chunk that was COLD, extract its per-layer
    K/V from the completed cache and register it as a standalone prefix
    so later requests can reuse it. Returns the number of chunks committed.
    """
    meta = _get_blend_metadata(request.cache)
    if meta is None:
        return 0

    num_layers = len(request.cache)
    committed = 0
    for chunk in meta.chunks:
        if chunk.kind != COLD:
            continue
        if chunk.tokens == getattr(request, "query_tokens", None):
            continue   # don't cache the query itself

        start, end = chunk.start_pos, chunk.start_pos + len(chunk.tokens)
        per_layer_kv = []
        for layer_idx in range(num_layers):
            layer_cache = request.cache[layer_idx]
            # Adapter: mlx-lm caches expose K/V under attribute names that
            # vary by cache type (e.g. `keys`, `values`, `_keys`, `_values`).
            # At implementation time, check the cache class and read the
            # correct attributes. Shape expected: [num_heads, total_len, head_dim].
            k = getattr(layer_cache, "keys", None)
            v = getattr(layer_cache, "values", None)
            if k is None or v is None:
                return committed   # unknown cache layout; skip auto-warm
            per_layer_kv.append((k[:, start:end, :], v[:, start:end, :]))

        try:
            prefix_cache.commit_chunk_as_standalone(chunk.tokens, per_layer_kv)
            committed += 1
        except Exception:   # noqa: BLE001
            logger.exception("cacheblend: failed to commit chunk on finish")
    return committed
```

- [ ] **Step 3: Wire into the scheduler's finish path**

At the finish/cleanup site identified in Step 1:

```python
        try:
            from omlx.patches.cacheblend import commit_blend_cold_chunks
            commit_blend_cold_chunks(request, self.prefix_cache)
        except Exception as exc:   # noqa: BLE001
            logger.exception("cacheblend: auto-warm commit failed: %s", exc)
```

- [ ] **Step 4: Add an integration test**

Append to `tests/test_cacheblend_integration.py`:

```python
def test_commit_blend_cold_chunks_writes_to_prefix_cache():
    meta = cacheblend.BlendMetadata(
        chunks=[
            cacheblend.ChunkInfo(tokens=[1, 2, 3], kind=cacheblend.COLD, start_pos=0),
            cacheblend.ChunkInfo(tokens=[4, 5], kind=cacheblend.COLD, start_pos=3),
        ],
        total_len=5,
    )

    class _LayerCache:
        def __init__(self):
            self.keys = mx.arange(5 * 4, dtype=mx.float32).reshape(1, 5, 4)
            self.values = mx.arange(5 * 4, dtype=mx.float32).reshape(1, 5, 4) * -1

        blend_metadata = None

    class _Req:
        def __init__(self, cache):
            self.cache = cache
            self.query_tokens = None

    cache = [_LayerCache(), _LayerCache()]
    cache[0].blend_metadata = meta

    committed_list = []

    class _Prefix:
        def commit_chunk_as_standalone(self, tokens, per_layer_kv):
            committed_list.append((tokens, len(per_layer_kv)))

    n = cacheblend.commit_blend_cold_chunks(_Req(cache), _Prefix())
    assert n == 2
    assert committed_list == [([1, 2, 3], 2), ([4, 5], 2)]
```

- [ ] **Step 5: Run tests, expect pass**

Run: `pytest tests/test_cacheblend_integration.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add omlx/engine/scheduler.py omlx/patches/cacheblend.py tests/test_cacheblend_integration.py
git commit -m "feat(cacheblend): auto-warm prefix cache on cold-chunk finish"
```

---

## Task 12: Honor `cacheblend_check_layers` config

**Files:**
- Modify: `omlx/patches/cacheblend.py`
- Modify: `tests/test_cacheblend_unit.py`

- [ ] **Step 1: Thread the check-layer index through metadata**

In `BlendMetadata`, add:
```python
    check_layer: int = 1
```

In `build_blend_metadata`, accept a `check_layer` argument (default 1) and set it on the returned metadata. In `try_cacheblend_prefill`, pass `settings.cacheblend_check_layers[0]` (MVP uses only index 0; list shape is reserved for future).

- [ ] **Step 2: Update `_run_blended_forward` to use `meta.check_layer`**

Replace the hardcoded `check_layer_idx = 1` with `check_layer_idx = meta.check_layer`.

- [ ] **Step 3: Write a test that sets check_layer=2 and confirms HKVD runs at layer 2**

Append to `tests/test_cacheblend_unit.py`:

```python
def test_check_layer_config_is_honored(monkeypatch):
    # Instrument hkvd_score to record which layer's K it was called with.
    seen = {}

    original = cacheblend.hkvd_score

    def spy(k_fresh, k_cached, cold_token_mask, recompute_ratio):
        seen["called"] = True
        return original(k_fresh, k_cached, cold_token_mask, recompute_ratio)

    monkeypatch.setattr(cacheblend, "hkvd_score", spy)
    # Rest of the test: construct a small model, a metadata with check_layer=2,
    # run forward, assert seen["called"] is True. This test depends on Task 9's
    # model integration test scaffolding; see that task for the load pattern.
```

Note: this test is a stub — the concrete assertion can only be made once Task 9's model-load scaffolding is reused. The implementer should copy the model-load block from `test_patched_forward_with_blend_metadata_runs_without_error` and extend it here.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cacheblend_unit.py -v -k check_layer`
Expected: PASS (once the stub is filled in).

- [ ] **Step 5: Commit**

```bash
git add omlx/patches/cacheblend.py tests/test_cacheblend_unit.py
git commit -m "feat(cacheblend): honor cacheblend_check_layers config"
```

---

## Task 13: End-to-end correctness test — Llama

**Files:**
- Create: `tests/test_cacheblend_correctness.py`
- Create: `tests/fixtures/cacheblend/rag_prompt_3doc.json`

- [ ] **Step 1: Create the prompt fixture**

```json
{
  "separator": " # # ",
  "docs": [
    "The Golden Gate Bridge is a suspension bridge in San Francisco, California. It was opened in 1937.",
    "Mount Everest is the highest mountain above sea level. Its peak is at 8,848 metres.",
    "The Pacific Ocean is the largest and deepest of Earth's oceans."
  ],
  "query": "Which of the three facts above mentions a year?"
}
```

- [ ] **Step 2: Write the correctness test**

```python
# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness: CacheBlend vs. full-recompute baseline."""

import json
from pathlib import Path

import mlx.core as mx
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


def _prewarm_chunks(model, tokenizer, prefix_cache, fixture):
    """Run each doc chunk as a standalone prefill, commit to standalone cache."""
    from mlx_lm.models.cache import make_prompt_cache
    for doc in fixture["docs"]:
        tokens = tokenizer.encode(doc)
        cache = make_prompt_cache(model)
        _ = model(mx.array([tokens]), cache=cache)
        # Extract per-layer KV and commit. See Task 11's commit helper for
        # the expected (k, v) per-layer attribute names.
        per_layer_kv = []
        for layer_cache in cache:
            k = getattr(layer_cache, "keys", None)
            v = getattr(layer_cache, "values", None)
            per_layer_kv.append((k[:, :len(tokens), :], v[:, :len(tokens), :]))
        prefix_cache.commit_chunk_as_standalone(tokens, per_layer_kv)


def _greedy_generate(model, tokenizer, prompt, cache, max_tokens: int):
    """Produce `max_tokens` greedy next tokens given a blended/full cache."""
    token_ids = tokenizer.encode(prompt)
    logits = model(mx.array([token_ids]), cache=cache)
    out = []
    for _ in range(max_tokens):
        next_tok = int(mx.argmax(logits[0, -1, :]).item())
        out.append(next_tok)
        logits = model(mx.array([[next_tok]]), cache=cache)
    return out


@pytest.mark.parametrize("model_id", [MODEL_ID_LLAMA, MODEL_ID_QWEN3])
def test_first_20_tokens_greedy_identical(model_id):
    pytest.importorskip("mlx_lm")
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from tests.fixtures.cacheblend.cache_factory import make_minimal_cache

    model, tokenizer = load(model_id)
    fixture = _load_fixture()
    prompt = _build_prompt(fixture)

    # Baseline: full recompute, no blend
    baseline_cache = make_prompt_cache(model)
    baseline_tokens = _greedy_generate(model, tokenizer, prompt, baseline_cache, max_tokens=20)

    # Blend: pre-warm, then run with metadata
    prefix_cache = make_minimal_cache()
    _prewarm_chunks(model, tokenizer, prefix_cache, fixture)
    cacheblend.patch_model_for_cacheblend(model)

    blend_cache = make_prompt_cache(model)
    # Attach metadata via try_cacheblend_prefill-like assembly
    tokenized_chunks = [tokenizer.encode(d) for d in fixture["docs"]]
    query_tokens = tokenizer.encode(fixture["query"])
    meta = cacheblend.build_blend_metadata(
        chunk_token_lists=tokenized_chunks,
        query_tokens=query_tokens,
        prefix_cache=prefix_cache,
        chunk_min_tokens=1,
    )
    assert meta is not None, "all chunks should be cached after prewarm"
    meta.recompute_ratio = 0.15
    for layer_cache in blend_cache:
        layer_cache.blend_metadata = meta

    blend_tokens = _greedy_generate(model, tokenizer, prompt, blend_cache, max_tokens=20)

    assert baseline_tokens == blend_tokens, (
        f"Blend divergence at first differing index; "
        f"baseline={baseline_tokens}, blend={blend_tokens}"
    )
```

- [ ] **Step 3: Run it, expect pass**

Run: `pytest tests/test_cacheblend_correctness.py -v -s`
Expected: both parametrized cases PASS. If one or both diverge, consult the open-questions section of the spec — this is exactly the risk we flagged.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cacheblend_correctness.py tests/fixtures/cacheblend/rag_prompt_3doc.json
git commit -m "test(cacheblend): end-to-end correctness vs full-recompute"
```

---

## Task 14: KL divergence test under sampling

**Files:**
- Modify: `tests/test_cacheblend_correctness.py`

- [ ] **Step 1: Add the KL test**

Append:

```python
@pytest.mark.parametrize("model_id", [MODEL_ID_LLAMA, MODEL_ID_QWEN3])
def test_first_20_tokens_kl_bounded_under_sampling(model_id):
    """At each of the first 20 positions, KL(blend || baseline) ≤ 0.05 nats."""
    pytest.importorskip("mlx_lm")
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from tests.fixtures.cacheblend.cache_factory import make_minimal_cache

    model, tokenizer = load(model_id)
    fixture = _load_fixture()
    prompt = _build_prompt(fixture)

    def _distributions(cache, max_steps: int):
        logits = model(mx.array([tokenizer.encode(prompt)]), cache=cache)
        dists = [mx.softmax(logits[0, -1, :], axis=-1)]
        for _ in range(max_steps - 1):
            # Use greedy to advance (we're comparing distributions at the same
            # token sequence, not exploring different rollouts).
            tok = int(mx.argmax(logits[0, -1, :]).item())
            logits = model(mx.array([[tok]]), cache=cache)
            dists.append(mx.softmax(logits[0, -1, :], axis=-1))
        return dists

    # Baseline
    baseline_cache = make_prompt_cache(model)
    p_dists = _distributions(baseline_cache, 20)

    # Blend
    prefix_cache = make_minimal_cache()
    _prewarm_chunks(model, tokenizer, prefix_cache, fixture)
    cacheblend.patch_model_for_cacheblend(model)
    blend_cache = make_prompt_cache(model)
    tokenized_chunks = [tokenizer.encode(d) for d in fixture["docs"]]
    query_tokens = tokenizer.encode(fixture["query"])
    meta = cacheblend.build_blend_metadata(
        chunk_token_lists=tokenized_chunks,
        query_tokens=query_tokens,
        prefix_cache=prefix_cache,
        chunk_min_tokens=1,
    )
    meta.recompute_ratio = 0.15
    for lc in blend_cache:
        lc.blend_metadata = meta
    q_dists = _distributions(blend_cache, 20)

    for i, (p, q) in enumerate(zip(p_dists, q_dists)):
        # Add tiny epsilon to q to avoid div-by-zero
        eps = 1e-12
        kl = float(mx.sum(p * (mx.log(p + eps) - mx.log(q + eps))).item())
        assert kl <= 0.05, f"KL at position {i} = {kl:.4f} exceeds 0.05 nats"
```

- [ ] **Step 2: Run, expect pass**

Run: `pytest tests/test_cacheblend_correctness.py::test_first_20_tokens_kl_bounded_under_sampling -v -s`
Expected: PASS on both models.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cacheblend_correctness.py
git commit -m "test(cacheblend): KL-bounded quality check under sampling"
```

---

## Task 15: TTFT micro-benchmark

**Files:**
- Create: `tests/bench/__init__.py` (empty)
- Create: `tests/bench/bench_cacheblend.py`

- [ ] **Step 1: Write the benchmark script**

```python
# SPDX-License-Identifier: Apache-2.0
"""CacheBlend TTFT micro-benchmark.

Runs 4 × 512-token chunks + 64-token query under:
  (a) full recompute (blend disabled)
  (b) cacheblend with pre-warmed chunks

Reports median + IQR of TTFT (token-0 latency), 30 runs each.
Ship gate: blend ≥ 1.5x speedup on the host.
"""

from __future__ import annotations

import json
import platform
import statistics
import string
import time
from pathlib import Path

import mlx.core as mx

from omlx.patches import cacheblend

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
NUM_CHUNKS = 4
TOKENS_PER_CHUNK = 512
QUERY_TOKENS = 64
RUNS = 30


def _synthetic_chunk(rng_seed: int, n: int) -> str:
    # Stable pseudorandom prose so hashes are reproducible across runs.
    import random
    r = random.Random(rng_seed)
    words = "".join(r.choices(string.ascii_lowercase + " ", k=n * 5))
    return words


def _synthetic_query(n: int) -> str:
    return "Summarize the above in one sentence. " * ((n // 8) + 1)


def measure_ttft(model, tokenizer, prompt, cache_factory, blend_setup=None) -> float:
    from mlx_lm.models.cache import make_prompt_cache
    cache = cache_factory(model)
    if blend_setup is not None:
        blend_setup(cache)
    tokens = tokenizer.encode(prompt)
    t0 = time.perf_counter()
    logits = model(mx.array([tokens]), cache=cache)
    _ = int(mx.argmax(logits[0, -1, :]).item())
    mx.eval(logits)   # force compute
    return time.perf_counter() - t0


def main() -> int:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from tests.fixtures.cacheblend.cache_factory import make_minimal_cache

    model, tokenizer = load(MODEL_ID)
    sep = " # # "
    chunks_text = [_synthetic_chunk(i, TOKENS_PER_CHUNK) for i in range(NUM_CHUNKS)]
    query_text = _synthetic_query(QUERY_TOKENS)
    prompt = sep.join(chunks_text + [query_text])

    # Pre-warm
    prefix_cache = make_minimal_cache()
    for text in chunks_text:
        toks = tokenizer.encode(text)
        c = make_prompt_cache(model)
        _ = model(mx.array([toks]), cache=c)
        per_layer_kv = []
        for lc in c:
            k = getattr(lc, "keys")
            v = getattr(lc, "values")
            per_layer_kv.append((k[:, :len(toks), :], v[:, :len(toks), :]))
        prefix_cache.commit_chunk_as_standalone(toks, per_layer_kv)

    cacheblend.patch_model_for_cacheblend(model)

    tokenized_chunks = [tokenizer.encode(t) for t in chunks_text]
    query_tokens = tokenizer.encode(query_text)

    def _blend_setup(cache):
        meta = cacheblend.build_blend_metadata(
            chunk_token_lists=tokenized_chunks,
            query_tokens=query_tokens,
            prefix_cache=prefix_cache,
            chunk_min_tokens=1,
        )
        assert meta is not None, "pre-warm failed"
        meta.recompute_ratio = 0.15
        for lc in cache:
            lc.blend_metadata = meta

    # Warmup
    for _ in range(3):
        measure_ttft(model, tokenizer, prompt, make_prompt_cache)
        measure_ttft(model, tokenizer, prompt, make_prompt_cache, blend_setup=_blend_setup)

    baseline = [measure_ttft(model, tokenizer, prompt, make_prompt_cache) for _ in range(RUNS)]
    blended = [measure_ttft(model, tokenizer, prompt, make_prompt_cache, blend_setup=_blend_setup) for _ in range(RUNS)]

    median_base = statistics.median(baseline)
    median_blend = statistics.median(blended)
    speedup = median_base / median_blend

    result = {
        "host": {"machine": platform.machine(), "platform": platform.platform()},
        "model_id": MODEL_ID,
        "config": {"chunks": NUM_CHUNKS, "tokens_per_chunk": TOKENS_PER_CHUNK,
                   "query_tokens": QUERY_TOKENS, "runs": RUNS, "recompute_ratio": 0.15},
        "baseline_ttft_s": {
            "median": median_base,
            "iqr": statistics.quantiles(baseline, n=4)[2] - statistics.quantiles(baseline, n=4)[0],
        },
        "blend_ttft_s": {
            "median": median_blend,
            "iqr": statistics.quantiles(blended, n=4)[2] - statistics.quantiles(blended, n=4)[0],
        },
        "speedup_x": speedup,
        "ship_gate_pass": speedup >= 1.5,
    }
    out_path = Path(__file__).parent / "bench_cacheblend_results.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["ship_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the benchmark**

Run: `python tests/bench/bench_cacheblend.py`
Expected: JSON result printed; `ship_gate_pass` is `true`. If false, the blend path has a perf regression — investigate before shipping.

- [ ] **Step 3: Commit results file and script**

```bash
git add tests/bench/__init__.py tests/bench/bench_cacheblend.py tests/bench/bench_cacheblend_results.json
git commit -m "bench(cacheblend): TTFT micro-benchmark with ship-gate assertion"
```

---

## Task 16: Full suite green + push

- [ ] **Step 1: Run the full CacheBlend test suite**

Run: `pytest tests/test_cacheblend_unit.py tests/test_cacheblend_cache.py tests/test_cacheblend_integration.py tests/test_cacheblend_correctness.py -v`
Expected: all green.

- [ ] **Step 2: Run the broader test suite to catch regressions**

Run: `pytest tests/ -q --ignore=tests/bench -x 2>&1 | tail -30`
Expected: no new failures introduced by the blend path (any failures must be pre-existing and unrelated).

- [ ] **Step 3: Push the feature branch to the fork**

Run: `git push fork cacheblend`
Expected: remote accepts the push (already tracking `fork/cacheblend`).

- [ ] **Step 4: Open PR against upstream (optional — only if ready for review)**

Run:
```bash
gh pr create --repo jundot/omlx --head intelc:cacheblend --title "feat: CacheBlend KV reuse for RAG (Llama + Qwen3)" --body "$(cat <<'EOF'
## Summary
- Adds CacheBlend (EuroSys '25) for per-chunk KV reuse in RAG-style prompts
- Supports llama and qwen3 architectures
- Graceful fallback to standard prefill for any edge case

## Test plan
- [x] Unit tests for HKVD, offset-RoPE, sparse-Q attention, chunk split, cache round-trip
- [x] End-to-end correctness vs. full-recompute baseline (20-token greedy identity + KL ≤ 0.05 under sampling)
- [x] TTFT micro-benchmark shows ≥1.5x speedup on Apple Silicon reference host
- [x] Config-disabled and no-separator paths verified byte-identical to today

## Design doc
See `docs/superpowers/specs/2026-04-22-cacheblend-design.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage check:** Every section of the spec maps to one or more tasks:

- Goals → Tasks 13, 14, 15
- Component layout → Tasks 2, 6, 9, 10
- Request flow → Tasks 7, 8, 9, 10, 11
- Cache extensions → Task 6
- Algorithm (HKVD, offset-RoPE, sparse-Q) → Tasks 3, 4, 5
- Configuration → Tasks 1, 12
- Error handling / fallback ladder → Task 2 (counter), Tasks 7, 8, 10, 11 (fallback sites)
- Testing (unit / correctness / benchmark) → Tasks 3–5 (unit), 13–14 (correctness), 15 (bench)
- Prerequisites (model download) → Task 0
- Mutual exclusion with SpecPrefill → Task 10 (integration test covers this)
- Rollout → Task 16

**Placeholder scan:** The plan has two deliberate `NotImplementedError` stubs in Task 9 (`_capture_fresh_k`, `_blended_layer_forward`) because these depend on mlx-lm's internal attention module layout. They are flagged with explicit adaptation notes citing `specprefill.py`'s `_OffsetAdjustedRoPE` as the template. Task 6 also has a deliberate `NotImplementedError` in `make_minimal_cache` with instructions to copy from `tests/test_cache_factory.py`. These are not "TODO later" placeholders — they are forcing the implementer to read specific existing files and adapt, which is the correct pattern when the upstream API can drift.

**Type consistency:** `BlendMetadata` gets a `recompute_ratio` attribute added dynamically in Tasks 10 and 13; this is a typed field on the dataclass only in Task 12 (`check_layer`). For consistency, add `recompute_ratio: float = 0.15` to `BlendMetadata` in Task 12 alongside `check_layer`. (Noted here; implementer should apply when they reach Task 12.)

**Scope:** 16 tasks, 2–5 minute steps, two 4-hour-ish task blocks (Task 9 model patching and Task 13/14 correctness). Fits within one implementation plan.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-22-cacheblend.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
