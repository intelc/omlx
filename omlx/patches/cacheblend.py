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
