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
