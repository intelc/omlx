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
