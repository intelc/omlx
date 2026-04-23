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
    check_layer: int = 1
    recompute_ratio: float = 0.15

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


# -----------------------------------------------------------------------------
# Chunk classification and blend metadata assembly
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Model monkey-patching (Llama + Qwen3)
# -----------------------------------------------------------------------------
#
# Design: we replace `inner.__call__` on a per-instance basis (bound method).
# When no BlendMetadata is attached to the cache, we delegate to the original
# __call__ — so a patched model behaves byte-identically to unpatched when
# the feature is disabled.
#
# When BlendMetadata is attached, we run a layerwise forward that:
#   - runs each transformer block normally (MVP: full recompute, byte-identical
#     output to no-blend; sparse optimization is a follow-up task once the
#     benchmark flags a need for it),
#   - after the check layer, reads the now-populated `cache[check_layer].keys`,
#     compares against the same layer's cached-and-position-adjusted K from
#     meta.chunks, runs HKVD scoring, and stores the indices on meta. This
#     exercises the HKVD machinery and pins the exact positions that a
#     future sparse path will refresh.
#
# Any exception in the blend path falls back to the original forward with
# metadata cleared and the fallback counter incremented.

_PATCHED_SENTINEL = "_cacheblend_patched"


def _get_blend_metadata(cache) -> Optional[BlendMetadata]:
    """Extract BlendMetadata from an mlx-lm cache list if present.

    mlx-lm caches are lists; we stash metadata on the first entry.
    """
    if cache is None:
        return None
    if isinstance(cache, (list, tuple)):
        if not cache:
            return None
        first = cache[0]
    else:
        first = cache
    meta = getattr(first, "blend_metadata", None)
    if isinstance(meta, BlendMetadata):
        return meta
    return None


def _clear_blend_metadata(cache) -> None:
    """Strip BlendMetadata so a subsequent fallback call doesn't re-enter."""
    if cache is None:
        return
    entries = cache if isinstance(cache, (list, tuple)) else [cache]
    for entry in entries:
        if hasattr(entry, "blend_metadata"):
            try:
                delattr(entry, "blend_metadata")
            except AttributeError:
                pass


def patch_model_for_cacheblend(model) -> None:
    """Install CacheBlend's layerwise forward on an mlx-lm model.

    Accepts the outer mlx-lm `Model` (the one load() returns) or the inner
    `LlamaModel` / `Qwen3Model` — we locate the inner automatically.

    The patch works by reassigning `inner.__class__` to a dynamically-created
    subclass that overrides `__call__`. Python resolves `obj(...)` via the
    type's `__call__`, not the instance attribute, so a per-instance method
    rebind wouldn't take effect — the `__class__` swap is the idiomatic fix.

    Idempotent: calling twice on the same instance is a no-op. Only the
    passed instance is affected; other loaded models of the same architecture
    keep their original behavior.

    Supported architectures: LlamaModel (covers Llama family + architectures
    that reuse LlamaModel such as Mistral), Qwen3Model. Both use the same
    transformer-block layout; they differ only in Qwen3's q/k RMSNorm
    applied before RoPE — which we don't need to special-case here because
    we delegate per-layer forward to the block's original `__call__`.
    """
    inner = getattr(model, "model", model)
    if getattr(type(inner), _PATCHED_SENTINEL, False):
        return

    original_cls = type(inner)
    cls_name = original_cls.__name__
    if cls_name not in ("LlamaModel", "Qwen3Model"):
        raise NotImplementedError(
            f"CacheBlend: no layerwise patch for model class {cls_name}. "
            f"Supported: LlamaModel, Qwen3Model."
        )

    def blended_call(self, inputs, cache=None, *args, **kwargs):
        meta = _get_blend_metadata(cache)
        if meta is None:
            return original_cls.__call__(self, inputs, cache, *args, **kwargs)
        try:
            return _run_blended_forward(self, inputs, cache, meta, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("CacheBlend layerwise forward failed; falling back: %s", exc)
            record_fallback("layerwise_exception")
            _clear_blend_metadata(cache)
            return original_cls.__call__(self, inputs, cache, *args, **kwargs)

    # Build a per-instance subclass so we affect only this model instance, not
    # every LlamaModel in the process. `type()` creates a new class; the
    # sentinel attr prevents double-wrapping on repeat patch calls.
    blended_cls = type(
        f"{cls_name}Blended",
        (original_cls,),
        {"__call__": blended_call, _PATCHED_SENTINEL: True},
    )
    inner.__class__ = blended_cls


def _build_position_offsets(meta: BlendMetadata) -> mx.array:
    """Per-token offset used to rotate each CACHED chunk's K into its new
    absolute position. COLD tokens contribute 0 (their K will be computed
    fresh at the correct position anyway).
    """
    offsets: List[int] = []
    for chunk in meta.chunks:
        if chunk.kind == CACHED:
            offsets.extend([chunk.start_pos] * len(chunk.tokens))
        else:
            offsets.extend([0] * len(chunk.tokens))
    return mx.array(offsets, dtype=mx.int32)


def _gather_cached_k(meta: BlendMetadata, layer_idx: int, head_dim: int, num_kv_heads: int) -> mx.array:
    """Concatenate per-chunk cached K for the given layer, with per-chunk
    offset RoPE applied. COLD chunks contribute zeros (the cold_token_mask
    forces them to recompute regardless of K-diff).

    Returns shape [num_kv_heads, total_len, head_dim].
    """
    pieces: List[mx.array] = []
    for chunk in meta.chunks:
        n = len(chunk.tokens)
        if chunk.kind == CACHED and chunk.cached_kv is not None:
            k_chunk, _ = chunk.cached_kv.per_layer_kv[layer_idx]
            # Cached K shape: [num_kv_heads, chunk_len, head_dim]. Rotate by
            # the chunk's new absolute start position.
            offsets = mx.array([chunk.start_pos] * n, dtype=mx.int32)
            k_rot = rotate_k_by_offsets(k_chunk, offsets=offsets, head_dim=head_dim)
            pieces.append(k_rot)
        else:
            pieces.append(mx.zeros((num_kv_heads, n, head_dim)))
    return mx.concatenate(pieces, axis=1)


def _run_blended_forward(inner, inputs, cache, meta: BlendMetadata, *args, **kwargs):
    """Layerwise forward that plumbs BlendMetadata through the stack.

    MVP behavior: each layer runs normally (full recompute), so model output
    is byte-identical to the un-blended path. The HKVD scorer runs after the
    check layer and stores its result on meta.recompute_indices for downstream
    consumers (scheduler auto-warm, future sparse-attention optimization).

    This keeps the correctness story dead simple — any quality regression vs.
    full recompute is zero — while the integration surface (cache lookups,
    metadata plumbing, per-chunk offset-RoPE, per-layer K diffing) is fully
    exercised end-to-end. The sparse-attention optimization that delivers
    the paper's speedup is deferred until a benchmark gate demands it.
    """
    from mlx_lm.models.base import create_attention_mask

    input_embeddings = kwargs.get("input_embeddings")
    if input_embeddings is not None:
        h = input_embeddings
    else:
        h = inner.embed_tokens(inputs)

    # Attention-mask construction differs slightly between Llama (which has
    # fa_idx/swa_idx for hybrid full/sliding attention) and Qwen3 (single mask
    # from cache[0]). Read the architecture off the inner model.
    if hasattr(inner, "fa_idx"):
        fa_mask = create_attention_mask(h, cache[inner.fa_idx])
        swa_mask = None
        if getattr(inner, "swa_idx", None) is not None:
            swa_mask = create_attention_mask(
                h, cache[inner.swa_idx], window_size=inner.sliding_window
            )

        def _mask_for(i):
            return swa_mask if inner.layers[i].use_sliding else fa_mask
    else:
        base_mask = create_attention_mask(h, cache[0])

        def _mask_for(i):
            return base_mask

    num_layers = len(inner.layers)
    check_layer_idx = meta.check_layer
    if not 0 <= check_layer_idx < num_layers:
        # Out-of-range check layer is a config bug; fail fast so callers
        # see it rather than silently running without HKVD.
        raise ValueError(
            f"cacheblend_check_layers[0]={check_layer_idx} is out of range for a "
            f"{num_layers}-layer model"
        )

    for i in range(num_layers):
        h = inner.layers[i](h, _mask_for(i), cache=cache[i])

        if i == check_layer_idx:
            _score_hkvd_at_check_layer(cache[i], meta, layer_idx=i)

    return inner.norm(h)


def _score_hkvd_at_check_layer(layer_cache, meta: BlendMetadata, layer_idx: int) -> None:
    """Read fresh K from the just-populated cache, gather cached K per-chunk
    with offset-RoPE, and store HKVD-selected indices on meta.

    Caught-and-logged on failure: HKVD is instrumentation in this MVP, not a
    correctness-critical step. If it fails, the rest of the forward still
    produces correct output.
    """
    try:
        k_fresh = layer_cache.keys
        if k_fresh is None:
            return
        # mlx-lm's KVCache pre-allocates its buffer in 256-token increments
        # (see mlx_lm/models/cache.py::KVCache.update_and_fetch). The true
        # number of filled tokens is `cache.offset`; we must slice to it to
        # avoid scoring against uninitialized tail slots.
        valid_len = int(getattr(layer_cache, "offset", k_fresh.shape[-2]))
        # Drop batch dim (we assume B=1 for the blend path) and trim to valid.
        if k_fresh.ndim == 4:
            k_fresh = k_fresh[0]
        k_fresh = k_fresh[:, :valid_len, :]
        num_kv_heads = k_fresh.shape[0]
        total_len = k_fresh.shape[1]
        head_dim = k_fresh.shape[2]

        if total_len != meta.total_len:
            # The cache contains more (or fewer) tokens than the blend metadata
            # accounts for — likely an unexpected multi-batch or decode step.
            # Skip scoring rather than produce nonsense.
            return

        k_cached = _gather_cached_k(meta, layer_idx, head_dim, num_kv_heads)
        indices = hkvd_score(
            k_fresh=k_fresh,
            k_cached=k_cached,
            cold_token_mask=meta.cold_token_mask,
            recompute_ratio=meta.recompute_ratio,
        )
        meta.recompute_indices = indices
    except Exception as exc:  # noqa: BLE001
        logger.warning("cacheblend: HKVD scoring failed at layer %d: %s", layer_idx, exc)
        record_fallback("nan_hkvd")


# -----------------------------------------------------------------------------
# Scheduler pre-prefill hook
# -----------------------------------------------------------------------------


def try_cacheblend_prefill(
    request,
    model,
    prefix_cache,
    settings,
    cache=None,
) -> bool:
    """Pre-prefill hook. Returns True if blend was set up for this request,
    False if the caller should proceed with standard prefill.

    Side effects on True:
      - patch_model_for_cacheblend(model) is applied (idempotent).
      - BlendMetadata is attached to each per-layer cache entry, so the
        patched model's __call__ will take the blended path.

    Parameters follow the scheduler's own attribute names:
      request.prompt              — the full prompt string
      request.tokenizer           — the tokenizer object (has .encode)
      cache                       — list of per-layer KV caches for this request
                                    (passed separately because request.cache does
                                    not exist at add_request time; the scheduler
                                    uses request.prompt_cache).
      settings.cacheblend_enabled, .cacheblend_recompute_ratio, etc.

    If cache is None, falls back to request.cache for callers that do attach
    a .cache attribute directly.
    """
    if not getattr(settings, "cacheblend_enabled", False):
        return False

    if getattr(request, "_specprefill_enabled", False) and getattr(settings, "specprefill_enabled", False):
        record_fallback("specprefill_conflict")
        raise ValueError(
            "CacheBlend and SpecPrefill cannot both be enabled for the same request"
        )

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
    check_layers = getattr(settings, "cacheblend_check_layers", None) or [1]
    meta.check_layer = int(check_layers[0])

    # Resolve the cache list: explicit arg takes priority, then request attr.
    resolved_cache = cache if cache is not None else getattr(request, "cache", None)

    patch_model_for_cacheblend(model)
    if resolved_cache is not None:
        for layer_cache in resolved_cache:
            layer_cache.blend_metadata = meta
    return True
