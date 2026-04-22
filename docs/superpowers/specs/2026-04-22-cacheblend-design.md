# CacheBlend for omlx — Design

Date: 2026-04-22
Author: intelc
Status: Proposed

## Motivation

omlx currently has rich prefix-based KV caching and SpecPrefill (attention-based sparse prefill on a single long prompt), but no way to reuse KV across *independent* chunks that appear mid-prompt. In RAG deployments this is the common shape: a request is `<doc1> + <doc2> + ... + <query>`, the docs are drawn from a retrieval index, and they rarely share a common prefix with the request being served.

CacheBlend (ACM EuroSys '25, LMCache's reference implementation) addresses this: take cached K/V for each chunk as if standalone, concatenate with correct per-chunk positional offsets, then recompute K/V only for the small fraction of tokens with the highest cross-attention deviation (HKVD). The paper reports TTFT 2.2–3.3× and throughput 2.8–5× over full recompute with negligible quality loss.

This design adds CacheBlend to omlx for `llama` and `qwen3` architectures, matching LMCache's coverage and allowing us to validate speedup on Apple Silicon.

## Goals (MVP)

1. Correctness: CacheBlend output on hand-crafted RAG prompts matches full-recompute baseline within a bounded logprob distance under sampling, and is token-identical under greedy decoding for at least the first N generated tokens.
2. Measured speedup: micro-benchmark shows ≥1.5× TTFT improvement over full recompute on the benchmark host for a 4-chunk × 512-token input with pre-warmed chunks.
3. Architecture coverage: `llama` (covers Llama 3.x family) and `qwen3`.
4. Graceful fallback: feature degrades to standard prefill for any edge case; never crashes a request.

## Non-Goals

- Partial block hits within a chunk. All-or-nothing chunk retrieval in MVP.
- More than one HKVD check layer. `check_layers=[1]` only; config is list-shaped to future-proof.
- VLM / multimodal RAG. Text-only in MVP.
- F1-quality validation on a real RAG dataset (2WikiMQA, Musique). Deferred to a follow-up.
- Streaming or concurrent blend of different requests sharing chunks. Correctness before optimization.
- Pre-population admin endpoint. Auto-warm on first cold request is sufficient for MVP.
- Interop with SpecPrefill on the same request. Mutually exclusive; documented and asserted.

## Prerequisites

No `llama` or `qwen3` checkpoints are currently present in this development environment. Before running the test/benchmark suite, the smallest suitable mlx-community checkpoints must be downloaded via omlx's existing model-download flow (`huggingface_hub.snapshot_download` through the admin API, or an equivalent `mlx_lm.load(...)` call during test setup). The implementation plan will specify exact model IDs when we reach the test stage; expected targets are a 1–3B Llama 3.2 variant and a small Qwen3 variant.

## Decisions Locked

| Decision | Choice | Rationale |
|---|---|---|
| MVP scope | Correctness + speedup micro-benchmark | Validate on Apple Silicon without committing to full quality eval |
| Chunk API | LMCache-compatible separator string | Zero client-side schema changes, A/B-testable against LMCache |
| Chunk warming | Hybrid auto-warm; cold request warms cache as side effect | Degrades gracefully; no new admin surface |
| Model scope | `llama` + `qwen3` | Direct line-by-line port from LMCache reference |
| Config surface | `model_settings.py` fields (SpecPrefill pattern) | Consistent with existing per-model config plumbing |
| Fork mechanics | In place; add `fork` remote pointing at `intelc/omlx` | Current checkout is clean; simpler than a second working copy |
| Layerwise forward | Runtime monkey-patch of `mlx_lm` model classes | Mirrors SpecPrefill; minimizes upstream-drift maintenance |
| Sparse attention | Generalize SpecPrefill's existing primitive | ~50 LOC, no new kernel needed |

## Component Layout

```
omlx/
├── patches/
│   └── cacheblend.py                    [NEW ~700 lines]
│       ├── Config loading (from model_settings)
│       ├── _patch_llama_forward()       # monkey-patches mlx_lm.models.llama
│       ├── _patch_qwen3_forward()       # monkey-patches mlx_lm.models.qwen3
│       ├── BlendMetadata dataclass      # attached to the cache object per request
│       ├── hkvd_score()                 # L2 diff on check layer → top-k indices
│       ├── sparse_attn_selected_q()     # selected Q vs. full K/V
│       ├── blend_chunk_kvs()            # concat per-chunk cached K/V with position fix-up
│       └── cleanup_blend_state()
│
├── cache/
│   └── prefix_cache.py                  [EDIT +~100 lines]
│       ├── lookup_chunk_by_standalone_hash()
│       └── commit_chunk_as_standalone()
│
├── engine/                              [EDIT ~50 lines]
│   └── scheduler.py or engine_core.py
│       └── split_on_separator_and_blend()  # pre-prefill hook
│
├── model_settings.py                    [EDIT +~30 lines]
│   └── cacheblend_enabled, cacheblend_recompute_ratio,
│       cacheblend_check_layers, cacheblend_special_str,
│       cacheblend_chunk_min_tokens
│
├── api/openai_models.py                 [EDIT ~20 lines]
│   └── Thread optional per-request separator override
│
└── tests/
    ├── test_cacheblend_unit.py          [NEW]
    ├── test_cacheblend_correctness.py   [NEW]
    └── bench/bench_cacheblend.py        [NEW]
```

**Invariants.**
- When `cacheblend_enabled=False` or prompt lacks the separator, control flow is byte-identical to today.
- All blend logic lives in `patches/` plus small additions to `cache/`. No new top-level directories.
- CacheBlend and SpecPrefill are mutually exclusive per request; enabling both fails fast at request admission.

## Request Flow

```
1. Request arrives at scheduler
   ↓
2. Pre-prefill hook (patches/cacheblend.py::split_on_separator_and_blend)
   • If cacheblend_enabled and prompt contains SPECIAL_STR:
       - Tokenize, split token stream on separator tokens
       - For each non-query chunk, compute standalone_hash (parent=empty)
       - Look each chunk up via prefix_cache.lookup_chunk_by_standalone_hash
       - Classify: CACHED or COLD
   • If no chunks CACHED: fall back to standard prefill, then on finish
     commit each chunk as standalone (hybrid auto-warm)
   ↓
3. If ≥1 chunk CACHED: attach BlendMetadata to the cache object
     { chunk_boundaries, cached_kvs, cold_chunk_ranges,
       recompute_indices: None (filled in after check layer) }
   ↓
4. Patched model forward (_patch_llama_forward / _patch_qwen3_forward)
   Layer 0:
     • Runs normally on all tokens, producing fresh K/V
   Check layer (default layer 1):
     • Compute fresh K at true absolute positions (per-chunk RoPE offset
       for CACHED tokens; direct for COLD)
     • diff_k = ||K_fresh - K_cached||² per token; ∞ for COLD tokens
     • top-k indices = argtop(diff_k, k=total_len * recompute_ratio)
     • Union with all COLD-chunk indices
     • Store in BlendMetadata.recompute_indices
   Layers 2..N:
     • sparse_attn_selected_q: only recompute_indices tokens refresh Q/K/V
     • Their Q attends against FULL K/V (CACHED + already-refreshed)
     • New K/V scattered back into cache at their positions
     • Non-recomputed tokens retain cached K/V as-is
   ↓
5. Decode proceeds normally against the blended cache
   ↓
6. On request finish: COLD chunks freshly computed this turn are
   committed via commit_chunk_as_standalone; future requests benefit
```

## Cache Extensions

Two new methods on `PrefixCache`:

```python
def lookup_chunk_by_standalone_hash(self, chunk_tokens: list[int]) -> Optional[ChunkKVHandle]:
    """
    Look up a chunk's KV as if it were a standalone prefix from root.
    Walks block-by-block with parent_hash = b"" at the start, following
    the same compute_block_hash recurrence used during insertion. Returns
    None if any block along the way is missing (partial hits not supported
    in MVP).
    """

def commit_chunk_as_standalone(self, chunk_tokens: list[int], per_layer_kvs) -> None:
    """
    Insert a freshly-computed chunk's KV into the prefix cache as if it
    had been a root-parented prefix. Makes it retrievable by
    lookup_chunk_by_standalone_hash on subsequent requests.
    """
```

These reuse existing block-hash and reference-counting machinery; they do not change how current prefix lookups behave. The SSD tier (`paged_ssd_cache.py`) is hash-addressed, so persistence and restoration work automatically once the block hash is computable. Partial last blocks inherit the existing `compute_block_hash` behavior — no special handling needed.

## Algorithm

**HKVD on the check layer:**
```python
# After fresh K is computed with correct absolute-position RoPE
diff_k = mx.sum((K_fresh.astype(mx.float32) - K_cached.astype(mx.float32))**2, axis=-1)
per_token_score = mx.mean(diff_k, axis=0)                 # mean over heads
per_token_score = mx.where(is_cold_token_mask, mx.inf, per_token_score)
k = int(total_len * recompute_ratio)
recompute_indices = mx.argsort(-per_token_score)[:k]
recompute_indices = mx.sort(recompute_indices)            # for cache locality
```

**Position fix-up (per-chunk RoPE offset):**
Each chunk C was RoPE'd at `[0..len(C))` when cached. At query time it occupies `[offset_C..offset_C + len(C))`. The cached K is rotated by `offset_C` using the existing angle table — a generalization of `_OffsetAdjustedRoPE` in `specprefill.py` to accept a per-chunk offset vector instead of a scalar.

**Sparse attention (layers 2..N):**
SpecPrefill currently does *"full Q, selected K/V"*. CacheBlend needs *"selected Q, full K/V"*. Both resolve to a masked `mx.fast.scaled_dot_product_attention` call after the right slicing:

```python
Q_selected = Q[:, recompute_indices, :]
out_selected = mx.fast.scaled_dot_product_attention(
    Q_selected, K_full, V_full, scale=scale, mask=causal_mask_at_absolute_positions
)
# Scatter out_selected back into the residual stream at recompute_indices
# Fresh K/V at these positions replaces the cached K/V in place
```

Two correctness details:
1. The mask for `Q_selected` is the *causal* mask against absolute positions — selected token at position m attends to all cached/fresh K at positions ≤ m. It is *not* derived from the `recompute_indices` ordering.
2. Recompute indices stay fixed across layers 2..N; each layer refines the K/V at those slots, and the K/V buffer handed to layer L+1 reflects layer L's scattered updates.

**What's reused from existing omlx code.**
- Offset-RoPE math (SpecPrefill already implements the scalar case).
- Sparse-Q attention with index scatter (small generalization of SpecPrefill's sparse prefill path).
- Attention mask construction utilities.
- Per-layer cache addressing from the paged cache.

**What's genuinely new.**
- HKVD scorer (~30 LOC).
- Per-chunk offset-RoPE generalization (~40 LOC on top of SpecPrefill).
- `Q_selected` vs. `K_full` primitive (~50 LOC).

## Configuration

In `model_settings.py`, matching the SpecPrefill pattern:
```python
cacheblend_enabled: bool = False
cacheblend_recompute_ratio: float = 0.15
cacheblend_check_layers: list[int] = [1]
cacheblend_special_str: str = " # # "
cacheblend_chunk_min_tokens: int = 32
```

Chunks shorter than `cacheblend_chunk_min_tokens` skip the blend path for that chunk — blend-machinery overhead outweighs savings on tiny chunks.

## Error Handling / Fallback

Any blend-path failure falls back to standard prefill and emits a counter:

1. Separator not found in prompt → standard prefill (no blend).
2. Zero chunks CACHED (all cold) → standard prefill + auto-warm on finish.
3. Tokenizer produces a separator split that doesn't round-cleanly → log warning, standard prefill.
4. Cached chunk's per-layer K/V shape mismatch (model config changed since cache write) → invalidate that chunk's entry, treat as COLD.
5. HKVD produces NaN/Inf → standard prefill, emit counter.
6. `cacheblend_enabled` and `specprefill_enabled` both true → fail fast at request admission.

Counter: `cacheblend_fallback_total{reason=...}` — lets us see in production which fallback path actually fires.

## Testing

**Unit tests (`tests/test_cacheblend_unit.py`).**
- `hkvd_score()` returns top-k indices matching a numpy reference on synthetic K tensors.
- `lookup_chunk_by_standalone_hash()` round-trips: insert a chunk, look it up, receive identical K/V.
- Offset-RoPE: applying offset θ to cached K and decoding gives attention scores identical to computing K fresh at position θ.
- Separator-split edge cases: empty chunks, leading/trailing separators, separator as the whole prompt.

**Correctness tests (`tests/test_cacheblend_correctness.py`).**
- Hand-crafted RAG prompt: 3 documents + 1 query, joined by the separator.
- Run once with CacheBlend enabled (chunks pre-warmed) and once disabled (full recompute).
- Assert, under greedy decoding: identical tokens for at least the first 20 generated tokens.
- Assert, under sampling (temperature 1.0): KL divergence between blend-on and blend-off next-token distributions at each of the first 20 positions is ≤ 0.05 nats.
- Cover both `llama` and `qwen3` checkpoints.

**Benchmark (`tests/bench/bench_cacheblend.py`).**
- Inputs: 4 chunks × 512 tokens + 64-token query, pre-warmed.
- Measure TTFT with blend on vs. off, 30 runs each, report median + IQR.
- Benchmark host: whichever machine runs the suite (recorded in the benchmark output for reproducibility). Initial reference host is the development machine running this work.
- Ship-pass threshold: ≥1.5× median TTFT speedup on the reference host.

## Rollout

1. Fork `jundot/omlx` to `intelc/omlx`; add `fork` remote in the current working copy.
2. Create branch `cacheblend` off `main`.
3. Implement in the order laid out by the implementation plan (to be produced next).
4. Merge to `fork/cacheblend`, open a PR against `jundot/omlx:main` when correctness + benchmark gates pass.

## Open Questions / Risks

- **Reproducibility risk.** LMCache issues (#2921, #2026) indicate the paper's speedup numbers have been hard to hit in the deployed implementation. If our MVP benchmark shows materially less than 1.5× on Apple Silicon, we need to decide whether to keep iterating or escalate to the "C" quality-eval cut to understand if something is wrong with our port vs. the algorithm itself.
- **Model-download logistics.** Tests and benchmarks require real checkpoints. The implementation plan must sequence model download ahead of any test-running step.
- **Block-size alignment.** The 256-token block boundary means a 500-token chunk stores as `[block(256), block(244-partial)]`. Chunks with lengths that round poorly against 256 may have different cache hit behavior than clean multiples. Worth confirming empirically.
