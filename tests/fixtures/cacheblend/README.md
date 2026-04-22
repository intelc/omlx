# CacheBlend test fixtures

Pinned model IDs used by correctness and benchmark tests:

- Llama: `mlx-community/Llama-3.2-1B-Instruct-4bit`
- Qwen3: `mlx-community/Qwen3-1.7B-4bit`

If either becomes unavailable, update the `MODEL_ID_*` constants in
`tests/test_cacheblend_correctness.py` and `tests/bench/bench_cacheblend.py`.
