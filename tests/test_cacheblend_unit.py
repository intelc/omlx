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
