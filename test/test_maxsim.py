#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test MaxSim (ColBERT late-interaction): nk.maxsim, nk.maxsim_pack, nk.maxsim_packed.

Covers dtypes: float32, bfloat16, float16.
Parametrized over: capability from possible_capabilities.

Precision notes:
    MaxSim computes angular distances via coarse screening + fine pass.
    float16 carries wider quantization noise. bfloat16 cannot be represented
    in NumPy; tests verify finiteness and non-negativity only.
"""

import atexit
import pytest

try:
    import numpy as np
except:
    np = None

import numkong as nk
from test_base import (
    numpy_available,
    possible_capabilities,
    randomized_repetitions_count,
    keep_one_capability,
    tolerances_for_dtype,
    create_stats,
    print_stats_report,
    seed_rng,
)

stats = create_stats()
atexit.register(print_stats_report, stats)

_MAXSIM_DTYPE = {"float32": "f32", "bfloat16": "bf16", "float16": "f16"}


def baseline_maxsim(queries, documents):
    """Pure NumPy reference: sum of per-query minimum angular distances."""
    total = 0.0
    for q in queries:
        q64 = q.astype(np.float64)
        norm_q = np.linalg.norm(q64)
        min_ang = float("inf")
        for d in documents:
            d64 = d.astype(np.float64)
            norm_d = np.linalg.norm(d64)
            dot = np.dot(q64, d64)
            angular = max(0.0, 1.0 - dot / (norm_q * norm_d)) if norm_q > 0 and norm_d > 0 else 1.0
            min_ang = min(min_ang, angular)
        total += min_ang
    return total


def _make_matrix(rows, cols, dtype):
    """Create a test matrix in the target dtype."""
    if dtype == "bfloat16":
        # NumPy has no bf16; create f16 array and reinterpret via nk.Tensor
        arr = np.random.randn(rows, cols).astype(np.float16)
        return nk.Tensor(arr, dtype="bf16")
    return np.random.randn(rows, cols).astype(dtype)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_pack_and_packed(dtype, capability):
    """Pack + compute vs baseline."""
    keep_one_capability(capability)
    n_q, n_d, depth = 8, 16, 64
    dtype_str = _MAXSIM_DTYPE[dtype]
    queries = _make_matrix(n_q, depth, dtype)
    documents = _make_matrix(n_d, depth, dtype)

    qp = nk.maxsim_pack(queries, dtype=dtype_str)
    dp = nk.maxsim_pack(documents, dtype=dtype_str)

    assert isinstance(qp, nk.MaxSimPackedMatrix)
    assert qp.vector_count == n_q
    assert qp.depth == depth
    assert qp.nbytes > 0

    result = nk.maxsim_packed(qp, dp)

    if dtype == "bfloat16":
        # NumPy cannot represent bf16; just verify result is finite and positive
        assert np.isfinite(result), f"maxsim_packed(bf16): got non-finite {result}"
        assert result >= 0, f"maxsim_packed(bf16): got negative {result}"
    else:
        q_np = np.array(queries, dtype=dtype)
        d_np = np.array(documents, dtype=dtype)
        expected = baseline_maxsim(q_np, d_np)
        atol, rtol = tolerances_for_dtype(dtype)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_convenience(dtype, capability):
    """Verify maxsim() matches maxsim_pack + maxsim_packed."""
    keep_one_capability(capability)
    n_q, n_d, depth = 4, 8, 32
    dtype_str = _MAXSIM_DTYPE[dtype]
    queries = _make_matrix(n_q, depth, dtype)
    documents = _make_matrix(n_d, depth, dtype)

    # Packed path
    qp = nk.maxsim_pack(queries, dtype=dtype_str)
    dp = nk.maxsim_pack(documents, dtype=dtype_str)
    packed_result = nk.maxsim_packed(qp, dp)

    # Convenience path
    conv_result = nk.maxsim(queries, documents, dtype=dtype_str)

    np.testing.assert_allclose(packed_result, conv_result, atol=1e-6, rtol=0)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_self_zero(capability):
    """Identical queries and documents should give score near zero."""
    keep_one_capability(capability)
    vectors = np.random.randn(8, 64).astype(np.float32)
    result = nk.maxsim(vectors, vectors, dtype="f32")
    np.testing.assert_allclose(result, 0.0, atol=0.1)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_type_errors(capability):
    """Wrong type and mismatched dtype/depth."""
    keep_one_capability(capability)
    q = np.random.randn(4, 32).astype(np.float32)
    d = np.random.randn(8, 32).astype(np.float32)
    qp = nk.maxsim_pack(q, dtype="f32")
    dp = nk.maxsim_pack(d, dtype="f32")

    # Wrong type for maxsim_packed
    with pytest.raises(TypeError):
        nk.maxsim_packed(q, dp)
    with pytest.raises(TypeError):
        nk.maxsim_packed(qp, d)

    # Mismatched depth
    d_wrong = np.random.randn(8, 16).astype(np.float32)
    dp_wrong = nk.maxsim_pack(d_wrong, dtype="f32")
    with pytest.raises(ValueError):
        nk.maxsim_packed(qp, dp_wrong)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_repr(capability):
    """Verify repr shows MaxSimPackedMatrix."""
    keep_one_capability(capability)
    q = np.random.randn(4, 32).astype(np.float32)
    qp = nk.maxsim_pack(q, dtype="f32")
    r = repr(qp)
    assert "MaxSimPackedMatrix" in r
    assert "vector_count=4" in r
    assert "depth=32" in r


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_packed_size(capability):
    """Verify classmethod packed_size returns a positive integer."""
    keep_one_capability(capability)
    size = nk.MaxSimPackedMatrix.packed_size(8, 64, dtype="bf16")
    assert isinstance(size, int)
    assert size > 0

    size_f32 = nk.MaxSimPackedMatrix.packed_size(8, 64, dtype="f32")
    assert size_f32 > 0
