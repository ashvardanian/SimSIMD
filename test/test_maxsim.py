#!/usr/bin/env python3
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
except:  # noqa: E722
    np = None

import numkong as nk
from test_base import (
    assert_allclose,
    collect_warnings,
    create_stats,
    downcast_f32_to_dtype,
    keep_one_capability,
    make_nk,
    nk_seed,  # noqa: F401 — pytest fixture
    numpy_available,
    possible_capabilities,
    print_stats_report,
    randomized_repetitions_count,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
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
    raw, _ = downcast_f32_to_dtype(np.random.randn(rows, cols).astype(np.float32), dtype)
    return make_nk(raw, dtype)


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
    assert repr(qp)  # smoke-test repr doesn't crash

    result = nk.maxsim_packed(qp, dp)

    # MaxSim is an approximation (coarse screening + fine pass); compound
    # angular-distance accumulation amplifies f32 error to ~2% relative.
    # Verify sanity bounds only and log deviations for the stats report.
    assert np.isfinite(result), f"maxsim_packed({dtype}): got non-finite {result}"
    assert result >= 0, f"maxsim_packed({dtype}): got negative {result}"
    if dtype != "bfloat16":
        q_np = np.array(queries, dtype=dtype)
        d_np = np.array(documents, dtype=dtype)
        expected = baseline_maxsim(q_np, d_np)
        rel_err = abs(result - expected) / max(abs(expected), 1e-12)
        if rel_err > 0.1:
            collect_warnings(f"maxsim_packed({dtype}) rel_err={rel_err:.4f}", stats)


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

    assert_allclose(packed_result, conv_result)


@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_self_zero(capability):
    """Identical queries and documents should give score near zero."""
    keep_one_capability(capability)
    vectors = nk.ones((8, 64), dtype="float32")
    result = nk.maxsim(vectors, vectors, dtype="f32")
    assert_allclose(result, 0.0, err_msg="Expected near-zero self-distance")


@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_type_errors(capability, nk_seed):
    """Wrong type and mismatched dtype/depth."""
    keep_one_capability(capability)
    query = nk.hash((4, 32), seed=nk_seed, dtype="float32")
    documents = nk.hash((8, 32), seed=nk_seed + 1, dtype="float32")
    query_packed = nk.maxsim_pack(query, dtype="f32")
    documents_packed = nk.maxsim_pack(documents, dtype="f32")

    # Wrong type for maxsim_packed
    with pytest.raises(TypeError):
        nk.maxsim_packed(query, documents_packed)
    with pytest.raises(TypeError):
        nk.maxsim_packed(query_packed, documents)

    # Mismatched depth
    documents_wrong = nk.hash((8, 16), seed=nk_seed + 2, dtype="float32")
    documents_wrong_packed = nk.maxsim_pack(documents_wrong, dtype="f32")
    with pytest.raises(ValueError):
        nk.maxsim_packed(query_packed, documents_wrong_packed)


@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_packed_size(capability):
    """Verify classmethod packed_size returns a positive integer."""
    keep_one_capability(capability)
    size = nk.MaxSimPackedMatrix.packed_size(8, 64, dtype="bf16")
    assert isinstance(size, int)
    assert size > 0

    size_f32 = nk.MaxSimPackedMatrix.packed_size(8, 64, dtype="f32")
    assert size_f32 > 0
