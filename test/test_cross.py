#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test cross-distance operations: cdist, batch, dots_symmetric, dots_packed.

Covers cdist and batch operations for float and complex dtypes.
Symmetric and packed dot products tested for all numeric dtypes
(float64, float32, float16, bfloat16, e4m3, e5m2, e2m3, e3m2, int8, uint8).

Precision notes:
    Floating-point dtypes use NK_ATOL/NK_RTOL (0.1/0.1).
    Integer output dtypes (cdist out_dtype) use atol=1.

Matches C++ suite: test_cross_*.cpp.
"""

import atexit
import decimal
import pytest

try:
    import numpy as np
except:
    np = None

import numkong as nk
from test_base import (
    numpy_available,
    scipy_available,
    dense_dimensions,
    possible_capabilities,
    randomized_repetitions_count,
    keep_one_capability,
    profile,
    scipy_metric_name,
    NK_ATOL,
    NK_RTOL,
    _DECIMAL_PRECISION,
    EXOTIC_DTYPES,
    make_random,
    make_nk,
    tolerances_for_dtype,
    collect_errors,
    create_stats,
    print_stats_report,
    _seed_rng,
)

try:
    import scipy.spatial.distance as spd
except ImportError:
    pass

_stats = create_stats()
atexit.register(print_stats_report, _stats)

baseline_dots_symmetric = lambda vectors: vectors @ vectors.T
baseline_dots_packed = lambda A, B: A @ B.T


def precise_matmul(A, B_T):
    """High-precision A @ B^T via Decimal. Returns 2D numpy array."""
    with decimal.localcontext() as ctx:
        ctx.prec = _DECIMAL_PRECISION
        D = decimal.Decimal
        m, k = A.shape
        n = B_T.shape[0]
        result = np.empty((m, n), dtype=np.float64)
        for i in range(m):
            da = [D.from_float(float(x)) for x in A[i]]
            for j in range(n):
                db = [D.from_float(float(x)) for x in B_T[j]]
                result[i, j] = float(sum(x * y for x, y in zip(da, db)))
        return result


def precise_dots_symmetric(vectors):
    """High-precision vectors @ vectors.T via Decimal."""
    return precise_matmul(vectors, vectors)


def precise_dots_packed(A, B):
    """High-precision A @ B.T via Decimal."""
    return precise_matmul(A, B)


_KERNELS_CROSS = {
    "dots_symmetric": (baseline_dots_symmetric, nk.dots_symmetric, precise_dots_symmetric),
    "dots_packed": (baseline_dots_packed, nk.dots_packed, precise_dots_packed),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_batch(ndim, dtype, capability):
    keep_one_capability(capability)

    # NxD vs NxD
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.random.randn(10, ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[i], B[i]) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(A, B)).astype(np.float64)
    np.testing.assert_allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # NxD vs 1xD
    B = np.random.randn(1, ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[i], B[0]) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(A, B)).astype(np.float64)
    np.testing.assert_allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # 1xD vs NxD
    A = np.random.randn(1, ndim).astype(dtype)
    B = np.random.randn(10, ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[0], B[i]) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(A, B)).astype(np.float64)
    np.testing.assert_allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # NxD vs D (1D)
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.random.randn(ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[i], B) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(A, B)).astype(np.float64)
    np.testing.assert_allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # D (1D) vs NxD
    B = np.random.randn(10, ndim).astype(dtype)
    A = np.random.randn(ndim).astype(dtype)
    result_np = [spd.sqeuclidean(B[i], A) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(B, A)).astype(np.float64)
    np.testing.assert_allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # Strided slices of bigger matrices
    A_extended = np.random.randn(10, ndim + 11).astype(dtype)
    B_extended = np.random.randn(10, ndim + 13).astype(dtype)
    A = A_extended[:, 1 : 1 + ndim]
    B = B_extended[:, 3 : 3 + ndim]
    assert A.base is A_extended and B.base is B_extended
    assert A.__array_interface__["strides"] is not None and B.__array_interface__["strides"] is not None
    result_np = [spd.sqeuclidean(A[i], B[i]) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(A, B)).astype(np.float64)
    np.testing.assert_allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # Transposed matrix
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.ascontiguousarray(np.random.randn(ndim, 10).astype(dtype).T)
    result_np = [spd.sqeuclidean(A[i], B[i]) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(A, B)).astype(np.float64)
    np.testing.assert_allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # Different output type
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.random.randn(10, ndim).astype(dtype)
    result_np = np.array([spd.sqeuclidean(A[i], B[i]) for i in range(10)]).astype(np.float32)
    result_simd = np.array(nk.sqeuclidean(A, B, out_dtype="float32"))
    np.testing.assert_allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)
    assert result_simd.dtype == result_np.dtype

    # Supplied output buffer
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.random.randn(10, ndim).astype(dtype)
    result_np = np.array([spd.sqeuclidean(A[i], B[i]) for i in range(10)]).astype(np.float32)
    result_simd = np.zeros(10, dtype=np.float32)
    assert nk.sqeuclidean(A, B, out=result_simd) is None
    np.testing.assert_allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)
    assert result_simd.dtype == result_np.dtype


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["float32", "float16"])
@pytest.mark.parametrize("out_dtype", [None, "float32", "int32"])
@pytest.mark.parametrize("metric", ["angular", "sqeuclidean"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist(ndim, input_dtype, out_dtype, metric, capability):
    keep_one_capability(capability)

    M, N = 10, 15
    A_extended = np.random.randn(M, ndim + 1).astype(input_dtype)
    B_extended = np.random.randn(N, ndim + 3).astype(input_dtype)
    A = A_extended[:, :ndim]
    B = B_extended[:, :ndim]

    is_integer_output = out_dtype in ("int32", "int64", "int16", "int8", "uint32", "uint64", "uint16", "uint8")
    scipy_metric = scipy_metric_name(metric)

    if out_dtype is None:
        expected = spd.cdist(A, B, scipy_metric)
        result = nk.cdist(A, B, metric)
        expected_out = np.zeros((M, N))
        result_out_extended = np.zeros((M, N + 7))
        result_out = result_out_extended[:, :N]
        assert spd.cdist(A, B, scipy_metric, out=expected_out) is not None
        assert nk.cdist(A, B, metric, out=result_out) is None
    else:
        scipy_result = spd.cdist(A, B, scipy_metric)
        expected = np.round(scipy_result).astype(out_dtype) if is_integer_output else scipy_result.astype(out_dtype)
        result = nk.cdist(A, B, metric, out_dtype=out_dtype)

        expected_out = np.zeros((M, N), dtype=np.float64)
        result_out_extended = np.zeros((M, N + 7), dtype=out_dtype)
        result_out = result_out_extended[:, :N]
        assert spd.cdist(A, B, scipy_metric, out=expected_out) is not None
        assert nk.cdist(A, B, metric, out=result_out) is None
        expected_out = np.round(expected_out).astype(out_dtype) if is_integer_output else expected_out.astype(out_dtype)

    atol = 1 if is_integer_output else NK_ATOL
    np.testing.assert_allclose(result, expected, atol=atol, rtol=NK_RTOL)
    np.testing.assert_allclose(result_out, expected_out, atol=atol, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["float32", "float16"])
@pytest.mark.parametrize("out_dtype", [None, "float32", "int32"])
@pytest.mark.parametrize("metric", ["angular", "sqeuclidean"])
def test_cdist_itself(ndim, input_dtype, out_dtype, metric):
    is_integer_output = out_dtype in ("int32", "int64", "int16", "int8", "uint32", "uint64", "uint16", "uint8")
    scipy_metric = scipy_metric_name(metric)

    A = np.random.randn(10, ndim + 1).astype(input_dtype)
    if out_dtype is None:
        expected = spd.cdist(A, A, scipy_metric)
        result = nk.cdist(A, A, metric=metric)
    else:
        scipy_result = spd.cdist(A, A, scipy_metric)
        expected = np.round(scipy_result).astype(out_dtype) if is_integer_output else scipy_result.astype(out_dtype)
        result = nk.cdist(A, A, metric=metric, out_dtype=out_dtype)

    atol = 1 if is_integer_output else NK_ATOL
    np.testing.assert_allclose(result, expected, atol=atol, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["complex128", "complex64"])
@pytest.mark.parametrize("out_dtype", [None, "complex128", "complex64"])
@pytest.mark.parametrize("metric", ["dot", "vdot"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_complex(ndim, input_dtype, out_dtype, metric, capability):
    keep_one_capability(capability)

    M, N = 10, 15
    A_extended = np.random.randn(M, ndim + 1).astype(input_dtype)
    B_extended = np.random.randn(N, ndim + 3).astype(input_dtype)
    A = A_extended[:, :ndim]
    B = B_extended[:, :ndim]
    C_extended = np.random.randn(M, N + 7).astype(out_dtype if out_dtype else np.complex128)
    C = C_extended[:, :N]

    expected = np.zeros((M, N), dtype=out_dtype if out_dtype else np.complex128)
    baseline_kernel = np.dot if metric == "dot" else np.vdot
    for i in range(M):
        for j in range(N):
            expected[i, j] = baseline_kernel(A[i], B[j])

    if out_dtype is None:
        result1d = nk.cdist(A[0], B[0], metric=metric)
        result2d = nk.cdist(A, B, metric=metric)
        assert nk.cdist(A, B, metric=metric, out=C) is None
    else:
        expected = expected.astype(out_dtype)
        result1d = nk.cdist(A[0], B[0], metric=metric, out_dtype=out_dtype)
        result2d = nk.cdist(A, B, metric=metric, out_dtype=out_dtype)
        assert nk.cdist(A, B, metric=metric, out_dtype=out_dtype, out=C) is None

    np.testing.assert_allclose(result1d, expected[0, 0], atol=NK_ATOL, rtol=NK_RTOL)
    np.testing.assert_allclose(result2d, expected, atol=NK_ATOL, rtol=NK_RTOL)
    np.testing.assert_allclose(C, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("out_dtype", [None, "float32", "float16", "int8"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_hamming(ndim, out_dtype, capability):
    keep_one_capability(capability)

    M, N = 10, 15
    A = np.random.randint(2, size=(M, ndim)).astype(np.uint8)
    B = np.random.randint(2, size=(N, ndim)).astype(np.uint8)
    A_bits, B_bits = np.packbits(A, axis=1), np.packbits(B, axis=1)

    if out_dtype is None:
        expected = spd.cdist(A, B, "hamming") * ndim
        result = nk.cdist(A_bits, B_bits, metric="hamming", dtype="uint1")
    else:
        expected = (spd.cdist(A, B, "hamming") * ndim).astype(out_dtype)
        result = nk.cdist(A_bits, B_bits, metric="hamming", dtype="uint1", out_dtype=out_dtype)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [
        "float64",
        "float32",
        "float16",
        "bfloat16",
        "e4m3",
        "e5m2",
        "e2m3",
        "e3m2",
        "int8",
        "uint8",
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dots_symmetric(dtype, capability):
    """Test nk.dots_symmetric against high-precision matmul (upper triangle)."""

    baseline_kernel, simd_kernel, precise_kernel = _KERNELS_CROSS["dots_symmetric"]
    n, d = 32, 64
    atol, rtol = tolerances_for_dtype(dtype)
    vectors_raw, vectors_baseline = make_random((n, d), dtype)

    keep_one_capability(capability)

    accurate_dt, accurate = profile(precise_kernel or baseline_kernel, vectors_baseline)

    result_dt, result = profile(simd_kernel, vectors_raw, dtype=dtype)
    result = np.asarray(result)

    mask = np.triu(np.ones((n, n), dtype=bool))
    np.testing.assert_allclose(result[mask], accurate[mask], atol=atol, rtol=rtol)
    collect_errors(
        "dots_symmetric",
        n * d,
        dtype,
        accurate[mask],
        accurate_dt,
        accurate[mask],
        accurate_dt,
        result[mask],
        result_dt,
        _stats,
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_hammings_symmetric(capability):
    """Test nk.hammings_symmetric against pairwise Hamming (upper triangle)."""
    n, d_bits = 16, 128
    bits = np.random.randint(2, size=(n, d_bits)).astype(np.uint8)
    packed = np.packbits(bits, axis=1)

    keep_one_capability(capability)
    result = np.asarray(nk.hammings_symmetric(packed, dtype="uint1"))

    mask = np.triu(np.ones((n, n), dtype=bool))
    expected = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            expected[i, j] = np.logical_xor(bits[i], bits[j]).sum()

    np.testing.assert_allclose(result[mask], expected[mask], atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [
        "float64",
        "float32",
        "float16",
        "bfloat16",
        "e4m3",
        "e5m2",
        "e2m3",
        "e3m2",
        "int8",
        "uint8",
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dots_packed(dtype, capability):
    """Test dots_pack + dots_packed against high-precision matmul."""

    _, _, precise_kernel = _KERNELS_CROSS["dots_packed"]
    m, n, k = 8, 16, 64
    atol, rtol = tolerances_for_dtype(dtype)
    A_raw, A_baseline = make_random((m, k), dtype)
    B_raw, B_baseline = make_random((n, k), dtype)

    keep_one_capability(capability)

    # SIMD path — exotic types need nk.Tensor wrappers because
    # dots_packed infers dtype from the tensor, not a kwarg.
    if dtype in EXOTIC_DTYPES:
        nk_A = make_nk(A_raw, dtype)
        nk_B = make_nk(B_raw, dtype)
        packed_B = nk.dots_pack(nk_B)
        result_dt, result = profile(nk.dots_packed, nk_A, packed_B)
        result = np.asarray(result)
    else:
        packed_B = nk.dots_pack(B_raw, dtype=dtype)
        result_dt, result = profile(nk.dots_packed, A_raw, packed_B)
        result = np.asarray(result)

    accurate_dt, accurate = profile(precise_kernel, A_baseline, B_baseline)

    np.testing.assert_allclose(result, accurate, atol=atol, rtol=rtol)
    collect_errors("dots_packed", m * k, dtype, accurate, accurate_dt, accurate, accurate_dt, result, result_dt, _stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dots_pack_matmul_operator(capability):
    """Test the @ operator with a PackedMatrix (Tensor @ PackedMatrix)."""
    m, n, k = 8, 16, 64
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(n, k).astype(np.float32)

    keep_one_capability(capability)
    nk_A = nk.zeros((m, k), dtype="float32")
    nk_A_np = np.asarray(nk_A)
    np.copyto(nk_A_np, A)

    packed_B = nk.dots_pack(B, dtype="float32")
    result = np.asarray(nk_A @ packed_B)
    expected = A @ B.T

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_hammings_pack_and_packed(capability):
    """Test hammings_pack + hammings_packed against pairwise Hamming."""
    m, n, d_bits = 8, 16, 128
    bits_A = np.random.randint(2, size=(m, d_bits)).astype(np.uint8)
    bits_B = np.random.randint(2, size=(n, d_bits)).astype(np.uint8)
    packed_A = np.packbits(bits_A, axis=1)
    packed_B_raw = np.packbits(bits_B, axis=1)

    keep_one_capability(capability)
    packed_B = nk.hammings_pack(packed_B_raw, dtype="uint1")
    result = np.asarray(nk.hammings_packed(packed_A, packed_B))

    expected = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            expected[i, j] = np.logical_xor(bits_A[i], bits_B[j]).sum()

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
