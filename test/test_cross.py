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
    DECIMAL_PRECISION,
    NATIVE_COMPUTE_DTYPE,
    make_random,
    make_nk,
    tolerances_for_dtype,
    collect_errors,
    create_stats,
    print_stats_report,
    seed_rng,
)

try:
    import scipy.spatial.distance as spd
except ImportError:
    pass

stats = create_stats()
atexit.register(print_stats_report, stats)

baseline_dots_symmetric = lambda vectors: vectors @ vectors.T
baseline_dots_packed = lambda A, B: A @ B.T


def precise_matmul(A, B_T):
    """High-precision A @ B^T via Decimal. Returns 2D numpy array."""
    with decimal.localcontext() as ctx:
        ctx.prec = DECIMAL_PRECISION
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


KERNELS_CROSS = {
    "dots_symmetric": (baseline_dots_symmetric, nk.dots_symmetric, precise_dots_symmetric),
    "dots_packed": (baseline_dots_packed, nk.dots_packed, precise_dots_packed),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_batch_sqeuclidean_broadcasting(ndim, dtype, capability):
    """Batch sqeuclidean with NxD-vs-NxD, NxD-vs-1xD, strided, transposed, and out_dtype scenarios."""
    keep_one_capability(capability)

    # NxD vs NxD
    a_matrix = np.random.randn(10, ndim).astype(dtype)
    b_matrix = np.random.randn(10, ndim).astype(dtype)
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    np.testing.assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # NxD vs 1xD
    b_matrix = np.random.randn(1, ndim).astype(dtype)
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix[0]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    np.testing.assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # 1xD vs NxD
    a_matrix = np.random.randn(1, ndim).astype(dtype)
    b_matrix = np.random.randn(10, ndim).astype(dtype)
    expected_distances = [spd.sqeuclidean(a_matrix[0], b_matrix[i]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    np.testing.assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # NxD vs D (1D)
    a_matrix = np.random.randn(10, ndim).astype(dtype)
    b_matrix = np.random.randn(ndim).astype(dtype)
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    np.testing.assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # D (1D) vs NxD
    b_matrix = np.random.randn(10, ndim).astype(dtype)
    a_matrix = np.random.randn(ndim).astype(dtype)
    expected_distances = [spd.sqeuclidean(b_matrix[i], a_matrix) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(b_matrix, a_matrix)).astype(np.float64)
    np.testing.assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # Strided slices of bigger matrices
    a_matrix_extended = np.random.randn(10, ndim + 11).astype(dtype)
    b_matrix_extended = np.random.randn(10, ndim + 13).astype(dtype)
    a_matrix = a_matrix_extended[:, 1 : 1 + ndim]
    b_matrix = b_matrix_extended[:, 3 : 3 + ndim]
    assert a_matrix.base is a_matrix_extended and b_matrix.base is b_matrix_extended
    assert a_matrix.__array_interface__["strides"] is not None and b_matrix.__array_interface__["strides"] is not None
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    np.testing.assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # Transposed matrix
    a_matrix = np.random.randn(10, ndim).astype(dtype)
    b_matrix = np.ascontiguousarray(np.random.randn(ndim, 10).astype(dtype).T)
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    np.testing.assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # Different output type
    a_matrix = np.random.randn(10, ndim).astype(dtype)
    b_matrix = np.random.randn(10, ndim).astype(dtype)
    expected_distances = np.array([spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]).astype(np.float32)
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix, out_dtype="float32"))
    np.testing.assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)
    assert simd_distances.dtype == expected_distances.dtype

    # Supplied output buffer
    a_matrix = np.random.randn(10, ndim).astype(dtype)
    b_matrix = np.random.randn(10, ndim).astype(dtype)
    expected_distances = np.array([spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]).astype(np.float32)
    output_buffer = np.zeros(10, dtype=np.float32)
    assert nk.sqeuclidean(a_matrix, b_matrix, out=output_buffer) is None
    np.testing.assert_allclose(output_buffer, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)
    assert output_buffer.dtype == expected_distances.dtype


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["float32", "float16"])
@pytest.mark.parametrize("out_dtype", [None, "float32", "int32"])
@pytest.mark.parametrize("metric", ["angular", "sqeuclidean"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_float_accuracy(ndim, input_dtype, out_dtype, metric, capability):
    """Pairwise cdist for float dtypes with out_dtype and out= buffer support."""
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 10, 15
    a_matrix_extended = np.random.randn(num_rows_a, ndim + 1).astype(input_dtype)
    b_matrix_extended = np.random.randn(num_rows_b, ndim + 3).astype(input_dtype)
    a_matrix = a_matrix_extended[:, :ndim]
    b_matrix = b_matrix_extended[:, :ndim]

    is_integer_output = out_dtype in ("int32", "int64", "int16", "int8", "uint32", "uint64", "uint16", "uint8")
    scipy_metric = scipy_metric_name(metric)

    if out_dtype is None:
        expected = spd.cdist(a_matrix, b_matrix, scipy_metric)
        result = nk.cdist(a_matrix, b_matrix, metric)
        expected_out = np.zeros((num_rows_a, num_rows_b))
        output_buffer_extended = np.zeros((num_rows_a, num_rows_b + 7))
        output_buffer = output_buffer_extended[:, :num_rows_b]
        assert spd.cdist(a_matrix, b_matrix, scipy_metric, out=expected_out) is not None
        assert nk.cdist(a_matrix, b_matrix, metric, out=output_buffer) is None
    else:
        scipy_result = spd.cdist(a_matrix, b_matrix, scipy_metric)
        expected = np.round(scipy_result).astype(out_dtype) if is_integer_output else scipy_result.astype(out_dtype)
        result = nk.cdist(a_matrix, b_matrix, metric, out_dtype=out_dtype)

        expected_out = np.zeros((num_rows_a, num_rows_b), dtype=np.float64)
        output_buffer_extended = np.zeros((num_rows_a, num_rows_b + 7), dtype=out_dtype)
        output_buffer = output_buffer_extended[:, :num_rows_b]
        assert spd.cdist(a_matrix, b_matrix, scipy_metric, out=expected_out) is not None
        assert nk.cdist(a_matrix, b_matrix, metric, out=output_buffer) is None
        expected_out = np.round(expected_out).astype(out_dtype) if is_integer_output else expected_out.astype(out_dtype)

    atol = 1 if is_integer_output else NK_ATOL
    np.testing.assert_allclose(result, expected, atol=atol, rtol=NK_RTOL)
    np.testing.assert_allclose(output_buffer, expected_out, atol=atol, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["float32", "float16"])
@pytest.mark.parametrize("out_dtype", [None, "float32", "int32"])
@pytest.mark.parametrize("metric", ["angular", "sqeuclidean"])
def test_cdist_self_distance(ndim, input_dtype, out_dtype, metric):
    """cdist(A, A) self-distance matrix against SciPy baseline."""
    is_integer_output = out_dtype in ("int32", "int64", "int16", "int8", "uint32", "uint64", "uint16", "uint8")
    scipy_metric = scipy_metric_name(metric)

    a_matrix = np.random.randn(10, ndim + 1).astype(input_dtype)
    if out_dtype is None:
        expected = spd.cdist(a_matrix, a_matrix, scipy_metric)
        result = nk.cdist(a_matrix, a_matrix, metric=metric)
    else:
        scipy_result = spd.cdist(a_matrix, a_matrix, scipy_metric)
        expected = np.round(scipy_result).astype(out_dtype) if is_integer_output else scipy_result.astype(out_dtype)
        result = nk.cdist(a_matrix, a_matrix, metric=metric, out_dtype=out_dtype)

    atol = 1 if is_integer_output else NK_ATOL
    np.testing.assert_allclose(result, expected, atol=atol, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["complex128", "complex64"])
@pytest.mark.parametrize("out_dtype", [None, "complex128", "complex64"])
@pytest.mark.parametrize("metric", ["dot", "vdot"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_complex(ndim, input_dtype, out_dtype, metric, capability):
    """cdist for complex dot and vdot with 1D and 2D inputs plus out= support."""
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 10, 15
    a_matrix_extended = np.random.randn(num_rows_a, ndim + 1).astype(input_dtype)
    b_matrix_extended = np.random.randn(num_rows_b, ndim + 3).astype(input_dtype)
    a_matrix = a_matrix_extended[:, :ndim]
    b_matrix = b_matrix_extended[:, :ndim]
    c_matrix_extended = np.random.randn(num_rows_a, num_rows_b + 7).astype(out_dtype if out_dtype else np.complex128)
    c_matrix = c_matrix_extended[:, :num_rows_b]

    expected = np.zeros((num_rows_a, num_rows_b), dtype=out_dtype if out_dtype else np.complex128)
    baseline_kernel = np.dot if metric == "dot" else np.vdot
    for i in range(num_rows_a):
        for j in range(num_rows_b):
            expected[i, j] = baseline_kernel(a_matrix[i], b_matrix[j])

    if out_dtype is None:
        result1d = nk.cdist(a_matrix[0], b_matrix[0], metric=metric)
        result2d = nk.cdist(a_matrix, b_matrix, metric=metric)
        assert nk.cdist(a_matrix, b_matrix, metric=metric, out=c_matrix) is None
    else:
        expected = expected.astype(out_dtype)
        result1d = nk.cdist(a_matrix[0], b_matrix[0], metric=metric, out_dtype=out_dtype)
        result2d = nk.cdist(a_matrix, b_matrix, metric=metric, out_dtype=out_dtype)
        assert nk.cdist(a_matrix, b_matrix, metric=metric, out_dtype=out_dtype, out=c_matrix) is None

    np.testing.assert_allclose(result1d, expected[0, 0], atol=NK_ATOL, rtol=NK_RTOL)
    np.testing.assert_allclose(result2d, expected, atol=NK_ATOL, rtol=NK_RTOL)
    np.testing.assert_allclose(c_matrix, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("out_dtype", [None, "float32", "float16", "int8"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_hamming(ndim, out_dtype, capability):
    """cdist for packed Hamming bits with optional out_dtype."""
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 10, 15
    a_bits = np.random.randint(2, size=(num_rows_a, ndim)).astype(np.uint8)
    b_bits = np.random.randint(2, size=(num_rows_b, ndim)).astype(np.uint8)
    a_packed_bits, b_packed_bits = np.packbits(a_bits, axis=1), np.packbits(b_bits, axis=1)

    if out_dtype is None:
        expected = spd.cdist(a_bits, b_bits, "hamming") * ndim
        result = nk.cdist(a_packed_bits, b_packed_bits, metric="hamming", dtype="uint1")
    else:
        expected = (spd.cdist(a_bits, b_bits, "hamming") * ndim).astype(out_dtype)
        result = nk.cdist(a_packed_bits, b_packed_bits, metric="hamming", dtype="uint1", out_dtype=out_dtype)

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

    baseline_kernel, simd_kernel, precise_kernel = KERNELS_CROSS["dots_symmetric"]
    num_vectors, vector_depth = 32, 64
    atol, rtol = tolerances_for_dtype(dtype)
    vectors_raw, vectors_baseline = make_random((num_vectors, vector_depth), dtype)

    keep_one_capability(capability)

    accurate_dt, accurate = profile(precise_kernel or baseline_kernel, vectors_baseline)

    native_dt = NATIVE_COMPUTE_DTYPE.get(dtype, np.float64)
    expected_dt, expected = profile(baseline_kernel, vectors_baseline.astype(native_dt))

    result_dt, result = profile(simd_kernel, vectors_raw, dtype=dtype)
    result = np.asarray(result)

    mask = np.triu(np.ones((num_vectors, num_vectors), dtype=bool))
    np.testing.assert_allclose(result[mask], accurate[mask], atol=atol, rtol=rtol)
    collect_errors(
        "dots_symmetric",
        num_vectors * vector_depth,
        dtype,
        accurate[mask],
        accurate_dt,
        expected[mask],
        expected_dt,
        result[mask],
        result_dt,
        stats,
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_hammings_symmetric(capability):
    """Test nk.hammings_symmetric against pairwise Hamming (upper triangle)."""
    num_vectors, bit_depth = 16, 128
    bits = np.random.randint(2, size=(num_vectors, bit_depth)).astype(np.uint8)
    packed = np.packbits(bits, axis=1)

    keep_one_capability(capability)
    result = np.asarray(nk.hammings_symmetric(packed, dtype="uint1"))

    mask = np.triu(np.ones((num_vectors, num_vectors), dtype=bool))
    expected = np.zeros((num_vectors, num_vectors), dtype=np.float64)
    for i in range(num_vectors):
        for j in range(i, num_vectors):
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
def test_dots_pack_and_packed(dtype, capability):
    """Test dots_pack + dots_packed against high-precision matmul."""

    _, _, precise_kernel = KERNELS_CROSS["dots_packed"]
    height, width, depth = 8, 16, 64
    atol, rtol = tolerances_for_dtype(dtype)
    a_raw, a_baseline = make_random((height, depth), dtype)
    b_raw, b_baseline = make_random((width, depth), dtype)

    keep_one_capability(capability)

    # SIMD path — wrap in nk.Tensor so dots_packed can infer dtype
    a_tensor = make_nk(a_raw, dtype)
    b_tensor = make_nk(b_raw, dtype)
    b_packed = nk.dots_pack(b_tensor, dtype=dtype)
    result_dt, result = profile(nk.dots_packed, a_tensor, b_packed)
    result = np.asarray(result)

    accurate_dt, accurate = profile(precise_kernel, a_baseline, b_baseline)

    native_dt = NATIVE_COMPUTE_DTYPE.get(dtype, np.float64)
    expected_dt, expected = profile(baseline_dots_packed, a_baseline.astype(native_dt), b_baseline.astype(native_dt))

    np.testing.assert_allclose(result, accurate, atol=atol, rtol=rtol)
    collect_errors(
        "dots_packed", height * depth, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dots_pack_matmul_operator(capability):
    """Test the @ operator with a PackedMatrix (Tensor @ PackedMatrix)."""
    height, width, depth = 8, 16, 64
    a_matrix = np.random.randn(height, depth).astype(np.float32)
    b_matrix = np.random.randn(width, depth).astype(np.float32)

    keep_one_capability(capability)
    a_tensor = nk.zeros((height, depth), dtype="float32")
    a_tensor_view = np.asarray(a_tensor)
    np.copyto(a_tensor_view, a_matrix)

    b_packed = nk.dots_pack(b_matrix, dtype="float32")
    result = np.asarray(a_tensor @ b_packed)
    expected = a_matrix @ b_matrix.T

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_hammings_pack_and_packed(capability):
    """Test hammings_pack + hammings_packed against pairwise Hamming."""
    num_rows_a, num_rows_b, bit_depth = 8, 16, 128
    a_bits = np.random.randint(2, size=(num_rows_a, bit_depth)).astype(np.uint8)
    b_bits = np.random.randint(2, size=(num_rows_b, bit_depth)).astype(np.uint8)
    a_packed = np.packbits(a_bits, axis=1)
    b_packed_raw = np.packbits(b_bits, axis=1)

    keep_one_capability(capability)
    b_packed = nk.hammings_pack(b_packed_raw, dtype="uint1")
    result = np.asarray(nk.hammings_packed(a_packed, b_packed))

    expected = np.zeros((num_rows_a, num_rows_b), dtype=np.float64)
    for i in range(num_rows_a):
        for j in range(num_rows_b):
            expected[i, j] = np.logical_xor(a_bits[i], b_bits[j]).sum()

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
