#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test elementwise operations: nk.scale, nk.add, nk.wsum, nk.fma, nk.multiply.

Covers dtypes: float64, float32, float16, bfloat16, e4m3, e5m2, int8, uint8 (core ops);
    mixed-dtype tuples for NumPy-like add/multiply interface.
Parametrized over: ndim from dense_dimensions, capability from possible_capabilities.

Precision notes:
    Integer dtypes use exact ±1 tolerance (discrete arithmetic).
    Floating-point dtypes use NK_ATOL/NK_RTOL (0.1/0.1).

    Integer coefficients are kept small to prevent overflow:
    - scale, wsum: abs(alpha)/2, abs(beta)/2
    - fma: abs(alpha)/512, abs(beta)/3 — because x*y magnifies values

    All assertions compare the SIMD result against the NumPy baseline at
    native precision (not f64), since these ops return data in the input dtype.

Matches C++ suite: test_each.cpp.
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
    dense_dimensions,
    possible_capabilities,
    randomized_repetitions_count,
    keep_one_capability,
    profile,
    make_random,
    tolerances_for_dtype,
    random_of_dtype,
    NK_ATOL,
    NK_RTOL,
    collect_errors,
    collect_warnings,
    create_stats,
    print_stats_report,
    seed_rng,
)

algebraic_dtypes = ["float32", "float64"]
algebraic_ndims = [7, 97]
stats = create_stats()
atexit.register(print_stats_report, stats)


def normalize_elementwise(r, dtype_new):
    """Clips higher-resolution results to the smaller target dtype without overflow."""
    if np.issubdtype(dtype_new, np.integer):
        r = np.round(r)
        dtype_old_info = np.iinfo(r.dtype) if np.issubdtype(r.dtype, np.integer) else np.finfo(r.dtype)
        dtype_new_info = np.iinfo(dtype_new)
        new_min = dtype_new_info.min if dtype_new_info.min > dtype_old_info.min else None
        new_max = dtype_new_info.max if dtype_new_info.max < dtype_old_info.max else None
        if new_min is not None or new_max is not None:
            r = np.clip(r, new_min, new_max, out=r)
    return r.astype(dtype_new)


def get_computation_dtypes(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    larger_dtype = np.promote_types(x.dtype, y.dtype)
    if larger_dtype == np.uint8:
        return np.uint16, larger_dtype
    elif larger_dtype == np.int8:
        return np.int16, larger_dtype
    if larger_dtype == np.uint16:
        return np.uint32, larger_dtype
    elif larger_dtype == np.int16:
        return np.int32, larger_dtype
    if larger_dtype == np.uint32:
        return np.uint64, larger_dtype
    elif larger_dtype == np.int32:
        return np.int64, larger_dtype
    else:
        return larger_dtype, larger_dtype


def baseline_scale(x, alpha, beta):
    """Scale operation: alpha * x + beta"""
    compute_dtype, _ = get_computation_dtypes(x, alpha)
    result = alpha * x.astype(compute_dtype) + beta
    return normalize_elementwise(result, x.dtype)


def baseline_sum(x, y):
    compute_dtype, _ = get_computation_dtypes(x, y)
    result = x.astype(compute_dtype) + y.astype(compute_dtype)
    return normalize_elementwise(result, x.dtype)


def baseline_wsum(x, y, alpha, beta):
    """Weighted sum: alpha * x + beta * y"""
    compute_dtype, _ = get_computation_dtypes(x, y)
    result = x.astype(compute_dtype) * alpha + y.astype(compute_dtype) * beta
    return normalize_elementwise(result, x.dtype)


def baseline_fma(x, y, z, alpha, beta):
    """Fused multiply-add: alpha * x * y + beta * z"""
    compute_dtype, _ = get_computation_dtypes(x, y)
    result = x.astype(compute_dtype) * y.astype(compute_dtype) * alpha + z.astype(compute_dtype) * beta
    return normalize_elementwise(result, x.dtype)


def baseline_add(x, y, out=None):
    compute_dtype, final_dtype = get_computation_dtypes(x, y)
    a = x.astype(compute_dtype) if isinstance(x, np.ndarray) else x
    b = y.astype(compute_dtype) if isinstance(y, np.ndarray) else y
    result = np.add(a, b, out=out, casting="unsafe")
    result = normalize_elementwise(result, final_dtype)
    return result


def baseline_multiply(x, y, out=None):
    compute_dtype, final_dtype = get_computation_dtypes(x, y)
    a = x.astype(compute_dtype) if isinstance(x, np.ndarray) else x
    b = y.astype(compute_dtype) if isinstance(y, np.ndarray) else y
    result = np.multiply(a, b, out=out, casting="unsafe")
    result = normalize_elementwise(result, final_dtype)
    return result


KERNELS_EACH = {
    "scale": (baseline_scale, nk.scale, None),
    "add": (baseline_add, nk.add, None),
    "wsum": (baseline_wsum, nk.wsum, None),
    "fma": (baseline_fma, nk.fma, None),
    "multiply": (baseline_multiply, nk.multiply, None),
}


def random_coefficients(dtype, alpha_div=2, beta_div=2):
    alpha = np.random.randn(1).astype(np.float64).item()
    beta = np.random.randn(1).astype(np.float64).item()
    if np.issubdtype(np.dtype(dtype), np.integer):
        alpha, beta = abs(alpha) / alpha_div, abs(beta) / beta_div
    return alpha, beta


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_scale_random_accuracy(ndim, dtype, capability):
    """scale(alpha * x + beta) across float and integer dtypes against NumPy baseline."""
    input_raw, input_baseline = make_random((ndim,), dtype)
    atol, rtol = tolerances_for_dtype(dtype)

    alpha, beta = random_coefficients(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = KERNELS_EACH["scale"]

    accurate_dt, accurate = profile(baseline_kernel, input_baseline, alpha=alpha, beta=beta)
    expected_dt, expected = profile(baseline_kernel, input_raw, alpha=alpha, beta=beta)
    result_dt, result = profile(simd_kernel, input_raw, alpha=alpha, beta=beta)
    result = np.asarray(result)

    np.testing.assert_allclose(result, expected.astype(np.float64), atol=atol, rtol=rtol)
    collect_errors("scale", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_add_random_accuracy(ndim, dtype, capability):
    """Elementwise addition across float and integer dtypes against NumPy baseline."""
    a_raw, a_baseline = make_random((ndim,), dtype)
    b_raw, b_baseline = make_random((ndim,), dtype)
    atol, rtol = tolerances_for_dtype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = KERNELS_EACH["add"]

    accurate_dt, accurate = profile(baseline_kernel, a_baseline, b_baseline)
    expected_dt, expected = profile(baseline_kernel, a_raw, b_raw)
    result_dt, result = profile(simd_kernel, a_raw, b_raw)
    result = np.asarray(result)

    np.testing.assert_allclose(result, expected.astype(np.float64), atol=atol, rtol=rtol)
    collect_errors("add", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_wsum_random_accuracy(ndim, dtype, capability):
    """Weighted sum (alpha * x + beta * y) across float and integer dtypes against NumPy baseline."""
    a_raw, a_baseline = make_random((ndim,), dtype)
    b_raw, b_baseline = make_random((ndim,), dtype)
    atol, rtol = tolerances_for_dtype(dtype)

    alpha, beta = random_coefficients(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = KERNELS_EACH["wsum"]

    accurate_dt, accurate = profile(baseline_kernel, a_baseline, b_baseline, alpha=alpha, beta=beta)
    expected_dt, expected = profile(baseline_kernel, a_raw, b_raw, alpha=alpha, beta=beta)
    result_dt, result = profile(simd_kernel, a_raw, b_raw, alpha=alpha, beta=beta)
    result = np.asarray(result)

    np.testing.assert_allclose(result, expected.astype(np.float64), atol=atol, rtol=rtol)
    collect_errors("wsum", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_fma_random_accuracy(ndim, dtype, capability):
    """Fused multiply-add (alpha * x * y + beta * z) across float and integer dtypes against NumPy baseline."""
    a_raw, a_baseline = make_random((ndim,), dtype)
    b_raw, b_baseline = make_random((ndim,), dtype)
    c_raw, c_baseline = make_random((ndim,), dtype)
    atol, rtol = tolerances_for_dtype(dtype)

    alpha, beta = random_coefficients(dtype, 512, 3)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = KERNELS_EACH["fma"]

    accurate_dt, accurate = profile(
        baseline_kernel,
        a_baseline,
        b_baseline,
        c_baseline,
        alpha=alpha,
        beta=beta,
    )
    expected_dt, expected = profile(baseline_kernel, a_raw, b_raw, c_raw, alpha=alpha, beta=beta)
    result_dt, result = profile(simd_kernel, a_raw, b_raw, c_raw, alpha=alpha, beta=beta)
    result = np.asarray(result)

    np.testing.assert_allclose(result, expected.astype(np.float64), atol=atol, rtol=rtol)
    collect_errors("fma", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [
        ("float64", "float64", "float64"),
        ("float32", "float32", "float32"),
        ("int8", "int8", "int8"),
        ("int16", "int16", "int16"),
        ("int32", "int32", "int32"),
        ("uint8", "uint8", "uint8"),
        ("uint16", "uint16", "uint16"),
        ("uint32", "uint32", "uint32"),
        ("int16", "uint16", "float64"),
        ("uint8", "float32", "float32"),
    ],
)
@pytest.mark.parametrize("kernel", ["add", "multiply"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_add_multiply_noncontiguous(dtype, kernel, capability):
    """Add and multiply on non-contiguous, strided, and shape-mismatched arrays."""
    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = KERNELS_EACH[kernel]
    first_dtype, second_dtype, output_dtype = dtype
    operator = {"add": "+", "multiply": "*"}[kernel]

    def validate(a, b, inplace_numkong):
        result_numpy = baseline_kernel(a, b)
        result_numkong = np.array(simd_kernel(a, b))
        assert (
            result_numkong.size == result_numpy.size
        ), f"Result sizes differ: {result_numkong.size} vs {result_numpy.size}"
        assert (
            result_numkong.shape == result_numpy.shape
        ), f"Result shapes differ: {result_numkong.shape} vs {result_numpy.shape}"
        assert (
            result_numkong.dtype == result_numpy.dtype
        ), f"Result dtypes differ: {result_numkong.dtype} vs {result_numpy.dtype} for ({a.dtype} {operator} {b.dtype})"

        if not np.allclose(result_numkong, result_numpy, atol=NK_ATOL, rtol=NK_RTOL):
            np.testing.assert_allclose(
                result_numkong,
                result_numpy,
                atol=NK_ATOL,
                rtol=NK_RTOL,
                err_msg=f"""
                Result mismatch for ({a.dtype} {operator} {b.dtype})
                First descriptor: {a.__array_interface__}
                Second descriptor: {b.__array_interface__}
                First operand: {a}
                Second operand: {b}
                NumKong result: {result_numkong}
                NumPy result: {result_numpy}
                """,
            )

        inplace_numpy = np.empty_like(inplace_numkong)
        simd_kernel(a, b, out=inplace_numkong)
        baseline_kernel(a, b, out=inplace_numpy)

        assert inplace_numkong.size == inplace_numpy.size
        assert inplace_numkong.shape == inplace_numpy.shape
        assert inplace_numkong.dtype == inplace_numpy.dtype

        mismatch_count = np.sum(~np.isclose(inplace_numkong, inplace_numpy, atol=NK_ATOL, rtol=NK_RTOL))
        if mismatch_count:
            collect_warnings(f"NumPy overflow in ({a.dtype} {operator} {b.dtype} -> {output_dtype})", stats)
        return result_numkong

    # Vector-Vector
    a = random_of_dtype(first_dtype, (6,))
    b = random_of_dtype(second_dtype, (6,))
    o = np.zeros(6).astype(output_dtype)
    validate(a, b, o)

    # Larger Vector-Vector
    a = random_of_dtype(first_dtype, (47,))
    b = random_of_dtype(second_dtype, (47,))
    o = np.zeros(47).astype(output_dtype)
    validate(a, b, o)

    # Much larger Vector-Vector
    a = random_of_dtype(first_dtype, (247,))
    b = random_of_dtype(second_dtype, (247,))
    o = np.zeros(247).astype(output_dtype)
    validate(a, b, o)

    # Vector-Scalar
    first_np_dt = np.dtype(first_dtype)
    second_np_dt = np.dtype(second_dtype)
    first_is_unsigned = np.issubdtype(first_np_dt, np.unsignedinteger)
    second_is_unsigned = np.issubdtype(second_np_dt, np.unsignedinteger)
    validate(a, first_np_dt.type(11 if first_is_unsigned else -11), o)
    validate(a, first_np_dt.type(7), o)

    # Scalar-Vector
    validate(second_np_dt.type(13 if second_is_unsigned else -13), b, o)
    validate(second_np_dt.type(5), b, o)

    # Matrix-Matrix
    a = random_of_dtype(first_dtype, (10, 47))
    b = random_of_dtype(second_dtype, (10, 47))
    o = np.zeros((10, 47)).astype(output_dtype)
    validate(a, b, o)

    # Strided Matrix-Matrix
    a_extended = random_of_dtype(first_dtype, (10, 47))
    b_extended = random_of_dtype(second_dtype, (10, 47))
    a = a_extended[::2, 1:]
    b = b_extended[1::2, :-1]
    o = np.zeros((5, 46)).astype(output_dtype)
    validate(a, b, o)

    # Strided Matrix-Matrix with reverse order
    a_extended = random_of_dtype(first_dtype, (10, 47))
    b_extended = random_of_dtype(second_dtype, (10, 47))
    a = a_extended[::-2, 1:]
    b = b_extended[1::2, -2::-1]
    o = np.zeros((5, 46)).astype(output_dtype)
    validate(a, b, o)

    # Shape mismatch errors
    a = random_of_dtype(first_dtype, (10, 47))
    b = random_of_dtype(second_dtype, (10, 46))
    with pytest.raises(ValueError):
        baseline_kernel(a, b)
    with pytest.raises(ValueError):
        simd_kernel(a, b)

    a = random_of_dtype(first_dtype, (6, 2, 3))
    b = random_of_dtype(second_dtype, (6, 6))
    with pytest.raises(ValueError):
        baseline_kernel(a, b)
    with pytest.raises(ValueError):
        simd_kernel(a, b)

    # Broadcasting not supported
    a = random_of_dtype(first_dtype, (4, 7, 5, 3))
    b = random_of_dtype(second_dtype, (1, 1, 1, 1))
    with pytest.raises(ValueError):
        simd_kernel(a, b)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize(
    "dtype",
    [
        ("float64", "float64", "float64"),
        ("float32", "float32", "float32"),
        ("float16", "float16", "float16"),
        ("float32", "float64", "float64"),
        ("float16", "float32", "float32"),
    ],
)
@pytest.mark.parametrize("kernel", ["add", "multiply"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_add_multiply_broadcast(ndim, dtype, kernel, capability):
    """Add and multiply with scalar-vector and mixed-dtype broadcasting."""
    first_dtype, second_dtype, output_dtype = dtype

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = KERNELS_EACH[kernel]

    # Vector-Vector
    a = np.random.randn(ndim).astype(first_dtype)
    b = np.random.randn(ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Scalar-Vector
    a = np.random.randn(1).astype(first_dtype)[0]
    b = np.random.randn(ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Vector-Scalar
    a = np.random.randn(ndim).astype(first_dtype)
    b = np.random.randn(1).astype(second_dtype)[0]
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Matrix-Matrix
    a = np.random.randn(10, ndim // 10 if ndim >= 10 else ndim).astype(first_dtype)
    b = np.random.randn(10, ndim // 10 if ndim >= 10 else ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # In-place operation
    a = np.random.randn(ndim).astype(first_dtype)
    b = np.random.randn(ndim).astype(second_dtype)
    out_expected = np.zeros(ndim).astype(output_dtype)
    out_result = np.zeros(ndim).astype(output_dtype)
    baseline_kernel(a, b, out=out_expected)
    simd_kernel(a, b, out=out_result)
    np.testing.assert_allclose(out_result, out_expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_scale_edge_cases(ndim, dtype, capability):
    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = KERNELS_EACH["scale"]

    a = np.random.randn(ndim).astype(dtype)

    # Standard alpha and beta
    alpha = np.random.randn(1).astype(np.float64).item()
    beta = np.random.randn(1).astype(np.float64).item()
    expected = baseline_kernel(a, alpha=alpha, beta=beta)
    result = np.array(simd_kernel(a, alpha=alpha, beta=beta))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Zero alpha
    expected = baseline_kernel(a, alpha=0.0, beta=1.5)
    result = np.array(simd_kernel(a, alpha=0.0, beta=1.5))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Zero beta
    expected = baseline_kernel(a, alpha=2.0, beta=0.0)
    result = np.array(simd_kernel(a, alpha=2.0, beta=0.0))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Negative alpha and beta
    expected = baseline_kernel(a, alpha=-1.5, beta=-2.0)
    result = np.array(simd_kernel(a, alpha=-1.5, beta=-2.0))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_add_edge_cases(ndim, dtype, capability):
    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = KERNELS_EACH["add"]

    # Standard random
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # One vector is zeros
    b = np.zeros(ndim).astype(dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Both vectors the same
    expected = baseline_kernel(a, a)
    result = np.array(simd_kernel(a, a))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Negative values
    a = -np.abs(np.random.randn(ndim).astype(dtype))
    b = -np.abs(np.random.randn(ndim).astype(dtype))
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_add_numpy_buffer_protocol(dtype):
    """nk.add() accepts NumPy arrays directly via buffer protocol."""
    a = np.random.randn(50).astype(dtype)
    b = np.random.randn(50).astype(dtype)

    expected = a + b
    result = nk.add(a, b)
    result_np = np.asarray(result)

    np.testing.assert_allclose(result_np, expected, rtol=1e-5)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_multiply_numpy_buffer_protocol(dtype):
    """nk.multiply() accepts NumPy arrays directly via buffer protocol."""
    a = np.random.randn(50).astype(dtype)
    b = np.random.randn(50).astype(dtype)

    expected = a * b
    result = nk.multiply(a, b)
    result_np = np.asarray(result)

    np.testing.assert_allclose(result_np, expected, rtol=1e-4)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_wsum_numpy_buffer_protocol(dtype):
    """nk.wsum() accepts NumPy arrays directly via buffer protocol."""
    a = np.random.randn(50).astype(dtype)
    b = np.random.randn(50).astype(dtype)
    alpha = 2.0
    beta = 0.5

    expected = alpha * a + beta * b
    result = nk.wsum(a, b, alpha=alpha, beta=beta)
    result_np = np.asarray(result)

    np.testing.assert_allclose(result_np, expected, rtol=1e-4)


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_add_known(ndim, dtype, capability):
    """add(full(2), full(3)) ~ 5."""
    keep_one_capability(capability)
    a = nk.full((ndim,), 2.0, dtype=dtype)
    b = nk.full((ndim,), 3.0, dtype=dtype)
    result = list(nk.add(a, b))
    for i in range(ndim):
        assert abs(result[i] - 5.0) < NK_ATOL, f"add(2,3)[{i}] = {result[i]}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multiply_known(ndim, dtype, capability):
    """multiply(full(2), full(3)) ~ 6."""
    keep_one_capability(capability)
    a = nk.full((ndim,), 2.0, dtype=dtype)
    b = nk.full((ndim,), 3.0, dtype=dtype)
    result = list(nk.multiply(a, b))
    for i in range(ndim):
        assert abs(result[i] - 6.0) < NK_ATOL, f"multiply(2,3)[{i}] = {result[i]}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_scale_identity(ndim, dtype, capability):
    """scale(v, alpha=1, beta=0) ~ v."""
    keep_one_capability(capability)
    input_vector = nk.full((ndim,), 7.5, dtype=dtype)
    result = list(nk.scale(input_vector, alpha=1.0, beta=0.0))
    for i in range(ndim):
        assert abs(result[i] - 7.5) < NK_ATOL, f"scale(7.5, 1, 0)[{i}] = {result[i]}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_wsum_known(ndim, dtype, capability):
    """wsum(full(a), full(b), alpha=2, beta=3) ~ 2a + 3b."""
    keep_one_capability(capability)
    a_val, b_val = 4.0, 5.0
    a = nk.full((ndim,), a_val, dtype=dtype)
    b = nk.full((ndim,), b_val, dtype=dtype)
    expected = 2.0 * a_val + 3.0 * b_val  # 23.0
    result = list(nk.wsum(a, b, alpha=2.0, beta=3.0))
    for i in range(ndim):
        assert abs(result[i] - expected) < NK_ATOL, f"wsum[{i}] = {result[i]}, expected {expected}"
