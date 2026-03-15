#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test elementwise operations: nk.scale, nk.add, nk.blend, nk.fma, nk.multiply.

Covers dtypes: float64, float32, float16, int8, uint8 (core ops);
    mixed-dtype tuples for NumPy-like add/multiply interface.
Parametrized over: ndim from dense_dimensions, capability from possible_capabilities.

Precision notes:
    Integer dtypes use exact ±1 tolerance (discrete arithmetic).
    Floating-point dtypes use NK_ATOL/NK_RTOL (0.1/0.1).

    Integer coefficients are kept small to prevent overflow:
    - scale, blend: abs(alpha)/2, abs(beta)/2
    - fma: abs(alpha)/512, abs(beta)/3 — because x*y magnifies values

    All assertions compare the SIMD result against the NumPy baseline at
    native precision (not f64), since these ops return data in the input dtype.

Matches C++ suite: test_each.cpp.
"""

import atexit
import decimal
import random
import pytest

try:
    import numpy as np
except:  # noqa: E722
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
    assert_allclose,
    nk_seed,  # noqa: F401 — pytest fixture
    NK_ATOL,
    NK_RTOL,
    collect_errors,
    collect_warnings,
    create_stats,
    print_stats_report,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
)

algebraic_dtypes = ["float32", "float64"]
algebraic_ndims = [7, 97]
stats = create_stats()
atexit.register(print_stats_report, stats)


def normalize_elementwise(r, dtype_new):
    """Clips higher-resolution results to the smaller target dtype without overflow."""
    if np.issubdtype(dtype_new, np.integer):
        dtype_new_info = np.iinfo(dtype_new)
        r = np.nan_to_num(r, nan=0.0, posinf=dtype_new_info.max, neginf=dtype_new_info.min)
        r = np.clip(r, dtype_new_info.min, dtype_new_info.max, out=r)
        r = np.rint(r)
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


def baseline_blend(x, y, alpha, beta):
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


_INT_CLIP_RANGES = {
    "int8": (-128, 127), "uint8": (0, 255),
    "int16": (-32768, 32767), "uint16": (0, 65535),
    "int32": (-2147483648, 2147483647), "uint32": (0, 4294967295),
}


def _clip_int(values, dtype):
    """Clip and round values to integer dtype range, mirroring normalize_elementwise."""
    clip_range = _INT_CLIP_RANGES.get(dtype)
    if clip_range is None:
        return values
    lo, hi = clip_range
    return [float(max(lo, min(hi, round(v)))) for v in values]


def precise_scale(a, alpha, beta, dtype=None):
    """High-precision scale: alpha * x + beta via Decimal."""
    with decimal.localcontext() as ctx:
        ctx.prec = 120
        D = decimal.Decimal
        da = D.from_float(float(alpha))
        db = D.from_float(float(beta))
        result = [float(da * D.from_float(float(x)) + db) for x in a]
    return _clip_int(result, dtype) if dtype else result


def precise_add(a, b, dtype=None):
    """High-precision elementwise add via Decimal."""
    with decimal.localcontext() as ctx:
        ctx.prec = 120
        D = decimal.Decimal
        result = [float(D.from_float(float(x)) + D.from_float(float(y))) for x, y in zip(a, b)]
    return _clip_int(result, dtype) if dtype else result


def precise_blend(a, b, alpha, beta, dtype=None):
    """High-precision blend: alpha * x + beta * y via Decimal."""
    with decimal.localcontext() as ctx:
        ctx.prec = 120
        D = decimal.Decimal
        da, db = D.from_float(float(alpha)), D.from_float(float(beta))
        result = [float(da * D.from_float(float(x)) + db * D.from_float(float(y))) for x, y in zip(a, b)]
    return _clip_int(result, dtype) if dtype else result


def precise_fma(a, b, c, alpha, beta, dtype=None):
    """High-precision FMA: alpha * x * y + beta * z via Decimal."""
    with decimal.localcontext() as ctx:
        ctx.prec = 120
        D = decimal.Decimal
        da, db = D.from_float(float(alpha)), D.from_float(float(beta))
        result = [float(da * D.from_float(float(x)) * D.from_float(float(y)) + db * D.from_float(float(z)))
                for x, y, z in zip(a, b, c)]
    return _clip_int(result, dtype) if dtype else result


def precise_multiply(a, b, dtype=None):
    """High-precision elementwise multiply via Decimal."""
    with decimal.localcontext() as ctx:
        ctx.prec = 120
        D = decimal.Decimal
        result = [float(D.from_float(float(x)) * D.from_float(float(y))) for x, y in zip(a, b)]
    return _clip_int(result, dtype) if dtype else result


KERNELS_EACH = {
    "scale": (baseline_scale if numpy_available else None, nk.scale, precise_scale),
    "add": (baseline_add if numpy_available else None, nk.add, precise_add),
    "blend": (baseline_blend if numpy_available else None, nk.blend, precise_blend),
    "fma": (baseline_fma if numpy_available else None, nk.fma, precise_fma),
    "multiply": (baseline_multiply if numpy_available else None, nk.multiply, precise_multiply),
}


def random_coefficients(dtype, alpha_div=2, beta_div=2):
    if numpy_available:
        alpha = np.random.randn(1).astype(np.float64).item()
        beta = np.random.randn(1).astype(np.float64).item()
        if np.issubdtype(np.dtype(dtype), np.integer):
            alpha, beta = abs(alpha) / alpha_div, abs(beta) / beta_div
    else:
        alpha = random.uniform(-2.0, 2.0)
        beta = random.uniform(-2.0, 2.0)
        if dtype in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"):
            alpha, beta = abs(alpha) / alpha_div, abs(beta) / beta_div
    return alpha, beta


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_scale_random_accuracy(ndim, dtype, capability, nk_seed):
    """scale(alpha * x + beta) across float and integer dtypes against high-precision Decimal baseline."""
    input_raw, input_baseline = make_random((ndim,), dtype, seed=nk_seed)
    atol, rtol = tolerances_for_dtype(dtype)

    alpha, beta = random_coefficients(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_EACH["scale"]

    # High-precision baseline
    accurate_dt, accurate = profile(precise_kernel, input_baseline, alpha=alpha, beta=beta, dtype=dtype)

    # Native precision baseline
    expected_dt, expected = profile(baseline_kernel, input_raw, alpha=alpha, beta=beta)

    result_dt, result = profile(simd_kernel, input_raw, alpha=alpha, beta=beta)

    assert_allclose(result, accurate, atol=atol, rtol=rtol)
    collect_errors("scale", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_add_random_accuracy(ndim, dtype, capability, nk_seed):
    """Elementwise addition across float and integer dtypes against high-precision Decimal baseline."""
    a_raw, a_baseline = make_random((ndim,), dtype, seed=nk_seed)
    b_raw, b_baseline = make_random((ndim,), dtype, seed=nk_seed + 1)
    atol, rtol = tolerances_for_dtype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_EACH["add"]

    # High-precision baseline
    accurate_dt, accurate = profile(precise_kernel, a_baseline, b_baseline, dtype=dtype)

    # Native precision baseline
    expected_dt, expected = profile(baseline_kernel, a_raw, b_raw)

    result_dt, result = profile(simd_kernel, a_raw, b_raw)

    assert_allclose(result, accurate, atol=atol, rtol=rtol)
    collect_errors("add", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_blend_random_accuracy(ndim, dtype, capability, nk_seed):
    """Weighted sum (alpha * x + beta * y) across float and integer dtypes against high-precision Decimal baseline."""
    a_raw, a_baseline = make_random((ndim,), dtype, seed=nk_seed)
    b_raw, b_baseline = make_random((ndim,), dtype, seed=nk_seed + 1)
    atol, rtol = tolerances_for_dtype(dtype)

    alpha, beta = random_coefficients(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_EACH["blend"]

    # High-precision baseline
    accurate_dt, accurate = profile(precise_kernel, a_baseline, b_baseline, alpha=alpha, beta=beta, dtype=dtype)

    # Native precision baseline
    expected_dt, expected = profile(baseline_kernel, a_raw, b_raw, alpha=alpha, beta=beta)

    result_dt, result = profile(simd_kernel, a_raw, b_raw, alpha=alpha, beta=beta)

    assert_allclose(result, accurate, atol=atol, rtol=rtol)
    collect_errors("blend", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_fma_random_accuracy(ndim, dtype, capability, nk_seed):
    """Fused multiply-add (alpha * x * y + beta * z) across float and integer dtypes against high-precision Decimal baseline."""
    a_raw, a_baseline = make_random((ndim,), dtype, seed=nk_seed)
    b_raw, b_baseline = make_random((ndim,), dtype, seed=nk_seed + 1)
    c_raw, c_baseline = make_random((ndim,), dtype, seed=nk_seed + 2)
    atol, rtol = tolerances_for_dtype(dtype)

    alpha, beta = random_coefficients(dtype, 512, 3)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_EACH["fma"]

    # High-precision baseline
    accurate_dt, accurate = profile(
        precise_kernel,
        a_baseline,
        b_baseline,
        c_baseline,
        alpha=alpha,
        beta=beta,
        dtype=dtype,
    )

    # Native precision baseline
    expected_dt, expected = profile(baseline_kernel, a_raw, b_raw, c_raw, alpha=alpha, beta=beta)

    result_dt, result = profile(simd_kernel, a_raw, b_raw, c_raw, alpha=alpha, beta=beta)

    assert_allclose(result, accurate, atol=atol, rtol=rtol)
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
            assert_allclose(
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
    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Scalar-Vector
    a = np.random.randn(1).astype(first_dtype)[0]
    b = np.random.randn(ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Vector-Scalar
    a = np.random.randn(ndim).astype(first_dtype)
    b = np.random.randn(1).astype(second_dtype)[0]
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Matrix-Matrix
    a = np.random.randn(10, ndim // 10 if ndim >= 10 else ndim).astype(first_dtype)
    b = np.random.randn(10, ndim // 10 if ndim >= 10 else ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # In-place operation
    a = np.random.randn(ndim).astype(first_dtype)
    b = np.random.randn(ndim).astype(second_dtype)
    out_expected = np.zeros(ndim).astype(output_dtype)
    out_result = np.zeros(ndim).astype(output_dtype)
    baseline_kernel(a, b, out=out_expected)
    simd_kernel(a, b, out=out_result)
    assert_allclose(out_result, out_expected, atol=NK_ATOL, rtol=NK_RTOL)


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
    assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Zero alpha
    expected = baseline_kernel(a, alpha=0.0, beta=1.5)
    result = np.array(simd_kernel(a, alpha=0.0, beta=1.5))
    assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Zero beta
    expected = baseline_kernel(a, alpha=2.0, beta=0.0)
    result = np.array(simd_kernel(a, alpha=2.0, beta=0.0))
    assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Negative alpha and beta
    expected = baseline_kernel(a, alpha=-1.5, beta=-2.0)
    result = np.array(simd_kernel(a, alpha=-1.5, beta=-2.0))
    assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)


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
    assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # One vector is zeros
    b = np.zeros(ndim).astype(dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Both vectors the same
    expected = baseline_kernel(a, a)
    result = np.array(simd_kernel(a, a))
    assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Negative values
    a = -np.abs(np.random.randn(ndim).astype(dtype))
    b = -np.abs(np.random.randn(ndim).astype(dtype))
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_add_numpy_buffer_protocol(dtype):
    """nk.add() accepts NumPy arrays directly via buffer protocol."""
    a = np.random.randn(50).astype(dtype)
    b = np.random.randn(50).astype(dtype)

    expected = a + b
    result = nk.add(a, b)
    result_np = np.asarray(result)

    assert_allclose(result_np, expected, rtol=1e-5)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_multiply_numpy_buffer_protocol(dtype):
    """nk.multiply() accepts NumPy arrays directly via buffer protocol."""
    a = np.random.randn(50).astype(dtype)
    b = np.random.randn(50).astype(dtype)

    expected = a * b
    result = nk.multiply(a, b)
    result_np = np.asarray(result)

    assert_allclose(result_np, expected, rtol=1e-4)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_blend_numpy_buffer_protocol(dtype):
    """nk.blend() accepts NumPy arrays directly via buffer protocol."""
    a = np.random.randn(50).astype(dtype)
    b = np.random.randn(50).astype(dtype)
    alpha = 2.0
    beta = 0.5

    expected = alpha * a + beta * b
    result = nk.blend(a, b, alpha=alpha, beta=beta)
    result_np = np.asarray(result)

    assert_allclose(result_np, expected, rtol=1e-4)


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
def test_blend_known(ndim, dtype, capability):
    """blend(full(a), full(b), alpha=2, beta=3) ~ 2a + 3b."""
    keep_one_capability(capability)
    a_val, b_val = 4.0, 5.0
    a = nk.full((ndim,), a_val, dtype=dtype)
    b = nk.full((ndim,), b_val, dtype=dtype)
    expected = 2.0 * a_val + 3.0 * b_val  # 23.0
    result = list(nk.blend(a, b, alpha=2.0, beta=3.0))
    for i in range(ndim):
        assert abs(result[i] - expected) < NK_ATOL, f"blend[{i}] = {result[i]}, expected {expected}"
