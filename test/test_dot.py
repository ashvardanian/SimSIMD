#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test inner/dot products: nk.inner, nk.dot, nk.vdot.

Covers dtypes: float64, float32, float16, bfloat16, e4m3, e5m2, e2m3, e3m2, int8, uint8.
Parametrized over: ndim from dense_dimensions, capability from possible_capabilities.

Precision notes:
    Integer dtypes use exact ±1 tolerance (discrete arithmetic with possible
    accumulator width differences). Floating-point dtypes use NK_ATOL/NK_RTOL
    (0.1/0.1). Sub-byte floats (bf16, e4m3, e5m2, e2m3, e3m2) carry wider
    quantization noise but are held to the same relative error bar.

Complex dot/vdot tested separately (different kernel API surface).
Overflow tests skipped pending issue #206.

Matches C++ suite: test_dot.cpp.
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
    make_random_buffer,
    numpy_available,
    dense_dimensions,
    possible_capabilities,
    randomized_repetitions_count,
    keep_one_capability,
    profile,
    NK_ATOL,
    NK_RTOL,
    _DECIMAL_PRECISION,
    NATIVE_COMPUTE_DTYPE,
    make_random,
    tolerances_for_dtype,
    hex_array,
    collect_errors,
    collect_warnings,
    create_stats,
    print_stats_report,
    LazyFormat,
    _seed_rng,
)
from test_spatial import baseline_euclidean, baseline_sqeuclidean, baseline_angular

_algebraic_dtypes = ["float32", "float64"]
_algebraic_ndims = [7, 97]

_stats = create_stats()
atexit.register(print_stats_report, _stats)

baseline_inner = np.inner if numpy_available else None


def precise_inner(a, b):
    """High-precision inner product via Python Decimal, exceeding f118 accuracy."""
    with decimal.localcontext() as ctx:
        ctx.prec = _DECIMAL_PRECISION
        D = decimal.Decimal
        da = [D.from_float(float(x)) for x in a]
        db = [D.from_float(float(x)) for x in b]
        return float(sum(x * y for x, y in zip(da, db)))


_KERNELS_DOT = {
    "inner": (baseline_inner, nk.inner, precise_inner),
}

_KERNELS_OVERFLOW = {
    "inner": (baseline_inner, nk.inner),
    "euclidean": (baseline_euclidean, nk.euclidean),
    "sqeuclidean": (baseline_sqeuclidean, nk.sqeuclidean),
    "angular": (baseline_angular, nk.angular),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
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
def test_inner(ndim, dtype, capability):
    """Inner product across all numeric dtypes.

    Integer dtypes use exact ±1 tolerance (discrete arithmetic).
    Sub-byte floats (bf16, e4m3, e5m2, e2m3, e3m2) carry wider quantization
    noise but are held to the same relative error bar.
    """
    a_raw, a_baseline = make_random((ndim,), dtype)
    b_raw, b_baseline = make_random((ndim,), dtype)
    atol, rtol = tolerances_for_dtype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = _KERNELS_DOT["inner"]

    # High-precision baseline (always f64)
    accurate_dt, accurate = profile(precise_kernel or baseline_kernel, a_baseline, b_baseline)

    # Baseline at native precision (for error stats)
    native_dt = NATIVE_COMPUTE_DTYPE.get(dtype, np.float64)
    expected_dt, expected = profile(baseline_kernel, a_baseline.astype(native_dt), b_baseline.astype(native_dt))

    # SIMD result — pass dtype for exotic types so the kernel knows the storage format
    result_dt, result = profile(simd_kernel, a_raw, b_raw, dtype)
    result = np.asarray(result)

    err_msg = LazyFormat(
        lambda: (
            f"\ninner({dtype}, ndim={ndim}):"
            f"\n  Accurate:  {accurate}"
            f"\n  Got:       {result}"
            f"\n  Raw a:     {hex_array(a_raw)}"
            f"\n  Raw b:     {hex_array(b_raw)}"
        )
    )

    np.testing.assert_allclose(result, accurate, atol=atol, rtol=rtol, err_msg=err_msg)
    collect_errors("inner", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, _stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dot_complex(ndim, dtype, capability):
    """Compares the nk.dot() and nk.vdot() against NumPy for complex numbers."""
    a = (np.random.randn(ndim) + 1.0j * np.random.randn(ndim)).astype(dtype)
    b = (np.random.randn(ndim) + 1.0j * np.random.randn(ndim)).astype(dtype)

    keep_one_capability(capability)
    accurate_dt, accurate = profile(np.dot, a.astype(np.complex128), b.astype(np.complex128))
    expected_dt, expected = profile(np.dot, a, b)
    result_dt, result = profile(nk.dot, a, b)
    result = np.asarray(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors("dot", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, _stats)

    accurate_dt, accurate = profile(np.vdot, a.astype(np.complex128), b.astype(np.complex128))
    expected_dt, expected = profile(np.vdot, a, b)
    result_dt, result = profile(nk.vdot, a, b)
    result = np.asarray(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors("vdot", ndim, dtype + "c", accurate, accurate_dt, expected, expected_dt, result, result_dt, _stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dot_complex_explicit(ndim, capability):
    """Compares the nk.dot() and nk.vdot() against NumPy for complex numbers."""
    a = np.random.randn(ndim * 2).astype(dtype=np.float32)
    b = np.random.randn(ndim * 2).astype(dtype=np.float32)

    keep_one_capability(capability)
    expected = np.dot(a.view(np.complex64), b.view(np.complex64))
    result = nk.dot(a, b, "complex64")

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    expected = np.vdot(a.view(np.complex64), b.view(np.complex64))
    result = nk.vdot(a, b, "complex64")

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skip(reason="Lacks overflow protection: https://github.com/ashvardanian/NumKong/issues/206")
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("metric", ["inner", "euclidean", "sqeuclidean", "angular"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_overflow(ndim, dtype, metric, capability):
    """Tests if the floating-point kernels are capable of detecting overflow yield the same ±inf result."""

    a = np.random.randn(ndim)
    b = np.random.randn(ndim)

    # Replace scalar at random position with infinity
    a[np.random.randint(ndim)] = np.inf
    a = a.astype(dtype)
    b = b.astype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = _KERNELS_OVERFLOW[metric]
    result = simd_kernel(a, b)
    assert np.isinf(result), f"Expected ±inf, but got {result}"

    #! In the Euclidean (L2) distance, SciPy raises a `ValueError` from the underlying
    #! NumPy function: `ValueError: array must not contain infs or NaNs`.
    try:
        expected_overflow = baseline_kernel(a, b)
        if not np.isinf(expected_overflow):
            collect_warnings("Overflow not detected in SciPy", _stats)
    except Exception as e:
        collect_warnings(f"Arbitrary error raised in SciPy: {e}", _stats)


@pytest.mark.parametrize("ndim", _algebraic_ndims)
@pytest.mark.parametrize("dtype", _algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_inner_known(ndim, dtype, capability):
    """inner(ones, ones) = n."""
    keep_one_capability(capability)
    o = nk.ones((ndim,), dtype=dtype)
    result = nk.inner(o, o)
    assert abs(result - ndim) < NK_ATOL + NK_RTOL * ndim


@pytest.mark.parametrize("ndim", _algebraic_ndims)
@pytest.mark.parametrize("dtype", _algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_inner_orthogonal(ndim, dtype, capability):
    """inner(ones, zeros) = 0."""
    keep_one_capability(capability)
    o = nk.ones((ndim,), dtype=dtype)
    z = nk.zeros((ndim,), dtype=dtype)
    result = nk.inner(o, z)
    assert abs(result) < NK_ATOL


@pytest.mark.parametrize("ndim", _algebraic_ndims)
@pytest.mark.parametrize("dtype", _algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_inner_symmetry(ndim, dtype, capability):
    """inner(a, b) = inner(b, a)."""
    keep_one_capability(capability)
    a = make_random_buffer(ndim, dtype)
    b = make_random_buffer(ndim, dtype)
    ab = nk.inner(a, b)
    ba = nk.inner(b, a)
    assert abs(ab - ba) < NK_ATOL, f"inner(a,b)={ab} != inner(b,a)={ba}"


@pytest.mark.parametrize("ndim", _algebraic_ndims)
@pytest.mark.parametrize("dtype", _algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_inner_cauchy_schwarz(ndim, dtype, capability):
    """|inner(a,b)|^2 <= inner(a,a) * inner(b,b)."""
    keep_one_capability(capability)
    a = make_random_buffer(ndim, dtype)
    b = make_random_buffer(ndim, dtype)
    ab = nk.inner(a, b)
    aa = nk.inner(a, a)
    bb = nk.inner(b, b)
    assert (
        ab * ab <= aa * bb + NK_ATOL
    ), f"Cauchy-Schwarz violated: |inner(a,b)|²={ab*ab} > inner(a,a)*inner(b,b)={aa*bb}"


@pytest.mark.skip(reason="Lacks overflow protection: https://github.com/ashvardanian/NumKong/issues/206")
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [131072, 262144])
@pytest.mark.parametrize("metric", ["inner", "euclidean", "sqeuclidean", "angular"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_overflow_i8(ndim, metric, capability):
    """Tests if the integral kernels are capable of detecting overflow yield the same ±inf result,
    as with 2^16 elements accumulating "u32(u16(u8)*u16(u8))+u32" products should overflow and the
    same is true for 2^17 elements with "i32(i15(i8))*i32(i15(i8))" products.
    """

    a = np.full(ndim, fill_value=-128, dtype=np.int8)
    b = np.full(ndim, fill_value=-128, dtype=np.int8)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = _KERNELS_OVERFLOW[metric]
    expected = baseline_kernel(a, b)
    result = simd_kernel(a, b)
    assert np.isinf(result), f"Expected ±inf, but got {result}"

    try:
        expected_overflow = baseline_kernel(a, b)
        if not np.isinf(expected_overflow):
            collect_warnings("Overflow not detected in SciPy", _stats)
    except Exception as e:
        collect_warnings(f"Arbitrary error raised in SciPy: {e}", _stats)
