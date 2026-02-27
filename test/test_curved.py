#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test curved-space distances: nk.bilinear, nk.mahalanobis.

Covers dtypes: float64, float32, float16 (with float32 compute), bfloat16,
    e4m3, e5m2, e2m3, e3m2.
Parametrized over: ndim from curved_dimensions, metric, capability.

Precision notes:
    bilinear computes ``x @ z @ y``, mahalanobis computes ``sqrt((x-y) @ z @ (x-y))``.
    Both take vectors a, b and a matrix c; the test constructs c as a symmetric
    positive-definite matrix (c = abs(random) @ abs(random).T) so mahalanobis
    produces real, finite results.

    All dtypes assert against an f64-precision accurate baseline.
    Complex bilinear tested separately (different operand construction).

Matches C++ suite: test_curved.cpp.
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
    curved_dimensions,
    possible_capabilities,
    randomized_repetitions_count,
    keep_one_capability,
    profile,
    NK_ATOL,
    NK_RTOL,
    _DECIMAL_PRECISION,
    EXOTIC_DTYPES,
    make_random,
    f32_downcast_to_bf16,
    hex_array,
    collect_errors,
    create_stats,
    print_stats_report,
    LazyFormat,
    _seed_rng,
)

_stats = create_stats()
atexit.register(print_stats_report, _stats)

baseline_bilinear = lambda x, y, z: x @ z @ y

try:
    import scipy.spatial.distance as spd

    def baseline_mahalanobis(x, y, z):
        try:
            result = spd.mahalanobis(x, y, z).astype(np.float64)
            if not np.isnan(result):
                return result
        except:
            pass
        pytest.skip(f"SciPy Mahalanobis distance returned {result} due to `sqrt` of a negative number")

except ImportError:

    def baseline_mahalanobis(x, y, z):
        diff = x - y
        return np.sqrt(diff @ z @ diff)


def make_psd(data: np.ndarray) -> np.ndarray:
    """Make a square matrix positive semi-definite in-place (Gershgorin diagonal dominance).

    Matches C++ make_psd() in test/test_curved.cpp.
    """
    n = data.shape[0]
    # Step 1: Symmetrize
    data = (data + data.T) * 0.5
    # Step 2: Diagonal dominance — each diagonal > absolute row sum of off-diagonals
    for i in range(n):
        row_sum = np.sum(np.abs(data[i])) - np.abs(data[i, i])
        data[i, i] = row_sum + 1.0
    return data


def precise_bilinear(a, b, c):
    """High-precision bilinear form x @ z @ y via Python Decimal."""
    with decimal.localcontext() as ctx:
        ctx.prec = _DECIMAL_PRECISION
        D = decimal.Decimal
        da = [D.from_float(float(x)) for x in a]
        db = [D.from_float(float(x)) for x in b]
        n = len(a)
        total = D(0)
        for i in range(n):
            for j in range(n):
                total += da[i] * D.from_float(float(c[i, j])) * db[j]
        return float(total)


def precise_mahalanobis(a, b, c):
    """High-precision Mahalanobis distance sqrt((a-b) @ c @ (a-b)) via Python Decimal."""
    with decimal.localcontext() as ctx:
        ctx.prec = _DECIMAL_PRECISION
        D = decimal.Decimal
        diff = [D.from_float(float(x)) - D.from_float(float(y)) for x, y in zip(a, b)]
        n = len(diff)
        total = D(0)
        for i in range(n):
            for j in range(n):
                total += diff[i] * D.from_float(float(c[i, j])) * diff[j]
        return float(total.sqrt())


_KERNELS_CURVED = {
    "bilinear": (baseline_bilinear, nk.bilinear, precise_bilinear),
    "mahalanobis": (baseline_mahalanobis, nk.mahalanobis, precise_mahalanobis),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", curved_dimensions)
@pytest.mark.parametrize(
    "dtypes",
    [
        ("float64", "float64"),
        ("float32", "float32"),
        ("float16", "float32"),
        ("bfloat16", "float32"),
    ],
)
@pytest.mark.parametrize("metric", ["bilinear", "mahalanobis"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_curved(ndim, dtypes, metric, capability):
    """Bilinear / Mahalanobis across float and exotic dtypes.

    Data is generated at f32 precision so normalisation and positive-definite
    construction are numerically stable, then cast to the target dtype.
    For bfloat16, ``f32_downcast_to_bf16`` rounds the f32 values and
    produces raw uint16 bytes for the SIMD kernel.
    """
    dtype, compute_dtype = dtypes

    # Generate structured data at f32
    a_f32 = np.abs(np.random.randn(ndim).astype(np.float32))
    b_f32 = np.abs(np.random.randn(ndim).astype(np.float32))
    a_f32 /= np.sum(a_f32)
    b_f32 /= np.sum(b_f32)
    c_f32 = make_psd(np.random.randn(ndim, ndim).astype(np.float32))

    if dtype in EXOTIC_DTYPES:
        a_f32r, a_raw = f32_downcast_to_bf16(a_f32)
        b_f32r, b_raw = f32_downcast_to_bf16(b_f32)
        c_f32r, c_raw = f32_downcast_to_bf16(c_f32)
        a_base, b_base, c_base = a_f32r.astype(np.float64), b_f32r.astype(np.float64), c_f32r.astype(np.float64)
    else:
        a_raw = a_f32.astype(dtype)
        b_raw = b_f32.astype(dtype)
        c_raw = c_f32.astype(dtype)
        a_base, b_base, c_base = a_raw.astype(np.float64), b_raw.astype(np.float64), c_raw.astype(np.float64)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = _KERNELS_CURVED[metric]

    # High-precision baseline (f64)
    accurate_dt, accurate = profile(precise_kernel or baseline_kernel, a_base, b_base, c_base)

    # Baseline at native compute precision (for error-stat collection)
    native_dt = np.dtype(compute_dtype).type
    expected_dt, expected = profile(
        baseline_kernel,
        a_base.astype(native_dt),
        b_base.astype(native_dt),
        c_base.astype(native_dt),
    )

    # SIMD result
    result_dt, result = profile(simd_kernel, a_raw, b_raw, c_raw, dtype)
    result = np.asarray(result)

    err_msg = LazyFormat(
        lambda: (
            f"\n{metric}({dtype}, ndim={ndim}):"
            f"\n  Accurate:  {accurate}"
            f"\n  Got:       {result}"
            f"\n  Raw a:     {hex_array(a_raw)}"
            f"\n  Raw b:     {hex_array(b_raw)}"
            f"\n  Raw c:     {hex_array(c_raw)}"
        )
    )

    np.testing.assert_allclose(result, accurate, atol=NK_ATOL, rtol=NK_RTOL, err_msg=err_msg)
    collect_errors(metric, ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, _stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", curved_dimensions)
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_curved_complex(ndim, dtype, capability):
    """Complex bilinear form — separate because operand construction differs."""
    a = (np.random.randn(ndim) + 1.0j * np.random.randn(ndim)).astype(dtype)
    b = (np.random.randn(ndim) + 1.0j * np.random.randn(ndim)).astype(dtype)
    c = (np.random.randn(ndim, ndim) + 1.0j * np.random.randn(ndim, ndim)).astype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = _KERNELS_CURVED["bilinear"]
    precise_dtype = np.clongdouble if dtype == "complex128" else np.complex128
    accurate_dt, accurate = profile(
        baseline_kernel,
        a.astype(precise_dtype),
        b.astype(precise_dtype),
        c.astype(precise_dtype),
    )
    expected_dt, expected = profile(baseline_kernel, a, b, c)
    result_dt, result = profile(simd_kernel, a, b, c)
    result = np.asarray(result)

    np.testing.assert_allclose(result, accurate, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors("bilinear", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, _stats)
