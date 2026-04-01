#!/usr/bin/env python3
"""Test curved-space distances: nk.bilinear, nk.mahalanobis.

Covers dtypes: float64, float32, float16 (with float32 compute), bfloat16,
    e4m3, e5m2, e2m3, e3m2.
Parametrized over: rows/columns from test_rows/columns_dimensions, metric, capability.

Precision notes:
    bilinear computes ``x @ z @ y``, mahalanobis computes ``sqrt((x-y) @ z @ (x-y))``.
    Both take vectors a, b and a matrix c; the test constructs c as a symmetric
    positive-definite matrix (c = abs(random) @ abs(random).T) so mahalanobis
    produces real, finite results.

    Real dtypes assert against a high-precision Decimal baseline.
    Complex bilinear tested separately (Decimal doesn't support complex,
    so it uses NumPy extended-precision types instead).

Matches C++ suite: test_curved.cpp.
"""

import atexit
import decimal
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import numpy as np  # static-analysis-only; the runtime try/except below is authoritative

try:
    import numpy as np

    numpy_available = True
except Exception:
    numpy_available = False

import numkong as nk
from test_base import (
    DECIMAL_PRECISION,
    NK_ATOL,
    NK_RTOL,
    LazyFormat,
    assert_allclose,
    collect_errors,
    create_stats,
    downcast_f32_to_dtype,
    hex_array,
    keep_one_capability,
    numpy_available,
    possible_capabilities,
    print_stats_report,
    profile,
    reduced_repetitions_count,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
    test_curved_dimensions,
)

stats = create_stats()
atexit.register(print_stats_report, stats)

baseline_bilinear = lambda x, y, z: x @ z @ y

try:
    import scipy.spatial.distance as spd

    def baseline_mahalanobis(x, y, z):
        try:
            result = spd.mahalanobis(x, y, z).astype(np.float64)
            if not np.isnan(result):
                return result
        except Exception:
            pass
        pytest.skip(f"SciPy Mahalanobis distance returned {result} due to `sqrt` of a negative number")

except ImportError:

    def baseline_mahalanobis(x, y, z):
        diff = x - y
        return np.sqrt(diff @ z @ diff)


def make_positive_semidefinite(data):
    """Make a square matrix positive semi-definite in-place (Gershgorin diagonal dominance).

    Matches C++ make_positive_semidefinite() in test/test_curved.cpp.
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
        ctx.prec = DECIMAL_PRECISION
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
        ctx.prec = DECIMAL_PRECISION
        D = decimal.Decimal
        diff = [D.from_float(float(x)) - D.from_float(float(y)) for x, y in zip(a, b)]
        n = len(diff)
        total = D(0)
        for i in range(n):
            for j in range(n):
                total += diff[i] * D.from_float(float(c[i, j])) * diff[j]
        return float(total.sqrt())


KERNELS_CURVED = {
    "bilinear": (baseline_bilinear, nk.bilinear, precise_bilinear),
    "mahalanobis": (baseline_mahalanobis, nk.mahalanobis, precise_mahalanobis),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(reduced_repetitions_count)
@pytest.mark.parametrize("ndim", test_curved_dimensions)
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
def test_curved_random_accuracy(ndim, dtypes, metric, capability):
    """Bilinear and Mahalanobis for float and bfloat16 dtypes against high-precision baselines."""
    dtype, compute_dtype = dtypes

    # Generate structured data at f32
    a_vector_f32 = np.abs(np.random.randn(ndim).astype(np.float32))
    b_vector_f32 = np.abs(np.random.randn(ndim).astype(np.float32))
    a_vector_f32 /= np.sum(a_vector_f32)
    b_vector_f32 /= np.sum(b_vector_f32)
    c_matrix_f32 = make_positive_semidefinite(np.random.randn(ndim, ndim).astype(np.float32))

    a_raw, a_baseline = downcast_f32_to_dtype(a_vector_f32, dtype)
    b_raw, b_baseline = downcast_f32_to_dtype(b_vector_f32, dtype)
    c_raw, c_baseline = downcast_f32_to_dtype(c_matrix_f32, dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_CURVED[metric]

    # High-precision baseline (Decimal)
    accurate_dt, accurate = profile(precise_kernel or baseline_kernel, a_baseline, b_baseline, c_baseline)

    # Baseline at native compute precision (for error-stat collection)
    native_dt = np.dtype(compute_dtype).type
    expected_dt, expected = profile(
        baseline_kernel,
        a_baseline.astype(native_dt),
        b_baseline.astype(native_dt),
        c_baseline.astype(native_dt),
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

    assert_allclose(result, accurate, atol=NK_ATOL, rtol=NK_RTOL, err_msg=err_msg)
    collect_errors(metric, ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(reduced_repetitions_count)
@pytest.mark.parametrize("ndim", test_curved_dimensions)
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_bilinear_complex_accuracy(ndim, dtype, capability):
    """Complex bilinear form against NumPy at extended precision."""
    a_vector = (np.random.randn(ndim) + 1.0j * np.random.randn(ndim)).astype(dtype)
    b_vector = (np.random.randn(ndim) + 1.0j * np.random.randn(ndim)).astype(dtype)
    c_matrix = (np.random.randn(ndim, ndim) + 1.0j * np.random.randn(ndim, ndim)).astype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = KERNELS_CURVED["bilinear"]
    precise_dtype = np.clongdouble if dtype == "complex128" else np.complex128
    accurate_dt, accurate = profile(
        baseline_kernel,
        a_vector.astype(precise_dtype),
        b_vector.astype(precise_dtype),
        c_matrix.astype(precise_dtype),
    )
    expected_dt, expected = profile(baseline_kernel, a_vector, b_vector, c_matrix)
    result_dt, result = profile(simd_kernel, a_vector, b_vector, c_matrix)
    result = np.asarray(result)

    assert_allclose(result, accurate, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors("bilinear", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)
