#!/usr/bin/env python3
"""Test spatial distances: nk.euclidean, nk.sqeuclidean, nk.angular.

Covers dtypes: float64, float32, float16, bfloat16, e4m3, e5m2, e2m3, e3m2, int8, uint8.
Parametrized over: ndim from dense_dimensions, metric, capability.

Precision notes:
    Integer dtypes use exact ±1 tolerance (discrete arithmetic with possible
    accumulator width differences). Floating-point dtypes use NK_ATOL/NK_RTOL
    (0.1/0.1). Sub-byte floats (bf16, e4m3, e5m2, e2m3, e3m2) carry wider
    quantization noise but are held to the same relative error bar.

Matches C++ suite: test_spatial.cpp.
"""

import atexit
import decimal
import math
import warnings

import pytest

try:
    import numpy as np
except:  # noqa: E722
    np = None

import numkong as nk
from test_base import (
    DECIMAL_PRECISION,
    NATIVE_COMPUTE_DTYPE,
    NK_ATOL,
    NK_RTOL,
    LazyFormat,
    assert_allclose,
    collect_errors,
    create_stats,
    dense_dimensions,
    keep_one_capability,
    make_random,
    make_random_buffer,
    nk_seed,  # noqa: F401 — pytest fixture
    numpy_available,
    possible_capabilities,
    print_stats_report,
    profile,
    randomized_repetitions_count,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
    tolerances_for_dtype,
)

algebraic_dtypes = ["float32", "float64"]
algebraic_ndims = [7, 97]
stats = create_stats()
atexit.register(print_stats_report, stats)

try:
    import scipy.spatial.distance as spd

    baseline_euclidean = lambda x, y: np.array(spd.euclidean(x, y))
    baseline_sqeuclidean = spd.sqeuclidean
    baseline_angular = spd.cosine
except ImportError:
    baseline_angular = lambda x, y: 1.0 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    baseline_euclidean = lambda x, y: np.array([np.sqrt(np.sum((x - y) ** 2))])
    baseline_sqeuclidean = lambda x, y: np.sum((x - y) ** 2)


def precise_sqeuclidean(a, b):
    """High-precision squared Euclidean distance via Python Decimal."""
    with decimal.localcontext() as ctx:
        ctx.prec = DECIMAL_PRECISION
        D = decimal.Decimal
        return float(sum((D.from_float(float(x)) - D.from_float(float(y))) ** 2 for x, y in zip(a, b)))


def precise_euclidean(a, b):
    return math.sqrt(precise_sqeuclidean(a, b))


def precise_angular(a, b):
    """High-precision angular/cosine distance via Python Decimal."""
    with decimal.localcontext() as ctx:
        ctx.prec = DECIMAL_PRECISION
        D = decimal.Decimal
        ab, aa, bb = D(0), D(0), D(0)
        for x, y in zip(a, b):
            dx = D.from_float(float(x))
            dy = D.from_float(float(y))
            ab += dx * dy
            aa += dx * dx
            bb += dy * dy
        denom = (aa * bb).sqrt()
        if aa == 0 and bb == 0:
            return 0.0
        if denom == 0:
            return 1.0
        return float(1 - ab / denom)


KERNELS_SPATIAL = {
    "euclidean": (baseline_euclidean if numpy_available else None, nk.euclidean, precise_euclidean),
    "sqeuclidean": (baseline_sqeuclidean if numpy_available else None, nk.sqeuclidean, precise_sqeuclidean),
    "angular": (baseline_angular if numpy_available else None, nk.angular, precise_angular),
}


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
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "angular"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_spatial_random_accuracy(ndim, dtype, metric, capability, nk_seed):
    """Spatial distances across all numeric dtypes against high-precision Decimal baselines."""
    a_raw, a_baseline = make_random((ndim,), dtype, seed=nk_seed)
    b_raw, b_baseline = make_random((ndim,), dtype, seed=nk_seed + 1)
    atol, rtol = tolerances_for_dtype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_SPATIAL[metric]

    # High-precision baseline (always f64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        accurate_dt, accurate = profile(precise_kernel or baseline_kernel, a_baseline, b_baseline)

    # Baseline at native precision (for error stats)
    if baseline_kernel is not None:
        native_dt = NATIVE_COMPUTE_DTYPE.get(dtype, np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected_dt, expected = profile(baseline_kernel, a_baseline.astype(native_dt), b_baseline.astype(native_dt))
    else:
        expected_dt, expected = 0, None

    # SIMD result
    result_dt, result = profile(simd_kernel, a_raw, b_raw, dtype)

    err_msg = LazyFormat(
        lambda: (
            f"\n{metric}({dtype}, ndim={ndim}):"
            f"\n  Accurate:  {accurate}"
            f"\n  Got:       {result}"
        )
    )

    assert_allclose(result, accurate, atol=atol, rtol=rtol, err_msg=err_msg)
    collect_errors(metric, ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_angular_zero_vector(ndim, dtype, capability):
    """Tests the nk.angular() function with zero vectors, to catch division by zero errors."""
    a = np.zeros(ndim, dtype=dtype)
    b = (np.random.randn(ndim) + 1).astype(dtype)
    keep_one_capability(capability)

    result = nk.angular(a, b)
    assert result == 1, f"Expected 1, but got {result}"

    result = nk.angular(a, a)
    assert result == 0, f"Expected 0 distance from itself, but got {result}"

    result = nk.angular(b, b)
    assert abs(result) < NK_ATOL, f"Expected 0 distance from itself, but got {result}"

    assert np.all(result >= 0), "Negative result for angular distance"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_spatial_self_distance_zero(ndim, dtype, capability):
    """d(v, v) should be 0 for euclidean, sqeuclidean, and angular."""
    keep_one_capability(capability)
    v = nk.full((ndim,), 1.5, dtype=dtype)
    atol = NK_ATOL
    assert abs(nk.euclidean(v, v)) < atol
    assert abs(nk.sqeuclidean(v, v)) < atol
    assert abs(nk.angular(v, v)) < atol


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_euclidean_known(ndim, dtype, capability):
    """euclidean(ones, zeros) = sqrt(n)."""
    keep_one_capability(capability)
    ones_vector = nk.ones((ndim,), dtype=dtype)
    zeros_vector = nk.zeros((ndim,), dtype=dtype)
    result = nk.euclidean(ones_vector, zeros_vector)
    expected = math.sqrt(ndim)
    assert abs(result - expected) < NK_ATOL + NK_RTOL * expected


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_sqeuclidean_known(ndim, dtype, capability):
    """sqeuclidean(ones, zeros) = n."""
    keep_one_capability(capability)
    ones_vector = nk.ones((ndim,), dtype=dtype)
    zeros_vector = nk.zeros((ndim,), dtype=dtype)
    result = nk.sqeuclidean(ones_vector, zeros_vector)
    assert abs(result - ndim) < NK_ATOL + NK_RTOL * ndim


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_spatial_symmetry(ndim, dtype, capability):
    """Commutativity: d(a, b) = d(b, a) for all spatial metrics."""
    keep_one_capability(capability)
    a = make_random_buffer(ndim, dtype)
    b = make_random_buffer(ndim, dtype)
    for metric_fn in [nk.euclidean, nk.sqeuclidean, nk.angular]:
        d_ab = metric_fn(a, b)
        d_ba = metric_fn(b, a)
        assert abs(d_ab - d_ba) < NK_ATOL, f"{metric_fn.__name__}: d(a,b)={d_ab} != d(b,a)={d_ba}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_spatial_non_negative(ndim, dtype, capability):
    """Non-negativity: d(a, b) >= 0 for all spatial metrics."""
    keep_one_capability(capability)
    a = make_random_buffer(ndim, dtype)
    b = make_random_buffer(ndim, dtype)
    for metric_fn in [nk.euclidean, nk.sqeuclidean, nk.angular]:
        distance = metric_fn(a, b)
        assert distance >= -NK_ATOL, f"{metric_fn.__name__}: d(a,b)={distance} is negative"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_euclidean_triangle_inequality(ndim, dtype, capability):
    """Triangle inequality: euclidean(a, c) <= euclidean(a, b) + euclidean(b, c)."""
    keep_one_capability(capability)
    a = make_random_buffer(ndim, dtype)
    b = make_random_buffer(ndim, dtype)
    c = make_random_buffer(ndim, dtype)
    d_ac = nk.euclidean(a, c)
    d_ab = nk.euclidean(a, b)
    d_bc = nk.euclidean(b, c)
    assert d_ac <= d_ab + d_bc + NK_ATOL, f"Triangle inequality violated: d(a,c)={d_ac} > d(a,b)+d(b,c)={d_ab + d_bc}"
