#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test trigonometric functions: nk.sin, nk.cos, nk.atan.

Covers dtypes: float32, float64.
Parametrized over: ndim from dense_dimensions, dtype, capability from possible_capabilities.

Precision notes:
    All trig functions tested on uniform random inputs in [-pi, pi].
    Assertions use NK_ATOL/NK_RTOL (0.1/0.1) against NumPy references.

Matches C++ suite: test_trigonometry.cpp.
"""

import atexit
import math
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
    collect_errors,
    create_stats,
    print_stats_report,
    seed_rng,
)

algebraic_dtypes = ["float32", "float64"]
algebraic_ndims = [7, 97]
stats = create_stats()
atexit.register(print_stats_report, stats)


def baseline_sin(a):
    """Reference sin via NumPy."""
    return np.sin(a)


def baseline_cos(a):
    """Reference cos via NumPy."""
    return np.cos(a)


def baseline_atan(a):
    """Reference arctan via NumPy."""
    return np.arctan(a)


KERNELS_TRIGONOMETRY = {
    "sin": (baseline_sin, nk.sin, None),
    "cos": (baseline_cos, nk.cos, None),
    "atan": (baseline_atan, nk.atan, None),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("metric", list(KERNELS_TRIGONOMETRY.keys()))
@pytest.mark.parametrize("capability", possible_capabilities)
def test_trigonometry_random_accuracy(ndim, dtype, metric, capability):
    """sin, cos, atan on random inputs in [-pi, pi] against NumPy baselines."""
    a = np.random.uniform(-np.pi, np.pi, ndim).astype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = KERNELS_TRIGONOMETRY[metric]

    accurate_dt, accurate = profile(baseline_kernel, a.astype(np.float64))
    expected_dt, expected = profile(baseline_kernel, a)
    result_dt, result = profile(simd_kernel, a)
    result = np.asarray(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(metric, ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_trigonometry_at_zero(ndim, dtype, capability):
    """sin(0)~0, cos(0)~1, atan(0)~0."""
    keep_one_capability(capability)
    zeros_vector = nk.zeros((ndim,), dtype=dtype)
    sin_values = list(nk.sin(zeros_vector))
    cos_values = list(nk.cos(zeros_vector))
    atan_values = list(nk.atan(zeros_vector))
    for i in range(ndim):
        assert abs(sin_values[i]) < NK_ATOL, f"sin(0)[{i}]={sin_values[i]}"
        assert abs(cos_values[i] - 1.0) < NK_ATOL, f"cos(0)[{i}]={cos_values[i]}"
        assert abs(atan_values[i]) < NK_ATOL, f"atan(0)[{i}]={atan_values[i]}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_trigonometry_known_values(ndim, dtype, capability):
    """sin(pi/2)~1, cos(pi/2)~0, atan(1)~pi/4 for all elements."""
    keep_one_capability(capability)
    half_pi = nk.full((ndim,), math.pi / 2, dtype=dtype)
    ones_vector = nk.ones((ndim,), dtype=dtype)
    sin_values = list(nk.sin(half_pi))
    cos_values = list(nk.cos(half_pi))
    atan_one = list(nk.atan(ones_vector))
    for i in range(ndim):
        assert abs(sin_values[i] - 1.0) < NK_ATOL, f"sin(pi/2)[{i}]={sin_values[i]}"
        assert abs(cos_values[i]) < NK_ATOL, f"cos(pi/2)[{i}]={cos_values[i]}"
        assert abs(atan_one[i] - math.pi / 4) < NK_ATOL, f"atan(1)[{i}]={atan_one[i]}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_pythagorean_identity(ndim, dtype, capability):
    """sin^2(x) + cos^2(x) ~ 1."""
    keep_one_capability(capability)
    input_angles = make_random_buffer(ndim, dtype)
    sin_values = list(nk.sin(input_angles))
    cos_values = list(nk.cos(input_angles))
    for i in range(ndim):
        identity = sin_values[i] ** 2 + cos_values[i] ** 2
        assert abs(identity - 1.0) < NK_ATOL, f"sin²+cos²={identity} at [{i}]"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_trigonometry_odd_even(ndim, dtype, capability):
    """sin(-x) ~ -sin(x) (odd), cos(-x) ~ cos(x) (even)."""
    keep_one_capability(capability)
    for random_angles in [0.5, 1.0, 2.0]:
        positive_input = nk.full((ndim,), random_angles, dtype=dtype)
        negative_input = nk.full((ndim,), -random_angles, dtype=dtype)
        sin_positive = list(nk.sin(positive_input))
        sin_negative = list(nk.sin(negative_input))
        cos_positive = list(nk.cos(positive_input))
        cos_negative = list(nk.cos(negative_input))
        for i in range(ndim):
            assert (
                abs(sin_negative[i] + sin_positive[i]) < NK_ATOL
            ), f"sin(-{random_angles}) + sin({random_angles}) = {sin_negative[i] + sin_positive[i]}"
            assert (
                abs(cos_negative[i] - cos_positive[i]) < NK_ATOL
            ), f"cos(-{random_angles}) - cos({random_angles}) = {cos_negative[i] - cos_positive[i]}"
