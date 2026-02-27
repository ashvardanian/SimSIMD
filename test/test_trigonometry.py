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
import math as _math
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
    _seed_rng,
)

_algebraic_dtypes = ["float32", "float64"]
_algebraic_ndims = [7, 97]
_stats = create_stats()
atexit.register(print_stats_report, _stats)


def baseline_sin(a):
    """Reference sin via NumPy."""
    return np.sin(a)


def baseline_cos(a):
    """Reference cos via NumPy."""
    return np.cos(a)


def baseline_atan(a):
    """Reference arctan via NumPy."""
    return np.arctan(a)


_KERNELS_TRIGONOMETRY = {
    "sin": (baseline_sin, nk.sin, None),
    "cos": (baseline_cos, nk.cos, None),
    "atan": (baseline_atan, nk.atan, None),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("metric", list(_KERNELS_TRIGONOMETRY.keys()))
@pytest.mark.parametrize("capability", possible_capabilities)
def test_trigonometric(ndim, dtype, metric, capability):
    """Test nk trig functions against NumPy."""
    a = np.random.uniform(-np.pi, np.pi, ndim).astype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = _KERNELS_TRIGONOMETRY[metric]

    accurate_dt, accurate = profile(baseline_kernel, a.astype(np.float64))
    expected_dt, expected = profile(baseline_kernel, a)
    result_dt, result = profile(simd_kernel, a)
    result = np.asarray(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(metric, ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, _stats)


@pytest.mark.parametrize("ndim", _algebraic_ndims)
@pytest.mark.parametrize("dtype", _algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_trig_at_zero(ndim, dtype, capability):
    """sin(0)~0, cos(0)~1, atan(0)~0."""
    keep_one_capability(capability)
    z = nk.zeros((ndim,), dtype=dtype)
    sin_z = list(nk.sin(z))
    cos_z = list(nk.cos(z))
    atan_z = list(nk.atan(z))
    for i in range(ndim):
        assert abs(sin_z[i]) < NK_ATOL, f"sin(0)[{i}]={sin_z[i]}"
        assert abs(cos_z[i] - 1.0) < NK_ATOL, f"cos(0)[{i}]={cos_z[i]}"
        assert abs(atan_z[i]) < NK_ATOL, f"atan(0)[{i}]={atan_z[i]}"


@pytest.mark.parametrize("ndim", _algebraic_ndims)
@pytest.mark.parametrize("dtype", _algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_sin_cos_known(ndim, dtype, capability):
    """sin(pi/2)~1, cos(pi/2)~0, atan(1)~pi/4."""
    keep_one_capability(capability)
    half_pi = nk.full((ndim,), _math.pi / 2, dtype=dtype)
    one_vec = nk.ones((ndim,), dtype=dtype)
    sin_hp = list(nk.sin(half_pi))
    cos_hp = list(nk.cos(half_pi))
    atan_one = list(nk.atan(one_vec))
    for i in range(ndim):
        assert abs(sin_hp[i] - 1.0) < NK_ATOL, f"sin(pi/2)[{i}]={sin_hp[i]}"
        assert abs(cos_hp[i]) < NK_ATOL, f"cos(pi/2)[{i}]={cos_hp[i]}"
        assert abs(atan_one[i] - _math.pi / 4) < NK_ATOL, f"atan(1)[{i}]={atan_one[i]}"


@pytest.mark.parametrize("ndim", _algebraic_ndims)
@pytest.mark.parametrize("dtype", _algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_pythagorean_identity(ndim, dtype, capability):
    """sin^2(x) + cos^2(x) ~ 1."""
    keep_one_capability(capability)
    x = make_random_buffer(ndim, dtype)
    sin_x = list(nk.sin(x))
    cos_x = list(nk.cos(x))
    for i in range(ndim):
        identity = sin_x[i] ** 2 + cos_x[i] ** 2
        assert abs(identity - 1.0) < NK_ATOL, f"sin²+cos²={identity} at [{i}]"


@pytest.mark.parametrize("ndim", _algebraic_ndims)
@pytest.mark.parametrize("dtype", _algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_trig_odd_even(ndim, dtype, capability):
    """sin(-x) ~ -sin(x) (odd), cos(-x) ~ cos(x) (even)."""
    keep_one_capability(capability)
    for c in [0.5, 1.0, 2.0]:
        pos = nk.full((ndim,), c, dtype=dtype)
        neg = nk.full((ndim,), -c, dtype=dtype)
        sin_pos = list(nk.sin(pos))
        sin_neg = list(nk.sin(neg))
        cos_pos = list(nk.cos(pos))
        cos_neg = list(nk.cos(neg))
        for i in range(ndim):
            assert abs(sin_neg[i] + sin_pos[i]) < NK_ATOL, f"sin(-{c}) + sin({c}) = {sin_neg[i] + sin_pos[i]}"
            assert abs(cos_neg[i] - cos_pos[i]) < NK_ATOL, f"cos(-{c}) - cos({c}) = {cos_neg[i] - cos_pos[i]}"
