#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test probability divergences: nk.jensenshannon.

Covers dtypes: float32, float16.
Parametrized over: ndim from dense_dimensions, capability from possible_capabilities.

Precision notes:
    Currently skipped — tolerance bounds for numerical errors in log-based
    divergence computation have not been reliably established.

Matches C++ suite: test_probability.cpp.
"""

import atexit
import pytest

try:
    import numpy as np
except:
    np = None

import numkong as nk
from test_base import (
    make_positive_buffer,
    numpy_available,
    scipy_available,
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

_algebraic_ndims = [7, 97]
_stats = create_stats()
atexit.register(print_stats_report, _stats)

try:
    import scipy.spatial.distance as spd

    baseline_jensenshannon = lambda x, y: spd.jensenshannon(x, y)
except ImportError:

    def _fallback_jensenshannon(p, q):
        """Jensen-Shannon distance (sqrt of JSD) — fallback when SciPy is unavailable."""
        m = 0.5 * (p + q)
        left = np.sum(np.where(p > 0, p * np.log(p / m), 0.0))
        right = np.sum(np.where(q > 0, q * np.log(q / m), 0.0))
        return np.sqrt(np.clip(0.5 * (left + right), 0.0, None))

    baseline_jensenshannon = _fallback_jensenshannon

_KERNELS_PROBABILITY = {
    "jensenshannon": (baseline_jensenshannon, nk.jensenshannon, None),
}


@pytest.mark.skip(reason="Problems inferring the tolerance bounds for numerical errors")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_jensen_shannon(ndim, dtype, capability):
    a = np.abs(np.random.randn(ndim)).astype(dtype)
    b = np.abs(np.random.randn(ndim)).astype(dtype)
    a /= np.sum(a)
    b /= np.sum(b)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = _KERNELS_PROBABILITY["jensenshannon"]
    accurate_dt, accurate = profile(baseline_kernel, a.astype(np.float64), b.astype(np.float64))
    expected_dt, expected = profile(baseline_kernel, a, b)
    result_dt, result = profile(simd_kernel, a, b)
    result = np.asarray(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(
        "jensenshannon", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, _stats
    )


@pytest.mark.parametrize("ndim", _algebraic_ndims)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_kl_self_zero(ndim, dtype, capability):
    """kullbackleibler(p, p) ~ 0 for uniform distribution."""
    keep_one_capability(capability)
    p = nk.full((ndim,), 1.0 / ndim, dtype=dtype)
    result = nk.kullbackleibler(p, p)
    assert abs(result) < NK_ATOL, f"KL(p,p) = {result}, expected ~0"


@pytest.mark.parametrize("ndim", _algebraic_ndims)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_js_self_zero(ndim, dtype, capability):
    """jensenshannon(p, p) ~ 0 for uniform distribution."""
    keep_one_capability(capability)
    p = nk.full((ndim,), 1.0 / ndim, dtype=dtype)
    result = nk.jensenshannon(p, p)
    assert abs(result) < NK_ATOL, f"JS(p,p) = {result}, expected ~0"


@pytest.mark.parametrize("ndim", _algebraic_ndims)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_js_symmetry_nonneg(ndim, dtype, capability):
    """js(p,q) = js(q,p) and js(p,q) >= 0."""
    keep_one_capability(capability)
    p = make_positive_buffer(ndim, dtype)
    q = make_positive_buffer(ndim, dtype)
    js_pq = nk.jensenshannon(p, q)
    js_qp = nk.jensenshannon(q, p)
    assert abs(js_pq - js_qp) < NK_ATOL, f"JS not symmetric: js(p,q)={js_pq}, js(q,p)={js_qp}"
    assert js_pq >= -NK_ATOL, f"JS negative: {js_pq}"
