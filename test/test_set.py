#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test set distances: nk.jaccard, nk.hamming (dense binary).

Covers dtypes: uint1 (packed uint8 bits), also verified via boolean view.
Parametrized over: ndim from dense_dimensions, metric, capability.

Precision notes:
    Hamming and Jaccard/Tanimoto on packed bits use NK_ATOL/NK_RTOL (0.1/0.1).
    Baseline computed at uint64 precision for the accurate reference.

Matches C++ suite: test_set.cpp.
"""

import array as _array
import atexit
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

    baseline_hamming = lambda x, y: spd.hamming(x, y) * len(x)
    baseline_jaccard = spd.jaccard
except ImportError:
    baseline_hamming = lambda x, y: np.logical_xor(x, y).sum()

    def baseline_jaccard(x, y):
        intersection = np.logical_and(x, y).sum()
        union = np.logical_or(x, y).sum()
        return 0.0 if union == 0 else 1.0 - float(intersection) / float(union)


_KERNELS_SET = {
    "jaccard": (baseline_jaccard, nk.jaccard, None),
    "hamming": (baseline_hamming, nk.hamming, None),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("metric", ["jaccard", "hamming"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dense_bits(ndim, metric, capability):
    """Tests Hamming and Jaccard/Tanimoto distances for dense bit arrays."""
    a = np.random.randint(2, size=ndim).astype(np.uint8)
    b = np.random.randint(2, size=ndim).astype(np.uint8)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = _KERNELS_SET[metric]
    accurate_dt, accurate = profile(baseline_kernel, a.astype(np.uint64), b.astype(np.uint64))
    expected_dt, expected = profile(baseline_kernel, a, b)
    result_dt, result = profile(simd_kernel, np.packbits(a), np.packbits(b), "uint1")
    result = np.asarray(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(metric, ndim, "uint1", accurate, accurate_dt, expected, expected_dt, result, result_dt, _stats)

    # Also verify with boolean view
    result_dt, result = profile(simd_kernel, np.packbits(a).view(np.bool_), np.packbits(b).view(np.bool_))
    result = np.asarray(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(metric, ndim, "uint1", accurate, accurate_dt, expected, expected_dt, result, result_dt, _stats)


@pytest.mark.parametrize("ndim", _algebraic_ndims)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_hamming_self_zero(ndim, capability):
    """hamming(v, v) = 0 for identical uint8 vectors."""
    keep_one_capability(capability)
    v = _array.array("B", [0xFF] * ndim)
    result = nk.hamming(v, v, "uint1")
    assert result == 0, f"hamming(v,v) = {result}, expected 0"


@pytest.mark.parametrize("ndim", _algebraic_ndims)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_jaccard_self_zero(ndim, capability):
    """jaccard(v, v) = 0 for identical non-zero uint8 vectors."""
    keep_one_capability(capability)
    v = _array.array("B", [0xAA] * ndim)
    result = nk.jaccard(v, v, "uint1")
    assert abs(result) < NK_ATOL, f"jaccard(v,v) = {result}, expected 0"
