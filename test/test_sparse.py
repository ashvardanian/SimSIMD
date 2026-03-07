#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test sparse operations: nk.sparse_dot, nk.intersect.

Covers dtypes: float32 values with uint32 indices (sparse_dot),
    uint16/uint32 indices (intersect).
Parametrized over: capability, index dtype, length bounds.

Precision notes:
    sparse_dot uses NK_ATOL/NK_RTOL against manual weighted intersection.
    intersect uses exact integer comparison (round to nearest int).

Matches C++ suite: test_sparse.cpp.
"""

import atexit
import platform

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
    is_running_under_qemu,
    keep_one_capability,
    profile,
    NK_ATOL,
    NK_RTOL,
    collect_errors,
    create_stats,
    print_stats_report,
    seed_rng,
)

stats = create_stats()
atexit.register(print_stats_report, stats)

baseline_intersect = lambda x, y: len(np.intersect1d(x, y))

KERNELS_SPARSE = {
    "intersect": (baseline_intersect, nk.intersect, None),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_sparse_dot(capability):
    """Test nk.sparse_dot against manual weighted intersection."""
    a_idx = np.unique(np.random.randint(0, 1000, size=50)).astype(np.uint32)
    b_idx = np.unique(np.random.randint(0, 1000, size=50)).astype(np.uint32)
    a_val = np.random.randn(len(a_idx)).astype(np.float32)
    b_val = np.random.randn(len(b_idx)).astype(np.float32)

    keep_one_capability(capability)
    result_dt, result = profile(nk.sparse_dot, a_idx, a_val, b_idx, b_val)

    def _sparse_dot_baseline(a_idx, a_val, b_idx, b_val):
        common = np.intersect1d(a_idx, b_idx)
        total = 0.0
        for idx in common:
            ai = np.searchsorted(a_idx, idx)
            bi = np.searchsorted(b_idx, idx)
            total += float(a_val[ai]) * float(b_val[bi])
        return total

    accurate_dt, accurate = profile(
        _sparse_dot_baseline, a_idx, a_val.astype(np.float64), b_idx, b_val.astype(np.float64)
    )
    expected_dt, expected = profile(_sparse_dot_baseline, a_idx, a_val, b_idx, b_val)

    np.testing.assert_allclose(result, accurate, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(
        "sparse_dot", len(a_idx), "float32", accurate, accurate_dt, expected, expected_dt, result, result_dt, stats
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("dtype", ["uint16", "uint32"])
@pytest.mark.parametrize("first_length_bound", [10, 100, 1000])
@pytest.mark.parametrize("second_length_bound", [10, 100, 1000])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_intersect(dtype, first_length_bound, second_length_bound, capability):
    """Compares the nk.intersect() function with numpy.intersect1d."""
    if is_running_under_qemu() and (platform.machine() == "aarch64" or platform.machine() == "arm64"):
        pytest.skip("In QEMU `aarch64` emulation on `x86_64` the `intersect` function is not reliable")

    a_length = np.random.randint(1, first_length_bound)
    b_length = np.random.randint(1, second_length_bound)
    a = np.random.randint(first_length_bound * 2, size=a_length, dtype=dtype)
    b = np.random.randint(second_length_bound * 2, size=b_length, dtype=dtype)

    a = np.unique(a)
    b = np.unique(b)

    keep_one_capability(capability)
    expected = baseline_intersect(a, b)
    result = nk.intersect(a, b)

    assert round(float(expected)) == round(float(result)), (
        f"Intersection count mismatch: expected {expected}, got {result}. "
        f"Intersection: {np.intersect1d(a, b)}, a={a}, b={b}"
    )
