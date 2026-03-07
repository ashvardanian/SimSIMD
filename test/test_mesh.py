#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test mesh alignment: nk.kabsch, nk.umeyama, nk.rmsd.

Covers dtypes: float64, float32.
Parametrized over: capability from possible_capabilities.

Precision notes:
    Kabsch on identical points expects near-zero RMSD and unit scale.
    Umeyama on 2x-scaled points expects scale in (1, 3).
    RMSD on identical point clouds expects near-zero result and unit scale.

Matches C++ suite: test_mesh.cpp.
"""

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
    possible_capabilities,
    keep_one_capability,
    create_stats,
    print_stats_report,
    seed_rng,
)

stats = create_stats()
atexit.register(print_stats_report, stats)

try:
    from scipy.spatial import procrustes as scipy_procrustes

    def baseline_kabsch(source, target):
        """Reference Kabsch via SciPy procrustes: returns disparity (should be ~0 for identical)."""
        _, _, disparity = scipy_procrustes(source.astype(np.float64), target.astype(np.float64))
        return disparity

except ImportError:
    baseline_kabsch = None

KERNELS_MESH = {
    "kabsch": (baseline_kabsch, nk.kabsch, None),
    "umeyama": (None, nk.umeyama, None),
    "rmsd": (None, nk.rmsd, None),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_mesh_kabsch(dtype, capability):
    """Kabsch on identical points vs SciPy procrustes: expects near-zero disparity."""
    point_cloud = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=dtype)
    keep_one_capability(capability)
    result = nk.kabsch(point_cloud, point_cloud.copy())
    scipy_disparity = baseline_kabsch(point_cloud, point_cloud.copy())

    assert hasattr(result, "rotation"), "Expected MeshAlignmentResult with .rotation attribute"
    assert abs(float(np.array(result.scale)) - 1.0) < 1e-4
    assert float(np.array(result.rmsd)) < 0.01
    assert scipy_disparity < 0.01


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_mesh_umeyama(dtype, capability):
    """Umeyama on 2x-scaled points: expects scale~2."""
    point_cloud = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=dtype)
    keep_one_capability(capability)
    result = nk.umeyama(point_cloud, (point_cloud * 2).astype(dtype))
    assert 1.0 < float(np.array(result.scale)) < 3.0


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_mesh_rmsd(dtype, capability):
    """RMSD on identical points: expects RMSD~0, scale=1."""
    point_cloud = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=dtype)
    keep_one_capability(capability)
    result = nk.rmsd(point_cloud, point_cloud.copy())
    rmsd_val = float(np.array(result.rmsd))
    assert not np.isnan(rmsd_val), "RMSD should not be NaN for identical point clouds"
    assert abs(float(np.array(result.scale)) - 1.0) < 1e-4
    assert rmsd_val < 1e-4


@pytest.mark.parametrize("capability", possible_capabilities)
def test_rmsd_self_zero(capability):
    """rmsd(cloud, cloud).rmsd ~ 0 for identical point clouds."""
    keep_one_capability(capability)
    point_cloud = nk.ones((4, 3), dtype="float64")
    result = nk.rmsd(point_cloud, point_cloud)
    rmsd_val = float(result.rmsd)
    assert rmsd_val < 1e-4, f"rmsd(cloud, cloud) = {rmsd_val}, expected ~0"
