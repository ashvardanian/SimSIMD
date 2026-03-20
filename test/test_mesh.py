#!/usr/bin/env python3
"""Test mesh alignment: nk.kabsch, nk.umeyama, nk.rmsd.

Covers dtypes: float64, float32, float16, bfloat16.
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
except:  # noqa: E722
    np = None

import numkong as nk
from test_base import (
    assert_allclose,
    create_stats,
    downcast_f32_to_dtype,
    keep_one_capability,
    make_nk,
    numpy_available,
    possible_capabilities,
    print_stats_report,
    scipy_available,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
    tolerances_for_dtype,
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


def _make_point_cloud(dtype, scale=1.0):
    """Create a (4,3) point cloud as an nk.Tensor in the requested dtype."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype="float32") * scale
    raw, _ = downcast_f32_to_dtype(pts, dtype)
    return make_nk(raw, dtype)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "bfloat16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_mesh_kabsch(dtype, capability):
    """Kabsch on identical points vs SciPy procrustes: expects near-zero disparity."""
    point_cloud = _make_point_cloud(dtype)
    keep_one_capability(capability)
    result = nk.kabsch(point_cloud, point_cloud)
    scipy_disparity = baseline_kabsch(np.array(point_cloud, dtype="float64"), np.array(point_cloud, dtype="float64"))

    atol, rtol = tolerances_for_dtype(dtype)
    assert hasattr(result, "rotation"), "Expected MeshAlignmentResult with .rotation attribute"
    assert_allclose(float(np.array(result.scale)), 1.0, atol=atol, rtol=rtol)
    assert_allclose(float(np.array(result.rmsd)), 0.0, atol=atol, rtol=rtol)
    assert scipy_disparity < 0.01


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "bfloat16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_mesh_umeyama(dtype, capability):
    """Umeyama on 2x-scaled points: expects scale~2."""
    point_cloud = _make_point_cloud(dtype)
    scaled = _make_point_cloud(dtype, scale=2.0)
    keep_one_capability(capability)
    result = nk.umeyama(point_cloud, scaled)
    assert 1.0 < float(np.array(result.scale)) < 3.0


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "bfloat16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_mesh_rmsd(dtype, capability):
    """RMSD on identical points: expects RMSD~0, scale=1."""
    point_cloud = _make_point_cloud(dtype)
    keep_one_capability(capability)
    result = nk.rmsd(point_cloud, point_cloud)
    rmsd_val = float(np.array(result.rmsd))
    atol, rtol = tolerances_for_dtype(dtype)
    assert not np.isnan(rmsd_val), "RMSD should not be NaN for identical point clouds"
    assert_allclose(float(np.array(result.scale)), 1.0, atol=atol, rtol=rtol)
    assert_allclose(rmsd_val, 0.0, atol=atol, rtol=rtol)


@pytest.mark.parametrize("capability", possible_capabilities)
def test_rmsd_self_zero(capability):
    """rmsd(cloud, cloud).rmsd ~ 0 for identical point clouds."""
    keep_one_capability(capability)
    point_cloud = nk.ones((4, 3), dtype="float64")
    result = nk.rmsd(point_cloud, point_cloud)
    rmsd_val = float(result.rmsd)
    assert_allclose(rmsd_val, 0.0)
