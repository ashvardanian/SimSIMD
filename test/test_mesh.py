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
    assert_allclose,
    create_stats,
    downcast_f32_to_dtype,
    keep_one_capability,
    make_nk,
    numpy_available,
    possible_capabilities,
    precise_decimal,
    print_stats_report,
    scipy_available,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
    dense_dimensions,
    tolerances_for_dtype,
)

stats = create_stats()
atexit.register(print_stats_report, stats)

try:
    from scipy.spatial import procrustes as scipy_procrustes

    def baseline_kabsch(source, target):
        """Reference Kabsch via SciPy procrustes: returns disparity (should be ~0 for identical).
        Returns None if SVD fails to converge (e.g., near-degenerate random point clouds).
        """
        try:
            _, _, disparity = scipy_procrustes(source.astype(np.float64), target.astype(np.float64))
            return disparity
        except np.linalg.LinAlgError:
            return None

except ImportError:
    baseline_kabsch = None


def baseline_rmsd(source, target):
    """NumPy reference RMSD: center both clouds, compute RMS of residuals."""
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    n_points = source.shape[0]
    return np.sqrt(np.sum((source_centered - target_centered) ** 2) / n_points)


def baseline_umeyama(source, target):
    """NumPy reference Umeyama: SVD-based similarity transform returning scale, rotation, RMSD."""
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    n_points = source.shape[0]
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    cross_covariance = source_centered.T @ target_centered / n_points
    left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(cross_covariance)
    determinant = np.linalg.det(right_vectors_transposed.T @ left_vectors.T)
    sign_vector = np.ones(3)
    sign_vector[-1] = np.sign(determinant)
    rotation = right_vectors_transposed.T @ np.diag(sign_vector) @ left_vectors.T
    source_variance = np.sum(source_centered**2) / n_points
    scale = np.sum(singular_values * sign_vector) / source_variance
    transformed = scale * (source_centered @ rotation.T)
    rmsd_valueue = np.sqrt(np.sum((transformed - target_centered) ** 2) / n_points)
    return {
        "rotation": rotation,
        "scale": scale,
        "rmsd": rmsd_valueue,
        "a_centroid": source_centroid,
        "b_centroid": target_centroid,
    }


def _decimal_jacobi_svd_3x3(matrix):
    """3x3 SVD via Jacobi eigendecomposition of A^T A, pure Decimal arithmetic.

    Returns (left_vectors, singular_values, right_vectors) as lists-of-lists / list.
    Convention: A ≈ left_vectors @ diag(singular_values) @ right_vectors^T.
    """
    D = decimal.Decimal
    symmetric = [[sum(matrix[k][i] * matrix[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    right_vectors = [[D(1) if i == j else D(0) for j in range(3)] for i in range(3)]

    for _ in range(30):
        for p in range(3):
            for q in range(p + 1, 3):
                if abs(symmetric[p][q]) < D("1e-80"):
                    continue
                difference = symmetric[q][q] - symmetric[p][p]
                if abs(difference) < D("1e-80"):
                    tangent = D(1)
                else:
                    tau = difference / (2 * symmetric[p][q])
                    tangent = (D(1) if tau >= 0 else D(-1)) / (abs(tau) + (1 + tau * tau).sqrt())
                cosine = D(1) / (1 + tangent * tangent).sqrt()
                sine = tangent * cosine
                rotated = [row[:] for row in symmetric]
                rotated[p][p] = (
                    cosine**2 * symmetric[p][p] - 2 * sine * cosine * symmetric[p][q] + sine**2 * symmetric[q][q]
                )
                rotated[q][q] = (
                    sine**2 * symmetric[p][p] + 2 * sine * cosine * symmetric[p][q] + cosine**2 * symmetric[q][q]
                )
                rotated[p][q] = rotated[q][p] = D(0)
                for r in range(3):
                    if r != p and r != q:
                        rotated[p][r] = rotated[r][p] = cosine * symmetric[p][r] - sine * symmetric[q][r]
                        rotated[q][r] = rotated[r][q] = sine * symmetric[p][r] + cosine * symmetric[q][r]
                symmetric = rotated
                for r in range(3):
                    old_p, old_q = right_vectors[r][p], right_vectors[r][q]
                    right_vectors[r][p] = cosine * old_p - sine * old_q
                    right_vectors[r][q] = sine * old_p + cosine * old_q

    singular_values = [symmetric[i][i].sqrt() if symmetric[i][i] > 0 else D(0) for i in range(3)]
    order = sorted(range(3), key=lambda i: -singular_values[i])
    singular_values = [singular_values[i] for i in order]
    right_vectors = [[right_vectors[r][order[c]] for c in range(3)] for r in range(3)]

    left_vectors = [[D(0)] * 3 for _ in range(3)]
    for i in range(3):
        if singular_values[i] > D("1e-80"):
            column = [sum(matrix[r][k] * right_vectors[k][i] for k in range(3)) for r in range(3)]
            for r in range(3):
                left_vectors[r][i] = column[r] / singular_values[i]

    return left_vectors, singular_values, right_vectors


def _decimal_center_and_covariance(source, target):
    """Convert points to Decimal, center, and compute cross-covariance. Returns
    (source_centered, target_centered, cross_covariance, n_points).
    """
    D = decimal.Decimal
    n_points = len(source)
    source_decimal = [[D.from_float(float(value)) for value in row] for row in source]
    target_decimal = [[D.from_float(float(value)) for value in row] for row in target]
    source_centroid = [sum(source_decimal[i][j] for i in range(n_points)) / n_points for j in range(3)]
    target_centroid = [sum(target_decimal[i][j] for i in range(n_points)) / n_points for j in range(3)]
    source_centered = [[source_decimal[i][j] - source_centroid[j] for j in range(3)] for i in range(n_points)]
    target_centered = [[target_decimal[i][j] - target_centroid[j] for j in range(3)] for i in range(n_points)]
    cross_covariance = [
        [sum(source_centered[k][i] * target_centered[k][j] for k in range(n_points)) for j in range(3)]
        for i in range(3)
    ]
    return source_centered, target_centered, cross_covariance, n_points


def _decimal_determinant_3x3(matrix):
    """Determinant of a 3x3 list-of-lists Decimal matrix."""
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def precise_rmsd(source, target):
    """High-precision RMSD via Python Decimal. No SVD needed."""
    with precise_decimal() as d:
        source_centered, target_centered, _, n_points = _decimal_center_and_covariance(source, target)
        total = d(0)
        for i in range(n_points):
            for j in range(3):
                difference = source_centered[i][j] - target_centered[i][j]
                total += difference**2
        return float((total / n_points).sqrt())


def precise_kabsch(source, target):
    """High-precision Kabsch via Decimal Jacobi SVD. Returns RMSD after optimal rotation."""
    with precise_decimal() as d:
        source_centered, target_centered, cross_covariance, n_points = _decimal_center_and_covariance(source, target)
        left_vectors, _, right_vectors = _decimal_jacobi_svd_3x3(cross_covariance)
        # R = V Uᵀ, fix reflection if det(R) < 0
        rotation = [
            [sum(right_vectors[r][k] * left_vectors[c][k] for k in range(3)) for c in range(3)] for r in range(3)
        ]
        if _decimal_determinant_3x3(rotation) < 0:
            for r in range(3):
                right_vectors[r][2] = -right_vectors[r][2]
            rotation = [
                [sum(right_vectors[r][k] * left_vectors[c][k] for k in range(3)) for c in range(3)] for r in range(3)
            ]
        # RMSD after rotation
        total = d(0)
        for i in range(n_points):
            rotated = [sum(rotation[r][k] * source_centered[i][k] for k in range(3)) for r in range(3)]
            for j in range(3):
                total += (rotated[j] - target_centered[i][j]) ** 2
        return float((total / n_points).sqrt())


def precise_umeyama(source, target):
    """High-precision Umeyama via Decimal Jacobi SVD. Returns scale factor."""
    with precise_decimal() as d:
        source_centered, target_centered, cross_covariance, n_points = _decimal_center_and_covariance(source, target)
        left_vectors, singular_values, right_vectors = _decimal_jacobi_svd_3x3(cross_covariance)
        rotation = [
            [sum(right_vectors[r][k] * left_vectors[c][k] for k in range(3)) for c in range(3)] for r in range(3)
        ]
        determinant_sign = d(1) if _decimal_determinant_3x3(rotation) >= 0 else d(-1)
        if determinant_sign < 0:
            for r in range(3):
                right_vectors[r][2] = -right_vectors[r][2]
            rotation = [
                [sum(right_vectors[r][k] * left_vectors[c][k] for k in range(3)) for c in range(3)] for r in range(3)
            ]
        source_variance = sum(source_centered[i][j] ** 2 for i in range(n_points) for j in range(3)) / n_points
        scale = (singular_values[0] + singular_values[1] + determinant_sign * singular_values[2]) / (
            n_points * source_variance
        )
        return float(scale)


KERNELS_MESH = {
    "kabsch": (baseline_kabsch, nk.kabsch, precise_kabsch),
    "umeyama": (baseline_umeyama if numpy_available else None, nk.umeyama, precise_umeyama),
    "rmsd": (baseline_rmsd if numpy_available else None, nk.rmsd, precise_rmsd),
}


def _make_point_cloud(n_points, dtype, scale=1.0):
    """Create an (n_points, 3) point cloud as an nk.Tensor in the requested dtype."""
    points_f32 = np.random.randn(n_points, 3).astype(np.float32) * scale
    points_raw, _ = downcast_f32_to_dtype(points_f32, dtype)
    return make_nk(points_raw, dtype)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("n_points", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "bfloat16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_mesh_kabsch(n_points, dtype, capability):
    """Kabsch on identical points vs SciPy procrustes: expects near-zero disparity."""
    if n_points < 3:
        pytest.skip("Kabsch requires at least 3 non-degenerate points")
    point_cloud = _make_point_cloud(n_points, dtype)
    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_MESH["kabsch"]
    result = simd_kernel(point_cloud, point_cloud)
    scipy_disparity = baseline_kernel(np.array(point_cloud, dtype="float64"), np.array(point_cloud, dtype="float64"))

    atol, rtol = tolerances_for_dtype(dtype)
    assert hasattr(result, "rotation"), "Expected MeshAlignmentResult with .rotation attribute"
    assert_allclose(float(np.array(result.scale)), 1.0, atol=atol, rtol=rtol)
    assert_allclose(float(np.array(result.rmsd)), 0.0, atol=atol, rtol=rtol)
    if scipy_disparity is not None:
        assert scipy_disparity < 0.01


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("n_points", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "bfloat16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_mesh_umeyama(n_points, dtype, capability):
    """Umeyama on 2x-scaled points: expects scale~2."""
    if n_points < 3:
        pytest.skip("Umeyama requires at least 3 non-degenerate points")
    points_f32 = np.random.randn(n_points, 3).astype(np.float32)
    original_raw, _ = downcast_f32_to_dtype(points_f32, dtype)
    original = make_nk(original_raw, dtype)
    scaled_raw, _ = downcast_f32_to_dtype(points_f32 * 2.0, dtype)
    scaled = make_nk(scaled_raw, dtype)
    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_MESH["umeyama"]
    result = simd_kernel(original, scaled)
    assert 1.0 < float(np.array(result.scale)) < 3.0


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("n_points", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "bfloat16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_mesh_rmsd(n_points, dtype, capability):
    """RMSD on identical points: expects RMSD~0, scale=1."""
    if n_points < 3:
        pytest.skip("RMSD requires at least 3 points")
    point_cloud = _make_point_cloud(n_points, dtype)
    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_MESH["rmsd"]
    result = simd_kernel(point_cloud, point_cloud)
    rmsd_value = float(np.array(result.rmsd))
    atol, rtol = tolerances_for_dtype(dtype)
    assert not np.isnan(rmsd_value), "RMSD should not be NaN for identical point clouds"
    assert_allclose(float(np.array(result.scale)), 1.0, atol=atol, rtol=rtol)
    assert_allclose(rmsd_value, 0.0, atol=atol, rtol=rtol)


@pytest.mark.parametrize("n_points", dense_dimensions)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_rmsd_self_zero(n_points, capability):
    """rmsd(cloud, cloud).rmsd ~ 0 for identical point clouds."""
    if n_points < 3:
        pytest.skip("RMSD requires at least 3 points")
    keep_one_capability(capability)
    point_cloud = nk.ones((n_points, 3), dtype="float64")
    result = nk.rmsd(point_cloud, point_cloud)
    rmsd_value = float(result.rmsd)
    assert_allclose(rmsd_value, 0.0)
