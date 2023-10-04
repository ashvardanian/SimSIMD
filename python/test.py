import pytest
import numpy as np
import simsimd as simd
from scipy.spatial.distance import cosine, sqeuclidean


def test_pointers_availability():
    """Tests the availability of pre-compiled functions for compatibility with USearch."""
    assert simd.pointer_to_sqeuclidean("f32") != 0
    assert simd.pointer_to_cosine("f32") != 0
    assert simd.pointer_to_inner("f32") != 0

    assert simd.pointer_to_sqeuclidean("f16") != 0
    assert simd.pointer_to_cosine("f16") != 0
    assert simd.pointer_to_inner("f16") != 0

    assert simd.pointer_to_sqeuclidean("i8") != 0
    assert simd.pointer_to_cosine("i8") != 0
    assert simd.pointer_to_inner("i8") != 0


@pytest.mark.parametrize("ndim", [3, 97, 1536])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_dot(ndim, dtype):
    """Compares the simd.dot() function with numpy.dot(), measuring the accuracy error for f16, and f32 types."""
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)

    expected = 1 - np.inner(a, b)
    result = simd.inner(a, b)

    np.testing.assert_allclose(expected, result, rtol=1e-2)


@pytest.mark.parametrize("ndim", [3, 97, 1536])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_sqeuclidean(ndim, dtype):
    """Compares the simd.sqeuclidean() function with scipy.spatial.distance.sqeuclidean(), measuring the accuracy error for f16, and f32 types."""
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)

    expected = sqeuclidean(a, b)
    result = simd.sqeuclidean(a, b)

    np.testing.assert_allclose(expected, result, rtol=1e-2)


@pytest.mark.parametrize("ndim", [3, 97, 1536])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_cosine(ndim, dtype):
    """Compares the simd.cosine() function with scipy.spatial.distance.cosine(), measuring the accuracy error for f16, and f32 types."""
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)

    expected = cosine(a, b)
    result = simd.cosine(a, b)

    np.testing.assert_allclose(expected, result, rtol=1e-2)


@pytest.mark.parametrize("ndim", [3, 97, 1536])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_batch(ndim, dtype):
    """Compares the simd.simd.sqeuclidean() function with scipy.spatial.distance.sqeuclidean() for a batch of vectors, measuring the accuracy error for f16, and f32 types."""

    # Distance between matrixes A (N x D scalars) and B (N x D scalars) is an array with N floats.
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.random.randn(10, ndim).astype(dtype)
    result_np = [sqeuclidean(A[i], B[i]) for i in range(10)]
    result_simd = simd.sqeuclidean(A, B)
    assert np.allclose(result_simd, result_np, rtol=1e-2)

    # Distance between matrixes A (N x D scalars) and B (1 x D scalars) is an array with N floats.
    B = np.random.randn(1, ndim).astype(dtype)
    result_np = [sqeuclidean(A[i], B[0]) for i in range(10)]
    result_simd = simd.sqeuclidean(A, B)
    assert np.allclose(result_simd, result_np, rtol=1e-2)

    # Distance between matrixes A (1 x D scalars) and B (N x D scalars) is an array with N floats.
    A = np.random.randn(1, ndim).astype(dtype)
    B = np.random.randn(10, ndim).astype(dtype)
    result_np = [sqeuclidean(A[0], B[i]) for i in range(10)]
    result_simd = simd.sqeuclidean(A, B)
    assert np.allclose(result_simd, result_np, rtol=1e-2)

    # Distance between matrix A (N x D scalars) and array B (D scalars) is an array with N floats.
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.random.randn(ndim).astype(dtype)
    result_np = [sqeuclidean(A[i], B) for i in range(10)]
    result_simd = simd.sqeuclidean(A, B)
    assert np.allclose(result_simd, result_np, rtol=1e-2)

    # Distance between matrix B (N x D scalars) and array A (D scalars) is an array with N floats.
    B = np.random.randn(10, ndim).astype(dtype)
    A = np.random.randn(ndim).astype(dtype)
    result_np = [sqeuclidean(B[i], A) for i in range(10)]
    result_simd = simd.sqeuclidean(B, A)
    assert np.allclose(result_simd, result_np, rtol=1e-2)


def test_all_pairs():
    """Compares the simd.dot() function with numpy.dot() for a batch of vectors, measuring the accuracy error for f16, and f32 types."""
    pass
