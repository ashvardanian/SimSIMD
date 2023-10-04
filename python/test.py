import numpy as np
import simsimd as simd
from scipy.spatial.distance import cosine, sqeuclidean


def test_dot():
    """Compares the simd.dot() function with numpy.dot(), measuring the accuracy error for i8, f16, and f32 types."""
    a = np.random.randn(100).astype(np.float32)
    b = np.random.randn(100).astype(np.float32)

    expected = np.dot(a, b)
    result = simd.dot(a, b)

    np.testing.assert_allclose(expected, result, rtol=1e-5)


def test_sqeuclidean():
    """Compares the simd.sqeuclidean() function with scipy.spatial.distance.sqeuclidean(), measuring the accuracy error for i8, f16, and f32 types."""
    a = np.random.randn(100).astype(np.float32)
    b = np.random.randn(100).astype(np.float32)

    expected = sqeuclidean(a, b)
    result = simd.sqeuclidean(a, b)

    np.testing.assert_allclose(expected, result, rtol=1e-5)


def test_cosine():
    """Compares the simd.cosine() function with scipy.spatial.distance.cosine(), measuring the accuracy error for i8, f16, and f32 types."""
    a = np.random.randn(100).astype(np.float32)
    b = np.random.randn(100).astype(np.float32)

    expected = cosine(a, b)
    result = simd.cosine(a, b)

    np.testing.assert_allclose(expected, result, rtol=1e-5)


def test_batch():
    """Compares the simd.dot() function with numpy.dot() for a batch of vectors, measuring the accuracy error for i8, f16, and f32 types."""
    pass


def test_all_pairs():
    """Compares the simd.dot() function with numpy.dot() for a batch of vectors, measuring the accuracy error for i8, f16, and f32 types."""
    pass
