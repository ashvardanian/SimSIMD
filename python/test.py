import pytest
import numpy as np
import simsimd as simd
import scipy.spatial.distance as spd

# For normalized distances we use the absolute tolerance, because the result is close to zero.
# For unnormalized ones (like squared Euclidean or Jaccard), we use the relative.
SIMSIMD_RTOL = 0.2
SIMSIMD_ATOL = 0.15


def test_pointers_availability():
    """Tests the availability of pre-compiled functions for compatibility with USearch."""
    assert simd.pointer_to_sqeuclidean("f64") != 0
    assert simd.pointer_to_cosine("f64") != 0
    assert simd.pointer_to_inner("f64") != 0

    assert simd.pointer_to_sqeuclidean("f32") != 0
    assert simd.pointer_to_cosine("f32") != 0
    assert simd.pointer_to_inner("f32") != 0

    assert simd.pointer_to_sqeuclidean("f16") != 0
    assert simd.pointer_to_cosine("f16") != 0
    assert simd.pointer_to_inner("f16") != 0

    assert simd.pointer_to_sqeuclidean("i8") != 0
    assert simd.pointer_to_cosine("i8") != 0
    assert simd.pointer_to_inner("i8") != 0


@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [3, 97, 1536])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_dot(ndim, dtype):
    """Compares the simd.dot() function with numpy.dot(), measuring the accuracy error for f16, and f32 types."""
    np.random.seed()
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    expected = 1 - np.inner(a.astype(np.float32), b.astype(np.float32))
    result = simd.inner(a, b)

    np.testing.assert_allclose(expected, result, atol=SIMSIMD_ATOL, rtol=0)


@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [3, 97, 1536])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_sqeuclidean(ndim, dtype):
    """Compares the simd.sqeuclidean() function with scipy.spatial.distance.sqeuclidean(), measuring the accuracy error for f16, and f32 types."""
    np.random.seed()
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)

    expected = spd.sqeuclidean(a.astype(np.float32), b.astype(np.float32))
    result = simd.sqeuclidean(a, b)

    np.testing.assert_allclose(expected, result, atol=0, rtol=SIMSIMD_RTOL)


@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [3, 97, 1536])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_cosine(ndim, dtype):
    """Compares the simd.cosine() function with scipy.spatial.distance.cosine(), measuring the accuracy error for f16, and f32 types."""
    np.random.seed()
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)

    expected = spd.cosine(a.astype(np.float32), b.astype(np.float32))
    result = simd.cosine(a, b)

    np.testing.assert_allclose(expected, result, atol=SIMSIMD_ATOL, rtol=0)


@pytest.mark.skip
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [3, 97, 1536])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_jensen_shannon(ndim, dtype):
    """Compares the simd.jensenshannon() function with scipy.spatial.distance.jensenshannon(), measuring the accuracy error for f16, and f32 types."""
    np.random.seed()
    a = np.abs(np.random.randn(ndim)).astype(dtype)
    b = np.abs(np.random.randn(ndim)).astype(dtype)
    a /= np.sum(a)
    b /= np.sum(b)

    expected = spd.jensenshannon(a, b) ** 2
    result = simd.jensenshannon(a, b)

    np.testing.assert_allclose(expected, result, atol=SIMSIMD_ATOL, rtol=0)


@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [3, 97, 1536])
def test_cosine_i8(ndim):
    """Compares the simd.cosine() function with scipy.spatial.distance.cosine(), measuring the accuracy error for 8-bit int types."""
    np.random.seed()
    a = np.random.randint(0, 100, size=ndim, dtype=np.int8)
    b = np.random.randint(0, 100, size=ndim, dtype=np.int8)

    expected = spd.cosine(a.astype(np.float32), b.astype(np.float32))
    result = simd.cosine(a, b)

    np.testing.assert_allclose(expected, result, atol=SIMSIMD_ATOL, rtol=0)


@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [3, 97, 1536])
def test_sqeuclidean_i8(ndim):
    """Compares the simd.sqeuclidean() function with scipy.spatial.distance.sqeuclidean(), measuring the accuracy error for 8-bit int types."""
    np.random.seed()
    a = np.random.randint(0, 100, size=ndim, dtype=np.int8)
    b = np.random.randint(0, 100, size=ndim, dtype=np.int8)

    expected = spd.sqeuclidean(a.astype(np.float32), b.astype(np.float32))
    result = simd.sqeuclidean(a, b)

    np.testing.assert_allclose(expected, result, atol=0, rtol=SIMSIMD_RTOL)


@pytest.mark.parametrize("ndim", [3, 97, 1536])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_cosine_zero_vector(ndim, dtype):
    """Tests the simd.cosine() function with zero vectors, to catch division by zero errors."""
    np.random.seed()
    a = np.zeros(ndim, dtype=dtype)
    b = np.random.randn(ndim).astype(dtype)

    # SciPy raises: "RuntimeWarning: invalid value encountered in scalar divide"
    with pytest.raises(RuntimeWarning):
        expected = spd.cosine(a, b)

    expected = 1
    result = simd.cosine(a, b)

    assert result == expected, f"Expected {expected}, but got {result}"


@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [3, 97, 1536])
def test_hamming(ndim):
    """Compares the simd.hamming() function with scipy.spatial.distance.hamming."""
    np.random.seed()
    a = np.random.randint(2, size=ndim).astype(np.uint8)
    b = np.random.randint(2, size=ndim).astype(np.uint8)

    expected = spd.hamming(a, b) * ndim
    result = simd.hamming(np.packbits(a), np.packbits(b))

    np.testing.assert_allclose(expected, result, atol=0, rtol=SIMSIMD_RTOL)


@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [3, 97, 1536])
def test_jaccard(ndim):
    """Compares the simd.jaccard() function with scipy.spatial.distance.jaccard."""
    np.random.seed()
    a = np.random.randint(2, size=ndim).astype(np.uint8)
    b = np.random.randint(2, size=ndim).astype(np.uint8)

    expected = spd.jaccard(a, b)
    result = simd.jaccard(np.packbits(a), np.packbits(b))

    np.testing.assert_allclose(expected, result, atol=SIMSIMD_ATOL, rtol=0)


@pytest.mark.parametrize("ndim", [3, 97, 1536])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_batch(ndim, dtype):
    """Compares the simd.simd.sqeuclidean() function with scipy.spatial.distance.sqeuclidean() for a batch of vectors, measuring the accuracy error for f16, and f32 types."""
    np.random.seed()

    # Distance between matrixes A (N x D scalars) and B (N x D scalars) is an array with N floats.
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.random.randn(10, ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[i], B[i]) for i in range(10)]
    result_simd = simd.sqeuclidean(A, B)
    assert np.allclose(result_simd, result_np, atol=0, rtol=SIMSIMD_RTOL)

    # Distance between matrixes A (N x D scalars) and B (1 x D scalars) is an array with N floats.
    B = np.random.randn(1, ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[i], B[0]) for i in range(10)]
    result_simd = simd.sqeuclidean(A, B)
    assert np.allclose(result_simd, result_np, atol=0, rtol=SIMSIMD_RTOL)

    # Distance between matrixes A (1 x D scalars) and B (N x D scalars) is an array with N floats.
    A = np.random.randn(1, ndim).astype(dtype)
    B = np.random.randn(10, ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[0], B[i]) for i in range(10)]
    result_simd = simd.sqeuclidean(A, B)
    assert np.allclose(result_simd, result_np, atol=0, rtol=SIMSIMD_RTOL)

    # Distance between matrix A (N x D scalars) and array B (D scalars) is an array with N floats.
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.random.randn(ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[i], B) for i in range(10)]
    result_simd = simd.sqeuclidean(A, B)
    assert np.allclose(result_simd, result_np, atol=0, rtol=SIMSIMD_RTOL)

    # Distance between matrix B (N x D scalars) and array A (D scalars) is an array with N floats.
    B = np.random.randn(10, ndim).astype(dtype)
    A = np.random.randn(ndim).astype(dtype)
    result_np = [spd.sqeuclidean(B[i], A) for i in range(10)]
    result_simd = simd.sqeuclidean(B, A)
    assert np.allclose(result_simd, result_np, atol=0, rtol=SIMSIMD_RTOL)


@pytest.mark.parametrize("ndim", [3, 97, 1536])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("metric", ["cosine"])
def test_cdist(ndim, dtype, metric):
    """Compares the simd.cdist() function with scipy.spatial.distance.cdist(), measuring the accuracy error for f16, and f32 types using sqeuclidean and cosine metrics."""
    np.random.seed()

    # Create random matrices A (M x D) and B (N x D).
    M, N = 10, 15  # or any other sizes you deem appropriate
    A = np.random.randn(M, ndim).astype(dtype)
    B = np.random.randn(N, ndim).astype(dtype)

    # Compute cdist using scipy.
    expected = spd.cdist(A, B, metric)

    # Compute cdist using simd.
    result = simd.cdist(A, B, metric=metric)

    # Assert they're close.
    np.testing.assert_allclose(expected, result, atol=SIMSIMD_ATOL, rtol=0)
