import os
import pytest
import simsimd as simd

# NumPy is available on most platforms and is required for most tests.
# When using PyPy on some platforms NumPy has internal issues, that will
# raise a weird error, not an `ImportError`. That's why we intentionally
# use a naked `except:`. Necessary evil!
try:
    import numpy as np

    numpy_available = True
except:
    # NumPy is not installed, most tests will be skipped
    numpy_available = False

# At the time of Python 3.12, SciPy doesn't support 32-bit Windows on any CPU,
# or 64-bit Windows on Arm. It also doesn't support `musllinux` distributions,
# like CentOS, RedHat OS, and many others.
try:
    import scipy.spatial.distance as spd

    scipy_available = True
    baseline_sqeuclidean = spd.sqeuclidean
    baseline_cosine = spd.cosine
    baseline_jensenshannon = spd.jensenshannon
    baseline_hamming = lambda x, y: spd.hamming(x, y) * len(x)
    baseline_jaccard = spd.jaccard

except:
    # SciPy is not installed, some tests will be skipped
    scipy_available = False
    baseline_cosine = lambda x, y: 1.0 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    baseline_sqeuclidean = lambda x, y: np.sum((x - y) ** 2)
    baseline_jensenshannon = lambda p, q: np.sqrt((np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / 2)
    baseline_hamming = lambda x, y: np.logical_xor(x, y).sum()

    def baseline_jaccard(x, y):
        intersection = np.logical_and(x, y).sum()
        union = np.logical_or(x, y).sum()
        return 0.0 if union == 0 else 1.0 - float(intersection) / float(union)


def is_running_under_qemu():
    return "SIMSIMD_IN_QEMU" in os.environ


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


def test_capabilities_list():
    """Tests the visibility of hardware capabilities."""
    assert "serial" in simd.get_capabilities()
    assert "neon" in simd.get_capabilities()
    assert "sve" in simd.get_capabilities()
    assert "sve2" in simd.get_capabilities()
    assert "haswell" in simd.get_capabilities()
    assert "ice" in simd.get_capabilities()
    assert "skylake" in simd.get_capabilities()
    assert "sapphire" in simd.get_capabilities()
    assert simd.get_capabilities().get("serial") == 1

    # Check the toggle:
    previous_value = simd.get_capabilities().get("neon")
    simd.enable_capability("neon")
    assert simd.get_capabilities().get("neon") == 1
    if not previous_value:
        simd.disable_capability("neon")


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_dot(ndim, dtype):
    """Compares the simd.dot() function with numpy.dot(), measuring the accuracy error for f64 and f32 types."""

    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    expected = np.inner(a, b).astype(np.float32)
    result = simd.inner(a, b)

    np.testing.assert_allclose(expected, result, atol=SIMSIMD_ATOL, rtol=0)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_sqeuclidean(ndim, dtype):
    """Compares the simd.sqeuclidean() function with scipy.spatial.distance.sqeuclidean(), measuring the accuracy error for f16, and f32 types."""

    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)

    expected = baseline_sqeuclidean(a, b).astype(np.float32)
    result = simd.sqeuclidean(a, b)

    np.testing.assert_allclose(expected, result, atol=0, rtol=SIMSIMD_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_cosine(ndim, dtype):
    """Compares the simd.cosine() function with scipy.spatial.distance.cosine(), measuring the accuracy error for f16, and f32 types."""

    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed(0)  # Use a fixed seed for reproducibility
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)

    expected = baseline_cosine(a, b).astype(np.float32)
    result = simd.cosine(a, b)

    np.testing.assert_allclose(expected, result, atol=SIMSIMD_ATOL, rtol=0)


@pytest.mark.skip(reason="Problems inferring the tolerance bounds for numerical errors")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_jensen_shannon(ndim, dtype):
    """Compares the simd.jensenshannon() function with scipy.spatial.distance.jensenshannon(), measuring the accuracy error for f16, and f32 types."""
    np.random.seed()
    a = np.abs(np.random.randn(ndim)).astype(dtype)
    b = np.abs(np.random.randn(ndim)).astype(dtype)
    a /= np.sum(a)
    b /= np.sum(b)

    expected = baseline_jensenshannon(a, b) ** 2
    result = simd.jensenshannon(a, b)

    np.testing.assert_allclose(expected, result, atol=SIMSIMD_ATOL, rtol=0)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
def test_cosine_i8(ndim):
    """Compares the simd.cosine() function with scipy.spatial.distance.cosine(), measuring the accuracy error for 8-bit int types."""
    np.random.seed()
    a = np.random.randint(0, 100, size=ndim, dtype=np.int8)
    b = np.random.randint(0, 100, size=ndim, dtype=np.int8)

    expected = baseline_cosine(a.astype(np.float32), b.astype(np.float32))
    result = simd.cosine(a, b)

    np.testing.assert_allclose(expected, result, atol=SIMSIMD_ATOL, rtol=0)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
def test_sqeuclidean_i8(ndim):
    """Compares the simd.sqeuclidean() function with scipy.spatial.distance.sqeuclidean(), measuring the accuracy error for 8-bit int types."""
    np.random.seed()
    a = np.random.randint(0, 100, size=ndim, dtype=np.int8)
    b = np.random.randint(0, 100, size=ndim, dtype=np.int8)

    expected = baseline_sqeuclidean(a.astype(np.float32), b.astype(np.float32))
    result = simd.sqeuclidean(a, b)

    np.testing.assert_allclose(expected, result, atol=0, rtol=SIMSIMD_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_cosine_zero_vector(ndim, dtype):
    """Tests the simd.cosine() function with zero vectors, to catch division by zero errors."""
    a = np.zeros(ndim, dtype=dtype)
    b = np.random.randn(ndim).astype(dtype)

    expected = 1
    result = simd.cosine(a, b)

    assert result == expected, f"Expected {expected}, but got {result}"


@pytest.mark.skipif(is_running_under_qemu(), reason="Complex math in QEMU fails")
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [22, 66, 1536])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dot_complex(ndim, dtype):
    """Compares the simd.dot() and simd.vdot() against NumPy for complex numbers."""
    np.random.seed()
    dtype_view = np.complex64 if dtype == "float32" else np.complex128
    a = np.random.randn(ndim).astype(dtype=dtype).view(dtype_view)
    b = np.random.randn(ndim).astype(dtype=dtype).view(dtype_view)

    expected = np.dot(a, b)
    result = simd.dot(a, b)

    np.testing.assert_allclose(expected, result, atol=0, rtol=SIMSIMD_RTOL)

    expected = np.vdot(a, b)
    result = simd.vdot(a, b)

    np.testing.assert_allclose(expected, result, atol=0, rtol=SIMSIMD_RTOL)


@pytest.mark.skipif(is_running_under_qemu(), reason="Complex math in QEMU fails")
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [22, 66, 1536])
def test_dot_complex_explicit(ndim):
    """Compares the simd.dot() and simd.vdot() against NumPy for complex numbers."""
    np.random.seed()
    a = np.random.randn(ndim).astype(dtype=np.float32)
    b = np.random.randn(ndim).astype(dtype=np.float32)

    expected = np.dot(a.view(np.complex64), b.view(np.complex64))
    result = simd.dot(a, b, "complex64")

    np.testing.assert_allclose(expected, result, atol=0, rtol=SIMSIMD_RTOL)

    expected = np.vdot(a.view(np.complex64), b.view(np.complex64))
    result = simd.vdot(a, b, "complex64")

    np.testing.assert_allclose(expected, result, atol=0, rtol=SIMSIMD_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
def test_hamming(ndim):
    """Compares the simd.hamming() function with scipy.spatial.distance.hamming."""
    np.random.seed()
    a = np.random.randint(2, size=ndim).astype(np.uint8)
    b = np.random.randint(2, size=ndim).astype(np.uint8)

    expected = baseline_hamming(a, b)
    result = simd.hamming(np.packbits(a), np.packbits(b))

    np.testing.assert_allclose(expected, result, atol=0, rtol=SIMSIMD_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
def test_jaccard(ndim):
    """Compares the simd.jaccard() function with scipy.spatial.distance.jaccard."""
    np.random.seed()
    a = np.random.randint(2, size=ndim).astype(np.uint8)
    b = np.random.randint(2, size=ndim).astype(np.uint8)

    expected = baseline_jaccard(a, b)
    result = simd.jaccard(np.packbits(a), np.packbits(b))

    np.testing.assert_allclose(expected, result, atol=SIMSIMD_ATOL, rtol=0)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_batch(ndim, dtype):
    """Compares the simd.simd.sqeuclidean() function with scipy.spatial.distance.sqeuclidean() for a batch of vectors, measuring the accuracy error for f16, and f32 types."""

    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

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


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("metric", ["cosine"])
def test_cdist(ndim, dtype, metric):
    """Compares the simd.cdist() function with scipy.spatial.distance.cdist(), measuring the accuracy error for f16, and f32 types using sqeuclidean and cosine metrics."""

    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

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


if __name__ == "__main__":
    pytest.main()
