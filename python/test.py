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
    baseline_intersect = lambda x, y: len(np.intersect1d(x, y))
    baseline_bilinear = lambda x, y, z: x @ z @ y
    baseline_mahalanobis = lambda x, y, z: spd.mahalanobis(x, y, z) ** 2

except:
    # SciPy is not installed, some tests will be skipped
    scipy_available = False
    baseline_cosine = lambda x, y: 1.0 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    baseline_sqeuclidean = lambda x, y: np.sum((x - y) ** 2)
    baseline_jensenshannon = lambda p, q: np.sqrt((np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / 2)
    baseline_hamming = lambda x, y: np.logical_xor(x, y).sum()
    baseline_bilinear = lambda x, y, z: x @ z @ y

    def baseline_mahalanobis(x, y, z):
        diff = x - y
        return diff @ z @ diff

    def baseline_jaccard(x, y):
        intersection = np.logical_and(x, y).sum()
        union = np.logical_or(x, y).sum()
        return 0.0 if union == 0 else 1.0 - float(intersection) / float(union)

    def baseline_intersect(arr1, arr2):
        i, j, intersection = 0, 0, 0
        while i < len(arr1) and j < len(arr2):
            if arr1[i] == arr2[j]:
                intersection += 1
                i += 1
                j += 1
            elif arr1[i] < arr2[j]:
                i += 1
            else:
                j += 1
        return intersection


def is_running_under_qemu():
    return "SIMSIMD_IN_QEMU" in os.environ


# For normalized distances we use the absolute tolerance, because the result is close to zero.
# For unnormalized ones (like squared Euclidean or Jaccard), we use the relative.
SIMSIMD_RTOL = 0.1
SIMSIMD_ATOL = 0.1


def f32_rounded_and_downcasted_to_bf16(array):
    """Converts an array of 32-bit floats into 16-bit brain-floats."""
    # NumPy doesn't natively support brain-float, so we need a trick!
    # Luckily, it's very easy to reduce the representation accuracy
    # by simply masking the low 16-bits of our 32-bit single-precision
    # numbers. We can also add `0x8000` to round the numbers.
    array_f32rounded = ((array.view(np.uint32) + 0x8000) & 0xFFFF0000).view(np.float32)
    # To represent them as brain-floats, we need to drop the second halfs
    array_bf16 = np.right_shift(array_f32rounded.view(np.uint32), 16).astype(np.uint16)
    return array_f32rounded, array_bf16


def hex_array(arr):
    """Converts numerical array into a string of comma-separated hexadecimal values for debugging."""
    printer = np.vectorize(hex)
    strings = printer(arr)
    return ", ".join(strings)


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
    assert "neon_f16" in simd.get_capabilities()
    assert "neon_bf16" in simd.get_capabilities()
    assert "neon_i8" in simd.get_capabilities()
    assert "sve" in simd.get_capabilities()
    assert "sve_f16" in simd.get_capabilities()
    assert "sve_bf16" in simd.get_capabilities()
    assert "sve_i8" in simd.get_capabilities()
    assert "haswell" in simd.get_capabilities()
    assert "ice" in simd.get_capabilities()
    assert "skylake" in simd.get_capabilities()
    assert "genoa" in simd.get_capabilities()
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
@pytest.mark.parametrize(
    "kernels",
    [
        (np.inner, simd.inner),
        (baseline_sqeuclidean, simd.sqeuclidean),
        (baseline_cosine, simd.cosine),
    ],
)
def test_dense(ndim, dtype, kernels):
    """Compares various SIMD kernels (like Dot-products, squared Euclidean, and Cosine distances)
    with their NumPy or baseline counterparts, testing accuracy for IEEE standard floating-point types."""

    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)

    baseline_kernel, simd_kernel = kernels
    expected = baseline_kernel(a, b).astype(np.float32)
    result = simd_kernel(a, b)

    np.testing.assert_allclose(result, expected, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [11, 16, 33])
@pytest.mark.parametrize(
    "dtype",
    [
        "float64",
        "float32",
        "float16",
    ],
)
@pytest.mark.parametrize(
    "kernels",
    [
        (baseline_bilinear, simd.bilinear),
        (baseline_mahalanobis, simd.mahalanobis),
    ],
)
def test_curved(ndim, dtype, kernels):
    """Compares various SIMD kernels (like Bilinear Forms and Mahalanobis distances) for curved spaces
    with their NumPy or baseline counterparts, testing accuracy for IEEE standard floating-point types."""

    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()

    # Let's generate some non-negative probability distirbutions
    a = np.abs(np.random.randn(ndim).astype(dtype))
    b = np.abs(np.random.randn(ndim).astype(dtype))
    a /= np.sum(a)
    b /= np.sum(b)

    # Let's compute the inverse of the covariance matrix, otherwise in the SciPy
    # implementation of the Mahalanobis we may face `sqrt` of a negative number.
    # We multiply the matrix by its transpose to get a positive-semi-definite matrix.
    c = np.abs(np.random.randn(ndim, ndim).astype(dtype))
    c = np.dot(c, c.T)

    simd.disable_capability("sapphire")

    baseline_kernel, simd_kernel = kernels
    expected = baseline_kernel(a, b, c).astype(np.float32)
    result = simd_kernel(a, b, c)

    np.testing.assert_allclose(result, expected, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize(
    "kernels",
    [
        (np.inner, simd.inner),
        (baseline_sqeuclidean, simd.sqeuclidean),
        (baseline_cosine, simd.cosine),
    ],
)
def test_dense_bf16(ndim, kernels):
    """Compares various SIMD kernels (like Dot-products, squared Euclidean, and Cosine distances)
    with their NumPy or baseline counterparts, testing accuracy for the Brain-float format not
    natively supported by NumPy."""
    np.random.seed()
    a = np.random.randn(ndim).astype(np.float32)
    b = np.random.randn(ndim).astype(np.float32)

    a_f32rounded, a_bf16 = f32_rounded_and_downcasted_to_bf16(a)
    b_f32rounded, b_bf16 = f32_rounded_and_downcasted_to_bf16(b)

    baseline_kernel, simd_kernel = kernels
    expected = baseline_kernel(a_f32rounded, b_f32rounded)
    result = simd_kernel(a_bf16, b_bf16, "bf16")

    np.testing.assert_allclose(
        result,
        expected,
        atol=SIMSIMD_ATOL,
        rtol=SIMSIMD_RTOL,
        err_msg=f"""
        First `f32` operand in hex:     {hex_array(a_f32rounded.view(np.uint32))}
        Second `f32` operand in hex:    {hex_array(b_f32rounded.view(np.uint32))}
        First `bf16` operand in hex:    {hex_array(a_bf16)}
        Second `bf16` operand in hex:   {hex_array(b_bf16)}
        """,
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize(
    "kernels",
    [
        (np.inner, simd.inner),
        (baseline_sqeuclidean, simd.sqeuclidean),
        (baseline_cosine, simd.cosine),
    ],
)
def test_dense_i8(ndim, kernels):
    """Compares various SIMD kernels (like Dot-products, squared Euclidean, and Cosine distances)
    with their NumPy or baseline counterparts, testing accuracy for small integer types, that can't
    be directly processed with other tools without overflowing."""

    np.random.seed()
    a = np.random.randint(-128, 127, size=(ndim), dtype=np.int8)
    b = np.random.randint(-128, 127, size=(ndim), dtype=np.int8)

    baseline_kernel, simd_kernel = kernels

    # Fun fact: SciPy doesn't actually raise an `OverflowError` when overflow happens
    # here, instead it raises `ValueError: math domain error` during the `sqrt` operation.
    try:
        expected_overflow = baseline_kernel(a, b)
    except OverflowError:
        expected_overflow = OverflowError()
    except ValueError:
        expected_overflow = ValueError()
    expected = baseline_kernel(a.astype(np.float64), b.astype(np.float64))
    result = simd_kernel(a, b)

    assert int(result) == int(expected), f"Expected {expected}, but got {result} (overflow: {expected_overflow})"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.repeat(50)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("kernels", [(baseline_hamming, simd.hamming), (baseline_jaccard, simd.jaccard)])
def test_dense_bits(ndim, kernels):
    """Compares various SIMD kernels (like Hamming and Jaccard/Tanimoto distances) for dense bit arrays
    with their NumPy or baseline counterparts, even though, they can't process sub-byte-sized scalars."""
    np.random.seed()
    a = np.random.randint(2, size=ndim).astype(np.uint8)
    b = np.random.randint(2, size=ndim).astype(np.uint8)

    baseline_kernel, simd_kernel = kernels
    expected = baseline_kernel(a, b)
    result = simd_kernel(np.packbits(a), np.packbits(b), "b8")

    np.testing.assert_allclose(result, expected, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)


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

    np.testing.assert_allclose(result, expected, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_cosine_zero_vector(ndim, dtype):
    """Tests the simd.cosine() function with zero vectors, to catch division by zero errors."""
    a = np.zeros(ndim, dtype=dtype)
    b = (np.random.randn(ndim) + 1).astype(dtype)

    result = simd.cosine(a, b)
    assert result == 1, f"Expected 1, but got {result}"

    result = simd.cosine(a, a)
    assert result == 0, f"Expected 0 distance from itself, but got {result}"

    result = simd.cosine(b, b)
    assert abs(result) < SIMSIMD_ATOL, f"Expected 0 distance from itself, but got {result}"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_cosine_tolerance(ndim, dtype):
    """Tests the simd.cosine() function analyzing its `rsqrt` approximation error."""
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)

    expected_f64 = baseline_cosine(a.astype(np.float64), b.astype(np.float64))
    result_f64 = simd.cosine(a, b)
    expected = np.array(expected_f64, dtype=dtype)
    result = np.array(result_f64, dtype=dtype)
    assert np.allclose(expected, result, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)


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

    np.testing.assert_allclose(result, expected, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)

    expected = np.vdot(a, b)
    result = simd.vdot(a, b)

    np.testing.assert_allclose(result, expected, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)


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

    np.testing.assert_allclose(result, expected, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)

    expected = np.vdot(a.view(np.complex64), b.view(np.complex64))
    result = simd.vdot(a, b, "complex64")

    np.testing.assert_allclose(result, expected, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(200)
@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_intersect(dtype):
    """Compares the simd.intersect() function with numpy.intersect1d."""
    np.random.seed()
    a_length = np.random.randint(1, 1024)
    b_length = np.random.randint(1, 1024)
    a = np.random.randint(2048, size=a_length, dtype=dtype)
    b = np.random.randint(2048, size=b_length, dtype=dtype)

    # Remove duplicates, converting into sorted arrays
    a = np.unique(a)
    b = np.unique(b)

    expected = baseline_intersect(a, b)
    result = simd.intersect(a, b)

    assert int(expected) == int(result)


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
    result_simd = np.array(simd.sqeuclidean(A, B)).astype(np.float32)
    assert np.allclose(result_simd, result_np, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)

    # Distance between matrixes A (N x D scalars) and B (1 x D scalars) is an array with N floats.
    B = np.random.randn(1, ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[i], B[0]) for i in range(10)]
    result_simd = np.array(simd.sqeuclidean(A, B)).astype(np.float32)
    assert np.allclose(result_simd, result_np, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)

    # Distance between matrixes A (1 x D scalars) and B (N x D scalars) is an array with N floats.
    A = np.random.randn(1, ndim).astype(dtype)
    B = np.random.randn(10, ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[0], B[i]) for i in range(10)]
    result_simd = np.array(simd.sqeuclidean(A, B)).astype(np.float32)
    assert np.allclose(result_simd, result_np, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)

    # Distance between matrix A (N x D scalars) and array B (D scalars) is an array with N floats.
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.random.randn(ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[i], B) for i in range(10)]
    result_simd = np.array(simd.sqeuclidean(A, B)).astype(np.float32)
    assert np.allclose(result_simd, result_np, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)

    # Distance between matrix B (N x D scalars) and array A (D scalars) is an array with N floats.
    B = np.random.randn(10, ndim).astype(dtype)
    A = np.random.randn(ndim).astype(dtype)
    result_np = [spd.sqeuclidean(B[i], A) for i in range(10)]
    result_simd = np.array(simd.sqeuclidean(B, A)).astype(np.float32)
    assert np.allclose(result_simd, result_np, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("input_dtype", ["float32", "float16"])
@pytest.mark.parametrize("result_dtype", [None, "float32", "float16", "int8"])
@pytest.mark.parametrize("metric", ["cosine"])
def test_cdist(ndim, input_dtype, result_dtype, metric):
    """Compares the simd.cdist() function with scipy.spatial.distance.cdist(), measuring the accuracy error for f16, and f32 types using sqeuclidean and cosine metrics."""

    if input_dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()

    # Create random matrices A (M x D) and B (N x D).
    M, N = 10, 15  # or any other sizes you deem appropriate
    A = np.random.randn(M, ndim).astype(input_dtype)
    B = np.random.randn(N, ndim).astype(input_dtype)

    if result_dtype is None:
        # Compute cdist using scipy.
        expected = spd.cdist(A, B, metric)

        # Compute cdist using simd.
        result = simd.cdist(A, B, metric=metric)
    else:
        # Compute cdist using scipy.
        expected = spd.cdist(A, B, metric).astype(result_dtype)

        # Compute cdist using simd.
        result = simd.cdist(A, B, metric=metric, dtype=result_dtype)

    # Assert they're close.
    np.testing.assert_allclose(result, expected, atol=SIMSIMD_ATOL, rtol=SIMSIMD_RTOL)


if __name__ == "__main__":
    pytest.main()
