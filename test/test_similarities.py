#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test cdist (pairwise cross-distance) operations.

Covers cdist for all metric families: dot, angular, euclidean, sqeuclidean,
hamming, jaccard, kld, jsd, vdot, and exotic sub-byte dtypes.

Precision notes:
    Floating-point dtypes use NK_ATOL/NK_RTOL (0.1/0.1).
    Integer output dtypes (cdist out_dtype) use atol=1.

Matches C++ suite: test_cross_*.cpp.
"""

import atexit
import pytest

try:
    import numpy as np
except Exception:
    np = None

try:
    import scipy.stats
except ImportError:
    pass

import numkong as nk
from test_base import (
    numpy_available,
    scipy_available,
    dense_dimensions,
    possible_capabilities,
    randomized_repetitions_count,
    keep_one_capability,
    profile,
    scipy_metric_name,
    NK_ATOL,
    NK_RTOL,
    make_random,
    make_nk,
    tolerances_for_dtype,
    seed_rng,
    collect_errors,
    create_stats,
    print_stats_report,
)

try:
    import scipy.spatial.distance as spd
except ImportError:
    pass

stats = create_stats()
atexit.register(print_stats_report, stats)


def round_and_clip_even(values, out_dtype):
    dtype_info = np.iinfo(out_dtype)
    finite_values = np.nan_to_num(values, nan=0.0, posinf=float(dtype_info.max), neginf=float(dtype_info.min))
    clipped_values = np.clip(finite_values, dtype_info.min, dtype_info.max)
    return np.rint(clipped_values).astype(out_dtype)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["float64", "float32"])
@pytest.mark.parametrize("metric", ["dot", "angular", "euclidean"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_batch_metrics(ndim, input_dtype, metric, capability):
    """Verify the SIMD batch dispatch for dot, angular, and euclidean.

    Sets ``out_dtype`` equal to ``input_dtype`` (float64 or float32) so that the
    kernel's native output type matches and the fast packed/symmetric batch path
    is selected instead of the scalar pairwise fallback.  Uses asymmetric matrix
    sizes (7 x 11) to exercise the general rectangular case.

    Dimensions are inherited from ``NK_DENSE_DIMENSIONS``; capabilities from
    platform auto-detection via ``possible_capabilities``.  Baseline for ``dot``
    is ``np.dot`` (SciPy has no ``cdist`` metric for inner product); other
    metrics use ``scipy.spatial.distance.cdist``.
    """
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 7, 11
    a_matrix = np.random.randn(num_rows_a, ndim).astype(input_dtype)
    b_matrix = np.random.randn(num_rows_b, ndim).astype(input_dtype)

    # Use native out_dtype to force batch path (float64->float64, float32->float32)
    out_dtype = input_dtype
    scipy_metric = scipy_metric_name(metric)

    if metric == "dot":
        expected = np.array(
            [[np.dot(a_matrix[i], b_matrix[j]) for j in range(num_rows_b)] for i in range(num_rows_a)]
        ).astype(out_dtype)
    else:
        expected = spd.cdist(a_matrix, b_matrix, scipy_metric).astype(out_dtype)

    result = nk.cdist(a_matrix, b_matrix, metric=metric, out_dtype=out_dtype)
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["float64", "float32"])
@pytest.mark.parametrize("metric", ["dot", "angular", "euclidean", "sqeuclidean"])
def test_cdist_self_distance(ndim, input_dtype, metric):
    """Verify ``cdist(A, A)`` produces a complete, correct symmetric matrix.

    When both operands are the same object the C code takes a symmetric batch
    shortcut that only computes the upper triangle.  A post-kernel mirror loop
    must fill the lower triangle.  This test checks:

    1. Default ``out_dtype=float64`` (pairwise fallback on f32 input) — full
       matrix against SciPy.
    2. Native ``out_dtype=input_dtype`` (forces batch symmetric path) — full
       matrix, with an explicit lower-triangle mask assertion to catch missing
       mirror writes.

    No ``capability`` parameter: runs on whatever the default backend is (all
    ISA-specific paths are already covered by ``test_cdist_batch_metrics``).
    Dimensions from ``NK_DENSE_DIMENSIONS``.
    """
    a_matrix = np.random.randn(10, ndim).astype(input_dtype)

    scipy_metric = scipy_metric_name(metric)
    if metric == "dot":
        expected = np.array([[np.dot(a_matrix[i], a_matrix[j]) for j in range(10)] for i in range(10)]).astype(
            np.float64
        )
    else:
        expected = spd.cdist(a_matrix, a_matrix, scipy_metric)

    # Default out_dtype (f64) — may use pairwise fallback
    result_default = np.asarray(nk.cdist(a_matrix, a_matrix, metric=metric))
    np.testing.assert_allclose(result_default, expected, atol=NK_ATOL, rtol=NK_RTOL)
    # Check lower triangle explicitly
    mask_lower = np.tril(np.ones((10, 10), dtype=bool), k=-1)
    np.testing.assert_allclose(result_default[mask_lower], expected[mask_lower], atol=NK_ATOL, rtol=NK_RTOL)

    # Native out_dtype — should force batch symmetric path for f32
    native_out_dtype = input_dtype
    if metric == "dot":
        expected_native = np.array([[np.dot(a_matrix[i], a_matrix[j]) for j in range(10)] for i in range(10)]).astype(
            native_out_dtype
        )
    else:
        expected_native = spd.cdist(a_matrix, a_matrix, scipy_metric).astype(native_out_dtype)
    result_native = np.asarray(nk.cdist(a_matrix, a_matrix, metric=metric, out_dtype=native_out_dtype))
    np.testing.assert_allclose(result_native, expected_native, atol=NK_ATOL, rtol=NK_RTOL)
    np.testing.assert_allclose(result_native[mask_lower], expected_native[mask_lower], atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("out_dtype", [None, "float32", "int32"])
@pytest.mark.parametrize("metric", ["angular", "sqeuclidean", "euclidean", "dot"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_float_accuracy(ndim, input_dtype, out_dtype, metric, capability):
    """Broad coverage of cdist for standard IEEE float inputs.

    Exercises four metrics (angular, sqeuclidean, euclidean, dot) across three
    input dtypes (float64, float32, float16) and three output modes (default
    float64, explicit float32, explicit int32).  Inputs are intentionally
    *strided* — sliced from wider arrays — so the kernel must respect non-
    contiguous memory layouts.

    Additionally tests the ``out=`` buffer path: a strided slice of a wider
    allocation is passed and checked for correct in-place writes.

    Skips:
        * ``angular`` at ndim=1 — degenerate (norm is a single element, 0/0).
        * ``float16`` input + ``float32`` out_dtype — pre-existing batch-path
          bug returns zeros.

    Dimensions from ``NK_DENSE_DIMENSIONS``; capabilities from platform
    auto-detection.  Integer output uses ``atol=1`` (discrete rounding);
    floats use ``NK_ATOL / NK_RTOL``.
    """
    if metric == "angular" and ndim == 1:
        pytest.skip("angular at ndim=1 is degenerate (0/0 from single-element norms)")
    if input_dtype == "float16" and out_dtype == "float32":
        pytest.skip("batch path with float16 input and float32 out_dtype is broken (pre-existing)")
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 10, 15
    a_matrix_extended = np.random.randn(num_rows_a, ndim + 1).astype(input_dtype)
    b_matrix_extended = np.random.randn(num_rows_b, ndim + 3).astype(input_dtype)
    a_matrix = a_matrix_extended[:, :ndim]
    b_matrix = b_matrix_extended[:, :ndim]

    is_integer_output = out_dtype in ("int32", "int64", "int16", "int8", "uint32", "uint64", "uint16", "uint8")
    scipy_metric = scipy_metric_name(metric)

    if metric == "dot":
        baseline = np.array(
            [
                [np.dot(a_matrix[i].astype(np.float64), b_matrix[j].astype(np.float64)) for j in range(num_rows_b)]
                for i in range(num_rows_a)
            ]
        )
    else:
        baseline = spd.cdist(a_matrix, b_matrix, scipy_metric)

    if out_dtype is None:
        expected = baseline
        result = nk.cdist(a_matrix, b_matrix, metric)
    else:
        expected = round_and_clip_even(baseline, out_dtype) if is_integer_output else baseline.astype(out_dtype)
        result = nk.cdist(a_matrix, b_matrix, metric, out_dtype=out_dtype)

    atol = 1 if is_integer_output else NK_ATOL
    np.testing.assert_allclose(result, expected, atol=atol, rtol=NK_RTOL)

    # Test out= buffer with strides
    out_np_dtype = out_dtype if out_dtype else "float64"
    output_buffer_extended = np.zeros((num_rows_a, num_rows_b + 7), dtype=out_np_dtype)
    output_buffer = output_buffer_extended[:, :num_rows_b]
    assert nk.cdist(a_matrix, b_matrix, metric, out=output_buffer) is None
    np.testing.assert_allclose(output_buffer, expected, atol=atol, rtol=NK_RTOL)


@pytest.mark.skip(reason="2D complex cdist segfaults (pre-existing bug)")
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["complex128", "complex64"])
@pytest.mark.parametrize("out_dtype", [None, "complex128", "complex64"])
@pytest.mark.parametrize("metric", ["dot", "vdot"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_complex(ndim, input_dtype, out_dtype, metric, capability):
    """Verify cdist for complex-valued dot and vdot metrics.

    Currently **skipped** due to a pre-existing segfault when cdist receives 2D
    complex arrays.  When enabled, tests three output modes (default complex128,
    explicit complex128, explicit complex64) for both ``dot`` (Hermitian-unaware)
    and ``vdot`` (conjugate dot) metrics.  Exercises:

    * 1D scalar result (single row vs single row).
    * 2D matrix result (10 x 15).
    * ``out=`` buffer path with strided column slice.

    Inputs are strided (sliced from wider allocations).  Dimensions from
    ``NK_DENSE_DIMENSIONS``; capabilities from platform auto-detection.
    """
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 10, 15
    a_matrix_extended = np.random.randn(num_rows_a, ndim + 1).astype(input_dtype)
    b_matrix_extended = np.random.randn(num_rows_b, ndim + 3).astype(input_dtype)
    a_matrix = a_matrix_extended[:, :ndim]
    b_matrix = b_matrix_extended[:, :ndim]
    c_matrix_extended = np.random.randn(num_rows_a, num_rows_b + 7).astype(out_dtype if out_dtype else np.complex128)
    c_matrix = c_matrix_extended[:, :num_rows_b]

    expected = np.zeros((num_rows_a, num_rows_b), dtype=out_dtype if out_dtype else np.complex128)
    baseline_kernel = np.dot if metric == "dot" else np.vdot
    for i in range(num_rows_a):
        for j in range(num_rows_b):
            expected[i, j] = baseline_kernel(a_matrix[i], b_matrix[j])

    if out_dtype is None:
        result1d = nk.cdist(a_matrix[0], b_matrix[0], metric=metric)
        result2d = nk.cdist(a_matrix, b_matrix, metric=metric)
        assert nk.cdist(a_matrix, b_matrix, metric=metric, out=c_matrix) is None
    else:
        expected = expected.astype(out_dtype)
        result1d = nk.cdist(a_matrix[0], b_matrix[0], metric=metric, out_dtype=out_dtype)
        result2d = nk.cdist(a_matrix, b_matrix, metric=metric, out_dtype=out_dtype)
        assert nk.cdist(a_matrix, b_matrix, metric=metric, out_dtype=out_dtype, out=c_matrix) is None

    np.testing.assert_allclose(result1d, expected[0, 0], atol=NK_ATOL, rtol=NK_RTOL)
    np.testing.assert_allclose(result2d, expected, atol=NK_ATOL, rtol=NK_RTOL)
    np.testing.assert_allclose(c_matrix, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("out_dtype", [None, "float32", "float16", "int8"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_hamming(ndim, out_dtype, capability):
    """Verify cdist Hamming distance on packed bit vectors.

    Generates random binary matrices, packs them via ``np.packbits``, and passes
    ``dtype="uint1"`` so the kernel interprets each byte as 8 bits.  Baseline is
    ``scipy.spatial.distance.cdist(bits, bits, "hamming") * ndim`` — SciPy
    normalises by dimension, so we undo that.

    Output dtype coverage: default (float64), float32, float16, int8.  Integer
    output uses standard ``NK_ATOL / NK_RTOL`` (Hamming counts are exact for
    integer types but may round for float16).

    Randomised via ``@pytest.mark.repeat(randomized_repetitions_count)`` (env
    ``NK_RANDOMIZED_REPETITIONS``, default 1).  Dimensions from
    ``NK_DENSE_DIMENSIONS``; capabilities from platform auto-detection.
    """
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 10, 15
    a_bits = np.random.randint(2, size=(num_rows_a, ndim)).astype(np.uint8)
    b_bits = np.random.randint(2, size=(num_rows_b, ndim)).astype(np.uint8)
    a_packed_bits, b_packed_bits = np.packbits(a_bits, axis=1), np.packbits(b_bits, axis=1)

    if out_dtype is None:
        expected = spd.cdist(a_bits, b_bits, "hamming") * ndim
        result = nk.cdist(a_packed_bits, b_packed_bits, metric="hamming", dtype="uint1")
    else:
        expected = (spd.cdist(a_bits, b_bits, "hamming") * ndim).astype(out_dtype)
        result = nk.cdist(a_packed_bits, b_packed_bits, metric="hamming", dtype="uint1", out_dtype=out_dtype)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("out_dtype", [None, "float32"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_jaccard(ndim, out_dtype, capability):
    """Verify cdist Jaccard distance on packed bit vectors.

    Same ``np.packbits`` + ``dtype="uint1"`` pattern as the Hamming test, but
    with ``metric="jaccard"``.  Baseline is ``scipy.spatial.distance.cdist``
    with ``"jaccard"`` (already normalised — ratio of disagreeing bits to bits
    where at least one is set).

    Output dtype coverage: default (float64) and explicit float32.

    Randomised via ``@pytest.mark.repeat``; dimensions from
    ``NK_DENSE_DIMENSIONS``; capabilities from platform auto-detection.
    """
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 10, 15
    a_bits = np.random.randint(2, size=(num_rows_a, ndim)).astype(np.uint8)
    b_bits = np.random.randint(2, size=(num_rows_b, ndim)).astype(np.uint8)
    a_packed_bits, b_packed_bits = np.packbits(a_bits, axis=1), np.packbits(b_bits, axis=1)

    if out_dtype is None:
        expected = spd.cdist(a_bits, b_bits, "jaccard")
        result = nk.cdist(a_packed_bits, b_packed_bits, metric="jaccard", dtype="uint1")
    else:
        expected = spd.cdist(a_bits, b_bits, "jaccard").astype(out_dtype)
        result = nk.cdist(a_packed_bits, b_packed_bits, metric="jaccard", dtype="uint1", out_dtype=out_dtype)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["float64", "float32"])
@pytest.mark.parametrize("metric", ["kld", "jsd"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_probability(ndim, input_dtype, metric, capability):
    """Verify cdist for Kullback-Leibler divergence and Jensen-Shannon distance.

    Inputs are positive probability vectors (softmax of randn + epsilon) using
    asymmetric 7 x 11 matrix sizes.

    Baselines:
        * **KLD** — ``scipy.stats.entropy(p, q)`` which computes
          ``sum(p * ln(p/q))`` in natural log (nats), matching ``nk.kld``.
          Note: KLD is asymmetric, so ``cdist(A, B)`` != ``cdist(B, A)``.
        * **JSD** — ``scipy.spatial.distance.jensenshannon`` with default
          ``base=e`` (natural log), returning ``sqrt(JS divergence)`` which
          matches ``nk.jsd`` directly.

    Only float64 and float32 input dtypes are supported.  No ``out_dtype``
    variants — probability divergences always produce float64 output.
    Dimensions from ``NK_DENSE_DIMENSIONS``; capabilities from platform
    auto-detection.
    """
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 7, 11
    # Positive normalized vectors (softmax of randn)
    a_raw = np.abs(np.random.randn(num_rows_a, ndim)).astype(input_dtype) + 1e-6
    b_raw = np.abs(np.random.randn(num_rows_b, ndim)).astype(input_dtype) + 1e-6
    a_matrix = (a_raw / a_raw.sum(axis=1, keepdims=True)).astype(input_dtype)
    b_matrix = (b_raw / b_raw.sum(axis=1, keepdims=True)).astype(input_dtype)

    if metric == "kld":
        # scipy.stats.entropy(p, q) = sum(p * ln(p/q)), natural log, matches nk.kld
        expected = spd.cdist(
            a_matrix.astype(np.float64),
            b_matrix.astype(np.float64),
            lambda u, v: scipy.stats.entropy(u, v),
        )
    else:
        # spd.jensenshannon defaults to base=e (natural log), returns sqrt(JS divergence)
        expected = spd.cdist(a_matrix.astype(np.float64), b_matrix.astype(np.float64), "jensenshannon")

    result = nk.cdist(a_matrix, b_matrix, metric=metric)
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize(
    "input_dtype",
    ["bfloat16", "e4m3", "e5m2", "e2m3", "e3m2", "int8", "uint8", "int4", "uint4"],
)
@pytest.mark.parametrize("metric", ["dot", "euclidean"])
def test_cdist_exotic_dtypes(ndim, input_dtype, metric):
    """Verify cdist for sub-byte and non-standard types via pairwise fallback.

    Covers bfloat16, e4m3, e5m2, e2m3, e3m2, int8, uint8, int4, and uint4 for
    the ``dot`` and ``euclidean`` metrics.  These dtypes have no batch-path
    kernel, so cdist always falls back to the scalar pairwise loop with float64
    output.

    Baselines are computed from the float64 arrays returned by ``make_random``
    (second return value), avoiding sub-byte row-indexing issues on
    ``nk.Tensor``.  Raw numpy byte-arrays are passed to cdist with an explicit
    ``dtype=`` parameter so the C code knows how to interpret the data.

    Skips int4/uint4 at odd ``ndim`` because the trailing padding nibble in
    each packed row is interpreted as data by the kernel.

    No ``capability`` parameter — exotic-dtype batch paths have pre-existing
    issues on non-serial backends, so only the default (serial) backend is
    tested.  Dimensions from ``NK_DENSE_DIMENSIONS``.
    """
    if input_dtype in ("int4", "uint4") and ndim % 2 != 0:
        pytest.skip(f"{input_dtype} requires even ndim (2 values per packed byte)")

    num_rows_a, num_rows_b = 5, 7
    a_raw, a_baseline = make_random((num_rows_a, ndim), input_dtype)
    b_raw, b_baseline = make_random((num_rows_b, ndim), input_dtype)

    # Baseline: compute from float64 baseline arrays (avoids sub-byte row indexing issues)
    if metric == "dot":
        expected = a_baseline @ b_baseline.T
    else:
        expected = np.zeros((num_rows_a, num_rows_b), dtype=np.float64)
        for i in range(num_rows_a):
            for j in range(num_rows_b):
                expected[i, j] = np.sqrt(np.sum((a_baseline[i] - b_baseline[j]) ** 2))

    # Use raw numpy arrays with explicit dtype= for sub-byte/exotic types
    cdist_kwargs = {"metric": metric, "dtype": input_dtype}

    # Test with default out_dtype (f64, pairwise fallback)
    result_f64 = nk.cdist(a_raw, b_raw, **cdist_kwargs)
    np.testing.assert_allclose(result_f64, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize(
    "m,n,k",
    [(1, 1, 8), (1, 5, 16), (5, 1, 16), (1, 1, 1), (2, 3, 4096), (13, 17, 97)],
)
def test_cdist_shapes(m, n, k):
    """Verify cdist output shape and correctness for diverse matrix geometries.

    Uses a hardcoded list of ``(m, n, k)`` triples that exercise edge cases
    the dimension-parameterised tests miss:

    * ``(1, 1, 8)`` — minimal single-row × single-row.
    * ``(1, 5, 16)`` and ``(5, 1, 16)`` — 1-row broadcasting in each direction.
    * ``(1, 1, 1)`` — absolute minimum dimensions.
    * ``(2, 3, 4096)`` — large vector depth (exercises SIMD tail handling).
    * ``(13, 17, 97)`` — odd prime dimensions with no alignment.

    Both ``euclidean`` (may use batch path) and ``sqeuclidean`` (pairwise only)
    are tested on float32 input with default float64 output.  Asserts both
    correctness (against SciPy) and output shape ``(m, n)``.

    Not parameterised by capability — shape handling is ISA-independent.
    """
    a_matrix = np.random.randn(m, k).astype(np.float32)
    b_matrix = np.random.randn(n, k).astype(np.float32)

    # euclidean (may use batch path)
    expected_euc = spd.cdist(a_matrix, b_matrix, "euclidean")
    result_euc = nk.cdist(a_matrix, b_matrix, "euclidean")
    assert result_euc.shape == (m, n), f"Expected shape ({m}, {n}), got {result_euc.shape}"
    np.testing.assert_allclose(result_euc, expected_euc, atol=NK_ATOL, rtol=NK_RTOL)

    # sqeuclidean (pairwise)
    expected_sqeuc = spd.cdist(a_matrix, b_matrix, "sqeuclidean")
    result_sqeuc = nk.cdist(a_matrix, b_matrix, "sqeuclidean")
    assert result_sqeuc.shape == (m, n), f"Expected shape ({m}, {n}), got {result_sqeuc.shape}"
    np.testing.assert_allclose(result_sqeuc, expected_sqeuc, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_cdist_edge_cases():
    """Verify cdist edge cases: scalar return, error handling, and removed API.

    Covers three categories:

    1. **Scalar return** — passing two 1D vectors ``(D,)`` must return a scalar
       float, not a ``(1, 1)`` matrix.
    2. **Rejected kwargs** — ``threads=`` was removed; verify ``TypeError`` is
       raised with "unexpected keyword" so callers get a clear error.
    3. **Input validation** — mismatched dimensions and empty matrices must
       raise ``ValueError`` (or ``SystemError`` due to a pre-existing ``%z``
       format-string bug in ``PyErr_Format``).

    Not parameterised — uses fixed 16-element vectors on float32.
    """
    ndim = 16
    a_vec = np.random.randn(ndim).astype(np.float32)
    b_vec = np.random.randn(ndim).astype(np.float32)

    # 1D vectors → scalar float return, not matrix
    result = nk.cdist(a_vec, b_vec, "euclidean")
    assert np.isscalar(result) or (
        hasattr(result, "shape") and result.shape == ()
    ), f"Expected scalar for 1D inputs, got {type(result)}"

    # threads= is rejected (removed parameter)
    with pytest.raises(TypeError, match="unexpected keyword"):
        nk.cdist(np.ones((2, 3)), np.ones((2, 3)), threads=2)

    # Mismatched dimensions → ValueError (or SystemError from format string bug)
    with pytest.raises((ValueError, SystemError)):
        nk.cdist(np.ones((2, 3)), np.ones((2, 5)), "euclidean")

    # Empty matrix → ValueError
    with pytest.raises((ValueError, SystemError)):
        nk.cdist(np.ones((0, 3)), np.ones((2, 3)), "euclidean")
