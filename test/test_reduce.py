#!/usr/bin/env python3
"""Test reductions: nk.moments, nk.minmax, nk.sum, nk.min, nk.max, nk.argmin, nk.argmax, nk.norm.

Covers dtypes: float64, float32, float16, int32 (axis reductions also test int32 norm).
Parametrized over: dtype, ndim from dense_dimensions, shape, axis.

Precision notes:
    Float reductions use rtol=1e-4 for f32, 1e-10 for f64.
    Integer reductions use exact equality.
    Sum always promotes to f64 internally, so even int32 sums are
    compared with assert_allclose against the f64 reference.

Matches C++ suite: test_reduce.cpp.
"""

import atexit
import decimal
import math

import pytest

try:
    import numpy as np
except:  # noqa: E722
    np = None

import numkong as nk
from test_base import (
    NK_ATOL,
    NK_RTOL,
    assert_allclose,
    create_stats,
    dense_dimensions,
    keep_one_capability,
    make_nk,
    make_random,
    nk_seed,  # noqa: F401 — pytest fixture
    numpy_available,
    possible_capabilities,
    print_stats_report,
    randomized_repetitions_count,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
    tolerances_for_dtype,
)

algebraic_dtypes = ["float32", "float64"]
algebraic_ndims = [7, 97]
stats = create_stats()
atexit.register(print_stats_report, stats)


def baseline_moments(a):
    """Reference moments: (sum, sum_of_squares) at f64 precision."""
    arr = np.asarray(a).astype(np.float64)
    return (np.sum(arr), np.sum(arr**2))


def baseline_sum(a):
    """Reference sum."""
    return np.sum(np.asarray(a))


def baseline_min(a):
    """Reference min."""
    return np.min(np.asarray(a))


def baseline_max(a):
    """Reference max."""
    return np.max(np.asarray(a))


def baseline_argmin(a):
    """Reference argmin."""
    return np.argmin(np.asarray(a))


def baseline_argmax(a):
    """Reference argmax."""
    return np.argmax(np.asarray(a))


def baseline_norm(a):
    """Reference L2 norm at f64 precision."""
    return np.linalg.norm(np.asarray(a).astype(np.float64))


def precise_sum(a):
    """High-precision sum via Decimal."""
    with decimal.localcontext() as ctx:
        ctx.prec = 120
        D = decimal.Decimal
        return float(sum(D.from_float(float(x)) for x in a))


def precise_min(a):
    return float(min(float(x) for x in a))


def precise_max(a):
    return float(max(float(x) for x in a))


def precise_argmin(a):
    vals = [float(x) for x in a]
    return vals.index(min(vals))


def precise_argmax(a):
    vals = [float(x) for x in a]
    return vals.index(max(vals))


def precise_norm(a):
    """High-precision L2 norm via Decimal."""
    with decimal.localcontext() as ctx:
        ctx.prec = 120
        D = decimal.Decimal
        return float(sum(D.from_float(float(x)) ** 2 for x in a).sqrt())


KERNELS_REDUCE = {
    "moments": (baseline_moments, nk.moments, None),
    "minmax": (
        lambda a: (np.asarray(a).min(), np.asarray(a).argmin(), np.asarray(a).max(), np.asarray(a).argmax()),
        nk.minmax,
        None,
    ),
    "sum": (baseline_sum, nk.sum, precise_sum),
    "min": (baseline_min, nk.min, precise_min),
    "max": (baseline_max, nk.max, precise_max),
    "argmin": (baseline_argmin, nk.argmin, precise_argmin),
    "argmax": (baseline_argmax, nk.argmax, precise_argmax),
    "norm": (baseline_norm, nk.norm, precise_norm),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize(
    "dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32"), pytest.param("float16", id="f16")]
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_moments(ndim, dtype, capability):
    """Test nk.moments() against NumPy sum and sum-of-squares."""
    keep_one_capability(capability)
    np_arr = np.random.randn(ndim).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)

    moments_result = nk.moments(nk_arr)
    nk_sum, nk_sum_sq = moments_result

    expected_sum, expected_sum_sq = baseline_moments(np_arr)
    atol, rtol = tolerances_for_dtype(dtype)
    assert_allclose(nk_sum, expected_sum, rtol=rtol, atol=atol)
    assert_allclose(nk_sum_sq, expected_sum_sq, rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize(
    "dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32"), pytest.param("float16", id="f16")]
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_minmax(ndim, dtype, capability):
    """Test nk.minmax() against NumPy min/argmin/max/argmax."""
    keep_one_capability(capability)
    np_arr = np.random.randn(ndim).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)

    minmax_result = nk.minmax(nk_arr)
    nk_min, nk_argmin, nk_max, nk_argmax = minmax_result

    atol, rtol = tolerances_for_dtype(dtype)
    assert_allclose(nk_min, np_arr.min(), rtol=rtol, atol=atol)
    assert int(nk_argmin) == np_arr.argmin()
    assert_allclose(nk_max, np_arr.max(), rtol=rtol, atol=atol)
    assert int(nk_argmax) == np_arr.argmax()


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", [1, 7, 16, 97])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("float16", id="f16"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_minmax_all_nan(ndim, dtype, capability):
    """All-NaN input returns None from minmax."""
    keep_one_capability(capability)
    np_arr = np.full(ndim, np.nan, dtype=dtype)
    nk_arr = make_nk(np_arr, dtype)
    assert nk.minmax(nk_arr) is None


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("float16", id="f16"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_minmax_mixed_nan(dtype, capability):
    """Mixed NaN + valid values returns correct min/max."""
    keep_one_capability(capability)
    np_arr = np.array([np.nan, 3.0, np.nan, 1.0, np.nan, 5.0, np.nan], dtype=dtype)
    nk_arr = make_nk(np_arr, dtype)
    result = nk.minmax(nk_arr)
    assert result is not None
    nk_min, nk_argmin, nk_max, nk_argmax = result
    assert float(nk_min) == pytest.approx(1.0, abs=0.1)
    assert int(nk_argmin) == 3
    assert float(nk_max) == pytest.approx(5.0, abs=0.1)
    assert int(nk_argmax) == 5


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", [1, 7, 16, 97])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("float16", id="f16"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_individual_reductions_all_nan(ndim, dtype, capability):
    """All-NaN input returns None from min, max, argmin, argmax."""
    keep_one_capability(capability)
    np_arr = np.full(ndim, np.nan, dtype=dtype)
    nk_arr = make_nk(np_arr, dtype)
    assert nk.min(nk_arr) is None
    assert nk.max(nk_arr) is None
    assert nk.argmin(nk_arr) is None
    assert nk.argmax(nk_arr) is None


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("float16", id="f16"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_module_level_reductions(ndim, dtype, capability, nk_seed):
    """Test nk.sum(), nk.min(), nk.max(), nk.argmin(), nk.argmax() module functions."""
    keep_one_capability(capability)
    raw, baseline = make_random((ndim,), dtype, seed=nk_seed)
    nk_arr = make_nk(raw, dtype) if numpy_available else raw

    assert_allclose(nk.sum(nk_arr), precise_sum(baseline), atol=NK_ATOL, rtol=NK_RTOL)
    assert_allclose(nk.min(nk_arr), precise_min(baseline), atol=NK_ATOL, rtol=NK_RTOL)
    assert_allclose(nk.max(nk_arr), precise_max(baseline), atol=NK_ATOL, rtol=NK_RTOL)
    assert nk.argmin(nk_arr) == precise_argmin(baseline)
    assert nk.argmax(nk_arr) == precise_argmax(baseline)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_sum_axis(dtype, capability):
    """sum(axis=) on 2D and 3D tensors vs NumPy."""
    keep_one_capability(capability)
    atol, rtol = tolerances_for_dtype(dtype)
    # 2D
    np_arr = np.random.randn(5, 7).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axis in [0, 1]:
        result = np.asarray(nk_arr.sum(axis=axis))
        expected = np_arr.astype(np.float64).sum(axis=axis)
        assert_allclose(result, expected, rtol=rtol, atol=atol)
    # 3D
    np_arr3 = np.random.randn(3, 4, 5).astype(dtype)
    nk_arr3 = make_nk(np_arr3, dtype)
    for axis in [0, 1, 2]:
        result = np.asarray(nk_arr3.sum(axis=axis))
        expected = np_arr3.astype(np.float64).sum(axis=axis)
        assert_allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_min_max_axis(dtype, capability):
    """min/max(axis=) on 2D tensors vs NumPy."""
    keep_one_capability(capability)
    np_arr = np.random.randn(6, 8).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    atol, rtol = tolerances_for_dtype(dtype)
    for axis in [0, 1]:
        assert_allclose(np.asarray(nk_arr.min(axis=axis)), np_arr.min(axis=axis), rtol=rtol, atol=atol)
        assert_allclose(np.asarray(nk_arr.max(axis=axis)), np_arr.max(axis=axis), rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_argmin_argmax_axis(dtype, capability):
    """argmin/argmax(axis=) on 2D tensors vs NumPy."""
    keep_one_capability(capability)
    np_arr = np.random.randn(6, 8).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axis in [0, 1]:
        nk_argmin = np.asarray(nk_arr.argmin(axis=axis))
        nk_argmax = np.asarray(nk_arr.argmax(axis=axis))
        np.testing.assert_array_equal(nk_argmin, np_arr.argmin(axis=axis))
        np.testing.assert_array_equal(nk_argmax, np_arr.argmax(axis=axis))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_norm_axis(dtype, capability):
    """norm(axis=) vs np.linalg.norm(x, axis=)."""
    keep_one_capability(capability)
    atol, rtol = tolerances_for_dtype(dtype)
    np_arr = np.random.randn(5, 7).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axis in [0, 1]:
        result = np.asarray(nk_arr.norm(axis=axis))
        expected = np.linalg.norm(np_arr.astype(np.float64), axis=axis)
        assert_allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_keepdims(dtype, capability):
    """keepdims=True preserves rank with size-1 at reduced axis."""
    keep_one_capability(capability)
    np_arr = np.random.randn(4, 5).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axis in [0, 1]:
        expected_shape = list(np_arr.shape)
        expected_shape[axis] = 1
        for op in ["sum", "min", "max", "norm"]:
            result = getattr(nk_arr, op)(axis=axis, keepdims=True)
            assert result.ndim == 2, f"{op} keepdims ndim"
            assert list(result.shape) == expected_shape, f"{op} keepdims shape"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_out_parameter(dtype, capability):
    """out= writes to pre-allocated tensor, returns it."""
    keep_one_capability(capability)
    np_arr = np.random.randn(4, 5).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    # sum along axis=0 -> shape (5,), dtype float64
    out_dtype = "float64"  # sum always promotes to f64
    out = nk.zeros((5,), dtype=out_dtype)
    ret = nk_arr.sum(axis=0, out=out)
    # ret should be the same object as out
    assert np.asarray(ret).ctypes.data == np.asarray(out).ctypes.data
    expected = np_arr.astype(np.float64).sum(axis=0)
    assert_allclose(np.asarray(out), expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_module_level_axis(capability):
    """nk.sum(a, axis=0), nk.min(a, axis=1), etc."""
    keep_one_capability(capability)
    np_arr = np.random.randn(4, 5).astype("float64")
    nk_arr = make_nk(np_arr, "float64")
    assert_allclose(np.asarray(nk.sum(nk_arr, axis=0)), np_arr.sum(axis=0))
    assert_allclose(np.asarray(nk.min(nk_arr, axis=1)), np_arr.min(axis=1))
    assert_allclose(np.asarray(nk.max(nk_arr, axis=0)), np_arr.max(axis=0))
    np.testing.assert_array_equal(np.asarray(nk.argmin(nk_arr, axis=1)), np_arr.argmin(axis=1))
    np.testing.assert_array_equal(np.asarray(nk.argmax(nk_arr, axis=0)), np_arr.argmax(axis=0))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_negative_axis(capability):
    """axis=-1 on 2D and axis=-2 on 3D reduce the correct dimension."""
    keep_one_capability(capability)
    # 2D, axis=-1
    np_arr = np.random.randn(3, 4).astype("float64")
    nk_arr = make_nk(np_arr, "float64")
    result = np.asarray(nk_arr.sum(axis=-1))
    expected = np_arr.sum(axis=-1)
    assert_allclose(result, expected)
    # 3D, axis=-2
    np_arr3 = np.random.randn(2, 3, 4).astype("float64")
    nk_arr3 = make_nk(np_arr3, "float64")
    result3 = np.asarray(nk_arr3.sum(axis=-2))
    expected3 = np_arr3.sum(axis=-2)
    assert_allclose(result3, expected3)


@pytest.mark.parametrize("capability", possible_capabilities)
def test_axis_error(capability):
    """axis out of range raises ValueError."""
    keep_one_capability(capability)
    nk_arr = nk.zeros((3, 4), dtype="float64")
    with pytest.raises(ValueError, match="axis.*out of range"):
        nk_arr.sum(axis=2)
    with pytest.raises(ValueError, match="axis.*out of range"):
        nk_arr.sum(axis=-3)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_integer_axis_reductions(capability):
    """sum/min/max/argmin/argmax along axis on int32 tensors."""
    keep_one_capability(capability)
    np_arr = np.array([[10, 3, 7], [1, 8, 5], [4, 9, 2]], dtype=np.int32)
    nk_arr = make_nk(np_arr, "int32")
    for axis in [0, 1]:
        assert_allclose(
            np.asarray(nk_arr.sum(axis=axis)), np_arr.astype(np.float64).sum(axis=axis)
        )
        np.testing.assert_array_equal(np.asarray(nk_arr.min(axis=axis)), np_arr.min(axis=axis))
        np.testing.assert_array_equal(np.asarray(nk_arr.max(axis=axis)), np_arr.max(axis=axis))
        np.testing.assert_array_equal(np.asarray(nk_arr.argmin(axis=axis)), np_arr.argmin(axis=axis))
        np.testing.assert_array_equal(np.asarray(nk_arr.argmax(axis=axis)), np_arr.argmax(axis=axis))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_norm_integer(capability):
    """norm(axis=) on int32 must include sqrt (the fix norm_slice was made for)."""
    keep_one_capability(capability)
    np_arr = np.array([[3, 4], [5, 12]], dtype=np.int32)
    nk_arr = make_nk(np_arr, "int32")
    for axis in [0, 1]:
        result = np.asarray(nk_arr.norm(axis=axis))
        expected = np.linalg.norm(np_arr.astype(np.float64), axis=axis)
        assert_allclose(result, expected)


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_sum_known(ndim, dtype, capability):
    """sum(ones(n)) ~ n."""
    keep_one_capability(capability)
    ones_tensor = nk.ones((ndim,), dtype=dtype)
    result = ones_tensor.sum()
    assert abs(result - ndim) < 0.1 + 0.1 * ndim


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_norm_known(ndim, dtype, capability):
    """norm(ones(n)) ~ sqrt(n)."""
    keep_one_capability(capability)
    ones_tensor = nk.ones((ndim,), dtype=dtype)
    result = nk.norm(ones_tensor)
    expected = math.sqrt(ndim)
    assert abs(result - expected) < 0.1 + 0.1 * expected


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_min_max_known(ndim, dtype, capability):
    """min(full(c)) = max(full(c)) = c."""
    keep_one_capability(capability)
    fill_value = 3.14
    constant_tensor = nk.full((ndim,), fill_value, dtype=dtype)
    assert abs(constant_tensor.min() - fill_value) < 0.01
    assert abs(constant_tensor.max() - fill_value) < 0.01


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_argmin_argmax_constant(ndim, dtype, capability):
    """For a constant tensor, argmin and argmax return valid indices in [0, n)."""
    keep_one_capability(capability)
    constant_tensor = nk.full((ndim,), 2.0, dtype=dtype)
    assert 0 <= constant_tensor.argmin() < ndim
    assert 0 <= constant_tensor.argmax() < ndim


# region Multi-axis reductions


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_sum(dtype, capability):
    """sum(axis=tuple) on 3D tensor vs NumPy."""
    keep_one_capability(capability)
    atol, rtol = tolerances_for_dtype(dtype)
    np_arr = np.random.randn(3, 4, 5).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axes in [(0, 1), (1, 2), (0, 2), (0, 1, 2)]:
        result = np.asarray(nk_arr.sum(axis=axes))
        expected = np_arr.astype(np.float64).sum(axis=axes)
        assert_allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_min_max(dtype, capability):
    """min/max(axis=tuple) on 3D tensor vs NumPy."""
    keep_one_capability(capability)
    atol, rtol = tolerances_for_dtype(dtype)
    np_arr = np.random.randn(3, 4, 5).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axes in [(0, 1), (1, 2), (0, 2)]:
        assert_allclose(np.asarray(nk_arr.min(axis=axes)), np_arr.min(axis=axes), rtol=rtol, atol=atol)
        assert_allclose(np.asarray(nk_arr.max(axis=axes)), np_arr.max(axis=axes), rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_norm(dtype, capability):
    """norm(axis=tuple) vs manual sqrt(sum(x**2, axis=axes))."""
    keep_one_capability(capability)
    atol, rtol = tolerances_for_dtype(dtype)
    np_arr = np.random.randn(3, 4, 5).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axes in [(0, 1), (1, 2), (0, 2)]:
        result = np.asarray(nk_arr.norm(axis=axes))
        expected = np.sqrt(np.sum(np_arr.astype(np.float64) ** 2, axis=axes))
        assert_allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_keepdims(capability):
    """keepdims=True preserves rank with size-1 at each reduced axis."""
    keep_one_capability(capability)
    np_arr = np.random.randn(3, 4, 5).astype(np.float64)
    nk_arr = make_nk(np_arr, "float64")
    for axes in [(0, 2), (0, 1), (1, 2), (0, 1, 2)]:
        result = nk_arr.sum(axis=axes, keepdims=True)
        expected = np_arr.sum(axis=axes, keepdims=True)
        assert result.shape == expected.shape, f"axes={axes}: {result.shape} != {expected.shape}"
        assert_allclose(np.asarray(result), expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_negative(capability):
    """Negative indices in axis tuple are normalized correctly."""
    keep_one_capability(capability)
    np_arr = np.random.randn(3, 4, 5).astype(np.float64)
    nk_arr = make_nk(np_arr, "float64")
    result = np.asarray(nk_arr.sum(axis=(-1, 0)))
    expected = np_arr.sum(axis=(0, 2))  # -1 -> 2, sorted -> (0, 2)
    assert_allclose(result, expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_single_element_tuple(capability):
    """axis=(1,) should behave identically to axis=1."""
    keep_one_capability(capability)
    np_arr = np.random.randn(3, 4, 5).astype(np.float64)
    nk_arr = make_nk(np_arr, "float64")
    result_tuple = np.asarray(nk_arr.sum(axis=(1,)))
    result_int = np.asarray(nk_arr.sum(axis=1))
    assert_allclose(result_tuple, result_int)


@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_errors(capability):
    """Error cases for multi-axis reductions."""
    keep_one_capability(capability)
    nk_arr = nk.zeros((3, 4, 5), dtype="float64")

    # Duplicate axes
    with pytest.raises(ValueError, match="duplicate"):
        nk_arr.sum(axis=(1, 1))

    # Out of range
    with pytest.raises(ValueError, match="out of range"):
        nk_arr.sum(axis=(0, 5))

    # Empty tuple
    with pytest.raises(ValueError, match="empty"):
        nk_arr.sum(axis=())

    # argmin/argmax reject tuple axis
    with pytest.raises(TypeError):
        nk_arr.argmin(axis=(0, 1))
    with pytest.raises(TypeError):
        nk_arr.argmax(axis=(0, 1))

    # Non-int in tuple
    with pytest.raises(TypeError):
        nk_arr.sum(axis=(0, 1.5))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_module_level(capability):
    """Module-level nk.sum(arr, axis=tuple) works for buffer-protocol inputs."""
    keep_one_capability(capability)
    np_arr = np.random.randn(3, 4, 5).astype(np.float64)
    result = np.asarray(nk.sum(np_arr, axis=(0, 2)))
    expected = np_arr.sum(axis=(0, 2))
    assert_allclose(result, expected)


# endregion Multi-axis reductions
