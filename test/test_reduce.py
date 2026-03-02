#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test reductions: nk.moments, nk.minmax, nk.sum, nk.min, nk.max, nk.argmin, nk.argmax, nk.norm.

Covers dtypes: float64, float32, float16, bfloat16, int32 (axis reductions also test int32 norm).
Parametrized over: dtype, ndim from dense_dimensions, shape, axis.

Precision notes:
    Float reductions use rtol=1e-4 for f32, 1e-10 for f64.
    Integer reductions use exact equality.
    Sum always promotes to f64 internally, so even int32 sums are
    compared with assert_allclose against the f64 reference.

Matches C++ suite: test_reduce.cpp.
"""

import atexit
import math
import pytest

try:
    import numpy as np
except:
    np = None

import numkong as nk
from test_base import (
    numpy_available,
    dense_dimensions,
    make_nk,
    create_stats,
    collect_errors,
    print_stats_report,
    seed_rng,
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


KERNELS_REDUCE = {
    "moments": (baseline_moments, nk.moments, None),
    "minmax": (
        lambda a: (np.asarray(a).min(), np.asarray(a).argmin(), np.asarray(a).max(), np.asarray(a).argmax()),
        nk.minmax,
        None,
    ),
    "sum": (baseline_sum, nk.sum, None),
    "min": (baseline_min, nk.min, None),
    "max": (baseline_max, nk.max, None),
    "argmin": (baseline_argmin, nk.argmin, None),
    "argmax": (baseline_argmax, nk.argmax, None),
    "norm": (baseline_norm, nk.norm, None),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize(
    "dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32"), pytest.param("float16", id="f16")]
)
def test_moments(ndim, dtype):
    """Test nk.moments() against NumPy sum and sum-of-squares."""
    np_arr = np.random.randn(ndim).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)

    moments_result = nk.moments(nk_arr)
    nk_sum, nk_sum_sq = moments_result

    expected_sum, expected_sum_sq = baseline_moments(np_arr)
    np.testing.assert_allclose(nk_sum, expected_sum, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(nk_sum_sq, expected_sum_sq, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize(
    "dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32"), pytest.param("float16", id="f16")]
)
def test_minmax(ndim, dtype):
    """Test nk.minmax() against NumPy min/argmin/max/argmax."""
    np_arr = np.random.randn(ndim).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)

    minmax_result = nk.minmax(nk_arr)
    nk_min, nk_argmin, nk_max, nk_argmax = minmax_result

    np.testing.assert_allclose(nk_min, np_arr.min(), rtol=1e-5)
    assert int(nk_argmin) == np_arr.argmin()
    np.testing.assert_allclose(nk_max, np_arr.max(), rtol=1e-5)
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
def test_minmax_all_nan(ndim, dtype):
    """All-NaN input returns None from minmax."""
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
def test_minmax_mixed_nan(dtype):
    """Mixed NaN + valid values returns correct min/max."""
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
def test_individual_reductions_all_nan(ndim, dtype):
    """All-NaN input returns None from min, max, argmin, argmax."""
    np_arr = np.full(ndim, np.nan, dtype=dtype)
    nk_arr = make_nk(np_arr, dtype)
    assert nk.min(nk_arr) is None
    assert nk.max(nk_arr) is None
    assert nk.argmin(nk_arr) is None
    assert nk.argmax(nk_arr) is None


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("float16", id="f16"),
    ],
)
def test_module_level_reductions(ndim, dtype):
    """Test nk.sum(), nk.min(), nk.max(), nk.argmin(), nk.argmax() module functions."""
    np_arr = np.random.randn(ndim).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)

    # For float16, the SIMD kernel accumulates in float32 so compare against a float32 reference
    sum_ref = np_arr.astype(np.float32).sum() if dtype == "float16" else np_arr.sum()
    np.testing.assert_allclose(nk.sum(nk_arr), sum_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(nk.min(nk_arr), np_arr.min(), rtol=1e-5)
    np.testing.assert_allclose(nk.max(nk_arr), np_arr.max(), rtol=1e-5)
    assert nk.argmin(nk_arr) == np_arr.argmin()
    assert nk.argmax(nk_arr) == np_arr.argmax()


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_sum_axis(dtype):
    """sum(axis=) on 2D and 3D tensors vs NumPy."""
    rtol = 1e-4 if dtype == "float32" else 1e-10
    # 2D
    np_arr = np.random.randn(5, 7).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axis in [0, 1]:
        result = np.asarray(nk_arr.sum(axis=axis))
        expected = np_arr.astype(np.float64).sum(axis=axis)
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=1e-6)
    # 3D
    np_arr3 = np.random.randn(3, 4, 5).astype(dtype)
    nk_arr3 = make_nk(np_arr3, dtype)
    for axis in [0, 1, 2]:
        result = np.asarray(nk_arr3.sum(axis=axis))
        expected = np_arr3.astype(np.float64).sum(axis=axis)
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=1e-6)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
def test_min_max_axis(dtype):
    """min/max(axis=) on 2D tensors vs NumPy."""
    np_arr = np.random.randn(6, 8).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axis in [0, 1]:
        np.testing.assert_allclose(np.asarray(nk_arr.min(axis=axis)), np_arr.min(axis=axis), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(nk_arr.max(axis=axis)), np_arr.max(axis=axis), rtol=1e-6)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
def test_argmin_argmax_axis(dtype):
    """argmin/argmax(axis=) on 2D tensors vs NumPy."""
    np_arr = np.random.randn(6, 8).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axis in [0, 1]:
        nk_argmin = np.asarray(nk_arr.argmin(axis=axis))
        nk_argmax = np.asarray(nk_arr.argmax(axis=axis))
        np.testing.assert_array_equal(nk_argmin, np_arr.argmin(axis=axis))
        np.testing.assert_array_equal(nk_argmax, np_arr.argmax(axis=axis))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
def test_norm_axis(dtype):
    """norm(axis=) vs np.linalg.norm(x, axis=)."""
    rtol = 1e-4 if dtype == "float32" else 1e-10
    np_arr = np.random.randn(5, 7).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axis in [0, 1]:
        result = np.asarray(nk_arr.norm(axis=axis))
        expected = np.linalg.norm(np_arr.astype(np.float64), axis=axis)
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=1e-6)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
def test_keepdims(dtype):
    """keepdims=True preserves rank with size-1 at reduced axis."""
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
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
def test_out_parameter(dtype):
    """out= writes to pre-allocated tensor, returns it."""
    np_arr = np.random.randn(4, 5).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    # sum along axis=0 -> shape (5,), dtype float64
    out_dtype = "float64"  # sum always promotes to f64
    out = nk.zeros((5,), dtype=out_dtype)
    ret = nk_arr.sum(axis=0, out=out)
    # ret should be the same object as out
    assert np.asarray(ret).ctypes.data == np.asarray(out).ctypes.data
    expected = np_arr.astype(np.float64).sum(axis=0)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-10, atol=1e-6)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_module_level_axis():
    """nk.sum(a, axis=0), nk.min(a, axis=1), etc."""
    np_arr = np.random.randn(4, 5).astype("float64")
    nk_arr = make_nk(np_arr, "float64")
    np.testing.assert_allclose(np.asarray(nk.sum(nk_arr, axis=0)), np_arr.sum(axis=0), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(nk.min(nk_arr, axis=1)), np_arr.min(axis=1), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(nk.max(nk_arr, axis=0)), np_arr.max(axis=0), rtol=1e-10)
    np.testing.assert_array_equal(np.asarray(nk.argmin(nk_arr, axis=1)), np_arr.argmin(axis=1))
    np.testing.assert_array_equal(np.asarray(nk.argmax(nk_arr, axis=0)), np_arr.argmax(axis=0))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_negative_axis():
    """axis=-1 on 2D and axis=-2 on 3D reduce the correct dimension."""
    # 2D, axis=-1
    np_arr = np.random.randn(3, 4).astype("float64")
    nk_arr = make_nk(np_arr, "float64")
    result = np.asarray(nk_arr.sum(axis=-1))
    expected = np_arr.sum(axis=-1)
    np.testing.assert_allclose(result, expected, rtol=1e-10)
    # 3D, axis=-2
    np_arr3 = np.random.randn(2, 3, 4).astype("float64")
    nk_arr3 = make_nk(np_arr3, "float64")
    result3 = np.asarray(nk_arr3.sum(axis=-2))
    expected3 = np_arr3.sum(axis=-2)
    np.testing.assert_allclose(result3, expected3, rtol=1e-10)


def test_axis_error():
    """axis out of range raises ValueError."""
    nk_arr = nk.zeros((3, 4), dtype="float64")
    with pytest.raises(ValueError, match="axis.*out of range"):
        nk_arr.sum(axis=2)
    with pytest.raises(ValueError, match="axis.*out of range"):
        nk_arr.sum(axis=-3)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_integer_axis_reductions():
    """sum/min/max/argmin/argmax along axis on int32 tensors."""
    np_arr = np.array([[10, 3, 7], [1, 8, 5], [4, 9, 2]], dtype=np.int32)
    nk_arr = make_nk(np_arr, "int32")
    for axis in [0, 1]:
        np.testing.assert_allclose(
            np.asarray(nk_arr.sum(axis=axis)), np_arr.astype(np.float64).sum(axis=axis), rtol=1e-10
        )
        np.testing.assert_array_equal(np.asarray(nk_arr.min(axis=axis)), np_arr.min(axis=axis))
        np.testing.assert_array_equal(np.asarray(nk_arr.max(axis=axis)), np_arr.max(axis=axis))
        np.testing.assert_array_equal(np.asarray(nk_arr.argmin(axis=axis)), np_arr.argmin(axis=axis))
        np.testing.assert_array_equal(np.asarray(nk_arr.argmax(axis=axis)), np_arr.argmax(axis=axis))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_norm_integer():
    """norm(axis=) on int32 must include sqrt (the fix norm_slice was made for)."""
    np_arr = np.array([[3, 4], [5, 12]], dtype=np.int32)
    nk_arr = make_nk(np_arr, "int32")
    for axis in [0, 1]:
        result = np.asarray(nk_arr.norm(axis=axis))
        expected = np.linalg.norm(np_arr.astype(np.float64), axis=axis)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_sum_known(ndim, dtype):
    """sum(ones(n)) ~ n."""
    ones_tensor = nk.ones((ndim,), dtype=dtype)
    result = ones_tensor.sum()
    assert abs(result - ndim) < 0.1 + 0.1 * ndim


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_norm_known(ndim, dtype):
    """norm(ones(n)) ~ sqrt(n)."""
    ones_tensor = nk.ones((ndim,), dtype=dtype)
    result = nk.norm(ones_tensor)
    expected = math.sqrt(ndim)
    assert abs(result - expected) < 0.1 + 0.1 * expected


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_min_max_known(ndim, dtype):
    """min(full(c)) = max(full(c)) = c."""
    fill_value = 3.14
    constant_tensor = nk.full((ndim,), fill_value, dtype=dtype)
    assert abs(constant_tensor.min() - fill_value) < 0.01
    assert abs(constant_tensor.max() - fill_value) < 0.01


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_argmin_argmax_constant(ndim, dtype):
    """For a constant tensor, argmin and argmax return valid indices in [0, n)."""
    constant_tensor = nk.full((ndim,), 2.0, dtype=dtype)
    assert 0 <= constant_tensor.argmin() < ndim
    assert 0 <= constant_tensor.argmax() < ndim
