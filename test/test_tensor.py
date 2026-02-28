#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test tensor API: constructors, slicing, ndarray marshalling, scalar types.

Covers: nk.zeros, nk.ones, nk.full, nk.empty, nk.cdist,
    nk.Tensor properties (shape, dtype, ndim, strides, etc.),
    indexing, iteration, transpose, reshape, copy, equality,
    NumPy interop (__array_interface__, memoryview, buffer protocol),
    arithmetic operators (+, -, *, unary +/-),
    strided/transposed/subview reductions,
    bfloat16/float8 scalar types and conversion vs ml_dtypes,
    GIL-free threading.
Covers dtypes: float64, float32, int8, int32 (constructors/reductions);
    bfloat16, float8_e4m3, float8_e5m2, float6_e2m3, float6_e3m2 (scalar types).
Parametrized over: dtype, shape, ndim.

Matches C++ suite: test_tensor.cpp.
"""

import sys
import time
import sysconfig
import multiprocessing
import concurrent.futures
import warnings

import pytest

try:
    import numpy as np
except:
    np = None

import numkong as nk
from test_base import (
    numpy_available,
    scipy_available,
    ml_dtypes_available,
    dense_dimensions,
    possible_capabilities,
    randomized_repetitions_count,
    keep_one_capability,
    to_array,
    NK_ATOL,
    NK_RTOL,
    f32_downcast_to_bf16,
    make_nk,
    seed_rng,
)

try:
    import ml_dtypes
except ImportError:
    pass

KERNELS_TENSOR = {
    "sum": (lambda a: np.sum(np.asarray(a)), lambda a: a.sum(), None),
    "min": (lambda a: np.min(np.asarray(a)), lambda a: a.min(), None),
    "max": (lambda a: np.max(np.asarray(a)), lambda a: a.max(), None),
    "argmin": (lambda a: np.argmin(np.asarray(a)), lambda a: a.argmin(), None),
    "argmax": (lambda a: np.argmax(np.asarray(a)), lambda a: a.argmax(), None),
    "add": (lambda a, b: np.add(np.asarray(a), np.asarray(b)), lambda a, b: a + b, None),
    "subtract": (lambda a, b: np.subtract(np.asarray(a), np.asarray(b)), lambda a, b: a - b, None),
    "multiply": (lambda a, b: np.multiply(np.asarray(a), np.asarray(b)), lambda a, b: a * b, None),
    "negate": (lambda a: -np.asarray(a), lambda a: -a, None),
}


def test_pointers_availability():
    """Tests the availability of pre-compiled functions for compatibility with USearch."""
    assert nk.pointer_to_sqeuclidean("float64") != 0
    assert nk.pointer_to_angular("float64") != 0
    assert nk.pointer_to_inner("float64") != 0

    assert nk.pointer_to_sqeuclidean("float32") != 0
    assert nk.pointer_to_angular("float32") != 0
    assert nk.pointer_to_inner("float32") != 0

    assert nk.pointer_to_sqeuclidean("float16") != 0
    assert nk.pointer_to_angular("float16") != 0
    assert nk.pointer_to_inner("float16") != 0

    assert nk.pointer_to_sqeuclidean("int8") != 0
    assert nk.pointer_to_angular("int8") != 0
    assert nk.pointer_to_inner("int8") != 0

    assert nk.pointer_to_sqeuclidean("uint8") != 0
    assert nk.pointer_to_angular("uint8") != 0
    assert nk.pointer_to_inner("uint8") != 0


def test_capabilities_list():
    """Tests the visibility of hardware capabilities."""
    assert "serial" in nk.get_capabilities()
    assert "neon" in nk.get_capabilities()
    assert "neonhalf" in nk.get_capabilities()
    assert "neonbfdot" in nk.get_capabilities()
    assert "neonsdot" in nk.get_capabilities()
    assert "sve" in nk.get_capabilities()
    assert "svehalf" in nk.get_capabilities()
    assert "svebfdot" in nk.get_capabilities()
    assert "svesdot" in nk.get_capabilities()
    assert "haswell" in nk.get_capabilities()
    assert "icelake" in nk.get_capabilities()
    assert "skylake" in nk.get_capabilities()
    assert "genoa" in nk.get_capabilities()
    assert "sapphire" in nk.get_capabilities()
    assert "turin" in nk.get_capabilities()
    assert nk.get_capabilities().get("serial") == 1

    previous_value = nk.get_capabilities().get("neon")
    nk.enable_capability("neon")
    assert nk.get_capabilities().get("neon") == 1
    if not previous_value:
        nk.disable_capability("neon")


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "function, expected_error, args, kwargs",
    [
        (nk.sqeuclidean, TypeError, (), {}),
        (nk.sqeuclidean, TypeError, (to_array([1.0]),), {}),
        (nk.sqeuclidean, ValueError, (to_array([1.0]), to_array([1.0]), "missing_dtype"), {}),
        (nk.sqeuclidean, TypeError, (to_array([1.0]), "invalid"), {}),
        (nk.sqeuclidean, TypeError, (to_array([1.0]), to_array([1.0])), {"invalid_kwarg": "value"}),
        (nk.enable_capability, TypeError, (123,), {}),
        (nk.disable_capability, TypeError, ([],), {}),
        (nk.mahalanobis, TypeError, (to_array([1.0]), to_array([1.0])), {}),
        (nk.bilinear, TypeError, (to_array([1.0]),), {}),
        (nk.angular, TypeError, (to_array([1.0]), to_array([1.0]), to_array([1.0])), {}),
        (nk.cdist, TypeError, (to_array([[1.0]]), to_array([[1.0]]), "l2", "dos"), {}),
        (nk.cdist, TypeError, (to_array([[1.0]]), to_array([[1.0]]), "l2"), {"metric": "l2"}),
        (nk.angular, LookupError, (to_array([1 + 2j]), to_array([1 + 2j])), {}),
        (nk.angular, ValueError, (to_array([1.0]), to_array([1.0, 2.0])), {}),
        (nk.angular, TypeError, (to_array([1.0]), to_array([1], "int8")), {}),
        (nk.angular, TypeError, (to_array([1], "float32"), to_array([1], "float16")), {}),
    ],
)
def test_invalid_argument_handling(function, expected_error, args, kwargs):
    """Test that functions raise appropriate errors with invalid arguments."""
    with pytest.raises(expected_error):
        function(*args, **kwargs)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
def test_bf16_conversion_vs_ml_dtypes(ndim):
    """Compare NumKong bfloat16 conversion with ml_dtypes reference implementation."""
    a_f32 = np.random.randn(ndim).astype(np.float32)
    _, a_nk_bf16 = f32_downcast_to_bf16(a_f32)
    a_ml_bf16 = a_f32.astype(ml_dtypes.bfloat16)
    ml_bits = np.asarray(a_ml_bf16.view(np.uint16), dtype=np.uint16)
    assert np.array_equal(a_nk_bf16, ml_bits), (
        f"BFloat16 conversion mismatch with ml_dtypes:\n"
        f"  NumKong bits: {a_nk_bf16[:5]}...\n"
        f"  ml_dtypes bits: {ml_bits[:5]}..."
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
def test_float8_e4m3_conversion_vs_ml_dtypes(ndim):
    """Compare NumKong float8_e4m3 conversion with ml_dtypes reference implementation."""
    a_f32 = (np.random.randn(ndim) * 10).astype(np.float32)
    a_f32 = np.clip(a_f32, -448, 448)
    a_ml_e4m3 = a_f32.astype(ml_dtypes.float8_e4m3fn)
    a_nk = nk.zeros((ndim,), dtype="e4m3")
    assert a_ml_e4m3.dtype == ml_dtypes.float8_e4m3fn
    assert a_nk.dtype == "e4m3"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
def test_float8_e5m2_conversion_vs_ml_dtypes(ndim):
    """Compare NumKong float8_e5m2 conversion with ml_dtypes reference implementation."""
    a_f32 = (np.random.randn(ndim) * 100).astype(np.float32)
    a_f32 = np.clip(a_f32, -57344, 57344)
    a_ml_e5m2 = a_f32.astype(ml_dtypes.float8_e5m2)
    a_nk = nk.zeros((ndim,), dtype="e5m2")
    assert a_ml_e5m2.dtype == ml_dtypes.float8_e5m2
    assert a_nk.dtype == "e5m2"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
def test_float6_e2m3_construction(ndim):
    """Verify that e2m3 tensors can be constructed and have the correct dtype."""
    a_nk = nk.zeros((ndim,), dtype="e2m3")
    assert a_nk.dtype == "e2m3"
    assert a_nk.shape == (ndim,)
    assert a_nk.ndim == 1


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
def test_float6_e3m2_construction(ndim):
    """Verify that e3m2 tensors can be constructed and have the correct dtype."""
    a_nk = nk.zeros((ndim,), dtype="e3m2")
    assert a_nk.dtype == "e3m2"
    assert a_nk.shape == (ndim,)
    assert a_nk.ndim == 1


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_properties():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    assert hasattr(result, "shape")
    assert hasattr(result, "dtype")
    assert hasattr(result, "ndim")
    assert hasattr(result, "size")
    assert hasattr(result, "nbytes")
    assert hasattr(result, "strides")
    assert hasattr(result, "itemsize")

    assert result.shape == (5, 7)
    assert result.dtype == "float64"
    assert result.ndim == 2
    assert result.size == 35
    assert result.nbytes == 35 * 8
    assert result.itemsize == 8
    assert isinstance(result.strides, tuple)
    assert len(result.strides) == 2


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_properties_1d():
    a = np.random.rand(10, 128).astype(np.float64)
    b = np.random.rand(10, 128).astype(np.float64)
    result = nk.sqeuclidean(a, b)

    assert result.shape == (10,)
    assert result.ndim == 1
    assert result.size == 10
    assert len(result.strides) == 1


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_len():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result_2d = nk.cdist(a, b, metric="sqeuclidean")
    assert len(result_2d) == 5

    a = np.random.rand(10, 128).astype(np.float64)
    b = np.random.rand(10, 128).astype(np.float64)
    result_1d = nk.sqeuclidean(a, b)
    assert len(result_1d) == 10


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_repr():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    repr_str = repr(result)
    assert "Tensor" in repr_str
    assert "shape=" in repr_str
    assert "dtype=" in repr_str
    assert "(5, 7)" in repr_str
    assert "float64" in repr_str


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_indexing():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")
    expected = np.asarray(result)

    row0 = result[0]
    assert hasattr(row0, "shape")
    assert row0.shape == (7,)

    row_last = result[-1]
    assert row_last.shape == (7,)
    np.testing.assert_allclose(np.asarray(row_last), expected[-1], rtol=1e-6)

    val = result[2, 3]
    assert isinstance(val, float)
    np.testing.assert_allclose(val, expected[2, 3], rtol=1e-6)

    val_neg = result[-1, -1]
    assert isinstance(val_neg, float)
    np.testing.assert_allclose(val_neg, expected[-1, -1], rtol=1e-6)

    with pytest.raises(IndexError):
        _ = result[10]
    with pytest.raises(IndexError):
        _ = result[0, 100]


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_iteration():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")
    expected = np.asarray(result)

    count = 0
    for i, row in enumerate(result):
        count += 1
        assert row.shape == (7,)
        np.testing.assert_allclose(np.asarray(row), expected[i], rtol=1e-6)
    assert count == 5

    a = np.random.rand(3, 128).astype(np.float64)
    b = np.random.rand(3, 128).astype(np.float64)
    result_1d = nk.sqeuclidean(a, b)
    expected_1d = np.asarray(result_1d)

    items = list(result_1d)
    assert len(items) == 3
    for i, item in enumerate(items):
        assert isinstance(item, float)
        np.testing.assert_allclose(item, expected_1d[i], rtol=1e-6)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_numpy_interop():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    arr = np.asarray(result)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (5, 7)
    assert arr.dtype == np.float64
    assert np.all(arr >= 0)

    mv = memoryview(result)
    assert mv.ndim == 2
    assert mv.shape == (5, 7)

    arr_from_mv = np.asarray(mv)
    np.testing.assert_array_equal(arr, arr_from_mv)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_scalar_conversion():
    a = np.random.rand(128).astype(np.float64)
    b = np.random.rand(128).astype(np.float64)
    result = nk.sqeuclidean(a, b)
    assert isinstance(result, float)
    assert result >= 0

    a2 = np.random.rand(3, 128).astype(np.float64)
    b2 = np.random.rand(5, 128).astype(np.float64)
    result2 = nk.cdist(a2, b2, metric="sqeuclidean")
    with pytest.raises(TypeError):
        float(result2)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_dtype_consistency():
    for input_dtype in [np.float32, np.float64]:
        a = np.random.rand(5, 128).astype(input_dtype)
        b = np.random.rand(7, 128).astype(input_dtype)
        result = nk.cdist(a, b, metric="sqeuclidean")
        assert result.dtype == "float64"
        arr = np.asarray(result)
        assert arr.dtype == np.float64


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_strides():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    strides = result.strides
    assert isinstance(strides, tuple)
    assert len(strides) == 2
    assert strides == (7 * 8, 8)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_array_interface():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    ai = result.__array_interface__
    assert isinstance(ai, dict)
    assert "shape" in ai
    assert "typestr" in ai
    assert "data" in ai
    assert "strides" in ai
    assert "version" in ai
    assert ai["shape"] == (5, 7)
    assert ai["typestr"] == "<f8"
    assert ai["version"] == 3
    assert isinstance(ai["data"], tuple)
    assert len(ai["data"]) == 2


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_transpose():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    t = result.T
    assert t.shape == (7, 5)
    result_arr = np.asarray(result)
    t_arr = np.asarray(t)
    np.testing.assert_allclose(result_arr.T, t_arr, rtol=1e-10)

    a1d = np.random.rand(10, 128).astype(np.float64)
    b1d = np.random.rand(10, 128).astype(np.float64)
    result1d = nk.sqeuclidean(a1d, b1d)
    t1d = result1d.T
    assert t1d.shape == result1d.shape


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_str():
    a = np.random.rand(3, 128).astype(np.float64)
    b = np.random.rand(4, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    s = str(result)
    assert "[" in s
    assert "]" in s
    assert any(c.isdigit() for c in s)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_equality():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    copy = result.copy()
    assert result == copy
    assert not (result != copy)

    a2 = np.random.rand(5, 128).astype(np.float64)
    b2 = np.random.rand(7, 128).astype(np.float64)
    result2 = nk.cdist(a2, b2, metric="sqeuclidean")
    assert result != result2

    a3 = np.random.rand(3, 128).astype(np.float64)
    b3 = np.random.rand(4, 128).astype(np.float64)
    result3 = nk.cdist(a3, b3, metric="sqeuclidean")
    assert result != result3


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_copy():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    copy = result.copy()
    assert copy.shape == result.shape
    assert copy.dtype == result.dtype
    assert result == copy

    result_arr = np.asarray(result)
    copy_arr = np.asarray(copy)
    np.testing.assert_array_equal(result_arr, copy_arr)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_reshape():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    flat = result.reshape(35)
    assert flat.shape == (35,)
    assert flat.size == 35

    back = flat.reshape(5, 7)
    assert back.shape == (5, 7)

    reshaped = result.reshape((7, 5))
    assert reshaped.shape == (7, 5)

    with pytest.raises(ValueError):
        result.reshape(10, 10)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_slicing():
    a = np.random.rand(10, 128).astype(np.float64)
    b = np.random.rand(8, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")
    expected = np.asarray(result)

    sliced = result[2:5]
    assert sliced.shape == (3, 8)
    np.testing.assert_allclose(np.asarray(sliced), expected[2:5], rtol=1e-10)

    step_sliced = result[::2]
    assert step_sliced.shape == (5, 8)
    np.testing.assert_allclose(np.asarray(step_sliced), expected[::2], rtol=1e-10)

    rev_sliced = result[::-1]
    assert rev_sliced.shape == (10, 8)
    np.testing.assert_allclose(np.asarray(rev_sliced), expected[::-1], rtol=1e-10)

    end_sliced = result[-3:]
    assert end_sliced.shape == (3, 8)
    np.testing.assert_allclose(np.asarray(end_sliced), expected[-3:], rtol=1e-10)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_zero_copy_views():
    a = np.random.rand(5, 64).astype(np.float64)
    b = np.random.rand(4, 64).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")
    orig_np = np.asarray(result)

    sliced = result[1:4]
    sliced_np = np.asarray(sliced)
    assert np.shares_memory(orig_np, sliced_np)

    step_sliced = result[::2]
    step_np = np.asarray(step_sliced)
    assert np.shares_memory(orig_np, step_np)

    rev_sliced = result[::-1]
    rev_np = np.asarray(rev_sliced)
    assert np.shares_memory(orig_np, rev_np)

    row = result[0]
    row_np = np.asarray(row)
    assert np.shares_memory(orig_np, row_np)

    transposed = result.T
    trans_np = np.asarray(transposed)
    assert np.shares_memory(orig_np, trans_np)

    flat = result.reshape(20)
    flat_np = np.asarray(flat)
    assert np.shares_memory(orig_np, flat_np)

    trans_reshaped = transposed.reshape(20)
    trans_reshaped_np = np.asarray(trans_reshaped)
    assert not np.shares_memory(orig_np, trans_reshaped_np)

    chained = result[1:4][1:]
    chained_np = np.asarray(chained)
    assert np.shares_memory(orig_np, chained_np)

    for i, row in enumerate(result):
        row_np = np.asarray(row)
        assert np.shares_memory(orig_np, row_np)

    orig_val = orig_np[2, 0].copy()
    view = result[2]
    view_np = np.asarray(view)
    view_np[0] = 999.0
    assert orig_np[2, 0] == 999.0
    orig_np[2, 0] = orig_val

    copied = result.copy()
    copied_np = np.asarray(copied)
    assert not np.shares_memory(orig_np, copied_np)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("int8", id="i8"),
        pytest.param("int32", id="i32"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [pytest.param((10,), id="1d-10"), pytest.param((5, 4), id="2d-5x4"), pytest.param((2, 3, 4), id="3d-2x3x4")],
)
def test_ndarray_zeros(dtype, shape):
    arr = nk.zeros(shape, dtype=dtype)
    arr_np = np.asarray(arr)
    assert arr.shape == shape
    assert arr.dtype == dtype
    assert np.all(arr_np == 0)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("int8", id="i8"),
        pytest.param("int32", id="i32"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [pytest.param((10,), id="1d-10"), pytest.param((5, 4), id="2d-5x4"), pytest.param((2, 3, 4), id="3d-2x3x4")],
)
def test_ndarray_ones(dtype, shape):
    arr = nk.ones(shape, dtype=dtype)
    arr_np = np.asarray(arr)
    assert arr.shape == shape
    assert arr.dtype == dtype
    assert np.all(arr_np == 1)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype,fill_value",
    [
        pytest.param("float64", 3.14, id="f64-pi"),
        pytest.param("float32", -2.5, id="f32-neg"),
        pytest.param("int8", 42, id="i8-42"),
        pytest.param("int32", -100, id="i32-neg"),
    ],
)
@pytest.mark.parametrize("shape", [pytest.param((10,), id="1d-10"), pytest.param((5, 4), id="2d-5x4")])
def test_ndarray_full(dtype, fill_value, shape):
    arr = nk.full(shape, fill_value, dtype=dtype)
    arr_np = np.asarray(arr)
    assert arr.shape == shape
    assert arr.dtype == dtype
    if dtype.startswith("float"):
        np.testing.assert_allclose(arr_np, fill_value, rtol=1e-5)
    else:
        expected = np.dtype(dtype).type(fill_value)
        assert np.all(arr_np == expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
@pytest.mark.parametrize("shape", [pytest.param((10,), id="1d-10"), pytest.param((5, 4), id="2d-5x4")])
def test_ndarray_empty(dtype, shape):
    arr = nk.empty(shape, dtype=dtype)
    assert arr.shape == shape
    assert arr.dtype == dtype


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("int8", id="i8"),
        pytest.param("int32", id="i32"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [pytest.param((100,), id="1d-100"), pytest.param((10, 10), id="2d-10x10"), pytest.param((4, 5, 5), id="3d-4x5x5")],
)
def test_ndarray_sum_method(dtype, shape):
    if dtype.startswith("float"):
        np_arr = np.random.randn(*shape).astype(dtype)
    else:
        dtype_info = np.iinfo(np.dtype(dtype))
        np_arr = np.random.randint(dtype_info.min // 2, dtype_info.max // 2, size=shape, dtype=dtype)
    nk_arr = make_nk(np_arr, dtype)

    expected = np_arr.sum()
    result = nk_arr.sum()

    if dtype.startswith("float"):
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
    else:
        assert result == expected


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("int8", id="i8"),
        pytest.param("int32", id="i32"),
    ],
)
@pytest.mark.parametrize("shape", [pytest.param((100,), id="1d-100"), pytest.param((10, 10), id="2d-10x10")])
def test_ndarray_min_max_methods(dtype, shape):
    if dtype.startswith("float"):
        np_arr = np.random.randn(*shape).astype(dtype)
    else:
        dtype_info = np.iinfo(np.dtype(dtype))
        np_arr = np.random.randint(dtype_info.min // 2, dtype_info.max // 2, size=shape, dtype=dtype)
    nk_arr = make_nk(np_arr, dtype)

    if dtype.startswith("float"):
        np.testing.assert_allclose(nk_arr.min(), np_arr.min(), rtol=1e-5)
        np.testing.assert_allclose(nk_arr.max(), np_arr.max(), rtol=1e-5)
    else:
        assert nk_arr.min() == np_arr.min()
        assert nk_arr.max() == np_arr.max()


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32"), pytest.param("int32", id="i32")]
)
@pytest.mark.parametrize("shape", [pytest.param((100,), id="1d-100"), pytest.param((10, 10), id="2d-10x10")])
def test_ndarray_argmin_argmax_methods(dtype, shape):
    if dtype.startswith("float"):
        np_arr = np.random.randn(*shape).astype(dtype)
    else:
        dtype_info = np.iinfo(np.dtype(dtype))
        np_arr = np.random.randint(dtype_info.min // 2, dtype_info.max // 2, size=shape, dtype=dtype)
    nk_arr = make_nk(np_arr, dtype)

    assert nk_arr.argmin() == np_arr.argmin()
    assert nk_arr.argmax() == np_arr.argmax()


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32"), pytest.param("int32", id="i32")]
)
def test_ndarray_add_operator(dtype):
    shape = (20,)
    if dtype.startswith("float"):
        np_a = np.random.randn(*shape).astype(dtype)
        np_b = np.random.randn(*shape).astype(dtype)
    else:
        np_a = np.random.randint(-50, 50, size=shape, dtype=dtype)
        np_b = np.random.randint(-50, 50, size=shape, dtype=dtype)
    nk_a = make_nk(np_a, dtype)
    nk_b = make_nk(np_b, dtype)

    expected = np_a + np_b
    result = nk_a + nk_b
    result_np = np.asarray(result)

    if dtype.startswith("float"):
        np.testing.assert_allclose(result_np, expected, rtol=1e-5)
    else:
        np.testing.assert_array_equal(result_np, expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32"), pytest.param("int32", id="i32")]
)
def test_ndarray_subtract_operator(dtype):
    shape = (20,)
    if dtype.startswith("float"):
        np_a = np.random.randn(*shape).astype(dtype)
        np_b = np.random.randn(*shape).astype(dtype)
    else:
        np_a = np.random.randint(-50, 50, size=shape, dtype=dtype)
        np_b = np.random.randint(-50, 50, size=shape, dtype=dtype)
    nk_a = make_nk(np_a, dtype)
    nk_b = make_nk(np_b, dtype)

    expected = np_a - np_b
    result = nk_a - nk_b
    result_np = np.asarray(result)

    if dtype.startswith("float"):
        np.testing.assert_allclose(result_np, expected, rtol=1e-5)
    else:
        np.testing.assert_array_equal(result_np, expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32"), pytest.param("int32", id="i32")]
)
def test_ndarray_multiply_operator(dtype):
    shape = (20,)
    if dtype.startswith("float"):
        np_a = np.random.randn(*shape).astype(dtype)
        np_b = np.random.randn(*shape).astype(dtype)
    else:
        np_a = np.random.randint(-10, 10, size=shape, dtype=dtype)
        np_b = np.random.randint(-10, 10, size=shape, dtype=dtype)
    nk_a = make_nk(np_a, dtype)
    nk_b = make_nk(np_b, dtype)

    expected = np_a * np_b
    result = nk_a * nk_b
    result_np = np.asarray(result)

    if dtype.startswith("float"):
        np.testing.assert_allclose(result_np, expected, rtol=1e-4)
    else:
        np.testing.assert_array_equal(result_np, expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32"), pytest.param("int32", id="i32")]
)
def test_ndarray_unary_operators(dtype):
    shape = (20,)
    if dtype.startswith("float"):
        np_a = np.random.randn(*shape).astype(dtype)
    else:
        np_a = np.random.randint(-50, 50, size=shape, dtype=dtype)
    nk_a = make_nk(np_a, dtype)

    expected_neg = -np_a
    result_neg = -nk_a
    result_neg_np = np.asarray(result_neg)

    if dtype.startswith("float"):
        np.testing.assert_allclose(result_neg_np, expected_neg, rtol=1e-5)
    else:
        np.testing.assert_array_equal(result_neg_np, expected_neg)

    result_pos = +nk_a
    result_pos_np = np.asarray(result_pos)
    np.testing.assert_array_equal(result_pos_np, np_a)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_reduction_on_strided_array(dtype):
    np_arr = np.random.randn(10, 10).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)

    np_strided = np_arr[::2]
    nk_strided = nk_arr[::2]

    np.testing.assert_allclose(nk_strided.sum(), np_strided.sum(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(nk_strided.min(), np_strided.min(), rtol=1e-5)
    np.testing.assert_allclose(nk_strided.max(), np_strided.max(), rtol=1e-5)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_reduction_on_transposed_array(dtype):
    np_arr = np.random.randn(5, 8).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)

    np_t = np_arr.T
    nk_t = nk_arr.T

    np.testing.assert_allclose(nk_t.sum(), np_t.sum(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(nk_t.min(), np_t.min(), rtol=1e-5)
    np.testing.assert_allclose(nk_t.max(), np_t.max(), rtol=1e-5)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_reduction_on_subview(dtype):
    np_arr = np.random.randn(20, 20).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)

    np_sub = np_arr[5:15, 5:15]
    nk_sub = nk_arr[5:15, 5:15]

    np.testing.assert_allclose(nk_sub.sum(), np_sub.sum(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(nk_sub.min(), np_sub.min(), rtol=1e-5)
    np.testing.assert_allclose(nk_sub.max(), np_sub.max(), rtol=1e-5)
    assert nk_sub.argmin() == np_sub.argmin()
    assert nk_sub.argmax() == np_sub.argmax()


def test_bfloat16_scalar_creation():
    bf = nk.bfloat16(3.14159)
    assert isinstance(bf, nk.bfloat16)
    assert abs(float(bf) - 3.14159) < 0.01
    assert int(bf) == 3


def test_bfloat16_scalar_repr():
    bf = nk.bfloat16(1.5)
    assert "bfloat16" in repr(bf)
    assert "1.5" in str(bf)


def test_bfloat16_scalar_arithmetic():
    a = nk.bfloat16(1.5)
    b = nk.bfloat16(2.5)

    result = a + b
    assert isinstance(result, nk.bfloat16)
    assert float(result) == 4.0

    result = b - a
    assert isinstance(result, nk.bfloat16)
    assert float(result) == 1.0

    result = a * b
    assert isinstance(result, nk.bfloat16)
    assert float(result) == 3.75

    result = b / a
    assert isinstance(result, nk.bfloat16)
    assert abs(float(result) - 1.6666) < 0.01


def test_bfloat16_scalar_unary():
    a = nk.bfloat16(1.5)
    assert float(-a) == -1.5
    assert float(+a) == 1.5
    assert float(abs(nk.bfloat16(-1.5))) == 1.5
    assert bool(a) == True
    assert bool(nk.bfloat16(0.0)) == False


def test_bfloat16_scalar_comparison():
    a = nk.bfloat16(1.5)
    b = nk.bfloat16(2.5)

    assert a < b
    assert a <= b
    assert a <= a
    assert b > a
    assert b >= a
    assert a == a
    assert a != b

    assert a == 1.5
    assert a < 2.0
    assert a > 1.0


def test_bfloat16_scalar_hash():
    a = nk.bfloat16(1.5)
    b = nk.bfloat16(1.5)
    s = {a, b}
    assert len(s) == 1
    d = {a: "value"}
    assert d[b] == "value"


def test_float8_e4m3_scalar():
    f8 = nk.float8_e4m3(1.5)
    assert isinstance(f8, nk.float8_e4m3)
    assert float(f8) == 1.5
    assert int(f8) == 1
    assert "float8_e4m3" in repr(f8)
    assert float(-f8) == -1.5
    assert bool(f8) == True
    assert bool(nk.float8_e4m3(0.0)) == False


def test_float8_e4m3_scalar_arithmetic():
    a = nk.float8_e4m3(1.5)
    b = nk.float8_e4m3(2.0)

    result = a + b
    assert isinstance(result, nk.float8_e4m3)
    assert abs(float(result) - 3.5) < 0.5

    result = b - a
    assert isinstance(result, nk.float8_e4m3)
    assert abs(float(result) - 0.5) < 0.5

    result = a * b
    assert isinstance(result, nk.float8_e4m3)
    assert abs(float(result) - 3.0) < 0.5

    result = b / a
    assert isinstance(result, nk.float8_e4m3)
    assert abs(float(result) - 1.33) < 0.5

    assert float(+a) == float(a)
    assert float(abs(nk.float8_e4m3(-1.5))) == 1.5


def test_float8_e5m2_scalar():
    f8 = nk.float8_e5m2(1.5)
    assert isinstance(f8, nk.float8_e5m2)
    assert float(f8) == 1.5
    assert int(f8) == 1
    assert "float8_e5m2" in repr(f8)
    assert float(-f8) == -1.5
    assert bool(f8) == True
    assert bool(nk.float8_e5m2(0.0)) == False


def test_float8_e5m2_scalar_arithmetic():
    a = nk.float8_e5m2(1.5)
    b = nk.float8_e5m2(2.0)

    result = a + b
    assert isinstance(result, nk.float8_e5m2)
    assert abs(float(result) - 3.5) < 0.5

    result = b - a
    assert isinstance(result, nk.float8_e5m2)
    assert abs(float(result) - 0.5) < 0.5

    result = a * b
    assert isinstance(result, nk.float8_e5m2)
    assert abs(float(result) - 3.0) < 0.5

    result = b / a
    assert isinstance(result, nk.float8_e5m2)
    assert abs(float(result) - 1.33) < 0.5

    assert float(+a) == float(a)
    assert float(abs(nk.float8_e5m2(-1.5))) == 1.5


@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes not installed")
def test_bfloat16_vs_ml_dtypes():
    test_values = [0.0, 1.0, -1.0, 3.14159, 100.0, 0.001, -65504.0]
    for val in test_values:
        nk_bf16 = nk.bfloat16(val)
        ml_bf16 = ml_dtypes.bfloat16(val)
        assert float(nk_bf16) == float(ml_bf16), f"Mismatch for {val}: nk={float(nk_bf16)}, ml={float(ml_bf16)}"


@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes not installed")
def test_float8_e4m3_vs_ml_dtypes():
    test_values = [0.0, 1.0, -1.0, 1.5, 2.0, 0.5]
    for val in test_values:
        nk_f8 = nk.float8_e4m3(val)
        ml_f8 = ml_dtypes.float8_e4m3fn(val)
        assert float(nk_f8) == float(ml_f8), f"Mismatch for {val}: nk={float(nk_f8)}, ml={float(ml_f8)}"


@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes not installed")
def test_float8_e5m2_vs_ml_dtypes():
    test_values = [0.0, 1.0, -1.0, 1.5, 2.0, 0.5]
    for val in test_values:
        nk_f8 = nk.float8_e5m2(val)
        ml_f8 = ml_dtypes.float8_e5m2(val)
        assert float(nk_f8) == float(ml_f8), f"Mismatch for {val}: nk={float(nk_f8)}, ml={float(ml_f8)}"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_gil_free_threading():
    """Test NumKong in Python 3.13t free-threaded mode if available."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 13:
        is_free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        if not is_free_threaded:
            pytest.skip("Uses non-free-threaded Python, skipping GIL-related tests")
        if sys._is_gil_enabled():
            pytest.skip("GIL is enabled, skipping GIL-related tests")
    else:
        pytest.skip("Python < 3.13t, skipping GIL-related tests")

    num_threads = multiprocessing.cpu_count()
    vectors_a = np.random.rand(32 * 1024 * num_threads, 1024).astype(np.float32)
    vectors_b = np.random.rand(32 * 1024 * num_threads, 1024).astype(np.float32)
    distances = np.zeros(vectors_a.shape[0], dtype=np.float32)

    def compute_batch(start_idx, end_idx) -> float:
        slice_a = vectors_a[start_idx:end_idx]
        slice_b = vectors_b[start_idx:end_idx]
        slice_distances = distances[start_idx:end_idx]
        nk.angular(slice_a, slice_b, out=slice_distances)
        return sum(slice_distances)

    def compute_with_threads(threads: int) -> float:
        chunk_size = len(vectors_a) // threads
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for i in range(threads):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < threads - 1 else len(vectors_a)
                futures.append(executor.submit(compute_batch, start_idx, end_idx))
        total_sum = 0.0
        for future in concurrent.futures.as_completed(futures):
            total_sum += future.result()
        return total_sum

    start_time = time.time()
    baseline_sum = compute_with_threads(2)
    end_time = time.time()
    baseline_duration = end_time - start_time

    start_time = time.time()
    multi_sum = compute_with_threads(num_threads)
    end_time = time.time()
    multi_duration = end_time - start_time

    assert np.allclose(
        baseline_sum, multi_sum, atol=NK_ATOL, rtol=NK_RTOL
    ), f"Results differ: baseline {baseline_sum} vs multi-threaded {multi_sum}"

    if baseline_duration < multi_duration:
        warnings.warn(
            f"{num_threads}-threaded execution took longer than 2-threaded baseline: {multi_duration:.2f}s vs {baseline_duration:.2f}s",
            UserWarning,
        )
