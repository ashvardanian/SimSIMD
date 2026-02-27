#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: test_base.py

Shared test infrastructure for the NumKong test suite.
Mirrors ``test.hpp`` from the C++ test suite.

Provides helpers, random data generators, capability detection,
and configuration used by all test files. Operation-specific baselines
live in their respective test files (test_dot.py, test_spatial.py, etc.).

Key exported symbols
--------------------
Helpers:
    make_random(shape, dtype)       Unified random-data factory → (raw, baseline).
    make_nk(np_arr, dtype)          Copy NumPy array into an ``nk.Tensor``.
    tolerances_for_dtype(dtype)     Returns ``(atol, rtol)`` for assertion checks.
    random_of_dtype(dtype, shape)   Legacy wrapper around ``make_random``.
Constants:
    NATIVE_COMPUTE_DTYPE            Maps dtype → NumPy dtype for native-precision baselines.
    EXOTIC_DTYPES                   Set of dtypes whose raw bytes aren't NumPy-readable.
    NK_ATOL, NK_RTOL                Default assertion tolerances.
    _DECIMAL_PRECISION              Shared decimal precision for high-accuracy baselines.

Profiling:
    profile(callable, ...)          Time a callable, return ``(ns, result)``.

Fixtures:
    _seed_rng                       Auto-seeds NumPy RNG (autouse).

Stats:
    create_stats / collect_errors / collect_warnings / print_stats_report

**Environment Variables** (parity with ``test.cpp`` / ``test.hpp``)::

    NK_DENSE_DIMENSIONS   Vector dims for dot/spatial/geo tests (default: "11,97,1536")
    NK_CURVED_DIMENSIONS  Vector dims for curved-space tests (default: "11,97")
    NK_MATRIX_HEIGHT      GEMM M dimensions (default: "1024")
    NK_MATRIX_WIDTH       GEMM N dimensions (default: "128")
    NK_MATRIX_DEPTH       GEMM K dimensions (default: "1536")
    NK_SEED               Deterministic seed for np.random (default: None = OS entropy)
    NK_REPETITIONS        Override randomized test repeat count (default: 10)
"""

import os
import sys
import math
import time
import platform
import collections
import warnings
import faulthandler

import array as _array
import random as _random

import tabulate
import pytest
import numkong as nk

faulthandler.enable()

_nk_seed_str = os.environ.get("NK_SEED")
nk_seed: int | None = int(_nk_seed_str) if _nk_seed_str is not None else None

_nk_reps_str = os.environ.get("NK_REPETITIONS")
randomized_repetitions_count: int = int(_nk_reps_str) if _nk_reps_str is not None else 10

_nk_dense_str = os.environ.get("NK_DENSE_DIMENSIONS")
dense_dimensions: list[int] = (
    [int(d) for d in _nk_dense_str.split(",")]
    if _nk_dense_str is not None
    else [1, 2, 3, 4, 7, 8, 15, 16, 11, 97, 1536]
)

_nk_curved_str = os.environ.get("NK_CURVED_DIMENSIONS")
curved_dimensions: list[int] = [int(d) for d in _nk_curved_str.split(",")] if _nk_curved_str is not None else [11, 97]

_nk_matrix_h_str = os.environ.get("NK_MATRIX_HEIGHT")
matrix_heights: list[int] = [int(d) for d in _nk_matrix_h_str.split(",")] if _nk_matrix_h_str is not None else [1024]

_nk_matrix_w_str = os.environ.get("NK_MATRIX_WIDTH")
matrix_widths: list[int] = [int(d) for d in _nk_matrix_w_str.split(",")] if _nk_matrix_w_str is not None else [128]

_nk_matrix_d_str = os.environ.get("NK_MATRIX_DEPTH")
matrix_depths: list[int] = [int(d) for d in _nk_matrix_d_str.split(",")] if _nk_matrix_d_str is not None else [1536]

try:
    import numpy as np

    numpy_available = True
except:
    np = None
    numpy_available = False

try:
    import scipy.spatial.distance

    scipy_available = True
except ImportError:
    scipy_available = False


try:
    import ml_dtypes

    ml_dtypes_available = True
except ImportError:
    ml_dtypes_available = False


NK_RTOL = 0.1
NK_ATOL = 0.1

# Map dtype → the NumPy dtype used for "native precision" baseline computation.
# f64 types compute at f64; everything else at f32 (for floats) or i64 (for ints).
NATIVE_COMPUTE_DTYPE: dict[str, type] = (
    {
        "float64": np.float64,
        "float32": np.float32,
        "float16": np.float32,
        "bfloat16": np.float32,
        "bf16": np.float32,
        "e4m3": np.float32,
        "e5m2": np.float32,
        "e2m3": np.float32,
        "e3m2": np.float32,
        "int8": np.int64,
        "uint8": np.int64,
        "int16": np.int64,
        "uint16": np.int64,
        "int32": np.int64,
        "uint32": np.int64,
        "int64": np.int64,
        "uint64": np.int64,
        "int4": np.int64,
        "uint4": np.int64,
        "complex64": np.complex128,
        "complex128": np.complex128,
    }
    if numpy_available
    else {}
)


def _is_numpy_native(name: str) -> bool:
    try:
        return np.dtype(name).type.__module__ == "numpy"
    except TypeError:
        return False


# True for any dtype whose raw representation isn't directly readable by NumPy.
EXOTIC_DTYPES: set[str] = {k for k in NATIVE_COMPUTE_DTYPE if not _is_numpy_native(k)}

_DECIMAL_PRECISION = 120


def is_running_under_qemu():
    return "NK_IN_QEMU" in os.environ


def profile(callable, *args, **kwargs) -> tuple:
    before = time.perf_counter_ns()
    result = callable(*args, **kwargs)
    after = time.perf_counter_ns()
    return after - before, result


def scipy_metric_name(metric: str) -> str:
    """Convert NumKong metric names to SciPy equivalents."""
    if metric == "angular":
        return "cosine"
    return metric


def to_array(x, dtype=None):
    if numpy_available:
        y = np.array(x)
        if dtype is not None:
            y = y.astype(dtype)
        return y


def tolerances_for_dtype(dtype: str) -> tuple[float, float]:
    """Returns ``(atol, rtol)`` appropriate for assertions on the given dtype.

    Integer dtypes: exact ±1 (discrete arithmetic, accumulator width differences).
    Everything else: ``(NK_ATOL, NK_RTOL)``.

    """
    if dtype in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "int4", "uint4"):
        return 1, 0
    return NK_ATOL, NK_RTOL


def random_of_dtype(dtype, shape):
    """Legacy helper — thin wrapper around :func:`make_random`."""
    raw, _ = make_random(shape, dtype)
    return raw


class LazyFormat:
    """Deferred string formatting — only evaluated when str() is called (on assertion failure)."""

    __slots__ = ("_fn",)

    def __init__(self, fn):
        self._fn = fn

    def __str__(self):
        return self._fn()


def f32_downcast_to_bf16(array):
    """Converts an array of 32-bit floats into 16-bit brain-floats."""
    array = np.asarray(array, dtype=np.float32)
    array_f32_rounded = ((array.view(np.uint32) + 0x8000) & 0xFFFF0000).view(np.float32)
    array_bf16 = np.right_shift(array_f32_rounded.view(np.uint32), 16).astype(np.uint16)
    return array_f32_rounded, array_bf16


def i8_downcast_to_i4(array):
    """Pack signed 8-bit integers into signed 4-bit pairs (2 per byte).

    Layout matches C ``nk_i4x2_t``: low nibble = even index, high nibble = odd index.
    Input values must be in [-8, 7].
    """
    array = np.asarray(array, dtype=np.int8)
    assert np.all(array >= -8) and np.all(array <= 7), "values must be in [-8, 7]"
    flat = array.ravel()
    if len(flat) % 2:
        flat = np.append(flat, np.int8(0))
    low = flat[0::2].astype(np.uint8) & 0x0F
    high = (flat[1::2].astype(np.uint8) & 0x0F) << 4
    return (low | high).astype(np.uint8)


def u8_downcast_to_u4(array):
    """Pack unsigned 8-bit integers into unsigned 4-bit pairs (2 per byte).

    Layout matches C ``nk_u4x2_t``: low nibble = even index, high nibble = odd index.
    Input values must be in [0, 15].
    """
    array = np.asarray(array, dtype=np.uint8)
    assert np.all(array <= 15), "values must be in [0, 15]"
    flat = array.ravel()
    if len(flat) % 2:
        flat = np.append(flat, np.uint8(0))
    low = flat[0::2] & 0x0F
    high = (flat[1::2] & 0x0F) << 4
    return (low | high).astype(np.uint8)


def hex_array(arr):
    """Converts numerical array into a string of comma-separated hexadecimal values for debugging."""
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        # View non-integer data as raw bytes for hex display
        shape = arr.shape
        arr = arr.view(np.uint8).reshape(shape + (-1,))
    printer = np.vectorize(hex)
    strings = printer(arr)
    if strings.ndim == 1:
        return ", ".join(strings)
    else:
        return "\n".join(", ".join(row) for row in strings.reshape(-1, strings.shape[-1]))


# Lookup tables for sub-byte float types.
# Built once from the encoding rules in ``include/numkong/cast/serial.h``.
# Each table maps a raw byte value to its float64 representation.
# NaN entries are stored as ``float('nan')``.


def _build_lut(sign_bit, exp_bits, mant_bits, bias, total_bits, has_inf=False, nan_only_max_mant=False):
    """Build a byte→float64 lookup table for a sub-byte float format.

    Args:
        sign_bit:           position of the sign bit (counting from bit 0)
        exp_bits:           number of exponent bits
        mant_bits:          number of mantissa bits
        bias:               exponent bias
        total_bits:         number of significant bits (6 for float6, 8 for float8)
        has_inf:            if True, max exponent with zero mantissa = ±∞
        nan_only_max_mant:  if True, only max_exp + max_mant is NaN (e4m3 rule)
    """
    n = 1 << total_bits
    exp_mask = (1 << exp_bits) - 1
    mant_mask = (1 << mant_bits) - 1
    max_exp = exp_mask
    max_mant = mant_mask
    lut = [0.0] * n
    for i in range(n):
        sign = (i >> sign_bit) & 1
        exp = (i >> mant_bits) & exp_mask
        mant = i & mant_mask
        if exp == 0:
            # Subnormal: value = (−1)^s × 2^(1−bias) × (mant / 2^mant_bits)
            val = (mant / (1 << mant_bits)) * (2.0 ** (1 - bias))
        elif exp == max_exp:
            if has_inf and mant == 0:
                lut[i] = -float("inf") if sign else float("inf")
                continue
            if has_inf and mant != 0:
                lut[i] = float("nan")
                continue
            if nan_only_max_mant and mant == max_mant:
                lut[i] = float("nan")
                continue
            # Finite max-exponent value
            val = 2.0 ** (exp - bias) * (1.0 + mant / (1 << mant_bits))
        else:
            val = 2.0 ** (exp - bias) * (1.0 + mant / (1 << mant_bits))
        lut[i] = -val if sign else val
    return lut


_LUT_E2M3 = _build_lut(sign_bit=5, exp_bits=2, mant_bits=3, bias=1, total_bits=6)
_LUT_E3M2 = _build_lut(sign_bit=5, exp_bits=3, mant_bits=2, bias=3, total_bits=6)
_LUT_E4M3 = _build_lut(sign_bit=7, exp_bits=4, mant_bits=3, bias=7, total_bits=8, nan_only_max_mant=True)
_LUT_E5M2 = _build_lut(sign_bit=7, exp_bits=5, mant_bits=2, bias=15, total_bits=8, has_inf=True)

_SUBBYTE_LUTS = {"e2m3": _LUT_E2M3, "e3m2": _LUT_E3M2, "e4m3": _LUT_E4M3, "e5m2": _LUT_E5M2}


def make_random(shape, dtype):
    """Unified random-data factory.

    Returns ``(raw, baseline)`` where:

    - *raw*: data in the dtype's storage format, suitable for SIMD kernels.
    - *baseline*: ``float64`` (or ``complex128``) array for reference comparison.

    For exotic types the raw array uses a NumPy-native storage dtype
    (``uint16`` for bf16, ``uint8`` for float8/float6).
    """
    if isinstance(shape, int):
        shape = (shape,)

    if dtype in ("float64", "float32", "float16"):
        raw = np.random.randn(*shape).astype(dtype)
        baseline = raw.astype(np.float64)
        return raw, baseline

    if dtype in ("bfloat16", "bf16"):
        f32_arr = np.random.randn(*shape).astype(np.float32)
        f32_rounded, bf16_raw = f32_downcast_to_bf16(f32_arr)
        baseline = f32_rounded.astype(np.float64)
        return bf16_raw, baseline

    if dtype in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"):
        info = np.iinfo(np.dtype(dtype))
        raw = np.random.randint(info.min, info.max, size=shape, dtype=dtype)
        baseline = raw.astype(np.float64)
        return raw, baseline

    if dtype in ("complex64", "complex128"):
        raw = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        baseline = raw.astype(np.complex128)
        return raw, baseline

    if dtype in ("e4m3", "e5m2", "e2m3", "e3m2"):
        lut = np.array(_SUBBYTE_LUTS[dtype])
        # Exclude NaN/±∞ entries from random generation
        finite_mask = np.isfinite(lut)
        valid_bytes = np.where(finite_mask)[0].astype(np.uint8)
        raw = valid_bytes[np.random.randint(0, len(valid_bytes), size=shape)]
        baseline = lut[raw.astype(int)]
        return raw, baseline

    if dtype == "int4":
        values = np.random.randint(-8, 8, size=shape, dtype=np.int8)
        baseline = values.astype(np.float64)
        raw = i8_downcast_to_i4(values)
        return raw, baseline

    if dtype == "uint4":
        values = np.random.randint(0, 16, size=shape, dtype=np.uint8)
        baseline = values.astype(np.float64)
        raw = u8_downcast_to_u4(values)
        return raw, baseline

    raise ValueError(f"Unsupported dtype for make_random: {dtype}")


def make_nk(np_arr, dtype=None):
    """Copy a NumPy array into a NumKong tensor.

    If *dtype* is ``None`` it is inferred from ``np_arr.dtype``.
    """
    if dtype is None:
        dtype = str(np_arr.dtype)
    nk_arr = nk.zeros(np_arr.shape, dtype=dtype)
    np.copyto(np.asarray(nk_arr), np_arr)
    return nk_arr


available_capabilities: dict[str, str] = nk.get_capabilities()

# fmt: off
possible_x86_capabilities: list[str] = [
    "haswell", "skylake", "icelake", "genoa", "sapphire", "sapphireamx", "graniteamx", "turin", "sierra",
]
possible_arm_capabilities: list[str] = [
    "neon", "neonhalf", "neonfhm", "neonbfdot", "neonsdot",
    "sve", "svehalf", "svebfdot", "svesdot", "sve2", "sve2p1",
    "sme", "sme2", "sme2p1", "smef64", "smehalf", "smebf16", "smelut2", "smefa64",
]
possible_rvv_capabilities: list[str] = ["rvv", "rvvhalf", "rvvbf16", "rvvbb"]
possible_wasm_capabilities: list[str] = ["v128relaxed"]
# fmt: on

possible_x86_capabilities = [c for c in possible_x86_capabilities if available_capabilities[c]]
possible_arm_capabilities = [c for c in possible_arm_capabilities if available_capabilities[c]]
possible_rvv_capabilities = [c for c in possible_rvv_capabilities if available_capabilities[c]]
possible_wasm_capabilities = [c for c in possible_wasm_capabilities if available_capabilities[c]]

hardware_capabilities: list[str] = []
_machine = platform.machine()

if sys.platform == "linux":
    if _machine == "x86_64":
        hardware_capabilities = possible_x86_capabilities
    elif _machine == "aarch64":
        hardware_capabilities = possible_arm_capabilities
    elif _machine == "riscv64":
        hardware_capabilities = possible_rvv_capabilities
elif sys.platform == "darwin":
    if _machine == "x86_64":
        hardware_capabilities = possible_x86_capabilities
    elif _machine == "arm64":
        hardware_capabilities = possible_arm_capabilities
elif sys.platform == "win32":
    if _machine == "AMD64":
        hardware_capabilities = possible_x86_capabilities
    elif _machine == "ARM64":
        hardware_capabilities = possible_arm_capabilities
elif sys.platform.startswith("freebsd"):
    if _machine == "amd64":
        hardware_capabilities = possible_x86_capabilities
    elif _machine == "arm64":
        hardware_capabilities = possible_arm_capabilities
elif sys.platform in ("emscripten", "wasi"):
    hardware_capabilities = possible_wasm_capabilities

possible_capabilities: list[str] = ["serial"] + hardware_capabilities

_current_capability: str | None = None


def keep_one_capability(cap: str):
    global _current_capability
    assert cap in possible_capabilities, f"Capability {cap} is not available on this platform."
    if cap == _current_capability:
        return
    for c in possible_capabilities:
        if c != cap and c != "serial":
            nk.disable_capability(c)
    if cap != "serial":
        nk.enable_capability(cap)
    _current_capability = cap


def create_stats():
    """Create a fresh stats dict for error collection."""
    return {
        "metric": [],
        "ndim": [],
        "dtype": [],
        "absolute_baseline_error": [],
        "relative_baseline_error": [],
        "absolute_nk_error": [],
        "relative_nk_error": [],
        "accurate_duration": [],
        "baseline_duration": [],
        "nk_duration": [],
        "warnings": [],
    }


def collect_errors(
    metric: str,
    ndim: int,
    dtype: str,
    accurate_result: float,
    accurate_duration: float,
    baseline_result: float,
    baseline_duration: float,
    nk_result: float,
    nk_duration: float,
    stats,
):
    """Calculates and aggregates errors for a given test."""
    accurate_result = np.asarray(accurate_result)
    eps = np.finfo(accurate_result.dtype).resolution
    absolute_baseline_error = np.max(np.abs(baseline_result - accurate_result))
    relative_baseline_error = np.max(np.abs(baseline_result - accurate_result) / (np.abs(accurate_result) + eps))
    absolute_nk_error = np.max(np.abs(nk_result - accurate_result))
    relative_nk_error = np.max(np.abs(nk_result - accurate_result) / (np.abs(accurate_result) + eps))

    stats["metric"].append(metric)
    stats["ndim"].append(ndim)
    stats["dtype"].append(dtype)
    stats["absolute_baseline_error"].append(absolute_baseline_error)
    stats["relative_baseline_error"].append(relative_baseline_error)
    stats["absolute_nk_error"].append(absolute_nk_error)
    stats["relative_nk_error"].append(relative_nk_error)
    stats["accurate_duration"].append(accurate_duration)
    stats["baseline_duration"].append(baseline_duration)
    stats["nk_duration"].append(nk_duration)


def collect_warnings(message: str, stats: dict):
    """Collects warnings for the final report."""
    full_name = os.environ.get("PYTEST_CURRENT_TEST", "unknown::unknown").split(" ")[0]
    function_name = full_name.split("::")[-1].split("[")[0]
    stats["warnings"].append((function_name, message))


def print_stats_report(stats):
    """Print the error aggregation report from collected stats."""
    if not stats["metric"]:
        return

    grouped_errors = collections.defaultdict(
        lambda: {
            "absolute_baseline_error": [],
            "relative_baseline_error": [],
            "absolute_nk_error": [],
            "relative_nk_error": [],
            "accurate_duration": [],
            "baseline_duration": [],
            "nk_duration": [],
        }
    )
    for (
        metric,
        ndim,
        dtype,
        absolute_baseline_error,
        relative_baseline_error,
        absolute_nk_error,
        relative_nk_error,
        accurate_duration,
        baseline_duration,
        nk_duration,
    ) in zip(
        stats["metric"],
        stats["ndim"],
        stats["dtype"],
        stats["absolute_baseline_error"],
        stats["relative_baseline_error"],
        stats["absolute_nk_error"],
        stats["relative_nk_error"],
        stats["accurate_duration"],
        stats["baseline_duration"],
        stats["nk_duration"],
    ):
        key = (metric, ndim, dtype)
        grouped_errors[key]["absolute_baseline_error"].append(absolute_baseline_error)
        grouped_errors[key]["relative_baseline_error"].append(relative_baseline_error)
        grouped_errors[key]["absolute_nk_error"].append(absolute_nk_error)
        grouped_errors[key]["relative_nk_error"].append(relative_nk_error)
        grouped_errors[key]["accurate_duration"].append(accurate_duration)
        grouped_errors[key]["baseline_duration"].append(baseline_duration)
        grouped_errors[key]["nk_duration"].append(nk_duration)

    final_results = []
    for key, errors in grouped_errors.items():
        n = len(errors["nk_duration"])
        baseline_errors = errors["relative_baseline_error"]
        nk_errors = errors["relative_nk_error"]
        baseline_mean = float(sum(baseline_errors)) / n
        nk_mean = float(sum(nk_errors)) / n
        baseline_std = math.sqrt(sum((x - baseline_mean) ** 2 for x in baseline_errors) / n)
        nk_std = math.sqrt(sum((x - nk_mean) ** 2 for x in nk_errors) / n)
        baseline_error_formatted = f"{baseline_mean:.2e} +/- {baseline_std:.2e}"
        nk_error_formatted = f"{nk_mean:.2e} +/- {nk_std:.2e}"

        accurate_durations = errors["accurate_duration"]
        baseline_durations = errors["baseline_duration"]
        nk_durations = errors["nk_duration"]
        accurate_mean_duration = sum(accurate_durations) / n
        baseline_mean_duration = sum(baseline_durations) / n
        nk_mean_duration = sum(nk_durations) / n
        accurate_std_duration = math.sqrt(sum((x - accurate_mean_duration) ** 2 for x in accurate_durations) / n)
        baseline_std_duration = math.sqrt(sum((x - baseline_mean_duration) ** 2 for x in baseline_durations) / n)
        nk_std_duration = math.sqrt(sum((x - nk_mean_duration) ** 2 for x in nk_durations) / n)
        accurate_duration_str = f"{accurate_mean_duration:.2e} +/- {accurate_std_duration:.2e}"
        baseline_duration_str = f"{baseline_mean_duration:.2e} +/- {baseline_std_duration:.2e}"
        nk_duration_str = f"{nk_mean_duration:.2e} +/- {nk_std_duration:.2e}"

        improvements = [baseline / numkong for baseline, numkong in zip(baseline_durations, nk_durations)]
        improvements_mean = sum(improvements) / n
        improvements_std = math.sqrt(sum((x - improvements_mean) ** 2 for x in improvements) / n)
        nk_speedup = f"{improvements_mean:.2f}x +/- {improvements_std:.2f}x"

        final_results.append(
            (
                *key,
                baseline_error_formatted,
                nk_error_formatted,
                accurate_duration_str,
                baseline_duration_str,
                nk_duration_str,
                nk_speedup,
            )
        )

    final_results.sort(key=lambda x: (x[0], x[1], x[2]))

    print("\n")
    print("Numerical Error Aggregation Report:")
    headers = [
        "Metric",
        "NDim",
        "DType",
        "Baseline Error",
        "NumKong Error",
        "Accurate Duration",
        "Baseline Duration",
        "NumKong Duration",
        "NumKong Speedup",
    ]
    print(tabulate.tabulate(final_results, headers=headers, tablefmt="pretty", showindex=True))

    warnings_list = stats.get("warnings", [])
    warnings_list = sorted(warnings_list)
    warnings_list = [f"{name}: {message}" for name, message in warnings_list]
    if len(warnings_list) != 0:
        print("\nWarnings:")
        warning_counts = collections.Counter(warnings_list)
        for warning, count in sorted(warning_counts.items()):
            print(f"- {count}x times: {warning}")


@pytest.fixture(autouse=True)
def _seed_rng(__pytest_repeat_step_number):
    """Auto-seed NumPy RNG before every test. When NK_SEED is set, each
    @pytest.mark.repeat() step gets a unique derived seed."""
    if not numpy_available:
        return
    step = __pytest_repeat_step_number or 0
    if nk_seed is not None:
        np.random.seed(nk_seed + step)


# Map nk dtype → (array.array typecode, low, high)
_ARRAY_TYPECODES = {
    "float32": ("f", -10.0, 10.0),
    "float64": ("d", -10.0, 10.0),
    "int8": ("b", -128, 127),
    "uint8": ("B", 0, 255),
}


def make_random_buffer(n, dtype="float32"):
    """Create a random array.array for the given dtype — no numpy needed."""
    tc, lo, hi = _ARRAY_TYPECODES[dtype]
    if tc in ("f", "d"):
        return _array.array(tc, [_random.uniform(lo, hi) for _ in range(n)])
    else:
        return _array.array(tc, [_random.randint(int(lo), int(hi)) for _ in range(n)])


def make_positive_buffer(n, dtype="float32"):
    """Create a random positive array.array — for probability distributions."""
    tc = "f" if dtype == "float32" else "d"
    vals = [_random.uniform(0.01, 1.0) for _ in range(n)]
    total = sum(vals)
    return _array.array(tc, [v / total for v in vals])
