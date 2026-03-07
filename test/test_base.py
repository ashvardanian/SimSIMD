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
    NK_ATOL, NK_RTOL                Default assertion tolerances.
    DECIMAL_PRECISION              Shared decimal precision for high-accuracy baselines.

Profiling:
    profile(callable, ...)          Time a callable, return ``(ns, result)``.

Fixtures:
    seed_rng                       Auto-seeds NumPy RNG (autouse).

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

from __future__ import annotations

import os
import sys
import math
import time
import platform
import collections
import warnings
import faulthandler

import array
import random

import pytest
import numkong as nk

faulthandler.enable()

nk_seed: int | None = int(s) if (s := os.environ.get("NK_SEED")) is not None else None

randomized_repetitions_count: int = int(s) if (s := os.environ.get("NK_REPETITIONS")) is not None else 10

dense_dimensions: list[int] = (
    [int(d) for d in s.split(",")]
    if (s := os.environ.get("NK_DENSE_DIMENSIONS")) is not None
    else [1, 2, 3, 4, 7, 8, 15, 16, 11, 97, 1536]
)

curved_dimensions: list[int] = (
    [int(d) for d in s.split(",")] if (s := os.environ.get("NK_CURVED_DIMENSIONS")) is not None else [11, 97]
)

matrix_heights: list[int] = (
    [int(d) for d in s.split(",")] if (s := os.environ.get("NK_MATRIX_HEIGHT")) is not None else [1024]
)

matrix_widths: list[int] = (
    [int(d) for d in s.split(",")] if (s := os.environ.get("NK_MATRIX_WIDTH")) is not None else [128]
)

matrix_depths: list[int] = (
    [int(d) for d in s.split(",")] if (s := os.environ.get("NK_MATRIX_DEPTH")) is not None else [1536]
)

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


DECIMAL_PRECISION = 120


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
    """Converts an array of 32-bit floats into 16-bit brain-floats.

    Uses IEEE 754 round-to-nearest-even (banker's rounding) to match
    ml_dtypes.bfloat16 behavior.
    """
    array = np.asarray(array, dtype=np.float32)
    u32 = array.view(np.uint32)
    lower = u32 & np.uint32(0xFFFF)
    # For exact ties (lower 16 bits == 0x8000), round to even:
    # only round up when the bf16 mantissa LSB (bit 16) is odd.
    is_tie = lower == np.uint32(0x8000)
    lsb = (u32 >> np.uint32(16)) & np.uint32(1)
    adjustment = np.where(is_tie, lsb << np.uint32(15), np.uint32(0x8000))
    rounded_u32 = (u32 + adjustment) & np.uint32(0xFFFF0000)
    array_f32_rounded = rounded_u32.view(np.float32)
    array_bf16 = np.right_shift(rounded_u32, 16).astype(np.uint16)
    return array_f32_rounded, array_bf16


def _pack_nibbles(array):
    """Pack pairs of nibbles along the last axis, preserving leading dimensions.

    Each pair of consecutive elements is packed into one byte:
    low nibble = even index, high nibble = odd index.
    Odd-length rows are zero-padded.  Returns a uint8 array.
    """
    shape = array.shape
    if array.ndim >= 2:
        rows = array.reshape(-1, shape[-1])
        cols = shape[-1]
        packed_cols = (cols + 1) // 2
        if cols % 2:
            rows = np.concatenate([rows, np.zeros((rows.shape[0], 1), dtype=np.uint8)], axis=1)
        low = rows[:, 0::2].astype(np.uint8) & 0x0F
        high = (rows[:, 1::2].astype(np.uint8) & 0x0F) << 4
        packed = (low | high).astype(np.uint8)
        return packed.reshape(*shape[:-1], packed_cols)
    else:
        flat = array.ravel()
        if len(flat) % 2:
            flat = np.append(flat, np.uint8(0))
        low = flat[0::2].astype(np.uint8) & 0x0F
        high = (flat[1::2].astype(np.uint8) & 0x0F) << 4
        return (low | high).astype(np.uint8)


def i8_downcast_to_i4(array):
    """Pack signed 8-bit integers into signed 4-bit pairs (2 per byte).

    Layout matches C ``nk_i4x2_t``: low nibble = even index, high nibble = odd index.
    Input values must be in [-8, 7].  Preserves leading dimensions for 2-D+ inputs.
    """
    array = np.asarray(array, dtype=np.int8)
    assert np.all(array >= -8) and np.all(array <= 7), "values must be in [-8, 7]"
    return _pack_nibbles(array)


def u8_downcast_to_u4(array):
    """Pack unsigned 8-bit integers into unsigned 4-bit pairs (2 per byte).

    Layout matches C ``nk_u4x2_t``: low nibble = even index, high nibble = odd index.
    Input values must be in [0, 15].  Preserves leading dimensions for 2-D+ inputs.
    """
    array = np.asarray(array, dtype=np.uint8)
    assert np.all(array <= 15), "values must be in [0, 15]"
    return _pack_nibbles(array)


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


def build_subbyte_float_lookup_table(
    sign_bit, exp_bits, mant_bits, bias, total_bits, has_inf=False, nan_only_max_mant=False
):
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


LOOKUP_TABLE_E2M3 = build_subbyte_float_lookup_table(sign_bit=5, exp_bits=2, mant_bits=3, bias=1, total_bits=6)
LOOKUP_TABLE_E3M2 = build_subbyte_float_lookup_table(sign_bit=5, exp_bits=3, mant_bits=2, bias=3, total_bits=6)
LOOKUP_TABLE_E4M3 = build_subbyte_float_lookup_table(
    sign_bit=7, exp_bits=4, mant_bits=3, bias=7, total_bits=8, nan_only_max_mant=True
)
LOOKUP_TABLE_E5M2 = build_subbyte_float_lookup_table(
    sign_bit=7, exp_bits=5, mant_bits=2, bias=15, total_bits=8, has_inf=True
)

SUBBYTE_LOOKUP_TABLES = {
    "e2m3": LOOKUP_TABLE_E2M3,
    "e3m2": LOOKUP_TABLE_E3M2,
    "e4m3": LOOKUP_TABLE_E4M3,
    "e5m2": LOOKUP_TABLE_E5M2,
}


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
        lut = np.array(SUBBYTE_LOOKUP_TABLES[dtype])
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
    """Copy a NumPy array into a NumKong tensor."""
    if dtype is None:
        dtype = str(np_arr.dtype)
    nk_arr = nk.zeros(np_arr.shape, dtype=dtype)
    dst = np.asarray(nk_arr)
    src = np.ascontiguousarray(np_arr)
    if dst.dtype != src.dtype:
        src = src.view(np.uint8).reshape(dst.shape)
    np.copyto(dst, src)
    return nk_arr


def downcast_f32_to_dtype(f32_arr, dtype):
    """Downcast an f32 array to *dtype*, returning (raw, f64_baseline).

    For native NumPy dtypes (float16/32/64, int*), casts directly.
    For bfloat16, uses round-to-nearest bf16 truncation.
    The baseline is always derived from the *actually stored* values
    (post-quantization), not the original f32.
    """
    if dtype in ("bfloat16", "bf16"):
        f32_rounded, raw = f32_downcast_to_bf16(f32_arr)
        return raw, f32_rounded.astype(np.float64)
    raw = f32_arr.astype(dtype)
    return raw, raw.astype(np.float64)


available_capabilities: dict[str, str] = nk.get_capabilities()

# fmt: off
possible_x86_capabilities: list[str] = [
    "haswell", "alder", "sierra",
    "skylake", "icelake", "genoa", "sapphire", "turin",
    "sapphireamx", "graniteamx",
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
machine_architecture = platform.machine()

if sys.platform == "linux":
    if machine_architecture == "x86_64":
        hardware_capabilities = possible_x86_capabilities
    elif machine_architecture == "aarch64":
        hardware_capabilities = possible_arm_capabilities
    elif machine_architecture == "riscv64":
        hardware_capabilities = possible_rvv_capabilities
elif sys.platform == "darwin":
    if machine_architecture == "x86_64":
        hardware_capabilities = possible_x86_capabilities
    elif machine_architecture == "arm64":
        hardware_capabilities = possible_arm_capabilities
elif sys.platform == "win32":
    if machine_architecture == "AMD64":
        hardware_capabilities = possible_x86_capabilities
    elif machine_architecture == "ARM64":
        hardware_capabilities = possible_arm_capabilities
elif sys.platform.startswith("freebsd"):
    if machine_architecture == "amd64":
        hardware_capabilities = possible_x86_capabilities
    elif machine_architecture == "arm64":
        hardware_capabilities = possible_arm_capabilities
elif sys.platform in ("emscripten", "wasi"):
    hardware_capabilities = possible_wasm_capabilities

possible_capabilities: list[str] = ["serial"] + hardware_capabilities

current_capability: str | None = None


def keep_one_capability(cap: str):
    global current_capability
    assert cap in possible_capabilities, f"Capability {cap} is not available on this platform."
    if cap == current_capability:
        return
    for c in possible_capabilities:
        if c != cap and c != "serial":
            nk.disable_capability(c)
    if cap != "serial":
        nk.enable_capability(cap)
    current_capability = cap


def create_stats():
    """Create a fresh stats dict for error collection."""
    return {
        "metric": [],
        "ndim": [],
        "dtype": [],
        "capability": [],
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
    eps = np.finfo(accurate_result.dtype).resolution if np.issubdtype(accurate_result.dtype, np.inexact) else 1.0
    absolute_baseline_error = np.max(np.abs(baseline_result - accurate_result))
    relative_baseline_error = np.max(np.abs(baseline_result - accurate_result) / (np.abs(accurate_result) + eps))
    absolute_nk_error = np.max(np.abs(nk_result - accurate_result))
    relative_nk_error = np.max(np.abs(nk_result - accurate_result) / (np.abs(accurate_result) + eps))

    stats["metric"].append(metric)
    stats["ndim"].append(ndim)
    stats["dtype"].append(dtype)
    stats["capability"].append(current_capability or "unknown")
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


def format_scientific(value):
    """Format a float as compact scientific notation (e.g. 7.4e-5). Return '0' for exact zero."""
    if value == 0:
        return "0"
    s = f"{value:.1e}"
    if "e" not in s:
        return s  # inf, nan, etc.
    mantissa, exp = s.split("e")
    exp_sign = exp[0]
    exp_digits = exp[1:].lstrip("0") or "0"
    return f"{mantissa}e{exp_sign}{exp_digits}"


def pad_with_ansi_color(visible, width, code):
    """Pad visible string to width first, then wrap in ANSI so escape codes don't break alignment."""
    padded = f"{visible:<{width}}"
    return f"\033[{code}m{padded}\033[0m"


CapabilityRecord = collections.namedtuple(
    "CapabilityRecord",
    ["metric", "ndim", "dtype", "capability", "baseline_error_mean", "nk_error_mean", "speedup_mean"],
)


def print_stats_report(stats):
    """Print a condensed error/speedup report: two rows per (metric, dtype) showing min/max ndim."""
    if not stats["metric"]:
        return

    # Stage 1: Group raw stats by (metric, ndim, dtype, capability) and compute per-group means.
    grouped = collections.defaultdict(
        lambda: {"relative_baseline_error": [], "relative_nk_error": [], "baseline_duration": [], "nk_duration": []}
    )
    for metric, ndim, dtype, capability, rel_base, rel_nk, base_dur, nk_dur in zip(
        stats["metric"],
        stats["ndim"],
        stats["dtype"],
        stats["capability"],
        stats["relative_baseline_error"],
        stats["relative_nk_error"],
        stats["baseline_duration"],
        stats["nk_duration"],
    ):
        key = (metric, ndim, dtype, capability)
        grouped[key]["relative_baseline_error"].append(rel_base)
        grouped[key]["relative_nk_error"].append(rel_nk)
        grouped[key]["baseline_duration"].append(base_dur)
        grouped[key]["nk_duration"].append(nk_dur)

    cap_records = []
    for (metric, ndim, dtype, capability), vals in grouped.items():
        n = len(vals["nk_duration"])
        base_err_mean = sum(vals["relative_baseline_error"]) / n
        nk_err_mean = sum(vals["relative_nk_error"]) / n
        speedups = [b / nk for b, nk in zip(vals["baseline_duration"], vals["nk_duration"]) if nk > 0]
        speedup_mean = sum(speedups) / len(speedups) if speedups else 0.0
        cap_records.append(CapabilityRecord(metric, ndim, dtype, capability, base_err_mean, nk_err_mean, speedup_mean))

    # Stage 2: Re-aggregate by (metric, dtype). For each, find min/max ndim and cross-capability aggregates.
    by_metric_dtype = collections.defaultdict(list)
    for rec in cap_records:
        by_metric_dtype[(rec.metric, rec.dtype)].append(rec)

    # Each output row: (metric_str, dtype_str, ndim_str, base_err_str, worst_nk_err_str, best_speedup_str)
    rows = []
    for (metric, dtype), recs in sorted(by_metric_dtype.items()):
        all_ndims = sorted(set(r.ndim for r in recs))
        min_ndim, max_ndim = all_ndims[0], all_ndims[-1]
        single_ndim = min_ndim == max_ndim
        target_ndims = [min_ndim] if single_ndim else [min_ndim, max_ndim]

        for i, target_ndim in enumerate(target_ndims):
            subset = [r for r in recs if r.ndim == target_ndim]
            if not subset:
                continue

            # Base error: average across capabilities
            base_err = sum(r.baseline_error_mean for r in subset) / len(subset)
            # Worst NK error: capability with highest mean NK error
            worst_rec = max(subset, key=lambda r: r.nk_error_mean)
            worst_nk_err = worst_rec.nk_error_mean
            worst_cap = worst_rec.capability
            # Best speedup: capability with highest mean speedup
            best_rec = max(subset, key=lambda r: r.speedup_mean)
            best_speedup = best_rec.speedup_mean
            best_cap = best_rec.capability
            best_cap_err = best_rec.nk_error_mean

            # Format strings
            if i == 0:
                metric_str = metric
                dtype_str = dtype
            else:
                metric_str = ""
                dtype_str = ""

            if single_ndim:
                ndim_str = str(target_ndim)
            elif i == 0:
                ndim_str = f"\u230a{target_ndim:>4}\u230b"
            else:
                ndim_str = f"\u2308{target_ndim:>4}\u2309"

            rows.append(
                (
                    metric_str,
                    dtype_str,
                    ndim_str,
                    base_err,
                    worst_nk_err,
                    worst_cap,
                    best_speedup,
                    best_cap,
                    best_cap_err,
                )
            )

    # Stage 3: Render
    col_w = {"kernel": 17, "dtype": 12, "ndim": 8, "base_err": 14, "worst_nk": 30, "best_spd": 34}
    header = (
        f"{'Kernel':<{col_w['kernel']}}"
        f"{'DType':<{col_w['dtype']}}"
        f"{'NDim':<{col_w['ndim']}}"
        f"{'Base Error':<{col_w['base_err']}}"
        f"{'Worst NK Error':<{col_w['worst_nk']}}"
        f"{'Best NK Speedup':<{col_w['best_spd']}}"
    )
    sep = "\u2500" * len(header)
    print(f"\n\n{header}")
    print(sep)

    for (
        metric_str,
        dtype_str,
        ndim_str,
        base_err,
        worst_nk_err,
        worst_cap,
        best_speedup,
        best_cap,
        best_cap_err,
    ) in rows:
        base_err_s = format_scientific(base_err)
        worst_nk_s = f"{format_scientific(worst_nk_err)} \u2039{worst_cap}\u203a"
        best_spd_s = f"{best_speedup:.1f}x \u2039{best_cap}, err {format_scientific(best_cap_err)}\u203a"

        # Color for worst NK error: red if NK error > base error
        nk_err_code = "31" if worst_nk_err > base_err else "0"
        # Color for speedup: green >=2x, yellow 1-2x, red <1x
        if best_speedup >= 2.0:
            spd_code = "32"
        elif best_speedup >= 1.0:
            spd_code = "33"
        else:
            spd_code = "31"

        line = (
            f"{metric_str:<{col_w['kernel']}}"
            f"{dtype_str:<{col_w['dtype']}}"
            f"{ndim_str:<{col_w['ndim']}}"
            f"{base_err_s:<{col_w['base_err']}}"
            f"{pad_with_ansi_color(worst_nk_s, col_w['worst_nk'], nk_err_code)}"
            f"{pad_with_ansi_color(best_spd_s, col_w['best_spd'], spd_code)}"
        )
        print(line)

    warnings_list = stats.get("warnings", [])
    warnings_list = sorted(warnings_list)
    warnings_list = [f"{name}: {message}" for name, message in warnings_list]
    if len(warnings_list) != 0:
        print("\nWarnings:")
        warning_counts = collections.Counter(warnings_list)
        for warning, count in sorted(warning_counts.items()):
            print(f"- {count}x times: {warning}")


@pytest.fixture(autouse=True)
def seed_rng(__pytest_repeat_step_number):
    """Auto-seed NumPy RNG before every test. When NK_SEED is set, each
    @pytest.mark.repeat() step gets a unique derived seed."""
    if not numpy_available:
        return
    step = __pytest_repeat_step_number or 0
    if nk_seed is not None:
        np.random.seed(nk_seed + step)


# Map nk dtype → (array.array typecode, low, high)
ARRAY_TYPECODES = {
    "float32": ("f", -10.0, 10.0),
    "float64": ("d", -10.0, 10.0),
    "int8": ("b", -128, 127),
    "uint8": ("B", 0, 255),
}


def make_random_buffer(n, dtype="float32"):
    """Create a random array.array for the given dtype — no numpy needed."""
    tc, lo, hi = ARRAY_TYPECODES[dtype]
    if tc in ("f", "d"):
        return array.array(tc, [random.uniform(lo, hi) for _ in range(n)])
    else:
        return array.array(tc, [random.randint(int(lo), int(hi)) for _ in range(n)])


def make_positive_buffer(n, dtype="float32"):
    """Create a random positive array.array — for probability distributions."""
    tc = "f" if dtype == "float32" else "d"
    vals = [random.uniform(0.01, 1.0) for _ in range(n)]
    total = sum(vals)
    return array.array(tc, [v / total for v in vals])
