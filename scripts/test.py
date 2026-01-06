#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: test.py

This module contains a suite of tests for the `numkong` package.
It compares various SIMD kernels (like Dot-products, squared Euclidean, and Cosine distances)
with their NumPy or baseline counterparts, testing accuracy for different data types including
floating-point, integer, and complex numbers.

The tests cover:

- **Dense Vector Operations**: Tests for `float64`, `float32`, `float16` data types using metrics like `inner`, `sqeuclidean`, and `angular`.
- **Brain Floating-Point Format (bfloat16)**: Tests for operations with the brain floating-point format not natively supported by NumPy.
- **Integer Operations**: Tests for `int8` data type, ensuring accuracy without overflow.
- **Bitwise Operations**: Tests for Hamming and Jaccard distances using bit arrays.
- **Complex Numbers**: Tests for complex dot products and vector dot products.
- **Batch Operations and Cross-Distance Computations**: Tests for batch processing and cross-distance computations using `cdist`.
- **Hardware Capabilities Verification**: Checks the availability of hardware capabilities and function pointers.

**Dependencies**:

- Python 3.x
- `numpy`
- `scipy`
- `pytest`
- `tabulate`
- `numkong` package

**Usage**:

Run the tests using pytest:

    pytest test.py

Or run the script directly:

    python test.py

"""
import os
import sys
import math
import time
import platform
import collections
import warnings
from typing import Dict, List
import faulthandler

import tabulate
import pytest
import numkong as nk

faulthandler.enable()
randomized_repetitions_count: int = 10

# NumPy is available on most platforms and is required for most tests.
# When using PyPy on some platforms NumPy has internal issues, that will
# raise a weird error, not an `ImportError`. That's why we intentionally
# use a naked `except:`. Necessary evil!
try:
    import numpy as np

    numpy_available = True

    baseline_inner = np.inner
    baseline_intersect = lambda x, y: len(np.intersect1d(x, y))
    baseline_bilinear = lambda x, y, z: x @ z @ y

    def _normalize_element_wise(r, dtype_new):
        """Clips higher-resolution results to the smaller target dtype without overflow."""
        if np.issubdtype(dtype_new, np.integer):
            r = np.round(r)
        #! We need non-overflowing saturating addition for small integers, that NumPy lacks:
        #! https://stackoverflow.com/questions/29611185/avoid-overflow-when-adding-numpy-arrays
        if np.issubdtype(dtype_new, np.integer):
            dtype_old_info = np.iinfo(r.dtype) if np.issubdtype(r.dtype, np.integer) else np.finfo(r.dtype)
            dtype_new_info = np.iinfo(dtype_new)
            new_min = dtype_new_info.min if dtype_new_info.min > dtype_old_info.min else None
            new_max = dtype_new_info.max if dtype_new_info.max < dtype_old_info.max else None
            if new_min is not None or new_max is not None:
                r = np.clip(r, new_min, new_max, out=r)
        return r.astype(dtype_new)

    def _computation_dtype(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        larger_dtype = np.promote_types(x.dtype, y.dtype)
        if larger_dtype == np.uint8:
            return np.uint16, larger_dtype
        elif larger_dtype == np.int8:
            return np.int16, larger_dtype
        if larger_dtype == np.uint16:
            return np.uint32, larger_dtype
        elif larger_dtype == np.int16:
            return np.int32, larger_dtype
        if larger_dtype == np.uint32:
            return np.uint64, larger_dtype
        elif larger_dtype == np.int32:
            return np.int64, larger_dtype
        else:
            return larger_dtype, larger_dtype

    def baseline_scale(x, alpha, beta):
        compute_dtype, _ = _computation_dtype(x, alpha)
        result = alpha * x.astype(compute_dtype) + beta
        return _normalize_element_wise(result, x.dtype)

    def baseline_sum(x, y):
        compute_dtype, _ = _computation_dtype(x, y)
        result = x.astype(compute_dtype) + y.astype(compute_dtype)
        return _normalize_element_wise(result, x.dtype)

    def baseline_wsum(x, y, alpha, beta):
        compute_dtype, _ = _computation_dtype(x, y)
        result = x.astype(compute_dtype) * alpha + y.astype(compute_dtype) * beta
        return _normalize_element_wise(result, x.dtype)

    def baseline_fma(x, y, z, alpha, beta):
        compute_dtype, _ = _computation_dtype(x, y)
        result = x.astype(compute_dtype) * y.astype(compute_dtype) * alpha + z.astype(compute_dtype) * beta
        return _normalize_element_wise(result, x.dtype)

    def baseline_add(x, y, out=None):
        compute_dtype, final_dtype = _computation_dtype(x, y)
        a = x.astype(compute_dtype) if isinstance(x, np.ndarray) else x
        b = y.astype(compute_dtype) if isinstance(y, np.ndarray) else y
        # If the input types are identical, we want to perform addition with saturation
        result = np.add(a, b, out=out, casting="unsafe")
        result = _normalize_element_wise(result, final_dtype)
        return result

    def baseline_multiply(x, y, out=None):
        compute_dtype, final_dtype = _computation_dtype(x, y)
        a = x.astype(compute_dtype) if isinstance(x, np.ndarray) else x
        b = y.astype(compute_dtype) if isinstance(y, np.ndarray) else y
        # If the input types are identical, we want to perform addition with saturation
        result = np.multiply(a, b, out=out, casting="unsafe")
        result = _normalize_element_wise(result, final_dtype)
        return result

except:
    # NumPy is not installed, most tests will be skipped
    numpy_available = False

    baseline_inner = lambda x, y: sum(x[i] * y[i] for i in range(len(x)))
    baseline_intersect = lambda x, y: len(set(x).intersection(y))

    def baseline_bilinear(x, y, z):
        result = 0
        for i in range(len(x)):
            for j in range(len(y)):
                result += x[i] * z[i][j] * y[j]
        return result

    def baseline_scale(x, alpha, beta):
        return [alpha * xi + beta for xi in x]

    def baseline_sum(x, y):
        return [xi + yi for xi, yi in zip(x, y)]

    def baseline_fma(x, y, z, alpha, beta):
        return [(alpha * xi) * yi + beta * zi for xi, yi, zi in zip(x, y, z)]

    def baseline_wsum(x, y, alpha, beta):
        return [(alpha * xi) + beta * yi for xi, yi in zip(x, y)]

    def baseline_add(x, y, out=None):
        result = [xi + yi for xi, yi in zip(x, y)]
        if out is not None:
            out[:] = result
        else:
            return out

    def baseline_multiply(x, y, out=None):
        result = [xi * yi for xi, yi in zip(x, y)]
        if out is not None:
            out[:] = result
        else:
            return out


# At the time of Python 3.12, SciPy doesn't support 32-bit Windows on any CPU,
# or 64-bit Windows on Arm. It also doesn't support `musllinux` distributions,
# like CentOS, RedHat OS, and many others.
try:
    import scipy.spatial.distance as spd

    scipy_available = True

    baseline_euclidean = lambda x, y: np.array(spd.euclidean(x, y))  #! SciPy returns a scalar
    baseline_sqeuclidean = spd.sqeuclidean
    baseline_angular = spd.cosine
    baseline_jensenshannon = lambda x, y: spd.jensenshannon(x, y)
    baseline_hamming = lambda x, y: spd.hamming(x, y) * len(x)
    baseline_jaccard = spd.jaccard

    def baseline_mahalanobis(x, y, z):
        # If there was an error, or the value is NaN, we skip the test.
        try:
            result = spd.mahalanobis(x, y, z).astype(np.float64)
            if not np.isnan(result):
                return result
        except:
            pass
        pytest.skip(f"SciPy Mahalanobis distance returned {result} due to `sqrt` of a negative number")

except:
    # SciPy is not installed, some tests will be skipped
    scipy_available = False

    baseline_angular = lambda x, y: 1.0 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    baseline_euclidean = lambda x, y: np.array([np.sqrt(np.sum((x - y) ** 2))])
    baseline_sqeuclidean = lambda x, y: np.sum((x - y) ** 2)
    baseline_jensenshannon = lambda p, q: (np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / 2
    baseline_hamming = lambda x, y: np.logical_xor(x, y).sum()

    def baseline_mahalanobis(x, y, z):
        diff = x - y
        return np.sqrt(diff @ z @ diff)

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


# ml_dtypes provides reference implementations of bfloat16 and float8 types.
# Used to validate our custom type implementations against Google's reference.
try:
    import ml_dtypes

    ml_dtypes_available = True
except:
    ml_dtypes_available = False


def is_running_under_qemu():
    return "NK_IN_QEMU" in os.environ


# Geospatial baseline functions (always defined, use NumPy only)
earth_radius_meters = 6335439.0  # Mean Earth radius in meters


def baseline_haversine(first_latitude, first_longitude, second_latitude, second_longitude):
    """Haversine distance using NumPy. All inputs in radians, output in meters."""
    latitude_difference = second_latitude - first_latitude
    longitude_difference = second_longitude - first_longitude
    haversine_term = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(first_latitude) * np.cos(second_latitude) * np.sin(longitude_difference / 2) ** 2
    )
    central_angle = 2 * np.arctan2(np.sqrt(haversine_term), np.sqrt(1 - haversine_term))
    return earth_radius_meters * central_angle


def baseline_vincenty(
    first_latitude, first_longitude, second_latitude, second_longitude, max_iterations=100, tolerance=1e-12
):
    """Vincenty distance using NumPy. All inputs in radians, output in meters."""
    # WGS84 ellipsoid parameters
    equatorial_radius = 6378136.6
    polar_radius = 6356751.9
    flattening = (equatorial_radius - polar_radius) / equatorial_radius

    # Reduced latitudes (parametric latitudes on auxiliary sphere)
    reduced_latitude_first = np.arctan((1 - flattening) * np.tan(first_latitude))
    reduced_latitude_second = np.arctan((1 - flattening) * np.tan(second_latitude))
    longitude_difference = second_longitude - first_longitude

    sin_reduced_first = np.sin(reduced_latitude_first)
    cos_reduced_first = np.cos(reduced_latitude_first)
    sin_reduced_second = np.sin(reduced_latitude_second)
    cos_reduced_second = np.cos(reduced_latitude_second)

    lambda_current = longitude_difference
    for _ in range(max_iterations):
        sin_lambda = np.sin(lambda_current)
        cos_lambda = np.cos(lambda_current)

        sin_sigma = np.sqrt(
            (cos_reduced_second * sin_lambda) ** 2
            + (cos_reduced_first * sin_reduced_second - sin_reduced_first * cos_reduced_second * cos_lambda) ** 2
        )
        cos_sigma = sin_reduced_first * sin_reduced_second + cos_reduced_first * cos_reduced_second * cos_lambda
        sigma = np.arctan2(sin_sigma, cos_sigma)

        sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda / sin_sigma
        cos_squared_azimuth = 1 - sin_azimuth**2
        cos_two_sigma_midpoint = (
            cos_sigma - 2 * sin_reduced_first * sin_reduced_second / cos_squared_azimuth
            if cos_squared_azimuth != 0
            else 0
        )

        correction_term = flattening / 16 * cos_squared_azimuth * (4 + flattening * (4 - 3 * cos_squared_azimuth))
        lambda_next = longitude_difference + (1 - correction_term) * flattening * sin_azimuth * (
            sigma
            + correction_term
            * sin_sigma
            * (cos_two_sigma_midpoint + correction_term * cos_sigma * (-1 + 2 * cos_two_sigma_midpoint**2))
        )

        if np.abs(lambda_next - lambda_current) < tolerance:
            break
        lambda_current = lambda_next

    u_squared = cos_squared_azimuth * (equatorial_radius**2 - polar_radius**2) / polar_radius**2
    coefficient_a = 1 + u_squared / 16384 * (4096 + u_squared * (-768 + u_squared * (320 - 175 * u_squared)))
    coefficient_b = u_squared / 1024 * (256 + u_squared * (-128 + u_squared * (74 - 47 * u_squared)))
    delta_sigma = (
        coefficient_b
        * sin_sigma
        * (
            cos_two_sigma_midpoint
            + coefficient_b
            / 4
            * (
                cos_sigma * (-1 + 2 * cos_two_sigma_midpoint**2)
                - coefficient_b
                / 6
                * cos_two_sigma_midpoint
                * (-3 + 4 * sin_sigma**2)
                * (-3 + 4 * cos_two_sigma_midpoint**2)
            )
        )
    )

    return polar_radius * coefficient_a * (sigma - delta_sigma)


def scipy_metric_name(metric: str) -> str:
    """Convert NumKong metric names to SciPy equivalents."""
    # NumKong uses 'angular' while SciPy uses 'cosine' for the same metric
    if metric == "angular":
        return "cosine"
    return metric


def profile(callable, *args, **kwargs) -> tuple:
    before = time.perf_counter_ns()
    result = callable(*args, **kwargs)
    after = time.perf_counter_ns()
    return after - before, result


@pytest.fixture(scope="session")
def stats_fixture():
    """Session-scoped fixture that collects errors during tests."""
    results = dict()
    results["metric"] = []
    results["ndim"] = []
    results["dtype"] = []
    results["absolute_baseline_error"] = []
    results["relative_baseline_error"] = []
    results["absolute_nk_error"] = []
    results["relative_nk_error"] = []
    results["accurate_duration"] = []
    results["baseline_duration"] = []
    results["nk_duration"] = []
    results["warnings"] = []
    yield results

    # Group the errors by (metric, ndim, dtype) to calculate the mean and std error.
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
        results["metric"],
        results["ndim"],
        results["dtype"],
        results["absolute_baseline_error"],
        results["relative_baseline_error"],
        results["absolute_nk_error"],
        results["relative_nk_error"],
        results["accurate_duration"],
        results["baseline_duration"],
        results["nk_duration"],
    ):
        key = (metric, ndim, dtype)
        grouped_errors[key]["absolute_baseline_error"].append(absolute_baseline_error)
        grouped_errors[key]["relative_baseline_error"].append(relative_baseline_error)
        grouped_errors[key]["absolute_nk_error"].append(absolute_nk_error)
        grouped_errors[key]["relative_nk_error"].append(relative_nk_error)
        grouped_errors[key]["accurate_duration"].append(accurate_duration)
        grouped_errors[key]["baseline_duration"].append(baseline_duration)
        grouped_errors[key]["nk_duration"].append(nk_duration)

    # Compute mean and the standard deviation for each task error
    final_results = []
    for key, errors in grouped_errors.items():
        n = len(errors["nk_duration"])

        # Mean and the standard deviation for errors
        baseline_errors = errors["relative_baseline_error"]
        nk_errors = errors["relative_nk_error"]
        #! On some platforms (like `cp312-musllinux_aarch64`) without casting via `float(x)`
        #! the subsequent `:.2e` string formatting code will fail due to:
        #! `TypeError: unsupported format string passed to numpy.ndarray.__format__`.
        baseline_mean = float(sum(baseline_errors)) / n
        nk_mean = float(sum(nk_errors)) / n
        baseline_std = math.sqrt(sum((x - baseline_mean) ** 2 for x in baseline_errors) / n)
        nk_std = math.sqrt(sum((x - nk_mean) ** 2 for x in nk_errors) / n)
        baseline_error_formatted = f"{baseline_mean:.2e} ± {baseline_std:.2e}"
        nk_error_formatted = f"{nk_mean:.2e} ± {nk_std:.2e}"

        # Log durations
        accurate_durations = errors["accurate_duration"]
        baseline_durations = errors["baseline_duration"]
        nk_durations = errors["nk_duration"]
        accurate_mean_duration = sum(accurate_durations) / n
        baseline_mean_duration = sum(baseline_durations) / n
        nk_mean_duration = sum(nk_durations) / n
        accurate_std_duration = math.sqrt(sum((x - accurate_mean_duration) ** 2 for x in accurate_durations) / n)
        baseline_std_duration = math.sqrt(sum((x - baseline_mean_duration) ** 2 for x in baseline_durations) / n)
        nk_std_duration = math.sqrt(sum((x - nk_mean_duration) ** 2 for x in nk_durations) / n)
        accurate_duration = f"{accurate_mean_duration:.2e} ± {accurate_std_duration:.2e}"
        baseline_duration = f"{baseline_mean_duration:.2e} ± {baseline_std_duration:.2e}"
        nk_duration = f"{nk_mean_duration:.2e} ± {nk_std_duration:.2e}"

        # Measure time improvement
        improvements = [baseline / numkong for baseline, numkong in zip(baseline_durations, nk_durations)]
        improvements_mean = sum(improvements) / n
        improvements_std = math.sqrt(sum((x - improvements_mean) ** 2 for x in improvements) / n)
        nk_speedup = f"{improvements_mean:.2f}x ± {improvements_std:.2f}x"

        # Calculate Improvement
        # improvement = abs(baseline_mean - nk_mean) / min(nk_mean, baseline_mean)
        # if baseline_mean < nk_mean:
        #     improvement *= -1
        # improvement_formatted = f"{improvement:+.2}x" if improvement != float("inf") else "N/A"

        final_results.append(
            (
                *key,
                baseline_error_formatted,
                nk_error_formatted,
                accurate_duration,
                baseline_duration,
                nk_duration,
                nk_speedup,
            )
        )

    # Sort results for consistent presentation
    final_results.sort(key=lambda x: (x[0], x[1], x[2]))

    # Output the final table after all tests are completed
    print("\n")
    print("Numerical Error Aggregation Report:")
    headers = [
        "Metric",
        "NDim",
        "DType",
        "Baseline Error",  # Printed as mean ± std deviation
        "NumKong Error",  # Printed as mean ± std deviation
        "Accurate Duration",  # Printed as mean ± std deviation
        "Baseline Duration",  # Printed as mean ± std deviation
        "NumKong Duration",  # Printed as mean ± std deviation
        "NumKong Speedup",
    ]
    print(tabulate.tabulate(final_results, headers=headers, tablefmt="pretty", showindex=True))

    # Show the additional grouped warnings
    warnings = results.get("warnings", [])
    warnings = sorted(warnings)
    warnings = [f"{name}: {message}" for name, message in warnings]
    if len(warnings) != 0:
        print("\nWarnings:")
        unique_warnings, warning_counts = np.unique(warnings, return_counts=True)
        for warning, count in zip(unique_warnings, warning_counts):
            print(f"- {count}x times: {warning}")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Custom hook to ensure that the error aggregator runs even for failed tests."""
    if call.when == "call":
        item.test_result = call.excinfo is None


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
    """Calculates and aggregates errors for a given test.

    What we want to know in the end of the day is:

    -   How much NumKong implementation is more/less accurate than baseline,
        when compared against the accurate result?
    -   TODO: How much faster is NumKong than the baseline kernel?
    -   TODO: How much faster is NumKong than the accurate kernel?
    """
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


def get_current_test():
    """Get's the current test filename, test name, and function name.
    Similar metadata can be obtained from the `request` fixture, but this
    solution uses environment variables."""
    full_name = os.environ.get("PYTEST_CURRENT_TEST").split(" ")[0]
    test_file = full_name.split("::")[0].split("/")[-1].split(".py")[0]
    test_name = full_name.split("::")[1]
    # The `test_name` may look like: "test_dense_i8[angular-1536-24-50]"
    function_name = test_name.split("[")[0]
    return test_file, test_name, function_name


def collect_warnings(message: str, stats: dict):
    """Collects warnings for the final report."""
    _, _, function_name = get_current_test()
    stats["warnings"].append((function_name, message))


# For normalized distances we use the absolute tolerance, because the result is close to zero.
# For unnormalized ones (like squared Euclidean or Jaccard), we use the relative.
NK_RTOL = 0.1
NK_ATOL = 0.1

# We will run all the tests many times using different instruction sets under the hood.
available_capabilities: Dict[str, str] = nk.get_capabilities()
possible_x86_capabilities: List[str] = ["haswell", "ice", "skylake", "sapphire", "turin", "genoa", "sierra"]
possible_arm_capabilities: List[str] = [
    "neon",
    "neonhalf",
    "neonbfdot",
    "neonsdot",
    "sve",
    "svehalf",
    "svebfdot",
    "svesdot",
]
possible_x86_capabilities: List[str] = [c for c in possible_x86_capabilities if available_capabilities[c]]
possible_arm_capabilities: List[str] = [c for c in possible_arm_capabilities if available_capabilities[c]]
possible_capabilities: List[str] = []

if sys.platform == "linux":
    if platform.machine() == "x86_64":
        possible_capabilities = possible_x86_capabilities
    elif platform.machine() == "aarch64":
        possible_capabilities = possible_arm_capabilities
elif sys.platform == "darwin":
    if platform.machine() == "x86_64":
        possible_capabilities = possible_x86_capabilities
    elif platform.machine() == "arm64":
        possible_capabilities = possible_arm_capabilities
elif sys.platform == "win32":
    if platform.machine() == "AMD64":
        possible_capabilities = possible_x86_capabilities
    elif platform.machine() == "ARM64":
        possible_capabilities = possible_arm_capabilities


def keep_one_capability(cap: str):
    assert cap in possible_capabilities or cap == "serial", f"Capability {cap} is not available on this platform."
    for c in possible_capabilities:
        if c != cap:
            nk.disable_capability(c)
    # Serial is always enabled, can't toggle it
    if cap != "serial":
        nk.enable_capability(cap)


def name_to_kernels(name: str):
    """
    Having a separate "helper" function to convert the kernel name is handy for PyTest decorators,
    that can't generally print non-trivial object (like function pointers) well.
    """
    if name == "inner":
        return baseline_inner, nk.inner
    elif name == "euclidean":
        return baseline_euclidean, nk.euclidean
    elif name == "sqeuclidean":
        return baseline_sqeuclidean, nk.sqeuclidean
    elif name == "angular":
        return baseline_angular, nk.angular
    elif name == "bilinear":
        return baseline_bilinear, nk.bilinear
    elif name == "mahalanobis":
        return baseline_mahalanobis, nk.mahalanobis
    elif name == "jaccard":
        return baseline_jaccard, nk.jaccard
    elif name == "hamming":
        return baseline_hamming, nk.hamming
    elif name == "intersect":
        return baseline_intersect, nk.intersect
    elif name == "scale":
        return baseline_scale, nk.scale
    elif name == "wsum":
        return baseline_wsum, nk.wsum
    elif name == "fma":
        return baseline_fma, nk.fma
    elif name == "add":
        return baseline_add, nk.add
    elif name == "multiply":
        return baseline_multiply, nk.multiply
    elif name == "jensenshannon":
        return baseline_jensenshannon, nk.jensenshannon
    elif name == "haversine":
        return baseline_haversine, nk.haversine
    elif name == "vincenty":
        return baseline_vincenty, nk.vincenty
    else:
        raise ValueError(f"Unknown kernel name: {name}")


def f32_downcast_to_bf16(array):
    """Converts an array of 32-bit floats into 16-bit brain-floats."""
    array = np.asarray(array, dtype=np.float32)
    # NumPy doesn't natively support brain-float, so we need a trick!
    # Luckily, it's very easy to reduce the representation accuracy
    # by simply masking the low 16-bits of our 32-bit single-precision
    # numbers. We can also add `0x8000` to round the numbers.
    array_f32_rounded = ((array.view(np.uint32) + 0x8000) & 0xFFFF0000).view(np.float32)
    # To represent them as brain-floats, we need to drop the second halves.
    array_bf16 = np.right_shift(array_f32_rounded.view(np.uint32), 16).astype(np.uint16)
    return array_f32_rounded, array_bf16


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
def test_bf16_conversion_vs_ml_dtypes(ndim):
    """Compare NumKong bfloat16 conversion with ml_dtypes reference implementation."""
    a_f32 = np.random.randn(ndim).astype(np.float32)

    # NumKong approach (using our bit manipulation helper)
    _, a_nk_bf16 = f32_downcast_to_bf16(a_f32)

    # ml_dtypes reference implementation
    a_ml_bf16 = a_f32.astype(ml_dtypes.bfloat16)

    # Compare raw bit patterns - they should match exactly
    assert np.array_equal(a_nk_bf16, a_ml_bf16.view(np.uint16)), (
        f"BFloat16 conversion mismatch with ml_dtypes:\n"
        f"  NumKong bits: {a_nk_bf16[:5]}...\n"
        f"  ml_dtypes bits: {a_ml_bf16.view(np.uint16)[:5]}..."
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
def test_float8_e4m3_conversion_vs_ml_dtypes(ndim):
    """Compare NumKong float8_e4m3 conversion with ml_dtypes reference implementation."""
    # Use values in a reasonable range for e4m3 (max ~448)
    a_f32 = (np.random.randn(ndim) * 10).astype(np.float32)
    a_f32 = np.clip(a_f32, -448, 448)

    # ml_dtypes reference implementation
    a_ml_e4m3 = a_f32.astype(ml_dtypes.float8_e4m3fn)

    # NumKong conversion via NDArray
    a_nk = nk.zeros((ndim,), dtype="e4m3")
    # TODO: Once we have NumPy dtype registration, we can directly compare
    # For now, we just verify ml_dtypes works and our types exist
    assert a_ml_e4m3.dtype == ml_dtypes.float8_e4m3fn
    assert a_nk.dtype == "e4m3"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
def test_float8_e5m2_conversion_vs_ml_dtypes(ndim):
    """Compare NumKong float8_e5m2 conversion with ml_dtypes reference implementation."""
    # Use values in a reasonable range for e5m2 (max ~57344)
    a_f32 = (np.random.randn(ndim) * 100).astype(np.float32)
    a_f32 = np.clip(a_f32, -57344, 57344)

    # ml_dtypes reference implementation
    a_ml_e5m2 = a_f32.astype(ml_dtypes.float8_e5m2)

    # NumKong conversion via NDArray
    a_nk = nk.zeros((ndim,), dtype="e5m2")
    # TODO: Once we have NumPy dtype registration, we can directly compare
    # For now, we just verify ml_dtypes works and our types exist
    assert a_ml_e5m2.dtype == ml_dtypes.float8_e5m2
    assert a_nk.dtype == "e5m2"


def i8_downcast_to_i4(array):
    """Converts an array of 8-bit integers into 4-bit integers, packing 2 per byte."""
    array = np.asarray(array, dtype=np.int8)
    assert np.all(array >= -8) and np.all(array <= 7), "Input array must be in the range [-8, 7]"


def hex_array(arr):
    """Converts numerical array into a string of comma-separated hexadecimal values for debugging.
    Supports 1D and 2D arrays.
    """
    printer = np.vectorize(hex)
    strings = printer(arr)

    if strings.ndim == 1:
        return ", ".join(strings)
    else:
        return "\n".join(", ".join(row) for row in strings)


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
    assert "ice" in nk.get_capabilities()
    assert "skylake" in nk.get_capabilities()
    assert "genoa" in nk.get_capabilities()
    assert "sapphire" in nk.get_capabilities()
    assert "turin" in nk.get_capabilities()
    assert nk.get_capabilities().get("serial") == 1

    # Check the toggle:
    previous_value = nk.get_capabilities().get("neon")
    nk.enable_capability("neon")
    assert nk.get_capabilities().get("neon") == 1
    if not previous_value:
        nk.disable_capability("neon")


def to_array(x, dtype=None):
    if numpy_available:
        y = np.array(x)
        if dtype is not None:
            y = y.astype(dtype)
        return y


def random_of_dtype(dtype, shape):
    if dtype == "float64" or dtype == "float32" or dtype == "float16":
        return np.random.randn(*shape).astype(dtype)
    elif (
        dtype == "int8"
        or dtype == "uint8"
        or dtype == "int16"
        or dtype == "uint16"
        or dtype == "int32"
        or dtype == "uint32"
        or dtype == "int64"
        or dtype == "uint64"
    ):
        return np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max, shape, dtype=dtype)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "function, expected_error, args, kwargs",
    [
        # Test missing positional arguments
        (nk.sqeuclidean, TypeError, (), {}),  # No arguments provided
        (nk.sqeuclidean, TypeError, (to_array([1.0]),), {}),  # Only one positional argument
        # Try missing type name
        (nk.sqeuclidean, ValueError, (to_array([1.0]), to_array([1.0]), "missing_dtype"), {}),
        # Test incorrect argument type
        (nk.sqeuclidean, TypeError, (to_array([1.0]), "invalid"), {}),  # Wrong type for second argument
        # Test invalid keyword argument name
        (nk.sqeuclidean, TypeError, (to_array([1.0]), to_array([1.0])), {"invalid_kwarg": "value"}),
        # Test wrong argument type for SIMD capability toggle
        (nk.enable_capability, TypeError, (123,), {}),  # Should expect a string
        (nk.disable_capability, TypeError, ([],), {}),  # Should expect a string
        # Test missing required argument for Mahalanobis
        (nk.mahalanobis, TypeError, (to_array([1.0]), to_array([1.0])), {}),  # Missing covariance matrix
        # Test missing required arguments for bilinear
        (nk.bilinear, TypeError, (to_array([1.0]),), {}),  # Missing second vector and metric tensor
        # Test passing too many arguments to a method
        (nk.angular, TypeError, (to_array([1.0]), to_array([1.0]), to_array([1.0])), {}),  # Too many arguments
        (nk.cdist, TypeError, (to_array([[1.0]]), to_array([[1.0]]), "l2", "dos"), {}),  # Too many arguments
        # Same argument as both positional and keyword
        (nk.cdist, TypeError, (to_array([[1.0]]), to_array([[1.0]]), "l2"), {"metric": "l2"}),
        # Applying real metric to complex numbers - missing kernel
        (nk.angular, LookupError, (to_array([1 + 2j]), to_array([1 + 2j])), {}),
        # Test incompatible vectors for angular
        (nk.angular, ValueError, (to_array([1.0]), to_array([1.0, 2.0])), {}),  # Different number of dimensions
        (nk.angular, TypeError, (to_array([1.0]), to_array([1], "int8")), {}),  # Floats and integers
        (nk.angular, TypeError, (to_array([1], "float32"), to_array([1], "float16")), {}),  # Different floats
    ],
)
def test_invalid_argument_handling(function, expected_error, args, kwargs):
    """Test that functions raise TypeError when called with invalid arguments."""
    with pytest.raises(expected_error):
        function(*args, **kwargs)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("metric", ["inner", "euclidean", "sqeuclidean", "angular"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dense(ndim, dtype, metric, capability, stats_fixture):
    """Compares various SIMD kernels (like Dot-products, squared Euclidean, and Cosine distances)
    with their NumPy or baseline counterparts, testing accuracy for IEEE standard floating-point types."""

    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(metric)

    accurate_dt, accurate = profile(baseline_kernel, a.astype(np.float64), b.astype(np.float64))
    expected_dt, expected = profile(baseline_kernel, a, b)
    result_dt, result = profile(simd_kernel, a, b)
    result = np.array(result)

    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(metric, ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats_fixture)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97])
@pytest.mark.parametrize(
    "dtypes",  # representation dtype and compute precision
    [
        ("float64", "float64"),
        ("float32", "float32"),
        ("float16", "float32"),  # otherwise NumPy keeps aggregating too much error
    ],
)
@pytest.mark.parametrize("metric", ["bilinear", "mahalanobis"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_curved(ndim, dtypes, metric, capability, stats_fixture):
    """Compares various SIMD kernels (like Bilinear Forms and Mahalanobis distances) for curved spaces
    with their NumPy or baseline counterparts, testing accuracy for IEEE standard floating-point types."""

    dtype, compute_dtype = dtypes
    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()

    # Let's generate some non-negative probability distributions
    a = np.abs(np.random.randn(ndim).astype(dtype))
    b = np.abs(np.random.randn(ndim).astype(dtype))
    a /= np.sum(a)
    b /= np.sum(b)

    # Let's compute the inverse of the covariance matrix, otherwise in the SciPy
    # implementation of the Mahalanobis we may face `sqrt` of a negative number.
    # We multiply the matrix by its transpose to get a positive-semi-definite matrix.
    c = np.abs(np.random.randn(ndim, ndim).astype(dtype))
    c = np.dot(c, c.T)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(metric)
    accurate_dt, accurate = profile(
        baseline_kernel,
        a.astype(np.float64),
        b.astype(np.float64),
        c.astype(np.float64),
    )
    expected_dt, expected = profile(
        baseline_kernel,
        a.astype(compute_dtype),
        b.astype(compute_dtype),
        c.astype(compute_dtype),
    )
    result_dt, result = profile(simd_kernel, a, b, c)
    result = np.array(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(metric, ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats_fixture)


@pytest.mark.skipif(is_running_under_qemu(), reason="Complex math in QEMU fails")
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97])
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_curved_complex(ndim, dtype, capability, stats_fixture):
    """Compares various SIMD kernels (like Bilinear Forms and Mahalanobis distances) for curved spaces
    with their NumPy or baseline counterparts, testing accuracy for complex IEEE standard floating-point types."""

    # Let's generate some uniform complex numbers
    np.random.seed()
    a = (np.random.randn(ndim) + 1.0j * np.random.randn(ndim)).astype(dtype)
    b = (np.random.randn(ndim) + 1.0j * np.random.randn(ndim)).astype(dtype)
    c = (np.random.randn(ndim, ndim) + 1.0j * np.random.randn(ndim, ndim)).astype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels("bilinear")
    accurate_dt, accurate = profile(
        baseline_kernel,
        a.astype(np.complex128),
        b.astype(np.complex128),
        c.astype(np.complex128),
    )
    expected_dt, expected = profile(baseline_kernel, a, b, c)
    result_dt, result = profile(simd_kernel, a, b, c)
    result = np.array(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(
        "bilinear", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats_fixture
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("metric", ["inner", "euclidean", "sqeuclidean", "angular"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dense_bf16(ndim, metric, capability, stats_fixture):
    """Compares various SIMD kernels (like Dot-products, squared Euclidean, and Cosine distances)
    with their NumPy or baseline counterparts, testing accuracy for the Brain-float format not
    natively supported by NumPy."""
    np.random.seed()
    a = np.random.randn(ndim).astype(np.float32)
    b = np.random.randn(ndim).astype(np.float32)

    a_f32_rounded, a_bf16 = f32_downcast_to_bf16(a)
    b_f32_rounded, b_bf16 = f32_downcast_to_bf16(b)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(metric)
    accurate_dt, accurate = profile(baseline_kernel, a_f32_rounded.astype(np.float64), b_f32_rounded.astype(np.float64))
    expected_dt, expected = profile(baseline_kernel, a_f32_rounded, b_f32_rounded)
    result_dt, result = profile(simd_kernel, a_bf16, b_bf16, "bf16")
    result = np.array(result)

    np.testing.assert_allclose(
        result,
        expected,
        atol=NK_ATOL,
        rtol=NK_RTOL,
        err_msg=f"""
        First `f32` operand in hex:     {hex_array(a_f32_rounded.view(np.uint32))}
        Second `f32` operand in hex:    {hex_array(b_f32_rounded.view(np.uint32))}
        First `bf16` operand in hex:    {hex_array(a_bf16)}
        Second `bf16` operand in hex:   {hex_array(b_bf16)}
        """,
    )
    collect_errors(
        metric, ndim, "bfloat16", accurate, accurate_dt, expected, expected_dt, result, result_dt, stats_fixture
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 16, 33])
@pytest.mark.parametrize("metric", ["bilinear", "mahalanobis"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_curved_bf16(ndim, metric, capability, stats_fixture):
    """Compares various SIMD kernels (like Bilinear Forms and Mahalanobis distances) for curved spaces
    with their NumPy or baseline counterparts, testing accuracy for the Brain-float format not
    natively supported by NumPy."""

    np.random.seed()

    # Let's generate some non-negative probability distributions
    a = np.abs(np.random.randn(ndim).astype(np.float32))
    b = np.abs(np.random.randn(ndim).astype(np.float32))
    a /= np.sum(a)
    b /= np.sum(b)

    # Let's compute the inverse of the covariance matrix, otherwise in the SciPy
    # implementation of the Mahalanobis we may face `sqrt` of a negative number.
    # We multiply the matrix by its transpose to get a positive-semi-definite matrix.
    c = np.abs(np.random.randn(ndim, ndim).astype(np.float32))
    c = np.dot(c, c.T)

    a_f32_rounded, a_bf16 = f32_downcast_to_bf16(a)
    b_f32_rounded, b_bf16 = f32_downcast_to_bf16(b)
    c_f32_rounded, c_bf16 = f32_downcast_to_bf16(c)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(metric)
    accurate_dt, accurate = profile(
        baseline_kernel,
        a_f32_rounded.astype(np.float64),
        b_f32_rounded.astype(np.float64),
        c_f32_rounded.astype(np.float64),
    )
    expected_dt, expected = profile(baseline_kernel, a_f32_rounded, b_f32_rounded, c_f32_rounded)
    result_dt, result = profile(simd_kernel, a_bf16, b_bf16, c_bf16, "bf16")
    result = np.array(result)

    np.testing.assert_allclose(
        result,
        expected,
        atol=NK_ATOL,
        rtol=NK_RTOL,
        err_msg=f"""
        First `f32` operand in hex:     {hex_array(a_f32_rounded.view(np.uint32))}
        Second `f32` operand in hex:    {hex_array(b_f32_rounded.view(np.uint32))}
        First `bf16` operand in hex:    {hex_array(a_bf16)}
        Second `bf16` operand in hex:   {hex_array(b_bf16)}
        Matrix `bf16` operand in hex:    {hex_array(c_bf16)}
        Matrix `bf16` operand in hex:   {hex_array(c_bf16)}
        """,
    )
    collect_errors(
        metric, ndim, "bfloat16", accurate, accurate_dt, expected, expected_dt, result, result_dt, stats_fixture
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize("metric", ["inner", "euclidean", "sqeuclidean", "angular"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dense_i8(ndim, dtype, metric, capability, stats_fixture):
    """Compares various SIMD kernels (like Dot-products, squared Euclidean, and Cosine distances)
    with their NumPy or baseline counterparts, testing accuracy for small integer types, that can't
    be directly processed with other tools without overflowing."""

    np.random.seed()
    if dtype == "int8":
        a = np.random.randint(-128, 127, size=(ndim), dtype=np.int8)
        b = np.random.randint(-128, 127, size=(ndim), dtype=np.int8)
    else:
        a = np.random.randint(0, 255, size=(ndim), dtype=np.uint8)
        b = np.random.randint(0, 255, size=(ndim), dtype=np.uint8)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(metric)

    accurate_dt, accurate = profile(baseline_kernel, a.astype(np.float64), b.astype(np.float64))
    expected_dt, expected = profile(baseline_kernel, a.astype(np.int64), b.astype(np.int64))
    result_dt, result = profile(simd_kernel, a, b)
    result = np.array(result)

    if metric == "inner":
        assert round(float(result)) == round(float(expected)), f"Expected {expected}, but got {result}"
    else:
        np.testing.assert_allclose(
            result, expected, atol=NK_ATOL, rtol=NK_RTOL
        ), f"Expected {expected}, but got {result}"
    collect_errors(metric, ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats_fixture)

    #! Fun fact: SciPy doesn't actually raise an `OverflowError` when overflow happens
    #! here, instead it raises `ValueError: math domain error` during the `sqrt` operation.
    try:
        expected_overflow = baseline_kernel(a, b)
        if np.isinf(expected_overflow):
            collect_warnings("Couldn't avoid overflow in SciPy", stats_fixture)
    except Exception as e:
        collect_warnings(f"Arbitrary error raised in SciPy: {e}", stats_fixture)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("metric", ["jaccard", "hamming"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dense_bits(ndim, metric, capability, stats_fixture):
    """Compares various SIMD kernels (like Hamming and Jaccard/Tanimoto distances) for dense bit arrays
    with their NumPy or baseline counterparts, even though, they can't process sub-byte-sized scalars."""
    np.random.seed()
    a = np.random.randint(2, size=ndim).astype(np.uint8)
    b = np.random.randint(2, size=ndim).astype(np.uint8)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(metric)
    accurate_dt, accurate = profile(baseline_kernel, a.astype(np.uint64), b.astype(np.uint64))
    expected_dt, expected = profile(baseline_kernel, a, b)
    result_dt, result = profile(simd_kernel, np.packbits(a), np.packbits(b), "bin8")
    result = np.array(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(metric, ndim, "bin8", accurate, accurate_dt, expected, expected_dt, result, result_dt, stats_fixture)

    # Aside from overriding the `dtype` parameter, we can also view as booleans
    result_dt, result = profile(simd_kernel, np.packbits(a).view(np.bool_), np.packbits(b).view(np.bool_))
    result = np.array(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(metric, ndim, "bin8", accurate, accurate_dt, expected, expected_dt, result, result_dt, stats_fixture)


@pytest.mark.skip(reason="Problems inferring the tolerance bounds for numerical errors")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_jensen_shannon(ndim, dtype, capability, stats_fixture):
    """Compares the nk.jensenshannon() function with scipy.spatial.distance.jensenshannon(), measuring the accuracy error for f16, and f32 types."""

    np.random.seed()
    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    a = np.abs(np.random.randn(ndim)).astype(dtype)
    b = np.abs(np.random.randn(ndim)).astype(dtype)
    a /= np.sum(a)
    b /= np.sum(b)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels("jensenshannon")
    accurate_dt, accurate = profile(baseline_kernel, a.astype(np.float64), b.astype(np.float64))
    expected_dt, expected = profile(baseline_kernel, a, b)
    result_dt, result = profile(simd_kernel, a, b)
    result = np.array(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(
        "jensenshannon", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats_fixture
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_angular_zero_vector(ndim, dtype, capability):
    """Tests the nk.angular() function with zero vectors, to catch division by zero errors."""
    a = np.zeros(ndim, dtype=dtype)
    b = (np.random.randn(ndim) + 1).astype(dtype)
    keep_one_capability(capability)

    result = nk.angular(a, b)
    assert result == 1, f"Expected 1, but got {result}"

    result = nk.angular(a, a)
    assert result == 0, f"Expected 0 distance from itself, but got {result}"

    result = nk.angular(b, b)
    assert abs(result) < NK_ATOL, f"Expected 0 distance from itself, but got {result}"

    # For the angular distance, the output must not be negative!
    assert np.all(result >= 0), f"Negative result for angular distance"


@pytest.mark.skip(reason="Lacks overflow protection: https://github.com/ashvardanian/NumKong/issues/206")  # TODO
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("metric", ["inner", "euclidean", "sqeuclidean", "angular"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_overflow(ndim, dtype, metric, capability):
    """Tests if the floating-point kernels are capable of detecting overflow yield the same ±inf result."""

    np.random.seed()
    a = np.random.randn(ndim)
    b = np.random.randn(ndim)

    # Replace scalar at random position with infinity
    a[np.random.randint(ndim)] = np.inf
    a = a.astype(dtype)
    b = b.astype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(metric)
    result = simd_kernel(a, b)
    assert np.isinf(result), f"Expected ±inf, but got {result}"

    #! In the Euclidean (L2) distance, SciPy raises a `ValueError` from the underlying
    #! NumPy function: `ValueError: array must not contain infs or NaNs`.
    try:
        expected_overflow = baseline_kernel(a, b)
        if not np.isinf(expected_overflow):
            collect_warnings("Overflow not detected in SciPy", stats_fixture)
    except Exception as e:
        collect_warnings(f"Arbitrary error raised in SciPy: {e}", stats_fixture)


@pytest.mark.skip(reason="Lacks overflow protection: https://github.com/ashvardanian/NumKong/issues/206")  # TODO
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [131072, 262144])
@pytest.mark.parametrize("metric", ["inner", "euclidean", "sqeuclidean", "angular"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_overflow_i8(ndim, metric, capability):
    """Tests if the integral kernels are capable of detecting overflow yield the same ±inf result,
    as with 2^16 elements accumulating "u32(u16(u8)*u16(u8))+u32" products should overflow and the
    same is true for 2^17 elements with "i32(i15(i8))*i32(i15(i8))" products.
    """

    np.random.seed()
    a = np.full(ndim, fill_value=-128, dtype=np.int8)
    b = np.full(ndim, fill_value=-128, dtype=np.int8)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(metric)
    expected = baseline_kernel(a, b)
    result = simd_kernel(a, b)
    assert np.isinf(result), f"Expected ±inf, but got {result}"

    try:
        expected_overflow = baseline_kernel(a, b)
        if not np.isinf(expected_overflow):
            collect_warnings("Overflow not detected in SciPy", stats_fixture)
    except Exception as e:
        collect_warnings(f"Arbitrary error raised in SciPy: {e}", stats_fixture)


@pytest.mark.skipif(is_running_under_qemu(), reason="Complex math in QEMU fails")
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [22, 66, 1536])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dot_complex(ndim, dtype, capability, stats_fixture):
    """Compares the nk.dot() and nk.vdot() against NumPy for complex numbers."""
    np.random.seed()
    a = (np.random.randn(ndim) + 1.0j * np.random.randn(ndim)).astype(dtype)
    b = (np.random.randn(ndim) + 1.0j * np.random.randn(ndim)).astype(dtype)

    keep_one_capability(capability)
    accurate_dt, accurate = profile(np.dot, a.astype(np.complex128), b.astype(np.complex128))
    expected_dt, expected = profile(np.dot, a, b)
    result_dt, result = profile(nk.dot, a, b)
    result = np.array(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors("dot", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats_fixture)

    accurate_dt, accurate = profile(np.vdot, a.astype(np.complex128), b.astype(np.complex128))
    expected_dt, expected = profile(np.vdot, a, b)
    result_dt, result = profile(nk.vdot, a, b)
    result = np.array(result)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(
        "vdot", ndim, dtype + "c", accurate, accurate_dt, expected, expected_dt, result, result_dt, stats_fixture
    )


@pytest.mark.skipif(is_running_under_qemu(), reason="Complex math in QEMU fails")
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [22, 66, 1536])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dot_complex_explicit(ndim, capability):
    """Compares the nk.dot() and nk.vdot() against NumPy for complex numbers."""
    np.random.seed()
    a = np.random.randn(ndim).astype(dtype=np.float32)
    b = np.random.randn(ndim).astype(dtype=np.float32)

    keep_one_capability(capability)
    expected = np.dot(a.view(np.complex64), b.view(np.complex64))
    result = nk.dot(a, b, "complex64")

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    expected = np.vdot(a.view(np.complex64), b.view(np.complex64))
    result = nk.vdot(a, b, "complex64")

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("dtype", ["uint16", "uint32"])
@pytest.mark.parametrize("first_length_bound", [10, 100, 1000])
@pytest.mark.parametrize("second_length_bound", [10, 100, 1000])
@pytest.mark.parametrize("capability", ["serial"] + possible_capabilities)
def test_intersect(dtype, first_length_bound, second_length_bound, capability):
    """Compares the nk.intersect() function with numpy.intersect1d."""

    if is_running_under_qemu() and (platform.machine() == "aarch64" or platform.machine() == "arm64"):
        pytest.skip("In QEMU `aarch64` emulation on `x86_64` the `intersect` function is not reliable")

    np.random.seed()

    a_length = np.random.randint(1, first_length_bound)
    b_length = np.random.randint(1, second_length_bound)
    a = np.random.randint(first_length_bound * 2, size=a_length, dtype=dtype)
    b = np.random.randint(second_length_bound * 2, size=b_length, dtype=dtype)

    # Remove duplicates, converting into sorted arrays
    a = np.unique(a)
    b = np.unique(b)

    keep_one_capability(capability)
    expected = baseline_intersect(a, b)
    result = nk.intersect(a, b)

    assert round(float(expected)) == round(float(result)), f"Missing {np.intersect1d(a, b)} from {a} and {b}"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
@pytest.mark.parametrize("kernel", ["scale"])
@pytest.mark.parametrize("capability", ["serial"] + possible_capabilities)
def test_scale(ndim, dtype, kernel, capability, stats_fixture):
    """"""

    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    if np.issubdtype(np.dtype(dtype), np.integer):
        dtype_info = np.iinfo(np.dtype(dtype))
        a = np.random.randint(dtype_info.min, dtype_info.max, size=ndim, dtype=dtype)
        alpha = abs(np.random.randn(1).astype(np.float64).item()) / 2
        beta = abs(np.random.randn(1).astype(np.float64).item()) / 2
        atol = 1  # ? Allow at most one rounding error per vector
        rtol = 0
    else:
        a = np.random.randn(ndim).astype(dtype)
        alpha = np.random.randn(1).astype(np.float64).item()
        beta = np.random.randn(1).astype(np.float64).item()
        atol = NK_ATOL
        rtol = NK_RTOL

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(kernel)

    accurate_dt, accurate = profile(
        baseline_kernel,
        a.astype(np.float64),
        alpha=alpha,
        beta=beta,
    )
    expected_dt, expected = profile(baseline_kernel, a, alpha=alpha, beta=beta)
    result_dt, result = profile(simd_kernel, a, alpha=alpha, beta=beta)
    result = np.array(result)

    np.testing.assert_allclose(result, expected.astype(np.float64), atol=atol, rtol=rtol)
    collect_errors(
        kernel,
        ndim,
        dtype,
        accurate,
        accurate_dt,
        expected,
        expected_dt,
        result,
        result_dt,
        stats_fixture,
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
@pytest.mark.parametrize("kernel", ["add"])
@pytest.mark.parametrize("capability", ["serial"] + possible_capabilities)
def test_add(ndim, dtype, kernel, capability, stats_fixture):
    """"""

    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    if np.issubdtype(np.dtype(dtype), np.integer):
        dtype_info = np.iinfo(np.dtype(dtype))
        a = np.random.randint(dtype_info.min, dtype_info.max, size=ndim, dtype=dtype)
        b = np.random.randint(dtype_info.min, dtype_info.max, size=ndim, dtype=dtype)
        atol = 1  # ? Allow at most one rounding error per vector
        rtol = 0
    else:
        a = np.random.randn(ndim).astype(dtype)
        b = np.random.randn(ndim).astype(dtype)
        atol = NK_ATOL
        rtol = NK_RTOL

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(kernel)

    accurate_dt, accurate = profile(
        baseline_kernel,
        a.astype(np.float64),
        b.astype(np.float64),
    )
    expected_dt, expected = profile(baseline_kernel, a, b)
    result_dt, result = profile(simd_kernel, a, b)
    result = np.array(result)

    np.testing.assert_allclose(result, expected.astype(np.float64), atol=atol, rtol=rtol)
    collect_errors(
        kernel,
        ndim,
        dtype,
        accurate,
        accurate_dt,
        expected,
        expected_dt,
        result,
        result_dt,
        stats_fixture,
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
@pytest.mark.parametrize("kernel", ["wsum"])
@pytest.mark.parametrize("capability", ["serial"] + possible_capabilities)
def test_wsum(ndim, dtype, kernel, capability, stats_fixture):
    """"""

    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    if np.issubdtype(np.dtype(dtype), np.integer):
        dtype_info = np.iinfo(np.dtype(dtype))
        a = np.random.randint(dtype_info.min, dtype_info.max, size=ndim, dtype=dtype)
        b = np.random.randint(dtype_info.min, dtype_info.max, size=ndim, dtype=dtype)
        alpha = abs(np.random.randn(1).astype(np.float64).item()) / 2
        beta = abs(np.random.randn(1).astype(np.float64).item()) / 2
        atol = 1  # ? Allow at most one rounding error per vector
        rtol = 0
    else:
        a = np.random.randn(ndim).astype(dtype)
        b = np.random.randn(ndim).astype(dtype)
        alpha = np.random.randn(1).astype(np.float64).item()
        beta = np.random.randn(1).astype(np.float64).item()
        atol = NK_ATOL
        rtol = NK_RTOL

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(kernel)

    accurate_dt, accurate = profile(
        baseline_kernel,
        a.astype(np.float64),
        b.astype(np.float64),
        alpha=alpha,
        beta=beta,
    )
    expected_dt, expected = profile(baseline_kernel, a, b, alpha=alpha, beta=beta)
    result_dt, result = profile(simd_kernel, a, b, alpha=alpha, beta=beta)
    result = np.array(result)

    np.testing.assert_allclose(result, expected.astype(np.float64), atol=atol, rtol=rtol)
    collect_errors(
        kernel,
        ndim,
        dtype,
        accurate,
        accurate_dt,
        expected,
        expected_dt,
        result,
        result_dt,
        stats_fixture,
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
@pytest.mark.parametrize("kernel", ["fma"])
@pytest.mark.parametrize("capability", ["serial"] + possible_capabilities)
def test_fma(ndim, dtype, kernel, capability, stats_fixture):
    """"""

    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    if np.issubdtype(np.dtype(dtype), np.integer):
        dtype_info = np.iinfo(np.dtype(dtype))
        a = np.random.randint(dtype_info.min, dtype_info.max, size=ndim, dtype=dtype)
        b = np.random.randint(dtype_info.min, dtype_info.max, size=ndim, dtype=dtype)
        c = np.random.randint(dtype_info.min, dtype_info.max, size=ndim, dtype=dtype)
        alpha = abs(np.random.randn(1).astype(np.float64).item()) / 512
        beta = abs(np.random.randn(1).astype(np.float64).item()) / 3
        atol = 1  # ? Allow at most one rounding error per vector
        rtol = 0
    else:
        a = np.random.randn(ndim).astype(dtype)
        b = np.random.randn(ndim).astype(dtype)
        c = np.random.randn(ndim).astype(dtype)
        alpha = np.random.randn(1).astype(np.float64).item()
        beta = np.random.randn(1).astype(np.float64).item()
        atol = NK_ATOL
        rtol = NK_RTOL

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(kernel)

    accurate_dt, accurate = profile(
        baseline_kernel,
        a.astype(np.float64),
        b.astype(np.float64),
        c.astype(np.float64),
        alpha=alpha,
        beta=beta,
    )
    expected_dt, expected = profile(baseline_kernel, a, b, c, alpha=alpha, beta=beta)
    result_dt, result = profile(simd_kernel, a, b, c, alpha=alpha, beta=beta)
    result = np.array(result)

    np.testing.assert_allclose(result, expected.astype(np.float64), atol=atol, rtol=rtol)
    collect_errors(
        kernel,
        ndim,
        dtype,
        accurate,
        accurate_dt,
        expected,
        expected_dt,
        result,
        result_dt,
        stats_fixture,
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_batch(ndim, dtype, capability):
    """Compares the nk.nk.sqeuclidean() function with scipy.spatial.distance.sqeuclidean() for a batch of vectors, measuring the accuracy error for f16, and f32 types."""

    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    keep_one_capability(capability)

    # Distance between matrixes A (N x D scalars) and B (N x D scalars) is an array with N floats.
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.random.randn(10, ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[i], B[i]) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(A, B)).astype(np.float64)
    assert np.allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # Distance between matrixes A (N x D scalars) and B (1 x D scalars) is an array with N floats.
    B = np.random.randn(1, ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[i], B[0]) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(A, B)).astype(np.float64)
    assert np.allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # Distance between matrixes A (1 x D scalars) and B (N x D scalars) is an array with N floats.
    A = np.random.randn(1, ndim).astype(dtype)
    B = np.random.randn(10, ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[0], B[i]) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(A, B)).astype(np.float64)
    assert np.allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # Distance between matrix A (N x D scalars) and array B (D scalars) is an array with N floats.
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.random.randn(ndim).astype(dtype)
    result_np = [spd.sqeuclidean(A[i], B) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(A, B)).astype(np.float64)
    assert np.allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # Distance between matrix B (N x D scalars) and array A (D scalars) is an array with N floats.
    B = np.random.randn(10, ndim).astype(dtype)
    A = np.random.randn(ndim).astype(dtype)
    result_np = [spd.sqeuclidean(B[i], A) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(B, A)).astype(np.float64)
    assert np.allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # Distance between matrixes A (N x D scalars) and B (N x D scalars) in slices of bigger matrices.
    A_extended = np.random.randn(10, ndim + 11).astype(dtype)
    B_extended = np.random.randn(10, ndim + 13).astype(dtype)
    A = A_extended[:, 1 : 1 + ndim]
    B = B_extended[:, 3 : 3 + ndim]
    assert A.base is A_extended and B.base is B_extended
    assert A.__array_interface__["strides"] is not None and B.__array_interface__["strides"] is not None
    result_np = [spd.sqeuclidean(A[i], B[i]) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(A, B)).astype(np.float64)
    assert np.allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # Distance between matrixes A (N x D scalars) and B (N x D scalars) in a transposed matrix.
    #! This requires calling `np.ascontiguousarray()` to ensure the matrix is in the right format.
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.ascontiguousarray(np.random.randn(ndim, 10).astype(dtype).T)
    result_np = [spd.sqeuclidean(A[i], B[i]) for i in range(10)]
    result_simd = np.array(nk.sqeuclidean(A, B)).astype(np.float64)
    assert np.allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)

    # Distance between matrixes A (N x D scalars) and B (N x D scalars) with a different output type.
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.random.randn(10, ndim).astype(dtype)
    result_np = np.array([spd.sqeuclidean(A[i], B[i]) for i in range(10)]).astype(np.float32)
    result_simd = np.array(nk.sqeuclidean(A, B, out_dtype="float32"))
    assert np.allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)
    assert result_simd.dtype == result_np.dtype

    # Distance between matrixes A (N x D scalars) and B (N x D scalars) with a supplied output buffer.
    A = np.random.randn(10, ndim).astype(dtype)
    B = np.random.randn(10, ndim).astype(dtype)
    result_np = np.array([spd.sqeuclidean(A[i], B[i]) for i in range(10)]).astype(np.float32)
    result_simd = np.zeros(10, dtype=np.float32)
    assert nk.sqeuclidean(A, B, out=result_simd) is None
    assert np.allclose(result_simd, result_np, atol=NK_ATOL, rtol=NK_RTOL)
    assert result_simd.dtype == result_np.dtype


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("input_dtype", ["float32", "float16"])
@pytest.mark.parametrize("out_dtype", [None, "float32", "int32"])
@pytest.mark.parametrize("metric", ["angular", "sqeuclidean"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist(ndim, input_dtype, out_dtype, metric, capability):
    """Compares the nk.cdist() function with scipy.spatial.distance.cdist(), measuring the accuracy error for f16, and f32 types using sqeuclidean and angular metrics."""

    if input_dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    keep_one_capability(capability)

    # We will work with random matrices A (M x D) and B (N x D).
    # To test their ability to handle strided inputs, we are going to add one extra dimension.
    M, N = 10, 15
    A_extended = np.random.randn(M, ndim + 1).astype(input_dtype)
    B_extended = np.random.randn(N, ndim + 3).astype(input_dtype)
    A = A_extended[:, :ndim]
    B = B_extended[:, :ndim]

    # Check if we need to round before casting to integer (to match NumKong's lround behavior)
    is_integer_output = out_dtype in ("int32", "int64", "int16", "int8", "uint32", "uint64", "uint16", "uint8")
    scipy_metric = scipy_metric_name(metric)

    if out_dtype is None:
        expected = spd.cdist(A, B, scipy_metric)
        result = nk.cdist(A, B, metric)
        #! Same functions can be used in-place, but SciPy doesn't support misaligned outputs
        expected_out = np.zeros((M, N))
        result_out_extended = np.zeros((M, N + 7))
        result_out = result_out_extended[:, :N]
        assert spd.cdist(A, B, scipy_metric, out=expected_out) is not None
        assert nk.cdist(A, B, metric, out=result_out) is None
    else:
        #! NumKong rounds to the nearest integer before casting
        scipy_result = spd.cdist(A, B, scipy_metric)
        expected = np.round(scipy_result).astype(out_dtype) if is_integer_output else scipy_result.astype(out_dtype)
        result = nk.cdist(A, B, metric, out_dtype=out_dtype)

        #! Same functions can be used in-place, but SciPy doesn't support misaligned outputs
        expected_out = np.zeros((M, N), dtype=np.float64)
        result_out_extended = np.zeros((M, N + 7), dtype=out_dtype)
        result_out = result_out_extended[:, :N]
        assert spd.cdist(A, B, scipy_metric, out=expected_out) is not None
        assert nk.cdist(A, B, metric, out=result_out) is None
        #! Moreover, SciPy supports only double-precision outputs, so we need to downcast afterwards.
        expected_out = np.round(expected_out).astype(out_dtype) if is_integer_output else expected_out.astype(out_dtype)

    # Assert they're close.
    # Integer outputs: allow ±1 tolerance since rounding differences are expected
    atol = 1 if is_integer_output else NK_ATOL
    np.testing.assert_allclose(result, expected, atol=atol, rtol=NK_RTOL)
    np.testing.assert_allclose(result_out, expected_out, atol=atol, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("input_dtype", ["float32", "float16"])
@pytest.mark.parametrize("out_dtype", [None, "float32", "int32"])
@pytest.mark.parametrize("metric", ["angular", "sqeuclidean"])
def test_cdist_itself(ndim, input_dtype, out_dtype, metric):
    """Compares the nk.cdist(A, A) function with scipy.spatial.distance.cdist(A, A), measuring the accuracy error for f16, and f32 types using sqeuclidean and angular metrics."""

    if input_dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()

    # Check if we need to round before casting to integer (to match NumKong's lround behavior)
    is_integer_output = out_dtype in ("int32", "int64", "int16", "int8", "uint32", "uint64", "uint16", "uint8")
    scipy_metric = scipy_metric_name(metric)

    A = np.random.randn(10, ndim + 1).astype(input_dtype)
    if out_dtype is None:
        expected = spd.cdist(A, A, scipy_metric)
        result = nk.cdist(A, A, metric=metric)
    else:
        #! NumKong rounds to the nearest integer before casting
        scipy_result = spd.cdist(A, A, scipy_metric)
        expected = np.round(scipy_result).astype(out_dtype) if is_integer_output else scipy_result.astype(out_dtype)
        result = nk.cdist(A, A, metric=metric, out_dtype=out_dtype)

    # Assert they're close.
    # Integer outputs: allow ±1 tolerance since rounding differences are expected
    atol = 1 if is_integer_output else NK_ATOL
    np.testing.assert_allclose(result, expected, atol=atol, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("input_dtype", ["complex128", "complex64"])
@pytest.mark.parametrize("out_dtype", [None, "complex128", "complex64"])
@pytest.mark.parametrize("metric", ["dot", "vdot"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_complex(ndim, input_dtype, out_dtype, metric, capability):
    """Compares the nk.cdist() for complex numbers to pure NumPy complex dot-products, as SciPy has no such functionality.
    The goal is to make sure that addressing multi-component numbers is done properly in both real and imaginary parts.
    """

    np.random.seed()
    keep_one_capability(capability)

    # We will work with random matrices A (M x D) and B (N x D).
    # To test their ability to handle strided inputs, we are going to add one extra dimension.
    M, N = 10, 15
    A_extended = np.random.randn(M, ndim + 1).astype(input_dtype)
    B_extended = np.random.randn(N, ndim + 3).astype(input_dtype)
    A = A_extended[:, :ndim]
    B = B_extended[:, :ndim]
    C_extended = np.random.randn(M, N + 7).astype(out_dtype if out_dtype else np.complex128)
    C = C_extended[:, :N]

    #! Unlike the `np.dot`, the `np.vdot` flattens multi-dimensional inputs into 1D arrays.
    #! So to compare the results we need to manually compute all the dot-products.
    expected = np.zeros((M, N), dtype=out_dtype if out_dtype else np.complex128)
    baseline_kernel = np.dot if metric == "dot" else np.vdot
    for i in range(M):
        for j in range(N):
            expected[i, j] = baseline_kernel(A[i], B[j])

    # Compute with NumKong:
    if out_dtype is None:
        result1d = nk.cdist(A[0], B[0], metric=metric)
        result2d = nk.cdist(A, B, metric=metric)
        assert nk.cdist(A, B, metric=metric, out=C) is None
    else:
        expected = expected.astype(out_dtype)
        result1d = nk.cdist(A[0], B[0], metric=metric, out_dtype=out_dtype)
        result2d = nk.cdist(A, B, metric=metric, out_dtype=out_dtype)
        assert nk.cdist(A, B, metric=metric, out_dtype=out_dtype, out=C) is None

    # Assert they're close.
    np.testing.assert_allclose(result1d, expected[0, 0], atol=NK_ATOL, rtol=NK_RTOL)
    np.testing.assert_allclose(result2d, expected, atol=NK_ATOL, rtol=NK_RTOL)
    np.testing.assert_allclose(C, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("out_dtype", [None, "float32", "float16", "int8"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_hamming(ndim, out_dtype, capability):
    """Compares various SIMD kernels (like Hamming and Jaccard/Tanimoto distances) for dense bit arrays
    with their NumPy or baseline counterparts, even though, they can't process sub-byte-sized scalars."""
    np.random.seed()
    keep_one_capability(capability)

    # Create random matrices A (M x D) and B (N x D).
    M, N = 10, 15
    A = np.random.randint(2, size=(M, ndim)).astype(np.uint8)
    B = np.random.randint(2, size=(N, ndim)).astype(np.uint8)
    A_bits, B_bits = np.packbits(A, axis=1), np.packbits(B, axis=1)

    if out_dtype is None:
        # SciPy divides the Hamming distance by the number of dimensions, so we need to multiply it back.
        expected = spd.cdist(A, B, "hamming") * ndim
        result = nk.cdist(A_bits, B_bits, metric="hamming", dtype="bin8")
    else:
        expected = (spd.cdist(A, B, "hamming") * ndim).astype(out_dtype)
        result = nk.cdist(A_bits, B_bits, metric="hamming", dtype="bin8", out_dtype=out_dtype)

    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [
        # Floats
        ("float64", "float64", "float64"),
        ("float32", "float32", "float32"),
        # Signed
        ("int8", "int8", "int8"),
        ("int16", "int16", "int16"),
        ("int32", "int32", "int32"),
        # Unsigned
        ("uint8", "uint8", "uint8"),
        ("uint16", "uint16", "uint16"),
        ("uint32", "uint32", "uint32"),
        # ! Can't reliably detect overflows in NumPy
        # ! ("int64", "int64", "int64"),
        # ! ("uint64", "uint64", "uint64"),
        # Mixed
        ("int16", "uint16", "float64"),
        ("uint8", "float32", "float32"),
    ],
)
@pytest.mark.parametrize(
    "kernel",
    [
        "add",
        "multiply",
    ],
)
@pytest.mark.parametrize("capability", ["serial"] + possible_capabilities)
def test_elementwise(dtype, kernel, capability, stats_fixture):
    """Tests NumPy-like compatibility interfaces on all kinds of non-contiguous arrays."""

    np.random.seed()
    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(kernel)
    first_dtype, second_dtype, output_dtype = dtype
    operator = {"add": "+", "multiply": "*"}[kernel]

    def validate(a, b, inplace_numkong):
        result_numpy = baseline_kernel(a, b)
        result_numkong = np.array(simd_kernel(a, b))
        assert (
            result_numkong.size == result_numpy.size
        ), f"Result sizes differ: {result_numkong.size} vs {result_numpy.size}"
        assert (
            result_numkong.shape == result_numpy.shape
        ), f"Result shapes differ: {result_numkong.shape} vs {result_numpy.shape}"
        assert (
            result_numkong.dtype == result_numpy.dtype
        ), f"Result dtypes differ: {result_numkong.dtype} vs {result_numpy.dtype} for ({a.dtype} {operator} {b.dtype})"

        if not np.allclose(result_numkong, result_numpy, atol=NK_ATOL, rtol=NK_RTOL):
            # ? Find the first mismatch and use it as an example in the error message
            np.testing.assert_allclose(
                result_numkong,
                result_numpy,
                atol=NK_ATOL,
                rtol=NK_RTOL,
                err_msg=f"""
                Result mismatch for ({a.dtype} {operator} {b.dtype})
                First descriptor: {a.__array_interface__}
                Second descriptor: {b.__array_interface__}
                First operand: {a}
                Second operand: {b}
                NumKong result: {result_numkong}
                NumPy result: {result_numpy}
                """,
            )

        #! NumPy constantly overflows in mixed-precision operations!
        inplace_numpy = np.empty_like(inplace_numkong)
        simd_kernel(a, b, out=inplace_numkong)
        baseline_kernel(a, b, out=inplace_numpy)

        assert (
            inplace_numkong.size == inplace_numpy.size
        ), f"Inplace sizes differ: {inplace_numkong.size} vs {inplace_numpy.size}"
        assert (
            inplace_numkong.shape == inplace_numpy.shape
        ), f"Inplace shapes differ: {inplace_numkong.shape} vs {inplace_numpy.shape}"
        assert (
            inplace_numkong.dtype == inplace_numpy.dtype
        ), f"Inplace dtypes differ: {inplace_numkong.dtype} vs {inplace_numpy.dtype} for ({a.dtype} {operator} {b.dtype})"

        # Let's count the number of overflows in NumPy:
        overflow_count = np.sum(np.isclose(inplace_numkong, inplace_numpy, atol=NK_ATOL, rtol=NK_RTOL))
        if overflow_count:
            collect_warnings(
                f"NumPy overflow in ({a.dtype} {operator} {b.dtype} -> {output_dtype})",
                stats_fixture,
            )
        return result_numkong

    # Vector-Vector addition
    a = random_of_dtype(first_dtype, (6,))
    b = random_of_dtype(second_dtype, (6,))
    o = np.zeros(6).astype(output_dtype)
    validate(a, b, o)

    # Larger Vector-Vector addition
    a = random_of_dtype(first_dtype, (47,))
    b = random_of_dtype(second_dtype, (47,))
    o = np.zeros(47).astype(output_dtype)
    validate(a, b, o)

    # Much larger Vector-Vector addition
    a = random_of_dtype(first_dtype, (247,))
    b = random_of_dtype(second_dtype, (247,))
    o = np.zeros(247).astype(output_dtype)
    validate(a, b, o)

    # Vector-Scalar addition
    validate(a, np.int8(-11), o)
    validate(a, np.uint8(11), o)
    validate(a, np.float32(11.0), o)

    # Scalar-Vector addition
    validate(np.int8(-13), b, o)
    validate(np.uint8(13), b, o)
    validate(np.float32(13.0), b, o)

    # Matrix-Matrix addition
    a = random_of_dtype(first_dtype, (10, 47))
    b = random_of_dtype(second_dtype, (10, 47))
    o = np.zeros((10, 47)).astype(output_dtype)
    validate(a, b, o)

    # Strided Matrix-Matrix addition
    a_extended = random_of_dtype(first_dtype, (10, 47))
    b_extended = random_of_dtype(second_dtype, (10, 47))
    a = a_extended[::2, 1:]  # Every second (even) row, all columns but the first
    b = b_extended[1::2, :-1]  # Every second (odd) row, all columns but the last
    o = np.zeros((5, 46)).astype(output_dtype)
    validate(a, b, o)

    # Strided Matrix-Matrix addition in with reverse order of different dimensions
    a_extended = random_of_dtype(first_dtype, (10, 47))
    b_extended = random_of_dtype(second_dtype, (10, 47))
    a = a_extended[::-2, 1:]  # Every second (even) row (reverse), all columns but the first
    b = b_extended[1::2, -2::-1]  # Every second (odd) row, all columns (reversed) but the last
    o = np.zeros((5, 46)).astype(output_dtype)
    validate(a, b, o)

    # Raise an error if shapes are different
    a = random_of_dtype(first_dtype, (10, 47))
    b = random_of_dtype(second_dtype, (10, 46))
    with pytest.raises(ValueError):
        baseline_kernel(a, b)
    with pytest.raises(ValueError):
        simd_kernel(a, b)

    # Raise an error if shapes are different
    a = random_of_dtype(first_dtype, (6, 2, 3))
    b = random_of_dtype(second_dtype, (6, 6))
    with pytest.raises(ValueError):
        baseline_kernel(a, b)
    with pytest.raises(ValueError):
        simd_kernel(a, b)

    # Make sure broadcasting works as expected for a single scalar
    a = random_of_dtype(first_dtype, (4, 7, 5, 3))
    b = random_of_dtype(second_dtype, (1,))
    o = np.zeros((4, 7, 5, 3)).astype(output_dtype)
    assert validate(a, b, o).shape == (4, 7, 5, 3)

    # Make sure broadcasting works as expected for a unit tensor
    a = random_of_dtype(first_dtype, (4, 7, 5, 3))
    b = random_of_dtype(second_dtype, (1, 1, 1, 1))
    o = np.zeros((4, 7, 5, 3)).astype(output_dtype)
    assert validate(a, b, o).shape == (4, 7, 5, 3)

    # Make sure broadcasting works as expected for 2 unit tensors of different rank
    a = random_of_dtype(first_dtype, (1, 1, 1, 1))
    b = random_of_dtype(second_dtype, (1, 1, 1))
    o = np.zeros((1, 1, 1, 1)).astype(output_dtype)
    assert validate(a, b, o).shape == (1, 1, 1, 1)

    # Make sure broadcasting works as expected for a unit tensor of different rank
    a = random_of_dtype(first_dtype, (4, 7, 5, 3))
    b = random_of_dtype(second_dtype, (1, 1, 1))
    o = np.zeros((4, 7, 5, 3)).astype(output_dtype)
    assert validate(a, b, o).shape == (4, 7, 5, 3)

    # Make sure broadcasting works as expected for an added dimension
    a = random_of_dtype(first_dtype, (4, 7, 5, 3))
    b = random_of_dtype(second_dtype, (1, 1, 1, 1, 1))
    o = np.zeros((1, 4, 7, 5, 3)).astype(output_dtype)
    assert validate(a, b, o).shape == (1, 4, 7, 5, 3)

    # Make sure broadcasting works as expected for mixed origin broadcasting
    a = random_of_dtype(first_dtype, (4, 7, 5, 3))
    b = random_of_dtype(second_dtype, (2, 1, 1, 1, 1))
    o = np.zeros((2, 4, 7, 5, 3)).astype(output_dtype)
    assert validate(a, b, o).shape == (2, 4, 7, 5, 3)

    # Make sure broadcasting works as expected
    a = random_of_dtype(first_dtype, (4, 7, 5, 1))
    b = random_of_dtype(second_dtype, (4, 1, 5, 3))
    o = np.zeros((4, 7, 5, 3)).astype(output_dtype)
    assert validate(a, b, o).shape == (4, 7, 5, 3)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("kernel", ["scale"])
@pytest.mark.parametrize("capability", ["serial"] + possible_capabilities)
def test_scale_extended(ndim, dtype, kernel, capability, stats_fixture):
    """Extended tests for scale() function with various dtypes and alpha/beta parameters.
    Tests the formula: result = alpha * x + beta
    """
    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(kernel)

    # Test with different alpha and beta values
    a = np.random.randn(ndim).astype(dtype)

    # Test case 1: Standard alpha and beta
    alpha = np.random.randn(1).astype(np.float64).item()
    beta = np.random.randn(1).astype(np.float64).item()
    expected = baseline_kernel(a, alpha=alpha, beta=beta)
    result = np.array(simd_kernel(a, alpha=alpha, beta=beta))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 2: Zero alpha (should give all beta)
    alpha = 0.0
    beta = 1.5
    expected = baseline_kernel(a, alpha=alpha, beta=beta)
    result = np.array(simd_kernel(a, alpha=alpha, beta=beta))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 3: Zero beta (should give alpha * x)
    alpha = 2.0
    beta = 0.0
    expected = baseline_kernel(a, alpha=alpha, beta=beta)
    result = np.array(simd_kernel(a, alpha=alpha, beta=beta))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 4: Negative alpha and beta
    alpha = -1.5
    beta = -2.0
    expected = baseline_kernel(a, alpha=alpha, beta=beta)
    result = np.array(simd_kernel(a, alpha=alpha, beta=beta))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("kernel", ["add"])
@pytest.mark.parametrize("capability", ["serial"] + possible_capabilities)
def test_add_extended(ndim, dtype, kernel, capability, stats_fixture):
    """Extended tests for add() function with various dtypes.
    Tests element-wise addition: result = x + y
    """
    if dtype == "float16" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(kernel)

    # Test case 1: Standard random vectors
    a = np.random.randn(ndim).astype(dtype)
    b = np.random.randn(ndim).astype(dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 2: One vector is zeros
    a = np.random.randn(ndim).astype(dtype)
    b = np.zeros(ndim).astype(dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 3: Both vectors are the same
    a = np.random.randn(ndim).astype(dtype)
    expected = baseline_kernel(a, a)
    result = np.array(simd_kernel(a, a))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 4: Negative values
    a = -np.abs(np.random.randn(ndim).astype(dtype))
    b = -np.abs(np.random.randn(ndim).astype(dtype))
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected.astype(np.float64), atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize(
    "dtype",
    [
        # Same dtypes
        ("float64", "float64", "float64"),
        ("float32", "float32", "float32"),
        ("float16", "float16", "float16"),
        # Mixed dtypes
        ("float32", "float64", "float64"),
        ("float16", "float32", "float32"),
    ],
)
@pytest.mark.parametrize("kernel", ["add"])
@pytest.mark.parametrize("capability", ["serial"] + possible_capabilities)
def test_add_extended(ndim, dtype, kernel, capability, stats_fixture):
    """Extended tests for add() function with broadcasting and various dtypes.
    Tests element-wise addition with broadcasting support: result = x + y
    """
    first_dtype, second_dtype, output_dtype = dtype
    if (first_dtype == "float16" or second_dtype == "float16") and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(kernel)

    # Test case 1: Vector-Vector addition
    a = np.random.randn(ndim).astype(first_dtype)
    b = np.random.randn(ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 2: Scalar-Vector addition (broadcast scalar to vector)
    a = np.random.randn(1).astype(first_dtype)[0]  # Extract scalar
    b = np.random.randn(ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 3: Vector-Scalar addition (broadcast scalar to vector)
    a = np.random.randn(ndim).astype(first_dtype)
    b = np.random.randn(1).astype(second_dtype)[0]  # Extract scalar
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 4: Matrix-Matrix addition
    a = np.random.randn(10, ndim // 10 if ndim >= 10 else ndim).astype(first_dtype)
    b = np.random.randn(10, ndim // 10 if ndim >= 10 else ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 5: In-place addition (with output buffer)
    a = np.random.randn(ndim).astype(first_dtype)
    b = np.random.randn(ndim).astype(second_dtype)
    out_expected = np.zeros(ndim).astype(output_dtype)
    out_result = np.zeros(ndim).astype(output_dtype)
    baseline_kernel(a, b, out=out_expected)
    simd_kernel(a, b, out=out_result)
    np.testing.assert_allclose(out_result, out_expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [11, 97, 1536])
@pytest.mark.parametrize(
    "dtype",
    [
        # Same dtypes
        ("float64", "float64", "float64"),
        ("float32", "float32", "float32"),
        ("float16", "float16", "float16"),
        # Mixed dtypes
        ("float32", "float64", "float64"),
        ("float16", "float32", "float32"),
    ],
)
@pytest.mark.parametrize("kernel", ["multiply"])
@pytest.mark.parametrize("capability", ["serial"] + possible_capabilities)
def test_multiply_extended(ndim, dtype, kernel, capability, stats_fixture):
    """Extended tests for multiply() function with broadcasting and various dtypes.
    Tests element-wise multiplication with broadcasting support: result = x * y
    """
    first_dtype, second_dtype, output_dtype = dtype
    if (first_dtype == "float16" or second_dtype == "float16") and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    keep_one_capability(capability)
    baseline_kernel, simd_kernel = name_to_kernels(kernel)

    # Test case 1: Vector-Vector multiplication
    a = np.random.randn(ndim).astype(first_dtype)
    b = np.random.randn(ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 2: Scalar-Vector multiplication (broadcast scalar to vector)
    a = np.random.randn(1).astype(first_dtype)[0]  # Extract scalar
    b = np.random.randn(ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 3: Vector-Scalar multiplication (broadcast scalar to vector)
    a = np.random.randn(ndim).astype(first_dtype)
    b = np.random.randn(1).astype(second_dtype)[0]  # Extract scalar
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 4: Matrix-Matrix multiplication
    a = np.random.randn(10, ndim // 10 if ndim >= 10 else ndim).astype(first_dtype)
    b = np.random.randn(10, ndim // 10 if ndim >= 10 else ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 5: In-place multiplication (with output buffer)
    a = np.random.randn(ndim).astype(first_dtype)
    b = np.random.randn(ndim).astype(second_dtype)
    out_expected = np.zeros(ndim).astype(output_dtype)
    out_result = np.zeros(ndim).astype(output_dtype)
    baseline_kernel(a, b, out=out_expected)
    simd_kernel(a, b, out=out_result)
    np.testing.assert_allclose(out_result, out_expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 6: Multiplication with zeros
    a = np.random.randn(ndim).astype(first_dtype)
    b = np.zeros(ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    # Test case 7: Multiplication with ones (should give original)
    a = np.random.randn(ndim).astype(first_dtype)
    b = np.ones(ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b))
    np.testing.assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


def test_gil_free_threading():
    """Test NumKong in Python 3.13t free-threaded mode if available."""
    import sys
    import sysconfig

    # Check if we're in a GIL-free environment
    # https://py-free-threading.github.io/running-gil-disabled/
    version = sys.version_info
    if version.major == 3 and version.minor >= 13:
        is_free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        if not is_free_threaded:
            pytest.skip("Uses non-free-threaded Python, skipping GIL-related tests")
        if sys._is_gil_enabled():
            pytest.skip("GIL is enabled, skipping GIL-related tests")
    else:
        pytest.skip("Python < 3.13t, skipping GIL-related tests")

    import multiprocessing
    import concurrent.futures

    num_threads = multiprocessing.cpu_count()
    vectors_a = np.random.rand(32 * 1024 * num_threads, 1024).astype(np.float32)
    vectors_b = np.random.rand(32 * 1024 * num_threads, 1024).astype(np.float32)
    distances = np.zeros(vectors_a.shape[0], dtype=np.float32)

    def compute_batch(start_idx, end_idx) -> float:
        """Compute angular distances for a batch."""
        slice_a = vectors_a[start_idx:end_idx]
        slice_b = vectors_b[start_idx:end_idx]
        slice_distances = distances[start_idx:end_idx]
        nk.angular(slice_a, slice_b, out=slice_distances)
        return sum(slice_distances)

    def compute_with_threads(threads: int) -> float:
        """Compute angular distances using multiple threads."""
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

    # Dual-threaded baseline is better than single-threaded,
    # as it will include the overhead of thread management.
    start_time = time.time()
    baseline_sum = compute_with_threads(2)
    end_time = time.time()
    baseline_duration = end_time - start_time

    # Multi-threaded execution, using all available threads
    start_time = time.time()
    multi_sum = compute_with_threads(num_threads)
    end_time = time.time()
    multi_duration = end_time - start_time

    # Verify results are the same length and reasonable
    assert np.allclose(
        baseline_sum, multi_sum, atol=NK_ATOL, rtol=NK_RTOL
    ), f"Results differ: baseline {baseline_sum} vs multi-threaded {multi_sum}"

    # Warn if multi-threaded execution is slower than the baseline
    if baseline_duration < multi_duration:
        warnings.warn(
            f"{num_threads}-threaded execution took longer than 2-threaded baseline: {multi_duration:.2f}s vs {baseline_duration:.2f}s",
            UserWarning,
        )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [1, 10, 100])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("capability", ["serial"] + possible_capabilities)
def test_haversine(ndim, dtype, capability, stats_fixture):
    """Tests Haversine (great-circle) distance computation for geospatial coordinates."""

    if dtype == "float32" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    keep_one_capability(capability)

    # Generate random coordinates in radians
    # Latitude: -π/2 to π/2, Longitude: -π to π
    first_latitudes = (np.random.rand(ndim) - 0.5) * np.pi
    first_longitudes = (np.random.rand(ndim) - 0.5) * 2 * np.pi
    second_latitudes = (np.random.rand(ndim) - 0.5) * np.pi
    second_longitudes = (np.random.rand(ndim) - 0.5) * 2 * np.pi

    first_latitudes = first_latitudes.astype(dtype)
    first_longitudes = first_longitudes.astype(dtype)
    second_latitudes = second_latitudes.astype(dtype)
    second_longitudes = second_longitudes.astype(dtype)

    # Compute expected results using baseline
    expected = np.array(
        [
            baseline_haversine(
                first_latitudes[i].astype(np.float64),
                first_longitudes[i].astype(np.float64),
                second_latitudes[i].astype(np.float64),
                second_longitudes[i].astype(np.float64),
            )
            for i in range(ndim)
        ]
    )

    # Compute using NumKong
    result = nk.haversine(first_latitudes, first_longitudes, second_latitudes, second_longitudes)
    result = np.array(result)

    # For geospatial, allow larger tolerance due to transcendental function differences
    # Different sin/cos/atan implementations can cause ~0.1% differences
    absolute_tolerance = 10.0  # 10 meter tolerance for small distances
    relative_tolerance = 1e-2  # 1% relative tolerance for numerical differences
    np.testing.assert_allclose(result, expected, atol=absolute_tolerance, rtol=relative_tolerance)

    # Record statistics
    accurate_dt, accurate = 0, expected
    expected_dt = 0
    result_dt = 0
    collect_errors(
        "haversine", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats_fixture
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [1, 10, 100])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("capability", ["serial"] + possible_capabilities)
def test_vincenty(ndim, dtype, capability, stats_fixture):
    """Tests Vincenty (ellipsoidal geodesic) distance computation for geospatial coordinates."""

    if dtype == "float32" and is_running_under_qemu():
        pytest.skip("Testing low-precision math isn't reliable in QEMU")

    np.random.seed()
    keep_one_capability(capability)

    # Generate random coordinates in radians
    # Latitude: -π/2 to π/2, Longitude: -π to π
    # Avoid antipodal points where Vincenty can have convergence issues
    first_latitudes = (np.random.rand(ndim) - 0.5) * np.pi * 0.9  # Slightly reduced range
    first_longitudes = (np.random.rand(ndim) - 0.5) * 2 * np.pi * 0.9
    second_latitudes = (np.random.rand(ndim) - 0.5) * np.pi * 0.9
    second_longitudes = (np.random.rand(ndim) - 0.5) * 2 * np.pi * 0.9

    first_latitudes = first_latitudes.astype(dtype)
    first_longitudes = first_longitudes.astype(dtype)
    second_latitudes = second_latitudes.astype(dtype)
    second_longitudes = second_longitudes.astype(dtype)

    # Compute expected results using baseline
    expected = np.array(
        [
            baseline_vincenty(
                first_latitudes[i].astype(np.float64),
                first_longitudes[i].astype(np.float64),
                second_latitudes[i].astype(np.float64),
                second_longitudes[i].astype(np.float64),
            )
            for i in range(ndim)
        ]
    )

    # Compute using NumKong
    result = nk.vincenty(first_latitudes, first_longitudes, second_latitudes, second_longitudes)
    result = np.array(result)

    # For geospatial, allow larger tolerance due to transcendental function differences
    # Different sin/cos/atan/tan implementations and iterative convergence can cause ~1% differences
    absolute_tolerance = 100.0  # 100 meter tolerance for complex iterative algorithm
    relative_tolerance = 1e-2  # 1% relative tolerance for numerical differences
    np.testing.assert_allclose(result, expected, atol=absolute_tolerance, rtol=relative_tolerance)

    # Record statistics
    accurate_dt, accurate = 0, expected
    expected_dt = 0
    result_dt = 0
    collect_errors(
        "vincenty", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats_fixture
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_haversine_known_values():
    """Tests Haversine distance with known reference values."""

    # NYC (40.7128°N, 74.0060°W) to LA (34.0522°N, 118.2437°W)
    # Expected distance: ~3940 km
    new_york_latitude = np.radians(40.7128)
    new_york_longitude = np.radians(-74.0060)
    los_angeles_latitude = np.radians(34.0522)
    los_angeles_longitude = np.radians(-118.2437)

    first_latitudes = np.array([new_york_latitude], dtype=np.float64)
    first_longitudes = np.array([new_york_longitude], dtype=np.float64)
    second_latitudes = np.array([los_angeles_latitude], dtype=np.float64)
    second_longitudes = np.array([los_angeles_longitude], dtype=np.float64)

    result = np.array(nk.haversine(first_latitudes, first_longitudes, second_latitudes, second_longitudes))
    result_kilometers = float(result[0]) / 1000

    # NYC to LA is approximately 3940 km (great circle)
    assert 3800 < result_kilometers < 4100, f"Expected ~3940 km, got {result_kilometers:.0f} km"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_geospatial_out_parameter():
    """Tests that the 'out' parameter works for geospatial functions."""

    count = 10
    np.random.seed(42)  # For reproducibility
    first_latitudes = np.random.rand(count).astype(np.float64) * np.pi - np.pi / 2
    first_longitudes = np.random.rand(count).astype(np.float64) * 2 * np.pi - np.pi
    second_latitudes = np.random.rand(count).astype(np.float64) * np.pi - np.pi / 2
    second_longitudes = np.random.rand(count).astype(np.float64) * 2 * np.pi - np.pi

    # Test with pre-allocated output
    output_distances = np.zeros(count, dtype=np.float64)
    result = nk.haversine(first_latitudes, first_longitudes, second_latitudes, second_longitudes, out=output_distances)
    assert result is None, "Expected None when using out parameter"
    assert np.all(output_distances >= 0), "Output should contain non-negative distances"

    # Compare with regular call
    expected = np.array(nk.haversine(first_latitudes, first_longitudes, second_latitudes, second_longitudes))
    np.testing.assert_allclose(output_distances, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_properties():
    """Tests that Tensor properties work correctly for various shapes."""
    np.random.seed(42)

    # Test with pairwise distances (2D result)
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    # Check basic properties
    assert hasattr(result, "shape"), "Tensor should have 'shape' property"
    assert hasattr(result, "dtype"), "Tensor should have 'dtype' property"
    assert hasattr(result, "ndim"), "Tensor should have 'ndim' property"
    assert hasattr(result, "size"), "Tensor should have 'size' property"
    assert hasattr(result, "nbytes"), "Tensor should have 'nbytes' property"
    assert hasattr(result, "strides"), "Tensor should have 'strides' property"
    assert hasattr(result, "itemsize"), "Tensor should have 'itemsize' property"

    # Check values
    assert result.shape == (5, 7), f"Expected shape (5, 7), got {result.shape}"
    assert result.dtype == "float64", f"Expected dtype 'float64', got {result.dtype}"
    assert result.ndim == 2, f"Expected ndim 2, got {result.ndim}"
    assert result.size == 35, f"Expected size 35, got {result.size}"
    assert result.nbytes == 35 * 8, f"Expected nbytes {35 * 8}, got {result.nbytes}"
    assert result.itemsize == 8, f"Expected itemsize 8, got {result.itemsize}"
    assert isinstance(result.strides, tuple), "strides should be a tuple"
    assert len(result.strides) == 2, "strides should have 2 elements for 2D tensor"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_properties_1d():
    """Tests Tensor properties for 1D results."""
    np.random.seed(42)

    # Test with row-wise distances (1D result)
    a = np.random.rand(10, 128).astype(np.float64)
    b = np.random.rand(10, 128).astype(np.float64)
    result = nk.sqeuclidean(a, b)

    assert result.shape == (10,), f"Expected shape (10,), got {result.shape}"
    assert result.ndim == 1, f"Expected ndim 1, got {result.ndim}"
    assert result.size == 10, f"Expected size 10, got {result.size}"
    assert len(result.strides) == 1, "strides should have 1 element for 1D tensor"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_len():
    """Tests that __len__() works correctly for Tensor."""
    np.random.seed(42)

    # 2D tensor
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result_2d = nk.cdist(a, b, metric="sqeuclidean")
    assert len(result_2d) == 5, f"Expected len 5, got {len(result_2d)}"

    # 1D tensor
    a = np.random.rand(10, 128).astype(np.float64)
    b = np.random.rand(10, 128).astype(np.float64)
    result_1d = nk.sqeuclidean(a, b)
    assert len(result_1d) == 10, f"Expected len 10, got {len(result_1d)}"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_repr():
    """Tests that __repr__() returns proper format."""
    np.random.seed(42)

    # 2D tensor
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    repr_str = repr(result)
    assert "Tensor" in repr_str, f"repr should contain 'Tensor', got: {repr_str}"
    assert "shape=" in repr_str, f"repr should contain 'shape=', got: {repr_str}"
    assert "dtype=" in repr_str, f"repr should contain 'dtype=', got: {repr_str}"
    assert "(5, 7)" in repr_str, f"repr should contain shape (5, 7), got: {repr_str}"
    assert "float64" in repr_str, f"repr should contain 'float64', got: {repr_str}"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_indexing():
    """Tests that __getitem__() works for various index types."""
    np.random.seed(42)

    # Create 2D result
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    # Convert to numpy for comparison
    expected = np.asarray(result)

    # Single integer index - returns sub-tensor (row)
    row0 = result[0]
    assert hasattr(row0, "shape"), "Single index should return Tensor"
    assert row0.shape == (7,), f"Expected shape (7,), got {row0.shape}"

    # Negative indexing
    row_last = result[-1]
    assert row_last.shape == (7,), f"Expected shape (7,), got {row_last.shape}"
    np.testing.assert_allclose(np.asarray(row_last), expected[-1], rtol=1e-6)

    # Full tuple indexing - returns scalar
    val = result[2, 3]
    assert isinstance(val, float), f"Full index should return Python float, got {type(val)}"
    np.testing.assert_allclose(val, expected[2, 3], rtol=1e-6)

    # Negative tuple indexing
    val_neg = result[-1, -1]
    assert isinstance(val_neg, float), f"Full index should return Python float, got {type(val_neg)}"
    np.testing.assert_allclose(val_neg, expected[-1, -1], rtol=1e-6)

    # Out of bounds should raise IndexError
    with pytest.raises(IndexError):
        _ = result[10]

    with pytest.raises(IndexError):
        _ = result[0, 100]


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_iteration():
    """Tests that __iter__() works correctly."""
    np.random.seed(42)

    # 2D tensor - iterates over rows
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")
    expected = np.asarray(result)

    count = 0
    for i, row in enumerate(result):
        count += 1
        assert row.shape == (7,), f"Row {i} should have shape (7,), got {row.shape}"
        np.testing.assert_allclose(np.asarray(row), expected[i], rtol=1e-6)

    assert count == 5, f"Expected 5 iterations, got {count}"

    # 1D tensor - iterates over scalars
    a = np.random.rand(3, 128).astype(np.float64)
    b = np.random.rand(3, 128).astype(np.float64)
    result_1d = nk.sqeuclidean(a, b)
    expected_1d = np.asarray(result_1d)

    items = list(result_1d)
    assert len(items) == 3, f"Expected 3 items, got {len(items)}"
    for i, item in enumerate(items):
        assert isinstance(item, float), f"1D iteration should yield floats, got {type(item)}"
        np.testing.assert_allclose(item, expected_1d[i], rtol=1e-6)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_numpy_interop():
    """Tests NumPy interoperability via buffer protocol."""
    np.random.seed(42)

    # Create tensor
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    # Convert to numpy array
    arr = np.asarray(result)
    assert isinstance(arr, np.ndarray), "np.asarray should return ndarray"
    assert arr.shape == (5, 7), f"Expected shape (5, 7), got {arr.shape}"
    assert arr.dtype == np.float64, f"Expected dtype float64, got {arr.dtype}"

    # Values should be valid squared Euclidean distances (non-negative)
    assert np.all(arr >= 0), "Squared Euclidean distances should be non-negative"

    # Test memoryview
    mv = memoryview(result)
    assert mv.ndim == 2, f"memoryview should be 2D, got {mv.ndim}"
    assert mv.shape == (5, 7), f"memoryview shape should be (5, 7), got {mv.shape}"

    # Verify data content matches
    arr_from_mv = np.asarray(mv)
    np.testing.assert_array_equal(arr, arr_from_mv)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_scalar_conversion():
    """Tests float() and int() conversion for scalar results."""
    np.random.seed(42)

    # Single-vector sqeuclidean returns a Python float directly, not a tensor
    a = np.random.rand(128).astype(np.float64)
    b = np.random.rand(128).astype(np.float64)
    result = nk.sqeuclidean(a, b)
    assert isinstance(result, float), f"Single pair should return float, got {type(result)}"
    assert result >= 0, "Squared Euclidean distance should be non-negative"

    # Test with 2D tensor - multi-element should not convert directly
    a2 = np.random.rand(3, 128).astype(np.float64)
    b2 = np.random.rand(5, 128).astype(np.float64)
    result2 = nk.cdist(a2, b2, metric="sqeuclidean")

    # Multi-element tensor should raise when trying float()
    with pytest.raises(TypeError):
        float(result2)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_dtype_consistency():
    """Tests that dtype property matches actual data type."""
    np.random.seed(42)

    # Test various input types - result should always be float64
    for input_dtype in [np.float32, np.float64]:
        a = np.random.rand(5, 128).astype(input_dtype)
        b = np.random.rand(7, 128).astype(input_dtype)
        result = nk.cdist(a, b, metric="sqeuclidean")

        assert result.dtype == "float64", f"dtype should be 'float64', got {result.dtype}"
        arr = np.asarray(result)
        assert arr.dtype == np.float64, f"numpy dtype should be float64, got {arr.dtype}"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_strides():
    """Tests that strides are correctly computed and usable."""
    np.random.seed(42)

    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    strides = result.strides
    assert isinstance(strides, tuple), "strides should be tuple"
    assert len(strides) == 2, "2D tensor should have 2 strides"

    # For row-major C-contiguous: stride[0] = ncols * itemsize, stride[1] = itemsize
    expected_stride0 = 7 * 8  # 7 columns * 8 bytes per float64
    expected_stride1 = 8  # 8 bytes per float64
    assert strides == (
        expected_stride0,
        expected_stride1,
    ), f"Expected strides ({expected_stride0}, {expected_stride1}), got {strides}"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_array_interface():
    """Tests __array_interface__ property for legacy NumPy interop."""
    np.random.seed(42)

    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    # Check __array_interface__ exists and has required keys
    ai = result.__array_interface__
    assert isinstance(ai, dict), "__array_interface__ should be a dict"
    assert "shape" in ai, "__array_interface__ should have 'shape'"
    assert "typestr" in ai, "__array_interface__ should have 'typestr'"
    assert "data" in ai, "__array_interface__ should have 'data'"
    assert "strides" in ai, "__array_interface__ should have 'strides'"
    assert "version" in ai, "__array_interface__ should have 'version'"

    # Verify values
    assert ai["shape"] == (5, 7), f"Expected shape (5, 7), got {ai['shape']}"
    assert ai["typestr"] == "<f8", f"Expected typestr '<f8' for float64, got {ai['typestr']}"
    assert ai["version"] == 3, f"Expected version 3, got {ai['version']}"
    assert isinstance(ai["data"], tuple), "data should be a tuple (ptr, readonly)"
    assert len(ai["data"]) == 2, "data should have 2 elements"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_transpose():
    """Tests .T property for tensor transpose."""
    np.random.seed(42)

    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    # Test transpose
    t = result.T
    assert t.shape == (7, 5), f"Transpose shape should be (7, 5), got {t.shape}"

    # Verify data is correctly transposed
    result_arr = np.asarray(result)
    t_arr = np.asarray(t)
    np.testing.assert_allclose(result_arr.T, t_arr, rtol=1e-10)

    # Test 1D tensor transpose (should be no-op)
    a1d = np.random.rand(10, 128).astype(np.float64)
    b1d = np.random.rand(10, 128).astype(np.float64)
    result1d = nk.sqeuclidean(a1d, b1d)
    t1d = result1d.T
    assert t1d.shape == result1d.shape, "1D tensor transpose should be no-op"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_str():
    """Tests __str__() pretty-print representation."""
    np.random.seed(42)

    a = np.random.rand(3, 128).astype(np.float64)
    b = np.random.rand(4, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    # Test str output
    s = str(result)
    assert "[" in s, "__str__ should produce array-like output with brackets"
    assert "]" in s, "__str__ should produce array-like output with brackets"
    # Should contain some numbers
    assert any(c.isdigit() for c in s), "__str__ should contain numeric values"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_equality():
    """Tests __eq__ and __ne__ comparison operators."""
    np.random.seed(42)

    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    # Test equality with copy
    copy = result.copy()
    assert result == copy, "Tensor should equal its copy"
    assert not (result != copy), "Tensor should not be != its copy"

    # Test inequality with different tensor
    a2 = np.random.rand(5, 128).astype(np.float64)
    b2 = np.random.rand(7, 128).astype(np.float64)
    result2 = nk.cdist(a2, b2, metric="sqeuclidean")
    assert result != result2, "Different tensors should not be equal"

    # Test with different shapes
    a3 = np.random.rand(3, 128).astype(np.float64)
    b3 = np.random.rand(4, 128).astype(np.float64)
    result3 = nk.cdist(a3, b3, metric="sqeuclidean")
    assert result != result3, "Tensors with different shapes should not be equal"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_copy():
    """Tests .copy() method for deep copying."""
    np.random.seed(42)

    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    # Test copy
    copy = result.copy()
    assert copy.shape == result.shape, "Copy should have same shape"
    assert copy.dtype == result.dtype, "Copy should have same dtype"
    assert result == copy, "Copy should equal original"

    # Verify it's a deep copy (modifying numpy view shouldn't affect original)
    result_arr = np.asarray(result)
    copy_arr = np.asarray(copy)
    np.testing.assert_array_equal(result_arr, copy_arr)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_reshape():
    """Tests .reshape() method."""
    np.random.seed(42)

    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    # Test reshape to 1D
    flat = result.reshape(35)
    assert flat.shape == (35,), f"Expected shape (35,), got {flat.shape}"
    assert flat.size == 35, f"Expected size 35, got {flat.size}"

    # Test reshape back to 2D
    back = flat.reshape(5, 7)
    assert back.shape == (5, 7), f"Expected shape (5, 7), got {back.shape}"

    # Test reshape with tuple argument
    reshaped = result.reshape((7, 5))
    assert reshaped.shape == (7, 5), f"Expected shape (7, 5), got {reshaped.shape}"

    # Test invalid reshape (wrong total size)
    with pytest.raises(ValueError):
        result.reshape(10, 10)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_slicing():
    """Tests slicing support with tensor[start:stop:step]."""
    np.random.seed(42)

    a = np.random.rand(10, 128).astype(np.float64)
    b = np.random.rand(8, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")
    expected = np.asarray(result)

    # Test basic slicing
    sliced = result[2:5]
    assert sliced.shape == (3, 8), f"Expected shape (3, 8), got {sliced.shape}"
    np.testing.assert_allclose(np.asarray(sliced), expected[2:5], rtol=1e-10)

    # Test slicing with step
    step_sliced = result[::2]
    assert step_sliced.shape == (5, 8), f"Expected shape (5, 8), got {step_sliced.shape}"
    np.testing.assert_allclose(np.asarray(step_sliced), expected[::2], rtol=1e-10)

    # Test negative step (reverse)
    rev_sliced = result[::-1]
    assert rev_sliced.shape == (10, 8), f"Expected shape (10, 8), got {rev_sliced.shape}"
    np.testing.assert_allclose(np.asarray(rev_sliced), expected[::-1], rtol=1e-10)

    # Test slice from end
    end_sliced = result[-3:]
    assert end_sliced.shape == (3, 8), f"Expected shape (3, 8), got {end_sliced.shape}"
    np.testing.assert_allclose(np.asarray(end_sliced), expected[-3:], rtol=1e-10)


def test_distances_tensor_zero_copy_views():
    """Tests that slicing, transpose, reshape, and iteration return zero-copy views."""
    np.random.seed(42)

    a = np.random.rand(5, 64).astype(np.float64)
    b = np.random.rand(4, 64).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")
    orig_np = np.asarray(result)

    # Test 1: Basic slicing shares memory
    sliced = result[1:4]
    sliced_np = np.asarray(sliced)
    assert np.shares_memory(orig_np, sliced_np), "Basic slice should share memory with original"

    # Test 2: Slice with step shares memory
    step_sliced = result[::2]
    step_np = np.asarray(step_sliced)
    assert np.shares_memory(orig_np, step_np), "Step slice should share memory with original"

    # Test 3: Reverse slice shares memory
    rev_sliced = result[::-1]
    rev_np = np.asarray(rev_sliced)
    assert np.shares_memory(orig_np, rev_np), "Reverse slice should share memory with original"

    # Test 4: Sub-tensor indexing (tensor[i]) shares memory
    row = result[0]
    row_np = np.asarray(row)
    assert np.shares_memory(orig_np, row_np), "Row indexing should share memory with original"

    # Test 5: Transpose shares memory
    transposed = result.T
    trans_np = np.asarray(transposed)
    assert np.shares_memory(orig_np, trans_np), "Transpose should share memory with original"

    # Test 6: Reshape of contiguous tensor shares memory
    flat = result.reshape(20)
    flat_np = np.asarray(flat)
    assert np.shares_memory(orig_np, flat_np), "Reshape of contiguous tensor should share memory"

    # Test 7: Reshape of non-contiguous tensor does NOT share memory (must copy)
    trans_reshaped = transposed.reshape(20)
    trans_reshaped_np = np.asarray(trans_reshaped)
    assert not np.shares_memory(orig_np, trans_reshaped_np), "Reshape of non-contiguous should copy"

    # Test 8: Chained slicing shares memory
    chained = result[1:4][1:]
    chained_np = np.asarray(chained)
    assert np.shares_memory(orig_np, chained_np), "Chained slicing should share memory"

    # Test 9: Iteration yields views that share memory
    for i, row in enumerate(result):
        row_np = np.asarray(row)
        assert np.shares_memory(orig_np, row_np), f"Iteration row {i} should share memory"

    # Test 10: Modification through view updates original
    orig_val = orig_np[2, 0].copy()
    view = result[2]
    view_np = np.asarray(view)
    view_np[0] = 999.0
    assert orig_np[2, 0] == 999.0, "Modification through view should update original"
    orig_np[2, 0] = orig_val  # Restore

    # Test 11: copy() does NOT share memory
    copied = result.copy()
    copied_np = np.asarray(copied)
    assert not np.shares_memory(orig_np, copied_np), "copy() should NOT share memory"


# region: NDArray Constructor Tests


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
    [
        pytest.param((10,), id="1d-10"),
        pytest.param((5, 4), id="2d-5x4"),
        pytest.param((2, 3, 4), id="3d-2x3x4"),
    ],
)
def test_ndarray_zeros(dtype, shape):
    """Test nk.zeros() constructor."""
    arr = nk.zeros(shape, dtype=dtype)
    arr_np = np.asarray(arr)

    assert arr.shape == shape, f"Shape mismatch: {arr.shape} vs {shape}"
    assert arr.dtype == dtype, f"Dtype mismatch: {arr.dtype} vs {dtype}"
    assert np.all(arr_np == 0), "Array should be all zeros"


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
    [
        pytest.param((10,), id="1d-10"),
        pytest.param((5, 4), id="2d-5x4"),
        pytest.param((2, 3, 4), id="3d-2x3x4"),
    ],
)
def test_ndarray_ones(dtype, shape):
    """Test nk.ones() constructor."""
    arr = nk.ones(shape, dtype=dtype)
    arr_np = np.asarray(arr)

    assert arr.shape == shape, f"Shape mismatch: {arr.shape} vs {shape}"
    assert arr.dtype == dtype, f"Dtype mismatch: {arr.dtype} vs {dtype}"
    assert np.all(arr_np == 1), "Array should be all ones"


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
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((10,), id="1d-10"),
        pytest.param((5, 4), id="2d-5x4"),
    ],
)
def test_ndarray_full(dtype, fill_value, shape):
    """Test nk.full() constructor."""
    arr = nk.full(shape, fill_value, dtype=dtype)
    arr_np = np.asarray(arr)

    assert arr.shape == shape, f"Shape mismatch: {arr.shape} vs {shape}"
    assert arr.dtype == dtype, f"Dtype mismatch: {arr.dtype} vs {dtype}"

    # For floats, use approximate comparison; for ints, exact
    if dtype.startswith("float"):
        np.testing.assert_allclose(arr_np, fill_value, rtol=1e-5)
    else:
        expected = np.dtype(dtype).type(fill_value)
        assert np.all(arr_np == expected), f"Array should be all {expected}"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((10,), id="1d-10"),
        pytest.param((5, 4), id="2d-5x4"),
    ],
)
def test_ndarray_empty(dtype, shape):
    """Test nk.empty() constructor - just verify shape and dtype, not contents."""
    arr = nk.empty(shape, dtype=dtype)

    assert arr.shape == shape, f"Shape mismatch: {arr.shape} vs {shape}"
    assert arr.dtype == dtype, f"Dtype mismatch: {arr.dtype} vs {dtype}"


# endregion

# region: NDArray Reduction Tests


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
    [
        pytest.param((100,), id="1d-100"),
        pytest.param((10, 10), id="2d-10x10"),
        pytest.param((4, 5, 5), id="3d-4x5x5"),
    ],
)
def test_ndarray_sum_method(dtype, shape):
    """Test NDArray.sum() method."""
    np.random.seed(42)

    # Create NumPy array and numkong array
    if dtype.startswith("float"):
        np_arr = np.random.randn(*shape).astype(dtype)
    else:
        dtype_info = np.iinfo(np.dtype(dtype))
        np_arr = np.random.randint(dtype_info.min // 2, dtype_info.max // 2, size=shape, dtype=dtype)

    nk_arr = nk.zeros(shape, dtype=dtype)
    nk_arr_np = np.asarray(nk_arr)
    np.copyto(nk_arr_np, np_arr)

    expected = np_arr.sum()
    result = nk_arr.sum()

    if dtype.startswith("float"):
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
    else:
        assert result == expected, f"sum mismatch: {result} vs {expected}"


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
    [
        pytest.param((100,), id="1d-100"),
        pytest.param((10, 10), id="2d-10x10"),
    ],
)
def test_ndarray_min_max_methods(dtype, shape):
    """Test NDArray.min() and NDArray.max() methods."""
    np.random.seed(42)

    if dtype.startswith("float"):
        np_arr = np.random.randn(*shape).astype(dtype)
    else:
        dtype_info = np.iinfo(np.dtype(dtype))
        np_arr = np.random.randint(dtype_info.min // 2, dtype_info.max // 2, size=shape, dtype=dtype)

    nk_arr = nk.zeros(shape, dtype=dtype)
    nk_arr_np = np.asarray(nk_arr)
    np.copyto(nk_arr_np, np_arr)

    # Test min
    expected_min = np_arr.min()
    result_min = nk_arr.min()
    if dtype.startswith("float"):
        np.testing.assert_allclose(result_min, expected_min, rtol=1e-5)
    else:
        assert result_min == expected_min, f"min mismatch: {result_min} vs {expected_min}"

    # Test max
    expected_max = np_arr.max()
    result_max = nk_arr.max()
    if dtype.startswith("float"):
        np.testing.assert_allclose(result_max, expected_max, rtol=1e-5)
    else:
        assert result_max == expected_max, f"max mismatch: {result_max} vs {expected_max}"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("int32", id="i32"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((100,), id="1d-100"),
        pytest.param((10, 10), id="2d-10x10"),
    ],
)
def test_ndarray_argmin_argmax_methods(dtype, shape):
    """Test NDArray.argmin() and NDArray.argmax() methods."""
    np.random.seed(42)

    if dtype.startswith("float"):
        np_arr = np.random.randn(*shape).astype(dtype)
    else:
        dtype_info = np.iinfo(np.dtype(dtype))
        np_arr = np.random.randint(dtype_info.min // 2, dtype_info.max // 2, size=shape, dtype=dtype)

    nk_arr = nk.zeros(shape, dtype=dtype)
    nk_arr_np = np.asarray(nk_arr)
    np.copyto(nk_arr_np, np_arr)

    # Test argmin (flat index)
    expected_argmin = np_arr.argmin()
    result_argmin = nk_arr.argmin()
    assert result_argmin == expected_argmin, f"argmin mismatch: {result_argmin} vs {expected_argmin}"

    # Test argmax (flat index)
    expected_argmax = np_arr.argmax()
    result_argmax = nk_arr.argmax()
    assert result_argmax == expected_argmax, f"argmax mismatch: {result_argmax} vs {expected_argmax}"


# endregion

# region: Module-level Reduction Tests


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
def test_module_level_reductions(dtype):
    """Test nk.sum(), nk.min(), nk.max(), nk.argmin(), nk.argmax() module functions."""
    np.random.seed(42)
    shape = (50,)
    np_arr = np.random.randn(*shape).astype(dtype)

    nk_arr = nk.zeros(shape, dtype=dtype)
    nk_arr_np = np.asarray(nk_arr)
    np.copyto(nk_arr_np, np_arr)

    # Test nk.sum()
    np.testing.assert_allclose(nk.sum(nk_arr), np_arr.sum(), rtol=1e-4, atol=1e-4)

    # Test nk.min()
    np.testing.assert_allclose(nk.min(nk_arr), np_arr.min(), rtol=1e-5)

    # Test nk.max()
    np.testing.assert_allclose(nk.max(nk_arr), np_arr.max(), rtol=1e-5)

    # Test nk.argmin()
    assert nk.argmin(nk_arr) == np_arr.argmin()

    # Test nk.argmax()
    assert nk.argmax(nk_arr) == np_arr.argmax()


# endregion

# region: NDArray Arithmetic Operator Tests


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("int32", id="i32"),
    ],
)
def test_ndarray_add_operator(dtype):
    """Test NDArray + operator."""
    np.random.seed(42)
    shape = (20,)

    if dtype.startswith("float"):
        np_a = np.random.randn(*shape).astype(dtype)
        np_b = np.random.randn(*shape).astype(dtype)
    else:
        np_a = np.random.randint(-50, 50, size=shape, dtype=dtype)
        np_b = np.random.randint(-50, 50, size=shape, dtype=dtype)

    # Create numkong arrays
    nk_a = nk.zeros(shape, dtype=dtype)
    nk_b = nk.zeros(shape, dtype=dtype)
    np.copyto(np.asarray(nk_a), np_a)
    np.copyto(np.asarray(nk_b), np_b)

    # Test a + b
    expected = np_a + np_b
    result = nk_a + nk_b
    result_np = np.asarray(result)

    if dtype.startswith("float"):
        np.testing.assert_allclose(result_np, expected, rtol=1e-5)
    else:
        np.testing.assert_array_equal(result_np, expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("int32", id="i32"),
    ],
)
def test_ndarray_subtract_operator(dtype):
    """Test NDArray - operator."""
    np.random.seed(42)
    shape = (20,)

    if dtype.startswith("float"):
        np_a = np.random.randn(*shape).astype(dtype)
        np_b = np.random.randn(*shape).astype(dtype)
    else:
        np_a = np.random.randint(-50, 50, size=shape, dtype=dtype)
        np_b = np.random.randint(-50, 50, size=shape, dtype=dtype)

    nk_a = nk.zeros(shape, dtype=dtype)
    nk_b = nk.zeros(shape, dtype=dtype)
    np.copyto(np.asarray(nk_a), np_a)
    np.copyto(np.asarray(nk_b), np_b)

    # Test a - b
    expected = np_a - np_b
    result = nk_a - nk_b
    result_np = np.asarray(result)

    if dtype.startswith("float"):
        np.testing.assert_allclose(result_np, expected, rtol=1e-5)
    else:
        np.testing.assert_array_equal(result_np, expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("int32", id="i32"),
    ],
)
def test_ndarray_multiply_operator(dtype):
    """Test NDArray * operator."""
    np.random.seed(42)
    shape = (20,)

    if dtype.startswith("float"):
        np_a = np.random.randn(*shape).astype(dtype)
        np_b = np.random.randn(*shape).astype(dtype)
    else:
        np_a = np.random.randint(-10, 10, size=shape, dtype=dtype)
        np_b = np.random.randint(-10, 10, size=shape, dtype=dtype)

    nk_a = nk.zeros(shape, dtype=dtype)
    nk_b = nk.zeros(shape, dtype=dtype)
    np.copyto(np.asarray(nk_a), np_a)
    np.copyto(np.asarray(nk_b), np_b)

    # Test a * b
    expected = np_a * np_b
    result = nk_a * nk_b
    result_np = np.asarray(result)

    if dtype.startswith("float"):
        np.testing.assert_allclose(result_np, expected, rtol=1e-4)
    else:
        np.testing.assert_array_equal(result_np, expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("int32", id="i32"),
    ],
)
def test_ndarray_unary_operators(dtype):
    """Test NDArray unary - and + operators."""
    np.random.seed(42)
    shape = (20,)

    if dtype.startswith("float"):
        np_a = np.random.randn(*shape).astype(dtype)
    else:
        np_a = np.random.randint(-50, 50, size=shape, dtype=dtype)

    nk_a = nk.zeros(shape, dtype=dtype)
    np.copyto(np.asarray(nk_a), np_a)

    # Test -a (unary negation)
    expected_neg = -np_a
    result_neg = -nk_a
    result_neg_np = np.asarray(result_neg)

    if dtype.startswith("float"):
        np.testing.assert_allclose(result_neg_np, expected_neg, rtol=1e-5)
    else:
        np.testing.assert_array_equal(result_neg_np, expected_neg)

    # Test +a (unary positive - should return copy)
    result_pos = +nk_a
    result_pos_np = np.asarray(result_pos)
    np.testing.assert_array_equal(result_pos_np, np_a)


# endregion

# region: Strided/Non-contiguous Array Tests


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
def test_reduction_on_strided_array(dtype):
    """Test reductions on non-contiguous (strided) arrays."""
    np.random.seed(42)

    # Create a 2D array and take a strided slice
    np_arr = np.random.randn(10, 10).astype(dtype)
    nk_arr = nk.zeros((10, 10), dtype=dtype)
    np.copyto(np.asarray(nk_arr), np_arr)

    # Take every other row (strided view)
    np_strided = np_arr[::2]  # Shape (5, 10), strided
    nk_strided = nk_arr[::2]

    # Test sum on strided array
    expected_sum = np_strided.sum()
    result_sum = nk_strided.sum()
    np.testing.assert_allclose(result_sum, expected_sum, rtol=1e-4, atol=1e-4)

    # Test min/max on strided array
    np.testing.assert_allclose(nk_strided.min(), np_strided.min(), rtol=1e-5)
    np.testing.assert_allclose(nk_strided.max(), np_strided.max(), rtol=1e-5)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
def test_reduction_on_transposed_array(dtype):
    """Test reductions on transposed (non-contiguous) arrays."""
    np.random.seed(42)

    np_arr = np.random.randn(5, 8).astype(dtype)
    nk_arr = nk.zeros((5, 8), dtype=dtype)
    np.copyto(np.asarray(nk_arr), np_arr)

    # Transpose
    np_t = np_arr.T
    nk_t = nk_arr.T

    # Test sum on transposed array
    expected_sum = np_t.sum()
    result_sum = nk_t.sum()
    np.testing.assert_allclose(result_sum, expected_sum, rtol=1e-4, atol=1e-4)

    # Test min/max on transposed array
    np.testing.assert_allclose(nk_t.min(), np_t.min(), rtol=1e-5)
    np.testing.assert_allclose(nk_t.max(), np_t.max(), rtol=1e-5)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
def test_reduction_on_subview(dtype):
    """Test reductions on subviews (sliced arrays)."""
    np.random.seed(42)

    np_arr = np.random.randn(20, 20).astype(dtype)
    nk_arr = nk.zeros((20, 20), dtype=dtype)
    np.copyto(np.asarray(nk_arr), np_arr)

    # Take a subview
    np_sub = np_arr[5:15, 5:15]
    nk_sub = nk_arr[5:15, 5:15]

    # Test sum
    np.testing.assert_allclose(nk_sub.sum(), np_sub.sum(), rtol=1e-4, atol=1e-4)

    # Test min/max
    np.testing.assert_allclose(nk_sub.min(), np_sub.min(), rtol=1e-5)
    np.testing.assert_allclose(nk_sub.max(), np_sub.max(), rtol=1e-5)

    # Test argmin/argmax
    assert nk_sub.argmin() == np_sub.argmin()
    assert nk_sub.argmax() == np_sub.argmax()


# endregion

# region: Buffer Protocol Tests (NumPy arrays as input)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
def test_add_with_numpy_arrays(dtype):
    """Test nk.add() with NumPy arrays via buffer protocol."""
    np.random.seed(42)
    shape = (50,)
    a = np.random.randn(*shape).astype(dtype)
    b = np.random.randn(*shape).astype(dtype)

    expected = a + b
    result = nk.add(a, b)
    result_np = np.asarray(result)

    np.testing.assert_allclose(result_np, expected, rtol=1e-5)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
def test_multiply_with_numpy_arrays(dtype):
    """Test nk.multiply() with NumPy arrays via buffer protocol."""
    np.random.seed(42)
    shape = (50,)
    a = np.random.randn(*shape).astype(dtype)
    b = np.random.randn(*shape).astype(dtype)

    expected = a * b
    result = nk.multiply(a, b)
    result_np = np.asarray(result)

    np.testing.assert_allclose(result_np, expected, rtol=1e-4)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
def test_wsum_with_numpy_arrays(dtype):
    """Test nk.wsum() with NumPy arrays via buffer protocol."""
    np.random.seed(42)
    shape = (50,)
    a = np.random.randn(*shape).astype(dtype)
    b = np.random.randn(*shape).astype(dtype)
    alpha = 2.0
    beta = 0.5

    expected = alpha * a + beta * b
    result = nk.wsum(a, b, alpha=alpha, beta=beta)
    result_np = np.asarray(result)

    np.testing.assert_allclose(result_np, expected, rtol=1e-4)


# endregion


def test_bfloat16_scalar_creation():
    """Test bfloat16 scalar creation and conversion."""
    bf = nk.bfloat16(3.14159)
    assert isinstance(bf, nk.bfloat16)
    # bfloat16 has limited precision, so we allow some tolerance
    assert abs(float(bf) - 3.14159) < 0.01
    assert int(bf) == 3


def test_bfloat16_scalar_repr():
    """Test bfloat16 repr and str."""
    bf = nk.bfloat16(1.5)
    assert "bfloat16" in repr(bf)
    assert "1.5" in str(bf)


def test_bfloat16_scalar_arithmetic():
    """Test bfloat16 arithmetic operations."""
    a = nk.bfloat16(1.5)
    b = nk.bfloat16(2.5)

    # Addition
    result = a + b
    assert isinstance(result, nk.bfloat16)
    assert float(result) == 4.0

    # Subtraction
    result = b - a
    assert isinstance(result, nk.bfloat16)
    assert float(result) == 1.0

    # Multiplication
    result = a * b
    assert isinstance(result, nk.bfloat16)
    assert float(result) == 3.75

    # Division
    result = b / a
    assert isinstance(result, nk.bfloat16)
    assert abs(float(result) - 1.6666) < 0.01


def test_bfloat16_scalar_unary():
    """Test bfloat16 unary operations."""
    a = nk.bfloat16(1.5)
    assert float(-a) == -1.5
    assert float(+a) == 1.5
    assert float(abs(nk.bfloat16(-1.5))) == 1.5
    assert bool(a) == True
    assert bool(nk.bfloat16(0.0)) == False


def test_bfloat16_scalar_comparison():
    """Test bfloat16 comparison operations."""
    a = nk.bfloat16(1.5)
    b = nk.bfloat16(2.5)

    assert a < b
    assert a <= b
    assert a <= a
    assert b > a
    assert b >= a
    assert a == a
    assert a != b

    # Comparison with Python floats
    assert a == 1.5
    assert a < 2.0
    assert a > 1.0


def test_bfloat16_scalar_hash():
    """Test bfloat16 can be used in sets and dicts."""
    a = nk.bfloat16(1.5)
    b = nk.bfloat16(1.5)
    s = {a, b}
    assert len(s) == 1

    d = {a: "value"}
    assert d[b] == "value"


def test_float8_e4m3_scalar():
    """Test float8_e4m3 scalar type."""
    f8 = nk.float8_e4m3(1.5)
    assert isinstance(f8, nk.float8_e4m3)
    assert float(f8) == 1.5
    assert int(f8) == 1
    assert "float8_e4m3" in repr(f8)

    # Negation
    assert float(-f8) == -1.5

    # Boolean
    assert bool(f8) == True
    assert bool(nk.float8_e4m3(0.0)) == False


def test_float8_e4m3_scalar_arithmetic():
    """Test float8_e4m3 arithmetic operations."""
    a = nk.float8_e4m3(1.5)
    b = nk.float8_e4m3(2.0)

    # Addition
    result = a + b
    assert isinstance(result, nk.float8_e4m3)
    assert abs(float(result) - 3.5) < 0.5  # e4m3 has limited precision

    # Subtraction
    result = b - a
    assert isinstance(result, nk.float8_e4m3)
    assert abs(float(result) - 0.5) < 0.5

    # Multiplication
    result = a * b
    assert isinstance(result, nk.float8_e4m3)
    assert abs(float(result) - 3.0) < 0.5

    # Division
    result = b / a
    assert isinstance(result, nk.float8_e4m3)
    assert abs(float(result) - 1.33) < 0.5

    # Unary
    assert float(+a) == float(a)
    assert float(abs(nk.float8_e4m3(-1.5))) == 1.5


def test_float8_e5m2_scalar():
    """Test float8_e5m2 scalar type."""
    f8 = nk.float8_e5m2(1.5)
    assert isinstance(f8, nk.float8_e5m2)
    assert float(f8) == 1.5
    assert int(f8) == 1
    assert "float8_e5m2" in repr(f8)

    # Negation
    assert float(-f8) == -1.5

    # Boolean
    assert bool(f8) == True
    assert bool(nk.float8_e5m2(0.0)) == False


def test_float8_e5m2_scalar_arithmetic():
    """Test float8_e5m2 arithmetic operations."""
    a = nk.float8_e5m2(1.5)
    b = nk.float8_e5m2(2.0)

    # Addition
    result = a + b
    assert isinstance(result, nk.float8_e5m2)
    assert abs(float(result) - 3.5) < 0.5  # e5m2 has limited precision

    # Subtraction
    result = b - a
    assert isinstance(result, nk.float8_e5m2)
    assert abs(float(result) - 0.5) < 0.5

    # Multiplication
    result = a * b
    assert isinstance(result, nk.float8_e5m2)
    assert abs(float(result) - 3.0) < 0.5

    # Division
    result = b / a
    assert isinstance(result, nk.float8_e5m2)
    assert abs(float(result) - 1.33) < 0.5

    # Unary
    assert float(+a) == float(a)
    assert float(abs(nk.float8_e5m2(-1.5))) == 1.5


@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes not installed")
def test_bfloat16_vs_ml_dtypes():
    """Compare NumKong bfloat16 with ml_dtypes reference implementation."""
    test_values = [0.0, 1.0, -1.0, 3.14159, 100.0, 0.001, -65504.0]

    for val in test_values:
        nk_bf16 = nk.bfloat16(val)
        ml_bf16 = ml_dtypes.bfloat16(val)

        nk_float = float(nk_bf16)
        ml_float = float(ml_bf16)

        # Both implementations should produce the same result
        assert nk_float == ml_float, f"Mismatch for {val}: nk={nk_float}, ml={ml_float}"


@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes not installed")
def test_float8_e4m3_vs_ml_dtypes():
    """Compare NumKong float8_e4m3 with ml_dtypes reference implementation."""
    test_values = [0.0, 1.0, -1.0, 1.5, 2.0, 0.5]

    for val in test_values:
        nk_f8 = nk.float8_e4m3(val)
        ml_f8 = ml_dtypes.float8_e4m3fn(val)

        nk_float = float(nk_f8)
        ml_float = float(ml_f8)

        # Both implementations should produce the same result
        assert nk_float == ml_float, f"Mismatch for {val}: nk={nk_float}, ml={ml_float}"


@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes not installed")
def test_float8_e5m2_vs_ml_dtypes():
    """Compare NumKong float8_e5m2 with ml_dtypes reference implementation."""
    test_values = [0.0, 1.0, -1.0, 1.5, 2.0, 0.5]

    for val in test_values:
        nk_f8 = nk.float8_e5m2(val)
        ml_f8 = ml_dtypes.float8_e5m2(val)

        nk_float = float(nk_f8)
        ml_float = float(ml_f8)

        # Both implementations should produce the same result
        assert nk_float == ml_float, f"Mismatch for {val}: nk={nk_float}, ml={ml_float}"


if __name__ == "__main__":
    pytest.main(
        [
            "-s",  # Print stdout
            "-x",  # Stop on first failure
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
        ]
    )
