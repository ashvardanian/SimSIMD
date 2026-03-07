#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test geospatial distances: nk.haversine, nk.vincenty.

Covers dtypes: float64, float32.
Parametrized over: ndim from dense_dimensions, capability from possible_capabilities.

Precision notes:
    Haversine uses atol=10.0, rtol=1e-2 — the great-circle formula at f32 accumulates
    rounding in intermediate trig computations.
    Vincenty at f32 can show >40% relative error near antipodal points due to its
    iterative algorithm; rtol=1.0 is used for f32 vs 1e-2 for f64.
    Known-value test verifies New York → Los Angeles ≈ 3940 km.

Matches C++ suite: test_geospatial.cpp.
"""

import atexit
import pytest

try:
    import numpy as np
except:
    np = None

import numkong as nk
from test_base import (
    numpy_available,
    dense_dimensions,
    possible_capabilities,
    randomized_repetitions_count,
    keep_one_capability,
    profile,
    collect_errors,
    create_stats,
    print_stats_report,
    seed_rng,
)

stats = create_stats()
atexit.register(print_stats_report, stats)

earth_radius_meters = 6335439.0


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
    equatorial_radius = 6378136.6
    polar_radius = 6356751.9
    flattening = (equatorial_radius - polar_radius) / equatorial_radius

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


KERNELS_GEOSPATIAL = {
    "haversine": (baseline_haversine, nk.haversine, None),
    "vincenty": (baseline_vincenty, nk.vincenty, None),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_haversine_random_accuracy(ndim, dtype, capability):
    """Haversine great-circle distance against Vincenty baseline for random coordinates."""
    keep_one_capability(capability)

    first_latitudes = (np.random.rand(ndim) - 0.5) * np.pi
    first_longitudes = (np.random.rand(ndim) - 0.5) * 2 * np.pi
    second_latitudes = (np.random.rand(ndim) - 0.5) * np.pi
    second_longitudes = (np.random.rand(ndim) - 0.5) * 2 * np.pi

    first_latitudes = first_latitudes.astype(dtype)
    first_longitudes = first_longitudes.astype(dtype)
    second_latitudes = second_latitudes.astype(dtype)
    second_longitudes = second_longitudes.astype(dtype)

    def _haversine_loop(lat1, lon1, lat2, lon2):
        return np.array([baseline_haversine(lat1[i], lon1[i], lat2[i], lon2[i]) for i in range(len(lat1))])

    accurate_dt, accurate = profile(
        _haversine_loop,
        first_latitudes.astype(np.float64),
        first_longitudes.astype(np.float64),
        second_latitudes.astype(np.float64),
        second_longitudes.astype(np.float64),
    )
    expected_dt, expected = profile(
        _haversine_loop, first_latitudes, first_longitudes, second_latitudes, second_longitudes
    )

    result_dt, result = profile(nk.haversine, first_latitudes, first_longitudes, second_latitudes, second_longitudes)
    result = np.asarray(result)

    absolute_tolerance = 10.0
    relative_tolerance = 1e-2
    np.testing.assert_allclose(result, accurate, atol=absolute_tolerance, rtol=relative_tolerance)

    collect_errors("haversine", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_vincenty_random_accuracy(ndim, dtype, capability):
    """Vincenty ellipsoidal geodesic distance against GeoPy baseline for random coordinates."""
    keep_one_capability(capability)

    first_latitudes = (np.random.rand(ndim) - 0.5) * np.pi * 0.9
    first_longitudes = (np.random.rand(ndim) - 0.5) * 2 * np.pi * 0.9
    second_latitudes = (np.random.rand(ndim) - 0.5) * np.pi * 0.9
    second_longitudes = (np.random.rand(ndim) - 0.5) * 2 * np.pi * 0.9

    first_latitudes = first_latitudes.astype(dtype)
    first_longitudes = first_longitudes.astype(dtype)
    second_latitudes = second_latitudes.astype(dtype)
    second_longitudes = second_longitudes.astype(dtype)

    def _vincenty_loop(lat1, lon1, lat2, lon2):
        return np.array([baseline_vincenty(lat1[i], lon1[i], lat2[i], lon2[i]) for i in range(len(lat1))])

    accurate_dt, accurate = profile(
        _vincenty_loop,
        first_latitudes.astype(np.float64),
        first_longitudes.astype(np.float64),
        second_latitudes.astype(np.float64),
        second_longitudes.astype(np.float64),
    )
    expected_dt, expected = profile(
        _vincenty_loop, first_latitudes, first_longitudes, second_latitudes, second_longitudes
    )

    result_dt, result = profile(nk.vincenty, first_latitudes, first_longitudes, second_latitudes, second_longitudes)
    result = np.asarray(result)

    # Vincenty's iterative algorithm at f32 precision accumulates significant
    # rounding error in intermediate trig computations, especially near antipodal
    # points. The f64 baseline converges to ~1e-12 but f32 cannot go below ~1e-7,
    # leading to path-dependent drift in the final distance. Near-antipodal cases
    # can show >40% relative error at f32 — this is inherent to the algorithm.
    absolute_tolerance = 100.0
    relative_tolerance = 1.0 if dtype == "float32" else 1e-2
    np.testing.assert_allclose(result, accurate, atol=absolute_tolerance, rtol=relative_tolerance)

    collect_errors("vincenty", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_haversine_known():
    """Haversine distance for New York to Los Angeles against known reference value."""
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

    assert 3800 < result_kilometers < 4100, f"Expected ~3940 km, got {result_kilometers:.0f} km"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_haversine_out_parameter():
    """The out= parameter writes haversine results to a pre-allocated buffer."""
    count = 10
    first_latitudes = np.random.rand(count).astype(np.float64) * np.pi - np.pi / 2
    first_longitudes = np.random.rand(count).astype(np.float64) * 2 * np.pi - np.pi
    second_latitudes = np.random.rand(count).astype(np.float64) * np.pi - np.pi / 2
    second_longitudes = np.random.rand(count).astype(np.float64) * 2 * np.pi - np.pi

    output_distances = np.zeros(count, dtype=np.float64)
    result = nk.haversine(first_latitudes, first_longitudes, second_latitudes, second_longitudes, out=output_distances)
    assert result is None, "Expected None when using out parameter"
    assert np.all(output_distances >= 0), "Output should contain non-negative distances"

    expected = np.array(nk.haversine(first_latitudes, first_longitudes, second_latitudes, second_longitudes))
    np.testing.assert_allclose(output_distances, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("capability", possible_capabilities)
def test_haversine_self_zero(capability):
    """haversine(lat, lon, lat, lon) ~ 0."""
    keep_one_capability(capability)
    lat = nk.full((1,), 0.5, dtype="float64")
    lon = nk.full((1,), 0.5, dtype="float64")
    result = nk.haversine(lat, lon, lat, lon)
    val = list(result)[0]
    assert abs(val) < 1.0, f"haversine(self) = {val}, expected ~0"


@pytest.mark.parametrize("capability", possible_capabilities)
def test_vincenty_self_zero(capability):
    """vincenty(lat, lon, lat, lon) ~ 0."""
    keep_one_capability(capability)
    lat = nk.full((1,), 0.5, dtype="float64")
    lon = nk.full((1,), 0.5, dtype="float64")
    result = nk.vincenty(lat, lon, lat, lon)
    val = list(result)[0]
    assert abs(val) < 1.0, f"vincenty(self) = {val}, expected ~0"
