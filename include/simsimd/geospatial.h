/**
 *  @file       geospatial.h
 *  @brief      SIMD-accelerated Geo-Spatial distance functions.
 *  @author     Ash Vardanian
 *  @date       July 1, 2023
 *
 *  Contains:
 *  - Haversine (Great Circle) distance for 2 points
 *  - Haversine (Great Circle) distance for 2 arrays of points
 *  - Vincenty's distance function for Oblate Spheroid Geodesics
 *  All outputs are in meters, and the input coordinates are in degrees.
 *
 *  For datatypes:
 *  - 32-bit IEEE-754 floating point
 *  - 64-bit IEEE-754 floating point
 *
 *  For hardware architectures:
 *  - Arm: NEON
 *  - x86: Haswell, Skylake
 *
 *  @section    Low-Accuracy High-Performance Haversine Similarity
 *
 *  In most cases, for distance computations, we don't need the exact Haversine formula.
 *  The very last part of the computation applies `asin(sqrt(x))` non-linear transformation.
 *  Both `asin` and `sqrt` are monotonically increasing functions, so their product is also
 *  monotonically increasing. This means, for relative similarity/closeness computation we
 *  can avoid that expensive last step.
 *
 *  @section    Trigonometric Approximations & SIMD Vectorization
 *
 *
 *
 *  @section    High-Precision Vincenty's Formulae & Earth Ellipsoid
 *
 *  Several approximations of the Earth Ellipsoid exist, each defined by the Equatorial radius (m),
 *  Polar radius (m), and Inverse flattening. The earliest ones date back to 1738, when Pierre Louis
 *  Maupertuis in France suggested a shape, that is only 0.3% different from the most accurate modern
 *  estimates by the International Earth Rotation and Reference Systems Service (IERS).
 *  The Global Positioning System (GPS) uses the World Geodetic Systems's (WGS) WGS-84 standard.
 *  SimSIMD uses the newer & more accurate @b IERS-2003 standard, but allows overriding default parameters:
 *
 *      #define SIMSIMD_EARTH_ELLIPSOID_EQUATORIAL_RADIUS (6378136.6)
 *      #define SIMSIMD_EARTH_ELLIPSOID_POLAR_RADIUS (6356751.9)
 *      #define SIMSIMD_EARTH_ELLIPSOID_INVERSE_FLATTENING (298.25642)
 *
 *  To revert from oblate spheroids to spheres, use:
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  Earth Ellipsoid: https://en.wikipedia.org/wiki/Earth_ellipsoid
 *  Oblate Spheroid Geodesic: https://mathworld.wolfram.com/OblateSpheroidGeodesic.html
 *  Staging experiments: https://github.com/ashvardanian/HaversineMathKong
 *  Speeding up atan2f by 50x: https://mazzo.li/posts/vectorized-atan2.html
 *  Simplifying the GNU C Sine Function: https://www.awelm.com/posts/simplifying-the-gnu-c-sine-function/
 */
#ifndef SIMSIMD_GEOSPATIAL_H
#define SIMSIMD_GEOSPATIAL_H

#include "types.h"

#include "trigonometry.h"

/*  Earth Ellipsoid Constants
 *  The default values use the IERS-2003 standard, but can be overridden before including this header.
 */
#ifndef SIMSIMD_EARTH_MEDIATORIAL_RADIUS
#define SIMSIMD_EARTH_MEDIATORIAL_RADIUS (6335439.0)
#endif
#ifndef SIMSIMD_EARTH_ELLIPSOID_EQUATORIAL_RADIUS
#define SIMSIMD_EARTH_ELLIPSOID_EQUATORIAL_RADIUS (6378136.6)
#endif
#ifndef SIMSIMD_EARTH_ELLIPSOID_POLAR_RADIUS
#define SIMSIMD_EARTH_ELLIPSOID_POLAR_RADIUS (6356751.9)
#endif
#ifndef SIMSIMD_EARTH_ELLIPSOID_INVERSE_FLATTENING
#define SIMSIMD_EARTH_ELLIPSOID_INVERSE_FLATTENING (298.25642)
#endif
#ifndef SIMSIMD_VINCENTY_MAX_ITERATIONS
#define SIMSIMD_VINCENTY_MAX_ITERATIONS 100
#endif
#ifndef SIMSIMD_VINCENTY_CONVERGENCE_THRESHOLD
#define SIMSIMD_VINCENTY_CONVERGENCE_THRESHOLD 1e-12
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*  Serial backends for all numeric types.
 *  By default they use 32-bit arithmetic, unless the arguments themselves contain 64-bit floats.
 *  For double-precision computation check out the "*_accurate" variants of those "*_serial" functions.
 */
SIMSIMD_PUBLIC void simsimd_haversine_f64_serial(             //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);
SIMSIMD_PUBLIC void simsimd_vincenty_f64_serial(              //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);
SIMSIMD_PUBLIC void simsimd_haversine_f32_serial(             //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);
SIMSIMD_PUBLIC void simsimd_vincenty_f32_serial(              //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);

SIMSIMD_PUBLIC void simsimd_haversine_f64_neon(               //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);
SIMSIMD_PUBLIC void simsimd_vincenty_f64_neon(                //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);
SIMSIMD_PUBLIC void simsimd_haversine_f32_neon(               //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);
SIMSIMD_PUBLIC void simsimd_vincenty_f32_neon(                //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic over 256-bit words.
 *  First demonstrated in 2011, at least one Haswell-based processor was still being sold in 2022 — the Pentium G3420.
 *  Practically all modern x86 CPUs support AVX2, FMA, and F16C, making it a perfect baseline for SIMD algorithms.
 *  On other hand, there is no need to implement AVX2 versions of `f32` and `f64` functions, as those are
 *  properly vectorized by recent compilers.
 */
SIMSIMD_PUBLIC void simsimd_haversine_f64_haswell(            //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);
SIMSIMD_PUBLIC void simsimd_vincenty_f64_haswell(             //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);
SIMSIMD_PUBLIC void simsimd_haversine_f32_haswell(            //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);
SIMSIMD_PUBLIC void simsimd_vincenty_f32_haswell(             //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);

/*  SIMD-powered backends for various generations of AVX512 CPUs.
 *  Skylake is handy, as it supports masked loads and other operations, avoiding the need for the tail loop.
 */
SIMSIMD_PUBLIC void simsimd_haversine_f64_skylake(            //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);
SIMSIMD_PUBLIC void simsimd_vincenty_f64_skylake(             //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);
SIMSIMD_PUBLIC void simsimd_haversine_f32_skylake(            //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);
SIMSIMD_PUBLIC void simsimd_vincenty_f32_skylake(             //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL
#endif // _SIMSIMD_TARGET_X86

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512dq", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512dq,bmi2"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_haversine_f64_skylake(            //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);
SIMSIMD_PUBLIC void simsimd_vincenty_f64_skylake(             //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);

SIMSIMD_INTERNAL __m512 _simsimd_haversine_f32x16_skylake(__m512 lat1, __m512 lon1, __m512 lat2, __m512 lon2) {
    __m512 const deg_to_rad = _mm512_set1_ps(3.14159265358979323846f / 180.0f); /// π / 180
    __m512 const earth_radius = _mm512_set1_ps(6335439);                        /// Mediatorial Earth Radius in meters

    // Convert degrees to radians
    lat1 = _mm512_mul_ps(lat1, deg_to_rad);
    lon1 = _mm512_mul_ps(lon1, deg_to_rad);
    lat2 = _mm512_mul_ps(lat2, deg_to_rad);
    lon2 = _mm512_mul_ps(lon2, deg_to_rad);

    // Compute delta latitudes and longitudes
    __m512 dlat = _mm512_sub_ps(lat2, lat1);
    __m512 dlon = _mm512_sub_ps(lon2, lon1);

    // Compute sin^2(dlat/2) and sin^2(dlon/2)
    __m512 dlat_half = _mm512_mul_ps(dlat, _mm512_set1_ps(0.5f));
    __m512 dlon_half = _mm512_mul_ps(dlon, _mm512_set1_ps(0.5f));

    __m512 sin_dlat_half = _simsimd_f32x16_sin_skylake(dlat_half);
    __m512 sin_dlon_half = _simsimd_f32x16_sin_skylake(dlon_half);

    __m512 sin2_dlat_half = _mm512_mul_ps(sin_dlat_half, sin_dlat_half);
    __m512 sin2_dlon_half = _mm512_mul_ps(sin_dlon_half, sin_dlon_half);

    // Compute cos(lat1) * cos(lat2)
    __m512 cos_lat1 = _simsimd_f32x16_cos_skylake(lat1);
    __m512 cos_lat2 = _simsimd_f32x16_cos_skylake(lat2);
    __m512 cos_lat1_lat2 = _mm512_mul_ps(cos_lat1, cos_lat2);

    // Compute a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
    __m512 a = _mm512_add_ps(sin2_dlat_half, _mm512_mul_ps(cos_lat1_lat2, sin2_dlon_half));

    // Compute c = 2 * atan2(sqrt(a), sqrt(1-a))
    __m512 sqrt_a = _mm512_sqrt_ps(a);
    __m512 sqrt_1_minus_a = _mm512_sqrt_ps(_mm512_sub_ps(_mm512_set1_ps(1.0f), a));
    __m512 c = _mm512_mul_ps(_mm512_set1_ps(2.0f), _simsimd_f32x16_atan2_skylake(sqrt_a, sqrt_1_minus_a));

    // Compute the distance: d = R * c
    __m512 distances = _mm512_mul_ps(earth_radius, c);
}

SIMSIMD_PUBLIC void simsimd_haversine_f32_skylake(            //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results) {

    __m512 lat1, lon1, lat2, lon2;

simsimd_haversine_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        lat1 = _mm512_maskz_loadu_ps(mask, a_lats);
        lon1 = _mm512_maskz_loadu_ps(mask, a_lons);
        lat2 = _mm512_maskz_loadu_ps(mask, b_lats);
        lon2 = _mm512_maskz_loadu_ps(mask, b_lons);
        n = 0;
    }
    else {
        lat1 = _mm512_loadu_ps(a_lats);
        lon1 = _mm512_loadu_ps(a_lons);
        lat2 = _mm512_loadu_ps(b_lats);
        lon2 = _mm512_loadu_ps(b_lons);
        a_lats += 16, a_lons += 16, b_lats += 16, b_lons += 16, n -= 16;
    }
}

SIMSIMD_PUBLIC void simsimd_vincenty_f32_skylake(             //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results);

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE

#ifdef __cplusplus
}
#endif

#endif
