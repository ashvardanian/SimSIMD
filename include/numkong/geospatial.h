/**
 *  @brief SIMD-accelerated Geo-Spatial distance functions.
 *  @file include/numkong/geospatial.h
 *  @author Ash Vardanian
 *  @date July 1, 2023
 *
 *  Contains following distance functions:
 *
 *  - Haversine (Great Circle) distance for 2 points
 *  - Haversine (Great Circle) distance for 2 arrays of points
 *  - Vincenty's distance function for Oblate Spheroid Geodesics
 *
 *  All outputs are in meters, and the input coordinates are in radians.
 *
 *  For dtypes:
 *
 *  - 64-bit IEEE-754 floating point → 64-bit
 *  - 32-bit IEEE-754 floating point → 32-bit
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON
 *  - x86: Haswell, Skylake
 *
 *  @section haversine_similarity Low-Accuracy High-Performance Haversine Similarity
 *
 *  In most cases, for distance computations, we don't need the exact Haversine formula.
 *  The very last part of the computation applies `asin(√x)` non-linear transformation.
 *  Both `asin` and `sqrt` are monotonically increasing functions, so their product is also
 *  monotonically increasing. This means, for relative similarity/closeness computation we
 *  can avoid that expensive last step.
 *
 *  @section trig_approximations Trigonometric Approximations & SIMD Vectorization
 *
 *  The trigonometric functions (sin, cos, atan2) use polynomial approximations with SLEEF-level
 *  error bounds (~3.5 ULP). For f64, this translates to ~1e-15 absolute error; for f32, ~1e-7.
 *
 *  @section accuracy_comparison Accuracy Comparison: Haversine vs Vincenty
 *
 *  Both algorithms compute geodesic distances, but with different Earth models:
 *
 *  - Haversine: Sphere (R=6335km), 0.3% - 0.6% vs WGS-84, fast approximation, ranking
 *  - Vincenty: WGS-84 Ellipsoid, 0.01% - 0.2% vs WGS-84, high-precision navigation
 *
 *  Vincenty is ~3-20x more accurate than Haversine for most routes. The improvement is most
 *  significant for long-distance routes and near-polar paths where Earth's oblateness matters.
 *
 *  @note   SIMD implementations may have slightly different results than serial due to
 *          floating-point ordering in iterative algorithms. For Vincenty, expect <0.001%
 *          difference between SIMD and serial implementations.
 *
 *  @section vincenty_precision High-Precision Vincenty's Formulae & Earth Ellipsoid
 *
 *  Several approximations of the Earth Ellipsoid exist, each defined by the Equatorial radius (m),
 *  Polar radius (m), and Inverse flattening. The earliest ones date back to 1738, when Pierre Louis
 *  Maupertuis in France suggested a shape, that is only 0.3% different from the most accurate modern
 *  estimates by the International Earth Rotation and Reference Systems Service (IERS).
 *  The Global Positioning System (GPS) uses the World Geodetic Systems's (WGS) WGS-84 standard.
 *  NumKong uses the newer & more accurate @b IERS-2003 standard, but allows overriding default parameters:
 *
 *      #define NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS (6378136.6)
 *      #define NK_EARTH_ELLIPSOID_POLAR_RADIUS (6356751.9)
 *      #define NK_EARTH_ELLIPSOID_INVERSE_FLATTENING (298.25642)
 *
 *  To revert from oblate spheroids to spheres, use `NK_EARTH_MEDIATORIAL_RADIUS`.
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  Haversine and Vincenty formulas require sqrt for the final distance calculation and division
 *  for Vincenty's iterative convergence. These are the most expensive operations (12-23 cycles)
 *  but only execute once per point-pair. The polynomial trig approximations use FMA chains.
 *  Note: ZMM sqrt is faster on Genoa (15c) than Ice Lake (19c) due to better 512-bit support.
 *
 *      Intrinsic               Instruction                     Ice         Genoa
 *      _mm256_sqrt_ps          VSQRTPS (YMM, YMM)              12c @ p0    15c @ p01
 *      _mm256_sqrt_pd          VSQRTPD (YMM, YMM)              13c @ p0    21c @ p01
 *      _mm512_sqrt_ps          VSQRTPS (ZMM, ZMM)              19c @ p05   15c @ p01
 *      _mm512_sqrt_pd          VSQRTPD (ZMM, ZMM)              23c @ p05   21c @ p01
 *      _mm256_div_ps           VDIVPS (YMM, YMM, YMM)          11c @ p0    11c @ p01
 *      _mm256_div_pd           VDIVPD (YMM, YMM, YMM)          13c @ p0    13c @ p01
 *      _mm256_fmadd_ps         VFMADD231PS (YMM, YMM, YMM)     4c @ p01    4c @ p01
 *      _mm256_fmadd_pd         VFMADD231PD (YMM, YMM, YMM)     4c @ p01    4c @ p01
 *
 *  @section arm_instructions Relevant ARM NEON/SVE Instructions
 *
 *  ARM sqrt (FSQRT) has low throughput as it uses a dedicated V02 execution unit. This is
 *  acceptable since sqrt only appears once per distance calculation. FMA chains for trig
 *  polynomial evaluation pipeline well across all 4 V-units.
 *
 *      Intrinsic               Instruction     M1 Firestorm    Graviton 3      Graviton 4
 *      vfmaq_f32               FMLA.S (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vfmaq_f64               FMLA.D (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vsqrtq_f32              FSQRT.S (vec)   10c @ V02       10c @ V02       9c @ V02
 *      vsqrtq_f64              FSQRT.D (vec)   13c @ V02       16c @ V02       16c @ V02
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  - Earth Ellipsoid: https://en.wikipedia.org/wiki/Earth_ellipsoid
 *  - Oblate Spheroid Geodesic: https://mathworld.wolfram.com/OblateSpheroidGeodesic.html
 *  - Staging experiments: https://github.com/ashvardanian/HaversineMathKong
 *  - Speeding up atan2f by 50x: https://mazzo.li/posts/vectorized-atan2.html
 *  - Simplifying the GNU C Sine Function: https://www.awelm.com/posts/simplifying-the-gnu-c-sine-function/
 *
 */
#ifndef NK_GEOSPATIAL_H
#define NK_GEOSPATIAL_H

#include "numkong/types.h"
#include "numkong/trigonometry.h"

/*  Earth Ellipsoid Constants
 *  The default values use the IERS-2003 standard, but can be overridden before including this header.
 */
#ifndef NK_EARTH_MEDIATORIAL_RADIUS
#define NK_EARTH_MEDIATORIAL_RADIUS (6335439.0)
#endif
#ifndef NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS
#define NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS (6378136.6)
#endif
#ifndef NK_EARTH_ELLIPSOID_POLAR_RADIUS
#define NK_EARTH_ELLIPSOID_POLAR_RADIUS (6356751.9)
#endif
#ifndef NK_EARTH_ELLIPSOID_INVERSE_FLATTENING
#define NK_EARTH_ELLIPSOID_INVERSE_FLATTENING (298.25642)
#endif
#ifndef NK_VINCENTY_MAX_ITERATIONS
#define NK_VINCENTY_MAX_ITERATIONS 100
#endif
#ifndef NK_VINCENTY_CONVERGENCE_THRESHOLD
#define NK_VINCENTY_CONVERGENCE_THRESHOLD 1e-12
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Haversine distance between two arrays of points on a sphere.
 *
 *  @param[in] a_lats Latitudes of the first points, in radians.
 *  @param[in] a_lons Longitudes of the first points, in radians.
 *  @param[in] b_lats Latitudes of the second points, in radians.
 *  @param[in] b_lons Longitudes of the second points, in radians.
 *  @param[in] n The number of point pairs.
 *  @param[out] results Output distances in meters, length `n`.
 *
 *  @note Inputs are in radians and outputs are in meters.
 */
NK_DYNAMIC void nk_haversine_f64(                   //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);

/** @copydoc nk_haversine_f64 */
NK_DYNAMIC void nk_haversine_f32(                   //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);

/**
 *  @brief Vincenty distance between two arrays of points on an oblate spheroid.
 *
 *  @param[in] a_lats Latitudes of the first points, in radians.
 *  @param[in] a_lons Longitudes of the first points, in radians.
 *  @param[in] b_lats Latitudes of the second points, in radians.
 *  @param[in] b_lons Longitudes of the second points, in radians.
 *  @param[in] n The number of point pairs.
 *  @param[out] results Output distances in meters, length `n`.
 *
 *  @note Inputs are in radians and outputs are in meters.
 *  @note Uses the Earth ellipsoid parameters configured via `NK_EARTH_ELLIPSOID_*`.
 */
NK_DYNAMIC void nk_vincenty_f64(                    //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);

/** @copydoc nk_vincenty_f64 */
NK_DYNAMIC void nk_vincenty_f32(                    //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);

/** @copydoc nk_haversine_f64 */
NK_PUBLIC void nk_haversine_f64_serial(             //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_vincenty_f64 */
NK_PUBLIC void nk_vincenty_f64_serial(              //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_haversine_f32 */
NK_PUBLIC void nk_haversine_f32_serial(             //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
/** @copydoc nk_vincenty_f32 */
NK_PUBLIC void nk_vincenty_f32_serial(              //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);

#if NK_TARGET_NEON
/** @copydoc nk_haversine_f64 */
NK_PUBLIC void nk_haversine_f64_neon(               //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_vincenty_f64 */
NK_PUBLIC void nk_vincenty_f64_neon(                //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_haversine_f32 */
NK_PUBLIC void nk_haversine_f32_neon(               //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
/** @copydoc nk_vincenty_f32 */
NK_PUBLIC void nk_vincenty_f32_neon(                //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
/** @copydoc nk_haversine_f64 */
NK_PUBLIC void nk_haversine_f64_haswell(            //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_vincenty_f64 */
NK_PUBLIC void nk_vincenty_f64_haswell(             //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_haversine_f32 */
NK_PUBLIC void nk_haversine_f32_haswell(            //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
/** @copydoc nk_vincenty_f32 */
NK_PUBLIC void nk_vincenty_f32_haswell(             //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_haversine_f64 */
NK_PUBLIC void nk_haversine_f64_skylake(            //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_vincenty_f64 */
NK_PUBLIC void nk_vincenty_f64_skylake(             //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_haversine_f32 */
NK_PUBLIC void nk_haversine_f32_skylake(            //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
/** @copydoc nk_vincenty_f32 */
NK_PUBLIC void nk_vincenty_f32_skylake(             //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
#endif // NK_TARGET_SKYLAKE

/*  Serial implementations of geospatial distance functions.
 *  These use the trigonometric functions from trigonometry.h for sin, cos, and atan2.
 */

NK_PUBLIC void nk_haversine_f64_serial(             //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    nk_f64_t const earth_radius = NK_EARTH_MEDIATORIAL_RADIUS;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t first_latitude = a_lats[i];
        nk_f64_t first_longitude = a_lons[i];
        nk_f64_t second_latitude = b_lats[i];
        nk_f64_t second_longitude = b_lons[i];

        nk_f64_t latitude_delta = second_latitude - first_latitude;
        nk_f64_t longitude_delta = second_longitude - first_longitude;

        // Haversine formula: a = sin²(Δlat/2) + cos(lat1)×cos(lat2)×sin²(Δlon/2)
        nk_f64_t sin_latitude_delta_half = nk_f64_sin(latitude_delta * 0.5);
        nk_f64_t sin_longitude_delta_half = nk_f64_sin(longitude_delta * 0.5);
        nk_f64_t cos_first_latitude = nk_f64_cos(first_latitude);
        nk_f64_t cos_second_latitude = nk_f64_cos(second_latitude);

        nk_f64_t haversine_term = sin_latitude_delta_half * sin_latitude_delta_half +
                                  cos_first_latitude * cos_second_latitude * sin_longitude_delta_half *
                                      sin_longitude_delta_half;
        // Clamp haversine_term to [0, 1] to prevent NaN from sqrt of negative values
        haversine_term = (haversine_term < 0.0) ? 0.0 : ((haversine_term > 1.0) ? 1.0 : haversine_term);

        // Central angle: c = 2 × atan2(√a, √(1-a))
        nk_f64_t sqrt_haversine = nk_f64_sqrt_serial(haversine_term);
        nk_f64_t sqrt_complement = nk_f64_sqrt_serial(1.0 - haversine_term);
        nk_f64_t central_angle = 2.0 * nk_f64_atan2(sqrt_haversine, sqrt_complement);

        results[i] = earth_radius * central_angle;
    }
}

NK_PUBLIC void nk_haversine_f32_serial(             //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    nk_f32_t const earth_radius = (nk_f32_t)NK_EARTH_MEDIATORIAL_RADIUS;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t first_latitude = a_lats[i];
        nk_f32_t first_longitude = a_lons[i];
        nk_f32_t second_latitude = b_lats[i];
        nk_f32_t second_longitude = b_lons[i];

        nk_f32_t latitude_delta = second_latitude - first_latitude;
        nk_f32_t longitude_delta = second_longitude - first_longitude;

        // Haversine formula: a = sin²(Δlat/2) + cos(lat1)×cos(lat2)×sin²(Δlon/2)
        nk_f32_t sin_latitude_delta_half = nk_f32_sin(latitude_delta * 0.5f);
        nk_f32_t sin_longitude_delta_half = nk_f32_sin(longitude_delta * 0.5f);
        nk_f32_t cos_first_latitude = nk_f32_cos(first_latitude);
        nk_f32_t cos_second_latitude = nk_f32_cos(second_latitude);

        nk_f32_t haversine_term = sin_latitude_delta_half * sin_latitude_delta_half +
                                  cos_first_latitude * cos_second_latitude * sin_longitude_delta_half *
                                      sin_longitude_delta_half;

        // Clamp to [0, 1] to avoid NaN from sqrt of negative numbers (due to floating point errors)
        if (haversine_term < 0.0f) haversine_term = 0.0f;
        if (haversine_term > 1.0f) haversine_term = 1.0f;

        // Central angle: c = 2 × atan2(√a, √(1-a))
        nk_f32_t sqrt_haversine = nk_f32_sqrt_serial(haversine_term);
        nk_f32_t sqrt_complement = nk_f32_sqrt_serial(1.0f - haversine_term);
        nk_f32_t central_angle = 2.0f * nk_f32_atan2(sqrt_haversine, sqrt_complement);

        results[i] = earth_radius * central_angle;
    }
}

NK_PUBLIC void nk_vincenty_f64_serial(              //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    nk_f64_t const equatorial_radius = NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
    nk_f64_t const polar_radius = NK_EARTH_ELLIPSOID_POLAR_RADIUS;
    nk_f64_t const flattening = 1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t first_latitude = a_lats[i];
        nk_f64_t second_latitude = b_lats[i];
        nk_f64_t longitude_difference = b_lons[i] - a_lons[i];

        // Reduced latitudes on the auxiliary sphere
        nk_f64_t tan_reduced_first = (1.0 - flattening) * nk_f64_tan(first_latitude);
        nk_f64_t tan_reduced_second = (1.0 - flattening) * nk_f64_tan(second_latitude);
        nk_f64_t cos_reduced_first = 1.0 / nk_f64_sqrt_serial(1.0 + tan_reduced_first * tan_reduced_first);
        nk_f64_t sin_reduced_first = tan_reduced_first * cos_reduced_first;
        nk_f64_t cos_reduced_second = 1.0 / nk_f64_sqrt_serial(1.0 + tan_reduced_second * tan_reduced_second);
        nk_f64_t sin_reduced_second = tan_reduced_second * cos_reduced_second;

        // Iterative convergence of lambda (difference in longitude on auxiliary sphere)
        nk_f64_t lambda = longitude_difference;
        nk_f64_t lambda_previous = longitude_difference;
        nk_f64_t sin_angular_distance, cos_angular_distance, angular_distance;
        nk_f64_t sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;
        nk_u32_t iteration = 0;

        // Check for coincident points early
        nk_u32_t coincident = 0;
        do {
            nk_f64_t sin_lambda = nk_f64_sin(lambda);
            nk_f64_t cos_lambda = nk_f64_cos(lambda);

            nk_f64_t cross_term = cos_reduced_second * sin_lambda;
            nk_f64_t mixed_term = cos_reduced_first * sin_reduced_second -
                                  sin_reduced_first * cos_reduced_second * cos_lambda;
            sin_angular_distance = nk_f64_sqrt_serial(cross_term * cross_term + mixed_term * mixed_term);

            if (sin_angular_distance == 0.0) {
                coincident = 1;
                break;
            }

            cos_angular_distance = sin_reduced_first * sin_reduced_second +
                                   cos_reduced_first * cos_reduced_second * cos_lambda;
            angular_distance = nk_f64_atan2(sin_angular_distance, cos_angular_distance);

            sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda / sin_angular_distance;
            cos_squared_azimuth = 1.0 - sin_azimuth * sin_azimuth;

            // Handle equatorial geodesic case
            cos_double_angular_midpoint = (cos_squared_azimuth != 0.0)
                                              ? cos_angular_distance -
                                                    2.0 * sin_reduced_first * sin_reduced_second / cos_squared_azimuth
                                              : 0.0;

            nk_f64_t correction_factor = flattening / 16.0 * cos_squared_azimuth *
                                         (4.0 + flattening * (4.0 - 3.0 * cos_squared_azimuth));

            lambda_previous = lambda;
            lambda = longitude_difference + (1.0 - correction_factor) * flattening * sin_azimuth *
                                                (angular_distance + correction_factor * sin_angular_distance *
                                                                        (cos_double_angular_midpoint +
                                                                         correction_factor * cos_angular_distance *
                                                                             (-1.0 + 2.0 * cos_double_angular_midpoint *
                                                                                         cos_double_angular_midpoint)));

            iteration++;
        } while (nk_f64_abs_(lambda - lambda_previous) > NK_VINCENTY_CONVERGENCE_THRESHOLD &&
                 iteration < NK_VINCENTY_MAX_ITERATIONS);

        if (coincident) {
            results[i] = 0.0;
            continue;
        }

        // Final distance calculation
        nk_f64_t u_squared = cos_squared_azimuth *
                             (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                             (polar_radius * polar_radius);
        nk_f64_t series_a = 1.0 + u_squared / 16384.0 *
                                      (4096.0 + u_squared * (-768.0 + u_squared * (320.0 - 175.0 * u_squared)));
        nk_f64_t series_b = u_squared / 1024.0 * (256.0 + u_squared * (-128.0 + u_squared * (74.0 - 47.0 * u_squared)));

        nk_f64_t angular_correction =
            series_b * sin_angular_distance *
            (cos_double_angular_midpoint +
             series_b / 4.0 *
                 (cos_angular_distance * (-1.0 + 2.0 * cos_double_angular_midpoint * cos_double_angular_midpoint) -
                  series_b / 6.0 * cos_double_angular_midpoint *
                      (-3.0 + 4.0 * sin_angular_distance * sin_angular_distance) *
                      (-3.0 + 4.0 * cos_double_angular_midpoint * cos_double_angular_midpoint)));

        results[i] = polar_radius * series_a * (angular_distance - angular_correction);
    }
}

NK_PUBLIC void nk_vincenty_f32_serial(              //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    nk_f32_t const equatorial_radius = (nk_f32_t)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
    nk_f32_t const polar_radius = (nk_f32_t)NK_EARTH_ELLIPSOID_POLAR_RADIUS;
    nk_f32_t const flattening = 1.0f / (nk_f32_t)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    nk_f32_t const convergence_threshold = (nk_f32_t)NK_VINCENTY_CONVERGENCE_THRESHOLD;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t first_latitude = a_lats[i];
        nk_f32_t second_latitude = b_lats[i];
        nk_f32_t longitude_difference = b_lons[i] - a_lons[i];

        // Reduced latitudes on the auxiliary sphere
        nk_f32_t tan_reduced_first = (1.0f - flattening) * nk_f32_tan(first_latitude);
        nk_f32_t tan_reduced_second = (1.0f - flattening) * nk_f32_tan(second_latitude);
        nk_f32_t cos_reduced_first = 1.0f / nk_f32_sqrt_serial(1.0f + tan_reduced_first * tan_reduced_first);
        nk_f32_t sin_reduced_first = tan_reduced_first * cos_reduced_first;
        nk_f32_t cos_reduced_second = 1.0f / nk_f32_sqrt_serial(1.0f + tan_reduced_second * tan_reduced_second);
        nk_f32_t sin_reduced_second = tan_reduced_second * cos_reduced_second;

        // Iterative convergence of lambda (difference in longitude on auxiliary sphere)
        nk_f32_t lambda = longitude_difference;
        nk_f32_t lambda_previous = longitude_difference;
        nk_f32_t sin_angular_distance, cos_angular_distance, angular_distance;
        nk_f32_t sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;
        nk_u32_t iteration = 0;

        // Check for coincident points early
        nk_u32_t coincident = 0;
        do {
            nk_f32_t sin_lambda = nk_f32_sin(lambda);
            nk_f32_t cos_lambda = nk_f32_cos(lambda);

            nk_f32_t cross_term = cos_reduced_second * sin_lambda;
            nk_f32_t mixed_term = cos_reduced_first * sin_reduced_second -
                                  sin_reduced_first * cos_reduced_second * cos_lambda;
            sin_angular_distance = nk_f32_sqrt_serial(cross_term * cross_term + mixed_term * mixed_term);

            if (sin_angular_distance == 0.0f) {
                coincident = 1;
                break;
            }

            cos_angular_distance = sin_reduced_first * sin_reduced_second +
                                   cos_reduced_first * cos_reduced_second * cos_lambda;
            angular_distance = nk_f32_atan2(sin_angular_distance, cos_angular_distance);

            sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda / sin_angular_distance;
            cos_squared_azimuth = 1.0f - sin_azimuth * sin_azimuth;

            // Handle equatorial geodesic case
            cos_double_angular_midpoint = (cos_squared_azimuth != 0.0f)
                                              ? cos_angular_distance -
                                                    2.0f * sin_reduced_first * sin_reduced_second / cos_squared_azimuth
                                              : 0.0f;

            nk_f32_t correction_factor = flattening / 16.0f * cos_squared_azimuth *
                                         (4.0f + flattening * (4.0f - 3.0f * cos_squared_azimuth));

            lambda_previous = lambda;
            lambda = longitude_difference +
                     (1.0f - correction_factor) * flattening * sin_azimuth *
                         (angular_distance +
                          correction_factor * sin_angular_distance *
                              (cos_double_angular_midpoint +
                               correction_factor * cos_angular_distance *
                                   (-1.0f + 2.0f * cos_double_angular_midpoint * cos_double_angular_midpoint)));

            iteration++;
        } while (nk_f32_abs_(lambda - lambda_previous) > convergence_threshold &&
                 iteration < NK_VINCENTY_MAX_ITERATIONS);

        if (coincident) {
            results[i] = 0.0;
            continue;
        }

        // Final distance calculation
        nk_f32_t u_squared = cos_squared_azimuth *
                             (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                             (polar_radius * polar_radius);
        nk_f32_t series_a = 1.0f + u_squared / 16384.0f *
                                       (4096.0f + u_squared * (-768.0f + u_squared * (320.0f - 175.0f * u_squared)));
        nk_f32_t series_b = u_squared / 1024.0f *
                            (256.0f + u_squared * (-128.0f + u_squared * (74.0f - 47.0f * u_squared)));

        nk_f32_t angular_correction =
            series_b * sin_angular_distance *
            (cos_double_angular_midpoint +
             series_b / 4.0f *
                 (cos_angular_distance * (-1.0f + 2.0f * cos_double_angular_midpoint * cos_double_angular_midpoint) -
                  series_b / 6.0f * cos_double_angular_midpoint *
                      (-3.0f + 4.0f * sin_angular_distance * sin_angular_distance) *
                      (-3.0f + 4.0f * cos_double_angular_midpoint * cos_double_angular_midpoint)));

        results[i] = polar_radius * series_a * (angular_distance - angular_correction);
    }
}

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

#include "numkong/trigonometry/neon.h"
#include "numkong/reduce/neon.h" // Partial load/store helpers

/*  NEON implementations using 2-wide f64 and 4-wide f32 SIMD.
 *  These require NEON trigonometric kernels from trigonometry/neon.h.
 */

NK_INTERNAL float64x2_t nk_haversine_f64x2_neon_(              //
    float64x2_t first_latitudes, float64x2_t first_longitudes, //
    float64x2_t second_latitudes, float64x2_t second_longitudes) {

    float64x2_t const earth_radius = vdupq_n_f64(NK_EARTH_MEDIATORIAL_RADIUS);
    float64x2_t const half = vdupq_n_f64(0.5);
    float64x2_t const one = vdupq_n_f64(1.0);
    float64x2_t const two = vdupq_n_f64(2.0);

    float64x2_t latitude_delta = vsubq_f64(second_latitudes, first_latitudes);
    float64x2_t longitude_delta = vsubq_f64(second_longitudes, first_longitudes);

    // Haversine terms: sin²(Δ/2)
    float64x2_t latitude_delta_half = vmulq_f64(latitude_delta, half);
    float64x2_t longitude_delta_half = vmulq_f64(longitude_delta, half);
    float64x2_t sin_latitude_delta_half = nk_f64x2_sin_neon_(latitude_delta_half);
    float64x2_t sin_longitude_delta_half = nk_f64x2_sin_neon_(longitude_delta_half);
    float64x2_t sin_squared_latitude_delta_half = vmulq_f64(sin_latitude_delta_half, sin_latitude_delta_half);
    float64x2_t sin_squared_longitude_delta_half = vmulq_f64(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    float64x2_t cos_first_latitude = nk_f64x2_cos_neon_(first_latitudes);
    float64x2_t cos_second_latitude = nk_f64x2_cos_neon_(second_latitudes);
    float64x2_t cos_latitude_product = vmulq_f64(cos_first_latitude, cos_second_latitude);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    float64x2_t haversine_term = vaddq_f64(sin_squared_latitude_delta_half,
                                           vmulq_f64(cos_latitude_product, sin_squared_longitude_delta_half));
    // Clamp haversine_term to [0, 1] to prevent NaN from sqrt of negative values
    float64x2_t zero = vdupq_n_f64(0.0);
    haversine_term = vmaxq_f64(zero, vminq_f64(one, haversine_term));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    float64x2_t sqrt_haversine = vsqrtq_f64(haversine_term);
    float64x2_t sqrt_complement = vsqrtq_f64(vsubq_f64(one, haversine_term));
    float64x2_t central_angle = vmulq_f64(two, nk_f64x2_atan2_neon_(sqrt_haversine, sqrt_complement));

    return vmulq_f64(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f64_neon(               //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 2) {
        float64x2_t first_latitudes = vld1q_f64(a_lats);
        float64x2_t first_longitudes = vld1q_f64(a_lons);
        float64x2_t second_latitudes = vld1q_f64(b_lats);
        float64x2_t second_longitudes = vld1q_f64(b_lons);

        float64x2_t distances = nk_haversine_f64x2_neon_(first_latitudes, first_longitudes, second_latitudes,
                                                         second_longitudes);
        vst1q_f64(results, distances);

        a_lats += 2, a_lons += 2, b_lats += 2, b_lons += 2, results += 2, n -= 2;
    }

    // Handle tail with partial loads (n can only be 0 or 1 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x2_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x2_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x2_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x2_serial_(b_lons, &b_lon_vec, n);
        float64x2_t distances = nk_haversine_f64x2_neon_(a_lat_vec.f64x2, a_lon_vec.f64x2, b_lat_vec.f64x2,
                                                         b_lon_vec.f64x2);
        result_vec.f64x2 = distances;
        nk_partial_store_b64x2_serial_(&result_vec, results, n);
    }
}

NK_INTERNAL float32x4_t nk_haversine_f32x4_neon_(              //
    float32x4_t first_latitudes, float32x4_t first_longitudes, //
    float32x4_t second_latitudes, float32x4_t second_longitudes) {

    float32x4_t const earth_radius = vdupq_n_f32((float)NK_EARTH_MEDIATORIAL_RADIUS);
    float32x4_t const half = vdupq_n_f32(0.5f);
    float32x4_t const one = vdupq_n_f32(1.0f);
    float32x4_t const two = vdupq_n_f32(2.0f);

    float32x4_t latitude_delta = vsubq_f32(second_latitudes, first_latitudes);
    float32x4_t longitude_delta = vsubq_f32(second_longitudes, first_longitudes);

    // Haversine terms: sin²(Δ/2)
    float32x4_t latitude_delta_half = vmulq_f32(latitude_delta, half);
    float32x4_t longitude_delta_half = vmulq_f32(longitude_delta, half);
    float32x4_t sin_latitude_delta_half = nk_f32x4_sin_neon_(latitude_delta_half);
    float32x4_t sin_longitude_delta_half = nk_f32x4_sin_neon_(longitude_delta_half);
    float32x4_t sin_squared_latitude_delta_half = vmulq_f32(sin_latitude_delta_half, sin_latitude_delta_half);
    float32x4_t sin_squared_longitude_delta_half = vmulq_f32(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    float32x4_t cos_first_latitude = nk_f32x4_cos_neon_(first_latitudes);
    float32x4_t cos_second_latitude = nk_f32x4_cos_neon_(second_latitudes);
    float32x4_t cos_latitude_product = vmulq_f32(cos_first_latitude, cos_second_latitude);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    float32x4_t haversine_term = vaddq_f32(sin_squared_latitude_delta_half,
                                           vmulq_f32(cos_latitude_product, sin_squared_longitude_delta_half));

    // Clamp to [0, 1] to avoid NaN from sqrt of negative numbers (due to floating point errors)
    float32x4_t zero = vdupq_n_f32(0.0f);
    haversine_term = vmaxq_f32(zero, vminq_f32(one, haversine_term));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    float32x4_t sqrt_haversine = vsqrtq_f32(haversine_term);
    float32x4_t sqrt_complement = vsqrtq_f32(vsubq_f32(one, haversine_term));
    float32x4_t central_angle = vmulq_f32(two, nk_f32x4_atan2_neon_(sqrt_haversine, sqrt_complement));

    return vmulq_f32(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f32_neon(               //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 4) {
        float32x4_t first_latitudes = vld1q_f32(a_lats);
        float32x4_t first_longitudes = vld1q_f32(a_lons);
        float32x4_t second_latitudes = vld1q_f32(b_lats);
        float32x4_t second_longitudes = vld1q_f32(b_lons);

        float32x4_t distances = nk_haversine_f32x4_neon_(first_latitudes, first_longitudes, second_latitudes,
                                                         second_longitudes);
        vst1q_f32(results, distances);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle tail with partial loads (n can be 0-3 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x4_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x4_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x4_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x4_serial_(b_lons, &b_lon_vec, n);
        float32x4_t distances = nk_haversine_f32x4_neon_(a_lat_vec.f32x4, a_lon_vec.f32x4, b_lat_vec.f32x4,
                                                         b_lon_vec.f32x4);
        result_vec.f32x4 = distances;
        nk_partial_store_b32x4_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  NEON helper for Vincenty's geodesic distance on 2 f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL float64x2_t nk_vincenty_f64x2_neon_(               //
    float64x2_t first_latitudes, float64x2_t first_longitudes, //
    float64x2_t second_latitudes, float64x2_t second_longitudes) {

    float64x2_t const equatorial_radius = vdupq_n_f64(NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    float64x2_t const polar_radius = vdupq_n_f64(NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    float64x2_t const flattening = vdupq_n_f64(1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    float64x2_t const convergence_threshold = vdupq_n_f64(NK_VINCENTY_CONVERGENCE_THRESHOLD);
    float64x2_t const one = vdupq_n_f64(1.0);
    float64x2_t const two = vdupq_n_f64(2.0);
    float64x2_t const three = vdupq_n_f64(3.0);
    float64x2_t const four = vdupq_n_f64(4.0);
    float64x2_t const six = vdupq_n_f64(6.0);
    float64x2_t const sixteen = vdupq_n_f64(16.0);
    float64x2_t const epsilon = vdupq_n_f64(1e-15);

    // Longitude difference
    float64x2_t longitude_difference = vsubq_f64(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    float64x2_t one_minus_f = vsubq_f64(one, flattening);
    float64x2_t tan_first = vdivq_f64(nk_f64x2_sin_neon_(first_latitudes), nk_f64x2_cos_neon_(first_latitudes));
    float64x2_t tan_second = vdivq_f64(nk_f64x2_sin_neon_(second_latitudes), nk_f64x2_cos_neon_(second_latitudes));
    float64x2_t tan_reduced_first = vmulq_f64(one_minus_f, tan_first);
    float64x2_t tan_reduced_second = vmulq_f64(one_minus_f, tan_second);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    float64x2_t cos_reduced_first = vdivq_f64(one, vsqrtq_f64(vfmaq_f64(one, tan_reduced_first, tan_reduced_first)));
    float64x2_t sin_reduced_first = vmulq_f64(tan_reduced_first, cos_reduced_first);
    float64x2_t cos_reduced_second = vdivq_f64(one, vsqrtq_f64(vfmaq_f64(one, tan_reduced_second, tan_reduced_second)));
    float64x2_t sin_reduced_second = vmulq_f64(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    float64x2_t lambda = longitude_difference;
    float64x2_t sin_angular_distance, cos_angular_distance, angular_distance;
    float64x2_t sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points using masks
    uint64x2_t converged_mask = vdupq_n_u64(0);
    uint64x2_t coincident_mask = vdupq_n_u64(0);

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        uint64_t converged_bits = vgetq_lane_u64(converged_mask, 0) & vgetq_lane_u64(converged_mask, 1);
        if (converged_bits) break;

        float64x2_t sin_lambda = nk_f64x2_sin_neon_(lambda);
        float64x2_t cos_lambda = nk_f64x2_cos_neon_(lambda);

        // sin²(angular_distance) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        float64x2_t cross_term = vmulq_f64(cos_reduced_second, sin_lambda);
        float64x2_t mixed_term = vsubq_f64(vmulq_f64(cos_reduced_first, sin_reduced_second),
                                           vmulq_f64(vmulq_f64(sin_reduced_first, cos_reduced_second), cos_lambda));
        float64x2_t sin_angular_dist_sq = vfmaq_f64(vmulq_f64(mixed_term, mixed_term), cross_term, cross_term);
        sin_angular_distance = vsqrtq_f64(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = vcltq_f64(sin_angular_distance, epsilon);

        // cos(angular_distance) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance = vfmaq_f64(vmulq_f64(sin_reduced_first, sin_reduced_second),
                                         vmulq_f64(cos_reduced_first, cos_reduced_second), cos_lambda);

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_f64x2_atan2_neon_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance)
        // Avoid division by zero by using blending
        float64x2_t safe_sin_angular = vbslq_f64(coincident_mask, one, sin_angular_distance);
        sin_azimuth = vdivq_f64(vmulq_f64(vmulq_f64(cos_reduced_first, cos_reduced_second), sin_lambda),
                                safe_sin_angular);
        cos_squared_azimuth = vsubq_f64(one, vmulq_f64(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α ≈ 0
        uint64x2_t equatorial_mask = vcltq_f64(cos_squared_azimuth, epsilon);
        float64x2_t safe_cos_sq_azimuth = vbslq_f64(equatorial_mask, one, cos_squared_azimuth);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        float64x2_t sin_product = vmulq_f64(sin_reduced_first, sin_reduced_second);
        cos_double_angular_midpoint = vsubq_f64(cos_angular_distance,
                                                vdivq_f64(vmulq_f64(two, sin_product), safe_cos_sq_azimuth));
        cos_double_angular_midpoint = vbslq_f64(equatorial_mask, vdupq_n_f64(0.0), cos_double_angular_midpoint);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        float64x2_t correction_factor = vmulq_f64(
            vdivq_f64(flattening, sixteen),
            vmulq_f64(cos_squared_azimuth, vfmaq_f64(four, flattening, vfmsq_f64(four, three, cos_squared_azimuth))));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        float64x2_t cos_2sm_sq = vmulq_f64(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2 × cos²(2σₘ)
        float64x2_t innermost = vfmaq_f64(vdupq_n_f64(-1.0), two, cos_2sm_sq);
        // middle = cos(2σₘ) + C × cos(σ) × innermost
        float64x2_t middle = vfmaq_f64(cos_double_angular_midpoint, vmulq_f64(correction_factor, cos_angular_distance),
                                       innermost);
        // inner = C × sin(σ) × middle
        float64x2_t inner = vmulq_f64(vmulq_f64(correction_factor, sin_angular_distance), middle);

        // λ' = L + (1-C) * f * sin_α * (σ + inner)
        float64x2_t lambda_new = vfmaq_f64(
            longitude_difference, vmulq_f64(vmulq_f64(vsubq_f64(one, correction_factor), flattening), sin_azimuth),
            vaddq_f64(angular_distance, inner));

        // Check convergence: |λ - λ'| < threshold
        float64x2_t lambda_diff = vsubq_f64(lambda_new, lambda);
        float64x2_t lambda_diff_abs = vabsq_f64(lambda_diff);
        uint64x2_t newly_converged = vcltq_f64(lambda_diff_abs, convergence_threshold);
        converged_mask = vorrq_u64(converged_mask, newly_converged);

        // Only update lambda for non-converged lanes
        lambda = vbslq_f64(converged_mask, lambda, lambda_new);
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    float64x2_t a_sq = vmulq_f64(equatorial_radius, equatorial_radius);
    float64x2_t b_sq = vmulq_f64(polar_radius, polar_radius);
    float64x2_t u_squared = vdivq_f64(vmulq_f64(cos_squared_azimuth, vsubq_f64(a_sq, b_sq)), b_sq);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    float64x2_t series_a = vfmaq_f64(vdupq_n_f64(320.0), u_squared, vdupq_n_f64(-175.0));
    series_a = vfmaq_f64(vdupq_n_f64(-768.0), u_squared, series_a);
    series_a = vfmaq_f64(vdupq_n_f64(4096.0), u_squared, series_a);
    series_a = vfmaq_f64(one, vdivq_f64(u_squared, vdupq_n_f64(16384.0)), series_a);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    float64x2_t series_b = vfmaq_f64(vdupq_n_f64(74.0), u_squared, vdupq_n_f64(-47.0));
    series_b = vfmaq_f64(vdupq_n_f64(-128.0), u_squared, series_b);
    series_b = vfmaq_f64(vdupq_n_f64(256.0), u_squared, series_b);
    series_b = vmulq_f64(vdivq_f64(u_squared, vdupq_n_f64(1024.0)), series_b);

    // Δσ = B × sin(σ) × (cos(2σₘ) + B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 +
    // 4 × cos²(2σₘ))))
    float64x2_t cos_2sm_sq = vmulq_f64(cos_double_angular_midpoint, cos_double_angular_midpoint);
    float64x2_t sin_sq = vmulq_f64(sin_angular_distance, sin_angular_distance);
    float64x2_t term1 = vfmaq_f64(vdupq_n_f64(-1.0), two, cos_2sm_sq);
    term1 = vmulq_f64(cos_angular_distance, term1);
    float64x2_t term2 = vfmaq_f64(vdupq_n_f64(-3.0), four, sin_sq);
    float64x2_t term3 = vfmaq_f64(vdupq_n_f64(-3.0), four, cos_2sm_sq);
    term2 = vmulq_f64(vmulq_f64(vdivq_f64(series_b, six), cos_double_angular_midpoint), vmulq_f64(term2, term3));
    float64x2_t delta_sigma = vmulq_f64(
        series_b,
        vmulq_f64(sin_angular_distance, vaddq_f64(cos_double_angular_midpoint,
                                                  vmulq_f64(vdivq_f64(series_b, four), vsubq_f64(term1, term2)))));

    // s = b * A * (σ - Δσ)
    float64x2_t distances = vmulq_f64(vmulq_f64(polar_radius, series_a), vsubq_f64(angular_distance, delta_sigma));

    // Set coincident points to zero
    distances = vbslq_f64(coincident_mask, vdupq_n_f64(0.0), distances);

    return distances;
}

NK_PUBLIC void nk_vincenty_f64_neon(                //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 2) {
        float64x2_t first_latitudes = vld1q_f64(a_lats);
        float64x2_t first_longitudes = vld1q_f64(a_lons);
        float64x2_t second_latitudes = vld1q_f64(b_lats);
        float64x2_t second_longitudes = vld1q_f64(b_lons);

        float64x2_t distances = nk_vincenty_f64x2_neon_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        vst1q_f64(results, distances);

        a_lats += 2, a_lons += 2, b_lats += 2, b_lons += 2, results += 2, n -= 2;
    }

    // Handle remaining elements with partial loads (n can only be 0 or 1 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x2_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x2_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x2_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x2_serial_(b_lons, &b_lon_vec, n);
        float64x2_t distances = nk_vincenty_f64x2_neon_(a_lat_vec.f64x2, a_lon_vec.f64x2, b_lat_vec.f64x2,
                                                        b_lon_vec.f64x2);
        result_vec.f64x2 = distances;
        nk_partial_store_b64x2_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  NEON helper for Vincenty's geodesic distance on 4 f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL float32x4_t nk_vincenty_f32x4_neon_(               //
    float32x4_t first_latitudes, float32x4_t first_longitudes, //
    float32x4_t second_latitudes, float32x4_t second_longitudes) {

    float32x4_t const equatorial_radius = vdupq_n_f32((float)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    float32x4_t const polar_radius = vdupq_n_f32((float)NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    float32x4_t const flattening = vdupq_n_f32(1.0f / (float)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    float32x4_t const convergence_threshold = vdupq_n_f32((float)NK_VINCENTY_CONVERGENCE_THRESHOLD);
    float32x4_t const one = vdupq_n_f32(1.0f);
    float32x4_t const two = vdupq_n_f32(2.0f);
    float32x4_t const three = vdupq_n_f32(3.0f);
    float32x4_t const four = vdupq_n_f32(4.0f);
    float32x4_t const six = vdupq_n_f32(6.0f);
    float32x4_t const sixteen = vdupq_n_f32(16.0f);
    float32x4_t const epsilon = vdupq_n_f32(1e-7f);

    // Longitude difference
    float32x4_t longitude_difference = vsubq_f32(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    float32x4_t one_minus_f = vsubq_f32(one, flattening);
    float32x4_t tan_first = vdivq_f32(nk_f32x4_sin_neon_(first_latitudes), nk_f32x4_cos_neon_(first_latitudes));
    float32x4_t tan_second = vdivq_f32(nk_f32x4_sin_neon_(second_latitudes), nk_f32x4_cos_neon_(second_latitudes));
    float32x4_t tan_reduced_first = vmulq_f32(one_minus_f, tan_first);
    float32x4_t tan_reduced_second = vmulq_f32(one_minus_f, tan_second);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    float32x4_t cos_reduced_first = vdivq_f32(one, vsqrtq_f32(vfmaq_f32(one, tan_reduced_first, tan_reduced_first)));
    float32x4_t sin_reduced_first = vmulq_f32(tan_reduced_first, cos_reduced_first);
    float32x4_t cos_reduced_second = vdivq_f32(one, vsqrtq_f32(vfmaq_f32(one, tan_reduced_second, tan_reduced_second)));
    float32x4_t sin_reduced_second = vmulq_f32(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    float32x4_t lambda = longitude_difference;
    float32x4_t sin_angular_distance, cos_angular_distance, angular_distance;
    float32x4_t sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points using masks
    uint32x4_t converged_mask = vdupq_n_u32(0);
    uint32x4_t coincident_mask = vdupq_n_u32(0);

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged (all bits set = 0xFFFFFFFF per lane)
        uint32_t converged_bits = vminvq_u32(converged_mask);
        if (converged_bits == 0xFFFFFFFF) break;

        float32x4_t sin_lambda = nk_f32x4_sin_neon_(lambda);
        float32x4_t cos_lambda = nk_f32x4_cos_neon_(lambda);

        // sin²(angular_distance) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        float32x4_t cross_term = vmulq_f32(cos_reduced_second, sin_lambda);
        float32x4_t mixed_term = vsubq_f32(vmulq_f32(cos_reduced_first, sin_reduced_second),
                                           vmulq_f32(vmulq_f32(sin_reduced_first, cos_reduced_second), cos_lambda));
        float32x4_t sin_angular_dist_sq = vfmaq_f32(vmulq_f32(mixed_term, mixed_term), cross_term, cross_term);
        sin_angular_distance = vsqrtq_f32(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = vcltq_f32(sin_angular_distance, epsilon);

        // cos(angular_distance) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance = vfmaq_f32(vmulq_f32(sin_reduced_first, sin_reduced_second),
                                         vmulq_f32(cos_reduced_first, cos_reduced_second), cos_lambda);

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_f32x4_atan2_neon_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance)
        float32x4_t safe_sin_angular = vbslq_f32(coincident_mask, one, sin_angular_distance);
        sin_azimuth = vdivq_f32(vmulq_f32(vmulq_f32(cos_reduced_first, cos_reduced_second), sin_lambda),
                                safe_sin_angular);
        cos_squared_azimuth = vsubq_f32(one, vmulq_f32(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α ≈ 0
        uint32x4_t equatorial_mask = vcltq_f32(cos_squared_azimuth, epsilon);
        float32x4_t safe_cos_sq_azimuth = vbslq_f32(equatorial_mask, one, cos_squared_azimuth);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        float32x4_t sin_product = vmulq_f32(sin_reduced_first, sin_reduced_second);
        cos_double_angular_midpoint = vsubq_f32(cos_angular_distance,
                                                vdivq_f32(vmulq_f32(two, sin_product), safe_cos_sq_azimuth));
        cos_double_angular_midpoint = vbslq_f32(equatorial_mask, vdupq_n_f32(0.0f), cos_double_angular_midpoint);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        float32x4_t correction_factor = vmulq_f32(
            vdivq_f32(flattening, sixteen),
            vmulq_f32(cos_squared_azimuth, vfmaq_f32(four, flattening, vfmsq_f32(four, three, cos_squared_azimuth))));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        float32x4_t cos_2sm_sq = vmulq_f32(cos_double_angular_midpoint, cos_double_angular_midpoint);
        float32x4_t innermost = vfmaq_f32(vdupq_n_f32(-1.0f), two, cos_2sm_sq);
        float32x4_t middle = vfmaq_f32(cos_double_angular_midpoint, vmulq_f32(correction_factor, cos_angular_distance),
                                       innermost);
        float32x4_t inner = vmulq_f32(vmulq_f32(correction_factor, sin_angular_distance), middle);

        float32x4_t lambda_new = vfmaq_f32(
            longitude_difference, vmulq_f32(vmulq_f32(vsubq_f32(one, correction_factor), flattening), sin_azimuth),
            vaddq_f32(angular_distance, inner));

        // Check convergence: |λ - λ'| < threshold
        float32x4_t lambda_diff = vsubq_f32(lambda_new, lambda);
        float32x4_t lambda_diff_abs = vabsq_f32(lambda_diff);
        uint32x4_t newly_converged = vcltq_f32(lambda_diff_abs, convergence_threshold);
        converged_mask = vorrq_u32(converged_mask, newly_converged);

        // Only update lambda for non-converged lanes
        lambda = vbslq_f32(converged_mask, lambda, lambda_new);
    }

    // Final distance calculation
    float32x4_t a_sq = vmulq_f32(equatorial_radius, equatorial_radius);
    float32x4_t b_sq = vmulq_f32(polar_radius, polar_radius);
    float32x4_t u_squared = vdivq_f32(vmulq_f32(cos_squared_azimuth, vsubq_f32(a_sq, b_sq)), b_sq);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    float32x4_t series_a = vfmaq_f32(vdupq_n_f32(320.0f), u_squared, vdupq_n_f32(-175.0f));
    series_a = vfmaq_f32(vdupq_n_f32(-768.0f), u_squared, series_a);
    series_a = vfmaq_f32(vdupq_n_f32(4096.0f), u_squared, series_a);
    series_a = vfmaq_f32(one, vdivq_f32(u_squared, vdupq_n_f32(16384.0f)), series_a);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    float32x4_t series_b = vfmaq_f32(vdupq_n_f32(74.0f), u_squared, vdupq_n_f32(-47.0f));
    series_b = vfmaq_f32(vdupq_n_f32(-128.0f), u_squared, series_b);
    series_b = vfmaq_f32(vdupq_n_f32(256.0f), u_squared, series_b);
    series_b = vmulq_f32(vdivq_f32(u_squared, vdupq_n_f32(1024.0f)), series_b);

    // Δσ calculation
    float32x4_t cos_2sm_sq = vmulq_f32(cos_double_angular_midpoint, cos_double_angular_midpoint);
    float32x4_t sin_sq = vmulq_f32(sin_angular_distance, sin_angular_distance);
    float32x4_t term1 = vfmaq_f32(vdupq_n_f32(-1.0f), two, cos_2sm_sq);
    term1 = vmulq_f32(cos_angular_distance, term1);
    float32x4_t term2 = vfmaq_f32(vdupq_n_f32(-3.0f), four, sin_sq);
    float32x4_t term3 = vfmaq_f32(vdupq_n_f32(-3.0f), four, cos_2sm_sq);
    term2 = vmulq_f32(vmulq_f32(vdivq_f32(series_b, six), cos_double_angular_midpoint), vmulq_f32(term2, term3));
    float32x4_t delta_sigma = vmulq_f32(
        series_b,
        vmulq_f32(sin_angular_distance, vaddq_f32(cos_double_angular_midpoint,
                                                  vmulq_f32(vdivq_f32(series_b, four), vsubq_f32(term1, term2)))));

    // s = b * A * (σ - Δσ)
    float32x4_t distances = vmulq_f32(vmulq_f32(polar_radius, series_a), vsubq_f32(angular_distance, delta_sigma));

    // Set coincident points to zero
    distances = vbslq_f32(coincident_mask, vdupq_n_f32(0.0f), distances);

    return distances;
}

NK_PUBLIC void nk_vincenty_f32_neon(                //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 4) {
        float32x4_t first_latitudes = vld1q_f32(a_lats);
        float32x4_t first_longitudes = vld1q_f32(a_lons);
        float32x4_t second_latitudes = vld1q_f32(b_lats);
        float32x4_t second_longitudes = vld1q_f32(b_lons);

        float32x4_t distances = nk_vincenty_f32x4_neon_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        vst1q_f32(results, distances);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle remaining elements with partial loads (n can be 1-3 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x4_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x4_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x4_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x4_serial_(b_lons, &b_lon_vec, n);
        float32x4_t distances = nk_vincenty_f32x4_neon_(a_lat_vec.f32x4, a_lon_vec.f32x4, b_lat_vec.f32x4,
                                                        b_lon_vec.f32x4);
        result_vec.f32x4 = distances;
        nk_partial_store_b32x4_serial_(&result_vec, results, n);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

/*  Haswell AVX2 implementations using 4-wide f64 and 8-wide f32 SIMD.
 *  These require AVX2 trigonometric kernels from trigonometry.h.
 */

NK_INTERNAL __m256d nk_haversine_f64x4_haswell_(       //
    __m256d first_latitudes, __m256d first_longitudes, //
    __m256d second_latitudes, __m256d second_longitudes) {

    __m256d const earth_radius = _mm256_set1_pd(NK_EARTH_MEDIATORIAL_RADIUS);
    __m256d const half = _mm256_set1_pd(0.5);
    __m256d const one = _mm256_set1_pd(1.0);
    __m256d const two = _mm256_set1_pd(2.0);

    __m256d latitude_delta = _mm256_sub_pd(second_latitudes, first_latitudes);
    __m256d longitude_delta = _mm256_sub_pd(second_longitudes, first_longitudes);

    // Haversine terms: sin²(Δ/2)
    __m256d latitude_delta_half = _mm256_mul_pd(latitude_delta, half);
    __m256d longitude_delta_half = _mm256_mul_pd(longitude_delta, half);
    __m256d sin_latitude_delta_half = nk_f64x4_sin_haswell_(latitude_delta_half);
    __m256d sin_longitude_delta_half = nk_f64x4_sin_haswell_(longitude_delta_half);
    __m256d sin_squared_latitude_delta_half = _mm256_mul_pd(sin_latitude_delta_half, sin_latitude_delta_half);
    __m256d sin_squared_longitude_delta_half = _mm256_mul_pd(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    __m256d cos_first_latitude = nk_f64x4_cos_haswell_(first_latitudes);
    __m256d cos_second_latitude = nk_f64x4_cos_haswell_(second_latitudes);
    __m256d cos_latitude_product = _mm256_mul_pd(cos_first_latitude, cos_second_latitude);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    __m256d haversine_term = _mm256_add_pd(sin_squared_latitude_delta_half,
                                           _mm256_mul_pd(cos_latitude_product, sin_squared_longitude_delta_half));
    // Clamp haversine_term to [0, 1] to prevent NaN from sqrt of negative values
    __m256d zero = _mm256_setzero_pd();
    haversine_term = _mm256_max_pd(zero, _mm256_min_pd(one, haversine_term));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    __m256d sqrt_haversine = _mm256_sqrt_pd(haversine_term);
    __m256d sqrt_complement = _mm256_sqrt_pd(_mm256_sub_pd(one, haversine_term));
    __m256d central_angle = _mm256_mul_pd(two, nk_f64x4_atan2_haswell_(sqrt_haversine, sqrt_complement));

    return _mm256_mul_pd(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f64_haswell(            //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 4) {
        __m256d first_latitudes = _mm256_loadu_pd(a_lats);
        __m256d first_longitudes = _mm256_loadu_pd(a_lons);
        __m256d second_latitudes = _mm256_loadu_pd(b_lats);
        __m256d second_longitudes = _mm256_loadu_pd(b_lons);

        __m256d distances = nk_haversine_f64x4_haswell_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        _mm256_storeu_pd(results, distances);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle remaining elements with partial loads (n can be 1-3 here)
    if (n > 0) {
        nk_b256_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x4_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x4_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x4_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x4_serial_(b_lons, &b_lon_vec, n);
        __m256d distances = nk_haversine_f64x4_haswell_(a_lat_vec.ymm_pd, a_lon_vec.ymm_pd, b_lat_vec.ymm_pd,
                                                        b_lon_vec.ymm_pd);
        result_vec.ymm_pd = distances;
        nk_partial_store_b64x4_serial_(&result_vec, results, n);
    }
}

NK_INTERNAL __m256 nk_haversine_f32x8_haswell_(      //
    __m256 first_latitudes, __m256 first_longitudes, //
    __m256 second_latitudes, __m256 second_longitudes) {

    __m256 const earth_radius = _mm256_set1_ps((float)NK_EARTH_MEDIATORIAL_RADIUS);
    __m256 const half = _mm256_set1_ps(0.5f);
    __m256 const one = _mm256_set1_ps(1.0f);
    __m256 const two = _mm256_set1_ps(2.0f);

    __m256 latitude_delta = _mm256_sub_ps(second_latitudes, first_latitudes);
    __m256 longitude_delta = _mm256_sub_ps(second_longitudes, first_longitudes);

    // Haversine terms: sin²(Δ/2)
    __m256 latitude_delta_half = _mm256_mul_ps(latitude_delta, half);
    __m256 longitude_delta_half = _mm256_mul_ps(longitude_delta, half);
    __m256 sin_latitude_delta_half = nk_f32x8_sin_haswell_(latitude_delta_half);
    __m256 sin_longitude_delta_half = nk_f32x8_sin_haswell_(longitude_delta_half);
    __m256 sin_squared_latitude_delta_half = _mm256_mul_ps(sin_latitude_delta_half, sin_latitude_delta_half);
    __m256 sin_squared_longitude_delta_half = _mm256_mul_ps(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    __m256 cos_first_latitude = nk_f32x8_cos_haswell_(first_latitudes);
    __m256 cos_second_latitude = nk_f32x8_cos_haswell_(second_latitudes);
    __m256 cos_latitude_product = _mm256_mul_ps(cos_first_latitude, cos_second_latitude);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    __m256 haversine_term = _mm256_add_ps(sin_squared_latitude_delta_half,
                                          _mm256_mul_ps(cos_latitude_product, sin_squared_longitude_delta_half));

    // Clamp to [0, 1] to avoid NaN from sqrt of negative numbers (due to floating point errors)
    __m256 zero = _mm256_setzero_ps();
    haversine_term = _mm256_max_ps(zero, _mm256_min_ps(one, haversine_term));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    __m256 sqrt_haversine = _mm256_sqrt_ps(haversine_term);
    __m256 sqrt_complement = _mm256_sqrt_ps(_mm256_sub_ps(one, haversine_term));
    __m256 central_angle = _mm256_mul_ps(two, nk_f32x8_atan2_haswell_(sqrt_haversine, sqrt_complement));

    return _mm256_mul_ps(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f32_haswell(            //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 8) {
        __m256 first_latitudes = _mm256_loadu_ps(a_lats);
        __m256 first_longitudes = _mm256_loadu_ps(a_lons);
        __m256 second_latitudes = _mm256_loadu_ps(b_lats);
        __m256 second_longitudes = _mm256_loadu_ps(b_lons);

        __m256 distances = nk_haversine_f32x8_haswell_(first_latitudes, first_longitudes, second_latitudes,
                                                       second_longitudes);
        _mm256_storeu_ps(results, distances);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with partial loads (n can be 1-7 here)
    if (n > 0) {
        nk_b256_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x8_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x8_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x8_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x8_serial_(b_lons, &b_lon_vec, n);
        __m256 distances = nk_haversine_f32x8_haswell_(a_lat_vec.ymm_ps, a_lon_vec.ymm_ps, b_lat_vec.ymm_ps,
                                                       b_lon_vec.ymm_ps);
        result_vec.ymm_ps = distances;
        nk_partial_store_b32x8_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  AVX2 helper for Vincenty's geodesic distance on 4 f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL __m256d nk_vincenty_f64x4_haswell_(        //
    __m256d first_latitudes, __m256d first_longitudes, //
    __m256d second_latitudes, __m256d second_longitudes) {

    __m256d const equatorial_radius = _mm256_set1_pd(NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m256d const polar_radius = _mm256_set1_pd(NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m256d const flattening = _mm256_set1_pd(1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m256d const convergence_threshold = _mm256_set1_pd(NK_VINCENTY_CONVERGENCE_THRESHOLD);
    __m256d const one = _mm256_set1_pd(1.0);
    __m256d const two = _mm256_set1_pd(2.0);
    __m256d const three = _mm256_set1_pd(3.0);
    __m256d const four = _mm256_set1_pd(4.0);
    __m256d const six = _mm256_set1_pd(6.0);
    __m256d const sixteen = _mm256_set1_pd(16.0);
    __m256d const epsilon = _mm256_set1_pd(1e-15);

    // Longitude difference
    __m256d longitude_difference = _mm256_sub_pd(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    __m256d one_minus_f = _mm256_sub_pd(one, flattening);
    __m256d tan_first = _mm256_div_pd(nk_f64x4_sin_haswell_(first_latitudes), nk_f64x4_cos_haswell_(first_latitudes));
    __m256d tan_second = _mm256_div_pd(nk_f64x4_sin_haswell_(second_latitudes),
                                       nk_f64x4_cos_haswell_(second_latitudes));
    __m256d tan_reduced_first = _mm256_mul_pd(one_minus_f, tan_first);
    __m256d tan_reduced_second = _mm256_mul_pd(one_minus_f, tan_second);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    __m256d cos_reduced_first = _mm256_div_pd(
        one, _mm256_sqrt_pd(_mm256_fmadd_pd(tan_reduced_first, tan_reduced_first, one)));
    __m256d sin_reduced_first = _mm256_mul_pd(tan_reduced_first, cos_reduced_first);
    __m256d cos_reduced_second = _mm256_div_pd(
        one, _mm256_sqrt_pd(_mm256_fmadd_pd(tan_reduced_second, tan_reduced_second, one)));
    __m256d sin_reduced_second = _mm256_mul_pd(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    __m256d lambda = longitude_difference;
    __m256d sin_angular_distance, cos_angular_distance, angular_distance;
    __m256d sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points using masks
    __m256d converged_mask = _mm256_setzero_pd();
    __m256d coincident_mask = _mm256_setzero_pd();

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        int converged_bits = _mm256_movemask_pd(converged_mask);
        if (converged_bits == 0xF) break;

        __m256d sin_lambda = nk_f64x4_sin_haswell_(lambda);
        __m256d cos_lambda = nk_f64x4_cos_haswell_(lambda);

        // sin²(angular_distance) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        __m256d cross_term = _mm256_mul_pd(cos_reduced_second, sin_lambda);
        __m256d mixed_term = _mm256_sub_pd(
            _mm256_mul_pd(cos_reduced_first, sin_reduced_second),
            _mm256_mul_pd(_mm256_mul_pd(sin_reduced_first, cos_reduced_second), cos_lambda));
        __m256d sin_angular_dist_sq = _mm256_fmadd_pd(cross_term, cross_term, _mm256_mul_pd(mixed_term, mixed_term));
        sin_angular_distance = _mm256_sqrt_pd(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = _mm256_cmp_pd(sin_angular_distance, epsilon, _CMP_LT_OS);

        // cos(angular_distance) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance = _mm256_fmadd_pd(_mm256_mul_pd(cos_reduced_first, cos_reduced_second), cos_lambda,
                                               _mm256_mul_pd(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_f64x4_atan2_haswell_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance)
        // Avoid division by zero by using blending
        __m256d safe_sin_angular = _mm256_blendv_pd(sin_angular_distance, one, coincident_mask);
        sin_azimuth = _mm256_div_pd(_mm256_mul_pd(_mm256_mul_pd(cos_reduced_first, cos_reduced_second), sin_lambda),
                                    safe_sin_angular);
        cos_squared_azimuth = _mm256_sub_pd(one, _mm256_mul_pd(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α ≈ 0
        __m256d equatorial_mask = _mm256_cmp_pd(cos_squared_azimuth, epsilon, _CMP_LT_OS);
        __m256d safe_cos_sq_azimuth = _mm256_blendv_pd(cos_squared_azimuth, one, equatorial_mask);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        __m256d sin_product = _mm256_mul_pd(sin_reduced_first, sin_reduced_second);
        cos_double_angular_midpoint = _mm256_sub_pd(
            cos_angular_distance, _mm256_div_pd(_mm256_mul_pd(two, sin_product), safe_cos_sq_azimuth));
        cos_double_angular_midpoint = _mm256_blendv_pd(cos_double_angular_midpoint, _mm256_setzero_pd(),
                                                       equatorial_mask);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        __m256d correction_factor = _mm256_mul_pd(
            _mm256_div_pd(flattening, sixteen),
            _mm256_mul_pd(cos_squared_azimuth,
                          _mm256_fmadd_pd(flattening, _mm256_fnmadd_pd(three, cos_squared_azimuth, four), four)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        __m256d cos_2sm_sq = _mm256_mul_pd(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2 × cos²(2σₘ)
        __m256d innermost = _mm256_fmadd_pd(two, cos_2sm_sq, _mm256_set1_pd(-1.0));
        // middle = cos(2σₘ) + C × cos(σ) × innermost
        __m256d middle = _mm256_fmadd_pd(_mm256_mul_pd(correction_factor, cos_angular_distance), innermost,
                                         cos_double_angular_midpoint);
        // inner = C × sin(σ) × middle
        __m256d inner = _mm256_mul_pd(_mm256_mul_pd(correction_factor, sin_angular_distance), middle);

        // λ' = L + (1-C) * f * sin_α * (σ + inner)
        __m256d lambda_new = _mm256_fmadd_pd(
            _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(one, correction_factor), flattening), sin_azimuth),
            _mm256_add_pd(angular_distance, inner), longitude_difference);

        // Check convergence: |λ - λ'| < threshold
        __m256d lambda_diff_abs = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_sub_pd(lambda_new, lambda));
        __m256d newly_converged = _mm256_cmp_pd(lambda_diff_abs, convergence_threshold, _CMP_LT_OS);
        converged_mask = _mm256_or_pd(converged_mask, newly_converged);

        // Only update lambda for non-converged lanes
        lambda = _mm256_blendv_pd(lambda_new, lambda, converged_mask);
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    __m256d a_sq = _mm256_mul_pd(equatorial_radius, equatorial_radius);
    __m256d b_sq = _mm256_mul_pd(polar_radius, polar_radius);
    __m256d u_squared = _mm256_div_pd(_mm256_mul_pd(cos_squared_azimuth, _mm256_sub_pd(a_sq, b_sq)), b_sq);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    __m256d series_a = _mm256_fmadd_pd(u_squared, _mm256_set1_pd(-175.0), _mm256_set1_pd(320.0));
    series_a = _mm256_fmadd_pd(u_squared, series_a, _mm256_set1_pd(-768.0));
    series_a = _mm256_fmadd_pd(u_squared, series_a, _mm256_set1_pd(4096.0));
    series_a = _mm256_fmadd_pd(_mm256_div_pd(u_squared, _mm256_set1_pd(16384.0)), series_a, one);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    __m256d series_b = _mm256_fmadd_pd(u_squared, _mm256_set1_pd(-47.0), _mm256_set1_pd(74.0));
    series_b = _mm256_fmadd_pd(u_squared, series_b, _mm256_set1_pd(-128.0));
    series_b = _mm256_fmadd_pd(u_squared, series_b, _mm256_set1_pd(256.0));
    series_b = _mm256_mul_pd(_mm256_div_pd(u_squared, _mm256_set1_pd(1024.0)), series_b);

    // Δσ = B × sin(σ) × (cos(2σₘ) +
    //      B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m256d cos_2sm_sq = _mm256_mul_pd(cos_double_angular_midpoint, cos_double_angular_midpoint);
    __m256d sin_sq = _mm256_mul_pd(sin_angular_distance, sin_angular_distance);
    __m256d term1 = _mm256_fmadd_pd(two, cos_2sm_sq, _mm256_set1_pd(-1.0));
    term1 = _mm256_mul_pd(cos_angular_distance, term1);
    __m256d term2 = _mm256_fmadd_pd(four, sin_sq, _mm256_set1_pd(-3.0));
    __m256d term3 = _mm256_fmadd_pd(four, cos_2sm_sq, _mm256_set1_pd(-3.0));
    term2 = _mm256_mul_pd(_mm256_mul_pd(_mm256_div_pd(series_b, six), cos_double_angular_midpoint),
                          _mm256_mul_pd(term2, term3));
    __m256d delta_sigma = _mm256_mul_pd(
        series_b, _mm256_mul_pd(sin_angular_distance, _mm256_add_pd(cos_double_angular_midpoint,
                                                                    _mm256_mul_pd(_mm256_div_pd(series_b, four),
                                                                                  _mm256_sub_pd(term1, term2)))));

    // s = b * A * (σ - Δσ)
    __m256d distances = _mm256_mul_pd(_mm256_mul_pd(polar_radius, series_a),
                                      _mm256_sub_pd(angular_distance, delta_sigma));

    // Set coincident points to zero
    distances = _mm256_blendv_pd(distances, _mm256_setzero_pd(), coincident_mask);

    return distances;
}

NK_PUBLIC void nk_vincenty_f64_haswell(             //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 4) {
        __m256d first_latitudes = _mm256_loadu_pd(a_lats);
        __m256d first_longitudes = _mm256_loadu_pd(a_lons);
        __m256d second_latitudes = _mm256_loadu_pd(b_lats);
        __m256d second_longitudes = _mm256_loadu_pd(b_lons);

        __m256d distances = nk_vincenty_f64x4_haswell_(first_latitudes, first_longitudes, second_latitudes,
                                                       second_longitudes);
        _mm256_storeu_pd(results, distances);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle remaining elements with partial loads (n can be 1-3 here)
    if (n > 0) {
        nk_b256_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x4_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x4_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x4_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x4_serial_(b_lons, &b_lon_vec, n);
        __m256d distances = nk_vincenty_f64x4_haswell_(a_lat_vec.ymm_pd, a_lon_vec.ymm_pd, b_lat_vec.ymm_pd,
                                                       b_lon_vec.ymm_pd);
        result_vec.ymm_pd = distances;
        nk_partial_store_b64x4_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  AVX2 helper for Vincenty's geodesic distance on 8 f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL __m256 nk_vincenty_f32x8_haswell_(       //
    __m256 first_latitudes, __m256 first_longitudes, //
    __m256 second_latitudes, __m256 second_longitudes) {

    __m256 const equatorial_radius = _mm256_set1_ps((float)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m256 const polar_radius = _mm256_set1_ps((float)NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m256 const flattening = _mm256_set1_ps(1.0f / (float)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m256 const convergence_threshold = _mm256_set1_ps((float)NK_VINCENTY_CONVERGENCE_THRESHOLD);
    __m256 const one = _mm256_set1_ps(1.0f);
    __m256 const two = _mm256_set1_ps(2.0f);
    __m256 const three = _mm256_set1_ps(3.0f);
    __m256 const four = _mm256_set1_ps(4.0f);
    __m256 const six = _mm256_set1_ps(6.0f);
    __m256 const sixteen = _mm256_set1_ps(16.0f);
    __m256 const epsilon = _mm256_set1_ps(1e-7f);

    // Longitude difference
    __m256 longitude_difference = _mm256_sub_ps(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    __m256 one_minus_f = _mm256_sub_ps(one, flattening);
    __m256 tan_first = _mm256_div_ps(nk_f32x8_sin_haswell_(first_latitudes), nk_f32x8_cos_haswell_(first_latitudes));
    __m256 tan_second = _mm256_div_ps(nk_f32x8_sin_haswell_(second_latitudes), nk_f32x8_cos_haswell_(second_latitudes));
    __m256 tan_reduced_first = _mm256_mul_ps(one_minus_f, tan_first);
    __m256 tan_reduced_second = _mm256_mul_ps(one_minus_f, tan_second);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    __m256 cos_reduced_first = _mm256_div_ps(
        one, _mm256_sqrt_ps(_mm256_fmadd_ps(tan_reduced_first, tan_reduced_first, one)));
    __m256 sin_reduced_first = _mm256_mul_ps(tan_reduced_first, cos_reduced_first);
    __m256 cos_reduced_second = _mm256_div_ps(
        one, _mm256_sqrt_ps(_mm256_fmadd_ps(tan_reduced_second, tan_reduced_second, one)));
    __m256 sin_reduced_second = _mm256_mul_ps(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    __m256 lambda = longitude_difference;
    __m256 sin_angular_distance, cos_angular_distance, angular_distance;
    __m256 sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points using masks
    __m256 converged_mask = _mm256_setzero_ps();
    __m256 coincident_mask = _mm256_setzero_ps();

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        int converged_bits = _mm256_movemask_ps(converged_mask);
        if (converged_bits == 0xFF) break;

        __m256 sin_lambda = nk_f32x8_sin_haswell_(lambda);
        __m256 cos_lambda = nk_f32x8_cos_haswell_(lambda);

        // sin²(angular_distance) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        __m256 cross_term = _mm256_mul_ps(cos_reduced_second, sin_lambda);
        __m256 mixed_term = _mm256_sub_ps(
            _mm256_mul_ps(cos_reduced_first, sin_reduced_second),
            _mm256_mul_ps(_mm256_mul_ps(sin_reduced_first, cos_reduced_second), cos_lambda));
        __m256 sin_angular_dist_sq = _mm256_fmadd_ps(cross_term, cross_term, _mm256_mul_ps(mixed_term, mixed_term));
        sin_angular_distance = _mm256_sqrt_ps(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = _mm256_cmp_ps(sin_angular_distance, epsilon, _CMP_LT_OS);

        // cos(angular_distance) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance = _mm256_fmadd_ps(_mm256_mul_ps(cos_reduced_first, cos_reduced_second), cos_lambda,
                                               _mm256_mul_ps(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_f32x8_atan2_haswell_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance)
        // Avoid division by zero by using blending
        __m256 safe_sin_angular = _mm256_blendv_ps(sin_angular_distance, one, coincident_mask);
        sin_azimuth = _mm256_div_ps(_mm256_mul_ps(_mm256_mul_ps(cos_reduced_first, cos_reduced_second), sin_lambda),
                                    safe_sin_angular);
        cos_squared_azimuth = _mm256_sub_ps(one, _mm256_mul_ps(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α ≈ 0
        __m256 equatorial_mask = _mm256_cmp_ps(cos_squared_azimuth, epsilon, _CMP_LT_OS);
        __m256 safe_cos_sq_azimuth = _mm256_blendv_ps(cos_squared_azimuth, one, equatorial_mask);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        __m256 sin_product = _mm256_mul_ps(sin_reduced_first, sin_reduced_second);
        cos_double_angular_midpoint = _mm256_sub_ps(
            cos_angular_distance, _mm256_div_ps(_mm256_mul_ps(two, sin_product), safe_cos_sq_azimuth));
        cos_double_angular_midpoint = _mm256_blendv_ps(cos_double_angular_midpoint, _mm256_setzero_ps(),
                                                       equatorial_mask);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        __m256 correction_factor = _mm256_mul_ps(
            _mm256_div_ps(flattening, sixteen),
            _mm256_mul_ps(cos_squared_azimuth,
                          _mm256_fmadd_ps(flattening, _mm256_fnmadd_ps(three, cos_squared_azimuth, four), four)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        __m256 cos_2sm_sq = _mm256_mul_ps(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2 × cos²(2σₘ)
        __m256 innermost = _mm256_fmadd_ps(two, cos_2sm_sq, _mm256_set1_ps(-1.0f));
        // middle = cos(2σₘ) + C × cos(σ) × innermost
        __m256 middle = _mm256_fmadd_ps(_mm256_mul_ps(correction_factor, cos_angular_distance), innermost,
                                        cos_double_angular_midpoint);
        // inner = C × sin(σ) × middle
        __m256 inner = _mm256_mul_ps(_mm256_mul_ps(correction_factor, sin_angular_distance), middle);

        // λ' = L + (1-C) * f * sin_α * (σ + inner)
        __m256 lambda_new = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(one, correction_factor), flattening), sin_azimuth),
            _mm256_add_ps(angular_distance, inner), longitude_difference);

        // Check convergence: |λ - λ'| < threshold
        __m256 lambda_diff_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_sub_ps(lambda_new, lambda));
        __m256 newly_converged = _mm256_cmp_ps(lambda_diff_abs, convergence_threshold, _CMP_LT_OS);
        converged_mask = _mm256_or_ps(converged_mask, newly_converged);

        // Only update lambda for non-converged lanes
        lambda = _mm256_blendv_ps(lambda_new, lambda, converged_mask);
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    __m256 a_sq = _mm256_mul_ps(equatorial_radius, equatorial_radius);
    __m256 b_sq = _mm256_mul_ps(polar_radius, polar_radius);
    __m256 u_squared = _mm256_div_ps(_mm256_mul_ps(cos_squared_azimuth, _mm256_sub_ps(a_sq, b_sq)), b_sq);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    __m256 series_a = _mm256_fmadd_ps(u_squared, _mm256_set1_ps(-175.0f), _mm256_set1_ps(320.0f));
    series_a = _mm256_fmadd_ps(u_squared, series_a, _mm256_set1_ps(-768.0f));
    series_a = _mm256_fmadd_ps(u_squared, series_a, _mm256_set1_ps(4096.0f));
    series_a = _mm256_fmadd_ps(_mm256_div_ps(u_squared, _mm256_set1_ps(16384.0f)), series_a, one);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    __m256 series_b = _mm256_fmadd_ps(u_squared, _mm256_set1_ps(-47.0f), _mm256_set1_ps(74.0f));
    series_b = _mm256_fmadd_ps(u_squared, series_b, _mm256_set1_ps(-128.0f));
    series_b = _mm256_fmadd_ps(u_squared, series_b, _mm256_set1_ps(256.0f));
    series_b = _mm256_mul_ps(_mm256_div_ps(u_squared, _mm256_set1_ps(1024.0f)), series_b);

    // Δσ = B × sin(σ) × (cos(2σₘ) +
    //      B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m256 cos_2sm_sq = _mm256_mul_ps(cos_double_angular_midpoint, cos_double_angular_midpoint);
    __m256 sin_sq = _mm256_mul_ps(sin_angular_distance, sin_angular_distance);
    __m256 term1 = _mm256_fmadd_ps(two, cos_2sm_sq, _mm256_set1_ps(-1.0f));
    term1 = _mm256_mul_ps(cos_angular_distance, term1);
    __m256 term2 = _mm256_fmadd_ps(four, sin_sq, _mm256_set1_ps(-3.0f));
    __m256 term3 = _mm256_fmadd_ps(four, cos_2sm_sq, _mm256_set1_ps(-3.0f));
    term2 = _mm256_mul_ps(_mm256_mul_ps(_mm256_div_ps(series_b, six), cos_double_angular_midpoint),
                          _mm256_mul_ps(term2, term3));
    __m256 delta_sigma = _mm256_mul_ps(
        series_b, _mm256_mul_ps(sin_angular_distance, _mm256_add_ps(cos_double_angular_midpoint,
                                                                    _mm256_mul_ps(_mm256_div_ps(series_b, four),
                                                                                  _mm256_sub_ps(term1, term2)))));

    // s = b * A * (σ - Δσ)
    __m256 distances = _mm256_mul_ps(_mm256_mul_ps(polar_radius, series_a),
                                     _mm256_sub_ps(angular_distance, delta_sigma));

    // Set coincident points to zero
    distances = _mm256_blendv_ps(distances, _mm256_setzero_ps(), coincident_mask);

    return distances;
}

NK_PUBLIC void nk_vincenty_f32_haswell(             //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 8) {
        __m256 first_latitudes = _mm256_loadu_ps(a_lats);
        __m256 first_longitudes = _mm256_loadu_ps(a_lons);
        __m256 second_latitudes = _mm256_loadu_ps(b_lats);
        __m256 second_longitudes = _mm256_loadu_ps(b_lons);

        __m256 distances = nk_vincenty_f32x8_haswell_(first_latitudes, first_longitudes, second_latitudes,
                                                      second_longitudes);
        _mm256_storeu_ps(results, distances);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with partial loads (n can be 1-7 here)
    if (n > 0) {
        nk_b256_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x8_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x8_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x8_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x8_serial_(b_lons, &b_lon_vec, n);
        __m256 distances = nk_vincenty_f32x8_haswell_(a_lat_vec.ymm_ps, a_lon_vec.ymm_ps, b_lat_vec.ymm_ps,
                                                      b_lon_vec.ymm_ps);
        result_vec.ymm_ps = distances;
        nk_partial_store_b32x8_serial_(&result_vec, results, n);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_

#if NK_TARGET_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

NK_INTERNAL __m512d nk_haversine_f64x8_skylake_(       //
    __m512d first_latitudes, __m512d first_longitudes, //
    __m512d second_latitudes, __m512d second_longitudes) {

    __m512d const earth_radius = _mm512_set1_pd(NK_EARTH_MEDIATORIAL_RADIUS);
    __m512d const half = _mm512_set1_pd(0.5);
    __m512d const one = _mm512_set1_pd(1.0);
    __m512d const two = _mm512_set1_pd(2.0);

    __m512d latitude_delta = _mm512_sub_pd(second_latitudes, first_latitudes);
    __m512d longitude_delta = _mm512_sub_pd(second_longitudes, first_longitudes);

    // Haversine terms: sin²(Δ/2)
    __m512d latitude_delta_half = _mm512_mul_pd(latitude_delta, half);
    __m512d longitude_delta_half = _mm512_mul_pd(longitude_delta, half);
    __m512d sin_latitude_delta_half = nk_f64x8_sin_skylake_(latitude_delta_half);
    __m512d sin_longitude_delta_half = nk_f64x8_sin_skylake_(longitude_delta_half);
    __m512d sin_squared_latitude_delta_half = _mm512_mul_pd(sin_latitude_delta_half, sin_latitude_delta_half);
    __m512d sin_squared_longitude_delta_half = _mm512_mul_pd(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    __m512d cos_first_latitude = nk_f64x8_cos_skylake_(first_latitudes);
    __m512d cos_second_latitude = nk_f64x8_cos_skylake_(second_latitudes);
    __m512d cos_latitude_product = _mm512_mul_pd(cos_first_latitude, cos_second_latitude);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    __m512d haversine_term = _mm512_add_pd(sin_squared_latitude_delta_half,
                                           _mm512_mul_pd(cos_latitude_product, sin_squared_longitude_delta_half));
    // Clamp haversine_term to [0, 1] to prevent NaN from sqrt of negative values
    __m512d zero = _mm512_setzero_pd();
    haversine_term = _mm512_max_pd(zero, _mm512_min_pd(one, haversine_term));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    __m512d sqrt_haversine = _mm512_sqrt_pd(haversine_term);
    __m512d sqrt_complement = _mm512_sqrt_pd(_mm512_sub_pd(one, haversine_term));
    __m512d central_angle = _mm512_mul_pd(two, nk_f64x8_atan2_skylake_(sqrt_haversine, sqrt_complement));

    return _mm512_mul_pd(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f64_skylake(            //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 8) {
        __m512d first_latitudes = _mm512_loadu_pd(a_lats);
        __m512d first_longitudes = _mm512_loadu_pd(a_lons);
        __m512d second_latitudes = _mm512_loadu_pd(b_lats);
        __m512d second_longitudes = _mm512_loadu_pd(b_lons);

        __m512d distances = nk_haversine_f64x8_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        _mm512_storeu_pd(results, distances);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with masked operations
    if (n > 0) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, n);
        __m512d first_latitudes = _mm512_maskz_loadu_pd(mask, a_lats);
        __m512d first_longitudes = _mm512_maskz_loadu_pd(mask, a_lons);
        __m512d second_latitudes = _mm512_maskz_loadu_pd(mask, b_lats);
        __m512d second_longitudes = _mm512_maskz_loadu_pd(mask, b_lons);

        __m512d distances = nk_haversine_f64x8_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        _mm512_mask_storeu_pd(results, mask, distances);
    }
}

/**
 *  @brief  AVX-512 helper for Vincenty's geodesic distance on 8 f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking.
 */
NK_INTERNAL __m512d nk_vincenty_f64x8_skylake_(        //
    __m512d first_latitudes, __m512d first_longitudes, //
    __m512d second_latitudes, __m512d second_longitudes) {

    __m512d const equatorial_radius = _mm512_set1_pd(NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m512d const polar_radius = _mm512_set1_pd(NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m512d const flattening = _mm512_set1_pd(1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m512d const convergence_threshold = _mm512_set1_pd(NK_VINCENTY_CONVERGENCE_THRESHOLD);
    __m512d const one = _mm512_set1_pd(1.0);
    __m512d const two = _mm512_set1_pd(2.0);
    __m512d const three = _mm512_set1_pd(3.0);
    __m512d const four = _mm512_set1_pd(4.0);
    __m512d const six = _mm512_set1_pd(6.0);
    __m512d const sixteen = _mm512_set1_pd(16.0);

    // Longitude difference
    __m512d longitude_difference = _mm512_sub_pd(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    __m512d one_minus_f = _mm512_sub_pd(one, flattening);
    __m512d tan_first = _mm512_div_pd(nk_f64x8_sin_skylake_(first_latitudes), nk_f64x8_cos_skylake_(first_latitudes));
    __m512d tan_second = _mm512_div_pd(nk_f64x8_sin_skylake_(second_latitudes),
                                       nk_f64x8_cos_skylake_(second_latitudes));
    __m512d tan_reduced_first = _mm512_mul_pd(one_minus_f, tan_first);
    __m512d tan_reduced_second = _mm512_mul_pd(one_minus_f, tan_second);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    __m512d cos_reduced_first = _mm512_div_pd(
        one, _mm512_sqrt_pd(_mm512_fmadd_pd(tan_reduced_first, tan_reduced_first, one)));
    __m512d sin_reduced_first = _mm512_mul_pd(tan_reduced_first, cos_reduced_first);
    __m512d cos_reduced_second = _mm512_div_pd(
        one, _mm512_sqrt_pd(_mm512_fmadd_pd(tan_reduced_second, tan_reduced_second, one)));
    __m512d sin_reduced_second = _mm512_mul_pd(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    __m512d lambda = longitude_difference;
    __m512d sin_angular_distance, cos_angular_distance, angular_distance;
    __m512d sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points
    __mmask8 converged_mask = 0;
    __mmask8 coincident_mask = 0;

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS && converged_mask != 0xFF; ++iteration) {
        __m512d sin_lambda = nk_f64x8_sin_skylake_(lambda);
        __m512d cos_lambda = nk_f64x8_cos_skylake_(lambda);

        // sin²(angular_distance) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        __m512d cross_term = _mm512_mul_pd(cos_reduced_second, sin_lambda);
        __m512d mixed_term = _mm512_sub_pd(
            _mm512_mul_pd(cos_reduced_first, sin_reduced_second),
            _mm512_mul_pd(_mm512_mul_pd(sin_reduced_first, cos_reduced_second), cos_lambda));
        __m512d sin_angular_dist_sq = _mm512_fmadd_pd(cross_term, cross_term, _mm512_mul_pd(mixed_term, mixed_term));
        sin_angular_distance = _mm512_sqrt_pd(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = _mm512_cmp_pd_mask(sin_angular_distance, _mm512_set1_pd(1e-15), _CMP_LT_OS);

        // cos(angular_distance) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance = _mm512_fmadd_pd(_mm512_mul_pd(cos_reduced_first, cos_reduced_second), cos_lambda,
                                               _mm512_mul_pd(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_f64x8_atan2_skylake_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance)
        sin_azimuth = _mm512_div_pd(_mm512_mul_pd(_mm512_mul_pd(cos_reduced_first, cos_reduced_second), sin_lambda),
                                    sin_angular_distance);
        cos_squared_azimuth = _mm512_sub_pd(one, _mm512_mul_pd(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α = 0
        __mmask8 equatorial_mask = _mm512_cmp_pd_mask(cos_squared_azimuth, _mm512_set1_pd(1e-15), _CMP_LT_OS);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        __m512d sin_product = _mm512_mul_pd(sin_reduced_first, sin_reduced_second);
        cos_double_angular_midpoint = _mm512_sub_pd(
            cos_angular_distance, _mm512_div_pd(_mm512_mul_pd(two, sin_product), cos_squared_azimuth));
        cos_double_angular_midpoint = _mm512_mask_blend_pd(equatorial_mask, cos_double_angular_midpoint,
                                                           _mm512_setzero_pd());

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        __m512d correction_factor = _mm512_mul_pd(
            _mm512_div_pd(flattening, sixteen),
            _mm512_mul_pd(cos_squared_azimuth,
                          _mm512_fmadd_pd(flattening, _mm512_fnmadd_pd(three, cos_squared_azimuth, four), four)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        __m512d cos_2sm_sq = _mm512_mul_pd(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2 × cos²(2σₘ)
        __m512d innermost = _mm512_fmadd_pd(two, cos_2sm_sq, _mm512_set1_pd(-1.0));
        // middle = cos(2σₘ) + C × cos(σ) × innermost
        __m512d middle = _mm512_fmadd_pd(_mm512_mul_pd(correction_factor, cos_angular_distance), innermost,
                                         cos_double_angular_midpoint);
        // inner = C × sin(σ) × middle
        __m512d inner = _mm512_mul_pd(_mm512_mul_pd(correction_factor, sin_angular_distance), middle);

        // λ' = L + (1-C) * f * sin_α * (σ + inner)
        __m512d lambda_new = _mm512_fmadd_pd(
            _mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(one, correction_factor), flattening), sin_azimuth),
            _mm512_add_pd(angular_distance, inner), longitude_difference);

        // Check convergence: |λ - λ'| < threshold
        __m512d lambda_diff = _mm512_abs_pd(_mm512_sub_pd(lambda_new, lambda));
        converged_mask = _mm512_cmp_pd_mask(lambda_diff, convergence_threshold, _CMP_LT_OS);

        lambda = lambda_new;
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    __m512d a_sq = _mm512_mul_pd(equatorial_radius, equatorial_radius);
    __m512d b_sq = _mm512_mul_pd(polar_radius, polar_radius);
    __m512d u_squared = _mm512_div_pd(_mm512_mul_pd(cos_squared_azimuth, _mm512_sub_pd(a_sq, b_sq)), b_sq);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    __m512d series_a = _mm512_fmadd_pd(u_squared, _mm512_set1_pd(-175.0), _mm512_set1_pd(320.0));
    series_a = _mm512_fmadd_pd(u_squared, series_a, _mm512_set1_pd(-768.0));
    series_a = _mm512_fmadd_pd(u_squared, series_a, _mm512_set1_pd(4096.0));
    series_a = _mm512_fmadd_pd(_mm512_div_pd(u_squared, _mm512_set1_pd(16384.0)), series_a, one);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    __m512d series_b = _mm512_fmadd_pd(u_squared, _mm512_set1_pd(-47.0), _mm512_set1_pd(74.0));
    series_b = _mm512_fmadd_pd(u_squared, series_b, _mm512_set1_pd(-128.0));
    series_b = _mm512_fmadd_pd(u_squared, series_b, _mm512_set1_pd(256.0));
    series_b = _mm512_mul_pd(_mm512_div_pd(u_squared, _mm512_set1_pd(1024.0)), series_b);

    // Δσ = B × sin(σ) × (cos(2σₘ) +
    //      B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m512d cos_2sm_sq = _mm512_mul_pd(cos_double_angular_midpoint, cos_double_angular_midpoint);
    __m512d sin_sq = _mm512_mul_pd(sin_angular_distance, sin_angular_distance);
    __m512d term1 = _mm512_fmadd_pd(two, cos_2sm_sq, _mm512_set1_pd(-1.0));
    term1 = _mm512_mul_pd(cos_angular_distance, term1);
    __m512d term2 = _mm512_fmadd_pd(four, sin_sq, _mm512_set1_pd(-3.0));
    __m512d term3 = _mm512_fmadd_pd(four, cos_2sm_sq, _mm512_set1_pd(-3.0));
    term2 = _mm512_mul_pd(_mm512_mul_pd(_mm512_div_pd(series_b, six), cos_double_angular_midpoint),
                          _mm512_mul_pd(term2, term3));
    __m512d delta_sigma = _mm512_mul_pd(
        series_b, _mm512_mul_pd(sin_angular_distance, _mm512_add_pd(cos_double_angular_midpoint,
                                                                    _mm512_mul_pd(_mm512_div_pd(series_b, four),
                                                                                  _mm512_sub_pd(term1, term2)))));

    // s = b * A * (σ - Δσ)
    __m512d distances = _mm512_mul_pd(_mm512_mul_pd(polar_radius, series_a),
                                      _mm512_sub_pd(angular_distance, delta_sigma));

    // Set coincident points to zero
    distances = _mm512_mask_blend_pd(coincident_mask, distances, _mm512_setzero_pd());

    return distances;
}

NK_PUBLIC void nk_vincenty_f64_skylake(             //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 8) {
        __m512d first_latitudes = _mm512_loadu_pd(a_lats);
        __m512d first_longitudes = _mm512_loadu_pd(a_lons);
        __m512d second_latitudes = _mm512_loadu_pd(b_lats);
        __m512d second_longitudes = _mm512_loadu_pd(b_lons);

        __m512d distances = nk_vincenty_f64x8_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                       second_longitudes);
        _mm512_storeu_pd(results, distances);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with masked operations
    if (n > 0) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, n);
        __m512d first_latitudes = _mm512_maskz_loadu_pd(mask, a_lats);
        __m512d first_longitudes = _mm512_maskz_loadu_pd(mask, a_lons);
        __m512d second_latitudes = _mm512_maskz_loadu_pd(mask, b_lats);
        __m512d second_longitudes = _mm512_maskz_loadu_pd(mask, b_lons);

        __m512d distances = nk_vincenty_f64x8_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                       second_longitudes);
        _mm512_mask_storeu_pd(results, mask, distances);
    }
}

NK_INTERNAL __m512 nk_haversine_f32x16_skylake_(     //
    __m512 first_latitudes, __m512 first_longitudes, //
    __m512 second_latitudes, __m512 second_longitudes) {

    __m512 const earth_radius = _mm512_set1_ps((float)NK_EARTH_MEDIATORIAL_RADIUS);
    __m512 const half = _mm512_set1_ps(0.5f);
    __m512 const one = _mm512_set1_ps(1.0f);
    __m512 const two = _mm512_set1_ps(2.0f);

    __m512 latitude_delta = _mm512_sub_ps(second_latitudes, first_latitudes);
    __m512 longitude_delta = _mm512_sub_ps(second_longitudes, first_longitudes);

    // Haversine terms: sin²(Δ/2)
    __m512 latitude_delta_half = _mm512_mul_ps(latitude_delta, half);
    __m512 longitude_delta_half = _mm512_mul_ps(longitude_delta, half);
    __m512 sin_latitude_delta_half = nk_f32x16_sin_skylake_(latitude_delta_half);
    __m512 sin_longitude_delta_half = nk_f32x16_sin_skylake_(longitude_delta_half);
    __m512 sin_squared_latitude_delta_half = _mm512_mul_ps(sin_latitude_delta_half, sin_latitude_delta_half);
    __m512 sin_squared_longitude_delta_half = _mm512_mul_ps(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    __m512 cos_first_latitude = nk_f32x16_cos_skylake_(first_latitudes);
    __m512 cos_second_latitude = nk_f32x16_cos_skylake_(second_latitudes);
    __m512 cos_latitude_product = _mm512_mul_ps(cos_first_latitude, cos_second_latitude);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    __m512 haversine_term = _mm512_add_ps(sin_squared_latitude_delta_half,
                                          _mm512_mul_ps(cos_latitude_product, sin_squared_longitude_delta_half));

    // Clamp to [0, 1] to avoid NaN from sqrt of negative numbers (due to floating point errors)
    __m512 zero = _mm512_setzero_ps();
    haversine_term = _mm512_max_ps(zero, _mm512_min_ps(one, haversine_term));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    __m512 sqrt_haversine = _mm512_sqrt_ps(haversine_term);
    __m512 sqrt_complement = _mm512_sqrt_ps(_mm512_sub_ps(one, haversine_term));
    __m512 central_angle = _mm512_mul_ps(two, nk_f32x16_atan2_skylake_(sqrt_haversine, sqrt_complement));

    return _mm512_mul_ps(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f32_skylake(            //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 16) {
        __m512 first_latitudes = _mm512_loadu_ps(a_lats);
        __m512 first_longitudes = _mm512_loadu_ps(a_lons);
        __m512 second_latitudes = _mm512_loadu_ps(b_lats);
        __m512 second_longitudes = _mm512_loadu_ps(b_lons);

        __m512 distances = nk_haversine_f32x16_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        _mm512_storeu_ps(results, distances);

        a_lats += 16, a_lons += 16, b_lats += 16, b_lons += 16, results += 16, n -= 16;
    }

    // Handle remaining elements with masked operations
    if (n > 0) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        __m512 first_latitudes = _mm512_maskz_loadu_ps(mask, a_lats);
        __m512 first_longitudes = _mm512_maskz_loadu_ps(mask, a_lons);
        __m512 second_latitudes = _mm512_maskz_loadu_ps(mask, b_lats);
        __m512 second_longitudes = _mm512_maskz_loadu_ps(mask, b_lons);

        __m512 distances = nk_haversine_f32x16_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        _mm512_mask_storeu_ps(results, mask, distances);
    }
}

/**
 *  @brief  AVX-512 helper for Vincenty's geodesic distance on 16 f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking.
 */
NK_INTERNAL __m512 nk_vincenty_f32x16_skylake_(      //
    __m512 first_latitudes, __m512 first_longitudes, //
    __m512 second_latitudes, __m512 second_longitudes) {

    __m512 const equatorial_radius = _mm512_set1_ps((float)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m512 const polar_radius = _mm512_set1_ps((float)NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m512 const flattening = _mm512_set1_ps(1.0f / (float)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m512 const convergence_threshold = _mm512_set1_ps((float)NK_VINCENTY_CONVERGENCE_THRESHOLD);
    __m512 const one = _mm512_set1_ps(1.0f);
    __m512 const two = _mm512_set1_ps(2.0f);
    __m512 const three = _mm512_set1_ps(3.0f);
    __m512 const four = _mm512_set1_ps(4.0f);
    __m512 const six = _mm512_set1_ps(6.0f);
    __m512 const sixteen = _mm512_set1_ps(16.0f);

    // Longitude difference
    __m512 longitude_difference = _mm512_sub_ps(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    __m512 one_minus_f = _mm512_sub_ps(one, flattening);
    __m512 tan_first = _mm512_div_ps(nk_f32x16_sin_skylake_(first_latitudes), nk_f32x16_cos_skylake_(first_latitudes));
    __m512 tan_second = _mm512_div_ps(nk_f32x16_sin_skylake_(second_latitudes),
                                      nk_f32x16_cos_skylake_(second_latitudes));
    __m512 tan_reduced_first = _mm512_mul_ps(one_minus_f, tan_first);
    __m512 tan_reduced_second = _mm512_mul_ps(one_minus_f, tan_second);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    __m512 cos_reduced_first = _mm512_div_ps(
        one, _mm512_sqrt_ps(_mm512_fmadd_ps(tan_reduced_first, tan_reduced_first, one)));
    __m512 sin_reduced_first = _mm512_mul_ps(tan_reduced_first, cos_reduced_first);
    __m512 cos_reduced_second = _mm512_div_ps(
        one, _mm512_sqrt_ps(_mm512_fmadd_ps(tan_reduced_second, tan_reduced_second, one)));
    __m512 sin_reduced_second = _mm512_mul_ps(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    __m512 lambda = longitude_difference;
    __m512 sin_angular_distance, cos_angular_distance, angular_distance;
    __m512 sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points
    __mmask16 converged_mask = 0;
    __mmask16 coincident_mask = 0;

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS && converged_mask != 0xFFFF; ++iteration) {
        __m512 sin_lambda = nk_f32x16_sin_skylake_(lambda);
        __m512 cos_lambda = nk_f32x16_cos_skylake_(lambda);

        // sin²(angular_distance) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        __m512 cross_term = _mm512_mul_ps(cos_reduced_second, sin_lambda);
        __m512 mixed_term = _mm512_sub_ps(
            _mm512_mul_ps(cos_reduced_first, sin_reduced_second),
            _mm512_mul_ps(_mm512_mul_ps(sin_reduced_first, cos_reduced_second), cos_lambda));
        __m512 sin_angular_dist_sq = _mm512_fmadd_ps(cross_term, cross_term, _mm512_mul_ps(mixed_term, mixed_term));
        sin_angular_distance = _mm512_sqrt_ps(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = _mm512_cmp_ps_mask(sin_angular_distance, _mm512_set1_ps(1e-7f), _CMP_LT_OS);

        // cos(angular_distance) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance = _mm512_fmadd_ps(_mm512_mul_ps(cos_reduced_first, cos_reduced_second), cos_lambda,
                                               _mm512_mul_ps(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_f32x16_atan2_skylake_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance)
        sin_azimuth = _mm512_div_ps(_mm512_mul_ps(_mm512_mul_ps(cos_reduced_first, cos_reduced_second), sin_lambda),
                                    sin_angular_distance);
        cos_squared_azimuth = _mm512_sub_ps(one, _mm512_mul_ps(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α = 0
        __mmask16 equatorial_mask = _mm512_cmp_ps_mask(cos_squared_azimuth, _mm512_set1_ps(1e-7f), _CMP_LT_OS);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        __m512 sin_product = _mm512_mul_ps(sin_reduced_first, sin_reduced_second);
        cos_double_angular_midpoint = _mm512_sub_ps(
            cos_angular_distance, _mm512_div_ps(_mm512_mul_ps(two, sin_product), cos_squared_azimuth));
        cos_double_angular_midpoint = _mm512_mask_blend_ps(equatorial_mask, cos_double_angular_midpoint,
                                                           _mm512_setzero_ps());

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        __m512 correction_factor = _mm512_mul_ps(
            _mm512_div_ps(flattening, sixteen),
            _mm512_mul_ps(cos_squared_azimuth,
                          _mm512_fmadd_ps(flattening, _mm512_fnmadd_ps(three, cos_squared_azimuth, four), four)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        __m512 cos_2sm_sq = _mm512_mul_ps(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2 × cos²(2σₘ)
        __m512 innermost = _mm512_fmadd_ps(two, cos_2sm_sq, _mm512_set1_ps(-1.0f));
        // middle = cos(2σₘ) + C × cos(σ) × innermost
        __m512 middle = _mm512_fmadd_ps(_mm512_mul_ps(correction_factor, cos_angular_distance), innermost,
                                        cos_double_angular_midpoint);
        // inner = C × sin(σ) × middle
        __m512 inner = _mm512_mul_ps(_mm512_mul_ps(correction_factor, sin_angular_distance), middle);

        // λ' = L + (1-C) * f * sin_α * (σ + inner)
        __m512 lambda_new = _mm512_fmadd_ps(
            _mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(one, correction_factor), flattening), sin_azimuth),
            _mm512_add_ps(angular_distance, inner), longitude_difference);

        // Check convergence: |λ - λ'| < threshold
        __m512 lambda_diff = _mm512_abs_ps(_mm512_sub_ps(lambda_new, lambda));
        converged_mask = _mm512_cmp_ps_mask(lambda_diff, convergence_threshold, _CMP_LT_OS);

        lambda = lambda_new;
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    __m512 a_sq = _mm512_mul_ps(equatorial_radius, equatorial_radius);
    __m512 b_sq = _mm512_mul_ps(polar_radius, polar_radius);
    __m512 u_squared = _mm512_div_ps(_mm512_mul_ps(cos_squared_azimuth, _mm512_sub_ps(a_sq, b_sq)), b_sq);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    __m512 series_a = _mm512_fmadd_ps(u_squared, _mm512_set1_ps(-175.0f), _mm512_set1_ps(320.0f));
    series_a = _mm512_fmadd_ps(u_squared, series_a, _mm512_set1_ps(-768.0f));
    series_a = _mm512_fmadd_ps(u_squared, series_a, _mm512_set1_ps(4096.0f));
    series_a = _mm512_fmadd_ps(_mm512_div_ps(u_squared, _mm512_set1_ps(16384.0f)), series_a, one);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    __m512 series_b = _mm512_fmadd_ps(u_squared, _mm512_set1_ps(-47.0f), _mm512_set1_ps(74.0f));
    series_b = _mm512_fmadd_ps(u_squared, series_b, _mm512_set1_ps(-128.0f));
    series_b = _mm512_fmadd_ps(u_squared, series_b, _mm512_set1_ps(256.0f));
    series_b = _mm512_mul_ps(_mm512_div_ps(u_squared, _mm512_set1_ps(1024.0f)), series_b);

    // Δσ = B × sin(σ) × (cos(2σₘ) +
    //      B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m512 cos_2sm_sq = _mm512_mul_ps(cos_double_angular_midpoint, cos_double_angular_midpoint);
    __m512 sin_sq = _mm512_mul_ps(sin_angular_distance, sin_angular_distance);
    __m512 term1 = _mm512_fmadd_ps(two, cos_2sm_sq, _mm512_set1_ps(-1.0f));
    term1 = _mm512_mul_ps(cos_angular_distance, term1);
    __m512 term2 = _mm512_fmadd_ps(four, sin_sq, _mm512_set1_ps(-3.0f));
    __m512 term3 = _mm512_fmadd_ps(four, cos_2sm_sq, _mm512_set1_ps(-3.0f));
    term2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_div_ps(series_b, six), cos_double_angular_midpoint),
                          _mm512_mul_ps(term2, term3));
    __m512 delta_sigma = _mm512_mul_ps(
        series_b, _mm512_mul_ps(sin_angular_distance, _mm512_add_ps(cos_double_angular_midpoint,
                                                                    _mm512_mul_ps(_mm512_div_ps(series_b, four),
                                                                                  _mm512_sub_ps(term1, term2)))));

    // s = b * A * (σ - Δσ)
    __m512 distances = _mm512_mul_ps(_mm512_mul_ps(polar_radius, series_a),
                                     _mm512_sub_ps(angular_distance, delta_sigma));

    // Set coincident points to zero
    distances = _mm512_mask_blend_ps(coincident_mask, distances, _mm512_setzero_ps());

    return distances;
}

NK_PUBLIC void nk_vincenty_f32_skylake(             //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 16) {
        __m512 first_latitudes = _mm512_loadu_ps(a_lats);
        __m512 first_longitudes = _mm512_loadu_ps(a_lons);
        __m512 second_latitudes = _mm512_loadu_ps(b_lats);
        __m512 second_longitudes = _mm512_loadu_ps(b_lons);

        __m512 distances = nk_vincenty_f32x16_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                       second_longitudes);
        _mm512_storeu_ps(results, distances);

        a_lats += 16, a_lons += 16, b_lats += 16, b_lons += 16, results += 16, n -= 16;
    }

    // Handle remaining elements with masked operations
    if (n > 0) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        __m512 first_latitudes = _mm512_maskz_loadu_ps(mask, a_lats);
        __m512 first_longitudes = _mm512_maskz_loadu_ps(mask, a_lons);
        __m512 second_latitudes = _mm512_maskz_loadu_ps(mask, b_lats);
        __m512 second_longitudes = _mm512_maskz_loadu_ps(mask, b_lons);

        __m512 distances = nk_vincenty_f32x16_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                       second_longitudes);
        _mm512_mask_storeu_ps(results, mask, distances);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SKYLAKE

/**
 *  @brief  Returns the output dtype for Haversine distance.
 */
NK_INTERNAL nk_dtype_t nk_haversine_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief  Returns the output dtype for Vincenty distance.
 */
NK_INTERNAL nk_dtype_t nk_vincenty_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_haversine_f64(                    //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {
#if NK_TARGET_SKYLAKE
    nk_haversine_f64_skylake(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_HASWELL
    nk_haversine_f64_haswell(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_NEON
    nk_haversine_f64_neon(a_lats, a_lons, b_lats, b_lons, n, results);
#else
    nk_haversine_f64_serial(a_lats, a_lons, b_lats, b_lons, n, results);
#endif
}

NK_PUBLIC void nk_haversine_f32(                    //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {
#if NK_TARGET_SKYLAKE
    nk_haversine_f32_skylake(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_HASWELL
    nk_haversine_f32_haswell(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_NEON
    nk_haversine_f32_neon(a_lats, a_lons, b_lats, b_lons, n, results);
#else
    nk_haversine_f32_serial(a_lats, a_lons, b_lats, b_lons, n, results);
#endif
}

NK_PUBLIC void nk_vincenty_f64(                     //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {
#if NK_TARGET_SKYLAKE
    nk_vincenty_f64_skylake(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_HASWELL
    nk_vincenty_f64_haswell(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_NEON
    nk_vincenty_f64_neon(a_lats, a_lons, b_lats, b_lons, n, results);
#else
    nk_vincenty_f64_serial(a_lats, a_lons, b_lats, b_lons, n, results);
#endif
}

NK_PUBLIC void nk_vincenty_f32(                     //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {
#if NK_TARGET_SKYLAKE
    nk_vincenty_f32_skylake(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_HASWELL
    nk_vincenty_f32_haswell(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_NEON
    nk_vincenty_f32_neon(a_lats, a_lons, b_lats, b_lons, n, results);
#else
    nk_vincenty_f32_serial(a_lats, a_lons, b_lats, b_lons, n, results);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
