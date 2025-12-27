/**
 *  @brief SIMD-accelerated Geo-Spatial distance functions.
 *  @file include/simsimd/geospatial.h
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
 *  For datatypes:
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
 *  The very last part of the computation applies `asin(sqrt(x))` non-linear transformation.
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
 *  SimSIMD uses the newer & more accurate @b IERS-2003 standard, but allows overriding default parameters:
 *
 *      #define SIMSIMD_EARTH_ELLIPSOID_EQUATORIAL_RADIUS (6378136.6)
 *      #define SIMSIMD_EARTH_ELLIPSOID_POLAR_RADIUS (6356751.9)
 *      #define SIMSIMD_EARTH_ELLIPSOID_INVERSE_FLATTENING (298.25642)
 *
 *  To revert from oblate spheroids to spheres, use `SIMSIMD_EARTH_MEDIATORIAL_RADIUS`.
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
SIMSIMD_DYNAMIC void simsimd_haversine_f64(                   //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results);

/** @copydoc simsimd_haversine_f64 */
SIMSIMD_DYNAMIC void simsimd_haversine_f32(                   //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results);

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
 *  @note Uses the Earth ellipsoid parameters configured via `SIMSIMD_EARTH_ELLIPSOID_*`.
 */
SIMSIMD_DYNAMIC void simsimd_vincenty_f64(                    //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results);

/** @copydoc simsimd_vincenty_f64 */
SIMSIMD_DYNAMIC void simsimd_vincenty_f32(                    //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results);

/** @copydoc simsimd_haversine_f64 */
SIMSIMD_PUBLIC void simsimd_haversine_f64_serial(             //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results);
/** @copydoc simsimd_vincenty_f64 */
SIMSIMD_PUBLIC void simsimd_vincenty_f64_serial(              //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results);
/** @copydoc simsimd_haversine_f32 */
SIMSIMD_PUBLIC void simsimd_haversine_f32_serial(             //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results);
/** @copydoc simsimd_vincenty_f32 */
SIMSIMD_PUBLIC void simsimd_vincenty_f32_serial(              //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results);

#if SIMSIMD_TARGET_NEON
/** @copydoc simsimd_haversine_f64 */
SIMSIMD_PUBLIC void simsimd_haversine_f64_neon(               //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results);
/** @copydoc simsimd_vincenty_f64 */
SIMSIMD_PUBLIC void simsimd_vincenty_f64_neon(                //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results);
/** @copydoc simsimd_haversine_f32 */
SIMSIMD_PUBLIC void simsimd_haversine_f32_neon(               //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results);
/** @copydoc simsimd_vincenty_f32 */
SIMSIMD_PUBLIC void simsimd_vincenty_f32_neon(                //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results);
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_HASWELL
/** @copydoc simsimd_haversine_f64 */
SIMSIMD_PUBLIC void simsimd_haversine_f64_haswell(            //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results);
/** @copydoc simsimd_vincenty_f64 */
SIMSIMD_PUBLIC void simsimd_vincenty_f64_haswell(             //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results);
/** @copydoc simsimd_haversine_f32 */
SIMSIMD_PUBLIC void simsimd_haversine_f32_haswell(            //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results);
/** @copydoc simsimd_vincenty_f32 */
SIMSIMD_PUBLIC void simsimd_vincenty_f32_haswell(             //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results);
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
/** @copydoc simsimd_haversine_f64 */
SIMSIMD_PUBLIC void simsimd_haversine_f64_skylake(            //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results);
/** @copydoc simsimd_vincenty_f64 */
SIMSIMD_PUBLIC void simsimd_vincenty_f64_skylake(             //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results);
/** @copydoc simsimd_haversine_f32 */
SIMSIMD_PUBLIC void simsimd_haversine_f32_skylake(            //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results);
/** @copydoc simsimd_vincenty_f32 */
SIMSIMD_PUBLIC void simsimd_vincenty_f32_skylake(             //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results);
#endif // SIMSIMD_TARGET_SKYLAKE

/*  Serial implementations of geospatial distance functions.
 *  These use the trigonometric functions from trigonometry.h for sin, cos, and atan2.
 */

SIMSIMD_PUBLIC void simsimd_haversine_f64_serial(             //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results) {

    simsimd_f64_t const earth_radius = SIMSIMD_EARTH_MEDIATORIAL_RADIUS;

    for (simsimd_size_t i = 0; i != n; ++i) {
        simsimd_f64_t first_latitude = a_lats[i];
        simsimd_f64_t first_longitude = a_lons[i];
        simsimd_f64_t second_latitude = b_lats[i];
        simsimd_f64_t second_longitude = b_lons[i];

        simsimd_f64_t latitude_delta = second_latitude - first_latitude;
        simsimd_f64_t longitude_delta = second_longitude - first_longitude;

        // Haversine formula: a = sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2)
        simsimd_f64_t sin_latitude_delta_half = simsimd_f64_sin(latitude_delta * 0.5);
        simsimd_f64_t sin_longitude_delta_half = simsimd_f64_sin(longitude_delta * 0.5);
        simsimd_f64_t cos_first_latitude = simsimd_f64_cos(first_latitude);
        simsimd_f64_t cos_second_latitude = simsimd_f64_cos(second_latitude);

        simsimd_f64_t haversine_term = sin_latitude_delta_half * sin_latitude_delta_half +
                                       cos_first_latitude * cos_second_latitude * sin_longitude_delta_half *
                                           sin_longitude_delta_half;

        // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
        simsimd_f64_t sqrt_haversine = SIMSIMD_F64_SQRT(haversine_term);
        simsimd_f64_t sqrt_complement = SIMSIMD_F64_SQRT(1.0 - haversine_term);
        simsimd_f64_t central_angle = 2.0 * simsimd_f64_atan2(sqrt_haversine, sqrt_complement);

        results[i] = earth_radius * central_angle;
    }
}

SIMSIMD_PUBLIC void simsimd_haversine_f32_serial(             //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results) {

    simsimd_f32_t const earth_radius = (simsimd_f32_t)SIMSIMD_EARTH_MEDIATORIAL_RADIUS;

    for (simsimd_size_t i = 0; i != n; ++i) {
        simsimd_f32_t first_latitude = a_lats[i];
        simsimd_f32_t first_longitude = a_lons[i];
        simsimd_f32_t second_latitude = b_lats[i];
        simsimd_f32_t second_longitude = b_lons[i];

        simsimd_f32_t latitude_delta = second_latitude - first_latitude;
        simsimd_f32_t longitude_delta = second_longitude - first_longitude;

        // Haversine formula: a = sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2)
        simsimd_f32_t sin_latitude_delta_half = simsimd_f32_sin(latitude_delta * 0.5f);
        simsimd_f32_t sin_longitude_delta_half = simsimd_f32_sin(longitude_delta * 0.5f);
        simsimd_f32_t cos_first_latitude = simsimd_f32_cos(first_latitude);
        simsimd_f32_t cos_second_latitude = simsimd_f32_cos(second_latitude);

        simsimd_f32_t haversine_term = sin_latitude_delta_half * sin_latitude_delta_half +
                                       cos_first_latitude * cos_second_latitude * sin_longitude_delta_half *
                                           sin_longitude_delta_half;

        // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
        simsimd_f32_t sqrt_haversine = SIMSIMD_F32_SQRT(haversine_term);
        simsimd_f32_t sqrt_complement = SIMSIMD_F32_SQRT(1.0f - haversine_term);
        simsimd_f32_t central_angle = 2.0f * simsimd_f32_atan2(sqrt_haversine, sqrt_complement);

        results[i] = earth_radius * central_angle;
    }
}

SIMSIMD_PUBLIC void simsimd_vincenty_f64_serial(              //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results) {

    simsimd_f64_t const equatorial_radius = SIMSIMD_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
    simsimd_f64_t const polar_radius = SIMSIMD_EARTH_ELLIPSOID_POLAR_RADIUS;
    simsimd_f64_t const flattening = 1.0 / SIMSIMD_EARTH_ELLIPSOID_INVERSE_FLATTENING;

    for (simsimd_size_t i = 0; i != n; ++i) {
        simsimd_f64_t first_latitude = a_lats[i];
        simsimd_f64_t second_latitude = b_lats[i];
        simsimd_f64_t longitude_difference = b_lons[i] - a_lons[i];

        // Reduced latitudes on the auxiliary sphere
        simsimd_f64_t tan_reduced_first = (1.0 - flattening) * SIMSIMD_F64_TAN(first_latitude);
        simsimd_f64_t tan_reduced_second = (1.0 - flattening) * SIMSIMD_F64_TAN(second_latitude);
        simsimd_f64_t cos_reduced_first = 1.0 / SIMSIMD_F64_SQRT(1.0 + tan_reduced_first * tan_reduced_first);
        simsimd_f64_t sin_reduced_first = tan_reduced_first * cos_reduced_first;
        simsimd_f64_t cos_reduced_second = 1.0 / SIMSIMD_F64_SQRT(1.0 + tan_reduced_second * tan_reduced_second);
        simsimd_f64_t sin_reduced_second = tan_reduced_second * cos_reduced_second;

        // Iterative convergence of lambda (difference in longitude on auxiliary sphere)
        simsimd_f64_t lambda = longitude_difference;
        simsimd_f64_t lambda_previous;
        simsimd_f64_t sin_angular_distance, cos_angular_distance, angular_distance;
        simsimd_f64_t sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;
        simsimd_u32_t iteration = 0;

        // Check for coincident points early
        simsimd_u32_t coincident = 0;
        do {
            simsimd_f64_t sin_lambda = simsimd_f64_sin(lambda);
            simsimd_f64_t cos_lambda = simsimd_f64_cos(lambda);

            simsimd_f64_t cross_term = cos_reduced_second * sin_lambda;
            simsimd_f64_t mixed_term = cos_reduced_first * sin_reduced_second -
                                       sin_reduced_first * cos_reduced_second * cos_lambda;
            sin_angular_distance = SIMSIMD_F64_SQRT(cross_term * cross_term + mixed_term * mixed_term);

            if (sin_angular_distance == 0.0) {
                coincident = 1;
                break;
            }

            cos_angular_distance = sin_reduced_first * sin_reduced_second +
                                   cos_reduced_first * cos_reduced_second * cos_lambda;
            angular_distance = simsimd_f64_atan2(sin_angular_distance, cos_angular_distance);

            sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda / sin_angular_distance;
            cos_squared_azimuth = 1.0 - sin_azimuth * sin_azimuth;

            // Handle equatorial geodesic case
            cos_double_angular_midpoint = (cos_squared_azimuth != 0.0)
                                              ? cos_angular_distance -
                                                    2.0 * sin_reduced_first * sin_reduced_second / cos_squared_azimuth
                                              : 0.0;

            simsimd_f64_t correction_factor = flattening / 16.0 * cos_squared_azimuth *
                                              (4.0 + flattening * (4.0 - 3.0 * cos_squared_azimuth));

            lambda_previous = lambda;
            lambda = longitude_difference + (1.0 - correction_factor) * flattening * sin_azimuth *
                                                (angular_distance + correction_factor * sin_angular_distance *
                                                                        (cos_double_angular_midpoint +
                                                                         correction_factor * cos_angular_distance *
                                                                             (-1.0 + 2.0 * cos_double_angular_midpoint *
                                                                                         cos_double_angular_midpoint)));

            iteration++;
        } while (SIMSIMD_F64_ABS(lambda - lambda_previous) > SIMSIMD_VINCENTY_CONVERGENCE_THRESHOLD &&
                 iteration < SIMSIMD_VINCENTY_MAX_ITERATIONS);

        if (coincident) {
            results[i] = 0.0;
            continue;
        }

        // Final distance calculation
        simsimd_f64_t u_squared = cos_squared_azimuth *
                                  (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                                  (polar_radius * polar_radius);
        simsimd_f64_t series_a = 1.0 + u_squared / 16384.0 *
                                           (4096.0 + u_squared * (-768.0 + u_squared * (320.0 - 175.0 * u_squared)));
        simsimd_f64_t series_b = u_squared / 1024.0 *
                                 (256.0 + u_squared * (-128.0 + u_squared * (74.0 - 47.0 * u_squared)));

        simsimd_f64_t angular_correction =
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

SIMSIMD_PUBLIC void simsimd_vincenty_f32_serial(              //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results) {

    simsimd_f32_t const equatorial_radius = (simsimd_f32_t)SIMSIMD_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
    simsimd_f32_t const polar_radius = (simsimd_f32_t)SIMSIMD_EARTH_ELLIPSOID_POLAR_RADIUS;
    simsimd_f32_t const flattening = 1.0f / (simsimd_f32_t)SIMSIMD_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    simsimd_f32_t const convergence_threshold = (simsimd_f32_t)SIMSIMD_VINCENTY_CONVERGENCE_THRESHOLD;

    for (simsimd_size_t i = 0; i != n; ++i) {
        simsimd_f32_t first_latitude = a_lats[i];
        simsimd_f32_t second_latitude = b_lats[i];
        simsimd_f32_t longitude_difference = b_lons[i] - a_lons[i];

        // Reduced latitudes on the auxiliary sphere
        simsimd_f32_t tan_reduced_first = (1.0f - flattening) * SIMSIMD_F32_TAN(first_latitude);
        simsimd_f32_t tan_reduced_second = (1.0f - flattening) * SIMSIMD_F32_TAN(second_latitude);
        simsimd_f32_t cos_reduced_first = 1.0f / SIMSIMD_F32_SQRT(1.0f + tan_reduced_first * tan_reduced_first);
        simsimd_f32_t sin_reduced_first = tan_reduced_first * cos_reduced_first;
        simsimd_f32_t cos_reduced_second = 1.0f / SIMSIMD_F32_SQRT(1.0f + tan_reduced_second * tan_reduced_second);
        simsimd_f32_t sin_reduced_second = tan_reduced_second * cos_reduced_second;

        // Iterative convergence of lambda (difference in longitude on auxiliary sphere)
        simsimd_f32_t lambda = longitude_difference;
        simsimd_f32_t lambda_previous;
        simsimd_f32_t sin_angular_distance, cos_angular_distance, angular_distance;
        simsimd_f32_t sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;
        simsimd_u32_t iteration = 0;

        // Check for coincident points early
        simsimd_u32_t coincident = 0;
        do {
            simsimd_f32_t sin_lambda = simsimd_f32_sin(lambda);
            simsimd_f32_t cos_lambda = simsimd_f32_cos(lambda);

            simsimd_f32_t cross_term = cos_reduced_second * sin_lambda;
            simsimd_f32_t mixed_term = cos_reduced_first * sin_reduced_second -
                                       sin_reduced_first * cos_reduced_second * cos_lambda;
            sin_angular_distance = SIMSIMD_F32_SQRT(cross_term * cross_term + mixed_term * mixed_term);

            if (sin_angular_distance == 0.0f) {
                coincident = 1;
                break;
            }

            cos_angular_distance = sin_reduced_first * sin_reduced_second +
                                   cos_reduced_first * cos_reduced_second * cos_lambda;
            angular_distance = simsimd_f32_atan2(sin_angular_distance, cos_angular_distance);

            sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda / sin_angular_distance;
            cos_squared_azimuth = 1.0f - sin_azimuth * sin_azimuth;

            // Handle equatorial geodesic case
            cos_double_angular_midpoint = (cos_squared_azimuth != 0.0f)
                                              ? cos_angular_distance -
                                                    2.0f * sin_reduced_first * sin_reduced_second / cos_squared_azimuth
                                              : 0.0f;

            simsimd_f32_t correction_factor = flattening / 16.0f * cos_squared_azimuth *
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
        } while (SIMSIMD_F32_ABS(lambda - lambda_previous) > convergence_threshold &&
                 iteration < SIMSIMD_VINCENTY_MAX_ITERATIONS);

        if (coincident) {
            results[i] = 0.0;
            continue;
        }

        // Final distance calculation
        simsimd_f32_t u_squared = cos_squared_azimuth *
                                  (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                                  (polar_radius * polar_radius);
        simsimd_f32_t series_a =
            1.0f + u_squared / 16384.0f * (4096.0f + u_squared * (-768.0f + u_squared * (320.0f - 175.0f * u_squared)));
        simsimd_f32_t series_b = u_squared / 1024.0f *
                                 (256.0f + u_squared * (-128.0f + u_squared * (74.0f - 47.0f * u_squared)));

        simsimd_f32_t angular_correction =
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

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

/*  Haswell AVX2 implementations using 4-wide f64 and 8-wide f32 SIMD.
 *  These require AVX2 trigonometric kernels from trigonometry.h.
 */

SIMSIMD_INTERNAL __m256d _simsimd_haversine_f64x4_haswell( //
    __m256d first_latitudes, __m256d first_longitudes,     //
    __m256d second_latitudes, __m256d second_longitudes) {

    __m256d const earth_radius = _mm256_set1_pd(SIMSIMD_EARTH_MEDIATORIAL_RADIUS);
    __m256d const half = _mm256_set1_pd(0.5);
    __m256d const one = _mm256_set1_pd(1.0);
    __m256d const two = _mm256_set1_pd(2.0);

    __m256d latitude_delta = _mm256_sub_pd(second_latitudes, first_latitudes);
    __m256d longitude_delta = _mm256_sub_pd(second_longitudes, first_longitudes);

    // Haversine terms: sin^2(delta/2)
    __m256d latitude_delta_half = _mm256_mul_pd(latitude_delta, half);
    __m256d longitude_delta_half = _mm256_mul_pd(longitude_delta, half);
    __m256d sin_latitude_delta_half = _simsimd_f64x4_sin_haswell(latitude_delta_half);
    __m256d sin_longitude_delta_half = _simsimd_f64x4_sin_haswell(longitude_delta_half);
    __m256d sin_squared_latitude_delta_half = _mm256_mul_pd(sin_latitude_delta_half, sin_latitude_delta_half);
    __m256d sin_squared_longitude_delta_half = _mm256_mul_pd(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    __m256d cos_first_latitude = _simsimd_f64x4_cos_haswell(first_latitudes);
    __m256d cos_second_latitude = _simsimd_f64x4_cos_haswell(second_latitudes);
    __m256d cos_latitude_product = _mm256_mul_pd(cos_first_latitude, cos_second_latitude);

    // a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
    __m256d haversine_term = _mm256_add_pd(sin_squared_latitude_delta_half,
                                           _mm256_mul_pd(cos_latitude_product, sin_squared_longitude_delta_half));

    // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a)) = 2 * atan(sqrt(a/(1-a)))
    __m256d sqrt_haversine = _mm256_sqrt_pd(haversine_term);
    __m256d sqrt_complement = _mm256_sqrt_pd(_mm256_sub_pd(one, haversine_term));
    __m256d ratio = _mm256_div_pd(sqrt_haversine, sqrt_complement);
    __m256d central_angle = _mm256_mul_pd(two, _simsimd_f64x4_atan_haswell(ratio));

    return _mm256_mul_pd(earth_radius, central_angle);
}

SIMSIMD_PUBLIC void simsimd_haversine_f64_haswell(            //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results) {

    while (n >= 4) {
        __m256d first_latitudes = _mm256_loadu_pd(a_lats);
        __m256d first_longitudes = _mm256_loadu_pd(a_lons);
        __m256d second_latitudes = _mm256_loadu_pd(b_lats);
        __m256d second_longitudes = _mm256_loadu_pd(b_lons);

        __m256d distances = _simsimd_haversine_f64x4_haswell(first_latitudes, first_longitudes, second_latitudes,
                                                             second_longitudes);
        _mm256_storeu_pd(results, distances);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle remaining elements with serial code
    if (n > 0) { simsimd_haversine_f64_serial(a_lats, a_lons, b_lats, b_lons, n, results); }
}

SIMSIMD_INTERNAL __m256 _simsimd_haversine_f32x8_haswell( //
    __m256 first_latitudes, __m256 first_longitudes,      //
    __m256 second_latitudes, __m256 second_longitudes) {

    __m256 const earth_radius = _mm256_set1_ps((float)SIMSIMD_EARTH_MEDIATORIAL_RADIUS);
    __m256 const half = _mm256_set1_ps(0.5f);
    __m256 const one = _mm256_set1_ps(1.0f);
    __m256 const two = _mm256_set1_ps(2.0f);

    __m256 latitude_delta = _mm256_sub_ps(second_latitudes, first_latitudes);
    __m256 longitude_delta = _mm256_sub_ps(second_longitudes, first_longitudes);

    // Haversine terms: sin^2(delta/2)
    __m256 latitude_delta_half = _mm256_mul_ps(latitude_delta, half);
    __m256 longitude_delta_half = _mm256_mul_ps(longitude_delta, half);
    __m256 sin_latitude_delta_half = _simsimd_f32x8_sin_haswell(latitude_delta_half);
    __m256 sin_longitude_delta_half = _simsimd_f32x8_sin_haswell(longitude_delta_half);
    __m256 sin_squared_latitude_delta_half = _mm256_mul_ps(sin_latitude_delta_half, sin_latitude_delta_half);
    __m256 sin_squared_longitude_delta_half = _mm256_mul_ps(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    __m256 cos_first_latitude = _simsimd_f32x8_cos_haswell(first_latitudes);
    __m256 cos_second_latitude = _simsimd_f32x8_cos_haswell(second_latitudes);
    __m256 cos_latitude_product = _mm256_mul_ps(cos_first_latitude, cos_second_latitude);

    // a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
    __m256 haversine_term = _mm256_add_ps(sin_squared_latitude_delta_half,
                                          _mm256_mul_ps(cos_latitude_product, sin_squared_longitude_delta_half));

    // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
    __m256 sqrt_haversine = _mm256_sqrt_ps(haversine_term);
    __m256 sqrt_complement = _mm256_sqrt_ps(_mm256_sub_ps(one, haversine_term));
    __m256 central_angle = _mm256_mul_ps(two, _simsimd_f32x8_atan2_haswell(sqrt_haversine, sqrt_complement));

    return _mm256_mul_ps(earth_radius, central_angle);
}

SIMSIMD_PUBLIC void simsimd_haversine_f32_haswell(            //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results) {

    while (n >= 8) {
        __m256 first_latitudes = _mm256_loadu_ps(a_lats);
        __m256 first_longitudes = _mm256_loadu_ps(a_lons);
        __m256 second_latitudes = _mm256_loadu_ps(b_lats);
        __m256 second_longitudes = _mm256_loadu_ps(b_lons);

        __m256 distances = _simsimd_haversine_f32x8_haswell(first_latitudes, first_longitudes, second_latitudes,
                                                            second_longitudes);
        _mm256_storeu_ps(results, distances);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with serial code
    if (n > 0) { simsimd_haversine_f32_serial(a_lats, a_lons, b_lats, b_lons, n, results); }
}

/**
 *  @brief  AVX2 helper for Vincenty's geodesic distance on 4 f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
SIMSIMD_INTERNAL __m256d _simsimd_vincenty_f64x4_haswell( //
    __m256d first_latitudes, __m256d first_longitudes,    //
    __m256d second_latitudes, __m256d second_longitudes) {

    __m256d const equatorial_radius = _mm256_set1_pd(SIMSIMD_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m256d const polar_radius = _mm256_set1_pd(SIMSIMD_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m256d const flattening = _mm256_set1_pd(1.0 / SIMSIMD_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m256d const convergence_threshold = _mm256_set1_pd(SIMSIMD_VINCENTY_CONVERGENCE_THRESHOLD);
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
    __m256d tan_first = _mm256_div_pd(_simsimd_f64x4_sin_haswell(first_latitudes),
                                      _simsimd_f64x4_cos_haswell(first_latitudes));
    __m256d tan_second = _mm256_div_pd(_simsimd_f64x4_sin_haswell(second_latitudes),
                                       _simsimd_f64x4_cos_haswell(second_latitudes));
    __m256d tan_reduced_first = _mm256_mul_pd(one_minus_f, tan_first);
    __m256d tan_reduced_second = _mm256_mul_pd(one_minus_f, tan_second);

    // cos(U) = 1/sqrt(1 + tan²(U)), sin(U) = tan(U) * cos(U)
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

    for (simsimd_u32_t iteration = 0; iteration < SIMSIMD_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        int converged_bits = _mm256_movemask_pd(converged_mask);
        if (converged_bits == 0xF) break;

        __m256d sin_lambda = _simsimd_f64x4_sin_haswell(lambda);
        __m256d cos_lambda = _simsimd_f64x4_cos_haswell(lambda);

        // sin_angular_distance² = (cos_U2 * sin_λ)² + (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_λ)²
        __m256d cross_term = _mm256_mul_pd(cos_reduced_second, sin_lambda);
        __m256d mixed_term = _mm256_sub_pd(
            _mm256_mul_pd(cos_reduced_first, sin_reduced_second),
            _mm256_mul_pd(_mm256_mul_pd(sin_reduced_first, cos_reduced_second), cos_lambda));
        __m256d sin_angular_dist_sq = _mm256_fmadd_pd(cross_term, cross_term, _mm256_mul_pd(mixed_term, mixed_term));
        sin_angular_distance = _mm256_sqrt_pd(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = _mm256_cmp_pd(sin_angular_distance, epsilon, _CMP_LT_OS);

        // cos_angular_distance = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_λ
        cos_angular_distance = _mm256_fmadd_pd(_mm256_mul_pd(cos_reduced_first, cos_reduced_second), cos_lambda,
                                               _mm256_mul_pd(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = _simsimd_f64x4_atan2_haswell(sin_angular_distance, cos_angular_distance);

        // sin_azimuth = cos_U1 * cos_U2 * sin_λ / sin_angular_distance
        // Avoid division by zero by using blending
        __m256d safe_sin_angular = _mm256_blendv_pd(sin_angular_distance, one, coincident_mask);
        sin_azimuth = _mm256_div_pd(_mm256_mul_pd(_mm256_mul_pd(cos_reduced_first, cos_reduced_second), sin_lambda),
                                    safe_sin_angular);
        cos_squared_azimuth = _mm256_sub_pd(one, _mm256_mul_pd(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α ≈ 0
        __m256d equatorial_mask = _mm256_cmp_pd(cos_squared_azimuth, epsilon, _CMP_LT_OS);
        __m256d safe_cos_sq_azimuth = _mm256_blendv_pd(cos_squared_azimuth, one, equatorial_mask);

        // cos_2σm = cos_σ - 2 * sin_U1 * sin_U2 / cos²α
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

        // λ' = L + (1-C) * f * sin_α * (σ + C * sin_σ * (cos_2σm + C * cos_σ * (-1 + 2*cos²_2σm)))
        __m256d cos_2sm_sq = _mm256_mul_pd(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2*cos²_2σm
        __m256d innermost = _mm256_fmadd_pd(two, cos_2sm_sq, _mm256_set1_pd(-1.0));
        // middle = cos_2σm + C * cos_σ * innermost
        __m256d middle = _mm256_fmadd_pd(_mm256_mul_pd(correction_factor, cos_angular_distance), innermost,
                                         cos_double_angular_midpoint);
        // inner = C * sin_σ * middle
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

    // Δσ = B * sin_σ * (cos_2σm + B/4 * (cos_σ*(-1 + 2*cos²_2σm) - B/6*cos_2σm*(-3 + 4*sin²σ)*(-3 + 4*cos²_2σm)))
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

SIMSIMD_PUBLIC void simsimd_vincenty_f64_haswell(             //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results) {

    while (n >= 4) {
        __m256d first_latitudes = _mm256_loadu_pd(a_lats);
        __m256d first_longitudes = _mm256_loadu_pd(a_lons);
        __m256d second_latitudes = _mm256_loadu_pd(b_lats);
        __m256d second_longitudes = _mm256_loadu_pd(b_lons);

        __m256d distances = _simsimd_vincenty_f64x4_haswell(first_latitudes, first_longitudes, second_latitudes,
                                                            second_longitudes);
        _mm256_storeu_pd(results, distances);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle remaining elements with serial code
    if (n > 0) { simsimd_vincenty_f64_serial(a_lats, a_lons, b_lats, b_lons, n, results); }
}

/**
 *  @brief  AVX2 helper for Vincenty's geodesic distance on 8 f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
SIMSIMD_INTERNAL __m256 _simsimd_vincenty_f32x8_haswell( //
    __m256 first_latitudes, __m256 first_longitudes,     //
    __m256 second_latitudes, __m256 second_longitudes) {

    __m256 const equatorial_radius = _mm256_set1_ps((float)SIMSIMD_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m256 const polar_radius = _mm256_set1_ps((float)SIMSIMD_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m256 const flattening = _mm256_set1_ps(1.0f / (float)SIMSIMD_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m256 const convergence_threshold = _mm256_set1_ps((float)SIMSIMD_VINCENTY_CONVERGENCE_THRESHOLD);
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
    __m256 tan_first = _mm256_div_ps(_simsimd_f32x8_sin_haswell(first_latitudes),
                                     _simsimd_f32x8_cos_haswell(first_latitudes));
    __m256 tan_second = _mm256_div_ps(_simsimd_f32x8_sin_haswell(second_latitudes),
                                      _simsimd_f32x8_cos_haswell(second_latitudes));
    __m256 tan_reduced_first = _mm256_mul_ps(one_minus_f, tan_first);
    __m256 tan_reduced_second = _mm256_mul_ps(one_minus_f, tan_second);

    // cos(U) = 1/sqrt(1 + tan²(U)), sin(U) = tan(U) * cos(U)
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

    for (simsimd_u32_t iteration = 0; iteration < SIMSIMD_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        int converged_bits = _mm256_movemask_ps(converged_mask);
        if (converged_bits == 0xFF) break;

        __m256 sin_lambda = _simsimd_f32x8_sin_haswell(lambda);
        __m256 cos_lambda = _simsimd_f32x8_cos_haswell(lambda);

        // sin_angular_distance² = (cos_U2 * sin_λ)² + (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_λ)²
        __m256 cross_term = _mm256_mul_ps(cos_reduced_second, sin_lambda);
        __m256 mixed_term = _mm256_sub_ps(
            _mm256_mul_ps(cos_reduced_first, sin_reduced_second),
            _mm256_mul_ps(_mm256_mul_ps(sin_reduced_first, cos_reduced_second), cos_lambda));
        __m256 sin_angular_dist_sq = _mm256_fmadd_ps(cross_term, cross_term, _mm256_mul_ps(mixed_term, mixed_term));
        sin_angular_distance = _mm256_sqrt_ps(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = _mm256_cmp_ps(sin_angular_distance, epsilon, _CMP_LT_OS);

        // cos_angular_distance = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_λ
        cos_angular_distance = _mm256_fmadd_ps(_mm256_mul_ps(cos_reduced_first, cos_reduced_second), cos_lambda,
                                               _mm256_mul_ps(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = _simsimd_f32x8_atan2_haswell(sin_angular_distance, cos_angular_distance);

        // sin_azimuth = cos_U1 * cos_U2 * sin_λ / sin_angular_distance
        // Avoid division by zero by using blending
        __m256 safe_sin_angular = _mm256_blendv_ps(sin_angular_distance, one, coincident_mask);
        sin_azimuth = _mm256_div_ps(_mm256_mul_ps(_mm256_mul_ps(cos_reduced_first, cos_reduced_second), sin_lambda),
                                    safe_sin_angular);
        cos_squared_azimuth = _mm256_sub_ps(one, _mm256_mul_ps(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α ≈ 0
        __m256 equatorial_mask = _mm256_cmp_ps(cos_squared_azimuth, epsilon, _CMP_LT_OS);
        __m256 safe_cos_sq_azimuth = _mm256_blendv_ps(cos_squared_azimuth, one, equatorial_mask);

        // cos_2σm = cos_σ - 2 * sin_U1 * sin_U2 / cos²α
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

        // λ' = L + (1-C) * f * sin_α * (σ + C * sin_σ * (cos_2σm + C * cos_σ * (-1 + 2*cos²_2σm)))
        __m256 cos_2sm_sq = _mm256_mul_ps(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2*cos²_2σm
        __m256 innermost = _mm256_fmadd_ps(two, cos_2sm_sq, _mm256_set1_ps(-1.0f));
        // middle = cos_2σm + C * cos_σ * innermost
        __m256 middle = _mm256_fmadd_ps(_mm256_mul_ps(correction_factor, cos_angular_distance), innermost,
                                        cos_double_angular_midpoint);
        // inner = C * sin_σ * middle
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

    // Δσ = B * sin_σ * (cos_2σm + B/4 * (cos_σ*(-1 + 2*cos²_2σm) - B/6*cos_2σm*(-3 + 4*sin²σ)*(-3 + 4*cos²_2σm)))
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

SIMSIMD_PUBLIC void simsimd_vincenty_f32_haswell(             //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results) {

    while (n >= 8) {
        __m256 first_latitudes = _mm256_loadu_ps(a_lats);
        __m256 first_longitudes = _mm256_loadu_ps(a_lons);
        __m256 second_latitudes = _mm256_loadu_ps(b_lats);
        __m256 second_longitudes = _mm256_loadu_ps(b_lons);

        __m256 distances = _simsimd_vincenty_f32x8_haswell(first_latitudes, first_longitudes, second_latitudes,
                                                           second_longitudes);
        _mm256_storeu_ps(results, distances);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with serial code
    if (n > 0) { simsimd_vincenty_f32_serial(a_lats, a_lons, b_lats, b_lons, n, results); }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL
#endif // _SIMSIMD_TARGET_X86

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,bmi2"))), \
                             apply_to = function)

SIMSIMD_INTERNAL __m512d _simsimd_haversine_f64x8_skylake( //
    __m512d first_latitudes, __m512d first_longitudes,     //
    __m512d second_latitudes, __m512d second_longitudes) {

    __m512d const earth_radius = _mm512_set1_pd(SIMSIMD_EARTH_MEDIATORIAL_RADIUS);
    __m512d const half = _mm512_set1_pd(0.5);
    __m512d const one = _mm512_set1_pd(1.0);
    __m512d const two = _mm512_set1_pd(2.0);

    __m512d latitude_delta = _mm512_sub_pd(second_latitudes, first_latitudes);
    __m512d longitude_delta = _mm512_sub_pd(second_longitudes, first_longitudes);

    // Haversine terms: sin^2(delta/2)
    __m512d latitude_delta_half = _mm512_mul_pd(latitude_delta, half);
    __m512d longitude_delta_half = _mm512_mul_pd(longitude_delta, half);
    __m512d sin_latitude_delta_half = _simsimd_f64x8_sin_skylake(latitude_delta_half);
    __m512d sin_longitude_delta_half = _simsimd_f64x8_sin_skylake(longitude_delta_half);
    __m512d sin_squared_latitude_delta_half = _mm512_mul_pd(sin_latitude_delta_half, sin_latitude_delta_half);
    __m512d sin_squared_longitude_delta_half = _mm512_mul_pd(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    __m512d cos_first_latitude = _simsimd_f64x8_cos_skylake(first_latitudes);
    __m512d cos_second_latitude = _simsimd_f64x8_cos_skylake(second_latitudes);
    __m512d cos_latitude_product = _mm512_mul_pd(cos_first_latitude, cos_second_latitude);

    // a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
    __m512d haversine_term = _mm512_add_pd(sin_squared_latitude_delta_half,
                                           _mm512_mul_pd(cos_latitude_product, sin_squared_longitude_delta_half));

    // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a)) = 2 * atan(sqrt(a/(1-a)))
    __m512d sqrt_haversine = _mm512_sqrt_pd(haversine_term);
    __m512d sqrt_complement = _mm512_sqrt_pd(_mm512_sub_pd(one, haversine_term));
    __m512d ratio = _mm512_div_pd(sqrt_haversine, sqrt_complement);
    __m512d central_angle = _mm512_mul_pd(two, _simsimd_f64x8_atan_skylake(ratio));

    return _mm512_mul_pd(earth_radius, central_angle);
}

SIMSIMD_PUBLIC void simsimd_haversine_f64_skylake(            //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results) {

    while (n >= 8) {
        __m512d first_latitudes = _mm512_loadu_pd(a_lats);
        __m512d first_longitudes = _mm512_loadu_pd(a_lons);
        __m512d second_latitudes = _mm512_loadu_pd(b_lats);
        __m512d second_longitudes = _mm512_loadu_pd(b_lons);

        __m512d distances = _simsimd_haversine_f64x8_skylake(first_latitudes, first_longitudes, second_latitudes,
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

        __m512d distances = _simsimd_haversine_f64x8_skylake(first_latitudes, first_longitudes, second_latitudes,
                                                             second_longitudes);
        _mm512_mask_storeu_pd(results, mask, distances);
    }
}

/**
 *  @brief  AVX-512 helper for Vincenty's geodesic distance on 8 f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking.
 */
SIMSIMD_INTERNAL __m512d _simsimd_vincenty_f64x8_skylake( //
    __m512d first_latitudes, __m512d first_longitudes,    //
    __m512d second_latitudes, __m512d second_longitudes) {

    __m512d const equatorial_radius = _mm512_set1_pd(SIMSIMD_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m512d const polar_radius = _mm512_set1_pd(SIMSIMD_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m512d const flattening = _mm512_set1_pd(1.0 / SIMSIMD_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m512d const convergence_threshold = _mm512_set1_pd(SIMSIMD_VINCENTY_CONVERGENCE_THRESHOLD);
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
    __m512d tan_first = _mm512_div_pd(_simsimd_f64x8_sin_skylake(first_latitudes),
                                      _simsimd_f64x8_cos_skylake(first_latitudes));
    __m512d tan_second = _mm512_div_pd(_simsimd_f64x8_sin_skylake(second_latitudes),
                                       _simsimd_f64x8_cos_skylake(second_latitudes));
    __m512d tan_reduced_first = _mm512_mul_pd(one_minus_f, tan_first);
    __m512d tan_reduced_second = _mm512_mul_pd(one_minus_f, tan_second);

    // cos(U) = 1/sqrt(1 + tan²(U)), sin(U) = tan(U) * cos(U)
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

    for (simsimd_u32_t iteration = 0; iteration < SIMSIMD_VINCENTY_MAX_ITERATIONS && converged_mask != 0xFF;
         ++iteration) {
        __m512d sin_lambda = _simsimd_f64x8_sin_skylake(lambda);
        __m512d cos_lambda = _simsimd_f64x8_cos_skylake(lambda);

        // sin_angular_distance² = (cos_U2 * sin_λ)² + (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_λ)²
        __m512d cross_term = _mm512_mul_pd(cos_reduced_second, sin_lambda);
        __m512d mixed_term = _mm512_sub_pd(
            _mm512_mul_pd(cos_reduced_first, sin_reduced_second),
            _mm512_mul_pd(_mm512_mul_pd(sin_reduced_first, cos_reduced_second), cos_lambda));
        __m512d sin_angular_dist_sq = _mm512_fmadd_pd(cross_term, cross_term, _mm512_mul_pd(mixed_term, mixed_term));
        sin_angular_distance = _mm512_sqrt_pd(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = _mm512_cmp_pd_mask(sin_angular_distance, _mm512_set1_pd(1e-15), _CMP_LT_OS);

        // cos_angular_distance = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_λ
        cos_angular_distance = _mm512_fmadd_pd(_mm512_mul_pd(cos_reduced_first, cos_reduced_second), cos_lambda,
                                               _mm512_mul_pd(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = _simsimd_f64x8_atan2_skylake(sin_angular_distance, cos_angular_distance);

        // sin_azimuth = cos_U1 * cos_U2 * sin_λ / sin_angular_distance
        sin_azimuth = _mm512_div_pd(_mm512_mul_pd(_mm512_mul_pd(cos_reduced_first, cos_reduced_second), sin_lambda),
                                    sin_angular_distance);
        cos_squared_azimuth = _mm512_sub_pd(one, _mm512_mul_pd(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α = 0
        __mmask8 equatorial_mask = _mm512_cmp_pd_mask(cos_squared_azimuth, _mm512_set1_pd(1e-15), _CMP_LT_OS);

        // cos_2σm = cos_σ - 2 * sin_U1 * sin_U2 / cos²α
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

        // λ' = L + (1-C) * f * sin_α * (σ + C * sin_σ * (cos_2σm + C * cos_σ * (-1 + 2*cos²_2σm)))
        __m512d cos_2sm_sq = _mm512_mul_pd(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2*cos²_2σm
        __m512d innermost = _mm512_fmadd_pd(two, cos_2sm_sq, _mm512_set1_pd(-1.0));
        // middle = cos_2σm + C * cos_σ * innermost
        __m512d middle = _mm512_fmadd_pd(_mm512_mul_pd(correction_factor, cos_angular_distance), innermost,
                                         cos_double_angular_midpoint);
        // inner = C * sin_σ * middle
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

    // Δσ = B * sin_σ * (cos_2σm + B/4 * (cos_σ*(-1 + 2*cos²_2σm) - B/6*cos_2σm*(-3 + 4*sin²σ)*(-3 + 4*cos²_2σm)))
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

SIMSIMD_PUBLIC void simsimd_vincenty_f64_skylake(             //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results) {

    while (n >= 8) {
        __m512d first_latitudes = _mm512_loadu_pd(a_lats);
        __m512d first_longitudes = _mm512_loadu_pd(a_lons);
        __m512d second_latitudes = _mm512_loadu_pd(b_lats);
        __m512d second_longitudes = _mm512_loadu_pd(b_lons);

        __m512d distances = _simsimd_vincenty_f64x8_skylake(first_latitudes, first_longitudes, second_latitudes,
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

        __m512d distances = _simsimd_vincenty_f64x8_skylake(first_latitudes, first_longitudes, second_latitudes,
                                                            second_longitudes);
        _mm512_mask_storeu_pd(results, mask, distances);
    }
}

SIMSIMD_INTERNAL __m512 _simsimd_haversine_f32x16_skylake( //
    __m512 first_latitudes, __m512 first_longitudes,       //
    __m512 second_latitudes, __m512 second_longitudes) {

    __m512 const earth_radius = _mm512_set1_ps((float)SIMSIMD_EARTH_MEDIATORIAL_RADIUS);
    __m512 const half = _mm512_set1_ps(0.5f);
    __m512 const one = _mm512_set1_ps(1.0f);
    __m512 const two = _mm512_set1_ps(2.0f);

    __m512 latitude_delta = _mm512_sub_ps(second_latitudes, first_latitudes);
    __m512 longitude_delta = _mm512_sub_ps(second_longitudes, first_longitudes);

    // Haversine terms: sin^2(delta/2)
    __m512 latitude_delta_half = _mm512_mul_ps(latitude_delta, half);
    __m512 longitude_delta_half = _mm512_mul_ps(longitude_delta, half);
    __m512 sin_latitude_delta_half = _simsimd_f32x16_sin_skylake(latitude_delta_half);
    __m512 sin_longitude_delta_half = _simsimd_f32x16_sin_skylake(longitude_delta_half);
    __m512 sin_squared_latitude_delta_half = _mm512_mul_ps(sin_latitude_delta_half, sin_latitude_delta_half);
    __m512 sin_squared_longitude_delta_half = _mm512_mul_ps(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    __m512 cos_first_latitude = _simsimd_f32x16_cos_skylake(first_latitudes);
    __m512 cos_second_latitude = _simsimd_f32x16_cos_skylake(second_latitudes);
    __m512 cos_latitude_product = _mm512_mul_ps(cos_first_latitude, cos_second_latitude);

    // a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
    __m512 haversine_term = _mm512_add_ps(sin_squared_latitude_delta_half,
                                          _mm512_mul_ps(cos_latitude_product, sin_squared_longitude_delta_half));

    // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
    __m512 sqrt_haversine = _mm512_sqrt_ps(haversine_term);
    __m512 sqrt_complement = _mm512_sqrt_ps(_mm512_sub_ps(one, haversine_term));
    __m512 central_angle = _mm512_mul_ps(two, _simsimd_f32x16_atan2_skylake(sqrt_haversine, sqrt_complement));

    return _mm512_mul_ps(earth_radius, central_angle);
}

SIMSIMD_PUBLIC void simsimd_haversine_f32_skylake(            //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results) {

    while (n >= 16) {
        __m512 first_latitudes = _mm512_loadu_ps(a_lats);
        __m512 first_longitudes = _mm512_loadu_ps(a_lons);
        __m512 second_latitudes = _mm512_loadu_ps(b_lats);
        __m512 second_longitudes = _mm512_loadu_ps(b_lons);

        __m512 distances = _simsimd_haversine_f32x16_skylake(first_latitudes, first_longitudes, second_latitudes,
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

        __m512 distances = _simsimd_haversine_f32x16_skylake(first_latitudes, first_longitudes, second_latitudes,
                                                             second_longitudes);
        _mm512_mask_storeu_ps(results, mask, distances);
    }
}

/**
 *  @brief  AVX-512 helper for Vincenty's geodesic distance on 16 f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking.
 */
SIMSIMD_INTERNAL __m512 _simsimd_vincenty_f32x16_skylake( //
    __m512 first_latitudes, __m512 first_longitudes,      //
    __m512 second_latitudes, __m512 second_longitudes) {

    __m512 const equatorial_radius = _mm512_set1_ps((float)SIMSIMD_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m512 const polar_radius = _mm512_set1_ps((float)SIMSIMD_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m512 const flattening = _mm512_set1_ps(1.0f / (float)SIMSIMD_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m512 const convergence_threshold = _mm512_set1_ps((float)SIMSIMD_VINCENTY_CONVERGENCE_THRESHOLD);
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
    __m512 tan_first = _mm512_div_ps(_simsimd_f32x16_sin_skylake(first_latitudes),
                                     _simsimd_f32x16_cos_skylake(first_latitudes));
    __m512 tan_second = _mm512_div_ps(_simsimd_f32x16_sin_skylake(second_latitudes),
                                      _simsimd_f32x16_cos_skylake(second_latitudes));
    __m512 tan_reduced_first = _mm512_mul_ps(one_minus_f, tan_first);
    __m512 tan_reduced_second = _mm512_mul_ps(one_minus_f, tan_second);

    // cos(U) = 1/sqrt(1 + tan²(U)), sin(U) = tan(U) * cos(U)
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

    for (simsimd_u32_t iteration = 0; iteration < SIMSIMD_VINCENTY_MAX_ITERATIONS && converged_mask != 0xFFFF;
         ++iteration) {
        __m512 sin_lambda = _simsimd_f32x16_sin_skylake(lambda);
        __m512 cos_lambda = _simsimd_f32x16_cos_skylake(lambda);

        // sin_angular_distance² = (cos_U2 * sin_λ)² + (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_λ)²
        __m512 cross_term = _mm512_mul_ps(cos_reduced_second, sin_lambda);
        __m512 mixed_term = _mm512_sub_ps(
            _mm512_mul_ps(cos_reduced_first, sin_reduced_second),
            _mm512_mul_ps(_mm512_mul_ps(sin_reduced_first, cos_reduced_second), cos_lambda));
        __m512 sin_angular_dist_sq = _mm512_fmadd_ps(cross_term, cross_term, _mm512_mul_ps(mixed_term, mixed_term));
        sin_angular_distance = _mm512_sqrt_ps(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = _mm512_cmp_ps_mask(sin_angular_distance, _mm512_set1_ps(1e-7f), _CMP_LT_OS);

        // cos_angular_distance = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_λ
        cos_angular_distance = _mm512_fmadd_ps(_mm512_mul_ps(cos_reduced_first, cos_reduced_second), cos_lambda,
                                               _mm512_mul_ps(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = _simsimd_f32x16_atan2_skylake(sin_angular_distance, cos_angular_distance);

        // sin_azimuth = cos_U1 * cos_U2 * sin_λ / sin_angular_distance
        sin_azimuth = _mm512_div_ps(_mm512_mul_ps(_mm512_mul_ps(cos_reduced_first, cos_reduced_second), sin_lambda),
                                    sin_angular_distance);
        cos_squared_azimuth = _mm512_sub_ps(one, _mm512_mul_ps(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α = 0
        __mmask16 equatorial_mask = _mm512_cmp_ps_mask(cos_squared_azimuth, _mm512_set1_ps(1e-7f), _CMP_LT_OS);

        // cos_2σm = cos_σ - 2 * sin_U1 * sin_U2 / cos²α
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

        // λ' = L + (1-C) * f * sin_α * (σ + C * sin_σ * (cos_2σm + C * cos_σ * (-1 + 2*cos²_2σm)))
        __m512 cos_2sm_sq = _mm512_mul_ps(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2*cos²_2σm
        __m512 innermost = _mm512_fmadd_ps(two, cos_2sm_sq, _mm512_set1_ps(-1.0f));
        // middle = cos_2σm + C * cos_σ * innermost
        __m512 middle = _mm512_fmadd_ps(_mm512_mul_ps(correction_factor, cos_angular_distance), innermost,
                                        cos_double_angular_midpoint);
        // inner = C * sin_σ * middle
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

    // Δσ = B * sin_σ * (cos_2σm + B/4 * (cos_σ*(-1 + 2*cos²_2σm) - B/6*cos_2σm*(-3 + 4*sin²σ)*(-3 + 4*cos²_2σm)))
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

SIMSIMD_PUBLIC void simsimd_vincenty_f32_skylake(             //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results) {

    while (n >= 16) {
        __m512 first_latitudes = _mm512_loadu_ps(a_lats);
        __m512 first_longitudes = _mm512_loadu_ps(a_lons);
        __m512 second_latitudes = _mm512_loadu_ps(b_lats);
        __m512 second_longitudes = _mm512_loadu_ps(b_lons);

        __m512 distances = _simsimd_vincenty_f32x16_skylake(first_latitudes, first_longitudes, second_latitudes,
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

        __m512 distances = _simsimd_vincenty_f32x16_skylake(first_latitudes, first_longitudes, second_latitudes,
                                                            second_longitudes);
        _mm512_mask_storeu_ps(results, mask, distances);
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE

#if !SIMSIMD_DYNAMIC_DISPATCH

SIMSIMD_PUBLIC void simsimd_haversine_f64(                    //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_haversine_f64_skylake(a_lats, a_lons, b_lats, b_lons, n, results);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_haversine_f64_haswell(a_lats, a_lons, b_lats, b_lons, n, results);
#else
    simsimd_haversine_f64_serial(a_lats, a_lons, b_lats, b_lons, n, results);
#endif
}

SIMSIMD_PUBLIC void simsimd_haversine_f32(                    //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_haversine_f32_skylake(a_lats, a_lons, b_lats, b_lons, n, results);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_haversine_f32_haswell(a_lats, a_lons, b_lats, b_lons, n, results);
#else
    simsimd_haversine_f32_serial(a_lats, a_lons, b_lats, b_lons, n, results);
#endif
}

SIMSIMD_PUBLIC void simsimd_vincenty_f64(                     //
    simsimd_f64_t const *a_lats, simsimd_f64_t const *a_lons, //
    simsimd_f64_t const *b_lats, simsimd_f64_t const *b_lons, //
    simsimd_size_t n, simsimd_f64_t *results) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_vincenty_f64_skylake(a_lats, a_lons, b_lats, b_lons, n, results);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_vincenty_f64_haswell(a_lats, a_lons, b_lats, b_lons, n, results);
#else
    simsimd_vincenty_f64_serial(a_lats, a_lons, b_lats, b_lons, n, results);
#endif
}

SIMSIMD_PUBLIC void simsimd_vincenty_f32(                     //
    simsimd_f32_t const *a_lats, simsimd_f32_t const *a_lons, //
    simsimd_f32_t const *b_lats, simsimd_f32_t const *b_lons, //
    simsimd_size_t n, simsimd_f32_t *results) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_vincenty_f32_skylake(a_lats, a_lons, b_lats, b_lons, n, results);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_vincenty_f32_haswell(a_lats, a_lons, b_lats, b_lons, n, results);
#else
    simsimd_vincenty_f32_serial(a_lats, a_lons, b_lats, b_lons, n, results);
#endif
}

#endif // !SIMSIMD_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
