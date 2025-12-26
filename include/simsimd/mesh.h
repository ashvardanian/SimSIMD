/**
 *  @brief SIMD-accelerated similarity measures for meshes and rigid 3D bodies.
 *  @file include/simsimd/mesh.h
 *  @author Ash Vardanian
 *  @date June 19, 2024
 *
 *  Contains:
 *
 *  - Root Mean Square Deviation (RMSD) for rigid body superposition
 *  - Kabsch algorithm for optimal rigid body alignment (rotation only)
 *  - Umeyama algorithm for similarity transform (rotation + uniform scaling)
 *
 *  For datatypes:
 *
 *  - 64-bit IEEE-754 floating point
 *  - 32-bit IEEE-754 floating point
 *  - 16-bit IEEE-754 floating point
 *  - 16-bit brain-floating point
 *
 *  For hardware architectures:
 *
 *  - x86 (AVX2, AVX512)
 *  - Arm (NEON, SVE)
 *
 *  @section applications Applications
 *
 *  These routines are the core of point-cloud alignment pipelines:
 *
 *  - Structural biology: protein backbone or ligand alignment (RMSD, Kabsch)
 *  - Computer graphics: mesh registration and deformation transfer
 *  - Robotics/SLAM: point-cloud registration and tracking
 *
 *  @section transformation_convention Transformation Convention
 *
 *  All functions compute a transformation that aligns the FIRST point cloud (a) to the SECOND (b).
 *  The transformation to apply is:
 *
 *      a'_i = scale * R * (a_i - a_centroid) + b_centroid
 *
 *  Where:
 *
 *  - R is a 3x3 rotation matrix (row-major, 9 values)
 *  - scale is a uniform scaling factor (1.0 for RMSD and Kabsch)
 *  - a_centroid, b_centroid are the centroids of the respective point clouds
 *
 *  @section algorithm_overview Algorithm Overview
 *
 *  - RMSD: Simple root mean square deviation without alignment. R = identity, scale = 1.0
 *  - Kabsch: Finds optimal rotation R minimizing ||R*(a - a_centroid) - (b - b_centroid)||. scale = 1.0
 *  - Umeyama: Finds optimal rotation R and scale c minimizing ||c*R*(a - a_centroid) - (b - b_centroid)||
 *
 *  Kabsch and Umeyama compute a 3x3 cross-covariance matrix H = sum (a_i - a_c)(b_i - b_c)^T
 *  and recover R from the SVD of H. Umeyama additionally estimates a uniform scale from the
 *  singular values and the variance of the centered source points.
 *
 *  The 3x3 SVD implementation is based on the McAdams et al. paper:
 *  "Computing the Singular Value Decomposition of 3x3 matrices with minimal branching
 *  and elementary floating point operations", University of Wisconsin - Madison TR1690, 2011.
 *
 *  @section numerical_notes Numerical Notes
 *
 *  - Accurate variants accumulate in double precision before producing single-precision outputs.
 *  - Reflections are handled by flipping the last singular vector when det(R) < 0.
 *  - For very small point sets, the loops are scalar-heavy and dominate over SIMD setup costs.
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  The SIMD kernels are dominated by FMA, permutes, and gathers:
 *
 *      Intrinsic                     Instruction        Notes
 *      _mm256_fmadd_ps/pd            VFMADD*            FMA on FP ports (Haswell/Skylake: ports 0/1)
 *      _mm256_i32gather_ps           VGATHERDPS         High-latency; memory-bound
 *      _mm512_permutex2var_ps/pd     VPERMT2*           Shuffle-heavy; can bottleneck on shuffle ports
 *      _mm512_reduce_add_ps/pd       (sequence)         Implemented via shuffles + adds
 *
 *  Gather-heavy tails are intentionally isolated to keep the steady-state loop on contiguous loads.
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef SIMSIMD_MESH_H
#define SIMSIMD_MESH_H

#include "types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief  Mesh superposition function signature (RMSD, Kabsch, Umeyama).
 *
 *  All mesh functions share this unified signature. The transformation aligns a to b:
 *      a'_i = scale * R * (a_i - a_centroid) + b_centroid
 *
 *  @param[in]  a           First point cloud (source), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in]  b           Second point cloud (target), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in]  n           Number of 3D points in each cloud.
 *  @param[out] a_centroid  Centroid of first cloud (3 values). Can be NULL.
 *  @param[out] b_centroid  Centroid of second cloud (3 values). Can be NULL.
 *  @param[out] rotation    Rotation matrix R, row-major 3x3 (9 values). Can be NULL.
 *                          - RMSD: identity matrix
 *                          - Kabsch/Umeyama: optimal rotation aligning a to b
 *  @param[out] scale       Uniform scale factor. Can be NULL.
 *                          - RMSD/Kabsch: 1.0
 *                          - Umeyama: optimal scale factor
 *  @param[out] result      RMSD after applying the transformation.
 */
SIMSIMD_DYNAMIC void simsimd_rmsd_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                      simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid, simsimd_f64_t *rotation,
                                      simsimd_distance_t *scale, simsimd_distance_t *result);
/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_DYNAMIC void simsimd_rmsd_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                      simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid, simsimd_f32_t *rotation,
                                      simsimd_distance_t *scale, simsimd_distance_t *result);
/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_DYNAMIC void simsimd_rmsd_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                      simsimd_f16_t *a_centroid, simsimd_f16_t *b_centroid, simsimd_f32_t *rotation,
                                      simsimd_distance_t *scale, simsimd_distance_t *result);
/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_DYNAMIC void simsimd_rmsd_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                       simsimd_bf16_t *a_centroid, simsimd_bf16_t *b_centroid, simsimd_f32_t *rotation,
                                       simsimd_distance_t *scale, simsimd_distance_t *result);

/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_DYNAMIC void simsimd_kabsch_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                        simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid, simsimd_f64_t *rotation,
                                        simsimd_distance_t *scale, simsimd_distance_t *result);
/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_DYNAMIC void simsimd_kabsch_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                        simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid, simsimd_f32_t *rotation,
                                        simsimd_distance_t *scale, simsimd_distance_t *result);
/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_DYNAMIC void simsimd_kabsch_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                        simsimd_f16_t *a_centroid, simsimd_f16_t *b_centroid, simsimd_f32_t *rotation,
                                        simsimd_distance_t *scale, simsimd_distance_t *result);
/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_DYNAMIC void simsimd_kabsch_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                         simsimd_bf16_t *a_centroid, simsimd_bf16_t *b_centroid,
                                         simsimd_f32_t *rotation, simsimd_distance_t *scale,
                                         simsimd_distance_t *result);

/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_DYNAMIC void simsimd_umeyama_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                         simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid, simsimd_f64_t *rotation,
                                         simsimd_distance_t *scale, simsimd_distance_t *result);
/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_DYNAMIC void simsimd_umeyama_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                         simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid, simsimd_f32_t *rotation,
                                         simsimd_distance_t *scale, simsimd_distance_t *result);
/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_DYNAMIC void simsimd_umeyama_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                         simsimd_f16_t *a_centroid, simsimd_f16_t *b_centroid, simsimd_f32_t *rotation,
                                         simsimd_distance_t *scale, simsimd_distance_t *result);
/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_DYNAMIC void simsimd_umeyama_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                          simsimd_bf16_t *a_centroid, simsimd_bf16_t *b_centroid,
                                          simsimd_f32_t *rotation, simsimd_distance_t *scale,
                                          simsimd_distance_t *result);

// clang-format off

/*  Serial backends for all numeric types.
 *  By default they use 32-bit arithmetic, unless the arguments themselves contain 64-bit floats.
 *  For double-precision computation check out the "*_accurate" variants of those "*_serial" functions.
 */
/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_PUBLIC void simsimd_rmsd_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_f64_t* a_centroid, simsimd_f64_t* b_centroid, simsimd_f64_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_kabsch_f64 */
SIMSIMD_PUBLIC void simsimd_kabsch_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_f64_t* a_centroid, simsimd_f64_t* b_centroid, simsimd_f64_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_umeyama_f64 */
SIMSIMD_PUBLIC void simsimd_umeyama_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_f64_t* a_centroid, simsimd_f64_t* b_centroid, simsimd_f64_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);

/** @copydoc simsimd_rmsd_f32 */
SIMSIMD_PUBLIC void simsimd_rmsd_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_kabsch_f32 */
SIMSIMD_PUBLIC void simsimd_kabsch_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_umeyama_f32 */
SIMSIMD_PUBLIC void simsimd_umeyama_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);

/** @copydoc simsimd_rmsd_f16 */
SIMSIMD_PUBLIC void simsimd_rmsd_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_f16_t* a_centroid, simsimd_f16_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_kabsch_f16 */
SIMSIMD_PUBLIC void simsimd_kabsch_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_f16_t* a_centroid, simsimd_f16_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_umeyama_f16 */
SIMSIMD_PUBLIC void simsimd_umeyama_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_f16_t* a_centroid, simsimd_f16_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);

/** @copydoc simsimd_rmsd_bf16 */
SIMSIMD_PUBLIC void simsimd_rmsd_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_bf16_t* a_centroid, simsimd_bf16_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_kabsch_bf16 */
SIMSIMD_PUBLIC void simsimd_kabsch_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_bf16_t* a_centroid, simsimd_bf16_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_umeyama_bf16 */
SIMSIMD_PUBLIC void simsimd_umeyama_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_bf16_t* a_centroid, simsimd_bf16_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);

/*  Double-precision serial backends for all numeric types.
 *  For single-precision computation check out the "*_serial" counterparts of those "*_accurate" functions.
 */
/** @copydoc simsimd_rmsd_f32 */
SIMSIMD_PUBLIC void simsimd_rmsd_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_kabsch_f32 */
SIMSIMD_PUBLIC void simsimd_kabsch_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_umeyama_f32 */
SIMSIMD_PUBLIC void simsimd_umeyama_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);

/** @copydoc simsimd_rmsd_f16 */
SIMSIMD_PUBLIC void simsimd_rmsd_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_f16_t* a_centroid, simsimd_f16_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_kabsch_f16 */
SIMSIMD_PUBLIC void simsimd_kabsch_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_f16_t* a_centroid, simsimd_f16_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_umeyama_f16 */
SIMSIMD_PUBLIC void simsimd_umeyama_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_f16_t* a_centroid, simsimd_f16_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);

/** @copydoc simsimd_rmsd_bf16 */
SIMSIMD_PUBLIC void simsimd_rmsd_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_bf16_t* a_centroid, simsimd_bf16_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_kabsch_bf16 */
SIMSIMD_PUBLIC void simsimd_kabsch_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_bf16_t* a_centroid, simsimd_bf16_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_umeyama_bf16 */
SIMSIMD_PUBLIC void simsimd_umeyama_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_bf16_t* a_centroid, simsimd_bf16_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);

/*  SIMD-powered backends for AVX512 CPUs of Skylake generation and newer.
 */
#if SIMSIMD_TARGET_SKYLAKE
/** @copydoc simsimd_rmsd_f32 */
SIMSIMD_PUBLIC void simsimd_rmsd_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_kabsch_f32 */
SIMSIMD_PUBLIC void simsimd_kabsch_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_umeyama_f32 */
SIMSIMD_PUBLIC void simsimd_umeyama_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);

/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_PUBLIC void simsimd_rmsd_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_f64_t* a_centroid, simsimd_f64_t* b_centroid, simsimd_f64_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_kabsch_f64 */
SIMSIMD_PUBLIC void simsimd_kabsch_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_f64_t* a_centroid, simsimd_f64_t* b_centroid, simsimd_f64_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_umeyama_f64 */
SIMSIMD_PUBLIC void simsimd_umeyama_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_f64_t* a_centroid, simsimd_f64_t* b_centroid, simsimd_f64_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
#endif // SIMSIMD_TARGET_SKYLAKE

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer.
 */
#if SIMSIMD_TARGET_HASWELL
/** @copydoc simsimd_rmsd_f32 */
SIMSIMD_PUBLIC void simsimd_rmsd_f32_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_kabsch_f32 */
SIMSIMD_PUBLIC void simsimd_kabsch_f32_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_umeyama_f32 */
SIMSIMD_PUBLIC void simsimd_umeyama_f32_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_f32_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);

/** @copydoc simsimd_rmsd_f64 */
SIMSIMD_PUBLIC void simsimd_rmsd_f64_haswell(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_f64_t* a_centroid, simsimd_f64_t* b_centroid, simsimd_f64_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_kabsch_f64 */
SIMSIMD_PUBLIC void simsimd_kabsch_f64_haswell(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_f64_t* a_centroid, simsimd_f64_t* b_centroid, simsimd_f64_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
/** @copydoc simsimd_umeyama_f64 */
SIMSIMD_PUBLIC void simsimd_umeyama_f64_haswell(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_f64_t* a_centroid, simsimd_f64_t* b_centroid, simsimd_f64_t* rotation, simsimd_distance_t* scale, simsimd_distance_t* result);
#endif // SIMSIMD_TARGET_HASWELL

// clang-format on

/*  Constants for the McAdams 3x3 SVD algorithm.
 *  gamma = (sqrt(8) + 3)^2 / 4 = 5.828427124
 *  cstar = cos(pi/8), sstar = sin(pi/8)
 */
#define SIMSIMD_SVD_GAMMA_F32   5.828427124f
#define SIMSIMD_SVD_CSTAR_F32   0.923879532f
#define SIMSIMD_SVD_SSTAR_F32   0.3826834323f
#define SIMSIMD_SVD_EPSILON_F32 1e-6f

#define SIMSIMD_SVD_GAMMA_F64   5.828427124
#define SIMSIMD_SVD_CSTAR_F64   0.923879532
#define SIMSIMD_SVD_SSTAR_F64   0.3826834323
#define SIMSIMD_SVD_EPSILON_F64 1e-12

/*  Internal helper: Fast reciprocal square root using the "magic number" method.
 *  Two Newton-Raphson iterations for high accuracy (~1e-7 relative error).
 */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_rsqrt_f32(simsimd_f32_t x) {
    simsimd_f32_t xhalf = 0.5f * x;
    union {
        simsimd_f32_t f;
        simsimd_i32_t i;
    } u;
    u.f = x;
    u.i = 0x5f375a82 - (u.i >> 1);
    u.f = u.f * (1.5f - xhalf * u.f * u.f); // First iteration: ~1e-3 relative error
    u.f = u.f * (1.5f - xhalf * u.f * u.f); // Second iteration: ~1e-7 relative error
    return u.f;
}

/*  Internal helper: Conditional swap without branching.
 */
SIMSIMD_INTERNAL void _simsimd_cond_swap_f32(int c, simsimd_f32_t *x, simsimd_f32_t *y) {
    simsimd_f32_t z = *x;
    *x = c ? *y : *x;
    *y = c ? z : *y;
}

/*  Internal helper: Conditional negating swap without branching.
 */
SIMSIMD_INTERNAL void _simsimd_cond_neg_swap_f32(int c, simsimd_f32_t *x, simsimd_f32_t *y) {
    simsimd_f32_t z = -*x;
    *x = c ? *y : *x;
    *y = c ? z : *y;
}

/*  Internal helper: Approximate Givens quaternion for Jacobi rotation.
 */
SIMSIMD_INTERNAL void _simsimd_approx_givens_quat_f32(       //
    simsimd_f32_t a11, simsimd_f32_t a12, simsimd_f32_t a22, //
    simsimd_f32_t *cos_half, simsimd_f32_t *sin_half) {      //
    *cos_half = 2 * (a11 - a22), *sin_half = a12;
    int use_givens = SIMSIMD_SVD_GAMMA_F32 * (*sin_half) * (*sin_half) < (*cos_half) * (*cos_half);
    simsimd_f32_t w = _simsimd_rsqrt_f32((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half));
    *cos_half = use_givens ? w * (*cos_half) : SIMSIMD_SVD_CSTAR_F32;
    *sin_half = use_givens ? w * (*sin_half) : SIMSIMD_SVD_SSTAR_F32;
}

/*  Internal helper: Jacobi conjugation step for symmetric eigenvalue problem.
 */
SIMSIMD_INTERNAL void _simsimd_jacobi_conjugation_f32(          //
    int x, int y, int z,                                        //
    simsimd_f32_t *s11, simsimd_f32_t *s21, simsimd_f32_t *s22, //
    simsimd_f32_t *s31, simsimd_f32_t *s32, simsimd_f32_t *s33, //
    simsimd_f32_t *q_v) {                                       //
    simsimd_f32_t cos_half, sin_half;
    _simsimd_approx_givens_quat_f32(*s11, *s21, *s22, &cos_half, &sin_half);

    simsimd_f32_t scale = cos_half * cos_half + sin_half * sin_half;
    simsimd_f32_t cos_theta = (cos_half * cos_half - sin_half * sin_half) / scale;
    simsimd_f32_t sin_theta = (2 * sin_half * cos_half) / scale;

    simsimd_f32_t s11_tmp = *s11, s21_tmp = *s21, s22_tmp = *s22;
    simsimd_f32_t s31_tmp = *s31, s32_tmp = *s32, s33_tmp = *s33;

    *s11 = cos_theta * (cos_theta * s11_tmp + sin_theta * s21_tmp) +
           sin_theta * (cos_theta * s21_tmp + sin_theta * s22_tmp);
    *s21 = cos_theta * (-sin_theta * s11_tmp + cos_theta * s21_tmp) +
           sin_theta * (-sin_theta * s21_tmp + cos_theta * s22_tmp);
    *s22 = -sin_theta * (-sin_theta * s11_tmp + cos_theta * s21_tmp) +
           cos_theta * (-sin_theta * s21_tmp + cos_theta * s22_tmp);
    *s31 = cos_theta * s31_tmp + sin_theta * s32_tmp;
    *s32 = -sin_theta * s31_tmp + cos_theta * s32_tmp;
    *s33 = s33_tmp;

    // Update quaternion accumulator
    simsimd_f32_t quat_tmp[3];
    quat_tmp[0] = q_v[0] * sin_half, quat_tmp[1] = q_v[1] * sin_half, quat_tmp[2] = q_v[2] * sin_half;
    sin_half *= q_v[3];
    q_v[0] *= cos_half, q_v[1] *= cos_half, q_v[2] *= cos_half, q_v[3] *= cos_half;
    q_v[z] += sin_half, q_v[3] -= quat_tmp[z], q_v[x] += quat_tmp[y], q_v[y] -= quat_tmp[x];

    // Cyclic permutation of matrix elements
    s11_tmp = *s22, s21_tmp = *s32, s22_tmp = *s33, s31_tmp = *s21, s32_tmp = *s31, s33_tmp = *s11;
    *s11 = s11_tmp, *s21 = s21_tmp, *s22 = s22_tmp, *s31 = s31_tmp, *s32 = s32_tmp, *s33 = s33_tmp;
}

/*  Internal helper: Convert quaternion to 3x3 rotation matrix.
 */
SIMSIMD_INTERNAL void _simsimd_quat_to_mat3_f32(simsimd_f32_t const *q_v, simsimd_f32_t *m) {
    simsimd_f32_t w = q_v[3], x = q_v[0], y = q_v[1], z = q_v[2];
    simsimd_f32_t q_xx = x * x, q_yy = y * y, q_zz = z * z;
    simsimd_f32_t q_xz = x * z, q_xy = x * y, q_yz = y * z;
    simsimd_f32_t q_wx = w * x, q_wy = w * y, q_wz = w * z;
    m[0] = 1 - 2 * (q_yy + q_zz), m[1] = 2 * (q_xy - q_wz), m[2] = 2 * (q_xz + q_wy);
    m[3] = 2 * (q_xy + q_wz), m[4] = 1 - 2 * (q_xx + q_zz), m[5] = 2 * (q_yz - q_wx);
    m[6] = 2 * (q_xz - q_wy), m[7] = 2 * (q_yz + q_wx), m[8] = 1 - 2 * (q_xx + q_yy);
}

/*  Internal helper: Jacobi eigenanalysis for symmetric 3x3 matrix.
 *  4 iterations of cyclic Jacobi rotations.
 */
SIMSIMD_INTERNAL void _simsimd_jacobi_eigenanalysis_f32(        //
    simsimd_f32_t *s11, simsimd_f32_t *s21, simsimd_f32_t *s22, //
    simsimd_f32_t *s31, simsimd_f32_t *s32, simsimd_f32_t *s33, //
    simsimd_f32_t *q_v) {                                       //
    q_v[0] = 0, q_v[1] = 0, q_v[2] = 0, q_v[3] = 1;
    for (int i = 0; i < 4; i++) {
        _simsimd_jacobi_conjugation_f32(0, 1, 2, s11, s21, s22, s31, s32, s33, q_v);
        _simsimd_jacobi_conjugation_f32(1, 2, 0, s11, s21, s22, s31, s32, s33, q_v);
        _simsimd_jacobi_conjugation_f32(2, 0, 1, s11, s21, s22, s31, s32, s33, q_v);
    }
    // Normalize quaternion to ensure orthogonal rotation matrix
    simsimd_f32_t norm = _simsimd_rsqrt_f32(q_v[0] * q_v[0] + q_v[1] * q_v[1] + q_v[2] * q_v[2] + q_v[3] * q_v[3]);
    q_v[0] *= norm, q_v[1] *= norm, q_v[2] *= norm, q_v[3] *= norm;
}

/*  Internal helper: QR Givens quaternion for rotation.
 */
SIMSIMD_INTERNAL void _simsimd_qr_givens_quat_f32(      //
    simsimd_f32_t a1, simsimd_f32_t a2,                 //
    simsimd_f32_t *cos_half, simsimd_f32_t *sin_half) { //
    simsimd_f32_t a1_sq_plus_a2_sq = a1 * a1 + a2 * a2;
    simsimd_f32_t rho = a1_sq_plus_a2_sq * _simsimd_rsqrt_f32(a1_sq_plus_a2_sq); // sqrt(a1^2 + a2^2)
    rho = a1_sq_plus_a2_sq > SIMSIMD_SVD_EPSILON_F32 ? rho : 0;

    *sin_half = rho > SIMSIMD_SVD_EPSILON_F32 ? a2 : 0;
    simsimd_f32_t abs_a1 = a1 < 0 ? -a1 : a1;
    simsimd_f32_t max_rho = rho > SIMSIMD_SVD_EPSILON_F32 ? rho : SIMSIMD_SVD_EPSILON_F32;
    *cos_half = abs_a1 + max_rho;

    int should_swap = a1 < 0;
    _simsimd_cond_swap_f32(should_swap, sin_half, cos_half);

    simsimd_f32_t w = _simsimd_rsqrt_f32((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half));
    *cos_half *= w, *sin_half *= w;
}

/*  Internal helper: Sort singular values and corresponding V columns.
 */
SIMSIMD_INTERNAL void _simsimd_sort_singular_values_f32(simsimd_f32_t *b, simsimd_f32_t *svd_v) {
    simsimd_f32_t rho1 = b[0] * b[0] + b[3] * b[3] + b[6] * b[6];
    simsimd_f32_t rho2 = b[1] * b[1] + b[4] * b[4] + b[7] * b[7];
    simsimd_f32_t rho3 = b[2] * b[2] + b[5] * b[5] + b[8] * b[8];
    int should_swap;
    // Sort columns by descending singular value magnitude (bubble sort with 3 comparisons)
    should_swap = rho1 < rho2;
    _simsimd_cond_neg_swap_f32(should_swap, &b[0], &b[1]),
        _simsimd_cond_neg_swap_f32(should_swap, &svd_v[0], &svd_v[1]);
    _simsimd_cond_neg_swap_f32(should_swap, &b[3], &b[4]),
        _simsimd_cond_neg_swap_f32(should_swap, &svd_v[3], &svd_v[4]);
    _simsimd_cond_neg_swap_f32(should_swap, &b[6], &b[7]),
        _simsimd_cond_neg_swap_f32(should_swap, &svd_v[6], &svd_v[7]);
    _simsimd_cond_swap_f32(should_swap, &rho1, &rho2);

    should_swap = rho1 < rho3;
    _simsimd_cond_neg_swap_f32(should_swap, &b[0], &b[2]),
        _simsimd_cond_neg_swap_f32(should_swap, &svd_v[0], &svd_v[2]);
    _simsimd_cond_neg_swap_f32(should_swap, &b[3], &b[5]),
        _simsimd_cond_neg_swap_f32(should_swap, &svd_v[3], &svd_v[5]);
    _simsimd_cond_neg_swap_f32(should_swap, &b[6], &b[8]),
        _simsimd_cond_neg_swap_f32(should_swap, &svd_v[6], &svd_v[8]);
    _simsimd_cond_swap_f32(should_swap, &rho1, &rho3);

    should_swap = rho2 < rho3;
    _simsimd_cond_neg_swap_f32(should_swap, &b[1], &b[2]),
        _simsimd_cond_neg_swap_f32(should_swap, &svd_v[1], &svd_v[2]);
    _simsimd_cond_neg_swap_f32(should_swap, &b[4], &b[5]),
        _simsimd_cond_neg_swap_f32(should_swap, &svd_v[4], &svd_v[5]);
    _simsimd_cond_neg_swap_f32(should_swap, &b[7], &b[8]),
        _simsimd_cond_neg_swap_f32(should_swap, &svd_v[7], &svd_v[8]);
}

/*  Internal helper: QR decomposition of 3x3 matrix using Givens rotations.
 *  Outputs Q (orthogonal) and R (upper triangular).
 */
SIMSIMD_INTERNAL void _simsimd_qr_decomposition_f32(simsimd_f32_t const *b, simsimd_f32_t *q, simsimd_f32_t *r) {
    simsimd_f32_t cos_half_1, sin_half_1, cos_half_2, sin_half_2, cos_half_3, sin_half_3;
    simsimd_f32_t cos_theta, sin_theta;
    simsimd_f32_t r_tmp[9], b_tmp[9];

    // First Givens rotation (zero b[3])
    _simsimd_qr_givens_quat_f32(b[0], b[3], &cos_half_1, &sin_half_1);
    cos_theta = 1 - 2 * sin_half_1 * sin_half_1, sin_theta = 2 * cos_half_1 * sin_half_1;
    r_tmp[0] = cos_theta * b[0] + sin_theta * b[3], r_tmp[1] = cos_theta * b[1] + sin_theta * b[4],
    r_tmp[2] = cos_theta * b[2] + sin_theta * b[5];
    r_tmp[3] = -sin_theta * b[0] + cos_theta * b[3], r_tmp[4] = -sin_theta * b[1] + cos_theta * b[4],
    r_tmp[5] = -sin_theta * b[2] + cos_theta * b[5];
    r_tmp[6] = b[6], r_tmp[7] = b[7], r_tmp[8] = b[8];

    // Second Givens rotation (zero r_tmp[6])
    _simsimd_qr_givens_quat_f32(r_tmp[0], r_tmp[6], &cos_half_2, &sin_half_2);
    cos_theta = 1 - 2 * sin_half_2 * sin_half_2, sin_theta = 2 * cos_half_2 * sin_half_2;
    b_tmp[0] = cos_theta * r_tmp[0] + sin_theta * r_tmp[6], b_tmp[1] = cos_theta * r_tmp[1] + sin_theta * r_tmp[7],
    b_tmp[2] = cos_theta * r_tmp[2] + sin_theta * r_tmp[8];
    b_tmp[3] = r_tmp[3], b_tmp[4] = r_tmp[4], b_tmp[5] = r_tmp[5];
    b_tmp[6] = -sin_theta * r_tmp[0] + cos_theta * r_tmp[6], b_tmp[7] = -sin_theta * r_tmp[1] + cos_theta * r_tmp[7],
    b_tmp[8] = -sin_theta * r_tmp[2] + cos_theta * r_tmp[8];

    // Third Givens rotation (zero b_tmp[7])
    _simsimd_qr_givens_quat_f32(b_tmp[4], b_tmp[7], &cos_half_3, &sin_half_3);
    cos_theta = 1 - 2 * sin_half_3 * sin_half_3, sin_theta = 2 * cos_half_3 * sin_half_3;
    r[0] = b_tmp[0], r[1] = b_tmp[1], r[2] = b_tmp[2];
    r[3] = cos_theta * b_tmp[3] + sin_theta * b_tmp[6], r[4] = cos_theta * b_tmp[4] + sin_theta * b_tmp[7],
    r[5] = cos_theta * b_tmp[5] + sin_theta * b_tmp[8];
    r[6] = -sin_theta * b_tmp[3] + cos_theta * b_tmp[6], r[7] = -sin_theta * b_tmp[4] + cos_theta * b_tmp[7],
    r[8] = -sin_theta * b_tmp[5] + cos_theta * b_tmp[8];

    // Construct Q = Q1 * Q2 * Q3 (using closed-form expressions for efficiency)
    simsimd_f32_t sin_half_1_sq = sin_half_1 * sin_half_1, sin_half_2_sq = sin_half_2 * sin_half_2,
                  sin_half_3_sq = sin_half_3 * sin_half_3;
    q[0] = (-1 + 2 * sin_half_1_sq) * (-1 + 2 * sin_half_2_sq);
    q[1] = 4 * cos_half_2 * cos_half_3 * (-1 + 2 * sin_half_1_sq) * sin_half_2 * sin_half_3 +
           2 * cos_half_1 * sin_half_1 * (-1 + 2 * sin_half_3_sq);
    q[2] = 4 * cos_half_1 * cos_half_3 * sin_half_1 * sin_half_3 -
           2 * cos_half_2 * (-1 + 2 * sin_half_1_sq) * sin_half_2 * (-1 + 2 * sin_half_3_sq);
    q[3] = 2 * cos_half_1 * sin_half_1 * (1 - 2 * sin_half_2_sq);
    q[4] = -8 * cos_half_1 * cos_half_2 * cos_half_3 * sin_half_1 * sin_half_2 * sin_half_3 +
           (-1 + 2 * sin_half_1_sq) * (-1 + 2 * sin_half_3_sq);
    q[5] = -2 * cos_half_3 * sin_half_3 +
           4 * sin_half_1 *
               (cos_half_3 * sin_half_1 * sin_half_3 + cos_half_1 * cos_half_2 * sin_half_2 * (-1 + 2 * sin_half_3_sq));
    q[6] = 2 * cos_half_2 * sin_half_2;
    q[7] = 2 * cos_half_3 * (1 - 2 * sin_half_2_sq) * sin_half_3;
    q[8] = (-1 + 2 * sin_half_2_sq) * (-1 + 2 * sin_half_3_sq);
}

/*  Internal helper: Compute SVD of 3x3 matrix A = U * S * V^T
 *  Using the McAdams algorithm with fixed 4 Jacobi iterations.
 */
SIMSIMD_INTERNAL void _simsimd_svd3x3_f32(simsimd_f32_t const *a, simsimd_f32_t *svd_u, simsimd_f32_t *svd_s,
                                          simsimd_f32_t *svd_v) {
    // Compute A^T * A (symmetric)
    simsimd_f32_t ata[9];
    ata[0] = a[0] * a[0] + a[3] * a[3] + a[6] * a[6];
    ata[1] = a[0] * a[1] + a[3] * a[4] + a[6] * a[7];
    ata[2] = a[0] * a[2] + a[3] * a[5] + a[6] * a[8];
    ata[3] = ata[1];
    ata[4] = a[1] * a[1] + a[4] * a[4] + a[7] * a[7];
    ata[5] = a[1] * a[2] + a[4] * a[5] + a[7] * a[8];
    ata[6] = ata[2];
    ata[7] = ata[5];
    ata[8] = a[2] * a[2] + a[5] * a[5] + a[8] * a[8];

    // Jacobi eigenanalysis of A^T * A
    simsimd_f32_t q_v[4];
    _simsimd_jacobi_eigenanalysis_f32(&ata[0], &ata[1], &ata[4], &ata[2], &ata[5], &ata[8], q_v);
    _simsimd_quat_to_mat3_f32(q_v, svd_v);

    // B = A * V
    simsimd_f32_t b[9];
    b[0] = a[0] * svd_v[0] + a[1] * svd_v[3] + a[2] * svd_v[6];
    b[1] = a[0] * svd_v[1] + a[1] * svd_v[4] + a[2] * svd_v[7];
    b[2] = a[0] * svd_v[2] + a[1] * svd_v[5] + a[2] * svd_v[8];
    b[3] = a[3] * svd_v[0] + a[4] * svd_v[3] + a[5] * svd_v[6];
    b[4] = a[3] * svd_v[1] + a[4] * svd_v[4] + a[5] * svd_v[7];
    b[5] = a[3] * svd_v[2] + a[4] * svd_v[5] + a[5] * svd_v[8];
    b[6] = a[6] * svd_v[0] + a[7] * svd_v[3] + a[8] * svd_v[6];
    b[7] = a[6] * svd_v[1] + a[7] * svd_v[4] + a[8] * svd_v[7];
    b[8] = a[6] * svd_v[2] + a[7] * svd_v[5] + a[8] * svd_v[8];

    // Sort singular values and update V
    _simsimd_sort_singular_values_f32(b, svd_v);

    // QR decomposition: B = U * S
    _simsimd_qr_decomposition_f32(b, svd_u, svd_s);
}

/*  Internal helper: Compute determinant of 3x3 matrix.
 */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_det3x3_f32(simsimd_f32_t const *m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

/*  RMSD (Root Mean Square Deviation) without optimal superposition.
 *  Simply computes the RMS of distances between corresponding points.
 */
#define SIMSIMD_MAKE_RMSD(name, input_type, accumulator_type, rotation_type, load_and_convert)          \
    SIMSIMD_PUBLIC void simsimd_rmsd_##input_type##_##name(                                             \
        simsimd_##input_type##_t const *a, simsimd_##input_type##_t const *b, simsimd_size_t n,         \
        simsimd_##input_type##_t *a_centroid, simsimd_##input_type##_t *b_centroid,                     \
        simsimd_##rotation_type##_t *rotation, simsimd_distance_t *scale, simsimd_distance_t *result) { \
        simsimd_##accumulator_type##_t sum_a_x = 0, sum_a_y = 0, sum_a_z = 0;                           \
        simsimd_##accumulator_type##_t sum_b_x = 0, sum_b_y = 0, sum_b_z = 0;                           \
        simsimd_##accumulator_type##_t val_a_x, val_a_y, val_a_z, val_b_x, val_b_y, val_b_z;            \
        for (simsimd_size_t i = 0; i < n; ++i) {                                                        \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(a + i * 3 + 1, &val_a_y);       \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 0, &val_b_x);       \
            load_and_convert(b + i * 3 + 1, &val_b_y), load_and_convert(b + i * 3 + 2, &val_b_z);       \
            sum_a_x += val_a_x, sum_a_y += val_a_y, sum_a_z += val_a_z;                                 \
            sum_b_x += val_b_x, sum_b_y += val_b_y, sum_b_z += val_b_z;                                 \
        }                                                                                               \
        simsimd_##accumulator_type##_t inv_n = (simsimd_##accumulator_type##_t)1.0 / n;                 \
        simsimd_##accumulator_type##_t centroid_a_x = sum_a_x * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_a_y = sum_a_y * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_a_z = sum_a_z * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_b_x = sum_b_x * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_b_y = sum_b_y * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_b_z = sum_b_z * inv_n;                                  \
        if (a_centroid) {                                                                               \
            a_centroid[0] = (simsimd_##input_type##_t)centroid_a_x;                                     \
            a_centroid[1] = (simsimd_##input_type##_t)centroid_a_y;                                     \
            a_centroid[2] = (simsimd_##input_type##_t)centroid_a_z;                                     \
        }                                                                                               \
        if (b_centroid) {                                                                               \
            b_centroid[0] = (simsimd_##input_type##_t)centroid_b_x;                                     \
            b_centroid[1] = (simsimd_##input_type##_t)centroid_b_y;                                     \
            b_centroid[2] = (simsimd_##input_type##_t)centroid_b_z;                                     \
        }                                                                                               \
        /* RMSD uses identity rotation and scale=1.0 */                                                 \
        if (rotation) {                                                                                 \
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;                                          \
            rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;                                          \
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;                                          \
        }                                                                                               \
        if (scale) *scale = 1.0;                                                                        \
        simsimd_##accumulator_type##_t sum_squared = 0;                                                 \
        for (simsimd_size_t i = 0; i < n; ++i) {                                                        \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(b + i * 3 + 0, &val_b_x);       \
            load_and_convert(a + i * 3 + 1, &val_a_y), load_and_convert(b + i * 3 + 1, &val_b_y);       \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 2, &val_b_z);       \
            simsimd_##accumulator_type##_t dx = (val_a_x - centroid_a_x) - (val_b_x - centroid_b_x);    \
            simsimd_##accumulator_type##_t dy = (val_a_y - centroid_a_y) - (val_b_y - centroid_b_y);    \
            simsimd_##accumulator_type##_t dz = (val_a_z - centroid_a_z) - (val_b_z - centroid_b_z);    \
            sum_squared += dx * dx + dy * dy + dz * dz;                                                 \
        }                                                                                               \
        *result = SIMSIMD_SQRT(sum_squared * inv_n);                                                    \
    }

/*  Kabsch algorithm for optimal rigid body superposition.
 *  Finds the rotation matrix R that minimizes RMSD between the two point sets.
 */
#define SIMSIMD_MAKE_KABSCH(name, input_type, accumulator_type, rotation_type, load_and_convert)        \
    SIMSIMD_PUBLIC void simsimd_kabsch_##input_type##_##name(                                           \
        simsimd_##input_type##_t const *a, simsimd_##input_type##_t const *b, simsimd_size_t n,         \
        simsimd_##input_type##_t *a_centroid, simsimd_##input_type##_t *b_centroid,                     \
        simsimd_##rotation_type##_t *rotation, simsimd_distance_t *scale, simsimd_distance_t *result) { \
        /* Step 1: Compute centroids */                                                                 \
        simsimd_##accumulator_type##_t sum_a_x = 0, sum_a_y = 0, sum_a_z = 0;                           \
        simsimd_##accumulator_type##_t sum_b_x = 0, sum_b_y = 0, sum_b_z = 0;                           \
        simsimd_##accumulator_type##_t val_a_x, val_a_y, val_a_z, val_b_x, val_b_y, val_b_z;            \
        for (simsimd_size_t i = 0; i < n; ++i) {                                                        \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(a + i * 3 + 1, &val_a_y);       \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 0, &val_b_x);       \
            load_and_convert(b + i * 3 + 1, &val_b_y), load_and_convert(b + i * 3 + 2, &val_b_z);       \
            sum_a_x += val_a_x, sum_a_y += val_a_y, sum_a_z += val_a_z;                                 \
            sum_b_x += val_b_x, sum_b_y += val_b_y, sum_b_z += val_b_z;                                 \
        }                                                                                               \
        simsimd_##accumulator_type##_t inv_n = (simsimd_##accumulator_type##_t)1.0 / n;                 \
        simsimd_##accumulator_type##_t centroid_a_x = sum_a_x * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_a_y = sum_a_y * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_a_z = sum_a_z * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_b_x = sum_b_x * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_b_y = sum_b_y * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_b_z = sum_b_z * inv_n;                                  \
        if (a_centroid) {                                                                               \
            a_centroid[0] = (simsimd_##input_type##_t)centroid_a_x;                                     \
            a_centroid[1] = (simsimd_##input_type##_t)centroid_a_y;                                     \
            a_centroid[2] = (simsimd_##input_type##_t)centroid_a_z;                                     \
        }                                                                                               \
        if (b_centroid) {                                                                               \
            b_centroid[0] = (simsimd_##input_type##_t)centroid_b_x;                                     \
            b_centroid[1] = (simsimd_##input_type##_t)centroid_b_y;                                     \
            b_centroid[2] = (simsimd_##input_type##_t)centroid_b_z;                                     \
        }                                                                                               \
        /* Step 2: Build 3x3 covariance matrix H = (A - centroid_A)^T * (B - centroid_B) */             \
        /* Use accumulator_type for high-precision accumulation */                                      \
        simsimd_##accumulator_type##_t h[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};                              \
        for (simsimd_size_t i = 0; i < n; ++i) {                                                        \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(b + i * 3 + 0, &val_b_x);       \
            load_and_convert(a + i * 3 + 1, &val_a_y), load_and_convert(b + i * 3 + 1, &val_b_y);       \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 2, &val_b_z);       \
            val_a_x -= centroid_a_x, val_a_y -= centroid_a_y, val_a_z -= centroid_a_z;                  \
            val_b_x -= centroid_b_x, val_b_y -= centroid_b_y, val_b_z -= centroid_b_z;                  \
            h[0] += val_a_x * val_b_x, h[1] += val_a_x * val_b_y, h[2] += val_a_x * val_b_z;            \
            h[3] += val_a_y * val_b_x, h[4] += val_a_y * val_b_y, h[5] += val_a_y * val_b_z;            \
            h[6] += val_a_z * val_b_x, h[7] += val_a_z * val_b_y, h[8] += val_a_z * val_b_z;            \
        }                                                                                               \
        /* Convert to f32 for SVD (SVD precision is adequate at f32) */                                 \
        simsimd_f32_t cross_covariance[9];                                                              \
        for (int j = 0; j < 9; ++j) cross_covariance[j] = (simsimd_f32_t)h[j];                          \
        /* Step 3: SVD of H = U * S * V^T */                                                            \
        simsimd_f32_t svd_u[9], svd_s[9], svd_v[9];                                                     \
        _simsimd_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);                                     \
        /* Step 4: R = V * U^T */                                                                       \
        simsimd_f32_t rotation_matrix[9];                                                               \
        rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];           \
        rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];           \
        rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];           \
        rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];           \
        rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];           \
        rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];           \
        rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];           \
        rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];           \
        rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];           \
        /* Handle reflection: if det(R) < 0, negate third column of V and recompute R */                \
        simsimd_f32_t rotation_det = _simsimd_det3x3_f32(rotation_matrix);                              \
        if (rotation_det < 0) {                                                                         \
            svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];                           \
            rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];       \
            rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];       \
            rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];       \
            rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];       \
            rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];       \
            rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];       \
            rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];       \
            rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];       \
            rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];       \
        }                                                                                               \
        /* Output rotation matrix and scale=1.0 */                                                      \
        if (rotation) {                                                                                 \
            for (int j = 0; j < 9; ++j) rotation[j] = (simsimd_##rotation_type##_t)rotation_matrix[j];  \
        }                                                                                               \
        if (scale) *scale = 1.0;                                                                        \
        /* Step 5: Compute RMSD after rotation */                                                       \
        simsimd_##accumulator_type##_t sum_squared = 0;                                                 \
        for (simsimd_size_t i = 0; i < n; ++i) {                                                        \
            simsimd_f32_t point_a[3], point_b[3], rotated_point_a[3];                                   \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(a + i * 3 + 1, &val_a_y);       \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 0, &val_b_x);       \
            load_and_convert(b + i * 3 + 1, &val_b_y), load_and_convert(b + i * 3 + 2, &val_b_z);       \
            point_a[0] = (simsimd_f32_t)(val_a_x - centroid_a_x);                                       \
            point_a[1] = (simsimd_f32_t)(val_a_y - centroid_a_y);                                       \
            point_a[2] = (simsimd_f32_t)(val_a_z - centroid_a_z);                                       \
            point_b[0] = (simsimd_f32_t)(val_b_x - centroid_b_x);                                       \
            point_b[1] = (simsimd_f32_t)(val_b_y - centroid_b_y);                                       \
            point_b[2] = (simsimd_f32_t)(val_b_z - centroid_b_z);                                       \
            rotated_point_a[0] = rotation_matrix[0] * point_a[0] + rotation_matrix[1] * point_a[1] +    \
                                 rotation_matrix[2] * point_a[2];                                       \
            rotated_point_a[1] = rotation_matrix[3] * point_a[0] + rotation_matrix[4] * point_a[1] +    \
                                 rotation_matrix[5] * point_a[2];                                       \
            rotated_point_a[2] = rotation_matrix[6] * point_a[0] + rotation_matrix[7] * point_a[1] +    \
                                 rotation_matrix[8] * point_a[2];                                       \
            simsimd_f32_t dx = rotated_point_a[0] - point_b[0];                                         \
            simsimd_f32_t dy = rotated_point_a[1] - point_b[1];                                         \
            simsimd_f32_t dz = rotated_point_a[2] - point_b[2];                                         \
            sum_squared += dx * dx + dy * dy + dz * dz;                                                 \
        }                                                                                               \
        *result = SIMSIMD_SQRT(sum_squared * inv_n);                                                    \
    }

/*  Umeyama algorithm for optimal similarity transformation (rotation + uniform scale).
 *  Finds the rotation matrix R and scale factor c that minimizes ||c*R*A - B||.
 *  Reference: S. Umeyama, "Least-squares estimation of transformation parameters
 *  between two point patterns", IEEE TPAMI 1991.
 */
#define SIMSIMD_MAKE_UMEYAMA(name, input_type, accumulator_type, rotation_type, load_and_convert)       \
    SIMSIMD_PUBLIC void simsimd_umeyama_##input_type##_##name(                                          \
        simsimd_##input_type##_t const *a, simsimd_##input_type##_t const *b, simsimd_size_t n,         \
        simsimd_##input_type##_t *a_centroid, simsimd_##input_type##_t *b_centroid,                     \
        simsimd_##rotation_type##_t *rotation, simsimd_distance_t *scale, simsimd_distance_t *result) { \
        /* Step 1: Compute centroids */                                                                 \
        simsimd_##accumulator_type##_t sum_a_x = 0, sum_a_y = 0, sum_a_z = 0;                           \
        simsimd_##accumulator_type##_t sum_b_x = 0, sum_b_y = 0, sum_b_z = 0;                           \
        simsimd_##accumulator_type##_t val_a_x, val_a_y, val_a_z, val_b_x, val_b_y, val_b_z;            \
        for (simsimd_size_t i = 0; i < n; ++i) {                                                        \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(a + i * 3 + 1, &val_a_y);       \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 0, &val_b_x);       \
            load_and_convert(b + i * 3 + 1, &val_b_y), load_and_convert(b + i * 3 + 2, &val_b_z);       \
            sum_a_x += val_a_x, sum_a_y += val_a_y, sum_a_z += val_a_z;                                 \
            sum_b_x += val_b_x, sum_b_y += val_b_y, sum_b_z += val_b_z;                                 \
        }                                                                                               \
        simsimd_##accumulator_type##_t inv_n = (simsimd_##accumulator_type##_t)1.0 / n;                 \
        simsimd_##accumulator_type##_t centroid_a_x = sum_a_x * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_a_y = sum_a_y * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_a_z = sum_a_z * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_b_x = sum_b_x * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_b_y = sum_b_y * inv_n;                                  \
        simsimd_##accumulator_type##_t centroid_b_z = sum_b_z * inv_n;                                  \
        if (a_centroid) {                                                                               \
            a_centroid[0] = (simsimd_##input_type##_t)centroid_a_x;                                     \
            a_centroid[1] = (simsimd_##input_type##_t)centroid_a_y;                                     \
            a_centroid[2] = (simsimd_##input_type##_t)centroid_a_z;                                     \
        }                                                                                               \
        if (b_centroid) {                                                                               \
            b_centroid[0] = (simsimd_##input_type##_t)centroid_b_x;                                     \
            b_centroid[1] = (simsimd_##input_type##_t)centroid_b_y;                                     \
            b_centroid[2] = (simsimd_##input_type##_t)centroid_b_z;                                     \
        }                                                                                               \
        /* Step 2: Build covariance matrix H and compute variance of A */                               \
        simsimd_##accumulator_type##_t h[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};                              \
        simsimd_##accumulator_type##_t variance_a = 0;                                                  \
        for (simsimd_size_t i = 0; i < n; ++i) {                                                        \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(b + i * 3 + 0, &val_b_x);       \
            load_and_convert(a + i * 3 + 1, &val_a_y), load_and_convert(b + i * 3 + 1, &val_b_y);       \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 2, &val_b_z);       \
            val_a_x -= centroid_a_x, val_a_y -= centroid_a_y, val_a_z -= centroid_a_z;                  \
            val_b_x -= centroid_b_x, val_b_y -= centroid_b_y, val_b_z -= centroid_b_z;                  \
            variance_a += val_a_x * val_a_x + val_a_y * val_a_y + val_a_z * val_a_z;                    \
            h[0] += val_a_x * val_b_x, h[1] += val_a_x * val_b_y, h[2] += val_a_x * val_b_z;            \
            h[3] += val_a_y * val_b_x, h[4] += val_a_y * val_b_y, h[5] += val_a_y * val_b_z;            \
            h[6] += val_a_z * val_b_x, h[7] += val_a_z * val_b_y, h[8] += val_a_z * val_b_z;            \
        }                                                                                               \
        variance_a *= inv_n;                                                                            \
        /* Convert to f32 for SVD */                                                                    \
        simsimd_f32_t cross_covariance[9];                                                              \
        for (int j = 0; j < 9; ++j) cross_covariance[j] = (simsimd_f32_t)h[j];                          \
        /* Step 3: SVD of H = U * S * V^T */                                                            \
        simsimd_f32_t svd_u[9], svd_s[9], svd_v[9];                                                     \
        _simsimd_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);                                     \
        /* Step 4: R = V * U^T */                                                                       \
        simsimd_f32_t rotation_matrix[9];                                                               \
        rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];           \
        rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];           \
        rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];           \
        rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];           \
        rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];           \
        rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];           \
        rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];           \
        rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];           \
        rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];           \
        /* Handle reflection and compute scale: c = trace(D*S) / variance_a */                          \
        /* D = diag(1, 1, det(R)), singular values are svd_s[0], svd_s[4], svd_s[8] (diagonal of S) */  \
        simsimd_f32_t rotation_det = _simsimd_det3x3_f32(rotation_matrix);                              \
        simsimd_f32_t sign_det = rotation_det < 0 ? -1.0f : 1.0f;                                       \
        simsimd_f32_t trace_scaled_s = svd_s[0] + svd_s[4] + sign_det * svd_s[8];                       \
        simsimd_##accumulator_type##_t scale_factor = (simsimd_##accumulator_type##_t)trace_scaled_s /  \
                                                      ((simsimd_##accumulator_type##_t)n * variance_a); \
        if (scale) *scale = scale_factor;                                                               \
        if (rotation_det < 0) {                                                                         \
            svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];                           \
            rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];       \
            rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];       \
            rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];       \
            rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];       \
            rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];       \
            rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];       \
            rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];       \
            rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];       \
            rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];       \
        }                                                                                               \
        /* Output rotation matrix */                                                                    \
        if (rotation) {                                                                                 \
            for (int j = 0; j < 9; ++j) rotation[j] = (simsimd_##rotation_type##_t)rotation_matrix[j];  \
        }                                                                                               \
        /* Step 5: Compute RMSD after similarity transform: ||c*R*a - b|| */                            \
        simsimd_##accumulator_type##_t sum_squared = 0;                                                 \
        for (simsimd_size_t i = 0; i < n; ++i) {                                                        \
            simsimd_f32_t point_a[3], point_b[3], rotated_point_a[3];                                   \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(a + i * 3 + 1, &val_a_y);       \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 0, &val_b_x);       \
            load_and_convert(b + i * 3 + 1, &val_b_y), load_and_convert(b + i * 3 + 2, &val_b_z);       \
            point_a[0] = (simsimd_f32_t)(val_a_x - centroid_a_x);                                       \
            point_a[1] = (simsimd_f32_t)(val_a_y - centroid_a_y);                                       \
            point_a[2] = (simsimd_f32_t)(val_a_z - centroid_a_z);                                       \
            point_b[0] = (simsimd_f32_t)(val_b_x - centroid_b_x);                                       \
            point_b[1] = (simsimd_f32_t)(val_b_y - centroid_b_y);                                       \
            point_b[2] = (simsimd_f32_t)(val_b_z - centroid_b_z);                                       \
            rotated_point_a[0] = (simsimd_f32_t)scale_factor *                                          \
                                 (rotation_matrix[0] * point_a[0] + rotation_matrix[1] * point_a[1] +   \
                                  rotation_matrix[2] * point_a[2]);                                     \
            rotated_point_a[1] = (simsimd_f32_t)scale_factor *                                          \
                                 (rotation_matrix[3] * point_a[0] + rotation_matrix[4] * point_a[1] +   \
                                  rotation_matrix[5] * point_a[2]);                                     \
            rotated_point_a[2] = (simsimd_f32_t)scale_factor *                                          \
                                 (rotation_matrix[6] * point_a[0] + rotation_matrix[7] * point_a[1] +   \
                                  rotation_matrix[8] * point_a[2]);                                     \
            simsimd_f32_t dx = rotated_point_a[0] - point_b[0];                                         \
            simsimd_f32_t dy = rotated_point_a[1] - point_b[1];                                         \
            simsimd_f32_t dz = rotated_point_a[2] - point_b[2];                                         \
            sum_squared += dx * dx + dy * dy + dz * dz;                                                 \
        }                                                                                               \
        *result = SIMSIMD_SQRT(sum_squared * inv_n);                                                    \
    }

SIMSIMD_MAKE_RMSD(serial, f64, f64, f64, SIMSIMD_ASSIGN_FROM_TO)    // simsimd_rmsd_f64_serial
SIMSIMD_MAKE_KABSCH(serial, f64, f64, f64, SIMSIMD_ASSIGN_FROM_TO)  // simsimd_kabsch_f64_serial
SIMSIMD_MAKE_UMEYAMA(serial, f64, f64, f64, SIMSIMD_ASSIGN_FROM_TO) // simsimd_umeyama_f64_serial

SIMSIMD_MAKE_RMSD(serial, f32, f32, f32, SIMSIMD_ASSIGN_FROM_TO)    // simsimd_rmsd_f32_serial
SIMSIMD_MAKE_KABSCH(serial, f32, f32, f32, SIMSIMD_ASSIGN_FROM_TO)  // simsimd_kabsch_f32_serial
SIMSIMD_MAKE_UMEYAMA(serial, f32, f32, f32, SIMSIMD_ASSIGN_FROM_TO) // simsimd_umeyama_f32_serial

SIMSIMD_MAKE_RMSD(serial, f16, f32, f32, simsimd_f16_to_f32)    // simsimd_rmsd_f16_serial
SIMSIMD_MAKE_KABSCH(serial, f16, f32, f32, simsimd_f16_to_f32)  // simsimd_kabsch_f16_serial
SIMSIMD_MAKE_UMEYAMA(serial, f16, f32, f32, simsimd_f16_to_f32) // simsimd_umeyama_f16_serial

SIMSIMD_MAKE_RMSD(serial, bf16, f32, f32, simsimd_bf16_to_f32)    // simsimd_rmsd_bf16_serial
SIMSIMD_MAKE_KABSCH(serial, bf16, f32, f32, simsimd_bf16_to_f32)  // simsimd_kabsch_bf16_serial
SIMSIMD_MAKE_UMEYAMA(serial, bf16, f32, f32, simsimd_bf16_to_f32) // simsimd_umeyama_bf16_serial

SIMSIMD_MAKE_RMSD(accurate, f32, f64, f32, SIMSIMD_ASSIGN_FROM_TO)    // simsimd_rmsd_f32_accurate
SIMSIMD_MAKE_KABSCH(accurate, f32, f64, f32, SIMSIMD_ASSIGN_FROM_TO)  // simsimd_kabsch_f32_accurate
SIMSIMD_MAKE_UMEYAMA(accurate, f32, f64, f32, SIMSIMD_ASSIGN_FROM_TO) // simsimd_umeyama_f32_accurate

SIMSIMD_MAKE_RMSD(accurate, f16, f64, f64, simsimd_f16_to_f64)    // simsimd_rmsd_f16_accurate
SIMSIMD_MAKE_KABSCH(accurate, f16, f64, f64, simsimd_f16_to_f64)  // simsimd_kabsch_f16_accurate
SIMSIMD_MAKE_UMEYAMA(accurate, f16, f64, f64, simsimd_f16_to_f64) // simsimd_umeyama_f16_accurate

SIMSIMD_MAKE_RMSD(accurate, bf16, f64, f64, simsimd_bf16_to_f64)    // simsimd_rmsd_bf16_accurate
SIMSIMD_MAKE_KABSCH(accurate, bf16, f64, f64, simsimd_bf16_to_f64)  // simsimd_kabsch_bf16_accurate
SIMSIMD_MAKE_UMEYAMA(accurate, bf16, f64, f64, simsimd_bf16_to_f64) // simsimd_umeyama_bf16_accurate

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "avx512bw", "avx512dq", "bmi2")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,avx512bw,avx512dq,bmi2"))), apply_to = function)

/*  Internal helper: Deinterleave 48 floats (16 xyz triplets) into separate x, y, z vectors.
 *  Uses permutex2var shuffles instead of gather for ~1.8x speedup.
 *
 *  Input: 48 contiguous floats [x0,y0,z0, x1,y1,z1, ..., x15,y15,z15]
 *  Output: x[16], y[16], z[16] vectors
 *
 *  Implementation: Load 3 registers (r0,r1,r2), use 6 permutex2var ops to separate.
 *  Phase analysis: r0 starts at float 0 (phase 0), r1 at float 16 (phase 1), r2 at float 32 (phase 2)
 *
 *  X elements at memory positions: 0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45
 *    = r0[0,3,6,9,12,15], r1[2,5,8,11,14], r2[1,4,7,10,13]
 *  Y elements at memory positions: 1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46
 *    = r0[1,4,7,10,13], r1[0,3,6,9,12,15], r2[2,5,8,11,14]
 *  Z elements at memory positions: 2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47
 *    = r0[2,5,8,11,14], r1[1,4,7,10,13], r2[0,3,6,9,12,15]
 */
SIMSIMD_INTERNAL void _simsimd_deinterleave_f32x16_skylake(simsimd_f32_t const *ptr, __m512 *x_out, __m512 *y_out,
                                                           __m512 *z_out) {
    __m512 r0 = _mm512_loadu_ps(ptr);
    __m512 r1 = _mm512_loadu_ps(ptr + 16);
    __m512 r2 = _mm512_loadu_ps(ptr + 32);

    // X: r0[0,3,6,9,12,15] + r1[2,5,8,11,14] -> 11 elements, then + r2[1,4,7,10,13] -> 16 elements
    // Indices for permutex2var: 0-15 = from first operand, 16-31 = from second operand
    __m512i idx_x_01 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
    __m512i idx_x_2 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29);
    __m512 x01 = _mm512_permutex2var_ps(r0, idx_x_01, r1);
    *x_out = _mm512_permutex2var_ps(x01, idx_x_2, r2);

    // Y: r0[1,4,7,10,13] + r1[0,3,6,9,12,15] -> 11 elements, then + r2[2,5,8,11,14] -> 16 elements
    __m512i idx_y_01 = _mm512_setr_epi32(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0);
    __m512i idx_y_2 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30);
    __m512 y01 = _mm512_permutex2var_ps(r0, idx_y_01, r1);
    *y_out = _mm512_permutex2var_ps(y01, idx_y_2, r2);

    // Z: r0[2,5,8,11,14] + r1[1,4,7,10,13] -> 10 elements, then + r2[0,3,6,9,12,15] -> 16 elements
    __m512i idx_z_01 = _mm512_setr_epi32(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0);
    __m512i idx_z_2 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31);
    __m512 z01 = _mm512_permutex2var_ps(r0, idx_z_01, r1);
    *z_out = _mm512_permutex2var_ps(z01, idx_z_2, r2);
}

/*  Internal helper: Deinterleave 8 f64 3D points from xyz,xyz,xyz... to separate x,y,z vectors.
 *  Input: 24 consecutive f64 values (8 points * 3 coordinates)
 *  Output: Three __m512d vectors containing the x, y, z coordinates separately.
 */
SIMSIMD_INTERNAL void _simsimd_deinterleave_f64x8_skylake(simsimd_f64_t const *ptr, __m512d *x_out, __m512d *y_out,
                                                          __m512d *z_out) {
    __m512d r0 = _mm512_loadu_pd(ptr);      // elements 0-7
    __m512d r1 = _mm512_loadu_pd(ptr + 8);  // elements 8-15
    __m512d r2 = _mm512_loadu_pd(ptr + 16); // elements 16-23

    // X: positions 0,3,6,9,12,15,18,21 -> r0[0,3,6] + r1[1,4,7] + r2[2,5]
    __m512i idx_x_01 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 0, 0);
    __m512i idx_x_2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 10, 13);
    __m512d x01 = _mm512_permutex2var_pd(r0, idx_x_01, r1);
    *x_out = _mm512_permutex2var_pd(x01, idx_x_2, r2);

    // Y: positions 1,4,7,10,13,16,19,22 -> r0[1,4,7] + r1[2,5] + r2[0,3,6]
    __m512i idx_y_01 = _mm512_setr_epi64(1, 4, 7, 10, 13, 0, 0, 0);
    __m512i idx_y_2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 8, 11, 14);
    __m512d y01 = _mm512_permutex2var_pd(r0, idx_y_01, r1);
    *y_out = _mm512_permutex2var_pd(y01, idx_y_2, r2);

    // Z: positions 2,5,8,11,14,17,20,23 -> r0[2,5] + r1[0,3,6] + r2[1,4,7]
    __m512i idx_z_01 = _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0);
    __m512i idx_z_2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 9, 12, 15);
    __m512d z01 = _mm512_permutex2var_pd(r0, idx_z_01, r1);
    *z_out = _mm512_permutex2var_pd(z01, idx_z_2, r2);
}

SIMSIMD_PUBLIC void simsimd_rmsd_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid,
                                             simsimd_f32_t *rotation, simsimd_distance_t *scale,
                                             simsimd_distance_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1;
        rotation[1] = 0;
        rotation[2] = 0;
        rotation[3] = 0;
        rotation[4] = 1;
        rotation[5] = 0;
        rotation[6] = 0;
        rotation[7] = 0;
        rotation[8] = 1;
    }
    if (scale) *scale = 1.0;
    // Optimized fused single-pass implementation.
    // Computes centroids and squared differences in one pass using the identity:
    //   RMSD = sqrt(E[(a-mean_a) - (b-mean_b)]^2)
    //        = sqrt(E[(a-b)^2] - (mean_a - mean_b)^2)
    __m512i const gather_idx = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
    __m512 const zeros = _mm512_setzero_ps();

    // Accumulators for centroids and squared differences
    __m512 sum_a_x = zeros, sum_a_y = zeros, sum_a_z = zeros;
    __m512 sum_b_x = zeros, sum_b_y = zeros, sum_b_z = zeros;
    __m512 sum_squared_x = zeros, sum_squared_y = zeros, sum_squared_z = zeros;

    __m512 a_x, a_y, a_z, b_x, b_y, b_z;
    simsimd_size_t i = 0;

    // Main loop with 2x unrolling for better latency hiding
    for (; i + 32 <= n; i += 32) {
        // Iteration 0
        _simsimd_deinterleave_f32x16_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f32x16_skylake(b + i * 3, &b_x, &b_y, &b_z);

        sum_a_x = _mm512_add_ps(sum_a_x, a_x);
        sum_a_y = _mm512_add_ps(sum_a_y, a_y);
        sum_a_z = _mm512_add_ps(sum_a_z, a_z);
        sum_b_x = _mm512_add_ps(sum_b_x, b_x);
        sum_b_y = _mm512_add_ps(sum_b_y, b_y);
        sum_b_z = _mm512_add_ps(sum_b_z, b_z);

        __m512 delta_x = _mm512_sub_ps(a_x, b_x);
        __m512 delta_y = _mm512_sub_ps(a_y, b_y);
        __m512 delta_z = _mm512_sub_ps(a_z, b_z);

        sum_squared_x = _mm512_fmadd_ps(delta_x, delta_x, sum_squared_x);
        sum_squared_y = _mm512_fmadd_ps(delta_y, delta_y, sum_squared_y);
        sum_squared_z = _mm512_fmadd_ps(delta_z, delta_z, sum_squared_z);

        // Iteration 1
        __m512 a_x1, a_y1, a_z1, b_x1, b_y1, b_z1;
        _simsimd_deinterleave_f32x16_skylake(a + (i + 16) * 3, &a_x1, &a_y1, &a_z1);
        _simsimd_deinterleave_f32x16_skylake(b + (i + 16) * 3, &b_x1, &b_y1, &b_z1);

        sum_a_x = _mm512_add_ps(sum_a_x, a_x1);
        sum_a_y = _mm512_add_ps(sum_a_y, a_y1);
        sum_a_z = _mm512_add_ps(sum_a_z, a_z1);
        sum_b_x = _mm512_add_ps(sum_b_x, b_x1);
        sum_b_y = _mm512_add_ps(sum_b_y, b_y1);
        sum_b_z = _mm512_add_ps(sum_b_z, b_z1);

        __m512 delta_x1 = _mm512_sub_ps(a_x1, b_x1);
        __m512 delta_y1 = _mm512_sub_ps(a_y1, b_y1);
        __m512 delta_z1 = _mm512_sub_ps(a_z1, b_z1);

        sum_squared_x = _mm512_fmadd_ps(delta_x1, delta_x1, sum_squared_x);
        sum_squared_y = _mm512_fmadd_ps(delta_y1, delta_y1, sum_squared_y);
        sum_squared_z = _mm512_fmadd_ps(delta_z1, delta_z1, sum_squared_z);
    }

    // Handle 16-point remainder
    for (; i + 16 <= n; i += 16) {
        _simsimd_deinterleave_f32x16_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f32x16_skylake(b + i * 3, &b_x, &b_y, &b_z);

        sum_a_x = _mm512_add_ps(sum_a_x, a_x);
        sum_a_y = _mm512_add_ps(sum_a_y, a_y);
        sum_a_z = _mm512_add_ps(sum_a_z, a_z);
        sum_b_x = _mm512_add_ps(sum_b_x, b_x);
        sum_b_y = _mm512_add_ps(sum_b_y, b_y);
        sum_b_z = _mm512_add_ps(sum_b_z, b_z);

        __m512 delta_x = _mm512_sub_ps(a_x, b_x);
        __m512 delta_y = _mm512_sub_ps(a_y, b_y);
        __m512 delta_z = _mm512_sub_ps(a_z, b_z);

        sum_squared_x = _mm512_fmadd_ps(delta_x, delta_x, sum_squared_x);
        sum_squared_y = _mm512_fmadd_ps(delta_y, delta_y, sum_squared_y);
        sum_squared_z = _mm512_fmadd_ps(delta_z, delta_z, sum_squared_z);
    }

    // Tail: use masked gather
    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail);
        simsimd_f32_t const *a_tail = a + i * 3;
        simsimd_f32_t const *b_tail = b + i * 3;

        a_x = _mm512_mask_i32gather_ps(zeros, mask, gather_idx, a_tail + 0, 4);
        a_y = _mm512_mask_i32gather_ps(zeros, mask, gather_idx, a_tail + 1, 4);
        a_z = _mm512_mask_i32gather_ps(zeros, mask, gather_idx, a_tail + 2, 4);
        b_x = _mm512_mask_i32gather_ps(zeros, mask, gather_idx, b_tail + 0, 4);
        b_y = _mm512_mask_i32gather_ps(zeros, mask, gather_idx, b_tail + 1, 4);
        b_z = _mm512_mask_i32gather_ps(zeros, mask, gather_idx, b_tail + 2, 4);

        sum_a_x = _mm512_add_ps(sum_a_x, a_x);
        sum_a_y = _mm512_add_ps(sum_a_y, a_y);
        sum_a_z = _mm512_add_ps(sum_a_z, a_z);
        sum_b_x = _mm512_add_ps(sum_b_x, b_x);
        sum_b_y = _mm512_add_ps(sum_b_y, b_y);
        sum_b_z = _mm512_add_ps(sum_b_z, b_z);

        __m512 delta_x = _mm512_sub_ps(a_x, b_x);
        __m512 delta_y = _mm512_sub_ps(a_y, b_y);
        __m512 delta_z = _mm512_sub_ps(a_z, b_z);

        sum_squared_x = _mm512_fmadd_ps(delta_x, delta_x, sum_squared_x);
        sum_squared_y = _mm512_fmadd_ps(delta_y, delta_y, sum_squared_y);
        sum_squared_z = _mm512_fmadd_ps(delta_z, delta_z, sum_squared_z);
    }

    // Reduce and compute centroids
    simsimd_f32_t inv_n = 1.0f / (simsimd_f32_t)n;
    simsimd_f32_t centroid_a_x = _mm512_reduce_add_ps(sum_a_x) * inv_n;
    simsimd_f32_t centroid_a_y = _mm512_reduce_add_ps(sum_a_y) * inv_n;
    simsimd_f32_t centroid_a_z = _mm512_reduce_add_ps(sum_a_z) * inv_n;
    simsimd_f32_t centroid_b_x = _mm512_reduce_add_ps(sum_b_x) * inv_n;
    simsimd_f32_t centroid_b_y = _mm512_reduce_add_ps(sum_b_y) * inv_n;
    simsimd_f32_t centroid_b_z = _mm512_reduce_add_ps(sum_b_z) * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Compute RMSD using the formula:
    // RMSD = sqrt(E[(a-b)^2] - (mean_a - mean_b)^2)
    simsimd_f32_t mean_diff_x = centroid_a_x - centroid_b_x;
    simsimd_f32_t mean_diff_y = centroid_a_y - centroid_b_y;
    simsimd_f32_t mean_diff_z = centroid_a_z - centroid_b_z;

    __m512 sum_squared_total = _mm512_add_ps(sum_squared_x, _mm512_add_ps(sum_squared_y, sum_squared_z));
    simsimd_f32_t sum_squared = _mm512_reduce_add_ps(sum_squared_total);
    simsimd_f32_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = SIMSIMD_SQRT((simsimd_distance_t)(sum_squared * inv_n - mean_diff_sq));
}

SIMSIMD_PUBLIC void simsimd_kabsch_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                               simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid,
                                               simsimd_f32_t *rotation, simsimd_distance_t *scale,
                                               simsimd_distance_t *result) {
    // Optimized fused single-pass implementation.
    // Computes centroids and covariance matrix in one pass using the identity:
    //   H_ij = sum((a_i - mean_a) * (b_j - mean_b))
    //        = sum(a_i * b_j) - sum(a_i) * sum(b_j) / n
    __m512i const gather_idx = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
    __m512 const zeros_f32 = _mm512_setzero_ps();
    __m512d const zeros_f64 = _mm512_setzero_pd();

    // Accumulators for centroids
    __m512d sum_a_x_vec = zeros_f64, sum_a_y_vec = zeros_f64, sum_a_z_vec = zeros_f64;
    __m512d sum_b_x_vec = zeros_f64, sum_b_y_vec = zeros_f64, sum_b_z_vec = zeros_f64;

    // Accumulators for covariance matrix (sum of outer products)
    __m512d cov_xx = zeros_f64, cov_xy = zeros_f64, cov_xz = zeros_f64;
    __m512d cov_yx = zeros_f64, cov_yy = zeros_f64, cov_yz = zeros_f64;
    __m512d cov_zx = zeros_f64, cov_zy = zeros_f64, cov_zz = zeros_f64;

    simsimd_size_t i = 0;
    __m512 a_x_vec, a_y_vec, a_z_vec, b_x_vec, b_y_vec, b_z_vec;

    // Fused single-pass: accumulate sums and outer products together
    for (; i + 16 <= n; i += 16) {
        _simsimd_deinterleave_f32x16_skylake(a + i * 3, &a_x_vec, &a_y_vec, &a_z_vec);
        _simsimd_deinterleave_f32x16_skylake(b + i * 3, &b_x_vec, &b_y_vec, &b_z_vec);

        // Convert to f64 - low 8 elements
        __m512d a_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_vec));
        __m512d a_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_vec));
        __m512d a_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_vec));
        __m512d b_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_vec));
        __m512d b_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_vec));
        __m512d b_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_vec));

        // Accumulate centroids
        sum_a_x_vec = _mm512_add_pd(sum_a_x_vec, a_x_lo);
        sum_a_y_vec = _mm512_add_pd(sum_a_y_vec, a_y_lo);
        sum_a_z_vec = _mm512_add_pd(sum_a_z_vec, a_z_lo);
        sum_b_x_vec = _mm512_add_pd(sum_b_x_vec, b_x_lo);
        sum_b_y_vec = _mm512_add_pd(sum_b_y_vec, b_y_lo);
        sum_b_z_vec = _mm512_add_pd(sum_b_z_vec, b_z_lo);

        // Accumulate outer products (raw, not centered)
        cov_xx = _mm512_fmadd_pd(a_x_lo, b_x_lo, cov_xx);
        cov_xy = _mm512_fmadd_pd(a_x_lo, b_y_lo, cov_xy);
        cov_xz = _mm512_fmadd_pd(a_x_lo, b_z_lo, cov_xz);
        cov_yx = _mm512_fmadd_pd(a_y_lo, b_x_lo, cov_yx);
        cov_yy = _mm512_fmadd_pd(a_y_lo, b_y_lo, cov_yy);
        cov_yz = _mm512_fmadd_pd(a_y_lo, b_z_lo, cov_yz);
        cov_zx = _mm512_fmadd_pd(a_z_lo, b_x_lo, cov_zx);
        cov_zy = _mm512_fmadd_pd(a_z_lo, b_y_lo, cov_zy);
        cov_zz = _mm512_fmadd_pd(a_z_lo, b_z_lo, cov_zz);

        // High 8 elements
        __m512d a_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_vec, 1));
        __m512d a_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_vec, 1));
        __m512d a_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_vec, 1));
        __m512d b_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_vec, 1));
        __m512d b_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_vec, 1));
        __m512d b_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_vec, 1));

        sum_a_x_vec = _mm512_add_pd(sum_a_x_vec, a_x_hi);
        sum_a_y_vec = _mm512_add_pd(sum_a_y_vec, a_y_hi);
        sum_a_z_vec = _mm512_add_pd(sum_a_z_vec, a_z_hi);
        sum_b_x_vec = _mm512_add_pd(sum_b_x_vec, b_x_hi);
        sum_b_y_vec = _mm512_add_pd(sum_b_y_vec, b_y_hi);
        sum_b_z_vec = _mm512_add_pd(sum_b_z_vec, b_z_hi);

        cov_xx = _mm512_fmadd_pd(a_x_hi, b_x_hi, cov_xx);
        cov_xy = _mm512_fmadd_pd(a_x_hi, b_y_hi, cov_xy);
        cov_xz = _mm512_fmadd_pd(a_x_hi, b_z_hi, cov_xz);
        cov_yx = _mm512_fmadd_pd(a_y_hi, b_x_hi, cov_yx);
        cov_yy = _mm512_fmadd_pd(a_y_hi, b_y_hi, cov_yy);
        cov_yz = _mm512_fmadd_pd(a_y_hi, b_z_hi, cov_yz);
        cov_zx = _mm512_fmadd_pd(a_z_hi, b_x_hi, cov_zx);
        cov_zy = _mm512_fmadd_pd(a_z_hi, b_y_hi, cov_zy);
        cov_zz = _mm512_fmadd_pd(a_z_hi, b_z_hi, cov_zz);
    }

    // Tail: masked gather for remaining points
    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail);
        simsimd_f32_t const *a_tail = a + i * 3;
        simsimd_f32_t const *b_tail = b + i * 3;
        a_x_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 0, 4);
        a_y_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 1, 4);
        a_z_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 2, 4);
        b_x_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 0, 4);
        b_y_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 1, 4);
        b_z_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 2, 4);

        __m512d a_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_vec));
        __m512d a_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_vec));
        __m512d a_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_vec));
        __m512d b_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_vec));
        __m512d b_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_vec));
        __m512d b_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_vec));

        sum_a_x_vec = _mm512_add_pd(sum_a_x_vec, a_x_lo);
        sum_a_y_vec = _mm512_add_pd(sum_a_y_vec, a_y_lo);
        sum_a_z_vec = _mm512_add_pd(sum_a_z_vec, a_z_lo);
        sum_b_x_vec = _mm512_add_pd(sum_b_x_vec, b_x_lo);
        sum_b_y_vec = _mm512_add_pd(sum_b_y_vec, b_y_lo);
        sum_b_z_vec = _mm512_add_pd(sum_b_z_vec, b_z_lo);

        cov_xx = _mm512_fmadd_pd(a_x_lo, b_x_lo, cov_xx);
        cov_xy = _mm512_fmadd_pd(a_x_lo, b_y_lo, cov_xy);
        cov_xz = _mm512_fmadd_pd(a_x_lo, b_z_lo, cov_xz);
        cov_yx = _mm512_fmadd_pd(a_y_lo, b_x_lo, cov_yx);
        cov_yy = _mm512_fmadd_pd(a_y_lo, b_y_lo, cov_yy);
        cov_yz = _mm512_fmadd_pd(a_y_lo, b_z_lo, cov_yz);
        cov_zx = _mm512_fmadd_pd(a_z_lo, b_x_lo, cov_zx);
        cov_zy = _mm512_fmadd_pd(a_z_lo, b_y_lo, cov_zy);
        cov_zz = _mm512_fmadd_pd(a_z_lo, b_z_lo, cov_zz);

        __m512d a_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_vec, 1));
        __m512d a_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_vec, 1));
        __m512d a_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_vec, 1));
        __m512d b_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_vec, 1));
        __m512d b_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_vec, 1));
        __m512d b_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_vec, 1));

        sum_a_x_vec = _mm512_add_pd(sum_a_x_vec, a_x_hi);
        sum_a_y_vec = _mm512_add_pd(sum_a_y_vec, a_y_hi);
        sum_a_z_vec = _mm512_add_pd(sum_a_z_vec, a_z_hi);
        sum_b_x_vec = _mm512_add_pd(sum_b_x_vec, b_x_hi);
        sum_b_y_vec = _mm512_add_pd(sum_b_y_vec, b_y_hi);
        sum_b_z_vec = _mm512_add_pd(sum_b_z_vec, b_z_hi);

        cov_xx = _mm512_fmadd_pd(a_x_hi, b_x_hi, cov_xx);
        cov_xy = _mm512_fmadd_pd(a_x_hi, b_y_hi, cov_xy);
        cov_xz = _mm512_fmadd_pd(a_x_hi, b_z_hi, cov_xz);
        cov_yx = _mm512_fmadd_pd(a_y_hi, b_x_hi, cov_yx);
        cov_yy = _mm512_fmadd_pd(a_y_hi, b_y_hi, cov_yy);
        cov_yz = _mm512_fmadd_pd(a_y_hi, b_z_hi, cov_yz);
        cov_zx = _mm512_fmadd_pd(a_z_hi, b_x_hi, cov_zx);
        cov_zy = _mm512_fmadd_pd(a_z_hi, b_y_hi, cov_zy);
        cov_zz = _mm512_fmadd_pd(a_z_hi, b_z_hi, cov_zz);
    }

    // Reduce centroids
    simsimd_f64_t inv_n_d = 1.0 / (simsimd_f64_t)n;
    simsimd_f64_t sum_a_x = _mm512_reduce_add_pd(sum_a_x_vec);
    simsimd_f64_t sum_a_y = _mm512_reduce_add_pd(sum_a_y_vec);
    simsimd_f64_t sum_a_z = _mm512_reduce_add_pd(sum_a_z_vec);
    simsimd_f64_t sum_b_x = _mm512_reduce_add_pd(sum_b_x_vec);
    simsimd_f64_t sum_b_y = _mm512_reduce_add_pd(sum_b_y_vec);
    simsimd_f64_t sum_b_z = _mm512_reduce_add_pd(sum_b_z_vec);

    simsimd_f32_t centroid_a_x = (simsimd_f32_t)(sum_a_x * inv_n_d);
    simsimd_f32_t centroid_a_y = (simsimd_f32_t)(sum_a_y * inv_n_d);
    simsimd_f32_t centroid_a_z = (simsimd_f32_t)(sum_a_z * inv_n_d);
    simsimd_f32_t centroid_b_x = (simsimd_f32_t)(sum_b_x * inv_n_d);
    simsimd_f32_t centroid_b_y = (simsimd_f32_t)(sum_b_y * inv_n_d);
    simsimd_f32_t centroid_b_z = (simsimd_f32_t)(sum_b_z * inv_n_d);

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Compute centered covariance matrix: H_ij = sum(a_i*b_j) - sum_a_i * sum_b_j / n
    simsimd_f32_t cross_covariance[9];
    cross_covariance[0] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_xx) - sum_a_x * sum_b_x * inv_n_d);
    cross_covariance[1] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_xy) - sum_a_x * sum_b_y * inv_n_d);
    cross_covariance[2] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_xz) - sum_a_x * sum_b_z * inv_n_d);
    cross_covariance[3] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_yx) - sum_a_y * sum_b_x * inv_n_d);
    cross_covariance[4] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_yy) - sum_a_y * sum_b_y * inv_n_d);
    cross_covariance[5] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_yz) - sum_a_y * sum_b_z * inv_n_d);
    cross_covariance[6] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_zx) - sum_a_z * sum_b_x * inv_n_d);
    cross_covariance[7] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_zy) - sum_a_z * sum_b_y * inv_n_d);
    cross_covariance[8] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_zz) - sum_a_z * sum_b_z * inv_n_d);

    // Step 3: SVD
    simsimd_f32_t svd_u[9], svd_s[9], svd_v[9];
    _simsimd_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);

    // Step 4: R = V * U^T
    simsimd_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection
    simsimd_f32_t det = _simsimd_det3x3_f32(r);
    if (det < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix and scale=1.0 */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }
    if (scale) *scale = 1.0;

    // Step 5: Compute RMSD after rotation using shuffle-based deinterleave
    __m512d sum_squared_vec = zeros_f64;

    __m512 r0_vec = _mm512_set1_ps(r[0]), r1_vec = _mm512_set1_ps(r[1]), r2_vec = _mm512_set1_ps(r[2]);
    __m512 r3_vec = _mm512_set1_ps(r[3]), r4_vec = _mm512_set1_ps(r[4]), r5_vec = _mm512_set1_ps(r[5]);
    __m512 r6_vec = _mm512_set1_ps(r[6]), r7_vec = _mm512_set1_ps(r[7]), r8_vec = _mm512_set1_ps(r[8]);
    __m512 centroid_a_x_vec = _mm512_set1_ps(centroid_a_x), centroid_a_y_vec = _mm512_set1_ps(centroid_a_y),
           centroid_a_z_vec = _mm512_set1_ps(centroid_a_z);
    __m512 centroid_b_x_vec = _mm512_set1_ps(centroid_b_x), centroid_b_y_vec = _mm512_set1_ps(centroid_b_y),
           centroid_b_z_vec = _mm512_set1_ps(centroid_b_z);

    // Main loop with shuffle-based deinterleave
    for (i = 0; i + 16 <= n; i += 16) {
        _simsimd_deinterleave_f32x16_skylake(a + i * 3, &a_x_vec, &a_y_vec, &a_z_vec);
        _simsimd_deinterleave_f32x16_skylake(b + i * 3, &b_x_vec, &b_y_vec, &b_z_vec);

        // Center points
        a_x_vec = _mm512_sub_ps(a_x_vec, centroid_a_x_vec);
        a_y_vec = _mm512_sub_ps(a_y_vec, centroid_a_y_vec);
        a_z_vec = _mm512_sub_ps(a_z_vec, centroid_a_z_vec);
        b_x_vec = _mm512_sub_ps(b_x_vec, centroid_b_x_vec);
        b_y_vec = _mm512_sub_ps(b_y_vec, centroid_b_y_vec);
        b_z_vec = _mm512_sub_ps(b_z_vec, centroid_b_z_vec);

        // R * a_centered
        __m512 rotated_a_x_vec = _mm512_fmadd_ps(r0_vec, a_x_vec,
                                                 _mm512_fmadd_ps(r1_vec, a_y_vec, _mm512_mul_ps(r2_vec, a_z_vec)));
        __m512 rotated_a_y_vec = _mm512_fmadd_ps(r3_vec, a_x_vec,
                                                 _mm512_fmadd_ps(r4_vec, a_y_vec, _mm512_mul_ps(r5_vec, a_z_vec)));
        __m512 rotated_a_z_vec = _mm512_fmadd_ps(r6_vec, a_x_vec,
                                                 _mm512_fmadd_ps(r7_vec, a_y_vec, _mm512_mul_ps(r8_vec, a_z_vec)));

        __m512 delta_x_vec = _mm512_sub_ps(rotated_a_x_vec, b_x_vec);
        __m512 delta_y_vec = _mm512_sub_ps(rotated_a_y_vec, b_y_vec);
        __m512 delta_z_vec = _mm512_sub_ps(rotated_a_z_vec, b_z_vec);

        // Accumulate in f64 for precision - low 8 elements
        __m512d delta_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(delta_x_vec));
        __m512d delta_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(delta_y_vec));
        __m512d delta_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(delta_z_vec));
        sum_squared_vec = _mm512_fmadd_pd(delta_x_lo, delta_x_lo, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_y_lo, delta_y_lo, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_z_lo, delta_z_lo, sum_squared_vec);
        // High 8 elements
        __m512d delta_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(delta_x_vec, 1));
        __m512d delta_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(delta_y_vec, 1));
        __m512d delta_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(delta_z_vec, 1));
        sum_squared_vec = _mm512_fmadd_pd(delta_x_hi, delta_x_hi, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_y_hi, delta_y_hi, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_z_hi, delta_z_hi, sum_squared_vec);
    }

    // Tail with masked gather
    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail);
        simsimd_f32_t const *a_tail = a + i * 3;
        simsimd_f32_t const *b_tail = b + i * 3;
        a_x_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 0, 4);
        a_y_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 1, 4);
        a_z_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 2, 4);
        b_x_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 0, 4);
        b_y_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 1, 4);
        b_z_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 2, 4);

        a_x_vec = _mm512_sub_ps(a_x_vec, centroid_a_x_vec);
        a_y_vec = _mm512_sub_ps(a_y_vec, centroid_a_y_vec);
        a_z_vec = _mm512_sub_ps(a_z_vec, centroid_a_z_vec);
        b_x_vec = _mm512_sub_ps(b_x_vec, centroid_b_x_vec);
        b_y_vec = _mm512_sub_ps(b_y_vec, centroid_b_y_vec);
        b_z_vec = _mm512_sub_ps(b_z_vec, centroid_b_z_vec);

        __m512 rotated_a_x_vec = _mm512_fmadd_ps(r0_vec, a_x_vec,
                                                 _mm512_fmadd_ps(r1_vec, a_y_vec, _mm512_mul_ps(r2_vec, a_z_vec)));
        __m512 rotated_a_y_vec = _mm512_fmadd_ps(r3_vec, a_x_vec,
                                                 _mm512_fmadd_ps(r4_vec, a_y_vec, _mm512_mul_ps(r5_vec, a_z_vec)));
        __m512 rotated_a_z_vec = _mm512_fmadd_ps(r6_vec, a_x_vec,
                                                 _mm512_fmadd_ps(r7_vec, a_y_vec, _mm512_mul_ps(r8_vec, a_z_vec)));

        __m512 delta_x_vec = _mm512_sub_ps(rotated_a_x_vec, b_x_vec);
        __m512 delta_y_vec = _mm512_sub_ps(rotated_a_y_vec, b_y_vec);
        __m512 delta_z_vec = _mm512_sub_ps(rotated_a_z_vec, b_z_vec);

        __m512d delta_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(delta_x_vec));
        __m512d delta_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(delta_y_vec));
        __m512d delta_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(delta_z_vec));
        sum_squared_vec = _mm512_fmadd_pd(delta_x_lo, delta_x_lo, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_y_lo, delta_y_lo, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_z_lo, delta_z_lo, sum_squared_vec);
        __m512d delta_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(delta_x_vec, 1));
        __m512d delta_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(delta_y_vec, 1));
        __m512d delta_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(delta_z_vec, 1));
        sum_squared_vec = _mm512_fmadd_pd(delta_x_hi, delta_x_hi, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_y_hi, delta_y_hi, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_z_hi, delta_z_hi, sum_squared_vec);
    }

    *result = SIMSIMD_SQRT((simsimd_distance_t)_mm512_reduce_add_pd(sum_squared_vec) * inv_n_d);
}

SIMSIMD_PUBLIC void simsimd_rmsd_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                             simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid,
                                             simsimd_f64_t *rotation, simsimd_distance_t *scale,
                                             simsimd_distance_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1;
        rotation[1] = 0;
        rotation[2] = 0;
        rotation[3] = 0;
        rotation[4] = 1;
        rotation[5] = 0;
        rotation[6] = 0;
        rotation[7] = 0;
        rotation[8] = 1;
    }
    if (scale) *scale = 1.0;
    // Optimized fused single-pass implementation for f64.
    // Computes centroids and squared differences in one pass using the identity:
    //   RMSD = sqrt(E[(a-mean_a) - (b-mean_b)]^2)
    //        = sqrt(E[(a-b)^2] - (mean_a - mean_b)^2)
    __m512i const gather_idx = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 18, 21);
    __m512d const zeros = _mm512_setzero_pd();

    // Accumulators for centroids and squared differences
    __m512d sum_a_x = zeros, sum_a_y = zeros, sum_a_z = zeros;
    __m512d sum_b_x = zeros, sum_b_y = zeros, sum_b_z = zeros;
    __m512d sum_squared_x = zeros, sum_squared_y = zeros, sum_squared_z = zeros;

    __m512d a_x, a_y, a_z, b_x, b_y, b_z;
    simsimd_size_t i = 0;

    // Main loop with 2x unrolling for better latency hiding
    for (; i + 16 <= n; i += 16) {
        // Iteration 0
        _simsimd_deinterleave_f64x8_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x8_skylake(b + i * 3, &b_x, &b_y, &b_z);

        sum_a_x = _mm512_add_pd(sum_a_x, a_x), sum_a_y = _mm512_add_pd(sum_a_y, a_y),
        sum_a_z = _mm512_add_pd(sum_a_z, a_z);
        sum_b_x = _mm512_add_pd(sum_b_x, b_x), sum_b_y = _mm512_add_pd(sum_b_y, b_y),
        sum_b_z = _mm512_add_pd(sum_b_z, b_z);

        __m512d delta_x = _mm512_sub_pd(a_x, b_x), delta_y = _mm512_sub_pd(a_y, b_y), delta_z = _mm512_sub_pd(a_z, b_z);
        sum_squared_x = _mm512_fmadd_pd(delta_x, delta_x, sum_squared_x);
        sum_squared_y = _mm512_fmadd_pd(delta_y, delta_y, sum_squared_y);
        sum_squared_z = _mm512_fmadd_pd(delta_z, delta_z, sum_squared_z);

        // Iteration 1
        __m512d a_x1, a_y1, a_z1, b_x1, b_y1, b_z1;
        _simsimd_deinterleave_f64x8_skylake(a + (i + 8) * 3, &a_x1, &a_y1, &a_z1);
        _simsimd_deinterleave_f64x8_skylake(b + (i + 8) * 3, &b_x1, &b_y1, &b_z1);

        sum_a_x = _mm512_add_pd(sum_a_x, a_x1), sum_a_y = _mm512_add_pd(sum_a_y, a_y1),
        sum_a_z = _mm512_add_pd(sum_a_z, a_z1);
        sum_b_x = _mm512_add_pd(sum_b_x, b_x1), sum_b_y = _mm512_add_pd(sum_b_y, b_y1),
        sum_b_z = _mm512_add_pd(sum_b_z, b_z1);

        __m512d delta_x1 = _mm512_sub_pd(a_x1, b_x1), delta_y1 = _mm512_sub_pd(a_y1, b_y1),
                delta_z1 = _mm512_sub_pd(a_z1, b_z1);
        sum_squared_x = _mm512_fmadd_pd(delta_x1, delta_x1, sum_squared_x);
        sum_squared_y = _mm512_fmadd_pd(delta_y1, delta_y1, sum_squared_y);
        sum_squared_z = _mm512_fmadd_pd(delta_z1, delta_z1, sum_squared_z);
    }

    // Handle 8-point remainder
    for (; i + 8 <= n; i += 8) {
        _simsimd_deinterleave_f64x8_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x8_skylake(b + i * 3, &b_x, &b_y, &b_z);

        sum_a_x = _mm512_add_pd(sum_a_x, a_x), sum_a_y = _mm512_add_pd(sum_a_y, a_y),
        sum_a_z = _mm512_add_pd(sum_a_z, a_z);
        sum_b_x = _mm512_add_pd(sum_b_x, b_x), sum_b_y = _mm512_add_pd(sum_b_y, b_y),
        sum_b_z = _mm512_add_pd(sum_b_z, b_z);

        __m512d delta_x = _mm512_sub_pd(a_x, b_x), delta_y = _mm512_sub_pd(a_y, b_y), delta_z = _mm512_sub_pd(a_z, b_z);
        sum_squared_x = _mm512_fmadd_pd(delta_x, delta_x, sum_squared_x);
        sum_squared_y = _mm512_fmadd_pd(delta_y, delta_y, sum_squared_y);
        sum_squared_z = _mm512_fmadd_pd(delta_z, delta_z, sum_squared_z);
    }

    // Tail: use masked gather
    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        simsimd_f64_t const *a_tail = a + i * 3;
        simsimd_f64_t const *b_tail = b + i * 3;

        a_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 0, 8);
        a_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 1, 8);
        a_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 2, 8);
        b_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 0, 8);
        b_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 1, 8);
        b_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 2, 8);

        sum_a_x = _mm512_add_pd(sum_a_x, a_x), sum_a_y = _mm512_add_pd(sum_a_y, a_y),
        sum_a_z = _mm512_add_pd(sum_a_z, a_z);
        sum_b_x = _mm512_add_pd(sum_b_x, b_x), sum_b_y = _mm512_add_pd(sum_b_y, b_y),
        sum_b_z = _mm512_add_pd(sum_b_z, b_z);

        __m512d delta_x = _mm512_sub_pd(a_x, b_x), delta_y = _mm512_sub_pd(a_y, b_y), delta_z = _mm512_sub_pd(a_z, b_z);
        sum_squared_x = _mm512_fmadd_pd(delta_x, delta_x, sum_squared_x);
        sum_squared_y = _mm512_fmadd_pd(delta_y, delta_y, sum_squared_y);
        sum_squared_z = _mm512_fmadd_pd(delta_z, delta_z, sum_squared_z);
    }

    // Reduce and compute centroids
    simsimd_f64_t inv_n = 1.0 / (simsimd_f64_t)n;
    simsimd_f64_t centroid_a_x = _mm512_reduce_add_pd(sum_a_x) * inv_n;
    simsimd_f64_t centroid_a_y = _mm512_reduce_add_pd(sum_a_y) * inv_n;
    simsimd_f64_t centroid_a_z = _mm512_reduce_add_pd(sum_a_z) * inv_n;
    simsimd_f64_t centroid_b_x = _mm512_reduce_add_pd(sum_b_x) * inv_n;
    simsimd_f64_t centroid_b_y = _mm512_reduce_add_pd(sum_b_y) * inv_n;
    simsimd_f64_t centroid_b_z = _mm512_reduce_add_pd(sum_b_z) * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute RMSD using the formula:
    // RMSD = sqrt(E[(a-b)^2] - (mean_a - mean_b)^2)
    simsimd_f64_t mean_diff_x = centroid_a_x - centroid_b_x, mean_diff_y = centroid_a_y - centroid_b_y,
                  mean_diff_z = centroid_a_z - centroid_b_z;
    __m512d sum_squared_total = _mm512_add_pd(sum_squared_x, _mm512_add_pd(sum_squared_y, sum_squared_z));
    simsimd_f64_t sum_squared = _mm512_reduce_add_pd(sum_squared_total);
    simsimd_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = SIMSIMD_SQRT((simsimd_distance_t)(sum_squared * inv_n - mean_diff_sq));
}

SIMSIMD_PUBLIC void simsimd_kabsch_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                               simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid,
                                               simsimd_f64_t *rotation, simsimd_distance_t *scale,
                                               simsimd_distance_t *result) {
    // Optimized fused single-pass implementation for f64.
    // Computes centroids and covariance matrix in one pass using the identity:
    //   H_ij = sum((a_i - mean_a) * (b_j - mean_b))
    //        = sum(a_i * b_j) - sum(a_i) * sum(b_j) / n
    __m512i const gather_idx = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 18, 21);
    __m512d const zeros = _mm512_setzero_pd();

    // Accumulators for centroids
    __m512d sum_a_x_vec = zeros, sum_a_y_vec = zeros, sum_a_z_vec = zeros;
    __m512d sum_b_x_vec = zeros, sum_b_y_vec = zeros, sum_b_z_vec = zeros;

    // Accumulators for covariance matrix (sum of outer products)
    __m512d cov_xx = zeros, cov_xy = zeros, cov_xz = zeros;
    __m512d cov_yx = zeros, cov_yy = zeros, cov_yz = zeros;
    __m512d cov_zx = zeros, cov_zy = zeros, cov_zz = zeros;

    simsimd_size_t i = 0;
    __m512d a_x, a_y, a_z, b_x, b_y, b_z;

    // Fused single-pass: accumulate sums and outer products together
    for (; i + 8 <= n; i += 8) {
        _simsimd_deinterleave_f64x8_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x8_skylake(b + i * 3, &b_x, &b_y, &b_z);

        // Accumulate centroids
        sum_a_x_vec = _mm512_add_pd(sum_a_x_vec, a_x), sum_a_y_vec = _mm512_add_pd(sum_a_y_vec, a_y),
        sum_a_z_vec = _mm512_add_pd(sum_a_z_vec, a_z);
        sum_b_x_vec = _mm512_add_pd(sum_b_x_vec, b_x), sum_b_y_vec = _mm512_add_pd(sum_b_y_vec, b_y),
        sum_b_z_vec = _mm512_add_pd(sum_b_z_vec, b_z);

        // Accumulate outer products (raw, not centered)
        cov_xx = _mm512_fmadd_pd(a_x, b_x, cov_xx), cov_xy = _mm512_fmadd_pd(a_x, b_y, cov_xy),
        cov_xz = _mm512_fmadd_pd(a_x, b_z, cov_xz);
        cov_yx = _mm512_fmadd_pd(a_y, b_x, cov_yx), cov_yy = _mm512_fmadd_pd(a_y, b_y, cov_yy),
        cov_yz = _mm512_fmadd_pd(a_y, b_z, cov_yz);
        cov_zx = _mm512_fmadd_pd(a_z, b_x, cov_zx), cov_zy = _mm512_fmadd_pd(a_z, b_y, cov_zy),
        cov_zz = _mm512_fmadd_pd(a_z, b_z, cov_zz);
    }

    // Tail: masked gather for remaining points
    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        simsimd_f64_t const *a_tail = a + i * 3;
        simsimd_f64_t const *b_tail = b + i * 3;

        a_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 0, 8);
        a_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 1, 8);
        a_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 2, 8);
        b_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 0, 8);
        b_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 1, 8);
        b_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 2, 8);

        sum_a_x_vec = _mm512_add_pd(sum_a_x_vec, a_x), sum_a_y_vec = _mm512_add_pd(sum_a_y_vec, a_y),
        sum_a_z_vec = _mm512_add_pd(sum_a_z_vec, a_z);
        sum_b_x_vec = _mm512_add_pd(sum_b_x_vec, b_x), sum_b_y_vec = _mm512_add_pd(sum_b_y_vec, b_y),
        sum_b_z_vec = _mm512_add_pd(sum_b_z_vec, b_z);

        cov_xx = _mm512_fmadd_pd(a_x, b_x, cov_xx), cov_xy = _mm512_fmadd_pd(a_x, b_y, cov_xy),
        cov_xz = _mm512_fmadd_pd(a_x, b_z, cov_xz);
        cov_yx = _mm512_fmadd_pd(a_y, b_x, cov_yx), cov_yy = _mm512_fmadd_pd(a_y, b_y, cov_yy),
        cov_yz = _mm512_fmadd_pd(a_y, b_z, cov_yz);
        cov_zx = _mm512_fmadd_pd(a_z, b_x, cov_zx), cov_zy = _mm512_fmadd_pd(a_z, b_y, cov_zy),
        cov_zz = _mm512_fmadd_pd(a_z, b_z, cov_zz);
    }

    // Reduce centroids
    simsimd_f64_t inv_n = 1.0 / (simsimd_f64_t)n;
    simsimd_f64_t sum_a_x = _mm512_reduce_add_pd(sum_a_x_vec), sum_a_y = _mm512_reduce_add_pd(sum_a_y_vec),
                  sum_a_z = _mm512_reduce_add_pd(sum_a_z_vec);
    simsimd_f64_t sum_b_x = _mm512_reduce_add_pd(sum_b_x_vec), sum_b_y = _mm512_reduce_add_pd(sum_b_y_vec),
                  sum_b_z = _mm512_reduce_add_pd(sum_b_z_vec);

    simsimd_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    simsimd_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance matrix: H_ij = sum(a_i*b_j) - sum_a_i * sum_b_j / n
    simsimd_f32_t cross_covariance[9];
    cross_covariance[0] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_xx) - sum_a_x * sum_b_x * inv_n);
    cross_covariance[1] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_xy) - sum_a_x * sum_b_y * inv_n);
    cross_covariance[2] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_xz) - sum_a_x * sum_b_z * inv_n);
    cross_covariance[3] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_yx) - sum_a_y * sum_b_x * inv_n);
    cross_covariance[4] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_yy) - sum_a_y * sum_b_y * inv_n);
    cross_covariance[5] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_yz) - sum_a_y * sum_b_z * inv_n);
    cross_covariance[6] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_zx) - sum_a_z * sum_b_x * inv_n);
    cross_covariance[7] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_zy) - sum_a_z * sum_b_y * inv_n);
    cross_covariance[8] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_zz) - sum_a_z * sum_b_z * inv_n);

    // SVD (f32 is sufficient for rotation matrix)
    simsimd_f32_t svd_u[9], svd_s[9], svd_v[9];
    _simsimd_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * U^T
    simsimd_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection
    if (_simsimd_det3x3_f32(r) < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix and scale=1.0 */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = (simsimd_f64_t)r[j];
    }
    if (scale) *scale = 1.0;

    // Compute RMSD after rotation using f64 throughout
    __m512d sum_squared_vec = zeros;
    __m512d r0_vec = _mm512_set1_pd(r[0]), r1_vec = _mm512_set1_pd(r[1]), r2_vec = _mm512_set1_pd(r[2]);
    __m512d r3_vec = _mm512_set1_pd(r[3]), r4_vec = _mm512_set1_pd(r[4]), r5_vec = _mm512_set1_pd(r[5]);
    __m512d r6_vec = _mm512_set1_pd(r[6]), r7_vec = _mm512_set1_pd(r[7]), r8_vec = _mm512_set1_pd(r[8]);
    __m512d centroid_a_x_vec = _mm512_set1_pd(centroid_a_x), centroid_a_y_vec = _mm512_set1_pd(centroid_a_y),
            centroid_a_z_vec = _mm512_set1_pd(centroid_a_z);
    __m512d centroid_b_x_vec = _mm512_set1_pd(centroid_b_x), centroid_b_y_vec = _mm512_set1_pd(centroid_b_y),
            centroid_b_z_vec = _mm512_set1_pd(centroid_b_z);

    for (i = 0; i + 8 <= n; i += 8) {
        _simsimd_deinterleave_f64x8_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x8_skylake(b + i * 3, &b_x, &b_y, &b_z);

        // Center points
        a_x = _mm512_sub_pd(a_x, centroid_a_x_vec), a_y = _mm512_sub_pd(a_y, centroid_a_y_vec),
        a_z = _mm512_sub_pd(a_z, centroid_a_z_vec);
        b_x = _mm512_sub_pd(b_x, centroid_b_x_vec), b_y = _mm512_sub_pd(b_y, centroid_b_y_vec),
        b_z = _mm512_sub_pd(b_z, centroid_b_z_vec);

        // R * a_centered
        __m512d rotated_a_x = _mm512_fmadd_pd(r0_vec, a_x, _mm512_fmadd_pd(r1_vec, a_y, _mm512_mul_pd(r2_vec, a_z)));
        __m512d rotated_a_y = _mm512_fmadd_pd(r3_vec, a_x, _mm512_fmadd_pd(r4_vec, a_y, _mm512_mul_pd(r5_vec, a_z)));
        __m512d rotated_a_z = _mm512_fmadd_pd(r6_vec, a_x, _mm512_fmadd_pd(r7_vec, a_y, _mm512_mul_pd(r8_vec, a_z)));

        __m512d delta_x = _mm512_sub_pd(rotated_a_x, b_x), delta_y = _mm512_sub_pd(rotated_a_y, b_y),
                delta_z = _mm512_sub_pd(rotated_a_z, b_z);
        sum_squared_vec = _mm512_fmadd_pd(delta_x, delta_x, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_y, delta_y, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_z, delta_z, sum_squared_vec);
    }

    // Tail with masked gather
    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        simsimd_f64_t const *a_tail = a + i * 3;
        simsimd_f64_t const *b_tail = b + i * 3;

        a_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 0, 8);
        a_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 1, 8);
        a_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 2, 8);
        b_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 0, 8);
        b_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 1, 8);
        b_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 2, 8);

        a_x = _mm512_sub_pd(a_x, centroid_a_x_vec), a_y = _mm512_sub_pd(a_y, centroid_a_y_vec),
        a_z = _mm512_sub_pd(a_z, centroid_a_z_vec);
        b_x = _mm512_sub_pd(b_x, centroid_b_x_vec), b_y = _mm512_sub_pd(b_y, centroid_b_y_vec),
        b_z = _mm512_sub_pd(b_z, centroid_b_z_vec);

        __m512d rotated_a_x = _mm512_fmadd_pd(r0_vec, a_x, _mm512_fmadd_pd(r1_vec, a_y, _mm512_mul_pd(r2_vec, a_z)));
        __m512d rotated_a_y = _mm512_fmadd_pd(r3_vec, a_x, _mm512_fmadd_pd(r4_vec, a_y, _mm512_mul_pd(r5_vec, a_z)));
        __m512d rotated_a_z = _mm512_fmadd_pd(r6_vec, a_x, _mm512_fmadd_pd(r7_vec, a_y, _mm512_mul_pd(r8_vec, a_z)));

        __m512d delta_x = _mm512_sub_pd(rotated_a_x, b_x), delta_y = _mm512_sub_pd(rotated_a_y, b_y),
                delta_z = _mm512_sub_pd(rotated_a_z, b_z);
        sum_squared_vec = _mm512_fmadd_pd(delta_x, delta_x, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_y, delta_y, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_z, delta_z, sum_squared_vec);
    }

    *result = SIMSIMD_SQRT((simsimd_distance_t)_mm512_reduce_add_pd(sum_squared_vec) * inv_n);
}

SIMSIMD_PUBLIC void simsimd_umeyama_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                                simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid,
                                                simsimd_f32_t *rotation, simsimd_distance_t *scale,
                                                simsimd_distance_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A
    __m512i const gather_idx = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
    __m512 const zeros_f32 = _mm512_setzero_ps();
    __m512d const zeros_f64 = _mm512_setzero_pd();

    __m512d sum_a_x_vec = zeros_f64, sum_a_y_vec = zeros_f64, sum_a_z_vec = zeros_f64;
    __m512d sum_b_x_vec = zeros_f64, sum_b_y_vec = zeros_f64, sum_b_z_vec = zeros_f64;
    __m512d cov_xx = zeros_f64, cov_xy = zeros_f64, cov_xz = zeros_f64;
    __m512d cov_yx = zeros_f64, cov_yy = zeros_f64, cov_yz = zeros_f64;
    __m512d cov_zx = zeros_f64, cov_zy = zeros_f64, cov_zz = zeros_f64;
    __m512d variance_a_acc = zeros_f64;

    simsimd_size_t i = 0;
    __m512 a_x_vec, a_y_vec, a_z_vec, b_x_vec, b_y_vec, b_z_vec;

    for (; i + 16 <= n; i += 16) {
        _simsimd_deinterleave_f32x16_skylake(a + i * 3, &a_x_vec, &a_y_vec, &a_z_vec);
        _simsimd_deinterleave_f32x16_skylake(b + i * 3, &b_x_vec, &b_y_vec, &b_z_vec);

        __m512d a_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_vec));
        __m512d a_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_vec));
        __m512d a_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_vec));
        __m512d b_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_vec));
        __m512d b_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_vec));
        __m512d b_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_vec));

        sum_a_x_vec = _mm512_add_pd(sum_a_x_vec, a_x_lo), sum_a_y_vec = _mm512_add_pd(sum_a_y_vec, a_y_lo);
        sum_a_z_vec = _mm512_add_pd(sum_a_z_vec, a_z_lo);
        sum_b_x_vec = _mm512_add_pd(sum_b_x_vec, b_x_lo), sum_b_y_vec = _mm512_add_pd(sum_b_y_vec, b_y_lo);
        sum_b_z_vec = _mm512_add_pd(sum_b_z_vec, b_z_lo);

        cov_xx = _mm512_fmadd_pd(a_x_lo, b_x_lo, cov_xx), cov_xy = _mm512_fmadd_pd(a_x_lo, b_y_lo, cov_xy);
        cov_xz = _mm512_fmadd_pd(a_x_lo, b_z_lo, cov_xz);
        cov_yx = _mm512_fmadd_pd(a_y_lo, b_x_lo, cov_yx), cov_yy = _mm512_fmadd_pd(a_y_lo, b_y_lo, cov_yy);
        cov_yz = _mm512_fmadd_pd(a_y_lo, b_z_lo, cov_yz);
        cov_zx = _mm512_fmadd_pd(a_z_lo, b_x_lo, cov_zx), cov_zy = _mm512_fmadd_pd(a_z_lo, b_y_lo, cov_zy);
        cov_zz = _mm512_fmadd_pd(a_z_lo, b_z_lo, cov_zz);
        variance_a_acc = _mm512_fmadd_pd(a_x_lo, a_x_lo, variance_a_acc);
        variance_a_acc = _mm512_fmadd_pd(a_y_lo, a_y_lo, variance_a_acc);
        variance_a_acc = _mm512_fmadd_pd(a_z_lo, a_z_lo, variance_a_acc);

        __m512d a_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_vec, 1));
        __m512d a_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_vec, 1));
        __m512d a_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_vec, 1));
        __m512d b_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_vec, 1));
        __m512d b_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_vec, 1));
        __m512d b_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_vec, 1));

        sum_a_x_vec = _mm512_add_pd(sum_a_x_vec, a_x_hi), sum_a_y_vec = _mm512_add_pd(sum_a_y_vec, a_y_hi);
        sum_a_z_vec = _mm512_add_pd(sum_a_z_vec, a_z_hi);
        sum_b_x_vec = _mm512_add_pd(sum_b_x_vec, b_x_hi), sum_b_y_vec = _mm512_add_pd(sum_b_y_vec, b_y_hi);
        sum_b_z_vec = _mm512_add_pd(sum_b_z_vec, b_z_hi);

        cov_xx = _mm512_fmadd_pd(a_x_hi, b_x_hi, cov_xx), cov_xy = _mm512_fmadd_pd(a_x_hi, b_y_hi, cov_xy);
        cov_xz = _mm512_fmadd_pd(a_x_hi, b_z_hi, cov_xz);
        cov_yx = _mm512_fmadd_pd(a_y_hi, b_x_hi, cov_yx), cov_yy = _mm512_fmadd_pd(a_y_hi, b_y_hi, cov_yy);
        cov_yz = _mm512_fmadd_pd(a_y_hi, b_z_hi, cov_yz);
        cov_zx = _mm512_fmadd_pd(a_z_hi, b_x_hi, cov_zx), cov_zy = _mm512_fmadd_pd(a_z_hi, b_y_hi, cov_zy);
        cov_zz = _mm512_fmadd_pd(a_z_hi, b_z_hi, cov_zz);
        variance_a_acc = _mm512_fmadd_pd(a_x_hi, a_x_hi, variance_a_acc);
        variance_a_acc = _mm512_fmadd_pd(a_y_hi, a_y_hi, variance_a_acc);
        variance_a_acc = _mm512_fmadd_pd(a_z_hi, a_z_hi, variance_a_acc);
    }

    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail);
        simsimd_f32_t const *a_tail = a + i * 3;
        simsimd_f32_t const *b_tail = b + i * 3;
        a_x_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 0, 4);
        a_y_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 1, 4);
        a_z_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 2, 4);
        b_x_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 0, 4);
        b_y_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 1, 4);
        b_z_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 2, 4);

        __m512d a_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_vec));
        __m512d a_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_vec));
        __m512d a_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_vec));
        __m512d b_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_vec));
        __m512d b_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_vec));
        __m512d b_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_vec));

        sum_a_x_vec = _mm512_add_pd(sum_a_x_vec, a_x_lo), sum_a_y_vec = _mm512_add_pd(sum_a_y_vec, a_y_lo);
        sum_a_z_vec = _mm512_add_pd(sum_a_z_vec, a_z_lo);
        sum_b_x_vec = _mm512_add_pd(sum_b_x_vec, b_x_lo), sum_b_y_vec = _mm512_add_pd(sum_b_y_vec, b_y_lo);
        sum_b_z_vec = _mm512_add_pd(sum_b_z_vec, b_z_lo);

        cov_xx = _mm512_fmadd_pd(a_x_lo, b_x_lo, cov_xx), cov_xy = _mm512_fmadd_pd(a_x_lo, b_y_lo, cov_xy);
        cov_xz = _mm512_fmadd_pd(a_x_lo, b_z_lo, cov_xz);
        cov_yx = _mm512_fmadd_pd(a_y_lo, b_x_lo, cov_yx), cov_yy = _mm512_fmadd_pd(a_y_lo, b_y_lo, cov_yy);
        cov_yz = _mm512_fmadd_pd(a_y_lo, b_z_lo, cov_yz);
        cov_zx = _mm512_fmadd_pd(a_z_lo, b_x_lo, cov_zx), cov_zy = _mm512_fmadd_pd(a_z_lo, b_y_lo, cov_zy);
        cov_zz = _mm512_fmadd_pd(a_z_lo, b_z_lo, cov_zz);
        variance_a_acc = _mm512_fmadd_pd(a_x_lo, a_x_lo, variance_a_acc);
        variance_a_acc = _mm512_fmadd_pd(a_y_lo, a_y_lo, variance_a_acc);
        variance_a_acc = _mm512_fmadd_pd(a_z_lo, a_z_lo, variance_a_acc);

        __m512d a_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_vec, 1));
        __m512d a_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_vec, 1));
        __m512d a_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_vec, 1));
        __m512d b_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_vec, 1));
        __m512d b_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_vec, 1));
        __m512d b_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_vec, 1));

        sum_a_x_vec = _mm512_add_pd(sum_a_x_vec, a_x_hi), sum_a_y_vec = _mm512_add_pd(sum_a_y_vec, a_y_hi);
        sum_a_z_vec = _mm512_add_pd(sum_a_z_vec, a_z_hi);
        sum_b_x_vec = _mm512_add_pd(sum_b_x_vec, b_x_hi), sum_b_y_vec = _mm512_add_pd(sum_b_y_vec, b_y_hi);
        sum_b_z_vec = _mm512_add_pd(sum_b_z_vec, b_z_hi);

        cov_xx = _mm512_fmadd_pd(a_x_hi, b_x_hi, cov_xx), cov_xy = _mm512_fmadd_pd(a_x_hi, b_y_hi, cov_xy);
        cov_xz = _mm512_fmadd_pd(a_x_hi, b_z_hi, cov_xz);
        cov_yx = _mm512_fmadd_pd(a_y_hi, b_x_hi, cov_yx), cov_yy = _mm512_fmadd_pd(a_y_hi, b_y_hi, cov_yy);
        cov_yz = _mm512_fmadd_pd(a_y_hi, b_z_hi, cov_yz);
        cov_zx = _mm512_fmadd_pd(a_z_hi, b_x_hi, cov_zx), cov_zy = _mm512_fmadd_pd(a_z_hi, b_y_hi, cov_zy);
        cov_zz = _mm512_fmadd_pd(a_z_hi, b_z_hi, cov_zz);
        variance_a_acc = _mm512_fmadd_pd(a_x_hi, a_x_hi, variance_a_acc);
        variance_a_acc = _mm512_fmadd_pd(a_y_hi, a_y_hi, variance_a_acc);
        variance_a_acc = _mm512_fmadd_pd(a_z_hi, a_z_hi, variance_a_acc);
    }

    // Reduce centroids
    simsimd_f64_t inv_n = 1.0 / (simsimd_f64_t)n;
    simsimd_f64_t sum_a_x = _mm512_reduce_add_pd(sum_a_x_vec), sum_a_y = _mm512_reduce_add_pd(sum_a_y_vec);
    simsimd_f64_t sum_a_z = _mm512_reduce_add_pd(sum_a_z_vec);
    simsimd_f64_t sum_b_x = _mm512_reduce_add_pd(sum_b_x_vec), sum_b_y = _mm512_reduce_add_pd(sum_b_y_vec);
    simsimd_f64_t sum_b_z = _mm512_reduce_add_pd(sum_b_z_vec);

    simsimd_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    simsimd_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid)
        a_centroid[0] = (simsimd_f32_t)centroid_a_x, a_centroid[1] = (simsimd_f32_t)centroid_a_y,
        a_centroid[2] = (simsimd_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (simsimd_f32_t)centroid_b_x, b_centroid[1] = (simsimd_f32_t)centroid_b_y,
        b_centroid[2] = (simsimd_f32_t)centroid_b_z;

    // Compute centered covariance and variance
    simsimd_f64_t variance_a_sum = _mm512_reduce_add_pd(variance_a_acc);
    simsimd_f64_t variance_a = variance_a_sum * inv_n - (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                         centroid_a_z * centroid_a_z);

    simsimd_f32_t cross_covariance[9];
    cross_covariance[0] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_xx) - sum_a_x * sum_b_x * inv_n);
    cross_covariance[1] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_xy) - sum_a_x * sum_b_y * inv_n);
    cross_covariance[2] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_xz) - sum_a_x * sum_b_z * inv_n);
    cross_covariance[3] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_yx) - sum_a_y * sum_b_x * inv_n);
    cross_covariance[4] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_yy) - sum_a_y * sum_b_y * inv_n);
    cross_covariance[5] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_yz) - sum_a_y * sum_b_z * inv_n);
    cross_covariance[6] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_zx) - sum_a_z * sum_b_x * inv_n);
    cross_covariance[7] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_zy) - sum_a_z * sum_b_y * inv_n);
    cross_covariance[8] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_zz) - sum_a_z * sum_b_z * inv_n);

    // SVD
    simsimd_f32_t svd_u[9], svd_s[9], svd_v[9];
    _simsimd_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * U^T
    simsimd_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Scale factor: c = trace(D*S) / (n * variance_a)
    simsimd_f32_t det = _simsimd_det3x3_f32(r);
    simsimd_f32_t d3 = det < 0 ? -1.0f : 1.0f;
    simsimd_f32_t trace_ds = svd_s[0] + svd_s[4] + d3 * svd_s[8];
    simsimd_f64_t c = (simsimd_f64_t)trace_ds / (n * variance_a);
    if (scale) *scale = c;

    // Handle reflection
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }

    // Compute RMSD with scaling
    __m512d sum_squared_vec = zeros_f64;
    __m512d c_vec = _mm512_set1_pd(c);
    __m512d r0_vec = _mm512_set1_pd(r[0]), r1_vec = _mm512_set1_pd(r[1]), r2_vec = _mm512_set1_pd(r[2]);
    __m512d r3_vec = _mm512_set1_pd(r[3]), r4_vec = _mm512_set1_pd(r[4]), r5_vec = _mm512_set1_pd(r[5]);
    __m512d r6_vec = _mm512_set1_pd(r[6]), r7_vec = _mm512_set1_pd(r[7]), r8_vec = _mm512_set1_pd(r[8]);
    __m512d centroid_a_x_vec = _mm512_set1_pd(centroid_a_x), centroid_a_y_vec = _mm512_set1_pd(centroid_a_y),
            centroid_a_z_vec = _mm512_set1_pd(centroid_a_z);
    __m512d centroid_b_x_vec = _mm512_set1_pd(centroid_b_x), centroid_b_y_vec = _mm512_set1_pd(centroid_b_y),
            centroid_b_z_vec = _mm512_set1_pd(centroid_b_z);

    for (i = 0; i + 16 <= n; i += 16) {
        _simsimd_deinterleave_f32x16_skylake(a + i * 3, &a_x_vec, &a_y_vec, &a_z_vec);
        _simsimd_deinterleave_f32x16_skylake(b + i * 3, &b_x_vec, &b_y_vec, &b_z_vec);

        __m512d a_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_vec));
        __m512d a_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_vec));
        __m512d a_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_vec));
        __m512d b_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_vec));
        __m512d b_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_vec));
        __m512d b_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_vec));

        a_x_lo = _mm512_sub_pd(a_x_lo, centroid_a_x_vec), a_y_lo = _mm512_sub_pd(a_y_lo, centroid_a_y_vec);
        a_z_lo = _mm512_sub_pd(a_z_lo, centroid_a_z_vec);
        b_x_lo = _mm512_sub_pd(b_x_lo, centroid_b_x_vec), b_y_lo = _mm512_sub_pd(b_y_lo, centroid_b_y_vec);
        b_z_lo = _mm512_sub_pd(b_z_lo, centroid_b_z_vec);

        __m512d rotated_a_x = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r0_vec, a_x_lo, _mm512_fmadd_pd(r1_vec, a_y_lo, _mm512_mul_pd(r2_vec, a_z_lo))));
        __m512d rotated_a_y = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r3_vec, a_x_lo, _mm512_fmadd_pd(r4_vec, a_y_lo, _mm512_mul_pd(r5_vec, a_z_lo))));
        __m512d rotated_a_z = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r6_vec, a_x_lo, _mm512_fmadd_pd(r7_vec, a_y_lo, _mm512_mul_pd(r8_vec, a_z_lo))));

        __m512d delta_x = _mm512_sub_pd(rotated_a_x, b_x_lo), delta_y = _mm512_sub_pd(rotated_a_y, b_y_lo),
                delta_z = _mm512_sub_pd(rotated_a_z, b_z_lo);
        sum_squared_vec = _mm512_fmadd_pd(delta_x, delta_x, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_y, delta_y, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_z, delta_z, sum_squared_vec);

        __m512d a_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_vec, 1));
        __m512d a_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_vec, 1));
        __m512d a_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_vec, 1));
        __m512d b_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_vec, 1));
        __m512d b_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_vec, 1));
        __m512d b_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_vec, 1));

        a_x_hi = _mm512_sub_pd(a_x_hi, centroid_a_x_vec), a_y_hi = _mm512_sub_pd(a_y_hi, centroid_a_y_vec);
        a_z_hi = _mm512_sub_pd(a_z_hi, centroid_a_z_vec);
        b_x_hi = _mm512_sub_pd(b_x_hi, centroid_b_x_vec), b_y_hi = _mm512_sub_pd(b_y_hi, centroid_b_y_vec);
        b_z_hi = _mm512_sub_pd(b_z_hi, centroid_b_z_vec);

        rotated_a_x = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r0_vec, a_x_hi, _mm512_fmadd_pd(r1_vec, a_y_hi, _mm512_mul_pd(r2_vec, a_z_hi))));
        rotated_a_y = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r3_vec, a_x_hi, _mm512_fmadd_pd(r4_vec, a_y_hi, _mm512_mul_pd(r5_vec, a_z_hi))));
        rotated_a_z = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r6_vec, a_x_hi, _mm512_fmadd_pd(r7_vec, a_y_hi, _mm512_mul_pd(r8_vec, a_z_hi))));

        delta_x = _mm512_sub_pd(rotated_a_x, b_x_hi), delta_y = _mm512_sub_pd(rotated_a_y, b_y_hi),
        delta_z = _mm512_sub_pd(rotated_a_z, b_z_hi);
        sum_squared_vec = _mm512_fmadd_pd(delta_x, delta_x, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_y, delta_y, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_z, delta_z, sum_squared_vec);
    }

    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail);
        simsimd_f32_t const *a_tail = a + i * 3;
        simsimd_f32_t const *b_tail = b + i * 3;
        a_x_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 0, 4);
        a_y_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 1, 4);
        a_z_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 2, 4);
        b_x_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 0, 4);
        b_y_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 1, 4);
        b_z_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 2, 4);

        // Mask for low 8 lanes: min(tail, 8) valid bits
        __mmask8 lo_mask = (__mmask8)_bzhi_u32(0xFF, tail < 8 ? tail : 8);

        __m512d a_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_vec));
        __m512d a_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_vec));
        __m512d a_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_vec));
        __m512d b_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_vec));
        __m512d b_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_vec));
        __m512d b_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_vec));

        a_x_lo = _mm512_sub_pd(a_x_lo, centroid_a_x_vec), a_y_lo = _mm512_sub_pd(a_y_lo, centroid_a_y_vec);
        a_z_lo = _mm512_sub_pd(a_z_lo, centroid_a_z_vec);
        b_x_lo = _mm512_sub_pd(b_x_lo, centroid_b_x_vec), b_y_lo = _mm512_sub_pd(b_y_lo, centroid_b_y_vec);
        b_z_lo = _mm512_sub_pd(b_z_lo, centroid_b_z_vec);

        __m512d rotated_a_x = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r0_vec, a_x_lo, _mm512_fmadd_pd(r1_vec, a_y_lo, _mm512_mul_pd(r2_vec, a_z_lo))));
        __m512d rotated_a_y = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r3_vec, a_x_lo, _mm512_fmadd_pd(r4_vec, a_y_lo, _mm512_mul_pd(r5_vec, a_z_lo))));
        __m512d rotated_a_z = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r6_vec, a_x_lo, _mm512_fmadd_pd(r7_vec, a_y_lo, _mm512_mul_pd(r8_vec, a_z_lo))));

        __m512d delta_x = _mm512_sub_pd(rotated_a_x, b_x_lo), delta_y = _mm512_sub_pd(rotated_a_y, b_y_lo),
                delta_z = _mm512_sub_pd(rotated_a_z, b_z_lo);
        // Use masked accumulation to avoid counting invalid tail lanes
        sum_squared_vec = _mm512_mask3_fmadd_pd(delta_x, delta_x, sum_squared_vec, lo_mask);
        sum_squared_vec = _mm512_mask3_fmadd_pd(delta_y, delta_y, sum_squared_vec, lo_mask);
        sum_squared_vec = _mm512_mask3_fmadd_pd(delta_z, delta_z, sum_squared_vec, lo_mask);

        // Only process high 8 if there are more than 8 tail elements
        if (tail > 8) {
            __mmask8 hi_mask = (__mmask8)_bzhi_u32(0xFF, tail - 8);

            __m512d a_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_vec, 1));
            __m512d a_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_vec, 1));
            __m512d a_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_vec, 1));
            __m512d b_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_vec, 1));
            __m512d b_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_vec, 1));
            __m512d b_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_vec, 1));

            a_x_hi = _mm512_sub_pd(a_x_hi, centroid_a_x_vec), a_y_hi = _mm512_sub_pd(a_y_hi, centroid_a_y_vec);
            a_z_hi = _mm512_sub_pd(a_z_hi, centroid_a_z_vec);
            b_x_hi = _mm512_sub_pd(b_x_hi, centroid_b_x_vec), b_y_hi = _mm512_sub_pd(b_y_hi, centroid_b_y_vec);
            b_z_hi = _mm512_sub_pd(b_z_hi, centroid_b_z_vec);

            rotated_a_x = _mm512_mul_pd(
                c_vec, _mm512_fmadd_pd(r0_vec, a_x_hi, _mm512_fmadd_pd(r1_vec, a_y_hi, _mm512_mul_pd(r2_vec, a_z_hi))));
            rotated_a_y = _mm512_mul_pd(
                c_vec, _mm512_fmadd_pd(r3_vec, a_x_hi, _mm512_fmadd_pd(r4_vec, a_y_hi, _mm512_mul_pd(r5_vec, a_z_hi))));
            rotated_a_z = _mm512_mul_pd(
                c_vec, _mm512_fmadd_pd(r6_vec, a_x_hi, _mm512_fmadd_pd(r7_vec, a_y_hi, _mm512_mul_pd(r8_vec, a_z_hi))));

            delta_x = _mm512_sub_pd(rotated_a_x, b_x_hi), delta_y = _mm512_sub_pd(rotated_a_y, b_y_hi),
            delta_z = _mm512_sub_pd(rotated_a_z, b_z_hi);
            sum_squared_vec = _mm512_mask3_fmadd_pd(delta_x, delta_x, sum_squared_vec, hi_mask);
            sum_squared_vec = _mm512_mask3_fmadd_pd(delta_y, delta_y, sum_squared_vec, hi_mask);
            sum_squared_vec = _mm512_mask3_fmadd_pd(delta_z, delta_z, sum_squared_vec, hi_mask);
        }
    }

    *result = SIMSIMD_SQRT((simsimd_distance_t)_mm512_reduce_add_pd(sum_squared_vec) * inv_n);
}

SIMSIMD_PUBLIC void simsimd_umeyama_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                                simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid,
                                                simsimd_f64_t *rotation, simsimd_distance_t *scale,
                                                simsimd_distance_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A
    __m512i const gather_idx = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 18, 21);
    __m512d const zeros = _mm512_setzero_pd();

    __m512d sum_a_x_vec = zeros, sum_a_y_vec = zeros, sum_a_z_vec = zeros;
    __m512d sum_b_x_vec = zeros, sum_b_y_vec = zeros, sum_b_z_vec = zeros;
    __m512d cov_xx = zeros, cov_xy = zeros, cov_xz = zeros;
    __m512d cov_yx = zeros, cov_yy = zeros, cov_yz = zeros;
    __m512d cov_zx = zeros, cov_zy = zeros, cov_zz = zeros;
    __m512d variance_a_acc = zeros;

    simsimd_size_t i = 0;
    __m512d a_x, a_y, a_z, b_x, b_y, b_z;

    for (; i + 8 <= n; i += 8) {
        _simsimd_deinterleave_f64x8_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x8_skylake(b + i * 3, &b_x, &b_y, &b_z);

        sum_a_x_vec = _mm512_add_pd(sum_a_x_vec, a_x), sum_a_y_vec = _mm512_add_pd(sum_a_y_vec, a_y);
        sum_a_z_vec = _mm512_add_pd(sum_a_z_vec, a_z);
        sum_b_x_vec = _mm512_add_pd(sum_b_x_vec, b_x), sum_b_y_vec = _mm512_add_pd(sum_b_y_vec, b_y);
        sum_b_z_vec = _mm512_add_pd(sum_b_z_vec, b_z);

        cov_xx = _mm512_fmadd_pd(a_x, b_x, cov_xx), cov_xy = _mm512_fmadd_pd(a_x, b_y, cov_xy);
        cov_xz = _mm512_fmadd_pd(a_x, b_z, cov_xz);
        cov_yx = _mm512_fmadd_pd(a_y, b_x, cov_yx), cov_yy = _mm512_fmadd_pd(a_y, b_y, cov_yy);
        cov_yz = _mm512_fmadd_pd(a_y, b_z, cov_yz);
        cov_zx = _mm512_fmadd_pd(a_z, b_x, cov_zx), cov_zy = _mm512_fmadd_pd(a_z, b_y, cov_zy);
        cov_zz = _mm512_fmadd_pd(a_z, b_z, cov_zz);
        variance_a_acc = _mm512_fmadd_pd(a_x, a_x, variance_a_acc);
        variance_a_acc = _mm512_fmadd_pd(a_y, a_y, variance_a_acc);
        variance_a_acc = _mm512_fmadd_pd(a_z, a_z, variance_a_acc);
    }

    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        simsimd_f64_t const *a_tail = a + i * 3;
        simsimd_f64_t const *b_tail = b + i * 3;

        a_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 0, 8);
        a_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 1, 8);
        a_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 2, 8);
        b_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 0, 8);
        b_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 1, 8);
        b_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 2, 8);

        sum_a_x_vec = _mm512_add_pd(sum_a_x_vec, a_x), sum_a_y_vec = _mm512_add_pd(sum_a_y_vec, a_y);
        sum_a_z_vec = _mm512_add_pd(sum_a_z_vec, a_z);
        sum_b_x_vec = _mm512_add_pd(sum_b_x_vec, b_x), sum_b_y_vec = _mm512_add_pd(sum_b_y_vec, b_y);
        sum_b_z_vec = _mm512_add_pd(sum_b_z_vec, b_z);

        cov_xx = _mm512_fmadd_pd(a_x, b_x, cov_xx), cov_xy = _mm512_fmadd_pd(a_x, b_y, cov_xy);
        cov_xz = _mm512_fmadd_pd(a_x, b_z, cov_xz);
        cov_yx = _mm512_fmadd_pd(a_y, b_x, cov_yx), cov_yy = _mm512_fmadd_pd(a_y, b_y, cov_yy);
        cov_yz = _mm512_fmadd_pd(a_y, b_z, cov_yz);
        cov_zx = _mm512_fmadd_pd(a_z, b_x, cov_zx), cov_zy = _mm512_fmadd_pd(a_z, b_y, cov_zy);
        cov_zz = _mm512_fmadd_pd(a_z, b_z, cov_zz);
        variance_a_acc = _mm512_fmadd_pd(a_x, a_x, variance_a_acc);
        variance_a_acc = _mm512_fmadd_pd(a_y, a_y, variance_a_acc);
        variance_a_acc = _mm512_fmadd_pd(a_z, a_z, variance_a_acc);
    }

    // Reduce centroids
    simsimd_f64_t inv_n = 1.0 / (simsimd_f64_t)n;
    simsimd_f64_t sum_a_x = _mm512_reduce_add_pd(sum_a_x_vec), sum_a_y = _mm512_reduce_add_pd(sum_a_y_vec);
    simsimd_f64_t sum_a_z = _mm512_reduce_add_pd(sum_a_z_vec);
    simsimd_f64_t sum_b_x = _mm512_reduce_add_pd(sum_b_x_vec), sum_b_y = _mm512_reduce_add_pd(sum_b_y_vec);
    simsimd_f64_t sum_b_z = _mm512_reduce_add_pd(sum_b_z_vec);

    simsimd_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    simsimd_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance and variance
    simsimd_f64_t variance_a_sum = _mm512_reduce_add_pd(variance_a_acc);
    simsimd_f64_t variance_a = variance_a_sum * inv_n - (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                         centroid_a_z * centroid_a_z);

    simsimd_f32_t cross_covariance[9];
    cross_covariance[0] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_xx) - sum_a_x * sum_b_x * inv_n);
    cross_covariance[1] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_xy) - sum_a_x * sum_b_y * inv_n);
    cross_covariance[2] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_xz) - sum_a_x * sum_b_z * inv_n);
    cross_covariance[3] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_yx) - sum_a_y * sum_b_x * inv_n);
    cross_covariance[4] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_yy) - sum_a_y * sum_b_y * inv_n);
    cross_covariance[5] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_yz) - sum_a_y * sum_b_z * inv_n);
    cross_covariance[6] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_zx) - sum_a_z * sum_b_x * inv_n);
    cross_covariance[7] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_zy) - sum_a_z * sum_b_y * inv_n);
    cross_covariance[8] = (simsimd_f32_t)(_mm512_reduce_add_pd(cov_zz) - sum_a_z * sum_b_z * inv_n);

    // SVD
    simsimd_f32_t svd_u[9], svd_s[9], svd_v[9];
    _simsimd_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * U^T
    simsimd_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Scale factor: c = trace(D*S) / (n * variance_a)
    simsimd_f32_t det = _simsimd_det3x3_f32(r);
    simsimd_f32_t d3 = det < 0 ? -1.0f : 1.0f;
    simsimd_f32_t trace_ds = svd_s[0] + svd_s[4] + d3 * svd_s[8];
    simsimd_f64_t c = (simsimd_f64_t)trace_ds / (n * variance_a);
    if (scale) *scale = c;

    // Handle reflection
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = (simsimd_f64_t)r[j];
    }

    // Compute RMSD with scaling
    __m512d sum_squared_vec = zeros;
    __m512d c_vec = _mm512_set1_pd(c);
    __m512d r0_vec = _mm512_set1_pd(r[0]), r1_vec = _mm512_set1_pd(r[1]), r2_vec = _mm512_set1_pd(r[2]);
    __m512d r3_vec = _mm512_set1_pd(r[3]), r4_vec = _mm512_set1_pd(r[4]), r5_vec = _mm512_set1_pd(r[5]);
    __m512d r6_vec = _mm512_set1_pd(r[6]), r7_vec = _mm512_set1_pd(r[7]), r8_vec = _mm512_set1_pd(r[8]);
    __m512d centroid_a_x_vec = _mm512_set1_pd(centroid_a_x), centroid_a_y_vec = _mm512_set1_pd(centroid_a_y),
            centroid_a_z_vec = _mm512_set1_pd(centroid_a_z);
    __m512d centroid_b_x_vec = _mm512_set1_pd(centroid_b_x), centroid_b_y_vec = _mm512_set1_pd(centroid_b_y),
            centroid_b_z_vec = _mm512_set1_pd(centroid_b_z);

    for (i = 0; i + 8 <= n; i += 8) {
        _simsimd_deinterleave_f64x8_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x8_skylake(b + i * 3, &b_x, &b_y, &b_z);

        a_x = _mm512_sub_pd(a_x, centroid_a_x_vec), a_y = _mm512_sub_pd(a_y, centroid_a_y_vec),
        a_z = _mm512_sub_pd(a_z, centroid_a_z_vec);
        b_x = _mm512_sub_pd(b_x, centroid_b_x_vec), b_y = _mm512_sub_pd(b_y, centroid_b_y_vec),
        b_z = _mm512_sub_pd(b_z, centroid_b_z_vec);

        __m512d rotated_a_x = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r0_vec, a_x, _mm512_fmadd_pd(r1_vec, a_y, _mm512_mul_pd(r2_vec, a_z))));
        __m512d rotated_a_y = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r3_vec, a_x, _mm512_fmadd_pd(r4_vec, a_y, _mm512_mul_pd(r5_vec, a_z))));
        __m512d rotated_a_z = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r6_vec, a_x, _mm512_fmadd_pd(r7_vec, a_y, _mm512_mul_pd(r8_vec, a_z))));

        __m512d delta_x = _mm512_sub_pd(rotated_a_x, b_x), delta_y = _mm512_sub_pd(rotated_a_y, b_y),
                delta_z = _mm512_sub_pd(rotated_a_z, b_z);
        sum_squared_vec = _mm512_fmadd_pd(delta_x, delta_x, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_y, delta_y, sum_squared_vec);
        sum_squared_vec = _mm512_fmadd_pd(delta_z, delta_z, sum_squared_vec);
    }

    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        simsimd_f64_t const *a_tail = a + i * 3;
        simsimd_f64_t const *b_tail = b + i * 3;

        a_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 0, 8);
        a_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 1, 8);
        a_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 2, 8);
        b_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 0, 8);
        b_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 1, 8);
        b_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 2, 8);

        a_x = _mm512_sub_pd(a_x, centroid_a_x_vec), a_y = _mm512_sub_pd(a_y, centroid_a_y_vec),
        a_z = _mm512_sub_pd(a_z, centroid_a_z_vec);
        b_x = _mm512_sub_pd(b_x, centroid_b_x_vec), b_y = _mm512_sub_pd(b_y, centroid_b_y_vec),
        b_z = _mm512_sub_pd(b_z, centroid_b_z_vec);

        __m512d rotated_a_x = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r0_vec, a_x, _mm512_fmadd_pd(r1_vec, a_y, _mm512_mul_pd(r2_vec, a_z))));
        __m512d rotated_a_y = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r3_vec, a_x, _mm512_fmadd_pd(r4_vec, a_y, _mm512_mul_pd(r5_vec, a_z))));
        __m512d rotated_a_z = _mm512_mul_pd(
            c_vec, _mm512_fmadd_pd(r6_vec, a_x, _mm512_fmadd_pd(r7_vec, a_y, _mm512_mul_pd(r8_vec, a_z))));

        __m512d delta_x = _mm512_sub_pd(rotated_a_x, b_x), delta_y = _mm512_sub_pd(rotated_a_y, b_y),
                delta_z = _mm512_sub_pd(rotated_a_z, b_z);
        // Use masked accumulation to avoid counting invalid tail lanes
        sum_squared_vec = _mm512_mask3_fmadd_pd(delta_x, delta_x, sum_squared_vec, mask);
        sum_squared_vec = _mm512_mask3_fmadd_pd(delta_y, delta_y, sum_squared_vec, mask);
        sum_squared_vec = _mm512_mask3_fmadd_pd(delta_z, delta_z, sum_squared_vec, mask);
    }

    *result = SIMSIMD_SQRT((simsimd_distance_t)_mm512_reduce_add_pd(sum_squared_vec) * inv_n);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "fma", "f16c", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,fma,f16c,bmi2"))), apply_to = function)

/*  Internal helper: Deinterleave 24 floats (8 xyz triplets) into separate x, y, z vectors.
 *  Uses AVX2 gather instructions for clean stride-3 access.
 *
 *  Input: 24 contiguous floats [x0,y0,z0, x1,y1,z1, ..., x7,y7,z7]
 *  Output: x[8], y[8], z[8] vectors
 */
SIMSIMD_INTERNAL void _simsimd_deinterleave_f32x8_haswell(simsimd_f32_t const *ptr, __m256 *x_out, __m256 *y_out,
                                                          __m256 *z_out) {
    // Gather indices: 0, 3, 6, 9, 12, 15, 18, 21 (stride 3)
    __m256i idx = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    *x_out = _mm256_i32gather_ps(ptr + 0, idx, 4);
    *y_out = _mm256_i32gather_ps(ptr + 1, idx, 4);
    *z_out = _mm256_i32gather_ps(ptr + 2, idx, 4);
}

/*  Internal helper: Deinterleave 12 f64 values (4 xyz triplets) into separate x, y, z vectors.
 *  Uses scalar extraction for simplicity as AVX2 lacks efficient stride-3 gather for f64.
 *
 *  Input: 12 contiguous f64 [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
 *  Output: x[4], y[4], z[4] vectors
 */
SIMSIMD_INTERNAL void _simsimd_deinterleave_f64x4_haswell(simsimd_f64_t const *ptr, __m256d *x_out, __m256d *y_out,
                                                          __m256d *z_out) {
    simsimd_f64_t x0 = ptr[0], x1 = ptr[3], x2 = ptr[6], x3 = ptr[9];
    simsimd_f64_t y0 = ptr[1], y1 = ptr[4], y2 = ptr[7], y3 = ptr[10];
    simsimd_f64_t z0 = ptr[2], z1 = ptr[5], z2 = ptr[8], z3 = ptr[11];

    *x_out = _mm256_setr_pd(x0, x1, x2, x3);
    *y_out = _mm256_setr_pd(y0, y1, y2, y3);
    *z_out = _mm256_setr_pd(z0, z1, z2, z3);
}

/*  Horizontal reduction sum for __m256 (8 floats) */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_reduce_add_f32x8_haswell(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

/*  Horizontal reduction sum for __m256d (4 doubles) */
SIMSIMD_INTERNAL simsimd_f64_t _simsimd_reduce_add_f64x4_haswell(__m256d v) {
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d sum = _mm_add_pd(lo, hi);
    sum = _mm_hadd_pd(sum, sum);
    return _mm_cvtsd_f64(sum);
}

SIMSIMD_PUBLIC void simsimd_rmsd_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid,
                                             simsimd_f32_t *rotation, simsimd_distance_t *scale,
                                             simsimd_distance_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1;
        rotation[1] = 0;
        rotation[2] = 0;
        rotation[3] = 0;
        rotation[4] = 1;
        rotation[5] = 0;
        rotation[6] = 0;
        rotation[7] = 0;
        rotation[8] = 1;
    }
    if (scale) *scale = 1.0;
    // Optimized fused single-pass implementation using AVX2.
    // Computes centroids and squared differences in one pass.
    __m256i const gather_idx = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    __m256 const zeros = _mm256_setzero_ps();

    // Accumulators for centroids and squared differences
    __m256 sum_a_x = zeros, sum_a_y = zeros, sum_a_z = zeros;
    __m256 sum_b_x = zeros, sum_b_y = zeros, sum_b_z = zeros;
    __m256 sum_squared_x = zeros, sum_squared_y = zeros, sum_squared_z = zeros;

    __m256 a_x, a_y, a_z, b_x, b_y, b_z;
    simsimd_size_t i = 0;

    // Main loop with 2x unrolling
    for (; i + 16 <= n; i += 16) {
        // Iteration 0
        _simsimd_deinterleave_f32x8_haswell(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f32x8_haswell(b + i * 3, &b_x, &b_y, &b_z);

        sum_a_x = _mm256_add_ps(sum_a_x, a_x);
        sum_a_y = _mm256_add_ps(sum_a_y, a_y);
        sum_a_z = _mm256_add_ps(sum_a_z, a_z);
        sum_b_x = _mm256_add_ps(sum_b_x, b_x);
        sum_b_y = _mm256_add_ps(sum_b_y, b_y);
        sum_b_z = _mm256_add_ps(sum_b_z, b_z);

        __m256 delta_x = _mm256_sub_ps(a_x, b_x);
        __m256 delta_y = _mm256_sub_ps(a_y, b_y);
        __m256 delta_z = _mm256_sub_ps(a_z, b_z);

        sum_squared_x = _mm256_fmadd_ps(delta_x, delta_x, sum_squared_x);
        sum_squared_y = _mm256_fmadd_ps(delta_y, delta_y, sum_squared_y);
        sum_squared_z = _mm256_fmadd_ps(delta_z, delta_z, sum_squared_z);

        // Iteration 1
        __m256 a_x1, a_y1, a_z1, b_x1, b_y1, b_z1;
        _simsimd_deinterleave_f32x8_haswell(a + (i + 8) * 3, &a_x1, &a_y1, &a_z1);
        _simsimd_deinterleave_f32x8_haswell(b + (i + 8) * 3, &b_x1, &b_y1, &b_z1);

        sum_a_x = _mm256_add_ps(sum_a_x, a_x1);
        sum_a_y = _mm256_add_ps(sum_a_y, a_y1);
        sum_a_z = _mm256_add_ps(sum_a_z, a_z1);
        sum_b_x = _mm256_add_ps(sum_b_x, b_x1);
        sum_b_y = _mm256_add_ps(sum_b_y, b_y1);
        sum_b_z = _mm256_add_ps(sum_b_z, b_z1);

        __m256 delta_x1 = _mm256_sub_ps(a_x1, b_x1);
        __m256 delta_y1 = _mm256_sub_ps(a_y1, b_y1);
        __m256 delta_z1 = _mm256_sub_ps(a_z1, b_z1);

        sum_squared_x = _mm256_fmadd_ps(delta_x1, delta_x1, sum_squared_x);
        sum_squared_y = _mm256_fmadd_ps(delta_y1, delta_y1, sum_squared_y);
        sum_squared_z = _mm256_fmadd_ps(delta_z1, delta_z1, sum_squared_z);
    }

    // Handle 8-point remainder
    for (; i + 8 <= n; i += 8) {
        _simsimd_deinterleave_f32x8_haswell(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f32x8_haswell(b + i * 3, &b_x, &b_y, &b_z);

        sum_a_x = _mm256_add_ps(sum_a_x, a_x);
        sum_a_y = _mm256_add_ps(sum_a_y, a_y);
        sum_a_z = _mm256_add_ps(sum_a_z, a_z);
        sum_b_x = _mm256_add_ps(sum_b_x, b_x);
        sum_b_y = _mm256_add_ps(sum_b_y, b_y);
        sum_b_z = _mm256_add_ps(sum_b_z, b_z);

        __m256 delta_x = _mm256_sub_ps(a_x, b_x);
        __m256 delta_y = _mm256_sub_ps(a_y, b_y);
        __m256 delta_z = _mm256_sub_ps(a_z, b_z);

        sum_squared_x = _mm256_fmadd_ps(delta_x, delta_x, sum_squared_x);
        sum_squared_y = _mm256_fmadd_ps(delta_y, delta_y, sum_squared_y);
        sum_squared_z = _mm256_fmadd_ps(delta_z, delta_z, sum_squared_z);
    }

    // Reduce vectors to scalars
    simsimd_f32_t total_ax = _simsimd_reduce_add_f32x8_haswell(sum_a_x);
    simsimd_f32_t total_ay = _simsimd_reduce_add_f32x8_haswell(sum_a_y);
    simsimd_f32_t total_az = _simsimd_reduce_add_f32x8_haswell(sum_a_z);
    simsimd_f32_t total_bx = _simsimd_reduce_add_f32x8_haswell(sum_b_x);
    simsimd_f32_t total_by = _simsimd_reduce_add_f32x8_haswell(sum_b_y);
    simsimd_f32_t total_bz = _simsimd_reduce_add_f32x8_haswell(sum_b_z);
    simsimd_f32_t total_sq_x = _simsimd_reduce_add_f32x8_haswell(sum_squared_x);
    simsimd_f32_t total_sq_y = _simsimd_reduce_add_f32x8_haswell(sum_squared_y);
    simsimd_f32_t total_sq_z = _simsimd_reduce_add_f32x8_haswell(sum_squared_z);

    // Scalar tail
    for (; i < n; ++i) {
        simsimd_f32_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        simsimd_f32_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        total_ax += ax;
        total_ay += ay;
        total_az += az;
        total_bx += bx;
        total_by += by;
        total_bz += bz;
        simsimd_f32_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        total_sq_x += delta_x * delta_x;
        total_sq_y += delta_y * delta_y;
        total_sq_z += delta_z * delta_z;
    }

    // Compute centroids
    simsimd_f32_t inv_n = 1.0f / (simsimd_f32_t)n;
    simsimd_f32_t centroid_a_x = total_ax * inv_n;
    simsimd_f32_t centroid_a_y = total_ay * inv_n;
    simsimd_f32_t centroid_a_z = total_az * inv_n;
    simsimd_f32_t centroid_b_x = total_bx * inv_n;
    simsimd_f32_t centroid_b_y = total_by * inv_n;
    simsimd_f32_t centroid_b_z = total_bz * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Compute RMSD
    simsimd_f32_t mean_diff_x = centroid_a_x - centroid_b_x;
    simsimd_f32_t mean_diff_y = centroid_a_y - centroid_b_y;
    simsimd_f32_t mean_diff_z = centroid_a_z - centroid_b_z;
    simsimd_f32_t sum_squared = total_sq_x + total_sq_y + total_sq_z;
    simsimd_f32_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = SIMSIMD_SQRT((simsimd_distance_t)(sum_squared * inv_n - mean_diff_sq));
}

SIMSIMD_PUBLIC void simsimd_rmsd_f64_haswell(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                             simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid,
                                             simsimd_f64_t *rotation, simsimd_distance_t *scale,
                                             simsimd_distance_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1;
        rotation[1] = 0;
        rotation[2] = 0;
        rotation[3] = 0;
        rotation[4] = 1;
        rotation[5] = 0;
        rotation[6] = 0;
        rotation[7] = 0;
        rotation[8] = 1;
    }
    if (scale) *scale = 1.0;
    __m256d const zeros = _mm256_setzero_pd();

    // Accumulators for centroids and squared differences
    __m256d sum_a_x = zeros, sum_a_y = zeros, sum_a_z = zeros;
    __m256d sum_b_x = zeros, sum_b_y = zeros, sum_b_z = zeros;
    __m256d sum_squared_x = zeros, sum_squared_y = zeros, sum_squared_z = zeros;

    __m256d a_x, a_y, a_z, b_x, b_y, b_z;
    simsimd_size_t i = 0;

    // Main loop with 2x unrolling
    for (; i + 8 <= n; i += 8) {
        // Iteration 0
        _simsimd_deinterleave_f64x4_haswell(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x4_haswell(b + i * 3, &b_x, &b_y, &b_z);

        sum_a_x = _mm256_add_pd(sum_a_x, a_x);
        sum_a_y = _mm256_add_pd(sum_a_y, a_y);
        sum_a_z = _mm256_add_pd(sum_a_z, a_z);
        sum_b_x = _mm256_add_pd(sum_b_x, b_x);
        sum_b_y = _mm256_add_pd(sum_b_y, b_y);
        sum_b_z = _mm256_add_pd(sum_b_z, b_z);

        __m256d delta_x = _mm256_sub_pd(a_x, b_x);
        __m256d delta_y = _mm256_sub_pd(a_y, b_y);
        __m256d delta_z = _mm256_sub_pd(a_z, b_z);

        sum_squared_x = _mm256_fmadd_pd(delta_x, delta_x, sum_squared_x);
        sum_squared_y = _mm256_fmadd_pd(delta_y, delta_y, sum_squared_y);
        sum_squared_z = _mm256_fmadd_pd(delta_z, delta_z, sum_squared_z);

        // Iteration 1
        __m256d a_x1, a_y1, a_z1, b_x1, b_y1, b_z1;
        _simsimd_deinterleave_f64x4_haswell(a + (i + 4) * 3, &a_x1, &a_y1, &a_z1);
        _simsimd_deinterleave_f64x4_haswell(b + (i + 4) * 3, &b_x1, &b_y1, &b_z1);

        sum_a_x = _mm256_add_pd(sum_a_x, a_x1);
        sum_a_y = _mm256_add_pd(sum_a_y, a_y1);
        sum_a_z = _mm256_add_pd(sum_a_z, a_z1);
        sum_b_x = _mm256_add_pd(sum_b_x, b_x1);
        sum_b_y = _mm256_add_pd(sum_b_y, b_y1);
        sum_b_z = _mm256_add_pd(sum_b_z, b_z1);

        __m256d delta_x1 = _mm256_sub_pd(a_x1, b_x1);
        __m256d delta_y1 = _mm256_sub_pd(a_y1, b_y1);
        __m256d delta_z1 = _mm256_sub_pd(a_z1, b_z1);

        sum_squared_x = _mm256_fmadd_pd(delta_x1, delta_x1, sum_squared_x);
        sum_squared_y = _mm256_fmadd_pd(delta_y1, delta_y1, sum_squared_y);
        sum_squared_z = _mm256_fmadd_pd(delta_z1, delta_z1, sum_squared_z);
    }

    // Handle 4-point remainder
    for (; i + 4 <= n; i += 4) {
        _simsimd_deinterleave_f64x4_haswell(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x4_haswell(b + i * 3, &b_x, &b_y, &b_z);

        sum_a_x = _mm256_add_pd(sum_a_x, a_x);
        sum_a_y = _mm256_add_pd(sum_a_y, a_y);
        sum_a_z = _mm256_add_pd(sum_a_z, a_z);
        sum_b_x = _mm256_add_pd(sum_b_x, b_x);
        sum_b_y = _mm256_add_pd(sum_b_y, b_y);
        sum_b_z = _mm256_add_pd(sum_b_z, b_z);

        __m256d delta_x = _mm256_sub_pd(a_x, b_x);
        __m256d delta_y = _mm256_sub_pd(a_y, b_y);
        __m256d delta_z = _mm256_sub_pd(a_z, b_z);

        sum_squared_x = _mm256_fmadd_pd(delta_x, delta_x, sum_squared_x);
        sum_squared_y = _mm256_fmadd_pd(delta_y, delta_y, sum_squared_y);
        sum_squared_z = _mm256_fmadd_pd(delta_z, delta_z, sum_squared_z);
    }

    // Reduce vectors to scalars
    simsimd_f64_t total_ax = _simsimd_reduce_add_f64x4_haswell(sum_a_x);
    simsimd_f64_t total_ay = _simsimd_reduce_add_f64x4_haswell(sum_a_y);
    simsimd_f64_t total_az = _simsimd_reduce_add_f64x4_haswell(sum_a_z);
    simsimd_f64_t total_bx = _simsimd_reduce_add_f64x4_haswell(sum_b_x);
    simsimd_f64_t total_by = _simsimd_reduce_add_f64x4_haswell(sum_b_y);
    simsimd_f64_t total_bz = _simsimd_reduce_add_f64x4_haswell(sum_b_z);
    simsimd_f64_t total_sq_x = _simsimd_reduce_add_f64x4_haswell(sum_squared_x);
    simsimd_f64_t total_sq_y = _simsimd_reduce_add_f64x4_haswell(sum_squared_y);
    simsimd_f64_t total_sq_z = _simsimd_reduce_add_f64x4_haswell(sum_squared_z);

    // Scalar tail
    for (; i < n; ++i) {
        simsimd_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        simsimd_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        total_ax += ax;
        total_ay += ay;
        total_az += az;
        total_bx += bx;
        total_by += by;
        total_bz += bz;
        simsimd_f64_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        total_sq_x += delta_x * delta_x;
        total_sq_y += delta_y * delta_y;
        total_sq_z += delta_z * delta_z;
    }

    // Compute centroids
    simsimd_f64_t inv_n = 1.0 / (simsimd_f64_t)n;
    simsimd_f64_t centroid_a_x = total_ax * inv_n;
    simsimd_f64_t centroid_a_y = total_ay * inv_n;
    simsimd_f64_t centroid_a_z = total_az * inv_n;
    simsimd_f64_t centroid_b_x = total_bx * inv_n;
    simsimd_f64_t centroid_b_y = total_by * inv_n;
    simsimd_f64_t centroid_b_z = total_bz * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Compute RMSD
    simsimd_f64_t mean_diff_x = centroid_a_x - centroid_b_x;
    simsimd_f64_t mean_diff_y = centroid_a_y - centroid_b_y;
    simsimd_f64_t mean_diff_z = centroid_a_z - centroid_b_z;
    simsimd_f64_t sum_squared = total_sq_x + total_sq_y + total_sq_z;
    simsimd_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = SIMSIMD_SQRT((simsimd_distance_t)(sum_squared * inv_n - mean_diff_sq));
}

SIMSIMD_PUBLIC void simsimd_kabsch_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                               simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid,
                                               simsimd_f32_t *rotation, simsimd_distance_t *scale,
                                               simsimd_distance_t *result) {
    // Optimized fused single-pass implementation using AVX2.
    // Computes centroids and covariance matrix in one pass.
    __m256 const zeros_f32 = _mm256_setzero_ps();
    __m256d const zeros_f64 = _mm256_setzero_pd();

    // Accumulators for centroids (f64 for precision)
    __m256d sum_a_x_vec = zeros_f64, sum_a_y_vec = zeros_f64, sum_a_z_vec = zeros_f64;
    __m256d sum_b_x_vec = zeros_f64, sum_b_y_vec = zeros_f64, sum_b_z_vec = zeros_f64;

    // Accumulators for covariance matrix (sum of outer products)
    __m256d cov_xx = zeros_f64, cov_xy = zeros_f64, cov_xz = zeros_f64;
    __m256d cov_yx = zeros_f64, cov_yy = zeros_f64, cov_yz = zeros_f64;
    __m256d cov_zx = zeros_f64, cov_zy = zeros_f64, cov_zz = zeros_f64;

    simsimd_size_t i = 0;
    __m256 a_x_vec, a_y_vec, a_z_vec, b_x_vec, b_y_vec, b_z_vec;

    // Fused single-pass: accumulate sums and outer products together
    for (; i + 8 <= n; i += 8) {
        _simsimd_deinterleave_f32x8_haswell(a + i * 3, &a_x_vec, &a_y_vec, &a_z_vec);
        _simsimd_deinterleave_f32x8_haswell(b + i * 3, &b_x_vec, &b_y_vec, &b_z_vec);

        // Convert to f64 - low 4 elements
        __m256d a_x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(a_x_vec));
        __m256d a_y_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(a_y_vec));
        __m256d a_z_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(a_z_vec));
        __m256d b_x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(b_x_vec));
        __m256d b_y_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(b_y_vec));
        __m256d b_z_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(b_z_vec));

        // Accumulate centroids
        sum_a_x_vec = _mm256_add_pd(sum_a_x_vec, a_x_lo);
        sum_a_y_vec = _mm256_add_pd(sum_a_y_vec, a_y_lo);
        sum_a_z_vec = _mm256_add_pd(sum_a_z_vec, a_z_lo);
        sum_b_x_vec = _mm256_add_pd(sum_b_x_vec, b_x_lo);
        sum_b_y_vec = _mm256_add_pd(sum_b_y_vec, b_y_lo);
        sum_b_z_vec = _mm256_add_pd(sum_b_z_vec, b_z_lo);

        // Accumulate outer products (raw, not centered)
        cov_xx = _mm256_fmadd_pd(a_x_lo, b_x_lo, cov_xx);
        cov_xy = _mm256_fmadd_pd(a_x_lo, b_y_lo, cov_xy);
        cov_xz = _mm256_fmadd_pd(a_x_lo, b_z_lo, cov_xz);
        cov_yx = _mm256_fmadd_pd(a_y_lo, b_x_lo, cov_yx);
        cov_yy = _mm256_fmadd_pd(a_y_lo, b_y_lo, cov_yy);
        cov_yz = _mm256_fmadd_pd(a_y_lo, b_z_lo, cov_yz);
        cov_zx = _mm256_fmadd_pd(a_z_lo, b_x_lo, cov_zx);
        cov_zy = _mm256_fmadd_pd(a_z_lo, b_y_lo, cov_zy);
        cov_zz = _mm256_fmadd_pd(a_z_lo, b_z_lo, cov_zz);

        // High 4 elements
        __m256d a_x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(a_x_vec, 1));
        __m256d a_y_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(a_y_vec, 1));
        __m256d a_z_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(a_z_vec, 1));
        __m256d b_x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(b_x_vec, 1));
        __m256d b_y_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(b_y_vec, 1));
        __m256d b_z_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(b_z_vec, 1));

        sum_a_x_vec = _mm256_add_pd(sum_a_x_vec, a_x_hi);
        sum_a_y_vec = _mm256_add_pd(sum_a_y_vec, a_y_hi);
        sum_a_z_vec = _mm256_add_pd(sum_a_z_vec, a_z_hi);
        sum_b_x_vec = _mm256_add_pd(sum_b_x_vec, b_x_hi);
        sum_b_y_vec = _mm256_add_pd(sum_b_y_vec, b_y_hi);
        sum_b_z_vec = _mm256_add_pd(sum_b_z_vec, b_z_hi);

        cov_xx = _mm256_fmadd_pd(a_x_hi, b_x_hi, cov_xx);
        cov_xy = _mm256_fmadd_pd(a_x_hi, b_y_hi, cov_xy);
        cov_xz = _mm256_fmadd_pd(a_x_hi, b_z_hi, cov_xz);
        cov_yx = _mm256_fmadd_pd(a_y_hi, b_x_hi, cov_yx);
        cov_yy = _mm256_fmadd_pd(a_y_hi, b_y_hi, cov_yy);
        cov_yz = _mm256_fmadd_pd(a_y_hi, b_z_hi, cov_yz);
        cov_zx = _mm256_fmadd_pd(a_z_hi, b_x_hi, cov_zx);
        cov_zy = _mm256_fmadd_pd(a_z_hi, b_y_hi, cov_zy);
        cov_zz = _mm256_fmadd_pd(a_z_hi, b_z_hi, cov_zz);
    }

    // Reduce vector accumulators
    simsimd_f64_t sum_a_x = _simsimd_reduce_add_f64x4_haswell(sum_a_x_vec);
    simsimd_f64_t sum_a_y = _simsimd_reduce_add_f64x4_haswell(sum_a_y_vec);
    simsimd_f64_t sum_a_z = _simsimd_reduce_add_f64x4_haswell(sum_a_z_vec);
    simsimd_f64_t sum_b_x = _simsimd_reduce_add_f64x4_haswell(sum_b_x_vec);
    simsimd_f64_t sum_b_y = _simsimd_reduce_add_f64x4_haswell(sum_b_y_vec);
    simsimd_f64_t sum_b_z = _simsimd_reduce_add_f64x4_haswell(sum_b_z_vec);

    simsimd_f64_t H00 = _simsimd_reduce_add_f64x4_haswell(cov_xx);
    simsimd_f64_t H01 = _simsimd_reduce_add_f64x4_haswell(cov_xy);
    simsimd_f64_t H02 = _simsimd_reduce_add_f64x4_haswell(cov_xz);
    simsimd_f64_t H10 = _simsimd_reduce_add_f64x4_haswell(cov_yx);
    simsimd_f64_t H11 = _simsimd_reduce_add_f64x4_haswell(cov_yy);
    simsimd_f64_t H12 = _simsimd_reduce_add_f64x4_haswell(cov_yz);
    simsimd_f64_t H20 = _simsimd_reduce_add_f64x4_haswell(cov_zx);
    simsimd_f64_t H21 = _simsimd_reduce_add_f64x4_haswell(cov_zy);
    simsimd_f64_t H22 = _simsimd_reduce_add_f64x4_haswell(cov_zz);

    // Scalar tail
    for (; i < n; ++i) {
        simsimd_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        simsimd_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        sum_a_x += ax;
        sum_a_y += ay;
        sum_a_z += az;
        sum_b_x += bx;
        sum_b_y += by;
        sum_b_z += bz;
        H00 += ax * bx;
        H01 += ax * by;
        H02 += ax * bz;
        H10 += ay * bx;
        H11 += ay * by;
        H12 += ay * bz;
        H20 += az * bx;
        H21 += az * by;
        H22 += az * bz;
    }

    // Compute centroids
    simsimd_f64_t inv_n = 1.0 / (simsimd_f64_t)n;
    simsimd_f64_t centroid_a_x = sum_a_x * inv_n;
    simsimd_f64_t centroid_a_y = sum_a_y * inv_n;
    simsimd_f64_t centroid_a_z = sum_a_z * inv_n;
    simsimd_f64_t centroid_b_x = sum_b_x * inv_n;
    simsimd_f64_t centroid_b_y = sum_b_y * inv_n;
    simsimd_f64_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) {
        a_centroid[0] = (simsimd_f32_t)centroid_a_x;
        a_centroid[1] = (simsimd_f32_t)centroid_a_y;
        a_centroid[2] = (simsimd_f32_t)centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = (simsimd_f32_t)centroid_b_x;
        b_centroid[1] = (simsimd_f32_t)centroid_b_y;
        b_centroid[2] = (simsimd_f32_t)centroid_b_z;
    }

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_b^T
    H00 -= n * centroid_a_x * centroid_b_x;
    H01 -= n * centroid_a_x * centroid_b_y;
    H02 -= n * centroid_a_x * centroid_b_z;
    H10 -= n * centroid_a_y * centroid_b_x;
    H11 -= n * centroid_a_y * centroid_b_y;
    H12 -= n * centroid_a_y * centroid_b_z;
    H20 -= n * centroid_a_z * centroid_b_x;
    H21 -= n * centroid_a_z * centroid_b_y;
    H22 -= n * centroid_a_z * centroid_b_z;

    // Compute SVD and optimal rotation
    simsimd_f32_t cross_covariance[9] = {(simsimd_f32_t)H00, (simsimd_f32_t)H01, (simsimd_f32_t)H02,
                                         (simsimd_f32_t)H10, (simsimd_f32_t)H11, (simsimd_f32_t)H12,
                                         (simsimd_f32_t)H20, (simsimd_f32_t)H21, (simsimd_f32_t)H22};
    simsimd_f32_t svd_u[9], svd_s[3], svd_v[9];
    _simsimd_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * U^T
    simsimd_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection: if det(R) < 0, negate third column of V and recompute R
    if (_simsimd_det3x3_f32(r) < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix and scale=1.0 */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }
    if (scale) *scale = 1.0;

    // Compute RMSD after optimal rotation
    simsimd_f64_t sum_squared = 0.0;
    for (simsimd_size_t j = 0; j < n; ++j) {
        simsimd_f32_t pa[3], pb[3], ra[3];
        pa[0] = a[j * 3 + 0] - (simsimd_f32_t)centroid_a_x;
        pa[1] = a[j * 3 + 1] - (simsimd_f32_t)centroid_a_y;
        pa[2] = a[j * 3 + 2] - (simsimd_f32_t)centroid_a_z;
        pb[0] = b[j * 3 + 0] - (simsimd_f32_t)centroid_b_x;
        pb[1] = b[j * 3 + 1] - (simsimd_f32_t)centroid_b_y;
        pb[2] = b[j * 3 + 2] - (simsimd_f32_t)centroid_b_z;

        ra[0] = r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2];
        ra[1] = r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2];
        ra[2] = r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2];

        simsimd_f32_t delta_x = ra[0] - pb[0];
        simsimd_f32_t delta_y = ra[1] - pb[1];
        simsimd_f32_t delta_z = ra[2] - pb[2];
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = SIMSIMD_SQRT(sum_squared * inv_n);
}

SIMSIMD_PUBLIC void simsimd_kabsch_f64_haswell(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                               simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid,
                                               simsimd_f64_t *rotation, simsimd_distance_t *scale,
                                               simsimd_distance_t *result) {
    __m256d const zeros = _mm256_setzero_pd();

    // Accumulators for centroids
    __m256d sum_a_x_vec = zeros, sum_a_y_vec = zeros, sum_a_z_vec = zeros;
    __m256d sum_b_x_vec = zeros, sum_b_y_vec = zeros, sum_b_z_vec = zeros;

    // Accumulators for covariance matrix (sum of outer products)
    __m256d cov_xx = zeros, cov_xy = zeros, cov_xz = zeros;
    __m256d cov_yx = zeros, cov_yy = zeros, cov_yz = zeros;
    __m256d cov_zx = zeros, cov_zy = zeros, cov_zz = zeros;

    simsimd_size_t i = 0;
    __m256d a_x, a_y, a_z, b_x, b_y, b_z;

    // Fused single-pass
    for (; i + 4 <= n; i += 4) {
        _simsimd_deinterleave_f64x4_haswell(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x4_haswell(b + i * 3, &b_x, &b_y, &b_z);

        sum_a_x_vec = _mm256_add_pd(sum_a_x_vec, a_x);
        sum_a_y_vec = _mm256_add_pd(sum_a_y_vec, a_y);
        sum_a_z_vec = _mm256_add_pd(sum_a_z_vec, a_z);
        sum_b_x_vec = _mm256_add_pd(sum_b_x_vec, b_x);
        sum_b_y_vec = _mm256_add_pd(sum_b_y_vec, b_y);
        sum_b_z_vec = _mm256_add_pd(sum_b_z_vec, b_z);

        cov_xx = _mm256_fmadd_pd(a_x, b_x, cov_xx);
        cov_xy = _mm256_fmadd_pd(a_x, b_y, cov_xy);
        cov_xz = _mm256_fmadd_pd(a_x, b_z, cov_xz);
        cov_yx = _mm256_fmadd_pd(a_y, b_x, cov_yx);
        cov_yy = _mm256_fmadd_pd(a_y, b_y, cov_yy);
        cov_yz = _mm256_fmadd_pd(a_y, b_z, cov_yz);
        cov_zx = _mm256_fmadd_pd(a_z, b_x, cov_zx);
        cov_zy = _mm256_fmadd_pd(a_z, b_y, cov_zy);
        cov_zz = _mm256_fmadd_pd(a_z, b_z, cov_zz);
    }

    // Reduce vector accumulators
    simsimd_f64_t sum_a_x = _simsimd_reduce_add_f64x4_haswell(sum_a_x_vec);
    simsimd_f64_t sum_a_y = _simsimd_reduce_add_f64x4_haswell(sum_a_y_vec);
    simsimd_f64_t sum_a_z = _simsimd_reduce_add_f64x4_haswell(sum_a_z_vec);
    simsimd_f64_t sum_b_x = _simsimd_reduce_add_f64x4_haswell(sum_b_x_vec);
    simsimd_f64_t sum_b_y = _simsimd_reduce_add_f64x4_haswell(sum_b_y_vec);
    simsimd_f64_t sum_b_z = _simsimd_reduce_add_f64x4_haswell(sum_b_z_vec);

    simsimd_f64_t H00 = _simsimd_reduce_add_f64x4_haswell(cov_xx);
    simsimd_f64_t H01 = _simsimd_reduce_add_f64x4_haswell(cov_xy);
    simsimd_f64_t H02 = _simsimd_reduce_add_f64x4_haswell(cov_xz);
    simsimd_f64_t H10 = _simsimd_reduce_add_f64x4_haswell(cov_yx);
    simsimd_f64_t H11 = _simsimd_reduce_add_f64x4_haswell(cov_yy);
    simsimd_f64_t H12 = _simsimd_reduce_add_f64x4_haswell(cov_yz);
    simsimd_f64_t H20 = _simsimd_reduce_add_f64x4_haswell(cov_zx);
    simsimd_f64_t H21 = _simsimd_reduce_add_f64x4_haswell(cov_zy);
    simsimd_f64_t H22 = _simsimd_reduce_add_f64x4_haswell(cov_zz);

    // Scalar tail
    for (; i < n; ++i) {
        simsimd_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        simsimd_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        sum_a_x += ax;
        sum_a_y += ay;
        sum_a_z += az;
        sum_b_x += bx;
        sum_b_y += by;
        sum_b_z += bz;
        H00 += ax * bx;
        H01 += ax * by;
        H02 += ax * bz;
        H10 += ay * bx;
        H11 += ay * by;
        H12 += ay * bz;
        H20 += az * bx;
        H21 += az * by;
        H22 += az * bz;
    }

    // Compute centroids
    simsimd_f64_t inv_n = 1.0 / (simsimd_f64_t)n;
    simsimd_f64_t centroid_a_x = sum_a_x * inv_n;
    simsimd_f64_t centroid_a_y = sum_a_y * inv_n;
    simsimd_f64_t centroid_a_z = sum_a_z * inv_n;
    simsimd_f64_t centroid_b_x = sum_b_x * inv_n;
    simsimd_f64_t centroid_b_y = sum_b_y * inv_n;
    simsimd_f64_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_b^T
    H00 -= n * centroid_a_x * centroid_b_x;
    H01 -= n * centroid_a_x * centroid_b_y;
    H02 -= n * centroid_a_x * centroid_b_z;
    H10 -= n * centroid_a_y * centroid_b_x;
    H11 -= n * centroid_a_y * centroid_b_y;
    H12 -= n * centroid_a_y * centroid_b_z;
    H20 -= n * centroid_a_z * centroid_b_x;
    H21 -= n * centroid_a_z * centroid_b_y;
    H22 -= n * centroid_a_z * centroid_b_z;

    // Compute SVD and optimal rotation (using f32 SVD for performance)
    simsimd_f32_t cross_covariance[9] = {(simsimd_f32_t)H00, (simsimd_f32_t)H01, (simsimd_f32_t)H02,
                                         (simsimd_f32_t)H10, (simsimd_f32_t)H11, (simsimd_f32_t)H12,
                                         (simsimd_f32_t)H20, (simsimd_f32_t)H21, (simsimd_f32_t)H22};
    simsimd_f32_t svd_u[9], svd_s[3], svd_v[9];
    _simsimd_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * U^T
    simsimd_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection: if det(R) < 0, negate third column of V and recompute R
    if (_simsimd_det3x3_f32(r) < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix and scale=1.0 */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = (simsimd_f64_t)r[j];
    }
    if (scale) *scale = 1.0;

    // Compute RMSD after optimal rotation (use f64 for precision)
    simsimd_f64_t sum_squared = 0.0;
    for (simsimd_size_t j = 0; j < n; ++j) {
        simsimd_f64_t pa[3], pb[3], ra[3];
        pa[0] = a[j * 3 + 0] - centroid_a_x;
        pa[1] = a[j * 3 + 1] - centroid_a_y;
        pa[2] = a[j * 3 + 2] - centroid_a_z;
        pb[0] = b[j * 3 + 0] - centroid_b_x;
        pb[1] = b[j * 3 + 1] - centroid_b_y;
        pb[2] = b[j * 3 + 2] - centroid_b_z;

        ra[0] = r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2];
        ra[1] = r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2];
        ra[2] = r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2];

        simsimd_f64_t delta_x = ra[0] - pb[0];
        simsimd_f64_t delta_y = ra[1] - pb[1];
        simsimd_f64_t delta_z = ra[2] - pb[2];
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = SIMSIMD_SQRT(sum_squared * inv_n);
}

SIMSIMD_PUBLIC void simsimd_umeyama_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                                simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid,
                                                simsimd_f32_t *rotation, simsimd_distance_t *scale,
                                                simsimd_distance_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A
    __m256i const gather_idx = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    __m256 const zeros_f32 = _mm256_setzero_ps();
    __m256d const zeros_f64 = _mm256_setzero_pd();

    __m256d sum_a_x_vec = zeros_f64, sum_a_y_vec = zeros_f64, sum_a_z_vec = zeros_f64;
    __m256d sum_b_x_vec = zeros_f64, sum_b_y_vec = zeros_f64, sum_b_z_vec = zeros_f64;
    __m256d cov_xx = zeros_f64, cov_xy = zeros_f64, cov_xz = zeros_f64;
    __m256d cov_yx = zeros_f64, cov_yy = zeros_f64, cov_yz = zeros_f64;
    __m256d cov_zx = zeros_f64, cov_zy = zeros_f64, cov_zz = zeros_f64;
    __m256d variance_a_acc = zeros_f64;

    simsimd_size_t i = 0;
    __m256 a_x_vec, a_y_vec, a_z_vec, b_x_vec, b_y_vec, b_z_vec;

    for (; i + 8 <= n; i += 8) {
        _simsimd_deinterleave_f32x8_haswell(a + i * 3, &a_x_vec, &a_y_vec, &a_z_vec);
        _simsimd_deinterleave_f32x8_haswell(b + i * 3, &b_x_vec, &b_y_vec, &b_z_vec);

        __m256d a_x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(a_x_vec));
        __m256d a_y_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(a_y_vec));
        __m256d a_z_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(a_z_vec));
        __m256d b_x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(b_x_vec));
        __m256d b_y_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(b_y_vec));
        __m256d b_z_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(b_z_vec));

        sum_a_x_vec = _mm256_add_pd(sum_a_x_vec, a_x_lo), sum_a_y_vec = _mm256_add_pd(sum_a_y_vec, a_y_lo);
        sum_a_z_vec = _mm256_add_pd(sum_a_z_vec, a_z_lo);
        sum_b_x_vec = _mm256_add_pd(sum_b_x_vec, b_x_lo), sum_b_y_vec = _mm256_add_pd(sum_b_y_vec, b_y_lo);
        sum_b_z_vec = _mm256_add_pd(sum_b_z_vec, b_z_lo);

        cov_xx = _mm256_fmadd_pd(a_x_lo, b_x_lo, cov_xx), cov_xy = _mm256_fmadd_pd(a_x_lo, b_y_lo, cov_xy);
        cov_xz = _mm256_fmadd_pd(a_x_lo, b_z_lo, cov_xz);
        cov_yx = _mm256_fmadd_pd(a_y_lo, b_x_lo, cov_yx), cov_yy = _mm256_fmadd_pd(a_y_lo, b_y_lo, cov_yy);
        cov_yz = _mm256_fmadd_pd(a_y_lo, b_z_lo, cov_yz);
        cov_zx = _mm256_fmadd_pd(a_z_lo, b_x_lo, cov_zx), cov_zy = _mm256_fmadd_pd(a_z_lo, b_y_lo, cov_zy);
        cov_zz = _mm256_fmadd_pd(a_z_lo, b_z_lo, cov_zz);
        variance_a_acc = _mm256_fmadd_pd(a_x_lo, a_x_lo, variance_a_acc);
        variance_a_acc = _mm256_fmadd_pd(a_y_lo, a_y_lo, variance_a_acc);
        variance_a_acc = _mm256_fmadd_pd(a_z_lo, a_z_lo, variance_a_acc);

        __m256d a_x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(a_x_vec, 1));
        __m256d a_y_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(a_y_vec, 1));
        __m256d a_z_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(a_z_vec, 1));
        __m256d b_x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(b_x_vec, 1));
        __m256d b_y_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(b_y_vec, 1));
        __m256d b_z_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(b_z_vec, 1));

        sum_a_x_vec = _mm256_add_pd(sum_a_x_vec, a_x_hi), sum_a_y_vec = _mm256_add_pd(sum_a_y_vec, a_y_hi);
        sum_a_z_vec = _mm256_add_pd(sum_a_z_vec, a_z_hi);
        sum_b_x_vec = _mm256_add_pd(sum_b_x_vec, b_x_hi), sum_b_y_vec = _mm256_add_pd(sum_b_y_vec, b_y_hi);
        sum_b_z_vec = _mm256_add_pd(sum_b_z_vec, b_z_hi);

        cov_xx = _mm256_fmadd_pd(a_x_hi, b_x_hi, cov_xx), cov_xy = _mm256_fmadd_pd(a_x_hi, b_y_hi, cov_xy);
        cov_xz = _mm256_fmadd_pd(a_x_hi, b_z_hi, cov_xz);
        cov_yx = _mm256_fmadd_pd(a_y_hi, b_x_hi, cov_yx), cov_yy = _mm256_fmadd_pd(a_y_hi, b_y_hi, cov_yy);
        cov_yz = _mm256_fmadd_pd(a_y_hi, b_z_hi, cov_yz);
        cov_zx = _mm256_fmadd_pd(a_z_hi, b_x_hi, cov_zx), cov_zy = _mm256_fmadd_pd(a_z_hi, b_y_hi, cov_zy);
        cov_zz = _mm256_fmadd_pd(a_z_hi, b_z_hi, cov_zz);
        variance_a_acc = _mm256_fmadd_pd(a_x_hi, a_x_hi, variance_a_acc);
        variance_a_acc = _mm256_fmadd_pd(a_y_hi, a_y_hi, variance_a_acc);
        variance_a_acc = _mm256_fmadd_pd(a_z_hi, a_z_hi, variance_a_acc);
    }

    // Reduce vector accumulators
    simsimd_f64_t sum_a_x = _simsimd_reduce_add_f64x4_haswell(sum_a_x_vec);
    simsimd_f64_t sum_a_y = _simsimd_reduce_add_f64x4_haswell(sum_a_y_vec);
    simsimd_f64_t sum_a_z = _simsimd_reduce_add_f64x4_haswell(sum_a_z_vec);
    simsimd_f64_t sum_b_x = _simsimd_reduce_add_f64x4_haswell(sum_b_x_vec);
    simsimd_f64_t sum_b_y = _simsimd_reduce_add_f64x4_haswell(sum_b_y_vec);
    simsimd_f64_t sum_b_z = _simsimd_reduce_add_f64x4_haswell(sum_b_z_vec);
    simsimd_f64_t H00 = _simsimd_reduce_add_f64x4_haswell(cov_xx);
    simsimd_f64_t H01 = _simsimd_reduce_add_f64x4_haswell(cov_xy);
    simsimd_f64_t H02 = _simsimd_reduce_add_f64x4_haswell(cov_xz);
    simsimd_f64_t H10 = _simsimd_reduce_add_f64x4_haswell(cov_yx);
    simsimd_f64_t H11 = _simsimd_reduce_add_f64x4_haswell(cov_yy);
    simsimd_f64_t H12 = _simsimd_reduce_add_f64x4_haswell(cov_yz);
    simsimd_f64_t H20 = _simsimd_reduce_add_f64x4_haswell(cov_zx);
    simsimd_f64_t H21 = _simsimd_reduce_add_f64x4_haswell(cov_zy);
    simsimd_f64_t H22 = _simsimd_reduce_add_f64x4_haswell(cov_zz);
    simsimd_f64_t variance_a_sum = _simsimd_reduce_add_f64x4_haswell(variance_a_acc);

    // Scalar tail
    for (; i < n; ++i) {
        simsimd_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        simsimd_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        sum_a_x += ax;
        sum_a_y += ay;
        sum_a_z += az;
        sum_b_x += bx;
        sum_b_y += by;
        sum_b_z += bz;
        H00 += ax * bx;
        H01 += ax * by;
        H02 += ax * bz;
        H10 += ay * bx;
        H11 += ay * by;
        H12 += ay * bz;
        H20 += az * bx;
        H21 += az * by;
        H22 += az * bz;
        variance_a_sum += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    simsimd_f64_t inv_n = 1.0 / (simsimd_f64_t)n;
    simsimd_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    simsimd_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid)
        a_centroid[0] = (simsimd_f32_t)centroid_a_x, a_centroid[1] = (simsimd_f32_t)centroid_a_y,
        a_centroid[2] = (simsimd_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (simsimd_f32_t)centroid_b_x, b_centroid[1] = (simsimd_f32_t)centroid_b_y,
        b_centroid[2] = (simsimd_f32_t)centroid_b_z;

    // Compute centered covariance and variance
    simsimd_f64_t variance_a = variance_a_sum * inv_n - (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                         centroid_a_z * centroid_a_z);

    simsimd_f32_t cross_covariance[9];
    cross_covariance[0] = (simsimd_f32_t)(H00 - n * centroid_a_x * centroid_b_x);
    cross_covariance[1] = (simsimd_f32_t)(H01 - n * centroid_a_x * centroid_b_y);
    cross_covariance[2] = (simsimd_f32_t)(H02 - n * centroid_a_x * centroid_b_z);
    cross_covariance[3] = (simsimd_f32_t)(H10 - n * centroid_a_y * centroid_b_x);
    cross_covariance[4] = (simsimd_f32_t)(H11 - n * centroid_a_y * centroid_b_y);
    cross_covariance[5] = (simsimd_f32_t)(H12 - n * centroid_a_y * centroid_b_z);
    cross_covariance[6] = (simsimd_f32_t)(H20 - n * centroid_a_z * centroid_b_x);
    cross_covariance[7] = (simsimd_f32_t)(H21 - n * centroid_a_z * centroid_b_y);
    cross_covariance[8] = (simsimd_f32_t)(H22 - n * centroid_a_z * centroid_b_z);

    // SVD
    simsimd_f32_t svd_u[9], svd_s[9], svd_v[9];
    _simsimd_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * U^T
    simsimd_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Scale factor: c = trace(D*S) / (n * variance_a)
    simsimd_f32_t det = _simsimd_det3x3_f32(r);
    simsimd_f32_t d3 = det < 0 ? -1.0f : 1.0f;
    simsimd_f32_t trace_ds = svd_s[0] + svd_s[4] + d3 * svd_s[8];
    simsimd_f64_t c = (simsimd_f64_t)trace_ds / (n * variance_a);
    if (scale) *scale = c;

    // Handle reflection
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }

    // Compute RMSD with scaling using serial loop (simpler for Haswell)
    simsimd_f64_t sum_squared = 0.0;
    for (simsimd_size_t j = 0; j < n; ++j) {
        simsimd_f64_t pa[3], pb[3], ra[3];
        pa[0] = a[j * 3 + 0] - centroid_a_x;
        pa[1] = a[j * 3 + 1] - centroid_a_y;
        pa[2] = a[j * 3 + 2] - centroid_a_z;
        pb[0] = b[j * 3 + 0] - centroid_b_x;
        pb[1] = b[j * 3 + 1] - centroid_b_y;
        pb[2] = b[j * 3 + 2] - centroid_b_z;

        ra[0] = c * (r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2]);
        ra[1] = c * (r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2]);
        ra[2] = c * (r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2]);

        simsimd_f64_t delta_x = ra[0] - pb[0];
        simsimd_f64_t delta_y = ra[1] - pb[1];
        simsimd_f64_t delta_z = ra[2] - pb[2];
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = SIMSIMD_SQRT(sum_squared * inv_n);
}

SIMSIMD_PUBLIC void simsimd_umeyama_f64_haswell(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                                simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid,
                                                simsimd_f64_t *rotation, simsimd_distance_t *scale,
                                                simsimd_distance_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A
    __m256d const zeros = _mm256_setzero_pd();

    __m256d sum_a_x_vec = zeros, sum_a_y_vec = zeros, sum_a_z_vec = zeros;
    __m256d sum_b_x_vec = zeros, sum_b_y_vec = zeros, sum_b_z_vec = zeros;
    __m256d cov_xx = zeros, cov_xy = zeros, cov_xz = zeros;
    __m256d cov_yx = zeros, cov_yy = zeros, cov_yz = zeros;
    __m256d cov_zx = zeros, cov_zy = zeros, cov_zz = zeros;
    __m256d variance_a_acc = zeros;

    simsimd_size_t i = 0;
    __m256d a_x, a_y, a_z, b_x, b_y, b_z;

    for (; i + 4 <= n; i += 4) {
        _simsimd_deinterleave_f64x4_haswell(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x4_haswell(b + i * 3, &b_x, &b_y, &b_z);

        sum_a_x_vec = _mm256_add_pd(sum_a_x_vec, a_x), sum_a_y_vec = _mm256_add_pd(sum_a_y_vec, a_y);
        sum_a_z_vec = _mm256_add_pd(sum_a_z_vec, a_z);
        sum_b_x_vec = _mm256_add_pd(sum_b_x_vec, b_x), sum_b_y_vec = _mm256_add_pd(sum_b_y_vec, b_y);
        sum_b_z_vec = _mm256_add_pd(sum_b_z_vec, b_z);

        cov_xx = _mm256_fmadd_pd(a_x, b_x, cov_xx), cov_xy = _mm256_fmadd_pd(a_x, b_y, cov_xy);
        cov_xz = _mm256_fmadd_pd(a_x, b_z, cov_xz);
        cov_yx = _mm256_fmadd_pd(a_y, b_x, cov_yx), cov_yy = _mm256_fmadd_pd(a_y, b_y, cov_yy);
        cov_yz = _mm256_fmadd_pd(a_y, b_z, cov_yz);
        cov_zx = _mm256_fmadd_pd(a_z, b_x, cov_zx), cov_zy = _mm256_fmadd_pd(a_z, b_y, cov_zy);
        cov_zz = _mm256_fmadd_pd(a_z, b_z, cov_zz);
        variance_a_acc = _mm256_fmadd_pd(a_x, a_x, variance_a_acc);
        variance_a_acc = _mm256_fmadd_pd(a_y, a_y, variance_a_acc);
        variance_a_acc = _mm256_fmadd_pd(a_z, a_z, variance_a_acc);
    }

    // Reduce vector accumulators
    simsimd_f64_t sum_a_x = _simsimd_reduce_add_f64x4_haswell(sum_a_x_vec);
    simsimd_f64_t sum_a_y = _simsimd_reduce_add_f64x4_haswell(sum_a_y_vec);
    simsimd_f64_t sum_a_z = _simsimd_reduce_add_f64x4_haswell(sum_a_z_vec);
    simsimd_f64_t sum_b_x = _simsimd_reduce_add_f64x4_haswell(sum_b_x_vec);
    simsimd_f64_t sum_b_y = _simsimd_reduce_add_f64x4_haswell(sum_b_y_vec);
    simsimd_f64_t sum_b_z = _simsimd_reduce_add_f64x4_haswell(sum_b_z_vec);
    simsimd_f64_t h00_s = _simsimd_reduce_add_f64x4_haswell(cov_xx);
    simsimd_f64_t h01_s = _simsimd_reduce_add_f64x4_haswell(cov_xy);
    simsimd_f64_t h02_s = _simsimd_reduce_add_f64x4_haswell(cov_xz);
    simsimd_f64_t h10_s = _simsimd_reduce_add_f64x4_haswell(cov_yx);
    simsimd_f64_t h11_s = _simsimd_reduce_add_f64x4_haswell(cov_yy);
    simsimd_f64_t h12_s = _simsimd_reduce_add_f64x4_haswell(cov_yz);
    simsimd_f64_t h20_s = _simsimd_reduce_add_f64x4_haswell(cov_zx);
    simsimd_f64_t h21_s = _simsimd_reduce_add_f64x4_haswell(cov_zy);
    simsimd_f64_t h22_s = _simsimd_reduce_add_f64x4_haswell(cov_zz);
    simsimd_f64_t variance_a_sum = _simsimd_reduce_add_f64x4_haswell(variance_a_acc);

    // Scalar tail loop for remaining points
    for (; i < n; i++) {
        simsimd_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        simsimd_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        sum_a_x += ax;
        sum_a_y += ay;
        sum_a_z += az;
        sum_b_x += bx;
        sum_b_y += by;
        sum_b_z += bz;
        h00_s += ax * bx;
        h01_s += ax * by;
        h02_s += ax * bz;
        h10_s += ay * bx;
        h11_s += ay * by;
        h12_s += ay * bz;
        h20_s += az * bx;
        h21_s += az * by;
        h22_s += az * bz;
        variance_a_sum += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    simsimd_f64_t inv_n = 1.0 / (simsimd_f64_t)n;

    simsimd_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    simsimd_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance and variance
    simsimd_f64_t variance_a = variance_a_sum * inv_n - (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                         centroid_a_z * centroid_a_z);

    simsimd_f32_t cross_covariance[9];
    cross_covariance[0] = (simsimd_f32_t)(h00_s - sum_a_x * sum_b_x * inv_n);
    cross_covariance[1] = (simsimd_f32_t)(h01_s - sum_a_x * sum_b_y * inv_n);
    cross_covariance[2] = (simsimd_f32_t)(h02_s - sum_a_x * sum_b_z * inv_n);
    cross_covariance[3] = (simsimd_f32_t)(h10_s - sum_a_y * sum_b_x * inv_n);
    cross_covariance[4] = (simsimd_f32_t)(h11_s - sum_a_y * sum_b_y * inv_n);
    cross_covariance[5] = (simsimd_f32_t)(h12_s - sum_a_y * sum_b_z * inv_n);
    cross_covariance[6] = (simsimd_f32_t)(h20_s - sum_a_z * sum_b_x * inv_n);
    cross_covariance[7] = (simsimd_f32_t)(h21_s - sum_a_z * sum_b_y * inv_n);
    cross_covariance[8] = (simsimd_f32_t)(h22_s - sum_a_z * sum_b_z * inv_n);

    // SVD
    simsimd_f32_t svd_u[9], svd_s[9], svd_v[9];
    _simsimd_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * U^T
    simsimd_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Scale factor: c = trace(D*S) / (n * variance_a)
    simsimd_f32_t det = _simsimd_det3x3_f32(r);
    simsimd_f32_t d3 = det < 0 ? -1.0f : 1.0f;
    simsimd_f32_t trace_ds = svd_s[0] + svd_s[4] + d3 * svd_s[8];
    simsimd_f64_t c = (simsimd_f64_t)trace_ds / (n * variance_a);
    if (scale) *scale = c;

    // Handle reflection
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = (simsimd_f64_t)r[j];
    }

    // Compute RMSD with scaling using serial loop
    simsimd_f64_t sum_squared = 0.0;
    for (simsimd_size_t j = 0; j < n; ++j) {
        simsimd_f64_t pa[3], pb[3], ra[3];
        pa[0] = a[j * 3 + 0] - centroid_a_x;
        pa[1] = a[j * 3 + 1] - centroid_a_y;
        pa[2] = a[j * 3 + 2] - centroid_a_z;
        pb[0] = b[j * 3 + 0] - centroid_b_x;
        pb[1] = b[j * 3 + 1] - centroid_b_y;
        pb[2] = b[j * 3 + 2] - centroid_b_z;

        ra[0] = c * (r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2]);
        ra[1] = c * (r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2]);
        ra[2] = c * (r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2]);

        simsimd_f64_t delta_x = ra[0] - pb[0];
        simsimd_f64_t delta_y = ra[1] - pb[1];
        simsimd_f64_t delta_z = ra[2] - pb[2];
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = SIMSIMD_SQRT(sum_squared * inv_n);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL
#endif // _SIMSIMD_TARGET_X86

#if !SIMSIMD_DYNAMIC_DISPATCH

SIMSIMD_PUBLIC void simsimd_rmsd_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                     simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid, simsimd_f64_t *rotation,
                                     simsimd_distance_t *scale, simsimd_distance_t *result) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_rmsd_f64_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_rmsd_f64_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    simsimd_rmsd_f64_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_rmsd_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                     simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid, simsimd_f32_t *rotation,
                                     simsimd_distance_t *scale, simsimd_distance_t *result) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_rmsd_f32_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_rmsd_f32_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    simsimd_rmsd_f32_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_rmsd_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                     simsimd_f16_t *a_centroid, simsimd_f16_t *b_centroid, simsimd_f32_t *rotation,
                                     simsimd_distance_t *scale, simsimd_distance_t *result) {
    simsimd_rmsd_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

SIMSIMD_PUBLIC void simsimd_rmsd_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                      simsimd_bf16_t *a_centroid, simsimd_bf16_t *b_centroid, simsimd_f32_t *rotation,
                                      simsimd_distance_t *scale, simsimd_distance_t *result) {
    simsimd_rmsd_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

SIMSIMD_PUBLIC void simsimd_kabsch_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                       simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid, simsimd_f64_t *rotation,
                                       simsimd_distance_t *scale, simsimd_distance_t *result) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_kabsch_f64_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_kabsch_f64_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    simsimd_kabsch_f64_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_kabsch_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                       simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid, simsimd_f32_t *rotation,
                                       simsimd_distance_t *scale, simsimd_distance_t *result) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_kabsch_f32_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_kabsch_f32_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    simsimd_kabsch_f32_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_kabsch_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                       simsimd_f16_t *a_centroid, simsimd_f16_t *b_centroid, simsimd_f32_t *rotation,
                                       simsimd_distance_t *scale, simsimd_distance_t *result) {
    simsimd_kabsch_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

SIMSIMD_PUBLIC void simsimd_kabsch_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                        simsimd_bf16_t *a_centroid, simsimd_bf16_t *b_centroid, simsimd_f32_t *rotation,
                                        simsimd_distance_t *scale, simsimd_distance_t *result) {
    simsimd_kabsch_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

SIMSIMD_PUBLIC void simsimd_umeyama_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                        simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid, simsimd_f64_t *rotation,
                                        simsimd_distance_t *scale, simsimd_distance_t *result) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_umeyama_f64_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_umeyama_f64_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    simsimd_umeyama_f64_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_umeyama_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                        simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid, simsimd_f32_t *rotation,
                                        simsimd_distance_t *scale, simsimd_distance_t *result) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_umeyama_f32_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_umeyama_f32_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    simsimd_umeyama_f32_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_umeyama_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                        simsimd_f16_t *a_centroid, simsimd_f16_t *b_centroid, simsimd_f32_t *rotation,
                                        simsimd_distance_t *scale, simsimd_distance_t *result) {
    simsimd_umeyama_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

SIMSIMD_PUBLIC void simsimd_umeyama_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                         simsimd_bf16_t *a_centroid, simsimd_bf16_t *b_centroid,
                                         simsimd_f32_t *rotation, simsimd_distance_t *scale,
                                         simsimd_distance_t *result) {
    simsimd_umeyama_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

#endif // !SIMSIMD_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif // SIMSIMD_MESH_H
