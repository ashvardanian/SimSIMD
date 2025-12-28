/**
 *  @brief SIMD-accelerated similarity measures for meshes and rigid 3D bodies.
 *  @file include/numkong/mesh.h
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
 *  - 64-bit IEEE-754 floating point → 64-bit
 *  - 32-bit IEEE-754 floating point → 32-bit
 *  - 16-bit IEEE-754 floating point → 32-bit
 *  - 16-bit brain-floating point → 32-bit
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
#ifndef NK_MESH_H
#define NK_MESH_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief RMSD mesh superposition function.
 *
 *  The transformation aligns a to b: a'_i = scale * R * (a_i - a_centroid) + b_centroid
 *
 *  @param[in] a First point cloud (source), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] b Second point cloud (target), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] n Number of 3D points in each cloud.
 *  @param[out] a_centroid Centroid of first cloud (3 values). Can be NULL.
 *  @param[out] b_centroid Centroid of second cloud (3 values). Can be NULL.
 *  @param[out] rotation Row-major 3x3 rotation matrix (9 values), always identity. Can be NULL.
 *  @param[out] scale Scale factor applied, always 1. Can be NULL.
 *  @param[out] result RMSD after applying the transformation.
 */
NK_DYNAMIC void nk_rmsd_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                            nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_rmsd_f64 */
NK_DYNAMIC void nk_rmsd_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_rmsd_f64 */
NK_DYNAMIC void nk_rmsd_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_rmsd_f64 */
NK_DYNAMIC void nk_rmsd_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/**
 *  @brief Kabsch mesh superposition function.
 *
 *  The transformation aligns a to b: a'_i = scale * R * (a_i - a_centroid) + b_centroid
 *
 *  @param[in] a First point cloud (source), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] b Second point cloud (target), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] n Number of 3D points in each cloud.
 *  @param[out] a_centroid Centroid of first cloud (3 values). Can be NULL.
 *  @param[out] b_centroid Centroid of second cloud (3 values). Can be NULL.
 *  @param[out] rotation Row-major 3x3 rotation matrix (9 values). Can be NULL.
 *  @param[out] scale Scale factor applied, always 1. Can be NULL.
 *  @param[out] result RMSD after applying the transformation.
 */
NK_DYNAMIC void nk_kabsch_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                              nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f64 */
NK_DYNAMIC void nk_kabsch_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_f64 */
NK_DYNAMIC void nk_kabsch_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_f64 */
NK_DYNAMIC void nk_kabsch_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/**
 *  @brief Umeyama mesh superposition function.
 *
 *  The transformation aligns a to b: a'_i = scale * R * (a_i - a_centroid) + b_centroid
 *
 *  @param[in] a First point cloud (source), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] b Second point cloud (target), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] n Number of 3D points in each cloud.
 *  @param[out] a_centroid Centroid of first cloud (3 values). Can be NULL.
 *  @param[out] b_centroid Centroid of second cloud (3 values). Can be NULL.
 *  @param[out] rotation Row-major 3x3 rotation matrix (9 values). Can be NULL.
 *  @param[out] scale Scale factor applied. Can be NULL.
 *  @param[out] result RMSD after applying the transformation.
 */
NK_DYNAMIC void nk_umeyama_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                               nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f64 */
NK_DYNAMIC void nk_umeyama_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_f64 */
NK_DYNAMIC void nk_umeyama_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_f64 */
NK_DYNAMIC void nk_umeyama_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/** @copydoc nk_rmsd_f64 */
NK_PUBLIC void nk_rmsd_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                  nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f64 */
NK_PUBLIC void nk_kabsch_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                    nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f64 */
NK_PUBLIC void nk_umeyama_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);

/** @copydoc nk_rmsd_f32 */
NK_PUBLIC void nk_rmsd_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_f32 */
NK_PUBLIC void nk_kabsch_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_f32 */
NK_PUBLIC void nk_umeyama_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/** @copydoc nk_rmsd_f16 */
NK_PUBLIC void nk_rmsd_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_f16 */
NK_PUBLIC void nk_kabsch_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_f16 */
NK_PUBLIC void nk_umeyama_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/** @copydoc nk_rmsd_bf16 */
NK_PUBLIC void nk_rmsd_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_bf16 */
NK_PUBLIC void nk_kabsch_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_bf16 */
NK_PUBLIC void nk_umeyama_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/** @copydoc nk_rmsd_f32 */
NK_PUBLIC void nk_rmsd_f32_accurate(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                    nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f32 */
NK_PUBLIC void nk_kabsch_f32_accurate(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                      nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f32 */
NK_PUBLIC void nk_umeyama_f32_accurate(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                       nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);

/** @copydoc nk_rmsd_f16 */
NK_PUBLIC void nk_rmsd_f16_accurate(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                    nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f16 */
NK_PUBLIC void nk_kabsch_f16_accurate(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                      nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f16 */
NK_PUBLIC void nk_umeyama_f16_accurate(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                       nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);

/** @copydoc nk_rmsd_bf16 */
NK_PUBLIC void nk_rmsd_bf16_accurate(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_bf16 */
NK_PUBLIC void nk_kabsch_bf16_accurate(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                       nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_bf16 */
NK_PUBLIC void nk_umeyama_bf16_accurate(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                        nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);

/*  SIMD-powered backends for AVX512 CPUs of Skylake generation and newer.
 */
#if NK_TARGET_SKYLAKE
/** @copydoc nk_rmsd_f32 */
NK_PUBLIC void nk_rmsd_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_f32 */
NK_PUBLIC void nk_kabsch_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_f32 */
NK_PUBLIC void nk_umeyama_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/** @copydoc nk_rmsd_f64 */
NK_PUBLIC void nk_rmsd_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f64 */
NK_PUBLIC void nk_kabsch_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f64 */
NK_PUBLIC void nk_umeyama_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                      nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
#endif // NK_TARGET_SKYLAKE

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer.
 */
#if NK_TARGET_HASWELL
/** @copydoc nk_rmsd_f32 */
NK_PUBLIC void nk_rmsd_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_f32 */
NK_PUBLIC void nk_kabsch_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_f32 */
NK_PUBLIC void nk_umeyama_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/** @copydoc nk_rmsd_f64 */
NK_PUBLIC void nk_rmsd_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f64 */
NK_PUBLIC void nk_kabsch_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f64 */
NK_PUBLIC void nk_umeyama_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                      nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
#endif // NK_TARGET_HASWELL

/*  SIMD-powered backends for Arm NEON CPUs.
 */
#if NK_TARGET_NEON
/** @copydoc nk_rmsd_f32 */
NK_PUBLIC void nk_rmsd_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_f32 */
NK_PUBLIC void nk_kabsch_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_f32 */
NK_PUBLIC void nk_umeyama_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/** @copydoc nk_rmsd_f64 */
NK_PUBLIC void nk_rmsd_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f64 */
NK_PUBLIC void nk_kabsch_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                  nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f64 */
NK_PUBLIC void nk_umeyama_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
#endif // NK_TARGET_NEON

/**
 *  @brief  Returns the output datatype for RMSD.
 */
NK_INTERNAL nk_datatype_t nk_rmsd_output_datatype(nk_datatype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    default: return nk_datatype_unknown_k;
    }
}

/**
 *  @brief  Returns the output datatype for Kabsch alignment.
 */
NK_INTERNAL nk_datatype_t nk_kabsch_output_datatype(nk_datatype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    default: return nk_datatype_unknown_k;
    }
}

/**
 *  @brief  Returns the output datatype for Umeyama alignment.
 */
NK_INTERNAL nk_datatype_t nk_umeyama_output_datatype(nk_datatype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    default: return nk_datatype_unknown_k;
    }
}

#include "numkong/mesh/serial.h"
#include "numkong/mesh/neon.h"
#include "numkong/mesh/haswell.h"
#include "numkong/mesh/skylake.h"

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_rmsd_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                           nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_rmsd_f64_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_rmsd_f64_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEON
    nk_rmsd_f64_neon(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_rmsd_f64_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_rmsd_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
#if NK_TARGET_SKYLAKE
    nk_rmsd_f32_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_rmsd_f32_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEON
    nk_rmsd_f32_neon(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_rmsd_f32_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_rmsd_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_rmsd_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_rmsd_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_rmsd_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_kabsch_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                             nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_kabsch_f64_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_kabsch_f64_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEON
    nk_kabsch_f64_neon(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_kabsch_f64_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_kabsch_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
#if NK_TARGET_SKYLAKE
    nk_kabsch_f32_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_kabsch_f32_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEON
    nk_kabsch_f32_neon(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_kabsch_f32_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_kabsch_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_kabsch_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_kabsch_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_kabsch_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_umeyama_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                              nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_umeyama_f64_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_umeyama_f64_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEON
    nk_umeyama_f64_neon(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_umeyama_f64_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_umeyama_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
#if NK_TARGET_SKYLAKE
    nk_umeyama_f32_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_umeyama_f32_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEON
    nk_umeyama_f32_neon(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_umeyama_f32_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_umeyama_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_umeyama_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_umeyama_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_umeyama_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif // NK_MESH_H
