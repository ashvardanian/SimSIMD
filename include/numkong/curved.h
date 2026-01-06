/**
 *  @brief SIMD-accelerated Similarity Measures for Curved Spaces.
 *  @file include/numkong/curved.h
 *  @author Ash Vardanian
 *  @date August 27, 2024
 *
 *  Contains following similarity measures:
 *
 *  - Mahalanobis distance
 *  - Bilinear form multiplication
 *  - Bilinear form multiplication over complex numbers
 *
 *  For dtypes:
 *
 *  - 64-bit floating point numbers → 64-bit floats
 *  - 32-bit floating point numbers → 32-bit floats
 *  - 16-bit floating point numbers → 32-bit floats
 *  - 16-bit brain-floating point numbers → 32-bit floats
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON
 *  - x86: Haswell, Ice Lake, Skylake, Genoa, Sapphire
 *
 *  @section usage Usage and Benefits
 *
 *  These kernels target BLAS level 2 patterns where vectors are combined with a metric
 *  tensor or covariance matrix. Using raw bilinear and Mahalanobis forms avoids constructing
 *  intermediates and keeps memory traffic low, which is often faster than a full GEMM path
 *  for small and medium sizes. Complex bilinear forms return a complex scalar as two reals,
 *  serving complex-valued signals without extra packing or unpacking.
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NK_CURVED_H
#define NK_CURVED_H

#include "numkong/types.h"
#include "numkong/dot.h"
#include "numkong/spatial.h" // nk_substract_bf16x32_genoa_
#include "numkong/reduce.h"  // nk_reduce_add_f16x8_neonhalf_

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Bilinear form between vectors a and b under metric tensor c.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] c The metric tensor or covariance matrix, stored row-major as an n-by-n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] result The output bilinear form value.
 *
 *  @note The output value can be negative.
 */
NK_DYNAMIC void nk_bilinear_f64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n, nk_f64_t *result);
/**
 *  @brief Bilinear form between vectors a and b under metric tensor c.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] c The metric tensor or covariance matrix, stored row-major as an n-by-n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] result The output bilinear form value.
 *
 *  @note The output value can be negative.
 */
NK_DYNAMIC void nk_bilinear_f32(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n, nk_f32_t *result);
/**
 *  @brief Bilinear form between vectors a and b under metric tensor c.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] c The metric tensor or covariance matrix, stored row-major as an n-by-n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] result The output bilinear form value.
 *
 *  @note The output value can be negative.
 */
NK_DYNAMIC void nk_bilinear_f16(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n, nk_f32_t *result);
/**
 *  @brief Bilinear form between vectors a and b under metric tensor c.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] c The metric tensor or covariance matrix, stored row-major as an n-by-n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] result The output bilinear form value.
 *
 *  @note The output value can be negative.
 */
NK_DYNAMIC void nk_bilinear_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                 nk_f32_t *result);
/**
 *  @brief Mahalanobis distance between vectors a and b under metric tensor c.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] c The metric tensor or covariance matrix, stored row-major as an n-by-n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] result The output distance value.
 *
 *  @note The output value is non-negative.
 *  @note The output value is zero if and only if the two vectors are identical.
 */
NK_DYNAMIC void nk_mahalanobis_f64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                   nk_f64_t *result);
/**
 *  @brief Mahalanobis distance between vectors a and b under metric tensor c.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] c The metric tensor or covariance matrix, stored row-major as an n-by-n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] result The output distance value.
 *
 *  @note The output value is non-negative.
 *  @note The output value is zero if and only if the two vectors are identical.
 */
NK_DYNAMIC void nk_mahalanobis_f32(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                   nk_f32_t *result);
/**
 *  @brief Mahalanobis distance between vectors a and b under metric tensor c.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] c The metric tensor or covariance matrix, stored row-major as an n-by-n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] result The output distance value.
 *
 *  @note The output value is non-negative.
 *  @note The output value is zero if and only if the two vectors are identical.
 */
NK_DYNAMIC void nk_mahalanobis_f16(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                   nk_f32_t *result);
/**
 *  @brief Mahalanobis distance between vectors a and b under metric tensor c.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] c The metric tensor or covariance matrix, stored row-major as an n-by-n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] result The output distance value.
 *
 *  @note The output value is non-negative.
 *  @note The output value is zero if and only if the two vectors are identical.
 */
NK_DYNAMIC void nk_mahalanobis_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                    nk_f32_t *result);

/** @copydoc nk_bilinear_f64 */
NK_PUBLIC void nk_bilinear_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                      nk_f64_t *result);
/** @copydoc nk_bilinear_f64c */
NK_PUBLIC void nk_bilinear_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                       nk_f64c_t *results);
/** @copydoc nk_mahalanobis_f64 */
NK_PUBLIC void nk_mahalanobis_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                         nk_f64_t *result);
/** @copydoc nk_bilinear_f32 */
NK_PUBLIC void nk_bilinear_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                      nk_f32_t *result);
/** @copydoc nk_bilinear_f32c */
NK_PUBLIC void nk_bilinear_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                       nk_f32c_t *results);
/** @copydoc nk_mahalanobis_f32 */
NK_PUBLIC void nk_mahalanobis_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                         nk_f32_t *result);
/** @copydoc nk_bilinear_f16 */
NK_PUBLIC void nk_bilinear_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                      nk_f32_t *result);
/** @copydoc nk_bilinear_f16c */
NK_PUBLIC void nk_bilinear_f16c_serial(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                       nk_f32c_t *results);
/** @copydoc nk_mahalanobis_f16 */
NK_PUBLIC void nk_mahalanobis_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                         nk_f32_t *result);
/** @copydoc nk_bilinear_bf16 */
NK_PUBLIC void nk_bilinear_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                       nk_f32_t *result);
/** @copydoc nk_bilinear_bf16c */
NK_PUBLIC void nk_bilinear_bf16c_serial(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c, nk_size_t n,
                                        nk_f32c_t *results);
/** @copydoc nk_mahalanobis_bf16 */
NK_PUBLIC void nk_mahalanobis_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                          nk_f32_t *result);
/** @copydoc nk_bilinear_f32 */
NK_PUBLIC void nk_bilinear_f32_accurate(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                        nk_f64_t *result);
/** @copydoc nk_bilinear_f32c */
NK_PUBLIC void nk_bilinear_f32c_accurate(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                         nk_f64c_t *results);
/** @copydoc nk_mahalanobis_f32 */
NK_PUBLIC void nk_mahalanobis_f32_accurate(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                           nk_f64_t *result);
/** @copydoc nk_bilinear_f16 */
NK_PUBLIC void nk_bilinear_f16_accurate(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                        nk_f64_t *result);
/** @copydoc nk_bilinear_f16c */
NK_PUBLIC void nk_bilinear_f16c_accurate(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                         nk_f64c_t *results);
/** @copydoc nk_mahalanobis_f16 */
NK_PUBLIC void nk_mahalanobis_f16_accurate(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                           nk_f64_t *result);
/** @copydoc nk_bilinear_bf16 */
NK_PUBLIC void nk_bilinear_bf16_accurate(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                         nk_f64_t *result);
/** @copydoc nk_bilinear_bf16c */
NK_PUBLIC void nk_bilinear_bf16c_accurate(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c, nk_size_t n,
                                          nk_f64c_t *results);
/** @copydoc nk_mahalanobis_bf16 */
NK_PUBLIC void nk_mahalanobis_bf16_accurate(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                            nk_f64_t *result);
#if NK_TARGET_NEON
/** @copydoc nk_bilinear_f32 */
NK_PUBLIC void nk_bilinear_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                    nk_f32_t *result);
/** @copydoc nk_bilinear_f32c */
NK_PUBLIC void nk_bilinear_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                     nk_f32c_t *results);
/** @copydoc nk_mahalanobis_f32 */
NK_PUBLIC void nk_mahalanobis_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                       nk_f32_t *result);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
/** @copydoc nk_bilinear_f16 */
NK_PUBLIC void nk_bilinear_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                        nk_f32_t *result);
/** @copydoc nk_bilinear_f16c */
NK_PUBLIC void nk_bilinear_f16c_neonhalf(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                         nk_f32c_t *results);
/** @copydoc nk_mahalanobis_f16 */
NK_PUBLIC void nk_mahalanobis_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                           nk_f32_t *result);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
/** @copydoc nk_bilinear_bf16 */
NK_PUBLIC void nk_bilinear_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                          nk_f32_t *result);
/** @copydoc nk_bilinear_bf16c */
NK_PUBLIC void nk_bilinear_bf16c_neonbfdot(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c, nk_size_t n,
                                           nk_f32c_t *results);
/** @copydoc nk_mahalanobis_bf16 */
NK_PUBLIC void nk_mahalanobis_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                             nk_f32_t *result);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_HASWELL
/** @copydoc nk_bilinear_f16 */
NK_PUBLIC void nk_bilinear_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                       nk_f32_t *result);
/** @copydoc nk_mahalanobis_f16 */
NK_PUBLIC void nk_mahalanobis_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                          nk_f32_t *result);
/** @copydoc nk_bilinear_bf16 */
NK_PUBLIC void nk_bilinear_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                        nk_f32_t *result);
/** @copydoc nk_mahalanobis_bf16 */
NK_PUBLIC void nk_mahalanobis_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                           nk_f32_t *result);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_bilinear_f64 */
NK_PUBLIC void nk_bilinear_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                       nk_f64_t *result);
/** @copydoc nk_bilinear_f64c */
NK_PUBLIC void nk_bilinear_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                        nk_f64c_t *results);
/** @copydoc nk_mahalanobis_f64 */
NK_PUBLIC void nk_mahalanobis_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                          nk_f64_t *result);
/** @copydoc nk_bilinear_f32 */
NK_PUBLIC void nk_bilinear_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                       nk_f32_t *result);
/** @copydoc nk_bilinear_f32c */
NK_PUBLIC void nk_bilinear_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                        nk_f32c_t *results);
/** @copydoc nk_mahalanobis_f32 */
NK_PUBLIC void nk_mahalanobis_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                          nk_f32_t *result);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_GENOA
/** @copydoc nk_bilinear_bf16 */
NK_PUBLIC void nk_bilinear_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                      nk_f32_t *result);
/** @copydoc nk_bilinear_bf16c */
NK_PUBLIC void nk_bilinear_bf16c_genoa(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c, nk_size_t n,
                                       nk_f32c_t *results);
/** @copydoc nk_mahalanobis_bf16 */
NK_PUBLIC void nk_mahalanobis_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                         nk_f32_t *result);
#endif // NK_TARGET_GENOA

#if NK_TARGET_SAPPHIRE
/** @copydoc nk_bilinear_f16 */
NK_PUBLIC void nk_bilinear_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                        nk_f32_t *result);
/** @copydoc nk_bilinear_f16c */
NK_PUBLIC void nk_bilinear_f16c_sapphire(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                         nk_f32c_t *results);
/** @copydoc nk_mahalanobis_f16 */
NK_PUBLIC void nk_mahalanobis_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                           nk_f32_t *result);
#endif // NK_TARGET_SAPPHIRE

#define NK_MAKE_BILINEAR(name, input_type, accumulator_type, load_and_convert)                                   \
    NK_PUBLIC void nk_bilinear_##input_type##_##name(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                     nk_##input_type##_t const *c, nk_size_t n,                  \
                                                     nk_##accumulator_type##_t *result) {                        \
        nk_##accumulator_type##_t sum = 0, a_i, b_j, c_ij;                                                       \
        for (nk_size_t i = 0; i != n; ++i) {                                                                     \
            nk_##accumulator_type##_t cb_j = 0;                                                                  \
            load_and_convert(a + i, &a_i);                                                                       \
            for (nk_size_t j = 0; j != n; ++j) {                                                                 \
                load_and_convert(b + j, &b_j);                                                                   \
                load_and_convert(c + i * n + j, &c_ij);                                                          \
                cb_j += c_ij * b_j;                                                                              \
            }                                                                                                    \
            sum += a_i * cb_j;                                                                                   \
        }                                                                                                        \
        *result = sum;                                                                                           \
    }

#define NK_MAKE_COMPLEX_BILINEAR(name, input_type, accumulator_type, load_and_convert)                              \
    NK_PUBLIC void nk_bilinear_##input_type##_##name(                                                               \
        nk_##input_type##_t const *a_pairs, nk_##input_type##_t const *b_pairs, nk_##input_type##_t const *c_pairs, \
        nk_size_t n, nk_##accumulator_type##c_t *results) {                                                         \
        nk_##accumulator_type##_t sum_real = 0;                                                                     \
        nk_##accumulator_type##_t sum_imag = 0;                                                                     \
        nk_##accumulator_type##_t a_i_real, a_i_imag, b_j_real, b_j_imag, c_ij_real, c_ij_imag;                     \
        for (nk_size_t i = 0; i != n; ++i) {                                                                        \
            nk_##accumulator_type##_t cb_j_real = 0;                                                                \
            nk_##accumulator_type##_t cb_j_imag = 0;                                                                \
            load_and_convert(&(a_pairs + i)->real, &a_i_real);                                                      \
            load_and_convert(&(a_pairs + i)->imag, &a_i_imag);                                                      \
            for (nk_size_t j = 0; j != n; ++j) {                                                                    \
                load_and_convert(&(b_pairs + j)->real, &b_j_real);                                                  \
                load_and_convert(&(b_pairs + j)->imag, &b_j_imag);                                                  \
                load_and_convert(&(c_pairs + i * n + j)->real, &c_ij_real);                                         \
                load_and_convert(&(c_pairs + i * n + j)->imag, &c_ij_imag);                                         \
                /* Complex multiplication: (c_ij * b_j) */                                                          \
                cb_j_real += c_ij_real * b_j_real - c_ij_imag * b_j_imag;                                           \
                cb_j_imag += c_ij_real * b_j_imag + c_ij_imag * b_j_real;                                           \
            }                                                                                                       \
            /* Complex multiplication: (a_i * cb_j) */                                                              \
            sum_real += a_i_real * cb_j_real - a_i_imag * cb_j_imag;                                                \
            sum_imag += a_i_real * cb_j_imag + a_i_imag * cb_j_real;                                                \
        }                                                                                                           \
        results->real = sum_real;                                                                                   \
        results->imag = sum_imag;                                                                                   \
    }

#define NK_MAKE_MAHALANOBIS(name, input_type, accumulator_type, load_and_convert)                                   \
    NK_PUBLIC void nk_mahalanobis_##input_type##_##name(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                        nk_##input_type##_t const *c, nk_size_t n,                  \
                                                        nk_##accumulator_type##_t *result) {                        \
        nk_##accumulator_type##_t sum = 0, v_ai, v_bi, v_aj, v_bj, v_cij;                                           \
        for (nk_size_t i = 0; i != n; ++i) {                                                                        \
            nk_##accumulator_type##_t cdiff_j = 0;                                                                  \
            load_and_convert(a + i, &v_ai);                                                                         \
            load_and_convert(b + i, &v_bi);                                                                         \
            nk_##accumulator_type##_t diff_i = v_ai - v_bi;                                                         \
            for (nk_size_t j = 0; j != n; ++j) {                                                                    \
                load_and_convert(a + j, &v_aj);                                                                     \
                load_and_convert(b + j, &v_bj);                                                                     \
                load_and_convert(c + i * n + j, &v_cij);                                                            \
                nk_##accumulator_type##_t diff_j = v_aj - v_bj;                                                     \
                cdiff_j += v_cij * diff_j;                                                                          \
            }                                                                                                       \
            sum += diff_i * cdiff_j;                                                                                \
        }                                                                                                           \
        *result = nk_##accumulator_type##_sqrt_serial(sum);                                                         \
    }

NK_MAKE_BILINEAR(serial, f64, f64, nk_assign_from_to_)          // nk_bilinear_f64_serial
NK_MAKE_COMPLEX_BILINEAR(serial, f64c, f64, nk_assign_from_to_) // nk_bilinear_f64c_serial
NK_MAKE_MAHALANOBIS(serial, f64, f64, nk_assign_from_to_)       // nk_mahalanobis_f64_serial

NK_MAKE_BILINEAR(serial, f32, f32, nk_assign_from_to_)          // nk_bilinear_f32_serial
NK_MAKE_COMPLEX_BILINEAR(serial, f32c, f32, nk_assign_from_to_) // nk_bilinear_f32c_serial
NK_MAKE_MAHALANOBIS(serial, f32, f32, nk_assign_from_to_)       // nk_mahalanobis_f32_serial

NK_MAKE_BILINEAR(serial, f16, f32, nk_f16_to_f32_serial)          // nk_bilinear_f16_serial
NK_MAKE_COMPLEX_BILINEAR(serial, f16c, f32, nk_f16_to_f32_serial) // nk_bilinear_f16c_serial
NK_MAKE_MAHALANOBIS(serial, f16, f32, nk_f16_to_f32_serial)       // nk_mahalanobis_f16_serial

NK_MAKE_BILINEAR(serial, bf16, f32, nk_bf16_to_f32_serial)          // nk_bilinear_bf16_serial
NK_MAKE_COMPLEX_BILINEAR(serial, bf16c, f32, nk_bf16_to_f32_serial) // nk_bilinear_bf16c_serial
NK_MAKE_MAHALANOBIS(serial, bf16, f32, nk_bf16_to_f32_serial)       // nk_mahalanobis_bf16_serial

NK_MAKE_BILINEAR(accurate, f32, f64, nk_assign_from_to_)          // nk_bilinear_f32_accurate
NK_MAKE_COMPLEX_BILINEAR(accurate, f32c, f64, nk_assign_from_to_) // nk_bilinear_f32c_accurate
NK_MAKE_MAHALANOBIS(accurate, f32, f64, nk_assign_from_to_)       // nk_mahalanobis_f32_accurate

NK_MAKE_BILINEAR(accurate, f16, f64, nk_f16_to_f64_)          // nk_bilinear_f16_accurate
NK_MAKE_COMPLEX_BILINEAR(accurate, f16c, f64, nk_f16_to_f64_) // nk_bilinear_f16c_accurate
NK_MAKE_MAHALANOBIS(accurate, f16, f64, nk_f16_to_f64_)       // nk_mahalanobis_f16_accurate

NK_MAKE_BILINEAR(accurate, bf16, f64, nk_bf16_to_f64_)          // nk_bilinear_bf16_accurate
NK_MAKE_COMPLEX_BILINEAR(accurate, bf16c, f64, nk_bf16_to_f64_) // nk_bilinear_bf16c_accurate
NK_MAKE_MAHALANOBIS(accurate, bf16, f64, nk_bf16_to_f64_)       // nk_mahalanobis_bf16_accurate

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#endif

NK_PUBLIC void nk_bilinear_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                    nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    for (nk_size_t i = 0; i != n; ++i) {
        float32x4_t a_f32x4 = vdupq_n_f32(a[i]);
        float32x4_t cb_j_f32x4 = vdupq_n_f32(0);
        for (nk_size_t j = 0; j + 4 <= n; j += 4) {
            float32x4_t b_f32x4 = vld1q_f32(b + j);
            float32x4_t c_f32x4 = vld1q_f32(c + i * n + j);
            cb_j_f32x4 = vmlaq_f32(cb_j_f32x4, b_f32x4, c_f32x4);
        }
        sum_f32x4 = vmlaq_f32(sum_f32x4, a_f32x4, cb_j_f32x4);
    }

    // Handle the tail of every row
    nk_f64_t sum = vaddvq_f32(sum_f32x4);
    nk_size_t const tail_length = n % 4;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t a_i = a[i];
            nk_f32_t cb_j = 0;
            for (nk_size_t j = tail_start; j != n; ++j) cb_j += b[j] * c[i * n + j];
            sum += a[i] * cb_j;
        }
    }

    *result = sum;
}

NK_PUBLIC void nk_mahalanobis_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                       nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    for (nk_size_t i = 0; i != n; ++i) {
        float32x4_t diff_i_f32x4 = vdupq_n_f32(a[i] - b[i]);
        float32x4_t cdiff_j_f32x4 = vdupq_n_f32(0);
        for (nk_size_t j = 0; j + 4 <= n; j += 4) {
            float32x4_t diff_j_f32x4 = vsubq_f32(vld1q_f32(a + j), vld1q_f32(b + j));
            float32x4_t c_f32x4 = vld1q_f32(c + i * n + j);
            cdiff_j_f32x4 = vmlaq_f32(cdiff_j_f32x4, diff_j_f32x4, c_f32x4);
        }

        sum_f32x4 = vmlaq_f32(sum_f32x4, diff_i_f32x4, cdiff_j_f32x4);
    }

    // Handle the tail of every row
    nk_f64_t sum = vaddvq_f32(sum_f32x4);
    nk_size_t const tail_length = n % 4;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t diff_i = a[i] - b[i];
            nk_f32_t cdiff_j = 0;
            for (nk_size_t j = tail_start; j != n; ++j) {
                nk_f32_t diff_j = a[j] - b[j];
                cdiff_j += diff_j * c[i * n + j];
            }
            sum += diff_i * cdiff_j;
        }
    }

    *result = nk_f64_sqrt_neon(sum);
}

NK_PUBLIC void nk_bilinear_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                     nk_f32c_t *results) {
    nk_f32_t sum_real = 0;
    nk_f32_t sum_imag = 0;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32c_t a_i = a[i];
        nk_f32c_t cb_j;
        float32x4_t cb_j_real_f32x4 = vdupq_n_f32(0);
        float32x4_t cb_j_imag_f32x4 = vdupq_n_f32(0);
        for (nk_size_t j = 0; j + 4 <= n; j += 4) {
            // Unpack the input arrays into real and imaginary parts:
            float32x4x2_t b_f32x4x2 = vld2q_f32((nk_f32_t const *)(b + j));
            float32x4x2_t c_f32x4x2 = vld2q_f32((nk_f32_t const *)(c + i * n + j));
            float32x4_t b_real_f32x4 = b_f32x4x2.val[0];
            float32x4_t b_imag_f32x4 = b_f32x4x2.val[1];
            float32x4_t c_real_f32x4 = c_f32x4x2.val[0];
            float32x4_t c_imag_f32x4 = c_f32x4x2.val[1];

            // Compute the dot product:
            cb_j_real_f32x4 = vfmaq_f32(cb_j_real_f32x4, c_real_f32x4, b_real_f32x4);
            cb_j_real_f32x4 = vfmsq_f32(cb_j_real_f32x4, c_imag_f32x4, b_imag_f32x4);
            cb_j_imag_f32x4 = vfmaq_f32(cb_j_imag_f32x4, c_real_f32x4, b_imag_f32x4);
            cb_j_imag_f32x4 = vfmaq_f32(cb_j_imag_f32x4, c_imag_f32x4, b_real_f32x4);
        }
        cb_j.real = vaddvq_f32(cb_j_real_f32x4);
        cb_j.imag = vaddvq_f32(cb_j_imag_f32x4);
        sum_real += a_i.real * cb_j.real - a_i.imag * cb_j.imag;
        sum_imag += a_i.real * cb_j.imag + a_i.imag * cb_j.real;
    }

    // Handle the tail of every row
    nk_size_t const tail_length = n % 4;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32c_t a_i = a[i];
            nk_f32c_t cb_j = {0, 0};
            for (nk_size_t j = tail_start; j != n; ++j) {
                nk_f32c_t b_j = b[j];
                nk_f32c_t c_ij = c[i * n + j];
                cb_j.real += b_j.real * c_ij.real - b_j.imag * c_ij.imag;
                cb_j.imag += b_j.real * c_ij.imag + b_j.imag * c_ij.real;
            }
            sum_real += a_i.real * cb_j.real - a_i.imag * cb_j.imag;
            sum_imag += a_i.real * cb_j.imag + a_i.imag * cb_j.real;
        }
    }

    results->real = sum_real;
    results->imag = sum_imag;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

NK_PUBLIC void nk_bilinear_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                        nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    for (nk_size_t i = 0; i != n; ++i) {
        // MSVC doesn't recognize `vdup_n_f16` as a valid intrinsic
        float32x4_t a_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(vdup_n_s16(*(short const *)(a + i))));
        float32x4_t cb_j_f32x4 = vdupq_n_f32(0);
        for (nk_size_t j = 0; j + 4 <= n; j += 4) {
            float32x4_t b_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)(b + j)));
            float32x4_t c_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)(c + i * n + j)));
            cb_j_f32x4 = vmlaq_f32(cb_j_f32x4, b_f32x4, c_f32x4);
        }
        sum_f32x4 = vmlaq_f32(sum_f32x4, a_f32x4, cb_j_f32x4);
    }

    // Handle the tail of every row
    nk_f64_t sum = vaddvq_f32(sum_f32x4);
    nk_size_t const tail_length = n % 4;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            float32x4_t a_i_f32x4, b_f32x4, c_f32x4;
            nk_partial_load_f16x4_to_f32x4_neonhalf_(a + i, 1, &a_i_f32x4);
            nk_partial_load_f16x4_to_f32x4_neonhalf_(b + tail_start, tail_length, &b_f32x4);
            nk_partial_load_f16x4_to_f32x4_neonhalf_(c + i * n + tail_start, tail_length, &c_f32x4);
            nk_f32_t a_i = vaddvq_f32(a_i_f32x4);
            nk_f32_t cb_j = vaddvq_f32(vmulq_f32(b_f32x4, c_f32x4));
            sum += a_i * cb_j;
        }
    }

    *result = sum;
}

NK_PUBLIC void nk_mahalanobis_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                           nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    for (nk_size_t i = 0; i != n; ++i) {
        // MSVC doesn't recognize `vdup_n_f16` as a valid intrinsic
        float32x4_t a_i_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(vdup_n_s16(*(short const *)(a + i))));
        float32x4_t b_i_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(vdup_n_s16(*(short const *)(b + i))));
        float32x4_t diff_i_f32x4 = vsubq_f32(a_i_f32x4, b_i_f32x4);
        float32x4_t cdiff_j_f32x4 = vdupq_n_f32(0);
        for (nk_size_t j = 0; j + 4 <= n; j += 4) {
            float32x4_t a_j_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)(a + j)));
            float32x4_t b_j_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)(b + j)));
            float32x4_t diff_j_f32x4 = vsubq_f32(a_j_f32x4, b_j_f32x4);
            float32x4_t c_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)(c + i * n + j)));
            cdiff_j_f32x4 = vmlaq_f32(cdiff_j_f32x4, diff_j_f32x4, c_f32x4);
        }
        sum_f32x4 = vmlaq_f32(sum_f32x4, diff_i_f32x4, cdiff_j_f32x4);
    }

    // Handle the tail of every row
    nk_f32_t sum = vaddvq_f32(sum_f32x4);
    nk_size_t const tail_length = n % 4;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            float32x4_t a_i_f32x4, b_i_f32x4, a_j_f32x4, b_j_f32x4, c_f32x4;
            nk_partial_load_f16x4_to_f32x4_neonhalf_(a + i, 1, &a_i_f32x4);
            nk_partial_load_f16x4_to_f32x4_neonhalf_(b + i, 1, &b_i_f32x4);
            nk_f32_t a_i = vaddvq_f32(a_i_f32x4);
            nk_f32_t b_i = vaddvq_f32(b_i_f32x4);
            nk_f32_t diff_i = a_i - b_i;
            nk_partial_load_f16x4_to_f32x4_neonhalf_(a + tail_start, tail_length, &a_j_f32x4);
            nk_partial_load_f16x4_to_f32x4_neonhalf_(b + tail_start, tail_length, &b_j_f32x4);
            float32x4_t diff_j_f32x4 = vsubq_f32(a_j_f32x4, b_j_f32x4);
            nk_partial_load_f16x4_to_f32x4_neonhalf_(c + i * n + tail_start, tail_length, &c_f32x4);
            nk_f32_t cdiff_j = vaddvq_f32(vmulq_f32(diff_j_f32x4, c_f32x4));
            sum += diff_i * cdiff_j;
        }
    }

    *result = nk_f32_sqrt_neon(sum);
}

NK_INTERNAL float16x8x2_t nk_partial_load_f16x8x2_neon_(nk_f16c_t const *x, nk_size_t n) {
    union {
        float16x8x2_t vecs;
        nk_f16_t scalars[2][8];
    } result;
    nk_size_t i = 0;
    for (; i < n; ++i) result.scalars[0][i] = x[i].real, result.scalars[1][i] = x[i].imag;
    for (; i < 8; ++i) result.scalars[0][i] = 0, result.scalars[1][i] = 0;
    return result.vecs;
}

NK_PUBLIC void nk_bilinear_f16c_neonhalf(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                         nk_f32c_t *results) {
    nk_f32_t sum_real = 0;
    nk_f32_t sum_imag = 0;
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32c_t a_i;
        nk_f16_to_f32_neon(&a[i].real, &a_i.real);
        nk_f16_to_f32_neon(&a[i].imag, &a_i.imag);
        float16x8_t cb_j_real_f16x8 = vdupq_n_f16(0);
        float16x8_t cb_j_imag_f16x8 = vdupq_n_f16(0);
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            // Unpack the input arrays into real and imaginary parts:
            float16x8x2_t b_f16x8x2 = vld2q_f16((float16_t const *)(b + j));
            float16x8x2_t c_f16x8x2 = vld2q_f16((float16_t const *)(c + i * n + j));
            float16x8_t b_real_f16x8 = b_f16x8x2.val[0];
            float16x8_t b_imag_f16x8 = b_f16x8x2.val[1];
            float16x8_t c_real_f16x8 = c_f16x8x2.val[0];
            float16x8_t c_imag_f16x8 = c_f16x8x2.val[1];

            // Compute the dot product:
            cb_j_real_f16x8 = vfmaq_f16(cb_j_real_f16x8, c_real_f16x8, b_real_f16x8);
            cb_j_real_f16x8 = vfmsq_f16(cb_j_real_f16x8, c_imag_f16x8, b_imag_f16x8);
            cb_j_imag_f16x8 = vfmaq_f16(cb_j_imag_f16x8, c_real_f16x8, b_imag_f16x8);
            cb_j_imag_f16x8 = vfmaq_f16(cb_j_imag_f16x8, c_imag_f16x8, b_real_f16x8);
        }
        // Handle row tails
        if (tail_length) {
            // Unpack the input arrays into real and imaginary parts:
            float16x8x2_t b_f16x8x2 = nk_partial_load_f16x8x2_neon_(b + tail_start, tail_length);
            float16x8x2_t c_f16x8x2 = nk_partial_load_f16x8x2_neon_(c + i * n + tail_start, tail_length);
            float16x8_t b_real_f16x8 = b_f16x8x2.val[0];
            float16x8_t b_imag_f16x8 = b_f16x8x2.val[1];
            float16x8_t c_real_f16x8 = c_f16x8x2.val[0];
            float16x8_t c_imag_f16x8 = c_f16x8x2.val[1];

            // Compute the dot product:
            cb_j_real_f16x8 = vfmaq_f16(cb_j_real_f16x8, c_real_f16x8, b_real_f16x8);
            cb_j_real_f16x8 = vfmsq_f16(cb_j_real_f16x8, c_imag_f16x8, b_imag_f16x8);
            cb_j_imag_f16x8 = vfmaq_f16(cb_j_imag_f16x8, c_real_f16x8, b_imag_f16x8);
            cb_j_imag_f16x8 = vfmaq_f16(cb_j_imag_f16x8, c_imag_f16x8, b_real_f16x8);
        }

        nk_f32c_t cb_j;
        cb_j.real = nk_reduce_add_f16x8_neonhalf_(cb_j_real_f16x8);
        cb_j.imag = nk_reduce_add_f16x8_neonhalf_(cb_j_imag_f16x8);
        sum_real += a_i.real * cb_j.real - a_i.imag * cb_j.imag;
        sum_imag += a_i.real * cb_j.imag + a_i.imag * cb_j.real;
    }

    results->real = sum_real;
    results->imag = sum_imag;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#endif

NK_PUBLIC void nk_bilinear_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                          nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_f32;
        nk_bf16_to_f32_serial(a + i, &a_f32);
        float32x4_t a_f32x4 = vdupq_n_f32(a_f32);
        float32x4_t cb_j_f32x4 = vdupq_n_f32(0);
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            bfloat16x8_t b_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)(b + j));
            bfloat16x8_t c_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)(c + i * n + j));
            cb_j_f32x4 = vbfdotq_f32(cb_j_f32x4, b_bf16x8, c_bf16x8);
        }
        sum_f32x4 = vmlaq_f32(sum_f32x4, a_f32x4, cb_j_f32x4);
    }

    // Handle the tail of every row
    nk_f64_t sum = vaddvq_f32(sum_f32x4);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t a_i;
            nk_bf16_to_f32_serial(a + i, &a_i);
            nk_b128_vec_t b_vec, c_vec;
            nk_partial_load_b16x8_neon_(b + tail_start, tail_length, &b_vec);
            nk_partial_load_b16x8_neon_(c + i * n + tail_start, tail_length, &c_vec);
            bfloat16x8_t b_bf16x8 = vreinterpretq_bf16_u16(b_vec.u16x8);
            bfloat16x8_t c_bf16x8 = vreinterpretq_bf16_u16(c_vec.u16x8);
            nk_f32_t cb_j = vaddvq_f32(vbfdotq_f32(vdupq_n_f32(0), b_bf16x8, c_bf16x8));
            sum += a_i * cb_j;
        }
    }

    *result = sum;
}

NK_PUBLIC void nk_mahalanobis_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                             nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_i, b_i;
        nk_bf16_to_f32_serial(a + i, &a_i);
        nk_bf16_to_f32_serial(b + i, &b_i);
        float32x4_t diff_i_f32x4 = vdupq_n_f32(a_i - b_i);
        float32x4_t cdiff_j_f32x4 = vdupq_n_f32(0);
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            bfloat16x8_t a_j_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)(a + j));
            bfloat16x8_t b_j_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)(b + j));

            // Arm NEON does not have a native subtraction instruction for `bf16`,
            // so we need to convert to `f32` first, subtract, and only then get back to `bf16`
            // for multiplication.
            float32x4_t a_j_high_f32x4 = vcvt_f32_bf16(vget_high_bf16(a_j_bf16x8));
            float32x4_t a_j_low_f32x4 = vcvt_f32_bf16(vget_low_bf16(a_j_bf16x8));
            float32x4_t b_j_high_f32x4 = vcvt_f32_bf16(vget_high_bf16(b_j_bf16x8));
            float32x4_t b_j_low_f32x4 = vcvt_f32_bf16(vget_low_bf16(b_j_bf16x8));
            float32x4_t diff_j_high_f32x4 = vsubq_f32(a_j_high_f32x4, b_j_high_f32x4);
            float32x4_t diff_j_low_f32x4 = vsubq_f32(a_j_low_f32x4, b_j_low_f32x4);
            bfloat16x8_t diff_j_bf16x8 = vcombine_bf16(vcvt_bf16_f32(diff_j_low_f32x4),
                                                       vcvt_bf16_f32(diff_j_high_f32x4));

            bfloat16x8_t c_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)(c + i * n + j));
            cdiff_j_f32x4 = vbfdotq_f32(cdiff_j_f32x4, diff_j_bf16x8, c_bf16x8);
        }
        sum_f32x4 = vmlaq_f32(sum_f32x4, diff_i_f32x4, cdiff_j_f32x4);
    }

    // Handle the tail of every row
    nk_f32_t sum = vaddvq_f32(sum_f32x4);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t a_i, b_i;
            nk_bf16_to_f32_serial(a + i, &a_i);
            nk_bf16_to_f32_serial(b + i, &b_i);
            nk_f32_t diff_i = a_i - b_i;
            nk_b128_vec_t a_j_vec, b_j_vec, c_vec;
            nk_partial_load_b16x8_neon_(a + tail_start, tail_length, &a_j_vec);
            nk_partial_load_b16x8_neon_(b + tail_start, tail_length, &b_j_vec);
            bfloat16x8_t a_j_bf16x8 = vreinterpretq_bf16_u16(a_j_vec.u16x8);
            bfloat16x8_t b_j_bf16x8 = vreinterpretq_bf16_u16(b_j_vec.u16x8);

            // Again, upcast for subtraction
            float32x4_t a_j_high_f32x4 = vcvt_f32_bf16(vget_high_bf16(a_j_bf16x8));
            float32x4_t a_j_low_f32x4 = vcvt_f32_bf16(vget_low_bf16(a_j_bf16x8));
            float32x4_t b_j_high_f32x4 = vcvt_f32_bf16(vget_high_bf16(b_j_bf16x8));
            float32x4_t b_j_low_f32x4 = vcvt_f32_bf16(vget_low_bf16(b_j_bf16x8));
            float32x4_t diff_j_high_f32x4 = vsubq_f32(a_j_high_f32x4, b_j_high_f32x4);
            float32x4_t diff_j_low_f32x4 = vsubq_f32(a_j_low_f32x4, b_j_low_f32x4);
            bfloat16x8_t diff_j_bf16x8 = vcombine_bf16(vcvt_bf16_f32(diff_j_low_f32x4),
                                                       vcvt_bf16_f32(diff_j_high_f32x4));

            nk_partial_load_b16x8_neon_(c + i * n + tail_start, tail_length, &c_vec);
            bfloat16x8_t c_bf16x8 = vreinterpretq_bf16_u16(c_vec.u16x8);
            nk_f32_t cdiff_j = vaddvq_f32(vbfdotq_f32(vdupq_n_f32(0), diff_j_bf16x8, c_bf16x8));
            sum += diff_i * cdiff_j;
        }
    }

    *result = nk_f32_sqrt_neon(sum);
}

NK_INTERNAL int16x4x2_t nk_partial_load_bf16x4x2_neon_(nk_bf16c_t const *x, nk_size_t n) {
    union {
        int16x4x2_t vec;
        nk_bf16_t scalars[2][4];
    } result;
    nk_size_t i = 0;
    for (; i < n; ++i) result.scalars[0][i] = x[i].real, result.scalars[1][i] = x[i].imag;
    for (; i < 4; ++i) result.scalars[1][i] = 0, result.scalars[1][i] = 0;
    return result.vec;
}

NK_PUBLIC void nk_bilinear_bf16c_neonbfdot(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c, nk_size_t n,
                                           nk_f32c_t *results) {
    nk_f32_t sum_real = 0;
    nk_f32_t sum_imag = 0;
    nk_size_t const tail_length = n % 4;
    nk_size_t const tail_start = n - tail_length;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32c_t a_i;
        nk_bf16_to_f32_serial(&a[i].real, &a_i.real);
        nk_bf16_to_f32_serial(&a[i].imag, &a_i.imag);
        // A nicer approach is to use `bf16` arithmetic for the dot product, but that requires
        // FMLA extensions available on Arm v8.3 and later. That we can also process 16 entries
        // at once. That's how the original implementation worked, but compiling it was a nightmare :)
        float32x4_t cb_j_real_f32x4 = vdupq_n_f32(0);
        float32x4_t cb_j_imag_f32x4 = vdupq_n_f32(0);
        for (nk_size_t j = 0; j + 4 <= n; j += 4) {
            // Unpack the input arrays into real and imaginary parts.
            // MSVC sadly doesn't recognize the `vld2_bf16`, so we load the  data as signed
            // integers of the same size and reinterpret with `vreinterpret_bf16_s16` afterwards.
            int16x4x2_t b_i16x4x2 = vld2_s16((short const *)(b + j));
            int16x4x2_t c_i16x4x2 = vld2_s16((short const *)(c + i * n + j));
            float32x4_t b_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[0]));
            float32x4_t b_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[1]));
            float32x4_t c_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(c_i16x4x2.val[0]));
            float32x4_t c_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(c_i16x4x2.val[1]));

            // Compute the dot product:
            cb_j_real_f32x4 = vfmaq_f32(cb_j_real_f32x4, c_real_f32x4, b_real_f32x4);
            cb_j_real_f32x4 = vfmsq_f32(cb_j_real_f32x4, c_imag_f32x4, b_imag_f32x4);
            cb_j_imag_f32x4 = vfmaq_f32(cb_j_imag_f32x4, c_real_f32x4, b_imag_f32x4);
            cb_j_imag_f32x4 = vfmaq_f32(cb_j_imag_f32x4, c_imag_f32x4, b_real_f32x4);
        }
        // Handle row tails
        if (tail_length) {
            // Unpack the input arrays into real and imaginary parts:
            int16x4x2_t b_i16x4x2 = nk_partial_load_bf16x4x2_neon_(b + tail_start, tail_length);
            int16x4x2_t c_i16x4x2 = nk_partial_load_bf16x4x2_neon_(c + i * n + tail_start, tail_length);
            float32x4_t b_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[0]));
            float32x4_t b_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[1]));
            float32x4_t c_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(c_i16x4x2.val[0]));
            float32x4_t c_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(c_i16x4x2.val[1]));

            // Compute the dot product:
            cb_j_real_f32x4 = vfmaq_f32(cb_j_real_f32x4, c_real_f32x4, b_real_f32x4);
            cb_j_real_f32x4 = vfmsq_f32(cb_j_real_f32x4, c_imag_f32x4, b_imag_f32x4);
            cb_j_imag_f32x4 = vfmaq_f32(cb_j_imag_f32x4, c_real_f32x4, b_imag_f32x4);
            cb_j_imag_f32x4 = vfmaq_f32(cb_j_imag_f32x4, c_imag_f32x4, b_real_f32x4);
        }

        nk_f32c_t cb_j;
        cb_j.real = vaddvq_f32(cb_j_real_f32x4);
        cb_j.imag = vaddvq_f32(cb_j_imag_f32x4);
        sum_real += a_i.real * cb_j.real - a_i.imag * cb_j.imag;
        sum_imag += a_i.real * cb_j.imag + a_i.imag * cb_j.real;
    }

    results->real = sum_real;
    results->imag = sum_imag;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEONBFDOT

#endif // NK_TARGET_ARM_

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_bilinear_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                       nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (nk_size_t i = 0; i != n; ++i) {
        __m256 a_f32x8 = _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(a + i)));
        __m256 cb_j_f32x8 = _mm256_setzero_ps();
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 b_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(b + j)));
            __m256 c_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(c + i * n + j)));
            cb_j_f32x8 = _mm256_fmadd_ps(b_f32x8, c_f32x8, cb_j_f32x8);
        }
        sum_f32x8 = _mm256_fmadd_ps(a_f32x8, cb_j_f32x8, sum_f32x8);
    }

    // Handle the tail of every row
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t a_i = _mm256_cvtss_f32(_mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(a + i))));
            __m256 b_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(b + tail_start, tail_length);
            __m256 c_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(c + i * n + tail_start, tail_length);
            nk_f32_t cb_j = nk_reduce_add_f32x8_haswell_(_mm256_mul_ps(b_f32x8, c_f32x8));
            sum += a_i * cb_j;
        }
    }

    *result = sum;
}

NK_PUBLIC void nk_mahalanobis_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                          nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (nk_size_t i = 0; i != n; ++i) {
        __m256 diff_i_f32x8 = _mm256_sub_ps(                          //
            _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(a + i))), //
            _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(b + i))));
        __m256 cdiff_j_f32x8 = _mm256_setzero_ps();
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 diff_j_f32x8 = _mm256_sub_ps( //
                _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(a + j))),
                _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(b + j))));
            __m256 c_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(c + i * n + j)));
            cdiff_j_f32x8 = _mm256_fmadd_ps(diff_j_f32x8, c_f32x8, cdiff_j_f32x8);
        }
        sum_f32x8 = _mm256_fmadd_ps(diff_i_f32x8, cdiff_j_f32x8, sum_f32x8);
    }

    // Handle the tail of every row
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t diff_i = _mm256_cvtss_f32(_mm256_sub_ps(             //
                _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(a + i))), //
                _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(b + i)))));
            __m256 diff_j_f32x8 = _mm256_sub_ps( //
                nk_partial_load_f16x8_to_f32x8_haswell_(a + tail_start, tail_length),
                nk_partial_load_f16x8_to_f32x8_haswell_(b + tail_start, tail_length));
            __m256 c_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(c + i * n + tail_start, tail_length);
            nk_f32_t cdiff_j = nk_reduce_add_f32x8_haswell_(_mm256_mul_ps(diff_j_f32x8, c_f32x8));
            sum += diff_i * cdiff_j;
        }
    }

    *result = nk_sqrt_f32_haswell_(sum);
}

NK_PUBLIC void nk_bilinear_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                        nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (nk_size_t i = 0; i != n; ++i) {
        // The `nk_bf16_to_f32_serial` is cheaper than `nk_bf16x8_to_f32x8_haswell_`
        nk_f32_t a_f32;
        nk_bf16_to_f32_serial(a + i, &a_f32);
        __m256 a_f32x8 = _mm256_set1_ps(a_f32);
        __m256 cb_j_f32x8 = _mm256_setzero_ps();
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 b_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(b + j)));
            __m256 c_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(c + i * n + j)));
            cb_j_f32x8 = _mm256_fmadd_ps(b_f32x8, c_f32x8, cb_j_f32x8);
        }
        sum_f32x8 = _mm256_fmadd_ps(a_f32x8, cb_j_f32x8, sum_f32x8);
    }

    // Handle the tail of every row
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t a_i;
            nk_bf16_to_f32_serial(a + i, &a_i);
            __m256 b_f32x8 = nk_partial_load_bf16x8_to_f32x8_haswell_(b + tail_start, tail_length);
            __m256 c_f32x8 = nk_partial_load_bf16x8_to_f32x8_haswell_(c + i * n + tail_start, tail_length);
            nk_f32_t cb_j = nk_reduce_add_f32x8_haswell_(_mm256_mul_ps(b_f32x8, c_f32x8));
            sum += a_i * cb_j;
        }
    }

    *result = sum;
}

NK_PUBLIC void nk_mahalanobis_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                           nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_i, b_i;
        nk_bf16_to_f32_serial(a + i, &a_i);
        nk_bf16_to_f32_serial(b + i, &b_i);
        __m256 diff_i_f32x8 = _mm256_sub_ps( //
            _mm256_set1_ps(a_i),             //
            _mm256_set1_ps(b_i));
        __m256 cdiff_j_f32x8 = _mm256_setzero_ps();
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 diff_j_f32x8 = _mm256_sub_ps(                                        //
                nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(a + j))), //
                nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(b + j))));
            __m256 c_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(c + i * n + j)));
            cdiff_j_f32x8 = _mm256_fmadd_ps(diff_j_f32x8, c_f32x8, cdiff_j_f32x8);
        }
        sum_f32x8 = _mm256_fmadd_ps(diff_i_f32x8, cdiff_j_f32x8, sum_f32x8);
    }

    // Handle the tail of every row
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t a_i, b_i;
            nk_bf16_to_f32_serial(a + i, &a_i);
            nk_bf16_to_f32_serial(b + i, &b_i);
            nk_f32_t diff_i = a_i - b_i;
            __m256 diff_j_f32x8 = _mm256_sub_ps( //
                nk_partial_load_bf16x8_to_f32x8_haswell_(a + tail_start, tail_length),
                nk_partial_load_bf16x8_to_f32x8_haswell_(b + tail_start, tail_length));
            __m256 c_f32x8 = nk_partial_load_bf16x8_to_f32x8_haswell_(c + i * n + tail_start, tail_length);
            nk_f32_t cdiff_j = nk_reduce_add_f32x8_haswell_(_mm256_mul_ps(diff_j_f32x8, c_f32x8));
            sum += diff_i * cdiff_j;
        }
    }

    *result = nk_sqrt_f32_haswell_(sum);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_bilinear_f32_skylake_under16unrolled(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c,
                                                       nk_size_t n, nk_f32_t *result) {
    // The goal of this optimization is to avoid horizontal accumulation of the cb_j sums
    // until the very end of the computation.
    __mmask16 const mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
    __m512 const b_f32x16 = _mm512_maskz_loadu_ps(mask, b);

    __m512 cb_j1_f32x16 = _mm512_setzero_ps();
    __m512 cb_j2_f32x16 = _mm512_setzero_ps();
    __m512 cb_j3_f32x16 = _mm512_setzero_ps();
    __m512 cb_j4_f32x16 = _mm512_setzero_ps();

    // Unroll the loop to process 4x ZMM registers at a time.
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        cb_j1_f32x16 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, c + n * (i + 0)),
                                       _mm512_mul_ps(b_f32x16, _mm512_set1_ps(a[i + 0])), cb_j1_f32x16);
        cb_j2_f32x16 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, c + n * (i + 1)),
                                       _mm512_mul_ps(b_f32x16, _mm512_set1_ps(a[i + 1])), cb_j2_f32x16);
        cb_j3_f32x16 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, c + n * (i + 2)),
                                       _mm512_mul_ps(b_f32x16, _mm512_set1_ps(a[i + 2])), cb_j3_f32x16);
        cb_j4_f32x16 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, c + n * (i + 3)),
                                       _mm512_mul_ps(b_f32x16, _mm512_set1_ps(a[i + 3])), cb_j4_f32x16);
    }

    if (i + 0 < n)
        cb_j1_f32x16 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, c + n * (i + 0)),
                                       _mm512_mul_ps(b_f32x16, _mm512_set1_ps(a[i + 0])), cb_j1_f32x16);
    if (i + 1 < n)
        cb_j2_f32x16 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, c + n * (i + 1)),
                                       _mm512_mul_ps(b_f32x16, _mm512_set1_ps(a[i + 1])), cb_j2_f32x16);
    if (i + 2 < n)
        cb_j3_f32x16 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, c + n * (i + 2)),
                                       _mm512_mul_ps(b_f32x16, _mm512_set1_ps(a[i + 2])), cb_j3_f32x16);
    if (i + 3 < n)
        cb_j4_f32x16 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, c + n * (i + 3)),
                                       _mm512_mul_ps(b_f32x16, _mm512_set1_ps(a[i + 3])), cb_j4_f32x16);

    // Combine cb_j sums
    __m512 sum_f32x16 = _mm512_add_ps(             //
        _mm512_add_ps(cb_j1_f32x16, cb_j2_f32x16), //
        _mm512_add_ps(cb_j3_f32x16, cb_j4_f32x16));
    *result = _mm512_reduce_add_ps(sum_f32x16);
}

NK_PUBLIC void nk_bilinear_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                       nk_f32_t *result) {

    // On modern x86 CPUs we have enough register space to load fairly large matrices with up to 16 cells
    // per row and 16 rows at a time, still keeping enough register space for temporaries.
    if (n <= 16) {
        nk_bilinear_f32_skylake_under16unrolled(a, b, c, n, result);
        return;
    }

    // Default case for arbitrary size `n`
    nk_size_t const tail_length = n % 16;
    nk_size_t const tail_start = n - tail_length;
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __mmask16 const tail_mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail_length);

    for (nk_size_t i = 0; i != n; ++i) {
        __m512 a_f32x16 = _mm512_set1_ps(a[i]);
        __m512 cb_j_f32x16 = _mm512_setzero_ps();
        __m512 b_f32x16, c_f32x16;
        nk_size_t j = 0;

    nk_bilinear_f32_skylake_cycle:
        if (j + 16 <= n) {
            b_f32x16 = _mm512_loadu_ps(b + j);
            c_f32x16 = _mm512_loadu_ps(c + i * n + j);
        }
        else {
            b_f32x16 = _mm512_maskz_loadu_ps(tail_mask, b + tail_start);
            c_f32x16 = _mm512_maskz_loadu_ps(tail_mask, c + i * n + tail_start);
        }
        cb_j_f32x16 = _mm512_fmadd_ps(b_f32x16, c_f32x16, cb_j_f32x16);
        j += 16;
        if (j < n) goto nk_bilinear_f32_skylake_cycle;
        sum_f32x16 = _mm512_fmadd_ps(a_f32x16, cb_j_f32x16, sum_f32x16);
    }

    *result = _mm512_reduce_add_ps(sum_f32x16);
}

NK_PUBLIC void nk_mahalanobis_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                          nk_f32_t *result) {
    nk_size_t const tail_length = n % 16;
    nk_size_t const tail_start = n - tail_length;
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __mmask16 const tail_mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail_length);

    for (nk_size_t i = 0; i != n; ++i) {
        __m512 diff_i_f32x16 = _mm512_set1_ps(a[i] - b[i]);
        __m512 cdiff_j_f32x16 = _mm512_setzero_ps(), cdiff_j_bot_f32x16 = _mm512_setzero_ps();
        __m512 a_j_f32x16, b_j_f32x16, diff_j_f32x16, c_f32x16;
        nk_size_t j = 0;

        // The nested loop is cleaner to implement with a `goto` in this case:
    nk_bilinear_f32_skylake_cycle:
        if (j + 16 <= n) {
            a_j_f32x16 = _mm512_loadu_ps(a + j);
            b_j_f32x16 = _mm512_loadu_ps(b + j);
            c_f32x16 = _mm512_loadu_ps(c + i * n + j);
        }
        else {
            a_j_f32x16 = _mm512_maskz_loadu_ps(tail_mask, a + tail_start);
            b_j_f32x16 = _mm512_maskz_loadu_ps(tail_mask, b + tail_start);
            c_f32x16 = _mm512_maskz_loadu_ps(tail_mask, c + i * n + tail_start);
        }
        diff_j_f32x16 = _mm512_sub_ps(a_j_f32x16, b_j_f32x16);
        cdiff_j_f32x16 = _mm512_fmadd_ps(diff_j_f32x16, c_f32x16, cdiff_j_f32x16);
        j += 16;
        if (j < n) goto nk_bilinear_f32_skylake_cycle;
        sum_f32x16 = _mm512_fmadd_ps(diff_i_f32x16, cdiff_j_f32x16, sum_f32x16);
    }

    *result = nk_sqrt_f64_haswell_(_mm512_reduce_add_ps(sum_f32x16));
}

NK_PUBLIC void nk_bilinear_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                        nk_f32c_t *results) {

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_i32x16 = _mm512_set1_epi64(0x8000000000000000);

    // Default case for arbitrary size `n`
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    __mmask16 const tail_mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail_length * 2);
    nk_f32_t sum_real = 0;
    nk_f32_t sum_imag = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t const a_i_real = a[i].real;
        nk_f32_t const a_i_imag = a[i].imag;
        __m512 cb_j_real_f32x16 = _mm512_setzero_ps();
        __m512 cb_j_imag_f32x16 = _mm512_setzero_ps();
        __m512 b_f32x16, c_f32x16;
        nk_size_t j = 0;

    nk_bilinear_f32c_skylake_cycle:
        if (j + 8 <= n) {
            b_f32x16 = _mm512_loadu_ps((nk_f32_t const *)(b + j));
            c_f32x16 = _mm512_loadu_ps((nk_f32_t const *)(c + i * n + j));
        }
        else {
            b_f32x16 = _mm512_maskz_loadu_ps(tail_mask, (nk_f32_t const *)(b + tail_start));
            c_f32x16 = _mm512_maskz_loadu_ps(tail_mask, (nk_f32_t const *)(c + i * n + tail_start));
        }
        // The real part of the product: b.real * c.real - b.imag * c.imag.
        // The subtraction will be performed later with a sign flip.
        cb_j_real_f32x16 = _mm512_fmadd_ps(c_f32x16, b_f32x16, cb_j_real_f32x16);
        // The imaginary part of the product: b.real * c.imag + b.imag * c.real.
        // Swap the imaginary and real parts of `c` before multiplication:
        c_f32x16 = _mm512_permute_ps(c_f32x16, 0xB1); //? Swap adjacent entries within each pair
        cb_j_imag_f32x16 = _mm512_fmadd_ps(c_f32x16, b_f32x16, cb_j_imag_f32x16);
        j += 8;
        if (j < n) goto nk_bilinear_f32c_skylake_cycle;
        // Flip the sign bit in every second scalar before accumulation:
        cb_j_real_f32x16 = _mm512_castsi512_ps(
            _mm512_xor_si512(_mm512_castps_si512(cb_j_real_f32x16), sign_flip_i32x16));
        // Horizontal sums are the expensive part of the computation:
        nk_f32_t const cb_j_real = _mm512_reduce_add_ps(cb_j_real_f32x16);
        nk_f32_t const cb_j_imag = _mm512_reduce_add_ps(cb_j_imag_f32x16);
        sum_real += a_i_real * cb_j_real - a_i_imag * cb_j_imag;
        sum_imag += a_i_real * cb_j_imag + a_i_imag * cb_j_real;
    }

    // Reduce horizontal sums:
    results->real = sum_real;
    results->imag = sum_imag;
}

NK_PUBLIC void nk_bilinear_f64_skylake_under8unrolled(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c,
                                                      nk_size_t n, nk_f64_t *result) {

    // The goal of this optimization is to avoid horizontal accumulation of the cb_j sums
    // until the very end of the computation.
    __mmask8 const row_mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
    __m512d const b_f64x8 = _mm512_maskz_loadu_pd(row_mask, b);

    __m512d cb_j1_f64x8 = _mm512_setzero_pd();
    __m512d cb_j2_f64x8 = _mm512_setzero_pd();
    __m512d cb_j3_f64x8 = _mm512_setzero_pd();
    __m512d cb_j4_f64x8 = _mm512_setzero_pd();

    if (n > 0)
        cb_j1_f64x8 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(row_mask, c + n * 0),
                                      _mm512_mul_pd(b_f64x8, _mm512_set1_pd(a[0])), cb_j1_f64x8);
    if (n > 1)
        cb_j2_f64x8 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(row_mask, c + n * 1),
                                      _mm512_mul_pd(b_f64x8, _mm512_set1_pd(a[1])), cb_j2_f64x8);
    if (n > 2)
        cb_j3_f64x8 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(row_mask, c + n * 2),
                                      _mm512_mul_pd(b_f64x8, _mm512_set1_pd(a[2])), cb_j3_f64x8);
    if (n > 3)
        cb_j4_f64x8 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(row_mask, c + n * 3),
                                      _mm512_mul_pd(b_f64x8, _mm512_set1_pd(a[3])), cb_j4_f64x8);

    if (n > 4)
        cb_j1_f64x8 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(row_mask, c + n * 4),
                                      _mm512_mul_pd(b_f64x8, _mm512_set1_pd(a[4])), cb_j1_f64x8);
    if (n > 5)
        cb_j2_f64x8 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(row_mask, c + n * 5),
                                      _mm512_mul_pd(b_f64x8, _mm512_set1_pd(a[5])), cb_j2_f64x8);
    if (n > 6)
        cb_j3_f64x8 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(row_mask, c + n * 6),
                                      _mm512_mul_pd(b_f64x8, _mm512_set1_pd(a[6])), cb_j3_f64x8);
    if (n > 7)
        cb_j4_f64x8 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(row_mask, c + n * 7),
                                      _mm512_mul_pd(b_f64x8, _mm512_set1_pd(a[7])), cb_j4_f64x8);

    // Combine cb_j sums
    __m512d sum_f64x8 = _mm512_add_pd(           //
        _mm512_add_pd(cb_j1_f64x8, cb_j2_f64x8), //
        _mm512_add_pd(cb_j3_f64x8, cb_j4_f64x8));
    *result = _mm512_reduce_add_pd(sum_f64x8);
}

NK_PUBLIC void nk_bilinear_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                       nk_f64_t *result) {

    // On modern x86 CPUs we have enough register space to load fairly large matrices with up to 16 cells
    // per row and 8 rows at a time, still keeping enough register space for temporaries.
    if (n <= 8) {
        nk_bilinear_f64_skylake_under8unrolled(a, b, c, n, result);
        return;
    }

    // Default case for arbitrary size `n`
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __mmask8 const tail_mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, tail_length);

    for (nk_size_t i = 0; i != n; ++i) {
        __m512d a_f64x8 = _mm512_set1_pd(a[i]);
        __m512d cb_j_f64x8 = _mm512_setzero_pd();
        __m512d b_f64x8, c_f64x8;
        nk_size_t j = 0;

    nk_bilinear_f64_skylake_cycle:
        if (j + 8 <= n) {
            b_f64x8 = _mm512_loadu_pd(b + j);
            c_f64x8 = _mm512_loadu_pd(c + i * n + j);
        }
        else {
            b_f64x8 = _mm512_maskz_loadu_pd(tail_mask, b + tail_start);
            c_f64x8 = _mm512_maskz_loadu_pd(tail_mask, c + i * n + tail_start);
        }
        cb_j_f64x8 = _mm512_fmadd_pd(b_f64x8, c_f64x8, cb_j_f64x8);
        j += 8;
        if (j < n) goto nk_bilinear_f64_skylake_cycle;
        sum_f64x8 = _mm512_fmadd_pd(a_f64x8, cb_j_f64x8, sum_f64x8);
    }

    *result = _mm512_reduce_add_pd(sum_f64x8);
}

NK_PUBLIC void nk_mahalanobis_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                          nk_f64_t *result) {
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    __mmask8 const tail_mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, tail_length);
    __m512d sum_f64x8 = _mm512_setzero_pd();

    for (nk_size_t i = 0; i != n; ++i) {
        __m512d diff_i_f64x8 = _mm512_set1_pd(a[i] - b[i]);
        __m512d cdiff_j_f64x8 = _mm512_setzero_pd();
        __m512d a_j_f64x8, b_j_f64x8, diff_j_f64x8, c_f64x8;
        nk_size_t j = 0;

        // The nested loop is cleaner to implement with a `goto` in this case:
    nk_bilinear_f64_skylake_cycle:
        if (j + 8 <= n) {
            a_j_f64x8 = _mm512_loadu_pd(a + j);
            b_j_f64x8 = _mm512_loadu_pd(b + j);
            c_f64x8 = _mm512_loadu_pd(c + i * n + j);
        }
        else {
            a_j_f64x8 = _mm512_maskz_loadu_pd(tail_mask, a + tail_start);
            b_j_f64x8 = _mm512_maskz_loadu_pd(tail_mask, b + tail_start);
            c_f64x8 = _mm512_maskz_loadu_pd(tail_mask, c + i * n + tail_start);
        }
        diff_j_f64x8 = _mm512_sub_pd(a_j_f64x8, b_j_f64x8);
        cdiff_j_f64x8 = _mm512_fmadd_pd(diff_j_f64x8, c_f64x8, cdiff_j_f64x8);
        j += 8;
        if (j < n) goto nk_bilinear_f64_skylake_cycle;
        sum_f64x8 = _mm512_fmadd_pd(diff_i_f64x8, cdiff_j_f64x8, sum_f64x8);
    }

    *result = nk_sqrt_f64_haswell_(_mm512_reduce_add_pd(sum_f64x8));
}

NK_PUBLIC void nk_bilinear_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                        nk_f64c_t *results) {

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_i64x8 = _mm512_set_epi64(                                   //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000  //
    );

    // Default case for arbitrary size `n`
    nk_size_t const tail_length = n % 4;
    nk_size_t const tail_start = n - tail_length;
    __mmask8 const tail_mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, tail_length * 2);
    nk_f64_t sum_real = 0;
    nk_f64_t sum_imag = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t const a_i_real = a[i].real;
        nk_f64_t const a_i_imag = a[i].imag;
        __m512d cb_j_real_f64x8 = _mm512_setzero_pd();
        __m512d cb_j_imag_f64x8 = _mm512_setzero_pd();
        __m512d b_f64x8, c_f64x8;
        nk_size_t j = 0;

    nk_bilinear_f64c_skylake_cycle:
        if (j + 4 <= n) {
            b_f64x8 = _mm512_loadu_pd((nk_f64_t const *)(b + j));
            c_f64x8 = _mm512_loadu_pd((nk_f64_t const *)(c + i * n + j));
        }
        else {
            b_f64x8 = _mm512_maskz_loadu_pd(tail_mask, (nk_f64_t const *)(b + tail_start));
            c_f64x8 = _mm512_maskz_loadu_pd(tail_mask, (nk_f64_t const *)(c + i * n + tail_start));
        }
        // The real part of the product: b.real * c.real - b.imag * c.imag.
        // The subtraction will be performed later with a sign flip.
        cb_j_real_f64x8 = _mm512_fmadd_pd(c_f64x8, b_f64x8, cb_j_real_f64x8);
        // The imaginary part of the product: b.real * c.imag + b.imag * c.real.
        // Swap the imaginary and real parts of `c` before multiplication:
        c_f64x8 = _mm512_permute_pd(c_f64x8, 0x55); //? Same as 0b01010101.
        cb_j_imag_f64x8 = _mm512_fmadd_pd(c_f64x8, b_f64x8, cb_j_imag_f64x8);
        j += 4;
        if (j < n) goto nk_bilinear_f64c_skylake_cycle;
        // Flip the sign bit in every second scalar before accumulation:
        cb_j_real_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(cb_j_real_f64x8), sign_flip_i64x8));
        // Horizontal sums are the expensive part of the computation:
        nk_f64_t const cb_j_real = _mm512_reduce_add_pd(cb_j_real_f64x8);
        nk_f64_t const cb_j_imag = _mm512_reduce_add_pd(cb_j_imag_f64x8);
        sum_real += a_i_real * cb_j_real - a_i_imag * cb_j_imag;
        sum_imag += a_i_real * cb_j_imag + a_i_imag * cb_j_real;
    }

    // Reduce horizontal sums:
    results->real = sum_real;
    results->imag = sum_imag;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_GENOA
#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512bf16,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512bf16", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_bilinear_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                      nk_f32_t *result) {
    nk_size_t const tail_length = n % 32;
    nk_size_t const tail_start = n - tail_length;
    __mmask32 const tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length);
    __m512 sum_f32x16 = _mm512_setzero_ps();

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_f32;
        nk_bf16_to_f32_serial(a + i, &a_f32);
        __m512 a_f32x16 = _mm512_set1_ps(a_f32);
        __m512 cb_j_f32x16 = _mm512_setzero_ps();
        __m512i b_bf16x32, c_bf16x32;
        nk_size_t j = 0;

    nk_bilinear_bf16_genoa_cycle:
        if (j + 32 <= n) {
            b_bf16x32 = _mm512_loadu_epi16(b + j);
            c_bf16x32 = _mm512_loadu_epi16(c + i * n + j);
        }
        else {
            b_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, b + tail_start);
            c_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, c + i * n + tail_start);
        }
        cb_j_f32x16 = _mm512_dpbf16_ps(cb_j_f32x16, (__m512bh)(b_bf16x32), (__m512bh)(c_bf16x32));
        j += 32;
        if (j < n) goto nk_bilinear_bf16_genoa_cycle;
        sum_f32x16 = _mm512_fmadd_ps(a_f32x16, cb_j_f32x16, sum_f32x16);
    }

    *result = _mm512_reduce_add_ps(sum_f32x16);
}

NK_PUBLIC void nk_mahalanobis_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                         nk_f32_t *result) {
    nk_size_t const tail_length = n % 32;
    nk_size_t const tail_start = n - tail_length;
    __mmask32 const tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length);
    __m512 sum_f32x16 = _mm512_setzero_ps();

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_i, b_i;
        nk_bf16_to_f32_serial(a + i, &a_i);
        nk_bf16_to_f32_serial(b + i, &b_i);
        __m512 diff_i_f32x16 = _mm512_set1_ps(a_i - b_i);
        __m512 cdiff_j_f32x16 = _mm512_setzero_ps();
        __m512i a_j_bf16x32, b_j_bf16x32, diff_j_bf16x32, c_bf16x32;
        nk_size_t j = 0;

        // The nested loop is cleaner to implement with a `goto` in this case:
    nk_mahalanobis_bf16_genoa_cycle:
        if (j + 32 <= n) {
            a_j_bf16x32 = _mm512_loadu_epi16(a + j);
            b_j_bf16x32 = _mm512_loadu_epi16(b + j);
            c_bf16x32 = _mm512_loadu_epi16(c + i * n + j);
        }
        else {
            a_j_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, a + tail_start);
            b_j_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, b + tail_start);
            c_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, c + i * n + tail_start);
        }
        diff_j_bf16x32 = nk_substract_bf16x32_genoa_(a_j_bf16x32, b_j_bf16x32);
        cdiff_j_f32x16 = _mm512_dpbf16_ps(cdiff_j_f32x16, (__m512bh)(diff_j_bf16x32), (__m512bh)(c_bf16x32));
        j += 32;
        if (j < n) goto nk_mahalanobis_bf16_genoa_cycle;
        sum_f32x16 = _mm512_fmadd_ps(diff_i_f32x16, cdiff_j_f32x16, sum_f32x16);
    }

    *result = nk_sqrt_f32_haswell_(_mm512_reduce_add_ps(sum_f32x16));
}

NK_PUBLIC void nk_bilinear_bf16c_genoa(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c, nk_size_t n,
                                       nk_f32c_t *results) {

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_i32x16 = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_i8x64 = _mm512_set_epi8(                //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

    // Default case for arbitrary size `n`
    nk_size_t const tail_length = n % 16;
    nk_size_t const tail_start = n - tail_length;
    __mmask32 const tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length * 2);
    nk_f64_t sum_real = 0;
    nk_f64_t sum_imag = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t const a_i_real = a[i].real;
        nk_f32_t const a_i_imag = a[i].imag;
        __m512 cb_j_real_f32x16 = _mm512_setzero_ps();
        __m512 cb_j_imag_f32x16 = _mm512_setzero_ps();
        __m512i b_bf16x32, c_bf16x32;
        nk_size_t j = 0;

    nk_bilinear_bf16c_skylake_cycle:
        if (j + 16 <= n) {
            b_bf16x32 = _mm512_loadu_epi16((nk_i16_t const *)(b + j));
            c_bf16x32 = _mm512_loadu_epi16((nk_i16_t const *)(c + i * n + j));
        }
        else {
            b_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, (nk_i16_t const *)(b + tail_start));
            c_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, (nk_i16_t const *)(c + i * n + tail_start));
        }
        cb_j_real_f32x16 = _mm512_dpbf16_ps(                           //
            cb_j_real_f32x16,                                          //
            (__m512bh)(_mm512_xor_si512(c_bf16x32, sign_flip_i32x16)), //
            (__m512bh)b_bf16x32);
        cb_j_imag_f32x16 = _mm512_dpbf16_ps(                                 //
            cb_j_imag_f32x16,                                                //
            (__m512bh)(_mm512_shuffle_epi8(c_bf16x32, swap_adjacent_i8x64)), //
            (__m512bh)b_bf16x32);
        j += 16;
        if (j < n) goto nk_bilinear_bf16c_skylake_cycle;
        // Horizontal sums are the expensive part of the computation:
        nk_f64_t const cb_j_real = nk_reduce_add_f32x16_skylake_(cb_j_real_f32x16);
        nk_f64_t const cb_j_imag = nk_reduce_add_f32x16_skylake_(cb_j_imag_f32x16);
        sum_real += a_i_real * cb_j_real - a_i_imag * cb_j_imag;
        sum_imag += a_i_real * cb_j_imag + a_i_imag * cb_j_real;
    }

    // Reduce horizontal sums:
    results->real = sum_real;
    results->imag = sum_imag;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_GENOA

#if NK_TARGET_SAPPHIRE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512fp16,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512fp16", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_bilinear_f16_sapphire_under32unrolled(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c,
                                                        nk_size_t const n, nk_f32_t *result) {
    // The goal of this optimization is to avoid horizontal accumulation of the cb_j sums
    // until the very end of the computation.
    __mmask32 const mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
    __m512h const b_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));

    // Independently accumulate the partial sums into separate variables to avoid data-dependencies.
    __m512h cb_j1_f16x32 = _mm512_setzero_ph();
    __m512h cb_j2_f16x32 = _mm512_setzero_ph();
    __m512h cb_j3_f16x32 = _mm512_setzero_ph();
    __m512h cb_j4_f16x32 = _mm512_setzero_ph();

    // Unroll the loop to process 4x ZMM registers at a time.
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        // If the code is compiled without native support for `_Float16`, we need a workaround
        // to avoid implicit casts from out `nk_f16_t` to `_Float16`.
        cb_j1_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 0))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 0]))),
            cb_j1_f16x32);
        cb_j2_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 1))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 1]))),
            cb_j2_f16x32);
        cb_j3_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 2))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 2]))),
            cb_j3_f16x32);
        cb_j4_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 3))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 3]))),
            cb_j4_f16x32);
    }

    // Handle the tail of the loop:
    if (i + 0 < n)
        cb_j1_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 0))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 0]))),
            cb_j1_f16x32);
    if (i + 1 < n)
        cb_j2_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 1))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 1]))),
            cb_j2_f16x32);
    if (i + 2 < n)
        cb_j3_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 2))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 2]))),
            cb_j3_f16x32);
    if (i + 3 < n)
        cb_j4_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 3))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 3]))),
            cb_j4_f16x32);

    // Combine cb_j sums
    __m512h sum_f16x32 = _mm512_add_ph(            //
        _mm512_add_ph(cb_j1_f16x32, cb_j2_f16x32), //
        _mm512_add_ph(cb_j3_f16x32, cb_j4_f16x32));
    *result = _mm512_reduce_add_ph(sum_f16x32);
}

NK_PUBLIC void nk_bilinear_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                        nk_f32_t *result) {

    // On modern x86 CPUs we have enough register space to load fairly large matrices with up to 32 cells
    // per row and 32 rows at a time, still keeping enough register space for temporaries.
    if (n <= 32) {
        nk_bilinear_f16_sapphire_under32unrolled(a, b, c, n, result);
        return;
    }

    nk_size_t const tail_length = n % 32;
    nk_size_t const tail_start = n - tail_length;
    __mmask32 const tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length);
    __m512h sum_f16x32 = _mm512_setzero_ph();

    for (nk_size_t i = 0; i != n; ++i) {
        __m512h a_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(*(short const *)(a + i)));
        __m512h cb_j_f16x32 = _mm512_setzero_ph();
        __m512i b_f16x32, c_f16x32;
        nk_size_t j = 0;

    nk_bilinear_f16_sapphire_cycle:
        if (j + 32 <= n) {
            b_f16x32 = _mm512_loadu_epi16(b + j);
            c_f16x32 = _mm512_loadu_epi16(c + i * n + j);
        }
        else {
            b_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, b + tail_start);
            c_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, c + i * n + tail_start);
        }
        cb_j_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(b_f16x32), _mm512_castsi512_ph(c_f16x32), cb_j_f16x32);
        j += 32;
        if (j < n) goto nk_bilinear_f16_sapphire_cycle;
        sum_f16x32 = _mm512_fmadd_ph(a_f16x32, cb_j_f16x32, sum_f16x32);
    }

    *result = _mm512_reduce_add_ph(sum_f16x32);
}

NK_PUBLIC void nk_mahalanobis_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                           nk_f32_t *result) {
    nk_size_t const tail_length = n % 32;
    nk_size_t const tail_start = n - tail_length;
    __mmask32 const tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length);
    __m512h sum_f16x32 = _mm512_setzero_ph();

    for (nk_size_t i = 0; i != n; ++i) {
        __m512h a_i_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(*(short const *)(a + i)));
        __m512h b_i_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(*(short const *)(b + i)));
        __m512h diff_i_f16x32 = _mm512_sub_ph(a_i_f16x32, b_i_f16x32);
        __m512h cdiff_j_f16x32 = _mm512_setzero_ph();
        __m512h diff_j_f16x32;
        __m512i a_j_f16x32, b_j_f16x32, c_f16x32;
        nk_size_t j = 0;

        // The nested loop is cleaner to implement with a `goto` in this case:
    nk_mahalanobis_f16_sapphire_cycle:
        if (j + 32 <= n) {
            a_j_f16x32 = _mm512_loadu_epi16(a + j);
            b_j_f16x32 = _mm512_loadu_epi16(b + j);
            c_f16x32 = _mm512_loadu_epi16(c + i * n + j);
        }
        else {
            a_j_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, a + tail_start);
            b_j_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, b + tail_start);
            c_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, c + i * n + tail_start);
        }
        diff_j_f16x32 = _mm512_sub_ph(_mm512_castsi512_ph(a_j_f16x32), _mm512_castsi512_ph(b_j_f16x32));
        cdiff_j_f16x32 = _mm512_fmadd_ph(diff_j_f16x32, _mm512_castsi512_ph(c_f16x32), cdiff_j_f16x32);
        j += 32;
        if (j < n) goto nk_mahalanobis_f16_sapphire_cycle;
        sum_f16x32 = _mm512_fmadd_ph(diff_i_f16x32, cdiff_j_f16x32, sum_f16x32);
    }

    *result = nk_sqrt_f32_haswell_(_mm512_reduce_add_ph(sum_f16x32));
}

NK_PUBLIC void nk_bilinear_f16c_sapphire(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                         nk_f32c_t *results) {

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_i32x16 = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_i8x64 = _mm512_set_epi8(                //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

    // Default case for arbitrary size `n`
    nk_size_t const tail_length = n % 16;
    nk_size_t const tail_start = n - tail_length;
    __mmask32 const tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length * 2);
    nk_f32_t sum_real = 0;
    nk_f32_t sum_imag = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t const a_i_real = a[i].real;
        nk_f32_t const a_i_imag = a[i].imag;
        __m512h cb_j_real_f16x32 = _mm512_setzero_ph();
        __m512h cb_j_imag_f16x32 = _mm512_setzero_ph();
        __m512i b_f16x32, c_f16x32;
        nk_size_t j = 0;

    nk_bilinear_f16c_skylake_cycle:
        if (j + 16 <= n) {
            b_f16x32 = _mm512_loadu_epi16((nk_i16_t const *)(b + j));
            c_f16x32 = _mm512_loadu_epi16((nk_i16_t const *)(c + i * n + j));
        }
        else {
            b_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, (nk_i16_t const *)(b + tail_start));
            c_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, (nk_i16_t const *)(c + i * n + tail_start));
        }
        cb_j_real_f16x32 = _mm512_fmadd_ph(                                    //
            _mm512_castsi512_ph(_mm512_xor_si512(c_f16x32, sign_flip_i32x16)), //
            _mm512_castsi512_ph(b_f16x32), cb_j_real_f16x32);
        cb_j_imag_f16x32 = _mm512_fmadd_ph(                                          //
            _mm512_castsi512_ph(_mm512_shuffle_epi8(c_f16x32, swap_adjacent_i8x64)), //
            _mm512_castsi512_ph(b_f16x32), cb_j_imag_f16x32);
        j += 16;
        if (j < n) goto nk_bilinear_f16c_skylake_cycle;
        // Horizontal sums are the expensive part of the computation:
        nk_f32_t const cb_j_real = _mm512_reduce_add_ph(cb_j_real_f16x32);
        nk_f32_t const cb_j_imag = _mm512_reduce_add_ph(cb_j_imag_f16x32);
        sum_real += a_i_real * cb_j_real - a_i_imag * cb_j_imag;
        sum_imag += a_i_real * cb_j_imag + a_i_imag * cb_j_real;
    }

    // Reduce horizontal sums:
    results->real = sum_real;
    results->imag = sum_imag;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_bilinear_f64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_bilinear_f64_skylake(a, b, c, n, result);
#else
    nk_bilinear_f64_serial(a, b, c, n, result);
#endif
}

NK_PUBLIC void nk_bilinear_f32(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SKYLAKE
    nk_bilinear_f32_skylake(a, b, c, n, result);
#elif NK_TARGET_NEON
    nk_bilinear_f32_neon(a, b, c, n, result);
#else
    nk_bilinear_f32_serial(a, b, c, n, result);
#endif
}

NK_PUBLIC void nk_bilinear_f16(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SAPPHIRE
    nk_bilinear_f16_sapphire(a, b, c, n, result);
#elif NK_TARGET_HASWELL
    nk_bilinear_f16_haswell(a, b, c, n, result);
#elif NK_TARGET_NEONHALF
    nk_bilinear_f16_neonhalf(a, b, c, n, result);
#else
    nk_bilinear_f16_serial(a, b, c, n, result);
#endif
}

NK_PUBLIC void nk_bilinear_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_bilinear_bf16_genoa(a, b, c, n, result);
#elif NK_TARGET_HASWELL
    nk_bilinear_bf16_haswell(a, b, c, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_bilinear_bf16_neonbfdot(a, b, c, n, result);
#else
    nk_bilinear_bf16_serial(a, b, c, n, result);
#endif
}

/**
 *  @brief Complex bilinear form between vectors a and b under metric tensor c.
 *
 *  @param[in] a The first complex vector.
 *  @param[in] b The second complex vector.
 *  @param[in] c The complex metric tensor, stored row-major as an n-by-n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] results The output array containing real and imaginary parts.
 *
 *  @note The output array stores real part in results[0] and imaginary part in results[1].
 */
NK_PUBLIC void nk_bilinear_f64c(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                nk_f64c_t *results) {
#if NK_TARGET_SKYLAKE
    nk_bilinear_f64c_skylake(a, b, c, n, results);
#else
    nk_bilinear_f64c_serial(a, b, c, n, results);
#endif
}

/**
 *  @brief Complex bilinear form between vectors a and b under metric tensor c.
 *
 *  @param[in] a The first complex vector.
 *  @param[in] b The second complex vector.
 *  @param[in] c The complex metric tensor, stored row-major as an n-by-n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] results The output array containing real and imaginary parts.
 *
 *  @note The output array stores real part in results[0] and imaginary part in results[1].
 */
NK_PUBLIC void nk_bilinear_f32c(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                nk_f32c_t *results) {
#if NK_TARGET_SKYLAKE
    nk_bilinear_f32c_skylake(a, b, c, n, results);
#elif NK_TARGET_NEON
    nk_bilinear_f32c_neon(a, b, c, n, results);
#else
    nk_bilinear_f32c_serial(a, b, c, n, results);
#endif
}

/**
 *  @brief Complex bilinear form between vectors a and b under metric tensor c.
 *
 *  @param[in] a The first complex vector.
 *  @param[in] b The second complex vector.
 *  @param[in] c The complex metric tensor, stored row-major as an n-by-n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] results The output array containing real and imaginary parts.
 *
 *  @note The output array stores real part in results[0] and imaginary part in results[1].
 */
NK_PUBLIC void nk_bilinear_f16c(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                nk_f32c_t *results) {
#if NK_TARGET_SAPPHIRE
    nk_bilinear_f16c_sapphire(a, b, c, n, results);
#elif NK_TARGET_NEONHALF
    nk_bilinear_f16c_neonhalf(a, b, c, n, results);
#else
    nk_bilinear_f16c_serial(a, b, c, n, results);
#endif
}

/**
 *  @brief Complex bilinear form between vectors a and b under metric tensor c.
 *
 *  @param[in] a The first complex vector.
 *  @param[in] b The second complex vector.
 *  @param[in] c The complex metric tensor, stored row-major as an n-by-n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] results The output array containing real and imaginary parts.
 *
 *  @note The output array stores real part in results[0] and imaginary part in results[1].
 */
NK_PUBLIC void nk_bilinear_bf16c(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c, nk_size_t n,
                                 nk_f32c_t *results) {
#if NK_TARGET_GENOA
    nk_bilinear_bf16c_genoa(a, b, c, n, results);
#elif NK_TARGET_NEONBFDOT
    nk_bilinear_bf16c_neonbfdot(a, b, c, n, results);
#else
    nk_bilinear_bf16c_serial(a, b, c, n, results);
#endif
}

NK_PUBLIC void nk_mahalanobis_f64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                  nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_mahalanobis_f64_skylake(a, b, c, n, result);
#else
    nk_mahalanobis_f64_serial(a, b, c, n, result);
#endif
}

NK_PUBLIC void nk_mahalanobis_f32(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                  nk_f32_t *result) {
#if NK_TARGET_SKYLAKE
    nk_mahalanobis_f32_skylake(a, b, c, n, result);
#elif NK_TARGET_NEON
    nk_mahalanobis_f32_neon(a, b, c, n, result);
#else
    nk_mahalanobis_f32_serial(a, b, c, n, result);
#endif
}

NK_PUBLIC void nk_mahalanobis_f16(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                  nk_f32_t *result) {
#if NK_TARGET_SAPPHIRE
    nk_mahalanobis_f16_sapphire(a, b, c, n, result);
#elif NK_TARGET_HASWELL
    nk_mahalanobis_f16_haswell(a, b, c, n, result);
#elif NK_TARGET_NEONHALF
    nk_mahalanobis_f16_neonhalf(a, b, c, n, result);
#else
    nk_mahalanobis_f16_serial(a, b, c, n, result);
#endif
}

NK_PUBLIC void nk_mahalanobis_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                   nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_mahalanobis_bf16_genoa(a, b, c, n, result);
#elif NK_TARGET_HASWELL
    nk_mahalanobis_bf16_haswell(a, b, c, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_mahalanobis_bf16_neonbfdot(a, b, c, n, result);
#else
    nk_mahalanobis_bf16_serial(a, b, c, n, result);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
