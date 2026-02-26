/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for Sierra Forest.
 *  @file include/numkong/spatial/sierra.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 */
#ifndef NK_SPATIAL_SIERRA_H
#define NK_SPATIAL_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA

#include "numkong/types.h"
#include "numkong/reduce/haswell.h" // `nk_reduce_add_i32x8_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni,avxvnniint8"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni", "avxvnniint8")
#endif

NK_PUBLIC void nk_angular_i8_sierra(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {

    __m256i dot_product_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_i32x8 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_loadu_si256((__m256i const *)(a + i));
        __m256i b_i8x32 = _mm256_loadu_si256((__m256i const *)(b + i));
        dot_product_i32x8 = _mm256_dpbssds_epi32(dot_product_i32x8, a_i8x32, b_i8x32);
        a_norm_sq_i32x8 = _mm256_dpbssds_epi32(a_norm_sq_i32x8, a_i8x32, a_i8x32);
        b_norm_sq_i32x8 = _mm256_dpbssds_epi32(b_norm_sq_i32x8, b_i8x32, b_i8x32);
    }

    // Further reduce to a single sum for each vector
    nk_i32_t dot_product_i32 = nk_reduce_add_i32x8_haswell_(dot_product_i32x8);
    nk_i32_t a_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(a_norm_sq_i32x8);
    nk_i32_t b_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(b_norm_sq_i32x8);

    // Take care of the tail:
    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = nk_angular_normalize_f32_haswell_(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_
#endif // NK_SPATIAL_SIERRA_H
