/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Sierra Forest CPUs.
 *  @file include/numkong/dot/sierra.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_SIERRA_H
#define NK_DOT_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA
#pragma GCC push_options
#pragma GCC target("avx2", "bmi2", "avxvnni")
#pragma clang attribute push(__attribute__((target("avx2,bmi2,avxvnni"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/reduce/haswell.h" // nk_reduce_add_i32x8_haswell_

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_i8_sierra(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                nk_i32_t *result) {

    __m256i sum_i32x8 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        __m256i a_i8x32 = _mm256_lddqu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_i8x32 = _mm256_lddqu_si256((__m256i const *)(b_scalars + idx_scalars));
        sum_i32x8 = _mm256_dpbssds_epi32(sum_i32x8, a_i8x32, b_i8x32);
    }

    // Further reduce to a single sum for each vector
    int sum_i32 = nk_reduce_add_i32x8_haswell_(sum_i32x8);

    // Take care of the tail:
    for (; idx_scalars < count_scalars; ++idx_scalars)
        sum_i32 += (int)(a_scalars[idx_scalars]) * b_scalars[idx_scalars];
    *result = sum_i32;
}

typedef struct nk_dot_i8x32_state_sierra_t {
    __m256i sum_i32x8;
} nk_dot_i8x32_state_sierra_t;

NK_INTERNAL void nk_dot_i8x32_init_sierra(nk_dot_i8x32_state_sierra_t *state) {
    state->sum_i32x8 = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_i8x32_update_sierra(nk_dot_i8x32_state_sierra_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    __m256i sum_i32x8 = state->sum_i32x8;
    __m256i a_i8x32 = _mm256_lddqu_si256((__m256i const *)(a.i8s));
    __m256i b_i8x32 = _mm256_lddqu_si256((__m256i const *)(b.i8s));
    state->sum_i32x8 = _mm256_dpbssds_epi32(sum_i32x8, a_i8x32, b_i8x32);
}

NK_INTERNAL void nk_dot_i8x32_finalize_sierra(                                              //
    nk_dot_i8x32_state_sierra_t const *state_a, nk_dot_i8x32_state_sierra_t const *state_b, //
    nk_dot_i8x32_state_sierra_t const *state_c, nk_dot_i8x32_state_sierra_t const *state_d, //
    nk_i32_t *results) {
    // ILP-optimized 4-way horizontal reduction for i32 in AVX2 (8 elements -> 1 scalar each)
    // Step 1: 8->4 for all 4 states (extract high 128-bit half and add to low half)
    __m128i sum_i32x4_a = _mm_add_epi32(_mm256_castsi256_si128(state_a->sum_i32x8),
                                        _mm256_extracti128_si256(state_a->sum_i32x8, 1));
    __m128i sum_i32x4_b = _mm_add_epi32(_mm256_castsi256_si128(state_b->sum_i32x8),
                                        _mm256_extracti128_si256(state_b->sum_i32x8, 1));
    __m128i sum_i32x4_c = _mm_add_epi32(_mm256_castsi256_si128(state_c->sum_i32x8),
                                        _mm256_extracti128_si256(state_c->sum_i32x8, 1));
    __m128i sum_i32x4_d = _mm_add_epi32(_mm256_castsi256_si128(state_d->sum_i32x8),
                                        _mm256_extracti128_si256(state_d->sum_i32x8, 1));
    // Step 2: Transpose 4x4 matrix of partial sums using integer shuffles
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_i32x4_a, sum_i32x4_b);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_i32x4_c, sum_i32x4_d);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_i32x4_a, sum_i32x4_b);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_i32x4_c, sum_i32x4_d);
    __m128i sum_lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    // Step 3: Vertical sum - each lane becomes the final i32 result for one state
    __m128i final_sum_i32x4 = _mm_add_epi32(_mm_add_epi32(sum_lane0_i32x4, sum_lane1_i32x4),
                                            _mm_add_epi32(sum_lane2_i32x4, sum_lane3_i32x4));
    // Store as i32
    _mm_storeu_si128((__m128i *)results, final_sum_i32x4);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_

#endif // NK_DOT_SIERRA_H