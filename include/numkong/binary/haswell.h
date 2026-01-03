/**
 *  @brief SIMD-accelerated Binary Similarity Measures optimized for Intel Haswell CPUs.
 *  @file include/numkong/binary/haswell.h
 *  @sa include/numkong/binary.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_BINARY_HASWELL_H
#define NK_BINARY_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("popcnt")
#pragma clang attribute push(__attribute__((target("popcnt"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/binary/serial.h" // `nk_popcount_b8`

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_hamming_b8_haswell(nk_b8_t const *a, nk_b8_t const *b, nk_size_t n_words, nk_u32_t *result) {
    // x86 supports unaligned loads and works just fine with the scalar version for small vectors.
    nk_u32_t differences = 0;
    for (; n_words >= 8; n_words -= 8, a += 8, b += 8)
        differences += _mm_popcnt_u64(*(nk_u64_t const *)a ^ *(nk_u64_t const *)b);
    for (; n_words; --n_words, ++a, ++b) differences += _mm_popcnt_u32(*a ^ *b);
    *result = differences;
}

NK_PUBLIC void nk_jaccard_b8_haswell(nk_b8_t const *a, nk_b8_t const *b, nk_size_t n_words, nk_f32_t *result) {
    // x86 supports unaligned loads and works just fine with the scalar version for small vectors.
    nk_u32_t intersection_count = 0, union_count = 0;
    for (; n_words >= 8; n_words -= 8, a += 8, b += 8)
        intersection_count += (nk_u32_t)_mm_popcnt_u64(*(nk_u64_t const *)a & *(nk_u64_t const *)b),
            union_count += (nk_u32_t)_mm_popcnt_u64(*(nk_u64_t const *)a | *(nk_u64_t const *)b);
    for (; n_words; --n_words, ++a, ++b)
        intersection_count += nk_popcount_b8(*a & *b), union_count += nk_popcount_b8(*a | *b);
    *result = (union_count != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)union_count : 1.0f;
}

NK_PUBLIC void nk_jaccard_u32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t intersection_count = 0;
    nk_size_t n_remaining = n;
    for (; n_remaining >= 4; n_remaining -= 4, a += 4, b += 4) {
        __m128i a_u32x4 = _mm_loadu_si128((__m128i const *)a);
        __m128i b_u32x4 = _mm_loadu_si128((__m128i const *)b);
        __m128i equality_u32x4 = _mm_cmpeq_epi32(a_u32x4, b_u32x4);
        int equality_mask = _mm_movemask_ps(_mm_castsi128_ps(equality_u32x4));
        intersection_count += (nk_u32_t)_mm_popcnt_u32((unsigned int)equality_mask);
    }
    for (; n_remaining; --n_remaining, ++a, ++b) intersection_count += (*a == *b);
    *result = (n != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)n : 1.0f;
}

typedef struct nk_jaccard_b256_state_haswell_t {
    nk_u32_t intersection_count;
} nk_jaccard_b256_state_haswell_t;

NK_INTERNAL void nk_jaccard_b256_init_haswell(nk_jaccard_b256_state_haswell_t *state) { state->intersection_count = 0; }

NK_INTERNAL void nk_jaccard_b256_update_haswell(nk_jaccard_b256_state_haswell_t *state, nk_b256_vec_t a,
                                                nk_b256_vec_t b) {
    // Process one 256-bit chunk (native Haswell AVX2 register size).
    //
    // Haswell port analysis:
    //   `_mm256_and_si256`:      p015, 1cy latency, 0.33cy throughput
    //   `_mm256_extracti128`:    p5, 3cy latency, 1cy throughput
    //   `_mm_cvtsi128_si64`:     p0, 2cy latency (first u64 extraction)
    //   `_mm_extract_epi64`:     p5, 3cy latency, 1cy throughput
    //   `_mm_popcnt_u64`:        p1 ONLY, 3cy latency, 1cy throughput (BOTTLENECK)
    //
    // With 4 popcounts per update, p1 is the bottleneck at 4 cycles minimum.

    // Step 1: Compute intersection bits (A AND B)
    __m256i intersection_u8x32 = _mm256_and_si256(a.ymm, b.ymm);

    // Step 2: Extract the two 128-bit halves
    __m128i intersection_low_u8x16 = _mm256_castsi256_si128(intersection_u8x32);       // FREE (register view)
    __m128i intersection_high_u8x16 = _mm256_extracti128_si256(intersection_u8x32, 1); // p5, 3cy

    // Step 3: Extract individual 64-bit words for scalar popcount
    nk_u64_t word_a = (nk_u64_t)_mm_cvtsi128_si64(intersection_low_u8x16);
    nk_u64_t word_b = (nk_u64_t)_mm_extract_epi64(intersection_low_u8x16, 1);
    nk_u64_t word_c = (nk_u64_t)_mm_cvtsi128_si64(intersection_high_u8x16);
    nk_u64_t word_d = (nk_u64_t)_mm_extract_epi64(intersection_high_u8x16, 1);

    // Step 4: Popcount each word (p1 bottleneck: 4 ops @ 1cy throughput = 4cy)
    nk_u32_t partial_a = (nk_u32_t)_mm_popcnt_u64(word_a);
    nk_u32_t partial_b = (nk_u32_t)_mm_popcnt_u64(word_b);
    nk_u32_t partial_c = (nk_u32_t)_mm_popcnt_u64(word_c);
    nk_u32_t partial_d = (nk_u32_t)_mm_popcnt_u64(word_d);

    // Step 5: Sum all partials (associative grouping for parallel adds)
    state->intersection_count += (partial_a + partial_b) + (partial_c + partial_d);
}

NK_INTERNAL void nk_jaccard_b256_finalize_haswell(nk_jaccard_b256_state_haswell_t const *state_a,
                                                  nk_jaccard_b256_state_haswell_t const *state_b,
                                                  nk_jaccard_b256_state_haswell_t const *state_c,
                                                  nk_jaccard_b256_state_haswell_t const *state_d,
                                                  nk_f32_t query_popcount, nk_f32_t target_popcount_a,
                                                  nk_f32_t target_popcount_b, nk_f32_t target_popcount_c,
                                                  nk_f32_t target_popcount_d, nk_f32_t *results) {

    // 4-way SIMD Jaccard computation with fast reciprocal.
    //
    // Haswell port analysis:
    //   `_mm_setr_ps`:     p5, 1cy (INSERTPS chain)
    //   `_mm_add_ps`:      p01, 3cy latency
    //   `_mm_sub_ps`:      p01, 3cy latency
    //   `_mm_rcp_ps`:      p0, 5cy latency, 1cy throughput
    //   `_mm_mul_ps`:      p01, 5cy latency, 0.5cy throughput
    //   `_mm_blendv_ps`:   p015, 2cy latency

    // Pack intersection counts and convert to float
    nk_f32_t intersection_a_f32 = (nk_f32_t)state_a->intersection_count;
    nk_f32_t intersection_b_f32 = (nk_f32_t)state_b->intersection_count;
    nk_f32_t intersection_c_f32 = (nk_f32_t)state_c->intersection_count;
    nk_f32_t intersection_d_f32 = (nk_f32_t)state_d->intersection_count;

    __m128 intersection_f32x4 = _mm_setr_ps(intersection_a_f32, intersection_b_f32, intersection_c_f32,
                                            intersection_d_f32);
    __m128 query_f32x4 = _mm_set1_ps(query_popcount);
    __m128 targets_f32x4 = _mm_setr_ps(target_popcount_a, target_popcount_b, target_popcount_c, target_popcount_d);
    __m128 union_f32x4 = _mm_sub_ps(_mm_add_ps(query_f32x4, targets_f32x4), intersection_f32x4);

    // Handle zero-union edge case
    __m128 zero_union_mask = _mm_cmpeq_ps(union_f32x4, _mm_setzero_ps());
    __m128 one_f32x4 = _mm_set1_ps(1.0f);
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 safe_union_f32x4 = _mm_blendv_ps(union_f32x4, one_f32x4, zero_union_mask);

    // Fast reciprocal with Newton-Raphson refinement:
    //   `_mm_rcp_ps`: ~12-bit precision, 5cy latency, 1cy throughput
    //   Newton-Raphson: rcp' = rcp * (2 - x * rcp), doubles precision to ~22-24 bits
    // Total: ~10cy vs `_mm_div_ps` 13cy latency, but NR has better throughput
    __m128 union_reciprocal_f32x4 = _mm_rcp_ps(safe_union_f32x4);
    __m128 newton_raphson_correction = _mm_sub_ps(two_f32x4, _mm_mul_ps(safe_union_f32x4, union_reciprocal_f32x4));
    union_reciprocal_f32x4 = _mm_mul_ps(union_reciprocal_f32x4, newton_raphson_correction);

    __m128 ratio_f32x4 = _mm_mul_ps(intersection_f32x4, union_reciprocal_f32x4);
    __m128 jaccard_f32x4 = _mm_sub_ps(one_f32x4, ratio_f32x4);
    __m128 result_f32x4 = _mm_blendv_ps(jaccard_f32x4, one_f32x4, zero_union_mask);

    _mm_storeu_ps(results, result_f32x4);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_

#endif // NK_BINARY_HASWELL_H
