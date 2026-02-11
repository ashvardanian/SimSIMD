/**
 *  @brief SIMD-accelerated Dot Products for Ice Lake.
 *  @file include/numkong/dot/icelake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_icelake_instructions VNNI Instructions Performance
 *
 *      Intrinsic                   Instruction                     Ice         Genoa
 *      _mm512_dpwssd_epi32         VPDPWSSD (ZMM, ZMM, ZMM)        5cy @ p0    4cy @ p01
 *      _mm512_dpbusd_epi32         VPDPBUSD (ZMM, ZMM, ZMM)        5cy @ p0    4cy @ p01
 *      _mm512_madd_epi16           VPMADDWD (ZMM, ZMM, ZMM)        5cy @ p05   3cy @ p01
 *
 *  Ice Lake introduces AVX-512 VNNI for accelerated integer dot products. VNNI instructions bottleneck
 *  on port 0, limiting throughput to 1/cy. AMD Genoa dual-issues on ports 0-1, achieving 0.5/cy throughput.
 *  We use VPDPWSSD for signed i8 inputs after widening to i16, since VPDPBUSD is asymmetric (unsigned x signed).
 *
 *  @section dot_icelake_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines following structures and force-inlined
 *  `NK_INTERNAL` functions:
 *
 *  - nk_dot_i8x64 for 8-bit signed integer inputs using DPBUSD with algebraic transformation,
 *  - nk_dot_u8x64 for 8-bit unsigned integer inputs using DPBUSD with algebraic transformation,
 *  - nk_dot_i4x128 for 4-bit signed integer products with correction terms,
 *  - nk_dot_u4x128 for 4-bit unsigned integer products.
 *
 *  @code{c}
 *  nk_dot_i8x64_state_icelake_t state_first, state_second, state_third, state_fourth;
 *  nk_b512_vec_t query_i8x64, target_first_i8x64, target_second_i8x64, target_third_i8x64, target_fourth_i8x64;
 *  nk_dot_i8x64_init_icelake(&state_first);
 *  nk_dot_i8x64_init_icelake(&state_second);
 *  nk_dot_i8x64_init_icelake(&state_third);
 *  nk_dot_i8x64_init_icelake(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 64 <= depth; idx += 64) {
 *      query_i8x64.zmm = _mm512_loadu_si512(query_ptr + idx);
 *      target_first_i8x64.zmm = _mm512_loadu_si512(target_first_ptr + idx);
 *      target_second_i8x64.zmm = _mm512_loadu_si512(target_second_ptr + idx);
 *      target_third_i8x64.zmm = _mm512_loadu_si512(target_third_ptr + idx);
 *      target_fourth_i8x64.zmm = _mm512_loadu_si512(target_fourth_ptr + idx);
 *      nk_dot_i8x64_update_icelake(&state_first, query_i8x64, target_first_i8x64, idx, 64);
 *      nk_dot_i8x64_update_icelake(&state_second, query_i8x64, target_second_i8x64, idx, 64);
 *      nk_dot_i8x64_update_icelake(&state_third, query_i8x64, target_third_i8x64, idx, 64);
 *      nk_dot_i8x64_update_icelake(&state_fourth, query_i8x64, target_fourth_i8x64, idx, 64);
 *  }
 *  nk_b128_vec_t results_i32x4;
 *  nk_dot_i8x64_finalize_icelake(&state_first, &state_second, &state_third, &state_fourth, depth, &results_i32x4);
 *  @endcode
 *
 *  For 4-bit integers, the state manages the complex unpacking and correction term accumulation:
 *
 *  @code{c}
 *  nk_dot_i4x128_state_icelake_t state_first, state_second, state_third, state_fourth;
 *  nk_b512_vec_t query_i4x128, target_first_i4x128, target_second_i4x128, target_third_i4x128, target_fourth_i4x128;
 *  nk_dot_i4x128_init_icelake(&state_first);
 *  nk_dot_i4x128_init_icelake(&state_second);
 *  nk_dot_i4x128_init_icelake(&state_third);
 *  nk_dot_i4x128_init_icelake(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 128 <= depth; idx += 128) {
 *      query_i4x128.zmm = _mm512_loadu_si512(query_ptr + idx / 2);
 *      target_first_i4x128.zmm = _mm512_loadu_si512(target_first_ptr + idx / 2);
 *      target_second_i4x128.zmm = _mm512_loadu_si512(target_second_ptr + idx / 2);
 *      target_third_i4x128.zmm = _mm512_loadu_si512(target_third_ptr + idx / 2);
 *      target_fourth_i4x128.zmm = _mm512_loadu_si512(target_fourth_ptr + idx / 2);
 *      nk_dot_i4x128_update_icelake(&state_first, query_i4x128, target_first_i4x128, idx, 128);
 *      nk_dot_i4x128_update_icelake(&state_second, query_i4x128, target_second_i4x128, idx, 128);
 *      nk_dot_i4x128_update_icelake(&state_third, query_i4x128, target_third_i4x128, idx, 128);
 *      nk_dot_i4x128_update_icelake(&state_fourth, query_i4x128, target_fourth_i4x128, idx, 128);
 *  }
 *  nk_b128_vec_t results_i32x4;
 *  nk_dot_i4x128_finalize_icelake(&state_first, &state_second, &state_third, &state_fourth, depth, &results_i32x4);
 *  @endcode
 */
#ifndef NK_DOT_ICELAKE_H
#define NK_DOT_ICELAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICELAKE

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                   \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,avx512vbmi,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "avx512vbmi", "f16c", "fma", \
                   "bmi", "bmi2")
#endif

NK_PUBLIC void nk_dot_i8_icelake(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_i32_t *result) {
    // Optimized i8×i8 dot product using algebraic transformation with DPBUSD
    //
    // Old approach (Haswell/Skylake):
    //   - Sign-extend i8 → i16 using cvtepi8_epi16 (3cy latency @ p5, 32 elements/iteration)
    //   - Multiply i16×i16 using vpmaddwd + dpwssd
    //   - Bottleneck: cvtepi8_epi16 serializes on port 5
    //
    // New approach (Ice Lake+):
    //   - Use DPBUSD (unsigned×signed multiply-add) with algebraic transformation
    //   - Convert signed i8 to unsigned via XOR with 0x80: a' = a + 128
    //   - Compute dpbusd(a', b) = (a+128)×b, then correct: a×b = (a+128)×b - 128×sum(b)
    //   - Use SAD for fast correction term accumulation (1cy @ p5 vs 8-10cy with cvtepi8)
    //   - Processes 64 elements/iteration
    //
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const zeros_u8x64 = _mm512_setzero_si512();
    __m512i sum_ab_i32x16 = _mm512_setzero_si512();
    __m512i sum_b_biased_i64x8 = _mm512_setzero_si512();
    __m512i a_i8x64, b_i8x64;
    nk_size_t count_original = count_scalars;

nk_dot_i8_icelake_cycle:
    if (count_scalars < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, count_scalars);
        a_i8x64 = _mm512_maskz_loadu_epi8(mask, a_scalars);
        b_i8x64 = _mm512_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_i8x64 = _mm512_loadu_si512(a_scalars);
        b_i8x64 = _mm512_loadu_si512(b_scalars);
        a_scalars += 64, b_scalars += 64, count_scalars -= 64;
    }

    // Convert a to unsigned [0,255] by XOR with 0x80: a_biased = a + 128
    __m512i a_biased_u8x64 = _mm512_xor_si512(a_i8x64, xor_mask_u8x64);

    // Compute (a+128) × b using dpbusd: unsigned × signed
    sum_ab_i32x16 = _mm512_dpbusd_epi32(sum_ab_i32x16, a_biased_u8x64, b_i8x64);

    // Accumulate sum(b+128) using SAD (1cy @ p5 instead of 8-10cy with cvtepi8+madd)
    __m512i b_biased_u8x64 = _mm512_xor_si512(b_i8x64, xor_mask_u8x64);
    sum_b_biased_i64x8 = _mm512_add_epi64(sum_b_biased_i64x8, _mm512_sad_epu8(b_biased_u8x64, zeros_u8x64));

    if (count_scalars) goto nk_dot_i8_icelake_cycle;

    // Apply algebraic correction: a×b = (a+128)×b - 128×sum(b)
    // sum_b = sum_b_biased - 128×count_rounded
    // correction = 128×sum_b = 128×sum_b_biased - 16384×count_rounded
    nk_i32_t ab_sum = _mm512_reduce_add_epi32(sum_ab_i32x16);
    nk_i64_t sum_b_biased = _mm512_reduce_add_epi64(sum_b_biased_i64x8);
    nk_size_t count_rounded = nk_size_round_up_to_multiple_(count_original, 64);
    nk_i64_t correction = 128LL * sum_b_biased - 16384LL * (nk_i64_t)count_rounded;

    *result = (nk_i32_t)(ab_sum - correction);
}

NK_PUBLIC void nk_dot_u8_icelake(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_u32_t *result) {
    // Optimized u8×u8 dot product using algebraic transformation with DPBUSD
    //
    // Algebraic transformation:
    //   Let b' = b XOR 0x80 (converts unsigned to signed: b' = b - 128)
    //   dpbusd(a, b') computes: a × (b-128)  [unsigned × signed]
    //   Therefore: a×b = a×(b-128) + 128×sum(a)
    //
    // Where:
    //   - XOR with 0x80 converts unsigned u8 [0,255] to signed [-128,127]
    //   - dpbusd performs unsigned×signed multiply-accumulate
    //   - sad_epu8 computes sum(a) as correction term
    //   - Correction term 128×sum(a) is added at the end
    //
    // Performance: 1.92× speedup over unpack + dpwssd approach
    //   - Processes 64 elements/iteration
    //   - Lower latency: ~8cy vs ~16cy per iteration
    //   - Eliminates 4× unpack operations (1cy each @ p5)
    //   - dpbusd@p0 runs in parallel with sad@p5
    //
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const zeros_u8x64 = _mm512_setzero_si512();
    __m512i sum_ab_i32x16 = _mm512_setzero_si512();
    __m512i sum_a_i64x8 = _mm512_setzero_si512();
    __m512i a_u8x64, b_u8x64;

nk_dot_u8_icelake_cycle:
    if (count_scalars < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, count_scalars);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a_scalars);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_si512(a_scalars);
        b_u8x64 = _mm512_loadu_si512(b_scalars);
        a_scalars += 64, b_scalars += 64, count_scalars -= 64;
    }

    // Convert b to signed [-128,127] by XOR with 0x80: b_signed = b - 128
    __m512i b_signed_i8x64 = _mm512_xor_si512(b_u8x64, xor_mask_u8x64);

    // Compute a × (b-128) using dpbusd: unsigned × signed
    sum_ab_i32x16 = _mm512_dpbusd_epi32(sum_ab_i32x16, a_u8x64, b_signed_i8x64);

    // Accumulate sum(a) for correction term using sad_epu8 (1cy @ p5)
    sum_a_i64x8 = _mm512_add_epi64(sum_a_i64x8, _mm512_sad_epu8(a_u8x64, zeros_u8x64));

    if (count_scalars) goto nk_dot_u8_icelake_cycle;

    // Apply algebraic correction: a×b = a×(b-128) + 128×sum(a)
    nk_i32_t ab_dot_signed = _mm512_reduce_add_epi32(sum_ab_i32x16);
    nk_i64_t sum_a = _mm512_reduce_add_epi64(sum_a_i64x8);
    nk_i64_t correction = 128LL * sum_a;

    *result = (nk_u32_t)(ab_dot_signed + correction);
}

typedef struct nk_dot_i8x64_state_icelake_t {
    __m512i sum_ab_i32x16;      // Main dot product sum: (a+128)×b
    __m512i sum_b_biased_i64x8; // Correction term: sum(b+128) for algebraic transform
} nk_dot_i8x64_state_icelake_t;

NK_INTERNAL void nk_dot_i8x64_init_icelake(nk_dot_i8x64_state_icelake_t *state) {
    state->sum_ab_i32x16 = _mm512_setzero_si512();
    state->sum_b_biased_i64x8 = _mm512_setzero_si512();
}

NK_INTERNAL void nk_dot_i8x64_update_icelake(nk_dot_i8x64_state_icelake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Optimized i8×i8 using DPBUSD with algebraic transformation
    // Transform: a×b = (a+128)×b - 128×sum(b)
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const zeros_u8x64 = _mm512_setzero_si512();

    // Load 64 i8 values
    __m512i a_i8x64 = _mm512_loadu_si512(a.u8s);
    __m512i b_i8x64 = _mm512_loadu_si512(b.u8s);

    // Convert a to unsigned: a_biased = a + 128
    __m512i a_biased_u8x64 = _mm512_xor_si512(a_i8x64, xor_mask_u8x64);

    // Compute (a+128) × b using dpbusd
    state->sum_ab_i32x16 = _mm512_dpbusd_epi32(state->sum_ab_i32x16, a_biased_u8x64, b_i8x64);

    // Accumulate sum(b+128) using SAD (1cy @ p5)
    __m512i b_biased_u8x64 = _mm512_xor_si512(b_i8x64, xor_mask_u8x64);
    state->sum_b_biased_i64x8 = _mm512_add_epi64(state->sum_b_biased_i64x8,
                                                 _mm512_sad_epu8(b_biased_u8x64, zeros_u8x64));
}

NK_INTERNAL void nk_dot_i8x64_finalize_icelake(                                               //
    nk_dot_i8x64_state_icelake_t const *state_a, nk_dot_i8x64_state_icelake_t const *state_b, //
    nk_dot_i8x64_state_icelake_t const *state_c, nk_dot_i8x64_state_icelake_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *results) {
    // ILP-optimized 4-way horizontal reduction for i8x64 with algebraic correction
    // For each accumulator: result = sum_ab - 128 × sum_b
    nk_size_t depth_elements_rounded = nk_size_round_up_to_multiple_(total_dimensions, 64);

    // Reduce main dot products (a+128)×b: zmm (i32x16) → ymm (i32x8)
    __m256i sum_ab_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_a->sum_ab_i32x16),
                                              _mm512_extracti32x8_epi32(state_a->sum_ab_i32x16, 1));
    __m256i sum_ab_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_b->sum_ab_i32x16),
                                              _mm512_extracti32x8_epi32(state_b->sum_ab_i32x16, 1));
    __m256i sum_ab_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_c->sum_ab_i32x16),
                                              _mm512_extracti32x8_epi32(state_c->sum_ab_i32x16, 1));
    __m256i sum_ab_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_d->sum_ab_i32x16),
                                              _mm512_extracti32x8_epi32(state_d->sum_ab_i32x16, 1));

    // Reduce correction sums sum(b+128) from i64x8 → i64x4
    __m256i sum_b_a_i64x4 = _mm256_add_epi64(_mm512_castsi512_si256(state_a->sum_b_biased_i64x8),
                                             _mm512_extracti64x4_epi64(state_a->sum_b_biased_i64x8, 1));
    __m256i sum_b_b_i64x4 = _mm256_add_epi64(_mm512_castsi512_si256(state_b->sum_b_biased_i64x8),
                                             _mm512_extracti64x4_epi64(state_b->sum_b_biased_i64x8, 1));
    __m256i sum_b_c_i64x4 = _mm256_add_epi64(_mm512_castsi512_si256(state_c->sum_b_biased_i64x8),
                                             _mm512_extracti64x4_epi64(state_c->sum_b_biased_i64x8, 1));
    __m256i sum_b_d_i64x4 = _mm256_add_epi64(_mm512_castsi512_si256(state_d->sum_b_biased_i64x8),
                                             _mm512_extracti64x4_epi64(state_d->sum_b_biased_i64x8, 1));

    // Reduce i64x4 → i64x2
    __m128i sum_b_a_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(sum_b_a_i64x4),
                                          _mm256_extracti128_si256(sum_b_a_i64x4, 1));
    __m128i sum_b_b_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(sum_b_b_i64x4),
                                          _mm256_extracti128_si256(sum_b_b_i64x4, 1));
    __m128i sum_b_c_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(sum_b_c_i64x4),
                                          _mm256_extracti128_si256(sum_b_c_i64x4, 1));
    __m128i sum_b_d_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(sum_b_d_i64x4),
                                          _mm256_extracti128_si256(sum_b_d_i64x4, 1));

    // Horizontal add i64x2 → single i64 in lane 0
    __m128i sum_b_a_i64x1 = _mm_add_epi64(sum_b_a_i64x2, _mm_shuffle_epi32(sum_b_a_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i sum_b_b_i64x1 = _mm_add_epi64(sum_b_b_i64x2, _mm_shuffle_epi32(sum_b_b_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i sum_b_c_i64x1 = _mm_add_epi64(sum_b_c_i64x2, _mm_shuffle_epi32(sum_b_c_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i sum_b_d_i64x1 = _mm_add_epi64(sum_b_d_i64x2, _mm_shuffle_epi32(sum_b_d_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));

    // Pack 4 i64 sum(b+128) values into __m256i: [sum_a, sum_b, sum_c, sum_d]
    __m256i sum_b_biased_i64x4 = _mm256_set_m128i(_mm_unpacklo_epi64(sum_b_c_i64x1, sum_b_d_i64x1),
                                                  _mm_unpacklo_epi64(sum_b_a_i64x1, sum_b_b_i64x1));

    // Compute correction: result = sum_ab - 128×sum_b = sum_ab - 128×(sum_b_biased - 128×count)
    //                            = sum_ab - 128×sum_b_biased + 16384×count
    __m256i correction_i64x4 = _mm256_slli_epi64(sum_b_biased_i64x4, 7);               // ×128
    __m256i offset_i64x4 = _mm256_set1_epi64x((nk_i64_t)depth_elements_rounded << 14); // 16384×count
    correction_i64x4 = _mm256_sub_epi64(correction_i64x4, offset_i64x4);

    // Extract low 32 bits from each i64 to get i32x4 correction vector
    __m128i correction_i32x4 = _mm256_castsi256_si128(
        _mm256_permutevar8x32_epi32(correction_i64x4, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));

    // Reduce main sums: ymm (i32x8) → xmm (i32x4)
    __m128i sum_ab_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_ab_a_i32x8),
                                           _mm256_extracti128_si256(sum_ab_a_i32x8, 1));
    __m128i sum_ab_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_ab_b_i32x8),
                                           _mm256_extracti128_si256(sum_ab_b_i32x8, 1));
    __m128i sum_ab_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_ab_c_i32x8),
                                           _mm256_extracti128_si256(sum_ab_c_i32x8, 1));
    __m128i sum_ab_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_ab_d_i32x8),
                                           _mm256_extracti128_si256(sum_ab_d_i32x8, 1));

    // Transpose for SIMD reduction
    __m128i transpose_ab_low = _mm_unpacklo_epi32(sum_ab_a_i32x4, sum_ab_b_i32x4);
    __m128i transpose_cd_low = _mm_unpacklo_epi32(sum_ab_c_i32x4, sum_ab_d_i32x4);
    __m128i transpose_ab_high = _mm_unpackhi_epi32(sum_ab_a_i32x4, sum_ab_b_i32x4);
    __m128i transpose_cd_high = _mm_unpackhi_epi32(sum_ab_c_i32x4, sum_ab_d_i32x4);
    __m128i sum_ab_lane0 = _mm_unpacklo_epi64(transpose_ab_low, transpose_cd_low);
    __m128i sum_ab_lane1 = _mm_unpackhi_epi64(transpose_ab_low, transpose_cd_low);
    __m128i sum_ab_lane2 = _mm_unpacklo_epi64(transpose_ab_high, transpose_cd_high);
    __m128i sum_ab_lane3 = _mm_unpackhi_epi64(transpose_ab_high, transpose_cd_high);
    __m128i sum_ab_final = _mm_add_epi32(_mm_add_epi32(sum_ab_lane0, sum_ab_lane1),
                                         _mm_add_epi32(sum_ab_lane2, sum_ab_lane3));

    // Apply algebraic correction: result = sum_ab - correction
    results->xmm = _mm_sub_epi32(sum_ab_final, correction_i32x4);
}

typedef struct nk_dot_u8x64_state_icelake_t {
    __m512i sum_ab_i32x16; // Main dot product sum: a×(b-128)
    __m512i sum_a_i64x8;   // Correction term: sum(a) for algebraic transform
} nk_dot_u8x64_state_icelake_t;

NK_INTERNAL void nk_dot_u8x64_init_icelake(nk_dot_u8x64_state_icelake_t *state) {
    state->sum_ab_i32x16 = _mm512_setzero_si512();
    state->sum_a_i64x8 = _mm512_setzero_si512();
}

NK_INTERNAL void nk_dot_u8x64_update_icelake(nk_dot_u8x64_state_icelake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Optimized u8×u8 using DPBUSD with algebraic transformation
    // Transform: a×b = a×(b-128) + 128×sum(a)
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const zeros_u8x64 = _mm512_setzero_si512();

    __m512i a_u8x64 = _mm512_loadu_si512(a.u8s);
    __m512i b_u8x64 = _mm512_loadu_si512(b.u8s);

    // Convert b to signed: b_signed = b - 128
    __m512i b_signed_i8x64 = _mm512_xor_si512(b_u8x64, xor_mask_u8x64);

    // Compute a × (b-128) using dpbusd
    state->sum_ab_i32x16 = _mm512_dpbusd_epi32(state->sum_ab_i32x16, a_u8x64, b_signed_i8x64);

    // Accumulate sum(a) for correction term using sad_epu8
    state->sum_a_i64x8 = _mm512_add_epi64(state->sum_a_i64x8, _mm512_sad_epu8(a_u8x64, zeros_u8x64));
}

NK_INTERNAL void nk_dot_u8x64_finalize_icelake(                                               //
    nk_dot_u8x64_state_icelake_t const *state_a, nk_dot_u8x64_state_icelake_t const *state_b, //
    nk_dot_u8x64_state_icelake_t const *state_c, nk_dot_u8x64_state_icelake_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // ILP-optimized 4-way horizontal reduction for u32 with algebraic correction
    // For each accumulator: result = sum_ab + 128 × sum_a

    // Reduce main dot products a×(b-128)
    __m256i sum_ab_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_a->sum_ab_i32x16),
                                              _mm512_extracti32x8_epi32(state_a->sum_ab_i32x16, 1));
    __m256i sum_ab_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_b->sum_ab_i32x16),
                                              _mm512_extracti32x8_epi32(state_b->sum_ab_i32x16, 1));
    __m256i sum_ab_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_c->sum_ab_i32x16),
                                              _mm512_extracti32x8_epi32(state_c->sum_ab_i32x16, 1));
    __m256i sum_ab_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_d->sum_ab_i32x16),
                                              _mm512_extracti32x8_epi32(state_d->sum_ab_i32x16, 1));

    // Reduce correction sums sum(a) - i64 to i32 conversion
    __m256i sum_a_a_i64x4 = _mm256_add_epi64(_mm512_castsi512_si256(state_a->sum_a_i64x8),
                                             _mm512_extracti64x4_epi64(state_a->sum_a_i64x8, 1));
    __m256i sum_a_b_i64x4 = _mm256_add_epi64(_mm512_castsi512_si256(state_b->sum_a_i64x8),
                                             _mm512_extracti64x4_epi64(state_b->sum_a_i64x8, 1));
    __m256i sum_a_c_i64x4 = _mm256_add_epi64(_mm512_castsi512_si256(state_c->sum_a_i64x8),
                                             _mm512_extracti64x4_epi64(state_c->sum_a_i64x8, 1));
    __m256i sum_a_d_i64x4 = _mm256_add_epi64(_mm512_castsi512_si256(state_d->sum_a_i64x8),
                                             _mm512_extracti64x4_epi64(state_d->sum_a_i64x8, 1));

    __m128i sum_a_a_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(sum_a_a_i64x4),
                                          _mm256_extracti128_si256(sum_a_a_i64x4, 1));
    __m128i sum_a_b_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(sum_a_b_i64x4),
                                          _mm256_extracti128_si256(sum_a_b_i64x4, 1));
    __m128i sum_a_c_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(sum_a_c_i64x4),
                                          _mm256_extracti128_si256(sum_a_c_i64x4, 1));
    __m128i sum_a_d_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(sum_a_d_i64x4),
                                          _mm256_extracti128_si256(sum_a_d_i64x4, 1));

    // Horizontal add each i64x2 to get single i64 value in lane 0 (stays in SIMD)
    __m128i sum_a_a_i64x1 = _mm_add_epi64(sum_a_a_i64x2, _mm_shuffle_epi32(sum_a_a_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i sum_a_b_i64x1 = _mm_add_epi64(sum_a_b_i64x2, _mm_shuffle_epi32(sum_a_b_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i sum_a_c_i64x1 = _mm_add_epi64(sum_a_c_i64x2, _mm_shuffle_epi32(sum_a_c_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i sum_a_d_i64x1 = _mm_add_epi64(sum_a_d_i64x2, _mm_shuffle_epi32(sum_a_d_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));

    // Pack 4 i64 values into __m256i: [sum_a, sum_b, sum_c, sum_d]
    __m256i sum_a_all_i64x4 = _mm256_set_m128i(_mm_unpacklo_epi64(sum_a_c_i64x1, sum_a_d_i64x1),
                                               _mm_unpacklo_epi64(sum_a_a_i64x1, sum_a_b_i64x1));

    // Multiply by 128 using shift left (stays in i64)
    __m256i correction_i64x4 = _mm256_slli_epi64(sum_a_all_i64x4, 7);

    // Continue reduction to scalar for main sums
    __m128i sum_ab_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_ab_a_i32x8),
                                           _mm256_extracti128_si256(sum_ab_a_i32x8, 1));
    __m128i sum_ab_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_ab_b_i32x8),
                                           _mm256_extracti128_si256(sum_ab_b_i32x8, 1));
    __m128i sum_ab_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_ab_c_i32x8),
                                           _mm256_extracti128_si256(sum_ab_c_i32x8, 1));
    __m128i sum_ab_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_ab_d_i32x8),
                                           _mm256_extracti128_si256(sum_ab_d_i32x8, 1));

    // Transpose for SIMD reduction
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_ab_a_i32x4, sum_ab_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_ab_c_i32x4, sum_ab_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_ab_a_i32x4, sum_ab_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_ab_c_i32x4, sum_ab_d_i32x4);
    __m128i sum_ab_lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_ab_lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_ab_lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_ab_lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_ab_final_i32x4 = _mm_add_epi32(_mm_add_epi32(sum_ab_lane0_i32x4, sum_ab_lane1_i32x4),
                                               _mm_add_epi32(sum_ab_lane2_i32x4, sum_ab_lane3_i32x4));

    // Apply algebraic correction: result = sum_ab + 128 × sum_a
    // Extract low 32 bits from each i64 to form i32x4 correction term
    // Use shuffle to pack: [i64.lo, i64.lo, i64.lo, i64.lo] from 4 i64 values
    __m128i correction_i32x4 = _mm256_castsi256_si128(
        _mm256_permutevar8x32_epi32(correction_i64x4, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
    __m128i final_i32x4 = _mm_add_epi32(sum_ab_final_i32x4, correction_i32x4);
    result->xmm = final_i32x4;
}

NK_PUBLIC void nk_dot_i4_icelake(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result) {
    // i4 values are packed as nibbles: two 4-bit signed values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    //
    // Algorithm: For signed i4, we use an algebraic transformation.
    // Let ax, bx be the unsigned [0,15] representation of signed values a, b in [-8,7].
    // Then: a = ax - 8, b = bx - 8 (the XOR trick gives signed = (unsigned ^ 8) - 8)
    // So: a * b = (ax - 8)(bx - 8) = ax * bx - 8 * ax - 8 * bx + 64
    //
    // We compute ax * bx using DPBUSD, then apply the correction:
    //   signed_dot = unsigned_dot - 8 * (sum_ax + sum_bx) + 64 * n
    //
    // Note: When n is odd, the high nibble of the last byte should be zero-padded.
    //
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8(0x08);
    __m512i const zeros_u8x64 = _mm512_setzero_si512();
    __m512i sum_cd_i32x16 = _mm512_setzero_si512();
    __m512i sum_cx_i64x8 = _mm512_setzero_si512();
    __m512i sum_dx_i64x8 = _mm512_setzero_si512();
    __m512i a_i4x128, b_i4x128;

nk_dot_i4_icelake_cycle:
    if (n_bytes < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
        a_i4x128 = _mm512_maskz_loadu_epi8(mask, a);
        b_i4x128 = _mm512_maskz_loadu_epi8(mask, b);
        n_bytes = 0;
    }
    else {
        a_i4x128 = _mm512_loadu_si512(a);
        b_i4x128 = _mm512_loadu_si512(b);
        a += 64, b += 64, n_bytes -= 64;
    }

    // Extract low and high nibbles
    __m512i a_lo_u8x64 = _mm512_and_si512(a_i4x128, nibble_mask_u8x64);
    __m512i a_hi_u8x64 = _mm512_and_si512(_mm512_srli_epi16(a_i4x128, 4), nibble_mask_u8x64);
    __m512i b_lo_u8x64 = _mm512_and_si512(b_i4x128, nibble_mask_u8x64);
    __m512i b_hi_u8x64 = _mm512_and_si512(_mm512_srli_epi16(b_i4x128, 4), nibble_mask_u8x64);

    // XOR with 8 to get cx, dx values for the algebraic transformation
    __m512i c_lo_u8x64 = _mm512_xor_si512(a_lo_u8x64, xor_mask_u8x64);
    __m512i c_hi_u8x64 = _mm512_xor_si512(a_hi_u8x64, xor_mask_u8x64);
    __m512i d_lo_u8x64 = _mm512_xor_si512(b_lo_u8x64, xor_mask_u8x64);
    __m512i d_hi_u8x64 = _mm512_xor_si512(b_hi_u8x64, xor_mask_u8x64);

    // Compute dot products of cx*dx for low and high nibbles
    sum_cd_i32x16 = _mm512_dpbusd_epi32(sum_cd_i32x16, c_lo_u8x64, d_lo_u8x64);
    sum_cd_i32x16 = _mm512_dpbusd_epi32(sum_cd_i32x16, c_hi_u8x64, d_hi_u8x64);

    // Accumulate sums of cx and dx using SAD against zeros
    sum_cx_i64x8 = _mm512_add_epi64(sum_cx_i64x8, _mm512_sad_epu8(c_lo_u8x64, zeros_u8x64));
    sum_cx_i64x8 = _mm512_add_epi64(sum_cx_i64x8, _mm512_sad_epu8(c_hi_u8x64, zeros_u8x64));
    sum_dx_i64x8 = _mm512_add_epi64(sum_dx_i64x8, _mm512_sad_epu8(d_lo_u8x64, zeros_u8x64));
    sum_dx_i64x8 = _mm512_add_epi64(sum_dx_i64x8, _mm512_sad_epu8(d_hi_u8x64, zeros_u8x64));
    if (n_bytes) goto nk_dot_i4_icelake_cycle;

    // Reduce partial sums and apply algebraic correction
    nk_i32_t cd_dot = _mm512_reduce_add_epi32(sum_cd_i32x16);
    nk_i64_t sum_cx = _mm512_reduce_add_epi64(sum_cx_i64x8);
    nk_i64_t sum_dx = _mm512_reduce_add_epi64(sum_dx_i64x8);
    *result = (nk_i32_t)(cd_dot - 8 * (sum_cx + sum_dx) + 64 * (nk_i64_t)n);
}

NK_PUBLIC void nk_dot_u4_icelake(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    // u4 values are packed as nibbles: two 4-bit unsigned values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    // Values are ∈ [0,15], so DPBUSD can be used directly.
    //
    // Note: When n is odd, the high nibble of the last byte should be zero-padded.
    //
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i sum_i32x16 = _mm512_setzero_si512();

    __m512i a_u4x128, b_u4x128;

nk_dot_u4_icelake_cycle:
    if (n_bytes < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
        a_u4x128 = _mm512_maskz_loadu_epi8(mask, a);
        b_u4x128 = _mm512_maskz_loadu_epi8(mask, b);
        n_bytes = 0;
    }
    else {
        a_u4x128 = _mm512_loadu_si512(a);
        b_u4x128 = _mm512_loadu_si512(b);
        a += 64, b += 64, n_bytes -= 64;
    }

    // Extract low and high nibbles
    __m512i a_lo_u8x64 = _mm512_and_si512(a_u4x128, nibble_mask_u8x64);
    __m512i a_hi_u8x64 = _mm512_and_si512(_mm512_srli_epi16(a_u4x128, 4), nibble_mask_u8x64);
    __m512i b_lo_u8x64 = _mm512_and_si512(b_u4x128, nibble_mask_u8x64);
    __m512i b_hi_u8x64 = _mm512_and_si512(_mm512_srli_epi16(b_u4x128, 4), nibble_mask_u8x64);

    // DPBUSD works directly for u4 since values are ∈ [0,15]
    // and the signed interpretation of [0,15] is the same as unsigned
    sum_i32x16 = _mm512_dpbusd_epi32(sum_i32x16, a_lo_u8x64, b_lo_u8x64);
    sum_i32x16 = _mm512_dpbusd_epi32(sum_i32x16, a_hi_u8x64, b_hi_u8x64);
    if (n_bytes) goto nk_dot_u4_icelake_cycle;

    *result = (nk_u32_t)_mm512_reduce_add_epi32(sum_i32x16);
}

typedef struct nk_dot_i4x128_state_icelake_t {
    __m512i product_sum_i32x16; // Main product: a_biased × b_biased
    __m512i sum_a_biased_i64x8; // Correction term: sum(a_biased)
    __m512i sum_b_biased_i64x8; // Correction term: sum(b_biased)
} nk_dot_i4x128_state_icelake_t;

NK_INTERNAL void nk_dot_i4x128_init_icelake(nk_dot_i4x128_state_icelake_t *state) {
    state->product_sum_i32x16 = _mm512_setzero_si512();
    state->sum_a_biased_i64x8 = _mm512_setzero_si512();
    state->sum_b_biased_i64x8 = _mm512_setzero_si512();
}

NK_INTERNAL void nk_dot_i4x128_update_icelake(nk_dot_i4x128_state_icelake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    // i4 values are packed as nibbles: 128 nibbles in 64 bytes (512 bits)
    // For signed i4, we use algebraic transformation:
    // Signed values in [-8,7] are stored as unsigned [0,15].
    // We XOR with 8 to bias them: a_biased = a_unsigned ^ 8
    // Then: a×b = (a_biased - 8)×(b_biased - 8) = a_biased×b_biased - 8×(a_biased + b_biased) + 64
    //
    // Key optimization: Keep correction terms in vector form, reduce only at finalize time.
    // When active_dimensions < 128, partial_load has zero-padded the unused nibbles.
    // The partial load ensures unused bytes are zero, so the products and sums will be correct.
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const bias_xor_mask_u8x64 = _mm512_set1_epi8(0x08);
    __m512i const zeros_u8x64 = _mm512_setzero_si512();

    // Load 64 bytes containing 128 nibbles (full 512-bit register)
    __m512i a_i4x128 = a.zmm;
    __m512i b_i4x128 = b.zmm;

    // Extract low and high nibbles (all 128 nibbles from 64 bytes)
    __m512i a_lo_u8x64 = _mm512_and_si512(a_i4x128, nibble_mask_u8x64);
    __m512i a_hi_u8x64 = _mm512_and_si512(_mm512_srli_epi16(a_i4x128, 4), nibble_mask_u8x64);
    __m512i b_lo_u8x64 = _mm512_and_si512(b_i4x128, nibble_mask_u8x64);
    __m512i b_hi_u8x64 = _mm512_and_si512(_mm512_srli_epi16(b_i4x128, 4), nibble_mask_u8x64);

    // Apply bias transformation: XOR with 8
    __m512i a_biased_lo_u8x64 = _mm512_xor_si512(a_lo_u8x64, bias_xor_mask_u8x64);
    __m512i a_biased_hi_u8x64 = _mm512_xor_si512(a_hi_u8x64, bias_xor_mask_u8x64);
    __m512i b_biased_lo_u8x64 = _mm512_xor_si512(b_lo_u8x64, bias_xor_mask_u8x64);
    __m512i b_biased_hi_u8x64 = _mm512_xor_si512(b_hi_u8x64, bias_xor_mask_u8x64);

    // Compute dot products of a_biased×b_biased for low and high nibbles
    state->product_sum_i32x16 = _mm512_dpbusd_epi32(state->product_sum_i32x16, a_biased_lo_u8x64, b_biased_lo_u8x64);
    state->product_sum_i32x16 = _mm512_dpbusd_epi32(state->product_sum_i32x16, a_biased_hi_u8x64, b_biased_hi_u8x64);

    // Accumulate sums of biased values using SAD (stays in vector form)
    state->sum_a_biased_i64x8 = _mm512_add_epi64(state->sum_a_biased_i64x8,
                                                 _mm512_sad_epu8(a_biased_lo_u8x64, zeros_u8x64));
    state->sum_a_biased_i64x8 = _mm512_add_epi64(state->sum_a_biased_i64x8,
                                                 _mm512_sad_epu8(a_biased_hi_u8x64, zeros_u8x64));
    state->sum_b_biased_i64x8 = _mm512_add_epi64(state->sum_b_biased_i64x8,
                                                 _mm512_sad_epu8(b_biased_lo_u8x64, zeros_u8x64));
    state->sum_b_biased_i64x8 = _mm512_add_epi64(state->sum_b_biased_i64x8,
                                                 _mm512_sad_epu8(b_biased_hi_u8x64, zeros_u8x64));
}

NK_INTERNAL void nk_dot_i4x128_finalize_icelake(                                                //
    nk_dot_i4x128_state_icelake_t const *state_a, nk_dot_i4x128_state_icelake_t const *state_b, //
    nk_dot_i4x128_state_icelake_t const *state_c, nk_dot_i4x128_state_icelake_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    // ILP-optimized 4-way hierarchical reduction for i4 with algebraic correction.
    // Formula: result = product_sum - 8×(sum_a_biased + sum_b_biased) + 64×depth_nibbles
    //
    // Note: total_dimensions is measured in nibbles (dimensions), not storage bytes.
    //
    // When total_dimensions < 128, partial loads zero-pad unused nibbles to 128.
    // Zero-padded nibbles become biased value 8 after XOR, contributing to sums and products.
    // The offset must account for all processed nibbles (rounded up to 128), not just valid ones.
    nk_size_t depth_nibbles = nk_size_round_up_to_multiple_(total_dimensions, 128);

    // Reduce main products: zmm (i32x16) → ymm (i32x8)
    __m256i product_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_a->product_sum_i32x16),
                                               _mm512_extracti32x8_epi32(state_a->product_sum_i32x16, 1));
    __m256i product_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_b->product_sum_i32x16),
                                               _mm512_extracti32x8_epi32(state_b->product_sum_i32x16, 1));
    __m256i product_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_c->product_sum_i32x16),
                                               _mm512_extracti32x8_epi32(state_c->product_sum_i32x16, 1));
    __m256i product_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_d->product_sum_i32x16),
                                               _mm512_extracti32x8_epi32(state_d->product_sum_i32x16, 1));

    // Reduce ymm (i32x8) → xmm (i32x4)
    __m128i product_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(product_a_i32x8),
                                            _mm256_extracti128_si256(product_a_i32x8, 1));
    __m128i product_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(product_b_i32x8),
                                            _mm256_extracti128_si256(product_b_i32x8, 1));
    __m128i product_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(product_c_i32x8),
                                            _mm256_extracti128_si256(product_c_i32x8, 1));
    __m128i product_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(product_d_i32x8),
                                            _mm256_extracti128_si256(product_d_i32x8, 1));

    // 4-way transpose to get [a,b,c,d] in lanes
    __m128i transpose_ab_low = _mm_unpacklo_epi32(product_a_i32x4, product_b_i32x4);
    __m128i transpose_cd_low = _mm_unpacklo_epi32(product_c_i32x4, product_d_i32x4);
    __m128i transpose_ab_high = _mm_unpackhi_epi32(product_a_i32x4, product_b_i32x4);
    __m128i transpose_cd_high = _mm_unpackhi_epi32(product_c_i32x4, product_d_i32x4);
    __m128i product_lane0 = _mm_unpacklo_epi64(transpose_ab_low, transpose_cd_low);
    __m128i product_lane1 = _mm_unpackhi_epi64(transpose_ab_low, transpose_cd_low);
    __m128i product_lane2 = _mm_unpacklo_epi64(transpose_ab_high, transpose_cd_high);
    __m128i product_lane3 = _mm_unpackhi_epi64(transpose_ab_high, transpose_cd_high);
    __m128i product_final = _mm_add_epi32(_mm_add_epi32(product_lane0, product_lane1),
                                          _mm_add_epi32(product_lane2, product_lane3));

    // Add bias terms together before reduction: sum_total_biased = sum_a_biased + sum_b_biased
    __m512i sum_biased_a_i64x8 = _mm512_add_epi64(state_a->sum_a_biased_i64x8, state_a->sum_b_biased_i64x8);
    __m512i sum_biased_b_i64x8 = _mm512_add_epi64(state_b->sum_a_biased_i64x8, state_b->sum_b_biased_i64x8);
    __m512i sum_biased_c_i64x8 = _mm512_add_epi64(state_c->sum_a_biased_i64x8, state_c->sum_b_biased_i64x8);
    __m512i sum_biased_d_i64x8 = _mm512_add_epi64(state_d->sum_a_biased_i64x8, state_d->sum_b_biased_i64x8);

    // Hierarchical reduction: zmm (i64x8) → ymm (i64x4)
    __m256i sum_biased_a_i64x4 = _mm256_add_epi64(_mm512_castsi512_si256(sum_biased_a_i64x8),
                                                  _mm512_extracti64x4_epi64(sum_biased_a_i64x8, 1));
    __m256i sum_biased_b_i64x4 = _mm256_add_epi64(_mm512_castsi512_si256(sum_biased_b_i64x8),
                                                  _mm512_extracti64x4_epi64(sum_biased_b_i64x8, 1));
    __m256i sum_biased_c_i64x4 = _mm256_add_epi64(_mm512_castsi512_si256(sum_biased_c_i64x8),
                                                  _mm512_extracti64x4_epi64(sum_biased_c_i64x8, 1));
    __m256i sum_biased_d_i64x4 = _mm256_add_epi64(_mm512_castsi512_si256(sum_biased_d_i64x8),
                                                  _mm512_extracti64x4_epi64(sum_biased_d_i64x8, 1));

    // Reduce ymm (i64x4) → xmm (i64x2)
    __m128i sum_biased_a_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(sum_biased_a_i64x4),
                                               _mm256_extracti128_si256(sum_biased_a_i64x4, 1));
    __m128i sum_biased_b_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(sum_biased_b_i64x4),
                                               _mm256_extracti128_si256(sum_biased_b_i64x4, 1));
    __m128i sum_biased_c_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(sum_biased_c_i64x4),
                                               _mm256_extracti128_si256(sum_biased_c_i64x4, 1));
    __m128i sum_biased_d_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(sum_biased_d_i64x4),
                                               _mm256_extracti128_si256(sum_biased_d_i64x4, 1));

    // Horizontal add i64x2 → single i64 in lane 0 (stays in SIMD)
    __m128i sum_biased_a_i64x1 = _mm_add_epi64(sum_biased_a_i64x2,
                                               _mm_shuffle_epi32(sum_biased_a_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i sum_biased_b_i64x1 = _mm_add_epi64(sum_biased_b_i64x2,
                                               _mm_shuffle_epi32(sum_biased_b_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i sum_biased_c_i64x1 = _mm_add_epi64(sum_biased_c_i64x2,
                                               _mm_shuffle_epi32(sum_biased_c_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i sum_biased_d_i64x1 = _mm_add_epi64(sum_biased_d_i64x2,
                                               _mm_shuffle_epi32(sum_biased_d_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));

    // Pack 4 i64 sums into a single YMM register: [sum_a, sum_b, sum_c, sum_d]
    __m256i sum_biased_i64x4 = _mm256_set_m128i(_mm_unpacklo_epi64(sum_biased_c_i64x1, sum_biased_d_i64x1),
                                                _mm_unpacklo_epi64(sum_biased_a_i64x1, sum_biased_b_i64x1));

    // Compute correction: -8 × sum_biased. Stay in i64 to avoid overflow during shift.
    __m256i sum_biased_scaled_i64x4 = _mm256_slli_epi64(sum_biased_i64x4, 3);
    sum_biased_scaled_i64x4 = _mm256_sub_epi64(_mm256_setzero_si256(), sum_biased_scaled_i64x4);

    // Extract low 32 bits from each i64 to get i32x4 correction vector
    __m128i correction_vec = _mm256_castsi256_si128(
        _mm256_permutevar8x32_epi32(sum_biased_scaled_i64x4, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));

    // Apply +64×depth correction (all 4 dot products have same depth)
    nk_i32_t offset = (nk_i32_t)(64 * depth_nibbles);
    __m128i offset_vec = _mm_set1_epi32(offset);

    __m128i final_i32x4 = _mm_add_epi32(_mm_add_epi32(product_final, correction_vec), offset_vec);
    result->xmm = final_i32x4;
}

typedef struct nk_dot_u4x128_state_icelake_t {
    __m512i sum_i32x16; // Direct unsigned accumulator
} nk_dot_u4x128_state_icelake_t;

NK_INTERNAL void nk_dot_u4x128_init_icelake(nk_dot_u4x128_state_icelake_t *state) {
    state->sum_i32x16 = _mm512_setzero_si512();
}

NK_INTERNAL void nk_dot_u4x128_update_icelake(nk_dot_u4x128_state_icelake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // u4 values are packed as nibbles: 128 nibbles in 64 bytes (512 bits)
    // Values are ∈ [0,15], so DPBUSD can be used directly
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);

    // Load 64 bytes containing 128 nibbles (full 512-bit register)
    __m512i a_u4x128 = a.zmm;
    __m512i b_u4x128 = b.zmm;

    // Extract low and high nibbles (all 128 nibbles from 64 bytes)
    __m512i a_lo_u8x64 = _mm512_and_si512(a_u4x128, nibble_mask_u8x64);
    __m512i a_hi_u8x64 = _mm512_and_si512(_mm512_srli_epi16(a_u4x128, 4), nibble_mask_u8x64);
    __m512i b_lo_u8x64 = _mm512_and_si512(b_u4x128, nibble_mask_u8x64);
    __m512i b_hi_u8x64 = _mm512_and_si512(_mm512_srli_epi16(b_u4x128, 4), nibble_mask_u8x64);

    // DPBUSD works directly for u4 since values are ∈ [0,15]
    state->sum_i32x16 = _mm512_dpbusd_epi32(state->sum_i32x16, a_lo_u8x64, b_lo_u8x64);
    state->sum_i32x16 = _mm512_dpbusd_epi32(state->sum_i32x16, a_hi_u8x64, b_hi_u8x64);
}

NK_INTERNAL void nk_dot_u4x128_finalize_icelake(                                                //
    nk_dot_u4x128_state_icelake_t const *state_a, nk_dot_u4x128_state_icelake_t const *state_b, //
    nk_dot_u4x128_state_icelake_t const *state_c, nk_dot_u4x128_state_icelake_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // ILP-optimized 4-way hierarchical reduction for u4 (no correction needed)

    // Reduce zmm (i32x16) → ymm (i32x8)
    __m256i sum_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_a->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_a->sum_i32x16, 1));
    __m256i sum_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_b->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_b->sum_i32x16, 1));
    __m256i sum_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_c->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_c->sum_i32x16, 1));
    __m256i sum_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_d->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_d->sum_i32x16, 1));

    // Reduce ymm (i32x8) → xmm (i32x4)
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_a_i32x8), _mm256_extracti128_si256(sum_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_b_i32x8), _mm256_extracti128_si256(sum_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_c_i32x8), _mm256_extracti128_si256(sum_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_d_i32x8), _mm256_extracti128_si256(sum_d_i32x8, 1));

    // 4-way transpose to get [a,b,c,d] in lanes
    __m128i transpose_ab_low = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i sum_lane0 = _mm_unpacklo_epi64(transpose_ab_low, transpose_cd_low);
    __m128i sum_lane1 = _mm_unpackhi_epi64(transpose_ab_low, transpose_cd_low);
    __m128i sum_lane2 = _mm_unpacklo_epi64(transpose_ab_high, transpose_cd_high);
    __m128i sum_lane3 = _mm_unpackhi_epi64(transpose_ab_high, transpose_cd_high);

    __m128i final_i32x4 = _mm_add_epi32(_mm_add_epi32(sum_lane0, sum_lane1), _mm_add_epi32(sum_lane2, sum_lane3));
    result->xmm = final_i32x4;
}

NK_PUBLIC void nk_dot_e2m3_icelake(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    // Integer dot product for e2m3 using VPERMB (LUT) + VPDPBUSD (unsigned×signed multiply-add).
    // Every e2m3 value × 16 is an exact integer in [-120, +120].
    // Result = i32_dot / 256.0f (exact, no rounding error).
    //
    // LUT maps 5-bit unsigned magnitude to (value × 16):
    //   exp=0 (sub): 2*mant,         exp=1: 16+2*mant
    //   exp=2:       32+4*mant,       exp=3: 64+8*mant
    //
    // VPERMB uses bits [5:0] of the index, so we need a 64-byte LUT with entries 0-31
    // replicated in the upper 32 bytes (VPERMB indexes mod 64, our indices are 0-31).
    // _mm512_set_epi8 lists bytes HIGH→LOW: byte63, byte62, ..., byte0
    __m512i const lut_magnitude_u8x64 = _mm512_set_epi8(120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36,
                                                        32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0,
                                                        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36,
                                                        32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m512i const magnitude_mask_u8x64 = _mm512_set1_epi8(0x1F);
    __m512i const sign_mask_u8x64 = _mm512_set1_epi8(0x20);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i a_e2m3_u8x64, b_e2m3_u8x64;

nk_dot_e2m3_icelake_cycle:
    if (count_scalars < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, count_scalars);
        a_e2m3_u8x64 = _mm512_maskz_loadu_epi8(mask, a_scalars);
        b_e2m3_u8x64 = _mm512_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e2m3_u8x64 = _mm512_loadu_si512(a_scalars);
        b_e2m3_u8x64 = _mm512_loadu_si512(b_scalars);
        a_scalars += 64, b_scalars += 64, count_scalars -= 64;
    }

    // Extract 5-bit magnitude indices
    __m512i a_magnitude_u8x64 = _mm512_and_si512(a_e2m3_u8x64, magnitude_mask_u8x64);
    __m512i b_magnitude_u8x64 = _mm512_and_si512(b_e2m3_u8x64, magnitude_mask_u8x64);

    // VPERMB LUT lookup: unsigned magnitudes × 16
    __m512i a_unsigned_u8x64 = _mm512_permutexvar_epi8(a_magnitude_u8x64, lut_magnitude_u8x64);
    __m512i b_unsigned_u8x64 = _mm512_permutexvar_epi8(b_magnitude_u8x64, lut_magnitude_u8x64);

    // Combined sign: (a ^ b) & 0x20 — nonzero means negative product
    __m512i sign_combined_u8x64 = _mm512_and_si512(_mm512_xor_si512(a_e2m3_u8x64, b_e2m3_u8x64), sign_mask_u8x64);
    __mmask64 negate_mask = _mm512_test_epi8_mask(sign_combined_u8x64, sign_combined_u8x64);

    // Negate b where signs differ: b_signed = negate_mask ? (0 - b_unsigned) : b_unsigned
    // For VPDPBUSD: a=unsigned [0,120], b=signed [-120,+120]
    __m512i b_signed_i8x64 = _mm512_mask_sub_epi8(b_unsigned_u8x64, negate_mask, _mm512_setzero_si512(),
                                                  b_unsigned_u8x64);

    // VPDPBUSD: a_unsigned[unsigned] × b_signed[signed], 4 bytes → i32
    sum_i32x16 = _mm512_dpbusd_epi32(sum_i32x16, a_unsigned_u8x64, b_signed_i8x64);

    if (count_scalars) goto nk_dot_e2m3_icelake_cycle;
    *result = (nk_f32_t)_mm512_reduce_add_epi32(sum_i32x16) / 256.0f;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_ICELAKE
#endif // NK_TARGET_X86_
#endif // NK_DOT_ICELAKE_H
