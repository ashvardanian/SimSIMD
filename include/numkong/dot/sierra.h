/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Sierra Forest CPUs.
 *  @file include/numkong/dot/sierra.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @section avx_vnni_instructions AVX-VNNI Instructions Performance
 *
 *      Intrinsic                   Instruction                     Alder Lake  Raptor Lake
 *      _mm256_dpbusd_epi32         VPDPBUSD (YMM, YMM, YMM)        4cy @ p05   4cy @ p05
 *      _mm256_madd_epi16           VPMADDWD (YMM, YMM, YMM)        4cy @ p05   4cy @ p05
 *      _mm256_sad_epu8             VPSADBW (YMM, YMM, YMM)         3cy @ p5    3cy @ p5
 *
 *  Sierra Forest and mainstream Intel CPUs (Alder Lake, Raptor Lake) support AVX-VNNI (256-bit VNNI)
 *  for accelerated integer dot products. This is the 256-bit variant of AVX-512 VNNI found on Ice Lake.
 *  We use VPDPBUSD for asymmetric unsigned×signed multiplication with algebraic transformations to
 *  handle signed×signed (i8) and unsigned×unsigned (u8) cases.
 *
 *  Performance improvements over previous approaches:
 *    - i8×i8: 1.3-1.4× speedup using dpbusd with XOR transformation (a+128)×b - 128×sum(b)
 *    - u8×u8: 1.8-2.0× speedup using dpbusd with XOR transformation a×(b-128) + 128×sum(a)
 *  These match the speedups achieved on Ice Lake (AVX-512 VNNI) but with 256-bit vectors.
 */
#ifndef NK_DOT_SIERRA_H
#define NK_DOT_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni,avxvnniint8"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni", "avxvnniint8")
#endif

#include "numkong/types.h"
#include "numkong/cast/serial.h"    // nk_partial_load_b8x32_serial_
#include "numkong/reduce/haswell.h" // nk_reduce_add_i32x8_haswell_

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_i8_sierra(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                nk_i32_t *result) {
    // Optimized i8×i8 dot product using algebraic transformation with DPBUSD
    //
    // Algebraic transformation:
    //   Let a' = a XOR 0x80 (interpreted as unsigned, gives a+128 mod 256)
    //   dpbusd(a', b) computes: (a+128) × b  [unsigned × signed]
    //   Therefore: a×b = (a+128)×b - 128×sum(b)
    //
    // Where:
    //   - XOR with 0x80 converts signed i8 [-128,127] to unsigned [0,255]
    //   - dpbusd performs unsigned×signed multiply-accumulate
    //   - Correction term 128×sum(b) is computed and subtracted at the end
    //
    // Performance: ~1.3-1.4× speedup expected over cvtepi8_epi16 + dpwssd approach
    //   - Processes 32 elements/iteration (AVX2 width)
    //   - Lower latency per iteration: 4 cy (VPDPBUSD @ p05) vs 3+4 = 7 cy (VPMOVSXBW @ p5 + VPMADDWD @ p05)
    //   - Better port utilization: VPDPBUSD (p05) runs in parallel with VPMOVSXBW (p5) + VPMADDWD (p05) for correction term,
    //     enabling dual-issue execution on p0 and p5 simultaneously. Old approach bottlenecked on p5 for sign extension.
    //
    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i sum_ab_i32x8 = _mm256_setzero_si256();
    __m256i sum_b_i32x8 = _mm256_setzero_si256();
    __m256i a_i8x32, b_i8x32;

nk_dot_i8_sierra_cycle:
    if (count_scalars < 32) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b8x32_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x32_serial_(b_scalars, &b_vec, count_scalars);
        a_i8x32 = _mm256_load_si256(&a_vec.ymm);
        b_i8x32 = _mm256_load_si256(&b_vec.ymm);
        count_scalars = 0;
    }
    else {
        a_i8x32 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_i8x32 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }

    // Convert a to unsigned [0,255] by XOR with 0x80: a_unsigned = a + 128
    __m256i a_unsigned_u8x32 = _mm256_xor_si256(a_i8x32, xor_mask_u8x32);

    // Compute (a+128) × b using dpbusd: unsigned × signed
    sum_ab_i32x8 = _mm256_dpbusd_epi32(sum_ab_i32x8, a_unsigned_u8x32, b_i8x32);

    // Accumulate sum(b) for correction term using sign-extension
    __m128i b_low_i8x16 = _mm256_castsi256_si128(b_i8x32);
    __m128i b_high_i8x16 = _mm256_extracti128_si256(b_i8x32, 1);
    __m256i b_low_i16x16 = _mm256_cvtepi8_epi16(b_low_i8x16);
    __m256i b_high_i16x16 = _mm256_cvtepi8_epi16(b_high_i8x16);
    __m256i ones_i16x16 = _mm256_set1_epi16(1);
    sum_b_i32x8 = _mm256_add_epi32(sum_b_i32x8, _mm256_madd_epi16(b_low_i16x16, ones_i16x16));
    sum_b_i32x8 = _mm256_add_epi32(sum_b_i32x8, _mm256_madd_epi16(b_high_i16x16, ones_i16x16));

    if (count_scalars) goto nk_dot_i8_sierra_cycle;

    // Apply algebraic correction: a×b = (a+128)×b - 128×sum(b)
    nk_i32_t ab_sum = nk_reduce_add_i32x8_haswell_(sum_ab_i32x8);
    nk_i64_t sum_b = nk_reduce_add_i32x8_haswell_(sum_b_i32x8);
    nk_i64_t correction = 128LL * sum_b;

    *result = (nk_i32_t)(ab_sum - correction);
}

struct nk_dot_i8x32_state_sierra_t {
    __m256i sum_ab_i32x8; // Main dot product sum: (a+128)×b
    __m256i sum_b_i32x8;  // Correction term: sum(b) for algebraic transform
};

NK_INTERNAL void nk_dot_i8x32_init_sierra(nk_dot_i8x32_state_sierra_t *state) {
    state->sum_ab_i32x8 = _mm256_setzero_si256();
    state->sum_b_i32x8 = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_i8x32_update_sierra(nk_dot_i8x32_state_sierra_t *state, nk_b256_vec_t a, nk_b256_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Optimized i8×i8 using DPBUSD with algebraic transformation
    // Transform: a×b = (a+128)×b - 128×sum(b)
    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const ones_i16x16 = _mm256_set1_epi16(1);

    // Load 32 i8 values
    __m256i a_i8x32 = a.ymm;
    __m256i b_i8x32 = b.ymm;

    // Convert a to unsigned: a_unsigned = a + 128
    __m256i a_unsigned_u8x32 = _mm256_xor_si256(a_i8x32, xor_mask_u8x32);

    // Compute (a+128) × b using dpbusd
    state->sum_ab_i32x8 = _mm256_dpbusd_epi32(state->sum_ab_i32x8, a_unsigned_u8x32, b_i8x32);

    // Accumulate sum(b) for correction term using sign-extension
    __m128i b_low_i8x16 = _mm256_castsi256_si128(b_i8x32);
    __m128i b_high_i8x16 = _mm256_extracti128_si256(b_i8x32, 1);
    __m256i b_low_i16x16 = _mm256_cvtepi8_epi16(b_low_i8x16);
    __m256i b_high_i16x16 = _mm256_cvtepi8_epi16(b_high_i8x16);
    state->sum_b_i32x8 = _mm256_add_epi32(state->sum_b_i32x8, _mm256_madd_epi16(b_low_i16x16, ones_i16x16));
    state->sum_b_i32x8 = _mm256_add_epi32(state->sum_b_i32x8, _mm256_madd_epi16(b_high_i16x16, ones_i16x16));
}

NK_INTERNAL void nk_dot_i8x32_finalize_sierra(                                              //
    nk_dot_i8x32_state_sierra_t const *state_a, nk_dot_i8x32_state_sierra_t const *state_b, //
    nk_dot_i8x32_state_sierra_t const *state_c, nk_dot_i8x32_state_sierra_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // ILP-optimized 4-way horizontal reduction for i32 with algebraic correction
    // For each accumulator: result = sum_ab - 128 × sum_b

    // Reduce main dot products (a+128)×b
    __m128i sum_ab_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_a->sum_ab_i32x8),
                                           _mm256_extracti128_si256(state_a->sum_ab_i32x8, 1));
    __m128i sum_ab_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_b->sum_ab_i32x8),
                                           _mm256_extracti128_si256(state_b->sum_ab_i32x8, 1));
    __m128i sum_ab_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_c->sum_ab_i32x8),
                                           _mm256_extracti128_si256(state_c->sum_ab_i32x8, 1));
    __m128i sum_ab_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_d->sum_ab_i32x8),
                                           _mm256_extracti128_si256(state_d->sum_ab_i32x8, 1));

    // Reduce correction sums sum(b)
    __m128i sum_b_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_a->sum_b_i32x8),
                                          _mm256_extracti128_si256(state_a->sum_b_i32x8, 1));
    __m128i sum_b_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_b->sum_b_i32x8),
                                          _mm256_extracti128_si256(state_b->sum_b_i32x8, 1));
    __m128i sum_b_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_c->sum_b_i32x8),
                                          _mm256_extracti128_si256(state_c->sum_b_i32x8, 1));
    __m128i sum_b_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_d->sum_b_i32x8),
                                          _mm256_extracti128_si256(state_d->sum_b_i32x8, 1));

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

    // Transpose and reduce correction terms
    __m128i transpose_b_ab_low_i32x4 = _mm_unpacklo_epi32(sum_b_a_i32x4, sum_b_b_i32x4);
    __m128i transpose_b_cd_low_i32x4 = _mm_unpacklo_epi32(sum_b_c_i32x4, sum_b_d_i32x4);
    __m128i transpose_b_ab_high_i32x4 = _mm_unpackhi_epi32(sum_b_a_i32x4, sum_b_b_i32x4);
    __m128i transpose_b_cd_high_i32x4 = _mm_unpackhi_epi32(sum_b_c_i32x4, sum_b_d_i32x4);
    __m128i sum_b_lane0_i32x4 = _mm_unpacklo_epi64(transpose_b_ab_low_i32x4, transpose_b_cd_low_i32x4);
    __m128i sum_b_lane1_i32x4 = _mm_unpackhi_epi64(transpose_b_ab_low_i32x4, transpose_b_cd_low_i32x4);
    __m128i sum_b_lane2_i32x4 = _mm_unpacklo_epi64(transpose_b_ab_high_i32x4, transpose_b_cd_high_i32x4);
    __m128i sum_b_lane3_i32x4 = _mm_unpackhi_epi64(transpose_b_ab_high_i32x4, transpose_b_cd_high_i32x4);
    __m128i sum_b_final_i32x4 = _mm_add_epi32(_mm_add_epi32(sum_b_lane0_i32x4, sum_b_lane1_i32x4),
                                              _mm_add_epi32(sum_b_lane2_i32x4, sum_b_lane3_i32x4));

    // Apply algebraic correction: result = sum_ab - 128 × sum_b
    __m128i correction_i32x4 = _mm_slli_epi32(sum_b_final_i32x4, 7); // multiply by 128
    __m128i final_i32x4 = _mm_sub_epi32(sum_ab_final_i32x4, correction_i32x4);
    result->xmm = final_i32x4;
}

NK_PUBLIC void nk_dot_u8_sierra(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
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
    // Performance: ~1.8-2.0× speedup expected over unpack + dpwssd approach
    //   - Processes 32 elements/iteration
    //   - Lower latency per iteration
    //   - Eliminates unpack operations
    //   - dpbusd runs in parallel with sad
    //
    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const zeros_u8x32 = _mm256_setzero_si256();
    __m256i sum_ab_i32x8 = _mm256_setzero_si256();
    __m256i sum_a_i64x4 = _mm256_setzero_si256();
    __m256i a_u8x32, b_u8x32;

nk_dot_u8_sierra_cycle:
    if (count_scalars < 32) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b8x32_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x32_serial_(b_scalars, &b_vec, count_scalars);
        a_u8x32 = _mm256_load_si256(&a_vec.ymm);
        b_u8x32 = _mm256_load_si256(&b_vec.ymm);
        count_scalars = 0;
    }
    else {
        a_u8x32 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_u8x32 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }

    // Convert b to signed [-128,127] by XOR with 0x80: b_signed = b - 128
    __m256i b_signed_i8x32 = _mm256_xor_si256(b_u8x32, xor_mask_u8x32);

    // Compute a × (b-128) using dpbusd: unsigned × signed
    sum_ab_i32x8 = _mm256_dpbusd_epi32(sum_ab_i32x8, a_u8x32, b_signed_i8x32);

    // Accumulate sum(a) for correction term using sad_epu8 (1cy @ p5)
    sum_a_i64x4 = _mm256_add_epi64(sum_a_i64x4, _mm256_sad_epu8(a_u8x32, zeros_u8x32));

    if (count_scalars) goto nk_dot_u8_sierra_cycle;

    // Apply algebraic correction: a×b = a×(b-128) + 128×sum(a)
    nk_i32_t ab_dot_signed = nk_reduce_add_i32x8_haswell_(sum_ab_i32x8);

    // Reduce sum_a from 4 i64 values to scalar
    __m128i sum_a_low_i64x2 = _mm256_castsi256_si128(sum_a_i64x4);
    __m128i sum_a_high_i64x2 = _mm256_extracti128_si256(sum_a_i64x4, 1);
    __m128i sum_a_i64x2 = _mm_add_epi64(sum_a_low_i64x2, sum_a_high_i64x2);
    __m128i sum_a_shuffled = _mm_shuffle_epi32(sum_a_i64x2, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i sum_a_final = _mm_add_epi64(sum_a_i64x2, sum_a_shuffled);
    nk_i64_t sum_a = _mm_cvtsi128_si64(sum_a_final);

    nk_i64_t correction = 128LL * sum_a;

    *result = (nk_u32_t)(ab_dot_signed + correction);
}

typedef struct nk_dot_u8x32_state_sierra_t {
    __m256i sum_ab_i32x8; // Main dot product sum: a×(b-128)
    __m256i sum_a_i64x4;  // Correction term: sum(a) for algebraic transform
} nk_dot_u8x32_state_sierra_t;

NK_INTERNAL void nk_dot_u8x32_init_sierra(nk_dot_u8x32_state_sierra_t *state) {
    state->sum_ab_i32x8 = _mm256_setzero_si256();
    state->sum_a_i64x4 = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_u8x32_update_sierra(nk_dot_u8x32_state_sierra_t *state, nk_b256_vec_t a, nk_b256_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Optimized u8×u8 using DPBUSD with algebraic transformation
    // Transform: a×b = a×(b-128) + 128×sum(a)
    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const zeros_u8x32 = _mm256_setzero_si256();

    __m256i a_u8x32 = a.ymm;
    __m256i b_u8x32 = b.ymm;

    // Convert b to signed: b_signed = b - 128
    __m256i b_signed_i8x32 = _mm256_xor_si256(b_u8x32, xor_mask_u8x32);

    // Compute a × (b-128) using dpbusd
    state->sum_ab_i32x8 = _mm256_dpbusd_epi32(state->sum_ab_i32x8, a_u8x32, b_signed_i8x32);

    // Accumulate sum(a) for correction term using sad_epu8
    state->sum_a_i64x4 = _mm256_add_epi64(state->sum_a_i64x4, _mm256_sad_epu8(a_u8x32, zeros_u8x32));
}

NK_INTERNAL void nk_dot_u8x32_finalize_sierra(                                              //
    nk_dot_u8x32_state_sierra_t const *state_a, nk_dot_u8x32_state_sierra_t const *state_b, //
    nk_dot_u8x32_state_sierra_t const *state_c, nk_dot_u8x32_state_sierra_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // ILP-optimized 4-way horizontal reduction for u32 with algebraic correction
    // For each accumulator: result = sum_ab + 128 × sum_a

    // Reduce main dot products a×(b-128)
    __m128i sum_ab_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_a->sum_ab_i32x8),
                                           _mm256_extracti128_si256(state_a->sum_ab_i32x8, 1));
    __m128i sum_ab_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_b->sum_ab_i32x8),
                                           _mm256_extracti128_si256(state_b->sum_ab_i32x8, 1));
    __m128i sum_ab_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_c->sum_ab_i32x8),
                                           _mm256_extracti128_si256(state_c->sum_ab_i32x8, 1));
    __m128i sum_ab_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_d->sum_ab_i32x8),
                                           _mm256_extracti128_si256(state_d->sum_ab_i32x8, 1));

    // Reduce correction sums sum(a) - i64 to i32 conversion
    __m128i sum_a_a_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(state_a->sum_a_i64x4),
                                          _mm256_extracti128_si256(state_a->sum_a_i64x4, 1));
    __m128i sum_a_b_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(state_b->sum_a_i64x4),
                                          _mm256_extracti128_si256(state_b->sum_a_i64x4, 1));
    __m128i sum_a_c_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(state_c->sum_a_i64x4),
                                          _mm256_extracti128_si256(state_c->sum_a_i64x4, 1));
    __m128i sum_a_d_i64x2 = _mm_add_epi64(_mm256_castsi256_si128(state_d->sum_a_i64x4),
                                          _mm256_extracti128_si256(state_d->sum_a_i64x4, 1));

    // Horizontal add each i64x2 to get single i64 value in lane 0 (stays in SIMD)
    __m128i sum_a_a_i64x1 = _mm_add_epi64(sum_a_a_i64x2, _mm_shuffle_epi32(sum_a_a_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i sum_a_b_i64x1 = _mm_add_epi64(sum_a_b_i64x2, _mm_shuffle_epi32(sum_a_b_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i sum_a_c_i64x1 = _mm_add_epi64(sum_a_c_i64x2, _mm_shuffle_epi32(sum_a_c_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i sum_a_d_i64x1 = _mm_add_epi64(sum_a_d_i64x2, _mm_shuffle_epi32(sum_a_d_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));

    // Pack 4 i64 values into __m128i x 2: [sum_a, sum_b] and [sum_c, sum_d]
    __m128i sum_a_ab_i64x2 = _mm_unpacklo_epi64(sum_a_a_i64x1, sum_a_b_i64x1);
    __m128i sum_a_cd_i64x2 = _mm_unpacklo_epi64(sum_a_c_i64x1, sum_a_d_i64x1);

    // Multiply by 128 using shift left (stays in i64)
    __m128i correction_ab_i64x2 = _mm_slli_epi64(sum_a_ab_i64x2, 7);
    __m128i correction_cd_i64x2 = _mm_slli_epi64(sum_a_cd_i64x2, 7);

    // Continue reduction to scalar for main sums
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
    __m128i correction_i32x4_ab = _mm_shuffle_epi32(correction_ab_i64x2, _MM_SHUFFLE(2, 0, 2, 0));
    __m128i correction_i32x4_cd = _mm_shuffle_epi32(correction_cd_i64x2, _MM_SHUFFLE(2, 0, 2, 0));
    __m128i correction_i32x4 = _mm_unpacklo_epi64(correction_i32x4_ab, correction_i32x4_cd);
    __m128i final_u32x4 = _mm_add_epi32(sum_ab_final_i32x4, correction_i32x4);
    result->xmm = final_u32x4;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_

#endif // NK_DOT_SIERRA_H
