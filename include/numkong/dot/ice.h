/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Ice Lake CPUs.
 *  @file include/numkong/dot/ice.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @section vnni_instructions VNNI Instructions Performance
 *
 *      Intrinsic                   Instruction                     Ice         Genoa
 *      _mm512_dpwssd_epi32         VPDPWSSD (ZMM, ZMM, ZMM)        5cy @ p0    4cy @ p01
 *      _mm512_dpbusd_epi32         VPDPBUSD (ZMM, ZMM, ZMM)        5cy @ p0    4cy @ p01
 *      _mm512_madd_epi16           VPMADDWD (ZMM, ZMM, ZMM)        5cy @ p05   3cy @ p01
 *
 *  Ice Lake introduces AVX-512 VNNI for accelerated integer dot products. VNNI instructions bottleneck
 *  on port 0, limiting throughput to 1/cy. AMD Genoa dual-issues on ports 0-1, achieving 0.5/cy throughput.
 *  We use VPDPWSSD for signed i8 inputs after widening to i16, since VPDPBUSD is asymmetric (unsigned x signed).
 */
#ifndef NK_DOT_ICE_H
#define NK_DOT_ICE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICE
#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "f16c", "fma", "bmi", "bmi2")
#endif

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_i8_ice(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                             nk_i32_t *result) {
    __m512i a_i16x32, b_i16x32;
    __m512i sum_i32x16 = _mm512_setzero_si512();

nk_dot_i8_ice_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a_scalars));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b_scalars));
        count_scalars = 0;
    }
    else {
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i const *)a_scalars));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i const *)b_scalars));
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Unfortunately we can't use the `_mm512_dpbusd_epi32` intrinsics here either,
    // as it's asymmetric with respect to the sign of the input arguments:
    //      Signed(ZeroExtend16(a_scalars.byte[4*j]) * SignExtend16(b_scalars.byte[4*j]))
    // So we have to use the `_mm512_dpwssd_epi32` intrinsics instead, upcasting
    // to 16-bit beforehand.
    sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, a_i16x32, b_i16x32);
    if (count_scalars) goto nk_dot_i8_ice_cycle;

    *result = _mm512_reduce_add_epi32(sum_i32x16);
}

NK_PUBLIC void nk_dot_u8_ice(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                             nk_u32_t *result) {
    __m512i a_u8x64, b_u8x64;
    __m512i a_low_i16x32, a_high_i16x32, b_low_i16x32, b_high_i16x32;
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i const zeros_i8x64 = _mm512_setzero_si512();

nk_dot_u8_ice_cycle:
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

    // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
    // instructions instead of extracts, as they are much faster and more efficient.
    a_low_i16x32 = _mm512_unpacklo_epi8(a_u8x64, zeros_i8x64);
    a_high_i16x32 = _mm512_unpackhi_epi8(a_u8x64, zeros_i8x64);
    b_low_i16x32 = _mm512_unpacklo_epi8(b_u8x64, zeros_i8x64);
    b_high_i16x32 = _mm512_unpackhi_epi8(b_u8x64, zeros_i8x64);
    // Unfortunately we can't use the `_mm512_dpbusd_epi32` intrinsics here either,
    // as it's asymmetric with respect to the sign of the input arguments:
    //      Signed(ZeroExtend16(a.byte[4*j]) * SignExtend16(b.byte[4*j]))
    // So we have to use the `_mm512_dpwssd_epi32` intrinsics instead, upcasting
    // to 16-bit beforehand.
    sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, a_low_i16x32, b_low_i16x32);
    sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, a_high_i16x32, b_high_i16x32);
    if (count_scalars) goto nk_dot_u8_ice_cycle;

    *result = (nk_u32_t)_mm512_reduce_add_epi32(sum_i32x16);
}

typedef struct nk_dot_i8x32_state_ice_t {
    __m512i sum_i32x16;
} nk_dot_i8x32_state_ice_t;

NK_INTERNAL void nk_dot_i8x32_init_ice(nk_dot_i8x32_state_ice_t *state) { state->sum_i32x16 = _mm512_setzero_si512(); }

NK_INTERNAL void nk_dot_i8x32_update_ice(nk_dot_i8x32_state_ice_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    __m512i a_i16x32 = _mm512_cvtepi8_epi16(a.ymm);
    __m512i b_i16x32 = _mm512_cvtepi8_epi16(b.ymm);
    state->sum_i32x16 = _mm512_dpwssd_epi32(state->sum_i32x16, a_i16x32, b_i16x32);
}

NK_INTERNAL void nk_dot_i8x32_finalize_ice(                                           //
    nk_dot_i8x32_state_ice_t const *state_a, nk_dot_i8x32_state_ice_t const *state_b, //
    nk_dot_i8x32_state_ice_t const *state_c, nk_dot_i8x32_state_ice_t const *state_d, //
    nk_b128_vec_t *results) {
    // ILP-optimized 4-way horizontal reduction for i32
    __m256i sum_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_a->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_a->sum_i32x16, 1));
    __m256i sum_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_b->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_b->sum_i32x16, 1));
    __m256i sum_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_c->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_c->sum_i32x16, 1));
    __m256i sum_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_d->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_d->sum_i32x16, 1));
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_a_i32x8), _mm256_extracti128_si256(sum_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_b_i32x8), _mm256_extracti128_si256(sum_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_c_i32x8), _mm256_extracti128_si256(sum_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_d_i32x8), _mm256_extracti128_si256(sum_d_i32x8, 1));
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i sum_lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    results->xmm = _mm_add_epi32(_mm_add_epi32(sum_lane0_i32x4, sum_lane1_i32x4),
                                 _mm_add_epi32(sum_lane2_i32x4, sum_lane3_i32x4));
}

typedef struct nk_dot_u8x64_state_ice_t {
    __m512i sum_i32x16;
} nk_dot_u8x64_state_ice_t;

NK_INTERNAL void nk_dot_u8x64_init_ice(nk_dot_u8x64_state_ice_t *state) { state->sum_i32x16 = _mm512_setzero_si512(); }

NK_INTERNAL void nk_dot_u8x64_update_ice(nk_dot_u8x64_state_ice_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    __m512i sum_i32x16 = state->sum_i32x16;
    __m512i const zeros_i8x64 = _mm512_setzero_si512();

    __m512i a_u8x64 = _mm512_loadu_si512(a.u8s);
    __m512i b_u8x64 = _mm512_loadu_si512(b.u8s);
    // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
    // instructions instead of extracts, as they are much faster and more efficient.
    __m512i a_low_i16x32 = _mm512_unpacklo_epi8(a_u8x64, zeros_i8x64);
    __m512i a_high_i16x32 = _mm512_unpackhi_epi8(a_u8x64, zeros_i8x64);
    __m512i b_low_i16x32 = _mm512_unpacklo_epi8(b_u8x64, zeros_i8x64);
    __m512i b_high_i16x32 = _mm512_unpackhi_epi8(b_u8x64, zeros_i8x64);
    // Unfortunately we can't use the `_mm512_dpbusd_epi32` intrinsics here either,
    // as it's asymmetric with respect to the sign of the input arguments:
    //      Signed(ZeroExtend16(a.byte[4*j]) * SignExtend16(b.byte[4*j]))
    // So we have to use the `_mm512_dpwssd_epi32` intrinsics instead, upcasting
    // to 16-bit beforehand.
    sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, a_low_i16x32, b_low_i16x32);
    state->sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, a_high_i16x32, b_high_i16x32);
}

NK_INTERNAL void nk_dot_u8x64_finalize_ice(                                           //
    nk_dot_u8x64_state_ice_t const *state_a, nk_dot_u8x64_state_ice_t const *state_b, //
    nk_dot_u8x64_state_ice_t const *state_c, nk_dot_u8x64_state_ice_t const *state_d, //
    nk_b128_vec_t *results) {
    // ILP-optimized 4-way horizontal reduction for u32
    __m256i sum_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_a->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_a->sum_i32x16, 1));
    __m256i sum_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_b->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_b->sum_i32x16, 1));
    __m256i sum_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_c->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_c->sum_i32x16, 1));
    __m256i sum_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_d->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_d->sum_i32x16, 1));
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_a_i32x8), _mm256_extracti128_si256(sum_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_b_i32x8), _mm256_extracti128_si256(sum_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_c_i32x8), _mm256_extracti128_si256(sum_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_d_i32x8), _mm256_extracti128_si256(sum_d_i32x8, 1));
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i sum_lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    results->xmm = _mm_add_epi32(_mm_add_epi32(sum_lane0_i32x4, sum_lane1_i32x4),
                                 _mm_add_epi32(sum_lane2_i32x4, sum_lane3_i32x4));
}

NK_PUBLIC void nk_dot_i4_ice(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result) {
    // i4 values are packed as nibbles: two 4-bit signed values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    //
    // Algorithm: For signed i4, we use an algebraic transformation.
    // Let ax, bx be the unsigned [0,15] representation of signed values a, b in [-8,7].
    // Then: a = ax - 8, b = bx - 8 (the XOR trick gives signed = (unsigned ^ 8) - 8)
    // So: a * b = (ax - 8)(bx - 8) = ax*bx - 8*ax - 8*bx + 64
    //
    // We compute ax*bx using DPBUSD, then apply the correction:
    //   signed_dot = unsigned_dot - 8*(sum_ax + sum_bx) + 64*n
    //
    // Note: When n is odd, the high nibble of the last byte should be zero-padded.
    //
    nk_size_t n_bytes = (n + 1) / 2;
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8(0x08);
    __m512i const zeros_u8x64 = _mm512_setzero_si512();
    __m512i sum_cd_i32x16 = _mm512_setzero_si512();
    __m512i sum_cx_i64x8 = _mm512_setzero_si512();
    __m512i sum_dx_i64x8 = _mm512_setzero_si512();

    // For signed i4 values, the transformation is: signed_val = (raw ^ 8) - 8
    // Let cx = ax ^ 8 and dx = bx ^ 8, then:
    // a * b = (cx - 8)(dx - 8) = cx*dx - 8*cx - 8*dx + 64
    // So: Σ(a*b) = Σ(cx*dx) - 8*Σcx - 8*Σdx + 64*n

    __m512i a_i4x128, b_i4x128;

nk_dot_i4_ice_cycle:
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
    if (n_bytes) goto nk_dot_i4_ice_cycle;

    // Reduce partial sums and apply algebraic correction
    nk_i32_t cd_dot = _mm512_reduce_add_epi32(sum_cd_i32x16);
    nk_i64_t sum_cx = _mm512_reduce_add_epi64(sum_cx_i64x8);
    nk_i64_t sum_dx = _mm512_reduce_add_epi64(sum_dx_i64x8);
    *result = (nk_i32_t)(cd_dot - 8 * (sum_cx + sum_dx) + 64 * (nk_i64_t)n);
}

NK_PUBLIC void nk_dot_u4_ice(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    // u4 values are packed as nibbles: two 4-bit unsigned values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    // Values are ∈ [0,15], so DPBUSD can be used directly.
    //
    // Note: When n is odd, the high nibble of the last byte should be zero-padded.
    //
    nk_size_t n_bytes = (n + 1) / 2;
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i sum_i32x16 = _mm512_setzero_si512();

    __m512i a_u4x128, b_u4x128;

nk_dot_u4_ice_cycle:
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
    if (n_bytes) goto nk_dot_u4_ice_cycle;

    *result = (nk_u32_t)_mm512_reduce_add_epi32(sum_i32x16);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_ICE
#endif // NK_TARGET_X86_

#endif // NK_DOT_ICE_H
