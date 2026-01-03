/**
 *  @brief SIMD-accelerated type conversions for FP8/BF16/F16 types optimized for Intel Haswell CPUs.
 *  @file include/numkong/cast/haswell.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 */
#ifndef NK_CAST_HASWELL_H
#define NK_CAST_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi2"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_partial_load_b16x16_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Convert f32 scalar to f16 bit pattern using F16C. */
NK_PUBLIC void nk_f32_to_f16_haswell(nk_f32_t const *from, nk_f16_t *to) {
    *to = _mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(*from), _MM_FROUND_TO_NEAREST_INT));
}

/** @brief Convert f16 bit pattern to f32 scalar using F16C. */
NK_PUBLIC void nk_f16_to_f32_haswell(nk_f16_t const *from, nk_f32_t *to) {
    *to = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(*from)));
}

#pragma region - Type Punned Loads and Stores

/** @brief Type-agnostic 256-bit full load (Haswell AVX2). */
NK_INTERNAL void nk_load_b256_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm = _mm256_loadu_si256((const __m256i *)src);
}

/** @brief Type-agnostic 128-bit full load (Haswell AVX2). */
NK_INTERNAL void nk_load_b128_haswell_(void const *src, nk_b128_vec_t *dst) {
    dst->xmm = _mm_loadu_si128((const __m128i *)src);
}

#pragma endregion - Type Punned Loads and Stores

#pragma region - Vectorized Conversions

/** @brief Convert 8x bf16 to 8x f32 by shifting left 16 bits (AVX2). */
NK_INTERNAL __m256 nk_bf16x8_to_f32x8_haswell_(__m128i bf16_i16x8) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_i16x8), 16));
}

/** @brief Convert 16x e4m3 to 16x bf16 via arithmetic + small LUT for subnormals (AVX2).
 *  E4M3 format: S EEEE MMM (bias=7). BF16: S EEEEEEEE MMMMMMM (bias=127).
 *  Normal values: BF16 = sign | ((lower7 << 4) + 0x3C00).
 *  Subnormals (8 values): looked up via vpshufb from an 8-entry LUT.
 *  Handles all corner cases: zero, subnormals, normals, and NaN. */
NK_INTERNAL __m256i nk_e4m3x16_to_bf16x16_haswell_(__m128i e4m3x16) {
    __m256i e4m3_i16x16 = _mm256_cvtepu8_epi16(e4m3x16);
    __m256i sign_i16x16 = _mm256_and_si256(e4m3_i16x16, _mm256_set1_epi16((short)0x80));
    __m256i lower7_i16x16 = _mm256_and_si256(e4m3_i16x16, _mm256_set1_epi16(0x7F));

    // Normal path: BF16 = ((lower7 << 4) + 0x3C00) | (sign << 8)
    __m256i normal_abs_i16x16 = _mm256_add_epi16(_mm256_slli_epi16(lower7_i16x16, 4), _mm256_set1_epi16(0x3C00));
    sign_i16x16 = _mm256_slli_epi16(sign_i16x16, 8);
    __m256i normal_i16x16 = _mm256_or_si256(sign_i16x16, normal_abs_i16x16);

    // Subnormal LUT via shuffle_epi8 (8 entries: mantissa 0-7 -> BF16)
    // E4M3 subnormal BF16 values: 0x0000, 0x3B00, 0x3B80, 0x3BC0, 0x3C00, 0x3C20, 0x3C40, 0x3C60
    // Split into low bytes and high bytes for reconstruction
    __m256i const lo_lut_i8x32 = _mm256_broadcastsi128_si256(_mm_set_epi8( //
        0x60, 0x40, 0x20, 0x00, (char)0xC0, (char)0x80, 0x00, 0x00,        //
        0x60, 0x40, 0x20, 0x00, (char)0xC0, (char)0x80, 0x00, 0x00));      //
    __m256i const hi_lut_i8x32 = _mm256_broadcastsi128_si256(_mm_set_epi8( //
        0x3C, 0x3C, 0x3C, 0x3C, 0x3B, 0x3B, 0x3B, 0x00,                    //
        0x3C, 0x3C, 0x3C, 0x3C, 0x3B, 0x3B, 0x3B, 0x00));                  //

    // Extract mantissa (bits 0-2) as byte indices for shuffle
    __m256i byte_idx_i8x32 = _mm256_and_si256(e4m3_i16x16, _mm256_set1_epi8(0x07));
    __m256i lo_bytes_i8x32 = _mm256_shuffle_epi8(lo_lut_i8x32, byte_idx_i8x32);
    __m256i hi_bytes_i8x32 = _mm256_shuffle_epi8(hi_lut_i8x32, byte_idx_i8x32);

    // Combine low and high bytes into 16-bit values
    __m256i subnorm_abs_i16x16 = _mm256_or_si256(                    //
        _mm256_and_si256(lo_bytes_i8x32, _mm256_set1_epi16(0x00FF)), //
        _mm256_slli_epi16(hi_bytes_i8x32, 8));                       //
    __m256i subnorm_i16x16 = _mm256_or_si256(subnorm_abs_i16x16, sign_i16x16);

    // Blend: if exponent == 0, use subnormal result; else use normal result
    __m256i exp_bits_i16x16 = _mm256_and_si256(e4m3_i16x16, _mm256_set1_epi16(0x78));
    __m256i is_subnormal_i16x16 = _mm256_cmpeq_epi16(exp_bits_i16x16, _mm256_setzero_si256());
    __m256i result_i16x16 = _mm256_blendv_epi8(normal_i16x16, subnorm_i16x16, is_subnormal_i16x16);

    // Handle NaN: E4M3 index 127 (0x7F) -> BF16 NaN (0x7FC0)
    __m256i is_nan_i16x16 = _mm256_cmpeq_epi16(lower7_i16x16, _mm256_set1_epi16(0x7F));
    __m256i nan_i16x16 = _mm256_or_si256(sign_i16x16, _mm256_set1_epi16(0x7FC0));
    return _mm256_blendv_epi8(result_i16x16, nan_i16x16, is_nan_i16x16);
}

/** @brief Convert 16x e5m2 to 16x bf16 via arithmetic + small LUT for subnormals (AVX2).
 *  E5M2 format: S EEEEE MM (bias=15). BF16: S EEEEEEEE MMMMMMM (bias=127).
 *  Normal values: BF16 = sign | ((lower7 << 5) + 0x3800).
 *  Subnormals (4 values): looked up via vpshufb from a 4-entry LUT.
 *  Handles all corner cases: zero, subnormals, normals, infinity, and NaN. */
NK_INTERNAL __m256i nk_e5m2x16_to_bf16x16_haswell_(__m128i e5m2x16) {
    __m256i e5m2_i16x16 = _mm256_cvtepu8_epi16(e5m2x16);
    __m256i sign_i16x16 = _mm256_and_si256(e5m2_i16x16, _mm256_set1_epi16((short)0x80));
    __m256i lower7_i16x16 = _mm256_and_si256(e5m2_i16x16, _mm256_set1_epi16(0x7F));

    // Normal path: BF16 = ((lower7 << 5) + 0x3800) | (sign << 8)
    __m256i normal_abs_i16x16 = _mm256_add_epi16(_mm256_slli_epi16(lower7_i16x16, 5), _mm256_set1_epi16(0x3800));
    sign_i16x16 = _mm256_slli_epi16(sign_i16x16, 8);
    __m256i normal_i16x16 = _mm256_or_si256(sign_i16x16, normal_abs_i16x16);

    // Subnormal LUT via shuffle_epi8 (4 entries: mantissa 0-3 -> BF16)
    // E5M2 subnormal BF16 values: 0x0000, 0x3780, 0x3800, 0x3840
    __m256i const lo_lut_i8x32 = _mm256_broadcastsi128_si256(_mm_set_epi8( //
        0x00, 0x00, 0x00, 0x00, 0x40, 0x00, (char)0x80, 0x00,              //
        0x00, 0x00, 0x00, 0x00, 0x40, 0x00, (char)0x80, 0x00));            //
    __m256i const hi_lut_i8x32 = _mm256_broadcastsi128_si256(_mm_set_epi8( //
        0x00, 0x00, 0x00, 0x00, 0x38, 0x38, 0x37, 0x00,                    //
        0x00, 0x00, 0x00, 0x00, 0x38, 0x38, 0x37, 0x00));                  //

    // Extract mantissa (bits 0-1) as byte indices for shuffle
    __m256i byte_idx_i8x32 = _mm256_and_si256(e5m2_i16x16, _mm256_set1_epi8(0x03));
    __m256i lo_bytes_i8x32 = _mm256_shuffle_epi8(lo_lut_i8x32, byte_idx_i8x32);
    __m256i hi_bytes_i8x32 = _mm256_shuffle_epi8(hi_lut_i8x32, byte_idx_i8x32);

    // Combine low and high bytes into 16-bit values
    __m256i subnorm_abs_i16x16 = _mm256_or_si256(                    //
        _mm256_and_si256(lo_bytes_i8x32, _mm256_set1_epi16(0x00FF)), //
        _mm256_slli_epi16(hi_bytes_i8x32, 8));                       //
    __m256i subnorm_i16x16 = _mm256_or_si256(subnorm_abs_i16x16, sign_i16x16);

    // Blend: if exponent == 0, use subnormal result; else use normal result
    __m256i exp_bits_i16x16 = _mm256_and_si256(e5m2_i16x16, _mm256_set1_epi16(0x7C));
    __m256i is_subnormal_i16x16 = _mm256_cmpeq_epi16(exp_bits_i16x16, _mm256_setzero_si256());
    __m256i result_i16x16 = _mm256_blendv_epi8(normal_i16x16, subnorm_i16x16, is_subnormal_i16x16);

    // Handle Inf (0x7C) and NaN (0x7D-0x7F)
    __m256i is_inf_i16x16 = _mm256_cmpeq_epi16(lower7_i16x16, _mm256_set1_epi16(0x7C));
    __m256i is_nan_i16x16 = _mm256_cmpgt_epi16(lower7_i16x16, _mm256_set1_epi16(0x7C));
    __m256i inf_i16x16 = _mm256_or_si256(sign_i16x16, _mm256_set1_epi16(0x7F80));
    __m256i nan_i16x16 = _mm256_or_si256(sign_i16x16, _mm256_set1_epi16(0x7FC0));
    result_i16x16 = _mm256_blendv_epi8(result_i16x16, inf_i16x16, is_inf_i16x16);
    return _mm256_blendv_epi8(result_i16x16, nan_i16x16, is_nan_i16x16);
}

/** @brief Convert 16x e4m3 to 16x f16 via arithmetic + small LUT for subnormals (AVX2).
 *  E4M3 format: S EEEE MMM (bias=7). F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Normal values: F16 = sign | ((lower7 << 7) + 0x2000).
 *  Subnormals (8 values): looked up via vpshufb from an 8-entry LUT.
 *  Handles all corner cases: zero, subnormals, normals, and NaN. */
NK_INTERNAL __m256i nk_e4m3x16_to_f16x16_haswell_(__m128i e4m3x16) {
    __m256i e4m3_i16x16 = _mm256_cvtepu8_epi16(e4m3x16);
    __m256i sign_i16x16 = _mm256_and_si256(e4m3_i16x16, _mm256_set1_epi16((short)0x80));
    __m256i lower7_i16x16 = _mm256_and_si256(e4m3_i16x16, _mm256_set1_epi16(0x7F));

    // Normal path: F16 = ((lower7 << 7) + 0x2000) | (sign << 8)
    __m256i normal_abs_i16x16 = _mm256_add_epi16(_mm256_slli_epi16(lower7_i16x16, 7), _mm256_set1_epi16(0x2000));
    sign_i16x16 = _mm256_slli_epi16(sign_i16x16, 8);
    __m256i normal_i16x16 = _mm256_or_si256(sign_i16x16, normal_abs_i16x16);

    // Subnormal LUT via shuffle_epi8 (8 entries: mantissa 0-7 -> F16)
    // E4M3 subnormal F16 values: 0x0000, 0x1800, 0x1C00, 0x1E00, 0x2000, 0x2100, 0x2200, 0x2300
    // All low bytes are 0x00, high bytes: 0x00, 0x18, 0x1C, 0x1E, 0x20, 0x21, 0x22, 0x23
    // _mm_set_epi8 order: b15..b8 (unused), b7=idx7, b6=idx6, ..., b0=idx0
    __m256i const lo_lut_i8x32 = _mm256_setzero_si256();
    __m256i const hi_lut_i8x32 = _mm256_broadcastsi128_si256(_mm_set_epi8( //
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,                    //
        0x23, 0x22, 0x21, 0x20, 0x1E, 0x1C, 0x18, 0x00));                  //

    // Extract mantissa (bits 0-2) as byte indices for shuffle
    __m256i byte_idx_i8x32 = _mm256_and_si256(e4m3_i16x16, _mm256_set1_epi8(0x07));
    __m256i lo_bytes_i8x32 = _mm256_shuffle_epi8(lo_lut_i8x32, byte_idx_i8x32);
    __m256i hi_bytes_i8x32 = _mm256_shuffle_epi8(hi_lut_i8x32, byte_idx_i8x32);

    // Combine low and high bytes into 16-bit values
    __m256i subnorm_abs_i16x16 = _mm256_or_si256(                    //
        _mm256_and_si256(lo_bytes_i8x32, _mm256_set1_epi16(0x00FF)), //
        _mm256_slli_epi16(hi_bytes_i8x32, 8));                       //
    __m256i subnorm_i16x16 = _mm256_or_si256(subnorm_abs_i16x16, sign_i16x16);

    // Blend: if exponent == 0, use subnormal result; else use normal result
    __m256i exp_bits_i16x16 = _mm256_and_si256(e4m3_i16x16, _mm256_set1_epi16(0x78));
    __m256i is_subnormal_i16x16 = _mm256_cmpeq_epi16(exp_bits_i16x16, _mm256_setzero_si256());
    __m256i result_i16x16 = _mm256_blendv_epi8(normal_i16x16, subnorm_i16x16, is_subnormal_i16x16);

    // Handle NaN: E4M3 index 127 (0x7F) -> F16 NaN (0x7E00)
    __m256i is_nan_i16x16 = _mm256_cmpeq_epi16(lower7_i16x16, _mm256_set1_epi16(0x7F));
    __m256i nan_i16x16 = _mm256_or_si256(sign_i16x16, _mm256_set1_epi16(0x7E00));
    return _mm256_blendv_epi8(result_i16x16, nan_i16x16, is_nan_i16x16);
}

/** @brief Convert 16x e5m2 to 16x f16 via simple bit shift (AVX2).
 *  E5M2 format: S EEEEE MM (bias=15). F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Same exponent bias means F16 = (lower7 << 8) | (sign << 15).
 *  Handles all corner cases: zero, subnormals, normals, infinity, and NaN. */
NK_INTERNAL __m256i nk_e5m2x16_to_f16x16_haswell_(__m128i e5m2x16) {
    __m256i e5m2_i16x16 = _mm256_cvtepu8_epi16(e5m2x16);
    __m256i sign_i16x16 = _mm256_and_si256(e5m2_i16x16, _mm256_set1_epi16((short)0x80));
    __m256i lower7_i16x16 = _mm256_and_si256(e5m2_i16x16, _mm256_set1_epi16(0x7F));

    // F16 = (lower7 << 8) | (sign << 15)
    // Works for all cases: subnormals, normals, infinity, and NaN
    __m256i result_i16x16 = _mm256_slli_epi16(lower7_i16x16, 8);
    sign_i16x16 = _mm256_slli_epi16(sign_i16x16, 8);
    return _mm256_or_si256(result_i16x16, sign_i16x16);
}

/** @brief Convert 8x e4m3 to 8x f32 via bit manipulation (AVX2).
 *  E4M3 format: S EEEE MMM (bias=7). F32: sign<<31, (exp+120)<<23, mant<<20.
 *  Subnormals (exp=0): value = mantissa * 2^(1-7) * 2^(-3) = mantissa / 512. */
NK_INTERNAL __m256 nk_e4m3x8_to_f32x8_haswell_(__m128i e4m3_i8x8) {
    __m256i e4m3_i32x8 = _mm256_cvtepu8_epi32(e4m3_i8x8);

    // Extract fields
    __m256i exp_i32x8 = _mm256_and_si256(_mm256_srli_epi32(e4m3_i32x8, 3), _mm256_set1_epi32(0x0F));
    __m256i mant_i32x8 = _mm256_and_si256(e4m3_i32x8, _mm256_set1_epi32(0x07));

    // Build F32 sign bit
    __m256i f32_sign_i32x8 = _mm256_slli_epi32(_mm256_srli_epi32(e4m3_i32x8, 7), 31);

    // Normal path: sign | ((exp+120)<<23) | (mant<<20)
    __m256i f32_exp_i32x8 = _mm256_slli_epi32(_mm256_add_epi32(exp_i32x8, _mm256_set1_epi32(120)), 23);
    __m256i f32_mant_i32x8 = _mm256_slli_epi32(mant_i32x8, 20);
    __m256i normal_bits_i32x8 = _mm256_or_si256(f32_sign_i32x8, _mm256_or_si256(f32_exp_i32x8, f32_mant_i32x8));

    // Subnormal path: value = mantissa / 512.0f, then apply sign
    __m256 subnorm_abs_f32x8 = _mm256_mul_ps(_mm256_cvtepi32_ps(mant_i32x8), _mm256_set1_ps(1.0f / 512.0f));
    __m256 subnorm_f32x8 = _mm256_or_ps(subnorm_abs_f32x8, _mm256_castsi256_ps(f32_sign_i32x8));

    // Blend: if exp==0, use subnormal result; otherwise use normal bits
    __m256i exp_zero_mask = _mm256_cmpeq_epi32(exp_i32x8, _mm256_setzero_si256());
    return _mm256_blendv_ps(_mm256_castsi256_ps(normal_bits_i32x8), subnorm_f32x8, _mm256_castsi256_ps(exp_zero_mask));
}

/** @brief Convert 8x e5m2 to 8x f32 via bit manipulation (AVX2).
 *  E5M2 format: S EEEEE MM (bias=15). F32: sign<<31, (exp+112)<<23, mant<<21.
 *  Subnormals (exp=0): value = mantissa * 2^(1-15) * 2^(-2) = mantissa / 65536. */
NK_INTERNAL __m256 nk_e5m2x8_to_f32x8_haswell_(__m128i e5m2_i8x8) {
    __m256i e5m2_i32x8 = _mm256_cvtepu8_epi32(e5m2_i8x8);

    // Extract fields
    __m256i exp_i32x8 = _mm256_and_si256(_mm256_srli_epi32(e5m2_i32x8, 2), _mm256_set1_epi32(0x1F));
    __m256i mant_i32x8 = _mm256_and_si256(e5m2_i32x8, _mm256_set1_epi32(0x03));

    // Build F32 sign bit
    __m256i f32_sign_i32x8 = _mm256_slli_epi32(_mm256_srli_epi32(e5m2_i32x8, 7), 31);

    // Normal path: sign | ((exp+112)<<23) | (mant<<21)
    __m256i f32_exp_i32x8 = _mm256_slli_epi32(_mm256_add_epi32(exp_i32x8, _mm256_set1_epi32(112)), 23);
    __m256i f32_mant_i32x8 = _mm256_slli_epi32(mant_i32x8, 21);
    __m256i normal_bits_i32x8 = _mm256_or_si256(f32_sign_i32x8, _mm256_or_si256(f32_exp_i32x8, f32_mant_i32x8));

    // Subnormal path: value = mantissa / 65536.0f, then apply sign
    __m256 subnorm_abs_f32x8 = _mm256_mul_ps(_mm256_cvtepi32_ps(mant_i32x8), _mm256_set1_ps(1.0f / 65536.0f));
    __m256 subnorm_f32x8 = _mm256_or_ps(subnorm_abs_f32x8, _mm256_castsi256_ps(f32_sign_i32x8));

    // Blend: if exp==0, use subnormal result; otherwise use normal bits
    __m256i exp_zero_mask = _mm256_cmpeq_epi32(exp_i32x8, _mm256_setzero_si256());
    return _mm256_blendv_ps(_mm256_castsi256_ps(normal_bits_i32x8), subnorm_f32x8, _mm256_castsi256_ps(exp_zero_mask));
}

/** @brief Convert 8x f32 to 8x e4m3 via bit manipulation (AVX2).
 *  E4M3 format: S EEEE MMM (bias=7). Handles normal, subnormal, and overflow cases.
 *  Subnormals (f32_exp <= 120): mantissa = round(abs_f32 * 512), clamped to [0,7]. */
NK_INTERNAL __m128i nk_f32x8_to_e4m3x8_haswell_(__m256 f32x8) {
    __m256i bits_i32x8 = _mm256_castps_si256(f32x8);
    __m256i sign_i32x8 = _mm256_srli_epi32(bits_i32x8, 31);
    __m256i f32_exp_i32x8 = _mm256_and_si256(_mm256_srli_epi32(bits_i32x8, 23), _mm256_set1_epi32(0xFF));

    // Round mantissa from 23 to 3 bits using RNE (round to nearest, ties to even)
    // RNE trick: add (half - 1 + lsb) where lsb is the bit that will become the new lsb after shift
    __m256i significand_i32x8 = _mm256_or_si256(_mm256_and_si256(bits_i32x8, _mm256_set1_epi32(0x007FFFFF)),
                                                _mm256_set1_epi32(0x00800000)); // Add implicit 1 bit
    __m256i lsb_i32x8 = _mm256_and_si256(_mm256_srli_epi32(significand_i32x8, 20), _mm256_set1_epi32(1));
    __m256i rounding_bias_i32x8 = _mm256_add_epi32(_mm256_set1_epi32(0x0007FFFF), lsb_i32x8);
    __m256i rounded_sig_i32x8 = _mm256_add_epi32(significand_i32x8, rounding_bias_i32x8);
    __m256i carry_i32x8 = _mm256_srli_epi32(rounded_sig_i32x8, 24); // Carry into exponent if bit 24 set
    __m256i f32_mantissa_i32x8 = _mm256_and_si256(_mm256_srli_epi32(rounded_sig_i32x8, 20), _mm256_set1_epi32(0x07));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    f32_mantissa_i32x8 = _mm256_andnot_si256(_mm256_slli_epi32(carry_i32x8, 31), f32_mantissa_i32x8);
    __m256i e4m3_exp_i32x8 = _mm256_sub_epi32(_mm256_add_epi32(f32_exp_i32x8, carry_i32x8), _mm256_set1_epi32(120));

    // Detect underflow (exp <= 0, maps to subnormal/zero) and overflow (exp > 15)
    __m256i is_subnormal_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32(1), e4m3_exp_i32x8);
    __m256i overflow_i32x8 = _mm256_cmpgt_epi32(e4m3_exp_i32x8, _mm256_set1_epi32(15));

    // Normal path: clamp exp to [1,15], extract mantissa bits
    // e4m3FN quirk: exp=15 with mantissa=7 is NaN (0x7F), so clamp mantissa to 6 when exp=15.
    __m256i clamped_exp_i32x8 = _mm256_max_epi32(e4m3_exp_i32x8, _mm256_set1_epi32(1));
    clamped_exp_i32x8 = _mm256_min_epi32(clamped_exp_i32x8, _mm256_set1_epi32(15));
    __m256i is_max_exp_i32x8 = _mm256_cmpeq_epi32(clamped_exp_i32x8, _mm256_set1_epi32(15));
    __m256i max_mantissa_i32x8 = _mm256_blendv_epi8(_mm256_set1_epi32(7), _mm256_set1_epi32(6), is_max_exp_i32x8);
    __m256i normal_mantissa_i32x8 = _mm256_min_epi32(f32_mantissa_i32x8, max_mantissa_i32x8);
    normal_mantissa_i32x8 = _mm256_blendv_epi8(normal_mantissa_i32x8, _mm256_set1_epi32(0x06), overflow_i32x8);
    __m256i normal_e4m3_i32x8 = _mm256_or_si256(
        _mm256_slli_epi32(sign_i32x8, 7),
        _mm256_or_si256(_mm256_slli_epi32(clamped_exp_i32x8, 3), normal_mantissa_i32x8));

    // Subnormal path: mantissa = round(abs_f32 * 512)
    // If mantissa rounds to 8 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x08
    __m256 abs_f32x8 = _mm256_and_ps(f32x8, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    __m256 scaled_f32x8 = _mm256_mul_ps(abs_f32x8, _mm256_set1_ps(512.0f));
    __m256i subnorm_mantissa_i32x8 = _mm256_cvtps_epi32(scaled_f32x8);
    __m256i promotes_to_normal_i32x8 = _mm256_cmpgt_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(7));
    subnorm_mantissa_i32x8 = _mm256_min_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(7));
    subnorm_mantissa_i32x8 = _mm256_max_epi32(subnorm_mantissa_i32x8, _mm256_setzero_si256());
    __m256i subnorm_e4m3_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 7), subnorm_mantissa_i32x8);
    // When mantissa rounds to 8, use first normal value (0x08) instead of clamped subnormal
    __m256i first_normal_e4m3_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 7), _mm256_set1_epi32(0x08));
    subnorm_e4m3_i32x8 = _mm256_blendv_epi8(subnorm_e4m3_i32x8, first_normal_e4m3_i32x8, promotes_to_normal_i32x8);

    // Blend: use subnormal result when exp <= 0, else normal
    __m256i e4m3_i32x8 = _mm256_blendv_epi8(normal_e4m3_i32x8, subnorm_e4m3_i32x8, is_subnormal_i32x8);

    // Pack 8 i32s to 8 unsigned i8s (use unsigned saturation to preserve values 128-255)
    __m128i low_i32x4 = _mm256_castsi256_si128(e4m3_i32x8);
    __m128i high_i32x4 = _mm256_extracti128_si256(e4m3_i32x8, 1);
    __m128i packed_i16x8 = _mm_packus_epi32(low_i32x4, high_i32x4);
    __m128i packed_i8x8 = _mm_packus_epi16(packed_i16x8, packed_i16x8);
    return packed_i8x8;
}

/** @brief Convert 8x f32 to 8x e5m2 via bit manipulation (AVX2).
 *  E5M2 format: S EEEEE MM (bias=15). Handles normal, subnormal, and overflow cases.
 *  Uses RNE (round to nearest even) for mantissa rounding. */
NK_INTERNAL __m128i nk_f32x8_to_e5m2x8_haswell_(__m256 f32x8) {
    __m256i bits_i32x8 = _mm256_castps_si256(f32x8);
    __m256i sign_i32x8 = _mm256_srli_epi32(bits_i32x8, 31);
    __m256i f32_exp_i32x8 = _mm256_and_si256(_mm256_srli_epi32(bits_i32x8, 23), _mm256_set1_epi32(0xFF));

    // Round mantissa from 23 to 2 bits using RNE (round to nearest, ties to even)
    // RNE trick: add (half - 1 + lsb) where lsb is the bit that will become the new lsb after shift
    __m256i significand_i32x8 = _mm256_or_si256(_mm256_and_si256(bits_i32x8, _mm256_set1_epi32(0x007FFFFF)),
                                                _mm256_set1_epi32(0x00800000)); // Add implicit 1 bit
    __m256i lsb_i32x8 = _mm256_and_si256(_mm256_srli_epi32(significand_i32x8, 21), _mm256_set1_epi32(1));
    __m256i rounding_bias_i32x8 = _mm256_add_epi32(_mm256_set1_epi32(0x000FFFFF), lsb_i32x8); // half = 0x100000
    __m256i rounded_sig_i32x8 = _mm256_add_epi32(significand_i32x8, rounding_bias_i32x8);
    __m256i carry_i32x8 = _mm256_srli_epi32(rounded_sig_i32x8, 24); // Carry into exponent if bit 24 set
    __m256i f32_mantissa_i32x8 = _mm256_and_si256(_mm256_srli_epi32(rounded_sig_i32x8, 21), _mm256_set1_epi32(0x03));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    f32_mantissa_i32x8 = _mm256_andnot_si256(_mm256_slli_epi32(carry_i32x8, 31), f32_mantissa_i32x8);
    __m256i e5m2_exp_i32x8 = _mm256_sub_epi32(_mm256_add_epi32(f32_exp_i32x8, carry_i32x8), _mm256_set1_epi32(112));

    // Detect subnormal (exp <= 0) and overflow (exp > 31)
    __m256i is_subnormal_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32(1), e5m2_exp_i32x8);
    __m256i overflow_i32x8 = _mm256_cmpgt_epi32(e5m2_exp_i32x8, _mm256_set1_epi32(31));

    // Normal path: clamp exp to [1,31], on overflow return infinity (exp=31, mantissa=0 = 0x7C)
    __m256i clamped_exp_i32x8 = _mm256_max_epi32(e5m2_exp_i32x8, _mm256_set1_epi32(1));
    clamped_exp_i32x8 = _mm256_min_epi32(clamped_exp_i32x8, _mm256_set1_epi32(31));
    __m256i normal_mantissa_i32x8 = _mm256_blendv_epi8(f32_mantissa_i32x8, _mm256_setzero_si256(), overflow_i32x8);
    __m256i normal_e5m2_i32x8 = _mm256_or_si256(
        _mm256_slli_epi32(sign_i32x8, 7),
        _mm256_or_si256(_mm256_slli_epi32(clamped_exp_i32x8, 2), normal_mantissa_i32x8));

    // Subnormal path: mantissa = round(abs_f32 * 65536)
    // If mantissa rounds to 4 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x04
    __m256 abs_f32x8 = _mm256_and_ps(f32x8, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    __m256 scaled_f32x8 = _mm256_mul_ps(abs_f32x8, _mm256_set1_ps(65536.0f));
    __m256i subnorm_mantissa_i32x8 = _mm256_cvtps_epi32(scaled_f32x8);
    __m256i promotes_to_normal_i32x8 = _mm256_cmpgt_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(3));
    subnorm_mantissa_i32x8 = _mm256_min_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(3));
    subnorm_mantissa_i32x8 = _mm256_max_epi32(subnorm_mantissa_i32x8, _mm256_setzero_si256());
    __m256i subnorm_e5m2_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 7), subnorm_mantissa_i32x8);
    // When mantissa rounds to 4, use first normal value (0x04) instead of clamped subnormal
    __m256i first_normal_e5m2_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 7), _mm256_set1_epi32(0x04));
    subnorm_e5m2_i32x8 = _mm256_blendv_epi8(subnorm_e5m2_i32x8, first_normal_e5m2_i32x8, promotes_to_normal_i32x8);

    // Blend: use subnormal result when exp <= 0
    __m256i e5m2_i32x8 = _mm256_blendv_epi8(normal_e5m2_i32x8, subnorm_e5m2_i32x8, is_subnormal_i32x8);

    // Pack 8 i32s to 8 unsigned i8s (use unsigned saturation to preserve values 128-255)
    __m128i low_i32x4 = _mm256_castsi256_si128(e5m2_i32x8);
    __m128i high_i32x4 = _mm256_extracti128_si256(e5m2_i32x8, 1);
    __m128i packed_i16x8 = _mm_packus_epi32(low_i32x4, high_i32x4);
    __m128i packed_i8x8 = _mm_packus_epi16(packed_i16x8, packed_i16x8);
    return packed_i8x8;
}

#pragma endregion - Vectorized Conversions

#pragma region - Converting Loads and Stores

/** @brief Partial load for f16 elements (up to 8) with conversion to f32 via F16C. */
NK_INTERNAL __m256 nk_partial_load_f16x8_to_f32x8_haswell_(nk_f16_t const *src, nk_size_t n) {
    nk_b256_vec_t vec;
    nk_partial_load_b16x16_serial_(src, &vec, n);
    return _mm256_cvtph_ps(vec.xmms[0]);
}

/** @brief Partial load for bf16 elements (up to 8) with conversion to f32. */
NK_INTERNAL __m256 nk_partial_load_bf16x8_to_f32x8_haswell_(nk_bf16_t const *src, nk_size_t n) {
    nk_b256_vec_t vec;
    nk_partial_load_b16x16_serial_(src, &vec, n);
    return nk_bf16x8_to_f32x8_haswell_(vec.xmms[0]);
}

/** @brief Partial load for e4m3 elements (up to 8) with conversion to f32. */
NK_INTERNAL __m256 nk_partial_load_e4m3x8_to_f32x8_haswell_(nk_e4m3_t const *src, nk_size_t n) {
    nk_b64_vec_t vec;
    nk_partial_load_b8x8_serial_(src, &vec, n);
    return nk_e4m3x8_to_f32x8_haswell_(_mm_cvtsi64_si128(vec.u64));
}

/** @brief Partial load for e5m2 elements (up to 8) with conversion to f32. */
NK_INTERNAL __m256 nk_partial_load_e5m2x8_to_f32x8_haswell_(nk_e5m2_t const *src, nk_size_t n) {
    nk_b64_vec_t vec;
    nk_partial_load_b8x8_serial_(src, &vec, n);
    return nk_e5m2x8_to_f32x8_haswell_(_mm_cvtsi64_si128(vec.u64));
}

#pragma endregion - Converting Loads and Stores

#pragma region - Public API

NK_PUBLIC void nk_cast_haswell(void const *from, nk_datatype_t from_type, nk_size_t n, void *to,
                               nk_datatype_t to_type) {
    return nk_cast_serial(from, from_type, n, to, to_type);
}

#pragma endregion - Public API

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_

#endif // NK_CAST_HASWELL_H
