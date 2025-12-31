/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Skylake-X CPUs.
 *  @file include/numkong/dot/skylake.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_SKYLAKE_H
#define NK_DOT_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,bmi2"))), \
                             apply_to = function)

#include "numkong/reduce/skylake.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Internal helper for f32x16-based finalize (used by f16, bf16, e4m3, e5m2 kernels).
 *
 *  These types accumulate to f32x16 and need a common horizontal reduction.
 */
NK_INTERNAL void nk_dot_f32x16_finalize_skylake_wout_compensation_( //
    __m512 const sum_a_f32x16, __m512 const sum_b_f32x16,           //
    __m512 const sum_c_f32x16, __m512 const sum_d_f32x16,           //
    nk_b128_vec_t *result) {

    // ILP-optimized 4-way horizontal reduction for f32x16 in AVX-512
    // Step 1: 16→8 for all 4 states (extract high 256-bit half and add to low half)
    __m256 sum_a_f32x8 = _mm256_add_ps(_mm512_castps512_ps256(sum_a_f32x16), _mm512_extractf32x8_ps(sum_a_f32x16, 1));
    __m256 sum_b_f32x8 = _mm256_add_ps(_mm512_castps512_ps256(sum_b_f32x16), _mm512_extractf32x8_ps(sum_b_f32x16, 1));
    __m256 sum_c_f32x8 = _mm256_add_ps(_mm512_castps512_ps256(sum_c_f32x16), _mm512_extractf32x8_ps(sum_c_f32x16, 1));
    __m256 sum_d_f32x8 = _mm256_add_ps(_mm512_castps512_ps256(sum_d_f32x16), _mm512_extractf32x8_ps(sum_d_f32x16, 1));
    // Step 2: 8→4 for all 4 states (extract high 128-bit half and add to low half)
    __m128 sum_a_f32x4 = _mm_add_ps(_mm256_castps256_ps128(sum_a_f32x8), _mm256_extractf128_ps(sum_a_f32x8, 1));
    __m128 sum_b_f32x4 = _mm_add_ps(_mm256_castps256_ps128(sum_b_f32x8), _mm256_extractf128_ps(sum_b_f32x8, 1));
    __m128 sum_c_f32x4 = _mm_add_ps(_mm256_castps256_ps128(sum_c_f32x8), _mm256_extractf128_ps(sum_c_f32x8, 1));
    __m128 sum_d_f32x4 = _mm_add_ps(_mm256_castps256_ps128(sum_d_f32x8), _mm256_extractf128_ps(sum_d_f32x8, 1));
    // Step 3: Transpose 4x4 and reduce to get final 4 scalars
    __m128 transpose_ab_low_f32x4 = _mm_unpacklo_ps(sum_a_f32x4, sum_b_f32x4);
    __m128 transpose_cd_low_f32x4 = _mm_unpacklo_ps(sum_c_f32x4, sum_d_f32x4);
    __m128 transpose_ab_high_f32x4 = _mm_unpackhi_ps(sum_a_f32x4, sum_b_f32x4);
    __m128 transpose_cd_high_f32x4 = _mm_unpackhi_ps(sum_c_f32x4, sum_d_f32x4);
    __m128 sum_lane0_f32x4 = _mm_movelh_ps(transpose_ab_low_f32x4, transpose_cd_low_f32x4);
    __m128 sum_lane1_f32x4 = _mm_movehl_ps(transpose_cd_low_f32x4, transpose_ab_low_f32x4);
    __m128 sum_lane2_f32x4 = _mm_movelh_ps(transpose_ab_high_f32x4, transpose_cd_high_f32x4);
    __m128 sum_lane3_f32x4 = _mm_movehl_ps(transpose_cd_high_f32x4, transpose_ab_high_f32x4);
    __m128 final_sum_f32x4 = _mm_add_ps(_mm_add_ps(sum_lane0_f32x4, sum_lane1_f32x4),
                                        _mm_add_ps(sum_lane2_f32x4, sum_lane3_f32x4));
    result->xmm = _mm_castps_si128(final_sum_f32x4);
}

NK_PUBLIC void nk_dot_f32_skylake(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m512d sum_f64x8 = _mm512_setzero_pd();

nk_dot_f32_skylake_cycle:
    if (count_scalars < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_f32x8 = _mm256_maskz_loadu_ps(mask, a_scalars);
        b_f32x8 = _mm256_maskz_loadu_ps(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_f32x8 = _mm256_loadu_ps(a_scalars);
        b_f32x8 = _mm256_loadu_ps(b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f64x8 = _mm512_fmadd_pd(_mm512_cvtps_pd(a_f32x8), _mm512_cvtps_pd(b_f32x8), sum_f64x8);
    if (count_scalars) goto nk_dot_f32_skylake_cycle;

    *result = (nk_f32_t)_mm512_reduce_add_pd(sum_f64x8);
}

NK_PUBLIC void nk_dot_f64_skylake(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f64_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated dot product
    __m512d a_f64x8, b_f64x8;
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d compensation_f64x8 = _mm512_setzero_pd();

nk_dot_f64_skylake_cycle:
    if (count_scalars < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a_scalars);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a_scalars);
        b_f64x8 = _mm512_loadu_pd(b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    // TwoProd: h = a * b, r = fma(a, b, -h) captures the rounding error
    __m512d product_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    __m512d product_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_f64x8, product_f64x8);
    // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
    __m512d t_f64x8 = _mm512_add_pd(sum_f64x8, product_f64x8);
    __m512d z_f64x8 = _mm512_sub_pd(t_f64x8, sum_f64x8);
    __m512d sum_error_f64x8 = _mm512_add_pd(_mm512_sub_pd(sum_f64x8, _mm512_sub_pd(t_f64x8, z_f64x8)),
                                            _mm512_sub_pd(product_f64x8, z_f64x8));
    // Update: sum = t, compensation += q + r
    sum_f64x8 = t_f64x8;
    compensation_f64x8 = _mm512_add_pd(compensation_f64x8, _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
    if (count_scalars) goto nk_dot_f64_skylake_cycle;

    // Reduce and combine sum + compensation
    *result = _mm512_reduce_add_pd(_mm512_add_pd(sum_f64x8, compensation_f64x8));
}

NK_PUBLIC void nk_dot_f32c_skylake(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f32c_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m512d sum_real_f64x8 = _mm512_setzero_pd();
    __m512d sum_imag_f64x8 = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f64x8 = _mm512_set_epi64(0x8000000000000000, 0, 0x8000000000000000, 0, 0x8000000000000000,
                                                     0, 0x8000000000000000, 0);
nk_dot_f32c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f32x8 = _mm256_maskz_loadu_ps(mask, (nk_f32_t const *)a_pairs);
        b_f32x8 = _mm256_maskz_loadu_ps(mask, (nk_f32_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_f32x8 = _mm256_loadu_ps((nk_f32_t const *)a_pairs);
        b_f32x8 = _mm256_loadu_ps((nk_f32_t const *)b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    __m512d a_f64x8 = _mm512_cvtps_pd(a_f32x8);
    __m512d b_f64x8 = _mm512_cvtps_pd(b_f32x8);
    __m512d b_swapped_f64x8 = _mm512_permute_pd(b_f64x8, 0b01010101);
    sum_real_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, sum_real_f64x8);
    sum_imag_f64x8 = _mm512_fmadd_pd(a_f64x8, b_swapped_f64x8, sum_imag_f64x8);
    if (count_pairs) goto nk_dot_f32c_skylake_cycle;

    // Flip the sign bit in every second f64 before accumulation:
    sum_real_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(sum_real_f64x8), sign_flip_f64x8));

    // Reduce horizontal sums:
    result->real = (nk_f32_t)_mm512_reduce_add_pd(sum_real_f64x8);
    result->imag = (nk_f32_t)_mm512_reduce_add_pd(sum_imag_f64x8);
}

NK_PUBLIC void nk_vdot_f32c_skylake(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m512d sum_real_f64x8 = _mm512_setzero_pd();
    __m512d sum_imag_f64x8 = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f64x8 = _mm512_set_epi64(0x8000000000000000, 0, 0x8000000000000000, 0, 0x8000000000000000,
                                                     0, 0x8000000000000000, 0);
nk_vdot_f32c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f32x8 = _mm256_maskz_loadu_ps(mask, (nk_f32_t const *)a_pairs);
        b_f32x8 = _mm256_maskz_loadu_ps(mask, (nk_f32_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_f32x8 = _mm256_loadu_ps((nk_f32_t const *)a_pairs);
        b_f32x8 = _mm256_loadu_ps((nk_f32_t const *)b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    __m512d a_f64x8 = _mm512_cvtps_pd(a_f32x8);
    __m512d b_f64x8 = _mm512_cvtps_pd(b_f32x8);
    sum_real_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, sum_real_f64x8);
    __m512d b_swapped_f64x8 = _mm512_permute_pd(b_f64x8, 0b01010101);
    sum_imag_f64x8 = _mm512_fmadd_pd(a_f64x8, b_swapped_f64x8, sum_imag_f64x8);
    if (count_pairs) goto nk_vdot_f32c_skylake_cycle;

    // Flip the sign bit in every second f64 before accumulation:
    sum_imag_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(sum_imag_f64x8), sign_flip_f64x8));

    // Reduce horizontal sums:
    result->real = (nk_f32_t)_mm512_reduce_add_pd(sum_real_f64x8);
    result->imag = (nk_f32_t)_mm512_reduce_add_pd(sum_imag_f64x8);
}

NK_PUBLIC void nk_dot_f64c_skylake(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f64c_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated complex dot product
    __m512d a_f64x8, b_f64x8;
    __m512d sum_real_f64x8 = _mm512_setzero_pd();
    __m512d sum_imag_f64x8 = _mm512_setzero_pd();
    __m512d compensation_real_f64x8 = _mm512_setzero_pd();
    __m512d compensation_imag_f64x8 = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f64x8 = _mm512_set_epi64(                                   //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000  //
    );
nk_dot_f64c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a_pairs);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a_pairs);
        b_f64x8 = _mm512_loadu_pd(b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    __m512d b_swapped_f64x8 = _mm512_permute_pd(b_f64x8, 0x55); //? Same as 0b01010101.

    // TwoProd for real part: a * b
    __m512d product_real_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    __m512d product_real_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_f64x8, product_real_f64x8);
    // TwoSum for real part
    __m512d t_real_f64x8 = _mm512_add_pd(sum_real_f64x8, product_real_f64x8);
    __m512d z_real_f64x8 = _mm512_sub_pd(t_real_f64x8, sum_real_f64x8);
    __m512d sum_real_error_f64x8 = _mm512_add_pd(
        _mm512_sub_pd(sum_real_f64x8, _mm512_sub_pd(t_real_f64x8, z_real_f64x8)),
        _mm512_sub_pd(product_real_f64x8, z_real_f64x8));
    sum_real_f64x8 = t_real_f64x8;
    compensation_real_f64x8 = _mm512_add_pd(compensation_real_f64x8,
                                            _mm512_add_pd(sum_real_error_f64x8, product_real_error_f64x8));

    // TwoProd for imag part: a * b_swapped
    __m512d product_imag_f64x8 = _mm512_mul_pd(a_f64x8, b_swapped_f64x8);
    __m512d product_imag_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_swapped_f64x8, product_imag_f64x8);
    // TwoSum for imag part
    __m512d t_imag_f64x8 = _mm512_add_pd(sum_imag_f64x8, product_imag_f64x8);
    __m512d z_imag_f64x8 = _mm512_sub_pd(t_imag_f64x8, sum_imag_f64x8);
    __m512d sum_imag_error_f64x8 = _mm512_add_pd(
        _mm512_sub_pd(sum_imag_f64x8, _mm512_sub_pd(t_imag_f64x8, z_imag_f64x8)),
        _mm512_sub_pd(product_imag_f64x8, z_imag_f64x8));
    sum_imag_f64x8 = t_imag_f64x8;
    compensation_imag_f64x8 = _mm512_add_pd(compensation_imag_f64x8,
                                            _mm512_add_pd(sum_imag_error_f64x8, product_imag_error_f64x8));

    if (count_pairs) goto nk_dot_f64c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation (to get a_r*b_r - a_i*b_i):
    sum_real_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(sum_real_f64x8), sign_flip_f64x8));
    compensation_real_f64x8 = _mm512_castsi512_pd(
        _mm512_xor_si512(_mm512_castpd_si512(compensation_real_f64x8), sign_flip_f64x8));

    // Reduce and combine: first vector-add sum+compensation, then horizontal reduce
    result->real = _mm512_reduce_add_pd(_mm512_add_pd(sum_real_f64x8, compensation_real_f64x8));
    result->imag = _mm512_reduce_add_pd(_mm512_add_pd(sum_imag_f64x8, compensation_imag_f64x8));
}

NK_PUBLIC void nk_vdot_f64c_skylake(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f64c_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated conjugate dot product
    __m512d a_f64x8, b_f64x8;
    __m512d sum_real_f64x8 = _mm512_setzero_pd();
    __m512d sum_imag_f64x8 = _mm512_setzero_pd();
    __m512d compensation_real_f64x8 = _mm512_setzero_pd();
    __m512d compensation_imag_f64x8 = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f64x8 = _mm512_set_epi64(                                   //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000  //
    );
nk_vdot_f64c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, (nk_f64_t const *)a_pairs);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, (nk_f64_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd((nk_f64_t const *)a_pairs);
        b_f64x8 = _mm512_loadu_pd((nk_f64_t const *)b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    __m512d b_swapped_f64x8 = _mm512_permute_pd(b_f64x8, 0x55); //? Same as 0b01010101.

    // TwoProd for real part: a * b
    __m512d product_real_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    __m512d product_real_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_f64x8, product_real_f64x8);
    // TwoSum for real part
    __m512d t_real_f64x8 = _mm512_add_pd(sum_real_f64x8, product_real_f64x8);
    __m512d z_real_f64x8 = _mm512_sub_pd(t_real_f64x8, sum_real_f64x8);
    __m512d sum_real_error_f64x8 = _mm512_add_pd(
        _mm512_sub_pd(sum_real_f64x8, _mm512_sub_pd(t_real_f64x8, z_real_f64x8)),
        _mm512_sub_pd(product_real_f64x8, z_real_f64x8));
    sum_real_f64x8 = t_real_f64x8;
    compensation_real_f64x8 = _mm512_add_pd(compensation_real_f64x8,
                                            _mm512_add_pd(sum_real_error_f64x8, product_real_error_f64x8));

    // TwoProd for imag part: a * b_swapped
    __m512d product_imag_f64x8 = _mm512_mul_pd(a_f64x8, b_swapped_f64x8);
    __m512d product_imag_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_swapped_f64x8, product_imag_f64x8);
    // TwoSum for imag part
    __m512d t_imag_f64x8 = _mm512_add_pd(sum_imag_f64x8, product_imag_f64x8);
    __m512d z_imag_f64x8 = _mm512_sub_pd(t_imag_f64x8, sum_imag_f64x8);
    __m512d sum_imag_error_f64x8 = _mm512_add_pd(
        _mm512_sub_pd(sum_imag_f64x8, _mm512_sub_pd(t_imag_f64x8, z_imag_f64x8)),
        _mm512_sub_pd(product_imag_f64x8, z_imag_f64x8));
    sum_imag_f64x8 = t_imag_f64x8;
    compensation_imag_f64x8 = _mm512_add_pd(compensation_imag_f64x8,
                                            _mm512_add_pd(sum_imag_error_f64x8, product_imag_error_f64x8));

    if (count_pairs) goto nk_vdot_f64c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation (to get a_r*b_i - a_i*b_r):
    sum_imag_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(sum_imag_f64x8), sign_flip_f64x8));
    compensation_imag_f64x8 = _mm512_castsi512_pd(
        _mm512_xor_si512(_mm512_castpd_si512(compensation_imag_f64x8), sign_flip_f64x8));

    // Reduce and combine: first vector-add sum+compensation, then horizontal reduce
    result->real = _mm512_reduce_add_pd(_mm512_add_pd(sum_real_f64x8, compensation_real_f64x8));
    result->imag = _mm512_reduce_add_pd(_mm512_add_pd(sum_imag_f64x8, compensation_imag_f64x8));
}

NK_PUBLIC void nk_dot_f16_skylake(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    __m256i a_f16x16, b_f16x16;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_f16_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_f16x16 = _mm256_maskz_loadu_epi16(mask, a_scalars);
        b_f16x16 = _mm256_maskz_loadu_epi16(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_f16x16 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_f16x16 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32x16 = _mm512_cvtph_ps(a_f16x16);
    __m512 b_f32x16 = _mm512_cvtph_ps(b_f16x16);
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto nk_dot_f16_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_bf16_skylake(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m256i a_bf16x16, b_bf16x16;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_bf16_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_bf16x16 = _mm256_maskz_loadu_epi16(mask, a_scalars);
        b_bf16x16 = _mm256_maskz_loadu_epi16(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_bf16x16 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_bf16x16 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
    __m512 b_f32x16 = nk_bf16x16_to_f32x16_skylake_(b_bf16x16);
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto nk_dot_bf16_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_e4m3_skylake(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m128i a_e4m3x16, b_e4m3x16;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_e4m3_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_e4m3x16 = _mm_maskz_loadu_epi8(mask, a_scalars);
        b_e4m3x16 = _mm_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e4m3x16 = _mm_loadu_si128((__m128i const *)a_scalars);
        b_e4m3x16 = _mm_loadu_si128((__m128i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32x16 = nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16);
    __m512 b_f32x16 = nk_e4m3x16_to_f32x16_skylake_(b_e4m3x16);
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto nk_dot_e4m3_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_e5m2_skylake(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m128i a_e5m2x16, b_e5m2x16;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_e5m2_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_e5m2x16 = _mm_maskz_loadu_epi8(mask, a_scalars);
        b_e5m2x16 = _mm_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e5m2x16 = _mm_loadu_si128((__m128i const *)a_scalars);
        b_e5m2x16 = _mm_loadu_si128((__m128i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32x16 = nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16);
    __m512 b_f32x16 = nk_e5m2x16_to_f32x16_skylake_(b_e5m2x16);
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto nk_dot_e5m2_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_i8_skylake(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_i32_t *result) {
    __m512i sum_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        // Load 32 bytes at a time and widen to i16
        __m256i a_i8x32 = _mm256_loadu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_i8x32 = _mm256_loadu_si256((__m256i const *)(b_scalars + idx_scalars));
        __m512i a_i16x32 = _mm512_cvtepi8_epi16(a_i8x32);
        __m512i b_i16x32 = _mm512_cvtepi8_epi16(b_i8x32);
        // _mm512_madd_epi16: multiply adjacent pairs of i16, add pairs to produce i32
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(a_i16x32, b_i16x32));
    }
    nk_i32_t sum = _mm512_reduce_add_epi32(sum_i32x16);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_i32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_u8_skylake(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_u32_t *result) {
    __m512i sum_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        // Load 32 bytes and zero-extend to i16 (u8 -> u16 via zero-extension)
        __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)(b_scalars + idx_scalars));
        __m512i a_u16x32 = _mm512_cvtepu8_epi16(a_u8x32);
        __m512i b_u16x32 = _mm512_cvtepu8_epi16(b_u8x32);
        // _mm512_madd_epi16: multiply adjacent pairs, add pairs to produce i32
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(a_u16x32, b_u16x32));
    }
    nk_u32_t sum = (nk_u32_t)_mm512_reduce_add_epi32(sum_i32x16);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_u32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

typedef struct nk_dot_f64x8_state_skylake_t {
    __m512d sum_f64x8;
    __m512d compensation_f64x8;
} nk_dot_f64x8_state_skylake_t;

NK_INTERNAL void nk_dot_f64x8_init_skylake(nk_dot_f64x8_state_skylake_t *state) {
    state->sum_f64x8 = _mm512_setzero_pd();
    state->compensation_f64x8 = _mm512_setzero_pd();
}

NK_INTERNAL void nk_dot_f64x8_update_skylake(nk_dot_f64x8_state_skylake_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    __m512d sum_f64x8 = state->sum_f64x8;
    __m512d compensation_f64x8 = state->compensation_f64x8;
    __m512d a_f64x8 = a.zmm_pd;
    __m512d b_f64x8 = b.zmm_pd;

    // TwoProd: h = a * b, r = fma(a, b, -h) captures the rounding error
    __m512d product_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    __m512d product_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_f64x8, product_f64x8);

    // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
    __m512d t_f64x8 = _mm512_add_pd(sum_f64x8, product_f64x8);
    __m512d z_f64x8 = _mm512_sub_pd(t_f64x8, sum_f64x8);
    __m512d sum_error_f64x8 = _mm512_add_pd(_mm512_sub_pd(sum_f64x8, _mm512_sub_pd(t_f64x8, z_f64x8)),
                                            _mm512_sub_pd(product_f64x8, z_f64x8));

    // Update: sum = t, compensation += q + r
    state->sum_f64x8 = t_f64x8;
    state->compensation_f64x8 = _mm512_add_pd(compensation_f64x8, _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
}

NK_INTERNAL void nk_dot_f64x8_finalize_skylake(                                               //
    nk_dot_f64x8_state_skylake_t const *state_a, nk_dot_f64x8_state_skylake_t const *state_b, //
    nk_dot_f64x8_state_skylake_t const *state_c, nk_dot_f64x8_state_skylake_t const *state_d, //
    nk_b256_vec_t *result) {
    // Combine sum + compensation before horizontal reduction
    __m512d sum_a_f64x8 = _mm512_add_pd(state_a->sum_f64x8, state_a->compensation_f64x8);
    __m512d sum_b_f64x8 = _mm512_add_pd(state_b->sum_f64x8, state_b->compensation_f64x8);
    __m512d sum_c_f64x8 = _mm512_add_pd(state_c->sum_f64x8, state_c->compensation_f64x8);
    __m512d sum_d_f64x8 = _mm512_add_pd(state_d->sum_f64x8, state_d->compensation_f64x8);

    // ILP-optimized 4-way horizontal reduction for f64
    // Step 1: 8->4 for all 4 states (extract high 256-bit half and add to low half)
    __m256d sum_a_f64x4 = _mm256_add_pd(_mm512_castpd512_pd256(sum_a_f64x8), _mm512_extractf64x4_pd(sum_a_f64x8, 1));
    __m256d sum_b_f64x4 = _mm256_add_pd(_mm512_castpd512_pd256(sum_b_f64x8), _mm512_extractf64x4_pd(sum_b_f64x8, 1));
    __m256d sum_c_f64x4 = _mm256_add_pd(_mm512_castpd512_pd256(sum_c_f64x8), _mm512_extractf64x4_pd(sum_c_f64x8, 1));
    __m256d sum_d_f64x4 = _mm256_add_pd(_mm512_castpd512_pd256(sum_d_f64x8), _mm512_extractf64x4_pd(sum_d_f64x8, 1));
    // Step 2: 4->2 for all 4 states (extract high 128-bit half and add to low half)
    __m128d sum_a_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_a_f64x4), _mm256_extractf128_pd(sum_a_f64x4, 1));
    __m128d sum_b_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_b_f64x4), _mm256_extractf128_pd(sum_b_f64x4, 1));
    __m128d sum_c_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_c_f64x4), _mm256_extractf128_pd(sum_c_f64x4, 1));
    __m128d sum_d_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_d_f64x4), _mm256_extractf128_pd(sum_d_f64x4, 1));
    // Step 3: Horizontal add pairs: [a0+a1, b0+b1] and [c0+c1, d0+d1]
    __m128d sum_ab_f64x2 = _mm_hadd_pd(sum_a_f64x2, sum_b_f64x2);
    __m128d sum_cd_f64x2 = _mm_hadd_pd(sum_c_f64x2, sum_d_f64x2);
    // Pack into 256-bit result vector
    result->ymm = _mm256_castpd_si256(_mm256_set_m128d(sum_cd_f64x2, sum_ab_f64x2));
}

typedef struct nk_dot_f32x8_state_skylake_t {
    __m512d sum_f64x8;
} nk_dot_f32x8_state_skylake_t;

NK_INTERNAL void nk_dot_f32x8_init_skylake(nk_dot_f32x8_state_skylake_t *state) {
    state->sum_f64x8 = _mm512_setzero_pd();
}

NK_INTERNAL void nk_dot_f32x8_update_skylake(nk_dot_f32x8_state_skylake_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    // Upcast 8 f32s to f64 for high-precision accumulation
    __m512d a_f64x8 = _mm512_cvtps_pd(a.ymm_ps);
    __m512d b_f64x8 = _mm512_cvtps_pd(b.ymm_ps);
    // Simple FMA accumulation in f64
    state->sum_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, state->sum_f64x8);
}

NK_INTERNAL void nk_dot_f32x8_finalize_skylake(                                               //
    nk_dot_f32x8_state_skylake_t const *state_a, nk_dot_f32x8_state_skylake_t const *state_b, //
    nk_dot_f32x8_state_skylake_t const *state_c, nk_dot_f32x8_state_skylake_t const *state_d, //
    nk_b128_vec_t *result) {
    // ILP-optimized 4-way horizontal reduction for f64
    // Step 1: 8->4 for all 4 states (extract high 256-bit half and add to low half)
    __m256d sum_a_f64x4 = _mm256_add_pd(_mm512_castpd512_pd256(state_a->sum_f64x8),
                                        _mm512_extractf64x4_pd(state_a->sum_f64x8, 1));
    __m256d sum_b_f64x4 = _mm256_add_pd(_mm512_castpd512_pd256(state_b->sum_f64x8),
                                        _mm512_extractf64x4_pd(state_b->sum_f64x8, 1));
    __m256d sum_c_f64x4 = _mm256_add_pd(_mm512_castpd512_pd256(state_c->sum_f64x8),
                                        _mm512_extractf64x4_pd(state_c->sum_f64x8, 1));
    __m256d sum_d_f64x4 = _mm256_add_pd(_mm512_castpd512_pd256(state_d->sum_f64x8),
                                        _mm512_extractf64x4_pd(state_d->sum_f64x8, 1));
    // Step 2: 4->2 for all 4 states (extract high 128-bit half and add to low half)
    __m128d sum_a_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_a_f64x4), _mm256_extractf128_pd(sum_a_f64x4, 1));
    __m128d sum_b_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_b_f64x4), _mm256_extractf128_pd(sum_b_f64x4, 1));
    __m128d sum_c_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_c_f64x4), _mm256_extractf128_pd(sum_c_f64x4, 1));
    __m128d sum_d_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_d_f64x4), _mm256_extractf128_pd(sum_d_f64x4, 1));
    // Step 3: Horizontal add pairs: [a0+a1, b0+b1] and [c0+c1, d0+d1]
    __m128d sum_ab_f64x2 = _mm_hadd_pd(sum_a_f64x2, sum_b_f64x2);
    __m128d sum_cd_f64x2 = _mm_hadd_pd(sum_c_f64x2, sum_d_f64x2);
    // Downcast f64 results to f32
    __m128 result_f32x4 = _mm_movelh_ps(_mm_cvtpd_ps(sum_ab_f64x2), _mm_cvtpd_ps(sum_cd_f64x2));
    result->xmm = _mm_castps_si128(result_f32x4);
}

typedef struct nk_dot_e4m3x16_state_skylake_t {
    __m512 sum_f32x16;
} nk_dot_e4m3x16_state_skylake_t;

NK_INTERNAL void nk_dot_e4m3x16_init_skylake(nk_dot_e4m3x16_state_skylake_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

NK_INTERNAL void nk_dot_e4m3x16_update_skylake(nk_dot_e4m3x16_state_skylake_t *state, nk_b128_vec_t a,
                                               nk_b128_vec_t b) {
    state->sum_f32x16 = _mm512_fmadd_ps(nk_e4m3x16_to_f32x16_skylake_(a.xmm), nk_e4m3x16_to_f32x16_skylake_(b.xmm),
                                        state->sum_f32x16);
}

NK_INTERNAL void nk_dot_e4m3x16_finalize_skylake(                                                 //
    nk_dot_e4m3x16_state_skylake_t const *state_a, nk_dot_e4m3x16_state_skylake_t const *state_b, //
    nk_dot_e4m3x16_state_skylake_t const *state_c, nk_dot_e4m3x16_state_skylake_t const *state_d, //
    nk_b128_vec_t *result) {
    nk_dot_f32x16_finalize_skylake_wout_compensation_(state_a->sum_f32x16, state_b->sum_f32x16, state_c->sum_f32x16,
                                                      state_d->sum_f32x16, result);
}

typedef struct nk_dot_e5m2x16_state_skylake_t {
    __m512 sum_f32x16;
} nk_dot_e5m2x16_state_skylake_t;

NK_INTERNAL void nk_dot_e5m2x16_init_skylake(nk_dot_e5m2x16_state_skylake_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

NK_INTERNAL void nk_dot_e5m2x16_update_skylake(nk_dot_e5m2x16_state_skylake_t *state, nk_b128_vec_t a,
                                               nk_b128_vec_t b) {
    state->sum_f32x16 = _mm512_fmadd_ps(nk_e5m2x16_to_f32x16_skylake_(a.xmm), nk_e5m2x16_to_f32x16_skylake_(b.xmm),
                                        state->sum_f32x16);
}

NK_INTERNAL void nk_dot_e5m2x16_finalize_skylake(                                                 //
    nk_dot_e5m2x16_state_skylake_t const *state_a, nk_dot_e5m2x16_state_skylake_t const *state_b, //
    nk_dot_e5m2x16_state_skylake_t const *state_c, nk_dot_e5m2x16_state_skylake_t const *state_d, //
    nk_b128_vec_t *result) {
    nk_dot_f32x16_finalize_skylake_wout_compensation_(state_a->sum_f32x16, state_b->sum_f32x16, state_c->sum_f32x16,
                                                      state_d->sum_f32x16, result);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_

#endif // NK_DOT_SKYLAKE_H