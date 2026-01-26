/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Haswell CPUs.
 *  @file include/numkong/dot/haswell.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @section haswell_dot_instructions Key AVX2/FMA Dot Product Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm256_fmadd_ps/pd          VFMADD (YMM, YMM, YMM)          5cy         0.5/cy      p01
 *      _mm256_mul_ps/pd            VMULPS/PD (YMM, YMM, YMM)       5cy         0.5/cy      p01
 *      _mm256_add_ps/pd            VADDPS/PD (YMM, YMM, YMM)       3cy         1/cy        p01
 *      _mm256_cvtph_ps             VCVTPH2PS (YMM, XMM)            5cy         1/cy        p01
 *      _mm256_cvtps_pd             VCVTPS2PD (YMM, XMM)            2cy         1/cy        p01
 *
 *  For small numeric types (F16, BF16, E4M3, E5M2) we use F32 accumulators. For F32 dot products,
 *  upcasting to F64 and downcasting back is faster than stable summation algorithms. For F64 we
 *  use the Dot2 algorithm (Ogita-Rump-Oishi, 2005) for compensated accumulation via TwoSum/TwoProd.
 */
#ifndef NK_DOT_HASWELL_H
#define NK_DOT_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

#include "numkong/types.h"
#include "numkong/dot/serial.h"
#include "numkong/reduce/haswell.h"
#include "numkong/cast/haswell.h" // `nk_f32x8_to_bf16x8_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Internal helper state for dot-products of low-precision types, where 32-bit accumulation is enough.
 *  @sa nk_dot_f16x8_state_haswell_t, nk_dot_bf16x8_state_haswell_t
 *  @sa nk_dot_e4m3x16_state_haswell_t, nk_dot_e5m2x16_state_haswell_t
 */
typedef struct nk_dot_through_f32_state_haswell_t_ {
    __m256 sum_f32x8;
} nk_dot_through_f32_state_haswell_t_;

/**
 *  @brief Initializes 32-bit accumulators for low-precision dot-products.
 *  @sa nk_dot_f16x8_init_haswell, nk_dot_bf16x8_init_haswell
 *  @sa nk_dot_e4m3x16_init_haswell, nk_dot_e5m2x16_init_haswell
 */
NK_INTERNAL void nk_dot_through_f32_init_haswell_(nk_dot_through_f32_state_haswell_t_ *state) {
    state->sum_f32x8 = _mm256_setzero_ps();
}

/**
 *  @brief Fuses 32-bit multiplication and accumulation for low-precision dot-products.
 *  @sa nk_dot_f16x8_update_haswell, nk_dot_bf16x8_update_haswell
 *  @sa nk_dot_e4m3x16_update_haswell, nk_dot_e5m2x16_update_haswell
 */
NK_INTERNAL void nk_dot_through_f32_update_haswell_(nk_dot_through_f32_state_haswell_t_ *state, nk_b256_vec_t a,
                                                    nk_b256_vec_t b, nk_size_t depth_offset,
                                                    nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    state->sum_f32x8 = _mm256_fmadd_ps(a.ymm_ps, b.ymm_ps, state->sum_f32x8);
}

/**
 *  @brief Finalizes 4x low-precision dot-products placing them into 4x consecutive 32-bit slots.
 *  @sa nk_dot_f16x8_finalize_haswell, nk_dot_bf16x8_finalize_haswell
 *  @sa nk_dot_e4m3x16_finalize_haswell, nk_dot_e5m2x16_finalize_haswell
 *
 *  The goal of this kernel is simple - compute 4x horizontal reductions, each involving 8x floats.
 *  The lack of vectorized horizontal instruction implies many consecutive shuffles producing a tree-like
 *  reduction. This kernel allows combining some of those operations between different dot products.
 */
NK_INTERNAL void nk_dot_through_f32_finalize_haswell_(                                                      //
    nk_dot_through_f32_state_haswell_t_ const *state_a, nk_dot_through_f32_state_haswell_t_ const *state_b, //
    nk_dot_through_f32_state_haswell_t_ const *state_c, nk_dot_through_f32_state_haswell_t_ const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);

    __m256 const sum_a_f32x8 = state_a->sum_f32x8, sum_b_f32x8 = state_b->sum_f32x8, sum_c_f32x8 = state_c->sum_f32x8,
                 sum_d_f32x8 = state_d->sum_f32x8;

    // ILP-optimized 4-way horizontal reduction for f32 in AVX2
    __m128 sum_a_f32x4 = _mm_add_ps(_mm256_castps256_ps128(sum_a_f32x8), _mm256_extractf128_ps(sum_a_f32x8, 1));
    __m128 sum_b_f32x4 = _mm_add_ps(_mm256_castps256_ps128(sum_b_f32x8), _mm256_extractf128_ps(sum_b_f32x8, 1));
    __m128 sum_c_f32x4 = _mm_add_ps(_mm256_castps256_ps128(sum_c_f32x8), _mm256_extractf128_ps(sum_c_f32x8, 1));
    __m128 sum_d_f32x4 = _mm_add_ps(_mm256_castps256_ps128(sum_d_f32x8), _mm256_extractf128_ps(sum_d_f32x8, 1));
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

NK_PUBLIC void nk_dot_f32_haswell(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count_scalars; idx_scalars += 4) {
        __m128 a_f32x4 = _mm_loadu_ps(a_scalars + idx_scalars);
        __m128 b_f32x4 = _mm_loadu_ps(b_scalars + idx_scalars);
        __m256d a_f64x4 = _mm256_cvtps_pd(a_f32x4);
        __m256d b_f64x4 = _mm256_cvtps_pd(b_f32x4);
        sum_f64x4 = _mm256_fmadd_pd(a_f64x4, b_f64x4, sum_f64x4);
    }
    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_f64_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = (nk_f32_t)sum;
}

NK_PUBLIC void nk_dot_f32c_haswell(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f32c_t *result) {
    // Using XOR to flip sign bits is cheaper than separate FMA/FMS. Throughput doubles from 2.5 GB/s to 5 GB/s.
    __m256d sum_real_f64x4 = _mm256_setzero_pd();
    __m256d sum_imag_f64x4 = _mm256_setzero_pd();
    __m256i sign_flip_i64x4 = _mm256_set_epi64x(0x8000000000000000, 0, 0x8000000000000000, 0);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 2 <= count_pairs; idx_pairs += 2) {
        __m128 a_f32x4 = _mm_loadu_ps((nk_f32_t const *)(a_pairs + idx_pairs));
        __m128 b_f32x4 = _mm_loadu_ps((nk_f32_t const *)(b_pairs + idx_pairs));
        __m256d a_f64x4 = _mm256_cvtps_pd(a_f32x4);
        __m256d b_f64x4 = _mm256_cvtps_pd(b_f32x4);
        __m256d b_swapped_f64x4 = _mm256_permute_pd(b_f64x4, 0x5); // 0b0101: swap adjacent pairs
        sum_real_f64x4 = _mm256_fmadd_pd(a_f64x4, b_f64x4, sum_real_f64x4);
        sum_imag_f64x4 = _mm256_fmadd_pd(a_f64x4, b_swapped_f64x4, sum_imag_f64x4);
    }
    // Flip the sign bit in every second f64 before accumulation:
    sum_real_f64x4 = _mm256_castsi256_pd(_mm256_xor_si256(_mm256_castpd_si256(sum_real_f64x4), sign_flip_i64x4));
    nk_f64_t sum_real = nk_reduce_add_f64x4_haswell_(sum_real_f64x4);
    nk_f64_t sum_imag = nk_reduce_add_f64x4_haswell_(sum_imag_f64x4);
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        sum_real += (nk_f64_t)a_pair.real * b_pair.real - (nk_f64_t)a_pair.imag * b_pair.imag;
        sum_imag += (nk_f64_t)a_pair.real * b_pair.imag + (nk_f64_t)a_pair.imag * b_pair.real;
    }
    result->real = (nk_f32_t)sum_real;
    result->imag = (nk_f32_t)sum_imag;
}

NK_PUBLIC void nk_vdot_f32c_haswell(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *result) {
    __m256d sum_real_f64x4 = _mm256_setzero_pd();
    __m256d sum_imag_f64x4 = _mm256_setzero_pd();
    __m256i sign_flip_i64x4 = _mm256_set_epi64x(0x8000000000000000, 0, 0x8000000000000000, 0);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 2 <= count_pairs; idx_pairs += 2) {
        __m128 a_f32x4 = _mm_loadu_ps((nk_f32_t const *)(a_pairs + idx_pairs));
        __m128 b_f32x4 = _mm_loadu_ps((nk_f32_t const *)(b_pairs + idx_pairs));
        __m256d a_f64x4 = _mm256_cvtps_pd(a_f32x4);
        __m256d b_f64x4 = _mm256_cvtps_pd(b_f32x4);
        sum_real_f64x4 = _mm256_fmadd_pd(a_f64x4, b_f64x4, sum_real_f64x4);
        __m256d b_swapped_f64x4 = _mm256_permute_pd(b_f64x4, 0x5); // 0b0101: swap adjacent pairs
        sum_imag_f64x4 = _mm256_fmadd_pd(a_f64x4, b_swapped_f64x4, sum_imag_f64x4);
    }
    // Flip the sign bit in every second f64 before accumulation:
    sum_imag_f64x4 = _mm256_castsi256_pd(_mm256_xor_si256(_mm256_castpd_si256(sum_imag_f64x4), sign_flip_i64x4));
    nk_f64_t sum_real = nk_reduce_add_f64x4_haswell_(sum_real_f64x4);
    nk_f64_t sum_imag = nk_reduce_add_f64x4_haswell_(sum_imag_f64x4);
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        sum_real += (nk_f64_t)a_pair.real * b_pair.real + (nk_f64_t)a_pair.imag * b_pair.imag;
        sum_imag += (nk_f64_t)a_pair.real * b_pair.imag - (nk_f64_t)a_pair.imag * b_pair.real;
    }
    result->real = (nk_f32_t)sum_real;
    result->imag = (nk_f32_t)sum_imag;
}

NK_PUBLIC void nk_dot_f64_haswell(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f64_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated dot product
    __m256d sum_f64x4 = _mm256_setzero_pd();
    __m256d compensation_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count_scalars; idx_scalars += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a_scalars + idx_scalars);
        __m256d b_f64x4 = _mm256_loadu_pd(b_scalars + idx_scalars);
        // TwoProd: h = a * b, r = fma(a, b, -h) captures the rounding error
        __m256d product_f64x4 = _mm256_mul_pd(a_f64x4, b_f64x4);
        __m256d product_error_f64x4 = _mm256_fmsub_pd(a_f64x4, b_f64x4, product_f64x4);
        // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
        __m256d t_f64x4 = _mm256_add_pd(sum_f64x4, product_f64x4);
        __m256d z_f64x4 = _mm256_sub_pd(t_f64x4, sum_f64x4);
        __m256d sum_error_f64x4 = _mm256_add_pd(_mm256_sub_pd(sum_f64x4, _mm256_sub_pd(t_f64x4, z_f64x4)),
                                                _mm256_sub_pd(product_f64x4, z_f64x4));
        // Update: sum = t, compensation += q + r
        sum_f64x4 = t_f64x4;
        compensation_f64x4 = _mm256_add_pd(compensation_f64x4, _mm256_add_pd(sum_error_f64x4, product_error_f64x4));
    }
    // Reduce and combine sum + compensation
    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(sum_f64x4, compensation_f64x4));
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_f64c_haswell(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f64c_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated complex dot product
    __m256d sum_real_f64x4 = _mm256_setzero_pd();
    __m256d sum_imag_f64x4 = _mm256_setzero_pd();
    __m256d compensation_real_f64x4 = _mm256_setzero_pd();
    __m256d compensation_imag_f64x4 = _mm256_setzero_pd();
    __m256i sign_flip_i64x4 = _mm256_set_epi64x(0x8000000000000000, 0, 0x8000000000000000, 0);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 2 <= count_pairs; idx_pairs += 2) {
        __m256d a_f64x4 = _mm256_loadu_pd((nk_f64_t const *)(a_pairs + idx_pairs));
        __m256d b_f64x4 = _mm256_loadu_pd((nk_f64_t const *)(b_pairs + idx_pairs));
        __m256d b_swapped_f64x4 = _mm256_permute_pd(b_f64x4, 0x5); // 0b0101: swap adjacent pairs

        // TwoProd for real part: a * b
        __m256d product_real_f64x4 = _mm256_mul_pd(a_f64x4, b_f64x4);
        __m256d product_real_error_f64x4 = _mm256_fmsub_pd(a_f64x4, b_f64x4, product_real_f64x4);
        // TwoSum for real part
        __m256d t_real_f64x4 = _mm256_add_pd(sum_real_f64x4, product_real_f64x4);
        __m256d z_real_f64x4 = _mm256_sub_pd(t_real_f64x4, sum_real_f64x4);
        __m256d sum_real_error_f64x4 = _mm256_add_pd(
            _mm256_sub_pd(sum_real_f64x4, _mm256_sub_pd(t_real_f64x4, z_real_f64x4)),
            _mm256_sub_pd(product_real_f64x4, z_real_f64x4));
        sum_real_f64x4 = t_real_f64x4;
        compensation_real_f64x4 = _mm256_add_pd(compensation_real_f64x4,
                                                _mm256_add_pd(sum_real_error_f64x4, product_real_error_f64x4));

        // TwoProd for imag part: a * b_swapped
        __m256d product_imag_f64x4 = _mm256_mul_pd(a_f64x4, b_swapped_f64x4);
        __m256d product_imag_error_f64x4 = _mm256_fmsub_pd(a_f64x4, b_swapped_f64x4, product_imag_f64x4);
        // TwoSum for imag part
        __m256d t_imag_f64x4 = _mm256_add_pd(sum_imag_f64x4, product_imag_f64x4);
        __m256d z_imag_f64x4 = _mm256_sub_pd(t_imag_f64x4, sum_imag_f64x4);
        __m256d sum_imag_error_f64x4 = _mm256_add_pd(
            _mm256_sub_pd(sum_imag_f64x4, _mm256_sub_pd(t_imag_f64x4, z_imag_f64x4)),
            _mm256_sub_pd(product_imag_f64x4, z_imag_f64x4));
        sum_imag_f64x4 = t_imag_f64x4;
        compensation_imag_f64x4 = _mm256_add_pd(compensation_imag_f64x4,
                                                _mm256_add_pd(sum_imag_error_f64x4, product_imag_error_f64x4));
    }
    // Flip sign in every second f64 for real part (to get a_r*b_r - a_i*b_i)
    sum_real_f64x4 = _mm256_castsi256_pd(_mm256_xor_si256(_mm256_castpd_si256(sum_real_f64x4), sign_flip_i64x4));
    compensation_real_f64x4 = _mm256_castsi256_pd(
        _mm256_xor_si256(_mm256_castpd_si256(compensation_real_f64x4), sign_flip_i64x4));
    // Reduce and combine: first vector-add sum+compensation, then horizontal reduce
    nk_f64_t sum_real = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(sum_real_f64x4, compensation_real_f64x4));
    nk_f64_t sum_imag = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(sum_imag_f64x4, compensation_imag_f64x4));
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f64c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        sum_real += a_pair.real * b_pair.real - a_pair.imag * b_pair.imag;
        sum_imag += a_pair.real * b_pair.imag + a_pair.imag * b_pair.real;
    }
    result->real = sum_real;
    result->imag = sum_imag;
}

NK_PUBLIC void nk_vdot_f64c_haswell(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f64c_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated conjugate dot product
    __m256d sum_real_f64x4 = _mm256_setzero_pd();
    __m256d sum_imag_f64x4 = _mm256_setzero_pd();
    __m256d compensation_real_f64x4 = _mm256_setzero_pd();
    __m256d compensation_imag_f64x4 = _mm256_setzero_pd();
    __m256i sign_flip_i64x4 = _mm256_set_epi64x(0x8000000000000000, 0, 0x8000000000000000, 0);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 2 <= count_pairs; idx_pairs += 2) {
        __m256d a_f64x4 = _mm256_loadu_pd((nk_f64_t const *)(a_pairs + idx_pairs));
        __m256d b_f64x4 = _mm256_loadu_pd((nk_f64_t const *)(b_pairs + idx_pairs));
        __m256d b_swapped_f64x4 = _mm256_permute_pd(b_f64x4, 0x5); // 0b0101: swap adjacent pairs

        // TwoProd for real part: a * b
        __m256d product_real_f64x4 = _mm256_mul_pd(a_f64x4, b_f64x4);
        __m256d product_real_error_f64x4 = _mm256_fmsub_pd(a_f64x4, b_f64x4, product_real_f64x4);
        // TwoSum for real part
        __m256d t_real_f64x4 = _mm256_add_pd(sum_real_f64x4, product_real_f64x4);
        __m256d z_real_f64x4 = _mm256_sub_pd(t_real_f64x4, sum_real_f64x4);
        __m256d sum_real_error_f64x4 = _mm256_add_pd(
            _mm256_sub_pd(sum_real_f64x4, _mm256_sub_pd(t_real_f64x4, z_real_f64x4)),
            _mm256_sub_pd(product_real_f64x4, z_real_f64x4));
        sum_real_f64x4 = t_real_f64x4;
        compensation_real_f64x4 = _mm256_add_pd(compensation_real_f64x4,
                                                _mm256_add_pd(sum_real_error_f64x4, product_real_error_f64x4));

        // TwoProd for imag part: a * b_swapped
        __m256d product_imag_f64x4 = _mm256_mul_pd(a_f64x4, b_swapped_f64x4);
        __m256d product_imag_error_f64x4 = _mm256_fmsub_pd(a_f64x4, b_swapped_f64x4, product_imag_f64x4);
        // TwoSum for imag part
        __m256d t_imag_f64x4 = _mm256_add_pd(sum_imag_f64x4, product_imag_f64x4);
        __m256d z_imag_f64x4 = _mm256_sub_pd(t_imag_f64x4, sum_imag_f64x4);
        __m256d sum_imag_error_f64x4 = _mm256_add_pd(
            _mm256_sub_pd(sum_imag_f64x4, _mm256_sub_pd(t_imag_f64x4, z_imag_f64x4)),
            _mm256_sub_pd(product_imag_f64x4, z_imag_f64x4));
        sum_imag_f64x4 = t_imag_f64x4;
        compensation_imag_f64x4 = _mm256_add_pd(compensation_imag_f64x4,
                                                _mm256_add_pd(sum_imag_error_f64x4, product_imag_error_f64x4));
    }
    // Flip sign in every second f64 for imag part (to get a_r*b_i - a_i*b_r)
    sum_imag_f64x4 = _mm256_castsi256_pd(_mm256_xor_si256(_mm256_castpd_si256(sum_imag_f64x4), sign_flip_i64x4));
    compensation_imag_f64x4 = _mm256_castsi256_pd(
        _mm256_xor_si256(_mm256_castpd_si256(compensation_imag_f64x4), sign_flip_i64x4));
    // Reduce and combine: first vector-add sum+compensation, then horizontal reduce
    nk_f64_t sum_real = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(sum_real_f64x4, compensation_real_f64x4));
    nk_f64_t sum_imag = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(sum_imag_f64x4, compensation_imag_f64x4));
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f64c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        sum_real += a_pair.real * b_pair.real + a_pair.imag * b_pair.imag;
        sum_imag += a_pair.real * b_pair.imag - a_pair.imag * b_pair.real;
    }
    result->real = sum_real;
    result->imag = sum_imag;
}

NK_PUBLIC void nk_dot_f16_haswell(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_dot_f16_haswell_cycle:
    if (count_scalars < 8) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_f16x8_to_f32x8_haswell_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_f16x8_to_f32x8_haswell_(b_scalars, &b_vec, count_scalars);
        a_f32x8 = a_vec.ymm_ps;
        b_f32x8 = b_vec.ymm_ps;
        count_scalars = 0;
    }
    else {
        a_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)a_scalars));
        b_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)b_scalars));
        count_scalars -= 8, a_scalars += 8, b_scalars += 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_f32x8);
    if (count_scalars) goto nk_dot_f16_haswell_cycle;
    *result = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

NK_PUBLIC void nk_dot_f16c_haswell(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f32c_t *result) {
    __m256 sum_real_f32x8 = _mm256_setzero_ps();
    __m256 sum_imag_f32x8 = _mm256_setzero_ps();
    __m256i sign_flip_i64x4 = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_i8x32 = _mm256_set_epi8( //
        11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
    while (count_pairs >= 4) {
        __m256 a_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)a_pairs));
        __m256 b_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)b_pairs));
        __m256 b_swapped_f32x8 = _mm256_castsi256_ps(
            _mm256_shuffle_epi8(_mm256_castps_si256(b_f32x8), swap_adjacent_i8x32));
        sum_real_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_real_f32x8);
        sum_imag_f32x8 = _mm256_fmadd_ps(a_f32x8, b_swapped_f32x8, sum_imag_f32x8);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Flip the sign bit in every second scalar before accumulation:
    sum_real_f32x8 = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(sum_real_f32x8), sign_flip_i64x4));
    nk_f32c_t tail_result;
    nk_dot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_real_f32x8);
    result->imag = tail_result.imag + (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_imag_f32x8);
}

NK_PUBLIC void nk_vdot_f16c_haswell(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *result) {
    __m256 sum_real_f32x8 = _mm256_setzero_ps();
    __m256 sum_imag_f32x8 = _mm256_setzero_ps();
    __m256i sign_flip_i64x4 = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_i8x32 = _mm256_set_epi8( //
        11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
    while (count_pairs >= 4) {
        __m256 a_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)a_pairs));
        __m256 b_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)b_pairs));
        sum_real_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_real_f32x8);
        b_f32x8 = _mm256_castsi256_ps(_mm256_shuffle_epi8(_mm256_castps_si256(b_f32x8), swap_adjacent_i8x32));
        sum_imag_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_imag_f32x8);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Flip the sign bit in every second scalar before accumulation:
    sum_imag_f32x8 = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(sum_imag_f32x8), sign_flip_i64x4));
    nk_f32c_t tail_result;
    nk_vdot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_real_f32x8);
    result->imag = tail_result.imag + (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_imag_f32x8);
}

NK_PUBLIC void nk_dot_i8_haswell(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_i32_t *result) {
    __m256i sum_low_i32x8 = _mm256_setzero_si256();
    __m256i sum_high_i32x8 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    // Use two 128-bit loads instead of 256-bit load + extract to avoid Port 5 contention.
    // VEXTRACTI128 uses Port 5; two smaller loads use Port 2/3 (2 ports available).
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        __m128i a_low_i8x16 = _mm_loadu_si128((__m128i const *)(a_scalars + idx_scalars));
        __m128i a_high_i8x16 = _mm_loadu_si128((__m128i const *)(a_scalars + idx_scalars + 16));
        __m128i b_low_i8x16 = _mm_loadu_si128((__m128i const *)(b_scalars + idx_scalars));
        __m128i b_high_i8x16 = _mm_loadu_si128((__m128i const *)(b_scalars + idx_scalars + 16));
        // Upcast `int8` to `int16` - no extracts needed
        __m256i a_low_i16x16 = _mm256_cvtepi8_epi16(a_low_i8x16);
        __m256i a_high_i16x16 = _mm256_cvtepi8_epi16(a_high_i8x16);
        __m256i b_low_i16x16 = _mm256_cvtepi8_epi16(b_low_i8x16);
        __m256i b_high_i16x16 = _mm256_cvtepi8_epi16(b_high_i8x16);
        // Multiply and accumulate at `int16` level, accumulate at `int32` level
        sum_low_i32x8 = _mm256_add_epi32(sum_low_i32x8, _mm256_madd_epi16(a_low_i16x16, b_low_i16x16));
        sum_high_i32x8 = _mm256_add_epi32(sum_high_i32x8, _mm256_madd_epi16(a_high_i16x16, b_high_i16x16));
    }
    nk_i32_t sum = nk_reduce_add_i32x8_haswell_(_mm256_add_epi32(sum_low_i32x8, sum_high_i32x8));
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_i32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_u8_haswell(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_u32_t *result) {
    __m256i sum_low_i32x8 = _mm256_setzero_si256();
    __m256i sum_high_i32x8 = _mm256_setzero_si256();
    __m256i const zeros_i8x32 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)(b_scalars + idx_scalars));
        // Upcast `uint8` to `int16`. Unpacking is faster than extracts.
        __m256i a_low_i16x16 = _mm256_unpacklo_epi8(a_u8x32, zeros_i8x32);
        __m256i a_high_i16x16 = _mm256_unpackhi_epi8(a_u8x32, zeros_i8x32);
        __m256i b_low_i16x16 = _mm256_unpacklo_epi8(b_u8x32, zeros_i8x32);
        __m256i b_high_i16x16 = _mm256_unpackhi_epi8(b_u8x32, zeros_i8x32);
        // Multiply and accumulate at `int16` level, accumulate at `int32` level
        sum_low_i32x8 = _mm256_add_epi32(sum_low_i32x8, _mm256_madd_epi16(a_low_i16x16, b_low_i16x16));
        sum_high_i32x8 = _mm256_add_epi32(sum_high_i32x8, _mm256_madd_epi16(a_high_i16x16, b_high_i16x16));
    }
    nk_u32_t sum = (nk_u32_t)nk_reduce_add_i32x8_haswell_(_mm256_add_epi32(sum_low_i32x8, sum_high_i32x8));
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_u32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_bf16_haswell(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m128i a_bf16x8, b_bf16x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_dot_bf16_haswell_cycle:
    if (count_scalars < 8) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b16x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b16x16_serial_(b_scalars, &b_vec, count_scalars);
        a_bf16x8 = a_vec.xmms[0];
        b_bf16x8 = b_vec.xmms[0];
        count_scalars = 0;
    }
    else {
        a_bf16x8 = _mm_loadu_si128((__m128i const *)a_scalars);
        b_bf16x8 = _mm_loadu_si128((__m128i const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(nk_bf16x8_to_f32x8_haswell_(a_bf16x8), nk_bf16x8_to_f32x8_haswell_(b_bf16x8),
                                sum_f32x8);
    if (count_scalars) goto nk_dot_bf16_haswell_cycle;
    *result = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

NK_PUBLIC void nk_dot_e4m3_haswell(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_dot_e4m3_haswell_cycle:
    if (count_scalars < 8) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_e4m3x8_to_f32x8_haswell_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_e4m3x8_to_f32x8_haswell_(b_scalars, &b_vec, count_scalars);
        a_f32x8 = a_vec.ymm_ps;
        b_f32x8 = b_vec.ymm_ps;
        count_scalars = 0;
    }
    else {
        a_f32x8 = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)a_scalars));
        b_f32x8 = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)b_scalars));
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_f32x8);
    if (count_scalars) goto nk_dot_e4m3_haswell_cycle;
    *result = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

NK_PUBLIC void nk_dot_e5m2_haswell(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_dot_e5m2_haswell_cycle:
    if (count_scalars < 8) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_e5m2x8_to_f32x8_haswell_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_e5m2x8_to_f32x8_haswell_(b_scalars, &b_vec, count_scalars);
        a_f32x8 = a_vec.ymm_ps;
        b_f32x8 = b_vec.ymm_ps;
        count_scalars = 0;
    }
    else {
        a_f32x8 = nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)a_scalars));
        b_f32x8 = nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)b_scalars));
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_f32x8);
    if (count_scalars) goto nk_dot_e5m2_haswell_cycle;
    *result = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

NK_PUBLIC void nk_dot_e2m3_haswell(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_dot_e2m3_haswell_cycle:
    if (count_scalars < 8) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_e2m3x8_to_f32x8_haswell_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_e2m3x8_to_f32x8_haswell_(b_scalars, &b_vec, count_scalars);
        a_f32x8 = a_vec.ymm_ps;
        b_f32x8 = b_vec.ymm_ps;
        count_scalars = 0;
    }
    else {
        a_f32x8 = nk_e2m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)a_scalars));
        b_f32x8 = nk_e2m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)b_scalars));
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_f32x8);
    if (count_scalars) goto nk_dot_e2m3_haswell_cycle;
    *result = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

NK_PUBLIC void nk_dot_e3m2_haswell(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_dot_e3m2_haswell_cycle:
    if (count_scalars < 8) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_e3m2x8_to_f32x8_haswell_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_e3m2x8_to_f32x8_haswell_(b_scalars, &b_vec, count_scalars);
        a_f32x8 = a_vec.ymm_ps;
        b_f32x8 = b_vec.ymm_ps;
        count_scalars = 0;
    }
    else {
        a_f32x8 = nk_e3m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)a_scalars));
        b_f32x8 = nk_e3m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)b_scalars));
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_f32x8);
    if (count_scalars) goto nk_dot_e3m2_haswell_cycle;
    *result = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

struct nk_dot_f32x4_state_haswell_t {
    __m256d sum_f64x4;
};

NK_INTERNAL void nk_dot_f32x4_init_haswell(nk_dot_f32x4_state_haswell_t *state) {
    state->sum_f64x4 = _mm256_setzero_pd();
}

NK_INTERNAL void nk_dot_f32x4_update_haswell(nk_dot_f32x4_state_haswell_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Upcast 4 f32s to f64s for high-precision accumulation
    __m256d a_f64x4 = _mm256_cvtps_pd(_mm_castsi128_ps(a.xmm));
    __m256d b_f64x4 = _mm256_cvtps_pd(_mm_castsi128_ps(b.xmm));
    // FMA accumulation in f64
    state->sum_f64x4 = _mm256_fmadd_pd(a_f64x4, b_f64x4, state->sum_f64x4);
}

NK_INTERNAL void nk_dot_f32x4_finalize_haswell(                                               //
    nk_dot_f32x4_state_haswell_t const *state_a, nk_dot_f32x4_state_haswell_t const *state_b, //
    nk_dot_f32x4_state_haswell_t const *state_c, nk_dot_f32x4_state_haswell_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // Horizontal reduction: 4 f64s → 1 f64 for each state
    // Then downcast final f64 results to f32
    __m256d sum_a_f64x4 = state_a->sum_f64x4;
    __m256d sum_b_f64x4 = state_b->sum_f64x4;
    __m256d sum_c_f64x4 = state_c->sum_f64x4;
    __m256d sum_d_f64x4 = state_d->sum_f64x4;

    // 4 → 2: add high 128-bit lane to low lane
    __m128d sum_a_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_a_f64x4), _mm256_extractf128_pd(sum_a_f64x4, 1));
    __m128d sum_b_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_b_f64x4), _mm256_extractf128_pd(sum_b_f64x4, 1));
    __m128d sum_c_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_c_f64x4), _mm256_extractf128_pd(sum_c_f64x4, 1));
    __m128d sum_d_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_d_f64x4), _mm256_extractf128_pd(sum_d_f64x4, 1));

    // 2 → 1: horizontal add
    __m128d sum_ab_f64x2 = _mm_hadd_pd(sum_a_f64x2, sum_b_f64x2); // [sum_a, sum_b]
    __m128d sum_cd_f64x2 = _mm_hadd_pd(sum_c_f64x2, sum_d_f64x2); // [sum_c, sum_d]

    // Combine into __m256d and convert to f32
    __m256d sum_abcd_f64x4 = _mm256_set_m128d(sum_cd_f64x2, sum_ab_f64x2);
    __m128 sum_f32x4 = _mm256_cvtpd_ps(sum_abcd_f64x4);

    result->xmm = _mm_castps_si128(sum_f32x4);
}

NK_INTERNAL void nk_load_f16x8_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm_ps = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)src));
}

NK_INTERNAL void nk_dots_partial_load_f16x8_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t f16_partial;
    nk_partial_load_b16x8_serial_(src, &f16_partial, n);
    dst->ymm_ps = _mm256_cvtph_ps(f16_partial.xmm);
}

NK_INTERNAL void nk_load_bf16x8_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm_ps = nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)src));
}

NK_INTERNAL void nk_dots_partial_load_bf16x8_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t bf16_partial;
    nk_partial_load_b16x8_serial_(src, &bf16_partial, n);
    dst->ymm_ps = nk_bf16x8_to_f32x8_haswell_(bf16_partial.xmm);
}

NK_INTERNAL void nk_load_e4m3x16_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm_ps = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)src));
}

NK_INTERNAL void nk_dots_partial_load_e4m3x16_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t e4m3_partial;
    nk_partial_load_b8x16_serial_(src, &e4m3_partial, n);
    dst->ymm_ps = nk_e4m3x8_to_f32x8_haswell_(e4m3_partial.xmm);
}

NK_INTERNAL void nk_load_e5m2x16_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm_ps = nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)src));
}

NK_INTERNAL void nk_dots_partial_load_e5m2x16_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t e5m2_partial;
    nk_partial_load_b8x16_serial_(src, &e5m2_partial, n);
    dst->ymm_ps = nk_e5m2x8_to_f32x8_haswell_(e5m2_partial.xmm);
}

NK_INTERNAL void nk_load_e2m3x16_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm_ps = nk_e2m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)src));
}

NK_INTERNAL void nk_dots_partial_load_e2m3x16_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t e2m3_partial;
    nk_partial_load_b8x16_serial_(src, &e2m3_partial, n);
    dst->ymm_ps = nk_e2m3x8_to_f32x8_haswell_(e2m3_partial.xmm);
}

NK_INTERNAL void nk_load_e3m2x16_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm_ps = nk_e3m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)src));
}

NK_INTERNAL void nk_dots_partial_load_e3m2x16_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t e3m2_partial;
    nk_partial_load_b8x16_serial_(src, &e3m2_partial, n);
    dst->ymm_ps = nk_e3m2x8_to_f32x8_haswell_(e3m2_partial.xmm);
}

struct nk_dot_i8x16_state_haswell_t {
    __m256i sum_i32x8;
};

NK_INTERNAL void nk_dot_i8x16_init_haswell(nk_dot_i8x16_state_haswell_t *state) {
    state->sum_i32x8 = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_i8x16_update_haswell(nk_dot_i8x16_state_haswell_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m256i a_i16x16 = _mm256_cvtepi8_epi16(a.xmm);
    __m256i b_i16x16 = _mm256_cvtepi8_epi16(b.xmm);
    state->sum_i32x8 = _mm256_add_epi32(state->sum_i32x8, _mm256_madd_epi16(a_i16x16, b_i16x16));
}

NK_INTERNAL void nk_dot_i8x16_finalize_haswell(                                               //
    nk_dot_i8x16_state_haswell_t const *state_a, nk_dot_i8x16_state_haswell_t const *state_b, //
    nk_dot_i8x16_state_haswell_t const *state_c, nk_dot_i8x16_state_haswell_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // ILP-optimized 4-way horizontal reduction for i32 in AVX2
    // Step 1: 8->4 for all 4 states
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_a->sum_i32x8),
                                        _mm256_extracti128_si256(state_a->sum_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_b->sum_i32x8),
                                        _mm256_extracti128_si256(state_b->sum_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_c->sum_i32x8),
                                        _mm256_extracti128_si256(state_c->sum_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_d->sum_i32x8),
                                        _mm256_extracti128_si256(state_d->sum_i32x8, 1));
    // Step 2: Transpose 4×4 matrix
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i sum_lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    // Step 3: Vertical sum and store as i32
    __m128i sum_i32x4 = _mm_add_epi32(_mm_add_epi32(sum_lane0_i32x4, sum_lane1_i32x4),
                                      _mm_add_epi32(sum_lane2_i32x4, sum_lane3_i32x4));
    result->xmm = sum_i32x4;
}

/**
 *  @brief Running state for 128-bit dot accumulation over u8 scalars on Haswell.
 */
struct nk_dot_u8x16_state_haswell_t {
    __m256i sum_i32x8;
};

NK_INTERNAL void nk_dot_u8x16_init_haswell(nk_dot_u8x16_state_haswell_t *state) {
    state->sum_i32x8 = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_u8x16_update_haswell(nk_dot_u8x16_state_haswell_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m256i a_i16x16 = _mm256_cvtepu8_epi16(a.xmm);
    __m256i b_i16x16 = _mm256_cvtepu8_epi16(b.xmm);
    state->sum_i32x8 = _mm256_add_epi32(state->sum_i32x8, _mm256_madd_epi16(a_i16x16, b_i16x16));
}

NK_INTERNAL void nk_dot_u8x16_finalize_haswell(                                               //
    nk_dot_u8x16_state_haswell_t const *state_a, nk_dot_u8x16_state_haswell_t const *state_b, //
    nk_dot_u8x16_state_haswell_t const *state_c, nk_dot_u8x16_state_haswell_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_dot_i8x16_finalize_haswell(                                                                    //
        (nk_dot_i8x16_state_haswell_t const *)state_a, (nk_dot_i8x16_state_haswell_t const *)state_b, //
        (nk_dot_i8x16_state_haswell_t const *)state_c, (nk_dot_i8x16_state_haswell_t const *)state_d, total_dimensions,
        result);
}

/**
 *  @brief Running state for 256-bit dot accumulation over f64 scalars on Haswell.
 *
 *  Uses the Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated dot product.
 */
typedef struct nk_dot_f64x4_state_haswell_t {
    __m256d sum_f64x4;
    __m256d compensation_f64x4; // Error accumulator for Dot2
} nk_dot_f64x4_state_haswell_t;

NK_INTERNAL void nk_dot_f64x4_init_haswell(nk_dot_f64x4_state_haswell_t *state) {
    state->sum_f64x4 = _mm256_setzero_pd();
    state->compensation_f64x4 = _mm256_setzero_pd();
}

NK_INTERNAL void nk_dot_f64x4_update_haswell(nk_dot_f64x4_state_haswell_t *state, nk_b256_vec_t a, nk_b256_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m256d sum_f64x4 = state->sum_f64x4;
    __m256d compensation_f64x4 = state->compensation_f64x4;
    __m256d a_f64x4 = a.ymm_pd;
    __m256d b_f64x4 = b.ymm_pd;

    // TwoProd: h = a * b, r = fma(a, b, -h) captures the rounding error
    __m256d product_f64x4 = _mm256_mul_pd(a_f64x4, b_f64x4);
    __m256d product_error_f64x4 = _mm256_fmsub_pd(a_f64x4, b_f64x4, product_f64x4);

    // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
    __m256d t_f64x4 = _mm256_add_pd(sum_f64x4, product_f64x4);
    __m256d z_f64x4 = _mm256_sub_pd(t_f64x4, sum_f64x4);
    __m256d sum_error_f64x4 = _mm256_add_pd(_mm256_sub_pd(sum_f64x4, _mm256_sub_pd(t_f64x4, z_f64x4)),
                                            _mm256_sub_pd(product_f64x4, z_f64x4));

    // Update: sum = t, compensation += q + r
    state->sum_f64x4 = t_f64x4;
    state->compensation_f64x4 = _mm256_add_pd(compensation_f64x4, _mm256_add_pd(sum_error_f64x4, product_error_f64x4));
}

NK_INTERNAL void nk_dot_f64x4_finalize_haswell(                                               //
    nk_dot_f64x4_state_haswell_t const *state_a, nk_dot_f64x4_state_haswell_t const *state_b, //
    nk_dot_f64x4_state_haswell_t const *state_c, nk_dot_f64x4_state_haswell_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    // Combine sum + compensation before horizontal reduction
    __m256d sum_a_f64x4 = _mm256_add_pd(state_a->sum_f64x4, state_a->compensation_f64x4);
    __m256d sum_b_f64x4 = _mm256_add_pd(state_b->sum_f64x4, state_b->compensation_f64x4);
    __m256d sum_c_f64x4 = _mm256_add_pd(state_c->sum_f64x4, state_c->compensation_f64x4);
    __m256d sum_d_f64x4 = _mm256_add_pd(state_d->sum_f64x4, state_d->compensation_f64x4);

    // ILP-optimized 4-way horizontal reduction for f64 in AVX2
    __m128d sum_a_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_a_f64x4), _mm256_extractf128_pd(sum_a_f64x4, 1));
    __m128d sum_b_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_b_f64x4), _mm256_extractf128_pd(sum_b_f64x4, 1));
    __m128d sum_c_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_c_f64x4), _mm256_extractf128_pd(sum_c_f64x4, 1));
    __m128d sum_d_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_d_f64x4), _mm256_extractf128_pd(sum_d_f64x4, 1));
    // Horizontal add pairs: [a0+a1, b0+b1] and [c0+c1, d0+d1]
    __m128d sum_ab_f64x2 = _mm_hadd_pd(sum_a_f64x2, sum_b_f64x2);
    __m128d sum_cd_f64x2 = _mm_hadd_pd(sum_c_f64x2, sum_d_f64x2);
    // Store results in ymm register
    result->ymm_pd = _mm256_set_m128d(sum_cd_f64x2, sum_ab_f64x2);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_

#endif // NK_DOT_HASWELL_H
