/**
 *  @brief SIMD-accelerated Dot Products for Haswell.
 *  @file include/numkong/dot/haswell.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_haswell_instructions Key AVX2/FMA Dot Product Instructions
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
 *
 *  @section dot_haswell_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines following structures and force-inlined
 *  `NK_INTERNAL` functions:
 *
 *  - nk_dot_f64x4 state with Dot2 stable dot-products,
 *  - nk_dot_f32x4 state with double-precision numerics,
 *  - nk_dot_through_f32 state for 16-, 8-, and 6-bit float inputs with single-precision numerics,
 *  - nk_dot_i8x16 for 8-bit signed integer inputs,
 *  - nk_dot_u8x16 for 8-bit unsigned integer inputs,
 *  - nk_dot_i4x32 for 4-bit signed integer products with 2 correction terms,
 *  - nk_dot_u4x32 for 4-bit unsigned integer products.
 *
 *  @code{c}
 *  nk_dot_i8x16_state_haswell_t state_first, state_second, state_third, state_fourth;
 *  nk_b128_vec_t query_i8x16, target_first_i8x16, target_second_i8x16, target_third_i8x16, target_fourth_i8x16,
 *  nk_dot_i8x16_init_haswell(&state_first);
 *  nk_dot_i8x16_init_haswell(&state_second);
 *  nk_dot_i8x16_init_haswell(&state_third);
 *  nk_dot_i8x16_init_haswell(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 16 <= depth; idx += 16) {
 *      query_i8x16.xmm = _mm_loadu_si128(query_ptr + idx);
 *      target_first_i8x16.xmm = _mm_loadu_si128(target_first_ptr + idx);
 *      target_second_i8x16.xmm = _mm_loadu_si128(target_second_ptr + idx);
 *      target_third_i8x16.xmm = _mm_loadu_si128(target_third_ptr + idx);
 *      target_fourth_i8x16.xmm = _mm_loadu_si128(target_fourth_ptr + idx);
 *      nk_dot_i8x16_update_haswell(&state_first, query_i8x16, target_first_i8x16, idx, 16);
 *      nk_dot_i8x16_update_haswell(&state_second, query_i8x16, target_second_i8x16, idx, 16);
 *      nk_dot_i8x16_update_haswell(&state_third, query_i8x16, target_third_i8x16, idx, 16);
 *      nk_dot_i8x16_update_haswell(&state_fourth, query_i8x16, target_fourth_i8x16, idx, 16);
 *  }
 *  nk_b128_vec_t results_i32x4;
 *  nk_dot_i8x16_finalize_haswell(&state_first, &state_second, &state_third, &state_fourth, depth, &results_i32x4);
 *  @endcode
 *
 *  Not every numeric type has dedicated dot-product SIMD circuitry on each ISA. Smaller float types
 *  like f16, bf16, e4m3, e5m2, e2m3, and e3m2 on Haswell use ISA-specific upcasting to f32 combined
 *  with native FMA instructions, sharing the `nk_dot_through_f32` accumulation logic:
 *
 *  @code{c}
 *  nk_dot_e4m3x16_state_haswell_t state_first, state_second, state_third, state_fourth;
 *  nk_b256_vec_t query_f32x8, target_first_f32x8, target_second_f32x8, target_third_f32x8, target_fourth_f32x8;
 *  nk_dot_through_f32_init_haswell_(&state_first);
 *  nk_dot_through_f32_init_haswell_(&state_second);
 *  nk_dot_through_f32_init_haswell_(&state_third);
 *  nk_dot_through_f32_init_haswell_(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 8 <= depth; idx += 8) {
 *      query_f32x8.ymm_ps = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64(query_ptr + idx));
 *      target_first_f32x8.ymm_ps = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64(target_first_ptr + idx));
 *      target_second_f32x8.ymm_ps = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64(target_second_ptr + idx));
 *      target_third_f32x8.ymm_ps = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64(target_third_ptr + idx));
 *      target_fourth_f32x8.ymm_ps = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64(target_fourth_ptr + idx));
 *      nk_dot_through_f32_update_haswell_(&state_first, query_f32x8, target_first_f32x8, idx, 8);
 *      nk_dot_through_f32_update_haswell_(&state_second, query_f32x8, target_second_f32x8, idx, 8);
 *      nk_dot_through_f32_update_haswell_(&state_third, query_f32x8, target_third_f32x8, idx, 8);
 *      nk_dot_through_f32_update_haswell_(&state_fourth, query_f32x8, target_fourth_f32x8, idx, 8);
 *  }
 *  nk_b128_vec_t results_f32x4;
 *  nk_dot_through_f32_finalize_haswell_(&state_first, &state_second, &state_third, &state_fourth,
 *      depth, &results_f32x4);
 *  @endcode
 */
#ifndef NK_DOT_HASWELL_H
#define NK_DOT_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/dot/serial.h"
#include "numkong/reduce/haswell.h"
#include "numkong/cast/haswell.h" // `nk_f32x8_to_bf16x8_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

#pragma region - Traditional Floats

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

typedef struct nk_dot_f32x4_state_haswell_t {
    __m256d sum_f64x4;
} nk_dot_f32x4_state_haswell_t;

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

#pragma endregion - Traditional Floats

#pragma region - Smaller Floats

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

NK_PUBLIC void nk_dot_bf16c_haswell(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *result) {
    // Convert BF16 to F32, then use F32 complex dot product with sign-flipping optimization.
    // Uses same XOR trick as f32c to double throughput by deferring sign flips until after loop.
    __m128i a_bf16x8, b_bf16x8;
    __m256 sum_real_f32x8 = _mm256_setzero_ps();
    __m256 sum_imag_f32x8 = _mm256_setzero_ps();
    __m256i const sign_flip_i64x4 = _mm256_set1_epi64x(0x8000000000000000);
    __m256i const swap_adjacent_i8x32 = _mm256_set_epi8( //
        11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);

nk_dot_bf16c_haswell_cycle:
    if (count_pairs < 4) {
        // Partial load using serial helper
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b16x16_serial_(a_pairs, &a_vec, count_pairs * 2);
        nk_partial_load_b16x16_serial_(b_pairs, &b_vec, count_pairs * 2);
        a_bf16x8 = a_vec.xmms[0];
        b_bf16x8 = b_vec.xmms[0];
        count_pairs = 0;
    }
    else {
        a_bf16x8 = _mm_loadu_si128((__m128i const *)a_pairs);
        b_bf16x8 = _mm_loadu_si128((__m128i const *)b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }

    // Convert BF16 to F32
    __m256 a_f32x8 = nk_bf16x8_to_f32x8_haswell_(a_bf16x8);
    __m256 b_f32x8 = nk_bf16x8_to_f32x8_haswell_(b_bf16x8);

    // Complex multiply-accumulate: swap b for imaginary part
    __m256 b_swapped_f32x8 = _mm256_castsi256_ps(
        _mm256_shuffle_epi8(_mm256_castps_si256(b_f32x8), swap_adjacent_i8x32));
    sum_real_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_real_f32x8);
    sum_imag_f32x8 = _mm256_fmadd_ps(a_f32x8, b_swapped_f32x8, sum_imag_f32x8);

    if (count_pairs) goto nk_dot_bf16c_haswell_cycle;

    // Flip the sign bit in every second scalar (real part: a_r*b_r - a_i*b_i)
    sum_real_f32x8 = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(sum_real_f32x8), sign_flip_i64x4));

    result->real = nk_reduce_add_f32x8_haswell_(sum_real_f32x8);
    result->imag = nk_reduce_add_f32x8_haswell_(sum_imag_f32x8);
}

NK_PUBLIC void nk_vdot_bf16c_haswell(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs, nk_size_t count_pairs,
                                     nk_f32c_t *result) {
    // Conjugate complex dot product: conj(a) * b
    __m128i a_bf16x8, b_bf16x8;
    __m256 sum_real_f32x8 = _mm256_setzero_ps();
    __m256 sum_imag_f32x8 = _mm256_setzero_ps();
    __m256i const sign_flip_i64x4 = _mm256_set1_epi64x(0x8000000000000000);
    __m256i const swap_adjacent_i8x32 = _mm256_set_epi8( //
        11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);

nk_vdot_bf16c_haswell_cycle:
    if (count_pairs < 4) {
        // Partial load using serial helper
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b16x16_serial_(a_pairs, &a_vec, count_pairs * 2);
        nk_partial_load_b16x16_serial_(b_pairs, &b_vec, count_pairs * 2);
        a_bf16x8 = a_vec.xmms[0];
        b_bf16x8 = b_vec.xmms[0];
        count_pairs = 0;
    }
    else {
        a_bf16x8 = _mm_loadu_si128((__m128i const *)a_pairs);
        b_bf16x8 = _mm_loadu_si128((__m128i const *)b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }

    // Convert BF16 to F32
    __m256 a_f32x8 = nk_bf16x8_to_f32x8_haswell_(a_bf16x8);
    __m256 b_f32x8 = nk_bf16x8_to_f32x8_haswell_(b_bf16x8);

    // Conjugate complex multiply-accumulate
    sum_real_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_real_f32x8);
    __m256 b_swapped_f32x8 = _mm256_castsi256_ps(
        _mm256_shuffle_epi8(_mm256_castps_si256(b_f32x8), swap_adjacent_i8x32));
    sum_imag_f32x8 = _mm256_fmadd_ps(a_f32x8, b_swapped_f32x8, sum_imag_f32x8);

    if (count_pairs) goto nk_vdot_bf16c_haswell_cycle;

    // Flip the sign bit in every second scalar (imag part: a_r*b_i - a_i*b_r)
    sum_imag_f32x8 = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(sum_imag_f32x8), sign_flip_i64x4));

    result->real = nk_reduce_add_f32x8_haswell_(sum_real_f32x8);
    result->imag = nk_reduce_add_f32x8_haswell_(sum_imag_f32x8);
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
    // Integer dot product for e2m3 using dual-VPSHUFB (LUT) + VPMADDUBSW (unsigned×signed).
    // Every e2m3 value × 16 is an exact integer in [-120, +120].
    // Result = i32_dot / 256.0f (exact, no rounding error).
    //
    // 32-entry LUT split into two 16-entry halves for VPSHUFB (which indexes 0-15):
    //   lut_lower[0..15]: {0,2,4,6,8,10,12,14, 16,18,20,22,24,26,28,30}
    //   lut_upper[0..15]: {32,36,40,44,48,52,56,60, 64,72,80,88,96,104,112,120}
    //
    __m256i const lut_lower_u8x32 = _mm256_set_epi8(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 30, 28,
                                                    26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m256i const lut_upper_u8x32 = _mm256_set_epi8(120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32,
                                                    120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32);
    __m256i const nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const magnitude_mask_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const half_select_u8x32 = _mm256_set1_epi8(0x10);
    __m256i const sign_mask_u8x32 = _mm256_set1_epi8(0x20);
    __m256i const ones_i16x16 = _mm256_set1_epi16(1);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i a_e2m3_u8x32, b_e2m3_u8x32;

nk_dot_e2m3_haswell_cycle:
    if (count_scalars < 32) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b8x32_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x32_serial_(b_scalars, &b_vec, count_scalars);
        a_e2m3_u8x32 = a_vec.ymm;
        b_e2m3_u8x32 = b_vec.ymm;
        count_scalars = 0;
    }
    else {
        a_e2m3_u8x32 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_e2m3_u8x32 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }

    // Extract 5-bit magnitude, then split into low 4 bits (VPSHUFB index) and bit 4 (hi/lo select)
    __m256i a_magnitude_u8x32 = _mm256_and_si256(a_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i b_magnitude_u8x32 = _mm256_and_si256(b_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i a_shuffle_index_u8x32 = _mm256_and_si256(a_magnitude_u8x32, nibble_mask_u8x32);
    __m256i b_shuffle_index_u8x32 = _mm256_and_si256(b_magnitude_u8x32, nibble_mask_u8x32);
    __m256i a_upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(a_magnitude_u8x32, half_select_u8x32),
                                                     half_select_u8x32);
    __m256i b_upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(b_magnitude_u8x32, half_select_u8x32),
                                                     half_select_u8x32);

    // Dual VPSHUFB: lookup in both halves, blend based on bit 4
    __m256i a_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, a_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, a_shuffle_index_u8x32),
                                                  a_upper_select_u8x32);
    __m256i b_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, b_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, b_shuffle_index_u8x32),
                                                  b_upper_select_u8x32);

    // Combined sign: (a ^ b) & 0x20, negate b where signs differ
    __m256i sign_combined_u8x32 = _mm256_and_si256(_mm256_xor_si256(a_e2m3_u8x32, b_e2m3_u8x32), sign_mask_u8x32);
    __m256i negate_mask_u8x32 = _mm256_cmpeq_epi8(sign_combined_u8x32, sign_mask_u8x32);
    __m256i b_negated_u8x32 = _mm256_sub_epi8(_mm256_setzero_si256(), b_unsigned_u8x32);
    __m256i b_signed_i8x32 = _mm256_blendv_epi8(b_unsigned_u8x32, b_negated_u8x32, negate_mask_u8x32);

    // VPMADDUBSW: a_unsigned[unsigned] × b_signed[signed] → i16 pairs (max |120×120| = 14400 < 32767, safe)
    __m256i products_i16x16 = _mm256_maddubs_epi16(a_unsigned_u8x32, b_signed_i8x32);
    // VPMADDWD with ones: i16 pairs → i32
    sum_i32x8 = _mm256_add_epi32(sum_i32x8, _mm256_madd_epi16(products_i16x16, ones_i16x16));

    if (count_scalars) goto nk_dot_e2m3_haswell_cycle;
    *result = (nk_f32_t)nk_reduce_add_i32x8_haswell_(sum_i32x8) / 256.0f;
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

/**
 *  @brief Running state for 128-bit dot accumulation over f16 scalars on Haswell.
 *  @note Alias of nk_dot_through_f32_state_haswell_t_
 */
typedef struct nk_dot_through_f32_state_haswell_t_ nk_dot_f16x8_state_haswell_t;

/**
 *  @brief Running state for 128-bit dot accumulation over bf16 scalars on Haswell.
 *  @note Alias of nk_dot_through_f32_state_haswell_t_
 */
typedef struct nk_dot_through_f32_state_haswell_t_ nk_dot_bf16x8_state_haswell_t;

/**
 *  @brief Running state for 128-bit dot accumulation over e4m3 scalars on Haswell.
 *  @note Alias of nk_dot_through_f32_state_haswell_t_
 */
typedef struct nk_dot_through_f32_state_haswell_t_ nk_dot_e4m3x16_state_haswell_t;

/**
 *  @brief Running state for 128-bit dot accumulation over e5m2 scalars on Haswell.
 *  @note Alias of nk_dot_through_f32_state_haswell_t_
 */
typedef struct nk_dot_through_f32_state_haswell_t_ nk_dot_e5m2x16_state_haswell_t;

/**
 *  @brief Running state for 128-bit dot accumulation over e2m3 scalars on Haswell.
 *  @note Alias of nk_dot_through_f32_state_haswell_t_
 */
typedef struct nk_dot_through_f32_state_haswell_t_ nk_dot_e2m3x16_state_haswell_t;

/**
 *  @brief Running state for 128-bit dot accumulation over e3m2 scalars on Haswell.
 *  @note Alias of nk_dot_through_f32_state_haswell_t_
 */
typedef struct nk_dot_through_f32_state_haswell_t_ nk_dot_e3m2x16_state_haswell_t;

#pragma endregion - Smaller Floats

#pragma region - Small Integers

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

NK_PUBLIC void nk_dot_i4_haswell(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result) {
    // i4 values are packed as nibbles: two 4-bit signed values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    //
    // Algorithm: For signed i4, we use an algebraic transformation (similar to Ice Lake).
    // Let ax, bx be the unsigned [0,15] representation of signed values a, b in [-8,7].
    // Then: a = ax - 8, b = bx - 8 (the XOR trick gives signed = (unsigned ^ 8) - 8)
    // So: a * b = (ax - 8)(bx - 8) = ax * bx - 8 * ax - 8 * bx + 64
    //
    // We compute ax * bx using widening multiply, then apply the correction:
    //   signed_dot = unsigned_dot - 8 * (sum_ax + sum_bx) + 64 * n
    //
    // Optimization: Process 16 bytes (32 nibbles) per iteration and use SAD for correction sums.
    // Benchmark shows 16-byte approach is 2× faster than 8-byte (10.7 GB/s vs 5.3 GB/s).
    // Better ILP and amortized loop overhead with wider operations.
    //
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    __m128i const nibble_mask_u8x16 = _mm_set1_epi8(0x0F);
    __m128i const xor_mask_u8x16 = _mm_set1_epi8(0x08);
    __m128i const zeros_u8x16 = _mm_setzero_si128();
    __m256i sum_cd_i32x8 = _mm256_setzero_si256();
    __m128i sum_cx_i64x2 = _mm_setzero_si128(); // Use i64 for SAD results
    __m128i sum_dx_i64x2 = _mm_setzero_si128();
    __m128i a_i4x32, b_i4x32;

nk_dot_i4_haswell_cycle:
    // Process 16 bytes (32 nibbles) per iteration
    if (n_bytes < 16) {
        // Partial load using serial helper
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n_bytes);
        nk_partial_load_b8x16_serial_(b, &b_vec, n_bytes);
        a_i4x32 = a_vec.xmm;
        b_i4x32 = b_vec.xmm;
        n_bytes = 0;
    }
    else {
        a_i4x32 = _mm_loadu_si128((__m128i const *)a); // Load full 16 bytes
        b_i4x32 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n_bytes -= 16;
    }

    // Extract low and high nibbles
    __m128i a_lo_u8x16 = _mm_and_si128(a_i4x32, nibble_mask_u8x16);
    __m128i a_hi_u8x16 = _mm_and_si128(_mm_srli_epi16(a_i4x32, 4), nibble_mask_u8x16);
    __m128i b_lo_u8x16 = _mm_and_si128(b_i4x32, nibble_mask_u8x16);
    __m128i b_hi_u8x16 = _mm_and_si128(_mm_srli_epi16(b_i4x32, 4), nibble_mask_u8x16);

    // XOR with 8 to get cx, dx values for the algebraic transformation
    __m128i c_lo_u8x16 = _mm_xor_si128(a_lo_u8x16, xor_mask_u8x16);
    __m128i c_hi_u8x16 = _mm_xor_si128(a_hi_u8x16, xor_mask_u8x16);
    __m128i d_lo_u8x16 = _mm_xor_si128(b_lo_u8x16, xor_mask_u8x16);
    __m128i d_hi_u8x16 = _mm_xor_si128(b_hi_u8x16, xor_mask_u8x16);

    // Widen u8 to i16 and multiply using MADD (2× instead of 4×)
    __m256i c_lo_i16x16 = _mm256_cvtepu8_epi16(c_lo_u8x16);
    __m256i c_hi_i16x16 = _mm256_cvtepu8_epi16(c_hi_u8x16);
    __m256i d_lo_i16x16 = _mm256_cvtepu8_epi16(d_lo_u8x16);
    __m256i d_hi_i16x16 = _mm256_cvtepu8_epi16(d_hi_u8x16);

    // Multiply i16×i16 and accumulate to i32 using MADD
    sum_cd_i32x8 = _mm256_add_epi32(sum_cd_i32x8, _mm256_madd_epi16(c_lo_i16x16, d_lo_i16x16));
    sum_cd_i32x8 = _mm256_add_epi32(sum_cd_i32x8, _mm256_madd_epi16(c_hi_i16x16, d_hi_i16x16));

    // Optimization: Use SAD for correction sums (5cy vs 24cy for 8× widenings)
    // PSADBW sums 8× u8 values to a single i64 in each 64-bit lane
    sum_cx_i64x2 = _mm_add_epi64(sum_cx_i64x2, _mm_sad_epu8(c_lo_u8x16, zeros_u8x16));
    sum_cx_i64x2 = _mm_add_epi64(sum_cx_i64x2, _mm_sad_epu8(c_hi_u8x16, zeros_u8x16));
    sum_dx_i64x2 = _mm_add_epi64(sum_dx_i64x2, _mm_sad_epu8(d_lo_u8x16, zeros_u8x16));
    sum_dx_i64x2 = _mm_add_epi64(sum_dx_i64x2, _mm_sad_epu8(d_hi_u8x16, zeros_u8x16));

    if (n_bytes) goto nk_dot_i4_haswell_cycle;

    // Reduce and apply algebraic correction
    nk_i32_t cd_dot = nk_reduce_add_i32x8_haswell_(sum_cd_i32x8);

    // Extract SAD results (already summed across 8 bytes per lane)
    nk_i64_t cx_sum = (nk_i64_t)_mm_extract_epi64(sum_cx_i64x2, 0) + (nk_i64_t)_mm_extract_epi64(sum_cx_i64x2, 1);
    nk_i64_t dx_sum = (nk_i64_t)_mm_extract_epi64(sum_dx_i64x2, 0) + (nk_i64_t)_mm_extract_epi64(sum_dx_i64x2, 1);

    *result = (nk_i32_t)(cd_dot - 8 * (cx_sum + dx_sum) + 64 * (nk_i64_t)n);
}

NK_PUBLIC void nk_dot_u4_haswell(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    // u4 values are packed as nibbles: two 4-bit unsigned values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    // Values are ∈ [0,15], so we can use direct unpacking and multiplication.
    //
    // Optimization: Process 16 bytes (32 nibbles) per iteration for better ILP.
    // Benchmark shows 16-byte approach provides best performance.
    //
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    __m128i const nibble_mask_u8x16 = _mm_set1_epi8(0x0F);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m128i a_u4x32, b_u4x32;

nk_dot_u4_haswell_cycle:
    // Process 16 bytes (32 nibbles) per iteration
    if (n_bytes < 16) {
        // Partial load using serial helper
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n_bytes);
        nk_partial_load_b8x16_serial_(b, &b_vec, n_bytes);
        a_u4x32 = a_vec.xmm;
        b_u4x32 = b_vec.xmm;
        n_bytes = 0;
    }
    else {
        a_u4x32 = _mm_loadu_si128((__m128i const *)a); // Load full 16 bytes
        b_u4x32 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n_bytes -= 16;
    }

    // Extract low and high nibbles
    __m128i a_lo_u8x16 = _mm_and_si128(a_u4x32, nibble_mask_u8x16);
    __m128i a_hi_u8x16 = _mm_and_si128(_mm_srli_epi16(a_u4x32, 4), nibble_mask_u8x16);
    __m128i b_lo_u8x16 = _mm_and_si128(b_u4x32, nibble_mask_u8x16);
    __m128i b_hi_u8x16 = _mm_and_si128(_mm_srli_epi16(b_u4x32, 4), nibble_mask_u8x16);

    // Widen u8 to i16
    __m256i a_lo_i16x16 = _mm256_cvtepu8_epi16(a_lo_u8x16);
    __m256i a_hi_i16x16 = _mm256_cvtepu8_epi16(a_hi_u8x16);
    __m256i b_lo_i16x16 = _mm256_cvtepu8_epi16(b_lo_u8x16);
    __m256i b_hi_i16x16 = _mm256_cvtepu8_epi16(b_hi_u8x16);

    // Multiply i16×i16 and accumulate to i32 using MADD
    sum_i32x8 = _mm256_add_epi32(sum_i32x8, _mm256_madd_epi16(a_lo_i16x16, b_lo_i16x16));
    sum_i32x8 = _mm256_add_epi32(sum_i32x8, _mm256_madd_epi16(a_hi_i16x16, b_hi_i16x16));

    if (n_bytes) goto nk_dot_u4_haswell_cycle;

    *result = (nk_u32_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
}

typedef struct nk_dot_i8x16_state_haswell_t {
    __m256i sum_i32x8;
} nk_dot_i8x16_state_haswell_t;

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

typedef struct nk_dot_u8x16_state_haswell_t {
    __m256i sum_i32x8;
} nk_dot_u8x16_state_haswell_t;

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
 *  @brief State for batched i4 dot products on Haswell.
 *  Processes 32 nibbles (16 bytes) per update iteration for optimal ILP.
 */
typedef struct nk_dot_i4x32_state_haswell_t {
    __m256i product_sum_i32x8; // Main product accumulator: c×d products
    __m128i sum_cx_i64x2;      // Correction term: sum(c) where c = a ^ 8
    __m128i sum_dx_i64x2;      // Correction term: sum(d) where d = b ^ 8
} nk_dot_i4x32_state_haswell_t;

NK_INTERNAL void nk_dot_i4x32_init_haswell(nk_dot_i4x32_state_haswell_t *state) {
    state->product_sum_i32x8 = _mm256_setzero_si256();
    state->sum_cx_i64x2 = _mm_setzero_si128();
    state->sum_dx_i64x2 = _mm_setzero_si128();
}

NK_INTERNAL void nk_dot_i4x32_update_haswell(nk_dot_i4x32_state_haswell_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    // Process 32 nibbles (16 bytes) from the full 128-bit vector
    // Algebraic transformation: a×b = (a_biased - 8)×(b_biased - 8)
    //                                = a_biased×b_biased - 8×(a_biased + b_biased) + 64
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);

    __m128i const nibble_mask_u8x16 = _mm_set1_epi8(0x0F);
    __m128i const xor_mask_u8x16 = _mm_set1_epi8(0x08);
    __m128i const zeros_u8x16 = _mm_setzero_si128();

    __m128i a_i4x32 = a.xmm;
    __m128i b_i4x32 = b.xmm;

    // Extract low and high nibbles
    __m128i a_lo_u8x16 = _mm_and_si128(a_i4x32, nibble_mask_u8x16);
    __m128i a_hi_u8x16 = _mm_and_si128(_mm_srli_epi16(a_i4x32, 4), nibble_mask_u8x16);
    __m128i b_lo_u8x16 = _mm_and_si128(b_i4x32, nibble_mask_u8x16);
    __m128i b_hi_u8x16 = _mm_and_si128(_mm_srli_epi16(b_i4x32, 4), nibble_mask_u8x16);

    // XOR with 8 for algebraic transformation
    __m128i c_lo_u8x16 = _mm_xor_si128(a_lo_u8x16, xor_mask_u8x16);
    __m128i c_hi_u8x16 = _mm_xor_si128(a_hi_u8x16, xor_mask_u8x16);
    __m128i d_lo_u8x16 = _mm_xor_si128(b_lo_u8x16, xor_mask_u8x16);
    __m128i d_hi_u8x16 = _mm_xor_si128(b_hi_u8x16, xor_mask_u8x16);

    // Widen u8 to i16 and multiply using MADD
    __m256i c_lo_i16x16 = _mm256_cvtepu8_epi16(c_lo_u8x16);
    __m256i c_hi_i16x16 = _mm256_cvtepu8_epi16(c_hi_u8x16);
    __m256i d_lo_i16x16 = _mm256_cvtepu8_epi16(d_lo_u8x16);
    __m256i d_hi_i16x16 = _mm256_cvtepu8_epi16(d_hi_u8x16);

    // Multiply and accumulate
    state->product_sum_i32x8 = _mm256_add_epi32(state->product_sum_i32x8, _mm256_madd_epi16(c_lo_i16x16, d_lo_i16x16));
    state->product_sum_i32x8 = _mm256_add_epi32(state->product_sum_i32x8, _mm256_madd_epi16(c_hi_i16x16, d_hi_i16x16));

    // Use SAD for correction sums (5cy vs 24cy)
    state->sum_cx_i64x2 = _mm_add_epi64(state->sum_cx_i64x2, _mm_sad_epu8(c_lo_u8x16, zeros_u8x16));
    state->sum_cx_i64x2 = _mm_add_epi64(state->sum_cx_i64x2, _mm_sad_epu8(c_hi_u8x16, zeros_u8x16));
    state->sum_dx_i64x2 = _mm_add_epi64(state->sum_dx_i64x2, _mm_sad_epu8(d_lo_u8x16, zeros_u8x16));
    state->sum_dx_i64x2 = _mm_add_epi64(state->sum_dx_i64x2, _mm_sad_epu8(d_hi_u8x16, zeros_u8x16));
}

NK_INTERNAL void nk_dot_i4x32_finalize_haswell(                                               //
    nk_dot_i4x32_state_haswell_t const *state_a, nk_dot_i4x32_state_haswell_t const *state_b, //
    nk_dot_i4x32_state_haswell_t const *state_c, nk_dot_i4x32_state_haswell_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {

    // 4-way ILP-optimized reduction with algebraic correction
    // Formula: result = product_sum - 8×(sum_cx + sum_dx) + 64×depth_nibbles
    nk_size_t depth_nibbles = nk_size_round_up_to_multiple_(total_dimensions, 32);

    // Reduce main products from ymm (i32x8) to xmm (i32x4)
    __m128i product_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_a->product_sum_i32x8),
                                            _mm256_extracti128_si256(state_a->product_sum_i32x8, 1));
    __m128i product_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_b->product_sum_i32x8),
                                            _mm256_extracti128_si256(state_b->product_sum_i32x8, 1));
    __m128i product_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_c->product_sum_i32x8),
                                            _mm256_extracti128_si256(state_c->product_sum_i32x8, 1));
    __m128i product_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_d->product_sum_i32x8),
                                            _mm256_extracti128_si256(state_d->product_sum_i32x8, 1));

    // 4-way transpose to get [a,b,c,d] in lanes
    __m128i transpose_ab_low = _mm_unpacklo_epi32(product_a_i32x4, product_b_i32x4);
    __m128i transpose_cd_low = _mm_unpacklo_epi32(product_c_i32x4, product_d_i32x4);
    __m128i transpose_ab_high = _mm_unpackhi_epi32(product_a_i32x4, product_b_i32x4);
    __m128i transpose_cd_high = _mm_unpackhi_epi32(product_c_i32x4, product_d_i32x4);
    __m128i product_lane0 = _mm_unpacklo_epi64(transpose_ab_low, transpose_cd_low);
    __m128i product_lane1 = _mm_unpackhi_epi64(transpose_ab_low, transpose_cd_low);
    __m128i product_lane2 = _mm_unpacklo_epi64(transpose_ab_high, transpose_cd_high);
    __m128i product_lane3 = _mm_unpackhi_epi64(transpose_ab_high, transpose_cd_high);

    // Sum product lanes
    __m128i product_sum_i32x4 = _mm_add_epi32(_mm_add_epi32(product_lane0, product_lane1),
                                              _mm_add_epi32(product_lane2, product_lane3));

    // Reduce correction sums from i64x2 to scalar and back to i32x4
    // Extract both lanes and sum them
    nk_i64_t cx_a = (nk_i64_t)_mm_extract_epi64(state_a->sum_cx_i64x2, 0) +
                    (nk_i64_t)_mm_extract_epi64(state_a->sum_cx_i64x2, 1);
    nk_i64_t cx_b = (nk_i64_t)_mm_extract_epi64(state_b->sum_cx_i64x2, 0) +
                    (nk_i64_t)_mm_extract_epi64(state_b->sum_cx_i64x2, 1);
    nk_i64_t cx_c = (nk_i64_t)_mm_extract_epi64(state_c->sum_cx_i64x2, 0) +
                    (nk_i64_t)_mm_extract_epi64(state_c->sum_cx_i64x2, 1);
    nk_i64_t cx_d = (nk_i64_t)_mm_extract_epi64(state_d->sum_cx_i64x2, 0) +
                    (nk_i64_t)_mm_extract_epi64(state_d->sum_cx_i64x2, 1);

    nk_i64_t dx_a = (nk_i64_t)_mm_extract_epi64(state_a->sum_dx_i64x2, 0) +
                    (nk_i64_t)_mm_extract_epi64(state_a->sum_dx_i64x2, 1);
    nk_i64_t dx_b = (nk_i64_t)_mm_extract_epi64(state_b->sum_dx_i64x2, 0) +
                    (nk_i64_t)_mm_extract_epi64(state_b->sum_dx_i64x2, 1);
    nk_i64_t dx_c = (nk_i64_t)_mm_extract_epi64(state_c->sum_dx_i64x2, 0) +
                    (nk_i64_t)_mm_extract_epi64(state_c->sum_dx_i64x2, 1);
    nk_i64_t dx_d = (nk_i64_t)_mm_extract_epi64(state_d->sum_dx_i64x2, 0) +
                    (nk_i64_t)_mm_extract_epi64(state_d->sum_dx_i64x2, 1);

    // Apply algebraic correction: result = product - 8×(cx + dx) + 64×depth
    nk_i64_t offset_term = 64 * (nk_i64_t)depth_nibbles;
    nk_i32_t result_a = (nk_i32_t)(_mm_extract_epi32(product_sum_i32x4, 0) - 8 * (cx_a + dx_a) + offset_term);
    nk_i32_t result_b = (nk_i32_t)(_mm_extract_epi32(product_sum_i32x4, 1) - 8 * (cx_b + dx_b) + offset_term);
    nk_i32_t result_c = (nk_i32_t)(_mm_extract_epi32(product_sum_i32x4, 2) - 8 * (cx_c + dx_c) + offset_term);
    nk_i32_t result_d = (nk_i32_t)(_mm_extract_epi32(product_sum_i32x4, 3) - 8 * (cx_d + dx_d) + offset_term);

    result->xmm = _mm_set_epi32(result_d, result_c, result_b, result_a);
}

/**
 *  @brief State for batched u4 dot products on Haswell.
 *  Processes 32 nibbles (16 bytes) per update iteration for optimal ILP.
 */
typedef struct nk_dot_u4x32_state_haswell_t {
    __m256i product_sum_i32x8; // Main product accumulator
} nk_dot_u4x32_state_haswell_t;

NK_INTERNAL void nk_dot_u4x32_init_haswell(nk_dot_u4x32_state_haswell_t *state) {
    state->product_sum_i32x8 = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_u4x32_update_haswell(nk_dot_u4x32_state_haswell_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    // Process 32 nibbles (16 bytes) from the full 128-bit vector
    // No algebraic transformation needed for unsigned values
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);

    __m128i const nibble_mask_u8x16 = _mm_set1_epi8(0x0F);

    __m128i a_u4x32 = a.xmm;
    __m128i b_u4x32 = b.xmm;

    // Extract low and high nibbles
    __m128i a_lo_u8x16 = _mm_and_si128(a_u4x32, nibble_mask_u8x16);
    __m128i a_hi_u8x16 = _mm_and_si128(_mm_srli_epi16(a_u4x32, 4), nibble_mask_u8x16);
    __m128i b_lo_u8x16 = _mm_and_si128(b_u4x32, nibble_mask_u8x16);
    __m128i b_hi_u8x16 = _mm_and_si128(_mm_srli_epi16(b_u4x32, 4), nibble_mask_u8x16);

    // Widen u8 to i16
    __m256i a_lo_i16x16 = _mm256_cvtepu8_epi16(a_lo_u8x16);
    __m256i a_hi_i16x16 = _mm256_cvtepu8_epi16(a_hi_u8x16);
    __m256i b_lo_i16x16 = _mm256_cvtepu8_epi16(b_lo_u8x16);
    __m256i b_hi_i16x16 = _mm256_cvtepu8_epi16(b_hi_u8x16);

    // Multiply and accumulate
    state->product_sum_i32x8 = _mm256_add_epi32(state->product_sum_i32x8, _mm256_madd_epi16(a_lo_i16x16, b_lo_i16x16));
    state->product_sum_i32x8 = _mm256_add_epi32(state->product_sum_i32x8, _mm256_madd_epi16(a_hi_i16x16, b_hi_i16x16));
}

NK_INTERNAL void nk_dot_u4x32_finalize_haswell(                                               //
    nk_dot_u4x32_state_haswell_t const *state_a, nk_dot_u4x32_state_haswell_t const *state_b, //
    nk_dot_u4x32_state_haswell_t const *state_c, nk_dot_u4x32_state_haswell_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);

    // 4-way ILP-optimized reduction (no algebraic correction needed for unsigned)
    // Reduce main products from ymm (i32x8) to xmm (i32x4)
    __m128i product_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_a->product_sum_i32x8),
                                            _mm256_extracti128_si256(state_a->product_sum_i32x8, 1));
    __m128i product_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_b->product_sum_i32x8),
                                            _mm256_extracti128_si256(state_b->product_sum_i32x8, 1));
    __m128i product_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_c->product_sum_i32x8),
                                            _mm256_extracti128_si256(state_c->product_sum_i32x8, 1));
    __m128i product_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_d->product_sum_i32x8),
                                            _mm256_extracti128_si256(state_d->product_sum_i32x8, 1));

    // 4-way transpose to get [a,b,c,d] in lanes
    __m128i transpose_ab_low = _mm_unpacklo_epi32(product_a_i32x4, product_b_i32x4);
    __m128i transpose_cd_low = _mm_unpacklo_epi32(product_c_i32x4, product_d_i32x4);
    __m128i transpose_ab_high = _mm_unpackhi_epi32(product_a_i32x4, product_b_i32x4);
    __m128i transpose_cd_high = _mm_unpackhi_epi32(product_c_i32x4, product_d_i32x4);
    __m128i product_lane0 = _mm_unpacklo_epi64(transpose_ab_low, transpose_cd_low);
    __m128i product_lane1 = _mm_unpackhi_epi64(transpose_ab_low, transpose_cd_low);
    __m128i product_lane2 = _mm_unpacklo_epi64(transpose_ab_high, transpose_cd_high);
    __m128i product_lane3 = _mm_unpackhi_epi64(transpose_ab_high, transpose_cd_high);

    // Sum product lanes
    result->xmm = _mm_add_epi32(_mm_add_epi32(product_lane0, product_lane1),
                                _mm_add_epi32(product_lane2, product_lane3));
}

#pragma endregion - Small Integers

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_DOT_HASWELL_H
