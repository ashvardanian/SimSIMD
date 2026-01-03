/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Ice Lake CPUs.
 *  @file include/numkong/elementwise/ice.h
 *  @sa include/numkong/elementwise.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_ELEMENTWISE_ICE_H
#define NK_ELEMENTWISE_ICE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), \
                             apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_sum_i8_ice(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result) {
    __mmask64 mask = 0xFFFFFFFFFFFFFFFF;
    __m512i a_i8_vec, b_i8_vec;
    __m512i sum_i8_vec;
nk_sum_i8_ice_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_i8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i8_vec = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_i8_vec = _mm512_loadu_epi8(a);
        b_i8_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }
    sum_i8_vec = _mm512_adds_epi8(a_i8_vec, b_i8_vec);
    _mm512_mask_storeu_epi8(result, mask, sum_i8_vec);
    result += 64;
    if (n) goto nk_sum_i8_ice_cycle;
}

NK_PUBLIC void nk_sum_u8_ice(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result) {
    __mmask64 mask = 0xFFFFFFFFFFFFFFFF;
    __m512i a_u8_vec, b_u8_vec;
    __m512i sum_u8_vec;
nk_sum_u8_ice_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_u8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_u8_vec = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_u8_vec = _mm512_loadu_epi8(a);
        b_u8_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }
    sum_u8_vec = _mm512_adds_epu8(a_u8_vec, b_u8_vec);
    _mm512_mask_storeu_epi8(result, mask, sum_u8_vec);
    result += 64;
    if (n) goto nk_sum_u8_ice_cycle;
}

NK_PUBLIC void nk_sum_i16_ice(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result) {
    __mmask32 mask = 0xFFFFFFFF;
    __m512i a_i16_vec, b_i16_vec;
    __m512i sum_i16_vec;
nk_sum_i16_ice_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_i16_vec = _mm512_loadu_epi16(a);
        b_i16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    sum_i16_vec = _mm512_adds_epi16(a_i16_vec, b_i16_vec);
    _mm512_mask_storeu_epi16(result, mask, sum_i16_vec);
    result += 32;
    if (n) goto nk_sum_i16_ice_cycle;
}

NK_PUBLIC void nk_sum_u16_ice(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result) {
    __mmask32 mask = 0xFFFFFFFF;
    __m512i a_u16_vec, b_u16_vec;
    __m512i sum_u16_vec;
nk_sum_u16_ice_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_u16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_u16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_u16_vec = _mm512_loadu_epi16(a);
        b_u16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    sum_u16_vec = _mm512_adds_epu16(a_u16_vec, b_u16_vec);
    _mm512_mask_storeu_epi16(result, mask, sum_u16_vec);
    result += 32;
    if (n) goto nk_sum_u16_ice_cycle;
}

NK_INTERNAL __m512i _mm512_adds_epi32_ice(__m512i a, __m512i b) {
    // ! There are many flavors of addition with saturation in AVX-512: i8, u8, i16, and u16.
    // ! But not for larger numeric types. We have to do it manually.
    // ! https://stackoverflow.com/a/56531252/2766161
    __m512i sum = _mm512_add_epi32(a, b);

    // Set constants for overflow and underflow limits
    __m512i max_val = _mm512_set1_epi32(2147483647);
    __m512i min_val = _mm512_set1_epi32(-2147483648);

    // TODO: Consider using ternary operator for performance.
    // Detect positive overflow: (a > 0) && (b > 0) && (sum < 0)
    __mmask16 a_is_positive = _mm512_cmpgt_epi32_mask(a, _mm512_setzero_si512());
    __mmask16 b_is_positive = _mm512_cmpgt_epi32_mask(b, _mm512_setzero_si512());
    __mmask16 sum_is_negative = _mm512_cmplt_epi32_mask(sum, _mm512_setzero_si512());
    __mmask16 pos_overflow_mask = _kand_mask16(_kand_mask16(a_is_positive, b_is_positive), sum_is_negative);

    // TODO: Consider using ternary operator for performance.
    // Detect negative overflow: (a < 0) && (b < 0) && (sum >= 0)
    __mmask16 a_is_negative = _mm512_cmplt_epi32_mask(a, _mm512_setzero_si512());
    __mmask16 b_is_negative = _mm512_cmplt_epi32_mask(b, _mm512_setzero_si512());
    __mmask16 sum_is_non_negative = _mm512_cmpge_epi32_mask(sum, _mm512_setzero_si512());
    __mmask16 neg_overflow_mask = _kand_mask16(_kand_mask16(a_is_negative, b_is_negative), sum_is_non_negative);

    // Apply saturation for positive overflow
    sum = _mm512_mask_blend_epi32(pos_overflow_mask, sum, max_val);
    // Apply saturation for negative overflow
    sum = _mm512_mask_blend_epi32(neg_overflow_mask, sum, min_val);
    return sum;
}

NK_INTERNAL __m512i _mm512_adds_epu32_ice(__m512i a, __m512i b) {
    __m512i sum = _mm512_add_epi32(a, b);
    __mmask16 overflow_mask = _mm512_cmp_epu32_mask(sum, a, _MM_CMPINT_LT); // sum < a means overflow
    __m512i max_val = _mm512_set1_epi32(4294967295u);
    return _mm512_mask_blend_epi32(overflow_mask, sum, max_val);
}

NK_INTERNAL __m512i _mm512_adds_epi64_ice(__m512i a, __m512i b) {
    __m512i sum = _mm512_add_epi64(a, b);
    __m512i sign_mask = _mm512_set1_epi64(0x8000000000000000);

    __m512i overflow = _mm512_and_si512(_mm512_xor_si512(a, b), sign_mask);  // Same sign inputs
    __m512i overflows = _mm512_or_si512(overflow, _mm512_xor_si512(sum, a)); // Overflow condition

    __m512i max_val = _mm512_set1_epi64(9223372036854775807ll);
    __m512i min_val = _mm512_set1_epi64(-9223372036854775807ll - 1);
    __m512i overflow_result = _mm512_mask_blend_epi64(_mm512_cmp_epi64_mask(sum, min_val, _MM_CMPINT_LT), max_val,
                                                      min_val);

    return _mm512_mask_blend_epi64(_mm512_test_epi64_mask(overflows, overflows), sum, overflow_result);
}

NK_INTERNAL __m512i _mm512_adds_epu64_ice(__m512i a, __m512i b) {
    __m512i sum = _mm512_add_epi64(a, b);
    __mmask8 overflow_mask = _mm512_cmp_epu64_mask(sum, a, _MM_CMPINT_LT); // sum < a means overflow
    __m512i max_val = _mm512_set1_epi64(18446744073709551615ull);
    return _mm512_mask_blend_epi64(overflow_mask, sum, max_val);
}

NK_PUBLIC void nk_sum_i32_ice(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result) {
    __mmask16 mask = 0xFFFF;
    __m512i a_i32_vec, b_i32_vec;
    __m512i sum_i32_vec;
nk_sum_i32_ice_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i32_vec = _mm512_maskz_loadu_epi32(mask, a);
        b_i32_vec = _mm512_maskz_loadu_epi32(mask, b);
        n = 0;
    }
    else {
        a_i32_vec = _mm512_loadu_epi32(a);
        b_i32_vec = _mm512_loadu_epi32(b);
        a += 16, b += 16, n -= 16;
    }
    sum_i32_vec = _mm512_adds_epi32_ice(a_i32_vec, b_i32_vec);
    _mm512_mask_storeu_epi32(result, mask, sum_i32_vec);
    result += 16;
    if (n) goto nk_sum_i32_ice_cycle;
}

NK_PUBLIC void nk_sum_u32_ice(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result) {
    __mmask16 mask = 0xFFFF;
    __m512i a_u32_vec, b_u32_vec;
    __m512i sum_u32_vec;
nk_sum_u32_ice_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u32_vec = _mm512_maskz_loadu_epi32(mask, a);
        b_u32_vec = _mm512_maskz_loadu_epi32(mask, b);
        n = 0;
    }
    else {
        a_u32_vec = _mm512_loadu_epi32(a);
        b_u32_vec = _mm512_loadu_epi32(b);
        a += 16, b += 16, n -= 16;
    }
    sum_u32_vec = _mm512_adds_epu32_ice(a_u32_vec, b_u32_vec);
    _mm512_mask_storeu_epi32(result, mask, sum_u32_vec);
    result += 16;
    if (n) goto nk_sum_u32_ice_cycle;
}

NK_PUBLIC void nk_sum_i64_ice(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result) {
    __mmask8 mask = 0xFF;
    __m512i a_i64_vec, b_i64_vec;
    __m512i sum_i64_vec;
nk_sum_i64_ice_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i64_vec = _mm512_maskz_loadu_epi64(mask, a);
        b_i64_vec = _mm512_maskz_loadu_epi64(mask, b);
        n = 0;
    }
    else {
        a_i64_vec = _mm512_loadu_epi64(a);
        b_i64_vec = _mm512_loadu_epi64(b);
        a += 8, b += 8, n -= 8;
    }
    sum_i64_vec = _mm512_adds_epi64_ice(a_i64_vec, b_i64_vec);
    _mm512_mask_storeu_epi64(result, mask, sum_i64_vec);
    result += 8;
    if (n) goto nk_sum_i64_ice_cycle;
}

NK_PUBLIC void nk_sum_u64_ice(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result) {
    __mmask8 mask = 0xFF;
    __m512i a_u64_vec, b_u64_vec;
    __m512i sum_u64_vec;
nk_sum_u64_ice_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u64_vec = _mm512_maskz_loadu_epi64(mask, a);
        b_u64_vec = _mm512_maskz_loadu_epi64(mask, b);
        n = 0;
    }
    else {
        a_u64_vec = _mm512_loadu_epi64(a);
        b_u64_vec = _mm512_loadu_epi64(b);
        a += 8, b += 8, n -= 8;
    }
    sum_u64_vec = _mm512_adds_epu64_ice(a_u64_vec, b_u64_vec);
    _mm512_mask_storeu_epi64(result, mask, sum_u64_vec);
    result += 8;
    if (n) goto nk_sum_u64_ice_cycle;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_ICE
#endif // NK_TARGET_X86_

#endif // NK_ELEMENTWISE_ICE_H
