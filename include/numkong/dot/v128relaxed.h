/**
 *  @brief SIMD-accelerated Dot Products for WASM.
 *  @file include/numkong/dot/v128relaxed.h
 *  @author Ash Vardanian
 *  @date January 31, 2026
 *
 *  Requires Emscripten 3.1.27+ with `-msimd128 -mrelaxed-simd` flags.
 *
 *  Key optimizations:
 *  - Uses relaxed FMA (f32x4.relaxed_madd, f64x2.relaxed_madd) for 2x throughput
 *  - Smart i8/u8 dot products using algebraic decomposition + correction terms
 *  - F64 upcasting variant for improved numerical precision (NEON-style)
 *
 *  Smart i8 optimization:
 *    Decompose: b = b_7bit - 128 × signbit
 *    Therefore: a·b = a·b_7bit - 128 × sum(a[i] where b[i] < 0)
 *    Uses fast relaxed_dot_i8x16_i7x16 + SAD-like correction
 *
 *  Smart u8 optimization:
 *    Decompose: b = b_7bit + 128 × highbit
 *    Therefore: a·b = a·b_7bit + 128 × sum(a[i] where b[i] >= 128)
 *    Simpler than i8 (positive correction, can use shift instead of mul)
 */

#ifndef NK_DOT_V128RELAXED_H
#define NK_DOT_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/reduce/v128relaxed.h"
#include "numkong/cast/serial.h"
#include "numkong/cast/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

NK_PUBLIC void nk_dot_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f64x2 = wasm_f64x2_splat(0.0);
    nk_f32_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_f32_vec, b_f32_vec;

nk_dot_f32_v128relaxed_cycle:
    if (count_scalars < 2) {
        nk_partial_load_b32x2_serial_(a_scalars, &a_f32_vec, count_scalars);
        nk_partial_load_b32x2_serial_(b_scalars, &b_f32_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b64_serial_(a_scalars, &a_f32_vec);
        nk_load_b64_serial_(b_scalars, &b_f32_vec);
        a_scalars += 2, b_scalars += 2, count_scalars -= 2;
    }
    v128_t a_f32x2 = wasm_v128_load64_zero(&a_f32_vec.u64);
    v128_t b_f32x2 = wasm_v128_load64_zero(&b_f32_vec.u64);
    v128_t a_f64x2 = wasm_f64x2_promote_low_f32x4(a_f32x2);
    v128_t b_f64x2 = wasm_f64x2_promote_low_f32x4(b_f32x2);
    sum_f64x2 = wasm_f64x2_relaxed_madd(a_f64x2, b_f64x2, sum_f64x2);
    if (count_scalars) goto nk_dot_f32_v128relaxed_cycle;

    *result = (nk_f32_t)nk_reduce_add_f64x2_v128relaxed_(sum_f64x2);
}

NK_PUBLIC void nk_dot_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    nk_f16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_f16_vec, b_f16_vec;

nk_dot_f16_v128relaxed_cycle:
    if (count_scalars < 4) {
        nk_partial_load_b16x4_serial_(a_scalars, &a_f16_vec, count_scalars);
        nk_partial_load_b16x4_serial_(b_scalars, &b_f16_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b64_serial_(a_scalars, &a_f16_vec);
        nk_load_b64_serial_(b_scalars, &b_f16_vec);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    nk_b128_vec_t a_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(a_f16_vec);
    nk_b128_vec_t b_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(b_f16_vec);
    sum_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, b_f32_vec.v128, sum_f32x4);
    if (count_scalars) goto nk_dot_f16_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_dot_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    nk_bf16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_bf16_vec, b_bf16_vec;

nk_dot_bf16_v128relaxed_cycle:
    if (count_scalars < 4) {
        nk_partial_load_b16x4_serial_(a_scalars, &a_bf16_vec, count_scalars);
        nk_partial_load_b16x4_serial_(b_scalars, &b_bf16_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b64_serial_(a_scalars, &a_bf16_vec);
        nk_load_b64_serial_(b_scalars, &b_bf16_vec);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    nk_b128_vec_t a_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(a_bf16_vec);
    nk_b128_vec_t b_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(b_bf16_vec);
    sum_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, b_f32_vec.v128, sum_f32x4);
    if (count_scalars) goto nk_dot_bf16_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_dot_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    v128_t sum_f64x2 = wasm_f64x2_splat(0.0);
    v128_t compensation_f64x2 = wasm_f64x2_splat(0.0);
    nk_f64_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b128_vec_t a_vec, b_vec;

nk_dot_f64_v128relaxed_cycle:
    if (count_scalars < 2) {
        nk_partial_load_b64x2_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b64x2_serial_(b_scalars, &b_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b128_serial_(a_scalars, &a_vec);
        nk_load_b128_serial_(b_scalars, &b_vec);
        a_scalars += 2, b_scalars += 2, count_scalars -= 2;
    }
    v128_t product_f64x2 = wasm_f64x2_mul(a_vec.v128, b_vec.v128);
    v128_t product_error_f64x2 = wasm_f64x2_sub(wasm_f64x2_relaxed_madd(a_vec.v128, b_vec.v128, wasm_f64x2_splat(0.0)),
                                                product_f64x2);
    v128_t t_f64x2 = wasm_f64x2_add(sum_f64x2, product_f64x2);
    v128_t z_f64x2 = wasm_f64x2_sub(t_f64x2, sum_f64x2);
    v128_t sum_error_f64x2 = wasm_f64x2_add(wasm_f64x2_sub(sum_f64x2, wasm_f64x2_sub(t_f64x2, z_f64x2)),
                                            wasm_f64x2_sub(product_f64x2, z_f64x2));
    sum_f64x2 = t_f64x2;
    compensation_f64x2 = wasm_f64x2_add(compensation_f64x2, wasm_f64x2_add(sum_error_f64x2, product_error_f64x2));
    if (count_scalars) goto nk_dot_f64_v128relaxed_cycle;

    v128_t final_sum_f64x2 = wasm_f64x2_add(sum_f64x2, compensation_f64x2);
    *result = nk_reduce_add_f64x2_v128relaxed_(final_sum_f64x2);
}

NK_PUBLIC void nk_dot_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result) {
    nk_i64_t sum_total = 0;
    nk_size_t i = 0;

    // Windowed accumulation loop
    while (i + 16 <= n) {
        v128_t sum_i32x4 = wasm_i32x4_splat(0);
        v128_t correction_i16x8 = wasm_i16x8_splat(0);

        // Inner loop: accumulate 127 iterations before widening correction
        nk_size_t cycle = 0;
        for (; cycle < 127 && i + 16 <= n; ++cycle, i += 16) {
            v128_t a_i8x16 = wasm_v128_load(a + i);
            v128_t b_i8x16 = wasm_v128_load(b + i);

            // Extract sign bit: b_neg_mask = (b < 0) ? 0xFF : 0x00
            v128_t b_neg_mask_i8x16 = wasm_i8x16_lt(b_i8x16, wasm_i8x16_splat(0));

            // b_7bit = b & 0x7F (clears sign bit)
            v128_t b_7bit_i8x16 = wasm_v128_and(b_i8x16, wasm_i8x16_splat(0x7F));

            // Fast path: a · b_7bit
            sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_i8x16, b_7bit_i8x16, sum_i32x4);

            // Accumulate correction in i16 (only ONE extadd per iteration!)
            v128_t a_where_b_neg_i8x16 = wasm_v128_and(a_i8x16, b_neg_mask_i8x16);
            v128_t a_neg_i16x8 = wasm_i16x8_extadd_pairwise_i8x16(a_where_b_neg_i8x16);
            correction_i16x8 = wasm_i16x8_add(correction_i16x8, a_neg_i16x8);
        }

        // Widen correction once per window: i16 → i32
        v128_t corr_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(correction_i16x8);

        // Apply correction: sum -= 128 × correction
        v128_t corr_scaled_i32x4 = wasm_i32x4_mul(corr_i32x4, wasm_i32x4_splat(-128));
        sum_i32x4 = wasm_i32x4_add(sum_i32x4, corr_scaled_i32x4);

        // Reduce window to scalar
        sum_total += (nk_i32_t)nk_reduce_add_i32x4_v128relaxed_(sum_i32x4);
    }

    // Handle tail elements
    for (; i < n; i++) { sum_total += (nk_i32_t)a[i] * (nk_i32_t)b[i]; }

    *result = (nk_i32_t)sum_total;
}

NK_PUBLIC void nk_dot_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    v128_t sum_dot_i32x4 = wasm_i32x4_splat(0);
    v128_t sum_a_i32x4 = wasm_i32x4_splat(0);
    nk_u8_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;

nk_dot_u8_v128relaxed_cycle:
    if (count_scalars >= 16) {
        v128_t a_u8x16 = wasm_v128_load(a_scalars);
        v128_t b_u8x16 = wasm_v128_load(b_scalars);
        v128_t b_signed_i8x16 = wasm_v128_xor(b_u8x16, wasm_i8x16_splat(0x80));
        sum_dot_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_u8x16, b_signed_i8x16, sum_dot_i32x4);
        v128_t a_u16x8 = wasm_u16x8_extadd_pairwise_u8x16(a_u8x16);
        v128_t a_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(a_u16x8);
        sum_a_i32x4 = wasm_i32x4_add(sum_a_i32x4, a_u32x4);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
        goto nk_dot_u8_v128relaxed_cycle;
    }

    nk_i32_t dot_biased = nk_reduce_add_i32x4_v128relaxed_(sum_dot_i32x4);
    nk_i32_t sum_a_total = nk_reduce_add_i32x4_v128relaxed_(sum_a_i32x4);
    nk_i32_t total = (nk_i32_t)dot_biased + 128LL * (nk_i32_t)sum_a_total;

    for (nk_size_t idx = 0; idx < count_scalars; idx++) {
        total += (nk_i32_t)a_scalars[idx] * (nk_i32_t)b_scalars[idx];
    }

    *result = (nk_u32_t)total;
}

NK_PUBLIC void nk_dot_e2m3_v128relaxed(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    // Integer dot product for e2m3 using relaxed SIMD: wasm_i32x4_relaxed_dot_i8x16_i7x16_add.
    // Every e2m3 value × 16 is an exact integer in [-120, +120].
    // The relaxed dot takes i8 × u7 (first signed, second unsigned [0,127]). Our magnitudes [0,120] fit.
    // Result = i32_dot / 256.0f (exact, no rounding error).
    //
    // 32-entry LUT split into two 16-entry halves for wasm_i8x16_swizzle (indexes 0-15).
    v128_t lut_lower_u8x16 = wasm_i8x16_const(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    v128_t lut_upper_u8x16 = wasm_i8x16_const(32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120);
    v128_t magnitude_mask_u8x16 = wasm_u8x16_splat(0x1F);
    v128_t nibble_mask_u8x16 = wasm_u8x16_splat(0x0F);
    v128_t half_select_u8x16 = wasm_u8x16_splat(0x10);
    v128_t sign_mask_u8x16 = wasm_u8x16_splat(0x20);
    v128_t sum_i32x4 = wasm_i32x4_splat(0);
    v128_t a_e2m3_u8x16, b_e2m3_u8x16;

nk_dot_e2m3_v128relaxed_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_e2m3_u8x16 = a_vec.v128;
        b_e2m3_u8x16 = b_vec.v128;
        count_scalars = 0;
    }
    else {
        a_e2m3_u8x16 = wasm_v128_load(a_scalars);
        b_e2m3_u8x16 = wasm_v128_load(b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }

    // Extract 5-bit magnitude indices
    v128_t a_magnitude_u8x16 = wasm_v128_and(a_e2m3_u8x16, magnitude_mask_u8x16);
    v128_t b_magnitude_u8x16 = wasm_v128_and(b_e2m3_u8x16, magnitude_mask_u8x16);

    // Dual swizzle + bitselect for 32-entry LUT (a)
    v128_t a_shuffle_index_u8x16 = wasm_v128_and(a_magnitude_u8x16, nibble_mask_u8x16);
    v128_t a_lower_u8x16 = wasm_i8x16_swizzle(lut_lower_u8x16, a_shuffle_index_u8x16);
    v128_t a_upper_u8x16 = wasm_i8x16_swizzle(lut_upper_u8x16, a_shuffle_index_u8x16);
    v128_t a_upper_select_u8x16 = wasm_i8x16_eq(wasm_v128_and(a_magnitude_u8x16, half_select_u8x16), half_select_u8x16);
    v128_t a_unsigned_u8x16 = wasm_v128_bitselect(a_upper_u8x16, a_lower_u8x16, a_upper_select_u8x16);

    // Dual swizzle + bitselect for 32-entry LUT (b)
    v128_t b_shuffle_index_u8x16 = wasm_v128_and(b_magnitude_u8x16, nibble_mask_u8x16);
    v128_t b_lower_u8x16 = wasm_i8x16_swizzle(lut_lower_u8x16, b_shuffle_index_u8x16);
    v128_t b_upper_u8x16 = wasm_i8x16_swizzle(lut_upper_u8x16, b_shuffle_index_u8x16);
    v128_t b_upper_select_u8x16 = wasm_i8x16_eq(wasm_v128_and(b_magnitude_u8x16, half_select_u8x16), half_select_u8x16);
    v128_t b_unsigned_u8x16 = wasm_v128_bitselect(b_upper_u8x16, b_lower_u8x16, b_upper_select_u8x16);

    // Combined sign: (a ^ b) & 0x20 — nonzero means negative product
    // Apply sign to a (relaxed_dot wants i8 × u7: a_signed, b_unsigned)
    v128_t sign_combined_u8x16 = wasm_v128_and(wasm_v128_xor(a_e2m3_u8x16, b_e2m3_u8x16), sign_mask_u8x16);
    v128_t negate_mask_u8x16 = wasm_i8x16_eq(sign_combined_u8x16, sign_mask_u8x16);
    v128_t a_negated_u8x16 = wasm_i8x16_neg(a_unsigned_u8x16);
    v128_t a_signed_i8x16 = wasm_v128_bitselect(a_negated_u8x16, a_unsigned_u8x16, negate_mask_u8x16);

    // relaxed_dot: a_signed[i8] × b_unsigned[u7] → i32 accumulate
    sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_signed_i8x16, b_unsigned_u8x16, sum_i32x4);

    if (count_scalars) goto nk_dot_e2m3_v128relaxed_cycle;
    *result = (nk_f32_t)nk_reduce_add_i32x4_v128relaxed_(sum_i32x4) / 256.0f;
}

NK_PUBLIC void nk_dot_e3m2_v128relaxed(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    // Integer dot product for e3m2 using i16 arithmetic with widening multiply.
    // Every e3m2 value × 16 is an exact integer, but magnitudes reach 448, requiring i16.
    // Result = i32_dot / 256.0f (exact, no rounding error).
    //
    // 32-entry magnitude LUT split into low-byte halves for dual swizzle lookup.
    // High byte is 0 for indices 0-27 and 1 for indices 28-31, so a simple comparison
    // replaces the high-byte LUT entirely.
    //
    // Low-byte LUT entries (magnitude[i] & 0xFF):
    //   [0,1,2,3,4,5,6,7,8,10,12,14,16,20,24,28] lower half
    //   [32,40,48,56,64,80,96,112,128,160,192,224,0,64,128,192] upper half
    v128_t lut_lo_lower_u8x16 = wasm_i8x16_const(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28);
    v128_t lut_lo_upper_u8x16 = wasm_u8x16_const(32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 0, 64, 128, 192);
    v128_t magnitude_mask_u8x16 = wasm_u8x16_splat(0x1F);
    v128_t nibble_mask_u8x16 = wasm_u8x16_splat(0x0F);
    v128_t half_select_u8x16 = wasm_u8x16_splat(0x10);
    v128_t hi_threshold_u8x16 = wasm_u8x16_splat(28);
    v128_t sign_mask_u8x16 = wasm_u8x16_splat(0x20);
    v128_t sum_i32x4 = wasm_i32x4_splat(0);
    v128_t a_e3m2_u8x16, b_e3m2_u8x16;

nk_dot_e3m2_v128relaxed_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_e3m2_u8x16 = a_vec.v128;
        b_e3m2_u8x16 = b_vec.v128;
        count_scalars = 0;
    }
    else {
        a_e3m2_u8x16 = wasm_v128_load(a_scalars);
        b_e3m2_u8x16 = wasm_v128_load(b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }

    // Extract 5-bit magnitude indices
    v128_t a_magnitude_u8x16 = wasm_v128_and(a_e3m2_u8x16, magnitude_mask_u8x16);
    v128_t b_magnitude_u8x16 = wasm_v128_and(b_e3m2_u8x16, magnitude_mask_u8x16);

    // Dual swizzle + bitselect for 32-entry low-byte LUT (a)
    v128_t a_shuffle_index_u8x16 = wasm_v128_and(a_magnitude_u8x16, nibble_mask_u8x16);
    v128_t a_lower_u8x16 = wasm_i8x16_swizzle(lut_lo_lower_u8x16, a_shuffle_index_u8x16);
    v128_t a_upper_u8x16 = wasm_i8x16_swizzle(lut_lo_upper_u8x16, a_shuffle_index_u8x16);
    v128_t a_upper_select_u8x16 = wasm_i8x16_eq(wasm_v128_and(a_magnitude_u8x16, half_select_u8x16), half_select_u8x16);
    v128_t a_lo_bytes_u8x16 = wasm_v128_bitselect(a_upper_u8x16, a_lower_u8x16, a_upper_select_u8x16);

    // High byte is 1 iff magnitude index >= 28 (values 256, 320, 384, 448), else 0
    v128_t a_hi_bytes_u8x16 = wasm_v128_and(wasm_u8x16_ge(a_magnitude_u8x16, hi_threshold_u8x16), wasm_u8x16_splat(1));

    // Dual swizzle + bitselect for 32-entry low-byte LUT (b)
    v128_t b_shuffle_index_u8x16 = wasm_v128_and(b_magnitude_u8x16, nibble_mask_u8x16);
    v128_t b_lower_u8x16 = wasm_i8x16_swizzle(lut_lo_lower_u8x16, b_shuffle_index_u8x16);
    v128_t b_upper_u8x16 = wasm_i8x16_swizzle(lut_lo_upper_u8x16, b_shuffle_index_u8x16);
    v128_t b_upper_select_u8x16 = wasm_i8x16_eq(wasm_v128_and(b_magnitude_u8x16, half_select_u8x16), half_select_u8x16);
    v128_t b_lo_bytes_u8x16 = wasm_v128_bitselect(b_upper_u8x16, b_lower_u8x16, b_upper_select_u8x16);

    // High byte is 1 iff magnitude index >= 28
    v128_t b_hi_bytes_u8x16 = wasm_v128_and(wasm_u8x16_ge(b_magnitude_u8x16, hi_threshold_u8x16), wasm_u8x16_splat(1));

    // Combine low and high bytes into i16 via byte interleave shuffle (little-endian: low byte first)
    v128_t a_unsigned_low_i16x8 = wasm_i8x16_shuffle(a_lo_bytes_u8x16, a_hi_bytes_u8x16, 0, 16, 1, 17, 2, 18, 3, 19, 4,
                                                     20, 5, 21, 6, 22, 7, 23);
    v128_t a_unsigned_high_i16x8 = wasm_i8x16_shuffle(a_lo_bytes_u8x16, a_hi_bytes_u8x16, 8, 24, 9, 25, 10, 26, 11, 27,
                                                      12, 28, 13, 29, 14, 30, 15, 31);
    v128_t b_unsigned_low_i16x8 = wasm_i8x16_shuffle(b_lo_bytes_u8x16, b_hi_bytes_u8x16, 0, 16, 1, 17, 2, 18, 3, 19, 4,
                                                     20, 5, 21, 6, 22, 7, 23);
    v128_t b_unsigned_high_i16x8 = wasm_i8x16_shuffle(b_lo_bytes_u8x16, b_hi_bytes_u8x16, 8, 24, 9, 25, 10, 26, 11, 27,
                                                      12, 28, 13, 29, 14, 30, 15, 31);

    // Combined sign: XOR sign bits, negate only b (saves ~15 ops vs independent negation)
    v128_t sign_combined_u8x16 = wasm_v128_and(wasm_v128_xor(a_e3m2_u8x16, b_e3m2_u8x16), sign_mask_u8x16);
    v128_t negate_mask_u8x16 = wasm_i8x16_eq(sign_combined_u8x16, sign_mask_u8x16);
    v128_t negate_low_i16x8 = wasm_i8x16_shuffle(negate_mask_u8x16, negate_mask_u8x16, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5,
                                                 5, 6, 6, 7, 7);
    v128_t negate_high_i16x8 = wasm_i8x16_shuffle(negate_mask_u8x16, negate_mask_u8x16, 8, 8, 9, 9, 10, 10, 11, 11, 12,
                                                  12, 13, 13, 14, 14, 15, 15);
    b_unsigned_low_i16x8 = wasm_v128_bitselect(wasm_i16x8_neg(b_unsigned_low_i16x8), b_unsigned_low_i16x8,
                                               negate_low_i16x8);
    b_unsigned_high_i16x8 = wasm_v128_bitselect(wasm_i16x8_neg(b_unsigned_high_i16x8), b_unsigned_high_i16x8,
                                                negate_high_i16x8);

    // Widening multiply: i16×i16 → i32, accumulate (a is unsigned magnitude, b has combined sign)
    sum_i32x4 = wasm_i32x4_add(sum_i32x4, wasm_i32x4_extmul_low_i16x8(a_unsigned_low_i16x8, b_unsigned_low_i16x8));
    sum_i32x4 = wasm_i32x4_add(sum_i32x4, wasm_i32x4_extmul_high_i16x8(a_unsigned_low_i16x8, b_unsigned_low_i16x8));
    sum_i32x4 = wasm_i32x4_add(sum_i32x4, wasm_i32x4_extmul_low_i16x8(a_unsigned_high_i16x8, b_unsigned_high_i16x8));
    sum_i32x4 = wasm_i32x4_add(sum_i32x4, wasm_i32x4_extmul_high_i16x8(a_unsigned_high_i16x8, b_unsigned_high_i16x8));

    if (count_scalars) goto nk_dot_e3m2_v128relaxed_cycle;
    *result = (nk_f32_t)nk_reduce_add_i32x4_v128relaxed_(sum_i32x4) / 256.0f;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_DOT_V128RELAXED_H
