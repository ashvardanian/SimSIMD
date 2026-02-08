/**
 *  @brief SIMD-accelerated Dot Products for RISC-V.
 *  @file include/numkong/dot/rvv.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  SpacemiT K1 and similar chips implement base RVV 1.0 without half-precision extensions.
 *  RVV uses vector length agnostic programming where:
 *  - `vsetvl_e*m*(n)` sets VL = min(n, VLMAX) and returns actual VL
 *  - Loads/stores with VL automatically handle partial vectors (tail elements)
 *  - No explicit masking needed for simple reductions
 *
 *  This file contains base RVV 1.0 operations (i8, u8, f32, f64).
 *  For f16 (Zvfh) see rvvhalf.h, for bf16 (Zvfbfwma) see rvvbf16.h.
 *
 *  Widening operations:
 *  - i8 ⨯ i8 → i16 via vwmul, then i16 reduction → i32 via vwredsum
 *  - f32 ⨯ f32 → f64 via vfwmul (for precision, like Skylake)
 */
#ifndef NK_DOT_RVV_H
#define NK_DOT_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/cast/rvv.h" // `nk_e4m3m1_to_f32m4_rvv_`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_i8_rvv(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                             nk_i32_t *result) {
    vint32m1_t sum_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vint8m1_t a_i8m1 = __riscv_vle8_v_i8m1(a_scalars, vector_length);
        vint8m1_t b_i8m1 = __riscv_vle8_v_i8m1(b_scalars, vector_length);
        // Widening multiply: i8 ⨯ i8 → i16
        vint16m2_t ab_i16m2 = __riscv_vwmul_vv_i16m2(a_i8m1, b_i8m1, vector_length);
        // Widening reduction sum: i16 vector → i32 scalar
        sum_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(ab_i16m2, sum_i32m1, vector_length);
    }
    *result = __riscv_vmv_x_s_i32m1_i32(sum_i32m1);
}

NK_PUBLIC void nk_dot_u8_rvv(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                             nk_u32_t *result) {
    vuint32m1_t sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1(a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1(b_scalars, vector_length);
        // Widening multiply: u8 ⨯ u8 → u16
        vuint16m2_t ab_u16m2 = __riscv_vwmulu_vv_u16m2(a_u8m1, b_u8m1, vector_length);
        // Widening reduction sum: u16 vector → u32 scalar
        sum_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(ab_u16m2, sum_u32m1, vector_length);
    }
    *result = __riscv_vmv_x_s_u32m1_u32(sum_u32m1);
}

NK_PUBLIC void nk_dot_f32_rvv(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                              nk_f32_t *result) {
    vfloat64m1_t sum_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count_scalars);
        vfloat32m1_t a_f32m1 = __riscv_vle32_v_f32m1(a_scalars, vector_length);
        vfloat32m1_t b_f32m1 = __riscv_vle32_v_f32m1(b_scalars, vector_length);
        // Widening multiply: f32 ⨯ f32 → f64
        vfloat64m2_t ab_f64m2 = __riscv_vfwmul_vv_f64m2(a_f32m1, b_f32m1, vector_length);
        // Ordered reduction sum
        sum_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(ab_f64m2, sum_f64m1, vector_length);
    }
    // Downcast f64 result to f32
    *result = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(sum_f64m1);
}

NK_PUBLIC void nk_dot_f64_rvv(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                              nk_f64_t *result) {
    // Accumulate partial sums into vector lanes, then one final horizontal reduction
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count_scalars);
        vfloat64m1_t a_f64m1 = __riscv_vle64_v_f64m1(a_scalars, vector_length);
        vfloat64m1_t b_f64m1 = __riscv_vle64_v_f64m1(b_scalars, vector_length);
        // Accumulate a ⨯ b into vector lanes
        sum_f64m1 = __riscv_vfmacc_vv_f64m1(sum_f64m1, a_f64m1, b_f64m1, vector_length);
    }
    // Single horizontal reduction at the end with VLMAX
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    *result = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_f64m1, zero_f64m1, vlmax));
}

NK_PUBLIC void nk_dot_f16_rvv(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                              nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);

        // Load f16 as u16 bits and convert to f32 via helper
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b_scalars, vector_length);
        vfloat32m2_t a_f32m2 = nk_f16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_f16m1_to_f32m2_rvv_(b_u16m1, vector_length);

        // Multiply and reduce in f32 (matching Skylake precision)
        vfloat32m2_t ab_f32m2 = __riscv_vfmul_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(ab_f32m2, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_dot_bf16_rvv(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);

        // Load bf16 as u16 and convert to f32 via helper
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b_scalars, vector_length);
        vfloat32m2_t a_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_bf16m1_to_f32m2_rvv_(b_u16m1, vector_length);

        // Multiply and reduce in f32 (matching Skylake precision)
        vfloat32m2_t ab_f32m2 = __riscv_vfmul_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(ab_f32m2, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_dot_e4m3_rvv(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e4m3 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e4m3m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e4m3m1_to_f32m4_rvv_(b_u8m1, vector_length);

        // Multiply and reduce in f32 (matching Skylake precision)
        vfloat32m4_t ab_f32m4 = __riscv_vfmul_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(ab_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_dot_e5m2_rvv(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e5m2 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e5m2m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e5m2m1_to_f32m4_rvv_(b_u8m1, vector_length);

        // Multiply and reduce in f32 (matching Skylake precision)
        vfloat32m4_t ab_f32m4 = __riscv_vfmul_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(ab_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_dot_e2m3_rvv(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e2m3 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e2m3m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e2m3m1_to_f32m4_rvv_(b_u8m1, vector_length);

        // Multiply and reduce in f32
        vfloat32m4_t ab_f32m4 = __riscv_vfmul_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(ab_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_dot_e3m2_rvv(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e3m2 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e3m2m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e3m2m1_to_f32m4_rvv_(b_u8m1, vector_length);

        // Multiply and reduce in f32
        vfloat32m4_t ab_f32m4 = __riscv_vfmul_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(ab_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_dot_i4_rvv(nk_i4x2_t const *a_scalars, nk_i4x2_t const *b_scalars, nk_size_t count_dimensions,
                             nk_i32_t *result) {
    // count_dimensions = number of 4-bit values, not bytes
    nk_size_t n_full_bytes = count_dimensions / 2;
    nk_i32_t tail_contribution = 0;

    // Handle odd tail: only low nibble of last byte contributes
    if (count_dimensions & 1) {
        nk_size_t last_byte = n_full_bytes;
        nk_i32_t a_low = (nk_i32_t)((a_scalars[last_byte] & 0x0F) ^ 8) - 8;
        nk_i32_t b_low = (nk_i32_t)((b_scalars[last_byte] & 0x0F) ^ 8) - 8;
        tail_contribution = a_low * b_low;
    }

    vint32m1_t sum_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    for (nk_size_t vector_length; n_full_bytes > 0;
         n_full_bytes -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n_full_bytes);

        vuint8m1_t a_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);

        vuint8m1_t a_high_u8m1 = __riscv_vsrl_vx_u8m1(a_packed_u8m1, 4, vector_length);
        vuint8m1_t b_high_u8m1 = __riscv_vsrl_vx_u8m1(b_packed_u8m1, 4, vector_length);
        vuint8m1_t a_low_u8m1 = __riscv_vand_vx_u8m1(a_packed_u8m1, 0x0F, vector_length);
        vuint8m1_t b_low_u8m1 = __riscv_vand_vx_u8m1(b_packed_u8m1, 0x0F, vector_length);

        // Sign extend 4-bit to 8-bit: (x ^ 8) - 8
        vint8m1_t a_high_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(a_high_u8m1), 8, vector_length), 8, vector_length);
        vint8m1_t b_high_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(b_high_u8m1), 8, vector_length), 8, vector_length);
        vint8m1_t a_low_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(a_low_u8m1), 8, vector_length), 8, vector_length);
        vint8m1_t b_low_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(b_low_u8m1), 8, vector_length), 8, vector_length);

        vint16m2_t ab_high_i16m2 = __riscv_vwmul_vv_i16m2(a_high_i8m1, b_high_i8m1, vector_length);
        vint16m2_t ab_low_i16m2 = __riscv_vwmul_vv_i16m2(a_low_i8m1, b_low_i8m1, vector_length);

        sum_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(ab_high_i16m2, sum_i32m1, vector_length);
        sum_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(ab_low_i16m2, sum_i32m1, vector_length);
    }
    *result = __riscv_vmv_x_s_i32m1_i32(sum_i32m1) + tail_contribution;
}

NK_PUBLIC void nk_dot_u4_rvv(nk_u4x2_t const *a_scalars, nk_u4x2_t const *b_scalars, nk_size_t count_dimensions,
                             nk_u32_t *result) {
    // count_dimensions = number of 4-bit values, not bytes
    nk_size_t n_full_bytes = count_dimensions / 2;
    nk_u32_t tail_contribution = 0;

    // Handle odd tail: only low nibble of last byte contributes
    if (count_dimensions & 1) {
        nk_size_t last_byte = n_full_bytes;
        nk_u32_t a_low = (nk_u32_t)(a_scalars[last_byte] & 0x0F);
        nk_u32_t b_low = (nk_u32_t)(b_scalars[last_byte] & 0x0F);
        tail_contribution = a_low * b_low;
    }

    vuint32m1_t sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    for (nk_size_t vector_length; n_full_bytes > 0;
         n_full_bytes -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n_full_bytes);

        vuint8m1_t a_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);

        vuint8m1_t a_high_u8m1 = __riscv_vsrl_vx_u8m1(a_packed_u8m1, 4, vector_length);
        vuint8m1_t b_high_u8m1 = __riscv_vsrl_vx_u8m1(b_packed_u8m1, 4, vector_length);
        vuint8m1_t a_low_u8m1 = __riscv_vand_vx_u8m1(a_packed_u8m1, 0x0F, vector_length);
        vuint8m1_t b_low_u8m1 = __riscv_vand_vx_u8m1(b_packed_u8m1, 0x0F, vector_length);

        vuint16m2_t ab_high_u16m2 = __riscv_vwmulu_vv_u16m2(a_high_u8m1, b_high_u8m1, vector_length);
        vuint16m2_t ab_low_u16m2 = __riscv_vwmulu_vv_u16m2(a_low_u8m1, b_low_u8m1, vector_length);

        sum_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(ab_high_u16m2, sum_u32m1, vector_length);
        sum_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(ab_low_u16m2, sum_u32m1, vector_length);
    }
    *result = __riscv_vmv_x_s_u32m1_u32(sum_u32m1) + tail_contribution;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_DOT_RVV_H
