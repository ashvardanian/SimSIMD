/**
 *  @brief SIMD-accelerated Dot Products for RISC-V BF16.
 *  @file include/numkong/dot/rvvbf16.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  Alibaba XuanTie C930 and similar chips implement RVV 1.0 with Zvfbfwma extension.
 *  Zvfbfwma provides widening bf16 fused multiply-accumulate to f32:
 *    vfwmaccbf16: f32 ← bf16 ⨯ bf16
 *
 *  For 6-bit types (e2m3, e3m2), conversion uses a 32-entry VLUXEI8 LUT gather.
 *  For 8-bit types (e4m3, e5m2), conversion uses hybrid arithmetic + small subnormal LUT.
 *  All variants then use vfwmaccbf16_vv for fused bf16 ⨯ bf16 → f32 multiply-accumulate.
 *
 *  Requires: RVV 1.0 + Zvfbfwma extension (GCC 14+ or Clang 18+)
 */
#ifndef NK_DOT_RVVBF16_H
#define NK_DOT_RVVBF16_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVVBF16

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_bf16_rvvbf16(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e16m1(count_scalars);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)a_scalars, vl);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)b_scalars, vl);
        vbfloat16m1_t a_bf16m1 = __riscv_vreinterpret_v_u16m1_bf16m1(a_u16m1);
        vbfloat16m1_t b_bf16m1 = __riscv_vreinterpret_v_u16m1_bf16m1(b_u16m1);
        // Widening bf16 FMA: f32 ← bf16 ⨯ bf16
        vfloat32m2_t acc_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        acc_f32m2 = __riscv_vfwmaccbf16_vv_f32m2(acc_f32m2, a_bf16m1, b_bf16m1, vl);
        // Reduction sum
        sum_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(acc_f32m2, sum_f32m1, vl);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

/**
 *  @brief Convert e2m3 (1-2-3 sign-exp-mantissa, 6-bit) to bf16 via 32-entry VLUXEI8 LUT.
 *  Strip sign bit (bit 5), look up 5-bit magnitude in 32-entry table, OR sign into bit 15.
 */
NK_INTERNAL vbfloat16m2_t nk_e2m3m1_to_bf16m2_rvvbf16_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    static nk_u16_t const table_magnitude_u16x32[32] = {
        0x0000, 0x3D80, 0x3E00, 0x3E40, 0x3E80, 0x3EA0, 0x3EC0, 0x3EE0, // exp=0: 0..7/16
        0x3F80, 0x3F90, 0x3FA0, 0x3FB0, 0x3FC0, 0x3FD0, 0x3FE0, 0x3FF0, // exp=1: 1.0..1.875
        0x4000, 0x4010, 0x4020, 0x4030, 0x4040, 0x4050, 0x4060, 0x4070, // exp=2: 2.0..3.75
        0x4080, 0x4090, 0x40A0, 0x40B0, 0x40C0, 0x40D0, 0x40E0, 0x40F0, // exp=3: 4.0..7.5
    };
    vuint8m1_t sign_u8m1 = __riscv_vsrl_vx_u8m1(raw_u8m1, 5, vector_length);
    vuint8m1_t magnitude_u8m1 = __riscv_vand_vx_u8m1(raw_u8m1, 0x1F, vector_length);
    vuint8m1_t offset_u8m1 = __riscv_vsll_vx_u8m1(magnitude_u8m1, 1, vector_length);
    vuint16m2_t value_u16m2 = __riscv_vluxei8_v_u16m2(table_magnitude_u16x32, offset_u8m1, vector_length);
    vuint16m2_t sign_u16m2 =
        __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 15, vector_length);
    vuint16m2_t result_u16m2 = __riscv_vor_vv_u16m2(value_u16m2, sign_u16m2, vector_length);
    return __riscv_vreinterpret_v_u16m2_bf16m2(result_u16m2);
}

/**
 *  @brief Convert e3m2 (1-3-2 sign-exp-mantissa, 6-bit) to bf16 via 32-entry VLUXEI8 LUT.
 *  Strip sign bit (bit 5), look up 5-bit magnitude in 32-entry table, OR sign into bit 15.
 */
NK_INTERNAL vbfloat16m2_t nk_e3m2m1_to_bf16m2_rvvbf16_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    static nk_u16_t const table_magnitude_u16x32[32] = {
        0x0000, 0x3D80, 0x3E00, 0x3E40, 0x3E80, 0x3EA0, 0x3EC0, 0x3EE0, // exp=0-1
        0x3F00, 0x3F20, 0x3F40, 0x3F60, 0x3F80, 0x3FA0, 0x3FC0, 0x3FE0, // exp=2-3
        0x4000, 0x4020, 0x4040, 0x4060, 0x4080, 0x40A0, 0x40C0, 0x40E0, // exp=4-5
        0x4100, 0x4120, 0x4140, 0x4160, 0x4180, 0x41A0, 0x41C0, 0x41E0, // exp=6-7
    };
    vuint8m1_t sign_u8m1 = __riscv_vsrl_vx_u8m1(raw_u8m1, 5, vector_length);
    vuint8m1_t magnitude_u8m1 = __riscv_vand_vx_u8m1(raw_u8m1, 0x1F, vector_length);
    vuint8m1_t offset_u8m1 = __riscv_vsll_vx_u8m1(magnitude_u8m1, 1, vector_length);
    vuint16m2_t value_u16m2 = __riscv_vluxei8_v_u16m2(table_magnitude_u16x32, offset_u8m1, vector_length);
    vuint16m2_t sign_u16m2 =
        __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 15, vector_length);
    vuint16m2_t result_u16m2 = __riscv_vor_vv_u16m2(value_u16m2, sign_u16m2, vector_length);
    return __riscv_vreinterpret_v_u16m2_bf16m2(result_u16m2);
}

/**
 *  @brief Convert e4m3 (1-4-3 sign-exp-mantissa, 8-bit) to bf16 via hybrid arithmetic + 8-entry subnormal LUT.
 *  Normal: bf16 = (lower7 << 4) + 0x3C00. Subnormal: 8-entry LUT. NaN: 0x7FC0.
 */
NK_INTERNAL vbfloat16m2_t nk_e4m3m1_to_bf16m2_rvvbf16_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    static nk_u16_t const table_subnormal_u16x8[8] = {0x0000, 0x3B00, 0x3B80, 0x3BC0,
                                                       0x3C00, 0x3C20, 0x3C40, 0x3C60};
    // Extract fields
    vuint8m1_t sign_u8m1 = __riscv_vsrl_vx_u8m1(raw_u8m1, 7, vector_length);
    vuint8m1_t lower7_u8m1 = __riscv_vand_vx_u8m1(raw_u8m1, 0x7F, vector_length);
    vuint16m2_t lower7_u16m2 = __riscv_vzext_vf2_u16m2(lower7_u8m1, vector_length);
    // Normal: bf16 = (lower7 << 4) + 0x3C00
    vuint16m2_t normal_u16m2 =
        __riscv_vadd_vx_u16m2(__riscv_vsll_vx_u16m2(lower7_u16m2, 4, vector_length), 0x3C00, vector_length);
    // Subnormal: 8-entry LUT via VLUXEI8
    vuint8m1_t mantissa_u8m1 = __riscv_vand_vx_u8m1(raw_u8m1, 0x07, vector_length);
    vuint8m1_t offset_u8m1 = __riscv_vsll_vx_u8m1(mantissa_u8m1, 1, vector_length);
    vuint16m2_t subnormal_u16m2 = __riscv_vluxei8_v_u16m2(table_subnormal_u16x8, offset_u8m1, vector_length);
    // Blend: subnormal where exp==0
    vuint8m1_t exponent_u8m1 = __riscv_vand_vx_u8m1(raw_u8m1, 0x78, vector_length);
    vbool8_t is_subnormal_b8 = __riscv_vmseq_vx_u8m1_b8(exponent_u8m1, 0, vector_length);
    vuint16m2_t result_u16m2 = __riscv_vmerge_vvm_u16m2(normal_u16m2, subnormal_u16m2, is_subnormal_b8, vector_length);
    // NaN: e4m3 index 127 → bf16 NaN (0x7FC0)
    vbool8_t is_nan_b8 = __riscv_vmseq_vx_u8m1_b8(lower7_u8m1, 0x7F, vector_length);
    result_u16m2 = __riscv_vmerge_vxm_u16m2(result_u16m2, 0x7FC0, is_nan_b8, vector_length);
    // Apply sign
    vuint16m2_t sign_u16m2 =
        __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 15, vector_length);
    result_u16m2 = __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
    return __riscv_vreinterpret_v_u16m2_bf16m2(result_u16m2);
}

/**
 *  @brief Convert e5m2 (1-5-2 sign-exp-mantissa, 8-bit) to bf16 via 4-entry subnormal LUT + arithmetic.
 *  Normal: bf16 = (lower7 << 5) + 0x3800. Subnormal: 4-entry LUT. Inf: 0x7F80. NaN: 0x7FC0.
 */
NK_INTERNAL vbfloat16m2_t nk_e5m2m1_to_bf16m2_rvvbf16_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    static nk_u16_t const table_subnormal_u16x4[4] = {0x0000, 0x3780, 0x3800, 0x3840};
    // Extract fields
    vuint8m1_t sign_u8m1 = __riscv_vsrl_vx_u8m1(raw_u8m1, 7, vector_length);
    vuint8m1_t lower7_u8m1 = __riscv_vand_vx_u8m1(raw_u8m1, 0x7F, vector_length);
    vuint16m2_t lower7_u16m2 = __riscv_vzext_vf2_u16m2(lower7_u8m1, vector_length);
    // Normal: bf16 = (lower7 << 5) + 0x3800
    vuint16m2_t normal_u16m2 =
        __riscv_vadd_vx_u16m2(__riscv_vsll_vx_u16m2(lower7_u16m2, 5, vector_length), 0x3800, vector_length);
    // Subnormal: 4-entry LUT via VLUXEI8
    vuint8m1_t mantissa_u8m1 = __riscv_vand_vx_u8m1(raw_u8m1, 0x03, vector_length);
    vuint8m1_t offset_u8m1 = __riscv_vsll_vx_u8m1(mantissa_u8m1, 1, vector_length);
    vuint16m2_t subnormal_u16m2 = __riscv_vluxei8_v_u16m2(table_subnormal_u16x4, offset_u8m1, vector_length);
    // Blend: subnormal where exp==0
    vuint8m1_t exponent_u8m1 = __riscv_vand_vx_u8m1(raw_u8m1, 0x7C, vector_length);
    vbool8_t is_subnormal_b8 = __riscv_vmseq_vx_u8m1_b8(exponent_u8m1, 0, vector_length);
    vuint16m2_t result_u16m2 = __riscv_vmerge_vvm_u16m2(normal_u16m2, subnormal_u16m2, is_subnormal_b8, vector_length);
    // Inf: e5m2 0x7C → bf16 inf (0x7F80)
    vbool8_t is_inf_b8 = __riscv_vmseq_vx_u8m1_b8(lower7_u8m1, 0x7C, vector_length);
    result_u16m2 = __riscv_vmerge_vxm_u16m2(result_u16m2, 0x7F80, is_inf_b8, vector_length);
    // NaN: e5m2 0x7D-0x7F → bf16 NaN (0x7FC0)
    vbool8_t is_nan_b8 = __riscv_vmsgtu_vx_u8m1_b8(lower7_u8m1, 0x7C, vector_length);
    result_u16m2 = __riscv_vmerge_vxm_u16m2(result_u16m2, 0x7FC0, is_nan_b8, vector_length);
    // Apply sign
    vuint16m2_t sign_u16m2 =
        __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 15, vector_length);
    result_u16m2 = __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
    return __riscv_vreinterpret_v_u16m2_bf16m2(result_u16m2);
}

NK_PUBLIC void nk_dot_e2m3_rvvbf16(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vbfloat16m2_t a_bf16m2 = nk_e2m3m1_to_bf16m2_rvvbf16_(a_u8m1, vector_length);
        vbfloat16m2_t b_bf16m2 = nk_e2m3m1_to_bf16m2_rvvbf16_(b_u8m1, vector_length);
        vfloat32m4_t acc_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vector_length);
        acc_f32m4 = __riscv_vfwmaccbf16_vv_f32m4(acc_f32m4, a_bf16m2, b_bf16m2, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(acc_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_dot_e3m2_rvvbf16(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vbfloat16m2_t a_bf16m2 = nk_e3m2m1_to_bf16m2_rvvbf16_(a_u8m1, vector_length);
        vbfloat16m2_t b_bf16m2 = nk_e3m2m1_to_bf16m2_rvvbf16_(b_u8m1, vector_length);
        vfloat32m4_t acc_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vector_length);
        acc_f32m4 = __riscv_vfwmaccbf16_vv_f32m4(acc_f32m4, a_bf16m2, b_bf16m2, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(acc_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_dot_e4m3_rvvbf16(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vbfloat16m2_t a_bf16m2 = nk_e4m3m1_to_bf16m2_rvvbf16_(a_u8m1, vector_length);
        vbfloat16m2_t b_bf16m2 = nk_e4m3m1_to_bf16m2_rvvbf16_(b_u8m1, vector_length);
        vfloat32m4_t acc_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vector_length);
        acc_f32m4 = __riscv_vfwmaccbf16_vv_f32m4(acc_f32m4, a_bf16m2, b_bf16m2, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(acc_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_dot_e5m2_rvvbf16(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vbfloat16m2_t a_bf16m2 = nk_e5m2m1_to_bf16m2_rvvbf16_(a_u8m1, vector_length);
        vbfloat16m2_t b_bf16m2 = nk_e5m2m1_to_bf16m2_rvvbf16_(b_u8m1, vector_length);
        vfloat32m4_t acc_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vector_length);
        acc_f32m4 = __riscv_vfwmaccbf16_vv_f32m4(acc_f32m4, a_bf16m2, b_bf16m2, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(acc_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_RVVBF16
#endif // NK_TARGET_RISCV_
#endif // NK_DOT_RVVBF16_H
