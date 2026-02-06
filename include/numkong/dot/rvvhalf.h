/**
 *  @brief SIMD-accelerated Dot Products for RISC-V FP16.
 *  @file include/numkong/dot/rvvhalf.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  SiFive P670/X280 and similar chips implement RVV 1.0 with Zvfh extension.
 *  Zvfh provides native half-precision (f16) vector operations.
 *  Uses widening multiply (f16 ⨯ f16 → f32) for precision, then reduces to f32.
 *
 *  For 6-bit types (e2m3, e3m2), conversion uses a 32-entry VLUXEI8 LUT gather.
 *  For 8-bit types (e4m3, e5m2), conversion uses hybrid arithmetic + small subnormal LUT.
 *  All variants then use vfwmul_vv for widening f16 ⨯ f16 → f32 multiply.
 *
 *  Requires: RVV 1.0 + Zvfh extension (GCC 14+ or Clang 18+)
 */
#ifndef NK_DOT_RVVHALF_H
#define NK_DOT_RVVHALF_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVVHALF

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_f16_rvvhalf(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e16m1(count_scalars);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)a_scalars, vl);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)b_scalars, vl);
        vfloat16m1_t a_f16m1 = __riscv_vreinterpret_v_u16m1_f16m1(a_u16m1);
        vfloat16m1_t b_f16m1 = __riscv_vreinterpret_v_u16m1_f16m1(b_u16m1);
        // Widening multiply: f16 ⨯ f16 → f32
        vfloat32m2_t ab_f32m2 = __riscv_vfwmul_vv_f32m2(a_f16m1, b_f16m1, vl);
        // Ordered reduction sum
        sum_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(ab_f32m2, sum_f32m1, vl);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

/**
 *  @brief Convert e2m3 (1-2-3 sign-exp-mantissa, 6-bit) to f16 via 32-entry VLUXEI8 LUT.
 *  Strip sign bit (bit 5), look up 5-bit magnitude in 32-entry table, OR sign into bit 15.
 */
NK_INTERNAL vfloat16m2_t nk_e2m3m1_to_f16m2_rvvhalf_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    static nk_u16_t const table_magnitude_u16x32[32] = {
        0x0000, 0x2C00, 0x3000, 0x3200, 0x3400, 0x3500, 0x3600, 0x3700, // exp=0: 0..7/16
        0x3C00, 0x3C80, 0x3D00, 0x3D80, 0x3E00, 0x3E80, 0x3F00, 0x3F80, // exp=1: 1.0..1.875
        0x4000, 0x4080, 0x4100, 0x4180, 0x4200, 0x4280, 0x4300, 0x4380, // exp=2: 2.0..3.75
        0x4400, 0x4480, 0x4500, 0x4580, 0x4600, 0x4680, 0x4700, 0x4780, // exp=3: 4.0..7.5
    };
    vuint8m1_t sign_u8m1 = __riscv_vsrl_vx_u8m1(raw_u8m1, 5, vector_length);
    vuint8m1_t magnitude_u8m1 = __riscv_vand_vx_u8m1(raw_u8m1, 0x1F, vector_length);
    vuint8m1_t offset_u8m1 = __riscv_vsll_vx_u8m1(magnitude_u8m1, 1, vector_length);
    vuint16m2_t value_u16m2 = __riscv_vluxei8_v_u16m2(table_magnitude_u16x32, offset_u8m1, vector_length);
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 15,
                                                   vector_length);
    vuint16m2_t result_u16m2 = __riscv_vor_vv_u16m2(value_u16m2, sign_u16m2, vector_length);
    return __riscv_vreinterpret_v_u16m2_f16m2(result_u16m2);
}

/**
 *  @brief Convert e3m2 (1-3-2 sign-exp-mantissa, 6-bit) to f16 via 32-entry VLUXEI8 LUT.
 *  Strip sign bit (bit 5), look up 5-bit magnitude in 32-entry table, OR sign into bit 15.
 */
NK_INTERNAL vfloat16m2_t nk_e3m2m1_to_f16m2_rvvhalf_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    static nk_u16_t const table_magnitude_u16x32[32] = {
        0x0000, 0x2C00, 0x3000, 0x3200, 0x3400, 0x3500, 0x3600, 0x3700, // exp=0-1
        0x3800, 0x3900, 0x3A00, 0x3B00, 0x3C00, 0x3D00, 0x3E00, 0x3F00, // exp=2-3
        0x4000, 0x4100, 0x4200, 0x4300, 0x4400, 0x4500, 0x4600, 0x4700, // exp=4-5
        0x4800, 0x4900, 0x4A00, 0x4B00, 0x4C00, 0x4D00, 0x4E00, 0x4F00, // exp=6-7
    };
    vuint8m1_t sign_u8m1 = __riscv_vsrl_vx_u8m1(raw_u8m1, 5, vector_length);
    vuint8m1_t magnitude_u8m1 = __riscv_vand_vx_u8m1(raw_u8m1, 0x1F, vector_length);
    vuint8m1_t offset_u8m1 = __riscv_vsll_vx_u8m1(magnitude_u8m1, 1, vector_length);
    vuint16m2_t value_u16m2 = __riscv_vluxei8_v_u16m2(table_magnitude_u16x32, offset_u8m1, vector_length);
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 15,
                                                   vector_length);
    vuint16m2_t result_u16m2 = __riscv_vor_vv_u16m2(value_u16m2, sign_u16m2, vector_length);
    return __riscv_vreinterpret_v_u16m2_f16m2(result_u16m2);
}

/**
 *  @brief Convert e4m3 (1-4-3 sign-exp-mantissa, 8-bit) to f16 via hybrid arithmetic + 8-entry subnormal LUT.
 *  Normal: f16 = (lower7 << 7) + 0x2000. Subnormal: 8-entry LUT. NaN: index 127 → 0x7E00.
 */
NK_INTERNAL vfloat16m2_t nk_e4m3m1_to_f16m2_rvvhalf_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    static nk_u16_t const table_subnormal_u16x8[8] = {0x0000, 0x1800, 0x1C00, 0x1E00, 0x2000, 0x2100, 0x2200, 0x2300};
    // Extract fields
    vuint8m1_t sign_u8m1 = __riscv_vsrl_vx_u8m1(raw_u8m1, 7, vector_length);
    vuint8m1_t lower7_u8m1 = __riscv_vand_vx_u8m1(raw_u8m1, 0x7F, vector_length);
    vuint16m2_t lower7_u16m2 = __riscv_vzext_vf2_u16m2(lower7_u8m1, vector_length);
    // Normal: f16 = (lower7 << 7) + 0x2000
    vuint16m2_t normal_u16m2 = __riscv_vadd_vx_u16m2(__riscv_vsll_vx_u16m2(lower7_u16m2, 7, vector_length), 0x2000,
                                                     vector_length);
    // Subnormal: 8-entry LUT via VLUXEI8
    vuint8m1_t mantissa_u8m1 = __riscv_vand_vx_u8m1(raw_u8m1, 0x07, vector_length);
    vuint8m1_t offset_u8m1 = __riscv_vsll_vx_u8m1(mantissa_u8m1, 1, vector_length);
    vuint16m2_t subnormal_u16m2 = __riscv_vluxei8_v_u16m2(table_subnormal_u16x8, offset_u8m1, vector_length);
    // Blend: subnormal where exp==0
    vuint8m1_t exponent_u8m1 = __riscv_vand_vx_u8m1(raw_u8m1, 0x78, vector_length);
    vbool8_t is_subnormal_b8 = __riscv_vmseq_vx_u8m1_b8(exponent_u8m1, 0, vector_length);
    vuint16m2_t result_u16m2 = __riscv_vmerge_vvm_u16m2(normal_u16m2, subnormal_u16m2, is_subnormal_b8, vector_length);
    // NaN: e4m3 index 127 → f16 NaN (0x7E00)
    vbool8_t is_nan_b8 = __riscv_vmseq_vx_u8m1_b8(lower7_u8m1, 0x7F, vector_length);
    result_u16m2 = __riscv_vmerge_vxm_u16m2(result_u16m2, 0x7E00, is_nan_b8, vector_length);
    // Apply sign
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 15,
                                                   vector_length);
    result_u16m2 = __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
    return __riscv_vreinterpret_v_u16m2_f16m2(result_u16m2);
}

/**
 *  @brief Convert e5m2 (1-5-2 sign-exp-mantissa, 8-bit) to f16 via pure shift (no LUT).
 *  Same exponent bias (15) means f16 = (lower7 << 8) | (sign << 15). Handles all cases.
 */
NK_INTERNAL vfloat16m2_t nk_e5m2m1_to_f16m2_rvvhalf_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    vuint16m2_t wide_u16m2 = __riscv_vzext_vf2_u16m2(raw_u8m1, vector_length);
    vuint16m2_t result_u16m2 = __riscv_vsll_vx_u16m2(wide_u16m2, 8, vector_length);
    return __riscv_vreinterpret_v_u16m2_f16m2(result_u16m2);
}

NK_PUBLIC void nk_dot_e2m3_rvvhalf(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat16m2_t a_f16m2 = nk_e2m3m1_to_f16m2_rvvhalf_(a_u8m1, vector_length);
        vfloat16m2_t b_f16m2 = nk_e2m3m1_to_f16m2_rvvhalf_(b_u8m1, vector_length);
        vfloat32m4_t ab_f32m4 = __riscv_vfwmul_vv_f32m4(a_f16m2, b_f16m2, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(ab_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_dot_e3m2_rvvhalf(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat16m2_t a_f16m2 = nk_e3m2m1_to_f16m2_rvvhalf_(a_u8m1, vector_length);
        vfloat16m2_t b_f16m2 = nk_e3m2m1_to_f16m2_rvvhalf_(b_u8m1, vector_length);
        vfloat32m4_t ab_f32m4 = __riscv_vfwmul_vv_f32m4(a_f16m2, b_f16m2, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(ab_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_dot_e4m3_rvvhalf(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat16m2_t a_f16m2 = nk_e4m3m1_to_f16m2_rvvhalf_(a_u8m1, vector_length);
        vfloat16m2_t b_f16m2 = nk_e4m3m1_to_f16m2_rvvhalf_(b_u8m1, vector_length);
        vfloat32m4_t ab_f32m4 = __riscv_vfwmul_vv_f32m4(a_f16m2, b_f16m2, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(ab_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_dot_e5m2_rvvhalf(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat16m2_t a_f16m2 = nk_e5m2m1_to_f16m2_rvvhalf_(a_u8m1, vector_length);
        vfloat16m2_t b_f16m2 = nk_e5m2m1_to_f16m2_rvvhalf_(b_u8m1, vector_length);
        vfloat32m4_t ab_f32m4 = __riscv_vfwmul_vv_f32m4(a_f16m2, b_f16m2, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(ab_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_RVVHALF
#endif // NK_TARGET_RISCV_
#endif // NK_DOT_RVVHALF_H
