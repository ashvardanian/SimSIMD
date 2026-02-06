/**
 *  @brief SIMD-accelerated Type Conversions for RISC-V.
 *  @file include/numkong/cast/rvv.h
 *  @author Ash Vardanian
 *  @date January 13, 2026
 *
 *  @sa include/numkong/cast.h
 *
 *  SpacemiT K1 and similar chips implement RVA22 profile with base RVV 1.0.
 *  This file provides vectorized type conversions for:
 *  - BF16 ↔ F32 (bit manipulation, no hardware support)
 *  - F16 ↔ F32 (bit manipulation, no hardware support)
 *  - E4M3 ↔ F32 (FP8 format for ML inference)
 *  - E5M2 ↔ F32 (FP8 format for ML training)
 *  - i4/u4 unpacking to i8/u8
 *
 *  @section rvv_cast_instructions Key RVV Cast Instructions
 *
 *      Intrinsic                       Purpose
 *      vzext_vf4_u32m4                 Zero-extend u8 → u32 (4x widening)
 *      vsext_vf4_i32m4                 Sign-extend i8 → i32 (4x widening)
 *      vsll_vx / vsrl_vx               Bit shifts for field extraction
 *      vand_vx                         Bit masking
 *      vor_vv                          Combining bit fields
 *      vfcvt_f_xu_v                    Unsigned int → float
 *      vmseq_vx                        Compare for conditional selection
 *      vmerge_vvm                      Conditional select (blend)
 */
#ifndef NK_CAST_RVV_H
#define NK_CAST_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_cast_serial`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region - Register-to-Register Helpers

/**
 *  @brief Convert bf16 (m1) to f32 (m2) register-to-register.
 *
 *  BF16 is the upper 16 bits of F32 (same sign + exponent + top 7 mantissa bits).
 *  Conversion is simply: f32_bits = bf16_bits << 16.
 */
NK_INTERNAL vfloat32m2_t nk_bf16m1_to_f32m2_rvv_(vuint16m1_t bf16_u16m1, nk_size_t vector_length) {
    vuint32m2_t bits_u32m2 = __riscv_vzext_vf2_u32m2(bf16_u16m1, vector_length);
    bits_u32m2 = __riscv_vsll_vx_u32m2(bits_u32m2, 16, vector_length);
    return __riscv_vreinterpret_v_u32m2_f32m2(bits_u32m2);
}

/**
 *  @brief Convert f32 (m2) to bf16 (m1) register-to-register.
 *
 *  Conversion with round-to-nearest-even (RNE): add (0x7FFF + lsb) to match hardware BF16 behavior.
 */
NK_INTERNAL vuint16m1_t nk_f32m2_to_bf16m1_rvv_(vfloat32m2_t f32_f32m2, nk_size_t vector_length) {
    vuint32m2_t bits_u32m2 = __riscv_vreinterpret_v_f32m2_u32m2(f32_f32m2);
    // Extract LSB of result (bit 16) for round-to-nearest-even
    vuint32m2_t lsb_u32m2 = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(bits_u32m2, 16, vector_length), 1,
                                                  vector_length);
    vuint32m2_t rounding_u32m2 = __riscv_vadd_vx_u32m2(lsb_u32m2, 0x7FFF, vector_length);
    vuint32m2_t rounded_u32m2 = __riscv_vadd_vv_u32m2(bits_u32m2, rounding_u32m2, vector_length);
    vuint32m2_t shifted_u32m2 = __riscv_vsrl_vx_u32m2(rounded_u32m2, 16, vector_length);
    return __riscv_vncvt_x_x_w_u16m1(shifted_u32m2, vector_length);
}

/**
 *  @brief Convert f16 (m1) to f32 (m2) register-to-register.
 *
 *  F16 format: S EEEEE MMMMMMMMMM (1 sign, 5 exponent bits with bias=15, 10 mantissa bits)
 *  F32 format: S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM (1 sign, 8 exponent bits with bias=127, 23 mantissa bits)
 *
 *  Conversion: Rebias exponent from 15 to 127, extend mantissa from 10 to 23 bits.
 */
NK_INTERNAL vfloat32m2_t nk_f16m1_to_f32m2_rvv_(vuint16m1_t f16_u16m1, nk_size_t vector_length) {
    // Widen to 32-bit for manipulation
    vuint32m2_t bits_u32m2 = __riscv_vzext_vf2_u32m2(f16_u16m1, vector_length);
    // Extract sign: (raw >> 15) << 31
    vuint32m2_t sign_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(bits_u32m2, 15, vector_length), 31,
                                                   vector_length);
    // Extract exponent: (raw >> 10) & 0x1F
    vuint32m2_t exponent_u32m2 = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(bits_u32m2, 10, vector_length), 0x1F,
                                                       vector_length);
    // Extract mantissa: raw & 0x3FF
    vuint32m2_t mantissa_u32m2 = __riscv_vand_vx_u32m2(bits_u32m2, 0x3FF, vector_length);
    // Rebias exponent (15 → 127): add 112
    vuint32m2_t f32_exponent_u32m2 = __riscv_vadd_vx_u32m2(exponent_u32m2, 112, vector_length);
    // Combine: sign | (exponent << 23) | (mantissa << 13)
    vuint32m2_t result_u32m2 = __riscv_vor_vv_u32m2(
        sign_u32m2,
        __riscv_vor_vv_u32m2(__riscv_vsll_vx_u32m2(f32_exponent_u32m2, 23, vector_length),
                             __riscv_vsll_vx_u32m2(mantissa_u32m2, 13, vector_length), vector_length),
        vector_length);
    return __riscv_vreinterpret_v_u32m2_f32m2(result_u32m2);
}

/**
 *  @brief Convert f32 (m2) to f16 (m1) register-to-register.
 *
 *  Conversion: Rebias exponent from 127 to 15, truncate mantissa from 23 to 10 bits with rounding.
 */
NK_INTERNAL vuint16m1_t nk_f32m2_to_f16m1_rvv_(vfloat32m2_t f32_f32m2, nk_size_t vector_length) {
    vuint32m2_t bits_u32m2 = __riscv_vreinterpret_v_f32m2_u32m2(f32_f32m2);
    // Extract sign: (raw >> 31) << 15
    vuint32m2_t sign_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(bits_u32m2, 31, vector_length), 15,
                                                   vector_length);
    // Extract exponent: (raw >> 23) & 0xFF
    vuint32m2_t exponent_u32m2 = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(bits_u32m2, 23, vector_length), 0xFF,
                                                       vector_length);
    // Extract mantissa: raw & 0x7FFFFF
    vuint32m2_t mantissa_u32m2 = __riscv_vand_vx_u32m2(bits_u32m2, 0x7FFFFF, vector_length);
    // Rebias exponent (127 → 15): subtract 112, clamp to [0, 31]
    // Note: This is a simplified conversion that doesn't handle subnormals or overflow properly
    vint32m2_t exponent_i32m2 = __riscv_vsub_vx_i32m2(__riscv_vreinterpret_v_u32m2_i32m2(exponent_u32m2), 112,
                                                      vector_length);
    exponent_i32m2 = __riscv_vmax_vx_i32m2(exponent_i32m2, 0, vector_length);
    vuint32m2_t f16_exponent_u32m2 = __riscv_vreinterpret_v_i32m2_u32m2(
        __riscv_vmin_vx_i32m2(exponent_i32m2, 31, vector_length));
    // Round mantissa: add 0x1000 (half of truncated bits) then shift
    vuint32m2_t rounded_mantissa_u32m2 = __riscv_vadd_vx_u32m2(mantissa_u32m2, 0x1000, vector_length);
    vuint32m2_t f16_mantissa_u32m2 = __riscv_vsrl_vx_u32m2(rounded_mantissa_u32m2, 13, vector_length);
    f16_mantissa_u32m2 = __riscv_vand_vx_u32m2(f16_mantissa_u32m2, 0x3FF, vector_length);
    // Combine: sign | (exponent << 10) | mantissa
    vuint32m2_t result_u32m2 = __riscv_vor_vv_u32m2(
        sign_u32m2,
        __riscv_vor_vv_u32m2(__riscv_vsll_vx_u32m2(f16_exponent_u32m2, 10, vector_length), f16_mantissa_u32m2,
                             vector_length),
        vector_length);
    return __riscv_vncvt_x_x_w_u16m1(result_u32m2, vector_length);
}

/**
 *  @brief Convert e4m3 (m1) to f32 (m4) register-to-register.
 *
 *  E4M3FN format: S EEEE MMM (1 sign, 4 exponent bits with bias=7, 3 mantissa bits)
 *  - Normal: value = (-1)^S * 2^(E-7) * (1 + M/8)
 *  - Subnormal (E=0): value = (-1)^S * M / 512
 *  - NaN: E=15 and M=7 (0x7F or 0xFF)
 *  - No infinity in E4M3FN
 */
NK_INTERNAL vfloat32m4_t nk_e4m3m1_to_f32m4_rvv_(vuint8m1_t e4m3_u8m1, nk_size_t vector_length) {
    // Widen to u32 for bit manipulation (4x widening: e8m1 → e32m4)
    vuint16m2_t e4m3_u16m2 = __riscv_vzext_vf2_u16m2(e4m3_u8m1, vector_length);
    vuint32m4_t e4m3_u32m4 = __riscv_vzext_vf2_u32m4(e4m3_u16m2, vector_length);

    // Extract sign: (raw >> 7) << 31
    vuint32m4_t sign_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vsrl_vx_u32m4(e4m3_u32m4, 7, vector_length), 31,
                                                   vector_length);

    // Extract exponent: (raw >> 3) & 0x0F
    vuint32m4_t exponent_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(e4m3_u32m4, 3, vector_length), 0x0F,
                                                       vector_length);

    // Extract mantissa: raw & 0x07
    vuint32m4_t mantissa_u32m4 = __riscv_vand_vx_u32m4(e4m3_u32m4, 0x07, vector_length);

    // Normal case: f32 = sign | ((exp + 120) << 23) | (mant << 20)
    vuint32m4_t f32_exponent_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vadd_vx_u32m4(exponent_u32m4, 120, vector_length),
                                                           23, vector_length);
    vuint32m4_t f32_mantissa_u32m4 = __riscv_vsll_vx_u32m4(mantissa_u32m4, 20, vector_length);
    vuint32m4_t normal_u32m4 = __riscv_vor_vv_u32m4(
        sign_u32m4, __riscv_vor_vv_u32m4(f32_exponent_u32m4, f32_mantissa_u32m4, vector_length), vector_length);

    // Subnormal case (exp == 0): value = sign | (mant / 512.0f as f32 bits)
    vfloat32m4_t subnorm_abs_f32m4 = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_xu_v_f32m4(mantissa_u32m4, vector_length),
                                                            1.0f / 512.0f, vector_length);
    vuint32m4_t subnorm_bits_u32m4 = __riscv_vreinterpret_v_f32m4_u32m4(subnorm_abs_f32m4);
    vuint32m4_t subnorm_u32m4 = __riscv_vor_vv_u32m4(sign_u32m4, subnorm_bits_u32m4, vector_length);

    // Select: if exp == 0, use subnormal; else use normal
    vbool8_t is_subnorm_b8 = __riscv_vmseq_vx_u32m4_b8(exponent_u32m4, 0, vector_length);
    vuint32m4_t result_u32m4 = __riscv_vmerge_vvm_u32m4(normal_u32m4, subnorm_u32m4, is_subnorm_b8, vector_length);

    // NaN case: E4M3FN has NaN when exp=15 AND mant=7
    vbool8_t exp_is_15_b8 = __riscv_vmseq_vx_u32m4_b8(exponent_u32m4, 15, vector_length);
    vbool8_t mant_is_7_b8 = __riscv_vmseq_vx_u32m4_b8(mantissa_u32m4, 7, vector_length);
    vbool8_t is_nan_b8 = __riscv_vmand_mm_b8(exp_is_15_b8, mant_is_7_b8, vector_length);
    vuint32m4_t nan_bits_u32m4 = __riscv_vor_vx_u32m4(sign_u32m4, 0x7FC00000, vector_length); // F32 quiet NaN
    result_u32m4 = __riscv_vmerge_vvm_u32m4(result_u32m4, nan_bits_u32m4, is_nan_b8, vector_length);

    return __riscv_vreinterpret_v_u32m4_f32m4(result_u32m4);
}

/**
 *  @brief Convert e5m2 (m1) to f32 (m4) register-to-register.
 *
 *  E5M2 format: S EEEEE MM (1 sign, 5 exponent bits with bias=15, 2 mantissa bits)
 *  - Normal: value = (-1)^S * 2^(E-15) * (1 + M/4)
 *  - Subnormal (E=0): value = (-1)^S * M / 65536
 *  - Infinity: E=31 and M=0
 *  - NaN: E=31 and M!=0
 */
NK_INTERNAL vfloat32m4_t nk_e5m2m1_to_f32m4_rvv_(vuint8m1_t e5m2_u8m1, nk_size_t vector_length) {
    // Widen to u32 for bit manipulation
    vuint16m2_t e5m2_u16m2 = __riscv_vzext_vf2_u16m2(e5m2_u8m1, vector_length);
    vuint32m4_t e5m2_u32m4 = __riscv_vzext_vf2_u32m4(e5m2_u16m2, vector_length);

    // Extract sign: (raw >> 7) << 31
    vuint32m4_t sign_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vsrl_vx_u32m4(e5m2_u32m4, 7, vector_length), 31,
                                                   vector_length);

    // Extract exponent: (raw >> 2) & 0x1F
    vuint32m4_t exponent_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(e5m2_u32m4, 2, vector_length), 0x1F,
                                                       vector_length);

    // Extract mantissa: raw & 0x03
    vuint32m4_t mantissa_u32m4 = __riscv_vand_vx_u32m4(e5m2_u32m4, 0x03, vector_length);

    // Normal case: f32 = sign | ((exp + 112) << 23) | (mant << 21)
    // E5M2 bias=15, F32 bias=127, so delta = 127-15 = 112
    vuint32m4_t f32_exponent_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vadd_vx_u32m4(exponent_u32m4, 112, vector_length),
                                                           23, vector_length);
    vuint32m4_t f32_mantissa_u32m4 = __riscv_vsll_vx_u32m4(mantissa_u32m4, 21, vector_length);
    vuint32m4_t normal_u32m4 = __riscv_vor_vv_u32m4(
        sign_u32m4, __riscv_vor_vv_u32m4(f32_exponent_u32m4, f32_mantissa_u32m4, vector_length), vector_length);

    // Subnormal case (exp == 0): value = sign | (mant / 65536.0f as f32 bits)
    vfloat32m4_t subnorm_abs_f32m4 = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_xu_v_f32m4(mantissa_u32m4, vector_length),
                                                            1.0f / 65536.0f, vector_length);
    vuint32m4_t subnorm_bits_u32m4 = __riscv_vreinterpret_v_f32m4_u32m4(subnorm_abs_f32m4);
    vuint32m4_t subnorm_u32m4 = __riscv_vor_vv_u32m4(sign_u32m4, subnorm_bits_u32m4, vector_length);

    // Select: if exp == 0, use subnormal; else use normal
    vbool8_t is_subnorm_b8 = __riscv_vmseq_vx_u32m4_b8(exponent_u32m4, 0, vector_length);
    vuint32m4_t result_u32m4 = __riscv_vmerge_vvm_u32m4(normal_u32m4, subnorm_u32m4, is_subnorm_b8, vector_length);

    // Infinity case: E=31 and M=0
    vbool8_t exp_is_31_b8 = __riscv_vmseq_vx_u32m4_b8(exponent_u32m4, 31, vector_length);
    vbool8_t mant_is_0_b8 = __riscv_vmseq_vx_u32m4_b8(mantissa_u32m4, 0, vector_length);
    vbool8_t is_inf_b8 = __riscv_vmand_mm_b8(exp_is_31_b8, mant_is_0_b8, vector_length);
    vuint32m4_t inf_bits_u32m4 = __riscv_vor_vx_u32m4(sign_u32m4, 0x7F800000, vector_length); // F32 infinity
    result_u32m4 = __riscv_vmerge_vvm_u32m4(result_u32m4, inf_bits_u32m4, is_inf_b8, vector_length);

    // NaN case: E=31 and M!=0
    vbool8_t mant_not_0_b8 = __riscv_vmsne_vx_u32m4_b8(mantissa_u32m4, 0, vector_length);
    vbool8_t is_nan_b8 = __riscv_vmand_mm_b8(exp_is_31_b8, mant_not_0_b8, vector_length);
    vuint32m4_t nan_bits_u32m4 = __riscv_vor_vx_u32m4(sign_u32m4, 0x7FC00000, vector_length); // F32 quiet NaN
    result_u32m4 = __riscv_vmerge_vvm_u32m4(result_u32m4, nan_bits_u32m4, is_nan_b8, vector_length);

    return __riscv_vreinterpret_v_u32m4_f32m4(result_u32m4);
}

/**
 *  @brief Convert e2m3 (m1) to f32 (m4) register-to-register.
 *
 *  E2M3FN format: S EE MMM (1 sign, 2 exponent bits with bias=1, 3 mantissa bits)
 *  - Normal: value = (-1)^S * 2^(E-1) * (1 + M/8)
 *  - Subnormal (E=0): value = (-1)^S * M / 8
 *  - No NaN or infinity in E2M3FN
 */
NK_INTERNAL vfloat32m4_t nk_e2m3m1_to_f32m4_rvv_(vuint8m1_t e2m3_u8m1, nk_size_t vector_length) {
    // Widen to u32 for bit manipulation (4x widening: e8m1 → e32m4)
    vuint16m2_t e2m3_u16m2 = __riscv_vzext_vf2_u16m2(e2m3_u8m1, vector_length);
    vuint32m4_t e2m3_u32m4 = __riscv_vzext_vf2_u32m4(e2m3_u16m2, vector_length);

    // Extract sign: ((raw >> 5) & 1) << 31  (sign bit is bit 5 in 6-bit format, mask needed for 8-bit storage)
    vuint32m4_t sign_u32m4 = __riscv_vsll_vx_u32m4(
        __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(e2m3_u32m4, 5, vector_length), 1, vector_length), 31,
        vector_length);

    // Extract exponent: (raw >> 3) & 0x03
    vuint32m4_t exponent_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(e2m3_u32m4, 3, vector_length), 0x03,
                                                       vector_length);

    // Extract mantissa: raw & 0x07
    vuint32m4_t mantissa_u32m4 = __riscv_vand_vx_u32m4(e2m3_u32m4, 0x07, vector_length);

    // Normal case: f32 = sign | ((exp + 126) << 23) | (mant << 20)
    // E2M3 bias=1, F32 bias=127, so delta = 127-1 = 126
    vuint32m4_t f32_exponent_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vadd_vx_u32m4(exponent_u32m4, 126, vector_length),
                                                           23, vector_length);
    vuint32m4_t f32_mantissa_u32m4 = __riscv_vsll_vx_u32m4(mantissa_u32m4, 20, vector_length);
    vuint32m4_t normal_u32m4 = __riscv_vor_vv_u32m4(
        sign_u32m4, __riscv_vor_vv_u32m4(f32_exponent_u32m4, f32_mantissa_u32m4, vector_length), vector_length);

    // Subnormal case (exp == 0): value = sign | (mant / 16.0f as f32 bits)
    // E2M3 subnormal: (-1)^S * 2^(-1) * (mantissa / 8) = mantissa / 16
    vfloat32m4_t subnorm_abs_f32m4 = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_xu_v_f32m4(mantissa_u32m4, vector_length),
                                                            1.0f / 16.0f, vector_length);
    vuint32m4_t subnorm_bits_u32m4 = __riscv_vreinterpret_v_f32m4_u32m4(subnorm_abs_f32m4);
    vuint32m4_t subnorm_u32m4 = __riscv_vor_vv_u32m4(sign_u32m4, subnorm_bits_u32m4, vector_length);

    // Select: if exp == 0, use subnormal; else use normal
    vbool8_t is_subnorm_b8 = __riscv_vmseq_vx_u32m4_b8(exponent_u32m4, 0, vector_length);
    vuint32m4_t result_u32m4 = __riscv_vmerge_vvm_u32m4(normal_u32m4, subnorm_u32m4, is_subnorm_b8, vector_length);

    return __riscv_vreinterpret_v_u32m4_f32m4(result_u32m4);
}

/**
 *  @brief Convert e3m2 (m1) to f32 (m4) register-to-register.
 *
 *  E3M2FN format: S EEE MM (1 sign, 3 exponent bits with bias=3, 2 mantissa bits)
 *  - Normal: value = (-1)^S * 2^(E-3) * (1 + M/4)
 *  - Subnormal (E=0): value = (-1)^S * M / 16
 *  - No NaN or infinity in E3M2FN
 */
NK_INTERNAL vfloat32m4_t nk_e3m2m1_to_f32m4_rvv_(vuint8m1_t e3m2_u8m1, nk_size_t vector_length) {
    // Widen to u32 for bit manipulation (4x widening: e8m1 → e32m4)
    vuint16m2_t e3m2_u16m2 = __riscv_vzext_vf2_u16m2(e3m2_u8m1, vector_length);
    vuint32m4_t e3m2_u32m4 = __riscv_vzext_vf2_u32m4(e3m2_u16m2, vector_length);

    // Extract sign: ((raw >> 5) & 1) << 31  (sign bit is bit 5 in 6-bit format, mask needed for 8-bit storage)
    vuint32m4_t sign_u32m4 = __riscv_vsll_vx_u32m4(
        __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(e3m2_u32m4, 5, vector_length), 1, vector_length), 31,
        vector_length);

    // Extract exponent: (raw >> 2) & 0x07
    vuint32m4_t exponent_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(e3m2_u32m4, 2, vector_length), 0x07,
                                                       vector_length);

    // Extract mantissa: raw & 0x03
    vuint32m4_t mantissa_u32m4 = __riscv_vand_vx_u32m4(e3m2_u32m4, 0x03, vector_length);

    // Normal case: f32 = sign | ((exp + 124) << 23) | (mant << 21)
    // E3M2 bias=3, F32 bias=127, so delta = 127-3 = 124
    vuint32m4_t f32_exponent_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vadd_vx_u32m4(exponent_u32m4, 124, vector_length),
                                                           23, vector_length);
    vuint32m4_t f32_mantissa_u32m4 = __riscv_vsll_vx_u32m4(mantissa_u32m4, 21, vector_length);
    vuint32m4_t normal_u32m4 = __riscv_vor_vv_u32m4(
        sign_u32m4, __riscv_vor_vv_u32m4(f32_exponent_u32m4, f32_mantissa_u32m4, vector_length), vector_length);

    // Subnormal case (exp == 0): value = sign | (mant / 16.0f as f32 bits)
    vfloat32m4_t subnorm_abs_f32m4 = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_xu_v_f32m4(mantissa_u32m4, vector_length),
                                                            1.0f / 16.0f, vector_length);
    vuint32m4_t subnorm_bits_u32m4 = __riscv_vreinterpret_v_f32m4_u32m4(subnorm_abs_f32m4);
    vuint32m4_t subnorm_u32m4 = __riscv_vor_vv_u32m4(sign_u32m4, subnorm_bits_u32m4, vector_length);

    // Select: if exp == 0, use subnormal; else use normal
    vbool8_t is_subnorm_b8 = __riscv_vmseq_vx_u32m4_b8(exponent_u32m4, 0, vector_length);
    vuint32m4_t result_u32m4 = __riscv_vmerge_vvm_u32m4(normal_u32m4, subnorm_u32m4, is_subnorm_b8, vector_length);

    return __riscv_vreinterpret_v_u32m4_f32m4(result_u32m4);
}

/**
 *  @brief Unpack i4 (m1) nibbles to i8 (m2) register-to-register.
 *
 *  Packed format: byte[i] contains two nibbles:
 *  - High nibble (bits [7:4]) → output[i*2]
 *  - Low nibble (bits [3:0]) → output[i*2+1]
 *
 *  Sign extension: 4-bit signed value [-8,7] extended to 8-bit.
 *  Trick: (x ^ 8) - 8 sign-extends a 4-bit value to larger type.
 *
 *  Returns a tuple of two m1 vectors (high nibbles, low nibbles) for segment store.
 */
NK_INTERNAL vint8m1x2_t nk_i4m1_to_i8m2_rvv_(vuint8m1_t packed_u8m1, nk_size_t vector_length) {
    // Extract high nibble (even indices in output)
    vuint8m1_t hi_u8m1 = __riscv_vsrl_vx_u8m1(packed_u8m1, 4, vector_length);
    // Sign extend: (x ^ 8) - 8
    vint8m1_t hi_i8m1 = __riscv_vsub_vx_i8m1(
        __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(hi_u8m1), 8, vector_length), 8, vector_length);

    // Extract low nibble (odd indices in output)
    vuint8m1_t lo_u8m1 = __riscv_vand_vx_u8m1(packed_u8m1, 0x0F, vector_length);
    // Sign extend: (x ^ 8) - 8
    vint8m1_t lo_i8m1 = __riscv_vsub_vx_i8m1(
        __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(lo_u8m1), 8, vector_length), 8, vector_length);

    return __riscv_vcreate_v_i8m1x2(hi_i8m1, lo_i8m1);
}

/**
 *  @brief Unpack u4 (m1) nibbles to u8 (m2) register-to-register.
 *
 *  Returns a tuple of two m1 vectors (high nibbles, low nibbles) for segment store.
 */
NK_INTERNAL vuint8m1x2_t nk_u4m1_to_u8m2_rvv_(vuint8m1_t packed_u8m1, nk_size_t vector_length) {
    // Extract high nibble (even indices in output)
    vuint8m1_t hi_u8m1 = __riscv_vsrl_vx_u8m1(packed_u8m1, 4, vector_length);

    // Extract low nibble (odd indices in output)
    vuint8m1_t lo_u8m1 = __riscv_vand_vx_u8m1(packed_u8m1, 0x0F, vector_length);

    return __riscv_vcreate_v_u8m1x2(hi_u8m1, lo_u8m1);
}

/**
 *  @brief Pack i8 (m2) to i4 (m1) nibbles register-to-register.
 *
 *  Takes a tuple of two m1 vectors (high nibbles, low nibbles from segment load).
 *  Values are clamped to [-8, 7] before packing.
 */
NK_INTERNAL vuint8m1_t nk_i8m2_to_i4m1_rvv_(vint8m1_t hi_i8m1, vint8m1_t lo_i8m1, nk_size_t vector_length) {
    // Clamp to [-8, 7]
    hi_i8m1 = __riscv_vmax_vx_i8m1(__riscv_vmin_vx_i8m1(hi_i8m1, 7, vector_length), -8, vector_length);
    lo_i8m1 = __riscv_vmax_vx_i8m1(__riscv_vmin_vx_i8m1(lo_i8m1, 7, vector_length), -8, vector_length);

    // Convert to unsigned nibbles: value & 0x0F
    vuint8m1_t hi_u4m1 = __riscv_vand_vx_u8m1(__riscv_vreinterpret_v_i8m1_u8m1(hi_i8m1), 0x0F, vector_length);
    vuint8m1_t lo_u4m1 = __riscv_vand_vx_u8m1(__riscv_vreinterpret_v_i8m1_u8m1(lo_i8m1), 0x0F, vector_length);

    // Pack: (hi << 4) | lo
    return __riscv_vor_vv_u8m1(__riscv_vsll_vx_u8m1(hi_u4m1, 4, vector_length), lo_u4m1, vector_length);
}

/**
 *  @brief Pack u8 (m2) to u4 (m1) nibbles register-to-register.
 *
 *  Takes a tuple of two m1 vectors (high nibbles, low nibbles from segment load).
 *  Values are clamped to [0, 15] before packing.
 */
NK_INTERNAL vuint8m1_t nk_u8m2_to_u4m1_rvv_(vuint8m1_t hi_u8m1, vuint8m1_t lo_u8m1, nk_size_t vector_length) {
    // Clamp to [0, 15]
    hi_u8m1 = __riscv_vminu_vx_u8m1(hi_u8m1, 15, vector_length);
    lo_u8m1 = __riscv_vminu_vx_u8m1(lo_u8m1, 15, vector_length);

    // Pack: (hi << 4) | lo
    return __riscv_vor_vv_u8m1(__riscv_vsll_vx_u8m1(hi_u8m1, 4, vector_length), lo_u8m1, vector_length);
}

#pragma endregion - Register - to - Register Helpers

#pragma region - Unified Cast Dispatcher

NK_PUBLIC void nk_cast_rvv(void const *from, nk_dtype_t from_type, nk_size_t count, void *to, nk_dtype_t to_type) {
    // bf16 → f32
    if (from_type == nk_bf16_k && to_type == nk_f32_k) {
        nk_bf16_t const *source = (nk_bf16_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e16m1(count);
            vuint16m1_t bf16_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)source, vector_length);
            vfloat32m2_t f32_f32m2 = nk_bf16m1_to_f32m2_rvv_(bf16_u16m1, vector_length);
            __riscv_vse32_v_f32m2(destination, f32_f32m2, vector_length);
        }
        return;
    }

    // f32 → bf16
    if (from_type == nk_f32_k && to_type == nk_bf16_k) {
        nk_f32_t const *source = (nk_f32_t const *)from;
        nk_bf16_t *destination = (nk_bf16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e32m2(count);
            vfloat32m2_t f32_f32m2 = __riscv_vle32_v_f32m2(source, vector_length);
            vuint16m1_t bf16_u16m1 = nk_f32m2_to_bf16m1_rvv_(f32_f32m2, vector_length);
            __riscv_vse16_v_u16m1((nk_u16_t *)destination, bf16_u16m1, vector_length);
        }
        return;
    }

    // f16 → f32
    if (from_type == nk_f16_k && to_type == nk_f32_k) {
        nk_f16_t const *source = (nk_f16_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e16m1(count);
            vuint16m1_t f16_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)source, vector_length);
            vfloat32m2_t f32_f32m2 = nk_f16m1_to_f32m2_rvv_(f16_u16m1, vector_length);
            __riscv_vse32_v_f32m2(destination, f32_f32m2, vector_length);
        }
        return;
    }

    // f32 → f16
    if (from_type == nk_f32_k && to_type == nk_f16_k) {
        nk_f32_t const *source = (nk_f32_t const *)from;
        nk_f16_t *destination = (nk_f16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e32m2(count);
            vfloat32m2_t f32_f32m2 = __riscv_vle32_v_f32m2(source, vector_length);
            vuint16m1_t f16_u16m1 = nk_f32m2_to_f16m1_rvv_(f32_f32m2, vector_length);
            __riscv_vse16_v_u16m1((nk_u16_t *)destination, f16_u16m1, vector_length);
        }
        return;
    }

    // e4m3 → f32
    if (from_type == nk_e4m3_k && to_type == nk_f32_k) {
        nk_e4m3_t const *source = (nk_e4m3_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e4m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vfloat32m4_t f32_f32m4 = nk_e4m3m1_to_f32m4_rvv_(e4m3_u8m1, vector_length);
            __riscv_vse32_v_f32m4(destination, f32_f32m4, vector_length);
        }
        return;
    }

    // e5m2 → f32
    if (from_type == nk_e5m2_k && to_type == nk_f32_k) {
        nk_e5m2_t const *source = (nk_e5m2_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e5m2_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vfloat32m4_t f32_f32m4 = nk_e5m2m1_to_f32m4_rvv_(e5m2_u8m1, vector_length);
            __riscv_vse32_v_f32m4(destination, f32_f32m4, vector_length);
        }
        return;
    }

    // i4 → i8
    if (from_type == nk_i4_k && to_type == nk_i8_k) {
        nk_i4x2_t const *source = (nk_i4x2_t const *)from;
        nk_i8_t *destination = (nk_i8_t *)to;
        nk_size_t n_bytes = count / 2;
        for (nk_size_t vector_length; n_bytes > 0;
             n_bytes -= vector_length, source += vector_length, destination += vector_length * 2) {
            vector_length = __riscv_vsetvl_e8m1(n_bytes);
            vuint8m1_t packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vint8m1x2_t unpacked_i8m1x2 = nk_i4m1_to_i8m2_rvv_(packed_u8m1, vector_length);
            __riscv_vsseg2e8_v_i8m1x2(destination, unpacked_i8m1x2, vector_length);
        }
        return;
    }

    // u4 → u8
    if (from_type == nk_u4_k && to_type == nk_u8_k) {
        nk_u4x2_t const *source = (nk_u4x2_t const *)from;
        nk_u8_t *destination = (nk_u8_t *)to;
        nk_size_t n_bytes = count / 2;
        for (nk_size_t vector_length; n_bytes > 0;
             n_bytes -= vector_length, source += vector_length, destination += vector_length * 2) {
            vector_length = __riscv_vsetvl_e8m1(n_bytes);
            vuint8m1_t packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint8m1x2_t unpacked_u8m1x2 = nk_u4m1_to_u8m2_rvv_(packed_u8m1, vector_length);
            __riscv_vsseg2e8_v_u8m1x2(destination, unpacked_u8m1x2, vector_length);
        }
        return;
    }

    // i8 → i4
    if (from_type == nk_i8_k && to_type == nk_i4_k) {
        nk_i8_t const *source = (nk_i8_t const *)from;
        nk_i4x2_t *destination = (nk_i4x2_t *)to;
        nk_size_t n_bytes = count / 2;
        for (nk_size_t vector_length; n_bytes > 0;
             n_bytes -= vector_length, source += vector_length * 2, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(n_bytes);
            vint8m1x2_t loaded_i8m1x2 = __riscv_vlseg2e8_v_i8m1x2(source, vector_length);
            vint8m1_t hi_i8m1 = __riscv_vget_v_i8m1x2_i8m1(loaded_i8m1x2, 0);
            vint8m1_t lo_i8m1 = __riscv_vget_v_i8m1x2_i8m1(loaded_i8m1x2, 1);
            vuint8m1_t packed_u8m1 = nk_i8m2_to_i4m1_rvv_(hi_i8m1, lo_i8m1, vector_length);
            __riscv_vse8_v_u8m1((nk_u8_t *)destination, packed_u8m1, vector_length);
        }
        return;
    }

    // u8 → u4
    if (from_type == nk_u8_k && to_type == nk_u4_k) {
        nk_u8_t const *source = (nk_u8_t const *)from;
        nk_u4x2_t *destination = (nk_u4x2_t *)to;
        nk_size_t n_bytes = count / 2;
        for (nk_size_t vector_length; n_bytes > 0;
             n_bytes -= vector_length, source += vector_length * 2, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(n_bytes);
            vuint8m1x2_t loaded_u8m1x2 = __riscv_vlseg2e8_v_u8m1x2(source, vector_length);
            vuint8m1_t hi_u8m1 = __riscv_vget_v_u8m1x2_u8m1(loaded_u8m1x2, 0);
            vuint8m1_t lo_u8m1 = __riscv_vget_v_u8m1x2_u8m1(loaded_u8m1x2, 1);
            vuint8m1_t packed_u8m1 = nk_u8m2_to_u4m1_rvv_(hi_u8m1, lo_u8m1, vector_length);
            __riscv_vse8_v_u8m1((nk_u8_t *)destination, packed_u8m1, vector_length);
        }
        return;
    }

    // Fallback to serial for unimplemented conversions
    nk_cast_serial(from, from_type, count, to, to_type);
}

#pragma endregion - Unified Cast Dispatcher

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_CAST_RVV_H
