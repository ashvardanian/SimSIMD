/**
 *  @brief SIMD-accelerated trigonometric element-wise operations optimized for Intel Sapphire Rapids CPUs.
 *  @file include/numkong/trigonometry/sapphire.h
 *  @sa include/numkong/trigonometry.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  Uses AVX-512 FP16 for native half-precision trigonometry:
 *  - 32 FP16 values per 512-bit register
 *  - Polynomial approximations tuned for FP16 precision (~3.3 decimal digits)
 *
 *  @section sapphire_trig_instructions Relevant Instructions
 *
 *      Intrinsic                   Instruction                     Sapphire    Genoa
 *      _mm512_mul_ph               VMULPH (ZMM, ZMM, ZMM)          4cy @ p05   3cy @ p01
 *      _mm512_fmadd_ph             VFMADD (ZMM, ZMM, ZMM)          4cy @ p05   4cy @ p01
 *      _mm512_fnmadd_ph            VFNMADD (ZMM, ZMM, ZMM)         4cy @ p05   4cy @ p01
 *      _mm512_fmsub_ph             VFMSUB (ZMM, ZMM, ZMM)          4cy @ p05   4cy @ p01
 *      _mm512_sub_ph               VSUBPH (ZMM, ZMM, ZMM)          4cy @ p05   3cy @ p01
 *      _mm512_div_ph               VDIVPH (ZMM, ZMM, ZMM)          10cy @ p0   10cy @ p0
 *      _mm512_abs_ph               VANDPS (ZMM, ZMM, ZMM)          1cy @ p05   1cy @ p0123
 *      _mm512_cvtph_epi16          VCVTPH2W (ZMM, ZMM)             4cy @ p05   4cy @ p01
 *      _mm512_roundscale_ph        VRNDSCALEPH (ZMM, ZMM, imm8)    4cy @ p05   4cy @ p01
 *      _mm512_cmp_ph_mask          VCMPPH (K, ZMM, ZMM, imm8)      3cy @ p5    3cy @ p0
 *      _mm512_test_epi16_mask      VPTESTMW (K, ZMM, ZMM)          3cy @ p5    3cy @ p0
 */
#ifndef NK_TRIGONOMETRY_SAPPHIRE_H
#define NK_TRIGONOMETRY_SAPPHIRE_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512fp16,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512fp16", "f16c", "fma", "bmi", "bmi2")
#endif

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief AVX-512 FP16 sine approximation for 32 half-precision values.
 *
 *  Uses a 3-term polynomial optimized for FP16 precision.
 *  Range reduction: angle mod π, then polynomial sin(x) ≈ x - x³/6 + x⁵/120
 */
NK_INTERNAL __m512h nk_f16x32_sin_sapphire_(__m512h const angles_radians) {
    // Constants for argument reduction (FP16 precision)
    __m512h const pi = _mm512_castsi512_ph(_mm512_set1_epi16(0x4248));             // 3.14159265
    __m512h const pi_reciprocal = _mm512_castsi512_ph(_mm512_set1_epi16(0x3518));  // 0.31830989
    __m512h const coeff_5 = _mm512_castsi512_ph(_mm512_set1_epi16((short)0x8A3A)); // -0.00019 (~-1/5!)
    __m512h const coeff_3 = _mm512_castsi512_ph(_mm512_set1_epi16(0x2044));        // +0.00833 (~1/3!)
    __m512h const coeff_1 = _mm512_castsi512_ph(_mm512_set1_epi16((short)0xB155)); // -0.16667 (~-1/3!)

    // Compute (multiples_of_pi) = round(angle / π)
    __m512h quotients = _mm512_mul_ph(angles_radians, pi_reciprocal);
    __m512h rounded_quotients = _mm512_roundscale_ph(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512i multiples_of_pi = _mm512_cvtph_epi16(rounded_quotients);

    // Reduce the angle to: (angle - (rounded_quotients * π)) in [-π/2, π/2]
    __m512h const angles = _mm512_fnmadd_ph(rounded_quotients, pi, angles_radians);
    __m512h const angles_squared = _mm512_mul_ph(angles, angles);
    __m512h const angles_cubed = _mm512_mul_ph(angles, angles_squared);

    // Compute the polynomial approximation: sin(x) ≈ x + x³(c₁ + x²(c₃ + x²c₅))
    __m512h polynomials = coeff_5;
    polynomials = _mm512_fmadd_ph(polynomials, angles_squared, coeff_3);
    polynomials = _mm512_fmadd_ph(polynomials, angles_squared, coeff_1);

    // If multiples_of_pi is odd, flip the sign of the results
    __mmask32 odd_mask = _mm512_test_epi16_mask(multiples_of_pi, _mm512_set1_epi16(1));
    __m512h results = _mm512_fmadd_ph(angles_cubed, polynomials, angles);
    results = _mm512_mask_sub_ph(results, odd_mask, _mm512_setzero_ph(), results);
    return results;
}

/**
 *  @brief AVX-512 FP16 cosine approximation for 32 half-precision values.
 *
 *  Uses cos(x) = sin(x + π/2) with the same polynomial.
 */
NK_INTERNAL __m512h nk_f16x32_cos_sapphire_(__m512h const angles_radians) {
    // Constants for argument reduction
    __m512h const pi = _mm512_castsi512_ph(_mm512_set1_epi16(0x4248));             // 3.14159265
    __m512h const pi_half = _mm512_castsi512_ph(_mm512_set1_epi16(0x3E48));        // 1.57079633
    __m512h const pi_reciprocal = _mm512_castsi512_ph(_mm512_set1_epi16(0x3518));  // 0.31830989
    __m512h const coeff_5 = _mm512_castsi512_ph(_mm512_set1_epi16((short)0x8A3A)); // -0.00019
    __m512h const coeff_3 = _mm512_castsi512_ph(_mm512_set1_epi16(0x2044));        // +0.00833
    __m512h const coeff_1 = _mm512_castsi512_ph(_mm512_set1_epi16((short)0xB155)); // -0.16667
    __m512h const half = _mm512_castsi512_ph(_mm512_set1_epi16(0x3800));           // 0.5

    // Compute (multiples_of_pi) = round((angle / π) - 0.5)
    __m512h quotients = _mm512_fmsub_ph(angles_radians, pi_reciprocal, half);
    __m512h rounded_quotients = _mm512_roundscale_ph(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512i multiples_of_pi = _mm512_cvtph_epi16(rounded_quotients);

    // Reduce the angle to: (angle - π/2 - (multiples_of_pi * π))
    __m512h const angles = _mm512_fnmadd_ph(rounded_quotients, pi, _mm512_sub_ph(angles_radians, pi_half));
    __m512h const angles_squared = _mm512_mul_ph(angles, angles);
    __m512h const angles_cubed = _mm512_mul_ph(angles, angles_squared);

    // Compute the polynomial approximation
    __m512h polynomials = coeff_5;
    polynomials = _mm512_fmadd_ph(polynomials, angles_squared, coeff_3);
    polynomials = _mm512_fmadd_ph(polynomials, angles_squared, coeff_1);
    __m512h results = _mm512_fmadd_ph(angles_cubed, polynomials, angles);

    // If multiples_of_pi is even, flip the sign of the results
    __mmask32 even_mask = _mm512_testn_epi16_mask(multiples_of_pi, _mm512_set1_epi16(1));
    results = _mm512_mask_sub_ph(results, even_mask, _mm512_setzero_ph(), results);
    return results;
}

/**
 *  @brief AVX-512 FP16 arctangent approximation for 32 half-precision values.
 *
 *  Uses a 4-term polynomial optimized for FP16 precision.
 */
NK_INTERNAL __m512h nk_f16x32_atan_sapphire_(__m512h const inputs) {
    // Polynomial coefficients (reduced precision for FP16)
    __m512h const coeff_4 = _mm512_castsi512_ph(_mm512_set1_epi16((short)0xB554)); // -0.333
    __m512h const coeff_3 = _mm512_castsi512_ph(_mm512_set1_epi16(0x3266));        // +0.200
    __m512h const coeff_2 = _mm512_castsi512_ph(_mm512_set1_epi16((short)0xB08B)); // -0.142
    __m512h const coeff_1 = _mm512_castsi512_ph(_mm512_set1_epi16(0x2EC9));        // +0.106
    __m512h const pi_half = _mm512_castsi512_ph(_mm512_set1_epi16(0x3E48));        // 1.5708
    __m512h const one = _mm512_castsi512_ph(_mm512_set1_epi16(0x3C00));            // 1.0

    // Quadrant adjustments
    __m512h values = inputs;
    __mmask32 const negative_mask = _mm512_cmp_ph_mask(values, _mm512_setzero_ph(), _CMP_LT_OS);
    values = _mm512_abs_ph(values);
    __mmask32 const reciprocal_mask = _mm512_cmp_ph_mask(values, one, _CMP_GT_OS);
    values = _mm512_mask_div_ph(values, reciprocal_mask, one, values);

    // Argument reduction
    __m512h const values_squared = _mm512_mul_ph(values, values);
    __m512h const values_cubed = _mm512_mul_ph(values, values_squared);

    // Polynomial evaluation: atan(x) ≈ x + x³(c₄ + x²(c₃ + x²(c₂ + x²c₁)))
    __m512h polynomials = coeff_1;
    polynomials = _mm512_fmadd_ph(polynomials, values_squared, coeff_2);
    polynomials = _mm512_fmadd_ph(polynomials, values_squared, coeff_3);
    polynomials = _mm512_fmadd_ph(polynomials, values_squared, coeff_4);

    // Compute result with quadrant adjustments
    __m512h result = _mm512_fmadd_ph(values_cubed, polynomials, values);
    result = _mm512_mask_sub_ph(result, reciprocal_mask, pi_half, result);
    result = _mm512_mask_sub_ph(result, negative_mask, _mm512_setzero_ph(), result);
    return result;
}

NK_PUBLIC void nk_sin_f16_sapphire(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512h angles = _mm512_loadu_ph(ins + i);
        __m512h results = nk_f16x32_sin_sapphire_(angles);
        _mm512_storeu_ph(outs + i, results);
    }
    if (i < n) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n - i);
        __m512h angles = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, ins + i));
        __m512h results = nk_f16x32_sin_sapphire_(angles);
        _mm512_mask_storeu_epi16(outs + i, mask, _mm512_castph_si512(results));
    }
}

NK_PUBLIC void nk_cos_f16_sapphire(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512h angles = _mm512_loadu_ph(ins + i);
        __m512h results = nk_f16x32_cos_sapphire_(angles);
        _mm512_storeu_ph(outs + i, results);
    }
    if (i < n) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n - i);
        __m512h angles = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, ins + i));
        __m512h results = nk_f16x32_cos_sapphire_(angles);
        _mm512_mask_storeu_epi16(outs + i, mask, _mm512_castph_si512(results));
    }
}

NK_PUBLIC void nk_atan_f16_sapphire(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512h values = _mm512_loadu_ph(ins + i);
        __m512h results = nk_f16x32_atan_sapphire_(values);
        _mm512_storeu_ph(outs + i, results);
    }
    if (i < n) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n - i);
        __m512h values = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, ins + i));
        __m512h results = nk_f16x32_atan_sapphire_(values);
        _mm512_mask_storeu_epi16(outs + i, mask, _mm512_castph_si512(results));
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_

#endif // NK_TRIGONOMETRY_SAPPHIRE_H
