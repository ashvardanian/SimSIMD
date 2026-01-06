/**
 *  @brief SIMD-accelerated trigonometric element-wise operations, based on SLEEF, optimized for Intel Skylake-X CPUs.
 *  @file include/numkong/elementwise/skylake.h
 *  @sa include/numkong/elementwise.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_TRIGONOMETRY_SKYLAKE_H
#define NK_TRIGONOMETRY_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL __m512 nk_f32x16_sin_skylake_(__m512 const angles_radians) {
    // Constants for argument reduction
    __m512 const pi = _mm512_set1_ps(3.14159265358979323846f);            // π
    __m512 const pi_reciprocal = _mm512_set1_ps(0.31830988618379067154f); // 1/π
    __m512 const coeff_5 = _mm512_set1_ps(-0.0001881748176f);             // Coefficient for x^5 term
    __m512 const coeff_3 = _mm512_set1_ps(+0.008323502727f);              // Coefficient for x^3 term
    __m512 const coeff_1 = _mm512_set1_ps(-0.1666651368f);                // Coefficient for x term

    // Compute (multiples_of_pi) = round(angle / π)
    __m512 quotients = _mm512_mul_ps(angles_radians, pi_reciprocal);
    __m512 rounded_quotients = _mm512_roundscale_ps(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512i multiples_of_pi = _mm512_cvtps_epi32(rounded_quotients);

    // Reduce the angle to: (angle - (rounded_quotients * π)) in [0, π]
    __m512 const angles = _mm512_fnmadd_ps(rounded_quotients, pi, angles_radians);
    __m512 const angles_squared = _mm512_mul_ps(angles, angles);
    __m512 const angles_cubed = _mm512_mul_ps(angles, angles_squared);

    // Compute the polynomial approximation
    __m512 polynomials = coeff_5;
    polynomials = _mm512_fmadd_ps(polynomials, angles_squared, coeff_3);
    polynomials = _mm512_fmadd_ps(polynomials, angles_squared, coeff_1);

    // If multiples_of_pi is odd, flip the sign of the results
    __mmask16 odd_mask = _mm512_test_epi32_mask(multiples_of_pi, _mm512_set1_epi32(1));
    __m512 results = _mm512_fmadd_ps(angles_cubed, polynomials, angles);
    results = _mm512_mask_sub_ps(results, odd_mask, _mm512_setzero_ps(), results);
    return results;
}

NK_INTERNAL __m512 nk_f32x16_cos_skylake_(__m512 const angles_radians) {
    // Constants for argument reduction
    __m512 const pi = _mm512_set1_ps(3.14159265358979323846f);            // π
    __m512 const pi_half = _mm512_set1_ps(1.57079632679489661923f);       // π/2
    __m512 const pi_reciprocal = _mm512_set1_ps(0.31830988618379067154f); // 1/π
    __m512 const coeff_5 = _mm512_set1_ps(-0.0001881748176f);             // Coefficient for x^5 term
    __m512 const coeff_3 = _mm512_set1_ps(+0.008323502727f);              // Coefficient for x^3 term
    __m512 const coeff_1 = _mm512_set1_ps(-0.1666651368f);                // Coefficient for x term

    // Compute (multiples_of_pi) = round((angle / π) - 0.5)
    __m512 quotients = _mm512_fmsub_ps(angles_radians, pi_reciprocal, _mm512_set1_ps(0.5f));
    __m512 rounded_quotients = _mm512_roundscale_ps(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512i multiples_of_pi = _mm512_cvtps_epi32(rounded_quotients);

    // Reduce the angle to: (angle - (multiples_of_pi * π)) in [-π/2, π/2]
    __m512 const angles = _mm512_fnmadd_ps(rounded_quotients, pi, _mm512_sub_ps(angles_radians, pi_half));
    __m512 const angles_squared = _mm512_mul_ps(angles, angles);
    __m512 const angles_cubed = _mm512_mul_ps(angles, angles_squared);

    // Compute the polynomials approximation
    __m512 polynomials = coeff_5;
    polynomials = _mm512_fmadd_ps(polynomials, angles_squared, coeff_3);
    polynomials = _mm512_fmadd_ps(polynomials, angles_squared, coeff_1);
    __m512 results = _mm512_fmadd_ps(angles_cubed, polynomials, angles);

    // If multiples_of_pi is even, flip the sign of the results
    __mmask16 even_mask = _mm512_testn_epi32_mask(multiples_of_pi, _mm512_set1_epi32(1));
    results = _mm512_mask_sub_ps(results, even_mask, _mm512_setzero_ps(), results);
    return results;
}

NK_INTERNAL __m512 nk_f32x16_atan_skylake_(__m512 const inputs) {
    // Polynomial coefficients
    __m512 const coeff_8 = _mm512_set1_ps(-0.333331018686294555664062f);
    __m512 const coeff_7 = _mm512_set1_ps(+0.199926957488059997558594f);
    __m512 const coeff_6 = _mm512_set1_ps(-0.142027363181114196777344f);
    __m512 const coeff_5 = _mm512_set1_ps(+0.106347933411598205566406f);
    __m512 const coeff_4 = _mm512_set1_ps(-0.0748900920152664184570312f);
    __m512 const coeff_3 = _mm512_set1_ps(+0.0425049886107444763183594f);
    __m512 const coeff_2 = _mm512_set1_ps(-0.0159569028764963150024414f);
    __m512 const coeff_1 = _mm512_set1_ps(+0.00282363896258175373077393f);

    // Adjust for quadrant
    __m512 values = inputs;
    __mmask16 const negative_mask = _mm512_fpclass_ps_mask(values, 0x40);
    values = _mm512_abs_ps(values);
    __mmask16 const reciprocal_mask = _mm512_cmp_ps_mask(values, _mm512_set1_ps(1.0f), _CMP_GT_OS);
    values = _mm512_mask_div_ps(values, reciprocal_mask, _mm512_set1_ps(1.0f), values);

    // Argument reduction
    __m512 const values_squared = _mm512_mul_ps(values, values);
    __m512 const values_cubed = _mm512_mul_ps(values, values_squared);

    // Polynomial evaluation
    __m512 polynomials = coeff_1;
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_2);
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_3);
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_4);
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_5);
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_6);
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_7);
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_8);

    // Adjust result for quadrants
    __m512 result = _mm512_fmadd_ps(values_cubed, polynomials, values);
    result = _mm512_mask_sub_ps(result, reciprocal_mask, _mm512_set1_ps(1.5707963267948966f), result);
    result = _mm512_mask_sub_ps(result, negative_mask, _mm512_setzero_ps(), result);
    return result;
}

NK_INTERNAL __m512 nk_f32x16_atan2_skylake_(__m512 const ys_inputs, __m512 const xs_inputs) {
    // Polynomial coefficients
    __m512 const coeff_8 = _mm512_set1_ps(-0.333331018686294555664062f);
    __m512 const coeff_7 = _mm512_set1_ps(+0.199926957488059997558594f);
    __m512 const coeff_6 = _mm512_set1_ps(-0.142027363181114196777344f);
    __m512 const coeff_5 = _mm512_set1_ps(+0.106347933411598205566406f);
    __m512 const coeff_4 = _mm512_set1_ps(-0.0748900920152664184570312f);
    __m512 const coeff_3 = _mm512_set1_ps(+0.0425049886107444763183594f);
    __m512 const coeff_2 = _mm512_set1_ps(-0.0159569028764963150024414f);
    __m512 const coeff_1 = _mm512_set1_ps(+0.00282363896258175373077393f);

    // Quadrant adjustments normalizing to absolute values of x and y
    __mmask16 const xs_negative_mask = _mm512_fpclass_ps_mask(xs_inputs, 0x40);
    __m512 xs = _mm512_abs_ps(xs_inputs);
    __m512 ys = _mm512_abs_ps(ys_inputs);
    // Ensure proper fraction where the numerator is smaller than the denominator
    __mmask16 const swap_mask = _mm512_cmp_ps_mask(ys, xs, _CMP_GT_OS);
    __m512 temps = xs;
    xs = _mm512_mask_blend_ps(swap_mask, xs, ys);
    ys = _mm512_mask_sub_ps(ys, swap_mask, _mm512_setzero_ps(), temps);

    // Compute ratio and ratio^2
    __m512 const ratio = _mm512_div_ps(ys, xs);
    __m512 const ratio_squared = _mm512_mul_ps(ratio, ratio);
    __m512 const ratio_cubed = _mm512_mul_ps(ratio, ratio_squared);

    // Polynomial evaluation
    __m512 polynomials = coeff_1;
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_2);
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_3);
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_4);
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_5);
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_6);
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_7);
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_8);

    // Compute the result, but unlike the serial version, we don't keep the quadrant index
    // in a form of an integer to compute the `quadrant * (π/2)` term, instead we use the
    // masks to achieve the same.
    __m512 results = _mm512_fmadd_ps(ratio_cubed, polynomials, ratio);
    results = _mm512_mask_sub_ps(results, xs_negative_mask, results, _mm512_set1_ps(3.14159265358979323846f));
    results = _mm512_mask_add_ps(results, swap_mask, results, _mm512_set1_ps(1.5707963267948966f));

    // Special cases handling doesn't even require constants, as AVX-512 can automatically classify infinities.
    // However, those `_mm512_fpclass_ps_mask` ~ `VFPCLASSPS (K, ZMM, I8)` instructions aren't free:
    //
    // - On Intel they generally cost 4 cycles and operate only on port 5.
    // - On AMD, its 5 cycles and two ports: 0 and 1.
    //
    // The alternative is to use equality comparions like `_mm512_cmpeq_ps_mask` ~ `VCMPPS (K, ZMM, ZMM, I8)`:
    //
    // - On Intel they generally cost 4 cycles and operate only on port 5.
    // - On AMD, its 5 cycles and two ports: 0 and 1.
    //
    // ! Same as before, so not much space for latency hiding!
    // ! Integer comparison for 32 bit types also have the same cost on the same ports.
    __mmask16 const xs_is_inf = _mm512_fpclass_ps_mask(xs, 0x18);
    __mmask16 const ys_is_inf = _mm512_fpclass_ps_mask(ys, 0x18);
    __mmask16 const xs_is_zero = _mm512_fpclass_ps_mask(xs, 0x06);
    __mmask16 const ys_is_zero = _mm512_fpclass_ps_mask(ys, 0x06);

    // Adjust sign based on x
    results = _mm512_mask_xor_ps(results, xs_negative_mask, results, _mm512_set1_ps(-0.0f));

    return results;
}

NK_PUBLIC void nk_sin_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 angles = _mm512_loadu_ps(ins + i);
        __m512 results = nk_f32x16_sin_skylake_(angles);
        _mm512_storeu_ps(outs + i, results);
    }
    if (i < n) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m512 angles = _mm512_maskz_loadu_ps(mask, ins + i);
        __m512 results = nk_f32x16_sin_skylake_(angles);
        _mm512_mask_storeu_ps(outs + i, mask, results);
    }
}
NK_PUBLIC void nk_cos_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 angles = _mm512_loadu_ps(ins + i);
        __m512 results = nk_f32x16_cos_skylake_(angles);
        _mm512_storeu_ps(outs + i, results);
    }
    if (i < n) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m512 angles = _mm512_maskz_loadu_ps(mask, ins + i);
        __m512 results = nk_f32x16_cos_skylake_(angles);
        _mm512_mask_storeu_ps(outs + i, mask, results);
    }
}
NK_PUBLIC void nk_atan_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 angles = _mm512_loadu_ps(ins + i);
        __m512 results = nk_f32x16_atan_skylake_(angles);
        _mm512_storeu_ps(outs + i, results);
    }
    if (i < n) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m512 angles = _mm512_maskz_loadu_ps(mask, ins + i);
        __m512 results = nk_f32x16_atan_skylake_(angles);
        _mm512_mask_storeu_ps(outs + i, mask, results);
    }
}

NK_INTERNAL __m512d nk_f64x8_sin_skylake_(__m512d const angles_radians) {
    // Constants for argument reduction
    __m512d const pi_high = _mm512_set1_pd(3.141592653589793116);         // High-digits part of π
    __m512d const pi_low = _mm512_set1_pd(1.2246467991473532072e-16);     // Low-digits part of π
    __m512d const pi_reciprocal = _mm512_set1_pd(0.31830988618379067154); // 1/π

    // Polynomial coefficients for sine/cosine approximation (minimax polynomial)
    __m512d const coeff_0 = _mm512_set1_pd(+0.00833333333333332974823815);
    __m512d const coeff_1 = _mm512_set1_pd(-0.000198412698412696162806809);
    __m512d const coeff_2 = _mm512_set1_pd(+2.75573192239198747630416e-06);
    __m512d const coeff_3 = _mm512_set1_pd(-2.50521083763502045810755e-08);
    __m512d const coeff_4 = _mm512_set1_pd(+1.60590430605664501629054e-10);
    __m512d const coeff_5 = _mm512_set1_pd(-7.64712219118158833288484e-13);
    __m512d const coeff_6 = _mm512_set1_pd(+2.81009972710863200091251e-15);
    __m512d const coeff_7 = _mm512_set1_pd(-7.97255955009037868891952e-18);
    __m512d const coeff_8 = _mm512_set1_pd(-0.166666666666666657414808);

    // Compute (rounded_quotients) = round(angle / π)
    __m512d const quotients = _mm512_mul_pd(angles_radians, pi_reciprocal);
    __m512d const rounded_quotients = _mm512_roundscale_pd(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Reduce the angle to: angle - (rounded_quotients * π_high + rounded_quotients * π_low)
    __m512d angles = angles_radians;
    angles = _mm512_fnmadd_pd(rounded_quotients, pi_high, angles);
    angles = _mm512_fnmadd_pd(rounded_quotients, pi_low, angles);

    // If rounded_quotients is odd (bit 0 set), negate the angle
    __mmask8 const sign_flip_mask = _mm256_test_epi32_mask(_mm512_cvtpd_epi32(rounded_quotients), _mm256_set1_epi32(1));
    angles = _mm512_mask_sub_pd(angles, sign_flip_mask, _mm512_setzero_pd(), angles);

    __m512d const angles_squared = _mm512_mul_pd(angles, angles);
    __m512d const angles_cubed = _mm512_mul_pd(angles, angles_squared);
    __m512d const angles_quadratic = _mm512_mul_pd(angles_squared, angles_squared);
    __m512d const angles_octic = _mm512_mul_pd(angles_quadratic, angles_quadratic);

    // Compute higher-degree polynomial terms
    __m512d const poly_67 = _mm512_fmadd_pd(angles_squared, coeff_7, coeff_6);
    __m512d const poly_45 = _mm512_fmadd_pd(angles_squared, coeff_5, coeff_4);
    __m512d const poly_4567 = _mm512_fmadd_pd(angles_quadratic, poly_67, poly_45);

    // Compute lower-degree polynomial terms
    __m512d const poly_23 = _mm512_fmadd_pd(angles_squared, coeff_3, coeff_2);
    __m512d const poly_01 = _mm512_fmadd_pd(angles_squared, coeff_1, coeff_0);
    __m512d const poly_0123 = _mm512_fmadd_pd(angles_quadratic, poly_23, poly_01);

    // Combine polynomial terms
    __m512d results = _mm512_fmadd_pd(angles_octic, poly_4567, poly_0123);
    results = _mm512_fmadd_pd(results, angles_squared, coeff_8);
    results = _mm512_fmadd_pd(results, angles_cubed, angles);

    // Handle the special case of negative zero input
    __mmask8 const non_zero_mask = _mm512_cmpneq_pd_mask(angles_radians, _mm512_setzero_pd());
    results = _mm512_maskz_mov_pd(non_zero_mask, results);
    return results;
}

NK_INTERNAL __m512d nk_f64x8_cos_skylake_(__m512d const angles_radians) {
    // Constants for argument reduction
    __m512d const pi_high_half = _mm512_set1_pd(3.141592653589793116 * 0.5);     // High-digits part of π
    __m512d const pi_low_half = _mm512_set1_pd(1.2246467991473532072e-16 * 0.5); // Low-digits part of π
    __m512d const pi_reciprocal = _mm512_set1_pd(0.31830988618379067154);        // 1/π

    // Polynomial coefficients for sine/cosine approximation (minimax polynomial)
    __m512d const coeff_0 = _mm512_set1_pd(+0.00833333333333332974823815);
    __m512d const coeff_1 = _mm512_set1_pd(-0.000198412698412696162806809);
    __m512d const coeff_2 = _mm512_set1_pd(+2.75573192239198747630416e-06);
    __m512d const coeff_3 = _mm512_set1_pd(-2.50521083763502045810755e-08);
    __m512d const coeff_4 = _mm512_set1_pd(+1.60590430605664501629054e-10);
    __m512d const coeff_5 = _mm512_set1_pd(-7.64712219118158833288484e-13);
    __m512d const coeff_6 = _mm512_set1_pd(+2.81009972710863200091251e-15);
    __m512d const coeff_7 = _mm512_set1_pd(-7.97255955009037868891952e-18);
    __m512d const coeff_8 = _mm512_set1_pd(-0.166666666666666657414808);

    // Compute (rounded_quotients) = 2 * round(angle / π - 0.5) + 1
    // Use fmsub: a*b - c = angles * (1/π) - 0.5
    __m512d const quotients = _mm512_fmsub_pd(angles_radians, pi_reciprocal, _mm512_set1_pd(0.5));
    __m512d const rounded_quotients = _mm512_fmadd_pd(                                  //
        _mm512_set1_pd(2),                                                              //
        _mm512_roundscale_pd(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), //
        _mm512_set1_pd(1));

    // Reduce the angle to: angle - (rounded_quotients * π_high + rounded_quotients * π_low)
    __m512d angles = angles_radians;
    angles = _mm512_fnmadd_pd(rounded_quotients, pi_high_half, angles);
    angles = _mm512_fnmadd_pd(rounded_quotients, pi_low_half, angles);
    __mmask8 const sign_flip_mask = _mm256_testn_epi32_mask(_mm512_cvtpd_epi32(rounded_quotients),
                                                            _mm256_set1_epi32(2));
    angles = _mm512_mask_sub_pd(angles, sign_flip_mask, _mm512_setzero_pd(), angles);
    __m512d const angles_squared = _mm512_mul_pd(angles, angles);
    __m512d const angles_cubed = _mm512_mul_pd(angles, angles_squared);
    __m512d const angles_quadratic = _mm512_mul_pd(angles_squared, angles_squared);
    __m512d const angles_octic = _mm512_mul_pd(angles_quadratic, angles_quadratic);

    // Compute higher-degree polynomial terms
    __m512d const poly_67 = _mm512_fmadd_pd(angles_squared, coeff_7, coeff_6);
    __m512d const poly_45 = _mm512_fmadd_pd(angles_squared, coeff_5, coeff_4);
    __m512d const poly_4567 = _mm512_fmadd_pd(angles_quadratic, poly_67, poly_45);

    // Compute lower-degree polynomial terms
    __m512d const poly_23 = _mm512_fmadd_pd(angles_squared, coeff_3, coeff_2);
    __m512d const poly_01 = _mm512_fmadd_pd(angles_squared, coeff_1, coeff_0);
    __m512d const poly_0123 = _mm512_fmadd_pd(angles_quadratic, poly_23, poly_01);

    // Combine polynomial terms
    __m512d results = _mm512_fmadd_pd(angles_octic, poly_4567, poly_0123);
    results = _mm512_fmadd_pd(results, angles_squared, coeff_8);
    results = _mm512_fmadd_pd(results, angles_cubed, angles);
    return results;
}

NK_INTERNAL __m512d nk_f64x8_atan_skylake_(__m512d const inputs) {
    // Polynomial coefficients for atan approximation
    __m512d const coeff_19 = _mm512_set1_pd(-1.88796008463073496563746e-05);
    __m512d const coeff_18 = _mm512_set1_pd(+0.000209850076645816976906797);
    __m512d const coeff_17 = _mm512_set1_pd(-0.00110611831486672482563471);
    __m512d const coeff_16 = _mm512_set1_pd(+0.00370026744188713119232403);
    __m512d const coeff_15 = _mm512_set1_pd(-0.00889896195887655491740809);
    __m512d const coeff_14 = _mm512_set1_pd(+0.016599329773529201970117);
    __m512d const coeff_13 = _mm512_set1_pd(-0.0254517624932312641616861);
    __m512d const coeff_12 = _mm512_set1_pd(+0.0337852580001353069993897);
    __m512d const coeff_11 = _mm512_set1_pd(-0.0407629191276836500001934);
    __m512d const coeff_10 = _mm512_set1_pd(+0.0466667150077840625632675);
    __m512d const coeff_9 = _mm512_set1_pd(-0.0523674852303482457616113);
    __m512d const coeff_8 = _mm512_set1_pd(+0.0587666392926673580854313);
    __m512d const coeff_7 = _mm512_set1_pd(-0.0666573579361080525984562);
    __m512d const coeff_6 = _mm512_set1_pd(+0.0769219538311769618355029);
    __m512d const coeff_5 = _mm512_set1_pd(-0.090908995008245008229153);
    __m512d const coeff_4 = _mm512_set1_pd(+0.111111105648261418443745);
    __m512d const coeff_3 = _mm512_set1_pd(-0.14285714266771329383765);
    __m512d const coeff_2 = _mm512_set1_pd(+0.199999999996591265594148);
    __m512d const coeff_1 = _mm512_set1_pd(-0.333333333333311110369124);

    // Quadrant adjustments
    __mmask8 negative_mask = _mm512_cmp_pd_mask(inputs, _mm512_setzero_pd(), _CMP_LT_OS);
    __m512d values = _mm512_abs_pd(inputs);
    __mmask8 reciprocal_mask = _mm512_cmp_pd_mask(values, _mm512_set1_pd(1.0), _CMP_GT_OS);
    values = _mm512_mask_div_pd(values, reciprocal_mask, _mm512_set1_pd(1.0), values);
    __m512d const values_squared = _mm512_mul_pd(values, values);
    __m512d const values_cubed = _mm512_mul_pd(values, values_squared);

    // Polynomial evaluation (argument reduction and approximation)
    __m512d polynomials = coeff_19;
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_18);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_17);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_16);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_15);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_14);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_13);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_12);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_11);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_10);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_9);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_8);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_7);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_6);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_5);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_4);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_3);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_2);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_1);

    // Compute atan approximation
    __m512d result = _mm512_fmadd_pd(values_cubed, polynomials, values);
    result = _mm512_mask_sub_pd(result, reciprocal_mask, _mm512_set1_pd(1.5707963267948966), result);
    result = _mm512_mask_sub_pd(result, negative_mask, _mm512_setzero_pd(), result);
    return result;
}

/**
 *  @brief AVX-512 implementation of atan2(y, x) for 8 double-precision values.
 *  @see Based on the f32x16 version with appropriate precision constants.
 */
NK_INTERNAL __m512d nk_f64x8_atan2_skylake_(__m512d const ys_inputs, __m512d const xs_inputs) {
    // Polynomial coefficients for atan approximation (higher precision than f32)
    __m512d const coeff_19 = _mm512_set1_pd(-1.88796008463073496563746e-05);
    __m512d const coeff_18 = _mm512_set1_pd(+0.000209850076645816976906797);
    __m512d const coeff_17 = _mm512_set1_pd(-0.00110611831486672482563471);
    __m512d const coeff_16 = _mm512_set1_pd(+0.00370026744188713119232403);
    __m512d const coeff_15 = _mm512_set1_pd(-0.00889896195887655491740809);
    __m512d const coeff_14 = _mm512_set1_pd(+0.016599329773529201970117);
    __m512d const coeff_13 = _mm512_set1_pd(-0.0254517624932312641616861);
    __m512d const coeff_12 = _mm512_set1_pd(+0.0337852580001353069993897);
    __m512d const coeff_11 = _mm512_set1_pd(-0.0407629191276836500001934);
    __m512d const coeff_10 = _mm512_set1_pd(+0.0466667150077840625632675);
    __m512d const coeff_9 = _mm512_set1_pd(-0.0523674852303482457616113);
    __m512d const coeff_8 = _mm512_set1_pd(+0.0587666392926673580854313);
    __m512d const coeff_7 = _mm512_set1_pd(-0.0666573579361080525984562);
    __m512d const coeff_6 = _mm512_set1_pd(+0.0769219538311769618355029);
    __m512d const coeff_5 = _mm512_set1_pd(-0.090908995008245008229153);
    __m512d const coeff_4 = _mm512_set1_pd(+0.111111105648261418443745);
    __m512d const coeff_3 = _mm512_set1_pd(-0.14285714266771329383765);
    __m512d const coeff_2 = _mm512_set1_pd(+0.199999999996591265594148);
    __m512d const coeff_1 = _mm512_set1_pd(-0.333333333333311110369124);

    // Quadrant adjustments normalizing to absolute values of x and y
    __mmask8 const xs_negative_mask = _mm512_cmp_pd_mask(xs_inputs, _mm512_setzero_pd(), _CMP_LT_OS);
    __m512d xs = _mm512_abs_pd(xs_inputs);
    __m512d ys = _mm512_abs_pd(ys_inputs);
    // Ensure proper fraction where the numerator is smaller than the denominator
    __mmask8 const swap_mask = _mm512_cmp_pd_mask(ys, xs, _CMP_GT_OS);
    __m512d temps = xs;
    xs = _mm512_mask_blend_pd(swap_mask, xs, ys);
    ys = _mm512_mask_sub_pd(ys, swap_mask, _mm512_setzero_pd(), temps);

    // Compute ratio and ratio^2
    __m512d const ratio = _mm512_div_pd(ys, xs);
    __m512d const ratio_squared = _mm512_mul_pd(ratio, ratio);
    __m512d const ratio_cubed = _mm512_mul_pd(ratio, ratio_squared);

    // Polynomial evaluation
    __m512d polynomials = coeff_19;
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_18);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_17);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_16);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_15);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_14);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_13);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_12);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_11);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_10);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_9);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_8);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_7);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_6);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_5);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_4);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_3);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_2);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_1);

    // Compute the result with quadrant adjustments
    __m512d results = _mm512_fmadd_pd(ratio_cubed, polynomials, ratio);
    results = _mm512_mask_sub_pd(results, xs_negative_mask, results, _mm512_set1_pd(3.14159265358979323846));
    results = _mm512_mask_add_pd(results, swap_mask, results, _mm512_set1_pd(1.5707963267948966));

    // Adjust sign based on original x and y signs (matching scalar atan2 behavior)
    __m512d const y_sign = _mm512_and_pd(ys_inputs, _mm512_set1_pd(-0.0));
    results = _mm512_mask_xor_pd(results, xs_negative_mask, results, _mm512_set1_pd(-0.0));
    results = _mm512_xor_pd(results, y_sign);

    return results;
}

NK_PUBLIC void nk_sin_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d angles = _mm512_loadu_pd(ins + i);
        __m512d results = nk_f64x8_sin_skylake_(angles);
        _mm512_storeu_pd(outs + i, results);
    }
    if (i < n) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFF, n - i);
        __m512d angles = _mm512_maskz_loadu_pd(mask, ins + i);
        __m512d results = nk_f64x8_sin_skylake_(angles);
        _mm512_mask_storeu_pd(outs + i, mask, results);
    }
}
NK_PUBLIC void nk_cos_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d angles = _mm512_loadu_pd(ins + i);
        __m512d results = nk_f64x8_cos_skylake_(angles);
        _mm512_storeu_pd(outs + i, results);
    }
    if (i < n) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFF, n - i);
        __m512d angles = _mm512_maskz_loadu_pd(mask, ins + i);
        __m512d results = nk_f64x8_cos_skylake_(angles);
        _mm512_mask_storeu_pd(outs + i, mask, results);
    }
}
NK_PUBLIC void nk_atan_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d angles = _mm512_loadu_pd(ins + i);
        __m512d results = nk_f64x8_atan_skylake_(angles);
        _mm512_storeu_pd(outs + i, results);
    }
    if (i < n) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFF, n - i);
        __m512d angles = _mm512_maskz_loadu_pd(mask, ins + i);
        __m512d results = nk_f64x8_atan_skylake_(angles);
        _mm512_mask_storeu_pd(outs + i, mask, results);
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_

#endif // NK_TRIGONOMETRY_SKYLAKE_H
