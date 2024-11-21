/**
 *  @file       trigonometry.h
 *  @brief      SIMD-accelerated trigonometric element-wise oeprations.
 *  @author     Ash Vardanian
 *  @date       July 1, 2023
 *
 *  Contains:
 *  - Sine and Cosine approximations: fast for `f32` vs accurate for `f64`
 *
 *  For datatypes:
 *  - 32-bit IEEE-754 floating point
 *  - 64-bit IEEE-754 floating point
 *
 *  For hardware architectures:
 *  - Arm: NEON
 *  - x86: Haswell, Skylake
 *
 *  Those functions partially complement the `elementwise.h` module, and are necessary for
 *  the `geospatial.h` module, among others. Both Haversine and Vincenty's formulas require
 *  trigonometric functions, and those are the most expensive part of the computation.
 *
 *  @section    GLibC IEEE-754-compliant Math Functions
 *
 *  The GNU C Library (GLibC) provides a set of IEEE-754-compliant math functions, like `sinf`, `cosf`,
 *  and double-precsion variants `sin`, `cos`. Those functions are accurate to ~0.55 ULP (units in the
 *  last place), but can be slow to evaluate. They use a combination of techniques, like:
 *
 *  - Taylor series expansions for small values.
 *  - Table lookups combined with corrections for moderate values.
 *  - Accurate modulo reduction for large values.
 *
 *  The precomputed tables may be the hardest part to accelerate with SIMD, as they contain 440x values,
 *  each 64-bit wide.
 *
 *  https://github.com/lattera/glibc/blob/895ef79e04a953cac1493863bcae29ad85657ee1/sysdeps/ieee754/dbl-64/branred.c#L54
 *  https://github.com/lattera/glibc/blob/895ef79e04a953cac1493863bcae29ad85657ee1/sysdeps/ieee754/dbl-64/s_sin.c#L84
 *
 *  @section    Approximation Algorithms
 *
 *  There are several ways to approximate trigonometric functions, and the choice depends on the
 *  target hardware and the desired precision. Notably:
 *
 *  - Taylor Series approximation is a series expansion of a sum of its derivatives at a target point.
 *    It's easy to derive for differentiable functions, works well for functions smooth around the
 *    expsansion point, but can perform poorly for functions with singularities or high-frequency
 *    oscillations.
 *
 *  - Pade approximations are rational functions that approximate a function by a ratio of polynomials.
 *    It often converges faster than Taylor for functions with singularities or steep changes, provides
 *    good approximations for both smooth and rational functions, but can be more computationally
 *    intensive to evaluate, and can have holes (undefined points).
 *
 *  Moreover, most approximations can be combined with Horner's methods of evaluating polynomials
 *  to reduce the number of multiplications and additions, and to improve the numerical stability.
 *  In trigonometry, the Payne-Hanek Range Reduction is another technique used to reduce the argument
 *  to a smaller range, where the approximation is more accurate.
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef SIMSIMD_TRIGONOMETRY_H
#define SIMSIMD_TRIGONOMETRY_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  Serial backends for all numeric types.
 *  By default they use 32-bit arithmetic, unless the arguments themselves contain 64-bit floats.
 *  For double-precision computation check out the "*_accurate" variants of those "*_serial" functions.
 */
SIMSIMD_PUBLIC void simsimd_sin_f64_serial(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t const *outs);
SIMSIMD_PUBLIC void simsimd_cos_f64_serial(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t const *outs);
SIMSIMD_PUBLIC void simsimd_sin_f32_serial(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t const *outs);
SIMSIMD_PUBLIC void simsimd_cos_f32_serial(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t const *outs);

SIMSIMD_PUBLIC void simsimd_sin_f64_neon(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t const *outs);
SIMSIMD_PUBLIC void simsimd_cos_f64_neon(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t const *outs);
SIMSIMD_PUBLIC void simsimd_sin_f32_neon(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t const *outs);
SIMSIMD_PUBLIC void simsimd_cos_f32_neon(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t const *outs);

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic over 256-bit words.
 *  First demonstrated in 2011, at least one Haswell-based processor was still being sold in 2022 — the Pentium G3420.
 *  Practically all modern x86 CPUs support AVX2, FMA, and F16C, making it a perfect baseline for SIMD algorithms.
 *  On other hand, there is no need to implement AVX2 versions of `f32` and `f64` functions, as those are
 *  properly vectorized by recent compilers.
 */
SIMSIMD_PUBLIC void simsimd_sin_f64_haswell(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t const *outs);
SIMSIMD_PUBLIC void simsimd_cos_f64_haswell(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t const *outs);
SIMSIMD_PUBLIC void simsimd_sin_f32_haswell(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t const *outs);
SIMSIMD_PUBLIC void simsimd_cos_f32_haswell(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t const *outs);

/*  SIMD-powered backends for various generations of AVX512 CPUs.
 *  Skylake is handy, as it supports masked loads and other operations, avoiding the need for the tail loop.
 */
SIMSIMD_PUBLIC void simsimd_sin_f64_skylake(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t const *outs);
SIMSIMD_PUBLIC void simsimd_cos_f64_skylake(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t const *outs);
SIMSIMD_PUBLIC void simsimd_sin_f32_skylake(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t const *outs);
SIMSIMD_PUBLIC void simsimd_cos_f32_skylake(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t const *outs);

/**
 *  @brief  Computes an approximate sine of the given angle in radians with @b 3-ULP error bound for [-2π, 2π].
 *  @see    Based on @b `xfastsinf_u3500` in SLEEF library.
 *  @param  angle The input angle in radians.
 *  @return The approximate sine of the input angle.
 */
SIMSIMD_PUBLIC simsimd_f32_t simsimd_f32_sin(simsimd_f32_t const angle_radians) {

    // Constants for argument reduction
    simsimd_f32_t const pi = 3.14159265358979323846f;            /// π
    simsimd_f32_t const pi_reciprocal = 0.31830988618379067154f; /// 1/π

    // Polynomial coefficients for sine/cosine approximation (minimax polynomial)
    simsimd_f32_t const coeff_5 = -0.0001881748176f; /// Coefficient for x^5 term
    simsimd_f32_t const coeff_3 = +0.008323502727f;  /// Coefficient for x^3 term
    simsimd_f32_t const coeff_1 = -0.1666651368f;    /// Coefficient for x term

    // Compute (multiple_of_pi) = round(angle / π)
    simsimd_f32_t const quotient = angle_radians * pi_reciprocal;
    int const multiple_of_pi = (int)(quotient < 0 ? quotient - 0.5f : quotient + 0.5f);

    // Reduce the angle to (angle - (multiple_of_pi * π)) in [0, π]
    simsimd_f32_t const angle = angle_radians - multiple_of_pi * pi;
    simsimd_f32_t const angle_squared = angle * angle;
    simsimd_f32_t const angle_cubed = angle * angle_squared;

    // Compute the polynomial approximation
    simsimd_f32_t polynomial = coeff_5;
    polynomial = polynomial * angle_squared + coeff_3;         // polynomial = (coeff_5 * x^2) + coeff_3
    polynomial = polynomial * angle_squared + coeff_1;         // polynomial = polynomial * x^2 + coeff_1
    simsimd_f32_t result = (angle_cubed * polynomial) + angle; // result = (x^3 * polynomial) + x

    // If multiple_of_pi is odd, flip the sign of the result
    if ((multiple_of_pi & 1) != 0) result = -result;
    return result;
}

/**
 *  @brief  Computes an approximate cosine of the given angle in radians with @b 3-ULP error bound for [-2π, 2π].
 *  @see    Based on @b `xfastcosf_u3500` in SLEEF library.
 *  @param  angle The input angle in radians.
 *  @return The approximate cosine of the input angle.
 */
SIMSIMD_PUBLIC simsimd_f32_t simsimd_f32_cos(simsimd_f32_t const angle_radians) {

    // Constants for argument reduction
    simsimd_f32_t const pi = 3.14159265358979323846f;            /// π
    simsimd_f32_t const pi_half = 1.57079632679489661923f;       /// π/2
    simsimd_f32_t const pi_reciprocal = 0.31830988618379067154f; /// 1/π

    // Polynomial coefficients for sine/cosine approximation (minimax polynomial)
    simsimd_f32_t const coeff_5 = -0.0001881748176f; /// Coefficient for x^5 term
    simsimd_f32_t const coeff_3 = +0.008323502727f;  /// Coefficient for x^3 term
    simsimd_f32_t const coeff_1 = -0.1666651368f;    /// Coefficient for x term

    // Compute (multiple_of_pi) = round(angle / π - 0.5)
    simsimd_f32_t const quotient = angle_radians * pi_reciprocal - 0.5f;
    int const multiple_of_pi = (int)(quotient < 0 ? quotient - 0.5f : quotient + 0.5f);

    // Reduce the angle to (angle - (multiple_of_pi * π)) in [-π/2, π/2]
    simsimd_f32_t const angle = angle_radians - pi_half - multiple_of_pi * pi;
    simsimd_f32_t const angle_squared = angle * angle;
    simsimd_f32_t const angle_cubed = angle * angle_squared;

    // Compute the polynomial approximation
    simsimd_f32_t polynomial = coeff_5;
    polynomial = polynomial * angle_squared + coeff_3;         // polynomial = (coeff_5 * x^2) + coeff_3
    polynomial = polynomial * angle_squared + coeff_1;         // polynomial = polynomial * x^2 + coeff_1
    simsimd_f32_t result = (angle_cubed * polynomial) + angle; // result = (x^3 * polynomial) + x

    // If multiple_of_pi is even, flip the sign of the result
    if ((multiple_of_pi & 1) == 0) result = -result;
    return result;
}

/**
 *  @brief  Computes an approximate cosine of the given angle in radians with @b 0-ULP error bound in [-2π, 2π].
 *  @see    Based on @b `xsin` in SLEEF library.
 *  @param  angle The input angle in radians.
 *  @return The approximate cosine of the input angle.
 */
SIMSIMD_PUBLIC simsimd_f64_t simsimd_f64_sin(simsimd_f64_t const angle_radians) {

    // Constants for argument reduction
    simsimd_f64_t const pi_high = 3.141592653589793116;                         // High-digits part of π
    simsimd_f64_t const pi_low = 1.2246467991473532072e-16;                     // Low-digits part of π
    simsimd_f64_t const pi_reciprocal = 0.318309886183790671537767526745028724; // 1/π
    simsimd_i64_t const negative_zero = 0x8000000000000000LL;                   // Hexadecimal value of -0.0 in IEEE 754

    // Polynomial coefficients for sine/cosine approximation (minimax polynomial)
    simsimd_f64_t const coeff_0 = +0.00833333333333332974823815;
    simsimd_f64_t const coeff_1 = -0.000198412698412696162806809;
    simsimd_f64_t const coeff_2 = +2.75573192239198747630416e-06;
    simsimd_f64_t const coeff_3 = -2.50521083763502045810755e-08;
    simsimd_f64_t const coeff_4 = +1.60590430605664501629054e-10;
    simsimd_f64_t const coeff_5 = -7.64712219118158833288484e-13;
    simsimd_f64_t const coeff_6 = +2.81009972710863200091251e-15;
    simsimd_f64_t const coeff_7 = -7.97255955009037868891952e-18;
    simsimd_f64_t const coeff_8 = -0.166666666666666657414808;

    // Compute (multiple_of_pi) = round(angle / π)
    simsimd_f64_t const quotient = angle_radians * pi_reciprocal;
    int const multiple_of_pi = (int)(quotient < 0 ? quotient - 0.5 : quotient + 0.5);

    // Reduce the angle to (angle - (multiple_of_pi * π)) in [0, π]
    simsimd_f64_t angle = angle_radians;
    angle = angle - (multiple_of_pi * pi_high);
    angle = angle - (multiple_of_pi * pi_low);
    if ((multiple_of_pi & 1) != 0) angle = -angle;
    simsimd_f64_t const angle_squared = angle * angle;
    simsimd_f64_t const angle_quartic = angle_squared * angle_squared;
    simsimd_f64_t const angle_octic = angle_quartic * angle_quartic;

    // Compute higher-degree polynomial terms
    simsimd_f64_t poly_67 = (angle_squared * coeff_7) + coeff_6;   // poly_67 = s * c7 + c6
    simsimd_f64_t poly_45 = (angle_squared * coeff_5) + coeff_4;   // poly_45 = s * c5 + c4
    simsimd_f64_t poly_4567 = (angle_quartic * poly_67) + poly_45; // poly_4567 = s^4 * poly_67 + poly_45

    // Compute lower-degree polynomial terms
    simsimd_f64_t poly_23 = (angle_squared * coeff_3) + coeff_2;   // poly_23 = s * c3 + c2
    simsimd_f64_t poly_01 = (angle_squared * coeff_1) + coeff_0;   // poly_01 = s * c1 + c0
    simsimd_f64_t poly_0123 = (angle_quartic * poly_23) + poly_01; // poly_0123 = s^4 * poly_23 + poly_01

    // Combine polynomial terms
    simsimd_f64_t result = (angle_octic * poly_4567) + poly_0123; // result = s^8 * poly_4567 + poly_0123
    result = (result * angle_squared) + coeff_8;                  // result = result * s + c8
    result = (angle_squared * (result * angle)) + angle;          // result = s * (result * angle) + angle

    // Handle the special case of negative zero input
    union {
        simsimd_f64_t f64;
        simsimd_i64_t i64;
    } converter;
    converter.f64 = angle;
    if (converter.i64 == negative_zero) result = angle;
    return result;
}

/**
 *  @brief  Computes an approximate cosine of the given angle in radians with @b 0-ULP error bound in [-2π, 2π].
 *  @see    Based on @b `xcos` in SLEEF library.
 *  @param  angle The input angle in radians.
 *  @return The approximate cosine of the input angle.
 */
SIMSIMD_PUBLIC simsimd_f64_t simsimd_f64_cos(simsimd_f64_t const angle_radians) {

    // Constants for argument reduction
    simsimd_f64_t const pi_high = 3.141592653589793116;                         // High-digits part of π
    simsimd_f64_t const pi_low = 1.2246467991473532072e-16;                     // Low-digits part of π
    simsimd_f64_t const pi_reciprocal = 0.318309886183790671537767526745028724; // 1/π

    // Polynomial coefficients for sine/cosine approximation (minimax polynomial)
    simsimd_f64_t const coeff_0 = +0.00833333333333332974823815;
    simsimd_f64_t const coeff_1 = -0.000198412698412696162806809;
    simsimd_f64_t const coeff_2 = +2.75573192239198747630416e-06;
    simsimd_f64_t const coeff_3 = -2.50521083763502045810755e-08;
    simsimd_f64_t const coeff_4 = +1.60590430605664501629054e-10;
    simsimd_f64_t const coeff_5 = -7.64712219118158833288484e-13;
    simsimd_f64_t const coeff_6 = +2.81009972710863200091251e-15;
    simsimd_f64_t const coeff_7 = -7.97255955009037868891952e-18;
    simsimd_f64_t const coeff_8 = -0.166666666666666657414808;

    // Compute (multiple_of_pi) = 2 * round(angle / π - 0.5) + 1
    simsimd_f64_t const quotient = angle_radians * pi_reciprocal - 0.5;
    int const multiple_of_pi = 2 * (int)(quotient < 0 ? quotient - 0.5 : quotient + 0.5) + 1;

    // Reduce the angle to (angle - (multiple_of_pi * π)) in [-π/2, π/2]
    simsimd_f64_t angle = angle_radians;
    angle = angle - (multiple_of_pi * pi_high * 0.5);
    angle = angle - (multiple_of_pi * pi_low * 0.5);
    if ((multiple_of_pi & 2) == 0) angle = -angle;
    simsimd_f64_t const angle_squared = angle * angle;
    simsimd_f64_t const angle_quartic = angle_squared * angle_squared;
    simsimd_f64_t const angle_octic = angle_quartic * angle_quartic;

    // Compute higher-degree polynomial terms
    simsimd_f64_t poly_67 = (angle_squared * coeff_7) + coeff_6;   // poly_67 = s * c7 + c6
    simsimd_f64_t poly_45 = (angle_squared * coeff_5) + coeff_4;   // poly_45 = s * c5 + c4
    simsimd_f64_t poly_4567 = (angle_quartic * poly_67) + poly_45; // poly_4567 = s^4 * poly_67 + poly_45

    // Compute lower-degree polynomial terms
    simsimd_f64_t poly_23 = (angle_squared * coeff_3) + coeff_2;   // poly_23 = s * c3 + c2
    simsimd_f64_t poly_01 = (angle_squared * coeff_1) + coeff_0;   // poly_01 = s * c1 + c0
    simsimd_f64_t poly_0123 = (angle_quartic * poly_23) + poly_01; // poly_0123 = s^4 * poly_23 + poly_01

    // Combine polynomial terms
    simsimd_f64_t result = (angle_octic * poly_4567) + poly_0123; // result = s^8 * poly_4567 + poly_0123
    result = (result * angle_squared) + coeff_8;                  // result = result * s + c8
    result = (angle_squared * (result * angle)) + angle;          // result = s * (result * angle) + angle
    return result;
}

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL
#endif // _SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif

#endif
