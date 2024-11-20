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
 *  @brief  Computes an approximate sine of the given angle in radians with 3 ULP error bound.
 *  @see    xfastsinf_u3500 in SLEEF library
 *  @param  angle The input angle in radians.
 *  @return The approximate sine of the input angle.
 */
SIMSIMD_PUBLIC simsimd_f32_t simsimd_f32_sin(simsimd_f32_t angle) {
    // Variables
    int multiple_of_pi;          // The integer multiple of π in the input angle
    simsimd_f32_t result;        // The final result of the sine computation
    simsimd_f32_t angle_squared; // Square of the reduced angle

    // Constants for argument reduction
    simsimd_f32_t const pi_reciprocal = 0.31830988618379067154f; // 1/π
    simsimd_f32_t const pi = 3.14159265358979323846f;            // π

    // Compute multiple_of_pi = round(angle / π)
    simsimd_f32_t quotient = angle * pi_reciprocal;
    if (quotient >= 0.0f) { multiple_of_pi = (int)(quotient + 0.5f); }
    else { multiple_of_pi = (int)(quotient - 0.5f); }

    // Reduce the angle: angle = angle - (multiple_of_pi * π)
    angle = angle - multiple_of_pi * pi;

    // Compute the square of the reduced angle
    angle_squared = angle * angle;

    // Polynomial coefficients for sine approximation (minimax polynomial)
    simsimd_f32_t const coeff_5 = -0.0001881748176f; // Coefficient for x^5 term
    simsimd_f32_t const coeff_3 = 0.008323502727f;   // Coefficient for x^3 term
    simsimd_f32_t const coeff_1 = -0.1666651368f;    // Coefficient for x term

    // Compute the polynomial approximation
    simsimd_f32_t polynomial = coeff_5;
    polynomial = polynomial * angle_squared + coeff_3; // polynomial = (coeff_5 * x^2) + coeff_3
    polynomial = polynomial * angle_squared + coeff_1; // polynomial = polynomial * x^2 + coeff_1

    // Compute the final result: sine approximation
    result = ((angle_squared * angle) * polynomial) + angle; // result = (x^3 * polynomial) + x

    // If multiple_of_pi is odd, flip the sign of the result
    if ((multiple_of_pi & 1) != 0) { result = -result; }

    return result;
}

/**
 *  @brief  Computes an approximate cosine of the given angle in radians with 3 ULP error bound.
 *  @see    xfastcosf_u3500 in SLEEF library
 *  @param  angle The input angle in radians.
 *  @return The approximate cosine of the input angle.
 */
SIMSIMD_PUBLIC simsimd_f32_t simsimd_f32_cos(simsimd_f32_t angle) {
    // Variables
    int multiple_of_pi;          // The integer multiple of π in the input angle
    simsimd_f32_t result;        // The final result of the cosine computation
    simsimd_f32_t angle_squared; // Square of the reduced angle
    simsimd_f32_t reduced_angle; // The angle reduced to the primary interval

    // Constants for argument reduction
    simsimd_f32_t const pi_reciprocal = 0.31830988618379067154f; // 1/π
    simsimd_f32_t const pi = 3.14159265358979323846f;            // π

    // Compute multiple_of_pi = round(angle * (1/π) - 0.5)
    simsimd_f32_t quotient = angle * pi_reciprocal - 0.5f;
    if (quotient >= 0.0f) { multiple_of_pi = (int)(quotient + 0.5f); }
    else { multiple_of_pi = (int)(quotient - 0.5f); }

    // Reduce the angle: angle = angle - (multiple_of_pi + 0.5) * π
    reduced_angle = angle - (multiple_of_pi + 0.5f) * pi;

    // Compute the square of the reduced angle
    angle_squared = reduced_angle * reduced_angle;

    // Polynomial coefficients for cosine approximation (minimax polynomial)
    simsimd_f32_t const coeff_5 = -0.0001881748176f; // Coefficient for x^5 term
    simsimd_f32_t const coeff_3 = 0.008323502727f;   // Coefficient for x^3 term
    simsimd_f32_t const coeff_1 = -0.1666651368f;    // Coefficient for x^1 term

    // Compute the polynomial approximation
    simsimd_f32_t polynomial = coeff_5;
    polynomial = polynomial * angle_squared + coeff_3; // polynomial = (coeff_5 * x^2) + coeff_3
    polynomial = polynomial * angle_squared + coeff_1; // polynomial = polynomial * x^2 + coeff_1

    // Compute the final result: cosine approximation
    result = ((angle_squared * reduced_angle) * polynomial) + reduced_angle; // result = (x^3 * polynomial) + x

    // If multiple_of_pi is even, flip the sign of the result
    if ((multiple_of_pi & 1) == 0) { result = -result; }

    return result;
}

/**
 *  @brief  Computes an approximate cosine of the given angle in radians with 0 ULP error bound.
 *  @see    `xsin` in SLEEF library
 *  @param  angle The input angle in radians.
 *  @return The approximate cosine of the input angle.
 */
SIMSIMD_PUBLIC simsimd_f64_t simsimd_f64_sin(simsimd_f64_t angle) {
    // Constants for bit manipulation
    simsimd_i64_t const negative_zero = 0x8000000000000000LL; // Hexadecimal value of -0.0 in IEEE 754

    // Union for bit manipulation between simsimd_f64_t and simsimd_i64_t
    union {
        simsimd_f64_t f64;
        simsimd_i64_t i64;
    } converter;

    // Preserve the original angle for special case handling (negative zero)
    simsimd_f64_t original_angle = angle;

    // Constants for argument reduction
    simsimd_f64_t const pi_reciprocal = 0.318309886183790671537767526745028724; // 1/π
    simsimd_f64_t const pi_high = 3.141592653589793116;                         // High-precision part of π
    simsimd_f64_t const pi_low = 1.2246467991473532072e-16;                     // Low-precision part of π

    // Compute multiple_of_pi = round(angle / π)
    simsimd_f64_t quotient = angle * pi_reciprocal;
    int multiple_of_pi = (int)(quotient < 0 ? quotient - 0.5 : quotient + 0.5);

    // Reduce the angle: angle = angle - (multiple_of_pi * π)
    angle = (multiple_of_pi * -pi_high) + angle;
    angle = (multiple_of_pi * -pi_low) + angle;

    // Compute the square of the reduced argument
    simsimd_f64_t argument_square = angle * angle;

    // Adjust the sign of the angle if multiple_of_pi is odd
    if ((multiple_of_pi & 1) != 0) { angle = -angle; }

    // Compute higher powers of the argument
    simsimd_f64_t argument_power_4 = argument_square * argument_square;   // angle^4
    simsimd_f64_t argument_power_8 = argument_power_4 * argument_power_4; // angle^8

    // Polynomial coefficients for sine approximation (minimax polynomial)
    simsimd_f64_t const coeff_0 = 0.00833333333333332974823815;
    simsimd_f64_t const coeff_1 = -0.000198412698412696162806809;
    simsimd_f64_t const coeff_2 = 2.75573192239198747630416e-06;
    simsimd_f64_t const coeff_3 = -2.50521083763502045810755e-08;
    simsimd_f64_t const coeff_4 = 1.60590430605664501629054e-10;
    simsimd_f64_t const coeff_5 = -7.64712219118158833288484e-13;
    simsimd_f64_t const coeff_6 = 2.81009972710863200091251e-15;
    simsimd_f64_t const coeff_7 = -7.97255955009037868891952e-18;
    simsimd_f64_t const coeff_8 = -0.166666666666666657414808;

    // Compute higher-degree polynomial terms
    simsimd_f64_t temp1 = (argument_square * coeff_7) + coeff_6;  // temp1 = s * c7 + c6
    simsimd_f64_t temp2 = (argument_square * coeff_5) + coeff_4;  // temp2 = s * c5 + c4
    simsimd_f64_t poly_high = (argument_power_4 * temp1) + temp2; // poly_high = s^4 * temp1 + temp2

    // Compute lower-degree polynomial terms
    simsimd_f64_t temp3 = (argument_square * coeff_3) + coeff_2; // temp3 = s * c3 + c2
    simsimd_f64_t temp4 = (argument_square * coeff_1) + coeff_0; // temp4 = s * c1 + c0
    simsimd_f64_t poly_low = (argument_power_4 * temp3) + temp4; // poly_low = s^4 * temp3 + temp4

    // Combine polynomial terms
    simsimd_f64_t result = (argument_power_8 * poly_high) + poly_low; // result = s^8 * poly_high + poly_low
    result = (result * argument_square) + coeff_8;                    // result = result * s + c8
    result = (argument_square * (result * angle)) + angle;            // result = s * (result * angle) + angle

    // Handle the special case of negative zero input
    converter.f64 = original_angle;
    if (converter.i64 == negative_zero) { result = original_angle; }

    return result;
}

/**
 *  @brief  Computes an approximate cosine of the given angle in radians with 0 ULP error bound.
 *  @see    `xcos` in SLEEF library
 *  @param  angle The input angle in radians.
 *  @return The approximate cosine of the input angle.
 */
SIMSIMD_PUBLIC simsimd_f64_t simsimd_f64_cos(simsimd_f64_t angle) {
    // Constants for bit manipulation
    simsimd_i64_t const negative_zero = 0x8000000000000000LL; // Hexadecimal value of -0.0 in IEEE 754

    // Union for bit manipulation between simsimd_f64_t and simsimd_i64_t
    union {
        simsimd_f64_t f64;
        simsimd_i64_t i64;
    } converter;

    // Variables
    simsimd_f64_t result;                 // The final result of the cosine computation
    simsimd_f64_t angle_squared;          // Square of the reduced angle
    simsimd_f64_t original_angle = angle; // Preserve the original angle for special case handling
    int multiple_of_pi;                   // The integer multiple of π in the input angle

    // Constants for argument reduction
    simsimd_f64_t const pi_reciprocal = 0.318309886183790671537767526745028724; // 1/π
    simsimd_f64_t const pi_high = 3.141592653589793116;                         // High-precision part of π
    simsimd_f64_t const pi_low = 1.2246467991473532072e-16;                     // Low-precision part of π

    // Compute multiple_of_pi = 2 * round(angle * (1/π) - 0.5) + 1
    simsimd_f64_t quotient = angle * pi_reciprocal - 0.5;
    int temp = (int)(quotient < 0 ? quotient - 0.5 : quotient + 0.5);
    multiple_of_pi = 2 * temp + 1;

    // Reduce the angle: angle = angle - multiple_of_pi * (π / 2)
    angle = angle - multiple_of_pi * (pi_high * 0.5);
    angle = angle - multiple_of_pi * (pi_low * 0.5);

    // Compute the square of the reduced angle
    angle_squared = angle * angle;

    // Adjust the sign of the angle if necessary
    if ((multiple_of_pi & 2) == 0) { angle = -angle; }

    // Compute higher powers of the argument
    simsimd_f64_t angle_power_2 = angle_squared;
    simsimd_f64_t angle_power_4 = angle_power_2 * angle_power_2; // angle^4
    simsimd_f64_t angle_power_8 = angle_power_4 * angle_power_4; // angle^8

    // Polynomial coefficients for cosine approximation (minimax polynomial)
    simsimd_f64_t const coeff_0 = 0.00833333333333332974823815;
    simsimd_f64_t const coeff_1 = -0.000198412698412696162806809;
    simsimd_f64_t const coeff_2 = 2.75573192239198747630416e-06;
    simsimd_f64_t const coeff_3 = -2.50521083763502045810755e-08;
    simsimd_f64_t const coeff_4 = 1.60590430605664501629054e-10;
    simsimd_f64_t const coeff_5 = -7.64712219118158833288484e-13;
    simsimd_f64_t const coeff_6 = 2.81009972710863200091251e-15;
    simsimd_f64_t const coeff_7 = -7.97255955009037868891952e-18;
    simsimd_f64_t const coeff_8 = -0.166666666666666657414808;

    // Compute higher-degree polynomial terms
    simsimd_f64_t temp1 = (angle_squared * coeff_7) + coeff_6; // temp1 = s * c7 + c6
    simsimd_f64_t temp2 = (angle_squared * coeff_5) + coeff_4; // temp2 = s * c5 + c4
    simsimd_f64_t poly_high = (angle_power_4 * temp1) + temp2; // poly_high = s^4 * temp1 + temp2

    // Compute lower-degree polynomial terms
    simsimd_f64_t temp3 = (angle_squared * coeff_3) + coeff_2; // temp3 = s * c3 + c2
    simsimd_f64_t temp4 = (angle_squared * coeff_1) + coeff_0; // temp4 = s * c1 + c0
    simsimd_f64_t poly_low = (angle_power_4 * temp3) + temp4;  // poly_low = s^4 * temp3 + temp4

    // Combine polynomial terms
    result = (angle_power_8 * poly_high) + poly_low;     // result = s^8 * poly_high + poly_low
    result = (result * angle_squared) + coeff_8;         // result = result * s + c8
    result = (angle_squared * (result * angle)) + angle; // result = s * (result * angle) + angle

    // Return the final result
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
