/**
 *  @file       trigonometry.h
 *  @brief      SIMD-accelerated trigonometric element-wise oeprations, based on SLEEF.
 *  @author     Ash Vardanian
 *  @date       July 1, 2023
 *
 *  Contains:
 *  - Sine and Cosine approximations: fast for `f32` vs accurate for `f64`
 *  - Tangent and the 2-argument arctangent: fast for `f32` vs accurate for `f64`
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
SIMSIMD_PUBLIC void simsimd_sin_f64_serial(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs);
SIMSIMD_PUBLIC void simsimd_cos_f64_serial(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs);
SIMSIMD_PUBLIC void simsimd_atan_f64_serial(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs);
SIMSIMD_PUBLIC void simsimd_sin_f32_serial(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs);
SIMSIMD_PUBLIC void simsimd_cos_f32_serial(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs);
SIMSIMD_PUBLIC void simsimd_atan_f32_serial(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs);

SIMSIMD_PUBLIC void simsimd_sin_f64_neon(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs);
SIMSIMD_PUBLIC void simsimd_cos_f64_neon(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs);
SIMSIMD_PUBLIC void simsimd_atan_f64_neon(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs);
SIMSIMD_PUBLIC void simsimd_sin_f32_neon(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs);
SIMSIMD_PUBLIC void simsimd_cos_f32_neon(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs);
SIMSIMD_PUBLIC void simsimd_atan_f32_neon(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs);

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic over 256-bit words.
 *  First demonstrated in 2011, at least one Haswell-based processor was still being sold in 2022 — the Pentium G3420.
 *  Practically all modern x86 CPUs support AVX2, FMA, and F16C, making it a perfect baseline for SIMD algorithms.
 *  On other hand, there is no need to implement AVX2 versions of `f32` and `f64` functions, as those are
 *  properly vectorized by recent compilers.
 */
SIMSIMD_PUBLIC void simsimd_sin_f64_haswell(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs);
SIMSIMD_PUBLIC void simsimd_cos_f64_haswell(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs);
SIMSIMD_PUBLIC void simsimd_atan_f64_haswell(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs);
SIMSIMD_PUBLIC void simsimd_sin_f32_haswell(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs);
SIMSIMD_PUBLIC void simsimd_cos_f32_haswell(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs);
SIMSIMD_PUBLIC void simsimd_atan_f32_haswell(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs);

/*  SIMD-powered backends for various generations of AVX512 CPUs.
 *  Skylake is handy, as it supports masked loads and other operations, avoiding the need for the tail loop.
 */
SIMSIMD_PUBLIC void simsimd_sin_f64_skylake(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs);
SIMSIMD_PUBLIC void simsimd_cos_f64_skylake(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs);
SIMSIMD_PUBLIC void simsimd_atan_f64_skylake(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs);
SIMSIMD_PUBLIC void simsimd_sin_f32_skylake(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs);
SIMSIMD_PUBLIC void simsimd_cos_f32_skylake(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs);
SIMSIMD_PUBLIC void simsimd_atan_f32_skylake(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs);

/**
 *  @brief  Computes an approximate sine of the given angle in radians with @b 3-ULP error bound for [-2π, 2π].
 *  @see    Based on @b `xfastsinf_u3500` in SLEEF library.
 *  @param  angle The input angle in radians.
 *  @return The approximate sine of the input angle in [-1, 1] range.
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
 *  @return The approximate cosine of the input angle in [-1, 1] range.
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
 *  @brief  Computes the arc-tangent of a value with @b 0-ULP error bound.
 *  @see    Based on @b `xatanf` in SLEEF library.
 *  @param  input The input value.
 *  @return The arc-tangent of the input value in [-π/2, π/2] radians range.
 */
SIMSIMD_PUBLIC simsimd_f32_t simsimd_f32_atan(simsimd_f32_t const input) {
    // Polynomial coefficients for atan approximation
    simsimd_f32_t const coeff_8 = -0.333331018686294555664062f;
    simsimd_f32_t const coeff_7 = +0.199926957488059997558594f;
    simsimd_f32_t const coeff_6 = -0.142027363181114196777344f;
    simsimd_f32_t const coeff_5 = +0.106347933411598205566406f;
    simsimd_f32_t const coeff_4 = -0.0748900920152664184570312f;
    simsimd_f32_t const coeff_3 = +0.0425049886107444763183594f;
    simsimd_f32_t const coeff_2 = -0.0159569028764963150024414f;
    simsimd_f32_t const coeff_1 = +0.00282363896258175373077393f;

    // Quadrant adjustment
    int quadrant = 0;
    simsimd_f32_t value = input;
    if (value < 0.0f) value = -value, quadrant |= 2;
    if (value > 1.0f) value = 1.0f / value, quadrant |= 1;

    // Argument reduction
    simsimd_f32_t const value_to_2nd = value * value;

    // Polynomial evaluation
    simsimd_f32_t polynomial_result = coeff_1;
    polynomial_result = polynomial_result * value_to_2nd + coeff_2;
    polynomial_result = polynomial_result * value_to_2nd + coeff_3;
    polynomial_result = polynomial_result * value_to_2nd + coeff_4;
    polynomial_result = polynomial_result * value_to_2nd + coeff_5;
    polynomial_result = polynomial_result * value_to_2nd + coeff_6;
    polynomial_result = polynomial_result * value_to_2nd + coeff_7;
    polynomial_result = polynomial_result * value_to_2nd + coeff_8;

    // Adjust for quadrant
    simsimd_f32_t result = value + value * value_to_2nd * polynomial_result;
    simsimd_f32_t const pi_half = 1.5707963267948966f; // π/2
    if ((quadrant & 1) != 0) result = pi_half - result;
    if ((quadrant & 2) != 0) result = -result;
    return result;
}

/**
 *  @brief  Computes the arc-tangent of (y/x) with @b 0-ULP error bound.
 *  @see    Based on @b `xatan2f` in SLEEF library.
 *  @param  y The input sine value.
 *  @param  x The input cosine value.
 *  @return The arc-tangent of (y/x) in [-π, π] radians range.
 */
SIMSIMD_PUBLIC simsimd_f32_t simsimd_f32_atan2(simsimd_f32_t const y, simsimd_f32_t const x) {
    // Polynomial coefficients for atan2 approximation
    simsimd_f32_t const coeff_8 = -0.333331018686294555664062f;
    simsimd_f32_t const coeff_7 = +0.199926957488059997558594f;
    simsimd_f32_t const coeff_6 = -0.142027363181114196777344f;
    simsimd_f32_t const coeff_5 = +0.106347933411598205566406f;
    simsimd_f32_t const coeff_4 = -0.0748900920152664184570312f;
    simsimd_f32_t const coeff_3 = +0.0425049886107444763183594f;
    simsimd_f32_t const coeff_2 = -0.0159569028764963150024414f;
    simsimd_f32_t const coeff_1 = +0.00282363896258175373077393f;

    simsimd_fui32_t x_bits, y_bits;
    x_bits.f = x, y_bits.f = y;
    simsimd_fui32_t x_abs, y_abs;
    y_abs.u = y_bits.u & 0x7FFFFFFFu;

    // Quadrant adjustment
    int quadrant = 0;
    if (x < 0.0f) { x_abs.f = -x, quadrant = -2; }
    else { x_abs.f = x; }
    // Ensure proper fraction where the numerator is smaller than the denominator
    if (y_abs.f > x_abs.f) {
        simsimd_f32_t temp = x_abs.f;
        x_abs.f = y_abs.f;
        y_abs.f = -temp;
        quadrant += 1;
    }

    // Argument reduction
    simsimd_f32_t const ratio = y_abs.f / x_abs.f;
    simsimd_f32_t const ratio_to_2nd = ratio * ratio;

    // Polynomial evaluation (fully unrolled)
    simsimd_f32_t polynomial_result = coeff_1;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_2;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_3;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_4;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_5;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_6;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_7;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_8;

    // Compute the result
    simsimd_f32_t result = ratio + ratio * ratio_to_2nd * polynomial_result;
    result += quadrant * (3.14159265358979323846f / 2.0f); // quadrant * (π/2)

    // Constants for special cases
    simsimd_u32_t const negative_zero = 0x80000000u;
    simsimd_u32_t const positive_infinity = 0x7F800000u;
    simsimd_u32_t const negative_infinity = 0xFF800000u;

    // Special cases handling using bit reinterpretation
    int const x_is_inf = (x_bits.u == positive_infinity) | (x_bits.u == negative_infinity);
    int const y_is_inf = (y_bits.u == positive_infinity) | (y_bits.u == negative_infinity);

    simsimd_fui32_t result_bits;
    result_bits.f = result;

    // Adjust sign based on x
    result_bits.u ^= x_bits.u & negative_zero;

    // Quadrant adjustments
    simsimd_f32_t const pi = 3.14159265358979323846f;     // π
    simsimd_f32_t const pi_half = 1.5707963267948966f;    // π/2
    simsimd_f32_t const pi_quarter = 0.7853981633974483f; // π/4

    if (x_is_inf || x_bits.f == 0.0f) {
        result_bits.f = pi_half - (x_is_inf ? ((x_bits.f < 0.0f) ? pi_half : 0.0f) : 0.0f);
    }
    if (y_is_inf) { result_bits.f = pi_half - (x_is_inf ? ((x_bits.f < 0.0f) ? pi_half : pi_quarter) : 0.0f); }
    if (y_bits.f == 0.0f) { result_bits.f = (x_bits.f < 0.0f) ? pi : 0.0f; }
    if (x_is_inf | y_is_inf) { result_bits.u = 0x7FC00000u; } // Set result to NaN
    // Adjust final result sign based on y
    else { result_bits.u ^= y_bits.u & negative_zero; }
    return result_bits.f;
}

/**
 *  @brief  Computes the sine of the given angle in radians with @b 0-ULP error bound in [-2π, 2π].
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
 *  @brief  Computes the cosine of the given angle in radians with @b 0-ULP error bound in [-2π, 2π].
 *  @see    Based on @b `xcos` in SLEEF library.
 *  @param  angle The input angle in radians.
 *  @return The approximate cosine of the input angle in [-1, 1] range.
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

/**
 *  @brief  Computes the arc-tangent of a value with @b 0-ULP error bound.
 *  @see    Based on @b `xatan` in SLEEF library.
 *  @param  input The input value.
 *  @return The arc-tangent of the input value in [-π/2, π/2] radians range.
 */
SIMSIMD_PUBLIC simsimd_f64_t simsimd_f64_atan(simsimd_f64_t const input) {
    // Polynomial coefficients for atan approximation
    simsimd_f64_t const coeff_19 = -1.88796008463073496563746e-05;
    simsimd_f64_t const coeff_18 = +0.000209850076645816976906797;
    simsimd_f64_t const coeff_17 = -0.00110611831486672482563471;
    simsimd_f64_t const coeff_16 = +0.00370026744188713119232403;
    simsimd_f64_t const coeff_15 = -0.00889896195887655491740809;
    simsimd_f64_t const coeff_14 = +0.016599329773529201970117;
    simsimd_f64_t const coeff_13 = -0.0254517624932312641616861;
    simsimd_f64_t const coeff_12 = +0.0337852580001353069993897;
    simsimd_f64_t const coeff_11 = -0.0407629191276836500001934;
    simsimd_f64_t const coeff_10 = +0.0466667150077840625632675;
    simsimd_f64_t const coeff_9 = -0.0523674852303482457616113;
    simsimd_f64_t const coeff_8 = +0.0587666392926673580854313;
    simsimd_f64_t const coeff_7 = -0.0666573579361080525984562;
    simsimd_f64_t const coeff_6 = +0.0769219538311769618355029;
    simsimd_f64_t const coeff_5 = -0.090908995008245008229153;
    simsimd_f64_t const coeff_4 = +0.111111105648261418443745;
    simsimd_f64_t const coeff_3 = -0.14285714266771329383765;
    simsimd_f64_t const coeff_2 = +0.199999999996591265594148;
    simsimd_f64_t const coeff_1 = -0.333333333333311110369124;

    // Quadrant adjustment
    int quadrant = 0;
    simsimd_f64_t value = input;
    if (value < 0) value = -value, quadrant |= 2;
    if (value > 1) value = 1.0 / value, quadrant |= 1;

    // Argument reduction
    simsimd_f64_t const value_to_2nd = value * value;
    simsimd_f64_t const value_to_4th = value_to_2nd * value_to_2nd;
    simsimd_f64_t const value_to_8th = value_to_4th * value_to_4th;
    simsimd_f64_t const value_to_16th = value_to_8th * value_to_8th;

    // Polynomial evaluation (fully unrolled)
    simsimd_f64_t polynomial_result = coeff_19 * value_to_2nd + coeff_18;
    polynomial_result = polynomial_result * value_to_2nd + coeff_17;
    polynomial_result = polynomial_result * value_to_2nd + coeff_16;
    polynomial_result = polynomial_result * value_to_2nd + coeff_15;
    polynomial_result = polynomial_result * value_to_2nd + coeff_14;
    polynomial_result = polynomial_result * value_to_2nd + coeff_13;
    polynomial_result = polynomial_result * value_to_2nd + coeff_12;
    polynomial_result = polynomial_result * value_to_2nd + coeff_11;
    polynomial_result = polynomial_result * value_to_2nd + coeff_10;
    polynomial_result = polynomial_result * value_to_2nd + coeff_9;
    polynomial_result = polynomial_result * value_to_2nd + coeff_8;
    polynomial_result = polynomial_result * value_to_2nd + coeff_7;
    polynomial_result = polynomial_result * value_to_2nd + coeff_6;
    polynomial_result = polynomial_result * value_to_2nd + coeff_5;
    polynomial_result = polynomial_result * value_to_2nd + coeff_4;
    polynomial_result = polynomial_result * value_to_2nd + coeff_3;
    polynomial_result = polynomial_result * value_to_2nd + coeff_2;
    polynomial_result = polynomial_result * value_to_2nd + coeff_1;

    // Adjust for quadrant
    simsimd_f64_t const pi_half = 1.5707963267948966; // π/2
    simsimd_f64_t result = value + value * value_to_2nd * polynomial_result;
    if (quadrant & 1) result = pi_half - result;
    if (quadrant & 2) result = -result;

    return result;
}

/**
 *  @brief  Computes the arc-tangent of (y/x) with @b 0-ULP error bound.
 *  @see    Based on @b `xatan2` in SLEEF library.
 *  @param  y The input sine value.
 *  @param  x The input cosine value.
 *  @return The arc-tangent of (y/x) in [-π/2, π/2] radians range.
 */
SIMSIMD_PUBLIC simsimd_f64_t simsimd_f64_atan2(simsimd_f64_t const y, simsimd_f64_t const x) {
    // Polynomial coefficients for atan2 approximation
    simsimd_f64_t const coeff_19 = -1.88796008463073496563746e-05;
    simsimd_f64_t const coeff_18 = +0.000209850076645816976906797;
    simsimd_f64_t const coeff_17 = -0.00110611831486672482563471;
    simsimd_f64_t const coeff_16 = +0.00370026744188713119232403;
    simsimd_f64_t const coeff_15 = -0.00889896195887655491740809;
    simsimd_f64_t const coeff_14 = +0.016599329773529201970117;
    simsimd_f64_t const coeff_13 = -0.0254517624932312641616861;
    simsimd_f64_t const coeff_12 = +0.0337852580001353069993897;
    simsimd_f64_t const coeff_11 = -0.0407629191276836500001934;
    simsimd_f64_t const coeff_10 = +0.0466667150077840625632675;
    simsimd_f64_t const coeff_9 = -0.0523674852303482457616113;
    simsimd_f64_t const coeff_8 = +0.0587666392926673580854313;
    simsimd_f64_t const coeff_7 = -0.0666573579361080525984562;
    simsimd_f64_t const coeff_6 = +0.0769219538311769618355029;
    simsimd_f64_t const coeff_5 = -0.090908995008245008229153;
    simsimd_f64_t const coeff_4 = +0.111111105648261418443745;
    simsimd_f64_t const coeff_3 = -0.14285714266771329383765;
    simsimd_f64_t const coeff_2 = +0.199999999996591265594148;
    simsimd_f64_t const coeff_1 = -0.333333333333311110369124;

    simsimd_fui64_t x_bits, y_bits;
    x_bits.f = x, y_bits.f = y;
    simsimd_fui64_t x_abs, y_abs;
    y_abs.u = y_bits.u & 0x7FFFFFFFFFFFFFFFull;

    // Quadrant adjustment
    int quadrant = 0;
    if (x < 0) { x_abs.f = -x, quadrant = -2; }
    else { x_abs.f = x; }
    // Now make sure its proper fraction, where the nominator is smaller than the denominator,
    // otherwise swap the absolute values that we will use down the road, but keep the `x_bits` and `y_bits`
    // as is for final qdrant re-adjustment.
    if (y_abs.f > x_abs.f) {
        simsimd_f64_t temp = x_abs.f;
        x_abs.f = y_abs.f;
        y_abs.f = -temp;
        quadrant += 1;
    }

    // Argument reduction
    simsimd_f64_t const ratio = y_abs.f / x_abs.f;
    simsimd_f64_t const ratio_to_2nd = ratio * ratio;
    simsimd_f64_t const ratio_to_4th = ratio_to_2nd * ratio_to_2nd;
    simsimd_f64_t const ratio_to_8th = ratio_to_4th * ratio_to_4th;
    simsimd_f64_t const ratio_to_16th = ratio_to_8th * ratio_to_8th;

    // Polynomial evaluation (fully unrolled)
    simsimd_f64_t polynomial_result = coeff_19 * ratio_to_2nd + coeff_18;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_17;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_16;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_15;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_14;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_13;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_12;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_11;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_10;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_9;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_8;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_7;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_6;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_5;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_4;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_3;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_2;
    polynomial_result = polynomial_result * ratio_to_2nd + coeff_1;

    // Adjust for quadrant
    simsimd_f64_t const epsilon = 1e-300;                // Near-zero threshold
    simsimd_f64_t const pi = 3.14159265358979323846;     // π
    simsimd_f64_t const pi_half = 1.5707963267948966;    // π/2
    simsimd_f64_t const pi_quarter = 0.7853981633974483; // π/4
    simsimd_u64_t const negative_zero = 0x8000000000000000ull;
    simsimd_u64_t const positive_infinity = 0x7FF0000000000000ull;
    simsimd_u64_t const negative_infinity = 0xFFF0000000000000ull;
    simsimd_f64_t result = polynomial_result * ratio_to_2nd * ratio + ratio;
    result += quadrant * pi_half;

    // Special cases handling using bit reinterpretation
    int const x_is_inf = (x_bits.u == positive_infinity) | (x_bits.u == negative_infinity);
    int const y_is_inf = (y_bits.u == positive_infinity) | (y_bits.u == negative_infinity);

    // Perform the sign multiplication and infer the right quadrant
    simsimd_fui64_t result_bits;
    result_bits.f = result;
    // Sign transfer:
    result_bits.u ^= x_bits.u & negative_zero;
    // Quadrant adjustments:
    if (x_is_inf | (x_bits.f == 0)) result_bits.f = pi_half - (x_is_inf ? (x_bits.f < 0 ? pi_half : 0) : 0);
    if (y_is_inf) result_bits.f = pi_half - (x_is_inf ? (x_bits.f < 0 ? pi_half : pi_quarter) : 0);
    if (y_bits.f == 0) result_bits.f = (x_bits.f < 0 ? pi : 0);
    if (x_is_inf | y_is_inf) result_bits.u = 0x7FF8000000000000ull;
    // Sign transfer back:
    else { result_bits.u ^= y_bits.u & negative_zero; }
    return result_bits.f;
}

SIMSIMD_PUBLIC void simsimd_sin_f32_serial(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs) {
    for (simsimd_size_t i = 0; i != n; ++i) outs[i] = simsimd_f32_sin(ins[i]);
}
SIMSIMD_PUBLIC void simsimd_cos_f32_serial(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs) {
    for (simsimd_size_t i = 0; i != n; ++i) outs[i] = simsimd_f32_cos(ins[i]);
}
SIMSIMD_PUBLIC void simsimd_atan_f32_serial(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs) {
    for (simsimd_size_t i = 0; i != n; ++i) outs[i] = simsimd_f32_atan(ins[i]);
}
SIMSIMD_PUBLIC void simsimd_sin_f64_serial(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs) {
    for (simsimd_size_t i = 0; i != n; ++i) outs[i] = simsimd_f64_sin(ins[i]);
}
SIMSIMD_PUBLIC void simsimd_cos_f64_serial(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs) {
    for (simsimd_size_t i = 0; i != n; ++i) outs[i] = simsimd_f64_cos(ins[i]);
}
SIMSIMD_PUBLIC void simsimd_atan_f64_serial(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs) {
    for (simsimd_size_t i = 0; i != n; ++i) outs[i] = simsimd_f64_atan(ins[i]);
}

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

SIMSIMD_INTERNAL __m512 _simsimd_f32x16_sin_skylake(__m512 const angles_radians) {
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

    // Reduce the angle to (angle - (rounded_quotients * π)) in [0, π]
    __m512 angles = _mm512_fnmadd_ps(rounded_quotients, pi, angles_radians);
    __m512 angles_squared = _mm512_mul_ps(angles, angles);
    __m512 angles_cubed = _mm512_mul_ps(angles, angles_squared);

    // Compute the polynomial approximation
    __m512 polynomials = coeff_5;
    polynomials = _mm512_fmadd_ps(polynomials, angles_squared, coeff_3);
    polynomials = _mm512_fmadd_ps(polynomials, angles_squared, coeff_1);
    __m512 results = _mm512_fmadd_ps(angles_cubed, polynomials, angles);

    // If multiples_of_pi is odd, flip the sign of the results
    __mmask16 odd_mask = _mm512_test_epi32_mask(multiples_of_pi, _mm512_set1_epi32(1));
    results = _mm512_mask_sub_ps(results, odd_mask, _mm512_setzero_ps(), results);
    return results;
}

SIMSIMD_INTERNAL __m512 _simsimd_f32x16_cos_skylake(__m512 const angles_radians) {
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

    // Reduce the angle to (angle - (multiples_of_pi * π)) in [-π/2, π/2]
    __m512 angles = _mm512_fnmadd_ps(rounded_quotients, pi, _mm512_sub_ps(angles_radians, pi_half));
    __m512 angles_squared = _mm512_mul_ps(angles, angles);
    __m512 angles_cubed = _mm512_mul_ps(angles, angles_squared);

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

SIMSIMD_INTERNAL __m512 _simsimd_f32x16_atan_skylake(__m512 const inputs) {
    // Polynomial coefficients
    __m512 const coeff_8 = _mm512_set1_ps(-0.333331018686294555664062f);
    __m512 const coeff_7 = _mm512_set1_ps(+0.199926957488059997558594f);
    __m512 const coeff_6 = _mm512_set1_ps(-0.142027363181114196777344f);
    __m512 const coeff_5 = _mm512_set1_ps(+0.106347933411598205566406f);
    __m512 const coeff_4 = _mm512_set1_ps(-0.0748900920152664184570312f);
    __m512 const coeff_3 = _mm512_set1_ps(+0.0425049886107444763183594f);
    __m512 const coeff_2 = _mm512_set1_ps(-0.0159569028764963150024414f);
    __m512 const coeff_1 = _mm512_set1_ps(+0.00282363896258175373077393f);
    __m512 const pi_half = _mm512_set1_ps(1.5707963267948966f); // π/2

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
    __m512 polynomial_result = coeff_1;
    polynomial_result = _mm512_fmadd_ps(polynomial_result, values_squared, coeff_2);
    polynomial_result = _mm512_fmadd_ps(polynomial_result, values_squared, coeff_3);
    polynomial_result = _mm512_fmadd_ps(polynomial_result, values_squared, coeff_4);
    polynomial_result = _mm512_fmadd_ps(polynomial_result, values_squared, coeff_5);
    polynomial_result = _mm512_fmadd_ps(polynomial_result, values_squared, coeff_6);
    polynomial_result = _mm512_fmadd_ps(polynomial_result, values_squared, coeff_7);
    polynomial_result = _mm512_fmadd_ps(polynomial_result, values_squared, coeff_8);

    // Adjust result for quadrants
    __m512 result = _mm512_fmadd_ps(values_cubed, polynomial_result, values);
    result = _mm512_mask_sub_ps(result, reciprocal_mask, pi_half, result);
    result = _mm512_mask_sub_ps(result, negative_mask, _mm512_setzero_ps(), result);
    return result;
}

SIMSIMD_PUBLIC void simsimd_sin_f32_skylake(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs) {
    simsimd_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 angles = _mm512_loadu_ps(ins + i);
        __m512 results = _simsimd_f32x16_sin_skylake(angles);
        _mm512_storeu_ps(outs + i, results);
    }
    if (i < n) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m512 angles = _mm512_maskz_loadu_ps(mask, ins + i);
        __m512 results = _simsimd_f32x16_sin_skylake(angles);
        _mm512_mask_storeu_ps(outs + i, mask, results);
    }
}
SIMSIMD_PUBLIC void simsimd_cos_f32_skylake(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs) {
    simsimd_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 angles = _mm512_loadu_ps(ins + i);
        __m512 results = _simsimd_f32x16_cos_skylake(angles);
        _mm512_storeu_ps(outs + i, results);
    }
    if (i < n) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m512 angles = _mm512_maskz_loadu_ps(mask, ins + i);
        __m512 results = _simsimd_f32x16_cos_skylake(angles);
        _mm512_mask_storeu_ps(outs + i, mask, results);
    }
}
SIMSIMD_PUBLIC void simsimd_atan_f32_skylake(simsimd_f32_t const *ins, simsimd_size_t n, simsimd_f32_t *outs) {
    simsimd_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 angles = _mm512_loadu_ps(ins + i);
        __m512 results = _simsimd_f32x16_atan_skylake(angles);
        _mm512_storeu_ps(outs + i, results);
    }
    if (i < n) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m512 angles = _mm512_maskz_loadu_ps(mask, ins + i);
        __m512 results = _simsimd_f32x16_atan_skylake(angles);
        _mm512_mask_storeu_ps(outs + i, mask, results);
    }
}

SIMSIMD_INTERNAL __m512d _simsimd_f64x8_sin_skylake(__m512d const angles_radians) {
    // Constants for argument reduction
    __m512d const pi_high = _mm512_set1_pd(3.141592653589793116);           // High-digits part of π
    __m512d const pi_low = _mm512_set1_pd(1.2246467991473532072e-16);       // Low-digits part of π
    __m512d const pi_reciprocal = _mm512_set1_pd(0.31830988618379067154);   // 1/π
    __m512d const coeff_8 = _mm512_set1_pd(-0.166666666666666657414808);    // Coefficient for x
    __m512d const coeff_0 = _mm512_set1_pd(0.00833333333333332974823815);   // Coefficient for x^3
    __m512d const coeff_1 = _mm512_set1_pd(-0.000198412698412696162806809); // Coefficient for x^5
    __m512d const coeff_2 = _mm512_set1_pd(2.75573192239198747630416e-06);  // Coefficient for x^7
    __m512d const coeff_3 = _mm512_set1_pd(-2.50521083763502045810755e-08); // Coefficient for x^9

    // Compute (multiple_of_pi) = round(angle / π)
    __m512d quotients = _mm512_mul_pd(angles_radians, pi_reciprocal);
    __m512d rounded_quotients = _mm512_roundscale_pd(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i multiples_of_pi = _mm512_cvtpd_epi32(rounded_quotients);

    // Reduce the angle: angle = angle - (multiple_of_pi * π_high + multiple_of_pi * π_low)
    __m512d reduced_angle = _mm512_fnmadd_pd(rounded_quotients, pi_high, angles_radians);
    reduced_angle = _mm512_fnmadd_pd(rounded_quotients, pi_low, reduced_angle);

    // Compute polynomial approximation
    __m512d angle_squared = _mm512_mul_pd(reduced_angle, reduced_angle);
    __m512d polynomial = coeff_3;
    polynomial = _mm512_fmadd_pd(polynomial, angle_squared, coeff_2);
    polynomial = _mm512_fmadd_pd(polynomial, angle_squared, coeff_1);
    polynomial = _mm512_fmadd_pd(polynomial, angle_squared, coeff_0);
    polynomial = _mm512_fmadd_pd(polynomial, angle_squared, coeff_8);
    __m512d results = _mm512_fmadd_pd(angle_squared, polynomial, reduced_angle);

    // Adjust for odd multiples of π
    __mmask8 odd_mask = _mm256_test_epi32_mask(multiples_of_pi, _mm256_set1_epi32(1));
    results = _mm512_mask_sub_pd(results, odd_mask, _mm512_setzero_pd(), results);
    return results;
}

SIMSIMD_INTERNAL __m512d _simsimd_f64x8_cos_skylake(__m512d const angles_radians) {
    // Constants for argument reduction
    __m512d const pi_high = _mm512_set1_pd(3.141592653589793116);           // High-digits part of π
    __m512d const pi_low = _mm512_set1_pd(1.2246467991473532072e-16);       // Low-digits part of π
    __m512d const pi_reciprocal = _mm512_set1_pd(0.31830988618379067154);   // 1/π
    __m512d const pi_half = _mm512_set1_pd(1.5707963267948966);             // π/2
    __m512d const coeff_8 = _mm512_set1_pd(-0.166666666666666657414808);    // Coefficient for x
    __m512d const coeff_0 = _mm512_set1_pd(0.00833333333333332974823815);   // Coefficient for x^3
    __m512d const coeff_1 = _mm512_set1_pd(-0.000198412698412696162806809); // Coefficient for x^5
    __m512d const coeff_2 = _mm512_set1_pd(2.75573192239198747630416e-06);  // Coefficient for x^7
    __m512d const coeff_3 = _mm512_set1_pd(-2.50521083763502045810755e-08); // Coefficient for x^9

    // Compute (multiple_of_pi) = round((angle / π) - 0.5)
    __m512d quotients = _mm512_fmsub_pd(angles_radians, pi_reciprocal, _mm512_set1_pd(0.5));
    __m512d rounded_quotients = _mm512_roundscale_pd(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i multiples_of_pi = _mm512_cvtpd_epi32(rounded_quotients);

    // Reduce the angle: angle = angle - (multiple_of_pi * π_high + multiple_of_pi * π_low)
    __m512d reduced_angle = _mm512_fnmadd_pd(rounded_quotients, pi_high, angles_radians);
    reduced_angle = _mm512_fnmadd_pd(rounded_quotients, pi_low, reduced_angle);

    // Compute polynomial approximation
    __m512d angle_squared = _mm512_mul_pd(reduced_angle, reduced_angle);
    __m512d polynomial = coeff_3;
    polynomial = _mm512_fmadd_pd(polynomial, angle_squared, coeff_2);
    polynomial = _mm512_fmadd_pd(polynomial, angle_squared, coeff_1);
    polynomial = _mm512_fmadd_pd(polynomial, angle_squared, coeff_0);
    polynomial = _mm512_fmadd_pd(polynomial, angle_squared, coeff_8);
    __m512d results = _mm512_fmadd_pd(angle_squared, polynomial, reduced_angle);

    // Adjust for even multiples of π
    __mmask8 even_mask = _mm256_testn_epi32_mask(multiples_of_pi, _mm256_set1_epi32(1));
    results = _mm512_mask_sub_pd(results, even_mask, results, _mm512_setzero_pd());
    return results;
}

SIMSIMD_INTERNAL __m512d _simsimd_f64x8_atan_skylake(__m512d const values) {
    // Polynomial coefficients for atan approximation
    __m512d const coeff_1 = _mm512_set1_pd(-0.333333333333311110369124);
    __m512d const coeff_2 = _mm512_set1_pd(0.199999999996591265594148);
    __m512d const coeff_3 = _mm512_set1_pd(-0.14285714266771329383765);
    __m512d const coeff_4 = _mm512_set1_pd(0.111111105648261418443745);
    __m512d const coeff_5 = _mm512_set1_pd(-0.090908995008245008229153);
    __m512d const coeff_6 = _mm512_set1_pd(0.0769219538311769618355029);
    __m512d const coeff_7 = _mm512_set1_pd(-0.0666573579361080525984562);
    __m512d const coeff_8 = _mm512_set1_pd(0.0587666392926673580854313);
    __m512d const coeff_9 = _mm512_set1_pd(-0.0523674852303482457616113);
    __m512d const coeff_10 = _mm512_set1_pd(0.0466667150077840625632675);
    __m512d const pi_half = _mm512_set1_pd(1.5707963267948966); // π/2

    // Quadrant adjustments
    __mmask8 negative_mask = _mm512_cmp_pd_mask(values, _mm512_setzero_pd(), _CMP_LT_OS);
    __m512d abs_values = _mm512_abs_pd(values);
    __mmask8 reciprocal_mask = _mm512_cmp_pd_mask(abs_values, _mm512_set1_pd(1.0), _CMP_GT_OS);
    abs_values = _mm512_mask_div_pd(abs_values, reciprocal_mask, _mm512_set1_pd(1.0), abs_values);

    // Polynomial evaluation (argument reduction and approximation)
    __m512d values_squared = _mm512_mul_pd(abs_values, abs_values);
    __m512d polynomial = coeff_10;
    polynomial = _mm512_fmadd_pd(polynomial, values_squared, coeff_9);
    polynomial = _mm512_fmadd_pd(polynomial, values_squared, coeff_8);
    polynomial = _mm512_fmadd_pd(polynomial, values_squared, coeff_7);
    polynomial = _mm512_fmadd_pd(polynomial, values_squared, coeff_6);
    polynomial = _mm512_fmadd_pd(polynomial, values_squared, coeff_5);
    polynomial = _mm512_fmadd_pd(polynomial, values_squared, coeff_4);
    polynomial = _mm512_fmadd_pd(polynomial, values_squared, coeff_3);
    polynomial = _mm512_fmadd_pd(polynomial, values_squared, coeff_2);
    polynomial = _mm512_fmadd_pd(polynomial, values_squared, coeff_1);

    // Compute atan approximation
    __m512d result = _mm512_fmadd_pd(values_squared, polynomial, abs_values);
    result = _mm512_mask_sub_pd(result, reciprocal_mask, pi_half, result);
    result = _mm512_mask_sub_pd(result, negative_mask, _mm512_setzero_pd(), result);
    return result;
}

SIMSIMD_PUBLIC void simsimd_sin_f64_skylake(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs) {
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d angles = _mm512_loadu_pd(ins + i);
        __m512d results = _simsimd_f64x8_sin_skylake(angles);
        _mm512_storeu_pd(outs + i, results);
    }
    if (i < n) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFF, n - i);
        __m512d angles = _mm512_maskz_loadu_pd(mask, ins + i);
        __m512d results = _simsimd_f64x8_sin_skylake(angles);
        _mm512_mask_storeu_pd(outs + i, mask, results);
    }
}
SIMSIMD_PUBLIC void simsimd_cos_f64_skylake(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs) {
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d angles = _mm512_loadu_pd(ins + i);
        __m512d results = _simsimd_f64x8_cos_skylake(angles);
        _mm512_storeu_pd(outs + i, results);
    }
    if (i < n) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFF, n - i);
        __m512d angles = _mm512_maskz_loadu_pd(mask, ins + i);
        __m512d results = _simsimd_f64x8_cos_skylake(angles);
        _mm512_mask_storeu_pd(outs + i, mask, results);
    }
}
SIMSIMD_PUBLIC void simsimd_atan_f64_skylake(simsimd_f64_t const *ins, simsimd_size_t n, simsimd_f64_t *outs) {
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d angles = _mm512_loadu_pd(ins + i);
        __m512d results = _simsimd_f64x8_atan_skylake(angles);
        _mm512_storeu_pd(outs + i, results);
    }
    if (i < n) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFF, n - i);
        __m512d angles = _mm512_maskz_loadu_pd(mask, ins + i);
        __m512d results = _simsimd_f64x8_atan_skylake(angles);
        _mm512_mask_storeu_pd(outs + i, mask, results);
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE
#endif // _SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif

#endif
