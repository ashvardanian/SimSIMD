/**
 *  @brief      SIMD-accelerated Similarity Measures for Complex Vectors.
 *  @author     Ash Vardanian
 *  @date       February 24, 2024
 *
 *  Contains:
 *  - Inner Product
 *  - Cosine Distance
 *
 *  For datatypes:
 *  - 32-bit floating point numbers
 *  - 16-bit floating point numbers
 *
 *  For hardware architectures:
 *  - Arm (NEON, SVE?)
 *  - x86 (AVX2?, AVX512?)
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef SIMSIMD_COMPLEX_H
#define SIMSIMD_COMPLEX_H

#include "types.h"

#define SIMSIMD_MAKE_COMPLEX_DOT(name, input_type, accumulator_type, converter, epsilon)                               \
    inline static void simsimd_##name##_##input_type##c_dot(simsimd_##input_type##_t const* a,                         \
                                                            simsimd_##input_type##_t const* b, simsimd_size_t n,       \
                                                            simsimd_f32_t* real_out, simsimd_f32_t* imag_out) {        \
        simsimd_##accumulator_type##_t ab_real, ab_imag;                                                               \
        for (simsimd_size_t i = 0; i + 2 <= n; i += 2) {                                                               \
            simsimd_##accumulator_type##_t ar = converter(a[i]);                                                       \
            simsimd_##accumulator_type##_t br = converter(b[i]);                                                       \
            simsimd_##accumulator_type##_t ai = converter(a[i + 1]);                                                   \
            simsimd_##accumulator_type##_t bi = converter(b[i + 1]);                                                   \
            ab_real += ar * br - ai * bi;                                                                              \
            ab_imag += ar * bi + ai * br;                                                                              \
        }                                                                                                              \
        *real_out = ab_real;                                                                                           \
        *imag_out = ab_imag;                                                                                           \
    }

#define SIMSIMD_MAKE_COMPLEX_VDOT(name, input_type, accumulator_type, converter, epsilon)                              \
    inline static void simsimd_##name##_##input_type##c_vdot(simsimd_##input_type##_t const* a,                        \
                                                             simsimd_##input_type##_t const* b, simsimd_size_t n,      \
                                                             simsimd_f32_t* real_out, simsimd_f32_t* imag_out) {       \
        simsimd_##accumulator_type##_t ab_real, ab_imag, a2_sum, b2_sum;                                               \
        for (simsimd_size_t i = 0; i + 2 <= n; i += 2) {                                                               \
            simsimd_##accumulator_type##_t ar = converter(a[i]);                                                       \
            simsimd_##accumulator_type##_t br = converter(b[i]);                                                       \
            simsimd_##accumulator_type##_t ai = converter(a[i + 1]);                                                   \
            simsimd_##accumulator_type##_t bi = converter(b[i + 1]);                                                   \
            ab_real += ar * br + ai * bi;                                                                              \
            ab_imag += ar * bi - ai * br;                                                                              \
        }                                                                                                              \
        *real_out = ab_real;                                                                                           \
        *imag_out = ab_imag;                                                                                           \
    }

#ifdef __cplusplus
extern "C" {
#endif

SIMSIMD_MAKE_COMPLEX_DOT(serial, f64, f64, SIMSIMD_IDENTIFY,
                         SIMSIMD_F32_DIVISION_EPSILON) // simsimd_serial_f64c_dot
SIMSIMD_MAKE_COMPLEX_VDOT(serial, f64, f64, SIMSIMD_IDENTIFY,
                          SIMSIMD_F32_DIVISION_EPSILON) // simsimd_serial_f64c_vdot

SIMSIMD_MAKE_COMPLEX_DOT(serial, f32, f32, SIMSIMD_IDENTIFY,
                         SIMSIMD_F32_DIVISION_EPSILON) // simsimd_serial_f32c_dot
SIMSIMD_MAKE_COMPLEX_VDOT(serial, f32, f32, SIMSIMD_IDENTIFY,
                          SIMSIMD_F32_DIVISION_EPSILON) // simsimd_serial_f32c_vdot

SIMSIMD_MAKE_COMPLEX_DOT(serial, f16, f32, SIMSIMD_UNCOMPRESS_F16,
                         SIMSIMD_F32_DIVISION_EPSILON) // simsimd_serial_f16c_dot
SIMSIMD_MAKE_COMPLEX_VDOT(serial, f16, f32, SIMSIMD_UNCOMPRESS_F16,
                          SIMSIMD_F32_DIVISION_EPSILON) // simsimd_serial_f16c_vdot

SIMSIMD_MAKE_COMPLEX_DOT(accurate, f32, f64, SIMSIMD_IDENTIFY,
                         SIMSIMD_F32_DIVISION_EPSILON) // simsimd_accurate_f32c_dot
SIMSIMD_MAKE_COMPLEX_VDOT(accurate, f32, f64, SIMSIMD_IDENTIFY,
                          SIMSIMD_F32_DIVISION_EPSILON) // simsimd_accurate_f32c_vdot

SIMSIMD_MAKE_COMPLEX_DOT(accurate, f16, f64, SIMSIMD_UNCOMPRESS_F16,
                         SIMSIMD_F32_DIVISION_EPSILON) // simsimd_accurate_f16c_dot
SIMSIMD_MAKE_COMPLEX_VDOT(accurate, f16, f64, SIMSIMD_UNCOMPRESS_F16,
                          SIMSIMD_F32_DIVISION_EPSILON) // simsimd_accurate_f16c_vdot

#if SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_ARM_NEON

/*
 *  @file   arm_neon_f32.h
 *  @brief  Arm NEON implementation of the most common similarity metrics for 32-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: Kullback-Leibler and Jensen–Shannon divergence.
 *  - Uses `f32` for storage and `f32` for accumulation.
 *  - Requires compiler capabilities: +simd.
 */

__attribute__((target("+simd"))) //
inline static void
simsimd_neon_f32c_dot(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, //
                      simsimd_f32_t* real_out, simsimd_f32_t* imag_out) {
    float32x4_t ab_real_vec = vdupq_n_f32(0);
    float32x4_t ab_imag_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // Unpack the input arrays into real and imaginary parts:
        float32x4x2_t a_vec = vld2q_f32(a + i);
        float32x4x2_t b_vec = vld2q_f32(b + i);
        float32x4_t a_real_vec = a_vec.val[0];
        float32x4_t a_imag_vec = a_vec.val[1];
        float32x4_t b_real_vec = b_vec.val[0];
        float32x4_t b_imag_vec = b_vec.val[1];

        // Compute the dot product:
        ab_real_vec = vfmaq_f32(ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = vfmsq_f32(ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = vfmaq_f32(ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = vfmaq_f32(ab_imag_vec, a_imag_vec, b_real_vec);
    }

    // Reduce horizontal sums:
    simsimd_f32_t ab_real = vaddvq_f32(ab_real_vec);
    simsimd_f32_t ab_imag = vaddvq_f32(ab_imag_vec);

    // Handle the tail:
    for (; i + 2 <= n; i += 2) {
        simsimd_f32_t ar = a[i], ai = a[i + 1], br = b[i], bi = b[i + 1];
        ab_real += ar * br - ai * bi;
        ab_imag += ar * bi + ai * br;
    }
    *real_out = ab_real;
    *imag_out = ab_imag;
}

__attribute__((target("+simd"))) //
inline static void
simsimd_neon_f32c_vdot(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, //
                       simsimd_f32_t* real_out, simsimd_f32_t* imag_out) {
    float32x4_t ab_real_vec = vdupq_n_f32(0);
    float32x4_t ab_imag_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // Unpack the input arrays into real and imaginary parts:
        float32x4x2_t a_vec = vld2q_f32(a + i);
        float32x4x2_t b_vec = vld2q_f32(b + i);
        float32x4_t a_real_vec = a_vec.val[0];
        float32x4_t a_imag_vec = a_vec.val[1];
        float32x4_t b_real_vec = b_vec.val[0];
        float32x4_t b_imag_vec = b_vec.val[1];

        // Compute the dot product:
        ab_real_vec = vfmaq_f32(ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = vfmaq_f32(ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = vfmaq_f32(ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = vfmsq_f32(ab_imag_vec, a_imag_vec, b_real_vec);
    }

    // Reduce horizontal sums:
    simsimd_f32_t ab_real = vaddvq_f32(ab_real_vec);
    simsimd_f32_t ab_imag = vaddvq_f32(ab_imag_vec);

    // Handle the tail:
    for (; i + 2 <= n; i += 2) {
        simsimd_f32_t ar = a[i], ai = a[i + 1], br = b[i], bi = b[i + 1];
        ab_real += ar * br + ai * bi;
        ab_imag += ar * bi - ai * br;
    }
    *real_out = ab_real;
    *imag_out = ab_imag;
}

#endif // SIMSIMD_TARGET_ARM_NEON
#endif // SIMSIMD_TARGET_ARM

#ifdef __cplusplus
}
#endif

#endif
