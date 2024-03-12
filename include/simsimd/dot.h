/**
 *  @file       dot.h
 *  @brief      SIMD-accelerated Dot Products for Real and Complex numbers.
 *  @author     Ash Vardanian
 *  @date       February 24, 2024
 *
 *  Contains:
 *  - Dot Product
 *  - Conjugate Dot Product for Complex vectors
 *
 *  For datatypes:
 *  - 64-bit IEEE floating point numbers
 *  - 32-bit IEEE floating point numbers
 *  - 16-bit floating point numbers
 *  - 8-bit signed integral numbers
 *
 *  For hardware architectures:
 *  - Arm (NEON, SVE?)
 *  - x86 (AVX2?, AVX512?)
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef SIMSIMD_DOT_H
#define SIMSIMD_DOT_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// clang-format off

/*  Serial backends for all numeric types.
 *  By default they use 32-bit arithmetic, unless the arguments themselves contain 64-bit floats.
 *  For double-precision computation check out the "*_accurate" variants of those "*_serial" functions.
 */
inline static void simsimd_dot_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const*, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f64c_serial(simsimd_f64_t const* a, simsimd_f64_t const*, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_vdot_f64c_serial(simsimd_f64_t const* a, simsimd_f64_t const*, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_dot_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f32c_serial(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_vdot_f32c_serial(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_dot_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f16c_serial(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_vdot_f16c_serial(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* results);

/*  Double-precision serial backends for all numeric types.
 *  For single-precision computation check out the "*_serial" counterparts of those "*_accurate" functions.
 */
inline static void simsimd_dot_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f32c_accurate(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_vdot_f32c_accurate(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_dot_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f16c_accurate(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_vdot_f16c_accurate(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* results);

/*  SIMD-powered backends for Arm NEON, mostly using 32-bit arithmetic over 128-bit words.
 *  By far the most portable backend, covering most Arm v8 devices, over a billion phones, and almost all
 *  server CPUs produced before 2023.
 */
inline static void simsimd_dot_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_i8_neon(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f32c_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_dot_f16c_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_vdot_f32c_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_vdot_f16c_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/*  SIMD-powered backends for Arm SVE, mostly using 32-bit arithmetic over variable-length platform-defined word sizes.
 *  Designed for Arm Graviton 3, Microsoft Cobalt, as well as Nvidia Grace and newer Ampere Altra CPUs.
 */
inline static void simsimd_dot_f32_sve(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f16_sve(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f64_sve(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result);

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic over 256-bit words.
 *  First demonstrated in 2011, at least one Haswell-based processor was still being sold in 2022 — the Pentium G3420.
 *  Practically all modern x86 CPUs support AVX2, FMA, and F16C, making it a perfect baseline for SIMD algorithms.
 *  On other hand, there is no need to implement AVX2 versions of `f32` and `f64` functions, as those are
 *  properly vectorized by recent compilers.
 */
inline static void simsimd_dot_i8_haswell(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f32c_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_vdot_f32c_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_dot_f16c_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* results);
inline static void simsimd_vdot_f16c_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/*  SIMD-powered backends for various generations of AVX512 CPUs.
 *  Skylake is handy, as it supports masked loads and other operations, avoiding the need for the tail loop.
 *  Ice Lake added VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG, VPCLMULQDQ, and other extensions for integral operations.
 *  Sapphire Rapids added tiled matrix operations, but we are most interested in the new mixed-precision FMA instructions.
 */
inline static void simsimd_dot_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f32c_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_vdot_f32c_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f64c_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_vdot_f64c_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result);

inline static void simsimd_dot_i8_ice(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* result);
inline static void simsimd_dot_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
// clang-format on

#define SIMSIMD_MAKE_DOT(name, input_type, accumulator_type, converter)                                                \
    inline static void simsimd_dot_##input_type##_##name(simsimd_##input_type##_t const* a,                            \
                                                         simsimd_##input_type##_t const* b, simsimd_size_t n,          \
                                                         simsimd_distance_t* result) {                                 \
        simsimd_##accumulator_type##_t ab = 0;                                                                         \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                      \
            simsimd_##accumulator_type##_t ai = converter(a[i]);                                                       \
            simsimd_##accumulator_type##_t bi = converter(b[i]);                                                       \
            ab += ai * bi;                                                                                             \
        }                                                                                                              \
        *result = ab;                                                                                                  \
    }

#define SIMSIMD_MAKE_COMPLEX_DOT(name, input_type, accumulator_type, converter)                                        \
    inline static void simsimd_dot_##input_type##c_##name(simsimd_##input_type##_t const* a,                           \
                                                          simsimd_##input_type##_t const* b, simsimd_size_t n,         \
                                                          simsimd_distance_t* results) {                               \
        simsimd_##accumulator_type##_t ab_real = 0, ab_imag = 0;                                                       \
        for (simsimd_size_t i = 0; i + 2 <= n; i += 2) {                                                               \
            simsimd_##accumulator_type##_t ar = converter(a[i]);                                                       \
            simsimd_##accumulator_type##_t br = converter(b[i]);                                                       \
            simsimd_##accumulator_type##_t ai = converter(a[i + 1]);                                                   \
            simsimd_##accumulator_type##_t bi = converter(b[i + 1]);                                                   \
            ab_real += ar * br - ai * bi;                                                                              \
            ab_imag += ar * bi + ai * br;                                                                              \
        }                                                                                                              \
        results[0] = ab_real;                                                                                          \
        results[1] = ab_imag;                                                                                          \
    }

#define SIMSIMD_MAKE_COMPLEX_VDOT(name, input_type, accumulator_type, converter)                                       \
    inline static void simsimd_vdot_##input_type##c_##name(simsimd_##input_type##_t const* a,                          \
                                                           simsimd_##input_type##_t const* b, simsimd_size_t n,        \
                                                           simsimd_distance_t* results) {                              \
        simsimd_##accumulator_type##_t ab_real = 0, ab_imag = 0;                                                       \
        for (simsimd_size_t i = 0; i + 2 <= n; i += 2) {                                                               \
            simsimd_##accumulator_type##_t ar = converter(a[i]);                                                       \
            simsimd_##accumulator_type##_t br = converter(b[i]);                                                       \
            simsimd_##accumulator_type##_t ai = converter(a[i + 1]);                                                   \
            simsimd_##accumulator_type##_t bi = converter(b[i + 1]);                                                   \
            ab_real += ar * br + ai * bi;                                                                              \
            ab_imag += ar * bi - ai * br;                                                                              \
        }                                                                                                              \
        results[0] = ab_real;                                                                                          \
        results[1] = ab_imag;                                                                                          \
    }

SIMSIMD_MAKE_DOT(serial, f64, f64, SIMSIMD_IDENTIFY)          // simsimd_dot_f64_serial
SIMSIMD_MAKE_COMPLEX_DOT(serial, f64, f64, SIMSIMD_IDENTIFY)  // simsimd_dot_f64c_serial
SIMSIMD_MAKE_COMPLEX_VDOT(serial, f64, f64, SIMSIMD_IDENTIFY) // simsimd_vdot_f64c_serial

SIMSIMD_MAKE_DOT(serial, f32, f32, SIMSIMD_IDENTIFY)          // simsimd_dot_f32_serial
SIMSIMD_MAKE_COMPLEX_DOT(serial, f32, f32, SIMSIMD_IDENTIFY)  // simsimd_dot_f32c_serial
SIMSIMD_MAKE_COMPLEX_VDOT(serial, f32, f32, SIMSIMD_IDENTIFY) // simsimd_vdot_f32c_serial

SIMSIMD_MAKE_DOT(serial, f16, f32, SIMSIMD_UNCOMPRESS_F16)          // simsimd_dot_f16_serial
SIMSIMD_MAKE_COMPLEX_DOT(serial, f16, f32, SIMSIMD_UNCOMPRESS_F16)  // simsimd_dot_f16c_serial
SIMSIMD_MAKE_COMPLEX_VDOT(serial, f16, f32, SIMSIMD_UNCOMPRESS_F16) // simsimd_vdot_f16c_serial

SIMSIMD_MAKE_DOT(accurate, f32, f64, SIMSIMD_IDENTIFY)          // simsimd_dot_f32_accurate
SIMSIMD_MAKE_COMPLEX_DOT(accurate, f32, f64, SIMSIMD_IDENTIFY)  // simsimd_dot_f32c_accurate
SIMSIMD_MAKE_COMPLEX_VDOT(accurate, f32, f64, SIMSIMD_IDENTIFY) // simsimd_vdot_f32c_accurate

SIMSIMD_MAKE_DOT(accurate, f16, f64, SIMSIMD_UNCOMPRESS_F16)          // simsimd_dot_f16_accurate
SIMSIMD_MAKE_COMPLEX_DOT(accurate, f16, f64, SIMSIMD_UNCOMPRESS_F16)  // simsimd_dot_f16c_accurate
SIMSIMD_MAKE_COMPLEX_VDOT(accurate, f16, f64, SIMSIMD_UNCOMPRESS_F16) // simsimd_vdot_f16c_accurate

#if SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON

__attribute__((target("+simd"))) //
inline static void
simsimd_dot_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    float32x4_t ab_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
    }
    simsimd_f32_t ab = vaddvq_f32(ab_vec);
    for (; i < n; ++i)
        ab += a[i] * b[i];
    *result = ab;
}

__attribute__((target("+simd"))) //
inline static void
simsimd_dot_f32c_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, //
                      simsimd_distance_t* results) {
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
    results[0] = ab_real;
    results[1] = ab_imag;
}

__attribute__((target("+simd"))) //
inline static void
simsimd_vdot_f32c_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, //
                       simsimd_distance_t* results) {
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
    results[0] = ab_real;
    results[1] = ab_imag;
}

/*
 *  @file   arm_neon_f16.h
 *  @brief  Arm NEON implementation of the most common similarity metrics for 16-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: L2 squared, inner product, cosine similarity.
 *  - Uses `f16` for storage and `f32` for accumulation, as the 16-bit FMA may not always be available.
 *  - Requires compiler capabilities: +simd+fp16.
 */

__attribute__((target("+simd+fp16"))) //
inline static void
simsimd_dot_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    float32x4_t ab_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vcvt_f32_f16(vld1_f16((float16_t const*)a + i));
        float32x4_t b_vec = vcvt_f32_f16(vld1_f16((float16_t const*)b + i));
        ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
    }

    // In case the software emulation for `f16` scalars is enabled, the `simsimd_uncompress_f16`
    // function will run. It is extremely slow, so even for the tail, let's combine serial
    // loads and stores with vectorized math.
    if (i < n) {
        union {
            float16x4_t f16_vec;
            simsimd_f16_t f16[4];
        } a_padded_tail, b_padded_tail;
        simsimd_size_t j = 0;
        for (; i < n; ++i, ++j)
            a_padded_tail.f16[j] = a[i], b_padded_tail.f16[j] = b[i];
        for (; j < 4; ++j)
            a_padded_tail.f16[j] = 0, b_padded_tail.f16[j] = 0;
        ab_vec = vfmaq_f32(ab_vec, vcvt_f32_f16(a_padded_tail.f16_vec), vcvt_f32_f16(b_padded_tail.f16_vec));
    }
    *result = vaddvq_f32(ab_vec);
}

/*
 *  @file   arm_neon_i8.h
 *  @brief  Arm NEON implementation of the most common similarity metrics for 8-bit signed integral numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: L2 squared, cosine similarity, inner product (same as cosine).
 *  - Uses `i8` for storage, `i16` for multiplication, and `i32` for accumulation, if no better option is available.
 *  - Requires compiler capabilities: +simd+dotprod.
 */

__attribute__((target("arch=armv8.2-a+dotprod"))) //
inline static void
simsimd_cos_i8_neon(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* result) {

    int32x4_t ab_vec = vdupq_n_s32(0);
    int32x4_t a2_vec = vdupq_n_s32(0);
    int32x4_t b2_vec = vdupq_n_s32(0);
    simsimd_size_t i = 0;

    // If the 128-bit `vdot_s32` intrinsic is unavailable, we can use the 64-bit `vdot_s32`.
    // for (simsimd_size_t i = 0; i != n; i += 8) {
    //     int16x8_t a_vec = vmovl_s8(vld1_s8(a + i));
    //     int16x8_t b_vec = vmovl_s8(vld1_s8(b + i));
    //     int16x8_t ab_part_vec = vmulq_s16(a_vec, b_vec);
    //     int16x8_t a2_part_vec = vmulq_s16(a_vec, a_vec);
    //     int16x8_t b2_part_vec = vmulq_s16(b_vec, b_vec);
    //     ab_vec = vaddq_s32(ab_vec, vaddq_s32(vmovl_s16(vget_high_s16(ab_part_vec)), //
    //                                          vmovl_s16(vget_low_s16(ab_part_vec))));
    //     a2_vec = vaddq_s32(a2_vec, vaddq_s32(vmovl_s16(vget_high_s16(a2_part_vec)), //
    //                                          vmovl_s16(vget_low_s16(a2_part_vec))));
    //     b2_vec = vaddq_s32(b2_vec, vaddq_s32(vmovl_s16(vget_high_s16(b2_part_vec)), //
    //                                          vmovl_s16(vget_low_s16(b2_part_vec))));
    // }
    for (; i + 15 < n; i += 16) {
        int8x16_t a_vec = vld1q_s8(a + i);
        int8x16_t b_vec = vld1q_s8(b + i);
        ab_vec = vdotq_s32(ab_vec, a_vec, b_vec);
        a2_vec = vdotq_s32(a2_vec, a_vec, a_vec);
        b2_vec = vdotq_s32(b2_vec, b_vec, b_vec);
    }

    int32_t ab = vaddvq_s32(ab_vec);
    int32_t a2 = vaddvq_s32(a2_vec);
    int32_t b2 = vaddvq_s32(b2_vec);

    // Take care of the tail:
    for (; i < n; ++i) {
        int32_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    // Avoid `simsimd_approximate_inverse_square_root` on Arm NEON
    simsimd_f32_t a2_b2_arr[2] = {(simsimd_f32_t)a2, (simsimd_f32_t)b2};
    float32x2_t a2_b2 = vld1_f32(a2_b2_arr);
    a2_b2 = vrsqrte_f32(a2_b2);
    vst1_f32(a2_b2_arr, a2_b2);
    *result = ab != 0 ? ab * a2_b2_arr[0] * a2_b2_arr[1] : 0;
}

#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_SVE

__attribute__((target("+sve"))) //
inline static void
simsimd_dot_f32_sve(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    simsimd_size_t i = 0;
    svfloat32_t ab_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
        svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
        ab_vec = svmla_f32_x(pg_vec, ab_vec, a_vec, b_vec);
        i += svcntw();
    } while (i < n);
    *result = svaddv_f32(svptrue_b32(), ab_vec);
}

__attribute__((target("+sve+fp16"))) //
inline static void
simsimd_dot_f16_sve(simsimd_f16_t const* a_enum, simsimd_f16_t const* b_enum, simsimd_size_t n,
                    simsimd_distance_t* result) {
    simsimd_size_t i = 0;
    svfloat16_t ab_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    simsimd_f16_t const* a = (simsimd_f16_t const*)(a_enum);
    simsimd_f16_t const* b = (simsimd_f16_t const*)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svfloat16_t a_vec = svld1_f16(pg_vec, (float16_t const*)a + i);
        svfloat16_t b_vec = svld1_f16(pg_vec, (float16_t const*)b + i);
        ab_vec = svmla_f16_x(pg_vec, ab_vec, a_vec, b_vec);
        i += svcnth();
    } while (i < n);
    simsimd_f16_t ab = svaddv_f16(svptrue_b16(), ab_vec);
    *result = ab;
}

__attribute__((target("+sve"))) //
inline static void
simsimd_dot_f64_sve(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    simsimd_size_t i = 0;
    svfloat64_t ab_vec = svdupq_n_f64(0.0, 0.0);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)i, (unsigned int)n);
        svfloat64_t a_vec = svld1_f64(pg_vec, a + i);
        svfloat64_t b_vec = svld1_f64(pg_vec, b + i);
        ab_vec = svmla_f64_x(pg_vec, ab_vec, a_vec, b_vec);
        i += svcntd();
    } while (i < n);
    *result = svaddv_f64(svptrue_b32(), ab_vec);
}

#endif // SIMSIMD_TARGET_SVE
#endif // SIMSIMD_TARGET_ARM

#if SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL

__attribute__((target("avx2,f16c,fma"))) //
inline static void
simsimd_dot_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    __m256 ab_vec = _mm256_setzero_ps();
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));
        ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
    }

    // In case the software emulation for `f16` scalars is enabled, the `simsimd_uncompress_f16`
    // function will run. It is extremely slow, so even for the tail, let's combine serial
    // loads and stores with vectorized math.
    if (i < n) {
        union {
            __m128i f16_vec;
            simsimd_f16_t f16[8];
        } a_padded_tail, b_padded_tail;
        simsimd_size_t j = 0;
        for (; i < n; ++i, ++j)
            a_padded_tail.f16[j] = a[i], b_padded_tail.f16[j] = b[i];
        for (; j < 8; ++j)
            a_padded_tail.f16[j] = 0, b_padded_tail.f16[j] = 0;
        __m256 a_vec = _mm256_cvtph_ps(a_padded_tail.f16_vec);
        __m256 b_vec = _mm256_cvtph_ps(b_padded_tail.f16_vec);
        ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
    }

    ab_vec = _mm256_add_ps(_mm256_permute2f128_ps(ab_vec, ab_vec, 1), ab_vec);
    ab_vec = _mm256_hadd_ps(ab_vec, ab_vec);
    ab_vec = _mm256_hadd_ps(ab_vec, ab_vec);

    simsimd_f32_t f32_result;
    _mm_store_ss(&f32_result, _mm256_castps256_ps128(ab_vec));
    *result = f32_result;
}

simsimd_f32_t _mm_reduce_add_ps(__m128 vec) {
    __m128 sum = _mm_hadd_ps(vec, vec);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

simsimd_f32_t _mm256_reduce_add_ps(__m256 vec) {
    __m128 low = _mm256_castps256_ps128(vec);
    __m128 high = _mm256_extractf128_ps(vec, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2,f16c,fma"))) //
inline static void
simsimd_dot_f32c_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                         simsimd_distance_t* results) {

    // The naive approach would be to use FMA and FMS instructions on different parts of the vectors.
    // Prior to that we would need to shuffle the input vectors to separate real and imaginary parts.
    // Both operations are quite expensive, and the resulting kernel would run at 2.5 GB/s.
    // __m128 ab_real_vec = _mm_setzero_ps();
    // __m128 ab_imag_vec = _mm_setzero_ps();
    // __m256i permute_vec = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    // simsimd_size_t i = 0;
    // for (; i + 8 <= n; i += 8) {
    //     __m256 a_vec = _mm256_loadu_ps(a + i);
    //     __m256 b_vec = _mm256_loadu_ps(b + i);
    //     __m256 a_shuffled = _mm256_permutevar8x32_ps(a_vec, permute_vec);
    //     __m256 b_shuffled = _mm256_permutevar8x32_ps(b_vec, permute_vec);
    //     __m128 a_real_vec = _mm256_extractf128_ps(a_shuffled, 0);
    //     __m128 a_imag_vec = _mm256_extractf128_ps(a_shuffled, 1);
    //     __m128 b_real_vec = _mm256_extractf128_ps(b_shuffled, 0);
    //     __m128 b_imag_vec = _mm256_extractf128_ps(b_shuffled, 1);
    //     ab_real_vec = _mm_fmadd_ps(a_real_vec, b_real_vec, ab_real_vec);
    //     ab_real_vec = _mm_fnmadd_ps(a_imag_vec, b_imag_vec, ab_real_vec);
    //     ab_imag_vec = _mm_fmadd_ps(a_real_vec, b_imag_vec, ab_imag_vec);
    //     ab_imag_vec = _mm_fmadd_ps(a_imag_vec, b_real_vec, ab_imag_vec);
    // }
    //
    // Instead, we take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    // Both operations are quite cheap, and the throughput doubles from 2.5 GB/s to 5 GB/s.
    __m256 ab_real_vec = _mm256_setzero_ps();
    __m256 ab_imag_vec = _mm256_setzero_ps();
    __m256i sign_flip_vec = _mm256_set1_epi64x(0x0000000080000000);
    __m256i swap_adjacent_f16s_vec = _mm256_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2, //
                                                     13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256i b_vec = _mm256_castps_si256(_mm256_loadu_ps(b + i));
        ab_real_vec = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_xor_si256(b_vec, sign_flip_vec)), a_vec, ab_real_vec);
        ab_imag_vec = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_shuffle_epi8(b_vec, swap_adjacent_f16s_vec)), a_vec,
                                      ab_imag_vec);
    }

    // Reduce horizontal sums:
    simsimd_f32_t ab_real = _mm256_reduce_add_ps(ab_real_vec);
    simsimd_f32_t ab_imag = _mm256_reduce_add_ps(ab_imag_vec);

    // Handle the tail:
    for (; i + 2 <= n; i += 2) {
        simsimd_f32_t ar = a[i], ai = a[i + 1], br = b[i], bi = b[i + 1];
        ab_real += ar * br - ai * bi;
        ab_imag += ar * bi + ai * br;
    }
    results[0] = ab_real;
    results[1] = ab_imag;
}

__attribute__((target("avx2,f16c,fma"))) //
inline static void
simsimd_vdot_f32c_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                          simsimd_distance_t* results) {
    __m128 ab_real_vec = _mm_setzero_ps();
    __m128 ab_imag_vec = _mm_setzero_ps();

    // Define indices for even and odd extraction
    __m256i permute_vec = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);

        // Shuffle to separate even (real) and odd (imaginary) elements
        __m256 a_shuffled = _mm256_permutevar8x32_ps(a_vec, permute_vec);
        __m256 b_shuffled = _mm256_permutevar8x32_ps(b_vec, permute_vec);

        // Extract the real and imaginary parts
        __m128 a_real_vec = _mm256_extractf128_ps(a_shuffled, 0);
        __m128 a_imag_vec = _mm256_extractf128_ps(a_shuffled, 1);
        __m128 b_real_vec = _mm256_extractf128_ps(b_shuffled, 0);
        __m128 b_imag_vec = _mm256_extractf128_ps(b_shuffled, 1);

        // Compute the dot product:
        ab_real_vec = _mm_fmadd_ps(a_real_vec, b_real_vec, ab_real_vec);
        ab_real_vec = _mm_fmadd_ps(a_imag_vec, b_imag_vec, ab_real_vec);
        ab_imag_vec = _mm_fmadd_ps(a_real_vec, b_imag_vec, ab_imag_vec);
        ab_imag_vec = _mm_fnmadd_ps(a_imag_vec, b_real_vec, ab_imag_vec);
    }

    // Reduce horizontal sums:
    simsimd_f32_t ab_real = _mm_reduce_add_ps(ab_real_vec);
    simsimd_f32_t ab_imag = _mm_reduce_add_ps(ab_imag_vec);

    // Handle the tail:
    for (; i + 2 <= n; i += 2) {
        simsimd_f32_t ar = a[i], ai = a[i + 1], br = b[i], bi = b[i + 1];
        ab_real += ar * br + ai * bi;
        ab_imag += ar * bi - ai * br;
    }
    results[0] = ab_real;
    results[1] = ab_imag;
}

__attribute__((target("avx2,f16c,fma"))) //
inline static void
simsimd_dot_f16c_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                         simsimd_distance_t* results) {

    // Ideally the implementation would load 256 bits worth of vector data at a time,
    // shuffle those within a register, split in halfs, and only then upcast.
    // That way, we are stepping through 32x 16-bit vector components at a time, or 16 dimensions.
    // Sadly, shuffling 16-bit entries in a YMM register is hard to implement efficiently.
    //
    // Simpler approach is to load 128 bits at a time, upcast, and then shuffle.
    // This mostly replicates the `simsimd_dot_f32c_haswell`.
    __m128 ab_real_vec = _mm_setzero_ps();
    __m128 ab_imag_vec = _mm_setzero_ps();

    // Define indices for even and odd extraction
    __m256i permute_vec = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));

        // Shuffle to separate even (real) and odd (imaginary) elements
        __m256 a_shuffled = _mm256_permutevar8x32_ps(a_vec, permute_vec);
        __m256 b_shuffled = _mm256_permutevar8x32_ps(b_vec, permute_vec);

        // Extract the real and imaginary parts
        __m128 a_real_vec = _mm256_extractf128_ps(a_shuffled, 0);
        __m128 a_imag_vec = _mm256_extractf128_ps(a_shuffled, 1);
        __m128 b_real_vec = _mm256_extractf128_ps(b_shuffled, 0);
        __m128 b_imag_vec = _mm256_extractf128_ps(b_shuffled, 1);

        // Compute the dot product:
        ab_real_vec = _mm_fmadd_ps(a_real_vec, b_real_vec, ab_real_vec);
        ab_real_vec = _mm_fnmadd_ps(a_imag_vec, b_imag_vec, ab_real_vec);
        ab_imag_vec = _mm_fmadd_ps(a_real_vec, b_imag_vec, ab_imag_vec);
        ab_imag_vec = _mm_fmadd_ps(a_imag_vec, b_real_vec, ab_imag_vec);
    }

    // Reduce horizontal sums:
    simsimd_f32_t ab_real = _mm_reduce_add_ps(ab_real_vec);
    simsimd_f32_t ab_imag = _mm_reduce_add_ps(ab_imag_vec);

    // Handle the tail:
    for (; i + 2 <= n; i += 2) {
        simsimd_f32_t ar = a[i], ai = a[i + 1], br = b[i], bi = b[i + 1];
        ab_real += ar * br - ai * bi;
        ab_imag += ar * bi + ai * br;
    }
    results[0] = ab_real;
    results[1] = ab_imag;
}

__attribute__((target("avx2,f16c,fma"))) //
inline static void
simsimd_vdot_f16c_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                          simsimd_distance_t* results) {

    // Ideally the implementation would load 256 bits worth of vector data at a time,
    // shuffle those within a register, split in halfs, and only then upcast.
    // That way, we are stepping through 32x 16-bit vector components at a time, or 16 dimensions.
    // Sadly, shuffling 16-bit entries in a YMM register is hard to implement efficiently.
    //
    // Simpler approach is to load 128 bits at a time, upcast, and then shuffle.
    // This mostly replicates the `simsimd_dot_f32c_haswell`.
    __m128 ab_real_vec = _mm_setzero_ps();
    __m128 ab_imag_vec = _mm_setzero_ps();

    // Define indices for even and odd extraction
    __m256i permute_vec = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));

        // Shuffle to separate even (real) and odd (imaginary) elements
        __m256 a_shuffled = _mm256_permutevar8x32_ps(a_vec, permute_vec);
        __m256 b_shuffled = _mm256_permutevar8x32_ps(b_vec, permute_vec);

        // Extract the real and imaginary parts
        __m128 a_real_vec = _mm256_extractf128_ps(a_shuffled, 0);
        __m128 a_imag_vec = _mm256_extractf128_ps(a_shuffled, 1);
        __m128 b_real_vec = _mm256_extractf128_ps(b_shuffled, 0);
        __m128 b_imag_vec = _mm256_extractf128_ps(b_shuffled, 1);

        // Compute the dot product:
        ab_real_vec = _mm_fmadd_ps(a_real_vec, b_real_vec, ab_real_vec);
        ab_real_vec = _mm_fmadd_ps(a_imag_vec, b_imag_vec, ab_real_vec);
        ab_imag_vec = _mm_fmadd_ps(a_real_vec, b_imag_vec, ab_imag_vec);
        ab_imag_vec = _mm_fnmadd_ps(a_imag_vec, b_real_vec, ab_imag_vec);
    }

    // Reduce horizontal sums:
    simsimd_f32_t ab_real = _mm_reduce_add_ps(ab_real_vec);
    simsimd_f32_t ab_imag = _mm_reduce_add_ps(ab_imag_vec);

    // Handle the tail:
    for (; i + 2 <= n; i += 2) {
        simsimd_f32_t ar = a[i], ai = a[i + 1], br = b[i], bi = b[i + 1];
        ab_real += ar * br + ai * bi;
        ab_imag += ar * bi - ai * br;
    }
    results[0] = ab_real;
    results[1] = ab_imag;
}

#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE

__attribute__((target("avx512f,avx512vl,bmi2"))) //
inline static void
simsimd_dot_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    __m512 ab_vec = _mm512_setzero();
    __m512 a_vec, b_vec;

simsimd_dot_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = _bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    } else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    ab_vec = _mm512_fmadd_ps(a_vec, b_vec, ab_vec);
    if (n)
        goto simsimd_dot_f32_skylake_cycle;

    *result = _mm512_reduce_add_ps(ab_vec);
}

__attribute__((target("avx512f,avx512vl,bmi2"))) //
inline static void
simsimd_dot_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    __m512d ab_vec = _mm512_setzero_pd();
    __m512d a_vec, b_vec;

simsimd_dot_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = _bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_pd(mask, a);
        b_vec = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    } else {
        a_vec = _mm512_loadu_pd(a);
        b_vec = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    ab_vec = _mm512_fmadd_pd(a_vec, b_vec, ab_vec);
    if (n)
        goto simsimd_dot_f64_skylake_cycle;

    *result = _mm512_reduce_add_pd(ab_vec);
}

__attribute__((target("avx512f,avx512vl,bmi2"))) //
inline static void
simsimd_dot_f32c_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
}

__attribute__((target("avx512f,avx512vl,bmi2"))) //
inline static void
simsimd_vdot_f32c_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                          simsimd_distance_t* result) {}

__attribute__((target("avx512f,avx512vl,bmi2"))) //
inline static void
simsimd_dot_f64c_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
}

__attribute__((target("avx512f,avx512vl,bmi2"))) //
inline static void
simsimd_vdot_f64c_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                          simsimd_distance_t* result) {}

#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_SAPPHIRE

__attribute__((target("avx512fp16,avx512vl,avx512f,bmi2"))) //
inline static void
simsimd_dot_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    __m512h ab_vec = _mm512_setzero_ph();
    __m512i a_i16_vec, b_i16_vec;

simsimd_dot_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = _bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    } else {
        a_i16_vec = _mm512_loadu_epi16(a);
        b_i16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    ab_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_i16_vec), _mm512_castsi512_ph(b_i16_vec), ab_vec);
    if (n)
        goto simsimd_dot_f16_sapphire_cycle;

    *result = _mm512_reduce_add_ph(ab_vec);
}

#endif // SIMSIMD_TARGET_SAPPHIRE

#endif // SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif

#endif
