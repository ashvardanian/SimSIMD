/**
 *  @file       curved.h
 *  @brief      SIMD-accelerated Similarity Measures for curved spaces.
 *  @author     Ash Vardanian
 *  @date       August 27, 2024
 *
 *  Contains:
 *  - Bilinear form multiplication
 *  - Mahalanobis distance
 *
 *  For datatypes:
 *  - 32-bit floating point numbers
 *  - 16-bit floating point numbers
 *  - 16-bit brain-floating point numbers
 *
 *  For hardware architectures:
 *  - Arm (NEON, SVE)
 *  - x86 (AVX2, AVX512)
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef SIMSIMD_CURVED_H
#define SIMSIMD_CURVED_H

#include "types.h"

#include "dot.h"     // `_simsimd_partial_load_f16x4_neon` and friends
#include "spatial.h" // `_simsimd_substract_bf16x32_genoa`

#ifdef __cplusplus
extern "C" {
#endif

// clang-format off

/*  Serial backends for all numeric types.
 *  By default they use 32-bit arithmetic, unless the arguments themselves contain 64-bit floats.
 *  For double-precision computation check out the "*_accurate" variants of those "*_serial" functions.
 */
SIMSIMD_PUBLIC void simsimd_bilinear_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_f64_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_f64_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_bilinear_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_bilinear_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_bilinear_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result);

/*  Double-precision serial backends for all numeric types.
 *  For single-precision computation check out the "*_serial" counterparts of those "*_accurate" functions.
 */
SIMSIMD_PUBLIC void simsimd_bilinear_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_bilinear_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_bilinear_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result);

/*  SIMD-powered backends for Arm NEON, mostly using 32-bit arithmetic over 128-bit words.
 *  By far the most portable backend, covering most Arm v8 devices, over a billion phones, and almost all
 *  server CPUs produced before 2023.
 */
SIMSIMD_PUBLIC void simsimd_bilinear_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_bilinear_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_bilinear_bf16_neon(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_bf16_neon(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result);

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic over 256-bit words.
 *  First demonstrated in 2011, at least one Haswell-based processor was still being sold in 2022 — the Pentium G3420.
 *  Practically all modern x86 CPUs support AVX2, FMA, and F16C, making it a perfect baseline for SIMD algorithms.
 *  On other hand, there is no need to implement AVX2 versions of `f32` and `f64` functions, as those are
 *  properly vectorized by recent compilers.
 */
SIMSIMD_PUBLIC void simsimd_bilinear_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_bilinear_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result);

/*  SIMD-powered backends for various generations of AVX512 CPUs.
 *  Skylake is handy, as it supports masked loads and other operations, avoiding the need for the tail loop.
 *  Ice Lake added VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG, VPCLMULQDQ, and other extensions for integral operations.
 *  Sapphire Rapids added tiled matrix operations, but we are most interested in the new mixed-precision FMA instructions.
 */
SIMSIMD_PUBLIC void simsimd_bilinear_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_bilinear_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_bilinear_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, simsimd_size_t n, simsimd_distance_t* result);


SIMSIMD_PUBLIC void simsimd_bilinear_bf16_sapphire(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_mahalanobis_bf16_sapphire(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result);
// clang-format on

#define SIMSIMD_MAKE_BILINEAR(name, input_type, accumulator_type, load_and_convert)                                    \
    SIMSIMD_PUBLIC void simsimd_bilinear_##input_type##_##name(                                                        \
        simsimd_##input_type##_t const* a, simsimd_##input_type##_t const* b, simsimd_##input_type##_t const* c,       \
        simsimd_size_t n, simsimd_distance_t* result) {                                                                \
        simsimd_##accumulator_type##_t sum = 0;                                                                        \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                      \
            simsimd_##accumulator_type##_t partial = 0;                                                                \
            simsimd_##accumulator_type##_t a_i = load_and_convert(a + i);                                              \
            for (simsimd_size_t j = 0; j != n; ++j) {                                                                  \
                simsimd_##accumulator_type##_t b_j = load_and_convert(b + j);                                          \
                simsimd_##accumulator_type##_t c_ij = load_and_convert(c + i * n + j);                                 \
                partial += c_ij * b_j;                                                                                 \
            }                                                                                                          \
            sum += a_i * partial;                                                                                      \
        }                                                                                                              \
        *result = (simsimd_distance_t)sum;                                                                             \
    }

#define SIMSIMD_MAKE_MAHALANOBIS(name, input_type, accumulator_type, load_and_convert)                                 \
    SIMSIMD_PUBLIC void simsimd_mahalanobis_##input_type##_##name(                                                     \
        simsimd_##input_type##_t const* a, simsimd_##input_type##_t const* b, simsimd_##input_type##_t const* c,       \
        simsimd_size_t n, simsimd_distance_t* result) {                                                                \
        simsimd_##accumulator_type##_t sum = 0;                                                                        \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                      \
            simsimd_##accumulator_type##_t partial = 0;                                                                \
            simsimd_##accumulator_type##_t diff_i = load_and_convert(a + i) - load_and_convert(b + i);                 \
            for (simsimd_size_t j = 0; j != n; ++j) {                                                                  \
                simsimd_##accumulator_type##_t diff_j = load_and_convert(a + j) - load_and_convert(b + j);             \
                simsimd_##accumulator_type##_t c_ij = load_and_convert(c + i * n + j);                                 \
                partial += c_ij * diff_j;                                                                              \
            }                                                                                                          \
            sum += diff_i * partial;                                                                                   \
        }                                                                                                              \
        *result = (simsimd_distance_t)sum;                                                                             \
    }

SIMSIMD_MAKE_BILINEAR(serial, f64, f64, SIMSIMD_DEREFERENCE)    // simsimd_bilinear_f64_serial
SIMSIMD_MAKE_MAHALANOBIS(serial, f64, f64, SIMSIMD_DEREFERENCE) // simsimd_mahalanobis_f64_serial

SIMSIMD_MAKE_BILINEAR(serial, f32, f32, SIMSIMD_DEREFERENCE)    // simsimd_bilinear_f32_serial
SIMSIMD_MAKE_MAHALANOBIS(serial, f32, f32, SIMSIMD_DEREFERENCE) // simsimd_mahalanobis_f32_serial

SIMSIMD_MAKE_BILINEAR(serial, f16, f32, SIMSIMD_F16_TO_F32)    // simsimd_bilinear_f16_serial
SIMSIMD_MAKE_MAHALANOBIS(serial, f16, f32, SIMSIMD_F16_TO_F32) // simsimd_mahalanobis_f16_serial

SIMSIMD_MAKE_BILINEAR(serial, bf16, f32, SIMSIMD_BF16_TO_F32)    // simsimd_bilinear_bf16_serial
SIMSIMD_MAKE_MAHALANOBIS(serial, bf16, f32, SIMSIMD_BF16_TO_F32) // simsimd_mahalanobis_bf16_serial

SIMSIMD_MAKE_BILINEAR(accurate, f32, f64, SIMSIMD_DEREFERENCE)    // simsimd_bilinear_f32_accurate
SIMSIMD_MAKE_MAHALANOBIS(accurate, f32, f64, SIMSIMD_DEREFERENCE) // simsimd_mahalanobis_f32_accurate

SIMSIMD_MAKE_BILINEAR(accurate, f16, f64, SIMSIMD_F16_TO_F32)    // simsimd_bilinear_f16_accurate
SIMSIMD_MAKE_MAHALANOBIS(accurate, f16, f64, SIMSIMD_F16_TO_F32) // simsimd_mahalanobis_f16_accurate

SIMSIMD_MAKE_BILINEAR(accurate, bf16, f64, SIMSIMD_BF16_TO_F32)    // simsimd_bilinear_bf16_accurate
SIMSIMD_MAKE_MAHALANOBIS(accurate, bf16, f64, SIMSIMD_BF16_TO_F32) // simsimd_mahalanobis_bf16_accurate

#if SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_bilinear_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c,
                                              simsimd_size_t n, simsimd_distance_t* result) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    for (simsimd_size_t i = 0; i != n; ++i) {
        float32x4_t a_vec = vdupq_n_f32(a[i]);
        float32x4_t partial_sum_vec = vdupq_n_f32(0);
        for (simsimd_size_t j = 0; j + 4 <= n; j += 4) {
            float32x4_t b_vec = vld1q_f32(b + j);
            float32x4_t c_vec = vld1q_f32(c + i * n + j);
            partial_sum_vec = vmlaq_f32(partial_sum_vec, b_vec, c_vec);
        }
        sum_vec = vmlaq_f32(sum_vec, a_vec, partial_sum_vec);
    }

    // Handle the tail of every row
    simsimd_f64_t sum = vaddvq_f32(sum_vec);
    simsimd_size_t tail_length = n % 4;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t a_i = a[i];
            simsimd_f32_t partial_sum = 0;
            for (simsimd_size_t j = tail_start; j != n; ++j)
                partial_sum += b[j] * c[i * n + j];
            sum += a[i] * partial_sum;
        }
    }

    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_mahalanobis_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c,
                                                 simsimd_size_t n, simsimd_distance_t* result) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    for (simsimd_size_t i = 0; i != n; ++i) {
        float32x4_t diff_i_vec = vdupq_n_f32(a[i] - b[i]);
        float32x4_t partial_sum_vec = vdupq_n_f32(0);
        for (simsimd_size_t j = 0; j + 4 <= n; j += 4) {
            float32x4_t diff_j_vec = vsubq_f32(vld1q_f32(a + j), vld1q_f32(b + j));
            float32x4_t c_vec = vld1q_f32(c + i * n + j);
            partial_sum_vec = vmlaq_f32(partial_sum_vec, diff_j_vec, c_vec);
        }

        sum_vec = vmlaq_f32(sum_vec, diff_i_vec, partial_sum_vec);
    }

    // Handle the tail of every row
    simsimd_f64_t sum = vaddvq_f32(sum_vec);
    simsimd_size_t tail_length = n % 4;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t diff_i = a[i] - b[i];
            simsimd_f32_t partial_sum = 0;
            for (simsimd_size_t j = tail_start; j != n; ++j) {
                simsimd_f32_t diff_j = a[j] - b[j];
                partial_sum += diff_j * c[i * n + j];
            }
            sum += diff_i * partial_sum;
        }
    }

    *result = sum;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_bilinear_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c,
                                              simsimd_size_t n, simsimd_distance_t* result) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    for (simsimd_size_t i = 0; i != n; ++i) {
        // MSVC doesn't recognize `vdup_n_f16` as a valid intrinsic
        float32x4_t a_vec = vcvt_f32_f16(vreinterpret_f16_s16(vdup_n_s16(*(short const*)(a + i))));
        float32x4_t partial_sum_vec = vdupq_n_f32(0);
        for (simsimd_size_t j = 0; j + 4 <= n; j += 4) {
            float32x4_t b_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const*)(b + j)));
            float32x4_t c_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const*)(c + i * n + j)));
            partial_sum_vec = vmlaq_f32(partial_sum_vec, b_vec, c_vec);
        }
        sum_vec = vmlaq_f32(sum_vec, a_vec, partial_sum_vec);
    }

    // Handle the tail of every row
    simsimd_f64_t sum = vaddvq_f32(sum_vec);
    simsimd_size_t tail_length = n % 4;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t a_i = vaddvq_f32(vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(a + i, 1)));
            float32x4_t b_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(b + tail_start, tail_length));
            float32x4_t c_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(c + i * n + tail_start, tail_length));
            simsimd_f32_t partial_sum = vaddvq_f32(vmulq_f32(b_vec, c_vec));
            sum += a_i * partial_sum;
        }
    }

    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_mahalanobis_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c,
                                                 simsimd_size_t n, simsimd_distance_t* result) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    for (simsimd_size_t i = 0; i != n; ++i) {
        // MSVC doesn't recognize `vdup_n_f16` as a valid intrinsic
        float32x4_t a_i_vec = vcvt_f32_f16(vreinterpret_f16_s16(vdup_n_s16(*(short const*)(a + i))));
        float32x4_t b_i_vec = vcvt_f32_f16(vreinterpret_f16_s16(vdup_n_s16(*(short const*)(b + i))));
        float32x4_t diff_i_vec = vsubq_f32(a_i_vec, b_i_vec);
        float32x4_t partial_sum_vec = vdupq_n_f32(0);
        for (simsimd_size_t j = 0; j + 4 <= n; j += 4) {
            float32x4_t a_j_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const*)(a + j)));
            float32x4_t b_j_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const*)(b + j)));
            float32x4_t diff_j_vec = vsubq_f32(a_j_vec, b_j_vec);
            float32x4_t c_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const*)(c + i * n + j)));
            partial_sum_vec = vmlaq_f32(partial_sum_vec, diff_j_vec, c_vec);
        }
        sum_vec = vmlaq_f32(sum_vec, diff_i_vec, partial_sum_vec);
    }

    // Handle the tail of every row
    simsimd_f64_t sum = vaddvq_f32(sum_vec);
    simsimd_size_t tail_length = n % 4;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t a_i = vaddvq_f32(vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(a + i, 1)));
            simsimd_f32_t b_i = vaddvq_f32(vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(b + i, 1)));
            simsimd_f32_t diff_i = a_i - b_i;
            float32x4_t a_j_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(a + tail_start, tail_length));
            float32x4_t b_j_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(b + tail_start, tail_length));
            float32x4_t diff_j_vec = vsubq_f32(a_j_vec, b_j_vec);
            float32x4_t c_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(c + i * n + tail_start, tail_length));
            simsimd_f32_t partial_sum = vaddvq_f32(vmulq_f32(diff_j_vec, c_vec));
            sum += diff_i * partial_sum;
        }
    }

    *result = sum;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_F16

#if SIMSIMD_TARGET_NEON_BF16
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_bilinear_bf16_neon(simsimd_bf16_t const* a, simsimd_bf16_t const* b,
                                               simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    for (simsimd_size_t i = 0; i != n; ++i) {
        float32x4_t a_vec = vdupq_n_f32(simsimd_bf16_to_f32(a + i));
        float32x4_t partial_sum_vec = vdupq_n_f32(0);
        for (simsimd_size_t j = 0; j + 8 <= n; j += 8) {
            bfloat16x8_t b_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)(b + j));
            bfloat16x8_t c_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)(c + i * n + j));
            partial_sum_vec = vbfdotq_f32(partial_sum_vec, b_vec, c_vec);
        }
        sum_vec = vmlaq_f32(sum_vec, a_vec, partial_sum_vec);
    }

    // Handle the tail of every row
    simsimd_f64_t sum = vaddvq_f32(sum_vec);
    simsimd_size_t tail_length = n % 8;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t a_i = simsimd_bf16_to_f32(a + i);
            bfloat16x8_t b_vec = _simsimd_partial_load_bf16x8_neon(b + tail_start, tail_length);
            bfloat16x8_t c_vec = _simsimd_partial_load_bf16x8_neon(c + i * n + tail_start, tail_length);
            simsimd_f32_t partial_sum = vaddvq_f32(vbfdotq_f32(vdupq_n_f32(0), b_vec, c_vec));
            sum += a_i * partial_sum;
        }
    }

    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_mahalanobis_bf16_neon(simsimd_bf16_t const* a, simsimd_bf16_t const* b,
                                                  simsimd_bf16_t const* c, simsimd_size_t n,
                                                  simsimd_distance_t* result) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    for (simsimd_size_t i = 0; i != n; ++i) {
        simsimd_f32_t a_i = simsimd_bf16_to_f32(a + i);
        simsimd_f32_t b_i = simsimd_bf16_to_f32(b + i);
        float32x4_t diff_i_vec = vdupq_n_f32(a_i - b_i);
        float32x4_t partial_sum_vec = vdupq_n_f32(0);
        for (simsimd_size_t j = 0; j + 8 <= n; j += 8) {
            bfloat16x8_t a_j_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)(a + j));
            bfloat16x8_t b_j_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)(b + j));

            // Arm NEON does not have a native subtraction instruction for `bf16`,
            // so we need to convert to `f32` first, subtract, and only then get back to `bf16`
            // for multiplication.
            float32x4_t a_j_vec_high = vcvt_f32_bf16(vget_high_bf16(a_j_vec));
            float32x4_t a_j_vec_low = vcvt_f32_bf16(vget_low_bf16(a_j_vec));
            float32x4_t b_j_vec_high = vcvt_f32_bf16(vget_high_bf16(b_j_vec));
            float32x4_t b_j_vec_low = vcvt_f32_bf16(vget_low_bf16(b_j_vec));
            float32x4_t diff_j_vec_high = vsubq_f32(a_j_vec_high, b_j_vec_high);
            float32x4_t diff_j_vec_low = vsubq_f32(a_j_vec_low, b_j_vec_low);
            bfloat16x8_t diff_j_vec = vcombine_bf16(vcvt_bf16_f32(diff_j_vec_low), vcvt_bf16_f32(diff_j_vec_high));

            bfloat16x8_t c_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)(c + i * n + j));
            partial_sum_vec = vbfdotq_f32(partial_sum_vec, diff_j_vec, c_vec);
        }
        sum_vec = vmlaq_f32(sum_vec, diff_i_vec, partial_sum_vec);
    }

    // Handle the tail of every row
    simsimd_f64_t sum = vaddvq_f32(sum_vec);
    simsimd_size_t tail_length = n % 8;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t a_i = simsimd_bf16_to_f32(a + i);
            simsimd_f32_t b_i = simsimd_bf16_to_f32(b + i);
            simsimd_f32_t diff_i = a_i - b_i;
            bfloat16x8_t a_j_vec = _simsimd_partial_load_bf16x8_neon(a + tail_start, tail_length);
            bfloat16x8_t b_j_vec = _simsimd_partial_load_bf16x8_neon(b + tail_start, tail_length);

            // Again, upcast for subtraction
            float32x4_t a_j_vec_high = vcvt_f32_bf16(vget_high_bf16(a_j_vec));
            float32x4_t a_j_vec_low = vcvt_f32_bf16(vget_low_bf16(a_j_vec));
            float32x4_t b_j_vec_high = vcvt_f32_bf16(vget_high_bf16(b_j_vec));
            float32x4_t b_j_vec_low = vcvt_f32_bf16(vget_low_bf16(b_j_vec));
            float32x4_t diff_j_vec_high = vsubq_f32(a_j_vec_high, b_j_vec_high);
            float32x4_t diff_j_vec_low = vsubq_f32(a_j_vec_low, b_j_vec_low);
            bfloat16x8_t diff_j_vec = vcombine_bf16(vcvt_bf16_f32(diff_j_vec_low), vcvt_bf16_f32(diff_j_vec_high));

            bfloat16x8_t c_vec = _simsimd_partial_load_bf16x8_neon(c + i * n + tail_start, tail_length);
            simsimd_f32_t partial_sum = vaddvq_f32(vbfdotq_f32(vdupq_n_f32(0), diff_j_vec, c_vec));
            sum += diff_i * partial_sum;
        }
    }

    *result = sum;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_BF16

#endif // SIMSIMD_TARGET_ARM

#if SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_bilinear_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c,
                                                 simsimd_size_t n, simsimd_distance_t* result) {
    __m256 sum_vec = _mm256_setzero_ps();
    for (simsimd_size_t i = 0; i != n; ++i) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_set1_epi16(*(short const*)(a + i)));
        __m256 partial_sum_vec = _mm256_setzero_ps();
        for (simsimd_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + j)));
            __m256 c_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(c + i * n + j)));
            partial_sum_vec = _mm256_fmadd_ps(b_vec, c_vec, partial_sum_vec);
        }
        sum_vec = _mm256_fmadd_ps(a_vec, partial_sum_vec, sum_vec);
    }

    // Handle the tail of every row
    simsimd_f64_t sum = _simsimd_reduce_f32x8_haswell(sum_vec);
    simsimd_size_t tail_length = n % 8;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t a_i = _mm256_cvtss_f32(_mm256_cvtph_ps(_mm_set1_epi16(*(short const*)(a + i))));
            __m256 b_vec = _simsimd_partial_load_f16x8_haswell(b + tail_start, tail_length);
            __m256 c_vec = _simsimd_partial_load_f16x8_haswell(c + i * n + tail_start, tail_length);
            simsimd_f32_t partial_sum = _simsimd_reduce_f32x8_haswell(_mm256_mul_ps(b_vec, c_vec));
            sum += a_i * partial_sum;
        }
    }

    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_mahalanobis_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b,
                                                    simsimd_f16_t const* c, simsimd_size_t n,
                                                    simsimd_distance_t* result) {
    __m256 sum_vec = _mm256_setzero_ps();
    for (simsimd_size_t i = 0; i != n; ++i) {
        __m256 diff_i_vec = _mm256_sub_ps(                           //
            _mm256_cvtph_ps(_mm_set1_epi16(*(short const*)(a + i))), //
            _mm256_cvtph_ps(_mm_set1_epi16(*(short const*)(b + i))));
        __m256 partial_sum_vec = _mm256_setzero_ps();
        for (simsimd_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 diff_j_vec = _mm256_sub_ps( //
                _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + j))),
                _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + j))));
            __m256 c_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(c + i * n + j)));
            partial_sum_vec = _mm256_fmadd_ps(diff_j_vec, c_vec, partial_sum_vec);
        }
        sum_vec = _mm256_fmadd_ps(diff_i_vec, partial_sum_vec, sum_vec);
    }

    // Handle the tail of every row
    simsimd_f64_t sum = _simsimd_reduce_f32x8_haswell(sum_vec);
    simsimd_size_t tail_length = n % 8;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t diff_i = _mm256_cvtss_f32(_mm256_sub_ps(       //
                _mm256_cvtph_ps(_mm_set1_epi16(*(short const*)(a + i))), //
                _mm256_cvtph_ps(_mm_set1_epi16(*(short const*)(b + i)))));
            __m256 diff_j_vec = _mm256_sub_ps( //
                _simsimd_partial_load_f16x8_haswell(a + tail_start, tail_length),
                _simsimd_partial_load_f16x8_haswell(b + tail_start, tail_length));
            __m256 c_vec = _simsimd_partial_load_f16x8_haswell(c + i * n + tail_start, tail_length);
            simsimd_f32_t partial_sum = _simsimd_reduce_f32x8_haswell(_mm256_mul_ps(diff_j_vec, c_vec));
            sum += diff_i * partial_sum;
        }
    }

    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_bilinear_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b,
                                                  simsimd_bf16_t const* c, simsimd_size_t n,
                                                  simsimd_distance_t* result) {
    __m256 sum_vec = _mm256_setzero_ps();
    for (simsimd_size_t i = 0; i != n; ++i) {
        // The `simsimd_bf16_to_f32` is cheaper than `_simsimd_bf16x8_to_f32x8_haswell`
        __m256 a_vec = _mm256_set1_ps(simsimd_bf16_to_f32(a + i));
        __m256 partial_sum_vec = _mm256_setzero_ps();
        for (simsimd_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 b_vec = _simsimd_bf16x8_to_f32x8_haswell(_mm_loadu_si128((__m128i const*)(b + j)));
            __m256 c_vec = _simsimd_bf16x8_to_f32x8_haswell(_mm_loadu_si128((__m128i const*)(c + i * n + j)));
            partial_sum_vec = _mm256_fmadd_ps(b_vec, c_vec, partial_sum_vec);
        }
        sum_vec = _mm256_fmadd_ps(a_vec, partial_sum_vec, sum_vec);
    }

    // Handle the tail of every row
    simsimd_f64_t sum = _simsimd_reduce_f32x8_haswell(sum_vec);
    simsimd_size_t tail_length = n % 8;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t a_i = simsimd_bf16_to_f32(a + i);
            __m256 b_vec = _simsimd_bf16x8_to_f32x8_haswell( //
                _simsimd_partial_load_bf16x8_haswell(b + tail_start, tail_length));
            __m256 c_vec = _simsimd_bf16x8_to_f32x8_haswell( //
                _simsimd_partial_load_bf16x8_haswell(c + i * n + tail_start, tail_length));
            simsimd_f32_t partial_sum = _simsimd_reduce_f32x8_haswell(_mm256_mul_ps(b_vec, c_vec));
            sum += a_i * partial_sum;
        }
    }

    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_mahalanobis_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b,
                                                     simsimd_bf16_t const* c, simsimd_size_t n,
                                                     simsimd_distance_t* result) {
    __m256 sum_vec = _mm256_setzero_ps();
    for (simsimd_size_t i = 0; i != n; ++i) {
        __m256 diff_i_vec = _mm256_sub_ps(              //
            _mm256_set1_ps(simsimd_bf16_to_f32(a + i)), //
            _mm256_set1_ps(simsimd_bf16_to_f32(b + i)));
        __m256 partial_sum_vec = _mm256_setzero_ps();
        for (simsimd_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 diff_j_vec = _mm256_sub_ps(                                              //
                _simsimd_bf16x8_to_f32x8_haswell(_mm_loadu_si128((__m128i const*)(a + j))), //
                _simsimd_bf16x8_to_f32x8_haswell(_mm_loadu_si128((__m128i const*)(b + j))));
            __m256 c_vec = _simsimd_bf16x8_to_f32x8_haswell(_mm_loadu_si128((__m128i const*)(c + i * n + j)));
            partial_sum_vec = _mm256_fmadd_ps(diff_j_vec, c_vec, partial_sum_vec);
        }
        sum_vec = _mm256_fmadd_ps(diff_i_vec, partial_sum_vec, sum_vec);
    }

    // Handle the tail of every row
    simsimd_f64_t sum = _simsimd_reduce_f32x8_haswell(sum_vec);
    simsimd_size_t tail_length = n % 8;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t diff_i = simsimd_bf16_to_f32(a + i) - simsimd_bf16_to_f32(b + i);
            __m256 diff_j_vec = _mm256_sub_ps( //
                _simsimd_bf16x8_to_f32x8_haswell(_simsimd_partial_load_bf16x8_haswell(a + tail_start, tail_length)),
                _simsimd_bf16x8_to_f32x8_haswell(_simsimd_partial_load_bf16x8_haswell(b + tail_start, tail_length)));
            __m256 c_vec = _simsimd_bf16x8_to_f32x8_haswell(
                _simsimd_partial_load_bf16x8_haswell(c + i * n + tail_start, tail_length));
            simsimd_f32_t partial_sum = _simsimd_reduce_f32x8_haswell(_mm256_mul_ps(diff_j_vec, c_vec));
            sum += diff_i * partial_sum;
        }
    }

    *result = sum;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_bilinear_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c,
                                                 simsimd_size_t n, simsimd_distance_t* result) {
    simsimd_size_t tail_length = n % 16;
    simsimd_size_t tail_start = n - tail_length;
    __m512 sum_vec = _mm512_setzero_ps();
    __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, tail_length);

    for (simsimd_size_t i = 0; i != n; ++i) {
        __m512 a_vec = _mm512_set1_ps(a[i]);
        __m512 partial_sum_vec = _mm512_setzero_ps();
        __m512 b_vec, c_vec;
        simsimd_size_t j = 0;

    simsimd_bilinear_f32_skylake_cycle:
        if (j + 16 <= n) {
            b_vec = _mm512_loadu_ps(b + j);
            c_vec = _mm512_loadu_ps(c + i * n + j);
        } else {
            b_vec = _mm512_maskz_loadu_ps(tail_mask, b + tail_start);
            c_vec = _mm512_maskz_loadu_ps(tail_mask, c + i * n + tail_start);
        }
        partial_sum_vec = _mm512_fmadd_ps(b_vec, c_vec, partial_sum_vec);
        j += 16;
        if (j < n)
            goto simsimd_bilinear_f32_skylake_cycle;
        sum_vec = _mm512_fmadd_ps(a_vec, partial_sum_vec, sum_vec);
    }

    *result = _mm512_reduce_add_ps(sum_vec);
}

SIMSIMD_PUBLIC void simsimd_mahalanobis_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b,
                                                    simsimd_f32_t const* c, simsimd_size_t n,
                                                    simsimd_distance_t* result) {
    simsimd_size_t tail_length = n % 16;
    simsimd_size_t tail_start = n - tail_length;
    __m512 sum_vec = _mm512_setzero_ps();
    __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, tail_length);

    for (simsimd_size_t i = 0; i != n; ++i) {
        __m512 diff_i_vec = _mm512_set1_ps(a[i] - b[i]);
        __m512 partial_sum_vec = _mm512_setzero_ps(), partial_sum_bot_vec = _mm512_setzero_ps();
        __m512 a_j_vec, b_j_vec, diff_j_vec, c_vec;
        simsimd_size_t j = 0;

        // The nested loop is cleaner to implement with a `goto` in this case:
    simsimd_bilinear_f32_skylake_cycle:
        if (j + 16 <= n) {
            a_j_vec = _mm512_loadu_ps(a + j);
            b_j_vec = _mm512_loadu_ps(b + j);
            c_vec = _mm512_loadu_ps(c + i * n + j);
        } else {
            a_j_vec = _mm512_maskz_loadu_ps(tail_mask, a + tail_start);
            b_j_vec = _mm512_maskz_loadu_ps(tail_mask, b + tail_start);
            c_vec = _mm512_maskz_loadu_ps(tail_mask, c + i * n + tail_start);
        }
        diff_j_vec = _mm512_sub_ps(a_j_vec, b_j_vec);
        partial_sum_vec = _mm512_fmadd_ps(diff_j_vec, c_vec, partial_sum_vec);
        j += 16;
        if (j < n)
            goto simsimd_bilinear_f32_skylake_cycle;
        sum_vec = _mm512_fmadd_ps(diff_i_vec, partial_sum_vec, sum_vec);
    }

    *result = _mm512_reduce_add_ps(sum_vec);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_GENOA
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512bf16")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512bf16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_bilinear_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b,
                                                simsimd_bf16_t const* c, simsimd_size_t n, simsimd_distance_t* result) {
    simsimd_size_t tail_length = n % 32;
    simsimd_size_t tail_start = n - tail_length;
    __m512 sum_vec = _mm512_setzero_ps();
    __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length);

    for (simsimd_size_t i = 0; i != n; ++i) {
        __m512 a_vec = _mm512_set1_ps(simsimd_bf16_to_f32(a + i));
        __m512 partial_sum_vec = _mm512_setzero_ps();
        __m512i b_vec, c_vec;
        simsimd_size_t j = 0;

    simsimd_bilinear_bf16_genoa_cycle:
        if (j + 32 <= n) {
            b_vec = _mm512_loadu_epi16(b + j);
            c_vec = _mm512_loadu_epi16(c + i * n + j);
        } else {
            b_vec = _mm512_maskz_loadu_epi16(tail_mask, b + tail_start);
            c_vec = _mm512_maskz_loadu_epi16(tail_mask, c + i * n + tail_start);
        }
        partial_sum_vec = _mm512_dpbf16_ps(partial_sum_vec, (__m512bh)(b_vec), (__m512bh)(c_vec));
        j += 32;
        if (j < n)
            goto simsimd_bilinear_bf16_genoa_cycle;
        sum_vec = _mm512_fmadd_ps(a_vec, partial_sum_vec, sum_vec);
    }

    *result = _mm512_reduce_add_ps(sum_vec);
}

SIMSIMD_PUBLIC void simsimd_mahalanobis_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b,
                                                   simsimd_bf16_t const* c, simsimd_size_t n,
                                                   simsimd_distance_t* result) {
    simsimd_size_t tail_length = n % 32;
    simsimd_size_t tail_start = n - tail_length;
    __m512 sum_vec = _mm512_setzero_ps();
    __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length);

    for (simsimd_size_t i = 0; i != n; ++i) {
        __m512 diff_i_vec = _mm512_set1_ps(simsimd_bf16_to_f32(a + i) - simsimd_bf16_to_f32(b + i));
        __m512 partial_sum_vec = _mm512_setzero_ps();
        __m512i a_j_vec, b_j_vec, diff_j_vec, c_vec;
        simsimd_size_t j = 0;

        // The nested loop is cleaner to implement with a `goto` in this case:
    simsimd_mahalanobis_bf16_genoa_cycle:
        if (j + 32 <= n) {
            a_j_vec = _mm512_loadu_epi16(a + j);
            b_j_vec = _mm512_loadu_epi16(b + j);
            c_vec = _mm512_loadu_epi16(c + i * n + j);
        } else {
            a_j_vec = _mm512_maskz_loadu_epi16(tail_mask, a + tail_start);
            b_j_vec = _mm512_maskz_loadu_epi16(tail_mask, b + tail_start);
            c_vec = _mm512_maskz_loadu_epi16(tail_mask, c + i * n + tail_start);
        }
        diff_j_vec = _simsimd_substract_bf16x32_genoa(a_j_vec, b_j_vec);
        partial_sum_vec = _mm512_dpbf16_ps(partial_sum_vec, (__m512bh)(diff_j_vec), (__m512bh)(c_vec));
        j += 32;
        if (j < n)
            goto simsimd_mahalanobis_bf16_genoa_cycle;
        sum_vec = _mm512_fmadd_ps(diff_i_vec, partial_sum_vec, sum_vec);
    }

    *result = _mm512_reduce_add_ps(sum_vec);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_GENOA

#if SIMSIMD_TARGET_SAPPHIRE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_bilinear_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b,
                                                  simsimd_f16_t const* c, simsimd_size_t n,
                                                  simsimd_distance_t* result) {
    simsimd_size_t tail_length = n % 32;
    simsimd_size_t tail_start = n - tail_length;
    __m512h sum_vec = _mm512_setzero_ph();
    __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length);

    for (simsimd_size_t i = 0; i != n; ++i) {
        __m512h a_vec = _mm512_castsi512_ph(_mm512_set1_epi16(*(short const*)(a + i)));
        __m512h partial_sum_vec = _mm512_setzero_ph();
        __m512i b_vec, c_vec;
        simsimd_size_t j = 0;

    simsimd_bilinear_f16_sapphire_cycle:
        if (j + 32 <= n) {
            b_vec = _mm512_loadu_epi16(b + j);
            c_vec = _mm512_loadu_epi16(c + i * n + j);
        } else {
            b_vec = _mm512_maskz_loadu_epi16(tail_mask, b + tail_start);
            c_vec = _mm512_maskz_loadu_epi16(tail_mask, c + i * n + tail_start);
        }
        partial_sum_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(b_vec), _mm512_castsi512_ph(c_vec), partial_sum_vec);
        j += 32;
        if (j < n)
            goto simsimd_bilinear_f16_sapphire_cycle;
        sum_vec = _mm512_fmadd_ph(a_vec, partial_sum_vec, sum_vec);
    }

    *result = _mm512_reduce_add_ph(sum_vec);
}

SIMSIMD_PUBLIC void simsimd_mahalanobis_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b,
                                                     simsimd_f16_t const* c, simsimd_size_t n,
                                                     simsimd_distance_t* result) {
    simsimd_size_t tail_length = n % 32;
    simsimd_size_t tail_start = n - tail_length;
    __m512h sum_vec = _mm512_setzero_ph();
    __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length);

    for (simsimd_size_t i = 0; i != n; ++i) {
        __m512h a_i_vec = _mm512_castsi512_ph(_mm512_set1_epi16(*(short const*)(a + i)));
        __m512h b_i_vec = _mm512_castsi512_ph(_mm512_set1_epi16(*(short const*)(b + i)));
        __m512h diff_i_vec = _mm512_sub_ph(a_i_vec, b_i_vec);
        __m512h partial_sum_vec = _mm512_setzero_ph();
        __m512h diff_j_vec;
        __m512i a_j_vec, b_j_vec, c_vec;
        simsimd_size_t j = 0;

        // The nested loop is cleaner to implement with a `goto` in this case:
    simsimd_mahalanobis_f16_sapphire_cycle:
        if (j + 32 <= n) {
            a_j_vec = _mm512_loadu_epi16(a + j);
            b_j_vec = _mm512_loadu_epi16(b + j);
            c_vec = _mm512_loadu_epi16(c + i * n + j);
        } else {
            a_j_vec = _mm512_maskz_loadu_epi16(tail_mask, a + tail_start);
            b_j_vec = _mm512_maskz_loadu_epi16(tail_mask, b + tail_start);
            c_vec = _mm512_maskz_loadu_epi16(tail_mask, c + i * n + tail_start);
        }
        diff_j_vec = _mm512_sub_ph(_mm512_castsi512_ph(a_j_vec), _mm512_castsi512_ph(b_j_vec));
        partial_sum_vec = _mm512_fmadd_ph(diff_j_vec, _mm512_castsi512_ph(c_vec), partial_sum_vec);
        j += 32;
        if (j < n)
            goto simsimd_mahalanobis_f16_sapphire_cycle;
        sum_vec = _mm512_fmadd_ph(diff_i_vec, partial_sum_vec, sum_vec);
    }

    *result = _mm512_reduce_add_ph(sum_vec);
}

SIMSIMD_PUBLIC void simsimd_bilinear_bf16_sapphire(simsimd_bf16_t const* a, simsimd_bf16_t const* b,
                                                   simsimd_bf16_t const* c, simsimd_size_t n,
                                                   simsimd_distance_t* result) {

    __m512 sum_vec = _mm512_setzero_ps();
    // Using Intel AMX instructions we can perform 16x32x16 brain-float multiplication using specialed
    // matrix-multiplication hardware, accumulating the results in a 16x16 single-precsion matrix.
    SIMSIMD_ALIGN64 simsimd_u8_t tiles_config[64];

    // Memset in one cycle, like a boss :)
    // std::memset(tiles_config, 0, sizeof(tiles_config));
    _mm512_storeu_si512((__m512i*)tiles_config, _mm512_setzero_si512());

    // Only one pallete is currently supported:
    simsimd_u8_t* palette_id_ptr = &tiles_config[0];
    *palette_id_ptr = 1;

    // The geniuses behind AMX decided to use different precisions for the rows and columns.
    // Wasted 2 hours of my life not noticing this!
    simsimd_u16_t* tiles_colsb_ptr = (simsimd_u16_t*)(&tiles_config[16]);
    simsimd_u8_t* tiles_rows_ptr = &tiles_config[48];

    // Important to note, AMX doesn't care about the real shape of our matrix,
    // it only cares about it's own tile shape. Keep it simple, otherwise
    // the next person reading this will be painting the walls with their brains :)
    tiles_rows_ptr[0] = 16;
    tiles_rows_ptr[1] = 16;
    tiles_rows_ptr[2] = 16;
    tiles_colsb_ptr[0] = 64;
    tiles_colsb_ptr[1] = 64;
    tiles_colsb_ptr[2] = 64;

    // Intialize the tiles configuration and zero-out the accumulators:
    _tile_loadconfig(&tiles_config);
    _tile_zero(2);

    // In every iteration we can load 16x32 = 512 values from the first vector.
    _tile_loadd(0, a, 64);

    // When multiplying matrix chains, like $X * Y * Z$, the order doesn't matter.
    // We can perform $(X * Y) * Z$ or $X * (Y * Z)$, the result will be the same.
    // The last variant is better in our case, as X and Z are much smaller than Y.
    //
    // The output of $YZ = Y * Z$ will be a single row in our case.
    simsimd_size_t const vertical_tiles = n / 16;
    simsimd_size_t const horizontal_tiles = n / 32;
    for (simsimd_size_t matrix_start_row = 0; matrix_start_row + 16 <= n; matrix_start_row += 16) {

        // The column vector `b` contains twice as many rows as the curvature matrix.
        simsimd_size_t vector_progress = matrix_start_row * 2;
        simsimd_bf16_t b_reshaped[16][16][2];
        for (simsimd_size_t i = 0; i < 32; ++i)
            for (simsimd_size_t j = 0; j < 16; ++j)
                // We are shrinking the number of rows in the second matrix by 2x for `bf16` and by 4x for `i8` and
                // `u8`. We are also practically making the rows longer by 2x for `bf16` and by 4x for `i8` and `u8`.
                b_reshaped[i / 2][j][i % 2] = b[vector_progress + i * 16 + j];
        _tile_loadd(1, &b_reshaped, 64);

        // Within a horizontal band of 16 rows in a curvature matrix,
        // we can compute 16 items of the YZ row in parallel, progressing
        // aggregating 32 pairs into each of the 16 items.
        for (simsimd_size_t matrix_start_col = 0; matrix_start_col + 32 <= n; matrix_start_col += 32) {
            // Load a 16x32 block from the curvature matrix.
            _tile_loadd(0, c + matrix_start_row * n + matrix_start_col, 64);
            // Perform the matrix multiplication.
            _tile_dpbf16ps(2, 0, 1);
        }

        simsimd_f32_t bc_expanded[16][16];
        _tile_stored(2, &bc_expanded, 64);

        // Now we need to accumulate the results in every column into the top row.
        // Each row is exactly 64 bytes or 512 bits, making it perfect for AVX-512.
        __m512 bc_collapsed_vec = _mm512_setzero_ps();
        for (simsimd_size_t i = 0; i < 16; ++i) {
            __m512 bc_row_vec = _mm512_loadu_ps(bc_expanded[i]);
            bc_collapsed_vec = _mm512_add_ps(bc_collapsed_vec, bc_row_vec);
        }

        // At this point the `bc_collapsed_vec` contains the product of a band of Y multiplied by the column Z.
        // To compute the overall bilinear form, we need to combine it with the respective slice of row X.
        __m512 a_vec = _simsimd_bf16x16_to_f32x16_skylake(_mm256_loadu_epi16(a + vector_progress));
        sum_vec = _mm512_fmadd_ps(a_vec, bc_collapsed_vec, sum_vec);
    }

    *result = _mm512_reduce_add_ps(sum_vec);
}

SIMSIMD_PUBLIC void simsimd_mahalanobis_bf16_sapphire(simsimd_bf16_t const* a, simsimd_bf16_t const* b,
                                                      simsimd_bf16_t const* c, simsimd_size_t n,
                                                      simsimd_distance_t* result) {}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SAPPHIRE
#endif // SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif

#endif
