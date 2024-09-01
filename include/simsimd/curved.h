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

#include "dot.h" // partial loads

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
// clang-format on

#define SIMSIMD_MAKE_BILINEAR(name, input_type, accumulator_type, converter)                                           \
    SIMSIMD_PUBLIC void simsimd_bilinear_##input_type##_##name(                                                        \
        simsimd_##input_type##_t const* a, simsimd_##input_type##_t const* b, simsimd_##input_type##_t const* c,       \
        simsimd_size_t n, simsimd_distance_t* result) {                                                                \
        simsimd_##accumulator_type##_t sum = 0;                                                                        \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                      \
            simsimd_##accumulator_type##_t partial = 0;                                                                \
            simsimd_##accumulator_type##_t a_i = converter(a[i]);                                                      \
            for (simsimd_size_t j = 0; j != n; ++j) {                                                                  \
                simsimd_##accumulator_type##_t b_j = converter(b[j]);                                                  \
                simsimd_##accumulator_type##_t c_ij = converter(c[i * n + j]);                                         \
                partial += c_ij * b_j;                                                                                 \
            }                                                                                                          \
            sum += a_i * partial;                                                                                      \
        }                                                                                                              \
        *result = (simsimd_distance_t)sum;                                                                             \
    }

#define SIMSIMD_MAKE_MAHALANOBIS(name, input_type, accumulator_type, converter)                                        \
    SIMSIMD_PUBLIC void simsimd_mahalanobis_##input_type##_##name(                                                     \
        simsimd_##input_type##_t const* a, simsimd_##input_type##_t const* b, simsimd_##input_type##_t const* c,       \
        simsimd_size_t n, simsimd_distance_t* result) {                                                                \
        simsimd_##accumulator_type##_t sum = 0;                                                                        \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                      \
            simsimd_##accumulator_type##_t partial = 0;                                                                \
            simsimd_##accumulator_type##_t diff_i = converter(a[i]) - converter(b[i]);                                 \
            for (simsimd_size_t j = 0; j != n; ++j) {                                                                  \
                simsimd_##accumulator_type##_t diff_j = converter(a[j]) - converter(b[j]);                             \
                simsimd_##accumulator_type##_t c_ij = converter(c[i * n + j]);                                         \
                partial += c_ij * diff_j;                                                                              \
            }                                                                                                          \
            sum += diff_i * partial;                                                                                   \
        }                                                                                                              \
        *result = (simsimd_distance_t)sum;                                                                             \
    }

SIMSIMD_MAKE_BILINEAR(serial, f64, f64, SIMSIMD_IDENTIFY)    // simsimd_bilinear_f64_serial
SIMSIMD_MAKE_MAHALANOBIS(serial, f64, f64, SIMSIMD_IDENTIFY) // simsimd_mahalanobis_f64_serial

SIMSIMD_MAKE_BILINEAR(serial, f32, f32, SIMSIMD_IDENTIFY)    // simsimd_bilinear_f32_serial
SIMSIMD_MAKE_MAHALANOBIS(serial, f32, f32, SIMSIMD_IDENTIFY) // simsimd_mahalanobis_f32_serial

SIMSIMD_MAKE_BILINEAR(serial, f16, f32, SIMSIMD_UNCOMPRESS_F16)    // simsimd_bilinear_f16_serial
SIMSIMD_MAKE_MAHALANOBIS(serial, f16, f32, SIMSIMD_UNCOMPRESS_F16) // simsimd_mahalanobis_f16_serial

SIMSIMD_MAKE_BILINEAR(serial, bf16, f32, SIMSIMD_UNCOMPRESS_BF16)    // simsimd_bilinear_bf16_serial
SIMSIMD_MAKE_MAHALANOBIS(serial, bf16, f32, SIMSIMD_UNCOMPRESS_BF16) // simsimd_mahalanobis_bf16_serial

SIMSIMD_MAKE_BILINEAR(accurate, f32, f64, SIMSIMD_IDENTIFY)    // simsimd_bilinear_f32_accurate
SIMSIMD_MAKE_MAHALANOBIS(accurate, f32, f64, SIMSIMD_IDENTIFY) // simsimd_mahalanobis_f32_accurate

SIMSIMD_MAKE_BILINEAR(accurate, f16, f64, SIMSIMD_UNCOMPRESS_F16)    // simsimd_bilinear_f16_accurate
SIMSIMD_MAKE_MAHALANOBIS(accurate, f16, f64, SIMSIMD_UNCOMPRESS_F16) // simsimd_mahalanobis_f16_accurate

SIMSIMD_MAKE_BILINEAR(accurate, bf16, f64, SIMSIMD_UNCOMPRESS_BF16)    // simsimd_bilinear_bf16_accurate
SIMSIMD_MAKE_MAHALANOBIS(accurate, bf16, f64, SIMSIMD_UNCOMPRESS_BF16) // simsimd_mahalanobis_bf16_accurate

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
        float32x4_t a_vec = vcvt_f32_f16(vdupq_n_f16((simsimd_f16_for_arm_simd_t const*)(a + i)));
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
            simsimd_f32_t a_i = vgetq_lane_f32(simsimd_partial_load_f16x4_neon(a + i, 1), 0);
            float32x4_t b_vec = simsimd_partial_load_f16x4_neon(b + tail_start, tail_length);
            float32x4_t c_vec = simsimd_partial_load_f16x4_neon(c + i * n + tail_start, tail_length);
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
        float32x4_t a_i_vec = vcvt_f32_f16(vdupq_n_f16((simsimd_f16_for_arm_simd_t const*)(a + i)));
        float32x4_t b_i_vec = vcvt_f32_f16(vdupq_n_f16((simsimd_f16_for_arm_simd_t const*)(b + i)));
        float32x4_t diff_i_vec = vsubq_f32(a_i_vec, b_i_vec);
        float32x4_t partial_sum_vec = vdupq_n_f32(0);
        for (simsimd_size_t j = 0; j + 4 <= n; j += 4) {
            float32x4_t a_j_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const*)(a + j)));
            float32x4_t b_j_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const*)(b + j)));
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
            simsimd_f32_t a_i = vgetq_lane_f32(simsimd_partial_load_f16x4_neon(a + i, 1), 0);
            simsimd_f32_t b_i = vgetq_lane_f32(simsimd_partial_load_f16x4_neon(b + i, 1), 0);
            simsimd_f32_t diff_i = a[i] - b[i];
            float32x4_t a_j_vec = simsimd_partial_load_f16x4_neon(a + tail_start, tail_length);
            float32x4_t b_j_vec = simsimd_partial_load_f16x4_neon(b + tail_start, tail_length);
            float32x4_t c_vec = simsimd_partial_load_f16x4_neon(c + i * n + tail_start, tail_length);
            simsimd_f32_t partial_sum = vaddvq_f32(vmulq_f32(b_vec, c_vec));
            sum += diff_i * partial_sum;
        }
    }

    *result = sum;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_F16
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
        __m256 a_vec = simsimd_partial_load_f16x8_haswell(a + i, 1);
        __m256 partial_sum_vec = _mm256_setzero_ps();
        for (simsimd_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + j)));
            __m256 c_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(c + i * n + j)));
            partial_sum_vec = _mm256_fmadd_ps(b_vec, c_vec, partial_sum_vec);
        }
        sum_vec = _mm256_fmadd_ps(a_vec, partial_sum_vec, sum_vec);
    }

    // Handle the tail of every row
    simsimd_f64_t sum = _mm256_reduce_add_ps_dbl(sum_vec);
    simsimd_size_t tail_length = n % 8;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t a_i = _mm256_cvtss_f32(simsimd_partial_load_f16x8_haswell(a + i, 1));
            __m256 b_vec = simsimd_partial_load_f16x8_haswell(b + tail_start, tail_length);
            __m256 c_vec = simsimd_partial_load_f16x8_haswell(c + i * n + tail_start, tail_length);
            simsimd_f32_t partial_sum = _mm256_reduce_add_ps_dbl(_mm256_mul_ps(b_vec, c_vec));
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
        __m256 diff_i_vec = _mm256_sub_ps(                //
            simsimd_partial_load_f16x8_haswell(a + i, 1), //
            simsimd_partial_load_f16x8_haswell(b + i, 1));
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
    simsimd_f64_t sum = _mm256_reduce_add_ps_dbl(sum_vec);
    simsimd_size_t tail_length = n % 8;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t diff_i = _mm256_cvtss_f32(_mm256_sub_ps( //
                simsimd_partial_load_f16x8_haswell(a + i, 1),      //
                simsimd_partial_load_f16x8_haswell(b + i, 1)));
            __m256 diff_j_vec = _mm256_sub_ps( //
                simsimd_partial_load_f16x8_haswell(a + tail_start, tail_length),
                simsimd_partial_load_f16x8_haswell(b + tail_start, tail_length));
            __m256 c_vec = simsimd_partial_load_f16x8_haswell(c + i * n + tail_start, tail_length);
            simsimd_f32_t partial_sum = _mm256_reduce_add_ps_dbl(_mm256_mul_ps(diff_j_vec, c_vec));
            partial_sum += diff_i * partial_sum;
        }
    }

    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_bilinear_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b,
                                                  simsimd_bf16_t const* c, simsimd_size_t n,
                                                  simsimd_distance_t* result) {
    __m256 sum_vec = _mm256_setzero_ps();
    for (simsimd_size_t i = 0; i != n; ++i) {
        // The `simsimd_uncompress_bf16` is cheaper than `simsimd_bf16x8_to_f32x8_haswell`
        __m256 a_vec = _mm256_set1_ps(simsimd_uncompress_bf16(a[i]));
        __m256 partial_sum_vec = _mm256_setzero_ps();
        for (simsimd_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 b_vec = simsimd_bf16x8_to_f32x8_haswell(_mm_loadu_si128((__m128i const*)(b + j)));
            __m256 c_vec = simsimd_bf16x8_to_f32x8_haswell(_mm_loadu_si128((__m128i const*)(c + i * n + j)));
            partial_sum_vec = _mm256_fmadd_ps(b_vec, c_vec, partial_sum_vec);
        }
        sum_vec = _mm256_fmadd_ps(a_vec, partial_sum_vec, sum_vec);
    }

    // Handle the tail of every row
    simsimd_f64_t sum = _mm256_reduce_add_ps_dbl(sum_vec);
    simsimd_size_t tail_length = n % 8;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t a_i = simsimd_uncompress_bf16(a[i]);
            __m256 b_vec = simsimd_bf16x8_to_f32x8_haswell( //
                simsimd_partial_load_bf16x8_haswell(b + tail_start, tail_length));
            __m256 c_vec = simsimd_bf16x8_to_f32x8_haswell( //
                simsimd_partial_load_bf16x8_haswell(c + i * n + tail_start, tail_length));
            simsimd_f32_t partial_sum = _mm256_reduce_add_ps_dbl(_mm256_mul_ps(b_vec, c_vec));
            sum += a_i * partial_sum;
        }
    }
}

SIMSIMD_PUBLIC void simsimd_mahalanobis_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b,
                                                     simsimd_bf16_t const* c, simsimd_size_t n,
                                                     simsimd_distance_t* result) {
    __m256 sum_vec = _mm256_setzero_ps();
    for (simsimd_size_t i = 0; i != n; ++i) {
        __m256 diff_i_vec = _mm256_sub_ps(                 //
            _mm256_set1_ps(simsimd_uncompress_bf16(a[i])), //
            _mm256_set1_ps(simsimd_uncompress_bf16(b[i])));
        __m256 partial_sum_vec = _mm256_setzero_ps();
        for (simsimd_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 diff_j_vec = _mm256_sub_ps(                                             //
                simsimd_bf16x8_to_f32x8_haswell(_mm_loadu_si128((__m128i const*)(a + j))), //
                simsimd_bf16x8_to_f32x8_haswell(_mm_loadu_si128((__m128i const*)(b + j))));
            __m256 c_vec = simsimd_bf16x8_to_f32x8_haswell(_mm_loadu_si128((__m128i const*)(c + i * n + j)));
            partial_sum_vec = _mm256_fmadd_ps(diff_j_vec, c_vec, partial_sum_vec);
        }
        sum_vec = _mm256_fmadd_ps(diff_i_vec, partial_sum_vec, sum_vec);
    }

    // Handle the tail of every row
    simsimd_f64_t sum = _mm256_reduce_add_ps_dbl(sum_vec);
    simsimd_size_t tail_length = n % 8;
    simsimd_size_t tail_start = n - tail_length;
    if (tail_length) {
        for (simsimd_size_t i = 0; i != n; ++i) {
            simsimd_f32_t diff_i = simsimd_uncompress_bf16(a[i]) - simsimd_uncompress_bf16(b[i]);
            __m256 diff_j_vec = _mm256_sub_ps( //
                simsimd_bf16x8_to_f32x8_haswell(simsimd_partial_load_bf16x8_haswell(a + tail_start, tail_length)),
                simsimd_bf16x8_to_f32x8_haswell(simsimd_partial_load_bf16x8_haswell(b + tail_start, tail_length)));
            __m256 c_vec = simsimd_bf16x8_to_f32x8_haswell(
                simsimd_partial_load_bf16x8_haswell(c + i * n + tail_start, tail_length));
            simsimd_f32_t partial_sum = _mm256_reduce_add_ps_dbl(_mm256_mul_ps(diff_j_vec, c_vec));
            sum += diff_i * partial_sum;
        }
    }
}

#endif // SIMSIMD_TARGET_HASWELL
#endif // SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif

#endif
