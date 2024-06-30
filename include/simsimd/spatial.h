/**
 *  @file       spatial.h
 *  @brief      SIMD-accelerated Spatial Similarity Measures.
 *  @author     Ash Vardanian
 *  @date       March 14, 2023
 *
 *  Contains:
 *  - L2 (Euclidean) squared distance
 *  - Cosine (Angular) similarity
 *
 *  For datatypes:
 *  - 64-bit IEEE floating point numbers
 *  - 32-bit IEEE floating point numbers
 *  - 16-bit IEEE floating point numbers
 *  - 16-bit brain floating point numbers
 *  - 8-bit signed integral numbers
 *
 *  For hardware architectures:
 *  - Arm (NEON, SVE)
 *  - x86 (AVX2, AVX512)
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef SIMSIMD_SPATIAL_H
#define SIMSIMD_SPATIAL_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// clang-format off

/*  Serial backends for all numeric types.
 *  By default they use 32-bit arithmetic, unless the arguments themselves contain 64-bit floats.
 *  For double-precision computation check out the "*_accurate" variants of those "*_serial" functions.
 */
SIMSIMD_PUBLIC void simsimd_l2sq_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_i8_serial(simsimd_i8_t const* a, simsimd_i8_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_i8_serial(simsimd_i8_t const* a, simsimd_i8_t const*, simsimd_size_t n, simsimd_distance_t* d);

/*  Double-precision serial backends for all numeric types.
 *  For single-precision computation check out the "*_serial" counterparts of those "*_accurate" functions.
 */
SIMSIMD_PUBLIC void simsimd_l2sq_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_i8_accurate(simsimd_i8_t const* a, simsimd_i8_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_i8_accurate(simsimd_i8_t const* a, simsimd_i8_t const*, simsimd_size_t n, simsimd_distance_t* d);

/*  SIMD-powered backends for Arm NEON, mostly using 32-bit arithmetic over 128-bit words.
 *  By far the most portable backend, covering most Arm v8 devices, over a billion phones, and almost all
 *  server CPUs produced before 2023.
 */
SIMSIMD_PUBLIC void simsimd_l2sq_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_i8_neon(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_i8_neon(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* d);

/*  SIMD-powered backends for Arm SVE, mostly using 32-bit arithmetic over variable-length platform-defined word sizes.
 *  Designed for Arm Graviton 3, Microsoft Cobalt, as well as Nvidia Grace and newer Ampere Altra CPUs.
 */
SIMSIMD_PUBLIC void simsimd_l2sq_f32_sve(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_cos_f32_sve(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_l2sq_f16_sve(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_cos_f16_sve(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_l2sq_f64_sve(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_cos_f64_sve(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t*);

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic over 256-bit words.
 *  First demonstrated in 2011, at least one Haswell-based processor was still being sold in 2022 — the Pentium G3420.
 *  Practically all modern x86 CPUs support AVX2, FMA, and F16C, making it a perfect baseline for SIMD algorithms.
 *  On other hand, there is no need to implement AVX2 versions of `f32` and `f64` functions, as those are
 *  properly vectorized by recent compilers.
 */
SIMSIMD_PUBLIC void simsimd_l2sq_i8_haswell(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_cos_i8_haswell(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_l2sq_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_cos_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_cos_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t*);

/*  SIMD-powered backends for AVX512 CPUs of Skylake generation and newer, using 32-bit arithmetic over 512-bit words.
 *  Skylake was launched in 2015, and discontinued in 2019. Skylake had support for F, CD, VL, DQ, and BW extensions,
 *  as well as masked operations. This is enough to supersede auto-vectorization on `f32` and `f64` types.
 */
SIMSIMD_PUBLIC void simsimd_l2sq_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_cos_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_l2sq_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_cos_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t*);

/*  SIMD-powered backends for AVX512 CPUs of Ice Lake generation and newer, using mixed arithmetic over 512-bit words.
 *  Ice Lake added VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG, VPCLMULQDQ, and other extensions for integral operations.
 *  Sapphire Rapids added tiled matrix operations, but we are most interested in the new mixed-precision FMA instructions.
 */
SIMSIMD_PUBLIC void simsimd_l2sq_i8_ice(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_cos_i8_ice(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_cos_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_l2sq_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t*);
SIMSIMD_PUBLIC void simsimd_cos_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t*);

// clang-format on

#define SIMSIMD_MAKE_L2SQ(name, input_type, accumulator_type, converter)                                               \
    SIMSIMD_PUBLIC void simsimd_l2sq_##input_type##_##name(simsimd_##input_type##_t const* a,                          \
                                                           simsimd_##input_type##_t const* b, simsimd_size_t n,        \
                                                           simsimd_distance_t* result) {                               \
        simsimd_##accumulator_type##_t d2 = 0;                                                                         \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                      \
            simsimd_##accumulator_type##_t ai = converter(a[i]);                                                       \
            simsimd_##accumulator_type##_t bi = converter(b[i]);                                                       \
            d2 += (ai - bi) * (ai - bi);                                                                               \
        }                                                                                                              \
        *result = d2;                                                                                                  \
    }

#define SIMSIMD_MAKE_COS(name, input_type, accumulator_type, converter)                                                \
    SIMSIMD_PUBLIC void simsimd_cos_##input_type##_##name(simsimd_##input_type##_t const* a,                           \
                                                          simsimd_##input_type##_t const* b, simsimd_size_t n,         \
                                                          simsimd_distance_t* result) {                                \
        simsimd_##accumulator_type##_t ab = 0, a2 = 0, b2 = 0;                                                         \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                      \
            simsimd_##accumulator_type##_t ai = converter(a[i]);                                                       \
            simsimd_##accumulator_type##_t bi = converter(b[i]);                                                       \
            ab += ai * bi;                                                                                             \
            a2 += ai * ai;                                                                                             \
            b2 += bi * bi;                                                                                             \
        }                                                                                                              \
        *result = ab != 0 ? (1 - ab * SIMSIMD_RSQRT(a2) * SIMSIMD_RSQRT(b2)) : 1;                                      \
    }

SIMSIMD_MAKE_L2SQ(serial, f64, f64, SIMSIMD_IDENTIFY) // simsimd_l2sq_f64_serial
SIMSIMD_MAKE_COS(serial, f64, f64, SIMSIMD_IDENTIFY)  // simsimd_cos_f64_serial

SIMSIMD_MAKE_L2SQ(serial, f32, f32, SIMSIMD_IDENTIFY) // simsimd_l2sq_f32_serial
SIMSIMD_MAKE_COS(serial, f32, f32, SIMSIMD_IDENTIFY)  // simsimd_cos_f32_serial

SIMSIMD_MAKE_L2SQ(serial, f16, f32, SIMSIMD_UNCOMPRESS_F16) // simsimd_l2sq_f16_serial
SIMSIMD_MAKE_COS(serial, f16, f32, SIMSIMD_UNCOMPRESS_F16)  // simsimd_cos_f16_serial

SIMSIMD_MAKE_L2SQ(serial, bf16, f32, SIMSIMD_UNCOMPRESS_BF16) // simsimd_l2sq_bf16_serial
SIMSIMD_MAKE_COS(serial, bf16, f32, SIMSIMD_UNCOMPRESS_BF16)  // simsimd_cos_bf16_serial

SIMSIMD_MAKE_L2SQ(serial, i8, i32, SIMSIMD_IDENTIFY) // simsimd_l2sq_i8_serial
SIMSIMD_MAKE_COS(serial, i8, i32, SIMSIMD_IDENTIFY)  // simsimd_cos_i8_serial

SIMSIMD_MAKE_L2SQ(accurate, f32, f64, SIMSIMD_IDENTIFY) // simsimd_l2sq_f32_accurate
SIMSIMD_MAKE_COS(accurate, f32, f64, SIMSIMD_IDENTIFY)  // simsimd_cos_f32_accurate

SIMSIMD_MAKE_L2SQ(accurate, f16, f64, SIMSIMD_UNCOMPRESS_F16) // simsimd_l2sq_f16_accurate
SIMSIMD_MAKE_COS(accurate, f16, f64, SIMSIMD_UNCOMPRESS_F16)  // simsimd_cos_f16_accurate

SIMSIMD_MAKE_L2SQ(accurate, bf16, f64, SIMSIMD_UNCOMPRESS_BF16) // simsimd_l2sq_bf16_accurate
SIMSIMD_MAKE_COS(accurate, bf16, f64, SIMSIMD_UNCOMPRESS_BF16)  // simsimd_cos_bf16_accurate

SIMSIMD_MAKE_L2SQ(accurate, i8, i32, SIMSIMD_IDENTIFY) // simsimd_l2sq_i8_accurate
SIMSIMD_MAKE_COS(accurate, i8, i32, SIMSIMD_IDENTIFY)  // simsimd_cos_i8_accurate

#if SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("+simd")
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2sq_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                          simsimd_distance_t* result) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
        sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
    }
    simsimd_f32_t sum = vaddvq_f32(sum_vec);
    for (; i < n; ++i) {
        simsimd_f32_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_cos_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                         simsimd_distance_t* result) {
    float32x4_t ab_vec = vdupq_n_f32(0), a2_vec = vdupq_n_f32(0), b2_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
        a2_vec = vfmaq_f32(a2_vec, a_vec, a_vec);
        b2_vec = vfmaq_f32(b2_vec, b_vec, b_vec);
    }
    simsimd_f32_t ab = vaddvq_f32(ab_vec), a2 = vaddvq_f32(a2_vec), b2 = vaddvq_f32(b2_vec);
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    // Avoid `simsimd_approximate_inverse_square_root` on Arm NEON
    simsimd_f32_t a2_b2_arr[2] = {a2, b2};
    vst1_f32(a2_b2_arr, vrsqrte_f32(vld1_f32(a2_b2_arr)));
    *result = ab != 0 ? 1 - ab * a2_b2_arr[0] * a2_b2_arr[1] : 1;
}

#pragma clang attribute pop
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("+simd+fp16")
#pragma clang attribute push(__attribute__((target("+simd+fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2sq_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                          simsimd_distance_t* result) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const*)a + i));
        float32x4_t b_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const*)b + i));
        float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
        sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
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
        float32x4_t diff_vec = vsubq_f32(vcvt_f32_f16(a_padded_tail.f16_vec), vcvt_f32_f16(b_padded_tail.f16_vec));
        sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
    }

    simsimd_f32_t sum = vaddvq_f32(sum_vec);
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_cos_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                         simsimd_distance_t* result) {
    float32x4_t ab_vec = vdupq_n_f32(0), a2_vec = vdupq_n_f32(0), b2_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const*)a + i));
        float32x4_t b_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const*)b + i));
        ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
        a2_vec = vfmaq_f32(a2_vec, a_vec, a_vec);
        b2_vec = vfmaq_f32(b2_vec, b_vec, b_vec);
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
        float32x4_t a_vec = vcvt_f32_f16(a_padded_tail.f16_vec);
        float32x4_t b_vec = vcvt_f32_f16(b_padded_tail.f16_vec);
        ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
        a2_vec = vfmaq_f32(a2_vec, a_vec, a_vec);
        b2_vec = vfmaq_f32(b2_vec, b_vec, b_vec);
    }

    // Avoid `simsimd_approximate_inverse_square_root` on Arm NEON
    simsimd_f32_t ab = vaddvq_f32(ab_vec), a2 = vaddvq_f32(a2_vec), b2 = vaddvq_f32(b2_vec);
    simsimd_f32_t a2_b2_arr[2] = {a2, b2};
    float32x2_t a2_b2 = vld1_f32(a2_b2_arr);
    a2_b2 = vrsqrte_f32(a2_b2);
    vst1_f32(a2_b2_arr, a2_b2);
    *result = ab != 0 ? 1 - ab * a2_b2_arr[0] * a2_b2_arr[1] : 1;
}

SIMSIMD_PUBLIC void simsimd_cos_bf16_neon(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                          simsimd_distance_t* result) {

    // Similar to `simsimd_cos_i8_neon`, we can use the `BFMMLA` instruction through
    // the `vbfmmlaq_f32` intrinsic to compute matrix products and later drop 1/4 of values.
    // The only difference is that `zip` isn't provided for `bf16` and we need to reinterpret back
    // and forth before zipping. Same as with integers, on modern Arm CPUs, this "smart"
    // approach is actually slower by around 25%.
    //
    //   float32x4_t products_low_vec = vdupq_n_f32(0.0f);
    //   float32x4_t products_high_vec = vdupq_n_f32(0.0f);
    //   for (; i + 8 <= n; i += 8) {
    //       bfloat16x8_t a_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)a + i);
    //       bfloat16x8_t b_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)b + i);
    //       int16x8_t a_vec_s16 = vreinterpretq_s16_bf16(a_vec);
    //       int16x8_t b_vec_s16 = vreinterpretq_s16_bf16(b_vec);
    //       int16x8x2_t y_w_vecs_s16 = vzipq_s16(a_vec_s16, b_vec_s16);
    //       bfloat16x8_t y_vec = vreinterpretq_bf16_s16(y_w_vecs_s16.val[0]);
    //       bfloat16x8_t w_vec = vreinterpretq_bf16_s16(y_w_vecs_s16.val[1]);
    //       bfloat16x4_t a_low = vget_low_bf16(a_vec);
    //       bfloat16x4_t b_low = vget_low_bf16(b_vec);
    //       bfloat16x4_t a_high = vget_high_bf16(a_vec);
    //       bfloat16x4_t b_high = vget_high_bf16(b_vec);
    //       bfloat16x8_t x_vec = vcombine_bf16(a_low, b_low);
    //       bfloat16x8_t v_vec = vcombine_bf16(a_high, b_high);
    //       products_low_vec = vbfmmlaq_f32(products_low_vec, x_vec, y_vec);
    //       products_high_vec = vbfmmlaq_f32(products_high_vec, v_vec, w_vec);
    //   }
    //   float32x4_t products_vec = vaddq_f32(products_high_vec, products_low_vec);
    //   simsimd_f32_t a2 = products_vec[0], ab = products_vec[1], b2 = products_vec[3];

    float32x4_t ab_high_vec = vdupq_n_f32(0), ab_low_vec = vdupq_n_f32(0);
    float32x4_t a2_high_vec = vdupq_n_f32(0), a2_low_vec = vdupq_n_f32(0);
    float32x4_t b2_high_vec = vdupq_n_f32(0), b2_low_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        bfloat16x8_t a_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)a + i);
        bfloat16x8_t b_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)b + i);
        ab_high_vec = vbfmlaltq_f32(ab_high_vec, a_vec, b_vec);
        ab_low_vec = vbfmlalbq_f32(ab_low_vec, a_vec, b_vec);
        a2_high_vec = vbfmlaltq_f32(a2_high_vec, a_vec, a_vec);
        a2_low_vec = vbfmlalbq_f32(a2_low_vec, a_vec, a_vec);
        b2_high_vec = vbfmlaltq_f32(b2_high_vec, b_vec, b_vec);
        b2_low_vec = vbfmlalbq_f32(b2_low_vec, b_vec, b_vec);
    }

    // In case the software emulation for `bf16` scalars is enabled, the `simsimd_uncompress_bf16`
    // function will run. It is extremely slow, so even for the tail, let's combine serial
    // loads and stores with vectorized math.
    if (i < n) {
        union {
            bfloat16x8_t bf16_vec;
            simsimd_bf16_t bf16[8];
        } a_padded_tail, b_padded_tail;
        simsimd_size_t j = 0;
        for (; i < n; ++i, ++j)
            a_padded_tail.bf16[j] = a[i], b_padded_tail.bf16[j] = b[i];
        for (; j < 8; ++j)
            a_padded_tail.bf16[j] = 0, b_padded_tail.bf16[j] = 0;
        ab_high_vec = vbfmlaltq_f32(ab_high_vec, a_padded_tail.bf16_vec, b_padded_tail.bf16_vec);
        ab_low_vec = vbfmlalbq_f32(ab_low_vec, a_padded_tail.bf16_vec, b_padded_tail.bf16_vec);
        a2_high_vec = vbfmlaltq_f32(a2_high_vec, a_padded_tail.bf16_vec, a_padded_tail.bf16_vec);
        a2_low_vec = vbfmlalbq_f32(a2_low_vec, a_padded_tail.bf16_vec, a_padded_tail.bf16_vec);
        b2_high_vec = vbfmlaltq_f32(b2_high_vec, b_padded_tail.bf16_vec, b_padded_tail.bf16_vec);
        b2_low_vec = vbfmlalbq_f32(b2_low_vec, b_padded_tail.bf16_vec, b_padded_tail.bf16_vec);
    }

    // Avoid `simsimd_approximate_inverse_square_root` on Arm NEON
    simsimd_f32_t ab = vaddvq_f32(vaddq_f32(ab_high_vec, ab_low_vec)),
                  a2 = vaddvq_f32(vaddq_f32(a2_high_vec, a2_low_vec)),
                  b2 = vaddvq_f32(vaddq_f32(b2_high_vec, b2_low_vec));
    simsimd_f32_t a2_b2_arr[2] = {a2, b2};
    float32x2_t a2_b2 = vld1_f32(a2_b2_arr);
    a2_b2 = vrsqrte_f32(a2_b2);
    vst1_f32(a2_b2_arr, a2_b2);
    *result = ab != 0 ? 1 - ab * a2_b2_arr[0] * a2_b2_arr[1] : 1;
}

SIMSIMD_PUBLIC void simsimd_l2sq_bf16_neon(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                           simsimd_distance_t* result) {
    float32x4_t diff_high_vec = vdupq_n_f32(0), diff_low_vec = vdupq_n_f32(0);
    float32x4_t sum_high_vec = vdupq_n_f32(0), sum_low_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        bfloat16x8_t a_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)a + i);
        bfloat16x8_t b_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)b + i);
        // We can't perform subtraction in `bf16`. One option would be to upcast to `f32`
        // and then subtract, converting back to `bf16` for computing the squared difference.
        diff_high_vec = vsubq_f32(vcvt_f32_bf16(vget_high_bf16(a_vec)), vcvt_f32_bf16(vget_high_bf16(b_vec)));
        diff_low_vec = vsubq_f32(vcvt_f32_bf16(vget_low_bf16(a_vec)), vcvt_f32_bf16(vget_low_bf16(b_vec)));
        sum_high_vec = vfmaq_f32(sum_high_vec, diff_high_vec, diff_high_vec);
        sum_low_vec = vfmaq_f32(sum_low_vec, diff_low_vec, diff_low_vec);
    }

    // In case the software emulation for `bf16` scalars is enabled, the `simsimd_uncompress_bf16`
    // function will run. It is extremely slow, so even for the tail, let's combine serial
    // loads and stores with vectorized math.
    if (i < n) {
        union {
            bfloat16x8_t bf16_vec;
            simsimd_bf16_t bf16[8];
        } a_padded_tail, b_padded_tail;
        simsimd_size_t j = 0;
        for (; i < n; ++i, ++j)
            a_padded_tail.bf16[j] = a[i], b_padded_tail.bf16[j] = b[i];
        for (; j < 8; ++j)
            a_padded_tail.bf16[j] = 0, b_padded_tail.bf16[j] = 0;
        diff_high_vec = vsubq_f32(vcvt_f32_bf16(vget_high_bf16(a_padded_tail.bf16_vec)),
                                  vcvt_f32_bf16(vget_high_bf16(b_padded_tail.bf16_vec)));
        diff_low_vec = vsubq_f32(vcvt_f32_bf16(vget_low_bf16(a_padded_tail.bf16_vec)),
                                 vcvt_f32_bf16(vget_low_bf16(b_padded_tail.bf16_vec)));
        sum_high_vec = vfmaq_f32(sum_high_vec, diff_high_vec, diff_high_vec);
        sum_low_vec = vfmaq_f32(sum_low_vec, diff_low_vec, diff_low_vec);
    }

    // Avoid `simsimd_approximate_inverse_square_root` on Arm NEON
    simsimd_f32_t sum = vaddvq_f32(vaddq_f32(sum_high_vec, sum_low_vec));
    *result = sum;
}
#pragma clang attribute pop
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod+i8mm")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod+i8mm"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2sq_i8_neon(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                         simsimd_distance_t* result) {
    int32x4_t d2_vec = vdupq_n_s32(0);
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t a_vec = vld1_s8(a + i);
        int8x8_t b_vec = vld1_s8(b + i);
        int16x8_t a_vec16 = vmovl_s8(a_vec);
        int16x8_t b_vec16 = vmovl_s8(b_vec);
        int16x8_t d_vec = vsubq_s16(a_vec16, b_vec16);
        int32x4_t d_low = vmull_s16(vget_low_s16(d_vec), vget_low_s16(d_vec));
        int32x4_t d_high = vmull_s16(vget_high_s16(d_vec), vget_high_s16(d_vec));
        d2_vec = vaddq_s32(d2_vec, vaddq_s32(d_low, d_high));
    }
    int32_t d2 = vaddvq_s32(d2_vec);
    for (; i < n; ++i) {
        int32_t n = a[i] - b[i];
        d2 += n * n;
    }
    *result = d2;
}

SIMSIMD_PUBLIC void simsimd_cos_i8_neon(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                        simsimd_distance_t* result) {

    simsimd_size_t i = 0;

    // Variant 1.
    // If the 128-bit `vdot_s32` intrinsic is unavailable, we can use the 64-bit `vdot_s32`.
    //
    //  int32x4_t ab_vec = vdupq_n_s32(0);
    //  int32x4_t a2_vec = vdupq_n_s32(0);
    //  int32x4_t b2_vec = vdupq_n_s32(0);
    //  for (simsimd_size_t i = 0; i != n; i += 8) {
    //      int16x8_t a_vec = vmovl_s8(vld1_s8(a + i));
    //      int16x8_t b_vec = vmovl_s8(vld1_s8(b + i));
    //      int16x8_t ab_part_vec = vmulq_s16(a_vec, b_vec);
    //      int16x8_t a2_part_vec = vmulq_s16(a_vec, a_vec);
    //      int16x8_t b2_part_vec = vmulq_s16(b_vec, b_vec);
    //      ab_vec = vaddq_s32(ab_vec, vaddq_s32(vmovl_s16(vget_high_s16(ab_part_vec)), //
    //                                           vmovl_s16(vget_low_s16(ab_part_vec))));
    //      a2_vec = vaddq_s32(a2_vec, vaddq_s32(vmovl_s16(vget_high_s16(a2_part_vec)), //
    //                                           vmovl_s16(vget_low_s16(a2_part_vec))));
    //      b2_vec = vaddq_s32(b2_vec, vaddq_s32(vmovl_s16(vget_high_s16(b2_part_vec)), //
    //                                           vmovl_s16(vget_low_s16(b2_part_vec))));
    //  }
    //
    // Variant 2.
    // With the 128-bit `vdotq_s32` intrinsic, we can use the following code:
    //
    //  for (; i + 16 <= n; i += 16) {
    //      int8x16_t a_vec = vld1q_s8(a + i);
    //      int8x16_t b_vec = vld1q_s8(b + i);
    //      ab_vec = vdotq_s32(ab_vec, a_vec, b_vec);
    //      a2_vec = vdotq_s32(a2_vec, a_vec, a_vec);
    //      b2_vec = vdotq_s32(b2_vec, b_vec, b_vec);
    //  }
    //
    // Variant 3.
    // To use MMLA instructions, we need to reorganize the contents of the vectors.
    // On input we have `a_vec` and `b_vec`:
    //
    //   a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]
    //   b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    //
    // We will be multiplying matrices of size 2x8 and 8x2. So we need to perform a few shuffles:
    //
    //   X =
    //      a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
    //      b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
    //   Y =
    //      a[0], b[0],
    //      a[1], b[1],
    //      a[2], b[2],
    //      a[3], b[3],
    //      a[4], b[4],
    //      a[5], b[5],
    //      a[6], b[6],
    //      a[7], b[7]
    //
    //   V =
    //      a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15],
    //      b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    //   W =
    //      a[8],   b[8],
    //      a[9],   b[9],
    //      a[10],  b[10],
    //      a[11],  b[11],
    //      a[12],  b[12],
    //      a[13],  b[13],
    //      a[14],  b[14],
    //      a[15],  b[15]
    //
    // Performing matrix multiplications we can aggregate into a matrix `products_low_vec` and `products_high_vec`:
    //
    //      X * X, X * Y                V * W, V * V
    //      Y * X, Y * Y                W * W, W * V
    //
    // Of those values we need only 3/4, as the (X * Y) and (Y * X) are the same.
    //
    //      int32x4_t products_low_vec = vdupq_n_s32(0), products_high_vec = vdupq_n_s32(0);
    //      int8x16_t a_low_b_low_vec, a_high_b_high_vec;
    //      for (; i + 16 <= n; i += 16) {
    //          int8x16_t a_vec = vld1q_s8(a + i);
    //          int8x16_t b_vec = vld1q_s8(b + i);
    //          int8x16x2_t y_w_vecs = vzipq_s8(a_vec, b_vec);
    //          int8x16_t x_vec = vcombine_s8(vget_low_s8(a_vec), vget_low_s8(b_vec));
    //          int8x16_t v_vec = vcombine_s8(vget_high_s8(a_vec), vget_high_s8(b_vec));
    //          products_low_vec = vmmlaq_s32(products_low_vec, x_vec, y_w_vecs.val[0]);
    //          products_high_vec = vmmlaq_s32(products_high_vec, v_vec, y_w_vecs.val[1]);
    //      }
    //      int32x4_t products_vec = vaddq_s32(products_high_vec, products_low_vec);
    //      int32_t a2 = products_vec[0];
    //      int32_t ab = products_vec[1];
    //      int32_t b2 = products_vec[3];
    //
    // That solution is elegant, but it requires the additional `+i8mm` extension and is currently slower,
    // at least on AWS Graviton 3.
    int32x4_t ab_vec = vdupq_n_s32(0);
    int32x4_t a2_vec = vdupq_n_s32(0);
    int32x4_t b2_vec = vdupq_n_s32(0);
    for (; i + 16 <= n; i += 16) {
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
    *result = ab != 0 ? 1 - ab * a2_b2_arr[0] * a2_b2_arr[1] : 1;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_SVE
#pragma GCC push_options
#pragma GCC target("+sve")
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2sq_f32_sve(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                         simsimd_distance_t* result) {
    simsimd_size_t i = 0;
    svfloat32_t d2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
        svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
        svfloat32_t a_minus_b_vec = svsub_f32_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f32_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcntw();
    } while (i < n);
    simsimd_f32_t d2 = svaddv_f32(svptrue_b32(), d2_vec);
    *result = d2;
}

SIMSIMD_PUBLIC void simsimd_cos_f32_sve(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                        simsimd_distance_t* result) {
    simsimd_size_t i = 0;
    svfloat32_t ab_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t a2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t b2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
        svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
        ab_vec = svmla_f32_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f32_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f32_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcntw();
    } while (i < n);

    simsimd_f32_t ab = svaddv_f32(svptrue_b32(), ab_vec);
    simsimd_f32_t a2 = svaddv_f32(svptrue_b32(), a2_vec);
    simsimd_f32_t b2 = svaddv_f32(svptrue_b32(), b2_vec);

    // Avoid `simsimd_approximate_inverse_square_root` on Arm NEON
    simsimd_f32_t a2_b2_arr[2] = {a2, b2};
    float32x2_t a2_b2 = vld1_f32(a2_b2_arr);
    a2_b2 = vrsqrte_f32(a2_b2);
    vst1_f32(a2_b2_arr, a2_b2);
    *result = ab != 0 ? 1 - ab * a2_b2_arr[0] * a2_b2_arr[1] : 1;
}

SIMSIMD_PUBLIC void simsimd_l2sq_f64_sve(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                         simsimd_distance_t* result) {
    simsimd_size_t i = 0;
    svfloat64_t d2_vec = svdupq_n_f64(0.0, 0.0);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)i, (unsigned int)n);
        svfloat64_t a_vec = svld1_f64(pg_vec, a + i);
        svfloat64_t b_vec = svld1_f64(pg_vec, b + i);
        svfloat64_t a_minus_b_vec = svsub_f64_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f64_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcntd();
    } while (i < n);
    simsimd_f64_t d2 = svaddv_f64(svptrue_b32(), d2_vec);
    *result = d2;
}

SIMSIMD_PUBLIC void simsimd_cos_f64_sve(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                        simsimd_distance_t* result) {
    simsimd_size_t i = 0;
    svfloat64_t ab_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t a2_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t b2_vec = svdupq_n_f64(0.0, 0.0);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)i, (unsigned int)n);
        svfloat64_t a_vec = svld1_f64(pg_vec, a + i);
        svfloat64_t b_vec = svld1_f64(pg_vec, b + i);
        ab_vec = svmla_f64_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f64_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f64_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcntd();
    } while (i < n);

    simsimd_f64_t ab = svaddv_f64(svptrue_b32(), ab_vec);
    simsimd_f64_t a2 = svaddv_f64(svptrue_b32(), a2_vec);
    simsimd_f64_t b2 = svaddv_f64(svptrue_b32(), b2_vec);

    // Avoid `simsimd_approximate_inverse_square_root` on Arm NEON
    simsimd_f64_t a2_b2_arr[2] = {a2, b2};
    float64x2_t a2_b2 = vld1q_f64(a2_b2_arr);
    a2_b2 = vrsqrteq_f64(a2_b2);
    vst1q_f64(a2_b2_arr, a2_b2);
    *result = ab != 0 ? 1 - ab * a2_b2_arr[0] * a2_b2_arr[1] : 1;
}

#pragma clang attribute pop
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("+sve+fp16")
#pragma clang attribute push(__attribute__((target("+sve+fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2sq_f16_sve(simsimd_f16_t const* a_enum, simsimd_f16_t const* b_enum, simsimd_size_t n,
                                         simsimd_distance_t* result) {
    simsimd_size_t i = 0;
    svfloat16_t d2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    simsimd_f16_for_arm_simd_t const* a = (simsimd_f16_for_arm_simd_t const*)(a_enum);
    simsimd_f16_for_arm_simd_t const* b = (simsimd_f16_for_arm_simd_t const*)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svfloat16_t a_vec = svld1_f16(pg_vec, a + i);
        svfloat16_t b_vec = svld1_f16(pg_vec, b + i);
        svfloat16_t a_minus_b_vec = svsub_f16_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f16_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcnth();
    } while (i < n);
    simsimd_f16_for_arm_simd_t d2_f16 = svaddv_f16(svptrue_b16(), d2_vec);
    *result = d2_f16;
}

SIMSIMD_PUBLIC void simsimd_cos_f16_sve(simsimd_f16_t const* a_enum, simsimd_f16_t const* b_enum, simsimd_size_t n,
                                        simsimd_distance_t* result) {
    simsimd_size_t i = 0;
    svfloat16_t ab_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    svfloat16_t a2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    svfloat16_t b2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    simsimd_f16_for_arm_simd_t const* a = (simsimd_f16_for_arm_simd_t const*)(a_enum);
    simsimd_f16_for_arm_simd_t const* b = (simsimd_f16_for_arm_simd_t const*)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svfloat16_t a_vec = svld1_f16(pg_vec, a + i);
        svfloat16_t b_vec = svld1_f16(pg_vec, b + i);
        ab_vec = svmla_f16_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f16_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f16_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcnth();
    } while (i < n);

    simsimd_f16_for_arm_simd_t ab = svaddv_f16(svptrue_b16(), ab_vec);
    simsimd_f16_for_arm_simd_t a2 = svaddv_f16(svptrue_b16(), a2_vec);
    simsimd_f16_for_arm_simd_t b2 = svaddv_f16(svptrue_b16(), b2_vec);

    // Avoid `simsimd_approximate_inverse_square_root` on Arm NEON
    simsimd_f32_t a2_b2_arr[2] = {a2, b2};
    float32x2_t a2_b2 = vld1_f32(a2_b2_arr);
    a2_b2 = vrsqrte_f32(a2_b2);
    vst1_f32(a2_b2_arr, a2_b2);
    *result = ab != 0 ? 1 - ab * a2_b2_arr[0] * a2_b2_arr[1] : 1;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SVE
#endif // SIMSIMD_TARGET_ARM

#if SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2sq_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                             simsimd_distance_t* result) {
    __m256 d2_vec = _mm256_setzero_ps();
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));
        __m256 d_vec = _mm256_sub_ps(a_vec, b_vec);
        d2_vec = _mm256_fmadd_ps(d_vec, d_vec, d2_vec);
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
        __m256 d_vec = _mm256_sub_ps(a_vec, b_vec);
        d2_vec = _mm256_fmadd_ps(d_vec, d_vec, d2_vec);
    }

    d2_vec = _mm256_add_ps(_mm256_permute2f128_ps(d2_vec, d2_vec, 1), d2_vec);
    d2_vec = _mm256_hadd_ps(d2_vec, d2_vec);
    d2_vec = _mm256_hadd_ps(d2_vec, d2_vec);

    simsimd_f32_t f32_result;
    _mm_store_ss(&f32_result, _mm256_castps256_ps128(d2_vec));
    *result = f32_result;
}

SIMSIMD_PUBLIC void simsimd_cos_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                            simsimd_distance_t* result) {

    __m256 ab_vec = _mm256_setzero_ps(), a2_vec = _mm256_setzero_ps(), b2_vec = _mm256_setzero_ps();
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));
        ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
        a2_vec = _mm256_fmadd_ps(a_vec, a_vec, a2_vec);
        b2_vec = _mm256_fmadd_ps(b_vec, b_vec, b2_vec);
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
        a2_vec = _mm256_fmadd_ps(a_vec, a_vec, a2_vec);
        b2_vec = _mm256_fmadd_ps(b_vec, b_vec, b2_vec);
    }

    // Horizontal reductions:
    ab_vec = _mm256_add_ps(_mm256_permute2f128_ps(ab_vec, ab_vec, 1), ab_vec);
    ab_vec = _mm256_hadd_ps(ab_vec, ab_vec);
    ab_vec = _mm256_hadd_ps(ab_vec, ab_vec);

    a2_vec = _mm256_add_ps(_mm256_permute2f128_ps(a2_vec, a2_vec, 1), a2_vec);
    a2_vec = _mm256_hadd_ps(a2_vec, a2_vec);
    a2_vec = _mm256_hadd_ps(a2_vec, a2_vec);

    b2_vec = _mm256_add_ps(_mm256_permute2f128_ps(b2_vec, b2_vec, 1), b2_vec);
    b2_vec = _mm256_hadd_ps(b2_vec, b2_vec);
    b2_vec = _mm256_hadd_ps(b2_vec, b2_vec);

    simsimd_f32_t ab, a2, b2;
    _mm_store_ss(&ab, _mm256_castps256_ps128(ab_vec));
    _mm_store_ss(&a2, _mm256_castps256_ps128(a2_vec));
    _mm_store_ss(&b2, _mm256_castps256_ps128(b2_vec));

    // Replace simsimd_approximate_inverse_square_root with `rsqrtss`
    __m128 a2_sqrt_recip = _mm_rsqrt_ss(_mm_set_ss((float)a2));
    __m128 b2_sqrt_recip = _mm_rsqrt_ss(_mm_set_ss((float)b2));
    __m128 result_vec = _mm_mul_ss(a2_sqrt_recip, b2_sqrt_recip); // Multiply the reciprocal square roots
    result_vec = _mm_mul_ss(result_vec, _mm_set_ss((float)ab));   // Multiply by ab
    result_vec = _mm_sub_ss(_mm_set_ss(1.0f), result_vec);        // Subtract from 1
    *result = ab != 0 ? _mm_cvtss_f32(result_vec) : 1;            // Extract the final result
}

SIMSIMD_PUBLIC void simsimd_l2sq_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                              simsimd_distance_t* result) {
    __m256 d2_vec = _mm256_setzero_ps();
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // Upcasting from `bf16` to `f32` is done by shifting the `bf16` values by 16 bits to the left, like:
        // x = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(x), 16))
        __m256 a_vec =
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i const*)(a + i))), 16));
        __m256 b_vec =
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i const*)(b + i))), 16));
        __m256 d_vec = _mm256_sub_ps(a_vec, b_vec);
        d2_vec = _mm256_fmadd_ps(d_vec, d_vec, d2_vec);
    }

    // In case the software emulation for `bf16` scalars is enabled, the `simsimd_uncompress_bf16`
    // function will run. It is extremely slow, so even for the tail, let's combine serial
    // loads and stores with vectorized math.
    if (i < n) {
        union {
            __m128i bf16_vec;
            simsimd_bf16_t bf16[8];
        } a_padded_tail, b_padded_tail;
        simsimd_size_t j = 0;
        for (; i < n; ++i, ++j)
            a_padded_tail.bf16[j] = a[i], b_padded_tail.bf16[j] = b[i];
        for (; j < 8; ++j)
            a_padded_tail.bf16[j] = 0, b_padded_tail.bf16[j] = 0;
        __m256 a_vec = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a_padded_tail.bf16_vec), 16));
        __m256 b_vec = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b_padded_tail.bf16_vec), 16));
        __m256 d_vec = _mm256_sub_ps(a_vec, b_vec);
        d2_vec = _mm256_fmadd_ps(d_vec, d_vec, d2_vec);
    }

    d2_vec = _mm256_add_ps(_mm256_permute2f128_ps(d2_vec, d2_vec, 1), d2_vec);
    d2_vec = _mm256_hadd_ps(d2_vec, d2_vec);
    d2_vec = _mm256_hadd_ps(d2_vec, d2_vec);

    simsimd_f32_t f32_result;
    _mm_store_ss(&f32_result, _mm256_castps256_ps128(d2_vec));
    *result = f32_result;
}

SIMSIMD_PUBLIC void simsimd_cos_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                             simsimd_distance_t* result) {

    __m256 ab_vec = _mm256_setzero_ps(), a2_vec = _mm256_setzero_ps(), b2_vec = _mm256_setzero_ps();
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // Upcasting from `bf16` to `f32` is done by shifting the `bf16` values by 16 bits to the left, like:
        // x = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(x), 16))
        __m256 a_vec =
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i const*)(a + i))), 16));
        __m256 b_vec =
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i const*)(b + i))), 16));
        ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
        a2_vec = _mm256_fmadd_ps(a_vec, a_vec, a2_vec);
        b2_vec = _mm256_fmadd_ps(b_vec, b_vec, b2_vec);
    }

    // In case the software emulation for `bf16` scalars is enabled, the `simsimd_uncompress_bf16`
    // function will run. It is extremely slow, so even for the tail, let's combine serial
    // loads and stores with vectorized math.
    if (i < n) {
        union {
            __m128i bf16_vec;
            simsimd_bf16_t bf16[8];
        } a_padded_tail, b_padded_tail;
        simsimd_size_t j = 0;
        for (; i < n; ++i, ++j)
            a_padded_tail.bf16[j] = a[i], b_padded_tail.bf16[j] = b[i];
        for (; j < 8; ++j)
            a_padded_tail.bf16[j] = 0, b_padded_tail.bf16[j] = 0;
        __m256 a_vec = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a_padded_tail.bf16_vec), 16));
        __m256 b_vec = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b_padded_tail.bf16_vec), 16));
        ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
        a2_vec = _mm256_fmadd_ps(a_vec, a_vec, a2_vec);
        b2_vec = _mm256_fmadd_ps(b_vec, b_vec, b2_vec);
    }

    // Horizontal reductions:
    ab_vec = _mm256_add_ps(_mm256_permute2f128_ps(ab_vec, ab_vec, 1), ab_vec);
    ab_vec = _mm256_hadd_ps(ab_vec, ab_vec);
    ab_vec = _mm256_hadd_ps(ab_vec, ab_vec);

    a2_vec = _mm256_add_ps(_mm256_permute2f128_ps(a2_vec, a2_vec, 1), a2_vec);
    a2_vec = _mm256_hadd_ps(a2_vec, a2_vec);
    a2_vec = _mm256_hadd_ps(a2_vec, a2_vec);

    b2_vec = _mm256_add_ps(_mm256_permute2f128_ps(b2_vec, b2_vec, 1), b2_vec);
    b2_vec = _mm256_hadd_ps(b2_vec, b2_vec);
    b2_vec = _mm256_hadd_ps(b2_vec, b2_vec);

    simsimd_f32_t ab, a2, b2;
    _mm_store_ss(&ab, _mm256_castps256_ps128(ab_vec));
    _mm_store_ss(&a2, _mm256_castps256_ps128(a2_vec));
    _mm_store_ss(&b2, _mm256_castps256_ps128(b2_vec));

    // Replace simsimd_approximate_inverse_square_root with `rsqrtss`
    __m128 a2_sqrt_recip = _mm_rsqrt_ss(_mm_set_ss((float)a2));
    __m128 b2_sqrt_recip = _mm_rsqrt_ss(_mm_set_ss((float)b2));
    __m128 result_vec = _mm_mul_ss(a2_sqrt_recip, b2_sqrt_recip); // Multiply the reciprocal square roots
    result_vec = _mm_mul_ss(result_vec, _mm_set_ss((float)ab));   // Multiply by ab
    result_vec = _mm_sub_ss(_mm_set_ss(1.0f), result_vec);        // Subtract from 1
    *result = ab != 0 ? _mm_cvtss_f32(result_vec) : 1;            // Extract the final result
}

SIMSIMD_PUBLIC void simsimd_l2sq_i8_haswell(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                            simsimd_distance_t* result) {

    __m256i d2_high_vec = _mm256_setzero_si256();
    __m256i d2_low_vec = _mm256_setzero_si256();

    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_vec = _mm256_loadu_si256((__m256i const*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((__m256i const*)(b + i));

        // Sign extend int8 to int16
        __m256i a_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a_vec));
        __m256i a_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_vec, 1));
        __m256i b_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_vec));
        __m256i b_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec, 1));

        // Subtract and multiply
        __m256i d_low = _mm256_sub_epi16(a_low, b_low);
        __m256i d_high = _mm256_sub_epi16(a_high, b_high);
        __m256i d2_low_part = _mm256_madd_epi16(d_low, d_low);
        __m256i d2_high_part = _mm256_madd_epi16(d_high, d_high);

        // Accumulate into int32 vectors
        d2_low_vec = _mm256_add_epi32(d2_low_vec, d2_low_part);
        d2_high_vec = _mm256_add_epi32(d2_high_vec, d2_high_part);
    }

    // Accumulate the 32-bit integers from `d2_high_vec` and `d2_low_vec`
    __m256i d2_vec = _mm256_add_epi32(d2_low_vec, d2_high_vec);
    __m128i d2_sum = _mm_add_epi32(_mm256_extracti128_si256(d2_vec, 0), _mm256_extracti128_si256(d2_vec, 1));
    d2_sum = _mm_hadd_epi32(d2_sum, d2_sum);
    d2_sum = _mm_hadd_epi32(d2_sum, d2_sum);
    int d2 = _mm_extract_epi32(d2_sum, 0);

    // Take care of the tail:
    for (; i < n; ++i) {
        int n = a[i] - b[i];
        d2 += n * n;
    }

    *result = (simsimd_f64_t)d2;
}

SIMSIMD_PUBLIC void simsimd_cos_i8_haswell(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                           simsimd_distance_t* result) {

    __m256i ab_low_vec = _mm256_setzero_si256();
    __m256i ab_high_vec = _mm256_setzero_si256();
    __m256i a2_low_vec = _mm256_setzero_si256();
    __m256i a2_high_vec = _mm256_setzero_si256();
    __m256i b2_low_vec = _mm256_setzero_si256();
    __m256i b2_high_vec = _mm256_setzero_si256();

    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_vec = _mm256_loadu_si256((__m256i const*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((__m256i const*)(b + i));

        // Unpack `int8` to `int16`
        __m256i a_low_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_vec, 0));
        __m256i a_high_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_vec, 1));
        __m256i b_low_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec, 0));
        __m256i b_high_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec, 1));

        // Multiply and accumulate as `int16`, accumulate products as `int32`
        ab_low_vec = _mm256_add_epi32(ab_low_vec, _mm256_madd_epi16(a_low_16, b_low_16));
        ab_high_vec = _mm256_add_epi32(ab_high_vec, _mm256_madd_epi16(a_high_16, b_high_16));
        a2_low_vec = _mm256_add_epi32(a2_low_vec, _mm256_madd_epi16(a_low_16, a_low_16));
        a2_high_vec = _mm256_add_epi32(a2_high_vec, _mm256_madd_epi16(a_high_16, a_high_16));
        b2_low_vec = _mm256_add_epi32(b2_low_vec, _mm256_madd_epi16(b_low_16, b_low_16));
        b2_high_vec = _mm256_add_epi32(b2_high_vec, _mm256_madd_epi16(b_high_16, b_high_16));
    }

    // Horizontal sum across the 256-bit register
    __m256i ab_vec = _mm256_add_epi32(ab_low_vec, ab_high_vec);
    __m128i ab_sum = _mm_add_epi32(_mm256_extracti128_si256(ab_vec, 0), _mm256_extracti128_si256(ab_vec, 1));
    ab_sum = _mm_hadd_epi32(ab_sum, ab_sum);
    ab_sum = _mm_hadd_epi32(ab_sum, ab_sum);

    __m256i a2_vec = _mm256_add_epi32(a2_low_vec, a2_high_vec);
    __m128i a2_sum = _mm_add_epi32(_mm256_extracti128_si256(a2_vec, 0), _mm256_extracti128_si256(a2_vec, 1));
    a2_sum = _mm_hadd_epi32(a2_sum, a2_sum);
    a2_sum = _mm_hadd_epi32(a2_sum, a2_sum);

    __m256i b2_vec = _mm256_add_epi32(b2_low_vec, b2_high_vec);
    __m128i b2_sum = _mm_add_epi32(_mm256_extracti128_si256(b2_vec, 0), _mm256_extracti128_si256(b2_vec, 1));
    b2_sum = _mm_hadd_epi32(b2_sum, b2_sum);
    b2_sum = _mm_hadd_epi32(b2_sum, b2_sum);

    // Further reduce to a single sum for each vector
    int ab = _mm_extract_epi32(ab_sum, 0);
    int a2 = _mm_extract_epi32(a2_sum, 0);
    int b2 = _mm_extract_epi32(b2_sum, 0);

    // Take care of the tail:
    for (; i < n; ++i) {
        int ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    // Compute the reciprocal of the square roots
    __m128 a2_sqrt_recip = _mm_rsqrt_ss(_mm_set_ss((float)a2));
    __m128 b2_sqrt_recip = _mm_rsqrt_ss(_mm_set_ss((float)b2));

    // Compute cosine similarity: ab / sqrt(a2 * b2)
    __m128 denom = _mm_mul_ss(a2_sqrt_recip, b2_sqrt_recip);      // Reciprocal of sqrt(a2 * b2)
    __m128 result_vec = _mm_mul_ss(_mm_set_ss((float)ab), denom); // ab * reciprocal of sqrt(a2 * b2)
    *result = ab != 0 ? 1 - _mm_cvtss_f32(result_vec) : 0;        // Extract the final result
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2sq_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                             simsimd_distance_t* result) {
    __m512 d2_vec = _mm512_setzero();
    __m512 a_vec, b_vec;

simsimd_l2sq_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    } else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    __m512 d_vec = _mm512_sub_ps(a_vec, b_vec);
    d2_vec = _mm512_fmadd_ps(d_vec, d_vec, d2_vec);
    if (n)
        goto simsimd_l2sq_f32_skylake_cycle;

    *result = _mm512_reduce_add_ps(d2_vec);
}

SIMSIMD_PUBLIC void simsimd_cos_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                            simsimd_distance_t* result) {
    __m512 ab_vec = _mm512_setzero();
    __m512 a2_vec = _mm512_setzero();
    __m512 b2_vec = _mm512_setzero();
    __m512 a_vec, b_vec;

simsimd_cos_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    } else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    ab_vec = _mm512_fmadd_ps(a_vec, b_vec, ab_vec);
    a2_vec = _mm512_fmadd_ps(a_vec, a_vec, a2_vec);
    b2_vec = _mm512_fmadd_ps(b_vec, b_vec, b2_vec);
    if (n)
        goto simsimd_cos_f32_skylake_cycle;

    simsimd_f32_t ab = _mm512_reduce_add_ps(ab_vec);
    simsimd_f32_t a2 = _mm512_reduce_add_ps(a2_vec);
    simsimd_f32_t b2 = _mm512_reduce_add_ps(b2_vec);

    // Compute the reciprocal square roots of a2 and b2
    // Mysteriously, MSVC has no `_mm_rsqrt14_ps` intrinsic, but has it's masked variants,
    // so let's use `_mm_maskz_rsqrt14_ps(0xFF, ...)` instead.
    __m128 rsqrts = _mm_maskz_rsqrt14_ps(0xFF, _mm_set_ps(0.f, 0.f, a2 + 1.e-9f, b2 + 1.e-9f));
    simsimd_f32_t rsqrt_a2 = _mm_cvtss_f32(rsqrts);
    simsimd_f32_t rsqrt_b2 = _mm_cvtss_f32(_mm_shuffle_ps(rsqrts, rsqrts, _MM_SHUFFLE(0, 0, 0, 1)));
    *result = 1 - ab * rsqrt_a2 * rsqrt_b2;
}

SIMSIMD_PUBLIC void simsimd_l2sq_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                             simsimd_distance_t* result) {
    __m512d d2_vec = _mm512_setzero_pd();
    __m512d a_vec, b_vec;

simsimd_l2sq_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_pd(mask, a);
        b_vec = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    } else {
        a_vec = _mm512_loadu_pd(a);
        b_vec = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    __m512d d_vec = _mm512_sub_pd(a_vec, b_vec);
    d2_vec = _mm512_fmadd_pd(d_vec, d_vec, d2_vec);
    if (n)
        goto simsimd_l2sq_f64_skylake_cycle;

    *result = _mm512_reduce_add_pd(d2_vec);
}

SIMSIMD_PUBLIC void simsimd_cos_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                            simsimd_distance_t* result) {
    __m512d ab_vec = _mm512_setzero_pd();
    __m512d a2_vec = _mm512_setzero_pd();
    __m512d b2_vec = _mm512_setzero_pd();
    __m512d a_vec, b_vec;

simsimd_cos_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_pd(mask, a);
        b_vec = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    } else {
        a_vec = _mm512_loadu_pd(a);
        b_vec = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    ab_vec = _mm512_fmadd_pd(a_vec, b_vec, ab_vec);
    a2_vec = _mm512_fmadd_pd(a_vec, a_vec, a2_vec);
    b2_vec = _mm512_fmadd_pd(b_vec, b_vec, b2_vec);
    if (n)
        goto simsimd_cos_f64_skylake_cycle;

    simsimd_f32_t ab = (simsimd_f32_t)_mm512_reduce_add_pd(ab_vec);
    simsimd_f32_t a2 = (simsimd_f32_t)_mm512_reduce_add_pd(a2_vec);
    simsimd_f32_t b2 = (simsimd_f32_t)_mm512_reduce_add_pd(b2_vec);

    // Compute the reciprocal square roots of a2 and b2
    // Mysteriously, MSVC has no `_mm_rsqrt14_ps` intrinsic, but has it's masked variants,
    // so let's use `_mm_maskz_rsqrt14_ps(0xFF, ...)` instead.
    __m128 rsqrts = _mm_maskz_rsqrt14_ps(0xFF, _mm_set_ps(0.f, 0.f, a2 + 1.e-9f, b2 + 1.e-9f));
    simsimd_f32_t rsqrt_a2 = _mm_cvtss_f32(rsqrts);
    simsimd_f32_t rsqrt_b2 = _mm_cvtss_f32(_mm_shuffle_ps(rsqrts, rsqrts, _MM_SHUFFLE(0, 0, 0, 1)));
    *result = 1 - ab * rsqrt_a2 * rsqrt_b2;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_GENOA
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512bf16")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512bf16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2sq_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                            simsimd_distance_t* result) {
    __m512 d2_top_vec = _mm512_setzero_ps(), d2_bot_vec = _mm512_setzero_ps();
    __m512 d_top_vec = _mm512_setzero_ps(), d_bot_vec = _mm512_setzero_ps();
    __m512 a_f32_top_vec, a_f32_bot_vec, b_f32_top_vec, b_f32_bot_vec;
    __m512i a_i16_vec, b_i16_vec;

simsimd_l2sq_bf16_genoa_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    } else {
        a_i16_vec = _mm512_loadu_epi16(a);
        b_i16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    // Let's perform the subtraction with single-precision, while the dot-product with half-precision.
    // For that we need to perform a couple of casts - each is a bitshift. To convert `bf16` to `f32`,
    // expand it to 32-bit integers, then shift the bits by 16 to the left. Then subtract as floats,
    // and shift back. During expansion, we will double the space, and should use separate registers
    // for top and bottom halves.
    a_f32_bot_vec =
        _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(a_i16_vec)), 16));
    b_f32_bot_vec =
        _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(b_i16_vec)), 16));

    // Some compilers don't have `_mm512_extracti32x8_epi32`, so we need to use `_mm512_extracti64x4_epi64`
    a_f32_top_vec =
        _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(a_i16_vec, 1)), 16));
    b_f32_top_vec =
        _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(b_i16_vec, 1)), 16));

    // Subtract and cast back
    d_top_vec = _mm512_sub_ps(a_f32_top_vec, b_f32_top_vec);
    d_bot_vec = _mm512_sub_ps(a_f32_bot_vec, b_f32_bot_vec);
    d_top_vec = _mm512_castsi512_ps(_mm512_srli_epi32(_mm512_castps_si512(d_top_vec), 16));
    d_bot_vec = _mm512_castsi512_ps(_mm512_srli_epi32(_mm512_castps_si512(d_bot_vec), 16));

    // Square and accumulate
    d2_top_vec = _mm512_dpbf16_ps(d2_top_vec, (__m512bh)(d_top_vec), (__m512bh)(d_top_vec));
    d2_bot_vec = _mm512_dpbf16_ps(d2_bot_vec, (__m512bh)(d_bot_vec), (__m512bh)(d_bot_vec));
    if (n)
        goto simsimd_l2sq_bf16_genoa_cycle;

    *result = _mm512_reduce_add_ps(d2_top_vec) + _mm512_reduce_add_ps(d2_bot_vec);
}

SIMSIMD_PUBLIC void simsimd_cos_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                           simsimd_distance_t* result) {
    __m512 ab_vec = _mm512_setzero_ps();
    __m512 a2_vec = _mm512_setzero_ps();
    __m512 b2_vec = _mm512_setzero_ps();
    __m512i a_i16_vec, b_i16_vec;

simsimd_cos_bf16_genoa_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    } else {
        a_i16_vec = _mm512_loadu_epi16(a);
        b_i16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    ab_vec = _mm512_dpbf16_ps(ab_vec, (__m512bh)(a_i16_vec), (__m512bh)(b_i16_vec));
    a2_vec = _mm512_dpbf16_ps(a2_vec, (__m512bh)(a_i16_vec), (__m512bh)(a_i16_vec));
    b2_vec = _mm512_dpbf16_ps(b2_vec, (__m512bh)(b_i16_vec), (__m512bh)(b_i16_vec));
    if (n)
        goto simsimd_cos_bf16_genoa_cycle;

    simsimd_f32_t ab = _mm512_reduce_add_ps(ab_vec);
    simsimd_f32_t a2 = _mm512_reduce_add_ps(a2_vec);
    simsimd_f32_t b2 = _mm512_reduce_add_ps(b2_vec);

    // Compute the reciprocal square roots of a2 and b2
    __m128 rsqrts = _mm_rsqrt14_ps(_mm_set_ps(0.f, 0.f, a2 + 1.e-9f, b2 + 1.e-9f));
    simsimd_f32_t rsqrt_a2 = _mm_cvtss_f32(rsqrts);
    simsimd_f32_t rsqrt_b2 = _mm_cvtss_f32(_mm_shuffle_ps(rsqrts, rsqrts, _MM_SHUFFLE(0, 0, 0, 1)));
    *result = ab != 0 ? 1 - ab * rsqrt_a2 * rsqrt_b2 : 0;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_GENOA

#if SIMSIMD_TARGET_SAPPHIRE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2sq_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                              simsimd_distance_t* result) {
    __m512h d2_vec = _mm512_setzero_ph();
    __m512i a_i16_vec, b_i16_vec;

simsimd_l2sq_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    } else {
        a_i16_vec = _mm512_loadu_epi16(a);
        b_i16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    __m512h d_vec = _mm512_sub_ph(_mm512_castsi512_ph(a_i16_vec), _mm512_castsi512_ph(b_i16_vec));
    d2_vec = _mm512_fmadd_ph(d_vec, d_vec, d2_vec);
    if (n)
        goto simsimd_l2sq_f16_sapphire_cycle;

    *result = _mm512_reduce_add_ph(d2_vec);
}

SIMSIMD_PUBLIC void simsimd_cos_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                             simsimd_distance_t* result) {
    __m512h ab_vec = _mm512_setzero_ph();
    __m512h a2_vec = _mm512_setzero_ph();
    __m512h b2_vec = _mm512_setzero_ph();
    __m512i a_i16_vec, b_i16_vec;

simsimd_cos_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    } else {
        a_i16_vec = _mm512_loadu_epi16(a);
        b_i16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    ab_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_i16_vec), _mm512_castsi512_ph(b_i16_vec), ab_vec);
    a2_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_i16_vec), _mm512_castsi512_ph(a_i16_vec), a2_vec);
    b2_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(b_i16_vec), _mm512_castsi512_ph(b_i16_vec), b2_vec);
    if (n)
        goto simsimd_cos_f16_sapphire_cycle;

    simsimd_f32_t ab = _mm512_reduce_add_ph(ab_vec);
    simsimd_f32_t a2 = _mm512_reduce_add_ph(a2_vec);
    simsimd_f32_t b2 = _mm512_reduce_add_ph(b2_vec);

    // Compute the reciprocal square roots of a2 and b2
    __m128 rsqrts = _mm_rsqrt14_ps(_mm_set_ps(0.f, 0.f, a2 + 1.e-9f, b2 + 1.e-9f));
    simsimd_f32_t rsqrt_a2 = _mm_cvtss_f32(rsqrts);
    simsimd_f32_t rsqrt_b2 = _mm_cvtss_f32(_mm_shuffle_ps(rsqrts, rsqrts, _MM_SHUFFLE(0, 0, 0, 1)));
    *result = ab != 0 ? 1 - ab * rsqrt_a2 * rsqrt_b2 : 0;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SAPPHIRE

#if SIMSIMD_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2sq_i8_ice(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                        simsimd_distance_t* result) {
    __m512i d2_i32s_vec = _mm512_setzero_si512();
    __m512i a_vec, b_vec, d_i16s_vec;

simsimd_l2sq_i8_ice_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a));
        b_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b));
        n = 0;
    } else {
        a_vec = _mm512_cvtepi8_epi16(_mm256_loadu_epi8(a));
        b_vec = _mm512_cvtepi8_epi16(_mm256_loadu_epi8(b));
        a += 32, b += 32, n -= 32;
    }
    d_i16s_vec = _mm512_sub_epi16(a_vec, b_vec);
    d2_i32s_vec = _mm512_dpwssd_epi32(d2_i32s_vec, d_i16s_vec, d_i16s_vec);
    if (n)
        goto simsimd_l2sq_i8_ice_cycle;

    *result = _mm512_reduce_add_epi32(d2_i32s_vec);
}

SIMSIMD_PUBLIC void simsimd_cos_i8_ice(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                       simsimd_distance_t* result) {
    __m512i ab_low_i32s_vec = _mm512_setzero_si512();
    __m512i ab_high_i32s_vec = _mm512_setzero_si512();
    __m512i a2_i32s_vec = _mm512_setzero_si512();
    __m512i b2_i32s_vec = _mm512_setzero_si512();
    __m512i a_vec, b_vec;
    __m512i a_abs_vec, b_abs_vec;

simsimd_cos_i8_ice_cycle:
    if (n < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_vec = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    } else {
        a_vec = _mm512_loadu_epi8(a);
        b_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }

    // We can't directly use the `_mm512_dpbusd_epi32` intrinsic everywhere,
    // as it's asymmetric with respect to the sign of the input arguments:
    //      Signed(ZeroExtend16(a.byte[4*j]) * SignExtend16(b.byte[4*j]))
    // Luckily to compute the squares, we just drop the sign bit of the second argument.
    a_abs_vec = _mm512_abs_epi8(a_vec);
    b_abs_vec = _mm512_abs_epi8(b_vec);
    a2_i32s_vec = _mm512_dpbusds_epi32(a2_i32s_vec, a_abs_vec, a_abs_vec);
    b2_i32s_vec = _mm512_dpbusds_epi32(b2_i32s_vec, b_abs_vec, b_abs_vec);

    // The same trick won't work for the primary dot-product, as the signs vector
    // components may differ significantly. So we have to use two `_mm512_dpwssd_epi32`
    // intrinsics instead, upcasting four chunks to 16-bit integers beforehand!
    ab_low_i32s_vec = _mm512_dpwssds_epi32(                  //
        ab_low_i32s_vec,                                     //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_vec)), //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_vec)));
    ab_high_i32s_vec = _mm512_dpwssds_epi32(                       //
        ab_high_i32s_vec,                                          //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_vec, 1)), //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_vec, 1)));
    if (n)
        goto simsimd_cos_i8_ice_cycle;

    int ab = _mm512_reduce_add_epi32(_mm512_add_epi32(ab_low_i32s_vec, ab_high_i32s_vec));
    int a2 = _mm512_reduce_add_epi32(a2_i32s_vec);
    int b2 = _mm512_reduce_add_epi32(b2_i32s_vec);

    // Compute the reciprocal square roots of a2 and b2
    // Mysteriously, MSVC has no `_mm_rsqrt14_ps` intrinsic, but has it's masked variants,
    // so let's use `_mm_maskz_rsqrt14_ps(0xFF, ...)` instead.
    __m128 rsqrts = _mm_maskz_rsqrt14_ps(0xFF, _mm_set_ps(0.f, 0.f, a2 + 1.e-9f, b2 + 1.e-9f));
    simsimd_f32_t rsqrt_a2 = _mm_cvtss_f32(rsqrts);
    simsimd_f32_t rsqrt_b2 = _mm_cvtss_f32(_mm_shuffle_ps(rsqrts, rsqrts, _MM_SHUFFLE(0, 0, 0, 1)));
    *result = ab != 0 ? 1 - ab * rsqrt_a2 * rsqrt_b2 : 0;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_ICE
#endif // SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif

#endif
