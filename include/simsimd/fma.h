/**
 *  @file       fma.h
 *  @brief      SIMD-accelerated mixed-precision Fused-Multiply-Add operations.
 *  @author     Ash Vardanian
 *  @date       October 16, 2024
 *
 *  Contains following element-wise operations:
 *  - WSum or Weighted-Sum: R[i] = Alpha * A[i] + Beta * B[i]
 *  - FMA or Fused-Multiply-Add: R[i] = Alpha * A[i] * B[i] + Beta * C[i]
 *
 *  This tiny set of operations if enough to implement a wide range of algorithms.
 *  To scale a vector by a scalar, just call WSum with $Beta$ = 0.
 *  To sum two vectors, just call WSum with $Alpha$ = $Beta$ = 1.
 *  To average two vectors, just call WSum with $Alpha$ = $Beta$ = 0.5.
 *  To multiply vectors element-wise, just call FMA with $Beta$ = 0.
 *
 *  For datatypes:
 *  - 64-bit IEEE floating point numbers
 *  - 32-bit IEEE floating point numbers
 *  - 16-bit IEEE floating point numbers
 *  - 16-bit brain floating point numbers
 *  - 8-bit unsigned integers
 *  - 8-bit signed integers
 *
 *  For hardware architectures:
 *  - Arm: NEON, SVE
 *  - x86: Haswell, Ice Lake, Skylake, Genoa, Sapphire
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef SIMSIMD_FMA_H
#define SIMSIMD_FMA_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

SIMSIMD_PUBLIC void simsimd_wsum_f64_serial(                          //
    simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t* result);
SIMSIMD_PUBLIC void simsimd_wsum_f32_serial(                          //
    simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t* result);
SIMSIMD_PUBLIC void simsimd_wsum_f16_serial(                          //
    simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t* result);
SIMSIMD_PUBLIC void simsimd_wsum_bf16_serial(                           //
    simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t* result);
SIMSIMD_PUBLIC void simsimd_wsum_i8_serial(                         //
    simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t* result);
SIMSIMD_PUBLIC void simsimd_wsum_u8_serial(                         //
    simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t* result);

SIMSIMD_PUBLIC void simsimd_fma_f64_serial(                                                   //
    simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_f64_t const* c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t* result);
SIMSIMD_PUBLIC void simsimd_fma_f32_serial(                                                   //
    simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t* result);
SIMSIMD_PUBLIC void simsimd_fma_f16_serial(                                                   //
    simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t* result);
SIMSIMD_PUBLIC void simsimd_fma_bf16_serial(                                                     //
    simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t* result);
SIMSIMD_PUBLIC void simsimd_fma_i8_serial(                                                 //
    simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_i8_t const* c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t* result);
SIMSIMD_PUBLIC void simsimd_fma_u8_serial(                                                 //
    simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_u8_t const* c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t* result);

#define SIMSIMD_MAKE_WSUM(name, input_type, accumulator_type, load_and_convert, convert_and_store)                     \
    SIMSIMD_PUBLIC void simsimd_wsum_##input_type##_##name(                                                            \
        simsimd_##input_type##_t const* a, simsimd_##input_type##_t const* b, simsimd_size_t n,                        \
        simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_##input_type##_t* result) {                         \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                      \
            simsimd_##accumulator_type##_t ai = load_and_convert(a + i);                                               \
            simsimd_##accumulator_type##_t bi = load_and_convert(b + i);                                               \
            simsimd_##accumulator_type##_t ai_weighted = (simsimd_##accumulator_type##_t)(ai * alpha);                 \
            simsimd_##accumulator_type##_t bi_weighted = (simsimd_##accumulator_type##_t)(bi * beta);                  \
            simsimd_##accumulator_type##_t sum = ai_weighted + bi_weighted;                                            \
            convert_and_store(sum, result + i);                                                                        \
        }                                                                                                              \
    }

#define SIMSIMD_MAKE_FMA(name, input_type, accumulator_type, load_and_convert, convert_and_store)                      \
    SIMSIMD_PUBLIC void simsimd_fma_##input_type##_##name(                                                             \
        simsimd_##input_type##_t const* a, simsimd_##input_type##_t const* b, simsimd_##input_type##_t const* c,       \
        simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_##input_type##_t* result) {       \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                      \
            simsimd_##accumulator_type##_t ai = load_and_convert(a + i);                                               \
            simsimd_##accumulator_type##_t bi = load_and_convert(b + i);                                               \
            simsimd_##accumulator_type##_t ci = load_and_convert(c + i);                                               \
            simsimd_##accumulator_type##_t abi_weighted = (simsimd_##accumulator_type##_t)(ai * bi * alpha);           \
            simsimd_##accumulator_type##_t ci_weighted = (simsimd_##accumulator_type##_t)(ci * beta);                  \
            simsimd_##accumulator_type##_t sum = abi_weighted + ci_weighted;                                           \
            convert_and_store(sum, result + i);                                                                        \
        }                                                                                                              \
    }

SIMSIMD_MAKE_WSUM(serial, f64, f64, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT)       // simsimd_wsum_f64_serial
SIMSIMD_MAKE_WSUM(serial, f32, f32, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT)       // simsimd_wsum_f32_serial
SIMSIMD_MAKE_WSUM(serial, f16, f32, SIMSIMD_F16_TO_F32, SIMSIMD_F32_TO_F16)    // simsimd_wsum_f16_serial
SIMSIMD_MAKE_WSUM(serial, bf16, f32, SIMSIMD_BF16_TO_F32, SIMSIMD_F32_TO_BF16) // simsimd_wsum_bf16_serial
SIMSIMD_MAKE_WSUM(serial, i8, f32, SIMSIMD_DEREFERENCE, SIMSIMD_F32_TO_I8)     // simsimd_wsum_i8_serial
SIMSIMD_MAKE_WSUM(serial, u8, f32, SIMSIMD_DEREFERENCE, SIMSIMD_F32_TO_U8)     // simsimd_wsum_u8_serial

SIMSIMD_MAKE_WSUM(accurate, f32, f64, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT)       // simsimd_wsum_f32_accurate
SIMSIMD_MAKE_WSUM(accurate, f16, f64, SIMSIMD_F16_TO_F32, SIMSIMD_F32_TO_F16)    // simsimd_wsum_f16_accurate
SIMSIMD_MAKE_WSUM(accurate, bf16, f64, SIMSIMD_BF16_TO_F32, SIMSIMD_F32_TO_BF16) // simsimd_wsum_bf16_accurate
SIMSIMD_MAKE_WSUM(accurate, i8, f64, SIMSIMD_DEREFERENCE, SIMSIMD_F64_TO_I8)     // simsimd_wsum_i8_accurate
SIMSIMD_MAKE_WSUM(accurate, u8, f64, SIMSIMD_DEREFERENCE, SIMSIMD_F64_TO_U8)     // simsimd_wsum_u8_accurate

SIMSIMD_MAKE_FMA(serial, f64, f64, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT)       // simsimd_fma_f64_serial
SIMSIMD_MAKE_FMA(serial, f32, f32, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT)       // simsimd_fma_f32_serial
SIMSIMD_MAKE_FMA(serial, f16, f32, SIMSIMD_F16_TO_F32, SIMSIMD_F32_TO_F16)    // simsimd_fma_f16_serial
SIMSIMD_MAKE_FMA(serial, bf16, f32, SIMSIMD_BF16_TO_F32, SIMSIMD_F32_TO_BF16) // simsimd_fma_bf16_serial
SIMSIMD_MAKE_FMA(serial, i8, f32, SIMSIMD_DEREFERENCE, SIMSIMD_F32_TO_I8)     // simsimd_fma_i8_serial
SIMSIMD_MAKE_FMA(serial, u8, f32, SIMSIMD_DEREFERENCE, SIMSIMD_F32_TO_U8)     // simsimd_fma_u8_serial

SIMSIMD_MAKE_FMA(accurate, f32, f64, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT)       // simsimd_fma_f32_accurate
SIMSIMD_MAKE_FMA(accurate, f16, f64, SIMSIMD_F16_TO_F32, SIMSIMD_F32_TO_F16)    // simsimd_fma_f16_accurate
SIMSIMD_MAKE_FMA(accurate, bf16, f64, SIMSIMD_BF16_TO_F32, SIMSIMD_F32_TO_BF16) // simsimd_fma_bf16_accurate
SIMSIMD_MAKE_FMA(accurate, i8, f64, SIMSIMD_DEREFERENCE, SIMSIMD_F64_TO_I8)     // simsimd_fma_i8_accurate
SIMSIMD_MAKE_FMA(accurate, u8, f64, SIMSIMD_DEREFERENCE, SIMSIMD_F64_TO_U8)     // simsimd_fma_u8_accurate

SIMSIMD_PUBLIC void simsimd_wsum_f64_haswell(                         //
    simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t* result);
SIMSIMD_PUBLIC void simsimd_wsum_f32_haswell(                         //
    simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t* result);
SIMSIMD_PUBLIC void simsimd_wsum_f16_haswell(                         //
    simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t* result);
SIMSIMD_PUBLIC void simsimd_wsum_bf16_haswell(                          //
    simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t* result);
SIMSIMD_PUBLIC void simsimd_wsum_i8_haswell(                        //
    simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t* result);
SIMSIMD_PUBLIC void simsimd_wsum_u8_haswell(                        //
    simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t* result);

SIMSIMD_PUBLIC void simsimd_fma_f64_haswell(                                                  //
    simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_f64_t const* c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t* result);
SIMSIMD_PUBLIC void simsimd_fma_f32_haswell(                                                  //
    simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t* result);
SIMSIMD_PUBLIC void simsimd_fma_f16_haswell(                                                  //
    simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t* result);
SIMSIMD_PUBLIC void simsimd_fma_bf16_haswell(                                                    //
    simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t* result);
SIMSIMD_PUBLIC void simsimd_fma_i8_haswell(                                                //
    simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_i8_t const* c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t* result);
SIMSIMD_PUBLIC void simsimd_fma_u8_haswell(                                                //
    simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_u8_t const* c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t* result);

#if SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_wsum_f32_haswell(                         //
    simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t* result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 a_scaled = _mm256_mul_ps(a_vec, alpha_vec);
        __m256 b_scaled = _mm256_mul_ps(b_vec, beta_vec);
        __m256 sum_vec = _mm256_add_ps(a_scaled, b_scaled);
        _mm256_storeu_ps(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i)
        result[i] = alpha_f32 * a[i] + beta_f32 * b[i];
}

SIMSIMD_PUBLIC void simsimd_wsum_f64_haswell(                         //
    simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t* result) {
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(a + i);
        __m256d b_vec = _mm256_loadu_pd(b + i);
        __m256d a_scaled = _mm256_mul_pd(a_vec, alpha_vec);
        __m256d b_scaled = _mm256_mul_pd(b_vec, beta_vec);
        __m256d sum_vec = _mm256_add_pd(a_scaled, b_scaled);
        _mm256_storeu_pd(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i)
        result[i] = alpha * a[i] + beta * b[i];
}

SIMSIMD_PUBLIC void simsimd_wsum_f16_haswell(                         //
    simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t* result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16 = _mm_loadu_si128((__m128i const*)(a + i));
        __m128i b_f16 = _mm_loadu_si128((__m128i const*)(b + i));
        __m256 a_vec = _mm256_cvtph_ps(a_f16);
        __m256 b_vec = _mm256_cvtph_ps(b_f16);
        __m256 a_scaled = _mm256_mul_ps(a_vec, alpha_vec);
        __m256 b_scaled = _mm256_mul_ps(b_vec, beta_vec);
        __m256 sum_vec = _mm256_add_ps(a_scaled, b_scaled);
        __m128i sum_f16 = _mm256_cvtps_ph(sum_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i*)(result + i), sum_f16);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = SIMSIMD_F16_TO_F32(a + i);
        simsimd_f32_t bi = SIMSIMD_F16_TO_F32(b + i);
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32 * bi;
        SIMSIMD_F32_TO_F16(sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_wsum_bf16_haswell(                          //
    simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t* result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16 = _mm_loadu_si128((__m128i const*)(a + i));
        __m128i b_bf16 = _mm_loadu_si128((__m128i const*)(b + i));
        __m256 a_vec = _simsimd_bf16x8_to_f32x8_haswell(a_bf16);
        __m256 b_vec = _simsimd_bf16x8_to_f32x8_haswell(b_bf16);
        __m256 a_scaled = _mm256_mul_ps(a_vec, alpha_vec);
        __m256 b_scaled = _mm256_mul_ps(b_vec, beta_vec);
        __m256 sum_vec = _mm256_add_ps(a_scaled, b_scaled);
        __m128i sum_bf16 = _simsimd_f32x8_to_bf16x8_haswell(sum_vec);
        _mm_storeu_si128((__m128i*)(result + i), sum_bf16);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = SIMSIMD_BF16_TO_F32(a + i);
        simsimd_f32_t bi = SIMSIMD_BF16_TO_F32(b + i);
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32 * bi;
        SIMSIMD_F32_TO_BF16(sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_wsum_i8_haswell(                        //
    simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t* result) {

    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // Assuming the "alpha" and "beta" are floating-point values, we will need
    // to convert the 8-bit integers to 32-bit integers first, and then use
    // the `_mm256_cvtepi32_ps`, which maps to "VCVTDQ2PS_EVEX (YMM, YMM)":
    //
    // - On Ice Lake: 4 cycles latency, ports: 1*p01
    // - On Genoa: 3 cycles latency, ports: 1*FP23
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_i8_vec = _mm_loadu_si128((__m128i const*)(a + i));
        __m128i b_i8_vec = _mm_loadu_si128((__m128i const*)(b + i));
        __m256i a_i32_low_vec = _mm256_cvtepi8_epi32(a_i8_vec);
        __m256i a_i32_high_vec = _mm256_cvtepi8_epi32(_mm256_permute4x64_epi64(a_i8_vec, _MM_MASK_SHUFFLE(1, 0, 3, 2)));
        __m256i b_i32_low_vec = _mm256_cvtepi8_epi32(b_i8_vec);
        __m256i b_i32_high_vec = _mm256_cvtepi8_epi32(_mm256_permute4x64_epi64(b_i8_vec, _MM_MASK_SHUFFLE(1, 0, 3, 2)));
        __m256 a_f32_low_vec = _mm256_cvtepi32_ps(a_i32_low_vec);
        __m256 a_f32_high_vec = _mm256_cvtepi32_ps(a_i32_high_vec);
        __m256 b_f32_low_vec = _mm256_cvtepi32_ps(b_i32_low_vec);
        __m256 b_f32_high_vec = _mm256_cvtepi32_ps(b_i32_high_vec);
        __m256 a_low_scaled = _mm256_mul_ps(a_low_vec, alpha_vec);
        __m256 a_high_scaled = _mm256_mul_ps(a_high_vec, alpha_vec);
        __m256 b_low_scaled = _mm256_mul_ps(b_low_vec, beta_vec);
        __m256 b_high_scaled = _mm256_mul_ps(b_high_vec, beta_vec);
        __m256 sum_low_vec = _mm256_add_ps(a_low_scaled, b_low_scaled);
        __m256 sum_high_vec = _mm256_add_ps(a_high_scaled, b_hight_scaled);
        // Now we need to convert the floats back to 8-bit integers:
        __m256i sum_low_i32_vec = _mm256_cvtps_epi32(sum_low_vec);
        __m256i sum_high_i32_vec = _mm256_cvtps_epi32(sum_high_vec);
        // TODO: Finish later
        // The packing instruction orders data within lanes
        __m256i sum_i16_vec = _mm256_packs_epi32( //
            _mm256_permute2x128_si256(sum_low_i32_vec, sum_high_i32_vec, _MM_MASK_SHUFFLE()),
            _mm256_permute2x128_si256(sum_low_i32_vec, sum_high_i32_vec, _MM_MASK_SHUFFLE()));

        _mm_storeu_si128((__m128i*)(result + i), );
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i], bi = b[i];
        result[i] = (simsimd_i8_t)(alpha_f32 * ai + beta_f32 * bi);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_f32_haswell(                                //
    simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t* result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 c_vec = _mm256_loadu_ps(c + i);
        __m256 ab_vec = _mm256_mul_ps(a_vec, b_vec);
        __m256 ab_scaled = _mm256_mul_ps(ab_vec, alpha_vec);
        __m256 c_scaled = _mm256_mul_ps(c_vec, beta_vec);
        __m256 sum_vec = _mm256_add_ps(ab_scaled, c_scaled);
        _mm256_storeu_ps(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i)
        result[i] = alpha_f32 * a[i] * b[i] + beta_f32 * c[i];
}

SIMSIMD_PUBLIC void simsimd_fma_f64_haswell(                                //
    simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_f64_t const* c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t* result) {
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(a + i);
        __m256d b_vec = _mm256_loadu_pd(b + i);
        __m256d c_vec = _mm256_loadu_pd(c + i);
        __m256d ab_vec = _mm256_mul_pd(a_vec, b_vec);
        __m256d ab_scaled = _mm256_mul_pd(ab_vec, alpha_vec);
        __m256d c_scaled = _mm256_mul_pd(c_vec, beta_vec);
        __m256d sum_vec = _mm256_add_pd(ab_scaled, c_scaled);
        _mm256_storeu_pd(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i)
        result[i] = alpha * a[i] * b[i] + beta * c[i];
}

SIMSIMD_PUBLIC void simsimd_fma_f16_haswell(                                //
    simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t* result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16 = _mm_loadu_si128((__m128i const*)(a + i));
        __m128i b_f16 = _mm_loadu_si128((__m128i const*)(b + i));
        __m128i c_f16 = _mm_loadu_si128((__m128i const*)(c + i));
        __m256 a_vec = _mm256_cvtph_ps(a_f16);
        __m256 b_vec = _mm256_cvtph_ps(b_f16);
        __m256 c_vec = _mm256_cvtph_ps(c_f16);
        __m256 ab_vec = _mm256_mul_ps(a_vec, b_vec);
        __m256 ab_scaled = _mm256_mul_ps(ab_vec, alpha_vec);
        __m256 c_scaled = _mm256_mul_ps(c_vec, beta_vec);
        __m256 sum_vec = _mm256_add_ps(ab_scaled, c_scaled);
        __m128i sum_f16 = _mm256_cvtps_ph(sum_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i*)(result + i), sum_f16);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = SIMSIMD_F16_TO_F32(a + i);
        simsimd_f32_t bi = SIMSIMD_F16_TO_F32(b + i);
        simsimd_f32_t ci = SIMSIMD_F16_TO_F32(c + i);
        simsimd_f32_t sum = alpha * ai * bi + beta * ci;
        SIMSIMD_F32_TO_F16(sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_bf16_haswell(                                  //
    simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_bf16_t const* c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t* result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16 = _mm_loadu_si128((__m128i const*)(a + i));
        __m128i b_bf16 = _mm_loadu_si128((__m128i const*)(b + i));
        __m128i c_bf16 = _mm_loadu_si128((__m128i const*)(c + i));
        __m256 a_vec = _simsimd_bf16x8_to_f32x8_haswell(a_bf16);
        __m256 b_vec = _simsimd_bf16x8_to_f32x8_haswell(b_bf16);
        __m256 c_vec = _simsimd_bf16x8_to_f32x8_haswell(c_bf16);
        __m256 ab_vec = _mm256_mul_ps(a_vec, b_vec);
        __m256 ab_scaled = _mm256_mul_ps(ab_vec, alpha_vec);
        __m256 c_scaled = _mm256_mul_ps(c_vec, beta_vec);
        __m256 sum_vec = _mm256_add_ps(ab_scaled, c_scaled);
        __m128i sum_bf16 = _simsimd_f32x8_to_bf16x8_haswell(sum_vec);
        _mm_storeu_si128((__m128i*)(result + i), sum_bf16);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = SIMSIMD_BF16_TO_F32(a + i);
        simsimd_f32_t bi = SIMSIMD_BF16_TO_F32(b + i);
        simsimd_f32_t ci = SIMSIMD_BF16_TO_F32(c + i);
        simsimd_f32_t sum = alpha * ai * bi + beta * ci;
        SIMSIMD_F32_TO_BF16(sum, result + i);
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL
#endif // SIMSIMD_TARGET_X86

#if SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_wsum_f32_neon(                            //
    simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t* result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t a_scaled_vec = vmulq_n_f32(a_vec, alpha_f32);
        float32x4_t b_scaled_vec = vmulq_n_f32(b_vec, beta_f32);
        float32x4_t sum_vec = vaddq_f32(a_scaled_vec, b_scaled_vec);
        vst1q_f32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i)
        result[i] = alpha_f32 * a[i] + beta_f32 * b[i];
}

SIMSIMD_PUBLIC void simsimd_fma_f32_neon(                                   //
    simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_f32_t const* c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t* result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t c_vec = vld1q_f32(c + i);
        float32x4_t ab_vec = vmulq_f32(a_vec, b_vec);
        float32x4_t ab_scaled_vec = vmulq_n_f32(ab_vec, alpha_f32);
        float32x4_t sum_vec = vfmaq_n_f32(ab_scaled_vec, c_vec, beta_f32);
        vst1q_f32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i)
        result[i] = alpha_f32 * a[i] * b[i] + beta_f32 * c[i];
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_wsum_f16_neon(                            //
    simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t* result) {
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_vec = vld1q_f16(a + i);
        float16x8_t b_vec = vld1q_f16(b + i);
        float16x8_t a_scaled_vec = vmulq_n_f16(a_vec, alpha_f16);
        float16x8_t b_scaled_vec = vmulq_n_f16(b_vec, beta_f16);
        float16x8_t sum_vec = vaddq_f16(a_scaled_vec, b_scaled_vec);
        vst1q_f16(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i)
        ((float16_t*)result)[i] = alpha_f16 * ((float16_t const*)a)[i] + beta_f16 * ((float16_t const*)b)[i];
}

SIMSIMD_PUBLIC void simsimd_fma_f16_neon(                                   //
    simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t const* c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t* result) {
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_vec = vld1q_f16(a + i);
        float16x8_t b_vec = vld1q_f16(b + i);
        float16x8_t c_vec = vld1q_f16(c + i);
        float16x8_t ab_vec = vmulq_f16(a_vec, b_vec);
        float16x8_t ab_scaled_vec = vmulq_n_f16(ab_vec, alpha_f16);
        float16x8_t sum_vec = vfmaq_n_f16(ab_scaled_vec, c_vec, beta_f16);
        vst1q_f16(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i)
        ((float16_t*)result)[i] =
            alpha_f16 * ((float16_t const*)a)[i] * ((float16_t const*)b)[i] + beta_f16 * ((float16_t const*)c)[i];
}

SIMSIMD_PUBLIC void simsimd_wsum_u8_neon(                           //
    simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t* result) {
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_u8_vec = vld1_u8(a + i);
        uint8x8_t b_u8_vec = vld1_u8(b + i);
        float16x8_t a_vec = vcvtq_f16_u16(vmovl_u8(a_u8_vec));
        float16x8_t b_vec = vcvtq_f16_u16(vmovl_u8(b_u8_vec));
        float16x8_t a_scaled_vec = vmulq_n_f16(a_vec, alpha_f16);
        float16x8_t b_scaled_vec = vmulq_n_f16(b_vec, beta_f16);
        float16x8_t sum_vec = vaddq_f16(a_scaled_vec, b_scaled_vec);
        uint8x8_t sum_u8_vec = vmovn_u16(vcvtaq_u16_f16(sum_vec));
        vst1_u8(result + i, sum_u8_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        SIMSIMD_F32_TO_U8(alpha_f16 * a[i] + beta_f16 * b[i], result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_u8_neon(                                 //
    simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_u8_t const* c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t* result) {
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_u8_vec = vld1_u8(a + i);
        uint8x8_t b_u8_vec = vld1_u8(b + i);
        uint8x8_t c_u8_vec = vld1_u8(c + i);
        float16x8_t a_vec = vcvtq_f16_u16(vmovl_u8(a_u8_vec));
        float16x8_t b_vec = vcvtq_f16_u16(vmovl_u8(b_u8_vec));
        float16x8_t c_vec = vcvtq_f16_u16(vmovl_u8(c_u8_vec));
        float16x8_t ab_vec = vmulq_f16(a_vec, b_vec);
        float16x8_t ab_scaled_vec = vmulq_n_f16(ab_vec, alpha_f16);
        float16x8_t sum_vec = vfmaq_n_f16(ab_scaled_vec, c_vec, beta_f16);
        uint8x8_t sum_u8_vec = vmovn_u16(vcvtaq_u16_f16(sum_vec));
        vst1_u8(result + i, sum_u8_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        SIMSIMD_F32_TO_U8(alpha_f16 * a[i] * b[i] + beta_f16 * c[i], result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_wsum_i8_neon(                           //
    simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t* result) {
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t a_i8_vec = vld1_s8(a + i);
        int8x8_t b_i8_vec = vld1_s8(b + i);
        float16x8_t a_vec = vcvtq_f16_s16(vmovl_s8(a_i8_vec));
        float16x8_t b_vec = vcvtq_f16_s16(vmovl_s8(b_i8_vec));
        float16x8_t a_scaled_vec = vmulq_n_f16(a_vec, alpha_f16);
        float16x8_t b_scaled_vec = vmulq_n_f16(b_vec, beta_f16);
        float16x8_t sum_vec = vaddq_f16(a_scaled_vec, b_scaled_vec);
        int8x8_t sum_i8_vec = vmovn_s16(vcvtaq_s16_f16(sum_vec));
        vst1_s8(result + i, sum_i8_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        SIMSIMD_F32_TO_I8(alpha_f16 * a[i] + beta_f16 * b[i], result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_i8_neon(                                 //
    simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_i8_t const* c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t* result) {
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t a_i8_vec = vld1_s8(a + i);
        int8x8_t b_i8_vec = vld1_s8(b + i);
        int8x8_t c_i8_vec = vld1_s8(c + i);
        float16x8_t a_vec = vcvtq_f16_s16(vmovl_s8(a_i8_vec));
        float16x8_t b_vec = vcvtq_f16_s16(vmovl_s8(b_i8_vec));
        float16x8_t c_vec = vcvtq_f16_s16(vmovl_s8(c_i8_vec));
        float16x8_t ab_vec = vmulq_f16(a_vec, b_vec);
        float16x8_t ab_scaled_vec = vmulq_n_f16(ab_vec, alpha_f16);
        float16x8_t sum_vec = vfmaq_n_f16(ab_scaled_vec, c_vec, beta_f16);
        int8x8_t sum_i8_vec = vmovn_s16(vcvtaq_s16_f16(sum_vec));
        vst1_s8(result + i, sum_i8_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        SIMSIMD_F32_TO_I8(alpha_f16 * a[i] * b[i] + beta_f16 * c[i], result + i);
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_F16

#endif // SIMSIMD_TARGET_ARM

#ifdef __cplusplus
}
#endif

#endif
