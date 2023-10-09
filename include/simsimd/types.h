#pragma once

// Compiling for Arm: SIMSIMD_TARGET_ARM
#if defined(__aarch64__) || defined(_M_ARM64)
#define SIMSIMD_TARGET_ARM 1

// Compiling for Arm: SIMSIMD_TARGET_ARM_NEON
#if !defined(SIMSIMD_TARGET_ARM_NEON)
#if defined(__ARM_NEON)
#define SIMSIMD_TARGET_ARM_NEON 1
#else
#define SIMSIMD_TARGET_ARM_NEON 0
#endif
#endif

// Compiling for Arm: SIMSIMD_TARGET_ARM_SVE
#if !defined(SIMSIMD_TARGET_ARM_SVE)
#if defined(__ARM_FEATURE_SVE)
#define SIMSIMD_TARGET_ARM_SVE 1
#else
#define SIMSIMD_TARGET_ARM_SVE 0
#endif
#endif

#undef SIMSIMD_TARGET_X86
#undef SIMSIMD_TARGET_X86_AVX2
#undef SIMSIMD_TARGET_X86_AVX512
#define SIMSIMD_TARGET_X86 0
#define SIMSIMD_TARGET_X86_AVX2 0
#define SIMSIMD_TARGET_X86_AVX512 0

// Compiling for x86: SIMSIMD_TARGET_X86
#elif defined(__x86_64__) || defined(_M_X64)
#define SIMSIMD_TARGET_X86 1

// Compiling for x86: SIMSIMD_TARGET_X86_AVX2
#if !defined(SIMSIMD_TARGET_X86_AVX2)
#if defined(__AVX2__)
#define SIMSIMD_TARGET_X86_AVX2 1
#else
#define SIMSIMD_TARGET_X86_AVX2 0
#endif
#endif

// Compiling for x86: SIMSIMD_TARGET_X86_AVX512
#if !defined(SIMSIMD_TARGET_X86_AVX512)
#if defined(__AVX512F__) || defined(__AVX512VPOPCNTDQ__)
#define SIMSIMD_TARGET_X86_AVX512 1
#else
#define SIMSIMD_TARGET_X86_AVX512 0
#endif
#endif

#undef SIMSIMD_TARGET_ARM
#undef SIMSIMD_TARGET_ARM_NEON
#undef SIMSIMD_TARGET_ARM_SVE
#define SIMSIMD_TARGET_ARM 0
#define SIMSIMD_TARGET_ARM_NEON 0
#define SIMSIMD_TARGET_ARM_SVE 0

// Compiling for unknown hardware architecture
#else
#error Unknown hardware architecture!
#endif

#ifdef _MSC_VER
#include <intrin.h>
#else

#if SIMSIMD_TARGET_ARM_NEON
#include <arm_neon.h>
#endif

#if SIMSIMD_TARGET_ARM_SVE
#include <arm_sve.h>
#endif

#if SIMSIMD_TARGET_X86_AVX2 || SIMSIMD_TARGET_X86_AVX512
#include <immintrin.h>
#endif

#endif

#ifndef SIMSIMD_RSQRT
#include <math.h>
#define SIMSIMD_RSQRT(x) (1 / sqrtf(x))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef int simsimd_i32_t;
typedef float simsimd_f32_t;
typedef double simsimd_f64_t;
typedef signed char simsimd_i8_t;
typedef unsigned char simsimd_b8_t;
typedef unsigned long long simsimd_size_t;

#if defined(__GNUC__) || defined(__clang__)
#if defined(__ARM_ARCH) || defined(__aarch64__)
#if defined(__ARM_FP16_FORMAT_IEEE)
typedef __fp16 simsimd_f16_t;
#else
#error "Enable -mfp16-format option for ARM targets to use __fp16."
#endif
#elif defined(__x86_64__) || defined(__i386__)
typedef _Float16 simsimd_f16_t;
#else
#error "Unsupported architecture for simsimd_f16_t."
#endif
#else
typedef _Float16 simsimd_f16_t; // This will be the fallback if not using GCC or Clang
#endif

typedef union {
    unsigned i;
    float f;
} simsimd_f32i32_t;

/**
 *  @brief  Computes `1/sqrt(x)` using the trick from Quake 3, replacing
 *          magic numbers with the ones suggested by Jan Kadlec.
 */
inline static simsimd_f32_t simsimd_approximate_inverse_square_root(simsimd_f32_t number) {
    simsimd_f32i32_t conv;
    conv.f = number;
    conv.i = 0x5F1FFFF9 - (conv.i >> 1);
    conv.f *= 0.703952253f * (2.38924456f - number * conv.f * conv.f);
    return conv.f;
}

#ifdef __cplusplus
} // extern "C"
#endif