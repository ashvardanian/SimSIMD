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

/**
 *  @brief  Half-precision floating-point type.
 *
 *  - GCC or Clang on 64-bit ARM: `__fp16`, may require `-mfp16-format` option.
 *  - GCC or Clang on 64-bit x86: `_Float16`.
 *  - Default: `unsigned short`.
 */
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__ARM_ARCH) || defined(__aarch64__)) &&                      \
    (defined(__ARM_FP16_FORMAT_IEEE))
#define SIMSIMD_NATIVE_F16 1
typedef __fp16 simsimd_f16_t;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
typedef _Float16 simsimd_f16_t;
#define SIMSIMD_NATIVE_F16 1
#else
typedef unsigned short simsimd_f16_t;
#define SIMSIMD_NATIVE_F16 0
#if defined(_MSC_VER)
#pragma message("Warning: Half-precision floating-point numbers not supported, and will be emulated.")
#else
#warning "Half-precision floating-point numbers not supported, and will be emulated."
#endif
#endif

/**
 *  @brief  Returns the value of the half-precision floating-point number,
 *          potentially decompressed into single-precision.
 */
#ifndef SIMSIMD_UNCOMPRESS_F16
#if SIMSIMD_NATIVE_F16
#define SIMSIMD_UNCOMPRESS_F16(x) (x)
#else
#define SIMSIMD_UNCOMPRESS_F16(x) simsimd_uncompress_f16(x)
#endif
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

/**
 *  @brief  For compilers that don't natively support the `_Float16` type,
 *          upcasts contents into a more conventional `float`.
 *
 *  https://stackoverflow.com/a/60047308
 *  https://gist.github.com/milhidaka/95863906fe828198f47991c813dbe233
 *  https://github.com/OpenCyphal/libcanard/blob/636795f4bc395f56af8d2c61d3757b5e762bb9e5/canard.c#L811-L834
 */
inline static simsimd_f32_t simsimd_uncompress_f16(unsigned short x) {
    unsigned int e = (x & 0x7C00) >> 10; // Exponent
    unsigned int m = (x & 0x03FF) << 13; // Mantissa
    // Evil log2 bit hack to count leading zeros in denormalized format
    float m_as_float = (float)m;
    unsigned int v = (*(unsigned int*)&m_as_float) >> 23;
    // Normalized format: sign : normalized : denormalized
    unsigned int result_as_int = //
        (x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
        ((e == 0) & (m != 0)) * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000));
    return *(float*)&result_as_int;
}

#ifdef __cplusplus
} // extern "C"
#endif