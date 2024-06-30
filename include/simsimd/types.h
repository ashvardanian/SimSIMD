/**
 *  @file       types.h
 *  @brief      Shared definitions for the SimSIMD library.
 *  @author     Ash Vardanian
 *  @date       October 2, 2023
 *
 *  Defines:
 *  - Sized aliases for numeric types, like: `simsimd_i32_t` and `simsimd_f64_t`.
 *  - Macros for compiler/hardware checks, like: `SIMSIMD_TARGET_NEON`
 */
#ifndef SIMSIMD_TYPES_H
#define SIMSIMD_TYPES_H

/*  Annotation for the public API symbols:
 *
 *  - `SIMSIMD_PUBLIC` is used for functions that are part of the public API.
 *  - `SIMSIMD_INTERNAL` is used for internal helper functions with unstable APIs.
 *  - `SIMSIMD_DYNAMIC` is used for functions that are part of the public API, but are dispatched at runtime.
 */
#if defined(_WIN32) || defined(__CYGWIN__)
#define SIMSIMD_DYNAMIC __declspec(dllexport)
#define SIMSIMD_PUBLIC inline static
#define SIMSIMD_INTERNAL inline static
#elif defined(__GNUC__) || defined(__clang__)
#define SIMSIMD_DYNAMIC __attribute__((visibility("default")))
#define SIMSIMD_PUBLIC __attribute__((unused)) inline static
#define SIMSIMD_INTERNAL __attribute__((always_inline)) inline static
#else
#define SIMSIMD_DYNAMIC inline static
#define SIMSIMD_PUBLIC inline static
#define SIMSIMD_INTERNAL inline static
#endif

// Compiling for Arm: SIMSIMD_TARGET_ARM
#if !defined(SIMSIMD_TARGET_ARM)
#if defined(__aarch64__) || defined(_M_ARM64)
#define SIMSIMD_TARGET_ARM 1
#else
#define SIMSIMD_TARGET_ARM 0
#endif // defined(__aarch64__) || defined(_M_ARM64)
#endif // !defined(SIMSIMD_TARGET_ARM)

// Compiling for x86: SIMSIMD_TARGET_X86
#if !defined(SIMSIMD_TARGET_X86)
#if defined(__x86_64__) || defined(_M_X64)
#define SIMSIMD_TARGET_X86 1
#else
#define SIMSIMD_TARGET_X86 0
#endif // defined(__x86_64__) || defined(_M_X64)
#endif // !defined(SIMSIMD_TARGET_X86)

// Compiling for Arm: SIMSIMD_TARGET_NEON
#if !defined(SIMSIMD_TARGET_NEON) || (SIMSIMD_TARGET_NEON && !SIMSIMD_TARGET_ARM)
#if defined(__ARM_NEON)
#define SIMSIMD_TARGET_NEON SIMSIMD_TARGET_ARM
#else
#undef SIMSIMD_TARGET_NEON
#define SIMSIMD_TARGET_NEON 0
#endif // defined(__ARM_NEON)
#endif // !defined(SIMSIMD_TARGET_NEON)

// Compiling for Arm: SIMSIMD_TARGET_SVE
#if !defined(SIMSIMD_TARGET_SVE) || (SIMSIMD_TARGET_SVE && !SIMSIMD_TARGET_ARM)
#if defined(__ARM_FEATURE_SVE)
#define SIMSIMD_TARGET_SVE SIMSIMD_TARGET_ARM
#else
#undef SIMSIMD_TARGET_SVE
#define SIMSIMD_TARGET_SVE 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(SIMSIMD_TARGET_SVE)

// Compiling for x86: SIMSIMD_TARGET_HASWELL
//
// Starting with Ivy Bridge, Intel supports the `F16C` extensions for fast half-precision
// to single-precision floating-point conversions. On AMD those instructions
// are supported on all CPUs starting with Jaguar 2009.
// Starting with Sandy Bridge, Intel adds basic AVX support in their CPUs and in 2013
// extends it with AVX2 in the Haswell generation. Moreover, Haswell adds FMA support.
#if !defined(SIMSIMD_TARGET_HASWELL) || (SIMSIMD_TARGET_HASWELL && !SIMSIMD_TARGET_X86)
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define SIMSIMD_TARGET_HASWELL 1
#else
#undef SIMSIMD_TARGET_HASWELL
#define SIMSIMD_TARGET_HASWELL 0
#endif // defined(__AVX2__)
#endif // !defined(SIMSIMD_TARGET_HASWELL)

// Compiling for x86: SIMSIMD_TARGET_SKYLAKE, SIMSIMD_TARGET_ICE, SIMSIMD_TARGET_SAPPHIRE
//
// To list all available macros for x86, take a recent compiler, like GCC 12 and run:
//      gcc-12 -march=sapphirerapids -dM -E - < /dev/null | egrep "SSE|AVX" | sort
// On Arm machines you may want to check for other flags:
//      gcc-12 -march=native -dM -E - < /dev/null | egrep "NEON|SVE|FP16|FMA" | sort
#if !defined(SIMSIMD_TARGET_SKYLAKE) || (SIMSIMD_TARGET_SKYLAKE && !SIMSIMD_TARGET_X86)
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512VL__) && defined(__AVX512DQ__) &&                 \
    defined(__AVX512BW__)
#define SIMSIMD_TARGET_SKYLAKE 1
#else
#undef SIMSIMD_TARGET_SKYLAKE
#define SIMSIMD_TARGET_SKYLAKE 0
#endif
#endif // !defined(SIMSIMD_TARGET_SKYLAKE)
#if !defined(SIMSIMD_TARGET_ICE) || (SIMSIMD_TARGET_ICE && !SIMSIMD_TARGET_X86)
#if defined(__AVX512VNNI__) && defined(__AVX512IFMA__) && defined(__AVX512BITALG__) && defined(__AVX512VBMI2__) &&     \
    defined(__AVX512VPOPCNTDQ__)
#define SIMSIMD_TARGET_ICE 1
#else
#undef SIMSIMD_TARGET_ICE
#define SIMSIMD_TARGET_ICE 0
#endif
#endif // !defined(SIMSIMD_TARGET_ICE)
#if !defined(SIMSIMD_TARGET_GENOA) || (SIMSIMD_TARGET_GENOA && !SIMSIMD_TARGET_X86)
#if defined(__AVX512BF16__)
#define SIMSIMD_TARGET_GENOA 1
#else
#undef SIMSIMD_TARGET_GENOA
#define SIMSIMD_TARGET_GENOA 0
#endif
#endif // !defined(SIMSIMD_TARGET_GENOA)
#if !defined(SIMSIMD_TARGET_SAPPHIRE) || (SIMSIMD_TARGET_SAPPHIRE && !SIMSIMD_TARGET_X86)
#if defined(__AVX512FP16__)
#define SIMSIMD_TARGET_SAPPHIRE 1
#else
#undef SIMSIMD_TARGET_SAPPHIRE
#define SIMSIMD_TARGET_SAPPHIRE 0
#endif
#endif // !defined(SIMSIMD_TARGET_SAPPHIRE)

#ifdef _MSC_VER
#include <intrin.h>
#else

#if SIMSIMD_TARGET_NEON
#include <arm_neon.h>
#endif

#if SIMSIMD_TARGET_SVE
#include <arm_sve.h>
#endif

#if SIMSIMD_TARGET_HASWELL || SIMSIMD_TARGET_SKYLAKE
#include <immintrin.h>
#endif

#endif

#ifndef SIMSIMD_RSQRT
#include <math.h>
#define SIMSIMD_RSQRT(x) (1 / sqrtf(x))
#endif

#ifndef SIMSIMD_LOG
#include <math.h>
#define SIMSIMD_LOG(x) (logf(x))
#endif

#ifndef SIMSIMD_F32_DIVISION_EPSILON
#define SIMSIMD_F32_DIVISION_EPSILON (1e-7)
#endif

#ifndef SIMSIMD_F16_DIVISION_EPSILON
#define SIMSIMD_F16_DIVISION_EPSILON (1e-3)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef int simsimd_i32_t;
typedef float simsimd_f32_t;
typedef double simsimd_f64_t;
typedef signed char simsimd_i8_t;
typedef unsigned char simsimd_b8_t;
typedef long long simsimd_i64_t;
typedef unsigned long long simsimd_u64_t;

typedef simsimd_u64_t simsimd_size_t;
typedef simsimd_f64_t simsimd_distance_t;

/*  @brief  Half-precision floating-point type.
 *
 *  - GCC or Clang on 64-bit Arm: `__fp16`, may require `-mfp16-format` option.
 *  - GCC or Clang on 64-bit x86: `_Float16`.
 *  - Default: `unsigned short`.
 */
#if !defined(SIMSIMD_NATIVE_F16) || SIMSIMD_NATIVE_F16
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__ARM_ARCH) || defined(__aarch64__)) &&                      \
    (defined(__ARM_FP16_FORMAT_IEEE))
#if !defined(SIMSIMD_NATIVE_F16)
#define SIMSIMD_NATIVE_F16 1
#endif
typedef __fp16 simsimd_f16_t;
#elif ((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) &&                      \
       (defined(__SSE2__) || defined(__AVX512F__)))
typedef _Float16 simsimd_f16_t;
#if !defined(SIMSIMD_NATIVE_F16)
#define SIMSIMD_NATIVE_F16 1
#endif
#else // Unknown compiler or architecture
#define SIMSIMD_NATIVE_F16 0
#endif // Unknown compiler or architecture
#endif // !SIMSIMD_NATIVE_F16

#if !SIMSIMD_NATIVE_F16
typedef unsigned short simsimd_f16_t;
#endif

#if !defined(SIMSIMD_NATIVE_BF16) || SIMSIMD_NATIVE_BF16
/**
 *  @brief  Half-precision brain-float type.
 *
 *  - GCC or Clang on 64-bit Arm: `__bf16`
 *  - GCC or Clang on 64-bit x86: `_BFloat16`.
 *  - Default: `unsigned short`.
 *
 *  @warning Apple Clang has hard time with bf16.
 *  https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
 *  https://forums.developer.apple.com/forums/thread/726201
 */
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__ARM_ARCH) || defined(__aarch64__)) &&                      \
    (defined(__ARM_BF16_FORMAT_ALTERNATIVE))
#if !defined(SIMSIMD_NATIVE_BF16)
#define SIMSIMD_NATIVE_BF16 1
#endif
typedef __fp16 simsimd_bf16_t;
#elif ((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) &&                      \
       (defined(__SSE2__) || defined(__AVX512F__)))
typedef _Float16 simsimd_bf16_t;
#if !defined(SIMSIMD_NATIVE_BF16)
#define SIMSIMD_NATIVE_BF16 1
#endif
#else // Unknown compiler or architecture
#define SIMSIMD_NATIVE_BF16 0
#endif // Unknown compiler or architecture
#endif // !SIMSIMD_NATIVE_BF16

#if !SIMSIMD_NATIVE_BF16
typedef unsigned short simsimd_bf16_t;
#endif

/**
 *  @brief  Alias for the half-precision floating-point type on Arm.
 *
 *  Clang and GCC bring the `float16_t` symbol when you compile for Aarch64.
 *  MSVC lacks it, and it's `vld1_f16`-like intrinsics are in reality macros,
 *  that cast to 16-bit integers internally, instead of using floats.
 *  Some of those are defined as aliases, so we use `#define` preprocessor
 *  directives instead of `typedef` to avoid errors.
 */
#if SIMSIMD_TARGET_ARM
#if defined(_MSC_VER)
#define simsimd_f16_for_arm_simd_t simsimd_f16_t
#define simsimd_bf16_for_arm_simd_t simsimd_bf16_t
#else
#define simsimd_f16_for_arm_simd_t float16_t
#define simsimd_bf16_for_arm_simd_t bfloat16_t
#endif
#endif

#define SIMSIMD_IDENTIFY(x) (x)

/**
 *  @brief  Returns the value of the half-precision floating-point number,
 *          potentially decompressed into single-precision.
 */
#ifndef SIMSIMD_UNCOMPRESS_F16
#if SIMSIMD_NATIVE_F16
#define SIMSIMD_UNCOMPRESS_F16(x) (SIMSIMD_IDENTIFY(x))
#else
#define SIMSIMD_UNCOMPRESS_F16(x) (simsimd_uncompress_f16(x))
#endif
#endif

/**
 *  @brief  Returns the value of the half-precision brain floating-point number,
 *          potentially decompressed into single-precision.
 */
#ifndef SIMSIMD_UNCOMPRESS_BF16
#if SIMSIMD_NATIVE_BF16
#define SIMSIMD_UNCOMPRESS_BF16(x) (SIMSIMD_IDENTIFY(x))
#else
#define SIMSIMD_UNCOMPRESS_BF16(x) (simsimd_uncompress_bf16(x))
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
SIMSIMD_PUBLIC simsimd_f32_t simsimd_approximate_inverse_square_root(simsimd_f32_t number) {
    simsimd_f32i32_t conv;
    conv.f = number;
    conv.i = 0x5F1FFFF9 - (conv.i >> 1);
    conv.f *= 0.703952253f * (2.38924456f - number * conv.f * conv.f);
    return conv.f;
}

/**
 *  @brief  Computes `log(x)` using the Mercator series.
 *          The series converges to the natural logarithm for args between -1 and 1.
 *          Published in 1668 in "Logarithmotechnia".
 */
SIMSIMD_PUBLIC simsimd_f32_t simsimd_approximate_log(simsimd_f32_t number) {
    simsimd_f32_t x = number - 1;
    simsimd_f32_t x2 = x * x;
    simsimd_f32_t x3 = x * x * x;
    return x - x2 / 2 + x3 / 3;
}

/**
 *  @brief  For compilers that don't natively support the `_Float16` type,
 *          upcasts contents into a more conventional `float`.
 *
 *  @warning  This function won't handle boundary conditions well.
 *
 *  https://stackoverflow.com/a/60047308
 *  https://gist.github.com/milhidaka/95863906fe828198f47991c813dbe233
 *  https://github.com/OpenCyphal/libcanard/blob/636795f4bc395f56af8d2c61d3757b5e762bb9e5/canard.c#L811-L834
 */
SIMSIMD_PUBLIC simsimd_f32_t simsimd_uncompress_f16(unsigned short x) {
    union float_or_unsigned_int_t {
        float f;
        unsigned int i;
    };
    unsigned int exponent = (x & 0x7C00) >> 10;
    unsigned int mantissa = (x & 0x03FF) << 13;
    union float_or_unsigned_int_t mantissa_union;
    mantissa_union.f = (float)mantissa;
    unsigned int v = (mantissa_union.i) >> 23;
    union float_or_unsigned_int_t result_union;
    result_union.i = (x & 0x8000) << 16 | (exponent != 0) * ((exponent + 112) << 23 | mantissa) |
                     ((exponent == 0) & (mantissa != 0)) * ((v - 37) << 23 | ((mantissa << (150 - v)) & 0x007FE000));
    return result_union.f;
}

/**
 *  @brief  Compresses a `float` to an `f16` representation (IEEE-754 16-bit floating-point format).
 *
 *  @warning  This function won't handle boundary conditions well.
 *
 *  https://stackoverflow.com/a/60047308
 *  https://gist.github.com/milhidaka/95863906fe828198f47991c813dbe233
 *  https://github.com/OpenCyphal/libcanard/blob/636795f4bc395f56af8d2c61d3757b5e762bb9e5/canard.c#L811-L834
 */
SIMSIMD_PUBLIC unsigned short simsimd_compress_f16(simsimd_f32_t x) {
    union float_or_unsigned_int_t {
        float f;
        unsigned int i;
    };

    unsigned int b = *(unsigned int*)&x + 0x00001000;
    unsigned int e = (b & 0x7F800000) >> 23;
    unsigned int m = b & 0x007FFFFF;
    unsigned short result = (b & 0x80000000) >> 16 | (e > 112) * (((e - 112) << 10) & 0x7C00 | m >> 13) |
                            ((e < 113) & (e > 101)) * (((0x007FF000 + m) >> (125 - e) + 1) >> 1) | (e > 143) * 0x7FFF;
    return result;
}

/**
 *  @brief  For compilers that don't natively support the `__bf16` type,
 *          upcasts contents into a more conventional `float`.
 *
 *  https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c/55254307#55254307
 *  https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
 */
SIMSIMD_PUBLIC simsimd_f32_t simsimd_uncompress_bf16(unsigned short x) {
    union float_or_unsigned_int_t {
        float f;
        unsigned int i;
    };
    union float_or_unsigned_int_t result_union;
    result_union.i = x << 16; // Zero extends the mantissa
    return result_union.f;
}

/**
 *  @brief  Compresses a `float` to a `bf16` representation.
 */
SIMSIMD_PUBLIC unsigned short simsimd_compress_bf16(simsimd_f32_t x) {
    union float_or_unsigned_int_t {
        float f;
        unsigned int i;
    };
    union float_or_unsigned_int_t value;
    value.f = x;
    value.i >>= 16;
    value.i &= 0xFFFF;
    return (unsigned short)value.i;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif
