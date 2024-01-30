#ifndef SIMSIMD_TYPES_H
#define SIMSIMD_TYPES_H

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

// Compiling for Arm: SIMSIMD_TARGET_ARM_NEON
#if !defined(SIMSIMD_TARGET_ARM_NEON) || !SIMSIMD_TARGET_ARM
#if defined(__ARM_NEON)
#define SIMSIMD_TARGET_ARM_NEON SIMSIMD_TARGET_ARM
#else
#define SIMSIMD_TARGET_ARM_NEON 0
#endif // defined(__ARM_NEON)
#endif // !defined(SIMSIMD_TARGET_ARM_NEON)

// Compiling for Arm: SIMSIMD_TARGET_ARM_SVE
#if !defined(SIMSIMD_TARGET_ARM_SVE) || !SIMSIMD_TARGET_ARM
#if defined(__ARM_FEATURE_SVE)
#define SIMSIMD_TARGET_ARM_SVE SIMSIMD_TARGET_ARM
#else
#define SIMSIMD_TARGET_ARM_SVE 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(SIMSIMD_TARGET_ARM_SVE)

// Compiling for x86: SIMSIMD_TARGET_X86_AVX2
#if !defined(SIMSIMD_TARGET_X86_AVX2) || !SIMSIMD_TARGET_X86
#if defined(__AVX2__)
#define SIMSIMD_TARGET_X86_AVX2 1
#else
#define SIMSIMD_TARGET_X86_AVX2 0
#endif // defined(__AVX2__)
#endif // !defined(SIMSIMD_TARGET_X86_AVX2)

// Compiling for x86: SIMSIMD_TARGET_X86_AVX512
#if !defined(SIMSIMD_TARGET_X86_AVX512) || !SIMSIMD_TARGET_X86
#if defined(__AVX512F__) && defined(__AVX512FP16__) && defined(__AVX512VNNI__) && defined(__AVX512VPOPCNTDQ__)
#define SIMSIMD_TARGET_X86_AVX512 1
#else
#define SIMSIMD_TARGET_X86_AVX512 0
#endif // defined(__AVX512F__) && defined(__AVX512FP16__) && defined(__AVX512VNNI__) && defined(__AVX512VPOPCNTDQ__)
#endif // !defined(SIMSIMD_TARGET_X86_AVX512)

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

#ifndef SIMSIMD_LOG
#include <math.h>
#define SIMSIMD_LOG(x) (logf(x))
#endif

#ifndef SIMSIMD_F32_DIVISION_EPSILON
#define SIMSIMD_F32_DIVISION_EPSILON 1e-7
#endif

#ifndef SIMSIMD_F16_DIVISION_EPSILON
#define SIMSIMD_F16_DIVISION_EPSILON 1e-3
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

#if !defined(SIMSIMD_NATIVE_F16) || SIMSIMD_NATIVE_F16
/**
 *  @brief  Half-precision floating-point type.
 *
 *  - GCC or Clang on 64-bit ARM: `__fp16`, may require `-mfp16-format` option.
 *  - GCC or Clang on 64-bit x86: `_Float16`.
 *  - Default: `unsigned short`.
 */
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

#define SIMSIMD_IDENTIFY(x) (x)

/**
 *  @brief  Returns the value of the half-precision floating-point number,
 *          potentially decompressed into single-precision.
 */
#ifndef SIMSIMD_UNCOMPRESS_F16
#if SIMSIMD_NATIVE_F16
#define SIMSIMD_UNCOMPRESS_F16(x) SIMSIMD_IDENTIFY(x)
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
 *  @brief  Computes `log(x)` using the Mercator series.
 *          The series converges to the natural logarithm for args between -1 and 1.
 *          Published in 1668 in "Logarithmotechnia".
 */
inline static simsimd_f32_t simsimd_approximate_log(simsimd_f32_t number) {
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
inline static float simsimd_uncompress_f16(unsigned short x) {
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

#ifdef __cplusplus
} // extern "C"
#endif

#endif
