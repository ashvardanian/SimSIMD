/**
 *  @file       types.h
 *  @brief      Shared definitions for the MathKong library.
 *  @author     Ash Vardanian
 *  @date       October 2, 2023
 *
 *  Defines:
 *  - Sized aliases for numeric types, like: `mathkong_i32_t` and `mathkong_f64_t`.
 *  - Macros for internal compiler/hardware checks, like: `_MATHKONG_TARGET_ARM`.
 *  - Macros for feature controls, like: `MATHKONG_TARGET_NEON`
 */
#ifndef MATHKONG_TYPES_H
#define MATHKONG_TYPES_H

// Inferring target OS: Windows, macOS, or Linux
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define _MATHKONG_DEFINED_WINDOWS 1
#elif defined(__APPLE__) && defined(__MACH__)
#define _MATHKONG_DEFINED_APPLE 1
#elif defined(__linux__)
#define _MATHKONG_DEFINED_LINUX 1
#endif

// Annotation for the public API symbols:
//
// - `MATHKONG_PUBLIC` is used for functions that are part of the public API.
// - `MATHKONG_INTERNAL` is used for internal helper functions with unstable APIs.
// - `MATHKONG_DYNAMIC` is used for functions that are part of the public API, but are dispatched at runtime.
//
// On GCC we mark the functions as `nonnull` informing that none of the arguments can be `NULL`.
// Marking with `pure` and `const` isn't possible as outputting to a pointer is a "side effect".
#if defined(_WIN32) || defined(__CYGWIN__)
#define MATHKONG_DYNAMIC __declspec(dllexport)
#define MATHKONG_PUBLIC inline static
#define MATHKONG_INTERNAL inline static
#elif defined(__GNUC__) || defined(__clang__)
#define MATHKONG_DYNAMIC __attribute__((visibility("default"))) __attribute__((nonnull))
#define MATHKONG_PUBLIC __attribute__((unused, nonnull)) inline static
#define MATHKONG_INTERNAL __attribute__((always_inline)) inline static
#else
#define MATHKONG_DYNAMIC
#define MATHKONG_PUBLIC inline static
#define MATHKONG_INTERNAL inline static
#endif

// Compiling for Arm: _MATHKONG_TARGET_ARM
#if !defined(_MATHKONG_TARGET_ARM)
#if defined(__aarch64__) || defined(_M_ARM64)
#define _MATHKONG_TARGET_ARM 1
#else
#define _MATHKONG_TARGET_ARM 0
#endif // defined(__aarch64__) || defined(_M_ARM64)
#endif // !defined(_MATHKONG_TARGET_ARM)

// Compiling for x86: _MATHKONG_TARGET_X86
#if !defined(_MATHKONG_TARGET_X86)
#if defined(__x86_64__) || defined(_M_X64)
#define _MATHKONG_TARGET_X86 1
#else
#define _MATHKONG_TARGET_X86 0
#endif // defined(__x86_64__) || defined(_M_X64)
#endif // !defined(_MATHKONG_TARGET_X86)

// Compiling for Arm: MATHKONG_TARGET_NEON
#if !defined(MATHKONG_TARGET_NEON) || (MATHKONG_TARGET_NEON && !_MATHKONG_TARGET_ARM)
#if defined(__ARM_NEON)
#define MATHKONG_TARGET_NEON _MATHKONG_TARGET_ARM
#else
#undef MATHKONG_TARGET_NEON
#define MATHKONG_TARGET_NEON 0
#endif // defined(__ARM_NEON)
#endif // !defined(MATHKONG_TARGET_NEON) || ...

// Compiling for Arm: MATHKONG_TARGET_NEON_I8
#if !defined(MATHKONG_TARGET_NEON_I8) || (MATHKONG_TARGET_NEON_I8 && !_MATHKONG_TARGET_ARM)
#if defined(__ARM_NEON)
#define MATHKONG_TARGET_NEON_I8 _MATHKONG_TARGET_ARM
#else
#undef MATHKONG_TARGET_NEON_I8
#define MATHKONG_TARGET_NEON_I8 0
#endif // defined(__ARM_NEON)
#endif // !defined(MATHKONG_TARGET_NEON_I8) || ...

// Compiling for Arm: MATHKONG_TARGET_NEON_F16
#if !defined(MATHKONG_TARGET_NEON_F16) || (MATHKONG_TARGET_NEON_F16 && !_MATHKONG_TARGET_ARM)
#if defined(__ARM_NEON)
#define MATHKONG_TARGET_NEON_F16 _MATHKONG_TARGET_ARM
#else
#undef MATHKONG_TARGET_NEON_F16
#define MATHKONG_TARGET_NEON_F16 0
#endif // defined(__ARM_NEON)
#endif // !defined(MATHKONG_TARGET_NEON_F16) || ...

// Compiling for Arm: MATHKONG_TARGET_NEON_BF16
#if !defined(MATHKONG_TARGET_NEON_BF16) || (MATHKONG_TARGET_NEON_BF16 && !_MATHKONG_TARGET_ARM)
#if defined(__ARM_NEON)
#define MATHKONG_TARGET_NEON_BF16 _MATHKONG_TARGET_ARM
#else
#undef MATHKONG_TARGET_NEON_BF16
#define MATHKONG_TARGET_NEON_BF16 0
#endif // defined(__ARM_NEON)
#endif // !defined(MATHKONG_TARGET_NEON_BF16) || ...

// Compiling for Arm: MATHKONG_TARGET_SVE
#if !defined(MATHKONG_TARGET_SVE) || (MATHKONG_TARGET_SVE && !_MATHKONG_TARGET_ARM)
#if defined(__ARM_FEATURE_SVE)
#define MATHKONG_TARGET_SVE _MATHKONG_TARGET_ARM
#else
#undef MATHKONG_TARGET_SVE
#define MATHKONG_TARGET_SVE 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(MATHKONG_TARGET_SVE) || ...

// Compiling for Arm: MATHKONG_TARGET_SVE_I8
#if !defined(MATHKONG_TARGET_SVE_I8) || (MATHKONG_TARGET_SVE_I8 && !_MATHKONG_TARGET_ARM)
#if defined(__ARM_FEATURE_SVE)
#define MATHKONG_TARGET_SVE_I8 _MATHKONG_TARGET_ARM
#else
#undef MATHKONG_TARGET_SVE_I8
#define MATHKONG_TARGET_SVE_I8 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(MATHKONG_TARGET_SVE_I8) || ...

// Compiling for Arm: MATHKONG_TARGET_SVE_F16
#if !defined(MATHKONG_TARGET_SVE_F16) || (MATHKONG_TARGET_SVE_F16 && !_MATHKONG_TARGET_ARM)
#if defined(__ARM_FEATURE_SVE)
#define MATHKONG_TARGET_SVE_F16 _MATHKONG_TARGET_ARM
#else
#undef MATHKONG_TARGET_SVE_F16
#define MATHKONG_TARGET_SVE_F16 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(MATHKONG_TARGET_SVE_F16) || ...

// Compiling for Arm: MATHKONG_TARGET_SVE_BF16
#if !defined(MATHKONG_TARGET_SVE_BF16) || (MATHKONG_TARGET_SVE_BF16 && !_MATHKONG_TARGET_ARM)
#if defined(__ARM_FEATURE_SVE)
#define MATHKONG_TARGET_SVE_BF16 _MATHKONG_TARGET_ARM
#else
#undef MATHKONG_TARGET_SVE_BF16
#define MATHKONG_TARGET_SVE_BF16 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(MATHKONG_TARGET_SVE_BF16) || ...

// Compiling for Arm: MATHKONG_TARGET_SVE2
#if !defined(MATHKONG_TARGET_SVE2) || (MATHKONG_TARGET_SVE2 && !_MATHKONG_TARGET_ARM)
#if defined(__ARM_FEATURE_SVE)
#define MATHKONG_TARGET_SVE2 _MATHKONG_TARGET_ARM
#else
#undef MATHKONG_TARGET_SVE2
#define MATHKONG_TARGET_SVE2 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(MATHKONG_TARGET_SVE2) || ...

// Compiling for x86: MATHKONG_TARGET_HASWELL
//
// Starting with Ivy Bridge, Intel supports the `F16C` extensions for fast half-precision
// to single-precision floating-point conversions. On AMD those instructions
// are supported on all CPUs starting with Jaguar 2009.
// Starting with Sandy Bridge, Intel adds basic AVX support in their CPUs and in 2013
// extends it with AVX2 in the Haswell generation. Moreover, Haswell adds FMA support.
#if !defined(MATHKONG_TARGET_HASWELL) || (MATHKONG_TARGET_HASWELL && !_MATHKONG_TARGET_X86)
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define MATHKONG_TARGET_HASWELL 1
#else
#undef MATHKONG_TARGET_HASWELL
#define MATHKONG_TARGET_HASWELL 0
#endif // defined(__AVX2__)
#endif // !defined(MATHKONG_TARGET_HASWELL) || ...

// Compiling for x86: MATHKONG_TARGET_SKYLAKE, MATHKONG_TARGET_ICE, MATHKONG_TARGET_GENOA,
// MATHKONG_TARGET_SAPPHIRE, MATHKONG_TARGET_TURIN, MATHKONG_TARGET_SIERRA
//
// To list all available macros for x86, take a recent compiler, like GCC 12 and run:
//      gcc-12 -march=sapphirerapids -dM -E - < /dev/null | egrep "SSE|AVX" | sort
// On Arm machines you may want to check for other flags:
//      gcc-12 -march=native -dM -E - < /dev/null | egrep "NEON|SVE|FP16|FMA" | sort
#if !defined(MATHKONG_TARGET_SKYLAKE) || (MATHKONG_TARGET_SKYLAKE && !_MATHKONG_TARGET_X86)
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && \
    defined(__AVX512BW__)
#define MATHKONG_TARGET_SKYLAKE 1
#else
#undef MATHKONG_TARGET_SKYLAKE
#define MATHKONG_TARGET_SKYLAKE 0
#endif
#endif // !defined(MATHKONG_TARGET_SKYLAKE) || ...
#if !defined(MATHKONG_TARGET_ICE) || (MATHKONG_TARGET_ICE && !_MATHKONG_TARGET_X86)
#if defined(__AVX512VNNI__) && defined(__AVX512IFMA__) && defined(__AVX512BITALG__) && defined(__AVX512VBMI2__) && \
    defined(__AVX512VPOPCNTDQ__)
#define MATHKONG_TARGET_ICE 1
#else
#undef MATHKONG_TARGET_ICE
#define MATHKONG_TARGET_ICE 0
#endif
#endif // !defined(MATHKONG_TARGET_ICE) || ...
#if !defined(MATHKONG_TARGET_GENOA) || (MATHKONG_TARGET_GENOA && !_MATHKONG_TARGET_X86)
#if defined(__AVX512BF16__)
#define MATHKONG_TARGET_GENOA 1
#else
#undef MATHKONG_TARGET_GENOA
#define MATHKONG_TARGET_GENOA 0
#endif
#endif // !defined(MATHKONG_TARGET_GENOA) || ...
#if !defined(MATHKONG_TARGET_SAPPHIRE) || (MATHKONG_TARGET_SAPPHIRE && !_MATHKONG_TARGET_X86)
#if defined(__AVX512FP16__)
#define MATHKONG_TARGET_SAPPHIRE 1
#else
#undef MATHKONG_TARGET_SAPPHIRE
#define MATHKONG_TARGET_SAPPHIRE 0
#endif
#endif // !defined(MATHKONG_TARGET_SAPPHIRE) || ...
#if !defined(MATHKONG_TARGET_TURIN) || (MATHKONG_TARGET_TURIN && !_MATHKONG_TARGET_X86)
#if defined(__AVX512VP2INTERSECT__)
#define MATHKONG_TARGET_TURIN 1
#else
#undef MATHKONG_TARGET_TURIN
#define MATHKONG_TARGET_TURIN 0
#endif
#endif // !defined(MATHKONG_TARGET_TURIN) || ...
#if !defined(MATHKONG_TARGET_SIERRA) || (MATHKONG_TARGET_SIERRA && !_MATHKONG_TARGET_X86)
#if defined(__AVX2_VNNI__)
#define MATHKONG_TARGET_SIERRA 1
#else
#undef MATHKONG_TARGET_SIERRA
#define MATHKONG_TARGET_SIERRA 0
#endif
#endif // !defined(MATHKONG_TARGET_SIERRA) || ...

#if defined(_MSC_VER)
#include <intrin.h>
#else

#if MATHKONG_TARGET_NEON
#include <arm_neon.h>
#endif

#if MATHKONG_TARGET_SVE || MATHKONG_TARGET_SVE2
#include <arm_sve.h>
#endif

#if MATHKONG_TARGET_HASWELL || MATHKONG_TARGET_SKYLAKE || MATHKONG_TARGET_ICE || MATHKONG_TARGET_GENOA || \
    MATHKONG_TARGET_SAPPHIRE || MATHKONG_TARGET_TURIN
#include <immintrin.h>
#endif

#endif

#if !defined(MATHKONG_SQRT)
#include <math.h>
#define MATHKONG_SQRT(x) (sqrt(x))
#endif

#if !defined(MATHKONG_RSQRT)
#include <math.h>
#define MATHKONG_RSQRT(x) (1 / MATHKONG_SQRT(x))
#endif

#if !defined(MATHKONG_LOG)
#include <math.h>
#define MATHKONG_LOG(x) (log(x))
#endif

// Copy 16 bits (2 bytes) from source to destination
#if defined(__GNUC__) || defined(__clang__)
#define MATHKONG_COPY16(destination_ptr, source_ptr) __builtin_memcpy((destination_ptr), (source_ptr), 2)
#else
#include <string.h> /* fallback for exotic compilers */
#define MATHKONG_COPY16(destination_ptr, source_ptr) memcpy((destination_ptr), (source_ptr), 2)
#endif

#if !defined(MATHKONG_F32_DIVISION_EPSILON)
#define MATHKONG_F32_DIVISION_EPSILON (1e-7)
#endif

#if !defined(MATHKONG_F16_DIVISION_EPSILON)
#define MATHKONG_F16_DIVISION_EPSILON (1e-3)
#endif

/**
 *  @brief  The compile-time constant defining the capacity of `mathkong_xd_index_t`.
 *          Matches `PyBUF_MAX_NDIM` by default.
 */
#if !defined(MATHKONG_NDARRAY_MAX_RANK)
#define MATHKONG_NDARRAY_MAX_RANK (64)
#endif

/**
 *  @brief  Aligns a variable to a 64-byte boundary using compiler extensions for
 *          compatibility with C 99, as `alignas(64)` is only available in C 11 or C++.
 *
 */
#ifdef _MSC_VER
#define MATHKONG_ALIGN64 __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
#define MATHKONG_ALIGN64 __attribute__((aligned(64)))
#endif

/**
 *  @brief  Similar to `static_assert`, but compatible with C 99.
 *          In C the `_Static_assert` is only available with C 11 and later.
 */
#define _MATHKONG_STATIC_ASSERT(expr, msg) typedef char static_assert_##msg[(expr) ? 1 : -1]

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char mathkong_b8_t;
typedef unsigned char mathkong_i4x2_t;

typedef signed char mathkong_i8_t;
typedef unsigned char mathkong_u8_t;
typedef signed short mathkong_i16_t;
typedef unsigned short mathkong_u16_t;
typedef signed int mathkong_i32_t;
typedef unsigned int mathkong_u32_t;
typedef signed long long mathkong_i64_t;
typedef unsigned long long mathkong_u64_t;

typedef float mathkong_f32_t;
typedef double mathkong_f64_t;

typedef mathkong_u64_t mathkong_size_t;
typedef mathkong_i64_t mathkong_ssize_t;
typedef mathkong_f64_t mathkong_distance_t;

/*  @brief  Half-precision floating-point type.
 *
 *  - GCC or Clang on 64-bit Arm: `__fp16`, may require `-mfp16-format` option.
 *  - GCC or Clang on 64-bit x86: `_Float16`.
 *  - Default: `unsigned short`.
 */
#if !defined(MATHKONG_NATIVE_F16) || MATHKONG_NATIVE_F16
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__ARM_ARCH) || defined(__aarch64__)) && \
    (defined(__ARM_FP16_FORMAT_IEEE))
#undef MATHKONG_NATIVE_F16
#define MATHKONG_NATIVE_F16 1
typedef __fp16 mathkong_f16_t;
#elif ((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && \
       (defined(__AVX512FP16__)))
typedef _Float16 mathkong_f16_t;
#undef MATHKONG_NATIVE_F16
#define MATHKONG_NATIVE_F16 1
#else                                       // Unknown compiler or architecture
#if defined(__GNUC__) || defined(__clang__) // Some compilers don't support warning pragmas
#warning "Unknown compiler or architecture for float16."
#endif
#undef MATHKONG_NATIVE_F16
#define MATHKONG_NATIVE_F16 0
#endif // Unknown compiler or architecture
#endif // !MATHKONG_NATIVE_F16

#if !MATHKONG_NATIVE_F16
typedef unsigned short mathkong_f16_t;
#endif

#if !defined(MATHKONG_NATIVE_BF16) || MATHKONG_NATIVE_BF16
/**
 *  @brief  Half-precision brain-float type.
 *
 *  - GCC or Clang on 64-bit Arm: `__bf16`
 *  - GCC or Clang on 64-bit x86: `_BFloat16`.
 *  - Default: `unsigned short`.
 *
 *  The compilers have added `__bf16` support in compliance with the x86-64 psABI spec.
 *  The motivation for this new special type is summed up as:
 *
 *      Currently `__bfloat16` is a typedef of short, which creates a problem where the
 *      compiler does not raise any alarms if it is used to add, subtract, multiply or
 *      divide, but the result of the calculation is actually meaningless.
 *      To solve this problem, a real scalar type `__Bfloat16` needs to be introduced.
 *      It is mainly used for intrinsics, not available for C standard operators.
 *      `__Bfloat16` will also be used for movement like passing parameter, load and store,
 *      vector initialization, vector shuffle, and etc. It creates a need for a
 *      corresponding psABI.
 *
 *  @warning Apple Clang has hard time with bf16.
 *  https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
 *  https://forums.developer.apple.com/forums/thread/726201
 *  https://www.phoronix.com/news/GCC-LLVM-bf16-BFloat16-Type
 */
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__ARM_ARCH) || defined(__aarch64__)) && \
    (defined(__ARM_BF16_FORMAT_ALTERNATIVE))
#undef MATHKONG_NATIVE_BF16
#define MATHKONG_NATIVE_BF16 1
typedef __bf16 mathkong_bf16_t;
#elif ((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && \
       (defined(__AVX512BF16__)))
typedef __bfloat16 mathkong_bf16_t;
#undef MATHKONG_NATIVE_BF16
#define MATHKONG_NATIVE_BF16 1
#else                                       // Unknown compiler or architecture
#if defined(__GNUC__) || defined(__clang__) // Some compilers don't support warning pragmas
#warning "Unknown compiler or architecture for bfloat16."
#endif
#undef MATHKONG_NATIVE_BF16
#define MATHKONG_NATIVE_BF16 0
#endif // Unknown compiler or architecture
#endif // !MATHKONG_NATIVE_BF16

#if !MATHKONG_NATIVE_BF16
typedef unsigned short mathkong_bf16_t;
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
#if _MATHKONG_TARGET_ARM
#if defined(_MSC_VER)
#define mathkong_f16_for_arm_simd_t mathkong_f16_t
#define mathkong_bf16_for_arm_simd_t mathkong_bf16_t
#else
#define mathkong_f16_for_arm_simd_t float16_t
#define mathkong_bf16_for_arm_simd_t bfloat16_t
#endif
#endif

/*
 *  Let's make sure the sizes of the types are as expected.
 *  In C the `_Static_assert` is only available with C11 and later.
 */
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_b8_t) == 1, mathkong_b8_t_must_be_1_byte);
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_i4x2_t) == 1, mathkong_i4x2_t_must_be_1_byte);
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_i8_t) == 1, mathkong_i8_t_must_be_1_byte);
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_u8_t) == 1, mathkong_u8_t_must_be_1_byte);
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_i16_t) == 2, mathkong_i16_t_must_be_2_bytes);
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_u16_t) == 2, mathkong_u16_t_must_be_2_bytes);
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_i32_t) == 4, mathkong_i32_t_must_be_4_bytes);
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_u32_t) == 4, mathkong_u32_t_must_be_4_bytes);
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_i64_t) == 8, mathkong_i64_t_must_be_8_bytes);
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_u64_t) == 8, mathkong_u64_t_must_be_8_bytes);
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_f32_t) == 4, mathkong_f32_t_must_be_4_bytes);
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_f64_t) == 8, mathkong_f64_t_must_be_8_bytes);
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_f16_t) == 2, mathkong_f16_t_must_be_2_bytes);
_MATHKONG_STATIC_ASSERT(sizeof(mathkong_bf16_t) == 2, mathkong_bf16_t_must_be_2_bytes);

/** @brief  Convenience type for single- and half-precision floating-point type conversions. */
typedef union {
    mathkong_f32_t f;
    mathkong_u32_t u;
    mathkong_i32_t i;
} mathkong_fui32_t;

/** @brief  Convenience type for double-precision floating-point type conversions. */
typedef union {
    mathkong_f64_t f;
    mathkong_u64_t u;
    mathkong_i64_t i;
} mathkong_fui64_t;

/** @brief  Convenience type addressing the real and imaginary parts of a half-precision complex number. */
typedef struct {
    mathkong_f16_t real;
    mathkong_f16_t imag;
} mathkong_f16c_t;

/** @brief  Convenience type addressing the real and imaginary parts of a half-precision brain-float complex number. */
typedef struct {
    mathkong_bf16_t real;
    mathkong_bf16_t imag;
} mathkong_bf16c_t;

/** @brief  Convenience type addressing the real and imaginary parts of a single-precision complex number. */
typedef struct {
    mathkong_f32_t real;
    mathkong_f32_t imag;
} mathkong_f32c_t;

/** @brief  Convenience type addressing the real and imaginary parts of a double-precision complex number. */
typedef struct {
    mathkong_f64_t real;
    mathkong_f64_t imag;
} mathkong_f64c_t;

/**
 *  @brief  Computes `1/sqrt(x)` @b Square-Root-Reciprocal using the trick from Quake 3,
 *          replacing the magic numbers with the ones suggested by Jan Kadlec.
 *
 *  Subsequent additions by hardware manufacturers have made this algorithm redundant for the most part.
 *  For example, on x86, Intel introduced the SSE instruction `rsqrtss` in 1999. In a 2009 benchmark on
 *  the Intel Core 2, this instruction took 0.85ns per float compared to 3.54ns for the fast inverse
 *  square root algorithm, and had less error. Carmack's Magic Number `rsqrt` had an average error
 *  of 0.0990%, while SSE `rsqrtss` had 0.0094%, a 10x improvement.
 *
 *  https://web.archive.org/web/20210208132927/http://assemblyrequired.crashworks.org/timing-square-root/
 *  https://stackoverflow.com/a/41460625/2766161
 */
MATHKONG_INTERNAL mathkong_f32_t mathkong_f32_rsqrt(mathkong_f32_t number) {
    mathkong_fui32_t conv;
    conv.f = number;
    conv.i = 0x5F1FFFF9 - (conv.i >> 1);
    // Refine using a Newton-Raphson step for better accuracy
    conv.f *= 0.703952253f * (2.38924456f - number * conv.f * conv.f);
    return conv.f;
}

/**
 *  @brief  Approximates `sqrt(x)` using the fast inverse square root trick
 *          with adjustments for direct square root approximation.
 *
 *  Similar to `rsqrt` approximation but multiplies by `number` to get `sqrt`.
 *  This technique is useful where `sqrt` approximation is needed in performance-critical
 *  code, though modern hardware provides optimized alternatives.
 */
MATHKONG_INTERNAL mathkong_f32_t mathkong_f32_sqrt(mathkong_f32_t number) {
    return number * mathkong_f32_rsqrt(number);
}

/**
 *  @brief  Computes `log(x)` using the Mercator series.
 *          The series converges to the natural logarithm for args between -1 and 1.
 *          Published in 1668 in "Logarithmotechnia".
 */
MATHKONG_INTERNAL mathkong_f32_t mathkong_f32_log(mathkong_f32_t number) {
    mathkong_f32_t x = number - 1;
    mathkong_f32_t x2 = x * x;
    mathkong_f32_t x3 = x * x * x;
    return x - x2 / 2 + x3 / 3;
}

#define _MATHKONG_ASSIGN_1_TO_2(x, y) *(y) = *(x)

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
MATHKONG_INTERNAL mathkong_f32_t mathkong_f16_to_f32_implementation(mathkong_f16_t const *x_ptr) {
    unsigned short x;
    MATHKONG_COPY16(&x, x_ptr);
    unsigned int exponent = (x & 0x7C00) >> 10;
    unsigned int mantissa = (x & 0x03FF) << 13;
    mathkong_fui32_t mantissa_conv;
    mantissa_conv.f = (float)mantissa;
    unsigned int v = (mantissa_conv.i) >> 23;
    mathkong_fui32_t conv;
    conv.i = (x & 0x8000) << 16 | (exponent != 0) * ((exponent + 112) << 23 | mantissa) |
             ((exponent == 0) & (mantissa != 0)) * ((v - 37) << 23 | ((mantissa << (150 - v)) & 0x007FE000));
    return conv.f;
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
MATHKONG_INTERNAL void mathkong_f32_to_f16_implementation(mathkong_f32_t x, mathkong_f16_t *result_ptr) {
    mathkong_fui32_t conv;
    conv.f = x;
    unsigned int b = conv.i + 0x00001000;
    unsigned int e = (b & 0x7F800000) >> 23;
    unsigned int m = b & 0x007FFFFF;
    unsigned short result = ((b & 0x80000000) >> 16) | (e > 112) * ((((e - 112) << 10) & 0x7C00) | (m >> 13)) |
                            ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
                            ((e > 143) * 0x7FFF);
    MATHKONG_COPY16(result_ptr, &result);
}

/**
 *  @brief  For compilers that don't natively support the `__bf16` type,
 *          upcasts contents into a more conventional `float`.
 *
 *  https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c/55254307#55254307
 *  https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
 */
MATHKONG_INTERNAL mathkong_f32_t mathkong_bf16_to_f32_implementation(mathkong_bf16_t const *x_ptr) {
    unsigned short x;
    MATHKONG_COPY16(&x, x_ptr);
    mathkong_fui32_t conv;
    conv.i = x << 16; // Zero extends the mantissa
    return conv.f;
}

/**
 *  @brief  Compresses a `float` to a `bf16` representation.
 *
 *  https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c/55254307#55254307
 *  https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
 */
MATHKONG_INTERNAL void mathkong_f32_to_bf16_implementation(mathkong_f32_t x, mathkong_bf16_t *result_ptr) {
    mathkong_fui32_t conv;
    conv.f = x;
    conv.i += 0x8000; // Rounding is optional
    conv.i >>= 16;
    // The top 16 bits will be zeroed out anyways
    // conv.i &= 0xFFFF;
    MATHKONG_COPY16(result_ptr, &conv.i);
}

MATHKONG_INTERNAL void _mathkong_f16_to_f64(mathkong_f16_t const *x, mathkong_f64_t *y) {
    mathkong_f32_t f32;
    mathkong_f16_to_f32(x, &f32);
    *y = (mathkong_f64_t)f32;
}
MATHKONG_INTERNAL void _mathkong_f64_to_f16(mathkong_f64_t const *x, mathkong_f16_t *y) {
    mathkong_f32_t f32 = (mathkong_f32_t)*x;
    mathkong_f32_to_f16(&f32, y);
}
MATHKONG_INTERNAL void _mathkong_bf16_to_f64(mathkong_bf16_t const *x, mathkong_f64_t *y) {
    mathkong_f32_t f32;
    mathkong_bf16_to_f32(x, &f32);
    *y = (mathkong_f64_t)f32;
}
MATHKONG_INTERNAL void _mathkong_f64_to_bf16(mathkong_f64_t const *x, mathkong_bf16_t *y) {
    mathkong_f32_t f32 = (mathkong_f32_t)*x;
    mathkong_f32_to_bf16(&f32, y);
}

/*  Convert floating pointer numbers to integers, clamping them to the range of signed
 *  and unsigned low-resolution integers, and rounding them to the nearest integer.
 *
 *  In C++ the analogous solution with STL could be: `*y = std::clamp(std::round(*x), -128, 127)`.
 *  In C, using the standard library: `*x = fminf(fmaxf(roundf(*x), -128), 127)`.
 */
MATHKONG_INTERNAL void _mathkong_f32_to_i8(mathkong_f32_t const *x, mathkong_i8_t *y) {
    *y = (mathkong_i8_t)(*x > 127 ? 127 : (*x < -128 ? -128 : (int)(*x + (*x < 0 ? -0.5f : 0.5f))));
}

MATHKONG_INTERNAL void _mathkong_f32_to_u8(mathkong_f32_t const *x, mathkong_u8_t *y) {
    *y = (mathkong_u8_t)(*x > 255 ? 255 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5f : 0.5f))));
}

MATHKONG_INTERNAL void _mathkong_f32_to_i16(mathkong_f32_t const *x, mathkong_i16_t *y) {
    *y = (mathkong_i16_t)(*x > 32767 ? 32767 : (*x < -32768 ? -32768 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

MATHKONG_INTERNAL void _mathkong_f32_to_u16(mathkong_f32_t const *x, mathkong_u16_t *y) {
    *y = (mathkong_u16_t)(*x > 65535 ? 65535 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

MATHKONG_INTERNAL void _mathkong_f64_to_i8(mathkong_f64_t const *x, mathkong_i8_t *y) {
    *y = (mathkong_i8_t)(*x > 127 ? 127 : (*x < -128 ? -128 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

MATHKONG_INTERNAL void _mathkong_f64_to_u8(mathkong_f64_t const *x, mathkong_u8_t *y) {
    *y = (mathkong_u8_t)(*x > 255 ? 255 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

MATHKONG_INTERNAL void _mathkong_f64_to_i16(mathkong_f64_t const *x, mathkong_i16_t *y) {
    *y = (mathkong_i16_t)(*x > 32767 ? 32767 : (*x < -32768 ? -32768 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

MATHKONG_INTERNAL void _mathkong_f64_to_u16(mathkong_f64_t const *x, mathkong_u16_t *y) {
    *y = (mathkong_u16_t)(*x > 65535 ? 65535 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

MATHKONG_INTERNAL void _mathkong_f64_to_i32(mathkong_f64_t const *x, mathkong_i32_t *y) {
    *y = (mathkong_i32_t)(*x > 2147483647 ? 2147483647
                                          : (*x < -2147483648 ? -2147483648 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

MATHKONG_INTERNAL void _mathkong_f64_to_u32(mathkong_f64_t const *x, mathkong_u32_t *y) {
    *y = (mathkong_u32_t)(*x > 4294967295 ? 4294967295 : (*x < 0 ? 0 : (unsigned int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

MATHKONG_INTERNAL void _mathkong_f64_to_i64(mathkong_f64_t const *x, mathkong_i64_t *y) {
    *y = (mathkong_i64_t)(*x > 9223372036854775807.0
                              ? 9223372036854775807ll
                              : (*x < -9223372036854775808.0 ? (-9223372036854775807ll - 1ll)
                                                             : (long long)(*x + (*x < 0 ? -0.5 : 0.5))));
}

MATHKONG_INTERNAL void _mathkong_f64_to_u64(mathkong_f64_t const *x, mathkong_u64_t *y) {
    *y =
        (mathkong_u64_t)(*x > 18446744073709551615.0 ? 18446744073709551615ull
                                                     : (*x < 0 ? 0 : (unsigned long long)(*x + (*x < 0 ? -0.5 : 0.5))));
}

MATHKONG_INTERNAL void _mathkong_i64_to_i8(mathkong_i64_t const *x, mathkong_i8_t *y) {
    *y = (mathkong_i8_t)(*x > 127ll ? 127ll : (*x < -128ll ? -128ll : *x));
}

MATHKONG_INTERNAL void _mathkong_i64_to_u8(mathkong_i64_t const *x, mathkong_u8_t *y) {
    *y = (mathkong_u8_t)(*x > 255ll ? 255ll : (*x < 0ll ? 0ll : *x));
}

MATHKONG_INTERNAL void _mathkong_i64_to_i16(mathkong_i64_t const *x, mathkong_i16_t *y) {
    *y = (mathkong_i16_t)(*x > 32767ll ? 32767ll : (*x < -32768ll ? -32768ll : *x));
}

MATHKONG_INTERNAL void _mathkong_i64_to_u16(mathkong_i64_t const *x, mathkong_u16_t *y) {
    *y = (mathkong_u16_t)(*x > 65535ll ? 65535ll : (*x < 0ll ? 0ll : *x));
}

MATHKONG_INTERNAL void _mathkong_i64_to_i32(mathkong_i64_t const *x, mathkong_i32_t *y) {
    *y = (mathkong_i32_t)(*x > 2147483647ll ? 2147483647ll : (*x < -2147483648ll ? -2147483648ll : *x));
}

MATHKONG_INTERNAL void _mathkong_i64_to_u32(mathkong_i64_t const *x, mathkong_u32_t *y) {
    *y = (mathkong_u32_t)(*x > 4294967295ll ? 4294967295ll : (*x < 0ll ? 0ll : *x));
}

MATHKONG_INTERNAL void _mathkong_u64_to_i8(mathkong_u64_t const *x, mathkong_i8_t *y) {
    *y = (mathkong_i8_t)(*x > 127ull ? 127ull : *x);
}

MATHKONG_INTERNAL void _mathkong_u64_to_u8(mathkong_u64_t const *x, mathkong_u8_t *y) {
    *y = (mathkong_u8_t)(*x > 255ull ? 255ull : *x);
}

MATHKONG_INTERNAL void _mathkong_u64_to_i16(mathkong_u64_t const *x, mathkong_i16_t *y) {
    *y = (mathkong_i16_t)(*x > 32767ull ? 32767ull : *x);
}

MATHKONG_INTERNAL void _mathkong_u64_to_u16(mathkong_u64_t const *x, mathkong_u16_t *y) {
    *y = (mathkong_u16_t)(*x > 65535ull ? 65535ull : *x);
}

MATHKONG_INTERNAL void _mathkong_u64_to_i32(mathkong_u64_t const *x, mathkong_i32_t *y) {
    *y = (mathkong_i32_t)(*x > 2147483647ull ? 2147483647ull : *x);
}

MATHKONG_INTERNAL void _mathkong_u64_to_u32(mathkong_u64_t const *x, mathkong_u32_t *y) {
    *y = (mathkong_u32_t)(*x > 4294967295ull ? 4294967295ull : *x);
}

MATHKONG_INTERNAL void _mathkong_f64_to_f32(mathkong_f64_t const *x, mathkong_f32_t *y) { *y = (mathkong_f32_t)*x; }
MATHKONG_INTERNAL void _mathkong_u64_to_f32(mathkong_u64_t const *x, mathkong_f32_t *y) { *y = (mathkong_f32_t)*x; }
MATHKONG_INTERNAL void _mathkong_i64_to_f32(mathkong_i64_t const *x, mathkong_f32_t *y) { *y = (mathkong_f32_t)*x; }

MATHKONG_INTERNAL void _mathkong_f32_to_f64(mathkong_f32_t const *x, mathkong_f64_t *y) { *y = (mathkong_f64_t)*x; }

MATHKONG_INTERNAL void _mathkong_f64_to_f64(mathkong_f64_t const *x, mathkong_f64_t *y) { *y = *x; }

MATHKONG_INTERNAL void _mathkong_i8_to_f64(mathkong_i8_t const *x, mathkong_f64_t *y) { *y = (mathkong_f64_t)*x; }
MATHKONG_INTERNAL void _mathkong_i16_to_f64(mathkong_i16_t const *x, mathkong_f64_t *y) { *y = (mathkong_f64_t)*x; }
MATHKONG_INTERNAL void _mathkong_i32_to_f64(mathkong_i32_t const *x, mathkong_f64_t *y) { *y = (mathkong_f64_t)*x; }
MATHKONG_INTERNAL void _mathkong_i64_to_f64(mathkong_i64_t const *x, mathkong_f64_t *y) { *y = (mathkong_f64_t)*x; }
MATHKONG_INTERNAL void _mathkong_u8_to_f64(mathkong_u8_t const *x, mathkong_f64_t *y) { *y = (mathkong_f64_t)*x; }
MATHKONG_INTERNAL void _mathkong_u16_to_f64(mathkong_u16_t const *x, mathkong_f64_t *y) { *y = (mathkong_f64_t)*x; }
MATHKONG_INTERNAL void _mathkong_u32_to_f64(mathkong_u32_t const *x, mathkong_f64_t *y) { *y = (mathkong_f64_t)*x; }
MATHKONG_INTERNAL void _mathkong_u64_to_f64(mathkong_u64_t const *x, mathkong_f64_t *y) { *y = (mathkong_f64_t)*x; }

MATHKONG_INTERNAL void _mathkong_i8_to_i64(mathkong_i8_t const *x, mathkong_i64_t *y) { *y = (mathkong_i64_t)*x; }
MATHKONG_INTERNAL void _mathkong_i16_to_i64(mathkong_i16_t const *x, mathkong_i64_t *y) { *y = (mathkong_i64_t)*x; }
MATHKONG_INTERNAL void _mathkong_i32_to_i64(mathkong_i32_t const *x, mathkong_i64_t *y) { *y = (mathkong_i64_t)*x; }
MATHKONG_INTERNAL void _mathkong_i64_to_i64(mathkong_i64_t const *x, mathkong_i64_t *y) { *y = *x; }
MATHKONG_INTERNAL void _mathkong_u8_to_i64(mathkong_u8_t const *x, mathkong_i64_t *y) { *y = (mathkong_i64_t)*x; }
MATHKONG_INTERNAL void _mathkong_u16_to_i64(mathkong_u16_t const *x, mathkong_i64_t *y) { *y = (mathkong_i64_t)*x; }
MATHKONG_INTERNAL void _mathkong_u32_to_i64(mathkong_u32_t const *x, mathkong_i64_t *y) { *y = (mathkong_i64_t)*x; }
MATHKONG_INTERNAL void _mathkong_u64_to_i64(mathkong_u64_t const *x, mathkong_i64_t *y) {
    *y = (mathkong_i64_t)(*x >= 9223372036854775807ull ? 9223372036854775807ll : *x);
}

MATHKONG_INTERNAL void _mathkong_i8_to_u64(mathkong_i8_t const *x, mathkong_u64_t *y) {
    *y = (mathkong_u64_t)(*x < 0 ? 0 : *x);
}
MATHKONG_INTERNAL void _mathkong_i16_to_u64(mathkong_i16_t const *x, mathkong_u64_t *y) {
    *y = (mathkong_u64_t)(*x < 0 ? 0 : *x);
}
MATHKONG_INTERNAL void _mathkong_i32_to_u64(mathkong_i32_t const *x, mathkong_u64_t *y) {
    *y = (mathkong_u64_t)(*x < 0 ? 0 : *x);
}
MATHKONG_INTERNAL void _mathkong_i64_to_u64(mathkong_i64_t const *x, mathkong_u64_t *y) {
    *y = (mathkong_u64_t)(*x < 0 ? 0 : *x);
}
MATHKONG_INTERNAL void _mathkong_u8_to_u64(mathkong_u8_t const *x, mathkong_u64_t *y) { *y = (mathkong_u64_t)*x; }
MATHKONG_INTERNAL void _mathkong_u16_to_u64(mathkong_u16_t const *x, mathkong_u64_t *y) { *y = (mathkong_u64_t)*x; }
MATHKONG_INTERNAL void _mathkong_u32_to_u64(mathkong_u32_t const *x, mathkong_u64_t *y) { *y = (mathkong_u64_t)*x; }
MATHKONG_INTERNAL void _mathkong_u64_to_u64(mathkong_u64_t const *x, mathkong_u64_t *y) { *y = *x; }

MATHKONG_INTERNAL void _mathkong_i64_to_f16(mathkong_i64_t const *x, mathkong_f16_t *y) {
    mathkong_f32_t f32 = (mathkong_f32_t)*x;
    mathkong_f32_to_f16(&f32, y);
}
MATHKONG_INTERNAL void _mathkong_i64_to_bf16(mathkong_i64_t const *x, mathkong_bf16_t *y) {
    mathkong_f32_t f32 = (mathkong_f32_t)*x;
    mathkong_f32_to_bf16(&f32, y);
}
MATHKONG_INTERNAL void _mathkong_u64_to_f16(mathkong_u64_t const *x, mathkong_f16_t *y) {
    mathkong_f32_t f32 = (mathkong_f32_t)*x;
    mathkong_f32_to_f16(&f32, y);
}
MATHKONG_INTERNAL void _mathkong_u64_to_bf16(mathkong_u64_t const *x, mathkong_bf16_t *y) {
    mathkong_f32_t f32 = (mathkong_f32_t)*x;
    mathkong_f32_to_bf16(&f32, y);
}

/**
 *  @brief  Helper structure for implementing strided matrix row lookups, with @b single-byte-level pointer math.
 */
MATHKONG_INTERNAL void *_mathkong_advance_by_bytes(void *ptr, mathkong_size_t bytes) {
    return (void *)((mathkong_u8_t *)ptr + bytes);
}

/**
 *  @brief  Divide and round up to the nearest integer.
 */
MATHKONG_INTERNAL mathkong_size_t _mathkong_divide_ceil(mathkong_size_t dividend, mathkong_size_t divisor) {
    return (dividend + divisor - 1) / divisor;
}

/**
 *  @brief Advances the Multi-Dimensional iterator to the next set of indicies.
 *  @param[in] extents The extents of the tensor, defined by an array with at least `rank` scalars.
 *  @param[in] strides The @b signed strides of the tensor in bytes, defined by an array with at least `rank`
 * scalars.
 *  @param[in] rank The number of dimensions in the tensor (its rank).
 *  @param[inout] coordinates The array of offsets along each of `rank` dimensions, which will be updated.
 *  @param[inout] byte_offset The @b signed byte offset of the current element, which will be advanced.
 *  @return 1 if the iterator was successfully advanced, 0 if the end of iteration was reached.
 *
 *  For flexibility, the API is decoupled from from the `mathkong_xd_index_t` structure, and
 *  can be used on any-rank tensors, independent of the `MATHKONG_NDARRAY_MAX_RANK` constant.
 */
MATHKONG_PUBLIC int mathkong_xd_index_next(                                                //
    mathkong_size_t const *extents, mathkong_ssize_t const *strides, mathkong_size_t rank, //
    mathkong_size_t *coordinates, mathkong_ssize_t *byte_offset) {
    // Start from last dimension and move backward
    for (mathkong_size_t i = rank; i-- > 0;) {
        coordinates[i]++;
        *byte_offset += strides[i];
        if (coordinates[i] < extents[i]) return 1; // Successfully moved to the next index
        coordinates[i] = 0;                        // Reset this dimension counter
        *byte_offset -= strides[i] * extents[i];   // Discard the running progress along this dimension
    }
    // If we reach here, we've iterated over all elements
    return 0; // End of iteration
}

/**
 *  @brief Advances the Multi-Dimensional iterator to the provided coordinates, updating the byte offset.
 *  @param[in] extents The extents of the tensor, defined by an array with at least `rank` scalars.
 *  @param[in] strides The @b signed strides of the tensor in bytes, defined by an array with at least `rank`
 * scalars.
 *  @param[in] rank The number of dimensions in the tensor (its rank).
 *  @param[in] coordinates The array of offsets along each of `rank` dimensions, which will be updated.
 *  @param[out] byte_offset The byte offset of the current element, which will be advanced.
 *  @return 1 if the offset was successfully advanced, 0 if the end of iteration was reached.
 */
MATHKONG_PUBLIC int mathkong_xd_index_linearize(                                           //
    mathkong_size_t const *extents, mathkong_ssize_t const *strides, mathkong_size_t rank, //
    mathkong_size_t const *coordinates, mathkong_ssize_t *byte_offset) {

    mathkong_ssize_t result = 0;
    for (mathkong_size_t i = 0; i < rank; i++) {
        // Ensure the coordinates is within bounds for the given dimension
        if (coordinates[i] >= extents[i]) return 0; // Invalid coordinates, out of bounds
        // Update the byte offset by multiplying the coordinates by the stride
        result += coordinates[i] * strides[i];
    }
    *byte_offset = result;
    return 1; // Successfully calculated global and byte offsets
}

/**
 *  @brief  A @b beefy structure to iterate through Multi-Dimensional arrays.
 *          Occupies 512 + 8 = 520 bytes on a 64-bit machine, or @b 9 cache-lines, by default.
 *
 *  When advancing through a structure, its overall size and strides should be stored somewhere else.
 *  The `byte_offset` starts at zero and grow monotonically during iteration, if the strides are positive.
 */
typedef struct mathkong_xd_index_t {
    mathkong_size_t coordinates[MATHKONG_NDARRAY_MAX_RANK]; // Coordinate offsets along each dimension
    mathkong_ssize_t byte_offset;                           // Byte offset of the current element
} mathkong_xd_index_t;

MATHKONG_PUBLIC void mathkong_xd_index_init(mathkong_xd_index_t *xd_index) {
    for (mathkong_size_t i = 0; i < MATHKONG_NDARRAY_MAX_RANK; i++) xd_index->coordinates[i] = 0;
    xd_index->byte_offset = 0;
}

/**
 *  @brief  A @b beefy structure describing the shape and memory layout of a Multi-Dimensional array.
 *          Similar to `md::span` in C++20 and `numpy.ndarray` in Python, but with a focus on compatibility.
 *          Occupies 512 + 512 + 8 = 2052 bytes on a 64-bit machine, or @b 17 cache-lines, by default.
 *
 *  Unlike NumPy and the CPython "Buffer Protocol", we don't use `suboffsets` for pointer indirection.
 *  The logic is that such layouts aren't friendly to conventional SIMD operations and dense matrix algorithms.
 *  If the tensor is sparse, consider using a different data structure or a different memory layout.
 *
 *  Most MathKong algorithms don't work with the entire structure, but expect the fields to be passed separately.
 *  It would also require storing the @b start-pointer and the @b datatype/item-size separately, as it's not
 *  stored inside the structure.
 */
typedef struct mathkong_xd_span_t {
    mathkong_size_t extents[MATHKONG_NDARRAY_MAX_RANK];  /// Number of elements along each dimension
    mathkong_ssize_t strides[MATHKONG_NDARRAY_MAX_RANK]; /// Strides of the tensor in bytes
    mathkong_size_t rank;                                /// Number of dimensions in the tensor
} mathkong_xd_span_t;

MATHKONG_PUBLIC void mathkong_xd_span_init(mathkong_xd_span_t *xd_span) {
    for (mathkong_size_t i = 0; i < MATHKONG_NDARRAY_MAX_RANK; i++) xd_span->extents[i] = 0, xd_span->strides[i] = 0;
    xd_span->rank = 0;
}

MATHKONG_INTERNAL mathkong_u32_t _mathkong_u32_rol(mathkong_u32_t *x, int n) { return (*x << n) | (*x >> (32 - n)); }
MATHKONG_INTERNAL mathkong_u16_t _mathkong_u16_rol(mathkong_u16_t *x, int n) { return (*x << n) | (*x >> (16 - n)); }
MATHKONG_INTERNAL mathkong_u8_t _mathkong_u8_rol(mathkong_u8_t *x, int n) { return (*x << n) | (*x >> (8 - n)); }
MATHKONG_INTERNAL mathkong_u32_t _mathkong_u32_ror(mathkong_u32_t *x, int n) { return (*x >> n) | (*x << (32 - n)); }
MATHKONG_INTERNAL mathkong_u16_t _mathkong_u16_ror(mathkong_u16_t *x, int n) { return (*x >> n) | (*x << (16 - n)); }
MATHKONG_INTERNAL mathkong_u8_t _mathkong_u8_ror(mathkong_u8_t *x, int n) { return (*x >> n) | (*x << (8 - n)); }

MATHKONG_INTERNAL void _mathkong_u8_sadd(mathkong_u8_t const *a, mathkong_u8_t const *b, mathkong_u8_t *r) {
    mathkong_u16_t result = (mathkong_u16_t)*a + (mathkong_u16_t)*b;
    *r = (result > 255u) ? (mathkong_u8_t)255u : (mathkong_u8_t)result;
}
MATHKONG_INTERNAL void _mathkong_u16_sadd(mathkong_u16_t const *a, mathkong_u16_t const *b, mathkong_u16_t *r) {
    mathkong_u32_t result = (mathkong_u32_t)*a + (mathkong_u32_t)*b;
    *r = (result > 65535u) ? (mathkong_u16_t)65535u : (mathkong_u16_t)result;
}
MATHKONG_INTERNAL void _mathkong_u32_sadd(mathkong_u32_t const *a, mathkong_u32_t const *b, mathkong_u32_t *r) {
    mathkong_u64_t result = (mathkong_u64_t)*a + (mathkong_u64_t)*b;
    *r = (result > 4294967295u) ? (mathkong_u32_t)4294967295u : (mathkong_u32_t)result;
}
MATHKONG_INTERNAL void _mathkong_u64_sadd(mathkong_u64_t const *a, mathkong_u64_t const *b, mathkong_u64_t *r) {
    *r = (*a + *b < *a) ? 18446744073709551615ull : (*a + *b);
}
MATHKONG_INTERNAL void _mathkong_i8_sadd(mathkong_i8_t const *a, mathkong_i8_t const *b, mathkong_i8_t *r) {
    mathkong_i16_t result = (mathkong_i16_t)*a + (mathkong_i16_t)*b;
    *r = (result > 127) ? 127 : (result < -128 ? -128 : result);
}
MATHKONG_INTERNAL void _mathkong_i16_sadd(mathkong_i16_t const *a, mathkong_i16_t const *b, mathkong_i16_t *r) {
    mathkong_i32_t result = (mathkong_i32_t)*a + (mathkong_i32_t)*b;
    *r = (result > 32767) ? 32767 : (result < -32768 ? -32768 : result);
}
MATHKONG_INTERNAL void _mathkong_i32_sadd(mathkong_i32_t const *a, mathkong_i32_t const *b, mathkong_i32_t *r) {
    mathkong_i64_t result = (mathkong_i64_t)*a + (mathkong_i64_t)*b;
    *r = (result > 2147483647ll) ? 2147483647ll : (result < -2147483648ll ? -2147483648ll : (mathkong_i32_t)result);
}
MATHKONG_INTERNAL void _mathkong_i64_sadd(mathkong_i64_t const *a, mathkong_i64_t const *b, mathkong_i64_t *r) {
    //? We can't just write `-9223372036854775808ll`, even though it's the smallest signed 64-bit value.
    //? The compiler will complain about the number being too large for the type, as it will process the
    //? constant and the sign separately. So we use the same hint that compilers use to define the `INT64_MIN`.
    if ((*b > 0) && (*a > (9223372036854775807ll) - *b)) { *r = 9223372036854775807ll; }                    // Overflow
    else if ((*b < 0) && (*a < (-9223372036854775807ll - 1ll) - *b)) { *r = -9223372036854775807ll - 1ll; } // Underflow
    else { *r = *a + *b; }
}
MATHKONG_INTERNAL void _mathkong_f32_sadd(mathkong_f32_t const *a, mathkong_f32_t const *b, mathkong_f32_t *r) {
    *r = *a + *b;
}
MATHKONG_INTERNAL void _mathkong_f64_sadd(mathkong_f64_t const *a, mathkong_f64_t const *b, mathkong_f64_t *r) {
    *r = *a + *b;
}
MATHKONG_INTERNAL void _mathkong_f16_sadd(mathkong_f16_t const *a, mathkong_f16_t const *b, mathkong_f16_t *r) {
    mathkong_f32_t a_f32, b_f32, r_f32;
    mathkong_f16_to_f32(a, &a_f32);
    mathkong_f16_to_f32(b, &b_f32);
    r_f32 = a_f32 + b_f32;
    mathkong_f32_to_f16(&r_f32, r);
}
MATHKONG_INTERNAL void _mathkong_bf16_sadd(mathkong_bf16_t const *a, mathkong_bf16_t const *b, mathkong_bf16_t *r) {
    mathkong_f32_t a_f32, b_f32, r_f32;
    mathkong_bf16_to_f32(a, &a_f32);
    mathkong_bf16_to_f32(b, &b_f32);
    r_f32 = a_f32 + b_f32;
    mathkong_f32_to_bf16(&r_f32, r);
}

MATHKONG_INTERNAL void _mathkong_u8_smul(mathkong_u8_t const *a, mathkong_u8_t const *b, mathkong_u8_t *r) {
    mathkong_u16_t result = (mathkong_u16_t)(*a) * (mathkong_u16_t)(*b);
    *r = (result > 255) ? 255 : (mathkong_u8_t)result;
}

MATHKONG_INTERNAL void _mathkong_u16_smul(mathkong_u16_t const *a, mathkong_u16_t const *b, mathkong_u16_t *r) {
    mathkong_u32_t result = (mathkong_u32_t)(*a) * (mathkong_u32_t)(*b);
    *r = (result > 65535) ? 65535 : (mathkong_u16_t)result;
}

MATHKONG_INTERNAL void _mathkong_u32_smul(mathkong_u32_t const *a, mathkong_u32_t const *b, mathkong_u32_t *r) {
    mathkong_u64_t result = (mathkong_u64_t)(*a) * (mathkong_u64_t)(*b);
    *r = (result > 4294967295u) ? 4294967295u : (mathkong_u32_t)result;
}

MATHKONG_INTERNAL void _mathkong_u64_smul(mathkong_u64_t const *a, mathkong_u64_t const *b, mathkong_u64_t *r) {
    // Split the inputs into high and low 32-bit parts
    mathkong_u64_t a_hi = *a >> 32;
    mathkong_u64_t a_lo = *a & 0xFFFFFFFF;
    mathkong_u64_t b_hi = *b >> 32;
    mathkong_u64_t b_lo = *b & 0xFFFFFFFF;

    // Compute partial products
    mathkong_u64_t hi_hi = a_hi * b_hi;
    mathkong_u64_t hi_lo = a_hi * b_lo;
    mathkong_u64_t lo_hi = a_lo * b_hi;
    mathkong_u64_t lo_lo = a_lo * b_lo;

    // Check if the high part of the result overflows
    if (hi_hi || (hi_lo >> 32) || (lo_hi >> 32) || ((hi_lo + lo_hi) >> 32)) { *r = 18446744073709551615ull; }
    else { *r = (hi_lo << 32) + (lo_hi << 32) + lo_lo; } // Combine parts if no overflow
}

MATHKONG_INTERNAL void _mathkong_i8_smul(mathkong_i8_t const *a, mathkong_i8_t const *b, mathkong_i8_t *r) {
    mathkong_i16_t result = (mathkong_i16_t)(*a) * (mathkong_i16_t)(*b);
    *r = (result > 127) ? 127 : (result < -128 ? -128 : (mathkong_i8_t)result);
}

MATHKONG_INTERNAL void _mathkong_i16_smul(mathkong_i16_t const *a, mathkong_i16_t const *b, mathkong_i16_t *r) {
    mathkong_i32_t result = (mathkong_i32_t)(*a) * (mathkong_i32_t)(*b);
    *r = (result > 32767) ? 32767 : (result < -32768 ? -32768 : (mathkong_i16_t)result);
}

MATHKONG_INTERNAL void _mathkong_i32_smul(mathkong_i32_t const *a, mathkong_i32_t const *b, mathkong_i32_t *r) {
    mathkong_i64_t result = (mathkong_i64_t)(*a) * (mathkong_i64_t)(*b);
    *r = (result > 2147483647ll) ? 2147483647ll : (result < -2147483648ll ? -2147483648ll : (mathkong_i32_t)result);
}

MATHKONG_INTERNAL void _mathkong_i64_smul(mathkong_i64_t const *a, mathkong_i64_t const *b, mathkong_i64_t *r) {
    int sign = ((*a < 0) ^ (*b < 0)) ? -1 : 1; // Track the sign of the result

    // Take absolute values for easy multiplication and overflow detection
    mathkong_u64_t abs_a = (*a < 0) ? -*a : *a;
    mathkong_u64_t abs_b = (*b < 0) ? -*b : *b;

    // Split the absolute values into high and low 32-bit parts
    mathkong_u64_t a_hi = abs_a >> 32;
    mathkong_u64_t a_lo = abs_a & 0xFFFFFFFF;
    mathkong_u64_t b_hi = abs_b >> 32;
    mathkong_u64_t b_lo = abs_b & 0xFFFFFFFF;

    // Compute partial products
    mathkong_u64_t hi_hi = a_hi * b_hi;
    mathkong_u64_t hi_lo = a_hi * b_lo;
    mathkong_u64_t lo_hi = a_lo * b_hi;
    mathkong_u64_t lo_lo = a_lo * b_lo;

    // Check for overflow and saturate based on sign
    if (hi_hi || (hi_lo >> 32) || (lo_hi >> 32) || ((hi_lo + lo_hi) >> 32)) {
        *r = (sign > 0) ? 9223372036854775807ll : (-9223372036854775807ll - 1ll);
    }
    // Combine parts if no overflow, then apply the sign
    else {
        mathkong_u64_t result = (hi_lo << 32) + (lo_hi << 32) + lo_lo;
        *r = (sign < 0) ? -((mathkong_i64_t)result) : (mathkong_i64_t)result;
    }
}

MATHKONG_INTERNAL void _mathkong_f32_smul(mathkong_f32_t const *a, mathkong_f32_t const *b, mathkong_f32_t *r) {
    *r = *a * *b;
}

MATHKONG_INTERNAL void _mathkong_f64_smul(mathkong_f64_t const *a, mathkong_f64_t const *b, mathkong_f64_t *r) {
    *r = *a * *b;
}

MATHKONG_INTERNAL void _mathkong_f16_smul(mathkong_f16_t const *a, mathkong_f16_t const *b, mathkong_f16_t *r) {
    mathkong_f32_t a_f32, b_f32, r_f32;
    mathkong_f16_to_f32(a, &a_f32);
    mathkong_f16_to_f32(b, &b_f32);
    r_f32 = a_f32 * b_f32;
    mathkong_f32_to_f16(&r_f32, r);
}

MATHKONG_INTERNAL void _mathkong_bf16_smul(mathkong_bf16_t const *a, mathkong_bf16_t const *b, mathkong_bf16_t *r) {
    mathkong_f32_t a_f32, b_f32, r_f32;
    mathkong_bf16_to_f32(a, &a_f32);
    mathkong_bf16_to_f32(b, &b_f32);
    r_f32 = a_f32 * b_f32;
    mathkong_f32_to_bf16(&r_f32, r);
}

#if MATHKONG_DYNAMIC_DISPATCH

/** @copydoc mathkong_f16_to_f32_implementation */
MATHKONG_DYNAMIC mathkong_f32_t mathkong_f16_to_f32(mathkong_f16_t const *x_ptr);

/** @copydoc mathkong_f32_to_f16_implementation */
MATHKONG_DYNAMIC void mathkong_f32_to_f16(mathkong_f32_t x, mathkong_f16_t *result_ptr);

/** @copydoc mathkong_bf16_to_f32_implementation */
MATHKONG_DYNAMIC mathkong_f32_t mathkong_bf16_to_f32(mathkong_bf16_t const *x_ptr);

/** @copydoc mathkong_f32_to_bf16_implementation */
MATHKONG_DYNAMIC void mathkong_f32_to_bf16(mathkong_f32_t x, mathkong_bf16_t *result_ptr);

#else // MATHKONG_DYNAMIC_DISPATCH

/** @copydoc mathkong_f16_to_f32_implementation */
MATHKONG_PUBLIC mathkong_f32_t mathkong_f16_to_f32(mathkong_f16_t const *x_ptr) {
    return mathkong_f16_to_f32_implementation(x_ptr);
}

/** @copydoc mathkong_f32_to_f16_implementation */
MATHKONG_PUBLIC void mathkong_f32_to_f16(mathkong_f32_t x, mathkong_f16_t *result_ptr) {
    mathkong_f32_to_f16_implementation(x, result_ptr);
}

/** @copydoc mathkong_bf16_to_f32_implementation */
MATHKONG_PUBLIC mathkong_f32_t mathkong_bf16_to_f32(mathkong_bf16_t const *x_ptr) {
    return mathkong_bf16_to_f32_implementation(x_ptr);
}

/** @copydoc mathkong_f32_to_bf16_implementation */
MATHKONG_PUBLIC void mathkong_f32_to_bf16(mathkong_f32_t x, mathkong_bf16_t *result_ptr) {
    mathkong_f32_to_bf16_implementation(x, result_ptr);
}

#endif // MATHKONG_DYNAMIC_DISPATCH

#ifdef __cplusplus
} // extern "C"
#endif

#endif
