/**
 *  @file       types.h
 *  @brief      Shared definitions for the SimSIMD library.
 *  @author     Ash Vardanian
 *  @date       October 2, 2023
 *
 *  Defines:
 *
 *  - Sized aliases for numeric types, like: `simsimd_i32_t` and `simsimd_f64_t`.
 *  - Macros for internal compiler/hardware checks, like: `_SIMSIMD_TARGET_ARM`.
 *  - Macros for feature controls, like: `SIMSIMD_TARGET_NEON`
 *
 */
#ifndef SIMSIMD_TYPES_H
#define SIMSIMD_TYPES_H

// Inferring target OS: Windows, macOS, or Linux
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define _SIMSIMD_DEFINED_WINDOWS 1
#elif defined(__APPLE__) && defined(__MACH__)
#define _SIMSIMD_DEFINED_APPLE 1
#elif defined(__linux__)
#define _SIMSIMD_DEFINED_LINUX 1
#endif

// Annotation for the public API symbols:
//
// - `SIMSIMD_PUBLIC` is used for functions that are part of the public API.
// - `SIMSIMD_INTERNAL` is used for internal helper functions with unstable APIs.
// - `SIMSIMD_DYNAMIC` is used for functions that are part of the public API, but are dispatched at runtime.
//
// On GCC we mark the functions as `nonnull` informing that none of the arguments can be `NULL`.
// Marking with `pure` and `const` isn't possible as outputting to a pointer is a "side effect".
#if defined(_WIN32) || defined(__CYGWIN__)
#define SIMSIMD_DYNAMIC __declspec(dllexport)
#define SIMSIMD_PUBLIC inline static
#define SIMSIMD_INTERNAL inline static
#elif defined(__GNUC__) || defined(__clang__)
#define SIMSIMD_DYNAMIC __attribute__((visibility("default"))) __attribute__((nonnull))
#define SIMSIMD_PUBLIC __attribute__((unused, nonnull)) inline static
#define SIMSIMD_INTERNAL __attribute__((always_inline)) inline static
#else
#define SIMSIMD_DYNAMIC
#define SIMSIMD_PUBLIC inline static
#define SIMSIMD_INTERNAL inline static
#endif

// Compiling for Arm: _SIMSIMD_TARGET_ARM
#if !defined(_SIMSIMD_TARGET_ARM)
#if defined(__aarch64__) || defined(_M_ARM64)
#define _SIMSIMD_TARGET_ARM 1
#else
#define _SIMSIMD_TARGET_ARM 0
#endif // defined(__aarch64__) || defined(_M_ARM64)
#endif // !defined(_SIMSIMD_TARGET_ARM)

// Compiling for x86: _SIMSIMD_TARGET_X86
#if !defined(_SIMSIMD_TARGET_X86)
#if defined(__x86_64__) || defined(_M_X64)
#define _SIMSIMD_TARGET_X86 1
#else
#define _SIMSIMD_TARGET_X86 0
#endif // defined(__x86_64__) || defined(_M_X64)
#endif // !defined(_SIMSIMD_TARGET_X86)

// Compiling for Arm: SIMSIMD_TARGET_NEON
#if !defined(SIMSIMD_TARGET_NEON) || (SIMSIMD_TARGET_NEON && !_SIMSIMD_TARGET_ARM)
#if defined(__ARM_NEON)
#define SIMSIMD_TARGET_NEON _SIMSIMD_TARGET_ARM
#else
#undef SIMSIMD_TARGET_NEON
#define SIMSIMD_TARGET_NEON 0
#endif // defined(__ARM_NEON)
#endif // !defined(SIMSIMD_TARGET_NEON) || ...

// Compiling for Arm: SIMSIMD_TARGET_NEON_I8
#if !defined(SIMSIMD_TARGET_NEON_I8) || (SIMSIMD_TARGET_NEON_I8 && !_SIMSIMD_TARGET_ARM)
#if defined(__ARM_NEON)
#define SIMSIMD_TARGET_NEON_I8 _SIMSIMD_TARGET_ARM
#else
#undef SIMSIMD_TARGET_NEON_I8
#define SIMSIMD_TARGET_NEON_I8 0
#endif // defined(__ARM_NEON)
#endif // !defined(SIMSIMD_TARGET_NEON_I8) || ...

// Compiling for Arm: SIMSIMD_TARGET_NEON_F16
#if !defined(SIMSIMD_TARGET_NEON_F16) || (SIMSIMD_TARGET_NEON_F16 && !_SIMSIMD_TARGET_ARM)
#if defined(__ARM_NEON)
#define SIMSIMD_TARGET_NEON_F16 _SIMSIMD_TARGET_ARM
#else
#undef SIMSIMD_TARGET_NEON_F16
#define SIMSIMD_TARGET_NEON_F16 0
#endif // defined(__ARM_NEON)
#endif // !defined(SIMSIMD_TARGET_NEON_F16) || ...

// Compiling for Arm: SIMSIMD_TARGET_NEON_BF16
#if !defined(SIMSIMD_TARGET_NEON_BF16) || (SIMSIMD_TARGET_NEON_BF16 && !_SIMSIMD_TARGET_ARM)
#if defined(__ARM_NEON)
#define SIMSIMD_TARGET_NEON_BF16 _SIMSIMD_TARGET_ARM
#else
#undef SIMSIMD_TARGET_NEON_BF16
#define SIMSIMD_TARGET_NEON_BF16 0
#endif // defined(__ARM_NEON)
#endif // !defined(SIMSIMD_TARGET_NEON_BF16) || ...

// Compiling for Arm: SIMSIMD_TARGET_SVE
#if !defined(SIMSIMD_TARGET_SVE) || (SIMSIMD_TARGET_SVE && !_SIMSIMD_TARGET_ARM)
#if defined(__ARM_FEATURE_SVE)
#define SIMSIMD_TARGET_SVE _SIMSIMD_TARGET_ARM
#else
#undef SIMSIMD_TARGET_SVE
#define SIMSIMD_TARGET_SVE 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(SIMSIMD_TARGET_SVE) || ...

// Compiling for Arm: SIMSIMD_TARGET_SVE_I8
#if !defined(SIMSIMD_TARGET_SVE_I8) || (SIMSIMD_TARGET_SVE_I8 && !_SIMSIMD_TARGET_ARM)
#if defined(__ARM_FEATURE_SVE)
#define SIMSIMD_TARGET_SVE_I8 _SIMSIMD_TARGET_ARM
#else
#undef SIMSIMD_TARGET_SVE_I8
#define SIMSIMD_TARGET_SVE_I8 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(SIMSIMD_TARGET_SVE_I8) || ...

// Compiling for Arm: SIMSIMD_TARGET_SVE_F16
#if !defined(SIMSIMD_TARGET_SVE_F16) || (SIMSIMD_TARGET_SVE_F16 && !_SIMSIMD_TARGET_ARM)
#if defined(__ARM_FEATURE_SVE)
#define SIMSIMD_TARGET_SVE_F16 _SIMSIMD_TARGET_ARM
#else
#undef SIMSIMD_TARGET_SVE_F16
#define SIMSIMD_TARGET_SVE_F16 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(SIMSIMD_TARGET_SVE_F16) || ...

// Compiling for Arm: SIMSIMD_TARGET_SVE_BF16
#if !defined(SIMSIMD_TARGET_SVE_BF16) || (SIMSIMD_TARGET_SVE_BF16 && !_SIMSIMD_TARGET_ARM)
#if defined(__ARM_FEATURE_SVE)
#define SIMSIMD_TARGET_SVE_BF16 _SIMSIMD_TARGET_ARM
#else
#undef SIMSIMD_TARGET_SVE_BF16
#define SIMSIMD_TARGET_SVE_BF16 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(SIMSIMD_TARGET_SVE_BF16) || ...

// Compiling for Arm: SIMSIMD_TARGET_SVE2
#if !defined(SIMSIMD_TARGET_SVE2) || (SIMSIMD_TARGET_SVE2 && !_SIMSIMD_TARGET_ARM)
#if defined(__ARM_FEATURE_SVE)
#define SIMSIMD_TARGET_SVE2 _SIMSIMD_TARGET_ARM
#else
#undef SIMSIMD_TARGET_SVE2
#define SIMSIMD_TARGET_SVE2 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(SIMSIMD_TARGET_SVE2) || ...

// Compiling for x86: SIMSIMD_TARGET_HASWELL
//
// Starting with Ivy Bridge, Intel supports the `F16C` extensions for fast half-precision
// to single-precision floating-point conversions. On AMD those instructions
// are supported on all CPUs starting with Jaguar 2009.
// Starting with Sandy Bridge, Intel adds basic AVX support in their CPUs and in 2013
// extends it with AVX2 in the Haswell generation. Moreover, Haswell adds FMA support.
#if !defined(SIMSIMD_TARGET_HASWELL) || (SIMSIMD_TARGET_HASWELL && !_SIMSIMD_TARGET_X86)
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define SIMSIMD_TARGET_HASWELL 1
#else
#undef SIMSIMD_TARGET_HASWELL
#define SIMSIMD_TARGET_HASWELL 0
#endif // defined(__AVX2__)
#endif // !defined(SIMSIMD_TARGET_HASWELL) || ...

// Compiling for x86: SIMSIMD_TARGET_SKYLAKE, SIMSIMD_TARGET_ICE, SIMSIMD_TARGET_GENOA,
// SIMSIMD_TARGET_SAPPHIRE, SIMSIMD_TARGET_TURIN, SIMSIMD_TARGET_SIERRA
//
// To list all available macros for x86, take a recent compiler, like GCC 12 and run:
//      gcc-12 -march=sapphirerapids -dM -E - < /dev/null | egrep "SSE|AVX" | sort
// On Arm machines you may want to check for other flags:
//      gcc-12 -march=native -dM -E - < /dev/null | egrep "NEON|SVE|FP16|FMA" | sort
#if !defined(SIMSIMD_TARGET_SKYLAKE) || (SIMSIMD_TARGET_SKYLAKE && !_SIMSIMD_TARGET_X86)
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && \
    defined(__AVX512BW__)
#define SIMSIMD_TARGET_SKYLAKE 1
#else
#undef SIMSIMD_TARGET_SKYLAKE
#define SIMSIMD_TARGET_SKYLAKE 0
#endif
#endif // !defined(SIMSIMD_TARGET_SKYLAKE) || ...
#if !defined(SIMSIMD_TARGET_ICE) || (SIMSIMD_TARGET_ICE && !_SIMSIMD_TARGET_X86)
#if defined(__AVX512VNNI__) && defined(__AVX512IFMA__) && defined(__AVX512BITALG__) && defined(__AVX512VBMI2__) && \
    defined(__AVX512VPOPCNTDQ__)
#define SIMSIMD_TARGET_ICE 1
#else
#undef SIMSIMD_TARGET_ICE
#define SIMSIMD_TARGET_ICE 0
#endif
#endif // !defined(SIMSIMD_TARGET_ICE) || ...
#if !defined(SIMSIMD_TARGET_GENOA) || (SIMSIMD_TARGET_GENOA && !_SIMSIMD_TARGET_X86)
#if defined(__AVX512BF16__)
#define SIMSIMD_TARGET_GENOA 1
#else
#undef SIMSIMD_TARGET_GENOA
#define SIMSIMD_TARGET_GENOA 0
#endif
#endif // !defined(SIMSIMD_TARGET_GENOA) || ...
#if !defined(SIMSIMD_TARGET_SAPPHIRE) || (SIMSIMD_TARGET_SAPPHIRE && !_SIMSIMD_TARGET_X86)
#if defined(__AVX512FP16__)
#define SIMSIMD_TARGET_SAPPHIRE 1
#else
#undef SIMSIMD_TARGET_SAPPHIRE
#define SIMSIMD_TARGET_SAPPHIRE 0
#endif
#endif // !defined(SIMSIMD_TARGET_SAPPHIRE) || ...
#if !defined(SIMSIMD_TARGET_SAPPHIRE_AMX) || (SIMSIMD_TARGET_SAPPHIRE_AMX && !_SIMSIMD_TARGET_X86)
#if defined(__AMX_TILE__) && defined(__AMX_BF16__) && defined(__AMX_INT8__)
#define SIMSIMD_TARGET_SAPPHIRE_AMX 1
#else
#undef SIMSIMD_TARGET_SAPPHIRE_AMX
#define SIMSIMD_TARGET_SAPPHIRE_AMX 0
#endif
#endif // !defined(SIMSIMD_TARGET_SAPPHIRE_AMX) || ...
#if !defined(SIMSIMD_TARGET_TURIN) || (SIMSIMD_TARGET_TURIN && !_SIMSIMD_TARGET_X86)
#if defined(__AVX512VP2INTERSECT__)
#define SIMSIMD_TARGET_TURIN 1
#else
#undef SIMSIMD_TARGET_TURIN
#define SIMSIMD_TARGET_TURIN 0
#endif
#endif // !defined(SIMSIMD_TARGET_TURIN) || ...
#if !defined(SIMSIMD_TARGET_SIERRA) || (SIMSIMD_TARGET_SIERRA && !_SIMSIMD_TARGET_X86)
#if defined(__AVX2_VNNI__)
#define SIMSIMD_TARGET_SIERRA 1
#else
#undef SIMSIMD_TARGET_SIERRA
#define SIMSIMD_TARGET_SIERRA 0
#endif
#endif // !defined(SIMSIMD_TARGET_SIERRA) || ...

#if defined(_MSC_VER)
#include <intrin.h>
#else

#if SIMSIMD_TARGET_NEON
#include <arm_neon.h>
#endif

#if SIMSIMD_TARGET_SVE || SIMSIMD_TARGET_SVE2
#include <arm_sve.h>
#endif

#if SIMSIMD_TARGET_HASWELL || SIMSIMD_TARGET_SKYLAKE || SIMSIMD_TARGET_ICE || SIMSIMD_TARGET_GENOA || \
    SIMSIMD_TARGET_SAPPHIRE || SIMSIMD_TARGET_TURIN
#include <immintrin.h>
#endif

#endif

#if !defined(SIMSIMD_SQRT)
#include <math.h>
#define SIMSIMD_SQRT(x) (sqrt(x))
#endif

#if !defined(SIMSIMD_RSQRT)
#include <math.h>
#define SIMSIMD_RSQRT(x) (1 / SIMSIMD_SQRT(x))
#endif

#if !defined(SIMSIMD_LOG)
#include <math.h>
#define SIMSIMD_LOG(x) (log(x))
#endif

#if !defined(SIMSIMD_TAN)
#include <math.h>
#define SIMSIMD_TAN(x) (tan(x))
#endif

#if !defined(SIMSIMD_FABS)
#include <math.h>
#define SIMSIMD_FABS(x) (fabs(x))
#endif

// Copy 16 bits (2 bytes) from source to destination
#if defined(__GNUC__) || defined(__clang__)
#define SIMSIMD_COPY16(destination_ptr, source_ptr) __builtin_memcpy((destination_ptr), (source_ptr), 2)
#else
#include <string.h> /* fallback for exotic compilers */
#define SIMSIMD_COPY16(destination_ptr, source_ptr) memcpy((destination_ptr), (source_ptr), 2)
#endif

#if !defined(SIMSIMD_F32_DIVISION_EPSILON)
#define SIMSIMD_F32_DIVISION_EPSILON (1e-7)
#endif

#if !defined(SIMSIMD_F16_DIVISION_EPSILON)
#define SIMSIMD_F16_DIVISION_EPSILON (1e-3)
#endif

/**
 *  @brief  The compile-time constant defining the capacity of `simsimd_xd_index_t`.
 *          Matches `PyBUF_MAX_NDIM` by default.
 */
#if !defined(SIMSIMD_NDARRAY_MAX_RANK)
#define SIMSIMD_NDARRAY_MAX_RANK (64)
#endif

/**
 *  @brief  Aligns a variable to a 64-byte boundary using compiler extensions for
 *          compatibility with C 99, as `alignas(64)` is only available in C 11 or C++.
 *
 */
#ifdef _MSC_VER
#define SIMSIMD_ALIGN64 __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
#define SIMSIMD_ALIGN64 __attribute__((aligned(64)))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char simsimd_b8_t;   /// ? Eight boolean values packed in one byte
typedef unsigned char simsimd_i4x2_t; /// ? Two 4-bit signed integers packed in one byte
typedef unsigned char simsimd_e4m3_t; /// ? FP8 E4M3 value encoded into one byte
typedef unsigned char simsimd_e5m2_t; /// ? FP8 E5M2 value encoded into one byte

typedef signed char simsimd_i8_t;
typedef unsigned char simsimd_u8_t;
typedef signed short simsimd_i16_t;
typedef unsigned short simsimd_u16_t;
typedef signed int simsimd_i32_t;
typedef unsigned int simsimd_u32_t;
typedef signed long long simsimd_i64_t;
typedef unsigned long long simsimd_u64_t;

typedef float simsimd_f32_t;
typedef double simsimd_f64_t;

typedef simsimd_u64_t simsimd_size_t;
typedef simsimd_i64_t simsimd_ssize_t;
typedef simsimd_f64_t simsimd_distance_t;

/*  @brief  Half-precision floating-point type.
 *
 *  - GCC or Clang on 64-bit Arm: `__fp16`, may require `-mfp16-format` option.
 *  - GCC or Clang on 64-bit x86: `_Float16`.
 *  - Default: `unsigned short`.
 *
 */
#if !defined(SIMSIMD_NATIVE_F16) || SIMSIMD_NATIVE_F16
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__ARM_ARCH) || defined(__aarch64__)) && \
    (defined(__ARM_FP16_FORMAT_IEEE))
#undef SIMSIMD_NATIVE_F16
#define SIMSIMD_NATIVE_F16 1
typedef __fp16 simsimd_f16_t;
#elif ((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && \
       (defined(__AVX512FP16__)))
typedef _Float16 simsimd_f16_t;
#undef SIMSIMD_NATIVE_F16
#define SIMSIMD_NATIVE_F16 1
#else                                       // Unknown compiler or architecture
#if defined(__GNUC__) || defined(__clang__) // Some compilers don't support warning pragmas
#warning "Unknown compiler or architecture for float16."
#endif
#undef SIMSIMD_NATIVE_F16
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
#undef SIMSIMD_NATIVE_BF16
#define SIMSIMD_NATIVE_BF16 1
typedef __bf16 simsimd_bf16_t;
#elif ((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && \
       (defined(__AVX512BF16__)))
typedef __bfloat16 simsimd_bf16_t;
#undef SIMSIMD_NATIVE_BF16
#define SIMSIMD_NATIVE_BF16 1
#else                                       // Unknown compiler or architecture
#if defined(__GNUC__) || defined(__clang__) // Some compilers don't support warning pragmas
#warning "Unknown compiler or architecture for bfloat16."
#endif
#undef SIMSIMD_NATIVE_BF16
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
#if _SIMSIMD_TARGET_ARM
#if defined(_MSC_VER)
#define simsimd_f16_for_arm_simd_t simsimd_f16_t
#define simsimd_bf16_for_arm_simd_t simsimd_bf16_t
#else
#define simsimd_f16_for_arm_simd_t float16_t
#define simsimd_bf16_for_arm_simd_t bfloat16_t
#endif
#endif

/*
 *  Let's make sure the sizes of the types are as expected.
 *  In C the `_Static_assert` is only available with C11 and later.
 */
#define SIMSIMD_STATIC_ASSERT(cond, msg) typedef char static_assertion_##msg[(cond) ? 1 : -1]
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_b8_t) == 1, simsimd_b8_t_must_be_1_byte);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_i4x2_t) == 1, simsimd_i4x2_t_must_be_1_byte);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_e4m3_t) == 1, simsimd_e4m3_t_must_be_1_byte);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_e5m2_t) == 1, simsimd_e5m2_t_must_be_1_byte);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_i8_t) == 1, simsimd_i8_t_must_be_1_byte);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_u8_t) == 1, simsimd_u8_t_must_be_1_byte);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_i16_t) == 2, simsimd_i16_t_must_be_2_bytes);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_u16_t) == 2, simsimd_u16_t_must_be_2_bytes);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_i32_t) == 4, simsimd_i32_t_must_be_4_bytes);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_u32_t) == 4, simsimd_u32_t_must_be_4_bytes);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_i64_t) == 8, simsimd_i64_t_must_be_8_bytes);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_u64_t) == 8, simsimd_u64_t_must_be_8_bytes);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_f32_t) == 4, simsimd_f32_t_must_be_4_bytes);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_f64_t) == 8, simsimd_f64_t_must_be_8_bytes);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_f16_t) == 2, simsimd_f16_t_must_be_2_bytes);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_bf16_t) == 2, simsimd_bf16_t_must_be_2_bytes);

#define SIMSIMD_DEREFERENCE(x) (*(x))
#define SIMSIMD_EXPORT(x, y) *(y) = x

/** @brief  Convenience type for single-precision floating-point bit manipulation. */
typedef union {
    simsimd_u32_t u;
    simsimd_f32_t f;
} simsimd_fui32_t;

/** @brief  Convenience type for double-precision floating-point bit manipulation. */
typedef union {
    simsimd_u64_t u;
    simsimd_f64_t f;
} simsimd_fui64_t;

/** @brief  Convenience type addressing the real and imaginary parts of a half-precision complex number. */
typedef struct {
    simsimd_f16_t real;
    simsimd_f16_t imag;
} simsimd_f16c_t;

/** @brief  Convenience type addressing the real and imaginary parts of a half-precision brain-float complex number. */
typedef struct {
    simsimd_bf16_t real;
    simsimd_bf16_t imag;
} simsimd_bf16c_t;

/** @brief  Convenience type addressing the real and imaginary parts of a single-precision complex number. */
typedef struct {
    simsimd_f32_t real;
    simsimd_f32_t imag;
} simsimd_f32c_t;

/** @brief  Convenience type addressing the real and imaginary parts of a double-precision complex number. */
typedef struct {
    simsimd_f64_t real;
    simsimd_f64_t imag;
} simsimd_f64c_t;

/** @brief  Small 64-byte memory slice viewable as different types.
 *
 *  On GCC and Clang we use `__transparent_union__` attribute to allow implicit conversions
 *  between the different vector types when passing them as function arguments. The most important side-effect
 *  of this is that the argument of such type is passed to functions using the calling convention of the first
 *  member of the union, which in our case is a register-based calling convention for SIMD types.
 */
typedef union __attribute__((__transparent_union__)) simsimd_b512_vec_t {
#if SIMSIMD_TARGET_SKYLAKE || SIMSIMD_TARGET_ICE || SIMSIMD_TARGET_GENOA || SIMSIMD_TARGET_SAPPHIRE || \
    SIMSIMD_TARGET_TURIN || SIMSIMD_TARGET_SIERRA
    __m512i zmm;
    __m512d zmm_pd;
    __m512 zmm_ps;
#endif
#if SIMSIMD_TARGET_SKYLAKE || SIMSIMD_TARGET_ICE || SIMSIMD_TARGET_GENOA || SIMSIMD_TARGET_SAPPHIRE || \
    SIMSIMD_TARGET_TURIN || SIMSIMD_TARGET_SIERRA || SIMSIMD_TARGET_HASWELL
    __m256i ymms[2];
    __m256d ymms_pd[2];
    __m256 ymms_ps[2];
#endif
#if SIMSIMD_TARGET_SKYLAKE || SIMSIMD_TARGET_ICE || SIMSIMD_TARGET_GENOA || SIMSIMD_TARGET_SAPPHIRE || \
    SIMSIMD_TARGET_TURIN || SIMSIMD_TARGET_SIERRA || SIMSIMD_TARGET_HASWELL
    __m128i xmms[4];
    __m128d xmms_pd[4];
    __m128 xmms_ps[4];
#endif
#if SIMSIMD_TARGET_NEON
    uint8x16_t u8x16s[4];
    uint16x8_t u16x8s[4];
    uint32x4_t u32x4s[4];
    uint64x2_t u64x2s[4];
#endif

    // Unsigned integers
    simsimd_u8_t u8s[64];
    simsimd_u16_t u16s[32];
    simsimd_u32_t u32s[16];
    simsimd_u64_t u64s[8];

    // Signed integers
    simsimd_i8_t i8s[64];
    simsimd_i16_t i16s[32];
    simsimd_i32_t i32s[16];
    simsimd_i64_t i64s[8];

    // Floating-point numbers
    simsimd_f16_t f16s[32];
    simsimd_bf16_t bf16s[32];
    simsimd_f32_t f32s[16];
    simsimd_f64_t f64s[8];
    simsimd_e4m3_t e4m3s[64];
    simsimd_e5m2_t e5m2s[64];

    // Boolean values
    simsimd_b8_t b8s[64];

} simsimd_b512_vec_t;

/**
 *  @brief  Computes `1/sqrt(x)` using the trick from Quake 3,
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
SIMSIMD_INTERNAL simsimd_f32_t simsimd_approximate_inverse_square_root(simsimd_f32_t number) {
    simsimd_fui32_t conv;
    conv.f = number;
    conv.u = 0x5F1FFFF9 - (conv.u >> 1);
    // Refine using a Newton-Raphson step for better accuracy
    conv.f *= 0.703952253f * (2.38924456f - number * conv.f * conv.f);
    return conv.f;
}

/**
 *  @brief  Approximates `sqrt(x)` using the fast inverse square root trick
 *          with adjustments for direct square root approximation.
 *
 *  Similar to `rsqrt` approximation but multiplies by `number` to get `sqrt`.
 *  This technique is useful where `sqrt` approximation is needed in performance-critical code,
 *  though modern hardware provides optimized alternatives.
 */
SIMSIMD_INTERNAL simsimd_f32_t simsimd_approximate_square_root(simsimd_f32_t number) {
    return number * simsimd_approximate_inverse_square_root(number);
}

/**
 *  @brief  Computes `log(x)` using the Mercator series.
 *          The series converges to the natural logarithm for args between -1 and 1.
 *          Published in 1668 in "Logarithmotechnia".
 */
SIMSIMD_INTERNAL simsimd_f32_t simsimd_approximate_log(simsimd_f32_t number) {
    simsimd_f32_t x = number - 1;
    simsimd_f32_t x2 = x * x;
    simsimd_f32_t x3 = x * x * x;
    return x - x2 / 2 + x3 / 3;
}

#define _SIMSIMD_ASSIGN_1_TO_2(x, y) *(y) = *(x)

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
SIMSIMD_INTERNAL void simsimd_f16_to_f32_implementation(simsimd_f16_t const *src, simsimd_f32_t *dest) {
    unsigned short x;
    SIMSIMD_COPY16(&x, src);
    unsigned int exponent = (x & 0x7C00) >> 10;
    unsigned int mantissa = (x & 0x03FF) << 13;
    simsimd_fui32_t mantissa_conv;
    mantissa_conv.f = (float)mantissa;
    unsigned int v = (mantissa_conv.u) >> 23;
    simsimd_fui32_t conv;
    conv.u = (x & 0x8000) << 16 | (exponent != 0) * ((exponent + 112) << 23 | mantissa) |
             ((exponent == 0) & (mantissa != 0)) * ((v - 37) << 23 | ((mantissa << (150 - v)) & 0x007FE000));
    *dest = conv.f;
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
SIMSIMD_INTERNAL void simsimd_f32_to_f16_implementation(simsimd_f32_t const *src, simsimd_f16_t *dest) {
    simsimd_fui32_t conv;
    conv.f = *src;
    unsigned int b = conv.u + 0x00001000;
    unsigned int e = (b & 0x7F800000) >> 23;
    unsigned int m = b & 0x007FFFFF;
    unsigned short result = ((b & 0x80000000) >> 16) | (e > 112) * ((((e - 112) << 10) & 0x7C00) | (m >> 13)) |
                            ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
                            ((e > 143) * 0x7FFF);
    SIMSIMD_COPY16(dest, &result);
}

/**
 *  @brief  For compilers that don't natively support the `__bf16` type,
 *          upcasts contents into a more conventional `float`.
 *
 *  https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c/55254307#55254307
 *  https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
 */
SIMSIMD_INTERNAL void simsimd_bf16_to_f32_implementation(simsimd_bf16_t const *src, simsimd_f32_t *dest) {
    unsigned short x;
    SIMSIMD_COPY16(&x, src);
    simsimd_fui32_t conv;
    conv.u = x << 16; // Zero extends the mantissa
    *dest = conv.f;
}

/**
 *  @brief  Compresses a `float` to a `bf16` representation.
 *
 *  https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c/55254307#55254307
 *  https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
 */
SIMSIMD_INTERNAL void simsimd_f32_to_bf16_implementation(simsimd_f32_t const *src, simsimd_bf16_t *dest) {
    simsimd_fui32_t conv;
    conv.f = *src;
    conv.u += 0x8000; // Rounding is optional
    conv.u >>= 16;
    // Use an intermediate variable to ensure correct behavior on big-endian systems.
    // Copying directly from `&conv.u` would copy the wrong bytes on big-endian,
    // since the lower 16 bits are at offset 2, not offset 0.
    unsigned short result = (unsigned short)conv.u;
    SIMSIMD_COPY16(dest, &result);
}

SIMSIMD_INTERNAL void simsimd_e4m3_to_f32_implementation(simsimd_e4m3_t const *src, simsimd_f32_t *dest) {
    simsimd_u8_t raw = *src;
    simsimd_u32_t sign = (simsimd_u32_t)(raw & 0x80) << 24;
    simsimd_u32_t exponent = (raw >> 3) & 0x0Fu;
    simsimd_u32_t mantissa = raw & 0x07u;
    simsimd_fui32_t conv;

    if (exponent == 0) {
        if (mantissa == 0) {
            conv.u = sign;
            *dest = conv.f;
            return;
        }
        simsimd_f32_t value = (simsimd_f32_t)mantissa * (1.0f / 512.0f);
        *dest = sign ? -value : value;
        return;
    }
    if (exponent == 0x0Fu) {
        if (mantissa == 0) { conv.u = sign | 0x7F800000u; }
        else { conv.u = sign | 0x7FC00000u; }
        *dest = conv.f;
        return;
    }

    simsimd_u32_t f32_exponent = (exponent + 120u) << 23;
    simsimd_u32_t f32_mantissa = mantissa << 20;
    conv.u = sign | f32_exponent | f32_mantissa;
    *dest = conv.f;
}

SIMSIMD_INTERNAL void simsimd_f32_to_e4m3_implementation(simsimd_f32_t const *src, simsimd_e4m3_t *dest) {
    simsimd_f32_t x = *src;
    simsimd_fui32_t conv;
    conv.f = x;
    simsimd_u32_t sign_bit = conv.u >> 31;
    simsimd_u32_t abs_bits = conv.u & 0x7FFFFFFFu;
    simsimd_u8_t sign = (simsimd_u8_t)(sign_bit << 7);

    if (abs_bits >= 0x7F800000u) {
        simsimd_u8_t mant = (abs_bits > 0x7F800000u) ? 0x01u : 0x00u;
        *dest = (simsimd_e4m3_t)(sign | 0x78u | mant);
        return;
    }

    if (abs_bits == 0) {
        *dest = (simsimd_e4m3_t)sign;
        return;
    }

    simsimd_f32_t abs_x = sign_bit ? -x : x;

    if (abs_x < (1.0f / 512.0f)) {
        *dest = (simsimd_e4m3_t)sign;
        return;
    }

    if (abs_x < (1.0f / 64.0f)) {
        simsimd_f32_t scaled = abs_x * 512.0f;
        simsimd_i32_t mant = (simsimd_i32_t)scaled;
        simsimd_f32_t frac = scaled - (simsimd_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        if (mant > 7) { mant = 7; }
        if (mant == 0) { *dest = (simsimd_e4m3_t)sign; }
        else { *dest = (simsimd_e4m3_t)(sign | (simsimd_u8_t)mant); }
        return;
    }

    simsimd_i32_t exp = (simsimd_i32_t)((abs_bits >> 23) & 0xFFu) - 127;
    simsimd_u32_t mantissa = abs_bits & 0x7FFFFFu;
    simsimd_u32_t significand = (1u << 23) | mantissa;
    simsimd_i32_t shift = 23 - 3;
    simsimd_u32_t remainder_mask = (1u << shift) - 1;
    simsimd_u32_t remainder = significand & remainder_mask;
    simsimd_u32_t halfway = 1u << (shift - 1);
    simsimd_u32_t significand_rounded = significand >> shift;
    if (remainder > halfway || (remainder == halfway && (significand_rounded & 1))) { ++significand_rounded; }
    if (significand_rounded == (1u << (3 + 1))) {
        significand_rounded >>= 1;
        ++exp;
    }
    if (exp > 7) {
        *dest = (simsimd_e4m3_t)(sign | 0x78u);
        return;
    }
    if (exp < -6) {
        simsimd_f32_t scaled = abs_x * 512.0f;
        simsimd_i32_t mant = (simsimd_i32_t)scaled;
        simsimd_f32_t frac = scaled - (simsimd_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        if (mant > 7) { mant = 7; }
        if (mant == 0) { *dest = (simsimd_e4m3_t)sign; }
        else { *dest = (simsimd_e4m3_t)(sign | (simsimd_u8_t)mant); }
        return;
    }

    simsimd_u8_t exp_field = (simsimd_u8_t)(exp + 7);
    simsimd_u8_t mant_field = (simsimd_u8_t)(significand_rounded & 0x07u);
    *dest = (simsimd_e4m3_t)(sign | (exp_field << 3) | mant_field);
}

SIMSIMD_INTERNAL void simsimd_e5m2_to_f32_implementation(simsimd_e5m2_t const *src, simsimd_f32_t *dest) {
    simsimd_u8_t raw = *src;
    simsimd_u32_t sign = (simsimd_u32_t)(raw & 0x80) << 24;
    simsimd_u32_t exponent = (raw >> 2) & 0x1Fu;
    simsimd_u32_t mantissa = raw & 0x03u;
    simsimd_fui32_t conv;

    if (exponent == 0) {
        if (mantissa == 0) {
            conv.u = sign;
            *dest = conv.f;
            return;
        }
        simsimd_f32_t value = (simsimd_f32_t)mantissa * (1.0f / 65536.0f);
        *dest = sign ? -value : value;
        return;
    }
    if (exponent == 0x1Fu) {
        if (mantissa == 0) { conv.u = sign | 0x7F800000u; }
        else { conv.u = sign | 0x7FC00000u; }
        *dest = conv.f;
        return;
    }

    simsimd_u32_t f32_exponent = (exponent + 112u) << 23;
    simsimd_u32_t f32_mantissa = mantissa << 21;
    conv.u = sign | f32_exponent | f32_mantissa;
    *dest = conv.f;
}

SIMSIMD_INTERNAL void simsimd_f32_to_e5m2_implementation(simsimd_f32_t const *src, simsimd_e5m2_t *dest) {
    simsimd_f32_t x = *src;
    simsimd_fui32_t conv;
    conv.f = x;
    simsimd_u32_t sign_bit = conv.u >> 31;
    simsimd_u32_t abs_bits = conv.u & 0x7FFFFFFFu;
    simsimd_u8_t sign = (simsimd_u8_t)(sign_bit << 7);

    if (abs_bits >= 0x7F800000u) {
        simsimd_u8_t mant = (abs_bits > 0x7F800000u) ? 0x01u : 0x00u;
        *dest = (simsimd_e5m2_t)(sign | 0x7Cu | mant);
        return;
    }

    if (abs_bits == 0) {
        *dest = (simsimd_e5m2_t)sign;
        return;
    }

    simsimd_f32_t abs_x = sign_bit ? -x : x;

    if (abs_x < (1.0f / 65536.0f)) {
        *dest = (simsimd_e5m2_t)sign;
        return;
    }

    if (abs_x < (1.0f / 16384.0f)) {
        simsimd_f32_t scaled = abs_x * 65536.0f;
        simsimd_i32_t mant = (simsimd_i32_t)scaled;
        simsimd_f32_t frac = scaled - (simsimd_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        if (mant > 3) { mant = 3; }
        if (mant == 0) { *dest = (simsimd_e5m2_t)sign; }
        else { *dest = (simsimd_e5m2_t)(sign | (simsimd_u8_t)mant); }
        return;
    }

    simsimd_i32_t exp = (simsimd_i32_t)((abs_bits >> 23) & 0xFFu) - 127;
    simsimd_u32_t mantissa = abs_bits & 0x7FFFFFu;
    simsimd_u32_t significand = (1u << 23) | mantissa;
    simsimd_i32_t shift = 23 - 2;
    simsimd_u32_t remainder_mask = (1u << shift) - 1;
    simsimd_u32_t remainder = significand & remainder_mask;
    simsimd_u32_t halfway = 1u << (shift - 1);
    simsimd_u32_t significand_rounded = significand >> shift;
    if (remainder > halfway || (remainder == halfway && (significand_rounded & 1))) { ++significand_rounded; }
    if (significand_rounded == (1u << (2 + 1))) {
        significand_rounded >>= 1;
        ++exp;
    }
    if (exp > 15) {
        *dest = (simsimd_e5m2_t)(sign | 0x7Cu);
        return;
    }
    if (exp < -14) {
        simsimd_f32_t scaled = abs_x * 65536.0f;
        simsimd_i32_t mant = (simsimd_i32_t)scaled;
        simsimd_f32_t frac = scaled - (simsimd_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        if (mant > 3) { mant = 3; }
        if (mant == 0) { *dest = (simsimd_e5m2_t)sign; }
        else { *dest = (simsimd_e5m2_t)(sign | (simsimd_u8_t)mant); }
        return;
    }

    simsimd_u8_t exp_field = (simsimd_u8_t)(exp + 15);
    simsimd_u8_t mant_field = (simsimd_u8_t)(significand_rounded & 0x03u);
    *dest = (simsimd_e5m2_t)(sign | (exp_field << 2) | mant_field);
}

// Forward declarations for conversion functions used in helper functions and wrapper macros
#if SIMSIMD_DYNAMIC_DISPATCH
SIMSIMD_DYNAMIC void simsimd_f16_to_f32(simsimd_f16_t const *src, simsimd_f32_t *dest);
SIMSIMD_DYNAMIC void simsimd_bf16_to_f32(simsimd_bf16_t const *src, simsimd_f32_t *dest);
SIMSIMD_DYNAMIC void simsimd_f32_to_f16(simsimd_f32_t const *src, simsimd_f16_t *dest);
SIMSIMD_DYNAMIC void simsimd_f32_to_bf16(simsimd_f32_t const *src, simsimd_bf16_t *dest);
SIMSIMD_DYNAMIC void simsimd_e4m3_to_f32(simsimd_e4m3_t const *src, simsimd_f32_t *dest);
SIMSIMD_DYNAMIC void simsimd_f32_to_e4m3(simsimd_f32_t const *src, simsimd_e4m3_t *dest);
SIMSIMD_DYNAMIC void simsimd_e5m2_to_f32(simsimd_e5m2_t const *src, simsimd_f32_t *dest);
SIMSIMD_DYNAMIC void simsimd_f32_to_e5m2(simsimd_f32_t const *src, simsimd_e5m2_t *dest);
#else
SIMSIMD_PUBLIC void simsimd_f16_to_f32(simsimd_f16_t const *src, simsimd_f32_t *dest);
SIMSIMD_PUBLIC void simsimd_bf16_to_f32(simsimd_bf16_t const *src, simsimd_f32_t *dest);
SIMSIMD_PUBLIC void simsimd_f32_to_f16(simsimd_f32_t const *src, simsimd_f16_t *dest);
SIMSIMD_PUBLIC void simsimd_f32_to_bf16(simsimd_f32_t const *src, simsimd_bf16_t *dest);
SIMSIMD_PUBLIC void simsimd_e4m3_to_f32(simsimd_e4m3_t const *src, simsimd_f32_t *dest);
SIMSIMD_PUBLIC void simsimd_f32_to_e4m3(simsimd_f32_t const *src, simsimd_e4m3_t *dest);
SIMSIMD_PUBLIC void simsimd_e5m2_to_f32(simsimd_e5m2_t const *src, simsimd_f32_t *dest);
SIMSIMD_PUBLIC void simsimd_f32_to_e5m2(simsimd_f32_t const *src, simsimd_e5m2_t *dest);
#endif

/**
 *  @brief  Returns the value of the half-precision floating-point number,
 *          potentially decompressed into single-precision.
 *
 *  The underlying conversion functions use pure two-pointer style, but these
 *  macros provide a convenient expression-returning interface for compatibility.
 */
#if !defined(SIMSIMD_F16_TO_F32)
#if SIMSIMD_NATIVE_F16
#define SIMSIMD_F16_TO_F32(x) (SIMSIMD_DEREFERENCE(x))
#define SIMSIMD_F32_TO_F16(x, y) (SIMSIMD_EXPORT(x, y))
#else
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_f16_to_f32_wrapper(simsimd_f16_t const *src) {
    simsimd_f32_t dest;
    simsimd_f16_to_f32(src, &dest);
    return dest;
}
SIMSIMD_INTERNAL void _simsimd_f32_to_f16_wrapper(simsimd_f32_t src, simsimd_f16_t *dest) {
    simsimd_f32_to_f16(&src, dest);
}
#define SIMSIMD_F16_TO_F32(x) (_simsimd_f16_to_f32_wrapper(x))
#define SIMSIMD_F32_TO_F16(x, y) (_simsimd_f32_to_f16_wrapper(x, y))
#endif
#endif

/**
 *  @brief  Returns the value of the half-precision brain floating-point number,
 *          potentially decompressed into single-precision.
 *
 *  The underlying conversion functions use pure two-pointer style, but these
 *  macros provide a convenient expression-returning interface for compatibility.
 */
#if !defined(SIMSIMD_BF16_TO_F32)
#if SIMSIMD_NATIVE_BF16
#define SIMSIMD_BF16_TO_F32(x) (SIMSIMD_DEREFERENCE(x))
#define SIMSIMD_F32_TO_BF16(x, y) (SIMSIMD_EXPORT(x, y))
#else
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_bf16_to_f32_wrapper(simsimd_bf16_t const *src) {
    simsimd_f32_t dest;
    simsimd_bf16_to_f32(src, &dest);
    return dest;
}
SIMSIMD_INTERNAL void _simsimd_f32_to_bf16_wrapper(simsimd_f32_t src, simsimd_bf16_t *dest) {
    simsimd_f32_to_bf16(&src, dest);
}
#define SIMSIMD_BF16_TO_F32(x) (_simsimd_bf16_to_f32_wrapper(x))
#define SIMSIMD_F32_TO_BF16(x, y) (_simsimd_f32_to_bf16_wrapper(x, y))
#endif
#endif

#if !defined(SIMSIMD_F32_TO_I8)
#define SIMSIMD_F32_TO_I8(x, y) \
    *(y) = (simsimd_i8_t)((x) > 127 ? 127 : ((x) < -128 ? -128 : (int)((x) + ((x) < 0 ? -0.5f : 0.5f))))
#endif
#if !defined(SIMSIMD_F32_TO_U8)
#define SIMSIMD_F32_TO_U8(x, y) \
    *(y) = (simsimd_u8_t)((x) > 255 ? 255 : ((x) < 0 ? 0 : (int)((x) + ((x) < 0 ? -0.5f : 0.5f))))
#endif
#if !defined(SIMSIMD_F64_TO_I8)
#define SIMSIMD_F64_TO_I8(x, y) \
    *(y) = (simsimd_i8_t)((x) > 127 ? 127 : ((x) < -128 ? -128 : (int)((x) + ((x) < 0 ? -0.5 : 0.5))))
#endif
#if !defined(SIMSIMD_F64_TO_U8)
#define SIMSIMD_F64_TO_U8(x, y) \
    *(y) = (simsimd_u8_t)((x) > 255 ? 255 : ((x) < 0 ? 0 : (int)((x) + ((x) < 0 ? -0.5 : 0.5))))
#endif

SIMSIMD_INTERNAL void _simsimd_f16_to_f64(simsimd_f16_t const *x, simsimd_f64_t *y) {
    simsimd_f32_t f32;
    simsimd_f16_to_f32(x, &f32);
    *y = (simsimd_f64_t)f32;
}
SIMSIMD_INTERNAL void _simsimd_f64_to_f16(simsimd_f64_t const *x, simsimd_f16_t *y) {
    simsimd_f32_t f32 = (simsimd_f32_t)*x;
    simsimd_f32_to_f16(&f32, y);
}
SIMSIMD_INTERNAL void _simsimd_bf16_to_f64(simsimd_bf16_t const *x, simsimd_f64_t *y) {
    simsimd_f32_t f32;
    simsimd_bf16_to_f32(x, &f32);
    *y = (simsimd_f64_t)f32;
}
SIMSIMD_INTERNAL void _simsimd_f64_to_bf16(simsimd_f64_t const *x, simsimd_bf16_t *y) {
    simsimd_f32_t f32 = (simsimd_f32_t)*x;
    simsimd_f32_to_bf16(&f32, y);
}

/*  Convert floating pointer numbers to integers, clamping them to the range of signed
 *  and unsigned low-resolution integers, and rounding them to the nearest integer.
 *
 *  In C++ the analogous solution with STL could be: `*y = std::clamp(std::round(*x), -128, 127)`.
 *  In C, using the standard library: `*x = fminf(fmaxf(roundf(*x), -128), 127)`.
 */
SIMSIMD_INTERNAL void _simsimd_f32_to_i8(simsimd_f32_t const *x, simsimd_i8_t *y) {
    *y = (simsimd_i8_t)(*x > 127 ? 127 : (*x < -128 ? -128 : (int)(*x + (*x < 0 ? -0.5f : 0.5f))));
}

SIMSIMD_INTERNAL void _simsimd_f32_to_u8(simsimd_f32_t const *x, simsimd_u8_t *y) {
    *y = (simsimd_u8_t)(*x > 255 ? 255 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5f : 0.5f))));
}

SIMSIMD_INTERNAL void _simsimd_f32_to_i16(simsimd_f32_t const *x, simsimd_i16_t *y) {
    *y = (simsimd_i16_t)(*x > 32767 ? 32767 : (*x < -32768 ? -32768 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

SIMSIMD_INTERNAL void _simsimd_f32_to_u16(simsimd_f32_t const *x, simsimd_u16_t *y) {
    *y = (simsimd_u16_t)(*x > 65535 ? 65535 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

SIMSIMD_INTERNAL void _simsimd_f64_to_i8(simsimd_f64_t const *x, simsimd_i8_t *y) {
    *y = (simsimd_i8_t)(*x > 127 ? 127 : (*x < -128 ? -128 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

SIMSIMD_INTERNAL void _simsimd_f64_to_u8(simsimd_f64_t const *x, simsimd_u8_t *y) {
    *y = (simsimd_u8_t)(*x > 255 ? 255 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

SIMSIMD_INTERNAL void _simsimd_f64_to_i16(simsimd_f64_t const *x, simsimd_i16_t *y) {
    *y = (simsimd_i16_t)(*x > 32767 ? 32767 : (*x < -32768 ? -32768 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

SIMSIMD_INTERNAL void _simsimd_f64_to_u16(simsimd_f64_t const *x, simsimd_u16_t *y) {
    *y = (simsimd_u16_t)(*x > 65535 ? 65535 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

SIMSIMD_INTERNAL void _simsimd_f64_to_i32(simsimd_f64_t const *x, simsimd_i32_t *y) {
    *y = (simsimd_i32_t)(*x > 2147483647 ? 2147483647
                                         : (*x < -2147483648 ? -2147483648 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

SIMSIMD_INTERNAL void _simsimd_f64_to_u32(simsimd_f64_t const *x, simsimd_u32_t *y) {
    *y = (simsimd_u32_t)(*x > 4294967295 ? 4294967295 : (*x < 0 ? 0 : (unsigned int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

SIMSIMD_INTERNAL void _simsimd_f64_to_i64(simsimd_f64_t const *x, simsimd_i64_t *y) {
    *y = (simsimd_i64_t)(*x > 9223372036854775807.0
                             ? 9223372036854775807ll
                             : (*x < -9223372036854775808.0 ? (-9223372036854775807ll - 1ll)
                                                            : (long long)(*x + (*x < 0 ? -0.5 : 0.5))));
}

SIMSIMD_INTERNAL void _simsimd_f64_to_u64(simsimd_f64_t const *x, simsimd_u64_t *y) {
    *y = (simsimd_u64_t)(*x > 18446744073709551615.0 ? 18446744073709551615ull
                                                     : (*x < 0 ? 0 : (unsigned long long)(*x + (*x < 0 ? -0.5 : 0.5))));
}

SIMSIMD_INTERNAL void _simsimd_i64_to_i8(simsimd_i64_t const *x, simsimd_i8_t *y) {
    *y = (simsimd_i8_t)(*x > 127ll ? 127ll : (*x < -128ll ? -128ll : *x));
}

SIMSIMD_INTERNAL void _simsimd_i64_to_u8(simsimd_i64_t const *x, simsimd_u8_t *y) {
    *y = (simsimd_u8_t)(*x > 255ll ? 255ll : (*x < 0ll ? 0ll : *x));
}

SIMSIMD_INTERNAL void _simsimd_i64_to_i16(simsimd_i64_t const *x, simsimd_i16_t *y) {
    *y = (simsimd_i16_t)(*x > 32767ll ? 32767ll : (*x < -32768ll ? -32768ll : *x));
}

SIMSIMD_INTERNAL void _simsimd_i64_to_u16(simsimd_i64_t const *x, simsimd_u16_t *y) {
    *y = (simsimd_u16_t)(*x > 65535ll ? 65535ll : (*x < 0ll ? 0ll : *x));
}

SIMSIMD_INTERNAL void _simsimd_i64_to_i32(simsimd_i64_t const *x, simsimd_i32_t *y) {
    *y = (simsimd_i32_t)(*x > 2147483647ll ? 2147483647ll : (*x < -2147483648ll ? -2147483648ll : *x));
}

SIMSIMD_INTERNAL void _simsimd_i64_to_u32(simsimd_i64_t const *x, simsimd_u32_t *y) {
    *y = (simsimd_u32_t)(*x > 4294967295ll ? 4294967295ll : (*x < 0ll ? 0ll : *x));
}

SIMSIMD_INTERNAL void _simsimd_u64_to_i8(simsimd_u64_t const *x, simsimd_i8_t *y) {
    *y = (simsimd_i8_t)(*x > 127ull ? 127ull : *x);
}

SIMSIMD_INTERNAL void _simsimd_u64_to_u8(simsimd_u64_t const *x, simsimd_u8_t *y) {
    *y = (simsimd_u8_t)(*x > 255ull ? 255ull : *x);
}

SIMSIMD_INTERNAL void _simsimd_u64_to_i16(simsimd_u64_t const *x, simsimd_i16_t *y) {
    *y = (simsimd_i16_t)(*x > 32767ull ? 32767ull : *x);
}

SIMSIMD_INTERNAL void _simsimd_u64_to_u16(simsimd_u64_t const *x, simsimd_u16_t *y) {
    *y = (simsimd_u16_t)(*x > 65535ull ? 65535ull : *x);
}

SIMSIMD_INTERNAL void _simsimd_u64_to_i32(simsimd_u64_t const *x, simsimd_i32_t *y) {
    *y = (simsimd_i32_t)(*x > 2147483647ull ? 2147483647ull : *x);
}

SIMSIMD_INTERNAL void _simsimd_u64_to_u32(simsimd_u64_t const *x, simsimd_u32_t *y) {
    *y = (simsimd_u32_t)(*x > 4294967295ull ? 4294967295ull : *x);
}

SIMSIMD_INTERNAL void _simsimd_f64_to_f32(simsimd_f64_t const *x, simsimd_f32_t *y) { *y = (simsimd_f32_t)*x; }
SIMSIMD_INTERNAL void _simsimd_u64_to_f32(simsimd_u64_t const *x, simsimd_f32_t *y) { *y = (simsimd_f32_t)*x; }
SIMSIMD_INTERNAL void _simsimd_i64_to_f32(simsimd_i64_t const *x, simsimd_f32_t *y) { *y = (simsimd_f32_t)*x; }

SIMSIMD_INTERNAL void _simsimd_f32_to_f64(simsimd_f32_t const *x, simsimd_f64_t *y) { *y = (simsimd_f64_t)*x; }

SIMSIMD_INTERNAL void _simsimd_f64_to_f64(simsimd_f64_t const *x, simsimd_f64_t *y) { *y = *x; }

SIMSIMD_INTERNAL void _simsimd_i8_to_f64(simsimd_i8_t const *x, simsimd_f64_t *y) { *y = (simsimd_f64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_i16_to_f64(simsimd_i16_t const *x, simsimd_f64_t *y) { *y = (simsimd_f64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_i32_to_f64(simsimd_i32_t const *x, simsimd_f64_t *y) { *y = (simsimd_f64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_i64_to_f64(simsimd_i64_t const *x, simsimd_f64_t *y) { *y = (simsimd_f64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_u8_to_f64(simsimd_u8_t const *x, simsimd_f64_t *y) { *y = (simsimd_f64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_u16_to_f64(simsimd_u16_t const *x, simsimd_f64_t *y) { *y = (simsimd_f64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_u32_to_f64(simsimd_u32_t const *x, simsimd_f64_t *y) { *y = (simsimd_f64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_u64_to_f64(simsimd_u64_t const *x, simsimd_f64_t *y) { *y = (simsimd_f64_t)*x; }

SIMSIMD_INTERNAL void _simsimd_i8_to_i64(simsimd_i8_t const *x, simsimd_i64_t *y) { *y = (simsimd_i64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_i16_to_i64(simsimd_i16_t const *x, simsimd_i64_t *y) { *y = (simsimd_i64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_i32_to_i64(simsimd_i32_t const *x, simsimd_i64_t *y) { *y = (simsimd_i64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_i64_to_i64(simsimd_i64_t const *x, simsimd_i64_t *y) { *y = *x; }
SIMSIMD_INTERNAL void _simsimd_u8_to_i64(simsimd_u8_t const *x, simsimd_i64_t *y) { *y = (simsimd_i64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_u16_to_i64(simsimd_u16_t const *x, simsimd_i64_t *y) { *y = (simsimd_i64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_u32_to_i64(simsimd_u32_t const *x, simsimd_i64_t *y) { *y = (simsimd_i64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_u64_to_i64(simsimd_u64_t const *x, simsimd_i64_t *y) {
    *y = (simsimd_i64_t)(*x >= 9223372036854775807ull ? 9223372036854775807ll : *x);
}

SIMSIMD_INTERNAL void _simsimd_i8_to_u64(simsimd_i8_t const *x, simsimd_u64_t *y) {
    *y = (simsimd_u64_t)(*x < 0 ? 0 : *x);
}
SIMSIMD_INTERNAL void _simsimd_i16_to_u64(simsimd_i16_t const *x, simsimd_u64_t *y) {
    *y = (simsimd_u64_t)(*x < 0 ? 0 : *x);
}
SIMSIMD_INTERNAL void _simsimd_i32_to_u64(simsimd_i32_t const *x, simsimd_u64_t *y) {
    *y = (simsimd_u64_t)(*x < 0 ? 0 : *x);
}
SIMSIMD_INTERNAL void _simsimd_i64_to_u64(simsimd_i64_t const *x, simsimd_u64_t *y) {
    *y = (simsimd_u64_t)(*x < 0 ? 0 : *x);
}
SIMSIMD_INTERNAL void _simsimd_u8_to_u64(simsimd_u8_t const *x, simsimd_u64_t *y) { *y = (simsimd_u64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_u16_to_u64(simsimd_u16_t const *x, simsimd_u64_t *y) { *y = (simsimd_u64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_u32_to_u64(simsimd_u32_t const *x, simsimd_u64_t *y) { *y = (simsimd_u64_t)*x; }
SIMSIMD_INTERNAL void _simsimd_u64_to_u64(simsimd_u64_t const *x, simsimd_u64_t *y) { *y = *x; }

SIMSIMD_INTERNAL void _simsimd_i64_to_f16(simsimd_i64_t const *x, simsimd_f16_t *y) {
    simsimd_f32_t f32 = (simsimd_f32_t)*x;
    simsimd_f32_to_f16(&f32, y);
}
SIMSIMD_INTERNAL void _simsimd_i64_to_bf16(simsimd_i64_t const *x, simsimd_bf16_t *y) {
    simsimd_f32_t f32 = (simsimd_f32_t)*x;
    simsimd_f32_to_bf16(&f32, y);
}
SIMSIMD_INTERNAL void _simsimd_u64_to_f16(simsimd_u64_t const *x, simsimd_f16_t *y) {
    simsimd_f32_t f32 = (simsimd_f32_t)*x;
    simsimd_f32_to_f16(&f32, y);
}
SIMSIMD_INTERNAL void _simsimd_u64_to_bf16(simsimd_u64_t const *x, simsimd_bf16_t *y) {
    simsimd_f32_t f32 = (simsimd_f32_t)*x;
    simsimd_f32_to_bf16(&f32, y);
}

/**
 *  @brief  Helper structure for implementing strided matrix row lookups, with @b single-byte-level pointer math.
 */
SIMSIMD_INTERNAL void *_simsimd_advance_by_bytes(void *ptr, simsimd_size_t bytes) {
    return (void *)((simsimd_u8_t *)ptr + bytes);
}

/**
 *  @brief  Divide and round up to the nearest integer.
 */
SIMSIMD_INTERNAL simsimd_size_t _simsimd_divide_ceil(simsimd_size_t dividend, simsimd_size_t divisor) {
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
 *  For flexibility, the API is decoupled from from the `simsimd_xd_index_t` structure, and
 *  can be used on any-rank tensors, independent of the `SIMSIMD_NDARRAY_MAX_RANK` constant.
 */
SIMSIMD_PUBLIC int simsimd_xd_index_next(                                               //
    simsimd_size_t const *extents, simsimd_ssize_t const *strides, simsimd_size_t rank, //
    simsimd_size_t *coordinates, simsimd_ssize_t *byte_offset) {
    // Start from last dimension and move backward
    for (simsimd_size_t i = rank; i-- > 0;) {
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
SIMSIMD_PUBLIC int simsimd_xd_index_linearize(                                          //
    simsimd_size_t const *extents, simsimd_ssize_t const *strides, simsimd_size_t rank, //
    simsimd_size_t const *coordinates, simsimd_ssize_t *byte_offset) {

    simsimd_ssize_t result = 0;
    for (simsimd_size_t i = 0; i < rank; i++) {
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
typedef struct simsimd_xd_index_t {
    simsimd_size_t coordinates[SIMSIMD_NDARRAY_MAX_RANK]; // Coordinate offsets along each dimension
    simsimd_ssize_t byte_offset;                          // Byte offset of the current element
} simsimd_xd_index_t;

SIMSIMD_PUBLIC void simsimd_xd_index_init(simsimd_xd_index_t *xd_index) {
    for (simsimd_size_t i = 0; i < SIMSIMD_NDARRAY_MAX_RANK; i++) xd_index->coordinates[i] = 0;
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
 *  Most SimSIMD algorithms don't work with the entire structure, but expect the fields to be passed separately.
 *  It would also require storing the @b start-pointer and the @b datatype/item-size separately, as it's not
 *  stored inside the structure.
 */
typedef struct simsimd_xd_span_t {
    simsimd_size_t extents[SIMSIMD_NDARRAY_MAX_RANK];  /// Number of elements along each dimension
    simsimd_ssize_t strides[SIMSIMD_NDARRAY_MAX_RANK]; /// Strides of the tensor in bytes
    simsimd_size_t rank;                               /// Number of dimensions in the tensor
} simsimd_xd_span_t;

SIMSIMD_PUBLIC void simsimd_xd_span_init(simsimd_xd_span_t *xd_span) {
    for (simsimd_size_t i = 0; i < SIMSIMD_NDARRAY_MAX_RANK; i++) xd_span->extents[i] = 0, xd_span->strides[i] = 0;
    xd_span->rank = 0;
}

SIMSIMD_INTERNAL simsimd_u32_t simsimd_u32_rol(simsimd_u32_t x, int n) { return (x << n) | (x >> (32 - n)); }
SIMSIMD_INTERNAL simsimd_u16_t simsimd_u16_rol(simsimd_u16_t x, int n) { return (x << n) | (x >> (16 - n)); }
SIMSIMD_INTERNAL simsimd_u8_t simsimd_u8_rol(simsimd_u8_t x, int n) { return (x << n) | (x >> (8 - n)); }
SIMSIMD_INTERNAL simsimd_u32_t simsimd_u32_ror(simsimd_u32_t x, int n) { return (x >> n) | (x << (32 - n)); }
SIMSIMD_INTERNAL simsimd_u16_t simsimd_u16_ror(simsimd_u16_t x, int n) { return (x >> n) | (x << (16 - n)); }
SIMSIMD_INTERNAL simsimd_u8_t simsimd_u8_ror(simsimd_u8_t x, int n) { return (x >> n) | (x << (8 - n)); }

SIMSIMD_INTERNAL void _simsimd_u8_sadd(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_u8_t *r) {
    simsimd_u16_t result = (simsimd_u16_t)*a + (simsimd_u16_t)*b;
    *r = (result > 255u) ? (simsimd_u8_t)255u : (simsimd_u8_t)result;
}
SIMSIMD_INTERNAL void _simsimd_u16_sadd(simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_u16_t *r) {
    simsimd_u32_t result = (simsimd_u32_t)*a + (simsimd_u32_t)*b;
    *r = (result > 65535u) ? (simsimd_u16_t)65535u : (simsimd_u16_t)result;
}
SIMSIMD_INTERNAL void _simsimd_u32_sadd(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_u32_t *r) {
    simsimd_u64_t result = (simsimd_u64_t)*a + (simsimd_u64_t)*b;
    *r = (result > 4294967295u) ? (simsimd_u32_t)4294967295u : (simsimd_u32_t)result;
}
SIMSIMD_INTERNAL void _simsimd_u64_sadd(simsimd_u64_t const *a, simsimd_u64_t const *b, simsimd_u64_t *r) {
    *r = (*a + *b < *a) ? 18446744073709551615ull : (*a + *b);
}
SIMSIMD_INTERNAL void _simsimd_i8_sadd(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_i8_t *r) {
    simsimd_i16_t result = (simsimd_i16_t)*a + (simsimd_i16_t)*b;
    *r = (result > 127) ? 127 : (result < -128 ? -128 : result);
}
SIMSIMD_INTERNAL void _simsimd_i16_sadd(simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_i16_t *r) {
    simsimd_i32_t result = (simsimd_i32_t)*a + (simsimd_i32_t)*b;
    *r = (result > 32767) ? 32767 : (result < -32768 ? -32768 : result);
}
SIMSIMD_INTERNAL void _simsimd_i32_sadd(simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_i32_t *r) {
    simsimd_i64_t result = (simsimd_i64_t)*a + (simsimd_i64_t)*b;
    *r = (result > 2147483647ll) ? 2147483647ll : (result < -2147483648ll ? -2147483648ll : (simsimd_i32_t)result);
}
SIMSIMD_INTERNAL void _simsimd_i64_sadd(simsimd_i64_t const *a, simsimd_i64_t const *b, simsimd_i64_t *r) {
    //? We can't just write `-9223372036854775808ll`, even though it's the smallest signed 64-bit value.
    //? The compiler will complain about the number being too large for the type, as it will process the
    //? constant and the sign separately. So we use the same hint that compilers use to define the `INT64_MIN`.
    if ((*b > 0) && (*a > (9223372036854775807ll) - *b)) { *r = 9223372036854775807ll; }                    // Overflow
    else if ((*b < 0) && (*a < (-9223372036854775807ll - 1ll) - *b)) { *r = -9223372036854775807ll - 1ll; } // Underflow
    else { *r = *a + *b; }
}
SIMSIMD_INTERNAL void _simsimd_f32_sadd(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_f32_t *r) {
    *r = *a + *b;
}
SIMSIMD_INTERNAL void _simsimd_f64_sadd(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_f64_t *r) {
    *r = *a + *b;
}
SIMSIMD_INTERNAL void _simsimd_f16_sadd(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_f16_t *r) {
    simsimd_f32_t a_f32, b_f32, r_f32;
    simsimd_f16_to_f32(a, &a_f32);
    simsimd_f16_to_f32(b, &b_f32);
    r_f32 = a_f32 + b_f32;
    simsimd_f32_to_f16(&r_f32, r);
}
SIMSIMD_INTERNAL void _simsimd_bf16_sadd(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_bf16_t *r) {
    simsimd_f32_t a_f32, b_f32, r_f32;
    simsimd_bf16_to_f32(a, &a_f32);
    simsimd_bf16_to_f32(b, &b_f32);
    r_f32 = a_f32 + b_f32;
    simsimd_f32_to_bf16(&r_f32, r);
}

SIMSIMD_INTERNAL void _simsimd_u8_smul(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_u8_t *r) {
    simsimd_u16_t result = (simsimd_u16_t)(*a) * (simsimd_u16_t)(*b);
    *r = (result > 255) ? 255 : (simsimd_u8_t)result;
}

SIMSIMD_INTERNAL void _simsimd_u16_smul(simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_u16_t *r) {
    simsimd_u32_t result = (simsimd_u32_t)(*a) * (simsimd_u32_t)(*b);
    *r = (result > 65535) ? 65535 : (simsimd_u16_t)result;
}

SIMSIMD_INTERNAL void _simsimd_u32_smul(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_u32_t *r) {
    simsimd_u64_t result = (simsimd_u64_t)(*a) * (simsimd_u64_t)(*b);
    *r = (result > 4294967295u) ? 4294967295u : (simsimd_u32_t)result;
}

SIMSIMD_INTERNAL void _simsimd_u64_smul(simsimd_u64_t const *a, simsimd_u64_t const *b, simsimd_u64_t *r) {
    // Split the inputs into high and low 32-bit parts
    simsimd_u64_t a_hi = *a >> 32;
    simsimd_u64_t a_lo = *a & 0xFFFFFFFF;
    simsimd_u64_t b_hi = *b >> 32;
    simsimd_u64_t b_lo = *b & 0xFFFFFFFF;

    // Compute partial products
    simsimd_u64_t hi_hi = a_hi * b_hi;
    simsimd_u64_t hi_lo = a_hi * b_lo;
    simsimd_u64_t lo_hi = a_lo * b_hi;
    simsimd_u64_t lo_lo = a_lo * b_lo;

    // Check if the high part of the result overflows
    if (hi_hi || (hi_lo >> 32) || (lo_hi >> 32) || ((hi_lo + lo_hi) >> 32)) { *r = 18446744073709551615ull; }
    else { *r = (hi_lo << 32) + (lo_hi << 32) + lo_lo; } // Combine parts if no overflow
}

SIMSIMD_INTERNAL void _simsimd_i8_smul(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_i8_t *r) {
    simsimd_i16_t result = (simsimd_i16_t)(*a) * (simsimd_i16_t)(*b);
    *r = (result > 127) ? 127 : (result < -128 ? -128 : (simsimd_i8_t)result);
}

SIMSIMD_INTERNAL void _simsimd_i16_smul(simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_i16_t *r) {
    simsimd_i32_t result = (simsimd_i32_t)(*a) * (simsimd_i32_t)(*b);
    *r = (result > 32767) ? 32767 : (result < -32768 ? -32768 : (simsimd_i16_t)result);
}

SIMSIMD_INTERNAL void _simsimd_i32_smul(simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_i32_t *r) {
    simsimd_i64_t result = (simsimd_i64_t)(*a) * (simsimd_i64_t)(*b);
    *r = (result > 2147483647ll) ? 2147483647ll : (result < -2147483648ll ? -2147483648ll : (simsimd_i32_t)result);
}

SIMSIMD_INTERNAL void _simsimd_i64_smul(simsimd_i64_t const *a, simsimd_i64_t const *b, simsimd_i64_t *r) {
    int sign = ((*a < 0) ^ (*b < 0)) ? -1 : 1; // Track the sign of the result

    // Take absolute values for easy multiplication and overflow detection
    simsimd_u64_t abs_a = (*a < 0) ? -*a : *a;
    simsimd_u64_t abs_b = (*b < 0) ? -*b : *b;

    // Split the absolute values into high and low 32-bit parts
    simsimd_u64_t a_hi = abs_a >> 32;
    simsimd_u64_t a_lo = abs_a & 0xFFFFFFFF;
    simsimd_u64_t b_hi = abs_b >> 32;
    simsimd_u64_t b_lo = abs_b & 0xFFFFFFFF;

    // Compute partial products
    simsimd_u64_t hi_hi = a_hi * b_hi;
    simsimd_u64_t hi_lo = a_hi * b_lo;
    simsimd_u64_t lo_hi = a_lo * b_hi;
    simsimd_u64_t lo_lo = a_lo * b_lo;

    // Check for overflow and saturate based on sign
    if (hi_hi || (hi_lo >> 32) || (lo_hi >> 32) || ((hi_lo + lo_hi) >> 32)) {
        *r = (sign > 0) ? 9223372036854775807ll : (-9223372036854775807ll - 1ll);
    }
    // Combine parts if no overflow, then apply the sign
    else {
        simsimd_u64_t result = (hi_lo << 32) + (lo_hi << 32) + lo_lo;
        *r = (sign < 0) ? -((simsimd_i64_t)result) : (simsimd_i64_t)result;
    }
}

SIMSIMD_INTERNAL void _simsimd_f32_smul(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_f32_t *r) {
    *r = *a * *b;
}

SIMSIMD_INTERNAL void _simsimd_f64_smul(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_f64_t *r) {
    *r = *a * *b;
}

SIMSIMD_INTERNAL void _simsimd_f16_smul(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_f16_t *r) {
    simsimd_f32_t a_f32, b_f32, r_f32;
    simsimd_f16_to_f32(a, &a_f32);
    simsimd_f16_to_f32(b, &b_f32);
    r_f32 = a_f32 * b_f32;
    simsimd_f32_to_f16(&r_f32, r);
}

SIMSIMD_INTERNAL void _simsimd_bf16_smul(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_bf16_t *r) {
    simsimd_f32_t a_f32, b_f32, r_f32;
    simsimd_bf16_to_f32(a, &a_f32);
    simsimd_bf16_to_f32(b, &b_f32);
    r_f32 = a_f32 * b_f32;
    simsimd_f32_to_bf16(&r_f32, r);
}

#if SIMSIMD_DYNAMIC_DISPATCH

/** @copydoc simsimd_f16_to_f32_implementation */
SIMSIMD_DYNAMIC void simsimd_f16_to_f32(simsimd_f16_t const *src, simsimd_f32_t *dest);

/** @copydoc simsimd_f32_to_f16_implementation */
SIMSIMD_DYNAMIC void simsimd_f32_to_f16(simsimd_f32_t const *src, simsimd_f16_t *dest);

/** @copydoc simsimd_bf16_to_f32_implementation */
SIMSIMD_DYNAMIC void simsimd_bf16_to_f32(simsimd_bf16_t const *src, simsimd_f32_t *dest);

/** @copydoc simsimd_f32_to_bf16_implementation */
SIMSIMD_DYNAMIC void simsimd_f32_to_bf16(simsimd_f32_t const *src, simsimd_bf16_t *dest);

/** @copydoc simsimd_e4m3_to_f32_implementation */
SIMSIMD_DYNAMIC void simsimd_e4m3_to_f32(simsimd_e4m3_t const *src, simsimd_f32_t *dest);

/** @copydoc simsimd_f32_to_e4m3_implementation */
SIMSIMD_DYNAMIC void simsimd_f32_to_e4m3(simsimd_f32_t const *src, simsimd_e4m3_t *dest);

/** @copydoc simsimd_e5m2_to_f32_implementation */
SIMSIMD_DYNAMIC void simsimd_e5m2_to_f32(simsimd_e5m2_t const *src, simsimd_f32_t *dest);

/** @copydoc simsimd_f32_to_e5m2_implementation */
SIMSIMD_DYNAMIC void simsimd_f32_to_e5m2(simsimd_f32_t const *src, simsimd_e5m2_t *dest);

#else // SIMSIMD_DYNAMIC_DISPATCH

/** @copydoc simsimd_f16_to_f32_implementation */
SIMSIMD_PUBLIC void simsimd_f16_to_f32(simsimd_f16_t const *src, simsimd_f32_t *dest) {
    simsimd_f16_to_f32_implementation(src, dest);
}

/** @copydoc simsimd_f32_to_f16_implementation */
SIMSIMD_PUBLIC void simsimd_f32_to_f16(simsimd_f32_t const *src, simsimd_f16_t *dest) {
    simsimd_f32_to_f16_implementation(src, dest);
}

/** @copydoc simsimd_bf16_to_f32_implementation */
SIMSIMD_PUBLIC void simsimd_bf16_to_f32(simsimd_bf16_t const *src, simsimd_f32_t *dest) {
    simsimd_bf16_to_f32_implementation(src, dest);
}

/** @copydoc simsimd_f32_to_bf16_implementation */
SIMSIMD_PUBLIC void simsimd_f32_to_bf16(simsimd_f32_t const *src, simsimd_bf16_t *dest) {
    simsimd_f32_to_bf16_implementation(src, dest);
}

/** @copydoc simsimd_e4m3_to_f32_implementation */
SIMSIMD_PUBLIC void simsimd_e4m3_to_f32(simsimd_e4m3_t const *src, simsimd_f32_t *dest) {
    simsimd_e4m3_to_f32_implementation(src, dest);
}

/** @copydoc simsimd_f32_to_e4m3_implementation */
SIMSIMD_PUBLIC void simsimd_f32_to_e4m3(simsimd_f32_t const *src, simsimd_e4m3_t *dest) {
    simsimd_f32_to_e4m3_implementation(src, dest);
}

/** @copydoc simsimd_e5m2_to_f32_implementation */
SIMSIMD_PUBLIC void simsimd_e5m2_to_f32(simsimd_e5m2_t const *src, simsimd_f32_t *dest) {
    simsimd_e5m2_to_f32_implementation(src, dest);
}

/** @copydoc simsimd_f32_to_e5m2_implementation */
SIMSIMD_PUBLIC void simsimd_f32_to_e5m2(simsimd_f32_t const *src, simsimd_e5m2_t *dest) {
    simsimd_f32_to_e5m2_implementation(src, dest);
}

#endif // SIMSIMD_DYNAMIC_DISPATCH

#ifdef __cplusplus
} // extern "C"
#endif

#endif
