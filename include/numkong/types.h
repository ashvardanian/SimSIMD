/**
 *  @brief Shared definitions for the NumKong library.
 *  @file numkong/types.h
 *  @author Ash Vardanian
 *  @date October 2, 2023
 *
 *  Defines:
 *
 *  - Sized aliases for numeric types, like: `nk_i32_t` and `nk_f64_t`.
 *  - Macros for internal compiler/hardware checks, like: `NK_TARGET_ARM_`.
 *  - Macros for feature controls, like: `NK_TARGET_NEON`
 *
 */
#ifndef NK_TYPES_H
#define NK_TYPES_H

// On Linux, `_GNU_SOURCE` must be defined before any system headers
// to expose `syscall` and other GNU extensions when C extensions are disabled.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

// Inferring target OS: Windows, macOS, or Linux
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define NK_DEFINED_WINDOWS_ 1
#elif defined(__APPLE__) && defined(__MACH__)
#define NK_DEFINED_APPLE_ 1
#elif defined(__linux__)
#define NK_DEFINED_LINUX_ 1
#endif

// Annotation for the public API symbols:
//
// - `NK_PUBLIC` is used for functions that are part of the public API.
// - `NK_INTERNAL` is used for internal helper functions with unstable APIs.
// - `NK_DYNAMIC` is used for functions that are part of the public API, but are dispatched at runtime.
//
// On GCC we mark the functions as `nonnull` informing that none of the arguments can be `NULL`.
// Marking with `pure` and `const` isn't possible as outputting to a pointer is a "side effect".
#if defined(__GNUC__) || defined(__clang__)
#define NK_PUBLIC   __attribute__((unused, nonnull)) inline static
#define NK_INTERNAL __attribute__((always_inline)) inline static
#else
#define NK_PUBLIC   inline static
#define NK_INTERNAL inline static
#endif

#if NK_DYNAMIC_DISPATCH
#if defined(_WIN32) || defined(__CYGWIN__)
#define NK_DYNAMIC __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define NK_DYNAMIC __attribute__((visibility("default"))) __attribute__((nonnull))
#else
#define NK_DYNAMIC NK_PUBLIC
#endif
#else
#define NK_DYNAMIC NK_PUBLIC
#endif

// Compiling for Arm: NK_TARGET_ARM_
#if !defined(NK_TARGET_ARM_)
#if defined(__aarch64__) || defined(_M_ARM64)
#define NK_TARGET_ARM_ 1
#else
#define NK_TARGET_ARM_ 0
#endif // defined(__aarch64__) || defined(_M_ARM64)
#endif // !defined(NK_TARGET_ARM_)

// Compiling for x86: NK_TARGET_X86_
#if !defined(NK_TARGET_X86_)
#if defined(__x86_64__) || defined(_M_X64)
#define NK_TARGET_X86_ 1
#else
#define NK_TARGET_X86_ 0
#endif // defined(__x86_64__) || defined(_M_X64)
#endif // !defined(NK_TARGET_X86_)

// Compiling for Arm: NK_TARGET_NEON
#if !defined(NK_TARGET_NEON) || (NK_TARGET_NEON && !NK_TARGET_ARM_)
#if defined(__ARM_NEON)
#define NK_TARGET_NEON NK_TARGET_ARM_
#else
#undef NK_TARGET_NEON
#define NK_TARGET_NEON 0
#endif // defined(__ARM_NEON)
#endif // !defined(NK_TARGET_NEON) || ...

// Compiling for Arm: NK_TARGET_NEONSDOT
#if !defined(NK_TARGET_NEONSDOT) || (NK_TARGET_NEONSDOT && !NK_TARGET_ARM_)
#if defined(__ARM_NEON)
#define NK_TARGET_NEONSDOT NK_TARGET_ARM_
#else
#undef NK_TARGET_NEONSDOT
#define NK_TARGET_NEONSDOT 0
#endif // defined(__ARM_NEON)
#endif // !defined(NK_TARGET_NEONSDOT) || ...

// Compiling for Arm: NK_TARGET_NEONHALF
#if !defined(NK_TARGET_NEONHALF) || (NK_TARGET_NEONHALF && !NK_TARGET_ARM_)
#if defined(__ARM_NEON)
#define NK_TARGET_NEONHALF NK_TARGET_ARM_
#else
#undef NK_TARGET_NEONHALF
#define NK_TARGET_NEONHALF 0
#endif // defined(__ARM_NEON)
#endif // !defined(NK_TARGET_NEONHALF) || ...

// Compiling for Arm: NK_TARGET_NEONFHM (FEAT_FHM - FMLAL/FMLSL widening ops)
#if !defined(NK_TARGET_NEONFHM) || (NK_TARGET_NEONFHM && !NK_TARGET_ARM_)
#if defined(__ARM_NEON)
#define NK_TARGET_NEONFHM NK_TARGET_ARM_
#else
#undef NK_TARGET_NEONFHM
#define NK_TARGET_NEONFHM 0
#endif // defined(__ARM_NEON)
#endif // !defined(NK_TARGET_NEONFHM) || ...

// Compiling for Arm: NK_TARGET_NEONBFDOT
#if !defined(NK_TARGET_NEONBFDOT) || (NK_TARGET_NEONBFDOT && !NK_TARGET_ARM_)
#if defined(__ARM_NEON)
#define NK_TARGET_NEONBFDOT NK_TARGET_ARM_
#else
#undef NK_TARGET_NEONBFDOT
#define NK_TARGET_NEONBFDOT 0
#endif // defined(__ARM_NEON)
#endif // !defined(NK_TARGET_NEONBFDOT) || ...

// Compiling for Arm: NK_TARGET_SVE
#if !defined(NK_TARGET_SVE) || (NK_TARGET_SVE && !NK_TARGET_ARM_)
#if defined(__ARM_FEATURE_SVE)
#define NK_TARGET_SVE NK_TARGET_ARM_
#else
#undef NK_TARGET_SVE
#define NK_TARGET_SVE 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NK_TARGET_SVE) || ...

// Compiling for Arm: NK_TARGET_SVESDOT
#if !defined(NK_TARGET_SVESDOT) || (NK_TARGET_SVESDOT && !NK_TARGET_ARM_)
#if defined(__ARM_FEATURE_SVE)
#define NK_TARGET_SVESDOT NK_TARGET_ARM_
#else
#undef NK_TARGET_SVESDOT
#define NK_TARGET_SVESDOT 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NK_TARGET_SVESDOT) || ...

// Compiling for Arm: NK_TARGET_SVEHALF
#if !defined(NK_TARGET_SVEHALF) || (NK_TARGET_SVEHALF && !NK_TARGET_ARM_)
#if defined(__ARM_FEATURE_SVE)
#define NK_TARGET_SVEHALF NK_TARGET_ARM_
#else
#undef NK_TARGET_SVEHALF
#define NK_TARGET_SVEHALF 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NK_TARGET_SVEHALF) || ...

// Compiling for Arm: NK_TARGET_SVEBFDOT
#if !defined(NK_TARGET_SVEBFDOT) || (NK_TARGET_SVEBFDOT && !NK_TARGET_ARM_)
#if defined(__ARM_FEATURE_SVE)
#define NK_TARGET_SVEBFDOT NK_TARGET_ARM_
#else
#undef NK_TARGET_SVEBFDOT
#define NK_TARGET_SVEBFDOT 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NK_TARGET_SVEBFDOT) || ...

// Compiling for Arm: NK_TARGET_SVE2
#if !defined(NK_TARGET_SVE2) || (NK_TARGET_SVE2 && !NK_TARGET_ARM_)
#if defined(__ARM_FEATURE_SVE)
#define NK_TARGET_SVE2 NK_TARGET_ARM_
#else
#undef NK_TARGET_SVE2
#define NK_TARGET_SVE2 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NK_TARGET_SVE2) || ...

// Compiling for Arm: NK_TARGET_SVE2P1
#if !defined(NK_TARGET_SVE2P1) || (NK_TARGET_SVE2P1 && !NK_TARGET_ARM_)
#if defined(__ARM_FEATURE_SVE)
#define NK_TARGET_SVE2P1 NK_TARGET_ARM_
#else
#undef NK_TARGET_SVE2P1
#define NK_TARGET_SVE2P1 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NK_TARGET_SVE2P1) || ...

// Compiling for Arm: NK_TARGET_SME (Scalable Matrix Extension)
#if !defined(NK_TARGET_SME)
#define NK_TARGET_SME 0
#endif

#if !defined(NK_TARGET_SME2)
#define NK_TARGET_SME2 0
#endif

#if !defined(NK_TARGET_SME2P1)
#define NK_TARGET_SME2P1 0
#endif

#if !defined(NK_TARGET_SMEF64)
#define NK_TARGET_SMEF64 0
#endif

#if !defined(NK_TARGET_SMEHALF)
#define NK_TARGET_SMEHALF 0
#endif

#if !defined(NK_TARGET_SMEBF16)
#define NK_TARGET_SMEBF16 0
#endif

#if !defined(NK_TARGET_SMELUT2)
#define NK_TARGET_SMELUT2 0
#endif

#if !defined(NK_TARGET_SMEFA64)
#define NK_TARGET_SMEFA64 0
#endif

// Compiling for x86: NK_TARGET_HASWELL
//
// Starting with Ivy Bridge, Intel supports the `F16C` extensions for fast half-precision
// to single-precision floating-point conversions. On AMD those instructions
// are supported on all CPUs starting with Jaguar 2009.
// Starting with Sandy Bridge, Intel adds basic AVX support in their CPUs and in 2013
// extends it with AVX2 in the Haswell generation. Moreover, Haswell adds FMA support.
#if !defined(NK_TARGET_HASWELL) || (NK_TARGET_HASWELL && !NK_TARGET_X86_)
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define NK_TARGET_HASWELL 1
#else
#undef NK_TARGET_HASWELL
#define NK_TARGET_HASWELL 0
#endif // defined(__AVX2__)
#endif // !defined(NK_TARGET_HASWELL) || ...

// Compiling for x86: NK_TARGET_SKYLAKE, NK_TARGET_ICE, NK_TARGET_GENOA,
// NK_TARGET_SAPPHIRE, NK_TARGET_TURIN, NK_TARGET_SIERRA
//
// To list all available macros for x86, take a recent compiler, like GCC 12 and run:
//      gcc-12 -march=sapphirerapids -dM -E - < /dev/null | egrep "SSE|AVX" | sort
// On Arm machines you may want to check for other flags:
//      gcc-12 -march=native -dM -E - < /dev/null | egrep "NEON|SVE|FP16|FMA" | sort
#if !defined(NK_TARGET_SKYLAKE) || (NK_TARGET_SKYLAKE && !NK_TARGET_X86_)
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && \
    defined(__AVX512BW__)
#define NK_TARGET_SKYLAKE 1
#else
#undef NK_TARGET_SKYLAKE
#define NK_TARGET_SKYLAKE 0
#endif
#endif // !defined(NK_TARGET_SKYLAKE) || ...
#if !defined(NK_TARGET_ICE) || (NK_TARGET_ICE && !NK_TARGET_X86_)
#if defined(__AVX512VNNI__) && defined(__AVX512IFMA__) && defined(__AVX512BITALG__) && defined(__AVX512VBMI2__) && \
    defined(__AVX512VPOPCNTDQ__)
#define NK_TARGET_ICE 1
#else
#undef NK_TARGET_ICE
#define NK_TARGET_ICE 0
#endif
#endif // !defined(NK_TARGET_ICE) || ...
#if !defined(NK_TARGET_GENOA) || (NK_TARGET_GENOA && !NK_TARGET_X86_)
#if defined(__AVX512BF16__)
#define NK_TARGET_GENOA 1
#else
#undef NK_TARGET_GENOA
#define NK_TARGET_GENOA 0
#endif
#endif // !defined(NK_TARGET_GENOA) || ...
#if !defined(NK_TARGET_SAPPHIRE) || (NK_TARGET_SAPPHIRE && !NK_TARGET_X86_)
#if defined(__AVX512FP16__)
#define NK_TARGET_SAPPHIRE 1
#else
#undef NK_TARGET_SAPPHIRE
#define NK_TARGET_SAPPHIRE 0
#endif
#endif // !defined(NK_TARGET_SAPPHIRE) || ...
#if !defined(NK_TARGET_SAPPHIRE_AMX) || (NK_TARGET_SAPPHIRE_AMX && !NK_TARGET_X86_)
#if defined(__AMX_TILE__) && defined(__AMX_BF16__) && defined(__AMX_INT8__)
#define NK_TARGET_SAPPHIRE_AMX 1
#else
#undef NK_TARGET_SAPPHIRE_AMX
#define NK_TARGET_SAPPHIRE_AMX 0
#endif
#endif // !defined(NK_TARGET_SAPPHIRE_AMX) || ...
#if !defined(NK_TARGET_GRANITE_AMX) || (NK_TARGET_GRANITE_AMX && !NK_TARGET_X86_)
#if defined(__AMX_TILE__) && defined(__AMX_FP16__)
#define NK_TARGET_GRANITE_AMX 1
#else
#undef NK_TARGET_GRANITE_AMX
#define NK_TARGET_GRANITE_AMX 0
#endif
#endif // !defined(NK_TARGET_GRANITE_AMX) || ...
#if !defined(NK_TARGET_TURIN) || (NK_TARGET_TURIN && !NK_TARGET_X86_)
#if defined(__AVX512VP2INTERSECT__)
#define NK_TARGET_TURIN 1
#else
#undef NK_TARGET_TURIN
#define NK_TARGET_TURIN 0
#endif
#endif // !defined(NK_TARGET_TURIN) || ...
#if !defined(NK_TARGET_SIERRA) || (NK_TARGET_SIERRA && !NK_TARGET_X86_)
#if defined(__AVX2_VNNI__)
#define NK_TARGET_SIERRA 1
#else
#undef NK_TARGET_SIERRA
#define NK_TARGET_SIERRA 0
#endif
#endif // !defined(NK_TARGET_SIERRA) || ...

#if defined(_MSC_VER)
#include <intrin.h>
#else

#if NK_TARGET_NEON
#include <arm_neon.h>
#endif

#if NK_TARGET_SVE || NK_TARGET_SVE2
#include <arm_sve.h>
#endif

#if NK_TARGET_HASWELL || NK_TARGET_SKYLAKE || NK_TARGET_ICE || NK_TARGET_GENOA || NK_TARGET_SAPPHIRE || NK_TARGET_TURIN
#include <immintrin.h>
#endif

#endif

#if !defined(NK_F32_SQRT)
#include <math.h>
#define NK_F32_SQRT(x) (sqrtf(x))
#endif

#if !defined(NK_F32_RSQRT)
#include <math.h>
#define NK_F32_RSQRT(x) (1 / NK_F32_SQRT(x))
#endif

#if !defined(NK_F64_SQRT)
#include <math.h>
#define NK_F64_SQRT(x) (sqrt(x))
#endif

#if !defined(NK_F64_RSQRT)
#include <math.h>
#define NK_F64_RSQRT(x) (1 / NK_F64_SQRT(x))
#endif

#if !defined(NK_F32_LOG)
#include <math.h>
#define NK_F32_LOG(x) (logf(x))
#endif

#if !defined(NK_F64_LOG)
#include <math.h>
#define NK_F64_LOG(x) (log(x))
#endif

#if !defined(NK_F32_TAN)
#include <math.h>
#define NK_F32_TAN(x) (tanf(x))
#endif

#if !defined(NK_F64_TAN)
#include <math.h>
#define NK_F64_TAN(x) (tan(x))
#endif

#if !defined(NK_F32_ABS)
#include <math.h>
#define NK_F32_ABS(x) (fabsf(x))
#endif

#if !defined(NK_F64_ABS)
#include <math.h>
#define NK_F64_ABS(x) (fabs(x))
#endif

#if !defined(NK_F32_DIVISION_EPSILON)
#define NK_F32_DIVISION_EPSILON (1e-7)
#endif

#if !defined(NK_F16_DIVISION_EPSILON)
#define NK_F16_DIVISION_EPSILON (1e-3)
#endif

/**
 *  @brief  The compile-time constant defining the capacity of `nk_xd_index_t`.
 *          Matches `PyBUF_MAX_NDIM` by default.
 */
#if !defined(NK_NDARRAY_MAX_RANK)
#define NK_NDARRAY_MAX_RANK (64)
#endif

/**
 *  @brief  Aligns a variable to a 64-byte boundary using compiler extensions for
 *          compatibility with C 99, as `alignas(64)` is only available in C 11 or C++.
 *          Used internally and recommended for external users.
 */
#if defined(_MSC_VER)
#define NK_ALIGN64 __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
#define NK_ALIGN64 __attribute__((aligned(64)))
#endif

/** Copy 16 bits (2 bytes) from source to destination */
#if defined(__GNUC__) || defined(__clang__)
#define nk_copy_bytes_(destination_ptr, source_ptr, count) __builtin_memcpy((destination_ptr), (source_ptr), count)
#else
#include <string.h> // fallback for exotic compilers
#define nk_copy_bytes_(destination_ptr, source_ptr, count) memcpy((destination_ptr), (source_ptr), count)
#endif

/**
 *  @brief C99 static array parameter annotation for minimum array size.
 *
 *  In C, expands to `static n` enabling compiler bounds checking.
 *  In C++, expands to nothing as this syntax is not supported.
 *  @see https://lwn.net/Articles/1046840/
 *
 *  Example usage:
 *  @code{.c}
 *      void hash_digest(uint8_t digest[nk_at_least_(32)]);
 *      void lookup(uint8_t const lut[nk_at_least_(256)]);
 *  @endcode
 */
#if defined(__cplusplus) || defined(_MSC_VER)
#define nk_at_least_(n)
#else
#define nk_at_least_(n) static n
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char nk_b8_t;   /// ? Eight boolean values packed in one byte
typedef unsigned char nk_i4x2_t; /// ? Two 4-bit signed integers packed in one byte
typedef unsigned char nk_e4m3_t; /// ? FP8 E4M3 value encoded into one byte
typedef unsigned char nk_e5m2_t; /// ? FP8 E5M2 value encoded into one byte

typedef signed char nk_i8_t;
typedef unsigned char nk_u8_t;
typedef signed short nk_i16_t;
typedef unsigned short nk_u16_t;
typedef signed int nk_i32_t;
typedef unsigned int nk_u32_t;
typedef signed long long nk_i64_t;
typedef unsigned long long nk_u64_t;

typedef float nk_f32_t;
typedef double nk_f64_t;

typedef nk_u64_t nk_size_t;
typedef nk_i64_t nk_ssize_t;
typedef nk_f64_t nk_fmax_t;

/**
 *  @brief  Enumeration of supported scalar data types.
 *
 *  Includes complex type descriptors which in C code would use the real counterparts,
 *  but the independent flags contain metadata to be passed between programming language
 *  interfaces.
 */
typedef enum {
    nk_datatype_unknown_k = 0, ///< Unknown data type
    nk_b8_k = 1 << 1,          ///< Single-bit values packed into 8-bit words
    nk_b1x8_k = nk_b8_k,       ///< Single-bit values packed into 8-bit words
    nk_i4x2_k = 1 << 19,       ///< 4-bit signed integers packed into 8-bit words

    nk_i8_k = 1 << 2,  ///< 8-bit signed integer
    nk_i16_k = 1 << 3, ///< 16-bit signed integer
    nk_i32_k = 1 << 4, ///< 32-bit signed integer
    nk_i64_k = 1 << 5, ///< 64-bit signed integer

    nk_u8_k = 1 << 6,  ///< 8-bit unsigned integer
    nk_u16_k = 1 << 7, ///< 16-bit unsigned integer
    nk_u32_k = 1 << 8, ///< 32-bit unsigned integer
    nk_u64_k = 1 << 9, ///< 64-bit unsigned integer

    nk_f64_k = 1 << 10,  ///< Double precision floating point
    nk_f32_k = 1 << 11,  ///< Single precision floating point
    nk_f16_k = 1 << 12,  ///< Half precision floating point
    nk_bf16_k = 1 << 13, ///< Brain floating point

    nk_e4m3_k = 1 << 14, ///< FP8 E4M3 floating point
    nk_e5m2_k = 1 << 15, ///< FP8 E5M2 floating point

    nk_f64c_k = 1 << 20,  ///< Complex double precision floating point
    nk_f32c_k = 1 << 21,  ///< Complex single precision floating point
    nk_f16c_k = 1 << 22,  ///< Complex half precision floating point
    nk_bf16c_k = 1 << 23, ///< Complex brain floating point
} nk_datatype_t;

/*  @brief  Half-precision floating-point type.
 *
 *  - GCC or Clang on 64-bit Arm: `__fp16`, may require `-mfp16-format` option.
 *  - GCC or Clang on 64-bit x86: `_Float16`.
 *  - Default: `unsigned short`.
 *
 */
#if !defined(NK_NATIVE_F16) || NK_NATIVE_F16
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__ARM_ARCH) || defined(__aarch64__)) && \
    (defined(__ARM_FP16_FORMAT_IEEE))
#undef NK_NATIVE_F16
#define NK_NATIVE_F16 1
typedef __fp16 nk_f16_t;
#elif ((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && \
       (defined(__AVX512FP16__)))
typedef _Float16 nk_f16_t;
#undef NK_NATIVE_F16
#define NK_NATIVE_F16 1
#else                                       // Unknown compiler or architecture
#if defined(__GNUC__) || defined(__clang__) // Some compilers don't support warning pragmas
#warning "Unknown compiler or architecture for float16."
#endif
#undef NK_NATIVE_F16
#define NK_NATIVE_F16 0
#endif // Unknown compiler or architecture
#endif // !NK_NATIVE_F16

#if !NK_NATIVE_F16
typedef unsigned short nk_f16_t;
#endif

#if !defined(NK_NATIVE_BF16) || NK_NATIVE_BF16
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
#undef NK_NATIVE_BF16
#define NK_NATIVE_BF16 1
typedef __bf16 nk_bf16_t;
#elif ((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && \
       (defined(__AVX512BF16__)))
typedef __bfloat16 nk_bf16_t;
#undef NK_NATIVE_BF16
#define NK_NATIVE_BF16 1
#else                                       // Unknown compiler or architecture
#if defined(__GNUC__) || defined(__clang__) // Some compilers don't support warning pragmas
#warning "Unknown compiler or architecture for bfloat16."
#endif
#undef NK_NATIVE_BF16
#define NK_NATIVE_BF16 0
#endif // Unknown compiler or architecture
#endif // !NK_NATIVE_BF16

#if !NK_NATIVE_BF16
typedef unsigned short nk_bf16_t;
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
#if NK_TARGET_ARM_
#if defined(_MSC_VER)
#define nk_f16_for_arm_simd_t  nk_f16_t
#define nk_bf16_for_arm_simd_t nk_bf16_t
#else
#define nk_f16_for_arm_simd_t  float16_t
#define nk_bf16_for_arm_simd_t bfloat16_t
#endif
#endif

/*
 *  Let's make sure the sizes of the types are as expected.
 *  In C the `_Static_assert` is only available with C11 and later.
 */
#define NK_STATIC_ASSERT(cond, msg) typedef char static_assertion_##msg[(cond) ? 1 : -1]
NK_STATIC_ASSERT(sizeof(nk_b8_t) == 1, nk_b8_t_must_be_1_byte);
NK_STATIC_ASSERT(sizeof(nk_i4x2_t) == 1, nk_i4x2_t_must_be_1_byte);
NK_STATIC_ASSERT(sizeof(nk_e4m3_t) == 1, nk_e4m3_t_must_be_1_byte);
NK_STATIC_ASSERT(sizeof(nk_e5m2_t) == 1, nk_e5m2_t_must_be_1_byte);
NK_STATIC_ASSERT(sizeof(nk_i8_t) == 1, nk_i8_t_must_be_1_byte);
NK_STATIC_ASSERT(sizeof(nk_u8_t) == 1, nk_u8_t_must_be_1_byte);
NK_STATIC_ASSERT(sizeof(nk_i16_t) == 2, nk_i16_t_must_be_2_bytes);
NK_STATIC_ASSERT(sizeof(nk_u16_t) == 2, nk_u16_t_must_be_2_bytes);
NK_STATIC_ASSERT(sizeof(nk_i32_t) == 4, nk_i32_t_must_be_4_bytes);
NK_STATIC_ASSERT(sizeof(nk_u32_t) == 4, nk_u32_t_must_be_4_bytes);
NK_STATIC_ASSERT(sizeof(nk_i64_t) == 8, nk_i64_t_must_be_8_bytes);
NK_STATIC_ASSERT(sizeof(nk_u64_t) == 8, nk_u64_t_must_be_8_bytes);
NK_STATIC_ASSERT(sizeof(nk_f32_t) == 4, nk_f32_t_must_be_4_bytes);
NK_STATIC_ASSERT(sizeof(nk_f64_t) == 8, nk_f64_t_must_be_8_bytes);
NK_STATIC_ASSERT(sizeof(nk_f16_t) == 2, nk_f16_t_must_be_2_bytes);
NK_STATIC_ASSERT(sizeof(nk_bf16_t) == 2, nk_bf16_t_must_be_2_bytes);

#define nk_assign_from_to_(src, dest) (*(dest) = *(src))

/** @brief  Convenience type for single-precision floating-point bit manipulation. */
typedef union {
    nk_u32_t u;
    nk_f32_t f;
} nk_fui32_t;

/** @brief  Convenience type for double-precision floating-point bit manipulation. */
typedef union {
    nk_u64_t u;
    nk_f64_t f;
} nk_fui64_t;

/** @brief  Convenience type addressing the real and imaginary parts of a half-precision complex number. */
typedef struct {
    nk_f16_t real;
    nk_f16_t imag;
} nk_f16c_t;

/** @brief  Convenience type addressing the real and imaginary parts of a half-precision brain-float complex number. */
typedef struct {
    nk_bf16_t real;
    nk_bf16_t imag;
} nk_bf16c_t;

/** @brief  Convenience type addressing the real and imaginary parts of a single-precision complex number. */
typedef struct {
    nk_f32_t real;
    nk_f32_t imag;
} nk_f32c_t;

/** @brief  Convenience type addressing the real and imaginary parts of a double-precision complex number. */
typedef struct {
    nk_f64_t real;
    nk_f64_t imag;
} nk_f64c_t;

/** @brief  Small 8-byte memory slice viewable as different types. */
typedef union nk_b64_vec_t {
#if NK_TARGET_NEON
    uint8x8_t u8x8;
    uint16x4_t u16x4;
    uint32x2_t u32x2;
    int8x8_t i8x8;
    int16x4_t i16x4;
    int32x2_t i32x2;
    float32x2_t f32x2;
#endif
#if NK_TARGET_NEONHALF
    float16x4_t f16x4;
#endif
    nk_u8_t u8s[8];
    nk_u16_t u16s[4];
    nk_u32_t u32s[2];
    nk_u64_t u64s[1];
    nk_i8_t i8s[8];
    nk_i16_t i16s[4];
    nk_i32_t i32s[2];
    nk_i64_t i64s[1];
    nk_f16_t f16s[4];
    nk_bf16_t bf16s[4];
    nk_f32_t f32s[2];
} nk_b64_vec_t;

/** @brief  Small 16-byte memory slice viewable as different types. */
typedef union nk_b128_vec_t {
#if NK_TARGET_HASWELL
    __m128i xmm;
    __m128d xmm_pd;
    __m128 xmm_ps;
#endif
#if NK_TARGET_NEON
    uint8x16_t u8x16;
    uint16x8_t u16x8;
    uint32x4_t u32x4;
    uint64x2_t u64x2;
    int8x16_t i8x16;
    int16x8_t i16x8;
    int32x4_t i32x4;
    int64x2_t i64x2;
    float32x4_t f32x4;
    float64x2_t f64x2;
#endif
    nk_u8_t u8s[16];
    nk_u16_t u16s[8];
    nk_u32_t u32s[4];
    nk_u64_t u64s[2];
    nk_i8_t i8s[16];
    nk_i16_t i16s[8];
    nk_i32_t i32s[4];
    nk_i64_t i64s[2];
    nk_f16_t f16s[8];
    nk_bf16_t bf16s[8];
    nk_e4m3_t e4m3s[16];
    nk_e5m2_t e5m2s[16];
    nk_f32_t f32s[4];
    nk_f64_t f64s[2];
    nk_b8_t b8s[16];
} nk_b128_vec_t;

/** @brief  Small 32-byte memory slice viewable as different types. */
typedef union nk_b256_vec_t {
#if NK_TARGET_HASWELL
    __m256i ymm;
    __m256d ymm_pd;
    __m256 ymm_ps;
    __m128i xmms[2];
#endif
#if NK_TARGET_NEON
    uint8x16_t u8x16s[2];
    uint16x8_t u16x8s[2];
    uint32x4_t u32x4s[2];
    uint64x2_t u64x2s[2];
    int8x16_t i8x16s[2];
    int16x8_t i16x8s[2];
    int32x4_t i32x4s[2];
    int64x2_t i64x2s[2];
    float32x4_t f32x4s[2];
    float64x2_t f64x2s[2];
#endif
    nk_u8_t u8s[32];
    nk_u16_t u16s[16];
    nk_u32_t u32s[8];
    nk_u64_t u64s[4];
    nk_i8_t i8s[32];
    nk_i16_t i16s[16];
    nk_i32_t i32s[8];
    nk_i64_t i64s[4];
    nk_f16_t f16s[16];
    nk_bf16_t bf16s[16];
    nk_e4m3_t e4m3s[32];
    nk_e5m2_t e5m2s[32];
    nk_f32_t f32s[8];
    nk_f64_t f64s[4];
    nk_b8_t b8s[32];
} nk_b256_vec_t;

/** @brief  Small 64-byte memory slice viewable as different types.
 *
 *  TODO: On GCC and Clang we use `__transparent_union__` attribute to allow implicit conversions
 *  between the different vector types when passing them as function arguments. The most important side-effect
 *  of this is that the argument of such type is passed to functions using the calling convention of the first
 *  member of the union, which in our case is a register-based calling convention for SIMD types.
 */
typedef union nk_b512_vec_t {
#if NK_TARGET_SKYLAKE || NK_TARGET_ICE || NK_TARGET_GENOA || NK_TARGET_SAPPHIRE || NK_TARGET_TURIN || NK_TARGET_SIERRA
    __m512i zmm;
    __m512d zmm_pd;
    __m512 zmm_ps;
#endif
#if NK_TARGET_HASWELL
    __m256i ymms[2];
    __m256d ymms_pd[2];
    __m256 ymms_ps[2];
    __m128i xmms[4];
    __m128d xmms_pd[4];
    __m128 xmms_ps[4];
#endif
#if NK_TARGET_NEON
    uint8x16_t u8x16s[4];
    uint16x8_t u16x8s[4];
    uint32x4_t u32x4s[4];
    uint64x2_t u64x2s[4];
#endif
    nk_u8_t u8s[64];
    nk_u16_t u16s[32];
    nk_u32_t u32s[16];
    nk_u64_t u64s[8];
    nk_i8_t i8s[64];
    nk_i16_t i16s[32];
    nk_i32_t i32s[16];
    nk_i64_t i64s[8];
    nk_f16_t f16s[32];
    nk_bf16_t bf16s[32];
    nk_f32_t f32s[16];
    nk_f64_t f64s[8];
    nk_e4m3_t e4m3s[64];
    nk_e5m2_t e5m2s[64];
    nk_b8_t b8s[64];
} nk_b512_vec_t;

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
NK_INTERNAL nk_f32_t nk_f32_approximate_inverse_square_root(nk_f32_t number) {
    nk_fui32_t conv;
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
NK_INTERNAL nk_f32_t nk_f32_approximate_square_root(nk_f32_t number) {
    return number * nk_f32_approximate_inverse_square_root(number);
}

/**
 *  @brief  Computes `log(x)` using the Mercator series.
 *          The series converges to the natural logarithm for args between -1 and 1.
 *          Published in 1668 in "Logarithmotechnia".
 */
NK_INTERNAL nk_f32_t nk_f32_approximate_log(nk_f32_t number) {
    nk_f32_t x = number - 1;
    nk_f32_t x2 = x * x;
    nk_f32_t x3 = x * x * x;
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
NK_INTERNAL void nk_f16_to_f32_(nk_f16_t const *src, nk_f32_t *dest) {
#if NK_NATIVE_F16
    *dest = (nk_f32_t)(*src);
#else
    unsigned short x;
    nk_copy_bytes_(&x, src, 2);
    unsigned int exponent = (x & 0x7C00) >> 10;
    unsigned int mantissa = (x & 0x03FF) << 13;
    nk_fui32_t mantissa_conv;
    mantissa_conv.f = (float)mantissa;
    unsigned int v = (mantissa_conv.u) >> 23;
    nk_fui32_t conv;
    conv.u = (x & 0x8000) << 16 | (exponent != 0) * ((exponent + 112) << 23 | mantissa) |
             ((exponent == 0) & (mantissa != 0)) * ((v - 37) << 23 | ((mantissa << (150 - v)) & 0x007FE000));
    *dest = conv.f;
#endif
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
NK_INTERNAL void nk_f32_to_f16_(nk_f32_t const *src, nk_f16_t *dest) {
#if NK_NATIVE_F16
    *dest = (nk_f16_t)(*src);
#else
    nk_fui32_t conv;
    conv.f = *src;
    unsigned int b = conv.u + 0x00001000;
    unsigned int e = (b & 0x7F800000) >> 23;
    unsigned int m = b & 0x007FFFFF;
    unsigned short result = ((b & 0x80000000) >> 16) | (e > 112) * ((((e - 112) << 10) & 0x7C00) | (m >> 13)) |
                            ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
                            ((e > 143) * 0x7FFF);
    nk_copy_bytes_(dest, &result, 2);
#endif
}

/**
 *  @brief  For compilers that don't natively support the `__bf16` type,
 *          upcasts contents into a more conventional `float`.
 *
 *  https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c/55254307#55254307
 *  https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
 */
NK_INTERNAL void nk_bf16_to_f32_(nk_bf16_t const *src, nk_f32_t *dest) {
#if NK_NATIVE_BF16
    *dest = (nk_f32_t)(*src);
#else
    unsigned short x;
    nk_copy_bytes_(&x, src, 2);
    nk_fui32_t conv;
    conv.u = x << 16; // Zero extends the mantissa
    *dest = conv.f;
#endif
}

/**
 *  @brief  Compresses a `float` to a `bf16` representation.
 *
 *  https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c/55254307#55254307
 *  https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
 */
NK_INTERNAL void nk_f32_to_bf16_(nk_f32_t const *src, nk_bf16_t *dest) {
#if NK_NATIVE_BF16
    *dest = (nk_bf16_t)(*src);
#else
    nk_fui32_t conv;
    conv.f = *src;
    conv.u += 0x8000; // Rounding is optional
    conv.u >>= 16;
    // Use an intermediate variable to ensure correct behavior on big-endian systems.
    // Copying directly from `&conv.u` would copy the wrong bytes on big-endian,
    // since the lower 16 bits are at offset 2, not offset 0.
    unsigned short result = (unsigned short)conv.u;
    nk_copy_bytes_(dest, &result, 2);
#endif
}

NK_INTERNAL void nk_e4m3_to_f32_(nk_e4m3_t const *src, nk_f32_t *dest) {
    nk_u8_t raw = *src;
    nk_u32_t sign = (nk_u32_t)(raw & 0x80) << 24;
    nk_u32_t exponent = (raw >> 3) & 0x0Fu;
    nk_u32_t mantissa = raw & 0x07u;
    nk_fui32_t conv;

    if (exponent == 0) {
        if (mantissa == 0) {
            conv.u = sign;
            *dest = conv.f;
            return;
        }
        nk_f32_t value = (nk_f32_t)mantissa * (1.0f / 512.0f);
        *dest = sign ? -value : value;
        return;
    }
    if (exponent == 0x0Fu) {
        if (mantissa == 0) { conv.u = sign | 0x7F800000u; }
        else { conv.u = sign | 0x7FC00000u; }
        *dest = conv.f;
        return;
    }

    nk_u32_t f32_exponent = (exponent + 120u) << 23;
    nk_u32_t f32_mantissa = mantissa << 20;
    conv.u = sign | f32_exponent | f32_mantissa;
    *dest = conv.f;
}

NK_INTERNAL void nk_f32_to_e4m3_(nk_f32_t const *src, nk_e4m3_t *dest) {
    nk_f32_t x = *src;
    nk_fui32_t conv;
    conv.f = x;
    nk_u32_t sign_bit = conv.u >> 31;
    nk_u32_t abs_bits = conv.u & 0x7FFFFFFFu;
    nk_u8_t sign = (nk_u8_t)(sign_bit << 7);

    if (abs_bits >= 0x7F800000u) {
        nk_u8_t mant = (abs_bits > 0x7F800000u) ? 0x01u : 0x00u;
        *dest = (nk_e4m3_t)(sign | 0x78u | mant);
        return;
    }

    if (abs_bits == 0) {
        *dest = (nk_e4m3_t)sign;
        return;
    }

    nk_f32_t abs_x = sign_bit ? -x : x;

    if (abs_x < (1.0f / 512.0f)) {
        *dest = (nk_e4m3_t)sign;
        return;
    }

    if (abs_x < (1.0f / 64.0f)) {
        nk_f32_t scaled = abs_x * 512.0f;
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        if (mant > 7) { mant = 7; }
        if (mant == 0) { *dest = (nk_e4m3_t)sign; }
        else { *dest = (nk_e4m3_t)(sign | (nk_u8_t)mant); }
        return;
    }

    nk_i32_t exp = (nk_i32_t)((abs_bits >> 23) & 0xFFu) - 127;
    nk_u32_t mantissa = abs_bits & 0x7FFFFFu;
    nk_u32_t significand = (1u << 23) | mantissa;
    nk_i32_t shift = 23 - 3;
    nk_u32_t remainder_mask = (1u << shift) - 1;
    nk_u32_t remainder = significand & remainder_mask;
    nk_u32_t halfway = 1u << (shift - 1);
    nk_u32_t significand_rounded = significand >> shift;
    if (remainder > halfway || (remainder == halfway && (significand_rounded & 1))) { ++significand_rounded; }
    if (significand_rounded == (1u << (3 + 1))) {
        significand_rounded >>= 1;
        ++exp;
    }
    if (exp > 7) {
        *dest = (nk_e4m3_t)(sign | 0x78u);
        return;
    }
    if (exp < -6) {
        nk_f32_t scaled = abs_x * 512.0f;
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        if (mant > 7) { mant = 7; }
        if (mant == 0) { *dest = (nk_e4m3_t)sign; }
        else { *dest = (nk_e4m3_t)(sign | (nk_u8_t)mant); }
        return;
    }

    nk_u8_t exp_field = (nk_u8_t)(exp + 7);
    nk_u8_t mant_field = (nk_u8_t)(significand_rounded & 0x07u);
    *dest = (nk_e4m3_t)(sign | (exp_field << 3) | mant_field);
}

NK_INTERNAL void nk_e5m2_to_f32_(nk_e5m2_t const *src, nk_f32_t *dest) {
    nk_u8_t raw = *src;
    nk_u32_t sign = (nk_u32_t)(raw & 0x80) << 24;
    nk_u32_t exponent = (raw >> 2) & 0x1Fu;
    nk_u32_t mantissa = raw & 0x03u;
    nk_fui32_t conv;

    if (exponent == 0) {
        if (mantissa == 0) {
            conv.u = sign;
            *dest = conv.f;
            return;
        }
        nk_f32_t value = (nk_f32_t)mantissa * (1.0f / 65536.0f);
        *dest = sign ? -value : value;
        return;
    }
    if (exponent == 0x1Fu) {
        if (mantissa == 0) { conv.u = sign | 0x7F800000u; }
        else { conv.u = sign | 0x7FC00000u; }
        *dest = conv.f;
        return;
    }

    nk_u32_t f32_exponent = (exponent + 112u) << 23;
    nk_u32_t f32_mantissa = mantissa << 21;
    conv.u = sign | f32_exponent | f32_mantissa;
    *dest = conv.f;
}

NK_INTERNAL void nk_f32_to_e5m2_(nk_f32_t const *src, nk_e5m2_t *dest) {
    nk_f32_t x = *src;
    nk_fui32_t conv;
    conv.f = x;
    nk_u32_t sign_bit = conv.u >> 31;
    nk_u32_t abs_bits = conv.u & 0x7FFFFFFFu;
    nk_u8_t sign = (nk_u8_t)(sign_bit << 7);

    if (abs_bits >= 0x7F800000u) {
        nk_u8_t mant = (abs_bits > 0x7F800000u) ? 0x01u : 0x00u;
        *dest = (nk_e5m2_t)(sign | 0x7Cu | mant);
        return;
    }

    if (abs_bits == 0) {
        *dest = (nk_e5m2_t)sign;
        return;
    }

    nk_f32_t abs_x = sign_bit ? -x : x;

    if (abs_x < (1.0f / 65536.0f)) {
        *dest = (nk_e5m2_t)sign;
        return;
    }

    if (abs_x < (1.0f / 16384.0f)) {
        nk_f32_t scaled = abs_x * 65536.0f;
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        if (mant > 3) { mant = 3; }
        if (mant == 0) { *dest = (nk_e5m2_t)sign; }
        else { *dest = (nk_e5m2_t)(sign | (nk_u8_t)mant); }
        return;
    }

    nk_i32_t exp = (nk_i32_t)((abs_bits >> 23) & 0xFFu) - 127;
    nk_u32_t mantissa = abs_bits & 0x7FFFFFu;
    nk_u32_t significand = (1u << 23) | mantissa;
    nk_i32_t shift = 23 - 2;
    nk_u32_t remainder_mask = (1u << shift) - 1;
    nk_u32_t remainder = significand & remainder_mask;
    nk_u32_t halfway = 1u << (shift - 1);
    nk_u32_t significand_rounded = significand >> shift;
    if (remainder > halfway || (remainder == halfway && (significand_rounded & 1))) { ++significand_rounded; }
    if (significand_rounded == (1u << (2 + 1))) {
        significand_rounded >>= 1;
        ++exp;
    }
    if (exp > 15) {
        *dest = (nk_e5m2_t)(sign | 0x7Cu);
        return;
    }
    if (exp < -14) {
        nk_f32_t scaled = abs_x * 65536.0f;
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        if (mant > 3) { mant = 3; }
        if (mant == 0) { *dest = (nk_e5m2_t)sign; }
        else { *dest = (nk_e5m2_t)(sign | (nk_u8_t)mant); }
        return;
    }

    nk_u8_t exp_field = (nk_u8_t)(exp + 15);
    nk_u8_t mant_field = (nk_u8_t)(significand_rounded & 0x03u);
    *dest = (nk_e5m2_t)(sign | (exp_field << 2) | mant_field);
}

#if NK_DYNAMIC_DISPATCH
NK_DYNAMIC void nk_f16_to_f32(nk_f16_t const *src, nk_f32_t *dest);
NK_DYNAMIC void nk_f16_to_f64(nk_f16_t const *src, nk_f64_t *dest);
NK_DYNAMIC void nk_bf16_to_f32(nk_bf16_t const *src, nk_f32_t *dest);
NK_DYNAMIC void nk_bf16_to_f64(nk_bf16_t const *src, nk_f64_t *dest);
NK_DYNAMIC void nk_f32_to_f16(nk_f32_t const *src, nk_f16_t *dest);
NK_DYNAMIC void nk_f32_to_bf16(nk_f32_t const *src, nk_bf16_t *dest);
NK_DYNAMIC void nk_e4m3_to_f32(nk_e4m3_t const *src, nk_f32_t *dest);
NK_DYNAMIC void nk_f32_to_e4m3(nk_f32_t const *src, nk_e4m3_t *dest);
NK_DYNAMIC void nk_e5m2_to_f32(nk_e5m2_t const *src, nk_f32_t *dest);
NK_DYNAMIC void nk_f32_to_e5m2(nk_f32_t const *src, nk_e5m2_t *dest);
#else
NK_PUBLIC void nk_f16_to_f32(nk_f16_t const *src, nk_f32_t *dest);
NK_PUBLIC void nk_f16_to_f64(nk_f16_t const *src, nk_f64_t *dest);
NK_PUBLIC void nk_bf16_to_f32(nk_bf16_t const *src, nk_f32_t *dest);
NK_PUBLIC void nk_bf16_to_f64(nk_bf16_t const *src, nk_f64_t *dest);
NK_PUBLIC void nk_f32_to_f16(nk_f32_t const *src, nk_f16_t *dest);
NK_PUBLIC void nk_f32_to_bf16(nk_f32_t const *src, nk_bf16_t *dest);
NK_PUBLIC void nk_e4m3_to_f32(nk_e4m3_t const *src, nk_f32_t *dest);
NK_PUBLIC void nk_f32_to_e4m3(nk_f32_t const *src, nk_e4m3_t *dest);
NK_PUBLIC void nk_e5m2_to_f32(nk_e5m2_t const *src, nk_f32_t *dest);
NK_PUBLIC void nk_f32_to_e5m2(nk_f32_t const *src, nk_e5m2_t *dest);
#endif

NK_INTERNAL void nk_f16_to_f64_(nk_f16_t const *x, nk_f64_t *y) {
    nk_f32_t f32;
    nk_f16_to_f32(x, &f32);
    *y = (nk_f64_t)f32;
}
NK_INTERNAL void nk_f64_to_f16_(nk_f64_t const *x, nk_f16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_f16(&f32, y);
}
NK_INTERNAL void nk_bf16_to_f64_(nk_bf16_t const *x, nk_f64_t *y) {
    nk_f32_t f32;
    nk_bf16_to_f32(x, &f32);
    *y = (nk_f64_t)f32;
}
NK_INTERNAL void nk_f64_to_bf16_(nk_f64_t const *x, nk_bf16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_bf16(&f32, y);
}

/*  Convert floating pointer numbers to integers, clamping them to the range of signed
 *  and unsigned low-resolution integers, and rounding them to the nearest integer.
 *
 *  In C++ the analogous solution with STL could be: `*y = std::clamp(std::round(*x), -128, 127)`.
 *  In C, using the standard library: `*x = fminf(fmaxf(roundf(*x), -128), 127)`.
 */
NK_INTERNAL void nk_f32_to_i8_(nk_f32_t const *x, nk_i8_t *y) {
    *y = (nk_i8_t)(*x > 127 ? 127 : (*x < -128 ? -128 : (int)(*x + (*x < 0 ? -0.5f : 0.5f))));
}

NK_INTERNAL void nk_f32_to_u8_(nk_f32_t const *x, nk_u8_t *y) {
    *y = (nk_u8_t)(*x > 255 ? 255 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5f : 0.5f))));
}

NK_INTERNAL void nk_f32_to_i16_(nk_f32_t const *x, nk_i16_t *y) {
    *y = (nk_i16_t)(*x > 32767 ? 32767 : (*x < -32768 ? -32768 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f32_to_u16_(nk_f32_t const *x, nk_u16_t *y) {
    *y = (nk_u16_t)(*x > 65535 ? 65535 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_i8_(nk_f64_t const *x, nk_i8_t *y) {
    *y = (nk_i8_t)(*x > 127 ? 127 : (*x < -128 ? -128 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_u8_(nk_f64_t const *x, nk_u8_t *y) {
    *y = (nk_u8_t)(*x > 255 ? 255 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_i16_(nk_f64_t const *x, nk_i16_t *y) {
    *y = (nk_i16_t)(*x > 32767 ? 32767 : (*x < -32768 ? -32768 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_u16_(nk_f64_t const *x, nk_u16_t *y) {
    *y = (nk_u16_t)(*x > 65535 ? 65535 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_i32_(nk_f64_t const *x, nk_i32_t *y) {
    *y = (nk_i32_t)(*x > 2147483647 ? 2147483647
                                    : (*x < -2147483648 ? -2147483648 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_u32_(nk_f64_t const *x, nk_u32_t *y) {
    *y = (nk_u32_t)(*x > 4294967295 ? 4294967295 : (*x < 0 ? 0 : (unsigned int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_i64_(nk_f64_t const *x, nk_i64_t *y) {
    *y = (nk_i64_t)(*x > 9223372036854775807.0
                        ? 9223372036854775807ll
                        : (*x < -9223372036854775808.0 ? (-9223372036854775807ll - 1ll)
                                                       : (long long)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_u64_(nk_f64_t const *x, nk_u64_t *y) {
    *y = (nk_u64_t)(*x > 18446744073709551615.0 ? 18446744073709551615ull
                                                : (*x < 0 ? 0 : (unsigned long long)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_i64_to_i8_(nk_i64_t const *x, nk_i8_t *y) {
    *y = (nk_i8_t)(*x > 127ll ? 127ll : (*x < -128ll ? -128ll : *x));
}

NK_INTERNAL void nk_i64_to_u8_(nk_i64_t const *x, nk_u8_t *y) {
    *y = (nk_u8_t)(*x > 255ll ? 255ll : (*x < 0ll ? 0ll : *x));
}

NK_INTERNAL void nk_i64_to_i16_(nk_i64_t const *x, nk_i16_t *y) {
    *y = (nk_i16_t)(*x > 32767ll ? 32767ll : (*x < -32768ll ? -32768ll : *x));
}

NK_INTERNAL void nk_i64_to_u16_(nk_i64_t const *x, nk_u16_t *y) {
    *y = (nk_u16_t)(*x > 65535ll ? 65535ll : (*x < 0ll ? 0ll : *x));
}

NK_INTERNAL void nk_i64_to_i32_(nk_i64_t const *x, nk_i32_t *y) {
    *y = (nk_i32_t)(*x > 2147483647ll ? 2147483647ll : (*x < -2147483648ll ? -2147483648ll : *x));
}

NK_INTERNAL void nk_i64_to_u32_(nk_i64_t const *x, nk_u32_t *y) {
    *y = (nk_u32_t)(*x > 4294967295ll ? 4294967295ll : (*x < 0ll ? 0ll : *x));
}

NK_INTERNAL void nk_u64_to_i8_(nk_u64_t const *x, nk_i8_t *y) { *y = (nk_i8_t)(*x > 127ull ? 127ull : *x); }
NK_INTERNAL void nk_u64_to_u8_(nk_u64_t const *x, nk_u8_t *y) { *y = (nk_u8_t)(*x > 255ull ? 255ull : *x); }
NK_INTERNAL void nk_u64_to_i16_(nk_u64_t const *x, nk_i16_t *y) { *y = (nk_i16_t)(*x > 32767ull ? 32767ull : *x); }
NK_INTERNAL void nk_u64_to_u16_(nk_u64_t const *x, nk_u16_t *y) { *y = (nk_u16_t)(*x > 65535ull ? 65535ull : *x); }

NK_INTERNAL void nk_u64_to_i32_(nk_u64_t const *x, nk_i32_t *y) {
    *y = (nk_i32_t)(*x > 2147483647ull ? 2147483647ull : *x);
}

NK_INTERNAL void nk_u64_to_u32_(nk_u64_t const *x, nk_u32_t *y) {
    *y = (nk_u32_t)(*x > 4294967295ull ? 4294967295ull : *x);
}

NK_INTERNAL void nk_f64_to_f32_(nk_f64_t const *x, nk_f32_t *y) { *y = (nk_f32_t)*x; }
NK_INTERNAL void nk_u64_to_f32_(nk_u64_t const *x, nk_f32_t *y) { *y = (nk_f32_t)*x; }
NK_INTERNAL void nk_i64_to_f32_(nk_i64_t const *x, nk_f32_t *y) { *y = (nk_f32_t)*x; }

NK_INTERNAL void nk_f32_to_f64_(nk_f32_t const *x, nk_f64_t *y) { *y = (nk_f64_t)*x; }
NK_INTERNAL void nk_f64_to_f64_(nk_f64_t const *x, nk_f64_t *y) { *y = *x; }

NK_INTERNAL void nk_i8_to_f64_(nk_i8_t const *x, nk_f64_t *y) { *y = (nk_f64_t)*x; }
NK_INTERNAL void nk_i16_to_f64_(nk_i16_t const *x, nk_f64_t *y) { *y = (nk_f64_t)*x; }
NK_INTERNAL void nk_i32_to_f64_(nk_i32_t const *x, nk_f64_t *y) { *y = (nk_f64_t)*x; }
NK_INTERNAL void nk_i64_to_f64_(nk_i64_t const *x, nk_f64_t *y) { *y = (nk_f64_t)*x; }
NK_INTERNAL void nk_u8_to_f64_(nk_u8_t const *x, nk_f64_t *y) { *y = (nk_f64_t)*x; }
NK_INTERNAL void nk_u16_to_f64_(nk_u16_t const *x, nk_f64_t *y) { *y = (nk_f64_t)*x; }
NK_INTERNAL void nk_u32_to_f64_(nk_u32_t const *x, nk_f64_t *y) { *y = (nk_f64_t)*x; }
NK_INTERNAL void nk_u64_to_f64_(nk_u64_t const *x, nk_f64_t *y) { *y = (nk_f64_t)*x; }

NK_INTERNAL void nk_i8_to_i64_(nk_i8_t const *x, nk_i64_t *y) { *y = (nk_i64_t)*x; }
NK_INTERNAL void nk_i16_to_i64_(nk_i16_t const *x, nk_i64_t *y) { *y = (nk_i64_t)*x; }
NK_INTERNAL void nk_i32_to_i64_(nk_i32_t const *x, nk_i64_t *y) { *y = (nk_i64_t)*x; }
NK_INTERNAL void nk_i64_to_i64_(nk_i64_t const *x, nk_i64_t *y) { *y = *x; }
NK_INTERNAL void nk_u8_to_i64_(nk_u8_t const *x, nk_i64_t *y) { *y = (nk_i64_t)*x; }
NK_INTERNAL void nk_u16_to_i64_(nk_u16_t const *x, nk_i64_t *y) { *y = (nk_i64_t)*x; }
NK_INTERNAL void nk_u32_to_i64_(nk_u32_t const *x, nk_i64_t *y) { *y = (nk_i64_t)*x; }
NK_INTERNAL void nk_u64_to_i64_(nk_u64_t const *x, nk_i64_t *y) {
    *y = (nk_i64_t)(*x >= 9223372036854775807ull ? 9223372036854775807ll : *x);
}

NK_INTERNAL void nk_i8_to_u64_(nk_i8_t const *x, nk_u64_t *y) { *y = (nk_u64_t)(*x < 0 ? 0 : *x); }
NK_INTERNAL void nk_i16_to_u64_(nk_i16_t const *x, nk_u64_t *y) { *y = (nk_u64_t)(*x < 0 ? 0 : *x); }
NK_INTERNAL void nk_i32_to_u64_(nk_i32_t const *x, nk_u64_t *y) { *y = (nk_u64_t)(*x < 0 ? 0 : *x); }
NK_INTERNAL void nk_i64_to_u64_(nk_i64_t const *x, nk_u64_t *y) { *y = (nk_u64_t)(*x < 0 ? 0 : *x); }
NK_INTERNAL void nk_u8_to_u64_(nk_u8_t const *x, nk_u64_t *y) { *y = (nk_u64_t)*x; }
NK_INTERNAL void nk_u16_to_u64_(nk_u16_t const *x, nk_u64_t *y) { *y = (nk_u64_t)*x; }
NK_INTERNAL void nk_u32_to_u64_(nk_u32_t const *x, nk_u64_t *y) { *y = (nk_u64_t)*x; }
NK_INTERNAL void nk_u64_to_u64_(nk_u64_t const *x, nk_u64_t *y) { *y = *x; }

NK_INTERNAL void nk_i64_to_f16_(nk_i64_t const *x, nk_f16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_f16(&f32, y);
}
NK_INTERNAL void nk_i64_to_bf16_(nk_i64_t const *x, nk_bf16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_bf16(&f32, y);
}
NK_INTERNAL void nk_u64_to_f16_(nk_u64_t const *x, nk_f16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_f16(&f32, y);
}
NK_INTERNAL void nk_u64_to_bf16_(nk_u64_t const *x, nk_bf16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_bf16(&f32, y);
}

/**
 *  @brief  Union for type-punned scalar values at language binding boundaries.
 *
 *  Used to bridge different type systems (Python, JavaScript, etc.) where
 *  scalars arrive as f64 but need to be passed to kernels as typed pointers.
 *  The caller fills the appropriate union member based on the target dtype,
 *  then passes the union address as `void const *` to kernel functions.
 */
typedef union nk_scalar_buffer_t {
    nk_u8_t bytes[8];
    nk_f64_t f64;
    nk_f32_t f32;
    nk_f16_t f16;
    nk_bf16_t bf16;
    nk_i64_t i64;
    nk_u64_t u64;
    nk_i32_t i32;
    nk_u32_t u32;
    nk_i16_t i16;
    nk_u16_t u16;
    nk_i8_t i8;
    nk_u8_t u8;
} nk_scalar_buffer_t;

/**
 *  @brief  Fill scalar buffer from f64, converting to the appropriate kernel parameter type.
 *
 *  For elementwise operations (scale, wsum, fma), the scalar parameter type depends on
 *  the input datatype: f64/i32/u32/i64/u64 use f64, all others use f32.
 *
 *  @param[out] buf     Pointer to the scalar buffer to fill.
 *  @param[in]  value   The f64 value to convert.
 *  @param[in]  dtype   The target datatype that determines the conversion.
 */
NK_INTERNAL void nk_scalar_buffer_set_f64(nk_scalar_buffer_t *buf, nk_f64_t value, nk_datatype_t dtype) {
    switch (dtype) {
    case nk_f64_k:
    case nk_i32_k:
    case nk_u32_k:
    case nk_i64_k:
    case nk_u64_k: buf->f64 = value; break;
    default: buf->f32 = (nk_f32_t)value; break;
    }
}

/**
 *  @brief  Helper structure for implementing strided matrix row lookups, with @b single-byte-level pointer math.
 */
NK_INTERNAL void *nk_advance_by_bytes_(void *ptr, nk_size_t bytes) { return (void *)((nk_u8_t *)ptr + bytes); }

/**
 *  @brief  Divide and round up to the nearest integer.
 */
NK_INTERNAL nk_size_t nk_divide_ceil_(nk_size_t dividend, nk_size_t divisor) {
    return (dividend + divisor - 1) / divisor;
}

/**
 *  @brief Advances the Multi-Dimensional iterator to the next set of indicies.
 *  @param[in] extents The extents of the tensor, defined by an array of at least `rank` scalars.
 *  @param[in] strides The @b signed strides of the tensor in bytes, defined by an array of at least `rank` scalars.
 *  @param[in] rank The number of dimensions in the tensor (its rank).
 *  @param[inout] coordinates The array of offsets along each of `rank` dimensions, which will be updated.
 *  @param[inout] byte_offset The @b signed byte offset of the current element, which will be advanced.
 *  @return 1 if the iterator was successfully advanced, 0 if the end of iteration was reached.
 *
 *  For flexibility, the API is decoupled from from the `nk_xd_index_t` structure, and
 *  can be used on any-rank tensors, independent of the `NK_NDARRAY_MAX_RANK` constant.
 */
NK_PUBLIC int nk_xd_index_next(                                          //
    nk_size_t const *extents, nk_ssize_t const *strides, nk_size_t rank, //
    nk_size_t *coordinates, nk_ssize_t *byte_offset) {
    // Start from last dimension and move backward
    for (nk_size_t i = rank; i-- > 0;) {
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
 *  @param[in] extents The extents of the tensor, defined by an array of at least `rank` scalars.
 *  @param[in] strides The @b signed strides of the tensor in bytes, defined by an array of at least `rank` scalars.
 *  @param[in] rank The number of dimensions in the tensor (its rank).
 *  @param[in] coordinates The array of offsets along each of `rank` dimensions, which will be updated.
 *  @param[out] byte_offset The byte offset of the current element, which will be advanced.
 *  @return 1 if the offset was successfully advanced, 0 if the end of iteration was reached.
 */
NK_PUBLIC int nk_xd_index_linearize(                                     //
    nk_size_t const *extents, nk_ssize_t const *strides, nk_size_t rank, //
    nk_size_t const *coordinates, nk_ssize_t *byte_offset) {

    nk_ssize_t result = 0;
    for (nk_size_t i = 0; i < rank; i++) {
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
typedef struct nk_xd_index_t {
    nk_size_t coordinates[NK_NDARRAY_MAX_RANK]; // Coordinate offsets along each dimension
    nk_ssize_t byte_offset;                     // Byte offset of the current element
} nk_xd_index_t;

NK_PUBLIC void nk_xd_index_init(nk_xd_index_t *xd_index) {
    for (nk_size_t i = 0; i < NK_NDARRAY_MAX_RANK; i++) xd_index->coordinates[i] = 0;
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
 *  Most NumKong algorithms don't work with the entire structure, but expect the fields to be passed separately.
 *  It would also require storing the @b start-pointer and the @b datatype/item-size separately, as it's not
 *  stored inside the structure.
 */
typedef struct nk_xd_span_t {
    nk_size_t extents[NK_NDARRAY_MAX_RANK];  /// Number of elements along each dimension
    nk_ssize_t strides[NK_NDARRAY_MAX_RANK]; /// Strides of the tensor in bytes
    nk_size_t rank;                          /// Number of dimensions in the tensor
} nk_xd_span_t;

NK_PUBLIC void nk_xd_span_init(nk_xd_span_t *xd_span) {
    for (nk_size_t i = 0; i < NK_NDARRAY_MAX_RANK; i++) xd_span->extents[i] = 0, xd_span->strides[i] = 0;
    xd_span->rank = 0;
}

NK_INTERNAL nk_u32_t nk_u32_rol(nk_u32_t x, int n) { return (x << n) | (x >> (32 - n)); }
NK_INTERNAL nk_u16_t nk_u16_rol(nk_u16_t x, int n) { return (x << n) | (x >> (16 - n)); }
NK_INTERNAL nk_u8_t nk_u8_rol(nk_u8_t x, int n) { return (x << n) | (x >> (8 - n)); }
NK_INTERNAL nk_u32_t nk_u32_ror(nk_u32_t x, int n) { return (x >> n) | (x << (32 - n)); }
NK_INTERNAL nk_u16_t nk_u16_ror(nk_u16_t x, int n) { return (x >> n) | (x << (16 - n)); }
NK_INTERNAL nk_u8_t nk_u8_ror(nk_u8_t x, int n) { return (x >> n) | (x << (8 - n)); }

NK_INTERNAL void nk_u8_sadd_(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t *r) {
    nk_u16_t result = (nk_u16_t)*a + (nk_u16_t)*b;
    *r = (result > 255u) ? (nk_u8_t)255u : (nk_u8_t)result;
}
NK_INTERNAL void nk_u16_sadd_(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t *r) {
    nk_u32_t result = (nk_u32_t)*a + (nk_u32_t)*b;
    *r = (result > 65535u) ? (nk_u16_t)65535u : (nk_u16_t)result;
}
NK_INTERNAL void nk_u32_sadd_(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t *r) {
    nk_u64_t result = (nk_u64_t)*a + (nk_u64_t)*b;
    *r = (result > 4294967295u) ? (nk_u32_t)4294967295u : (nk_u32_t)result;
}
NK_INTERNAL void nk_u64_sadd_(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t *r) {
    *r = (*a + *b < *a) ? 18446744073709551615ull : (*a + *b);
}
NK_INTERNAL void nk_i8_sadd_(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t *r) {
    nk_i16_t result = (nk_i16_t)*a + (nk_i16_t)*b;
    *r = (result > 127) ? 127 : (result < -128 ? -128 : result);
}
NK_INTERNAL void nk_i16_sadd_(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t *r) {
    nk_i32_t result = (nk_i32_t)*a + (nk_i32_t)*b;
    *r = (result > 32767) ? 32767 : (result < -32768 ? -32768 : result);
}
NK_INTERNAL void nk_i32_sadd_(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t *r) {
    nk_i64_t result = (nk_i64_t)*a + (nk_i64_t)*b;
    *r = (result > 2147483647ll) ? 2147483647ll : (result < -2147483648ll ? -2147483648ll : (nk_i32_t)result);
}
NK_INTERNAL void nk_i64_sadd_(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t *r) {
    //? We can't just write `-9223372036854775808ll`, even though it's the smallest signed 64-bit value.
    //? The compiler will complain about the number being too large for the type, as it will process the
    //? constant and the sign separately. So we use the same hint that compilers use to define the `INT64_MIN`.
    if ((*b > 0) && (*a > (9223372036854775807ll) - *b)) { *r = 9223372036854775807ll; }                    // Overflow
    else if ((*b < 0) && (*a < (-9223372036854775807ll - 1ll) - *b)) { *r = -9223372036854775807ll - 1ll; } // Underflow
    else { *r = *a + *b; }
}
NK_INTERNAL void nk_f32_sadd_(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t *r) { *r = *a + *b; }
NK_INTERNAL void nk_f64_sadd_(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t *r) { *r = *a + *b; }
NK_INTERNAL void nk_f16_sadd_(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t *r) {
    nk_f32_t a_f32, b_f32, r_f32;
    nk_f16_to_f32(a, &a_f32);
    nk_f16_to_f32(b, &b_f32);
    r_f32 = a_f32 + b_f32;
    nk_f32_to_f16(&r_f32, r);
}
NK_INTERNAL void nk_bf16_sadd_(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t *r) {
    nk_f32_t a_f32, b_f32, r_f32;
    nk_bf16_to_f32(a, &a_f32);
    nk_bf16_to_f32(b, &b_f32);
    r_f32 = a_f32 + b_f32;
    nk_f32_to_bf16(&r_f32, r);
}

NK_INTERNAL void nk_u8_smul_(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t *r) {
    nk_u16_t result = (nk_u16_t)(*a) * (nk_u16_t)(*b);
    *r = (result > 255) ? 255 : (nk_u8_t)result;
}

NK_INTERNAL void nk_u16_smul_(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t *r) {
    nk_u32_t result = (nk_u32_t)(*a) * (nk_u32_t)(*b);
    *r = (result > 65535) ? 65535 : (nk_u16_t)result;
}

NK_INTERNAL void nk_u32_smul_(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t *r) {
    nk_u64_t result = (nk_u64_t)(*a) * (nk_u64_t)(*b);
    *r = (result > 4294967295u) ? 4294967295u : (nk_u32_t)result;
}

NK_INTERNAL void nk_u64_smul_(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t *r) {
    // Split the inputs into high and low 32-bit parts
    nk_u64_t a_hi = *a >> 32;
    nk_u64_t a_lo = *a & 0xFFFFFFFF;
    nk_u64_t b_hi = *b >> 32;
    nk_u64_t b_lo = *b & 0xFFFFFFFF;

    // Compute partial products
    nk_u64_t hi_hi = a_hi * b_hi;
    nk_u64_t hi_lo = a_hi * b_lo;
    nk_u64_t lo_hi = a_lo * b_hi;
    nk_u64_t lo_lo = a_lo * b_lo;

    // Check if the high part of the result overflows
    if (hi_hi || (hi_lo >> 32) || (lo_hi >> 32) || ((hi_lo + lo_hi) >> 32)) { *r = 18446744073709551615ull; }
    else { *r = (hi_lo << 32) + (lo_hi << 32) + lo_lo; } // Combine parts if no overflow
}

/**
 *  @brief  SWAR population count for 64-bit integers.
 *
 *  Classic algorithm from Hacker's Delight using parallel bit summation:
 *  - Step 1: Count bits in pairs (2-bit sums)
 *  - Step 2: Count bits in nibbles (4-bit sums)
 *  - Step 3: Count bits in bytes (8-bit sums)
 *  - Step 4: Horizontal sum via multiply - each byte contributes to bits 56-63
 *
 *  Cost: ~12 ALU ops, zero memory access (vs 8 table lookups for byte-wise).
 */
NK_INTERNAL nk_u64_t nk_u64_popcount_(nk_u64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full;
    return (x * 0x0101010101010101ull) >> 56;
}

NK_INTERNAL void nk_i8_smul_(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t *r) {
    nk_i16_t result = (nk_i16_t)(*a) * (nk_i16_t)(*b);
    *r = (result > 127) ? 127 : (result < -128 ? -128 : (nk_i8_t)result);
}

NK_INTERNAL void nk_i16_smul_(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t *r) {
    nk_i32_t result = (nk_i32_t)(*a) * (nk_i32_t)(*b);
    *r = (result > 32767) ? 32767 : (result < -32768 ? -32768 : (nk_i16_t)result);
}

NK_INTERNAL void nk_i32_smul_(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t *r) {
    nk_i64_t result = (nk_i64_t)(*a) * (nk_i64_t)(*b);
    *r = (result > 2147483647ll) ? 2147483647ll : (result < -2147483648ll ? -2147483648ll : (nk_i32_t)result);
}

NK_INTERNAL void nk_i64_smul_(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t *r) {
    int sign = ((*a < 0) ^ (*b < 0)) ? -1 : 1; // Track the sign of the result

    // Take absolute values for easy multiplication and overflow detection
    nk_u64_t abs_a = (*a < 0) ? -*a : *a;
    nk_u64_t abs_b = (*b < 0) ? -*b : *b;

    // Split the absolute values into high and low 32-bit parts
    nk_u64_t a_hi = abs_a >> 32;
    nk_u64_t a_lo = abs_a & 0xFFFFFFFF;
    nk_u64_t b_hi = abs_b >> 32;
    nk_u64_t b_lo = abs_b & 0xFFFFFFFF;

    // Compute partial products
    nk_u64_t hi_hi = a_hi * b_hi;
    nk_u64_t hi_lo = a_hi * b_lo;
    nk_u64_t lo_hi = a_lo * b_hi;
    nk_u64_t lo_lo = a_lo * b_lo;

    // Check for overflow and saturate based on sign
    if (hi_hi || (hi_lo >> 32) || (lo_hi >> 32) || ((hi_lo + lo_hi) >> 32)) {
        *r = (sign > 0) ? 9223372036854775807ll : (-9223372036854775807ll - 1ll);
    }
    // Combine parts if no overflow, then apply the sign
    else {
        nk_u64_t result = (hi_lo << 32) + (lo_hi << 32) + lo_lo;
        *r = (sign < 0) ? -((nk_i64_t)result) : (nk_i64_t)result;
    }
}

NK_INTERNAL void nk_f32_smul_(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t *r) { *r = *a * *b; }
NK_INTERNAL void nk_f64_smul_(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t *r) { *r = *a * *b; }

NK_INTERNAL void nk_f16_smul_(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t *r) {
    nk_f32_t a_f32, b_f32, r_f32;
    nk_f16_to_f32(a, &a_f32);
    nk_f16_to_f32(b, &b_f32);
    r_f32 = a_f32 * b_f32;
    nk_f32_to_f16(&r_f32, r);
}

NK_INTERNAL void nk_bf16_smul_(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t *r) {
    nk_f32_t a_f32, b_f32, r_f32;
    nk_bf16_to_f32(a, &a_f32);
    nk_bf16_to_f32(b, &b_f32);
    r_f32 = a_f32 * b_f32;
    nk_f32_to_bf16(&r_f32, r);
}

#if !NK_DYNAMIC_DISPATCH
NK_PUBLIC void nk_f16_to_f32(nk_f16_t const *src, nk_f32_t *dest) { nk_f16_to_f32_(src, dest); }
NK_PUBLIC void nk_f16_to_f64(nk_f16_t const *src, nk_f64_t *dest) { nk_f16_to_f64_(src, dest); }
NK_PUBLIC void nk_f32_to_f16(nk_f32_t const *src, nk_f16_t *dest) { nk_f32_to_f16_(src, dest); }
NK_PUBLIC void nk_bf16_to_f32(nk_bf16_t const *src, nk_f32_t *dest) { nk_bf16_to_f32_(src, dest); }
NK_PUBLIC void nk_bf16_to_f64(nk_bf16_t const *src, nk_f64_t *dest) { nk_bf16_to_f64_(src, dest); }
NK_PUBLIC void nk_f32_to_bf16(nk_f32_t const *src, nk_bf16_t *dest) { nk_f32_to_bf16_(src, dest); }
NK_PUBLIC void nk_e4m3_to_f32(nk_e4m3_t const *src, nk_f32_t *dest) { nk_e4m3_to_f32_(src, dest); }
NK_PUBLIC void nk_f32_to_e4m3(nk_f32_t const *src, nk_e4m3_t *dest) { nk_f32_to_e4m3_(src, dest); }
NK_PUBLIC void nk_e5m2_to_f32(nk_e5m2_t const *src, nk_f32_t *dest) { nk_e5m2_to_f32_(src, dest); }
NK_PUBLIC void nk_f32_to_e5m2(nk_f32_t const *src, nk_e5m2_t *dest) { nk_f32_to_e5m2_(src, dest); }
#endif // NK_DYNAMIC_DISPATCH

#ifdef __cplusplus
} // extern "C"
#endif

#endif
