/**
 *  @brief Collection of Similarity Measures, SIMD-accelerated with SSE, AVX, NEON, SVE.
 *
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  Detecting target CPU features at compile time: https://stackoverflow.com/a/28939692/2766161
 */

#pragma once
#include "types.h"

#include "autovec.h"

#ifdef _MSC_VER
#include <intrin.h>
#define popcount32 __popcnt
#define popcount64 __popcnt64
#else
#define popcount32 __builtin_popcount
#define popcount64 __builtin_popcountll
#endif

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

#if SIMSIMD_TARGET_ARM_NEON
#include "arm_neon_f16.h"
#include "arm_neon_f32.h"
#include "arm_neon_i8.h"
#endif

#if SIMSIMD_TARGET_ARM_SVE
#include "arm_sve_f16.h"
#include "arm_sve_f32.h"
#endif

#if SIMSIMD_TARGET_X86_AVX2
#include "x86_avx2_f16.h"
#include "x86_avx2_i8.h"
#endif

#if SIMSIMD_TARGET_X86_AVX512
#include "x86_avx512_f16.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* simsimd_metric_punned_t;
typedef simsimd_f32_t (*simsimd_metric_variable_t)(void const*, void const*, simsimd_size_t, simsimd_size_t);

typedef enum {
    unknown_k = 0,

    // Classics:
    ip_k = 'i',
    cos_k = 'c',
    l2sq_k = 'e',

    // Sets:
    hamming_k = 'b',
    tanimoto_k = 't',
} simsimd_metric_kind_t;

typedef enum {
    autovec_k,

    arm_neon_k,
    arm_sve_k,

    x86_avx2_k,
    x86_avx512_k,
    x86_avx2fp16_k,
    x86_avx512fp16_k,

    arm_sme_k,
    x86_amx_k,

} simsimd_capability_t;

typedef enum {
    f32_k,
    f16_k,
    i8_k,
    b1_k,
} simsimd_datatype_t;

/**
 *  @brief  Depending on the hardware capabilities of the CPU detected at runtime, the list of pre-compiled metrics,
 *          and list of preferences from the `capability_preferences`, determines the best suited distance function
 *          implementation.
 */
inline static void simsimd_metric_punned(simsimd_metric_punned_t* punned_output, simsimd_metric_kind_t kind,
                                         simsimd_datatype_t datatype, simsimd_capability_t* capability_preferences) {
    *capability_preferences = autovec_k;
}

inline static simsimd_f32_t simsimd_measure_fixed( //
    simsimd_metric_punned_t punned,                //
    void const* first_pointer, void const* second_pointer) {
    simsimd_metric_variable_t metric = (simsimd_metric_variable_t)punned;
    simsimd_size_t first_dimensions = 0;
    simsimd_size_t second_dimensions = 0;
    return metric(first_pointer, second_pointer, first_dimensions, second_dimensions);
}

inline static simsimd_f32_t simsimd_measure_equidimensional( //
    simsimd_metric_punned_t punned,                          //
    void const* first_pointer, void const* second_pointer, simsimd_size_t dimensions) {
    simsimd_metric_variable_t metric = (simsimd_metric_variable_t)punned;
    return metric(first_pointer, second_pointer, dimensions, dimensions);
}

inline static simsimd_f32_t simsimd_measure_variable(      //
    simsimd_metric_punned_t punned,                        //
    void const* first_pointer, void const* second_pointer, //
    simsimd_size_t first_dimensions, simsimd_size_t second_dimensions) {
    simsimd_metric_variable_t metric = (simsimd_metric_variable_t)punned;
    return metric(first_pointer, second_pointer, first_dimensions, second_dimensions);
}

#ifdef __cplusplus
}
#endif