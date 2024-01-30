/**
 *  @brief      SIMD-accelerated Similarity Measures and Distance Functions.
 *  @file       simsimd.h
 *  @author     Ash Vardanian
 *  @date       March 14, 2023
 *  @copyright  Copyright (c) 2023
 *
 *  References:
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  Detecting target CPU features at compile time: https://stackoverflow.com/a/28939692/2766161
 */

#ifndef SIMSIMD_H
#define SIMSIMD_H

#define SIMSIMD_VERSION_MAJOR 3
#define SIMSIMD_VERSION_MINOR 7
#define SIMSIMD_VERSION_PATCH 5

#include "binary.h"      // Hamming, Jaccard
#include "probability.h" // Kullback-Leibler, Jensen–Shannon
#include "spatial.h"     // L2, Inner Product, Cosine

#if SIMSIMD_TARGET_ARM
#ifdef __linux__
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief  Enumeration of supported metric kinds.
 */
typedef enum {
    simsimd_metric_unknown_k = 0, ///< Unknown metric kind

    // Classics:
    simsimd_metric_ip_k = 'i',    ///< Inner product
    simsimd_metric_dot_k = 'i',   ///< Inner product alias
    simsimd_metric_inner_k = 'i', ///< Inner product alias

    simsimd_metric_cos_k = 'c',     ///< Cosine similarity
    simsimd_metric_cosine_k = 'c',  ///< Cosine similarity alias
    simsimd_metric_angular_k = 'c', ///< Cosine similarity alias

    simsimd_metric_l2sq_k = 'e',        ///< Squared Euclidean distance
    simsimd_metric_sqeuclidean_k = 'e', ///< Squared Euclidean distance alias

    // Binary:
    simsimd_metric_hamming_k = 'b', ///< Hamming distance
    simsimd_metric_jaccard_k = 'j', ///< Jaccard coefficient

    // Probability:
    simsimd_metric_kl_k = 'k',               ///< Kullback-Leibler divergence
    simsimd_metric_kullback_leibler_k = 'k', ///< Kullback-Leibler divergence alias

    simsimd_metric_js_k = 's',             ///< Jensen-Shannon divergence
    simsimd_metric_jensen_shannon_k = 's', ///< Jensen-Shannon divergence alias

} simsimd_metric_kind_t;

/**
 *  @brief  Enumeration of SIMD capabilities of the target architecture.
 */
typedef enum {
    simsimd_cap_serial_k = 1,       ///< Serial (non-SIMD) capability
    simsimd_cap_any_k = 0xFFFFFFFF, ///< Mask representing any capability

    simsimd_cap_arm_neon_k = 1 << 10, ///< ARM NEON capability
    simsimd_cap_arm_sve_k = 1 << 11,  ///< ARM SVE capability
    simsimd_cap_arm_sve2_k = 1 << 12, ///< ARM SVE2 capability

    simsimd_cap_x86_avx2_k = 1 << 20,            ///< x86 AVX2 capability
    simsimd_cap_x86_avx512_k = 1 << 21,          ///< x86 AVX512 capability
    simsimd_cap_x86_avx2fp16_k = 1 << 22,        ///< x86 AVX2 with FP16 capability
    simsimd_cap_x86_avx512fp16_k = 1 << 23,      ///< x86 AVX512 with FP16 capability
    simsimd_cap_x86_avx512vpopcntdq_k = 1 << 24, ///< x86 AVX512 VPOPCNTDQ instruction capability
    simsimd_cap_x86_avx512vnni_k = 1 << 25,      ///< x86 AVX512 VNNI instruction capability

} simsimd_capability_t;

/**
 *  @brief  Enumeration of supported data types.
 */
typedef enum {
    simsimd_datatype_unknown_k, ///< Unknown data type
    simsimd_datatype_f64_k,     ///< Double precision floating point
    simsimd_datatype_f32_k,     ///< Single precision floating point
    simsimd_datatype_f16_k,     ///< Half precision floating point
    simsimd_datatype_i8_k,      ///< 8-bit integer
    simsimd_datatype_b8_k,      ///< Single-bit values packed into 8-bit words
} simsimd_datatype_t;

/**
 *  @brief  Type-punned function pointer accepting two vectors and outputting their similarity/distance.
 *
 *  @param[in] a Pointer to the first data array.
 *  @param[in] b Pointer to the second data array.
 *  @param[in] size_a Size of the first data array.
 *  @param[in] size_b Size of the second data array.
 *  @return Computed metric as a single-precision floating-point value.
 */
typedef simsimd_f32_t (*simsimd_metric_punned_t)(void const* a, void const* b, simsimd_size_t size_a,
                                                 simsimd_size_t size_b);

/**
 *  @brief  Function to determine the SIMD capabilities of the current machine at @b runtime.
 *  @return A bitmask of the SIMD capabilities represented as a `simsimd_capability_t` enum value.
 */
inline static simsimd_capability_t simsimd_capabilities() {

#if SIMSIMD_TARGET_X86

    /// The states of 4 registers populated for a specific "cpuid" assmebly call
    union four_registers_t {
        int array[4];
        struct separate_t {
            unsigned eax, ebx, ecx, edx;
        } named;
    } info1, info7;

#ifdef _MSC_VER
    __cpuidex(info1.array, 1, 0);
    __cpuidex(info7.array, 7, 0);
#else
    __asm__ __volatile__("cpuid"
                         : "=a"(info1.named.eax), "=b"(info1.named.ebx), "=c"(info1.named.ecx), "=d"(info1.named.edx)
                         : "a"(1), "c"(0));
    __asm__ __volatile__("cpuid"
                         : "=a"(info7.named.eax), "=b"(info7.named.ebx), "=c"(info7.named.ecx), "=d"(info7.named.edx)
                         : "a"(7), "c"(0));
#endif

    // Check for AVX2 (Function ID 7, EBX register)
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L148
    unsigned supports_avx2 = (info7.named.ebx & 0x00000020) != 0;
    // Check for F16C (Function ID 1, ECX register)
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L107
    unsigned supports_f16c = (info1.named.ecx & 0x20000000) != 0;
    // Check for AVX512F (Function ID 7, EBX register)
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L155
    unsigned supports_avx512f = (info7.named.ebx & 0x00010000) != 0;
    // Check for AVX512FP16 (Function ID 7, EDX register)
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L198C9-L198C23
    unsigned supports_avx512fp16 = (info7.named.edx & 0x00800000) != 0;
    // Check for VPOPCNTDQ (Function ID 1, ECX register)
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L182C30-L182C40
    unsigned supports_avx512vpopcntdq = (info1.named.ecx & 0x00004000) != 0;
    // Check for VNNI (Function ID 1, ECX register)
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L180
    unsigned supports_avx512vnni = (info1.named.ecx & 0x00000800) != 0;

    return (simsimd_capability_t)(                                                   //
        (simsimd_cap_x86_avx2_k * supports_avx2) |                                   //
        (simsimd_cap_x86_avx512_k * supports_avx512f) |                              //
        (simsimd_cap_x86_avx2fp16_k * (supports_avx2 && supports_f16c)) |            //
        (simsimd_cap_x86_avx512fp16_k * (supports_avx512fp16 && supports_avx512f)) | //
        (simsimd_cap_x86_avx512vpopcntdq_k * (supports_avx512vpopcntdq)) |           //
        (simsimd_cap_x86_avx512vnni_k * (supports_avx512vnni)) |                     //
        (simsimd_cap_serial_k));

#endif // SIMSIMD_TARGET_X86

#if SIMSIMD_TARGET_ARM

    // Every 64-bit Arm CPU supports NEON
    unsigned supports_neon = 1;
    unsigned supports_sve = 0;
    unsigned supports_sve2 = 0;

#ifdef __linux__
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    supports_sve = (hwcap & HWCAP_SVE) != 0;
    supports_sve2 = (hwcap2 & HWCAP2_SVE2) != 0;
#endif

    return (simsimd_capability_t)(                 //
        (simsimd_cap_arm_neon_k * supports_neon) | //
        (simsimd_cap_arm_sve_k * supports_sve) |   //
        (simsimd_cap_arm_sve2_k * supports_sve2) | //
        (simsimd_cap_serial_k));

#endif // SIMSIMD_TARGET_ARM

    return simsimd_cap_serial_k;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-function-type"

/**
 *  @brief  Determines the best suited metric implementation based on the given datatype,
 *          supported and allowed by hardware capabilities.
 *
 *  @param kind The kind of metric to be evaluated.
 *  @param datatype The data type for which the metric needs to be evaluated.
 *  @param supported The hardware capabilities supported by the CPU.
 *  @param allowed The hardware capabilities allowed for use.
 *  @param metric_output Output variable for the selected similarity function.
 *  @param capability_output Output variable for the utilized hardware capabilities.
 */
inline static void simsimd_find_metric_punned( //
    simsimd_metric_kind_t kind,                //
    simsimd_datatype_t datatype,               //
    simsimd_capability_t supported,            //
    simsimd_capability_t allowed,              //
    simsimd_metric_punned_t* metric_output,    //
    simsimd_capability_t* capability_output) {

    simsimd_metric_punned_t* m = metric_output;
    simsimd_capability_t* c = capability_output;
    simsimd_capability_t viable = (simsimd_capability_t)(supported & allowed);
    *m = (simsimd_metric_punned_t)0;
    *c = (simsimd_capability_t)0;

    // clang-format off
    switch (datatype) {

    case simsimd_datatype_unknown_k: break;

    // Double-precision floating-point vectors
    case simsimd_datatype_f64_k:

    #if SIMSIMD_TARGET_X86_AVX512
        if (viable & simsimd_cap_x86_avx512_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_f64_ip, *c = simsimd_cap_x86_avx512_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_f64_cos, *c = simsimd_cap_x86_avx512_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_f64_l2sq, *c = simsimd_cap_x86_avx512_k; return;
            default: break;
            }
    #endif
        if (viable & simsimd_cap_serial_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f64_ip, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f64_cos, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f64_l2sq, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_js_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f64_js, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_kl_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f64_kl, *c = simsimd_cap_serial_k; return;
            default: break;
            }

        break;

    // Single-precision floating-point vectors
    case simsimd_datatype_f32_k:

    #if SIMSIMD_TARGET_ARM_NEON
        if (viable & simsimd_cap_arm_neon_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_neon_f32_ip, *c = simsimd_cap_arm_neon_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_neon_f32_cos, *c = simsimd_cap_arm_neon_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_neon_f32_l2sq, *c = simsimd_cap_arm_neon_k; return;
            case simsimd_metric_js_k: *m = (simsimd_metric_punned_t)&simsimd_neon_f32_js, *c = simsimd_cap_arm_neon_k; return;
            case simsimd_metric_kl_k: *m = (simsimd_metric_punned_t)&simsimd_neon_f32_kl, *c = simsimd_cap_arm_neon_k; return;
            default: break;
            }
    #endif
    #if SIMSIMD_TARGET_X86_AVX512
        if (viable & simsimd_cap_x86_avx512_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_f32_ip, *c = simsimd_cap_x86_avx512_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_f32_cos, *c = simsimd_cap_x86_avx512_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_f32_l2sq, *c = simsimd_cap_x86_avx512_k; return;
            case simsimd_metric_js_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_f32_js, *c = simsimd_cap_x86_avx512_k; return;
            case simsimd_metric_kl_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_f32_kl, *c = simsimd_cap_x86_avx512_k; return;
            default: break;
            }
    #endif
        if (viable & simsimd_cap_serial_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f32_ip, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f32_cos, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f32_l2sq, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_js_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f32_js, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_kl_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f32_kl, *c = simsimd_cap_serial_k; return;
            default: break;
            }

        break;

    // Half-precision floating-point vectors
    case simsimd_datatype_f16_k:

    #if SIMSIMD_TARGET_ARM_SVE
        if (viable & simsimd_cap_arm_sve_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_sve_f16_ip, *c = simsimd_cap_arm_sve_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_sve_f16_cos, *c = simsimd_cap_arm_sve_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_sve_f16_l2sq, *c = simsimd_cap_arm_sve_k; return;
            default: break;
            }
    #endif
    #if SIMSIMD_TARGET_ARM_NEON
        if (viable & simsimd_cap_arm_neon_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_neon_f16_ip, *c = simsimd_cap_arm_neon_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_neon_f16_cos, *c = simsimd_cap_arm_neon_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_neon_f16_l2sq, *c = simsimd_cap_arm_neon_k; return;
            case simsimd_metric_js_k: *m = (simsimd_metric_punned_t)&simsimd_neon_f16_js, *c = simsimd_cap_arm_neon_k; return;
            case simsimd_metric_kl_k: *m = (simsimd_metric_punned_t)&simsimd_neon_f16_kl, *c = simsimd_cap_arm_neon_k; return;
            default: break;
            }
    #endif
    #if SIMSIMD_TARGET_X86_AVX512
        if (viable & simsimd_cap_x86_avx512fp16_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_f16_ip, *c = simsimd_cap_x86_avx512fp16_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_f16_cos, *c = simsimd_cap_x86_avx512fp16_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_f16_l2sq, *c = simsimd_cap_x86_avx512fp16_k; return;
            case simsimd_metric_js_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_f16_js, *c = simsimd_cap_x86_avx512fp16_k; return;
            case simsimd_metric_kl_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_f16_kl, *c = simsimd_cap_x86_avx512fp16_k; return;
            default: break;
            }
    #endif
    #if SIMSIMD_TARGET_X86_AVX2
        if (viable & simsimd_cap_x86_avx2fp16_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_avx2_f16_ip, *c = simsimd_cap_x86_avx2fp16_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_avx2_f16_cos, *c = simsimd_cap_x86_avx2fp16_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_avx2_f16_l2sq, *c = simsimd_cap_x86_avx2fp16_k; return;
            case simsimd_metric_js_k: *m = (simsimd_metric_punned_t)&simsimd_avx2_f16_js, *c = simsimd_cap_x86_avx2fp16_k; return;
            case simsimd_metric_kl_k: *m = (simsimd_metric_punned_t)&simsimd_avx2_f16_kl, *c = simsimd_cap_x86_avx2fp16_k; return;
            default: break;
            }
    #endif

        if (viable & simsimd_cap_serial_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f16_ip, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f16_cos, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f16_l2sq, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_js_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f16_js, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_kl_k: *m = (simsimd_metric_punned_t)&simsimd_serial_f16_kl, *c = simsimd_cap_serial_k; return;
            default: break;
            }
        
        break;

    // Single-byte integer vectors
    case simsimd_datatype_i8_k:
    #if SIMSIMD_TARGET_ARM_NEON
        if (viable & simsimd_cap_arm_neon_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_neon_i8_ip, *c = simsimd_cap_arm_neon_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_neon_i8_cos, *c = simsimd_cap_arm_neon_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_neon_i8_l2sq, *c = simsimd_cap_arm_neon_k; return;
            default: break;
            }
    #endif
    #if SIMSIMD_TARGET_X86_AVX512
        if (viable & simsimd_cap_x86_avx512vnni_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_i8_ip, *c = simsimd_cap_x86_avx512vnni_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_i8_cos, *c = simsimd_cap_x86_avx512vnni_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_i8_l2sq, *c = simsimd_cap_x86_avx512vnni_k; return;
            default: break;
            }
    #endif
    #if SIMSIMD_TARGET_X86_AVX2
        if (viable & simsimd_cap_x86_avx2_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_avx2_i8_ip, *c = simsimd_cap_x86_avx2_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_avx2_i8_cos, *c = simsimd_cap_x86_avx2_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_avx2_i8_l2sq, *c = simsimd_cap_x86_avx2_k; return;
            default: break;
            }
    #endif

        if (viable & simsimd_cap_serial_k)
            switch (kind) {
            case simsimd_metric_ip_k: *m = (simsimd_metric_punned_t)&simsimd_serial_i8_ip, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_cos_k: *m = (simsimd_metric_punned_t)&simsimd_serial_i8_cos, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_l2sq_k: *m = (simsimd_metric_punned_t)&simsimd_serial_i8_l2sq, *c = simsimd_cap_serial_k; return;
            default: break;
            }
        
        break;

    // Binary vectors
    case simsimd_datatype_b8_k:

    #if SIMSIMD_TARGET_ARM_NEON
        if (viable & simsimd_cap_arm_neon_k)
            switch (kind) {
            case simsimd_metric_hamming_k: *m = (simsimd_metric_punned_t)&simsimd_neon_b8_hamming, *c = simsimd_cap_arm_neon_k; return;
            case simsimd_metric_jaccard_k: *m = (simsimd_metric_punned_t)&simsimd_neon_b8_jaccard, *c = simsimd_cap_arm_neon_k; return;
            default: break;
            }
    #endif
    #if SIMSIMD_TARGET_ARM_SVE
        if (viable & simsimd_cap_arm_sve_k)
            switch (kind) {
            case simsimd_metric_hamming_k: *m = (simsimd_metric_punned_t)&simsimd_sve_b8_hamming, *c = simsimd_cap_arm_sve_k; return;
            case simsimd_metric_jaccard_k: *m = (simsimd_metric_punned_t)&simsimd_sve_b8_jaccard, *c = simsimd_cap_arm_sve_k; return;
            default: break;
            }
    #endif
    #if SIMSIMD_TARGET_X86_AVX512
        if (viable & simsimd_cap_x86_avx512vpopcntdq_k)
            switch (kind) {
            case simsimd_metric_hamming_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_b8_hamming, *c = simsimd_cap_x86_avx512vpopcntdq_k; return;
            case simsimd_metric_jaccard_k: *m = (simsimd_metric_punned_t)&simsimd_avx512_b8_jaccard, *c = simsimd_cap_x86_avx512vpopcntdq_k; return;
            default: break;
            }
    #endif

        if (viable & simsimd_cap_serial_k)
            switch (kind) {
            case simsimd_metric_hamming_k: *m = (simsimd_metric_punned_t)&simsimd_serial_b8_hamming, *c = simsimd_cap_serial_k; return;
            case simsimd_metric_jaccard_k: *m = (simsimd_metric_punned_t)&simsimd_serial_b8_jaccard, *c = simsimd_cap_serial_k; return;
            default: break;
            }
        
        break;
    }
    // clang-format on
}

#pragma clang diagnostic pop
#pragma GCC diagnostic pop

/**
 *  @brief  Selects the most suitable metric implementation based on the given metric kind, datatype,
 *          and allowed capabilities. @b Don't call too often and prefer caching the `simsimd_capabilities()`.
 *
 *  @param kind The kind of metric to be evaluated.
 *  @param datatype The data type for which the metric needs to be evaluated.
 *  @param allowed The hardware capabilities allowed for use.
 *  @return A function pointer to the selected metric implementation.
 */
inline static simsimd_metric_punned_t simsimd_metric_punned( //
    simsimd_metric_kind_t kind,                              //
    simsimd_datatype_t datatype,                             //
    simsimd_capability_t allowed) {

    simsimd_metric_punned_t result = 0;
    simsimd_capability_t c = simsimd_cap_serial_k;
    simsimd_capability_t supported = simsimd_capabilities();
    simsimd_find_metric_punned(kind, datatype, supported, allowed, &result, &c);
    return result;
}

#ifdef __cplusplus
}
#endif

#endif
