/**
 *  @brief SIMD-accelerated Trigonometric Functions for NEON FP16.
 *  @file include/numkong/trigonometry/neonhalf.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/trigonometry.h
 *  @see https://sleef.org
 *
 *  @section trigonometry_neonhalf_instructions ARM NEON FP16 Instructions (ARMv8.2-FP16)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vld1q_f16                   LD1 (V.8H)                      4cy         2/cy        3/cy
 *      vst1q_f16                   ST1 (V.8H)                      2cy         2/cy        3/cy
 *      vfmaq_f16                   FMLA (V.8H, V.8H, V.8H)         4cy         2/cy        4/cy
 *      vmulq_f16                   FMUL (V.8H, V.8H, V.8H)         3cy         2/cy        4/cy
 *      vaddq_f16                   FADD (V.8H, V.8H, V.8H)         2cy         2/cy        4/cy
 *      vsubq_f16                   FSUB (V.8H, V.8H, V.8H)         2cy         2/cy        4/cy
 *      vcvt_f32_f16                FCVTL (V.4S, V.4H)              4cy         2/cy        4/cy
 *
 *  The ARMv8.2-FP16 extension provides native half-precision trigonometric operations, processing
 *  8 FP16 elements per instruction compared to 4 F32 elements. This doubles throughput for
 *  polynomial-based approximations of sin, cos, tan, and their inverses.
 *
 *  Trigonometric functions are implemented using SLEEF-derived polynomial approximations with
 *  Horner's method, leveraging FMA chains for accuracy and performance. The longer dependency
 *  chains in polynomial evaluation benefit from the 4-cycle FMA latency being hidden by
 *  instruction-level parallelism when processing independent elements.
 */
#ifndef NK_TRIGONOMETRY_NEONHALF_H
#define NK_TRIGONOMETRY_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_
#endif // NK_TRIGONOMETRY_NEONHALF_H
