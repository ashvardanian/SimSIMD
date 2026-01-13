/**
 *  @brief SIMD-accelerated trigonometric element-wise operations, based on SLEEF, optimized for Arm SVE-capable CPUs.
 *  @file include/numkong/trigonometry/svehalf.h
 *  @sa include/numkong/each.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @section trigonometry_svehalf_instructions ARM SVE+FP16 Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      svld1_f16                   LD1H (Z.H, P/Z, [Xn])           4-6cy       2/cy
 *      svst1_f16                   ST1H (Z.H, P, [Xn])             4cy         1/cy
 *      svadd_f16_x                 FADD (Z.H, P/M, Z.H, Z.H)       3cy         2/cy
 *      svsub_f16_x                 FSUB (Z.H, P/M, Z.H, Z.H)       3cy         2/cy
 *      svmul_f16_x                 FMUL (Z.H, P/M, Z.H, Z.H)       4cy         2/cy
 *      svmla_f16_x                 FMLA (Z.H, P/M, Z.H, Z.H)       4cy         2/cy
 *      svdiv_f16_x                 FDIV (Z.H, P/M, Z.H, Z.H)       10-14cy     0.2/cy
 *      svsqrt_f16_x                FSQRT (Z.H, P/M, Z.H)           12-16cy     0.2/cy
 *      svdup_f16                   DUP (Z.H, #imm)                 1cy         2/cy
 *      svwhilelt_b16               WHILELT (P.H, Xn, Xm)           2cy         1/cy
 *      svptrue_b16                 PTRUE (P.H, pattern)            1cy         2/cy
 *      svcnth                      CNTH (Xd)                       1cy         2/cy
 */
#ifndef NK_TRIGONOMETRY_SVEHALF_H
#define NK_TRIGONOMETRY_SVEHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVEHALF
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#endif

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SVEHALF
#endif // NK_TARGET_ARM_

#endif // NK_TRIGONOMETRY_SVEHALF_H
