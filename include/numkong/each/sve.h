/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm SVE-capable CPUs.
 *  @file include/numkong/each/sve.h
 *  @sa include/numkong/each.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @section elementwise_sve_instructions ARM SVE Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      svld1_f32                   LD1W (Z.S, P/Z, [Xn])           4-6cy       2/cy
 *      svst1_f32                   ST1W (Z.S, P, [Xn])             4cy         1/cy
 *      svadd_f32_x                 FADD (Z.S, P/M, Z.S, Z.S)       3cy         2/cy
 *      svsub_f32_x                 FSUB (Z.S, P/M, Z.S, Z.S)       3cy         2/cy
 *      svmul_f32_x                 FMUL (Z.S, P/M, Z.S, Z.S)       4cy         2/cy
 *      svmla_f32_x                 FMLA (Z.S, P/M, Z.S, Z.S)       4cy         2/cy
 *      svdiv_f32_x                 FDIV (Z.S, P/M, Z.S, Z.S)       10-14cy     0.2/cy
 *      svsqrt_f32_x                FSQRT (Z.S, P/M, Z.S)           12-16cy     0.2/cy
 *      svabs_f32_x                 FABS (Z.S, P/M, Z.S)            2cy         2/cy
 *      svneg_f32_x                 FNEG (Z.S, P/M, Z.S)            2cy         2/cy
 *      svdup_f32                   DUP (Z.S, #imm)                 1cy         2/cy
 *      svwhilelt_b32               WHILELT (P.S, Xn, Xm)           2cy         1/cy
 *      svptrue_b32                 PTRUE (P.S, pattern)            1cy         2/cy
 *      svcntw                      CNTW (Xd)                       1cy         2/cy
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  Division and square root are significantly slower (10-16cy) with limited throughput
 *  (0.2/cy), so consider reciprocal approximations for performance-critical code.
 */
#ifndef NK_EACH_SVE_H
#define NK_EACH_SVE_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
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
#endif // NK_TARGET_SVE
#endif // NK_TARGET_ARM_

#endif // NK_EACH_SVE_H
