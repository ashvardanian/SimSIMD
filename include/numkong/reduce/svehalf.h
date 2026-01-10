/**
 *  @brief SIMD-accelerated horizontal reduction operations for Arm SVE-capable CPUs.
 *  @file include/numkong/reduce/sve.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @section reduce_svehalf_instructions ARM SVE+FP16 Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      svld1_f16                   LD1H (Z.H, P/Z, [Xn])           4-6cy       2/cy
 *      svaddv_f16                  FADDV (H, P, Z.H)               6cy         1/cy
 *      svmaxv_f16                  FMAXV (H, P, Z.H)               6cy         1/cy
 *      svminv_f16                  FMINV (H, P, Z.H)               6cy         1/cy
 *      svmaxnmv_f16                FMAXNMV (H, P, Z.H)             6cy         1/cy
 *      svminnmv_f16                FMINNMV (H, P, Z.H)             6cy         1/cy
 *      svadd_f16_x                 FADD (Z.H, P/M, Z.H, Z.H)       3cy         2/cy
 *      svmax_f16_x                 FMAX (Z.H, P/M, Z.H, Z.H)       3cy         2/cy
 *      svmin_f16_x                 FMIN (Z.H, P/M, Z.H, Z.H)       3cy         2/cy
 *      svdup_f16                   DUP (Z.H, #imm)                 1cy         2/cy
 *      svwhilelt_b16               WHILELT (P.H, Xn, Xm)           2cy         1/cy
 *      svptrue_b16                 PTRUE (P.H, pattern)            1cy         2/cy
 *      svcnth                      CNTH (Xd)                       1cy         2/cy
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  FP16 reductions process twice as many elements per vector, but the horizontal reduction
 *  latency remains constant, making vectorized accumulation even more beneficial.
 */
#ifndef NK_REDUCE_SVEHALF_H
#define NK_REDUCE_SVEHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVEHALF
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#endif

#include "numkong/types.h"
#include "numkong/reduce/serial.h" // `nk_popcount_u1`
#include "numkong/reduce/neon.h"   // `nk_reduce_add_u8x16_neon_`

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

#endif // NK_REDUCE_SVEHALF_H
