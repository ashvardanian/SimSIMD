/**
 *  @brief SIMD-accelerated Vector Reductions for SVE.
 *  @file include/numkong/reduce/sve.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/reduce.h
 *
 *  @section reduce_sve_instructions ARM SVE Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      svld1_f32                   LD1W (Z.S, P/Z, [Xn])           4-6cy       2/cy
 *      svaddv_f32                  FADDV (S, P, Z.S)               6cy         1/cy
 *      svmaxv_f32                  FMAXV (S, P, Z.S)               6cy         1/cy
 *      svminv_f32                  FMINV (S, P, Z.S)               6cy         1/cy
 *      svmaxnmv_f32                FMAXNMV (S, P, Z.S)             6cy         1/cy
 *      svminnmv_f32                FMINNMV (S, P, Z.S)             6cy         1/cy
 *      svadd_f32_x                 FADD (Z.S, P/M, Z.S, Z.S)       3cy         2/cy
 *      svmax_f32_x                 FMAX (Z.S, P/M, Z.S, Z.S)       3cy         2/cy
 *      svmin_f32_x                 FMIN (Z.S, P/M, Z.S, Z.S)       3cy         2/cy
 *      svdup_f32                   DUP (Z.S, #imm)                 1cy         2/cy
 *      svwhilelt_b32               WHILELT (P.S, Xn, Xm)           2cy         1/cy
 *      svptrue_b32                 PTRUE (P.S, pattern)            1cy         2/cy
 *      svcntw                      CNTW (Xd)                       1cy         2/cy
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  Horizontal reductions (FADDV, FMAXV, FMINV) have 6-cycle latency and 1/cycle throughput,
 *  making them the bottleneck for reduction operations on short vectors.
 */
#ifndef NK_REDUCE_SVE_H
#define NK_REDUCE_SVE_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

#include "numkong/types.h"
#include "numkong/reduce/serial.h" // `nk_u1x8_popcount_`
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
#endif // NK_TARGET_SVE
#endif // NK_TARGET_ARM_

#endif // NK_REDUCE_SVE_H
