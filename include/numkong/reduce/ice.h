/**
 *  @brief SIMD-accelerated horizontal reduction operations for Intel Ice Lake CPUs.
 *  @file include/numkong/reduce/ice.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_ICE_H
#define NK_REDUCE_ICE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), \
                             apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_ICE
#endif // NK_TARGET_X86_

#endif // NK_REDUCE_ICE_H