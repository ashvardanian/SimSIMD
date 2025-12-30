/**
 *  @brief SIMD-accelerated horizontal reduction operations for Intel Sierra Forest CPUs.
 *  @file include/numkong/reduce/sierra.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_SIERRA_H
#define NK_REDUCE_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA
#pragma GCC push_options
#pragma GCC target("avx2", "bmi2", "avxvnni")
#pragma clang attribute push(__attribute__((target("avx2,bmi2,avxvnni"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_

#endif // NK_REDUCE_SIERRA_H