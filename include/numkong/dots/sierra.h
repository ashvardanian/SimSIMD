/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Sierra Forest CPUs.
 *  @file include/numkong/dots/sierra.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOTS_SIERRA_H
#define NK_DOTS_SIERRA_H

#if _NK_TARGET_X86
#if NK_TARGET_SIERRA
#pragma GCC push_options
#pragma GCC target("avx2", "bmi2", "avx2vnni")
#pragma clang attribute push(__attribute__((target("avx2,bmi2,avx2vnni"))), apply_to = function)

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
#endif // _NK_TARGET_X86

#endif // NK_DOTS_SIERRA_H