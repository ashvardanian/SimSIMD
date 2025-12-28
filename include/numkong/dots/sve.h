/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm SVE-capable CPUs.
 *  @file include/numkong/dots/sve.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOTS_SVE_H
#define NK_DOTS_SVE_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVE
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SVE
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_SVE_H