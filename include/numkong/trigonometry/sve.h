/**
 *  @brief SIMD-accelerated trigonometric element-wise operations, based on SLEEF, optimized for Arm SVE-capable CPUs.
 *  @file include/numkong/elementwise/sve.h
 *  @sa include/numkong/elementwise.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_TRIGONOMETRY_SVE_H
#define NK_TRIGONOMETRY_SVE_H

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

#endif // NK_TRIGONOMETRY_SVE_H
