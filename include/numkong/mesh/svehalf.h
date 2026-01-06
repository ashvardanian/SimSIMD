/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm SVE-capable CPUs.
 *  @file include/numkong/mesh/sve.h
 *  @sa include/numkong/mesh.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_MESH_SVEHALF_H
#define NK_MESH_SVEHALF_H

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

#endif // NK_MESH_SVEHALF_H
