/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/mesh/neon_f16.h
 *  @sa include/numkong/mesh.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_MESH_NEON_F16_H
#define NK_MESH_NEON_F16_H

#if _NK_TARGET_ARM
#if NK_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON_F16
#endif // _NK_TARGET_ARM

#endif // NK_MESH_NEON_F16_H