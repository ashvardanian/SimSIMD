/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/dots/neonhalf.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOTS_NEONHALF_H
#define NK_DOTS_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

#include "numkong/dot/neonhalf.h"

#if defined(__cplusplus)
extern "C" {
#endif

// F16 GEMM: k_tile=4 (4 f16s = 8 bytes = 64-bit input for direct f32 conversion)
nk_make_dots_pack_size_(neonhalf, f16, f32)
nk_make_dots_pack_(neonhalf, f16, f32)
nk_make_dots_inner_vectors_(f16f16f32_neonhalf, f16, f32, nk_b64_vec_t, nk_dot_f16x4_state_neonhalf_t, nk_b128_vec_t,
                            nk_dot_f16x4_init_neonhalf, nk_load_b64_neon_, nk_partial_load_b16x4_neon_,
                            nk_dot_f16x4_update_neonhalf, nk_dot_f16x4_finalize_neonhalf, nk_partial_store_b32x4_neon_,
                            /*k_tile=*/4)

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_NEONHALF_H
