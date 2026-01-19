/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON CPUs with FMLAL.
 *  @file include/numkong/dots/neonfhm.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *
 *  Uses FMLAL (FEAT_FHM) for widening fp16->f32 multiply-accumulate, which is 20-48% faster
 *  than the convert-then-FMA approach used in neonhalf.h.
 */
#ifndef NK_DOTS_NEONFHM_H
#define NK_DOTS_NEONFHM_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONFHM
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

#include "numkong/dot/neonfhm.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* F16 GEMM using FMLAL: simd_width=8 (8 f16s = 16 bytes = NEON register width) */
nk_define_dots_pack_size_(neonfhm, f16, f32, f32)
nk_define_dots_pack_(neonfhm, f16, f16, f32, nk_assign_from_to_)
nk_define_dots_symmetric_(f16_neonfhm, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_neonfhm_t, nk_b128_vec_t,
                          nk_dot_f16x8_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                          nk_dot_f16x8_update_neonfhm, nk_dot_f16x8_finalize_neonfhm,
                          /*simd_width=*/8)
nk_define_dots_packed_(f16_neonfhm, f16, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_neonfhm_t, nk_b128_vec_t,
                       nk_dot_f16x8_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b16x8_serial_, nk_load_b128_neon_,
                       nk_partial_load_b16x8_serial_, nk_dot_f16x8_update_neonfhm, nk_dot_f16x8_finalize_neonfhm,
                       nk_partial_store_b32x4_serial_,
                       /*simd_width=*/8)

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_NEONFHM_H
