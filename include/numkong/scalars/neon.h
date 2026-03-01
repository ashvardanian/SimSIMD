/**
 *  @brief SIMD-accelerated Scalar Math Helpers for NEON.
 *  @file include/numkong/scalars/neon.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalars.h
 *
 *  @section scalars_neon_instructions Key NEON Scalar Instructions
 *
 *      Intrinsic           Instruction                     Latency     Throughput
 *      vsqrt_f32           FSQRT (S)                       9-12cy      0.25/cy
 *      vsqrt_f64           FSQRT (D)                       12-18cy     0.25/cy
 *      vfmas_f32           FMADD (S, S, S, S)              4cy         2/cy
 *      vfmad_f64           FMADD (D, D, D, D)              4cy         2/cy
 *      vqaddb_u8           UQADD (B)                       1cy         4/cy
 *      vqaddb_s8           SQADD (B)                       1cy         4/cy
 */
#ifndef NK_SCALARS_NEON_H
#define NK_SCALARS_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

NK_PUBLIC nk_f32_t nk_f32_sqrt_neon(nk_f32_t x) { return vget_lane_f32(vsqrt_f32(vdup_n_f32(x)), 0); }
NK_PUBLIC nk_f64_t nk_f64_sqrt_neon(nk_f64_t x) { return vget_lane_f64(vsqrt_f64(vdup_n_f64(x)), 0); }
NK_PUBLIC nk_f32_t nk_f32_rsqrt_neon(nk_f32_t x) { return 1.0f / nk_f32_sqrt_neon(x); }
NK_PUBLIC nk_f64_t nk_f64_rsqrt_neon(nk_f64_t x) { return 1.0 / nk_f64_sqrt_neon(x); }
NK_PUBLIC nk_f32_t nk_f32_fma_neon(nk_f32_t a, nk_f32_t b, nk_f32_t c) { return vfmas_f32(c, a, b); }
NK_PUBLIC nk_f64_t nk_f64_fma_neon(nk_f64_t a, nk_f64_t b, nk_f64_t c) { return vfmad_f64(c, a, b); }

NK_PUBLIC nk_u8_t nk_u8_saturating_add_neon(nk_u8_t a, nk_u8_t b) { return vqaddb_u8(a, b); }
NK_PUBLIC nk_i8_t nk_i8_saturating_add_neon(nk_i8_t a, nk_i8_t b) { return vqaddb_s8(a, b); }
NK_PUBLIC nk_u16_t nk_u16_saturating_add_neon(nk_u16_t a, nk_u16_t b) { return vqaddh_u16(a, b); }
NK_PUBLIC nk_i16_t nk_i16_saturating_add_neon(nk_i16_t a, nk_i16_t b) { return vqaddh_s16(a, b); }
NK_PUBLIC nk_u32_t nk_u32_saturating_add_neon(nk_u32_t a, nk_u32_t b) { return vqadds_u32(a, b); }
NK_PUBLIC nk_i32_t nk_i32_saturating_add_neon(nk_i32_t a, nk_i32_t b) { return vqadds_s32(a, b); }
NK_PUBLIC nk_u64_t nk_u64_saturating_add_neon(nk_u64_t a, nk_u64_t b) { return vqaddd_u64(a, b); }
NK_PUBLIC nk_i64_t nk_i64_saturating_add_neon(nk_i64_t a, nk_i64_t b) { return vqaddd_s64(a, b); }

NK_PUBLIC void nk_f16_to_f32_neon(nk_f16_t const *src, nk_f32_t *dest) {
    float16x4_t f16vec = vld1_dup_f16((nk_f16_for_arm_simd_t const *)src);
    float32x4_t f32vec = vcvt_f32_f16(f16vec);
    *dest = vgetq_lane_f32(f32vec, 0);
}
NK_PUBLIC void nk_f32_to_f16_neon(nk_f32_t const *src, nk_f16_t *dest) {
    float32x4_t f32vec = vdupq_n_f32(*src);
    float16x4_t f16vec = vcvt_f16_f32(f32vec);
    vst1_lane_f16((nk_f16_for_arm_simd_t *)dest, f16vec, 0);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_SCALARS_NEON_H
