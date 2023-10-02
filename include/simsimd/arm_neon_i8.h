/**
 *  @file   arm_neon_i8.h
 *  @brief  Arm NEON implementation of the most common similarity metrics for 8-bit signed integral numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: L2 squared, cosine similarity, inner product (same as cosine).
 *  - Uses `i8` for storage, `i16` for multiplication, and `i32` for accumulation, if no better option is available.
 *  - Requires compiler capabilities: +simd.
 */
#include <arm_neon.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

inline static simsimd_f32_t simsimd_neon_i8_l2sq(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    int32x4_t d2_vec = vdupq_n_s32(0);
    simsimd_size_t i = 0;
    for (; i + 7 < d; i += 8) {
        int8x8_t a_vec = vld1_s8(a + i);
        int8x8_t b_vec = vld1_s8(b + i);
        int16x8_t a_vec16 = vmovl_s8(a_vec);
        int16x8_t b_vec16 = vmovl_s8(b_vec);
        int16x8_t d_vec = vsubq_s16(a_vec16, b_vec16);
        int32x4_t d_low = vmull_s16(vget_low_s16(d_vec), vget_low_s16(d_vec));
        int32x4_t d_high = vmull_s16(vget_high_s16(d_vec), vget_high_s16(d_vec));
        d2_vec = vaddq_s32(d2_vec, vaddq_s32(d_low, d_high));
    }
    int32_t d2 = vaddvq_s32(d2_vec);
    for (; i < d; ++i) {
        int16_t d = a[i] - b[i];
        d2 += d * d;
    }
    return d2;
}

inline static simsimd_f32_t simsimd_neon_i8_cos(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {

    int32x4_t ab_vec = vdupq_n_s32(0);
    int32x4_t a2_vec = vdupq_n_s32(0);
    int32x4_t b2_vec = vdupq_n_s32(0);
    simsimd_size_t i = 0;

    // If the 128-bit `vdot_s32` intrinsic is unavailable, we can use the 64-bit `vdot_s32`.
    // for (simsimd_size_t i = 0; i != d; i += 8) {
    //     int16x8_t a_vec = vmovl_s8(vld1_s8(a + i));
    //     int16x8_t b_vec = vmovl_s8(vld1_s8(b + i));
    //     int16x8_t ab_part_vec = vmulq_s16(a_vec, b_vec);
    //     int16x8_t a2_part_vec = vmulq_s16(a_vec, a_vec);
    //     int16x8_t b2_part_vec = vmulq_s16(b_vec, b_vec);
    //     ab_vec = vaddq_s32(ab_vec, vaddq_s32(vmovl_s16(vget_high_s16(ab_part_vec)), //
    //                                          vmovl_s16(vget_low_s16(ab_part_vec))));
    //     a2_vec = vaddq_s32(a2_vec, vaddq_s32(vmovl_s16(vget_high_s16(a2_part_vec)), //
    //                                          vmovl_s16(vget_low_s16(a2_part_vec))));
    //     b2_vec = vaddq_s32(b2_vec, vaddq_s32(vmovl_s16(vget_high_s16(b2_part_vec)), //
    //                                          vmovl_s16(vget_low_s16(b2_part_vec))));
    // }
    for (; i + 15 < d; i += 16) {
        int8x16_t a_vec = vld1q_s8(a + i);
        int8x16_t b_vec = vld1q_s8(b + i);
        ab_vec = vdotq_s32(ab_vec, a_vec, b_vec);
        a2_vec = vdotq_s32(a2_vec, a_vec, a_vec);
        b2_vec = vdotq_s32(b2_vec, b_vec, b_vec);
    }

    int32_t ab = vaddvq_s32(ab_vec);
    int32_t a2 = vaddvq_s32(a2_vec);
    int32_t b2 = vaddvq_s32(b2_vec);

    // Take care of the tail:
    for (; i < d; ++i) {
        int32_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    return 1 - ab * simsimd_approximate_inverse_square_root(a2 * b2);
}

inline static simsimd_f32_t simsimd_neon_i8_ip(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    return simsimd_neon_i8_cos(a, b, d);
}

#ifdef __cplusplus
} // extern "C"
#endif