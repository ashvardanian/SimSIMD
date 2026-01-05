/**
 *  @brief SIMD-accelerated Spatial Distances for bf16 using XuanTie (RVV + Zvfbfwma).
 *  @file include/numkong/spatial/xuantie.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  Zvfbfwma provides widening bf16 fused multiply-accumulate to f32:
 *    vfwmaccbf16: f32 ← bf16 × bf16
 *
 *  For L2 distance, we use the identity: (a−b)² = a² + b² − 2×a×b
 *  This allows us to use vfwmaccbf16 for all computations.
 *
 *  Requires: RVV 1.0 + Zvfbfwma extension (GCC 14+ or Clang 18+)
 */
#ifndef NK_SPATIAL_XUANTIE_H
#define NK_SPATIAL_XUANTIE_H

#if NK_TARGET_RISCV_
#if NK_TARGET_XUANTIE

#include <math.h>

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief  L2 squared distance of two bf16 vectors with f32 accumulation on XuanTie.
 *
 *  L2²(a,b) = Σ(a[i] - b[i])² = Σ(a² + b² − 2×a×b)
 *  Uses vfwmaccbf16 for all bf16 multiply-accumulate operations.
 */
NK_PUBLIC void nk_l2sq_bf16_xuantie(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    // Accumulate a² + b² and a*b separately
    vfloat32m1_t sq_sum_f32x1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);  // a² + b²
    vfloat32m1_t ab_sum_f32x1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);  // a*b

    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e16m1(count_scalars);
        vbfloat16m1_t a_bf16x1 = __riscv_vle16_v_bf16m1((bfloat16_t const *)a_scalars, vl);
        vbfloat16m1_t b_bf16x1 = __riscv_vle16_v_bf16m1((bfloat16_t const *)b_scalars, vl);

        // Compute a² using vfwmaccbf16
        vfloat32m2_t aa_f32x2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        aa_f32x2 = __riscv_vfwmaccbf16_vv_f32m2(aa_f32x2, a_bf16x1, a_bf16x1, vl);
        sq_sum_f32x1 = __riscv_vfredusum_vs_f32m2_f32m1(aa_f32x2, sq_sum_f32x1, vl);

        // Compute b² using vfwmaccbf16
        vfloat32m2_t bb_f32x2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        bb_f32x2 = __riscv_vfwmaccbf16_vv_f32m2(bb_f32x2, b_bf16x1, b_bf16x1, vl);
        sq_sum_f32x1 = __riscv_vfredusum_vs_f32m2_f32m1(bb_f32x2, sq_sum_f32x1, vl);

        // Compute a*b using vfwmaccbf16
        vfloat32m2_t ab_f32x2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        ab_f32x2 = __riscv_vfwmaccbf16_vv_f32m2(ab_f32x2, a_bf16x1, b_bf16x1, vl);
        ab_sum_f32x1 = __riscv_vfredusum_vs_f32m2_f32m1(ab_f32x2, ab_sum_f32x1, vl);
    }

    // Final: (a² + b²) - 2*(a*b)
    nk_f32_t sq_sum = __riscv_vfmv_f_s_f32m1_f32(sq_sum_f32x1);
    nk_f32_t ab_sum = __riscv_vfmv_f_s_f32m1_f32(ab_sum_f32x1);
    *result = sq_sum - 2.0f * ab_sum;
}

/**
 *  @brief  L2 distance of two bf16 vectors on XuanTie.
 */
NK_PUBLIC void nk_l2_bf16_xuantie(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    nk_f32_t l2sq;
    nk_l2sq_bf16_xuantie(a_scalars, b_scalars, count_scalars, &l2sq);
    // Handle potential negative values from floating point errors
    *result = l2sq > 0.0f ? sqrtf(l2sq) : 0.0f;
}

/**
 *  @brief  Angular distance of two bf16 vectors on XuanTie.
 *
 *  Angular = 1 - dot(a,b) / (||a|| × ||b||)
 *  Uses vfwmaccbf16 for all bf16 multiply-accumulate operations.
 */
NK_PUBLIC void nk_angular_bf16_xuantie(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f32_t *result) {
    vfloat32m1_t dot_f32x1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t a_sq_f32x1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t b_sq_f32x1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);

    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e16m1(count_scalars);
        vbfloat16m1_t a_bf16x1 = __riscv_vle16_v_bf16m1((bfloat16_t const *)a_scalars, vl);
        vbfloat16m1_t b_bf16x1 = __riscv_vle16_v_bf16m1((bfloat16_t const *)b_scalars, vl);

        // dot += a * b
        vfloat32m2_t ab_f32x2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        ab_f32x2 = __riscv_vfwmaccbf16_vv_f32m2(ab_f32x2, a_bf16x1, b_bf16x1, vl);
        dot_f32x1 = __riscv_vfredusum_vs_f32m2_f32m1(ab_f32x2, dot_f32x1, vl);

        // a_sq += a * a
        vfloat32m2_t aa_f32x2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        aa_f32x2 = __riscv_vfwmaccbf16_vv_f32m2(aa_f32x2, a_bf16x1, a_bf16x1, vl);
        a_sq_f32x1 = __riscv_vfredusum_vs_f32m2_f32m1(aa_f32x2, a_sq_f32x1, vl);

        // b_sq += b * b
        vfloat32m2_t bb_f32x2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        bb_f32x2 = __riscv_vfwmaccbf16_vv_f32m2(bb_f32x2, b_bf16x1, b_bf16x1, vl);
        b_sq_f32x1 = __riscv_vfredusum_vs_f32m2_f32m1(bb_f32x2, b_sq_f32x1, vl);
    }

    // Finalize: 1 - dot / sqrt(a_sq * b_sq)
    nk_f32_t dot = __riscv_vfmv_f_s_f32m1_f32(dot_f32x1);
    nk_f32_t a_sq = __riscv_vfmv_f_s_f32m1_f32(a_sq_f32x1);
    nk_f32_t b_sq = __riscv_vfmv_f_s_f32m1_f32(b_sq_f32x1);

    nk_f32_t denom = sqrtf(a_sq * b_sq);
    // Handle edge cases: zero vectors
    if (denom < 1e-12f) {
        *result = 0.0f;
        return;
    }
    nk_f32_t cos_sim = dot / denom;
    // Clamp to [-1, 1] to avoid NaN from floating point errors
    if (cos_sim > 1.0f) cos_sim = 1.0f;
    if (cos_sim < -1.0f) cos_sim = -1.0f;
    *result = 1.0f - cos_sim;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_XUANTIE
#endif // NK_TARGET_RISCV_

#endif // NK_SPATIAL_XUANTIE_H
