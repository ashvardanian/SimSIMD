/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for RISC-V BF16.
 *  @file include/numkong/spatial/rvvbf16.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  @sa include/numkong/spatial.h
 *
 *  Zvfbfwma provides widening bf16 fused multiply-accumulate to f32:
 *    vfwmaccbf16: f32 ← bf16 × bf16
 *
 *  For L2 distance, we use the identity: (a−b)² = a² + b² − 2 × a × b
 *  This allows us to use vfwmaccbf16 for all computations.
 *
 *  Requires: RVV 1.0 + Zvfbfwma extension (GCC 14+ or Clang 18+)
 */
#ifndef NK_SPATIAL_RVVBF16_H
#define NK_SPATIAL_RVVBF16_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVVBF16

#include "numkong/types.h"
#include "numkong/spatial/rvv.h" // `nk_f32_sqrt_rvv`

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_sqeuclidean_bf16_rvvbf16(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars,
                                           nk_size_t count_scalars, nk_f32_t *result) {
    // Accumulate a² + b² and a × b separately
    vfloat32m1_t sq_sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1); // a² + b²
    vfloat32m1_t ab_sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1); // a × b

    for (nk_size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e16m1(count_scalars);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)a_scalars, vl);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)b_scalars, vl);
        vbfloat16m1_t a_bf16m1 = __riscv_vreinterpret_v_u16m1_bf16m1(a_u16m1);
        vbfloat16m1_t b_bf16m1 = __riscv_vreinterpret_v_u16m1_bf16m1(b_u16m1);

        // Compute a² using vfwmaccbf16
        vfloat32m2_t aa_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        aa_f32m2 = __riscv_vfwmaccbf16_vv_f32m2(aa_f32m2, a_bf16m1, a_bf16m1, vl);
        sq_sum_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(aa_f32m2, sq_sum_f32m1, vl);

        // Compute b² using vfwmaccbf16
        vfloat32m2_t bb_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        bb_f32m2 = __riscv_vfwmaccbf16_vv_f32m2(bb_f32m2, b_bf16m1, b_bf16m1, vl);
        sq_sum_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(bb_f32m2, sq_sum_f32m1, vl);

        // Compute a × b using vfwmaccbf16
        vfloat32m2_t ab_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        ab_f32m2 = __riscv_vfwmaccbf16_vv_f32m2(ab_f32m2, a_bf16m1, b_bf16m1, vl);
        ab_sum_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(ab_f32m2, ab_sum_f32m1, vl);
    }

    // Final: (a² + b²) − 2 × (a × b)
    nk_f32_t sq_sum = __riscv_vfmv_f_s_f32m1_f32(sq_sum_f32m1);
    nk_f32_t ab_sum = __riscv_vfmv_f_s_f32m1_f32(ab_sum_f32m1);
    *result = sq_sum - 2.0f * ab_sum;
}

NK_PUBLIC void nk_euclidean_bf16_rvvbf16(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result) {
    nk_sqeuclidean_bf16_rvvbf16(a_scalars, b_scalars, count_scalars, result);
    // Handle potential negative values from floating point errors
    *result = *result > 0.0f ? nk_f32_sqrt_rvv(*result) : 0.0f;
}

NK_PUBLIC void nk_angular_bf16_rvvbf16(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    vfloat32m1_t dot_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t a_sq_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t b_sq_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);

    for (nk_size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e16m1(count_scalars);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)a_scalars, vl);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)b_scalars, vl);
        vbfloat16m1_t a_bf16m1 = __riscv_vreinterpret_v_u16m1_bf16m1(a_u16m1);
        vbfloat16m1_t b_bf16m1 = __riscv_vreinterpret_v_u16m1_bf16m1(b_u16m1);

        // dot += a × b
        vfloat32m2_t ab_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        ab_f32m2 = __riscv_vfwmaccbf16_vv_f32m2(ab_f32m2, a_bf16m1, b_bf16m1, vl);
        dot_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(ab_f32m2, dot_f32m1, vl);

        // a_sq += a × a
        vfloat32m2_t aa_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        aa_f32m2 = __riscv_vfwmaccbf16_vv_f32m2(aa_f32m2, a_bf16m1, a_bf16m1, vl);
        a_sq_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(aa_f32m2, a_sq_f32m1, vl);

        // b_sq += b × b
        vfloat32m2_t bb_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        bb_f32m2 = __riscv_vfwmaccbf16_vv_f32m2(bb_f32m2, b_bf16m1, b_bf16m1, vl);
        b_sq_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(bb_f32m2, b_sq_f32m1, vl);
    }

    // Finalize: 1 − dot / √(‖a‖² × ‖b‖²)
    nk_f32_t dot = __riscv_vfmv_f_s_f32m1_f32(dot_f32m1);
    nk_f32_t a_sq = __riscv_vfmv_f_s_f32m1_f32(a_sq_f32m1);
    nk_f32_t b_sq = __riscv_vfmv_f_s_f32m1_f32(b_sq_f32m1);

    // Normalize: 1 − dot / √(‖a‖² × ‖b‖²)
    if (a_sq == 0.0f && b_sq == 0.0f) { *result = 0.0f; }
    else if (dot == 0.0f) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - dot * nk_f32_rsqrt_rvv(a_sq) * nk_f32_rsqrt_rvv(b_sq);
        *result = unclipped > 0.0f ? unclipped : 0.0f;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_RVVBF16
#endif // NK_TARGET_RISCV_
#endif // NK_SPATIAL_RVVBF16_H
