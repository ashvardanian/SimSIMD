/**
 *  @brief SIMD-accelerated Spatial Distances for f16 optimized for SiFive (RVV + Zvfh) CPUs.
 *  @file include/numkong/spatial/sifive.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  Zvfh provides native half-precision (f16) vector operations.
 *  Uses widening operations (f16 → f32) for precision accumulation.
 *
 *  Requires: RVV 1.0 + Zvfh extension (GCC 14+ or Clang 18+)
 */
#ifndef NK_SPATIAL_SIFIVE_H
#define NK_SPATIAL_SIFIVE_H

#if NK_TARGET_RISCV_
#if NK_TARGET_SIFIVE

#include "numkong/types.h"
#include "numkong/spatial/spacemit.h" // `nk_f32_sqrt_spacemit`

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_l2sq_f16_sifive(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    vfloat32m1_t sum_f32x1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e16m1(count_scalars);
        vfloat16m1_t a_f16x1 = __riscv_vle16_v_f16m1((float16_t const *)a_scalars, vl);
        vfloat16m1_t b_f16x1 = __riscv_vle16_v_f16m1((float16_t const *)b_scalars, vl);
        // Difference in f16
        vfloat16m1_t diff_f16x1 = __riscv_vfsub_vv_f16m1(a_f16x1, b_f16x1, vl);
        // Widening square: f16 × f16 → f32
        vfloat32m2_t sq_f32x2 = __riscv_vfwmul_vv_f32m2(diff_f16x1, diff_f16x1, vl);
        // Reduction sum
        sum_f32x1 = __riscv_vfredusum_vs_f32m2_f32m1(sq_f32x2, sum_f32x1, vl);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32x1);
}

NK_PUBLIC void nk_l2_f16_sifive(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                nk_f32_t *result) {
    nk_l2sq_f16_sifive(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_spacemit(*result);
}

NK_PUBLIC void nk_angular_f16_sifive(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    vfloat32m1_t dot_f32x1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t a_sq_f32x1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t b_sq_f32x1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);

    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e16m1(count_scalars);
        vfloat16m1_t a_f16x1 = __riscv_vle16_v_f16m1((float16_t const *)a_scalars, vl);
        vfloat16m1_t b_f16x1 = __riscv_vle16_v_f16m1((float16_t const *)b_scalars, vl);

        // dot += a * b (widened to f32)
        vfloat32m2_t ab_f32x2 = __riscv_vfwmul_vv_f32m2(a_f16x1, b_f16x1, vl);
        dot_f32x1 = __riscv_vfredusum_vs_f32m2_f32m1(ab_f32x2, dot_f32x1, vl);

        // a_sq += a * a
        vfloat32m2_t aa_f32x2 = __riscv_vfwmul_vv_f32m2(a_f16x1, a_f16x1, vl);
        a_sq_f32x1 = __riscv_vfredusum_vs_f32m2_f32m1(aa_f32x2, a_sq_f32x1, vl);

        // b_sq += b * b
        vfloat32m2_t bb_f32x2 = __riscv_vfwmul_vv_f32m2(b_f16x1, b_f16x1, vl);
        b_sq_f32x1 = __riscv_vfredusum_vs_f32m2_f32m1(bb_f32x2, b_sq_f32x1, vl);
    }

    // Finalize: 1 − dot / √(‖a‖² × ‖b‖²)
    nk_f32_t dot = __riscv_vfmv_f_s_f32m1_f32(dot_f32x1);
    nk_f32_t a_sq = __riscv_vfmv_f_s_f32m1_f32(a_sq_f32x1);
    nk_f32_t b_sq = __riscv_vfmv_f_s_f32m1_f32(b_sq_f32x1);

    // Normalize: 1 − dot / √(‖a‖² × ‖b‖²)
    if (a_sq == 0.0f && b_sq == 0.0f) { *result = 0.0f; }
    else if (dot == 0.0f) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - dot * nk_f32_rsqrt_spacemit(a_sq) * nk_f32_rsqrt_spacemit(b_sq);
        *result = unclipped > 0.0f ? unclipped : 0.0f;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SIFIVE
#endif // NK_TARGET_RISCV_

#endif // NK_SPATIAL_SIFIVE_H
