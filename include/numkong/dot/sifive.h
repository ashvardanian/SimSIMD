/**
 *  @brief SIMD-accelerated Dot Products for f16 using SiFive (RVV + Zvfh).
 *  @file include/numkong/dot/sifive.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  SiFive P670/X280 and similar chips implement RVV 1.0 with Zvfh extension.
 *  Zvfh provides native half-precision (f16) vector operations.
 *  Uses widening multiply (f16 × f16 → f32) for precision, then reduces to f32.
 *
 *  Requires: RVV 1.0 + Zvfh extension (GCC 14+ or Clang 18+)
 */
#ifndef NK_DOT_SIFIVE_H
#define NK_DOT_SIFIVE_H

#if NK_TARGET_RISCV_
#if NK_TARGET_SIFIVE

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief  Dot product of two f16 vectors with f32 accumulation on SiFive.
 *
 *  Uses widening multiply (f16 × f16 → f32) for precision.
 *  VL-based loop handles all tail elements automatically.
 */
NK_PUBLIC void nk_dot_f16_sifive(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    vfloat32m1_t sum_f32x1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e16m1(count_scalars);
        vfloat16m1_t a_f16x1 = __riscv_vle16_v_f16m1((float16_t const *)a_scalars, vl);
        vfloat16m1_t b_f16x1 = __riscv_vle16_v_f16m1((float16_t const *)b_scalars, vl);
        // Widening multiply: f16 × f16 → f32
        vfloat32m2_t ab_f32x2 = __riscv_vfwmul_vv_f32m2(a_f16x1, b_f16x1, vl);
        // Ordered reduction sum
        sum_f32x1 = __riscv_vfredusum_vs_f32m2_f32m1(ab_f32x2, sum_f32x1, vl);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32x1);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SIFIVE
#endif // NK_TARGET_RISCV_

#endif // NK_DOT_SIFIVE_H
