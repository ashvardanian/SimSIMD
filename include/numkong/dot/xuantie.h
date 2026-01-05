/**
 *  @brief SIMD-accelerated Dot Products for bf16 using XuanTie (RVV + Zvfbfwma).
 *  @file include/numkong/dot/xuantie.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  Alibaba XuanTie C930 and similar chips implement RVV 1.0 with Zvfbfwma extension.
 *  Zvfbfwma provides widening bf16 fused multiply-accumulate to f32:
 *    vfwmaccbf16: f32 ← bf16 × bf16
 *
 *  Requires: RVV 1.0 + Zvfbfwma extension (GCC 14+ or Clang 18+)
 */
#ifndef NK_DOT_XUANTIE_H
#define NK_DOT_XUANTIE_H

#if NK_TARGET_RISCV_
#if NK_TARGET_XUANTIE

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief  Dot product of two bf16 vectors with f32 accumulation on XuanTie.
 *
 *  Uses vfwmaccbf16 for widening bf16 multiply-accumulate to f32.
 *  VL-based loop handles all tail elements automatically.
 */
NK_PUBLIC void nk_dot_bf16_xuantie(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f32_t *result) {
    vfloat32m1_t sum_f32x1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e16m1(count_scalars);
        vbfloat16m1_t a_bf16x1 = __riscv_vle16_v_bf16m1((bfloat16_t const *)a_scalars, vl);
        vbfloat16m1_t b_bf16x1 = __riscv_vle16_v_bf16m1((bfloat16_t const *)b_scalars, vl);
        // Widening bf16 FMA: f32 ← bf16 × bf16
        vfloat32m2_t acc_f32x2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        acc_f32x2 = __riscv_vfwmaccbf16_vv_f32m2(acc_f32x2, a_bf16x1, b_bf16x1, vl);
        // Reduction sum
        sum_f32x1 = __riscv_vfredusum_vs_f32m2_f32m1(acc_f32x2, sum_f32x1, vl);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32x1);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_XUANTIE
#endif // NK_TARGET_RISCV_

#endif // NK_DOT_XUANTIE_H
