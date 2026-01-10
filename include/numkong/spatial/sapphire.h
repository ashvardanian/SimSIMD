/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Intel Sapphire Rapids CPUs.
 *  @file include/numkong/spatial/sapphire.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  Sapphire Rapids adds native FP16 support via AVX-512 FP16 extension.
 *  For e4m3 L2 distance, we can leverage F16 for the subtraction step:
 *  - e4m3 differences fit in F16 (max |a−b| = 896 < 65504)
 *  - But squared differences overflow F16 (896² = 802816 > 65504)
 *  - So: subtract in F16, convert to F32, then square and accumulate
 *
 *  @section sapphire_spatial_instructions Relevant Instructions
 *
 *      Intrinsic                   Instruction                     Sapphire    Genoa
 *      _mm256_sub_ph               VSUBPH (YMM, YMM, YMM)          4cy @ p05   3cy @ p01
 *      _mm512_cvtph_ps             VCVTPH2PS (ZMM, YMM)            5cy @ p05   5cy @ p01
 *      _mm512_fmadd_ps             VFMADD (ZMM, ZMM, ZMM)          4cy @ p05   4cy @ p01
 *      _mm512_reduce_add_ps        (pseudo: VHADDPS chain)         ~8cy        ~8cy
 *      _mm_maskz_loadu_epi8        VMOVDQU8 (XMM {K}, M128)        7cy @ p23   7cy @ p23
 */
#ifndef NK_SPATIAL_SAPPHIRE_H
#define NK_SPATIAL_SAPPHIRE_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512fp16,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512fp16", "f16c", "fma", "bmi", "bmi2")
#endif

#include "numkong/types.h"
#include "numkong/cast/sapphire.h" // `nk_e4m3x16_to_f16x16_sapphire_`

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_l2sq_e4m3_sapphire(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    __m128i a_e4m3x16, b_e4m3x16;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_l2sq_e4m3_sapphire_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_e4m3x16 = _mm_maskz_loadu_epi8(mask, a_scalars);
        b_e4m3x16 = _mm_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e4m3x16 = _mm_loadu_epi8(a_scalars);
        b_e4m3x16 = _mm_loadu_epi8(b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }

    // Convert e4m3 → f16
    __m256h a_f16x16 = nk_e4m3x16_to_f16x16_sapphire_(a_e4m3x16);
    __m256h b_f16x16 = nk_e4m3x16_to_f16x16_sapphire_(b_e4m3x16);

    // Subtract in F16 - differences fit (max 896 < 65504)
    __m256h diff_f16x16 = _mm256_sub_ph(a_f16x16, b_f16x16);

    // Convert to F32 before squaring (896² = 802816 overflows F16!)
    __m512 diff_f32x16 = _mm512_cvtph_ps(_mm256_castph_si256(diff_f16x16));

    // Square and accumulate in F32
    sum_f32x16 = _mm512_fmadd_ps(diff_f32x16, diff_f32x16, sum_f32x16);

    if (count_scalars) goto nk_l2sq_e4m3_sapphire_cycle;

    *result = _mm512_reduce_add_ps(sum_f32x16);
}

NK_PUBLIC void nk_l2_e4m3_sapphire(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_l2sq_e4m3_sapphire(a_scalars, b_scalars, count_scalars, result);
    *result = nk_sqrt_f32_haswell_(*result);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_

#endif // NK_SPATIAL_SAPPHIRE_H
