/**
 *  @brief SIMD-accelerated Scalar Math Helpers for Sapphire Rapids.
 *  @file include/numkong/scalars/sapphire.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalars.h
 *
 *  Provides native AVX-512 FP16 scalar conversions via `VCVTSS2SH` / `VCVTSH2SS`.
 */
#ifndef NK_SCALARS_SAPPHIRE_H
#define NK_SCALARS_SAPPHIRE_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512fp16,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512fp16", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_f32_to_f16_sapphire(nk_f32_t const *from, nk_f16_t *to) {
    *to = _mm_cvtsi128_si32(_mm_castph_si128(_mm_cvtss_sh(_mm_setzero_ph(), _mm_set_ss(*from))));
}
NK_PUBLIC void nk_f16_to_f32_sapphire(nk_f16_t const *from, nk_f32_t *to) {
    *to = _mm_cvtss_f32(_mm_cvtsh_ss(_mm_setzero_ps(), _mm_castsi128_ph(_mm_cvtsi32_si128(*from))));
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_
#endif // NK_SCALARS_SAPPHIRE_H
