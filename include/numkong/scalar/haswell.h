/**
 *  @brief SIMD-accelerated Scalar Math Helpers for Haswell.
 *  @file include/numkong/scalar/haswell.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalar.h
 *
 *  @section scalars_haswell_instructions Key AVX2/FMA Scalar Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm_sqrt_ps                 VSQRTPS (XMM, XMM)              11cy        7cy         p0
 *      _mm_sqrt_pd                 VSQRTPD (XMM, XMM)              16cy        12cy        p0
 *      _mm_fmadd_ss                VFMADD (XMM, XMM, XMM)          5cy         0.5/cy      p01
 *      _mm_fmadd_sd                VFMADD (XMM, XMM, XMM)          5cy         0.5/cy      p01
 *      _mm_cvtps_ph                VCVTPS2PH (XMM, XMM, I8)        4cy         1/cy        p01+p5
 *      _mm_cvtph_ps                VCVTPH2PS (XMM, XMM)            5cy         1/cy        p01
 */
#ifndef NK_SCALAR_HASWELL_H
#define NK_SCALAR_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC nk_f32_t nk_f32_sqrt_haswell(nk_f32_t x) { return _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(x))); }
NK_PUBLIC nk_f64_t nk_f64_sqrt_haswell(nk_f64_t x) { return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(x))); }
NK_PUBLIC nk_f32_t nk_f32_rsqrt_haswell(nk_f32_t x) { return 1.0f / nk_f32_sqrt_haswell(x); }
NK_PUBLIC nk_f64_t nk_f64_rsqrt_haswell(nk_f64_t x) { return 1.0 / nk_f64_sqrt_haswell(x); }
NK_PUBLIC nk_f32_t nk_f32_fma_haswell(nk_f32_t a, nk_f32_t b, nk_f32_t c) {
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}
NK_PUBLIC nk_f64_t nk_f64_fma_haswell(nk_f64_t a, nk_f64_t b, nk_f64_t c) {
    return _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)));
}

NK_PUBLIC void nk_f32_to_f16_haswell(nk_f32_t const *from, nk_f16_t *to) {
    *to = _mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(*from), _MM_FROUND_TO_NEAREST_INT));
}
NK_PUBLIC void nk_f16_to_f32_haswell(nk_f16_t const *from, nk_f32_t *to) {
    *to = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(*from)));
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_SCALAR_HASWELL_H
