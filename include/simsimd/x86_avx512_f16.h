/**
 *  @file   x86_avx512_f16.h
 *  @brief  x86 AVX-512 implementation of the most common similarity metrics for 16-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: L2 squared, inner product, cosine similarity.
 *  - Uses `_mm512_maskz_loadu_epi16` intrinsics to perform masked unaligned loads.
 *  - Uses `f16` for both storage and accumulation, assuming it's resolution is enough for average case.
 *  - Requires compiler capabilities: avx512fp16, avx512f.
 */
#include <immintrin.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

static simsimd_f32_t simsimd_avx512_f16_l2sq(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t d) {
    __m512h diff_vec = _mm512_set1_ph(0);
    simsimd_size_t i = 0;

    do {
        simsimd_size_t elements_to_process = d - i > 32 ? 32 : d - i;
        __mmask32 mask = (1 << elements_to_process) - 1;
        __m512i a_vec = _mm512_maskz_loadu_epi16(mask, a + i);
        __m512i b_vec = _mm512_maskz_loadu_epi16(mask, b + i);
        __m512h sub_vec = _mm512_sub_ph(_mm512_castsi512_ph(a_vec), _mm512_castsi512_ph(b_vec));
        diff_vec = _mm512_fmadd_ph(sub_vec, sub_vec, diff_vec);

        i += 32;
    } while (i < d);

    return _mm512_reduce_add_ph(diff_vec);
}

static simsimd_f32_t simsimd_avx512_f16_ip(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t d) {
    __m512h ab_vec = _mm512_set1_ph(0);
    simsimd_size_t i = 0;

    do {
        simsimd_size_t elements_to_process = d - i > 32 ? 32 : d - i;
        __mmask32 mask = (1 << elements_to_process) - 1;
        __m512i a_vec = _mm512_maskz_loadu_epi16(mask, a + i);
        __m512i b_vec = _mm512_maskz_loadu_epi16(mask, b + i);
        ab_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_vec), _mm512_castsi512_ph(b_vec), ab_vec);

        i += 32;
    } while (i < d);

    return 1 - _mm512_reduce_add_ph(ab_vec);
}

static simsimd_f32_t simsimd_avx512_f16_cos(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t d) {
    __m512h ab_vec = _mm512_set1_ph(0);
    __m512h a2_vec = _mm512_set1_ph(0);
    __m512h b2_vec = _mm512_set1_ph(0);
    simsimd_size_t i = 0;

    do {
        simsimd_size_t elements_to_process = d - i > 32 ? 32 : d - i;
        __mmask32 mask = (1 << elements_to_process) - 1;
        __m512i a_vec = _mm512_maskz_loadu_epi16(mask, a + i);
        __m512i b_vec = _mm512_maskz_loadu_epi16(mask, b + i);
        ab_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_vec), _mm512_castsi512_ph(b_vec), ab_vec);
        a2_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_vec), _mm512_castsi512_ph(a_vec), a2_vec);
        b2_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(b_vec), _mm512_castsi512_ph(b_vec), b2_vec);

        i += 32;
    } while (i < d);

    simsimd_f32_t ab = _mm512_reduce_add_ph(ab_vec);
    simsimd_f32_t a2 = _mm512_reduce_add_ph(a2_vec);
    simsimd_f32_t b2 = _mm512_reduce_add_ph(b2_vec);
    return 1 - ab * simsimd_approximate_inverse_square_root(a2 * b2);
}

#ifdef __cplusplus
} // extern "C"
#endif