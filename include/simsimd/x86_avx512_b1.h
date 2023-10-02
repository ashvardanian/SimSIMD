/**
 *  @file   x86_avx512_b1.h
 *  @brief  x86 AVX-512 implementation of the most common similarity metrics for binary vectors.
 *  @author Ash Vardanian
 *
 *  - Implements: Hamming.
 *  - Requires compiler capabilities: +sve.
 */
#include <immintrin.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

inline static simsimd_f32_t simsimd_sve_b1_hamming(simsimd_b1_t const* a, simsimd_b1_t const* b, simsimd_size_t d) {
    simsimd_size_t i = 0;
    svuint8_t d_vec = svdupq_n_u8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    svbool_t pg_vec = svwhilelt_b8(i, d);
    do {
        svuint8_t a_vec = svld1_u8(pg_vec, a + i);
        svuint8_t b_vec = svld1_u8(pg_vec, b + i);
        svuint8_t a_xor_b_vec = sveor_u8_m(pg_vec, a_vec, b_vec);
        d_vec = svadd_u8_m(pg_vec, d_vec, svcnt_u8_x(pg_vec, a_xor_b_vec));
        i += svcntb() * __CHAR_BIT__;
        pg_vec = svwhilelt_b32(i, d);
    } while (svptest_any(svptrue_b32(), pg_vec));
    return 1 - svaddv_u8(svptrue_b32(), d_vec);
}

#ifdef __cplusplus
} // extern "C"
#endif