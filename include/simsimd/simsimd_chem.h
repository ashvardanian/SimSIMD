/**
 *  @brief Collection of Chemistry-oriented Similarity Measures, SIMD-accelerated with SSE, AVX, NEON, SVE.
 *
 *  @author Ashot Vardanian
 *  @date July 1, 2023
 */

#pragma once
#include <simsimd/simsimd.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief  Optimized version for Tanimoto distance on @b exactly 166 bits,
 *          forming 21 incomplete bytes for the MACCS fingerprints used in
 *          computation chemistry.
 */
inline static simsimd_f32_t simsimd_tanimoto_maccs_naive(uint8_t const* a_chars, uint8_t const* b_chars) {
    unsigned long a[3] = {0};
    unsigned long b[3] = {0};
    __builtin_memcpy(&a[0], a_chars, 21);
    __builtin_memcpy(&b[0], b_chars, 21);
    float and_count =                       //
        __builtin_popcountll(a[0] & b[0]) + //
        __builtin_popcountll(a[1] & b[1]) + //
        __builtin_popcountll(a[2] & b[2]);
    float or_count =                        //
        __builtin_popcountll(a[0] | b[0]) + //
        __builtin_popcountll(a[1] | b[1]) + //
        __builtin_popcountll(a[2] | b[2]);
    return 1 - and_count / or_count;
}

/**
 *  @brief  Optimized version for Tanimoto distance on @b exactly 166 bits,
 *          forming 21 incomplete bytes for the MACCS fingerprints used in
 *          computation chemistry, accelerated for Arm NEON ISA.
 */
inline static simsimd_f32_t simsimd_tanimoto_maccs_neon(uint8_t const* a_chars, uint8_t const* b_chars) {
#if defined(__ARM_NEON)
    unsigned char a[32] = {0};
    unsigned char b[32] = {0};
    __builtin_memcpy(&a[0], a_chars, 21);
    __builtin_memcpy(&b[0], b_chars, 21);
    uint8x16_t a_first = vld1q_u8(&a[0]);
    uint8x16_t a_second = vld1q_u8(&a[16]);
    uint8x16_t b_first = vld1q_u8(&b[0]);
    uint8x16_t b_second = vld1q_u8(&b[16]);
    uint8x16_t a_and_b_first = vandq_u8(a_first, b_first);
    uint8x16_t a_and_b_second = vandq_u8(a_second, b_second);
    uint8x16_t a_or_b_first = vorrq_u8(a_first, b_first);
    uint8x16_t a_or_b_second = vorrq_u8(a_second, b_second);
    float and_populations = vaddvq_u8(vaddq_u8(vcntq_u8(a_and_b_first), vcntq_u8(a_and_b_second)));
    float or_populations = vaddvq_u8(vaddq_u8(vcntq_u8(a_or_b_first), vcntq_u8(a_or_b_second)));
    return 1 - and_populations / or_populations;
#else
    (void)a, (void)b;
    return 0;
#endif
}

/**
 *  @brief  Optimized version for Tanimoto distance on @b exactly 166 bits,
 *          forming 21 incomplete bytes for the MACCS fingerprints used in
 *          computation chemistry, accelerated for Arm NEON ISA.
 */
inline static simsimd_f32_t simsimd_tanimoto_maccs_sve(uint8_t const* a_chars, uint8_t const* b_chars) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg_vec = svwhilelt_b8(0ul, 21ul);
    svuint8_t a_vec = svld1_u8(pg_vec, (uint8_t const*)a_chars);
    svuint8_t b_vec = svld1_u8(pg_vec, (uint8_t const*)b_chars);
    svuint8_t a_and_b_vec = svand_u8_m(pg_vec, a_vec, b_vec);
    svuint8_t a_or_b_vec = svorr_u8_m(pg_vec, a_vec, b_vec);
    svuint8_t and_populations_vec = svcnt_u8_x(pg_vec, a_and_b_vec);
    svuint8_t or_populations_vec = svcnt_u8_x(pg_vec, a_or_b_vec);

    // We are gonna need at most two cycles like this, regardless of the architecture,
    // assuming the SVE registers are at least 128 bits (16 bytes), and we onlu need 21 bytes.
    pg_vec = svwhilelt_b8(svcntb(), 21ul);
    a_vec = svld1_u8(pg_vec, (uint8_t const*)a_chars + svcntb());
    b_vec = svld1_u8(pg_vec, (uint8_t const*)b_chars + svcntb());
    a_and_b_vec = svand_u8_m(pg_vec, a_vec, b_vec);
    a_or_b_vec = svorr_u8_m(pg_vec, a_vec, b_vec);
    and_populations_vec = svadd_u8_m(pg_vec, svcnt_u8_x(pg_vec, a_and_b_vec), and_populations_vec);
    or_populations_vec = svadd_u8_m(pg_vec, svcnt_u8_x(pg_vec, a_or_b_vec), or_populations_vec);

    float and_populations = svaddv_u8(svptrue_b8(), and_populations_vec);
    float or_populations = svaddv_u8(svptrue_b8(), or_populations_vec);
    return 1 - and_populations / or_populations;
#else
    (void)a, (void)b;
    return 0;
#endif
}

/**
 *  @brief  Optimized version for Tanimoto distance on @b exactly 166 bits,
 *          forming 21 incomplete bytes for the MACCS fingerprints used in
 *          computation chemistry, accelerated with AVX-512 population count
 *          instructions.
 */
inline static simsimd_f32_t simsimd_tanimoto_maccs_avx512(uint8_t const* a, uint8_t const* b) {
#if defined(__AVX512VPOPCNTDQ__)
    __m256i a_vec = _mm256_maskz_loadu_epi8(0b11111111111111111111100000000000, a);
    __m256i b_vec = _mm256_maskz_loadu_epi8(0b11111111111111111111100000000000, b);
    __m256i and_vec _mm256_and_si256(a_vec, b_vec);
    __m256i or_vec _mm256_or_si256(a_vec, b_vec);
    __m256i and_counts = _mm256_popcnt_epi8(and_vec);
    __m256i or_counts = _mm256_popcnt_epi8(or_vec);
    return 1 - float(_mm256_reduce_add_epi8(and_counts)) / _mm256_reduce_add_epi8(or_counts);
#else
    (void)a, (void)b;
    return 0;
#endif
}

#ifdef __cplusplus
}
#endif