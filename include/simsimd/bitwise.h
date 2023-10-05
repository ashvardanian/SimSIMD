/**
 *  @brief Collection of Chemistry-oriented Similarity Measures, SIMD-accelerated with SSE, AVX, NEON, SVE.
 *
 *  @author Ashot Vardanian
 *  @date July 1, 2023
 */

#pragma once
#include <string.h> // `memcpy`

#include <simsimd/simsimd.h>

#ifdef _MSC_VER
#include <intrin.h>
#define popcount64 __popcnt64
#else
#define popcount64 __builtin_popcountll
#endif

#ifdef __cplusplus
extern "C" {
#endif

simsimd_f32_t simsimd_sve_b1_hamming(simsimd_b1_t const* a, simsimd_b1_t const* b, simsimd_size_t d) {
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

simsimd_f32_t simsimd_avx512_b1_hamming(simsimd_b1_t const* a, simsimd_b1_t const* b, simsimd_size_t d) {
    simsimd_size_t words = d / 128;
    uint64_t const* a64 = (uint64_t const*)(a);
    uint64_t const* b64 = (uint64_t const*)(b);
    /// Contains 2x 64-bit integers with running population count sums.
    __m128i d_vec = _mm_set_epi64x(0, 0);
    for (simsimd_size_t i = 0; i != words; i += 2)
        d_vec = _mm_add_epi64( //
            d_vec,             //
            _mm_popcnt_epi64(  //
                _mm_xor_si128( //
                    _mm_load_si128((__m128i const*)(a64 + i)), _mm_load_si128((__m128i const*)(b64 + i)))));
    return _mm_cvtm64_si64(_mm_movepi64_pi64(d_vec)) + _mm_extract_epi64(d_vec, 1);
}

/**
 *  @brief  Optimized version for Tanimoto distance on @b exactly 166 bits,
 *          forming 21 incomplete bytes for the MACCS fingerprints used in
 *          computation chemistry.
 */
inline static simsimd_f32_t //
simsimd_tanimoto_maccs_naive(uint8_t const* a_chars, uint8_t const* b_chars) {
    unsigned long a[3] = {0};
    unsigned long b[3] = {0};
    memcpy(&a[0], a_chars, 21);
    memcpy(&b[0], b_chars, 21);
    float and_count = popcount64(a[0] & b[0]) + popcount64(a[1] & b[1]) + popcount64(a[2] & b[2]);
    float or_count = popcount64(a[0] | b[0]) + popcount64(a[1] | b[1]) + popcount64(a[2] | b[2]);
    return 1 - and_count / or_count;
}

/**
 *  @brief  Weird Tanimoto distance implementation for concatenated MACCS and ECFP4
 *          representations, forming a grand-total of 56 bytes. The last chunk is evaluated
 *          if only the first one matches well.
 *
 *              - First 166 bits in 21 bytes is a MACCS vector.
 *              - After that 3 empty bytes of a padding are expected.
 *              - After which, 256 bytes (32 words) of ECFP4 follow.
 */
inline static simsimd_f32_t //
simsimd_tanimoto_conditional_naive(uint8_t const* a_chars, uint8_t const* b_chars) {
    float const threshold = 0.2;
    unsigned long const* a = (unsigned long const*)(a_chars);
    unsigned long const* b = (unsigned long const*)(b_chars);
    float and_count = popcount64(a[0] & b[0]) + popcount64(a[1] & b[1]) + popcount64(a[2] & b[2]);
    float or_count = popcount64(a[0] | b[0]) + popcount64(a[1] | b[1]) + popcount64(a[2] | b[2]);
    float result = 1 - and_count / or_count;
    if (result > threshold)
        return result;

    // Start comparing ECFP4
    and_count = 0, or_count = 0;
    for (int i = 0; i != 32; ++i)
        and_count += popcount64(a[3 + i] & b[3 + i]), or_count += popcount64(a[3 + i] | b[3 + i]);
    result = 1 - and_count / or_count;
    return result * threshold;
}

#if SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_ARM_NEON

/**
 *  @brief  Optimized version for Tanimoto distance on @b exactly 166 bits,
 *          forming 21 incomplete bytes for the MACCS fingerprints used in
 *          computation chemistry, accelerated for Arm NEON ISA.
 */
inline static simsimd_f32_t //
simsimd_tanimoto_maccs_neon(uint8_t const* a_chars, uint8_t const* b_chars) {
    unsigned char a[32] = {0};
    unsigned char b[32] = {0};
    memcpy(&a[0], a_chars, 21);
    memcpy(&b[0], b_chars, 21);
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
}

#endif // SIMSIMD_TARGET_ARM_NEON

#if SIMSIMD_TARGET_ARM_SVE

/**
 *  @brief  Optimized version for Tanimoto distance on @b exactly 166 bits,
 *          forming 21 incomplete bytes for the MACCS fingerprints used in
 *          computation chemistry, accelerated for Arm NEON ISA.
 */
__attribute__((target("+sve"))) inline static simsimd_f32_t //
simsimd_tanimoto_maccs_sve(uint8_t const* a_chars, uint8_t const* b_chars) {
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
}
#endif
#endif // SIMSIMD_TARGET_ARM

#if SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_X86_AVX512

/**
 *  @brief  Optimized version for Tanimoto distance on @b exactly 166 bits,
 *          forming 21 incomplete bytes for the MACCS fingerprints used in
 *          computation chemistry, accelerated with AVX-512 population count
 *          instructions.
 */
__attribute__((target("avx512vpopcntdq")))                     //
__attribute__((target("avx512vl")))                            //
__attribute__((target("avx512bw")))                            //
__attribute__((target("avx512f"))) inline static simsimd_f32_t //
simsimd_tanimoto_maccs_avx512(uint8_t const* a, uint8_t const* b) {
    __m256i a_vec = _mm256_maskz_loadu_epi8(0b11111111111111111111100000000000, a);
    __m256i b_vec = _mm256_maskz_loadu_epi8(0b11111111111111111111100000000000, b);
    __m256i and_vec = _mm256_and_si256(a_vec, b_vec);
    __m256i or_vec = _mm256_or_si256(a_vec, b_vec);
    __m256i and_counts_vec = _mm256_popcnt_epi64(and_vec);
    __m256i or_counts_vec = _mm256_popcnt_epi64(or_vec);

    float and_counts = _mm256_extract_epi64(and_counts_vec, 0) + _mm256_extract_epi64(and_counts_vec, 1) +
                       _mm256_extract_epi64(and_counts_vec, 2) + _mm256_extract_epi64(and_counts_vec, 3);
    float or_counts = _mm256_extract_epi64(or_counts_vec, 0) + _mm256_extract_epi64(or_counts_vec, 1) +
                      _mm256_extract_epi64(or_counts_vec, 2) + _mm256_extract_epi64(or_counts_vec, 3);

    return 1 - and_counts / or_counts;
}

#endif // SIMSIMD_TARGET_X86_AVX512
#endif // SIMSIMD_TARGET_X86

#undef popcount64

#ifdef __cplusplus
}
#endif