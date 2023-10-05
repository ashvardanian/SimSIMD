/**
 *  @file   arm_sve_b1.h
 *  @brief  Arm SVE implementation of the most common similarity metrics for binary vectors.
 *  @author Ash Vardanian
 *
 *  - Implements: Hamming.
 *  - Requires compiler capabilities: avx512vpopcntdq, avx512vl, avx512f.
 */
#include <arm_sve.h>

#include "types.h"

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
