/**
 *  @brief Collection of Biology-oriented Similarity Measures, SIMD-accelerated with SSE, AVX, NEON, SVE.
 *
 *  @author Ash Vardanian
 *  @date August 8, 2023
 */

#pragma once

#include <simsimd/simsimd_chem.h>

#ifdef _MSC_VER
#define hash32
#define hash64
#else
#define hash32
#define hash64
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief  Used to compute the similarity of two DNA or RNA sequences in "TwoBit" form.
 *
 *  By design uses a separate bit-set for every string, expanding the compressed
 *  representation into bitsets, and then computes the tanimoto distance on those.
 *
 *  Optimal `length_hash` should be derived from `max(a_length, b_length)`, as
 */
inline static simsimd_f32_t simsimd_tanimoto_twobit( //
    uint8_t const* a_chars, uint8_t const* b_chars,  //
    size_t a_length, size_t b_length,                //
    uint8_t* a_hash, uint8_t* b_hash, size_t length_hash) {

    // uint64 -> 32-gram
    // uint32 -> 16-gram
    uint64_t last_word = 0;
    for (size_t i = 0; i + 8 <= a_length; i += 8) {
        uint64_t a_word;
        memcpy(&a_word, a_chars, 8);
    }

    size_t b_words = (b_length + 7) / 8;

    return simsimd_tanimoto_b1x8_naive(a_hash, b_hash, length_hash);
}

#undef popcount64

#ifdef __cplusplus
}
#endif