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

inline static void simsimd_4gram_hash_string( //
    char const* chars, size_t chars_length,   //
    uint8_t* hash, size_t hash_length) {

    // Skip misaligned addresses in a naive fashion
    char const* const chars_end = chars + chars_length;
    for (; uintptr_t(chars) % 4 && chars != chars_end; ++chars) {
        uint32_t word;
        memcpy(&word, chars, 4);
        hash;
    }

    uint32_t last_word = *a_chars;

    for (size_t i = 0; i + 8 <= a_length; i += 8) {
        uint32_t new_word = *a_chars;

        size_t hash_first = hash(last_word);
        size_t hash_second = hash((last_word << 8) | (new_word >> 24));
        size_t hash_third = hash((last_word << 16) | (new_word >> 16));
        size_t hash_fourth = hash((last_word << 16) | (new_word >> 16));

        uint64_t a_word;
        memcpy(&a_word, a_chars, 8);
    }
}

/**
 *  @brief  Similarity measure for different length strings, that compares them
 *          by hashing the N-grams first and computing the Tanimoto distance of
 *          hashes.
 *
 *  By design uses a separate bit-set for every string. Those have to be of identical
 *  length derived from `l = max(a_length, b_length)`, as `(l + N - 1) * 2` for any
 *  N-gram, to have less than half of bits populated.
 *
 *  Inspired by Rabin-Karp algorithm, and continues the approaches implemented in String~illa.
 */
inline static simsimd_f32_t simsimd_4gram_hashes_naive( //
    uint8_t const* a_chars, uint8_t const* b_chars,     //
    size_t a_length, size_t b_length,                   //
    uint8_t* a_hash, uint8_t* b_hash, size_t length_hash) {

    // Skip through misaligned part

    uint32_t last_word = *a_chars;

    for (size_t i = 0; i + 8 <= a_length; i += 8) {
        uint32_t new_word = *a_chars;

        size_t hash_first = hash(last_word);
        size_t hash_second = hash((last_word << 8) | (new_word >> 24));
        size_t hash_third = hash((last_word << 16) | (new_word >> 16));
        size_t hash_fourth = hash((last_word << 16) | (new_word >> 16));

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