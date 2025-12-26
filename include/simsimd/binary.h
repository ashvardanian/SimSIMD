/**
 *  @brief SIMD-accelerated Binary Similarity Measures.
 *  @file include/simsimd/binary.h
 *  @author Ash Vardanian
 *  @date July 1, 2023
 *
 *  Contains following similarity measures:
 *
 *  - Bit-level Hamming distance
 *  - Bit-level Jaccard distance (Tanimoto coefficient)
 *  - Jaccard distance for `u32` integral MinHash vectors from StringZilla
 *  - TODO: Weighted Jaccard distance for `u32` integral Count-Min-Sketch vectors
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, SVE
 *  - x86: Haswell, Ice Lake
 *
 *  @section popcount_strategies Population Count Strategies
 *
 *  Jaccard distances are extremely common and also fairly cheap to compute on binary vectors.
 *  The hardest part of optimizing binary similarity measures is the population count operation.
 *  It's natively supported by almost every instruction set, but the throughput and latency can
 *  be suboptimal. There are several ways to optimize this operation:
 *
 *  - Lookup tables, mostly using nibbles (4-bit lookups)
 *  - Harley-Seal population counts using Carry-Save Adders (CSA)
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  On binary vectors, when computing Jaccard distance we can clearly see how the CPU struggles
 *  to compute that many population counts. There are several instructions we should keep in mind:
 *
 *      Intrinsic                   Instruction                     Ice         Genoa
 *      _mm512_popcnt_epi64         VPOPCNTQ (ZMM, K, ZMM)          3c @ p5     2c @ p01
 *      _mm512_shuffle_epi8         VPSHUFB (ZMM, ZMM, ZMM)         1c @ p5     2c @ p12
 *      _mm512_sad_epu8             VPSADBW (ZMM, ZMM, ZMM)         3c @ p5     3c @ p01
 *      _mm512_ternarylogic_epi64   VPTERNLOGQ (ZMM, ZMM, ZMM, I8)  1c @ p05    1c @ p0123
 *      _mm512_gf2p8mul_epi8        VGF2P8MULB (ZMM, ZMM, ZMM)      5c @ p0     3c @ p01
 *
 *  On Ice Lake, VPOPCNTQ bottlenecks on port 5. On AMD Genoa/Turin, it dual-issues
 *  on ports 0-1, making native popcount significantly faster without CSA tricks.
 *
 *  @section harley_seal Harley-Seal Carry-Save Adders
 *
 *  The Harley-Seal algorithm uses Carry-Save Adders (CSA) to accumulate population counts
 *  with fewer VPOPCNTQ instructions. A CSA computes (a + b + c) as (sum, carry) using only
 *  bitwise operations, deferring expensive popcounts to the final reduction.
 *
 *  Performance varies significantly by architecture and buffer size (cycles/byte):
 *
 *      Method              Buffer      Ice Lake    Sapphire    Genoa
 *      Native VPOPCNTQ     any         ~0.12       ~0.10       ~0.06
 *      Harley-Seal CSA     1 KB        0.107       0.095       0.08
 *      Harley-Seal CSA     4 KB        0.056       0.052       0.05
 *      VPSHUFB lookup      4 KB        0.063       0.058       0.07
 *
 *  For small buffers (<1KB), loop overhead dominates and unrolled native VPOPCNTQ wins.
 *  Harley-Seal shines on large buffers where CSA chains amortize the setup cost.
 *  On AMD Genoa, native VPOPCNTQ is competitive even for large buffers.
 *
 *  @section jaccard_norms Jaccard Optimization via Norms
 *
 *  There is a trivial optimization to halve the number of population counts needed for
 *  binary Jaccard distance, if one knows the set magnitudes ahead of time:
 *
 *      J = |A ∩ B| / |A ∪ B| = |A ∩ B| / (|A| + |B| - |A ∩ B|)
 *
 *  At that point the problem reduces to optimizing memory accesses and register usage.
 *  For such cases, we provide additional function variants designed exclusively for compile-time
 *  dispatch in heavily inlined code, operating on wider vectors with known sizes:
 *
 *  - simsimd_jaccard_b512_state_<isa>_t - Smallest optimal running state
 *  - simsimd_jaccard_b512_init_<isa> - Initializes the running state
 *  - simsimd_jaccard_b512_update_<isa> - Updates the running state with 2 new 512-bit vectors
 *  - simsimd_jaccard_b512_finalize_<isa> - Finalizes the running state and produces the distance
 *
 *  @section streaming_api Streaming API
 *
 *  The streaming variants aren't always strictly equivalent to their counterparts above
 *  and their usage also differs quite drastically. For large-scale batch processing where
 *  vectors won't be reused, consider non-temporal loads (`_mm512_stream_load_si512`) to
 *  bypass the cache and avoid pollution. This is especially beneficial when computing
 *  distances across millions of vectors in a single pass.
 *
 *  @code{.c}
 *  simsimd_b8_t a[128], b[128]; // 1024 bits each filled with random bits
 *
 *  simsimd_jaccard_b512_state_ice_t state_a, state_b, state_c, state_d;
 *  simsimd_jaccard_b512_init_ice(&state_a);
 *  simsimd_jaccard_b512_init_ice(&state_b);
 *  simsimd_jaccard_b512_init_ice(&state_c);
 *  simsimd_jaccard_b512_init_ice(&state_d);
 *  simsimd_jaccard_b512_update_ice(&state_a, &a[0], &b[0]); // First 512 bits
 *  simsimd_jaccard_b512_update_ice(&state_a, &a[64], &b[64]); // Second 512 bits
 *  // ... update state_b, state_c, state_d similarly ...
 *
 *  simsimd_distance_t results[4];
 *  simsimd_jaccard_b512_finalize_ice(&state_a, &state_b, &state_c, &state_d, results);
 *  @endcode
 *
 *  @section tail_handling Tail Handling
 *
 *  The trickiest part is handling the tails of the vectors when their size isn't divisible
 *  by our step size. In such cases, it's recommended to use masked loads when supported by
 *  the ISA, or fall back to scalar code and a local on-stack buffer.
 *
 *  @section references References
 *
 *  - Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  - Arm Intrinsics Reference: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  - Muła et al. "Faster Population Counts": https://arxiv.org/pdf/1611.07612
 *  - Muła SSE POPCOUNT experiments: https://github.com/WojciechMula/sse-popcount
 *  - SimSIMD binary R&D tracker: https://github.com/ashvardanian/SimSIMD/pull/138
 *
 */
#ifndef SIMSIMD_BINARY_H
#define SIMSIMD_BINARY_H

#include "types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Binary Hamming distance computing the number of differing bits between two binary vectors.
 *
 *  @param[in] a The first binary vector.
 *  @param[in] b The second binary vector.
 *  @param[in] n_words The number of 8-bit words in the vectors.
 *  @param[out] result The output distance value.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
SIMSIMD_DYNAMIC void simsimd_hamming_b8(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                        simsimd_distance_t *result);

/**
 *  @brief Binary Jaccard distance computing the ratio of differing bits to the union of bits.
 *
 *  @param[in] a The first binary vector.
 *  @param[in] b The second binary vector.
 *  @param[in] n_words The number of 8-bit words in the vectors.
 *  @param[out] result The output distance value.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
SIMSIMD_DYNAMIC void simsimd_jaccard_b8(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                        simsimd_distance_t *result);

/**
 *  @brief Integral Jaccard distance computing the ratio of differing bits to the union of bits.
 *
 *  @param[in] a The first binary vector.
 *  @param[in] b The second binary vector.
 *  @param[in] n The number of 32-bit scalars in the vectors.
 *  @param[out] result The output distance value.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
SIMSIMD_DYNAMIC void simsimd_jaccard_u32(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                         simsimd_distance_t *result);

// clang-format off

/** @copydoc simsimd_hamming_b8 */
SIMSIMD_PUBLIC void simsimd_hamming_b8_serial(simsimd_b8_t const* a, simsimd_b8_t const* b, simsimd_size_t n_words, simsimd_distance_t* result);
/** @copydoc simsimd_jaccard_b8 */
SIMSIMD_PUBLIC void simsimd_jaccard_b8_serial(simsimd_b8_t const* a, simsimd_b8_t const* b, simsimd_size_t n_words, simsimd_distance_t* result);
/** @copydoc simsimd_jaccard_u32 */
SIMSIMD_PUBLIC void simsimd_jaccard_u32_serial(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t n, simsimd_distance_t* result);

#if SIMSIMD_TARGET_NEON
/** @copydoc simsimd_hamming_b8 */
SIMSIMD_PUBLIC void simsimd_hamming_b8_neon(simsimd_b8_t const* a, simsimd_b8_t const* b, simsimd_size_t n_words, simsimd_distance_t* result);
/** @copydoc simsimd_jaccard_b8 */
SIMSIMD_PUBLIC void simsimd_jaccard_b8_neon(simsimd_b8_t const* a, simsimd_b8_t const* b, simsimd_size_t n_words, simsimd_distance_t* result);
/** @copydoc simsimd_jaccard_u32 */
SIMSIMD_PUBLIC void simsimd_jaccard_u32_neon(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t n, simsimd_distance_t* result);

typedef struct simsimd_jaccard_b512_state_neon_t simsimd_jaccard_b512_state_neon_t;
/** @copydoc simsimd_jaccard_b512_state_neon_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b512_init_neon(simsimd_jaccard_b512_state_neon_t *state);
/** @copydoc simsimd_jaccard_b512_state_neon_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b512_update_neon(simsimd_jaccard_b512_state_neon_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_jaccard_b512_state_neon_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b512_finalize_neon(simsimd_jaccard_b512_state_neon_t const *state_a, simsimd_jaccard_b512_state_neon_t const *state_b, simsimd_jaccard_b512_state_neon_t const *state_c, simsimd_jaccard_b512_state_neon_t const *state_d, simsimd_distance_t *results);
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_SVE
/** @copydoc simsimd_hamming_b8 */
SIMSIMD_PUBLIC void simsimd_hamming_b8_sve(simsimd_b8_t const* a, simsimd_b8_t const* b, simsimd_size_t n_words, simsimd_distance_t* result);
/** @copydoc simsimd_jaccard_b8 */
SIMSIMD_PUBLIC void simsimd_jaccard_b8_sve(simsimd_b8_t const* a, simsimd_b8_t const* b, simsimd_size_t n_words, simsimd_distance_t* result);
/** @copydoc simsimd_jaccard_u32 */
SIMSIMD_PUBLIC void simsimd_jaccard_u32_sve(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
#endif // SIMSIMD_TARGET_SVE

#if SIMSIMD_TARGET_HASWELL
/** @copydoc simsimd_hamming_b8 */
SIMSIMD_PUBLIC void simsimd_hamming_b8_haswell(simsimd_b8_t const* a, simsimd_b8_t const* b, simsimd_size_t n_words, simsimd_distance_t* result);
/** @copydoc simsimd_jaccard_b8 */
SIMSIMD_PUBLIC void simsimd_jaccard_b8_haswell(simsimd_b8_t const* a, simsimd_b8_t const* b, simsimd_size_t n_words, simsimd_distance_t* result);
/** @copydoc simsimd_jaccard_u32 */
SIMSIMD_PUBLIC void simsimd_jaccard_u32_haswell(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t n, simsimd_distance_t* result);

typedef struct simsimd_jaccard_b512_state_haswell_t simsimd_jaccard_b512_state_haswell_t;
/** @copydoc simsimd_jaccard_b512_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b512_init_haswell(simsimd_jaccard_b512_state_haswell_t *state);
/** @copydoc simsimd_jaccard_b512_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b512_update_haswell(simsimd_jaccard_b512_state_haswell_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_jaccard_b512_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b512_finalize_haswell(simsimd_jaccard_b512_state_haswell_t const *state_a, simsimd_jaccard_b512_state_haswell_t const *state_b, simsimd_jaccard_b512_state_haswell_t const *state_c, simsimd_jaccard_b512_state_haswell_t const *state_d, simsimd_distance_t *results);
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_ICE
/** @copydoc simsimd_hamming_b8 */
SIMSIMD_PUBLIC void simsimd_hamming_b8_ice(simsimd_b8_t const* a, simsimd_b8_t const* b, simsimd_size_t n_words, simsimd_distance_t* result);
/** @copydoc simsimd_jaccard_b8 */
SIMSIMD_PUBLIC void simsimd_jaccard_b8_ice(simsimd_b8_t const* a, simsimd_b8_t const* b, simsimd_size_t n_words, simsimd_distance_t* result);
/** @copydoc simsimd_jaccard_u32 */
SIMSIMD_PUBLIC void simsimd_jaccard_u32_ice(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t n, simsimd_distance_t* result);

typedef struct simsimd_jaccard_b512_state_ice_t simsimd_jaccard_b512_state_ice_t;
/** @copydoc simsimd_jaccard_b512_state_ice_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b512_init_ice(simsimd_jaccard_b512_state_ice_t *state);
/** @copydoc simsimd_jaccard_b512_state_ice_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b512_update_ice(simsimd_jaccard_b512_state_ice_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_jaccard_b512_state_ice_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b512_finalize_ice(simsimd_jaccard_b512_state_ice_t const *state_a, simsimd_jaccard_b512_state_ice_t const *state_b, simsimd_jaccard_b512_state_ice_t const *state_c, simsimd_jaccard_b512_state_ice_t const *state_d, simsimd_distance_t *results);
#endif // SIMSIMD_TARGET_ICE

// clang-format on

SIMSIMD_PUBLIC unsigned char simsimd_popcount_b8(simsimd_b8_t x) {
    static unsigned char lookup_table[] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, //
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};
    return lookup_table[x];
}

SIMSIMD_PUBLIC void simsimd_hamming_b8_serial(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                              simsimd_distance_t *result) {
    simsimd_u32_t differences = 0;
    for (simsimd_size_t i = 0; i != n_words; ++i) differences += simsimd_popcount_b8(a[i] ^ b[i]);
    *result = differences;
}

SIMSIMD_PUBLIC void simsimd_jaccard_b8_serial(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                              simsimd_distance_t *result) {
    simsimd_u32_t intersection_count = 0, union_count = 0;
    for (simsimd_size_t i = 0; i != n_words; ++i)
        intersection_count += simsimd_popcount_b8(a[i] & b[i]), union_count += simsimd_popcount_b8(a[i] | b[i]);
    *result = (union_count != 0) ? 1 - (simsimd_f64_t)intersection_count / (simsimd_f64_t)union_count : 1;
}

SIMSIMD_PUBLIC void simsimd_jaccard_u32_serial(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                               simsimd_distance_t *result) {
    simsimd_u32_t intersection_count = 0;
    for (simsimd_size_t i = 0; i != n; ++i) intersection_count += (a[i] == b[i]);
    *result = (n != 0) ? 1 - (simsimd_f64_t)intersection_count / (simsimd_f64_t)n : 1;
}

#if _SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)

SIMSIMD_INTERNAL simsimd_u32_t _simsimd_reduce_u8x16_neon(uint8x16_t vec) {
    // Split the vector into two halves and widen to `uint16x8_t`
    uint16x8_t low_half = vmovl_u8(vget_low_u8(vec));   // widen lower 8 elements
    uint16x8_t high_half = vmovl_u8(vget_high_u8(vec)); // widen upper 8 elements

    // Sum the widened halves
    uint16x8_t sum16 = vaddq_u16(low_half, high_half);

    // Now reduce the `uint16x8_t` to a single `simsimd_u32_t`
    uint32x4_t sum32 = vpaddlq_u16(sum16);       // pairwise add into 32-bit integers
    uint64x2_t sum64 = vpaddlq_u32(sum32);       // pairwise add into 64-bit integers
    simsimd_u32_t final_sum = vaddvq_u64(sum64); // final horizontal add to 32-bit result
    return final_sum;
}

SIMSIMD_PUBLIC void simsimd_hamming_b8_neon(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                            simsimd_distance_t *result) {
    simsimd_i32_t differences = 0;
    simsimd_size_t i = 0;
    // In each 8-bit word we may have up to 8 differences.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the differences into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= n_words) {
        uint8x16_t differences_cycle_vec = vdupq_n_u8(0);
        for (simsimd_size_t cycle = 0; cycle < 31 && i + 16 <= n_words; ++cycle, i += 16) {
            uint8x16_t a_vec = vld1q_u8(a + i);
            uint8x16_t b_vec = vld1q_u8(b + i);
            uint8x16_t xor_count_vec = vcntq_u8(veorq_u8(a_vec, b_vec));
            differences_cycle_vec = vaddq_u8(differences_cycle_vec, xor_count_vec);
        }
        differences += _simsimd_reduce_u8x16_neon(differences_cycle_vec);
    }
    // Handle the tail
    for (; i != n_words; ++i) differences += simsimd_popcount_b8(a[i] ^ b[i]);
    *result = differences;
}

SIMSIMD_PUBLIC void simsimd_jaccard_b8_neon(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                            simsimd_distance_t *result) {
    simsimd_i32_t intersection_count = 0, union_count = 0;
    simsimd_size_t i = 0;
    // In each 8-bit word we may have up to 8 intersections/unions.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the intersections/unions into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= n_words) {
        uint8x16_t intersection_popcount_vec = vdupq_n_u8(0);
        uint8x16_t union_popcount_vec = vdupq_n_u8(0);
        for (simsimd_size_t cycle = 0; cycle < 31 && i + 16 <= n_words; ++cycle, i += 16) {
            uint8x16_t a_vec = vld1q_u8(a + i);
            uint8x16_t b_vec = vld1q_u8(b + i);
            uint8x16_t intersection_bits_popcount = vcntq_u8(vandq_u8(a_vec, b_vec));
            uint8x16_t union_bits_popcount = vcntq_u8(vorrq_u8(a_vec, b_vec));
            intersection_popcount_vec = vaddq_u8(intersection_popcount_vec, intersection_bits_popcount);
            union_popcount_vec = vaddq_u8(union_popcount_vec, union_bits_popcount);
        }
        intersection_count += _simsimd_reduce_u8x16_neon(intersection_popcount_vec);
        union_count += _simsimd_reduce_u8x16_neon(union_popcount_vec);
    }
    // Handle the tail
    for (; i != n_words; ++i)
        intersection_count += simsimd_popcount_b8(a[i] & b[i]), union_count += simsimd_popcount_b8(a[i] | b[i]);
    *result = (union_count != 0) ? 1 - (simsimd_f64_t)intersection_count / (simsimd_f64_t)union_count : 1;
}

SIMSIMD_PUBLIC void simsimd_jaccard_u32_neon(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                             simsimd_distance_t *result) {
    simsimd_size_t intersection_count = 0;
    simsimd_size_t i = 0;
    uint32x4_t intersection_count_vec = vdupq_n_u32(0);
    for (; i + 4 <= n; i += 4) {
        uint32x4_t a_vec = vld1q_u32(a + i);
        uint32x4_t b_vec = vld1q_u32(b + i);
        uint32x4_t equality_vec = vceqq_u32(a_vec, b_vec);
        intersection_count_vec = vaddq_u32(intersection_count_vec, vshrq_n_u32(equality_vec, 31));
    }
    intersection_count += vaddvq_u32(intersection_count_vec);
    for (; i != n; ++i) intersection_count += (a[i] == b[i]);
    *result = (n != 0) ? 1 - (simsimd_f64_t)intersection_count / (simsimd_f64_t)n : 1;
}

/**
 *  @brief Running state for 512-bit Jaccard accumulation on NEON.
 *  @code{.c}
 *  simsimd_b512_vec_t a_block, b_block;
 *
 *  simsimd_jaccard_b512_state_neon_t state_a, state_b, state_c, state_d;
 *  simsimd_jaccard_b512_init_neon(&state_a);
 *  simsimd_jaccard_b512_init_neon(&state_b);
 *  simsimd_jaccard_b512_init_neon(&state_c);
 *  simsimd_jaccard_b512_init_neon(&state_d);
 *  simsimd_jaccard_b512_update_neon(&state_a, a_block, b_block);
 *  // ... update state_b, state_c, state_d similarly ...
 *
 *  simsimd_distance_t results[4];
 *  simsimd_jaccard_b512_finalize_neon(&state_a, &state_b, &state_c, &state_d, results);
 *  @endcode
 */
typedef struct simsimd_jaccard_b512_state_neon_t {
    uint32x4_t intersection_count;
} simsimd_jaccard_b512_state_neon_t;

SIMSIMD_INTERNAL void simsimd_jaccard_b512_init_neon(simsimd_jaccard_b512_state_neon_t *state) {
    state->intersection_count = vdupq_n_u32(0);
}

SIMSIMD_INTERNAL void simsimd_jaccard_b512_update_neon(simsimd_jaccard_b512_state_neon_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    // Compute popcount for each 128-bit chunk of the intersection
    uint8x16_t intersection_popcount_0 = vcntq_u8(vandq_u8(a.u8x16s[0], b.u8x16s[0]));
    uint8x16_t intersection_popcount_1 = vcntq_u8(vandq_u8(a.u8x16s[1], b.u8x16s[1]));
    uint8x16_t intersection_popcount_2 = vcntq_u8(vandq_u8(a.u8x16s[2], b.u8x16s[2]));
    uint8x16_t intersection_popcount_3 = vcntq_u8(vandq_u8(a.u8x16s[3], b.u8x16s[3]));
    // Pairwise horizontal add to combine chunks
    uint8x16_t intersection_popcount_01 = vaddq_u8(intersection_popcount_0, intersection_popcount_1);
    uint8x16_t intersection_popcount_23 = vaddq_u8(intersection_popcount_2, intersection_popcount_3);
    uint8x16_t intersection_popcount_total = vaddq_u8(intersection_popcount_01, intersection_popcount_23);
    // Widen from u8 -> u16 -> u32 and accumulate
    state->intersection_count = vaddq_u32(state->intersection_count,
                                          vpaddlq_u16(vpaddlq_u8(intersection_popcount_total)));
}

SIMSIMD_INTERNAL void simsimd_jaccard_b512_finalize_neon(simsimd_jaccard_b512_state_neon_t const *state_a,
                                                         simsimd_jaccard_b512_state_neon_t const *state_b,
                                                         simsimd_jaccard_b512_state_neon_t const *state_c,
                                                         simsimd_jaccard_b512_state_neon_t const *state_d,
                                                         simsimd_distance_t *results) {
    // Loop-unrolled reduction for all 4 states
    simsimd_i32_t intersection_count_a = vaddvq_u32(state_a->intersection_count);
    simsimd_i32_t intersection_count_b = vaddvq_u32(state_b->intersection_count);
    simsimd_i32_t intersection_count_c = vaddvq_u32(state_c->intersection_count);
    simsimd_i32_t intersection_count_d = vaddvq_u32(state_d->intersection_count);

    // Store results - explicit unrolled assignments
    results[0] = (simsimd_distance_t)intersection_count_a;
    results[1] = (simsimd_distance_t)intersection_count_b;
    results[2] = (simsimd_distance_t)intersection_count_c;
    results[3] = (simsimd_distance_t)intersection_count_d;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_SVE
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_hamming_b8_sve(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                           simsimd_distance_t *result) {

    // On very small register sizes, NEON is at least as fast as SVE.
    simsimd_size_t const words_per_register = svcntb();
    if (words_per_register <= 32) {
        simsimd_hamming_b8_neon(a, b, n_words, result);
        return;
    }

    // On larger register sizes, SVE is faster.
    simsimd_size_t i = 0, cycle = 0;
    simsimd_i32_t differences = 0;
    svuint8_t differences_cycle_vec = svdup_n_u8(0);
    svbool_t const all_vec = svptrue_b8();
    while (i < n_words) {
        do {
            svbool_t pg_vec = svwhilelt_b8((unsigned int)i, (unsigned int)n_words);
            svuint8_t a_vec = svld1_u8(pg_vec, a + i);
            svuint8_t b_vec = svld1_u8(pg_vec, b + i);
            differences_cycle_vec = svadd_u8_z(all_vec, differences_cycle_vec,
                                               svcnt_u8_x(all_vec, sveor_u8_m(all_vec, a_vec, b_vec)));
            i += words_per_register;
            ++cycle;
        } while (i < n_words && cycle < 31);
        differences += svaddv_u8(all_vec, differences_cycle_vec);
        differences_cycle_vec = svdup_n_u8(0);
        cycle = 0; // Reset the cycle counter.
    }

    *result = differences;
}

SIMSIMD_PUBLIC void simsimd_jaccard_b8_sve(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                           simsimd_distance_t *result) {

    // On very small register sizes, NEON is at least as fast as SVE.
    simsimd_size_t const words_per_register = svcntb();
    if (words_per_register <= 32) {
        simsimd_jaccard_b8_neon(a, b, n_words, result);
        return;
    }

    // On larger register sizes, SVE is faster.
    simsimd_size_t i = 0, cycle = 0;
    simsimd_i32_t intersection_count = 0, union_count = 0;
    svuint8_t intersection_popcount_vec = svdup_n_u8(0);
    svuint8_t union_popcount_vec = svdup_n_u8(0);
    svbool_t const predicate_all = svptrue_b8();
    while (i < n_words) {
        do {
            svbool_t predicate_active = svwhilelt_b8((unsigned int)i, (unsigned int)n_words);
            svuint8_t a_vec = svld1_u8(predicate_active, a + i);
            svuint8_t b_vec = svld1_u8(predicate_active, b + i);
            intersection_popcount_vec = svadd_u8_z(predicate_all, intersection_popcount_vec,
                                                   svcnt_u8_x(predicate_all, svand_u8_m(predicate_all, a_vec, b_vec)));
            union_popcount_vec = svadd_u8_z(predicate_all, union_popcount_vec,
                                            svcnt_u8_x(predicate_all, svorr_u8_m(predicate_all, a_vec, b_vec)));
            i += words_per_register;
            ++cycle;
        } while (i < n_words && cycle < 31);
        intersection_count += svaddv_u8(predicate_all, intersection_popcount_vec);
        intersection_popcount_vec = svdup_n_u8(0);
        union_count += svaddv_u8(predicate_all, union_popcount_vec);
        union_popcount_vec = svdup_n_u8(0);
        cycle = 0; // Reset the cycle counter.
    }

    *result = (union_count != 0) ? 1 - (simsimd_f64_t)intersection_count / (simsimd_f64_t)union_count : 1;
}

SIMSIMD_PUBLIC void simsimd_jaccard_u32_sve(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    simsimd_size_t const words_per_register = svcntw();
    simsimd_size_t i = 0;
    simsimd_size_t intersection_count = 0;
    while (i < n) {
        svbool_t predicate_active = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svuint32_t a_vec = svld1_u32(predicate_active, a + i);
        svuint32_t b_vec = svld1_u32(predicate_active, b + i);
        svbool_t equality_predicate = svcmpeq_u32(predicate_active, a_vec, b_vec);
        intersection_count += svcntp_b32(predicate_active, equality_predicate);
        i += words_per_register;
    }
    *result = (n != 0) ? 1 - (simsimd_f64_t)intersection_count / (simsimd_f64_t)n : 1;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SVE
#endif // _SIMSIMD_TARGET_ARM

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vpopcntdq")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512vpopcntdq"))), \
                             apply_to = function)

SIMSIMD_PUBLIC void simsimd_hamming_b8_ice(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                           simsimd_distance_t *result) {

    simsimd_size_t xor_count;
    // It's harder to squeeze out performance from tiny representations, so we unroll the loops for binary metrics.
    if (n_words <= 64) { // Up to 512 bits.
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
        __m512i a_vec = _mm512_maskz_loadu_epi8(mask, a);
        __m512i b_vec = _mm512_maskz_loadu_epi8(mask, b);
        __m512i xor_count_vec = _mm512_popcnt_epi64(_mm512_xor_si512(a_vec, b_vec));
        xor_count = _mm512_reduce_add_epi64(xor_count_vec);
    }
    else if (n_words <= 128) { // Up to 1024 bits.
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words - 64);
        __m512i a1_vec = _mm512_loadu_epi8(a);
        __m512i b1_vec = _mm512_loadu_epi8(b);
        __m512i a2_vec = _mm512_maskz_loadu_epi8(mask, a + 64);
        __m512i b2_vec = _mm512_maskz_loadu_epi8(mask, b + 64);
        __m512i xor1_count_vec = _mm512_popcnt_epi64(_mm512_xor_si512(a1_vec, b1_vec));
        __m512i xor2_count_vec = _mm512_popcnt_epi64(_mm512_xor_si512(a2_vec, b2_vec));
        xor_count = _mm512_reduce_add_epi64(_mm512_add_epi64(xor2_count_vec, xor1_count_vec));
    }
    else if (n_words <= 192) { // Up to 1536 bits.
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words - 128);
        __m512i a1_vec = _mm512_loadu_epi8(a);
        __m512i b1_vec = _mm512_loadu_epi8(b);
        __m512i a2_vec = _mm512_loadu_epi8(a + 64);
        __m512i b2_vec = _mm512_loadu_epi8(b + 64);
        __m512i a3_vec = _mm512_maskz_loadu_epi8(mask, a + 128);
        __m512i b3_vec = _mm512_maskz_loadu_epi8(mask, b + 128);
        __m512i xor1_count_vec = _mm512_popcnt_epi64(_mm512_xor_si512(a1_vec, b1_vec));
        __m512i xor2_count_vec = _mm512_popcnt_epi64(_mm512_xor_si512(a2_vec, b2_vec));
        __m512i xor3_count_vec = _mm512_popcnt_epi64(_mm512_xor_si512(a3_vec, b3_vec));
        xor_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(xor3_count_vec, _mm512_add_epi64(xor2_count_vec, xor1_count_vec)));
    }
    else if (n_words <= 256) { // Up to 2048 bits.
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words - 192);
        __m512i a1_vec = _mm512_loadu_epi8(a);
        __m512i b1_vec = _mm512_loadu_epi8(b);
        __m512i a2_vec = _mm512_loadu_epi8(a + 64);
        __m512i b2_vec = _mm512_loadu_epi8(b + 64);
        __m512i a3_vec = _mm512_loadu_epi8(a + 128);
        __m512i b3_vec = _mm512_loadu_epi8(b + 128);
        __m512i a4_vec = _mm512_maskz_loadu_epi8(mask, a + 192);
        __m512i b4_vec = _mm512_maskz_loadu_epi8(mask, b + 192);
        __m512i xor1_count_vec = _mm512_popcnt_epi64(_mm512_xor_si512(a1_vec, b1_vec));
        __m512i xor2_count_vec = _mm512_popcnt_epi64(_mm512_xor_si512(a2_vec, b2_vec));
        __m512i xor3_count_vec = _mm512_popcnt_epi64(_mm512_xor_si512(a3_vec, b3_vec));
        __m512i xor4_count_vec = _mm512_popcnt_epi64(_mm512_xor_si512(a4_vec, b4_vec));
        xor_count = _mm512_reduce_add_epi64(_mm512_add_epi64(_mm512_add_epi64(xor4_count_vec, xor3_count_vec),
                                                             _mm512_add_epi64(xor2_count_vec, xor1_count_vec)));
    }
    else {
        __m512i xor_count_vec = _mm512_setzero_si512();
        __m512i a_vec, b_vec;

    simsimd_hamming_b8_ice_cycle:
        if (n_words < 64) {
            __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
            a_vec = _mm512_maskz_loadu_epi8(mask, a);
            b_vec = _mm512_maskz_loadu_epi8(mask, b);
            n_words = 0;
        }
        else {
            a_vec = _mm512_loadu_epi8(a);
            b_vec = _mm512_loadu_epi8(b);
            a += 64, b += 64, n_words -= 64;
        }
        __m512i xor_vec = _mm512_xor_si512(a_vec, b_vec);
        xor_count_vec = _mm512_add_epi64(xor_count_vec, _mm512_popcnt_epi64(xor_vec));
        if (n_words) goto simsimd_hamming_b8_ice_cycle;

        xor_count = _mm512_reduce_add_epi64(xor_count_vec);
    }
    *result = xor_count;
}

SIMSIMD_PUBLIC void simsimd_jaccard_b8_ice(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                           simsimd_distance_t *result) {

    simsimd_size_t intersection_count = 0, union_count = 0;
    //  It's harder to squeeze out performance from tiny representations, so we unroll the loops for binary metrics.
    if (n_words <= 64) { // Up to 512 bits.
        __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
        __m512i a_vec = _mm512_maskz_loadu_epi8(load_mask, a);
        __m512i b_vec = _mm512_maskz_loadu_epi8(load_mask, b);
        __m512i intersection_popcount_vec = _mm512_popcnt_epi64(_mm512_and_si512(a_vec, b_vec));
        __m512i union_popcount_vec = _mm512_popcnt_epi64(_mm512_or_si512(a_vec, b_vec));
        intersection_count = _mm512_reduce_add_epi64(intersection_popcount_vec);
        union_count = _mm512_reduce_add_epi64(union_popcount_vec);
    }
    else if (n_words <= 128) { // Up to 1024 bits.
        __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words - 64);
        __m512i a_chunk_1 = _mm512_loadu_epi8(a);
        __m512i b_chunk_1 = _mm512_loadu_epi8(b);
        __m512i a_chunk_2 = _mm512_maskz_loadu_epi8(load_mask, a + 64);
        __m512i b_chunk_2 = _mm512_maskz_loadu_epi8(load_mask, b + 64);
        __m512i intersection_popcount_1 = _mm512_popcnt_epi64(_mm512_and_si512(a_chunk_1, b_chunk_1));
        __m512i union_popcount_1 = _mm512_popcnt_epi64(_mm512_or_si512(a_chunk_1, b_chunk_1));
        __m512i intersection_popcount_2 = _mm512_popcnt_epi64(_mm512_and_si512(a_chunk_2, b_chunk_2));
        __m512i union_popcount_2 = _mm512_popcnt_epi64(_mm512_or_si512(a_chunk_2, b_chunk_2));
        intersection_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(intersection_popcount_2, intersection_popcount_1));
        union_count = _mm512_reduce_add_epi64(_mm512_add_epi64(union_popcount_2, union_popcount_1));
    }
    else if (n_words <= 192) { // Up to 1536 bits.
        __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words - 128);
        __m512i a_chunk_1 = _mm512_loadu_epi8(a);
        __m512i b_chunk_1 = _mm512_loadu_epi8(b);
        __m512i a_chunk_2 = _mm512_loadu_epi8(a + 64);
        __m512i b_chunk_2 = _mm512_loadu_epi8(b + 64);
        __m512i a_chunk_3 = _mm512_maskz_loadu_epi8(load_mask, a + 128);
        __m512i b_chunk_3 = _mm512_maskz_loadu_epi8(load_mask, b + 128);
        __m512i intersection_popcount_1 = _mm512_popcnt_epi64(_mm512_and_si512(a_chunk_1, b_chunk_1));
        __m512i union_popcount_1 = _mm512_popcnt_epi64(_mm512_or_si512(a_chunk_1, b_chunk_1));
        __m512i intersection_popcount_2 = _mm512_popcnt_epi64(_mm512_and_si512(a_chunk_2, b_chunk_2));
        __m512i union_popcount_2 = _mm512_popcnt_epi64(_mm512_or_si512(a_chunk_2, b_chunk_2));
        __m512i intersection_popcount_3 = _mm512_popcnt_epi64(_mm512_and_si512(a_chunk_3, b_chunk_3));
        __m512i union_popcount_3 = _mm512_popcnt_epi64(_mm512_or_si512(a_chunk_3, b_chunk_3));
        intersection_count = _mm512_reduce_add_epi64( //
            _mm512_add_epi64(intersection_popcount_3,
                             _mm512_add_epi64(intersection_popcount_2, intersection_popcount_1)));
        union_count = _mm512_reduce_add_epi64( //
            _mm512_add_epi64(union_popcount_3, _mm512_add_epi64(union_popcount_2, union_popcount_1)));
    }
    else if (n_words <= 256) { // Up to 2048 bits.
        __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words - 192);
        __m512i a_chunk_1 = _mm512_loadu_epi8(a);
        __m512i b_chunk_1 = _mm512_loadu_epi8(b);
        __m512i a_chunk_2 = _mm512_loadu_epi8(a + 64);
        __m512i b_chunk_2 = _mm512_loadu_epi8(b + 64);
        __m512i a_chunk_3 = _mm512_loadu_epi8(a + 128);
        __m512i b_chunk_3 = _mm512_loadu_epi8(b + 128);
        __m512i a_chunk_4 = _mm512_maskz_loadu_epi8(load_mask, a + 192);
        __m512i b_chunk_4 = _mm512_maskz_loadu_epi8(load_mask, b + 192);
        __m512i intersection_popcount_1 = _mm512_popcnt_epi64(_mm512_and_si512(a_chunk_1, b_chunk_1));
        __m512i union_popcount_1 = _mm512_popcnt_epi64(_mm512_or_si512(a_chunk_1, b_chunk_1));
        __m512i intersection_popcount_2 = _mm512_popcnt_epi64(_mm512_and_si512(a_chunk_2, b_chunk_2));
        __m512i union_popcount_2 = _mm512_popcnt_epi64(_mm512_or_si512(a_chunk_2, b_chunk_2));
        __m512i intersection_popcount_3 = _mm512_popcnt_epi64(_mm512_and_si512(a_chunk_3, b_chunk_3));
        __m512i union_popcount_3 = _mm512_popcnt_epi64(_mm512_or_si512(a_chunk_3, b_chunk_3));
        __m512i intersection_popcount_4 = _mm512_popcnt_epi64(_mm512_and_si512(a_chunk_4, b_chunk_4));
        __m512i union_popcount_4 = _mm512_popcnt_epi64(_mm512_or_si512(a_chunk_4, b_chunk_4));
        intersection_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(_mm512_add_epi64(intersection_popcount_4, intersection_popcount_3),
                             _mm512_add_epi64(intersection_popcount_2, intersection_popcount_1)));
        union_count = _mm512_reduce_add_epi64(_mm512_add_epi64(_mm512_add_epi64(union_popcount_4, union_popcount_3),
                                                               _mm512_add_epi64(union_popcount_2, union_popcount_1)));
    }
    else {
        __m512i intersection_popcount_vec = _mm512_setzero_si512();
        __m512i union_popcount_vec = _mm512_setzero_si512();
        __m512i a_vec, b_vec;

    simsimd_jaccard_b8_ice_cycle:
        if (n_words < 64) {
            __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
            a_vec = _mm512_maskz_loadu_epi8(load_mask, a);
            b_vec = _mm512_maskz_loadu_epi8(load_mask, b);
            n_words = 0;
        }
        else {
            a_vec = _mm512_loadu_epi8(a);
            b_vec = _mm512_loadu_epi8(b);
            a += 64, b += 64, n_words -= 64;
        }
        __m512i intersection_bits = _mm512_and_si512(a_vec, b_vec);
        __m512i union_bits = _mm512_or_si512(a_vec, b_vec);
        intersection_popcount_vec = _mm512_add_epi64(intersection_popcount_vec, _mm512_popcnt_epi64(intersection_bits));
        union_popcount_vec = _mm512_add_epi64(union_popcount_vec, _mm512_popcnt_epi64(union_bits));
        if (n_words) goto simsimd_jaccard_b8_ice_cycle;

        intersection_count = _mm512_reduce_add_epi64(intersection_popcount_vec);
        union_count = _mm512_reduce_add_epi64(union_popcount_vec);
    }
    *result = (union_count != 0) ? 1 - (simsimd_f64_t)intersection_count / (simsimd_f64_t)union_count : 1;
}

SIMSIMD_PUBLIC void simsimd_jaccard_u32_ice(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    simsimd_size_t total_elements = n;
    simsimd_size_t intersection_count = 0;
    for (; n >= 16; n -= 16, a += 16, b += 16) {
        __m512i a_vec = _mm512_loadu_epi32(a);
        __m512i b_vec = _mm512_loadu_epi32(b);
        __mmask16 equality_mask = _mm512_cmpeq_epi32_mask(a_vec, b_vec);
        intersection_count += _mm_popcnt_u32((unsigned int)equality_mask);
    }
    if (n) {
        __mmask16 load_mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        __m512i a_vec = _mm512_maskz_loadu_epi32(load_mask, a);
        __m512i b_vec = _mm512_maskz_loadu_epi32(load_mask, b);
        __mmask16 equality_mask = _mm512_cmpeq_epi32_mask(a_vec, b_vec) & load_mask;
        intersection_count += _mm_popcnt_u32((unsigned int)equality_mask);
    }
    *result = (total_elements != 0) ? 1 - (simsimd_f64_t)intersection_count / (simsimd_f64_t)total_elements : 1;
}

/**
 *  @brief Running state for 512-bit Jaccard accumulation on Ice Lake.
 *  @code{.c}
 *  simsimd_b512_vec_t a_block, b_block;
 *
 *  simsimd_jaccard_b512_state_ice_t state_a, state_b, state_c, state_d;
 *  simsimd_jaccard_b512_init_ice(&state_a);
 *  simsimd_jaccard_b512_init_ice(&state_b);
 *  simsimd_jaccard_b512_init_ice(&state_c);
 *  simsimd_jaccard_b512_init_ice(&state_d);
 *  simsimd_jaccard_b512_update_ice(&state_a, a_block, b_block);
 *  // ... update state_b, state_c, state_d similarly ...
 *
 *  simsimd_distance_t results[4];
 *  simsimd_jaccard_b512_finalize_ice(&state_a, &state_b, &state_c, &state_d, results);
 *  @endcode
 */
typedef struct simsimd_jaccard_b512_state_ice_t {
    __m512i intersection_count;
} simsimd_jaccard_b512_state_ice_t;

SIMSIMD_INTERNAL void simsimd_jaccard_b512_init_ice(simsimd_jaccard_b512_state_ice_t *state) {
    state->intersection_count = _mm512_setzero_si512();
}

SIMSIMD_INTERNAL void simsimd_jaccard_b512_update_ice(simsimd_jaccard_b512_state_ice_t *state, simsimd_b512_vec_t a,
                                                      simsimd_b512_vec_t b) {
    state->intersection_count = _mm512_add_epi64(state->intersection_count,
                                                 _mm512_popcnt_epi64(_mm512_and_si512(a.zmm, b.zmm)));
}

SIMSIMD_INTERNAL void simsimd_jaccard_b512_finalize_ice(simsimd_jaccard_b512_state_ice_t const *state_a,
                                                        simsimd_jaccard_b512_state_ice_t const *state_b,
                                                        simsimd_jaccard_b512_state_ice_t const *state_c,
                                                        simsimd_jaccard_b512_state_ice_t const *state_d,
                                                        simsimd_distance_t *results) {
    // ILP-optimized 4-way horizontal reduction for i64 using SIMD operations.
    // Avoids serial _mm512_reduce_add_epi64 calls by staying in SIMD as long as possible.
    // Step 1: 8->4 for all 4 states (extract high 256-bit half and add to low half)
    __m256i intersection_count_a = _mm256_add_epi64(_mm512_castsi512_si256(state_a->intersection_count),
                                                    _mm512_extracti64x4_epi64(state_a->intersection_count, 1));
    __m256i intersection_count_b = _mm256_add_epi64(_mm512_castsi512_si256(state_b->intersection_count),
                                                    _mm512_extracti64x4_epi64(state_b->intersection_count, 1));
    __m256i intersection_count_c = _mm256_add_epi64(_mm512_castsi512_si256(state_c->intersection_count),
                                                    _mm512_extracti64x4_epi64(state_c->intersection_count, 1));
    __m256i intersection_count_d = _mm256_add_epi64(_mm512_castsi512_si256(state_d->intersection_count),
                                                    _mm512_extracti64x4_epi64(state_d->intersection_count, 1));
    // Step 2: 4->2 for all 4 states (extract high 128-bit half and add to low half)
    __m128i partial_a = _mm_add_epi64(_mm256_castsi256_si128(intersection_count_a),
                                      _mm256_extracti128_si256(intersection_count_a, 1));
    __m128i partial_b = _mm_add_epi64(_mm256_castsi256_si128(intersection_count_b),
                                      _mm256_extracti128_si256(intersection_count_b, 1));
    __m128i partial_c = _mm_add_epi64(_mm256_castsi256_si128(intersection_count_c),
                                      _mm256_extracti128_si256(intersection_count_c, 1));
    __m128i partial_d = _mm_add_epi64(_mm256_castsi256_si128(intersection_count_d),
                                      _mm256_extracti128_si256(intersection_count_d, 1));
    // Step 3: Transpose and reduce 2x2 pairs into final 4 results.
    // Each partial has [sum_lo, sum_hi]. We need to add them.
    // Transpose: combine [a_lo, a_hi] and [b_lo, b_hi] into [a_lo, b_lo] and [a_hi, b_hi]
    __m128i transpose_ab_lo = _mm_unpacklo_epi64(partial_a, partial_b); // [a_lo, b_lo]
    __m128i transpose_ab_hi = _mm_unpackhi_epi64(partial_a, partial_b); // [a_hi, b_hi]
    __m128i transpose_cd_lo = _mm_unpacklo_epi64(partial_c, partial_d); // [c_lo, d_lo]
    __m128i transpose_cd_hi = _mm_unpackhi_epi64(partial_c, partial_d); // [c_hi, d_hi]
    // Final horizontal add for each pair
    __m128i final_ab = _mm_add_epi64(transpose_ab_lo, transpose_ab_hi); // [a_total, b_total]
    __m128i final_cd = _mm_add_epi64(transpose_cd_lo, transpose_cd_hi); // [c_total, d_total]
    // Convert to f64 and store
    __m128d result_ab = _mm_cvtepi64_pd(final_ab);
    __m128d result_cd = _mm_cvtepi64_pd(final_cd);
    _mm_storeu_pd(results, result_ab);
    _mm_storeu_pd(results + 2, result_cd);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_ICE

#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("popcnt")
#pragma clang attribute push(__attribute__((target("popcnt"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_hamming_b8_haswell(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                               simsimd_distance_t *result) {
    // x86 supports unaligned loads and works just fine with the scalar version for small vectors.
    simsimd_size_t differences = 0;
    for (; n_words >= 8; n_words -= 8, a += 8, b += 8)
        differences += _mm_popcnt_u64(*(simsimd_u64_t const *)a ^ *(simsimd_u64_t const *)b);
    for (; n_words; --n_words, ++a, ++b) differences += _mm_popcnt_u32(*a ^ *b);
    *result = differences;
}

SIMSIMD_PUBLIC void simsimd_jaccard_b8_haswell(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                               simsimd_distance_t *result) {
    // x86 supports unaligned loads and works just fine with the scalar version for small vectors.
    simsimd_size_t intersection_count = 0, union_count = 0;
    for (; n_words >= 8; n_words -= 8, a += 8, b += 8)
        intersection_count += _mm_popcnt_u64(*(simsimd_u64_t const *)a & *(simsimd_u64_t const *)b),
            union_count += _mm_popcnt_u64(*(simsimd_u64_t const *)a | *(simsimd_u64_t const *)b);
    for (; n_words; --n_words, ++a, ++b)
        intersection_count += _mm_popcnt_u32(*a & *b), union_count += _mm_popcnt_u32(*a | *b);
    *result = (union_count != 0) ? 1 - (simsimd_f64_t)intersection_count / (simsimd_f64_t)union_count : 1;
}

SIMSIMD_PUBLIC void simsimd_jaccard_u32_haswell(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                                simsimd_distance_t *result) {
    simsimd_size_t total_elements = n;
    simsimd_size_t intersection_count = 0;
    for (; n >= 4; n -= 4, a += 4, b += 4) {
        __m128i a_vec = _mm_loadu_si128((__m128i const *)a);
        __m128i b_vec = _mm_loadu_si128((__m128i const *)b);
        __m128i equality_vec = _mm_cmpeq_epi32(a_vec, b_vec);
        int equality_mask = _mm_movemask_ps(_mm_castsi128_ps(equality_vec));
        intersection_count += _mm_popcnt_u32((unsigned int)equality_mask);
    }
    for (; n; --n, ++a, ++b) intersection_count += (*a == *b);
    *result = (total_elements != 0) ? 1 - (simsimd_f64_t)intersection_count / (simsimd_f64_t)total_elements : 1;
}

/**
 *  @brief Running state for 512-bit Jaccard accumulation on Haswell.
 *  @code{.c}
 *  simsimd_b512_vec_t a_block, b_block;
 *
 *  simsimd_jaccard_b512_state_haswell_t state_a, state_b, state_c, state_d;
 *  simsimd_jaccard_b512_init_haswell(&state_a);
 *  simsimd_jaccard_b512_init_haswell(&state_b);
 *  simsimd_jaccard_b512_init_haswell(&state_c);
 *  simsimd_jaccard_b512_init_haswell(&state_d);
 *  simsimd_jaccard_b512_update_haswell(&state_a, a_block, b_block);
 *  // ... update state_b, state_c, state_d similarly ...
 *
 *  simsimd_distance_t results[4];
 *  simsimd_jaccard_b512_finalize_haswell(&state_a, &state_b, &state_c, &state_d, results);
 *  @endcode
 */
typedef struct simsimd_jaccard_b512_state_haswell_t {
    simsimd_size_t intersection_count[2];
} simsimd_jaccard_b512_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_jaccard_b512_init_haswell(simsimd_jaccard_b512_state_haswell_t *state) {
    state->intersection_count[0] = 0;
    state->intersection_count[1] = 0;
}

SIMSIMD_INTERNAL void simsimd_jaccard_b512_update_haswell(simsimd_jaccard_b512_state_haswell_t *state,
                                                          simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    // Process all 8 u64 words, alternating accumulation to expose ILP
    state->intersection_count[0] += _mm_popcnt_u64(a.u64s[0] & b.u64s[0]);
    state->intersection_count[1] += _mm_popcnt_u64(a.u64s[1] & b.u64s[1]);
    state->intersection_count[0] += _mm_popcnt_u64(a.u64s[2] & b.u64s[2]);
    state->intersection_count[1] += _mm_popcnt_u64(a.u64s[3] & b.u64s[3]);
    state->intersection_count[0] += _mm_popcnt_u64(a.u64s[4] & b.u64s[4]);
    state->intersection_count[1] += _mm_popcnt_u64(a.u64s[5] & b.u64s[5]);
    state->intersection_count[0] += _mm_popcnt_u64(a.u64s[6] & b.u64s[6]);
    state->intersection_count[1] += _mm_popcnt_u64(a.u64s[7] & b.u64s[7]);
}

SIMSIMD_INTERNAL void simsimd_jaccard_b512_finalize_haswell(simsimd_jaccard_b512_state_haswell_t const *state_a,
                                                            simsimd_jaccard_b512_state_haswell_t const *state_b,
                                                            simsimd_jaccard_b512_state_haswell_t const *state_c,
                                                            simsimd_jaccard_b512_state_haswell_t const *state_d,
                                                            simsimd_distance_t *results) {
    // Loop-unrolled extraction for all 4 states
    simsimd_size_t intersection_count_a = state_a->intersection_count[0] + state_a->intersection_count[1];
    simsimd_size_t intersection_count_b = state_b->intersection_count[0] + state_b->intersection_count[1];
    simsimd_size_t intersection_count_c = state_c->intersection_count[0] + state_c->intersection_count[1];
    simsimd_size_t intersection_count_d = state_d->intersection_count[0] + state_d->intersection_count[1];

    // Store results - explicit unrolled assignments
    results[0] = (simsimd_distance_t)intersection_count_a;
    results[1] = (simsimd_distance_t)intersection_count_b;
    results[2] = (simsimd_distance_t)intersection_count_c;
    results[3] = (simsimd_distance_t)intersection_count_d;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL
#endif // _SIMSIMD_TARGET_X86

#if !SIMSIMD_DYNAMIC_DISPATCH

SIMSIMD_PUBLIC void simsimd_hamming_b8(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n,
                                       simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SVE
    simsimd_hamming_b8_sve(a, b, n, d);
#elif SIMSIMD_TARGET_NEON
    simsimd_hamming_b8_neon(a, b, n, d);
#elif SIMSIMD_TARGET_ICE
    simsimd_hamming_b8_ice(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_hamming_b8_haswell(a, b, n, d);
#else
    simsimd_hamming_b8_serial(a, b, n, d);
#endif
}

SIMSIMD_PUBLIC void simsimd_jaccard_b8(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n,
                                       simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SVE
    simsimd_jaccard_b8_sve(a, b, n, d);
#elif SIMSIMD_TARGET_NEON
    simsimd_jaccard_b8_neon(a, b, n, d);
#elif SIMSIMD_TARGET_ICE
    simsimd_jaccard_b8_ice(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_jaccard_b8_haswell(a, b, n, d);
#else
    simsimd_jaccard_b8_serial(a, b, n, d);
#endif
}

SIMSIMD_PUBLIC void simsimd_jaccard_u32(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                        simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SVE
    simsimd_jaccard_u32_sve(a, b, n, d);
#elif SIMSIMD_TARGET_NEON
    simsimd_jaccard_u32_neon(a, b, n, d);
#elif SIMSIMD_TARGET_ICE
    simsimd_jaccard_u32_ice(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_jaccard_u32_haswell(a, b, n, d);
#else
    simsimd_jaccard_u32_serial(a, b, n, d);
#endif
}

#endif // !SIMSIMD_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
